#!/bin/bash

# =====================================================================
# Распределенный рой для подготовки Датасетов (Walk-Forward) [Адаптировано под 5m]
# =====================================================================

# Оставляем пару ядер свободными, чтобы система не зависла намертво
NUM_WORKERS=4

# --- Жесткое удушение внутренних потоков Python ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LIGHTGBM_NUM_THREADS=1
PYTHON_WORKERS=1

# --- Базовые настройки (для 5m таймфрейма) ---
TIMEFRAME="5m"
# ВАЖНО: Теперь интервалы измеряются в МЕСЯЦАХ (согласно обновленной логике init_dataset.py)
VAL_INTERVAL=2      # Валидация = 2 месяца
SPLIT_INTERVAL=3    # Смещение окна (шаг walk-forward) = 3 месяца (квартал)

CORR_THRESHOLD=0.85
CUM_THRESHOLD=0.99
PERCENTILE=75

# Конфигурации "Lookback:Horizon"
# 72 свечи = 6 часов, 12 свечей = 1 час (для 5m)
CONFIGS=("72:12")

# Генерируем даты сплитов (каждый квартал начиная с 2020 года)
START_YEAR=2020
END_YEAR=2026

SPLIT_DATES=()
for (( y=START_YEAR; y<=END_YEAR; y++ )); do
    for m in 01 04 07 10; do
        SPLIT_DATES+=("${y}-${m}-01")
    done
done

# Создаем папки для логов и подскриптов
mkdir -p _logs_data_prep/scripts

echo "==================================================="
echo "🛡️ ФАЗА 1: Сборка базовых кэшей (Защита от Race Condition)"
echo "==================================================="
for config in "${CONFIGS[@]}"; do
    IFS=":" read -r LOOKBACK HORIZON <<< "$config"
    echo "   Собираю кэш для $LOOKBACK:$HORIZON..."
    python -m _tools.init_dataset \
        --timeframe $TIMEFRAME \
        --lookback $LOOKBACK \
        --horizon $HORIZON \
        --auto --percentile $PERCENTILE \
        --init_split "${SPLIT_DATES[0]}" \
        --val_interval $VAL_INTERVAL \
        --split_interval $SPLIT_INTERVAL \
        --endpoint "${SPLIT_DATES[0]}" \
        --corr_threshold $CORR_THRESHOLD \
        --cum_threshold $CUM_THRESHOLD \
        --workers $PYTHON_WORKERS > _logs_data_prep/base_build_${LOOKBACK}_${HORIZON}.log 2>&1
done

echo "==================================================="
echo "🚀 ФАЗА 2: Распределение задач (Round-Robin)"
echo "==================================================="
rm -f _logs_data_prep/scripts/worker_*.sh

# Инициализируем скрипты для каждого воркера
for ((i=0; i<NUM_WORKERS; i++)); do
    echo "#!/bin/bash" > "_logs_data_prep/scripts/worker_${i}.sh"
    chmod +x "_logs_data_prep/scripts/worker_${i}.sh"
done

# Раскидываем задачи по воркерам
JOB_INDEX=0
for config in "${CONFIGS[@]}"; do
    IFS=":" read -r LOOKBACK HORIZON <<< "$config"
    for date in "${SPLIT_DATES[@]}"; do
        
        WORKER_ID=$((JOB_INDEX % NUM_WORKERS))
        SCRIPT_FILE="_logs_data_prep/scripts/worker_${WORKER_ID}.sh"
        
        # Заменяем дефисы на подчеркивания для имени лог-файла
        CLEAN_DATE=$(echo $date | tr '-' '_')
        LOG_FILE="_logs_data_prep/fold_${LOOKBACK}_${HORIZON}_${CLEAN_DATE}.log"
        
        # Фокус: передавая init_split и endpoint одной и той же датой, 
        # мы заставляем скрипт просчитать ровно 1 фолд!
        cat <<EOF >> "$SCRIPT_FILE"
echo "▶️ [Worker $WORKER_ID] Старт: $LOOKBACK:$HORIZON фолд $date..."
python -m _tools.init_dataset \\
    --timeframe $TIMEFRAME \\
    --lookback $LOOKBACK \\
    --horizon $HORIZON \\
    --auto --percentile $PERCENTILE \\
    --init_split "${date}" \\
    --val_interval $VAL_INTERVAL \\
    --split_interval $SPLIT_INTERVAL \\
    --endpoint "${date}" \\
    --corr_threshold $CORR_THRESHOLD \\
    --cum_threshold $CUM_THRESHOLD \\
    --workers $PYTHON_WORKERS > "$LOG_FILE" 2>&1
echo "✅ [Worker $WORKER_ID] Завершено: $LOOKBACK:$HORIZON фолд $date"
EOF
        ((JOB_INDEX++))
    done
done

echo "==================================================="
echo "🔥 ФАЗА 3: Запуск $NUM_WORKERS tmux-сессий..."
echo "==================================================="

# Проверка наличия tmux
if ! command -v tmux &> /dev/null; then
    echo "❌ Ошибка: tmux не установлен! Установите его (sudo apt install tmux) или измените скрипт."
    exit 1
fi

for ((i=0; i<NUM_WORKERS; i++)); do
    SESSION_NAME="data_worker_${i}"
    SESSION_LOG_FILE="_logs_data_prep/worker_${i}_session.log"
    
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
    if [ $? != 0 ]; then
        echo "   Поднимаю воркера $i..."
        tmux new-session -d -s "$SESSION_NAME"
        
        tmux send-keys -t "$SESSION_NAME" "bash _logs_data_prep/scripts/worker_${i}.sh > \"$SESSION_LOG_FILE\" 2>&1; echo '🎉 ВСЕ ЗАДАЧИ ВОРКЕРА ЗАВЕРШЕНЫ' >> \"$SESSION_LOG_FILE\"" C-m
        sleep 0.5
    else
        echo "   ⚠️ Сессия $SESSION_NAME уже существует! Для перезапуска сначала выполните stop_data_swarm.sh"
    fi
done

echo "==================================================="
echo "✅ Рой из $NUM_WORKERS дата-воркеров успешно запущен!"
echo "Всего задач распределено: $JOB_INDEX"
echo "👉 Мониторинг всех сессий:   tmux ls"
echo "👉 Подключиться к воркеру 0: tmux attach -t data_worker_0"
echo "👉 Следить за общим прогрессом воркера 0:"
echo "   tail -f _logs_data_prep/worker_0_session.log"
echo "==================================================="