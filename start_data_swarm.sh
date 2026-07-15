#!/bin/bash

# =====================================================================
# Распределенный рой для подготовки Датасетов (Walk-Forward)
# =====================================================================

# Оставляем пару ядер свободными, чтобы система не зависла намертво
NUM_WORKERS=1

# --timeframe 1d       : Интервал данных — дневные свечи.
# --lookback 60        : Глубина истории — модель смотрит на 60 дней назад.
# --horizon 10         : Горизонт прогноза — ищем выход по барьерам в течение 10 дней.
# --auto               : Режим автоматического расчета уровней TP/SL на основе волатильности.
# --percentile 75      : Перцентиль волатильности для отсечения аномальных выбросов при авто-разметке.
# --init_split         : Дата начала первого разделения данных на Train и Val.
# --val_interval 2     : Продолжительность валидационного периода в годах.
# --split_interval 2   : Шаг смещения окна Walk-Forward в годах.
# --endpoint           : Дата окончания формирования всех временных интервалов.
# --corr_threshold     : Порог удаления коррелирующих признаков (убираем дубликаты > 85%).
# --cum_threshold      : Порог кумулятивной важности (оставляем топ фичей, дающих 99% влияния).
# --force              : Раскомментируйте параметр ниже для полной перезаписи кэшированных данных.

# --- Жесткое удушение внутренних потоков Python ---
# Так как мы параллелим задачи в Bash, нам нужно запретить 
# Numpy, Pandas и LightGBM плодить дочерние потоки.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LIGHTGBM_NUM_THREADS=1
PYTHON_WORKERS=1

# --- Базовые настройки ---
TIMEFRAME="1d"
VAL_INTERVAL=2
SPLIT_INTERVAL=1
CORR_THRESHOLD=0.85
CUM_THRESHOLD=0.99
PERCENTILE=75

# Конфигурации и годы
#CONFIGS=("60:10" "30:5" "18:3" "10:1")
CONFIGS=("90:15")
YEARS=($(seq 2010 2026))

# Создаем папки для логов и подскриптов
mkdir -p _logs_data_prep/scripts

echo "==================================================="
echo "🛡️ ФАЗА 1: Сборка базовых кэшей (Защита от Race Condition)"
echo "==================================================="
# Запускаем скрипт синхронно ровно на 1 год для каждой конфигурации.
# Это заставит Python безопасно собрать raw_combined.csv без конкуренции.
for config in "${CONFIGS[@]}"; do
    IFS=":" read -r LOOKBACK HORIZON <<< "$config"
    echo "   Собираю кэш для $LOOKBACK:$HORIZON..."
    python -m _tools.init_dataset \
        --timeframe $TIMEFRAME \
        --lookback $LOOKBACK \
        --horizon $HORIZON \
        --auto --percentile $PERCENTILE \
        --init_split "${YEARS[0]}-01-01" \
        --val_interval $VAL_INTERVAL \
        --split_interval $SPLIT_INTERVAL \
        --endpoint "${YEARS[0]}-01-01" \
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

# Раскидываем 60 задач (4 конфига * 15 лет) по 14 воркерам
JOB_INDEX=0
for config in "${CONFIGS[@]}"; do
    IFS=":" read -r LOOKBACK HORIZON <<< "$config"
    for year in "${YEARS[@]}"; do
        
        WORKER_ID=$((JOB_INDEX % NUM_WORKERS))
        SCRIPT_FILE="_logs_data_prep/scripts/worker_${WORKER_ID}.sh"
        LOG_FILE="_logs_data_prep/fold_${LOOKBACK}_${HORIZON}_${year}.log"
        
        # Фокус: передавая init_split и endpoint одним и тем же годом, 
        # мы заставляем скрипт просчитать ровно 1 фолд!
        cat <<EOF >> "$SCRIPT_FILE"
echo "▶️ [Worker $WORKER_ID] Старт: $LOOKBACK:$HORIZON фолд $year..."
python -m _tools.init_dataset \\
    --timeframe $TIMEFRAME \\
    --lookback $LOOKBACK \\
    --horizon $HORIZON \\
    --auto --percentile $PERCENTILE \\
    --init_split "${year}-01-01" \\
    --val_interval $VAL_INTERVAL \\
    --split_interval $SPLIT_INTERVAL \\
    --endpoint "${year}-01-01" \\
    --corr_threshold $CORR_THRESHOLD \\
    --cum_threshold $CUM_THRESHOLD \\
    --workers $PYTHON_WORKERS > "$LOG_FILE" 2>&1
echo "✅ [Worker $WORKER_ID] Завершено: $LOOKBACK:$HORIZON фолд $year"
EOF
        ((JOB_INDEX++))
    done
done

echo "==================================================="
echo "🔥 ФАЗА 3: Запуск $NUM_WORKERS tmux-сессий..."
echo "==================================================="

for ((i=0; i<NUM_WORKERS; i++)); do
    SESSION_NAME="data_worker_${i}"
    
    # Файл, куда будет писаться общий лог всей bash-сессии воркера
    SESSION_LOG_FILE="_logs_data_prep/worker_${i}_session.log"
    
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
    if [ $? != 0 ]; then
        echo "   Поднимаю воркера $i..."
        tmux new-session -d -s "$SESSION_NAME"
        
        # Передаем воркеру его пачку задач, перенаправляя ВЕСЬ вывод в лог-файл сессии
        tmux send-keys -t "$SESSION_NAME" "bash _logs_data_prep/scripts/worker_${i}.sh > \"$SESSION_LOG_FILE\" 2>&1; echo '🎉 ВСЕ ЗАДАЧИ ВОРКЕРА ЗАВЕРШЕНЫ' >> \"$SESSION_LOG_FILE\"" C-m
        sleep 0.5
    else
        echo "   ⚠️ Сессия $SESSION_NAME уже существует!"
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