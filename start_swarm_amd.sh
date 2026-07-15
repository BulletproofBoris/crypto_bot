#!/bin/bash

# Генерируем уникальный ID для этого пула воркеров
export SWARM_ID="s_$(date +'%d_%H%M%S')"

# --- Значения по умолчанию ---
VRAM_PER_WORKER=1700
OS_BUFFER=1500 
STAGGER_DELAY=10
PYTHON_ARGS=""

# --- Функция парсинга аргументов ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vram) VRAM_PER_WORKER="$2"; shift ;;
        --stagger) STAGGER_DELAY="$2"; shift ;;
        *) PYTHON_ARGS="$PYTHON_ARGS $1" ;; # Собираем всё остальное для Python
    esac
    shift
done

# Если PYTHON_ARGS остались пустыми, задаем дефолтные настройки
if [ -z "$PYTHON_ARGS" ]; then
    echo "⚠️ Аргументы не переданы! Использую дефолтные настройки..."
    PYTHON_ARGS="--dataset_dir data/processed/2000_2026_1d_20_1 --bonus_ratio 0.2 --runs 200 --epochs 100 --l2_reg 1e-5 --lr 4e-3 --start_fold fold_2020 --append"
fi

CMD="python run_walkforward.py $PYTHON_ARGS"

TOTAL_VRAM=12000

if [ -z "$TOTAL_VRAM" ]; then
    echo "❌ Ошибка: Не удалось получить данные от nvidia-smi."
    exit 1
fi

AVAILABLE_VRAM=$((TOTAL_VRAM - OS_BUFFER))
MAX_WORKERS=$((AVAILABLE_VRAM / VRAM_PER_WORKER))

# Ограничиваем сверху 16 воркерами (защита от перегрева CPU/системы)
[ "$MAX_WORKERS" -gt 16 ] && MAX_WORKERS=16

echo "📊 VRAM всего: ${TOTAL_VRAM} MB"
echo "📊 VRAM доступно (без ОС): ${AVAILABLE_VRAM} MB"
echo "🎯 Рассчитано воркеров: ${MAX_WORKERS} (по ${VRAM_PER_WORKER} MB каждый, задержка ${STAGGER_DELAY}с)"
echo "---------------------------------------------------"
echo "🚀 Запуск с командой: $CMD"
echo "🏷️ Идентификатор сессии (Swarm ID): $SWARM_ID"
echo "---------------------------------------------------"

for (( i=1; i<=MAX_WORKERS; i++ ))
do
    SESSION="worker_swarm_$i"
    
    tmux has-session -t "$SESSION" 2>/dev/null
    
    if [ $? != 0 ]; then
        echo "🚀 [Воркер $i/$MAX_WORKERS] Создаю сессию: $SESSION"
        tmux new-session -d -s "$SESSION"
        
        # Запуск команды
        tmux send-keys -t "$SESSION" "$CMD >> ${SESSION}.log 2>&1" C-m
        
        if [ "$i" -lt "$MAX_WORKERS" ]; then
            echo "⏳ Жду $STAGGER_DELAY сек..."
            sleep $STAGGER_DELAY
        fi
    else
        echo "⚠️ Сессия $SESSION уже существует, пропускаю."
    fi
done

echo "==================================================="
echo "✅ Рой из $MAX_WORKERS процессов успешно запущен!"
echo "👉 Подключиться к первому: tmux attach -t worker_swarm_1"