#!/bin/bash

# Префикс сессий, которые мы хотим убить (для подготовки данных)
SESSION_PREFIX="data_worker_"

echo "🛑 Останавливаю рой подготовки данных..."

# Получаем список сессий, фильтруем по префиксу и вытаскиваем их имена
SESSIONS=$(tmux ls 2>/dev/null | grep "^$SESSION_PREFIX" | cut -d: -f1)

if [ -n "$SESSIONS" ]; then
    # Убиваем каждую найденную сессию
    for SESSION in $SESSIONS
    do
        echo "🔪 Завершаю сессию: $SESSION"
        tmux kill-session -t "$SESSION"
    done
else
    echo "✅ Активных tmux-сессий с префиксом '$SESSION_PREFIX' не найдено."
fi

# Для надежности добиваем фоновые процессы Python, которые могли отвязаться от tmux
PIDS=$(pgrep -f "python -m _tools.init_dataset")
if [ -n "$PIDS" ]; then
    echo "🔪 Добиваю зависшие процессы подготовки данных..."
    pkill -f "python -m _tools.init_dataset"
fi

echo "==================================================="
echo "✅ Рой подготовки данных успешно остановлен!"
echo "==================================================="