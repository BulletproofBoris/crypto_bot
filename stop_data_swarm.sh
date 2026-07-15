#!/bin/bash

# Префикс сессий, которые мы хотим убить (для подготовки данных)
SESSION_PREFIX="data_worker_"

echo "🛑 Останавливаю рой подготовки данных..."

# Получаем список сессий, фильтруем по префиксу и вытаскиваем их имена
SESSIONS=$(tmux ls 2>/dev/null | grep "^$SESSION_PREFIX" | cut -d: -f1)

if [ -z "$SESSIONS" ]; then
    echo "✅ Активных воркеров с префиксом '$SESSION_PREFIX' не найдено."
    exit 0
fi

# Убиваем каждую найденную сессию
for SESSION in $SESSIONS
do
    echo "🔪 Завершаю сессию: $SESSION"
    tmux kill-session -t "$SESSION"
done

echo "==================================================="
echo "✅ Рой подготовки данных успешно остановлен!"
echo "==================================================="