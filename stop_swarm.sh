#!/bin/bash

# Префикс сессий, которые создал наш менеджер
SESSION_PREFIX="worker_swarm_"

echo "🛑 Запуск протокола остановки Swarm-пула..."

# Получаем список всех tmux-сессий, фильтруем по префиксу и забираем только их имена
SESSIONS=$(tmux ls 2>/dev/null | grep "^$SESSION_PREFIX" | cut -d: -f1)

# Проверяем, есть ли вообще такие сессии
if [ -z "$SESSIONS" ]; then
    echo "✅ Активных воркеров с префиксом '$SESSION_PREFIX' не найдено. Память чиста."
    exit 0
fi

# Проходимся по каждой найденной сессии и убиваем её
for SESSION in $SESSIONS
do
    echo "🔪 Завершаю работу воркера: $SESSION"
    tmux kill-session -t "$SESSION"
done

echo "==================================================="
echo "✅ Рой успешно деактивирован. Вся VRAM и CPU освобождены!"