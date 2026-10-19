#!/bin/bash
export SSHPASS='cbqeoR6392Lr'
OPTS="-o StrictHostKeyChecking=no"
HOST="root@81.19.137.18"

echo "1. Installing aiohttp..."
sshpass -e ssh $OPTS $HOST "/opt/crypto_bot/venv/bin/pip install aiohttp"

echo "2. Uploading script..."
sshpass -e scp $OPTS _tools/collect_safetrade_realtime.py $HOST:/opt/crypto_bot/_tools/collect_safetrade_realtime.py

echo "3. Restarting service..."
sshpass -e ssh $OPTS $HOST "systemctl restart safetrade-collector && sleep 2 && systemctl status safetrade-collector --no-pager"
