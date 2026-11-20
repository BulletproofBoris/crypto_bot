#!/usr/bin/env bash
set -uo pipefail
cd /home/restorator/crypto_bot
mkdir -p /home/restorator/crypto_bot/data/logs
echo "=== [$(date '+%Y-%m-%d %H:%M:%S %Z')] RESUME BigONE 1st-Order Swarm (--only bo --lgb-workers 2 --bo-runs 4) ===" | tee -a /home/restorator/crypto_bot/data/logs/swarm_st_bo_retrain.log
/home/restorator/crypto_bot/.venv/bin/python -u /home/restorator/crypto_bot/_tools/train_st_missing_and_bo_full_swarm.py --only bo --phase all --lgb-workers 2 --bo-runs 4 2>&1 | tee -a /home/restorator/crypto_bot/data/logs/swarm_st_bo_retrain.log
echo "=== [$(date '+%Y-%m-%d %H:%M:%S %Z')] DONE BigONE + SafeTrade Summary ===" | tee -a /home/restorator/crypto_bot/data/logs/swarm_st_bo_retrain.log
/home/restorator/crypto_bot/.venv/bin/python -u -c "from _tools.train_st_missing_and_bo_full_swarm import print_swarm_summary; print_swarm_summary('prl', '_mmap_cache'); print_swarm_summary('prl_bo', '_mmap_cache_bo')" 2>&1 | tee -a /home/restorator/crypto_bot/data/logs/swarm_st_bo_retrain.log
exec bash
