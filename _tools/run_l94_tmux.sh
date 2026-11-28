#!/usr/bin/env bash
set -euo pipefail
cd /home/restorator/crypto_bot
export CUDA_VISIBLE_DEVICES=1
export L94_GPU_ID=1
mkdir -p logs/l94_ledger_and_split

{
  echo "=== [$(date -Is)] RESUME L94 STEP A+B ON GPU #1 ONLY (N=4,211,230 READY) ==="
  /home/restorator/crypto_bot/.venv/bin/python _tools/build_l94_write_once_ledger.py --force-rebuild
  /home/restorator/crypto_bot/.venv/bin/python _tools/regime_purged_split_l94.py
  /home/restorator/crypto_bot/.venv/bin/python _tools/test_order3_sim_live_parity.py models/prl/release_champion_v1.pt
  echo "=== [$(date -Is)] L94 STEP A+B COMPLETED SUCCESSFULLY ==="
} 2>&1 | tee -a logs/l94_ledger_and_split/tmux_l94.log
