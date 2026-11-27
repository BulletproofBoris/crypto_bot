#!/usr/bin/env bash
set -euo pipefail
cd /home/restorator/crypto_bot
rm -rf models/prl/order3_champion/bak_l93b_polish logs/l93_order35_cma_vault/l93a_*.log
/home/restorator/crypto_bot/.venv/bin/python _tools/train_l93_order35_cma_vault.py --iters-vault 10 --iters-polish 0 2>&1 | tee logs/l93_order35_cma_vault/tmux_session.log
echo "=== TMUX L93 Q2 COMPLETED AT $(date -Is) ===" | tee -a logs/l93_order35_cma_vault/tmux_session.log
