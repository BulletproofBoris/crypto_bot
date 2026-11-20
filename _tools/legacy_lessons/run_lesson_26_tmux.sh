#!/usr/bin/env bash
set -uo pipefail
cd /home/restorator/crypto_bot
mkdir -p /home/restorator/crypto_bot/data/logs
echo "=== [$(date '+%Y-%m-%d %H:%M:%S %Z')] START Lesson 26 Endogenous Nonlinear Trainer ==="
/home/restorator/crypto_bot/.venv/bin/python -u /home/restorator/crypto_bot/_tools/lesson_26_endogenous_nonlinear_trainer.py --iterations 560 --phase1-iters 70 --phase2-iters 260 --phase3-iters 440 --plateau-patience 80 --plateau-min-delta-bp 0.25
echo "=== [$(date '+%Y-%m-%d %H:%M:%S %Z')] DONE Lesson 26 Endogenous Nonlinear Trainer ==="
exec bash
