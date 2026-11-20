#!/usr/bin/env bash
set -uo pipefail
cd /home/restorator/crypto_bot
mkdir -p /home/restorator/crypto_bot/data/logs
echo "=== [$(date '+%Y-%m-%d %H:%M:%S %Z')] START Lesson 27 Multimodal MoE 8.5h Multi-Wave Trainer ==="
/home/restorator/crypto_bot/.venv/bin/python -u /home/restorator/crypto_bot/_tools/lesson_27_multimodal_moe_trainer.py \
  --time-budget-hours 8.5 \
  --max-iters 1600 \
  --wakeup-iters 45 \
  --explore-max-iters 140 \
  --explore-patience 40 \
  --zoom1-iters 34 \
  --zoom2-iters 28 \
  --min-delta-bp 0.18
echo "=== [$(date '+%Y-%m-%d %H:%M:%S %Z')] DONE Lesson 27 Multimodal MoE 8.5h Multi-Wave Trainer ==="
exec bash
