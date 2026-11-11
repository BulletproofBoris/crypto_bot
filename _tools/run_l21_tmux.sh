#!/usr/bin/env bash
cd /home/restorator/crypto_bot || exit 1
/home/restorator/crypto_bot/.venv/bin/python -u _tools/lesson_21_pareto_decoupled_4venue_trainer.py --iters 400 --seed 2026212 --no-early-stop
exec bash
