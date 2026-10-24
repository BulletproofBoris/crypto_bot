#!/usr/bin/env bash
set -e

cd /home/restorator/crypto_bot
mkdir -p logs

LOG_FILE="logs/overnight_train_$(date +%Y%m%d_%H%M%S).log"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "🌙 ЗАПУСК НОЧНОГО ПОСЛЕДОВАТЕЛЬНОГО ОБУЧЕНИЯ НОВЫХ ОКОН (H=5, 12, 25)" | tee -a "$LOG_FILE"
echo "   Старт: $(date)" | tee -a "$LOG_FILE"
echo "   Лог:   $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

# 1. Обучение LightGBM Rolling Swarm для H=5 (24 окна), H=12 (8 окон), H=25 (4 окна)
echo -e "\n[1/5] 🌲 Обучение пула экспертов LightGBM (H=5, 12, 25)..." | tee -a "$LOG_FILE"
.venv/bin/python -u _tools/train_rolling_swarm.py --coins prl --horizons 5,12,25 2>&1 | tee -a "$LOG_FILE"

# 2. Обучение новых окон нейросетей H=5 (новые окна: 0..4, 7, 19..23; обученные 5..18 пропускаются автоматически)
echo -e "\n[2/5] 🧬 Обучение новых окон нейросетей H=5s (всего 24 окна)..." | tee -a "$LOG_FILE"
.venv/bin/python -u -m _tools.colab_neural_trainer --data-file data/prl_neural_colab_data.npz --horizon 5 --windows all 2>&1 | tee -a "$LOG_FILE"

# 3. Обучение новых окон нейросетей H=12 (новые окна: 0, 1; обученные 2..7 пропускаются автоматически)
echo -e "\n[3/5] 🧬 Обучение новых окон нейросетей H=12s (всего 8 окон)..." | tee -a "$LOG_FILE"
.venv/bin/python -u -m _tools.colab_neural_trainer --data-file data/prl_neural_colab_data.npz --horizon 12 --windows all 2>&1 | tee -a "$LOG_FILE"

# 4. Обучение новых окон нейросетей H=25 (новое окно: 0; обученные 1..3 пропускаются автоматически)
echo -e "\n[4/5] 🧬 Обучение новых окон нейросетей H=25s (всего 4 окна)..." | tee -a "$LOG_FILE"
.venv/bin/python -u -m _tools.colab_neural_trainer --data-file data/prl_neural_colab_data.npz --horizon 25 --windows all 2>&1 | tee -a "$LOG_FILE"

# 5. Очистка старых кэшей вероятностей и пересборка 72-часового триадного кэша для RL-агента
echo -e "\n[5/5] 🔄 Пересборка триадного кэша RL-агента с новыми окнами..." | tee -a "$LOG_FILE"
rm -f data/cache/lgb_probs_prl_*.npy data/cache/nn_preds_prl_*.npy data/cache/rl_triad_lob_cache_prl_72h.npz
.venv/bin/python -u -c "
from _tools.train_rl_execution import build_or_load_rl_cache
build_or_load_rl_cache(coin='prl', tail_hours=72, force_rebuild=True)
" 2>&1 | tee -a "$LOG_FILE"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "🎉 НОЧНОЕ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО: $(date)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
