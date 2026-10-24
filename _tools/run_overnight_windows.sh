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

# 1. Экспорт свежих окон в NPZ и сброс старого mmap-кэша
echo -e "\n[1/5] 📦 Сборка актуального NPZ-архива окон для нейросетей..." | tee -a "$LOG_FILE"
rm -rf data/_mmap_cache
.venv/bin/python -u -m _tools.export_colab_dataset 2>&1 | tee -a "$LOG_FILE"

# 2. Обучение LightGBM Rolling Swarm для H=5 (28 окон), H=12 (9 окон), H=25 (4 окна)
echo -e "\n[2/5] 🌲 Обучение пула экспертов LightGBM (H=5, 12, 25)..." | tee -a "$LOG_FILE"
.venv/bin/python -u _tools/train_rolling_swarm.py --coins prl --horizons 5,12,25 2>&1 | tee -a "$LOG_FILE"

# 3. Обучение 4 новых окон нейросетей H=5 (win_00..win_03; обученные win_04..win_27 пропускаются автоматически)
echo -e "\n[3/5] 🧬 Обучение 4 новых окон нейросетей H=5s (win_00..win_03 из 28)..." | tee -a "$LOG_FILE"
.venv/bin/python -u -m _tools.colab_neural_trainer --data-file data/prl_neural_colab_data.npz --horizon 5 --windows all 2>&1 | tee -a "$LOG_FILE"

# 4. Обучение 1 нового окна нейросетей H=12 (win_00; обученные win_01..win_08 пропускаются автоматически)
echo -e "\n[4/5] 🧬 Обучение 1 нового окна нейросетей H=12s (win_00 из 9)..." | tee -a "$LOG_FILE"
.venv/bin/python -u -m _tools.colab_neural_trainer --data-file data/prl_neural_colab_data.npz --horizon 12 --windows all 2>&1 | tee -a "$LOG_FILE"

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
