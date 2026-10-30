#!/usr/bin/env bash
# ==============================================================================
# Полное переобучение всех моделей PRL/USDT на GPU (NVIDIA RTX 3080)
# Окна нарезаются строго с самого начала датасета (tr_s_pos = 0, step, 2*step...)
# Количество ранов нейросетей: 20 на каждую архитектуру в каждом окне
# Режим: 2 параллельные GPU-сессии (по 4.2 ГБ VRAM каждая = 8.4 ГБ из 10 ГБ)
# ==============================================================================
set -e

cd "$(dirname "$0")/.."
mkdir -p logs
PYTHON=".venv/bin/python"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_MEMORY_LIMIT_MB=${TF_GPU_MEMORY_LIMIT_MB:-4200}
export TF_CPP_MIN_LOG_LEVEL=2

RUNS=${RUNS:-20}
EPOCHS=${EPOCHS:-10}
PRUNE_EPOCH=${PRUNE_EPOCH:-5}
CLEAN_START=${CLEAN_START:-0}
SKIP_PREP=${SKIP_PREP:-0}
WAIT_FOR_WIN15=${WAIT_FOR_WIN15:-0}

echo "========================================================================================"
echo "🚀 ЗАПУСК ПАРАЛЛЕЛЬНОГО ПЕРЕОБУЧЕНИЯ МОДЕЛЕЙ PRL/USDT НА GPU (RTX 3080 — 2 СЕССИИ)"
echo "   Режим: Нарезка окон с самого начала (index 0) | Ранов на архитектуру: ${RUNS}"
echo "   Лимит VRAM на воркер: ${TF_GPU_MEMORY_LIMIT_MB} MiB (2 воркера x 4.2 ГБ = 8.4 ГБ)"
echo "========================================================================================"

if [ "${WAIT_FOR_WIN15}" = "1" ]; then
    echo "⏳ Ожидание завершения текущего окна win_15 (H=5s) перед переключением на 2 параллельные сессии..."
    while [ ! -f "models/prl/mlp_bottleneck_h5s/meta_win_15.json" ]; do
        sleep 2
    done
    sleep 2
    echo "✅ Окно win_15 полностью завершено! Останавливаем одиночную сессию и запускаем 2 параллельных потока..." | tee -a logs/gpu_retrain.log
    pkill -9 -f "_tools.colab_neural_trainer" || true
    sleep 3
fi

if [ "${CLEAN_START}" = "1" ]; then
    echo "🧹 [Шаг 1/5] Очистка старых чекпоинтов окон и временных кэшей..."
    rm -f models/prl/h*s_k*/model_win_*.txt models/prl/h*s_k*/meta_win_*.json
    rm -f models/prl/*_h*s/model_win_*.keras models/prl/*_h*s/meta_win_*.json models/prl/*_h*s/progress_win_*.json
    rm -rf prl_ensemble_pack_h*s prl_ensemble_pack_h*s.zip
    rm -rf data/neural_mmap_cache data/cache/lgb_probs_prl_* data/cache/nn_preds_prl_*
fi

if [ "${SKIP_PREP}" != "1" ]; then
    echo ""
    echo "📦 [Шаг 2/5] Экспорт полного пакета окон с самого начала в data/prl_neural_colab_data.npz..."
    $PYTHON -u _tools/export_colab_dataset.py

    echo ""
    echo "🌲 [Шаг 3/5] Обучение пула экспертов LightGBM (Учителей) по всем окнам (H = 5, 12, 25, 60с)..."
    $PYTHON -u _tools/train_rolling_swarm.py --coins prl --horizons 5,12,25,60
fi

echo "" | tee -a logs/gpu_retrain.log
echo "🧬 [Шаг 4/5] Последовательное обучение оставшихся длинных горизонтов (H=25s, H=60s) в 1 поток GPU (вся VRAM RTX 3080, --runs ${RUNS})..." | tee -a logs/gpu_retrain.log

echo "⚡ Проверка mmap-кэша data/neural_mmap_cache..." | tee -a logs/gpu_retrain.log
CUDA_VISIBLE_DEVICES="-1" $PYTHON -u -c '
from _tools.colab_neural_trainer import ensure_mmap_cache
ensure_mmap_cache("data/prl_neural_colab_data.npz")
'

for SPEC in \
    "25:2,3,4,5,6" \
    "60:0,1"; do
    H="${SPEC%%:*}"
    WINS="${SPEC#*:}"
    echo "🔥 [GPU 1-ПОТОК] Горизонт H=${H}с | Окна: ${WINS}" | tee -a logs/gpu_retrain.log
    $PYTHON -u -m _tools.colab_neural_trainer \
        --data-file data/prl_neural_colab_data.npz \
        --horizon "${H}" \
        --windows "${WINS}" \
        --archs "conv1d+gru,causal_cnn,attention,mlp_bottleneck" \
        --runs "${RUNS}" \
        --epochs "${EPOCHS}" \
        --prune-epoch "${PRUNE_EPOCH}" 2>&1 | tee -a logs/gpu_retrain.log
done

echo "" | tee -a logs/gpu_retrain.log
echo "🤖 [Шаг 5/5] Пересборка RL-кэша триады экспертов и переобучение MaskablePPO на RTX 3080..." | tee -a logs/gpu_retrain.log
$PYTHON -u _tools/train_rl_execution.py --coin prl --rebuild-cache --steps 500000 2>&1 | tee -a logs/gpu_retrain.log

echo "" | tee -a logs/gpu_retrain.log
echo "========================================================================================" | tee -a logs/gpu_retrain.log
echo "✅ ПОЛНОЕ ПЕРЕОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ PRL/USDT УСПЕШНО ЗАВЕРШЕНО!" | tee -a logs/gpu_retrain.log
echo "========================================================================================" | tee -a logs/gpu_retrain.log
