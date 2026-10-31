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
    "5:40,41,42,43" \
    "12:13,14" \
    "25:7"; do
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
echo "📊 [Шаг 5/5] Переоценка валидационных метрик win_06 (H=25s) на полном OOS-хвосте 187,500 сек..." | tee -a logs/gpu_retrain.log
CUDA_VISIBLE_DEVICES="0" $PYTHON -u -c '
import json, shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from _tools.colab_neural_trainer import ensure_mmap_cache, load_window_arrays, evaluate_oos_probs, get_horizon_defaults
from _tools.colab_models import make_gpu_sequence_dataset

cache_dir = ensure_mmap_cache("data/prl_neural_colab_data.npz")
H, win_idx = 25, 6
_, seq_len = get_horizon_defaults(H)
w_data = load_window_arrays(cache_dir, H, win_idx)
X_tr, X_te = w_data["X_tr"], w_data["X_te"]
y_te, dt_te, ts_te = w_data["y_te"], w_data["dt_te"], w_data["ts_te"]
mean = np.nanmean(X_tr, axis=0)
std = np.nanstd(X_tr, axis=0) + 1e-6
X_te_norm = np.clip((X_te - mean) / std, -5.0, 5.0).astype(np.float16)
y_te_oh = tf.keras.utils.to_categorical(y_te, num_classes=7).astype(np.float32)
val_ds = make_gpu_sequence_dataset(X_te_norm, y_te_oh[seq_len - 1:], seq_len=seq_len, batch_size=2048, shuffle=False, sample_stride=1)
y_true_oos = y_te[seq_len - 1:]
dt_true_oos = dt_te[seq_len - 1:]
new_test_ts = [str(ts_te[0])[:19], str(ts_te[-1])[:19]]

for arch in ["conv1d_gru", "causal_cnn", "attention", "mlp_bottleneck"]:
    k_path = Path(f"models/prl/{arch}_h25s/model_win_06.keras")
    m_path = Path(f"models/prl/{arch}_h25s/meta_win_06.json")
    p_m_path = Path(f"prl_ensemble_pack_h25s/{arch}_h25s/meta_win_06.json")
    model = tf.keras.models.load_model(k_path, compile=False)
    probs = model.predict(val_ds, verbose=0)
    m = evaluate_oos_probs(probs, y_true_oos, dt_true_oos)
    meta = json.load(open(m_path))
    old_m = meta.get("metrics_oos", {})
    meta["test_ts_range"] = new_test_ts
    meta["metrics_oos"] = {
        "ic_dt_ticks": round(float(m["ic"]), 4),
        "roc_auc_ovr": round(float(m["auc"]), 4),
        "log_loss": round(float(m["log_loss"]), 4),
        "top2pct_mean_ticks": round(float(m["top2_pnl"]), 2),
    }
    json.dump(meta, open(m_path, "w"), indent=2)
    if p_m_path.parent.exists():
        json.dump(meta, open(p_m_path, "w"), indent=2)
    old_ic, new_ic = old_m.get("ic_dt_ticks", 0.0), meta["metrics_oos"]["ic_dt_ticks"]
    old_auc, new_auc = old_m.get("roc_auc_ovr", 0.0), meta["metrics_oos"]["roc_auc_ovr"]
    old_t2, new_t2 = old_m.get("top2pct_mean_ticks", 0.0), meta["metrics_oos"]["top2pct_mean_ticks"]
    print(f"  ✅ [{arch:15s} | win_06] OOS (187,500s): IC {old_ic:+.4f} -> {new_ic:+.4f} | AUC {old_auc:.4f} -> {new_auc:.4f} | Top2 {old_t2:+.2f} -> {new_t2:+.2f}t")
' 2>&1 | tee -a logs/gpu_retrain.log

echo "" | tee -a logs/gpu_retrain.log
echo "========================================================================================" | tee -a logs/gpu_retrain.log
echo "✅ ПОЛНОЕ ПЕРЕОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ PRL/USDT УСПЕШНО ЗАВЕРШЕНО!" | tee -a logs/gpu_retrain.log
echo "========================================================================================" | tee -a logs/gpu_retrain.log
