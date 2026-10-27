#!/usr/bin/env bash
# ==============================================================================
# Конвейер последовательного обучения 3 новых строгих Walk-Forward окон (100% OOS)
#   1. Идемпотентный сдвиг уже обученных нейросетевых чекпоинтов:
#        - H=5s:  win_00..win_31 -> win_02..win_33 (+2 новых окна win_00, win_01)
#        - H=12s: win_00..win_09 -> win_01..win_10 (+1 новое окно win_00)
#        - H=25s: без сдвига (сохраняются 5 строгих 100%-OOS окон win_00..win_04)
#   2. Экспорт актуального архива признаков data/prl_neural_colab_data.npz
#   3. Обучение ансамбля LightGBM (H=5: 34 окна, H=12: 11 окон, H=25: 5 окон)
#   4. Последовательное обучение нейросетей (Conv1D+GRU, Causal CNN, Attention, MLP):
#        - H=5s  (2 новых окна: win_00, win_01)
#        - H=12s (1 новое окно: win_00)
#   5. Пересборка каузального кэша Триады (415ч) и дообучение Maskable PPO агента
#
# Запуск в фоне:
#   nohup bash _tools/run_overnight_windows.sh > /dev/null 2>&1 &
# Мониторинг лога:
#   tail -f logs/overnight_windows.log
# ==============================================================================
set -euo pipefail
{

cd /home/restorator/crypto_bot
mkdir -p logs

LOG_FILE="logs/overnight_windows.log"
PYTHON_BIN="/home/restorator/crypto_bot/.venv/bin/python"
export MPLBACKEND="Agg"
export TF_CPP_MIN_LOG_LEVEL="2"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================================"
echo "🚀 СТАРТ КОНВЕЙЕРА WALK-FORWARD ОБУЧЕНИЯ (3 НОВЫХ СТРОГИХ ОКНА 100% OOS): $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================================"

echo ""
echo "🔄 [Этап 1/5] Идемпотентный сдвиг индексов ранее обученных нейросетевых чекпоинтов..."
"$PYTHON_BIN" -u -c '
import json
import shutil
from pathlib import Path

base = Path("/home/restorator/crypto_bot")
archs = ["conv1d_gru", "causal_cnn", "attention", "mlp_bottleneck"]

# (horizon, shift_by, old_count, expected_win00_train_end_prefix)
shift_plan = [
    (5,  2, 32, "2026-10-25 23:52"),
    (12, 1, 10, "2026-10-24 20:51"),
]

for H, shift, old_cnt, ts_prefix in shift_plan:
    roots = [
        base / "models" / "prl",
        base / f"prl_ensemble_pack_h{H}s",
    ]
    for root in roots:
        for arch in archs:
            d = root / f"{arch}_h{H}s"
            m0 = d / "meta_win_00.json"
            if not m0.exists():
                continue
            with open(m0) as f:
                meta0 = json.load(f)
            tr_end = str(meta0.get("train_ts_range", ["", ""])[1])
            if not tr_end.startswith(ts_prefix):
                print(f"  ⏩ [{d.relative_to(base)}] Сдвиг уже выполнен ранее (win_00 train_end={tr_end})")
                continue
            print(f"  📦 [{d.relative_to(base)}] Сдвиг {old_cnt} окон на +{shift} (win_00..win_{old_cnt-1:02d} -> win_{shift:02d}..win_{old_cnt+shift-1:02d})...")
            for k in range(old_cnt - 1, -1, -1):
                old_tag = f"win_{k:02d}"
                new_tag = f"win_{k + shift:02d}"
                old_k = d / f"model_{old_tag}.keras"
                new_k = d / f"model_{new_tag}.keras"
                old_m = d / f"meta_{old_tag}.json"
                new_m = d / f"meta_{new_tag}.json"
                if old_k.exists():
                    shutil.move(str(old_k), str(new_k))
                if old_m.exists():
                    with open(old_m) as mf:
                        mdata = json.load(mf)
                    mdata["window_id"] = new_tag
                    mdata["weights_file"] = f"model_{new_tag}.keras"
                    with open(new_m, "w") as mf:
                        json.dump(mdata, mf, indent=2)
                    if old_m != new_m and old_m.exists():
                        old_m.unlink()
'

echo ""
echo "📦 [Этап 2/5] Проверка сжатого архива признаков data/prl_neural_colab_data.npz ($(date '+%H:%M:%S'))..."
if [ -f "data/prl_neural_colab_data.npz" ]; then
    echo "  ⏩ Архив data/prl_neural_colab_data.npz уже существует (пропуск повторной упаковки)."
else
    "$PYTHON_BIN" -u _tools/export_colab_dataset.py
fi

echo ""
echo "🌲 [Этап 3/5] Проверка роя LightGBM по всем окнам H=5 (34), H=12 (11), H=25 (5), H=60 (1) ($(date '+%H:%M:%S'))..."
if [ -d "models/prl/h60s_k3" ] && [ -d "models/prl/h12s_k3.5" ]; then
    echo "  ⏩ Все ансамбли LightGBM (H=5, 12, 25, 60) уже обучены (пропуск)."
else
    "$PYTHON_BIN" -u _tools/train_rolling_swarm.py --coins prl --horizons 5,12,25,60
fi

echo ""
echo "🧬 [Этап 4.1/5] Проверка нейросетей H=5s: 2 новых окна win_00, win_01 ($(date '+%H:%M:%S'))..."
if [ -f "models/prl/mlp_bottleneck_h5s/model_win_01.keras" ]; then
    echo "  ⏩ Все 8 нейросетей H=5s (win_00, win_01) уже обучены (пропуск)."
else
    "$PYTHON_BIN" -u -m _tools.colab_neural_trainer \
        --data-file data/prl_neural_colab_data.npz \
        --horizon 5 \
        --windows 0,1 \
        --runs 10 \
        --epochs 10 \
        --prune-epoch 5
fi

echo ""
echo "🧬 [Этап 4.2/5] Дообучение прерванного строгого окна H=12s (win_00: attention r02..10, mlp_bottleneck r01..10) ($(date '+%H:%M:%S'))..."
"$PYTHON_BIN" -u -m _tools.colab_neural_trainer \
    --data-file data/prl_neural_colab_data.npz \
    --horizon 12 \
    --windows 0 \
    --runs 10 \
    --epochs 10 \
    --prune-epoch 5

echo ""
echo "🧬 [Этап 4.3/5] Обучение нового строгого окна H=60s (K=3.0, win_00, с авто-подхватом r08 из бэкапа) ($(date '+%H:%M:%S'))..."
"$PYTHON_BIN" -u -m _tools.colab_neural_trainer \
    --data-file data/prl_neural_colab_data.npz \
    --horizon 60 \
    --windows 0 \
    --runs 10 \
    --epochs 10 \
    --prune-epoch 5

echo ""
echo "🤖 [Этап 5/5] Пересборка каузального кэша Триады (421ч) и обучение Maskable PPO v4 ($(date '+%H:%M:%S'))..."
"$PYTHON_BIN" -u _tools/train_rl_execution.py \
    --coin prl \
    --tail-hours 421.0 \
    --epochs 25 \
    --steps 16384 \
    --split-ts "2026-10-25 00:00:00+03:00"

echo ""
echo "========================================================================================"
echo "✅ КОНВЕЙЕР НОЧНОГО ОБУЧЕНИЯ СТРОГИХ ОКОН УСПЕШНО ЗАВЕРШЕН: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================================"
exit 0
}
