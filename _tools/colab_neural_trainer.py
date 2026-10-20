#!/usr/bin/env python3
"""
Движок обучения нейросетевого роя (Conv1D+GRU, Causal CNN, Attention, MLP Bottleneck)
по асинхронным Walk-Forward окнам (H=5: 12 окон, H=12: 5 окон, H=25: 2 окна) для Google Colab / GPU.
Все основные гиперпараметры вынесены в аргументы функции и CLI.
"""

import os
import gc
import time
import json
import shutil
import sqlite3
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from _tools.colab_models import (
    MODEL_FACTORY,
    make_gpu_sequence_dataset,
    SmartBacktrackCallback,
    ElasticPatienceProfiler,
    GPUMonitorCallback,
)

CLASSES_VEC = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)


def get_horizon_defaults(horizon: int) -> tuple[float, int]:
    """Возвращает канонический K и длину последовательности L = H * K."""
    if int(horizon) == 25:
        return 3.0, 75
    elif int(horizon) in (12, 13, 14, 15):
        return 3.5, int(round(horizon * 3.5))
    return 3.0, int(round(horizon * 3.0))


def evaluate_oos_probs(probs: np.ndarray, y_true: np.ndarray, dt_ticks: np.ndarray) -> dict:
    """Считает OOS IC, ROC AUC OVR, Log-Loss и доходность Top-2% импульсов."""
    p_clip = np.clip(probs, 1e-7, 1.0 - 1e-7)
    p_clip = p_clip / np.sum(p_clip, axis=1, keepdims=True)
    y_oh = tf.keras.utils.to_categorical(y_true, num_classes=7).astype(np.float32)
    ez = np.dot(p_clip, CLASSES_VEC)
    ic = float(np.corrcoef(ez, dt_ticks)[0, 1]) if np.std(ez) > 1e-6 else 0.0
    try:
        auc = float(roc_auc_score(y_true, p_clip, multi_class="ovr"))
    except Exception:
        auc = 0.5
    ll = float(-np.mean(np.sum(y_oh * np.log(p_clip), axis=1)))
    top2_th = np.quantile(ez, 0.98)
    top2_mask = ez >= top2_th
    top2_pnl = float(np.mean(dt_ticks[top2_mask])) if np.sum(top2_mask) > 0 else 0.0
    return {"ic": ic, "auc": auc, "log_loss": ll, "top2_pnl": top2_pnl}


def load_window_arrays(raw_npz, horizon: int, win_idx: int = 0) -> dict:
    """Извлекает обучающую и тестовую выборку для заданного горизонта H и индекса окна win_idx."""
    h_key = f"h{int(horizon)}"
    if "X_all" in raw_npz.files and f"{h_key}_tr_idx_{win_idx:02d}" in raw_npz.files:
        tr_idx = raw_npz[f"{h_key}_tr_idx_{win_idx:02d}"]
        te_idx = raw_npz[f"{h_key}_te_idx_{win_idx:02d}"]
        X_all = raw_npz["X_all"]
        ts_all = raw_npz["ts_all"]
        X_tr, X_te = X_all[tr_idx].astype(np.float32), X_all[te_idx].astype(np.float32)
        y_tr = raw_npz[f"y_all_{h_key}"][tr_idx].astype(np.int32)
        y_te = raw_npz[f"y_all_{h_key}"][te_idx].astype(np.int32)
        dt_te = raw_npz[f"dt_all_{h_key}"][te_idx].astype(np.float32)
        ts_tr, ts_te = ts_all[tr_idx], ts_all[te_idx]
    else:
        X_tr, X_te = raw_npz["X_train"].astype(np.float32), raw_npz["X_test"].astype(np.float32)
        y_tr = raw_npz[f"y_train_{h_key}"].astype(np.int32)
        y_te = raw_npz[f"y_test_{h_key}"].astype(np.int32)
        dt_te = raw_npz[f"dt_test_{h_key}"].astype(np.float32)
        ts_tr, ts_te = raw_npz["ts_train"], raw_npz["ts_test"]
    return {
        "X_tr": X_tr, "X_te": X_te,
        "y_tr": y_tr, "y_te": y_te, "dt_te": dt_te,
        "ts_tr": ts_tr, "ts_te": ts_te,
    }



def train_window_swarm(
    raw_npz,
    horizon: int = 12,
    win_idx: int = 0,
    archs: list[str] = None,
    runs_per_arch: int = 10,
    max_epochs: int = 10,
    hard_prune_epoch: int = 5,
    top_k_per_arch: int = 3,
    batch_size: int = 2048,
    lr: float = 2e-3,
    l2_reg: float = 1e-4,
    pack_dir: str = None,
) -> dict:
    """Обучает пул нейросетевых архитектур для одного окна win_idx и сохраняет model_win_XX.keras + meta_win_XX.json."""
    if archs is None:
        archs = ["conv1d+gru", "causal_cnn", "attention", "mlp_bottleneck"]
    k_ratio, seq_len = get_horizon_defaults(horizon)
    win_tag = f"win_{win_idx:02d}"
    pack_dir = pack_dir or f"prl_ensemble_pack_h{horizon}s"
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs(pack_dir, exist_ok=True)

    w_data = load_window_arrays(raw_npz, horizon, win_idx)
    X_tr, X_te = w_data["X_tr"], w_data["X_te"]
    y_tr, y_te, dt_te = w_data["y_tr"], w_data["y_te"], w_data["dt_te"]
    ts_tr, ts_te = w_data["ts_tr"], w_data["ts_te"]
    n_features = X_tr.shape[1]

    print(f"\n{'='*88}")
    print(f"🚀 ОБУЧЕНИЕ ОКНА {win_tag.upper()} | H={horizon}с (K={k_ratio}, L={seq_len}с) | Train={len(X_tr):,} ({ts_tr[0][:16]}..{ts_tr[-1][:16]})")
    print(f"{'='*88}")

    mean = np.nanmean(X_tr, axis=0)
    std = np.nanstd(X_tr, axis=0) + 1e-6
    X_tr_norm = np.clip((X_tr - mean) / std, -5.0, 5.0).astype(np.float16)
    X_te_norm = np.clip((X_te - mean) / std, -5.0, 5.0).astype(np.float16)
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes=7).astype(np.float32)
    y_te_oh = tf.keras.utils.to_categorical(y_te, num_classes=7).astype(np.float32)

    base_w = compute_class_weight("balanced", classes=np.arange(7), y=y_tr[seq_len - 1:])
    custom_weights = dict(enumerate(base_w))
    custom_weights[0] *= 1.35
    custom_weights[1] *= 1.25
    custom_weights[2] *= 1.10

    train_ds = make_gpu_sequence_dataset(X_tr_norm, y_tr_oh[seq_len - 1:], seq_len=seq_len, batch_size=batch_size, shuffle=True)
    val_ds = make_gpu_sequence_dataset(X_te_norm, y_te_oh[seq_len - 1:], seq_len=seq_len, batch_size=batch_size, shuffle=False)
    y_true_oos = y_te[seq_len - 1:]
    dt_true_oos = dt_te[seq_len - 1:]

    leaderboard = []
    oos_predictions = {}

    for arch in archs:
        arch_candidates = []
        arch_losses = []
        for run_idx in range(1, runs_per_arch + 1):
            tf.keras.backend.clear_session()
            gc.collect()
            tf.random.set_seed(1000 + win_idx * 100 + run_idx * 17)
            np.random.seed(1000 + win_idx * 100 + run_idx * 17)
            t_run = time.time()

            model = MODEL_FACTORY[arch](seq_len, n_features, n_classes=7, l2_reg=l2_reg)
            opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0)
            model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.06), metrics=["accuracy"])

            temp_w = f"trained_models/tmp_{arch.replace('+', '_')}_{win_tag}_r{run_idx}.weights.h5"
            profiler = ElasticPatienceProfiler(arch_losses, max_epochs=max_epochs, hard_prune_epoch=hard_prune_epoch, hard_prune_z=4.0, max_z_score=2.0)
            cbs = [
                tf.keras.callbacks.ModelCheckpoint(filepath=temp_w, save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True, verbose=0),
                SmartBacktrackCallback(best_weights_path=temp_w, monitor_loss="val_loss", factor=0.5, patience=2, min_lr=1e-5, max_rollbacks=3),
                tf.keras.callbacks.TerminateOnNaN(),
                profiler,
            ]
            model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, class_weight=custom_weights, callbacks=cbs, verbose=0)
            elapsed = round(time.time() - t_run, 1)
            if os.path.exists(temp_w):
                model.load_weights(temp_w)
                os.remove(temp_w)

            probs = model.predict(val_ds, verbose=0)
            metrics = evaluate_oos_probs(probs, y_true_oos, dt_true_oos)
            arch_losses.append(metrics["log_loss"])
            cand_path = f"trained_models/cand_{arch.replace('+', '_')}_{win_tag}_r{run_idx:02d}.keras"
            model.save(cand_path)
            arch_candidates.append((metrics["log_loss"], -metrics["ic"], run_idx, cand_path, probs, metrics, elapsed, model.count_params()))
            print(f"  [{arch.upper()} | {win_tag} | r{run_idx:02d}/{runs_per_arch}] ({elapsed}с) -> Loss: {metrics['log_loss']:.4f} | IC: {metrics['ic']:+.4f} | AUC: {metrics['auc']:.4f} | Top-2%: {metrics['top2_pnl']:+.2f}")
            del model

        arch_candidates.sort(key=lambda x: (round(x[0], 3), x[1]))
        _, _, best_r, best_path, best_probs, best_m, _, best_params = arch_candidates[0]
        arch_dir = os.path.join(pack_dir, f"{arch.replace('+', '_')}_h{horizon}s")
        os.makedirs(arch_dir, exist_ok=True)
        shutil.copy2(best_path, os.path.join(arch_dir, f"model_{win_tag}.keras"))

        meta = {
            "window_id": win_tag, "model_type": arch, "coin": "PRL", "H": int(horizon), "seq_len": seq_len,
            "train_ts_range": [str(ts_tr[0])[:19], str(ts_tr[-1])[:19]],
            "test_ts_range": [str(ts_te[0])[:19], str(ts_te[-1])[:19]],
            "metrics_oos": {"ic_dt_ticks": round(best_m["ic"], 4), "roc_auc_ovr": round(best_m["auc"], 4), "log_loss": round(best_m["log_loss"], 4), "top2pct_mean_ticks": round(best_m["top2_pnl"], 2)},
            "feature_norm": {"mean": mean.tolist(), "std": std.tolist()},
            "weights_file": f"model_{win_tag}.keras",
        }
        with open(os.path.join(arch_dir, f"meta_{win_tag}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        for c_item in arch_candidates:
            if os.path.exists(c_item[3]):
                os.remove(c_item[3])
        oos_predictions[arch] = best_probs
        leaderboard.append({"window": win_tag, "model": f"{arch} (r{best_r})", "params": best_params, **best_m})

    return {"window": win_tag, "leaderboard": leaderboard, "oos_predictions": oos_predictions}



def run_colab_pipeline(
    data_file: str = "prl_neural_colab_data.npz",
    file_id: str = "1N2YvZqCiZuKcTYoNjne2-Q4dc-ZzS38N",
    horizon: int = 12,
    windows: str = "all",
    archs: str = "conv1d+gru,causal_cnn,attention,mlp_bottleneck",
    runs_per_arch: int = 10,
    max_epochs: int = 10,
    hard_prune_epoch: int = 5,
    top_k_per_arch: int = 3,
    batch_size: int = 2048,
    lr: float = 2e-3,
    l2_reg: float = 1e-4,
):
    """Запускает полный цикл загрузки данных, обучения выбранных окон и упаковки архива."""
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    for cand in [data_file, f"data/{data_file}", f"/content/{data_file}", f"/content/drive/MyDrive/{data_file}"]:
        if os.path.exists(cand):
            data_file = cand
            break
    if not os.path.exists(data_file):
        import subprocess
        direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        print(f"☁️ Прямое скачивание {data_file} с Google Drive (ID={file_id})...")
        subprocess.run(["curl", "-L", "-s", "-o", data_file, direct_url], check=False)
        if not os.path.exists(data_file) or os.path.getsize(data_file) < 10_000_000:
            import gdown
            gdown.download(id=file_id, output=data_file, quiet=False, fuzzy=True)

    raw_npz = np.load(data_file, allow_pickle=True)
    h_key = f"h{int(horizon)}"
    if windows.strip().lower() == "all":
        default_counts = {5: 12, 12: 5, 25: 2}
        n_w = int(raw_npz[f"{h_key}_n_windows"]) if f"{h_key}_n_windows" in raw_npz.files else default_counts.get(int(horizon), 1)
        win_list = list(range(n_w))
    else:
        win_list = [int(x.strip()) for x in windows.split(",") if x.strip()]

    arch_list = [a.strip() for a in archs.split(",") if a.strip()]
    pack_dir = f"prl_ensemble_pack_h{horizon}s"
    all_lb = []

    for w_idx in win_list:
        res = train_window_swarm(
            raw_npz=raw_npz, horizon=int(horizon), win_idx=w_idx, archs=arch_list,
            runs_per_arch=runs_per_arch, max_epochs=max_epochs, hard_prune_epoch=hard_prune_epoch,
            top_k_per_arch=top_k_per_arch, batch_size=batch_size, lr=lr, l2_reg=l2_reg, pack_dir=pack_dir,
        )
        all_lb.extend(res["leaderboard"])

    df_lb = pd.DataFrame(all_lb)
    print("\n🏆 СВОДНЫЙ ЛИДЕРБОРД WALK-FORWARD ОКОН:")
    print(df_lb.to_string(index=False))
    shutil.make_archive(pack_dir, "zip", root_dir=".", base_dir=pack_dir)
    print(f"\n📦 Архив {pack_dir}.zip успешно собран!")
    return df_lb


def main():
    parser = argparse.ArgumentParser(description="Colab Neural Swarm Trainer (Walk-Forward Windows)")
    parser.add_argument("--data-file", type=str, default="prl_neural_colab_data.npz")
    parser.add_argument("--file-id", type=str, default="1N2YvZqCiZuKcTYoNjne2-Q4dc-ZzS38N")
    parser.add_argument("--horizon", type=int, default=12, help="Горизонт H: 5, 12 или 25")
    parser.add_argument("--windows", type=str, default="all", help="Окна через запятую (0,1,2) или all")
    parser.add_argument("--archs", type=str, default="conv1d+gru,causal_cnn,attention,mlp_bottleneck")
    parser.add_argument("--runs", type=int, default=10, help="Ранов на архитектуру")
    parser.add_argument("--epochs", type=int, default=10, help="Макс. эпох на ран")
    parser.add_argument("--prune-epoch", type=int, default=5, help="Эпоха отсечения слабых ранов")
    parser.add_argument("--top-k", type=int, default=3, help="Топ-K ранов внутри архитектуры")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--l2-reg", type=float, default=1e-4)
    args = parser.parse_args()

    run_colab_pipeline(
        data_file=args.data_file, file_id=args.file_id, horizon=args.horizon, windows=args.windows,
        archs=args.archs, runs_per_arch=args.runs, max_epochs=args.epochs, hard_prune_epoch=args.prune_epoch,
        top_k_per_arch=args.top_k, batch_size=args.batch_size, lr=args.lr, l2_reg=args.l2_reg,
    )


if __name__ == "__main__":
    main()


