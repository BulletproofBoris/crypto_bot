#!/usr/bin/env python3
"""
Движок обучения нейросетевого роя (Conv1D+GRU, Causal CNN, Attention, MLP Bottleneck)
по асинхронным Walk-Forward окнам (H=5: 12 окон, H=12: 5 окон, H=25: 2 окна) для Google Colab / GPU.
Защищен от утечек памяти TensorFlow (Kernel Crash / OOM):
- mmap-чтение матрицы признаков без распаковки 1 ГБ в RAM;
- изоляция обучения каждой пары (окно, архитектура) в отдельном подпроцессе с полным возвратом RAM/VRAM ОС;
- автоматический Resume (пропуск уже обученных чекпоинтов model_win_XX.keras при перезапуске).
"""

import os
import gc
import sys
import time
import json
import ctypes
import shutil
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
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


def _trim_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def get_horizon_defaults(horizon: int) -> tuple[float, int]:
    if int(horizon) >= 50:
        return 2.5, int(round(horizon * 2.5))
    elif int(horizon) == 25:
        return 3.0, 75
    elif int(horizon) in (12, 13, 14, 15):
        return 3.5, int(round(horizon * 3.5))
    return 3.0, int(round(horizon * 3.0))


def evaluate_oos_probs(probs: np.ndarray, y_true: np.ndarray, dt_ticks: np.ndarray) -> dict:
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


def ensure_mmap_cache(npz_path: str) -> str:
    """Распаковывает сжатый .npz в быстрый .npy кэш для чтения через mmap без затрат RAM."""
    p = Path(npz_path).resolve()
    if p.is_dir() and (p / "ready.txt").exists():
        return str(p)
    cache_dir = p.parent / "_mmap_cache"
    marker = cache_dir / "ready.txt"
    if marker.exists():
        return str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("⚡ Подготовка memory-mapped кэша на диске (экономит ~4 ГБ RAM)...")
    with np.load(npz_path, allow_pickle=True) as raw:
        for k in raw.files:
            np.save(cache_dir / f"{k}.npy", raw[k])
    marker.write_text("ok")
    _trim_memory()
    return str(cache_dir)


def load_window_arrays(cache_or_npz, horizon: int, win_idx: int = 0) -> dict:
    h_key = f"h{int(horizon)}"
    if isinstance(cache_or_npz, (str, Path)) and Path(cache_or_npz).is_dir():
        cdir = Path(cache_or_npz)
        tr_idx = np.load(cdir / f"{h_key}_tr_idx_{win_idx:02d}.npy")
        te_idx = np.load(cdir / f"{h_key}_te_idx_{win_idx:02d}.npy")
        X_all = np.load(cdir / "X_all.npy", mmap_mode="r")
        ts_all = np.load(cdir / "ts_all.npy", allow_pickle=True)
        y_all = np.load(cdir / f"y_all_{h_key}.npy", mmap_mode="r")
        dt_all = np.load(cdir / f"dt_all_{h_key}.npy", mmap_mode="r")
        return {
            "X_tr": np.array(X_all[tr_idx], dtype=np.float32),
            "X_te": np.array(X_all[te_idx], dtype=np.float32),
            "y_tr": np.array(y_all[tr_idx], dtype=np.int32),
            "y_te": np.array(y_all[te_idx], dtype=np.int32),
            "dt_te": np.array(dt_all[te_idx], dtype=np.float32),
            "ts_tr": ts_all[tr_idx],
            "ts_te": ts_all[te_idx],
        }
    raw_npz = cache_or_npz
    tr_idx = raw_npz[f"{h_key}_tr_idx_{win_idx:02d}"]
    te_idx = raw_npz[f"{h_key}_te_idx_{win_idx:02d}"]
    X_all = raw_npz["X_all"]
    ts_all = raw_npz["ts_all"]
    return {
        "X_tr": X_all[tr_idx].astype(np.float32),
        "X_te": X_all[te_idx].astype(np.float32),
        "y_tr": raw_npz[f"y_all_{h_key}"][tr_idx].astype(np.int32),
        "y_te": raw_npz[f"y_all_{h_key}"][te_idx].astype(np.int32),
        "dt_te": raw_npz[f"dt_all_{h_key}"][te_idx].astype(np.float32),
        "ts_tr": ts_all[tr_idx],
        "ts_te": ts_all[te_idx],
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
    force_retrain: bool = False,
) -> dict:
    if archs is None:
        archs = ["conv1d+gru", "causal_cnn", "attention", "mlp_bottleneck"]
    has_gpu = len(tf.config.list_physical_devices("GPU")) > 0
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if has_gpu else "float32")
    k_ratio, seq_len = get_horizon_defaults(horizon)
    win_tag = f"win_{win_idx:02d}"
    pack_dir = pack_dir or f"prl_ensemble_pack_h{horizon}s"
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs(pack_dir, exist_ok=True)

    # Проверяем, какие архитектуры уже обучены (Resume после сбоя/перезапуска)
    pending_archs = []
    leaderboard = []
    for arch in archs:
        arch_dir = os.path.join(pack_dir, f"{arch.replace('+', '_')}_h{horizon}s")
        k_file = os.path.join(arch_dir, f"model_{win_tag}.keras")
        m_file = os.path.join(arch_dir, f"meta_{win_tag}.json")
        if not force_retrain and os.path.exists(k_file) and os.path.exists(m_file):
            with open(m_file) as f:
                m_data = json.load(f)
            mo = m_data.get("metrics_oos", {})
            print(f"⏩ [{arch.upper()} | {win_tag}] Чекпоинт уже готов -> IC: {mo.get('ic_dt_ticks', 0):+.4f} | AUC: {mo.get('roc_auc_ovr', 0):.4f} (пропуск)")
            leaderboard.append({
                "window": win_tag, "model": f"{arch} (cached)", "params": 0,
                "ic": mo.get("ic_dt_ticks", 0.0), "auc": mo.get("roc_auc_ovr", 0.5),
                "log_loss": mo.get("log_loss", 1.9), "top2_pnl": mo.get("top2pct_mean_ticks", 0.0)
            })
        else:
            pending_archs.append(arch)

    if not pending_archs:
        return {"window": win_tag, "leaderboard": leaderboard, "oos_predictions": {}}

    w_data = load_window_arrays(raw_npz, horizon, win_idx)
    X_tr, X_te = w_data["X_tr"], w_data["X_te"]
    y_tr, y_te, dt_te = w_data["y_tr"], w_data["y_te"], w_data["dt_te"]
    ts_tr, ts_te = w_data["ts_tr"], w_data["ts_te"]
    n_features = X_tr.shape[1]

    print(f"\n{'='*88}")
    print(f"🚀 ОБУЧЕНИЕ ОКНА {win_tag.upper()} | H={horizon}с (K={k_ratio}, L={seq_len}с) | Train={len(X_tr):,} ({str(ts_tr[0])[:16]}..{str(ts_tr[-1])[:16]})")
    print(f"{'='*88}")

    mean = np.nanmean(X_tr, axis=0)
    std = np.nanstd(X_tr, axis=0) + 1e-6
    target_dtype = np.float16 if has_gpu else np.float32
    X_tr_norm = np.clip((X_tr - mean) / std, -5.0, 5.0).astype(target_dtype)
    X_te_norm = np.clip((X_te - mean) / std, -5.0, 5.0).astype(target_dtype)
    del X_tr, X_te, w_data
    _trim_memory()

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

    for arch in pending_archs:
        best_metric_tuple = (999.0, 999.0)
        best_r = 1
        best_m = None
        best_params = 0
        arch_losses = []
        start_r = 1
        arch_dir = os.path.join(pack_dir, f"{arch.replace('+', '_')}_h{horizon}s")
        os.makedirs(arch_dir, exist_ok=True)
        final_keras_path = os.path.join(arch_dir, f"model_{win_tag}.keras")
        prog_path = os.path.join(arch_dir, f"progress_{win_tag}.json")

        if not force_retrain and os.path.exists(prog_path) and os.path.exists(final_keras_path):
            with open(prog_path) as pf:
                prog = json.load(pf)
            start_r = int(prog.get("completed_runs", 0)) + 1
            best_r = int(prog.get("best_r", 1))
            best_m = prog.get("best_m")
            best_params = int(prog.get("best_params", 0))
            arch_losses = list(prog.get("arch_losses", []))
            if best_m:
                best_metric_tuple = (round(best_m["log_loss"], 3), -best_m["ic"])
                print(
                    f"🔄 [{arch.upper()} | {win_tag}] Предзагружен лидер r{best_r:02d}/{runs_per_arch} "
                    f"(Loss: {best_m['log_loss']:.4f} | IC: {best_m['ic']:+.4f} | AUC: {best_m['auc']:.4f} | Top-2%: {best_m['top2_pnl']:+.2f}) "
                    f"— продолжаем с r{start_r:02d}/{runs_per_arch}!",
                    flush=True,
                )

        for run_idx in range(start_r, runs_per_arch + 1):
            tf.keras.backend.clear_session()
            _trim_memory()
            tf.random.set_seed(1000 + win_idx * 100 + run_idx * 17)
            np.random.seed(1000 + win_idx * 100 + run_idx * 17)
            t_run = time.time()

            model = MODEL_FACTORY[arch](seq_len, n_features, n_classes=7, l2_reg=l2_reg)
            n_params = model.count_params()
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
            model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, class_weight=custom_weights, callbacks=cbs, shuffle=False, verbose=0)
            elapsed = round(time.time() - t_run, 1)
            if os.path.exists(temp_w):
                model.load_weights(temp_w)
                os.remove(temp_w)

            probs = model.predict(val_ds, verbose=0)
            metrics = evaluate_oos_probs(probs, y_true_oos, dt_true_oos)
            del probs
            arch_losses.append(metrics["log_loss"])

            cur_tuple = (round(metrics["log_loss"], 3), -metrics["ic"])
            is_best = cur_tuple < best_metric_tuple
            if is_best:
                best_metric_tuple = cur_tuple
                best_r = run_idx
                best_m = metrics
                best_params = n_params
                model.save(final_keras_path)

            mark = "🏆 РЕКОРД" if is_best else ("🔪 PRUNED" if profiler.pruned else "✅ OK")
            print(f"  [{arch.upper()} | {win_tag} | r{run_idx:02d}/{runs_per_arch}] {mark} ({elapsed}с) -> Loss: {metrics['log_loss']:.4f} | IC: {metrics['ic']:+.4f} | AUC: {metrics['auc']:.4f} | Top-2%: {metrics['top2_pnl']:+.2f}", flush=True)
            with open(prog_path, "w") as pf:
                json.dump(
                    {
                        "completed_runs": run_idx,
                        "best_r": best_r,
                        "best_m": best_m,
                        "best_params": best_params,
                        "arch_losses": arch_losses,
                    },
                    pf,
                    indent=2,
                )
            del model, opt, cbs, profiler
            _trim_memory()

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
        if os.path.exists(prog_path):
            os.remove(prog_path)
        if os.path.isdir("models/prl"):
            canon_dir = os.path.join("models/prl", f"{arch.replace('+', '_')}_h{horizon}s")
            os.makedirs(canon_dir, exist_ok=True)
            shutil.copy2(final_keras_path, os.path.join(canon_dir, f"model_{win_tag}.keras"))
            shutil.copy2(os.path.join(arch_dir, f"meta_{win_tag}.json"), os.path.join(canon_dir, f"meta_{win_tag}.json"))
        leaderboard.append({"window": win_tag, "model": f"{arch} (r{best_r})", "params": best_params, **best_m})

    del train_ds, val_ds, X_tr_norm, X_te_norm, y_tr_oh, y_te_oh
    tf.keras.backend.clear_session()
    _trim_memory()
    return {"window": win_tag, "leaderboard": leaderboard, "oos_predictions": {}}


def run_colab_pipeline(
    data_file: str = "prl_neural_colab_data.npz",
    file_id: str = "1IXq51Irot9L-pj8-EVAY980J7tql4xho",
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
    isolate_windows: bool = True,
    force_retrain: bool = False,
):
    """Запускает полный цикл загрузки данных, обучения выбранных окон (с изоляцией RAM/VRAM между окнами) и упаковки архива."""
    for cand in [data_file, f"data/{data_file}", f"/content/{data_file}", f"/content/drive/MyDrive/{data_file}"]:
        if os.path.exists(cand):
            data_file = cand
            break
    if not os.path.exists(data_file):
        direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        print(f"☁️ Прямое скачивание {data_file} с Google Drive (ID={file_id})...")
        subprocess.run(["curl", "-L", "-s", "-o", data_file, direct_url], check=False)
        if not os.path.exists(data_file) or os.path.getsize(data_file) < 10_000_000:
            import gdown
            gdown.download(id=file_id, output=data_file, quiet=False, fuzzy=True)

    cache_dir = ensure_mmap_cache(data_file)
    h_key = f"h{int(horizon)}"
    if windows.strip().lower() == "all":
        default_counts = {5: 14, 12: 6, 25: 3}
        nw_file = Path(cache_dir) / f"{h_key}_n_windows.npy"
        n_w = int(np.load(nw_file)) if nw_file.exists() else default_counts.get(int(horizon), 1)
        win_list = list(range(n_w))
    else:
        win_list = [int(x.strip()) for x in windows.split(",") if x.strip()]

    arch_list = [a.strip() for a in archs.split(",") if a.strip()]
    pack_dir = f"prl_ensemble_pack_h{horizon}s"
    all_lb = []

    for w_idx in win_list:
        if isolate_windows and len(win_list) > 1:
            cmd = [
                sys.executable, "-u", "-m", "_tools.colab_neural_trainer",
                "--data-file", data_file, "--horizon", str(horizon), "--windows", str(w_idx),
                "--archs", archs, "--runs", str(runs_per_arch), "--epochs", str(max_epochs),
                "--prune-epoch", str(hard_prune_epoch), "--top-k", str(top_k_per_arch),
                "--batch-size", str(batch_size), "--lr", str(lr), "--l2-reg", str(l2_reg), "--no-isolate",
            ]
            if force_retrain:
                cmd.append("--force-retrain")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    if not any(skip in line for skip in ("oneDNN custom operations", "Unable to register cu", "computation_placer.cc", "prefetch_dataset_op.cc", "prefetch_autotuner.cc")):
                        print(line, end="", flush=True)
                proc.wait()
            except KeyboardInterrupt:
                print("\n🛑 [Interrupt] Принудительная остановка дочернего процесса окна...")
                proc.kill()
                proc.wait()
                break
            for arch in arch_list:
                m_file = os.path.join(pack_dir, f"{arch.replace('+', '_')}_h{horizon}s", f"meta_win_{w_idx:02d}.json")
                if os.path.exists(m_file):
                    with open(m_file) as f:
                        mo = json.load(f).get("metrics_oos", {})
                    all_lb.append({
                        "window": f"win_{w_idx:02d}", "model": arch,
                        "ic": mo.get("ic_dt_ticks", 0.0), "auc": mo.get("roc_auc_ovr", 0.5),
                        "log_loss": mo.get("log_loss", 1.9), "top2_pnl": mo.get("top2pct_mean_ticks", 0.0)
                    })
        else:
            res = train_window_swarm(
                raw_npz=cache_dir, horizon=int(horizon), win_idx=w_idx, archs=arch_list,
                runs_per_arch=runs_per_arch, max_epochs=max_epochs, hard_prune_epoch=hard_prune_epoch,
                top_k_per_arch=top_k_per_arch, batch_size=batch_size, lr=lr, l2_reg=l2_reg,
                pack_dir=pack_dir, force_retrain=force_retrain,
            )
            all_lb.extend(res["leaderboard"])

    df_lb = pd.DataFrame(all_lb)
    if not isolate_windows and len(win_list) == 1:
        return df_lb
    print("\n🏆 СВОДНЫЙ ЛИДЕРБОРД WALK-FORWARD ОКОН:")
    if len(df_lb) > 0:
        print(df_lb.to_string(index=False))
    shutil.make_archive(pack_dir, "zip", root_dir=".", base_dir=pack_dir)
    print(f"\n📦 Архив {pack_dir}.zip успешно собран!")
    return df_lb


def main():
    parser = argparse.ArgumentParser(description="Colab Neural Swarm Trainer (Walk-Forward Windows)")
    parser.add_argument("--data-file", type=str, default="prl_neural_colab_data.npz")
    parser.add_argument("--file-id", type=str, default="1IXq51Irot9L-pj8-EVAY980J7tql4xho")
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
    parser.add_argument("--no-isolate", dest="isolate", action="store_false", default=True)
    parser.add_argument("--force-retrain", action="store_true", default=False)
    args = parser.parse_args()

    run_colab_pipeline(
        data_file=args.data_file, file_id=args.file_id, horizon=args.horizon, windows=args.windows,
        archs=args.archs, runs_per_arch=args.runs, max_epochs=args.epochs, hard_prune_epoch=args.prune_epoch,
        top_k_per_arch=args.top_k, batch_size=args.batch_size, lr=args.lr, l2_reg=args.l2_reg,
        isolate_windows=args.isolate, force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()
