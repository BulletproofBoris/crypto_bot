#!/usr/bin/env python3
"""
_tools/train_bo_full34d_micro_swarms.py
=============================================================================
Полный 34-дневный проход обучения малых окон `H = 1s (K=3.0)` и `H = 2s (K=3.0)`
для площадки `BigONE (prl_bo -> data/_mmap_cache_bo -> models/prl_bo/)`, где родной
шаг цены `$0.0001` действовал на протяжении всей 34-суточной истории (`820.8 ч`).

Архитектура максимальной пропускной способности:
  1. Фаза A (CPU): Векторная разметка всех 3.06M секунд BigONE в кванте `eff_tick = $0.0005`
     и бесшовная нарезка чистых каузальных окон (`~386` окон для `H=1s`, `~192` окна для `H=2s`).
  2. Фаза B (CPU Parallel LightGBM): Параллельное обучение всех `~578` окон LightGBM
     в пуле из `4` воркеров × `4` треда (`16` ядер CPU) с векторизованными паспортами режимов.
  3. Фаза C (Persistent Multi-Worker GPU Pool): Калибровка числа персистентных GPU-процессов
     по доступной видеопамяти `RTX 5060 Ti` (`4–5` параллельных процессов × мультистарты в CUDA Streams,
     однократная инициализация `CUDA` и `mmap` на воркер без накладных расходов).
=============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MICRO_PAIRS = [(1, 3.0), (2, 3.0)]
MICRO_EFF_TICK = 0.0005
NEURAL_ARCHS = ["mlp_bottleneck", "wavenet_tcn", "cross_mixer", "causal_cnn", "attention", "conv1d+gru"]


def _numeric_win_sort_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def _build_full_bo_micro_windows(df: pd.DataFrame, H: int, K: float, n_target: int = 5000) -> list[dict]:
    """Строит полную хронологическую сетку окон H in {1, 2}s по всей 34-дневной чистой истории BigONE."""
    from _tools.train_rolling_swarm import find_clean_segments

    L = int(round(H * K))
    t_train = int(n_target * L)
    step = t_train // 2
    buffer = H + 180
    min_test_sec = max(3600, step // 2)
    seam_purge = 360

    segs = sorted(find_clean_segments(df), key=lambda x: x[0])
    bo_mid = np.nan_to_num(df["BO_mid_price"].values.astype(np.float64), nan=0.0)
    bo_stale = np.nan_to_num(df["BO_sec_since_tick"].values.astype(np.float64), nan=999.0)

    clean_blocks = []
    for s_idx, (s, e) in enumerate(segs):
        s_clean = s + (seam_purge if s_idx > 0 else 0)
        e_clean = e - (seam_purge if s_idx < len(segs) - 1 else 0)
        if e_clean - s_clean > 1000:
            blk = np.arange(s_clean, e_clean, dtype=np.int32)
            valid = (bo_mid[blk] > 0.01) & (bo_stale[blk] <= 60.0)
            if np.sum(valid) > 1000:
                clean_blocks.append(blk[valid])

    clean_idx = np.concatenate(clean_blocks)
    n_clean = len(clean_idx)
    windows: list[dict] = []

    tr_s_pos = 0
    while tr_s_pos + t_train + buffer + min_test_sec <= n_clean:
        tr_e_pos = tr_s_pos + t_train
        te_s_pos = tr_e_pos + buffer
        next_tr_s = tr_s_pos + step
        if next_tr_s + t_train + buffer + min_test_sec > n_clean:
            te_e_pos = n_clean
        else:
            te_e_pos = min(n_clean, te_s_pos + step)

        windows.append({
            "segment_id": len(windows),
            "tr_idx": clean_idx[tr_s_pos:tr_e_pos].astype(np.int32),
            "te_idx": clean_idx[te_s_pos:te_e_pos].astype(np.int32),
            "H": int(H),
            "K": float(K),
            "L": int(L),
        })
        tr_s_pos += step

    return windows


def phase_a_prepare_bo_full34d_cache():
    """Фаза A (CPU): разметка полной 34-дневной истории BigONE (H=1s, 2s) и обновление data/_mmap_cache_bo."""
    from _tools.prepare_dual_venue_first_order_caches import compute_bigone_single_horizon_labels

    t0 = time.time()
    print("=" * 108, flush=True)
    print("📐 [ФАЗА A: CPU] РАЗМЕТКА 34-ДНЕВНОЙ ИСТОРИИ BIGONE H=1s, 2s ($0.0001 РОДНОЙ ТИК, eff_tick=$0.0005)", flush=True)
    print("=" * 108, flush=True)

    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    cache_bo = BASE_DIR / "data" / "_mmap_cache_bo"
    ts_all = np.load(cache_bo / "ts_all.npy", allow_pickle=True)

    need_cols = [
        "sample_split", "is_data_gap", "ST_mid_price", "ST_trade_vol",
        "BO_mid_price", "BO_spread", "BO_bid_1_price", "BO_ask_1_price",
        "BO_bid_1_qty", "BO_ask_1_qty", "BO_bid_2_qty", "BO_ask_2_qty",
        "BO_bids_vol_top5", "BO_asks_vol_top5", "BO_trade_vol", "BO_sec_since_tick",
    ]
    schema_cols = set(pq.read_schema(labeled_path).names)
    df = pq.read_table(labeled_path, columns=[c for c in need_cols if c in schema_cols]).to_pandas()

    # Очистим старые 6-оконные артефакты prl_bo h1/h2 перед записью полного 34-дневного пула
    for h, k in MICRO_PAIRS:
        for old_idx in cache_bo.glob(f"h{h}_*_idx_*.npy"):
            old_idx.unlink(missing_ok=True)
        lgb_dir = BASE_DIR / "models" / "prl_bo" / f"h{h}s_k{k:g}"
        if lgb_dir.exists():
            shutil.rmtree(lgb_dir)
        for arch in NEURAL_ARCHS:
            a_dir = BASE_DIR / "models" / "prl_bo" / f"{arch.replace('+', '_')}_h{h}s"
            if a_dir.exists():
                shutil.rmtree(a_dir)

    for h, k in MICRO_PAIRS:
        lbl = compute_bigone_single_horizon_labels(df, h, k, eff_tick=MICRO_EFF_TICK, min_vol=50.0)
        y_bo = (lbl["class_7"] + 3).astype(np.int8)
        dt_bo = lbl["dt_ticks"].astype(np.float16)
        wins = _build_full_bo_micro_windows(df, h, k)

        np.save(cache_bo / f"y_all_h{h}.npy", y_bo)
        np.save(cache_bo / f"dt_all_h{h}.npy", dt_bo)
        np.save(cache_bo / f"h{h}_n_windows.npy", np.int32(len(wins)))
        for w_i, w_obj in enumerate(wins):
            np.save(cache_bo / f"h{h}_tr_idx_{w_i:02d}.npy", w_obj["tr_idx"])
            np.save(cache_bo / f"h{h}_te_idx_{w_i:02d}.npy", w_obj["te_idx"])

        nz_pct = float(np.mean(lbl["class_7"] != 0) * 100.0)
        print(
            f"  • PRL_BO H={h}s K={k:g}: сформировано {len(wins)} окон за 34 суток "
            f"[{str(ts_all[wins[0]['tr_idx'][0]])[:19]} -> {str(ts_all[wins[-1]['te_idx'][-1]])[:19]}] "
            f"| импульсных меток: {nz_pct:.2f}%",
            flush=True,
        )

    del df
    gc.collect()
    print(f"✅ [ФАЗА A] 34-дневный mmap-кэш BigONE H=1s, 2s собран за {time.time() - t0:.1f}с!", flush=True)


def phase_b_train_bo_lightgbm_parallel(total_cpu_threads: int = 16):
    """Фаза B (CPU Parallel): быстрое обучение всех ~578 окон LightGBM (4 параллельных окна × 4 потока)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight

    t0 = time.time()
    print("\n" + "=" * 108, flush=True)
    print(f"🌲 [ФАЗА B: CPU PARALLEL] ОБУЧЕНИЕ LIGHTGBM ПО ВСЕЙ 34-ДНЕВНОЙ ИСТОРИИ BIGONE ({total_cpu_threads} ПОТОКОВ)", flush=True)
    print("=" * 108, flush=True)

    cache_bo = BASE_DIR / "data" / "_mmap_cache_bo"
    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    feats_json = BASE_DIR / "data" / "features" / "selected_features_prl_bo_h8s.json"
    feature_cols = json.loads(feats_json.read_text())["feature_order"]
    X_all = np.load(cache_bo / "X_all.npy", mmap_mode="r")
    ts_all = np.load(cache_bo / "ts_all.npy", allow_pickle=True)

    fp_cols = [
        "ST_vol_300s_ticks", "ST_vol_60s_ticks", "ST_spread", "ST_spread_pct",
        "ST_ofi_norm_300s", "ST_cvd_5m", "ST_cvd_norm_20s",
        "BO_spread", "BO_mid_price",
    ]
    schema_cols = set(pq.read_schema(labeled_path).names)
    fp_cols = [c for c in fp_cols if c in schema_cols]
    fp_df = pq.read_table(labeled_path, columns=fp_cols).to_pandas()
    fp_arrays = {c: fp_df[c].values.astype(np.float32) for c in fp_cols}
    del fp_df

    def _fast_fp(idx_arr: np.ndarray, y_zero_based: np.ndarray) -> dict:
        fp = {}
        for c_name, arr in fp_arrays.items():
            sub = arr[idx_arr]
            fp[f"median_{c_name}"] = float(np.nanmedian(sub))
            fp[f"mean_{c_name}"] = float(np.nanmean(sub))
            fp[f"std_{c_name}"] = float(np.nanstd(sub))
        bc = np.bincount(y_zero_based, minlength=7).astype(np.float64) / max(1, len(y_zero_based))
        for cls_i in range(7):
            fp[f"pct_class_{cls_i - 3:+d}"] = float(bc[cls_i] * 100.0)
        return fp

    win_workers = 4
    lgb_threads = max(2, total_cpu_threads // win_workers)
    classes_values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)

    for h, k in MICRO_PAIRS:
        th0 = time.time()
        n_w = int(np.load(cache_bo / f"h{h}_n_windows.npy"))
        y_all = np.load(cache_bo / f"y_all_h{h}.npy", mmap_mode="r")
        dt_all = np.load(cache_bo / f"dt_all_h{h}.npy", mmap_mode="r")
        out_dir = BASE_DIR / "models" / "prl_bo" / f"h{h}s_k{k:g}"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _fit_win(w_i: int) -> tuple[int, dict]:
            win_id = f"win_{w_i:02d}"
            meta_p = out_dir / f"meta_{win_id}.json"
            model_p = out_dir / f"model_{win_id}.txt"
            if meta_p.exists() and model_p.exists():
                return w_i, json.loads(meta_p.read_text())

            tr_idx = np.load(cache_bo / f"h{h}_tr_idx_{w_i:02d}.npy")
            te_idx = np.load(cache_bo / f"h{h}_te_idx_{w_i:02d}.npy")
            X_tr = np.asarray(X_all[tr_idx], dtype=np.float32)
            y_tr = np.asarray(y_all[tr_idx], dtype=np.int32)
            X_te = np.asarray(X_all[te_idx], dtype=np.float32)
            y_te = np.asarray(y_all[te_idx], dtype=np.int32)
            dt_te = np.asarray(dt_all[te_idx], dtype=np.float32)

            pres_tr = np.unique(y_tr)
            base_w = compute_class_weight("balanced", classes=pres_tr, y=y_tr)
            cw = {int(c): float(v) for c, v in zip(pres_tr, base_w)}
            for c_b, mult in ((0, 1.35), (1, 1.25), (2, 1.10)):
                if c_b in cw:
                    cw[c_b] *= mult

            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=7,
                class_weight=cw,
                n_estimators=450,
                learning_rate=0.07,
                num_leaves=31,
                max_depth=6,
                min_child_samples=40,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=2042 + w_i,
                n_jobs=lgb_threads,
                max_bin=255,
                verbose=-1,
            )
            clf.fit(X_tr, y_tr, feature_name=feature_cols)
            probs = clf.booster_.predict(X_te)
            exp_z = np.dot(probs, classes_values)
            ic = float(np.corrcoef(exp_z, dt_te)[0, 1]) if np.std(exp_z) > 1e-6 and np.std(dt_te) > 1e-6 else 0.0
            try:
                pres_te = [c for c in range(7) if np.sum(y_te == c) > 0]
                if len(pres_te) > 1:
                    ps = probs[:, pres_te]
                    ps = ps / np.maximum(ps.sum(axis=1, keepdims=True), 1e-7)
                    auc = float(roc_auc_score(y_te, ps, multi_class="ovr", labels=pres_te))
                else:
                    auc = 0.5
            except Exception:
                auc = 0.5
            q98 = float(np.quantile(exp_z, 0.98))
            top2 = float(np.mean(dt_te[exp_z >= q98])) if np.any(exp_z >= q98) else 0.0

            clf.booster_.save_model(str(model_p))
            meta = {
                "window_id": win_id,
                "coin": "PRL_BO",
                "H": h,
                "K": k,
                "L": h * k,
                "eff_tick": MICRO_EFF_TICK,
                "train_ts_range": [str(ts_all[tr_idx[0]])[:19], str(ts_all[tr_idx[-1]])[:19]],
                "test_ts_range": [str(ts_all[te_idx[0]])[:19], str(ts_all[te_idx[-1]])[:19]],
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "metrics_oos": {
                    "ic_dt_ticks": round(ic, 4),
                    "roc_auc_ovr": round(auc, 4),
                    "top2pct_mean_ticks": round(top2, 2),
                },
                "regime_fingerprint_train": _fast_fp(tr_idx, y_tr),
                "regime_fingerprint_test": _fast_fp(te_idx, y_te),
                "model_file": model_p.name,
            }
            meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return w_i, meta

        slots = [None] * n_w
        done = 0
        with ThreadPoolExecutor(max_workers=win_workers) as ex:
            futs = {ex.submit(_fit_win, i): i for i in range(n_w)}
            for fut in as_completed(futs):
                w_i, m_obj = fut.result()
                slots[w_i] = m_obj
                done += 1
                if done % 25 == 0 or done == n_w:
                    ics = [s["metrics_oos"]["ic_dt_ticks"] for s in slots if s is not None]
                    aucs = [s["metrics_oos"]["roc_auc_ovr"] for s in slots if s is not None]
                    print(
                        f"  🌲 [PRL_BO LGB H={h}s] {done:3d}/{n_w} окон ({time.time()-th0:5.1f}s) "
                        f"| Mean IC={np.mean(ics):+.4f} | Mean AUC={np.mean(aucs):.4f}",
                        flush=True,
                    )
        (out_dir / "manifest.json").write_text(json.dumps(slots, indent=2), encoding="utf-8")

    gc.collect()
    print(f"✅ [ФАЗА B] Все 34-дневные пулы LightGBM BigONE обучены за {(time.time() - t0)/60.0:.2f} мин!", flush=True)


def _persistent_gpu_worker_init():
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)


def _persistent_gpu_worker_step(task: dict) -> dict:
    import torch
    from _tools.fast_neural_engine import train_pytorch_window_fast

    t0 = time.time()
    if torch.cuda.is_initialized():
        try:
            torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            pass
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = train_pytorch_window_fast(
            cache_dir=task["cache_dir"],
            horizon=task["horizon"],
            win_idx=task["win_idx"],
            arch=task["arch"],
            runs=task["runs"],
            epochs=task["epochs"],
            batch_size=task["batch_size"],
            lr=2.2e-3,
            force=False,
            coin="prl_bo",
        )
    peak_res_mib = float(torch.cuda.max_memory_reserved() / (1024 * 1024))
    free_mib, total_mib = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
    return {
        **task,
        "elapsed": round(time.time() - t0, 2),
        "peak_reserved_mib": round(peak_res_mib, 1),
        "free_mib_after": round(free_mib, 1),
        "total_mib": round(total_mib, 1),
        "result": res,
    }


def phase_c_train_bo_full34d_gpu(runs: int = 8, epochs: int = 10, max_workers_cap: int = 6):
    """
    Фаза C: Параллельное обучение всей 34-дневной истории BigONE (H=1s, 2s)
    на персистентном пуле GPU-воркеров с раздельной калибровкой VRAM для H=1s и H=2s!
    (H=1s потребляет вдвое меньше VRAM на окно, чем H=2s, что позволяет поднять больше параллельных воркеров!).
    """
    t0 = time.time()
    cache_bo = BASE_DIR / "data" / "_mmap_cache_bo"
    ctx = mp.get_context("spawn")

    print("\n" + "=" * 108, flush=True)
    print(f"🚀 [ФАЗА C: PERSISTENT PARALLEL GPU] ПОЛНЫЙ 34-ДНЕВНЫЙ ПРОХОД BIGONE H=1s, 2s (runs={runs})", flush=True)
    print("=" * 108, flush=True)

    for h, _ in MICRO_PAIRS:
        th0 = time.time()
        n_w = int(np.load(cache_bo / f"h{h}_n_windows.npy"))
        tasks_h: list[dict] = []
        import zipfile
        for arch in NEURAL_ARCHS:
            a_dir = BASE_DIR / "models" / "prl_bo" / f"{arch.replace('+', '_')}_h{h}s"
            ep = 8 if arch == "conv1d+gru" else epochs
            for w_i in range(n_w):
                kp = a_dir / f"model_win_{w_i:02d}.keras"
                mp_f = a_dir / f"meta_win_{w_i:02d}.json"
                valid_pair = False
                if kp.exists() and mp_f.exists() and zipfile.is_zipfile(kp):
                    try:
                        _m_chk = json.loads(mp_f.read_text())
                        valid_pair = bool(isinstance(_m_chk, dict) and _m_chk.get("metrics_oos"))
                    except Exception:
                        valid_pair = False
                if valid_pair:
                    continue
                if kp.exists():
                    kp.unlink(missing_ok=True)
                if mp_f.exists():
                    mp_f.unlink(missing_ok=True)
                tasks_h.append({
                    "cache_dir": str(cache_bo),
                    "horizon": h,
                    "win_idx": w_i,
                    "arch": arch,
                    "runs": runs,
                    "epochs": ep,
                    "batch_size": 2048,
                })

        if not tasks_h:
            print(f"⏩ [PRL_BO H={h}s] Все {n_w * 6} моделей уже обучены, проверяем манифесты...", flush=True)
        else:
            # Пробный замер VRAM на тяжелой архитектуре attention данного горизонта H
            probe_idx = next((i for i, t in enumerate(tasks_h) if t["arch"] == "attention"), 0)
            probe_task = tasks_h.pop(probe_idx)
            with ctx.Pool(processes=1, initializer=_persistent_gpu_worker_init) as p1:
                p_out = p1.apply(_persistent_gpu_worker_step, (probe_task,))

            free_before = p_out["free_mib_after"] + p_out["peak_reserved_mib"] + 420.0
            worker_budget = max(800.0, p_out["peak_reserved_mib"] + 620.0)
            n_workers = int(np.clip(int((free_before - 1400.0) // worker_budget), 2, max_workers_cap))
            print(
                f"🔍 [Калибровка H={h}s, задач={len(tasks_h)+1}] Всего VRAM={p_out['total_mib']:.0f}M | "
                f"Бюджет 1 воркера={worker_budget:.0f}M -> Запускаем {n_workers} ПЕРСИСТЕНТНЫХ GPU-ВОРКЕРОВ параллельно!",
                flush=True,
            )

            done = 1
            total_h = len(tasks_h) + 1
            recent_ics = [float((p_out["result"] or {}).get("ic_dt_ticks", 0.0))]
            if tasks_h:
                with ctx.Pool(processes=n_workers, initializer=_persistent_gpu_worker_init) as pool:
                    for out in pool.imap_unordered(_persistent_gpu_worker_step, tasks_h, chunksize=1):
                        done += 1
                        mo = out["result"] or {}
                        recent_ics.append(float(mo.get("ic_dt_ticks", 0.0)))
                        if done % 30 == 0 or done == total_h:
                            elapsed_h = time.time() - th0
                            speed = done / max(1e-3, elapsed_h)
                            eta_min = (total_h - done) / max(1e-3, speed) / 60.0
                            print(
                                f"  ⚡ [PRL_BO H={h}s] {done:4d}/{total_h:4d} ({done*100.0/total_h:5.1f}%) | "
                                f"скорость={speed:.2f} окон/с | ETA={eta_min:.1f} мин | freeVRAM={out['free_mib_after']:.0f}M | "
                                f"Mean IC={np.mean(recent_ics):+.4f}",
                                flush=True,
                            )

        # Сборка manifest.json и альянсов строго по численному порядку окон 0..n_w-1
        selected_models = []
        weights = {}
        for arch in NEURAL_ARCHS:
            a_dir = BASE_DIR / "models" / "prl_bo" / f"{arch.replace('+', '_')}_h{h}s"
            manifest = []
            for w_i in range(n_w):
                mp_f = a_dir / f"meta_win_{w_i:02d}.json"
                if mp_f.exists() and (a_dir / f"model_win_{w_i:02d}.keras").exists():
                    m_obj = json.loads(mp_f.read_text())
                    manifest.append(m_obj)
                    mo_dict = m_obj.get("metrics_oos") or {}
                    ic = float(mo_dict.get("ic_dt_ticks", 0.0))
                    if ic > 0.02:
                        key = f"{arch}_win_{w_i:02d}"
                        selected_models.append(key)
                        weights[key] = round(max(0.001, ic ** 2), 6)
            (a_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        w_sum = sum(weights.values()) + 1e-12
        weights = {k_m: round(v_m / w_sum, 6) for k_m, v_m in weights.items()}
        payload = {
            "coin": "PRL_BO",
            "horizon_sec": h,
            "eff_tick": MICRO_EFF_TICK,
            "n_windows": n_w,
            "selected_models": selected_models,
            "weights": weights,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for fn in (f"ensemble_alliance_h{h}s.json", f"ensemble_alliance_live_h{h}s.json"):
            (BASE_DIR / "models" / "prl_bo" / fn).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  🤝 [PRL_BO H={h}s] Альянс собран: {len(selected_models)} активных экспертов из {n_w * 6}!", flush=True)

    print(f"\n🎉 Полный 34-дневный цикл GPU-обучения BigONE (H=1s, 2s) завершен за {(time.time() - t0)/60.0:.1f} мин!", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "cpu", "gpu"], default="all")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-gpu-workers", type=int, default=6)
    ap.add_argument("--cpu-threads", type=int, default=14)
    args = ap.parse_args()

    if args.phase in ("all", "cpu"):
        phase_a_prepare_bo_full34d_cache()
        phase_b_train_bo_lightgbm_parallel(total_cpu_threads=args.cpu_threads)

    if args.phase in ("all", "gpu"):
        phase_c_train_bo_full34d_gpu(runs=args.runs, epochs=args.epochs, max_workers_cap=args.max_gpu_workers)
        from _tools.train_micro_1s_2s_swarms import print_micro_summary
        print_micro_summary()


if __name__ == "__main__":
    main()
