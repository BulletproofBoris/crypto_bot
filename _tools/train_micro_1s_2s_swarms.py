#!/usr/bin/env python3
"""
_tools/train_micro_1s_2s_swarms.py
=============================================================================
Обучение легковесных микро-окон 1-го порядка `H = 1s (K=3.0)` и `H = 2s (K=3.0)`
для обеих площадок — `SafeTrade (prl)` и `BigONE (prl_bo)` — строго с момента
перехода на единый шаг цены `$0.0001` (`2026-11-13 21:56:26 UTC` / `2026-11-14 00:56:26+03:00`).

Ключевые особенности:
  1. Экономический квант разметки `eff_tick = $0.0005` (5 микро-тиков `$0.0001`),
     обеспечивающий симметричную наполненность всех 7 классов `[-3..+3]`.
  2. Атомарное дообогащение `data/_mmap_cache` и `data/_mmap_cache_bo` ключами `h1_*` и `h2_*`
     без перезаписи существующих горизонтов `H = 5..120s`.
  3. Обучение LightGBM (`450` деревьев) в чистом CPU-контуре до инициализации CUDA.
  4. Параллельное мультипроцессное GPU-обучение всех 6 нейросетевых архитектур
     (`mlp_bottleneck`, `wavenet_tcn`, `cross_mixer`, `causal_cnn`, `attention`, `conv1d+gru`,
     `sl = 15`, шаблон от `h5s`) с автоматическим расчетом числа одновременных воркеров
     по реально измеренной свободной видеопамяти (VRAM).
=============================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _find_subcent_era_start_idx(df: pd.DataFrame, ts_arr: np.ndarray) -> int:
    """Находит точный индекс старта эпохи шага `$0.0001` на SafeTrade."""
    st_sp = np.nan_to_num(df["ST_spread"].values.astype(np.float64), nan=0.0)
    sub_idx = np.where((st_sp > 0.0) & (np.round(st_sp * 10000.0) % 100 != 0))[0]
    if len(sub_idx) > 0:
        return int(sub_idx[0])
    # Fallback по известной метке времени UTC 2026-11-13 21:56:25 -> MSK 2026-11-14 00:56:25+03:00
    ts_pd = pd.to_datetime(pd.Series(ts_arr), utc=True)
    cand = np.where(ts_pd >= pd.Timestamp("2026-11-13 21:56:25+00:00"))[0]
    if len(cand) == 0:
        raise RuntimeError("В датасете не найден срез эпохи $0.0001!")
    return int(cand[0])


def _build_micro_windows(clean_idx: np.ndarray, H: int, K: float, n_target: int = 5000) -> list[dict]:
    """
    Генератор каузальных скользящих окон для быстрых горизонтов H in {1, 2}s:
      - t_train = 5000 * L (15,000s = 4.17h для H=1s; 30,000s = 8.33h для H=2s)
      - buffer  = H + 180s (3 мин embargo >= 30*L, полностью исключает утечку меток и признаков)
      - step    = t_train // 2
      - min_te  = max(3600, step // 3) (минимум 1..1.5 часа чистого OOS-теста = 600..1800 независимых блоков L)
    При наличии свежего хвоста >= 4500s после последнего окна добавляет актуальное фронт-окно `win_last`.
    """
    L = int(round(H * K))
    t_train = int(n_target * L)
    step = t_train // 2
    buffer = H + 180
    min_test_sec = max(3600, step // 3)
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
            "tr_start": int(clean_idx[tr_s_pos]),
            "tr_end": int(clean_idx[tr_e_pos - 1]) + 1,
            "te_start": int(clean_idx[te_s_pos]),
            "te_end": int(clean_idx[te_e_pos - 1]) + 1,
            "t_train": int(t_train),
            "t_test": int(te_e_pos - te_s_pos),
            "H": int(H),
            "K": float(K),
            "L": int(L),
        })
        tr_s_pos += step

    # Для H=2s при 11.65ч истории базовый шаг дает 1 окно с 11,760с теста; добавим сдвинутое фронт-окно win_01
    if windows:
        last_tr_e_pos = (len(windows) - 1) * step + t_train
        front_te_len = max(4500, step // 3)
        cand_tr_e = n_clean - buffer - front_te_len
        min_shift = max(4500, step // 3)
        if cand_tr_e - last_tr_e_pos >= min_shift and cand_tr_e >= t_train:
            cand_tr_s = cand_tr_e - t_train
            cand_te_s = cand_tr_e + buffer
            # Сузим OOS предыдущего окна до начала его естественного шага, чтобы каждое окно тестировалось честно
            prev_w = windows[-1]
            prev_te_end_pos = min(n_clean, last_tr_e_pos + buffer + step)
            prev_te_s_pos = last_tr_e_pos + buffer
            prev_w["te_idx"] = clean_idx[prev_te_s_pos:prev_te_end_pos].astype(np.int32)
            prev_w["te_end"] = int(clean_idx[prev_te_end_pos - 1]) + 1
            prev_w["t_test"] = int(prev_te_end_pos - prev_te_s_pos)

            windows.append({
                "segment_id": len(windows),
                "tr_idx": clean_idx[cand_tr_s:cand_tr_e].astype(np.int32),
                "te_idx": clean_idx[cand_te_s:n_clean].astype(np.int32),
                "tr_start": int(clean_idx[cand_tr_s]),
                "tr_end": int(clean_idx[cand_tr_e - 1]) + 1,
                "te_start": int(clean_idx[cand_te_s]),
                "te_end": int(clean_idx[n_clean - 1]) + 1,
                "t_train": int(t_train),
                "t_test": int(n_clean - cand_te_s),
                "H": int(H),
                "K": float(K),
                "L": int(L),
            })

    return windows


def phase_a_prepare_micro_caches() -> dict:
    """Фаза A (100% CPU): разметка H=1s, 2s в масштабе $0.0005 и запись h1_*, h2_* в оба mmap-кэша."""
    from _tools.data_labeler import compute_single_horizon_labels
    from _tools.prepare_dual_venue_first_order_caches import compute_bigone_single_horizon_labels
    from _tools.train_rolling_swarm import compute_regime_fingerprint

    t0 = time.time()
    print("=" * 105, flush=True)
    print("📐 [ФАЗА A: CPU] РАЗМЕТКА МИКРО-ГОРИЗОНТОВ H=1s, 2s ($0.0001 ЭРА, eff_tick=$0.0005) И СБОРКА MMAP-ИНДЕКСОВ", flush=True)
    print("=" * 105, flush=True)

    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    cache_st = BASE_DIR / "data" / "_mmap_cache"
    cache_bo = BASE_DIR / "data" / "_mmap_cache_bo"
    ts_all = np.load(cache_st / "ts_all.npy", allow_pickle=True)
    n_total = len(ts_all)

    need_cols = [
        "sample_split", "is_data_gap",
        "ST_mid_price", "ST_spread", "ST_spread_pct",
        "ST_bid_1_price", "ST_ask_1_price", "ST_bid_1_qty", "ST_ask_1_qty",
        "ST_bid_2_qty", "ST_ask_2_qty", "ST_bids_vol_top5", "ST_asks_vol_top5", "ST_trade_vol",
        "ST_vol_300s_ticks", "ST_vol_60s_ticks", "ST_ofi_norm_300s", "ST_cvd_5m", "ST_cvd_norm_20s",
        "BO_mid_price", "BO_spread", "BO_bid_1_price", "BO_ask_1_price",
        "BO_bid_1_qty", "BO_ask_1_qty", "BO_bid_2_qty", "BO_ask_2_qty",
        "BO_bids_vol_top5", "BO_asks_vol_top5", "BO_trade_vol", "BO_sec_since_tick",
    ]
    schema_cols = set(pq.read_schema(labeled_path).names)
    load_cols = [c for c in need_cols if c in schema_cols]
    df = pq.read_table(labeled_path, columns=load_cols).to_pandas()
    if len(df) != n_total:
        raise RuntimeError(f"Рассинхрон длины prl_labeled.parquet ({len(df)}) и ts_all.npy ({n_total})!")

    era_start = _find_subcent_era_start_idx(df, ts_all)
    df_era = df.iloc[era_start:].reset_index(drop=True)
    n_era = len(df_era)
    print(
        f"✅ Эра шага $0.0001: старт idx={era_start:,} ({ts_all[era_start]}) -> конец idx={n_total - 1:,} ({ts_all[-1]}) "
        f"| длина = {n_era:,} с ({n_era / 3600.0:.2f} ч)",
        flush=True,
    )

    valid_st_rel = np.where((df_era["is_data_gap"].fillna(0).values == 0) & (df_era["ST_mid_price"].fillna(0).values > 0.01))[0]
    valid_bo_rel = np.where(
        (df_era["is_data_gap"].fillna(0).values == 0)
        & (df_era["BO_mid_price"].fillna(0).values > 0.01)
        & (df_era["BO_sec_since_tick"].fillna(999).values <= 60.0)
    )[0]
    clean_idx_st = (valid_st_rel + era_start).astype(np.int32)
    clean_idx_bo = (valid_bo_rel + era_start).astype(np.int32)

    summary_meta: dict = {"prl": {}, "prl_bo": {}}

    for h, k in MICRO_PAIRS:
        # 1. SafeTrade (prl) — точный непрерывный dt_ticks в единицах $0.0005
        st_lbl = compute_single_horizon_labels(df_era, "prl", h, k, 50.0, MICRO_EFF_TICK)
        mid_st = df_era["ST_mid_price"].values.astype(np.float64)
        dt_st_era = np.zeros(n_era, dtype=np.float32)
        dt_st_era[:-h] = ((mid_st[h:] - mid_st[:-h]) / MICRO_EFF_TICK).astype(np.float32)
        dt_st_era = np.where(st_lbl["is_void"], 0.0, dt_st_era)

        y_st_full = np.full(n_total, 3, dtype=np.int8)
        dt_st_full = np.zeros(n_total, dtype=np.float16)
        y_st_full[era_start:] = (st_lbl["class_7"] + 3).astype(np.int8)
        dt_st_full[era_start:] = dt_st_era.astype(np.float16)

        # 2. BigONE (prl_bo)
        bo_lbl = compute_bigone_single_horizon_labels(df_era, h, k, eff_tick=MICRO_EFF_TICK, min_vol=50.0)
        y_bo_full = np.full(n_total, 3, dtype=np.int8)
        dt_bo_full = np.zeros(n_total, dtype=np.float16)
        y_bo_full[era_start:] = (bo_lbl["class_7"] + 3).astype(np.int8)
        dt_bo_full[era_start:] = bo_lbl["dt_ticks"].astype(np.float16)

        wins_st = _build_micro_windows(clean_idx_st, h, k)
        wins_bo = _build_micro_windows(clean_idx_bo, h, k)

        for cache_dir, y_arr, dt_arr, wins, coin_key in (
            (cache_st, y_st_full, dt_st_full, wins_st, "prl"),
            (cache_bo, y_bo_full, dt_bo_full, wins_bo, "prl_bo"),
        ):
            for old_p in cache_dir.glob(f"h{h}_*_idx_*.npy"):
                old_p.unlink(missing_ok=True)
            np.save(cache_dir / f"y_all_h{h}.npy", y_arr)
            np.save(cache_dir / f"dt_all_h{h}.npy", dt_arr)
            np.save(cache_dir / f"h{h}_n_windows.npy", np.int32(len(wins)))
            fps_per_win = []
            for w_i, w_obj in enumerate(wins):
                np.save(cache_dir / f"h{h}_tr_idx_{w_i:02d}.npy", w_obj["tr_idx"])
                np.save(cache_dir / f"h{h}_te_idx_{w_i:02d}.npy", w_obj["te_idx"])
                df_tr = df.iloc[w_obj["tr_idx"]].copy()
                df_te = df.iloc[w_obj["te_idx"]].copy()
                df_tr["target_class"] = (y_arr[w_obj["tr_idx"]].astype(np.int8) - 3)
                df_te["target_class"] = (y_arr[w_obj["te_idx"]].astype(np.int8) - 3)
                fps_per_win.append((compute_regime_fingerprint(df_tr), compute_regime_fingerprint(df_te)))
            summary_meta[coin_key][h] = {"windows": wins, "fingerprints": fps_per_win}

        st_nz = float(np.mean(st_lbl["class_7"] != 0) * 100.0)
        bo_nz = float(np.mean(bo_lbl["class_7"] != 0) * 100.0)
        print(
            f"  • H={h}s K={k:g}: ST окон={len(wins_st)} (импульсных меток {st_nz:.2f}%) | "
            f"BO окон={len(wins_bo)} (импульсных меток {bo_nz:.2f}%)",
            flush=True,
        )
        for w_i, w_obj in enumerate(wins_st):
            print(
                f"      win_{w_i:02d}: Train [{str(ts_all[w_obj['tr_idx'][0]])[:19]} .. {str(ts_all[w_obj['tr_idx'][-1]])[:19]}] ({len(w_obj['tr_idx']):,}s) "
                f"-> Test [{str(ts_all[w_obj['te_idx'][0]])[:19]} .. {str(ts_all[w_obj['te_idx'][-1]])[:19]}] ({len(w_obj['te_idx']):,}s)",
                flush=True,
            )

    del df, df_era
    gc.collect()
    print(f"✅ [ФАЗА A] Завершена за {time.time() - t0:.1f}с!", flush=True)
    return summary_meta


def phase_b_train_lightgbm(summary_meta: dict, lgb_threads: int = 8):
    """Фаза B (100% CPU до инициализации CUDA): обучение LightGBM для H=1s, 2s на ST и BO."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight

    t0 = time.time()
    print("\n" + "=" * 105, flush=True)
    print("🌲 [ФАЗА B: CPU] ОБУЧЕНИЕ LIGHTGBM (450 ДЕРЕВЬЕВ) ДЛЯ H=1s И H=2s НА SAFETRADE И BIGONE", flush=True)
    print("=" * 105, flush=True)

    classes_values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)

    for coin_l, cache_sub in (("prl", "_mmap_cache"), ("prl_bo", "_mmap_cache_bo")):
        cache_dir = BASE_DIR / "data" / cache_sub
        feats_json = BASE_DIR / "data" / "features" / f"selected_features_{coin_l}_h8s.json"
        feature_cols = json.loads(feats_json.read_text())["feature_order"]
        X_all = np.load(cache_dir / "X_all.npy", mmap_mode="r")
        ts_all = np.load(cache_dir / "ts_all.npy", allow_pickle=True)

        for h, k in MICRO_PAIRS:
            y_all = np.load(cache_dir / f"y_all_h{h}.npy", mmap_mode="r")
            dt_all = np.load(cache_dir / f"dt_all_h{h}.npy", mmap_mode="r")
            wins = summary_meta[coin_l][h]["windows"]
            fps = summary_meta[coin_l][h]["fingerprints"]

            out_dir = BASE_DIR / "models" / coin_l / f"h{h}s_k{k:g}"
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest = []

            for w_i, w_obj in enumerate(wins):
                win_id = f"win_{w_i:02d}"
                tr_idx = w_obj["tr_idx"]
                te_idx = w_obj["te_idx"]
                X_tr = np.asarray(X_all[tr_idx], dtype=np.float32)
                y_tr = np.asarray(y_all[tr_idx], dtype=np.int32)
                X_te = np.asarray(X_all[te_idx], dtype=np.float32)
                y_te = np.asarray(y_all[te_idx], dtype=np.int32)
                dt_te = np.asarray(dt_all[te_idx], dtype=np.float32)

                present_tr = np.unique(y_tr)
                base_w = compute_class_weight("balanced", classes=present_tr, y=y_tr)
                custom_weights = {int(c): float(w_val) for c, w_val in zip(present_tr, base_w)}
                for c_boost, mult in ((0, 1.35), (1, 1.25), (2, 1.10)):
                    if c_boost in custom_weights:
                        custom_weights[c_boost] *= mult

                clf = lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=7,
                    class_weight=custom_weights,
                    n_estimators=450,
                    learning_rate=0.07,
                    num_leaves=31,
                    max_depth=6,
                    min_child_samples=40,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=1042 + w_i,
                    n_jobs=lgb_threads,
                    max_bin=255,
                    verbose=-1,
                )
                clf.fit(X_tr, y_tr, feature_name=feature_cols)
                probs = clf.predict_proba(X_te)
                full_probs = np.zeros((len(X_te), 7), dtype=np.float32)
                for col_i, cls_val in enumerate(clf.classes_):
                    if 0 <= int(cls_val) < 7:
                        full_probs[:, int(cls_val)] = probs[:, col_i]

                exp_z = np.dot(full_probs, classes_values)
                ic = float(np.corrcoef(exp_z, dt_te)[0, 1]) if np.std(exp_z) > 1e-6 and np.std(dt_te) > 1e-6 else 0.0
                try:
                    pres_te = [c for c in range(7) if np.sum(y_te == c) > 0]
                    if len(pres_te) > 1:
                        p_sub = full_probs[:, pres_te]
                        p_sub = p_sub / np.maximum(p_sub.sum(axis=1, keepdims=True), 1e-7)
                        auc = float(roc_auc_score(y_te, p_sub, multi_class="ovr", labels=pres_te))
                    else:
                        auc = 0.5
                except Exception:
                    auc = 0.5

                top2_thr = float(np.quantile(exp_z, 0.98))
                top2_mask = exp_z >= top2_thr
                top2_pnl = float(np.mean(dt_te[top2_mask])) if np.any(top2_mask) else 0.0

                model_path = out_dir / f"model_{win_id}.txt"
                clf.booster_.save_model(str(model_path))
                fp_tr, fp_te = fps[w_i]
                ts_tr_range = [str(ts_all[tr_idx[0]])[:19], str(ts_all[tr_idx[-1]])[:19]]
                ts_te_range = [str(ts_all[te_idx[0]])[:19], str(ts_all[te_idx[-1]])[:19]]

                meta = {
                    "window_id": win_id,
                    "coin": coin_l.upper(),
                    "H": h,
                    "K": k,
                    "L": h * k,
                    "eff_tick": MICRO_EFF_TICK,
                    "train_ts_range": ts_tr_range,
                    "test_ts_range": ts_te_range,
                    "n_train": int(len(X_tr)),
                    "n_test": int(len(X_te)),
                    "metrics_oos": {
                        "ic_dt_ticks": round(ic, 4),
                        "roc_auc_ovr": round(auc, 4),
                        "top2pct_mean_ticks": round(top2_pnl, 2),
                    },
                    "regime_fingerprint_train": fp_tr,
                    "regime_fingerprint_test": fp_te,
                    "model_file": model_path.name,
                }
                (out_dir / f"meta_{win_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                manifest.append(meta)
                print(
                    f"  🌲 [{coin_l.upper():<6} LGB H={h}s {win_id}] IC={ic:+6.4f} | AUC={auc:6.4f} | Top2%={top2_pnl:+6.2f}t "
                    f"| Test: {ts_te_range[0]} -> {ts_te_range[1]}",
                    flush=True,
                )
                del clf, X_tr, X_te, probs, full_probs
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gc.collect()
    print(f"✅ [ФАЗА B] Все модели LightGBM H=1s, 2s обучены за {time.time() - t0:.1f}с!", flush=True)


def _seed_h5s_templates_for_micro_horizons():
    """
    Копирует один эталонный `.keras` из `models/prl/{arch}_h5s` (`sl = 15`) в целевые папки
    `{arch}_h1s` и `{arch}_h2s` как `template_seed_sl15.keras` (удаляется после обучения),
    чтобы `mlp_bottleneck` и остальные 5 архитектур гарантированно имели входную размерность `sl = 15`,
    а не `sl = 180` от `h60s`!
    """
    seeded = []
    for coin_l in ("prl", "prl_bo"):
        for h, _ in MICRO_PAIRS:
            for arch in NEURAL_ARCHS:
                a_clean = arch.replace("+", "_")
                src_dir = BASE_DIR / "models" / "prl" / f"{a_clean}_h5s"
                src_list = sorted(src_dir.glob("model_win_*.keras"))
                if not src_list:
                    raise FileNotFoundError(f"Не найден шаблон sl=15 в {src_dir}")
                dst_dir = BASE_DIR / "models" / coin_l / f"{a_clean}_h{h}s"
                dst_dir.mkdir(parents=True, exist_ok=True)
                # Если в папке еще нет model_win_*.keras, временно кладем model_win_99.keras (sl=15)
                existing = sorted(dst_dir.glob("model_win_*.keras"))
                if not existing:
                    seed_p = dst_dir / "model_win_99.keras"
                    shutil.copy2(src_list[-1], seed_p)
                    seeded.append(seed_p)
    return seeded


def _gpu_worker_run_task(task: dict) -> dict:
    """Изолированный GPU-воркер (spawn): вызывает эталонный `train_pytorch_window_fast` с замером пикового VRAM."""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import io
    import contextlib
    import torch
    torch.set_num_threads(2)
    from _tools.fast_neural_engine import train_pytorch_window_fast

    t0 = time.time()
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
            force=True,
            coin=task["coin"],
        )
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    reserved_mib = float(torch.cuda.max_memory_reserved() / (1024 * 1024))
    free_mib, total_mib = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
    return {
        **task,
        "elapsed": round(time.time() - t0, 2),
        "peak_alloc_mib": round(peak_mib, 1),
        "peak_reserved_mib": round(reserved_mib, 1),
        "free_mib_after": round(free_mib, 1),
        "total_mib": round(total_mib, 1),
        "result": res,
    }


def _run_fne_with_sl15(
    fne,
    cache_dir: str,
    horizon: int,
    win_idx: int,
    arch: str,
    runs: int,
    epochs: int,
    batch_size: int,
    coin: str,
) -> dict:
    """
    Обучает микро-окно `H in {1, 2}` с рецептивным полем `sl = 15` (15 секунд истории книги и ленты),
    полностью совместимым с архитектурами `wavenet_tcn`, `causal_cnn`, `mlp_bottleneck`, `cross_mixer`,
    `attention` и `conv1d+gru` и боевым `FastKerasModel`.
    """
    import json
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight

    sl = 15
    win_tag = f"win_{win_idx:02d}"
    arch_dir = BASE_DIR / "models" / coin.lower() / f"{arch.replace('+', '_')}_h{horizon}s"
    arch_dir.mkdir(parents=True, exist_ok=True)
    out_keras = arch_dir / f"model_{win_tag}.keras"
    out_meta = arch_dir / f"meta_{win_tag}.json"

    src_h5_dir = BASE_DIR / "models" / "prl" / f"{arch.replace('+', '_')}_h5s"
    templates = sorted(src_h5_dir.glob("model_win_*.keras"))
    if not templates:
        raise FileNotFoundError(f"Нет шаблона sl=15 в {src_h5_dir}")
    tpl_keras = templates[-1]

    cdir = Path(cache_dir)
    h_key = f"h{int(horizon)}"
    tr_idx = np.load(cdir / f"{h_key}_tr_idx_{win_idx:02d}.npy")
    te_idx = np.load(cdir / f"{h_key}_te_idx_{win_idx:02d}.npy")
    X_all = np.load(cdir / "X_all.npy", mmap_mode="r")
    ts_all = np.load(cdir / "ts_all.npy", allow_pickle=True)
    y_all = np.load(cdir / f"y_all_{h_key}.npy", mmap_mode="r")
    dt_all = np.load(cdir / f"dt_all_{h_key}.npy", mmap_mode="r")

    X_tr = np.array(X_all[tr_idx], dtype=np.float32)
    y_tr = np.array(y_all[tr_idx], dtype=np.int64)
    ts_tr_range = [str(ts_all[tr_idx[0]])[:19], str(ts_all[tr_idx[-1]])[:19]]
    ts_te_range = [str(ts_all[te_idx[0]])[:19], str(ts_all[te_idx[-1]])[:19]]
    del ts_all

    mean = np.nanmean(X_tr, axis=0).astype(np.float32)
    std = (np.nanstd(X_tr, axis=0) + 1e-6).astype(np.float32)
    X_tr -= mean
    X_tr /= std
    np.clip(X_tr, -5.0, 5.0, out=X_tr)
    np.nan_to_num(X_tr, copy=False)

    X_te = np.array(X_all[te_idx], dtype=np.float32)
    y_te = np.array(y_all[te_idx], dtype=np.int64)
    dt_te = np.array(dt_all[te_idx], dtype=np.float32)
    X_te -= mean
    X_te /= std
    np.clip(X_te, -5.0, 5.0, out=X_te)
    np.nan_to_num(X_te, copy=False)

    n_tr, n_te = len(X_tr), len(X_te)
    val_split = int(n_tr * 0.85)
    valid_starts_tr = np.arange(sl - 1, val_split, dtype=np.int64)
    valid_starts_va = np.arange(max(sl - 1, val_split), n_tr, dtype=np.int64)
    valid_starts_te = np.arange(0, n_te, dtype=np.int64)

    present = np.unique(y_tr[:val_split])
    cw_vals = compute_class_weight("balanced", classes=present, y=y_tr[:val_split])
    cw_np = np.ones(7, dtype=np.float32)
    for c_i, w_v in zip(present, cw_vals):
        cw_np[int(c_i)] = float(np.clip(w_v, 0.35, 6.0))
    cw_np[0] *= 1.25
    cw_np[6] *= 1.25

    dev = torch.device("cuda")
    x_tr_gpu = torch.as_tensor(X_tr, dtype=torch.float16, device=dev)
    x_te_pad = np.vstack([np.repeat(X_te[:1], sl - 1, axis=0), X_te])
    x_te_gpu = torch.as_tensor(x_te_pad, dtype=torch.float16, device=dev)
    y_tr_gpu = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
    y_va_np = y_tr[valid_starts_va]
    dt_va_np = np.array(dt_all[tr_idx[valid_starts_va]], dtype=np.float32)
    cw_t = torch.as_tensor(cw_np, dtype=torch.float32, device=dev)
    del X_tr, X_te

    starts_tr_gpu = torch.as_tensor(valid_starts_tr, dtype=torch.long, device=dev)
    starts_va_gpu = torch.as_tensor(valid_starts_va, dtype=torch.long, device=dev)
    starts_te_gpu = torch.as_tensor(valid_starts_te + (sl - 1), dtype=torch.long, device=dev)

    cls_vals = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
    best_score = -1e9
    best_state = None
    best_metrics = None

    for r_i in range(runs):
        gen = torch.Generator(device=dev)
        gen.manual_seed(202611 + win_idx * 97 + r_i * 1009)
        model = fne._TrainableKerasGraph(tpl_keras, arch).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=2e-4)
        n_steps = max(1, (len(starts_tr_gpu) + batch_size - 1) // batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * n_steps))

        best_va_ic = -1e9
        best_r_state = None
        for _ep in range(epochs):
            model.train()
            perm = torch.randperm(len(starts_tr_gpu), generator=gen, device=dev)
            shuf_idx = starts_tr_gpu[perm]
            for s_b in range(0, len(shuf_idx), batch_size):
                b_idx = shuf_idx[s_b : s_b + batch_size]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.forward_batch(x_tr_gpu, b_idx, sl)
                    loss = F.cross_entropy(logits.float(), y_tr_gpu[b_idx], weight=cw_t, label_smoothing=0.02)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

            model.eval()
            va_probs_l = []
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for s_b in range(0, len(starts_va_gpu), 4096):
                    b_idx = starts_va_gpu[s_b : s_b + 4096]
                    va_probs_l.append(F.softmax(model.forward_batch(x_tr_gpu, b_idx, sl).float(), dim=-1).cpu().numpy())
            va_p = np.vstack(va_probs_l)
            va_ez = va_p @ cls_vals
            va_ic = float(np.corrcoef(va_ez, dt_va_np)[0, 1]) if np.std(va_ez) > 1e-6 and np.std(dt_va_np) > 1e-6 else 0.0
            if va_ic > best_va_ic or best_r_state is None:
                best_va_ic = va_ic
                best_r_state = {k_p: v_p.detach().cpu().clone() for k_p, v_p in model.state_dict().items()}

        model.load_state_dict({k_p: v_p.to(dev) for k_p, v_p in best_r_state.items()})
        model.eval()
        te_probs_l = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for s_b in range(0, len(starts_te_gpu), 4096):
                b_idx = starts_te_gpu[s_b : s_b + 4096]
                te_probs_l.append(F.softmax(model.forward_batch(x_te_gpu, b_idx, sl).float(), dim=-1).cpu().numpy())
        probs = np.vstack(te_probs_l)
        exp_z = probs @ cls_vals
        ic = float(np.corrcoef(exp_z, dt_te)[0, 1]) if np.std(exp_z) > 1e-6 and np.std(dt_te) > 1e-6 else 0.0
        try:
            pres = [c for c in range(7) if np.sum(y_te == c) > 0]
            if len(pres) > 1:
                ps = probs[:, pres]
                ps = ps / np.maximum(ps.sum(axis=1, keepdims=True), 1e-7)
                auc = float(roc_auc_score(y_te, ps, multi_class="ovr", labels=pres))
            else:
                auc = 0.5
        except Exception:
            auc = 0.5
        q98 = float(np.quantile(exp_z, 0.98))
        top2 = float(np.mean(dt_te[exp_z >= q98])) if np.any(exp_z >= q98) else 0.0
        eps_p = np.clip(probs[np.arange(len(y_te)), y_te], 1e-7, 1.0)
        ce_loss = float(-np.mean(np.log(eps_p)))

        # Выбор лучшего мультистарта строго по валидационному best_va_ic (0% утечки OOS!),
        # при равенстве — по комбинированной стабильности
        if best_va_ic > best_score or best_state is None:
            best_score = best_va_ic
            best_state = best_r_state
            best_metrics = {
                "ic_dt_ticks": round(ic, 4),
                "roc_auc_ovr": round(auc, 4),
                "log_loss": round(ce_loss, 4),
                "top2pct_mean_ticks": round(top2, 2),
                "val_ic": round(best_va_ic, 4),
            }

    export_model = fne._TrainableKerasGraph(tpl_keras, arch)
    export_model.load_state_dict(best_state)
    export_model.export_to_keras(out_keras)
    meta_payload = {
        "window_id": win_tag,
        "arch": arch,
        "coin": coin.upper(),
        "H": int(horizon),
        "seq_len": sl,
        "eff_tick": MICRO_EFF_TICK,
        "runs": int(runs),
        "train_ts_range": ts_tr_range,
        "test_ts_range": ts_te_range,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "metrics_oos": best_metrics,
    }
    out_meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    del x_tr_gpu, x_te_gpu, y_tr_gpu, starts_tr_gpu, starts_va_gpu, starts_te_gpu
    torch.cuda.empty_cache()
    return best_metrics


def _probe_vram_and_choose_workers(sample_task: dict, hard_cap: int = 7) -> tuple[int, dict]:
    """
    Измеряет свободную видеопамять и реальный расход VRAM на 1 пробном воркере,
    затем вычисляет безопасное число одновременных процессов с резервом >= 1,250 MiB.
    """
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        probe_res = ex.submit(_gpu_worker_run_task, sample_task).result()

    # Реальный вклад одного воркера в системную VRAM = CUDA context (~420 MiB) + peak_reserved_mib
    free_before_est = probe_res["free_mib_after"] + probe_res["peak_reserved_mib"] + 430.0
    per_worker_vram = max(650.0, probe_res["peak_reserved_mib"] + 460.0)
    safety_margin_mib = 1250.0
    usable_mib = max(per_worker_vram, free_before_est - safety_margin_mib)
    n_workers = int(np.clip(int(usable_mib // per_worker_vram), 2, hard_cap))
    info = {
        "total_vram_mib": probe_res["total_mib"],
        "estimated_free_vram_mib": round(free_before_est, 1),
        "probe_peak_alloc_mib": probe_res["peak_alloc_mib"],
        "probe_peak_reserved_mib": probe_res["peak_reserved_mib"],
        "per_worker_budget_mib": round(per_worker_vram, 1),
        "selected_parallel_workers": n_workers,
    }
    return n_workers, probe_res | {"vram_sizing": info}


def phase_c_parallel_gpu_neural_training(runs: int = 12, epochs: int = 10, max_workers_cap: int = 7):
    """Фаза C: Параллельное мультипроцессное GPU-обучение всех нейросетей H=1s, 2s."""
    t0 = time.time()
    print("\n" + "=" * 105, flush=True)
    print("🚀 [ФАЗА C: PARALLEL GPU] КАЛИБРОВКА VRAM И ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ 6 АРХИТЕКТУР (H=1s, 2s, ST + BO)", flush=True)
    print("=" * 105, flush=True)

    all_tasks: list[dict] = []
    for coin_l, cache_sub in (("prl", "_mmap_cache"), ("prl_bo", "_mmap_cache_bo")):
        cache_dir = BASE_DIR / "data" / cache_sub
        for h, _ in MICRO_PAIRS:
            n_w = int(np.load(cache_dir / f"h{h}_n_windows.npy"))
            for arch in NEURAL_ARCHS:
                ep = 8 if arch == "conv1d+gru" else epochs
                for w_i in range(n_w):
                    all_tasks.append({
                        "coin": coin_l,
                        "cache_dir": str(cache_dir),
                        "horizon": h,
                        "win_idx": w_i,
                        "arch": arch,
                        "runs": runs,
                        "epochs": ep,
                        "batch_size": 2048,
                    })

    print(f"📋 Всего нейросетевых задач к обучению: {len(all_tasks)} (runs={runs})", flush=True)
    # Выберем задачу H=2s attention как верхнюю оценку пикового потребления VRAM среди H=1s, 2s
    probe_idx = next(
        (i for i, t in enumerate(all_tasks) if t["horizon"] == 2 and t["arch"] == "attention"),
        0,
    )
    probe_task = all_tasks.pop(probe_idx)
    n_workers, probe_out = _probe_vram_and_choose_workers(probe_task, hard_cap=max_workers_cap)
    vs = probe_out["vram_sizing"]
    m0 = probe_out["result"]
    print(
        f"🔍 Замер VRAM (пробная задача {probe_out['coin'].upper()} {probe_out['arch'].upper()} H={probe_out['horizon']}s win_{probe_out['win_idx']:02d}):\n"
        f"   • Всего VRAM: {vs['total_vram_mib']:.0f} MiB | Доступно: ~{vs['estimated_free_vram_mib']:.0f} MiB\n"
        f"   • Пик тензоров воркера: alloc={vs['probe_peak_alloc_mib']:.1f} MiB, reserved={vs['probe_peak_reserved_mib']:.1f} MiB "
        f"(полный бюджет с CUDA-контекстом ~{vs['per_worker_budget_mib']:.0f} MiB/воркер)\n"
        f"   • ⚡ Выбрано параллельных GPU-процессов: {n_workers} (резерв безопасности 1,250 MiB)\n"
        f"   • Пробный результат ({probe_out['elapsed']:.1f}s): IC={m0['ic_dt_ticks']:+.4f} | AUC={m0['roc_auc_ovr']:.4f} | Top2%={m0['top2pct_mean_ticks']:+.2f}t",
        flush=True,
    )

    done_cnt = 1
    total_cnt = len(all_tasks) + 1
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futs = {pool.submit(_gpu_worker_run_task, t): t for t in all_tasks}
        for fut in as_completed(futs):
            out = fut.result()
            done_cnt += 1
            mo = out["result"]
            print(
                f"  [{done_cnt:2d}/{total_cnt:2d}] 🧠 {out['coin'].upper():<6} {out['arch'].upper():<15} "
                f"H={out['horizon']}s win_{out['win_idx']:02d} ({out['elapsed']:4.1f}s, freeVRAM={out['free_mib_after']:4.0f}M) "
                f"-> IC={mo['ic_dt_ticks']:+6.4f} | AUC={mo['roc_auc_ovr']:6.4f} | Top2%={mo['top2pct_mean_ticks']:+6.2f}t",
                flush=True,
            )

    # Сборка manifest.json и ensemble_alliance_live_h{1,2}s.json
    for coin_l, cache_sub in (("prl", "_mmap_cache"), ("prl_bo", "_mmap_cache_bo")):
        cache_dir = BASE_DIR / "data" / cache_sub
        for h, _ in MICRO_PAIRS:
            n_w = int(np.load(cache_dir / f"h{h}_n_windows.npy"))
            selected_models = []
            weights = {}
            for arch in NEURAL_ARCHS:
                a_dir = BASE_DIR / "models" / coin_l / f"{arch.replace('+', '_')}_h{h}s"
                manifest = []
                for w_i in range(n_w):
                    mp_f = a_dir / f"meta_win_{w_i:02d}.json"
                    if mp_f.exists():
                        m_obj = json.loads(mp_f.read_text())
                        manifest.append(m_obj)
                        ic = float(m_obj.get("metrics_oos", {}).get("ic_dt_ticks", 0.0))
                        if ic > 0.015:
                            key = f"{arch}_win_{w_i:02d}"
                            selected_models.append(key)
                            weights[key] = round(max(0.001, ic ** 2), 6)
                (a_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            w_sum = sum(weights.values()) + 1e-12
            weights = {k_m: round(v_m / w_sum, 6) for k_m, v_m in weights.items()}
            payload = {
                "coin": coin_l.upper(),
                "horizon_sec": h,
                "eff_tick": MICRO_EFF_TICK,
                "n_windows": n_w,
                "selected_models": selected_models,
                "weights": weights,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for fn in (f"ensemble_alliance_h{h}s.json", f"ensemble_alliance_live_h{h}s.json"):
                (BASE_DIR / "models" / coin_l / fn).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"✅ [ФАЗА C] Параллельное обучение {total_cnt} нейросетей завершено за {(time.time() - t0) / 60.0:.2f} мин!", flush=True)


def print_micro_summary():
    for coin_l, cache_sub in (("prl", "_mmap_cache"), ("prl_bo", "_mmap_cache_bo")):
        cache_dir = BASE_DIR / "data" / cache_sub
        print("\n" + "=" * 112, flush=True)
        print(f"📊 ИТОГОВАЯ OOS-СВОДКА МИКРО-ГОРИЗОНТОВ ($0.0001 ЭРА): {coin_l.upper()} (models/{coin_l}/)", flush=True)
        print("=" * 112, flush=True)
        print(f"{'Горизонт':<12} | {'Окон':>4} | {'LGB Mean IC':>11} | {'LGB Mean AUC':>12} | {'NN Best IC':>10} | {'NN Mean AUC':>11} | {'Экспертов в альянсе':>20}", flush=True)
        print("-" * 112, flush=True)
        for h, k in MICRO_PAIRS:
            n_w = int(np.load(cache_dir / f"h{h}_n_windows.npy"))
            lgb_dir = BASE_DIR / "models" / coin_l / f"h{h}s_k{k:g}"
            lgb_ics, lgb_aucs = [], []
            for w_i in range(n_w):
                mp_f = lgb_dir / f"meta_win_{w_i:02d}.json"
                if mp_f.exists():
                    mo = json.loads(mp_f.read_text()).get("metrics_oos", {})
                    lgb_ics.append(float(mo.get("ic_dt_ticks", 0.0)))
                    lgb_aucs.append(float(mo.get("roc_auc_ovr", 0.5)))

            nn_ics, nn_aucs = [], []
            for arch in NEURAL_ARCHS:
                a_dir = BASE_DIR / "models" / coin_l / f"{arch.replace('+', '_')}_h{h}s"
                for w_i in range(n_w):
                    mp_f = a_dir / f"meta_win_{w_i:02d}.json"
                    if mp_f.exists():
                        mo = json.loads(mp_f.read_text()).get("metrics_oos", {})
                        nn_ics.append(float(mo.get("ic_dt_ticks", 0.0)))
                        nn_aucs.append(float(mo.get("roc_auc_ovr", 0.5)))

            al_p = BASE_DIR / "models" / coin_l / f"ensemble_alliance_live_h{h}s.json"
            n_sel = len(json.loads(al_p.read_text()).get("selected_models", [])) if al_p.exists() else 0
            print(
                f"H={h:3d}s K={k:<3g} | {n_w:4d} | {np.mean(lgb_ics) if lgb_ics else 0.0:+11.4f} | "
                f"{np.mean(lgb_aucs) if lgb_aucs else 0.5:12.4f} | {np.max(nn_ics) if nn_ics else 0.0:+10.4f} | "
                f"{np.mean(nn_aucs) if nn_aucs else 0.5:11.4f} | {n_sel:20d}",
                flush=True,
            )
        print("=" * 112, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "cpu", "gpu"], default="all")
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-gpu-workers", type=int, default=6)
    ap.add_argument("--lgb-threads", type=int, default=6)
    args = ap.parse_args()

    if args.phase in ("all", "cpu"):
        meta = phase_a_prepare_micro_caches()
        phase_b_train_lightgbm(meta, lgb_threads=args.lgb_threads)

    if args.phase in ("all", "gpu"):
        phase_c_parallel_gpu_neural_training(
            runs=args.runs,
            epochs=args.epochs,
            max_workers_cap=args.max_gpu_workers,
        )
        print_micro_summary()


if __name__ == "__main__":
    main()
