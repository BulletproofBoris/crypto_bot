#!/usr/bin/env python3
"""
_tools/build_hybrid_v7_v8_cache.py
=============================================================================
Сборщик единого гибридного кэша (v8 Spatial-Temporal LOB + v7 Walk-Forward Swarm
с 8-фазным Классификатором-Роутером + CarouselNet v3 + HazardNet v3 + ExecutionRealismNet).

Формирует 58-мерную каузальную матрицу v7_signals_58d на всей глубине истории (681+ ч)
потоковыми чанками по 300,000 тиков (без переполнения RAM):
  - [0..17]  (18D): H=5s, 12s, 25s (ez, p_up, p_down, p_tp, p_tn, ent)
  - [18..27] (10D): Межбиржевые и микроструктурные признаки (SafeTrade + BigONE)
  - [28..33]  (6D): H=60s (ez, p_up, p_down, p_tp, p_tn, ent)
  - [34..39]  (6D): H=90s (ez, p_up, p_down, p_tp, p_tn, ent)
  - [40..45]  (6D): H=120s (ez, p_up, p_down, p_tp, p_tn, ent)
  - [46..49]  (4D): CarouselNet v3 + HazardNet v3 (X_v52)
  - [50..51]  (2D): ExecutionRealismNet (X_v70_realism: p_fill_real, p_frontrun)
  - [52..57]  (6D): Индикаторы зрелости/открытия горизонтов ln(1 + N_avail_win(t)) / 4.0
                    для H=5s, 12s, 25s, 60s, 90s, 120s.
=============================================================================
"""

import gc
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CACHE_600H_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
CACHE_176H_PATH = BASE_DIR / "data" / "cache" / "rl_triad_lob_cache_prl_176h.npz"


def compute_horizon_maturity_6d(ts_unix_sec: np.ndarray, coin: str = "prl") -> np.ndarray:
    """
    Для каждой секунды t вычисляет 6-мерный вектор зрелости горизонтов:
    log1p(число уже обученных к секунде t окон) / 4.0 для H = 5, 12, 25, 60, 90, 120s.
    До завершения win_00 соответствующего горизонта значение строго равно 0.0!
    """
    n = len(ts_unix_sec)
    maturity = np.zeros((n, 6), dtype=np.float32)
    horizons = [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]

    for h_idx, (H, K) in enumerate(horizons):
        m_dir = BASE_DIR / "models" / coin / f"h{H}s_k{K:g}"
        metas = sorted(m_dir.glob("meta_win_*.json"))
        if not metas:
            continue
        try:
            manifest = [json.loads(p.read_text(encoding="utf-8")) for p in metas]
            tr_ends = []
            for item in manifest:
                t_range = item.get("train_ts_range", ["1970-01-01", "1970-01-01"])
                t_end_str = t_range[1] if isinstance(t_range, list) and len(t_range) > 1 else "1970-01-01"
                tr_ends.append(int(pd.Timestamp(t_end_str, tz="UTC").timestamp()))
            if not tr_ends:
                continue
            tr_ends_arr = np.sort(np.array(tr_ends, dtype=np.int64))
            n_open = np.searchsorted(tr_ends_arr, ts_unix_sec, side="left").astype(np.float32)
            maturity[:, h_idx] = np.log1p(n_open) / 4.0
        except Exception as e:
            print(f"⚠️ Ошибка чтения зрелости H={H}s: {e}")

    return maturity


def _extract_6d_from_ens(ens_out: dict) -> np.ndarray:
    probs = ens_out["probs"]
    ez = ens_out["expected_z"]
    p_up = probs[:, 4] + probs[:, 5] + probs[:, 6]
    p_down = probs[:, 0] + probs[:, 1] + probs[:, 2]
    p_tp = probs[:, 5] + probs[:, 6]
    p_tn = probs[:, 0] + probs[:, 1]
    p_clip = np.clip(probs, 1e-7, 1.0)
    ent = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(7.0)
    return np.column_stack([ez, p_up, p_down, p_tp, p_tn, ent]).astype(np.float32)


def build_and_inject_v7_signals_58d(coin: str = "prl", force_recompute: bool = True, chunk_size: int = 300000) -> Path:
    """
    Потоковый чанковый расчёт 58D каузальных сигналов v7:
      1) Для свежих 176ч (633,600 тиков) напрямую переиспользует полный гибридный кэш rl_triad_lob_cache_prl_176h.npz
         (LightGBM + 6 нейросетей + CarouselNet v3 + HazardNet v3 + ExecutionRealismNet).
      2) Для исторического префикса (0 .. N - 633,600) вычисляет сигналы роя с 8-фазным классификатором-роутером
         + CarouselNet v3 + HazardNet v3 + ExecutionRealismNet потоковыми чанками по 300,000 строк (< 600 МБ RAM).
      3) Налагает строгую каузальную маску открытия горизонтов и сохраняет v7_signals_58d в rl_guard_lob_cache_600h.npz.
    """
    from _tools.swarm_router import SwarmRegimeRouter
    from _tools.train_rl_execution import _compute_v52_signals_matrix, _compute_v70_realism_matrix

    t0 = time.time()
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    raw_600 = np.load(CACHE_600H_PATH, allow_pickle=True)
    ts_600 = raw_600["ts"].astype(np.int64)
    n_total = len(ts_600)
    prev_v7_52d = None
    if "v7_signals_58d" in raw_600.files and len(raw_600["v7_signals_58d"]) == n_total:
        prev_v7_52d = raw_600["v7_signals_58d"][:, :52].copy()
    del raw_600

    if np.max(ts_600) < 10**9 or np.min(ts_600) < 10**9:
        ts_raw = pq.read_table(labeled_path, columns=["timestamp"]).column("timestamp").to_numpy()
        ts_600 = pd.to_datetime(ts_raw[:n_total], utc=True).as_unit("s").astype("int64").to_numpy()
    elif np.max(ts_600) > 10**12:
        ts_600 = pd.to_datetime(ts_600, utc=True).as_unit("s").astype("int64").to_numpy()

    print(f"🔗 Потоковая сборка 58D каузальных сигналов v7 для {n_total:,} тиков ({n_total / 3600:.1f} ч)...")
    aligned_52d = np.zeros((n_total, 52), dtype=np.float32)

    # 1. Сохраняем уже посчитанный инкрементальный хвост [2451560:n_total] и 176ч блок [1818560:2451560]
    tail_matched_start = n_total
    if prev_v7_52d is not None and n_total > 2451560 and np.any(prev_v7_52d[2451560:] != 0):
        aligned_52d[2451560:] = prev_v7_52d[2451560:]
        tail_matched_start = 2451560
        print(f"   ✅ Инкрементальный хвост ({n_total - 2451560:,} тиков: [2451560:{n_total}]) сохранен.")
    del prev_v7_52d

    if CACHE_176H_PATH.exists():
        c176 = np.load(CACHE_176H_PATH, allow_pickle=True)
        xm = np.nan_to_num(c176["X_market"].astype(np.float32), nan=0.0, posinf=3.0, neginf=-3.0)
        x60 = np.nan_to_num(c176["X_h60"].astype(np.float32), nan=0.0, posinf=3.0, neginf=-3.0)
        x90 = np.nan_to_num(c176["X_h90"].astype(np.float32), nan=0.0, posinf=3.0, neginf=-3.0)
        x120 = np.nan_to_num(c176["X_h120"].astype(np.float32), nan=0.0, posinf=3.0, neginf=-3.0)
        xv52 = np.nan_to_num(c176["X_v52"][:, :4].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        xv70 = np.nan_to_num(c176["X_v70_realism"][:, :2].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        tail_52d = np.column_stack([xm[:, :28], x60[:, :6], x90[:, :6], x120[:, :6], xv52, xv70]).astype(np.float32)
        if tail_matched_start == 2451560 and len(tail_52d) >= 633000:
            aligned_52d[1818560:2451560] = tail_52d[:633000]
            tail_matched_start = 1818560
            print(f"   ✅ Блок 176ч (633,000 тиков: [1818560:2451560]) переиспользован из {CACHE_176H_PATH.name}.")
        else:
            n_tail = len(tail_52d)
            tail_matched_start = max(0, n_total - n_tail)
            aligned_52d[tail_matched_start:] = tail_52d[: n_total - tail_matched_start]
            print(f"   ✅ Свежий хвост 176ч ({n_total - tail_matched_start:,} тиков: [{tail_matched_start}:{n_total}]) переиспользован из {CACHE_176H_PATH.name}.")
        del c176, xm, x60, x90, x120, xv52, xv70, tail_52d
        gc.collect()

    # 2. Потоковый чанковый расчёт исторического префикса [0 : tail_matched_start]
    if tail_matched_start > 0:
        mmap_dir = BASE_DIR / "data" / "_mmap_cache"
        X_all_mm = np.load(mmap_dir / "X_all.npy", mmap_mode="r")
        pf = pq.ParquetFile(labeled_path)
        schema_cols = set(pf.schema_arrow.names)
        meta_cols = [
            "timestamp", "ST_spread", "ST_vol_60s_ticks", "ST_ofi_norm_300s",
            "ST_vol_15s_ticks", "ST_ofi_15s", "ST_ofi_60s", "ST_depth_ratio_5",
            "BO_ofi_15s", "BO_ret_15s", "LL_zscore_30s", "BO_sec_since_tick",
            "ST_vol_300s_ticks", "ST_trade_vol", "ST_buy_vol", "ST_sell_vol", "ST_vwap_1s",
        ]
        for lvl in range(1, 11):
            meta_cols.extend([f"ST_bid_{lvl}_price", f"ST_bid_{lvl}_qty", f"ST_ask_{lvl}_price", f"ST_ask_{lvl}_qty"])
        meta_cols = [c for c in list(dict.fromkeys(meta_cols)) if c in schema_cols]
        df_meta_all = pq.read_table(labeled_path, columns=meta_cols).slice(0, tail_matched_start).to_pandas()
        if "timestamp" not in df_meta_all.columns:
            df_meta_all = df_meta_all.reset_index()
            if "timestamp" not in df_meta_all.columns and "index" in df_meta_all.columns:
                df_meta_all = df_meta_all.rename(columns={"index": "timestamp"})

        horizons = [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]
        routers = [SwarmRegimeRouter(coin, H, K, temperature=0.25, use_neural=False, neural_weight=0.0) for H, K in horizons]

        for s in range(0, tail_matched_start, chunk_size):
            e = min(tail_matched_start, s + chunk_size)
            print(f"   ⚡ Чанк истории [{s:,}:{e:,}] / {tail_matched_start:,} ({(e-s)/3600:.1f} ч)...", flush=True)
            x_chunk = np.array(X_all_mm[s:e], dtype=np.float32)
            df_chunk = df_meta_all.iloc[s:e].reset_index(drop=True)

            h_mats = []
            for r in routers:
                ens = r.predict_ensemble(x_chunk, df_chunk, mode="router")
                r._lgb_cache_probs = None
                h_mats.append(_extract_6d_from_ens(ens))
                del ens

            bid_p = np.column_stack([np.nan_to_num(df_chunk[f"ST_bid_{l}_price"].values, nan=0.0) for l in range(1, 11)])
            bid_q = np.column_stack([np.nan_to_num(df_chunk[f"ST_bid_{l}_qty"].values, nan=0.0) for l in range(1, 11)])
            ask_p = np.column_stack([np.nan_to_num(df_chunk[f"ST_ask_{l}_price"].values, nan=0.0) for l in range(1, 11)])
            ask_q = np.column_stack([np.nan_to_num(df_chunk[f"ST_ask_{l}_qty"].values, nan=0.0) for l in range(1, 11)])

            spread_ticks = np.clip((ask_p[:, 0] - bid_p[:, 0]) / 0.01, 0.0, 50.0)
            b1_q, a1_q = np.maximum(bid_q[:, 0], 0.0), np.maximum(ask_q[:, 0], 0.0)
            b5_q, a5_q = np.sum(np.maximum(bid_q[:, :5], 0.0), axis=1), np.sum(np.maximum(ask_q[:, :5], 0.0), axis=1)
            b10_q, a10_q = np.sum(np.maximum(bid_q, 0.0), axis=1), np.sum(np.maximum(ask_q, 0.0), axis=1)
            vol_300 = np.nan_to_num(df_chunk["ST_vol_300s_ticks"].values, nan=1.0)
            ofi_300 = np.nan_to_num(df_chunk["ST_ofi_norm_300s"].values, nan=0.0)
            x_lob = np.column_stack([
                spread_ticks / 10.0,
                (b1_q - a1_q) / (b1_q + a1_q + 1e-6),
                (b5_q - a5_q) / (b5_q + a5_q + 1e-6),
                (b10_q - a10_q) / (b10_q + a10_q + 1e-6),
                np.log1p(b1_q) / 10.0, np.log1p(a1_q) / 10.0,
                np.log1p(b5_q) / 12.0, np.log1p(a5_q) / 12.0,
                np.clip(vol_300 / 5.0, 0.0, 5.0), np.clip(ofi_300, -3.0, 3.0),
            ]).astype(np.float32)

            xm_c = np.hstack([h_mats[0], h_mats[1], h_mats[2], x_lob]).astype(np.float32)
            vwap_arr = (
                np.nan_to_num(df_chunk["ST_vwap_1s"].values, nan=0.0)
                if "ST_vwap_1s" in df_chunk.columns
                else (bid_p[:, 0] + ask_p[:, 0]) * 0.5
            )
            b_vol_arr = (
                np.nan_to_num(df_chunk["ST_buy_vol"].values, nan=0.0)
                if "ST_buy_vol" in df_chunk.columns
                else np.zeros(len(df_chunk), dtype=np.float32)
            )
            s_vol_arr = (
                np.nan_to_num(df_chunk["ST_sell_vol"].values, nan=0.0)
                if "ST_sell_vol" in df_chunk.columns
                else np.zeros(len(df_chunk), dtype=np.float32)
            )
            chunk_dict = {
                "X_market": xm_c, "X_h60": h_mats[3], "X_h90": h_mats[4], "X_h120": h_mats[5],
                "bid_p": bid_p, "bid_q": bid_q, "ask_p": ask_p, "ask_q": ask_q,
                "buy_vol": b_vol_arr,
                "sell_vol": s_vol_arr,
                "vwap": vwap_arr,
                "tick_size": 0.01,
            }
            xv52_c = _compute_v52_signals_matrix(chunk_dict, coin=coin)[:, :4].astype(np.float32)
            xv70_c = _compute_v70_realism_matrix(chunk_dict, coin=coin)[:, :2].astype(np.float32)
            aligned_52d[s:e] = np.column_stack([xm_c, h_mats[3], h_mats[4], h_mats[5], xv52_c, xv70_c])
            del x_chunk, df_chunk, h_mats, chunk_dict, xv52_c, xv70_c
            gc.collect()
        del df_meta_all, routers, X_all_mm
        gc.collect()

    maturity_6d = compute_horizon_maturity_6d(ts_600, coin=coin)
    aligned_52d[maturity_6d[:, 0] <= 0.0, 0:6] = 0.0
    aligned_52d[maturity_6d[:, 1] <= 0.0, 6:12] = 0.0
    aligned_52d[maturity_6d[:, 2] <= 0.0, 12:18] = 0.0
    aligned_52d[maturity_6d[:, 3] <= 0.0, 28:34] = 0.0
    aligned_52d[maturity_6d[:, 4] <= 0.0, 34:40] = 0.0
    aligned_52d[maturity_6d[:, 5] <= 0.0, 40:46] = 0.0
    aligned_52d[maturity_6d[:, 2] <= 0.0, 46:52] = 0.0

    v7_signals_58d = np.clip(np.column_stack([aligned_52d, maturity_6d]), -5.0, 5.0).astype(np.float32)
    del aligned_52d, maturity_6d
    gc.collect()

    raw_600 = np.load(CACHE_600H_PATH, allow_pickle=True)
    base_dict = {k: raw_600[k] for k in raw_600.files}
    del raw_600
    base_dict["ts"] = ts_600
    base_dict["v7_signals_58d"] = v7_signals_58d
    tmp_path = CACHE_600H_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **base_dict)
    tmp_path.replace(CACHE_600H_PATH)
    print(f"✅ Гибридный кэш v7+v8 (v7_signals_58d: {v7_signals_58d.shape}) сохранен за {time.time() - t0:.1f}с!")
    return CACHE_600H_PATH


def incremental_append_600h_cache(coin: str = "prl") -> Path:
    """
    Быстрое инкрементальное достраивание rl_guard_lob_cache_600h.npz для новых тиков в хвосте
    prl_labeled.parquet без полного пересчёта 2.45M исторических строк.
    """
    from _tools.swarm_router import SwarmRegimeRouter
    from _tools.train_rl_execution import _compute_v52_signals_matrix, _compute_v70_realism_matrix

    t0 = time.time()
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    pf = pq.ParquetFile(labeled_path)
    n_new = pf.metadata.num_rows

    old_npz = np.load(CACHE_600H_PATH, allow_pickle=True)
    n_old = len(old_npz["bid_p"])
    has_bo = "bo_bid_p" in old_npz.files and len(old_npz["bo_bid_p"]) == n_old
    if n_new <= n_old and "v7_signals_58d" in old_npz.files and len(old_npz["v7_signals_58d"]) == n_new and has_bo:
        print(f"✅ Кэш {CACHE_600H_PATH.name} уже актуален ({n_new:,} тиков, включая BigONE L2).")
        return CACHE_600H_PATH

    overlap = 600
    s_idx = max(0, n_old - overlap)
    n_tail = n_new - s_idx
    print(f"⚡ Инкрементальное обновление {CACHE_600H_PATH.name}: {n_old:,} ➔ {n_new:,} тиков (хвост {n_tail:,} строк)...", flush=True)

    b_p_cols = [f"ST_bid_{i}_price" for i in range(1, 11)]
    b_q_cols = [f"ST_bid_{i}_qty" for i in range(1, 11)]
    a_p_cols = [f"ST_ask_{i}_price" for i in range(1, 11)]
    a_q_cols = [f"ST_ask_{i}_qty" for i in range(1, 11)]
    bo_bp_cols = [f"BO_bid_{i}_price" for i in range(1, 6)]
    bo_bq_cols = [f"BO_bid_{i}_qty" for i in range(1, 6)]
    bo_ap_cols = [f"BO_ask_{i}_price" for i in range(1, 6)]
    bo_aq_cols = [f"BO_ask_{i}_qty" for i in range(1, 6)]
    bo_extra_cols = ["BO_buy_vol", "BO_sell_vol", "BO_sec_since_tick"]
    meta_cols = [
        "timestamp", "ST_spread", "ST_vol_60s_ticks", "ST_ofi_norm_300s",
        "ST_vol_15s_ticks", "ST_ofi_15s", "ST_ofi_60s", "ST_depth_ratio_5",
        "BO_ofi_15s", "BO_ret_15s", "LL_zscore_30s", "BO_sec_since_tick",
        "ST_vol_300s_ticks", "ST_trade_vol", "ST_buy_vol", "ST_sell_vol", "ST_vwap_1s",
        "ST_cvd_1m", "ST_cvd_5m",
    ] + b_p_cols + b_q_cols + a_p_cols + a_q_cols + bo_bp_cols + bo_bq_cols + bo_ap_cols + bo_aq_cols + bo_extra_cols
    schema_cols = set(pf.schema_arrow.names)
    avail_cols = [c for c in list(dict.fromkeys(meta_cols)) if c in schema_cols]

    df_tail = pq.read_table(labeled_path, columns=avail_cols).slice(s_idx, n_tail).to_pandas()
    if "timestamp" not in df_tail.columns:
        df_tail = df_tail.reset_index()
        if "timestamp" not in df_tail.columns and "index" in df_tail.columns:
            df_tail = df_tail.rename(columns={"index": "timestamp"})
    df_tail = df_tail.reset_index(drop=True)
    ts_tail = pd.to_datetime(df_tail["timestamp"], utc=True).dt.as_unit("s").astype("int64").values

    tick = 0.01
    bid_p = np.column_stack([np.nan_to_num(df_tail[c].values, nan=0.0) for c in b_p_cols]).astype(np.float32)
    bid_q = np.column_stack([np.nan_to_num(df_tail[c].values, nan=0.0) for c in b_q_cols]).astype(np.float32)
    ask_p = np.column_stack([np.nan_to_num(df_tail[c].values, nan=0.0) for c in a_p_cols]).astype(np.float32)
    ask_q = np.column_stack([np.nan_to_num(df_tail[c].values, nan=0.0) for c in a_q_cols]).astype(np.float32)
    buy_v = np.nan_to_num(df_tail["ST_buy_vol"].values, nan=0.0).astype(np.float32)
    sell_v = np.nan_to_num(df_tail["ST_sell_vol"].values, nan=0.0).astype(np.float32)

    b1 = bid_p[:, 0]
    y_sweep_12s = np.zeros(n_tail, dtype=np.float32)
    for k in range(1, 13):
        b1_fwd = np.pad(b1[k:], (0, k), mode="edge")
        y_sweep_12s = np.maximum(y_sweep_12s, ((b1 - b1_fwd) >= (tick * 0.95)).astype(np.float32))

    min_drop_60s = np.zeros(n_tail, dtype=np.float32)
    max_rise_60s = np.zeros(n_tail, dtype=np.float32)
    for k in range(1, 61):
        b1_fwd = np.pad(b1[k:], (0, k), mode="edge")
        min_drop_60s = np.maximum(min_drop_60s, (b1 - b1_fwd) / tick)
        max_rise_60s = np.maximum(max_rise_60s, (b1_fwd - b1) / tick)
    y_bleed_60s = ((min_drop_60s >= 1.95) & (max_rise_60s < 0.95)).astype(np.float32)

    min_drop_300s = np.zeros(n_tail, dtype=np.float32)
    for k in range(5, 301, 5):
        b1_fwd = np.pad(b1[k:], (0, k), mode="edge")
        min_drop_300s = np.maximum(min_drop_300s, (b1 - b1_fwd) / tick)
    y_crash_300s = (min_drop_300s >= 4.95).astype(np.float32)

    min_drop_30s = np.zeros(n_tail, dtype=np.float32)
    for k in range(1, 31):
        b1_fwd = np.pad(b1[k:], (0, k), mode="edge")
        min_drop_30s = np.maximum(min_drop_30s, (b1 - b1_fwd) / tick)
    y_exit_30s = (min_drop_30s >= 1.95).astype(np.float32)

    X_all_mm = np.load(BASE_DIR / "data" / "_mmap_cache" / "X_all.npy", mmap_mode="r")
    x_tail = np.array(X_all_mm[s_idx:n_new], dtype=np.float32)

    horizons = [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]
    routers = [SwarmRegimeRouter(coin, H, K, temperature=0.25, use_neural=True, neural_weight=0.15) for H, K in horizons]
    h_mats = []
    for r in routers:
        ens = r.predict_ensemble(x_tail, df_tail, mode="router")
        r._lgb_cache_probs = None
        h_mats.append(_extract_6d_from_ens(ens))

    spread_ticks = np.clip((ask_p[:, 0] - bid_p[:, 0]) / 0.01, 0.0, 50.0)
    b1_q, a1_q = np.maximum(bid_q[:, 0], 0.0), np.maximum(ask_q[:, 0], 0.0)
    b5_q, a5_q = np.sum(np.maximum(bid_q[:, :5], 0.0), axis=1), np.sum(np.maximum(ask_q[:, :5], 0.0), axis=1)
    b10_q, a10_q = np.sum(np.maximum(bid_q, 0.0), axis=1), np.sum(np.maximum(ask_q, 0.0), axis=1)
    vol_300 = np.nan_to_num(df_tail["ST_vol_300s_ticks"].values, nan=1.0)
    ofi_300 = np.nan_to_num(df_tail["ST_ofi_norm_300s"].values, nan=0.0)
    x_lob = np.column_stack([
        spread_ticks / 10.0,
        (b1_q - a1_q) / (b1_q + a1_q + 1e-6),
        (b5_q - a5_q) / (b5_q + a5_q + 1e-6),
        (b10_q - a10_q) / (b10_q + a10_q + 1e-6),
        np.log1p(b1_q) / 10.0, np.log1p(a1_q) / 10.0,
        np.log1p(b5_q) / 12.0, np.log1p(a5_q) / 12.0,
        np.clip(vol_300 / 5.0, 0.0, 5.0), np.clip(ofi_300, -3.0, 3.0),
    ]).astype(np.float32)

    xm_c = np.hstack([h_mats[0], h_mats[1], h_mats[2], x_lob]).astype(np.float32)
    vwap_arr = np.nan_to_num(df_tail["ST_vwap_1s"].values, nan=0.0) if "ST_vwap_1s" in df_tail.columns else (bid_p[:, 0] + ask_p[:, 0]) * 0.5
    chunk_dict = {
        "X_market": xm_c, "X_h60": h_mats[3], "X_h90": h_mats[4], "X_h120": h_mats[5],
        "bid_p": bid_p, "bid_q": bid_q, "ask_p": ask_p, "ask_q": ask_q,
        "buy_vol": buy_v, "sell_vol": sell_v, "vwap": vwap_arr, "tick_size": 0.01,
    }
    xv52_c = _compute_v52_signals_matrix(chunk_dict, coin=coin)[:, :4].astype(np.float32)
    xv70_c = _compute_v70_realism_matrix(chunk_dict, coin=coin)[:, :2].astype(np.float32)
    tail_52d = np.column_stack([xm_c, h_mats[3], h_mats[4], h_mats[5], xv52_c, xv70_c]).astype(np.float32)
    mat_tail = compute_horizon_maturity_6d(ts_tail, coin=coin)
    v7_tail_58d = np.clip(np.column_stack([tail_52d, mat_tail]), -5.0, 5.0).astype(np.float32)

    def _cat(key_name, new_tail_arr):
        return np.concatenate([old_npz[key_name][:s_idx], new_tail_arr], axis=0)

    bo_bp_tail = np.column_stack([np.nan_to_num(df_tail[c].values, nan=0.0) if c in df_tail.columns else np.zeros(n_tail) for c in bo_bp_cols]).astype(np.float32)
    bo_bq_tail = np.column_stack([np.nan_to_num(df_tail[c].values, nan=0.0) if c in df_tail.columns else np.zeros(n_tail) for c in bo_bq_cols]).astype(np.float32)
    bo_ap_tail = np.column_stack([np.nan_to_num(df_tail[c].values, nan=0.0) if c in df_tail.columns else np.zeros(n_tail) for c in bo_ap_cols]).astype(np.float32)
    bo_aq_tail = np.column_stack([np.nan_to_num(df_tail[c].values, nan=0.0) if c in df_tail.columns else np.zeros(n_tail) for c in bo_aq_cols]).astype(np.float32)
    bo_bv_tail = (np.nan_to_num(df_tail["BO_buy_vol"].values, nan=0.0) if "BO_buy_vol" in df_tail.columns else np.zeros(n_tail)).astype(np.float32)
    bo_sv_tail = (np.nan_to_num(df_tail["BO_sell_vol"].values, nan=0.0) if "BO_sell_vol" in df_tail.columns else np.zeros(n_tail)).astype(np.float32)
    bo_sec_tail = (np.nan_to_num(df_tail["BO_sec_since_tick"].values, nan=999.0) if "BO_sec_since_tick" in df_tail.columns else np.full(n_tail, 999.0)).astype(np.float32)

    if has_bo:
        bo_bp_full = _cat("bo_bid_p", bo_bp_tail)
        bo_bq_full = _cat("bo_bid_q", bo_bq_tail)
        bo_ap_full = _cat("bo_ask_p", bo_ap_tail)
        bo_aq_full = _cat("bo_ask_q", bo_aq_tail)
        bo_bv_full = _cat("bo_buy_vol", bo_bv_tail)
        bo_sv_full = _cat("bo_sell_vol", bo_sv_tail)
        bo_sec_full = _cat("bo_sec_since_tick", bo_sec_tail)
    else:
        print("   🌐 Разовая догрузка исторических колонок BigONE L1..L5 + Trades для всего 600h+ массива...", flush=True)
        bo_need = [c for c in (bo_bp_cols + bo_bq_cols + bo_ap_cols + bo_aq_cols + bo_extra_cols) if c in schema_cols]
        tbl_bo = pq.read_table(labeled_path, columns=bo_need)
        bo_bp_full = np.column_stack([np.nan_to_num(tbl_bo.column(c).to_numpy(zero_copy_only=False), nan=0.0) for c in bo_bp_cols]).astype(np.float32)
        bo_bq_full = np.column_stack([np.nan_to_num(tbl_bo.column(c).to_numpy(zero_copy_only=False), nan=0.0) for c in bo_bq_cols]).astype(np.float32)
        bo_ap_full = np.column_stack([np.nan_to_num(tbl_bo.column(c).to_numpy(zero_copy_only=False), nan=0.0) for c in bo_ap_cols]).astype(np.float32)
        bo_aq_full = np.column_stack([np.nan_to_num(tbl_bo.column(c).to_numpy(zero_copy_only=False), nan=0.0) for c in bo_aq_cols]).astype(np.float32)
        bo_bv_full = np.nan_to_num(tbl_bo.column("BO_buy_vol").to_numpy(zero_copy_only=False), nan=0.0).astype(np.float32)
        bo_sv_full = np.nan_to_num(tbl_bo.column("BO_sell_vol").to_numpy(zero_copy_only=False), nan=0.0).astype(np.float32)
        bo_sec_full = np.nan_to_num(tbl_bo.column("BO_sec_since_tick").to_numpy(zero_copy_only=False), nan=999.0).astype(np.float32)
        del tbl_bo

    out_dict = {
        "ts": _cat("ts", ts_tail),
        "bid_p": _cat("bid_p", bid_p),
        "bid_q": _cat("bid_q", bid_q),
        "ask_p": _cat("ask_p", ask_p),
        "ask_q": _cat("ask_q", ask_q),
        "buy_vol": _cat("buy_vol", buy_v),
        "sell_vol": _cat("sell_vol", sell_v),
        "bo_bid_p": bo_bp_full,
        "bo_bid_q": bo_bq_full,
        "bo_ask_p": bo_ap_full,
        "bo_ask_q": bo_aq_full,
        "bo_buy_vol": bo_bv_full,
        "bo_sell_vol": bo_sv_full,
        "bo_sec_since_tick": bo_sec_full,
        "y_sweep_12s": _cat("y_sweep_12s", y_sweep_12s),
        "y_bleed_60s": _cat("y_bleed_60s", y_bleed_60s),
        "y_crash_300s": _cat("y_crash_300s", y_crash_300s),
        "y_exit_30s": _cat("y_exit_30s", y_exit_30s),
        "v7_signals_58d": _cat("v7_signals_58d", v7_tail_58d),
        "tick_size": np.float32(0.01),
    }
    del old_npz
    gc.collect()
    tmp_path = CACHE_600H_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **out_dict)
    tmp_path.replace(CACHE_600H_PATH)
    print(f"🎉 Инкрементальный кэш {CACHE_600H_PATH.name} ({n_new:,} тиков) сохранен за {time.time() - t0:.1f}с!", flush=True)
    return CACHE_600H_PATH


if __name__ == "__main__":
    incremental_append_600h_cache(coin="prl")
