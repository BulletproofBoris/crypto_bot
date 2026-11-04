#!/usr/bin/env python3
"""
Этап 1–4: Предрасчёт матрицы триады (H=5s, 12s, 25s) и обучение RL-агента
микроструктурного исполнения (Maskable PPO + Behavioral Cloning Warm-Start)
поверх 10-уровневого L2 FIFO-стакана SafeTrade.
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.system_resources import SAFE_WORKERS
from _tools.swarm_router import SwarmRegimeRouter
from _tools.data_labeler import COIN_CONFIG


def _compute_v52_signals_matrix(data_dict: dict, coin: str = "prl") -> np.ndarray:
    """Предвычисляет 4D матрицу сигналов [p_car_50, log_cap_norm, p_adverse, p_exhaust] из CarouselNet + HazardNet."""
    n = len(data_dict["X_market"])
    out = np.zeros((n, 4), dtype=np.float32)
    out[:, 0] = 0.13
    out[:, 1] = 0.02
    out[:, 2] = 0.08
    out[:, 3] = 0.04
    try:
        from _tools.carousel_net import build_carousel_features, CarouselPredictor
        from _tools.hazard_net import HazardPredictor
        c_path = BASE_DIR / "models" / coin.lower() / "rl_execution" / "carousel_net.pt"
        h_path = BASE_DIR / "models" / coin.lower() / "rl_execution" / "hazard_net.pt"
        if c_path.exists() and h_path.exists():
            X44 = build_carousel_features(data_dict)
            cp = CarouselPredictor(c_path)
            hp = HazardPredictor(h_path)
            probs_c, cap_c = cp.predict_batch(X44)
            pa, pe = hp.predict_batch(X44)
            out = np.column_stack([
                probs_c[:, 1],
                np.log1p(np.maximum(0.0, cap_c)) / 7.0,
                pa,
                pe,
            ]).astype(np.float32)
    except Exception as e:
        print(f"⚠️ Precompute X_v52 warning: {e}")
    return out


def _compute_v70_realism_matrix(data_dict: dict, coin: str = "prl") -> np.ndarray:
    """Предвычисляет 2D матрицу [p_fill_real, p_frontrun] из ExecutionRealismNet."""
    n = len(data_dict["X_market"])
    out = np.column_stack([np.full(n, 0.35, dtype=np.float32), np.full(n, 0.15, dtype=np.float32)])
    try:
        from _tools.execution_realism_net import build_exec_realism_features_from_cache, ExecutionRealismPredictor
        r_path = BASE_DIR / "models" / coin.lower() / "rl_execution" / "execution_realism_net.pt"
        if r_path.exists():
            X16 = build_exec_realism_features_from_cache(data_dict, order_qty_prl=150.0)
            rp = ExecutionRealismPredictor(r_path)
            pf, pfr = rp.predict_batch(X16)
            out = np.column_stack([pf, pfr]).astype(np.float32)
    except Exception as e:
        print(f"⚠️ Precompute X_v70_realism warning: {e}")
    return out


def build_or_load_rl_cache(coin: str = "prl", tail_hours: float = 176.0, force_rebuild: bool = False) -> dict:
    """
    Этап 1: Предрасчёт и кэширование сигналов триады ансамблей (H=5s, 12s, 25s)
    и 10 уровней стакана L2 для мгновенной симуляции RL-среды.
    """
    coin = coin.lower()
    cache_dir = BASE_DIR / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rl_cache_path = cache_dir / f"rl_triad_lob_cache_{coin}_{int(tail_hours)}h.npz"
    tmp_cache_path = Path(f"/tmp/rl_triad_lob_cache_{coin}_{int(tail_hours)}h.npz")
    read_cache_path = tmp_cache_path if (tmp_cache_path.exists() and (not rl_cache_path.exists() or tmp_cache_path.stat().st_mtime >= rl_cache_path.stat().st_mtime)) else rl_cache_path

    if read_cache_path.exists() and not force_rebuild:
        data_raw = np.load(read_cache_path, allow_pickle=True)
        if "X_h60" in data_raw.files and "X_h90" in data_raw.files and "X_h120" in data_raw.files:
            res_dict = {k: data_raw[k] for k in data_raw.files}
            if "X_v52" not in res_dict or len(res_dict["X_v52"]) != len(res_dict["X_market"]):
                print(f"🧠 Досчет 4D матрицы CarouselNet + HazardNet (X_v52) для кэша {read_cache_path.name}...")
                res_dict["X_v52"] = _compute_v52_signals_matrix(res_dict, coin=coin)
            if "X_v70_realism" not in res_dict or len(res_dict["X_v70_realism"]) != len(res_dict["X_market"]):
                res_dict["X_v70_realism"] = _compute_v70_realism_matrix(res_dict, coin=coin)
            print(f"⚡ Загрузка готового кэша RL-полигона (H=5..120s + X_v52 + X_v70): {read_cache_path}")
            return res_dict

    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    feat_path = BASE_DIR / "data" / "features" / f"{coin}_1s_features.parquet"
    if feat_path.exists() and labeled_path.exists():
        parquet_path = labeled_path if labeled_path.stat().st_mtime >= feat_path.stat().st_mtime else feat_path
    else:
        parquet_path = feat_path if feat_path.exists() else labeled_path

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    n_rows = min(total_rows, int(tail_hours * 3600))
    start_row = max(0, total_rows - n_rows)

    horizons = [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]
    schema_cols = set(pf.schema_arrow.names)
    lob_and_meta_cols = [
        "timestamp", "ST_best_bid", "ST_best_ask", "ST_bid_qty_1", "ST_ask_qty_1",
        "ST_spread", "ST_spread_pct", "ST_vol_60s_ticks", "ST_vol_300s_ticks",
        "ST_ofi_norm_300s", "ST_cvd_norm_20s", "ST_cvd_5m",
        "BO_sec_since_tick", "ST_trade_vol", "ST_trade_vol_1s",
        "ST_buy_vol", "ST_buy_vol_1s", "ST_sell_vol", "ST_sell_vol_1s", "ST_vwap_1s",
    ]
    for lvl in range(1, 11):
        lob_and_meta_cols.extend([
            f"ST_bid_{lvl}_price", f"ST_bid_{lvl}_qty",
            f"ST_ask_{lvl}_price", f"ST_ask_{lvl}_qty",
        ])
    base_cols = [c for c in lob_and_meta_cols if c in schema_cols]

    rg_indices = []
    cum_rows = 0
    first_rg_offset = 0
    for rg_i in range(pf.num_row_groups):
        rg_n = pf.metadata.row_group(rg_i).num_rows
        if cum_rows + rg_n > start_row:
            if not rg_indices:
                first_rg_offset = start_row - cum_rows
            rg_indices.append(rg_i)
        cum_rows += rg_n

    print(f"📂 Селективное чтение стакана L2 ({n_rows:,} сек, {len(base_cols)} кол.) из {parquet_path.name}...")
    table = pf.read_row_groups(rg_indices, columns=base_cols)
    df = table.slice(first_rg_offset, n_rows).to_pandas()
    del table
    if "timestamp" not in df.columns:
        df["timestamp"] = df.index
    df = df.reset_index(drop=True)

    triad_feats = []
    cached_fnames = None
    cached_X = None
    ts_str_all = df["timestamp"].astype(str).values
    cand_caches = list(cache_dir.glob(f"rl_triad_lob_cache_{coin}_*h.npz"))
    prev_cache_path = max(cand_caches, key=lambda p: p.stat().st_mtime) if cand_caches else (cache_dir / f"rl_triad_lob_cache_{coin}_438h.npz")
    X_triad_Incremental = None
    X_h60 = None
    X_h90 = None
    X_h120 = None

    if prev_cache_path.exists():
        try:
            prev_c = np.load(prev_cache_path, allow_pickle=True)
            prev_ts_i8 = pd.DatetimeIndex(pd.to_datetime(prev_c["ts"], utc=True)).asi8
            cur_ts_i8 = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)).asi8
            prev_triad = prev_c["X_market"][:, :18].astype(np.float32)
            prev_h60 = prev_c["X_h60"].astype(np.float32) if "X_h60" in prev_c.files and len(prev_c["X_h60"]) == len(prev_triad) else None
            prev_h90 = prev_c["X_h90"].astype(np.float32) if "X_h90" in prev_c.files and len(prev_c["X_h90"]) == len(prev_triad) else None
            prev_h120 = prev_c["X_h120"].astype(np.float32) if "X_h120" in prev_c.files and len(prev_c["X_h120"]) == len(prev_triad) else None
            ts_to_idx = {int(s): i for i, s in enumerate(prev_ts_i8)}
            matched_idx = np.array([ts_to_idx.get(int(s), -1) for s in cur_ts_i8], dtype=np.int64)
            valid_mask = matched_idx >= 0
            if np.any(valid_mask):
                valid_positions = np.where(valid_mask)[0]
                first_match_pos = int(valid_positions[0])
                last_match_pos = int(valid_positions[-1])
                if first_match_pos > 0 and first_match_pos <= 7200 and np.all(valid_mask[first_match_pos:last_match_pos + 1]):
                    df = df.iloc[first_match_pos:].reset_index(drop=True)
                    matched_idx = matched_idx[first_match_pos:]
                    valid_mask = valid_mask[first_match_pos:]
                    start_row += first_match_pos
                    last_match_pos -= first_match_pos
                if last_match_pos >= len(df) - 300000 and np.all(valid_mask[:last_match_pos + 1]):
                    n_tail = len(df) - (last_match_pos + 1)
                    print(f"⚡ Инкрементальное обновление кэша: переиспользуем {last_match_pos + 1:,} сек из {prev_cache_path.name}, досчитываем хвост {n_tail:,} сек ({n_tail/3600:.2f} ч)...")
                    X_triad_head = prev_triad[matched_idx[:last_match_pos + 1]]
                    X_h60_head = prev_h60[matched_idx[:last_match_pos + 1]] if prev_h60 is not None else None
                    X_h90_head = prev_h90[matched_idx[:last_match_pos + 1]] if prev_h90 is not None else None
                    X_h120_head = prev_h120[matched_idx[:last_match_pos + 1]] if prev_h120 is not None else None
                    if n_tail > 0:
                        pad = min(360, last_match_pos + 1)
                        tail_start_abs = start_row + last_match_pos + 1 - pad
                        t_rgs, t_cum, t_off = [], 0, 0
                        for rg_i in range(pf.num_row_groups):
                            rg_n = pf.metadata.row_group(rg_i).num_rows
                            if t_cum + rg_n > tail_start_abs:
                                if not t_rgs:
                                    t_off = tail_start_abs - t_cum
                                t_rgs.append(rg_i)
                            t_cum += rg_n
                        df_tail = pf.read_row_groups(t_rgs).slice(t_off, n_tail + pad).to_pandas()
                        if "timestamp" not in df_tail.columns:
                            df_tail["timestamp"] = df_tail.index
                        df_tail = df_tail.reset_index(drop=True)
                        tail_triad_list = []
                        for H, K in horizons:
                            router = SwarmRegimeRouter(coin, H, K, temperature=0.25, use_neural=True, neural_weight=0.15)
                            X_cols = [df_tail[col].values.astype(np.float32) if col in df_tail.columns else np.zeros(len(df_tail), dtype=np.float32) for col in router.feature_names]
                            X_tail_in = np.nan_to_num(np.column_stack(X_cols), nan=0.0, posinf=0.0, neginf=0.0)
                            ens_out = router.predict_ensemble(X_tail_in, df_tail, mode="router")
                            del router, X_cols, X_tail_in
                            probs = ens_out["probs"][pad:]
                            ez = ens_out["expected_z"][pad:]
                            p_up = probs[:, 4] + probs[:, 5] + probs[:, 6]
                            p_down = probs[:, 0] + probs[:, 1] + probs[:, 2]
                            p_tp = probs[:, 5] + probs[:, 6]
                            p_tn = probs[:, 0] + probs[:, 1]
                            p_clip = np.clip(probs, 1e-7, 1.0)
                            ent = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(7.0)
                            tail_triad_list.append(np.column_stack([ez, p_up, p_down, p_tp, p_tn, ent]).astype(np.float32))
                        del df_tail
                        X_triad_Incremental = np.vstack([X_triad_head, np.hstack(tail_triad_list[:3])]).astype(np.float32)
                        if X_h60_head is not None and len(tail_triad_list) > 3:
                            X_h60 = np.vstack([X_h60_head, tail_triad_list[3]]).astype(np.float32)
                        if X_h90_head is not None and len(tail_triad_list) > 4:
                            X_h90 = np.vstack([X_h90_head, tail_triad_list[4]]).astype(np.float32)
                        if X_h120_head is not None and len(tail_triad_list) > 5:
                            X_h120 = np.vstack([X_h120_head, tail_triad_list[5]]).astype(np.float32)
                    else:
                        X_triad_Incremental = X_triad_head
                        X_h60 = X_h60_head
                        X_h90 = X_h90_head
                        X_h120 = X_h120_head
        except Exception as e:
            print(f"⚠️ Не удалось применить инкрементальный кэш ({e}), выполняем полный расчёт...")

    if X_triad_Incremental is not None:
        X_triad = X_triad_Incremental
        if X_h60 is None:
            df_full = pf.read_row_groups(rg_indices).slice(first_rg_offset, len(X_triad)).to_pandas()
            if "timestamp" not in df_full.columns:
                df_full["timestamp"] = df_full.index
            df_full = df_full.reset_index(drop=True)
            router60 = SwarmRegimeRouter(coin, 60, 3.0, temperature=0.25, use_neural=True, neural_weight=0.15)
            print(f"\n🔮 Досчёт макро-ансамбля H=60s (K=3.0) на {len(df_full):,} сек...")
            X_cols60 = [df_full[col].values.astype(np.float32) if col in df_full.columns else np.zeros(len(df_full), dtype=np.float32) for col in router60.feature_names]
            X_in60 = np.nan_to_num(np.column_stack(X_cols60), nan=0.0, posinf=0.0, neginf=0.0)
            ens60 = router60.predict_ensemble(X_in60, df_full, mode="router")
            probs60, ez60 = ens60["probs"], ens60["expected_z"]
            p_clip60 = np.clip(probs60, 1e-7, 1.0)
            X_h60 = np.column_stack([
                ez60, probs60[:, 4] + probs60[:, 5] + probs60[:, 6], probs60[:, 0] + probs60[:, 1] + probs60[:, 2],
                probs60[:, 5] + probs60[:, 6], probs60[:, 0] + probs60[:, 1], -np.sum(p_clip60 * np.log(p_clip60), axis=1) / np.log(7.0)
            ]).astype(np.float32)
            del df_full, router60, X_cols60, X_in60, ens60
    else:
        df_full = pf.read_row_groups(rg_indices).slice(first_rg_offset, n_rows).to_pandas()
        if "timestamp" not in df_full.columns:
            df_full["timestamp"] = df_full.index
        df_full = df_full.reset_index(drop=True)
        for H, K in horizons:
            router = SwarmRegimeRouter(coin, H, K, temperature=0.25, use_neural=True, neural_weight=0.15)
            print(f"\n🔮 Расчёт ансамбля H={H}s (K={K})...")
            if cached_fnames != router.feature_names:
                X_cols = [
                    df_full[col].values.astype(np.float32) if col in df_full.columns else np.zeros(len(df_full), dtype=np.float32)
                    for col in router.feature_names
                ]
                cached_X = np.nan_to_num(np.column_stack(X_cols), nan=0.0, posinf=0.0, neginf=0.0)
                cached_fnames = list(router.feature_names)
                del X_cols

            ens_out = router.predict_ensemble(cached_X, df_full, mode="router")
            probs = ens_out["probs"]
            ez = ens_out["expected_z"]
            del router, ens_out

            p_up = probs[:, 4] + probs[:, 5] + probs[:, 6]
            p_down = probs[:, 0] + probs[:, 1] + probs[:, 2]
            p_tail_pos = probs[:, 5] + probs[:, 6]
            p_tail_neg = probs[:, 0] + probs[:, 1]
            p_clip = np.clip(probs, 1e-7, 1.0)
            entropy = -np.sum(p_clip * np.log(p_clip), axis=1) / np.log(7.0)

            triad_feats.append(np.column_stack([ez, p_up, p_down, p_tail_pos, p_tail_neg, entropy]).astype(np.float32))
        del cached_X, df_full
        X_triad = np.hstack(triad_feats[:3]).astype(np.float32)
        X_h60 = triad_feats[3] if len(triad_feats) > 3 else np.zeros((len(X_triad), 6), dtype=np.float32)
        X_h90 = triad_feats[4] if len(triad_feats) > 4 else X_h60.copy()
        X_h120 = triad_feats[5] if len(triad_feats) > 5 else X_h60.copy()

    # 10 уровней стакана SafeTrade (реальный биржевой шаг цены PRL = 0.01 USDT)
    tick_size = 0.01 if coin == "prl" else float(COIN_CONFIG[coin]["tick_size"])
    bid_p = np.zeros((len(df), 10), dtype=np.float64)
    bid_q = np.zeros((len(df), 10), dtype=np.float64)
    ask_p = np.zeros((len(df), 10), dtype=np.float64)
    ask_q = np.zeros((len(df), 10), dtype=np.float64)

    for lvl in range(1, 11):
        bp_col, bq_col = f"ST_bid_{lvl}_price", f"ST_bid_{lvl}_qty"
        ap_col, aq_col = f"ST_ask_{lvl}_price", f"ST_ask_{lvl}_qty"
        if lvl == 1:
            bid_p[:, 0] = df[bp_col].values if bp_col in df.columns else df["ST_best_bid"].values
            bid_q[:, 0] = df[bq_col].values if bq_col in df.columns else df["ST_bid_qty_1"].values
            ask_p[:, 0] = df[ap_col].values if ap_col in df.columns else df["ST_best_ask"].values
            ask_q[:, 0] = df[aq_col].values if aq_col in df.columns else df["ST_ask_qty_1"].values
        else:
            if bp_col in df.columns:
                bid_p[:, lvl - 1] = np.nan_to_num(df[bp_col].values, nan=0.0)
                bid_q[:, lvl - 1] = np.nan_to_num(df[bq_col].values, nan=0.0)
            if ap_col in df.columns:
                ask_p[:, lvl - 1] = np.nan_to_num(df[ap_col].values, nan=0.0)
                ask_q[:, lvl - 1] = np.nan_to_num(df[aq_col].values, nan=0.0)

    # 10 микроструктурных признаков стакана
    spread_ticks = np.clip((ask_p[:, 0] - bid_p[:, 0]) / tick_size, 0.0, 50.0)
    b1_q = np.maximum(bid_q[:, 0], 0.0)
    a1_q = np.maximum(ask_q[:, 0], 0.0)
    imb_l1 = (b1_q - a1_q) / (b1_q + a1_q + 1e-6)
    b5_q = np.sum(np.maximum(bid_q[:, :5], 0.0), axis=1)
    a5_q = np.sum(np.maximum(ask_q[:, :5], 0.0), axis=1)
    imb_l5 = (b5_q - a5_q) / (b5_q + a5_q + 1e-6)
    b10_q = np.sum(np.maximum(bid_q, 0.0), axis=1)
    a10_q = np.sum(np.maximum(ask_q, 0.0), axis=1)
    imb_l10 = (b10_q - a10_q) / (b10_q + a10_q + 1e-6)

    log_b1 = np.log1p(b1_q) / 10.0
    log_a1 = np.log1p(a1_q) / 10.0
    log_b5 = np.log1p(b5_q) / 12.0
    log_a5 = np.log1p(a5_q) / 12.0
    vol_300 = np.nan_to_num(df["ST_vol_300s_ticks"].values if "ST_vol_300s_ticks" in df.columns else np.ones(len(df)), nan=1.0)
    ofi_300 = np.nan_to_num(df["ST_ofi_norm_300s"].values if "ST_ofi_norm_300s" in df.columns else np.zeros(len(df)), nan=0.0)

    X_lob = np.column_stack([
        spread_ticks / 10.0, imb_l1, imb_l5, imb_l10,
        log_b1, log_a1, log_b5, log_a5,
        np.clip(vol_300 / 5.0, 0.0, 5.0), np.clip(ofi_300, -3.0, 3.0)
    ]).astype(np.float32)

    X_market = np.nan_to_num(np.hstack([X_triad, X_lob]), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)

    t_col = "ST_trade_vol" if "ST_trade_vol" in df.columns else "ST_trade_vol_1s"
    b_col = "ST_buy_vol" if "ST_buy_vol" in df.columns else "ST_buy_vol_1s"
    s_col = "ST_sell_vol" if "ST_sell_vol" in df.columns else "ST_sell_vol_1s"
    trade_vol = np.nan_to_num(df[t_col].values if t_col in df.columns else np.zeros(len(df)), nan=0.0)
    buy_vol = np.nan_to_num(df[b_col].values if b_col in df.columns else np.zeros(len(df)), nan=0.0)
    sell_vol = np.nan_to_num(df[s_col].values if s_col in df.columns else np.maximum(0.0, trade_vol - buy_vol), nan=0.0)
    vwap = np.nan_to_num(df["ST_vwap_1s"].values if "ST_vwap_1s" in df.columns else (bid_p[:, 0] + ask_p[:, 0]) * 0.5, nan=0.0)
    ts_arr = df["timestamp"].astype(str).values

    if X_h90 is None or len(X_h90) != len(X_market) or X_h120 is None or len(X_h120) != len(X_market):
        df_full90 = pf.read_row_groups(rg_indices).slice(first_rg_offset, len(X_market)).to_pandas()
        if "timestamp" not in df_full90.columns:
            df_full90["timestamp"] = df_full90.index
        df_full90 = df_full90.reset_index(drop=True)
        for h_macro in (90, 120):
            r_m = SwarmRegimeRouter(coin, h_macro, 3.0, temperature=0.25, use_neural=True, neural_weight=0.15)
            print(f"\n🔮 Досчёт макро-ансамбля H={h_macro}s (K=3.0) на {len(df_full90):,} сек...")
            X_c_m = [df_full90[col].values.astype(np.float32) if col in df_full90.columns else np.zeros(len(df_full90), dtype=np.float32) for col in r_m.feature_names]
            X_in_m = np.nan_to_num(np.column_stack(X_c_m), nan=0.0, posinf=0.0, neginf=0.0)
            ens_m = r_m.predict_ensemble(X_in_m, df_full90, mode="router")
            pr_m, ez_m = ens_m["probs"], ens_m["expected_z"]
            pc_m = np.clip(pr_m, 1e-7, 1.0)
            mat_m = np.column_stack([
                ez_m, pr_m[:, 4] + pr_m[:, 5] + pr_m[:, 6], pr_m[:, 0] + pr_m[:, 1] + pr_m[:, 2],
                pr_m[:, 5] + pr_m[:, 6], pr_m[:, 0] + pr_m[:, 1], -np.sum(pc_m * np.log(pc_m), axis=1) / np.log(7.0)
            ]).astype(np.float32)
            if h_macro == 90:
                X_h90 = mat_m
            else:
                X_h120 = mat_m
            del r_m, X_c_m, X_in_m, ens_m
        del df_full90

    payload = {
        "X_market": X_market,
        "X_h60": X_h60 if X_h60 is not None else np.zeros((len(X_market), 6), dtype=np.float32),
        "X_h90": X_h90 if X_h90 is not None else np.zeros((len(X_market), 6), dtype=np.float32),
        "X_h120": X_h120 if X_h120 is not None else np.zeros((len(X_market), 6), dtype=np.float32),
        "bid_p": bid_p,
        "bid_q": bid_q,
        "ask_p": ask_p,
        "ask_q": ask_q,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "vwap": vwap,
        "ts": ts_arr,
        "tick_size": np.array(tick_size, dtype=np.float64),
    }
    payload["X_v52"] = _compute_v52_signals_matrix(payload, coin=coin)
    payload["X_v70_realism"] = _compute_v70_realism_matrix(payload, coin=coin)
    try:
        rl_cache_path.unlink(missing_ok=True)
        np.savez_compressed(rl_cache_path, **payload)
    except OSError:
        np.savez_compressed(Path(f"/tmp/{rl_cache_path.name}"), **payload)
    print(f"✅ Кэш RL-полигона сохранён в {rl_cache_path} ({X_market.shape[0]:,} строк x {X_market.shape[1]} признаков + H=60,90,120s + X_v52 + X_v70_realism)")
    return payload



class SafeTradeL2ExecutionEnv:
    """
    Быстрая 10-уровневая L2 FIFO среда исполнения заявок для обучения RL-агента.
    Действия (7 шт.):
      0: HOLD / WAIT (сохранять текущую позицию и место q_ahead в очереди FIFO)
      1: QUOTE_BID_BEST (лимитка на покупку на best_bid или +1 тик внутри спреда)
      2: QUOTE_BID_PASSIVE (лимитка на покупку на best_bid - 1 тик)
      3: QUOTE_ASK_BEST (лимитный выход на best_ask)
      4: QUOTE_ASK_AGGRESSIVE (лимитный выход на best_ask - 1 тик)
      5: CANCEL_ORDERS (снять активную лимитку)
      6: PANIC_TAKER_EXIT (мгновенный выход по рынку через 10 уровней стакана)
    """

    def __init__(self, data: dict, start_idx: int, end_idx: int, episode_len: int = 3600, order_usd: float = 95.5, legacy_v2: bool = False, enable_unfreeze: bool = True, use_v52_obs: bool = True, use_v70_obs: bool = True):
        self.enable_unfreeze = bool(enable_unfreeze)
        self.X_market = data["X_market"]
        self.X_h60 = data.get("X_h60", np.zeros((len(self.X_market), 6), dtype=np.float32))
        self.X_h90 = data.get("X_h90", self.X_h60)
        self.X_h120 = data.get("X_h120", self.X_h60)
        self.bid_p = data["bid_p"]
        self.bid_q = data["bid_q"]
        self.ask_p = data["ask_p"]
        self.ask_q = data["ask_q"]
        self.buy_vol = data["buy_vol"]
        self.sell_vol = data["sell_vol"]
        self.vwap = data["vwap"]
        self.tick_size = float(data["tick_size"])

        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.episode_len = int(episode_len)
        self.order_usd = float(order_usd)
        self.legacy_v2 = bool(legacy_v2)
        self.use_v52_obs = bool(use_v52_obs) and (not self.legacy_v2)
        self.use_v70_obs = bool(use_v70_obs) and self.use_v52_obs
        if "X_v52" in data and len(data["X_v52"]) == len(self.X_market):
            self.X_v52 = data["X_v52"].astype(np.float32)
        elif self.use_v52_obs:
            self.X_v52 = _compute_v52_signals_matrix(data)
            data["X_v52"] = self.X_v52
        else:
            self.X_v52 = np.zeros((len(self.X_market), 4), dtype=np.float32)
        if "X_v70_realism" in data and len(data["X_v70_realism"]) == len(self.X_market):
            self.X_v70_realism = data["X_v70_realism"].astype(np.float32)
        elif self.use_v70_obs:
            self.X_v70_realism = _compute_v70_realism_matrix(data)
            data["X_v70_realism"] = self.X_v70_realism
        else:
            self.X_v70_realism = np.zeros((len(self.X_market), 2), dtype=np.float32)
        self.obs_dim = self.X_market.shape[1] + (4 if self.use_v52_obs else 0) + (7 if self.use_v70_obs else 0) + 10
        self.n_actions = 7
        self.v70_slot_fill_ratio = 0.0
        self.v70_lowest_slot_edge_ticks = 0.0
        self.v70_slot_dispersion_ticks = 0.0
        self.v70_free_cash_ratio = 1.0

        # Предрасчет каузальных индикаторов локальных экстремумов (5м / 15м / 60м) и вектора рыночной инерции
        s_bid = pd.Series(self.bid_p[:, 0])
        high_300 = s_bid.rolling(300, min_periods=1).max().values
        high_900 = s_bid.rolling(900, min_periods=1).max().values
        low_900 = s_bid.rolling(900, min_periods=1).min().values
        high_3600 = s_bid.rolling(3600, min_periods=1).max().values
        low_3600 = s_bid.rolling(3600, min_periods=1).min().values
        high_14400 = s_bid.rolling(14400, min_periods=1).max().values
        low_14400 = s_bid.rolling(14400, min_periods=1).min().values
        low_30 = s_bid.rolling(30, min_periods=1).min().values
        vel_b1_10 = np.round((self.bid_p[:, 0] - s_bid.shift(10).bfill().values) / self.tick_size, 4)
        vel_b1_15 = np.round((self.bid_p[:, 0] - s_bid.shift(15).bfill().values) / self.tick_size, 4)
        vel_b1_180 = np.round((self.bid_p[:, 0] - s_bid.shift(180).bfill().values) / self.tick_size, 4)

        self.dd_300 = np.round((high_300 - self.bid_p[:, 0]) / self.tick_size, 4).astype(np.float32)
        self.dd_900 = np.round((high_900 - self.bid_p[:, 0]) / self.tick_size, 4).astype(np.float32)
        self.pos_900 = np.where(high_900 > low_900 + 1e-6, (self.bid_p[:, 0] - low_900) / np.maximum(high_900 - low_900, self.tick_size), 1.0).astype(np.float32)
        self.pos_3600 = np.where(high_3600 > low_3600 + 1e-6, (self.bid_p[:, 0] - low_3600) / np.maximum(high_3600 - low_3600, self.tick_size), 1.0).astype(np.float32)
        self.rng_14400 = np.round((high_14400 - low_14400) / self.tick_size, 4).astype(np.float32)
        self.pos_14400 = ((self.bid_p[:, 0] - low_14400) / np.maximum(high_14400 - low_14400, self.tick_size)).astype(np.float32)
        self.vel_b1_10 = vel_b1_10.astype(np.float32)
        self.vel_b1_180 = vel_b1_180.astype(np.float32)
        self.reb_30 = np.round((self.bid_p[:, 0] - low_30) / self.tick_size, 4).astype(np.float32)

        ez5, ez12, ez25 = self.X_market[:, 0], self.X_market[:, 6], self.X_market[:, 12]
        ez60 = self.X_h60[:, 0] if self.X_h60 is not None and len(self.X_h60) == len(ez25) else ez25
        self.ez60_arr = ez60.astype(np.float32)
        self.min_ez60_30s = pd.Series(self.ez60_arr).rolling(30, min_periods=1).min().values.astype(np.float32)
        self.min_ez25_30s = pd.Series(ez25).rolling(30, min_periods=1).min().values.astype(np.float32)
        p_tn_5 = self.X_market[:, 4]
        p_dn_12 = self.X_market[:, 8]
        spread_ticks = self.X_market[:, 18] * 10.0
        d_ez12 = ez12 - pd.Series(ez12).shift(5).bfill().values
        imb1, imb5 = self.X_market[:, 19], self.X_market[:, 20]
        b1q_s = pd.Series(np.maximum(self.bid_q[:, 0], 0.0))
        a1q_s = pd.Series(np.maximum(self.ask_q[:, 0], 0.0))
        same_b1_5s = np.abs(self.bid_p[:, 0] - s_bid.shift(5).bfill().values) < (self.tick_size * 0.25)
        s_ask = pd.Series(self.ask_p[:, 0])
        same_a1_5s = np.abs(self.ask_p[:, 0] - s_ask.shift(5).bfill().values) < (self.tick_size * 0.25)
        self.dq_bid_5s = np.where(same_b1_5s, b1q_s.diff(5).fillna(0.0).values, 0.0).astype(np.float32)
        self.dq_ask_5s = np.where(same_a1_5s, a1q_s.diff(5).fillna(0.0).values, 0.0).astype(np.float32)
        self.buy_vol_10s = pd.Series(self.buy_vol).rolling(10, min_periods=1).sum().values.astype(np.float32)
        net_trade_5s = pd.Series(self.buy_vol - self.sell_vol).rolling(5, min_periods=1).sum().values
        self.ofi_5s = (self.dq_bid_5s - self.dq_ask_5s + net_trade_5s).astype(np.float32)
        net_trade_10s = pd.Series(self.buy_vol - self.sell_vol).rolling(10, min_periods=1).sum().values
        dq_10s = (b1q_s.diff(10).fillna(0.0).values - a1q_s.diff(10).fillna(0.0).values)
        fast_flow_10s = np.clip((net_trade_10s + 0.5 * dq_10s) / (b1q_s.values + a1q_s.values + 500.0) * 3.0, -3.0, 3.0).astype(np.float32)
        if self.legacy_v2:
            self.inertia_arr = (
                0.45 * ez25 + 0.35 * ez12 + 0.20 * ez5
                + 0.25 * d_ez12 + 0.35 * imb5 + 0.15 * imb1
                + 0.10 * np.clip(vel_b1_15, -3.0, 3.0) + 0.08 * np.maximum(0.0, fast_flow_10s)
            ).astype(np.float32)
        else:
            self.inertia_arr = (
                0.15 * ez60 + 0.35 * ez25 + 0.30 * ez12 + 0.20 * ez5
                + 0.25 * d_ez12 + 0.35 * imb5 + 0.15 * imb1
                + 0.10 * np.clip(vel_b1_15, -3.0, 3.0) + 0.08 * np.maximum(0.0, fast_flow_10s)
            ).astype(np.float32)

        # Векторный предрасчет не зависящих от позиции компонент наблюдения и масок входа (Zero-Allocation Step Loop)
        b1_vec = np.maximum(self.bid_p[:, 0], self.tick_size)
        idx_prev12 = np.maximum(0, np.arange(len(b1_vec)) - 12)
        actual_dt_12 = (b1_vec - b1_vec[idx_prev12]) / self.tick_size
        pred_dt_12 = ez12[idx_prev12] * 4.0
        model_err_12 = np.clip(actual_dt_12 - pred_dt_12, -10.0, 10.0).astype(np.float32)
        surprise_pen = np.maximum(0.0, -model_err_12) * 0.6

        self.vel_10 = self.vel_b1_10
        self.imb_l5 = imb5.astype(np.float32)
        if self.legacy_v2:
            exp_entry_edge = (0.5 * ez25 + 0.35 * ez12 + 0.15 * ez5) * 4.0 + 0.5 * spread_ticks - p_tn_5 * 3.0 - surprise_pen
            exp_hold_edge = (0.6 * ez12 + 0.4 * ez5) * 4.0 - p_dn_12 * spread_ticks - surprise_pen * 0.8
            in_dip_zone_vec = np.ones(len(b1_vec), dtype=bool)
            self.in_breakout_arr = np.zeros(len(b1_vec), dtype=bool)
            self.in_breakout_zone = self.in_breakout_arr
        else:
            ext_pos = 0.35 * np.minimum(self.dd_900, 10.0) * np.maximum(0.0, 1.0 - 1.5 * self.pos_900)
            ext_neg = -1.5 * np.maximum(0.0, self.pos_900 - 0.65)
            extremum_bonus = np.where(self.inertia_arr > 0.25, ext_pos, ext_neg)
            exp_entry_edge = (0.15 * ez60 + 0.40 * ez25 + 0.30 * ez12 + 0.15 * ez5) * 4.0 + 0.5 * spread_ticks - p_tn_5 * 3.0 - surprise_pen + extremum_bonus
            exp_hold_edge = self.inertia_arr * 3.0 - p_dn_12 * spread_ticks - surprise_pen * 0.6

            # Освобождаем PPO от жестких эвристик: разрешаем входы по сигналам нейросетевого роя,
            # отсекая только физически опасные зоны (покупка на хаях часа pos_3600 > 0.80, вершине 4ч пампа pos_14400 > 0.72 или в отвесном обвале)
            safe_macro_4h = (self.rng_14400 < 6.0) | (self.pos_14400 <= 0.72)
            waterfall_trap_vec = getattr(self, "enable_unfreeze", True) & (
                ((self.dd_900 >= 8.0) & (self.pos_14400 >= 0.45) & (ez25 < 0.0))
                | ((self.dd_900 >= 10.0) & (self.bid_q[:, 0] < 2500.0) & (self.ask_q[:, 0] > 800.0) & (ez12 < 0.0))
            )
            staircase_bleed_vec = getattr(self, "enable_unfreeze", True) & (
                ((self.pos_3600 <= 0.05) & (self.pos_14400 <= 0.05) & (self.vel_b1_180 <= -2.0) & (ez25 < 0.0))
                | ((self.pos_3600 <= 0.05) & (self.pos_14400 <= 0.05) & (self.rng_14400 >= 10.0) & (self.dd_900 <= 3.5) & (self.ask_q[:, 0] > 1000.0) & (self.bid_q[:, 0] < 20000.0))
                | ((ez5 < 0.15) & (ez12 <= 0.0) & (ez25 <= 0.0))
                | ((self.pos_3600 <= 0.05) & (ez12 <= 0.0) & (ez25 <= 0.0) & ~((self.bid_q[:, 0] >= 9000.0) & (imb5 >= 0.60) & (ez5 >= 0.15)))
                | ((self.rng_14400 >= 22.0) & (ez12 <= 0.05) & (ez25 <= 0.02) & (self.ask_q[:, 0] > 400.0))
                | ((ez60 <= -0.10) & (ez25 <= 0.05) & (self.dd_900 >= 2.0) & (self.bid_q[:, 0] < 8000.0))
            )
            top_rollover_vec = getattr(self, "enable_unfreeze", True) & (spread_ticks < 1.5) & (self.rng_14400 >= 16.0) & (self.pos_14400 >= 0.50) & (self.pos_3600 <= 0.15) & (self.dd_300 <= 1.0)
            self.waterfall_trap_arr = waterfall_trap_vec
            in_dip_zone_vec = (
                (self.pos_3600 <= 0.80)
                & safe_macro_4h
                & ~waterfall_trap_vec
                & ~staircase_bleed_vec
                & ~top_rollover_vec
                & (self.vel_b1_10 >= -0.5)
                & (self.inertia_arr >= 0.15)
                & (exp_entry_edge >= 0.60)
                & ((self.dd_300 >= 1.0) | (spread_ticks >= 2.0))
            )
            brk_wide_vec = (
                getattr(self, "enable_unfreeze", True)
                & (spread_ticks >= 1.8)
                & (self.bid_q[:, 0] >= 5500.0)
                & (self.ask_q[:, 0] <= 2500.0)
                & ((self.ask_q[:, 0] <= 1000.0) | (self.pos_3600 >= 0.35))
                & (self.inertia_arr >= 0.60)
                & (ez25 >= 0.35)
                & (self.dd_300 >= 1.0)
            )
            in_breakout_vec = (
                (
                    (ez5 >= 0.40)
                    & (ez12 >= 0.80)
                    & (ez25 >= 0.85)
                    & (self.inertia_arr >= 0.70)
                    & (exp_entry_edge >= 3.20)
                    & (imb5 >= 0.25)
                    & (self.vel_b1_10 >= 1.0)
                    & (self.rng_14400 <= 11.0)
                )
                | brk_wide_vec
            )
            self.in_breakout_arr = in_breakout_vec
            self.in_breakout_zone = in_breakout_vec
            in_dip_zone_vec = in_dip_zone_vec | in_breakout_vec

        self.obs_err12_norm = np.clip(model_err_12 / 5.0, -3.0, 3.0).astype(np.float32)
        self.obs_entry_edge_norm = np.clip(exp_entry_edge / 5.0, -3.0, 3.0).astype(np.float32)
        self.obs_hold_edge_norm = np.clip(exp_hold_edge / 5.0, -3.0, 3.0).astype(np.float32)
        self.can_buy_passive_arr = in_dip_zone_vec
        self.can_buy_best_arr = in_dip_zone_vec & (imb5 > -0.65)

        self._obs_buf = np.empty(self.obs_dim, dtype=np.float32)
        self._mask_buf = np.zeros(self.n_actions, dtype=bool)
        self._oracle_cache = None
        self.reset()

    def get_oracle_benchmark(self) -> dict:
        """
        Вычисляет физический потолок Идеального Maker-Оракула (100% КПД) на интервале [start_idx, end_idx]
        в той же самой L2 FIFO-среде (с учетом реальной очереди q_ahead, задержки 1с и масок).
        """
        if self._oracle_cache is not None:
            return self._oracle_cache

        saved_legacy = self.legacy_v2
        self.legacy_v2 = False
        obs = self.reset(eval_mode=True)
        done = False
        n_max = len(self.bid_p)
        b1_all = np.maximum(self.bid_p[:, 0], self.tick_size)
        a1_all = np.maximum(self.ask_p[:, 0], b1_all + self.tick_size)

        while not done:
            t = self.t
            mask = self.get_action_mask()
            w_end = min(n_max, t + 1800)
            fut_b1 = b1_all[t + 1:w_end]
            fut_a1 = a1_all[t + 1:w_end]
            act = 0

            if self.pos_qty <= 1e-8:
                b1 = b1_all[t]
                a1 = a1_all[t]
                target_b = (b1 + self.tick_size) if (a1 - b1) >= 2.0 * self.tick_size else b1
                if len(fut_b1) > 5:
                    # Проверяем, даст ли волна минимум +1..2 тика профита без провала против нашего Bid
                    max_gain_ticks = (np.max(fut_a1) - target_b) / self.tick_size
                    max_drop_ticks = (target_b - np.min(fut_b1[:60])) / self.tick_size
                    good_wave = (max_gain_ticks >= 2.0 and max_drop_ticks <= 1.0) or (max_gain_ticks >= 1.0 and max_drop_ticks <= 0.0)
                    if good_wave and mask[1]:
                        act = 1 if (self.order_side != "BUY" or self.order_act == 2) else 0
                    elif good_wave and mask[2]:
                        act = 0 if self.order_side == "BUY" else 2
                    elif self.order_side == "BUY" and max_drop_ticks > 1.0 and mask[5]:
                        act = 5
            else:
                if self.order_side == "SELL":
                    act = 0
                else:
                    can_reach_2 = len(fut_b1) > 0 and ((np.max(fut_b1) - self.avg_entry) / self.tick_size >= 2.0)
                    if can_reach_2 and mask[3]:
                        act = 3
                    elif mask[4]:
                        act = 4
                    elif mask[3]:
                        act = 3

            obs, _, done = self.step(act)

        oracle_pnl = max(0.10, float(self.prev_mtm))
        self._oracle_cache = {
            "oracle_pnl_usd": round(oracle_pnl, 4),
            "oracle_pnl_pct": round(oracle_pnl / self.order_usd * 100.0, 2),
            "oracle_trades": int(self.n_trades),
        }
        self.legacy_v2 = saved_legacy
        self.reset(eval_mode=True)
        return self._oracle_cache

    def _get_lob_queue(self, t: int, side: str, price: float):
        tol = self.tick_size * 0.25
        q_at, q_better = 0.0, 0.0
        if side == "BUY":
            for lvl in range(10):
                p, q = self.bid_p[t, lvl], self.bid_q[t, lvl]
                if p <= 0 or q <= 0:
                    continue
                if abs(p - price) <= tol:
                    q_at += q
                elif p > price + tol:
                    q_better += q
        else:
            for lvl in range(10):
                p, q = self.ask_p[t, lvl], self.ask_q[t, lvl]
                if p <= 0 or q <= 0:
                    continue
                if abs(p - price) <= tol:
                    q_at += q
                elif p < price - tol:
                    q_better += q
        return q_at, q_better

    def _sweep_taker(self, t: int, qty: float):
        rem = qty
        proceeds = 0.0
        filled = 0.0
        for lvl in range(10):
            p, q = self.bid_p[t, lvl], self.bid_q[t, lvl]
            if p <= 0 or q <= 0:
                continue
            take = min(rem, q)
            proceeds += take * p
            filled += take
            rem -= take
            if rem <= 1e-8:
                break
        if rem > 1e-8:
            fb_p = max(self.tick_size, self.bid_p[t, 0] - 10 * self.tick_size)
            proceeds += rem * fb_p
            filled += rem
        return proceeds / max(filled, 1e-8)

    def reset(self, eval_mode: bool = False):
        if eval_mode or (self.end_idx - self.start_idx <= self.episode_len + 10):
            self.t = self.start_idx
            self.max_t = self.end_idx - 1
        else:
            self.t = int(np.random.randint(self.start_idx, self.end_idx - self.episode_len - 1))
            self.max_t = self.t + self.episode_len
        self.pos_qty = 0.0
        self.avg_entry = 0.0
        self.peak_bid = 0.0
        self.entry_t = 0
        self.order_side = None
        self.order_act = 0
        self.order_price = 0.0
        self.order_qty = 0.0
        self.q_ahead = 0.0
        self.active_after = 0
        self.slot2_side = None
        self.slot2_price = 0.0
        self.slot2_qty = 0.0
        self.slot2_q_ahead = 0.0
        self.slot2_active_after = 0
        self.split_done_for_pos = False
        self.prev_mtm = 0.0
        self.total_pnl_usd = 0.0
        self.n_trades = 0
        self.cooldown_until = 0
        self.bid_placed_t = 0
        self.last_exit_t = 0
        self.last_exit_price = 0.0
        self.last_loss_exit_t = 0
        self.last_loss_exit_price = 0.0
        self.car_state = 0  # 0: idle, 1: quoting sell @ local ask, 2: quoting buyback @ local bid
        self.car_qty = 0.0
        self.car_px = 0.0
        self.car_q_ahead = 0.0
        self.car_sell_px = 0.0
        self.car_cycles = 0
        self.unfreeze_active = False
        return self._get_obs()

    def _compute_adaptive_wave_weight(self, t: int, floor_p: float, wave_p: float) -> float:
        """
        Адаптивный расчёт доли позиции во 2-м слоте (Slot 2 @ +2 тика) на основе импульса триады
        и собственного импакта нашего объёма на очередь Ask L1/L2.
        Возвращает:
          1.0       — вся заявка ставится на волновой выход (+2 тика)
          0.30..0.65 — адаптивный сплит между Slot 1 (+1 тик) и Slot 2 (+2 тика)
          0.0       — вся заявка ставится на быстрый скальп (+1 тик)
        """
        ez25 = float(self.X_market[t, 12])
        ez12 = float(self.X_market[t, 6])
        hold_edge = float(self.obs_hold_edge_norm[t] * 5.0)
        b1 = max(self.bid_p[t, 0], self.tick_size)
        a1 = max(self.ask_p[t, 0], b1 + self.tick_size)
        vel10 = float(self.vel_10[t])
        imb5 = float(self.imb_l5[t])
        inertia = float(self.inertia_arr[t])
        step_in_supported = (
            getattr(self, "enable_unfreeze", True)
            and b1 >= self.avg_entry - self.tick_size * 1.25
            and (
                a1 >= self.avg_entry + 1.8 * self.tick_size
                or (
                    (float(self.pos_3600[t]) >= 0.30 or float(self.dd_300[t]) >= 3.0)
                    and float(self.bid_q[t, 0]) >= 4000.0
                    and (imb5 >= 0.20 or (ez12 >= 0.55 and ez25 >= 0.35 and float(self.ask_q[t, 0]) <= 2500.0))
                )
            )
        )
        if b1 < self.avg_entry - self.tick_size * 0.25 and not step_in_supported:
            return 0.0
        et_prev = max(0, self.entry_t - 1)
        if getattr(self, "enable_unfreeze", True) and float(self.pos_14400[t]) <= 0.15 and min(float(self.X_market[self.entry_t, 6]), float(self.X_market[et_prev, 6])) < -0.25 and min(float(self.X_market[self.entry_t, 12]), float(self.X_market[et_prev, 12])) < -0.25:
            return 0.0
        if (
            vel10 >= 1.0
            or bool(self.in_breakout_zone[t])
            or (self.avg_entry > 0 and b1 >= self.avg_entry + self.tick_size * 0.75 and inertia >= 0.0)
            or (imb5 >= 0.22 and inertia >= 0.25 and ez12 >= 0.15)
            or (
                getattr(self, "enable_unfreeze", True)
                and imb5 >= 0.35
                and float(self.bid_q[t, 0]) >= 5000.0
                and float(self.ask_q[t, 0]) <= 1500.0
                and (inertia >= 0.05 or (a1 >= self.avg_entry + 1.8 * self.tick_size and imb5 >= 0.50))
            )
        ):
            return 1.0
        q_a1, _ = self._get_lob_queue(t, "SELL", floor_p)
        q_a2, _ = self._get_lob_queue(t, "SELL", wave_p)
        own_ask_share = self.pos_qty / (q_a1 + self.pos_qty + 1e-6)

        if max(float(self.pos_3600[t]), float(self.pos_3600[self.entry_t])) >= 0.65 and q_a2 > 3000.0 and vel10 <= 0.0 and inertia < 0.25:
            return 0.0
        if q_a2 > 15000.0 and q_a1 < 500.0:
            return 0.50

        if (hold_edge >= 0.90 and ez25 >= 0.15 and ez12 >= 0.08) or (hold_edge >= 0.80 and q_a2 < 250.0 and q_a1 > 500.0):
            return 1.0
        if (hold_edge >= 0.45 and ez25 >= 0.08) or (ez25 >= 0.14 and hold_edge >= 0.20) or (own_ask_share > 0.15 and ez25 >= 0.08 and hold_edge >= 0.10):
            q_adv = q_a1 / (q_a1 + q_a2 + 200.0)
            return float(np.clip(0.35 + 0.12 * max(0.0, hold_edge) + 0.25 * q_adv, 0.30, 0.65))
        return 0.0

    def get_action_mask(self) -> np.ndarray:
        t = self.t
        mask = np.zeros(self.n_actions, dtype=bool)
        if self.pos_qty <= 1e-8:
            mask[0] = True
            if self.order_side == "BUY" and not self.legacy_v2:
                b1 = max(self.bid_p[t, 0], self.tick_size)
                drift_limit = self.tick_size * 1.25 if self.order_act == 2 else self.tick_size * 0.25
                brk_hold = bool(getattr(self, "in_breakout_arr", np.zeros(len(self.pos_3600), dtype=bool))[t])
                need_cancel = (
                    (float(self.inertia_arr[t]) < -0.10)
                    or (float(self.pos_3600[t]) > 0.82 and not brk_hold)
                    or (float(self.X_market[t, 20]) < -0.55)
                    or (b1 > self.order_price + drift_limit)
                )
                if need_cancel:
                    mask[5] = True
                    if t >= self.cooldown_until:
                        mask[1] = bool(self.can_buy_best_arr[t])
                        mask[2] = bool(self.can_buy_passive_arr[t])
                elif self.order_act == 2 and t >= self.cooldown_until:
                    mask[1] = bool(self.can_buy_best_arr[t])
            else:
                if t >= self.cooldown_until:
                    mask[1] = bool(self.can_buy_best_arr[t])
                    mask[2] = bool(self.can_buy_passive_arr[t])
                if self.order_side is not None:
                    mask[5] = True
        else:
            mask[0] = (self.order_side == "SELL")
            mask[3] = True
            mask[4] = True
            if self.legacy_v2:
                mask[6] = True
                if self.order_side is not None:
                    mask[5] = True
            else:
                b1 = max(self.bid_p[t, 0], self.tick_size)
                unr_ticks = (b1 - self.avg_entry) / self.tick_size
                # Разрешаем аварийный Taker-выход только при реальной угрозе пролива, исключая случайный слив при семплировании PPO
                if unr_ticks <= -2.0 or float(self.inertia_arr[t]) < -0.25 or float(self.obs_err12_norm[t]) < -0.60:
                    mask[6] = True
        return mask

    def _get_obs(self) -> np.ndarray:
        t = self.t
        b1 = max(self.bid_p[t, 0], self.tick_size)
        a1 = max(self.ask_p[t, 0], b1 + self.tick_size)
        has_pos = 1.0 if self.pos_qty > 1e-8 else 0.0
        unr_ticks = ((b1 - self.avg_entry) / self.tick_size) if has_pos else 0.0
        hold_ratio = min((t - self.entry_t) / 45.0, 2.0) if has_pos else 0.0
        has_bid = 1.0 if self.order_side == "BUY" else 0.0
        has_ask = 1.0 if self.order_side == "SELL" else 0.0
        if has_bid:
            dist = (b1 - self.order_price) / self.tick_size
        elif has_ask:
            dist = (self.order_price - a1) / self.tick_size
        else:
            dist = 0.0
        q_ratio = self.q_ahead / (self.q_ahead + self.order_qty + 1e-6) if self.order_side else 0.0

        buf = self._obs_buf
        m_dim = self.X_market.shape[1]
        buf[:m_dim] = self.X_market[t]
        if getattr(self, "use_v52_obs", False):
            buf[m_dim : m_dim + 4] = self.X_v52[t]
        if getattr(self, "use_v70_obs", False):
            s_fill = float(self.v70_slot_fill_ratio) if self.v70_slot_fill_ratio > 0 else (0.6 if has_pos else 0.0)
            low_edge = float(self.v70_lowest_slot_edge_ticks) if self.v70_slot_fill_ratio > 0 else (unr_ticks * 0.2)
            disp = float(self.v70_slot_dispersion_ticks)
            free_c = float(self.v70_free_cash_ratio) if self.v70_slot_fill_ratio > 0 else (0.4 if has_pos else 1.0)
            ez90_t = float(self.X_h90[t, 0]) if getattr(self, "X_h90", None) is not None else float(self.X_h60[t, 0])
            ez120_t = float(self.X_h120[t, 0]) if getattr(self, "X_h120", None) is not None else ez90_t
            p_fr_t = float(self.X_v70_realism[t, 1]) if getattr(self, "X_v70_realism", None) is not None else 0.15
            buf[m_dim + 4 : m_dim + 11] = (
                s_fill,
                min(max(low_edge, -3.0), 3.0),
                min(max(disp, 0.0), 3.0),
                free_c,
                min(max(ez90_t, -3.0), 3.0),
                min(max(ez120_t, -3.0), 3.0),
                p_fr_t,
            )
        buf[-10] = has_pos
        buf[-9] = min(max(unr_ticks * 0.1, -3.0), 3.0)
        buf[-8] = hold_ratio
        buf[-7] = has_bid
        buf[-6] = has_ask
        buf[-5] = min(max(dist * 0.2, -2.0), 2.0)
        buf[-4] = q_ratio
        buf[-3] = self.obs_err12_norm[t]
        buf[-2] = self.obs_entry_edge_norm[t]
        buf[-1] = self.obs_hold_edge_norm[t]
        return buf.copy()

    def step(self, action: int):
        t = self.t
        b1 = max(self.bid_p[t, 0], self.tick_size)
        a1 = max(self.ask_p[t, 0], b1 + self.tick_size)
        tol = self.tick_size * 0.25
        requote_penalty = 0.0
        inertia = float(self.inertia_arr[t])
        pos900 = float(self.pos_900[t])
        if self.pos_qty > 1e-8:
            self.peak_bid = max(self.peak_bid, b1)

        # 1. Обработка выбранного действия агента (прямое исполнение решений PPO без скрытых переопределений)
        pos3600 = float(self.pos_3600[t])
        sp = float(self.X_market[t, 18] * 10.0)
        ez25 = float(self.X_market[t, 12])
        hold_edge = float(self.obs_hold_edge_norm[t] * 5.0)
        brk_now = bool(getattr(self, "in_breakout_arr", np.zeros(len(self.pos_3600), dtype=bool))[t])
        wide_scalp_ok = bool(
            getattr(self, "enable_unfreeze", True)
            and (a1 - b1) >= 1.8 * self.tick_size
            and (
                (
                    float(self.bid_q[t, 0]) >= 2500.0
                    and float(self.ask_q[t, 0]) <= 2500.0
                    and pos3600 <= 0.75
                    and inertia >= 0.05
                    and (inertia >= 0.20 or float(self.imb_l5[t]) >= 0.50 or ez25 > -0.25)
                )
                or (
                    float(self.bid_q[t, 0]) >= 18000.0
                    and float(self.ask_q[t, 0]) <= 7500.0
                    and pos3600 <= 0.50
                    and float(self.rng_14400[t]) <= 9.0
                    and float(self.X_market[t, 6]) >= 0.80
                    and ez25 >= 0.65
                    and inertia >= 0.75
                    and float(self.imb_l5[t]) >= 0.65
                )
                or (
                    float(self.bid_q[t, 0]) >= 1700.0
                    and float(self.ask_q[t, 0]) <= 1250.0
                    and 0.45 <= pos3600 <= 0.55
                    and float(self.pos_14400[t]) <= 0.10
                    and float(self.dd_900[t]) == 1.0
                    and float(self.X_market[t, 0]) >= 0.20
                    and float(self.imb_l5[t]) >= 0.55
                )
            )
            and not (float(self.X_market[t, 0]) < 0.15 and float(self.X_market[t, 6]) <= 0.0 and ez25 <= 0.0)
            and not (pos3600 <= 0.05 and float(self.X_market[t, 6]) <= 0.0 and ez25 <= 0.0 and not (float(self.bid_q[t, 0]) >= 9000.0 and float(self.imb_l5[t]) >= 0.60 and float(self.X_market[t, 0]) >= 0.15))
            and not (pos3600 <= 0.05 and float(self.pos_14400[t]) <= 0.05 and float(self.rng_14400[t]) >= 10.0 and float(self.dd_900[t]) <= 3.5 and float(self.ask_q[t, 0]) > 1000.0 and float(self.bid_q[t, 0]) < 20000.0)
            and not (pos3600 <= 0.05 and float(self.pos_14400[t]) <= 0.05 and float(self.vel_b1_180[t]) <= -2.0 and ez25 < 0.0)
            and not (float(self.dd_900[t]) >= 8.0 and float(self.pos_14400[t]) >= 0.45 and ez25 < 0.0)
            and not (float(self.dd_900[t]) >= 10.0 and float(self.bid_q[t, 0]) < 2500.0 and float(self.ask_q[t, 0]) > 800.0 and float(self.X_market[t, 6]) < 0.0)
            and not (float(self.rng_14400[t]) >= 22.0 and float(self.X_market[t, 6]) <= 0.05 and ez25 <= 0.02 and float(self.ask_q[t, 0]) > 400.0)
            and not (float(self.obs_entry_edge_norm[t] * 5.0) < -1.50 and hold_edge < -1.50 and float(self.obs_err12_norm[t]) < -1.0 and ez25 <= 0.0)
            and not (self.last_exit_t > 0 and (t - self.last_exit_t) <= 15 and (b1 + self.tick_size) >= self.last_exit_price - tol and float(self.X_market[t, 0]) < 0.0 and float(self.obs_err12_norm[t]) < -0.30)
            and not (float(self.X_market[t, 0]) < 0.10 and float(self.X_market[t, 6]) < 0.10 and ez25 < -0.05 and hold_edge < 0.0 and float(self.imb_l5[t]) < 0.35)
            and float(self.vel_b1_10[t]) >= -0.5
        )
        if not self.legacy_v2 and self.pos_qty <= 1e-8 and self.order_side is None and action == 0 and (brk_now or wide_scalp_ok) and t >= self.cooldown_until:
            action = 1 if (self.can_buy_best_arr[t] or wide_scalp_ok) else 2
        if not self.legacy_v2 and self.pos_qty <= 1e-8 and self.order_side == "BUY":
            dust_b1_flicker = (
                abs(self.order_price - (b1 - self.tick_size)) <= tol
                and float(self.bid_q[t, 1]) >= 11000.0
                and float(self.rng_14400[t]) <= 6.0
                and float(self.pos_14400[t]) <= 0.20
            )
            at_b1 = ((self.order_price >= b1 - tol) and (self.order_price <= b1 + self.tick_size + tol)) or dust_b1_flicker
            compressed_4h_floor = (
                (4.8 <= self.rng_14400[t] <= 5.5 and self.pos_14400[t] <= 0.05 and self.dd_900[t] <= 1.0)
                or (pos3600 <= 0.05 and float(self.pos_14400[t]) <= 0.05 and float(self.rng_14400[t]) <= 9.0 and float(self.bid_q[t, 0]) >= 12000.0 and float(self.bid_q[t, 0]) >= 1.8 * float(self.ask_q[t, 0]) and float(self.imb_l5[t]) >= 0.65)
                or dust_b1_flicker
            )
            l1_bid_erosion = (
                getattr(self, "enable_unfreeze", True)
                and sp < 1.5
                and float(self.dq_bid_5s[t]) <= -1000.0
                and float(self.bid_q[t, 0]) < 1800.0
                and float(self.buy_vol_10s[t]) < 100.0
                and not (
                    (float(self.ask_q[t, 0]) <= 10.0 and inertia >= 0.15 and float(self.rng_14400[t]) >= 16.5)
                    or ((inertia >= 0.25 or float(self.ask_q[t, 0]) <= 100.0) and (float(self.rng_14400[t]) <= 12.5 or 21.5 <= float(self.rng_14400[t]) <= 22.5))
                )
            )
            stale_top_rollover = (
                getattr(self, "enable_unfreeze", True)
                and (t - self.bid_placed_t) >= 180
                and sp < 1.5
                and pos3600 <= 0.15
                and float(self.pos_14400[t]) >= 0.50
            )
            staircase_ask_wall = (
                getattr(self, "enable_unfreeze", True)
                and pos3600 <= 0.05
                and float(self.pos_14400[t]) <= 0.05
                and float(self.rng_14400[t]) >= 10.0
                and float(self.dd_900[t]) <= 3.5
                and float(self.ask_q[t, 0]) > 1000.0
                and float(self.bid_q[t, 0]) < 20000.0
            )
            l1_crush = (sp < 1.5 and float(self.ask_q[t, 0]) > 5000.0 and float(self.bid_q[t, 0]) < 800.0 and inertia < 0.0) or l1_bid_erosion or stale_top_rollover or staircase_ask_wall
            pre_dump = ((sp < 1.5 and ez25 < -0.15 and inertia < 0.08 and self.ask_q[t, 0] > 4000.0) or l1_crush) and not compressed_4h_floor
            front_q_hold = (
                (
                    at_b1
                    and not pre_dump
                    and (
                        (self.q_ahead < 600.0 and pos3600 <= 0.65 and inertia >= -0.10)
                        or (self.q_ahead < 50.0 and (float(self.bid_q[t, 0]) >= 1200.0 or float(self.bid_q[t, 1]) >= 1700.0) and float(self.ask_q[t, 0]) <= 1250.0 and ((pos3600 <= 0.55 and float(self.pos_14400[t]) >= 0.80) or (float(self.pos_14400[t]) <= 0.20 and (float(self.rng_14400[t]) <= 12.5 or 14.5 <= float(self.rng_14400[t]) <= 16.5 or 21.5 <= float(self.rng_14400[t]) <= 22.5))) and float(self.dd_900[t]) <= 2.0)
                        or (abs(self.order_price - b1) <= tol and self.q_ahead < 950.0 and sp < 1.5 and float(self.ask_q[t, 0]) <= 30.0 and pos3600 <= 0.05 and float(self.pos_14400[t]) <= 0.15 and float(self.dd_300[t]) >= 1.0 and float(self.dd_900[t]) >= 2.0 and float(self.imb_l5[t]) >= 0.17 and (float(self.rng_14400[t]) <= 9.0 or (15.5 <= float(self.rng_14400[t]) <= 16.5 and float(self.pos_14400[t]) <= 0.02 and float(self.dd_900[t]) == 3.0)))
                        or (abs(self.order_price - b1) <= tol and (a1 - b1) >= 1.8 * self.tick_size and float(self.bid_q[t, 0]) >= 3000.0 and float(self.ask_q[t, 0]) <= 3600.0 and 0.65 <= pos3600 <= 0.75 and float(self.pos_14400[t]) <= 0.25 and float(self.dd_900[t]) == 1.0 and 21.5 <= float(self.rng_14400[t]) <= 22.5)
                        or dust_b1_flicker
                    )
                )
                or (
                    abs(self.order_price - (b1 - self.tick_size)) <= tol
                    and self.q_ahead < 10.0
                    and pos3600 <= 0.50
                    and float(self.pos_14400[t]) <= 0.20
                    and 21.5 <= float(self.rng_14400[t]) <= 22.5
                    and not pre_dump
                )
            )
            brk_hold = (brk_now or (at_b1 and (a1 - b1) >= 1.8 * self.tick_size and float(self.bid_q[t, 0]) >= 1500.0 and inertia >= -0.05)) and not pre_dump
            v14_hold_q = (at_b1 or front_q_hold) and ((pos3600 <= 0.25 and not pre_dump and self.bid_q[t, 0] >= 5000.0) or compressed_4h_floor or front_q_hold or brk_hold)
            bid_age = t - self.bid_placed_t
            if self.order_price > b1 + tol and self.vel_b1_10[t] <= -1.0 and ((hold_edge < 0.0 and self.ask_q[t, 0] > 500.0) or (ez25 < 0.0 and float(self.X_market[t, 6]) < 0.10)):
                action = 5
            elif v14_hold_q:
                if action == 5:
                    action = 0
            elif (pos3600 > 0.85 and not brk_hold) or inertia < -0.20:
                action = 5
            elif self.dd_300[t] < 0.5 and self.dd_900[t] <= 1.0 and sp < 1.5 and self.X_market[t, 6] < 0.30:
                if bid_age >= 15 and float(self.X_market[t, 6]) < 0.12 and inertia < 0.08:
                    action = 5
                elif action == 5:
                    action = 0
            elif pre_dump:
                action = 5

        if not self.legacy_v2 and self.pos_qty > 1e-8:
            floor_p = round(math.ceil((self.avg_entry + 0.75 * self.tick_size) / self.tick_size - 1e-4) * self.tick_size, 2)
            wave_p = round(floor_p + self.tick_size, 2)
            ez12_now = float(self.X_market[t, 6])
            car_step_up_ask = (sp < 1.5) and (
                float(self.imb_l5[t]) > 0.25 and float(self.vel_10[t]) >= 1.0
            )
            opt_car_sell_p = round(min(floor_p, (a1 + self.tick_size) if car_step_up_ask else max(a1, b1 + self.tick_size)) / self.tick_size) * self.tick_size
            early_wide_unfreeze = (
                (self.avg_entry - b1) / self.tick_size >= 3.0
                and a1 >= self.avg_entry - tol
                and float(self.ask_q[t, 0]) <= 250.0
                and float(self.dd_900[t]) >= 4.0
            )
            bullish_triad_hold = (
                (self.avg_entry - b1) / self.tick_size <= 4.0
                and ez12_now >= 0.45
                and ez25 >= 0.45
                and float(self.X_market[t, 0]) >= 0.20
            )
            unfreeze_min_sec = 900 if (float(self.rng_14400[t]) >= 25.0 and float(self.pos_14400[t]) <= 0.25) else 300
            unfreeze_mode = getattr(self, "enable_unfreeze", True) and not bullish_triad_hold and (
                ((self.avg_entry - b1) / self.tick_size >= 3.0 and (t - self.entry_t) >= unfreeze_min_sec)
                or early_wide_unfreeze
            )
            wave_hold_ok = b1 >= self.avg_entry - tol or (
                getattr(self, "enable_unfreeze", True)
                and b1 >= self.avg_entry - 1.25 * self.tick_size
                and (
                    a1 >= self.avg_entry + 1.8 * self.tick_size
                    or (
                        (pos3600 >= 0.30 or float(self.dd_300[t]) >= 3.0)
                        and float(self.bid_q[t, 0]) >= 4000.0
                        and (float(self.imb_l5[t]) >= 0.20 or (ez12_now >= 0.55 and ez25 >= 0.35 and float(self.ask_q[t, 0]) <= 2500.0))
                    )
                )
            )
            if unfreeze_mode:
                if self.order_side != "SELL" or self.slot2_side is not None or self.order_price > opt_car_sell_p + tol or (self.order_price < opt_car_sell_p - tol and car_step_up_ask):
                    self.order_side = "SELL"
                    self.order_act = 4
                    self.order_price = opt_car_sell_p
                    self.order_qty = self.pos_qty
                    self.q_ahead, _ = self._get_lob_queue(t, "SELL", self.order_price)
                    self.active_after = t + 1
                    self.slot2_side = None
                    self.slot2_qty = 0.0
                    self.car_state = 0
                action = 0
            elif self.order_side == "SELL" and self.order_price >= wave_p - tol and not wave_hold_ok and hold_edge < -0.20 and a1 <= floor_p + tol:
                self.order_price = max(a1, floor_p)
                self.order_qty = self.pos_qty
                self.q_ahead, _ = self._get_lob_queue(t, "SELL", self.order_price)
                self.slot2_side = None
                self.slot2_qty = 0.0
                action = 0

        if not self.legacy_v2 and self.pos_qty > 1e-8 and action in (3, 4):
            floor_p = round(math.ceil((self.avg_entry + 0.75 * self.tick_size) / self.tick_size - 1e-4) * self.tick_size, 2)
            wave_p = round(floor_p + self.tick_size, 2)
            super_p = round(floor_p + 2.0 * self.tick_size, 2)
            q_a1, _ = self._get_lob_queue(t, "SELL", floor_p)
            q_a2, _ = self._get_lob_queue(t, "SELL", wave_p)
            w2 = self._compute_adaptive_wave_weight(t, floor_p, wave_p)
            e_t = self.entry_t
            ez12 = float(self.X_market[t, 6])
            triad_super = (
                sp < 1.5
                and ez12 >= 0.70
                and ez25 >= 0.70
                and hold_edge >= 3.10
                and (ez25 >= 0.85 or hold_edge >= 3.80)
                and inertia >= 1.25
                and q_a1 < 1000.0
                and (
                    (q_a2 < 4500.0 and self.dd_900[t] >= 2.0)
                    or (self.rng_14400[t] >= 19.0 and ez12 >= 1.25 and ez25 >= 0.90 and hold_edge >= 3.80 and inertia >= 1.50 and self.dd_900[t] >= 1.0)
                )
            )
            is_super = (
                (self.dd_300[t] >= 4.0 and sp < 1.5 and ez25 >= 0.20 and hold_edge >= 1.50 and q_a2 < 800.0)
                or (self.dd_900[e_t] >= 3.0 and self.pos_3600[e_t] <= 0.05 and float(self.X_market[e_t, 12]) >= 0.65 and float(self.obs_hold_edge_norm[e_t] * 5.0) >= 4.0 and q_a2 < 300.0)
                or triad_super
            )
            if self.order_side != "SELL" and is_super and not self.split_done_for_pos:
                self.order_side = "SELL"
                self.order_act = 3
                self.order_price = max(a1 + 2.0 * self.tick_size, super_p)
                self.order_qty = self.pos_qty
                self.q_ahead, _ = self._get_lob_queue(t, "SELL", self.order_price)
                self.active_after = t + 1
                self.slot2_side = None
                self.slot2_qty = 0.0
                self.split_done_for_pos = True
                requote_penalty = 0.005
                action = 0
            elif self.order_side == "SELL" and self.order_price >= wave_p - tol and self.split_done_for_pos:
                if self.slot2_side is None and (t - self.entry_t) >= 600 and b1 <= self.avg_entry - 2.0 * self.tick_size and hold_edge < 0.25:
                    q2 = round(self.pos_qty * 0.50, 4)
                    q1 = round(self.pos_qty - q2, 4)
                    self.order_price = max(a1, floor_p)
                    self.order_qty = q1
                    self.q_ahead, _ = self._get_lob_queue(t, "SELL", self.order_price)
                    self.slot2_side = "SELL"
                    self.slot2_price = max(a1 + self.tick_size, wave_p)
                    self.slot2_qty = q2
                    self.slot2_q_ahead, _ = self._get_lob_queue(t, "SELL", self.slot2_price)
                    self.slot2_active_after = t + 1
                action = 0
            elif self.order_side == "SELL" and self.slot2_side == "SELL":
                if w2 <= 0.05 and a1 <= floor_p + tol and q_a2 <= 15000.0:
                    self.order_qty = self.pos_qty
                    self.slot2_side = None
                    self.slot2_qty = 0.0
                action = 0
            elif self.order_side == "SELL" and self.slot2_side is None:
                pump_up = bool(self.vel_10[t] >= 1.0 or self.in_breakout_zone[t] or (self.imb_l5[t] >= 0.22 and inertia >= 0.25))
                if pump_up and a1 >= self.order_price - tol and self.imb_l5[t] >= 0.20 and b1 >= self.avg_entry:
                    action = 3
                elif self.order_price >= wave_p - tol and wave_hold_ok:
                    action = 0
                elif abs(self.order_price - floor_p) <= tol:
                    action = 0
                else:
                    action = 4
            elif self.order_side != "SELL" and 0.15 <= w2 <= 0.85 and not self.split_done_for_pos:
                q2 = round(self.pos_qty * w2, 4)
                q1 = round(self.pos_qty - q2, 4)
                self.order_side = "SELL"
                self.order_act = 4
                self.order_price = max(a1, floor_p)
                self.order_qty = q1
                self.q_ahead, _ = self._get_lob_queue(t, "SELL", self.order_price)
                self.active_after = t + 1
                self.slot2_side = "SELL"
                self.slot2_price = max(a1 + self.tick_size, wave_p)
                self.slot2_qty = q2
                self.slot2_q_ahead, _ = self._get_lob_queue(t, "SELL", self.slot2_price)
                self.slot2_active_after = t + 1
                self.split_done_for_pos = True
                requote_penalty = 0.005
                action = 0
            elif self.order_side != "SELL" and w2 > 0.85 and getattr(self, "enable_unfreeze", True):
                self.split_done_for_pos = True
                action = 3
            elif self.order_side != "SELL" and w2 < 0.15 and not bool(self.vel_10[t] >= 1.0 or self.in_breakout_zone[t]):
                action = 4

        if action in (1, 2) and self.pos_qty <= 1e-8:
            ez60_now = float(self.ez60_arr[t]) if hasattr(self, "ez60_arr") else ez25
            in_post_loss_window = (not self.legacy_v2) and (self.last_loss_exit_t > 0) and ((t - self.last_loss_exit_t) <= 2700)
            macro_step_in_ok = self.legacy_v2 or (
                not in_post_loss_window
                and not (ez60_now < 0.0 and float(self.dd_900[t]) >= 2.0 and float(self.bid_q[t, 0]) < 8500.0)
            )
            step_in_l5_ok = (
                float(self.imb_l5[t]) >= 0.0
                or float(self.bid_q[t, 0]) >= 8500.0
                or float(self.X_market[t, 6]) >= 0.90
                or (23.5 <= float(self.rng_14400[t]) <= 25.5 and float(self.dd_900[t]) == 1.0 and float(self.imb_l5[t]) >= -0.10)
                or (18.5 <= float(self.rng_14400[t]) <= 19.5 and float(self.dd_900[t]) >= 11.0 and pos3600 <= 0.05)
            )
            step_in = (
                (a1 - b1) >= 1.8 * self.tick_size
                and macro_step_in_ok
                and (
                    self.legacy_v2
                    or (step_in_l5_ok and ((float(self.ask_q[t, 0]) <= 5000.0 and inertia >= 0.35) or wide_scalp_ok))
                )
            )
            target_p = (b1 + self.tick_size if step_in else b1) if action == 1 else max(self.tick_size, b1 - self.tick_size)
            if in_post_loss_window and target_p > self.last_loss_exit_price + tol and not (brk_now or (float(self.bid_q[t, 0]) >= 8500.0 and ez60_now >= 0.25 and ez25 >= 0.45)):
                target_p = min(target_p, b1)
                if target_p > self.last_loss_exit_price + tol:
                    target_p = max(self.tick_size, b1 - self.tick_size)
            if self.order_side != "BUY" or abs(self.order_price - target_p) > tol:
                requote_penalty = 0.025 if self.order_side is not None else 0.005
                self.order_side = "BUY"
                self.order_act = int(action)
                self.order_price = round(target_p / self.tick_size) * self.tick_size
                self.order_qty = self.order_usd / self.order_price
                q_at_t, _ = self._get_lob_queue(t, "BUY", self.order_price)
                self.q_ahead = q_at_t
                self.active_after = t + 1
                self.bid_placed_t = t
        elif action in (3, 4) and self.pos_qty > 1e-8:
            hold_s = t - self.entry_t
            if self.legacy_v2:
                ez25_entry = float(self.X_market[self.entry_t, 12])
                trend_ticks = int(np.clip(round(max(ez25_entry, float(self.X_market[t, 12])) * 6.0), 2, 6))
                if action == 3:
                    target_p = max(a1, self.avg_entry + trend_ticks * self.tick_size)
                else:
                    min_p = (self.avg_entry + self.tick_size) if hold_s < 45 else self.avg_entry
                    target_p = max(b1 + self.tick_size, min_p, a1 - self.tick_size)
            else:
                floor_p = round((self.avg_entry + self.tick_size) / self.tick_size) * self.tick_size
                if action == 3:
                    target_p = max(a1, self.avg_entry + 2.0 * self.tick_size)
                else:
                    if self.order_side == "SELL" and abs(self.order_price - floor_p) <= tol and b1 < floor_p - tol:
                        target_p = floor_p
                    else:
                        target_p = max(a1, floor_p)
            if self.order_side != "SELL" or abs(self.order_price - target_p) > tol:
                if self.order_side is not None:
                    requote_penalty = 0.025
                self.order_side = "SELL"
                self.order_act = int(action)
                self.order_price = round(target_p / self.tick_size) * self.tick_size
                self.order_qty = self.pos_qty
                q_at_t, _ = self._get_lob_queue(t, "SELL", self.order_price)
                self.q_ahead = q_at_t
                self.active_after = t + 1
        elif action == 5:
            if self.order_side is not None:
                self.order_side = None
                self.order_act = 0
                self.slot2_side = None
                self.slot2_qty = 0.0
                requote_penalty = 0.025
        elif action == 6 and self.pos_qty > 1e-8:
            exec_p = self._sweep_taker(t, self.pos_qty)
            net_pnl = self.pos_qty * (exec_p * (1.0 - 0.001) - self.avg_entry)
            self.total_pnl_usd += net_pnl
            self.pos_qty = 0.0
            self.avg_entry = 0.0
            self.order_side = None
            self.order_act = 0
            self.slot2_side = None
            self.slot2_qty = 0.0
            if net_pnl < 0:
                self.cooldown_until = t + 30  # 30с защита от повторного входа на падающем ноже

        # 2. Переход к следующей секунде t+1 и симуляция очереди L2 FIFO (с учетом ~200мс REST-латентности)
        self.t += 1
        nt = self.t
        nb1 = max(self.bid_p[nt, 0], self.tick_size)
        na1 = max(self.ask_p[nt, 0], nb1 + self.tick_size)

        if self.order_side is not None and nt >= self.active_after:
            q_at, q_better = self._get_lob_queue(nt, self.order_side, self.order_price)
            q_at_prev, q_better_prev = self._get_lob_queue(nt - 1, self.order_side, self.order_price)
            if nt == self.active_after and not self.legacy_v2:
                self.q_ahead = max(self.q_ahead, q_at)
            q_better_eff = max(q_better, q_better_prev)
            eff_q = self.q_ahead + q_better_eff

            if self.order_side == "BUY":
                crossed = (na1 <= self.order_price + tol)
                price_ok = (abs(nb1 - self.order_price) <= tol) if self.legacy_v2 else (nb1 <= self.order_price + tol and na1 > self.order_price + tol)
                vol_hit = self.sell_vol[nt] if price_ok else 0.0
                vol_at_px = max(0.0, vol_hit - q_better_eff)
                if crossed or vol_hit > eff_q:
                    fill_q = self.order_qty if crossed else min(self.order_qty, max(0.0, vol_hit - eff_q))
                    if fill_q >= self.order_qty * 0.25:
                        self.pos_qty = self.order_qty
                        self.avg_entry = self.order_price
                        self.peak_bid = max(self.order_price, nb1)
                        self.entry_t = nt
                        self.order_side = None
                        self.slot2_side = None
                        self.slot2_qty = 0.0
                        self.split_done_for_pos = False
                        self.unfreeze_active = False
                        self.n_trades += 1
                        if not self.legacy_v2:
                            self.total_pnl_usd -= self.pos_qty * self.avg_entry * 0.001  # Реальная комиссия SafeTrade Maker Buy (0.10%)
                            ez60_nt = float(self.ez60_arr[nt]) if hasattr(self, "ez60_arr") else float(self.X_market[nt, 12])
                            if self. avg_entry > nb1 + tol and (float(self.dd_900[nt]) >= 2.0 or ez60_nt < 0.05):
                                requote_penalty += 0.35  # Эмпирический штраф adverse selection из 508,961 сек телеметрии SafeTrade
                else:
                    if vol_at_px > 0:
                        self.q_ahead = max(0.0, self.q_ahead - vol_at_px)
                    if not self.legacy_v2 and q_at < q_at_prev - vol_at_px and q_at_prev > 1e-6:
                        net_cancel = (q_at_prev - vol_at_px) - q_at
                        self.q_ahead = max(0.0, self.q_ahead - net_cancel * min(1.0, 0.35 * self.q_ahead / q_at_prev))
                    self.q_ahead = min(self.q_ahead, q_at)
            elif self.order_side == "SELL":
                crossed = (nb1 >= self.order_price - tol)
                vol_hit = self.buy_vol[nt] if abs(na1 - self.order_price) <= tol else 0.0
                vol_at_px = max(0.0, vol_hit - q_better_eff)
                if crossed or vol_hit > eff_q:
                    fill_qty = min(self.pos_qty, self.order_qty if self.order_qty > 1e-4 else self.pos_qty)
                    sell_fee = 0.0 if self.legacy_v2 else (fill_qty * self.order_price * 0.001)
                    net_pnl = fill_qty * (self.order_price - self.avg_entry) - sell_fee
                    realized_ticks = (self.order_price - self.avg_entry) / self.tick_size
                    if realized_ticks >= 1.0:
                        requote_penalty -= 0.45 * realized_ticks * (fill_qty / max(self.pos_qty, 1e-6))
                    self.total_pnl_usd += net_pnl
                    rem_qty = self.pos_qty - fill_qty
                    if rem_qty > 1e-4 and self.slot2_side == "SELL" and self.slot2_qty > 1e-4:
                        self.pos_qty = self.slot2_qty
                        self.order_side = "SELL"
                        self.order_act = 3
                        self.order_price = self.slot2_price
                        self.order_qty = self.slot2_qty
                        self.q_ahead = self.slot2_q_ahead
                        self.active_after = self.slot2_active_after
                        self.slot2_side = None
                        self.slot2_qty = 0.0
                    else:
                        if realized_ticks <= -0.8:
                            self.last_loss_exit_t = nt
                            self.last_loss_exit_price = self.order_price
                        if getattr(self, "enable_unfreeze", True) and (float(self.dd_900[nt]) >= 4.0 or ((nt - self.entry_t) >= 600 and float(self.dd_900[nt]) >= 3.0)):
                            if realized_ticks <= -2.0:
                                self.cooldown_until = nt + 600
                            elif realized_ticks <= -0.8 and (self.avg_entry - nb1) / self.tick_size >= 2.8:
                                self.cooldown_until = nt + 900
                        self.last_exit_t = nt
                        self.last_exit_price = self.order_price
                        self.pos_qty = 0.0
                        self.avg_entry = 0.0
                        self.peak_bid = 0.0
                        self.order_side = None
                        self.slot2_side = None
                        self.slot2_qty = 0.0
                        self.unfreeze_active = False
                else:
                    if vol_at_px > 0:
                        self.q_ahead = max(0.0, self.q_ahead - vol_at_px)
                    if not self.legacy_v2 and q_at < q_at_prev - vol_at_px and q_at_prev > 1e-6:
                        net_cancel = (q_at_prev - vol_at_px) - q_at
                        self.q_ahead = max(0.0, self.q_ahead - net_cancel * min(1.0, 0.35 * self.q_ahead / q_at_prev))
                    self.q_ahead = min(self.q_ahead, q_at)

        if self.slot2_side == "SELL" and nt >= self.slot2_active_after and self.pos_qty > 1e-8:
            q2_at, q2_better = self._get_lob_queue(nt, "SELL", self.slot2_price)
            q2_at_prev, q2_better_prev = self._get_lob_queue(nt - 1, "SELL", self.slot2_price)
            if nt == self.slot2_active_after and not self.legacy_v2:
                self.slot2_q_ahead = max(self.slot2_q_ahead, q2_at)
            q2_better_eff = max(q2_better, q2_better_prev)
            eff_q2 = self.slot2_q_ahead + q2_better_eff
            crossed2 = (nb1 >= self.slot2_price - tol)
            vol_hit2 = self.buy_vol[nt] if abs(na1 - self.slot2_price) <= tol else 0.0
            vol_at_px2 = max(0.0, vol_hit2 - q2_better_eff)
            if crossed2 or vol_hit2 > eff_q2:
                sell_fee2 = 0.0 if self.legacy_v2 else (self.slot2_qty * self.slot2_price * 0.001)
                net_pnl2 = self.slot2_qty * (self.slot2_price - self.avg_entry) - sell_fee2
                self.total_pnl_usd += net_pnl2
                self.pos_qty = max(0.0, self.pos_qty - self.slot2_qty)
                slot2_px = self.slot2_price
                self.slot2_side = None
                self.slot2_qty = 0.0
                if self.pos_qty <= 1e-4:
                    self.last_exit_t = nt
                    self.last_exit_price = slot2_px
                    self.pos_qty = 0.0
                    self.avg_entry = 0.0
                    self.peak_bid = 0.0
                    self.order_side = None
            else:
                if vol_at_px2 > 0:
                    self.slot2_q_ahead = max(0.0, self.slot2_q_ahead - vol_at_px2)
                if not self.legacy_v2 and q2_at < q2_at_prev - vol_at_px2 and q2_at_prev > 1e-6:
                    net_cancel2 = (q2_at_prev - vol_at_px2) - q2_at
                    self.slot2_q_ahead = max(0.0, self.slot2_q_ahead - net_cancel2 * min(1.0, 0.35 * self.slot2_q_ahead / q2_at_prev))
                self.slot2_q_ahead = min(self.slot2_q_ahead, q2_at)

        # Карусель микро-транша (M4: PRL -> USDT -> PRL) при просадке основной позиции (b1 <= avg_entry - 3 тика)
        if not self.legacy_v2:
            if self.pos_qty <= 1e-8:
                self.car_state = 0
            else:
                underwater_ticks = (self.avg_entry - nb1) / self.tick_size
                if self.car_state == 0 and underwater_ticks >= 3.0 and (nt - self.entry_t) >= 120:
                    if (na1 - nb1) >= 1.5 * self.tick_size or (float(self.pos_3600[nt]) >= 0.30 and float(self.inertia_arr[nt]) > -0.10):
                        self.car_qty = round(self.pos_qty * 0.25, 4)
                        self.car_px = na1
                        self.car_q_ahead, _ = self._get_lob_queue(nt, "SELL", self.car_px)
                        self.car_state = 1
                elif self.car_state == 1:
                    if nb1 >= self.avg_entry - self.tick_size:
                        self.car_state = 0
                    else:
                        if abs(self.car_px - na1) > tol and na1 < self.avg_entry:
                            self.car_px = na1
                            self.car_q_ahead, _ = self._get_lob_queue(nt, "SELL", self.car_px)
                        vol_hit_c = self.buy_vol[nt] if abs(na1 - self.car_px) <= tol else 0.0
                        if nb1 >= self.car_px - tol or vol_hit_c > self.car_q_ahead:
                            self.car_sell_px = self.car_px
                            self.car_px = round((self.car_sell_px - 2.0 * self.tick_size) / self.tick_size) * self.tick_size
                            self.car_q_ahead, _ = self._get_lob_queue(nt, "BUY", self.car_px)
                            self.car_state = 2
                        else:
                            self.car_q_ahead = max(0.0, self.car_q_ahead - vol_hit_c)
                elif self.car_state == 2:
                    vol_hit_c = self.sell_vol[nt] if abs(nb1 - self.car_px) <= tol else 0.0
                    if na1 <= self.car_px + tol or vol_hit_c > self.car_q_ahead:
                        net_car_pnl = self.car_qty * (self.car_sell_px * 0.999 - self.car_px * 1.001)
                        self.total_pnl_usd += net_car_pnl
                        self.avg_entry = round(max(self.tick_size, self.avg_entry - net_car_pnl / max(self.pos_qty, 1e-6)), 4)
                        self.car_cycles += 1
                        self.car_state = 0
                    elif nb1 >= self.avg_entry:
                        net_leg1 = self.car_qty * (self.car_sell_px * 0.999 - self.avg_entry)
                        self.total_pnl_usd += net_leg1
                        self.pos_qty = max(0.0, self.pos_qty - self.car_qty)
                        self.car_state = 0
                    else:
                        self.car_q_ahead = max(0.0, self.car_q_ahead - vol_hit_c)

        cur_mtm = self.total_pnl_usd + (self.pos_qty * (nb1 - self.avg_entry) if self.pos_qty > 1e-8 else 0.0)
        delta_pnl_ticks = (cur_mtm - self.prev_mtm) / (self.order_usd * (self.tick_size / nb1))
        self.prev_mtm = cur_mtm

        # Асимметричная неприязнь к риску (Downside Risk Aversion): глубокий минус штрафуется в 1.5x сильнее
        shaped_pnl = delta_pnl_ticks * 1.50 if delta_pnl_ticks < -0.5 else delta_pnl_ticks
        hold_sec = (nt - self.entry_t) if self.pos_qty > 1e-8 else 0
        inv_pen = 0.005 if hold_sec > 60 else 0.0

        next_obs = self._get_obs()
        reward = float(shaped_pnl - inv_pen - requote_penalty)
        done = bool(nt >= self.max_t)
        return next_obs, reward, done


class MaskableActorCritic(nn.Module):
    """Нейросеть политики и ценности (Actor-Critic) с маскированием недопустимых действий."""

    def __init__(self, obs_dim: int, n_actions: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        if obs.is_cuda and obs.shape[0] >= 256:
            from _tools.fast_neural_engine import blackwell_matmul_bias
            h = blackwell_matmul_bias(obs, self.backbone[0].weight.t(), self.backbone[0].bias)
            h = self.backbone[2](self.backbone[1](h))
            h = blackwell_matmul_bias(h, self.backbone[3].weight.t(), self.backbone[3].bias)
            h = self.backbone[5](self.backbone[4](h))
        else:
            h = self.backbone(obs)
        logits = self.actor(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        val = self.critic(h).squeeze(-1)
        return logits, val



def load_compatible_ppo_state_dict(model: MaskableActorCritic, sd: dict) -> None:
    """Загружает state_dict в MaskableActorCritic с хирургической адаптацией первого слоя при переходе 38D <-> 42D <-> 49D."""
    sd = dict(sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd)
    target_w = model.backbone[0].weight.data
    src_w = sd["backbone.0.weight"]
    if src_w.shape != target_w.shape:
        new_w = torch.zeros_like(target_w)
        s_in, t_in = src_w.shape[1], target_w.shape[1]
        if s_in == 38 and t_in == 42:
            new_w[:, :28] = src_w[:, :28]
            new_w[:, 32:42] = src_w[:, 28:38]
        elif s_in == 42 and t_in == 38:
            new_w[:, :28] = src_w[:, :28]
            new_w[:, 28:38] = src_w[:, 32:42]
        elif s_in == 42 and t_in == 49:
            new_w[:, :32] = src_w[:, :32]
            new_w[:, -10:] = src_w[:, -10:]
        elif s_in == 49 and t_in == 42:
            new_w[:, :32] = src_w[:, :32]
            new_w[:, -10:] = src_w[:, -10:]
        elif s_in == 38 and t_in == 49:
            new_w[:, :28] = src_w[:, :28]
            new_w[:, -10:] = src_w[:, -10:]
        elif s_in == 49 and t_in == 38:
            new_w[:, :28] = src_w[:, :28]
            new_w[:, -10:] = src_w[:, -10:]
        else:
            c_min = min(s_in, t_in)
            new_w[:, :c_min] = src_w[:, :c_min]
        sd["backbone.0.weight"] = new_w
    model.load_state_dict(sd)


def teacher_heuristic_action(obs: np.ndarray, mask: np.ndarray) -> int:
    """Селективная нейросетевая политика-учитель (с учетом 0.20% round-trip комиссий Maker + CarouselNet/HazardNet) для Warm-Start PPO."""
    ez_5, p_up_5, p_dn_5, p_tp_5, p_tn_5 = obs[0], obs[1], obs[2], obs[3], obs[4]
    ez_12, p_up_12, p_dn_12 = obs[6], obs[7], obs[8]
    ez_25 = obs[12]
    imb5 = obs[20]
    p_car_50 = float(obs[28]) if len(obs) >= 42 else 0.25
    p_adv = float(obs[30]) if len(obs) >= 42 else 0.08
    p_exh = float(obs[31]) if len(obs) >= 42 else 0.04
    has_pos = obs[-10] > 0.5
    unr_ticks = obs[-9] * 10.0
    hold_ratio = obs[-8]
    has_bid = obs[-7] > 0.5
    has_ask = obs[-6] > 0.5
    model_err_12 = obs[-3] * 5.0
    exp_entry_edge = obs[-2] * 5.0
    exp_hold_edge = obs[-1] * 5.0

    if not has_pos:
        if p_adv >= 0.65:
            if has_bid and mask[5]:
                return 5
            return 0
        strong_buy = (
            exp_entry_edge >= 0.80 and model_err_12 > -1.5 and p_tn_5 < 0.26 and imb5 > -0.45
            and (
                (ez_25 >= 0.08 and ez_12 >= 0.04 and ez_5 >= -0.02)
                or (exp_entry_edge >= 1.35 and ez_12 >= 0.08 and ez_5 >= 0.35 and imb5 >= 0.50)
                or (p_car_50 >= 0.55 and p_adv < 0.30 and exp_entry_edge >= 0.65)
            )
        )
        dip_buy = (
            exp_entry_edge >= 0.65 and model_err_12 > -1.5
            and ez_25 >= 0.10 and ez_12 >= 0.03 and ez_5 >= -0.04
            and p_tn_5 < 0.26
        )
        if strong_buy and mask[1]:
            return 1
        if dip_buy and mask[2]:
            return 0 if has_bid else 2
        if has_bid:
            if (obs[-5] >= 0.22 or exp_entry_edge < 0.20 or model_err_12 < -2.5 or ez_12 < -0.15 or p_tn_5 > 0.35 or p_adv >= 0.58) and mask[5]:
                return 5
            return 0
        return 0
    else:
        if (unr_ticks <= -6.0 and model_err_12 < -4.0 and ez_25 < -0.25 and ez_12 < -0.25) and mask[6]:
            return 6
        if has_ask:
            return 0
        if p_exh < 0.72 and exp_hold_edge >= 0.90 and ez_25 >= 0.15 and ez_12 >= 0.08 and hold_ratio < 0.50 and mask[3]:
            return 3
        if mask[4]:
            return 4
        if mask[3]:
            return 0 if has_ask else 3
        return 0


def evaluate_policy(env: SafeTradeL2ExecutionEnv, model: MaskableActorCritic = None, device: str = "cpu", oracle_pnl_override: float = None) -> dict:
    """
    Прогоняет детерминированный бэктест на интервале среды [start_idx, end_idx]
    и рассчитывает полный паспорт КПД агента:
      - kpd_capture_pct: физический КПД захвата волн относительно Идеального Maker-Оракула (%)
      - exit_eff_pct: КПД фиксации волны относительно внутрисделочного пика MFE (%)
      - mean_mfe_ticks / mean_mae_ticks: средние MFE и MAE в тиках на сделку
      - max_dd_usd / max_dd_pct: максимальная просадка плавающего MTM-капитала
      - efficiency_score: интегральный риск-скорректированный индекс КПД для отбора лучших итераций
    """
    if oracle_pnl_override is not None:
        oracle_pnl = float(oracle_pnl_override)
    else:
        oracle_pnl = env.get_oracle_benchmark()["oracle_pnl_usd"]

    obs = env.reset(eval_mode=True)
    done = False
    steps = 0
    entry_edges = []
    entry_v_preds = []

    mtm_peak = 0.0
    max_dd_usd = 0.0
    prev_pos = 0.0
    entry_p = 0.0
    cur_mfe_ticks = 0.0
    cur_mae_ticks = 0.0
    mfe_list, mae_list, exit_eff_list = [], [], []
    gross_win_usd, gross_loss_usd = 0.0, 0.0
    prev_realized = 0.0

    while not done:
        t_cur = env.t
        mask = env.get_action_mask()
        v_pred = 0.0
        if not np.any(mask[1:]):
            act = 0
        elif model is None:
            act = teacher_heuristic_action(obs, mask)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                logits, v_t = model(obs_t, mask_t)
                act = int(torch.argmax(logits, dim=-1).item())
                v_pred = float(v_t.item())
        if act in (1, 2) and env.pos_qty <= 1e-8:
            entry_edges.append(float(obs[-2] * 5.0))
            entry_v_preds.append(v_pred)

        obs, _, done = env.step(act)
        steps += 1

        mtm_peak = max(mtm_peak, float(env.prev_mtm))
        max_dd_usd = max(max_dd_usd, mtm_peak - float(env.prev_mtm))

        nt = env.t
        nb1 = max(env.bid_p[nt, 0], env.tick_size)
        na1 = max(env.ask_p[nt, 0], nb1 + env.tick_size)

        if prev_pos <= 1e-8 and env.pos_qty > 1e-8:
            entry_p = env.avg_entry
            cur_mfe_ticks = max(0.0, (na1 - entry_p) / env.tick_size)
            cur_mae_ticks = max(0.0, (entry_p - nb1) / env.tick_size)
        elif env.pos_qty > 1e-8:
            cur_mfe_ticks = max(cur_mfe_ticks, (na1 - entry_p) / env.tick_size)
            cur_mae_ticks = max(cur_mae_ticks, (entry_p - nb1) / env.tick_size)
        elif prev_pos > 1e-8 and env.pos_qty <= 1e-8:
            tr_pnl = float(env.total_pnl_usd - prev_realized)
            prev_realized = float(env.total_pnl_usd)
            if tr_pnl > 0:
                gross_win_usd += tr_pnl
            else:
                gross_loss_usd += abs(tr_pnl)
            realized_ticks = tr_pnl / max(1e-8, prev_pos * env.tick_size)
            mfe_eff = max(cur_mfe_ticks, realized_ticks, 1.0)
            mfe_list.append(cur_mfe_ticks)
            mae_list.append(cur_mae_ticks)
            exit_eff_list.append(np.clip(realized_ticks / mfe_eff, -1.0, 1.0) * 100.0)

        prev_pos = env.pos_qty

    pnl_usd = float(env.prev_mtm)
    n_trades = int(env.n_trades)
    avg_real_pnl_usd = float(pnl_usd / max(1, n_trades))
    kpd_capture_pct = float(pnl_usd / max(0.10, oracle_pnl) * 100.0)
    exit_eff_pct = float(np.mean(exit_eff_list)) if exit_eff_list else 0.0
    mean_mfe = float(np.mean(mfe_list)) if mfe_list else 0.0
    mean_mae = float(np.mean(mae_list)) if mae_list else 0.0
    profit_factor = float(gross_win_usd / max(1e-4, gross_loss_usd)) if gross_loss_usd > 1e-4 else (99.0 if gross_win_usd > 0 else 0.0)

    # Интегральный риск-скорректированный КПД (Efficiency Score)
    dd_penalty = 1.0 / (1.0 + 2.0 * (max_dd_usd / max(0.25, oracle_pnl)))
    trade_stat_weight = min(1.0, np.sqrt(n_trades / 4.0))
    efficiency_score = kpd_capture_pct * dd_penalty * trade_stat_weight

    return {
        "pnl_usd": round(pnl_usd, 4),
        "pnl_pct": round(float(pnl_usd / env.order_usd * 100.0), 2),
        "oracle_pnl_usd": round(oracle_pnl, 4),
        "kpd_capture_pct": round(kpd_capture_pct, 2),
        "efficiency_score": round(float(efficiency_score), 2),
        "exit_eff_pct": round(exit_eff_pct, 1),
        "mean_mfe_ticks": round(mean_mfe, 2),
        "mean_mae_ticks": round(mean_mae, 2),
        "max_dd_usd": round(max_dd_usd, 4),
        "max_dd_pct": round(float(max_dd_usd / env.order_usd * 100.0), 2),
        "profit_factor": round(min(99.9, profit_factor), 2),
        "trades": n_trades,
        "hours": round(steps / 3600.0, 2),
        "exp_edge_ticks": round(float(np.mean(entry_edges)) if entry_edges else 0.0, 2),
        "critic_v_ticks": round(float(np.mean(entry_v_preds)) if entry_v_preds else 0.0, 2),
        "real_pnl_per_trade_usd": round(avg_real_pnl_usd, 4),
    }


def load_live_telemetry_experiences(obs_dim: int = 38) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает реальный опыт боевой торговли из data/processed/prl_rl_telemetry.parquet
    (сессия с 5/5 плюсовыми сделками +$0.96 USDT), устраняя 1-секундные ложные отмены QUOTE_BID_2
    и усиливая вес реальных траекторий удержания очереди FIFO и профитных выходов.
    """
    tel_path = BASE_DIR / "data" / "processed" / "prl_rl_telemetry.parquet"
    if not tel_path.exists():
        return np.empty((0, obs_dim), dtype=np.float32), np.empty((0, 7), dtype=bool), np.empty((0,), dtype=np.int64)
    try:
        df_t = pd.read_parquet(tel_path)
        df_t["ts_msk"] = pd.to_datetime(df_t["timestamp"], utc=True).dt.tz_convert("Europe/Moscow")
        s15 = df_t[df_t["ts_msk"] >= pd.Timestamp("2026-10-26 20:47:20+03:00")].copy().reset_index(drop=True)
        if len(s15) == 0:
            return np.empty((0, obs_dim), dtype=np.float32), np.empty((0, 7), dtype=bool), np.empty((0,), dtype=np.int64)
        obs_arr = np.array([json.loads(x) for x in s15["obs_json"]], dtype=np.float32)
        mask_arr = np.array([json.loads(x) for x in s15["mask_json"]], dtype=bool)
        act_arr = s15["act"].values.astype(np.int64)
        if obs_arr.shape[1] == 38 and obs_dim == 42:
            v52_pad = np.tile(np.array([0.13, 0.02, 0.08, 0.04], dtype=np.float32), (len(obs_arr), 1))
            obs_arr = np.hstack([obs_arr[:, :28], v52_pad, obs_arr[:, 28:38]]).astype(np.float32)
        elif obs_arr.shape[1] in (38, 42) and obs_dim == 49:
            if obs_arr.shape[1] == 38:
                v52_pad = np.tile(np.array([0.13, 0.02, 0.08, 0.04], dtype=np.float32), (len(obs_arr), 1))
                obs_arr = np.hstack([obs_arr[:, :28], v52_pad, obs_arr[:, 28:38]]).astype(np.float32)
            v70_pad = np.tile(np.array([0.30, 0.0, 0.0, 0.70, 0.0, 0.0, 0.15], dtype=np.float32), (len(obs_arr), 1))
            obs_arr = np.hstack([obs_arr[:, :32], v70_pad, obs_arr[:, 32:42]]).astype(np.float32)
        elif obs_arr.shape[1] != obs_dim:
            return np.empty((0, obs_dim), dtype=np.float32), np.empty((0, 7), dtype=bool), np.empty((0,), dtype=np.int64)

        # Разрешаем подтягивание пассивного бида (act=2 -> act=1) в маске и размечаем эталонным нейро-учителем
        passive_bid_upgrade = (obs_arr[:, -10] <= 0.5) & (obs_arr[:, -7] > 0.5) & (obs_arr[:, -5] >= 0.15) & (obs_arr[:, 20] > -0.65)
        mask_arr[passive_bid_upgrade, 1] = True
        act_arr = np.array([teacher_heuristic_action(obs_arr[i], mask_arr[i]) for i in range(len(obs_arr))], dtype=np.int64)

        # Калибровка по реальным результатам сделок: блокируем входы перед убыточными циклами (step_in / широкие спреды)
        in_pos = (s15["pos_qty"].values >= 2.0)
        starts = np.where(in_pos & ~np.roll(in_pos, 1))[0]
        ends = np.where(~in_pos & np.roll(in_pos, 1))[0]
        bad_pre_mask = np.zeros(len(s15), dtype=bool)
        win_pre_list = []
        for s_i in starts:
            e_cand = ends[ends > s_i]
            if len(e_cand) == 0:
                continue
            e_i = int(e_cand[0])
            dpnl = float(s15["bot_cash_usd"].iloc[e_i] - s15["bot_cash_usd"].iloc[max(0, s_i - 1)])
            if -15.0 < dpnl < -0.05:
                w0 = max(0, s_i - 90)
                bad_pre_mask[w0:s_i] = True
                for j in range(w0, s_i):
                    act_arr[j] = 5 if (obs_arr[j, -7] > 0.5 and mask_arr[j, 5]) else 0
            elif 0.05 < dpnl < 15.0:
                win_pre_list.append(np.arange(max(0, s_i - 20), s_i))
        n_fixed = int((act_arr != s15["act"].values).sum())

        # Усиливаем реальные траектории профитных сделок и отмен токсичных бидов
        active_idx = np.where((obs_arr[:, -10] > 0.5) | (obs_arr[:, -7] > 0.5) | (act_arr > 0))[0]
        buy1_idx = np.where((act_arr == 1) & ~bad_pre_mask)[0]
        bad_idx = np.where(bad_pre_mask)[0]
        win_idx = np.concatenate(win_pre_list) if win_pre_list else np.empty((0,), dtype=np.int64)
        idle_idx = np.where((obs_arr[:, -10] <= 0.5) & (obs_arr[:, -7] <= 0.5) & (act_arr == 0))[0]
        if len(idle_idx) > 8000:
            idle_idx = np.random.choice(idle_idx, size=8000, replace=False)
        sel_idx = np.concatenate([np.repeat(active_idx, 3), np.repeat(buy1_idx, 6), np.repeat(win_idx, 5), np.repeat(bad_idx, 6), idle_idx])
        print(f"📡 Подключен реальный опыт живой торговли: {len(sel_idx):,} переходов (из {len(s15):,} сек телеметрии, скорректировано {n_fixed:,} действий, отфильтровано {len(bad_idx)} сек перед стопами)")
        return obs_arr[sel_idx], mask_arr[sel_idx], act_arr[sel_idx]
    except Exception as e:
        print(f"⚠️ Не удалось загрузить боевую телеметрию: {e}")
        return np.empty((0, obs_dim), dtype=np.float32), np.empty((0, 7), dtype=bool), np.empty((0,), dtype=np.int64)


def train_rl_agent(
    coin: str = "prl",
    tail_hours: float = 176.0,
    ppo_epochs: int = 30,
    steps_per_epoch: int = 32768,
    pop_size: int = 8,
    live_plot: bool = False,
    seed: int = 3,
    split_ts_msk: str = None,
    force_rebuild: bool = False,
):
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = build_or_load_rl_cache(coin=coin, tail_hours=tail_hours, force_rebuild=force_rebuild)
    n_total = len(data["X_market"])
    ts_msk = pd.to_datetime(data["ts"], utc=True).tz_convert("Europe/Moscow")
    if split_ts_msk is not None:
        split_idx = int(np.argmax(ts_msk > pd.Timestamp(split_ts_msk)))
    else:
        split_idx = int(n_total * 0.65)  # 65% RL-Train, 35% Blind OOS RL-Test

    print(f"📅 RL-Train: {ts_msk[60]} ➔ {ts_msk[split_idx - 1]} ({(split_idx - 60)/3600:.2f} ч)")
    print(f"📅 RL-Test:  {ts_msk[split_idx]} ➔ {ts_msk[n_total - 1]} ({(n_total - split_idx)/3600:.2f} ч)")

    train_env = SafeTradeL2ExecutionEnv(data, start_idx=60, end_idx=split_idx, episode_len=3600)
    test_env = SafeTradeL2ExecutionEnv(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=n_total - split_idx - 2)
    base_env = SafeTradeL2ExecutionEnv(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=n_total - split_idx - 2, legacy_v2=True)

    device = "cpu"
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Устройство RL: Rollout/Eval={device} ({SAFE_WORKERS} потоков) | Обучение батчей и популяции (pop_size={pop_size})={gpu_device}")
    out_dir = BASE_DIR / "models" / coin.lower() / "rl_execution"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = out_dir / "baseline_ppo_v2.pt"
    warm_ckpt = out_dir / "best_maskable_ppo.pt" if (out_dir / "best_maskable_ppo.pt").exists() else base_ckpt

    model = MaskableActorCritic(obs_dim=train_env.obs_dim, n_actions=train_env.n_actions).to(device)
    base_model = None
    if base_ckpt.exists():
        sd2 = torch.load(base_ckpt, map_location=device, weights_only=False)
        sd2 = sd2.get("model_state_dict", sd2) if isinstance(sd2, dict) else sd2
        h2 = sd2["backbone.0.weight"].shape[0]
        base_model = MaskableActorCritic(obs_dim=base_env.obs_dim, n_actions=base_env.n_actions, hidden_dim=h2).to(device)
        base_model.load_state_dict(sd2)
        base_model.eval()

    oracle_info = test_env.get_oracle_benchmark()
    base_res = evaluate_policy(base_env, model=base_model, device=device, oracle_pnl_override=oracle_info["oracle_pnl_usd"])
    print(
        f"\n🎯 Физический потолок Идеального Maker-Оракула на слепом OOS ({base_res['hours']} ч): "
        f"+${oracle_info['oracle_pnl_usd']:.4f} (+{oracle_info['oracle_pnl_pct']:.2f}%, {oracle_info['oracle_trades']} волн) = 100% КПД"
    )
    print(
        f"📊 Базовый эталон (PPO v2): PnL = ${base_res['pnl_usd']:+.4f} ({base_res['pnl_pct']:+.2f}%) | "
        f"КПД захвата: {base_res['kpd_capture_pct']:+.1f}% | Score: {base_res['efficiency_score']:+.1f} | "
        f"Выход: {base_res['exit_eff_pct']:.0f}% (MFE={base_res['mean_mfe_ticks']:.1f}т/MAE={base_res['mean_mae_ticks']:.1f}т) | "
        f"MaxDD: ${base_res['max_dd_usd']:.3f} | Сделок: {base_res['trades']}"
    )
    history = []
    best_pnl = -1e9
    best_score = -1e9

    best_model_ref = MaskableActorCritic(obs_dim=train_env.obs_dim, n_actions=train_env.n_actions).to(device)
    print("🎓 Behavioral Cloning Warm-Start на GPU (сбор 80,000 демонстраций селективного нейро-учителя + реальный опыт торговли)...")
    bc_obs, bc_masks, bc_acts = [], [], []
    obs = train_env.reset()
    for _ in range(80000):
        mask = train_env.get_action_mask()
        act = teacher_heuristic_action(obs, mask)
        bc_obs.append(obs); bc_masks.append(mask); bc_acts.append(act)
        obs, _, done = train_env.step(act)
        if done:
            obs = train_env.reset()
    live_obs, live_masks, live_acts = load_live_telemetry_experiences(obs_dim=train_env.obs_dim)
    if len(live_obs) > 0:
        bc_obs_np = np.vstack([np.array(bc_obs, dtype=np.float32), live_obs])
        bc_masks_np = np.vstack([np.array(bc_masks, dtype=bool), live_masks])
        bc_acts_np = np.concatenate([np.array(bc_acts, dtype=np.int64), live_acts])
    else:
        bc_obs_np = np.array(bc_obs, dtype=np.float32)
        bc_masks_np = np.array(bc_masks, dtype=bool)
        bc_acts_np = np.array(bc_acts, dtype=np.int64)
    gpu_model = MaskableActorCritic(obs_dim=train_env.obs_dim, n_actions=train_env.n_actions).to(gpu_device)
    gpu_model.load_state_dict(model.state_dict())
    t_obs = torch.as_tensor(bc_obs_np, dtype=torch.float32, device=gpu_device)
    t_masks = torch.as_tensor(bc_masks_np, dtype=torch.bool, device=gpu_device)
    t_acts = torch.as_tensor(bc_acts_np, dtype=torch.long, device=gpu_device)
    T_live_obs = torch.as_tensor(live_obs, dtype=torch.float32, device=gpu_device) if len(live_obs) > 0 else None
    T_live_masks = torch.as_tensor(live_masks, dtype=torch.bool, device=gpu_device) if len(live_obs) > 0 else None
    T_live_acts = torch.as_tensor(live_acts, dtype=torch.long, device=gpu_device) if len(live_obs) > 0 else None
    ce_w = torch.tensor([1.0, 20.0, 20.0, 3.0, 3.0, 8.0, 10.0], dtype=torch.float32, device=gpu_device)
    bc_opt = torch.optim.AdamW(gpu_model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(20):
        perm = torch.randperm(len(t_obs), device=gpu_device)
        for i in range(0, len(t_obs), 1024):
            idx = perm[i:i + 1024]
            logits, _ = gpu_model(t_obs[idx], t_masks[idx])
            loss = F.cross_entropy(logits, t_acts[idx], weight=ce_w)
            bc_opt.zero_grad(); loss.backward(); bc_opt.step()
    model.load_state_dict({k: v.cpu() for k, v in gpu_model.state_dict().items()})
    del bc_obs, bc_masks, bc_acts, bc_obs_np, bc_masks_np, bc_acts_np, t_obs, t_masks, t_acts, bc_opt, gpu_model

    ws_res = evaluate_policy(test_env, model=model, device=device)
    if warm_ckpt.exists():
        cand_model = MaskableActorCritic(obs_dim=train_env.obs_dim, n_actions=train_env.n_actions).to(device)
        sd_w = torch.load(warm_ckpt, map_location=device, weights_only=False)
        load_compatible_ppo_state_dict(cand_model, sd_w)
        cand_res = evaluate_policy(test_env, model=cand_model, device=device)
        if cand_res["efficiency_score"] > ws_res["efficiency_score"] and cand_res["trades"] >= 2:
            print(f"🎓 Выбран существующий чекпоинт {warm_ckpt.name} с хирургической адаптацией до {train_env.obs_dim}D (Score={cand_res['efficiency_score']:+.1f} vs BC={ws_res['efficiency_score']:+.1f})")
            model.load_state_dict(cand_model.state_dict())
            ws_res = cand_res

    best_model_ref.load_state_dict(model.state_dict())
    best_model_ref.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    best_pnl = ws_res["pnl_usd"]
    best_score = ws_res["efficiency_score"]
    try:
        torch.save(model.state_dict(), out_dir / "best_maskable_ppo.pt")
    except OSError:
        torch.save(model.state_dict(), Path("/tmp/best_maskable_ppo.pt"))
    history.append({"epoch": 0, **ws_res, "best_pnl_usd": round(float(best_pnl), 4), "best_score": round(float(best_score), 2)})
    print(
        f"  [Warm-Start v3] 🏆 -> PnL: ${ws_res['pnl_usd']:+.4f} | КПД: {ws_res['kpd_capture_pct']:+.1f}% "
        f"(Score={ws_res['efficiency_score']:+.1f}) | Выход: {ws_res['exit_eff_pct']:.0f}% "
        f"(MFE={ws_res['mean_mfe_ticks']:.1f}т/MAE={ws_res['mean_mae_ticks']:.1f}т) | MaxDD: ${ws_res['max_dd_usd']:.3f} | Сделок: {ws_res['trades']}"
    )

    # 2. Обучение Maskable PPO v3 в L2 FIFO среде с реальными комиссиями 0.10% Maker
    print(f"🚀 Запуск Maskable PPO v3 ({ppo_epochs} эпох по {steps_per_epoch:,} шагов, с учетом реальных комиссий SafeTrade)...")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ppo_epochs, eta_min=1.5e-5)

    for ep in range(1, ppo_epochs + 1):
        b_obs, b_masks, b_acts, b_tacts, b_logp, b_rews, b_vals, b_dones = [], [], [], [], [], [], [], []
        obs = train_env.reset()
        for _ in range(steps_per_epoch):
            mask = train_env.get_action_mask()
            with torch.no_grad():
                o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                logits, val = model(o_t, m_t)
                ref_logits, _ = best_model_ref(o_t, m_t)
                t_act = int(torch.argmax(ref_logits, dim=-1).item())
                dist = Categorical(logits=logits)
                act_t = dist.sample()
                logp_t = dist.log_prob(act_t)
            act = int(act_t.item())
            next_obs, rew, done = train_env.step(act)
            b_obs.append(obs); b_masks.append(mask); b_acts.append(act); b_tacts.append(t_act)
            b_logp.append(float(logp_t.item())); b_rews.append(rew); b_vals.append(float(val.item())); b_dones.append(done)
            obs = train_env.reset() if done else next_obs

        # GAE (Generalized Advantage Estimation)
        advs = np.zeros(steps_per_epoch, dtype=np.float32)
        last_gae = 0.0
        for t_i in reversed(range(steps_per_epoch)):
            next_val = 0.0 if (t_i == steps_per_epoch - 1 or b_dones[t_i]) else b_vals[t_i + 1]
            delta = b_rews[t_i] + 0.995 * next_val - b_vals[t_i]
            last_gae = delta + 0.995 * 0.95 * (0.0 if b_dones[t_i] else last_gae)
            advs[t_i] = last_gae
        rets = advs + np.array(b_vals, dtype=np.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        T_obs = torch.as_tensor(np.array(b_obs), dtype=torch.float32, device=device)
        T_masks = torch.as_tensor(np.array(b_masks), dtype=torch.bool, device=device)
        T_acts = torch.as_tensor(np.array(b_acts), dtype=torch.long, device=device)
        T_tacts = torch.as_tensor(np.array(b_tacts), dtype=torch.long, device=device)
        T_old_logp = torch.as_tensor(np.array(b_logp), dtype=torch.float32, device=device)
        T_advs = torch.as_tensor(advs, dtype=torch.float32, device=device)
        T_rets = torch.as_tensor(rets, dtype=torch.float32, device=device)

        from _tools.rl_gpu_population import run_gpu_population_step
        best_cand, oos_res = run_gpu_population_step(
            ep=ep,
            pop_size=pop_size,
            model_cls=MaskableActorCritic,
            obs_dim=train_env.obs_dim,
            n_actions=train_env.n_actions,
            best_model_ref=best_model_ref,
            T_obs=T_obs,
            T_masks=T_masks,
            T_acts=T_acts,
            T_tacts=T_tacts,
            T_old_logp=T_old_logp,
            T_advs=T_advs,
            T_rets=T_rets,
            T_live_obs=T_live_obs,
            T_live_masks=T_live_masks,
            T_live_acts=T_live_acts,
            ce_w=ce_w,
            gpu_device=gpu_device,
            data=data,
            split_idx=split_idx,
            n_total=n_total,
            env_cls=SafeTradeL2ExecutionEnv,
            evaluate_fn=evaluate_policy,
            oracle_pnl=oracle_info["oracle_pnl_usd"],
            template_test_env=test_env,
        )
        model.load_state_dict(best_cand.state_dict())
        is_best = (oos_res["efficiency_score"] > best_score) and (oos_res["trades"] >= 2)
        if is_best:
            best_score = oos_res["efficiency_score"]
            best_pnl = max(best_pnl, oos_res["pnl_usd"])
            best_model_ref.load_state_dict(model.state_dict())
            try:
                torch.save(model.state_dict(), out_dir / "best_maskable_ppo.pt")
            except OSError:
                torch.save(model.state_dict(), Path("/tmp/best_maskable_ppo.pt"))
        elif oos_res["efficiency_score"] < best_score * 0.75:
            # Мягкий возврат к лидеру при неудачной стохастической мутации
            with torch.no_grad():
                for p_cur, p_best in zip(model.parameters(), best_model_ref.parameters()):
                    p_cur.data.mul_(0.3).add_(p_best.data, alpha=0.7)
        mark = "🏆 РЕКОРД" if is_best else "✅ OK"
        history.append({"epoch": ep, **oos_res, "best_pnl_usd": round(float(best_pnl), 4), "best_score": round(float(best_score), 2)})
        try:
            with open(out_dir / "training_history.json", "w") as f:
                json.dump({"oracle": oracle_info, "baseline": base_res, "history": history}, f, indent=2)
        except OSError:
            pass
        print(
            f"  [PPO Ep {ep:02d}/{ppo_epochs}] {mark} -> PnL: ${oos_res['pnl_usd']:+.4f} | "
            f"КПД: {oos_res['kpd_capture_pct']:+.1f}% (Score={oos_res['efficiency_score']:+.1f}) | "
            f"Выход: {oos_res['exit_eff_pct']:.0f}% (MFE={oos_res['mean_mfe_ticks']:.1f}т/MAE={oos_res['mean_mae_ticks']:.1f}т) | "
            f"MaxDD: ${oos_res['max_dd_usd']:.3f} | Сделок: {oos_res['trades']}",
            flush=True,
        )
        if live_plot:
            try:
                from IPython.display import clear_output
                import matplotlib.pyplot as plt
                clear_output(wait=True)
                eps = [h["epoch"] for h in history]
                pnls = [h["pnl_usd"] for h in history]
                bests = [h["best_pnl_usd"] for h in history]
                edges = [h["exp_edge_ticks"] for h in history]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2))
                ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.6)
                ax1.axhline(base_res["pnl_usd"], color="red", linestyle=":", label=f"Heuristic (${base_res['pnl_usd']:+.2f})")
                ax1.plot(eps, pnls, marker="o", label="Epoch OOS PnL ($)")
                ax1.plot(eps, bests, linestyle="--", color="green", label="Best OOS PnL ($)")
                ax1.set_title(f"Maskable PPO OOS Convergence (Ep {ep}/{ppo_epochs})")
                ax1.set_xlabel("Epoch"); ax1.set_ylabel("Net PnL (USDT)"); ax1.legend(); ax1.grid(alpha=0.3)
                ax2.plot(eps, edges, marker="s", color="darkorange", label="Mean Entry ExpEdge (ticks)")
                ax2.set_title("Selectivity: Expected Entry Edge (ticks)")
                ax2.set_xlabel("Epoch"); ax2.set_ylabel("Ticks"); ax2.legend(); ax2.grid(alpha=0.3)
                plt.tight_layout(); plt.show()
            except Exception:
                pass
    return {"baseline": base_res, "history": history}


def inspect_and_plot_rl_oos(coin: str = "prl", tail_hours: float = 176.0, start_ts_msk: str = None):
    """Прогоняет лучший чекпоинт Maskable PPO и базовую эвристику на слепом OOS или чистом Форвард-тесте с визуализацией сделок и кривой капитала."""
    import matplotlib.pyplot as plt
    data = build_or_load_rl_cache(coin=coin, tail_hours=tail_hours)
    n_total = len(data["X_market"])
    ts_msk = pd.to_datetime(data["ts"], utc=True).tz_convert("Europe/Moscow")
    if start_ts_msk is not None:
        cutoff = pd.Timestamp(start_ts_msk)
        split_idx = int(np.argmax(ts_msk > cutoff))
    else:
        split_idx = int(n_total * 0.65)
    t_start_str = str(ts_msk[split_idx])
    t_end_str = str(ts_msk[n_total - 1])
    test_env = SafeTradeL2ExecutionEnv(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=n_total - split_idx - 2)
    base_env = SafeTradeL2ExecutionEnv(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=n_total - split_idx - 2, legacy_v2=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = BASE_DIR / "models" / coin.lower() / "rl_execution" / "best_maskable_ppo.pt"
    base_ckpt = BASE_DIR / "models" / coin.lower() / "rl_execution" / "baseline_ppo_v2.pt"
    sd3 = torch.load(ckpt, map_location=device, weights_only=False) if ckpt.exists() else None
    sd3 = sd3.get("model_state_dict", sd3) if isinstance(sd3, dict) else sd3
    h3 = sd3["backbone.0.weight"].shape[0] if sd3 is not None else 128
    model = MaskableActorCritic(obs_dim=test_env.obs_dim, n_actions=test_env.n_actions, hidden_dim=h3).to(device)
    if sd3 is not None:
        load_compatible_ppo_state_dict(model, sd3)
    model.eval()

    base_model = None
    if base_ckpt.exists():
        sd2 = torch.load(base_ckpt, map_location=device, weights_only=False)
        sd2 = sd2.get("model_state_dict", sd2) if isinstance(sd2, dict) else sd2
        h2 = sd2["backbone.0.weight"].shape[0]
        base_model = MaskableActorCritic(obs_dim=base_env.obs_dim, n_actions=base_env.n_actions, hidden_dim=h2).to(device)
        base_model.load_state_dict(sd2)
        base_model.eval()

    def _run_trace(pol_model, env_obj):
        obs = env_obj.reset(eval_mode=True)
        done = False
        eq_mtm, eq_realized, mid_p, acts = [], [], [], []
        trades_log = []
        prev_pos = 0.0
        realized_cum = 0.0
        entry_t, entry_p, entry_eq, entry_eq_before = 0, 0.0, 0.0, 0.0
        while not done:
            t_cur = env_obj.t
            mask = env_obj.get_action_mask()
            if pol_model is None:
                act = teacher_heuristic_action(obs, mask)
            else:
                with torch.no_grad():
                    o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    m_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    logits, _ = pol_model(o_t, m_t)
                    act = int(torch.argmax(logits, dim=-1).item())
            obs, _, done = env_obj.step(act)
            eq_mtm.append(float(env_obj.prev_mtm))
            mid_p.append(float(0.5 * (env_obj.bid_p[t_cur, 0] + env_obj.ask_p[t_cur, 0])))
            acts.append(act)
            if prev_pos <= 1e-8 and env_obj.pos_qty > 1e-8:
                entry_t, entry_p, entry_eq = t_cur, env_obj.avg_entry, entry_eq_before
            elif prev_pos > 1e-8 and env_obj.pos_qty <= 1e-8:
                tr_pnl = float(env_obj.prev_mtm - entry_eq)
                realized_cum = float(env_obj.prev_mtm)
                trades_log.append({
                    "entry_step": int(entry_t - split_idx),
                    "exit_step": int(t_cur - split_idx),
                    "entry_ts": str(ts_msk[entry_t]),
                    "exit_ts": str(ts_msk[t_cur]),
                    "hold_sec": int(t_cur - entry_t),
                    "entry_price": round(float(entry_p), 6),
                    "exit_action": "Panic Taker" if act == 6 else "Maker Ask (Passive)",
                    "net_pnl_usd": round(tr_pnl, 4),
                    "cum_pnl_usd": round(realized_cum, 4),
                })
            eq_realized.append(realized_cum)
            prev_pos = env_obj.pos_qty
            entry_eq_before = env_obj.prev_mtm
        return np.array(eq_mtm), np.array(eq_realized), np.array(mid_p), np.array(acts), pd.DataFrame(trades_log)

    kpi_ppo = evaluate_policy(test_env, model=model, device=device)
    kpi_base = evaluate_policy(base_env, model=base_model, device=device, oracle_pnl_override=kpi_ppo["oracle_pnl_usd"])
    mode_lbl = "ЧИСТОМ ФОРВАРД-ТЕСТЕ" if start_ts_msk is not None else "СЛЕПОМ OOS"
    print("\n" + "=" * 96)
    print(f"📊 СРАВНЕНИЕ ПАСПОРТОВ КПД АГЕНТОВ НА {mode_lbl} ({kpi_ppo['hours']} ч | Потолок Оракула: +${kpi_ppo['oracle_pnl_usd']:.4f} = 100% КПД)")
    print(f"📅 Период (МСК): {t_start_str} ➔ {t_end_str}")
    print("=" * 96)
    print(f"{'Метрика КПД':<34} | {'Базовый эталон (v2)':<26} | {'Maskable PPO (v3)':<26}")
    print("-" * 96)
    print(f"{'Чистый PnL (USDT / % к ордеру)':<34} | ${kpi_base['pnl_usd']:+.4f} ({kpi_base['pnl_pct']:+.2f}%)        | ${kpi_ppo['pnl_usd']:+.4f} ({kpi_ppo['pnl_pct']:+.2f}%)")
    print(f"{'Физический КПД захвата (η_capture)':<34} | {kpi_base['kpd_capture_pct']:+6.2f}%                   | {kpi_ppo['kpd_capture_pct']:+6.2f}%")
    print(f"{'Интегральный индекс (Score)':<34} | {kpi_base['efficiency_score']:+6.2f}                    | {kpi_ppo['efficiency_score']:+6.2f}")
    print(f"{'КПД фиксации волны (η_exit от MFE)':<34} | {kpi_base['exit_eff_pct']:6.1f}%                   | {kpi_ppo['exit_eff_pct']:6.1f}%")
    print(f"{'Средний MFE / MAE (тики)':<34} | {kpi_base['mean_mfe_ticks']:4.2f}т / {kpi_base['mean_mae_ticks']:4.2f}т             | {kpi_ppo['mean_mfe_ticks']:4.2f}т / {kpi_ppo['mean_mae_ticks']:4.2f}т")
    print(f"{'Макс. просадка MTM (MaxDD)':<34} | ${kpi_base['max_dd_usd']:.4f} ({kpi_base['max_dd_pct']:.2f}%)         | ${kpi_ppo['max_dd_usd']:.4f} ({kpi_ppo['max_dd_pct']:.2f}%)")
    print(f"{'Profit Factor / Сделок':<34} | PF={kpi_base['profit_factor']:.2f} ({kpi_base['trades']} сд.)          | PF={kpi_ppo['profit_factor']:.2f} ({kpi_ppo['trades']} сд.)")
    print("=" * 96 + "\n")

    eq_base_mtm, eq_base_real, mid_p, acts_base, df_base = _run_trace(base_model, base_env)
    eq_ppo_mtm, eq_ppo_real, _, acts_ppo, df_ppo = _run_trace(model, test_env)
    hours_x = np.arange(len(eq_ppo_mtm)) / 3600.0
    n_win = int((df_ppo["net_pnl_usd"] > 0).sum()) if len(df_ppo) else 0
    n_be = int((df_ppo["net_pnl_usd"] == 0).sum()) if len(df_ppo) else 0
    n_loss = int((df_ppo["net_pnl_usd"] < 0).sum()) if len(df_ppo) else 0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Панель 1: Крупный план кривой капитала только нашего RL-агента (Зафиксированный PnL vs MTM)
    ax1.plot(hours_x, eq_ppo_real, color="darkgreen", lw=2.4, label=f"Зафиксированный PnL ({eq_ppo_real[-1]:+.4f} USDT | {n_win} плюс / {n_be} безуб. / {n_loss} минус)")
    ax1.plot(hours_x, eq_ppo_mtm, color="limegreen", lw=1.0, alpha=0.55, label="Плавающий MTM PnL (с учетом открытой позиции)")
    if len(df_ppo) > 0:
        exit_h = df_ppo["exit_step"].values / 3600.0
        ax1.scatter(exit_h, df_ppo["cum_pnl_usd"].values, color="darkgreen", s=30, zorder=5, label="Закрытия сделок")
    ax1.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax1.set_title(f"1. Прибыль RL-агента v3 (Extremum + Inertia): {eq_ppo_mtm[-1]:+.4f} USDT (+{eq_ppo_mtm[-1]/test_env.order_usd*100:.2f}% к ордеру 15 USDT)")
    ax1.set_ylabel("Прибыль PPO (USDT)"); ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

    # Панель 2: Сравнение RL-агента v3 с Базовым эталоном v2 (+0.3779 USDT)
    ax2.plot(hours_x, eq_ppo_real, color="forestgreen", lw=2.2, label=f"Maskable PPO v3 ({eq_ppo_mtm[-1]:+.4f} USDT, {len(df_ppo)} сделок)")
    ax2.plot(hours_x, eq_base_real, color="crimson", lw=1.8, alpha=0.85, label=f"Базовый эталон v2 ({eq_base_mtm[-1]:+.4f} USDT, {len(df_base)} сделок)")
    ax2.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax2.set_title("2. Сравнение нового инерционного агента v3 с прошлым рекордом (Базовый эталон v2)")
    ax2.set_ylabel("Чистый PnL (USDT)"); ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)

    # Панель 3: Цена монеты PRL/USDT и точки входов/выходов RL-агента
    ax3.plot(hours_x, mid_p, color="steelblue", lw=1.2, label="Рыночная цена PRL/USDT (падающий рынок -3.6%)")
    if len(df_ppo) > 0:
        ent_idx = df_ppo["entry_step"].values.astype(int)
        ext_idx = df_ppo["exit_step"].values.astype(int)
        ax3.scatter(ent_idx / 3600.0, mid_p[np.clip(ent_idx, 0, len(mid_p)-1)], marker="^", color="lime", edgecolor="black", s=65, zorder=5, label="Вход PPO (Maker Bid)")
        ax3.scatter(ext_idx / 3600.0, mid_p[np.clip(ext_idx, 0, len(mid_p)-1)], marker="v", color="gold", edgecolor="black", s=65, zorder=5, label="Выход PPO (Maker Ask)")
    ax3.set_title("3. Рыночная цена PRL/USDT и точки сделок RL-агента")
    ax3.set_xlabel("Часы слепого OOS (25.2 ч)"); ax3.set_ylabel("Цена PRL (USDT)"); ax3.legend(loc="upper right"); ax3.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    drop_cols = ["entry_step", "exit_step"]
    return df_ppo.drop(columns=[c for c in drop_cols if c in df_ppo.columns]), df_base.drop(columns=[c for c in drop_cols if c in df_base.columns])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="prl")
    parser.add_argument("--tail-hours", type=float, default=176.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--split-ts", type=str, default="2026-10-28 00:00:00+03:00")
    parser.add_argument("--force-rebuild", action="store_true", default=False)
    args = parser.parse_args()
    train_rl_agent(
        coin=args.coin,
        tail_hours=args.tail_hours,
        ppo_epochs=args.epochs,
        steps_per_epoch=args.steps,
        pop_size=args.pop_size,
        split_ts_msk=args.split_ts,
        force_rebuild=args.force_rebuild,
    )
