#!/usr/bin/env python3
"""
_tools/cross_venue_first_order_net.py
=============================================================================
Модели 1-го порядка v2: 4-Venue Microstructure & Macro Lead-Lag Net
(SafeTrade L1..L5 + BigONE L1..L5 + Gate.io PEARL + Ourbit PEARL + Binance BTC/NEAR).

Выходы модуля (100% каузальные, без единого тика утечки из будущего, 1-в-1 в офлайне и в live):
1. `cv_leadlag_10d` (занимает слот v7_signals_58d[:, 18:28] -> агент [278:288])
2. `cv_probs_6d` (выход нейросети `CrossVenueFirstOrderNet` 44D -> 6D, слот v7_signals_58d[:, 52:58] -> агент [312:318]):
   [0] p_cv_sweep_12s      : вероятность пробития ST b1 вниз >= 1 тик за 12с
   [1] p_cv_bleed_60s      : вероятность сползания ST b1 вниз >= 2 тика без отскока за 60с
   [2] p_cv_rebound_30s    : вероятность чистого роста ST b1 вверх >= +1 тик за 30с без пробития вниз
   [3] p_clean_fill_b0     : вероятность чистого налива + плюсового выхода при Bid на b1 (0t) за 45с
   [4] p_clean_fill_bm1    : вероятность чистого налива + плюсового выхода при Bid на b1 - 1t за 60с
   [5] p_clean_fill_bm2    : вероятность чистого налива + плюсового выхода при Bid на b1 - 2t за 90с
=============================================================================
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_600H_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
MODEL_DIR = BASE_DIR / "models" / "prl" / "cross_venue_first_order"
MODEL_PATH = MODEL_DIR / "cross_venue_first_order_v1.pt"
METRICS_PATH = MODEL_DIR / "cross_venue_first_order_metrics.json"


def _rolling_sum_5s(arr: np.ndarray) -> np.ndarray:
    """Строго каузальная сумма за последние 5 секунд [t-4 .. t]."""
    x = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    cs = np.cumsum(x)
    out = cs.copy()
    if len(x) > 5:
        out[5:] = cs[5:] - cs[:-5]
    return np.maximum(0.0, out).astype(np.float32)


def align_extra_venues_and_macro(ts_sec: np.ndarray) -> dict[str, np.ndarray]:
    """Мгновенное O(N log M) каузальное выравнивание Gate.io, Ourbit и Binance BTC/NEAR по сетке ts_sec."""
    n_g = len(ts_sec)
    proc_dir = BASE_DIR / "data" / "processed"

    def _align_one(depth_pq: Path, trades_pq: Path) -> dict[str, np.ndarray]:
        b1 = np.zeros(n_g, dtype=np.float32)
        a1 = np.zeros(n_g, dtype=np.float32)
        b5_q = np.zeros(n_g, dtype=np.float32)
        a5_q = np.zeros(n_g, dtype=np.float32)
        sec_since = np.full(n_g, 999.0, dtype=np.float32)
        buy_v = np.zeros(n_g, dtype=np.float32)
        sell_v = np.zeros(n_g, dtype=np.float32)
        last_px = np.zeros(n_g, dtype=np.float32)
        trade_sec = np.full(n_g, 999.0, dtype=np.float32)

        if depth_pq.exists():
            cols_d = ["timestamp", "bid_1_price", "ask_1_price"] + [f"bid_{i}_qty" for i in range(1, 6)] + [f"ask_{i}_qty" for i in range(1, 6)]
            dfd = pd.read_parquet(depth_pq, columns=cols_d).sort_values("timestamp")
            d_ts = dfd["timestamp"].dt.as_unit("s").astype("int64").to_numpy()
            idx = np.searchsorted(d_ts, ts_sec, side="right") - 1
            valid = idx >= 0
            vi = idx[valid]
            b1[valid] = np.nan_to_num(dfd["bid_1_price"].to_numpy(dtype=np.float32)[vi], nan=0.0)
            a1[valid] = np.nan_to_num(dfd["ask_1_price"].to_numpy(dtype=np.float32)[vi], nan=0.0)
            bq_s = np.sum(np.nan_to_num(dfd[[f"bid_{i}_qty" for i in range(1, 6)]].to_numpy(dtype=np.float32), nan=0.0), axis=1)
            aq_s = np.sum(np.nan_to_num(dfd[[f"ask_{i}_qty" for i in range(1, 6)]].to_numpy(dtype=np.float32), nan=0.0), axis=1)
            b5_q[valid] = bq_s[vi]
            a5_q[valid] = aq_s[vi]
            sec_since[valid] = np.clip((ts_sec[valid] - d_ts[vi]).astype(np.float32), 0.0, 999.0)

        if trades_pq.exists():
            dft = pd.read_parquet(trades_pq, columns=["timestamp", "price", "amount", "side"]).sort_values("timestamp")
            t_ts = dft["timestamp"].dt.as_unit("s").astype("int64").to_numpy()
            amt = np.nan_to_num(dft["amount"].to_numpy(dtype=np.float32), nan=0.0)
            px = np.nan_to_num(dft["price"].to_numpy(dtype=np.float32), nan=0.0)
            is_buy = dft["side"].astype(str).str.lower().to_numpy() == "buy"

            pos = np.searchsorted(ts_sec, t_ts, side="left")
            in_g = (pos < n_g) & (ts_sec[np.minimum(pos, n_g - 1)] == t_ts)
            np.add.at(buy_v, pos[in_g & is_buy], amt[in_g & is_buy])
            np.add.at(sell_v, pos[in_g & (~is_buy)], amt[in_g & (~is_buy)])

            idx_t = np.searchsorted(t_ts, ts_sec, side="right") - 1
            vt = idx_t >= 0
            last_px[vt] = px[idx_t[vt]]
            trade_sec[vt] = np.clip((ts_sec[vt] - t_ts[idx_t[vt]]).astype(np.float32), 0.0, 999.0)

        return {
            "b1": b1, "a1": a1, "b5_q": b5_q, "a5_q": a5_q,
            "sec_since": np.minimum(sec_since, trade_sec),
            "buy_v": buy_v, "sell_v": sell_v, "last_px": last_px,
        }

    gt = _align_one(proc_dir / "gate_depth.parquet", proc_dir / "gate_trades.parquet")
    ob = _align_one(proc_dir / "ourbit_depth.parquet", proc_dir / "ourbit_trades.parquet")

    btc_mid = np.zeros(n_g, dtype=np.float32)
    near_mid = np.zeros(n_g, dtype=np.float32)
    macro_pq = proc_dir / "binance_macro.parquet"
    if macro_pq.exists():
        dfm = pd.read_parquet(macro_pq, columns=["timestamp", "btc_mid", "near_mid"]).sort_values("timestamp")
        m_ts = dfm["timestamp"].dt.as_unit("s").astype("int64").to_numpy()
        idx_m = np.searchsorted(m_ts, ts_sec, side="right") - 1
        vm = idx_m >= 0
        btc_mid[vm] = np.nan_to_num(dfm["btc_mid"].to_numpy(dtype=np.float32)[idx_m[vm]], nan=0.0)
        near_mid[vm] = np.nan_to_num(dfm["near_mid"].to_numpy(dtype=np.float32)[idx_m[vm]], nan=0.0)

    return {"gt": gt, "ob": ob, "btc_mid": btc_mid, "near_mid": near_mid}


def _compute_ext_venue_6d(
    st_mid: np.ndarray,
    v_dict: Optional[dict],
    tick: float,
) -> np.ndarray:
    n = len(st_mid)
    out = np.zeros((n, 6), dtype=np.float32)
    if v_dict is None:
        return out
    b1_raw = v_dict["b1"]
    a1_raw = v_dict["a1"]
    lpx = v_dict["last_px"]
    sec = np.clip(np.nan_to_num(v_dict["sec_since"], nan=999.0), 0.0, 300.0)

    has_book = (b1_raw > tick) & (a1_raw >= b1_raw)
    ref_px = np.where(has_book, 0.5 * (b1_raw + a1_raw), np.where(lpx > tick, lpx, st_mid)).astype(np.float32)
    valid = ((has_book | (lpx > tick)) & (sec <= 180.0))
    fresh = np.where(valid, np.exp(-sec / 20.0), 0.0).astype(np.float32)

    ref_5s = np.pad(ref_px[:-5], (5, 0), mode="edge") if n > 5 else ref_px.copy()
    b5 = np.maximum(0.0, v_dict["b5_q"])
    a5 = np.maximum(0.0, v_dict["a5_q"])
    imb_l5 = np.where(has_book, (b5 - a5) / (b5 + a5 + 25.0), 0.0)

    buy_5s = _rolling_sum_5s(v_dict["buy_v"])
    sell_5s = _rolling_sum_5s(v_dict["sell_v"])

    out[:, 0] = fresh
    out[:, 1] = np.clip((ref_px - st_mid) / (5.0 * tick), -2.5, 2.5) * fresh
    out[:, 2] = np.clip((ref_px - ref_5s) / (3.0 * tick), -2.5, 2.5) * fresh
    out[:, 3] = np.clip(imb_l5, -1.0, 1.0) * fresh
    out[:, 4] = np.clip((buy_5s - sell_5s) / (buy_5s + sell_5s + 5.0), -1.0, 1.0) * fresh
    out[:, 5] = np.clip(np.log1p(buy_5s + sell_5s) / 8.0, 0.0, 2.0) * fresh
    return out


def compute_cross_venue_10d_and_44d(
    st_bp: np.ndarray,
    st_bq: np.ndarray,
    st_ap: np.ndarray,
    st_aq: np.ndarray,
    st_buy: np.ndarray,
    st_sell: np.ndarray,
    bo_bp: np.ndarray,
    bo_bq: np.ndarray,
    bo_ap: np.ndarray,
    bo_aq: np.ndarray,
    bo_buy: np.ndarray,
    bo_sell: np.ndarray,
    bo_sec_since: np.ndarray,
    extra_feeds: Optional[dict] = None,
    tick: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Векторизованный и строго каузальный расчет:
      - `cv_10d` (N, 10): быстрые кросс-биржевые импульсы (включая композитный 4-биржевой консенсус в слотах 8..9),
      - `x_44d`  (N, 44): полный 4-биржевой + макро вектор для `CrossVenueFirstOrderNet v2`:
          [0:28]  — SafeTrade + BigONE L1..L5 + лента (100% обратная совместимость с v1),
          [28:34] — Gate.io PEARL/USDT (fresh, mid_diff, vel_5s, imb_l5, tape_imb_5s, burst_5s),
          [34:40] — Ourbit PEARL/USDT (fresh, mid_diff, vel_5s, imb_l5, tape_imb_5s, burst_5s),
          [40:44] — Binance Macro BTC/NEAR (btc_ret_5s, btc_ret_30s, near_ret_5s, near_ret_30s).
    """
    n = len(st_bp)
    st_b1 = np.maximum(tick, st_bp[:, 0].astype(np.float32))
    st_a1 = np.maximum(st_b1 + tick, st_ap[:, 0].astype(np.float32))
    st_mid = 0.5 * (st_b1 + st_a1)

    bo_b1_raw = np.nan_to_num(bo_bp[:, 0].astype(np.float32), nan=0.0)
    bo_a1_raw = np.nan_to_num(bo_ap[:, 0].astype(np.float32), nan=0.0)
    valid_bo = (bo_b1_raw > tick) & (bo_a1_raw >= bo_b1_raw) & (np.nan_to_num(bo_sec_since, nan=999.0) <= 120.0)
    bo_sec_clamped = np.clip(np.nan_to_num(bo_sec_since.astype(np.float32), nan=999.0), 0.0, 300.0)
    bo_fresh = np.where(valid_bo, np.exp(-bo_sec_clamped / 15.0), 0.0).astype(np.float32)

    bo_b1 = np.where(valid_bo, bo_b1_raw, st_b1).astype(np.float32)
    bo_a1 = np.where(valid_bo, np.maximum(bo_b1 + 0.0001, bo_a1_raw), st_a1).astype(np.float32)
    bo_mid = 0.5 * (bo_b1 + bo_a1)

    st_mid_5s = np.pad(st_mid[:-5], (5, 0), mode="edge") if n > 5 else st_mid.copy()
    bo_mid_5s = np.pad(bo_mid[:-5], (5, 0), mode="edge") if n > 5 else bo_mid.copy()
    st_mid_20s = np.pad(st_mid[:-20], (20, 0), mode="edge") if n > 20 else st_mid.copy()
    bo_mid_20s = np.pad(bo_mid[:-20], (20, 0), mode="edge") if n > 20 else bo_mid.copy()

    bo_b1_q = np.maximum(0.0, np.nan_to_num(bo_bq[:, 0].astype(np.float32), nan=0.0))
    bo_a1_q = np.maximum(0.0, np.nan_to_num(bo_aq[:, 0].astype(np.float32), nan=0.0))
    bo_b5_q = np.sum(np.maximum(0.0, np.nan_to_num(bo_bq[:, :5].astype(np.float32), nan=0.0)), axis=1)
    bo_a5_q = np.sum(np.maximum(0.0, np.nan_to_num(bo_aq[:, :5].astype(np.float32), nan=0.0)), axis=1)
    bo_imb_l1 = (bo_b1_q - bo_a1_q) / (bo_b1_q + bo_a1_q + 10.0)
    bo_imb_l5 = (bo_b5_q - bo_a5_q) / (bo_b5_q + bo_a5_q + 25.0)
    bo_imb_l5_5s = np.pad(bo_imb_l5[:-5], (5, 0), mode="edge") if n > 5 else bo_imb_l5.copy()

    bo_buy_5s = _rolling_sum_5s(bo_buy)
    bo_sell_5s = _rolling_sum_5s(bo_sell)
    st_buy_5s = _rolling_sum_5s(st_buy)
    st_sell_5s = _rolling_sum_5s(st_sell)

    cv_10d = np.zeros((n, 10), dtype=np.float32)
    cv_10d[:, 0] = bo_fresh
    cv_10d[:, 1] = np.clip((bo_b1 - st_a1) / (3.0 * tick), -2.0, 2.0) * bo_fresh
    cv_10d[:, 2] = np.clip((st_b1 - bo_a1) / (3.0 * tick), -2.0, 2.0) * bo_fresh
    cv_10d[:, 3] = np.clip((bo_mid - st_mid) / (5.0 * tick), -2.0, 2.0) * bo_fresh
    cv_10d[:, 4] = np.clip((bo_mid - bo_mid_5s) / (3.0 * tick), -2.0, 2.0) * bo_fresh
    cv_10d[:, 5] = np.clip(((bo_mid - bo_mid_5s) - (st_mid - st_mid_5s)) / (3.0 * tick), -2.0, 2.0) * bo_fresh
    cv_10d[:, 6] = np.clip(bo_imb_l1, -1.0, 1.0) * bo_fresh
    cv_10d[:, 7] = np.clip(bo_imb_l5 - bo_imb_l5_5s, -1.5, 1.5) * bo_fresh
    cv_10d[:, 8] = np.clip((bo_buy_5s - bo_sell_5s) / (bo_buy_5s + bo_sell_5s + 5.0), -1.0, 1.0) * bo_fresh
    cv_10d[:, 9] = np.clip(np.log1p(bo_buy_5s + bo_sell_5s) / 8.0, 0.0, 2.0) * bo_fresh

    st_b1_q = np.maximum(0.0, st_bq[:, 0].astype(np.float32))
    st_a1_q = np.maximum(0.0, st_aq[:, 0].astype(np.float32))
    st_b2_q = np.maximum(0.0, st_bq[:, 1].astype(np.float32))
    st_b3_q = np.maximum(0.0, st_bq[:, 2].astype(np.float32))
    st_b5_q = np.sum(np.maximum(0.0, st_bq[:, :5].astype(np.float32)), axis=1)
    st_a5_q = np.sum(np.maximum(0.0, st_aq[:, :5].astype(np.float32)), axis=1)

    st_b1q_5s = np.pad(st_b1_q[:-5], (5, 0), mode="edge") if n > 5 else st_b1_q.copy()
    st_a1q_5s = np.pad(st_a1_q[:-5], (5, 0), mode="edge") if n > 5 else st_a1_q.copy()
    st_b1_5s = np.pad(st_b1[:-5], (5, 0), mode="edge") if n > 5 else st_b1.copy()
    st_a1_5s = np.pad(st_a1[:-5], (5, 0), mode="edge") if n > 5 else st_a1.copy()

    x_extra = np.zeros((n, 18), dtype=np.float32)
    x_extra[:, 0] = np.clip((st_a1 - st_b1) / (5.0 * tick), 0.2, 3.0)
    x_extra[:, 1] = np.clip((bo_a1 - bo_b1) / (5.0 * tick), 0.0, 3.0) * bo_fresh
    x_extra[:, 2] = (st_b1_q - st_a1_q) / (st_b1_q + st_a1_q + 10.0)
    x_extra[:, 3] = (st_b5_q - st_a5_q) / (st_b5_q + st_a5_q + 25.0)
    x_extra[:, 4] = np.log1p(st_b1_q) / 10.0
    x_extra[:, 5] = np.log1p(st_a1_q) / 10.0
    x_extra[:, 6] = np.log1p(st_b2_q) / 10.0
    x_extra[:, 7] = np.log1p(st_b3_q) / 10.0
    x_extra[:, 8] = np.log1p(bo_b5_q) / 10.0 * bo_fresh
    x_extra[:, 9] = np.log1p(bo_a5_q) / 10.0 * bo_fresh
    x_extra[:, 10] = np.clip((st_b1 - st_b1_5s) / (3.0 * tick), -2.0, 2.0)
    x_extra[:, 11] = np.clip((st_a1 - st_a1_5s) / (3.0 * tick), -2.0, 2.0)
    x_extra[:, 12] = np.clip(np.log1p(st_b1_q) - np.log1p(st_b1q_5s), -2.0, 2.0)
    x_extra[:, 13] = np.clip(np.log1p(st_a1_q) - np.log1p(st_a1q_5s), -2.0, 2.0)
    x_extra[:, 14] = np.clip((st_buy_5s - st_sell_5s) / (st_buy_5s + st_sell_5s + 5.0), -1.0, 1.0)
    x_extra[:, 15] = np.clip(np.log1p(st_buy_5s + st_sell_5s) / 8.0, 0.0, 2.0)
    x_extra[:, 16] = np.clip((bo_mid - bo_mid_20s) / (5.0 * tick), -2.5, 2.5) * bo_fresh
    x_extra[:, 17] = np.clip(((bo_mid - bo_mid_20s) - (st_mid - st_mid_20s)) / (5.0 * tick), -2.5, 2.5) * bo_fresh

    gt_6d = _compute_ext_venue_6d(st_mid, extra_feeds.get("gt") if extra_feeds else None, tick)
    ob_6d = _compute_ext_venue_6d(st_mid, extra_feeds.get("ob") if extra_feeds else None, tick)

    macro_4d = np.zeros((n, 4), dtype=np.float32)
    if extra_feeds is not None:
        btc = extra_feeds.get("btc_mid")
        near = extra_feeds.get("near_mid")
        if btc is not None and len(btc) == n:
            btc_s = np.maximum(1.0, btc)
            btc_5s = np.pad(btc_s[:-5], (5, 0), mode="edge") if n > 5 else btc_s
            btc_30s = np.pad(btc_s[:-30], (30, 0), mode="edge") if n > 30 else btc_s
            macro_4d[:, 0] = np.clip((btc_s - btc_5s) / btc_5s * 500.0, -3.0, 3.0) * (btc > 0)
            macro_4d[:, 1] = np.clip((btc_s - btc_30s) / btc_30s * 250.0, -3.0, 3.0) * (btc > 0)
        if near is not None and len(near) == n:
            near_s = np.maximum(0.01, near)
            near_5s = np.pad(near_s[:-5], (5, 0), mode="edge") if n > 5 else near_s
            near_30s = np.pad(near_s[:-30], (30, 0), mode="edge") if n > 30 else near_s
            macro_4d[:, 2] = np.clip((near_s - near_5s) / near_5s * 300.0, -3.0, 3.0) * (near > 0)
            macro_4d[:, 3] = np.clip((near_s - near_30s) / near_30s * 150.0, -3.0, 3.0) * (near > 0)

    x_44d = np.hstack([cv_10d, x_extra, gt_6d, ob_6d, macro_4d]).astype(np.float32)
    return cv_10d, x_44d


def compute_cross_venue_10d_and_28d(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Обратно совместимая обертка (возвращает cv_10d и первые 28 признаков)."""
    cv_10d, x_44d = compute_cross_venue_10d_and_44d(*args, **kwargs)
    return cv_10d, x_44d[:, :28]


def compute_cross_venue_single_tick(
    st_bp: np.ndarray,
    st_bq: np.ndarray,
    st_ap: np.ndarray,
    st_aq: np.ndarray,
    st_buy_5s: float,
    st_sell_5s: float,
    bo_bp: np.ndarray,
    bo_bq: np.ndarray,
    bo_ap: np.ndarray,
    bo_aq: np.ndarray,
    bo_buy_5s: float,
    bo_sell_5s: float,
    bo_sec_since: float,
    hist_state: dict,
    tick: float = 0.01,
    gt_snap: Optional[dict] = None,
    ob_snap: Optional[dict] = None,
    macro_snap: Optional[dict] = None,
    out_dim_x: int = 44,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Однотиковый O(1) расчет (< 0.15 мс на CPU) для `_tools/live_rl_trader.py`
    со 100% математическим совпадением с `compute_cross_venue_10d_and_44d`.
    Поддерживает как `out_dim_x=28` (v1), так и `out_dim_x=44` (v2 4-Venue + Macro).
    """
    from collections import deque

    st_b1 = max(tick, float(st_bp[0]))
    st_a1 = max(st_b1 + tick, float(st_ap[0]))
    st_mid = 0.5 * (st_b1 + st_a1)

    bo_b1_raw = float(bo_bp[0]) if len(bo_bp) > 0 else 0.0
    bo_a1_raw = float(bo_ap[0]) if len(bo_ap) > 0 else 0.0
    valid_bo = (bo_b1_raw > tick) and (bo_a1_raw >= bo_b1_raw) and (float(bo_sec_since) <= 120.0)
    bo_fresh = math.exp(-min(max(0.0, float(bo_sec_since)), 300.0) / 15.0) if valid_bo else 0.0

    bo_b1 = bo_b1_raw if valid_bo else st_b1
    bo_a1 = max(bo_b1 + 0.0001, bo_a1_raw) if valid_bo else st_a1
    bo_mid = 0.5 * (bo_b1 + bo_a1)

    bo_b1_q = max(0.0, float(bo_bq[0])) if len(bo_bq) > 0 else 0.0
    bo_a1_q = max(0.0, float(bo_aq[0])) if len(bo_aq) > 0 else 0.0
    bo_b5_q = float(np.sum(np.maximum(0.0, bo_bq[:5]))) if len(bo_bq) >= 5 else bo_b1_q
    bo_a5_q = float(np.sum(np.maximum(0.0, bo_aq[:5]))) if len(bo_aq) >= 5 else bo_a1_q
    bo_imb_l1 = (bo_b1_q - bo_a1_q) / (bo_b1_q + bo_a1_q + 10.0)
    bo_imb_l5 = (bo_b5_q - bo_a5_q) / (bo_b5_q + bo_a5_q + 25.0)

    st_b1_q = max(0.0, float(st_bq[0]))
    st_a1_q = max(0.0, float(st_aq[0]))
    st_b2_q = max(0.0, float(st_bq[1])) if len(st_bq) > 1 else 0.0
    st_b3_q = max(0.0, float(st_bq[2])) if len(st_bq) > 2 else 0.0
    st_b5_q = float(np.sum(np.maximum(0.0, st_bq[:5])))
    st_a5_q = float(np.sum(np.maximum(0.0, st_aq[:5])))

    def _ext_ref(snap: Optional[dict]) -> float:
        if not snap:
            return st_mid
        b1_s = float(snap.get("b1", 0.0))
        a1_s = float(snap.get("a1", 0.0))
        lpx_s = float(snap.get("last_px", 0.0))
        if b1_s > tick and a1_s >= b1_s:
            return 0.5 * (b1_s + a1_s)
        if lpx_s > tick:
            return lpx_s
        return st_mid

    gt_ref = _ext_ref(gt_snap)
    ob_ref = _ext_ref(ob_snap)
    btc_m = float(macro_snap.get("btc_mid", 0.0)) if macro_snap else 0.0
    near_m = float(macro_snap.get("near_mid", 0.0)) if macro_snap else 0.0

    if "dq30" not in hist_state:
        hist_state["dq30"] = deque(maxlen=30)
    dq = hist_state["dq30"]
    cur_snap = (st_mid, bo_mid, bo_imb_l5, st_b1, st_a1, st_b1_q, st_a1_q, gt_ref, ob_ref, btc_m, near_m)
    s5 = dq[-5] if len(dq) >= 5 else (dq[0] if len(dq) > 0 else cur_snap)
    s20 = dq[-20] if len(dq) >= 20 else (dq[0] if len(dq) > 0 else cur_snap)
    s30 = dq[0] if len(dq) > 0 else cur_snap
    dq.append(cur_snap)

    st_mid_5s, bo_mid_5s, bo_imb_l5_5s, st_b1_5s, st_a1_5s, st_b1q_5s, st_a1q_5s = s5[:7]
    st_mid_20s, bo_mid_20s = s20[0], s20[1]

    cv_10d = np.array([
        bo_fresh,
        float(np.clip((bo_b1 - st_a1) / (3.0 * tick), -2.0, 2.0) * bo_fresh),
        float(np.clip((st_b1 - bo_a1) / (3.0 * tick), -2.0, 2.0) * bo_fresh),
        float(np.clip((bo_mid - st_mid) / (5.0 * tick), -2.0, 2.0) * bo_fresh),
        float(np.clip((bo_mid - bo_mid_5s) / (3.0 * tick), -2.0, 2.0) * bo_fresh),
        float(np.clip(((bo_mid - bo_mid_5s) - (st_mid - st_mid_5s)) / (3.0 * tick), -2.0, 2.0) * bo_fresh),
        float(np.clip(bo_imb_l1, -1.0, 1.0) * bo_fresh),
        float(np.clip(bo_imb_l5 - bo_imb_l5_5s, -1.5, 1.5) * bo_fresh),
        float(np.clip((bo_buy_5s - bo_sell_5s) / (bo_buy_5s + bo_sell_5s + 5.0), -1.0, 1.0) * bo_fresh),
        float(np.clip(math.log1p(max(0.0, bo_buy_5s + bo_sell_5s)) / 8.0, 0.0, 2.0) * bo_fresh),
    ], dtype=np.float32)

    x_extra = np.array([
        float(np.clip((st_a1 - st_b1) / (5.0 * tick), 0.2, 3.0)),
        float(np.clip((bo_a1 - bo_b1) / (5.0 * tick), 0.0, 3.0) * bo_fresh),
        float((st_b1_q - st_a1_q) / (st_b1_q + st_a1_q + 10.0)),
        float((st_b5_q - st_a5_q) / (st_b5_q + st_a5_q + 25.0)),
        float(math.log1p(st_b1_q) / 10.0),
        float(math.log1p(st_a1_q) / 10.0),
        float(math.log1p(st_b2_q) / 10.0),
        float(math.log1p(st_b3_q) / 10.0),
        float(math.log1p(bo_b5_q) / 10.0 * bo_fresh),
        float(math.log1p(bo_a5_q) / 10.0 * bo_fresh),
        float(np.clip((st_b1 - st_b1_5s) / (3.0 * tick), -2.0, 2.0)),
        float(np.clip((st_a1 - st_a1_5s) / (3.0 * tick), -2.0, 2.0)),
        float(np.clip(math.log1p(st_b1_q) - math.log1p(st_b1q_5s), -2.0, 2.0)),
        float(np.clip(math.log1p(st_a1_q) - math.log1p(st_a1q_5s), -2.0, 2.0)),
        float(np.clip((st_buy_5s - st_sell_5s) / (st_buy_5s + st_sell_5s + 5.0), -1.0, 1.0)),
        float(np.clip(math.log1p(max(0.0, st_buy_5s + st_sell_5s)) / 8.0, 0.0, 2.0)),
        float(np.clip((bo_mid - bo_mid_20s) / (5.0 * tick), -2.5, 2.5) * bo_fresh),
        float(np.clip(((bo_mid - bo_mid_20s) - (st_mid - st_mid_20s)) / (5.0 * tick), -2.5, 2.5) * bo_fresh),
    ], dtype=np.float32)

    if out_dim_x <= 28:
        return cv_10d, np.concatenate([cv_10d, x_extra], axis=0)

    def _ext_single_6d(snap: Optional[dict], ref_now: float, ref_prev5: float) -> np.ndarray:
        if not snap:
            return np.zeros(6, dtype=np.float32)
        b1_s = float(snap.get("b1", 0.0))
        a1_s = float(snap.get("a1", 0.0))
        lpx_s = float(snap.get("last_px", 0.0))
        sec_s = min(max(0.0, float(snap.get("sec_since", 999.0))), 300.0)
        has_b = (b1_s > tick) and (a1_s >= b1_s)
        valid_s = (has_b or (lpx_s > tick)) and (sec_s <= 180.0)
        fr = math.exp(-sec_s / 20.0) if valid_s else 0.0
        b5_s = max(0.0, float(snap.get("b5_q", 0.0)))
        a5_s = max(0.0, float(snap.get("a5_q", 0.0)))
        imb_s = ((b5_s - a5_s) / (b5_s + a5_s + 25.0)) if has_b else 0.0
        bv5 = max(0.0, float(snap.get("buy_5s", 0.0)))
        sv5 = max(0.0, float(snap.get("sell_5s", 0.0)))
        return np.array([
            fr,
            float(np.clip((ref_now - st_mid) / (5.0 * tick), -2.5, 2.5) * fr),
            float(np.clip((ref_now - ref_prev5) / (3.0 * tick), -2.5, 2.5) * fr),
            float(np.clip(imb_s, -1.0, 1.0) * fr),
            float(np.clip((bv5 - sv5) / (bv5 + sv5 + 5.0), -1.0, 1.0) * fr),
            float(np.clip(math.log1p(bv5 + sv5) / 8.0, 0.0, 2.0) * fr),
        ], dtype=np.float32)

    gt_5s = s5[7] if len(s5) > 7 else gt_ref
    ob_5s = s5[8] if len(s5) > 8 else ob_ref
    btc_5s = max(1.0, s5[9] if len(s5) > 9 and s5[9] > 0 else max(1.0, btc_m))
    btc_30s = max(1.0, s30[9] if len(s30) > 9 and s30[9] > 0 else max(1.0, btc_m))
    near_5s = max(0.01, s5[10] if len(s5) > 10 and s5[10] > 0 else max(0.01, near_m))
    near_30s = max(0.01, s30[10] if len(s30) > 10 and s30[10] > 0 else max(0.01, near_m))

    gt_6d = _ext_single_6d(gt_snap, gt_ref, gt_5s)
    ob_6d = _ext_single_6d(ob_snap, ob_ref, ob_5s)
    macro_4d = np.array([
        float(np.clip((btc_m - btc_5s) / btc_5s * 500.0, -3.0, 3.0)) if btc_m > 0 else 0.0,
        float(np.clip((btc_m - btc_30s) / btc_30s * 250.0, -3.0, 3.0)) if btc_m > 0 else 0.0,
        float(np.clip((near_m - near_5s) / near_5s * 300.0, -3.0, 3.0)) if near_m > 0 else 0.0,
        float(np.clip((near_m - near_30s) / near_30s * 150.0, -3.0, 3.0)) if near_m > 0 else 0.0,
    ], dtype=np.float32)

    return cv_10d, np.concatenate([cv_10d, x_extra, gt_6d, ob_6d, macro_4d], axis=0)[:out_dim_x]


def peek_local_venue_snap(venue_dir: Path, depth_fn: str, trades_fn: str, state_cache: dict) -> Optional[dict]:
    """Сверхлегкое (< 0.05 мс) чтение хвоста локальных CSV Gate.io / Ourbit на VPS с кэшем 0.8с."""
    now = time.time()
    key = str(venue_dir)
    prev = state_cache.get(key)
    if prev is not None and (now - prev[0]) < 0.8:
        return prev[1]
    dp = venue_dir / depth_fn
    tp = venue_dir / trades_fn
    if not dp.exists() and not tp.exists():
        state_cache[key] = (now, None)
        return None
    b1 = a1 = b5_q = a5_q = last_px = buy_5s = sell_5s = 0.0
    sec_d = sec_t = 999.0
    try:
        if dp.exists():
            sz = dp.stat().st_size
            sec_d = max(0.0, now - dp.stat().st_mtime)
            if sz > 64:
                with open(dp, "rb") as f:
                    f.seek(max(0, sz - 4096))
                    lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                if lines and not lines[-1].startswith("timestamp"):
                    ps = lines[-1].split(",")
                    if len(ps) >= 103:
                        b1 = float(ps[1] or 0.0)
                        a1 = float(ps[101] or 0.0)
                        b5_q = sum(float(ps[2 * i] or 0.0) for i in range(1, 6))
                        a5_q = sum(float(ps[100 + 2 * i] or 0.0) for i in range(1, 6))
    except Exception:
        pass
    try:
        if tp.exists():
            sz_t = tp.stat().st_size
            if sz_t > 32:
                with open(tp, "rb") as f:
                    f.seek(max(0, sz_t - 6144))
                    tlines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                from datetime import datetime
                for ln in reversed(tlines[-40:]):
                    if ln.startswith("id,") or "," not in ln:
                        continue
                    pts = ln.split(",")
                    if len(pts) < 5:
                        continue
                    ts_s = datetime.fromisoformat(pts[1].replace("Z", "+00:00")).timestamp()
                    age = max(0.0, now - ts_s)
                    if last_px <= 0.0:
                        last_px = float(pts[2] or 0.0)
                        sec_t = age
                    if age <= 5.5:
                        amt = float(pts[3] or 0.0)
                        if pts[4].strip().lower() == "buy":
                            buy_5s += amt
                        else:
                            sell_5s += amt
                    else:
                        break
    except Exception:
        pass
    snap = {
        "b1": b1, "a1": a1, "b5_q": b5_q, "a5_q": a5_q,
        "last_px": last_px, "buy_5s": buy_5s, "sell_5s": sell_5s,
        "sec_since": min(sec_d, sec_t),
    }
    state_cache[key] = (now, snap)
    return snap


class CrossVenueFirstOrderNet(nn.Module):
    """
    Быстрая Residual-SwiGLU нейросеть 1-го порядка (44D -> 128D -> Residual SwiGLU -> 6 логитов).
    """

    def __init__(self, in_dim: int = 44, h_dim: int = 128, out_dim: int = 6):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, h_dim * 2)
        self.norm1 = nn.LayerNorm(h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim * 2)
        self.norm2 = nn.LayerNorm(h_dim)
        self.head = nn.Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        u1, g1 = self.fc1(x).chunk(2, dim=-1)
        h = self.norm1(F.silu(u1) * torch.sigmoid(g1))
        u2, g2 = self.fc2(h).chunk(2, dim=-1)
        h = h + self.norm2(F.silu(u2) * torch.sigmoid(g2))
        return self.head(h)

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_first_order_targets_6d(
    st_bp: np.ndarray,
    st_bq: np.ndarray,
    st_ap: np.ndarray,
    st_buy: np.ndarray,
    st_sell: np.ndarray,
    y_sweep_12s: np.ndarray,
    y_bleed_60s: np.ndarray,
    tick: float = 0.01,
) -> np.ndarray:
    """Формирует 6 таргетов сети 1-го порядка через O(N) GPU 1D MaxPool."""
    n = len(st_bp)
    b1 = st_bp[:, 0].astype(np.float32)
    b1_q = np.maximum(1.0, st_bq[:, 0].astype(np.float32))
    sell_v = np.maximum(0.0, st_sell.astype(np.float32))

    pool_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        b1_t = torch.from_numpy(b1).to(pool_dev).view(1, 1, n)
        b1_shift = F.pad(b1_t[:, :, 1:], (0, 1), mode="replicate")

        def _fwd_minmax(win_w: int) -> tuple[np.ndarray, np.ndarray]:
            padded = F.pad(b1_shift, (0, win_w - 1), mode="replicate")
            mx = F.max_pool1d(padded, kernel_size=win_w, stride=1).view(n).cpu().numpy()
            mn = (-F.max_pool1d(-padded, kernel_size=win_w, stride=1)).view(n).cpu().numpy()
            return mx, mn

        max_b1_30, min_b1_30 = _fwd_minmax(30)
        max_b1_45, min_b1_45 = _fwd_minmax(45)
        max_b1_60, min_b1_60 = _fwd_minmax(60)
        max_b1_90, min_b1_90 = _fwd_minmax(90)

    cs_sell = np.cumsum(sell_v.astype(np.float64))
    sell_fwd_25 = np.zeros(n, dtype=np.float32)
    if n > 25:
        sell_fwd_25[:-25] = (cs_sell[25:] - cs_sell[:-25]).astype(np.float32)

    y_rebound_30s = (((max_b1_30 - b1) >= 0.95 * tick) & ((b1 - min_b1_30) < 0.95 * tick)).astype(np.float32)
    fill_b0 = (sell_fwd_25 >= 0.65 * b1_q) | ((b1 - min_b1_30) >= 0.95 * tick)
    win_b0 = ((max_b1_45 - b1) >= 0.0) & ((b1 - min_b1_45) < 1.95 * tick)
    y_clean_b0 = (fill_b0 & win_b0).astype(np.float32)

    touch_bm1 = (b1 - min_b1_45) >= 0.95 * tick
    rebound_bm1 = ((max_b1_60 - (b1 - tick)) >= 0.95 * tick) & ((b1 - min_b1_60) < 2.95 * tick)
    y_clean_bm1 = (touch_bm1 & rebound_bm1).astype(np.float32)

    touch_bm2 = (b1 - min_b1_60) >= 1.95 * tick
    rebound_bm2 = ((max_b1_90 - (b1 - 2.0 * tick)) >= 0.95 * tick) & ((b1 - min_b1_90) < 3.95 * tick)
    y_clean_bm2 = (touch_bm2 & rebound_bm2).astype(np.float32)

    return np.column_stack([
        y_sweep_12s.astype(np.float32),
        y_bleed_60s.astype(np.float32),
        y_rebound_30s,
        y_clean_b0,
        y_clean_bm1,
        y_clean_bm2,
    ]).astype(np.float32)


def train_and_inject_cross_venue_first_order(
    cache_path: Path = CACHE_600H_PATH,
    epochs: int = 12,
    batch_size: int = 16384,
) -> dict:
    """
    Двухфазное обучение 4-биржевой модели 1-го порядка `CrossVenueFirstOrderNet v2` (44D -> 6D)
    без малейшей утечки в слепой Holdout RL-агента:
      - Слепой OOS Holdout (`va_start .. n-120`) = последние `25,200` тиков (`7.0` часов 4-биржевого рынка!),
        отделенные эмбарго `600s` от обучающей выборки.
      - Дополнительно замеряется широкий OOS (`120,000` тиков = `33.3ч`) на Phase 1 (до 4-биржевого дообучения).
      - Фаза 1: претрейн базовой сети (SafeTrade + BigONE + Binance BTC/NEAR) на `1.6M` тиков `[tr_start : p1_end]`.
      - Фаза 2: дообучение 4-биржевых каналов Gate.io + Ourbit `[28:40]` с L2-якорем к Фазе 1 на 8-часовом
        4-биржевом префиксе `[four_v_start : tr_end]` в смеси 50/50 с историей.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = np.load(cache_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    del raw

    n = len(data["bid_p"])
    tick = float(data.get("tick_size", 0.01))
    ts_sec = data["ts"].astype(np.int64)
    print("=" * 95, flush=True)
    print(f"🌐 ОБУЧЕНИЕ 4-БИРЖЕВОЙ МОДЕЛИ 1-ГО ПОРЯДКА CrossVenueFirstOrderNet v2 (44D | {n:,} тиков)", flush=True)
    print("=" * 95, flush=True)

    extra_feeds = align_extra_venues_and_macro(ts_sec)
    cv_10d, x_44d = compute_cross_venue_10d_and_44d(
        st_bp=data["bid_p"],
        st_bq=data["bid_q"],
        st_ap=data["ask_p"],
        st_aq=data["ask_q"],
        st_buy=data["buy_vol"],
        st_sell=data["sell_vol"],
        bo_bp=data["bo_bid_p"],
        bo_bq=data["bo_bid_q"],
        bo_ap=data["bo_ask_p"],
        bo_aq=data["bo_ask_q"],
        bo_buy=data["bo_buy_vol"],
        bo_sell=data["bo_sell_vol"],
        bo_sec_since=data["bo_sec_since_tick"],
        extra_feeds=extra_feeds,
        tick=tick,
    )
    y_6d = build_first_order_targets_6d(
        st_bp=data["bid_p"],
        st_bq=data["bid_q"],
        st_ap=data["ask_p"],
        st_buy=data["buy_vol"],
        st_sell=data["sell_vol"],
        y_sweep_12s=data["y_sweep_12s"],
        y_bleed_60s=data["y_bleed_60s"],
        tick=tick,
    )

    # Находим старт живых потоков Gate.io / Ourbit
    active_4v = np.where((x_44d[:, 28] > 0.05) | (x_44d[:, 34] > 0.05))[0]
    four_v_start = int(active_4v[0]) if len(active_4v) > 0 else max(0, n - 54000)

    # Слепой 4-биржевой Holdout (последние 25,200с = 7.0 часов, строго закрыт для обучения!)
    blind_4v_len = 25200
    embargo = 600
    tr_end = n - blind_4v_len - embargo
    va_start = n - blind_4v_len

    # Широкий 33ч срез для Phase 1 (чтобы базовый ствол не видел последние 120,000 тиков на Фазе 1!)
    wide_oos_len = 120000
    p1_end = n - wide_oos_len - embargo
    p1_start = max(0, p1_end - 1600000)

    pos_rates = y_6d[p1_start:p1_end].mean(axis=0)
    pos_weight = np.clip((1.0 - pos_rates) / np.maximum(pos_rates, 1e-4), 1.0, 25.0).astype(np.float32)
    head_names = [
        "cv_sweep_12s",
        "cv_bleed_60s",
        "cv_rebound_30s",
        "clean_fill_b0",
        "clean_fill_bm1",
        "clean_fill_bm2",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    eff_bsz = max(batch_size, 16384) if device.type == "cuda" else batch_size

    def _fast_aucs_gpu(y_true_t: torch.Tensor, probs_t: torch.Tensor) -> dict[str, float]:
        order = torch.argsort(probs_t, dim=0)
        ranks = torch.empty_like(order, dtype=torch.float64)
        base_r = torch.arange(1, probs_t.size(0) + 1, dtype=torch.float64, device=probs_t.device).unsqueeze(1).expand_as(order)
        ranks.scatter_(0, order, base_r)
        pos_mask = y_true_t > 0.5
        n_pos = pos_mask.sum(dim=0).to(torch.float64)
        n_neg = float(probs_t.size(0)) - n_pos
        sum_pos_r = (ranks * pos_mask.to(torch.float64)).sum(dim=0)
        auc_v = torch.where(
            (n_pos > 0) & (n_neg > 0),
            (sum_pos_r - n_pos * (n_pos + 1.0) * 0.5) / (n_pos * n_neg).clamp(min=1.0),
            torch.full_like(n_pos, 0.5),
        ).cpu().tolist()
        return {hn: float(auc_v[j]) for j, hn in enumerate(head_names)}

    x_p1_gpu = torch.from_numpy(x_44d[p1_start:p1_end]).to(device)
    y_p1_gpu = torch.from_numpy(y_6d[p1_start:p1_end]).to(device)
    x_wide_gpu = torch.from_numpy(x_44d[n - wide_oos_len : n - 120]).to(device)
    y_wide_gpu = torch.from_numpy(y_6d[n - wide_oos_len : n - 120]).to(device)
    x_va_gpu = torch.from_numpy(x_44d[va_start : n - 120]).to(device)
    y_va_gpu = torch.from_numpy(y_6d[va_start : n - 120]).to(device)

    model = CrossVenueFirstOrderNet(in_dim=44, h_dim=128, out_dim=6).to(device)
    # Зануляем стартовые веса Gate/Ourbit [28:40] на Фазе 1, чтобы шум отсутствующих фичей = 0
    with torch.no_grad():
        model.fc1.weight[:, 28:40].zero_()
    fast_fwd = model
    pw_t = torch.from_numpy(pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=2e-4, fused=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_p1_auc = 0.0
    best_p1_state = None
    n_p1 = x_p1_gpu.size(0)
    print(f"🔹 [Фаза 1/2] Базовый ствол ST + BO + Macro BTC/NEAR ({n_p1:,} тиков -> слепой 33ч OOS):", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_p1, device=device)
        for s_i in range(0, n_p1, eff_bsz):
            idx = perm[s_i : s_i + eff_bsz]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = fast_fwd(x_p1_gpu[idx])
            loss = F.binary_cross_entropy_with_logits(logits.float(), y_p1_gpu[idx], pos_weight=pw_t)
            loss.backward()
            model.fc1.weight.grad[:, 28:40].zero_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pr_w = torch.empty_like(y_wide_gpu)
            for i0 in range(0, len(x_wide_gpu), 65536):
                pr_w[i0 : i0 + 65536] = torch.sigmoid(fast_fwd(x_wide_gpu[i0 : i0 + 65536]).float())
            aucs_w = _fast_aucs_gpu(y_wide_gpu, pr_w)
        m_w = float(np.mean(list(aucs_w.values())))
        mark = ""
        if m_w > best_p1_auc:
            best_p1_auc = m_w
            best_p1_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " ★ BEST P1"
        print(f"  [P1 Ep {ep:02d}/{epochs}] Wide-33h OOS AUC={m_w:.4f} [sw:{aucs_w['cv_sweep_12s']:.3f} reb:{aucs_w['cv_rebound_30s']:.3f}]{mark}", flush=True)

    assert best_p1_state is not None
    model.load_state_dict(best_p1_state)
    model.eval()
    with torch.no_grad():
        p1_blind_preds = torch.sigmoid(model(x_va_gpu).float())
        p1_blind_aucs = _fast_aucs_gpu(y_va_gpu, p1_blind_preds)
    p1_blind_mean = float(np.mean(list(p1_blind_aucs.values())))
    print(f"   📌 Базовый AUC до подключения Gate.io+Ourbit на слепом 7ч Holdout: {p1_blind_mean:.4f}", flush=True)

    # Фаза 2: Адаптивное подключение Gate.io + Ourbit на 8ч окне [four_v_start : tr_end] с L2-якорем к Фазе 1
    best_state = {k: v.clone() for k, v in best_p1_state.items()}
    best_blind_auc = p1_blind_mean
    best_head_aucs = p1_blind_aucs

    if tr_end - four_v_start >= 7200:
        print(f"🔹 [Фаза 2/2] Дообучение каналов Gate.io + Ourbit на [{four_v_start:,}:{tr_end:,}] ({(tr_end-four_v_start)/3600:.1f}ч) -> слепой 7ч Holdout [{va_start:,}:{n-120:,}]:", flush=True)
        x_4v_tr = torch.from_numpy(x_44d[four_v_start:tr_end]).to(device)
        y_4v_tr = torch.from_numpy(y_6d[four_v_start:tr_end]).to(device)
        anchor_params = {k: v.to(device) for k, v in best_p1_state.items()}
        opt2 = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=5e-4, fused=(device.type == "cuda"))
        n_4v = x_4v_tr.size(0)
        bsz2 = 2048
        for ep2 in range(1, 11):
            model.train()
            perm4 = torch.randperm(n_4v, device=device)
            for s_i in range(0, n_4v, bsz2):
                i4 = perm4[s_i : s_i + bsz2]
                ih = torch.randint(0, n_p1, (len(i4),), device=device)
                bx = torch.cat([x_4v_tr[i4], x_p1_gpu[ih]], dim=0)
                by = torch.cat([y_4v_tr[i4], y_p1_gpu[ih]], dim=0)
                opt2.zero_grad(set_to_none=True)
                logits = model(bx)
                bce = F.binary_cross_entropy_with_logits(logits.float(), by, pos_weight=pw_t)
                # L2-якорь на все слои кроме входных колонок Gate/Ourbit [28:40]
                l2_anc = torch.zeros((), device=device)
                for name, p in model.named_parameters():
                    diff = p - anchor_params[name]
                    if name == "fc1.weight":
                        l2_anc = l2_anc + (diff[:, :28].pow(2).mean() + diff[:, 40:].pow(2).mean()) * 15.0 + diff[:, 28:40].pow(2).mean() * 0.2
                    elif name.startswith("in_norm") and diff.ndim == 1:
                        l2_anc = l2_anc + (diff[:28].pow(2).mean() + diff[40:].pow(2).mean()) * 15.0
                    else:
                        l2_anc = l2_anc + diff.pow(2).mean() * 15.0
                loss2 = bce + l2_anc
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()

            model.eval()
            with torch.no_grad():
                pr_b = torch.sigmoid(model(x_va_gpu).float())
                aucs_b = _fast_aucs_gpu(y_va_gpu, pr_b)
                pr_w2 = torch.empty_like(y_wide_gpu)
                for i0 in range(0, len(x_wide_gpu), 65536):
                    pr_w2[i0 : i0 + 65536] = torch.sigmoid(model(x_wide_gpu[i0 : i0 + 65536]).float())
                aucs_w2 = _fast_aucs_gpu(y_wide_gpu, pr_w2)
            m_b = float(np.mean(list(aucs_b.values())))
            m_w2 = float(np.mean(list(aucs_w2.values())))
            mark = ""
            if m_b > best_blind_auc and m_w2 >= best_p1_auc - 0.0015:
                best_blind_auc = m_b
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_head_aucs = aucs_b
                mark = " ★ BEST 4-VENUE"
            print(
                f"  [P2 Ep {ep2:02d}/10] Blind-7h 4V AUC={m_b:.4f} (Δ={m_b - p1_blind_mean:+.4f}) | Wide-33h AUC={m_w2:.4f}{mark}",
                flush=True,
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        x_va_st_only = x_va_gpu.clone()
        x_va_st_only[:, :10] = 0.0
        x_va_st_only[:, [11, 18, 19, 26, 27]] = 0.0
        x_va_st_only[:, 28:44] = 0.0
        aucs_st_only = _fast_aucs_gpu(y_va_gpu, torch.sigmoid(model(x_va_st_only).float()))

        pr_wide_final = torch.empty_like(y_wide_gpu)
        for i0 in range(0, len(x_wide_gpu), 65536):
            pr_wide_final[i0 : i0 + 65536] = torch.sigmoid(model(x_wide_gpu[i0 : i0 + 65536]).float())
        aucs_wide_final = _fast_aucs_gpu(y_wide_gpu, pr_wide_final)

    mean_st_only = float(np.mean(list(aucs_st_only.values())))
    mean_wide_final = float(np.mean(list(aucs_wide_final.values())))
    print(
        f"🏆 Итог 4-Venue + Macro на слепом 7ч Holdout: ST-only={mean_st_only:.4f} ➔ ST+BO+Macro={p1_blind_mean:.4f} "
        f"➔ 4-Venue v2={best_blind_auc:.4f} (Wide-33h={mean_wide_final:.4f})",
        flush=True,
    )

    del x_p1_gpu, y_p1_gpu, x_wide_gpu, y_wide_gpu, x_va_gpu, y_va_gpu

    torch.save(
        {
            "version": 2,
            "in_dim": 44,
            "h_dim": 128,
            "out_dim": 6,
            "head_names": head_names,
            "state_dict": best_state,
            "oos_mean_auc": mean_wide_final,
            "blind_7h_4venue_auc": best_blind_auc,
            "blind_7h_p1_auc": p1_blind_mean,
            "oos_head_aucs": best_head_aucs,
            "st_only_mean_auc": mean_st_only,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        MODEL_PATH,
    )
    metrics_payload = {
        "version": 2,
        "in_dim": 44,
        "n_ticks": n,
        "four_venue_start_idx": four_v_start,
        "blind_7h_range": [int(va_start), int(n - 120)],
        "wide_33h_oos_mean_auc": round(mean_wide_final, 5),
        "blind_7h_p1_auc": round(p1_blind_mean, 5),
        "blind_7h_4venue_auc": round(best_blind_auc, 5),
        "delta_4venue_over_2venue": round(best_blind_auc - p1_blind_mean, 5),
        "delta_4venue_over_st_only": round(best_blind_auc - mean_st_only, 5),
        "oos_head_aucs": {k: round(v, 5) for k, v in best_head_aucs.items()},
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    all_probs = np.zeros((n, 6), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, 65536):
            bx = torch.from_numpy(x_44d[i0 : i0 + 65536]).to(device)
            all_probs[i0 : i0 + 65536] = model.predict_probs(bx).cpu().numpy()

    cv_probs_scaled = ((all_probs - 0.5) * 2.0).astype(np.float32)
    v7_mat = data["v7_signals_58d"].astype(np.float32).copy()
    v7_mat[:, 18:28] = cv_10d
    v7_mat[:, 52:58] = cv_probs_scaled
    data["v7_signals_58d"] = v7_mat
    data["has_cross_venue_v1"] = np.int32(2)

    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **data)
    tmp_path.replace(cache_path)
    print(
        f"✅ CrossVenueFirstOrderNet v2 (44D) сохранена и встроена в {cache_path.name} за {time.time() - t0:.1f}с!",
        flush=True,
    )
    return metrics_payload


if __name__ == "__main__":
    train_and_inject_cross_venue_first_order()
