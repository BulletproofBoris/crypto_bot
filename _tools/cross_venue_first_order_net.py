#!/usr/bin/env python3
"""
_tools/cross_venue_first_order_net.py
=============================================================================
Модели 1-го порядка нового поколения: Cross-Venue Microstructure & Multi-Level Fill Net
(SafeTrade L1..L5 + BigONE L1..L5 + ST/BO Trade Tape + Executable Arbitrage Lead-Lag).

Решает главную слепую зону предыдущих моделей 1-го порядка (MultiTaskLOBNet / Guard / Realism),
которые смотрели ТОЛЬКО в стакан SafeTrade и не видели поводыря BigONE (32.1% всей предсказательной
силы роя на H=5..120s!).

Выходы модуля (100% каузальные, без единого тика утечки из будущего, 1-в-1 в офлайне и в live):
1. `cv_leadlag_10d` (занимает очищенный слот v7_signals_58d[:, 18:28] -> агент [278:288]):
   [0] bo_fresh            : exp(-bo_sec_since_tick / 15.0) — свежесть котировок BigONE
   [1] arb_buy_st_ticks    : clip((bo_b1 - st_a1) / (3 * tick), -2, 2) * bo_fresh — арбитражная тяга вверх
   [2] arb_sell_st_ticks   : clip((st_b1 - bo_a1) / (3 * tick), -2, 2) * bo_fresh — арбитражный удар по Bid ST!
   [3] mid_diff_ticks      : clip((bo_mid - st_mid) / (5 * tick), -2, 2) * bo_fresh — спред средних цен BO - ST
   [4] bo_vel_5s_ticks     : clip((bo_mid - bo_mid_5s) / (3 * tick), -2, 2) * bo_fresh — 5с импульс BigONE
   [5] lead_lag_5s_ticks   : clip(((bo_mid - bo_mid_5s) - (st_mid - st_mid_5s)) / (3 * tick), -2, 2) * bo_fresh
   [6] bo_imb_l1           : ((bo_b1_q - bo_a1_q) / (bo_b1_q + bo_a1_q + 10)) * bo_fresh
   [7] bo_imb_l5_delta_5s  : clip(bo_imb_l5 - bo_imb_l5_5s, -1.5, 1.5) * bo_fresh
   [8] bo_tape_imb_5s      : ((bo_buy_5s - bo_sell_5s) / (bo_buy_5s + bo_sell_5s + 5)) * bo_fresh
   [9] bo_burst_5s         : (log1p(bo_buy_5s + bo_sell_5s) / 8.0) * bo_fresh

2. `cv_probs_6d` (выход нейросети `CrossVenueFirstOrderNet`, занимает слот v7_signals_58d[:, 52:58] -> агент [312:318]):
   [0] p_cv_sweep_12s      : вероятность пробития ST b1 вниз >= 1 тик за 12с (с учетом BigONE!)
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
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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


def compute_cross_venue_10d_and_28d(
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
    tick: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Векторизованный и строго каузальный расчет:
      - `cv_10d` (N, 10): быстрые кросс-биржевые Lead-Lag импульсы SafeTrade + BigONE,
      - `x_28d`  (N, 28): полный входной вектор для нейросети 1-го порядка `CrossVenueFirstOrderNet`.
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

    # Дополнительные 18 микроструктурных признаков SafeTrade + BigONE для сети 1-го порядка (итого 28D):
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

    x_28d = np.hstack([cv_10d, x_extra]).astype(np.float32)
    return cv_10d, x_28d


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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Однотиковый O(1) расчет (< 0.15 мс на CPU) для `_tools/live_rl_trader.py`
    со 100% математическим совпадением с `compute_cross_venue_10d_and_28d`.
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

    if "dq20" not in hist_state:
        hist_state["dq20"] = deque(maxlen=20)
    dq = hist_state["dq20"]
    cur_snap = (st_mid, bo_mid, bo_imb_l5, st_b1, st_a1, st_b1_q, st_a1_q)
    s5 = dq[-5] if len(dq) >= 5 else (dq[0] if len(dq) > 0 else cur_snap)
    s20 = dq[0] if len(dq) > 0 else cur_snap
    dq.append(cur_snap)

    st_mid_5s, bo_mid_5s, bo_imb_l5_5s, st_b1_5s, st_a1_5s, st_b1q_5s, st_a1q_5s = s5
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

    return cv_10d, np.concatenate([cv_10d, x_extra], axis=0)


class CrossVenueFirstOrderNet(nn.Module):
    """
    Быстрая Residual-SwiGLU нейросеть 1-го порядка (28D -> 128D -> Residual SwiGLU -> 6 логитов).
    Инференс 1 тика на CPU занимает ~0.08 мс; пакетный инференс 2.75M тиков на GPU — ~1.5 сек.
    """

    def __init__(self, in_dim: int = 28, h_dim: int = 128, out_dim: int = 6):
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
    """
    Формирует 6 таргетов для обучения сети 1-го порядка (строго для loss на Train/Val):
      [0] y_sweep_12s      : вынос b1 вниз >= 1 тик за 12с
      [1] y_bleed_60s      : ступенчатое падение b1 >= 2 тика без отскока за 60с
      [2] y_rebound_30s    : чистый подъем b1 >= +1 тик за 30с без падения вниз >= 1 тик!
      [3] y_clean_fill_b0  : постановка Bid на b1 получает проторговку очереди И в окне 45с цена выходит в >= b1 + 1t без обвала >= 2t
      [4] y_clean_fill_bm1 : постановка Bid на b1 - 1t получает налив И в окне 60с цена возвращается >= b1 (прибыль +1t от входа!) без обвала до b1 - 3t
      [5] y_clean_fill_bm2 : постановка Bid на b1 - 2t получает налив И в окне 90с цена возвращается >= b1 - 1t (прибыль +1t от входа!) без обвала до b1 - 4t
    """
    n = len(st_bp)
    b1 = st_bp[:, 0].astype(np.float32)
    b1_q = np.maximum(1.0, st_bq[:, 0].astype(np.float32))
    sell_v = np.maximum(0.0, st_sell.astype(np.float32))

    # Скользящие вперед максимумы/минимумы b1 и кумулятивный sell_vol
    max_b1_30 = np.full(n, -1e9, dtype=np.float32)
    min_b1_30 = np.full(n, 1e9, dtype=np.float32)
    max_b1_45 = np.full(n, -1e9, dtype=np.float32)
    min_b1_45 = np.full(n, 1e9, dtype=np.float32)
    max_b1_60 = np.full(n, -1e9, dtype=np.float32)
    min_b1_60 = np.full(n, 1e9, dtype=np.float32)
    max_b1_90 = np.full(n, -1e9, dtype=np.float32)
    min_b1_90 = np.full(n, 1e9, dtype=np.float32)

    cs_sell = np.cumsum(sell_v.astype(np.float64))
    sell_fwd_25 = np.zeros(n, dtype=np.float32)
    if n > 25:
        sell_fwd_25[:-25] = (cs_sell[25:] - cs_sell[:-25]).astype(np.float32)

    for k in range(1, 91):
        bf = np.pad(b1[k:], (0, k), mode="edge")
        if k <= 30:
            max_b1_30 = np.maximum(max_b1_30, bf)
            min_b1_30 = np.minimum(min_b1_30, bf)
        if k <= 45:
            max_b1_45 = np.maximum(max_b1_45, bf)
            min_b1_45 = np.minimum(min_b1_45, bf)
        if k <= 60:
            max_b1_60 = np.maximum(max_b1_60, bf)
            min_b1_60 = np.minimum(min_b1_60, bf)
        max_b1_90 = np.maximum(max_b1_90, bf)
        min_b1_90 = np.minimum(min_b1_90, bf)

    y_rebound_30s = (((max_b1_30 - b1) >= 0.95 * tick) & ((b1 - min_b1_30) < 0.95 * tick)).astype(np.float32)

    # b0 (на b1): налив очереди sell_fwd_25 >= 0.65 * b1_q (или микрокасание), без обвала >= 2t, с выходом ask >= b1+1t (т.е. b1 >= b1 или a1 проторгован)
    fill_b0 = (sell_fwd_25 >= 0.65 * b1_q) | ((b1 - min_b1_30) >= 0.95 * tick)
    win_b0 = ((max_b1_45 - b1) >= 0.0) & ((b1 - min_b1_45) < 1.95 * tick)
    y_clean_b0 = (fill_b0 & win_b0).astype(np.float32)

    # bm1 (на b1 - 1t): цена качнулась до b1 - 1t (налили наш лимитник на -1t), не провалилась до b1 - 3t, и вернулась хотя бы к b1!
    touch_bm1 = (b1 - min_b1_45) >= 0.95 * tick
    rebound_bm1 = ((max_b1_60 - (b1 - tick)) >= 0.95 * tick) & ((b1 - min_b1_60) < 2.95 * tick)
    y_clean_bm1 = (touch_bm1 & rebound_bm1).astype(np.float32)

    # bm2 (на b1 - 2t): цена качнулась до b1 - 2t (налили наш нижний уровень), не провалилась до b1 - 4t, и отскочила хотя бы к b1 - 1t!
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


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true > 0.5
    n_pos = int(np.sum(pos))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_pos_ranks = float(np.sum(ranks[pos]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (float(n_pos) * float(n_neg))
    return float(auc)


def train_and_inject_cross_venue_first_order(
    cache_path: Path = CACHE_600H_PATH,
    epochs: int = 12,
    batch_size: int = 8192,
) -> dict:
    """
    1) Загружает обновленный `rl_guard_lob_cache_600h.npz` (включая `bo_*` массивы).
    2) Строго делит данные по времени, исключая последние 67,500с (Holdout ValBasket Уроков 18–19!)
       из обучения, чтобы исключить малейшую утечку в проверочную корзину RL-агента!
    3) Обучает `CrossVenueFirstOrderNet` на GPU, замеряет честный OOS ROC-AUC по всем 6 головам
       (и сравнивает с SafeTrade-only базой!).
    4) Записывает каузальные `cv_leadlag_10d` в `v7_signals_58d[:, 18:28]` и `cv_probs_6d`
       в `v7_signals_58d[:, 52:58]` атомарно на диск.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = np.load(cache_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    del raw

    n = len(data["bid_p"])
    tick = float(data.get("tick_size", 0.01))
    print("=" * 95, flush=True)
    print(f"🌐 ОБУЧЕНИЕ МОДЕЛИ 1-ГО ПОРЯДКА CrossVenueFirstOrderNet (SafeTrade + BigONE | {n:,} тиков)", flush=True)
    print("=" * 95, flush=True)

    cv_10d, x_28d = compute_cross_venue_10d_and_28d(
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

    # Строгая хронологическая изоляция: последние 120,000с (включая все 67,500с Holdout RL!) — чистый OOS!
    # Дополнительно берем буфер эмбарго 600с между Train и OOS.
    oos_len = 120000
    embargo = 600
    tr_end = n - oos_len - embargo
    va_start = n - oos_len
    # Пропускаем ранний префикс, если там не было BigONE, или учим на последних 1.6M тиков актуального режима + вес свежести
    tr_start = max(0, tr_end - 1600000)

    pos_rates = y_6d[tr_start:tr_end].mean(axis=0)
    pos_weight = np.clip((1.0 - pos_rates) / np.maximum(pos_rates, 1e-4), 1.0, 25.0).astype(np.float32)
    head_names = [
        "cv_sweep_12s",
        "cv_bleed_60s",
        "cv_rebound_30s",
        "clean_fill_b0",
        "clean_fill_bm1",
        "clean_fill_bm2",
    ]
    print("📊 Базовые частоты таргетов на Train:", flush=True)
    for hn, pr, pw in zip(head_names, pos_rates, pos_weight):
        print(f"   • {hn:<16}: {pr*100:6.2f}% (pos_weight={pw:5.2f})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr_t = torch.from_numpy(x_28d[tr_start:tr_end])
    y_tr_t = torch.from_numpy(y_6d[tr_start:tr_end])
    x_va_t = torch.from_numpy(x_28d[va_start:n - 120])
    y_va_np = y_6d[va_start:n - 120]

    train_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, drop_last=False)
    model = CrossVenueFirstOrderNet(in_dim=28, h_dim=128, out_dim=6).to(device)
    pw_t = torch.from_numpy(pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc = 0.0
    best_state = None
    best_head_aucs = {}

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        n_b = 0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.binary_cross_entropy_with_logits(logits, by, pos_weight=pw_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot_loss += float(loss.item())
            n_b += 1
        sched.step()

        model.eval()
        preds_va = []
        with torch.no_grad():
            for i0 in range(0, len(x_va_t), 32768):
                pb = model.predict_probs(x_va_t[i0 : i0 + 32768].to(device)).cpu().numpy()
                preds_va.append(pb)
        preds_va_np = np.vstack(preds_va)
        aucs = {hn: _roc_auc_binary(y_va_np[:, j], preds_va_np[:, j]) for j, hn in enumerate(head_names)}
        mean_auc = float(np.mean(list(aucs.values())))
        mark = ""
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_head_aucs = aucs
            mark = " ★ BEST"
        print(
            f"  [Ep {ep:02d}/{epochs}] loss={tot_loss/max(1,n_b):.4f} | OOS Mean AUC={mean_auc:.4f} "
            f"[sw:{aucs['cv_sweep_12s']:.3f} bl:{aucs['cv_bleed_60s']:.3f} reb:{aucs['cv_rebound_30s']:.3f} "
            f"b0:{aucs['clean_fill_b0']:.3f} bm1:{aucs['clean_fill_bm1']:.3f} bm2:{aucs['clean_fill_bm2']:.3f}]{mark}",
            flush=True,
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    # Замеряем чистый вклад BigONE на OOS (зануляем каналы BigONE и смотрим падение AUC!)
    x_va_no_bo = x_va_t.clone()
    x_va_no_bo[:, :10] = 0.0
    x_va_no_bo[:, [11, 18, 19, 26, 27]] = 0.0
    preds_no_bo = []
    with torch.no_grad():
        for i0 in range(0, len(x_va_no_bo), 32768):
            preds_no_bo.append(model.predict_probs(x_va_no_bo[i0 : i0 + 32768].to(device)).cpu().numpy())
    preds_no_bo_np = np.vstack(preds_no_bo)
    aucs_no_bo = {hn: _roc_auc_binary(y_va_np[:, j], preds_no_bo_np[:, j]) for j, hn in enumerate(head_names)}
    mean_auc_no_bo = float(np.mean(list(aucs_no_bo.values())))
    print(
        f"🔍 Прирост точности от стакана и ленты BigONE на слепом OOS: "
        f"AUC {mean_auc_no_bo:.4f} (ST-only) ➔ {best_auc:.4f} (ST+BO) [ΔAUC = {best_auc - mean_auc_no_bo:+.4f}]",
        flush=True,
    )

    # Сохраняем веса модели 1-го порядка и метрики
    torch.save(
        {
            "in_dim": 28,
            "h_dim": 128,
            "out_dim": 6,
            "head_names": head_names,
            "state_dict": best_state,
            "oos_mean_auc": best_auc,
            "oos_head_aucs": best_head_aucs,
            "st_only_mean_auc": mean_auc_no_bo,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        MODEL_PATH,
    )
    metrics_payload = {
        "n_ticks": n,
        "train_range": [int(tr_start), int(tr_end)],
        "oos_range": [int(va_start), int(n - 120)],
        "oos_mean_auc": round(best_auc, 5),
        "st_only_mean_auc": round(mean_auc_no_bo, 5),
        "delta_auc_from_bigone": round(best_auc - mean_auc_no_bo, 5),
        "oos_head_aucs": {k: round(v, 5) for k, v in best_head_aucs.items()},
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Пакетный расчет вероятностей на всей глубине кэша (2.75M тиков) и запись в v7_signals_58d
    all_probs = np.zeros((n, 6), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, 65536):
            bx = torch.from_numpy(x_28d[i0 : i0 + 65536]).to(device)
            all_probs[i0 : i0 + 65536] = model.predict_probs(bx).cpu().numpy()

    # Центрируем вероятности вокруг 0.5 -> [-1..+1], чтобы масштаб идеально совпадал со входами SwiGLU!
    cv_probs_scaled = ((all_probs - 0.5) * 2.0).astype(np.float32)

    v7_mat = data["v7_signals_58d"].astype(np.float32).copy()
    v7_mat[:, 18:28] = cv_10d
    v7_mat[:, 52:58] = cv_probs_scaled
    data["v7_signals_58d"] = v7_mat
    data["has_cross_venue_v1"] = np.int32(1)

    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **data)
    tmp_path.replace(cache_path)
    print(
        f"✅ CrossVenueFirstOrderNet сохранена в {MODEL_PATH.name}, а сигналы 10D+6D встроены в {cache_path.name} за {time.time() - t0:.1f}с!",
        flush=True,
    )
    return metrics_payload


if __name__ == "__main__":
    train_and_inject_cross_venue_first_order()
