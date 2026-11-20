#!/usr/bin/env python3
"""
_tools/hierarchical_regime_tree.py
=============================================================================
Урок 28 (Компонент 1): Масштабно-инвариантный Иерархический Древовидный
Классификатор-Роутер 7 горизонтов `H = [120s -> 60s -> 25s -> 12s -> 5s -> 2s -> 1s]`
(без коллинеарного `90s`) для `SafeTrade (prl)` и `BigONE (prl_bo)`.

Закрывает все 5 архитектурных уязвимостей маршрутизации 1-го порядка:
  1. Scale-Matched Signatures + Soft Hierarchical Prior + Global Crisis Bypass:
     каждый этаж дерева сопоставляется со своим временным масштабом (Macro 60..300s,
     Meso 20..60s, Micro 1..20s), а кризисные якоря (`is_crisis_anchor=True`) имеют
     прямой байпас мимо размытых 21-дневных корней `120s`.
  2. Venue-Native & Tick-Invariant `bps` State:
     никаких чужих `ST_*` признаков в паспортах BigONE и никаких захардкоженных `tick=0.01`;
     все спреды и волатильности измеряются строго в базисных пунктах (`bps = 1e4 * dP / P`).
  3. Verified Causal Skill Weighting:
     вес окна пропорционален подтверждённому OOS-качеству (`IC * (AUC - 0.5)`), которое
     вступает в силу строго каузально при `ts > test_end_sec` (до этого для живого листа
     действует нейтральный приор без заглядывания в будущее).
  4. Physical `bps / sqrt(H)` Coherence + Exact Hybrid Leaf Parity (`LGB + Top FastKerasModel`):
     приводит `expected_z` к единой физической шкале (учитывая `eff_tick = $0.0005` на `1s/2s`
     и `$0.0050` на `5s..120s`) и исполняет ровно `Top-2 Tree + 1 Live T_now` гибридных
     узлов одинаково в офлайн-кэше и в живом `1 Гц` роботе (`~6–8 мс` на CPU).
  5. Multi-Timescale Causal Logit EMA:
     сглаживает логиты переключения ветвей с постоянной времени масштаба уровня
     (`30s -> 15s -> 8s -> 4s -> 2s -> 1s -> 0s`), исключая посекундный дребезг.
=============================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.fast_neural_engine import FastKerasModel
from _tools.system_resources import SAFE_WORKERS

# Сверху вниз для дерева (от корней к листьям) и снизу вверх для выходного тензора 42D:
TREE_LEVELS_DESC = [(120, 3.0), (60, 3.0), (25, 3.0), (12, 3.5), (5, 3.0), (2, 3.0), (1, 3.0)]
L28_HORIZONS_ASC = [(1, 3.0), (2, 3.0), (5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (120, 3.0)]

EFF_TICK_BY_H = {1: 0.0005, 2: 0.0005, 5: 0.0050, 12: 0.0050, 25: 0.0050, 60: 0.0050, 120: 0.0050}
EMA_TAU_BY_H = {120: 30.0, 60: 15.0, 25: 8.0, 12: 4.0, 5: 2.0, 2: 1.0, 1: 0.0}
ACT_SCALE_L28 = {1: 90.0, 2: 65.0, 5: 180.0, 12: 38.0, 25: 15.0, 60: 5.0, 120: 3.5}
CLASS_VALS_7 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
NEURAL_ARCHS = ["mlp_bottleneck", "wavenet_tcn", "cross_mixer", "causal_cnn", "attention", "conv1d+gru"]


def _win_num(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def _causal_ema_2d(mat: np.ndarray, tau_sec: float) -> np.ndarray:
    """Быстрый каузальный экспоненциальный фильтр вдоль оси времени (axis=0)."""
    if tau_sec <= 0.25 or len(mat) <= 1:
        return mat
    alpha = float(1.0 - math.exp(-1.0 / max(0.25, tau_sec)))
    one_m = 1.0 - alpha
    out = np.empty_like(mat, dtype=np.float32)
    cur = mat[0].astype(np.float32).copy()
    out[0] = cur
    # Векторный проход по строкам (при M <= 80 на старших уровнях занимает миллисекунды; для 391 окна BO H=1s tau=0!)
    for i in range(1, len(mat)):
        cur = one_m * cur + alpha * mat[i]
        out[i] = cur
    return out


def required_router_state_columns() -> list[str]:
    cols = ["timestamp", "BO_sec_since_tick", "arb_eff_100_pct", "mid_price_diff"]
    for vp in ("ST", "BO"):
        cols.extend([
            f"{vp}_mid_price", f"{vp}_spread_pct",
            f"{vp}_bid_1_qty", f"{vp}_ask_1_qty",
            f"{vp}_bids_vol_top5", f"{vp}_asks_vol_top5",
            f"{vp}_wall_vol_imbalance", f"{vp}_microprice_offset_ticks",
            f"{vp}_vol_5s_ticks", f"{vp}_vol_20s_ticks", f"{vp}_vol_60s_ticks", f"{vp}_vol_300s_ticks",
            f"{vp}_ofi_norm_5s", f"{vp}_ofi_top5_norm_5s", f"{vp}_ofi_norm_20s", f"{vp}_ofi_norm_60s", f"{vp}_ofi_norm_300s",
            f"{vp}_cvd_norm_5s", f"{vp}_cvd_norm_20s", f"{vp}_cvd_norm_60s", f"{vp}_cvd_norm_300s",
            f"{vp}_vol_intensity_5s", f"{vp}_vol_intensity_60s",
        ])
    return cols


def extract_venue_invariant_states(df: pd.DataFrame, coin: str) -> dict[str, np.ndarray]:
    """
    Извлекает раздельные для площадки (`ST` или `BO`) масштабно-инвариантные векторы состояния
    в базисных пунктах (`bps`) и безразмерных дисбалансах на 3 этажах времени:
      - `macro` [N, 8]: для корней H = 120s, 60s
      - `meso`  [N, 8]: для средних уровней H = 25s, 12s
      - `micro` [N, 8]: для быстрых листьев H = 5s, 2s, 1s
      - `crisis_intensity` [N]: мгновенный детектор экстремального режима для Global Crisis Bypass
    """
    n = len(df)
    vp = "BO" if coin.lower().endswith("_bo") else "ST"

    def _c(name: str, dval: float = 0.0) -> np.ndarray:
        if name in df.columns:
            return np.nan_to_num(df[name].to_numpy(dtype=np.float32, copy=False), nan=dval, posinf=dval, neginf=dval)
        return np.full(n, dval, dtype=np.float32)

    mid = np.maximum(_c(f"{vp}_mid_price", 1.5), 0.05)
    # В prl_1s_features.parquet `*_vol_*_ticks` рассчитывались через фикс-делитель 0.01 -> переводим строго в bps!
    px_to_bps = 100.0 / mid  # т.к. (ticks * 0.01 / mid) * 1e4 = ticks * (100.0 / mid)

    sp_bps = np.clip(_c(f"{vp}_spread_pct", 0.004) * 1e4, 0.5, 400.0)
    log_sp = np.log1p(sp_bps)

    vol5_bps = np.clip(_c(f"{vp}_vol_5s_ticks", 0.1) * px_to_bps, 0.0, 300.0)
    vol20_bps = np.clip(_c(f"{vp}_vol_20s_ticks", 0.2) * px_to_bps, 0.0, 400.0)
    vol60_bps = np.clip(_c(f"{vp}_vol_60s_ticks", 0.4) * px_to_bps, 0.0, 500.0)
    vol300_bps = np.clip(_c(f"{vp}_vol_300s_ticks", 0.8) * px_to_bps, 0.0, 600.0)

    b5 = np.maximum(0.0, _c(f"{vp}_bids_vol_top5", 50.0))
    a5 = np.maximum(0.0, _c(f"{vp}_asks_vol_top5", 50.0))
    imb_top5 = (b5 - a5) / (b5 + a5 + 10.0)
    wall_imb = np.clip(_c(f"{vp}_wall_vol_imbalance", 0.0), -1.0, 1.0)

    b1 = np.maximum(0.0, _c(f"{vp}_bid_1_qty", 10.0))
    a1 = np.maximum(0.0, _c(f"{vp}_ask_1_qty", 10.0))
    imb_bbo = 0.5 * ((b1 - a1) / (b1 + a1 + 5.0)) + 0.5 * imb_top5

    micro_bps = np.clip(_c(f"{vp}_microprice_offset_ticks", 0.0) * px_to_bps, -100.0, 100.0) / 25.0
    arb_bps = np.clip(_c("arb_eff_100_pct", 0.0) * 1e4, -300.0, 300.0) / 50.0
    if vp == "BO":
        arb_bps = -arb_bps

    ofi5 = np.clip(_c(f"{vp}_ofi_norm_5s", 0.0), -4.0, 4.0)
    ofi5_t5 = np.clip(_c(f"{vp}_ofi_top5_norm_5s", 0.0), -4.0, 4.0)
    ofi20 = np.clip(_c(f"{vp}_ofi_norm_20s", 0.0), -4.0, 4.0)
    ofi60 = np.clip(_c(f"{vp}_ofi_norm_60s", 0.0), -4.0, 4.0)
    ofi300 = np.clip(_c(f"{vp}_ofi_norm_300s", 0.0), -4.0, 4.0)

    cvd5 = np.clip(_c(f"{vp}_cvd_norm_5s", 0.0), -4.0, 4.0)
    cvd20 = np.clip(_c(f"{vp}_cvd_norm_20s", 0.0), -4.0, 4.0)
    cvd60 = np.clip(_c(f"{vp}_cvd_norm_60s", 0.0), -4.0, 4.0)
    cvd300 = np.clip(_c(f"{vp}_cvd_norm_300s", 0.0), -4.0, 4.0)

    vint5 = np.clip(np.log1p(np.maximum(0.0, _c(f"{vp}_vol_intensity_5s", 0.0))), 0.0, 4.0)
    vint60 = np.clip(np.log1p(np.maximum(0.0, _c(f"{vp}_vol_intensity_60s", 0.0))), 0.0, 4.0)

    macro = np.column_stack([
        log_sp, np.log1p(vol300_bps), np.log1p(vol60_bps),
        ofi300, cvd300, imb_top5, wall_imb, arb_bps,
    ]).astype(np.float32)

    meso = np.column_stack([
        log_sp, np.log1p(vol60_bps), np.log1p(vol20_bps),
        ofi60, ofi20, cvd60, imb_top5, vint60,
    ]).astype(np.float32)

    micro = np.column_stack([
        log_sp, np.log1p(vol5_bps), ofi5, ofi5_t5,
        cvd5, imb_bbo, micro_bps, vint5,
    ]).astype(np.float32)

    # Индекс экстремального кризиса/сквиза [0..1] для активации Global Crisis Bypass
    crisis_raw = 0.40 * (vol20_bps / 35.0) + 0.35 * (np.abs(ofi20) / 1.8) + 0.25 * (vint5 / 1.8)
    crisis_intensity = np.clip(crisis_raw - 0.55, 0.0, 1.5).astype(np.float32)

    return {
        "macro": np.nan_to_num(macro, nan=0.0, posinf=4.0, neginf=-4.0),
        "meso": np.nan_to_num(meso, nan=0.0, posinf=4.0, neginf=-4.0),
        "micro": np.nan_to_num(micro, nan=0.0, posinf=4.0, neginf=-4.0),
        "mid_price": mid.astype(np.float32),
        "crisis_intensity": crisis_intensity,
    }


def extract_venue_invariant_state_1row(row_map, coin: str) -> dict[str, np.ndarray | float]:
    """Сверхбыстрый (<0.08 мс) 1-строчный расчёт bps-состояния без накладных расходов pandas DataFrame."""
    vp = "BO" if coin.lower().endswith("_bo") else "ST"

    def _f(name: str, dval: float = 0.0) -> float:
        raw = row_map.get(name, dval) if hasattr(row_map, "get") else dval
        if raw is None:
            return dval
        val = float(raw)
        if not math.isfinite(val):
            return dval
        return val

    mid = max(_f(f"{vp}_mid_price", 1.5), 0.05)
    px_to_bps = 100.0 / mid
    sp_bps = min(max(_f(f"{vp}_spread_pct", 0.004) * 1e4, 0.5), 400.0)
    log_sp = math.log1p(sp_bps)

    vol5_bps = min(max(_f(f"{vp}_vol_5s_ticks", 0.1) * px_to_bps, 0.0), 300.0)
    vol20_bps = min(max(_f(f"{vp}_vol_20s_ticks", 0.2) * px_to_bps, 0.0), 400.0)
    vol60_bps = min(max(_f(f"{vp}_vol_60s_ticks", 0.4) * px_to_bps, 0.0), 500.0)
    vol300_bps = min(max(_f(f"{vp}_vol_300s_ticks", 0.8) * px_to_bps, 0.0), 600.0)

    b5 = max(0.0, _f(f"{vp}_bids_vol_top5", 50.0))
    a5 = max(0.0, _f(f"{vp}_asks_vol_top5", 50.0))
    imb_top5 = (b5 - a5) / (b5 + a5 + 10.0)
    wall_imb = min(max(_f(f"{vp}_wall_vol_imbalance", 0.0), -1.0), 1.0)

    b1 = max(0.0, _f(f"{vp}_bid_1_qty", 10.0))
    a1 = max(0.0, _f(f"{vp}_ask_1_qty", 10.0))
    imb_bbo = 0.5 * ((b1 - a1) / (b1 + a1 + 5.0)) + 0.5 * imb_top5

    micro_bps = min(max(_f(f"{vp}_microprice_offset_ticks", 0.0) * px_to_bps, -100.0), 100.0) / 25.0
    arb_bps = min(max(_f("arb_eff_100_pct", 0.0) * 1e4, -300.0), 300.0) / 50.0
    if vp == "BO":
        arb_bps = -arb_bps

    ofi5 = min(max(_f(f"{vp}_ofi_norm_5s", 0.0), -4.0), 4.0)
    ofi5_t5 = min(max(_f(f"{vp}_ofi_top5_norm_5s", 0.0), -4.0), 4.0)
    ofi20 = min(max(_f(f"{vp}_ofi_norm_20s", 0.0), -4.0), 4.0)
    ofi60 = min(max(_f(f"{vp}_ofi_norm_60s", 0.0), -4.0), 4.0)
    ofi300 = min(max(_f(f"{vp}_ofi_norm_300s", 0.0), -4.0), 4.0)

    cvd5 = min(max(_f(f"{vp}_cvd_norm_5s", 0.0), -4.0), 4.0)
    cvd60 = min(max(_f(f"{vp}_cvd_norm_60s", 0.0), -4.0), 4.0)
    cvd300 = min(max(_f(f"{vp}_cvd_norm_300s", 0.0), -4.0), 4.0)

    vint5 = min(max(math.log1p(max(0.0, _f(f"{vp}_vol_intensity_5s", 0.0))), 0.0), 4.0)
    vint60 = min(max(math.log1p(max(0.0, _f(f"{vp}_vol_intensity_60s", 0.0))), 0.0), 4.0)

    macro = np.array([log_sp, math.log1p(vol300_bps), math.log1p(vol60_bps), ofi300, cvd300, imb_top5, wall_imb, arb_bps], dtype=np.float32)
    meso = np.array([log_sp, math.log1p(vol60_bps), math.log1p(vol20_bps), ofi60, ofi20, cvd60, imb_top5, vint60], dtype=np.float32)
    micro = np.array([log_sp, math.log1p(vol5_bps), ofi5, ofi5_t5, cvd5, imb_bbo, micro_bps, vint5], dtype=np.float32)

    crisis_raw = 0.40 * (vol20_bps / 35.0) + 0.35 * (abs(ofi20) / 1.8) + 0.25 * (vint5 / 1.8)
    crisis_intensity = min(max(crisis_raw - 0.55, 0.0), 1.5)
    return {
        "macro": macro,
        "meso": meso,
        "micro": micro,
        "mid_price": float(mid),
        "crisis_intensity": float(crisis_intensity),
    }


def _scale_key_for_h(H: int) -> str:
    if H >= 60:
        return "macro"
    if H >= 12:
        return "meso"
    return "micro"


def extract_6d_from_probs_l28(probs: np.ndarray, H: int, mid_price: np.ndarray | float = 1.50) -> np.ndarray:
    """
    Преобразует 7-классовые вероятности [N, 7] в унифицированный 6D вектор:
      [0] dir_log_odds  — направленное лог-отношение шансов [-1..+1]
      [1] tail_log_odds — хвостовое лог-отношение шансов (±2, ±3) [-1..+1]
      [2] vel_bps_sqrtH — физически соизмеримая ожидаемая скорость в bps / sqrt(H), масштабированная в [-3..+3]
      [3] act_up        — лог-активность импульса вверх [0..2]
      [4] act_dn        — лог-активность импульса вниз [0..2]
      [5] norm_entropy  — нормированная энтропия распределения [0..1]
    """
    p = np.clip(np.asarray(probs, dtype=np.float32), 1e-7, 1.0)
    if p.ndim == 1:
        p = p[np.newaxis, :]
    p = p / np.maximum(np.sum(p, axis=1, keepdims=True), 1e-7)

    p_dn = p[:, 0] + p[:, 1] + p[:, 2]
    p_up = p[:, 4] + p[:, 5] + p[:, 6]
    p_tn = p[:, 0] + p[:, 1]
    p_tp = p[:, 5] + p[:, 6]
    ez_cls = p @ CLASS_VALS_7

    eff_tick = EFF_TICK_BY_H.get(int(H), 0.0050)
    mid_arr = np.maximum(np.asarray(mid_price, dtype=np.float32), 0.05)
    # Ожидаемый сдвиг в bps, нормированный на диффузионный корень sqrt(H):
    ez_bps = ez_cls * (eff_tick / mid_arr) * 1e4
    vel_phys = np.clip((ez_bps / math.sqrt(float(H))) * 0.45, -3.0, 3.0)

    act_sc = ACT_SCALE_L28.get(int(H), 10.0)
    dir_lo = np.tanh(0.45 * np.log((p_up + 1e-5) / (p_dn + 1e-5)))
    tail_lo = np.tanh(0.45 * np.log((p_tp + 1e-5) / (p_tn + 1e-5)))
    act_u = np.clip(np.log1p(p_up * act_sc) / 2.5, 0.0, 2.0)
    act_d = np.clip(np.log1p(p_dn * act_sc) / 2.5, 0.0, 2.0)
    ent = -np.sum(p * np.log(p), axis=1) / math.log(7.0)
    return np.column_stack([dir_lo, tail_lo, vel_phys, act_u, act_d, ent]).astype(np.float32)


def compute_tree_coherence_4d(tree_42d: np.ndarray) -> np.ndarray:
    """
    Вычисляет 4D вектор межуровневой согласованности дерева СТРОГО по открытым/активным горизонтам:
      [0] global_dir_consensus  — среднее согласие направления активных горизонтов [-1..+1]
      [1] micro_macro_div       — расхождение активных быстрых листьев (из 1s, 2s, 5s) и макро-корней (60s, 120s)
      [2] cascade_acceleration  — градиент физической скорости от макро (25s..120s) к быстрым листьям (1s/2s, fallback 5s)
      [3] mean_tree_confidence  — средняя уверенность (1 - entropy) строго по активным уровням [0..1] (0.0 при отсутствии активных)
    """
    t = np.asarray(tree_42d, dtype=np.float32)
    if t.ndim == 1:
        t = t[np.newaxis, :]
    # Индексы горизонтов в L28_HORIZONS_ASC: 0=1s, 1=2s, 2=5s, 3=12s, 4=25s, 5=60s, 6=120s
    dirs = t[:, 0::6]   # [N, 7]
    vels = t[:, 2::6]   # [N, 7]
    ents = t[:, 5::6]   # [N, 7]

    active_mask = (np.abs(dirs) + np.abs(vels) + np.abs(ents)) > 1e-5
    any_act = np.any(active_mask, axis=1)
    n_act = np.maximum(np.sum(active_mask, axis=1), 1.0)
    consensus = np.where(any_act, np.sum(dirs * active_mask, axis=1) / n_act, 0.0)

    # Микро-направление по реально активным из {1s, 2s, 5s} (в эпоху $0.01 без выключенных 1s/2s не делится на 3!)
    mic_m = active_mask[:, :3]
    n_mic = np.sum(mic_m, axis=1)
    micro_dir = np.where(n_mic > 0, np.sum(dirs[:, :3] * mic_m, axis=1) / np.maximum(n_mic, 1.0), 0.0)

    mac_m = active_mask[:, 5:7]
    n_mac = np.sum(mac_m, axis=1)
    macro_dir = np.where(n_mac > 0, np.sum(dirs[:, 5:7] * mac_m, axis=1) / np.maximum(n_mac, 1.0), 0.0)
    div = np.where((n_mic > 0) & (n_mac > 0), np.clip(micro_dir - macro_dir, -2.0, 2.0), 0.0)

    # Быстрая скорость: {1s, 2s}, а если они ещё не открыты (историческая эпоха $0.01) -> чистый fallback на {5s, 12s}
    fast12_m = active_mask[:, :2]
    n_f12 = np.sum(fast12_m, axis=1)
    v_fast12 = np.sum(vels[:, :2] * fast12_m, axis=1) / np.maximum(n_f12, 1.0)
    fast512_m = active_mask[:, 2:4]
    n_f512 = np.sum(fast512_m, axis=1)
    v_fast512 = np.sum(vels[:, 2:4] * fast512_m, axis=1) / np.maximum(n_f512, 1.0)
    has_fast = (n_f12 > 0) | (n_f512 > 0)
    v_fast = np.where(n_f12 > 0, v_fast12, np.where(n_f512 > 0, v_fast512, 0.0))

    slow_m = active_mask[:, 4:7]
    n_slow = np.sum(slow_m, axis=1)
    v_slow = np.where(n_slow > 0, np.sum(vels[:, 4:7] * slow_m, axis=1) / np.maximum(n_slow, 1.0), 0.0)
    accel = np.where(has_fast & (n_slow > 0), np.clip(v_fast - v_slow, -3.0, 3.0) * 0.5, 0.0)

    conf = np.where(any_act, np.clip(np.sum((1.0 - ents) * active_mask, axis=1) / n_act, 0.0, 1.0), 0.0)
    return np.column_stack([consensus, div, accel, conf]).astype(np.float32)


def build_or_load_tree_index(coin: str, force_rebuild: bool = False) -> dict:
    """
    Строит и сохраняет масштабно-инвариантный иерархический индекс дерева для `prl` или `prl_bo`:
      - пересчитывает `mu_8d`, `std_8d` каждого окна строго по ЕГО площадке и ЕГО масштабу времени;
      - связывает родительские и дочерние окна от `120s` к `1s`;
      - консолидирует избыточные спокойные старые листья (> 8 на ветвь) в кластерные медоиды,
        оставляя неприкосновенными свежие окна и кризисные якоря (`is_crisis_anchor=True`);
      - находит лучший быстрый нейроэксперт (`best_nn`) для каждого окна.
    """
    coin_l = coin.lower()
    idx_path = BASE_DIR / "models" / coin_l / "hierarchical_tree_index_l28.json"
    if idx_path.exists() and not force_rebuild:
        return json.loads(idx_path.read_text(encoding="utf-8"))

    t0 = time.time()
    cache_sub = "_mmap_cache_bo" if coin_l.endswith("_bo") else "_mmap_cache"
    cache_dir = BASE_DIR / "data" / cache_sub
    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"

    req_cols = [c for c in required_router_state_columns() if c in set(pq.read_schema(labeled_path).names)]
    df_state = pq.read_table(labeled_path, columns=req_cols).to_pandas()
    states = extract_venue_invariant_states(df_state, coin_l)
    raw_ts = df_state["timestamp"] if "timestamp" in df_state.columns else df_state.index
    ts_sec_all = pd.to_datetime(np.asarray(raw_ts), utc=True).as_unit("s").astype("int64").to_numpy()
    del df_state
    gc.collect()

    levels_data: dict[str, list[dict]] = {}
    for H, K in TREE_LEVELS_DESC:
        h_key = str(H)
        scale_key = _scale_key_for_h(H)
        s_mat = states[scale_key]
        cr_vec = states["crisis_intensity"]

        lgb_dir = BASE_DIR / "models" / coin_l / f"h{H}s_k{K:g}"
        metas = sorted(lgb_dir.glob("meta_win_*.json"), key=_win_num)
        n_w_cache = int(np.load(cache_dir / f"h{H}_n_windows.npy"))
        metas = [m for m in metas if _win_num(m) < n_w_cache]

        # Карта ВСЕХ нейроэкспертов по номеру окна (полный ансамбль, Variant B)
        all_nn_by_win: dict[int, list[dict]] = {}
        best_nn_by_win: dict[int, dict] = {}
        for arch in NEURAL_ARCHS:
            a_dir = BASE_DIR / "models" / coin_l / f"{arch.replace('+', '_')}_h{H}s"
            if not a_dir.exists():
                continue
            for w_i in range(n_w_cache):
                mp_nn = a_dir / f"meta_win_{w_i:02d}.json"
                kp_nn = a_dir / f"model_win_{w_i:02d}.keras"
                if mp_nn.exists() and kp_nn.exists():
                    m_nn = json.loads(mp_nn.read_text(encoding="utf-8"))
                    mo = m_nn.get("metrics_oos") or {}
                    ic_nn = float(mo.get("ic_dt_ticks", 0.0))
                    nn_entry = {
                        "arch": arch,
                        "keras_rel": f"{arch.replace('+', '_')}_h{H}s/model_win_{w_i:02d}.keras",
                        "meta_rel": f"{arch.replace('+', '_')}_h{H}s/meta_win_{w_i:02d}.json",
                        "seq_len": int(m_nn.get("seq_len", 15)),
                        "ic": round(ic_nn, 4),
                    }
                    if ic_nn > 0.02:
                        all_nn_by_win.setdefault(w_i, []).append(nn_entry)
                        if w_i not in best_nn_by_win or ic_nn > best_nn_by_win[w_i]["ic"]:
                            best_nn_by_win[w_i] = nn_entry

        nodes = []
        vol_scores = []
        for mp_lgb in metas:
            w_i = _win_num(mp_lgb)
            m_obj = json.loads(mp_lgb.read_text(encoding="utf-8"))
            tr_idx_p = cache_dir / f"h{H}_tr_idx_{w_i:02d}.npy"
            te_idx_p = cache_dir / f"h{H}_te_idx_{w_i:02d}.npy"
            if not tr_idx_p.exists():
                continue
            tr_idx = np.load(tr_idx_p)
            te_idx = np.load(te_idx_p) if te_idx_p.exists() else tr_idx[-1000:]

            sub_s = s_mat[tr_idx]
            mu = np.mean(sub_s, axis=0).astype(np.float32)
            sd = np.maximum(np.std(sub_s, axis=0), 0.08).astype(np.float32)
            cr_q90 = float(np.quantile(cr_vec[tr_idx], 0.90))

            mo_lgb = m_obj.get("metrics_oos") or {}
            ic_lgb = float(mo_lgb.get("ic_dt_ticks", 0.0))
            auc_lgb = float(mo_lgb.get("roc_auc_ovr", 0.5))
            nn_info = best_nn_by_win.get(w_i)
            all_nn_list = sorted(all_nn_by_win.get(w_i, []), key=lambda x: -x["ic"])
            ic_best = max(ic_lgb, float(nn_info["ic"]) if nn_info else 0.0)
            # Каузальный подтвержденный skill (0 для шумовых окон с IC <= 0.005!)
            verified_skill = float(max(0.0, ic_best - 0.005) * (0.5 + max(0.0, auc_lgb - 0.5)))

            vol_scores.append(cr_q90)
            nodes.append({
                "win_idx": int(w_i),
                "window_id": f"win_{w_i:02d}",
                "H": int(H),
                "K": float(K),
                "tr_start_sec": int(ts_sec_all[tr_idx[0]]),
                "tr_end_sec": int(ts_sec_all[tr_idx[-1]]),
                "te_end_sec": int(ts_sec_all[te_idx[-1]]),
                "mu": [round(float(x), 5) for x in mu],
                "sd": [round(float(x), 5) for x in sd],
                "crisis_q90": round(cr_q90, 4),
                "ic_lgb": round(ic_lgb, 4),
                "auc_lgb": round(auc_lgb, 4),
                "verified_skill": round(verified_skill, 5),
                "prior_skill": round(max(0.02, float(np.median([0.06, max(0.02, ic_best * 0.5)]))), 5),
                "lgb_rel": f"h{H}s_k{K:g}/model_win_{w_i:02d}.txt",
                "best_nn": nn_info,
                "all_nn": all_nn_list,  # полный ансамбль (Variant B)
                "is_crisis_anchor": False,
                "active_in_pool": True,
                "parent_win_idx": -1,
            })

        if nodes:
            thr_crisis = float(np.quantile(vol_scores, 0.85)) if len(vol_scores) >= 6 else 999.0
            for nd in nodes:
                if nd["crisis_q90"] >= thr_crisis and nd["verified_skill"] > 0.005:
                    nd["is_crisis_anchor"] = True
        levels_data[h_key] = nodes

    # Привязка родительских ветвей H_parent -> H_child и кластерная консолидация старых спокойных листьев
    for lvl_i in range(1, len(TREE_LEVELS_DESC)):
        p_h = str(TREE_LEVELS_DESC[lvl_i - 1][0])
        c_h = str(TREE_LEVELS_DESC[lvl_i][0])
        parents = [p for p in levels_data[p_h] if p.get("active_in_pool", True)]
        children = levels_data[c_h]
        if not parents or not children:
            continue

        by_parent: dict[int, list[dict]] = {p["win_idx"]: [] for p in parents}
        p_mus = np.array([p["mu"] for p in parents], dtype=np.float32)

        for ch in children:
            c_s, c_e = ch["tr_start_sec"], ch["tr_end_sec"]
            best_p = -1
            best_ov = -1
            for p in parents:
                ov = min(c_e, p["tr_end_sec"]) - max(c_s, p["tr_start_sec"])
                if ov > best_ov:
                    best_ov = ov
                    best_p = p["win_idx"]
            if best_ov <= 0:
                # Fallback по близости нормированного вектора состояния
                c_mu = np.array(ch["mu"], dtype=np.float32)
                d2 = np.mean((p_mus - c_mu[None, :]) ** 2, axis=1)
                best_p = parents[int(np.argmin(d2))]["win_idx"]
            ch["parent_win_idx"] = int(best_p)
            by_parent[int(best_p)].append(ch)

        # Кластерная консолидация избыточных плотных уровней (например 391 окно BO H=1s и 194 окна H=2s):
        # сохраняем все свежие окна хвоста, лучшие кризисные якоря, 20 разнообразных медоидов по децилям режима
        # и равномерную историческую сетку T_now, сжимая 391 -> ~58 репрезентативных прототипов без потери эпох!
        n_ch_total = len(children)
        if n_ch_total > 40:
            fresh_ids = {nd["win_idx"] for nd in children[-10:]}
            crisis_cands = sorted([nd for nd in children if nd["is_crisis_anchor"]], key=lambda d: d["verified_skill"], reverse=True)
            crisis_ids = {nd["win_idx"] for nd in crisis_cands[:10]}
            calm_cands = [nd for nd in children if nd["win_idx"] not in fresh_ids and nd["win_idx"] not in crisis_ids]
            calm_sorted = sorted(calm_cands, key=lambda d: (d["mu"][0] + d["mu"][1] + 0.35 * d["mu"][5]))
            n_k = 20
            medoid_ids = set()
            for b_i in range(n_k):
                s_b = (b_i * len(calm_sorted)) // n_k
                e_b = ((b_i + 1) * len(calm_sorted)) // n_k
                bucket = calm_sorted[s_b:e_b]
                if bucket:
                    medoid_ids.add(max(bucket, key=lambda d: d["verified_skill"])["win_idx"])
            epoch_stride = max(1, n_ch_total // 16)
            epoch_ids = {children[k]["win_idx"] for k in range(0, n_ch_total, epoch_stride)}
            keep_all = fresh_ids | crisis_ids | medoid_ids | epoch_ids
            for nd in children:
                nd["active_in_pool"] = bool(nd["win_idx"] in keep_all)

    tree_doc = {
        "coin": coin_l,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horizons_desc": [h for h, _ in TREE_LEVELS_DESC],
        "levels": levels_data,
    }
    idx_path.write_text(json.dumps(tree_doc, indent=2), encoding="utf-8")
    tot_raw = sum(len(v) for v in levels_data.values())
    tot_act = sum(sum(1 for x in v if x["active_in_pool"]) for v in levels_data.values())
    print(f"🌳 [{coin_l.upper()}] Иерархический индекс дерева L28 построен за {time.time() - t0:.1f}с: {tot_raw} окон -> {tot_act} активных прототипов/якорей.", flush=True)
    return tree_doc


class HierarchicalRegimeTreeRouter:
    """
    Исполняемый иерархический древовидный роутер 7 горизонтов `[1s, 2s, 5s, 12s, 25s, 60s, 120s]`:
      - 100% идентичная логика выбора `Top-2 Tree + 1 Live T_now` и гибридного предсказания
        (`LightGBM + Top FastKerasModel`) в пакетной сборке кэша и в живом `1 Гц` трейдере!
    """

    def __init__(self, coin: str, use_hybrid_nn: bool = True, nn_blend_weight: float = 0.22, live_anchor_weight: float = 0.35):
        self.coin = coin.lower()
        self.use_hybrid_nn = bool(use_hybrid_nn)
        self.nn_blend_weight = float(nn_blend_weight)
        # Адаптивные веса NN-ансамбля по горизонту (Variant B, исследование 2026-11-16)
        # Средний оптимальный LGB-вес: H=5s→82%, H=25s→64%, H=60s→60%, H=120s→38%
        self._nn_weight_by_h: dict[int, float] = {
            1: 0.18, 2: 0.18, 5: 0.18,       # короткие: LGB доминирует
            12: 0.35, 25: 0.36,               # средние: NN набирают силу
            60: 0.40, 90: 0.50, 120: 0.62,    # длинные: NN-ансамбль сильнее LGB
        }
        self.live_anchor_weight = float(live_anchor_weight)
        self.models_root = BASE_DIR / "models" / self.coin
        self.tree_doc = build_or_load_tree_index(self.coin, force_rebuild=False)

        feats_json = BASE_DIR / "data" / "features" / f"selected_features_{self.coin}_h8s.json"
        self.feature_names = json.loads(feats_json.read_text(encoding="utf-8"))["feature_order"]

        self._lgb_cache: dict[str, lgb.Booster] = {}
        self._nn_cache: dict[str, tuple[FastKerasModel, np.ndarray, np.ndarray, int]] = {}
        self._live_ema_state: dict[int, np.ndarray] = {}
        self._pool_by_h: dict[int, list[dict]] = {}
        self._pool_arrays_by_h: dict[int, dict[str, np.ndarray]] = {}
        self._all_tr_ends_by_h: dict[int, np.ndarray] = {}
        self._precompile_pools()

    def _precompile_pools(self):
        for H, _ in TREE_LEVELS_DESC:
            raw_nodes = self.tree_doc["levels"].get(str(H), [])
            self._all_tr_ends_by_h[H] = np.sort(np.array([nd["tr_end_sec"] for nd in raw_nodes], dtype=np.int64)) if raw_nodes else np.zeros(0, dtype=np.int64)
            pool = [nd for nd in raw_nodes if nd.get("active_in_pool", True)]
            self._pool_by_h[H] = pool
            if pool:
                self._pool_arrays_by_h[H] = {
                    "mu": np.array([nd["mu"] for nd in pool], dtype=np.float32),
                    "sd": np.array([nd["sd"] for nd in pool], dtype=np.float32),
                    "tr_ends": np.array([nd["tr_end_sec"] for nd in pool], dtype=np.int64),
                    "te_ends": np.array([nd["te_end_sec"] for nd in pool], dtype=np.int64),
                    "v_skill": np.array([nd["verified_skill"] for nd in pool], dtype=np.float32),
                    "p_skill": np.array([nd["prior_skill"] for nd in pool], dtype=np.float32),
                    "is_crisis": np.array([1.0 if nd["is_crisis_anchor"] else 0.0 for nd in pool], dtype=np.float32),
                }

    def _get_booster(self, rel_path: str) -> lgb.Booster:
        if rel_path not in self._lgb_cache:
            self._lgb_cache[rel_path] = lgb.Booster(model_file=str(self.models_root / rel_path))
        return self._lgb_cache[rel_path]

    def _get_nn(self, nn_info: dict, batch_gpu: bool = False) -> tuple[FastKerasModel, np.ndarray, np.ndarray, int]:
        rel = nn_info["keras_rel"]
        if rel not in self._nn_cache:
            meta = json.loads((self.models_root / nn_info["meta_rel"]).read_text(encoding="utf-8"))
            mean = np.asarray(meta["feature_norm"]["mean"], dtype=np.float32)
            std = np.asarray(meta["feature_norm"]["std"], dtype=np.float32)
            fkm = FastKerasModel(self.models_root / rel, nn_info["arch"])
            self._nn_cache[rel] = (fkm, mean, std, int(nn_info["seq_len"]))
        fkm, mean, std, sl = self._nn_cache[rel]
        if batch_gpu and torch.cuda.is_available():
            fkm.to_device("cuda")
        return fkm, mean, std, sl

    def _predict_node_probs_batch(self, node: dict, X_full: np.ndarray, start_i: int, end_i: int) -> np.ndarray:
        """Пакетный прогноз одного гибридного узла: LGB + полный NN-ансамбль (Variant B)."""
        x_slice = np.nan_to_num(np.asarray(X_full[start_i:end_i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        p_lgb = np.asarray(self._get_booster(node["lgb_rel"]).predict(x_slice, num_threads=SAFE_WORKERS), dtype=np.float32)

        nn_list = node.get("all_nn") or ([node["best_nn"]] if node.get("best_nn") else [])
        if not self.use_hybrid_nn or not nn_list:
            return p_lgb

        H = node.get("H", 5)
        w_nn_total = self._nn_weight_by_h.get(H, self.nn_blend_weight)

        nn_preds = []
        nn_ics = []
        for nn_info in nn_list:
            try:
                fkm, mean, std, sl = self._get_nn(nn_info, batch_gpu=True)
                ctx_s = max(0, start_i - (sl - 1))
                need_left = (sl - 1) - (start_i - ctx_s)
                x_ctx = np.nan_to_num(np.asarray(X_full[ctx_s:end_i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                if need_left > 0:
                    x_ctx = np.vstack([np.repeat(x_ctx[:1], need_left, axis=0), x_ctx])
                x_norm = np.clip((x_ctx - mean) / std, -5.0, 5.0)
                p_nn = fkm.predict_sequence_2d(x_norm, seq_len=sl, pre_padded=True)
                nn_preds.append(p_nn)
                nn_ics.append(max(0.001, nn_info.get("ic", 0.05)))
            except Exception:
                continue

        if not nn_preds:
            return p_lgb

        nn_ics_arr = np.array(nn_ics, dtype=np.float32)
        nn_weights = nn_ics_arr / nn_ics_arr.sum()
        p_nn_ensemble = sum(w * p for w, p in zip(nn_weights, nn_preds))
        return ((1.0 - w_nn_total) * p_lgb + w_nn_total * p_nn_ensemble).astype(np.float32)

    def _predict_node_probs_live_1row(self, node: dict, x_ring_2d: np.ndarray) -> np.ndarray:
        """Онлайн-прогноз одного гибридного узла: LGB + полный NN-ансамбль (Variant B)."""
        x_last = np.nan_to_num(np.asarray(x_ring_2d[-1:], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        p_lgb = np.asarray(self._get_booster(node["lgb_rel"]).predict(x_last, num_threads=1), dtype=np.float32)[0]

        nn_list = node.get("all_nn") or ([node["best_nn"]] if node.get("best_nn") else [])
        if not self.use_hybrid_nn or not nn_list:
            return p_lgb

        # Адаптивный вес NN по горизонту
        H = node.get("H", 5)
        w_nn_total = self._nn_weight_by_h.get(H, self.nn_blend_weight)

        # Предсказания всех NN с IC-взвешиванием
        nn_preds = []
        nn_ics = []
        for nn_info in nn_list:
            try:
                fkm, mean, std, sl = self._get_nn(nn_info)
                if len(x_ring_2d) >= sl:
                    seq_raw = x_ring_2d[-sl:]
                else:
                    pad = np.repeat(x_ring_2d[:1], sl - len(x_ring_2d), axis=0)
                    seq_raw = np.vstack([pad, x_ring_2d])
                seq_clean = np.nan_to_num(np.asarray(seq_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                seq_n = np.clip((seq_clean - mean) / std, -5.0, 5.0).astype(np.float32)
                p_nn = fkm.predict_numpy(seq_n[np.newaxis, :, :])[0]
                nn_preds.append(p_nn)
                nn_ics.append(max(0.001, nn_info.get("ic", 0.05)))
            except Exception:
                continue

        if not nn_preds:
            return p_lgb

        # IC-взвешенное усреднение NN
        nn_ics = np.array(nn_ics, dtype=np.float32)
        nn_weights = nn_ics / nn_ics.sum()
        p_nn_ensemble = sum(w * p for w, p in zip(nn_weights, nn_preds))

        return ((1.0 - w_nn_total) * p_lgb + w_nn_total * p_nn_ensemble).astype(np.float32)

    def compute_hierarchical_weights_batch(
        self,
        states: dict[str, np.ndarray],
        ts_sec: np.ndarray,
        ema_init_by_h: dict[int, np.ndarray] | None = None,
    ) -> tuple[dict[int, list[dict]], dict[int, np.ndarray], np.ndarray]:
        """
        Нисходящий каузальный расчёт весов по дереву 120s -> 60s -> 25s -> 12s -> 5s -> 2s -> 1s
        с выбором не более 3 активных узлов на каждую секунду (Top-2 Tree + 1 Live T_now).
        Поддерживает продолжение каузального EMA через `ema_init_by_h` для инкрементального режима!
        """
        n = len(ts_sec)
        nodes_by_h: dict[int, list[dict]] = {}
        weights_by_h: dict[int, np.ndarray] = {}
        parent_log_prior_by_win: dict[int, dict[int, np.ndarray]] = {}
        ema_last_out: dict[int, np.ndarray] = {}
        crisis_vec = states["crisis_intensity"]

        for H, _ in TREE_LEVELS_DESC:
            pool = self._pool_by_h.get(H, [])
            nodes_by_h[H] = pool
            m_cnt = len(pool)
            if m_cnt == 0:
                weights_by_h[H] = np.zeros((n, 0), dtype=np.float32)
                continue

            arrs = self._pool_arrays_by_h[H]
            s_mat = states[_scale_key_for_h(H)]
            mu_mat = arrs["mu"]
            sd_mat = arrs["sd"]
            tr_ends = arrs["tr_ends"]
            te_ends = arrs["te_ends"]
            v_skill = arrs["v_skill"]
            p_skill = arrs["p_skill"]
            is_crisis = arrs["is_crisis"]

            # Чанковый расчёт локальных логитов без пиков RAM (даже при N=3.1M и M=200):
            w_out = np.zeros((n, m_cnt), dtype=np.float32)
            p_map = parent_log_prior_by_win.get(H, {})

            chunk_sz = 200000
            ema_prev = None
            if ema_init_by_h is not None and H in ema_init_by_h and len(ema_init_by_h[H]) == m_cnt:
                ema_prev = np.asarray(ema_init_by_h[H], dtype=np.float32).copy()
            tau_h = EMA_TAU_BY_H.get(H, 0.0)
            alpha_ema = float(1.0 - math.exp(-1.0 / max(0.25, tau_h))) if tau_h > 0.25 else 1.0

            for c0 in range(0, n, chunk_sz):
                c1 = min(n, c0 + chunk_sz)
                s_c = s_mat[c0:c1]
                ts_c = ts_sec[c0:c1]
                cr_c = crisis_vec[c0:c1]

                # Масштабно-согласованное расстояние Махаланобиса
                diff = np.clip((s_c[:, None, :] - mu_mat[None, :, :]) / sd_mat[None, :, :], -3.5, 3.5)
                sim_logits = -np.mean(diff * diff, axis=-1) / 0.45  # [C, M]

                # Мягкий иерархический приор от родителя + Global Crisis Bypass!
                if p_map:
                    inh = np.zeros((c1 - c0, m_cnt), dtype=np.float32)
                    for j_m, nd in enumerate(pool):
                        pw = nd["parent_win_idx"]
                        if pw in p_map:
                            inh[:, j_m] = p_map[pw][c0:c1]
                    # При высоком crisis_intensity кризисные якоря игнорируют штраф от спокойного 21-дневного родителя:
                    bypass_boost = (cr_c[:, None] * is_crisis[None, :]) * 1.35
                    hier_term = np.where(
                        (is_crisis[None, :] > 0.5) & (cr_c[:, None] > 0.25),
                        np.maximum(inh, -0.25) + bypass_boost,
                        inh,
                    )
                    raw_logits = sim_logits + 0.55 * hier_term
                else:
                    raw_logits = sim_logits + (cr_c[:, None] * is_crisis[None, :]) * 0.90

                # Каузальное многомасштабное EMA сглаживание логитов
                if alpha_ema < 0.999:
                    if ema_prev is not None:
                        raw_logits[0] = (1.0 - alpha_ema) * ema_prev + alpha_ema * raw_logits[0]
                    raw_logits = _causal_ema_2d(raw_logits, tau_h)
                    ema_prev = raw_logits[-1].copy()

                # Каузальная маска доступности окна (ts > tr_end) и каузального вступления в силу OOS Skill (ts > te_end)
                avail = ts_c[:, None] > tr_ends[None, :]
                skill_known = ts_c[:, None] > te_ends[None, :]
                eff_skill = np.where(skill_known, v_skill[None, :], p_skill[None, :])
                # Отсекаем окна с нулевым подтвержденным skill!
                valid_node = avail & (eff_skill > 1e-4)

                logits_masked = np.where(valid_node, raw_logits + 0.65 * np.log(np.maximum(eff_skill, 1e-4)), -1e9)

                # Выбираем Top-2 из дерева + 1 каузальный якорь современности `T_now`
                any_valid = np.any(valid_node, axis=1)
                # Индекс самого свежего доступного окна `T_now` (по максимальному tr_end < ts)
                order_tr = np.argsort(tr_ends)
                tr_sorted = tr_ends[order_tr]
                pos_lat = np.searchsorted(tr_sorted, ts_c - 1, side="right") - 1
                has_any_avail = pos_lat >= 0
                lat_pool_idx = order_tr[np.maximum(pos_lat, 0)]

                # Top-2 узла дерева в каждой строке
                if m_cnt <= 2:
                    top2_idx = np.tile(np.arange(m_cnt)[None, :], (c1 - c0, 1))
                else:
                    top2_idx = np.argpartition(logits_masked, -2, axis=1)[:, -2:]

                top2_vals = np.take_along_axis(logits_masked, top2_idx, axis=1)
                vmax = np.max(top2_vals, axis=1, keepdims=True)
                e2 = np.exp(np.clip(top2_vals - vmax, -20.0, 0.0)) * (top2_vals > -1e8)
                s2 = np.sum(e2, axis=1, keepdims=True)
                w2 = np.where(s2 > 1e-8, e2 / np.maximum(s2, 1e-8), 0.0)

                w_c = np.zeros((c1 - c0, m_cnt), dtype=np.float32)
                rows_arange = np.arange(c1 - c0)
                a_live = self.live_anchor_weight
                for k_col in range(top2_idx.shape[1]):
                    w_c[rows_arange, top2_idx[:, k_col]] += np.where(any_valid, (1.0 - a_live) * w2[:, k_col], 0.0)
                w_c[rows_arange, lat_pool_idx] += np.where(has_any_avail, np.where(any_valid, a_live, 1.0), 0.0)

                w_out[c0:c1] = w_c

            weights_by_h[H] = w_out
            if ema_prev is not None:
                ema_last_out[H] = ema_prev.copy()

            # Формируем лог-приор для следующего дочернего этажа дерева
            idx_cur = TREE_LEVELS_DESC.index((H, dict(TREE_LEVELS_DESC)[H]))
            if idx_cur + 1 < len(TREE_LEVELS_DESC):
                next_h = TREE_LEVELS_DESC[idx_cur + 1][0]
                p_dict = {}
                for j_m, nd in enumerate(pool):
                    p_dict[nd["win_idx"]] = np.log(np.clip(w_out[:, j_m], 0.02, 1.0)).astype(np.float32)
                parent_log_prior_by_win[next_h] = p_dict

        self._last_batch_ema_by_h = ema_last_out

        # Зрелость 7 горизонтов L28_HORIZONS_ASC
        maturity_7d = np.zeros((n, 7), dtype=np.float32)
        for h_i, (H, _) in enumerate(L28_HORIZONS_ASC):
            raw_nodes = self.tree_doc["levels"].get(str(H), [])
            if raw_nodes:
                tr_e = np.sort(np.array([nd["tr_end_sec"] for nd in raw_nodes], dtype=np.int64))
                n_open = np.searchsorted(tr_e, ts_sec - 1, side="right").astype(np.float32)
                maturity_7d[:, h_i] = np.log1p(n_open) / 4.0

        return nodes_by_h, weights_by_h, maturity_7d

    def predict_all_7_horizons_batch(
        self,
        X_all: np.ndarray,
        states: dict[str, np.ndarray],
        ts_sec: np.ndarray,
        stale_mask: np.ndarray | None = None,
        min_eval_row: int = 0,
        ema_init_by_h: dict[int, np.ndarray] | None = None,
        x_global_offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Турбо-пакетный каузальный прогон 7 горизонтов дерева (с полной поддержкой инкрементального режима!):
          1) Однократная материализация рабочего среза `X_work` (`float32`, без повторных `nan_to_num`);
          2) Для каждого прототипа LightGBM вызывается РОВНО 1 РАЗ по вектору его активных строк `act_idx`;
          3) Нейросеть `FastKerasModel` держит веса резидентно в VRAM и считается по склеенным сегментам
             (`gaps > 8192`) с `pre_padded=True` без `unfold`-дублирования и без `cuda.empty_cache()`.
        """
        n = len(ts_sec)
        stale_1d = np.asarray(stale_mask, dtype=bool).reshape(-1)[:n] if stale_mask is not None else None
        nodes_by_h, weights_by_h, maturity_7d = self.compute_hierarchical_weights_batch(
            states, ts_sec, ema_init_by_h=ema_init_by_h
        )
        mid_px = states["mid_price"]
        tree_42d = np.zeros((n, 42), dtype=np.float32)

        # Один раз выгружаем и чистим рабочий срез признаков в RAM с запасом 360 строк под максимальный seq_len:
        max_ctx = 360
        work_g0 = max(0, (x_global_offset + min_eval_row) - max_ctx)
        work_g1 = x_global_offset + n
        X_work = np.nan_to_num(np.asarray(X_all[work_g0:work_g1], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        w_nn_default = self.nn_blend_weight
        for h_i, (H, _) in enumerate(L28_HORIZONS_ASC):
            th0 = time.time()
            pool = nodes_by_h[H]
            w_mat = weights_by_h[H]  # [N, M]
            if not pool:
                continue

            w_nn = self._nn_weight_by_h.get(H, w_nn_default)
            probs_acc = np.zeros((n, 7), dtype=np.float32)
            has_w = np.sum(w_mat, axis=1) > 1e-5
            if min_eval_row > 0:
                has_w[:min_eval_row] = False

            n_calls = 0
            for j_m, nd in enumerate(pool):
                wj = w_mat[:, j_m]
                act_idx = np.where(wj > 1e-4)[0]
                if min_eval_row > 0:
                    act_idx = act_idx[act_idx >= min_eval_row]
                if len(act_idx) == 0:
                    continue

                # 1) Табличный LightGBM — РОВНО 1 вызов на все активные тики прототипа!
                act_w_idx = (x_global_offset + act_idx) - work_g0
                p_lgb_act = np.asarray(
                    self._get_booster(nd["lgb_rel"]).predict(X_work[act_w_idx], num_threads=SAFE_WORKERS),
                    dtype=np.float32,
                )

                nn_list = nd.get("all_nn") or ([nd["best_nn"]] if nd.get("best_nn") else [])
                use_nn_node = self.use_hybrid_nn and bool(nn_list) and (w_nn > 0.0)
                if not use_nn_node:
                    probs_acc[act_idx] += wj[act_idx, None] * p_lgb_act
                    n_calls += 1
                    continue

                probs_acc[act_idx] += ((1.0 - w_nn) * wj[act_idx, None]) * p_lgb_act

                # 2) NN-ансамбль: IC-взвешенное усреднение всех доступных архитектур
                nn_ics = np.array([max(0.001, info.get("ic", 0.05)) for info in nn_list], dtype=np.float32)
                nn_weights_arr = nn_ics / nn_ics.sum()

                for nn_idx, nn_info in enumerate(nn_list):
                    try:
                        fkm, mean, std, sl = self._get_nn(nn_info, batch_gpu=True)
                    except Exception:
                        continue
                    nn_w_coeff = w_nn * float(nn_weights_arr[nn_idx])
                    gaps = np.where(np.diff(act_idx) > 8192)[0]
                    seg_starts = np.r_[act_idx[0], act_idx[gaps + 1]]
                    seg_ends = np.r_[act_idx[gaps] + 1, act_idx[-1] + 1]
                    for s_seg, e_seg in zip(seg_starts, seg_ends):
                        for b0 in range(int(s_seg), int(e_seg), 262144):
                            b1 = min(int(e_seg), b0 + 262144)
                            g_b0 = x_global_offset + b0
                            g_b1 = x_global_offset + b1
                            ctx_g0 = max(work_g0, g_b0 - (sl - 1))
                            need_left = (sl - 1) - (g_b0 - ctx_g0)
                            x_ctx = X_work[(ctx_g0 - work_g0) : (g_b1 - work_g0)]
                            if need_left > 0:
                                x_ctx = np.vstack([np.repeat(x_ctx[:1], need_left, axis=0), x_ctx])
                            x_norm = np.clip((x_ctx - mean) / std, -5.0, 5.0)
                            p_nn = fkm.predict_sequence_2d(x_norm, seq_len=sl, pre_padded=True)
                            w_sl = wj[b0:b1]
                            nz = w_sl > 1e-4
                            if np.all(nz):
                                probs_acc[b0:b1] += (nn_w_coeff * w_sl[:, None]) * p_nn
                            else:
                                probs_acc[b0:b1][nz] += (nn_w_coeff * w_sl[nz, None]) * p_nn[nz]
                            n_calls += 1

            h_6d = np.zeros((n, 6), dtype=np.float32)
            if np.any(has_w):
                h_6d[has_w] = extract_6d_from_probs_l28(probs_acc[has_w], H, mid_px[has_w])
            if stale_1d is not None:
                h_6d[stale_1d] = 0.0

            tree_42d[:, h_i * 6 : (h_i + 1) * 6] = h_6d
            self._lgb_cache.clear()
            self._nn_cache.clear()
            gc.collect()
            print(
                f"  🌳 [{self.coin.upper():<6} H={H:3d}s] пул={len(pool):3d} узлов | сегментных батчей={n_calls:4d} "
                f"| активных тиков={int(np.sum(has_w)):>9,} ({time.time() - th0:.1f}с)",
                flush=True,
            )

        del X_work
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        coherence_4d = compute_tree_coherence_4d(tree_42d)
        if stale_1d is not None:
            coherence_4d[stale_1d] = 0.0
        return tree_42d, maturity_7d, coherence_4d

    def predict_live_1row(
        self,
        x_ring_2d: np.ndarray,
        live_row,
        now_ts_sec: int,
        is_stale: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Живой пошаговый расчёт (1 Гц) с 100% математическим паритетом к `predict_all_7_horizons_batch`.
        Принимает dict/Series `live_row` (или 1-row DataFrame) и работает за ~5–8 мс на CPU.
        """
        if is_stale:
            return np.zeros(42, dtype=np.float32), np.zeros(7, dtype=np.float32), np.zeros(4, dtype=np.float32)

        if isinstance(live_row, pd.DataFrame):
            st_1 = extract_venue_invariant_states(live_row.iloc[-1:], self.coin)
            mid_1 = float(st_1["mid_price"][0])
            cr_1 = float(st_1["crisis_intensity"][0])
            s_by_k = {k: st_1[k][0] for k in ("macro", "meso", "micro")}
        else:
            st_1 = extract_venue_invariant_state_1row(live_row, self.coin)
            mid_1 = float(st_1["mid_price"])
            cr_1 = float(st_1["crisis_intensity"])
            s_by_k = {k: st_1[k] for k in ("macro", "meso", "micro")}

        probs_by_h: dict[int, np.ndarray] = {}
        parent_prior: dict[int, float] = {}

        for H, _ in TREE_LEVELS_DESC:
            pool = self._pool_by_h.get(H, [])
            if not pool:
                continue
            arrs = self._pool_arrays_by_h[H]
            s_vec = s_by_k[_scale_key_for_h(H)]
            mu_mat = arrs["mu"]
            sd_mat = arrs["sd"]
            is_cr = arrs["is_crisis"]
            diff = np.clip((s_vec[None, :] - mu_mat) / sd_mat, -3.5, 3.5)
            sim_l = -np.mean(diff * diff, axis=-1) / 0.45

            if parent_prior:
                inh = np.array([parent_prior.get(nd["parent_win_idx"], 0.0) for nd in pool], dtype=np.float32)
                hier = np.where((is_cr > 0.5) & (cr_1 > 0.25), np.maximum(inh, -0.25) + cr_1 * is_cr * 1.35, inh)
                raw_l = sim_l + 0.55 * hier
            else:
                raw_l = sim_l + cr_1 * is_cr * 0.90

            tau_h = EMA_TAU_BY_H.get(H, 0.0)
            if tau_h > 0.25:
                alpha = float(1.0 - math.exp(-1.0 / tau_h))
                if H in self._live_ema_state and len(self._live_ema_state[H]) == len(raw_l):
                    raw_l = (1.0 - alpha) * self._live_ema_state[H] + alpha * raw_l
                self._live_ema_state[H] = raw_l.copy()

            tr_ends = arrs["tr_ends"]
            te_ends = arrs["te_ends"]
            v_sk = arrs["v_skill"]
            p_sk = arrs["p_skill"]

            avail = now_ts_sec > tr_ends
            eff_sk = np.where(now_ts_sec > te_ends, v_sk, p_sk)
            valid = avail & (eff_sk > 1e-4)
            if not np.any(avail):
                continue

            masked_l = np.where(valid, raw_l + 0.65 * np.log(np.maximum(eff_sk, 1e-4)), -1e9)
            avail_idx = np.where(avail)[0]
            lat_i = int(avail_idx[np.argmax(tr_ends[avail_idx])])

            w_vec = np.zeros(len(pool), dtype=np.float32)
            if np.any(valid):
                top2 = np.argsort(masked_l)[-2:] if len(pool) > 2 else np.arange(len(pool))
                v2 = masked_l[top2]
                e2 = np.exp(np.clip(v2 - np.max(v2), -20.0, 0.0)) * (v2 > -1e8)
                e2 /= max(float(np.sum(e2)), 1e-8)
                for k_i, idx_n in enumerate(top2):
                    w_vec[idx_n] += (1.0 - self.live_anchor_weight) * float(e2[k_i])
                w_vec[lat_i] += self.live_anchor_weight
            else:
                w_vec[lat_i] = 1.0

            p_blend = np.zeros(7, dtype=np.float32)
            for idx_n in np.where(w_vec > 1e-4)[0]:
                p_blend += w_vec[idx_n] * self._predict_node_probs_live_1row(pool[int(idx_n)], x_ring_2d)
            probs_by_h[H] = p_blend
            parent_prior = {pool[i_n]["win_idx"]: float(math.log(max(float(w_vec[i_n]), 0.02))) for i_n in range(len(pool))}

        out_42 = np.zeros(42, dtype=np.float32)
        mat_7 = np.zeros(7, dtype=np.float32)
        for h_i, (H, _) in enumerate(L28_HORIZONS_ASC):
            tr_e = self._all_tr_ends_by_h.get(H)
            if tr_e is not None and len(tr_e) > 0:
                n_op = int(np.searchsorted(tr_e, now_ts_sec - 1, side="right"))
                mat_7[h_i] = math.log1p(n_op) / 4.0
            if H in probs_by_h:
                out_42[h_i * 6 : (h_i + 1) * 6] = extract_6d_from_probs_l28(probs_by_h[H], H, mid_1)[0]

        coh_4 = compute_tree_coherence_4d(out_42)[0]
        return out_42, mat_7, coh_4


def run_self_test_and_benchmark():
    """Самопроверка индексов ST и BO, замер live-латентности и сравнение OOS IC на свежем хвосте."""
    print("=" * 108, flush=True)
    print("🧪 САМОПРОВЕРКА HIERARCHICAL REGIME TREE ROUTER (ST + BO, 7 ГОРИЗОНТОВ БЕЗ 90s)", flush=True)
    print("=" * 108, flush=True)
    for c_name in ("prl", "prl_bo"):
        build_or_load_tree_index(c_name, force_rebuild=True)

    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    pf = pq.ParquetFile(labeled_path)
    n_tot = pf.metadata.num_rows
    tail_n = 45000  # последние 12.5 часов чистой эры $0.0001
    s_idx = n_tot - tail_n

    req_cols = [c for c in required_router_state_columns() if c in set(pf.schema_arrow.names)]
    df_tail = pq.read_table(labeled_path, columns=req_cols).slice(s_idx, tail_n).to_pandas()
    raw_tail_ts = df_tail["timestamp"] if "timestamp" in df_tail.columns else df_tail.index
    ts_tail = pd.to_datetime(np.asarray(raw_tail_ts), utc=True).as_unit("s").astype("int64").to_numpy()
    tail_records = df_tail.iloc[-350:].to_dict(orient="records")

    for coin_l, csub in (("prl", "_mmap_cache"), ("prl_bo", "_mmap_cache_bo")):
        cdir = BASE_DIR / "data" / csub
        X_mm = np.load(cdir / "X_all.npy", mmap_mode="r")
        X_tail = np.asarray(X_mm[s_idx:n_tot], dtype=np.float32)
        states = extract_venue_invariant_states(df_tail, coin_l)

        router = HierarchicalRegimeTreeRouter(coin_l, use_hybrid_nn=True)
        tree_42, mat_7, coh_4 = router.predict_all_7_horizons_batch(X_tail, states, ts_tail)

        # Замер OOS IC по всем 7 горизонтам на хвосте 45,000 с
        ics_str = []
        for h_i, (H, _) in enumerate(L28_HORIZONS_ASC):
            dt_true = np.asarray(np.load(cdir / f"dt_all_h{H}.npy", mmap_mode="r")[s_idx:n_tot], dtype=np.float32)
            vel_pred = tree_42[:, h_i * 6 + 2]
            ic = float(np.corrcoef(vel_pred, dt_true)[0, 1]) if np.std(vel_pred) > 1e-6 and np.std(dt_true) > 1e-6 else 0.0
            ics_str.append(f"H={H}s:{ic:+.4f}")
        print(f"📈 [{coin_l.upper():<6} Tail-45k OOS IC] " + " | ".join(ics_str), flush=True)

        # Проверка паритета Batch vs Live 1-row (при прогретом EMA 350с = 11.6*tau_max) + замер латентности 1 Гц шага
        router._live_ema_state.clear()
        start_w = tail_n - 350
        for off_i, k_w in enumerate(range(start_w, tail_n - 1)):
            ring_w = X_tail[max(0, k_w - 360 + 1) : k_w + 1]
            router.predict_live_1row(ring_w, tail_records[off_i], int(ts_tail[k_w]))

        ema_snapshot = {k_e: v_e.copy() for k_e, v_e in router._live_ema_state.items()}
        ring_last = X_tail[-360:]
        last_rec = tail_records[-1]
        o42_live, _, c4_live = router.predict_live_1row(ring_last, last_rec, int(ts_tail[-1]))
        max_diff = float(np.max(np.abs(o42_live - tree_42[-1])))

        t_lat = []
        for _ in range(25):
            router._live_ema_state = {k_e: v_e.copy() for k_e, v_e in ema_snapshot.items()}
            tl0 = time.perf_counter()
            router.predict_live_1row(ring_last, last_rec, int(ts_tail[-1]))
            t_lat.append((time.perf_counter() - tl0) * 1000.0)
        print(
            f"⚡ [{coin_l.upper():<6} Live Parity & Latency] median_step={np.median(t_lat):.2f} ms | max|Batch - Live|={max_diff:.6f} | coh={np.round(c4_live, 3).tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        run_self_test_and_benchmark()
