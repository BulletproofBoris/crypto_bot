#!/usr/bin/env python3
"""
_tools/order_orchestrator.py
=============================================================================
Детерминированный Контур Оркестрации Капитала, Защиты Безубытка и Сетки Заявок
(Deterministic Capital, Break-Even & Avellaneda-Stoikov Grid Orchestrator v2):

Принимает непрерывные выходы обученного чемпиона Order-3 (plan) и накладывает
строгие математические инварианты исполнения без случайных весов:
  1. Защита безубытка (Break-Even Floor Guard): ни один лимитный Ask не ставится
     ниже себестоимости транша + 2x комиссии + минимальной маржи (+12 bps).
  2. Инвентарный перекос Авелланеды-Стойкова (Inventory Skew): при росте позиции
     свыше 25% капитала Bid-сетка автоматически отодвигается глубже, а при >=62%
     или высокой токсичности стакана новые покупки блокируются.
  3. Сжатие рабочей глубины (Dynamic Grid Squeeze): концентрация слотов в рабочем
     коридоре волатильности (-0.05% .. -1.40%) вместо мёртвого бункера -3.5%.
  4. Анти-пыль (Anti-Dust Guarantee): min $25+ на активный Bid-ордер (>= 2.0 PRL).
  5. Строгая монотонность цен: P_bid[0] > P_bid[1] > ... и P_ask[0] < P_ask[1] < ...
=============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ORCH_MKT_DIM = 40
ORCH_ACCT_DIM = 8
ORCH_IN_DIM = ORCH_MKT_DIM + ORCH_ACCT_DIM  # 48D

MIN_WORKING_ORDER_USD = 25.0
MIN_PRL_AMOUNT = 2.0
MIN_NET_PROFIT_BPS = 12.0  # +12 bps чистого профита сверх двойной комиссии мейкера

GRID_MODES = {
    0: {"name": "Sniper_2Slots",    "b_slots": 2,  "a_slots": 2,  "front_share": 0.65, "mid_share": 0.35, "deep_share": 0.00},
    1: {"name": "Ladder_4Slots",    "b_slots": 4,  "a_slots": 4,  "front_share": 0.45, "mid_share": 0.40, "deep_share": 0.15},
    2: {"name": "Balanced_6Slots",  "b_slots": 6,  "a_slots": 6,  "front_share": 0.35, "mid_share": 0.35, "deep_share": 0.30},
    3: {"name": "DeepGrid_12Slots", "b_slots": 12, "a_slots": 12, "front_share": 0.25, "mid_share": 0.35, "deep_share": 0.40},
}


class OrderOrchestratorNet(nn.Module):
    """Сохранено для обратной совместимости модульных тестов (с фиксированным seed)."""

    def __init__(self, in_dim: int = ORCH_IN_DIM):
        super().__init__()
        self.in_dim = in_dim
        gen = torch.Generator().manual_seed(20261118)
        self.fc1 = nn.Linear(in_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.head = nn.Linear(32, 7)
        with torch.no_grad():
            for m in (self.fc1, self.fc2, self.head):
                w = torch.randn(m.weight.shape, generator=gen)
                q, _ = torch.linalg.qr(w.T if w.shape[0] < w.shape[1] else w)
                m.weight.copy_((q.T if w.shape[0] < w.shape[1] else q)[: w.shape[0], : w.shape[1]] * 0.25)
                m.bias.zero_()
            self.head.bias[0] = 0.85
            self.head.bias[2] = 1.40
            self.head.bias[3] = 1.15

    def forward(self, x_48d: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.silu(self.ln1(self.fc1(x_48d)))
        h = F.silu(self.ln2(self.fc2(h)))
        out = self.head(h)
        capital_exp = 0.15 + 0.80 * torch.sigmoid(out[:, 0])
        mode_probs = F.softmax(out[:, 1:5], dim=-1)
        echelon_bias = torch.tanh(out[:, 5])
        spread_tight = torch.sigmoid(out[:, 6])
        return {
            "capital_exp": capital_exp,
            "mode_probs": mode_probs,
            "echelon_bias": echelon_bias,
            "spread_tight": spread_tight,
        }


def build_orchestrator_acct_features(
    budget_usd: float,
    cash_usd: float,
    pos_qty: float,
    mid_px: float,
    avg_entry: float,
    hold_sec: float,
    spread_bps: float,
    vol_60s: float,
) -> np.ndarray:
    z = np.zeros(ORCH_ACCT_DIM, dtype=np.float32)
    tot_eq = max(100.0, cash_usd + pos_qty * mid_px)
    z[0] = float(np.clip(cash_usd / tot_eq, 0.0, 1.0))
    z[1] = float(np.clip((pos_qty * mid_px) / tot_eq, 0.0, 1.0))
    z[2] = float(math.log1p(tot_eq / 1000.0))
    if pos_qty >= MIN_PRL_AMOUNT and avg_entry > 1e-4:
        z[3] = float(np.clip(((mid_px - avg_entry) / avg_entry) * 20.0, -2.0, 2.0))
    z[4] = float(np.clip(hold_sec / 300.0, 0.0, 3.0))
    z[5] = float(np.clip(spread_bps / 50.0, 0.0, 3.0))
    z[6] = float(np.clip(vol_60s * 100.0, 0.0, 3.0))
    z[7] = 1.0 if pos_qty >= MIN_PRL_AMOUNT else 0.0
    return z


def compute_breakeven_floor_px(
    entry_px: float,
    tick_sz: float = 0.0001,
    maker_fee: float = 0.0010,
    min_profit_bps: float = MIN_NET_PROFIT_BPS,
) -> float:
    """
    Строгая точка безубытка продажи: покрывает входную комиссию, выходную комиссию
    и минимальную чистую маржу (+12 bps по умолчанию), округленная вверх до тика.
    """
    if entry_px <= 1e-6:
        return 0.0
    raw_floor = entry_px * ((1.0 + maker_fee) / max(1e-6, 1.0 - maker_fee)) * (1.0 + min_profit_bps * 1e-4)
    return round(math.ceil((raw_floor - 1e-9) / tick_sz) * tick_sz, 8)


def orchestrate_grid_orders(
    budget_usd: float,
    cash_usd: float,
    pos_qty: float,
    b1: float,
    a1: float,
    capital_exp: float,
    grid_mode_idx: int = 1,
    spread_tight: float = 0.5,
    tick_sz: float = 0.0001,
    maker_fee: float = 0.0010,
    min_order_usd: float = MIN_WORKING_ORDER_USD,
    avg_entry: float = 0.0,
    low_tranche_px: float = 0.0,
    low_tranche_qty: float = 0.0,
    toxicity_score: float = 0.0,
    model_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Превращает предсказания обученного чемпиона Order-3 (`model_plan`) или макро-директивы
    в строго монотонную, защищенную от убыточных продаж и перегруза инвентаря сетку 24 слотов.
    """
    tot_eq = max(100.0, budget_usd, cash_usd + pos_qty * b1)
    pos_usd = pos_qty * b1
    pos_ratio = float(np.clip(pos_usd / tot_eq, 0.0, 1.5))
    tox = float(np.clip(toxicity_score, 0.0, 1.0))

    # 1. Интеграция с обученным мозгом Order-3 Champion (если передан model_plan):
    if model_plan is not None:
        raw_exp = float(model_plan.get("exp_bid_frac", capital_exp))
        pi_b = np.asarray(model_plan["pi_bid"], dtype=np.float64).copy()
        pi_a = np.asarray(model_plan["pi_ask"], dtype=np.float64).copy()
        active_b_cnt = int(np.sum(pi_b >= 0.06))
        if active_b_cnt <= 2:
            grid_mode_idx = 0
        elif active_b_cnt <= 4:
            grid_mode_idx = 1
        elif active_b_cnt <= 7:
            grid_mode_idx = 2
        else:
            grid_mode_idx = 3
        capital_exp = float(np.clip(max(0.35, raw_exp), 0.15, 0.90))
    else:
        pi_b = None
        pi_a = None

    mode_cfg = GRID_MODES.get(grid_mode_idx, GRID_MODES[1])
    n_b_active = int(mode_cfg["b_slots"])
    n_a_active = int(mode_cfg["a_slots"])

    # 2. Avellaneda-Stoikov Inventory Skew & Toxicity Brake:
    # При росте инвентаря > 25% снижаем целевую покупку; при >= 62% или шторме >= 0.72 — стоп покупкам!
    inv_brake = 1.0
    if pos_ratio >= 0.62 or tox >= 0.72:
        inv_brake = 0.0
    elif pos_ratio > 0.25:
        inv_brake = max(0.15, 1.0 - 1.85 * (pos_ratio - 0.25))
    if tox > 0.45:
        inv_brake *= max(0.20, 1.0 - 1.40 * (tox - 0.45))

    eff_exp = float(np.clip(capital_exp * inv_brake, 0.0, 0.92))
    target_bid_pool = max(0.0, tot_eq * eff_exp - pos_usd * 0.85) if inv_brake > 0.0 else 0.0
    target_bid_pool = min(target_bid_pool, cash_usd * 0.996)

    while n_b_active > 1 and (target_bid_pool / n_b_active) < min_order_usd:
        n_b_active = max(1, n_b_active - 1)

    # 3. Распределение долларовых аллокаций Bid без пыли (< $25 -> 0)
    bid_usd_alloc = np.zeros(12, dtype=np.float32)
    if target_bid_pool >= min_order_usd and n_b_active > 0:
        if pi_b is not None and np.sum(pi_b[:n_b_active]) > 1e-4:
            w_raw = pi_b[:n_b_active].copy()
            # Ограничиваем доминирование слота #00 максимум 35%, чтобы работала вся лестница #00..#04!
            w_raw = np.minimum(w_raw, 0.35 * np.sum(w_raw) if n_b_active >= 3 else w_raw)
            weights = (w_raw / np.sum(w_raw)).astype(np.float32)
        elif n_b_active == 1:
            weights = np.array([1.0], dtype=np.float32)
        elif n_b_active == 2:
            weights = np.array([0.55, 0.45], dtype=np.float32)
        elif n_b_active == 3:
            weights = np.array([0.36, 0.34, 0.30], dtype=np.float32)
        elif n_b_active == 4:
            weights = np.array([0.28, 0.26, 0.24, 0.22], dtype=np.float32)
        elif n_b_active <= 6:
            n_b_active = 6
            weights = np.array([0.22, 0.18, 0.18, 0.16, 0.14, 0.12], dtype=np.float32)
        else:
            weights = np.zeros(n_b_active, dtype=np.float32) + (1.0 / n_b_active)
        bid_usd_alloc[:n_b_active] = target_bid_pool * weights

    # 4. Сжатая рабочая геометрия цен Bid с асимметричным сдвигом от инвентаря и строгой монотонностью:
    # Рабочие отступы эшелонов: от 0..10 bps на фронте до ~115 bps на глубоком слоте (вместо мертвых -350 bps)
    squeezed_rel_offsets = [0.0000, 0.0012, 0.0026, 0.0042, 0.0062, 0.0085, 0.0110, 0.0140, 0.0175, 0.0215, 0.0260, 0.0310]
    inv_skew_rel = max(0.0, pos_ratio - 0.20) * 0.0080 + max(0.0, tox - 0.35) * 0.0060

    bid_px = np.zeros(12, dtype=np.float32)
    bid_qty = np.zeros(12, dtype=np.float32)

    wide_sp = (a1 - b1) >= 3.0 * tick_sz
    # Penny-jump разрешен только при низкой позиции (<25%), низком риске и широком спреде >= 3 тиков
    if wide_sp and spread_tight > 0.55 and pos_ratio < 0.25 and tox < 0.35:
        front_b = min(a1 - 2.0 * tick_sz, b1 + tick_sz)
    else:
        front_b = b1 * (1.0 - inv_skew_rel)
    front_b = round(math.floor((front_b + 1e-9) / tick_sz) * tick_sz, 8)

    prev_bp = front_b + tick_sz
    for s in range(12):
        if bid_usd_alloc[s] >= min_order_usd:
            if model_plan is not None and "t_bid_px" in model_plan:
                # Блендинг обученной цены чемпиона с компактным рабочим коридором:
                m_bp = float(model_plan["t_bid_px"][s])
                compact_bp = b1 * (1.0 - inv_skew_rel - squeezed_rel_offsets[s])
                raw_p = 0.35 * max(m_bp, b1 * 0.982) + 0.65 * compact_bp if s > 0 else min(front_b, max(m_bp, front_b - 4.0 * tick_sz))
            else:
                raw_p = b1 * (1.0 - inv_skew_rel - squeezed_rel_offsets[s]) if s > 0 else front_b

            p_s = round(math.floor((min(raw_p, prev_bp - tick_sz, a1 - tick_sz) + 1e-9) / tick_sz) * tick_sz, 8)
            p_s = max(tick_sz, p_s)
            q_s = math.floor((bid_usd_alloc[s] / (p_s * (1.0 + maker_fee))) * 10000.0) / 10000.0
            if q_s >= MIN_PRL_AMOUNT:
                bid_px[s] = p_s
                bid_qty[s] = q_s
                prev_bp = p_s

    # 5. Расчет Ask-слотов с ЖЁСТКИМ BREAK-EVEN FLOOR GUARD (0 продаж в минус!)
    ask_px = np.zeros(12, dtype=np.float32)
    ask_qty = np.zeros(12, dtype=np.float32)
    core_floor_px = compute_breakeven_floor_px(avg_entry, tick_sz=tick_sz, maker_fee=maker_fee) if avg_entry > 0 else 0.0
    lifo_floor_px = (
        compute_breakeven_floor_px(low_tranche_px, tick_sz=tick_sz, maker_fee=maker_fee)
        if (low_tranche_qty >= MIN_PRL_AMOUNT and 0.0 < low_tranche_px < avg_entry)
        else core_floor_px
    )

    if pos_qty >= MIN_PRL_AMOUNT:
        pos_val_now = pos_qty * max(b1, avg_entry)
        max_a_by_size = max(1, min(6, int(pos_val_now // 30.0) + 1))
        n_a_real = min(n_a_active, max_a_by_size)

        if pi_a is not None and np.sum(pi_a[:n_a_real]) > 1e-4:
            wa = pi_a[:n_a_real].copy()
            wa = np.minimum(wa, 0.45 * np.sum(wa) if n_a_real >= 3 else wa)
            shares = (wa / np.sum(wa)).astype(np.float32)
        else:
            shares = np.ones(n_a_real, dtype=np.float32) / float(n_a_real)

        ask_rel_steps = [0.0000, 0.0014, 0.0030, 0.0050, 0.0075, 0.0105, 0.0140, 0.0180, 0.0225, 0.0275, 0.0330, 0.0400]
        mkt_front_a = max(b1 + tick_sz, a1)
        prev_ap = 0.0
        rem_q = pos_qty

        for s in range(n_a_real):
            # Слот #0 может разгружать только объем дешевого LIFO-транша по lifo_floor_px; всё остальное строго >= core_floor_px!
            if s == 0 and low_tranche_qty >= MIN_PRL_AMOUNT and lifo_floor_px < core_floor_px:
                slot_floor = lifo_floor_px
                q_raw = min(float(pos_qty * shares[s]), float(low_tranche_qty))
            else:
                slot_floor = core_floor_px
                q_raw = float(pos_qty * shares[s])

            q_s = math.floor(min(q_raw, rem_q) * 10000.0) / 10000.0
            if s == n_a_real - 1 and rem_q >= MIN_PRL_AMOUNT:
                q_s = math.floor(rem_q * 10000.0) / 10000.0

            if q_s >= MIN_PRL_AMOUNT and rem_q >= MIN_PRL_AMOUNT:
                base_a = max(mkt_front_a * (1.0 + ask_rel_steps[s]), slot_floor * (1.0 + 0.5 * ask_rel_steps[s]))
                if s > 0 and model_plan is not None and "t_ask_px" in model_plan:
                    m_ap = float(model_plan["t_ask_px"][s])
                    base_a = max(slot_floor, 0.35 * min(m_ap, mkt_front_a * 1.018) + 0.65 * base_a)
                p_a = round(math.ceil((max(base_a, slot_floor, b1 + tick_sz, prev_ap + tick_sz) - 1e-9) / tick_sz) * tick_sz, 8)
                ask_px[s] = p_a
                ask_qty[s] = q_s
                rem_q = max(0.0, rem_q - q_s)
                prev_ap = p_a

        if rem_q >= MIN_PRL_AMOUNT:
            # Остаток всегда добавляем в самый верхний активный слот (>= core_floor_px), чтобы не нарушать защиту безубытка
            last_valid = max([idx for idx in range(12) if ask_qty[idx] > 0] or [0])
            if ask_qty[last_valid] > 0:
                ask_qty[last_valid] = math.floor((ask_qty[last_valid] + rem_q) * 10000.0) / 10000.0
                ask_px[last_valid] = max(ask_px[last_valid], core_floor_px)
            else:
                p_a0 = round(math.ceil((max(mkt_front_a, core_floor_px, b1 + tick_sz) - 1e-9) / tick_sz) * tick_sz, 8)
                ask_px[0] = p_a0
                ask_qty[0] = math.floor(rem_q * 10000.0) / 10000.0

    return {
        "capital_exp": eff_exp,
        "raw_capital_exp": capital_exp,
        "inv_brake": inv_brake,
        "pos_ratio": pos_ratio,
        "core_floor_px": core_floor_px,
        "lifo_floor_px": lifo_floor_px,
        "grid_mode_name": mode_cfg["name"],
        "n_b_active": int(np.sum(bid_qty >= MIN_PRL_AMOUNT)),
        "n_a_active": int(np.sum(ask_qty >= MIN_PRL_AMOUNT)),
        "total_bid_usd": float(np.sum(bid_px * bid_qty)),
        "total_ask_prl": float(np.sum(ask_qty)),
        "bid_px": bid_px,
        "bid_qty": bid_qty,
        "ask_px": ask_px,
        "ask_qty": ask_qty,
    }
