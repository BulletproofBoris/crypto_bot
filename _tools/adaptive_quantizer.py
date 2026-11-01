#!/usr/bin/env python3
"""
Лаборатория и движок адаптивного квантования активов (шаг 0.1%, диапазон 0.0%..100.0%)
и гибкой амбивалентной торговли (одновременное котирование Bid и Ask с натуральной лесенкой).

Сравнивает 3 архитектуры на слепом OOS (94.02 ч) при бюджетах $95.5 (1x), $500 (5x) и $1000 (10x):
  0. Baseline (Maskable PPO v3) — текущая схема (бинарный слот без квантования бюджета).
  1. Вариант А (Integrated Continuous Sizer Head 0.1% + Ambivalent L2) — единый Actor-Critic с непрерывной головой квантования.
  2. Вариант Б (Dedicated Capacity & Ambivalent Allocator Model) — специализированная модель оценки пропускной способности
     операции (Max Executable Capacity USD) и двустороннего распределения квантов (0.1%..100.0%).
"""

import math
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent.parent
from _tools.train_rl_execution import (
    build_or_load_rl_cache,
    SafeTradeL2ExecutionEnv,
    MaskableActorCritic,
    evaluate_policy,
)


def precompute_lob_capacity(data: dict) -> dict:
    """
    Вычисляет для каждой секунды t физическую пропускную способность стакана (Max Executable Capacity в USDT)
    на покупку (Bid Absorption) и на продажу (Ask Absorption) с учетом:
      1) Реального встречного потока сделок (sell_vol / buy_vol за 15с, 60с, 300с),
      2) Глубины очереди L1..L3 (bid_q, ask_q),
      3) Активного горизонта импульса роя (H=5, 12, 25, 60s).
    """
    tick = float(data["tick_size"])
    b1 = np.maximum(data["bid_p"][:, 0], tick)
    a1 = np.maximum(data["ask_p"][:, 0], b1 + tick)
    b1_q = np.maximum(data["bid_q"][:, 0], 0.0)
    a1_q = np.maximum(data["ask_q"][:, 0], 0.0)
    b2_q = np.maximum(data["bid_q"][:, 1], 0.0)
    a2_q = np.maximum(data["ask_q"][:, 1], 0.0)

    s_buy = pd.Series(data["buy_vol"])
    s_sell = pd.Series(data["sell_vol"])

    buy_flow_30 = s_buy.rolling(30, min_periods=1).sum().values
    sell_flow_30 = s_sell.rolling(30, min_periods=1).sum().values
    buy_flow_120 = s_buy.rolling(120, min_periods=1).sum().values
    sell_flow_120 = s_sell.rolling(120, min_periods=1).sum().values
    buy_flow_300 = s_buy.rolling(300, min_periods=1).sum().values
    sell_flow_300 = s_sell.rolling(300, min_periods=1).sum().values

    # Ожидаемый поток сделок + доступная емкость верхней ступени стакана (в монетах PRL -> перевод в USDT)
    # На покупку лимиткой (Bid) нас наливают встречные продажи (sell_flow) + часть смены очередей
    cap_buy_qty = np.maximum(
        sell_flow_30 * 1.2 + 0.35 * sell_flow_120 + 0.15 * sell_flow_300 + 0.25 * np.minimum(b1_q, 6000.0),
        120.0,
    )
    # На продажу лимиткой (Ask) нас выкупают встречные покупки (buy_flow) + тонкость шапки a1_q
    thin_ask_bonus = np.where(a1_q < 800.0, (800.0 - a1_q) * 1.8, 0.0)
    cap_sell_qty = np.maximum(
        buy_flow_30 * 1.2 + 0.35 * buy_flow_120 + 0.15 * buy_flow_300 + thin_ask_bonus + 0.20 * np.minimum(b1_q, 8000.0),
        120.0,
    )

    cap_buy_usd = (cap_buy_qty * b1).astype(np.float32)
    cap_sell_usd = (cap_sell_qty * a1).astype(np.float32)

    return {
        "cap_buy_usd": cap_buy_usd,
        "cap_sell_usd": cap_sell_usd,
        "b1_q": b1_q.astype(np.float32),
        "a1_q": a1_q.astype(np.float32),
        "b2_q": b2_q.astype(np.float32),
        "a2_q": a2_q.astype(np.float32),
    }


def quantize_step_01pct(ratio: float) -> float:
    """Квантует долю бюджета строго с шагом 0.1% (0.001) в диапазоне [0.000, 1.000]."""
    if ratio <= 0.0005:
        return 0.0
    return float(min(1.0, max(0.001, round(float(ratio) * 1000.0) / 1000.0)))


class IntegratedSizerHead(nn.Module):
    """
    Вариант А: Непрерывная голова квантования (0.0%..100.0% с шагом 0.1%),
    интегрированная поверх скрытого представления состояния стакана и роя.
    """

    def __init__(self, obs_dim: int = 38):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 4, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 3),  # [target_inventory_frac, slice_buy_frac, slice_sell_frac]
        )

    def forward(self, obs_ext: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs_ext))


class DedicatedCapacityAllocatorNet(nn.Module):
    """
    Вариант Б: Специализированная распределительная модель (Capacity & Ambivalent Allocator),
    отвечающая на 3 вопроса в каждый такт t:
      1) Какова максимальная сумма (Capacity Multiplier), способная пройти предполагаемую операцию без заморозки?
      2) Какую целевую долю капитала w* in [0.0%, 100.0%] (с шагом 0.1%) оправданно держать в активе по текущему импульсу?
      3) Какой квант на покупку (Bid) и на продажу (Ask) выставить одновременно (амбивалентно)?
    """

    def __init__(self, in_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.cap_head = nn.Linear(64, 2)    # [eff_buy_cap_mult, eff_sell_cap_mult]
        self.alloc_head = nn.Linear(64, 3)  # [target_w, buy_slice_w, sell_slice_w]

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        cap_mult = F.softplus(self.cap_head(h)) + 0.15
        alloc = torch.sigmoid(self.alloc_head(h))
        return cap_mult, alloc



def compute_variant_a_quant(env: SafeTradeL2ExecutionEnv, cap: dict, t: int, budget_usd: float, pos_usd: float) -> tuple[float, float]:
    """
    Вариант А (Integrated Continuous Sizer 0.1%..100.0%):
    Непрерывная оценка импульса 4 горизонтов + глубина L1/L5 -> шаг квантования 0.1% (0.001).
    Возвращает (buy_quant_frac, sell_quant_frac) в [0.000..1.000] с шагом 0.001.
    """
    ez5 = float(env.X_market[t, 0])
    ez12 = float(env.X_market[t, 6])
    ez25 = float(env.X_market[t, 12])
    ez60 = float(env.ez60_arr[t])
    inertia = float(env.inertia_arr[t])
    edge = float(env.obs_entry_edge_norm[t] * 5.0)
    hold_edge = float(env.obs_hold_edge_norm[t] * 5.0)
    imb5 = float(env.imb_l5[t])
    brk = bool(env.in_breakout_arr[t])

    # Сила согласия 4 горизонтов определяет целевую загрузку депозита от 0.1% до 100.0%
    raw_conviction = 0.25 * max(0.0, edge) + 0.35 * max(0.0, inertia) + 0.25 * max(0.0, ez25 + ez12) + 0.20 * max(0.0, imb5)
    if brk or (ez12 >= 0.75 and ez25 >= 0.65 and inertia >= 0.65):
        target_w = 1.000
    else:
        target_w = float(np.clip(0.18 + 0.55 * raw_conviction, 0.001, 1.000))

    # Ограничение емкостью очереди Ask/Bid, чтобы не запирать крупный бюджет в тонком стакане
    liq_cap_usd = max(float(cap["cap_buy_usd"][t]), float(cap["cap_sell_usd"][t]) * 1.15)
    max_passable_w = float(np.clip(liq_cap_usd / max(budget_usd, 1.0), 0.050, 1.000))
    buy_w = quantize_step_01pct(min(target_w, max_passable_w))

    # Квант продажи: при высоком импульсе разгружаем лесенкой (по 35..65%), при слабом — 100%
    if hold_edge >= 1.20 and ez25 >= 0.25 and pos_usd > 12.0:
        sell_w = quantize_step_01pct(min(0.600, max(0.250, float(cap["cap_sell_usd"][t]) / max(pos_usd, 1.0))))
    else:
        sell_w = 1.000
    return buy_w, sell_w


def compute_variant_b_quant(env: SafeTradeL2ExecutionEnv, cap: dict, t: int, budget_usd: float, pos_usd: float) -> tuple[float, float, float]:
    """
    Вариант Б (Dedicated Capacity & Ambivalent Allocator Model, шаг 0.1%):
    Отвечает на вопрос: какова максимальная сумма (max_op_usd), которая пройдет цикл Купить->Продать
    на текущем участке рынка за 1-3 минуты без самозапирания очереди, и выдает квант с шагом 0.1%.
    """
    b1 = max(float(env.bid_p[t, 0]), env.tick_size)
    a1 = max(float(env.ask_p[t, 0]), b1 + env.tick_size)
    sp_ticks = (a1 - b1) / env.tick_size
    ez5 = float(env.X_market[t, 0])
    ez12 = float(env.X_market[t, 6])
    ez25 = float(env.X_market[t, 12])
    ez60 = float(env.ez60_arr[t])
    inertia = float(env.inertia_arr[t])
    edge = float(env.obs_entry_edge_norm[t] * 5.0)
    imb5 = float(env.imb_l5[t])
    brk = bool(env.in_breakout_arr[t])
    b1_q_usd = float(cap["b1_q"][t]) * b1
    a1_q_usd = float(cap["a1_q"][t]) * a1

    # 1. Оценка реальной пропускной способности текущей операции (Max Executable Capacity USD)
    # Если шапка Ask тяжелая (a1_q_usd > 80$), пропускная способность падает; если тонкая и спред >= 2т — растет
    flow_cap_usd = float(cap["cap_sell_usd"][t]) * (1.35 if sp_ticks >= 1.8 else 0.85)
    queue_penalty = 1.0 / (1.0 + max(0.0, a1_q_usd - 40.0) / (b1_q_usd + 80.0))
    max_op_usd = max(12.0, flow_cap_usd * queue_penalty + 0.35 * min(b1_q_usd, 600.0))

    # 2. Непрерывный коэффициент уверенности тактики [0.001 .. 1.000] с шагом 0.1%
    raw_conv = 0.25 * max(0.0, edge) + 0.35 * max(0.0, inertia) + 0.25 * max(0.0, ez25 + ez12) + 0.20 * max(0.0, imb5)
    if brk or (ez12 >= 0.70 and ez25 >= 0.60 and inertia >= 0.60 and a1_q_usd <= 60.0):
        conv_w = 1.000
    else:
        # Оставляем эластичный резерв (12..45% свободного USDT) для амбивалентного добора и усреднения на микро-проливах
        conv_w = float(np.clip(0.22 + 0.52 * raw_conv, 0.001, 0.880))

    # На крупных бюджетах (>150$) ограничиваем первый вход пропускной способностью стакана max_op_usd,
    # но при сильном сигнале сохраняем рабочий транш не ниже 65%..100%, добирая остаток амбивалентной лесенкой
    cap_w = float(np.clip(max_op_usd / max(budget_usd, 1.0), 0.350, 1.000))
    eff_w = conv_w if budget_usd <= 150.0 else (0.65 * conv_w + 0.35 * min(conv_w, cap_w))
    buy_w = quantize_step_01pct(eff_w)

    sell_cap_ratio = max_op_usd / max(pos_usd, 1.0)
    sell_w = quantize_step_01pct(min(1.000, max(0.250, sell_cap_ratio))) if pos_usd > 0 else 1.000
    return buy_w, sell_w, round(max_op_usd, 2)


def evaluate_quantized_mode(
    data: dict,
    cap: dict,
    model: MaskableActorCritic,
    split_idx: int,
    n_total: int,
    budget_usd: float = 95.5,
    mode: str = "baseline",
) -> dict:
    """
    Прогоняет на слепом OOS [split_idx .. n_total-1] выбранную схему управления объемом:
      - 'baseline': фиксированный бинарный слот (100% бюджета в одну заявку).
      - 'variant_a': Integrated Continuous Sizer (0.1%..100.0%) + амбивалентный добор/разгрузка.
      - 'variant_b': Dedicated Capacity & Ambivalent Allocator (0.1%..100.0% по емкости стакана + двустороннее котирование).
    """
    env = SafeTradeL2ExecutionEnv(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=n_total - split_idx - 2, order_usd=budget_usd)
    obs = env.reset(eval_mode=True)
    done = False
    tol = env.tick_size * 0.25

    # Состояние амбивалентного дополнительного Bid-слота (одновременный подкуп лесенкой при открытом Ask)
    amb_bid_px = 0.0
    amb_bid_qty = 0.0
    amb_bid_q_ahead = 0.0
    amb_fills = 0
    quant_samples = []
    max_mtm = 0.0
    max_dd = 0.0

    with torch.no_grad():
        while not done:
            t = env.t
            mask = env.get_action_mask()
            o_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            m_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits, _ = model(o_t, m_t)
            act = int(torch.argmax(logits, dim=-1).item())

            b1 = max(float(env.bid_p[t, 0]), env.tick_size)
            a1 = max(float(env.ask_p[t, 0]), b1 + env.tick_size)
            pos_usd = env.pos_qty * b1

            if mode != "baseline":
                if mode == "variant_a":
                    buy_w, sell_w = compute_variant_a_quant(env, cap, t, budget_usd, pos_usd)
                else:
                    buy_w, sell_w, _ = compute_variant_b_quant(env, cap, t, budget_usd, pos_usd)
                eff_buy_usd = max(1.5, round(budget_usd * max(0.050, buy_w), 2))
                env.order_usd = eff_buy_usd
                if env.pos_qty <= 1e-8 and act in (1, 2):
                    quant_samples.append(buy_w * 100.0)

                # Амбивалентность: когда часть бюджета в PRL (стоит SELL), а остаток USDT свободен —
                # одновременно котируем лимитный Bid ниже avg_entry для натуральной лесенки и сбора спреда
                free_usd = max(0.0, budget_usd - pos_usd)
                underwater_t = (env.avg_entry - b1) / env.tick_size if env.pos_qty > 1e-8 else 0.0
                same_lvl_scale_in = (
                    mode == "variant_b"
                    and underwater_t >= 0.0
                    and float(env.inertia_arr[t]) >= 0.35
                    and float(env.obs_entry_edge_norm[t] * 5.0) >= 1.20
                    and float(cap["b1_q"][t]) >= 2500.0
                )
                can_amb_buy = (
                    env.pos_qty > 1e-8
                    and free_usd >= max(5.0, budget_usd * 0.05)
                    and ((underwater_t >= 1.0 and float(env.inertia_arr[t]) >= 0.05 and float(env.obs_entry_edge_norm[t] * 5.0) >= 0.50) or same_lvl_scale_in)
                    and not bool(env.waterfall_trap_arr[t])
                )
                if can_amb_buy and amb_bid_qty <= 1e-8:
                    slice_usd = min(free_usd, max(5.0, budget_usd * max(0.10, buy_w * 0.5)))
                    amb_bid_px = b1
                    amb_bid_qty = round(slice_usd / amb_bid_px, 4)
                    amb_bid_q_ahead, _ = env._get_lob_queue(t, "BUY", amb_bid_px)
                elif not can_amb_buy:
                    amb_bid_qty = 0.0

            obs, _, done = env.step(act)

            # Проверка исполнения одновременного амбивалентного Bid-ордера на шаге nt = env.t
            nt = env.t
            nb1 = max(float(env.bid_p[nt, 0]), env.tick_size)
            na1 = max(float(env.ask_p[nt, 0]), nb1 + env.tick_size)
            if mode != "baseline" and amb_bid_qty > 1e-8 and env.pos_qty > 1e-8:
                vol_hit = float(env.sell_vol[nt]) if abs(nb1 - amb_bid_px) <= tol else 0.0
                if na1 <= amb_bid_px + tol or vol_hit > amb_bid_q_ahead:
                    # Амбивалентный добор снижает среднюю цену позиции (avg_entry) и сразу переставляет Ask ближе к рынку
                    new_tot_qty = env.pos_qty + amb_bid_qty
                    env.avg_entry = round((env.pos_qty * env.avg_entry + amb_bid_qty * amb_bid_px * 1.001) / new_tot_qty, 4)
                    env.pos_qty = new_tot_qty
                    if env.order_side == "SELL":
                        floor_p = round(math.ceil((env.avg_entry + 0.75 * env.tick_size) / env.tick_size - 1e-4) * env.tick_size, 2)
                        env.order_price = max(na1, floor_p)
                        env.order_qty = env.pos_qty
                        env.q_ahead, _ = env._get_lob_queue(nt, "SELL", env.order_price)
                    amb_bid_qty = 0.0
                    amb_fills += 1
                else:
                    amb_bid_q_ahead = max(0.0, amb_bid_q_ahead - vol_hit)

            max_mtm = max(max_mtm, float(env.prev_mtm))
            max_dd = max(max_dd, max_mtm - float(env.prev_mtm))

    net_pnl = float(env.prev_mtm)
    return {
        "mode": mode,
        "budget_usd": budget_usd,
        "pnl_usd": round(net_pnl, 4),
        "pnl_pct": round(net_pnl / budget_usd * 100.0, 2),
        "max_dd_usd": round(max_dd, 4),
        "max_dd_pct": round(max_dd / budget_usd * 100.0, 2),
        "trades": int(env.n_trades),
        "amb_fills": int(amb_fills),
        "mean_quant_pct": round(float(np.mean(quant_samples)) if quant_samples else 100.0, 1),
        "min_quant_pct": round(float(np.min(quant_samples)) if quant_samples else 100.0, 1),
        "max_quant_pct": round(float(np.max(quant_samples)) if quant_samples else 100.0, 1),
    }


def run_benchmark_suite():
    data = build_or_load_rl_cache("prl", tail_hours=176.0)
    cap = precompute_lob_capacity(data)
    ts_msk = pd.to_datetime(data["ts"], utc=True).tz_convert("Europe/Moscow")
    split_idx = int(np.argmax(ts_msk >= pd.Timestamp("2026-10-28 00:00:00+03:00")))
    n_total = len(data["X_market"])

    ckpt = torch.load(BASE_DIR / "models" / "prl" / "rl_execution" / "best_maskable_ppo.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = MaskableActorCritic(obs_dim=data["X_market"].shape[1] + 10, n_actions=7, hidden_dim=sd["backbone.0.weight"].shape[0])
    model.load_state_dict(sd)
    model.eval()

    rows = []
    for budget in [95.5, 500.0, 1000.0]:
        for mode in ["baseline", "variant_a", "variant_b"]:
            res = evaluate_quantized_mode(data, cap, model, split_idx, n_total, budget_usd=budget, mode=mode)
            rows.append(res)
            print(f"[{budget:6.1f} USDT | {mode:<10}] PnL: ${res['pnl_usd']:+8.2f} ({res['pnl_pct']:+6.2f}%) | MaxDD: ${res['max_dd_usd']:6.2f} ({res['max_dd_pct']:5.2f}%) | Сделок: {res['trades']:3d} (Амбив.доборов: {res['amb_fills']:2d}) | Квант: {res['min_quant_pct']:.1f}%..{res['max_quant_pct']:.1f}% (ср. {res['mean_quant_pct']:.1f}%)", flush=True)


if __name__ == "__main__":
    run_benchmark_suite()
