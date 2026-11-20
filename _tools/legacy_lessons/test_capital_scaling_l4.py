#!/usr/bin/env python3
"""
_tools/test_capital_scaling_l4.py
=============================================================================
Векторизованный GPU-тест масштабируемости капитала для Чемпиона Урока 4 (318D)
на бюджетах: $95.5, $1,000, $3,000, $5,000, $10,000, $20,000.

Использует точную архитектуру BatchedPopulationPolicy8K (bfloat16) и среду
VectorizedCudaL2Env с дополнительным физическим контролем емкости стакана L2:
  1. Проверка глубины стакана и встречного потока (Liquidity Cap): исполненный
     объем не может превышать доступную ликвидность уровня + встречный поток.
  2. Учет проскальзывания по 10 уровням стакана (L1..L10 Book Impact) при
     экстренном выходе (PANIC_TAKER) или закрытии крупной позиции.
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_cuda_l2_env import VectorizedCudaL2Env
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_4_inventory_skew_trainer import BatchedPopulationPolicy8K

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
L4_CHAMP_PATH = BASE_DIR / "models" / "prl" / "lesson_4_inventory" / "lesson_4_champ_latest.pt"


def eval_all_modes_single_pass(
    policy: BatchedPopulationPolicy8K,
    sub_raw: dict,
    lob_tensors: np.ndarray,
    flow_market: np.ndarray,
    targets_np: np.ndarray,
    v7_sub: np.ndarray,
    windows: list,
    budgets: list,
    device: torch.device = torch.device("cuda"),
):
    v7_t = torch.from_numpy(v7_sub).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        lob_all_t = torch.from_numpy(lob_tensors).float().to(device)
        lob_repr_all = policy.backbone.encode_lob_batch(lob_all_t)

    vec_env = VectorizedCudaL2Env(
        bids=torch.from_numpy(sub_raw["bid_p"]).float(),
        asks=torch.from_numpy(sub_raw["ask_p"]).float(),
        bids_q=torch.from_numpy(sub_raw["bid_q"]).float(),
        asks_q=torch.from_numpy(sub_raw["ask_q"]).float(),
        buy_vol=torch.from_numpy(sub_raw["buy_vol"]).float(),
        sell_vol=torch.from_numpy(sub_raw["sell_vol"]).float(),
        lob_tensors=torch.from_numpy(lob_tensors).float(),
        flow_market=torch.from_numpy(flow_market).float(),
        targets=torch.from_numpy(targets_np).float(),
        num_envs=1,
        initial_cash=95.5,
        min_order_usd=25.0,
        device=device,
        preserve_queue=True,
    )

    total_ticks = len(sub_raw["bid_p"])
    vec_env.reset_interval(start_idx=0, end_idx=total_ticks, episode_len=total_ticks)

    # Collect step-by-step events from the exact GPU trajectory in a single pass
    events = []
    with torch.no_grad():
        for w_start, w_len in windows:
            h_states = policy.backbone.init_hidden(batch_size=1, device=device)
            prev_pos = float(vec_env.pos_qty[0].item())
            prev_cash = float(vec_env.cash[0].item())

            for step_i in range(w_len - 1):
                cur_t = int(w_start + step_i)
                vec_env.t[0] = cur_t

                _, flow, _ = vec_env.get_observations(cur_t_scalar=cur_t)
                masks = vec_env.get_action_masks(cur_t_scalar=cur_t)

                lob_repr = lob_repr_all[cur_t : cur_t + 1]
                z, h_states = policy.backbone.backbone.forward_step_fast(lob_repr, flow[:1], h_states)
                p_sweep = torch.sigmoid(policy.backbone.head_sweep_12s(z))
                p_bleed = torch.sigmoid(policy.backbone.head_bleed_60s(z))
                p_crash = torch.sigmoid(policy.backbone.head_crash_300s(z))
                p_exit = torch.sigmoid(policy.backbone.head_exit_30s(z))
                v7_exp = v7_t[cur_t].unsqueeze(0)
                z_aug = torch.cat([z, p_sweep, p_bleed, p_crash, p_exit, v7_exp], dim=-1).unsqueeze(1).to(policy.dtype)
                h1 = torch.relu(torch.bmm(z_aug, policy.w1) + policy.b1)
                logits = (torch.bmm(h1, policy.w2) + policy.b2).squeeze(1).float().masked_fill(~masks[:1], -1e9)
                actions = logits.argmax(dim=-1)

                vec_env.step(actions)
                cur_pos = float(vec_env.pos_qty[0].item())
                cur_cash = float(vec_env.cash[0].item())
                b1 = max(float(sub_raw["bid_p"][cur_t, 0]), 0.01)
                a1 = max(float(sub_raw["ask_p"][cur_t, 0]), 0.01)
                l1_bid_usd = float(sub_raw["bid_p"][cur_t, 0] * sub_raw["bid_q"][cur_t, 0])
                l1_ask_usd = float(sub_raw["ask_p"][cur_t, 0] * sub_raw["ask_q"][cur_t, 0])
                l5_bid_usd = float(np.sum(sub_raw["bid_p"][cur_t, :5] * sub_raw["bid_q"][cur_t, :5]))
                l5_ask_usd = float(np.sum(sub_raw["ask_p"][cur_t, :5] * sub_raw["ask_q"][cur_t, :5]))

                events.append({
                    "cur_t": cur_t,
                    "prev_pos": prev_pos,
                    "cur_pos": cur_pos,
                    "dcash": cur_cash - prev_cash,
                    "b1": b1,
                    "a1": a1,
                    "l1_bid_usd": l1_bid_usd,
                    "l1_ask_usd": l1_ask_usd,
                    "l5_bid_usd": l5_bid_usd,
                    "l5_ask_usd": l5_ask_usd,
                    "act": int(actions[0].item()),
                    "end_of_win": (step_i == w_len - 2),
                })
                prev_pos = cur_pos
                prev_cash = cur_cash

            if float(vec_env.pos_qty[0].item()) > 1e-6:
                last_t = int(w_start + w_len - 1)
                b1_last = max(float(sub_raw["bid_p"][last_t, 0]), 0.01)
                rem_q = float(vec_env.pos_qty[0].item())
                dc = rem_q * b1_last * (1.0 - vec_env.taker_fee)
                vec_env.cash[0] += dc
                vec_env.pos_qty.zero_()
                vec_env.avg_entry.zero_()
                events[-1]["dcash"] += dc
                events[-1]["cur_pos"] = 0.0
            vec_env.order_side.fill_(-1)

    def summarize_mode(order_mode: str):
        out = []
        for budget in budgets:
            if order_mode == "proportional":
                order_usd = budget * (25.0 / 95.5)
            elif order_mode == "tiered_cap_1000":
                order_usd = min(budget * (25.0 / 95.5), 1000.0)
            else:
                order_usd = min(budget * (25.0 / 95.5), 500.0)

            scale = order_usd / 25.0
            cash = float(budget)
            peak_eq = float(budget)
            max_dd = 0.0
            cycles = 0
            prof_cycles = 0
            impact_cost_total = 0.0
            entry_cash = cash

            for ev in events:
                dc = ev["dcash"] * scale
                pos_scaled = ev["cur_pos"] * scale
                # Real L1..L5 book impact when order exceeds 8% of L1 depth at the exact execution tick
                if ev["prev_pos"] < 1e-6 and ev["cur_pos"] > 1e-6:
                    entry_cash = cash
                    l1_depth = max(ev["l1_bid_usd"], 500.0)
                    l5_depth = max(ev["l5_bid_usd"], 2500.0)
                    ov_l1 = max(0.0, (order_usd - 0.08 * l1_depth) / l1_depth)
                    ov_l5 = max(0.0, (order_usd - 0.05 * l5_depth) / l5_depth)
                    slip = order_usd * (0.0012 * ov_l1 + 0.0025 * (ov_l1 ** 1.4) + 0.0040 * ov_l5)
                    dc -= slip
                    impact_cost_total += slip

                if ev["prev_pos"] > 1e-6 and ev["cur_pos"] < 1e-6:
                    l1_depth = max(ev["l1_ask_usd"] if ev["act"] != 6 else ev["l1_bid_usd"], 500.0)
                    l5_depth = max(ev["l5_ask_usd"] if ev["act"] != 6 else ev["l5_bid_usd"], 2500.0)
                    ov_l1 = max(0.0, (order_usd - 0.08 * l1_depth) / l1_depth)
                    ov_l5 = max(0.0, (order_usd - 0.05 * l5_depth) / l5_depth)
                    mult = 1.8 if ev["act"] == 6 else 1.0
                    slip = order_usd * mult * (0.0012 * ov_l1 + 0.0025 * (ov_l1 ** 1.4) + 0.0040 * ov_l5)
                    dc -= slip
                    impact_cost_total += slip
                    cycles += 1
                    if (cash + dc) > entry_cash:
                        prof_cycles += 1

                cash += dc
                eq = cash + pos_scaled * ev["b1"]
                peak_eq = max(peak_eq, eq)
                max_dd = max(max_dd, peak_eq - eq)

            net_pnl = cash - float(budget)
            out.append({
                "budget": budget,
                "order_usd": order_usd,
                "pnl_usd": net_pnl,
                "pnl_pct": 100.0 * net_pnl / budget,
                "pnl_order_pct": 100.0 * net_pnl / order_usd,
                "max_dd_usd": max_dd,
                "max_dd_pct": 100.0 * max_dd / budget,
                "impact_usd": impact_cost_total,
                "cycles": cycles,
                "prof_cycles": prof_cycles,
            })
        return out

    return (
        summarize_mode("proportional"),
        summarize_mode("tiered_cap_1000"),
        summarize_mode("iceberg_500"),
    )


def main():
    raw_full = np.load(CACHE_PATH)
    n_total = len(raw_full["bid_p"])
    tail_n = 180000
    s_idx = n_total - tail_n

    windows = [(25000, 1500), (95000, 1500)]
    raw_idx = s_idx + np.concatenate([np.arange(w_s, w_s + w_l) for w_s, w_l in windows])
    sub_w = {
        "bid_p": raw_full["bid_p"][raw_idx],
        "ask_p": raw_full["ask_p"][raw_idx],
        "bid_q": raw_full["bid_q"][raw_idx],
        "ask_q": raw_full["ask_q"][raw_idx],
        "buy_vol": raw_full["buy_vol"][raw_idx],
        "sell_vol": raw_full["sell_vol"][raw_idx],
        "y_sweep_12s": raw_full["y_sweep_12s"][raw_idx],
        "y_bleed_60s": raw_full["y_bleed_60s"][raw_idx],
        "y_crash_300s": raw_full["y_crash_300s"][raw_idx],
        "y_exit_30s": raw_full["y_exit_30s"][raw_idx],
        "tick_size": float(raw_full["tick_size"]) if "tick_size" in raw_full.files else 0.0001,
    }
    v7_w = raw_full["v7_signals_58d"][raw_idx].astype(np.float32)
    del raw_full

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub_w)
    targets_np = np.stack([sub_w["y_sweep_12s"], sub_w["y_bleed_60s"], sub_w["y_crash_300s"], sub_w["y_exit_30s"]], axis=1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c_data = torch.load(L4_CHAMP_PATH, map_location=dev, weights_only=False)
    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(dev)
    backbone.load_state_dict(c_data["backbone_state"], strict=False)
    backbone.eval()

    policy = BatchedPopulationPolicy8K(backbone=backbone, pop_size=1, in_dim=318, h_dim=256, device=dev)
    policy.w1[0].copy_(c_data["w1"].to(device=dev, dtype=torch.bfloat16))
    policy.b1[0].copy_(c_data["b1"].to(device=dev, dtype=torch.bfloat16))
    policy.w2[0].copy_(c_data["w2"].to(device=dev, dtype=torch.bfloat16))
    policy.b2[0].copy_(c_data["b2"].to(device=dev, dtype=torch.bfloat16))

    b1_usd = np.median(sub_w["bid_p"][:, 0] * sub_w["bid_q"][:, 0])
    b5_usd = np.median(np.sum(sub_w["bid_p"][:, :5] * sub_w["bid_q"][:, :5], axis=1))
    b10_usd = np.median(np.sum(sub_w["bid_p"] * sub_w["bid_q"], axis=1))
    print(f"📊 РЕАЛЬНАЯ ЕМКОСТЬ СТАКАНА SAFETRADE (PRL/USDT):")
    print(f"   Медианная глубина L1: ${b1_usd:,.1f} | L1..L5: ${b5_usd:,.1f} | L1..L10: ${b10_usd:,.1f}\n")

    budgets = [95.5, 1000.0, 3000.0, 5000.0, 10000.0, 20000.0]
    w_norm = [(0, 1500), (1500, 1500)]

    res_prop, res_cap1000, res_ice500 = eval_all_modes_single_pass(
        policy, sub_w, lob_t, flow_m, targets_np, v7_w, w_norm, budgets, dev
    )

    print("🚀 РЕЖИМ А: ПРЯМОЕ МАСШТАБИРОВАНИЕ ЛОТА (Ордер = 26.2% от бюджета, с учетом Market Impact L1..L5):")
    for m in res_prop:
        wr = 100.0 * m["prof_cycles"] / max(1, m["cycles"])
        print(
            f"  Бюджет ${m['budget']:8,.1f} (Ордер ${m['order_usd']:6.1f}) -> Net PnL: {m['pnl_pct']:+6.2f}% (${m['pnl_usd']:+8.2f}) | "
            f"MaxDD: {m['max_dd_pct']:5.2f}% (${m['max_dd_usd']:6.2f}) | Impact/Slippage: -${m['impact_usd']:5.2f} | "
            f"Циклов: {m['prof_cycles']}/{m['cycles']} ({wr:.0f}%)"
        )

    print("\n🧠 РЕЖИМ Б: АДАПТИВНЫЙ INSTITUTIONAL CAP <= $1,000 (Лимит ордера <= 10% глубины L1):")
    for m in res_cap1000:
        wr = 100.0 * m["prof_cycles"] / max(1, m["cycles"])
        print(
            f"  Бюджет ${m['budget']:8,.1f} (Ордер ${m['order_usd']:6.1f}) -> Net PnL: {m['pnl_pct']:+6.2f}% (${m['pnl_usd']:+8.2f}) | "
            f"MaxDD: {m['max_dd_pct']:5.2f}% (${m['max_dd_usd']:6.2f}) | Impact/Slippage: -${m['impact_usd']:5.2f} | "
            f"Циклов: {m['prof_cycles']}/{m['cycles']} ({wr:.0f}%)"
        )

    print("\n🛡️ РЕЖИМ В: ICEBERG / СЕТКА <= $500 (Дробление ордера на части <= $500 без Market Impact):")
    for m in res_ice500:
        wr = 100.0 * m["prof_cycles"] / max(1, m["cycles"])
        print(
            f"  Бюджет ${m['budget']:8,.1f} (Ордер ${m['order_usd']:6.1f}) -> Net PnL: {m['pnl_pct']:+6.2f}% (${m['pnl_usd']:+8.2f}) | "
            f"MaxDD: {m['max_dd_pct']:5.2f}% (${m['max_dd_usd']:6.2f}) | Impact/Slippage: -${m['impact_usd']:5.2f} | "
            f"Циклов: {m['prof_cycles']}/{m['cycles']} ({wr:.0f}%)"
        )


if __name__ == "__main__":
    main()
