#!/usr/bin/env python3
"""
_tools/eval_l5_champion_budgets.py
=============================================================================
Параллельный тест текущего Чемпиона Урока 5 (lesson_5_champ_latest.pt, 321D, 10 действий)
в честной среде VectorizedWhaleL2Env (с частичными наливами Partial Fills и пробоем L1..L10)
одновременно на 6 бюджетах: $95.5, $1,000, $3,000, $5,000, $10,000, $20,000.

Оценивает два временных горизонта:
1. 50 минут (3,000 секунд = стандартный интервал 1 итерации обучения);
2. 4 часа (14,400 секунд = 4 часовых окна на разных участках рынка).
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
from _tools.vectorized_whale_l2_env import VectorizedWhaleL2Env
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_5_whale_sizing_trainer import BatchedWhalePolicy8K

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
L5_CHAMP_PATH = BASE_DIR / "models" / "prl" / "lesson_5_whale" / "lesson_5_champ_latest.pt"


def eval_interval(policy, backbone, sub, lob_t, flow_m, targets_np, v7_sub, win_list, budgets, dev):
    B = len(budgets)
    tick_sz = float(sub["tick_size"])
    vec_env = VectorizedWhaleL2Env(
        bids=torch.from_numpy(sub["bid_p"]).float(),
        asks=torch.from_numpy(sub["ask_p"]).float(),
        bids_q=torch.from_numpy(sub["bid_q"]).float(),
        asks_q=torch.from_numpy(sub["ask_q"]).float(),
        buy_vol=torch.from_numpy(sub["buy_vol"]).float(),
        sell_vol=torch.from_numpy(sub["sell_vol"]).float(),
        lob_tensors=torch.from_numpy(lob_t).float(),
        flow_market=torch.from_numpy(flow_m).float(),
        targets=torch.from_numpy(targets_np).float(),
        num_envs=B,
        tick_size=tick_sz,
        device=dev,
    )
    bud_t = torch.tensor(budgets, dtype=torch.float32, device=dev)
    v7_t = torch.from_numpy(v7_sub).to(device=dev, dtype=policy.dtype)

    with torch.no_grad():
        lob_repr_all = backbone.encode_lob_batch(vec_env.lob_tensors)

    total_pnl_usd = torch.zeros(B, dtype=torch.float32, device=dev)
    max_dd_usd = torch.zeros(B, dtype=torch.float32, device=dev)
    cycles = torch.zeros(B, dtype=torch.long, device=dev)
    prof_cycles = torch.zeros(B, dtype=torch.long, device=dev)
    vec_env.n_trades.zero_()
    vec_env.partial_fills.zero_()
    vec_env.iceberg_unloads.zero_()

    with torch.no_grad():
        for w_start, w_len in win_list:
            vec_env.budget.copy_(bud_t)
            vec_env.cash.copy_(bud_t)
            vec_env.pos_qty.zero_()
            vec_env.avg_entry.zero_()
            vec_env.entry_t.zero_()
            vec_env.order_side.fill_(-1)
            vec_env.order_act.zero_()
            vec_env.order_price.zero_()
            vec_env.order_qty.zero_()
            vec_env.q_ahead.zero_()
            vec_env.active_after.zero_()

            h_states = backbone.init_hidden(batch_size=B, device=dev)
            prev_pos = vec_env.pos_qty.clone()
            prev_cash = vec_env.cash.clone()
            entry_cash = vec_env.cash.clone()
            peak_eq = bud_t.clone()

            for s in range(w_len - 1):
                cur_t = int(w_start + s)
                _, flow, _, liq_3d = vec_env.get_observations(cur_t_scalar=cur_t)
                masks = vec_env.get_action_masks(cur_t_scalar=cur_t)

                lob_r = lob_repr_all[cur_t:cur_t + 1].expand(B, -1)
                z, h_states = backbone.backbone.forward_step_fast(lob_r, flow, h_states)
                ps = torch.sigmoid(backbone.head_sweep_12s(z))
                pb = torch.sigmoid(backbone.head_bleed_60s(z))
                pc = torch.sigmoid(backbone.head_crash_300s(z))
                pe = torch.sigmoid(backbone.head_exit_30s(z))
                v7_exp = v7_t[cur_t].unsqueeze(0).expand(B, -1)
                z_aug = torch.cat([z, ps, pb, pc, pe, v7_exp, liq_3d], dim=-1).unsqueeze(1).to(policy.dtype)

                h1 = torch.relu(torch.bmm(z_aug, policy.w1) + policy.b1)
                logits = (torch.bmm(h1, policy.w2) + policy.b2).squeeze(1).float().masked_fill(~masks, -1e9)
                actions = logits.argmax(dim=-1)

                vec_env.step(actions, cur_t_scalar=cur_t)
                cur_pos = vec_env.pos_qty
                cur_cash = vec_env.cash
                b1 = torch.clamp(vec_env.bids[cur_t, 0], min=tick_sz)

                in_p = cur_pos > 1e-6
                eq = cur_cash + cur_pos * b1
                peak_eq = torch.maximum(peak_eq, eq)
                max_dd_usd = torch.maximum(max_dd_usd, peak_eq - eq)

                just_opened = (prev_pos <= 1e-6) & in_p
                entry_cash = torch.where(just_opened, prev_cash, entry_cash)

                just_closed = (prev_pos > 1e-6) & (~in_p)
                cycles += just_closed.long()
                prof_cycles += (just_closed & ((cur_cash - entry_cash) > 0.0)).long()

                prev_pos.copy_(cur_pos)
                prev_cash.copy_(cur_cash)

            end_t = int(w_start + w_len - 1)
            rem = vec_env.pos_qty > 1e-6
            if rem.any():
                rem_q = vec_env.pos_qty[rem].clone()
                proceeds = torch.zeros_like(rem_q)
                for lvl in range(10):
                    lp = torch.clamp(vec_env.bids[end_t, lvl], min=tick_sz)
                    lq = torch.clamp(vec_env.bids_q[end_t, lvl], min=1.0)
                    tq = torch.minimum(rem_q, lq if lvl < 9 else rem_q)
                    proceeds += tq * lp
                    rem_q = torch.clamp(rem_q - tq, min=0.0)
                vec_env.cash[rem] += proceeds * (1.0 - vec_env.taker_fee)
                vec_env.pos_qty[rem] = 0.0

            total_pnl_usd += (vec_env.cash - bud_t)

    out = []
    for i, b in enumerate(budgets):
        pu = float(total_pnl_usd[i].item())
        du = float(max_dd_usd[i].item())
        c_i = int(cycles[i].item())
        p_i = int(prof_cycles[i].item())
        out.append({
            "budget": b,
            "pnl_usd": pu,
            "pnl_pct": 100.0 * pu / b,
            "dd_usd": du,
            "dd_pct": 100.0 * du / b,
            "cycles": c_i,
            "prof_cycles": p_i,
            "part_fills": int(vec_env.partial_fills[i].item()),
            "iceberg": int(vec_env.iceberg_unloads[i].item()),
        })
    return out


def main():
    raw = np.load(CACHE_PATH)
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.0001

    # 4 окна по 3,600 секунд = 14,400 секунд (4 часа реального рынка)
    win_starts = [n_total - 160000, n_total - 115000, n_total - 70000, n_total - 25000]
    idx = np.concatenate([np.arange(ws, ws + 3600) for ws in win_starts])
    keys = ["bid_p", "ask_p", "bid_q", "ask_q", "buy_vol", "sell_vol", "y_sweep_12s", "y_bleed_60s", "y_crash_300s", "y_exit_30s"]
    sub = {k: raw[k][idx] for k in keys}
    sub["tick_size"] = tick_sz
    v7_sub = raw["v7_signals_58d"][idx].astype(np.float32)
    del raw

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budgets = [95.5, 1000.0, 3000.0, 5000.0, 10000.0, 20000.0]
    B = len(budgets)

    c_data = torch.load(L5_CHAMP_PATH, map_location=dev, weights_only=False)
    it_num = c_data.get("iteration", "?")
    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(dev)
    backbone.load_state_dict(c_data["backbone_state"], strict=False)
    backbone.eval()

    policy = BatchedWhalePolicy8K(backbone=backbone, pop_size=B, in_dim=321, h_dim=256, n_actions=10, device=dev)
    policy.w1.copy_(c_data["w1"].to(dev, policy.dtype).unsqueeze(0).expand(B, -1, -1))
    policy.b1.copy_(c_data["b1"].to(dev, policy.dtype).unsqueeze(0).expand(B, -1, -1))
    policy.w2.copy_(c_data["w2"].to(dev, policy.dtype).unsqueeze(0).expand(B, -1, -1))
    policy.b2.copy_(c_data["b2"].to(dev, policy.dtype).unsqueeze(0).expand(B, -1, -1))

    print(f"🐋 ТЕСТ ЧЕМПИОНА УРОКА 5 (Итерация {it_num}/1000) В ЧЕСТНОЙ СРЕДЕ С ЧАСТИЧНЫМИ НАЛИВАМИ (Partial Fills + L1..L10 Impact)")
    print("=" * 105)

    w_50m = [(0, 1500), (3600, 1500)]
    print("\n⏱️ 1. ИНТЕРВАЛ 50 МИНУТ (3,000 секунд = 2 окна по 25 мин, масштаб одной итерации обучения):")
    for m in eval_interval(policy, backbone, sub, lob_t, flow_m, targets_np, v7_sub, w_50m, budgets, dev):
        wr = 100.0 * m["prof_cycles"] / max(1, m["cycles"])
        print(
            f"  Бюджет ${m['budget']:8,.1f} | Net PnL: {m['pnl_pct']:+6.2f}% (${m['pnl_usd']:+8.2f}) | "
            f"MaxDD: {m['dd_pct']:5.2f}% (${m['dd_usd']:6.2f}) | Циклов: {m['prof_cycles']:2d}/{m['cycles']:2d} ({wr:4.0f}%) | "
            f"PartFills: {m['part_fills']:2d} | Iceberg: {m['iceberg']:2d}"
        )

    w_4h = [(0, 3600), (3600, 3600), (7200, 3600), (10800, 3600)]
    print("\n⏱️ 2. ИНТЕРВАЛ 4 ЧАСА (14,400 секунд = 4 полных часовых окна на разных фазах рынка):")
    for m in eval_interval(policy, backbone, sub, lob_t, flow_m, targets_np, v7_sub, w_4h, budgets, dev):
        wr = 100.0 * m["prof_cycles"] / max(1, m["cycles"])
        print(
            f"  Бюджет ${m['budget']:8,.1f} | Net PnL: {m['pnl_pct']:+6.2f}% (${m['pnl_usd']:+8.2f}) | "
            f"MaxDD: {m['dd_pct']:5.2f}% (${m['dd_usd']:6.2f}) | Циклов: {m['prof_cycles']:2d}/{m['cycles']:2d} ({wr:4.0f}%) | "
            f"PartFills: {m['part_fills']:2d} | Iceberg: {m['iceberg']:2d}"
        )


if __name__ == "__main__":
    main()
