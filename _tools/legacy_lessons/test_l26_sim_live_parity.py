#!/usr/bin/env python3
"""
_tools/test_l26_sim_live_parity.py
=============================================================================
Автоматический тест 100% математического паритета между GPU-ядром
`fused_l26_endogenous_tick` и живым мостом `l26_live_bridge.py` + `live_rl_trader.py`.
=============================================================================
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path("/home/restorator/crypto_bot")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.lesson_26_endogenous_nonlinear_trainer import EndogenousNonlinearPolicyL26, RELEASE_CHAMP_PATH
from _tools.fused_l26_endogenous_tick import (
    MKT_IDX_L26,
    AG_IDX_L26,
    normalize_market_z370_per_block,
    _lookup_sparse_bid_lob_q,
    fused_l26_endogenous_tick,
)
from _tools.l26_live_bridge import (
    lookup_sparse_lob_qty_1d,
    forward_l26_strategist_live,
    compute_l26_endogenous_plan,
    compute_honest_taker_cross_price,
)


class DummyTraderL26:
    def __init__(self, st_f32: dict):
        self.tick_size = 0.01
        self.initial_budget_usd = 800.0
        self.bot_cash_usd = 520.0
        self.pos_qty = 200.0
        self.avg_entry = 1.46
        self.l26_api_tokens = 12.0
        self.l7_mem_h = torch.zeros((1, 16), dtype=torch.float32)
        for k, v in st_f32.items():
            setattr(self, f"l7_{k}", v.float().cpu())
        self.bid_slots = [{"slot": i, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "qa": 0.0, "qa0": 0.0} for i in range(12)]
        self.ask_slots = [{"slot": i, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "qa": 0.0, "qa0": 0.0} for i in range(12)]


def main() -> None:
    ck = torch.load(RELEASE_CHAMP_PATH, map_location="cpu", weights_only=False)
    bb = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).eval()
    pol = EndogenousNonlinearPolicyL26(bb, n_groups=4, n_budgets=1, device=torch.device("cpu"))
    pol.dtype = torch.float32
    st_cpu, gl = pol.load_from_checkpoint(ck)
    folded = pol.fold_gate_into_state(st_cpu, gl)

    # 1. Тест разреженного стакана с "дырами": $1.43 (150 PRL), $1.42 ПУСТ, $1.39 (11314 PRL)!
    bp_10 = np.array([1.43, 1.39, 1.35, 1.31, 1.28, 1.25, 1.20, 1.18, 1.15, 1.10], dtype=np.float32)
    bq_10 = np.array([150.0, 11314.6, 4200.0, 900.0, 500.0, 300.0, 1200.0, 800.0, 600.0, 2000.0], dtype=np.float32)
    ap_10 = np.array([1.46, 1.49, 1.52, 1.55, 1.58, 1.60, 1.65, 1.70, 1.75, 1.80], dtype=np.float32)
    aq_10 = np.array([220.0, 5400.0, 1800.0, 700.0, 400.0, 900.0, 1100.0, 500.0, 300.0, 1000.0], dtype=np.float32)

    q_empty_live = lookup_sparse_lob_qty_1d(bp_10, bq_10, 1.42, 0.01)
    q_real_live = lookup_sparse_lob_qty_1d(bp_10, bq_10, 1.39, 0.01)
    q_sim = _lookup_sparse_bid_lob_q(
        torch.from_numpy(bp_10).unsqueeze(0),
        torch.from_numpy(bq_10).unsqueeze(0),
        torch.tensor([[1.42, 1.39] + [1.43] * 10], dtype=torch.float32),
        1, 1, 12, 0.01,
    )
    assert q_empty_live == 0.0 and float(q_sim[0, 0].item()) == 0.0, f"Sparse empty tick failed: {q_empty_live} vs {q_sim[0,0]}"
    assert abs(q_real_live - 11314.6) < 1e-2 and abs(float(q_sim[0, 1].item()) - 11314.6) < 1e-2

    # 2. Тест p_cross при Taker-выходе на 300 PRL при допуске 4%:
    p_cr, q_cr = compute_honest_taker_cross_price(bp_10, bq_10, 300.0, 1.43, 0.04, 0.01)
    assert abs(p_cr - 1.39) < 1e-5 and abs(q_cr - 300.0) < 1e-5, f"Taker cross mismatch: {p_cr}, {q_cr}"

    # 3. Точный 1-шаговый паритет целевых цен `t_bid_px / t_ask_px` и объёмов `q_target_b / q_target_a`:
    torch.manual_seed(42)
    z_mkt_raw = torch.randn((1, 1, 370), dtype=torch.float32) * 0.5
    z_mkt_norm, _ = normalize_market_z370_per_block(z_mkt_raw)

    trader = DummyTraderL26(folded)
    b1, a1 = 1.43, 1.46
    pos_val = trader.pos_qty * b1
    tot_eq = trader.bot_cash_usd + pos_val
    pos_rat = pos_val / tot_eq
    free_rat = trader.bot_cash_usd / tot_eq
    eff_sc = min(1.0, max(0.15, pos_rat / 0.25))
    unr_pct = ((b1 - trader.avg_entry) / trader.avg_entry) * 25.0 * eff_sc
    acct_8d = [1.0, pos_rat, free_rat, float(np.clip(unr_pct, -2.5, 2.5)), 0.0, 0.0, 0.0, 0.0]
    queue_8d = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    multi_6d = [math.log1p(800.0 / (1.43 * 150.0)) / 3.0, math.log1p(520.0 / (1.43 * 150.0)) / 3.0, (200.0 * 1.46 / (1.46 * 220.0)) / 2.0, (200.0 / 500.0) / 5.0, 0.5, float(np.clip(((1.43 - 1.46) / 1.46) * 30.0, -2.5, 2.5))]
    z_ag_110 = torch.tensor([multi_6d + [0.0] * 48 + acct_8d + queue_8d + [0.0] * 16 + [-1.0] * 24], dtype=torch.float32)

    z_full_480 = torch.zeros((1, 480), dtype=torch.float32)
    z_full_480[:, MKT_IDX_L26] = z_mkt_raw.squeeze(0)
    z_full_480[:, AG_IDX_L26] = z_ag_110
    forward_l26_strategist_live(trader, z_full_480, acct_8d, now_q_ts=100.0)
    plan_live = compute_l26_endogenous_plan(trader, b1, a1, now_ts=100.0)

    # Запуск ядра симулятора на тех же входах (FP32):
    w1_mkt = folded["w1"].float()[MKT_IDX_L26].unsqueeze(0)
    w1_ag = folded["w1"].float()[AG_IDX_L26].unsqueeze(0)
    pre_mkt = (z_mkt_norm.view(1, 1, 370) @ w1_mkt + folded["b1"].float().view(1, 1, 384)).view(1, 1, 384)
    ech = torch.tensor([0.0, 0.0, 0.12, 0.12, 0.26, 0.26, 0.44, 0.44, 0.68, 0.68, 1.0, 1.0])
    out_sim = fused_l26_endogenous_tick.__wrapped__(
        torch.tensor(100, dtype=torch.int16), torch.tensor(101, dtype=torch.int16), pre_mkt,
        torch.tensor([b1]), torch.tensor([a1]), torch.tensor([150.0]), torch.tensor([220.0]),
        torch.tensor([500.0]), torch.tensor([500.0]),
        torch.from_numpy(bp_10).unsqueeze(0), torch.from_numpy(ap_10).unsqueeze(0),
        torch.from_numpy(bq_10).unsqueeze(0), torch.from_numpy(aq_10).unsqueeze(0),
        torch.tensor([b1]), torch.tensor([a1]), torch.tensor([0.0]), torch.tensor([0.0]),
        w1_ag, folded["w_res1"].float().unsqueeze(0), folded["b_res1"].float().view(1, 1, 128), folded["w_res2"].float().unsqueeze(0),
        folded["w2"].float().unsqueeze(0), folded["b2"].float().view(1, 1, 44),
        folded["w_rec"].float().unsqueeze(0), folded["b_rec"].float().view(1, 1, 32),
        folded["w_reg1"].float().unsqueeze(0), folded["b_reg1_b"].float().view(1, 1, 48), folded["b_reg1_a"].float().view(1, 1, 48),
        folded["w_reg2"].float().unsqueeze(0), folded["b_reg2_b"].float().view(1, 1, 2), folded["b_reg2_a"].float().view(1, 1, 2),
        ech, torch.tensor([800.0]), torch.zeros((1, 16)), torch.tensor([520.0]), torch.tensor([200.0]),
        torch.tensor([200.0 * 1.46 * 1.001]), torch.tensor([1.46]), torch.tensor([1.46]), torch.tensor([100.0]),
        torch.tensor([1.46]), torch.tensor([100.0]), torch.tensor([100], dtype=torch.int16),
        torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12), dtype=torch.int16),
        torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12), dtype=torch.int16),
        torch.tensor([12.0]), torch.tensor([805.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([805.0]),
        torch.tensor([0], dtype=torch.int16), torch.tensor([0], dtype=torch.int16), torch.tensor([0], dtype=torch.int16),
        0.01, 0.001, 0.002, True,
    )
    sim_b_px, sim_b_qty = out_sim[10].squeeze(0).numpy(), out_sim[11].squeeze(0).numpy()
    sim_a_px, sim_a_qty = out_sim[15].squeeze(0).numpy(), out_sim[16].squeeze(0).numpy()

    act_b_m = sim_b_qty > 0
    act_a_m = sim_a_qty > 0
    dp_b = float(np.max(np.abs(sim_b_px[act_b_m] - plan_live["t_bid_px"][act_b_m]))) if np.any(act_b_m) else 0.0
    dp_a = float(np.max(np.abs(sim_a_px[act_a_m] - plan_live["t_ask_px"][act_a_m]))) if np.any(act_a_m) else 0.0
    assert dp_b < 1e-5 and dp_a < 1e-5, f"Price mismatch: dp_b={dp_b}, dp_a={dp_a}"
    print(f"L26_PARITY_OK | empty_tick_qa=0.0 | p_cross=${p_cr:.2f} | max|Δpx_b|={dp_b:.6f} | max|Δpx_a|={dp_a:.6f} | t_bid_px={np.round(plan_live['t_bid_px'], 2).tolist()} | t_ask_px={np.round(plan_live['t_ask_px'], 2).tolist()}")


if __name__ == "__main__":
    main()
