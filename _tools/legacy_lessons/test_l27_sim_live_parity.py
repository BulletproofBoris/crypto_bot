#!/usr/bin/env python3
"""
_tools/test_l27_sim_live_parity.py
=============================================================================
Автоматический тест 100% математического паритета между GPU-ядром
`fused_l27_multimodal_tick` (`K=4` MoE + Router + Fanned Echelons + 26D Reg)
и боевым мостом `l27_live_bridge.py` + `live_rl_trader.py`.
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
from _tools.lesson_27_multimodal_moe_trainer import MultimodalMoEPolicyL27, RELEASE_CHAMP_PATH, L26_EXPERT_POLES
from _tools.fused_l27_multimodal_tick import (
    MKT_IDX_L27,
    AG_IDX_L27,
    K_MODES_L27,
    MOE_OUT_DIM_L27,
    ROUTER_H_DIM_L27,
    TOKEN_BUCKET_CAP_L27,
    normalize_market_z370_per_block,
    fused_l27_multimodal_tick,
)
from _tools.l27_live_bridge import (
    forward_l27_multimodal_live,
    compute_l27_multimodal_plan,
)


class DummyTraderL27:
    def __init__(self, st_f32: dict):
        self.tick_size = 0.01
        self.initial_budget_usd = 800.0
        self.bot_cash_usd = 520.0
        self.pos_qty = 200.0
        self.avg_entry = 1.46
        self.l27_api_tokens = TOKEN_BUCKET_CAP_L27
        self.l7_mem_h = torch.zeros((1, 16), dtype=torch.float32)
        for k, v in st_f32.items():
            setattr(self, f"l7_{k}", v.float().cpu())
        self.bid_slots = [{"slot": i, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "qa": 0.0, "qa0": 0.0} for i in range(12)]
        self.ask_slots = [{"slot": i, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "qa": 0.0, "qa0": 0.0} for i in range(12)]


def main() -> None:
    ck = torch.load(RELEASE_CHAMP_PATH, map_location="cpu", weights_only=False)
    bb = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).eval()
    pol = MultimodalMoEPolicyL27(bb, n_groups=4, n_budgets=1, device=torch.device("cpu"))
    pol.dtype = torch.float32
    st_cpu, gl = pol.load_from_checkpoint(ck, expert_pole_paths=L26_EXPERT_POLES)
    folded = pol.fold_gate_into_state(st_cpu, gl)

    bp_10 = np.array([1.43, 1.39, 1.35, 1.31, 1.28, 1.25, 1.20, 1.18, 1.15, 1.10], dtype=np.float32)
    bq_10 = np.array([150.0, 11314.6, 4200.0, 900.0, 500.0, 300.0, 1200.0, 800.0, 600.0, 2000.0], dtype=np.float32)
    ap_10 = np.array([1.46, 1.49, 1.52, 1.55, 1.58, 1.60, 1.65, 1.70, 1.75, 1.80], dtype=np.float32)
    aq_10 = np.array([220.0, 5400.0, 1800.0, 700.0, 400.0, 900.0, 1100.0, 500.0, 300.0, 1000.0], dtype=np.float32)

    torch.manual_seed(42)
    z_mkt_raw = torch.randn((1, 1, 370), dtype=torch.float32) * 0.5
    z_mkt_norm, _ = normalize_market_z370_per_block(z_mkt_raw)

    trader = DummyTraderL27(folded)
    b1, a1 = 1.43, 1.46
    pos_val = trader.pos_qty * b1
    tot_eq = trader.bot_cash_usd + pos_val
    pos_rat = pos_val / tot_eq
    free_rat = trader.bot_cash_usd / tot_eq
    eff_sc = min(1.0, max(0.15, pos_rat / 0.25))
    unr_pct = ((b1 - trader.avg_entry) / trader.avg_entry) * 25.0 * eff_sc
    acct_8d = [1.0, pos_rat, free_rat, float(np.clip(unr_pct, -2.5, 2.5)), 0.0, 0.0, 0.0, 0.0]
    queue_8d = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    multi_6d = [
        math.log1p(800.0 / (1.43 * 150.0)) / 3.0,
        math.log1p(520.0 / (1.43 * 150.0)) / 3.0,
        (200.0 * 1.46 / (1.46 * 220.0)) / 2.0,
        (200.0 / 500.0) / 5.0,
        0.5,
        float(np.clip(((1.43 - 1.46) / 1.46) * 30.0, -2.5, 2.5)),
    ]
    z_ag_110 = torch.tensor([multi_6d + [0.0] * 48 + acct_8d + queue_8d + [0.0] * 16 + [-1.0] * 24], dtype=torch.float32)

    z_full_480 = torch.zeros((1, 480), dtype=torch.float32)
    z_full_480[:, MKT_IDX_L27] = z_mkt_raw.squeeze(0)
    z_full_480[:, AG_IDX_L27] = z_ag_110
    forward_l27_multimodal_live(trader, z_full_480, acct_8d, now_q_ts=100.0)
    # На шаге симулятора до исполнения api_tokens пополняется на +0.35 с капом 10.0 -> остаётся 10.0:
    plan_live = compute_l27_multimodal_plan(trader, b1, a1, now_ts=100.0, api_tokens_override=10.0)

    w1_mkt = folded["w1"].float()[MKT_IDX_L27].unsqueeze(0)
    w1_ag = folded["w1"].float()[AG_IDX_L27].unsqueeze(0)
    pre_mkt = (z_mkt_norm.view(1, 1, 370) @ w1_mkt + folded["b1"].float().view(1, 1, 384)).view(1, 1, 384)
    ech = torch.tensor([0.0, 0.0, 0.12, 0.12, 0.26, 0.26, 0.44, 0.44, 0.68, 0.68, 1.0, 1.0])
    out_sim = fused_l27_multimodal_tick.__wrapped__(
        torch.tensor(100, dtype=torch.int16), torch.tensor(101, dtype=torch.int16), pre_mkt,
        torch.tensor([b1]), torch.tensor([a1]), torch.tensor([150.0]), torch.tensor([220.0]),
        torch.tensor([500.0]), torch.tensor([500.0]),
        torch.from_numpy(bp_10).unsqueeze(0), torch.from_numpy(ap_10).unsqueeze(0),
        torch.from_numpy(bq_10).unsqueeze(0), torch.from_numpy(aq_10).unsqueeze(0),
        torch.tensor([b1]), torch.tensor([a1]), torch.tensor([0.0]), torch.tensor([0.0]),
        w1_ag, folded["w_res1"].float().unsqueeze(0), folded["b_res1"].float().view(1, 1, 128), folded["w_res2"].float().unsqueeze(0),
        folded["w2_moe"].float().unsqueeze(0), folded["b2_moe"].float().view(1, 1, MOE_OUT_DIM_L27),
        folded["w_gate1"].float().unsqueeze(0), folded["b_gate1"].float().view(1, 1, ROUTER_H_DIM_L27),
        folded["w_gate2"].float().unsqueeze(0), folded["b_gate2"].float().view(1, 1, K_MODES_L27),
        folded["w_rec"].float().unsqueeze(0), folded["b_rec"].float().view(1, 1, 32),
        folded["w_reg1"].float().unsqueeze(0), folded["b_reg1_b"].float().view(1, 1, 48), folded["b_reg1_a"].float().view(1, 1, 48),
        folded["w_reg2"].float().unsqueeze(0), folded["b_reg2_b"].float().view(1, 1, 2), folded["b_reg2_a"].float().view(1, 1, 2),
        ech, torch.tensor([800.0]), torch.zeros((1, 16)), torch.tensor([520.0]), torch.tensor([200.0]),
        torch.tensor([200.0 * 1.46 * 1.001]), torch.tensor([1.46]), torch.tensor([1.46]), torch.tensor([100.0]),
        torch.tensor([1.46]), torch.tensor([100.0]), torch.tensor([100], dtype=torch.int16),
        torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12), dtype=torch.int16),
        torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12)), torch.zeros((1, 12), dtype=torch.int16),
        torch.tensor([10.0]), torch.tensor([805.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([805.0]),
        torch.zeros((1, K_MODES_L27)), torch.tensor([0.0]),
        torch.tensor([0], dtype=torch.int16), torch.tensor([0], dtype=torch.int16), torch.tensor([0], dtype=torch.int16),
        0.01, 0.001, 0.002, True,
    )
    sim_b_px, sim_b_qty = out_sim[10].squeeze(0).numpy(), out_sim[11].squeeze(0).numpy()
    sim_a_px, sim_a_qty = out_sim[15].squeeze(0).numpy(), out_sim[16].squeeze(0).numpy()
    sim_g_mode = out_sim[29].squeeze(0).numpy()

    dg_mode = float(np.max(np.abs(sim_g_mode - plan_live["g_mode"])))
    act_b_m = sim_b_qty > 0
    act_a_m = sim_a_qty > 0
    dp_b = float(np.max(np.abs(sim_b_px[act_b_m] - plan_live["t_bid_px"][act_b_m]))) if np.any(act_b_m) else 0.0
    dp_a = float(np.max(np.abs(sim_a_px[act_a_m] - plan_live["t_ask_px"][act_a_m]))) if np.any(act_a_m) else 0.0
    assert dg_mode < 1e-5, f"Mode router mismatch: {dg_mode} ({sim_g_mode} vs {plan_live['g_mode']})"
    assert dp_b < 1e-5 and dp_a < 1e-5, f"Price mismatch: dp_b={dp_b}, dp_a={dp_a}"
    print(
        f"L27_PARITY_OK | max|Δg_mode|={dg_mode:.7f} | g_mode={np.round(sim_g_mode, 3).tolist()} | "
        f"max|Δpx_b|={dp_b:.6f} | max|Δpx_a|={dp_a:.6f} | max_pi_bid={float(np.max(plan_live['pi_bid'])):.3f} | "
        f"active_bids={int(np.sum(act_b_m))} | active_asks={int(np.sum(act_a_m))}"
    )


if __name__ == "__main__":
    main()
