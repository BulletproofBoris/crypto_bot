#!/usr/bin/env python3
"""
_tools/eval_l28_blind_coronation.py
=============================================================================
Фаза 4 — Коронационный Слепой OOS-Турнир Урока 28:
Впервые распечатывает физически изолированный хвост `Final SubCent Blind OOS`
(`[3,076,379 : 3,096,179]`, `19,800с = 5.50ч`, 50 уровней `$0.0001`,
`exec_tick=0.0001`, `true_latency=True`) + историческую корзину макро-крахов
и сквизов (`exec_tick=0.01`, `true_latency=True`) на бюджетах `[$835, $1,500, $5,000]`.

Честное сравнение в идентичной субцентовой физике исполнения (с хотфиксом):
  1) `L28 Cross-Wave Soup` (`508D` без `90s`, `HierarchicalRegimeTree 1s..120s`),
  2) `L28 Best 4-Fold Val` (`508D` без `90s`),
  3) `L28 Latest Champion` (`508D` без `90s`),
  4) `Live L27 Cross-Wave Champion (with hotfix)` (`480D` на нативных `370D` признаках),
  5) `L27 Soup Top-3 (with hotfix)` (`480D` на нативных `370D` признаках),
  6) `L26 Endogenous Champion (with hotfix)` (`480D` на нативных `370D` признаках).
=============================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path("/home/restorator/crypto_bot")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_free_multi_order_env import VectorizedFreeMultiOrderEnv
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_18_multistack_lifo_survival_trainer import FusedBF16BackboneRunner
from _tools.lesson_23_pump_capture_penny_l1_trainer import _build_roll_sell_30s
from _tools.fused_l25_regulator_tick import (
    MKT_IDX_L25 as MKT_IDX_480,
    AG_IDX_L25 as AG_IDX_480,
    precompute_window_market_z370,
)
from _tools.fused_l26_endogenous_tick import normalize_market_z370_per_block
from _tools.build_l28_hierarchical_cache import read_l28_cache_global_slice
from _tools.fused_l28_hierarchical_subcent_tick import (
    MKT_DIM_L28,
    MKT_IDX_L28,
    AG_IDX_L28,
    K_MODES_L28,
    EXPERT_OUT_DIM_L28,
    MOE_OUT_DIM_L28,
    ROUTER_IN_DIM_L28,
    ROUTER_H_DIM_L28,
    REG_IN_DIM_L28,
    TOKEN_BUCKET_CAP_L28,
    build_ext_mkt_138_tensor,
    precompute_window_market_z398,
    normalize_market_z398_per_block,
    fused_l28_hierarchical_subcent_tick,
)

BASE_LOB_CACHE_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
L28_CACHE_PATH = BASE_DIR / "data/cache/l28_hierarchical_subcent_cache.npz"
L28_DIR = BASE_DIR / "models/prl/lesson_28_hierarchical_subcent"
L27_DIR = BASE_DIR / "models/prl/lesson_27_multimodal_moe"
L26_DIR = BASE_DIR / "models/prl/lesson_26_endogenous_nonlinear"

BLIND_OOS_DURATION_SEC = 19_800
SUBCENT_GLOBAL_START = 3_016_277
MSK_TZ = timezone(timedelta(hours=3))


def _load_480d_model_as_moe_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """
    Загружает `480D` чекпоинт `L27` или `L26` в единый канонический MoE-словарь:
    для `L26` (`1-mode 45D` + `22D` регулятор) все 4 эксперта MoE получают идентичные
    веса `w2/b2`, а матрица `w_reg1` дополняется 4 нулевыми строками под `g_mode`,
    что математически даёт побитово тождественный выход при любом `sum(g_mode) = 1`.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st: dict[str, torch.Tensor] = {
        "w1": ck["w1"].float(),
        "b1": ck["b1"].float().view(1, 384),
        "w_res1": ck["w_res1"].float(),
        "b_res1": ck["b_res1"].float().view(1, 128),
        "w_res2": ck["w_res2"].float(),
        "w_rec": ck["w_rec"].float(),
        "b_rec": ck["b_rec"].float().view(1, 32),
        "b_reg1_b": ck["b_reg1_b"].float().view(1, 48),
        "b_reg1_a": ck["b_reg1_a"].float().view(1, 48),
        "w_reg2": ck["w_reg2"].float(),
        "b_reg2_b": ck["b_reg2_b"].float().view(1, 2),
        "b_reg2_a": ck["b_reg2_a"].float().view(1, 2),
    }
    if "w2_moe" in ck and ck["w2_moe"].shape[1] == MOE_OUT_DIM_L28:
        st["w2_moe"] = ck["w2_moe"].float()
        st["b2_moe"] = ck["b2_moe"].float().view(1, MOE_OUT_DIM_L28)
        st["w_gate1"] = ck["w_gate1"].float()
        st["b_gate1"] = ck["b_gate1"].float().view(1, ROUTER_H_DIM_L28)
        st["w_gate2"] = ck["w_gate2"].float()
        st["b_gate2"] = ck["b_gate2"].float().view(1, K_MODES_L28)
    else:
        w2_single = ck["w2"].float()
        b2_single = ck["b2"].float().view(1, -1)
        if w2_single.shape[1] == 44:
            pad_w = torch.zeros((256, 1), dtype=torch.float32)
            pad_b = torch.full((1, 1), 2.5, dtype=torch.float32)
            w2_single = torch.cat([w2_single, pad_w], dim=1)
            b2_single = torch.cat([b2_single, pad_b], dim=1)
        st["w2_moe"] = w2_single.repeat(1, K_MODES_L28)
        st["b2_moe"] = b2_single.repeat(1, K_MODES_L28)
        st["w_gate1"] = torch.zeros((ROUTER_IN_DIM_L28, ROUTER_H_DIM_L28), dtype=torch.float32)
        st["b_gate1"] = torch.zeros((1, ROUTER_H_DIM_L28), dtype=torch.float32)
        st["w_gate2"] = torch.zeros((ROUTER_H_DIM_L28, K_MODES_L28), dtype=torch.float32)
        st["b_gate2"] = torch.zeros((1, K_MODES_L28), dtype=torch.float32)

    w_reg1 = ck["w_reg1"].float()
    if w_reg1.shape[0] == 22:
        w_reg1 = torch.cat([w_reg1, torch.zeros((4, 48), dtype=torch.float32)], dim=0)
    st["w_reg1"] = w_reg1
    return st


@torch.no_grad()
def _eval_universal_group(
    val_env: VectorizedFreeMultiOrderEnv,
    bf16_runner: FusedBF16BackboneRunner,
    lob_fu_all: torch.Tensor,
    ext138_all: torch.Tensor | None,
    v7_74_all: torch.Tensor | None,
    bo36_all: torch.Tensor | None,
    roll_sell_30s_all: torch.Tensor,
    subcent_books: dict[str, torch.Tensor] | None,
    subcent_local_start: int,
    cand_states: list[dict[str, torch.Tensor]],
    is_508d: bool,
    windows: list[tuple[str, int, float, float]],
    win_len: int,
    exec_tick: float,
    feat_tick: float,
    budget_tiers: torch.Tensor,
    device: torch.device,
    true_latency: bool = True,
) -> dict[str, torch.Tensor]:
    dt = lob_fu_all.dtype
    C = len(cand_states)
    W = len(windows)
    B_bud = budget_tiers.numel()
    E = C * B_bud
    N = W * E
    S = 12
    echelon_k01 = torch.tensor(
        [0.0, 0.0, 0.12, 0.12, 0.26, 0.26, 0.44, 0.44, 0.68, 0.68, 1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )

    mkt_idx_t = torch.tensor(MKT_IDX_L28 if is_508d else MKT_IDX_480, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_L28 if is_508d else AG_IDX_480, dtype=torch.long, device=device)

    w1_c = torch.stack([st["w1"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    w1_mkt_c = w1_c.index_select(1, mkt_idx_t).contiguous()
    w1_ag_c = w1_c.index_select(1, ag_idx_t).contiguous()
    b1_c = torch.stack([st["b1"].to(device=device, dtype=dt).view(1, 384) for st in cand_states], dim=0).contiguous()
    wr1_c = torch.stack([st["w_res1"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    br1_c = torch.stack([st["b_res1"].to(device=device, dtype=dt).view(1, 128) for st in cand_states], dim=0).contiguous()
    wr2_c = torch.stack([st["w_res2"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    w2m_c = torch.stack([st["w2_moe"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    b2m_c = torch.stack([st["b2_moe"].to(device=device, dtype=dt).view(1, MOE_OUT_DIM_L28) for st in cand_states], dim=0).contiguous()
    wg1_c = torch.stack([st["w_gate1"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    bg1_c = torch.stack([st["b_gate1"].to(device=device, dtype=dt).view(1, ROUTER_H_DIM_L28) for st in cand_states], dim=0).contiguous()
    wg2_c = torch.stack([st["w_gate2"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    bg2_c = torch.stack([st["b_gate2"].to(device=device, dtype=dt).view(1, K_MODES_L28) for st in cand_states], dim=0).contiguous()
    wrec_c = torch.stack([st["w_rec"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    brec_c = torch.stack([st["b_rec"].to(device=device, dtype=dt).view(1, 32) for st in cand_states], dim=0).contiguous()
    wreg1_c = torch.stack([st["w_reg1"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    breg1b_c = torch.stack([st["b_reg1_b"].to(device=device, dtype=dt).view(1, 48) for st in cand_states], dim=0).contiguous()
    breg1a_c = torch.stack([st["b_reg1_a"].to(device=device, dtype=dt).view(1, 48) for st in cand_states], dim=0).contiguous()
    wreg2_c = torch.stack([st["w_reg2"].to(device=device, dtype=dt) for st in cand_states], dim=0).contiguous()
    breg2b_c = torch.stack([st["b_reg2_b"].to(device=device, dtype=dt).view(1, 2) for st in cand_states], dim=0).contiguous()
    breg2a_c = torch.stack([st["b_reg2_a"].to(device=device, dtype=dt).view(1, 2) for st in cand_states], dim=0).contiguous()

    w_starts = torch.tensor([int(ws) for _, ws, _, _ in windows], dtype=torch.long, device=device)
    t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
    t_steps_cl = torch.clamp(t_steps, max=val_env.max_steps - 1)

    flow_w_all = val_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
    lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()

    # Вычисляем `adv_risk` из `MultiTaskLOBNet` идентично для всех моделей:
    assert ext138_all is not None
    ext_w_for_adv = ext138_all[t_steps_cl[: win_len - 1]].contiguous()
    z_mkt_398_raw, adv_risk_w_all = precompute_window_market_z398(flow_w_all, lob_fu_w_all, ext_w_for_adv, bf16_runner)

    if is_508d:
        z_mkt_norm, _ = normalize_market_z398_per_block(z_mkt_398_raw)
        mkt_dim_cur = MKT_DIM_L28
    else:
        del z_mkt_398_raw
        assert v7_74_all is not None and bo36_all is not None
        v7_w_all = v7_74_all[t_steps_cl[: win_len - 1]].contiguous()
        bo_w_all = bo36_all[t_steps_cl[: win_len - 1]].contiguous()
        z_mkt_370_raw = precompute_window_market_z370(flow_w_all, lob_fu_w_all, v7_w_all, bo_w_all, bf16_runner)
        z_mkt_norm, _ = normalize_market_z370_per_block(z_mkt_370_raw)
        mkt_dim_cur = 370

    z_mkt_c_in = z_mkt_norm.view((win_len - 1) * W, mkt_dim_cur).unsqueeze(0).expand(C, -1, -1)
    pre_mkt_all = (torch.bmm(z_mkt_c_in, w1_mkt_c) + b1_c).view(C, win_len - 1, W, 384).permute(1, 0, 2, 3).contiguous()

    b1q_cent_w_all = val_env.bids_q[t_steps_cl][:, :, 0].contiguous()
    a1q_cent_w_all = val_env.asks_q[t_steps_cl][:, :, 0].contiguous()

    if exec_tick < feat_tick * 0.5 and subcent_books is not None:
        sub_idx = torch.clamp(t_steps_cl - int(subcent_local_start), min=0, max=subcent_books["bid_p50"].shape[0] - 1)
        bL_w_all = subcent_books["bid_p50"][sub_idx].contiguous()
        aL_w_all = subcent_books["ask_p50"][sub_idx].contiguous()
        bqL_w_all = subcent_books["bid_q50"][sub_idx].contiguous()
        aqL_w_all = subcent_books["ask_q50"][sub_idx].contiguous()
    else:
        bL_w_all = val_env.bids[t_steps_cl].contiguous()
        aL_w_all = val_env.asks[t_steps_cl].contiguous()
        bqL_w_all = val_env.bids_q[t_steps_cl].contiguous()
        aqL_w_all = val_env.asks_q[t_steps_cl].contiguous()

    b1_w_all = bL_w_all[:, :, 0].contiguous()
    a1_w_all = aL_w_all[:, :, 0].contiguous()
    rb_w_all = val_env.roll_buy_30s[t_steps_cl].contiguous()
    rs_w_all = roll_sell_30s_all[t_steps_cl].contiguous()
    sv_w_all = val_env.sell_vol[t_steps_cl].contiguous()
    bv_w_all = val_env.buy_vol[t_steps_cl].contiguous()

    env_budgets_e = budget_tiers.unsqueeze(0).expand(C, B_bud).reshape(E)
    budget = env_budgets_e.unsqueeze(0).expand(W, E).reshape(N).contiguous()

    mem_h = torch.zeros((N, 16), dtype=dt, device=device)
    cash = budget.clone()
    pos_qty = torch.zeros(N, dtype=torch.float32, device=device)
    pos_cost_usd = torch.zeros(N, dtype=torch.float32, device=device)
    avg_entry = torch.zeros(N, dtype=torch.float32, device=device)
    min_buy_px = torch.zeros(N, dtype=torch.float32, device=device)
    low_tranche_qty = torch.zeros(N, dtype=torch.float32, device=device)
    mid_buy_px = torch.zeros(N, dtype=torch.float32, device=device)
    mid_tranche_qty = torch.zeros(N, dtype=torch.float32, device=device)
    entry_s = torch.zeros(N, dtype=torch.int16, device=device)

    for wi, (_, _, rho_0, rel_off) in enumerate(windows):
        if rho_0 > 0.01:
            s0_w, s1_w = wi * E, (wi + 1) * E
            b1_0 = float(max(exec_tick, float(b1_w_all[0, wi].item())))
            epx = max(exec_tick, round((b1_0 * (1.0 + rel_off)) / exec_tick) * exec_tick)
            iqty = torch.floor((env_budgets_e * float(rho_0) / b1_0) * 100.0) / 100.0
            v_m = iqty >= 2.0
            rcash = torch.clamp(env_budgets_e - iqty * b1_0 * (1.0 - val_env.maker_fee), min=0.0)
            cash[s0_w:s1_w] = torch.where(v_m, rcash, env_budgets_e)
            pos_qty[s0_w:s1_w] = torch.where(v_m, iqty, torch.zeros_like(iqty))
            avg_entry[s0_w:s1_w] = torch.where(v_m, torch.full_like(iqty, epx), torch.zeros_like(iqty))
            pos_cost_usd[s0_w:s1_w] = torch.where(v_m, iqty * epx * (1.0 + val_env.maker_fee), torch.zeros_like(iqty))
            min_buy_px[s0_w:s1_w] = torch.where(v_m, torch.full_like(iqty, epx), torch.zeros_like(iqty))
            low_tranche_qty[s0_w:s1_w] = torch.where(v_m, iqty * 0.5, torch.zeros_like(iqty))
            mid_buy_px[s0_w:s1_w] = torch.where(v_m, torch.full_like(iqty, epx), torch.zeros_like(iqty))
            mid_tranche_qty[s0_w:s1_w] = torch.where(v_m, iqty * 0.5, torch.zeros_like(iqty))
            entry_s[s0_w:s1_w] = -120

    bid_px = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qty = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qa = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qa0 = torch.zeros((N, S), dtype=dt, device=device)
    bid_act_s = torch.zeros((N, S), dtype=torch.int16, device=device)
    ask_px = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qty = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qa = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qa0 = torch.zeros((N, S), dtype=dt, device=device)
    ask_act_s = torch.zeros((N, S), dtype=torch.int16, device=device)
    api_tokens = torch.full((N,), TOKEN_BUCKET_CAP_L28, dtype=torch.float32, device=device)

    init_b1 = torch.clamp(b1_w_all[0], min=exec_tick).unsqueeze(1).expand(W, E).reshape(N)
    start_mtm_eq = torch.clamp(cash + pos_qty * init_b1 * (1.0 - val_env.maker_fee), min=1.0)
    peak_mtm_eq = start_mtm_eq.clone()
    prev_mtm_eq = start_mtm_eq.clone()
    prev_val_pred = torch.zeros(N, dtype=torch.float32, device=device)

    sum_log_dd = torch.zeros(N, dtype=torch.float32, device=device)
    max_dd_pct = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mut = torch.zeros(N, dtype=torch.float32, device=device)
    sum_fqp = torch.zeros(N, dtype=torch.float32, device=device)
    sum_val_err = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mode_p = torch.zeros((N, K_MODES_L28), dtype=torch.float32, device=device)
    sum_mode_ent = torch.zeros(N, dtype=torch.float32, device=device)
    n_fills = torch.zeros(N, dtype=torch.int16, device=device)
    n_cancels = torch.zeros(N, dtype=torch.int16, device=device)
    n_requeues = torch.zeros(N, dtype=torch.int16, device=device)
    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)

    for s_t in range(win_len - 1):
        s_nxt = s_t + 1
        (
            mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
            min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
            bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
            ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
            api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
            sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
            sum_mode_p, sum_mode_ent,
            n_fills, n_cancels, n_requeues,
        ) = fused_l28_hierarchical_subcent_tick(
            s_steps_i16[s_t], s_steps_i16[s_nxt], pre_mkt_all[s_t], adv_risk_w_all[s_t],
            b1_w_all[s_t], a1_w_all[s_t], b1q_cent_w_all[s_t], a1q_cent_w_all[s_t],
            rb_w_all[s_t], rs_w_all[s_t],
            bL_w_all[s_t], aL_w_all[s_t], bqL_w_all[s_t], aqL_w_all[s_t],
            b1_w_all[s_nxt], a1_w_all[s_nxt], sv_w_all[s_nxt], bv_w_all[s_nxt],
            w1_ag_c, wr1_c, br1_c, wr2_c,
            w2m_c, b2m_c, wg1_c, bg1_c, wg2_c, bg2_c,
            wrec_c, brec_c,
            wreg1_c, breg1b_c, breg1a_c, wreg2_c, breg2b_c, breg2a_c,
            echelon_k01, budget,
            mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
            min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
            bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
            ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
            api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
            sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
            sum_mode_p, sum_mode_ent,
            n_fills, n_cancels, n_requeues,
            exec_tick, feat_tick, val_env.maker_fee, val_env.taker_fee,
            1.0, 0, true_latency,
        )

    end_b1 = torch.clamp(b1_w_all[win_len - 1], min=exec_tick).unsqueeze(1).expand(W, E).reshape(N)
    rem = pos_qty > 1e-6
    end_eq = torch.clamp(torch.where(rem, cash + pos_qty * end_b1 * (1.0 - val_env.maker_fee), cash), min=1.0)
    n_steps_f = float(win_len - 1)
    dlog_mat = torch.log(end_eq / start_mtm_eq).view(W, C, B_bud)
    m_log_dd = (sum_log_dd / n_steps_f).view(W, C, B_bud)
    return {
        "traj_growth_bp": 10000.0 * (dlog_mat - 0.55 * m_log_dd),
        "mtm_delta_pct": (100.0 * (end_eq - start_mtm_eq) / budget).view(W, C, B_bud),
        "pnl_usd": (end_eq - start_mtm_eq).view(W, C, B_bud),
        "max_dd_pct": max_dd_pct.view(W, C, B_bud),
        "fills": n_fills.float().view(W, C, B_bud),
        "cancels": n_cancels.float().view(W, C, B_bud),
        "requeues": n_requeues.float().view(W, C, B_bud),
        "fqp": sum_fqp.view(W, C, B_bud),
        "mode_p": (sum_mode_p / n_steps_f).view(W, C, B_bud, K_MODES_L28),
        "mode_ent": (sum_mode_ent / n_steps_f).view(W, C, B_bud),
    }


def _summarize_model_metrics(
    sub_out: dict[str, torch.Tensor],
    crs_out: dict[str, torch.Tensor],
    ci: int,
    sub_wins: list[tuple[str, int, float, float]],
    crs_wins: list[tuple[str, int, float, float]],
    budget_list: list[float],
) -> dict:
    ln2 = math.log(2.0)
    sub_clean_idx = [i for i, (_, _, r0, _) in enumerate(sub_wins) if r0 <= 0.01]
    sub_inv_idx = [i for i, (_, _, r0, _) in enumerate(sub_wins) if r0 > 0.01]

    per_budget = {}
    for bi, bud in enumerate(budget_list):
        sub_ev_b = float(sub_out["mtm_delta_pct"][:, ci, bi].mean().item())
        sub_cl_b = float(sub_out["mtm_delta_pct"][sub_clean_idx, ci, bi].mean().item()) if sub_clean_idx else sub_ev_b
        sub_inv_b = float(sub_out["mtm_delta_pct"][sub_inv_idx, ci, bi].mean().item()) if sub_inv_idx else 0.0
        sub_pnl_b = float(sub_out["pnl_usd"][:, ci, bi].sum().item())
        sub_dd_b = float(sub_out["max_dd_pct"][:, ci, bi].max().item())
        sub_f_b = float(sub_out["fills"][:, ci, bi].sum().item())
        sub_c_b = float(sub_out["cancels"][:, ci, bi].sum().item())
        crs_ev_b = float(crs_out["mtm_delta_pct"][:, ci, bi].mean().item())
        crs_dd_b = float(crs_out["max_dd_pct"][:, ci, bi].max().item())
        per_budget[f"${int(bud)}"] = {
            "subcent_blind_ev_pct": round(sub_ev_b, 4),
            "subcent_clean_ev_pct": round(sub_cl_b, 4),
            "subcent_inv_ev_pct": round(sub_inv_b, 4),
            "subcent_total_pnl_usd": round(sub_pnl_b, 2),
            "subcent_max_dd_pct": round(sub_dd_b, 3),
            "subcent_fills_total": round(sub_f_b, 1),
            "subcent_cancels_total": round(sub_c_b, 1),
            "subcent_cnl_per_fill": round(sub_c_b / max(sub_f_b, 0.5), 2),
            "crisis_ev_pct": round(crs_ev_b, 4),
            "crisis_max_dd_pct": round(crs_dd_b, 3),
        }

    g_all = torch.cat([sub_out["traj_growth_bp"][:, ci, :], crs_out["traj_growth_bp"][:, ci, :]], dim=0).reshape(-1)
    k_cvar = max(1, g_all.numel() // 4)
    cvar25_bp = float(torch.topk(g_all, k_cvar, largest=False).values.mean().item())
    mean_g_bp = float(g_all.mean().item())
    q50_bp = float(torch.quantile(g_all, 0.50).item())
    down_semi = float(torch.sqrt(torch.mean(torch.clamp(-g_all, min=0.0) ** 2)).item())

    sub_ev_all = float(sub_out["mtm_delta_pct"][:, ci, :].mean().item())
    sub_cl_all = float(sub_out["mtm_delta_pct"][sub_clean_idx, ci, :].mean().item())
    sub_inv_all = float(sub_out["mtm_delta_pct"][sub_inv_idx, ci, :].mean().item())
    crs_ev_all = float(crs_out["mtm_delta_pct"][:, ci, :].mean().item())
    sub_dd_mean = float(sub_out["max_dd_pct"][:, ci, :].max(dim=0).values.mean().item())
    crs_dd_mean = float(crs_out["max_dd_pct"][:, ci, :].max(dim=0).values.mean().item())

    tot_f_sub = float(sub_out["fills"][:, ci, :].sum(dim=0).mean().item())
    tot_c_sub = float(sub_out["cancels"][:, ci, :].sum(dim=0).mean().item())
    cnl_pf_sub = tot_c_sub / max(tot_f_sub, 0.5)
    fqp_sub = float(sub_out["fqp"][:, ci, :].sum(dim=0).mean().item())
    eta_q_sub = 100.0 * min(1.0, fqp_sub / max(tot_f_sub, 1.0)) if tot_f_sub >= 0.5 else 100.0

    mp_bar = sub_out["mode_p"][:, ci, :, :].reshape(-1, K_MODES_L28).mean(dim=0)
    h_marg = float(-(mp_bar * torch.log(torch.clamp(mp_bar, min=1e-6))).sum().item())
    h_cond = float(sub_out["mode_ent"][:, ci, :].mean().item())
    mi_bits = max(0.0, (h_marg - h_cond) / ln2)

    churn_pen = max(0.0, cnl_pf_sub - 6.0) * 0.55
    coronation_geo_bp = (
        0.50 * mean_g_bp
        + 0.25 * cvar25_bp
        + 0.20 * q50_bp
        + 30.0 * sub_cl_all
        + 18.0 * sub_inv_all
        + 28.0 * sub_ev_all
        + 14.0 * crs_ev_all
        - 0.18 * down_semi
        - churn_pen
    )
    return {
        "coronation_geo_bp": round(coronation_geo_bp, 2),
        "cvar25_bp": round(cvar25_bp, 2),
        "subcent_blind_ev_pct": round(sub_ev_all, 4),
        "subcent_clean_ev_pct": round(sub_cl_all, 4),
        "subcent_inv_ev_pct": round(sub_inv_all, 4),
        "subcent_dd_pct": round(sub_dd_mean, 3),
        "subcent_fills": round(tot_f_sub, 1),
        "subcent_cancels": round(tot_c_sub, 1),
        "subcent_cnl_per_fill": round(cnl_pf_sub, 2),
        "subcent_eta_queue_pct": round(eta_q_sub, 1),
        "crisis_ev_pct": round(crs_ev_all, 4),
        "crisis_dd_pct": round(crs_dd_mean, 3),
        "mode_probs": [round(float(x), 4) for x in mp_bar.tolist()],
        "mode_mi_bits": round(mi_bits, 4),
        "per_budget": per_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-len", type=int, default=2400)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    win_len = int(args.win_len)

    print("=" * 124)
    print("👑 ФАЗА 4 — КОРОНАЦИОННЫЙ СЛЕПОЙ OOS-ТУРНИР УРОКА 28 (UNSEALING FINAL 5.50H SUBCENT BLIND OOS)")
    print("=" * 124)

    raw_lob = np.load(BASE_LOB_CACHE_PATH, allow_pickle=True)
    n_total = len(raw_lob["bid_p"])
    feat_tick = float(raw_lob["tick_size"]) if "tick_size" in raw_lob.files else 0.01

    # Берём хвост 1,200,000 секунд ВКЛЮЧАЯ физически запечатанный `Final SubCent Blind OOS` `[3,076,379 : 3,096,179]`:
    s0 = max(0, n_total - 1_200_000)
    ts_sub = np.array(raw_lob["ts"][s0:n_total], dtype=np.float64, copy=True)
    sub = {
        k: np.array(raw_lob[k][s0:n_total], dtype=np.float32, copy=True)
        for k in ["bid_p", "ask_p", "bid_q", "ask_q", "buy_vol", "sell_vol", "y_sweep_12s", "y_bleed_60s", "y_crash_300s", "y_exit_30s"]
    }
    sub["tick_size"] = feat_tick
    v7_raw_np = np.array(raw_lob["v7_signals_58d"][s0:n_total], dtype=np.float32, copy=True)
    bo36_np = np.array(raw_lob["bo_signals_36d"][s0:n_total], dtype=np.float32, copy=True)
    raw_lob.close()
    del raw_lob

    l28_npz = np.load(L28_CACHE_PATH, allow_pickle=False)
    st_tree_42 = read_l28_cache_global_slice(l28_npz, "st_tree_42d", s0, n_total)
    bo_tree_42 = read_l28_cache_global_slice(l28_npz, "bo_tree_42d", s0, n_total)
    sub_micro_16 = read_l28_cache_global_slice(l28_npz, "subcent_micro_16d", s0, n_total)

    subcent_local_start = int(l28_npz["subcent_era_start"]) - s0
    blind_global_start = n_total - BLIND_OOS_DURATION_SEC
    blind_local_start = blind_global_start - s0
    n_slice = n_total - s0

    sub["bid_p"][subcent_local_start:] = np.array(l28_npz["cent_bid_p10"], dtype=np.float32)
    sub["ask_p"][subcent_local_start:] = np.array(l28_npz["cent_ask_p10"], dtype=np.float32)
    sub["bid_q"][subcent_local_start:] = np.array(l28_npz["cent_bid_q10"], dtype=np.float32)
    sub["ask_q"][subcent_local_start:] = np.array(l28_npz["cent_ask_q10"], dtype=np.float32)

    subcent_books = {
        "bid_p50": torch.from_numpy(np.array(l28_npz["subcent_bid_p50"], dtype=np.float32, copy=True)).to(device),
        "ask_p50": torch.from_numpy(np.array(l28_npz["subcent_ask_p50"], dtype=np.float32, copy=True)).to(device),
        "bid_q50": torch.from_numpy(np.array(l28_npz["subcent_bid_q50"], dtype=np.float32, copy=True)).to(device),
        "ask_q50": torch.from_numpy(np.array(l28_npz["subcent_ask_q50"], dtype=np.float32, copy=True)).to(device),
    }
    l28_npz.close()
    del l28_npz
    gc.collect()

    print(
        f"🔓 Распечатан `Final SubCent Blind OOS`: [{blind_global_start:,} : {n_total:,}] = {n_total - blind_global_start:,}с "
        f"({datetime.fromtimestamp(ts_sub[blind_local_start], MSK_TZ).strftime('%Y-%m-%d %H:%M:%S')} -> "
        f"{datetime.fromtimestamp(ts_sub[-1], MSK_TZ).strftime('%Y-%m-%d %H:%M:%S')} MSK)"
    )

    bp_t = torch.from_numpy(sub["bid_p"][:, :5]).to(device)
    ap_t = torch.from_numpy(sub["ask_p"][:, :5]).to(device)
    bq_t = torch.from_numpy(sub["bid_q"][:, :5]).to(device)
    aq_t = torch.from_numpy(sub["ask_q"][:, :5]).to(device)
    bv_t = torch.from_numpy(sub["buy_vol"]).to(device)
    sv_t = torch.from_numpy(sub["sell_vol"]).to(device)

    lob_vel8 = torch.zeros((n_slice, 8), dtype=torch.float32, device=device)
    lob_vel8[5:, 0] = torch.clamp((bp_t[5:, 0] - bp_t[:-5, 0]) / (3.0 * feat_tick), -2.0, 2.0)
    lob_vel8[5:, 1] = torch.clamp((ap_t[5:, 0] - ap_t[:-5, 0]) / (3.0 * feat_tick), -2.0, 2.0)
    lob_vel8[5:, 2] = torch.clamp(torch.log1p(torch.clamp(bq_t[5:, 0], min=0.0)) - torch.log1p(torch.clamp(bq_t[:-5, 0], min=0.0)), -2.0, 2.0)
    lob_vel8[5:, 3] = torch.clamp(torch.log1p(torch.clamp(aq_t[5:, 0], min=0.0)) - torch.log1p(torch.clamp(aq_t[:-5, 0], min=0.0)), -2.0, 2.0)
    lob_vel8[:, 4] = torch.clamp((bq_t[:, 0] - aq_t[:, 0]) / (bq_t[:, 0] + aq_t[:, 0] + 10.0), -1.0, 1.0)
    b_deep = bq_t[:, 1:5].sum(dim=1)
    a_deep = aq_t[:, 1:5].sum(dim=1)
    lob_vel8[:, 5] = torch.clamp((b_deep - a_deep) / (b_deep + a_deep + 50.0), -1.0, 1.0)
    lob_vel8[:, 6] = torch.clamp(torch.log1p(torch.clamp(bv_t, min=0.0)) / 8.0, 0.0, 2.0)
    lob_vel8[:, 7] = torch.clamp(torch.log1p(torch.clamp(sv_t, min=0.0)) / 8.0, 0.0, 2.0)
    del bp_t, ap_t, bq_t, aq_t, bv_t, sv_t, b_deep, a_deep

    v7_raw_t = torch.from_numpy(v7_raw_np).to(device)
    st_t = torch.from_numpy(st_tree_42).to(device)
    bo_t = torch.from_numpy(bo_tree_42).to(device)
    sm_t = torch.from_numpy(sub_micro_16).to(device)
    del st_tree_42, bo_tree_42, sub_micro_16

    # 1) Признаки `ext138` для L28 (508D без 90s — 1-в-1 с тренером и лайвом):
    ext138_all = build_ext_mkt_138_tensor(st_t, bo_t, v7_raw_t, lob_vel8, sm_t).to(dtype=dt).contiguous()
    del st_t, bo_t, sm_t

    # 2) Нативные признаки `v7_74` + `bo36` для честного тестирования `Live L27` и `L26`:
    v7_vel8 = torch.zeros((n_slice, 8), dtype=torch.float32, device=device)
    vel_cols_legacy = [0, 2, 4, 6, 8, 12, 34, 42]
    v7_vel8[5:] = torch.clamp(v7_raw_t[5:, vel_cols_legacy] - v7_raw_t[:-5, vel_cols_legacy], -3.0, 3.0)
    v7_74_all = torch.cat([v7_raw_t, v7_vel8, lob_vel8], dim=-1).to(dtype=dt).contiguous()
    bo36_all = torch.from_numpy(bo36_np).to(device=device, dtype=dt).contiguous()
    del v7_raw_np, bo36_np, v7_raw_t, v7_vel8, lob_vel8

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    ref_ck = torch.load(L27_DIR / "l27_xwave_0722_0700_0171.pt", map_location="cpu", weights_only=False)
    if "backbone_state" in ref_ck:
        backbone.load_state_dict(ref_ck["backbone_state"], strict=False)

    bf16_runner = FusedBF16BackboneRunner(backbone, dt, device)
    lob_fu_chunks = []
    with torch.no_grad():
        for i0 in range(0, len(lob_t), 100_000):
            chunk_gpu = torch.from_numpy(lob_t[i0 : i0 + 100_000]).float().to(device)
            lr_ch = backbone.encode_lob_batch(chunk_gpu).to(dtype=dt)
            lob_fu_chunks.append((lr_ch @ bf16_runner.w_fu_lob + bf16_runner.b_fu).contiguous())
            del chunk_gpu, lr_ch
    del lob_t
    lob_fu_all = torch.cat(lob_fu_chunks, dim=0)
    del lob_fu_chunks

    val_env = VectorizedFreeMultiOrderEnv(
        bids=torch.from_numpy(sub["bid_p"]).float(),
        asks=torch.from_numpy(sub["ask_p"]).float(),
        bids_q=torch.from_numpy(sub["bid_q"]).float(),
        asks_q=torch.from_numpy(sub["ask_q"]).float(),
        buy_vol=torch.from_numpy(sub["buy_vol"]).float(),
        sell_vol=torch.from_numpy(sub["sell_vol"]).float(),
        lob_tensors=torch.empty((1, 10, 4), dtype=torch.float32),
        flow_market=torch.from_numpy(flow_m).float(),
        targets=torch.from_numpy(targets_np).float(),
        num_envs=18,
        n_side_slots=12,
        tick_size=feat_tick,
        device=device,
        allow_scratch_stop_asks=True,
    )
    roll_sell_30s_all = _build_roll_sell_30s(val_env.sell_vol)

    # Строим плотную непересекающуюся сетку по всему запечатанному `Final SubCent Blind OOS` (`19,800s`):
    blind_starts = list(range(blind_local_start, n_slice - win_len - 10, 1200))
    subcent_blind_windows: list[tuple[str, int, float, float]] = []
    for wi, ws in enumerate(blind_starts):
        # Каждую точку проверяем и с чистого листа (`rho=0`), и со стартовым инвентарём (`rho in {0.35, 0.65}`):
        subcent_blind_windows.append((f"BLIND_CL_{wi:02d}", int(ws), 0.0, 0.0))
        rho_inv = 0.35 if (wi % 2 == 0) else 0.65
        off_inv = 0.008 if (wi % 2 == 0) else 0.025
        subcent_blind_windows.append((f"BLIND_INV_{wi:02d}", int(ws), rho_inv, off_inv))

    # Историческая корзина стресс-крахов и сквизов (`$0.01` 10-lvl):
    mid_hist = 0.5 * (sub["bid_p"][:subcent_local_start, 0] + sub["ask_p"][:subcent_local_start, 0])
    stride_h = 1800
    cand_h = np.arange(1200, subcent_local_start - win_len - 1200, stride_h)
    rng_h = np.array(
        [(float(np.max(mid_hist[s : s + win_len]) - np.min(mid_hist[s : s + win_len])) / max(0.05, float(mid_hist[s]))) for s in cand_h],
        dtype=np.float32,
    )
    top_crs_idx = np.argsort(-rng_h)[:12]
    crisis_windows: list[tuple[str, int, float, float]] = []
    for ci_w, idx_h in enumerate(top_crs_idx):
        ws_c = int(cand_h[int(idx_h)])
        r0_c = 0.0 if (ci_w % 2 == 0) else 0.55
        of_c = 0.0 if (ci_w % 2 == 0) else 0.035
        crisis_windows.append((f"CRS_{ci_w:02d}", ws_c, r0_c, of_c))

    del sub, flow_m, targets_np
    gc.collect()
    torch.cuda.empty_cache()

    budget_list = [835.0, 1500.0, 5000.0]
    budget_tiers = torch.tensor(budget_list, dtype=torch.float32, device=device)

    l28_models: list[tuple[str, Path]] = []
    for label, p in [
        ("L28_14h_Best_4Fold_Val", L28_DIR / "l28_best_val.pt"),
        ("L28_14h_CrossWave_Soup", L28_DIR / "l28_crosswave_soup.pt"),
        ("L28_14h_SubCent_Sniper", L28_DIR / "l28_wave_13_specialist.pt"),
        ("L28_Champ_Iter0800", L28_DIR / "l28_champ_iter_0800.pt"),
        ("L28_8h_Milestone_BestVal", L28_DIR / "l28_8h_milestone_best_val.pt"),
        ("L28_8h_Milestone_Soup", L28_DIR / "l28_8h_milestone_soup.pt"),
    ]:
        if p.exists():
            l28_models.append((label, p))

    legacy_models: list[tuple[str, Path]] = []
    vps_champ_p = BASE_DIR / "models/prl/release_champion_v1.pt"
    for label, p in [
        ("VPS_Live_Server_Champ_L27", vps_champ_p if vps_champ_p.exists() else (L27_DIR / "l27_xwave_0722_0700_0171.pt")),
        ("L27_Soup_Top3_Hotfix", L27_DIR / "l27_soup_top3.pt"),
        ("L26_Champion_0027_Hotfix", L26_DIR / "l26_champ_iter_0027.pt"),
    ]:
        if p.exists():
            legacy_models.append((label, p))

    all_summaries: dict[str, dict] = {}

    if l28_models:
        l28_states = []
        for _, p in l28_models:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            l28_states.append({k: ck[k].float() for k in [
                "w1", "b1", "w_res1", "b_res1", "w_res2", "w2_moe", "b2_moe",
                "w_gate1", "b_gate1", "w_gate2", "b_gate2", "w_rec", "b_rec",
                "w_reg1", "b_reg1_b", "b_reg1_a", "w_reg2", "b_reg2_b", "b_reg2_a",
            ]})
        print(f"⚡ Прогон {len(l28_models)} моделей Урока 28 (`508D` без `90s`) на {len(subcent_blind_windows)} окнах Blind SubCent + {len(crisis_windows)} Crisis...")
        sub_l28 = _eval_universal_group(
            val_env, bf16_runner, lob_fu_all, ext138_all, None, None, roll_sell_30s_all,
            subcent_books, subcent_local_start, l28_states, True,
            subcent_blind_windows, win_len, 0.0001, feat_tick, budget_tiers, device, True,
        )
        crs_l28 = _eval_universal_group(
            val_env, bf16_runner, lob_fu_all, ext138_all, None, None, roll_sell_30s_all,
            None, subcent_local_start, l28_states, True,
            crisis_windows, win_len, feat_tick, feat_tick, budget_tiers, device, True,
        )
        for ci, (label, p) in enumerate(l28_models):
            m = _summarize_model_metrics(sub_l28, crs_l28, ci, subcent_blind_windows, crisis_windows, budget_list)
            m["ckpt_path"] = str(p)
            m["arch"] = "508D_L28_No90s"
            all_summaries[label] = m

    if legacy_models:
        leg_states = [_load_480d_model_as_moe_state(p) for _, p in legacy_models]
        print(f"⚡ Прогон {len(legacy_models)} базовых моделей L27/L26 (`480D` на нативных `370D` признаках) в идентичной `$0.0001` Blind OOS физике...")
        sub_leg = _eval_universal_group(
            val_env, bf16_runner, lob_fu_all, ext138_all, v7_74_all, bo36_all, roll_sell_30s_all,
            subcent_books, subcent_local_start, leg_states, False,
            subcent_blind_windows, win_len, 0.0001, feat_tick, budget_tiers, device, True,
        )
        crs_leg = _eval_universal_group(
            val_env, bf16_runner, lob_fu_all, ext138_all, v7_74_all, bo36_all, roll_sell_30s_all,
            None, subcent_local_start, leg_states, False,
            crisis_windows, win_len, feat_tick, feat_tick, budget_tiers, device, True,
        )
        for ci, (label, p) in enumerate(legacy_models):
            m = _summarize_model_metrics(sub_leg, crs_leg, ci, subcent_blind_windows, crisis_windows, budget_list)
            m["ckpt_path"] = str(p)
            m["arch"] = "480D_Legacy"
            all_summaries[label] = m

    print("\n" + "=" * 136)
    print(
        f"{'Модель':<28} | {'Geo(bp)':>8} | {'BlindSubEV':>10} | {'CleanEV':>9} | {'InvEV':>9} | "
        f"{'$835 EV':>8} | {'$1500 EV':>9} | {'$5000 EV':>9} | {'SubDD':>6} | {'Cnl/F':>6} | {'CrisisEV':>9}"
    )
    print("-" * 136)
    ranked = sorted(all_summaries.items(), key=lambda kv: kv[1]["coronation_geo_bp"], reverse=True)
    for name, m in ranked:
        pb = m["per_budget"]
        print(
            f"{name:<28} | {m['coronation_geo_bp']:>+8.2f} | {m['subcent_blind_ev_pct']:>+9.4f}% | "
            f"{m['subcent_clean_ev_pct']:>+8.4f}% | {m['subcent_inv_ev_pct']:>+8.4f}% | "
            f"{pb['$835']['subcent_blind_ev_pct']:>+7.3f}% | {pb['$1500']['subcent_blind_ev_pct']:>+8.3f}% | "
            f"{pb['$5000']['subcent_blind_ev_pct']:>+8.3f}% | {m['subcent_dd_pct']:>5.2f}% | "
            f"{m['subcent_cnl_per_fill']:>6.2f} | {m['crisis_ev_pct']:>+8.4f}%"
        )
    print("=" * 136)

    out_json = {
        "evaluated_at_msk": datetime.now(MSK_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 2),
        "blind_oos_range": [int(blind_global_start), int(n_total)],
        "blind_oos_seconds": int(n_total - blind_global_start),
        "n_subcent_blind_windows": len(subcent_blind_windows),
        "n_crisis_windows": len(crisis_windows),
        "budgets_usd": budget_list,
        "ranking": [name for name, _ in ranked],
        "models": all_summaries,
    }
    L28_DIR.mkdir(parents=True, exist_ok=True)
    out_path = L28_DIR / "l28_blind_coronation_results.json"
    out_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Результаты слепого коронационного турнира сохранены -> {out_path} ({time.time() - t0:.1f}с)")


if __name__ == "__main__":
    main()
