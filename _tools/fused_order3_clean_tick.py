#!/usr/bin/env python3
"""
_tools/fused_order3_clean_tick.py
=============================================================================
Order-3 Fused Multi-Modal GPU Tick & Basket Evaluator (39D Market + 110D Agent = 149D).
Чистая реализация без эвристических ограничений и без устаревших зависимостей.
=============================================================================
"""
from __future__ import annotations

import math
import os
os.environ.setdefault('TRITON_CACHE_DIR', '/home/restorator/crypto_bot/scratch/.triton_cache')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/home/restorator/crypto_bot/scratch/.inductor_cache')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn.functional as F

MKT_DIM_ORDER3 = 96
AG_DIM_ORDER3 = 110
TOTAL_DIM_ORDER3 = 206

MEM_DIM_ORDER3 = 64
K_MODES_ORDER3 = 8
EXPERT_OUT_DIM_ORDER3 = 58
MOE_OUT_DIM_ORDER3 = K_MODES_ORDER3 * EXPERT_OUT_DIM_ORDER3  # 464
MOE_MLP_H_ORDER3 = 48
MOE_MLP_UP_DIM_ORDER3 = K_MODES_ORDER3 * (2 * MOE_MLP_H_ORDER3)  # 8 * 96 = 768
ROUTER_EXTRA_DIM_ORDER3 = 42
ROUTER_IN_DIM_ORDER3 = 256 + 8 + 8 + MEM_DIM_ORDER3 + ROUTER_EXTRA_DIM_ORDER3 + K_MODES_ORDER3  # 386
ROUTER_H_DIM_ORDER3 = 128
ROUTER_RES_H_ORDER3 = 64
ROUTER_OUT_DIM_ORDER3 = 2 * K_MODES_ORDER3  # 16: 8 mode logits + 8 liquid update gates!
REG_IN_DIM_ORDER3 = 28 + K_MODES_ORDER3     # 36D per-slot features
REG_H_DIM_ORDER3 = 64
TOKEN_REFILL_PER_SEC_ORDER3 = 1.2
TOKEN_BUCKET_CAP_ORDER3 = 8.0


def _fanned_slot_softmax(
    logits: torch.Tensor,
    tau: torch.Tensor,
    max_slot_share: torch.Tensor,
    floor: float = 0.008,
) -> torch.Tensor:
    scaled = logits / tau
    p_raw = torch.softmax(scaled, dim=-1)
    p_cap1 = torch.minimum(p_raw, max_slot_share)
    rem1 = torch.clamp(1.0 - p_cap1.sum(dim=-1, keepdim=True), min=0.0)
    room1 = torch.clamp(max_slot_share - p_cap1, min=0.0)
    p_redist = p_cap1 + rem1 * (room1 * (p_raw + 0.02)) / torch.clamp((room1 * (p_raw + 0.02)).sum(dim=-1, keepdim=True), min=1e-6)
    p_keep = torch.where(p_redist >= floor, p_redist, torch.zeros_like(p_redist))
    s_keep = p_keep.sum(dim=-1, keepdim=True)
    return torch.where(s_keep > 1e-5, p_keep / torch.clamp(s_keep, min=1e-6), p_redist)

def _normalize_agent_z110_per_block(z_ag_110: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    zf = z_ag_110.float()
    b0 = zf[:, 0:6] / torch.clamp(torch.sqrt(torch.mean(zf[:, 0:6] * zf[:, 0:6], dim=-1, keepdim=True) + 1e-4), min=0.10)
    b1 = zf[:, 6:30] / torch.clamp(torch.sqrt(torch.mean(zf[:, 6:30] * zf[:, 6:30], dim=-1, keepdim=True) + 1e-4), min=0.08)
    b2 = zf[:, 30:54] / torch.clamp(torch.sqrt(torch.mean(zf[:, 30:54] * zf[:, 30:54], dim=-1, keepdim=True) + 1e-4), min=0.10)
    b3 = zf[:, 54:62] / torch.clamp(torch.sqrt(torch.mean(zf[:, 54:62] * zf[:, 54:62], dim=-1, keepdim=True) + 1e-4), min=0.12)
    b4 = zf[:, 62:70] / torch.clamp(torch.sqrt(torch.mean(zf[:, 62:70] * zf[:, 62:70], dim=-1, keepdim=True) + 1e-4), min=0.12)
    b5 = zf[:, 70:86] / torch.clamp(torch.sqrt(torch.mean(zf[:, 70:86] * zf[:, 70:86], dim=-1, keepdim=True) + 1e-4), min=0.10)
    b6 = zf[:, 86:110] / torch.clamp(torch.sqrt(torch.mean(zf[:, 86:110] * zf[:, 86:110], dim=-1, keepdim=True) + 1e-4), min=0.15)
    return torch.cat([b0, b1, b2, b3, b4, b5, b6], dim=-1).to(dt)


def _scale_mkt_44d(z_mkt: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """Signal Stack v2 (Lesson 41): z-калибровка всех 60D каналов (0 мертвых и 0 заглушенных входов, 100% паритет Sim/Live)."""
    zm = z_mkt.float().clone()
    zm[..., 0] = torch.clamp((zm[..., 0] - 0.0020) * 200.0, -0.50, 3.0)
    if zm.shape[-1] >= 60:
        # Разблокировка заглушенных микро-возвратов ch[22..28] (IC=-0.288) и базиса BigONE ch[12..13] через гладкий tanh:
        zm[..., 12] = 2.0 * torch.tanh(zm[..., 12] * 12.5)
        zm[..., 13] = torch.clamp((zm[..., 13] - 0.0020) * 150.0, -0.50, 2.5)
        zm[..., 22:29] = 2.0 * torch.tanh(zm[..., 22:29] * 12.5)
    if zm.shape[-1] >= 44:
        zm[..., 40] = torch.clamp(zm[..., 40] * 12.0, -3.0, 3.0)
        zm[..., 41] = torch.clamp(zm[..., 41] * 10.0, -3.0, 3.0)
        zm[..., 43] = torch.clamp(zm[..., 43] * 4.0, 0.0, 3.0)
    if zm.shape[-1] >= 60:
        zm[..., 44:50] = (zm[..., 44:50] - 0.25) * 2.5
        zm[..., 50:52] = 2.0 * torch.tanh(zm[..., 50:52] * 2.5)
        zm[..., 52:55] = (zm[..., 52:55] - 0.60) * 2.5
        zm[..., 55] = zm[..., 55] * 1.5
        zm[..., 56] = zm[..., 56] * 2.5
        zm[..., 57] = zm[..., 57] * 1.2
        zm[..., 58:60] = zm[..., 58:60] * 3.0
    if zm.shape[-1] >= 72:
        # Signal Stack v3 (12D Wave Radar 5m..60m + геометрия макро-волн):
        zm[..., 60:64] = zm[..., 60:64] * 2.2
        zm[..., 64] = (zm[..., 64] - 0.18) * 3.2
        zm[..., 65] = (zm[..., 65] - 0.10) * 4.0
        zm[..., 66:69] = zm[..., 66:69] * 2.2
        zm[..., 69] = zm[..., 69] * 1.5
        zm[..., 70:72] = (zm[..., 70:72] - 0.20) * 2.5
    if zm.shape[-1] >= 84:
        # Signal Stack v4 (L71: 12D Extended Order-1 [320s..1800s] + Order-2 [bleed_180s..carousel_240s]):
        zm[..., 72:76] = zm[..., 72:76] * 2.0
        zm[..., 76:78] = (zm[..., 76:78] - 0.95) * 8.0
        zm[..., 78] = zm[..., 78] * 3.0
        zm[..., 79:84] = (zm[..., 79:84] - 0.40) * 2.5
    if zm.shape[-1] >= 88:
        # Signal Stack v5 (L75 Order-2.5: Penny-Jump Small/Whale + Real Live Buy Trap & Sell Regret):
        zm[..., 84:88] = (zm[..., 84:88] - 0.30) * 2.5
    if zm.shape[-1] >= 94:
        # Signal Stack v6 (L78 Medium-Term 5m..30m Structure & Tape CVD):
        zm[..., 88:91] = zm[..., 88:91] * 2.5
        zm[..., 91:94] = zm[..., 91:94] * 2.0
    if zm.shape[-1] >= 96:
        # Signal Stack v7 (L80 2-Hour Exhaustion/Capitulation & Tape CVD):
        zm[..., 94] = zm[..., 94] * 2.5
        zm[..., 95] = zm[..., 95] * 2.0
    return zm.to(dt)


def _sparse_slot_softmax(logits: torch.Tensor, tau: torch.Tensor, floor: float = 0.025) -> torch.Tensor:
    scaled = logits / tau
    p_raw = torch.softmax(scaled, dim=-1)
    p_keep = torch.where(p_raw >= floor, p_raw, torch.zeros_like(p_raw))
    s_keep = p_keep.sum(dim=-1, keepdim=True)
    return torch.where(s_keep > 1e-5, p_keep / torch.clamp(s_keep, min=1e-6), p_raw)


def _lookup_sparse_bid_lob_q(b_p10_1w: torch.Tensor, b_q10_1w: torch.Tensor, px_ns: torch.Tensor, W: int, E: int, S: int, tick_size: float) -> torch.Tensor:
    """Точный поиск объёма на ценовом уровне `px_ns` в разреженном Bid-стакане: пустой тик -> 0.0 PRL!"""
    px_wes = px_ns.view(W, E * S)
    idx = torch.clamp(torch.searchsorted(-b_p10_1w, -px_wes, right=False), 0, 9)
    mp = torch.gather(b_p10_1w, 1, idx)
    mq = torch.gather(b_q10_1w, 1, idx)
    matched = torch.abs(mp - px_wes) < (0.45 * tick_size)
    return torch.where(matched, mq, torch.zeros_like(mq)).view(-1, S)


def _lookup_sparse_ask_lob_q(a_p10_1w: torch.Tensor, a_q10_1w: torch.Tensor, px_ns: torch.Tensor, W: int, E: int, S: int, tick_size: float) -> torch.Tensor:
    """Точный поиск объёма на ценовом уровне `px_ns` в разреженном Ask-стакане: пустой тик -> 0.0 PRL!"""
    px_wes = px_ns.view(W, E * S)
    idx = torch.clamp(torch.searchsorted(a_p10_1w, px_wes, right=False), 0, 9)
    mp = torch.gather(a_p10_1w, 1, idx)
    mq = torch.gather(a_q10_1w, 1, idx)
    matched = torch.abs(mp - px_wes) < (0.45 * tick_size)
    return torch.where(matched, mq, torch.zeros_like(mq)).view(-1, S)



def _order3_clean_tick_step(
    s_cur_i16: torch.Tensor,
    s_nxt_i16: torch.Tensor,
    pre_mkt_gw: torch.Tensor,
    mkt_w0: torch.Tensor,
    b1_w0: torch.Tensor,
    a1_w0: torch.Tensor,
    b1q_w0: torch.Tensor,
    a1q_w0: torch.Tensor,
    rb_w0: torch.Tensor,
    rs_w0: torch.Tensor,
    bids_10_w: torch.Tensor,
    asks_10_w: torch.Tensor,
    bids_q_10_w: torch.Tensor,
    asks_q_10_w: torch.Tensor,
    nb1_w0: torch.Tensor,
    na1_w0: torch.Tensor,
    nb1q_w0: torch.Tensor,
    na1q_w0: torch.Tensor,
    sv_w0: torch.Tensor,
    bv_w0: torch.Tensor,
    w1_ag_g: torch.Tensor,
    wr1_g: torch.Tensor,
    br1_g: torch.Tensor,
    wr2_g: torch.Tensor,
    w2_moe_g: torch.Tensor,
    b2_moe_g: torch.Tensor,
    w_moe_up_g: torch.Tensor,
    b_moe_up_g: torch.Tensor,
    w_moe_dn_g: torch.Tensor,
    w_gate1_g: torch.Tensor,
    b_gate1_g: torch.Tensor,
    w_gate_res1_g: torch.Tensor,
    b_gate_res_g: torch.Tensor,
    w_gate_res2_g: torch.Tensor,
    w_gate2_g: torch.Tensor,
    b_gate2_g: torch.Tensor,
    w_rec_g: torch.Tensor,
    b_rec_g: torch.Tensor,
    w_mem2h_g: torch.Tensor,
    w_reg1_g: torch.Tensor,
    b_reg1_b_g: torch.Tensor,
    b_reg1_a_g: torch.Tensor,
    w_reg_res1_g: torch.Tensor,
    b_reg_res_g: torch.Tensor,
    w_reg_res2_g: torch.Tensor,
    w_reg2_g: torch.Tensor,
    b_reg2_b_g: torch.Tensor,
    b_reg2_a_g: torch.Tensor,
    echelon_k01: torch.Tensor,
    budget: torch.Tensor,
    mem_h: torch.Tensor,
    gate_state: torch.Tensor,
    strat_state: torch.Tensor,
    cash: torch.Tensor,
    pos_qty: torch.Tensor,
    pos_cost_usd: torch.Tensor,
    avg_entry: torch.Tensor,
    min_buy_px: torch.Tensor,
    low_tranche_qty: torch.Tensor,
    mid_buy_px: torch.Tensor,
    mid_tranche_qty: torch.Tensor,
    entry_s: torch.Tensor,
    bid_px: torch.Tensor,
    bid_qty: torch.Tensor,
    bid_qa: torch.Tensor,
    bid_qa0: torch.Tensor,
    bid_act_s: torch.Tensor,
    ask_px: torch.Tensor,
    ask_qty: torch.Tensor,
    ask_qa: torch.Tensor,
    ask_qa0: torch.Tensor,
    ask_act_s: torch.Tensor,
    api_tokens: torch.Tensor,
    peak_mtm_eq: torch.Tensor,
    sum_log_dd: torch.Tensor,
    max_mtm_dd_pct: torch.Tensor,
    sum_mut_load: torch.Tensor,
    sum_fill_qprog: torch.Tensor,
    sum_val_err: torch.Tensor,
    prev_val_pred: torch.Tensor,
    prev_mtm_eq: torch.Tensor,
    sum_mode_p: torch.Tensor,
    sum_mode_ent: torch.Tensor,
    n_fills: torch.Tensor,
    n_cancels: torch.Tensor,
    n_requeues: torch.Tensor,
    last_sell_px: torch.Tensor,
    last_sell_s: torch.Tensor,
    reg_stats: torch.Tensor,
    tick_size: float,
    maker_fee: float,
    taker_fee: float,
    true_latency: bool = True,
    harsh_stress: bool = False,
):
    W = b1_w0.shape[0]
    G = w1_ag_g.shape[0]
    N = cash.shape[0]
    E = N // W
    B = E // G
    M = W * B
    S = echelon_k01.shape[0]
    dt = mem_h.dtype

    b1_1w = torch.clamp(b1_w0, min=tick_size)
    a1_1w = torch.clamp(a1_w0, min=b1_1w + tick_size)
    b1_q_1w = torch.clamp(b1q_w0, min=0.0)
    a1_q_1w = torch.clamp(a1q_w0, min=0.0)
    l1_bid_usd_1w = torch.clamp(b1_1w * b1_q_1w, min=100.0)
    l1_ask_usd_1w = torch.clamp(a1_1w * a1_q_1w, min=100.0)
    roll_buy_1w = torch.clamp(rb_w0, min=10.0)
    roll_sell_1w = torch.clamp(rs_w0, min=10.0)

    b_p10_1w = torch.clamp(bids_10_w[:, :10], min=tick_size)
    b_q10_raw_1w = torch.clamp(bids_q_10_w[:, :10], min=0.0)
    a_p10_1w = torch.clamp(asks_10_w[:, :10], min=b1_1w.unsqueeze(1) + tick_size)
    a_q10_raw_1w = torch.clamp(asks_q_10_w[:, :10], min=0.0)

    b1 = b1_1w.unsqueeze(1).expand(W, E).reshape(N)
    a1 = a1_1w.unsqueeze(1).expand(W, E).reshape(N)
    b1_q = b1_q_1w.unsqueeze(1).expand(W, E).reshape(N)
    a1_q = a1_q_1w.unsqueeze(1).expand(W, E).reshape(N)
    l1_bid_usd = l1_bid_usd_1w.unsqueeze(1).expand(W, E).reshape(N)
    l1_ask_usd = l1_ask_usd_1w.unsqueeze(1).expand(W, E).reshape(N)
    roll_buy_n = roll_buy_1w.unsqueeze(1).expand(W, E).reshape(N)
    roll_sell_n = roll_sell_1w.unsqueeze(1).expand(W, E).reshape(N)
    mkt_w = mkt_w0.unsqueeze(1).expand(W, E, MKT_DIM_ORDER3).reshape(N, MKT_DIM_ORDER3)

    # Физический Token-Bucket лимита API биржи (+0.35 токена/сек, всплесковый резерв 10.0):
    api_tokens = torch.clamp(api_tokens + TOKEN_REFILL_PER_SEC_ORDER3, 0.0, TOKEN_BUCKET_CAP_ORDER3)

    s_cur_f = s_cur_i16.float()
    has_p = pos_qty > 1e-6
    has_pos = has_p.float()
    pos_val = pos_qty * b1
    tot_eq = torch.clamp(cash + pos_val, min=1.0)
    pos_ratio = torch.clamp(pos_val / tot_eq, 0.0, 1.5)
    res_cash_now = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    unreserved_cash_now = torch.clamp(cash - res_cash_now, min=0.0)
    free_cash_ratio = torch.clamp(unreserved_cash_now / tot_eq, 0.0, 1.0)

    eff_pos_scale = torch.clamp(pos_ratio / 0.25, 0.15, 1.0)
    net_pos_ret = ((b1 - avg_entry) / torch.clamp(avg_entry, min=tick_size)) - 2.0 * maker_fee
    unr_pct = torch.where(has_p, net_pos_ret * 25.0 * eff_pos_scale, torch.zeros_like(has_pos))
    hold_ratio = torch.where(has_p, torch.clamp((s_cur_f - entry_s.float()) / 300.0, 0.0, 2.0) * eff_pos_scale, torch.zeros_like(has_pos))

    vis_b_lvl_q0 = _lookup_sparse_bid_lob_q(b_p10_1w, b_q10_raw_1w, bid_px, W, E, S, tick_size)
    in_book_b0 = (bid_qty > 1e-6) & (bid_px <= b1.unsqueeze(1) + 0.1 * tick_size)
    bid_qa = torch.where(in_book_b0, torch.minimum(bid_qa, vis_b_lvl_q0), bid_qa)

    vis_a_lvl_q0 = _lookup_sparse_ask_lob_q(a_p10_1w, a_q10_raw_1w, ask_px, W, E, S, tick_size)
    in_book_a0 = (ask_qty > 1e-6) & (ask_px >= a1.unsqueeze(1) - 0.1 * tick_size)
    ask_qa = torch.where(in_book_a0, torch.minimum(ask_qa, vis_a_lvl_q0), ask_qa)
    act_b_mask = bid_qty > 1e-6
    act_a_mask = ask_qty > 1e-6
    has_bid = act_b_mask.any(dim=1).float()
    has_ask = act_a_mask.any(dim=1).float()
    n_bid_slots = act_b_mask.float().sum(dim=1) / 12.0
    n_ask_slots = act_a_mask.float().sum(dim=1) / 12.0

    acct_8d = torch.stack(
        [
            has_pos,
            pos_ratio,
            free_cash_ratio,
            torch.clamp(unr_pct, -2.5, 2.5),
            hold_ratio,
            has_bid,
            has_ask,
            n_bid_slots - n_ask_slots,
        ],
        dim=-1,
    ).to(dt)

    best_b_idx = torch.argmin(torch.where(act_b_mask, b1.unsqueeze(1) - bid_px, torch.full_like(bid_px, 999.0)), dim=1, keepdim=True)
    best_b_act = torch.gather(bid_act_s, 1, best_b_idx).squeeze(1).float()
    best_b_age = torch.where(has_bid > 0.5, torch.clamp((s_cur_f - best_b_act) / 60.0, 0.0, 2.0), torch.zeros_like(has_bid))

    best_a_idx = torch.argmin(torch.where(act_a_mask, ask_px - a1.unsqueeze(1), torch.full_like(ask_px, 999.0)), dim=1, keepdim=True)
    best_a_act = torch.gather(ask_act_s, 1, best_a_idx).squeeze(1).float()
    best_a_age = torch.where(has_ask > 0.5, torch.clamp((s_cur_f - best_a_act) / 60.0, 0.0, 2.0), torch.zeros_like(has_ask))

    bo_basis = mkt_w[:, 12]
    bo_spread = mkt_w[:, 13]
    st_flow_30s = mkt_w[:, 10]
    st_imb5 = mkt_w[:, 3]
    wave_exh = mkt_w[:, 44]
    car_succ = mkt_w[:, 48]

    queue_8d = torch.stack([best_b_age, best_a_age, bo_basis, bo_spread, st_flow_30s, st_imb5, wave_exh, car_succ], dim=-1).to(dt)

    bud_l1 = torch.log1p(budget / l1_bid_usd) / 3.0
    cash_l1 = torch.log1p(unreserved_cash_now / l1_bid_usd) / 3.0
    pos_l1_ask = torch.clamp((pos_qty * a1) / l1_ask_usd, 0.0, 4.0) / 2.0
    pos_tape = torch.clamp(pos_qty / roll_buy_n, 0.0, 10.0) / 5.0
    low_tr_ratio = torch.where(has_p, torch.clamp(low_tranche_qty / torch.clamp(pos_qty, min=1e-6), 0.0, 1.0), torch.zeros_like(has_pos))
    net_lifo_ret = ((b1 - min_buy_px) / torch.clamp(min_buy_px, min=tick_size)) - 2.0 * maker_fee
    lifo_edge_pct = torch.where(has_p, torch.clamp(net_lifo_ret * 30.0, -2.5, 2.5), torch.zeros_like(has_pos))
    multi_6d = torch.stack([bud_l1, cash_l1, pos_l1_ask, pos_tape, low_tr_ratio, lifo_edge_pct], dim=-1).to(dt)

    b_slot_12 = torch.clamp((bid_qty * b1.unsqueeze(1)) / torch.clamp(budget.unsqueeze(1), min=25.0), 0.0, 1.0).to(dt)
    a_slot_12 = torch.clamp(ask_qty / torch.clamp(pos_qty.unsqueeze(1), min=2.0), 0.0, 1.0).to(dt)
    b_dist_12 = torch.where(act_b_mask, torch.log1p(torch.clamp((b1.unsqueeze(1) - bid_px) / tick_size, min=0.0)) / 3.5, torch.zeros_like(bid_qty)).to(dt)
    a_dist_12 = torch.where(act_a_mask, torch.log1p(torch.clamp((ask_px - a1.unsqueeze(1)) / tick_size, min=0.0)) / 3.5, torch.zeros_like(ask_qty)).to(dt)

    b_slot_age_12 = torch.where(act_b_mask, torch.clamp((s_cur_f - bid_act_s.float()) / 60.0, 0.0, 2.0), torch.full_like(bid_qty, -1.0)).to(dt)
    a_slot_age_12 = torch.where(act_a_mask, torch.clamp((s_cur_f - ask_act_s.float()) / 60.0, 0.0, 2.0), torch.full_like(ask_qty, -1.0)).to(dt)

    # 1. Прямой проход 60D+110D ствола Signal Stack v2 + 4-горизонтная латентная память mem_h (32D: ~3.7s, ~45s, ~180s, ~735s):
    z_ag_raw = torch.cat(
        [multi_6d, b_slot_12, a_slot_12, b_dist_12, a_dist_12, acct_8d, queue_8d, mem_h[:, :16], b_slot_age_12, a_slot_age_12],
        dim=-1,
    )
    z_ag = _normalize_agent_z110_per_block(z_ag_raw, dt)
    z_ag_g = z_ag.view(W, G, B, 110).permute(1, 0, 2, 3).reshape(G, M, 110)
    pre_ag = torch.bmm(z_ag_g, w1_ag_g).view(G, W, B, 384)
    pre_all = (pre_ag + pre_mkt_gw.unsqueeze(2)).reshape(G, M, 384)

    h_base = torch.relu(pre_all[:, :, :256])
    h_gate = (2.0 * torch.sigmoid(pre_all[:, :, 256:384])).repeat_interleave(2, dim=-1)
    h1 = h_base * h_gate

    res_pre = torch.bmm(h1, wr1_g) + br1_g
    res_mid = torch.relu(res_pre[:, :, :64]) * (2.0 * torch.sigmoid(res_pre[:, :, 64:128]))
    h2 = h1 + torch.bmm(res_mid, wr2_g)

    rec_raw = torch.bmm(h2, w_rec_g) + b_rec_g
    rec_u = torch.tanh(rec_raw[:, :, :MEM_DIM_ORDER3])
    rec_gate = torch.cat(
        [
            torch.sigmoid(rec_raw[:, :, 64:72] + 1.0),
            torch.sigmoid(rec_raw[:, :, 72:80] + 3.8),
            torch.sigmoid(rec_raw[:, :, 80:88] + 5.2),
            torch.sigmoid(rec_raw[:, :, 88:96] + 6.6),
            torch.sigmoid(rec_raw[:, :, 96:104] + 2.2),
            torch.sigmoid(rec_raw[:, :, 104:112] + 4.5),
            torch.sigmoid(rec_raw[:, :, 112:120] + 7.5),
            torch.sigmoid(rec_raw[:, :, 120:128] + 8.9),
        ],
        dim=-1,
    )
    mem_prev_g = mem_h.view(W, G, B, MEM_DIM_ORDER3).permute(1, 0, 2, 3).reshape(G, M, MEM_DIM_ORDER3)
    mem_next_g = rec_gate * mem_prev_g + (1.0 - rec_gate) * rec_u
    mem_h = mem_next_g.view(G, W, B, MEM_DIM_ORDER3).permute(1, 0, 2, 3).reshape(N, MEM_DIM_ORDER3)

    # Прямая обратная связь 64D многомасштабной памяти в экспертов и роутер (снимает 1-секундную слепоту MoE!):
    h2_ctx = h2 + torch.bmm(mem_next_g, w_mem2h_g)

    # 1B. Двухкаскадный Вентильный Рекуррентный Маршрутизатор (386D -> 128 SiLU + ResSwiGLU -> 16):
    acct_g = acct_8d.view(W, G, B, 8).permute(1, 0, 2, 3).reshape(G, M, 8)
    queue_g = queue_8d.view(W, G, B, 8).permute(1, 0, 2, 3).reshape(G, M, 8)
    r_extra_30d = torch.cat(
        [
            mkt_w[:, 0:1], mkt_w[:, 40:44], mkt_w[:, 44:45], mkt_w[:, 48:49],
            mkt_w[:, 29:34], mkt_w[:, 36:37], mkt_w[:, 39:40], mkt_w[:, 55:56], mkt_w[:, 12:13],
            mkt_w[:, 60:66],
            mkt_w[:, 72:76], mkt_w[:, 79:82], mkt_w[:, 83:84],
            mkt_w[:, 84:88],
            mkt_w[:, 88:96],
        ],
        dim=-1,
    ).to(dt)
    r_extra_g = r_extra_30d.view(W, G, B, ROUTER_EXTRA_DIM_ORDER3).permute(1, 0, 2, 3).reshape(G, M, ROUTER_EXTRA_DIM_ORDER3)
    gate_prev_g = gate_state.view(W, G, B, K_MODES_ORDER3).permute(1, 0, 2, 3).reshape(G, M, K_MODES_ORDER3)
    prev_g_mode_g = torch.softmax(gate_prev_g, dim=-1).to(dt)

    router_in_g = torch.cat([h2_ctx, acct_g, queue_g, mem_next_g, r_extra_g, prev_g_mode_g], dim=-1)  # [G, M, 386]
    r_hid1_g = F.silu(torch.bmm(router_in_g, w_gate1_g) + b_gate1_g)  # [G, M, 128]
    r_res_pre_g = torch.bmm(r_hid1_g, w_gate_res1_g) + b_gate_res_g   # [G, M, 128]
    r_res_act_g = F.silu(r_res_pre_g[:, :, :ROUTER_RES_H_ORDER3]) * (2.0 * torch.sigmoid(r_res_pre_g[:, :, ROUTER_RES_H_ORDER3:]))
    r_hid_g = r_hid1_g + torch.bmm(r_res_act_g, w_gate_res2_g)

    gate_out_g = torch.bmm(r_hid_g, w_gate2_g).float() + b_gate2_g.float()  # [G, M, 16]
    raw_logits_g = gate_out_g[:, :, :K_MODES_ORDER3]
    beta_gate_g = torch.sigmoid(gate_out_g[:, :, K_MODES_ORDER3:])
    gate_next_g = (1.0 - beta_gate_g) * gate_prev_g + beta_gate_g * raw_logits_g
    g_mode_g = torch.softmax(gate_next_g, dim=-1)  # [G, M, K]
    gate_state = gate_next_g.view(G, W, B, K_MODES_ORDER3).permute(1, 0, 2, 3).reshape(N, K_MODES_ORDER3)

    moe_lin_g = (torch.bmm(h2_ctx, w2_moe_g).float() + b2_moe_g.float()).view(G, M, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    moe_up_raw = (torch.bmm(h2_ctx, w_moe_up_g) + b_moe_up_g).view(G, M, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    moe_up_act = F.silu(moe_up_raw[:, :, :, :MOE_MLP_H_ORDER3]) * (2.0 * torch.sigmoid(moe_up_raw[:, :, :, MOE_MLP_H_ORDER3:]))
    moe_mlp_g = torch.matmul(moe_up_act.permute(0, 2, 1, 3), w_moe_dn_g).permute(0, 2, 1, 3).float()
    moe_all_g = moe_lin_g + moe_mlp_g
    strat_inst_g = (g_mode_g.unsqueeze(-1) * moe_all_g).sum(dim=2)  # [G, M, 58]
    strat_inst = strat_inst_g.view(G, W, B, EXPERT_OUT_DIM_ORDER3).permute(1, 0, 2, 3).reshape(N, EXPERT_OUT_DIM_ORDER3)

    # Адаптивный нелинейный демпфер плана котирования (разрывает петлю PLACE -> free_cash -> CANCEL, оставляя Taker мгновенным!):
    alpha_bid = torch.sigmoid(strat_inst[:, 50:51])
    alpha_ask = torch.sigmoid(strat_inst[:, 51:52])
    bid_blend = (1.0 - alpha_bid) * strat_state + alpha_bid * strat_inst
    ask_blend = (1.0 - alpha_ask) * strat_state + alpha_ask * strat_inst
    strat_raw = torch.cat(
        [
            bid_blend[:, 0:13],     # [0..12] exp_bid + bid_dist
            ask_blend[:, 13:25],    # [13..24] ask_dist
            strat_inst[:, 25:26],   # [25] u_taker (мгновенный!)
            bid_blend[:, 26:35],    # [26..34] reg ctx + tau_b
            ask_blend[:, 35:36],    # [35] tau_a
            strat_inst[:, 36:37],   # [36] max_slip (мгновенный!)
            bid_blend[:, 37:41],    # [37..40] inside_b, retreat_b, span_b, gamma_b
            ask_blend[:, 41:44],    # [41..43] skew_blend, z42, inside_a
            bid_blend[:, 44:45],    # [44] max_share
            ask_blend[:, 45:47],    # [45..46] span_a_off, gamma_a_off
            bid_blend[:, 47:48],    # [47] renorm_blend
            strat_inst[:, 48:49],   # [48] u_taker_buy (мгновенный!)
            ask_blend[:, 49:50],    # [49] wave_hold_pct
            strat_inst[:, 50:52],   # [50..51] сами вентили
            ask_blend[:, 52:53],    # [52] tranche_carousel_z (независимая карусель нижнего транша!)
            bid_blend[:, 53:54],    # [53] sniper_front_z (подавление отступа трусости у Bid1!)
            ask_blend[:, 54:55],    # [54] unwind_front_z (L78 ранний мейкер-сброс передних эшелонов при сломе тренда!)
            bid_blend[:, 55:56],    # [55] micro_sniper_z (L79 изолированный микро-снайпер широкого спреда на слоте #00!)
            bid_blend[:, 56:57],    # [56] recycle_anchor_z (L80 якорь рециклинга: потолок откупа ниже last_sell_px!)
            bid_blend[:, 57:58],    # [57] deep_hyst_z (L80 эндогенный гистерезис перестановки глубоких заявок!)
        ],
        dim=-1,
    )
    strat_state = strat_raw
    g_mode = g_mode_g.view(G, W, B, K_MODES_ORDER3).permute(1, 0, 2, 3).reshape(N, K_MODES_ORDER3)

    # Накопление статистики модального разнообразия I(State; Mode) без ручных ярлыков:
    sum_mode_p = sum_mode_p + g_mode
    mode_ent_step = -(g_mode * torch.log(torch.clamp(g_mode, min=1e-6))).sum(dim=-1)
    sum_mode_ent = sum_mode_ent + mode_ent_step

    # 1C. 100% Эндогенная экспозиция [0..1] + L79 Wide-Spread Micro-Sniper [55] (до 6.5% на слоте #00 при спреде >35..80 bp):
    rel_sp_now = (a1 - b1) / torch.clamp(b1, min=tick_size)
    wide_sp_gate = torch.clamp((rel_sp_now - 0.0035) / 0.0045, 0.0, 1.0)
    eff_micro_s = torch.sigmoid(strat_raw[:, 55]) * wide_sp_gate
    exp_bid_frac = torch.maximum(torch.sigmoid(strat_raw[:, 0]), 0.065 * eff_micro_s)
    renorm_blend = torch.sigmoid(strat_raw[:, 47:48])

    tau_b = 0.08 + 2.42 * torch.sigmoid(strat_raw[:, 34:35])
    tau_a = 0.08 + 2.42 * torch.sigmoid(strat_raw[:, 35:36])
    max_share_b = 0.08 + 0.92 * torch.sigmoid(strat_raw[:, 44:45])
    max_share_a = 0.08 + 0.92 * torch.sigmoid(strat_raw[:, 44:45] + 0.25)
    logits_b_12 = torch.cat([strat_raw[:, 1:2] + (5.0 * eff_micro_s).unsqueeze(1), strat_raw[:, 2:13]], dim=-1)
    pi_bid = _fanned_slot_softmax(logits_b_12, tau_b, max_share_b, floor=0.005)
    pi_ask = _fanned_slot_softmax(strat_raw[:, 13:25], tau_a, max_share_a, floor=0.005)
    # 2. Непрерывный эндогенный Taker-вентиль БЕЗ карательного сноса обеспеченных Ask-очередей при частичном хеджировании:
    u_taker = torch.tanh(F.relu(strat_raw[:, 25]))
    max_slip_pct = 0.005 + 0.145 * torch.sigmoid(strat_raw[:, 36])
    raw_sell_q = pos_qty * u_taker
    is_taker = (raw_sell_q >= 2.0) & (api_tokens >= 1.0)
    sell_q_req = torch.where((pos_qty - raw_sell_q) < 2.0, pos_qty, raw_sell_q)

    b_p10 = b_p10_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    b_q10_honest = b_q10_raw_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    min_cross_px = torch.clamp(b1 * (1.0 - max_slip_pct), min=tick_size).unsqueeze(1)
    elig_q10 = torch.where(b_p10 >= min_cross_px - 0.1 * tick_size, b_q10_honest, torch.zeros_like(b_q10_honest))
    cum_q10 = torch.cumsum(elig_q10, dim=1)
    prev_q10 = cum_q10 - elig_q10
    take_10 = torch.clamp(torch.minimum(sell_q_req.unsqueeze(1) - prev_q10, elig_q10), min=0.0)
    actual_taker_q = take_10.sum(dim=1)
    exec_taker = is_taker & (actual_taker_q >= 2.0)

    eff_t_fee = taker_fee
    proceeds = (take_10 * b_p10).sum(dim=1) * (1.0 - eff_t_fee)
    tk_sell_px = (take_10 * b_p10).sum(dim=1) / torch.clamp(actual_taker_q, min=1e-6)
    last_sell_px = torch.where(exec_taker, tk_sell_px, last_sell_px)
    last_sell_s = torch.where(exec_taker, s_nxt_i16.expand_as(last_sell_s), last_sell_s)
    rem_pos = torch.clamp(pos_qty - actual_taker_q, min=0.0)
    rem_frac = torch.where(pos_qty > 1e-6, rem_pos / torch.clamp(pos_qty, min=1e-6), torch.zeros_like(pos_qty))
    full_exit = rem_pos < 2.0

    cash = torch.where(exec_taker, cash + proceeds, cash)
    pos_qty = torch.where(exec_taker, rem_pos, pos_qty)
    pos_cost_usd = torch.where(exec_taker, pos_cost_usd * rem_frac, pos_cost_usd)
    avg_entry = torch.where(exec_taker & (rem_pos <= 1e-5), torch.zeros_like(avg_entry), avg_entry)
    min_buy_px = torch.where(exec_taker & (rem_pos <= 1e-5), torch.zeros_like(min_buy_px), min_buy_px)
    low_tranche_qty = torch.where(exec_taker, low_tranche_qty * rem_frac, low_tranche_qty)
    mid_buy_px = torch.where(exec_taker & (rem_pos <= 1e-5), torch.zeros_like(mid_buy_px), mid_buy_px)
    mid_tranche_qty = torch.where(exec_taker, mid_tranche_qty * rem_frac, mid_tranche_qty)

    # Снимаем ТОЛЬКО те Ask-слоты, которым не хватает оставшегося rem_pos (передние очереди выживают!):
    cum_ask_pre = torch.cumsum(ask_qty, dim=1)
    keep_ask_on_taker = (~full_exit.unsqueeze(1)) & (cum_ask_pre <= rem_pos.unsqueeze(1) + 1e-5) & (ask_qty >= 2.0)
    drop_ask_taker = exec_taker.unsqueeze(1) & (ask_qty > 1e-6) & (~keep_ask_on_taker)
    had_ask_taker_cnt = drop_ask_taker.long().sum(dim=1)
    ask_qty = torch.where(drop_ask_taker, torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(drop_ask_taker, torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(drop_ask_taker, torch.zeros_like(ask_qa), ask_qa)
    ask_qa0 = torch.where(drop_ask_taker, torch.zeros_like(ask_qa0), ask_qa0)
    api_tokens = torch.where(exec_taker, torch.clamp(api_tokens - 1.0 - had_ask_taker_cnt.float(), min=0.0), api_tokens)
    n_fills = torch.clamp(n_fills.long() + exec_taker.long(), max=2000000).to(torch.int32)

    # 2B. Эндогенный импульсный вход в волну u_taker_buy [ch 48] по реальному 10-уровневому Ask-стакану:
    u_taker_buy = torch.tanh(F.relu(strat_raw[:, 48] - 1.2))
    tot_bud_pre_tb = torch.maximum(budget, cash + pos_qty * b1)
    res_b_pre_tb = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    free_c_tb = torch.clamp(cash * 0.996 - res_b_pre_tb, min=0.0)
    rem_exp_tb = torch.clamp(tot_bud_pre_tb * 0.994 * exp_bid_frac - pos_qty * a1 - res_b_pre_tb, min=0.0)
    want_tb_usd = torch.minimum(free_c_tb, rem_exp_tb) * u_taker_buy
    is_tb = (~exec_taker) & (u_taker_buy > 1e-4) & (want_tb_usd >= 2.0 * a1 * (1.0 + taker_fee)) & (api_tokens >= 1.5)

    a_p10 = a_p10_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    a_q10_honest = a_q10_raw_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    max_buy_cross_px = (a1 * 1.012).unsqueeze(1)
    elig_aq10 = torch.where(a_p10 <= max_buy_cross_px + 0.1 * tick_size, a_q10_honest, torch.zeros_like(a_q10_honest))
    lvl_cost10 = elig_aq10 * a_p10 * (1.0 + taker_fee)
    cum_cost10 = torch.cumsum(lvl_cost10, dim=1)
    prev_cost10 = cum_cost10 - lvl_cost10
    rem_usd10 = torch.clamp(want_tb_usd.unsqueeze(1) - prev_cost10, min=0.0)
    buy_take_10 = torch.minimum(elig_aq10, rem_usd10 / torch.clamp(a_p10 * (1.0 + taker_fee), min=tick_size))
    actual_tb_q = buy_take_10.sum(dim=1)
    exec_tb = is_tb & (actual_tb_q >= 2.0)
    tb_cost_raw = (buy_take_10 * a_p10).sum(dim=1)
    tb_cost_tot = tb_cost_raw * (1.0 + eff_t_fee)
    tb_avg_px = tb_cost_raw / torch.clamp(actual_tb_q, min=1e-6)

    old_q_tb = pos_qty
    new_q_tb = old_q_tb + actual_tb_q
    entry_s = torch.where(exec_tb & (old_q_tb <= 1e-6), s_cur_i16.expand_as(entry_s), entry_s)
    avg_entry = torch.where(exec_tb, (old_q_tb * avg_entry + tb_cost_raw) / torch.clamp(new_q_tb, min=1e-6), avg_entry)
    min_buy_px = torch.where(exec_tb, torch.where(old_q_tb <= 1e-6, tb_avg_px, torch.minimum(min_buy_px, tb_avg_px)), min_buy_px)
    low_tranche_qty = torch.where(exec_tb & (old_q_tb <= 1e-6), actual_tb_q, low_tranche_qty)
    pos_cost_usd = torch.where(exec_tb, pos_cost_usd + tb_cost_tot, pos_cost_usd)
    cash = torch.where(exec_tb, torch.clamp(cash - tb_cost_tot, min=0.0), cash)
    pos_qty = torch.where(exec_tb, new_q_tb, pos_qty)
    api_tokens = torch.where(exec_tb, torch.clamp(api_tokens - 1.0, min=0.0), api_tokens)
    n_fills = torch.clamp(n_fills.long() + exec_tb.long(), max=2000000).to(torch.int32)

    # 3. 100% Эндогенная полно-спредовая + волновая геометрия 24 слотов + L72 Tranche Carousel [52] & Spread Sniper [53]:
    spread_room = torch.clamp(a1 - b1 - tick_size, min=0.0)
    inside_b = spread_room * torch.sigmoid(strat_raw[:, 37])
    sniper_front = torch.sigmoid(strat_raw[:, 53])
    retreat_b = b1 * 0.04 * torch.sigmoid(strat_raw[:, 38]) * (1.0 - sniper_front)
    span_b_usd = b1 * (0.0005 + 0.2495 * torch.sigmoid(strat_raw[:, 39]))
    gamma_b = 0.40 + 2.10 * torch.sigmoid(strat_raw[:, 40])

    curve_b = torch.pow(echelon_k01.unsqueeze(0), gamma_b.unsqueeze(1))
    ech0_m = (echelon_k01 < 0.01).float().unsqueeze(0)  # [1, 12]: 1.0 только на слоте #00!
    eff_retreat_k = retreat_b.unsqueeze(1) * (1.0 - ech0_m * eff_micro_s.unsqueeze(1))
    micro_step_in = torch.minimum(spread_room, torch.full_like(spread_room, 2.0 * tick_size)) * eff_micro_s
    eff_inside_k = torch.minimum(spread_room.unsqueeze(1), inside_b.unsqueeze(1) + ech0_m * micro_step_in.unsqueeze(1))
    raw_t_bid = (b1.unsqueeze(1) + eff_inside_k - eff_retreat_k) - span_b_usd.unsqueeze(1) * curve_b
    # T0.6: Якорь рециклинга [56] — потолок откупа ниже last_sell_px после продажи всплеска (Mode F, 0.00e+00 при b2=-12):
    sell_age_s = torch.clamp(s_cur_f - last_sell_s.float(), min=0.0)
    has_rec_anc = (last_sell_px > 1e-4) & (sell_age_s < 3600.0)
    rec_w = (torch.sigmoid(strat_raw[:, 56]) * torch.exp(-sell_age_s / 1800.0) * has_rec_anc.float()).unsqueeze(1) * (1.0 - ech0_m * eff_micro_s.unsqueeze(1))
    rec_disc = 0.0040 + 0.0160 * torch.sigmoid(strat_raw[:, 56] - 1.0)
    rec_cap_px = (last_sell_px * (1.0 - rec_disc)).unsqueeze(1)
    raw_t_bid = (1.0 - rec_w) * raw_t_bid + rec_w * torch.minimum(raw_t_bid, rec_cap_px)
    max_maker_bid = torch.clamp(a1 - tick_size, min=tick_size).unsqueeze(1)
    t_bid_px = torch.clamp(torch.round(torch.minimum(raw_t_bid, max_maker_bid) / tick_size) * tick_size, min=tick_size)

    hp_now = pos_qty >= 2.0
    hp_now_mask = hp_now.unsqueeze(1)
    ref_entry = torch.where(hp_now & (avg_entry > 1e-4), avg_entry * (1.0 + 2.0 * maker_fee), a1)
    has_low_tr = hp_now & (low_tranche_qty >= 2.0) & (min_buy_px > 1e-4)
    low_ref_entry = torch.where(has_low_tr, min_buy_px * (1.0 + 2.0 * maker_fee), ref_entry)
    ech_front_sq = torch.pow(1.0 - echelon_k01, 2.0).unsqueeze(0)  # [1, 12]: 1.00 на #00 -> 0.30 на #05 -> 0.00 на #11
    tranche_w_k = (torch.sigmoid(strat_raw[:, 52]) * has_low_tr.float()).unsqueeze(1) * ech_front_sq
    ref_entry_k = ref_entry.unsqueeze(1) * (1.0 - tranche_w_k) + low_ref_entry.unsqueeze(1) * tranche_w_k
    # L78 Controlled Maker Stop-Unwind [54]: плавно подтягивает передние Ask-эшелоны к a1 ТОЛЬКО при просадке (ref_entry_k > a1)!
    unwind_w_k = (torch.sigmoid(strat_raw[:, 54]).unsqueeze(1) * ech_front_sq) * (ref_entry_k > a1.unsqueeze(1)).float()
    eff_ref_entry_k = ref_entry_k * (1.0 - unwind_w_k) + a1.unsqueeze(1) * unwind_w_k

    skew_blend = torch.sigmoid(strat_raw[:, 41])
    z42 = strat_raw[:, 42]
    t42_slow = torch.tanh(z42 * 0.25)
    wave_hold_pct = 0.10 * torch.sigmoid(strat_raw[:, 49] - 1.2)
    ask_edge_pct = 0.0035 * torch.tanh(z42) + 0.0765 * (t42_slow * t42_slow * t42_slow) + wave_hold_pct
    span_a_usd = a1 * (0.0005 + 0.2495 * torch.sigmoid(strat_raw[:, 39] + 0.5 * z42 + strat_raw[:, 45]))
    gamma_a = 0.40 + 2.10 * torch.sigmoid(strat_raw[:, 40] + strat_raw[:, 46])

    inside_a = spread_room * torch.sigmoid(strat_raw[:, 43])
    anchor_base_a_k = a1.unsqueeze(1) + skew_blend.unsqueeze(1) * (eff_ref_entry_k - a1.unsqueeze(1)) + (a1 * ask_edge_pct - inside_a).unsqueeze(1)
    min_maker_ask = (b1 + tick_size).unsqueeze(1)
    curve_a = torch.pow(echelon_k01.unsqueeze(0), gamma_a.unsqueeze(1))
    raw_t_ask = anchor_base_a_k + span_a_usd.unsqueeze(1) * curve_a
    t_ask_px = torch.maximum(min_maker_ask, torch.round(raw_t_ask / tick_size) * tick_size)

    val_pred_now = torch.zeros_like(b1)

    tot_bud_eff = torch.maximum(budget, cash + pos_qty * b1)
    target_bid_pool_usd = torch.clamp(tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1, min=0.0)
    q_target_b = (target_bid_pool_usd.unsqueeze(1) * pi_bid) / (t_bid_px * (1.0 + maker_fee))
    q_target_b = torch.where(q_target_b >= 2.0, q_target_b, torch.zeros_like(q_target_b))

    q_target_a = torch.where(hp_now_mask, pos_qty.unsqueeze(1) * pi_ask, torch.zeros_like(pi_ask))
    q_target_a = torch.where(q_target_a >= 2.0, q_target_a, torch.zeros_like(q_target_a))

    # 4. Модально-контекстный Послотовый Нейро-Регулятор (26D -> 48 -> 2D с гистерезисом ценности очереди):
    ctx_4d_b = torch.tanh(strat_raw[:, 26:30])
    ctx_4d_a = torch.tanh(strat_raw[:, 30:34])
    s_2d = s_cur_i16.view(1, 1)
    next_s_2d = s_nxt_i16.view(1, 1)
    base_lat_ticks = 2 if true_latency else 0
    act_lat_2d = next_s_2d + base_lat_ticks

    had_b_pre = bid_qty > 1e-6
    had_a_pre = ask_qty > 1e-6
    age_b_norm = torch.where(had_b_pre, torch.log1p(torch.clamp((s_2d - bid_act_s).float(), min=0.0)) / 6.0, torch.zeros_like(bid_qty))
    age_a_norm = torch.where(had_a_pre, torch.log1p(torch.clamp((s_2d - ask_act_s).float(), min=0.0)) / 6.0, torch.zeros_like(ask_qty))

    bud_2d = torch.clamp(tot_bud_eff.unsqueeze(1), min=25.0)
    pos_2d = torch.clamp(pos_qty.unsqueeze(1), min=2.0)
    cur_val_b = torch.clamp((bid_qty * bid_px) / bud_2d, 0.0, 1.5)
    targ_val_b = torch.clamp((q_target_b * t_bid_px) / bud_2d, 0.0, 1.5)
    cur_val_a = torch.clamp(ask_qty / pos_2d, 0.0, 1.5)
    targ_val_a = torch.clamp(q_target_a / pos_2d, 0.0, 1.5)

    rel_delta_b = torch.clamp((q_target_b - bid_qty) / torch.maximum(torch.maximum(bid_qty, q_target_b), torch.full_like(bid_qty, 2.0)), -2.0, 2.0)
    rel_delta_a = torch.clamp((q_target_a - ask_qty) / torch.maximum(torch.maximum(ask_qty, q_target_a), torch.full_like(ask_qty, 2.0)), -2.0, 2.0)

    drift_b = torch.where(had_b_pre, torch.clamp(((t_bid_px - bid_px) / b1.unsqueeze(1)) * 25.0, -2.0, 2.0), torch.zeros_like(bid_qty))
    drift_a = torch.where(had_a_pre, torch.clamp(((ask_px - t_ask_px) / a1.unsqueeze(1)) * 25.0, -2.0, 2.0), torch.zeros_like(ask_qty))

    cur_dist_b = torch.where(had_b_pre, torch.clamp(((b1.unsqueeze(1) - bid_px) / b1.unsqueeze(1)) * 12.0, -0.5, 3.0), torch.zeros_like(bid_qty))
    cur_dist_a = torch.where(had_a_pre, torch.clamp(((ask_px - a1.unsqueeze(1)) / a1.unsqueeze(1)) * 12.0, -0.5, 3.0), torch.zeros_like(ask_qty))
    b_age_s = torch.where(had_b_pre, torch.clamp((s_cur_f - bid_act_s.float()) / 60.0, 0.0, 2.0), torch.zeros_like(bid_qty))
    a_age_s = torch.where(had_a_pre, torch.clamp((s_cur_f - ask_act_s.float()) / 60.0, 0.0, 2.0), torch.zeros_like(ask_qty))

    depth_b_norm = torch.clamp(((b1.unsqueeze(1) - t_bid_px) / b1.unsqueeze(1)) * 8.0, -0.2, 2.5)
    depth_a_norm = torch.clamp(((t_ask_px - a1.unsqueeze(1)) / a1.unsqueeze(1)) * 8.0, -0.2, 2.5)
    front_b_fl = (t_bid_px >= b1.unsqueeze(1) - 0.5 * tick_size).float()
    front_a_fl = (t_ask_px <= a1.unsqueeze(1) + 0.5 * tick_size).float()
    free_b_2d = free_cash_ratio.unsqueeze(1).expand(N, S)
    free_a_2d = torch.clamp((pos_qty - ask_qty.sum(dim=1)) / torch.clamp(pos_qty, min=2.0), 0.0, 1.0).unsqueeze(1).expand(N, S)

    tot_side_b = cur_val_b.sum(dim=1, keepdim=True).expand(N, S)
    tot_side_a = cur_val_a.sum(dim=1, keepdim=True).expand(N, S)
    near_b_mask = (torch.abs(bid_px.unsqueeze(2) - t_bid_px.unsqueeze(1)) <= 1.1 * tick_size) & had_b_pre.unsqueeze(1)
    near_a_mask = (torch.abs(ask_px.unsqueeze(2) - t_ask_px.unsqueeze(1)) <= 1.1 * tick_size) & had_a_pre.unsqueeze(1)
    eye_s = torch.eye(S, dtype=torch.bool, device=bid_px.device).unsqueeze(0)
    neigh_val_b = torch.clamp((near_b_mask & (~eye_s)).float().bmm(cur_val_b.unsqueeze(2)).squeeze(2), 0.0, 1.5)
    neigh_val_a = torch.clamp((near_a_mask & (~eye_s)).float().bmm(cur_val_a.unsqueeze(2)).squeeze(2), 0.0, 1.5)
    had_b_cnt = had_b_pre.float().sum(dim=1, keepdim=True)
    mean_age_b = torch.where(had_b_cnt > 0.5, (b_age_s * had_b_pre.float()).sum(dim=1, keepdim=True) / torch.clamp(had_b_cnt, min=1.0), torch.zeros_like(tot_side_b)).expand(N, S)
    had_a_cnt = had_a_pre.float().sum(dim=1, keepdim=True)
    mean_age_a = torch.where(had_a_cnt > 0.5, (a_age_s * had_a_pre.float()).sum(dim=1, keepdim=True) / torch.clamp(had_a_cnt, min=1.0), torch.zeros_like(tot_side_a)).expand(N, S)
    api_ratio_2d = (api_tokens / TOKEN_BUCKET_CAP_ORDER3).unsqueeze(1).expand(N, S)
    ech_2d = echelon_k01.unsqueeze(0).expand(N, S)
    sp_rel_2d = torch.clamp(((a1 - b1) / b1) * 40.0, 0.0, 3.0).unsqueeze(1).expand(N, S)
    g_mode_s = g_mode.unsqueeze(1).expand(N, S, K_MODES_ORDER3)

    o1_slot_imp = torch.tanh(
        0.30 * (mkt_w[:, 34:35].float() + mkt_w[:, 36:37].float())
        + 0.35 * (mkt_w[:, 61:62].float() + mkt_w[:, 62:63].float())
        + 0.25 * (mkt_w[:, 72:73].float() + mkt_w[:, 74:75].float())
    ).to(dt)
    slot_edge_b = torch.clamp((((a1.unsqueeze(1) - t_bid_px) / torch.clamp(t_bid_px, min=tick_size)) - 2.0 * maker_fee) * 40.0, -2.0, 3.0)
    slot_exec_b = (0.50 * mkt_w[:, 48:49] + 0.25 * mkt_w[:, 82:83] + 0.25 * mkt_w[:, 83:84] + 0.55 * o1_slot_imp).expand(N, S)
    slot_edge_a = torch.where(
        hp_now_mask & (avg_entry.unsqueeze(1) > 1e-4),
        torch.clamp((((t_ask_px - avg_entry.unsqueeze(1)) / torch.clamp(avg_entry.unsqueeze(1), min=tick_size)) - 2.0 * maker_fee) * 40.0, -2.5, 3.0),
        torch.zeros_like(t_ask_px),
    )
    slot_exec_a = (0.55 * mkt_w[:, 45:46] + 0.25 * mkt_w[:, 80:81] - 0.55 * o1_slot_imp).expand(N, S)

    # 4D датчики реальной FIFO-очереди и долгого горизонта удержания (снимает слепоту регулятора к qprog и стенам L2!):
    qprog_b = torch.where(had_b_pre, torch.clamp(1.0 - bid_qa / (bid_qa0.float() + 10.0), 0.0, 1.0), torch.zeros_like(bid_qty))
    qprog_a = torch.where(had_a_pre, torch.clamp(1.0 - ask_qa / (ask_qa0.float() + 10.0), 0.0, 1.0), torch.zeros_like(ask_qty))
    qa_eta_b = torch.where(had_b_pre, torch.clamp(torch.log1p(bid_qa / roll_sell_n.unsqueeze(1)) / 4.0, 0.0, 2.5), torch.zeros_like(bid_qty))
    qa_eta_a = torch.where(had_a_pre, torch.clamp(torch.log1p(ask_qa / roll_buy_n.unsqueeze(1)) / 4.0, 0.0, 2.5), torch.zeros_like(ask_qty))
    t_vis_bq = _lookup_sparse_bid_lob_q(b_p10_1w, b_q10_raw_1w, t_bid_px, W, E, S, tick_size)
    t_vis_aq = _lookup_sparse_ask_lob_q(a_p10_1w, a_q10_raw_1w, t_ask_px, W, E, S, tick_size)
    wall_b_log = torch.clamp(torch.log1p((t_vis_bq * t_bid_px) / bud_2d) / 2.0, 0.0, 2.5)
    wall_a_log = torch.clamp(torch.log1p((t_vis_aq * t_ask_px) / bud_2d) / 2.0, 0.0, 2.5)
    hold_long_1d = torch.where(hp_now, torch.clamp(torch.log1p(torch.clamp(s_cur_f - entry_s.float(), min=0.0) / 60.0) / 4.0, 0.0, 2.0), torch.zeros_like(b1))
    hold_long_2d = hold_long_1d.unsqueeze(1).expand(N, S)

    feat_b_36 = torch.cat(
        [
            torch.stack(
                [
                    torch.zeros_like(bid_qty), had_b_pre.float(), cur_val_b, targ_val_b, rel_delta_b, drift_b,
                    cur_dist_b, b_age_s, age_b_norm, depth_b_norm, front_b_fl, free_b_2d,
                ],
                dim=-1,
            ),
            ctx_4d_b.unsqueeze(1).expand(N, S, 4),
            torch.stack([tot_side_b, neigh_val_b, mean_age_b, api_ratio_2d, ech_2d, sp_rel_2d, slot_edge_b, slot_exec_b, qprog_b, qa_eta_b, wall_b_log, hold_long_2d], dim=-1),
            g_mode_s,
        ],
        dim=-1,
    ).to(dt)
    feat_a_36 = torch.cat(
        [
            torch.stack(
                [
                    torch.ones_like(ask_qty), had_a_pre.float(), cur_val_a, targ_val_a, rel_delta_a, drift_a,
                    cur_dist_a, a_age_s, age_a_norm, depth_a_norm, front_a_fl, free_a_2d,
                ],
                dim=-1,
            ),
            ctx_4d_a.unsqueeze(1).expand(N, S, 4),
            torch.stack([tot_side_a, neigh_val_a, mean_age_a, api_ratio_2d, ech_2d, sp_rel_2d, slot_edge_a, slot_exec_a, qprog_a, qa_eta_a, wall_a_log, hold_long_2d], dim=-1),
            g_mode_s,
        ],
        dim=-1,
    ).to(dt)

    fb_g = feat_b_36.view(W, G, B, S, REG_IN_DIM_ORDER3).permute(1, 0, 2, 3, 4).reshape(G, M * S, REG_IN_DIM_ORDER3)
    fa_g = feat_a_36.view(W, G, B, S, REG_IN_DIM_ORDER3).permute(1, 0, 2, 3, 4).reshape(G, M * S, REG_IN_DIM_ORDER3)
    rb_h0 = F.silu(torch.bmm(fb_g, w_reg1_g) + b_reg1_b_g)
    ra_h0 = F.silu(torch.bmm(fa_g, w_reg1_g) + b_reg1_a_g)
    rb_res_p = torch.bmm(rb_h0, w_reg_res1_g) + b_reg_res_g
    ra_res_p = torch.bmm(ra_h0, w_reg_res1_g) + b_reg_res_g
    rb_h = rb_h0 + torch.bmm(torch.relu(rb_res_p[:, :, :32]) * (2.0 * torch.sigmoid(rb_res_p[:, :, 32:64])), w_reg_res2_g)
    ra_h = ra_h0 + torch.bmm(torch.relu(ra_res_p[:, :, :32]) * (2.0 * torch.sigmoid(ra_res_p[:, :, 32:64])), w_reg_res2_g)
    rb_out = (torch.bmm(rb_h, w_reg2_g).float() + b_reg2_b_g.float()).view(G, W, B, S, 2).permute(1, 0, 2, 3, 4).reshape(N, S, 2)
    ra_out = (torch.bmm(ra_h, w_reg2_g).float() + b_reg2_a_g.float()).view(G, W, B, S, 2).permute(1, 0, 2, 3, 4).reshape(N, S, 2)
    # T0.4: Эндогенный гистерезис перестановки глубоких заявок [57] (снижает churn отмен в 10-15x, 0.00e+00 при b2=-12):
    hyst_band = 0.0080 * torch.sigmoid(strat_raw[:, 57]).unsqueeze(1)
    abs_drift_b = torch.abs(t_bid_px - bid_px) / torch.clamp(b1.unsqueeze(1), min=tick_size)
    deep_rest_b = (had_b_pre & (bid_px < b1.unsqueeze(1) * 0.992) & (hyst_band > 2e-5)).float()
    hyst_damp_b = 8.0 * torch.sigmoid((hyst_band - abs_drift_b) * 1200.0) * deep_rest_b
    rb_cnl_adj = rb_out[:, :, 1] - hyst_damp_b
    g_plc_b = rb_out[:, :, 0] > 0.0
    g_cnl_b = rb_cnl_b_mask = rb_cnl_adj > 0.0
    g_plc_a = ra_out[:, :, 0] > 0.0
    g_cnl_a = ra_out[:, :, 1] > 0.0
    # 5. Ранжированное по срочности снятие ордеров (в полёте HTTP до bid_act_s/ask_act_s ордер ещё не подтверждён движком Peatio -> рутинный cancel ждёт оседания в книге!):
    settled_b = (s_2d >= bid_act_s) if true_latency else torch.ones_like(had_b_pre)
    crossed_b = bid_px >= (a1.unsqueeze(1) - 0.1 * tick_size)
    want_cnl_b = had_b_pre & (crossed_b | (settled_b & g_cnl_b))
    prio_cnl_b = torch.where(want_cnl_b, rb_cnl_adj + crossed_b.float() * 100.0, torch.full_like(bid_qty, -1e4))
    ord_cb = torch.argsort(prio_cnl_b, dim=1, descending=True)
    inv_cb = torch.argsort(ord_cb, dim=1)
    want_cb_s = torch.gather(want_cnl_b.float(), 1, ord_cb)
    allow_cb_s = (want_cb_s.cumsum(dim=1) <= api_tokens.unsqueeze(1) + 1e-4) & (want_cb_s > 0.5)
    clear_b = torch.gather(allow_cb_s, 1, inv_cb)
    api_tokens = torch.clamp(api_tokens - clear_b.float().sum(dim=1), min=0.0)

    settled_a = (s_2d >= ask_act_s) if true_latency else torch.ones_like(had_a_pre)
    crossed_a = ask_px <= (b1.unsqueeze(1) + 0.1 * tick_size)
    want_cnl_a = had_a_pre & ((~hp_now_mask) | crossed_a | (settled_a & g_cnl_a))
    prio_cnl_a = torch.where(want_cnl_a, ra_out[:, :, 1] + (crossed_a | (~hp_now_mask)).float() * 100.0, torch.full_like(ask_qty, -1e4))
    ord_ca = torch.argsort(prio_cnl_a, dim=1, descending=True)
    inv_ca = torch.argsort(ord_ca, dim=1)
    want_ca_s = torch.gather(want_cnl_a.float(), 1, ord_ca)
    allow_ca_s = (~hp_now_mask) & (want_ca_s > 0.5) | ((want_ca_s.cumsum(dim=1) <= api_tokens.unsqueeze(1) + 1e-4) & (want_ca_s > 0.5))
    clear_a = torch.gather(allow_ca_s, 1, inv_ca)
    api_tokens = torch.clamp(api_tokens - (clear_a & hp_now_mask).float().sum(dim=1), min=0.0)

    n_cnl_b_step = clear_b.long().sum(dim=1)
    n_cnl_a_step = clear_a.long().sum(dim=1) + had_ask_taker_cnt

    # Peatio v2 Cancel-Lock Quarantine: при отмене на шаге s слот НЕ перезанимается в ту же миллисекунду, а часть locked-маржи ждёт async-воркера!
    cnl_lock_frac = 0.65 if true_latency else 0.0
    cnl_locked_b_usd = cnl_lock_frac * torch.where(clear_b, bid_qty * bid_px * (1.0 + maker_fee), torch.zeros_like(bid_qty)).sum(dim=1)
    cnl_locked_a_qty = cnl_lock_frac * torch.where(clear_a, ask_qty, torch.zeros_like(ask_qty)).sum(dim=1)

    bid_qty = torch.where(clear_b, torch.zeros_like(bid_qty), bid_qty)
    bid_px = torch.where(clear_b, torch.zeros_like(bid_px), bid_px)
    bid_qa = torch.where(clear_b, torch.zeros_like(bid_qa), bid_qa)
    bid_qa0 = torch.where(clear_b, torch.zeros_like(bid_qa0), bid_qa0)

    ask_qty = torch.where(clear_a, torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(clear_a, torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(clear_a, torch.zeros_like(ask_qa), ask_qa)
    ask_qa0 = torch.where(clear_a, torch.zeros_like(ask_qa0), ask_qa0)

    res_cash_after_cnl = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1) + cnl_locked_b_usd
    free_cash_for_b = torch.clamp(cash * 0.998 - res_cash_after_cnl, min=0.0)
    rem_exp_for_b = torch.clamp(tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1 - res_cash_after_cnl, min=0.0)
    avail_b_usd = torch.minimum(free_cash_for_b, rem_exp_for_b)

    empty_b_now = (bid_qty <= 1e-6) & ((~clear_b) if true_latency else torch.ones_like(clear_b))
    cand_plc_b = empty_b_now & g_plc_b & (pi_bid > 1e-4) & ((avail_b_usd.unsqueeze(1) / (t_bid_px * (1.0 + maker_fee))) >= 2.0)
    ord_b_idx = torch.argsort(torch.where(cand_plc_b, pi_bid, torch.full_like(pi_bid, -1.0)), dim=1, descending=True)
    inv_b_idx = torch.argsort(ord_b_idx, dim=1)
    cand_b_sorted = torch.gather(cand_plc_b.float(), 1, ord_b_idx)
    allow_b_tok_sorted = (cand_b_sorted.cumsum(dim=1) <= api_tokens.unsqueeze(1) + 1e-4) & (cand_b_sorted > 0.5)
    want_place_b = torch.gather(allow_b_tok_sorted, 1, inv_b_idx)

    pi_sel_b = torch.where(want_place_b, pi_bid + 1e-5, torch.zeros_like(pi_bid))
    sum_pi_sel_b = pi_sel_b.sum(dim=1, keepdim=True)
    full_share_b = torch.where(sum_pi_sel_b > 1e-6, pi_sel_b / torch.clamp(sum_pi_sel_b, min=1e-6), torch.zeros_like(pi_sel_b))
    raw_share_b = torch.where(want_place_b, torch.minimum(pi_bid, full_share_b), torch.zeros_like(pi_bid))
    share_sel_b = renorm_blend * full_share_b + (1.0 - renorm_blend) * raw_share_b
    plc_b_qty = torch.floor(((avail_b_usd.unsqueeze(1) * share_sel_b * 0.999) / (t_bid_px * (1.0 + maker_fee))) * 10000.0) / 10000.0
    valid_b = want_place_b & (plc_b_qty >= 2.0)
    plc_b_qty = torch.where(valid_b, plc_b_qty, torch.zeros_like(plc_b_qty))
    n_plc_b_step = valid_b.long().sum(dim=1)
    api_tokens = torch.clamp(api_tokens - valid_b.float().sum(dim=1), min=0.0)
    n_req_b_step = (clear_b & valid_b).long().sum(dim=1)

    # Интерактивная Order-2.5 физика очереди L2: rival penny-jumping на пассивных BBO-плитах (0 штрафа при шаге внутрь спреда inside_sp!):
    p_pj_s = torch.clamp(mkt_w[:, 84:85].float() * 0.40 + 0.30, 0.0, 1.0)
    p_pj_w = torch.clamp(mkt_w[:, 85:86].float() * 0.40 + 0.30, 0.0, 1.0)

    # Транспортный 2-тиковый зазор от встречного BBO при широком спреде (1-в-1 с execute_live_24slot_order3 в order3_live_bridge.py):
    flight_guard = torch.where((a1 - b1) >= 2.9 * tick_size, torch.full_like(a1, 2.0), torch.ones_like(a1)).unsqueeze(1)
    t_bid_px = torch.minimum(t_bid_px, torch.clamp(a1.unsqueeze(1) - flight_guard * tick_size, min=tick_size))
    inside_sp_b = t_bid_px > (b1.unsqueeze(1) + 0.1 * tick_size)
    at_bbo_b = (~inside_sp_b) & (t_bid_px >= (b1.unsqueeze(1) - 12.0 * tick_size))
    slot_usd_b = plc_b_qty * t_bid_px
    w_frac_b = torch.clamp((slot_usd_b - 35.0) / 215.0, 0.0, 1.0)
    eff_pj_b = p_pj_s + w_frac_b * (p_pj_w - p_pj_s)
    pj_mult = 1.0
    # Audit 1A/1B: 56.5% penny-jump at BBO AND 23.0% inside spread (<0 bp) + whale book-push!
    rival_qa_b = torch.where(
        at_bbo_b,
        pj_mult * torch.clamp(eff_pj_b - 0.22, min=0.0) * (18.0 + 0.85 * plc_b_qty * (0.35 + 0.65 * w_frac_b)),
        torch.zeros_like(t_vis_bq),
    )
    rival_in_sp_b = pj_mult * torch.clamp(eff_pj_b - 0.28, min=0.0) * (8.0 + 0.45 * plc_b_qty * (0.25 + 0.75 * w_frac_b))
    base_b_qa = torch.where(inside_sp_b, rival_in_sp_b, t_vis_bq + rival_qa_b)

    bid_px = torch.where(valid_b, t_bid_px, bid_px)
    bid_qty = torch.where(valid_b, plc_b_qty, bid_qty)
    bid_qa = torch.where(valid_b, base_b_qa, bid_qa)
    bid_qa0 = torch.where(valid_b, base_b_qa.to(dt), bid_qa0)
    bid_act_s = torch.where(valid_b, act_lat_2d.expand_as(bid_act_s), bid_act_s)

    fill_sync_lag = 1 if true_latency else 0
    fill_synced_mask = ((s_2d >= entry_s.unsqueeze(1) + fill_sync_lag) if fill_sync_lag > 0 else torch.ones_like(hp_now_mask))
    unreserved_pos = torch.clamp(pos_qty - ask_qty.sum(dim=1) - cnl_locked_a_qty, min=0.0).unsqueeze(1)
    empty_a_now = (ask_qty <= 1e-6) & ((~clear_a) if true_latency else torch.ones_like(clear_a))
    cand_plc_a = empty_a_now & hp_now_mask & fill_synced_mask & g_plc_a & (pi_ask > 1e-4) & (unreserved_pos >= 2.0)
    ord_a_idx = torch.argsort(torch.where(cand_plc_a, pi_ask, torch.full_like(pi_ask, -1.0)), dim=1, descending=True)
    inv_a_idx = torch.argsort(ord_a_idx, dim=1)
    cand_a_sorted = torch.gather(cand_plc_a.float(), 1, ord_a_idx)
    allow_a_tok_sorted = (cand_a_sorted.cumsum(dim=1) <= api_tokens.unsqueeze(1) + 1e-4) & (cand_a_sorted > 0.5)
    want_place_a = torch.gather(allow_a_tok_sorted, 1, inv_a_idx)

    pi_sel_a = torch.where(want_place_a, pi_ask + 1e-5, torch.zeros_like(pi_ask))
    sum_pi_sel_a = pi_sel_a.sum(dim=1, keepdim=True)
    full_share_a = torch.where(sum_pi_sel_a > 1e-6, pi_sel_a / torch.clamp(sum_pi_sel_a, min=1e-6), torch.zeros_like(pi_sel_a))
    raw_share_a = torch.where(want_place_a, torch.minimum(pi_ask, full_share_a), torch.zeros_like(pi_ask))
    share_sel_a = renorm_blend * full_share_a + (1.0 - renorm_blend) * raw_share_a
    plc_a_qty = torch.floor((unreserved_pos * share_sel_a) * 10000.0) / 10000.0
    valid_a = want_place_a & (plc_a_qty >= 2.0)
    plc_a_qty = torch.where(valid_a, plc_a_qty, torch.zeros_like(plc_a_qty))
    n_plc_a_step = valid_a.long().sum(dim=1)
    api_tokens = torch.clamp(api_tokens - valid_a.float().sum(dim=1), min=0.0)
    n_req_a_step = (clear_a & valid_a).long().sum(dim=1)

    t_ask_px = torch.maximum(t_ask_px, b1.unsqueeze(1) + flight_guard * tick_size)
    inside_sp_a = t_ask_px < (a1.unsqueeze(1) - 0.1 * tick_size)
    at_bbo_a = (~inside_sp_a) & (t_ask_px <= (a1.unsqueeze(1) + 12.0 * tick_size))
    slot_usd_a = plc_a_qty * t_ask_px
    w_frac_a = torch.clamp((slot_usd_a - 35.0) / 215.0, 0.0, 1.0)
    eff_pj_a = p_pj_s + w_frac_a * (p_pj_w - p_pj_s)
    rival_qa_a = torch.where(
        at_bbo_a,
        pj_mult * torch.clamp(eff_pj_a - 0.22, min=0.0) * (18.0 + 0.85 * plc_a_qty * (0.35 + 0.65 * w_frac_a)),
        torch.zeros_like(t_vis_aq),
    )
    rival_in_sp_a = pj_mult * torch.clamp(eff_pj_a - 0.28, min=0.0) * (8.0 + 0.45 * plc_a_qty * (0.25 + 0.75 * w_frac_a))
    base_a_qa = torch.where(inside_sp_a, rival_in_sp_a, t_vis_aq + rival_qa_a)

    ask_px = torch.where(valid_a, t_ask_px, ask_px)
    ask_qty = torch.where(valid_a, plc_a_qty, ask_qty)
    ask_qa = torch.where(valid_a, base_a_qa, ask_qa)
    ask_qa0 = torch.where(valid_a, base_a_qa.to(dt), ask_qa0)
    ask_act_s = torch.where(valid_a, act_lat_2d.expand_as(ask_act_s), ask_act_s)

    n_mut_tot = n_cnl_b_step + n_cnl_a_step + n_plc_b_step + n_plc_a_step
    if true_latency:
        burst_delay = ((n_mut_tot >= 2) | (api_tokens < 3.0)).unsqueeze(1)
        lat_med = 3
        lat_hi = 4
        extra_s = torch.where((n_mut_tot >= 5).unsqueeze(1) | (api_tokens < 1.5).unsqueeze(1), (next_s_2d + lat_hi).expand_as(bid_act_s), (next_s_2d + lat_med).expand_as(bid_act_s))
        bid_act_s = torch.where(valid_b & burst_delay, extra_s, bid_act_s)
        ask_act_s = torch.where(valid_a & burst_delay, extra_s, ask_act_s)

    # Непрерывный дрейф соперников вперёд нашей стоячей плиты у края стакана (penny-jump erosion):
    rest_near_b = (bid_qty > 1e-6) & (~valid_b) & (bid_px >= b1.unsqueeze(1) - 6.0 * tick_size)
    rest_near_a = (ask_qty > 1e-6) & (~valid_a) & (ask_px <= a1.unsqueeze(1) + 6.0 * tick_size)
    drip_rate = 0.75
    bid_qa = torch.where(rest_near_b, bid_qa + drip_rate * torch.clamp(p_pj_s - 0.25, min=0.0), bid_qa)
    ask_qa = torch.where(rest_near_a, ask_qa + drip_rate * torch.clamp(p_pj_s - 0.25, min=0.0), ask_qa)

    # 6. Калиброванное исполнение FIFO-очередей (единая физика попадания в сделку в calibrated и harsh_stress: 0 ложного уклонения от дампов и 0 случайного HODL!):
    nb1_1w = torch.clamp(nb1_w0, min=tick_size)
    na1_1w = torch.clamp(na1_w0, min=nb1_1w + tick_size)
    nb1 = nb1_1w.unsqueeze(1).expand(W, E).reshape(N, 1)
    na1 = na1_1w.unsqueeze(1).expand(W, E).reshape(N, 1)
    nb1_q = torch.clamp(nb1q_w0, min=0.0).unsqueeze(1).expand(W, E).reshape(N, 1)
    na1_q = torch.clamp(na1q_w0, min=0.0).unsqueeze(1).expand(W, E).reshape(N, 1)
    sv = sv_w0.unsqueeze(1).expand(W, E).reshape(N, 1)
    bv = bv_w0.unsqueeze(1).expand(W, E).reshape(N, 1)

    tril_s = torch.tril(torch.ones((S, S), dtype=torch.bool, device=bid_px.device), diagonal=-1).unsqueeze(0)

    act_b = (bid_qty > 1e-6) & (next_s_2d >= bid_act_s)
    b1_2d = b1.unsqueeze(1)
    a1_2d = a1.unsqueeze(1)
    b1_q_2d = b1_q.unsqueeze(1)
    a1_q_2d = a1_q.unsqueeze(1)

    # Закон сохранения массы стакана SafeTrade (отсечение 82.7% mid-spread self-wash объёма, где b1/b1_q и a1/a1_q не сдвинулись ни на йоту!):
    b1_cleared = nb1 < (b1_2d - 0.1 * tick_size)
    a1_cleared = na1 > (a1_2d + 0.1 * tick_size)
    same_b1 = torch.abs(nb1 - b1_2d) <= 0.1 * tick_size
    same_a1 = torch.abs(na1 - a1_2d) <= 0.1 * tick_size
    depl_b1 = torch.where(same_b1, torch.clamp(b1_q_2d - nb1_q, min=0.0), torch.zeros_like(nb1_q))
    depl_a1 = torch.where(same_a1, torch.clamp(a1_q_2d - na1_q, min=0.0), torch.zeros_like(na1_q))
    pure_wash_tick = same_b1 & same_a1 & (torch.abs(nb1_q - b1_q_2d) < 1e-3) & (torch.abs(na1_q - a1_q_2d) < 1e-3)

    in_sp_ord_b = bid_px > (b1_2d + 0.1 * tick_size)
    price_through_b = nb1 < (bid_px - 0.5 * tick_size)
    touch_only_b = (~price_through_b) & (nb1 <= bid_px + 1e-6)
    deep_cross_b = na1 <= (bid_px - 2.5 * tick_size)
    is_toxic_dump_b = b1_cleared | ((~pure_wash_tick) & (sv > 1.50 * bv + 15.0) & (depl_b1 > 5.0))

    in_sp_org_frac = 0.022
    depl_real_frac = 0.24
    touch_frac_fav = 0.18
    touch_frac_tox = 0.88
    cross_q_cap = 0.12
    qa_evap_tox_b = 0.42
    qa_comp_fav = 1.12

    # При неизменном b1 падение b1_q (depl_b1) частично вызвано отменами, а объём сделки СТРОГО ограничен сверху реальной лентой sv!
    # Живая калибровка по канарейке (26.11): ордер внутри спреда впереди b1 (in_sp_ord_b) забирает 62.7% реальных продаж!
    in_sp_bid_cap = 0.62
    clr_sv_cap = torch.minimum(sv, b1_q_2d + 0.35 * sv)
    org_sv_at_book = torch.where(b1_cleared, clr_sv_cap, torch.minimum(sv, depl_real_frac * depl_b1))
    org_sv_in_sp = torch.where(
        b1_cleared,
        clr_sv_cap,
        torch.where(pure_wash_tick, torch.zeros_like(sv), torch.minimum(sv, torch.maximum(depl_real_frac * depl_b1, in_sp_bid_cap * sv))),
    )
    org_sv_slot = torch.where(in_sp_ord_b, org_sv_in_sp, org_sv_at_book)

    hit_b = act_b & (deep_cross_b | ((price_through_b | touch_only_b) & (org_sv_slot > 1e-5)))
    eff_sv_slot_b = torch.where(
        in_sp_ord_b | (price_through_b & b1_cleared & is_toxic_dump_b),
        org_sv_slot,
        torch.where(is_toxic_dump_b, touch_frac_tox * org_sv_slot, touch_frac_fav * org_sv_slot),
    )
    tot_flow_b = torch.where(deep_cross_b, torch.maximum(eff_sv_slot_b, cross_q_cap * na1_q), eff_sv_slot_b)
    # Sweep-through (проверено прогоном 27k реальных ордеров 23–25 ноября): если реальные продажи увели книгу строго ниже нашей цены,
    # весь объём ленты sv прошёл через уровни от b1 до nb1:
    sweep_b = act_b & price_through_b & b1_cleared & (sv > 1e-5) & (nb1 < bid_px - 0.5 * tick_size)
    hit_b = hit_b | sweep_b
    tot_flow_b = torch.where(sweep_b, torch.maximum(tot_flow_b, sv), tot_flow_b)

    same_px_earlier_b = (
        (torch.abs(bid_px.unsqueeze(2) - bid_px.unsqueeze(1)) < 0.1 * tick_size)
        & act_b.unsqueeze(1)
        & tril_s
    ).any(dim=2)
    # Асимметрия спотового стакана: при токсичном сливе чужие биды снимаются (подставляя нас под нож), а на спокойном дне конкурируют за вход!
    adj_bid_qa = torch.where(is_toxic_dump_b, bid_qa * qa_evap_tox_b, bid_qa * qa_comp_fav)
    inc_qa_b = torch.where(same_px_earlier_b, torch.zeros_like(adj_bid_qa), adj_bid_qa)
    eff_qa_b = torch.where(hit_b, inc_qa_b, torch.zeros_like(bid_qa))
    eff_qty_b = torch.where(hit_b, bid_qty, torch.zeros_like(bid_qty))

    slot_need_b = eff_qa_b + eff_qty_b
    cum_need_b = torch.cumsum(slot_need_b, dim=1)
    flow_before_slot_b = cum_need_b - slot_need_b

    cum_bq_1w = torch.cat([torch.zeros((W, 1), dtype=torch.float32, device=bid_px.device), torch.cumsum(b_q10_raw_1w, dim=1)], dim=1)
    cnt_above_b = torch.searchsorted(-b_p10_1w, -(bid_px.view(W, E * S) + 0.5 * tick_size), right=False)
    mkt_above_b = torch.gather(cum_bq_1w, 1, cnt_above_b).view(N, S)
    rem_flow_at_slot_b = torch.clamp(tot_flow_b - torch.maximum(flow_before_slot_b, mkt_above_b), min=0.0)

    pre_fill_qp_b = torch.clamp(1.0 - bid_qa / (bid_qa0.float() + 10.0), 0.0, 1.0)
    qa_eaten_b = torch.minimum(bid_qa, rem_flow_at_slot_b)
    bid_qa = torch.where(hit_b, torch.clamp(bid_qa - qa_eaten_b, min=0.0), bid_qa)

    want_b = torch.minimum(eff_qty_b, torch.clamp(rem_flow_at_slot_b - eff_qa_b, min=0.0))
    # Точная бухгалтерия SafeTrade (Peatio v2); весь дополнительный стресс-фрикцион harsh_stress накапливается монотонно в prev_val_pred:
    near_opp_b = (next_s_2d == bid_act_s) & (bid_px >= a1.unsqueeze(1) - 2.1 * tick_size)
    slot_fee_b = torch.where(near_opp_b, torch.full_like(bid_px, 0.5 * (maker_fee + taker_fee)), torch.full_like(bid_px, maker_fee))
    unit_px_b = torch.clamp(bid_px, min=tick_size)
    want_b_q4 = torch.floor(want_b * 10000.0) / 10000.0
    want_cost_b = want_b_q4 * unit_px_b
    cum_cost_b = torch.cumsum(want_cost_b, dim=1)
    avail_usd_slot_b = torch.clamp(cash.unsqueeze(1) * 0.999 - (cum_cost_b - want_cost_b), min=0.0)
    f_q = torch.floor(torch.minimum(want_b_q4, avail_usd_slot_b / unit_px_b) * 10000.0) / 10000.0
    f_q_net = torch.floor(f_q * (1.0 - slot_fee_b) * 10000.0) / 10000.0

    has_slot_b = f_q_net > 1e-5
    tot_fq = f_q_net.sum(dim=1)
    tot_cost_raw = (f_q_net * bid_px).sum(dim=1)
    tot_cost = (f_q * unit_px_b).sum(dim=1)
    has_buy = tot_fq > 1e-5
    n_fills = torch.clamp(n_fills.long() + has_buy.long(), max=2000000).to(torch.int32)
    fill_qp_b_step = torch.where(has_buy, (f_q_net * pre_fill_qp_b).sum(dim=1) / torch.clamp(tot_fq, min=1e-6), torch.zeros_like(tot_fq))
    old_q = pos_qty
    pos_pre_buy = old_q
    new_q = torch.floor((old_q + tot_fq) * 10000.0) / 10000.0
    avg_fill_b_px = tot_cost_raw / torch.clamp(tot_fq, min=1e-6)
    first_buy = has_buy & (old_q <= 1e-6)
    no_low_yet = has_buy & (old_q > 1e-6) & (low_tranche_qty <= 1e-6)
    has_low_already = has_buy & (old_q > 1e-6) & (low_tranche_qty > 1e-6)

    init_dip = no_low_yet & (avg_fill_b_px <= avg_entry - 0.35 * tick_size)
    init_higher = no_low_yet & (~init_dip) & (old_q >= 1.95) & (avg_fill_b_px >= avg_entry + 0.35 * tick_size)
    deeper_dip = has_low_already & (avg_fill_b_px < min_buy_px - 0.35 * tick_size)
    same_low = has_low_already & (~deeper_dip) & (torch.abs(avg_fill_b_px - min_buy_px) <= 0.35 * tick_size)
    mid_dip = has_low_already & (~deeper_dip) & (~same_low) & (avg_fill_b_px <= avg_entry - 0.35 * tick_size)

    pushed_mid_q = mid_tranche_qty + low_tranche_qty
    pushed_mid_px = (mid_tranche_qty * mid_buy_px + low_tranche_qty * min_buy_px) / torch.clamp(pushed_mid_q, min=1e-6)
    added_mid_q = mid_tranche_qty + tot_fq
    added_mid_px = (mid_tranche_qty * mid_buy_px + tot_fq * avg_fill_b_px) / torch.clamp(added_mid_q, min=1e-6)

    new_mid_q = torch.where(first_buy, torch.zeros_like(mid_tranche_qty), torch.where(deeper_dip, pushed_mid_q, torch.where(mid_dip, added_mid_q, mid_tranche_qty)))
    new_mid_px = torch.where(first_buy, torch.zeros_like(mid_buy_px), torch.where(deeper_dip, pushed_mid_px, torch.where(mid_dip, added_mid_px, mid_buy_px)))
    same_low_q = low_tranche_qty + tot_fq
    same_low_px = (low_tranche_qty * min_buy_px + tot_fq * avg_fill_b_px) / torch.clamp(same_low_q, min=1e-6)

    low_tranche_qty = torch.where(first_buy, torch.zeros_like(low_tranche_qty), torch.where(init_dip | deeper_dip, tot_fq, torch.where(init_higher, old_q, torch.where(same_low, same_low_q, low_tranche_qty))))
    min_buy_px = torch.where(first_buy | init_dip | deeper_dip, avg_fill_b_px, torch.where(init_higher, avg_entry, torch.where(same_low, same_low_px, min_buy_px)))
    mid_tranche_qty = new_mid_q
    mid_buy_px = new_mid_px
    avg_entry = torch.where(has_buy, (old_q * avg_entry + tot_cost_raw) / torch.clamp(new_q, min=1e-6), avg_entry)
    pos_cost_usd = torch.where(has_buy, pos_cost_usd + tot_cost, pos_cost_usd)
    entry_s = torch.where(has_buy & (old_q <= 1e-6), s_nxt_i16.expand_as(entry_s), entry_s)
    cash = torch.where(has_buy, torch.clamp(cash - tot_cost, min=0.0), cash)
    pos_qty = torch.where(has_buy, new_q, pos_qty)

    rem_ord_b = torch.clamp(bid_qty - f_q, min=0.0)
    bid_qty = torch.where(has_slot_b, torch.where(rem_ord_b >= 2.0, rem_ord_b, torch.zeros_like(rem_ord_b)), bid_qty)
    # Ask может продавать ТОЛЬКО монеты pos_pre_buy, уже находившиеся на балансе до начала этой секунды (0 внутрисекундного wash-круга!):
    act_a = (ask_qty > 1e-6) & (next_s_2d >= ask_act_s) & (pos_pre_buy.unsqueeze(1) > 1e-6)
    in_sp_ord_a = ask_px < (a1_2d - 0.1 * tick_size)
    price_through_a = na1 > (ask_px + 0.5 * tick_size)
    touch_only_a = (~price_through_a) & (na1 >= ask_px - 1e-6)
    deep_cross_a = nb1 >= (ask_px + 2.5 * tick_size)
    is_fomo_rip_a = a1_cleared | ((~pure_wash_tick) & (bv > 1.50 * sv + 15.0) & (depl_a1 > 5.0))

    # На споте выход через Ask — это конкуренция с другими продавцами за реального покупателя (живая калибровка по канарейке 26.11):
    # возле высокой полки Ask1-1тик (sp_pos_a -> 1.0) без пробоя a1 выкупов почти нет (~50 мин ожидания), поток растёт только при спуске к середине/Bid1!
    sp_pos_a = torch.clamp((ask_px - b1_2d) / torch.clamp(a1_2d - b1_2d, min=tick_size), 0.0, 1.0)
    in_sp_ask_frac = 0.0012 + 0.36 * torch.pow(1.0 - sp_pos_a, 3.0)
    clr_bv_cap = torch.minimum(bv, 0.72 * a1_q_2d + 0.18 * bv)
    org_bv_at_book = torch.where(a1_cleared, clr_bv_cap, torch.minimum(bv, depl_real_frac * depl_a1))
    org_bv_in_sp = torch.where(
        a1_cleared,
        clr_bv_cap,
        torch.where(pure_wash_tick, torch.zeros_like(bv), torch.minimum(bv, torch.maximum(depl_real_frac * depl_a1, in_sp_ask_frac * bv))),
    )
    org_bv_slot = torch.where(in_sp_ord_a, org_bv_in_sp, org_bv_at_book)

    hit_a = act_a & (deep_cross_a | ((price_through_a | touch_only_a) & (org_bv_slot > 1e-5)))
    ask_fill_frac = 0.68
    eff_bv_slot_a = torch.where(
        price_through_a & a1_cleared,
        ask_fill_frac * org_bv_slot,
        touch_frac_fav * org_bv_slot,
    )
    tot_flow_a = torch.where(deep_cross_a, torch.maximum(eff_bv_slot_a, cross_q_cap * nb1_q), eff_bv_slot_a)
    sweep_a = act_a & price_through_a & a1_cleared & (bv > 1e-5) & (na1 > ask_px + 0.5 * tick_size)
    hit_a = hit_a | sweep_a
    tot_flow_a = torch.where(sweep_a, torch.maximum(tot_flow_a, bv), tot_flow_a)

    same_px_earlier_a = (
        (torch.abs(ask_px.unsqueeze(2) - ask_px.unsqueeze(1)) < 0.1 * tick_size)
        & act_a.unsqueeze(1)
        & tril_s
    ).any(dim=2)
    adj_ask_qa = ask_qa * qa_comp_fav
    inc_qa_a = torch.where(same_px_earlier_a, torch.zeros_like(adj_ask_qa), adj_ask_qa)
    eff_qa_a = torch.where(hit_a, inc_qa_a, torch.zeros_like(ask_qa))
    eff_qty_a = torch.where(hit_a, ask_qty, torch.zeros_like(ask_qty))

    slot_need_a = eff_qa_a + eff_qty_a
    cum_need_a = torch.cumsum(slot_need_a, dim=1)
    flow_before_slot_a = cum_need_a - slot_need_a

    cum_aq_1w = torch.cat([torch.zeros((W, 1), dtype=torch.float32, device=ask_px.device), torch.cumsum(a_q10_raw_1w, dim=1)], dim=1)
    cnt_below_a = torch.searchsorted(a_p10_1w, ask_px.view(W, E * S) - 0.5 * tick_size, right=False)
    mkt_below_a = torch.gather(cum_aq_1w, 1, cnt_below_a).view(N, S)
    rem_flow_at_slot_a = torch.clamp(tot_flow_a - torch.maximum(flow_before_slot_a, mkt_below_a), min=0.0)

    pre_fill_qp_a = torch.clamp(1.0 - ask_qa / (ask_qa0.float() + 10.0), 0.0, 1.0)
    qa_eaten_a = torch.minimum(ask_qa, rem_flow_at_slot_a)
    ask_qa = torch.where(hit_a, torch.clamp(ask_qa - qa_eaten_a, min=0.0), ask_qa)

    want_a = torch.floor(torch.minimum(eff_qty_a, torch.clamp(rem_flow_at_slot_a - eff_qa_a, min=0.0)) * 10000.0) / 10000.0
    cum_want_a = torch.cumsum(want_a, dim=1)
    f_qa = torch.floor(torch.minimum(want_a, torch.clamp(pos_pre_buy.unsqueeze(1) - (cum_want_a - want_a), min=0.0)) * 10000.0) / 10000.0

    has_slot_a = f_qa > 1e-5
    tot_fqa = f_qa.sum(dim=1)
    near_opp_a = (next_s_2d == ask_act_s) & (ask_px <= b1.unsqueeze(1) + 2.1 * tick_size)
    slot_fee_a = torch.where(near_opp_a, torch.full_like(ask_px, 0.5 * (maker_fee + taker_fee)), torch.full_like(ask_px, maker_fee))
    tot_rev = (f_qa * ask_px * (1.0 - slot_fee_a)).sum(dim=1)
    has_sell = tot_fqa > 1e-5
    mk_sell_px = (f_qa * ask_px).sum(dim=1) / torch.clamp(tot_fqa, min=1e-6)
    last_sell_px = torch.where(has_sell, mk_sell_px, last_sell_px)
    last_sell_s = torch.where(has_sell, s_nxt_i16.expand_as(last_sell_s), last_sell_s)
    n_fills = torch.clamp(n_fills.long() + has_sell.long(), max=2000000).to(torch.int32)
    fill_qp_a_step = torch.where(has_sell, (f_qa * pre_fill_qp_a).sum(dim=1) / torch.clamp(tot_fqa, min=1e-6), torch.zeros_like(tot_fqa))
    sum_fill_qprog = sum_fill_qprog + fill_qp_b_step + fill_qp_a_step

    lifo_f_qa = f_qa[:, :6].sum(dim=1)
    sold_low_q = torch.minimum(lifo_f_qa, low_tranche_qty)
    rem_lifo_qa = torch.clamp(lifo_f_qa - sold_low_q, min=0.0)
    sold_mid_q = torch.minimum(rem_lifo_qa, mid_tranche_qty)
    sold_high_q = torch.clamp(tot_fqa - sold_low_q - sold_mid_q, min=0.0)

    core_qty_pre = torch.clamp(pos_qty - low_tranche_qty - mid_tranche_qty, min=0.0)
    low_mid_cost_pre = (low_tranche_qty * min_buy_px + mid_tranche_qty * mid_buy_px) * (1.0 + maker_fee)
    core_cost_pre = torch.clamp(pos_cost_usd - low_mid_cost_pre, min=0.0)
    core_unit_cost = torch.where(core_qty_pre > 1e-5, core_cost_pre / torch.clamp(core_qty_pre, min=1e-6), avg_entry * (1.0 + maker_fee))
    sold_cost_usd = torch.minimum(
        pos_cost_usd,
        sold_low_q * min_buy_px * (1.0 + maker_fee) + sold_mid_q * mid_buy_px * (1.0 + maker_fee) + sold_high_q * core_unit_cost,
    )

    cash = torch.where(has_sell, cash + tot_rev, cash)
    rem_p = torch.clamp(pos_qty - tot_fqa, min=0.0)
    pos_qty = torch.where(has_sell, rem_p, pos_qty)
    pos_cost_usd = torch.where(has_sell, torch.where(rem_p > 1e-6, torch.clamp(pos_cost_usd - sold_cost_usd, min=0.0), torch.zeros_like(pos_cost_usd)), pos_cost_usd)
    avg_entry = torch.where(has_sell, torch.where(rem_p > 1e-6, pos_cost_usd / (rem_p * (1.0 + maker_fee)), torch.zeros_like(avg_entry)), avg_entry)

    rem_low_q = torch.where(has_sell, torch.where(rem_p > 1e-6, torch.clamp(low_tranche_qty - sold_low_q, min=0.0), torch.zeros_like(low_tranche_qty)), low_tranche_qty)
    rem_mid_q = torch.where(has_sell, torch.where(rem_p > 1e-6, torch.clamp(mid_tranche_qty - sold_mid_q, min=0.0), torch.zeros_like(mid_tranche_qty)), mid_tranche_qty)
    promote_mid = has_sell & (rem_p > 1e-6) & (rem_low_q < 1.95) & (rem_mid_q >= 1.95)
    prom_q = rem_low_q + rem_mid_q
    prom_px = (rem_low_q * min_buy_px + rem_mid_q * mid_buy_px) / torch.clamp(prom_q, min=1e-6)

    low_tranche_qty = torch.where(promote_mid, prom_q, rem_low_q)
    min_buy_px = torch.where(pos_qty > 1e-6, torch.where(promote_mid, prom_px, torch.where(low_tranche_qty > 1e-4, min_buy_px, avg_entry)), torch.zeros_like(min_buy_px))
    mid_tranche_qty = torch.where(promote_mid | (pos_qty <= 1e-6), torch.zeros_like(rem_mid_q), rem_mid_q)
    mid_buy_px = torch.where((pos_qty > 1e-6) & (mid_tranche_qty > 1e-4), mid_buy_px, torch.zeros_like(mid_buy_px))

    rem_ord_a = torch.clamp(ask_qty - f_qa, min=0.0)
    ask_qty = torch.where(has_slot_a, torch.where((rem_ord_a >= 2.0) & (pos_qty.unsqueeze(1) >= 2.0), torch.minimum(rem_ord_a, pos_qty.unsqueeze(1)), torch.zeros_like(rem_ord_a)), ask_qty)

    # 7. Честная траекторная просадка по реальному Bid-ликвидационному курсу nb1 + монотонный счётчик диссипативного трения (единый CUDA Graph без bfloat16-расхождения веток!):
    nb1_1d = nb1.squeeze(1)
    tox_b_1d = is_toxic_dump_b.squeeze(1).float()
    taker_usd_step = torch.where(exec_taker, proceeds, torch.zeros_like(proceeds)) + torch.where(exec_tb, tb_cost_raw, torch.zeros_like(tb_cost_raw))
    stress_step_usd = (
        0.50 * maker_fee * (tot_cost_raw + tot_rev)
        + 0.75 * tick_size * (tot_fq + tot_fqa)
        + 1.15 * maker_fee * tot_cost_raw * tox_b_1d
        + 0.45 * taker_fee * taker_usd_step
    )
    stress_acc_usd = prev_val_pred + stress_step_usd
    mtm_eq = torch.clamp(cash + pos_qty * nb1_1d * (1.0 - maker_fee), min=1e-2)
    peak_mtm_eq = torch.maximum(peak_mtm_eq, mtm_eq)
    log_dd_step = torch.clamp(torch.log(peak_mtm_eq / mtm_eq), min=0.0)
    sum_log_dd = sum_log_dd + log_dd_step
    dd_pct = 100.0 * torch.clamp(peak_mtm_eq - mtm_eq, min=0.0) / budget
    max_mtm_dd_pct = torch.maximum(max_mtm_dd_pct, dd_pct)

    na1_1d = na1.squeeze(1)
    near_b_usd = torch.where((bid_qty > 1e-6) & (bid_px >= nb1_1d.unsqueeze(1) * 0.9945), bid_qty * bid_px, torch.zeros_like(bid_qty)).sum(dim=1)
    near_a_q = torch.where((ask_qty > 1e-6) & (ask_px <= na1_1d.unsqueeze(1) * 1.0085), ask_qty, torch.zeros_like(ask_qty)).sum(dim=1)
    healthy_wave_pos = (pos_qty > 1e-6) & (avg_entry > 1e-4) & (nb1_1d >= avg_entry * 0.990)
    working_pos_usd = torch.where(healthy_wave_pos, pos_qty * nb1_1d, torch.minimum(pos_qty, near_a_q) * nb1_1d)
    raw_work_frac = torch.clamp((working_pos_usd + near_b_usd) / torch.clamp(mtm_eq, min=25.0), 0.0, 1.0)
    is_spike_danger = (mkt_w[:, 31] > 0.36) | (mkt_w[:, 30] > 0.44) | (mkt_w[:, 65] > 0.34) | (mkt_w[:, 80] > 0.38) | (mkt_w[:, 81] > 0.38)
    # L78 Smart Bear-Trend Cash Immunity: освобождает сидение в 100% USDT на ступенчатых даунтрендах (Wave Down / Lower Highs / BuyTrap+CVD<0)!
    is_macro_bear = (
        ((mkt_w[:, 61] < -0.16) & (mkt_w[:, 62] < -0.14))
        | ((mkt_w[:, 89] < -0.16) & (mkt_w[:, 90] > 0.10))
        | ((mkt_w[:, 86] > 0.30) & (mkt_w[:, 92] < -0.14))
    )
    is_danger_mkt = is_spike_danger | is_macro_bear
    sum_val_err = sum_val_err + torch.where(is_danger_mkt, torch.ones_like(raw_work_frac), raw_work_frac)
    # T0.2 & T0.8: Порежимная атрибуция dEq (UP/DN/FL), время с запасом в RISK-OFF и исполнения глубоких Bid при спокойном vs падающем BigONE:
    deq_step = mtm_eq - prev_mtm_eq
    is_up_30m = mkt_w[:, 88] > 0.445
    is_dn_30m = mkt_w[:, 88] < -0.445
    is_risk_off = is_dn_30m | (mkt_w[:, 81] > 0.285)
    is_deep_fill = has_buy & (avg_fill_b_px < b1 * 0.988)
    bo_dropped = mkt_w[:, 21] <= -1.25
    d_reg = torch.stack(
        [
            torch.where(is_up_30m, deq_step, torch.zeros_like(deq_step)),
            torch.where(is_dn_30m, deq_step, torch.zeros_like(deq_step)),
            torch.where((~is_up_30m) & (~is_dn_30m), deq_step, torch.zeros_like(deq_step)),
            torch.where(is_risk_off, torch.clamp((pos_qty * nb1_1d) / torch.clamp(budget, min=25.0), 0.0, 1.5), torch.zeros_like(deq_step)),
            (is_deep_fill & (~bo_dropped)).float(),
            (is_deep_fill & bo_dropped).float(),
        ],
        dim=-1,
    )
    reg_stats = reg_stats + d_reg
    prev_val_pred = stress_acc_usd
    prev_mtm_eq = mtm_eq

    n_req_step = n_req_b_step + n_req_a_step
    traded_usd_step = tot_cost_raw + tot_rev + torch.where(exec_tb, tb_cost_raw, torch.zeros_like(tb_cost_raw))
    sum_mut_load = sum_mut_load + traded_usd_step
    n_cancels = torch.clamp(n_cancels.long() + n_cnl_b_step + n_cnl_a_step, max=2000000).to(torch.int32)
    n_requeues = torch.clamp(n_requeues.long() + n_req_step, max=2000000).to(torch.int32)

    return (
        mem_h, gate_state, strat_state, cash, pos_qty, pos_cost_usd, avg_entry,
        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
        api_tokens, peak_mtm_eq, sum_log_dd, max_mtm_dd_pct, sum_mut_load,
        sum_fill_qprog, sum_val_err, prev_val_pred, prev_mtm_eq,
        sum_mode_p, sum_mode_ent,
        n_fills, n_cancels, n_requeues,
        last_sell_px, last_sell_s, reg_stats,
    )


fused_order3_clean_tick = torch.compile(_order3_clean_tick_step, dynamic=False)
_ORDER3_CUDAGRAPH_CACHE: dict[tuple, dict] = {}


def _get_or_build_order3_cudagraph(
    W: int, C: int, win_len: int, t_blk_sz: int, dt: torch.dtype, device: torch.device,
    tick_sz: float, maker_fee: float, taker_fee: float, true_latency: bool, harsh_stress: bool = False,
) -> dict:
    """Строит и кэширует аппаратный CUDA Graph одного тика симуляции с индексацией на GPU (0 Python overhead в цикле!)."""
    key = (W, C, win_len, t_blk_sz, dt, str(device), tick_sz, maker_fee, taker_fee, true_latency)
    if key in _ORDER3_CUDAGRAPH_CACHE:
        return _ORDER3_CUDAGRAPH_CACHE[key]

    B_bud, S = 6, 12
    E = C * B_bud
    N = W * E

    buf = {
        "s_idx": torch.zeros((1,), dtype=torch.long, device=device),
        "sn_idx": torch.ones((1,), dtype=torch.long, device=device),
        "u_idx": torch.zeros((1,), dtype=torch.long, device=device),
        "s_steps_i32": torch.arange(win_len, dtype=torch.int32, device=device),
        "pre_mkt_blk": torch.zeros((t_blk_sz, C, W, 384), dtype=dt, device=device),
        "z_mkt_w": torch.zeros((win_len - 1, W, MKT_DIM_ORDER3), dtype=dt, device=device),
        "b1_w_all": torch.ones((win_len, W), dtype=torch.float32, device=device),
        "a1_w_all": torch.full((win_len, W), 1.001, dtype=torch.float32, device=device),
        "b1q_w_all": torch.ones((win_len, W), dtype=torch.float32, device=device),
        "a1q_w_all": torch.ones((win_len, W), dtype=torch.float32, device=device),
        "rb_w_all": torch.ones((win_len, W), dtype=torch.float32, device=device),
        "rs_w_all": torch.ones((win_len, W), dtype=torch.float32, device=device),
        "b10_w_all": torch.ones((win_len, W, 10), dtype=torch.float32, device=device),
        "a10_w_all": torch.full((win_len, W, 10), 1.001, dtype=torch.float32, device=device),
        "bq10_w_all": torch.ones((win_len, W, 10), dtype=torch.float32, device=device),
        "aq10_w_all": torch.ones((win_len, W, 10), dtype=torch.float32, device=device),
        "sv_w_all": torch.zeros((win_len, W), dtype=torch.float32, device=device),
        "bv_w_all": torch.zeros((win_len, W), dtype=torch.float32, device=device),
        "w1_ag_c": torch.zeros((C, AG_DIM_ORDER3, 384), dtype=dt, device=device),
        "wr1_c": torch.zeros((C, 256, 128), dtype=dt, device=device),
        "br1_c": torch.zeros((C, 1, 128), dtype=dt, device=device),
        "wr2_c": torch.zeros((C, 64, 256), dtype=dt, device=device),
        "w2m_c": torch.zeros((C, 256, MOE_OUT_DIM_ORDER3), dtype=dt, device=device),
        "b2m_c": torch.zeros((C, 1, MOE_OUT_DIM_ORDER3), dtype=torch.float32, device=device),
        "wmup_c": torch.zeros((C, 256, MOE_MLP_UP_DIM_ORDER3), dtype=dt, device=device),
        "bmup_c": torch.zeros((C, 1, MOE_MLP_UP_DIM_ORDER3), dtype=dt, device=device),
        "wmdn_c": torch.zeros((C, K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=dt, device=device),
        "wg1_c": torch.zeros((C, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), dtype=dt, device=device),
        "bg1_c": torch.zeros((C, 1, ROUTER_H_DIM_ORDER3), dtype=dt, device=device),
        "wgres1_c": torch.zeros((C, ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3), dtype=dt, device=device),
        "bgres_c": torch.zeros((C, 1, 2 * ROUTER_RES_H_ORDER3), dtype=dt, device=device),
        "wgres2_c": torch.zeros((C, ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3), dtype=dt, device=device),
        "wg2_c": torch.zeros((C, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3), dtype=dt, device=device),
        "bg2_c": torch.zeros((C, 1, ROUTER_OUT_DIM_ORDER3), dtype=torch.float32, device=device),
        "wrec_c": torch.zeros((C, 256, 2 * MEM_DIM_ORDER3), dtype=dt, device=device),
        "brec_c": torch.zeros((C, 1, 2 * MEM_DIM_ORDER3), dtype=dt, device=device),
        "wmem2h_c": torch.zeros((C, MEM_DIM_ORDER3, 256), dtype=dt, device=device),
        "wreg1_c": torch.zeros((C, REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3), dtype=dt, device=device),
        "breg1b_c": torch.zeros((C, 1, REG_H_DIM_ORDER3), dtype=dt, device=device),
        "breg1a_c": torch.zeros((C, 1, REG_H_DIM_ORDER3), dtype=dt, device=device),
        "wreg_res1_c": torch.zeros((C, REG_H_DIM_ORDER3, 64), dtype=dt, device=device),
        "breg_res_c": torch.zeros((C, 1, 64), dtype=dt, device=device),
        "wreg_res2_c": torch.zeros((C, 32, REG_H_DIM_ORDER3), dtype=dt, device=device),
        "wreg2_c": torch.zeros((C, REG_H_DIM_ORDER3, 2), dtype=dt, device=device),
        "breg2b_c": torch.zeros((C, 1, 2), dtype=torch.float32, device=device),
        "breg2a_c": torch.zeros((C, 1, 2), dtype=torch.float32, device=device),
        "echelon_k01": torch.linspace(0.0, 1.0, S, dtype=torch.float32, device=device),
        "budget": torch.full((N,), 1000.0, dtype=torch.float32, device=device),
        "mem_h": torch.zeros((N, MEM_DIM_ORDER3), dtype=dt, device=device),
        "gate_state": torch.zeros((N, K_MODES_ORDER3), dtype=torch.float32, device=device),
        "strat_state": torch.zeros((N, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32, device=device),
        "cash": torch.full((N,), 1000.0, dtype=torch.float32, device=device),
        "pos_qty": torch.zeros(N, dtype=torch.float32, device=device),
        "pos_cost_usd": torch.zeros(N, dtype=torch.float32, device=device),
        "avg_entry": torch.zeros(N, dtype=torch.float32, device=device),
        "min_buy_px": torch.zeros(N, dtype=torch.float32, device=device),
        "low_tranche_qty": torch.zeros(N, dtype=torch.float32, device=device),
        "mid_buy_px": torch.zeros(N, dtype=torch.float32, device=device),
        "mid_tranche_qty": torch.zeros(N, dtype=torch.float32, device=device),
        "entry_s": torch.zeros(N, dtype=torch.int32, device=device),
        "bid_px": torch.zeros((N, S), dtype=torch.float32, device=device),
        "bid_qty": torch.zeros((N, S), dtype=torch.float32, device=device),
        "bid_qa": torch.zeros((N, S), dtype=torch.float32, device=device),
        "bid_qa0": torch.zeros((N, S), dtype=dt, device=device),
        "bid_act_s": torch.zeros((N, S), dtype=torch.int32, device=device),
        "ask_px": torch.zeros((N, S), dtype=torch.float32, device=device),
        "ask_qty": torch.zeros((N, S), dtype=torch.float32, device=device),
        "ask_qa": torch.zeros((N, S), dtype=torch.float32, device=device),
        "ask_qa0": torch.zeros((N, S), dtype=dt, device=device),
        "ask_act_s": torch.zeros((N, S), dtype=torch.int32, device=device),
        "api_tokens": torch.full((N,), TOKEN_BUCKET_CAP_ORDER3, dtype=torch.float32, device=device),
        "peak_mtm_eq": torch.full((N,), 1000.0, dtype=torch.float32, device=device),
        "sum_log_dd": torch.zeros(N, dtype=torch.float32, device=device),
        "max_dd_pct": torch.zeros(N, dtype=torch.float32, device=device),
        "sum_mut": torch.zeros(N, dtype=torch.float32, device=device),
        "sum_fqp": torch.zeros(N, dtype=torch.float32, device=device),
        "sum_val_err": torch.zeros(N, dtype=torch.float32, device=device),
        "prev_val_pred": torch.zeros(N, dtype=torch.float32, device=device),
        "prev_mtm_eq": torch.full((N,), 1000.0, dtype=torch.float32, device=device),
        "sum_mode_p": torch.zeros((N, K_MODES_ORDER3), dtype=torch.float32, device=device),
        "sum_mode_ent": torch.zeros(N, dtype=torch.float32, device=device),
        "n_fills": torch.zeros(N, dtype=torch.int32, device=device),
        "n_cancels": torch.zeros(N, dtype=torch.int32, device=device),
        "n_requeues": torch.zeros(N, dtype=torch.int32, device=device),
        "last_sell_px": torch.zeros(N, dtype=torch.float32, device=device),
        "last_sell_s": torch.full((N,), -9999, dtype=torch.int32, device=device),
        "reg_stats": torch.zeros((N, 6), dtype=torch.float32, device=device),
    }

    state_keys = [
        "mem_h", "gate_state", "strat_state", "cash", "pos_qty", "pos_cost_usd", "avg_entry",
        "min_buy_px", "low_tranche_qty", "mid_buy_px", "mid_tranche_qty", "entry_s",
        "bid_px", "bid_qty", "bid_qa", "bid_qa0", "bid_act_s",
        "ask_px", "ask_qty", "ask_qa", "ask_qa0", "ask_act_s",
        "api_tokens", "peak_mtm_eq", "sum_log_dd", "max_dd_pct", "sum_mut",
        "sum_fqp", "sum_val_err", "prev_val_pred", "prev_mtm_eq",
        "sum_mode_p", "sum_mode_ent",
        "n_fills", "n_cancels", "n_requeues",
        "last_sell_px", "last_sell_s", "reg_stats",
    ]

    def _run_step():
        si = buf["s_idx"]
        sni = buf["sn_idx"]
        ui = buf["u_idx"]
        outs = fused_order3_clean_tick(
            buf["s_steps_i32"].index_select(0, si).squeeze(0),
            buf["s_steps_i32"].index_select(0, sni).squeeze(0),
            buf["pre_mkt_blk"].index_select(0, ui).squeeze(0),
            buf["z_mkt_w"].index_select(0, si).squeeze(0),
            buf["b1_w_all"].index_select(0, si).squeeze(0),
            buf["a1_w_all"].index_select(0, si).squeeze(0),
            buf["b1q_w_all"].index_select(0, si).squeeze(0),
            buf["a1q_w_all"].index_select(0, si).squeeze(0),
            buf["rb_w_all"].index_select(0, si).squeeze(0),
            buf["rs_w_all"].index_select(0, si).squeeze(0),
            buf["b10_w_all"].index_select(0, si).squeeze(0),
            buf["a10_w_all"].index_select(0, si).squeeze(0),
            buf["bq10_w_all"].index_select(0, si).squeeze(0),
            buf["aq10_w_all"].index_select(0, si).squeeze(0),
            buf["b1_w_all"].index_select(0, sni).squeeze(0),
            buf["a1_w_all"].index_select(0, sni).squeeze(0),
            buf["b1q_w_all"].index_select(0, sni).squeeze(0),
            buf["a1q_w_all"].index_select(0, sni).squeeze(0),
            buf["sv_w_all"].index_select(0, sni).squeeze(0),
            buf["bv_w_all"].index_select(0, sni).squeeze(0),
            buf["w1_ag_c"], buf["wr1_c"], buf["br1_c"], buf["wr2_c"],
            buf["w2m_c"], buf["b2m_c"], buf["wmup_c"], buf["bmup_c"], buf["wmdn_c"],
            buf["wg1_c"], buf["bg1_c"], buf["wgres1_c"], buf["bgres_c"], buf["wgres2_c"], buf["wg2_c"], buf["bg2_c"],
            buf["wrec_c"], buf["brec_c"], buf["wmem2h_c"],
            buf["wreg1_c"], buf["breg1b_c"], buf["breg1a_c"],
            buf["wreg_res1_c"], buf["breg_res_c"], buf["wreg_res2_c"],
            buf["wreg2_c"], buf["breg2b_c"], buf["breg2a_c"],
            buf["echelon_k01"], buf["budget"],
            *(buf[k] for k in state_keys),
            tick_sz, maker_fee, taker_fee, true_latency, False,
        )
        for k, v in zip(state_keys, outs):
            buf[k].copy_(v)

    # 3 шага прогрева для завершения JIT-компиляции Triton до захвата CUDA Graph
    s_warm = torch.cuda.Stream(device=device)
    s_warm.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(s_warm):
        for _ in range(3):
            _run_step()
    torch.cuda.current_stream(device=device).wait_stream(s_warm)
    torch.cuda.synchronize(device)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s_warm):
        _run_step()
    torch.cuda.synchronize(device)

    entry = {"graph": g, "buf": buf, "state_keys": state_keys}
    _ORDER3_CUDAGRAPH_CACHE[key] = entry
    return entry




@torch.no_grad()
def evaluate_order3_basket(
    mkt_39d_all: torch.Tensor,
    b10_all: torch.Tensor,
    a10_all: torch.Tensor,
    bq10_all: torch.Tensor,
    aq10_all: torch.Tensor,
    sv_all: torch.Tensor,
    bv_all: torch.Tensor,
    cand_states: list[dict],
    val_windows: list[tuple[str, int, float, float]],
    win_len: int,
    tick_sz: float,
    maker_fee: float = 0.0010,
    taker_fee: float = 0.0020,
    device: torch.device = torch.device('cuda'),
    true_latency: bool = True,
    pad_to_max_c: bool = True,
    calib_jitter_std: float = 0.0,
    enforce_activity_floor: bool = False,
    harsh_stress: bool = False,
    budget_override: float | None = None,
    trace_out: list | None = None,
    risk_inv_pen_w: float = 0.0,
) -> list[dict]:
    # Монолитный батч до 192 групп (1,152 агента и 18,432 стакана за 1 проход на GPU!) благодаря потоковому pre_mkt_blk
    C_total = len(cand_states)
    max_chunk_c = 160
    if C_total > max_chunk_c:
        all_res = []
        for c0 in range(0, C_total, max_chunk_c):
            c_sub = cand_states[c0 : c0 + max_chunk_c]
            sub_res = evaluate_order3_basket(
                mkt_39d_all=mkt_39d_all,
                b10_all=b10_all,
                a10_all=a10_all,
                bq10_all=bq10_all,
                aq10_all=aq10_all,
                sv_all=sv_all,
                bv_all=bv_all,
                cand_states=c_sub,
                val_windows=val_windows,
                win_len=win_len,
                tick_sz=tick_sz,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                device=device,
                true_latency=true_latency,
                pad_to_max_c=pad_to_max_c,
                calib_jitter_std=calib_jitter_std,
                enforce_activity_floor=enforce_activity_floor,
                harsh_stress=harsh_stress,
                budget_override=budget_override,
                risk_inv_pen_w=risk_inv_pen_w,
            )
            all_res.extend(sub_res)
        return all_res

    # Авто-паддинг малых батчей (C=1 на старте, C=32 на Val) под единый CUDAGraph(160): 1 компиляция на весь запуск и 0s eager-простоя!
    orig_c = C_total
    if pad_to_max_c and device.type == "cuda" and 0 < C_total < max_chunk_c:
        target_c = max_chunk_c
        for k_cg in _ORDER3_CUDAGRAPH_CACHE:
            if k_cg[0] == len(val_windows) and k_cg[2] == win_len and k_cg[5] == str(device):
                target_c = k_cg[1]
                break
        if C_total < target_c:
            cand_states = list(cand_states) + [cand_states[0]] * (target_c - C_total)

    W = len(val_windows)
    C = len(cand_states)
    B_bud = 6
    E = C * B_bud
    N = W * E
    S = 12
    dt = mkt_39d_all.dtype

    echelon_k01 = torch.linspace(0.0, 1.0, 12, dtype=torch.float32, device=device)

    w1_l, b1_l, wr1_l, br1_l, wr2_l, w2m_l, b2m_l = [], [], [], [], [], [], []
    wmup_l, bmup_l, wmdn_l = [], [], []
    wg1_l, bg1_l, wgres1_l, bgres_l, wgres2_l, wg2_l, bg2_l = [], [], [], [], [], [], []
    wrec_l, brec_l, wmem2h_l = [], [], []
    wreg1_l, breg1b_l, breg1a_l, wregr1_l, bregr_l, wregr2_l, wreg2_l, breg2b_l, breg2a_l = [], [], [], [], [], [], [], [], []
    for st in cand_states:
        w1_t = st['w1'].to(device=device, dtype=torch.float32)
        if w1_t.shape[0] != TOTAL_DIM_ORDER3:
            old_mkt_d = w1_t.shape[0] - AG_DIM_ORDER3
            w1_pad = torch.zeros((TOTAL_DIM_ORDER3, 384), dtype=torch.float32, device=device)
            cm = min(MKT_DIM_ORDER3, max(0, old_mkt_d))
            w1_pad[:cm, :] = w1_t[:cm, :]
            w1_pad[MKT_DIM_ORDER3:, :] = w1_t[old_mkt_d:, :]
            w1_t = w1_pad
        w1_l.append(w1_t.to(dtype=dt))
        b1_l.append(st['b1'].to(device=device, dtype=dt).view(1, 384))
        wr1_l.append(st['w_res1'].to(device=device, dtype=dt))
        br1_l.append(st['b_res1'].to(device=device, dtype=dt).view(1, 128))
        wr2_l.append(st['w_res2'].to(device=device, dtype=dt))
        w2m_t = st['w2_moe'].to(device=device, dtype=torch.float32)
        b2m_t = st['b2_moe'].to(device=device, dtype=torch.float32)
        if b2m_t.numel() != MOE_OUT_DIM_ORDER3:
            old_exp_d = b2m_t.numel() // K_MODES_ORDER3
            w2m_old = w2m_t.view(256, K_MODES_ORDER3, old_exp_d)
            b2m_old = b2m_t.view(1, K_MODES_ORDER3, old_exp_d)
            w2m_pad = torch.zeros((256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32, device=device)
            b2m_pad = torch.zeros((1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32, device=device)
            c_ed = min(old_exp_d, EXPERT_OUT_DIM_ORDER3)
            w2m_pad[:, :, :c_ed] = w2m_old[:, :, :c_ed]
            b2m_pad[:, :, :c_ed] = b2m_old[:, :, :c_ed]
            if old_exp_d <= 45 and EXPERT_OUT_DIM_ORDER3 >= 48:
                b2m_pad[:, :, 47] = 6.0
            if old_exp_d <= 48 and EXPERT_OUT_DIM_ORDER3 >= 50:
                b2m_pad[:, :, 48:50] = -12.0
            if old_exp_d <= 50 and EXPERT_OUT_DIM_ORDER3 >= 52:
                b2m_pad[:, :, 50:52] = 12.0  # sigmoid(12)=1.0 -> 0.00e+00 паритет старых чекпоинтов L70!
            if old_exp_d <= 52 and EXPERT_OUT_DIM_ORDER3 >= 54:
                b2m_pad[:, :, 52:54] = -12.0  # sigmoid(-12)=0.0 -> 0.00e+00 паритет чекпоинтов L70/L71!
            if old_exp_d <= 54 and EXPERT_OUT_DIM_ORDER3 >= 55:
                b2m_pad[:, :, 54] = -12.0
            if old_exp_d <= 55 and EXPERT_OUT_DIM_ORDER3 >= 56:
                b2m_pad[:, :, 55] = -12.0
            if old_exp_d <= 56 and EXPERT_OUT_DIM_ORDER3 >= 58:
                b2m_pad[:, :, 56:58] = -12.0
            w2m_t = w2m_pad.reshape(256, MOE_OUT_DIM_ORDER3)
            b2m_t = b2m_pad.reshape(1, MOE_OUT_DIM_ORDER3)
        w2m_l.append(w2m_t.to(dtype=dt))
        b2m_l.append(b2m_t.view(1, MOE_OUT_DIM_ORDER3))
        if 'w_moe_up' in st and st['w_moe_up'].shape == (256, MOE_MLP_UP_DIM_ORDER3):
            wmup_l.append(st['w_moe_up'].to(device=device, dtype=dt))
            bmup_l.append(st['b_moe_up'].to(device=device, dtype=dt).view(1, MOE_MLP_UP_DIM_ORDER3))
            wmdn_l.append(st['w_moe_dn'].to(device=device, dtype=dt).view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3))
        else:
            wmup_l.append(torch.zeros((256, MOE_MLP_UP_DIM_ORDER3), dtype=dt, device=device))
            bmup_l.append(torch.zeros((1, MOE_MLP_UP_DIM_ORDER3), dtype=dt, device=device))
            wmdn_l.append(torch.zeros((K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=dt, device=device))

        wg1_t = st['w_gate1'].to(device=device, dtype=torch.float32)
        bg1_t = st['b_gate1'].to(device=device, dtype=torch.float32).view(1, -1)
        wg2_t = st['w_gate2'].to(device=device, dtype=torch.float32)
        bg2_t = st['b_gate2'].to(device=device, dtype=torch.float32).view(1, -1)
        if wg1_t.shape != (ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3):
            old_rin, old_rh = wg1_t.shape
            wg1_pad = torch.zeros((ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), dtype=torch.float32, device=device)
            bg1_pad = torch.zeros((1, ROUTER_H_DIM_ORDER3), dtype=torch.float32, device=device)
            ch_r = min(old_rh, ROUTER_H_DIM_ORDER3)
            bg1_pad[:, :ch_r] = bg1_t[:, :ch_r]
            if old_rin == 370 and ROUTER_IN_DIM_ORDER3 == 386:
                # 48D mem (idx 272..320) -> 64D mem (idx 272..336), extra+modes (idx 320..370) -> (idx 336..386)
                wg1_pad[:320, :ch_r] = wg1_t[:320, :ch_r]
                wg1_pad[336:386, :ch_r] = wg1_t[320:370, :ch_r]
            elif old_rin == 368 and ROUTER_IN_DIM_ORDER3 == 386:
                wg1_pad[:320, :ch_r] = wg1_t[:320, :ch_r]
                wg1_pad[336:376, :ch_r] = wg1_t[320:360, :ch_r]
                wg1_pad[378:386, :ch_r] = wg1_t[360:368, :ch_r]
            elif old_rin == 368 and ROUTER_IN_DIM_ORDER3 == 370:
                wg1_pad[:360, :ch_r] = wg1_t[:360, :ch_r]
                wg1_pad[362:370, :ch_r] = wg1_t[360:368, :ch_r]
            elif old_rin == 362 and ROUTER_IN_DIM_ORDER3 >= 368:
                wg1_pad[:320, :ch_r] = wg1_t[:320, :ch_r]
                wg1_pad[336:370, :ch_r] = wg1_t[320:354, :ch_r]
                wg1_pad[ROUTER_IN_DIM_ORDER3 - 8 : ROUTER_IN_DIM_ORDER3, :ch_r] = wg1_t[354:362, :ch_r]
            elif old_rin == 358 and ROUTER_IN_DIM_ORDER3 >= 362:
                wg1_pad[:320, :ch_r] = wg1_t[:320, :ch_r]
                wg1_pad[336:366, :ch_r] = wg1_t[320:350, :ch_r]
                wg1_pad[ROUTER_IN_DIM_ORDER3 - 8 : ROUTER_IN_DIM_ORDER3, :ch_r] = wg1_t[350:358, :ch_r]
            elif old_rin == 326 and ROUTER_IN_DIM_ORDER3 >= 358:
                wg1_pad[:304, :ch_r] = wg1_t[:304, :ch_r]
                wg1_pad[336:358, :ch_r] = wg1_t[304:326, :ch_r]
            elif old_rin == 295:
                wg1_pad[:288, :ch_r] = wg1_t[:288, :ch_r]
                wg1_pad[336:343, :ch_r] = wg1_t[288:295, :ch_r]
            else:
                cr = min(old_rin, ROUTER_IN_DIM_ORDER3)
                wg1_pad[:cr, :ch_r] = wg1_t[:cr, :ch_r]
            wg1_t, bg1_t = wg1_pad, bg1_pad
        if wg2_t.shape != (ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3):
            old_rh2, old_ro = wg2_t.shape
            wg2_pad = torch.zeros((ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3), dtype=torch.float32, device=device)
            bg2_pad = torch.zeros((1, ROUTER_OUT_DIM_ORDER3), dtype=torch.float32, device=device)
            wg2_pad[:min(old_rh2, ROUTER_H_DIM_ORDER3), :min(old_ro, ROUTER_OUT_DIM_ORDER3)] = wg2_t[:min(old_rh2, ROUTER_H_DIM_ORDER3), :min(old_ro, ROUTER_OUT_DIM_ORDER3)]
            bg2_pad[:, :min(old_ro, ROUTER_OUT_DIM_ORDER3)] = bg2_t[:, :min(old_ro, ROUTER_OUT_DIM_ORDER3)]
            if old_ro <= K_MODES_ORDER3:
                bg2_pad[:, K_MODES_ORDER3:] = 12.0  # beta_gate=1.0 -> 0.00e+00 паритет со старым безынерционным роутером!
            wg2_t, bg2_t = wg2_pad, bg2_pad
        wg1_l.append(wg1_t.to(dtype=dt))
        bg1_l.append(bg1_t.to(dtype=dt).view(1, ROUTER_H_DIM_ORDER3))
        if 'w_gate_res1' in st and st['w_gate_res1'].shape == (ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3):
            wgres1_l.append(st['w_gate_res1'].to(device=device, dtype=dt))
            bgres_l.append(st['b_gate_res'].to(device=device, dtype=dt).view(1, 2 * ROUTER_RES_H_ORDER3))
            wgres2_l.append(st['w_gate_res2'].to(device=device, dtype=dt))
        else:
            wgres1_l.append(torch.zeros((ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3), dtype=dt, device=device))
            bgres_l.append(torch.zeros((1, 2 * ROUTER_RES_H_ORDER3), dtype=dt, device=device))
            wgres2_l.append(torch.zeros((ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3), dtype=dt, device=device))
        wg2_l.append(wg2_t.to(dtype=dt))
        bg2_l.append(bg2_t.view(1, ROUTER_OUT_DIM_ORDER3))

        wrec_t = st['w_rec'].to(device=device, dtype=torch.float32)
        brec_t = st['b_rec'].to(device=device, dtype=torch.float32).view(1, -1)
        if brec_t.numel() != 2 * MEM_DIM_ORDER3:
            old_m = brec_t.numel() // 2
            wrec_pad = torch.zeros((256, 2 * MEM_DIM_ORDER3), dtype=torch.float32, device=device)
            brec_pad = torch.zeros((1, 2 * MEM_DIM_ORDER3), dtype=torch.float32, device=device)
            cm_r = min(old_m, MEM_DIM_ORDER3)
            wrec_pad[:, :cm_r] = wrec_t[:, :cm_r]
            wrec_pad[:, MEM_DIM_ORDER3 : MEM_DIM_ORDER3 + cm_r] = wrec_t[:, old_m : old_m + cm_r]
            brec_pad[:, :cm_r] = brec_t[:, :cm_r]
            brec_pad[:, MEM_DIM_ORDER3 : MEM_DIM_ORDER3 + cm_r] = brec_t[:, old_m : old_m + cm_r]
            wrec_t, brec_t = wrec_pad, brec_pad
        wrec_l.append(wrec_t.to(dtype=dt))
        brec_l.append(brec_t.to(dtype=dt).view(1, 2 * MEM_DIM_ORDER3))
        if 'w_mem2h' in st:
            wm2h_raw = st['w_mem2h'].to(device=device, dtype=torch.float32)
            if wm2h_raw.shape == (MEM_DIM_ORDER3, 256):
                wmem2h_l.append(wm2h_raw.to(dtype=dt))
            else:
                wm2h_pad = torch.zeros((MEM_DIM_ORDER3, 256), dtype=torch.float32, device=device)
                cm2 = min(wm2h_raw.shape[0], MEM_DIM_ORDER3)
                wm2h_pad[:cm2, :] = wm2h_raw[:cm2, :]
                wmem2h_l.append(wm2h_pad.to(dtype=dt))
        else:
            wmem2h_l.append(torch.zeros((MEM_DIM_ORDER3, 256), dtype=dt, device=device))

        wreg1_t = st['w_reg1'].to(device=device, dtype=torch.float32)
        breg1b_t = st['b_reg1_b'].to(device=device, dtype=torch.float32).view(1, -1)
        breg1a_t = st['b_reg1_a'].to(device=device, dtype=torch.float32).view(1, -1)
        wreg2_t = st['w_reg2'].to(device=device, dtype=torch.float32)
        if wreg1_t.shape != (REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3):
            old_ri, old_rh = wreg1_t.shape
            wreg1_pad = torch.zeros((REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3), dtype=torch.float32, device=device)
            breg1b_pad = torch.zeros((1, REG_H_DIM_ORDER3), dtype=torch.float32, device=device)
            breg1a_pad = torch.zeros((1, REG_H_DIM_ORDER3), dtype=torch.float32, device=device)
            wreg2_pad = torch.zeros((REG_H_DIM_ORDER3, 2), dtype=torch.float32, device=device)
            cr_h = min(old_rh, REG_H_DIM_ORDER3)
            if old_ri == 32 and REG_IN_DIM_ORDER3 == 36:
                wreg1_pad[:24, :cr_h] = wreg1_t[:24, :cr_h]
                wreg1_pad[28:36, :cr_h] = wreg1_t[24:32, :cr_h]
            else:
                cr_i = min(old_ri, REG_IN_DIM_ORDER3)
                wreg1_pad[:cr_i, :cr_h] = wreg1_t[:cr_i, :cr_h]
            breg1b_pad[:, :cr_h] = breg1b_t[:, :cr_h]
            breg1a_pad[:, :cr_h] = breg1a_t[:, :cr_h]
            wreg2_pad[:cr_h, :] = wreg2_t[:cr_h, :]
            wreg1_t, breg1b_t, breg1a_t, wreg2_t = wreg1_pad, breg1b_pad, breg1a_pad, wreg2_pad
        wreg1_l.append(wreg1_t.to(dtype=dt))
        breg1b_l.append(breg1b_t.to(dtype=dt).view(1, REG_H_DIM_ORDER3))
        breg1a_l.append(breg1a_t.to(dtype=dt).view(1, REG_H_DIM_ORDER3))
        if 'w_reg_res1' in st and st['w_reg_res1'].shape == (REG_H_DIM_ORDER3, 64):
            wregr1_l.append(st['w_reg_res1'].to(device=device, dtype=dt))
            bregr_l.append(st['b_reg_res'].to(device=device, dtype=dt).view(1, 64))
            wregr2_l.append(st['w_reg_res2'].to(device=device, dtype=dt))
        else:
            wregr1_l.append(torch.zeros((REG_H_DIM_ORDER3, 64), dtype=dt, device=device))
            bregr_l.append(torch.zeros((1, 64), dtype=dt, device=device))
            wregr2_l.append(torch.zeros((32, REG_H_DIM_ORDER3), dtype=dt, device=device))
        wreg2_l.append(wreg2_t.to(dtype=dt))
        breg2b_l.append(st['b_reg2_b'].to(device=device, dtype=torch.float32).view(1, 2))
        breg2a_l.append(st['b_reg2_a'].to(device=device, dtype=torch.float32).view(1, 2))

    w1_c = torch.stack(w1_l, dim=0).contiguous()
    w1_mkt_c = w1_c[:, :MKT_DIM_ORDER3, :].contiguous()
    w1_ag_c = w1_c[:, MKT_DIM_ORDER3:, :].contiguous()
    b1_c = torch.stack(b1_l, dim=0).contiguous()
    wr1_c = torch.stack(wr1_l, dim=0).contiguous()
    br1_c = torch.stack(br1_l, dim=0).contiguous()
    wr2_c = torch.stack(wr2_l, dim=0).contiguous()
    w2m_c = torch.stack(w2m_l, dim=0).contiguous()
    b2m_c = torch.stack(b2m_l, dim=0).contiguous()
    wmup_c = torch.stack(wmup_l, dim=0).contiguous()
    bmup_c = torch.stack(bmup_l, dim=0).contiguous()
    wmdn_c = torch.stack(wmdn_l, dim=0).contiguous()
    wg1_c = torch.stack(wg1_l, dim=0).contiguous()
    bg1_c = torch.stack(bg1_l, dim=0).contiguous()
    wgres1_c = torch.stack(wgres1_l, dim=0).contiguous()
    bgres_c = torch.stack(bgres_l, dim=0).contiguous()
    wgres2_c = torch.stack(wgres2_l, dim=0).contiguous()
    wg2_c = torch.stack(wg2_l, dim=0).contiguous()
    bg2_c = torch.stack(bg2_l, dim=0).contiguous()
    wrec_c = torch.stack(wrec_l, dim=0).contiguous()
    brec_c = torch.stack(brec_l, dim=0).contiguous()
    wmem2h_c = torch.stack(wmem2h_l, dim=0).contiguous()
    wreg1_c = torch.stack(wreg1_l, dim=0).contiguous()
    breg1b_c = torch.stack(breg1b_l, dim=0).contiguous()
    breg1a_c = torch.stack(breg1a_l, dim=0).contiguous()
    wreg_res1_c = torch.stack(wregr1_l, dim=0).contiguous()
    breg_res_c = torch.stack(bregr_l, dim=0).contiguous()
    wreg_res2_c = torch.stack(wregr2_l, dim=0).contiguous()
    wreg2_c = torch.stack(wreg2_l, dim=0).contiguous()
    breg2b_c = torch.stack(breg2b_l, dim=0).contiguous()
    breg2a_c = torch.stack(breg2a_l, dim=0).contiguous()

    w_starts = torch.tensor([ws for _, ws, _, _ in val_windows], dtype=torch.long, device=device)
    total_max_t = len(mkt_39d_all) - 1
    t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
    t_steps_cl = torch.clamp(t_steps, max=total_max_t)

    # Потоковый расчет z_mkt_w (pre_mkt считается блоками по 240 шагов: 10x экономия VRAM, 0 OOM на C=192!)
    raw_mkt_slice = mkt_39d_all[t_steps_cl[:win_len - 1]]
    if raw_mkt_slice.shape[-1] < MKT_DIM_ORDER3:
        pad_m = torch.zeros((*raw_mkt_slice.shape[:-1], MKT_DIM_ORDER3 - raw_mkt_slice.shape[-1]), dtype=raw_mkt_slice.dtype, device=device)
        raw_mkt_slice = torch.cat([raw_mkt_slice, pad_m], dim=-1)
    if calib_jitter_std > 0.0 and raw_mkt_slice.shape[-1] >= 56:
        # L70/L71 Anti-Spoof LOB Dropout (15% маскировка сырого BBO дисбаланса ch[1..4] на Train -> принуждает опираться на Order-1 / Wave Stack!)
        raw_mkt_slice = raw_mkt_slice.float().clone()
        lob_drop_mask = (torch.rand((raw_mkt_slice.shape[0], W, 1), device=device) > 0.15).float()
        lob_noise = torch.randn((raw_mkt_slice.shape[0], W, 4), device=device) * (1.25 * calib_jitter_std)
        raw_mkt_slice[..., 1:5] = torch.clamp(raw_mkt_slice[..., 1:5] * lob_drop_mask + lob_noise, -1.0, 1.0)
        cj_m = 0.30 * calib_jitter_std
        s_o2a = 1.0 + torch.randn((1, W, 5), device=device) * cj_m
        raw_mkt_slice[..., 29:34] = torch.clamp(raw_mkt_slice[..., 29:34] * s_o2a, 0.0, 1.0)
        s_o1 = 1.0 + torch.randn((1, W, 6), device=device) * cj_m
        raw_mkt_slice[..., 34:40] = torch.clamp(raw_mkt_slice[..., 34:40] * s_o1, -3.0, 3.0)
        if raw_mkt_slice.shape[-1] >= 84:
            s_o1e = 1.0 + torch.randn((1, W, 4), device=device) * cj_m
            raw_mkt_slice[..., 72:76] = torch.clamp(raw_mkt_slice[..., 72:76] * s_o1e, -3.0, 3.0)
            s_o2e = 1.0 + torch.randn((1, W, 5), device=device) * cj_m
            raw_mkt_slice[..., 79:84] = torch.clamp(raw_mkt_slice[..., 79:84] * s_o2e, 0.0, 1.0)
        if raw_mkt_slice.shape[-1] >= 88:
            s_o25 = 1.0 + torch.randn((1, W, 4), device=device) * cj_m
            raw_mkt_slice[..., 84:88] = torch.clamp(raw_mkt_slice[..., 84:88] * s_o25, 0.0, 1.0)
    z_mkt_w = _scale_mkt_44d(raw_mkt_slice, dt).contiguous()  # [win_len - 1, W, 88]

    b10_w_all = b10_all[t_steps_cl].contiguous()
    a10_w_all = a10_all[t_steps_cl].contiguous()
    bq10_w_all = bq10_all[t_steps_cl].contiguous()
    aq10_w_all = aq10_all[t_steps_cl].contiguous()
    b1_w_all = b10_w_all[:, :, 0].contiguous()
    a1_w_all = a10_w_all[:, :, 0].contiguous()
    b1q_w_all = bq10_w_all[:, :, 0].contiguous()
    a1q_w_all = aq10_w_all[:, :, 0].contiguous()

    # Честное каузальное 10-секундное скользящее среднее потока (1-в-1 с order3_live_bridge.py: sum(last_10s) / 10.0):
    k10 = torch.arange(-9, 1, dtype=torch.long, device=device).view(1, 1, 10)
    t_10s_idx = torch.clamp(t_steps_cl.unsqueeze(-1) + k10, min=0, max=total_max_t)
    rb_w_all = bv_all[t_10s_idx].mean(dim=-1).contiguous()
    rs_w_all = sv_all[t_10s_idx].mean(dim=-1).contiguous()
    sv_w_all = sv_all[t_steps_cl].contiguous()
    bv_w_all = bv_all[t_steps_cl].contiguous()

    budget_tiers = torch.tensor([1200.0, 1500.0, 1800.0, 2150.0, 2550.0, 3000.0], device=device)
    if budget_override is not None:
        # Диагностический режим сверки с боевым сервером: все 6 сред получают точный бюджет live-счёта.
        budget_tiers = torch.full((6,), float(budget_override), device=device)
    env_budgets_e = budget_tiers.unsqueeze(0).expand(C, B_bud).reshape(E)
    budget = env_budgets_e.unsqueeze(0).expand(W, E).reshape(N).contiguous()

    # Блок 1 L70/L71: Честное каузальное дополнение окна слева [ws - 1800 .. ws) для 100% разогрева 48D памяти и Liquid-вентилей!
    warm_stride = 10
    n_warm_steps = 180
    w_offs = torch.arange(-n_warm_steps * warm_stride, 0, warm_stride, dtype=torch.long, device=device).unsqueeze(1)
    t_warm_idx = torch.clamp(w_starts.unsqueeze(0) + w_offs, min=0, max=total_max_t)  # [180, W]
    raw_warm = mkt_39d_all[t_warm_idx]
    if raw_warm.shape[-1] < MKT_DIM_ORDER3:
        raw_warm = torch.cat([raw_warm, torch.zeros((*raw_warm.shape[:-1], MKT_DIM_ORDER3 - raw_warm.shape[-1]), dtype=raw_warm.dtype, device=device)], dim=-1)
    z_warm = _scale_mkt_44d(raw_warm, dt).contiguous()  # [180, W, 88]
    z_warm_in = z_warm.view(n_warm_steps * W, MKT_DIM_ORDER3).unsqueeze(0).expand(C, -1, -1)
    pre_warm = (torch.bmm(z_warm_in, w1_mkt_c) + b1_c).view(C, n_warm_steps, W, 384)
    mem_cw = torch.zeros((C, W, MEM_DIM_ORDER3), dtype=dt, device=device)
    for uw in range(n_warm_steps):
        pw_u = pre_warm[:, uw]  # [C, W, 384]
        h1_w = torch.relu(pw_u[:, :, :256]) * (2.0 * torch.sigmoid(pw_u[:, :, 256:384])).repeat_interleave(2, dim=-1)
        rp_w = torch.bmm(h1_w, wr1_c) + br1_c
        h2_w = h1_w + torch.bmm(torch.relu(rp_w[:, :, :64]) * (2.0 * torch.sigmoid(rp_w[:, :, 64:128])), wr2_c)
        rr_w = torch.bmm(h2_w, wrec_c) + brec_c
        ru_w = torch.tanh(rr_w[:, :, :MEM_DIM_ORDER3])
        rg_1s = torch.cat(
            [
                torch.sigmoid(rr_w[:, :, 64:72].float() + 1.0),
                torch.sigmoid(rr_w[:, :, 72:80].float() + 3.8),
                torch.sigmoid(rr_w[:, :, 80:88].float() + 5.2),
                torch.sigmoid(rr_w[:, :, 88:96].float() + 6.6),
                torch.sigmoid(rr_w[:, :, 96:104].float() + 2.2),
                torch.sigmoid(rr_w[:, :, 104:112].float() + 4.5),
                torch.sigmoid(rr_w[:, :, 112:120].float() + 7.5),
                torch.sigmoid(rr_w[:, :, 120:128].float() + 8.9),
            ],
            dim=-1,
        )
        rg_step = torch.pow(rg_1s, float(warm_stride)).to(dt)
        mem_cw = rg_step * mem_cw + (1.0 - rg_step) * ru_w

    # Равновесный прогрев начальных состояний gate_state и strat_state на последнем шаге перед ws:
    h2_w_ctx = h2_w + torch.bmm(mem_cw, wmem2h_c)
    zw_last = z_warm[-1]  # [W, 94]
    rex_w = torch.cat(
        [
            zw_last[:, 0:1], zw_last[:, 40:44], zw_last[:, 44:45], zw_last[:, 48:49],
            zw_last[:, 29:34], zw_last[:, 36:37], zw_last[:, 39:40], zw_last[:, 55:56], zw_last[:, 12:13],
            zw_last[:, 60:66],
            zw_last[:, 72:76], zw_last[:, 79:82], zw_last[:, 83:84],
            zw_last[:, 84:88],
            zw_last[:, 88:96],
        ],
        dim=-1,
    ).to(dt).unsqueeze(0).expand(C, W, ROUTER_EXTRA_DIM_ORDER3)
    z16_w = torch.zeros((C, W, 16), dtype=dt, device=device)
    p0_mode_w = torch.full((C, W, K_MODES_ORDER3), 1.0 / float(K_MODES_ORDER3), dtype=dt, device=device)
    rin_w = torch.cat([h2_w_ctx, z16_w, mem_cw, rex_w, p0_mode_w], dim=-1)
    rh1_w = F.silu(torch.bmm(rin_w, wg1_c) + bg1_c)
    rres_pre_w = torch.bmm(rh1_w, wgres1_c) + bgres_c
    rres_act_w = F.silu(rres_pre_w[:, :, :ROUTER_RES_H_ORDER3]) * (2.0 * torch.sigmoid(rres_pre_w[:, :, ROUTER_RES_H_ORDER3:]))
    rh_w = rh1_w + torch.bmm(rres_act_w, wgres2_c)
    go_w = (torch.bmm(rh_w, wg2_c).float() + bg2_c.float())[:, :, :K_MODES_ORDER3]
    gm_w = torch.softmax(go_w, dim=-1)
    moe_lin_w = (torch.bmm(h2_w_ctx, w2m_c).float() + b2m_c.float()).view(C, W, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    moe_up_w = (torch.bmm(h2_w_ctx, wmup_c) + bmup_c).view(C, W, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    moe_act_w = F.silu(moe_up_w[:, :, :, :MOE_MLP_H_ORDER3]) * (2.0 * torch.sigmoid(moe_up_w[:, :, :, MOE_MLP_H_ORDER3:]))
    moe_mlp_w = torch.matmul(moe_act_w.permute(0, 2, 1, 3), wmdn_c).permute(0, 2, 1, 3).float()
    moe_w = moe_lin_w + moe_mlp_w
    st_w = (gm_w.unsqueeze(-1) * moe_w).sum(dim=2)

    mem_h = mem_cw.permute(1, 0, 2).unsqueeze(2).expand(W, C, B_bud, MEM_DIM_ORDER3).reshape(N, MEM_DIM_ORDER3).contiguous()
    gate_state = go_w.permute(1, 0, 2).unsqueeze(2).expand(W, C, B_bud, K_MODES_ORDER3).reshape(N, K_MODES_ORDER3).contiguous()
    strat_state = st_w.permute(1, 0, 2).unsqueeze(2).expand(W, C, B_bud, EXPERT_OUT_DIM_ORDER3).reshape(N, EXPERT_OUT_DIM_ORDER3).contiguous()
    cash = budget.clone()
    pos_qty = torch.zeros(N, dtype=torch.float32, device=device)
    pos_cost_usd = torch.zeros(N, dtype=torch.float32, device=device)
    avg_entry = torch.zeros(N, dtype=torch.float32, device=device)
    min_buy_px = torch.zeros(N, dtype=torch.float32, device=device)
    low_tranche_qty = torch.zeros(N, dtype=torch.float32, device=device)
    mid_buy_px = torch.zeros(N, dtype=torch.float32, device=device)
    mid_tranche_qty = torch.zeros(N, dtype=torch.float32, device=device)
    entry_s = torch.zeros(N, dtype=torch.int32, device=device)

    for wi, (_, ws, rho_0, rel_off) in enumerate(val_windows):
        if rho_0 > 0.01:
            s0_w, s1_w = wi * E, (wi + 1) * E
            b1_0 = float(max(tick_sz, float(b10_all[int(ws), 0].item())))
            a1_0 = float(max(b1_0 + tick_sz, float(a10_all[int(ws), 0].item())))
            mid_0 = 0.5 * (b1_0 + a1_0)
            epx = max(tick_sz, round((mid_0 * (1.0 + rel_off)) / tick_sz) * tick_sz)
            iqty = torch.floor((env_budgets_e * float(rho_0) / mid_0) * 100.0) / 100.0
            v_m = iqty >= 2.0
            rcash = torch.clamp(env_budgets_e - iqty * mid_0 * (1.0 - maker_fee), min=0.0)
            cash[s0_w:s1_w] = torch.where(v_m, rcash, env_budgets_e)
            pos_qty[s0_w:s1_w] = torch.where(v_m, iqty, torch.zeros_like(iqty))
            avg_entry[s0_w:s1_w] = torch.where(v_m, torch.full_like(iqty, epx), torch.zeros_like(iqty))
            pos_cost_usd[s0_w:s1_w] = torch.where(v_m, iqty * epx * (1.0 + maker_fee), torch.zeros_like(iqty))
            min_buy_px[s0_w:s1_w] = torch.where(v_m, torch.full_like(iqty, epx), torch.zeros_like(iqty))
            low_tranche_qty[s0_w:s1_w] = torch.where(v_m, iqty * 0.5, torch.zeros_like(iqty))
            mid_buy_px[s0_w:s1_w] = torch.where(v_m, torch.full_like(iqty, epx), torch.zeros_like(iqty))
            mid_tranche_qty[s0_w:s1_w] = torch.where(v_m, iqty * 0.5, torch.zeros_like(iqty))
            entry_s[s0_w:s1_w] = -120

    bid_px = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qty = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qa = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qa0 = torch.zeros((N, S), dtype=dt, device=device)
    bid_act_s = torch.zeros((N, S), dtype=torch.int32, device=device)
    ask_px = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qty = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qa = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qa0 = torch.zeros((N, S), dtype=dt, device=device)
    ask_act_s = torch.zeros((N, S), dtype=torch.int32, device=device)
    api_tokens = torch.full((N,), TOKEN_BUCKET_CAP_ORDER3, dtype=torch.float32, device=device)

    init_b1 = torch.clamp(b1_w_all[0], min=tick_sz).unsqueeze(1).expand(W, E).reshape(N)
    init_a1 = torch.maximum(torch.clamp(a1_w_all[0], min=tick_sz).unsqueeze(1).expand(W, E).reshape(N), init_b1 + tick_sz)
    init_mid = 0.5 * (init_b1 + init_a1)
    init_cash_0 = cash.clone()
    init_pos_0 = pos_qty.clone()
    start_mtm_eq = torch.clamp(init_cash_0 + init_pos_0 * init_mid * (1.0 - maker_fee), min=1.0)
    peak_mtm_eq = start_mtm_eq.clone()
    prev_mtm_eq = start_mtm_eq.clone()
    prev_val_pred = torch.zeros(N, dtype=torch.float32, device=device)
    sum_log_dd = torch.zeros(N, dtype=torch.float32, device=device)
    max_dd_pct = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mut = torch.zeros(N, dtype=torch.float32, device=device)
    sum_fqp = torch.zeros(N, dtype=torch.float32, device=device)
    sum_val_err = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mode_p = torch.zeros((N, K_MODES_ORDER3), dtype=torch.float32, device=device)
    sum_mode_ent = torch.zeros(N, dtype=torch.float32, device=device)
    n_fills = torch.zeros(N, dtype=torch.int32, device=device)
    n_cancels = torch.zeros(N, dtype=torch.int32, device=device)
    n_requeues = torch.zeros(N, dtype=torch.int32, device=device)
    last_sell_px = torch.zeros(N, dtype=torch.float32, device=device)
    last_sell_s = torch.full((N,), -9999, dtype=torch.int32, device=device)
    reg_stats = torch.zeros((N, 6), dtype=torch.float32, device=device)

    s_steps_i32 = torch.arange(win_len, dtype=torch.int32, device=device)
    total_sim_steps = win_len - 1
    t_blk_sz = 240

    if device.type == "cuda" and (C >= 32 or not pad_to_max_c):
        cg_entry = _get_or_build_order3_cudagraph(
            W=W, C=C, win_len=win_len, t_blk_sz=t_blk_sz, dt=dt, device=device,
            tick_sz=tick_sz, maker_fee=maker_fee, taker_fee=taker_fee, true_latency=true_latency, harsh_stress=harsh_stress,
        )
        g_replay = cg_entry["graph"]
        cb = cg_entry["buf"]

        cb["z_mkt_w"].copy_(z_mkt_w)
        cb["b1_w_all"].copy_(b1_w_all)
        cb["a1_w_all"].copy_(a1_w_all)
        cb["b1q_w_all"].copy_(b1q_w_all)
        cb["a1q_w_all"].copy_(a1q_w_all)
        cb["rb_w_all"].copy_(rb_w_all)
        cb["rs_w_all"].copy_(rs_w_all)
        cb["b10_w_all"].copy_(b10_w_all)
        cb["a10_w_all"].copy_(a10_w_all)
        cb["bq10_w_all"].copy_(bq10_w_all)
        cb["aq10_w_all"].copy_(aq10_w_all)
        cb["sv_w_all"].copy_(sv_w_all)
        cb["bv_w_all"].copy_(bv_w_all)

        cb["w1_ag_c"].copy_(w1_ag_c)
        cb["wr1_c"].copy_(wr1_c); cb["br1_c"].copy_(br1_c); cb["wr2_c"].copy_(wr2_c)
        cb["w2m_c"].copy_(w2m_c); cb["b2m_c"].copy_(b2m_c)
        cb["wmup_c"].copy_(wmup_c); cb["bmup_c"].copy_(bmup_c); cb["wmdn_c"].copy_(wmdn_c)
        cb["wg1_c"].copy_(wg1_c); cb["bg1_c"].copy_(bg1_c)
        cb["wgres1_c"].copy_(wgres1_c); cb["bgres_c"].copy_(bgres_c); cb["wgres2_c"].copy_(wgres2_c)
        cb["wg2_c"].copy_(wg2_c); cb["bg2_c"].copy_(bg2_c)
        cb["wrec_c"].copy_(wrec_c); cb["brec_c"].copy_(brec_c); cb["wmem2h_c"].copy_(wmem2h_c)
        cb["wreg1_c"].copy_(wreg1_c); cb["breg1b_c"].copy_(breg1b_c); cb["breg1a_c"].copy_(breg1a_c)
        cb["wreg_res1_c"].copy_(wreg_res1_c); cb["breg_res_c"].copy_(breg_res_c); cb["wreg_res2_c"].copy_(wreg_res2_c)
        cb["wreg2_c"].copy_(wreg2_c); cb["breg2b_c"].copy_(breg2b_c); cb["breg2a_c"].copy_(breg2a_c)
        cb["budget"].copy_(budget)

        init_states = [
            mem_h, gate_state, strat_state, cash, pos_qty, pos_cost_usd, avg_entry,
            min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
            bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
            ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
            api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
            sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
            sum_mode_p, sum_mode_ent,
            n_fills, n_cancels, n_requeues,
            last_sell_px, last_sell_s, reg_stats,
        ]
        for k_s, t_s in zip(cg_entry["state_keys"], init_states):
            cb[k_s].copy_(t_s)

        si_buf, sni_buf, ui_buf = cb["s_idx"], cb["sn_idx"], cb["u_idx"]
        # Диагностическая трасса сверки с live: 4 маленьких device-to-device копии на шаг, без синхронизации с CPU (CUDA Graph не ломается).
        tr_pos = tr_cash = tr_nf = tr_nc = None
        if trace_out is not None:
            tr_pos = torch.zeros((total_sim_steps, N), dtype=torch.float32, device=device)
            tr_cash = torch.zeros((total_sim_steps, N), dtype=torch.float32, device=device)
            tr_nf = torch.zeros((total_sim_steps, N), dtype=torch.int32, device=device)
            tr_nc = torch.zeros((total_sim_steps, N), dtype=torch.int32, device=device)
            tr_slots = torch.zeros((total_sim_steps, 4, N, 12), dtype=torch.float32, device=device)
            tr_pol = torch.zeros((total_sim_steps, N, 2 + K_MODES_ORDER3), dtype=torch.float32, device=device)
        for s_blk0 in range(0, total_sim_steps, t_blk_sz):
            s_blk1 = min(total_sim_steps, s_blk0 + t_blk_sz)
            blk_len = s_blk1 - s_blk0
            z_blk_in = z_mkt_w[s_blk0:s_blk1].view(blk_len * W, MKT_DIM_ORDER3).unsqueeze(0).expand(C, -1, -1)
            pre_mkt_blk = (torch.bmm(z_blk_in, w1_mkt_c) + b1_c).view(C, blk_len, W, 384).permute(1, 0, 2, 3)
            cb["pre_mkt_blk"][:blk_len].copy_(pre_mkt_blk)
            for u in range(blk_len):
                s = s_blk0 + u
                ui_buf.fill_(u)
                si_buf.fill_(s)
                sni_buf.fill_(s + 1)
                g_replay.replay()
                if tr_pos is not None:
                    tr_pos[s].copy_(cb["pos_qty"])
                    tr_cash[s].copy_(cb["cash"])
                    tr_nf[s].copy_(cb["n_fills"])
                    tr_nc[s].copy_(cb["n_cancels"])
                    tr_slots[s, 0].copy_(cb["bid_px"])
                    tr_slots[s, 1].copy_(cb["bid_qty"])
                    tr_slots[s, 2].copy_(cb["ask_px"])
                    tr_slots[s, 3].copy_(cb["ask_qty"])
                    tr_pol[s, :, 0].copy_(torch.sigmoid(cb["strat_state"][:, 0].float()))
                    tr_pol[s, :, 1].copy_(cb["api_tokens"])
                    tr_pol[s, :, 2:].copy_(torch.softmax(cb["gate_state"][:, :K_MODES_ORDER3].float(), dim=-1))
        if tr_pos is not None:
            p_np, c_np, f_np, k_np = tr_pos.cpu().numpy(), tr_cash.cpu().numpy(), tr_nf.cpu().numpy(), tr_nc.cpu().numpy()
            sl_np = tr_slots.cpu().numpy()
            pol_np = tr_pol.cpu().numpy()
            trace_out.extend((s + 1, p_np[s], c_np[s], f_np[s], k_np[s], sl_np[s], pol_np[s]) for s in range(total_sim_steps))

        (
            mem_h, gate_state, strat_state, cash, pos_qty, pos_cost_usd, avg_entry,
            min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
            bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
            ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
            api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
            sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
            sum_mode_p, sum_mode_ent,
            n_fills, n_cancels, n_requeues,
            last_sell_px, last_sell_s, reg_stats,
        ) = [cb[k_s] for k_s in cg_entry["state_keys"]]
    else:
        for s_blk0 in range(0, total_sim_steps, t_blk_sz):
            s_blk1 = min(total_sim_steps, s_blk0 + t_blk_sz)
            blk_len = s_blk1 - s_blk0
            z_blk_in = z_mkt_w[s_blk0:s_blk1].view(blk_len * W, MKT_DIM_ORDER3).unsqueeze(0).expand(C, -1, -1)
            pre_mkt_blk = (torch.bmm(z_blk_in, w1_mkt_c) + b1_c).view(C, blk_len, W, 384).permute(1, 0, 2, 3).contiguous()
            for u in range(blk_len):
                s = s_blk0 + u
                (
                    mem_h, gate_state, strat_state, cash, pos_qty, pos_cost_usd, avg_entry,
                    min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                    bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                    ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                    api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
                    sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
                    sum_mode_p, sum_mode_ent,
                    n_fills, n_cancels, n_requeues,
                    last_sell_px, last_sell_s, reg_stats,
                ) = _order3_clean_tick_step(
                    s_steps_i32[s], s_steps_i32[s + 1], pre_mkt_blk[u], z_mkt_w[s],
                    b1_w_all[s], a1_w_all[s], b1q_w_all[s], a1q_w_all[s],
                    rb_w_all[s], rs_w_all[s],
                    b10_w_all[s], a10_w_all[s], bq10_w_all[s], aq10_w_all[s],
                    b1_w_all[s + 1], a1_w_all[s + 1], b1q_w_all[s + 1], a1q_w_all[s + 1], sv_w_all[s + 1], bv_w_all[s + 1],
                    w1_ag_c, wr1_c, br1_c, wr2_c,
                    w2m_c, b2m_c, wmup_c, bmup_c, wmdn_c,
                    wg1_c, bg1_c, wgres1_c, bgres_c, wgres2_c, wg2_c, bg2_c,
                    wrec_c, brec_c, wmem2h_c,
                    wreg1_c, breg1b_c, breg1a_c, wreg_res1_c, breg_res_c, wreg_res2_c, wreg2_c, breg2b_c, breg2a_c,
                    echelon_k01, budget,
                    mem_h, gate_state, strat_state, cash, pos_qty, pos_cost_usd, avg_entry,
                    min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                    bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                    ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                    api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
                    sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
                    sum_mode_p, sum_mode_ent,
                    n_fills, n_cancels, n_requeues,
                    last_sell_px, last_sell_s, reg_stats,
                    tick_sz, maker_fee, taker_fee, true_latency, harsh_stress,
                )
                if trace_out is not None:
                    trace_out.append((s + 1, pos_qty.detach().float().cpu().numpy().copy(), cash.detach().float().cpu().numpy().copy(), n_fills.detach().cpu().numpy().copy(), n_cancels.detach().cpu().numpy().copy()))

    # 100% честная терминальная ликвидация по реальной 10-уровневой книге покупателей end_b10/end_bq10:
    # На коротких тренировочных окнах (enforce_activity_floor=True) рабочий ММ-инвентарь (<=35% бюджета, не в глубоком пробое)
    # оценивается по ожидаемому лимитному выходу (mid * (1 - 1.2*maker_fee)), снимая искусственный штраф "казни по таймеру",
    # который загонял exp_bid_frac -> 0.0001! Весь избыточный/аварийный объем и все слепые OOS-тесты идут строго по L2 end_b10!
    end_b10 = torch.clamp(b10_w_all[win_len - 1], min=tick_sz).unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    end_a1 = torch.clamp(a1_w_all[win_len - 1], min=tick_sz).unsqueeze(1).expand(W, E).reshape(N)
    end_b1 = end_b10[:, 0]
    end_mid = 0.5 * (end_b1 + torch.maximum(end_a1, end_b1 + tick_sz))
    end_bq10 = torch.clamp(bq10_w_all[win_len - 1], min=0.0).unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    cum_end_bq = torch.cumsum(end_bq10, dim=1)
    prev_end_bq = cum_end_bq - end_bq10
    liq_10_q = torch.clamp(torch.minimum(pos_qty.unsqueeze(1) - prev_end_bq, end_bq10), min=0.0)
    liq_10_tot = liq_10_q.sum(dim=1)
    excess_end_q = torch.clamp(pos_qty - liq_10_tot, min=0.0)
    tail_px = torch.clamp(end_b10[:, -1] * 0.985, min=tick_sz)
    strict_l2_carry = (liq_10_q * end_b10).sum(dim=1) * (1.0 - maker_fee) + excess_end_q * tail_px * (1.0 - taker_fee)
    raw_max_dd_pct = max_dd_pct
    rem = pos_qty > 1e-6

    n_steps_f = float(win_len - 1)
    # Точный побитовый эталон пассивного портфеля (L2 ликвидация + точная траекторная просадка по всем шагам s=1..win_len-1 -> 0 сделок === 0.0000 bp!):
    init_liq_10_q = torch.clamp(torch.minimum(init_pos_0.unsqueeze(1) - prev_end_bq, end_bq10), min=0.0)
    init_exc_q = torch.clamp(init_pos_0 - init_liq_10_q.sum(dim=1), min=0.0)
    passive_carry_val = (init_liq_10_q * end_b10).sum(dim=1) * (1.0 - maker_fee) + init_exc_q * tail_px * (1.0 - taker_fee)
    passive_end_eq = torch.clamp(init_cash_0 + passive_carry_val, min=1.0)
    ref_eq = torch.where(init_pos_0 > 1e-6, passive_end_eq, start_mtm_eq)

    # Пошаговая траекторная просадка пассивного портфеля имеет размер всего [win_len-1, W, B_bud] (~4 МБ VRAM, 0 нагрузки на C!):
    ic_wb = init_cash_0.view(W, C, B_bud)[:, 0, :]
    ip_wb = init_pos_0.view(W, C, B_bud)[:, 0, :]
    se_wb = start_mtm_eq.view(W, C, B_bud)[:, 0, :]
    b1_seq_tw = torch.clamp(b1_w_all[1:win_len], min=tick_sz).unsqueeze(2)  # [T, W, 1]
    p_eq_twb = torch.clamp(ic_wb.unsqueeze(0) + ip_wb.unsqueeze(0) * b1_seq_tw * (1.0 - maker_fee), min=1e-2)
    p_peak_twb = torch.maximum(se_wb.unsqueeze(0), torch.cummax(p_eq_twb, dim=0).values)
    pass_mld_wb = torch.clamp(torch.log(p_peak_twb / p_eq_twb), min=0.0).mean(dim=0)  # [W, B_bud]
    pass_mld_we = pass_mld_wb.unsqueeze(1).expand(W, C, B_bud).reshape(W, E)

    raw_mean_log_dd = (sum_log_dd / n_steps_f).view(W, E)
    rel_mean_log_dd = torch.clamp(raw_mean_log_dd - pass_mld_we, min=-0.0025)
    w_fills_mat = n_fills.float().view(W, E)
    w_cancels_mat = n_cancels.float().view(W, E)
    w_requeues_mat = n_requeues.float().view(W, E)
    w_fqp_mat = sum_fqp.view(W, E)
    mode_p_mat = (sum_mode_p / n_steps_f).view(W, E, K_MODES_ORDER3)
    mode_ent_mat = (sum_mode_ent / n_steps_f).view(W, E)

    clean_idx = [wi for wi, (_, _, r0, _) in enumerate(val_windows) if r0 <= 0.01]
    inv_idx = [wi for wi, (_, _, r0, _) in enumerate(val_windows) if r0 > 0.01]

    beta_risk = 0.06
    ln2 = math.log(2.0)
    M_wb = W * B_bud
    fills_wcb = w_fills_mat.view(W, C, B_bud)
    fills_v = fills_wcb.sum(dim=0).mean(dim=1)
    cl_fills_v = fills_wcb[clean_idx].sum(dim=0).mean(dim=1) if clean_idx else fills_v
    cl_fills_pw = cl_fills_v / max(1.0, float(len(clean_idx)))
    turn_wcb = (sum_mut / torch.clamp(budget, min=100.0)).view(W, C, B_bud)
    cl_turn_v = turn_wcb[clean_idx].mean(dim=(0, 2)) if clean_idx else turn_wcb.mean(dim=(0, 2))
    cancels_v = w_cancels_mat.view(W, C, B_bud).sum(dim=0).mean(dim=1)
    requeues_v = w_requeues_mat.view(W, C, B_bud).sum(dim=0).mean(dim=1)
    fqp_v = w_fqp_mat.view(W, C, B_bud).sum(dim=0).mean(dim=1)

    mp_bar_mat = mode_p_mat.view(W, C, B_bud, K_MODES_ORDER3).mean(dim=(0, 2))  # [C, K]
    h_marg_v = -(mp_bar_mat * torch.log(torch.clamp(mp_bar_mat, min=1e-6))).sum(dim=1)
    h_cond_v = mode_ent_mat.view(W, C, B_bud).mean(dim=(0, 2))
    mi_bits_v = torch.clamp((h_marg_v - h_cond_v) / ln2, min=0.0)

    work_wcb = (sum_val_err / n_steps_f).view(W, C, B_bud)
    cl_work_v = work_wcb[clean_idx].mean(dim=(0, 2)) if clean_idx else work_wcb.mean(dim=(0, 2))
    win_hrs = max(0.25, float(win_len - 1) / 3600.0)
    cl_fills_phr = cl_fills_pw / win_hrs
    cl_turn_phr = cl_turn_v / win_hrs

    if enforce_activity_floor:
        work_cap_q = (0.96 * budget) / torch.clamp(end_mid, min=tick_sz)
        healthy_mm = (init_pos_0 <= 1e-6) & (avg_entry > 1e-4) & (end_mid >= avg_entry * 0.985)
        maker_q = torch.where(healthy_mm, torch.minimum(pos_qty, work_cap_q), torch.zeros_like(pos_qty))
        rem_l2_q = torch.clamp(pos_qty - maker_q, min=0.0)
        rem_liq_10 = torch.clamp(torch.minimum(rem_l2_q.unsqueeze(1) - prev_end_bq, end_bq10), min=0.0)
        rem_exc = torch.clamp(rem_l2_q - rem_liq_10.sum(dim=1), min=0.0)
        cost_neutral_px = torch.minimum(end_mid * (1.0 - 1.2 * maker_fee), avg_entry * (1.0 + maker_fee))
        calib_carry_val = (
            maker_q * cost_neutral_px
            + (rem_liq_10 * end_b10).sum(dim=1) * (1.0 - maker_fee)
            + rem_exc * tail_px * (1.0 - taker_fee)
        )
    else:
        calib_carry_val = strict_l2_carry

    harsh_carry_val = torch.where(
        init_pos_0 <= 1e-6,
        torch.minimum(strict_l2_carry * (1.0 - taker_fee), pos_qty * avg_entry * (1.0 - maker_fee * 1.65)),
        strict_l2_carry * (1.0 - taker_fee),
    )

    def _reduce_terminal_mode(is_h: bool):
        c_val = harsh_carry_val if is_h else calib_carry_val
        s_ded = prev_val_pred if is_h else torch.zeros_like(prev_val_pred)
        e_cash = torch.clamp(torch.where(rem, cash + c_val, cash) - s_ded, min=1.0)
        e_dd_pct = 100.0 * torch.clamp(peak_mtm_eq - e_cash, min=0.0) / budget
        m_dd_pct = torch.maximum(raw_max_dd_pct + (100.0 * s_ded / budget), e_dd_pct)

        dl_eq = torch.log(e_cash / ref_eq).view(W, E)
        tg_bp = 10000.0 * (dl_eq - 0.30 * rel_mean_log_dd)
        md_pct = (100.0 * (e_cash - ref_eq) / budget).view(W, E)
        wd_mat = m_dd_pct.view(W, E)

        gm = tg_bp.view(W, C, B_bud).permute(1, 0, 2).reshape(C, M_wb)
        ent_v = (-torch.logsumexp(-beta_risk * gm, dim=1) + math.log(float(M_wb))) / beta_risk
        qm = torch.quantile(gm, torch.tensor([0.15, 0.35, 0.50, 0.65, 0.85], device=device), dim=1)
        kc = max(1, M_wb // 4)
        cv25_v = torch.topk(gm, kc, dim=1, largest=False).values.mean(dim=1)

        mw = md_pct.view(W, C, B_bud)
        cev_v = mw[clean_idx].mean(dim=(0, 2)) if clean_idx else torch.zeros(C, device=device)
        iev_v = mw[inv_idx].mean(dim=(0, 2)) if inv_idx else torch.zeros(C, device=device)
        fev_v = mw.mean(dim=(0, 2))

        if clean_idx:
            cgm = tg_bp[clean_idx].view(len(clean_idx), C, B_bud).permute(1, 0, 2).reshape(C, len(clean_idx) * B_bud)
            cqm = torch.quantile(cgm, torch.tensor([0.15, 0.35, 0.50], device=device), dim=1)
            cbrd_v = torch.where(cgm > 0.0, 250.0 * torch.log1p(cgm / 250.0), cgm).mean(dim=1)
            ccv25_v = torch.topk(cgm, max(1, cgm.shape[1] // 4), dim=1, largest=False).values.mean(dim=1)
            ccv15_v = torch.topk(cgm, max(1, (cgm.shape[1] * 15) // 100), dim=1, largest=False).values.mean(dim=1)
            ccv10_v = torch.topk(cgm, max(1, cgm.shape[1] // 10), dim=1, largest=False).values.mean(dim=1)
            ccv05_v = torch.topk(cgm, max(1, cgm.shape[1] // 20), dim=1, largest=False).values.mean(dim=1)
            cdwn_v = torch.sqrt(torch.mean(torch.clamp(-cgm, min=0.0) ** 2, dim=1))
            ctail_p = (
                0.35 * torch.clamp(-110.0 - ccv15_v, min=0.0)
                + 0.35 * torch.clamp(-185.0 - ccv10_v, min=0.0)
                + 0.45 * torch.clamp(-265.0 - ccv05_v, min=0.0)
            )
            cms = mw[clean_idx]
            pw_f = (cms > 0.05).float().mean(dim=(0, 2))
            bw_f = (cms > 0.65).float().mean(dim=(0, 2))
            br_b = 16.0 * pw_f * torch.sigmoid(cev_v * 12.0) + 32.0 * bw_f * torch.sigmoid(cev_v * 8.0)
            cer_v = torch.where(cms > 8.0, 8.0 + 2.5 * torch.log1p((cms - 8.0) / 2.5), cms).mean(dim=(0, 2))
            cmr_v = torch.where(cgm > 800.0, 800.0 + 250.0 * torch.log1p((cgm - 800.0) / 250.0), cgm).mean(dim=1)
            g_cl = (
                0.40 * cbrd_v + 0.14 * cqm[2] + 0.10 * cqm[1] + 0.68 * cmr_v
                + 220.0 * cer_v + br_b + 0.24 * ccv25_v - 0.18 * cdwn_v - 0.70 * ctail_p
            )
        else:
            g_cl = torch.zeros(C, device=device)

        if inv_idx:
            igm = tg_bp[inv_idx].view(len(inv_idx), C, B_bud).permute(1, 0, 2).reshape(C, len(inv_idx) * B_bud)
            imr_v = torch.where(igm > 500.0, 500.0 + 150.0 * torch.log1p((igm - 500.0) / 150.0), igm).mean(dim=1)
            ims = mw[inv_idx]
            ier_v = torch.where(ims > 5.0, 5.0 + 1.5 * torch.log1p((ims - 5.0) / 1.5), ims).mean(dim=(0, 2))
            iq50_v = torch.median(igm, dim=1).values
            icv25_v = torch.topk(igm, max(1, igm.shape[1] // 4), dim=1, largest=False).values.mean(dim=1)
            icv15_v = torch.topk(igm, max(1, (igm.shape[1] * 15) // 100), dim=1, largest=False).values.mean(dim=1)
            g_in = 0.30 * imr_v + 0.20 * iq50_v + 45.0 * ier_v + 0.20 * icv25_v + 0.30 * icv15_v
        else:
            g_in = torch.zeros(C, device=device)

        dd_m = wd_mat.view(W, C, B_bud).max(dim=0).values.mean(dim=1)
        rs_wcb = reg_stats.view(W, C, B_bud, 6)
        bud_wcb = budget.view(W, C, B_bud)
        up_bp_c = (10000.0 * rs_wcb[clean_idx, :, :, 0] / bud_wcb[clean_idx]).mean(dim=(0, 2)) if clean_idx else torch.zeros(C, device=device)
        dn_bp_c = (10000.0 * rs_wcb[clean_idx, :, :, 1] / bud_wcb[clean_idx]).mean(dim=(0, 2)) if clean_idx else torch.zeros(C, device=device)
        fl_bp_c = (10000.0 * rs_wcb[clean_idx, :, :, 2] / bud_wcb[clean_idx]).mean(dim=(0, 2)) if clean_idx else torch.zeros(C, device=device)
        rinv_c = (rs_wcb[clean_idx, :, :, 3] / n_steps_f).mean(dim=(0, 2)) if clean_idx else (rs_wcb[:, :, :, 3] / n_steps_f).mean(dim=(0, 2))
        bo_calm_c = rs_wcb[:, :, :, 4].sum(dim=0).mean(dim=1)
        bo_drop_c = rs_wcb[:, :, :, 5].sum(dim=0).mean(dim=1)

        if enforce_activity_floor and len(clean_idx) >= 4:
            # Директива D6: бездействие на сжатом флэте НЕ штрафуется! Штрафуются только лишние отмены, DD, ловля ножа при падении BO и запас в RISK-OFF:
            tr_phr = 0.16
            ept_bp = 100.0 * cev_v / torch.clamp(cl_turn_v, min=0.02)
            qg = torch.sigmoid((ept_bp - 6.0) / 5.0) * torch.sigmoid(cev_v * 18.0)
            vm = 0.50 * torch.clamp(cl_turn_phr / tr_phr, min=0.35, max=3.20) + 0.50 * torch.sqrt(torch.clamp(cl_turn_phr / tr_phr, min=0.15, max=12.0))
            cpf_v = cancels_v / torch.clamp(fills_v, min=1.0)
            cph_v = (cancels_v / max(1.0, float(W))) / win_hrs
            ch_pen = 0.25 * torch.clamp(cpf_v - 35.0, min=0.0, max=80.0) + 0.14 * torch.clamp(cph_v - 65.0, min=0.0, max=450.0)
            h_dd_p = (18.0 * torch.clamp(dd_m - 2.20, min=0.0) + 0.55 * torch.clamp(-35.0 - cv25_v, min=0.0)) if is_h else torch.zeros(C, device=device)
            dn_bleed_p = 0.45 * torch.clamp(-dn_bp_c - 15.0, min=0.0)
            rinv_p = float(risk_inv_pen_w) * 100.0 * rinv_c
            bo_guard_p = 1.8 * (bo_drop_c / max(1.0, float(W)))
            dm_pen = (
                ch_pen + h_dd_p + dn_bleed_p + rinv_p + bo_guard_p
                + 24.0 * torch.clamp(cl_turn_phr - 1.50, min=0.0) * torch.sigmoid(-cev_v * 35.0)
            )
            pv_bon = (
                260.0 * torch.clamp(cev_v, min=0.0, max=12.0) * vm
                + 24.0 * torch.clamp(bo_calm_c / max(1.0, float(W)), min=0.0, max=4.0) * qg
            )
        else:
            dm_pen = float(risk_inv_pen_w) * 100.0 * rinv_c
            pv_bon = torch.zeros(C, device=device)

        if clean_idx and inv_idx:
            am_gate = torch.sigmoid((g_cl + 10.0) / 4.0) * torch.sigmoid(cev_v * 20.0)
            gv = 0.85 * (g_cl + pv_bon - dm_pen) + 0.15 * torch.where(g_in > 0.0, g_in * am_gate, g_in)
        elif clean_idx:
            gv = g_cl + pv_bon - dm_pen
        else:
            gv = g_in
        return gv, ent_v, cv25_v, qm, fev_v, cev_v, iev_v, dd_m, up_bp_c, dn_bp_c, fl_bp_c, rinv_c, bo_calm_c, bo_drop_c

    geo_v, ent_bp_v, cvar25_v, q_mat, full_ev_v, cl_ev_v, inv_ev_v, dd_v, up_bp_v, dn_bp_v, fl_bp_v, rinv_v, bo_calm_v, bo_drop_v = _reduce_terminal_mode(harsh_stress)
    if harsh_stress:
        cal_geo_v, _, _, _, _, cal_cl_ev_v, _, cal_dd_v, _, _, _, _, _, _ = _reduce_terminal_mode(False)
    else:
        cal_geo_v, cal_cl_ev_v, cal_dd_v = geo_v, cl_ev_v, dd_v

    # Единый перенос компактной матрицы метрик [C, 26] на CPU:
    stats_cpu = torch.stack(
        [
            geo_v, ent_bp_v, cvar25_v, q_mat[0], q_mat[2], q_mat[4],
            full_ev_v, cl_ev_v, inv_ev_v, dd_v, fills_v, cancels_v, requeues_v, fqp_v, mi_bits_v, cl_turn_v, cl_work_v,
            cal_geo_v, cal_cl_ev_v, cal_dd_v,
            up_bp_v, dn_bp_v, fl_bp_v, 100.0 * rinv_v, bo_calm_v, bo_drop_v,
        ],
        dim=1,
    ).cpu().tolist()
    mp_cpu = mp_bar_mat.cpu().tolist()

    results = []
    for ci in range(C):
        st = stats_cpu[ci]
        fills_f = float(st[10])
        cancels_f = float(st[11])
        fqp_f = float(st[13])
        up_b, dn_b, fl_b = float(st[20]), float(st[21]), float(st[22])
        denom_b = abs(up_b) + abs(dn_b) + abs(fl_b) + 1e-4
        beta_sh = (abs(up_b) + abs(dn_b)) / denom_b
        results.append({
            'geo_score_bp': float(st[0]),
            'entropic_growth_bp': float(st[1]),
            'cvar25_bp': float(st[2]),
            'q15_bp': float(st[3]),
            'q50_bp': float(st[4]),
            'q85_bp': float(st[5]),
            'full_ev_pct': float(st[6]),
            'clean_ev_pct': float(st[7]),
            'inv_delta_ev_pct': float(st[8]),
            'dd_pct': float(st[9]),
            'fills': fills_f,
            'cancels': cancels_f,
            'requeues': float(st[12]),
            'cnl_per_fill': cancels_f / max(fills_f, 0.5),
            'eta_queue_pct': 100.0 * min(1.0, fqp_f / max(fills_f, 1.0)) if fills_f >= 0.5 else 100.0,
            'mode_probs': [round(float(x), 4) for x in mp_cpu[ci]],
            'mode_mi_bits': round(float(st[14]), 4),
            'turn_mult': round(float(st[15]), 4),
            'work_frac': round(float(st[16]), 4),
            'calib_geo_score_bp': float(st[17]),
            'calib_clean_ev_pct': float(st[18]),
            'calib_dd_pct': float(st[19]),
            'pnl_up_bp': round(up_b, 2),
            'pnl_dn_bp': round(dn_b, 2),
            'pnl_fl_bp': round(fl_b, 2),
            'beta_share': round(beta_sh, 4),
            'risk_inv_pct': round(float(st[23]), 3),
            'bo_calm_fills': round(float(st[24]), 1),
            'bo_drop_fills': round(float(st[25]), 1),
        })
    return results[:orig_c]
