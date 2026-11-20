#!/usr/bin/env python3
"""
_tools/fused_l27_multimodal_tick.py
=============================================================================
Урок 27: Мультимодальная архитектура латентных экспертов (`K = 4 Latent MoE`),
контекстный маршрутизатор (`288D -> 64 -> 4D`), эндогенное веерное
эшелонирование капитала, регулятор `26D -> 48 -> 2D` и физический Token-Bucket
(`+0.35 токена/с`, ёмкость `10.0`) против 2-секундного тремора отмен.
=============================================================================
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor_cache")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

from _tools.legacy_lessons.fused_l25_regulator_tick import (
    MKT_IDX_L25 as MKT_IDX_L27,
    AG_IDX_L25 as AG_IDX_L27,
    precompute_window_market_z370,
)
from _tools.legacy_lessons.fused_l26_endogenous_tick import (
    MKT_SUBBLOCKS_370,
    AG_SUBBLOCKS_110,
    normalize_market_z370_per_block,
    _normalize_agent_z110_per_block,
    _lookup_sparse_bid_lob_q,
    _lookup_sparse_ask_lob_q,
)

K_MODES_L27 = 4
EXPERT_OUT_DIM_L27 = 45
MOE_OUT_DIM_L27 = K_MODES_L27 * EXPERT_OUT_DIM_L27  # 180
ROUTER_IN_DIM_L27 = 288  # h2(256) + acct_8d(8) + queue_8d(8) + mem_h(16)
ROUTER_H_DIM_L27 = 64
REG_IN_DIM_L27 = 26  # 22D базовых признаков слота + 4D контекстный вектор g_mode
TOKEN_REFILL_PER_SEC_L27 = 0.35
TOKEN_BUCKET_CAP_L27 = 10.0


def _fanned_slot_softmax_l27(
    logits: torch.Tensor,
    tau: torch.Tensor,
    max_slot_share: torch.Tensor,
    floor: float = 0.008,
) -> torch.Tensor:
    """
    Гладкое распределение долей по 12 слотам с эндогенным мягким потолком концентрации
    `max_slot_share` (выучивается экспертом по каналу #44 в диапазоне [0.18 .. 0.92]):
    излишек доли с единственного перегруженного тика плавно перетекает в соседние эшелоны!
    """
    scaled = logits / tau
    p_raw = torch.softmax(scaled, dim=-1)
    # Двухшаговая гладкая проекция на симплекс с верхним пределом доли одного слота:
    p_cap1 = torch.minimum(p_raw, max_slot_share)
    rem1 = torch.clamp(1.0 - p_cap1.sum(dim=-1, keepdim=True), min=0.0)
    room1 = torch.clamp(max_slot_share - p_cap1, min=0.0)
    p_redist = p_cap1 + rem1 * (room1 * (p_raw + 0.02)) / torch.clamp((room1 * (p_raw + 0.02)).sum(dim=-1, keepdim=True), min=1e-6)

    p_keep = torch.where(p_redist >= floor, p_redist, torch.zeros_like(p_redist))
    s_keep = p_keep.sum(dim=-1, keepdim=True)
    return torch.where(s_keep > 1e-5, p_keep / torch.clamp(s_keep, min=1e-6), p_redist)


@torch.compile(dynamic=False)
def fused_l27_multimodal_tick(
    s_cur_i16: torch.Tensor,
    s_nxt_i16: torch.Tensor,
    pre_mkt_gw: torch.Tensor,
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
    sv_w0: torch.Tensor,
    bv_w0: torch.Tensor,
    w1_ag_g: torch.Tensor,
    wr1_g: torch.Tensor,
    br1_g: torch.Tensor,
    wr2_g: torch.Tensor,
    w2_moe_g: torch.Tensor,
    b2_moe_g: torch.Tensor,
    w_gate1_g: torch.Tensor,
    b_gate1_g: torch.Tensor,
    w_gate2_g: torch.Tensor,
    b_gate2_g: torch.Tensor,
    w_rec_g: torch.Tensor,
    b_rec_g: torch.Tensor,
    w_reg1_g: torch.Tensor,
    b_reg1_b_g: torch.Tensor,
    b_reg1_a_g: torch.Tensor,
    w_reg2_g: torch.Tensor,
    b_reg2_b_g: torch.Tensor,
    b_reg2_a_g: torch.Tensor,
    echelon_k01: torch.Tensor,
    budget: torch.Tensor,
    mem_h: torch.Tensor,
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
    tick_size: float,
    maker_fee: float,
    taker_fee: float,
    true_latency: bool = True,
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

    # Физический Token-Bucket лимита API биржи (+0.35 токена/сек, всплесковый резерв 10.0):
    api_tokens = torch.clamp(api_tokens + TOKEN_REFILL_PER_SEC_L27, 0.0, TOKEN_BUCKET_CAP_L27)

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
    unr_pct = torch.where(has_p, ((b1 - avg_entry) / torch.clamp(avg_entry, min=tick_size)) * 25.0 * eff_pos_scale, torch.zeros_like(has_pos))
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

    b_qa_rat_all = torch.where(act_b_mask, torch.clamp(bid_qa / (bid_qa0.float() + 10.0), 0.0, 1.5), torch.full_like(bid_qa, 2.0))
    best_b_idx = torch.argmin(torch.where(act_b_mask, b1.unsqueeze(1) - bid_px, torch.full_like(bid_px, 999.0)), dim=1, keepdim=True)
    best_b_qa = torch.gather(bid_qa, 1, best_b_idx).squeeze(1)
    best_b_qa0 = torch.gather(bid_qa0.float(), 1, best_b_idx).squeeze(1)
    best_b_act = torch.gather(bid_act_s, 1, best_b_idx).squeeze(1).float()
    best_b_rat = torch.where(has_bid > 0.5, torch.clamp(best_b_qa / (best_b_qa0 + 10.0), 0.0, 1.5), torch.ones_like(has_bid))
    best_b_tape = torch.where(has_bid > 0.5, torch.clamp(torch.log1p(best_b_qa / roll_sell_n) / 3.0, 0.0, 2.0), torch.zeros_like(has_bid))
    best_b_age = torch.where(has_bid > 0.5, torch.clamp((s_cur_f - best_b_act) / 120.0, 0.0, 2.0), torch.zeros_like(has_bid))
    deep_b_act = act_b_mask[:, 4:]
    deep_b_cnt = deep_b_act.float().sum(dim=1)
    deep_b_rat = torch.where(
        deep_b_cnt > 0.5,
        torch.where(deep_b_act, b_qa_rat_all[:, 4:], torch.zeros_like(b_qa_rat_all[:, 4:])).sum(dim=1) / torch.clamp(deep_b_cnt, min=1.0),
        torch.ones_like(has_bid),
    )

    a_qa_rat_all = torch.where(act_a_mask, torch.clamp(ask_qa / (ask_qa0.float() + 10.0), 0.0, 1.5), torch.full_like(ask_qa, 2.0))
    best_a_idx = torch.argmin(torch.where(act_a_mask, ask_px - a1.unsqueeze(1), torch.full_like(ask_px, 999.0)), dim=1, keepdim=True)
    best_a_qa = torch.gather(ask_qa, 1, best_a_idx).squeeze(1)
    best_a_qa0 = torch.gather(ask_qa0.float(), 1, best_a_idx).squeeze(1)
    best_a_act = torch.gather(ask_act_s, 1, best_a_idx).squeeze(1).float()
    best_a_rat = torch.where(has_ask > 0.5, torch.clamp(best_a_qa / (best_a_qa0 + 10.0), 0.0, 1.5), torch.ones_like(has_ask))
    best_a_tape = torch.where(has_ask > 0.5, torch.clamp(torch.log1p(best_a_qa / roll_buy_n) / 3.0, 0.0, 2.0), torch.zeros_like(has_ask))
    best_a_age = torch.where(has_ask > 0.5, torch.clamp((s_cur_f - best_a_act) / 120.0, 0.0, 2.0), torch.zeros_like(has_ask))
    q_imb_adv = (1.0 - torch.clamp(best_a_rat, 0.0, 1.0)) * has_ask - (1.0 - torch.clamp(best_b_rat, 0.0, 1.0)) * has_bid

    queue_8d = torch.stack([best_b_rat, best_b_tape, best_b_age, deep_b_rat, best_a_rat, best_a_tape, best_a_age, q_imb_adv], dim=-1).to(dt)

    bud_l1 = torch.log1p(budget / l1_bid_usd) / 3.0
    cash_l1 = torch.log1p(unreserved_cash_now / l1_bid_usd) / 3.0
    pos_l1_ask = torch.clamp((pos_qty * a1) / l1_ask_usd, 0.0, 4.0) / 2.0
    pos_tape = torch.clamp(pos_qty / roll_buy_n, 0.0, 10.0) / 5.0
    low_tr_ratio = torch.where(has_p, torch.clamp(low_tranche_qty / torch.clamp(pos_qty, min=1e-6), 0.0, 1.0), torch.zeros_like(has_pos))
    lifo_edge_pct = torch.where(has_p, torch.clamp(((b1 - min_buy_px) / torch.clamp(min_buy_px, min=tick_size)) * 30.0, -2.5, 2.5), torch.zeros_like(has_pos))
    multi_6d = torch.stack([bud_l1, cash_l1, pos_l1_ask, pos_tape, low_tr_ratio, lifo_edge_pct], dim=-1).to(dt)

    b_slot_12 = torch.clamp((bid_qty * b1.unsqueeze(1)) / torch.clamp(budget.unsqueeze(1), min=25.0), 0.0, 1.0).to(dt)
    a_slot_12 = torch.clamp(ask_qty / torch.clamp(pos_qty.unsqueeze(1), min=2.0), 0.0, 1.0).to(dt)
    b_dist_12 = torch.where(act_b_mask, torch.clamp(((b1.unsqueeze(1) - bid_px) / b1.unsqueeze(1)) * 12.0, -0.5, 3.0), torch.zeros_like(bid_qty)).to(dt)
    a_dist_12 = torch.where(act_a_mask, torch.clamp(((ask_px - a1.unsqueeze(1)) / a1.unsqueeze(1)) * 12.0, -0.5, 3.0), torch.zeros_like(ask_qty)).to(dt)

    b_qprog_12_f = torch.where(act_b_mask, torch.clamp(1.0 - b_qa_rat_all, -0.5, 1.0), torch.full_like(bid_qty, -1.0))
    a_qprog_12_f = torch.where(act_a_mask, torch.clamp(1.0 - a_qa_rat_all, -0.5, 1.0), torch.full_like(ask_qty, -1.0))

    # 1. Прямой проход 480D ствола L27 + латентная микропамять mem_h (16D):
    z_ag_raw = torch.cat(
        [multi_6d, b_slot_12, a_slot_12, b_dist_12, a_dist_12, acct_8d, queue_8d, mem_h, b_qprog_12_f.to(dt), a_qprog_12_f.to(dt)],
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
    rec_u = torch.tanh(rec_raw[:, :, :16])
    rec_gate = torch.sigmoid(rec_raw[:, :, 16:32] + 1.5)
    mem_prev_g = mem_h.view(W, G, B, 16).permute(1, 0, 2, 3).reshape(G, M, 16)
    mem_next_g = rec_gate * mem_prev_g + (1.0 - rec_gate) * rec_u
    mem_h = mem_next_g.view(G, W, B, 16).permute(1, 0, 2, 3).reshape(N, 16)

    # 1B. Контекстный Маршрутизатор (288D -> 64 -> 4D) и 4 Латентных Эксперта (K = 4 x 45D = 180D):
    acct_g = acct_8d.view(W, G, B, 8).permute(1, 0, 2, 3).reshape(G, M, 8)
    queue_g = queue_8d.view(W, G, B, 8).permute(1, 0, 2, 3).reshape(G, M, 8)
    router_in_g = torch.cat([h2, acct_g, queue_g, mem_next_g], dim=-1)  # [G, M, 288]
    r_hid_g = F.silu(torch.bmm(router_in_g, w_gate1_g) + b_gate1_g)     # [G, M, 64]
    gate_logits_g = (torch.bmm(r_hid_g, w_gate2_g) + b_gate2_g).float()  # [G, M, 4]
    g_mode_g = torch.softmax(gate_logits_g, dim=-1)                      # [G, M, 4]

    moe_all_g = (torch.bmm(h2, w2_moe_g) + b2_moe_g).float().view(G, M, K_MODES_L27, EXPERT_OUT_DIM_L27)
    strat_raw_g = (g_mode_g.unsqueeze(-1) * moe_all_g).sum(dim=2)        # [G, M, 45]
    strat_raw = strat_raw_g.view(G, W, B, EXPERT_OUT_DIM_L27).permute(1, 0, 2, 3).reshape(N, EXPERT_OUT_DIM_L27)
    g_mode = g_mode_g.view(G, W, B, K_MODES_L27).permute(1, 0, 2, 3).reshape(N, K_MODES_L27)

    # Накопление статистики модального разнообразия I(State; Mode) без ручных ярлыков:
    sum_mode_p = sum_mode_p + g_mode
    mode_ent_step = -(g_mode * torch.log(torch.clamp(g_mode, min=1e-6))).sum(dim=-1)
    sum_mode_ent = sum_mode_ent + mode_ent_step

    # Разреженная экспозиция E_bid + эндогенное веерное распределение долей по эшелонам:
    raw_ebid = torch.sigmoid(strat_raw[:, 0])
    exp_bid_frac = torch.clamp((raw_ebid - 0.03) / 0.97, 0.0, 1.0)

    tau_b = 0.30 + 2.00 * torch.sigmoid(strat_raw[:, 34:35])
    tau_a = 0.30 + 2.00 * torch.sigmoid(strat_raw[:, 35:36])
    # Канал #44 управляет эндогенным потолком концентрации одного слота [0.18 .. 0.88] (по умолчанию ~0.42 при 0.0):
    max_share_b = 0.18 + 0.70 * torch.sigmoid(strat_raw[:, 44:45] - 0.65)
    max_share_a = 0.22 + 0.72 * torch.sigmoid(strat_raw[:, 44:45] - 0.25)
    pi_bid = _fanned_slot_softmax_l27(strat_raw[:, 1:13], tau_b, max_share_b, floor=0.008)
    pi_ask = _fanned_slot_softmax_l27(strat_raw[:, 13:25], tau_a, max_share_a, floor=0.008)
    # 2. Непрерывный эндогенный Taker-вентиль и честный пробой 10 уровней стакана БЕЗ плиты 1e9:
    u_taker = torch.clamp(torch.tanh(torch.relu(strat_raw[:, 25] - 1.10)), 0.0, 1.0)
    max_slip_pct = 0.005 + 0.145 * torch.sigmoid(strat_raw[:, 36])
    is_taker = (u_taker > 0.02) & (pos_qty >= 1.95) & (api_tokens >= 1.0)
    raw_sell_q = pos_qty * u_taker
    sell_q_req = torch.where((pos_qty - raw_sell_q) < 2.0, pos_qty, raw_sell_q)

    b_p10 = b_p10_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    b_q10_honest = b_q10_raw_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    min_cross_px = torch.clamp(b1 * (1.0 - max_slip_pct), min=tick_size).unsqueeze(1)
    elig_q10 = torch.where(b_p10 >= min_cross_px - 0.1 * tick_size, b_q10_honest, torch.zeros_like(b_q10_honest))
    cum_q10 = torch.cumsum(elig_q10, dim=1)
    prev_q10 = cum_q10 - elig_q10
    take_10 = torch.clamp(torch.minimum(sell_q_req.unsqueeze(1) - prev_q10, elig_q10), min=0.0)
    actual_taker_q = take_10.sum(dim=1)
    exec_taker = is_taker & (actual_taker_q >= 1e-4)

    proceeds = (take_10 * b_p10).sum(dim=1) * (1.0 - taker_fee)
    rem_pos = torch.clamp(pos_qty - actual_taker_q, min=0.0)
    rem_frac = torch.where(pos_qty > 1e-6, rem_pos / torch.clamp(pos_qty, min=1e-6), torch.zeros_like(pos_qty))
    full_exit = rem_pos <= 1e-5

    cash = torch.where(exec_taker, cash + proceeds, cash)
    pos_qty = torch.where(exec_taker, rem_pos, pos_qty)
    pos_cost_usd = torch.where(exec_taker, pos_cost_usd * rem_frac, pos_cost_usd)
    avg_entry = torch.where(exec_taker & full_exit, torch.zeros_like(avg_entry), avg_entry)
    min_buy_px = torch.where(exec_taker & full_exit, torch.zeros_like(min_buy_px), min_buy_px)
    low_tranche_qty = torch.where(exec_taker, low_tranche_qty * rem_frac, low_tranche_qty)
    mid_buy_px = torch.where(exec_taker & full_exit, torch.zeros_like(mid_buy_px), mid_buy_px)
    mid_tranche_qty = torch.where(exec_taker, mid_tranche_qty * rem_frac, mid_tranche_qty)
    had_ask_taker_cnt = torch.where(exec_taker, (ask_qty > 1e-6).long().sum(dim=1), torch.zeros(N, dtype=torch.long, device=ask_qty.device))
    ask_qty = torch.where(exec_taker.unsqueeze(1), torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(exec_taker.unsqueeze(1), torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(exec_taker.unsqueeze(1), torch.zeros_like(ask_qa), ask_qa)
    ask_qa0 = torch.where(exec_taker.unsqueeze(1), torch.zeros_like(ask_qa0), ask_qa0)
    api_tokens = torch.where(exec_taker, torch.clamp(api_tokens - 1.0 - had_ask_taker_cnt.float(), min=0.0), api_tokens)
    n_fills = torch.clamp(n_fills.long() + exec_taker.long(), max=30000).to(torch.int16)

    # 3. Эндогенная нелинейная геометрия 24 слотов (безразмерный размах до 25% от b1, БЕЗ замка base_tp!):
    wide_sp = (a1 - b1) >= 2.0 * tick_size
    penny_room_b = torch.where(wide_sp, torch.full_like(b1, tick_size), torch.zeros_like(b1))

    front_b_shift = -penny_room_b * torch.sigmoid(strat_raw[:, 37]) + b1 * 0.04 * F.softplus(strat_raw[:, 38] - 1.8) * 0.35
    span_b_usd = b1 * (0.015 + 0.235 * torch.sigmoid(strat_raw[:, 39]))
    gamma_b = 0.65 + 1.65 * torch.sigmoid(strat_raw[:, 40])

    curve_b = torch.pow(echelon_k01.unsqueeze(0), gamma_b.unsqueeze(1))
    raw_t_bid = b1.unsqueeze(1) - front_b_shift.unsqueeze(1) - span_b_usd.unsqueeze(1) * curve_b
    max_penny_b = torch.clamp(a1 - tick_size, min=tick_size).unsqueeze(1)
    t_bid_px = torch.clamp(torch.round(torch.minimum(raw_t_bid, max_penny_b) / tick_size) * tick_size, min=tick_size)

    hp_now = pos_qty >= 1.95
    hp_now_mask = hp_now.unsqueeze(1)
    ref_entry = torch.where(hp_now, avg_entry, a1)
    skew_blend = torch.sigmoid(strat_raw[:, 41] + 1.2)
    ask_edge_pct = 0.04 * torch.tanh(strat_raw[:, 42])
    span_a_usd = a1 * (0.012 + 0.238 * torch.sigmoid(strat_raw[:, 39] + 0.5 * strat_raw[:, 42]))
    gamma_a = 0.65 + 1.65 * torch.sigmoid(strat_raw[:, 40])

    anchor_base_a = a1 + skew_blend * (ref_entry - a1) + a1 * ask_edge_pct - penny_room_b * torch.sigmoid(-strat_raw[:, 41])
    min_maker_ask = (b1 + tick_size).unsqueeze(1)
    curve_a = torch.pow(echelon_k01.unsqueeze(0), gamma_a.unsqueeze(1))
    raw_t_ask = anchor_base_a.unsqueeze(1) + span_a_usd.unsqueeze(1) * curve_a
    t_ask_px = torch.maximum(min_maker_ask, torch.round(raw_t_ask / tick_size) * tick_size)

    val_pred_now = 0.05 * torch.tanh(strat_raw[:, 43])

    tot_bud_eff = torch.maximum(budget, cash + pos_qty * b1)
    target_bid_pool_usd = torch.clamp(tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1, min=0.0)
    q_target_b = (target_bid_pool_usd.unsqueeze(1) * pi_bid) / (t_bid_px * (1.0 + maker_fee))
    pair_small_b = ((q_target_b[:, 0::2] > 0.05) & (q_target_b[:, 0::2] < 2.0)) | ((q_target_b[:, 1::2] > 0.05) & (q_target_b[:, 1::2] < 2.0))
    pair_comb_b = q_target_b[:, 0::2] + q_target_b[:, 1::2]
    even_win_b = pi_bid[:, 0::2] >= pi_bid[:, 1::2]
    qb_ev = torch.where(pair_small_b, torch.where(even_win_b, pair_comb_b, torch.zeros_like(pair_comb_b)), q_target_b[:, 0::2])
    qb_od = torch.where(pair_small_b, torch.where(even_win_b, torch.zeros_like(pair_comb_b), pair_comb_b), q_target_b[:, 1::2])
    q_target_b = torch.stack([qb_ev, qb_od], dim=2).reshape(N, S)
    q_target_b = torch.where(q_target_b >= 2.0, q_target_b, torch.zeros_like(q_target_b))

    q_target_a = torch.where(hp_now_mask, pos_qty.unsqueeze(1) * pi_ask, torch.zeros_like(pi_ask))
    dust_a_mask = (q_target_a > 0.0) & (q_target_a < 2.0)
    dust_a_sum = torch.where(dust_a_mask, q_target_a, torch.zeros_like(q_target_a)).sum(dim=1, keepdim=True)
    q_target_a = torch.where(dust_a_mask, torch.zeros_like(q_target_a), q_target_a)
    best_a_t_idx = torch.argmax(pi_ask, dim=1, keepdim=True)
    best_a_oh = torch.zeros_like(q_target_a).scatter_(1, best_a_t_idx, 1.0)
    q_target_a = torch.where(hp_now_mask & (dust_a_sum >= 0.01) & (best_a_oh > 0.5), torch.minimum(pos_qty.unsqueeze(1), q_target_a + dust_a_sum), q_target_a)
    q_target_a = torch.where(q_target_a >= 2.0, q_target_a, torch.zeros_like(q_target_a))

    # 4. Модально-контекстный Послотовый Нейро-Регулятор (26D -> 48 -> 2D с гистерезисом ценности очереди):
    ctx_4d_b = torch.tanh(strat_raw[:, 26:30])
    ctx_4d_a = torch.tanh(strat_raw[:, 30:34])
    s_2d = s_cur_i16.view(1, 1)
    next_s_2d = s_nxt_i16.view(1, 1)
    act_lat_2d = (next_s_2d + 1) if true_latency else next_s_2d

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

    qprog_b_01 = torch.where(had_b_pre, torch.clamp(1.0 - b_qa_rat_all, 0.0, 1.0), torch.zeros_like(bid_qty))
    qprog_a_01 = torch.where(had_a_pre, torch.clamp(1.0 - a_qa_rat_all, 0.0, 1.0), torch.zeros_like(ask_qty))
    log_qa_b = torch.where(had_b_pre, torch.log1p(torch.clamp(bid_qa, min=0.0)) / 10.0, torch.zeros_like(bid_qty))
    log_qa_a = torch.where(had_a_pre, torch.log1p(torch.clamp(ask_qa, min=0.0)) / 10.0, torch.zeros_like(ask_qty))

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
    mean_qp_b = ((qprog_b_01 * cur_val_b).sum(dim=1, keepdim=True) / torch.clamp(cur_val_b.sum(dim=1, keepdim=True), min=0.05)).expand(N, S)
    mean_qp_a = ((qprog_a_01 * cur_val_a).sum(dim=1, keepdim=True) / torch.clamp(cur_val_a.sum(dim=1, keepdim=True), min=0.05)).expand(N, S)
    api_ratio_2d = (api_tokens / TOKEN_BUCKET_CAP_L27).unsqueeze(1).expand(N, S)
    ech_2d = echelon_k01.unsqueeze(0).expand(N, S)
    sp_rel_2d = torch.clamp(((a1 - b1) / b1) * 40.0, 0.0, 3.0).unsqueeze(1).expand(N, S)
    g_mode_s = g_mode.unsqueeze(1).expand(N, S, K_MODES_L27)

    feat_b_26 = torch.cat(
        [
            torch.stack(
                [
                    torch.zeros_like(bid_qty), had_b_pre.float(), cur_val_b, targ_val_b, rel_delta_b, drift_b,
                    qprog_b_01, log_qa_b, age_b_norm, depth_b_norm, front_b_fl, free_b_2d,
                ],
                dim=-1,
            ),
            ctx_4d_b.unsqueeze(1).expand(N, S, 4),
            torch.stack([tot_side_b, neigh_val_b, mean_qp_b, api_ratio_2d, ech_2d, sp_rel_2d], dim=-1),
            g_mode_s,
        ],
        dim=-1,
    ).to(dt)
    feat_a_26 = torch.cat(
        [
            torch.stack(
                [
                    torch.ones_like(ask_qty), had_a_pre.float(), cur_val_a, targ_val_a, rel_delta_a, drift_a,
                    qprog_a_01, log_qa_a, age_a_norm, depth_a_norm, front_a_fl, free_a_2d,
                ],
                dim=-1,
            ),
            ctx_4d_a.unsqueeze(1).expand(N, S, 4),
            torch.stack([tot_side_a, neigh_val_a, mean_qp_a, api_ratio_2d, ech_2d, sp_rel_2d], dim=-1),
            g_mode_s,
        ],
        dim=-1,
    ).to(dt)

    fb_g = feat_b_26.view(W, G, B, S, REG_IN_DIM_L27).permute(1, 0, 2, 3, 4).reshape(G, M * S, REG_IN_DIM_L27)
    fa_g = feat_a_26.view(W, G, B, S, REG_IN_DIM_L27).permute(1, 0, 2, 3, 4).reshape(G, M * S, REG_IN_DIM_L27)
    rb_h = F.silu(torch.bmm(fb_g, w_reg1_g) + b_reg1_b_g)
    ra_h = F.silu(torch.bmm(fa_g, w_reg1_g) + b_reg1_a_g)
    rb_out = (torch.bmm(rb_h, w_reg2_g) + b_reg2_b_g).float().view(G, W, B, S, 2).permute(1, 0, 2, 3, 4).reshape(N, S, 2)
    ra_out = (torch.bmm(ra_h, w_reg2_g) + b_reg2_a_g).float().view(G, W, B, S, 2).permute(1, 0, 2, 3, 4).reshape(N, S, 2)
    g_plc_b = rb_out[:, :, 0] > 0.0
    g_cnl_b = rb_out[:, :, 1] > 0.0
    g_plc_a = ra_out[:, :, 0] > 0.0
    g_cnl_a = ra_out[:, :, 1] > 0.0
    # 5. Физический Token-Bucket API (1 токен = 1 операция на бирже) и постановка/отмена 24 слотов:
    want_cnl_b = had_b_pre & g_cnl_b
    want_cnl_a = had_a_pre & ((~hp_now_mask) | g_cnl_a)
    cum_cnl = want_cnl_b.float().cumsum(dim=1)
    clear_b = want_cnl_b & (cum_cnl <= api_tokens.unsqueeze(1) + 1e-4)
    api_tokens = torch.clamp(api_tokens - clear_b.float().sum(dim=1), min=0.0)

    cum_cnl_a = want_cnl_a.float().cumsum(dim=1)
    clear_a = want_cnl_a & ((~hp_now_mask) | (cum_cnl_a <= api_tokens.unsqueeze(1) + 1e-4))
    api_tokens = torch.clamp(api_tokens - (clear_a & hp_now_mask).float().sum(dim=1), min=0.0)

    n_cnl_b_step = clear_b.long().sum(dim=1)
    n_cnl_a_step = clear_a.long().sum(dim=1) + had_ask_taker_cnt

    bid_qty = torch.where(clear_b, torch.zeros_like(bid_qty), bid_qty)
    bid_px = torch.where(clear_b, torch.zeros_like(bid_px), bid_px)
    bid_qa = torch.where(clear_b, torch.zeros_like(bid_qa), bid_qa)
    bid_qa0 = torch.where(clear_b, torch.zeros_like(bid_qa0), bid_qa0)

    ask_qty = torch.where(clear_a, torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(clear_a, torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(clear_a, torch.zeros_like(ask_qa), ask_qa)
    ask_qa0 = torch.where(clear_a, torch.zeros_like(ask_qa0), ask_qa0)

    res_cash_after_cnl = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    free_cash_for_b = torch.clamp(cash * 0.998 - res_cash_after_cnl, min=0.0)
    rem_exp_for_b = torch.clamp(tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1 - res_cash_after_cnl, min=0.0)
    avail_b_usd = torch.minimum(free_cash_for_b, rem_exp_for_b)

    empty_b_now = bid_qty <= 1e-6
    cand_plc_b = empty_b_now & g_plc_b & (q_target_b >= 2.0)
    ord_b_idx = torch.argsort(torch.where(cand_plc_b, pi_bid, torch.full_like(pi_bid, -1.0)), dim=1, descending=True)
    inv_b_idx = torch.argsort(ord_b_idx, dim=1)
    cand_b_sorted = torch.gather(cand_plc_b.float(), 1, ord_b_idx)
    allow_b_tok_sorted = (cand_b_sorted.cumsum(dim=1) <= api_tokens.unsqueeze(1) + 1e-4) & (cand_b_sorted > 0.5)
    want_place_b = torch.gather(allow_b_tok_sorted, 1, inv_b_idx)

    want_b_usd = torch.where(want_place_b, q_target_b * t_bid_px * (1.0 + maker_fee), torch.zeros_like(q_target_b))
    tot_want_b_usd = want_b_usd.sum(dim=1, keepdim=True)
    scale_cap_b = torch.where(tot_want_b_usd > avail_b_usd.unsqueeze(1), avail_b_usd.unsqueeze(1) / torch.clamp(tot_want_b_usd, min=1e-6), torch.ones_like(tot_want_b_usd))
    plc_b_qty = (want_b_usd * scale_cap_b) / (t_bid_px * (1.0 + maker_fee))
    valid_b = want_place_b & (plc_b_qty >= 2.0)
    plc_b_qty = torch.where(valid_b, plc_b_qty, torch.zeros_like(plc_b_qty))
    n_plc_b_step = valid_b.long().sum(dim=1)
    api_tokens = torch.clamp(api_tokens - valid_b.float().sum(dim=1), min=0.0)
    n_req_b_step = (clear_b & valid_b).long().sum(dim=1)

    # Price-Matched инициализация очереди + физическое самонасыщение при вливании > L1 глубины в 1 тик:
    exact_b_q_2d = _lookup_sparse_bid_lob_q(b_p10_1w, b_q10_raw_1w, t_bid_px, W, E, S, tick_size)
    inside_sp_b = t_bid_px > (b1.unsqueeze(1) + 0.1 * tick_size)
    base_b_qa = torch.where(inside_sp_b, torch.zeros_like(exact_b_q_2d), exact_b_q_2d)
    plc_b_usd = plc_b_qty * t_bid_px
    b_over50 = torch.sigmoid((plc_b_usd - 50.0) / 18.0)
    b_depth_sat = F.softplus(plc_b_usd / torch.clamp(l1_bid_usd.unsqueeze(1), min=100.0) - 0.85)
    b_self_impact_qa = base_b_qa * (1.0 + 0.75 * b_over50) + b_over50 * (0.45 * plc_b_qty) + 0.35 * b_depth_sat * plc_b_qty

    bid_px = torch.where(valid_b, t_bid_px, bid_px)
    bid_qty = torch.where(valid_b, plc_b_qty, bid_qty)
    bid_qa = torch.where(valid_b, b_self_impact_qa, bid_qa)
    bid_qa0 = torch.where(valid_b, b_self_impact_qa.to(dt), bid_qa0)
    bid_act_s = torch.where(valid_b, act_lat_2d.expand_as(bid_act_s), bid_act_s)

    unreserved_pos = torch.clamp(pos_qty - ask_qty.sum(dim=1), min=0.0).unsqueeze(1)
    empty_a_now = ask_qty <= 1e-6
    cand_plc_a = empty_a_now & hp_now_mask & g_plc_a & (q_target_a >= 2.0)
    ord_a_idx = torch.argsort(torch.where(cand_plc_a, pi_ask, torch.full_like(pi_ask, -1.0)), dim=1, descending=True)
    inv_a_idx = torch.argsort(ord_a_idx, dim=1)
    cand_a_sorted = torch.gather(cand_plc_a.float(), 1, ord_a_idx)
    allow_a_tok_sorted = (cand_a_sorted.cumsum(dim=1) <= api_tokens.unsqueeze(1) + 1e-4) & (cand_a_sorted > 0.5)
    want_place_a = torch.gather(allow_a_tok_sorted, 1, inv_a_idx)

    want_a_q = torch.where(want_place_a, q_target_a, torch.zeros_like(q_target_a))
    tot_want_a_q = want_a_q.sum(dim=1, keepdim=True)
    scale_cap_a = torch.where(tot_want_a_q > unreserved_pos, unreserved_pos / torch.clamp(tot_want_a_q, min=1e-6), torch.ones_like(tot_want_a_q))
    plc_a_qty = want_a_q * scale_cap_a
    valid_a = want_place_a & (plc_a_qty >= 2.0)
    plc_a_qty = torch.where(valid_a, plc_a_qty, torch.zeros_like(plc_a_qty))
    n_plc_a_step = valid_a.long().sum(dim=1)
    api_tokens = torch.clamp(api_tokens - valid_a.float().sum(dim=1), min=0.0)
    n_req_a_step = (clear_a & valid_a).long().sum(dim=1)

    exact_a_q_2d = _lookup_sparse_ask_lob_q(a_p10_1w, a_q10_raw_1w, t_ask_px, W, E, S, tick_size)
    inside_sp_a = t_ask_px < (a1.unsqueeze(1) - 0.1 * tick_size)
    base_a_qa = torch.where(inside_sp_a, torch.zeros_like(exact_a_q_2d), exact_a_q_2d)
    plc_a_usd = plc_a_qty * t_ask_px
    a_over50 = torch.sigmoid((plc_a_usd - 50.0) / 18.0)
    a_depth_sat = F.softplus(plc_a_usd / torch.clamp(l1_ask_usd.unsqueeze(1), min=100.0) - 0.85)
    a_self_impact_qa = base_a_qa * (1.0 + 0.75 * a_over50) + a_over50 * (0.45 * plc_a_qty) + 0.35 * a_depth_sat * plc_a_qty

    ask_px = torch.where(valid_a, t_ask_px, ask_px)
    ask_qty = torch.where(valid_a, plc_a_qty, ask_qty)
    ask_qa = torch.where(valid_a, a_self_impact_qa, ask_qa)
    ask_qa0 = torch.where(valid_a, a_self_impact_qa.to(dt), ask_qa0)
    ask_act_s = torch.where(valid_a, act_lat_2d.expand_as(ask_act_s), ask_act_s)

    n_mut_tot = n_cnl_b_step + n_cnl_a_step + n_plc_b_step + n_plc_a_step
    if true_latency:
        burst_delay = ((n_mut_tot > 4) | (api_tokens < 2.0)).unsqueeze(1)
        extra_s = torch.where((api_tokens < 1.0).unsqueeze(1), (next_s_2d + 3).expand_as(bid_act_s), (next_s_2d + 2).expand_as(bid_act_s))
        bid_act_s = torch.where(valid_b & burst_delay, extra_s, bid_act_s)
        ask_act_s = torch.where(valid_a & burst_delay, extra_s, ask_act_s)

    # 6. Исполнение FIFO-очередей в произвольном порядке цен слотов + 3-траншевый LIFO:
    nb1_1w = torch.clamp(nb1_w0, min=tick_size)
    na1_1w = torch.clamp(na1_w0, min=nb1_1w + tick_size)
    nb1 = nb1_1w.unsqueeze(1).expand(W, E).reshape(N, 1)
    na1 = na1_1w.unsqueeze(1).expand(W, E).reshape(N, 1)
    sv = sv_w0.unsqueeze(1).expand(W, E).reshape(N, 1)
    bv = bv_w0.unsqueeze(1).expand(W, E).reshape(N, 1)

    act_b = (bid_qty > 1e-6) & (next_s_2d >= bid_act_s)
    strict_cross_b = na1 <= (bid_px - 0.5 * tick_size)
    hit_b = act_b & (strict_cross_b | ((nb1 <= bid_px + 1e-6) & (sv > 0.0)))
    any_strict_b = (act_b & strict_cross_b).any(dim=1, keepdim=True)
    tot_flow_b = torch.where(any_strict_b, sv + 0.12 * b1_q.unsqueeze(1), sv)

    even_act_b = act_b[:, 0::2] & (torch.abs(bid_px[:, 1::2] - bid_px[:, 0::2]) < 0.1 * tick_size)
    inc_qa_b = bid_qa.clone()
    inc_qa_b[:, 1::2] = torch.where(even_act_b, torch.clamp(bid_qa[:, 1::2] - bid_qa[:, 0::2], min=0.0), bid_qa[:, 1::2])
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
    qa_eaten_b = torch.minimum(eff_qa_b, rem_flow_at_slot_b)
    bid_qa = torch.where(hit_b, torch.clamp(bid_qa - qa_eaten_b, min=0.0), bid_qa)
    bid_qa[:, 1::2] = torch.where(act_b[:, 1::2] & even_act_b, torch.clamp(bid_qa[:, 1::2] - qa_eaten_b[:, 0::2], min=0.0), bid_qa[:, 1::2])

    want_b = torch.minimum(eff_qty_b, torch.clamp(rem_flow_at_slot_b - eff_qa_b, min=0.0))
    cum_want_b = torch.cumsum(want_b, dim=1)
    max_afford_b = torch.clamp(cash * 0.998 / (b1 * (1.0 + maker_fee)), min=0.0).unsqueeze(1)
    f_q = torch.minimum(want_b, torch.clamp(max_afford_b - (cum_want_b - want_b), min=0.0))

    has_slot_b = f_q > 1e-5
    tot_fq = f_q.sum(dim=1)
    tot_cost_raw = (f_q * bid_px).sum(dim=1)
    tot_cost = tot_cost_raw * (1.0 + maker_fee)
    has_buy = tot_fq > 1e-5
    n_fills = torch.clamp(n_fills.long() + has_buy.long(), max=30000).to(torch.int16)
    fill_qp_b_step = torch.where(has_buy, (f_q * pre_fill_qp_b).sum(dim=1) / torch.clamp(tot_fq, min=1e-6), torch.zeros_like(tot_fq))
    old_q = pos_qty
    new_q = old_q + tot_fq
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
    bid_qty = torch.where(has_slot_b, torch.where(rem_ord_b >= 1.0, rem_ord_b, torch.zeros_like(rem_ord_b)), bid_qty)
    act_a = (ask_qty > 1e-6) & (next_s_2d >= ask_act_s) & (pos_qty.unsqueeze(1) > 1e-6)
    strict_cross_a = nb1 >= (ask_px + 0.5 * tick_size)
    hit_a = act_a & (strict_cross_a | ((na1 >= ask_px - 1e-6) & (bv > 0.0)))
    any_strict_a = (act_a & strict_cross_a).any(dim=1, keepdim=True)
    tot_flow_a = torch.where(any_strict_a, bv + 0.12 * a1_q.unsqueeze(1), bv)

    even_act_a = act_a[:, 0::2] & (torch.abs(ask_px[:, 1::2] - ask_px[:, 0::2]) < 0.1 * tick_size)
    inc_qa_a = ask_qa.clone()
    inc_qa_a[:, 1::2] = torch.where(even_act_a, torch.clamp(ask_qa[:, 1::2] - ask_qa[:, 0::2], min=0.0), ask_qa[:, 1::2])
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
    qa_eaten_a = torch.minimum(eff_qa_a, rem_flow_at_slot_a)
    ask_qa = torch.where(hit_a, torch.clamp(ask_qa - qa_eaten_a, min=0.0), ask_qa)
    ask_qa[:, 1::2] = torch.where(act_a[:, 1::2] & even_act_a, torch.clamp(ask_qa[:, 1::2] - qa_eaten_a[:, 0::2], min=0.0), ask_qa[:, 1::2])

    want_a = torch.minimum(eff_qty_a, torch.clamp(rem_flow_at_slot_a - eff_qa_a, min=0.0))
    cum_want_a = torch.cumsum(want_a, dim=1)
    f_qa = torch.minimum(want_a, torch.clamp(pos_qty.unsqueeze(1) - (cum_want_a - want_a), min=0.0))

    has_slot_a = f_qa > 1e-5
    tot_fqa = f_qa.sum(dim=1)
    tot_rev = (f_qa * ask_px).sum(dim=1) * (1.0 - maker_fee)
    has_sell = tot_fqa > 1e-5
    n_fills = torch.clamp(n_fills.long() + has_sell.long(), max=30000).to(torch.int16)
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
    ask_qty = torch.where(has_slot_a, torch.where((rem_ord_a >= 1.0) & (pos_qty.unsqueeze(1) >= 1.0), torch.minimum(rem_ord_a, pos_qty.unsqueeze(1)), torch.zeros_like(rem_ord_a)), ask_qty)

    # 7. Безразмерная логарифмическая траекторная просадка и самокалибровка критика V_risk:
    mtm_eq = torch.clamp(cash + pos_qty * nb1.squeeze(1) * (1.0 - maker_fee), min=1e-2)
    peak_mtm_eq = torch.maximum(peak_mtm_eq, mtm_eq)
    log_dd_step = torch.clamp(torch.log(peak_mtm_eq / mtm_eq), min=0.0)
    sum_log_dd = sum_log_dd + log_dd_step
    dd_pct = 100.0 * torch.clamp(peak_mtm_eq - mtm_eq, min=0.0) / budget
    max_mtm_dd_pct = torch.maximum(max_mtm_dd_pct, dd_pct)

    realized_dlog = torch.log(mtm_eq / torch.clamp(prev_mtm_eq, min=1e-2))
    td_diff = realized_dlog + 0.985 * val_pred_now - prev_val_pred
    sum_val_err = sum_val_err + td_diff * td_diff
    prev_val_pred = val_pred_now
    prev_mtm_eq = mtm_eq

    n_req_step = n_req_b_step + n_req_a_step
    sum_mut_load = sum_mut_load + n_mut_tot.float()
    n_cancels = torch.clamp(n_cancels.long() + n_cnl_b_step + n_cnl_a_step, max=30000).to(torch.int16)
    n_requeues = torch.clamp(n_requeues.long() + n_req_step, max=30000).to(torch.int16)

    return (
        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
        api_tokens, peak_mtm_eq, sum_log_dd, max_mtm_dd_pct, sum_mut_load,
        sum_fill_qprog, sum_val_err, prev_val_pred, prev_mtm_eq,
        sum_mode_p, sum_mode_ent,
        n_fills, n_cancels, n_requeues,
    )


@torch.no_grad()
def evaluate_l27_basket(
    val_env,
    bf16_runner,
    lob_fu_all: torch.Tensor,
    v7_all: torch.Tensor,
    bo36_all: torch.Tensor,
    roll_sell_30s_all: torch.Tensor,
    cand_states: list[dict],
    val_windows: list[tuple[str, int, float, float]],
    win_len: int,
    tick_sz: float,
    device: torch.device,
    true_latency: bool = True,
) -> list[dict]:
    """
    Внешняя независимая оценка мультимодальных кандидатов Урока 27:
    вычисляет Asymmetric Geometric Growth (`geo_score_bp`), `entropic_growth_bp`,
    CVaR25, Q15/Q50/Q85, взаимную информацию маршрутизатора `mode_mi_bits` и чистые EV-метрики.
    """
    W = len(val_windows)
    C = len(cand_states)
    B_bud = 6
    E = C * B_bud
    N = W * E
    S = 12
    dt = lob_fu_all.dtype

    mkt_idx_t = torch.tensor(MKT_IDX_L27, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_L27, dtype=torch.long, device=device)
    echelon_k01 = torch.tensor([0.0, 0.0, 0.12, 0.12, 0.26, 0.26, 0.44, 0.44, 0.68, 0.68, 1.0, 1.0], dtype=torch.float32, device=device)

    w1_l, b1_l, wr1_l, br1_l, wr2_l, w2m_l, b2m_l = [], [], [], [], [], [], []
    wg1_l, bg1_l, wg2_l, bg2_l, wrec_l, brec_l = [], [], [], [], [], []
    wreg1_l, breg1b_l, breg1a_l, wreg2_l, breg2b_l, breg2a_l = [], [], [], [], [], []
    for st in cand_states:
        w1_l.append(st["w1"].to(device=device, dtype=dt))
        b1_l.append(st["b1"].to(device=device, dtype=dt).view(1, 384))
        wr1_l.append(st["w_res1"].to(device=device, dtype=dt))
        br1_l.append(st["b_res1"].to(device=device, dtype=dt).view(1, 128))
        wr2_l.append(st["w_res2"].to(device=device, dtype=dt))
        w2m_l.append(st["w2_moe"].to(device=device, dtype=dt))
        b2m_l.append(st["b2_moe"].to(device=device, dtype=dt).view(1, MOE_OUT_DIM_L27))
        wg1_l.append(st["w_gate1"].to(device=device, dtype=dt))
        bg1_l.append(st["b_gate1"].to(device=device, dtype=dt).view(1, ROUTER_H_DIM_L27))
        wg2_l.append(st["w_gate2"].to(device=device, dtype=dt))
        bg2_l.append(st["b_gate2"].to(device=device, dtype=dt).view(1, K_MODES_L27))
        wrec_l.append(st["w_rec"].to(device=device, dtype=dt))
        brec_l.append(st["b_rec"].to(device=device, dtype=dt).view(1, 32))
        wreg1_l.append(st["w_reg1"].to(device=device, dtype=dt))
        breg1b_l.append(st["b_reg1_b"].to(device=device, dtype=dt).view(1, 48))
        breg1a_l.append(st["b_reg1_a"].to(device=device, dtype=dt).view(1, 48))
        wreg2_l.append(st["w_reg2"].to(device=device, dtype=dt))
        breg2b_l.append(st["b_reg2_b"].to(device=device, dtype=dt).view(1, 2))
        breg2a_l.append(st["b_reg2_a"].to(device=device, dtype=dt).view(1, 2))

    w1_c = torch.stack(w1_l, dim=0).contiguous()
    w1_mkt_c = w1_c.index_select(1, mkt_idx_t).contiguous()
    w1_ag_c = w1_c.index_select(1, ag_idx_t).contiguous()
    b1_c = torch.stack(b1_l, dim=0).contiguous()
    wr1_c = torch.stack(wr1_l, dim=0).contiguous()
    br1_c = torch.stack(br1_l, dim=0).contiguous()
    wr2_c = torch.stack(wr2_l, dim=0).contiguous()
    w2m_c = torch.stack(w2m_l, dim=0).contiguous()
    b2m_c = torch.stack(b2m_l, dim=0).contiguous()
    wg1_c = torch.stack(wg1_l, dim=0).contiguous()
    bg1_c = torch.stack(bg1_l, dim=0).contiguous()
    wg2_c = torch.stack(wg2_l, dim=0).contiguous()
    bg2_c = torch.stack(bg2_l, dim=0).contiguous()
    wrec_c = torch.stack(wrec_l, dim=0).contiguous()
    brec_c = torch.stack(brec_l, dim=0).contiguous()
    wreg1_c = torch.stack(wreg1_l, dim=0).contiguous()
    breg1b_c = torch.stack(breg1b_l, dim=0).contiguous()
    breg1a_c = torch.stack(breg1a_l, dim=0).contiguous()
    wreg2_c = torch.stack(wreg2_l, dim=0).contiguous()
    breg2b_c = torch.stack(breg2b_l, dim=0).contiguous()
    breg2a_c = torch.stack(breg2a_l, dim=0).contiguous()

    w_starts = torch.tensor([int(ws) for _, ws, _, _ in val_windows], dtype=torch.long, device=device)
    t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
    t_steps_cl = torch.clamp(t_steps, max=val_env.max_steps - 1)

    flow_w_all = val_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
    lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()
    v7_w_all = v7_all[t_steps_cl[: win_len - 1]].contiguous()
    bo36_w_all = bo36_all[t_steps_cl[: win_len - 1]].contiguous()

    z_mkt_raw = precompute_window_market_z370(flow_w_all, lob_fu_w_all, v7_w_all, bo36_w_all, bf16_runner)
    z_mkt_norm, _ = normalize_market_z370_per_block(z_mkt_raw)
    z_mkt_c_in = z_mkt_norm.view((win_len - 1) * W, 370).unsqueeze(0).expand(C, -1, -1)
    pre_mkt_all = (torch.bmm(z_mkt_c_in, w1_mkt_c) + b1_c).view(C, win_len - 1, W, 384).permute(1, 0, 2, 3).contiguous()

    b10_w_all = val_env.bids[t_steps_cl].contiguous()
    a10_w_all = val_env.asks[t_steps_cl].contiguous()
    bq10_w_all = val_env.bids_q[t_steps_cl].contiguous()
    aq10_w_all = val_env.asks_q[t_steps_cl].contiguous()
    b1_w_all = b10_w_all[:, :, 0].contiguous()
    a1_w_all = a10_w_all[:, :, 0].contiguous()
    b1q_w_all = bq10_w_all[:, :, 0].contiguous()
    a1q_w_all = aq10_w_all[:, :, 0].contiguous()
    rb_w_all = val_env.roll_buy_30s[t_steps_cl].contiguous()
    rs_w_all = roll_sell_30s_all[t_steps_cl].contiguous()
    sv_w_all = val_env.sell_vol[t_steps_cl].contiguous()
    bv_w_all = val_env.buy_vol[t_steps_cl].contiguous()

    budget_tiers = torch.tensor([100.0, 300.0, 625.0, 1000.0, 1500.0, 2500.0], device=device)
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

    for wi, (_, ws, rho_0, rel_off) in enumerate(val_windows):
        if rho_0 > 0.01:
            s0_w, s1_w = wi * E, (wi + 1) * E
            b1_0 = float(max(tick_sz, float(val_env.bids[int(ws), 0].item())))
            epx = max(tick_sz, round((b1_0 * (1.0 + rel_off)) / tick_sz) * tick_sz)
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
    api_tokens = torch.full((N,), TOKEN_BUCKET_CAP_L27, dtype=torch.float32, device=device)

    init_b1 = torch.clamp(b1_w_all[0], min=tick_sz).unsqueeze(1).expand(W, E).reshape(N)
    start_mtm_eq = torch.clamp(cash + pos_qty * init_b1 * (1.0 - val_env.maker_fee), min=1.0)
    peak_mtm_eq = start_mtm_eq.clone()
    prev_mtm_eq = start_mtm_eq.clone()
    prev_val_pred = torch.zeros(N, dtype=torch.float32, device=device)
    sum_log_dd = torch.zeros(N, dtype=torch.float32, device=device)
    max_dd_pct = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mut = torch.zeros(N, dtype=torch.float32, device=device)
    sum_fqp = torch.zeros(N, dtype=torch.float32, device=device)
    sum_val_err = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mode_p = torch.zeros((N, K_MODES_L27), dtype=torch.float32, device=device)
    sum_mode_ent = torch.zeros(N, dtype=torch.float32, device=device)
    n_fills = torch.zeros(N, dtype=torch.int16, device=device)
    n_cancels = torch.zeros(N, dtype=torch.int16, device=device)
    n_requeues = torch.zeros(N, dtype=torch.int16, device=device)

    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    for s in range(win_len - 1):
        (
            mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
            min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
            bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
            ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
            api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
            sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
            sum_mode_p, sum_mode_ent,
            n_fills, n_cancels, n_requeues,
        ) = fused_l27_multimodal_tick(
            s_steps_i16[s], s_steps_i16[s + 1], pre_mkt_all[s],
            b1_w_all[s], a1_w_all[s], b1q_w_all[s], a1q_w_all[s],
            rb_w_all[s], rs_w_all[s],
            b10_w_all[s], a10_w_all[s], bq10_w_all[s], aq10_w_all[s],
            b1_w_all[s + 1], a1_w_all[s + 1], sv_w_all[s + 1], bv_w_all[s + 1],
            w1_ag_c, wr1_c, br1_c, wr2_c, w2m_c, b2m_c, wg1_c, bg1_c, wg2_c, bg2_c, wrec_c, brec_c,
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
            tick_sz, val_env.maker_fee, val_env.taker_fee, true_latency,
        )

    end_b1 = torch.clamp(b1_w_all[win_len - 1], min=tick_sz).unsqueeze(1).expand(W, E).reshape(N)
    rem = pos_qty > 1e-6
    end_cash = torch.clamp(torch.where(rem, cash + pos_qty * end_b1 * (1.0 - val_env.maker_fee), cash), min=1.0)
    end_dd_pct = 100.0 * torch.clamp(peak_mtm_eq - end_cash, min=0.0) / budget
    max_dd_pct = torch.maximum(max_dd_pct, end_dd_pct)

    n_steps_f = float(win_len - 1)
    dlog_eq = torch.log(end_cash / start_mtm_eq).view(W, E)
    mean_log_dd = (sum_log_dd / n_steps_f).view(W, E)
    traj_growth_bp = 10000.0 * (dlog_eq - 0.55 * mean_log_dd)
    mtm_delta_pct = (100.0 * (end_cash - start_mtm_eq) / budget).view(W, E)
    w_dd_mat = max_dd_pct.view(W, E)
    w_fills_mat = n_fills.float().view(W, E)
    w_cancels_mat = n_cancels.float().view(W, E)
    w_requeues_mat = n_requeues.float().view(W, E)
    w_fqp_mat = sum_fqp.view(W, E)
    mode_p_mat = (sum_mode_p / n_steps_f).view(W, E, K_MODES_L27)
    mode_ent_mat = (sum_mode_ent / n_steps_f).view(W, E)

    clean_idx = [wi for wi, (_, _, r0, _) in enumerate(val_windows) if r0 <= 0.01]
    inv_idx = [wi for wi, (_, _, r0, _) in enumerate(val_windows) if r0 > 0.01]

    results = []
    beta_risk = 0.06
    ln2 = math.log(2.0)
    for ci in range(C):
        s_i, e_i = ci * B_bud, (ci + 1) * B_bud
        g_vec = traj_growth_bp[:, s_i:e_i].reshape(-1)
        ent_bp = float((-torch.logsumexp(-beta_risk * g_vec, dim=0) + torch.log(torch.tensor(float(g_vec.numel()), device=device))).item() / beta_risk)
        q_vals = torch.quantile(g_vec, torch.tensor([0.15, 0.50, 0.85], device=device))

        k_cvar = max(1, g_vec.numel() // 4)
        cvar25_bp = float(torch.topk(g_vec, k_cvar, largest=False).values.mean().item())
        mean_g_bp = float(g_vec.mean().item())
        downside_semi = float(torch.sqrt(torch.mean(torch.clamp(-g_vec, min=0.0) ** 2)).item())

        cl_ev = float(mtm_delta_pct[clean_idx, s_i:e_i].mean().item()) if clean_idx else 0.0
        inv_ev = float(mtm_delta_pct[inv_idx, s_i:e_i].mean().item()) if inv_idx else 0.0
        full_ev = float(mtm_delta_pct[:, s_i:e_i].mean().item())
        dd_p = float(w_dd_mat[:, s_i:e_i].max(dim=0).values.mean().item())
        fills = float(w_fills_mat[:, s_i:e_i].sum(dim=0).mean().item())
        cancels = float(w_cancels_mat[:, s_i:e_i].sum(dim=0).mean().item())
        requeues = float(w_requeues_mat[:, s_i:e_i].sum(dim=0).mean().item())
        fqp_s = float(w_fqp_mat[:, s_i:e_i].sum(dim=0).mean().item())

        # Асимметричная геометрическая полезность (Сортино-Келли): вознаграждает чистый + подводный EV,
        # защищает нижний хвост CVaR25 и не отдаёт победу пассивному 0-трейд сидению в кэше!
        geo_score_bp = (
            0.55 * mean_g_bp
            + 0.25 * cvar25_bp
            + 0.20 * float(q_vals[1].item())
            + 35.0 * cl_ev
            + 20.0 * inv_ev
            - 0.18 * downside_semi
        )

        mp_bar = mode_p_mat[:, s_i:e_i, :].reshape(-1, K_MODES_L27).mean(dim=0)
        h_marginal = float(-(mp_bar * torch.log(torch.clamp(mp_bar, min=1e-6))).sum().item())
        h_cond = float(mode_ent_mat[:, s_i:e_i].mean().item())
        mi_bits = max(0.0, (h_marginal - h_cond) / ln2)

        results.append({
            "geo_score_bp": float(geo_score_bp),
            "entropic_growth_bp": ent_bp,
            "cvar25_bp": cvar25_bp,
            "q15_bp": float(q_vals[0].item()),
            "q50_bp": float(q_vals[1].item()),
            "q85_bp": float(q_vals[2].item()),
            "full_ev_pct": full_ev,
            "clean_ev_pct": cl_ev,
            "inv_delta_ev_pct": inv_ev,
            "dd_pct": dd_p,
            "fills": fills,
            "cancels": cancels,
            "requeues": requeues,
            "cnl_per_fill": cancels / max(fills, 0.5),
            "eta_queue_pct": 100.0 * min(1.0, fqp_s / max(fills, 1.0)) if fills >= 0.5 else 100.0,
            "mode_probs": [round(float(x), 4) for x in mp_bar.tolist()],
            "mode_mi_bits": round(mi_bits, 4),
        })
    return results
