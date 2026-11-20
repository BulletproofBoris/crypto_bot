#!/usr/bin/env python3
"""
_tools/fused_l25_regulator_tick.py
=============================================================================
Урок 25: Двухроевой Стратег (480D = 370D Рынок + 110D Агент) +
Послотовый Нейро-Регулятор Очереди (`Shared Per-Slot Micro-Regulator`).

Главные архитектурные новшества Урока 25:
  1. Расширение входной шины до `480D`: добавлены `+36D` каузальных сигналов
     роя 1-го порядка BigONE (`models/prl_bo/`, H=5..120s) в индексы `[444:480]`.
  2. Полная развязка масштаба и геометрии слотов в Макро-Стратеге (`256D -> 35D`):
     - `E_bid = sigmoid(out[0]) ∈ [0, 1]` — целевая доля капитала в риске,
     - `π_bid[12] = sparse_softmax(out[1:13] / τ_b)` — свободные непрерывные доли по всем 12 Bid-слотам,
     - `π_ask[12] = sparse_softmax(out[13:25] / τ_a)` — свободные непрерывные доли по всем 12 Ask-слотам,
     - `panic_prob = sigmoid(out[25])` (`panic_thresh = 0.85`),
     - `ctx_reg[8] = tanh(out[26:34])` + `log_tau = out[34]`.
     Целевой объём каждого Bid-слота рассчитывается от целевого бюджета `budget * E_bid - pos_qty * b1`,
     НЕ завися от того, какие соседние слоты сейчас стоят или пустуют!
  3. Нейросетевой `Shared Per-Slot Micro-Regulator` (`[N, 24, 16] -> 32 -> [N, 24, 2]`):
     Полный отказ от эвристики `upsize_b`, жёстких таймеров (`30s`, `180s`) и лимита `top-3`.
     Регулятор взвешивает прогресс в FIFO-очереди `qprog_s`, непрерывный лог-возраст, сдвиг цены
     и расхождение объёма `(q_target - q_cur)` и выдаёт явное решение по каждому слоту:
     `HOLD_QUEUE`, `CANCEL_NOW` или `REQUOTE` (`1-в-1` с `live_rl_trader.py`).
=============================================================================
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor_cache")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

# 480 каналов головы политики:
# - 370 чисто рыночных каналов (334 прежних + 36 новых каналов роя BigONE [444:480])
# - 110 динамических агентских каналов (портфель, 24 слота, FIFO-очередь 8D + микро-память 16D + послотовый прогресс 24D)
MKT_IDX_L25 = list(range(0, 318)) + list(range(348, 356)) + list(range(380, 388)) + list(range(444, 480))
AG_IDX_L25 = (
    list(range(318, 324))  # multi_6d (6)
    + list(range(324, 348))  # b_slot_12 + a_slot_12 (24)
    + list(range(356, 380))  # b_dist_12 + a_dist_12 (24)
    + list(range(388, 396))  # acct_8d direct (8)
    + list(range(396, 404))  # queue_ahead_8d FIFO (8)
    + list(range(404, 420))  # policy micro-memory mem_h (16)
    + list(range(420, 444))  # per-slot FIFO queue progress 24D (24)
)
assert len(MKT_IDX_L25) == 370 and len(AG_IDX_L25) == 110


@torch.no_grad()
def precompute_window_market_z370(
    flow_mkt_w_all: torch.Tensor,
    lob_fu_w_all: torch.Tensor,
    v7_w_all: torch.Tensor,
    bo36_w_all: torch.Tensor,
    bf16_runner,
) -> torch.Tensor:
    """
    Предвычисляет 370D рыночный ствол (334D GRU+SafeTrade Swarm+CrossVenue + 36D BigONE Swarm)
    для всех `T_steps = win_len - 1` и `W` окон одновременно (`[T_steps, W, 370]` в BF16).
    """
    t_steps, w_cnt, _ = flow_mkt_w_all.shape
    dt = lob_fu_w_all.dtype
    dev = lob_fu_w_all.device

    flow_16_all = torch.zeros((t_steps, w_cnt, 16), dtype=dt, device=dev)
    flow_16_all[:, :, :8] = flow_mkt_w_all[:, :, :8].to(dt)

    fl_all = F.gelu(F.layer_norm(flow_16_all @ bf16_runner.w_fl + bf16_runner.b_fl, [64], bf16_runner.ln_fl_w, bf16_runner.ln_fl_b))
    fu_raw_all = lob_fu_w_all + (fl_all @ bf16_runner.w_fu_fl)
    x_all = F.gelu(F.layer_norm(fu_raw_all, [256], bf16_runner.ln_fu_w, bf16_runner.ln_fu_b))
    gi_all = x_all @ bf16_runner.w_ih + bf16_runner.b_ih

    h = bf16_runner.init_hidden(batch_size=w_cnt)
    h_list = []
    for s in range(t_steps):
        gh = h @ bf16_runner.w_hh + bf16_runner.b_hh
        i_r, i_z, i_n = gi_all[s].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = torch.sigmoid(i_r + h_r)
        z_gate = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1.0 - z_gate) * n + z_gate * h
        h_list.append(h)

    h_stack = torch.stack(h_list, dim=0)  # [T, W, 256]
    n_tw = t_steps * w_cnt
    hh = F.gelu(h_stack.view(n_tw, 256) @ bf16_runner.w_h1 + bf16_runner.b_h1).view(n_tw, 4, 64).transpose(0, 1)
    p4 = torch.sigmoid(torch.bmm(hh, bf16_runner.w_h2).squeeze(-1).t() + bf16_runner.b_h2).view(t_steps, w_cnt, 4)

    v7_58 = v7_w_all[:t_steps, :, :58].to(dt)
    v7_vel = v7_w_all[:t_steps, :, 58:66].to(dt)
    v7_lob = v7_w_all[:t_steps, :, 66:74].to(dt)
    bo_36 = bo36_w_all[:t_steps, :, :36].to(dt)
    return torch.cat([h_stack, p4, v7_58, v7_vel, v7_lob, bo_36], dim=-1).contiguous()


def _sparse_slot_softmax(logits: torch.Tensor, tau: torch.Tensor, floor: float = 0.03) -> torch.Tensor:
    """Непрерывное свободное распределение долей слотов с подавлением микро-пыли < 3%."""
    scaled = logits / tau
    p_raw = torch.softmax(scaled, dim=-1)
    p_keep = torch.where(p_raw >= floor, p_raw, torch.zeros_like(p_raw))
    s_keep = p_keep.sum(dim=-1, keepdim=True)
    return torch.where(s_keep > 1e-5, p_keep / torch.clamp(s_keep, min=1e-6), p_raw)


@torch.compile(dynamic=False)
def fused_l25_regulator_tick(
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
    w2_g: torch.Tensor,
    b2_g: torch.Tensor,
    w_rec_g: torch.Tensor,
    b_rec_g: torch.Tensor,
    w_reg1_g: torch.Tensor,
    b_reg1_g: torch.Tensor,
    w_reg2_g: torch.Tensor,
    b_reg2_g: torch.Tensor,
    slot_tick_offsets: torch.Tensor,
    slot_lob_lvl: torch.Tensor,
    is_penny_slot: torch.Tensor,
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
    peak_mtm_eq: torch.Tensor,
    sum_sq_dd_pct: torch.Tensor,
    max_mtm_dd_pct: torch.Tensor,
    sum_mut_load: torch.Tensor,
    sum_fill_qprog: torch.Tensor,
    n_fills: torch.Tensor,
    n_cancels: torch.Tensor,
    n_requeues: torch.Tensor,
    tick_size: float,
    maker_fee: float,
    taker_fee: float,
    true_latency: bool = True,
    allow_scratch_stop_asks: bool = False,
    panic_thresh: float = 0.85,
):
    W = b1_w0.shape[0]
    G = w1_ag_g.shape[0]
    N = cash.shape[0]
    E = N // W
    B = E // G
    M = W * B
    S = slot_tick_offsets.shape[0]
    dt = mem_h.dtype

    b1_1w = torch.clamp(b1_w0, min=tick_size)
    a1_1w = torch.clamp(a1_w0, min=b1_1w + tick_size)
    b1_q_1w = torch.clamp(b1q_w0, min=0.0)
    a1_q_1w = torch.clamp(a1q_w0, min=0.0)
    l1_bid_usd_1w = torch.clamp(b1_1w * b1_q_1w, min=100.0)
    l1_ask_usd_1w = torch.clamp(a1_1w * a1_q_1w, min=100.0)
    roll_buy_1w = torch.clamp(rb_w0, min=10.0)
    roll_sell_1w = torch.clamp(rs_w0, min=10.0)

    b1 = b1_1w.unsqueeze(1).expand(W, E).reshape(N)
    a1 = a1_1w.unsqueeze(1).expand(W, E).reshape(N)
    b1_q = b1_q_1w.unsqueeze(1).expand(W, E).reshape(N)
    a1_q = a1_q_1w.unsqueeze(1).expand(W, E).reshape(N)
    l1_bid_usd = l1_bid_usd_1w.unsqueeze(1).expand(W, E).reshape(N)
    l1_ask_usd = l1_ask_usd_1w.unsqueeze(1).expand(W, E).reshape(N)
    roll_buy_n = roll_buy_1w.unsqueeze(1).expand(W, E).reshape(N)
    roll_sell_n = roll_sell_1w.unsqueeze(1).expand(W, E).reshape(N)

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
    unr_ticks = torch.where(has_p, ((b1 - avg_entry) / tick_size) * eff_pos_scale, torch.zeros_like(has_pos))
    hold_ratio = torch.where(
        has_p,
        torch.clamp((s_cur_f - entry_s.float()) / 300.0, 0.0, 2.0) * eff_pos_scale,
        torch.zeros_like(has_pos),
    )

    # Естественное сжатие чужой очереди ПЕРЕД расчётом наблюдения и решений Микро-Регулятора:
    cur_b_lvl0 = torch.clamp(torch.round((b1.unsqueeze(1) - bid_px) / tick_size).long(), 0, 9)
    bids_q_1w_raw = torch.clamp(bids_q_10_w[:, :10], min=0.0)
    vis_b_lvl_q0 = torch.gather(bids_q_1w_raw, 1, cur_b_lvl0.view(W, E * S)).view(N, S)
    in_book_b0 = (bid_qty > 1e-6) & (bid_px <= b1.unsqueeze(1) + 0.1 * tick_size)
    bid_qa = torch.where(in_book_b0, torch.minimum(bid_qa, vis_b_lvl_q0), bid_qa)

    cur_a_lvl0 = torch.clamp(torch.round((ask_px - a1.unsqueeze(1)) / tick_size).long(), 0, 9)
    asks_q_1w_raw = torch.clamp(asks_q_10_w[:, :10], min=0.0)
    vis_a_lvl_q0 = torch.gather(asks_q_1w_raw, 1, cur_a_lvl0.view(W, E * S)).view(N, S)
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
            torch.clamp(unr_ticks / 5.0, -2.0, 2.0),
            hold_ratio,
            has_bid,
            has_ask,
            n_bid_slots - n_ask_slots,
        ],
        dim=-1,
    ).to(dt)

    b_qa_rat_all = torch.where(
        act_b_mask,
        torch.clamp(bid_qa / (bid_qa0.float() + 10.0), 0.0, 1.5),
        torch.full_like(bid_qa, 2.0),
    )
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

    a_qa_rat_all = torch.where(
        act_a_mask,
        torch.clamp(ask_qa / (ask_qa0.float() + 10.0), 0.0, 1.5),
        torch.full_like(ask_qa, 2.0),
    )
    best_a_idx = torch.argmin(torch.where(act_a_mask, ask_px - a1.unsqueeze(1), torch.full_like(ask_px, 999.0)), dim=1, keepdim=True)
    best_a_qa = torch.gather(ask_qa, 1, best_a_idx).squeeze(1)
    best_a_qa0 = torch.gather(ask_qa0.float(), 1, best_a_idx).squeeze(1)
    best_a_act = torch.gather(ask_act_s, 1, best_a_idx).squeeze(1).float()
    best_a_rat = torch.where(has_ask > 0.5, torch.clamp(best_a_qa / (best_a_qa0 + 10.0), 0.0, 1.5), torch.ones_like(has_ask))
    best_a_tape = torch.where(has_ask > 0.5, torch.clamp(torch.log1p(best_a_qa / roll_buy_n) / 3.0, 0.0, 2.0), torch.zeros_like(has_ask))
    best_a_age = torch.where(has_ask > 0.5, torch.clamp((s_cur_f - best_a_act) / 120.0, 0.0, 2.0), torch.zeros_like(has_ask))
    q_imb_adv = (1.0 - torch.clamp(best_a_rat, 0.0, 1.0)) * has_ask - (1.0 - torch.clamp(best_b_rat, 0.0, 1.0)) * has_bid

    queue_8d = torch.stack(
        [best_b_rat, best_b_tape, best_b_age, deep_b_rat, best_a_rat, best_a_tape, best_a_age, q_imb_adv],
        dim=-1,
    ).to(dt)

    bud_l1 = torch.log1p(budget / l1_bid_usd) / 3.0
    cash_l1 = torch.log1p(unreserved_cash_now / l1_bid_usd) / 3.0
    pos_l1_ask = torch.clamp((pos_qty * a1) / l1_ask_usd, 0.0, 4.0) / 2.0
    pos_tape = torch.clamp(pos_qty / roll_buy_n, 0.0, 10.0) / 5.0
    low_tr_ratio = torch.where(
        has_p, torch.clamp(low_tranche_qty / torch.clamp(pos_qty, min=1e-6), 0.0, 1.0), torch.zeros_like(has_pos)
    )
    lifo_edge_ticks = torch.where(
        has_p, torch.clamp((b1 - min_buy_px) / (3.0 * tick_size), -2.0, 2.0), torch.zeros_like(has_pos)
    )
    multi_6d = torch.stack([bud_l1, cash_l1, pos_l1_ask, pos_tape, low_tr_ratio, lifo_edge_ticks], dim=-1).to(dt)

    b_slot_12 = torch.clamp((bid_qty * b1.unsqueeze(1)) / torch.clamp(budget.unsqueeze(1), min=25.0), 0.0, 1.0).to(dt)
    a_slot_12 = torch.clamp(ask_qty / torch.clamp(pos_qty.unsqueeze(1), min=2.0), 0.0, 1.0).to(dt)
    b_dist_12 = torch.where(
        act_b_mask,
        torch.clamp((b1.unsqueeze(1) - bid_px) / (5.0 * tick_size), -0.5, 1.5),
        torch.zeros_like(bid_qty),
    ).to(dt)
    a_dist_12 = torch.where(
        act_a_mask,
        torch.clamp((ask_px - a1.unsqueeze(1)) / (5.0 * tick_size), -0.5, 1.5),
        torch.zeros_like(ask_qty),
    ).to(dt)

    b_qprog_12_f = torch.where(act_b_mask, torch.clamp(1.0 - b_qa_rat_all, -0.5, 1.0), torch.full_like(bid_qty, -1.0))
    a_qprog_12_f = torch.where(act_a_mask, torch.clamp(1.0 - a_qa_rat_all, -0.5, 1.0), torch.full_like(ask_qty, -1.0))
    b_qprog_12 = b_qprog_12_f.to(dt)
    a_qprog_12 = a_qprog_12_f.to(dt)

    # 1. Прямой проход Макро-Стратега (480D -> 256D -> ResSwiGLU -> 35D)
    z_ag = torch.cat([multi_6d, b_slot_12, a_slot_12, b_dist_12, a_dist_12, acct_8d, queue_8d, mem_h, b_qprog_12, a_qprog_12], dim=-1)
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

    strat_raw_g = (torch.bmm(h2, w2_g) + b2_g).float()
    strat_raw = strat_raw_g.view(G, W, B, 35).permute(1, 0, 2, 3).reshape(N, 35)

    exp_bid_frac = torch.sigmoid(strat_raw[:, 0])
    tau_slot = 0.30 + 1.70 * torch.sigmoid(strat_raw[:, 34:35])
    pi_bid = _sparse_slot_softmax(strat_raw[:, 1:13], tau_slot, floor=0.03)
    pi_ask = _sparse_slot_softmax(strat_raw[:, 13:25], tau_slot, floor=0.03)
    panic_prob = torch.sigmoid(strat_raw[:, 25])
    ctx_4d_b = torch.tanh(strat_raw[:, 26:30])
    ctx_4d_a = torch.tanh(strat_raw[:, 30:34])

    # 2. Экстренный Panic Taker Exit при структурном обвале (> 0.85)
    is_panic = (panic_prob > panic_thresh) & (pos_qty > 1e-6)
    panic_frac = torch.clamp((panic_prob - 0.80) * 5.0, 0.25, 1.0)
    raw_sell_q = pos_qty * panic_frac
    sell_q_panic = torch.where((pos_qty - raw_sell_q) < 2.0, pos_qty, raw_sell_q)

    b_p10_1w = torch.clamp(bids_10_w[:, :10], min=tick_size)
    b_q10_1w = torch.clamp(bids_q_10_w[:, :10], min=1.0).clone()
    b_q10_1w[:, 9] = 1e9
    cum_q10_1w = torch.cumsum(b_q10_1w, dim=1)
    prev_q10_1w = cum_q10_1w - b_q10_1w

    b_p10 = b_p10_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    b_q10 = b_q10_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)
    prev_q10 = prev_q10_1w.unsqueeze(1).expand(W, E, 10).reshape(N, 10)

    take_10 = torch.clamp(torch.minimum(sell_q_panic.unsqueeze(1) - prev_q10, b_q10), min=0.0)
    proceeds = (take_10 * b_p10).sum(dim=1) * (1.0 - taker_fee)
    rem_pos = torch.clamp(pos_qty - sell_q_panic, min=0.0)
    rem_frac = torch.where(pos_qty > 1e-6, rem_pos / torch.clamp(pos_qty, min=1e-6), torch.zeros_like(pos_qty))
    full_exit = rem_pos <= 1e-5

    cash = torch.where(is_panic, cash + proceeds, cash)
    pos_qty = torch.where(is_panic, rem_pos, pos_qty)
    pos_cost_usd = torch.where(is_panic, pos_cost_usd * rem_frac, pos_cost_usd)
    avg_entry = torch.where(is_panic & full_exit, torch.zeros_like(avg_entry), avg_entry)
    min_buy_px = torch.where(is_panic & full_exit, torch.zeros_like(min_buy_px), min_buy_px)
    low_tranche_qty = torch.where(is_panic, low_tranche_qty * rem_frac, low_tranche_qty)
    mid_buy_px = torch.where(is_panic & full_exit, torch.zeros_like(mid_buy_px), mid_buy_px)
    mid_tranche_qty = torch.where(is_panic, mid_tranche_qty * rem_frac, mid_tranche_qty)
    had_ask_panic_cnt = torch.where(is_panic, (ask_qty > 1e-6).long().sum(dim=1), torch.zeros(N, dtype=torch.long, device=ask_qty.device))
    ask_qty = torch.where(is_panic.unsqueeze(1), torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(is_panic.unsqueeze(1), torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(is_panic.unsqueeze(1), torch.zeros_like(ask_qa), ask_qa)
    ask_qa0 = torch.where(is_panic.unsqueeze(1), torch.zeros_like(ask_qa0), ask_qa0)
    n_fills = torch.clamp(n_fills.long() + is_panic.long(), max=30000).to(torch.int16)

    # 3. Расчёт свободных целевых цен и объёмов Q_target (НЕ зависящих от стоящих соседних ордеров!)
    wide_sp_1w = (a1_1w - b1_1w) >= 2.0 * tick_size
    penny_add_1w = torch.where(wide_sp_1w, torch.full_like(b1_1w, tick_size), torch.zeros_like(b1_1w))
    t_bid_px_1w = torch.where(
        is_penny_slot.unsqueeze(0),
        (b1_1w + penny_add_1w).unsqueeze(1),
        torch.clamp(b1_1w.unsqueeze(1) - slot_tick_offsets.unsqueeze(0), min=tick_size),
    )
    t_bid_px = t_bid_px_1w.unsqueeze(1).expand(W, E, S).reshape(N, S)

    tot_bud_eff = torch.maximum(budget, cash + pos_qty * b1)
    target_bid_pool_usd = torch.clamp(tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1, min=0.0)
    q_target_b = (target_bid_pool_usd.unsqueeze(1) * pi_bid) / (t_bid_px * (1.0 + maker_fee))
    # Укрупнение внутри ценовых пар (2k, 2k+1), если сумма >= 2.0 PRL, а один из парных слотов < 2.0 PRL
    pair_small_b = ((q_target_b[:, 0::2] > 0.05) & (q_target_b[:, 0::2] < 2.0)) | ((q_target_b[:, 1::2] > 0.05) & (q_target_b[:, 1::2] < 2.0))
    pair_comb_b = q_target_b[:, 0::2] + q_target_b[:, 1::2]
    even_win_b = pi_bid[:, 0::2] >= pi_bid[:, 1::2]
    qb_ev = torch.where(pair_small_b, torch.where(even_win_b, pair_comb_b, torch.zeros_like(pair_comb_b)), q_target_b[:, 0::2])
    qb_od = torch.where(pair_small_b, torch.where(even_win_b, torch.zeros_like(pair_comb_b), pair_comb_b), q_target_b[:, 1::2])
    q_target_b = torch.stack([qb_ev, qb_od], dim=2).reshape(N, S)
    q_target_b = torch.where(q_target_b >= 2.0, q_target_b, torch.zeros_like(q_target_b))

    hp_now = pos_qty >= 1.95
    hp_now_mask = hp_now.unsqueeze(1)
    has_low_tr = hp_now & (low_tranche_qty >= 1.95) & (min_buy_px < avg_entry - 0.1 * tick_size)
    avg_tp_tick = torch.ceil((avg_entry + 0.55 * tick_size) / tick_size - 1e-5) * tick_size
    lifo_tp_tick = torch.ceil((min_buy_px + 0.55 * tick_size) / tick_size - 1e-5) * tick_size
    base_avg_tp = torch.where(hp_now, avg_tp_tick, a1)
    base_lifo_tp = torch.where(has_low_tr, lifo_tp_tick, base_avg_tp)

    if allow_scratch_stop_asks:
        avg_round_tick = torch.clamp(torch.round(avg_entry / tick_size) * tick_size, min=tick_size)
        stop_1_px = torch.where(hp_now, torch.clamp(avg_round_tick - tick_size, min=tick_size), base_avg_tp).unsqueeze(1)
        scratch_0_px = torch.where(hp_now, avg_round_tick, base_avg_tp).unsqueeze(1)
        slot_base_tp = torch.cat([stop_1_px, scratch_0_px, base_lifo_tp.unsqueeze(1).expand(N, 4), base_avg_tp.unsqueeze(1).expand(N, S - 6)], dim=1)
    else:
        slot_base_tp = torch.cat([base_lifo_tp.unsqueeze(1).expand(N, 6), base_avg_tp.unsqueeze(1).expand(N, S - 6)], dim=1)

    is_ask_penny = (is_penny_slot | (slot_lob_lvl == 0)).unsqueeze(0)
    raw_slot_a_px_1w = torch.where(
        is_ask_penny,
        (a1_1w - penny_add_1w).unsqueeze(1),
        a1_1w.unsqueeze(1) + slot_tick_offsets.unsqueeze(0),
    )
    raw_slot_a_px = raw_slot_a_px_1w.unsqueeze(1).expand(W, E, S).reshape(N, S)
    t_ask_px = torch.maximum(raw_slot_a_px, slot_base_tp + slot_tick_offsets.unsqueeze(0))

    q_target_a = torch.where(hp_now_mask, pos_qty.unsqueeze(1) * pi_ask, torch.zeros_like(pi_ask))
    need_lifo_cap = has_low_tr & (base_lifo_tp < base_avg_tp - 0.1 * tick_size)
    if allow_scratch_stop_asks:
        want_low_sum = q_target_a[:, 2:6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        low_cap_scale = torch.where(need_lifo_cap.unsqueeze(1) & (want_low_sum > low_tranche_qty.unsqueeze(1)), low_tranche_qty.unsqueeze(1) / want_low_sum, torch.ones_like(want_low_sum))
        spill_q = (q_target_a[:, 2:6] * (1.0 - low_cap_scale)).sum(dim=1, keepdim=True)
        core_pi_sum = pi_ask[:, 6:].sum(dim=1, keepdim=True).clamp(min=1e-6)
        q_target_a = torch.cat([q_target_a[:, :2], q_target_a[:, 2:6] * low_cap_scale, q_target_a[:, 6:] + spill_q * (pi_ask[:, 6:] / core_pi_sum)], dim=1)
    else:
        want_low_sum = q_target_a[:, :6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        low_cap_scale = torch.where(need_lifo_cap.unsqueeze(1) & (want_low_sum > low_tranche_qty.unsqueeze(1)), low_tranche_qty.unsqueeze(1) / want_low_sum, torch.ones_like(want_low_sum))
        spill_q = (q_target_a[:, :6] * (1.0 - low_cap_scale)).sum(dim=1, keepdim=True)
        core_pi_sum = pi_ask[:, 6:].sum(dim=1, keepdim=True).clamp(min=1e-6)
        q_target_a = torch.cat([q_target_a[:, :6] * low_cap_scale, q_target_a[:, 6:] + spill_q * (pi_ask[:, 6:] / core_pi_sum)], dim=1)

    dust_a_mask = (q_target_a > 0.0) & (q_target_a < 2.0)
    dust_a_sum = torch.where(dust_a_mask, q_target_a, torch.zeros_like(q_target_a)).sum(dim=1, keepdim=True)
    q_target_a = torch.where(dust_a_mask, torch.zeros_like(q_target_a), q_target_a)
    best_a_t_idx = torch.argmax(pi_ask, dim=1, keepdim=True)
    best_a_oh = torch.zeros_like(q_target_a).scatter_(1, best_a_t_idx, 1.0)
    q_target_a = torch.where(hp_now_mask & (dust_a_sum >= 0.01) & (best_a_oh > 0.5), torch.minimum(pos_qty.unsqueeze(1), q_target_a + dust_a_sum), q_target_a)
    q_target_a = torch.where(q_target_a >= 2.0, q_target_a, torch.zeros_like(q_target_a))

    # 4. Послотовый Нейро-Регулятор Очереди (Shared Per-Slot Micro-Regulator [N, 24, 16] -> [N, 24, 2])
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

    drift_b = torch.where(had_b_pre, torch.clamp((t_bid_px - bid_px) / tick_size, -5.0, 5.0) / 5.0, torch.zeros_like(bid_qty))
    drift_a = torch.where(had_a_pre, torch.clamp((ask_px - t_ask_px) / tick_size, -5.0, 5.0) / 5.0, torch.zeros_like(ask_qty))

    qprog_b_01 = torch.where(had_b_pre, torch.clamp(1.0 - b_qa_rat_all, 0.0, 1.0), torch.zeros_like(bid_qty))
    qprog_a_01 = torch.where(had_a_pre, torch.clamp(1.0 - a_qa_rat_all, 0.0, 1.0), torch.zeros_like(ask_qty))
    log_qa_b = torch.where(had_b_pre, torch.log1p(torch.clamp(bid_qa, min=0.0)) / 10.0, torch.zeros_like(bid_qty))
    log_qa_a = torch.where(had_a_pre, torch.log1p(torch.clamp(ask_qa, min=0.0)) / 10.0, torch.zeros_like(ask_qty))

    depth_norm_1s = (slot_tick_offsets / (6.0 * tick_size)).unsqueeze(0).expand(N, S)
    front_1s = (torch.arange(S, device=bid_px.device) < 6).float().unsqueeze(0).expand(N, S)
    free_b_2d = free_cash_ratio.unsqueeze(1).expand(N, S)
    free_a_2d = torch.clamp((pos_qty - ask_qty.sum(dim=1)) / torch.clamp(pos_qty, min=2.0), 0.0, 1.0).unsqueeze(1).expand(N, S)

    feat_b_16 = torch.cat(
        [
            torch.stack(
                [
                    torch.zeros_like(bid_qty),
                    had_b_pre.float(),
                    cur_val_b,
                    targ_val_b,
                    rel_delta_b,
                    drift_b,
                    qprog_b_01,
                    log_qa_b,
                    age_b_norm,
                    depth_norm_1s,
                    front_1s,
                    free_b_2d,
                ],
                dim=-1,
            ),
            ctx_4d_b.unsqueeze(1).expand(N, S, 4),
        ],
        dim=-1,
    )
    feat_a_16 = torch.cat(
        [
            torch.stack(
                [
                    torch.ones_like(ask_qty),
                    had_a_pre.float(),
                    cur_val_a,
                    targ_val_a,
                    rel_delta_a,
                    drift_a,
                    qprog_a_01,
                    log_qa_a,
                    age_a_norm,
                    depth_norm_1s,
                    front_1s,
                    free_a_2d,
                ],
                dim=-1,
            ),
            ctx_4d_a.unsqueeze(1).expand(N, S, 4),
        ],
        dim=-1,
    )
    feat_24_16 = torch.cat([feat_b_16, feat_a_16], dim=1).to(dt)  # [N, 24, 16]
    feat_reg_g = feat_24_16.view(W, G, B, 2 * S, 16).permute(1, 0, 2, 3, 4).reshape(G, M * 2 * S, 16)
    reg_h = F.silu(torch.bmm(feat_reg_g, w_reg1_g) + b_reg1_g)
    reg_out_g = (torch.bmm(reg_h, w_reg2_g) + b_reg2_g).float()
    reg_out = reg_out_g.view(G, W, B, 2 * S, 2).permute(1, 0, 2, 3, 4).reshape(N, 2 * S, 2)

    g_plc_b = reg_out[:, :S, 0] > 0.0
    g_cnl_b = reg_out[:, :S, 1] > 0.0
    g_plc_a = reg_out[:, S:, 0] > 0.0
    g_cnl_a = reg_out[:, S:, 1] > 0.0

    # 5. Исполнение решений Микро-Регулятора по 12 Bid-слотам (HOLD_QUEUE / CANCEL_NOW / REQUOTE / PLACE)
    clear_b = had_b_pre & g_cnl_b
    requeue_b = clear_b & g_plc_b & (q_target_b >= 2.0)
    n_cnl_b_step = clear_b.long().sum(dim=1)
    n_req_b_step = requeue_b.long().sum(dim=1)

    bid_qty = torch.where(clear_b, torch.zeros_like(bid_qty), bid_qty)
    bid_px = torch.where(clear_b, torch.zeros_like(bid_px), bid_px)
    bid_qa = torch.where(clear_b, torch.zeros_like(bid_qa), bid_qa)
    bid_qa0 = torch.where(clear_b, torch.zeros_like(bid_qa0), bid_qa0)

    res_cash_after_cnl = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    free_cash_for_b = torch.clamp(cash * 0.998 - res_cash_after_cnl, min=0.0)
    rem_exp_for_b = torch.clamp(tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1 - res_cash_after_cnl, min=0.0)
    avail_b_usd = torch.minimum(free_cash_for_b, rem_exp_for_b)

    empty_b_now = bid_qty <= 1e-6
    want_place_b = empty_b_now & g_plc_b & (q_target_b >= 2.0)
    want_b_usd = torch.where(want_place_b, q_target_b * t_bid_px * (1.0 + maker_fee), torch.zeros_like(q_target_b))
    tot_want_b_usd = want_b_usd.sum(dim=1, keepdim=True)
    scale_cap_b = torch.where(
        tot_want_b_usd > avail_b_usd.unsqueeze(1),
        avail_b_usd.unsqueeze(1) / torch.clamp(tot_want_b_usd, min=1e-6),
        torch.ones_like(tot_want_b_usd),
    )
    plc_b_qty = (want_b_usd * scale_cap_b) / (t_bid_px * (1.0 + maker_fee))
    valid_b = want_place_b & (plc_b_qty >= 2.0)
    plc_b_qty = torch.where(valid_b, plc_b_qty, torch.zeros_like(plc_b_qty))
    n_plc_b_step = valid_b.long().sum(dim=1)

    eff_b_lvl = torch.clamp(torch.round((b1.unsqueeze(1) - t_bid_px) / tick_size).long(), 0, 9)
    bids_q_1w = torch.clamp(bids_q_10_w[:, :10], min=10.0)
    lvl_q_2d = torch.gather(bids_q_1w, 1, eff_b_lvl.view(W, E * S)).view(N, S)
    inside_sp_b = t_bid_px > (b1.unsqueeze(1) + 0.1 * tick_size)
    base_b_qa = torch.where(inside_sp_b, torch.zeros_like(lvl_q_2d), lvl_q_2d)

    plc_b_usd = plc_b_qty * t_bid_px
    b_over50 = torch.sigmoid((plc_b_usd - 50.0) / 18.0)
    b_self_impact_qa = base_b_qa * (1.0 + 0.75 * b_over50) + b_over50 * (0.45 * plc_b_qty)

    bid_px = torch.where(valid_b, t_bid_px, bid_px)
    bid_qty = torch.where(valid_b, plc_b_qty, bid_qty)
    bid_qa = torch.where(valid_b, b_self_impact_qa, bid_qa)
    bid_qa0 = torch.where(valid_b, b_self_impact_qa.to(dt), bid_qa0)
    bid_act_s = torch.where(valid_b, act_lat_2d.expand_as(bid_act_s), bid_act_s)

    # 6. Исполнение решений Микро-Регулятора по 12 Ask-слотам (с защитой безубытка under_tp_a)
    under_tp_a = had_a_pre & (ask_px < slot_base_tp - 0.4 * tick_size)
    clear_a = had_a_pre & ((~hp_now_mask) | under_tp_a | g_cnl_a)
    requeue_a = clear_a & hp_now_mask & g_plc_a & (q_target_a >= 2.0)
    n_cnl_a_step = clear_a.long().sum(dim=1) + had_ask_panic_cnt
    n_req_a_step = requeue_a.long().sum(dim=1)

    ask_qty = torch.where(clear_a, torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(clear_a, torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(clear_a, torch.zeros_like(ask_qa), ask_qa)
    ask_qa0 = torch.where(clear_a, torch.zeros_like(ask_qa0), ask_qa0)

    unreserved_pos = torch.clamp(pos_qty - ask_qty.sum(dim=1), min=0.0).unsqueeze(1)
    empty_a_now = ask_qty <= 1e-6
    want_place_a = empty_a_now & hp_now_mask & g_plc_a & (q_target_a >= 2.0)
    want_a_q = torch.where(want_place_a, q_target_a, torch.zeros_like(q_target_a))

    if allow_scratch_stop_asks:
        lifo_placed = ask_qty[:, 2:6].sum(dim=1, keepdim=True)
        rem_low_tr = torch.clamp(low_tranche_qty.unsqueeze(1) - lifo_placed, min=0.0)
        w_low_s = want_a_q[:, 2:6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        l_sc = torch.where(need_lifo_cap.unsqueeze(1) & (w_low_s > rem_low_tr), rem_low_tr / w_low_s, torch.ones_like(w_low_s))
        want_a_q = torch.cat([want_a_q[:, :2], want_a_q[:, 2:6] * l_sc, want_a_q[:, 6:]], dim=1)
    else:
        lifo_placed = ask_qty[:, :6].sum(dim=1, keepdim=True)
        rem_low_tr = torch.clamp(low_tranche_qty.unsqueeze(1) - lifo_placed, min=0.0)
        w_low_s = want_a_q[:, :6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        l_sc = torch.where(need_lifo_cap.unsqueeze(1) & (w_low_s > rem_low_tr), rem_low_tr / w_low_s, torch.ones_like(w_low_s))
        want_a_q = torch.cat([want_a_q[:, :6] * l_sc, want_a_q[:, 6:]], dim=1)

    tot_want_a_q = want_a_q.sum(dim=1, keepdim=True)
    scale_cap_a = torch.where(tot_want_a_q > unreserved_pos, unreserved_pos / torch.clamp(tot_want_a_q, min=1e-6), torch.ones_like(tot_want_a_q))
    plc_a_qty = want_a_q * scale_cap_a
    valid_a = want_place_a & (plc_a_qty >= 2.0)
    plc_a_qty = torch.where(valid_a, plc_a_qty, torch.zeros_like(plc_a_qty))
    n_plc_a_step = valid_a.long().sum(dim=1)

    eff_a_lvl = torch.clamp(torch.round((t_ask_px - a1.unsqueeze(1)) / tick_size).long(), 0, 9)
    asks_q_1w = torch.clamp(asks_q_10_w[:, :10], min=10.0)
    lvl_a_q_2d = torch.gather(asks_q_1w, 1, eff_a_lvl.view(W, E * S)).view(N, S)
    inside_sp_a = t_ask_px < (a1.unsqueeze(1) - 0.1 * tick_size)
    base_a_qa = torch.where(inside_sp_a, torch.zeros_like(lvl_a_q_2d), lvl_a_q_2d)

    plc_a_usd = plc_a_qty * t_ask_px
    a_over50 = torch.sigmoid((plc_a_usd - 50.0) / 18.0)
    a_self_impact_qa = base_a_qa * (1.0 + 0.75 * a_over50) + a_over50 * (0.45 * plc_a_qty)

    ask_px = torch.where(valid_a, t_ask_px, ask_px)
    ask_qty = torch.where(valid_a, plc_a_qty, ask_qty)
    ask_qa = torch.where(valid_a, a_self_impact_qa, ask_qa)
    ask_qa0 = torch.where(valid_a, a_self_impact_qa.to(dt), ask_qa0)
    ask_act_s = torch.where(valid_a, act_lat_2d.expand_as(ask_act_s), ask_act_s)

    n_mut_tot = n_cnl_b_step + n_cnl_a_step + n_plc_b_step + n_plc_a_step
    if true_latency:
        burst_delay = (n_mut_tot > 4).unsqueeze(1)
        bid_act_s = torch.where(valid_b & burst_delay, (next_s_2d + 2).expand_as(bid_act_s), bid_act_s)
        ask_act_s = torch.where(valid_a & burst_delay, (next_s_2d + 2).expand_as(ask_act_s), ask_act_s)

    # 7. Честное исполнение FIFO-очередей и 3-траншевый LIFO-учёт сделок (1-в-1 с live_rl_trader.py)
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
    Incremental_qa_b = bid_qa.clone()
    Incremental_qa_b[:, 1::2] = torch.where(
        even_act_b, torch.clamp(bid_qa[:, 1::2] - bid_qa[:, 0::2], min=0.0), bid_qa[:, 1::2]
    )
    eff_qa_b = torch.where(hit_b, Incremental_qa_b, torch.zeros_like(bid_qa))
    eff_qty_b = torch.where(hit_b, bid_qty, torch.zeros_like(bid_qty))

    slot_need_b = eff_qa_b + eff_qty_b
    cum_need_b = torch.cumsum(slot_need_b, dim=1)
    flow_before_slot_b = cum_need_b - slot_need_b

    b_q10_lvl_1w = torch.clamp(bids_q_10_w[:, :10], min=0.0)
    cum_bq_1w = torch.cat([torch.zeros((W, 1), dtype=torch.float32, device=bid_px.device), torch.cumsum(b_q10_lvl_1w, dim=1)], dim=1)
    cnt_above_b = torch.searchsorted(-b_p10_1w, -(bid_px.view(W, E * S) + 0.5 * tick_size), right=False)
    mkt_above_b = torch.gather(cum_bq_1w, 1, cnt_above_b).view(N, S)

    rem_flow_at_slot_b = torch.clamp(tot_flow_b - torch.maximum(flow_before_slot_b, mkt_above_b), min=0.0)

    pre_fill_qp_b = torch.clamp(1.0 - bid_qa / (bid_qa0.float() + 10.0), 0.0, 1.0)
    qa_eaten_b = torch.minimum(eff_qa_b, rem_flow_at_slot_b)
    bid_qa = torch.where(hit_b, torch.clamp(bid_qa - qa_eaten_b, min=0.0), bid_qa)
    bid_qa[:, 1::2] = torch.where(
        act_b[:, 1::2] & even_act_b, torch.clamp(bid_qa[:, 1::2] - qa_eaten_b[:, 0::2], min=0.0), bid_qa[:, 1::2]
    )

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

    new_mid_q = torch.where(
        first_buy,
        torch.zeros_like(mid_tranche_qty),
        torch.where(deeper_dip, pushed_mid_q, torch.where(mid_dip, added_mid_q, mid_tranche_qty)),
    )
    new_mid_px = torch.where(
        first_buy,
        torch.zeros_like(mid_buy_px),
        torch.where(deeper_dip, pushed_mid_px, torch.where(mid_dip, added_mid_px, mid_buy_px)),
    )

    same_low_q = low_tranche_qty + tot_fq
    same_low_px = (low_tranche_qty * min_buy_px + tot_fq * avg_fill_b_px) / torch.clamp(same_low_q, min=1e-6)

    low_tranche_qty = torch.where(
        first_buy,
        torch.zeros_like(low_tranche_qty),
        torch.where(
            init_dip | deeper_dip,
            tot_fq,
            torch.where(init_higher, old_q, torch.where(same_low, same_low_q, low_tranche_qty)),
        ),
    )
    min_buy_px = torch.where(
        first_buy | init_dip | deeper_dip,
        avg_fill_b_px,
        torch.where(init_higher, avg_entry, torch.where(same_low, same_low_px, min_buy_px)),
    )
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
    Incremental_qa_a = ask_qa.clone()
    Incremental_qa_a[:, 1::2] = torch.where(
        even_act_a, torch.clamp(ask_qa[:, 1::2] - ask_qa[:, 0::2], min=0.0), ask_qa[:, 1::2]
    )
    eff_qa_a = torch.where(hit_a, Incremental_qa_a, torch.zeros_like(ask_qa))
    eff_qty_a = torch.where(hit_a, ask_qty, torch.zeros_like(ask_qty))

    slot_need_a = eff_qa_a + eff_qty_a
    cum_need_a = torch.cumsum(slot_need_a, dim=1)
    flow_before_slot_a = cum_need_a - slot_need_a

    a_p10_1w = torch.clamp(asks_10_w[:, :10], min=b1_1w.unsqueeze(1) + tick_size)
    a_q10_lvl_1w = torch.clamp(asks_q_10_w[:, :10], min=0.0)
    cum_aq_1w = torch.cat([torch.zeros((W, 1), dtype=torch.float32, device=ask_px.device), torch.cumsum(a_q10_lvl_1w, dim=1)], dim=1)
    cnt_below_a = torch.searchsorted(a_p10_1w, ask_px.view(W, E * S) - 0.5 * tick_size, right=False)
    mkt_below_a = torch.gather(cum_aq_1w, 1, cnt_below_a).view(N, S)

    rem_flow_at_slot_a = torch.clamp(tot_flow_a - torch.maximum(flow_before_slot_a, mkt_below_a), min=0.0)

    pre_fill_qp_a = torch.clamp(1.0 - ask_qa / (ask_qa0.float() + 10.0), 0.0, 1.0)
    qa_eaten_a = torch.minimum(eff_qa_a, rem_flow_at_slot_a)
    ask_qa = torch.where(hit_a, torch.clamp(ask_qa - qa_eaten_a, min=0.0), ask_qa)
    ask_qa[:, 1::2] = torch.where(
        act_a[:, 1::2] & even_act_a, torch.clamp(ask_qa[:, 1::2] - qa_eaten_a[:, 0::2], min=0.0), ask_qa[:, 1::2]
    )

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

    lifo_f_qa = f_qa[:, 2:6].sum(dim=1) if allow_scratch_stop_asks else f_qa[:, :6].sum(dim=1)
    sold_low_q = torch.minimum(lifo_f_qa, low_tranche_qty)
    rem_lifo_qa = torch.clamp(lifo_f_qa - sold_low_q, min=0.0)
    sold_mid_q = torch.minimum(rem_lifo_qa, mid_tranche_qty)
    sold_high_q = torch.clamp(tot_fqa - sold_low_q - sold_mid_q, min=0.0)

    core_qty_pre = torch.clamp(pos_qty - low_tranche_qty - mid_tranche_qty, min=0.0)
    low_mid_cost_pre = (low_tranche_qty * min_buy_px + mid_tranche_qty * mid_buy_px) * (1.0 + maker_fee)
    core_cost_pre = torch.clamp(pos_cost_usd - low_mid_cost_pre, min=0.0)
    core_unit_cost = torch.where(
        core_qty_pre > 1e-5,
        core_cost_pre / torch.clamp(core_qty_pre, min=1e-6),
        avg_entry * (1.0 + maker_fee),
    )
    sold_cost_usd = torch.minimum(
        pos_cost_usd,
        sold_low_q * min_buy_px * (1.0 + maker_fee)
        + sold_mid_q * mid_buy_px * (1.0 + maker_fee)
        + sold_high_q * core_unit_cost,
    )

    cash = torch.where(has_sell, cash + tot_rev, cash)
    rem_p = torch.clamp(pos_qty - tot_fqa, min=0.0)
    pos_qty = torch.where(has_sell, rem_p, pos_qty)
    pos_cost_usd = torch.where(
        has_sell,
        torch.where(rem_p > 1e-6, torch.clamp(pos_cost_usd - sold_cost_usd, min=0.0), torch.zeros_like(pos_cost_usd)),
        pos_cost_usd,
    )
    avg_entry = torch.where(
        has_sell,
        torch.where(rem_p > 1e-6, pos_cost_usd / (rem_p * (1.0 + maker_fee)), torch.zeros_like(avg_entry)),
        avg_entry,
    )
    rem_low_q = torch.where(
        has_sell,
        torch.where(rem_p > 1e-6, torch.clamp(low_tranche_qty - sold_low_q, min=0.0), torch.zeros_like(low_tranche_qty)),
        low_tranche_qty,
    )
    rem_mid_q = torch.where(
        has_sell,
        torch.where(rem_p > 1e-6, torch.clamp(mid_tranche_qty - sold_mid_q, min=0.0), torch.zeros_like(mid_tranche_qty)),
        mid_tranche_qty,
    )
    promote_mid = has_sell & (rem_p > 1e-6) & (rem_low_q < 1.95) & (rem_mid_q >= 1.95)
    prom_q = rem_low_q + rem_mid_q
    prom_px = (rem_low_q * min_buy_px + rem_mid_q * mid_buy_px) / torch.clamp(prom_q, min=1e-6)

    low_tranche_qty = torch.where(promote_mid, prom_q, rem_low_q)
    min_buy_px = torch.where(
        pos_qty > 1e-6,
        torch.where(promote_mid, prom_px, torch.where(low_tranche_qty > 1e-4, min_buy_px, avg_entry)),
        torch.zeros_like(min_buy_px),
    )
    mid_tranche_qty = torch.where(promote_mid | (pos_qty <= 1e-6), torch.zeros_like(rem_mid_q), rem_mid_q)
    mid_buy_px = torch.where((pos_qty > 1e-6) & (mid_tranche_qty > 1e-4), mid_buy_px, torch.zeros_like(mid_buy_px))

    rem_ord_a = torch.clamp(ask_qty - f_qa, min=0.0)
    ask_qty = torch.where(
        has_slot_a,
        torch.where(
            (rem_ord_a >= 1.0) & (pos_qty.unsqueeze(1) >= 1.0),
            torch.minimum(rem_ord_a, pos_qty.unsqueeze(1)),
            torch.zeros_like(rem_ord_a),
        ),
        ask_qty,
    )

    mtm_eq = cash + pos_qty * nb1.squeeze(1) * (1.0 - maker_fee)
    peak_mtm_eq = torch.maximum(peak_mtm_eq, mtm_eq)
    dd_pct = 100.0 * torch.clamp(peak_mtm_eq - mtm_eq, min=0.0) / budget
    sum_sq_dd_pct = sum_sq_dd_pct + dd_pct * dd_pct
    max_mtm_dd_pct = torch.maximum(max_mtm_dd_pct, dd_pct)

    # Честная биржевая REST-нагрузка + штраф за тремор перепостановок (requeue = 2 мутации за 1с):
    n_req_step = n_req_b_step + n_req_a_step
    burst_over = torch.clamp(n_mut_tot.float() - 2.0, min=0.0)
    mut_load_now = (n_mut_tot.float() + 0.35 * n_req_step.float() + 0.20 * burst_over * burst_over) / 4.0
    sum_mut_load = sum_mut_load + mut_load_now
    n_cancels = torch.clamp(n_cancels.long() + n_cnl_b_step + n_cnl_a_step, max=30000).to(torch.int16)
    n_requeues = torch.clamp(n_requeues.long() + n_req_step, max=30000).to(torch.int16)

    return (
        mem_h,
        cash,
        pos_qty,
        pos_cost_usd,
        avg_entry,
        min_buy_px,
        low_tranche_qty,
        mid_buy_px,
        mid_tranche_qty,
        entry_s,
        bid_px,
        bid_qty,
        bid_qa,
        bid_qa0,
        bid_act_s,
        ask_px,
        ask_qty,
        ask_qa,
        ask_qa0,
        ask_act_s,
        peak_mtm_eq,
        sum_sq_dd_pct,
        max_mtm_dd_pct,
        sum_mut_load,
        sum_fill_qprog,
        n_fills,
        n_cancels,
        n_requeues,
    )


@torch.no_grad()
def evaluate_l25_dual_basket(
    val_env,
    bf16_runner,
    lob_fu_all: torch.Tensor,
    v7_all: torch.Tensor,
    bo36_all: torch.Tensor,
    roll_sell_30s_all: torch.Tensor,
    cand_states: list[dict],
    val_windows: list[tuple[str, int]],
    win_len: int,
    tick_sz: float,
    device: torch.device,
    true_latency: bool = True,
    allow_scratch_stop_asks: bool = False,
    panic_thresh: float = 0.85,
) -> list[dict]:
    """
    Двухконтурная оценка кандидатов Урока 25 (`480D Dual-Swarm Strategist + Per-Slot Micro-Regulator`):
    оценивает чистый EV, выживание на проливах/каскадах, спасение подводного портфеля `UW`,
    коэффициент суеты `Cnl/Fill`, число `Requeue/Win` и истинный КПД налитой FIFO-очереди `η_queue`.
    """
    W = len(val_windows)
    C = len(cand_states)
    B_bud = 6
    E = C * B_bud
    N = W * E
    S = int(val_env.n_side_slots)
    dt = lob_fu_all.dtype

    mkt_idx_t = torch.tensor(MKT_IDX_L25, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_L25, dtype=torch.long, device=device)

    w1_l, b1_l, wr1_l, br1_l, wr2_l, w2_l, b2_l, wrec_l, brec_l = [], [], [], [], [], [], [], [], []
    wreg1_l, breg1_l, wreg2_l, breg2_l = [], [], [], []
    for st in cand_states:
        w1_raw = st["w1"].to(device=device, dtype=dt)
        if w1_raw.shape[0] < 480:
            pad = torch.zeros((480 - w1_raw.shape[0], w1_raw.shape[1]), dtype=dt, device=device)
            w1_raw = torch.cat([w1_raw, pad], dim=0)
        w1_l.append(w1_raw)
        b1_l.append(st["b1"].to(device=device, dtype=dt).view(1, 384))
        wr1_l.append(st["w_res1"].to(device=device, dtype=dt))
        br1_l.append(st["b_res1"].to(device=device, dtype=dt).view(1, 128))
        wr2_l.append(st["w_res2"].to(device=device, dtype=dt))
        w2_l.append(st["w2"].to(device=device, dtype=dt))
        b2_l.append(st["b2"].to(device=device, dtype=dt).view(1, 35))
        wrec_l.append(st["w_rec"].to(device=device, dtype=dt))
        brec_l.append(st["b_rec"].to(device=device, dtype=dt).view(1, 32))
        wreg1_l.append(st["w_reg1"].to(device=device, dtype=dt))
        breg1_l.append(st["b_reg1"].to(device=device, dtype=dt).view(1, 32))
        wreg2_l.append(st["w_reg2"].to(device=device, dtype=dt))
        breg2_l.append(st["b_reg2"].to(device=device, dtype=dt).view(1, 2))

    w1_c = torch.stack(w1_l, dim=0).contiguous()
    w1_mkt_c = w1_c.index_select(1, mkt_idx_t).contiguous()
    w1_ag_c = w1_c.index_select(1, ag_idx_t).contiguous()
    b1_c = torch.stack(b1_l, dim=0).contiguous()
    wr1_c = torch.stack(wr1_l, dim=0).contiguous()
    br1_c = torch.stack(br1_l, dim=0).contiguous()
    wr2_c = torch.stack(wr2_l, dim=0).contiguous()
    w2_c = torch.stack(w2_l, dim=0).contiguous()
    b2_c = torch.stack(b2_l, dim=0).contiguous()
    wrec_c = torch.stack(wrec_l, dim=0).contiguous()
    brec_c = torch.stack(brec_l, dim=0).contiguous()
    wreg1_c = torch.stack(wreg1_l, dim=0).contiguous()
    breg1_c = torch.stack(breg1_l, dim=0).contiguous()
    wreg2_c = torch.stack(wreg2_l, dim=0).contiguous()
    breg2_c = torch.stack(breg2_l, dim=0).contiguous()

    w_starts = torch.tensor([int(ws) for _, ws in val_windows], dtype=torch.long, device=device)
    t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
    t_steps_cl = torch.clamp(t_steps, max=val_env.max_steps - 1)

    flow_w_all = val_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
    lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()
    v7_w_all = v7_all[t_steps_cl[: win_len - 1]].contiguous()
    bo36_w_all = bo36_all[t_steps_cl[: win_len - 1]].contiguous()

    z_mkt_all = precompute_window_market_z370(flow_w_all, lob_fu_w_all, v7_w_all, bo36_w_all, bf16_runner)
    z_mkt_c_in = z_mkt_all.view((win_len - 1) * W, 370).unsqueeze(0).expand(C, -1, -1)
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

    for wi, (reg_name, ws) in enumerate(val_windows):
        if reg_name.startswith("UW_") or reg_name.endswith("_UW"):
            s0_w = wi * E
            s1_w = (wi + 1) * E
            b1_w0 = float(max(tick_sz, float(val_env.bids[int(ws), 0].item())))
            epx = round((b1_w0 + 5.0 * tick_sz) / tick_sz) * tick_sz
            iqty = torch.floor((env_budgets_e * 0.55 / b1_w0) * 100.0) / 100.0
            v_mask = iqty >= 2.0
            rcash = torch.clamp(env_budgets_e - iqty * b1_w0 * (1.0 - val_env.maker_fee), min=0.0)
            cash[s0_w:s1_w] = torch.where(v_mask, rcash, env_budgets_e)
            pos_qty[s0_w:s1_w] = torch.where(v_mask, iqty, torch.zeros_like(iqty))
            avg_entry[s0_w:s1_w] = torch.where(v_mask, torch.full_like(iqty, epx), torch.zeros_like(iqty))
            pos_cost_usd[s0_w:s1_w] = torch.where(v_mask, iqty * epx * (1.0 + val_env.maker_fee), torch.zeros_like(iqty))
            min_buy_px[s0_w:s1_w] = torch.where(v_mask, torch.full_like(iqty, epx), torch.zeros_like(iqty))
            entry_s[s0_w:s1_w] = -180

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

    init_b1 = torch.clamp(b1_w_all[0], min=tick_sz).unsqueeze(1).expand(W, E).reshape(N)
    start_mtm_eq = cash + pos_qty * init_b1 * (1.0 - val_env.maker_fee)
    peak_mtm_eq = start_mtm_eq.clone()
    sum_sq_dd = torch.zeros(N, dtype=torch.float32, device=device)
    max_dd_pct = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mut = torch.zeros(N, dtype=torch.float32, device=device)
    sum_fqp = torch.zeros(N, dtype=torch.float32, device=device)
    n_fills = torch.zeros(N, dtype=torch.int16, device=device)
    n_cancels = torch.zeros(N, dtype=torch.int16, device=device)
    n_requeues = torch.zeros(N, dtype=torch.int16, device=device)

    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    for s in range(win_len - 1):
        (
            mem_h,
            cash,
            pos_qty,
            pos_cost_usd,
            avg_entry,
            min_buy_px,
            low_tranche_qty,
            mid_buy_px,
            mid_tranche_qty,
            entry_s,
            bid_px,
            bid_qty,
            bid_qa,
            bid_qa0,
            bid_act_s,
            ask_px,
            ask_qty,
            ask_qa,
            ask_qa0,
            ask_act_s,
            peak_mtm_eq,
            sum_sq_dd,
            max_dd_pct,
            sum_mut,
            sum_fqp,
            n_fills,
            n_cancels,
            n_requeues,
        ) = fused_l25_regulator_tick(
            s_steps_i16[s],
            s_steps_i16[s + 1],
            pre_mkt_all[s],
            b1_w_all[s],
            a1_w_all[s],
            b1q_w_all[s],
            a1q_w_all[s],
            rb_w_all[s],
            rs_w_all[s],
            b10_w_all[s],
            a10_w_all[s],
            bq10_w_all[s],
            aq10_w_all[s],
            b1_w_all[s + 1],
            a1_w_all[s + 1],
            sv_w_all[s + 1],
            bv_w_all[s + 1],
            w1_ag_c,
            wr1_c,
            br1_c,
            wr2_c,
            w2_c,
            b2_c,
            wrec_c,
            brec_c,
            wreg1_c,
            breg1_c,
            wreg2_c,
            breg2_c,
            val_env.slot_tick_offsets,
            val_env.slot_lob_lvl,
            val_env.is_penny_slot,
            budget,
            mem_h,
            cash,
            pos_qty,
            pos_cost_usd,
            avg_entry,
            min_buy_px,
            low_tranche_qty,
            mid_buy_px,
            mid_tranche_qty,
            entry_s,
            bid_px,
            bid_qty,
            bid_qa,
            bid_qa0,
            bid_act_s,
            ask_px,
            ask_qty,
            ask_qa,
            ask_qa0,
            ask_act_s,
            peak_mtm_eq,
            sum_sq_dd,
            max_dd_pct,
            sum_mut,
            sum_fqp,
            n_fills,
            n_cancels,
            n_requeues,
            tick_sz,
            val_env.maker_fee,
            val_env.taker_fee,
            true_latency,
            allow_scratch_stop_asks,
            panic_thresh,
        )

    end_b1 = torch.clamp(b1_w_all[win_len - 1], min=tick_sz).unsqueeze(1).expand(W, E).reshape(N)
    rem = pos_qty > 1e-6
    end_cash = torch.where(rem, cash + pos_qty * end_b1 * (1.0 - val_env.maker_fee), cash)
    end_dd_pct = 100.0 * torch.clamp(peak_mtm_eq - end_cash, min=0.0) / budget
    max_dd_pct = torch.maximum(max_dd_pct, end_dd_pct)

    w_pnl_pct_mat = (100.0 * (end_cash - budget) / budget).view(W, E)
    w_mtm_delta_pct_mat = (100.0 * (end_cash - start_mtm_eq) / budget).view(W, E)
    w_ulcer_mat = torch.sqrt(sum_sq_dd.view(W, E) / float(win_len - 1) + 1e-8)
    w_dd_mat = max_dd_pct.view(W, E)
    w_api_mat = (sum_mut / float(win_len - 1)).view(W, E)
    w_fills_mat = n_fills.float().view(W, E)
    w_cancels_mat = n_cancels.float().view(W, E)
    w_requeues_mat = n_requeues.float().view(W, E)
    w_fqp_mat = sum_fqp.view(W, E)

    clean_idx = [wi for wi, (r, _) in enumerate(val_windows) if not (r.startswith("UW_") or r.endswith("_UW"))]
    uw_idx = [wi for wi, (r, _) in enumerate(val_windows) if (r.startswith("UW_") or r.endswith("_UW"))]
    norm_clean_idx = [wi for wi, (r, _) in enumerate(val_windows) if r == "NORM"]
    dump_clean_idx = [wi for wi, (r, _) in enumerate(val_windows) if r in ("DUMP", "CASC")]
    pump_clean_idx = [wi for wi, (r, _) in enumerate(val_windows) if r == "PUMP"]

    results = []
    for ci in range(C):
        s_i = ci * B_bud
        e_i = s_i + B_bud

        def _reg_mean(idxs):
            if not idxs:
                return 0.0
            return float(w_pnl_pct_mat[idxs, s_i:e_i].mean().item())

        n_c = _reg_mean(norm_clean_idx)
        d_c = _reg_mean(dump_clean_idx)
        p_c = _reg_mean(pump_clean_idx)
        clean_ev = 0.65 * n_c + 0.20 * p_c + 0.15 * d_c

        if clean_idx:
            cl_pnl_w = w_pnl_pct_mat[clean_idx, s_i:e_i]
            cl_ulc = float(w_ulcer_mat[clean_idx, s_i:e_i].mean().item())
            cl_dd = float(w_dd_mat[clean_idx, s_i:e_i].max(dim=0).values.mean().item())
            cl_api = float(w_api_mat[clean_idx, s_i:e_i].mean().item())
            cl_fills = float(w_fills_mat[clean_idx, s_i:e_i].sum(dim=0).mean().item())
            cl_cancels = float(w_cancels_mat[clean_idx, s_i:e_i].sum(dim=0).mean().item())
            cl_requeues = float(w_requeues_mat[clean_idx, s_i:e_i].sum(dim=0).mean().item())
            cl_fqp_sum = float(w_fqp_mat[clean_idx, s_i:e_i].sum(dim=0).mean().item())
            cl_cnl_per_fill = cl_cancels / max(cl_fills, 0.5)
            cl_eta_queue = (
                100.0 * min(1.0, cl_fqp_sum / max(cl_fills, 1.0))
                if cl_fills >= 0.5
                else (100.0 if cl_cancels <= 1.0 else 0.0)
            )
            cl_act_win_frac = float((w_fills_mat[clean_idx, s_i:e_i] > 0).float().mean().item())
            k_cvar = max(1, len(clean_idx) // 4)
            cl_cvar25 = float(torch.topk(cl_pnl_w.mean(dim=1), k_cvar, largest=False).values.mean().item())
            tremor_excess = max(0.0, cl_cnl_per_fill - 14.0) * 0.008
            clean_util_bp = 100.0 * (
                clean_ev + 0.25 * cl_cvar25 - 0.55 * cl_ulc - 0.04 * (cl_dd * cl_dd) - 0.25 * cl_api - tremor_excess
            )
        else:
            cl_ulc = cl_dd = cl_api = cl_fills = cl_cancels = cl_requeues = cl_cnl_per_fill = cl_eta_queue = cl_act_win_frac = cl_cvar25 = clean_util_bp = 0.0

        if uw_idx:
            uw_delta_w = w_mtm_delta_pct_mat[uw_idx, s_i:e_i]
            uw_raw_pnl = float(w_pnl_pct_mat[uw_idx, s_i:e_i].mean().item())
            uw_ev = float(uw_delta_w.mean().item())
            uw_worst = float(uw_delta_w.mean(dim=1).min().item())
            uw_ulc = float(w_ulcer_mat[uw_idx, s_i:e_i].mean().item())
            uw_dd = float(w_dd_mat[uw_idx, s_i:e_i].max(dim=0).values.mean().item())
            uw_api = float(w_api_mat[uw_idx, s_i:e_i].mean().item())
            uw_util_bp = 100.0 * (
                uw_ev + 0.25 * uw_worst - 0.45 * uw_ulc - 0.035 * (uw_dd * uw_dd) - 0.25 * uw_api
            )
        else:
            uw_raw_pnl = uw_ev = uw_worst = uw_ulc = uw_dd = uw_api = uw_util_bp = 0.0

        n_all_idx = [wi for wi, (r, _) in enumerate(val_windows) if r in ("NORM", "UW_NORM")]
        d_all_idx = [wi for wi, (r, _) in enumerate(val_windows) if r in ("DUMP", "CASC", "UW_DUMP")]
        n_all = _reg_mean(n_all_idx)
        d_all = _reg_mean(d_all_idx)
        p_all = p_c
        full_dd = float(w_dd_mat[:, s_i:e_i].max(dim=0).values.mean().item())
        realistic_ev_full = 0.65 * n_all + 0.20 * p_all + 0.15 * d_all
        pareto_util_bp = (0.72 * clean_util_bp + 0.28 * uw_util_bp) if uw_idx else clean_util_bp

        results.append({
            "pareto_util_bp": pareto_util_bp,
            "clean_util_bp": clean_util_bp,
            "uw_util_bp": uw_util_bp,
            "clean_ev": clean_ev,
            "clean_norm_pct": n_c,
            "clean_dump_pct": d_c,
            "clean_pump_pct": p_c,
            "clean_cvar25_pct": cl_cvar25,
            "clean_dd_pct": cl_dd,
            "clean_fills": cl_fills,
            "clean_cancels": cl_cancels,
            "clean_requeues": cl_requeues,
            "clean_cnl_per_fill": cl_cnl_per_fill,
            "clean_eta_queue_pct": cl_eta_queue,
            "clean_api": cl_api,
            "clean_act_frac": cl_act_win_frac,
            "uw_delta_ev": uw_ev,
            "uw_raw_pct": uw_raw_pnl,
            "realistic_ev": realistic_ev_full,
            "norm_pct": n_all,
            "dump_pct": d_all,
            "pump_pct": p_all,
            "dd_pct": full_dd,
        })
    return results
