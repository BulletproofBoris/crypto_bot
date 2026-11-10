#!/usr/bin/env python3
"""
_tools/vectorized_free_multi_order_env.py
=============================================================================
VectorizedFreeMultiOrderEnv (Урок 6 v2 — 24 Независимых Слота [12 Bid + 12 Ask],
Полная Свобода Объема Заявок + Безветвевая GPU-Векторизация + SelfImpactRealismNet v2):

Ключевые особенности:
1. 24 ОДНОВРЕМЕННЫХ НЕЗАВИСИМЫХ ОРДЕРА В КНИГЕ (4x расширение: 12 Bid + 12 Ask):
   - 6 ценовых эшелонов × 2 независимых транша (айсберга) на каждом эшелоне:
     * Слоты 0, 1 : Penny-Jump (`b1 + tick` / `a1 - tick` при широком спреде, иначе L1)
     * Слоты 2, 3 : Best L1 (`b1` / `a1`)
     * Слоты 4, 5 : Passive L2 (`b1 - 1 tick` / `a1 + 1 tick`)
     * Слоты 6, 7 : Deep L3 (`b1 - 2 ticks` / `a1 + 2 ticks`)
     * Слоты 8, 9 : Deep L4 (`b1 - 3 ticks` / `a1 + 3 ticks`)
     * Слоты 10,11: Knife/Pump L5 (`b1 - 5 ticks` / `a1 + 5 ticks`)
2. 100% БЕЗВЕТВЕВАЯ МАТРИЧНАЯ ВЕКТОРИЗАЦИЯ (ZERO CPU-GPU SYNC):
   - Полностью устранены вызовы `.any()`, `bool()` и `torch.nonzero()` внутри шага `step()`.
   - Каскадный налив по всем 12 слотам Bid и 12 слотам Ask вычисляется точно через
     матричный водопад `torch.cumsum(..., dim=1)` за один проход ядра CUDA без потери точности.
=============================================================================
"""

import os

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor_cache")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from typing import Tuple


@torch.compile(dynamic=True)
def _jit_compute_obs_flow_and_multi(
    flow_mkt_1d: torch.Tensor,
    b1: torch.Tensor,
    a1: torch.Tensor,
    b1_q: torch.Tensor,
    a1_q: torch.Tensor,
    roll_buy_t: torch.Tensor,
    pos_qty: torch.Tensor,
    budget: torch.Tensor,
    bid_qty: torch.Tensor,
    bid_px: torch.Tensor,
    ask_qty: torch.Tensor,
    cash: torch.Tensor,
    avg_entry: torch.Tensor,
    entry_t: torch.Tensor,
    low_tranche_qty: torch.Tensor,
    min_buy_px: torch.Tensor,
    cur_t_scalar: int,
    num_envs: int,
    n_side_slots: int,
    tick_size: float,
    maker_fee: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b1_c = torch.clamp(b1, min=tick_size)
    a1_c = torch.clamp(a1, min=b1_c + tick_size)
    l1_bid_usd = torch.clamp(b1_c * b1_q, min=100.0)
    l1_ask_usd = torch.clamp(a1_c * a1_q, min=100.0)

    has_pos = (pos_qty > 1e-6).float()
    curr_pos_usd = pos_qty * b1_c
    pos_ratio = torch.clamp(curr_pos_usd / torch.clamp(budget, min=25.0), 0.0, 1.0)
    reserved_cash_now = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    unreserved_cash_now = torch.clamp(cash - reserved_cash_now, min=0.0)
    free_cash_ratio = torch.clamp(unreserved_cash_now / torch.clamp(budget, min=25.0), 0.0, 1.0)

    eff_pos_scale = torch.clamp(pos_ratio / 0.25, 0.15, 1.0)
    unr_ticks = torch.where(has_pos > 0.5, ((b1_c - avg_entry) / tick_size) * eff_pos_scale, torch.zeros_like(has_pos))
    hold_ratio = torch.where(
        has_pos > 0.5,
        torch.clamp((float(cur_t_scalar) - entry_t.float()) / 300.0, 0.0, 2.0) * eff_pos_scale,
        torch.zeros_like(has_pos),
    )

    tot_bid_q = bid_qty.sum(dim=1)
    tot_ask_q = ask_qty.sum(dim=1)
    has_bid = (tot_bid_q > 1e-6).float()
    has_ask = (tot_ask_q > 1e-6).float()
    n_bid_slots = (bid_qty > 1e-6).float().sum(dim=1) / float(n_side_slots)
    n_ask_slots = (ask_qty > 1e-6).float().sum(dim=1) / float(n_side_slots)

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
    )
    flow = torch.cat([flow_mkt_1d[:, :8].expand(num_envs, 8), acct_8d], dim=-1)

    bud_l1 = torch.log1p(budget / l1_bid_usd) / 3.0
    cash_l1 = torch.log1p(unreserved_cash_now / l1_bid_usd) / 3.0
    pos_l1_ask = torch.clamp((pos_qty * a1_c) / l1_ask_usd, 0.0, 4.0) / 2.0
    pos_tape = torch.clamp(pos_qty / roll_buy_t, 0.0, 10.0) / 5.0
    low_tr_ratio = torch.where(
        has_pos > 0.5,
        torch.clamp(low_tranche_qty / torch.clamp(pos_qty, min=1e-6), 0.0, 1.0),
        torch.zeros_like(has_pos),
    )
    lifo_edge_ticks = torch.where(
        has_pos > 0.5,
        torch.clamp((b1_c - min_buy_px) / (3.0 * tick_size), -2.0, 2.0),
        torch.zeros_like(has_pos),
    )
    multi_6d = torch.stack([bud_l1, cash_l1, pos_l1_ask, pos_tape, low_tr_ratio, lifo_edge_ticks], dim=-1)
    return flow, multi_6d


@torch.compile(dynamic=True)
def _jit_env_step(
    raw_outputs: torch.Tensor,
    bids_t: torch.Tensor,
    asks_t: torch.Tensor,
    bids_q_t: torch.Tensor,
    asks_q_t: torch.Tensor,
    bids_next: torch.Tensor,
    asks_next: torch.Tensor,
    sv_next: torch.Tensor,
    bv_next: torch.Tensor,
    budget: torch.Tensor,
    cash: torch.Tensor,
    pos_qty: torch.Tensor,
    pos_cost_usd: torch.Tensor,
    avg_entry: torch.Tensor,
    min_buy_px: torch.Tensor,
    low_tranche_qty: torch.Tensor,
    entry_t: torch.Tensor,
    bid_px: torch.Tensor,
    bid_qty: torch.Tensor,
    bid_qa: torch.Tensor,
    bid_act_t: torch.Tensor,
    ask_px: torch.Tensor,
    ask_qty: torch.Tensor,
    ask_qa: torch.Tensor,
    ask_act_t: torch.Tensor,
    slot_tick_offsets: torch.Tensor,
    slot_lob_lvl: torch.Tensor,
    is_penny_slot: torch.Tensor,
    n_trades: torch.Tensor,
    partial_fills: torch.Tensor,
    multi_orders_used: torch.Tensor,
    frontrun_hits: torch.Tensor,
    sell_cycles: torch.Tensor,
    prof_sell_cycles: torch.Tensor,
    cur_t_scalar: int,
    next_t: int,
    num_envs: int,
    S: int,
    tick_size: float,
    maker_fee: float,
    taker_fee: float,
    allow_scratch_stop_asks: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    b1 = torch.clamp(bids_t[0], min=tick_size)
    a1 = torch.clamp(asks_t[0], min=b1 + tick_size)
    l1_bid_q = torch.clamp(bids_q_t[0], min=10.0)
    l1_ask_q = torch.clamp(asks_q_t[0], min=10.0)
    wide_sp = (a1 - b1) >= 2.0 * tick_size

    is_panic = (raw_outputs[:, 2 * S + 1] > 0.82) & (pos_qty > 1e-6)
    b_p10 = torch.clamp(bids_t[:10], min=tick_size)
    b_q10 = torch.clamp(bids_q_t[:10], min=1.0).clone()
    b_q10[9] = 1e9
    cum_q10 = torch.cumsum(b_q10, dim=0)
    prev_q10 = cum_q10 - b_q10
    take_10 = torch.clamp(
        torch.minimum(pos_qty.unsqueeze(1) - prev_q10.unsqueeze(0), b_q10.unsqueeze(0)),
        min=0.0,
    )
    proceeds = (take_10 * b_p10.unsqueeze(0)).sum(dim=1) * (1.0 - taker_fee)

    cash = torch.where(is_panic, cash + proceeds, cash)
    pos_qty = torch.where(is_panic, torch.zeros_like(pos_qty), pos_qty)
    pos_cost_usd = torch.where(is_panic, torch.zeros_like(pos_cost_usd), pos_cost_usd)
    avg_entry = torch.where(is_panic, torch.zeros_like(avg_entry), avg_entry)
    min_buy_px = torch.where(is_panic, torch.zeros_like(min_buy_px), min_buy_px)
    low_tranche_qty = torch.where(is_panic, torch.zeros_like(low_tranche_qty), low_tranche_qty)
    ask_qty = torch.where(is_panic.unsqueeze(1), torch.zeros_like(ask_qty), ask_qty)
    n_trades = n_trades + is_panic.long()

    cancel_bids = raw_outputs[:, 2 * S] > 0.75
    cb_mask = cancel_bids.unsqueeze(1)
    b_w = torch.where(cb_mask, torch.zeros_like(raw_outputs[:, :S]), raw_outputs[:, :S])

    penny_add = torch.where(wide_sp, b1.new_full((), tick_size), b1.new_zeros(()))
    slot_b_px_1d = torch.where(
        is_penny_slot,
        b1 + penny_add,
        torch.clamp(b1 - slot_tick_offsets, min=tick_size),
    )
    t_bid_px = slot_b_px_1d.unsqueeze(0).expand(num_envs, S)

    slot_off_b = cb_mask | (b_w < 0.06)
    grid_shift_b = (
        (bid_qty > 1e-6)
        & ((cur_t_scalar - bid_act_t) >= 10)
        & (torch.abs(t_bid_px - bid_px) >= 1.9 * tick_size)
    )
    clear_b = slot_off_b | grid_shift_b
    bid_qty = torch.where(clear_b, torch.zeros_like(bid_qty), bid_qty)
    bid_px = torch.where(clear_b, torch.zeros_like(bid_px), bid_px)
    bid_qa = torch.where(clear_b, torch.zeros_like(bid_qa), bid_qa)

    empty_b = bid_qty <= 1e-6
    place_b = empty_b & (b_w >= 0.12)
    b_w_active = torch.where(place_b, b_w, torch.zeros_like(b_w))
    b_total_sum = torch.where(b_w >= 0.08, b_w, torch.zeros_like(b_w)).sum(dim=1)
    target_exp_frac = torch.clamp(b_total_sum, 0.0, 1.0)
    target_exp_usd = budget * 0.996 * target_exp_frac

    b_sum = b_w_active.sum(dim=1, keepdim=True)
    scale_b = torch.where(
        b_sum > 1e-5,
        1.0 / torch.clamp(b_sum, min=1e-6),
        torch.zeros_like(b_sum),
    )
    b_alloc = b_w_active * scale_b

    lvl_b_q = torch.clamp(bids_q_t[slot_lob_lvl], min=10.0)
    base_b_qa_1d = torch.where(is_penny_slot & wide_sp, torch.zeros_like(lvl_b_q), lvl_b_q)
    base_b_qa = base_b_qa_1d.unsqueeze(0).expand(num_envs, S)

    reserved_cash = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    pos_cost_now = pos_qty * avg_entry
    committed_usd = pos_cost_now + reserved_cash
    avail_exp_usd = torch.clamp(target_exp_usd - committed_usd, min=0.0)
    unreserved_cash = torch.minimum(
        torch.clamp(cash * 0.996 - reserved_cash, min=0.0),
        avail_exp_usd,
    ).unsqueeze(1)
    t_bid_usd = unreserved_cash * b_alloc
    valid_b = place_b & (t_bid_usd >= 2.0)
    t_bid_qty = torch.where(valid_b, t_bid_usd / t_bid_px, torch.zeros_like(t_bid_usd))

    b_over65 = torch.sigmoid((t_bid_usd - 65.0) / 25.0)
    b_self_impact_qa = base_b_qa + b_over65 * (0.065 * l1_bid_q + 0.15 * t_bid_qty)

    bid_px = torch.where(valid_b, t_bid_px, bid_px)
    bid_qty = torch.where(valid_b, t_bid_qty, bid_qty)
    bid_qa = torch.where(valid_b, b_self_impact_qa, bid_qa)
    bid_act_t = torch.where(valid_b, torch.full_like(bid_act_t, cur_t_scalar + 1), bid_act_t)

    has_p = (pos_qty > 1e-6) & (~is_panic)
    hp_mask = has_p.unsqueeze(1)
    a_w = torch.where(hp_mask, raw_outputs[:, S : 2 * S], torch.zeros_like(raw_outputs[:, S : 2 * S]))

    has_low_tr = (pos_qty > 1e-6) & (low_tranche_qty >= 1.95) & (min_buy_px < avg_entry - 0.1 * tick_size)
    avg_tp_tick = torch.ceil((avg_entry + 0.55 * tick_size) / tick_size - 1e-5) * tick_size
    lifo_tp_tick = torch.ceil((min_buy_px + 0.55 * tick_size) / tick_size - 1e-5) * tick_size
    base_avg_tp = torch.where(pos_qty > 1e-6, avg_tp_tick, a1)
    base_lifo_tp = torch.where(has_low_tr, lifo_tp_tick, base_avg_tp)

    if allow_scratch_stop_asks:
        avg_round_tick = torch.clamp(torch.round(avg_entry / tick_size) * tick_size, min=tick_size)
        stop_1_px = torch.where(pos_qty > 1e-6, torch.clamp(avg_round_tick - tick_size, min=tick_size), base_avg_tp).unsqueeze(1)
        scratch_0_px = torch.where(pos_qty > 1e-6, avg_round_tick, base_avg_tp).unsqueeze(1)
        slot_base_tp = torch.cat(
            [
                stop_1_px,
                scratch_0_px,
                base_lifo_tp.unsqueeze(1).expand(num_envs, 4),
                base_avg_tp.unsqueeze(1).expand(num_envs, S - 6),
            ],
            dim=1,
        )
    else:
        slot_base_tp = torch.cat(
            [
                base_lifo_tp.unsqueeze(1).expand(num_envs, 6),
                base_avg_tp.unsqueeze(1).expand(num_envs, S - 6),
            ],
            dim=1,
        )

    raw_slot_a_px = torch.where(
        is_penny_slot,
        a1 - penny_add,
        a1 + slot_tick_offsets,
    ).unsqueeze(0).expand(num_envs, S)
    t_ask_px = torch.maximum(raw_slot_a_px, slot_base_tp + slot_tick_offsets.unsqueeze(0))

    slot_off_a = (~hp_mask) | ((a_w < 0.04) & ((cur_t_scalar - ask_act_t) >= 180))
    under_tp_a = (ask_qty > 1e-6) & (ask_px < slot_base_tp - 0.4 * tick_size)
    grid_shift_a = (
        (ask_qty > 1e-6)
        & ((cur_t_scalar - ask_act_t) >= 15)
        & (torch.abs(ask_px - t_ask_px) >= 1.9 * tick_size)
    ) | under_tp_a
    clear_a = slot_off_a | grid_shift_a
    ask_qty = torch.where(clear_a, torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(clear_a, torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(clear_a, torch.zeros_like(ask_qa), ask_qa)

    empty_a = ask_qty <= 1e-6
    place_a = empty_a & (a_w >= 0.12) & hp_mask
    a_w_active = torch.where(place_a, a_w, torch.zeros_like(a_w))
    a_sum = a_w_active.sum(dim=1, keepdim=True)
    scale_a = torch.where(
        a_sum > 1e-5,
        1.0 / torch.clamp(a_sum, min=1e-6),
        torch.zeros_like(a_sum),
    )
    a_alloc = a_w_active * scale_a

    eff_a_lvl = torch.clamp(torch.round((t_ask_px - a1) / tick_size).long(), 0, 9)
    asks_q_row = torch.clamp(asks_q_t, min=10.0)
    lvl_a_q_2d = asks_q_row[eff_a_lvl]
    inside_sp_a = t_ask_px < (a1 - 0.1 * tick_size)
    base_a_qa = torch.where(inside_sp_a, torch.zeros_like(lvl_a_q_2d), lvl_a_q_2d)

    unreserved_pos = torch.clamp(pos_qty - ask_qty.sum(dim=1), min=0.0).unsqueeze(1)
    t_ask_qty = unreserved_pos * a_alloc
    need_lifo_cap = has_low_tr & (base_lifo_tp < base_avg_tp - 0.1 * tick_size)
    if allow_scratch_stop_asks:
        lifo_already_placed = ask_qty[:, 2:6].sum(dim=1, keepdim=True)
        rem_low_tr = torch.clamp(low_tranche_qty.unsqueeze(1) - lifo_already_placed, min=0.0)
        want_low_sum = t_ask_qty[:, 2:6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        low_cap_scale = torch.where(
            need_lifo_cap.unsqueeze(1) & (want_low_sum > rem_low_tr),
            rem_low_tr / want_low_sum,
            torch.ones_like(want_low_sum),
        )
        t_ask_qty = torch.cat([t_ask_qty[:, :2], t_ask_qty[:, 2:6] * low_cap_scale, t_ask_qty[:, 6:]], dim=1)
    else:
        lifo_already_placed = ask_qty[:, :6].sum(dim=1, keepdim=True)
        rem_low_tr = torch.clamp(low_tranche_qty.unsqueeze(1) - lifo_already_placed, min=0.0)
        want_low_sum = t_ask_qty[:, :6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        low_cap_scale = torch.where(
            need_lifo_cap.unsqueeze(1) & (want_low_sum > rem_low_tr),
            rem_low_tr / want_low_sum,
            torch.ones_like(want_low_sum),
        )
        t_ask_qty = torch.cat([t_ask_qty[:, :6] * low_cap_scale, t_ask_qty[:, 6:]], dim=1)

    t_ask_usd = t_ask_qty * t_ask_px
    valid_a = place_a & (t_ask_qty >= 1.0)
    t_ask_qty = torch.where(valid_a, t_ask_qty, torch.zeros_like(t_ask_qty))

    a_over50 = torch.sigmoid((t_ask_usd - 50.0) / 18.0)
    a_self_impact_qa = base_a_qa * (1.0 + 0.75 * a_over50) + a_over50 * (0.45 * t_ask_qty)
    frontrun_hits = frontrun_hits + ((valid_a & (t_ask_usd > 60.0)).any(dim=1)).long()

    ask_px = torch.where(valid_a, t_ask_px, ask_px)
    ask_qty = torch.where(valid_a, t_ask_qty, ask_qty)
    ask_qa = torch.where(valid_a, a_self_impact_qa, ask_qa)
    ask_act_t = torch.where(valid_a, torch.full_like(ask_act_t, cur_t_scalar + 1), ask_act_t)

    tot_active_orders = (bid_qty > 1e-6).long().sum(dim=1) + (ask_qty > 1e-6).long().sum(dim=1)
    multi_orders_used = multi_orders_used + (tot_active_orders >= 2).long()

    nb1 = torch.clamp(bids_next[0], min=tick_size)
    na1 = torch.clamp(asks_next[0], min=nb1 + tick_size)
    sv = sv_next
    bv = bv_next

    act_b = (bid_qty > 1e-6) & (next_t >= bid_act_t)
    strict_cross_b = na1 <= (bid_px - 0.5 * tick_size)
    hit_b = act_b & (strict_cross_b | ((nb1 <= bid_px + 1e-6) & (sv > 0.0)))

    any_strict_b = (act_b & strict_cross_b).any(dim=1, keepdim=True)
    tot_flow_b = torch.where(any_strict_b, sv + 0.12 * l1_bid_q, sv)

    even_act_b = act_b[:, 0::2] & (torch.abs(bid_px[:, 1::2] - bid_px[:, 0::2]) < 0.1 * tick_size)
    Incremental_qa_b = bid_qa.clone()
    Incremental_qa_b[:, 1::2] = torch.where(
        even_act_b,
        torch.clamp(bid_qa[:, 1::2] - bid_qa[:, 0::2], min=0.0),
        bid_qa[:, 1::2],
    )
    eff_qa_b = torch.where(hit_b, Incremental_qa_b, torch.zeros_like(bid_qa))
    eff_qty_b = torch.where(hit_b, bid_qty, torch.zeros_like(bid_qty))

    slot_need_b = eff_qa_b + eff_qty_b
    cum_need_b = torch.cumsum(slot_need_b, dim=1)
    flow_before_slot_b = cum_need_b - slot_need_b
    rem_flow_at_slot_b = torch.clamp(tot_flow_b - flow_before_slot_b, min=0.0)

    qa_eaten_b = torch.minimum(eff_qa_b, rem_flow_at_slot_b)
    bid_qa = torch.where(hit_b, torch.clamp(bid_qa - qa_eaten_b, min=0.0), bid_qa)
    bid_qa[:, 1::2] = torch.where(
        act_b[:, 1::2] & even_act_b,
        torch.clamp(bid_qa[:, 1::2] - qa_eaten_b[:, 0::2], min=0.0),
        bid_qa[:, 1::2],
    )

    want_b = torch.minimum(eff_qty_b, torch.clamp(rem_flow_at_slot_b - eff_qa_b, min=0.0))
    cum_want_b = torch.cumsum(want_b, dim=1)
    prev_cum_b = cum_want_b - want_b
    max_afford_b = torch.clamp(cash * 0.998 / (b1 * (1.0 + maker_fee)), min=0.0).unsqueeze(1)
    f_q = torch.minimum(want_b, torch.clamp(max_afford_b - prev_cum_b, min=0.0))

    has_slot_b = f_q > 1e-5
    tot_fq = f_q.sum(dim=1)
    tot_cost_raw = (f_q * bid_px).sum(dim=1)
    tot_cost = tot_cost_raw * (1.0 + maker_fee)
    has_buy = tot_fq > 1e-5

    old_q = pos_qty
    new_q = old_q + tot_fq
    avg_fill_b_px = tot_cost_raw / torch.clamp(tot_fq, min=1e-6)

    is_dip_buy = (
        has_buy
        & (old_q > 1e-6)
        & (
            (avg_fill_b_px <= avg_entry - 0.15 * tick_size)
            | ((low_tranche_qty > 1e-6) & (avg_fill_b_px <= min_buy_px + 0.15 * tick_size))
        )
    )
    is_higher_buy_from_single = (
        has_buy
        & (old_q >= 1.95)
        & (low_tranche_qty <= 1e-6)
        & (~is_dip_buy)
        & (avg_fill_b_px >= avg_entry + 0.35 * tick_size)
    )
    low_tranche_qty = torch.where(
        has_buy & (old_q <= 1e-6),
        torch.zeros_like(low_tranche_qty),
        torch.where(
            is_dip_buy,
            low_tranche_qty + tot_fq,
            torch.where(is_higher_buy_from_single, old_q, low_tranche_qty),
        ),
    )
    min_buy_px = torch.where(
        has_buy,
        torch.where(
            old_q <= 1e-6,
            avg_fill_b_px,
            torch.where(
                is_dip_buy,
                torch.where(low_tranche_qty > tot_fq + 1e-5, torch.minimum(min_buy_px, avg_fill_b_px), avg_fill_b_px),
                torch.where(is_higher_buy_from_single, avg_entry, min_buy_px),
            ),
        ),
        min_buy_px,
    )

    avg_entry = torch.where(
        has_buy,
        (old_q * avg_entry + tot_cost_raw) / torch.clamp(new_q, min=1e-6),
        avg_entry,
    )
    pos_cost_usd = torch.where(has_buy, pos_cost_usd + tot_cost, pos_cost_usd)
    entry_t = torch.where(has_buy & (old_q <= 1e-6), torch.full_like(entry_t, next_t), entry_t)
    cash = torch.where(has_buy, torch.clamp(cash - tot_cost, min=0.0), cash)
    pos_qty = torch.where(has_buy, new_q, pos_qty)

    rem_ord_b = torch.clamp(bid_qty - f_q, min=0.0)
    partial_fills = partial_fills + (has_slot_b & (rem_ord_b > 1e-4)).long().sum(dim=1)
    bid_qty = torch.where(
        has_slot_b,
        torch.where(rem_ord_b >= 1.0, rem_ord_b, torch.zeros_like(rem_ord_b)),
        bid_qty,
    )
    n_trades = n_trades + has_slot_b.long().sum(dim=1)

    act_a = (ask_qty > 1e-6) & (next_t >= ask_act_t) & (pos_qty.unsqueeze(1) > 1e-6)
    strict_cross_a = nb1 >= (ask_px + 0.5 * tick_size)
    hit_a = act_a & (strict_cross_a | ((na1 >= ask_px - 1e-6) & (bv > 0.0)))

    any_strict_a = (act_a & strict_cross_a).any(dim=1, keepdim=True)
    tot_flow_a = torch.where(any_strict_a, bv + 0.12 * l1_ask_q, bv)

    even_act_a = act_a[:, 0::2] & (torch.abs(ask_px[:, 1::2] - ask_px[:, 0::2]) < 0.1 * tick_size)
    Incremental_qa_a = ask_qa.clone()
    Incremental_qa_a[:, 1::2] = torch.where(
        even_act_a,
        torch.clamp(ask_qa[:, 1::2] - ask_qa[:, 0::2], min=0.0),
        ask_qa[:, 1::2],
    )
    eff_qa_a = torch.where(hit_a, Incremental_qa_a, torch.zeros_like(ask_qa))
    eff_qty_a = torch.where(hit_a, ask_qty, torch.zeros_like(ask_qty))

    slot_need_a = eff_qa_a + eff_qty_a
    cum_need_a = torch.cumsum(slot_need_a, dim=1)
    flow_before_slot_a = cum_need_a - slot_need_a
    rem_flow_at_slot_a = torch.clamp(tot_flow_a - flow_before_slot_a, min=0.0)

    qa_eaten_a = torch.minimum(eff_qa_a, rem_flow_at_slot_a)
    ask_qa = torch.where(hit_a, torch.clamp(ask_qa - qa_eaten_a, min=0.0), ask_qa)
    ask_qa[:, 1::2] = torch.where(
        act_a[:, 1::2] & even_act_a,
        torch.clamp(ask_qa[:, 1::2] - qa_eaten_a[:, 0::2], min=0.0),
        ask_qa[:, 1::2],
    )

    want_a = torch.minimum(eff_qty_a, torch.clamp(rem_flow_at_slot_a - eff_qa_a, min=0.0))
    cum_want_a = torch.cumsum(want_a, dim=1)
    prev_cum_a = cum_want_a - want_a
    cap_a = pos_qty.unsqueeze(1)
    f_qa = torch.minimum(want_a, torch.clamp(cap_a - prev_cum_a, min=0.0))

    has_slot_a = f_qa > 1e-5
    tot_fqa = f_qa.sum(dim=1)
    tot_rev = (f_qa * ask_px).sum(dim=1) * (1.0 - maker_fee)
    has_sell = tot_fqa > 1e-5

    lifo_f_qa = f_qa[:, 2:6].sum(dim=1) if allow_scratch_stop_asks else f_qa[:, :6].sum(dim=1)
    sold_low_q = torch.minimum(lifo_f_qa, low_tranche_qty)
    sold_high_q = torch.clamp(tot_fqa - sold_low_q, min=0.0)
    sold_cost_usd = torch.minimum(
        pos_cost_usd,
        sold_low_q * torch.minimum(min_buy_px, avg_entry) * (1.0 + maker_fee)
        + sold_high_q * avg_entry * (1.0 + maker_fee),
    )
    is_net_prof_sell = has_sell & (tot_rev > sold_cost_usd + 1e-5)
    sell_cycles = sell_cycles + has_sell.long()
    prof_sell_cycles = prof_sell_cycles + is_net_prof_sell.long()

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
    low_tranche_qty = torch.where(
        has_sell,
        torch.where(rem_p > 1e-6, torch.clamp(low_tranche_qty - sold_low_q, min=0.0), torch.zeros_like(low_tranche_qty)),
        low_tranche_qty,
    )
    min_buy_px = torch.where(
        pos_qty > 1e-6,
        torch.where(low_tranche_qty > 1e-4, min_buy_px, avg_entry),
        torch.zeros_like(min_buy_px),
    )

    rem_ord_a = torch.clamp(ask_qty - f_qa, min=0.0)
    partial_fills = partial_fills + (has_slot_a & (rem_ord_a > 1e-4)).long().sum(dim=1)
    ask_qty = torch.where(
        has_slot_a,
        torch.where(
            (rem_ord_a >= 1.0) & (pos_qty.unsqueeze(1) >= 1.0),
            torch.minimum(rem_ord_a, pos_qty.unsqueeze(1)),
            torch.zeros_like(rem_ord_a),
        ),
        ask_qty,
    )
    n_trades = n_trades + has_slot_a.long().sum(dim=1)

    return (
        cash,
        pos_qty,
        pos_cost_usd,
        avg_entry,
        min_buy_px,
        low_tranche_qty,
        entry_t,
        bid_px,
        bid_qty,
        bid_qa,
        bid_act_t,
        ask_px,
        ask_qty,
        ask_qa,
        ask_act_t,
        n_trades,
        partial_fills,
        multi_orders_used,
        frontrun_hits,
        sell_cycles,
        prof_sell_cycles,
    )


class VectorizedFreeMultiOrderEnv:
    def __init__(
        self,
        bids: torch.Tensor,
        asks: torch.Tensor,
        bids_q: torch.Tensor,
        asks_q: torch.Tensor,
        buy_vol: torch.Tensor,
        sell_vol: torch.Tensor,
        lob_tensors: torch.Tensor,
        flow_market: torch.Tensor,
        targets: torch.Tensor,
        num_envs: int = 8192,
        n_side_slots: int = 12,
        tick_size: float = 0.01,
        maker_fee: float = 0.0010,
        taker_fee: float = 0.0020,
        init_budget: float = 1000.0,
        device: torch.device = torch.device("cuda"),
        allow_scratch_stop_asks: bool = False,
    ):
        self.device = device
        self.num_envs = num_envs
        self.n_side_slots = n_side_slots  # 12 слотов Bid + 12 слотов Ask = 24 слота одновременно
        self.out_dim = 2 * n_side_slots + 2  # 12 Bid + 12 Ask + Cancel Bid gate + Panic Sweep gate = 26
        self.tick_size = float(tick_size)
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.allow_scratch_stop_asks = bool(allow_scratch_stop_asks)

        self.bids = bids.to(device)
        self.asks = asks.to(device)
        self.bids_q = bids_q.to(device)
        self.asks_q = asks_q.to(device)
        self.buy_vol = buy_vol.to(device)
        self.sell_vol = sell_vol.to(device)
        self.lob_tensors = lob_tensors.to(device)
        self.flow_market = flow_market.to(device)
        self.targets = targets.to(device)
        # CPU-копия флагов кровотечения/обвала, чтобы не вызывать CPU-GPU синхронизацию на каждом тике
        t_cpu = targets.detach().cpu()
        self.is_bleed_cpu = ((t_cpu[:, 1] > 0.5) | (t_cpu[:, 2] > 0.5)).numpy()
        self.max_steps = len(self.bids)

        # Скользящий 30-секундный поток для оценки емкости
        bv_cpu = buy_vol.cpu().float()
        sv_cpu = sell_vol.cpu().float()
        cs_b = torch.cat([torch.zeros(1), torch.cumsum(bv_cpu, dim=0)])
        cs_s = torch.cat([torch.zeros(1), torch.cumsum(sv_cpu, dim=0)])
        idx = torch.arange(self.max_steps)
        self.roll_buy_30s = torch.clamp(cs_b[idx + 1] - cs_b[torch.clamp(idx - 29, min=0)], min=10.0).to(device)
        self.roll_sell_30s = torch.clamp(cs_s[idx + 1] - cs_s[torch.clamp(idx - 29, min=0)], min=10.0).to(device)

        self.budget = torch.full((num_envs,), float(init_budget), dtype=torch.float32, device=device)
        self.cash = torch.full((num_envs,), float(init_budget), dtype=torch.float32, device=device)
        self.pos_qty = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.pos_cost_usd = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.avg_entry = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.min_buy_px = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.low_tranche_qty = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.entry_t = torch.zeros(num_envs, dtype=torch.long, device=device)

        # 12 независимых слотов Bid [B, 12] и 12 независимых слотов Ask [B, 12]
        self.bid_px = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.bid_qty = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.bid_qa = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.bid_act_t = torch.zeros((num_envs, n_side_slots), dtype=torch.long, device=device)

        self.ask_px = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.ask_qty = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.ask_qa = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.ask_act_t = torch.zeros((num_envs, n_side_slots), dtype=torch.long, device=device)

        # Предвычисленные смещения в тиках и индексы уровней стакана для 12 слотов (6 эшелонов по 2 транша)
        # Эшелоны: [0: Penny (+1/-1), 1: L1 (0), 2: L2 (-1/+1), 3: L3 (-2/+2), 4: L4 (-3/+3), 5: L5 (-5/+5)]
        self.slot_tick_offsets = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0],
            dtype=torch.float32,
            device=device,
        ) * self.tick_size
        self.slot_lob_lvl = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.long, device=device)
        self.is_penny_slot = torch.tensor(
            [True, True, False, False, False, False, False, False, False, False, False, False],
            dtype=torch.bool,
            device=device,
        )

        # Метрики
        self.n_trades = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.partial_fills = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.multi_orders_used = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.frontrun_hits = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.sell_cycles = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prof_sell_cycles = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset_with_budgets(self, budgets_t: torch.Tensor):
        self.budget.copy_(budgets_t)
        self.cash.copy_(budgets_t)
        self.pos_qty.zero_()
        self.pos_cost_usd.zero_()
        self.avg_entry.zero_()
        self.min_buy_px.zero_()
        self.low_tranche_qty.zero_()
        self.entry_t.zero_()
        self.bid_px.zero_()
        self.bid_qty.zero_()
        self.bid_qa.zero_()
        self.bid_act_t.zero_()
        self.ask_px.zero_()
        self.ask_qty.zero_()
        self.ask_qa.zero_()
        self.ask_act_t.zero_()

    def get_observations(self, cur_t_scalar: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает LOB [1, 10, 4], Flow [B, 16], Targets [1, 4], MultiSlotSignals [B, 6]."""
        lob = self.lob_tensors[:1]
        tgt = self.targets[cur_t_scalar:cur_t_scalar + 1]
        flow, multi_6d = _jit_compute_obs_flow_and_multi(
            self.flow_market[cur_t_scalar:cur_t_scalar + 1],
            self.bids[cur_t_scalar, 0],
            self.asks[cur_t_scalar, 0],
            self.bids_q[cur_t_scalar, 0],
            self.asks_q[cur_t_scalar, 0],
            self.roll_buy_30s[cur_t_scalar],
            self.pos_qty,
            self.budget,
            self.bid_qty,
            self.bid_px,
            self.ask_qty,
            self.cash,
            self.avg_entry,
            self.entry_t,
            self.low_tranche_qty,
            self.min_buy_px,
            cur_t_scalar,
            self.num_envs,
            self.n_side_slots,
            self.tick_size,
            self.maker_fee,
        )
        return lob, flow, tgt, multi_6d

    def step(self, raw_outputs: torch.Tensor, cur_t_scalar: int):
        """
        100% Безветвевой GPU-шаг для 24 слотов (12 Bid + 12 Ask), скомпилированный в C++ TorchScript:
        raw_outputs: [B, 26]:
          0..11 : веса свободного кэша для 12 Bid-слотов
          12..23: веса свободной позиции для 12 Ask-слотов
          24    : cancel_all_bids_gate (> 0.65)
          25    : panic_sweep_gate (> 0.82)
        """
        next_t = min(cur_t_scalar + 1, self.max_steps - 1)
        (
            self.cash,
            self.pos_qty,
            self.pos_cost_usd,
            self.avg_entry,
            self.min_buy_px,
            self.low_tranche_qty,
            self.entry_t,
            self.bid_px,
            self.bid_qty,
            self.bid_qa,
            self.bid_act_t,
            self.ask_px,
            self.ask_qty,
            self.ask_qa,
            self.ask_act_t,
            self.n_trades,
            self.partial_fills,
            self.multi_orders_used,
            self.frontrun_hits,
            self.sell_cycles,
            self.prof_sell_cycles,
        ) = _jit_env_step(
            raw_outputs,
            self.bids[cur_t_scalar],
            self.asks[cur_t_scalar],
            self.bids_q[cur_t_scalar],
            self.asks_q[cur_t_scalar],
            self.bids[next_t],
            self.asks[next_t],
            self.sell_vol[next_t],
            self.buy_vol[next_t],
            self.budget,
            self.cash,
            self.pos_qty,
            self.pos_cost_usd,
            self.avg_entry,
            self.min_buy_px,
            self.low_tranche_qty,
            self.entry_t,
            self.bid_px,
            self.bid_qty,
            self.bid_qa,
            self.bid_act_t,
            self.ask_px,
            self.ask_qty,
            self.ask_qa,
            self.ask_act_t,
            self.slot_tick_offsets,
            self.slot_lob_lvl,
            self.is_penny_slot,
            self.n_trades,
            self.partial_fills,
            self.multi_orders_used,
            self.frontrun_hits,
            self.sell_cycles,
            self.prof_sell_cycles,
            cur_t_scalar,
            next_t,
            self.num_envs,
            self.n_side_slots,
            self.tick_size,
            self.maker_fee,
            self.taker_fee,
            self.allow_scratch_stop_asks,
        )

