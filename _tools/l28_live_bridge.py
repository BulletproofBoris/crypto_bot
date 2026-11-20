#!/usr/bin/env python3
"""
_tools/l28_live_bridge.py
=============================================================================
Урок 28 (Компонент 3b): Боевой мост для `live_rl_trader.py` с 100% математическим
паритетом к `_tools/fused_l28_hierarchical_subcent_tick.py`:
  - `508D` = `398D Market` (14 семантических подблоков, 0 каналов `90s`) + `110D Agent`;
  - Иерархические деревья `SafeTrade` и `BigONE` по 7 горизонтам `[1s, 2s, 5s, 12s, 25s, 60s, 120s]`;
  - 50-уровневая сырая субцентовая книга `$0.0001` для `Sparse-LOB` и Taker-пробоя + `$0.01` корзины для `MultiTaskLOBNet`;
  - Коэволюционные `min_fee_tp` (round-trip комиссия + 4 bps) и `calm_bid_hold`.
=============================================================================
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from _tools.build_l28_hierarchical_cache import compute_subcent_micro_8d_single
from _tools.fused_l28_hierarchical_subcent_tick import (
    AG_DIM_L28,
    EXPERT_OUT_DIM_L28,
    IN_DIM_L28,
    K_MODES_L28,
    MKT_DIM_L28,
    TOKEN_BUCKET_CAP_L28,
    TOKEN_REFILL_PER_SEC_L28,
    _fanned_slot_softmax_l28,
    normalize_full_508d_live,
)
from _tools.l26_live_bridge import (
    ECHELON_K01_NP,
    compute_honest_taker_cross_price,
    lookup_sparse_lob_qty_1d,
)


def build_l28_agent_features_from_trader(
    trader,
    b1: float,
    a1: float,
    b1_q_cent: float,
    a1_q_cent: float,
    roll_b30: float,
    roll_s30: float,
    now_q_ts: float,
    maker_fee: float = 0.001,
) -> dict[str, np.ndarray | list[float]]:
    """
    Вычисляет все агентские подблоки (`multi_6d`, `b_slot_12`, `a_slot_12`, `b_dist_12`, `a_dist_12`,
    `acct_8d`, `queue_8d`, `slot_qprog_24`) на живом `trader` в 100% математическом паритете с
    `fused_l28_hierarchical_subcent_tick.py:394-507` (без устаревшего `1.002` и `tot_bud` из старых уроков).
    """
    feat_tick = float(trader.tick_size)
    exec_tick = float(getattr(trader, "exec_tick_size", feat_tick))
    b1_c = max(exec_tick, float(b1))
    a1_c = max(b1_c + exec_tick, float(a1))

    cash_v = float(trader.bot_cash_usd)
    pos_q = max(0.0, float(trader.pos_qty))
    bud_v = max(25.0, float(trader.initial_budget_usd))
    has_p = pos_q > 1e-6
    has_pos_f = 1.0 if has_p else 0.0
    pos_val = pos_q * b1_c
    tot_eq = max(1.0, cash_v + pos_val)
    pos_ratio = float(np.clip(pos_val / tot_eq, 0.0, 1.5))

    act_b_m = [s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-6 for s in trader.bid_slots]
    act_a_m = [s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-6 for s in trader.ask_slots]
    b_qty = np.array([float(trader.bid_slots[i].get("qty", 0.0)) if act_b_m[i] else 0.0 for i in range(12)], dtype=np.float32)
    b_px = np.array([float(trader.bid_slots[i].get("px", 0.0)) if act_b_m[i] else 0.0 for i in range(12)], dtype=np.float32)
    a_qty = np.array([float(trader.ask_slots[i].get("qty", 0.0)) if act_a_m[i] else 0.0 for i in range(12)], dtype=np.float32)
    a_px = np.array([float(trader.ask_slots[i].get("px", 0.0)) if act_a_m[i] else 0.0 for i in range(12)], dtype=np.float32)

    res_cash_now = float(np.sum(b_qty * b_px * (1.0 + maker_fee)))
    unres_cash_now = max(0.0, cash_v - res_cash_now)
    free_cash_ratio = float(np.clip(unres_cash_now / tot_eq, 0.0, 1.0))

    eff_pos_scale = float(np.clip(pos_ratio / 0.25, 0.15, 1.0))
    avg_ent = float(trader.avg_entry)
    unr_pct = (((b1_c - avg_ent) / max(avg_ent, feat_tick)) * 25.0 * eff_pos_scale) if (has_p and avg_ent > 0.0) else 0.0
    hold_ratio = (min(max(0.0, float(trader.hold_sec)) / 300.0, 2.0) * eff_pos_scale) if has_p else 0.0

    has_bid_f = 1.0 if any(act_b_m) else 0.0
    has_ask_f = 1.0 if any(act_a_m) else 0.0
    n_b_sl = float(sum(1 for x in act_b_m if x)) / 12.0
    n_a_sl = float(sum(1 for x in act_a_m if x)) / 12.0
    acct_8d = [
        has_pos_f,
        pos_ratio,
        free_cash_ratio,
        float(np.clip(unr_pct, -2.5, 2.5)),
        float(np.clip(hold_ratio, 0.0, 2.0)),
        has_bid_f,
        has_ask_f,
        n_b_sl - n_a_sl,
    ]

    rb = max(10.0, float(roll_b30))
    rs = max(10.0, float(roll_s30))

    b_qa_rat = np.full(12, 2.0, dtype=np.float32)
    for i in range(12):
        if act_b_m[i]:
            qa_i = float(trader.bid_slots[i].get("qa", 0.0))
            qa0_i = float(trader.bid_slots[i].get("qa0", qa_i))
            b_qa_rat[i] = float(np.clip(qa_i / (qa0_i + 10.0), 0.0, 1.5))

    if has_bid_f > 0.5:
        bi = min((i for i in range(12) if act_b_m[i]), key=lambda i: (b1_c - float(b_px[i]), i))
        bb_qa = float(trader.bid_slots[bi].get("qa", 0.0))
        bb_act = float(trader.bid_slots[bi].get("act_ts", now_q_ts))
        best_b_rat = float(b_qa_rat[bi])
        best_b_tape = float(np.clip(math.log1p(max(0.0, bb_qa) / rs) / 3.0, 0.0, 2.0))
        best_b_age = float(np.clip((now_q_ts - bb_act) / 120.0, 0.0, 2.0))
    else:
        best_b_rat, best_b_tape, best_b_age = 1.0, 0.0, 0.0

    deep_idx = [i for i in range(4, 12) if act_b_m[i]]
    deep_b_rat = float(np.mean([b_qa_rat[i] for i in deep_idx])) if deep_idx else 1.0

    a_qa_rat = np.full(12, 2.0, dtype=np.float32)
    for i in range(12):
        if act_a_m[i]:
            qa_i = float(trader.ask_slots[i].get("qa", 0.0))
            qa0_i = float(trader.ask_slots[i].get("qa0", qa_i))
            a_qa_rat[i] = float(np.clip(qa_i / (qa0_i + 10.0), 0.0, 1.5))

    if has_ask_f > 0.5:
        ai = min((i for i in range(12) if act_a_m[i]), key=lambda i: (float(a_px[i]) - a1_c, i))
        ba_qa = float(trader.ask_slots[ai].get("qa", 0.0))
        ba_act = float(trader.ask_slots[ai].get("act_ts", now_q_ts))
        best_a_rat = float(a_qa_rat[ai])
        best_a_tape = float(np.clip(math.log1p(max(0.0, ba_qa) / rb) / 3.0, 0.0, 2.0))
        best_a_age = float(np.clip((now_q_ts - ba_act) / 120.0, 0.0, 2.0))
    else:
        best_a_rat, best_a_tape, best_a_age = 1.0, 0.0, 0.0

    q_imb_adv = (1.0 - min(1.0, max(0.0, best_a_rat))) * has_ask_f - (1.0 - min(1.0, max(0.0, best_b_rat))) * has_bid_f
    queue_8d = [best_b_rat, best_b_tape, best_b_age, deep_b_rat, best_a_rat, best_a_tape, best_a_age, q_imb_adv]

    l1_b_usd = max(100.0, b1_c * max(0.0, float(b1_q_cent)))
    l1_a_usd = max(100.0, a1_c * max(0.0, float(a1_q_cent)))
    low_tr_q = float(getattr(trader, "tr_low_qty", 0.0))
    min_b_px = float(getattr(trader, "tr_low_px", 0.0))
    low_tr_rat = float(np.clip(low_tr_q / max(pos_q, 1e-6), 0.0, 1.0)) if has_p else 0.0
    lifo_ref = min_b_px if (low_tr_q > 1e-4 and min_b_px > 0.0) else avg_ent
    lifo_edge_t = float(np.clip(((b1_c - lifo_ref) / max(lifo_ref, feat_tick)) * 30.0, -2.5, 2.5)) if (has_p and lifo_ref > 0.0) else 0.0
    multi_6d = np.array(
        [
            float(math.log1p(bud_v / l1_b_usd) / 3.0),
            float(math.log1p(unres_cash_now / l1_b_usd) / 3.0),
            float(np.clip((pos_q * a1_c) / l1_a_usd, 0.0, 4.0) / 2.0),
            float(np.clip(pos_q / rb, 0.0, 10.0) / 5.0),
            low_tr_rat,
            lifo_edge_t,
        ],
        dtype=np.float32,
    )

    b_slot_12 = np.clip((b_qty * b1_c) / bud_v, 0.0, 1.0).astype(np.float32)
    a_slot_12 = np.clip(a_qty / max(pos_q, 2.0), 0.0, 1.0).astype(np.float32)
    b_dist_12 = np.where(act_b_m, np.clip(((b1_c - b_px) / max(b1_c, feat_tick)) * 12.0, -0.5, 3.0), 0.0).astype(np.float32)
    a_dist_12 = np.where(act_a_m, np.clip(((a_px - a1_c) / max(a1_c, feat_tick)) * 12.0, -0.5, 3.0), 0.0).astype(np.float32)
    b_qp_12 = np.where(act_b_m, np.clip(1.0 - b_qa_rat, -0.5, 1.0), -1.0).astype(np.float32)
    a_qp_12 = np.where(act_a_m, np.clip(1.0 - a_qa_rat, -0.5, 1.0), -1.0).astype(np.float32)
    slot_qprog_24 = np.concatenate([b_qp_12, a_qp_12], axis=0).astype(np.float32)

    return {
        "multi_6d": multi_6d,
        "b_slot_12": b_slot_12,
        "a_slot_12": a_slot_12,
        "b_dist_12": b_dist_12,
        "a_dist_12": a_dist_12,
        "acct_8d": acct_8d,
        "queue_8d": queue_8d,
        "slot_qprog_24": slot_qprog_24,
    }


def assemble_l28_508d_live(
    z_t_256: torch.Tensor,
    ps_t: torch.Tensor,
    pb_t: torch.Tensor,
    pc_t: torch.Tensor,
    pe_t: torch.Tensor,
    st_tree_42: np.ndarray,
    st_coh_4: np.ndarray,
    bo_tree_42: np.ndarray,
    bo_coh_4: np.ndarray,
    slot_18_28: list[float] | np.ndarray,
    v52_4d: list[float] | np.ndarray,
    real_2d: list[float] | np.ndarray,
    slot_52_58: list[float] | np.ndarray,
    lvel_8d: list[float] | np.ndarray,
    bp_exec_np: np.ndarray,
    bq_exec_np: np.ndarray,
    ap_exec_np: np.ndarray,
    aq_exec_np: np.ndarray,
    vb_cur: float,
    vs_cur: float,
    exec_tick: float,
    vel8_hist_deque: deque,
    multi_6d_np: np.ndarray,
    b_slot_12: list[float] | np.ndarray,
    a_slot_12: list[float] | np.ndarray,
    b_dist_12: list[float] | np.ndarray,
    a_dist_12: list[float] | np.ndarray,
    acct_8d_list: list[float] | np.ndarray,
    queue_8d_list: list[float] | np.ndarray,
    mem_h_t: torch.Tensor,
    slot_qprog_24: list[float] | np.ndarray,
) -> tuple[torch.Tensor, float]:
    """
    Собирает живой вектор `[1, 508]` (`398D Market + 110D Agent`) 1-в-1 с `build_ext_mkt_138_tensor`
    и `fused_l28_hierarchical_subcent_tick.py`. Возвращает `(z_508_t, adv_risk)`.
    """
    st_f = np.asarray(st_tree_42, dtype=np.float32).reshape(42)
    bo_f = np.asarray(bo_tree_42, dtype=np.float32).reshape(42)
    st_c4 = np.asarray(st_coh_4, dtype=np.float32).reshape(4)
    bo_c4 = np.asarray(bo_coh_4, dtype=np.float32).reshape(4)

    vel_now_8 = np.array(
        [st_f[2], st_f[8], st_f[14], st_f[20], st_f[26], st_f[32], st_f[38], bo_f[14]],
        dtype=np.float32,
    )
    vel_prev_8 = vel8_hist_deque[0] if len(vel8_hist_deque) > 0 else vel_now_8
    tree_vel_8 = np.clip(vel_now_8 - vel_prev_8, -3.0, 3.0).astype(np.float32)
    vel8_hist_deque.append(vel_now_8.copy())

    micro_8 = compute_subcent_micro_8d_single(bp_exec_np, bq_exec_np, ap_exec_np, aq_exec_np, vb_cur, vs_cur, exec_tick)
    st_1s2s_open = 1.0 if float(np.sum(np.abs(st_f[:12]))) > 1e-5 else 0.0
    micro_8[7] = 0.5 * float(micro_8[7]) + 0.5 * st_1s2s_open
    subcent_micro_16 = np.concatenate([micro_8, st_c4, bo_c4], axis=0).astype(np.float32)

    car_real_6 = np.concatenate([np.asarray(v52_4d, dtype=np.float32), np.asarray(real_2d, dtype=np.float32)], axis=0)
    ext_138 = np.concatenate(
        [
            st_f[0:12],                                  # ST 1s, 2s (12)
            st_f[12:30],                                 # ST 5s, 12s, 25s (18)
            np.asarray(slot_18_28, dtype=np.float32),    # CrossVenue 5s LeadLag (10)
            st_f[30:42],                                 # ST 60s, 120s (12)
            car_real_6,                                  # Carousel + Realism (6)
            np.asarray(slot_52_58, dtype=np.float32),    # CrossVenue 1stOrder 6D (6)
            tree_vel_8,                                  # Tree 5s Velocity 8D (8)
            np.asarray(lvel_8d, dtype=np.float32),       # Fast 5s LOB Impulse (8)
            bo_f[0:12],                                  # BO 1s, 2s (12)
            bo_f[12:30],                                 # BO 5s, 12s, 25s (18)
            bo_f[30:42],                                 # BO 60s, 120s (12)
            subcent_micro_16,                            # SubCent Micro + Coherence (16)
        ],
        axis=0,
    ).astype(np.float32)
    assert ext_138.shape == (138,), f"Bad ext_138 shape: {ext_138.shape}"

    z_mkt_398 = torch.cat(
        [
            z_t_256.float().view(1, 256),
            ps_t.float().view(1, 1),
            pb_t.float().view(1, 1),
            pc_t.float().view(1, 1),
            pe_t.float().view(1, 1),
            torch.from_numpy(ext_138).view(1, 138),
        ],
        dim=-1,
    )
    ag_94_pre = np.concatenate(
        [
            np.asarray(multi_6d_np, dtype=np.float32).reshape(6),
            np.asarray(b_slot_12, dtype=np.float32).reshape(12),
            np.asarray(a_slot_12, dtype=np.float32).reshape(12),
            np.asarray(b_dist_12, dtype=np.float32).reshape(12),
            np.asarray(a_dist_12, dtype=np.float32).reshape(12),
            np.asarray(acct_8d_list, dtype=np.float32).reshape(8),
            np.asarray(queue_8d_list, dtype=np.float32).reshape(8),
        ],
        axis=0,
    ).astype(np.float32)
    qprog_24_np = np.asarray(slot_qprog_24, dtype=np.float32).reshape(24)
    z_ag_110 = torch.cat(
        [
            torch.from_numpy(ag_94_pre).view(1, 70),
            mem_h_t.float().view(1, 16),
            torch.from_numpy(qprog_24_np).view(1, 24),
        ],
        dim=-1,
    )
    z_508 = torch.cat([z_mkt_398, z_ag_110], dim=-1)
    assert z_508.shape == (1, IN_DIM_L28)

    adv_risk = max(float(ps_t.item()), float(pb_t.item()), float(car_real_6[2]))
    return z_508, adv_risk


def forward_l28_hierarchical_live(
    trader,
    z_508: torch.Tensor,
    acct_8d_list: list[float],
    now_q_ts: float,
    adv_risk: float,
) -> dict:
    """
    Прямой проход иерархического Стратега Урока 28:
    508D -> 384 -> ResSwiGLU 256 + GRU 16 -> Router(288->64->4) + 4x45D Experts -> 45D.
    """
    zf = z_508.float()
    ag_raw_110 = zf[:, MKT_DIM_L28:]
    acct_8d_t = ag_raw_110[:, 54:62]
    queue_8d_t = ag_raw_110[:, 62:70]

    z_norm = normalize_full_508d_live(zf)
    pre_all = z_norm @ trader.l7_w1 + trader.l7_b1.squeeze(0)
    h_base = torch.relu(pre_all[..., :256])
    h_gate = (2.0 * torch.sigmoid(pre_all[..., 256:384])).repeat_interleave(2, dim=-1)
    h1_t = h_base * h_gate

    res_pre = h1_t @ trader.l7_w_res1 + trader.l7_b_res1.squeeze(0)
    res_mid = torch.relu(res_pre[..., :64]) * (2.0 * torch.sigmoid(res_pre[..., 64:128]))
    h2_t = h1_t + (res_mid @ trader.l7_w_res2)

    rec_raw = h2_t @ trader.l7_w_rec + trader.l7_b_rec.squeeze(0)
    rec_u = torch.tanh(rec_raw[..., :16])
    rec_gate = torch.sigmoid(rec_raw[..., 16:32] + 1.5)
    trader.l7_mem_h = (rec_gate * trader.l7_mem_h + (1.0 - rec_gate) * rec_u).detach()

    router_in = torch.cat([h2_t, acct_8d_t, queue_8d_t, trader.l7_mem_h], dim=-1)  # [1, 288]
    r_hid = F.silu(router_in @ trader.l7_w_gate1 + trader.l7_b_gate1.view(1, -1))
    gate_logits = (r_hid @ trader.l7_w_gate2 + trader.l7_b_gate2.view(1, K_MODES_L28)).float()
    g_mode_t = torch.softmax(gate_logits, dim=-1)

    moe_all = (h2_t @ trader.l7_w2_moe + trader.l7_b2_moe.view(1, -1)).float().view(1, K_MODES_L28, EXPERT_OUT_DIM_L28)
    strat_raw = (g_mode_t.unsqueeze(-1) * moe_all).sum(dim=1)  # [1, 45]

    raw_ebid = float(torch.sigmoid(strat_raw[0, 0]).item())
    exp_bid_frac = max(0.0, min(1.0, (raw_ebid - 0.03) / 0.97))

    tau_b = 0.30 + 2.00 * torch.sigmoid(strat_raw[:, 34:35])
    tau_a = 0.30 + 2.00 * torch.sigmoid(strat_raw[:, 35:36])
    max_share_b = 0.18 + 0.70 * torch.sigmoid(strat_raw[:, 44:45] - 0.65)
    max_share_a = 0.22 + 0.72 * torch.sigmoid(strat_raw[:, 44:45] - 0.25)

    pi_b_np = _fanned_slot_softmax_l28(strat_raw[:, 1:13], tau_b, max_share_b, floor=0.008).squeeze(0).numpy().astype(np.float32)
    pi_a_np = _fanned_slot_softmax_l28(strat_raw[:, 13:25], tau_a, max_share_a, floor=0.008).squeeze(0).numpy().astype(np.float32)
    g_mode_np = g_mode_t.squeeze(0).numpy().astype(np.float32)

    u_taker = float(torch.clamp(torch.tanh(torch.relu(strat_raw[0, 25] - 1.10)), 0.0, 1.0).item())
    max_slip_pct = float((0.005 + 0.145 * torch.sigmoid(strat_raw[0, 36])).item())
    ctx_b_np = torch.tanh(strat_raw[0, 26:30]).numpy().astype(np.float32)
    ctx_a_np = torch.tanh(strat_raw[0, 30:34]).numpy().astype(np.float32)

    st = {
        "strat_raw": strat_raw.squeeze(0).numpy().astype(np.float32),
        "g_mode": g_mode_np,
        "max_share_b": float(max_share_b.item()),
        "max_share_a": float(max_share_a.item()),
        "exp_bid_frac": exp_bid_frac,
        "pi_bid": pi_b_np,
        "pi_ask": pi_a_np,
        "u_taker": u_taker,
        "max_slip_pct": max_slip_pct,
        "ctx_4d_b": ctx_b_np,
        "ctx_4d_a": ctx_a_np,
        "adv_risk": float(adv_risk),
        "free_cash_ratio": float(acct_8d_list[2]) if len(acct_8d_list) >= 3 else 1.0,
        "now_q_ts": float(now_q_ts),
    }
    trader.last_l28_state = st
    trader.last_l27_state = st
    compat_26 = np.zeros(26, dtype=np.float32)
    compat_26[:12] = pi_b_np * exp_bid_frac
    compat_26[12:24] = pi_a_np
    compat_26[24] = 1.0 if exp_bid_frac <= 1e-4 else 0.0
    compat_26[25] = 0.95 if u_taker > 0.02 else 0.0
    trader.last_l7_out = compat_26
    trader.last_l7_target_exp_frac = exp_bid_frac
    return st


def compute_l28_hierarchical_plan(
    trader,
    b1: float,
    a1: float,
    now_ts: float | None = None,
    maker_fee: float = 0.001,
    api_tokens_override: float | None = None,
) -> dict:
    """Точный расчёт плана 24 слотов L28 (1-в-1 с `fused_l28_hierarchical_subcent_tick.py`)."""
    st = getattr(trader, "last_l28_state", None) or trader.last_l27_state
    if now_ts is None:
        now_ts = float(st.get("now_q_ts", time.time()))
    feat_tick = float(trader.tick_size)
    tick = float(getattr(trader, "exec_tick_size", feat_tick))
    neigh_tol = max(1.1 * tick, 1.1 * feat_tick)
    b1_c = max(tick, float(b1))
    a1_c = max(b1_c + tick, float(a1))

    sr = np.asarray(st["strat_raw"], dtype=np.float32)
    g_mode = np.asarray(st["g_mode"], dtype=np.float32)
    exp_bid_frac = float(st["exp_bid_frac"])
    pi_bid = np.asarray(st["pi_bid"], dtype=np.float32)
    pi_ask = np.asarray(st["pi_ask"], dtype=np.float32)
    u_taker = float(st["u_taker"])
    max_slip_pct = float(st["max_slip_pct"])
    ctx_4d_b = np.asarray(st["ctx_4d_b"], dtype=np.float32)
    ctx_4d_a = np.asarray(st["ctx_4d_a"], dtype=np.float32)
    adv_risk = float(st.get("adv_risk", 0.0))
    api_tok = float(api_tokens_override if api_tokens_override is not None else getattr(trader, "l27_api_tokens", TOKEN_BUCKET_CAP_L28))

    wide_sp = (a1_c - b1_c) >= (2.0 * tick - 1e-7)
    penny_room_b = tick if wide_sp else 0.0

    sig = lambda x: 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(x)))))
    softplus = lambda x: float(x) if float(x) > 20.0 else math.log1p(math.exp(max(-30.0, float(x))))

    front_b_shift = -penny_room_b * sig(sr[37]) + b1_c * 0.04 * softplus(sr[38] - 1.8) * 0.35
    span_b_usd = b1_c * (0.015 + 0.235 * sig(sr[39]))
    gamma_b = 0.65 + 1.65 * sig(sr[40])
    curve_b = np.power(ECHELON_K01_NP, gamma_b, dtype=np.float32)
    raw_t_bid = b1_c - front_b_shift - span_b_usd * curve_b
    max_penny_b = max(tick, a1_c - tick)
    t_bid_px = np.maximum(tick, np.round(np.minimum(raw_t_bid, max_penny_b) / tick) * tick).astype(np.float32)

    pos_q = max(0.0, float(trader.pos_qty))
    hp_now = pos_q >= 1.95
    ref_entry = float(trader.avg_entry) if (hp_now and trader.avg_entry > 0.0) else a1_c
    skew_blend = sig(sr[41] + 1.2)
    ask_edge_pct = 0.04 * math.tanh(float(sr[42]))
    span_a_usd = a1_c * (0.012 + 0.238 * sig(sr[39] + 0.5 * sr[42]))
    gamma_a = 0.65 + 1.65 * sig(sr[40])
    anchor_base_a = a1_c + skew_blend * (ref_entry - a1_c) + a1_c * ask_edge_pct - penny_room_b * sig(-sr[41])

    min_fee_tp = 0.0
    is_subcent_exec = tick < feat_tick * 0.5
    if is_subcent_exec and hp_now and float(trader.avg_entry) > 0.0 and u_taker <= 0.02 and adv_risk < 0.30:
        roundtrip_fee_mult = (1.0 + maker_fee) / max(1e-6, 1.0 - maker_fee) + 0.0004
        min_fee_tp = math.ceil((ref_entry * roundtrip_fee_mult) / tick - 1e-6) * tick
        anchor_base_a = max(anchor_base_a, min_fee_tp)

    curve_a = np.power(ECHELON_K01_NP, gamma_a, dtype=np.float32)
    raw_t_ask = anchor_base_a + span_a_usd * curve_a
    floor_ask = max(b1_c + tick, min_fee_tp)
    t_ask_px = np.maximum(floor_ask, np.round(raw_t_ask / tick) * tick).astype(np.float32)

    cash_tot = max(0.0, float(trader.bot_cash_usd))
    tot_bud_eff = max(float(trader.initial_budget_usd), cash_tot + pos_q * b1_c, 25.0)
    target_bid_pool_usd = max(0.0, tot_bud_eff * 0.996 * exp_bid_frac - pos_q * b1_c)
    q_target_b = (target_bid_pool_usd * pi_bid) / (t_bid_px * (1.0 + maker_fee))
    for k in range(0, 12, 2):
        q0, q1 = float(q_target_b[k]), float(q_target_b[k + 1])
        if (0.05 < q0 < 2.0) or (0.05 < q1 < 2.0):
            if float(pi_bid[k]) >= float(pi_bid[k + 1]):
                q_target_b[k], q_target_b[k + 1] = q0 + q1, 0.0
            else:
                q_target_b[k], q_target_b[k + 1] = 0.0, q0 + q1
    q_target_b = np.where(q_target_b >= 2.0, q_target_b, 0.0).astype(np.float32)

    q_target_a = (pos_q * pi_ask).astype(np.float32) if hp_now else np.zeros(12, dtype=np.float32)
    dust_m = (q_target_a > 0.0) & (q_target_a < 2.0)
    dust_s = float(np.sum(np.where(dust_m, q_target_a, 0.0)))
    q_target_a = np.where(dust_m, 0.0, q_target_a)
    if hp_now and dust_s >= 0.01:
        bi = int(np.argmax(pi_ask))
        q_target_a[bi] = min(pos_q, float(q_target_a[bi]) + dust_s)
    q_target_a = np.where(q_target_a >= 2.0, q_target_a, 0.0).astype(np.float32)

    had_b = np.array([s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-6 for s in trader.bid_slots], dtype=bool)
    had_a = np.array([s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-6 for s in trader.ask_slots], dtype=bool)
    b_qty = np.array([float(s.get("qty", 0.0)) if had_b[i] else 0.0 for i, s in enumerate(trader.bid_slots)], dtype=np.float32)
    b_px = np.array([float(s.get("px", 0.0)) if had_b[i] else 0.0 for i, s in enumerate(trader.bid_slots)], dtype=np.float32)
    a_qty = np.array([float(s.get("qty", 0.0)) if had_a[i] else 0.0 for i, s in enumerate(trader.ask_slots)], dtype=np.float32)
    a_px = np.array([float(s.get("px", 0.0)) if had_a[i] else 0.0 for i, s in enumerate(trader.ask_slots)], dtype=np.float32)

    cur_val_b = np.clip((b_qty * b_px) / max(tot_bud_eff, 25.0), 0.0, 1.5)
    targ_val_b = np.clip((q_target_b * t_bid_px) / max(tot_bud_eff, 25.0), 0.0, 1.5)
    cur_val_a = np.clip(a_qty / max(pos_q, 2.0), 0.0, 1.5)
    targ_val_a = np.clip(q_target_a / max(pos_q, 2.0), 0.0, 1.5)

    qp_b = np.array([float(np.clip(1.0 - float(s.get("qa", 0.0)) / (float(s.get("qa0", 0.0)) + 10.0), 0.0, 1.0)) if had_b[i] else 0.0 for i, s in enumerate(trader.bid_slots)], dtype=np.float32)
    qp_a = np.array([float(np.clip(1.0 - float(s.get("qa", 0.0)) / (float(s.get("qa0", 0.0)) + 10.0), 0.0, 1.0)) if had_a[i] else 0.0 for i, s in enumerate(trader.ask_slots)], dtype=np.float32)
    tot_side_b = float(np.sum(cur_val_b))
    tot_side_a = float(np.sum(cur_val_a))
    mean_qp_b = float(np.sum(qp_b * cur_val_b) / max(0.05, tot_side_b))
    mean_qp_a = float(np.sum(qp_a * cur_val_a) / max(0.05, tot_side_a))
    res_now_u = float(np.sum(b_qty * b_px * (1.0 + maker_fee)))
    free_b = float(np.clip(max(0.0, cash_tot - res_now_u) / max(1.0, cash_tot + pos_q * b1_c), 0.0, 1.0))
    free_a = float(np.clip((pos_q - float(np.sum(a_qty))) / max(pos_q, 2.0), 0.0, 1.0))
    api_rat = api_tok / TOKEN_BUCKET_CAP_L28
    sp_rel = float(np.clip(((a1_c - b1_c) / b1_c) * 40.0, 0.0, 3.0))
    gm_list = g_mode.tolist()

    fb_rows, fa_rows = [], []
    for i in range(12):
        sl_b = trader.bid_slots[i]
        age_b = math.log1p(max(0.0, now_ts - float(sl_b.get("act_ts", now_ts)))) / 6.0 if had_b[i] else 0.0
        lqa_b = math.log1p(max(0.0, float(sl_b.get("qa", 0.0)))) / 10.0 if had_b[i] else 0.0
        dr_b = float(np.clip(((t_bid_px[i] - b_px[i]) / b1_c) * 25.0, -2.0, 2.0)) if had_b[i] else 0.0
        rd_b = float(np.clip((q_target_b[i] - b_qty[i]) / max(b_qty[i], q_target_b[i], 2.0), -2.0, 2.0))
        dp_b = float(np.clip(((b1_c - t_bid_px[i]) / b1_c) * 8.0, -0.2, 2.5))
        fr_b = 1.0 if t_bid_px[i] >= b1_c - 0.5 * tick else 0.0
        nv_b = float(np.clip(sum(cur_val_b[j] for j in range(12) if j != i and had_b[j] and abs(b_px[j] - t_bid_px[i]) <= neigh_tol), 0.0, 1.5))
        fb_rows.append([0.0, float(had_b[i]), cur_val_b[i], targ_val_b[i], rd_b, dr_b, qp_b[i], lqa_b, age_b, dp_b, fr_b, free_b, *ctx_4d_b.tolist(), tot_side_b, nv_b, mean_qp_b, api_rat, float(ECHELON_K01_NP[i]), sp_rel, *gm_list])

        sl_a = trader.ask_slots[i]
        age_a = math.log1p(max(0.0, now_ts - float(sl_a.get("act_ts", now_ts)))) / 6.0 if had_a[i] else 0.0
        lqa_a = math.log1p(max(0.0, float(sl_a.get("qa", 0.0)))) / 10.0 if had_a[i] else 0.0
        dr_a = float(np.clip(((a_px[i] - t_ask_px[i]) / a1_c) * 25.0, -2.0, 2.0)) if had_a[i] else 0.0
        rd_a = float(np.clip((q_target_a[i] - a_qty[i]) / max(a_qty[i], q_target_a[i], 2.0), -2.0, 2.0))
        dp_a = float(np.clip(((t_ask_px[i] - a1_c) / a1_c) * 8.0, -0.2, 2.5))
        fr_a = 1.0 if t_ask_px[i] <= a1_c + 0.5 * tick else 0.0
        nv_a = float(np.clip(sum(cur_val_a[j] for j in range(12) if j != i and had_a[j] and abs(a_px[j] - t_ask_px[i]) <= neigh_tol), 0.0, 1.5))
        fa_rows.append([1.0, float(had_a[i]), cur_val_a[i], targ_val_a[i], rd_a, dr_a, qp_a[i], lqa_a, age_a, dp_a, fr_a, free_a, *ctx_4d_a.tolist(), tot_side_a, nv_a, mean_qp_a, api_rat, float(ECHELON_K01_NP[i]), sp_rel, *gm_list])

    with torch.no_grad():
        fb_t = torch.tensor(fb_rows, dtype=torch.float32)
        fa_t = torch.tensor(fa_rows, dtype=torch.float32)
        rb_h = F.silu(fb_t @ trader.l7_w_reg1 + trader.l7_b_reg1_b.view(1, -1))
        ra_h = F.silu(fa_t @ trader.l7_w_reg1 + trader.l7_b_reg1_a.view(1, -1))
        rb_o = (rb_h @ trader.l7_w_reg2 + trader.l7_b_reg2_b.view(1, 2)).numpy()
        ra_o = (ra_h @ trader.l7_w_reg2 + trader.l7_b_reg2_a.view(1, 2)).numpy()

    g_plc_b, g_cnl_b = rb_o[:, 0] > 0.0, rb_o[:, 1] > 0.0
    g_plc_a, g_cnl_a = ra_o[:, 0] > 0.0, ra_o[:, 1] > 0.0

    calm_bid_hold = is_subcent_exec and (exp_bid_frac > 0.08) and (adv_risk < 0.12) and (u_taker <= 0.02)
    eff_cnl_b = g_cnl_b.copy()
    if calm_bid_hold:
        for s in range(2, 12):
            if had_b[s] and abs(float(t_bid_px[s]) - float(b_px[s])) < 0.0020 and float(q_target_b[s]) >= 0.65 * float(b_qty[s]):
                eff_cnl_b[s] = False

    return {
        "g_mode": g_mode,
        "exp_bid_frac": exp_bid_frac,
        "pi_bid": pi_bid,
        "pi_ask": pi_ask,
        "t_bid_px": t_bid_px,
        "t_ask_px": t_ask_px,
        "q_target_b": q_target_b,
        "q_target_a": q_target_a,
        "g_plc_b": g_plc_b,
        "g_cnl_b": eff_cnl_b,
        "g_plc_a": g_plc_a,
        "g_cnl_a": g_cnl_a,
        "u_taker": u_taker,
        "max_slip_pct": max_slip_pct,
        "adv_risk": adv_risk,
        "calm_bid_hold": calm_bid_hold,
        "api_tokens": api_tok,
        "had_b": had_b,
        "had_a": had_a,
        "hp_now": hp_now,
        "min_fee_tp": float(min_fee_tp),
        "tot_bud_eff": tot_bud_eff,
    }


def execute_live_24slot_l28(trader, b1: float, a1: float) -> None:
    """Боевое исполнение 24 слотов Урока 28 с 50-уровневым Sparse-LOB $0.0001 и Token-Bucket API."""
    now_ts = time.time()
    last_tok_ts = float(getattr(trader, "l27_last_tok_ts", now_ts - 1.0))
    dt_s = max(0.0, min(5.0, now_ts - last_tok_ts))
    trader.l27_api_tokens = min(
        TOKEN_BUCKET_CAP_L28,
        float(getattr(trader, "l27_api_tokens", TOKEN_BUCKET_CAP_L28)) + TOKEN_REFILL_PER_SEC_L28 * dt_s,
    )
    trader.l27_last_tok_ts = now_ts

    exec_tick = float(getattr(trader, "exec_tick_size", trader.tick_size))
    p_prec = int(getattr(trader, "price_precision", 4 if getattr(trader, "coin", "prl") == "prl" else 2))
    a_prec = int(getattr(trader, "amount_precision", 4))
    min_amt = float(getattr(trader, "min_order_amount", 2.0))

    plan = compute_l28_hierarchical_plan(trader, b1, a1, now_ts=now_ts)
    bp_ref = getattr(trader, "last_clean_bp_10", np.full(10, b1, dtype=np.float32))
    bq_ref = getattr(trader, "last_clean_bq_10", np.zeros(10, dtype=np.float32))
    ap_ref = getattr(trader, "last_clean_ap_10", np.full(10, a1, dtype=np.float32))
    aq_ref = getattr(trader, "last_clean_aq_10", np.zeros(10, dtype=np.float32))
    l1_bid_u = max(100.0, float(b1) * max(0.0, float(bq_ref[0]) if len(bq_ref) > 0 else 0.0))
    l1_ask_u = max(100.0, float(a1) * max(0.0, float(aq_ref[0]) if len(aq_ref) > 0 else 0.0))

    if plan["u_taker"] > 0.02 and trader.pos_qty >= 1.95 and trader.l27_api_tokens >= 1.0:
        avail_pos = min(trader.pos_qty, max(trader.ex_coin_free, trader.pos_qty))
        raw_sq = avail_pos * plan["u_taker"]
        sell_q = avail_pos if (avail_pos - raw_sq) < min_amt else raw_sq
        sell_q = math.floor(sell_q * 10000.0) / 10000.0
        p_cross, cross_avail_q = compute_honest_taker_cross_price(bp_ref, bq_ref, sell_q, b1, plan["max_slip_pct"], exec_tick)
        if sell_q >= min_amt and cross_avail_q >= 1e-2:
            ask_oids = [s["id"] for s in trader.ask_slots if s.get("id") is not None]
            if ask_oids:
                list(trader._io_pool.map(trader._safe_cancel_oid, ask_oids))
                for s in trader.ask_slots:
                    s["id"], s["px"], s["qty"], s["act_ts"], s["acc_fill"] = None, 0.0, 0.0, 0.0, 0.0
                trader._sync_24slot_orders(b1, count_hold=False)
                trader.sync_account_balances(b1)
            trader.l27_api_tokens = max(0.0, trader.l27_api_tokens - 1.0 - float(len(ask_oids)))
            try:
                res_t = trader.client.place_limit_order(
                    trader.market, "sell", sell_q, p_cross,
                    price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt,
                )
                trader.ask_slots[0] = {
                    "slot": 0, "id": res_t.get("id"), "px": float(res_t.get("price", p_cross)),
                    "qty": float(res_t.get("origin_amount", sell_q)), "act_ts": now_ts, "acc_fill": 0.0,
                }
                print(f"  🚨 [L28 TAKER CROSS (u={plan['u_taker']:.0%}, slip<={plan['max_slip_pct']:.1%})] id={trader.ask_slots[0]['id']} | {sell_q:.4f} PRL @ ${p_cross:.{p_prec}f}")
            except Exception as et:
                print(f"  ⚠️ [L28 TAKER WARN]: {et}")
            trader._sync_24slot_orders(b1, count_hold=False)
            trader._save_wallet()
            return

    min_fee_tp_live = float(plan.get("min_fee_tp", 0.0))
    to_cnl_b, to_cnl_a = [], []
    for s in range(12):
        if plan["had_b"][s] and plan["g_cnl_b"][s] and trader.l27_api_tokens >= 1.0:
            to_cnl_b.append((s, trader.bid_slots[s]["id"]))
            trader.l27_api_tokens = max(0.0, trader.l27_api_tokens - 1.0)
    for s in range(12):
        cur_apx = float(trader.ask_slots[s].get("px", 0.0)) if plan["had_a"][s] else 0.0
        under_fee_tp = plan["hp_now"] and (min_fee_tp_live > 0.0) and (cur_apx < min_fee_tp_live - 0.4 * exec_tick)
        must_clear = (not plan["hp_now"]) or under_fee_tp
        if plan["had_a"][s] and (must_clear or (plan["g_cnl_a"][s] and trader.l27_api_tokens >= 1.0)):
            to_cnl_a.append((s, trader.ask_slots[s]["id"]))
            if not must_clear:
                trader.l27_api_tokens = max(0.0, trader.l27_api_tokens - 1.0)

    if to_cnl_b or to_cnl_a:
        all_oids = [x[1] for x in (to_cnl_b + to_cnl_a)]
        c_map = {oid: rc for oid, rc in zip(all_oids, trader._io_pool.map(trader._safe_cancel_oid, all_oids)) if isinstance(rc, dict)}
        need_sync = False
        for s, oid in to_cnl_b:
            ci = c_map.get(oid)
            if isinstance(ci, dict) and float(ci.get("filled_amount", 0.0)) - float(trader.bid_slots[s].get("acc_fill", 0.0)) <= 1e-4:
                trader.ex_usdt_free += float(trader.bid_slots[s].get("qty", 0.0)) * float(trader.bid_slots[s].get("px", 0.0))
                trader.bid_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
            else:
                need_sync = True
            print(f"  ✖️ [L28 CANCEL BID #{s:02d}] id={oid}")
        for s, oid in to_cnl_a:
            ci = c_map.get(oid)
            if isinstance(ci, dict) and float(ci.get("filled_amount", 0.0)) - float(trader.ask_slots[s].get("acc_fill", 0.0)) <= 1e-4:
                trader.ex_coin_free += float(trader.ask_slots[s].get("qty", 0.0))
                trader.ask_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
            else:
                need_sync = True
            print(f"  ✖️ [L28 CANCEL ASK #{s:02d}] id={oid}")
        if need_sync:
            trader._sync_24slot_orders(b1, count_hold=False, pre_fetched_orders=c_map)
        else:
            time.sleep(0.12)
        plan = compute_l28_hierarchical_plan(trader, b1, a1, now_ts=now_ts)

    res_after_b = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) * 1.001 for s in trader.bid_slots if s.get("id") is not None)
    avail_u = min(
        max(0.0, trader.bot_cash_usd * 0.998 - res_after_b),
        max(0.0, plan["tot_bud_eff"] * 0.996 * plan["exp_bid_frac"] - trader.pos_qty * b1 - res_after_b),
    )
    if trader.ex_usdt_free > 0.0:
        avail_u = min(avail_u, trader.ex_usdt_free)

    b_tasks = []
    for s in sorted(range(12), key=lambda idx: float(plan["pi_bid"][idx]), reverse=True):
        if trader.bid_slots[s].get("id") is None and plan["g_plc_b"][s] and plan["q_target_b"][s] >= min_amt and trader.l27_api_tokens >= 1.0:
            px_s = float(plan["t_bid_px"][s])
            q_s = math.floor(min(float(plan["q_target_b"][s]), (avail_u / (px_s * 1.0015)) * 0.999) * 10000.0) / 10000.0
            if q_s >= min_amt:
                b_tasks.append((s, q_s, px_s))
                avail_u = max(0.0, avail_u - q_s * px_s * 1.0015)
                trader.l27_api_tokens = max(0.0, trader.l27_api_tokens - 1.0)

    softplus = lambda x: float(x) if float(x) > 20.0 else math.log1p(math.exp(max(-30.0, float(x))))
    for s, res_o, q_req, p_req in (list(trader._io_pool.map(lambda it: trader._place_slot_order("buy", it[0], it[1], it[2]), b_tasks)) if b_tasks else []):
        if res_o and res_o.get("id") is not None:
            oid_n, q_a, p_a = res_o.get("id"), float(res_o.get("origin_amount", q_req)), float(res_o.get("price", p_req))
            base_qa = 0.0 if p_a > b1 + 0.1 * exec_tick else lookup_sparse_lob_qty_1d(bp_ref, bq_ref, p_a, exec_tick)
            ov = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (q_a * p_a - 50.0) / 18.0))))
            dsat = softplus((q_a * p_a) / l1_bid_u - 0.85)
            init_qa = base_qa * (1.0 + 0.75 * ov) + ov * (0.45 * q_a) + 0.35 * dsat * q_a
            trader.ex_usdt_free = max(0.0, trader.ex_usdt_free - q_a * p_a * 1.001)
            trader.bid_slots[s] = {"slot": s, "id": oid_n, "px": p_a, "qty": q_a, "act_ts": now_ts, "acc_fill": 0.0, "qa_oid": oid_n, "qa0": init_qa, "qa": init_qa}
            print(f"  📥 [L28 PLACE BID #{s:02d}] id={oid_n} | {q_a:.4f} PRL @ ${p_a:.{p_prec}f} (qa0={init_qa:.1f}, mode={int(np.argmax(plan['g_mode']))})")

    avail_c = max(0.0, trader.ex_coin_free - 0.0005) if trader.ex_coin_free > 0.0 else max(0.0, trader.pos_qty - sum(float(s.get("qty", 0.0)) for s in trader.ask_slots if s.get("id") is not None))
    a_tasks = []
    for s in sorted(range(12), key=lambda idx: float(plan["pi_ask"][idx]), reverse=True):
        if trader.ask_slots[s].get("id") is None and plan["hp_now"] and plan["g_plc_a"][s] and plan["q_target_a"][s] >= min_amt and trader.l27_api_tokens >= 1.0:
            px_s = float(plan["t_ask_px"][s])
            q_s = math.floor(min(float(plan["q_target_a"][s]), avail_c) * 10000.0) / 10000.0
            if q_s >= min_amt:
                a_tasks.append((s, q_s, px_s))
                avail_c = max(0.0, avail_c - q_s)
                trader.l27_api_tokens = max(0.0, trader.l27_api_tokens - 1.0)

    for s, res_o, q_req, p_req in (list(trader._io_pool.map(lambda it: trader._place_slot_order("sell", it[0], it[1], it[2]), a_tasks)) if a_tasks else []):
        if res_o and res_o.get("id") is not None:
            oid_n, q_a, p_a = res_o.get("id"), float(res_o.get("origin_amount", q_req)), float(res_o.get("price", p_req))
            base_qa = 0.0 if p_a < a1 - 0.1 * exec_tick else lookup_sparse_lob_qty_1d(ap_ref, aq_ref, p_a, exec_tick)
            ov = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (q_a * p_a - 50.0) / 18.0))))
            dsat = softplus((q_a * p_a) / l1_ask_u - 0.85)
            init_qa = base_qa * (1.0 + 0.75 * ov) + ov * (0.45 * q_a) + 0.35 * dsat * q_a
            trader.ex_coin_free = max(0.0, trader.ex_coin_free - q_a)
            trader.ask_slots[s] = {"slot": s, "id": oid_n, "px": p_a, "qty": q_a, "act_ts": now_ts, "acc_fill": 0.0, "qa_oid": oid_n, "qa0": init_qa, "qa": init_qa}
            print(f"  📤 [L28 PLACE ASK #{s:02d}] id={oid_n} | {q_a:.4f} PRL @ ${p_a:.{p_prec}f} (qa0={init_qa:.1f}, mode={int(np.argmax(plan['g_mode']))})")

    trader._refresh_legacy_slot_mirrors()
    if to_cnl_b or to_cnl_a or b_tasks or a_tasks:
        trader._save_wallet()
