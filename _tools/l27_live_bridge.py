#!/usr/bin/env python3
"""
_tools/l27_live_bridge.py
=============================================================================
Боевой мост Урока 27 для `live_rl_trader.py`: 1-в-1 математический паритет
с `_tools/fused_l27_multimodal_tick.py` (`K = 4` латентных экспертов,
маршрутизатор `288D -> 64 -> 4D`, веерное распределение `_fanned_slot_softmax_l27`,
`26D -> 48 -> 2D` Регулятор и Token-Bucket `+0.35 токена/с`, ёмкость `10.0`).
=============================================================================
"""

from __future__ import annotations

import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from _tools.fused_l27_multimodal_tick import (
    MKT_IDX_L27,
    AG_IDX_L27,
    K_MODES_L27,
    EXPERT_OUT_DIM_L27,
    TOKEN_REFILL_PER_SEC_L27,
    TOKEN_BUCKET_CAP_L27,
    _fanned_slot_softmax_l27,
)
from _tools.l26_live_bridge import (
    ECHELON_K01_NP,
    lookup_sparse_lob_qty_1d,
    normalize_full_480d_live,
    compute_honest_taker_cross_price,
)


def forward_l27_multimodal_live(trader, z_aug_480: torch.Tensor, acct_8d_list: list[float], now_q_ts: float) -> dict:
    """
    Прямой проход мультимодального Стратега L27:
    480D -> 384 -> ResSwiGLU 256 + GRU 16 -> Router(288->64->4) + 4x45D Experts -> 45D.
    """
    zf = z_aug_480.float()
    ag_raw_110 = zf[:, AG_IDX_L27]
    acct_8d_t = ag_raw_110[:, 54:62]
    queue_8d_t = ag_raw_110[:, 62:70]

    z_norm = normalize_full_480d_live(zf)
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
    gate_logits = (r_hid @ trader.l7_w_gate2 + trader.l7_b_gate2.view(1, K_MODES_L27)).float()
    g_mode_t = torch.softmax(gate_logits, dim=-1)  # [1, 4]

    moe_all = (h2_t @ trader.l7_w2_moe + trader.l7_b2_moe.view(1, -1)).float().view(1, K_MODES_L27, EXPERT_OUT_DIM_L27)
    strat_raw = (g_mode_t.unsqueeze(-1) * moe_all).sum(dim=1)  # [1, 45]

    raw_ebid = float(torch.sigmoid(strat_raw[0, 0]).item())
    exp_bid_frac = max(0.0, min(1.0, (raw_ebid - 0.03) / 0.97))

    tau_b = 0.30 + 2.00 * torch.sigmoid(strat_raw[:, 34:35])
    tau_a = 0.30 + 2.00 * torch.sigmoid(strat_raw[:, 35:36])
    max_share_b = 0.18 + 0.70 * torch.sigmoid(strat_raw[:, 44:45] - 0.65)
    max_share_a = 0.22 + 0.72 * torch.sigmoid(strat_raw[:, 44:45] - 0.25)

    pi_b_np = _fanned_slot_softmax_l27(strat_raw[:, 1:13], tau_b, max_share_b, floor=0.008).squeeze(0).numpy().astype(np.float32)
    pi_a_np = _fanned_slot_softmax_l27(strat_raw[:, 13:25], tau_a, max_share_a, floor=0.008).squeeze(0).numpy().astype(np.float32)
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
        "free_cash_ratio": float(acct_8d_list[2]) if len(acct_8d_list) >= 3 else 1.0,
        "now_q_ts": float(now_q_ts),
    }
    trader.last_l27_state = st
    compat_26 = np.zeros(26, dtype=np.float32)
    compat_26[:12] = pi_b_np * exp_bid_frac
    compat_26[12:24] = pi_a_np
    compat_26[24] = 1.0 if exp_bid_frac <= 1e-4 else 0.0
    compat_26[25] = 0.95 if u_taker > 0.02 else 0.0
    trader.last_l7_out = compat_26
    trader.last_l7_target_exp_frac = exp_bid_frac
    return st


def compute_l27_multimodal_plan(
    trader,
    b1: float,
    a1: float,
    now_ts: float | None = None,
    maker_fee: float = 0.001,
    api_tokens_override: float | None = None,
) -> dict:
    """Точный расчёт плана 24 слотов L27 (1-в-1 с `fused_l27_multimodal_tick.py`)."""
    st = trader.last_l27_state
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
    api_tok = float(api_tokens_override if api_tokens_override is not None else getattr(trader, "l27_api_tokens", TOKEN_BUCKET_CAP_L27))

    wide_sp = (a1_c - b1_c) >= (2.0 * tick - 1e-6)
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
    adv_risk = float(getattr(trader, "last_l7_adv", 0.0))
    if is_subcent_exec and hp_now and float(trader.avg_entry) > 0.0 and u_taker <= 0.02 and adv_risk < 0.30:
        roundtrip_fee_mult = (1.0 + maker_fee) / max(1e-6, 1.0 - maker_fee) + 0.0004
        min_fee_tp = math.ceil((ref_entry * roundtrip_fee_mult) / tick - 1e-6) * tick
        anchor_base_a = max(anchor_base_a, min_fee_tp)
    curve_a = np.power(ECHELON_K01_NP, gamma_a, dtype=np.float32)
    raw_t_ask = anchor_base_a + span_a_usd * curve_a
    floor_ask = max(b1_c + tick, min_fee_tp) if min_fee_tp > 0.0 else (b1_c + tick)
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
    api_rat = api_tok / TOKEN_BUCKET_CAP_L27
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
        "g_cnl_b": g_cnl_b,
        "g_plc_a": g_plc_a,
        "g_cnl_a": g_cnl_a,
        "u_taker": u_taker,
        "max_slip_pct": max_slip_pct,
        "api_tokens": api_tok,
        "had_b": had_b,
        "had_a": had_a,
        "hp_now": hp_now,
        "min_fee_tp": float(min_fee_tp),
        "tot_bud_eff": tot_bud_eff,
    }


def execute_live_24slot_l27(trader, b1: float, a1: float) -> None:
    """Боевое исполнение 24 слотов Урока 27 с Sparse-LOB и Token-Bucket API (+0.35/с, cap=10.0)."""
    now_ts = time.time()
    last_tok_ts = float(getattr(trader, "l27_last_tok_ts", now_ts - 1.0))
    dt_s = max(0.0, min(5.0, now_ts - last_tok_ts))
    trader.l27_api_tokens = min(
        TOKEN_BUCKET_CAP_L27,
        float(getattr(trader, "l27_api_tokens", TOKEN_BUCKET_CAP_L27)) + TOKEN_REFILL_PER_SEC_L27 * dt_s,
    )
    trader.l27_last_tok_ts = now_ts

    exec_tick = float(getattr(trader, "exec_tick_size", trader.tick_size))
    p_prec = int(getattr(trader, "price_precision", 4 if getattr(trader, "coin", "prl") == "prl" else 2))
    a_prec = int(getattr(trader, "amount_precision", 4))
    min_amt = float(getattr(trader, "min_order_amount", 2.0))

    plan = compute_l27_multimodal_plan(trader, b1, a1, now_ts=now_ts)
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
                print(f"  🚨 [L27 TAKER CROSS (u={plan['u_taker']:.0%}, slip<={plan['max_slip_pct']:.1%})] id={trader.ask_slots[0]['id']} | {sell_q:.4f} PRL @ ${p_cross:.{p_prec}f}")
            except Exception as et:
                print(f"  ⚠️ [L27 TAKER WARN]: {et}")
            trader._sync_24slot_orders(b1, count_hold=False)
            trader._save_wallet()
            return

    is_subcent_live = exec_tick < float(trader.tick_size) * 0.5
    adv_live = float(getattr(trader, "last_l7_adv", 0.0))
    calm_bid_hold = is_subcent_live and (plan["exp_bid_frac"] > 0.08) and (adv_live < 0.12) and (plan["u_taker"] <= 0.02)
    min_fee_tp_live = float(plan.get("min_fee_tp", 0.0))

    to_cnl_b, to_cnl_a = [], []
    for s in range(12):
        if plan["had_b"][s] and plan["g_cnl_b"][s] and trader.l27_api_tokens >= 1.0:
            cur_bpx = float(trader.bid_slots[s].get("px", 0.0))
            cur_bq = float(trader.bid_slots[s].get("qty", 0.0))
            if calm_bid_hold and s >= 2 and abs(float(plan["t_bid_px"][s]) - cur_bpx) < 0.0020 and float(plan["q_target_b"][s]) >= 0.65 * cur_bq:
                continue
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
            print(f"  ✖️ [L27 CANCEL BID #{s:02d}] id={oid}")
        for s, oid in to_cnl_a:
            ci = c_map.get(oid)
            if isinstance(ci, dict) and float(ci.get("filled_amount", 0.0)) - float(trader.ask_slots[s].get("acc_fill", 0.0)) <= 1e-4:
                trader.ex_coin_free += float(trader.ask_slots[s].get("qty", 0.0))
                trader.ask_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
            else:
                need_sync = True
            print(f"  ✖️ [L27 CANCEL ASK #{s:02d}] id={oid}")
        if need_sync:
            trader._sync_24slot_orders(b1, count_hold=False, pre_fetched_orders=c_map)
        else:
            time.sleep(0.12)
        plan = compute_l27_multimodal_plan(trader, b1, a1, now_ts=now_ts)

    res_after_b = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) * 1.001 for s in trader.bid_slots if s.get("id") is not None)
    avail_u = min(max(0.0, trader.bot_cash_usd * 0.998 - res_after_b), max(0.0, plan["tot_bud_eff"] * 0.996 * plan["exp_bid_frac"] - trader.pos_qty * b1 - res_after_b))
    if trader.ex_usdt_free > 0.0:
        avail_u = min(avail_u, trader.ex_usdt_free)

    b_tasks = []
    for s in sorted(range(12), key=lambda idx: float(plan["pi_bid"][idx]), reverse=True):
        if trader.bid_slots[s].get("id") is None and plan["g_plc_b"][s] and plan["q_target_b"][s] >= 2.0 and trader.l27_api_tokens >= 1.0:
            px_s = float(plan["t_bid_px"][s])
            q_s = math.floor(min(float(plan["q_target_b"][s]), (avail_u / (px_s * 1.0015)) * 0.999) * 10000.0) / 10000.0
            if q_s >= 2.0:
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
            print(f"  📥 [L27 PLACE BID #{s:02d}] id={oid_n} | {q_a:.4f} PRL @ ${p_a:.{p_prec}f} (qa0={init_qa:.1f}, mode={int(np.argmax(plan['g_mode']))})")

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
            print(f"  📤 [L27 PLACE ASK #{s:02d}] id={oid_n} | {q_a:.4f} PRL @ ${p_a:.{p_prec}f} (qa0={init_qa:.1f}, mode={int(np.argmax(plan['g_mode']))})")

    trader._refresh_legacy_slot_mirrors()
    if to_cnl_b or to_cnl_a or b_tasks or a_tasks:
        trader._save_wallet()
