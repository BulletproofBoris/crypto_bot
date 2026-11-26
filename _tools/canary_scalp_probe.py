#!/usr/bin/env python3
"""
_tools/canary_scalp_probe.py
=============================================================================
Изолированный живой канареечный зонд микро-скальпа широкого спреда (Режим B)
для калибровки очереди и перешагивания (penny-jumping) SafeTrade под Урок L83.

Параметры безопасности и изоляции:
  - Фиксированный микро-лот: 8.0 PRL (~$16, <1.5% от счёта $1,100).
  - Полная изоляция от основного чемпиона L85: основной движок видит
    core_pos_qty = max(0, trader.pos_qty - canary_pos_qty) и не отменяет
    ордера канарейки через bulk cancel_all_orders.
  - Вход ТОЛЬКО при истинном внешнем спреде >= 50 bp и выключенном RISK-OFF
    (ret_30m > -1.5%, guard_crash_900s < 0.42, BigONE спокоен).
  - Выход: динамический мейкер Ask1 - 1 тик с полом +22 bp (0..120с),
    линейным спуском пола до -30 bp (120..300с) и аварийным стопом -2.0%.
  - Автоматический Kill-Switch: отключение при 80 кругах ИЛИ cum_pnl <= -$2.50.
  - Телеметрия: data/logs/canary_scalp_telemetry.jsonl (penny_jump_dt,
    real_queue_capture_ratio, markout 10s/60s/180s, чистый PnL круга в bp/$).
=============================================================================
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CANARY_STATE_PATH = BASE_DIR / "data" / "logs" / "canary_scalp_state.json"
CANARY_LOG_PATH = BASE_DIR / "data" / "logs" / "canary_scalp_telemetry.jsonl"

CANARY_ENABLED_DEFAULT = True
CANARY_LOT_PRL = 8.0
CANARY_CASH_RESERVE_USD = 22.0
CANARY_ENTRY_SPREAD_BP = 50.0
CANARY_KEEP_SPREAD_BP = 36.0
CANARY_MAX_ROUNDS = 80
CANARY_MAX_LOSS_USD = -2.50
CANARY_MAKER_FEE = 0.0010
CANARY_TAKER_FEE = 0.0020


def _msk_iso() -> str:
    return datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _append_telemetry(ev: dict[str, Any]) -> None:
    try:
        CANARY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": round(time.time(), 3), "msk": _msk_iso(), **ev}
        with open(CANARY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_state() -> dict[str, Any]:
    default_st = {
        "enabled": CANARY_ENABLED_DEFAULT,
        "disabled_reason": "",
        "completed_rounds": 0,
        "cum_pnl_usd": 0.0,
        "cum_pnl_bp": 0.0,
        "pos_qty": 0.0,
        "entry_px": 0.0,
        "entry_ts": 0.0,
        "entry_sp_bp": 0.0,
        "entry_mid": 0.0,
        "round_id": 0,
        "bid_oid": None,
        "bid_px": 0.0,
        "bid_qty": 0.0,
        "bid_place_ts": 0.0,
        "bid_acc_fill": 0.0,
        "bid_init_sp_bp": 0.0,
        "bid_penny_jumped": False,
        "bid_penny_jump_dt": -1.0,
        "bid_tape_sell_vol": 0.0,
        "ask_oid": None,
        "ask_px": 0.0,
        "ask_qty": 0.0,
        "ask_place_ts": 0.0,
        "ask_acc_fill": 0.0,
        "pending_markouts": [],
    }
    if CANARY_STATE_PATH.exists():
        try:
            loaded = json.loads(CANARY_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                default_st.update(loaded)
        except Exception:
            pass
    return default_st


def _save_state(st: dict[str, Any]) -> None:
    try:
        CANARY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CANARY_STATE_PATH.with_name(f"{CANARY_STATE_PATH.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CANARY_STATE_PATH)
    except Exception:
        pass


def _compute_external_bbo(
    b1: float,
    a1: float,
    bp10: np.ndarray,
    bq10: np.ndarray,
    ap10: np.ndarray,
    aq10: np.ndarray,
    own_bid_px: float,
    own_bid_qty: float,
    own_ask_px: float,
    own_ask_qty: float,
    tick_sz: float,
) -> tuple[float, float]:
    """Computes external BBO after subtracting the canary's own resting order from L1."""
    ext_b1 = float(b1)
    if own_bid_qty > 1e-4 and own_bid_px > 0.0 and len(bp10) >= 2:
        for i in range(len(bp10)):
            px_i = float(bp10[i])
            q_i = float(bq10[i]) if i < len(bq10) else 0.0
            if px_i <= 0.0 or q_i <= 1e-6:
                continue
            if abs(px_i - own_bid_px) <= 0.45 * tick_sz:
                rem_q = q_i - own_bid_qty
                if rem_q >= 1.0:
                    ext_b1 = px_i
                    break
            else:
                ext_b1 = px_i
                break

    ext_a1 = float(a1)
    if own_ask_qty > 1e-4 and own_ask_px > 0.0 and len(ap10) >= 2:
        for i in range(len(ap10)):
            px_i = float(ap10[i])
            q_i = float(aq10[i]) if i < len(aq10) else 0.0
            if px_i <= 0.0 or q_i <= 1e-6:
                continue
            if abs(px_i - own_ask_px) <= 0.45 * tick_sz:
                rem_q = q_i - own_ask_qty
                if rem_q >= 1.0:
                    ext_a1 = px_i
                    break
            else:
                ext_a1 = px_i
                break

    ext_b1 = max(tick_sz, ext_b1)
    ext_a1 = max(ext_b1 + tick_sz, ext_a1)
    return ext_b1, ext_a1


def _cancel_canary_oid(trader, oid: Any) -> dict[str, Any] | None:
    if oid is None:
        return None
    try:
        rc = trader._safe_cancel_oid(oid)
        if isinstance(rc, dict):
            return rc
    except Exception:
        pass
    try:
        o = trader.client.get_order(oid)
        if isinstance(o, dict):
            return o
    except Exception:
        pass
    return None


def step_canary_scalp(
    trader,
    b1: float,
    a1: float,
    z_mkt: np.ndarray,
    bp10_live: np.ndarray,
    bq10_live: np.ndarray,
    ap10_live: np.ndarray,
    aq10_live: np.ndarray,
    roll_sell_1s: float,
    now_ts: float,
) -> dict[str, Any]:
    """
    Executes 1 tick of the isolated Canary Wide-Spread Micro-Scalper.
    Returns reservation metadata for L85 isolation:
      {
        "canary_pos_qty": float,
        "canary_cash_reserve": float,
        "canary_bid_oid": Any,
        "canary_ask_oid": Any,
      }
    """
    st = getattr(trader, "_canary_scalp_st", None)
    if st is None:
        st = _load_state()
        trader._canary_scalp_st = st

    t_sz = float(getattr(trader, "tick_size", 0.0001))
    p_prec = int(getattr(trader, "price_precision", 4))
    a_prec = int(getattr(trader, "amount_precision", 4))
    min_amt = float(getattr(trader, "min_order_amount", 2.0))
    changed = False

    # Guard against spot manual close / external wipe when restart happened
    real_spot_pos = float(getattr(trader, "pos_qty", 0.0))
    if float(st.get("pos_qty", 0.0)) > real_spot_pos + 0.5 and st.get("ask_oid") is None:
        st["pos_qty"] = max(0.0, min(float(st["pos_qty"]), real_spot_pos))
        if st["pos_qty"] < min_amt:
            st["pos_qty"] = 0.0
            st["entry_px"] = 0.0
        changed = True

    mid_now = 0.5 * (b1 + a1)

    # 1. Process pending markouts (10s / 60s / 180s post-fill)
    pm_list = list(st.get("pending_markouts", []))
    keep_pm = []
    for pm in pm_list:
        dt_m = now_ts - float(pm["fill_ts"])
        epx = max(1e-6, float(pm["fill_px"]))
        for h_s in (10, 60, 180):
            k_h = f"m_{h_s}s_bp"
            if k_h not in pm and dt_m >= float(h_s):
                pm[k_h] = round(((mid_now - epx) / epx) * 1e4, 2)
                changed = True
        if "m_180s_bp" in pm:
            _append_telemetry({
                "event": "entry_markouts",
                "round_id": pm.get("round_id"),
                "fill_px": pm.get("fill_px"),
                "entry_sp_bp": pm.get("entry_sp_bp"),
                "markout_10s_bp": pm.get("m_10s_bp"),
                "markout_60s_bp": pm.get("m_60s_bp"),
                "markout_180s_bp": pm.get("m_180s_bp"),
            })
            changed = True
        else:
            keep_pm.append(pm)
    st["pending_markouts"] = keep_pm

    # 2. Sync active canary bid order
    if st.get("bid_oid") is not None:
        st["bid_tape_sell_vol"] = float(st.get("bid_tape_sell_vol", 0.0)) + max(0.0, float(roll_sell_1s))
        try:
            ob = trader.client.get_order(st["bid_oid"])
        except Exception:
            ob = None
        if isinstance(ob, dict):
            filled_b = float(ob.get("filled_amount", 0.0))
            prev_b = float(st.get("bid_acc_fill", 0.0))
            d_b = max(0.0, filled_b - prev_b)
            state_b = str(ob.get("state", "")).lower()
            if d_b > 1e-4:
                px_b = float(ob.get("avg_price", 0.0)) or float(ob.get("price", st["bid_px"]))
                fee_b = float(ob.get("maker_fee", CANARY_MAKER_FEE) or CANARY_MAKER_FEE)
                net_q = math.floor(d_b * (1.0 - fee_b) * 10000.0) / 10000.0
                cost_u = d_b * px_b
                old_cq = float(st.get("pos_qty", 0.0))
                if old_cq < 1e-4:
                    st["round_id"] = int(st.get("round_id", 0)) + 1
                    st["entry_ts"] = now_ts
                    st["entry_px"] = px_b
                    st["entry_sp_bp"] = float(st.get("bid_init_sp_bp", 0.0))
                    st["entry_mid"] = mid_now
                    st["pending_markouts"].append({
                        "round_id": st["round_id"],
                        "fill_ts": now_ts,
                        "fill_px": px_b,
                        "entry_sp_bp": st["entry_sp_bp"],
                    })
                else:
                    st["entry_px"] = (old_cq * float(st["entry_px"]) + net_q * px_b) / max(1e-6, old_cq + net_q)
                st["pos_qty"] = round(old_cq + net_q, 4)
                st["bid_acc_fill"] = filled_b
                # Sync trader accounting so L85 & spot balance stay consistent
                trader.bot_cash_usd = max(0.0, float(trader.bot_cash_usd) - cost_u)
                trader.pos_qty = math.floor((float(trader.pos_qty) + net_q) * 10000.0) / 10000.0
                trader.ex_coin_free = max(0.0, float(getattr(trader, "ex_coin_free", 0.0)) + net_q)
                tape_sv = max(1e-4, float(st.get("bid_tape_sell_vol", 0.0)))
                cap_ratio = round(min(1.0, filled_b / max(filled_b, tape_sv)), 4)
                _append_telemetry({
                    "event": "canary_bid_fill",
                    "round_id": st["round_id"],
                    "oid": st["bid_oid"],
                    "fill_qty": round(net_q, 4),
                    "fill_px": round(px_b, 4),
                    "rest_sec": round(now_ts - float(st.get("bid_place_ts", now_ts)), 2),
                    "init_sp_bp": round(float(st.get("bid_init_sp_bp", 0.0)), 1),
                    "penny_jumped": bool(st.get("bid_penny_jumped", False)),
                    "penny_jump_dt": round(float(st.get("bid_penny_jump_dt", -1.0)), 2),
                    "tape_sell_vol": round(tape_sv, 2),
                    "capture_ratio": cap_ratio,
                })
                print(
                    f"  🐤🟢 [CANARY FILL BID | R#{st['round_id']}] +{net_q:.4f} PRL @ ${px_b:.4f} "
                    f"(sp={st.get('bid_init_sp_bp',0):.1f}bp, pj_dt={st.get('bid_penny_jump_dt',-1):.1f}s, cap={cap_ratio:.0%}) | t={_msk_iso()}"
                )
                changed = True
            if state_b in ("done", "cancel"):
                rem_unfilled = max(0.0, float(st.get("bid_qty", 0.0)) - filled_b)
                trader.ex_usdt_free = float(getattr(trader, "ex_usdt_free", 0.0)) + rem_unfilled * float(st.get("bid_px", 0.0))
                st["bid_oid"] = None
                st["bid_px"] = 0.0
                st["bid_qty"] = 0.0
                st["bid_acc_fill"] = 0.0
                changed = True

    # 3. Sync active canary ask order
    if st.get("ask_oid") is not None:
        try:
            oa = trader.client.get_order(st["ask_oid"])
        except Exception:
            oa = None
        if isinstance(oa, dict):
            filled_a = float(oa.get("filled_amount", 0.0))
            prev_a = float(st.get("ask_acc_fill", 0.0))
            d_a = max(0.0, filled_a - prev_a)
            state_a = str(oa.get("state", "")).lower()
            if d_a > 1e-4:
                px_a = float(oa.get("avg_price", 0.0)) or float(oa.get("price", st["ask_px"]))
                is_tk = px_a <= b1 + 0.25 * t_sz and (now_ts - float(st.get("ask_place_ts", now_ts))) < 1.5
                fee_a = CANARY_TAKER_FEE if is_tk else float(oa.get("maker_fee", CANARY_MAKER_FEE) or CANARY_MAKER_FEE)
                proceeds_u = d_a * px_a * (1.0 - fee_a)
                epx = max(1e-6, float(st.get("entry_px", px_a)))
                # Entry cost included buy maker fee via reduced net_q (~0.10%), so net round pnl:
                entry_cost_u = (d_a / (1.0 - CANARY_MAKER_FEE)) * epx
                pnl_u = proceeds_u - entry_cost_u
                pnl_bp = (pnl_u / max(1e-6, entry_cost_u)) * 1e4

                st["pos_qty"] = max(0.0, round(float(st.get("pos_qty", 0.0)) - d_a, 4))
                st["ask_acc_fill"] = filled_a
                st["cum_pnl_usd"] = round(float(st.get("cum_pnl_usd", 0.0)) + pnl_u, 4)
                st["cum_pnl_bp"] = round(float(st.get("cum_pnl_bp", 0.0)) + pnl_bp, 2)
                trader.bot_cash_usd = float(trader.bot_cash_usd) + proceeds_u
                trader.ex_usdt_free = float(getattr(trader, "ex_usdt_free", 0.0)) + proceeds_u
                trader.pos_qty = max(0.0, math.floor((float(trader.pos_qty) - d_a) * 10000.0) / 10000.0)

                hold_dur = round(now_ts - float(st.get("entry_ts", now_ts)), 2)
                if st["pos_qty"] < min_amt:
                    st["pos_qty"] = 0.0
                    st["completed_rounds"] = int(st.get("completed_rounds", 0)) + 1
                    _append_telemetry({
                        "event": "canary_round_complete",
                        "round_id": st.get("round_id"),
                        "completed_rounds": st["completed_rounds"],
                        "entry_px": round(epx, 4),
                        "exit_px": round(px_a, 4),
                        "qty": round(d_a, 4),
                        "hold_sec": hold_dur,
                        "entry_sp_bp": round(float(st.get("entry_sp_bp", 0.0)), 1),
                        "net_pnl_bp": round(pnl_bp, 2),
                        "net_pnl_usd": round(pnl_u, 4),
                        "cum_pnl_usd": st["cum_pnl_usd"],
                        "mean_pnl_bp": round(st["cum_pnl_bp"] / max(1, st["completed_rounds"]), 2),
                    })
                    print(
                        f"  🐤💰 [CANARY EXIT ASK | R#{st.get('round_id')} ({st['completed_rounds']}/{CANARY_MAX_ROUNDS})] "
                        f"-{d_a:.4f} PRL @ ${px_a:.4f} (hold={hold_dur:.0f}s) | Round={pnl_bp:+.1f}bp (${pnl_u:+.4f}) | "
                        f"Cum=${st['cum_pnl_usd']:+.3f} | t={_msk_iso()}"
                    )
                changed = True
            if state_a in ("done", "cancel"):
                rem_unfilled_a = max(0.0, float(st.get("ask_qty", 0.0)) - filled_a)
                trader.ex_coin_free = float(getattr(trader, "ex_coin_free", 0.0)) + rem_unfilled_a
                st["ask_oid"] = None
                st["ask_px"] = 0.0
                st["ask_qty"] = 0.0
                st["ask_acc_fill"] = 0.0
                changed = True

    # 4. Check Kill-Switch (only when flat on canary position)
    if bool(st.get("enabled", True)) and float(st.get("pos_qty", 0.0)) < min_amt:
        if int(st.get("completed_rounds", 0)) >= CANARY_MAX_ROUNDS:
            st["enabled"] = False
            st["disabled_reason"] = f"target_sample_reached_{CANARY_MAX_ROUNDS}_rounds"
            _append_telemetry({"event": "canary_kill_switch", "reason": st["disabled_reason"], "cum_pnl_usd": st["cum_pnl_usd"]})
            print(f"  🐤🏁 [CANARY TARGET REACHED] Набрано {CANARY_MAX_ROUNDS} кругов (Cum=${st['cum_pnl_usd']:+.3f}). Канарейка выключена.")
            changed = True
        elif float(st.get("cum_pnl_usd", 0.0)) <= CANARY_MAX_LOSS_USD:
            st["enabled"] = False
            st["disabled_reason"] = f"max_loss_hit_{st['cum_pnl_usd']:.3f}"
            _append_telemetry({"event": "canary_kill_switch", "reason": st["disabled_reason"], "cum_pnl_usd": st["cum_pnl_usd"]})
            print(f"  🐤🛑 [CANARY KILL-SWITCH] Достигнут лимит убытка ${st['cum_pnl_usd']:+.3f} <= ${CANARY_MAX_LOSS_USD:.2f}. Канарейка выключена.")
            changed = True

    # Compute true external BBO (subtracting canary's own order!)
    ext_b1, ext_a1 = _compute_external_bbo(
        b1=b1,
        a1=a1,
        bp10=bp10_live,
        bq10=bq10_live,
        ap10=ap10_live,
        aq10=aq10_live,
        own_bid_px=float(st.get("bid_px", 0.0)) if st.get("bid_oid") is not None else 0.0,
        own_bid_qty=float(st.get("bid_qty", 0.0)) - float(st.get("bid_acc_fill", 0.0)) if st.get("bid_oid") is not None else 0.0,
        own_ask_px=float(st.get("ask_px", 0.0)) if st.get("ask_oid") is not None else 0.0,
        own_ask_qty=float(st.get("ask_qty", 0.0)) - float(st.get("ask_acc_fill", 0.0)) if st.get("ask_oid") is not None else 0.0,
        tick_sz=t_sz,
    )
    ext_mid = 0.5 * (ext_b1 + ext_a1)
    ext_sp_bp = ((ext_a1 - ext_b1) / max(1e-6, ext_mid)) * 1e4

    # Detect penny-jumping by rival bots while our canary bid is resting
    if st.get("bid_oid") is not None and not bool(st.get("bid_penny_jumped", False)):
        if ext_b1 > float(st.get("bid_px", 0.0)) + 0.45 * t_sz:
            pj_dt = round(now_ts - float(st.get("bid_place_ts", now_ts)), 2)
            st["bid_penny_jumped"] = True
            st["bid_penny_jump_dt"] = pj_dt
            _append_telemetry({
                "event": "penny_jump_detected",
                "oid": st["bid_oid"],
                "our_bid_px": st["bid_px"],
                "rival_bid_px": round(ext_b1, 4),
                "penny_jump_dt": pj_dt,
                "rem_ext_sp_bp": round(ext_sp_bp, 1),
            })
            changed = True

    # Risk-OFF filter from causal 96D market vector:
    # z[88] = tanh(ret_30m * 12) -> -0.178 corresponds to ret_30m = -1.5%
    # z[81] = guard_crash_900s, z[31] = guard_crash_300s, z[12] = bo_basis*10, z[21] = bo_ret_30s
    z88 = float(z_mkt[88]) if len(z_mkt) > 88 else 0.0
    z81 = float(z_mkt[81]) if len(z_mkt) > 81 else 0.0
    z31 = float(z_mkt[31]) if len(z_mkt) > 31 else 0.0
    z12 = float(z_mkt[12]) if len(z_mkt) > 12 else 0.0
    z21 = float(z_mkt[21]) if len(z_mkt) > 21 else 0.0
    risk_off = (z88 < -0.178) or (z81 > 0.42) or (z31 > 0.55) or (z12 < -0.15) or (z21 < -0.50)

    # 5A. EXIT MANAGEMENT when holding canary position (>= min_amt)
    if float(st.get("pos_qty", 0.0)) >= min_amt:
        # Cancel any leftover bid immediately
        if st.get("bid_oid") is not None:
            rc_b = _cancel_canary_oid(trader, st["bid_oid"])
            if rc_b is not None:
                st["bid_oid"] = None
                st["bid_px"] = 0.0
                st["bid_qty"] = 0.0
                changed = True

        epx = max(t_sz, float(st.get("entry_px", ext_b1)))
        hold_s = max(0.0, now_ts - float(st.get("entry_ts", now_ts)))

        # Emergency -2.0% stop-loss
        if ext_b1 <= epx * 0.980:
            if st.get("ask_oid") is not None:
                _cancel_canary_oid(trader, st["ask_oid"])
                st["ask_oid"] = None
                st["ask_px"] = 0.0
                st["ask_qty"] = 0.0
            q_stop = math.floor(min(float(st["pos_qty"]), max(float(st["pos_qty"]), float(getattr(trader, "ex_coin_free", 0.0)))) * 10000.0) / 10000.0
            if q_stop >= min_amt:
                stop_px = round(max(t_sz, ext_b1 * 0.994), p_prec)
                try:
                    res_s = trader.client.place_limit_order(trader.market, "sell", q_stop, stop_px, price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt)
                    if res_s and res_s.get("id") is not None:
                        st["ask_oid"] = res_s.get("id")
                        st["ask_px"] = float(res_s.get("price", stop_px))
                        st["ask_qty"] = float(res_s.get("origin_amount", q_stop))
                        st["ask_place_ts"] = now_ts
                        st["ask_acc_fill"] = 0.0
                        print(f"  🐤⚡ [CANARY STOP-LOSS -2%] id={st['ask_oid']} | {q_stop:.4f} PRL @ >=${stop_px:.4f} | t={_msk_iso()}")
                        changed = True
                except Exception:
                    pass
        else:
            # Dynamic chasing maker ask: +22 bp floor for 0..120s, linear ramp to -30 bp at 300s, -60 bp >300s
            if hold_s <= 120.0:
                floor_mult = 1.0022
            elif hold_s <= 300.0:
                frac = (hold_s - 120.0) / 180.0
                floor_mult = 1.0022 * (1.0 - frac) + 0.9970 * frac
            else:
                floor_mult = 0.9940
            floor_px = math.ceil((epx * floor_mult) / t_sz) * t_sz
            min_safe_a = ext_b1 + 2.0 * t_sz if (ext_a1 - ext_b1) >= 2.9 * t_sz else ext_b1 + t_sz
            targ_ask = round(max(min_safe_a, floor_px, ext_a1 - t_sz), p_prec)

            if st.get("ask_oid") is not None:
                cur_apx = float(st.get("ask_px", 0.0))
                age_a = now_ts - float(st.get("ask_place_ts", 0.0))
                crossed_b = cur_apx <= ext_b1 + 0.1 * t_sz
                need_reprice_a = crossed_b or (abs(targ_ask - cur_apx) >= 1.9 * t_sz and age_a >= 3.5)
                if need_reprice_a and trader.order3_engine.api_tokens >= 2.0:
                    rc_a = _cancel_canary_oid(trader, st["ask_oid"])
                    trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                    if rc_a is not None and float(rc_a.get("filled_amount", 0.0)) <= float(st.get("ask_acc_fill", 0.0)) + 1e-4:
                        trader.ex_coin_free = float(getattr(trader, "ex_coin_free", 0.0)) + float(st.get("ask_qty", 0.0))
                        st["ask_oid"] = None
                        st["ask_px"] = 0.0
                        st["ask_qty"] = 0.0
                        changed = True

            if st.get("ask_oid") is None and trader.order3_engine.api_tokens >= 1.5:
                q_sell = math.floor(float(st["pos_qty"]) * 10000.0) / 10000.0
                if q_sell >= min_amt:
                    try:
                        res_a = trader.client.place_limit_order(trader.market, "sell", q_sell, targ_ask, price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt)
                        trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                        if res_a and res_a.get("id") is not None:
                            st["ask_oid"] = res_a.get("id")
                            st["ask_px"] = float(res_a.get("price", targ_ask))
                            st["ask_qty"] = float(res_a.get("origin_amount", q_sell))
                            st["ask_place_ts"] = now_ts
                            st["ask_acc_fill"] = 0.0
                            trader.ex_coin_free = max(0.0, float(getattr(trader, "ex_coin_free", 0.0)) - st["ask_qty"])
                            print(f"  🐤📤 [CANARY PLACE ASK] id={st['ask_oid']} | {st['ask_qty']:.4f} PRL @ ${st['ask_px']:.4f} (hold={hold_s:.0f}s) | t={_msk_iso()}")
                            changed = True
                    except Exception:
                        pass

    # 5B. ENTRY MANAGEMENT when flat on canary position
    else:
        if not bool(st.get("enabled", True)):
            if st.get("bid_oid") is not None:
                _cancel_canary_oid(trader, st["bid_oid"])
                st["bid_oid"] = None
                st["bid_px"] = 0.0
                st["bid_qty"] = 0.0
                changed = True
        else:
            max_safe_b = max(t_sz, ext_a1 - 2.0 * t_sz) if (ext_a1 - ext_b1) >= 2.9 * t_sz else max(t_sz, ext_a1 - t_sz)
            targ_bid = round(min(max_safe_b, ext_b1 + t_sz), p_prec)

            if st.get("bid_oid") is not None:
                cur_bpx = float(st.get("bid_px", 0.0))
                age_b = now_ts - float(st.get("bid_place_ts", 0.0))
                crossed_a = cur_bpx >= ext_a1 - 0.1 * t_sz
                should_pull = risk_off or (ext_sp_bp < CANARY_KEEP_SPREAD_BP) or crossed_a
                # Re-step only if rival jumped ahead AND spread is still >= 50bp (rate-limited >= 4.5s) or book dropped >15bp
                need_restep = (
                    (ext_b1 > cur_bpx + 0.5 * t_sz and ext_sp_bp >= CANARY_ENTRY_SPREAD_BP and age_b >= 4.5)
                    or ((cur_bpx - ext_b1) > 0.0015 * ext_b1 and age_b >= 3.0)
                )
                if (should_pull or need_restep) and trader.order3_engine.api_tokens >= 2.0:
                    rc_b = _cancel_canary_oid(trader, st["bid_oid"])
                    trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                    if rc_b is not None and float(rc_b.get("filled_amount", 0.0)) <= float(st.get("bid_acc_fill", 0.0)) + 1e-4:
                        trader.ex_usdt_free = float(getattr(trader, "ex_usdt_free", 0.0)) + float(st.get("bid_qty", 0.0)) * cur_bpx
                        st["bid_oid"] = None
                        st["bid_px"] = 0.0
                        st["bid_qty"] = 0.0
                        changed = True

            if (
                st.get("bid_oid") is None
                and (not risk_off)
                and ext_sp_bp >= CANARY_ENTRY_SPREAD_BP
                and trader.order3_engine.api_tokens >= 2.5
                and float(getattr(trader, "ex_usdt_free", 0.0)) >= CANARY_LOT_PRL * targ_bid * 1.02
            ):
                q_buy = CANARY_LOT_PRL
                try:
                    res_b = trader.client.place_limit_order(trader.market, "buy", q_buy, targ_bid, price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt)
                    trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                    if res_b and res_b.get("id") is not None:
                        st["bid_oid"] = res_b.get("id")
                        st["bid_px"] = float(res_b.get("price", targ_bid))
                        st["bid_qty"] = float(res_b.get("origin_amount", q_buy))
                        st["bid_place_ts"] = now_ts
                        st["bid_acc_fill"] = 0.0
                        st["bid_init_sp_bp"] = round(ext_sp_bp, 2)
                        st["bid_penny_jumped"] = False
                        st["bid_penny_jump_dt"] = -1.0
                        st["bid_tape_sell_vol"] = 0.0
                        trader.ex_usdt_free = max(0.0, float(getattr(trader, "ex_usdt_free", 0.0)) - st["bid_qty"] * st["bid_px"] * 1.001)
                        _append_telemetry({
                            "event": "canary_bid_place",
                            "oid": st["bid_oid"],
                            "px": st["bid_px"],
                            "qty": st["bid_qty"],
                            "ext_b1": round(ext_b1, 4),
                            "ext_a1": round(ext_a1, 4),
                            "ext_sp_bp": round(ext_sp_bp, 1),
                        })
                        print(f"  🐤📥 [CANARY PLACE BID] id={st['bid_oid']} | {st['bid_qty']:.1f} PRL @ ${st['bid_px']:.4f} (ext_sp={ext_sp_bp:.1f}bp) | t={_msk_iso()}")
                        changed = True
                except Exception:
                    pass

    if changed:
        _save_state(st)

    return {
        "canary_pos_qty": float(st.get("pos_qty", 0.0)),
        "canary_cash_reserve": CANARY_CASH_RESERVE_USD if bool(st.get("enabled", True)) else 0.0,
        "canary_bid_oid": st.get("bid_oid"),
        "canary_ask_oid": st.get("ask_oid"),
    }
