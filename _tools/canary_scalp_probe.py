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
CANARY_NEURAL_MODEL_PATH = BASE_DIR / "models" / "prl" / "canary_neural_v1.pt"
CANARY_PAUSE_FLAG_PATH = BASE_DIR / "data" / "logs" / "canary_pause.flag"
# Canary v2 (Canary_Neural_v2_PureRaw): переобучена на честной физике и шаге $0.0001; для ручной паузы на лету используйте файл data/logs/canary_pause.flag
CANARY_PAUSED = False

CANARY_ENABLED_DEFAULT = True
CANARY_LOT_PRL = 8.0
CANARY_CASH_RESERVE_USD = 22.0
CANARY_MAX_ROUNDS = 80
CANARY_MAX_LOSS_USD = -2.50
CANARY_MAKER_FEE = 0.0010
CANARY_TAKER_FEE = 0.0020
# Физические защитные рубежи канарейки (страховка, а не торговая логика):
CANARY_HARD_STOP_PCT = -0.020   # аварийный выход при падении Bid на 2% ниже цены входа
CANARY_MAX_HOLD_S = 1800.0      # предельное удержание позиции 30 мин (живые данные: markout < 0 после 180с, убыточные круги висели 70-130 мин)
CANARY_STEP_INTERVAL_S = 2.0    # инференс нейро-канарейки раз в 2с (<2 мс CPU/с)
CANARY_ORDER_POLL_S = 2.5       # опрос REST get_order раз в 2.5с (или мгновенно при касании цены/потоке сделок)

_CANARY_ENGINE_CACHE: dict[str, Any] = {"engine": None, "mtime": 0.0}


def _get_canary_neural_engine():
    if not CANARY_NEURAL_MODEL_PATH.exists():
        return None
    mt = float(CANARY_NEURAL_MODEL_PATH.stat().st_mtime)
    if _CANARY_ENGINE_CACHE["engine"] is None or abs(mt - float(_CANARY_ENGINE_CACHE["mtime"])) > 1e-3:
        from _tools.order3_live_bridge import Order3LiveInferenceEngine
        eng = Order3LiveInferenceEngine(CANARY_NEURAL_MODEL_PATH, device="cpu")
        _CANARY_ENGINE_CACHE["engine"] = eng
        _CANARY_ENGINE_CACHE["mtime"] = mt
        print(f"  🐤🧠 [CANARY NEURAL LOADED] Активирован автономный нейро-движок {CANARY_NEURAL_MODEL_PATH.name} (Mode#0 + Персональный Оркестратор)!")
    return _CANARY_ENGINE_CACHE["engine"]


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

    # ВАЖНО: реальный биржевой шаг цены ($0.0001) = exec_tick_size; trader.tick_size — это признаковый шаг $0.01 (НЕ для ордеров!).
    t_sz = float(getattr(trader, "exec_tick_size", 0.0) or 0.0) or float(getattr(trader, "tick_size", 0.0001))
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

    # 2. Sync active canary bid order (throttled REST poll unless tape/touch indicates potential fill)
    if st.get("bid_oid") is not None:
        st["bid_tape_sell_vol"] = float(st.get("bid_tape_sell_vol", 0.0)) + max(0.0, float(roll_sell_1s))
        bid_px_cur = float(st.get("bid_px", 0.0))
        need_b_poll = (
            (now_ts - float(st.get("last_bid_poll_ts", 0.0))) >= CANARY_ORDER_POLL_S
            or (roll_sell_1s > 1e-4 and b1 <= bid_px_cur + 1.5 * t_sz)
            or (a1 <= bid_px_cur + 0.5 * t_sz)
        )
        ob = None
        if need_b_poll:
            st["last_bid_poll_ts"] = now_ts
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
                    st["round_pnl_usd"] = 0.0
                    st["round_cost_usd"] = 0.0
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

    # 3. Sync active canary ask order (throttled REST poll unless price touches ask)
    if st.get("ask_oid") is not None:
        ask_px_cur = float(st.get("ask_px", 0.0))
        need_a_poll = (
            (now_ts - float(st.get("last_ask_poll_ts", 0.0))) >= CANARY_ORDER_POLL_S
            or (a1 >= ask_px_cur - 1.5 * t_sz)
            or (b1 >= ask_px_cur - 0.5 * t_sz)
        )
        oa = None
        if need_a_poll:
            st["last_ask_poll_ts"] = now_ts
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
                # Раунд считается целиком (сумма всех частичных продаж), а не по каждому частичному исполнению:
                st["round_pnl_usd"] = float(st.get("round_pnl_usd", 0.0)) + pnl_u
                st["round_cost_usd"] = float(st.get("round_cost_usd", 0.0)) + entry_cost_u
                trader.bot_cash_usd = float(trader.bot_cash_usd) + proceeds_u
                trader.ex_usdt_free = float(getattr(trader, "ex_usdt_free", 0.0)) + proceeds_u
                trader.pos_qty = max(0.0, math.floor((float(trader.pos_qty) - d_a) * 10000.0) / 10000.0)

                hold_dur = round(now_ts - float(st.get("entry_ts", now_ts)), 2)
                already_closed = st.get("last_closed_round") == st.get("round_id")
                if st["pos_qty"] < min_amt and already_closed:
                    # Хвост частично исполненного Ask после закрытия раунда: учитываем деньги, но НЕ считаем новый раунд.
                    st["pos_qty"] = 0.0
                    st["round_pnl_usd"] = 0.0
                    st["round_cost_usd"] = 0.0
                    _append_telemetry({
                        "event": "canary_exit_residual",
                        "round_id": st.get("round_id"),
                        "qty": round(d_a, 4),
                        "exit_px": round(px_a, 4),
                        "net_pnl_usd": round(pnl_u, 4),
                        "cum_pnl_usd": st["cum_pnl_usd"],
                    })
                elif st["pos_qty"] < min_amt:
                    st["pos_qty"] = 0.0
                    st["completed_rounds"] = int(st.get("completed_rounds", 0)) + 1
                    st["last_closed_round"] = st.get("round_id")
                    r_cost = max(1e-6, float(st.get("round_cost_usd", entry_cost_u)))
                    r_pnl = float(st.get("round_pnl_usd", pnl_u))
                    pnl_u = r_pnl
                    pnl_bp = (r_pnl / r_cost) * 1e4
                    st["cum_pnl_bp"] = round(float(st.get("cum_pnl_bp", 0.0)) + pnl_bp, 2)
                    st["round_pnl_usd"] = 0.0
                    st["round_cost_usd"] = 0.0
                    forced_r = str(st.get("force_reason", "") or "")
                    st["force_reason"] = ""
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
                        "forced_exit": forced_r,
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

    # 5. 100% ЭНДОГЕННОЕ НЕЙРОСЕТЕВОЕ УПРАВЛЕНИЕ (без единого эмпирического порога спреда/таймера!):
    # ПАУЗА: при CANARY_PAUSED=True или наличии файла data/logs/canary_pause.flag новые входы запрещены,
    # стоящая заявка на покупку снимается; уже открытая позиция (если есть) доводится до выхода с защитными рубежами.
    is_paused = CANARY_PAUSED or CANARY_PAUSE_FLAG_PATH.exists()
    if is_paused and float(st.get("pos_qty", 0.0)) < min_amt:
        if st.get("bid_oid") is not None:
            rc_bp = _cancel_canary_oid(trader, st["bid_oid"])
            if rc_bp is not None:
                trader.ex_usdt_free = float(getattr(trader, "ex_usdt_free", 0.0)) + max(0.0, float(st.get("bid_qty", 0.0)) - float(st.get("bid_acc_fill", 0.0))) * float(st.get("bid_px", 0.0))
                st["bid_oid"] = None; st["bid_px"] = 0.0; st["bid_qty"] = 0.0
                changed = True
        if not st.get("pause_logged"):
            st["pause_logged"] = True
            _append_telemetry({"event": "canary_paused", "cum_pnl_usd": st.get("cum_pnl_usd"), "completed_rounds": st.get("completed_rounds")})
            print(f"  🐤⏸️ [CANARY PAUSED] Канарейка на паузе: новых входов нет (Cum=${float(st.get('cum_pnl_usd', 0.0)):+.3f}) | t={_msk_iso()}")
            changed = True
        if changed:
            _save_state(st)
        return {"canary_pos_qty": 0.0, "canary_cash_reserve": 0.0, "canary_bid_oid": st.get("bid_oid"), "canary_ask_oid": st.get("ask_oid")}
    if not is_paused and st.get("pause_logged"):
        st["pause_logged"] = False
        changed = True
    # собственный экземпляр Order3LiveInferenceEngine(canary_neural_v1.pt) сам оценивает 96D рынок +
    # свой микро-портфель ($22 / 8 PRL, hold_sec, entry_px) и через свой персональный Оркестратор
    # переключает Mode #0 (нелинейный Скальпер) <-> Mode #3 (Ожидание / Выход 0% покупок).
    can_eng = _get_canary_neural_engine()
    if can_eng is None:
        if changed:
            _save_state(st)
        return {
            "canary_pos_qty": float(st.get("pos_qty", 0.0)),
            "canary_cash_reserve": CANARY_CASH_RESERVE_USD if bool(st.get("enabled", True)) else 0.0,
            "canary_bid_oid": st.get("bid_oid"),
            "canary_ask_oid": st.get("ask_oid"),
        }

    c_pos = max(0.0, float(st.get("pos_qty", 0.0)))
    c_epx = max(t_sz, float(st.get("entry_px", ext_b1))) if c_pos >= min_amt else 0.0
    c_hold_s = max(0.0, now_ts - float(st.get("entry_ts", now_ts))) if c_pos >= min_amt else 0.0
    c_cash = max(0.0, CANARY_CASH_RESERVE_USD - c_pos * (c_epx if c_epx > 0 else ext_b1))

    # Дросселирование инференса канарейки раз в 2с (<2 мс CPU/с), кроме мгновенного срабатывания страховочных рубежей:
    urgent_rail = (c_pos >= min_amt) and (
        (c_epx > 0 and ext_b1 <= c_epx * (1.0 + CANARY_HARD_STOP_PCT))
        or (c_hold_s >= CANARY_MAX_HOLD_S)
    )
    if (not urgent_rail) and ((now_ts - float(st.get("last_nn_step_ts", 0.0))) < CANARY_STEP_INTERVAL_S):
        if changed:
            _save_state(st)
        return {
            "canary_pos_qty": c_pos,
            "canary_cash_reserve": CANARY_CASH_RESERVE_USD if bool(st.get("enabled", True)) else 0.0,
            "canary_bid_oid": st.get("bid_oid"),
            "canary_ask_oid": st.get("ask_oid"),
        }
    st["last_nn_step_ts"] = now_ts

    b_px_12 = np.zeros(12, dtype=np.float32)
    b_qty_12 = np.zeros(12, dtype=np.float32)
    b_act_12 = np.zeros(12, dtype=np.float64)
    a_px_12 = np.zeros(12, dtype=np.float32)
    a_qty_12 = np.zeros(12, dtype=np.float32)
    a_act_12 = np.zeros(12, dtype=np.float64)

    if st.get("bid_oid") is not None:
        b_px_12[0] = float(st.get("bid_px", 0.0))
        b_qty_12[0] = max(0.0, float(st.get("bid_qty", 0.0)) - float(st.get("bid_acc_fill", 0.0)))
        b_act_12[0] = float(st.get("bid_place_ts", now_ts))
    if st.get("ask_oid") is not None:
        a_px_12[0] = float(st.get("ask_px", 0.0))
        a_qty_12[0] = max(0.0, float(st.get("ask_qty", 0.0)) - float(st.get("ask_acc_fill", 0.0)))
        a_act_12[0] = float(st.get("ask_place_ts", now_ts))

    # Build 100% exact 110D agent vector for the isolated $22 canary account (1-to-1 with fused_order3_clean_tick)
    import torch
    from _tools.fused_order3_clean_tick import AG_DIM_ORDER3, _scale_mkt_44d

    z_ag_c = np.zeros(AG_DIM_ORDER3, dtype=np.float32)
    b1_q_ext = max(1.0, float(bq10_live[0]) if len(bq10_live) > 0 else 10.0)
    a1_q_ext = max(1.0, float(aq10_live[0]) if len(aq10_live) > 0 else 10.0)
    l1_b_usd = max(100.0, ext_b1 * b1_q_ext)
    l1_a_usd = max(100.0, ext_a1 * a1_q_ext)
    res_b_u = float(b_qty_12[0] * b_px_12[0] * 1.001)
    unres_u = max(0.0, c_cash - res_b_u)
    rb_1s = max(10.0, float(getattr(trader, "last_buy_vol_10s", 100.0)) / 10.0)
    has_cp = c_pos >= min_amt
    tot_c_eq = max(CANARY_CASH_RESERVE_USD, c_cash + c_pos * ext_b1)
    pos_rat_c = float(np.clip((c_pos * ext_b1) / max(tot_c_eq, 1.0), 0.0, 1.5))
    eff_ps_c = float(np.clip(pos_rat_c / 0.25, 0.15, 1.0))

    z_ag_c[0] = math.log1p(tot_c_eq / l1_b_usd) / 3.0
    z_ag_c[1] = math.log1p(unres_u / l1_b_usd) / 3.0
    z_ag_c[2] = min(4.0, (c_pos * ext_a1) / l1_a_usd) / 2.0
    z_ag_c[3] = min(10.0, c_pos / rb_1s) / 5.0
    z_ag_c[4] = 1.0 if has_cp else 0.0
    z_ag_c[5] = float(np.clip((((ext_b1 - c_epx) / max(c_epx, t_sz)) - 2.0 * CANARY_MAKER_FEE) * 30.0, -2.5, 2.5)) if (has_cp and c_epx > 0) else 0.0
    z_ag_c[6:18] = np.clip((b_qty_12 * ext_b1) / max(tot_c_eq, 25.0), 0.0, 1.0)
    z_ag_c[18:30] = np.clip(a_qty_12 / max(c_pos, 2.0), 0.0, 1.0)
    z_ag_c[30:42] = np.where(b_qty_12 > 1e-6, np.log1p(np.maximum(0.0, (ext_b1 - b_px_12) / t_sz)) / 3.5, 0.0)
    z_ag_c[42:54] = np.where(a_qty_12 > 1e-6, np.log1p(np.maximum(0.0, (a_px_12 - ext_a1) / t_sz)) / 3.5, 0.0)
    z_ag_c[54] = 1.0 if has_cp else 0.0
    z_ag_c[55] = pos_rat_c
    z_ag_c[56] = float(np.clip(unres_u / max(tot_c_eq, 1.0), 0.0, 1.0))
    z_ag_c[57] = float(np.clip(((((ext_b1 - c_epx) / max(t_sz, c_epx)) - 2.0 * CANARY_MAKER_FEE) * 25.0 * eff_ps_c), -2.5, 2.5)) if (has_cp and c_epx > 0) else 0.0
    z_ag_c[58] = (min(2.0, max(0.0, c_hold_s / 300.0)) * eff_ps_c) if has_cp else 0.0
    z_ag_c[59] = 1.0 if b_qty_12[0] > 1e-6 else 0.0
    z_ag_c[60] = 1.0 if a_qty_12[0] > 1e-6 else 0.0
    z_ag_c[61] = ((1.0 if b_qty_12[0] > 1e-6 else 0.0) - (1.0 if a_qty_12[0] > 1e-6 else 0.0)) / 12.0
    z_ag_c[62] = min(2.0, max(0.0, (now_ts - b_act_12[0]) / 60.0)) if (b_qty_12[0] > 1e-6 and b_act_12[0] > 0) else 0.0
    z_ag_c[63] = min(2.0, max(0.0, (now_ts - a_act_12[0]) / 60.0)) if (a_qty_12[0] > 1e-6 and a_act_12[0] > 0) else 0.0

    zm_sc_c = _scale_mkt_44d(torch.from_numpy(z_mkt).float(), torch.float32).numpy()
    for ci, mi in ((64, 12), (65, 13), (66, 10), (67, 3), (68, 44), (69, 48)):
        z_ag_c[ci] = float(zm_sc_c[mi]) if len(zm_sc_c) > mi else 0.0
    z_ag_c[70:86] = can_eng.mem_h[:, :16].squeeze(0).cpu().numpy()
    z_ag_c[86:98] = np.where(b_qty_12 > 1e-6, np.clip((now_ts - b_act_12) / 60.0, 0.0, 2.0), -1.0).astype(np.float32)
    z_ag_c[98:110] = np.where(a_qty_12 > 1e-6, np.clip((now_ts - a_act_12) / 60.0, 0.0, 2.0), -1.0).astype(np.float32)

    can_eng.refill_tokens(now_ts)
    dec = can_eng.forward_tick(
        z_mkt_39d=z_mkt,
        z_ag_110d=z_ag_c,
        b1=ext_b1,
        a1=ext_a1,
        budget=CANARY_CASH_RESERVE_USD,
        pos_qty=c_pos,
        cash=c_cash,
        bid_px_12=b_px_12,
        bid_qty_12=b_qty_12,
        ask_px_12=a_px_12,
        ask_qty_12=a_qty_12,
        tick_sz=t_sz,
        maker_fee=CANARY_MAKER_FEE,
        trader_entry=c_epx,
        bid_act_12=b_act_12,
        ask_act_12=a_act_12,
        now_ts=now_ts,
        bp_10=bp10_live,
        bq_10=bq10_live,
        ap_10=ap10_live,
        aq_10=aq10_live,
        roll_sell_1s=max(1.0, float(roll_sell_1s)),
        roll_buy_1s=rb_1s,
        hold_sec=c_hold_s,
        min_buy_px=c_epx,
        low_tranche_qty=c_pos,
    )
    gm_c = dec["g_mode"]
    p_m0 = float(gm_c[0])
    p_m3 = float(gm_c[3])

    # 5A. ВЫХОД ПО НЕЙРОСЕТИ при наличии позиции канарейки (>= min_amt)
    if has_cp:
        if st.get("bid_oid") is not None:
            rc_b = _cancel_canary_oid(trader, st["bid_oid"])
            if rc_b is not None:
                st["bid_oid"] = None; st["bid_px"] = 0.0; st["bid_qty"] = 0.0
                changed = True

        u_tk_c = float(dec["u_taker"])
        targ_ask = round(max(ext_b1 + t_sz, float(dec["t_ask_px"][0])), p_prec)
        want_cnl_a = bool(dec["g_cnl_a"][0]) or (u_tk_c > 0.25)

        # Физические защитные рубежи (страховка, НЕ торговая стратегия): аварийный стоп -2% и предельное время удержания.
        force_reason = ""
        if c_epx > 0 and ext_b1 <= c_epx * (1.0 + CANARY_HARD_STOP_PCT):
            force_reason = "hard_stop_-2pct"
        elif c_hold_s >= CANARY_MAX_HOLD_S:
            force_reason = f"max_hold_{int(CANARY_MAX_HOLD_S)}s"
        force_px = 0.0
        if force_reason:
            force_px = round(max(t_sz, ext_b1 * (0.994 if force_reason.startswith("hard_stop") else 1.0)), p_prec)
            cur_apx_f = float(st.get("ask_px", 0.0)) if st.get("ask_oid") is not None else 0.0
            already_forced = st.get("ask_oid") is not None and cur_apx_f <= force_px + 0.1 * t_sz
            u_tk_c = 1.0
            want_cnl_a = not already_forced
            if st.get("force_reason") != force_reason:
                st["force_reason"] = force_reason
                _append_telemetry({
                    "event": "canary_forced_exit",
                    "reason": force_reason,
                    "round_id": st.get("round_id"),
                    "entry_px": round(c_epx, 4),
                    "ext_b1": round(ext_b1, 4),
                    "hold_sec": round(c_hold_s, 1),
                })
                print(f"  🐤⚡ [CANARY SAFETY RAIL] {force_reason}: принудительный выход {c_pos:.4f} PRL @ ~${force_px:.4f} (entry=${c_epx:.4f}, hold={c_hold_s:.0f}s) | t={_msk_iso()}")
                changed = True

        if st.get("ask_oid") is not None and want_cnl_a and trader.order3_engine.api_tokens >= 2.0 and can_eng.api_tokens >= 1.0:
            cur_apx = float(st.get("ask_px", 0.0))
            if (u_tk_c > 0.25) or (abs(targ_ask - cur_apx) >= 0.9 * t_sz):
                rc_a = _cancel_canary_oid(trader, st["ask_oid"])
                trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                can_eng.api_tokens = max(0.0, can_eng.api_tokens - 1.0)
                if rc_a is not None and float(rc_a.get("filled_amount", 0.0)) <= float(st.get("ask_acc_fill", 0.0)) + 1e-4:
                    trader.ex_coin_free = float(getattr(trader, "ex_coin_free", 0.0)) + float(st.get("ask_qty", 0.0))
                    st["ask_oid"] = None; st["ask_px"] = 0.0; st["ask_qty"] = 0.0
                    changed = True

        if st.get("ask_oid") is None and trader.order3_engine.api_tokens >= 1.5 and can_eng.api_tokens >= 1.0:
            q_sell = math.floor(c_pos * 10000.0) / 10000.0
            if force_reason:
                px_out = force_px
            else:
                px_out = round(max(t_sz, ext_b1 * (1.0 - float(dec["max_slip_pct"]))), p_prec) if u_tk_c > 0.25 else targ_ask
            if q_sell >= min_amt and (bool(dec["g_plc_a"][0]) or u_tk_c > 0.25):
                try:
                    res_a = trader.client.place_limit_order(trader.market, "sell", q_sell, px_out, price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt)
                    trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                    can_eng.api_tokens = max(0.0, can_eng.api_tokens - 1.0)
                    if res_a and res_a.get("id") is not None:
                        st["ask_oid"] = res_a.get("id")
                        st["ask_px"] = float(res_a.get("price", px_out))
                        st["ask_qty"] = float(res_a.get("origin_amount", q_sell))
                        st["ask_place_ts"] = now_ts
                        st["ask_acc_fill"] = 0.0
                        trader.ex_coin_free = max(0.0, float(getattr(trader, "ex_coin_free", 0.0)) - st["ask_qty"])
                        print(f"  🐤📤 [CANARY NEURAL ASK] id={st['ask_oid']} | {st['ask_qty']:.4f} PRL @ ${st['ask_px']:.4f} (M0={p_m0:.0%} M3={p_m3:.0%}, hold={c_hold_s:.0f}s) | t={_msk_iso()}")
                        changed = True
                except Exception:
                    pass

    # 5B. ВХОД ПО НЕЙРОСЕТИ И ЕЁ ОРКЕСТРАТОРУ при пустой позиции канарейки
    else:
        if not bool(st.get("enabled", True)):
            if st.get("bid_oid") is not None:
                _cancel_canary_oid(trader, st["bid_oid"])
                st["bid_oid"] = None; st["bid_px"] = 0.0; st["bid_qty"] = 0.0
                changed = True
        else:
            targ_bid = round(min(max(t_sz, ext_a1 - t_sz), float(dec["t_bid_px"][0])), p_prec)
            raw_qb = float(dec["q_target_b"][0])
            want_qb = min(CANARY_LOT_PRL, math.floor(raw_qb * 10000.0) / 10000.0) if raw_qb >= min_amt else 0.0
            want_cnl_b = bool(dec["g_cnl_b"][0]) or (want_qb < min_amt)

            if st.get("bid_oid") is not None and want_cnl_b and trader.order3_engine.api_tokens >= 2.0 and can_eng.api_tokens >= 1.0:
                cur_bpx = float(st.get("bid_px", 0.0))
                if (want_qb < min_amt) or (abs(targ_bid - cur_bpx) >= 0.9 * t_sz):
                    rc_b = _cancel_canary_oid(trader, st["bid_oid"])
                    trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                    can_eng.api_tokens = max(0.0, can_eng.api_tokens - 1.0)
                    if rc_b is not None and float(rc_b.get("filled_amount", 0.0)) <= float(st.get("bid_acc_fill", 0.0)) + 1e-4:
                        trader.ex_usdt_free = float(getattr(trader, "ex_usdt_free", 0.0)) + float(st.get("bid_qty", 0.0)) * cur_bpx
                        st["bid_oid"] = None; st["bid_px"] = 0.0; st["bid_qty"] = 0.0
                        changed = True

            if (
                st.get("bid_oid") is None
                and st.get("ask_oid") is None
                and want_qb >= min_amt
                and bool(dec["g_plc_b"][0])
                and trader.order3_engine.api_tokens >= 2.5
                and can_eng.api_tokens >= 1.0
                and float(getattr(trader, "ex_usdt_free", 0.0)) >= want_qb * targ_bid * 1.02
            ):
                try:
                    res_b = trader.client.place_limit_order(trader.market, "buy", want_qb, targ_bid, price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt)
                    trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                    can_eng.api_tokens = max(0.0, can_eng.api_tokens - 1.0)
                    if res_b and res_b.get("id") is not None:
                        st["bid_oid"] = res_b.get("id")
                        st["bid_px"] = float(res_b.get("price", targ_bid))
                        st["bid_qty"] = float(res_b.get("origin_amount", want_qb))
                        st["bid_place_ts"] = now_ts
                        st["bid_acc_fill"] = 0.0
                        st["bid_init_sp_bp"] = round(ext_sp_bp, 2)
                        st["bid_penny_jumped"] = False
                        st["bid_penny_jump_dt"] = -1.0
                        st["bid_tape_sell_vol"] = 0.0
                        trader.ex_usdt_free = max(0.0, float(getattr(trader, "ex_usdt_free", 0.0)) - st["bid_qty"] * st["bid_px"] * 1.001)
                        _append_telemetry({
                            "event": "canary_bid_place",
                            "engine": "canary_neural_v1",
                            "oid": st["bid_oid"],
                            "px": st["bid_px"],
                            "qty": st["bid_qty"],
                            "ext_b1": round(ext_b1, 4),
                            "ext_a1": round(ext_a1, 4),
                            "ext_sp_bp": round(ext_sp_bp, 1),
                            "p_m0": round(p_m0, 3),
                            "p_m3": round(p_m3, 3),
                        })
                        print(f"  🐤📥 [CANARY NEURAL BID] id={st['bid_oid']} | {st['bid_qty']:.2f} PRL @ ${st['bid_px']:.4f} (ext_sp={ext_sp_bp:.1f}bp | M0={p_m0:.0%} M3={p_m3:.0%}) | t={_msk_iso()}")
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
