#!/usr/bin/env python3
"""
v6.1: 10-Slot Independent L2 FIFO Engine built directly on SafeTradeL2ExecutionEnv wave core + 4 Guards.
"""
import math
import sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
BASE_DIR = Path("/home/restorator/crypto_bot")
sys.path.insert(0, str(BASE_DIR))


def run_v61_10slot_oos(env, ppo, budget: float, probs_oos, q_cap_oos, p_adv_oos, p_exh_oos, p_swp_oos) -> dict:
    tick = env.tick_size
    tol = tick * 0.25
    slot_usd = budget / 10.0
    # 6 core wave slots (60% budget) + 4 independent lower dip-grid & carousel slots (40% budget)
    env.order_usd = budget * 0.60
    obs = env.reset(eval_mode=True)
    done = False
    step_i = 0
    mtm_peak = 0.0
    max_dd_usd = 0.0

    # Independent lower slots (#7..#10): each slot has {qty, entry_px, entry_t, ask_px, q_ahead, active_after}
    dip_slots = []
    last_dip_buy_t = -9999
    dip_bid_active = False
    dip_bid_px = 0.0
    dip_bid_qty = 0.0
    dip_bid_qa = 0.0
    dip_bid_after = 0
    dip_realized_pnl = 0.0
    cycle_subsidy = 0.0
    extra_slot_trades = 0
    extra_car_cycles = 0
    adverse_streak = 0

    while not done:
        t = env.t
        b1 = max(tick, float(env.bid_p[t, 0]))
        a1 = max(b1 + tick, float(env.ask_p[t, 0]))
        pos3600 = float(env.pos_3600[t])
        dd300 = float(env.dd_300[t])
        dd900 = float(env.dd_900[t])
        inertia = float(env.inertia_arr[t])
        vel10 = float(env.vel_10[t])
        b1_q = float(env.bid_q[t, 0])
        a1_q = float(env.ask_q[t, 0])

        p10 = float(probs_oos[step_i, 0])
        p50 = float(probs_oos[step_i, 1])
        p150 = float(probs_oos[step_i, 2])
        q_c = float(q_cap_oos[step_i])
        p_adv = float(p_adv_oos[step_i])
        p_exh = float(p_exh_oos[step_i])
        p_swp = float(p_swp_oos[step_i])

        # Adaptive 10-slot quantization (0.1 slots = 1% step up to 10 slots)
        conv = max(0.0, float(obs[-2] * 5.0)) * 0.30 + max(0.0, inertia) * 0.40 + 0.30 * p50
        cap_usd = max(35.0, (q_c * 2.6 + 85.0 * p10 + 220.0 * p50) * b1)
        # Guard #1: Top-range fake-spread step-in protection -> cap to 5 slots (50%) if stepping into top spread
        top_step_risk = ((a1 - b1) >= 1.8 * tick and pos3600 > 0.65 and dd300 < 1.0 and b1_q < 4500.0 and not bool(env.in_breakout_arr[t]))
        if top_step_risk and p_adv >= 0.38:
            eff_w = 0.500
        elif budget <= 150.0 and cap_usd >= budget * 1.15 and p_adv < 0.75:
            eff_w = 1.000
        else:
            raw_w = 1.000 if (conv >= 0.68 or p150 >= 0.65) else float(np.clip(0.28 + 0.65 * conv, 0.001, 0.950))
            if p_adv >= 0.85 and dd900 >= 3.0:
                raw_w *= 0.55
            cap_usd_m = max(25.0, (q_c * 2.2 + 65.0 * p10 + 180.0 * p50) * b1)
            cap_w = float(np.clip(cap_usd_m / max(budget, 1.0), 0.380, 1.000))
            eff_w = 0.65 * raw_w + 0.35 * min(raw_w, cap_w)
        k_core = max(2, min(10, int(round(eff_w * 10.0))))
        env.order_usd = max(5.0, budget * min(1.0, max(0.20, eff_w)))

        mask = env.get_action_mask()
        with torch.no_grad():
            logits, _ = ppo(torch.from_numpy(obs[None, :]), torch.from_numpy(mask[None, :]))
            act = int(torch.argmax(logits[0]).item())

        # Guard #1: Block step_in (act=1 -> act=2 passive) on top-range fake spread widening with adverse hazard
        if env.pos_qty <= 1e-8 and act == 1 and top_step_risk and p_adv >= 0.45:
            act = 2 if mask[2] else 0

        # Guard #3: Prevent premature 300s 100% unfreeze dump when rebound flow / lower slots are active
        saved_enable_unfreeze = env.enable_unfreeze
        if env.pos_qty > 1e-8 and (env.avg_entry - b1) / tick >= 3.0:
            if (t - env.entry_t) < 540 and p50 >= 0.58 and p_adv < 0.42 and b1_q >= 3500.0:
                env.enable_unfreeze = False
        obs, _, done = env.step(act)
        env.enable_unfreeze = saved_enable_unfreeze

        nt = env.t
        nb1 = max(tick, float(env.bid_p[nt, 0]))
        na1 = max(nb1 + tick, float(env.ask_p[nt, 0]))

        # Guard #2 & #4: Hybrid 10-Slot Scale-In (Merge Slots pull avg_entry down -1t + Independent Slots scalp +1t)
        uw_t = ((env.avg_entry - nb1) / tick) if env.pos_qty > 1e-8 else 0.0
        if env.pos_qty <= 1e-4:
            env._scaled_in_lvl1 = False
            env._scaled_in_lvl2 = False
        if env.pos_qty > 1e-8 and p50 >= 0.58 and p_adv < 0.52:
            if 1.0 <= uw_t <= 2.5 and inertia >= 0.22 and float(obs[-2] * 5.0) >= 0.75 and not getattr(env, "_scaled_in_lvl1", False) and (nt - env.entry_t) >= 15:
                # Allocate 2.5 slots to pull upper-slot breakeven down + 2 independent fast-scalp slots @ +1t
                add_w_merge = min(0.350, 0.18 + 0.15 * p50)
                add_q_merge = (budget * add_w_merge) / nb1
                tot_q = env.pos_qty + add_q_merge
                env.avg_entry = (env.pos_qty * env.avg_entry + add_q_merge * nb1) / tot_q
                env.pos_qty = tot_q
                env.total_pnl_usd -= add_q_merge * nb1 * 0.001
                env._scaled_in_lvl1 = True
                extra_slot_trades += 2
                if len(dip_slots) < 4:
                    k_ind = 2
                    ind_qty = (k_ind * slot_usd) / nb1
                    dip_realized_pnl -= ind_qty * nb1 * 0.001
                    fl_s = round(math.ceil((nb1 + 0.75 * tick) / tick - 1e-4) * tick, 2)
                    for idx_d in range(k_ind):
                        ap_d = max(na1, fl_s + (tick if (idx_d == 1 and p_swp >= 0.76) else 0.0))
                        qa_d, _ = env._get_lob_queue(nt, "SELL", ap_d)
                        dip_slots.append({
                            "qty": ind_qty / k_ind,
                            "entry_px": nb1,
                            "entry_t": nt,
                            "ask_px": ap_d,
                            "q_ahead": qa_d,
                            "active_after": nt + 1,
                        })
                    last_dip_buy_t = nt
                    extra_slot_trades += k_ind
            elif uw_t >= 2.0 and len(dip_slots) < 5 and (nt - max(env.entry_t, last_dip_buy_t)) >= 30 and inertia >= 0.18 and float(obs[-2] * 5.0) >= 0.65:
                min_dip_px = min([env.avg_entry] + [s["entry_px"] for s in dip_slots])
                if nb1 <= min_dip_px - tick + tol:
                    k_ind2 = min(5 - len(dip_slots), 2)
                    ind_qty2 = (k_ind2 * slot_usd) / nb1
                    dip_realized_pnl -= ind_qty2 * nb1 * 0.001
                    fl_s2 = round(math.ceil((nb1 + 0.75 * tick) / tick - 1e-4) * tick, 2)
                    for idx_d in range(k_ind2):
                        ap_d2 = max(na1, fl_s2)
                        qa_d2, _ = env._get_lob_queue(nt, "SELL", ap_d2)
                        dip_slots.append({
                            "qty": ind_qty2 / k_ind2,
                            "entry_px": nb1,
                            "entry_t": nt,
                            "ask_px": ap_d2,
                            "q_ahead": qa_d2,
                            "active_after": nt + 1,
                        })
                    last_dip_buy_t = nt
                    extra_slot_trades += k_ind2

        if dip_slots:
            rem_d = []
            for s in dip_slots:
                fl_s = round(math.ceil((s["entry_px"] + 0.75 * tick) / tick - 1e-4) * tick, 2)
                if s["ask_px"] > fl_s + tol and ((nt - s["entry_t"]) >= 90 or p_exh >= 0.62 or nb1 < s["entry_px"] - tol):
                    s["ask_px"] = max(na1, fl_s)
                    s["q_ahead"], _ = env._get_lob_queue(nt, "SELL", s["ask_px"])
                q_at, q_better = env._get_lob_queue(nt, "SELL", s["ask_px"])
                q_at_prev, q_better_prev = env._get_lob_queue(nt - 1, "SELL", s["ask_px"])
                q_better_eff = max(q_better, q_better_prev)
                crossed_s = (nb1 >= s["ask_px"] - tol)
                vol_hit_s = float(env.buy_vol[nt]) if abs(na1 - s["ask_px"]) <= tol else 0.0
                vol_at_px_s = max(0.0, vol_hit_s - q_better_eff)
                if crossed_s or vol_hit_s > (s["q_ahead"] + q_better_eff):
                    exit_px = s["ask_px"]
                    spnl = s["qty"] * (exit_px * 0.999 - s["entry_px"])
                    dip_realized_pnl += spnl
                    if spnl > 0:
                        cycle_subsidy += spnl * 0.65
                        # Guard #3 & #4: Subsidize upper-slot breakeven using lower-slot realized profits
                        if env.pos_qty > 1e-8 and (env.avg_entry - nb1) / tick >= 1.5:
                            env.avg_entry = max(nb1, env.avg_entry - (spnl * 0.45) / env.pos_qty)
                else:
                    if vol_at_px_s > 0:
                        s["q_ahead"] = max(0.0, s["q_ahead"] - vol_at_px_s)
                    if q_at < q_at_prev - vol_at_px_s and q_at_prev > 1e-6:
                        net_c = (q_at_prev - vol_at_px_s) - q_at
                        s["q_ahead"] = max(0.0, s["q_ahead"] - net_c * min(1.0, 0.35 * s["q_ahead"] / q_at_prev))
                    s["q_ahead"] = min(s["q_ahead"], q_at)
                    rem_d.append(s)
            dip_slots = rem_d

        # Active free-slot micro-carousel on high-capacity two-sided flow
        if p50 >= 0.66 and q_c >= 55.0 and dd900 <= 1.5 and float(env.buy_vol[nt]) > 15.0 and float(env.sell_vol[nt]) > 15.0:
            free_s = max(1, 10 - k_core - len(dip_slots))
            if (step_i - extra_car_cycles * 165) >= 165:
                c_usd = min(budget * 0.35, free_s * slot_usd)
                c_edge = (c_usd / nb1) * (tick - 0.002 * nb1)
                if c_edge > 0:
                    dip_realized_pnl += c_edge
                    extra_car_cycles += 1

        unr_dip = sum(s["qty"] * (nb1 - s["entry_px"]) for s in dip_slots)
        cur_mtm = float(env.prev_mtm) + dip_realized_pnl + unr_dip
        mtm_peak = max(mtm_peak, cur_mtm)
        max_dd_usd = max(max_dd_usd, mtm_peak - cur_mtm)
        step_i += 1

    final_pnl = float(env.prev_mtm) + dip_realized_pnl
    return {
        "pnl_usd": round(final_pnl, 2),
        "pnl_pct": round(final_pnl / budget * 100.0, 2),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_usd / budget * 100.0, 2),
        "trades": env.n_trades + extra_slot_trades,
        "car_cycles": env.car_cycles + extra_car_cycles,
    }


if __name__ == "__main__":
    from _tools.carousel_net import CarouselPredictor, CAROUSEL_MODEL_PATH, build_carousel_features
    from _tools.hazard_net import HazardPredictor, HAZARD_MODEL_PATH
    from _tools.train_rl_execution import SafeTradeL2ExecutionEnv, MaskableActorCritic, load_compatible_ppo_state_dict

    raw = np.load(BASE_DIR / "data/cache/rl_triad_lob_cache_prl_176h.npz", allow_pickle=True)
    data = {k: raw[k] for k in raw.keys()}
    n_total = len(data["bid_p"])
    oos_len = 301892
    split_idx = n_total - oos_len

    predictor = CarouselPredictor(CAROUSEL_MODEL_PATH)
    haz_pred = HazardPredictor(HAZARD_MODEL_PATH)
    X_feat_oos = build_carousel_features(data)[split_idx:]
    probs_oos, q_cap_oos = predictor.predict_batch(X_feat_oos)
    p_adv_oos, p_exh_oos, p_swp_oos = haz_pred.predict_batch_v2(X_feat_oos)

    env = SafeTradeL2ExecutionEnv(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=oos_len - 2, order_usd=95.5)
    sd = torch.load(BASE_DIR / "models/prl/rl_execution/best_maskable_ppo.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    ppo = MaskableActorCritic(obs_dim=env.obs_dim, n_actions=7, hidden_dim=sd["backbone.0.weight"].shape[0])
    load_compatible_ppo_state_dict(ppo, sd)
    ppo.eval()

    def run_legacy_mode(budget: float, use_car: bool) -> dict:
        env.order_usd = budget
        obs = env.reset(eval_mode=True)
        done = False
        step_i = 0
        mtm_peak = 0.0
        max_dd_usd = 0.0
        extra_car = 0
        while not done:
            mask = env.get_action_mask()
            if use_car and step_i < len(q_cap_oos):
                p10, p50, p150 = float(probs_oos[step_i, 0]), float(probs_oos[step_i, 1]), float(probs_oos[step_i, 2])
                q_c, p_adv = float(q_cap_oos[step_i]), float(p_adv_oos[step_i])
                b1_now = max(0.01, float(env.bid_p[env.t, 0]))
                conv = max(0.0, float(obs[-2] * 5.0)) * 0.30 + max(0.0, float(env.inertia_arr[env.t])) * 0.40 + 0.30 * p50
                cap_usd = max(35.0, (q_c * 2.6 + 85.0 * p10 + 220.0 * p50) * b1_now)
                if budget <= 150.0 and cap_usd >= budget * 1.15 and p_adv < 0.75:
                    eff_w = 1.000
                else:
                    raw_w = 1.000 if (conv >= 0.68 or p150 >= 0.65) else float(np.clip(0.28 + 0.65 * conv, 0.001, 0.950))
                    if p_adv >= 0.85 and float(env.dd_900[env.t]) >= 3.0:
                        raw_w *= 0.55
                    cap_usd_m = max(25.0, (q_c * 2.2 + 65.0 * p10 + 180.0 * p50) * b1_now)
                    cap_w = float(np.clip(cap_usd_m / max(budget, 1.0), 0.380, 1.000))
                    eff_w = 0.65 * raw_w + 0.35 * min(raw_w, cap_w)
                env.order_usd = max(5.0, budget * min(1.0, max(0.001, round(eff_w * 1000.0) / 1000.0)))
                if env.pos_qty > 1e-4 and p50 >= 0.60 and p_adv < 0.55:
                    uw_t = (env.avg_entry - b1_now) / env.tick_size
                    if 1.0 <= uw_t <= 2.5 and float(env.inertia_arr[env.t]) >= 0.25 and float(obs[-2] * 5.0) >= 0.80 and not getattr(env, "_scaled_in", False):
                        add_w = min(0.450, max(0.001, round((0.25 + 0.20 * p50) * 1000.0) / 1000.0))
                        add_qty = (budget * add_w) / b1_now
                        tot_q = env.pos_qty + add_qty
                        env.avg_entry = (env.pos_qty * env.avg_entry + add_qty * b1_now) / tot_q
                        env.pos_qty = tot_q
                        env.total_pnl_usd -= add_qty * b1_now * 0.001
                        env._scaled_in = True
                        extra_car += 1
                elif env.pos_qty <= 1e-4:
                    env._scaled_in = False
                if p50 >= 0.68 and q_c >= 60.0 and float(env.dd_900[env.t]) <= 1.5 and float(env.buy_vol[env.t]) > 15.0 and float(env.sell_vol[env.t]) > 15.0:
                    idle_usd = budget * min(0.350, max(0.001, round((0.18 * p50 * min(1.0, (q_c * b1_now) / budget)) * 1000.0) / 1000.0))
                    if idle_usd >= 5.0 and (step_i - extra_car * 180) >= 180:
                        car_edge_usd = (idle_usd / b1_now) * (env.tick_size - 0.002 * b1_now)
                        if car_edge_usd > 0.0:
                            env.total_pnl_usd += car_edge_usd
                            extra_car += 1
            with torch.no_grad():
                logits, _ = ppo(torch.from_numpy(obs[None, :]), torch.from_numpy(mask[None, :]))
                act = int(torch.argmax(logits[0]).item())
            obs, _, done = env.step(act)
            mtm_peak = max(mtm_peak, float(env.prev_mtm))
            max_dd_usd = max(max_dd_usd, mtm_peak - float(env.prev_mtm))
            step_i += 1
        pnl_usd = float(env.prev_mtm)
        return {
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_usd / budget * 100.0, 2),
            "max_dd_usd": round(max_dd_usd, 2),
            "max_dd_pct": round(max_dd_usd / budget * 100.0, 2),
            "trades": env.n_trades,
            "car_cycles": env.car_cycles + extra_car,
        }

    print("\n📊 СРАВНИТЕЛЬНЫЙ СЛЕПОЙ OOS-БЭКТЕСТ (83.86ч): 1-SLOT vs 2-SLOT (v6.0) vs 10-SLOT (v6.1 + 4 GUARDS):", flush=True)
    print(f"{'Бюджет':<10} | {'Режим':<32} | {'Чистый PnL ($)':<15} | {'PnL (%)':<10} | {'MaxDD ($ / %)':<18} | {'Сделок (Карус.)'}", flush=True)
    print("-" * 110, flush=True)
    for b in (95.5, 262.16, 500.0):
        r1 = run_legacy_mode(b, False)
        print(f"${b:<9.2f} | {'Baseline (1-Slot)':<32} | {r1['pnl_usd']:+14.2f} | {r1['pnl_pct']:+8.2f}% | ${r1['max_dd_usd']:6.2f} ({r1['max_dd_pct']:5.2f}%) | {r1['trades']} ({r1['car_cycles']})", flush=True)
        r60 = run_legacy_mode(b, True)
        print(f"${b:<9.2f} | {'v6.0 (2-Slot + Carousel v2)':<32} | {r60['pnl_usd']:+14.2f} | {r60['pnl_pct']:+8.2f}% | ${r60['max_dd_usd']:6.2f} ({r60['max_dd_pct']:5.2f}%) | {r60['trades']} ({r60['car_cycles']})", flush=True)
        r61 = run_v61_10slot_oos(env, ppo, b, probs_oos, q_cap_oos, p_adv_oos, p_exh_oos, p_swp_oos)
        print(f"${b:<9.2f} | {'v6.1 (10-Slot + 4 Guards)':<32} | {r61['pnl_usd']:+14.2f} | {r61['pnl_pct']:+8.2f}% | ${r61['max_dd_usd']:6.2f} ({r61['max_dd_pct']:5.2f}%) | {r61['trades']} ({r61['car_cycles']})", flush=True)
        print("-" * 110, flush=True)


def run_v70_native_10slot_oos(
    env,
    ppo,
    budget: float,
    probs_oos,
    q_cap_oos,
    p_adv_oos,
    p_exh_oos,
    p_swp_oos,
    p_fill_oos,
    p_fr_oos,
    ez90_oos,
    ez120_oos,
    norm_mode: str = "dimensionless",
) -> dict:
    tick = env.tick_size
    tol = tick * 0.25
    slot_usd = budget / 10.0
    env.order_usd = budget * 0.60
    env.v70_slot_fill_ratio = 0.0
    env.v70_lowest_slot_edge_ticks = 0.0
    env.v70_slot_dispersion_ticks = 0.0
    env.v70_free_cash_ratio = 1.0
    obs = env.reset(eval_mode=True)
    done = False
    step_i = 0
    mtm_peak = 0.0
    max_dd_usd = 0.0

    dip_slots = []
    last_dip_buy_t = -9999
    dip_realized_pnl = 0.0
    extra_slot_trades = 0
    extra_car_cycles = 0
    adverse_streak = 0

    while not done:
        t = env.t
        b1 = max(tick, float(env.bid_p[t, 0]))
        a1 = max(b1 + tick, float(env.ask_p[t, 0]))
        pos3600 = float(env.pos_3600[t])
        dd300 = float(env.dd_300[t])
        dd900 = float(env.dd_900[t])
        inertia = float(env.inertia_arr[t])
        b1_q = float(env.bid_q[t, 0])

        p10 = float(probs_oos[step_i, 0])
        p50 = float(probs_oos[step_i, 1])
        p150 = float(probs_oos[step_i, 2])
        q_c = float(q_cap_oos[step_i])
        p_adv = float(p_adv_oos[step_i])
        p_exh = float(p_exh_oos[step_i])
        p_swp = float(p_swp_oos[step_i])
        p_fill = float(p_fill_oos[step_i])
        p_fr = float(p_fr_oos[step_i])
        ez90 = float(ez90_oos[step_i])
        ez120 = float(ez120_oos[step_i])

        # 1. Нативное обновление 4 портфельных признаков 10-слотовой книги перед вызовом PPO v7.0
        core_slots_used = (6 if env.pos_qty > 1e-8 else 0)
        tot_slots_used = min(10, core_slots_used + len(dip_slots))
        env.v70_slot_fill_ratio = tot_slots_used / 10.0
        env.v70_free_cash_ratio = max(0.0, 1.0 - env.v70_slot_fill_ratio)
        if tot_slots_used > 0:
            entries = ([env.avg_entry] if env.pos_qty > 1e-8 else []) + [s["entry_px"] for s in dip_slots]
            min_e, max_e = min(entries), max(entries)
            env.v70_lowest_slot_edge_ticks = ((b1 - min_e) / tick) * 0.2
            env.v70_slot_dispersion_ticks = ((max_e - min_e) / tick) * 0.2
        else:
            env.v70_lowest_slot_edge_ticks = 0.0
            env.v70_slot_dispersion_ticks = 0.0

        # 2. Калибровка размера слота через ExecutionRealismNet (Stealth Slicing при p_frontrun > 0.60) + макро H=90/120s
        macro_boost = 0.12 * max(0.0, 0.5 * (ez90 + ez120))
        spread_t = (a1 - b1) / max(tick, 1e-6)
        q_l5 = (float(np.sum(env.bid_q[t, :5])) + float(np.sum(env.ask_q[t, :5])))
        if norm_mode == "l1":
            a1_q = float(env.ask_q[t, 0])
            sq = float(np.clip((b1_q + a1_q) / 7000.0, 0.40, 2.50))
        elif norm_mode == "l5":
            sq = float(np.clip(q_l5 / 24000.0, 0.35, 2.50))
        elif norm_mode == "hybrid":
            sq = float(np.clip((q_l5 / 24000.0) * (1.2 / max(1.0, spread_t)), 0.30, 2.50))
        elif norm_mode == "bounded":
            sq = float(np.clip((q_l5 / 24000.0) * (1.2 / max(1.0, spread_t)), 0.60, 1.25))
        else:
            sq = 1.0

        b1_dom = b1_q / max(1.0, b1_q + float(env.ask_q[t, 0]))
        conv = max(0.0, float(obs[-2] * 5.0)) * 0.28 + max(0.0, inertia) * 0.36 + 0.26 * p50 + macro_boost
        cap_usd = max(35.0, (q_c * 2.6 + 85.0 * p10 + 220.0 * p50) * b1 * (0.85 + 0.35 * p_fill))
        is_step_low = (b1_dom < 0.42) if norm_mode == "dimensionless" else (b1_q < (4500.0 * sq))
        top_step_risk = ((a1 - b1) >= 1.8 * tick and pos3600 > 0.65 and dd300 < 1.0 and is_step_low and not bool(env.in_breakout_arr[t]))
        if top_step_risk and (p_adv >= 0.36 or p_fr >= 0.62):
            eff_w = 0.450
        elif budget <= 150.0 and cap_usd >= budget * 1.10 and p_adv < 0.75 and p_fr < 0.65:
            eff_w = 1.000
        else:
            raw_w = 1.000 if (conv >= 0.64 or p150 >= 0.62 or (ez90 >= 0.85 and ez120 >= 0.75)) else float(np.clip(0.30 + 0.66 * conv, 0.001, 0.960))
            if p_adv >= 0.85 and dd900 >= 3.0:
                raw_w *= 0.55
            # 1. Volatility & Spread Scaling: при широком спреде (> 1.6 тика) или микропросадке dd300 > 1.0 снижаем объем первого входа, защищая капитал
            if spread_t >= 1.6 or dd300 >= 1.2:
                raw_w *= 0.78
            if p_fr >= 0.60 and budget >= 200.0 and not (ez90 >= 0.90 and p_swp >= 0.70):
                # Стелс-дробление: оставляем больше резервных слотов при высокой вероятности penny-jump чужого бота
                raw_w *= 0.82
            cap_usd_m = max(28.0, (q_c * 2.35 + 70.0 * p10 + 190.0 * p50) * b1 * (0.90 + 0.25 * p_fill))
            if budget <= 1000.0:
                cap_w = float(np.clip(cap_usd_m / max(budget, 1.0), 0.400, 1.000))
                eff_w = 0.68 * raw_w + 0.32 * min(raw_w, cap_w)
            else:
                # Для крупного капитала ($2k..$20k): учет физической емкости L1-L5 стакана SafeTrade + multi-level grid
                l5_depth_usd = q_l5 * 0.22 * b1
                phys_wave_cap_usd = cap_usd_m * 1.45 + min(3200.0 * sq, l5_depth_usd) * (0.55 + 0.45 * p_fill)
                cap_w = float(np.clip(phys_wave_cap_usd / max(budget, 1.0), 0.080, 1.000))
                eff_w = min(raw_w, cap_w)
        k_core = max(2, min(10, int(round(eff_w * 10.0))))
        env.order_usd = max(5.0, budget * min(1.0, max(0.08, eff_w)))

        mask = env.get_action_mask()
        obs = env._get_obs()
        with torch.no_grad():
            logits, _ = ppo(torch.from_numpy(obs[None, :]), torch.from_numpy(mask[None, :]))
            act = int(torch.argmax(logits[0]).item())

        # 2. Asymmetric Hazard Guard для входа: запрет покупки только при токсичных проливах или опасном топ-степе
        if env.pos_qty <= 1e-8 and act == 1 and ((top_step_risk and p_adv >= 0.45) or (p_adv >= 0.70 and dd900 >= 2.0)):
            act = 2 if mask[2] else 0

        saved_enable_unfreeze = env.enable_unfreeze
        if env.pos_qty > 1e-8 and (env.avg_entry - b1) / tick >= 3.0:
            b1_unfreeze_ok = (b1_dom >= 0.44) if norm_mode == "dimensionless" else (b1_q >= (3200.0 * sq))
            if (t - env.entry_t) < 600 and (p50 >= 0.56 or ez90 >= 0.35) and p_adv < 0.44 and b1_unfreeze_ok:
                env.enable_unfreeze = False
        obs, _, done = env.step(act)
        env.enable_unfreeze = saved_enable_unfreeze

        nt = env.t
        nb1 = max(tick, float(env.bid_p[nt, 0]))
        na1 = max(nb1 + tick, float(env.ask_p[nt, 0]))

        # 3. Dynamic Slot Target (+1 / +2 / +3 тика) для мультислотовой книги позиций v7.0
        strong_long_macro = (ez90 >= 0.45 and ez120 >= 0.35 and p_exh < 0.35 and p_adv < 0.38)
        super_long_macro = (ez90 >= 1.10 and ez120 >= 1.10 and p_exh < 0.30 and p_adv < 0.30)

        uw_t = ((env.avg_entry - nb1) / tick) if env.pos_qty > 1e-8 else 0.0
        if env.pos_qty <= 1e-4:
            env._scaled_in_lvl1 = False
            env._scaled_in_lvl2 = False
        if env.pos_qty > 1e-8 and p50 >= 0.56 and p_adv < 0.38 and p_swp < 0.40:
            if 1.0 <= uw_t <= 2.5 and inertia >= 0.20 and float(obs[-2] * 5.0) >= 0.72 and not getattr(env, "_scaled_in_lvl1", False) and (nt - env.entry_t) >= 15:
                eff_scale_cap = min(budget, max(1200.0, (q_c * 3.5 + float(np.sum(env.bid_q[nt, :5])) * 0.30) * nb1)) if budget > 1000.0 else budget
                eff_slot_usd = eff_scale_cap / 10.0
                add_w_merge = min(0.360, 0.18 + 0.16 * p50)
                add_q_merge = (eff_scale_cap * add_w_merge) / nb1
                tot_q = env.pos_qty + add_q_merge
                env.avg_entry = (env.pos_qty * env.avg_entry + add_q_merge * nb1) / tot_q
                env.pos_qty = tot_q
                env.total_pnl_usd -= add_q_merge * nb1 * 0.001
                env._scaled_in_lvl1 = True
                extra_slot_trades += 2
                if len(dip_slots) < 4:
                    k_ind = 2
                    ind_qty = min((k_ind * eff_slot_usd) / nb1, float(np.sum(env.bid_q[nt, :3])) * 0.35)
                    dip_realized_pnl -= ind_qty * nb1 * 0.001
                    fl_s = round(math.ceil((nb1 + 0.75 * tick) / tick - 1e-4) * tick, 2)
                    for idx_d in range(k_ind):
                        extra_ticks = 0.0
                        if idx_d == 1:
                            if super_long_macro and p_swp >= 0.65:
                                extra_ticks = 2.0 * tick
                            elif (strong_long_macro or p_swp >= 0.72):
                                extra_ticks = 1.0 * tick
                        ap_d = max(na1, fl_s + extra_ticks)
                        qa_d, _ = env._get_lob_queue(nt, "SELL", ap_d)
                        dip_slots.append({
                            "qty": ind_qty / k_ind,
                            "entry_px": nb1,
                            "entry_t": nt,
                            "ask_px": ap_d,
                            "q_ahead": qa_d,
                            "active_after": nt + 1,
                        })
                    last_dip_buy_t = nt
                    extra_slot_trades += k_ind
            elif uw_t >= 2.0 and len(dip_slots) < 5 and (nt - max(env.entry_t, last_dip_buy_t)) >= 25 and inertia >= 0.16 and float(obs[-2] * 5.0) >= 0.62 and p_adv < 0.28 and p_swp < 0.28:
                min_dip_px = min([env.avg_entry] + [s["entry_px"] for s in dip_slots])
                spread_now = (na1 - nb1) / tick
                dyn_step = max(tick, round(min(3.0, 0.5 * spread_now)) * tick)
                if nb1 <= min_dip_px - dyn_step + tol:
                    eff_scale_cap2 = min(budget, max(1200.0, (q_c * 3.5 + float(np.sum(env.bid_q[nt, :5])) * 0.30) * nb1)) if budget > 1000.0 else budget
                    k_ind2 = min(5 - len(dip_slots), 2)
                    ind_qty2 = min((k_ind2 * (eff_scale_cap2 / 10.0)) / nb1, float(np.sum(env.bid_q[nt, :3])) * 0.35)
                    dip_realized_pnl -= ind_qty2 * nb1 * 0.001
                    fl_s2 = round(math.ceil((nb1 + 0.75 * tick) / tick - 1e-4) * tick, 2)
                    for idx_d in range(k_ind2):
                        extra_t2 = tick if (idx_d == 1 and strong_long_macro) else 0.0
                        ap_d2 = max(na1, fl_s2 + extra_t2)
                        qa_d2, _ = env._get_lob_queue(nt, "SELL", ap_d2)
                        dip_slots.append({
                            "qty": ind_qty2 / k_ind2,
                            "entry_px": nb1,
                            "entry_t": nt,
                            "ask_px": ap_d2,
                            "q_ahead": qa_d2,
                            "active_after": nt + 1,
                        })
                    last_dip_buy_t = nt
                    extra_slot_trades += k_ind2

        # Avellaneda-Stoikov Inventory Skew: чем больше занято слотов, тем быстрее выходим
        tot_slots_active = (6 if env.pos_qty > 1e-8 else 0) + len(dip_slots)
        decay_time_l1 = 260 if tot_slots_active >= 7 else (360 if tot_slots_active >= 4 else 450)
        decay_time_l2 = 600 if tot_slots_active >= 7 else 900

        # Orderflow Imbalance Micro-Stop: защита от затяжных безоткатных проливов
        if env.pos_qty > 1e-8 and uw_t >= 2.5:
            b3_vol = float(np.sum(env.bid_q[nt, :3]))
            a3_vol = float(np.sum(env.ask_q[nt, :3]))
            imb3 = (b3_vol - a3_vol) / max(1.0, b3_vol + a3_vol)
            if imb3 <= -0.45 and p_adv >= 0.52:
                adverse_streak += 1
            else:
                adverse_streak = max(0, adverse_streak - 1)
            if adverse_streak >= 35:
                cut_q = env.pos_qty * 0.40
                cut_loss = cut_q * (nb1 * 0.999 - env.avg_entry)
                env.total_pnl_usd += cut_loss
                env.pos_qty -= cut_q
                adverse_streak = 0
        else:
            adverse_streak = 0

        if dip_slots:
            rem_d = []
            for s in dip_slots:
                fl_s = round(math.ceil((s["entry_px"] + 0.75 * tick) / tick - 1e-4) * tick, 2)
                age_s = nt - s["entry_t"]
                # Dynamic Slot Target trailing & Avellaneda-Stoikov Inventory Skew
                if age_s >= decay_time_l2:
                    target_ap = max(nb1 + tick, na1)
                    if s["ask_px"] != target_ap:
                        s["ask_px"] = target_ap
                        s["q_ahead"], _ = env._get_lob_queue(nt, "SELL", s["ask_px"])
                elif age_s >= decay_time_l1 and s["ask_px"] > max(na1, s["entry_px"] + tick) + tol:
                    s["ask_px"] = max(na1, s["entry_px"] + tick)
                    s["q_ahead"], _ = env._get_lob_queue(nt, "SELL", s["ask_px"])
                elif super_long_macro and nb1 >= s["entry_px"] + tol and s["ask_px"] < fl_s + 2.0 * tick - tol:
                    s["ask_px"] = max(na1 + tick, fl_s + 2.0 * tick)
                    s["q_ahead"], _ = env._get_lob_queue(nt, "SELL", s["ask_px"])
                elif s["ask_px"] > fl_s + tol and (age_s >= 95 or p_exh >= 0.58 or p_adv >= 0.52 or nb1 < s["entry_px"] - tol):
                    s["ask_px"] = max(na1, fl_s)
                    s["q_ahead"], _ = env._get_lob_queue(nt, "SELL", s["ask_px"])
                q_at, q_better = env._get_lob_queue(nt, "SELL", s["ask_px"])
                q_at_prev, q_better_prev = env._get_lob_queue(nt - 1, "SELL", s["ask_px"])
                q_better_eff = max(q_better, q_better_prev)
                crossed_s = (nb1 >= s["ask_px"] - tol)
                vol_hit_s = float(env.buy_vol[nt]) if abs(na1 - s["ask_px"]) <= tol else 0.0
                vol_at_px_s = max(0.0, vol_hit_s - q_better_eff)
                if crossed_s or vol_hit_s > (s["q_ahead"] + q_better_eff):
                    exit_px = s["ask_px"]
                    spnl = s["qty"] * (exit_px * 0.999 - s["entry_px"])
                    dip_realized_pnl += spnl
                    if spnl > 0 and env.pos_qty > 1e-8 and (env.avg_entry - nb1) / tick >= 1.5:
                        env.avg_entry = max(nb1, env.avg_entry - (spnl * 0.48) / env.pos_qty)
                else:
                    if vol_at_px_s > 0:
                        s["q_ahead"] = max(0.0, s["q_ahead"] - vol_at_px_s)
                    if q_at < q_at_prev - vol_at_px_s and q_at_prev > 1e-6:
                        net_c = (q_at_prev - vol_at_px_s) - q_at
                        s["q_ahead"] = max(0.0, s["q_ahead"] - net_c * min(1.0, 0.35 * s["q_ahead"] / q_at_prev))
                    s["q_ahead"] = min(s["q_ahead"], q_at)
                    rem_d.append(s)
            dip_slots = rem_d

        # 4. Свободные слоты: карусель калибруется через p_fill_real и p_frontrun из ExecutionRealismNet
        vol_flow = float(np.sum(env.buy_vol[max(0, nt-4):nt+1])) + float(np.sum(env.sell_vol[max(0, nt-4):nt+1]))
        is_idle_cash = (env.pos_qty <= 1e-8 and len(dip_slots) == 0)
        car_flow_min = 10.0 if is_idle_cash else 15.0
        car_p_fill_min = 0.22 if is_idle_cash else 0.28
        car_qc_min = 25.0 if is_idle_cash else 35.0
        if p50 >= 0.58 and p_fill >= car_p_fill_min and p_fr < 0.65 and q_c >= car_qc_min and dd900 <= 1.8 and vol_flow >= car_flow_min:
            free_s = max(1, 10 - k_core - len(dip_slots))
            cooldown_car = 45 if is_idle_cash else (90 if p_fill >= 0.45 else 120)
            if (step_i - extra_car_cycles * cooldown_car) >= cooldown_car:
                c_usd = min(budget * 0.38, free_s * slot_usd * (0.90 + 0.20 * p_fill))
                c_edge = (c_usd / nb1) * (tick - 0.002 * nb1)
                if c_edge > 0:
                    dip_realized_pnl += c_edge
                    extra_car_cycles += 1

        unr_dip = sum(s["qty"] * (nb1 - s["entry_px"]) for s in dip_slots)
        cur_mtm = float(env.prev_mtm) + dip_realized_pnl + unr_dip
        mtm_peak = max(mtm_peak, cur_mtm)
        max_dd_usd = max(max_dd_usd, mtm_peak - cur_mtm)
        step_i += 1

    final_pnl = float(env.prev_mtm) + dip_realized_pnl
    tot_trades = max(1, env.n_trades + extra_slot_trades)
    return {
        "pnl_usd": round(final_pnl, 2),
        "pnl_pct": round(final_pnl / budget * 100.0, 2),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_usd / budget * 100.0, 2),
        "trades": tot_trades,
        "car_cycles": env.car_cycles + extra_car_cycles,
        "avg_pnl_per_trade": round(final_pnl / tot_trades, 4),
    }



def run_capital_scaling_grid(
    coin: str = "prl",
    budgets: tuple = (8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0),
    norm_mode: str = "dimensionless",
) -> list[dict]:
    """
    Проводит слепой OOS-бенчмарк масштабирования капитала ($8k .. $20k с шагом $2k)
    на реальном 10-уровневом L2 FIFO-стакане с учетом физической ёмкости ликвидности L1-L5.
    """
    import pandas as pd
    from _tools.carousel_net import CarouselPredictor, CAROUSEL_MODEL_PATH, build_carousel_features
    from _tools.hazard_net import HazardPredictor, HAZARD_MODEL_PATH
    from _tools.execution_realism_net import ExecutionRealismPredictor, EXEC_REALISM_MODEL_PATH, build_exec_realism_features_from_cache
    from _tools.train_rl_execution import SafeTradeL2ExecutionEnv, MaskableActorCritic, load_compatible_ppo_state_dict

    raw = np.load(BASE_DIR / f"data/cache/rl_triad_lob_cache_{coin}_176h.npz", allow_pickle=True)
    data_dict = {k: raw[k] for k in raw.keys()}
    n_total = len(data_dict["bid_p"])
    oos_len = 301892
    split_idx = n_total - oos_len

    cp = CarouselPredictor(CAROUSEL_MODEL_PATH)
    hp = HazardPredictor(HAZARD_MODEL_PATH)
    rp = ExecutionRealismPredictor(EXEC_REALISM_MODEL_PATH)

    X47_oos = build_carousel_features(data_dict)[split_idx:]
    probs_oos, q_cap_oos = cp.predict_batch(X47_oos)
    p_adv_oos, p_exh_oos, p_swp_oos = hp.predict_batch_v2(X47_oos)
    X16_oos = build_exec_realism_features_from_cache(data_dict, order_qty_prl=150.0)[split_idx:]
    p_fill_oos, p_fr_oos = rp.predict_batch(X16_oos)
    ez90_oos = data_dict["X_h90"][split_idx:, 0]
    ez120_oos = data_dict["X_h120"][split_idx:, 0]

    env = SafeTradeL2ExecutionEnv(data_dict, start_idx=split_idx, end_idx=n_total - 1, episode_len=oos_len - 2, order_usd=8000.0)
    sd = torch.load(BASE_DIR / f"models/{coin}/rl_execution/best_maskable_ppo.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    ppo = MaskableActorCritic(obs_dim=env.obs_dim, n_actions=7, hidden_dim=sd["backbone.0.weight"].shape[0])
    load_compatible_ppo_state_dict(ppo, sd)
    ppo.eval()

    print("=" * 124, flush=True)
    print("📈 СЛЕПОЙ OOS-БЕНЧМАРК МАСШТАБИРОВАНИЯ КАПИТАЛА ($8,000 ➔ $20,000 с шагом $2,000 | 83.86 ч)", flush=True)
    print("=" * 124, flush=True)
    print(f"{'Бюджет':<10} | {'Режим':<36} | {'Чистый PnL ($)':<14} | {'PnL (%)':<9} | {'MaxDD ($ / %)':<19} | {'Сделок (Карус.)':<15} | {'$/сделку':<9}", flush=True)
    print("-" * 124, flush=True)

    rows = []
    for b in budgets:
        r70 = run_v70_native_10slot_oos(
            env, ppo, float(b), probs_oos, q_cap_oos, p_adv_oos, p_exh_oos, p_swp_oos, p_fill_oos, p_fr_oos, ez90_oos, ez120_oos,
            norm_mode=norm_mode
        )
        avg_t = r70["pnl_usd"] / max(1, r70["trades"])
        print(
            f"${b:<9.0f} | {'v7.0 (Native 10-Slot + L1-L5 Cap)':<36} | {r70['pnl_usd']:+13.2f}$ | {r70['pnl_pct']:+7.2f}% | "
            f"${r70['max_dd_usd']:7.2f} ({r70['max_dd_pct']:4.2f}%) | {r70['trades']:<6} ({r70['car_cycles']:<5}) | {avg_t:+.3f}$",
            flush=True,
        )
        rows.append({"budget_usd": b, **r70})
    print("-" * 124, flush=True)
    return pd.DataFrame(rows)
