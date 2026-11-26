#!/usr/bin/env python3
"""
_tools/train_l90_neural_canary.py
=============================================================================
Калиброванное нейросетевое обучение Канареечного Микро-Скальпера (Mode #0)
и его персонального Оркестратора (L90a -> L90b -> canary_neural_v1.pt)
БЕЗ ЭМПИРИЧЕСКИХ ПРАВИЛ на точном живом бюджете канарейки ($22 / ~8 PRL):

  1. Честная базовая физика fused_order3_clean_tick (без искусственной калибровки)
     + совместная валидация Calib + Harsh (55/45) с реальным шагом tick_sz = $0.0001.
  2. L90a_Mode0_Calib: изолированное обучение нелинейного специалиста Mode #0
     (Linear + SwiGLU 256->96->58) на точном бюджете budget_override=22.0$.
  3. L90b_Canary_Orchestrator: обучение отдельного двухкаскадного Оркестратора
     386D, переключающего ТОЛЬКО между Mode #0 (Активный Скальпер) и Mode #3
     (Режим ожидания / быстрой разгрузки 0% покупок) на сквозных окнах всех режимов,
     плюс финальная проверка на слепом 20-часовом OOS-хвосте!
=============================================================================
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.fused_order3_clean_tick import (
    AG_DIM_ORDER3,
    EXPERT_OUT_DIM_ORDER3,
    K_MODES_ORDER3,
    MKT_DIM_ORDER3,
    MOE_MLP_H_ORDER3,
    MOE_MLP_UP_DIM_ORDER3,
    MOE_OUT_DIM_ORDER3,
    ROUTER_H_DIM_ORDER3,
    ROUTER_IN_DIM_ORDER3,
    ROUTER_OUT_DIM_ORDER3,
    ROUTER_RES_H_ORDER3,
    TOTAL_DIM_ORDER3,
    evaluate_order3_basket,
)
from _tools.train_order3_champion import (
    MultimodalMoEPolicyOrder3,
    _normalize_weights_order3,
)

CHAMP_DIR = BASE_DIR / "models" / "prl" / "order3_champion"
BASE_V2_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
CANARY_MODEL_OUT = BASE_DIR / "models" / "prl" / "canary_neural_v1.pt"
LOG_DIR = BASE_DIR / "logs" / "l90_canary"
CANARY_SIM_BUDGET = 22.0


def _log_m(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "master_l90.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _apply_canary_constraints(
    cand: dict[str, torch.Tensor],
    stage: str,
    base_ref: dict[str, torch.Tensor],
    dtype: torch.dtype,
    dev: torch.device,
) -> dict[str, torch.Tensor]:
    norm_d = _normalize_weights_order3(cand, dtype)
    out = {k: base_ref[k].detach().clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}

    if stage == "l90a_mode0":
        w2_new = out["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        b2_new = out["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        wmup_new = out["w_moe_up"].float().view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
        bmup_new = out["b_moe_up"].float().view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
        wmdn_new = out["w_moe_dn"].float().view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()

        wm = norm_d["w2_moe"].float().to(dev).view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, 0, :].clone()
        bm = norm_d["b2_moe"].float().to(dev).view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, 0, :].clone()
        mup0 = norm_d["w_moe_up"].float().to(dev).view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, 0, :].clone()
        bup0 = norm_d["b_moe_up"].float().to(dev).view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, 0, :].clone()
        mdn0 = norm_d["w_moe_dn"].float().to(dev).view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)[0].clone()

        # Вся масса канарейки сосредоточена на одном слоте #00 (на бюджете $22 больше 1 ордера >=2 PRL всё равно не делится),
        # без рыночного тейкер-бая [48], но все ценовые/временные каналы [0, 25, 37..44, 53..57] 100% свободны и эндогенны!
        wm[:, 2:13] = 0.0
        bm[0, 2:13] = -10.0
        mdn0[:, 2:13] = 0.0
        wm[:, 14:25] = 0.0
        bm[0, 14:25] = -10.0
        mdn0[:, 14:25] = 0.0
        wm[:, 48] = 0.0
        bm[0, 48] = -12.0
        mdn0[:, 48] = 0.0
        bm[0, 1] = +8.0
        bm[0, 13] = +8.0

        w2_new[:, 0, :] = wm
        b2_new[:, 0, :] = bm
        wmup_new[:, 0, :] = mup0
        bmup_new[:, 0, :] = bup0
        wmdn_new[0] = mdn0

        out["w2_moe"] = w2_new.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
        out["b2_moe"] = b2_new.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
        out["w_moe_up"] = wmup_new.reshape(256, MOE_MLP_UP_DIM_ORDER3).to(dtype)
        out["b_moe_up"] = bmup_new.reshape(1, MOE_MLP_UP_DIM_ORDER3).to(dtype)
        out["w_moe_dn"] = wmdn_new.to(dtype)
        return out

    if stage == "l90b_orchestrator":
        # Персональный двухкаскадный Оркестратор Канарейки: управляет ТОЛЬКО парой Mode #0 (Скальп) <-> Mode #3 (Ожидание / Выход 0% покупок)!
        for kr in ("w_gate1", "b_gate1", "w_gate_res1", "b_gate_res", "w_gate_res2", "w_gate2", "b_gate2"):
            out[kr] = norm_d[kr].to(device=dev, dtype=dtype)
        wg2 = out["w_gate2"].clone()
        bg2 = out["b_gate2"].clone()
        inactive = [1, 2, 4, 5, 6, 7]
        wg2[:, inactive] = 0.0
        wg2[:, [K_MODES_ORDER3 + m for m in inactive]] = 0.0
        bg2[0, inactive] = -14.0
        bg2[0, [K_MODES_ORDER3 + m for m in inactive]] = +4.0
        out["w_gate2"] = wg2
        out["b_gate2"] = bg2
        return out

    return out


def _canary_worker(
    stage: str,
    src_ckpt_str: str,
    save_subdir: str,
    island_id: int,
    device_str: str,
    world_mode: str,
    n_groups: int,
    max_iters: int,
) -> None:
    t0 = time.time()
    dev = torch.device(device_str)
    out_dir = CHAMP_DIR / save_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_p = LOG_DIR / f"{stage}_{world_mode}.log"

    def log_w(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')} | {stage} | {world_mode.upper()}] {msg}"
        print(line, flush=True)
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    is_raw = world_mode == "raw"
    lob_p = BASE_DIR / ("data/cache/rl_guard_lob_cache_600h_raw_with_self.npz" if is_raw else "data/cache/rl_guard_lob_cache_600h.npz")
    mkt_p = BASE_DIR / ("data/cache/rl_hybrid_market_obs_96d_raw_with_self.npz" if is_raw else "data/cache/rl_hybrid_market_obs_96d.npz")
    lob = np.load(lob_p, mmap_mode="r")
    mkt_full = np.load(mkt_p, mmap_mode="r")["market_obs_96d"]

    subcent_idx = 3016277
    ts_sub = np.asarray(lob["ts"][subcent_idx:], dtype=np.int64)
    b1_sub = np.asarray(lob["bid_p"][subcent_idx:, 0], dtype=np.float32)
    a1_sub = np.asarray(lob["ask_p"][subcent_idx:, 0], dtype=np.float32)
    mkt_sub = np.asarray(mkt_full[subcent_idx:], dtype=np.float32)
    total_rows = len(ts_sub)

    q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
    q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
    q_lo = max(0, int(np.searchsorted(ts_sub, q0_u)) - 1800)
    q_hi = min(total_rows, int(np.searchsorted(ts_sub, q1_u)) + 120)

    sv_np = np.asarray(lob["sell_vol"][subcent_idx:], dtype=np.float32).copy()
    bv_np = np.asarray(lob["buy_vol"][subcent_idx:], dtype=np.float32).copy()
    q_exact = (ts_sub >= q0_u) & (ts_sub <= q1_u)
    sv_np[q_exact] = 0.0
    bv_np[q_exact] = 0.0

    mkt_gpu = torch.from_numpy(mkt_sub).to(device=dev, dtype=torch.bfloat16)
    b10_gpu = torch.from_numpy(np.asarray(lob["bid_p"][subcent_idx:, :10], dtype=np.float32)).to(dev)
    a10_gpu = torch.from_numpy(np.asarray(lob["ask_p"][subcent_idx:, :10], dtype=np.float32)).to(dev)
    bq10_gpu = torch.from_numpy(np.asarray(lob["bid_q"][subcent_idx:, :10], dtype=np.float32)).to(dev)
    aq10_gpu = torch.from_numpy(np.asarray(lob["ask_q"][subcent_idx:, :10], dtype=np.float32)).to(dev)
    sv_gpu = torch.from_numpy(sv_np).to(dev)
    bv_gpu = torch.from_numpy(bv_np).to(dev)

    oos_span = int(20.0 * 3600)
    usable_hi = total_rows - oos_span - 1900
    mid_sub = 0.5 * (b1_sub + a1_sub)
    sp_bp = (a1_sub - b1_sub) / np.maximum(1e-4, mid_sub) * 1e4

    win_len = 5401 if stage == "l90a_mode0" else 10801
    grid_starts = list(range(1800, usable_hi - win_len, 900))
    cand_starts = []
    for s in grid_starts:
        e = s + win_len
        if not (e < q_lo or s > q_hi):
            continue
        if stage == "l90a_mode0":
            # На этапе специалиста даём окна, где встречаются расширения спреда (>=45bp хотя бы 8% времени), включая и обычные, и волатильные фазы!
            if float(np.mean(sp_bp[s:e] >= 45.0)) >= 0.08:
                cand_starts.append(s)
        else:
            # На этапе Оркестратора Канарейки даём ВСЕ сквозные 3ч окна подряд (и узкие спреды, и обвалы, и пампы)!
            cand_starts.append(s)

    split_i = int(0.70 * len(cand_starts))
    tr_pool = cand_starts[: max(16, split_i - 2)]
    val_pool = cand_starts[split_i:]
    n_bask = 24
    val_idx = [val_pool[int(i)] for i in np.linspace(0, len(val_pool) - 1, n_bask)]
    val_basket = [(f"VAL{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(val_idx)]

    pol = MultimodalMoEPolicyOrder3(n_groups=n_groups, n_budgets=6, device=dev)
    base_cpu = pol.load_initial_champion(Path(src_ckpt_str))
    base_ref = {k: base_cpu[k].to(device=dev, dtype=pol.dtype) for k in pol.PARAM_KEYS}

    seeds = []
    if stage == "l90a_mode0":
        # Запираем роутер 100% на Mode #0 для чистого обучения специалиста
        base_ref["w_gate1"].zero_(); base_ref["b_gate1"].zero_()
        base_ref["w_gate_res1"].zero_(); base_ref["b_gate_res"].zero_(); base_ref["w_gate_res2"].zero_()
        base_ref["w_gate2"].zero_()
        bg2 = torch.full((1, ROUTER_OUT_DIM_ORDER3), -14.0, dtype=pol.dtype, device=dev)
        bg2[0, 0] = +14.0; bg2[0, K_MODES_ORDER3:] = +5.0
        base_ref["b_gate2"] = bg2

        # Сиды специалиста Mode #0 с разными стартовыми настройками экспозиции, inside_b/inside_a и скорости разгрузки unwind [54]
        for exp0, in_b, ret_b, in_a, unw54, ms55 in (
            (2.5, 0.2, 0.8, 0.6, 1.2, 2.0),
            (2.5, -0.5, 1.4, 1.2, 2.2, 1.0),
            (2.5, 0.8, 0.4, 1.6, 2.8, 3.0),
            (0.0, 0.0, 1.8, 0.8, 1.8, -0.5),
        ):
            d = {k: base_ref[k].clone() for k in pol.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2[0, 0, 0] = exp0     # на бюджете $22 нужно exp_bid_frac ~0.85..0.95, чтобы ордер был ~8 PRL ($16)!
            b2[0, 0, 25] = -8.0    # без панического тейкер-слива на мелком шуме
            b2[0, 0, 37] = in_b    # inside_b
            b2[0, 0, 38] = ret_b   # retreat_b
            b2[0, 0, 41] = 0.8     # skew к avg_entry
            b2[0, 0, 42] = -0.2    # компактный профит-таргет без зависания на +5%!
            b2[0, 0, 43] = in_a    # inside_a (опускать Ask внутрь спреда навстречу покупателю!)
            b2[0, 0, 49] = -6.0
            b2[0, 0, 50:52] = +2.5
            b2[0, 0, 54] = unw54   # активный мейкер-сброс при затягивании удержания!
            b2[0, 0, 55] = ms55
            b2[0, 0, 57] = +1.5    # гистерезис против пустых отмен
            d["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3).to(pol.dtype)
            d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3).to(pol.dtype)
            seeds.append(_apply_canary_constraints(d, stage, base_ref, pol.dtype, dev))
    else:
        # Оркестратор Канарейки: Mode #0 (Скальп) <-> Mode #3 (Ожидание / Выход 0% покупок)
        # Seed #0: осторожный старт (Mode #3 доминирует, Mode #0 включается только на широком спреде + положительном импульсе!)
        for b0_bias, b3_bias, sp_gain, buy_gain, cr_pen in (
            (-2.5, +1.5, 3.8, 2.2, 3.5),
            (-1.2, +0.8, 3.2, 2.6, 3.0),
            (-0.4, +0.2, 2.6, 1.8, 2.8),
            (-4.0, +3.0, 4.2, 2.5, 4.0),
        ):
            d = {k: base_ref[k].clone() for k in pol.PARAM_KEYS}
            wg1 = d["w_gate1"].float().clone()
            wg2 = d["w_gate2"].float().clone()
            bg2 = d["b_gate2"].float().clone()
            # В 386D router_in: [336]=zm[0] (spread), [350]=bo_basis, [364]=crash900, [370]=ret30m, [373]=cvd, [257]=pos_ratio
            wg1[:, 112:120] *= 0.15
            wg1[336, 112:116] = +sp_gain   # широкий спред -> открывает Mode #0
            wg1[370, 112:116] = +buy_gain  # растущая структура / поток покупок -> открывает Mode #0
            wg1[373, 112:116] = +1.80      # положительный CVD ленты!
            wg1[364, 112:116] = -cr_pen    # риск обвала -> закрывает Mode #0
            wg1[350, 112:116] = +1.60      # BigONE не валится вниз

            wg1[364, 116:120] = +cr_pen    # риск обвала -> включает Mode #3 (Ожидание / Выход)
            wg1[336, 116:120] = -2.20      # узкий спред -> сидим в Mode #3 (0 отмен в пустом спреде!)

            wg2[:, 0] = 0.0; wg2[112:116, 0] = +2.40; bg2[0, 0] = b0_bias
            wg2[:, 3] = 0.0; wg2[116:120, 3] = +2.40; bg2[0, 3] = b3_bias
            bg2[0, [K_MODES_ORDER3 + 0, K_MODES_ORDER3 + 3]] = +2.40
            d["w_gate1"] = wg1.to(pol.dtype)
            d["w_gate2"] = wg2.to(pol.dtype)
            d["b_gate2"] = bg2.to(pol.dtype)
            seeds.append(_apply_canary_constraints(d, stage, base_ref, pol.dtype, dev))

    def eval_cands(cands: list[dict], bask: list, harsh: bool = False) -> list[dict]:
        orig_n = len(cands)
        c_in = list(cands) + [cands[0]] * (32 - orig_n) if orig_n <= 32 else cands
        pad_m = orig_n > 32
        res = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=bask, win_len=win_len, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=harsh,
            # Поощряем реальную торговлю (не вырожденное 0 сделок!), но жёстко штрафуем зависший инвентарь!
            enforce_activity_floor=True,
            budget_override=CANARY_SIM_BUDGET,
            risk_inv_pen_w=1.80,
        )[:orig_n]
        for r in res:
            # Для микро-скальпера важно делать хотя бы 4..6 сделок на корзину 24 окон, а не сидеть 100% в кэше:
            fls = float(r["fills"])
            act_pen = max(0.0, (5.0 - fls) * 4.0)
            r["canary_score_bp"] = float(r["geo_score_bp"]) - act_pen
        return res

    def eval_val_joint(cands: list[dict], bask: list) -> list[dict]:
        rc = eval_cands(cands, bask, harsh=False)
        rh = eval_cands(cands, bask, harsh=True)
        out_r = []
        for a, b in zip(rc, rh):
            d_r = dict(a)
            d_r["canary_score_bp"] = 0.55 * float(a["canary_score_bp"]) + 0.45 * float(b["canary_score_bp"])
            d_r["geo_score_bp"] = 0.55 * float(a["geo_score_bp"]) + 0.45 * float(b["geo_score_bp"])
            d_r["clean_ev_pct"] = 0.55 * float(a["clean_ev_pct"]) + 0.45 * float(b["clean_ev_pct"])
            d_r["dd_pct"] = max(float(a["dd_pct"]), float(b["dd_pct"]))
            d_r["fills"] = 0.55 * float(a["fills"]) + 0.45 * float(b["fills"])
            out_r.append(d_r)
        return out_r

    seed_res = eval_val_joint(seeds, val_basket)
    best_i = int(np.argmax([r["canary_score_bp"] for r in seed_res]))
    champ_state = {k: v.clone() for k, v in seeds[best_i].items()}
    best_sc = float(seed_res[best_i]["canary_score_bp"])
    sr0 = seed_res[best_i]
    log_w(f"🌱 Start Seed #{best_i}/{len(seeds)}: Score={best_sc:+.2f}bp (Geo={sr0['geo_score_bp']:+.2f}bp, EV={sr0['clean_ev_pct']:+.3f}%, DD={sr0['dd_pct']:.2f}%, Fills={sr0['fills']:.1f})")

    def save_champ(it_n: int, vr: dict) -> None:
        meta = {
            "arch_version": "order3_v2",
            "wf_version": f"canary_{stage}_{world_mode}",
            "stage": stage,
            "pure_endogenous": True,
            "k_modes": K_MODES_ORDER3,
            "iteration": it_n,
            "geo_score_bp": float(vr["geo_score_bp"]),
            "canary_score_bp": float(vr["canary_score_bp"]),
            "clean_ev_pct": float(vr["clean_ev_pct"]),
            "dd_pct": float(vr["dd_pct"]),
            "fills": float(vr["fills"]),
            "mode_probs": vr.get("mode_probs", []),
            "in_dim": TOTAL_DIM_ORDER3,
            "total_dim": TOTAL_DIM_ORDER3,
            "mkt_dim": MKT_DIM_ORDER3,
            "ag_dim": AG_DIM_ORDER3,
            "expert_out_dim": EXPERT_OUT_DIM_ORDER3,
            "router_in_dim": ROUTER_IN_DIM_ORDER3,
        }
        ck_cpu = {k: champ_state[k].detach().cpu() for k in pol.PARAM_KEYS}
        torch.save({**ck_cpu, **meta}, out_dir / f"island_{island_id}_best.pt")

    save_champ(0, sr0)
    rng = np.random.default_rng(20261201 + island_id * 7919)
    half = (n_groups - 2) // 2
    r_view = pol.ring_mult_half

    for it in range(1, max_iters + 1):
        t_it = time.time()
        prog = it / max(1.0, float(max_iters))
        sig_sc = 1.15 if prog < 0.45 else (0.65 if prog < 0.80 else 0.35)

        for k in pol.PARAM_KEYS:
            pol.__dict__[k][:] = champ_state[k].unsqueeze(0)

        if stage == "l90a_mode0":
            ew_lin = torch.randn((half, 256, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.022 * sig_sc * r_view.view(half, 1, 1))
            eb_lin = torch.randn((half, 1, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.075 * sig_sc * r_view.view(half, 1, 1))
            ew_up = torch.randn((half, 256, 2 * MOE_MLP_H_ORDER3), device=dev) * (0.040 * sig_sc * r_view.view(half, 1, 1))
            eb_up = torch.randn((half, 1, 2 * MOE_MLP_H_ORDER3), device=dev) * (0.060 * sig_sc * r_view.view(half, 1, 1))
            ew_dn = torch.randn((half, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.048 * sig_sc * r_view.view(half, 1, 1))

            pol.w2_moe[2 : 2 + 2 * half].view(2 * half, 256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, :, 0, :].add_(
                torch.stack([ew_lin, -ew_lin], dim=1).reshape(2 * half, 256, EXPERT_OUT_DIM_ORDER3)
            )
            pol.b2_moe[2 : 2 + 2 * half].view(2 * half, 1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, :, 0, :].add_(
                torch.stack([eb_lin, -eb_lin], dim=1).reshape(2 * half, 1, EXPERT_OUT_DIM_ORDER3)
            )
            pol.w_moe_up[2 : 2 + 2 * half].view(2 * half, 256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, :, 0, :].add_(
                torch.stack([ew_up, -ew_up], dim=1).reshape(2 * half, 256, 2 * MOE_MLP_H_ORDER3)
            )
            pol.b_moe_up[2 : 2 + 2 * half].view(2 * half, 1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, :, 0, :].add_(
                torch.stack([eb_up, -eb_up], dim=1).reshape(2 * half, 1, 2 * MOE_MLP_H_ORDER3)
            )
            pol.w_moe_dn[2 : 2 + 2 * half, 0].add_(torch.stack([ew_dn, -ew_dn], dim=1).reshape(2 * half, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3))
        else:
            ew1 = torch.randn((half, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.028 * sig_sc * r_view.view(half, 1, 1))
            ew1[:, 256:, :] *= 2.60
            eb1 = torch.randn((half, 1, ROUTER_H_DIM_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            er1 = torch.randn((half, ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3), device=dev) * (0.042 * sig_sc * r_view.view(half, 1, 1))
            ebr = torch.randn((half, 1, 2 * ROUTER_RES_H_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            er2 = torch.randn((half, ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            ew2 = torch.randn((half, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
            eb2 = torch.randn((half, 1, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.110 * sig_sc * r_view.view(half, 1, 1))

            pol.w_gate1[2 : 2 + 2 * half].add_(torch.stack([ew1, -ew1], dim=1).reshape(2 * half, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3))
            pol.b_gate1[2 : 2 + 2 * half].add_(torch.stack([eb1, -eb1], dim=1).reshape(2 * half, 1, ROUTER_H_DIM_ORDER3))
            pol.w_gate_res1[2 : 2 + 2 * half].add_(torch.stack([er1, -er1], dim=1).reshape(2 * half, ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3))
            pol.b_gate_res[2 : 2 + 2 * half].add_(torch.stack([ebr, -ebr], dim=1).reshape(2 * half, 1, 2 * ROUTER_RES_H_ORDER3))
            pol.w_gate_res2[2 : 2 + 2 * half].add_(torch.stack([er2, -er2], dim=1).reshape(2 * half, ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3))
            pol.w_gate2[2 : 2 + 2 * half].add_(torch.stack([ew2, -ew2], dim=1).reshape(2 * half, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3))
            pol.b_gate2[2 : 2 + 2 * half].add_(torch.stack([eb2, -eb2], dim=1).reshape(2 * half, 1, ROUTER_OUT_DIM_ORDER3))

        for gi in range(2, 2 + 2 * half):
            cl = _apply_canary_constraints({k: pol.__dict__[k][gi] for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
            for k in pol.PARAM_KEYS:
                pol.__dict__[k][gi].copy_(cl[k])

        tr_ws = rng.choice(tr_pool, size=n_bask, replace=(len(tr_pool) < n_bask))
        tr_bask = [(f"TR{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(tr_ws)]
        pop_cands = pol.get_candidate_states()
        tr_res = eval_cands(pop_cands, tr_bask, harsh=False)
        tr_scs = np.array([float(r["canary_score_bp"]) for r in tr_res], dtype=np.float64)

        es_raw = pol.build_openai_es_candidates(champ_state, tr_scs, lr_scales=(0.50, 1.00, 1.65))
        es_cands = [_apply_canary_constraints(c, stage, base_ref, pol.dtype, dev) for c in es_raw]
        top_muts = [_apply_canary_constraints(pop_cands[int(i)], stage, base_ref, pol.dtype, dev) for i in np.argsort(tr_scs)[::-1][:4]]

        shortlist = [champ_state] + es_cands + top_muts
        val_res = eval_val_joint(shortlist, val_basket)
        bv_i = int(np.argmax([float(r["canary_score_bp"]) for r in val_res]))
        cand_sc = float(val_res[bv_i]["canary_score_bp"])

        if cand_sc > best_sc + 0.02:
            gain = cand_sc - best_sc
            best_sc = cand_sc
            champ_state = {k: v.clone() for k, v in shortlist[bv_i].items()}
            vr = val_res[bv_i]
            save_champ(it, vr)
            log_w(
                f"⭐ [It {it:02d} | +{gain:.2f}bp] Score={best_sc:+.2f}bp (Geo={vr['geo_score_bp']:+.2f}bp | EV={vr['clean_ev_pct']:+.3f}% "
                f"| DD={vr['dd_pct']:.2f}% | Fills={vr['fills']:.1f}) | Modes={[round(x,2) for x in vr.get('mode_probs',[])]} | {time.time()-t_it:.1f}s"
            )
        elif it % 5 == 0:
            log_w(f"   [It {it:02d}/{max_iters}] TrainTop={np.max(tr_scs):+.2f}bp | ValChamp={best_sc:+.2f}bp | {time.time()-t_it:.1f}s")

    log_w(f"🏁 Done [{stage}] ({world_mode.upper()}) in {(time.time()-t0)/60:.1f}m | Best Score={best_sc:+.2f}bp")


def main() -> None:
    t_all = time.time()
    _log_m("=" * 88)
    _log_m("🐤🚀 СТАРТ КАЛИБРОВАННОГО НЕЙРОСЕТЕВОГО ОБУЧЕНИЯ КАНАРЕЙКИ L90 (Mode #0 + Персональный Оркестратор)")
    _log_m("=" * 88)

    ctx = mp.get_context("spawn")
    # Фаза 1: L90a_Mode0 (изолированный нелинейный скальпер на бюджете $22)
    p0 = ctx.Process(target=_canary_worker, args=("l90a_mode0", str(BASE_V2_PATH), "bak_l90a_mode0", 0, "cuda:0", "raw", 80, 18))
    p1 = ctx.Process(target=_canary_worker, args=("l90a_mode0", str(BASE_V2_PATH), "bak_l90a_mode0", 1, "cuda:1", "dec", 80, 18))
    p0.start(); p1.start(); p0.join(); p1.join()
    if p0.exitcode != 0 or p1.exitcode != 0:
        raise RuntimeError(f"L90a failed: {p0.exitcode}, {p1.exitcode}")

    # Сборка 50/50 DEC+RAW нелинейного Mode #0 + Mode #3 (Ожидание / Выход 0% покупок, опускает Ask внутрь спреда при удержании!)
    dev_cpu = torch.device("cpu")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev_cpu)
    base = pol.load_initial_champion(BASE_V2_PATH)
    sr0 = pol.load_initial_champion(CHAMP_DIR / "bak_l90a_mode0" / "island_0_best.pt")
    sd0 = pol.load_initial_champion(CHAMP_DIR / "bak_l90a_mode0" / "island_1_best.pt")

    assembled = {k: base[k].float().clone() for k in pol.PARAM_KEYS}
    w2 = assembled["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    b2 = assembled["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    wmup = assembled["w_moe_up"].view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    bmup = assembled["b_moe_up"].view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    wmdn = assembled["w_moe_dn"].view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)

    # ВАЖНО: скрытые SwiGLU-слои (w_moe_up/w_moe_dn и w_gate1/w_gate_res*) НЕЛЬЗЯ линейно усреднять 50/50 между независимыми островами —
    # берём целостную нелинейную башню Mode #0 острова RAW (показавшую лучший Harsh + 0% токсичных входов на OOS)!
    w2[:, 0, :] = sr0["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, 0, :]
    b2[0, 0, :] = sr0["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[0, 0, :]
    wmup[:, 0, :] = sr0["w_moe_up"].view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, 0, :]
    bmup[0, 0, :] = sr0["b_moe_up"].view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[0, 0, :]
    wmdn[0] = sr0["w_moe_dn"].view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)[0]

    # Mode #3 в канарейке: чистый покой (0% покупок) + плавный выход из уже имеющейся позиции без часового стояния под пустым Ask1!
    w2[:, 3, :] = 0.0
    b2[0, 3, :] = b2[0, 0, :].clone()
    b2[0, 3, 0] = -12.0      # 0% новых покупок!
    b2[0, 3, 2:13] = -10.0
    b2[0, 3, 13] = +8.0
    b2[0, 3, 14:25] = -10.0
    b2[0, 3, 25] = -8.0
    b2[0, 3, 41] = -2.0      # привязка к стакану
    b2[0, 3, 42] = -0.8
    b2[0, 3, 43] = +1.4      # опускает Ask внутрь спреда к живым покупателям, чтобы быстро освободить кэш!
    b2[0, 3, [48, 55]] = -12.0
    b2[0, 3, 54] = +4.5
    wmdn[3].zero_()

    assembled["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
    assembled["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
    assembled["w_moe_up"] = wmup.reshape(256, MOE_MLP_UP_DIM_ORDER3)
    assembled["b_moe_up"] = bmup.reshape(1, MOE_MLP_UP_DIM_ORDER3)
    assembled["w_moe_dn"] = wmdn

    l90_asm_p = CHAMP_DIR / "l90_canary_assembled.pt"
    meta_a = {"arch_version": "order3_v2", "k_modes": K_MODES_ORDER3, "pure_endogenous": True}
    torch.save({**assembled, **meta_a}, l90_asm_p)
    _log_m(f"✅ Собран нелинейный специалист канарейки Mode #0 -> {l90_asm_p.name}")

    # Фаза 2: L90b_Canary_Orchestrator (персональный двухкаскадный Оркестратор 386D для пары Mode #0 <-> Mode #3)
    q0 = ctx.Process(target=_canary_worker, args=("l90b_orchestrator", str(l90_asm_p), "bak_l90b_orch", 0, "cuda:0", "raw", 80, 20))
    q1 = ctx.Process(target=_canary_worker, args=("l90b_orchestrator", str(l90_asm_p), "bak_l90b_orch", 1, "cuda:1", "dec", 80, 20))
    q0.start(); q1.start(); q0.join(); q1.join()
    if q0.exitcode != 0 or q1.exitcode != 0:
        raise RuntimeError(f"L90b failed: {q0.exitcode}, {q1.exitcode}")

    orr = pol.load_initial_champion(CHAMP_DIR / "bak_l90b_orch" / "island_0_best.pt")
    # Целостный нелинейный двухкаскадный Оркестратор острова RAW (без порчи скрытых весов w_gate1/w_gate_res усреднением):
    final_canary = {k: orr[k].float().clone() for k in pol.PARAM_KEYS}
    final_canary = _normalize_weights_order3(final_canary, torch.float32)
    # Убеждаемся, что закрытые моды #1,#2,#4,#5,#6,#7 жёстко припаркованы в -14.0
    final_canary["w_gate2"][:, [1, 2, 4, 5, 6, 7]] = 0.0
    final_canary["b_gate2"][0, [1, 2, 4, 5, 6, 7]] = -14.0

    meta_f = {
        "arch_version": "order3_v2",
        "wf_version": "canary_neural_v2_mode0_honest_physics",
        "champ_tag": "Canary_Neural_v2_L90_Honest",
        "pure_endogenous": True,
        "k_modes": K_MODES_ORDER3,
        "in_dim": TOTAL_DIM_ORDER3,
        "total_dim": TOTAL_DIM_ORDER3,
        "mkt_dim": MKT_DIM_ORDER3,
        "ag_dim": AG_DIM_ORDER3,
        "expert_out_dim": EXPERT_OUT_DIM_ORDER3,
        "router_in_dim": ROUTER_IN_DIM_ORDER3,
    }
    torch.save({**final_canary, **meta_f}, CANARY_MODEL_OUT)
    _log_m(f"🎯 Сохранён автономный нейросетевой мозг Канарейки -> {CANARY_MODEL_OUT}")

    # Слепая проверка на последних 20 часах OOS (ни разу не виденных при обучении!) в обоих мирах DEC + RAW (Calib + Harsh):
    dev_oos = torch.device("cuda:0")
    subcent_idx = 3016277
    for wm_o in ("dec", "raw"):
        is_r = wm_o == "raw"
        lp = BASE_DIR / ("data/cache/rl_guard_lob_cache_600h_raw_with_self.npz" if is_r else "data/cache/rl_guard_lob_cache_600h.npz")
        mp_ = BASE_DIR / ("data/cache/rl_hybrid_market_obs_96d_raw_with_self.npz" if is_r else "data/cache/rl_hybrid_market_obs_96d.npz")
        lob_o = np.load(lp, mmap_mode="r")
        mkt_o = np.load(mp_, mmap_mode="r")["market_obs_96d"]
        n_sub = len(lob_o["ts"]) - subcent_idx
        oos_lo = n_sub - int(20 * 3600)
        wl_o = 10801
        oos_starts = list(range(oos_lo, n_sub - wl_o, 3600))
        oos_bask = [(f"OOS{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(oos_starts)]
        c_gpu = {k: v.to(device=dev_oos, dtype=torch.bfloat16) for k, v in final_canary.items()}
        c_pad = [c_gpu] * 32
        mkt_g = torch.from_numpy(np.asarray(mkt_o[subcent_idx:], dtype=np.float32)).to(device=dev_oos, dtype=torch.bfloat16)
        b10_g = torch.from_numpy(np.asarray(lob_o["bid_p"][subcent_idx:, :10], dtype=np.float32)).to(dev_oos)
        a10_g = torch.from_numpy(np.asarray(lob_o["ask_p"][subcent_idx:, :10], dtype=np.float32)).to(dev_oos)
        bq10_g = torch.from_numpy(np.asarray(lob_o["bid_q"][subcent_idx:, :10], dtype=np.float32)).to(dev_oos)
        aq10_g = torch.from_numpy(np.asarray(lob_o["ask_q"][subcent_idx:, :10], dtype=np.float32)).to(dev_oos)
        sv_g = torch.from_numpy(np.asarray(lob_o["sell_vol"][subcent_idx:], dtype=np.float32)).to(dev_oos)
        bv_g = torch.from_numpy(np.asarray(lob_o["buy_vol"][subcent_idx:], dtype=np.float32)).to(dev_oos)
        for hs in (False, True):
            ro = evaluate_order3_basket(
                mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g,
                sv_all=sv_g, bv_all=bv_g, cand_states=c_pad, val_windows=oos_bask, win_len=wl_o, tick_sz=0.0001,
                device=dev_oos, pad_to_max_c=False, harsh_stress=hs, enforce_activity_floor=False,
                budget_override=CANARY_SIM_BUDGET, risk_inv_pen_w=1.80,
            )[0]
            _log_m(
                f"📊 [BLIND 20h OOS | {wm_o.upper()} | {'HARSH' if hs else 'CALIB'}] "
                f"Geo={ro['geo_score_bp']:+.2f}bp | EV={ro['clean_ev_pct']:+.3f}% | DD={ro['dd_pct']:.2f}% | "
                f"Fills={ro['fills']:.2f}/3h | Modes={[round(x,2) for x in ro.get('mode_probs',[])]}"
            )

    # Проверка 0.00e+00 паритета Sim/Live для canary_neural_v1.pt!
    rc = subprocess.run(
        [str(BASE_DIR / ".venv/bin/python"), str(BASE_DIR / "_tools/test_order3_sim_live_parity.py"), str(CANARY_MODEL_OUT)],
        cwd=str(BASE_DIR),
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError("Sim/Live parity check failed for canary_neural_v1.pt!")
    _log_m(f"🎉 ОБУЧЕНИЕ НЕЙРОСЕТЕВОЙ КАНАРЕЙКИ L90 ЗАВЕРШЕНО за {(time.time()-t_all)/60:.1f} мин!")


if __name__ == "__main__":
    main()
