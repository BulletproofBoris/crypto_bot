#!/usr/bin/env python3
"""
_tools/train_l83_l84_specialists_and_router.py
=============================================================================
Модульная Школа Изолированных Специалистов (L83a..L83c) и Оркестратора (L84):
  1. mode0_scalp    : Изолированное обучение Моды #0 (Микро-скальп широкого спреда >=38..50bp)
                      на профильных окнах широкого спреда без обвалов.
  2. mode1_midwick  : Изолированное обучение Моды #1 (Умеренные проливы -1.6%..-2.6%
                      в спокойные/зелёные дни без мега-крашей, без убегания в -4%).
  3. mode2_upcarry  : Изолированное обучение Моды #2 (UP-Carry + Продавец всплесков вверх F
                      через якорь рециклинга ch[56] + выход на 2ч перегреве ch[94]).
  4. assemble       : Хирургическая сборка 6 активных ролей (#0,#1,#2 + #3 Бункер + #4,#7 L85)
                      и консервация резерва (#5,#6 -> -14.0) поверх эталонного ствола L85.
  5. router         : Изолированный Урок Оркестратора (L84_Router) — 100% заморозка ствола
                      и всех 8 экспертов, обучение ТОЛЬКО шлюзов w_gate1/b_gate1/w_gate2/b_gate2
                      на сквозных мультирежимных корзинах (Macro + Micro + 24x6h Slide).
=============================================================================
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.fused_order3_clean_tick import (
    AG_DIM_ORDER3,
    EXPERT_OUT_DIM_ORDER3,
    K_MODES_ORDER3,
    MEM_DIM_ORDER3,
    MKT_DIM_ORDER3,
    MOE_OUT_DIM_ORDER3,
    ROUTER_H_DIM_ORDER3,
    ROUTER_IN_DIM_ORDER3,
    ROUTER_OUT_DIM_ORDER3,
    TOTAL_DIM_ORDER3,
    evaluate_order3_basket,
)
from _tools.train_order3_champion import (
    MultimodalMoEPolicyOrder3,
    _normalize_weights_order3,
)

CHAMP_DIR = BASE_DIR / "models" / "prl" / "order3_champion"
LIVE_L85_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"


def _lock_router_to_single_mode(st: dict[str, torch.Tensor], target_mode: int, dtype: torch.dtype, dev: torch.device) -> dict[str, torch.Tensor]:
    """Forces Liquid Router to route 99.9999% weight into target_mode during specialist training."""
    d = {k: v.detach().clone().to(device=dev, dtype=dtype) for k, v in st.items()}
    d["w_gate1"].zero_()
    d["b_gate1"].zero_()
    d["w_gate2"].zero_()
    bg2 = torch.full((1, ROUTER_OUT_DIM_ORDER3), -14.0, dtype=dtype, device=dev)
    bg2[0, target_mode] = +14.0
    bg2[0, K_MODES_ORDER3:] = +5.0  # fast beta_gate=0.993 so target_mode is active from tick 1
    d["b_gate2"] = bg2
    return d


def _clamp_specialist_constraints(d: dict[str, torch.Tensor], stage: str, base_ref: dict[str, torch.Tensor], dtype: torch.dtype, dev: torch.device) -> dict[str, torch.Tensor]:
    """Guarantees 100% bitwise freeze on trunk and non-target modes, plus specialist structural bounds."""
    out = {}
    if stage in ("mode0_scalp", "mode1_midwick", "mode2_upcarry"):
        t_m = {"mode0_scalp": 0, "mode1_midwick": 1, "mode2_upcarry": 2}[stage]
        for k in MultimodalMoEPolicyOrder3.PARAM_KEYS:
            if k not in ("w2_moe", "b2_moe"):
                out[k] = base_ref[k].detach().clone().to(device=dev, dtype=dtype)
        w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        w2_ref = base_ref["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).to(dev)
        b2_ref = base_ref["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).to(dev)

        # Normalize only target mode column, copy all other 7 modes bit-for-bit from base_ref!
        w2_new = w2_ref.clone()
        b2_new = b2_ref.clone()
        wm = w2[:, t_m, :].to(dev)
        bm = torch.clamp(b2[:, t_m, :].to(dev), -12.0, 8.5)
        n_wm = torch.linalg.vector_norm(wm, dim=0, keepdim=True).clamp(min=1e-6)
        wm = wm * torch.clamp(5.2 / n_wm, max=1.0)

        # Always forbid impulse taker buy [48]
        wm[:, 48] = 0.0
        bm[0, 48] = -10.0

        if stage == "mode0_scalp":
            # Mode #0: Wide-spread micro-scalper (cap total exposure <= 14%, keep deep ladder slots 2..11 off)
            bm[0, 0] = torch.clamp(bm[0, 0], -4.2, -1.8)   # exp <= 14%
            wm[:, 2:13] = 0.0
            bm[0, 2:13] = -9.0
            bm[0, 38] = torch.clamp(bm[0, 38], 1.5, 4.5)   # non-sniper fallback stays deep
            bm[0, 55] = torch.clamp(bm[0, 55], -1.5, 5.0)  # active micro_sniper_z [55]!
        elif stage == "mode1_midwick":
            # Mode #1: Moderate dip harvester (-1.6%..-2.7%) — prevent saturation to -4.0%!
            n_r38 = torch.linalg.vector_norm(wm[:, 38]).clamp(min=1e-6)
            wm[:, 38] = wm[:, 38] * min(1.0, float(1.15 / n_r38))
            bm[0, 38] = torch.clamp(bm[0, 38], -0.45, +0.50)  # sigmoid(-0.45..+0.50)*4% = 1.56%..2.49% base!
            wm[:, 3:13] = 0.0
            bm[0, 3:13] = -8.5  # max 2 concentrated bid slots (D2)
            wm[:, 55] = 0.0
            bm[0, 55] = -10.0
        elif stage == "mode2_upcarry":
            # Mode #2: UP-Carry + Up-Wick Seller F (ch[56] recycle anchor active, max 2 bid slots, no scalp [55])
            wm[:, 3:13] = 0.0
            bm[0, 3:13] = -8.5
            wm[:, 55] = 0.0
            bm[0, 55] = -10.0
            bm[0, 56] = torch.clamp(bm[0, 56], -0.5, 4.5)  # active recycle anchor!

        w2_new[:, t_m, :] = wm
        b2_new[:, t_m, :] = bm
        out["w2_moe"] = w2_new.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
        out["b2_moe"] = b2_new.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
        return out

    elif stage == "router":
        # 100% freeze trunk, memory, slot regulator, AND all 8 experts! Train ONLY w_gate1, b_gate1, w_gate2, b_gate2!
        norm_d = _normalize_weights_order3(d, dtype)
        router_keys = {"w_gate1", "b_gate1", "w_gate2", "b_gate2"}
        for k in MultimodalMoEPolicyOrder3.PARAM_KEYS:
            if k in router_keys:
                out[k] = norm_d[k].to(device=dev, dtype=dtype)
            else:
                out[k] = base_ref[k].detach().clone().to(device=dev, dtype=dtype)
        # Keep reserve slots #5 and #6 strictly parked at -14.0 (0.000% probability, zero noise!)
        wg2 = out["w_gate2"].clone()
        bg2 = out["b_gate2"].clone()
        wg2[:, [5, 6]] = 0.0
        wg2[:, [K_MODES_ORDER3 + 5, K_MODES_ORDER3 + 6]] = 0.0
        bg2[0, [5, 6]] = -14.0
        bg2[0, [K_MODES_ORDER3 + 5, K_MODES_ORDER3 + 6]] = +4.0
        out["w_gate2"] = wg2
        out["b_gate2"] = bg2
        return out

    return _normalize_weights_order3(d, dtype)


def _build_specialist_seeds(base_st: dict[str, torch.Tensor], stage: str, dtype: torch.dtype, dev: torch.device) -> list[dict[str, torch.Tensor]]:
    t_m = {"mode0_scalp": 0, "mode1_midwick": 1, "mode2_upcarry": 2}[stage]
    locked_base = _lock_router_to_single_mode(base_st, t_m, dtype, dev)
    w2_b = locked_base["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    b2_b = locked_base["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    caution_dir = -w2_b[:, 4, 0].clone()
    bull_dir = w2_b[:, 7, 0].clone()

    seeds = []
    if stage == "mode0_scalp":
        for mic_z, exp_z, z42_v, unw_z in [
            (2.40, -2.60, -3.20, 1.80),
            (3.20, -2.30, -2.40, 1.40),
            (1.60, -2.90, -3.50, 2.20),
            (0.80, -3.20, -1.80, 1.20),
        ]:
            d = {k: locked_base[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            # Start from Mode #7 exit/queue discipline, reconfigured for slot #0 wide-spread micro-scalp
            w2[:, 0, :] = 0.25 * w2_b[:, 7, :]
            b2[0, 0, :] = b2_b[0, 7, :]
            b2[0, 0, 0] = exp_z
            b2[0, 0, 1] = 8.0
            b2[0, 0, 2:13] = -9.0
            b2[0, 0, 13] = 8.0
            b2[0, 0, 14:25] = -6.5
            b2[0, 0, 25] = -8.5
            b2[0, 0, 37] = -4.0
            b2[0, 0, 38] = 3.0
            b2[0, 0, 41] = -6.5   # quote relative to Ask1 for fast spread capture
            b2[0, 0, 42] = z42_v
            b2[0, 0, 43] = 2.5    # step 1 tick inside spread on ask exit
            b2[0, 0, 54] = unw_z  # fast maker stop-unwind if mid drifts against fill
            w2[:, 0, 55] = -0.15 * caution_dir
            b2[0, 0, 55] = mic_z
            b2[0, 0, 56] = -10.0
            b2[0, 0, 57] = 0.50
            d["w2_moe"] = w2.reshape(256, -1).to(dtype)
            d["b2_moe"] = b2.reshape(1, -1).to(dtype)
            seeds.append(_clamp_specialist_constraints(d, stage, locked_base, dtype, dev))

    elif stage == "mode1_midwick":
        for r_b, exp_b, h57 in [
            (-0.10, -0.45, 1.85),   # ~1.9% / 2.9% concentrated 2-slot dip catcher
            (+0.15, -0.35, 1.95),   # ~2.15% / 3.15%
            (+0.35, -0.25, 2.05),   # ~2.35% / 3.35%
            (-0.30, -0.60, 1.75),   # ~1.70% / 2.70%
        ]:
            d = {k: locked_base[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            w2[:, 1, :] = w2_b[:, 7, :].clone()
            b2[0, 1, :] = b2_b[0, 7, :].clone()
            # Desaturate w2[:, 1, 38] so Mode #1 genuinely quotes at -1.8%..-2.5% instead of saturating to -4.0%!
            w2[:, 1, 38] = 0.12 * w2_b[:, 7, 38]
            b2[0, 1, 38] = r_b
            b2[0, 1, 0] = exp_b
            b2[0, 1, 1:3] = 8.0
            b2[0, 1, 3:13] = -8.5
            b2[0, 1, 57] = h57
            d["w2_moe"] = w2.reshape(256, -1).to(dtype)
            d["b2_moe"] = b2.reshape(1, -1).to(dtype)
            seeds.append(_clamp_specialist_constraints(d, stage, locked_base, dtype, dev))

    elif stage == "mode2_upcarry":
        st_rel_old = None
        old_p = BASE_DIR / "models" / "prl" / "release_champion_v1_backup_l79_b20.pt"
        if old_p.exists():
            pol_tmp = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)
            st_rel_old = pol_tmp.load_initial_champion(old_p)
        for r_b, z42_v, w49_v, rec56_v in [
            (0.10, 2.10, -0.40, 1.50),
            (0.35, 2.50,  0.20, 1.85),
            (-0.15, 1.80, -0.80, 1.35),
            (0.55, 2.80,  0.50, 2.10),
        ]:
            d = {k: locked_base[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            w2[:, 2, :] = 0.70 * w2_b[:, 7, :]
            b2[0, 2, :] = b2_b[0, 7, :].clone()
            if st_rel_old is not None:
                w2_o = st_rel_old["w2_moe"].float().view(256, K_MODES_ORDER3, -1).to(dev)
                ed = min(w2_o.shape[-1], EXPERT_OUT_DIM_ORDER3)
                w2[:, 2, :ed] = 0.65 * w2[:, 2, :ed] + 0.35 * w2_o[:, 1, :ed]
            w2[:, 2, 38] = 0.18 * w2_b[:, 7, 38]
            b2[0, 2, 38] = r_b
            b2[0, 2, 0] = -0.30
            b2[0, 2, 1:3] = 8.0
            b2[0, 2, 3:13] = -8.5
            b2[0, 2, 41] = 1.50    # respect avg_entry floor in UP-carry
            b2[0, 2, 42] = z42_v   # sell upward spikes +1.8%..+2.8% (Mode F)
            b2[0, 2, 49] = w49_v   # smooth wave_hold
            b2[0, 2, 54] = -0.50   # controlled stop-unwind on trend break
            w2[:, 2, 56] = 0.12 * bull_dir
            b2[0, 2, 56] = rec56_v # active recycle anchor below last_sell_px!
            b2[0, 2, 57] = 1.75
            d["w2_moe"] = w2.reshape(256, -1).to(dtype)
            d["b2_moe"] = b2.reshape(1, -1).to(dtype)
            seeds.append(_clamp_specialist_constraints(d, stage, locked_base, dtype, dev))

    return seeds


def run_assemble() -> Path:
    """Assembles the 6 orthogonal specialists + 2 parked reserve slots onto the frozen L85 DualSoup trunk."""
    dev = torch.device("cpu")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)
    base_l85 = pol.load_initial_champion(LIVE_L85_PATH)
    out = {k: base_l85[k].float().clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
    w2 = out["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    b2 = out["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)

    # Blend 50% DEC + 50% RAW specialist columns for Modes #0, #1, #2
    stage_dirs = [
        (0, CHAMP_DIR / "bak_l83a_mode0"),
        (1, CHAMP_DIR / "bak_l83b_mode1"),
        (2, CHAMP_DIR / "bak_l83c_mode2"),
    ]
    for m_idx, s_dir in stage_dirs:
        p_raw = s_dir / "island_0_best.pt"
        p_dec = s_dir / "island_1_best.pt"
        assert p_raw.exists() and p_dec.exists(), f"Missing specialist checkpoints in {s_dir}"
        st_r = pol.load_initial_champion(p_raw)
        st_d = pol.load_initial_champion(p_dec)
        w2_r = st_r["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2_r = st_r["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        w2_d = st_d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2_d = st_d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        w2[:, m_idx, :] = 0.50 * w2_d[:, m_idx, :] + 0.50 * w2_r[:, m_idx, :]
        b2[0, m_idx, :] = 0.50 * b2_d[0, m_idx, :] + 0.50 * b2_r[0, m_idx, :]
        print(f"  ✅ Подключён обученный Специалист Mode #{m_idx} (50% DEC + 50% RAW) из {s_dir.name}")

    # Mode #3: Analytical "Pure Bunker" (0% new buys during BigONE+SafeTrade crash, rapid maker unload near Ask1)
    w2[:, 3, :] = 0.0
    b2[0, 3, :] = b2[0, 4, :].clone()
    b2[0, 3, 0] = -12.0     # 0.0006% bid exposure -> zero new buys!
    b2[0, 3, 1:13] = -10.0
    b2[0, 3, 13] = 8.0
    b2[0, 3, 14:25] = -8.0
    b2[0, 3, 25] = -10.0    # no panic taker dump, pure maker unwind
    b2[0, 3, 38] = +8.0     # max retreat 4.0%
    b2[0, 3, 41] = -6.5     # quote at Ask1
    b2[0, 3, 42] = -3.0
    b2[0, 3, 48] = -12.0
    b2[0, 3, 54] = +4.5     # rapid maker stop-unwind if holding inventory
    b2[0, 3, 55] = -12.0
    b2[0, 3, 56] = -10.0
    b2[0, 3, 57] = +3.0
    print("  ✅ Подключён аналитический Специалист Mode #3 ('Чистый Бункер 0% покупок' для синхронных обвалов)")
    print("  🔒 Моды #4 и #7 сохранены побитово от боевого чемпиона L85_DualSoup_50D_50R")

    # Modes #5, #6: Reserved slots (zeroed & parked at -14.0 in router)
    w2[:, 5:7, :] = 0.0
    b2[0, 5:7, :] = b2[0, 4:5, :].clone()
    b2[0, 5:7, 55] = -12.0

    out["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
    out["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)

    # Restore original L85 router as Seed #0 baseline, with Modes #0,#1,#2,#3,#5,#6 initially quiet so Seed #0 == L85!
    wg2 = base_l85["w_gate2"].float().clone()
    bg2 = base_l85["b_gate2"].float().clone()
    wg2[:, [5, 6]] = 0.0
    wg2[:, [K_MODES_ORDER3 + 5, K_MODES_ORDER3 + 6]] = 0.0
    bg2[0, [5, 6]] = -14.0
    bg2[0, [K_MODES_ORDER3 + 5, K_MODES_ORDER3 + 6]] = +4.0
    out["w_gate2"] = wg2
    out["b_gate2"] = bg2

    dst = CHAMP_DIR / "l83_assembled_6modes.pt"
    CHAMP_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "arch_version": "order3_v1",
        "wf_version": "pure_wf_v54_l83_assembled_6specialists_96d",
        "champ_tag": "L83_Assembled_6Modes",
        "pure_endogenous": True,
        "k_modes": K_MODES_ORDER3,
        "in_dim": TOTAL_DIM_ORDER3,
        "total_dim": TOTAL_DIM_ORDER3,
        "mkt_dim": MKT_DIM_ORDER3,
        "ag_dim": AG_DIM_ORDER3,
        "expert_out_dim": EXPERT_OUT_DIM_ORDER3,
        "router_in_dim": ROUTER_IN_DIM_ORDER3,
    }
    torch.save({**out, **meta}, dst)
    print(f"🎯 Собранный чекпоинт 6 специалистов сохранён -> {dst}")
    return dst


def _build_router_seeds(assembled_st: dict[str, torch.Tensor], dtype: torch.dtype, dev: torch.device) -> list[dict[str, torch.Tensor]]:
    """
    Builds Seed Pool for L84_Router:
      - Seed #0: Zero-regression L85 router ( suppresses #0..#3 to -9.0 so #4/#7 reproduce 100% of L85_DualSoup!)
      - Seed #1..#4: Semantically wired Orchestrators using direct skip-inputs in r_extra (indices 320..361 of router_in):
          * idx 320 =zm[0] (spread) -> boosts Mode #0 (wide-spread scalper)
          * idx 354 =zm[88] (ret_30m), idx 357=zm[91] (cvd) -> boosts Mode #2 (UP-carry) & Mode #1 (calm green)
          * idx 334 =zm[12] (BigONE basis), idx 347..349=zm[79..81] (crash900) -> boosts Mode #3 (Bunker) on BO drop!
    """
    seeds = []
    # Seed #0: Exact L85_DualSoup anchor (zero regression guarantee!)
    s0 = {k: assembled_st[k].clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
    wg2_0 = s0["w_gate2"].float().clone()
    bg2_0 = s0["b_gate2"].float().clone()
    wg2_0[:, [0, 1, 2, 3, 5, 6]] = 0.0
    bg2_0[0, [0, 1, 2, 3]] = -10.0
    bg2_0[0, [5, 6]] = -14.0
    s0["w_gate2"] = wg2_0.to(dtype)
    s0["b_gate2"] = bg2_0.to(dtype)
    seeds.append(_clamp_specialist_constraints(s0, "router", assembled_st, dtype, dev))

    # Seeds #1..#4: Multi-specialist orchestrators with sharp gating and varying activation thresholds
    configs = [
        # (b_m0_scalp, b_m1_mid, b_m2_up, b_m3_bunk, sharp_mult)
        (-2.2, -1.4, -1.6, -2.0, 1.35),
        (-1.5, -0.9, -1.1, -1.5, 1.65),
        (-2.8, -1.8, -2.0, -2.4, 1.20),
        (-1.2, -0.5, -0.8, -1.2, 2.00),
    ]
    for b0, b1_m, b2_m, b3, sh_m in configs:
        d = {k: assembled_st[k].clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        wg1 = d["w_gate1"].float().clone()
        wg2 = d["w_gate2"].float().clone()
        bg2 = d["b_gate2"].float().clone()

        # Sharpen existing #4 / #7 gates
        wg2[:, [4, 7]] *= sh_m
        bg2[0, 4] *= sh_m
        bg2[0, 7] *= sh_m

        # Wire dedicated hidden channels 80..95 of router (96D) to direct regime sensors in rin [320..361]
        # rin layout: [0:256]=h2_ctx, [256:272]=za16, [272:320]=mem48, [320:362]=r_extra(42D), [362:370]=prev_gm(8D)
        # In r_extra: [0]=zm[0](spread) -> rin[320]; [14]=zm[12](bo_basis) -> rin[334];
        # [9]=zm[31](crash300) -> rin[329]; [28]=zm[81](crash900) -> rin[348];
        # [34]=zm[88](ret30m) -> rin[354]; [40]=zm[94](ret120m) -> rin[360]
        wg1[:, 80:92] *= 0.25
        # Ch 80..82: Wide-spread & calm sensor -> Mode #0
        wg1[320, 80:83] = +3.20   # wide spread
        wg1[348, 80:83] = -2.80   # low crash900
        wg1[334, 80:83] = +1.40   # non-negative BO basis
        # Ch 83..85: Calm green / flat dip sensor -> Mode #1
        wg1[354, 83:86] = +1.80   # non-falling 30m
        wg1[348, 83:86] = -3.00   # low crash900
        wg1[329, 83:86] = -2.40   # low crash300
        # Ch 86..88: Strong UP-trend & non-overheated 2h -> Mode #2
        wg1[354, 86:89] = +3.40   # strong ret_30m > 0
        wg1[360, 86:89] = -1.60   # step off on extreme 2h blowoff
        wg1[348, 86:89] = -2.20
        # Ch 89..91: Cross-venue BigONE crash / waterfall -> Mode #3 Bunker
        wg1[334, 89:92] = -3.80   # BO basis deeply negative (BO leading down!)
        wg1[348, 89:92] = +2.60   # high crash900
        wg1[329, 89:92] = +2.40   # high crash300

        wg2[:, 0] = 0.0; wg2[80:83, 0] = +2.40 * sh_m; bg2[0, 0] = b0
        wg2[:, 1] = 0.0; wg2[83:86, 1] = +2.20 * sh_m; bg2[0, 1] = b1_m
        wg2[:, 2] = 0.0; wg2[86:89, 2] = +2.40 * sh_m; bg2[0, 2] = b2_m
        wg2[:, 3] = 0.0; wg2[89:92, 3] = +2.60 * sh_m; bg2[0, 3] = b3
        wg2[:, [5, 6]] = 0.0; bg2[0, [5, 6]] = -14.0
        bg2[0, K_MODES_ORDER3 : K_MODES_ORDER3 + 4] = +2.20  # responsive liquid switching

        d["w_gate1"] = wg1.to(dtype)
        d["w_gate2"] = wg2.to(dtype)
        d["b_gate2"] = bg2.to(dtype)
        seeds.append(_clamp_specialist_constraints(d, "router", assembled_st, dtype, dev))

    return seeds


def run_stage_training(
    stage: str,
    device_str: str,
    n_groups: int,
    max_iters: int,
    hours: float,
    island_id: int,
    num_islands: int,
    world_mode: str,
    save_subdir: str,
    log_file: str | None,
) -> None:
    t_start = time.time()
    dev = torch.device(device_str)
    out_dir = CHAMP_DIR / save_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def log_msg(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    log_msg("=" * 90)
    log_msg(f"🚀 MODULAR STAGE [{stage.upper()}] | ISLAND #{island_id}/{num_islands} (WORLD={world_mode.upper()}) | POP={n_groups*6}")
    log_msg("=" * 90)

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

    # Quarantine 2026-11-24 20:05..21:10 MSK (17:05..18:10 UTC)
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
    oos_start = total_rows - oos_span - 120
    usable_hi = min(q_lo - 100, oos_start - 1800)

    # Precompute window regime descriptors every 600s over [1800 .. usable_hi]
    mid_sub = 0.5 * (b1_sub + a1_sub)
    sp_bp = (a1_sub - b1_sub) / np.maximum(1e-4, mid_sub) * 1e4
    ret30m = np.zeros(total_rows, dtype=np.float32)
    ret30m[1800:] = mid_sub[1800:] / np.maximum(1e-4, mid_sub[:-1800]) - 1.0
    crash900 = mkt_sub[:, 81]

    win_len = 3601 if stage == "mode0_scalp" else (7201 if stage in ("mode1_midwick", "mode2_upcarry") else 9001)
    grid_starts = list(range(1800, usable_hi - win_len, 900))
    cand_starts = []
    for s in grid_starts:
        e = s + win_len
        if not (e < q_lo or s > q_hi):
            continue
        sp_w = sp_bp[s:e]
        r30_w = ret30m[s:e]
        cr_w = crash900[s:e]
        dip_min = float(np.min(mid_sub[s:e] / np.maximum.accumulate(mid_sub[s:e]) - 1.0))
        if stage == "mode0_scalp":
            if float(np.mean(sp_w >= 38.0)) >= 0.14 and float(np.min(r30_w)) > -0.025 and float(np.mean(cr_w)) < 0.42:
                cand_starts.append(s)
        elif stage == "mode1_midwick":
            if -0.042 <= dip_min <= -0.015 and float(np.min(r30_w)) > -0.035 and float(np.mean(cr_w)) < 0.40:
                cand_starts.append(s)
        elif stage == "mode2_upcarry":
            if float(np.max(r30_w)) >= 0.014 and (mid_sub[e - 1] / mid_sub[s] - 1.0) >= -0.005:
                cand_starts.append(s)
        else:
            cand_starts.append(s)

    assert len(cand_starts) >= 32, f"Too few regime windows ({len(cand_starts)}) for stage={stage}"
    # Split 70% Train / 30% Val chronologically with purge gap
    n_tot_w = len(cand_starts)
    split_i = int(0.70 * n_tot_w)
    tr_pool = cand_starts[: max(16, split_i - 2)]
    val_pool_raw = cand_starts[split_i:]
    n_bask = 24
    val_idx = [val_pool_raw[int(i)] for i in np.linspace(0, len(val_pool_raw) - 1, n_bask)]
    val_basket = [(f"VAL{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(val_idx)]

    # For router stage, also include 24 x 6h Slide basket so Orchestrator proves itself on multi-hour transitions & crashes!
    val_slide_basket = None
    if stage == "router":
        slide_len = 21601
        s_lo = max(360000, usable_hi - 24 * 16000)
        s_hi = max(s_lo + 1000, usable_hi - slide_len)
        s_idx = [int(x) for x in np.linspace(s_lo, s_hi, n_bask)]
        val_slide_basket = [(f"SLD{k:02d}", ws, 0.0, 0.0) for k, ws in enumerate(s_idx)]

    log_msg(f"📊 Корзина режима [{stage}]: найдено {n_tot_w} профильных окон (win={win_len}с) -> TrainPool={len(tr_pool)}, Val={len(val_basket)}w")

    pol = MultimodalMoEPolicyOrder3(n_groups=n_groups, n_budgets=6, device=dev)
    src_path = (CHAMP_DIR / "l83_assembled_6modes.pt") if stage == "router" else LIVE_L85_PATH
    base_cpu = pol.load_initial_champion(src_path)
    base_ref = {k: base_cpu[k].to(device=dev, dtype=pol.dtype) for k in pol.PARAM_KEYS}
    if stage in ("mode0_scalp", "mode1_midwick", "mode2_upcarry"):
        t_m = {"mode0_scalp": 0, "mode1_midwick": 1, "mode2_upcarry": 2}[stage]
        base_ref = _lock_router_to_single_mode(base_ref, t_m, pol.dtype, dev)
        seeds = _build_specialist_seeds(base_ref, stage, pol.dtype, dev)
    else:
        seeds = _build_router_seeds(base_ref, pol.dtype, dev)

    def eval_cands(cands: list[dict], bask: list, w_l: int, slide_b: list | None = None, harsh: bool = True) -> list[dict]:
        orig_n = len(cands)
        if orig_n <= 32:
            c_in = list(cands) + [cands[0]] * (32 - orig_n)
            pad_m = False
        else:
            c_in = cands
            pad_m = True
        r_main = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=bask, win_len=w_l, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=harsh, enforce_activity_floor=False,
            risk_inv_pen_w=(1.40 if stage == "router" else 0.35),
        )[:orig_n]
        if slide_b is None:
            return r_main
        r_sld = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=slide_b, win_len=21601, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=harsh, enforce_activity_floor=False, risk_inv_pen_w=1.65,
        )[:orig_n]
        merged = []
        for a, s in zip(r_main, r_sld):
            # Router score: 50% multi-regime windows + 50% 6h slide/transition windows + sharp mode bonus (avoid 50/50 blur!)
            mp = np.asarray(s.get("mode_probs", [1.0]), dtype=np.float64)
            mp = mp / max(1e-9, mp.sum())
            ent = -float(np.sum(mp * np.log(np.clip(mp, 1e-9, 1.0))))
            # Gentle bonus when router uses >=2 distinct modes across 144h (mi_bits > 0.15) while keeping individual tick entropy controlled
            mi_b = float(s.get("mode_mi_bits", 0.0))
            sharp_bonus = min(6.0, mi_b * 8.0) - max(0.0, (ent - 1.15) * 8.0)
            g_comb = 0.50 * float(a["geo_score_bp"]) + 0.50 * float(s["geo_score_bp"]) + sharp_bonus
            d_m = dict(s)
            d_m["geo_score_bp"] = g_comb
            d_m["main_geo_bp"] = float(a["geo_score_bp"])
            d_m["slide_geo_bp"] = float(s["geo_score_bp"])
            d_m["clean_ev_pct"] = 0.50 * float(a["clean_ev_pct"]) + 0.50 * float(s["clean_ev_pct"])
            d_m["dd_pct"] = max(float(a["dd_pct"]), float(s["dd_pct"]))
            d_m["fills"] = float(a["fills"]) + float(s["fills"])
            merged.append(d_m)
        return merged

    seed_res = eval_cands(seeds, val_basket, win_len, slide_b=val_slide_basket, harsh=True)
    for si, sr in enumerate(seed_res):
        log_msg(
            f"   [Seed #{si}] HarshGeo={sr['geo_score_bp']:+.2f}bp | EV={sr['clean_ev_pct']:+.3f}% | DD={sr['dd_pct']:.2f}% | "
            f"Fills={sr['fills']:.1f} | Reg=[UP:{sr.get('pnl_up_bp',0):+.1f} DN:{sr.get('pnl_dn_bp',0):+.1f} FL:{sr.get('pnl_fl_bp',0):+.1f}bp] | "
            f"Modes={[round(x,2) for x in sr.get('mode_probs',[])]}"
        )
    best_i = int(np.argmax([r["geo_score_bp"] for r in seed_res]))
    champ_state = {k: v.clone() for k, v in seeds[best_i].items()}
    best_geo = float(seed_res[best_i]["geo_score_bp"])
    best_cal = eval_cands([champ_state], val_basket, win_len, slide_b=val_slide_basket, harsh=False)[0]
    log_msg(f"🌱 Старт с Seed #{best_i}: HarshGeo={best_geo:+.2f}bp | CalibEV={best_cal['clean_ev_pct']:+.3f}% (CalibDD={best_cal['dd_pct']:.2f}%)")

    def save_champ(it_n: int, h_res: dict, c_res: dict) -> None:
        meta = {
            "arch_version": "order3_v1",
            "wf_version": f"modular_{stage}_{world_mode}",
            "stage": stage,
            "pure_endogenous": True,
            "k_modes": K_MODES_ORDER3,
            "iteration": it_n,
            "geo_score_bp": float(h_res["geo_score_bp"]),
            "clean_ev_pct": float(h_res["clean_ev_pct"]),
            "dd_pct": float(h_res["dd_pct"]),
            "calib_clean_ev_pct": float(c_res["clean_ev_pct"]),
            "calib_dd_pct": float(c_res["dd_pct"]),
            "mode_probs": h_res.get("mode_probs", []),
            "in_dim": TOTAL_DIM_ORDER3,
            "total_dim": TOTAL_DIM_ORDER3,
            "mkt_dim": MKT_DIM_ORDER3,
            "ag_dim": AG_DIM_ORDER3,
            "expert_out_dim": EXPERT_OUT_DIM_ORDER3,
            "router_in_dim": ROUTER_IN_DIM_ORDER3,
        }
        ck_cpu = {k: champ_state[k].detach().cpu() for k in pol.PARAM_KEYS}
        torch.save({**ck_cpu, **meta}, out_dir / f"island_{island_id}_best.pt")

    save_champ(0, seed_res[best_i], best_cal)
    rng = np.random.default_rng(20261183 + island_id * 7919)
    half = (n_groups - 2) // 2

    for it in range(1, max_iters + 1):
        if (time.time() - t_start) / 3600.0 >= hours:
            log_msg(f"⏱️ Достигнут лимит времени {hours:.2f}ч на шаге {it}. Завершение.")
            break
        t_it = time.time()

        # Cross-world migration every 6 iterations
        if it % 6 == 0 and num_islands >= 2:
            nb_p = out_dir / f"island_{(island_id - 1) % num_islands}_best.pt"
            if nb_p.exists():
                try:
                    nb_cpu = pol.load_initial_champion(nb_p)
                    nb_st = _clamp_specialist_constraints({k: nb_cpu[k].to(dev, pol.dtype) for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
                    nb_r = eval_cands([nb_st], val_basket, win_len, slide_b=val_slide_basket, harsh=True)[0]
                    if float(nb_r["geo_score_bp"]) > best_geo + 0.05:
                        champ_state = nb_st
                        best_geo = float(nb_r["geo_score_bp"])
                        c_nb = eval_cands([champ_state], val_basket, win_len, slide_b=val_slide_basket, harsh=False)[0]
                        save_champ(it, nb_r, c_nb)
                        log_msg(f"🧬 [It {it:03d}] Межмировая миграция улучшила лидера -> HarshGeo={best_geo:+.2f}bp (CalibEV={c_nb['clean_ev_pct']:+.3f}%, DD={c_nb['dd_pct']:.2f}%)!")
                except Exception:
                    pass

        # Annealed sigma: P1 explore -> P2 focus -> P3 zoom
        prog = it / max(1.0, float(max_iters))
        sig_sc = 1.15 if prog < 0.45 else (0.65 if prog < 0.80 else 0.35)

        # Sample strictly isolated antithetic perturbations!
        for k in pol.PARAM_KEYS:
            pol.__dict__[k][:] = champ_state[k].unsqueeze(0)

        r_view = pol.ring_mult_half
        if stage in ("mode0_scalp", "mode1_midwick", "mode2_upcarry"):
            t_m = {"mode0_scalp": 0, "mode1_midwick": 1, "mode2_upcarry": 2}[stage]
            eps_w = torch.randn((half, 256, EXPERT_OUT_DIM_ORDER3), device=dev, dtype=torch.float32) * (0.022 * sig_sc * r_view.view(half, 1, 1))
            eps_b = torch.randn((half, 1, EXPERT_OUT_DIM_ORDER3), device=dev, dtype=torch.float32) * (0.085 * sig_sc * r_view.view(half, 1, 1))
            w_all = torch.stack([eps_w, -eps_w], dim=1).reshape(2 * half, 256, EXPERT_OUT_DIM_ORDER3)
            b_all = torch.stack([eps_b, -eps_b], dim=1).reshape(2 * half, 1, EXPERT_OUT_DIM_ORDER3)
            w2_v = pol.w2_moe[2 : 2 + 2 * half].view(2 * half, 256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2_v = pol.b2_moe[2 : 2 + 2 * half].view(2 * half, 1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            w2_v[:, :, t_m, :] = w2_v[:, :, t_m, :] + w_all
            b2_v[:, :, t_m, :] = b2_v[:, :, t_m, :] + b_all
            for gi in range(2, 2 + 2 * half):
                cl = _clamp_specialist_constraints({k: pol.__dict__[k][gi] for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
                pol.w2_moe[gi].copy_(cl["w2_moe"])
                pol.b2_moe[gi].copy_(cl["b2_moe"])
        else:
            # Router-only perturbations (w_gate1, b_gate1, w_gate2, b_gate2)
            ew1 = torch.randn((half, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.028 * sig_sc * r_view.view(half, 1, 1))
            ew1[:, 320:, :] *= 2.80  # strong sensitivity to direct regime skip-sensors!
            eb1 = torch.randn((half, 1, ROUTER_H_DIM_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            ew2 = torch.randn((half, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
            eb2 = torch.randn((half, 1, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.120 * sig_sc * r_view.view(half, 1, 1))
            pol.w_gate1[2 : 2 + 2 * half].add_(torch.stack([ew1, -ew1], dim=1).reshape(2 * half, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3))
            pol.b_gate1[2 : 2 + 2 * half].add_(torch.stack([eb1, -eb1], dim=1).reshape(2 * half, 1, ROUTER_H_DIM_ORDER3))
            pol.w_gate2[2 : 2 + 2 * half].add_(torch.stack([ew2, -ew2], dim=1).reshape(2 * half, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3))
            pol.b_gate2[2 : 2 + 2 * half].add_(torch.stack([eb2, -eb2], dim=1).reshape(2 * half, 1, ROUTER_OUT_DIM_ORDER3))
            pol.w_gate2[2 : 2 + 2 * half, :, 5] = 0.0
            pol.w_gate2[2 : 2 + 2 * half, :, 6] = 0.0
            pol.b_gate2[2 : 2 + 2 * half, 0, 5] = -14.0
            pol.b_gate2[2 : 2 + 2 * half, 0, 6] = -14.0

        # Sample random train basket from regime pool
        tr_ws = rng.choice(tr_pool, size=n_bask, replace=(len(tr_pool) < n_bask))
        tr_bask = [(f"TR{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(tr_ws)]
        pop_cands = pol.get_candidate_states()
        tr_res = eval_cands(pop_cands, tr_bask, win_len, slide_b=None, harsh=True)
        tr_geos = np.array([float(r["geo_score_bp"]) for r in tr_res], dtype=np.float64)

        es_raw = pol.build_openai_es_candidates(champ_state, tr_geos, lr_scales=(0.50, 1.00, 1.65))
        es_cands = [_clamp_specialist_constraints(c, stage, base_ref, pol.dtype, dev) for c in es_raw]
        top_mut_idx = [int(i) for i in np.argsort(tr_geos)[::-1][:4]]
        top_muts = [_clamp_specialist_constraints(pop_cands[i], stage, base_ref, pol.dtype, dev) for i in top_mut_idx]

        shortlist = [champ_state] + es_cands + top_muts
        val_res = eval_cands(shortlist, val_basket, win_len, slide_b=val_slide_basket, harsh=True)
        best_v_idx = int(np.argmax([float(r["geo_score_bp"]) for r in val_res]))
        cand_geo = float(val_res[best_v_idx]["geo_score_bp"])

        if cand_geo > best_geo + 0.02:
            gain = cand_geo - best_geo
            best_geo = cand_geo
            champ_state = {k: v.clone() for k, v in shortlist[best_v_idx].items()}
            cal_r = eval_cands([champ_state], val_basket, win_len, slide_b=val_slide_basket, harsh=False)[0]
            save_champ(it, val_res[best_v_idx], cal_r)
            vr = val_res[best_v_idx]
            log_msg(
                f"⭐ [It {it:03d} | +{gain:.2f}bp] NEW [{stage.upper()}] CHAMP! HarshGeo={best_geo:+.2f}bp "
                f"| CalibEV={cal_r['clean_ev_pct']:+.3f}% (DD={cal_r['dd_pct']:.2f}%) | Fills={vr['fills']:.1f} "
                f"| Reg=[UP:{vr.get('pnl_up_bp',0):+.1f} DN:{vr.get('pnl_dn_bp',0):+.1f} FL:{vr.get('pnl_fl_bp',0):+.1f}bp] "
                f"| Modes={[round(x,2) for x in vr.get('mode_probs',[])]} | {time.time()-t_it:.1f}s"
            )
        elif it % 5 == 0:
            log_msg(f"   [It {it:03d}/{max_iters}] TrainTop={np.max(tr_geos):+.2f}bp | ValChamp={best_geo:+.2f}bp | {time.time()-t_it:.1f}s")

    log_msg(f"🏁 Завершён этап [{stage}] на Island #{island_id} ({world_mode.upper()}) за {(time.time()-t_start)/60:.1f} мин! Best HarshGeo={best_geo:+.2f}bp")


def main():
    p = argparse.ArgumentParser(description="Modular L83 Specialists & L84 Orchestrator Trainer")
    p.add_argument("--stage", type=str, required=True, choices=["mode0_scalp", "mode1_midwick", "mode2_upcarry", "assemble", "router"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--groups", type=int, default=96)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--hours", type=float, default=0.55)
    p.add_argument("--island-id", type=int, default=0)
    p.add_argument("--num-islands", type=int, default=2)
    p.add_argument("--world-mode", type=str, default="dec", choices=["dec", "raw"])
    p.add_argument("--save-subdir", type=str, default="bak_l83a_mode0")
    p.add_argument("--log-file", type=str, default=None)
    args = p.parse_args()

    if args.stage == "assemble":
        run_assemble()
    else:
        run_stage_training(
            stage=args.stage,
            device_str=args.device,
            n_groups=args.groups,
            max_iters=args.iters,
            hours=args.hours,
            island_id=args.island_id,
            num_islands=args.num_islands,
            world_mode=args.world_mode,
            save_subdir=args.save_subdir,
            log_file=args.log_file,
        )


if __name__ == "__main__":
    main()
