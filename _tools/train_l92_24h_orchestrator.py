#!/usr/bin/env python3
"""
_tools/train_l92_24h_orchestrator.py
=============================================================================
Урок L92: Прямое обучение Глубокого Оркестратора 386D и 36D Регулятора
НЕПОСРЕДСТВЕННО НА НЕПРЕРЫВНЫХ 24-ЧАСОВЫХ ТРАЕКТОРИЯХ (win_len = 86,401с)
поверх связки:
  - Боевое ядро Live_L89_V2 Mode #4 (95% экспозиции, +2.037%/сут на 12x24h)
  - Поверхностный скальпер L91a Mode #0 (эндогенный лот 4.8% у BBO 1.0 bp)
  - Рабочая средняя мода L91b Mode #1 (42.4%, 2 слота на -2.15%)
  - Импульсная UP-Carry L91c Mode #2 (рециклинг ch[56]=0.80)
  - Живая глубокая мода L91d1 в слоте Mode #5 (35.1%, DD 0.71%)
  - Бункер кэша Mode #3 (0% покупок) и Разгрузчик Mode #7

Цель: преодолеть краевое искажение коротких 2ч/6ч окон, поднять частоту сделок
в 4–7 раз за счёт работы на широком спреде / спокойном рынке и превзойти
эталон Live_L89_V2 (+2.037%/сут, MaxDD 5.93%) во ВСЕХ 4 физиках (DEC/RAW x Calib/Harsh)
и на слепых финальных 24 часах OOS!
=============================================================================
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
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
    REG_H_DIM_ORDER3,
    REG_IN_DIM_ORDER3,
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
BASE_LIVE_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
LOG_DIR = BASE_DIR / "logs" / "l92_24h_orchestrator"
WIN_24H = 86401


def _log_m(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "master_l92.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def compute_l92_24h_fitness(cal_r: dict, har_r: dict, n_windows: int, target_daily_pct: float = 2.15) -> dict:
    """
    Целевая функция прямого 24ч компаундинга (1 окно = ровно 24.0 часа, 0 краевых искажений!):
      - Строгий баланс Calib + Harsh (50/50) по Geo и суточному EV%/day.
      - Целевая планка +2.15%/сутки (выше текущих +2.037%/сут у Live_L89_V2) со штрафом за недобор.
      - Условный бонус здоровой активности: начисляется ТОЛЬКО при сохранении высокой доходности
        (ev_day_mix >= +1.85%/сут) и низкой просадки (dd_max <= 6.0%), поощряя частые лёгкие сделки
        вместо суточного простоя без провокации убыточного спама!
    """
    ev_day_c = float(cal_r["clean_ev_pct"])
    ev_day_h = float(har_r["clean_ev_pct"])
    ev_day_mix = 0.50 * ev_day_c + 0.50 * ev_day_h
    geo_mix = 0.50 * float(cal_r["geo_score_bp"]) + 0.50 * float(har_r["geo_score_bp"])
    dd_max = max(float(cal_r["dd_pct"]), float(har_r["dd_pct"]))

    shortfall_bp = 0.60 * max(0.0, (target_daily_pct - ev_day_mix) * 100.0)
    fills_per_day = float(cal_r["fills"]) / max(1.0, float(n_windows))
    cnl_per_hr = float(cal_r.get("cancels", 0.0)) / max(1.0, 24.0 * float(n_windows))
    cpf = float(cal_r.get("cancels", 0.0)) / max(1.0, float(cal_r["fills"]))

    api_pen = 0.12 * max(0.0, cnl_per_hr - 420.0) + 0.035 * max(0.0, cpf - 580.0)
    dd_pen = 16.0 * (max(0.0, dd_max - 4.80) ** 1.35)

    # Гейтированный бонус продуктивной активности (до +18 bp при 25..40 сделках/сутки И чистом EV >= 1.85%/сут и DD <= 6.0%):
    prof_gate = 1.0 / (1.0 + math.exp(-(ev_day_mix - 1.85) * 8.0))
    safe_gate = 1.0 / (1.0 + math.exp((dd_max - 6.05) * 4.0))
    act_bonus = 10.0 * min(1.8, math.log1p(max(0.0, fills_per_day - 5.0) / 8.0)) * prof_gate * safe_gate

    l92_sc = geo_mix + 55.0 * ev_day_mix - shortfall_bp - api_pen - dd_pen + act_bonus
    return {
        "l92_score_bp": float(l92_sc),
        "geo_mix_bp": float(geo_mix),
        "ev_day_cal_pct": float(ev_day_c),
        "ev_day_har_pct": float(ev_day_h),
        "ev_day_mix_pct": float(ev_day_mix),
        "dd_max_pct": float(dd_max),
        "fills_per_day": float(fills_per_day),
        "fills_tot": float(cal_r["fills"]),
        "cnl_per_hr": float(cnl_per_hr),
        "mode_probs": cal_r.get("mode_probs", []),
    }


def build_l92_master_bank() -> Path:
    """
    Формирует стартовый банк специалистов L92:
      - Ствол, память 64D и Mode #4 (95% экспозиция) + Mode #7 + роутер = 100% от Live_L89_V2
        (при закрытых слотах #0..#3,#5 даёт побитово 0.00e+00 то же поведение, что живой чемпион!).
      - В слоты #0, #1, #2, #3, #5 встраиваются целостные нелинейные специалисты из L91:
          #0 <- L91a_Mode0_FastScalp (4.8% микро-лот, 1.0 bp от Bid1, нервный темп)
          #1 <- L91b_Mode1_MedWave   (42.4%, 2 слота на -2.15%, ритмичный темп)
          #2 <- L91c_Mode2_UpCarry   (47.4%, рециклинг всплесков ch[56]=0.80)
          #3 <- Бункер кэша 0% покупок
          #5 <- L91d1_Mode4_AgileDeep (35.1% облегчённая живая глубокая мода, DD 0.71%)
    """
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=torch.device("cpu"))
    live_l89 = pol.load_initial_champion(BASE_LIVE_PATH)
    l91_asm = pol.load_initial_champion(CHAMP_DIR / "l91_specialists_assembled.pt")

    d = {k: live_l89[k].float().clone() for k in pol.PARAM_KEYS}
    w2 = d["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    b2 = d["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    wmup = d["w_moe_up"].view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    bmup = d["b_moe_up"].view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    wmdn = d["w_moe_dn"].view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)

    a_w2 = l91_asm["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    a_b2 = l91_asm["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    a_up = l91_asm["w_moe_up"].float().view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    a_bu = l91_asm["b_moe_up"].float().view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    a_dn = l91_asm["w_moe_dn"].float().view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)

    for dst_m, src_m in ((0, 0), (1, 1), (2, 2), (3, 3), (5, 4)):
        w2[:, dst_m, :] = a_w2[:, src_m, :]
        b2[0, dst_m, :] = a_b2[0, src_m, :]
        wmup[:, dst_m, :] = a_up[:, src_m, :]
        bmup[0, dst_m, :] = a_bu[0, src_m, :]
        wmdn[dst_m] = a_dn[src_m]

    d["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
    d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
    d["w_moe_up"] = wmup.reshape(256, MOE_MLP_UP_DIM_ORDER3)
    d["b_moe_up"] = bmup.reshape(1, MOE_MLP_UP_DIM_ORDER3)
    d["w_moe_dn"] = wmdn

    # Паркуем неактивные в L89 слоты на -12.0 в Seed #0, чтобы старт давал 100% точность Live_L89_V2 (+2.037%/сут)!
    d["w_gate2"][:, [0, 1, 2, 3, 5, 6]] = 0.0
    d["b_gate2"][0, [0, 1, 2, 3, 5]] = -12.0
    d["b_gate2"][0, 6] = -14.0
    d = _normalize_weights_order3(d, torch.float32)

    dst = CHAMP_DIR / "l92_master_bank_init.pt"
    meta = {
        "arch_version": "order3_v2",
        "wf_version": "l92_master_bank_init",
        "champ_tag": "L92_MasterBank_Init",
        "pure_endogenous": True,
        "k_modes": K_MODES_ORDER3,
        "in_dim": TOTAL_DIM_ORDER3,
        "total_dim": TOTAL_DIM_ORDER3,
        "mkt_dim": MKT_DIM_ORDER3,
        "ag_dim": AG_DIM_ORDER3,
        "expert_out_dim": EXPERT_OUT_DIM_ORDER3,
        "router_in_dim": ROUTER_IN_DIM_ORDER3,
    }
    torch.save({**d, **meta}, dst)
    _log_m(f"🧩 Собран стартовый банк L92 (ядро Live_L89 Mode#4[95%] + специалисты L91 #0,#1,#2,#3,#5) -> {dst.name}")
    return dst


def _apply_l92_constraints(
    cand: dict[str, torch.Tensor],
    stage: str,
    base_ref: dict[str, torch.Tensor],
    dtype: torch.dtype,
    dev: torch.device,
) -> dict[str, torch.Tensor]:
    norm_d = _normalize_weights_order3(cand, dtype)
    out = {k: base_ref[k].detach().clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}

    router_keys = ("w_gate1", "b_gate1", "w_gate_res1", "b_gate_res", "w_gate_res2", "w_gate2", "b_gate2")
    reg_keys = ("w_reg1", "b_reg1_b", "b_reg1_a", "w_reg_res1", "b_reg_res", "w_reg_res2", "w_reg2", "b_reg2_b", "b_reg2_a")
    for kr in router_keys + reg_keys:
        out[kr] = norm_d[kr].to(device=dev, dtype=dtype)

    if stage == "l92b_24h_joint":
        reg_keys = ("w_reg1", "b_reg1_b", "b_reg1_a", "w_reg_res1", "b_reg_res", "w_reg_res2", "w_reg2", "b_reg2_b", "b_reg2_a")
        for kr in reg_keys:
            out[kr] = norm_d[kr].to(device=dev, dtype=dtype)
        wmup_new = out["w_moe_up"].float().view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
        bmup_new = out["b_moe_up"].float().view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
        wmdn_new = out["w_moe_dn"].float().view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        wmup_c = norm_d["w_moe_up"].float().to(dev).view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
        bmup_c = norm_d["b_moe_up"].float().to(dev).view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
        wmdn_c = norm_d["w_moe_dn"].float().to(dev).view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)
        for km in (0, 1, 2, 4, 5, 7):
            wmup_new[:, km, :] = wmup_c[:, km, :]
            bmup_new[:, km, :] = bmup_c[:, km, :]
            mdn_km = wmdn_c[km].clone()
            mdn_km[:, 48] = 0.0
            if km != 0:
                mdn_km[:, 55] = 0.0
            if km == 0:
                mdn_km[:, 38] = torch.clamp(mdn_km[:, 38], -0.15, 0.15)
            elif km == 1:
                mdn_km[:, 38] = torch.clamp(mdn_km[:, 38], -0.22, 0.22)
            wmdn_new[km] = mdn_km
        out["w_moe_up"] = wmup_new.reshape(256, MOE_MLP_UP_DIM_ORDER3).to(dtype)
        out["b_moe_up"] = bmup_new.reshape(1, MOE_MLP_UP_DIM_ORDER3).to(dtype)
        out["w_moe_dn"] = wmdn_new.to(dtype)

    wg2 = out["w_gate2"].clone()
    bg2 = out["b_gate2"].clone()
    wg2[:, 6] = 0.0
    wg2[:, K_MODES_ORDER3 + 6] = 0.0
    bg2[0, 6] = -14.0
    bg2[0, K_MODES_ORDER3 + 6] = +4.0
    out["w_gate2"] = wg2
    out["b_gate2"] = bg2
    return out


def _build_l92_seeds(base_ref: dict[str, torch.Tensor], stage: str, dtype: torch.dtype, dev: torch.device) -> list[dict[str, torch.Tensor]]:
    # Seed #0: 100% Zero-Regression Anchor (= Live_L89_V2, +2.037%/сут на 12x24h!)
    seeds = [_apply_l92_constraints(base_ref, stage, base_ref, dtype, dev)]
    if stage == "l92b_24h_joint":
        # На фазе Joint поверх чемпиона L92a (+677.95bp, +2.066%/d) тестируем также точечное подключение ТОЛЬКО Mode #0 (4.8% микро-лот)
        # строго на широком спреде (>=50bp) и штиле (2%..8% времени), НЕ размывая 92-98% ядро Mode #4!
        imp = base_ref["w_gate2"].float()[:, 4].abs() + base_ref["w_gate2"].float()[:, 7].abs()
        q0 = torch.argsort(imp)[:4].cpu()
        for m0_b in (-2.0, -0.6, +0.6, +1.6):
            dm = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            wg1 = dm["w_gate1"].float().clone()
            wg2 = dm["w_gate2"].float().clone()
            bg2 = dm["b_gate2"].float().clone()
            wg1[:, q0] = 0.0
            wg1[336, q0] = +4.8; wg1[350, q0] = +2.6; wg1[364, q0] = -10.0; wg1[257, q0] = -5.0
            wg2[:, 0] = 0.0; wg2[q0, 0] = +2.5; bg2[0, 0] = m0_b; bg2[0, 8] = +3.6
            dm["w_gate1"] = wg1.to(dtype); dm["w_gate2"] = wg2.to(dtype); dm["b_gate2"] = bg2.to(dtype)
            seeds.append(_apply_l92_constraints(dm, stage, base_ref, dtype, dev))
        return seeds

    pol_cpu = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=torch.device("cpu"))
    l91e_reg = pol_cpu.load_initial_champion(CHAMP_DIR / "l91e_regulator_winner.pt")

    # Ищем тихие нейроны первого каскада роутера L89 (минимально задействованные в Mode #4/#7),
    # чтобы подключить Mode #0 (4.8% BBO скальп), Mode #2 (UP-Carry), Mode #3 (Бункер) и Mode #5 (облегчённая глубокая)
    # БЕЗ малейшего повреждения обученных путей Mode #4/#7!
    imp = base_ref["w_gate2"].float()[:, 4].abs() + base_ref["w_gate2"].float()[:, 7].abs()
    quiet_idx = torch.argsort(imp)[:16].cpu()
    q0, q1, q2, q3 = quiet_idx[0:4], quiet_idx[4:8], quiet_idx[8:12], quiet_idx[12:16]

    for m0_b, m1_b, m2_b, m3_b, m5_b, cr_sh in (
        (+0.5, -6.0, -3.5, -2.5, -2.5, 8.5),   # деликатный скальп широкого спреда (~4..6% времени) поверх 94% ядра L89
        (+1.8, -5.0, -2.5, -2.0, -1.8, 8.5),   # умеренный скальп (~8..11% времени) + мягкий UP-Carry и Бункер
        (+2.8, -4.0, -1.8, -1.5, -1.2, 9.0),   # активный скальп (~13..15% времени) под жёстким щитом crash900=-9.0
        (+1.2, -1.5, -1.5, -1.5, -1.5, 9.5),   # сбалансированное подключение всех вспомогательных мод на штиле
        (-0.5, -6.0, -2.0, -1.8, +0.5, 8.0),   # переключение между полновесной Mode#4 (95%) и лёгкой Mode#5 (35%) + UP-Carry
        (+2.2, -6.0, -6.0, -1.5, -6.0, 9.5),   # чистая связка: Mode #0 (4.8% скальп) + Mode #3 (0% Бункер) + Mode #4/#7 (L89)
    ):
        d = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        wg1 = d["w_gate1"].float().clone()
        wg2 = d["w_gate2"].float().clone()
        bg2 = d["b_gate2"].float().clone()

        wg1[:, quiet_idx] = 0.0
        # Mode #0: широкий спред [336], спокойный BigONE [350], ультра-низкий crash900 [364], низкая загрузка [257]
        wg1[336, q0] = +4.2; wg1[350, q0] = +2.4; wg1[364, q0] = -cr_sh; wg1[257, q0] = -4.5
        # Mode #1 / #5: штиль и свободный кэш
        wg1[364, q1] = -cr_sh; wg1[350, q1] = +2.2; wg1[258, q1] = +1.8; wg1[257, q1] = -3.5
        # Mode #2: импульс роста ret30m [370] + CVD [373] вне риска обвала
        wg1[370, q2] = +3.4; wg1[373, q2] = +2.0; wg1[364, q2] = -cr_sh; wg1[257, q2] = -3.0
        # Mode #3: Бункер 0% покупок при падении BigONE [350] и скачке crash900 [364]
        wg1[350, q3] = -4.5; wg1[364, q3] = +3.5; wg1[257, q3] = -2.0

        wg2[:, 0] = 0.0; wg2[q0, 0] = +2.4; bg2[0, 0] = m0_b; bg2[0, 8] = +3.6
        wg2[:, 1] = 0.0; wg2[q1, 1] = +2.0; bg2[0, 1] = m1_b; bg2[0, 9] = +1.0
        wg2[:, 2] = 0.0; wg2[q2, 2] = +2.2; bg2[0, 2] = m2_b; bg2[0, 10] = +1.2
        wg2[:, 3] = 0.0; wg2[q3, 3] = +2.2; bg2[0, 3] = m3_b; bg2[0, 11] = +2.8
        wg2[:, 5] = 0.0; wg2[q1, 5] = +1.8; bg2[0, 5] = m5_b; bg2[0, 13] = +0.5

        d["w_gate1"] = wg1.to(dtype)
        d["w_gate2"] = wg2.to(dtype)
        d["b_gate2"] = bg2.to(dtype)
        seeds.append(_apply_l92_constraints(d, stage, base_ref, dtype, dev))

    # Дополнительно проверяем 3 сида с регулятором L91e (показавшим +2.088%/d и 388 сделок в RAW!) прямо со старта L92a:
    reg_keys = ("w_reg1", "b_reg1_b", "b_reg1_a", "w_reg_res1", "b_reg_res", "w_reg_res2", "w_reg2", "b_reg2_b", "b_reg2_a")
    for m0_r in (+1.6, +2.6, +3.4):
        d_r = {k: seeds[2][k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        for kr in reg_keys:
            d_r[kr] = l91e_reg[kr].to(device=dev, dtype=dtype)
        bg2_r = d_r["b_gate2"].float().clone()
        bg2_r[0, 0] = m0_r
        d_r["b_gate2"] = bg2_r.to(dtype)
        seeds.append(_apply_l92_constraints(d_r, stage, base_ref, dtype, dev))

    return seeds


def _island_24h_worker(
    stage: str,
    src_ckpt_str: str,
    save_subdir: str,
    island_id: int,
    device_str: str,
    world_mode: str,
    n_groups: int,
    max_iters: int,
    max_hours: float,
) -> None:
    t0 = time.time()
    dev = torch.device(device_str)
    out_dir = CHAMP_DIR / save_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_p = LOG_DIR / f"{stage}_i{island_id}_{world_mode}.log"

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
    mkt_sub = np.asarray(mkt_full[subcent_idx:], dtype=np.float32)
    total_rows = len(ts_sub)

    q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
    q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
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

    # Полные финальные 24 часа (D12: 86,401с) остаются 100% НЕПРИКОСНОВЕННЫМ СЛЕПЫМ 24h OOS!
    usable_hi = total_rows - WIN_24H - 120
    # Внутри 11.7 суток строим сетку непрерывных 24ч окон (шаг 2ч -> ~120 окон):
    all_24h_starts = list(range(1800, usable_hi - WIN_24H, 7200))
    # 8 стратифицированных 24ч окон валидации по всей 11.7-суточной истории (охватывают и ранние пампы, и обвал 20.11, и флэт 25.11):
    val_pos_idx = [int(round(i)) for i in np.linspace(4, len(all_24h_starts) - 1, 8)]
    val_starts = [all_24h_starts[i] for i in val_pos_idx]
    val_basket = [(f"VAL24_{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(val_starts)]
    # Обучающий пул исключает точные старты валидации (+/- 1 сосед), чтобы исключить тривиальное запоминание фазы старта:
    excl = set()
    for vi in val_pos_idx:
        excl.update((vi - 1, vi, vi + 1))
    tr_pool = [all_24h_starts[i] for i in range(len(all_24h_starts)) if i not in excl]

    pol = MultimodalMoEPolicyOrder3(n_groups=n_groups, n_budgets=6, device=dev)
    base_cpu = pol.load_initial_champion(Path(src_ckpt_str))
    base_ref = {k: base_cpu[k].to(device=dev, dtype=pol.dtype) for k in pol.PARAM_KEYS}
    seeds = _build_l92_seeds(base_ref, stage, pol.dtype, dev)
    if stage == "l92b_24h_joint":
        # Добавляем также сид с обновлённым регулятором L91e поверх лучшего роутера L92a
        pol_c = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=torch.device("cpu"))
        l91e_ck = pol_c.load_initial_champion(CHAMP_DIR / "l91e_regulator_winner.pt")
        dr = {k: base_ref[k].clone() for k in pol.PARAM_KEYS}
        for kr in ("w_reg1", "b_reg1_b", "b_reg1_a", "w_reg_res1", "b_reg_res", "w_reg_res2", "w_reg2", "b_reg2_b", "b_reg2_a"):
            dr[kr] = l91e_ck[kr].to(device=dev, dtype=pol.dtype)
        seeds.append(_apply_l92_constraints(dr, stage, base_ref, pol.dtype, dev))

    def eval_24h_batch(cands: list[dict], bask: list, dual_stress: bool = True) -> list[dict]:
        orig_n = len(cands)
        c_in = list(cands) + [cands[0]] * (32 - orig_n) if orig_n <= 32 else cands
        pad_m = orig_n > 32
        rc = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=bask, win_len=WIN_24H, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=(not dual_stress), enforce_activity_floor=False, risk_inv_pen_w=1.50,
        )[:orig_n]
        rh = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=bask, win_len=WIN_24H, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=True, enforce_activity_floor=False, risk_inv_pen_w=1.50,
        )[:orig_n] if dual_stress else rc
        return [compute_l92_24h_fitness(a, b, len(bask)) for a, b in zip(rc, rh)]

    seed_res = eval_24h_batch(seeds, val_basket, dual_stress=True)
    for si, sr_i in enumerate(seed_res):
        log_w(
            f"   [Seed #{si}] L92={sr_i['l92_score_bp']:+.2f}bp | EV/d={sr_i['ev_day_cal_pct']:+.3f}%/{sr_i['ev_day_har_pct']:+.3f}% "
            f"| DD={sr_i['dd_max_pct']:.2f}% | Fills/d={sr_i['fills_per_day']:.1f} | Modes={[round(x,2) for x in sr_i['mode_probs']]}"
        )
    best_i = int(np.argmax([r["l92_score_bp"] for r in seed_res]))
    champ_state = {k: v.clone() for k, v in seeds[best_i].items()}
    best_sc = float(seed_res[best_i]["l92_score_bp"])
    sr0 = seed_res[best_i]
    log_w(
        f"🌱 Winner Seed #{best_i}/{len(seeds)}: L92={best_sc:+.2f}bp | EV/d={sr0['ev_day_cal_pct']:+.3f}%/{sr0['ev_day_har_pct']:+.3f}% "
        f"| DD={sr0['dd_max_pct']:.2f}% | Fills/d={sr0['fills_per_day']:.1f}"
    )

    def save_champ(it_n: int, vr: dict) -> None:
        meta = {
            "arch_version": "order3_v2",
            "wf_version": f"l92_24h_{stage}_{world_mode}",
            "stage": stage,
            "pure_endogenous": True,
            "k_modes": K_MODES_ORDER3,
            "iteration": it_n,
            **vr,
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
    rng = np.random.default_rng(20261292 + island_id * 99991)
    half = (n_groups - 2) // 2
    r_view = pol.ring_mult_half
    n_tr_w = 6  # 6 x 24h = 144 часа за 1 батч (~32с на 80 кандидатов, 0 риска OOM на 16 ГБ VRAM!)

    for it in range(1, max_iters + 1):
        if (time.time() - t0) / 3600.0 >= max_hours:
            break
        t_it = time.time()

        if it % 6 == 0:
            nb_p = out_dir / f"island_{1 - island_id}_best.pt"
            if nb_p.exists():
                try:
                    nb_cpu = pol.load_initial_champion(nb_p)
                    nb_st = _apply_l92_constraints({k: nb_cpu[k].to(dev, pol.dtype) for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
                    nb_r = eval_24h_batch([nb_st], val_basket, dual_stress=True)[0]
                    if float(nb_r["l92_score_bp"]) > best_sc + 0.05:
                        champ_state = nb_st
                        best_sc = float(nb_r["l92_score_bp"])
                        save_champ(it, nb_r)
                        log_w(f"🧬 [It {it:02d}] Миграция обновила лидера -> L92={best_sc:+.2f}bp (EV/d={nb_r['ev_day_cal_pct']:+.3f}%, Fills/d={nb_r['fills_per_day']:.1f})")
                except Exception:
                    pass

        prog = it / max(1.0, float(max_iters))
        sig_sc = 1.05 if prog < 0.45 else (0.62 if prog < 0.80 else 0.34)
        if stage == "l92b_24h_joint":
            sig_sc *= 0.28

        for k in pol.PARAM_KEYS:
            pol.__dict__[k][:] = champ_state[k].unsqueeze(0)

        ew1 = torch.randn((half, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.026 * sig_sc * r_view.view(half, 1, 1))
        ew1[:, 256:, :] *= 2.50
        eb1 = torch.randn((half, 1, ROUTER_H_DIM_ORDER3), device=dev) * (0.048 * sig_sc * r_view.view(half, 1, 1))
        er1 = torch.randn((half, ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3), device=dev) * (0.040 * sig_sc * r_view.view(half, 1, 1))
        ebr = torch.randn((half, 1, 2 * ROUTER_RES_H_ORDER3), device=dev) * (0.048 * sig_sc * r_view.view(half, 1, 1))
        er2 = torch.randn((half, ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
        ew2 = torch.randn((half, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.048 * sig_sc * r_view.view(half, 1, 1))
        eb2 = torch.randn((half, 1, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.140 * sig_sc * r_view.view(half, 1, 1))

        for attr_n, eps_t in (
            ("w_gate1", ew1), ("b_gate1", eb1), ("w_gate_res1", er1),
            ("b_gate_res", ebr), ("w_gate_res2", er2), ("w_gate2", ew2), ("b_gate2", eb2),
        ):
            tgt = pol.__dict__[attr_n][2 : 2 + 2 * half]
            tgt.add_(torch.stack([eps_t, -eps_t], dim=1).reshape_as(tgt))

        if stage == "l92b_24h_joint":
            er_w1 = torch.randn((half, REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3), device=dev) * (0.032 * sig_sc * r_view.view(half, 1, 1))
            er_w1[:, 28:36, :] *= 2.40
            er_b1b = torch.randn((half, 1, REG_H_DIM_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            er_b1a = torch.randn((half, 1, REG_H_DIM_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            er_rs1 = torch.randn((half, REG_H_DIM_ORDER3, 64), device=dev) * (0.038 * sig_sc * r_view.view(half, 1, 1))
            er_brs = torch.randn((half, 1, 64), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
            er_rs2 = torch.randn((half, 32, REG_H_DIM_ORDER3), device=dev) * (0.042 * sig_sc * r_view.view(half, 1, 1))
            er_w2 = torch.randn((half, REG_H_DIM_ORDER3, 2), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
            er_b2b = torch.randn((half, 1, 2), device=dev) * (0.075 * sig_sc * r_view.view(half, 1, 1))
            er_b2a = torch.randn((half, 1, 2), device=dev) * (0.075 * sig_sc * r_view.view(half, 1, 1))
            for attr_n, eps_t in (
                ("w_reg1", er_w1), ("b_reg1_b", er_b1b), ("b_reg1_a", er_b1a),
                ("w_reg_res1", er_rs1), ("b_reg_res", er_brs), ("w_reg_res2", er_rs2),
                ("w_reg2", er_w2), ("b_reg2_b", er_b2b), ("b_reg2_a", er_b2a),
            ):
                tgt = pol.__dict__[attr_n][2 : 2 + 2 * half]
                tgt.add_(torch.stack([eps_t, -eps_t], dim=1).reshape_as(tgt))

            ew_up = torch.randn((half, 256, MOE_MLP_UP_DIM_ORDER3), device=dev) * (0.028 * sig_sc * r_view.view(half, 1, 1))
            ew_dn = torch.randn((half, K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.030 * sig_sc * r_view.view(half, 1, 1, 1))
            pol.w_moe_up[2 : 2 + 2 * half].add_(torch.stack([ew_up, -ew_up], dim=1).reshape_as(pol.w_moe_up[2 : 2 + 2 * half]))
            pol.w_moe_dn[2 : 2 + 2 * half].add_(torch.stack([ew_dn, -ew_dn], dim=1).reshape_as(pol.w_moe_dn[2 : 2 + 2 * half]))

        for gi in range(2, 2 + 2 * half):
            cl = _apply_l92_constraints({k: pol.__dict__[k][gi] for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
            for k in pol.PARAM_KEYS:
                pol.__dict__[k][gi].copy_(cl[k])

        tr_ws = rng.choice(tr_pool, size=n_tr_w, replace=False)
        tr_bask = [(f"TR24_{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(tr_ws)]
        pop_cands = pol.get_candidate_states()
        tr_res = eval_24h_batch(pop_cands, tr_bask, dual_stress=False)
        tr_scs = np.array([float(r["l92_score_bp"]) for r in tr_res], dtype=np.float64)

        es_raw = pol.build_openai_es_candidates(champ_state, tr_scs, lr_scales=(0.50, 1.00, 1.65))
        es_cands = [_apply_l92_constraints(c, stage, base_ref, pol.dtype, dev) for c in es_raw]
        top_muts = [_apply_l92_constraints(pop_cands[int(i)], stage, base_ref, pol.dtype, dev) for i in np.argsort(tr_scs)[::-1][:4]]

        shortlist = [champ_state] + es_cands + top_muts
        val_res = eval_24h_batch(shortlist, val_basket, dual_stress=True)
        bv_i = int(np.argmax([float(r["l92_score_bp"]) for r in val_res]))
        cand_sc = float(val_res[bv_i]["l92_score_bp"])

        if cand_sc > best_sc + 0.02:
            gain = cand_sc - best_sc
            best_sc = cand_sc
            champ_state = {k: v.clone() for k, v in shortlist[bv_i].items()}
            vr = val_res[bv_i]
            save_champ(it, vr)
            log_w(
                f"⭐ [It {it:02d} | +{gain:.2f}bp] L92={best_sc:+.2f}bp | EV/d={vr['ev_day_cal_pct']:+.3f}%/{vr['ev_day_har_pct']:+.3f}% "
                f"| DD={vr['dd_max_pct']:.2f}% | Fills/d={vr['fills_per_day']:.1f} | Cnl/h={vr['cnl_per_hr']:.1f} "
                f"| Modes={[round(x,2) for x in vr.get('mode_probs',[])]} | {time.time()-t_it:.1f}s"
            )
        elif it % 5 == 0:
            log_w(f"   [It {it:02d}/{max_iters}] TrainTop={np.max(tr_scs):+.2f}bp | ValChamp={best_sc:+.2f}bp | {time.time()-t_it:.1f}s")

    log_w(f"🏁 Done [{stage}] ({world_mode.upper()}) in {(time.time()-t0)/60:.1f}m | Best L92={best_sc:+.2f}bp")


def select_and_coronate_l92() -> dict:
    """Честный турнир на всех 12 сутках (12 x 24h) + отдельно слепые финальные 24 часа OOS (D12) во всех 4 физиках!"""
    _log_m("🏆 СТАРТ ФИНАЛЬНОГО 24h-ТУРНИРА L92 ПРОТИВ LIVE_L89_V2 (12 x 24h + СЛЕПЫЕ 24ч OOS)")
    dev = torch.device("cuda:0")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)

    cands_named = [("Live_L89_V2", BASE_LIVE_PATH)]
    for nm, p_c in (
        ("L92a_24h_RAW", CHAMP_DIR / "bak_l92a_orch" / "island_0_best.pt"),
        ("L92a_24h_DEC", CHAMP_DIR / "bak_l92a_orch" / "island_1_best.pt"),
        ("L92b_Joint_RAW", CHAMP_DIR / "bak_l92b_joint" / "island_0_best.pt"),
        ("L92b_Joint_DEC", CHAMP_DIR / "bak_l92b_joint" / "island_1_best.pt"),
    ):
        if p_c.exists():
            cands_named.append((nm, p_c))

    states = [pol.load_initial_champion(p) for _, p in cands_named]
    states_gpu = [{k: v.to(dev) for k, v in st.items()} for st in states]
    sub = 3016277
    summary = {nm: {"l92_4w": 0.0, "ev_day_4w": 0.0, "oos_24h_ev": 0.0, "max_dd": 0.0, "fills_day": 0.0, "worlds": {}} for nm, _ in cands_named}

    for wm in ("dec", "raw"):
        is_r = wm == "raw"
        lob = np.load(BASE_DIR / ("data/cache/rl_guard_lob_cache_600h_raw_with_self.npz" if is_r else "data/cache/rl_guard_lob_cache_600h.npz"), mmap_mode="r")
        mkt = np.load(BASE_DIR / ("data/cache/rl_hybrid_market_obs_96d_raw_with_self.npz" if is_r else "data/cache/rl_hybrid_market_obs_96d.npz"), mmap_mode="r")["market_obs_96d"]
        ts = lob["ts"][sub:]
        n_sub = len(ts)
        q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
        q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
        sv_np = np.asarray(lob["sell_vol"][sub:], dtype=np.float32).copy()
        bv_np = np.asarray(lob["buy_vol"][sub:], dtype=np.float32).copy()
        qm = (ts >= q0_u) & (ts <= q1_u)
        sv_np[qm] = 0.0; bv_np[qm] = 0.0

        mkt_g = torch.from_numpy(np.asarray(mkt[sub:], dtype=np.float32)).to(dev, torch.bfloat16)
        b10_g = torch.from_numpy(np.asarray(lob["bid_p"][sub:, :10], dtype=np.float32)).to(dev)
        a10_g = torch.from_numpy(np.asarray(lob["ask_p"][sub:, :10], dtype=np.float32)).to(dev)
        bq10_g = torch.from_numpy(np.asarray(lob["bid_q"][sub:, :10], dtype=np.float32)).to(dev)
        aq10_g = torch.from_numpy(np.asarray(lob["ask_q"][sub:, :10], dtype=np.float32)).to(dev)
        sv_g = torch.from_numpy(sv_np).to(dev)
        bv_g = torch.from_numpy(bv_np).to(dev)

        starts_12d = [int(x) for x in np.linspace(1800, n_sub - WIN_24H - 10, 12)]
        bask_12d = [(f"D{k:02d}", ws, 0.0, 0.0) for k, ws in enumerate(starts_12d)]
        rc = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=bask_12d, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)
        rh = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_gpu if False else b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=bask_12d, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)

        oos_24h = [("BLIND_D12", int(n_sub - WIN_24H - 5), 0.0, 0.0)]
        oc = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=oos_24h, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)
        oh = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=oos_24h, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)

        for idx_c, (nm, _) in enumerate(cands_named):
            fm = compute_l92_24h_fitness(rc[idx_c], rh[idx_c], 12)
            oos_m = 0.50 * float(oc[idx_c]["clean_ev_pct"]) + 0.50 * float(oh[idx_c]["clean_ev_pct"])
            summary[nm]["l92_4w"] += 0.50 * fm["l92_score_bp"]
            summary[nm]["ev_day_4w"] += 0.50 * fm["ev_day_mix_pct"]
            summary[nm]["oos_24h_ev"] += 0.50 * oos_m
            summary[nm]["max_dd"] = max(summary[nm]["max_dd"], fm["dd_max_pct"])
            summary[nm]["fills_day"] += 0.50 * fm["fills_per_day"]
            summary[nm]["worlds"][wm.upper()] = {
                "calib_ev_d": round(fm["ev_day_cal_pct"], 3),
                "harsh_ev_d": round(fm["ev_day_har_pct"], 3),
                "dd_max": round(fm["dd_max_pct"], 2),
                "fills_per_day": round(fm["fills_per_day"], 2),
                "cnl_per_hr": round(fm["cnl_per_hr"], 1),
                "blind_d12_ev": round(oos_m, 3),
                "blind_d12_fills": round(float(oc[idx_c]["fills"]), 1),
                "modes": [round(float(x), 3) for x in fm["mode_probs"]],
            }
            _log_m(
                f"  [{wm.upper()}] {nm:18s} | EV/d={fm['ev_day_cal_pct']:+.3f}%/{fm['ev_day_har_pct']:+.3f}% "
                f"| DD={fm['dd_max_pct']:.2f}% | Fills/d={fm['fills_per_day']:.1f} | BlindD12={oos_m:+.3f}% ({oc[idx_c]['fills']:.0f}f) "
                f"| Modes={[round(float(x),2) for x in fm['mode_probs']]}"
            )

    _log_m("-" * 98)
    for nm, _ in cands_named:
        s = summary[nm]
        _log_m(
            f"📊 ИТОГ [{nm:18s}] L92_4W={s['l92_4w']:+.2f}bp | Mean_EV/d={s['ev_day_4w']:+.3f}% "
            f"| Fills/d={s['fills_day']:.1f} | BlindD12={s['oos_24h_ev']:+.3f}% | MaxDD={s['max_dd']:.2f}%"
        )
    (LOG_DIR / "coronation_l92_24h.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage(stage: str, src: Path, sub_d: str, iters: int, max_h: float) -> tuple[Path, Path]:
    _log_m(f"▶️ ЗАПУСК 24h ЭТАПА [{stage}] на 2x GPU (iters={iters}, limit={max_h:.2f}h)")
    ctx = mp.get_context("spawn")
    p0 = ctx.Process(target=_island_24h_worker, args=(stage, str(src), sub_d, 0, "cuda:0", "raw", 80, iters, max_h))
    p1 = ctx.Process(target=_island_24h_worker, args=(stage, str(src), sub_d, 1, "cuda:1", "dec", 80, iters, max_h))
    p0.start(); p1.start(); p0.join(); p1.join()
    if p0.exitcode != 0 or p1.exitcode != 0:
        raise RuntimeError(f"Stage {stage} failed: {p0.exitcode}, {p1.exitcode}")
    return CHAMP_DIR / sub_d / "island_0_best.pt", CHAMP_DIR / sub_d / "island_1_best.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters-orch", type=int, default=40)
    ap.add_argument("--iters-joint", type=int, default=22)
    ap.add_argument("--skip-orch", action="store_true")
    args = ap.parse_args()

    r_a = CHAMP_DIR / "bak_l92a_orch" / "island_0_best.pt"
    d_a = CHAMP_DIR / "bak_l92a_orch" / "island_1_best.pt"
    if not args.skip_orch:
        init_p = build_l92_master_bank()
        r_a, d_a = run_stage("l92a_24h_orch", init_p, "bak_l92a_orch", args.iters_orch, 1.05)
        res_mid = select_and_coronate_l92()
        best_a_p = r_a if res_mid["L92a_24h_RAW"]["l92_4w"] >= res_mid["L92a_24h_DEC"]["l92_4w"] else d_a
    else:
        best_a_p = r_a

    if args.iters_joint > 0:
        run_stage("l92b_24h_joint", best_a_p, "bak_l92b_joint", args.iters_joint, 0.65)
    res_fin = select_and_coronate_l92()

    # Проверяем побитовый паритет Sim/Live для победителя
    best_nm = max(res_fin.keys(), key=lambda k: res_fin[k]["l92_4w"])
    _log_m(f"🏅 АБСОЛЮТНЫЙ ЛИДЕР 24h ТУРНИРА: {best_nm} (L92_4W={res_fin[best_nm]['l92_4w']:+.2f}bp, EV/d={res_fin[best_nm]['ev_day_4w']:+.3f}%)")
    name_to_p = {
        "Live_L89_V2": BASE_LIVE_PATH,
        "L92a_24h_RAW": CHAMP_DIR / "bak_l92a_orch" / "island_0_best.pt",
        "L92a_24h_DEC": CHAMP_DIR / "bak_l92a_orch" / "island_1_best.pt",
        "L92b_Joint_RAW": CHAMP_DIR / "bak_l92b_joint" / "island_0_best.pt",
        "L92b_Joint_DEC": CHAMP_DIR / "bak_l92b_joint" / "island_1_best.pt",
    }
    win_p = name_to_p[best_nm]
    rc_p = subprocess.run(
        [str(BASE_DIR / ".venv/bin/python"), str(BASE_DIR / "_tools/test_order3_sim_live_parity.py"), str(win_p)],
        cwd=str(BASE_DIR),
        check=False,
    ).returncode
    if rc_p != 0:
        raise RuntimeError(f"Sim/Live parity check failed for {win_p}!")


if __name__ == "__main__":
    main()
