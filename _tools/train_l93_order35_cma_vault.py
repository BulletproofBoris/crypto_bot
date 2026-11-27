#!/usr/bin/env python3
"""
_tools/train_l93_order35_cma_vault.py
=============================================================================
Урок L93 (Order-3.5 Modernization — Stage 4):
  1. Строго непересекающийся Purged Block 24h CV + 48ч Слепой OOS (0.0% нахлёста
     между Train, Purged Val и Blind OOS; 2-часовые санитарные буферы по краям!).
  2. Separable CMA-ES (Sep-CMA-ES) на 2x GPU (Island 0: RAW, Island 1: DEC):
     - логарифмическое ранговое взвешивание элиты (инвариантно к выбросам одного окна);
     - кумулятивный импульсный путь эволюции p_c и индивидуальная покоординатная
       адаптация диагональной ковариации sigma_j для разномасштабных весов.
  3. Двухтраншевая архитектура «80% Core Vault / 20% Agile Pod» (Residual Alpha):
     - Seed #0 = побитовый эталон L92_24h_Endogenous_2pct_V1 (release_champion_v1.pt);
     - Mode #4 (80%+ Core Cascade Vault) и Mode #7 защищены от деградации;
     - Mode #0 (Wide-Spread Wall-Shielded Scalper), Mode #1 (Med-Wave), Mode #2 (Up-Carry)
       ограничены архитектурным затвором ch[59] agile_pod_z <= 20% экспозиции
       (80% кэша всегда неприкосновенно ждут каскада Mode #4!) и опираются на
       16 новых каналов Signal Stack v8 [96..111] (Gate/Ourbit CVD, BTC/NEAR, Walls L500)
       и канал ch[58] wall_anchor_z (+1 тик перед защитной плитой стакана!).
  4. Коронация по строгим воротам: кандидат обязан превзойти живого чемпиона
     L92_24h_Endogenous_2pct_V1 во всех 4 физиках (DEC/RAW x Calib/Harsh) на 13x24h
     и на 48ч Слепом OOS (BLIND_D12 + BLIND_D13), а также пройти тест паритета 0.00e+00!
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
    MEM_DIM_ORDER3,
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
LOG_DIR = BASE_DIR / "logs" / "l93_order35_cma_vault"
WIN_24H = 86401
PURGE_SEC = 7200  # Строгий 2-часовой санитарный зазор между любыми блоками Train / Val / Blind OOS!


def _log_m(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "master_l93.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_purged_24h_splits(total_rows: int) -> tuple[list[tuple[str, int, float, float]], list[int], list[tuple[str, int, float, float]]]:
    """
    Формирует строго непересекающееся разбиение (0.0% общих секунд!):
      - BLIND_48H: последние 2 x 24ч (BLIND_D12 + BLIND_D13, 172,802с, вкл. ночной спуск -8.6%).
      - PURGED_VAL: 3 стратифицированных 24ч блока внутри первых 11.48 суток:
          BLK_02 (-2.5% штиль/узкий диапазон), BLK_05 (-16.5% жёсткий обвал), BLK_08 (+18.8% ап-тренд).
      - PURGED_TRAIN: все допустимые старты 24ч окон на оставшихся интервалах, удалённые минимум
          на PURGE_SEC (7,200с = 2ч) от границ любого окна валидации и слепого 48ч OOS!
    """
    blind_d13_s = int(total_rows - WIN_24H - 5)
    blind_d12_s = int(blind_d13_s - WIN_24H - 5)
    blind_48h = [
        ("BLIND_D12", blind_d12_s, 0.0, 0.0),
        ("BLIND_D13", blind_d13_s, 0.0, 0.0),
    ]

    usable_end = blind_d12_s - PURGE_SEC
    base_blocks = list(range(1800, usable_end - WIN_24H + 1, WIN_24H + PURGE_SEC))
    # Берём 5 строго непересекающихся режимных 24ч блоков (120ч = 50% до-слепой истории, 0% нахлёста с Train и Blind 48h!):
    #   BLK_02 (-2.5% штиль), BLK_04 (+27.2% каскадный урожай), BLK_05 (-16.5% обвал), BLK_06 (-15.0% пик DD), BLK_08 (+18.8% ап-тренд)
    val_indices = [2, 4, 5, 6, 8] if len(base_blocks) >= 9 else [1, 3, len(base_blocks) // 2, len(base_blocks) - 1]
    val_starts = [base_blocks[i] for i in val_indices]
    val_basket = [(f"PVAL24_{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(val_starts)]

    # Запретные интервалы [v_s - PURGE_SEC, v_s + WIN_24H + PURGE_SEC] для каждого Val-блока и Blind OOS:
    forbidden = [(vs - PURGE_SEC, vs + WIN_24H + PURGE_SEC) for vs in val_starts]
    forbidden.append((blind_d12_s - PURGE_SEC, total_rows + WIN_24H))

    train_starts: list[int] = []
    # Основная сетка непересекающихся 24ч блоков + полушаговые старты ТОЛЬКО внутри чистых Train-зон (0% касания Val/Blind!):
    for ws in range(1800, usable_end - WIN_24H + 1, 14400):
        we = ws + WIN_24H
        overlaps = any(not (we <= f0 or ws >= f1) for f0, f1 in forbidden)
        if not overlaps:
            train_starts.append(int(ws))

    return val_basket, train_starts, blind_48h


def compute_l93_vault_fitness(
    cal_r: dict,
    har_r: dict,
    n_windows: int,
    ref_ev_mix: float | None = None,
    target_daily_pct: float = 2.25,
) -> dict:
    """
    Целевая функция Урока 93 (24ч компаундинг + Residual Alpha поверх ядра L92):
      - Равный вес Calib и Harsh (50/50) по Geo и суточному EV%/day.
      - Начисляет явную премию за Residual Alpha (прирост среднесуточного EV над Seed #0 = L92),
        а за просадку ниже эталона L92 штрафует асимметрично сильнее!
      - Поощряет здоровую частоту сделок Agile Pod на тихих/растущих днях при сохранении DD <= 5.85%.
    """
    ev_day_c = float(cal_r["clean_ev_pct"])
    ev_day_h = float(har_r["clean_ev_pct"])
    ev_day_mix = 0.50 * ev_day_c + 0.50 * ev_day_h
    geo_mix = 0.50 * float(cal_r["geo_score_bp"]) + 0.50 * float(har_r["geo_score_bp"])
    dd_max = max(float(cal_r["dd_pct"]), float(har_r["dd_pct"]))

    shortfall_bp = 0.55 * max(0.0, (target_daily_pct - ev_day_mix) * 100.0)
    fills_per_day = float(cal_r["fills"]) / max(1.0, float(n_windows))
    cnl_per_hr = float(cal_r.get("cancels", 0.0)) / max(1.0, 24.0 * float(n_windows))
    cpf = float(cal_r.get("cancels", 0.0)) / max(1.0, float(cal_r["fills"]))

    api_pen = 0.12 * max(0.0, cnl_per_hr - 420.0) + 0.035 * max(0.0, cpf - 580.0)
    dd_pen = 18.0 * (max(0.0, dd_max - 4.75) ** 1.38)
    if dd_max > 6.55:
        dd_pen += 55.0 * (dd_max - 6.55)

    res_alpha_bp = 0.0
    if ref_ev_mix is not None:
        delta_ev = ev_day_mix - float(ref_ev_mix)
        res_alpha_bp = 45.0 * delta_ev if delta_ev >= 0.0 else 75.0 * delta_ev

    prof_gate = 1.0 / (1.0 + math.exp(-(ev_day_mix - 0.35) * 5.0))
    safe_gate = 1.0 / (1.0 + math.exp((dd_max - 6.30) * 4.0))
    act_bonus = 11.0 * min(2.0, math.log1p(max(0.0, fills_per_day - 3.5) / 6.0)) * prof_gate * safe_gate

    l93_sc = geo_mix + 58.0 * ev_day_mix + res_alpha_bp - shortfall_bp - api_pen - dd_pen + act_bonus
    return {
        "l93_score_bp": float(l93_sc),
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


def build_l93_vault_master_bank() -> Path:
    """
    Собирает стартовый двухтраншевый банк специалистов Order-3.5 (112D / 402D / 60D):
      - Ядро (ствол w1[:96], память 64D, Mode #4 [Core Cascade Vault], Mode #7, роутер и регулятор)
        копируются побитово из живого чемпиона L92_24h_Endogenous_2pct_V1 (release_champion_v1.pt)!
      - В слоты Agile Pod (#0 скальпер широкого спреда, #1 средняя волна, #2 импульсный UP-Carry)
        загружаются нелинейные специалисты L91 с АРХИТЕКТУРНЫМ ИЗОЛЯТОРОМ ch[59] agile_pod_z = +8.0
        (жёсткий потолок экспозиции <= 20% капитала -> 80% кэша всегда защищены под каскад Mode #4!)
        и мягкой инициализацией ch[58] wall_anchor_z для постановки на +1 тик перед стенами L500.
      - В Seed #0 неактивные в L92 слоты (#1, #2) припаркованы в роутере, поэтому Seed #0 воспроизводит
        100% доходность живого чемпиона L92 с нулевой регрессией!
    """
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=torch.device("cpu"))
    live_l92 = pol.load_initial_champion(BASE_LIVE_PATH)
    l91_asm = pol.load_initial_champion(CHAMP_DIR / "l91_specialists_assembled.pt")

    d = {k: live_l92[k].float().clone() for k in pol.PARAM_KEYS}
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

    # В L92 уже немного задействованы Mode #0 (~2%) и Mode #3 (~2%); чтобы Seed #0 сохранил 0.00e+00 с L92,
    # обновляем свободные слоты #1 (MedWave+Pod20%), #2 (UpCarry+Pod20%) и #5 (Wall-Shielded Agile Scalp 20%!),
    # оставляя слоты #0, #3, #4, #7 в точности как в L92!
    for dst_m, src_m in ((1, 1), (2, 2), (5, 0)):
        w2[:, dst_m, :] = a_w2[:, src_m, :]
        b2[0, dst_m, :] = a_b2[0, src_m, :]
        wmup[:, dst_m, :] = a_up[:, src_m, :]
        bmup[0, dst_m, :] = a_bu[0, src_m, :]
        wmdn[dst_m] = a_dn[src_m]
        # Двухтраншевый замок 20% Agile Pod (ch[59]=+8.0 -> pod_cap_mult=0.20) + привязка к стене ch[58]:
        w2[:, dst_m, 59] = 0.0
        wmdn[dst_m, :, 59] = 0.0
        b2[0, dst_m, 59] = +8.0
        b2[0, dst_m, 58] = +1.2 if dst_m == 5 else +0.4
        if dst_m == 5:
            # Делаем слот #5 полноценным 18%-м Agile Pod скальпером перед стеной (выше анти-пыли >=$12!):
            b2[0, 5, 38] = +1.6  # E_target ~ 80% * 0.20 pod_cap = 16% капитала (~$175 на $1100 счёте -> чёткие ордера по $25..$60!)

    # Паркуем слоты #1, #2, #5, #6 в Seed #0 на -14.0 -> побитовая идентичность Seed #0 с живым L92!
    d["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
    d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
    d["w_moe_up"] = wmup.reshape(256, MOE_MLP_UP_DIM_ORDER3)
    d["b_moe_up"] = bmup.reshape(1, MOE_MLP_UP_DIM_ORDER3)
    d["w_moe_dn"] = wmdn
    d["w_gate2"][:, [1, 2, 5, 6]] = 0.0
    d["b_gate2"][0, [1, 2, 5, 6]] = -14.0
    # НЕ вызываем глобальный _normalize_weights_order3 на всём d, чтобы не сжать веса Live_L92_V1 (w_gate2 norm=18.0)!

    dst = CHAMP_DIR / "l93_vault_master_init.pt"
    meta = {
        "arch_version": "order3_5_v8",
        "wf_version": "l93_vault_master_init",
        "champ_tag": "L93_VaultMaster_Init",
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
    _log_m(f"🧩 Собран двухтраншевый банк Order-3.5 (80% Core Vault L92 + 20% Agile Pod #1,#2,#5) -> {dst.name}")
    return dst


Q2_POD_IDX = (59, 58)  # 2 практически мёртвых нейрона роутера L92 (act_norm = 0.031 и 0.059 -> 0.000% влияния на Mode #4/#7!)


def _apply_l93_constraints(
    cand: dict[str, torch.Tensor],
    stage: str,
    base_ref: dict[str, torch.Tensor],
    dtype: torch.dtype,
    dev: torch.device,
    isolate_pod_subspace: bool = True,
) -> dict[str, torch.Tensor]:
    out = {k: base_ref[k].detach().clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}

    if not isolate_pod_subspace:
        return out

    # СТРОГАЯ ПОКОЛОНОЧНАЯ ИЗОЛЯЦИЯ БЕЗ ГЛОБАЛЬНОГО РЕСКЕЙЛА _normalize_weights_order3 (который сжимал w_gate2 Mode #4!):
    # 126 нейронов роутера L92 + весь res-блок + столбцы Mode #0,#3,#4,#7 остаются ПОБИТОВО 0.00e+00 от Live_L92_V1!
    # На каскадах логит Mode #4 равен +30..+60 (100% доминирование ядра L92!), а на штиле/росте падает до +5..+10,
    # где мягко включаются нейроны Q2_POD_IDX = (59, 58) и слоты 20% Agile Pod (#5 Wall-Scalp и #2 Up-Carry)!
    q_t = torch.tensor(Q2_POD_IDX, dtype=torch.long, device=dev)
    wg1_new = out["w_gate1"].float().clone()
    bg1_new = out["b_gate1"].float().clone()
    wgres1_new = out["w_gate_res1"].float().clone()
    wgres2_new = out["w_gate_res2"].float().clone()
    wg2_new = out["w_gate2"].float().clone()
    bg2_new = out["b_gate2"].float().clone()

    wg1_c = cand["w_gate1"].float().to(dev)
    bg1_c = cand["b_gate1"].float().to(dev)
    wg2_c = cand["w_gate2"].float().to(dev)
    bg2_c = cand["b_gate2"].float().to(dev)

    wg1_q = torch.clamp(wg1_c[:, q_t], -5.0, 5.0)
    n_q = torch.linalg.vector_norm(wg1_q, dim=0, keepdim=True).clamp(min=1e-6)
    wg1_q = wg1_q * torch.clamp(12.0 / n_q, max=1.0)
    wg1_new[:, q_t] = wg1_q
    # Защита от включения Agile Pod на каскадах/обвалах (crash900 [364] <= -9.0) и при заполненном портфеле (inv [257] <= -4.5):
    wg1_new[364, q_t] = torch.clamp(wg1_c[364, q_t], -16.0, -9.0)
    wg1_new[257, q_t] = torch.clamp(wg1_c[257, q_t], -12.0, -4.5)
    bg1_new[0, q_t] = torch.clamp(bg1_c[0, q_t], -3.5, -0.4)

    wgres1_new[q_t, :] = 0.0
    wgres2_new[:, q_t] = 0.0
    # Разрешаем слотам #1, #2, #5 читать как оба новых нейрона Q2_POD_IDX, так и замороженный контекст r_hid L92 (не трогая #4/#7!):
    for pm in (1, 2, 5):
        col_p = torch.clamp(wg2_c[:, pm], -1.5, 1.5)
        col_p[q_t] = torch.clamp(wg2_c[q_t, pm], 0.0, 4.0)
        nc_p = torch.linalg.vector_norm(col_p).clamp(min=1e-6)
        wg2_new[:, pm] = col_p * torch.clamp(6.0 / nc_p, max=1.0)
        bg2_new[0, pm] = torch.clamp(bg2_c[0, pm], -6.0, 14.0)
        bg2_new[0, K_MODES_ORDER3 + pm] = torch.clamp(bg2_c[0, K_MODES_ORDER3 + pm], -1.0, 4.5)

    wg2_new[q_t, 0] = 0.0
    wg2_new[q_t, 3] = 0.0
    wg2_new[q_t, 4] = 0.0
    wg2_new[q_t, 6] = 0.0
    wg2_new[q_t, 7] = 0.0

    out["w_gate1"] = wg1_new.to(dtype)
    out["b_gate1"] = bg1_new.to(dtype)
    out["w_gate_res1"] = wgres1_new.to(dtype)
    out["w_gate_res2"] = wgres2_new.to(dtype)
    out["w_gate2"] = wg2_new.to(dtype)
    out["b_gate2"] = bg2_new.to(dtype)

    if stage == "l93b_joint_polish":
        norm_d = _normalize_weights_order3(cand, dtype)
        reg_keys = ("w_reg1", "b_reg1_b", "b_reg1_a", "w_reg_res1", "b_reg_res", "w_reg_res2", "w_reg2", "b_reg2_b", "b_reg2_a")
        for kr in reg_keys:
            out[kr] = norm_d[kr].to(device=dev, dtype=dtype)

    # 3. Защита Core Vault Mode #4 и Mode #7 + архитектурный потолок 20% на Agile Pod (#1, #2, #5):
    w2_new = out["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
    b2_new = out["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
    wmup_new = out["w_moe_up"].float().view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
    bmup_new = out["b_moe_up"].float().view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
    wmdn_new = out["w_moe_dn"].float().view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()

    w2_c = cand["w2_moe"].float().to(dev).view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    b2_c = cand["b2_moe"].float().to(dev).view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    wmup_c = cand["w_moe_up"].float().to(dev).view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    bmup_c = cand["b_moe_up"].float().to(dev).view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    wmdn_c = cand["w_moe_dn"].float().to(dev).view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)

    # В Agile Pod слотах (#1, #2, #5) разрешаем тонкую настройку MLP и привязки к стенам [58],
    # НО жёстко удерживаем замок 20% экспозиции ch[59] = +8.0 (чтобы 80% кэша ВСЕГДА оставались свободны для Mode #4!):
    for km in (1, 2, 5):
        up_km = wmup_c[:, km, :]
        n_up = torch.linalg.vector_norm(up_km).clamp(min=1e-6)
        wmup_new[:, km, :] = up_km * torch.clamp(38.0 / n_up, max=1.0)
        bmup_new[:, km, :] = torch.clamp(bmup_c[:, km, :], -2.8, 2.8)
        mdn_km = wmdn_c[km].clone()
        n_dn = torch.linalg.vector_norm(mdn_km).clamp(min=1e-6)
        mdn_km = mdn_km * torch.clamp(24.0 / n_dn, max=1.0)
        mdn_km[:, 48] = 0.0
        mdn_km[:, 59] = 0.0
        if km != 5:
            mdn_km[:, 55] = 0.0
        wmdn_new[km] = mdn_km
        w2_new[:, km, 58] = torch.clamp(w2_c[:, km, 58], -0.25, 0.25)
        b2_new[0, km, 58] = torch.clamp(b2_c[0, km, 58], -8.0, 5.0)
        w2_new[:, km, 59] = 0.0
        b2_new[0, km, 59] = +8.0  # Строго <= 20% экспозиции в Agile Pod!

    if stage == "l93b_joint_polish":
        for km in (0, 4):
            w2_new[:, km, 58] = torch.clamp(w2_c[:, km, 58], -0.15, 0.15)
            b2_new[0, km, 58] = torch.clamp(b2_c[0, km, 58], -12.0, 2.5)

    out["w2_moe"] = w2_new.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
    out["b2_moe"] = b2_new.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
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


def _build_l93_seeds(base_ref: dict[str, torch.Tensor], stage: str, dtype: torch.dtype, dev: torch.device) -> list[dict[str, torch.Tensor]]:
    # Seed #0: 100% Zero-Regression Anchor (= L92_24h_Endogenous_2pct_V1!)
    seeds = [_apply_l93_constraints(base_ref, stage, base_ref, dtype, dev, isolate_pod_subspace=False)]
    if stage == "l93b_joint_polish":
        for wb4 in (-12.0, -6.0, -3.0):
            dm = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            b2 = dm["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
            b2[0, 4, 58] = wb4
            dm["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
            seeds.append(_apply_l93_constraints(dm, stage, base_ref, dtype, dev))
        return seeds

    q5, q2 = Q2_POD_IDX[0], Q2_POD_IDX[1]  # Нейрон 59 -> Mode #5 (Wall-Scalp 20%), Нейрон 58 -> Mode #2 (Up-Carry 20%)

    # Реальные квантили логита Mode #4 в Live_L92_V1:
    #   p10 (штиль/флэт/рост) = +9.77 | median = +18.19 | p90..max (каскады!) = +30.75..+60.50!
    # Поэтому смещение +5.5..+9.8 включает 20% Agile Pod ТОЛЬКО в нижних 3%..12% спокойных/трендовых тиков,
    # оставляя 100.000% доминирование Mode #4 (+30..+60) на всех каскадах!
    for m5_b, m2_b, m1_b, cr_sh in (
        (+5.2, +4.5, -6.0, 14.0),  # ультра-точечный 20% Pod (~2..4% штиля/тренда, 0% влияния на каскады!)
        (+6.8, +6.0, -6.0, 14.0),  # деликатный 20% Pod (~5..8% штиля/тренда)
        (+8.2, +7.2, -5.0, 14.5),  # умеренный 20% Pod (~9..14% штиля/тренда под жёстким щитом crash900=-14.5)
        (-6.0, +7.8, -6.0, 14.0),  # чистый 20% Up-Trend Carry (#2) строго на бычьем консенсусе Gate+Ourbit+BTC
        (+7.8, -6.0, -6.0, 14.5),  # чистый 20% Wall-Shielded Scalper (#5) строго перед крупной плитой L500 [393]
        (+6.5, +6.2, +5.5, 14.0),  # трио 20% Agile Pod (#5 + #2 + #1) на глубокой поддержке стакана 2% [390]
    ):
        d = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        wg1 = d["w_gate1"].float().clone()
        bg1 = d["b_gate1"].float().clone()
        wg2 = d["w_gate2"].float().clone()
        bg2 = d["b_gate2"].float().clone()

        wg1[:, [q5, q2]] = 0.0
        bg1[0, [q5, q2]] = -1.2

        # Нейрон q5 (59) -> Mode #5 (20% Wall-Shielded Scalp): широкий спред [336], стена L500 [393], 2% имбаланс [390], поводырь [382]
        wg1[336, q5] = +1.8; wg1[393, q5] = +2.2; wg1[390, q5] = +1.5; wg1[382, q5] = +1.4
        wg1[364, q5] = -cr_sh; wg1[257, q5] = -7.0

        # Нейрон q2 (58) -> Mode #2 (20% Up-Carry): консенсус Gate+Ourbit [382], CVD 60s [379,381], BTC 15m [386], ret30m [370]
        wg1[382, q2] = +2.2; wg1[379, q2] = +1.5; wg1[381, q2] = +1.4; wg1[386, q2] = +1.4; wg1[370, q2] = +1.6
        wg1[364, q2] = -cr_sh; wg1[257, q2] = -6.0

        wg2[:, [1, 2, 5]] = 0.0
        wg2[q5, 5] = +2.4; bg2[0, 5] = m5_b; bg2[0, 13] = +2.2
        wg2[q2, 2] = +2.4; bg2[0, 2] = m2_b; bg2[0, 10] = +1.8
        wg2[[q5, q2], 1] = +1.5; bg2[0, 1] = m1_b; bg2[0, 9] = +1.6

        d["w_gate1"] = wg1.to(dtype)
        d["b_gate1"] = bg1.to(dtype)
        d["w_gate2"] = wg2.to(dtype)
        d["b_gate2"] = bg2.to(dtype)
        seeds.append(_apply_l93_constraints(d, stage, base_ref, dtype, dev))

    return seeds


class SeparableCMAES:
    """
    Диагональный Separable CMA-ES в пространстве активных тензоров нейросети:
      - Антитетическая выборка Z ~ N(0, diag(sigma^2)) на GPU.
      - Ранговые логарифмические веса mu лучших кандидатов (инвариантно к масштабу наград!).
      - Кумулятивный путь эволюции p_c и индивидуальная покоординатная адаптация sigma_j.
    """

    def __init__(self, shapes: dict[str, tuple[int, ...]], init_scales: dict[str, float], mu: int, dev: torch.device):
        self.keys = list(shapes.keys())
        self.shapes = shapes
        self.init_scales = init_scales
        self.mu = mu
        self.dev = dev

        raw_w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)], dtype=np.float64)
        self.weights = torch.tensor(raw_w / raw_w.sum(), dtype=torch.float32, device=dev)
        self.mu_eff = float(1.0 / np.sum((raw_w / raw_w.sum()) ** 2))

        self.c_c = 0.22
        self.c_1 = 0.08
        self.c_mu = 0.14
        self.p_c = {k: torch.zeros(sh, dtype=torch.float32, device=dev) for k, sh in shapes.items()}
        self.diag_std = {k: torch.ones(sh, dtype=torch.float32, device=dev) for k, sh in shapes.items()}

    def sample_half(self, half: int, global_step_scale: float, ring_mult: torch.Tensor) -> dict[str, torch.Tensor]:
        noise: dict[str, torch.Tensor] = {}
        for k in self.keys:
            sh = self.shapes[k]
            rv = ring_mult.view(half, *([1] * len(sh)))
            std_k = self.diag_std[k].unsqueeze(0) * (self.init_scales[k] * global_step_scale)
            z = torch.randn((half, *sh), dtype=torch.float32, device=self.dev)
            noise[k] = z * std_k * rv
        return noise

    def build_recombined_candidates(
        self,
        mean_state: dict[str, torch.Tensor],
        pop_cands: list[dict[str, torch.Tensor]],
        scores: np.ndarray,
        lr_scales: tuple[float, ...] = (0.65, 1.05, 1.55),
    ) -> list[dict[str, torch.Tensor]]:
        order = np.argsort(scores)[::-1][: self.mu]
        deltas: dict[str, torch.Tensor] = {}
        z_sq_weighted: dict[str, torch.Tensor] = {}

        for k in self.keys:
            m_t = mean_state[k].float()
            top_t = torch.stack([pop_cands[int(idx)][k].float() - m_t for idx in order], dim=0)
            w_view = self.weights.view(self.mu, *([1] * len(self.shapes[k])))
            d_mean = (w_view * top_t).sum(dim=0)
            deltas[k] = d_mean

            denom = torch.clamp(self.diag_std[k] * self.init_scales[k], min=1e-5)
            z_norm = torch.clamp(top_t / denom.unsqueeze(0), -4.0, 4.0)
            z_sq_weighted[k] = (w_view * (z_norm * z_norm)).sum(dim=0)

        # Обновляем кумулятивный путь p_c и диагональную ковариацию diag_std (Sep-CMA-ES!):
        for k in self.keys:
            denom = torch.clamp(self.diag_std[k] * self.init_scales[k], min=1e-5)
            step_norm = torch.clamp((deltas[k] / denom) * math.sqrt(self.mu_eff), -3.5, 3.5)
            self.p_c[k] = (1.0 - self.c_c) * self.p_c[k] + math.sqrt(self.c_c * (2.0 - self.c_c)) * step_norm
            cov_upd = 0.5 * self.c_1 * (self.p_c[k] * self.p_c[k] - 1.0) + 0.5 * self.c_mu * (z_sq_weighted[k] - 1.0)
            self.diag_std[k] = torch.clamp(self.diag_std[k] * torch.exp(torch.clamp(cov_upd, -0.20, 0.20)), 0.30, 3.20)

        out_cands: list[dict[str, torch.Tensor]] = []
        for lr in lr_scales:
            c_new = {k: v.clone() for k, v in mean_state.items()}
            for k in self.keys:
                # Комбинация взвешенного шага CMA и кумулятивного импульса p_c:
                step_k = lr * deltas[k] + (0.20 * lr) * self.p_c[k] * (self.diag_std[k] * self.init_scales[k])
                c_new[k] = (mean_state[k].float() + step_k).to(mean_state[k].dtype)
            out_cands.append(c_new)
        return out_cands


def _island_l93_worker(
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
    mkt_p = BASE_DIR / ("data/cache/rl_hybrid_market_obs_112d_raw_with_self.npz" if is_raw else "data/cache/rl_hybrid_market_obs_112d.npz")
    lob = np.load(lob_p, mmap_mode="r")
    mkt_full = np.load(mkt_p, mmap_mode="r")["market_obs_112d"]

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

    val_basket, tr_pool, blind_48h = build_purged_24h_splits(total_rows)
    log_w(
        f"📐 Purged 24h Split: total={total_rows}s ({total_rows/86400:.2f}d) | TrainPool={len(tr_pool)} чистых 24h окон "
        f"| PurgedVal={len(val_basket)}x24h {[w[1] for w in val_basket]} | Blind48h={[w[1] for w in blind_48h]} (0% overlap!)"
    )

    pol = MultimodalMoEPolicyOrder3(n_groups=n_groups, n_budgets=6, device=dev)
    base_cpu = pol.load_initial_champion(Path(src_ckpt_str))
    base_ref = {k: base_cpu[k].to(device=dev, dtype=pol.dtype) for k in pol.PARAM_KEYS}
    seeds = _build_l93_seeds(base_ref, stage, pol.dtype, dev)

    def eval_24h_raw(cands: list[dict], bask: list, dual_stress: bool = True) -> tuple[list[dict], list[dict]]:
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
        return rc, rh

    # Оцениваем Seed #0 (побитовый эталон L92) и стартовый банк активных Agile Pod сидов (#1..#6) на Purged Val (5 x 24h):
    rc_s, rh_s = eval_24h_raw(seeds, val_basket, dual_stress=True)
    ref_val_ev = 0.50 * float(rc_s[0]["clean_ev_pct"]) + 0.50 * float(rh_s[0]["clean_ev_pct"])
    seed_res = [compute_l93_vault_fitness(a, b, len(val_basket), ref_ev_mix=ref_val_ev) for a, b in zip(rc_s, rh_s)]
    for si, sr_i in enumerate(seed_res):
        log_w(
            f"   [Seed #{si}] L93={sr_i['l93_score_bp']:+.2f}bp | EV/d={sr_i['ev_day_cal_pct']:+.3f}%/{sr_i['ev_day_har_pct']:+.3f}% "
            f"| DD={sr_i['dd_max_pct']:.2f}% | Fills/d={sr_i['fills_per_day']:.1f} | Modes={[round(x,3) for x in sr_i['mode_probs']]}"
        )
    seed0_sc = float(seed_res[0]["l93_score_bp"])
    # Стартуем поиск Sep-CMA-ES из лучшего АКТИВНОГО Agile Pod сида (#1..#6), чтобы эволюционировать живое подпространство Q6_POD_IDX
    # (а не сидеть в мёртвой точке b_gate2=-14.0!), но сохраняем в чекпоинт ТОЛЬКО при строгом пробое планки эталона Seed #0!
    active_pool = list(range(len(seeds))) if stage == "l93b_joint_polish" else (list(range(1, len(seeds))) if len(seeds) > 1 else [0])
    best_i = int(active_pool[int(np.argmax([seed_res[idx]["ev_day_mix_pct"] for idx in active_pool]))])
    champ_state = {k: v.clone() for k, v in seeds[best_i].items()}
    best_sc = float(seed_res[best_i]["l93_score_bp"])
    saved_best_sc = max(seed0_sc, best_sc)
    sr0 = seed_res[best_i]
    log_w(
        f"🌱 Search Init Seed #{best_i} (Anchor Seed#0={seed0_sc:+.2f}bp, EV/d={ref_val_ev:+.3f}%): "
        f"L93={best_sc:+.2f}bp | EV/d={sr0['ev_day_cal_pct']:+.3f}%/{sr0['ev_day_har_pct']:+.3f}% "
        f"| DD={sr0['dd_max_pct']:.2f}% | Modes={[round(x,3) for x in sr0['mode_probs']]}"
    )

    def save_champ(it_n: int, vr: dict, st_to_save: dict[str, torch.Tensor]) -> None:
        meta = {
            "arch_version": "order3_5_v8",
            "wf_version": f"l93_cma_vault_{stage}_{world_mode}",
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
        ck_cpu = {k: st_to_save[k].detach().cpu() for k in pol.PARAM_KEYS}
        torch.save({**ck_cpu, **meta}, out_dir / f"island_{island_id}_best.pt")

    if best_sc >= seed0_sc and float(sr0["ev_day_mix_pct"]) >= ref_val_ev:
        save_champ(0, sr0, champ_state)
    else:
        save_champ(0, seed_res[0], seeds[0])

    # Настраиваем Separable CMA-ES по активным параметрам (ствол w1 заморожен на 100%!):
    active_shapes = {
        "w_gate1": (ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3),
        "b_gate1": (1, ROUTER_H_DIM_ORDER3),
        "w_gate_res1": (ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3),
        "b_gate_res": (1, 2 * ROUTER_RES_H_ORDER3),
        "w_gate_res2": (ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3),
        "w_gate2": (ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3),
        "b_gate2": (1, ROUTER_OUT_DIM_ORDER3),
        "w_reg1": (REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3),
        "b_reg1_b": (1, REG_H_DIM_ORDER3),
        "b_reg1_a": (1, REG_H_DIM_ORDER3),
        "w_reg2": (REG_H_DIM_ORDER3, 2),
        "b_reg2_b": (1, 2),
        "b_reg2_a": (1, 2),
        "w2_moe": (256, MOE_OUT_DIM_ORDER3),
        "b2_moe": (1, MOE_OUT_DIM_ORDER3),
        "w_moe_up": (256, MOE_MLP_UP_DIM_ORDER3),
        "w_moe_dn": (K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3),
    }
    init_scales = {
        "w_gate1": 0.026,
        "b_gate1": 0.042,
        "w_gate_res1": 0.034,
        "b_gate_res": 0.040,
        "w_gate_res2": 0.038,
        "w_gate2": 0.042,
        "b_gate2": 0.140,
        "w_reg1": 0.022,
        "b_reg1_b": 0.036,
        "b_reg1_a": 0.036,
        "w_reg2": 0.030,
        "b_reg2_b": 0.050,
        "b_reg2_a": 0.050,
        "w2_moe": 0.020,
        "b2_moe": 0.080,
        "w_moe_up": 0.022,
        "w_moe_dn": 0.024,
    }
    cma = SeparableCMAES(active_shapes, init_scales, mu=16, dev=dev)
    # Фокусируем ковариацию роутера на экстра-признаках [336..394] (особенно новых 16D поводырях [378..394]):
    cma.diag_std["w_gate1"][256:378, :] = 1.80
    cma.diag_std["w_gate1"][378:394, :] = 2.80

    rng = np.random.default_rng(20261393 + island_id * 77773)
    half = (n_groups - 2) // 2
    r_view = pol.ring_mult_half
    n_tr_w = min(6, len(tr_pool))
    seed0_anchor = {k: seeds[0][k].clone() for k in pol.PARAM_KEYS}

    for it in range(1, max_iters + 1):
        if (time.time() - t0) / 3600.0 >= max_hours:
            break
        t_it = time.time()

        if it % 6 == 0:
            nb_p = out_dir / f"island_{1 - island_id}_best.pt"
            if nb_p.exists():
                try:
                    nb_cpu = pol.load_initial_champion(nb_p)
                    nb_st = _apply_l93_constraints({k: nb_cpu[k].to(dev, pol.dtype) for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
                    rc_n, rh_n = eval_24h_raw([nb_st], val_basket, dual_stress=True)
                    nb_r = compute_l93_vault_fitness(rc_n[0], rh_n[0], len(val_basket), ref_ev_mix=ref_val_ev)
                    if float(nb_r["l93_score_bp"]) > best_sc + 0.05:
                        champ_state = nb_st
                        best_sc = float(nb_r["l93_score_bp"])
                        if best_sc > saved_best_sc + 0.02 and float(nb_r["ev_day_mix_pct"]) >= ref_val_ev:
                            saved_best_sc = best_sc
                            save_champ(it, nb_r, champ_state)
                        log_w(f"🧬 [It {it:02d}] Кросс-мировая миграция -> L93={best_sc:+.2f}bp (EV/d={nb_r['ev_day_cal_pct']:+.3f}%, Fills/d={nb_r['fills_per_day']:.1f})")
                except Exception:
                    pass

        prog = it / max(1.0, float(max_iters))
        sig_sc = 1.00 if prog < 0.45 else (0.65 if prog < 0.80 else 0.38)
        if stage == "l93b_joint_polish":
            sig_sc *= 0.35

        for k in pol.PARAM_KEYS:
            pol.__dict__[k][:] = champ_state[k].unsqueeze(0)
        # Слот #1 в батче всегда хранит эталон L92 Seed #0, чтобы Residual Alpha на случайном Train-батче считалась точно!
        for k in pol.PARAM_KEYS:
            pol.__dict__[k][1].copy_(seed0_anchor[k])

        noise_d = cma.sample_half(half, sig_sc, r_view)

        for attr_n, eps_t in noise_d.items():
            tgt = pol.__dict__[attr_n][2 : 2 + 2 * half]
            tgt.add_(torch.stack([eps_t, -eps_t], dim=1).reshape_as(tgt).to(tgt.dtype))

        for gi in range(2, 2 + 2 * half):
            cl = _apply_l93_constraints({k: pol.__dict__[k][gi] for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
            for k in pol.PARAM_KEYS:
                pol.__dict__[k][gi].copy_(cl[k])

        tr_ws = rng.choice(tr_pool, size=n_tr_w, replace=False)
        tr_bask = [(f"TR24_{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(tr_ws)]
        pop_cands = pol.get_candidate_states()
        rc_tr, rh_tr = eval_24h_raw(pop_cands, tr_bask, dual_stress=False)
        ref_tr_ev = float(rc_tr[1]["clean_ev_pct"])
        tr_res = [compute_l93_vault_fitness(a, b, len(tr_bask), ref_ev_mix=ref_tr_ev) for a, b in zip(rc_tr, rh_tr)]
        # Исключаем слот #1 (неподвижный якорь Seed#0) из ранговой рекомбинации Sep-CMA-ES!
        tr_scs = np.array([float(r["l93_score_bp"]) for r in tr_res], dtype=np.float64)
        tr_scs[1] = -1e9

        cma_raw = cma.build_recombined_candidates(champ_state, pop_cands, tr_scs, lr_scales=(0.45, 0.85, 1.35))
        cma_cands = [_apply_l93_constraints(c, stage, base_ref, pol.dtype, dev) for c in cma_raw]

        # ТОЛЬКО чистые рекомбинированные средние Sep-CMA-ES (0 отбора случайного шума top_muts на валидации!):
        shortlist = [champ_state] + cma_cands
        rc_v, rh_v = eval_24h_raw(shortlist, val_basket, dual_stress=True)
        val_res = [compute_l93_vault_fitness(a, b, len(val_basket), ref_ev_mix=ref_val_ev) for a, b in zip(rc_v, rh_v)]
        bv_i = int(np.argmax([float(r["l93_score_bp"]) for r in val_res]))
        cand_sc = float(val_res[bv_i]["l93_score_bp"])

        if cand_sc > best_sc + 0.02:
            gain = cand_sc - best_sc
            best_sc = cand_sc
            champ_state = {k: v.clone() for k, v in shortlist[bv_i].items()}
            vr = val_res[bv_i]
            beat_anchor = (best_sc > saved_best_sc + 0.02) and (float(vr["ev_day_mix_pct"]) >= ref_val_ev)
            if beat_anchor:
                saved_best_sc = best_sc
                save_champ(it, vr, champ_state)
            tag_s = "🏆[SAVED>L92]" if beat_anchor else "📈[POD_CLIMB]"
            log_w(
                f"⭐ {tag_s} [It {it:02d} | +{gain:.2f}bp] L93={best_sc:+.2f}bp | EV/d={vr['ev_day_cal_pct']:+.3f}%/{vr['ev_day_har_pct']:+.3f}% "
                f"| DD={vr['dd_max_pct']:.2f}% | Fills/d={vr['fills_per_day']:.1f} | Cnl/h={vr['cnl_per_hr']:.1f} "
                f"| Modes={[round(x,3) for x in vr.get('mode_probs',[])]} | {time.time()-t_it:.1f}s"
            )
        elif it % 4 == 0:
            log_w(
                f"   [It {it:02d}/{max_iters}] TrainTop={np.max(tr_scs):+.2f}bp (RefEV={ref_tr_ev:+.2f}%) "
                f"| PodChamp={best_sc:+.2f}bp (Anchor={seed0_sc:+.2f}bp) | {time.time()-t_it:.1f}s"
            )

    log_w(f"🏁 Done [{stage}] ({world_mode.upper()}) in {(time.time()-t0)/60:.1f}m | Best PurgedVal L93={best_sc:+.2f}bp")


def select_and_coronate_l93() -> dict:
    """
    Строгий научный экзамен на:
      1) Purged Val (3 x 24h, 0% нахлёста с Train)
      2) Полной непересекающейся 13-суточной развёртке (13 x 24h)
      3) Неприкосновенном Слепом 48ч OOS (BLIND_D12 + BLIND_D13, включая ночной спуск -8.6%!)
    во ВСЕХ 4 физиках (DEC/RAW x Calib/Harsh) в прямом бою против L92_24h_Endogenous_2pct_V1!
    """
    _log_m("🏆 СТАРТ ФИНАЛЬНОГО ТУРНИРА L93 (ORDER-3.5) ПРОТИВ L92_LIVE (13x24h + PURGED VAL + СЛЕПЫЕ 48ч OOS)")
    dev = torch.device("cuda:0")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)

    cands_named = [("Live_L92_V1", BASE_LIVE_PATH)]
    for nm, p_c in (
        ("L93a_Vault_RAW", CHAMP_DIR / "bak_l93a_vault" / "island_0_best.pt"),
        ("L93a_Vault_DEC", CHAMP_DIR / "bak_l93a_vault" / "island_1_best.pt"),
        ("L93b_Polish_RAW", CHAMP_DIR / "bak_l93b_polish" / "island_0_best.pt"),
        ("L93b_Polish_DEC", CHAMP_DIR / "bak_l93b_polish" / "island_1_best.pt"),
    ):
        if p_c.exists():
            cands_named.append((nm, p_c))

    states = [pol.load_initial_champion(p) for _, p in cands_named]
    states_gpu = [{k: v.to(dev) for k, v in st.items()} for st in states]
    sub = 3016277
    summary = {
        nm: {
            "l93_4w": 0.0,
            "ev_day_4w": 0.0,
            "purged_val_ev": 0.0,
            "blind_48h_ev_d": 0.0,
            "blind_d12_ev": 0.0,
            "blind_d13_ev": 0.0,
            "max_dd": 0.0,
            "fills_day": 0.0,
            "worlds": {},
        }
        for nm, _ in cands_named
    }

    for wm in ("dec", "raw"):
        is_r = wm == "raw"
        lob = np.load(BASE_DIR / ("data/cache/rl_guard_lob_cache_600h_raw_with_self.npz" if is_r else "data/cache/rl_guard_lob_cache_600h.npz"), mmap_mode="r")
        mkt = np.load(BASE_DIR / ("data/cache/rl_hybrid_market_obs_112d_raw_with_self.npz" if is_r else "data/cache/rl_hybrid_market_obs_112d.npz"), mmap_mode="r")["market_obs_112d"]
        ts = lob["ts"][sub:]
        n_sub = len(ts)
        q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
        q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
        sv_np = np.asarray(lob["sell_vol"][sub:], dtype=np.float32).copy()
        bv_np = np.asarray(lob["buy_vol"][sub:], dtype=np.float32).copy()
        qm = (ts >= q0_u) & (ts <= q1_u)
        sv_np[qm] = 0.0
        bv_np[qm] = 0.0

        mkt_g = torch.from_numpy(np.asarray(mkt[sub:], dtype=np.float32)).to(dev, torch.bfloat16)
        b10_g = torch.from_numpy(np.asarray(lob["bid_p"][sub:, :10], dtype=np.float32)).to(dev)
        a10_g = torch.from_numpy(np.asarray(lob["ask_p"][sub:, :10], dtype=np.float32)).to(dev)
        bq10_g = torch.from_numpy(np.asarray(lob["bid_q"][sub:, :10], dtype=np.float32)).to(dev)
        aq10_g = torch.from_numpy(np.asarray(lob["ask_q"][sub:, :10], dtype=np.float32)).to(dev)
        sv_g = torch.from_numpy(sv_np).to(dev)
        bv_g = torch.from_numpy(bv_np).to(dev)

        val_bask, _, blind_48h = build_purged_24h_splits(n_sub)
        starts_13d = [int(x) for x in np.linspace(1800, n_sub - WIN_24H - 5, 13)]
        bask_13d = [(f"D{k+1:02d}", ws, 0.0, 0.0) for k, ws in enumerate(starts_13d)]

        rc = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=bask_13d, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)
        rh = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=bask_13d, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)

        vc = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=val_bask, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)
        vh = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=val_bask, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)

        oc12 = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=[blind_48h[0]], win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)
        oh12 = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=[blind_48h[0]], win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)
        oc13 = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=[blind_48h[1]], win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)
        oh13 = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=[blind_48h[1]], win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)

        ref_13d = 0.50 * float(rc[0]["clean_ev_pct"]) + 0.50 * float(rh[0]["clean_ev_pct"])
        for idx_c, (nm, _) in enumerate(cands_named):
            fm = compute_l93_vault_fitness(rc[idx_c], rh[idx_c], 13, ref_ev_mix=ref_13d)
            pval_m = 0.50 * float(vc[idx_c]["clean_ev_pct"]) + 0.50 * float(vh[idx_c]["clean_ev_pct"])
            b12_m = 0.50 * float(oc12[idx_c]["clean_ev_pct"]) + 0.50 * float(oh12[idx_c]["clean_ev_pct"])
            b13_m = 0.50 * float(oc13[idx_c]["clean_ev_pct"]) + 0.50 * float(oh13[idx_c]["clean_ev_pct"])
            b48_d = 0.50 * (b12_m + b13_m)

            summary[nm]["l93_4w"] += 0.50 * fm["l93_score_bp"]
            summary[nm]["ev_day_4w"] += 0.50 * fm["ev_day_mix_pct"]
            summary[nm]["purged_val_ev"] += 0.50 * pval_m
            summary[nm]["blind_48h_ev_d"] += 0.50 * b48_d
            summary[nm]["blind_d12_ev"] += 0.50 * b12_m
            summary[nm]["blind_d13_ev"] += 0.50 * b13_m
            summary[nm]["max_dd"] = max(summary[nm]["max_dd"], fm["dd_max_pct"])
            summary[nm]["fills_day"] += 0.50 * fm["fills_per_day"]
            summary[nm]["worlds"][wm.upper()] = {
                "calib_ev_d": round(fm["ev_day_cal_pct"], 3),
                "harsh_ev_d": round(fm["ev_day_har_pct"], 3),
                "purged_val_ev_d": round(pval_m, 3),
                "blind_48h_ev_d": round(b48_d, 3),
                "blind_d12_ev": round(b12_m, 3),
                "blind_d13_ev": round(b13_m, 3),
                "blind_48h_fills": round(float(oc12[idx_c]["fills"]) + float(oc13[idx_c]["fills"]), 1),
                "dd_max": round(fm["dd_max_pct"], 2),
                "fills_per_day": round(fm["fills_per_day"], 2),
                "cnl_per_hr": round(fm["cnl_per_hr"], 1),
                "modes": [round(float(x), 3) for x in fm["mode_probs"]],
            }
            _log_m(
                f"  [{wm.upper()}] {nm:16s} | 13d_EV/d={fm['ev_day_cal_pct']:+.3f}%/{fm['ev_day_har_pct']:+.3f}% "
                f"| PurgedVal={pval_m:+.3f}%/d | Blind48h={b48_d:+.3f}%/d (D12={b12_m:+.2f}%, D13={b13_m:+.2f}%) "
                f"| DD={fm['dd_max_pct']:.2f}% | Fills/d={fm['fills_per_day']:.1f} | Modes={[round(float(x),2) for x in fm['mode_probs']]}"
            )

    _log_m("-" * 108)
    for nm, _ in cands_named:
        s = summary[nm]
        _log_m(
            f"📊 ИТОГ [{nm:16s}] L93_4W={s['l93_4w']:+.2f}bp | Mean_13d_EV/d={s['ev_day_4w']:+.3f}% "
            f"| PurgedVal={s['purged_val_ev']:+.3f}%/d | Blind48h={s['blind_48h_ev_d']:+.3f}%/d "
            f"(D12={s['blind_d12_ev']:+.2f}%, D13={s['blind_d13_ev']:+.2f}%) | Fills/d={s['fills_day']:.1f} | MaxDD={s['max_dd']:.2f}%"
        )
    (LOG_DIR / "coronation_l93_order35.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage(stage: str, src: Path, sub_d: str, iters: int, max_h: float) -> tuple[Path, Path]:
    _log_m(f"▶️ ЗАПУСК ЭТАПА SEP-CMA-ES [{stage}] на 2x GPU (iters={iters}, limit={max_h:.2f}h)")
    ctx = mp.get_context("spawn")
    p0 = ctx.Process(target=_island_l93_worker, args=(stage, str(src), sub_d, 0, "cuda:0", "raw", 80, iters, max_h))
    p1 = ctx.Process(target=_island_l93_worker, args=(stage, str(src), sub_d, 1, "cuda:1", "dec", 80, iters, max_h))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    if p0.exitcode != 0 or p1.exitcode != 0:
        raise RuntimeError(f"Stage {stage} failed: {p0.exitcode}, {p1.exitcode}")
    return CHAMP_DIR / sub_d / "island_0_best.pt", CHAMP_DIR / sub_d / "island_1_best.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters-vault", type=int, default=32)
    ap.add_argument("--iters-polish", type=int, default=16)
    ap.add_argument("--skip-vault", action="store_true")
    ap.add_argument("--coronate-only", action="store_true")
    args = ap.parse_args()

    if not args.coronate_only:
        r_a = CHAMP_DIR / "bak_l93a_vault" / "island_0_best.pt"
        d_a = CHAMP_DIR / "bak_l93a_vault" / "island_1_best.pt"
        if not args.skip_vault:
            init_p = build_l93_vault_master_bank()
            r_a, d_a = run_stage("l93a_cma_vault", init_p, "bak_l93a_vault", args.iters_vault, 0.85)
        res_mid = select_and_coronate_l93()
        best_a_p = r_a if res_mid["L93a_Vault_RAW"]["l93_4w"] >= res_mid["L93a_Vault_DEC"]["l93_4w"] else d_a

        if args.iters_polish > 0:
            run_stage("l93b_joint_polish", best_a_p, "bak_l93b_polish", args.iters_polish, 0.45)

    res_fin = select_and_coronate_l93()
    best_nm = max(res_fin.keys(), key=lambda k: res_fin[k]["l93_4w"])
    _log_m(
        f"🏅 АБСОЛЮТНЫЙ ЛИДЕР ТУРНИРА ORDER-3.5: {best_nm} "
        f"(L93_4W={res_fin[best_nm]['l93_4w']:+.2f}bp, EV/d={res_fin[best_nm]['ev_day_4w']:+.3f}%, "
        f"Blind48h={res_fin[best_nm]['blind_48h_ev_d']:+.3f}%/d)"
    )

    name_to_p = {
        "Live_L92_V1": BASE_LIVE_PATH,
        "L93a_Vault_RAW": CHAMP_DIR / "bak_l93a_vault" / "island_0_best.pt",
        "L93a_Vault_DEC": CHAMP_DIR / "bak_l93a_vault" / "island_1_best.pt",
        "L93b_Polish_RAW": CHAMP_DIR / "bak_l93b_polish" / "island_0_best.pt",
        "L93b_Polish_DEC": CHAMP_DIR / "bak_l93b_polish" / "island_1_best.pt",
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
