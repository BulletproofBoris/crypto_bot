#!/usr/bin/env python3
"""
_tools/train_l91_endogenous_2pct.py
=============================================================================
8-часовой исследовательско-тренировочный конвейер Order-3 v2.1 (Урок L91):
  - Целевой среднесуточный прирост капитала +2.0%/сутки со штрафом за недобор
    (отменяет бесплатное отсиживание в 0 сделок при сохранении жёсткого щита DD/Harsh).
  - 100% эндогенное управление размером заявки (через спектр сидов экспозиции 4%..85%
    и непрерывное смешивание мод с кэш-бункером Mode #3 в Оркестраторе 386D).
  - 100% эндогенное управление темпом заявок (нервный Mode #0, ритмичный Mode #1/#2,
    спокойный, но не засыпающий Mode #4) через разблокированный 36D Регулятор w_reg*.
  - Строгий запрет порчи нелинейных SwiGLU-слоёв линейным 50/50 усреднением:
    выбор целостного победителя по кросс-мировому 4-физическому экзамену (DEC+RAW x Calib+Harsh).
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
LOG_DIR = BASE_DIR / "logs" / "l91_endogenous_2pct"

TARGET_DAILY_PCT = 2.00


def _log_m(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "master_l91.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def compute_l91_fitness(
    cal_r: dict,
    har_r: dict,
    win_len: int,
    n_windows: int,
    stage: str,
    target_daily_pct: float = TARGET_DAILY_PCT,
) -> dict:
    """
    Калиброванная в Блоке 0 (Study 0.4) целевая функция L91 (+2.0%/сутки):
      - Совмещает Calib + Harsh (50/50) по Geo и суточному EV%/day.
      - Плавно штрафует за недобор до +2.0%/сутки (на этапе приёмника разгрузки l91d2_mode7
        цель — чистый выход выше пассивного удержания без штрафа за оборот покупок).
      - Штрафует только патологический перерасход API (>420 отмен/ч), оставляя быстрым модам
        естественную свободу реакции внутри физического Token-Bucket 0.35/с (630 пар/ч).
      - Держит нелинейный щит просадки MaxDD > 4.5%.
    """
    hrs = max(0.25, float(win_len - 1) / 3600.0)
    day_mult = 24.0 / hrs
    ev_day_c = float(cal_r["clean_ev_pct"]) * day_mult
    ev_day_h = float(har_r["clean_ev_pct"]) * day_mult
    ev_day_mix = 0.55 * ev_day_c + 0.45 * ev_day_h
    geo_mix = 0.50 * float(cal_r["geo_score_bp"]) + 0.50 * float(har_r["geo_score_bp"])
    dd_max = max(float(cal_r["dd_pct"]), float(har_r["dd_pct"]))

    if stage == "l91d2_mode7":
        # Приёмник разгрузки стартует с готовым портфелем rho_0: его задача — продать лучше пассивного холд-эталона с мин. DD
        shortfall_bp = 0.0
    else:
        # Для отдельных узких специалистов цель пропорциональна их вкладу (1.4%/сут), для полного ансамбля/роутера — полные 2.0%/сут!
        eff_target = target_daily_pct if stage in ("l91e_regulator", "l91f_orchestrator", "l91f2_multimode", "l91g_joint", "coronate") else 1.40
        w_sf = 0.55 if stage in ("l91e_regulator", "l91f_orchestrator", "l91f2_multimode", "l91g_joint", "coronate") else 0.42
        shortfall_bp = w_sf * max(0.0, (eff_target - ev_day_mix) * 100.0)
        # Микро-штраф за полное вырождение в 0 сделок на рабочих окнах специалиста (< 6 сделок на корзину):
        fl_tot = float(cal_r["fills"])
        if fl_tot < 6.0:
            shortfall_bp += (6.0 - fl_tot) * 6.0

    tot_hrs = max(1.0, hrs * float(n_windows))
    cnl_per_hr = float(cal_r.get("cancels", 0.0)) / tot_hrs
    cpf = float(cal_r.get("cancels", 0.0)) / max(1.0, float(cal_r["fills"]))
    api_pen = 0.12 * max(0.0, cnl_per_hr - 420.0) + 0.035 * max(0.0, cpf - 580.0)
    dd_pen = 14.0 * (max(0.0, dd_max - 4.50) ** 1.35)

    l91_sc = geo_mix + 45.0 * ev_day_mix - shortfall_bp - api_pen - dd_pen
    return {
        "l91_score_bp": float(l91_sc),
        "geo_mix_bp": float(geo_mix),
        "ev_day_cal_pct": float(ev_day_c),
        "ev_day_har_pct": float(ev_day_h),
        "dd_max_pct": float(dd_max),
        "fills": float(cal_r["fills"]),
        "cnl_per_hr": float(cnl_per_hr),
        "shortfall_bp": float(shortfall_bp),
        "mode_probs": cal_r.get("mode_probs", []),
    }


def _lock_router_single_mode(st: dict[str, torch.Tensor], target_mode: int, dtype: torch.dtype, dev: torch.device) -> dict[str, torch.Tensor]:
    d = {k: v.detach().clone().to(device=dev, dtype=dtype) for k, v in st.items()}
    d["w_gate1"].zero_()
    d["b_gate1"].zero_()
    d["w_gate_res1"].zero_()
    d["b_gate_res"].zero_()
    d["w_gate_res2"].zero_()
    d["w_gate2"].zero_()
    bg2 = torch.full((1, ROUTER_OUT_DIM_ORDER3), -14.0, dtype=dtype, device=dev)
    bg2[0, target_mode] = +14.0
    bg2[0, K_MODES_ORDER3:] = +5.0
    d["b_gate2"] = bg2
    return d


def _apply_l91_constraints(
    cand: dict[str, torch.Tensor],
    stage: str,
    base_ref: dict[str, torch.Tensor],
    dtype: torch.dtype,
    dev: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Побитовая изоляция модулей без жёстких зажимов размера заявки (ch[0]) или таймеров отмен!
    Только мягкое разведение эшелонов глубины на этапах первичного обучения специалистов,
    чтобы Mode #0, Mode #1 и Mode #4 не слиплись в одинаковых клонов на -4%.
    """
    norm_d = _normalize_weights_order3(cand, dtype)
    out = {k: base_ref[k].detach().clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}

    spec_modes = {
        "l91a_mode0": 0,
        "l91b_mode1": 1,
        "l91c_mode2": 2,
        "l91d1_mode4": 4,
        "l91d2_mode7": 7,
    }
    if stage in spec_modes:
        tm = spec_modes[stage]
        w2_new = out["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        b2_new = out["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        wmup_new = out["w_moe_up"].float().view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
        bmup_new = out["b_moe_up"].float().view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
        wmdn_new = out["w_moe_dn"].float().view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()

        wm = norm_d["w2_moe"].float().to(dev).view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, tm, :].clone()
        bm = norm_d["b2_moe"].float().to(dev).view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, tm, :].clone()
        mup = norm_d["w_moe_up"].float().to(dev).view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, tm, :].clone()
        bup = norm_d["b_moe_up"].float().to(dev).view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, tm, :].clone()
        mdn = norm_d["w_moe_dn"].float().to(dev).view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)[tm].clone()

        # Рыночный тейкер-бай [48] всегда выключен (чистый Maker-вход 0.10% против 0.20% Taker + спред)
        wm[:, 48] = 0.0
        bm[0, 48] = -12.0
        mdn[:, 48] = 0.0

        if stage == "l91a_mode0":
            # Поверхностный эшелон вблизи BBO (retreat_b <= ~45 bp за счёт sniper_front [53] / малого [38]),
            # но размер заявки ch[0], темп [26:34, 50:52, 57] и выход [41..43, 54] 100% эндогенны!
            n38 = torch.linalg.vector_norm(wm[:, 38]).clamp(min=1e-6)
            wm[:, 38] = wm[:, 38] * min(1.0, float(0.80 / n38))
            bm[0, 38] = torch.clamp(bm[0, 38], -6.0, -2.1)  # 400bp * sigmoid(-2.1) <= 44 bp от Bid1!
            mdn[:, 38] = torch.clamp(mdn[:, 38], -0.15, 0.15)
            bm[0, 39] = torch.clamp(bm[0, 39], -6.0, -3.2)  # компактный веер слотов у края стакана
        else:
            wm[:, 55] = 0.0
            bm[0, 55] = -11.0
            mdn[:, 55] = 0.0
            if stage == "l91b_mode1":
                # Рабочий умеренный эшелон (-0.45%..-2.25%): не сваливается в -4.3% засаду Mode #4,
                # сохраняя 100% свободу в размере позиции ch[0], числе слотов [1:13] и темпе!
                n38 = torch.linalg.vector_norm(wm[:, 38]).clamp(min=1e-6)
                wm[:, 38] = wm[:, 38] * min(1.0, float(1.15 / n38))
                bm[0, 38] = torch.clamp(bm[0, 38], -1.80, +0.25)  # 4% * sigmoid ∈ [0.57%..2.25%]
                mdn[:, 38] = torch.clamp(mdn[:, 38], -0.22, 0.22)
                bm[0, 53] = torch.clamp(bm[0, 53], -6.0, -1.5)
            elif stage == "l91c_mode2":
                # Импульсный UP-Carry + рециклинг [56]: рабочий диапазон откупа -0.3%..-2.4% + память last_sell_px
                n38 = torch.linalg.vector_norm(wm[:, 38]).clamp(min=1e-6)
                wm[:, 38] = wm[:, 38] * min(1.0, float(1.25 / n38))
                bm[0, 38] = torch.clamp(bm[0, 38], -2.20, +0.35)
                mdn[:, 38] = torch.clamp(mdn[:, 38], -0.25, 0.25)
                bm[0, 56] = torch.clamp(bm[0, 56], -0.50, +4.50)

        w2_new[:, tm, :] = wm
        b2_new[:, tm, :] = bm
        wmup_new[:, tm, :] = mup
        bmup_new[:, tm, :] = bup
        wmdn_new[tm] = mdn

        out["w2_moe"] = w2_new.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
        out["b2_moe"] = b2_new.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
        out["w_moe_up"] = wmup_new.reshape(256, MOE_MLP_UP_DIM_ORDER3).to(dtype)
        out["b_moe_up"] = bmup_new.reshape(1, MOE_MLP_UP_DIM_ORDER3).to(dtype)
        out["w_moe_dn"] = wmdn_new.to(dtype)
        return out

    if stage == "l91e_regulator":
        # Разблокирован 36D Послотовый Нейро-Регулятор (принимающий g_mode[0:8] + ctx_4d[26:34])
        # + темповые каналы экспертов [26:34, 50:52, 57]!
        reg_keys = ("w_reg1", "b_reg1_b", "b_reg1_a", "w_reg_res1", "b_reg_res", "w_reg_res2", "w_reg2", "b_reg2_b", "b_reg2_a")
        for kr in reg_keys:
            out[kr] = norm_d[kr].to(device=dev, dtype=dtype)
        w2_new = out["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        b2_new = out["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        w2_c = norm_d["w2_moe"].float().to(dev).view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2_c = norm_d["b2_moe"].float().to(dev).view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        tempo_ch = [26, 27, 28, 29, 30, 31, 32, 33, 50, 51, 57]
        for km in (0, 1, 2, 4, 7):
            w2_new[:, km, tempo_ch] = w2_c[:, km, tempo_ch]
            b2_new[:, km, tempo_ch] = b2_c[:, km, tempo_ch]
        out["w2_moe"] = w2_new.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
        out["b2_moe"] = b2_new.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
        return out

    if stage in ("l91f_orchestrator", "l91f2_multimode", "l91g_joint"):
        router_keys = ("w_gate1", "b_gate1", "w_gate_res1", "b_gate_res", "w_gate_res2", "w_gate2", "b_gate2")
        for kr in router_keys:
            out[kr] = norm_d[kr].to(device=dev, dtype=dtype)

        if stage == "l91g_joint":
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

        # В главном Оркестраторе ОТКРЫТЫ ВСЕ 7 рабочих ролей (#0 Скальп, #1 Средняя, #2 Импульс,
        # #3 Бункер 0% кэша, #4 Глубокая засада, #5 Анти-каскад, #7 Разгрузчик); припаркован только пустой слот #6!
        wg2 = out["w_gate2"].clone()
        bg2 = out["b_gate2"].clone()
        wg2[:, 6] = 0.0
        wg2[:, K_MODES_ORDER3 + 6] = 0.0
        bg2[0, 6] = -14.0
        bg2[0, K_MODES_ORDER3 + 6] = +4.0
        out["w_gate2"] = wg2
        out["b_gate2"] = bg2
        return out

    return out


def _build_l91_seeds(
    base_ref: dict[str, torch.Tensor],
    stage: str,
    dtype: torch.dtype,
    dev: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """
    Формирует богатый стартовый спектр по доле капитала (exp ∈ [4%..85%]) и темпу (nervous -> medium -> calm),
    чтобы эволюция не застревала в старом смещении b2[0]=+3.5 (97%), а сама находила оптимальный сайзинг и темп!
    """
    seeds = [_apply_l91_constraints(base_ref, stage, base_ref, dtype, dev)]

    if stage == "l91a_mode0":
        # Спектр долей капитала для быстрой поверхностной моды: 4%, 8%, 15%, 26% + нервный/быстрый темп
        for exp_f, in_b, ret_b, in_a, unw, ms55 in (
            (0.04, +1.2, -3.8, +1.6, +2.6, +2.5),
            (0.08, +0.6, -3.0, +1.4, +2.2, +2.0),
            (0.15, +0.2, -2.5, +1.2, +2.4, +1.5),
            (0.26, -0.4, -2.2, +1.5, +3.0, +1.0),
            (0.07, +1.5, -4.5, +2.0, +1.8, +3.0),
        ):
            d = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            w2[:, 0, 0] *= 0.25
            b2[0, 0, 0] = math.log(exp_f / (1.0 - exp_f))
            b2[0, 0, 1] = +7.0
            b2[0, 0, 2:13] = -8.0
            b2[0, 0, 13] = +7.0
            b2[0, 0, 14:25] = -8.0
            b2[0, 0, 25] = -8.0
            b2[0, 0, 26:30] = +2.0   # нервный быстрый отклик регулятора
            b2[0, 0, 37] = in_b
            b2[0, 0, 38] = ret_b
            b2[0, 0, 39] = -4.8
            b2[0, 0, 41] = +1.0
            b2[0, 0, 42] = -0.35
            b2[0, 0, 43] = in_a
            b2[0, 0, 47] = +2.5
            b2[0, 0, 50:52] = +3.5   # мгновенная реакция alpha_bid/ask
            b2[0, 0, 53] = +2.2
            b2[0, 0, 54] = unw
            b2[0, 0, 55] = ms55
            b2[0, 0, 57] = -5.0      # нулевой гистерезис для верхушки стакана
            d["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
            d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
            seeds.append(_apply_l91_constraints(d, stage, base_ref, dtype, dev))

    elif stage == "l91b_mode1":
        # Спектр долей капитала для средней рабочей моды (-0.7%..-2.1%): 7%, 14%, 25%, 42%, 60% и 1..3 слота, средний темп
        for exp_f, b38, n_sl, alpha_t, hyst_t, z42 in (
            (0.07, -1.10, 2, +1.2, -1.5, -0.10),
            (0.14, -0.75, 2, +0.8, -0.8, +0.05),
            (0.25, -0.35, 2, +0.6, -0.4, +0.15),
            (0.42, +0.05, 3, +0.4, +0.0, +0.20),
            (0.60, +0.20, 2, +0.5, +0.2, +0.10),
        ):
            d = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            w2[:, 1, 0] *= 0.30
            b2[0, 1, 0] = math.log(exp_f / (1.0 - exp_f))
            b2[0, 1, 1:13] = -8.0
            for s_i in range(n_sl):
                b2[0, 1, 1 + s_i] = +4.5
            b2[0, 1, 13] = +6.5
            b2[0, 1, 14:25] = -7.0
            b2[0, 1, 25] = -8.0
            b2[0, 1, 26:30] = +0.5
            b2[0, 1, 37] = -2.0
            b2[0, 1, 38] = b38
            b2[0, 1, 39] = -3.4 if n_sl <= 2 else -2.2
            b2[0, 1, 41] = +1.1
            b2[0, 1, 42] = z42
            b2[0, 1, 43] = +1.3
            b2[0, 1, 47] = +2.5
            b2[0, 1, 50:52] = alpha_t
            b2[0, 1, 53] = -4.0
            b2[0, 1, 54] = +2.2
            b2[0, 1, 57] = hyst_t
            d["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
            d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
            seeds.append(_apply_l91_constraints(d, stage, base_ref, dtype, dev))

    elif stage == "l91c_mode2":
        # Спектр долей капитала для импульсной UP-Carry + рециклинг всплесков [56]: 10%, 20%, 35%, 52%
        for exp_f, b38, rec56, z42, unw in (
            (0.10, -1.40, +2.5, +0.35, +2.4),
            (0.20, -0.90, +3.0, +0.50, +2.0),
            (0.35, -0.45, +2.2, +0.65, +1.8),
            (0.52, -0.10, +1.8, +0.40, +2.6),
        ):
            d = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            w2[:, 2, 0] *= 0.35
            b2[0, 2, 0] = math.log(exp_f / (1.0 - exp_f))
            b2[0, 2, 1:3] = +4.5
            b2[0, 2, 3:13] = -7.5
            b2[0, 2, 13:15] = +5.0
            b2[0, 2, 15:25] = -7.0
            b2[0, 2, 26:30] = +0.6
            b2[0, 2, 38] = b38
            b2[0, 2, 39] = -3.0
            b2[0, 2, 41] = +0.8
            b2[0, 2, 42] = z42
            b2[0, 2, 43] = +0.9
            b2[0, 2, 47] = +2.5
            b2[0, 2, 50:52] = +0.9
            b2[0, 2, 53] = -4.0
            b2[0, 2, 54] = unw
            b2[0, 2, 56] = rec56
            b2[0, 2, 57] = -0.6
            d["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
            d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
            seeds.append(_apply_l91_constraints(d, stage, base_ref, dtype, dev))

    elif stage == "l91d1_mode4":
        # Seed #0 = эталон Mode #4; Seeds #1..#4 варьируют "живое спокойствие" (hyst_band [57], ctx_4d [26:30], alpha [50:52]),
        # чтобы глубокая мода не засыпала при уходе рынка вверх (Study 0.3)!
        for alpha_t, hyst_t, ctx_t in (
            (+0.6, -0.5, +0.3),
            (+0.2, +0.2, +0.0),
            (-0.4, +0.8, -0.3),
            (+1.0, -1.0, +0.6),
        ):
            d = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2[0, 4, 50:52] = alpha_t
            b2[0, 4, 57] = hyst_t
            b2[0, 4, 26:30] = ctx_t
            d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
            seeds.append(_apply_l91_constraints(d, stage, base_ref, dtype, dev))

    elif stage in ("l91e_regulator", "l91f_orchestrator"):
        if stage == "l91f_orchestrator":
            # 1) Zero-Regression Anchor от боевого роутера L89 поверх новых специалистов L91 + 36D регулятора L91e:
            pol_tmp = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=torch.device("cpu"))
            live_ck = pol_tmp.load_initial_champion(BASE_LIVE_PATH)
            d_l89 = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            for kr in ("w_gate1", "b_gate1", "w_gate_res1", "b_gate_res", "w_gate_res2", "w_gate2", "b_gate2"):
                d_l89[kr] = live_ck[kr].to(device=dev, dtype=dtype)
            seeds.append(_apply_l91_constraints(d_l89, stage, base_ref, dtype, dev))

            # 2) Хирургическое раскрытие роутера L89: сохраняем ядро Mode #4/#7 и плавно открываем рабочую среднюю Mode #1, импульс Mode #2 и скальп Mode #0!
            for b0_o, b1_o, b2_o, b3_o in (
                (-3.8, -1.8, -2.6, -2.5),
                (-2.8, -0.6, -1.6, -1.8),
                (-2.0, +0.4, -0.8, -1.4),
            ):
                dh = {k: d_l89[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
                wg1_h = dh["w_gate1"].float().clone()
                wg2_h = dh["w_gate2"].float().clone()
                bg2_h = dh["b_gate2"].float().clone()
                wg1_h[:, 92:116] *= 0.15
                wg1_h[336, 92:98] = +3.2; wg1_h[364, 92:98] = -3.0; wg1_h[350, 92:98] = +1.4   # Mode #0: wide spread & calm
                wg1_h[364, 98:104] = -2.8; wg1_h[258, 98:104] = +1.6; wg1_h[350, 98:104] = +1.4 # Mode #1: calm working market
                wg1_h[370, 104:110] = +2.8; wg1_h[373, 104:110] = +1.6; wg1_h[257, 104:110] = -2.2 # Mode #2: UP-Carry
                wg1_h[350, 110:116] = -3.6; wg1_h[364, 110:116] = +2.8                          # Mode #3: BO drop bunker
                for m_i, sl_h, bv, btv in (
                    (0, slice(92, 98), b0_o, +3.2),
                    (1, slice(98, 104), b1_o, +0.9),
                    (2, slice(104, 110), b2_o, +0.9),
                    (3, slice(110, 116), b3_o, +2.6),
                ):
                    wg2_h[:, m_i] = 0.0
                    wg2_h[sl_h, m_i] = +2.20
                    bg2_h[0, m_i] = bv
                    bg2_h[0, K_MODES_ORDER3 + m_i] = btv
                bg2_h[0, K_MODES_ORDER3 + 4] = -0.6  # спокойная инерция для глубокой Mode #4
                dh["w_gate1"] = wg1_h.to(dtype)
                dh["w_gate2"] = wg2_h.to(dtype)
                dh["b_gate2"] = bg2_h.to(dtype)
                seeds.append(_apply_l91_constraints(dh, stage, base_ref, dtype, dev))

        # Сиды сбалансированного мультимодального запуска (с полноценным весом глубокой Mode #4 b4=+2.2..+3.0 и средней Mode #1)!
        for b0, b1, b2_m, b3, b4, b7 in (
            (-2.4, +0.8, -0.8, -1.6, +2.6, +1.2),
            (-1.4, +1.4, +0.0, -1.2, +2.2, +1.4),
            (-3.0, +0.4, -1.2, -1.8, +3.2, +1.0),
            (-0.8, +1.2, +0.2, -0.8, +1.8, +1.5),
        ):
            d = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            wg1 = d["w_gate1"].float().clone()
            wg2 = d["w_gate2"].float().clone()
            bg2 = d["b_gate2"].float().clone()

            wg1[:, 88:124] *= 0.15
            # [336]=spread, [350]=bo_ret, [364]=crash900, [370]=ret30m, [373]=cvd, [257]=pos_ratio, [258]=free_cash
            # Mode #0: широкий спред + низкий риск + быстрый beta_gate (+3.5)
            wg1[336, 88:94] = +3.4; wg1[364, 88:94] = -3.0; wg1[350, 88:94] = +1.5
            # Mode #1: спокойный/умеренный рынок (низкий crash900, свободный кэш) + средний beta_gate (+0.8)
            wg1[364, 94:100] = -2.8; wg1[258, 94:100] = +1.8; wg1[350, 94:100] = +1.4
            # Mode #2: импульс роста ret30m + положительный CVD + свободный портфель
            wg1[370, 100:106] = +3.0; wg1[373, 100:106] = +1.8; wg1[257, 100:106] = -2.2; wg1[364, 100:106] = -2.4
            # Mode #3: Бункер 0% кэша при падении BigONE или высоком риске обвала
            wg1[350, 106:112] = -3.6; wg1[364, 106:112] = +2.6; wg1[257, 106:112] = -1.5
            # Mode #4: глубокая засада при повышенной турбулентности + плавный спокойный beta_gate (-1.0)
            wg1[364, 112:118] = +1.6; wg1[336, 112:118] = +1.0
            # Mode #7: разгрузка набранной позиции (высокий pos_ratio[257]) + быстрый beta_gate (+2.5)
            wg1[257, 118:124] = +3.6; wg1[364, 118:124] = +1.4

            for m_i, sl_h, bias_v, beta_v in (
                (0, slice(88, 94), b0, +3.2),     # быстрая мода — нервное быстрое переключение
                (1, slice(94, 100), b1, +0.8),    # средняя мода — умеренная частота смены решений
                (2, slice(100, 106), b2_m, +0.8),
                (3, slice(106, 112), b3, +2.6),   # защитный бункер — включается быстро при опасности
                (4, slice(112, 118), b4, -0.8),   # глубокая мода — спокойная выдержка без суеты
                (7, slice(118, 124), b7, +2.4),   # разгрузчик — оперативный приём позиции
            ):
                wg2[:, m_i] *= 0.25
                wg2[sl_h, m_i] = +2.10
                bg2[0, m_i] = bias_v
                bg2[0, K_MODES_ORDER3 + m_i] = beta_v

            d["w_gate1"] = wg1.to(dtype)
            d["w_gate2"] = wg2.to(dtype)
            d["b_gate2"] = bg2.to(dtype)
            if stage == "l91e_regulator":
                # Для обучения регулятора фиксируем мультимодальный роутер сида #1 на base_ref, чтобы все 5 мод активно работали!
                seeds.append(_apply_l91_constraints(d, stage, d, dtype, dev))
            else:
                seeds.append(_apply_l91_constraints(d, stage, base_ref, dtype, dev))

    elif stage == "l91f2_multimode":
        # Раскрытие быстрых/средних мод (#0, #1, #2, #3) поверх чемпиона L91f (+2.68%/d, где bg2[4]=-0.13, bg2[7]=+0.21, а [0..3] были заперты на -11.0):
        # Выводим bg2[0..3] из мёртвой зоны -11.0 в рабочую пороговую окрестность [-2.4 .. -0.4] под жёстким щитом crash900/BigONE!
        for b0_s, b1_s, b2_s, b3_s, sh_g in (
            (-2.4, -1.2, -1.8, -1.6, 4.5),
            (-1.8, -0.5, -1.2, -1.2, 5.0),
            (-1.2, +0.1, -0.6, -0.8, 5.5),
            (-2.8, -0.2, -1.4, -1.4, 6.0),
            (-1.5, +0.5, -0.3, -0.6, 5.5),
            (-0.9, -0.8, -1.0, -1.0, 5.0),
        ):
            dm = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            wg1_m = dm["w_gate1"].float().clone()
            wg2_m = dm["w_gate2"].float().clone()
            bg2_m = dm["b_gate2"].float().clone()

            # Используем тихие нейроны 104..128 первого каскада роутера, чтобы не задеть обученные пути Mode #4/#7!
            wg1_m[:, 104:128] = 0.0
            # Mode #0 (104:110): широкий спред [336], спокойный BigONE [350], низкий crash900 [364], пустой портфель [257]
            wg1_m[336, 104:110] = +3.2; wg1_m[350, 104:110] = +2.2; wg1_m[364, 104:110] = -sh_g; wg1_m[257, 104:110] = -3.0
            # Mode #1 (110:116): спокойный рынок (низкий crash900 [364], спокойный BigONE [350], свободный кэш [258])
            wg1_m[364, 110:116] = -sh_g; wg1_m[350, 110:116] = +2.4; wg1_m[258, 110:116] = +1.8; wg1_m[257, 110:116] = -2.2
            # Mode #2 (116:122): растущий тренд [370] + положительный CVD [373] + низкий crash900 [364]
            wg1_m[370, 116:122] = +3.0; wg1_m[373, 116:122] = +1.8; wg1_m[364, 116:122] = -sh_g; wg1_m[257, 116:122] = -2.5
            # Mode #3 (122:128): Бункер 0% кэша при падении BigONE [350] и всплеске crash900 [364]
            wg1_m[350, 122:128] = -4.2; wg1_m[364, 122:128] = +3.2; wg1_m[257, 122:128] = -2.0

            for m_i, sl_h, bv, btv in (
                (0, slice(104, 110), b0_s, +3.4),
                (1, slice(110, 116), b1_s, +1.0),
                (2, slice(116, 122), b2_s, +1.0),
                (3, slice(122, 128), b3_s, +2.8),
            ):
                wg2_m[:, m_i] = 0.0
                wg2_m[sl_h, m_i] = +1.80
                bg2_m[0, m_i] = bv
                bg2_m[0, K_MODES_ORDER3 + m_i] = btv

            dm["w_gate1"] = wg1_m.to(dtype)
            dm["w_gate2"] = wg2_m.to(dtype)
            dm["b_gate2"] = bg2_m.to(dtype)
            seeds.append(_apply_l91_constraints(dm, stage, base_ref, dtype, dev))

    elif stage == "l91g_joint":
        # Seed #0 = рекордный L91f_orchestrator_winner.pt (+312.29bp, +2.68%/d Calib, +2.43%/d Harsh)!
        # Доп. сиды #1..#3 деликатно подключают лёгкий скальп Mode #0 (exp=4.8%) на широком спреде и Mode #1 на штиле,
        # НЕ затирая 85%+ ядро Mode #4/#7 на сквозных всех режимах!
        for b0_j, b1_j in ((-2.6, -2.8), (-1.9, -2.2), (-1.4, -1.7)):
            dj = {k: base_ref[k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            wg1_j = dj["w_gate1"].float().clone()
            wg2_j = dj["w_gate2"].float().clone()
            bg2_j = dj["b_gate2"].float().clone()
            wg1_j[:, 116:128] = 0.0
            wg1_j[336, 116:122] = +3.5; wg1_j[364, 116:122] = -6.0; wg1_j[350, 116:122] = +2.0; wg1_j[257, 116:122] = -3.0
            wg1_j[364, 122:128] = -6.5; wg1_j[350, 122:128] = +2.4; wg1_j[257, 122:128] = -3.0
            wg2_j[:, 0] = 0.0; wg2_j[116:122, 0] = +1.60; bg2_j[0, 0] = b0_j; bg2_j[0, K_MODES_ORDER3 + 0] = +3.4
            wg2_j[:, 1] = 0.0; wg2_j[122:128, 1] = +1.40; bg2_j[0, 1] = b1_j; bg2_j[0, K_MODES_ORDER3 + 1] = +1.2
            dj["w_gate1"] = wg1_j.to(dtype)
            dj["w_gate2"] = wg2_j.to(dtype)
            dj["b_gate2"] = bg2_j.to(dtype)
            seeds.append(_apply_l91_constraints(dj, stage, base_ref, dtype, dev))

    return seeds


def _island_worker(
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

    # ИСПРАВЛЕНО: включаем все данные вплоть до 26.11 (пропуская карантин q_lo..q_hi в цикле окон),
    # оставляя последние 20 часов неприкосновенным слепым OOS!
    oos_span = int(20.0 * 3600)
    usable_hi = total_rows - oos_span - 1900

    mid_sub = 0.5 * (b1_sub + a1_sub)
    sp_bp = (a1_sub - b1_sub) / np.maximum(1e-4, mid_sub) * 1e4
    ret30m = np.zeros(total_rows, dtype=np.float32)
    ret30m[1800:] = mid_sub[1800:] / np.maximum(1e-4, mid_sub[:-1800]) - 1.0
    crash900 = mkt_sub[:, 81]

    if stage in ("l91a_mode0", "l91b_mode1", "l91c_mode2", "l91d1_mode4", "l91d2_mode7"):
        win_len = 7201   # 2ч окна для прицельного обучения специалистов
    else:
        win_len = 10801  # 3ч окна (+ 6ч слайды) для регулятора, оркестратора и совместной притирки

    grid_starts = list(range(1800, usable_hi - win_len, 900))
    cand_windows: list[tuple[int, float, float]] = []
    for s in grid_starts:
        e = s + win_len
        if not (e < q_lo or s > q_hi):
            continue
        m_seg = mid_sub[s:e]
        split_tail = s + int(0.65 * win_len)
        dip_core = float(np.min(mid_sub[s:split_tail] / np.maximum.accumulate(mid_sub[s:split_tail]) - 1.0))
        dip_full = float(np.min(m_seg / np.maximum.accumulate(m_seg) - 1.0))
        up_core = float(np.max(ret30m[s:split_tail]))
        tail_ret = float(mid_sub[e - 1] / max(1e-4, mid_sub[split_tail]) - 1.0)
        cr_mean = float(np.mean(crash900[s:split_tail]))
        f_sp40 = float(np.mean(sp_bp[s:e] >= 40.0))

        if stage == "l91a_mode0":
            # Окна, где есть и широкие спреды (>=40bp >= 12%), и обычные фазы (чтобы учиться не застревать!)
            if f_sp40 >= 0.12 and dip_full > -0.065:
                cand_windows.append((s, 0.0, 0.0))
        elif stage == "l91b_mode1":
            # Спокойные и умеренно-дышащие окна (-0.65%..-2.8% откаты без катастрофического обвала) — наш ежедневный рынок!
            if -0.032 <= dip_core <= -0.0065 and dip_full > -0.045 and cr_mean < 0.45:
                cand_windows.append((s, 0.0, 0.0))
        elif stage == "l91c_mode2":
            # Растущие волны + фаза остывания/передачи в хвосте
            if up_core >= 0.011 and tail_ret >= -0.035:
                cand_windows.append((s, 0.0, 0.0))
        elif stage == "l91d1_mode4":
            # Глубокие проливы И дрейфующие вверх окна (чтобы Mode #4 учился подтягивать сетку, не засыпая!)
            if dip_full <= -0.026 or (up_core >= 0.018 and dip_full <= -0.018):
                cand_windows.append((s, 0.0, 0.0))
        elif stage == "l91d2_mode7":
            # Приёмник разгрузки стартует с разной величиной переданной позиции (и малый 14% от скальпа/средней, и крупный 55% от глубокой!)
            rhos = (0.14, 0.28, 0.46, 0.64)
            offs = (-0.002, +0.003, -0.001, +0.005)
            idx_c = len(cand_windows) % 4
            cand_windows.append((s, rhos[idx_c], offs[idx_c]))
        elif stage == "l91f2_multimode":
            # Фокус 3ч корзины на спокойных/умеренных/растущих окнах (без обвалов глубже -3.2%, где Mode #4 молчит!),
            # тогда как вторая половина оценки (val_slide_basket 6ч) одновременно страхует все обвалы!
            if dip_full > -0.032 and cr_mean < 0.42:
                cand_windows.append((s, 0.0, 0.0))
        else:
            cand_windows.append((s, 0.0, 0.0))

    assert len(cand_windows) >= 28, f"Too few windows ({len(cand_windows)}) for {stage}"
    split_i = int(0.70 * len(cand_windows))
    tr_pool = cand_windows[: max(18, split_i - 2)]
    val_pool = cand_windows[split_i:]
    n_bask = 20
    val_idx = [int(i) for i in np.linspace(0, len(val_pool) - 1, n_bask)]
    val_basket = [(f"VAL{k:02d}", val_pool[idx][0], val_pool[idx][1], val_pool[idx][2]) for k, idx in enumerate(val_idx)]

    val_slide_basket = None
    if stage in ("l91e_regulator", "l91f_orchestrator", "l91f2_multimode", "l91g_joint"):
        slide_len = 21601
        raw_s = [int(x) for x in np.linspace(1800, usable_hi - slide_len, 36)]
        clean_s = [ws for ws in raw_s if (ws + slide_len < q_lo or ws > q_hi)]
        s_sel = [clean_s[int(i)] for i in np.linspace(0, len(clean_s) - 1, n_bask)]
        val_slide_basket = [(f"SLD{k:02d}", ws, 0.0, 0.0) for k, ws in enumerate(s_sel)]

    pol = MultimodalMoEPolicyOrder3(n_groups=n_groups, n_budgets=6, device=dev)
    base_cpu = pol.load_initial_champion(Path(src_ckpt_str))
    base_ref = {k: base_cpu[k].to(device=dev, dtype=pol.dtype) for k in pol.PARAM_KEYS}

    spec_modes = {"l91a_mode0": 0, "l91b_mode1": 1, "l91c_mode2": 2, "l91d1_mode4": 4, "l91d2_mode7": 7}
    if stage in spec_modes:
        base_ref = _lock_router_single_mode(base_ref, spec_modes[stage], pol.dtype, dev)

    seeds = _build_l91_seeds(base_ref, stage, pol.dtype, dev)
    if stage == "l91e_regulator" and len(seeds) >= 2:
        # Фиксируем сбалансированный мультимодальный роутер на время урока 36D регулятора, чтобы все моды тренировались одновременно!
        base_ref = {k: seeds[1][k].clone() for k in pol.PARAM_KEYS}
        seeds = [_apply_l91_constraints(s, stage, base_ref, pol.dtype, dev) for s in seeds]

    inv_pen = 1.55 if stage in ("l91a_mode0", "l91b_mode1", "l91c_mode2", "l91f_orchestrator", "l91g_joint") else 0.95

    def eval_l91_batch(cands: list[dict], bask: list, w_l: int, slide_b: list | None = None, dual_stress: bool = True) -> list[dict]:
        orig_n = len(cands)
        c_in = list(cands) + [cands[0]] * (32 - orig_n) if orig_n <= 32 else cands
        pad_m = orig_n > 32
        rc = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=bask, win_len=w_l, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=(not dual_stress), enforce_activity_floor=False, risk_inv_pen_w=inv_pen,
        )[:orig_n]
        rh = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=bask, win_len=w_l, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=True, enforce_activity_floor=False, risk_inv_pen_w=inv_pen,
        )[:orig_n] if dual_stress else rc
        fit_main = [compute_l91_fitness(a, b, w_l, len(bask), stage) for a, b in zip(rc, rh)]
        if slide_b is None:
            return fit_main

        sc = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=slide_b, win_len=21601, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=False, enforce_activity_floor=False, risk_inv_pen_w=1.60,
        )[:orig_n]
        sh = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
            sv_all=sv_gpu, bv_all=bv_gpu, cand_states=c_in, val_windows=slide_b, win_len=21601, tick_sz=0.0001,
            device=dev, pad_to_max_c=pad_m, harsh_stress=True, enforce_activity_floor=False, risk_inv_pen_w=1.60,
        )[:orig_n] if dual_stress else sc
        fit_sld = [compute_l91_fitness(a, b, 21601, len(slide_b), stage) for a, b in zip(sc, sh)]

        merged = []
        for fm, fs in zip(fit_main, fit_sld):
            merged.append({
                "l91_score_bp": 0.45 * fm["l91_score_bp"] + 0.55 * fs["l91_score_bp"],
                "geo_mix_bp": 0.45 * fm["geo_mix_bp"] + 0.55 * fs["geo_mix_bp"],
                "ev_day_cal_pct": 0.45 * fm["ev_day_cal_pct"] + 0.55 * fs["ev_day_cal_pct"],
                "ev_day_har_pct": 0.45 * fm["ev_day_har_pct"] + 0.55 * fs["ev_day_har_pct"],
                "dd_max_pct": max(fm["dd_max_pct"], fs["dd_max_pct"]),
                "fills": fm["fills"] + fs["fills"],
                "cnl_per_hr": 0.50 * (fm["cnl_per_hr"] + fs["cnl_per_hr"]),
                "shortfall_bp": 0.45 * fm["shortfall_bp"] + 0.55 * fs["shortfall_bp"],
                "mode_probs": fs["mode_probs"],
            })
        return merged

    seed_res = eval_l91_batch(seeds, val_basket, win_len, slide_b=val_slide_basket, dual_stress=True)
    best_i = int(np.argmax([r["l91_score_bp"] for r in seed_res]))
    champ_state = {k: v.clone() for k, v in seeds[best_i].items()}
    best_sc = float(seed_res[best_i]["l91_score_bp"])
    sr0 = seed_res[best_i]
    log_w(
        f"🌱 Start Seed #{best_i}/{len(seeds)} ({len(cand_windows)}w): L91={best_sc:+.2f}bp | "
        f"EV/d={sr0['ev_day_cal_pct']:+.3f}%/{sr0['ev_day_har_pct']:+.3f}% | DD={sr0['dd_max_pct']:.2f}% | "
        f"Fills={sr0['fills']:.1f} | Cnl/h={sr0['cnl_per_hr']:.1f}"
    )

    def save_champ(it_n: int, vr: dict) -> None:
        meta = {
            "arch_version": "order3_v2",
            "wf_version": f"l91_{stage}_{world_mode}",
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
    rng = np.random.default_rng(20261291 + island_id * 104729)
    half = (n_groups - 2) // 2
    r_view = pol.ring_mult_half

    for it in range(1, max_iters + 1):
        if (time.time() - t0) / 3600.0 >= max_hours:
            break
        t_it = time.time()

        # Межмировая миграция ЦЕЛОСТНЫХ сетей каждые 6 итераций (без линейного смешивания весов!)
        if it % 6 == 0:
            nb_p = out_dir / f"island_{1 - island_id}_best.pt"
            if nb_p.exists():
                try:
                    nb_cpu = pol.load_initial_champion(nb_p)
                    nb_st = _apply_l91_constraints({k: nb_cpu[k].to(dev, pol.dtype) for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
                    nb_r = eval_l91_batch([nb_st], val_basket, win_len, slide_b=val_slide_basket, dual_stress=True)[0]
                    if float(nb_r["l91_score_bp"]) > best_sc + 0.05:
                        champ_state = nb_st
                        best_sc = float(nb_r["l91_score_bp"])
                        save_champ(it, nb_r)
                        log_w(f"🧬 [It {it:02d}] Миграция обновила лидера -> L91={best_sc:+.2f}bp (EV/d={nb_r['ev_day_cal_pct']:+.3f}%, DD={nb_r['dd_max_pct']:.2f}%)")
                except Exception:
                    pass

        prog = it / max(1.0, float(max_iters))
        sig_sc = 1.15 if prog < 0.45 else (0.68 if prog < 0.80 else 0.36)
        if stage == "l91g_joint":
            sig_sc *= 0.28

        for k in pol.PARAM_KEYS:
            pol.__dict__[k][:] = champ_state[k].unsqueeze(0)

        if stage in spec_modes:
            tm = spec_modes[stage]
            ew_lin = torch.randn((half, 256, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.024 * sig_sc * r_view.view(half, 1, 1))
            eb_lin = torch.randn((half, 1, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.085 * sig_sc * r_view.view(half, 1, 1))
            # Повышенная свобода поиска по каналу размера экспозиции [0], резерва [47] и темпа [50:52, 57]
            eb_lin[:, :, [0, 47, 50, 51, 57]] *= 1.80
            ew_up = torch.randn((half, 256, 2 * MOE_MLP_H_ORDER3), device=dev) * (0.042 * sig_sc * r_view.view(half, 1, 1))
            eb_up = torch.randn((half, 1, 2 * MOE_MLP_H_ORDER3), device=dev) * (0.062 * sig_sc * r_view.view(half, 1, 1))
            ew_dn = torch.randn((half, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))

            pol.w2_moe[2 : 2 + 2 * half].view(2 * half, 256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, :, tm, :].add_(
                torch.stack([ew_lin, -ew_lin], dim=1).reshape(2 * half, 256, EXPERT_OUT_DIM_ORDER3)
            )
            pol.b2_moe[2 : 2 + 2 * half].view(2 * half, 1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, :, tm, :].add_(
                torch.stack([eb_lin, -eb_lin], dim=1).reshape(2 * half, 1, EXPERT_OUT_DIM_ORDER3)
            )
            pol.w_moe_up[2 : 2 + 2 * half].view(2 * half, 256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, :, tm, :].add_(
                torch.stack([ew_up, -ew_up], dim=1).reshape(2 * half, 256, 2 * MOE_MLP_H_ORDER3)
            )
            pol.b_moe_up[2 : 2 + 2 * half].view(2 * half, 1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, :, tm, :].add_(
                torch.stack([eb_up, -eb_up], dim=1).reshape(2 * half, 1, 2 * MOE_MLP_H_ORDER3)
            )
            pol.w_moe_dn[2 : 2 + 2 * half, tm].add_(torch.stack([ew_dn, -ew_dn], dim=1).reshape(2 * half, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3))

        if stage in ("l91e_regulator", "l91g_joint"):
            er_w1 = torch.randn((half, REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3), device=dev) * (0.036 * sig_sc * r_view.view(half, 1, 1))
            er_w1[:, 28:36, :] *= 2.40  # акцент на входы g_mode[0:8] внутри 36D регулятора!
            er_b1b = torch.randn((half, 1, REG_H_DIM_ORDER3), device=dev) * (0.055 * sig_sc * r_view.view(half, 1, 1))
            er_b1a = torch.randn((half, 1, REG_H_DIM_ORDER3), device=dev) * (0.055 * sig_sc * r_view.view(half, 1, 1))
            er_rs1 = torch.randn((half, REG_H_DIM_ORDER3, 64), device=dev) * (0.042 * sig_sc * r_view.view(half, 1, 1))
            er_brs = torch.randn((half, 1, 64), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            er_rs2 = torch.randn((half, 32, REG_H_DIM_ORDER3), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
            er_w2 = torch.randn((half, REG_H_DIM_ORDER3, 2), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            er_b2b = torch.randn((half, 1, 2), device=dev) * (0.085 * sig_sc * r_view.view(half, 1, 1))
            er_b2a = torch.randn((half, 1, 2), device=dev) * (0.085 * sig_sc * r_view.view(half, 1, 1))

            for attr_n, eps_t in (
                ("w_reg1", er_w1), ("b_reg1_b", er_b1b), ("b_reg1_a", er_b1a),
                ("w_reg_res1", er_rs1), ("b_reg_res", er_brs), ("w_reg_res2", er_rs2),
                ("w_reg2", er_w2), ("b_reg2_b", er_b2b), ("b_reg2_a", er_b2a),
            ):
                tgt = pol.__dict__[attr_n][2 : 2 + 2 * half]
                tgt.add_(torch.stack([eps_t, -eps_t], dim=1).reshape_as(tgt))

            if stage == "l91e_regulator":
                eb_t = torch.randn((half, 1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.070 * sig_sc * r_view.view(half, 1, 1, 1))
                pol.b2_moe[2 : 2 + 2 * half].view(2 * half, 1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).add_(
                    torch.stack([eb_t, -eb_t], dim=1).reshape(2 * half, 1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
                )

        if stage in ("l91f_orchestrator", "l91f2_multimode", "l91g_joint"):
            ew1 = torch.randn((half, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.028 * sig_sc * r_view.view(half, 1, 1))
            ew1[:, 256:, :] *= 2.50  # акцент на портфель (pos_ratio/cash), очередь, 64D память и 30D риск-сигналы
            eb1 = torch.randn((half, 1, ROUTER_H_DIM_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            er1 = torch.randn((half, ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3), device=dev) * (0.042 * sig_sc * r_view.view(half, 1, 1))
            ebr = torch.randn((half, 1, 2 * ROUTER_RES_H_ORDER3), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            er2 = torch.randn((half, ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.048 * sig_sc * r_view.view(half, 1, 1))
            ew2 = torch.randn((half, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.048 * sig_sc * r_view.view(half, 1, 1))
            eb2 = torch.randn((half, 1, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.115 * sig_sc * r_view.view(half, 1, 1))
            if stage == "l91f2_multimode":
                eb2[:, :, [0, 1, 2, 3]] *= 2.40
                ew2[:, :, [0, 1, 2, 3]] *= 1.80

            for attr_n, eps_t in (
                ("w_gate1", ew1), ("b_gate1", eb1), ("w_gate_res1", er1),
                ("b_gate_res", ebr), ("w_gate_res2", er2), ("w_gate2", ew2), ("b_gate2", eb2),
            ):
                tgt = pol.__dict__[attr_n][2 : 2 + 2 * half]
                tgt.add_(torch.stack([eps_t, -eps_t], dim=1).reshape_as(tgt))

            if stage == "l91g_joint":
                ew_up = torch.randn((half, 256, MOE_MLP_UP_DIM_ORDER3), device=dev) * (0.030 * sig_sc * r_view.view(half, 1, 1))
                ew_dn = torch.randn((half, K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.032 * sig_sc * r_view.view(half, 1, 1, 1))
                pol.w_moe_up[2 : 2 + 2 * half].add_(torch.stack([ew_up, -ew_up], dim=1).reshape_as(pol.w_moe_up[2 : 2 + 2 * half]))
                pol.w_moe_dn[2 : 2 + 2 * half].add_(torch.stack([ew_dn, -ew_dn], dim=1).reshape_as(pol.w_moe_dn[2 : 2 + 2 * half]))

        for gi in range(2, 2 + 2 * half):
            cl = _apply_l91_constraints({k: pol.__dict__[k][gi] for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
            for k in pol.PARAM_KEYS:
                pol.__dict__[k][gi].copy_(cl[k])

        tr_sel = rng.choice(len(tr_pool), size=n_bask, replace=(len(tr_pool) < n_bask))
        tr_bask = [(f"TR{k:02d}", tr_pool[int(idx)][0], tr_pool[int(idx)][1], tr_pool[int(idx)][2]) for k, idx in enumerate(tr_sel)]
        pop_cands = pol.get_candidate_states()
        # Для высокой скорости поиска на Train используем Harsh-стресс (самый строгий фильтр), а на Val — полный Dual Calib+Harsh (+ 6h Slide)!
        tr_res = eval_l91_batch(pop_cands, tr_bask, win_len, slide_b=None, dual_stress=False)
        tr_scs = np.array([float(r["l91_score_bp"]) for r in tr_res], dtype=np.float64)

        es_raw = pol.build_openai_es_candidates(champ_state, tr_scs, lr_scales=(0.50, 1.00, 1.65))
        es_cands = [_apply_l91_constraints(c, stage, base_ref, pol.dtype, dev) for c in es_raw]
        top_muts = [_apply_l91_constraints(pop_cands[int(i)], stage, base_ref, pol.dtype, dev) for i in np.argsort(tr_scs)[::-1][:4]]

        shortlist = [champ_state] + es_cands + top_muts
        val_res = eval_l91_batch(shortlist, val_basket, win_len, slide_b=val_slide_basket, dual_stress=True)
        bv_i = int(np.argmax([float(r["l91_score_bp"]) for r in val_res]))
        cand_sc = float(val_res[bv_i]["l91_score_bp"])

        if cand_sc > best_sc + 0.02:
            gain = cand_sc - best_sc
            best_sc = cand_sc
            champ_state = {k: v.clone() for k, v in shortlist[bv_i].items()}
            vr = val_res[bv_i]
            save_champ(it, vr)
            log_w(
                f"⭐ [It {it:02d} | +{gain:.2f}bp] L91={best_sc:+.2f}bp | EV/d={vr['ev_day_cal_pct']:+.3f}%/{vr['ev_day_har_pct']:+.3f}% "
                f"| DD={vr['dd_max_pct']:.2f}% | Fills={vr['fills']:.1f} | Cnl/h={vr['cnl_per_hr']:.1f} "
                f"| Modes={[round(x,2) for x in vr.get('mode_probs',[])]} | {time.time()-t_it:.1f}s"
            )
        elif it % 5 == 0:
            log_w(f"   [It {it:02d}/{max_iters}] TrainTop={np.max(tr_scs):+.2f}bp | ValChamp={best_sc:+.2f}bp | {time.time()-t_it:.1f}s")

    log_w(f"🏁 Done [{stage}] ({world_mode.upper()}) in {(time.time()-t0)/60:.1f}m | Best L91={best_sc:+.2f}bp")


def run_dual_gpu_stage(stage: str, src_ckpt: Path, save_subdir: str, n_groups: int, max_iters: int, max_hours: float) -> tuple[Path, Path]:
    _log_m(f"▶️ СТАРТ ЭТАПА [{stage}] на 2x GPU (cuda:0=RAW, cuda:1=DEC | groups={n_groups}, iters={max_iters}, limit={max_hours:.2f}h)")
    ctx = mp.get_context("spawn")
    p0 = ctx.Process(target=_island_worker, args=(stage, str(src_ckpt), save_subdir, 0, "cuda:0", "raw", n_groups, max_iters, max_hours))
    p1 = ctx.Process(target=_island_worker, args=(stage, str(src_ckpt), save_subdir, 1, "cuda:1", "dec", n_groups, max_iters, max_hours))
    p0.start(); p1.start(); p0.join(); p1.join()
    if p0.exitcode != 0 or p1.exitcode != 0:
        raise RuntimeError(f"Stage {stage} failed: rc0={p0.exitcode}, rc1={p1.exitcode}")
    out_dir = CHAMP_DIR / save_subdir
    return out_dir / "island_0_best.pt", out_dir / "island_1_best.pt"


def select_intact_cross_world_winner(p_raw: Path, p_dec: Path, dst: Path, stage: str, tag: str) -> dict:
    """
    Кросс-мировой 4-физический отбор ЦЕЛОСТНОГО победителя (DEC_Calib, DEC_Harsh, RAW_Calib, RAW_Harsh)
    + проверка на слепом 20ч OOS БЕЗ линейного усреднения скрытых SwiGLU-слоёв!
    """
    dev = torch.device("cuda:0")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=torch.device("cpu"))
    c_r = pol.load_initial_champion(p_raw)
    c_d = pol.load_initial_champion(p_dec)
    spec_modes = {"l91a_mode0": 0, "l91b_mode1": 1, "l91c_mode2": 2, "l91d1_mode4": 4, "l91d2_mode7": 7}
    if stage in spec_modes:
        c_r = _lock_router_single_mode(c_r, spec_modes[stage], torch.float32, torch.device("cpu"))
        c_d = _lock_router_single_mode(c_d, spec_modes[stage], torch.float32, torch.device("cpu"))

    sub = 3016277
    cands_gpu = [{k: v.to(dev, torch.bfloat16) for k, v in st.items()} for st in (c_r, c_d)]
    c_pad = cands_gpu + [cands_gpu[0]] * 30

    scores = [0.0, 0.0]
    oos_ev = [0.0, 0.0]
    rep = {"RAW_island": {}, "DEC_island": {}}

    for wm in ("dec", "raw"):
        is_r = wm == "raw"
        lob = np.load(BASE_DIR / ("data/cache/rl_guard_lob_cache_600h_raw_with_self.npz" if is_r else "data/cache/rl_guard_lob_cache_600h.npz"), mmap_mode="r")
        mkt = np.load(BASE_DIR / ("data/cache/rl_hybrid_market_obs_96d_raw_with_self.npz" if is_r else "data/cache/rl_hybrid_market_obs_96d.npz"), mmap_mode="r")["market_obs_96d"]
        ts = lob["ts"][sub:]
        n_sub = len(ts)
        q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
        q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
        q_lo = max(0, int(np.searchsorted(ts, q0_u)) - 1800)
        q_hi = min(n_sub, int(np.searchsorted(ts, q1_u)) + 120)
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

        wl = 21601
        usable_hi = n_sub - 20 * 3600 - 1900
        raw_s = [int(x) for x in np.linspace(1800, usable_hi - wl, 28)]
        clean_s = [ws for ws in raw_s if (ws + wl < q_lo or ws > q_hi)][:20]
        if stage == "l91d2_mode7":
            bask = [(f"W{k:02d}", ws, 0.35, 0.002) for k, ws in enumerate(clean_s)]
        elif stage in ("l91a_mode0", "l91b_mode1", "l91c_mode2", "l91d1_mode4"):
            # Для узких специалистов проверяем перенос между мирами DEC<->RAW на их целевом классе режимов (val-половина)!
            wl = 7201
            b1_s = np.asarray(lob["bid_p"][sub:, 0], dtype=np.float32)
            a1_s = np.asarray(lob["ask_p"][sub:, 0], dtype=np.float32)
            mid_s = 0.5 * (b1_s + a1_s)
            sp_s = (a1_s - b1_s) / np.maximum(1e-4, mid_s) * 1e4
            r30_s = np.zeros(n_sub, dtype=np.float32)
            r30_s[1800:] = mid_s[1800:] / np.maximum(1e-4, mid_s[:-1800]) - 1.0
            cr_s = np.asarray(mkt[sub:, 81], dtype=np.float32)
            dom_w = []
            for s_i in range(int(0.65 * usable_hi), usable_hi - wl, 900):
                if not (s_i + wl < q_lo or s_i > q_hi):
                    continue
                ms_g = mid_s[s_i : s_i + wl]
                st_i = s_i + int(0.65 * wl)
                dc = float(np.min(mid_s[s_i:st_i] / np.maximum.accumulate(mid_s[s_i:st_i]) - 1.0))
                df = float(np.min(ms_g / np.maximum.accumulate(ms_g) - 1.0))
                uc = float(np.max(r30_s[s_i:st_i]))
                tr_v = float(mid_s[s_i + wl - 1] / max(1e-4, mid_s[st_i]) - 1.0)
                cm_v = float(np.mean(cr_s[s_i:st_i]))
                if stage == "l91a_mode0" and float(np.mean(sp_s[s_i : s_i + wl] >= 40.0)) >= 0.12 and df > -0.065:
                    dom_w.append(s_i)
                elif stage == "l91b_mode1" and -0.032 <= dc <= -0.0065 and df > -0.045 and cm_v < 0.45:
                    dom_w.append(s_i)
                elif stage == "l91c_mode2" and uc >= 0.011 and tr_v >= -0.035:
                    dom_w.append(s_i)
                elif stage == "l91d1_mode4" and (df <= -0.026 or (uc >= 0.018 and df <= -0.018)):
                    dom_w.append(s_i)
            sel_d = [dom_w[int(i)] for i in np.linspace(0, len(dom_w) - 1, min(20, len(dom_w)))] if dom_w else clean_s
            bask = [(f"DOM{k:02d}", ws, 0.0, 0.0) for k, ws in enumerate(sel_d)]
        else:
            bask = [(f"W{k:02d}", ws, 0.0, 0.0) for k, ws in enumerate(clean_s)]

        rc = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=c_pad, val_windows=bask, win_len=wl, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False, risk_inv_pen_w=1.50)[:2]
        rh = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=c_pad, val_windows=bask, win_len=wl, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False, risk_inv_pen_w=1.50)[:2]

        oos_bask = [(f"OOS{k:02d}", int(ws), 0.35 if stage == "l91d2_mode7" else 0.0, 0.0) for k, ws in enumerate(range(n_sub - 20 * 3600, n_sub - 10801, 3600))]
        ro = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=c_pad, val_windows=oos_bask, win_len=10801, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False, risk_inv_pen_w=1.50)[:2]

        for idx_c, nm_c in enumerate(("RAW_island", "DEC_island")):
            f_m = compute_l91_fitness(rc[idx_c], rh[idx_c], wl, len(bask), stage)
            scores[idx_c] += 0.50 * f_m["l91_score_bp"]
            oos_d = float(ro[idx_c]["clean_ev_pct"]) * 8.0
            oos_ev[idx_c] += 0.50 * oos_d
            rep[nm_c][wm.upper()] = {**f_m, "blind_20h_oos_ev_day_pct": round(oos_d, 3), "blind_20h_oos_fills": round(float(ro[idx_c]["fills"]), 1)}

    win_idx = 0 if (scores[0] + 15.0 * oos_ev[0]) >= (scores[1] + 15.0 * oos_ev[1]) else 1
    win_nm = "RAW_island" if win_idx == 0 else "DEC_island"
    win_st = _normalize_weights_order3(c_r if win_idx == 0 else c_d, torch.float32)
    meta = {
        "arch_version": "order3_v2",
        "wf_version": f"l91_{tag.lower()}",
        "champ_tag": tag,
        "selected_island": win_nm,
        "cross_4w_l91_score_bp": round(scores[win_idx], 2),
        "blind_20h_oos_ev_day_pct": round(oos_ev[win_idx], 3),
        "pure_endogenous": True,
        "k_modes": K_MODES_ORDER3,
        "in_dim": TOTAL_DIM_ORDER3,
        "total_dim": TOTAL_DIM_ORDER3,
        "mkt_dim": MKT_DIM_ORDER3,
        "ag_dim": AG_DIM_ORDER3,
        "expert_out_dim": EXPERT_OUT_DIM_ORDER3,
        "router_in_dim": ROUTER_IN_DIM_ORDER3,
    }
    torch.save({**win_st, **meta}, dst)
    _log_m(
        f"🏆 [{tag}] Целостный победитель: {win_nm} (L91_4W={scores[win_idx]:+.2f}bp, OOS_EV/d={oos_ev[win_idx]:+.3f}% "
        f"vs другой остров {scores[1-win_idx]:+.2f}bp, OOS={oos_ev[1-win_idx]:+.3f}%) -> {dst.name}"
    )
    (LOG_DIR / f"audit_{stage}.json").write_text(json.dumps({"winner": win_nm, "scores": scores, "oos_ev_day": oos_ev, "report": rep}, indent=2), encoding="utf-8")
    return {"winner": win_nm, "scores": scores, "oos_ev_day": oos_ev, "report": rep}


def assemble_l91_specialists() -> Path:
    """Собирает целостные нелинейные башни L91a..d2 в единый банк мод без порчи весов усреднением!"""
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=torch.device("cpu"))
    base = pol.load_initial_champion(BASE_LIVE_PATH)
    out = {k: base[k].float().clone() for k in pol.PARAM_KEYS}

    w2 = out["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    b2 = out["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    wmup = out["w_moe_up"].view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    bmup = out["b_moe_up"].view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    wmdn = out["w_moe_dn"].view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)

    # Сохраняем также оригинальный L89 Mode #4 в слот #5 как резервный глубокий якорь, а обновлённый L91d1 ставим в #4!
    w2[:, 5, :] = w2[:, 4, :].clone()
    b2[0, 5, :] = b2[0, 4, :].clone()
    wmup[:, 5, :] = wmup[:, 4, :].clone()
    bmup[0, 5, :] = bmup[0, 4, :].clone()
    wmdn[5] = wmdn[4].clone()

    for tm, ck_name in (
        (0, "l91a_mode0_winner.pt"),
        (1, "l91b_mode1_winner.pt"),
        (2, "l91c_mode2_winner.pt"),
        (4, "l91d1_mode4_winner.pt"),
        (7, "l91d2_mode7_winner.pt"),
    ):
        p_s = CHAMP_DIR / ck_name
        if not p_s.exists():
            raise FileNotFoundError(f"Missing specialist winner: {p_s}")
        st_s = pol.load_initial_champion(p_s)
        w2[:, tm, :] = st_s["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, tm, :]
        b2[0, tm, :] = st_s["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[0, tm, :]
        wmup[:, tm, :] = st_s["w_moe_up"].float().view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, tm, :]
        bmup[0, tm, :] = st_s["b_moe_up"].float().view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[0, tm, :]
        wmdn[tm] = st_s["w_moe_dn"].float().view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)[tm]
        exp0_pct = 100.0 / (1.0 + math.exp(-float(b2[0, tm, 0].item())))
        _log_m(f"  🧩 Встроен целостный нелинейный специалист Mode #{tm} из {ck_name} (базовый bias exp={exp0_pct:.1f}%)")

    # Mode #3: аналитический Бункер кэша (0% покупок, мягкая разгрузка имеющейся позиции)
    w2[:, 3, :] = 0.0
    b2[0, 3, :] = b2[0, 7, :].clone()
    b2[0, 3, 0] = -12.0
    b2[0, 3, 1:13] = -10.0
    b2[0, 3, 13] = +8.0
    b2[0, 3, 14:25] = -8.0
    b2[0, 3, 25] = -10.0
    b2[0, 3, [48, 55, 56]] = -12.0
    b2[0, 3, 54] = +4.5
    wmdn[3].zero_()

    # Mode #6: резервный пустой слот
    w2[:, 6, :] = 0.0
    b2[0, 6, :] = b2[0, 4, :].clone()
    b2[0, 6, 55] = -12.0
    wmdn[6].zero_()

    out["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
    out["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
    out["w_moe_up"] = wmup.reshape(256, MOE_MLP_UP_DIM_ORDER3)
    out["b_moe_up"] = bmup.reshape(1, MOE_MLP_UP_DIM_ORDER3)
    out["w_moe_dn"] = wmdn

    dst = CHAMP_DIR / "l91_specialists_assembled.pt"
    meta = {
        "arch_version": "order3_v2",
        "wf_version": "l91_specialists_assembled_intact",
        "champ_tag": "L91_Specialists_Assembled",
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
    _log_m(f"🎯 Сборка целостных специалистов L91 завершена -> {dst.name}")
    return dst


def coronate_12d_and_oos() -> dict:
    """Финальный 12-суточный 4-физический экзамен (12 x 24h) + слепой 20ч OOS против живого L89_V2 и L85."""
    _log_m("🏆 СТАРТ ГЕНЕРАЛЬНОГО 12-СУТОЧНОГО ЭКЗАМЕНА (12 x 24h, 4 ФИЗИКИ + СЛЕПОЙ 20ч OOS)")
    dev = torch.device("cuda:0")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)

    cands_named = [("Live_L89_V2", BASE_LIVE_PATH)]
    for nm, rel_p in (
        ("Ref_L85_Active", CHAMP_DIR / "l85_dualsoup_50d_50r.pt"),
        ("L91f_Orchestrator", CHAMP_DIR / "l91f_orchestrator_winner.pt"),
        ("L91f2_MultiMode", CHAMP_DIR / "l91f2_multimode_winner.pt"),
        ("L91f2_MM_RAW", CHAMP_DIR / "bak_l91f2_mm" / "island_0_best.pt"),
        ("L91f2_MM_DEC", CHAMP_DIR / "bak_l91f2_mm" / "island_1_best.pt"),
        ("L91g_Joint_Winner", CHAMP_DIR / "l91g_joint_winner.pt"),
        ("L91g_Joint_RAW", CHAMP_DIR / "bak_l91g_joint" / "island_0_best.pt"),
        ("L91g_Joint_DEC", CHAMP_DIR / "bak_l91g_joint" / "island_1_best.pt"),
    ):
        if rel_p.exists():
            cands_named.append((nm, rel_p))

    states = [pol.load_initial_champion(p) for _, p in cands_named]
    states_gpu = [{k: v.to(dev) for k, v in st.items()} for st in states]
    c_pad = states_gpu + [states_gpu[0]] * (32 - len(states_gpu))

    sub = 3016277
    win_24h = 86401
    summary = {nm: {"l91_4w": 0.0, "geo_4w": 0.0, "ev_day_4w": 0.0, "oos_ev_day": 0.0, "max_dd": 0.0, "worlds": {}} for nm, _ in cands_named}

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

        starts_12d = [int(x) for x in np.linspace(1800, n_sub - win_24h - 10, 12)]
        bask_12d = [(f"D{k:02d}", ws, 0.0, 0.0) for k, ws in enumerate(starts_12d)]
        rc = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=bask_12d, win_len=win_24h, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)
        rh = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=states_gpu, val_windows=bask_12d, win_len=win_24h, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)

        oos_bask = [(f"OOS{k:02d}", int(ws), 0.0, 0.0) for k, ws in enumerate(range(n_sub - 20 * 3600, n_sub - 10801, 3600))]
        ro_c = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=c_pad, val_windows=oos_bask, win_len=10801, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)[:len(cands_named)]
        ro_h = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=c_pad, val_windows=oos_bask, win_len=10801, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)[:len(cands_named)]

        for idx_c, (nm, _) in enumerate(cands_named):
            f_m = compute_l91_fitness(rc[idx_c], rh[idx_c], win_24h, 12, "coronate")
            oos_mix_d = (0.55 * float(ro_c[idx_c]["clean_ev_pct"]) + 0.45 * float(ro_h[idx_c]["clean_ev_pct"])) * 8.0
            summary[nm]["l91_4w"] += 0.50 * f_m["l91_score_bp"]
            summary[nm]["geo_4w"] += 0.25 * (float(rc[idx_c]["geo_score_bp"]) + float(rh[idx_c]["geo_score_bp"]))
            summary[nm]["ev_day_4w"] += 0.25 * (float(rc[idx_c]["clean_ev_pct"]) + float(rh[idx_c]["clean_ev_pct"]))
            summary[nm]["oos_ev_day"] += 0.50 * oos_mix_d
            summary[nm]["max_dd"] = max(summary[nm]["max_dd"], f_m["dd_max_pct"])
            summary[nm]["worlds"][wm.upper()] = {
                "calib_ev_d": round(float(rc[idx_c]["clean_ev_pct"]), 3),
                "harsh_ev_d": round(float(rh[idx_c]["clean_ev_pct"]), 3),
                "calib_dd": round(float(rc[idx_c]["dd_pct"]), 2),
                "harsh_dd": round(float(rh[idx_c]["dd_pct"]), 2),
                "fills_12d": round(float(rc[idx_c]["fills"]), 1),
                "cnl_per_hr": round(f_m["cnl_per_hr"], 1),
                "oos_ev_d": round(oos_mix_d, 3),
                "oos_fills": round(float(ro_c[idx_c]["fills"]), 1),
                "modes": [round(float(x), 3) for x in rc[idx_c].get("mode_probs", [])],
            }
            _log_m(
                f"  [{wm.upper()}] {nm:20s} | EV/d={rc[idx_c]['clean_ev_pct']:+.3f}%/{rh[idx_c]['clean_ev_pct']:+.3f}% "
                f"| DD={f_m['dd_max_pct']:.2f}% | Fills={rc[idx_c]['fills']:.0f} | OOS_EV/d={oos_mix_d:+.3f}% ({ro_c[idx_c]['fills']:.0f}f)"
            )

    _log_m("-" * 96)
    for nm, _ in cands_named:
        s = summary[nm]
        _log_m(f"📊 ИТОГ [{nm:20s}] L91_4W={s['l91_4w']:+.2f}bp | Mean_EV/d={s['ev_day_4w']:+.3f}% | OOS_EV/d={s['oos_ev_day']:+.3f}% | MaxDD={s['max_dd']:.2f}%")

    (LOG_DIR / "coronation_12d_l91.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Order-3 v2.1 (L91) 8-Hour Modular Research & Curriculum Runner")
    p.add_argument("--stage", type=str, required=True, choices=[
        "l91a_mode0", "l91b_mode1", "l91c_mode2", "l91d1_mode4", "l91d2_mode7",
        "assemble_l91", "l91e_regulator", "l91f_orchestrator", "l91f2_multimode", "l91g_joint", "coronate_12d",
    ])
    p.add_argument("--iters", type=int, default=48)
    p.add_argument("--max-hours", type=float, default=1.05)
    args = p.parse_args()

    st = args.stage
    if st == "l91a_mode0":
        r, d = run_dual_gpu_stage(st, BASE_LIVE_PATH, "bak_l91a_mode0", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91a_mode0_winner.pt", st, "L91a_Mode0_FastScalp")
    elif st == "l91b_mode1":
        r, d = run_dual_gpu_stage(st, BASE_LIVE_PATH, "bak_l91b_mode1", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91b_mode1_winner.pt", st, "L91b_Mode1_MedWave")
    elif st == "l91c_mode2":
        r, d = run_dual_gpu_stage(st, BASE_LIVE_PATH, "bak_l91c_mode2", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91c_mode2_winner.pt", st, "L91c_Mode2_UpCarry")
    elif st == "l91d1_mode4":
        r, d = run_dual_gpu_stage(st, BASE_LIVE_PATH, "bak_l91d1_mode4", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91d1_mode4_winner.pt", st, "L91d1_Mode4_AgileDeep")
    elif st == "l91d2_mode7":
        r, d = run_dual_gpu_stage(st, BASE_LIVE_PATH, "bak_l91d2_mode7", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91d2_mode7_winner.pt", st, "L91d2_Mode7_Unwind")
    elif st == "assemble_l91":
        assemble_l91_specialists()
    elif st == "l91e_regulator":
        src = CHAMP_DIR / "l91_specialists_assembled.pt"
        r, d = run_dual_gpu_stage(st, src, "bak_l91e_reg", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91e_regulator_winner.pt", st, "L91e_Regulator36D")
    elif st == "l91f_orchestrator":
        src = CHAMP_DIR / "l91e_regulator_winner.pt"
        r, d = run_dual_gpu_stage(st, src, "bak_l91f_orch", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91f_orchestrator_winner.pt", st, "L91f_Orchestrator386D")
    elif st == "l91f2_multimode":
        src = CHAMP_DIR / "l91f_orchestrator_winner.pt"
        r, d = run_dual_gpu_stage(st, src, "bak_l91f2_mm", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91f2_multimode_winner.pt", st, "L91f2_ShieldedMultiMode")
    elif st == "l91g_joint":
        # Стартуем строго от проверенного на всех 4 физиках L91f (+312.29bp), отбраковав L91f2 (-591bp, где выключилась Mode #4)!
        src = CHAMP_DIR / "l91f_orchestrator_winner.pt"
        r, d = run_dual_gpu_stage(st, src, "bak_l91g_joint", 80, args.iters, args.max_hours)
        select_intact_cross_world_winner(r, d, CHAMP_DIR / "l91g_joint_winner.pt", st, "L91g_Joint_Polish")
    elif st == "coronate_12d":
        coronate_12d_and_oos()


if __name__ == "__main__":
    main()
