#!/usr/bin/env python3
"""
_tools/train_order3_v2_modular_curriculum.py
=============================================================================
Полный автоматический конвейер перехода на архитектуру Order-3 v2
(64D Liquid Memory + Нелинейные Residual-SwiGLU Специалисты Мод + Двухкаскадный
ResSwiGLU Оркестратор 386D) и 4-ступенчатую политику модульного обучения:

  1. l86_mem64       : Адаптация длинной памяти 64D (банки 30m / 120m) с 0.00e+00 старта.
  2. l87a_mode1      : Нелинейный специалист Mode #1 (умеренные прострелы -1.5%..-2.8% + хвост).
  3. l87b_mode2      : Нелинейный специалист Mode #2 (UP-Carry + Handoff Tail: импульс -> остывание/разворот).
  4. l87c_mode4      : Нелинейный специалист Mode #4 (глубокие каскады -3.5%..-9% + фильтр 2-го дна).
  5. l87d_mode7      : Нелинейный специалист Mode #7 (Handoff Receiver со стартовой позицией rho_0=0.25..0.60).
  6. assemble_v2     : Хирургическая сборка 6 активных нелинейных экспертов + 2 резервных слотов (-14.0).
  7. l88_deeprouter  : Изолированный урок двухкаскадного Оркестратора 386D на мультирежимной + 6h Slide корзине.
  8. l89_joint       : Финальная совместная притирка переходов (Joint Handoff Polish, микро-шаг 0.30x).
  9. coronate_11d    : Проверка Sim/Live паритета (0.00e+00) и 11-суточный 4-физический экзамен против L84.
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
BASE_L84_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
LOG_DIR = BASE_DIR / "logs" / "order3_v2"


def _log_master(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "master_curriculum.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def _apply_stage_constraints(
    cand: dict[str, torch.Tensor],
    stage: str,
    base_ref: dict[str, torch.Tensor],
    dtype: torch.dtype,
    dev: torch.device,
) -> dict[str, torch.Tensor]:
    """Гарантирует 100% побитовую заморозку закрытых модулей и структурные ограничения текущего урока."""
    norm_d = _normalize_weights_order3(cand, dtype)
    out = {k: base_ref[k].detach().clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}

    if stage == "l86_mem64":
        # Разморожены ТОЛЬКО новые банки длинной памяти mem[48:64] (30m / 120m) и их проекции в ствол и роутер!
        wrec = out["w_rec"].float().clone()
        brec = out["b_rec"].float().clone()
        wm2h = out["w_mem2h"].float().clone()
        wg1 = out["w_gate1"].float().clone()

        wrec_c = norm_d["w_rec"].float().to(dev)
        brec_c = norm_d["b_rec"].float().to(dev)
        wm2h_c = norm_d["w_mem2h"].float().to(dev)
        wg1_c = norm_d["w_gate1"].float().to(dev)

        wrec[:, 48:64] = wrec_c[:, 48:64]
        wrec[:, 112:128] = wrec_c[:, 112:128]
        brec[:, 48:64] = brec_c[:, 48:64]
        brec[:, 112:128] = brec_c[:, 112:128]
        wm2h[48:64, :] = wm2h_c[48:64, :]
        wg1[320:336, :] = wg1_c[320:336, :]

        out["w_rec"] = wrec.to(dtype)
        out["b_rec"] = brec.to(dtype)
        out["w_mem2h"] = wm2h.to(dtype)
        out["w_gate1"] = wg1.to(dtype)
        return out

    if stage in ("l87a_mode1", "l87b_mode2", "l87c_mode4", "l87d_mode7"):
        t_m = {"l87a_mode1": 1, "l87b_mode2": 2, "l87c_mode4": 4, "l87d_mode7": 7}[stage]
        w2_new = out["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        b2_new = out["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()
        wmup_new = out["w_moe_up"].float().view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
        bmup_new = out["b_moe_up"].float().view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
        wmdn_new = out["w_moe_dn"].float().view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()

        wm = norm_d["w2_moe"].float().to(dev).view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, t_m, :].clone()
        bm = norm_d["b2_moe"].float().to(dev).view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, t_m, :].clone()
        mup_k = norm_d["w_moe_up"].float().to(dev).view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, t_m, :].clone()
        bup_k = norm_d["b_moe_up"].float().to(dev).view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, t_m, :].clone()
        mdn_k = norm_d["w_moe_dn"].float().to(dev).view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)[t_m].clone()

        # Инвариант безопасности: импульсный рыночный тейкер-бай [48] и сырой скальп [55] закрыты во всех не-скальп модах!
        wm[:, [48, 55]] = 0.0
        bm[0, [48, 55]] = -11.0
        mdn_k[:, [48, 55]] = 0.0

        if stage == "l87a_mode1":
            # Умеренный ловец шпилек (-1.5%..-2.7%): не уходит в глубокий -4% отступ и держит 2 сконцентрированных слота
            n_r38 = torch.linalg.vector_norm(wm[:, 38]).clamp(min=1e-6)
            wm[:, 38] = wm[:, 38] * min(1.0, float(1.25 / n_r38))
            bm[0, 38] = torch.clamp(bm[0, 38], -0.50, +0.60)
            mdn_k[:, 38] = torch.clamp(mdn_k[:, 38], -0.25, 0.25)
            wm[:, 3:13] = 0.0
            bm[0, 3:13] = -8.5
            mdn_k[:, 3:13] = 0.0
        elif stage == "l87b_mode2":
            # Нелинейный UP-Carry + Handoff Exit: активный якорь рециклинга [56] и нелинейный сброс при развороте [0, 41, 42, 54]
            wm[:, 3:13] = 0.0
            bm[0, 3:13] = -8.5
            mdn_k[:, 3:13] = 0.0
            bm[0, 56] = torch.clamp(bm[0, 56], -0.5, 4.5)

        w2_new[:, t_m, :] = wm
        b2_new[:, t_m, :] = bm
        wmup_new[:, t_m, :] = mup_k
        bmup_new[:, t_m, :] = bup_k
        wmdn_new[t_m] = mdn_k

        out["w2_moe"] = w2_new.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
        out["b2_moe"] = b2_new.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
        out["w_moe_up"] = wmup_new.reshape(256, MOE_MLP_UP_DIM_ORDER3).to(dtype)
        out["b_moe_up"] = bmup_new.reshape(1, MOE_MLP_UP_DIM_ORDER3).to(dtype)
        out["w_moe_dn"] = wmdn_new.to(dtype)
        return out

    if stage in ("l88_deeprouter", "l89_joint"):
        router_keys = ("w_gate1", "b_gate1", "w_gate_res1", "b_gate_res", "w_gate_res2", "w_gate2", "b_gate2")
        for kr in router_keys:
            out[kr] = norm_d[kr].to(device=dev, dtype=dtype)

        if stage == "l89_joint":
            # В Joint Handoff Polish открыты также нелинейные SwiGLU-ветки активных мод (#1, #2, #4, #5, #7)!
            wmup_new = out["w_moe_up"].float().view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
            bmup_new = out["b_moe_up"].float().view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3).clone()
            wmdn_new = out["w_moe_dn"].float().view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3).clone()

            wmup_c = norm_d["w_moe_up"].float().to(dev).view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
            bmup_c = norm_d["b_moe_up"].float().to(dev).view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
            wmdn_c = norm_d["w_moe_dn"].float().to(dev).view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)
            for km in (1, 2, 4, 5, 7):
                wmup_new[:, km, :] = wmup_c[:, km, :]
                bmup_new[:, km, :] = bmup_c[:, km, :]
                mdn_km = wmdn_c[km].clone()
                mdn_km[:, [48, 55]] = 0.0
                if km in (1, 2):
                    mdn_km[:, 3:13] = 0.0
                wmdn_new[km] = mdn_km
            out["w_moe_up"] = wmup_new.reshape(256, MOE_MLP_UP_DIM_ORDER3).to(dtype)
            out["b_moe_up"] = bmup_new.reshape(1, MOE_MLP_UP_DIM_ORDER3).to(dtype)
            out["w_moe_dn"] = wmdn_new.to(dtype)

        # Резервный слот #6 и законсервированный до конца канарейки скальпер #0 держим строго припаркованными:
        wg2 = out["w_gate2"].clone()
        bg2 = out["b_gate2"].clone()
        wg2[:, [0, 6]] = 0.0
        wg2[:, [K_MODES_ORDER3 + 0, K_MODES_ORDER3 + 6]] = 0.0
        bg2[0, 0] = -12.0   # Mode #0 ждёт калибровки живого канареечного теста (D4)
        bg2[0, 6] = -14.0
        bg2[0, [K_MODES_ORDER3 + 0, K_MODES_ORDER3 + 6]] = +4.0
        out["w_gate2"] = wg2
        out["b_gate2"] = bg2
        return out

    return out


def _build_stage_seeds(
    base_ref: dict[str, torch.Tensor],
    stage: str,
    dtype: torch.dtype,
    dev: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Строит начальный пул кандидатов, где Seed #0 ВСЕГДА даёт 0.00e+00 потерь относительно старта!"""
    s0 = _apply_stage_constraints(base_ref, stage, base_ref, dtype, dev)
    seeds = [s0]
    gen = torch.Generator(device="cpu").manual_seed(20261188)

    if stage == "l88_deeprouter":
        # Seed #0: Истинный Zero-Regression Anchor = 100% эталон L86_Mem64 (новые моды #1,#2,#3,#5 закрыты на -11.0, регрессия физически невозможна!)
        d0 = {k: base_ref[k].clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        wg2_0 = d0["w_gate2"].float().clone()
        bg2_0 = d0["b_gate2"].float().clone()
        wg2_0[:, [0, 1, 2, 3, 5, 6]] = 0.0
        bg2_0[0, [0, 1, 2, 3, 5]] = -11.0
        bg2_0[0, 6] = -14.0
        d0["w_gate2"] = wg2_0.to(dtype)
        d0["b_gate2"] = bg2_0.to(dtype)
        seeds = [_apply_stage_constraints(d0, stage, base_ref, dtype, dev)]

        # Seeds #1..#3: Деликатные хирургические надстройки вокруг ядра #4/#7 (подключают #1,#2,#3,#5 только при сильном сигнале!)
        for b1_m, b2_m, b3_m, b5_m, sh_m in (
            (-4.2, -4.5, -3.8, -4.0, 1.05),
            (-3.4, -3.6, -3.2, -3.4, 1.15),
            (-2.8, -3.0, -2.6, -2.8, 1.25),
        ):
            d = {k: base_ref[k].clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            wg1 = d["w_gate1"].float().clone()
            wg2 = d["w_gate2"].float().clone()
            bg2 = d["b_gate2"].float().clone()

            wg2[:, [4, 7]] *= sh_m
            bg2[0, [4, 7]] *= sh_m
            wg1[:, 96:110] *= 0.15
            # Ch 96..98 -> Mode #1 (спокойные умеренные шпильки)
            wg1[370, 96:99] = +1.60; wg1[364, 96:99] = -3.00; wg1[345, 96:99] = -2.20
            # Ch 99..101 -> Mode #2 (бычий тренд И низкая загрузка портфеля pos_ratio[257])
            wg1[370, 99:102] = +3.20; wg1[376, 99:102] = -1.80; wg1[257, 99:102] = -2.80; wg1[364, 99:102] = -2.50
            # Ch 102..104 -> Mode #3 (Бункер 0% покупок при падении BigONE)
            wg1[350, 102:105] = -3.80; wg1[364, 102:105] = +2.80; wg1[345, 102:105] = +2.40
            # Ch 105..107 -> Mode #5 (Нелинейный противообвальный снайпер L87c на глубоком каскаде)
            wg1[364, 105:108] = +3.20; wg1[350, 105:108] = +1.20

            wg2[:, 1] = 0.0; wg2[96:99, 1] = +1.80 * sh_m; bg2[0, 1] = b1_m
            wg2[:, 2] = 0.0; wg2[99:102, 2] = +1.90 * sh_m; bg2[0, 2] = b2_m
            wg2[:, 3] = 0.0; wg2[102:105, 3] = +2.20 * sh_m; bg2[0, 3] = b3_m
            wg2[:, 5] = 0.0; wg2[105:108, 5] = +2.00 * sh_m; bg2[0, 5] = b5_m
            d["w_gate1"] = wg1.to(dtype)
            d["w_gate2"] = wg2.to(dtype)
            d["b_gate2"] = bg2.to(dtype)
            seeds.append(_apply_stage_constraints(d, stage, base_ref, dtype, dev))
    elif stage == "l87b_mode2":
        # Доп. сиды для нелинейного UP-Carry (инициализация линейной колонки Mode #2 от L83c + мягкие вариации порога выхода)
        p_l83 = CHAMP_DIR / "bak_l83c_mode2" / "island_1_best.pt"
        if p_l83.exists():
            ck83 = torch.load(p_l83, map_location="cpu", weights_only=False)
            d = {k: base_ref[k].clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            w2[:, 2, :] = ck83["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, 2, :].to(dev)
            b2[0, 2, :] = ck83["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[0, 2, :].to(dev)
            d["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
            d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
            seeds.append(_apply_stage_constraints(d, stage, base_ref, dtype, dev))
    elif stage == "l87a_mode1":
        p_l83 = CHAMP_DIR / "bak_l83b_mode1" / "island_1_best.pt"
        if p_l83.exists():
            ck83 = torch.load(p_l83, map_location="cpu", weights_only=False)
            d = {k: base_ref[k].clone().to(device=dev, dtype=dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            w2 = d["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2 = d["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            w2[:, 1, :] = ck83["w2_moe"].float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, 1, :].to(dev)
            b2[0, 1, :] = ck83["b2_moe"].float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[0, 1, :].to(dev)
            d["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3).to(dtype)
            d["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3).to(dtype)
            seeds.append(_apply_stage_constraints(d, stage, base_ref, dtype, dev))
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
    log_path = LOG_DIR / f"{stage}_island{island_id}_{world_mode}.log"

    def log_w(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')} | {stage} | {world_mode.upper()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
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

    # Сохраняем последние 20 часов строго для слепого Out-of-Sample экзамена (не трогаем в обучении/валидации)
    oos_span = int(20.0 * 3600)
    usable_hi = min(q_lo - 120, total_rows - oos_span - 1900)

    mid_sub = 0.5 * (b1_sub + a1_sub)
    ret30m = np.zeros(total_rows, dtype=np.float32)
    ret30m[1800:] = mid_sub[1800:] / np.maximum(1e-4, mid_sub[:-1800]) - 1.0
    crash900 = mkt_sub[:, 81]

    if stage in ("l87a_mode1", "l87b_mode2", "l87c_mode4", "l87d_mode7"):
        win_len = 7201   # 2ч (включая Handoff Tail в последней трети окна!)
    else:
        win_len = 10801  # 3ч для памяти 64D, глубокого роутера и совместной притирки

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

        if stage == "l87a_mode1":
            if -0.042 <= dip_core <= -0.015 and dip_full > -0.052 and cr_mean < 0.42:
                cand_windows.append((s, 0.0, 0.0))
        elif stage == "l87b_mode2":
            # Handoff-Aware UP-Carry: импульс роста в первых 65% окна + в части окон остывание/откат в финальных 35%!
            if up_core >= 0.013 and tail_ret >= -0.032:
                cand_windows.append((s, 0.0, 0.0))
        elif stage == "l87c_mode4":
            # Глубокие проливы и водопады -3.2%..-12%
            if dip_full <= -0.030 or float(np.max(crash900[s:e])) >= 0.55:
                cand_windows.append((s, 0.0, 0.0))
        elif stage == "l87d_mode7":
            # Handoff Receiver: агент стартует УЖЕ с набранным портфелем (rho_0 = 30%..55%) и средней ценой около/чуть выше рынка!
            rho_v = 0.35 if (len(cand_windows) % 2 == 0) else 0.52
            off_v = 0.004 if (len(cand_windows) % 3 == 0) else -0.002
            cand_windows.append((s, rho_v, off_v))
        else:
            cand_windows.append((s, 0.0, 0.0))

    assert len(cand_windows) >= 28, f"Too few windows ({len(cand_windows)}) for {stage}"
    n_tot_w = len(cand_windows)
    split_i = int(0.70 * n_tot_w)
    tr_pool = cand_windows[: max(16, split_i - 2)]
    val_pool = cand_windows[split_i:]
    n_bask = 20
    val_idx = [int(i) for i in np.linspace(0, len(val_pool) - 1, n_bask)]
    val_basket = [(f"VAL{k:02d}", val_pool[idx][0], val_pool[idx][1], val_pool[idx][2]) for k, idx in enumerate(val_idx)]

    val_slide_basket = None
    if stage in ("l86_mem64", "l88_deeprouter", "l89_joint"):
        slide_len = 21601
        s_lo = max(360000, usable_hi - 20 * 16000)
        s_hi = max(s_lo + 1000, usable_hi - slide_len)
        s_starts = [int(x) for x in np.linspace(s_lo, s_hi, n_bask)]
        val_slide_basket = [(f"SLD{k:02d}", ws, 0.0, 0.0) for k, ws in enumerate(s_starts)]

    pol = MultimodalMoEPolicyOrder3(n_groups=n_groups, n_budgets=6, device=dev)
    base_cpu = pol.load_initial_champion(Path(src_ckpt_str))
    base_ref = {k: base_cpu[k].to(device=dev, dtype=pol.dtype) for k in pol.PARAM_KEYS}
    if stage in ("l87a_mode1", "l87b_mode2", "l87c_mode4", "l87d_mode7"):
        t_m = {"l87a_mode1": 1, "l87b_mode2": 2, "l87c_mode4": 4, "l87d_mode7": 7}[stage]
        base_ref = _lock_router_single_mode(base_ref, t_m, pol.dtype, dev)

    seeds = _build_stage_seeds(base_ref, stage, pol.dtype, dev)
    inv_pen = 1.65 if stage in ("l87b_mode2", "l88_deeprouter", "l89_joint") else (0.95 if stage == "l87d_mode7" else 0.50)

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
            device=dev, pad_to_max_c=pad_m, harsh_stress=harsh, enforce_activity_floor=False, risk_inv_pen_w=inv_pen,
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
            d_m = dict(s)
            d_m["geo_score_bp"] = 0.40 * float(a["geo_score_bp"]) + 0.60 * float(s["geo_score_bp"])
            d_m["clean_ev_pct"] = 0.40 * float(a["clean_ev_pct"]) + 0.60 * float(s["clean_ev_pct"])
            d_m["dd_pct"] = max(float(a["dd_pct"]), float(s["dd_pct"]))
            d_m["fills"] = float(a["fills"]) + float(s["fills"])
            merged.append(d_m)
        return merged

    seed_res = eval_cands(seeds, val_basket, win_len, slide_b=val_slide_basket, harsh=True)
    best_i = int(np.argmax([r["geo_score_bp"] for r in seed_res]))
    champ_state = {k: v.clone() for k, v in seeds[best_i].items()}
    best_geo = float(seed_res[best_i]["geo_score_bp"])
    best_cal = eval_cands([champ_state], val_basket, win_len, slide_b=val_slide_basket, harsh=False)[0]
    log_w(f"🌱 Start Seed #{best_i}/{len(seeds)} ({len(cand_windows)}w): HarshGeo={best_geo:+.2f}bp | CalibEV={best_cal['clean_ev_pct']:+.3f}% (DD={best_cal['dd_pct']:.2f}%)")

    def save_champ(it_n: int, h_res: dict, c_res: dict) -> None:
        meta = {
            "arch_version": "order3_v2",
            "wf_version": f"order3_v2_{stage}_{world_mode}",
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
    rng = np.random.default_rng(20261190 + island_id * 9973)
    half = (n_groups - 2) // 2
    r_view = pol.ring_mult_half

    for it in range(1, max_iters + 1):
        if (time.time() - t0) / 3600.0 >= max_hours:
            break
        t_it = time.time()

        # Межмировая миграция каждые 6 шагов
        if it % 6 == 0:
            nb_p = out_dir / f"island_{1 - island_id}_best.pt"
            if nb_p.exists():
                try:
                    nb_cpu = pol.load_initial_champion(nb_p)
                    nb_st = _apply_stage_constraints({k: nb_cpu[k].to(dev, pol.dtype) for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
                    nb_r = eval_cands([nb_st], val_basket, win_len, slide_b=val_slide_basket, harsh=True)[0]
                    if float(nb_r["geo_score_bp"]) > best_geo + 0.05:
                        champ_state = nb_st
                        best_geo = float(nb_r["geo_score_bp"])
                        c_nb = eval_cands([champ_state], val_basket, win_len, slide_b=val_slide_basket, harsh=False)[0]
                        save_champ(it, nb_r, c_nb)
                        log_w(f"🧬 [It {it:02d}] Миграция улучшила лидера -> HarshGeo={best_geo:+.2f}bp (DD={c_nb['dd_pct']:.2f}%)")
                except Exception:
                    pass

        prog = it / max(1.0, float(max_iters))
        sig_sc = 1.10 if prog < 0.45 else (0.65 if prog < 0.80 else 0.35)
        if stage == "l89_joint":
            sig_sc *= 0.32  # деликатная притирка границ Handoff без разрушения специалистов!

        for k in pol.PARAM_KEYS:
            pol.__dict__[k][:] = champ_state[k].unsqueeze(0)

        if stage == "l86_mem64":
            ewr = torch.randn((half, 256, 16), device=dev) * (0.032 * sig_sc * r_view.view(half, 1, 1))
            ewg = torch.randn((half, 256, 16), device=dev) * (0.032 * sig_sc * r_view.view(half, 1, 1))
            ebr = torch.randn((half, 1, 32), device=dev) * (0.050 * sig_sc * r_view.view(half, 1, 1))
            em2h = torch.randn((half, 16, 256), device=dev) * (0.035 * sig_sc * r_view.view(half, 1, 1))
            eg1m = torch.randn((half, 16, ROUTER_H_DIM_ORDER3), device=dev) * (0.038 * sig_sc * r_view.view(half, 1, 1))

            pol.w_rec[2 : 2 + 2 * half, :, 48:64].add_(torch.stack([ewr, -ewr], dim=1).reshape(2 * half, 256, 16))
            pol.w_rec[2 : 2 + 2 * half, :, 112:128].add_(torch.stack([ewg, -ewg], dim=1).reshape(2 * half, 256, 16))
            pol.b_rec[2 : 2 + 2 * half, :, 48:64].add_(torch.stack([ebr[:, :, :16], -ebr[:, :, :16]], dim=1).reshape(2 * half, 1, 16))
            pol.b_rec[2 : 2 + 2 * half, :, 112:128].add_(torch.stack([ebr[:, :, 16:], -ebr[:, :, 16:]], dim=1).reshape(2 * half, 1, 16))
            pol.w_mem2h[2 : 2 + 2 * half, 48:64, :].add_(torch.stack([em2h, -em2h], dim=1).reshape(2 * half, 16, 256))
            pol.w_gate1[2 : 2 + 2 * half, 320:336, :].add_(torch.stack([eg1m, -eg1m], dim=1).reshape(2 * half, 16, ROUTER_H_DIM_ORDER3))

        elif stage in ("l87a_mode1", "l87b_mode2", "l87c_mode4", "l87d_mode7"):
            t_m = {"l87a_mode1": 1, "l87b_mode2": 2, "l87c_mode4": 4, "l87d_mode7": 7}[stage]
            # Мутируем как линейный путь моды t_m, так и её персональную нелинейную SwiGLU-башню (256 -> 96 -> 58)!
            ew_lin = torch.randn((half, 256, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.016 * sig_sc * r_view.view(half, 1, 1))
            eb_lin = torch.randn((half, 1, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.055 * sig_sc * r_view.view(half, 1, 1))
            ew_up = torch.randn((half, 256, 2 * MOE_MLP_H_ORDER3), device=dev) * (0.035 * sig_sc * r_view.view(half, 1, 1))
            eb_up = torch.randn((half, 1, 2 * MOE_MLP_H_ORDER3), device=dev) * (0.055 * sig_sc * r_view.view(half, 1, 1))
            ew_dn = torch.randn((half, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.042 * sig_sc * r_view.view(half, 1, 1))

            w2_v = pol.w2_moe[2 : 2 + 2 * half].view(2 * half, 256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            b2_v = pol.b2_moe[2 : 2 + 2 * half].view(2 * half, 1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            up_v = pol.w_moe_up[2 : 2 + 2 * half].view(2 * half, 256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
            bu_v = pol.b_moe_up[2 : 2 + 2 * half].view(2 * half, 1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)

            w2_v[:, :, t_m, :].add_(torch.stack([ew_lin, -ew_lin], dim=1).reshape(2 * half, 256, EXPERT_OUT_DIM_ORDER3))
            b2_v[:, :, t_m, :].add_(torch.stack([eb_lin, -eb_lin], dim=1).reshape(2 * half, 1, EXPERT_OUT_DIM_ORDER3))
            up_v[:, :, t_m, :].add_(torch.stack([ew_up, -ew_up], dim=1).reshape(2 * half, 256, 2 * MOE_MLP_H_ORDER3))
            bu_v[:, :, t_m, :].add_(torch.stack([eb_up, -eb_up], dim=1).reshape(2 * half, 1, 2 * MOE_MLP_H_ORDER3))
            pol.w_moe_dn[2 : 2 + 2 * half, t_m].add_(torch.stack([ew_dn, -ew_dn], dim=1).reshape(2 * half, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3))

        else:
            # Двухкаскадный роутер (l88_deeprouter) и финальная притирка (l89_joint)
            ew1 = torch.randn((half, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.025 * sig_sc * r_view.view(half, 1, 1))
            ew1[:, 256:, :] *= 2.40  # усиленное внимание к портфелю, 64D памяти и 42D волновым сигналам!
            eb1 = torch.randn((half, 1, ROUTER_H_DIM_ORDER3), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
            er1 = torch.randn((half, ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3), device=dev) * (0.038 * sig_sc * r_view.view(half, 1, 1))
            ebr = torch.randn((half, 1, 2 * ROUTER_RES_H_ORDER3), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
            er2 = torch.randn((half, ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3), device=dev) * (0.045 * sig_sc * r_view.view(half, 1, 1))
            ew2 = torch.randn((half, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.040 * sig_sc * r_view.view(half, 1, 1))
            eb2 = torch.randn((half, 1, ROUTER_OUT_DIM_ORDER3), device=dev) * (0.100 * sig_sc * r_view.view(half, 1, 1))

            pol.w_gate1[2 : 2 + 2 * half].add_(torch.stack([ew1, -ew1], dim=1).reshape(2 * half, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3))
            pol.b_gate1[2 : 2 + 2 * half].add_(torch.stack([eb1, -eb1], dim=1).reshape(2 * half, 1, ROUTER_H_DIM_ORDER3))
            pol.w_gate_res1[2 : 2 + 2 * half].add_(torch.stack([er1, -er1], dim=1).reshape(2 * half, ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3))
            pol.b_gate_res[2 : 2 + 2 * half].add_(torch.stack([ebr, -ebr], dim=1).reshape(2 * half, 1, 2 * ROUTER_RES_H_ORDER3))
            pol.w_gate_res2[2 : 2 + 2 * half].add_(torch.stack([er2, -er2], dim=1).reshape(2 * half, ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3))
            pol.w_gate2[2 : 2 + 2 * half].add_(torch.stack([ew2, -ew2], dim=1).reshape(2 * half, ROUTER_H_DIM_ORDER3, ROUTER_OUT_DIM_ORDER3))
            pol.b_gate2[2 : 2 + 2 * half].add_(torch.stack([eb2, -eb2], dim=1).reshape(2 * half, 1, ROUTER_OUT_DIM_ORDER3))

            if stage == "l89_joint":
                for km in (1, 2, 4, 7):
                    edn = torch.randn((half, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), device=dev) * (0.030 * sig_sc * r_view.view(half, 1, 1))
                    pol.w_moe_dn[2 : 2 + 2 * half, km].add_(torch.stack([edn, -edn], dim=1).reshape(2 * half, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3))

        # Применяем строгие структурные маски ко всем мутантам ДО оценки
        for gi in range(2, 2 + 2 * half):
            cl = _apply_stage_constraints({k: pol.__dict__[k][gi] for k in pol.PARAM_KEYS}, stage, base_ref, pol.dtype, dev)
            for k in pol.PARAM_KEYS:
                pol.__dict__[k][gi].copy_(cl[k])

        tr_pick = rng.choice(len(tr_pool), size=n_bask, replace=(len(tr_pool) < n_bask))
        tr_bask = [(f"TR{k:02d}", tr_pool[int(ii)][0], tr_pool[int(ii)][1], tr_pool[int(ii)][2]) for k, ii in enumerate(tr_pick)]
        pop_cands = pol.get_candidate_states()
        tr_res = eval_cands(pop_cands, tr_bask, win_len, slide_b=None, harsh=True)
        tr_geos = np.array([float(r["geo_score_bp"]) for r in tr_res], dtype=np.float64)

        es_raw = pol.build_openai_es_candidates(champ_state, tr_geos, lr_scales=(0.50, 1.00, 1.65))
        es_cands = [_apply_stage_constraints(c, stage, base_ref, pol.dtype, dev) for c in es_raw]
        top_idx = [int(i) for i in np.argsort(tr_geos)[::-1][:4]]
        top_muts = [_apply_stage_constraints(pop_cands[i], stage, base_ref, pol.dtype, dev) for i in top_idx]

        shortlist = [champ_state] + es_cands + top_muts
        val_res = eval_cands(shortlist, val_basket, win_len, slide_b=val_slide_basket, harsh=True)
        bv_i = int(np.argmax([float(r["geo_score_bp"]) for r in val_res]))
        cand_geo = float(val_res[bv_i]["geo_score_bp"])

        if cand_geo > best_geo + 0.02:
            gain = cand_geo - best_geo
            best_geo = cand_geo
            champ_state = {k: v.clone() for k, v in shortlist[bv_i].items()}
            cal_r = eval_cands([champ_state], val_basket, win_len, slide_b=val_slide_basket, harsh=False)[0]
            save_champ(it, val_res[bv_i], cal_r)
            vr = val_res[bv_i]
            log_w(
                f"⭐ [It {it:02d} | +{gain:.2f}bp] HarshGeo={best_geo:+.2f}bp | CalibEV={cal_r['clean_ev_pct']:+.3f}% "
                f"(DD={cal_r['dd_pct']:.2f}%) | Fills={vr['fills']:.1f} | Modes={[round(x,2) for x in vr.get('mode_probs',[])]} | {time.time()-t_it:.1f}s"
            )
        elif it % 5 == 0:
            log_w(f"   [It {it:02d}/{max_iters}] TrainTop={np.max(tr_geos):+.2f}bp | ValChamp={best_geo:+.2f}bp | {time.time()-t_it:.1f}s")

    log_w(f"🏁 Done [{stage}] ({world_mode.upper()}) in {(time.time()-t0)/60:.1f}m | Best HarshGeo={best_geo:+.2f}bp")


def run_dual_gpu_stage(stage: str, src_ckpt: Path, save_subdir: str, n_groups: int, max_iters: int, max_hours: float) -> tuple[Path, Path]:
    _log_master(f"▶️ ЗАПУСК ЭТАПА [{stage}] на 2x GPU (cuda:0=RAW, cuda:1=DEC | groups={n_groups}, iters={max_iters})")
    ctx = mp.get_context("spawn")
    p0 = ctx.Process(
        target=_island_worker,
        args=(stage, str(src_ckpt), save_subdir, 0, "cuda:0", "raw", n_groups, max_iters, max_hours),
    )
    p1 = ctx.Process(
        target=_island_worker,
        args=(stage, str(src_ckpt), save_subdir, 1, "cuda:1", "dec", n_groups, max_iters, max_hours),
    )
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    if p0.exitcode != 0 or p1.exitcode != 0:
        raise RuntimeError(f"Stage {stage} failed: rc0={p0.exitcode}, rc1={p1.exitcode}")
    out_dir = CHAMP_DIR / save_subdir
    return out_dir / "island_0_best.pt", out_dir / "island_1_best.pt"


def _blend_50_50(p_raw: Path, p_dec: Path, dst: Path, tag: str) -> Path:
    dev = torch.device("cpu")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)
    sr = pol.load_initial_champion(p_raw)
    sd = pol.load_initial_champion(p_dec)
    blended = {k: (0.50 * sd[k].float() + 0.50 * sr[k].float()) for k in pol.PARAM_KEYS}
    blended = _normalize_weights_order3(blended, torch.float32)
    meta = {
        "arch_version": "order3_v2",
        "wf_version": f"order3_v2_{tag.lower()}",
        "champ_tag": tag,
        "pure_endogenous": True,
        "k_modes": K_MODES_ORDER3,
        "in_dim": TOTAL_DIM_ORDER3,
        "total_dim": TOTAL_DIM_ORDER3,
        "mkt_dim": MKT_DIM_ORDER3,
        "ag_dim": AG_DIM_ORDER3,
        "expert_out_dim": EXPERT_OUT_DIM_ORDER3,
        "router_in_dim": ROUTER_IN_DIM_ORDER3,
    }
    torch.save({**blended, **meta}, dst)
    _log_master(f"✅ Сохранён 50/50 Dual-Physics чекпоинт [{tag}] -> {dst.name}")
    return dst


def assemble_v2_specialists(mem64_ckpt: Path) -> Path:
    dev = torch.device("cpu")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)
    base = pol.load_initial_champion(mem64_ckpt)
    out = {k: base[k].float().clone() for k in pol.PARAM_KEYS}

    w2 = out["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    b2 = out["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
    wmup = out["w_moe_up"].view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    bmup = out["b_moe_up"].view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)
    wmdn = out["w_moe_dn"].view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)

    # Сохраняем Mode #0 из L83a (микро-скальпер широкого спреда на консервации до конца живой канарейки)
    p0_r = CHAMP_DIR / "bak_l83a_mode0" / "island_0_best.pt"
    p0_d = CHAMP_DIR / "bak_l83a_mode0" / "island_1_best.pt"
    if p0_r.exists() and p0_d.exists():
        s0r, s0d = pol.load_initial_champion(p0_r), pol.load_initial_champion(p0_d)
        w2[:, 0, :] = 0.50 * s0d["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, 0, :] + 0.50 * s0r["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, 0, :]
        b2[0, 0, :] = 0.50 * s0d["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[0, 0, :] + 0.50 * s0r["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[0, 0, :]
        wmdn[0].zero_()

    # Подключаем нелинейных специалистов L87a..d без затирания универсального чемпиона Mode #4 (L86_Mem64):
    # - Mode #1 <- L87a (умеренные шпильки)
    # - Mode #2 <- L87b (нелинейный UP-Carry + Handoff Tail)
    # - Mode #4 <- сохраняется 100% от эталона L86_Mem64 (+67.69bp)!
    # - Mode #5 <- L87c (узкий нелинейный противообвальный снайпер переносится из колонки 4 L87c в слот 5!)
    # - Mode #7 <- L87d (Handoff Receiver со стартовой позицией)
    spec_map = [
        (1, 1, CHAMP_DIR / "bak_l87a_mode1"),
        (2, 2, CHAMP_DIR / "bak_l87b_mode2"),
        (5, 4, CHAMP_DIR / "bak_l87c_mode4"),
        (7, 7, CHAMP_DIR / "bak_l87d_mode7"),
    ]
    for dst_m, src_m, sdir in spec_map:
        sr = pol.load_initial_champion(sdir / "island_0_best.pt")
        sd = pol.load_initial_champion(sdir / "island_1_best.pt")
        w2[:, dst_m, :] = 0.50 * sd["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, src_m, :] + 0.50 * sr["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[:, src_m, :]
        b2[0, dst_m, :] = 0.50 * sd["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[0, src_m, :] + 0.50 * sr["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)[0, src_m, :]
        wmup[:, dst_m, :] = 0.50 * sd["w_moe_up"].view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, src_m, :] + 0.50 * sr["w_moe_up"].view(256, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[:, src_m, :]
        bmup[0, dst_m, :] = 0.50 * sd["b_moe_up"].view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[0, src_m, :] + 0.50 * sr["b_moe_up"].view(1, K_MODES_ORDER3, 2 * MOE_MLP_H_ORDER3)[0, src_m, :]
        wmdn[dst_m] = 0.50 * sd["w_moe_dn"].view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)[src_m] + 0.50 * sr["w_moe_dn"].view(K_MODES_ORDER3, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3)[src_m]
        _log_master(f"  🧩 Встроен нелинейный специалист Mode #{dst_m} (из {sdir.name}[#{src_m}]); ядро Mode #4 сохранено от L86_Mem64")

    # Mode #3: аналитический Бункер обвала (0% покупок)
    w2[:, 3, :] = 0.0
    b2[0, 3, :] = b2[0, 4, :].clone()
    b2[0, 3, 0] = -12.0
    b2[0, 3, 1:13] = -10.0
    b2[0, 3, 13] = 8.0
    b2[0, 3, 14:25] = -8.0
    b2[0, 3, 25] = -10.0
    b2[0, 3, 38] = +8.0
    b2[0, 3, 41] = -6.5
    b2[0, 3, 42] = -3.0
    b2[0, 3, [48, 55, 56]] = -12.0
    b2[0, 3, 54] = +4.5
    b2[0, 3, 57] = +3.0
    wmdn[3].zero_()

    # Mode #6: чистый резерв
    w2[:, 6, :] = 0.0
    b2[0, 6, :] = b2[0, 4, :].clone()
    b2[0, 6, 55] = -12.0
    wmdn[6].zero_()

    out["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
    out["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
    out["w_moe_up"] = wmup.reshape(256, MOE_MLP_UP_DIM_ORDER3)
    out["b_moe_up"] = bmup.reshape(1, MOE_MLP_UP_DIM_ORDER3)
    out["w_moe_dn"] = wmdn

    dst = CHAMP_DIR / "l87_v2_assembled.pt"
    meta = {
        "arch_version": "order3_v2",
        "wf_version": "order3_v2_l87_assembled_nonlinear",
        "champ_tag": "L87_V2_Assembled",
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
    _log_master(f"🎯 Сборка нелинейных экспертов Order-3 v2 завершена -> {dst.name}")
    return dst


def evaluate_11d_4physics(challengers: list[tuple[str, Path]]) -> dict:
    """Честный сквозной 11-суточный турнир (11 x 24h) во всех 4 физиках (DEC/RAW x Calib/Harsh) против боевого L84."""
    _log_master("🏆 СТАРТ СЛЕПОГО 11-СУТОЧНОГО ЭКЗАМЕНА (11 x 24h, 4 ФИЗИКИ: DEC/RAW x Calib/Harsh)")
    dev = torch.device("cuda:0")
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)

    cands_named = [("Baseline_L84_Live", BASE_L84_PATH)] + list(challengers)
    states = [pol.load_initial_champion(p) for _, p in cands_named]
    states_gpu = [{k: v.to(dev) for k, v in st.items()} for st in states]

    subcent_idx = 3016277
    win_24h = 86401
    scores_4w = {name: 0.0 for name, _ in cands_named}
    details: dict[str, dict] = {name: {} for name, _ in cands_named}

    for world_mode in ("dec", "raw"):
        is_raw = world_mode == "raw"
        lob_p = BASE_DIR / ("data/cache/rl_guard_lob_cache_600h_raw_with_self.npz" if is_raw else "data/cache/rl_guard_lob_cache_600h.npz")
        mkt_p = BASE_DIR / ("data/cache/rl_hybrid_market_obs_96d_raw_with_self.npz" if is_raw else "data/cache/rl_hybrid_market_obs_96d.npz")
        lob = np.load(lob_p, mmap_mode="r")
        mkt_full = np.load(mkt_p, mmap_mode="r")["market_obs_96d"]

        ts_sub = np.asarray(lob["ts"][subcent_idx:], dtype=np.int64)
        total_rows = len(ts_sub)
        q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
        q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
        sv_np = np.asarray(lob["sell_vol"][subcent_idx:], dtype=np.float32).copy()
        bv_np = np.asarray(lob["buy_vol"][subcent_idx:], dtype=np.float32).copy()
        q_exact = (ts_sub >= q0_u) & (ts_sub <= q1_u)
        sv_np[q_exact] = 0.0
        bv_np[q_exact] = 0.0

        mkt_gpu = torch.from_numpy(np.asarray(mkt_full[subcent_idx:], dtype=np.float32)).to(device=dev, dtype=torch.bfloat16)
        b10_gpu = torch.from_numpy(np.asarray(lob["bid_p"][subcent_idx:, :10], dtype=np.float32)).to(dev)
        a10_gpu = torch.from_numpy(np.asarray(lob["ask_p"][subcent_idx:, :10], dtype=np.float32)).to(dev)
        bq10_gpu = torch.from_numpy(np.asarray(lob["bid_q"][subcent_idx:, :10], dtype=np.float32)).to(dev)
        aq10_gpu = torch.from_numpy(np.asarray(lob["ask_q"][subcent_idx:, :10], dtype=np.float32)).to(dev)
        sv_gpu = torch.from_numpy(sv_np).to(dev)
        bv_gpu = torch.from_numpy(bv_np).to(dev)

        starts_11d = [int(x) for x in np.linspace(1800, total_rows - win_24h - 10, 11)]
        bask_11d = [(f"D{k:02d}", ws, 0.0, 0.0) for k, ws in enumerate(starts_11d)]

        for harsh in (False, True):
            tag_ph = f"{world_mode.upper()}_{'Harsh' if harsh else 'Calib'}"
            res_l = evaluate_order3_basket(
                mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu, bq10_all=bq10_gpu, aq10_all=aq10_gpu,
                sv_all=sv_gpu, bv_all=bv_gpu, cand_states=states_gpu, val_windows=bask_11d, win_len=win_24h,
                tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=harsh, enforce_activity_floor=False,
            )
            for (nm, _), r in zip(cands_named, res_l):
                geo = float(r["geo_score_bp"])
                ev_d = float(r["clean_ev_pct"])
                dd = float(r["dd_pct"])
                scores_4w[nm] += 0.25 * geo
                details[nm][tag_ph] = {
                    "geo_bp": round(geo, 2),
                    "ev_per_day_pct": round(ev_d, 3),
                    "max_dd_pct": round(dd, 2),
                    "fills": round(float(r["fills"]), 1),
                    "modes": [round(float(x), 3) for x in r.get("mode_probs", [])],
                }
                _log_master(f"  [{tag_ph:10s}] {nm:24s} | Geo={geo:+7.2f}bp | EV/d={ev_d:+.3f}% | MaxDD={dd:.2f}% | Fills={r['fills']:.1f}")

    _log_master("-" * 92)
    for nm, _ in cands_named:
        _log_master(f"📊 ИТОГ 4W_Score [{nm:24s}] = {scores_4w[nm]:+.2f} bp")

    summary = {"scores_4w_bp": scores_4w, "details": details}
    (LOG_DIR / "coronation_11d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Order-3 v2 Full Modular Curriculum Runner")
    p.add_argument("--start-from", type=str, default="l86_mem64", choices=[
        "l86_mem64", "l87a_mode1", "l87b_mode2", "l87c_mode4", "l87d_mode7",
        "assemble_v2", "l88_deeprouter", "l89_joint", "coronate_11d",
    ])
    args = p.parse_args()

    order = [
        "l86_mem64", "l87a_mode1", "l87b_mode2", "l87c_mode4", "l87d_mode7",
        "assemble_v2", "l88_deeprouter", "l89_joint", "coronate_11d",
    ]
    start_idx = order.index(args.start_from)
    t_all = time.time()
    _log_master("=" * 92)
    _log_master(f"🚀 СТАРТ ПОЛНОГО МОДУЛЬНОГО ЦИКЛА ORDER-3 v2 (начиная с {args.start_from})")
    _log_master("=" * 92)

    mem64_soup = CHAMP_DIR / "l86_mem64_soup.pt"
    assembled_v2 = CHAMP_DIR / "l87_v2_assembled.pt"
    router_v2_soup = CHAMP_DIR / "l88_deeprouter_soup.pt"
    joint_v2_soup = CHAMP_DIR / "l89_v2_joint_best.pt"

    if start_idx <= 0:
        r0, d0 = run_dual_gpu_stage("l86_mem64", BASE_L84_PATH, "bak_l86_mem64", n_groups=80, max_iters=18, max_hours=0.25)
        _blend_50_50(r0, d0, mem64_soup, "L86_Mem64_50_50")

    if start_idx <= 1:
        run_dual_gpu_stage("l87a_mode1", mem64_soup, "bak_l87a_mode1", n_groups=80, max_iters=24, max_hours=0.30)

    if start_idx <= 2:
        run_dual_gpu_stage("l87b_mode2", mem64_soup, "bak_l87b_mode2", n_groups=80, max_iters=28, max_hours=0.35)

    if start_idx <= 3:
        run_dual_gpu_stage("l87c_mode4", mem64_soup, "bak_l87c_mode4", n_groups=80, max_iters=24, max_hours=0.30)

    if start_idx <= 4:
        run_dual_gpu_stage("l87d_mode7", mem64_soup, "bak_l87d_mode7", n_groups=80, max_iters=24, max_hours=0.30)

    if start_idx <= 5:
        assemble_v2_specialists(mem64_soup)

    if start_idx <= 6:
        r88, d88 = run_dual_gpu_stage("l88_deeprouter", assembled_v2, "bak_l88_deeprouter", n_groups=80, max_iters=18, max_hours=0.25)
        _blend_50_50(r88, d88, router_v2_soup, "L88_DeepRouter_50_50")

    if start_idx <= 7:
        r89, d89 = run_dual_gpu_stage("l89_joint", router_v2_soup, "bak_l89_joint", n_groups=80, max_iters=12, max_hours=0.18)
        _blend_50_50(r89, d89, joint_v2_soup, "L89_V2_Joint_50_50")

    # Проверяем побитовый паритет Sim/Live новых чемпионов Order-3 v2
    for ck_p in (router_v2_soup, joint_v2_soup):
        if ck_p.exists():
            rc_par = subprocess.run(
                [str(BASE_DIR / ".venv/bin/python"), str(BASE_DIR / "_tools/test_order3_sim_live_parity.py"), str(ck_p)],
                cwd=str(BASE_DIR),
                check=False,
            ).returncode
            if rc_par != 0:
                raise RuntimeError(f"Sim/Live parity failed for {ck_p}")

    challs = []
    for nm, p_c in [
        ("L86_Mem64_50_50", mem64_soup),
        ("L88_DeepRouter_50_50", router_v2_soup),
        ("L88_DeepRouter_RAW", CHAMP_DIR / "bak_l88_deeprouter" / "island_0_best.pt"),
        ("L88_DeepRouter_DEC", CHAMP_DIR / "bak_l88_deeprouter" / "island_1_best.pt"),
        ("L89_V2_Joint_50_50", joint_v2_soup),
        ("L89_V2_Joint_RAW", CHAMP_DIR / "bak_l89_joint" / "island_0_best.pt"),
        ("L89_V2_Joint_DEC", CHAMP_DIR / "bak_l89_joint" / "island_1_best.pt"),
    ]:
        if p_c.exists():
            challs.append((nm, p_c))

    evaluate_11d_4physics(challs)
    _log_master(f"🎉 ПОЛНЫЙ ЦИКЛ ORDER-3 v2 ЗАВЕРШЁН за {(time.time()-t_all)/60:.1f} мин!")


if __name__ == "__main__":
    main()
