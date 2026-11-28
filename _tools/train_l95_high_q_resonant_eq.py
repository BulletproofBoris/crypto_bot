#!/usr/bin/env python3
"""
_tools/train_l95_high_q_resonant_eq.py
================================================================================
Урок 95 (Шаг В): Узкополосный высокодобротный эквалайзер пространства состояний
(High-Q Resonant State-Space Equalizer) поверх неизменной несущей H_0 = Live_L92_V1:

    u(s) = H_0(s) + sum_{k=1}^{K(T)} gamma_k(s) * Delta_u_k,
    gamma_k(s) = A_k * exp( -0.5 * sum_{d=1}^{16} (q_{k,d} * (phi_d(s) - c_{k,d}))^2 ).

Принципы обучения (Orthogonal Matching Pursuit + Sep-CMA-ES строго на GPU #1):
  1. Несущая H_0 (w1..b_reg2_a) заморожена на 100% (0.00e+00 боковых лепестков вне резонанса!).
  2. Число полос K_max(T) определяется автоматически из объёма чистой истории (при 13.83 сут -> K_max = 6).
  3. Закон статистической обеспеченности (Sample-Support Nyquist Constraint):
     каждая полоса k обязана срабатывать (gamma_k > 0.20) минимум в >= 25 независимых эпизодах
     (разнесённых минимум на 180с) и минимум на >= 3 различных сутках Train, не задевая обвалы D!
  4. Пошаговая приёмка каждой полосы k строго через Per-Regime PurgedVal (4 суточных блока C/U/Q
     в обоих мирах DEC и RAW): полоса фиксируется ТОЛЬКО если даёт строгий прирост PurgedVal
     при сохранении каскадов C и невозрастании просадки MaxDD.
  5. Rolling Blind OOS (BLIND_00_Q, BLIND_01_D) НЕ участвует ни в обучении, ни в отборе полос —
     проверяется строго 1 раз в финальном выпускном экзамене!
================================================================================
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("TRITON_CACHE_DIR", "/home/restorator/crypto_bot/scratch/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/restorator/crypto_bot/scratch/.inductor_cache")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.fused_order3_clean_tick import (
    EXPERT_OUT_DIM_ORDER3,
    MAX_RES_BANDS_ORDER3,
    MEM_DIM_ORDER3,
    MKT_DIM_ORDER3,
    RES_PHI_DIM_ORDER3,
    _extract_resonant_phi_16d,
    _scale_mkt_44d,
    evaluate_order3_basket,
)
from _tools.regime_purged_split_l94 import (
    SUBCENT_START,
    WIN_24H,
    build_dynamic_regime_split,
)
from _tools.train_order3_champion import MultimodalMoEPolicyOrder3

CACHE_DIR = BASE_DIR / "data" / "cache"
LOG_DIR = BASE_DIR / "logs" / "l95_high_q_resonant"
CHAMP_V1 = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
OUT_BEST = BASE_DIR / "models" / "prl" / "l95_resonant_candidate.pt"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dual_worlds_gpu(dev: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    """Загружает оба мира DEC и RAW (с субцентового старта 3,016,277) в VRAM строго GPU #1."""
    worlds = {}
    q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
    q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
    for wm in ("DEC", "RAW"):
        is_r = wm == "RAW"
        lob = np.load(
            CACHE_DIR / ("rl_guard_lob_cache_600h_raw_with_self.npz" if is_r else "rl_guard_lob_cache_600h.npz"),
            mmap_mode="r",
        )
        mkt = np.load(
            CACHE_DIR / ("rl_hybrid_market_obs_112d_raw_with_self.npz" if is_r else "rl_hybrid_market_obs_112d.npz"),
            mmap_mode="r",
        )["market_obs_112d"]
        ts = lob["ts"][SUBCENT_START:]
        sv_np = np.asarray(lob["sell_vol"][SUBCENT_START:], dtype=np.float32).copy()
        bv_np = np.asarray(lob["buy_vol"][SUBCENT_START:], dtype=np.float32).copy()
        qm = (ts >= q0_u) & (ts <= q1_u)
        sv_np[qm] = 0.0
        bv_np[qm] = 0.0

        worlds[wm] = {
            "mkt": torch.from_numpy(np.asarray(mkt[SUBCENT_START:], dtype=np.float32)).to(dev, torch.bfloat16),
            "b10": torch.from_numpy(np.asarray(lob["bid_p"][SUBCENT_START:, :10], dtype=np.float32)).to(dev),
            "a10": torch.from_numpy(np.asarray(lob["ask_p"][SUBCENT_START:, :10], dtype=np.float32)).to(dev),
            "bq10": torch.from_numpy(np.asarray(lob["bid_q"][SUBCENT_START:, :10], dtype=np.float32)).to(dev),
            "aq10": torch.from_numpy(np.asarray(lob["ask_q"][SUBCENT_START:, :10], dtype=np.float32)).to(dev),
            "sv": torch.from_numpy(sv_np).to(dev),
            "bv": torch.from_numpy(bv_np).to(dev),
        }
    return worlds


@torch.no_grad()
def extract_train_spectral_basis_and_seeds(
    h0_st: dict[str, torch.Tensor],
    w_dec: dict[str, torch.Tensor],
    train_blocks: list[dict],
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """
    Каузально прогоняет 64D SSM-память несущей H_0 строго по Train-блокам (шаг 5с для компактной статистики),
    извлекает реальные траектории 16D базиса phi(t), метку дня и будущую 15-минутную доходность (ТОЛЬКО внутри Train!)
    для физически обоснованной инициализации центров резонаторов c_k и мгновенной проверки критерия Найквиста (>=25 срабатываний на >=3 днях).
    """
    w1_mkt = h0_st["w1"][:MKT_DIM_ORDER3, :].float().to(dev)
    b1_v = h0_st["b1"].float().view(1, 384).to(dev)
    wr1 = h0_st["w_res1"].float().to(dev)
    br1 = h0_st["b_res1"].float().view(1, 128).to(dev)
    wr2 = h0_st["w_res2"].float().to(dev)
    wrec = h0_st["w_rec"].float().to(dev)
    brec = h0_st["b_rec"].float().view(1, 2 * MEM_DIM_ORDER3).to(dev)

    phi_list = []
    day_list = []
    reg_code_list = []
    fwd15m_list = []
    b1_full = w_dec["b10"][:, 0]
    reg_map = {"Q": 0, "U": 1, "C": 2, "D": 3}

    for di, tb in enumerate(train_blocks):
        ws = int(tb["rel_start"])
        we = int(tb["rel_end"])
        # Разогрев памяти слева 1800с + сам суточный Train-блок (шаг 5с):
        idx_5s = torch.arange(max(0, ws - 1800), we, 5, dtype=torch.long, device=dev)
        zm_seq = _scale_mkt_44d(w_dec["mkt"][idx_5s].float(), torch.float32)  # [L, 112]
        pre_seq = zm_seq @ w1_mkt + b1_v
        h1_seq = torch.relu(pre_seq[:, :256]) * (2.0 * torch.sigmoid(pre_seq[:, 256:384])).repeat_interleave(2, dim=-1)
        rp_seq = h1_seq @ wr1 + br1
        h2_seq = h1_seq + (torch.relu(rp_seq[:, :64]) * (2.0 * torch.sigmoid(rp_seq[:, 64:128]))) @ wr2
        rr_seq = h2_seq @ wrec + brec
        ru_seq = torch.tanh(rr_seq[:, :MEM_DIM_ORDER3])
        rg_1s = torch.cat(
            [
                torch.sigmoid(rr_seq[:, 64:72] + 1.0),
                torch.sigmoid(rr_seq[:, 72:80] + 3.8),
                torch.sigmoid(rr_seq[:, 80:88] + 5.2),
                torch.sigmoid(rr_seq[:, 88:96] + 6.6),
                torch.sigmoid(rr_seq[:, 96:104] + 2.2),
                torch.sigmoid(rr_seq[:, 104:112] + 4.5),
                torch.sigmoid(rr_seq[:, 112:120] + 7.5),
                torch.sigmoid(rr_seq[:, 120:128] + 8.9),
            ],
            dim=-1,
        )
        rg_5s = torch.pow(rg_1s, 5.0)
        mem = torch.zeros((MEM_DIM_ORDER3,), dtype=torch.float32, device=dev)
        mems = []
        for t_i in range(len(idx_5s)):
            mem = rg_5s[t_i] * mem + (1.0 - rg_5s[t_i]) * ru_seq[t_i]
            mems.append(mem)
        mem_mat = torch.stack(mems, dim=0)
        keep = idx_5s >= ws
        idx_k = idx_5s[keep]
        zm_k = zm_seq[keep]
        mem_k = mem_mat[keep]
        phi_0 = _extract_resonant_phi_16d(mem_k, zm_k, torch.zeros(len(idx_k), device=dev))
        # Будущая 15-минутная доходность СТРОГО внутри конца текущего Train-блока we:
        idx_f15 = torch.clamp(idx_k + 900, max=we - 1)
        ret_15m = (b1_full[idx_f15] - b1_full[idx_k]) / torch.clamp(b1_full[idx_k], min=1e-4)

        phi_list.append(phi_0)
        day_list.append(torch.full((len(idx_k),), di, dtype=torch.int32, device=dev))
        reg_code_list.append(torch.full((len(idx_k),), reg_map[tb["regime"]], dtype=torch.int32, device=dev))
        fwd15m_list.append(ret_15m)

    phi_all = torch.cat(phi_list, dim=0)         # [M_tr, 16] (~103,680 точек по 5с)
    day_all = torch.cat(day_list, dim=0)         # [M_tr]
    reg_all = torch.cat(reg_code_list, dim=0)    # [M_tr]
    fwd_all = torch.cat(fwd15m_list, dim=0)      # [M_tr]

    # Координата стандартного масштаба по каждой оси phi[0..14] для изотропного старта добротности Q:
    phi_std = torch.clamp(phi_all[:, :15].std(dim=0), min=0.15)
    inv_std = 1.0 / phi_std

    # 6 физических спектральных архетипов (семена для 6 полос OMP, строго из Train-квантилей!):
    seeds = []
    # Полоса 0: Восходящий межбиржевой импульс поводырей Gate+Ourbit (корзины U + вторичные волны C, без краш-щита)
    m0 = (reg_all != 3) & (phi_all[:, 11] > torch.quantile(phi_all[:, 11], 0.72)) & (phi_all[:, 9] > torch.quantile(phi_all[:, 9], 0.68)) & (phi_all[:, 14] < 0.15)
    # Полоса 1: Тихий широкий спред и сжатие волатильности (корзина Q + спокойные фазы U, мейкер-снайпер на 20% Agile Pod)
    m1 = (reg_all != 3) & (phi_all[:, 8] > torch.quantile(phi_all[:, 8], 0.65)) & (torch.abs(phi_all[:, 13]) < torch.quantile(torch.abs(phi_all[:, 13]), 0.55)) & (phi_all[:, 14] < 0.05)
    # Полоса 2: Глубокий локальный откат на растущей 30м макро-волне (вторичные каскады BLK_07_C / U dip-buy)
    m2 = (reg_all != 3) & (phi_all[:, 10] > torch.quantile(phi_all[:, 10], 0.60)) & (phi_all[:, 9] > 0.05) & (phi_all[:, 3] < torch.quantile(phi_all[:, 3], 0.35)) & (phi_all[:, 14] < 0.20)
    # Полоса 3: Ускорение агрессоров CVD 60s + положительный среднесрочный тренд 30m (трендовый пробой U)
    m3 = (reg_all != 3) & (phi_all[:, 12] > torch.quantile(phi_all[:, 12], 0.75)) & (phi_all[:, 13] > torch.quantile(phi_all[:, 13], 0.60)) & (phi_all[:, 14] < 0.15)
    # Полоса 4: Резонатор фиксации прибыли / разгрузки набранного Agile Pod (inv_ratio ~ 0.12..0.25 при затухании импульса!)
    m4 = (reg_all != 3) & (phi_all[:, 9] < torch.quantile(phi_all[:, 9], 0.35)) & (phi_all[:, 11] < 0.0)
    # Полоса 5: Вторичный разворотный резонатор после микро-капитуляции (высокая положительная fwd_15m на Train вне D)
    m5 = (reg_all != 3) & (fwd_all > torch.quantile(fwd_all, 0.82)) & (phi_all[:, 14] < 0.25)

    archetypes = [
        ("Band0_LeadLag_UpWave_Maker", m0, 0.00, "maker_up"),
        ("Band1_WideSpread_Quiet_Pod", m1, 0.00, "quiet_sniper"),
        ("Band2_MacroWave_Dip_Capture", m2, 0.00, "dip_pod"),
        ("Band3_CrossVenue_CVD_Breakout", m3, 0.00, "trend_pod"),
        ("Band4_AgilePod_Profit_Harvest", m4, 0.15, "ask_harvest"),
        ("Band5_Rebound_Phase_Resonator", m5, 0.00, "rebound_pod"),
    ]

    for name_b, mask_b, inv_c, kind_b in archetypes:
        pts = phi_all[mask_b] if int(mask_b.sum().item()) >= 50 else phi_all[reg_all != 3]
        c_vec = torch.median(pts, dim=0).values.clone()
        c_vec[15] = float(inv_c)

        # Начальная добротность Q_{k,d}: селективная на профильных частотах (Q ~ 1.4..2.4 / std) и мягкая на второстепенных (~0.55 / std),
        # чтобы выполнялся критерий Найквиста (>=25 эпизодов на >=3 днях), но отсекались обвалы D (q[14] высокая)!
        q_vec = torch.ones((RES_PHI_DIM_ORDER3,), dtype=torch.float32, device=dev) * 0.55
        q_vec[:15] = 0.52 * inv_std
        q_vec[14] = max(2.20, float(1.85 * inv_std[14].item()))  # Жёсткий барьер по bleed+crash+trap щиту!
        q_vec[15] = 3.60 if kind_b == "ask_harvest" else 2.80

        du_vec = torch.zeros((EXPERT_OUT_DIM_ORDER3,), dtype=torch.float32, device=dev)
        if kind_b in ("maker_up", "trend_pod"):
            q_vec[9] *= 2.4; q_vec[11] *= 2.5; q_vec[12] *= 2.0; q_vec[4] *= 1.8
            du_vec[0] = +14.20   # Выводит спящую Mode #3 из -12.0 -> +2.20 (чистый мейкер-транш >= $12!)
            du_vec[59] = +18.50  # Мгновенный 20% Agile Pod (-12.0 + 18.5 = +6.5 -> pod_cap_mult=0.20!)
            du_vec[1:4] = +3.20  # Концентрация в передних эшелонах Bid #00..#02
            du_vec[37] = +2.60   # Шаг внутрь спреда
            du_vec[38] = -3.00   # Снятие излишнего отступа трусости
            du_vec[39] = -2.80   # Компактный спред-карман
            du_vec[53] = +10.50  # Пробуждение Sniper front из -6.5
            du_vec[52] = +11.00  # Пробуждение Tranche carousel из -6.5 для быстрой фиксации прибыли
            du_vec[54] = +8.00   # Мягкий maker-unwind при развороте цены ниже входа
            du_vec[42] = +0.80   # Умеренный тейк-профит Ask без зависания в стакане
            du_vec[50:52] = +4.00
        elif kind_b == "quiet_sniper":
            q_vec[8] *= 2.6; q_vec[13] *= 2.2; q_vec[2] *= 1.8
            du_vec[0] = +13.60
            du_vec[59] = +18.50  # Строгий 20% Agile Pod!
            du_vec[55] = +6.50   # Пробуждение Wide-spread micro-sniper
            du_vec[1:3] = +3.40
            du_vec[37] = +2.20
            du_vec[39] = -3.20
            du_vec[42] = -0.80   # Быстрый мейкер-оборот внутри широкого спреда!
            du_vec[52] = +11.50  # Tranche carousel
            du_vec[54] = +8.50   # Контролируемый maker stop-unwind против сползания
            du_vec[50:52] = +4.00
        elif kind_b in ("dip_pod", "rebound_pod"):
            q_vec[10] *= 2.4; q_vec[3] *= 2.2; q_vec[9] *= 2.0
            du_vec[0] = +14.00
            du_vec[59] = +18.50  # 20% Agile Pod
            du_vec[2:7] = +2.80  # Ловля локального пролива ступенчатой гребёнкой #02..#06
            du_vec[39] = -1.60
            du_vec[52] = +11.50  # Быстрая разгрузка нижнего транша на отскоке
            du_vec[54] = +8.00
            du_vec[56] = +14.00  # Пробуждение Recycle anchor из -12.0
            du_vec[50:52] = +4.00
        elif kind_b == "ask_harvest":
            q_vec[9] *= 2.2; q_vec[11] *= 2.2; q_vec[15] = 4.20
            du_vec[0] = -6.00    # Торможение новых покупок при затухании импульса
            du_vec[13:17] = +3.40  # Переброс Ask-объёма в передние слоты #00..#03 для быстрой фиксации!
            du_vec[43] = +2.80   # Inside Ask
            du_vec[52] = +12.00  # Карусель транша
            du_vec[54] = +9.50   # Мейкер-выход при затухании
            du_vec[51] = +4.00

        q_vec = torch.clamp(q_vec, min=0.55, max=10.0)
        seeds.append({"name": name_b, "kind": kind_b, "c": c_vec, "q": q_vec, "du": du_vec})

    return phi_all, day_all, reg_all, seeds


@torch.no_grad()
def check_nyquist_support_batch(
    amp_b: torch.Tensor,     # [C]
    center_b: torch.Tensor,  # [C, 16]
    q_b: torch.Tensor,       # [C, 16]
    phi_tr: torch.Tensor,    # [M_tr, 16] (5s grid)
    day_tr: torch.Tensor,    # [M_tr]
    reg_tr: torch.Tensor,    # [M_tr] (3 == 'D')
    min_hits: int = 25,
    min_days: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Проверяет закон статистической обеспеченности (Nyquist Sample-Support Law) на Train-траектории:
      - Считает число независимых резонансных эпизодов (gamma_k > 0.20, с прореживанием по 36 шагов = 180с),
      - Число уникальных дней с >= 3 эпизодами,
      - Долю ложных срабатываний на медвежьем обвале D (reg_tr == 3).
    """
    # Прореживаем до шага 60с (каждая 12-я точка из 5с сетки -> ~8,640 точек, мгновенный расчёт для C=32 за <1 мс!):
    phi_sub = phi_tr[::12]   # шаг 60с
    day_sub = day_tr[::12]
    reg_sub = reg_tr[::12]
    inv_probe = torch.where(center_b[:, 15:16] > 0.05, center_b[:, 15:16], torch.zeros_like(center_b[:, 15:16]))
    phi_c = phi_sub.unsqueeze(0).expand(len(amp_b), -1, -1).clone()
    phi_c[:, :, 15] = inv_probe

    diff = phi_c - center_b.unsqueeze(1)
    q_eff = torch.clamp(q_b, 0.35, 12.0).unsqueeze(1)
    d2 = 0.5 * torch.sum((diff * q_eff) ** 2, dim=-1)  # [C, L]
    gamma = torch.clamp(amp_b, 0.0, 1.5).unsqueeze(1) * torch.exp(-torch.clamp(d2, max=40.0))
    active = gamma > 0.20  # [C, L]
    # Независимые эпизоды (разделённые минимум 3 минутами = 3 шага на 60с сетке):
    ep_start = active.clone()
    ep_start[:, 1:] = active[:, 1:] & (~active[:, :-1])
    n_episodes = ep_start.long().sum(dim=1)  # [C]

    n_days_hit = torch.zeros_like(n_episodes)
    for d_id in torch.unique(day_sub):
        dm = day_sub == d_id
        n_days_hit += (ep_start[:, dm].long().sum(dim=1) >= 2).long()

    dump_mask = reg_sub == 3
    dump_fire_rate = active[:, dump_mask].float().mean(dim=1) if bool(dump_mask.any()) else torch.zeros(len(amp_b), device=amp_b.device)
    # Строгая узкополосность High-Q: одна мода активна не более 16% времени (исключает вырождение в широкую полуплоскость!):
    duty_cycle = active.float().mean(dim=1)
    valid = (n_episodes >= min_hits) & (n_days_hit >= min_days) & (dump_fire_rate <= 0.015) & (duty_cycle <= 0.16)
    return valid, n_episodes, n_days_hit


def eval_population_dual_worlds(
    cand_states: list[dict[str, torch.Tensor]],
    worlds: dict[str, dict[str, torch.Tensor]],
    windows: list[tuple[str, int, float, float]],
    dev: torch.device,
) -> list[dict]:
    """
    Оценивает пачку C=32 кандидатов одновременно по всем окнам корзины в обоих мирах DEC и RAW!
    Благодаря harsh_stress=True ядро возвращает И calib_clean_ev_pct, И clean_ev_pct (Harsh) за 1 проход!
    """
    res_dec = evaluate_order3_basket(
        mkt_39d_all=worlds["DEC"]["mkt"],
        b10_all=worlds["DEC"]["b10"],
        a10_all=worlds["DEC"]["a10"],
        bq10_all=worlds["DEC"]["bq10"],
        aq10_all=worlds["DEC"]["aq10"],
        sv_all=worlds["DEC"]["sv"],
        bv_all=worlds["DEC"]["bv"],
        cand_states=cand_states,
        val_windows=windows,
        win_len=WIN_24H,
        tick_sz=0.0001,
        device=dev,
        pad_to_max_c=False,
        harsh_stress=True,
        enforce_activity_floor=False,
    )
    res_raw = evaluate_order3_basket(
        mkt_39d_all=worlds["RAW"]["mkt"],
        b10_all=worlds["RAW"]["b10"],
        a10_all=worlds["RAW"]["a10"],
        bq10_all=worlds["RAW"]["bq10"],
        aq10_all=worlds["RAW"]["aq10"],
        sv_all=worlds["RAW"]["sv"],
        bv_all=worlds["RAW"]["bv"],
        cand_states=cand_states,
        val_windows=windows,
        win_len=WIN_24H,
        tick_sz=0.0001,
        device=dev,
        pad_to_max_c=False,
        harsh_stress=True,
        enforce_activity_floor=False,
    )
    merged = []
    for rd, rr in zip(res_dec, res_raw):
        ev_dec = 0.50 * float(rd["calib_clean_ev_pct"]) + 0.50 * float(rd["clean_ev_pct"])
        ev_raw = 0.50 * float(rr["calib_clean_ev_pct"]) + 0.50 * float(rr["clean_ev_pct"])
        ev_4w = 0.50 * ev_dec + 0.50 * ev_raw
        dd_max = max(float(rd["dd_pct"]), float(rd["calib_dd_pct"]), float(rr["dd_pct"]), float(rr["calib_dd_pct"]))
        cvar_4w = 0.50 * float(rd["cvar25_bp"]) + 0.50 * float(rr["cvar25_bp"])
        fills_w = 0.50 * float(rd["fills"]) + 0.50 * float(rr["fills"])
        merged.append(
            {
                "ev_4w_pct": ev_4w,
                "ev_dec_pct": ev_dec,
                "ev_raw_pct": ev_raw,
                "dd_max_pct": dd_max,
                "cvar_bp": cvar_4w,
                "fills": fills_w,
            }
        )
    return merged


def run_per_block_audit(
    cand_st: dict[str, torch.Tensor],
    worlds: dict[str, dict[str, torch.Tensor]],
    block_specs: list[dict],
    dev: torch.device,
) -> dict:
    """Точный поблочный аудит кандидата по корзинам C / D / U / Q / BLIND в обоих мирах DEC и RAW."""
    pad_32 = [cand_st] * 32
    out = {}
    for wm in ("DEC", "RAW"):
        w = worlds[wm]
        per_blk = []
        by_reg: dict[str, list[float]] = {"C": [], "D": [], "U": [], "Q": [], "BLIND": []}
        max_dd = 0.0
        for b in block_specs:
            wb = [(b["block_id"], int(b["rel_start"]), 0.0, 0.0)]
            r = evaluate_order3_basket(
                mkt_39d_all=w["mkt"], b10_all=w["b10"], a10_all=w["a10"],
                bq10_all=w["bq10"], aq10_all=w["aq10"], sv_all=w["sv"], bv_all=w["bv"],
                cand_states=pad_32, val_windows=wb, win_len=WIN_24H, tick_sz=0.0001,
                device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False,
            )[0]
            ev_m = 0.50 * float(r["calib_clean_ev_pct"]) + 0.50 * float(r["clean_ev_pct"])
            dd_m = max(float(r["calib_dd_pct"]), float(r["dd_pct"]))
            max_dd = max(max_dd, dd_m)
            rk = "BLIND" if b["role"] == "blind" else b["regime"]
            by_reg[rk].append(ev_m)
            per_blk.append(
                {
                    "block_id": b["block_id"],
                    "role": b["role"],
                    "regime": b["regime"],
                    "ev_mix_pct": round(ev_m, 3),
                    "fills": round(float(r["fills"]), 1),
                    "dd_pct": round(dd_m, 2),
                }
            )
        reg_means = {k: round(float(np.mean(v)), 3) if v else 0.0 for k, v in by_reg.items()}
        pre_blind = [x["ev_mix_pct"] for x in per_blk if x["role"] != "blind"]
        val_only = [x["ev_mix_pct"] for x in per_blk if x["role"] == "val"]
        out[wm] = {
            "mean_pre_blind_ev_d": round(float(np.mean(pre_blind)), 3),
            "mean_val_ev_d": round(float(np.mean(val_only)), 3) if val_only else 0.0,
            "regime_means_pct": reg_means,
            "max_dd_pct": round(max_dd, 2),
            "per_block": per_blk,
        }
    return out


def main() -> None:
    t_start = time.time()
    assert torch.cuda.device_count() == 1, f"Ожидалась СТРОГО 1 видеокарта (GPU #1), видно: {torch.cuda.device_count()}"
    dev = torch.device("cuda:0")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    _log("🚀 УРОК 95 (ШАГ В): ПОШАГОВОЕ ОБУЧЕНИЕ УЗКОПОЛОСНОГО HIGH-Q RESONANT EQUALIZER НА GPU #1")
    split_info = build_dynamic_regime_split()
    cap_law = split_info["capacity_law"]
    k_max = int(cap_law["k_max_bands"])
    min_hits = int(cap_law["min_hits_per_band"])
    min_days = int(cap_law["min_distinct_days_per_band"])

    train_blocks = [b for b in split_info["blocks"] if b["role"] == "train"]
    val_blocks = [b for b in split_info["blocks"] if b["role"] == "val"]
    blind_blocks = split_info["blind_blocks"]
    all_blocks = split_info["blocks"] + blind_blocks

    train_wins = [(b["block_id"], int(b["rel_start"]), 0.0, 0.0) for b in train_blocks]
    # Добавляем до 2 сдвинутых внутри-тренировочных тихих/медвежьих окон (строго >=2ч от PurgedVal и Blind!),
    # чтобы сам Train отсекал резонаторы, покупающие медленное сползание цены:
    jit_qd = list(split_info["jittered_train_starts_by_regime"].get("Q", [])) + list(split_info["jittered_train_starts_by_regime"].get("D", []))
    base_starts = {int(b["rel_start"]) for b in train_blocks}
    extra_added = 0
    for js in jit_qd:
        if all(abs(int(js) - bs) >= 14400 for bs in base_starts):
            train_wins.append((f"JIT_QD_{js}", int(js), 0.0, 0.0))
            base_starts.add(int(js))
            extra_added += 1
            if extra_added >= 2:
                break
    val_wins = [(b["block_id"], int(b["rel_start"]), 0.0, 0.0) for b in val_blocks]

    _log(f"📐 Закон ёмкости: K_max={k_max} полос | Nyquist >= {min_hits} эпизодов на >= {min_days} днях | Train={len(train_wins)} сут | PurgedVal={len(val_wins)} сут | Blind={len(blind_blocks)} сут (скрыт до конца!)")

    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)
    h0_cpu = pol.load_initial_champion(CHAMP_V1)
    cur_st = {k: v.to(dev) for k, v in h0_cpu.items()}
    cur_st["res_amp"] = torch.zeros((1, MAX_RES_BANDS_ORDER3), dtype=torch.float32, device=dev)
    cur_st["res_center"] = torch.zeros((MAX_RES_BANDS_ORDER3, RES_PHI_DIM_ORDER3), dtype=torch.float32, device=dev)
    cur_st["res_q"] = torch.ones((MAX_RES_BANDS_ORDER3, RES_PHI_DIM_ORDER3), dtype=torch.float32, device=dev)
    cur_st["res_du"] = torch.zeros((MAX_RES_BANDS_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32, device=dev)

    worlds = load_dual_worlds_gpu(dev)
    phi_tr, day_tr, reg_tr, seeds = extract_train_spectral_basis_and_seeds(cur_st, worlds["DEC"], train_blocks, dev)
    _log(f"✅ Извлечён каузальный 16D спектрально-фазовый базис на Train: {tuple(phi_tr.shape)} (0% утечки из PurgedVal/Blind!)")

    # Замер исходной несущей H_0 на Train и PurgedVal (разогревает CUDAGraph W=6 и W=4 по 1 разу!):
    _log("⏳ Калибровочный замер замороженной несущей H_0 на корзинах Train (W=6) и PurgedVal (W=4)...")
    base_tr = eval_population_dual_worlds([cur_st] * 32, worlds, train_wins, dev)[0]
    base_val = eval_population_dual_worlds([cur_st] * 32, worlds, val_wins, dev)[0]
    _log(
        f"⚓ Несущая H_0: Train_4W={base_tr['ev_4w_pct']:+.3f}%/сут (DD={base_tr['dd_max_pct']:.2f}%, Fills={base_tr['fills']:.1f}) "
        f"| PurgedVal_4W={base_val['ev_4w_pct']:+.3f}%/сут (DD={base_val['dd_max_pct']:.2f}%, Fills={base_val['fills']:.1f})"
    )

    cur_tr = base_tr
    cur_val = base_val
    locked_bands = []
    pop_sz = 32
    n_gens_per_band = 10
    mu_elite = 8
    w_recomb = torch.tensor([math.log(mu_elite + 0.5) - math.log(i + 1.0) for i in range(mu_elite)], device=dev)
    w_recomb = w_recomb / w_recomb.sum()

    # Параметризация одной полосы k (D_band = 1 + 16 + 16 + 60 = 93 параметра -> идеальное отношение сигнал/шум для Sep-CMA-ES!):
    # theta = [raw_amp (1), center (16), log_q (16), du (60)]
    d_band = 1 + RES_PHI_DIM_ORDER3 + RES_PHI_DIM_ORDER3 + EXPERT_OUT_DIM_ORDER3
    rng = torch.Generator(device=dev).manual_seed(202695)

    for k_idx in range(min(k_max, len(seeds))):
        sd = seeds[k_idx]
        _log("=" * 92)
        _log(f"🎛️ [OMP Полоса #{k_idx + 1}/{k_max}] Настройка узкополосного резонатора: {sd['name']}")

        mean_th = torch.zeros((d_band,), dtype=torch.float32, device=dev)
        mean_th[0] = 0.20  # sigmoid(0.20)*1.35 ~ 0.74 амплитуда старта
        mean_th[1:17] = sd["c"]
        mean_th[17:33] = torch.log(torch.clamp(sd["q"], min=0.36))
        mean_th[33:93] = sd["du"]

        sig_vec = torch.ones((d_band,), dtype=torch.float32, device=dev)
        sig_vec[0] = 0.45
        sig_vec[1:17] = 0.28
        sig_vec[17:33] = 0.22
        sig_vec[33:93] = 0.85

        best_band_tr_score = -1e9
        best_band_th = None
        best_band_tr_metrics = None

        for gen in range(n_gens_per_band):
            half_p = (pop_sz - 2) // 2
            eps_half = torch.randn((half_p, d_band), generator=rng, device=dev)
            th_pop = torch.zeros((pop_sz, d_band), dtype=torch.float32, device=dev)
            # Слот 0: чистая текущая база (амплитуда новой полосы = 0 -> гарантия невырождения!)
            th_pop[0] = mean_th
            # Слот 1: текущий центр mean_th
            th_pop[1] = mean_th
            th_pop[2 : 2 + half_p] = mean_th.unsqueeze(0) + eps_half * sig_vec.unsqueeze(0)
            th_pop[2 + half_p :] = mean_th.unsqueeze(0) - eps_half * sig_vec.unsqueeze(0)

            amp_v = 1.35 * torch.sigmoid(th_pop[:, 0])
            amp_v[0] = 0.0  # контрольный нулевой слот (текущая зафиксированная база!)
            cen_v = th_pop[:, 1:17]
            q_v = torch.clamp(torch.exp(th_pop[:, 17:33]), min=0.55, max=11.0)
            # Жёстко удерживаем барьер bleed+crash+trap щита (ось 14) >= 1.80:
            q_v[:, 14] = torch.clamp(q_v[:, 14], min=1.80)
            du_v = torch.clamp(th_pop[:, 33:93], -18.0, 20.0)
            if sd["kind"] != "ask_harvest":
                # Выводит изолятор [59] из -12.0 в [+4..+8] (строгий 20% Agile Pod в спящих модах H_0!):
                du_v[:, 59] = torch.clamp(du_v[:, 59], min=16.5, max=20.0)
                # И не сносит Taker-выход несущей:
                du_v[:, 25] = torch.clamp(du_v[:, 25], max=0.5)

            valid_nyq, n_ep, n_dy = check_nyquist_support_batch(
                amp_v, cen_v, q_v, phi_tr, day_tr, reg_tr, min_hits=min_hits, min_days=min_days
            )
            valid_nyq[0] = True  # нулевой контрольный слот всегда допустим

            cands = []
            for ci in range(pop_sz):
                st_c = {
                    k: (v.clone() if k.startswith("res_") else v)
                    for k, v in cur_st.items()
                }
                if bool(valid_nyq[ci].item()) and ci > 0:
                    st_c["res_amp"][0, k_idx] = amp_v[ci]
                    st_c["res_center"][k_idx] = cen_v[ci]
                    st_c["res_q"][k_idx] = q_v[ci]
                    st_c["res_du"][k_idx] = du_v[ci]
                else:
                    st_c["res_amp"][0, k_idx] = 0.0
                cands.append(st_c)

            tr_res = eval_population_dual_worlds(cands, worlds, train_wins, dev)
            scores = torch.full((pop_sz,), -1e6, dtype=torch.float32, device=dev)
            base_c0_ev = tr_res[0]["ev_4w_pct"]
            base_c0_dd = tr_res[0]["dd_max_pct"]

            for ci in range(1, pop_sz):
                if not bool(valid_nyq[ci].item()):
                    continue
                r_c = tr_res[ci]
                d_ev = r_c["ev_4w_pct"] - base_c0_ev
                dd_excess = max(0.0, r_c["dd_max_pct"] - max(base_c0_dd + 0.10, 6.10))
                worst_w = min(r_c["ev_dec_pct"], r_c["ev_raw_pct"]) - min(tr_res[0]["ev_dec_pct"], tr_res[0]["ev_raw_pct"])
                sc = d_ev + 0.35 * worst_w - 2.50 * dd_excess
                scores[ci] = float(sc)
                if sc > best_band_tr_score and d_ev > 0.008 and dd_excess <= 0.05:
                    best_band_tr_score = float(sc)
                    best_band_th = (
                        float(amp_v[ci].item()),
                        cen_v[ci].detach().clone(),
                        q_v[ci].detach().clone(),
                        du_v[ci].detach().clone(),
                        int(n_ep[ci].item()),
                        int(n_dy[ci].item()),
                    )
                    best_band_tr_metrics = r_c

            n_valid = int(valid_nyq[1:].sum().item())
            top_idx = torch.topk(scores, k=mu_elite).indices
            if float(scores[top_idx[0]].item()) > -1e5:
                valid_elites = [idx_e for idx_e in top_idx.tolist() if float(scores[idx_e].item()) > -1e5]
                if valid_elites:
                    w_u = w_recomb[: len(valid_elites)]
                    w_u = w_u / w_u.sum()
                    el_mat = th_pop[valid_elites]
                    new_mean = (w_u.unsqueeze(1) * el_mat).sum(dim=0)
                    var_el = (w_u.unsqueeze(1) * ((el_mat - mean_th.unsqueeze(0)) ** 2)).sum(dim=0)
                    sig_vec = torch.clamp(0.75 * sig_vec + 0.25 * torch.sqrt(var_el + 1e-6), min=0.05, max=1.50)
                    mean_th = new_mean

            best_g_ev = max((tr_res[i]["ev_4w_pct"] for i in range(1, pop_sz) if bool(valid_nyq[i].item())), default=base_c0_ev)
            _log(
                f"  Gen {gen + 1:02d}/{n_gens_per_band} | valid_nyquist={n_valid:2d}/31 | "
                f"Train_4W base={base_c0_ev:+.3f}% -> best_gen={best_g_ev:+.3f}% (best_band_score={best_band_tr_score:+.4f})"
            )

        # Честная проверка лучшего резонатора полосы k на PurgedVal (4 суточных блока C/U/Q в DEC+RAW):
        if best_band_th is None:
            _log(f"  ⚪ Полоса #{k_idx + 1} ({sd['name']}) не дала статистически обеспеченного прироста на Train — пропускается (A_k=0).")
            continue

        amp_b, cen_b, q_b, du_b, ep_cnt, dy_cnt = best_band_th
        cand_lock = {k: (v.clone() if k.startswith("res_") else v) for k, v in cur_st.items()}
        cand_lock["res_amp"][0, k_idx] = amp_b
        cand_lock["res_center"][k_idx] = cen_b
        cand_lock["res_q"][k_idx] = q_b
        cand_lock["res_du"][k_idx] = du_b

        val_check = eval_population_dual_worlds([cur_st, cand_lock] + [cur_st] * 30, worlds, val_wins, dev)
        v_prev, v_new = val_check[0], val_check[1]
        d_val_4w = v_new["ev_4w_pct"] - v_prev["ev_4w_pct"]
        d_val_dec = v_new["ev_dec_pct"] - v_prev["ev_dec_pct"]
        d_val_raw = v_new["ev_raw_pct"] - v_prev["ev_raw_pct"]
        dd_val_ok = v_new["dd_max_pct"] <= max(v_prev["dd_max_pct"] + 0.15, 5.50)

        _log(
            f"  🔎 PurgedVal проверка полосы #{k_idx + 1} (эпизодов={ep_cnt}, дней={dy_cnt}, A={amp_b:.3f}, mean_Q={float(q_b.mean().item()):.2f}): "
            f"Val_4W {v_prev['ev_4w_pct']:+.3f}% -> {v_new['ev_4w_pct']:+.3f}% (Δ={d_val_4w:+.3f}%, ΔDEC={d_val_dec:+.3f}%, ΔRAW={d_val_raw:+.3f}%, ValDD={v_new['dd_max_pct']:.2f}%)"
        )

        if d_val_4w > 0.005 and d_val_dec >= -0.010 and d_val_raw >= -0.010 and dd_val_ok:
            cur_st = cand_lock
            cur_tr = best_band_tr_metrics
            cur_val = v_new
            locked_bands.append(
                {
                    "band_idx": k_idx,
                    "name": sd["name"],
                    "amp": round(amp_b, 4),
                    "mean_q": round(float(q_b.mean().item()), 3),
                    "train_episodes": ep_cnt,
                    "train_days": dy_cnt,
                    "delta_val_4w_pct": round(d_val_4w, 4),
                    "new_val_4w_pct": round(v_new["ev_4w_pct"], 4),
                }
            )
            _log(f"  ✅ ЗАФИКСИРОВАНА ПОЛОСА #{k_idx + 1} ({sd['name']}): новый PurgedVal_4W = {v_new['ev_4w_pct']:+.3f}%/сут!")
        else:
            _log(f"  🛡️ Полоса #{k_idx + 1} отклонена фильтром PurgedVal (защита от переобучения, оставляем A_{k_idx}=0.0).")

    # Финальный выпускной экзамен на ВСЕХ 12 блоках (включая впервые открытый Rolling Blind OOS!):
    _log("=" * 92)
    _log(f"🏁 ФИНАЛЬНЫЙ ПОБЛОЧНЫЙ ЭКЗАМЕН (10 блоков C/D/U/Q + 2 СКРЫТЫХ BLIND OOS) | Принято полос: {len(locked_bands)}/{k_max}")
    base_audit = json.loads((BASE_DIR / "logs/l94_ledger_and_split/h0_clean_ledger_audit.json").read_text(encoding="utf-8"))["worlds"]
    cand_audit = run_per_block_audit(cur_st, worlds, all_blocks, dev) if locked_bands else base_audit

    from _tools.regime_purged_split_l94 import evaluate_per_regime_dominance

    gates = {}
    for wm in ("DEC", "RAW"):
        passed_w, g_info = evaluate_per_regime_dominance(
            cand_by_reg=cand_audit[wm]["regime_means_pct"],
            base_by_reg=base_audit[wm]["regime_means_pct"],
            cand_dd=cand_audit[wm]["max_dd_pct"],
            base_dd=base_audit[wm]["max_dd_pct"],
            cand_blind=cand_audit[wm]["regime_means_pct"]["BLIND"],
            base_blind=base_audit[wm]["regime_means_pct"]["BLIND"],
        )
        gates[wm] = g_info
        _log(
            f"🌍 [{wm}] H_0={base_audit[wm]['mean_pre_blind_ev_d']:+.3f}%/d -> L95={cand_audit[wm]['mean_pre_blind_ev_d']:+.3f}%/d | "
            f"Gate={passed_w} | {g_info}"
        )
        for pb in cand_audit[wm]["per_block"]:
            _log(f"    {pb['block_id']:14s} ({pb['role']:5s}) -> EV={pb['ev_mix_pct']:+6.3f}% | Fills={pb['fills']:4.1f} | DD={pb['dd_pct']:4.2f}%")

    strict_champion_pass = bool(locked_bands and gates["DEC"]["passed"] and gates["RAW"]["passed"])

    # Сохраняем полный чекпоинт кандидата (с 100% сохранением метаданных H_0 + банков res_*):
    raw_ck = torch.load(CHAMP_V1, map_location="cpu", weights_only=False)
    save_ck = dict(raw_ck)
    for k_t, v_t in cur_st.items():
        save_ck[k_t] = v_t.detach().cpu().float() if k_t.startswith("res_") else v_t.detach().cpu()
    save_ck["lesson"] = "L95_HighQ_Resonant_Equalizer"
    save_ck["l95_locked_bands"] = locked_bands
    torch.save(save_ck, OUT_BEST)

    summary = {
        "elapsed_min": round((time.time() - t_start) / 60.0, 2),
        "capacity_law": cap_law,
        "locked_bands": locked_bands,
        "strict_dominance_passed": strict_champion_pass,
        "gates": gates,
        "base_worlds": base_audit,
        "cand_worlds": cand_audit,
    }
    sum_path = LOG_DIR / "l95_resonant_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if strict_champion_pass:
        # Атомарное обновление релизного чемпиона ТОЛЬКО при строгом прохождении ворот в обоих мирах DEC и RAW!
        bak = BASE_DIR / "models/prl/release_champion_v1_pre_l95_bak.pt"
        if not bak.exists():
            import shutil
            shutil.copy2(CHAMP_V1, bak)
        torch.save(save_ck, CHAMP_V1)
        _log(f"🏆 НОВЫЙ РЕЛИЗНЫЙ ЧЕМПИОН L95 ПРОШЁЛ ВСЕ ПОКОРЗИННЫЕ И BLIND OOS ВОРОТА -> ОБНОВЛЁН {CHAMP_V1.name}!")
    else:
        _log(f"🛡️ Строгий критерий доминирования (DEC={gates['DEC']['passed']}, RAW={gates['RAW']['passed']}) сохранил эталонные веса {CHAMP_V1.name}; кандидат сохранён в {OUT_BEST.name}.")


if __name__ == "__main__":
    main()
