#!/usr/bin/env python3
"""
_tools/train_order3_champion.py
=============================================================================
Order-3 Multimodal MoE RL Champion Trainer (39D Market + 110D Agent = 149D)
=============================================================================
Ключевые особенности:
1. 100% Data-Driven (0 эмпирических порогов, 0 хардкодных весов бирж).
2. Прямой доступ к 39D кэшу (16D Alpha 1 порядка + 11D Risk 2 порядка + 12D LOB/Basis).
3. 24 слота исполнения (12 Bid + 12 Ask) с Token-Bucket (+0.35/с) и защитой от тремора.
4. K=4 Latent MoE + Fanned Slot Softmax + Per-Slot Regulator.
5. Surgical warm-start из L27 champion (`w1_ag` переносится 1-в-1, `w1_mkt` обучается с нуля).
6. Population-Based Optimization (PBT) с кольцевой антитетической мутацией.
=============================================================================
"""

from __future__ import annotations

import gc
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.fused_order3_clean_tick import (
    evaluate_order3_basket,
    MKT_DIM_ORDER3,
    AG_DIM_ORDER3,
    TOTAL_DIM_ORDER3,
    MEM_DIM_ORDER3,
    K_MODES_ORDER3,
    EXPERT_OUT_DIM_ORDER3,
    MOE_OUT_DIM_ORDER3,
    ROUTER_IN_DIM_ORDER3,
    ROUTER_H_DIM_ORDER3,
    REG_IN_DIM_ORDER3,
    REG_H_DIM_ORDER3,
    TOKEN_REFILL_PER_SEC_ORDER3,
    TOKEN_BUCKET_CAP_ORDER3,
)

SAVE_DIR = BASE_DIR / "models" / "prl" / "order3_champion"
LATEST_CHAMP_PATH = SAVE_DIR / "order3_champ_latest.pt"
RESUME_CHAMP_PATH = SAVE_DIR / "order3_champ_resume.pt"
SOUP_CHAMP_PATH = SAVE_DIR / "order3_soup_top3.pt"
LOG_PATH = BASE_DIR / "data" / "logs" / "order3_champion_train.log"

L27_SEED_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1_backup_l27.pt"
ORDER3_SEED_PATH = BASE_DIR / "models" / "prl" / "release_champion_v2.pt"

MSK_TZ = timezone(timedelta(hours=3))


def log_msg(msg: str) -> None:
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts_str}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _normalize_weights_order3(state: dict, dtype: torch.dtype) -> dict:
    out = {}
    w1_f = state["w1"].float()
    w1_mkt = w1_f[..., :MKT_DIM_ORDER3, :]
    w1_ag = w1_f[..., MKT_DIM_ORDER3:, :]

    # Нормировка блока рынка и блока агента раздельно
    cap_mkt = 6.5 * math.sqrt(float(MKT_DIM_ORDER3))
    nb_mkt = torch.linalg.vector_norm(w1_mkt, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    w1_mkt_norm = w1_mkt * torch.clamp(cap_mkt / nb_mkt, max=1.0)

    cap_ag = 6.5 * math.sqrt(float(AG_DIM_ORDER3))
    nb_ag = torch.linalg.vector_norm(w1_ag, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    w1_ag_norm = w1_ag * torch.clamp(cap_ag / nb_ag, max=1.0)

    out["w1"] = torch.cat([w1_mkt_norm, w1_ag_norm], dim=-2).to(dtype)
    out["b1"] = torch.clamp(state["b1"].float(), -2.2, 2.2).to(dtype)

    w2m_f = state["w2_moe"].float()
    w2m_parts = []
    for k_i in range(K_MODES_ORDER3):
        s_k, e_k = k_i * EXPERT_OUT_DIM_ORDER3, (k_i + 1) * EXPERT_OUT_DIM_ORDER3
        wk = w2m_f[..., s_k:e_k]
        nk = torch.linalg.vector_norm(wk, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        w2m_parts.append(wk * torch.clamp(62.0 / nk, max=1.0))
    out["w2_moe"] = torch.cat(w2m_parts, dim=-1).to(dtype)
    out["b2_moe"] = torch.clamp(state["b2_moe"].float(), -7.5, 6.5).to(dtype)

    for k, cap in (
        ("w_res1", 65.0),
        ("w_res2", 40.0),
        ("w_gate1", 28.0),
        ("w_gate2", 14.0),
        ("w_rec", 30.0),
        ("w_reg1", 26.0),
        ("w_reg2", 14.0),
    ):
        wf = state[k].float()
        nw = torch.linalg.vector_norm(wf, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        out[k] = (wf * torch.clamp(cap / nw, max=1.0)).to(dtype)

    out["b_res1"] = torch.clamp(state["b_res1"].float(), -2.0, 2.0).to(dtype)
    out["b_gate1"] = torch.clamp(state["b_gate1"].float(), -2.5, 2.5).to(dtype)
    out["b_gate2"] = torch.clamp(state["b_gate2"].float(), -2.8, 2.8).to(dtype)
    out["b_rec"] = torch.clamp(state["b_rec"].float(), -2.5, 2.5).to(dtype)
    out["b_reg1_b"] = torch.clamp(state["b_reg1_b"].float(), -3.0, 3.0).to(dtype)
    out["b_reg1_a"] = torch.clamp(state["b_reg1_a"].float(), -3.0, 3.0).to(dtype)
    out["b_reg2_b"] = torch.clamp(state["b_reg2_b"].float(), -6.0, 3.0).to(dtype)
    out["b_reg2_a"] = torch.clamp(state["b_reg2_a"].float(), -6.0, 3.0).to(dtype)
    return out


class MultimodalMoEPolicyOrder3:
    PARAM_KEYS = (
        "w1", "b1", "w_res1", "b_res1", "w_res2",
        "w2_moe", "b2_moe",
        "w_gate1", "b_gate1", "w_gate2", "b_gate2",
        "w_rec", "b_rec",
        "w_reg1", "b_reg1_b", "b_reg1_a", "w_reg2", "b_reg2_b", "b_reg2_a",
    )

    def __init__(self, n_groups: int = 256, n_budgets: int = 6, device: torch.device = torch.device("cuda:0")):
        self.n_groups = n_groups
        self.n_budgets = n_budgets
        self.pop_size = n_groups * n_budgets
        self.device = device
        # Храним мастер-популяцию мутаций и OpenAI-ES строго в fp32 (предотвращает обнуление мелких шагов квантованием bf16!)
        self.dtype = torch.float32

        self.w1 = torch.empty((n_groups, TOTAL_DIM_ORDER3, 384), dtype=self.dtype, device=device)
        self.b1 = torch.empty((n_groups, 1, 384), dtype=self.dtype, device=device)
        self.w_res1 = torch.empty((n_groups, 256, 128), dtype=self.dtype, device=device)
        self.b_res1 = torch.empty((n_groups, 1, 128), dtype=self.dtype, device=device)
        self.w_res2 = torch.empty((n_groups, 64, 256), dtype=self.dtype, device=device)

        self.w2_moe = torch.empty((n_groups, 256, MOE_OUT_DIM_ORDER3), dtype=self.dtype, device=device)
        self.b2_moe = torch.empty((n_groups, 1, MOE_OUT_DIM_ORDER3), dtype=self.dtype, device=device)
        self.w_gate1 = torch.empty((n_groups, ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), dtype=self.dtype, device=device)
        self.b_gate1 = torch.empty((n_groups, 1, ROUTER_H_DIM_ORDER3), dtype=self.dtype, device=device)
        self.w_gate2 = torch.empty((n_groups, ROUTER_H_DIM_ORDER3, K_MODES_ORDER3), dtype=self.dtype, device=device)
        self.b_gate2 = torch.empty((n_groups, 1, K_MODES_ORDER3), dtype=self.dtype, device=device)

        self.w_rec = torch.empty((n_groups, 256, 2 * MEM_DIM_ORDER3), dtype=self.dtype, device=device)
        self.b_rec = torch.empty((n_groups, 1, 2 * MEM_DIM_ORDER3), dtype=self.dtype, device=device)

        self.w_reg1 = torch.empty((n_groups, REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3), dtype=self.dtype, device=device)
        self.b_reg1_b = torch.empty((n_groups, 1, REG_H_DIM_ORDER3), dtype=self.dtype, device=device)
        self.b_reg1_a = torch.empty((n_groups, 1, REG_H_DIM_ORDER3), dtype=self.dtype, device=device)
        self.w_reg2 = torch.empty((n_groups, REG_H_DIM_ORDER3, 2), dtype=self.dtype, device=device)
        self.b_reg2_b = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)
        self.b_reg2_a = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)

        # Кольца радиусов: 30% Fine (0.50x), 40% Core (1.00x), 30% Bold (1.85x)
        half = max(1, (self.n_groups - 2) // 2)
        n_fine = max(1, int(round(0.30 * half)))
        n_bold = max(1, int(round(0.30 * half)))
        n_core = max(0, half - n_fine - n_bold)
        rings = [0.50] * n_fine + [1.00] * n_core + [1.85] * n_bold
        self.ring_mult_half = torch.tensor(rings[:half], dtype=torch.float32, device=device)

        # 🧠 True Diagonal Adam-ES Moments (m_t, v_t) — без искажения антитетической симметрии +eps / -eps!
        self.momentum = {k: torch.zeros_like(self.__dict__[k][0]) for k in self.PARAM_KEYS}
        self.adam_m = {k: torch.zeros_like(self.__dict__[k][0], dtype=torch.float32) for k in self.PARAM_KEYS}
        self.adam_v = {k: torch.zeros_like(self.__dict__[k][0], dtype=torch.float32) for k in self.PARAM_KEYS}
        self.adam_t = 0
        self.has_momentum = False

    def update_momentum(self, prev_state: dict, new_state: dict, beta: float = 0.80) -> None:
        """Обновление вектора импульса вдоль градиента успеха для ускорения сходимости."""
        for k in self.PARAM_KEYS:
            if k not in prev_state or k not in new_state:
                continue
            if prev_state[k].shape != new_state[k].shape:
                continue
            diff = (new_state[k].to(device=self.device, dtype=self.dtype) - prev_state[k].to(device=self.device, dtype=self.dtype))
            norm = torch.linalg.norm(diff.float()).clamp(min=1e-6)
            diff_cl = diff * torch.clamp(0.20 / norm, max=1.0)
            if not self.has_momentum:
                self.momentum[k] = diff_cl
            else:
                self.momentum[k] = (beta * self.momentum[k].float() + (1.0 - beta) * diff_cl.float()).to(self.dtype)
        self.has_momentum = True

    def load_initial_champion(self, ck_path: Path) -> dict:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        is_order3 = str(ck.get("arch_version", "")).startswith("order3")

        if (
            is_order3
            and ck["w1"].shape[0] == TOTAL_DIM_ORDER3
            and ck["w_gate1"].shape[0] == ROUTER_IN_DIM_ORDER3
            and ck["w_reg1"].shape == (REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3)
            and ck["b2_moe"].numel() == MOE_OUT_DIM_ORDER3
            and ck["b_rec"].numel() == 2 * MEM_DIM_ORDER3
            and ck["b_gate2"].numel() == K_MODES_ORDER3
        ):
            state = {k: ck[k].to(dtype=self.dtype, device=self.device) for k in self.PARAM_KEYS}
        else:
            # Хирургический перенос L46/L45 -> L48 (32D память, 48D эксперты, 311D роутер, 36->64->2 регулятор FIFO-очереди):
            gen = torch.Generator().manual_seed(20261148)
            w1_mkt = torch.randn((MKT_DIM_ORDER3, 384), generator=gen) * 0.005
            b1_vec = ck["b1"].view(1, 384).float().clone()
            if is_order3:
                old_mkt_d = max(0, min(MKT_DIM_ORDER3, int(ck["w1"].shape[0]) - AG_DIM_ORDER3))
                if old_mkt_d > 0:
                    w1_mkt[:old_mkt_d, :] = ck["w1"][:old_mkt_d, :].float()
                w1_ag = ck["w1"][ck["w1"].shape[0] - AG_DIM_ORDER3 :, :].float().clone()
                if old_mkt_d < 60:
                    w1_mkt[12, :] = w1_mkt[12, :] / 25.0
                    b1_vec = b1_vec + 0.0020 * w1_mkt[13:14, :]
                    w1_mkt[13, :] = w1_mkt[13, :] / 150.0
                    w1_mkt[22:29, :] = w1_mkt[22:29, :] / 25.0
            else:
                w1_ag = ck["w1"][370:, :].float().clone()
            w1_full = torch.cat([w1_mkt, w1_ag], dim=0)

            wg1_old = ck["w_gate1"].float()
            if wg1_old.shape[0] != ROUTER_IN_DIM_ORDER3:
                wg1_new = torch.zeros((ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), dtype=torch.float32)
                if wg1_old.shape[0] == 295 and ROUTER_IN_DIM_ORDER3 >= 311:
                    wg1_new[:288, :] = wg1_old[:288, :]
                    wg1_new[304:311, :] = wg1_old[288:295, :]
                else:
                    cr = min(ROUTER_IN_DIM_ORDER3, wg1_old.shape[0])
                    wg1_new[:cr, :] = wg1_old[:cr, :]
            else:
                wg1_new = wg1_old

            old_k = int(ck["b_gate2"].numel())
            old_exp_d = int(ck["b2_moe"].numel()) // old_k
            w2m_old = ck["w2_moe"].float().view(256, old_k, old_exp_d)
            b2m_old = ck["b2_moe"].float().view(1, old_k, old_exp_d)
            wg2_old = ck["w_gate2"].float()
            bg2_old = ck["b_gate2"].float().view(1, old_k)

            w2m_new = torch.zeros((256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32)
            b2m_new = torch.zeros((1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32)
            wg2_new = torch.randn((ROUTER_H_DIM_ORDER3, K_MODES_ORDER3), generator=gen) * 0.03
            bg2_new = torch.zeros((1, K_MODES_ORDER3), dtype=torch.float32)

            ck_copy = min(old_k, K_MODES_ORDER3)
            ed_copy = min(old_exp_d, EXPERT_OUT_DIM_ORDER3)
            w2m_new[:, :ck_copy, :ed_copy] = w2m_old[:, :ck_copy, :ed_copy]
            b2m_new[:, :ck_copy, :ed_copy] = b2m_old[:, :ck_copy, :ed_copy]
            wg2_new[:, :ck_copy] = wg2_old[:, :ck_copy]
            bg2_new[:, :ck_copy] = bg2_old[:, :ck_copy]

            if old_exp_d <= 45 and EXPERT_OUT_DIM_ORDER3 >= 48:
                # Инициализация [47] renorm_blend = +4.5 -> 98.9% стартовый паритет с L46 с открытым градиентом для каскадного резерва!
                b2m_new[:, :, 47] = 4.50
            if old_exp_d <= 48 and EXPERT_OUT_DIM_ORDER3 >= 50:
                b2m_new[:, :, 48:50] = -7.0

            if old_k < K_MODES_ORDER3:
                ref_m = min(1, old_k - 1)
                for km in range(ck_copy, K_MODES_ORDER3):
                    w2m_new[:, km, :ed_copy] = w2m_old[:, ref_m, :ed_copy] + torch.randn((256, ed_copy), generator=gen) * 0.025
                    b2m_new[:, km, :ed_copy] = b2m_old[:, ref_m, :ed_copy]
                    wg2_new[:, km] = wg2_old[:, km % old_k] + torch.randn((ROUTER_H_DIM_ORDER3,), generator=gen) * 0.06
                bg2_new[0, ck_copy:] = float(bg2_old.mean().item()) - 0.65

            w2m_flat = w2m_new.reshape(256, MOE_OUT_DIM_ORDER3)
            b2m_flat = b2m_new.reshape(1, MOE_OUT_DIM_ORDER3)

            wrec_old = ck["w_rec"].float()
            brec_old = ck["b_rec"].float().view(1, -1)
            if brec_old.numel() != 2 * MEM_DIM_ORDER3:
                wrec_new = torch.zeros((256, 2 * MEM_DIM_ORDER3), dtype=torch.float32)
                brec_new = torch.zeros((1, 2 * MEM_DIM_ORDER3), dtype=torch.float32)
                wrec_new[:, :16] = wrec_old[:, :16]
                wrec_new[:, 32:48] = wrec_old[:, 16:32]
                brec_new[:, :16] = brec_old[:, :16]
                brec_new[:, 32:48] = brec_old[:, 16:32]
                # Мягкая инициализация 3-минутной [16..24] и 12-минутной [24..32] макро-памяти:
                wrec_new[:, 16:32] = torch.randn((256, 16), generator=gen) * 0.020
            else:
                wrec_new, brec_new = wrec_old, brec_old

            wreg1_old = ck["w_reg1"].float()
            breg1b_old = ck["b_reg1_b"].float().view(1, -1)
            breg1a_old = ck["b_reg1_a"].float().view(1, -1)
            wreg2_old = ck["w_reg2"].float()
            if wreg1_old.shape != (REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3):
                old_ri, old_rh = wreg1_old.shape
                wreg1_new = torch.zeros((REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3), dtype=torch.float32)
                breg1b_new = torch.zeros((1, REG_H_DIM_ORDER3), dtype=torch.float32)
                breg1a_new = torch.zeros((1, REG_H_DIM_ORDER3), dtype=torch.float32)
                wreg2_new = torch.zeros((REG_H_DIM_ORDER3, 2), dtype=torch.float32)
                cr_h = min(old_rh, REG_H_DIM_ORDER3)
                if old_ri == 32 and REG_IN_DIM_ORDER3 == 36:
                    wreg1_new[:24, :cr_h] = wreg1_old[:24, :cr_h]
                    wreg1_new[28:36, :cr_h] = wreg1_old[24:32, :cr_h]
                elif old_ri == 26:
                    wreg1_new[:22, :cr_h] = wreg1_old[:22, :cr_h]
                    wreg1_new[28:32, :cr_h] = wreg1_old[22:26, :cr_h]
                else:
                    cr_i = min(old_ri, REG_IN_DIM_ORDER3)
                    wreg1_new[:cr_i, :cr_h] = wreg1_old[:cr_i, :cr_h]
                breg1b_new[:, :cr_h] = breg1b_old[:, :cr_h]
                breg1a_new[:, :cr_h] = breg1a_old[:, :cr_h]
                wreg2_new[:cr_h, :] = wreg2_old[:cr_h, :]
                if REG_H_DIM_ORDER3 > cr_h:
                    wreg1_new[:, cr_h:] = torch.randn((REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3 - cr_h), generator=gen) * 0.015
            else:
                wreg1_new, breg1b_new, breg1a_new, wreg2_new = wreg1_old, breg1b_old, breg1a_old, wreg2_old

            state = {
                "w1": w1_full.to(dtype=self.dtype, device=self.device),
                "b1": b1_vec.to(dtype=self.dtype, device=self.device),
                "w_res1": ck["w_res1"].to(dtype=self.dtype, device=self.device),
                "b_res1": ck["b_res1"].view(1, 128).to(dtype=self.dtype, device=self.device),
                "w_res2": ck["w_res2"].to(dtype=self.dtype, device=self.device),
                "w2_moe": w2m_flat.to(dtype=self.dtype, device=self.device),
                "b2_moe": b2m_flat.to(dtype=self.dtype, device=self.device),
                "w_gate1": wg1_new.to(dtype=self.dtype, device=self.device),
                "b_gate1": ck["b_gate1"].view(1, ROUTER_H_DIM_ORDER3).to(dtype=self.dtype, device=self.device),
                "w_gate2": wg2_new.to(dtype=self.dtype, device=self.device),
                "b_gate2": bg2_new.to(dtype=self.dtype, device=self.device),
                "w_rec": wrec_new.to(dtype=self.dtype, device=self.device),
                "b_rec": brec_new.to(dtype=self.dtype, device=self.device),
                "w_reg1": wreg1_new.to(dtype=self.dtype, device=self.device),
                "b_reg1_b": breg1b_new.to(dtype=self.dtype, device=self.device),
                "b_reg1_a": breg1a_new.to(dtype=self.dtype, device=self.device),
                "w_reg2": wreg2_new.to(dtype=self.dtype, device=self.device),
                "b_reg2_b": ck["b_reg2_b"].view(1, 2).to(dtype=self.dtype, device=self.device),
                "b_reg2_a": ck["b_reg2_a"].view(1, 2).to(dtype=self.dtype, device=self.device),
            }

        state = _normalize_weights_order3(state, self.dtype)
        cpu_st = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.sample_population(cpu_st, stage="explore", sigma_scale=1.0)
        return cpu_st

    def sample_population(
        self,
        center_state: dict,
        stage: str = "explore",
        sigma_scale: float = 1.0,
        alt_elite_state: dict | None = None,
    ) -> None:
        base_gpu = {k: v.to(device=self.device, dtype=self.dtype) for k, v in center_state.items()}
        alt_gpu = {k: v.to(device=self.device, dtype=self.dtype) for k, v in alt_elite_state.items()} if alt_elite_state is not None else base_gpu

        for k in self.PARAM_KEYS:
            self.__dict__[k][0].copy_(base_gpu[k])
            self.__dict__[k][1].copy_(alt_gpu[k])

        half = (self.n_groups - 2) // 2
        is_zoom = stage in ("zoom1", "zoom2")
        ring_h = (torch.clamp(self.ring_mult_half, max=1.25) * 0.85) if is_zoom else self.ring_mult_half
        sig_trunk = 0.0055 * float(sigma_scale)

        def _anti_ring(shape: tuple[int, ...], scale_f: float, low_rank: int = 0) -> torch.Tensor:
            # Гибридный Low-Rank (70%) + Isotropic (30%) антитетический шум +eps / -eps: кратно повышает SNR OpenAI-ES на больших матрицах!
            r_view = ring_h.view((half,) + (1,) * len(shape))
            if low_rank > 0 and len(shape) == 2 and min(shape) > 2 * low_rank:
                d_in, d_out = shape
                u_lr = torch.randn((half, d_in, low_rank), device=self.device, dtype=torch.float32)
                v_lr = torch.randn((half, low_rank, d_out), device=self.device, dtype=torch.float32)
                lr_part = torch.bmm(u_lr, v_lr) / math.sqrt(float(low_rank))
                iso_part = torch.randn((half, d_in, d_out), device=self.device, dtype=torch.float32)
                raw_eps = 0.8366 * lr_part + 0.5477 * iso_part  # 0.8366^2 + 0.5477^2 = 0.70 + 0.30 = 1.0 (единичная дисперсия!)
                eps = raw_eps * (scale_f * r_view)
            else:
                eps = torch.randn((half, *shape), device=self.device, dtype=torch.float32) * (scale_f * r_view)
            return torch.stack([eps, -eps], dim=1).reshape(2 * half, *shape)

        dw1 = _anti_ring((TOTAL_DIM_ORDER3, 384), sig_trunk, low_rank=16).to(self.dtype)
        dw1[:, 0:1, :] *= 2.40
        dw1[:, 1:40, :] *= 1.50
        dw1[:, 40:60, :] *= 2.20
        if MKT_DIM_ORDER3 > 60:
            dw1[:, 60:MKT_DIM_ORDER3, :] *= 4.20  # ускоренное подключение 12D Wave Radar + 30m/60m структуры волн!
        dw1[:, MKT_DIM_ORDER3:, :] *= 0.90

        self.w1[2 : 2 + 2 * half].copy_(base_gpu["w1"].unsqueeze(0) + dw1)
        self.b1[2 : 2 + 2 * half].copy_(base_gpu["b1"].unsqueeze(0) + _anti_ring((1, 384), sig_trunk * 0.6).to(self.dtype))

        self.w_res1[2 : 2 + 2 * half].copy_(base_gpu["w_res1"].unsqueeze(0) + _anti_ring((256, 128), sig_trunk, low_rank=12).to(self.dtype))
        self.b_res1[2 : 2 + 2 * half].copy_(base_gpu["b_res1"].unsqueeze(0) + _anti_ring((1, 128), sig_trunk * 0.5).to(self.dtype))
        self.w_res2[2 : 2 + 2 * half].copy_(base_gpu["w_res2"].unsqueeze(0) + _anti_ring((64, 256), sig_trunk, low_rank=12).to(self.dtype))

        moe_sig = 0.0085 * float(sigma_scale)
        dw2m = _anti_ring((256, MOE_OUT_DIM_ORDER3), moe_sig, low_rank=16)
        db2m = _anti_ring((1, MOE_OUT_DIM_ORDER3), moe_sig * 1.35)
        if EXPERT_OUT_DIM_ORDER3 >= 50:
            dw2m_v = dw2m.view(-1, 256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            db2m_v = db2m.view(-1, 1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
            dw2m_v[:, :, :7, 25] *= 0.15  # защита рабочих волновых мод 0..6 от случайного включения тайм-стопа u_taker!
            db2m_v[:, :, :7, 25] *= 0.15
            dw2m_v[:, :, :, 48] *= 0.35
            db2m_v[:, :, :, 48] *= 0.35
            dw2m_v[:, :, :, 49] *= 3.40   # активная калибровка гладкого wave_hold_pct [0..+10%] по модам!
            db2m_v[:, :, :, 49] *= 3.80
            dw2m_v[:, :, :, 0] *= 1.80
            db2m_v[:, :, :, 0] *= 2.00
        self.w2_moe[2 : 2 + 2 * half].copy_(base_gpu["w2_moe"].unsqueeze(0) + dw2m.to(self.dtype))
        self.b2_moe[2 : 2 + 2 * half].copy_(base_gpu["b2_moe"].unsqueeze(0) + db2m.to(self.dtype))

        dw_g1 = _anti_ring((ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), moe_sig * 1.85, low_rank=12).to(self.dtype)
        dw_g1[:, 272:311, :] *= 2.40  # активный фокус на 32D многомасштабной памяти и макро/карусельных каналах!
        if ROUTER_IN_DIM_ORDER3 > 311:
            dw_g1[:, 311:min(320, ROUTER_IN_DIM_ORDER3), :] *= 2.80
        if ROUTER_IN_DIM_ORDER3 > 320:
            dw_g1[:, 320:, :] *= 4.50  # прямое подключение 6D Wave Radar [w_5m..p_top_15m] в роутер мод K=8!
        self.w_gate1[2 : 2 + 2 * half].copy_(base_gpu["w_gate1"].unsqueeze(0) + dw_g1)
        self.b_gate1[2 : 2 + 2 * half].copy_(base_gpu["b_gate1"].unsqueeze(0) + _anti_ring((1, ROUTER_H_DIM_ORDER3), moe_sig * 1.35).to(self.dtype))
        dw_g2 = _anti_ring((ROUTER_H_DIM_ORDER3, K_MODES_ORDER3), moe_sig * 2.00)
        db_g2 = _anti_ring((1, K_MODES_ORDER3), moe_sig * 1.40)
        self.w_gate2[2 : 2 + 2 * half].copy_(base_gpu["w_gate2"].unsqueeze(0) + dw_g2.to(self.dtype))
        self.b_gate2[2 : 2 + 2 * half].copy_(base_gpu["b_gate2"].unsqueeze(0) + db_g2.to(self.dtype))

        dw_rec = _anti_ring((256, 2 * MEM_DIM_ORDER3), sig_trunk * 1.10, low_rank=12)
        dw_rec[:, 16:32] *= 2.20  # ускоряем обучение новых 3m/12m ячеек макро-памяти
        self.w_rec[2 : 2 + 2 * half].copy_(base_gpu["w_rec"].unsqueeze(0) + dw_rec.to(self.dtype))
        self.b_rec[2 : 2 + 2 * half].copy_(base_gpu["b_rec"].unsqueeze(0) + _anti_ring((1, 2 * MEM_DIM_ORDER3), sig_trunk * 0.5).to(self.dtype))

        bias_reg_sig = 0.040 * float(sigma_scale)
        dw_reg1 = _anti_ring((REG_IN_DIM_ORDER3, REG_H_DIM_ORDER3), sig_trunk * 1.75)
        dw_reg1[:, 24:28, :] *= 2.80  # повышенная пластичность 4D каналов реальной FIFO-очереди (qprog, qa_eta, wall, hold_long)!
        self.w_reg1[2 : 2 + 2 * half].copy_(base_gpu["w_reg1"].unsqueeze(0) + dw_reg1.to(self.dtype))
        self.b_reg1_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_b"].unsqueeze(0) + _anti_ring((1, REG_H_DIM_ORDER3), sig_trunk * 1.10).to(self.dtype))
        self.b_reg1_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_a"].unsqueeze(0) + _anti_ring((1, REG_H_DIM_ORDER3), sig_trunk * 1.10).to(self.dtype))
        self.w_reg2[2 : 2 + 2 * half].copy_(base_gpu["w_reg2"].unsqueeze(0) + _anti_ring((REG_H_DIM_ORDER3, 2), sig_trunk * 1.60).to(self.dtype))
        self.b_reg2_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_b"].unsqueeze(0) + _anti_ring((1, 2), bias_reg_sig).to(self.dtype))
        self.b_reg2_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_a"].unsqueeze(0) + _anti_ring((1, 2), bias_reg_sig).to(self.dtype))

    def get_candidate_states(self) -> list[dict]:
        out = []
        for gi in range(self.n_groups):
            out.append({k: self.__dict__[k][gi].clone() for k in self.PARAM_KEYS})
        return out

    def build_openai_es_candidates(self, center_state: dict, tr_geos: np.ndarray, lr_scales: tuple[float, ...] = (0.45, 0.90, 1.45)) -> list[dict]:
        """
        Centered-Rank OpenAI-ES: агрегирует антитетичный градиент по всем (n_groups - 2)/2 парам (+eps / -eps)
        и строит синтетические градиентные точки с разными масштабами шага обучения.
        """
        half = (self.n_groups - 2) // 2
        if half < 4 or len(tr_geos) < 2 + 2 * half:
            return []
        pair_scores = tr_geos[2 : 2 + 2 * half]
        n_m = len(pair_scores)
        order = np.argsort(pair_scores)
        ranks = np.empty(n_m, dtype=np.float32)
        ranks[order] = np.arange(n_m, dtype=np.float32)
        u_centered = (ranks / max(1.0, float(n_m - 1))) - 0.5  # [-0.5, +0.5]
        w_pair = torch.from_numpy(u_centered[0::2] - u_centered[1::2]).to(device=self.device, dtype=torch.float32)
        inv_norm = 1.0 / math.sqrt(float(half))

        grad_step: dict[str, torch.Tensor] = {}
        adam_step: dict[str, torch.Tensor] = {}
        self.adam_t += 1
        b1_ad, b2_ad = 0.82, 0.96
        bc1 = 1.0 - (b1_ad ** self.adam_t)
        bc2 = 1.0 - (b2_ad ** self.adam_t)

        for k in self.PARAM_KEYS:
            p_all = self.__dict__[k][2 : 2 + 2 * half].float()
            p_plus = p_all[0::2]
            p_minus = p_all[1::2]
            eps_k = 0.5 * (p_plus - p_minus)
            w_view = w_pair.view((half,) + (1,) * (eps_k.ndim - 1))
            g_k = (w_view * eps_k).sum(dim=0) * inv_norm
            grad_step[k] = g_k

            self.adam_m[k] = b1_ad * self.adam_m[k] + (1.0 - b1_ad) * g_k
            self.adam_v[k] = b2_ad * self.adam_v[k] + (1.0 - b2_ad) * (g_k * g_k)
            m_hat = self.adam_m[k] / bc1
            v_hat = self.adam_v[k] / bc2
            raw_ad = m_hat / (torch.sqrt(v_hat) + 1e-6)
            # Масштабируем диагональный шаг Adam к физическому RMS градиента слоя g_k:
            rms_g = torch.sqrt(torch.mean(g_k * g_k) + 1e-12)
            rms_a = torch.sqrt(torch.mean(raw_ad * raw_ad) + 1e-12)
            adam_step[k] = raw_ad * (rms_g / rms_a)

        es_cands = []
        for lr_s in lr_scales:
            raw_st = {
                k: (center_state[k].to(device=self.device, dtype=torch.float32) + float(lr_s) * grad_step[k])
                for k in self.PARAM_KEYS
            }
            norm_st = _normalize_weights_order3(raw_st, self.dtype)
            es_cands.append({k: v.clone() for k, v in norm_st.items()})
        for lr_a in (0.55, 1.10, 1.80):
            ad_st = {
                k: (center_state[k].to(device=self.device, dtype=torch.float32) + float(lr_a) * adam_step[k])
                for k in self.PARAM_KEYS
            }
            norm_ad = _normalize_weights_order3(ad_st, self.dtype)
            es_cands.append({k: v.clone() for k, v in norm_ad.items()})
        return es_cands


def train_order3(
    max_iters: int = 500,
    n_groups: int = 160,
    win_len: int = 2400,
    n_val_windows: int = 28,
    device_str: str = "cuda:0",
    time_budget_hours: float = 2.0,
    island_id: int = 0,
    num_islands: int = 2,
    island_mode: str = "exploit",
    early_stop_minutes: float = 35.0,
    log_file: str | None = None,
    wf_version: str = "pure_wf_v19_l49_cvar_shield",
    oos_hours: float = 17.0,
    inv_stress_frac: float = 0.35,
) -> None:
    t_start = time.time()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    if log_file:
        LOG_PATH = Path(log_file)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_msg("=" * 90)
    log_msg(f"🚀 ORDER-3 WFv2 ISLAND #{island_id}/{num_islands} ({island_mode.upper()}) ON {device} | GROUPS={n_groups} (POP={n_groups * 6})")
    log_msg("=" * 90)

    # 1. Загрузка строго каузальных 72D данных Signal Stack v3 (Wave Radar + Micro + Macro) на тике $0.0001
    log_msg("📦 Загрузка каузальной 72D матрицы Signal Stack v3 (Wave Radar) и выделение эпохи суб-центового тика $0.0001...")
    mkt_raw = np.load(BASE_DIR / "data/cache/rl_hybrid_market_obs_72d.npz", mmap_mode="r")
    lob_raw = np.load(BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz", mmap_mode="r")

    b1_full = lob_raw["bid_p"][:, 0]
    subcent_hits = np.where(np.abs(b1_full * 100.0 - np.round(b1_full * 100.0)) > 1e-3)[0]
    subcent_offset = int(subcent_hits[0]) if len(subcent_hits) > 0 else 0
    log_msg(f"🔍 Обнаружена граница перехода биржи на тик $0.0001: idx={subcent_offset:,} (отсечено {subcent_offset:,} старых $0.01 тиков!)")

    mkt_sub = np.asarray(mkt_raw["market_obs_72d"][subcent_offset:], dtype=np.float32)
    total_rows = len(mkt_sub)
    tick_sz = 0.0001
    if win_len > total_rows // 8:
        win_len = 9001

    mkt_gpu = torch.from_numpy(mkt_sub).to(device=device, dtype=torch.bfloat16)
    b10_gpu = torch.from_numpy(np.asarray(lob_raw["bid_p"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    a10_gpu = torch.from_numpy(np.asarray(lob_raw["ask_p"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    bq10_gpu = torch.from_numpy(np.asarray(lob_raw["bid_q"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    aq10_gpu = torch.from_numpy(np.asarray(lob_raw["ask_q"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    sv_gpu = torch.from_numpy(np.asarray(lob_raw["sell_vol"][subcent_offset:], dtype=np.float32)).to(device=device)
    bv_gpu = torch.from_numpy(np.asarray(lob_raw["buy_vol"][subcent_offset:], dtype=np.float32)).to(device=device)

    # 2. Автоматическое Purged Multi-Regime волновое разбиение (Macro = 5401с = 90м волновой цикл, Micro = 1801с = 30м свинг!):
    macro_win_len = min(win_len, 5401)
    micro_win_len = 1801
    n_bask_w = max(24, n_val_windows)
    oos_ticks = max(macro_win_len + 7200, int(round(oos_hours * 3600.0)))
    oos_start = max(360000, total_rows - oos_ticks) if total_rows >= 480000 else int(total_rows * 0.80)
    oos_end_macro = total_rows - macro_win_len - 100
    oos_end_micro = total_rows - micro_win_len - 100
    oos_span = max(1, total_rows - oos_start)
    n_oos_w = max(n_bask_w, min(72, int(round(n_bask_w * (oos_span / 61200.0)))))

    # Равномерная каузальная решетка из 8 зон Val и 9 зон Train со строгим Purge >= macro_win_len + 250 (= 9251с):
    purge_gap = macro_win_len + 250
    val_span_starts = 4200
    avail_pre_oos = oos_start - purge_gap
    n_val_zones = 8
    v_lo_grid = np.linspace(38000, min(599500, max(90000, avail_pre_oos - val_span_starts - 16000)), num=n_val_zones, dtype=int)
    val_zones = []
    tr_segments = []
    prev_tr_lo = 600
    for z_i in range(n_val_zones):
        v_lo = max(prev_tr_lo + 2500 + purge_gap, int(v_lo_grid[z_i]))
        v_hi = v_lo + val_span_starts
        tr_hi = max(prev_tr_lo + 2000, v_lo - purge_gap)
        tr_segments.append((prev_tr_lo, tr_hi))
        val_zones.append((v_lo, v_hi))
        prev_tr_lo = v_hi + purge_gap
    if avail_pre_oos - prev_tr_lo >= 2500:
        tr_segments.append((prev_tr_lo, avail_pre_oos))

    # Стратифицированные профили стартового портфеля (65% чистый кэш + 35% перегруженный/просевший инвентарь для прививки от ножей!):
    inv_profiles = [(0.25, 0.004), (0.45, 0.008), (0.65, 0.014)]

    def _assign_stratified_inv(idx_arr: np.ndarray, prefix: str, frac_inv: float) -> list[tuple[str, int, float, float]]:
        out_b = []
        n_inv_target = int(round(len(idx_arr) * frac_inv))
        inv_slots = set(np.linspace(1, len(idx_arr) - 1, num=n_inv_target, dtype=int).tolist()) if n_inv_target > 0 else set()
        inv_cnt = 0
        for k_i, idx_v in enumerate(idx_arr):
            if k_i in inv_slots:
                r0, roff = inv_profiles[inv_cnt % len(inv_profiles)]
                inv_cnt += 1
                out_b.append((f"{prefix}{k_i:02d}", int(idx_v), float(r0), float(roff)))
            else:
                out_b.append((f"{prefix}{k_i:02d}", int(idx_v), 0.0, 0.0))
        return out_b

    # Двухфазная шахматная анти-оверфит корзина Val из 56 пар = 112 окон (ровно 2 чанка по 28, включая памп $2.39 и водопад $2.18->$1.59!):
    base_zc = [4, 5, 6, 7, 8, 8, 9, 9] if len(val_zones) == 8 else ([5, 5, 6, 7, 8, 9, 9] if len(val_zones) == 7 else [max(4, (7 * n_bask_w) // (4 * len(val_zones)))] * len(val_zones))
    mac_parts, mic_parts = [], []
    for (zlo, zhi), zc in zip(val_zones, base_zc):
        g_m = np.linspace(zlo, zhi, num=zc, dtype=int)
        # Шахматное фазовое смещение нечетных окон внутри зоны [zlo..zhi] ломает синхронизацию входа под шаг сетки!
        phase_off = np.array([(337 if (j % 2 == 1) else 0) for j in range(zc)], dtype=int)
        mac_parts.append(np.clip(g_m + phase_off, zlo, zhi))
        mic_parts.append(np.clip(np.linspace(zlo + 250, zhi + 3600, num=zc, dtype=int) + phase_off, zlo, zhi + 3800))
    val_macro_idx = np.concatenate(mac_parts)
    val_micro_idx = np.concatenate(mic_parts)
    val_macro_basket = _assign_stratified_inv(val_macro_idx, "VMAC", inv_stress_frac)
    val_micro_basket = _assign_stratified_inv(val_micro_idx, "VMIC", inv_stress_frac)

    oos_macro_idx = np.linspace(oos_start, oos_end_macro, num=n_oos_w, dtype=int)
    oos_micro_idx = np.linspace(oos_start + 250, oos_end_micro, num=n_oos_w, dtype=int)
    oos_macro_basket = _assign_stratified_inv(oos_macro_idx, "OMAC", 0.0)
    oos_micro_basket = _assign_stratified_inv(oos_micro_idx, "OMIC", 0.0)

    # Повышенный вес свежих волатильных и медвежьих зон (сегменты 4..8 = 73% выборки Train!):
    rng_train = np.random.default_rng(20261150 + island_id * 1009 + (int(time.time()) % 9973))
    raw_w = np.array([0.05, 0.06, 0.07, 0.09, 0.12, 0.14, 0.15, 0.16, 0.16][:len(tr_segments)], dtype=np.float64)
    tr_seg_probs = raw_w / raw_w.sum()

    def sample_train_basket(is_micro: bool = False) -> list[tuple[str, int, float, float]]:
        seg_ids = rng_train.choice(len(tr_segments), size=n_bask_w, p=tr_seg_probs)
        out_tr = []
        extra_hi = 3500 if is_micro else 0
        for k_t, sid in enumerate(seg_ids):
            lo_s, hi_s = tr_segments[int(sid)]
            s_t = int(rng_train.integers(lo_s, max(lo_s + 1, hi_s + extra_hi + 1)))
            if rng_train.random() < inv_stress_frac:
                r0, roff = inv_profiles[int(rng_train.integers(0, len(inv_profiles)))]
            else:
                r0, roff = 0.0, 0.0
            out_tr.append((f"{'TMIC' if is_micro else 'TMAC'}{k_t:02d}", s_t, float(r0), float(roff)))
        return out_tr

    def _eval_basket_chunked(
        cands: list[dict], bask: list, w_len: int, calib_jitter_std: float = 0.0, enforce_activity_floor: bool = False
    ) -> list[dict]:
        # Строго фиксированный W = n_bask_w гарантирует 100% переиспользование одного CUDA Graph в 16 ГБ VRAM без OOM!
        if len(bask) == n_bask_w:
            return evaluate_order3_basket(
                mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu,
                bq10_all=bq10_gpu, aq10_all=aq10_gpu, sv_all=sv_gpu, bv_all=bv_gpu,
                cand_states=cands, val_windows=bask, win_len=w_len, tick_sz=tick_sz, device=device,
                calib_jitter_std=calib_jitter_std, enforce_activity_floor=enforce_activity_floor,
            )
        chunks_res = []
        weights = []
        for s_c in range(0, len(bask), n_bask_w):
            sub_b = bask[s_c : s_c + n_bask_w]
            pad_n = n_bask_w - len(sub_b)
            sub_padded = sub_b + [sub_b[-1]] * pad_n if pad_n > 0 else sub_b
            r_sub = evaluate_order3_basket(
                mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu,
                bq10_all=bq10_gpu, aq10_all=aq10_gpu, sv_all=sv_gpu, bv_all=bv_gpu,
                cand_states=cands, val_windows=sub_padded, win_len=w_len, tick_sz=tick_sz, device=device,
                calib_jitter_std=calib_jitter_std, enforce_activity_floor=enforce_activity_floor,
            )
            chunks_res.append(r_sub)
            weights.append(float(len(sub_b)))
        tot_w = sum(weights)
        out_m = []
        for c_i in range(len(cands)):
            d = {}
            for k in chunks_res[0][c_i]:
                v0 = chunks_res[0][c_i][k]
                if k in ("fills", "cancels", "requeues", "clean_ev_pct", "full_ev_pct", "turn_mult"):
                    d[k] = sum(ch[c_i][k] * (w / float(n_bask_w)) for ch, w in zip(chunks_res, weights))
                elif k == "dd_pct":
                    d[k] = max(float(ch[c_i][k]) for ch in chunks_res)
                elif isinstance(v0, list):
                    d[k] = [sum(float(ch[c_i][k][j]) * w for ch, w in zip(chunks_res, weights)) / tot_w for j in range(len(v0))]
                else:
                    d[k] = sum(float(ch[c_i][k]) * w for ch, w in zip(chunks_res, weights)) / tot_w
            out_m.append(d)
        return out_m

    def eval_dual_scale(
        cands: list[dict],
        mac_bask: list,
        mic_bask: list,
        mac_len: int | None = None,
        enforce_mi: bool = False,
        calib_jitter_std: float = 0.0,
    ) -> list[dict]:
        r_mac = _eval_basket_chunked(
            cands, mac_bask, (mac_len or macro_win_len), calib_jitter_std=calib_jitter_std, enforce_activity_floor=enforce_mi
        )
        r_mic = _eval_basket_chunked(
            cands, mic_bask, micro_win_len, calib_jitter_std=calib_jitter_std, enforce_activity_floor=enforce_mi
        )
        merged = []
        for a, b in zip(r_mac, r_mic):
            # Сбалансированный двухмасштабный скор (Macro 100m + Micro 25m) + барьер против коллапса K=8 роутера в одну моду:
            mi_join = 0.5 * float(a["mode_mi_bits"]) + 0.5 * float(b["mode_mi_bits"])
            tot_f = float(a["fills"]) + float(b["fills"])
            mi_pen = (3.5 * max(0.0, 0.55 - mi_join)) if (enforce_mi and tot_f >= 2.0) else 0.0
            joint_geo = 0.50 * float(a["geo_score_bp"]) + 0.50 * (1.55 * float(b["geo_score_bp"])) - mi_pen
            tot_c = float(a["cancels"]) + float(b["cancels"])
            mp_join = [round(0.5 * float(x) + 0.5 * float(y), 4) for x, y in zip(a["mode_probs"], b["mode_probs"])]
            merged.append({
                "geo_score_bp": joint_geo,
                "macro_geo_bp": float(a["geo_score_bp"]),
                "micro_geo_bp": float(b["geo_score_bp"]),
                "entropic_growth_bp": 0.5 * float(a["entropic_growth_bp"]) + 0.5 * float(b["entropic_growth_bp"]),
                "cvar25_bp": 0.5 * float(a["cvar25_bp"]) + 0.5 * float(b["cvar25_bp"]),
                "q15_bp": 0.5 * float(a["q15_bp"]) + 0.5 * float(b["q15_bp"]),
                "q50_bp": 0.5 * float(a["q50_bp"]) + 0.5 * float(b["q50_bp"]),
                "q85_bp": 0.5 * float(a["q85_bp"]) + 0.5 * float(b["q85_bp"]),
                "full_ev_pct": float(a["full_ev_pct"]) + float(b["full_ev_pct"]),
                "clean_ev_pct": float(a["clean_ev_pct"]) + float(b["clean_ev_pct"]),
                "macro_ev_pct": float(a["clean_ev_pct"]),
                "micro_ev_pct": float(b["clean_ev_pct"]),
                "inv_delta_ev_pct": float(a.get("inv_delta_ev_pct", 0.0)) + float(b.get("inv_delta_ev_pct", 0.0)),
                "dd_pct": max(float(a["dd_pct"]), float(b["dd_pct"])),
                "fills": tot_f,
                "cancels": tot_c,
                "requeues": float(a["requeues"]) + float(b["requeues"]),
                "cnl_per_fill": tot_c / max(tot_f, 0.5),
                "eta_queue_pct": 0.5 * float(a["eta_queue_pct"]) + 0.5 * float(b["eta_queue_pct"]),
                "mode_probs": mp_join,
                "mode_mi_bits": round(mi_join, 4),
                "turn_mult": round(float(a.get("turn_mult", 0.0)) + float(b.get("turn_mult", 0.0)), 4),
            })
        return merged

    log_msg(
        f"📊 Purged Multi-Regime K={K_MODES_ORDER3} Split [{wf_version}] ({total_rows:,} тиков = {total_rows/3600:.1f}ч, InvStress={inv_stress_frac:.0%}, CVaR15 Shield!): "
        f"Train={len(tr_segments)} purged zones [600..{tr_segments[-1][1]:,}] | Val={len(val_macro_basket)+len(val_micro_basket)}w | "
        f"Blind OOS={len(oos_macro_basket)+len(oos_micro_basket)}w ({oos_span/3600:.1f}ч) [{oos_start:,}..{oos_end_micro:,}]"
    )

    # 3. Коронационный старт L56 от четверки триумфаторов всех слепых турниров (L53 Isl1 + L54 Soup + Chimera54_50 + Chimera54_53 + L53 Isl0):
    policy = MultimodalMoEPolicyOrder3(n_groups=n_groups, n_budgets=6, device=device)
    l46_bak_p = BASE_DIR / "models" / "prl" / "release_champion_v1_bak_l46.pt"
    live_v1_p = l46_bak_p if l46_bak_p.exists() else (BASE_DIR / "models" / "prl" / "release_champion_v1.pt")
    l54_soup_p = SAVE_DIR / "bak_l54" / "soup_isl0.pt"
    l53_isl1_p = SAVE_DIR / "bak_l53" / "island_1_best.pt"
    l53_isl0_p = SAVE_DIR / "bak_l53" / "island_0_best.pt"
    ch_54_50_p = SAVE_DIR / "bak_l55" / "chimera_54_50.pt"
    ch_54_53_p = SAVE_DIR / "bak_l55" / "chimera_54_53.pt"
    l50_glob_p = SAVE_DIR / "bak_l50" / "global_best.pt"
    is_warm_l43 = True
    seed_l46 = live_v1_p if live_v1_p.exists() else LATEST_CHAMP_PATH
    log_msg(f"🧩 Коронационный старт [{wf_version}]: L53 Isl1 + L54 Soup + Chimera(R54+B50) + Chimera(R54+B53) + L53 Isl0...")

    st_l46_base = policy.load_initial_champion(seed_l46)
    st_l54_soup = policy.load_initial_champion(l54_soup_p) if l54_soup_p.exists() else st_l46_base
    st_l53_isl1 = policy.load_initial_champion(l53_isl1_p) if l53_isl1_p.exists() else st_l54_soup
    st_l53_isl0 = policy.load_initial_champion(l53_isl0_p) if l53_isl0_p.exists() else st_l53_isl1
    st_ch_54_50 = policy.load_initial_champion(ch_54_50_p) if ch_54_50_p.exists() else st_l54_soup
    st_ch_54_53 = policy.load_initial_champion(ch_54_53_p) if ch_54_53_p.exists() else st_l54_soup
    st_l50_glob = policy.load_initial_champion(l50_glob_p) if l50_glob_p.exists() else st_l46_base

    def _build_router_preserving_blend(st_router: dict, st_body: dict, body_w: float = 0.40) -> dict:
        gate_keys = {"w_gate1", "b_gate1", "w_gate2", "b_gate2"}
        st_m = {}
        for k in MultimodalMoEPolicyOrder3.PARAM_KEYS:
            if k in gate_keys:
                st_m[k] = st_router[k].detach().cpu().float().clone()
            else:
                st_m[k] = (1.0 - body_w) * st_router[k].detach().cpu().float() + body_w * st_body[k].detach().cpu().float()
        st_n = _normalize_weights_order3(st_m, policy.dtype)
        return {k: v.detach().cpu().clone() for k, v in st_n.items()}

    l52_isl1_p = SAVE_DIR / "bak_l52" / "island_1_best.pt"
    l56_isl0_p = SAVE_DIR / "bak_l56" / "island_0_best.pt"
    l57_isl0_p = SAVE_DIR / "bak_l57" / "island_0_best.pt"
    l57_soup_p = SAVE_DIR / "bak_l57" / "order3_soup_top3.pt"
    st_l52_isl1 = policy.load_initial_champion(l52_isl1_p) if l52_isl1_p.exists() else st_l46_base
    st_l56_isl0 = policy.load_initial_champion(l56_isl0_p) if l56_isl0_p.exists() else st_l53_isl0
    st_l57_isl0 = policy.load_initial_champion(l57_isl0_p) if l57_isl0_p.exists() else st_l53_isl0
    st_l57_soup = policy.load_initial_champion(l57_soup_p) if l57_soup_p.exists() else st_l57_isl0

    def _build_bid_ask_split_chimera(st_bid: dict, st_ask: dict) -> dict:
        """Хирургический синтез: ствол + роутер + все Bid-каналы от st_bid (L53), Ask/Exit-каналы от st_ask (L57)!"""
        ask_ch = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 35, 36, 41, 42, 43, 45, 46]
        out_c = {k: st_bid[k].detach().cpu().float().clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        w2_b = out_c["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2_b = out_c["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        w2_a = st_ask["w2_moe"].detach().cpu().float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2_a = st_ask["b2_moe"].detach().cpu().float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        for ch_i in ask_ch:
            w2_b[:, :, ch_i] = w2_a[:, :, ch_i]
            b2_b[:, :, ch_i] = b2_a[:, :, ch_i]
        out_c["w2_moe"] = w2_b.reshape(256, MOE_OUT_DIM_ORDER3)
        out_c["b2_moe"] = b2_b.reshape(1, MOE_OUT_DIM_ORDER3)
        out_c["b_reg1_a"] = st_ask["b_reg1_a"].detach().cpu().float().clone()
        out_c["b_reg2_a"] = st_ask["b_reg2_a"].detach().cpu().float().clone()
        st_n = _normalize_weights_order3(out_c, policy.dtype)
        return {k: v.to(device=device, dtype=policy.dtype) for k, v in st_n.items()}

    st_chim_b53_a57 = _build_bid_ask_split_chimera(st_l53_isl0, st_l57_isl0)
    l58_isl1_p = SAVE_DIR / "bak_l58" / "island_1_best.pt"
    l58_isl0_p = SAVE_DIR / "bak_l58" / "island_0_best.pt"
    l58_soup_p = SAVE_DIR / "bak_l58" / "order3_soup_top3.pt"
    st_l58_isl1 = policy.load_initial_champion(l58_isl1_p) if l58_isl1_p.exists() else st_chim_b53_a57
    st_l58_isl0 = policy.load_initial_champion(l58_isl0_p) if l58_isl0_p.exists() else st_chim_b53_a57
    st_l58_soup = policy.load_initial_champion(l58_soup_p) if l58_soup_p.exists() else st_l58_isl1
    st_chim_b53_a58i1 = _build_bid_ask_split_chimera(st_l53_isl0, st_l58_isl1)
    st_ms_53i0_52i1 = {k: v.to(device=device, dtype=policy.dtype) for k, v in _build_router_preserving_blend(st_l53_isl0, st_l52_isl1, body_w=0.28).items()}
    st_ms_53i0_56i0 = {k: v.to(device=device, dtype=policy.dtype) for k, v in _build_router_preserving_blend(st_l53_isl0, st_l56_isl0, body_w=0.25).items()}

    l61_soup_p = SAVE_DIR / "bak_l61" / "order3_soup_top3.pt"
    l61_isl0_p = SAVE_DIR / "bak_l61" / "island_0_best.pt"
    l61_isl1_p = SAVE_DIR / "bak_l61" / "island_1_best.pt"
    l60_isl0_p = SAVE_DIR / "bak_l60" / "island_0_best.pt"
    l47_live_p = SAVE_DIR / "bak_l47_lcm" / "global_best.pt"
    st_l60_isl0 = policy.load_initial_champion(l60_isl0_p) if l60_isl0_p.exists() else st_l58_isl1
    st_l61_soup = policy.load_initial_champion(l61_soup_p) if l61_soup_p.exists() else st_l60_isl0
    st_l61_isl0 = policy.load_initial_champion(l61_isl0_p) if l61_isl0_p.exists() else st_l61_soup
    st_l61_isl1 = policy.load_initial_champion(l61_isl1_p) if l61_isl1_p.exists() else st_l61_soup
    st_l47_live = policy.load_initial_champion(l47_live_p) if l47_live_p.exists() else st_l61_soup

    def _build_high_turnover_seed(st_base: dict, exp_b: float, ret_b: float, span_b: float, edge_a: float, span_a: float, damp: float = 0.30) -> dict:
        """
        Высокооборотная двухсторонняя инициализация L62 (x50..x100 к обороту с защитой безубытка):
        - Моды 0..6: активная загрузка капитала, близкая сетка Bid/Ask, запрет рыночного слива u_taker,
          строгая привязка Ask выше avg_entry*(1+2*maker_fee) (skew_blend -> 1.0, inside_a -> 0),
        - Мода #7: аварийный щит эвакуации водопадов,
        - Регулятор: нейрон подтяжки зависших ордеров (>90с и уход цены >1.2%) -> 0 залипания кэша на растущем рынке!
        """
        out_a = {k: st_base[k].detach().cpu().float().clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        w2 = out_a["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2 = out_a["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        for ch, val, d_f in [
            (0, exp_b, damp),
            (25, -5.5, 0.15),
            (38, ret_b, damp),
            (39, span_b, damp),
            (41, 4.2, 0.20),
            (42, edge_a, damp),
            (43, -4.2, 0.20),
            (45, span_a, damp),
        ]:
            w2[:, :7, ch] = w2[:, :7, ch] * d_f
            b2[:, :7, ch] = val
        b2[:, 7, 0] = -4.0

        w1_r = out_a["w_reg1"]
        bb_r = out_a["b_reg1_b"]
        ba_r = out_a["b_reg1_a"]
        w2_r = out_a["w_reg2"]
        imp = w1_r.abs().sum(dim=0) * w2_r.abs().sum(dim=1)
        u = int(torch.argmin(imp).item())
        w1_r[:, u] = 0.0
        w2_r[u, 0] = 0.0
        w1_r[1, u] = 4.0
        w1_r[5, u] = 6.0
        w1_r[7, u] = 4.0
        bb_r[0, u] = -(4.0 + 6.0 * 0.28 + 4.0 * 1.45)
        ba_r[0, u] = -14.0
        w2_r[u, 1] = 4.8
        out_a["b_reg2_b"][0, 0] = max(float(out_a["b_reg2_b"][0, 0]), 1.0)
        out_a["b_reg2_a"][0, 0] = max(float(out_a["b_reg2_a"][0, 0]), 1.0)

        out_a["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
        out_a["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
        st_n = _normalize_weights_order3(out_a, policy.dtype)
        return {k: v.to(device=device, dtype=policy.dtype) for k, v in st_n.items()}

    st_ht_bal = _build_high_turnover_seed(st_l61_soup, exp_b=1.4, ret_b=-2.6, span_b=-2.4, edge_a=0.65, span_a=-2.6, damp=0.35)
    st_ht_fast = _build_high_turnover_seed(st_l61_isl0, exp_b=1.9, ret_b=-3.3, span_b=-3.0, edge_a=0.80, span_a=-3.2, damp=0.25)
    st_ht_turbo = _build_high_turnover_seed(st_l61_isl1, exp_b=2.2, ret_b=-3.8, span_b=-3.5, edge_a=0.95, span_a=-3.6, damp=0.20)
    st_ht_l47 = _build_high_turnover_seed(_build_bid_ask_split_chimera(st_l47_live, st_l61_soup), exp_b=1.6, ret_b=-2.9, span_b=-2.8, edge_a=0.70, span_a=-3.0, damp=0.35)
    l62_isl1_p = SAVE_DIR / "bak_l62" / "island_1_best.pt"
    l62_isl0_p = SAVE_DIR / "bak_l62" / "island_0_best.pt"
    l62_soup_p = SAVE_DIR / "bak_l62" / "order3_soup_top3.pt"
    st_l62_isl1 = policy.load_initial_champion(l62_isl1_p) if l62_isl1_p.exists() else st_ht_fast
    st_l62_isl0 = policy.load_initial_champion(l62_isl0_p) if l62_isl0_p.exists() else st_ht_fast
    st_l62_soup = policy.load_initial_champion(l62_soup_p) if l62_soup_p.exists() else st_l62_isl1

    def _build_strict_edge_bridge(st_hi_turn: dict, st_prof: dict, alpha_turn: float, inside_b_bias: float = -3.0, min_edge_a: float = 0.78) -> dict:
        """
        Золотой мост L63 между высокооборотным ядром L62 (50x оборот) и профитным L61 (+1.13%):
        - Убирает агрессивный перехлест Bid под Ask1-1tick (inside_b <= 5% спреда -> 0 переплаты на входе!),
        - Жестко фиксирует Ask от точки безубытка с двойной комиссией (skew_blend -> 0.99, inside_a -> 0, edge_a >= +23 bp)!
        """
        gate_k = {"w_gate1", "b_gate1", "w_gate2", "b_gate2"}
        out_b = {}
        for k in MultimodalMoEPolicyOrder3.PARAM_KEYS:
            if k in gate_k:
                out_b[k] = st_hi_turn[k].detach().cpu().float().clone()
            else:
                out_b[k] = (1.0 - alpha_turn) * st_prof[k].detach().cpu().float() + alpha_turn * st_hi_turn[k].detach().cpu().float()
        w2 = out_b["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2 = out_b["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        w2[:, :7, 25] *= 0.15
        b2[:, :7, 25] = -5.5
        w2[:, :7, 37] *= 0.25
        b2[:, :7, 37] = inside_b_bias
        w2[:, :7, 41] *= 0.20
        b2[:, :7, 41] = 4.5
        w2[:, :7, 43] *= 0.20
        b2[:, :7, 43] = -4.5
        b2[:, :7, 42] = torch.clamp(b2[:, :7, 42], min=min_edge_a)
        out_b["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
        out_b["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
        st_n = _normalize_weights_order3(out_b, policy.dtype)
        return {k: v.to(device=device, dtype=policy.dtype) for k, v in st_n.items()}

    st_br_100 = _build_strict_edge_bridge(st_l62_isl1, st_l61_soup, alpha_turn=1.00, inside_b_bias=-2.8, min_edge_a=0.75)
    st_br_80 = _build_strict_edge_bridge(st_l62_isl0, st_l61_soup, alpha_turn=0.80, inside_b_bias=-3.2, min_edge_a=0.82)
    st_br_65 = _build_strict_edge_bridge(st_l62_soup, st_l61_soup, alpha_turn=0.65, inside_b_bias=-3.4, min_edge_a=0.88)
    st_br_50 = _build_strict_edge_bridge(st_l62_isl1, st_l61_soup, alpha_turn=0.50, inside_b_bias=-3.5, min_edge_a=0.92)
    l63_isl1_p = SAVE_DIR / "bak_l63" / "island_1_best.pt"
    l63_soup_p = SAVE_DIR / "bak_l63" / "order3_soup_top3.pt"
    st_l63_isl1 = policy.load_initial_champion(l63_isl1_p) if l63_isl1_p.exists() else st_br_100
    st_l63_soup = policy.load_initial_champion(l63_soup_p) if l63_soup_p.exists() else st_l63_isl1

    def _build_l64_coronation_seed(st_act: dict, st_shield: dict, ret_add: float, edge_min: float, shield_modes: tuple[int, ...] = (6, 7)) -> dict:
        """
        Коронационный мультимодальный синтез L64:
        - Рабочие моды (0..5): высокооборотное ядро L63 с расширенной чистой маржой Ask (+35..+75 bp сверх 2*fee!),
        - Защитные краш-моды (shield_modes): щит L61 (+$11.91 на V-цикле!) для мгновенного ухода от ножей -25.9%!
        """
        out_c = {k: st_act[k].detach().cpu().float().clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        w2 = out_c["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2 = out_c["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        w2_sh = st_shield["w2_moe"].detach().cpu().float().view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2_sh = st_shield["b2_moe"].detach().cpu().float().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        for m_i in range(K_MODES_ORDER3):
            if m_i in shield_modes:
                w2[:, m_i, :] = w2_sh[:, m_i, :]
                b2[:, m_i, :] = b2_sh[:, m_i, :]
            else:
                b2[:, m_i, 37] = -4.5
                b2[:, m_i, 38] = b2[:, m_i, 38] + ret_add
                b2[:, m_i, 41] = 5.0
                b2[:, m_i, 43] = -5.0
                b2[:, m_i, 42] = torch.clamp(b2[:, m_i, 42] + edge_min, min=edge_min)
        out_c["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
        out_c["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
        st_n = _normalize_weights_order3(out_c, policy.dtype)
        return {k: v.to(device=device, dtype=policy.dtype) for k, v in st_n.items()}

    l64_isl1_p = SAVE_DIR / "bak_l64" / "island_1_best.pt"
    l64_plus_p = SAVE_DIR / "bak_l64" / "l64_plus_margin.pt"
    l64_sw_p = SAVE_DIR / "bak_l64" / "l64_sweetspot_a.pt"
    st_l64_isl1 = policy.load_initial_champion(l64_isl1_p) if l64_isl1_p.exists() else st_wm_bal
    st_l64_plus = policy.load_initial_champion(l64_plus_p) if l64_plus_p.exists() else st_wm_pos
    st_l64_sw = policy.load_initial_champion(l64_sw_p) if l64_sw_p.exists() else st_wm_fast

    l65_isl0_p = SAVE_DIR / "bak_l65" / "island_0_best.pt"
    st_l65_isl0 = policy.load_initial_champion(l65_isl0_p) if l65_isl0_p.exists() else st_l64_plus

    def _build_l66_patient_wave_seed(
        base_st: dict,
        exp_bias: float,
        ret_b: float,
        span_b: float,
        wave49_z: float,
        edge42: float = 1.8,
        keep_mode7_crash: bool = True,
        requote_bias: float = -10.2,
    ) -> dict:
        """
        Терпеливый волновой компаундер L66 (+$34.69 на 45ч V-цикле, +$112.65 на 7 сутках еще ДО старта ES!):
        - Отключает преждевременный тайм-стоп u_taker (w2[:, :7, 25] = 0.0) в рабочих модах 0..6, позволяя удерживать волны +2.5%..+6.0%,
        - Сохраняет защитную краш-моду #7 для эвакуации водопадов -25.9%,
        - Регулятор переставляет отставшие Bid каждые ~60-90с вслед за растущим рынком, не трогая стоящие Ask!
        """
        st_p = {k: base_st[k].detach().cpu().float().clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
        w2 = st_p["w2_moe"].view(256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2 = st_p["b2_moe"].view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        b2[:, :, 48] = -7.0
        b2[:, :, 49] = -7.0

        m_end = 7 if keep_mode7_crash else 8
        w2[:, :m_end, 25] = 0.0
        b2[:, :m_end, 25] = -7.2
        if keep_mode7_crash:
            w2[:, 7, 25] *= 0.45
            b2[:, 7, 25] -= 1.2

        for m in range(m_end):
            w2[:, m, 0] *= 0.35
            b2[:, m, 0] = exp_bias if m < 5 else (exp_bias - 1.2)
            w2[:, m, 37] *= 0.25
            b2[:, m, 37] = -4.0
            w2[:, m, 38] *= 0.30
            b2[:, m, 38] = ret_b
            w2[:, m, 39] *= 0.30
            b2[:, m, 39] = span_b
            w2[:, m, 41] *= 0.20
            b2[:, m, 41] = 5.5
            w2[:, m, 42] *= 0.35
            b2[:, m, 42] = edge42
            w2[:, m, 43] *= 0.20
            b2[:, m, 43] = -5.5
            w2[:, m, 45] *= 0.35
            b2[:, m, 45] = -2.2
            b2[:, m, 49] = wave49_z if m != 2 else -4.0

        w1_r, bb_r, ba_r, w2_r = st_p["w_reg1"], st_p["b_reg1_b"], st_p["b_reg1_a"], st_p["w_reg2"]
        imp_r = w1_r.abs().sum(dim=0) * w2_r.abs().sum(dim=1)
        u = int(torch.argmin(imp_r).item())
        w1_r[:, u] = 0.0
        w2_r[u, 0] = 0.0
        w1_r[1, u] = 4.0
        w1_r[5, u] = 6.0
        w1_r[7, u] = 4.0
        bb_r[0, u] = requote_bias
        ba_r[0, u] = -16.0
        w2_r[u, 1] = 4.6
        st_p["b_reg2_b"][0, 0] = max(float(st_p["b_reg2_b"][0, 0]), 1.2)
        st_p["b_reg2_a"][0, 0] = max(float(st_p["b_reg2_a"][0, 0]), 1.2)

        st_p["w2_moe"] = w2.reshape(256, MOE_OUT_DIM_ORDER3)
        st_p["b2_moe"] = b2.reshape(1, MOE_OUT_DIM_ORDER3)
        st_n = _normalize_weights_order3(st_p, policy.dtype)
        return {k: v.to(device=device, dtype=policy.dtype) for k, v in st_n.items()}

    st_pw_notkr = _build_l66_patient_wave_seed(st_l64_plus, exp_bias=1.6, ret_b=-2.6, span_b=-2.5, wave49_z=0.1, edge42=1.8, keep_mode7_crash=False)
    st_pw_l53 = _build_l66_patient_wave_seed(st_l53_isl0, exp_bias=1.8, ret_b=-3.0, span_b=-2.8, wave49_z=0.7, edge42=1.8, keep_mode7_crash=True)
    st_pw_l65 = _build_l66_patient_wave_seed(st_l65_isl0, exp_bias=1.6, ret_b=-2.8, span_b=-2.6, wave49_z=0.1, edge42=1.8, keep_mode7_crash=True)
    st_pw_l61_25 = _build_l66_patient_wave_seed(st_l61_soup, exp_bias=1.8, ret_b=-3.0, span_b=-2.8, wave49_z=0.1, edge42=1.8, keep_mode7_crash=True)
    st_pw_l61_38 = _build_l66_patient_wave_seed(st_l61_soup, exp_bias=2.0, ret_b=-3.2, span_b=-2.8, wave49_z=0.7, edge42=2.0, keep_mode7_crash=True)
    def _load_gpu_seed(p_seed: Path, fallback_st: dict) -> dict:
        if p_seed.exists():
            st_cpu = policy.load_initial_champion(p_seed)
            return {k: v.to(device=device, dtype=policy.dtype) for k, v in st_cpu.items()}
        return fallback_st

    st_l66_soup = _load_gpu_seed(SAVE_DIR / "bak_l66" / "order3_soup_top3.pt", st_pw_l53)
    st_hyb_60pw_40l66 = _load_gpu_seed(SAVE_DIR / "bak_l66" / "hyb_60pw25_40l66.pt", st_pw_notkr)
    st_hyb_50_50 = _load_gpu_seed(SAVE_DIR / "bak_l66" / "hyb_50pw38_50pw25.pt", st_pw_notkr)
    st_hyb_60l66_40pw = _load_gpu_seed(SAVE_DIR / "bak_l66" / "hyb_60l66_40pw25.pt", st_l66_soup)
    st_ms_53i0_52i1 = st_hyb_60pw_40l66
    st_ms_53i0_56i0 = st_hyb_50_50
    is_warm_l43 = True

    # L67 Grand Kelly Realized Compounder: объединение +$312 (+24.22% за 7 сут = $44.57/сут!) и щита L66_Soup (DD 3.57%-8.11%)!
    seed_pool = (
        [st_hyb_60pw_40l66, st_hyb_50_50, st_hyb_60l66_40pw, st_l66_soup, st_pw_l53, st_pw_notkr]
        if island_id == 0
        else [st_hyb_50_50, st_hyb_60pw_40l66, st_pw_notkr, st_hyb_60l66_40pw, st_l66_soup, st_pw_l61_38]
    )
    resume_iter = 0
    for res_p in [SAVE_DIR / "islands" / f"island_{island_id}_best.pt", SAVE_DIR / "global_best.pt", RESUME_CHAMP_PATH]:
        if res_p.exists():
            try:
                r_ck = torch.load(res_p, map_location="cpu", weights_only=False)
                if r_ck.get("wf_version") == wf_version:
                    st_res = policy.load_initial_champion(res_p)
                    seed_pool.insert(0, st_res)
                    resume_iter = max(resume_iter, int(r_ck.get("iteration", 0)))
                    log_msg(f"♻️ Найден чекпоинт продолжения {wf_version} ({res_p.name}, it={r_ck.get('iteration', 0)}, ValGeo={float(r_ck.get('geo_score_bp', 0.0)):+.2f} bp)!")
            except Exception as e_res:
                log_msg(f"⚠️ Пропуск чтения чекпоинта продолжения {res_p.name}: {e_res}")

    seed_val = eval_dual_scale(seed_pool, val_macro_basket, val_micro_basket, enforce_mi=True)
    best_seed_i = int(np.argmax([r["geo_score_bp"] for r in seed_val]))
    champion_state = seed_pool[best_seed_i]
    val_basket = val_macro_basket + val_micro_basket
    oos_basket = oos_macro_basket + oos_micro_basket
    log_msg(
        f"🌱 Выбрано стартовое семя #{best_seed_i} (TrainJointGeo={seed_val[best_seed_i]['geo_score_bp']:+.2f} bp | "
        f"Macro={seed_val[best_seed_i]['macro_geo_bp']:+.2f} | Micro={seed_val[best_seed_i]['micro_geo_bp']:+.2f} | "
        f"Turn={seed_val[best_seed_i].get('turn_mult', 0.0):.2f}x)"
    )

    # 4. Базовая оценка на Validation и расширенном Blind OOS
    log_msg(f"⚡ Оценка базового K=8 чемпиона на {len(val_basket)} окнах Validation и {len(oos_basket)} окнах Blind OOS...")
    t0_eval = time.time()
    init_res = eval_dual_scale([champion_state], val_macro_basket, val_micro_basket, enforce_mi=True)[0]
    init_oos = eval_dual_scale([champion_state], oos_macro_basket, oos_micro_basket, enforce_mi=False)[0]
    dt_eval0 = time.time() - t0_eval

    best_geo_bp = float(init_res["geo_score_bp"])
    best_clean_ev = float(init_res["clean_ev_pct"])
    baseline_oos_geo = float(init_oos["geo_score_bp"])
    best_state = {k: v.clone() for k, v in champion_state.items()}
    best_metrics = dict(init_res)

    log_msg(
        f"🏆 Базовый K=8 результат: ValJointGeo={best_geo_bp:+.2f} bp (Macro={init_res['macro_geo_bp']:+.2f}, Micro={init_res['micro_geo_bp']:+.2f}, EV={best_clean_ev:+.4f}%, MI={init_res['mode_mi_bits']:.2f}b) | "
        f"OOS_JointGeo={baseline_oos_geo:+.2f} bp (OOS_EV={init_oos['clean_ev_pct']:+.4f}%) | "
        f"Fills={init_res['fills']:.1f} | EvalTime={dt_eval0:.2f}s"
    )

    elite_archive = [(best_geo_bp, {k: v.clone() for k, v in best_state.items()})]
    aux_elites = [st_ms_53i0_52i1, st_ms_53i0_56i0] if island_id == 0 else [st_l53_isl0, st_l56_isl0]
    aux_res = eval_dual_scale(aux_elites, val_macro_basket, val_micro_basket, enforce_mi=True)
    for ax_st, ax_r in zip(aux_elites, aux_res):
        elite_archive.append((float(ax_r["geo_score_bp"]), {k: v.clone() for k, v in ax_st.items()}))

    # 5. Главный цикл оптимизации с Low-Rank OpenAI-ES + Auto-Phase Step-Down (P1 -> P2 -> P3) и полным чекпоинтом состояния
    patience = 0
    phase_idx = 0
    prior_elapsed_sec = 0.0
    sync_path = SAVE_DIR / "global_best.pt"
    prog_dir = SAVE_DIR / "islands"
    prog_dir.mkdir(parents=True, exist_ok=True)
    prog_path = prog_dir / f"island_{island_id}_progress.pt"
    if prog_path.exists():
        try:
            pr_st = torch.load(prog_path, map_location="cpu", weights_only=False)
            if pr_st.get("wf_version") == wf_version:
                resume_iter = max(resume_iter, int(pr_st.get("iteration", resume_iter)))
                phase_idx = int(pr_st.get("phase_idx", 0))
                patience = int(pr_st.get("patience", 0))
                prior_elapsed_sec = float(pr_st.get("elapsed_sec", 0.0))
                policy.adam_t = int(pr_st.get("adam_t", 0))
                for k_ad in MultimodalMoEPolicyOrder3.PARAM_KEYS:
                    if k_ad in pr_st.get("adam_m", {}):
                        policy.adam_m[k_ad] = pr_st["adam_m"][k_ad].to(device=device, dtype=torch.float32)
                        policy.adam_v[k_ad] = pr_st["adam_v"][k_ad].to(device=device, dtype=torch.float32)
                for ea_geo, ea_w in pr_st.get("elite_archive", []):
                    elite_archive.append((float(ea_geo), {k: v.to(device=device, dtype=policy.dtype) for k, v in ea_w.items()}))
                log_msg(f"🔄 Восстановлено состояние оптимизатора Island #{island_id}: next_it={resume_iter+1}, phase=P{phase_idx+1}, elapsed={prior_elapsed_sec/60:.1f}m")
        except Exception as e_pr:
            log_msg(f"⚠️ Не удалось прочитать progress.pt: {e_pr}")

    last_improve_ts = time.time()
    phase_enter_ts = time.time()
    w_cnt_f = max(1.0, float(len(val_basket)))

    start_it = max(25, resume_iter) + 1 if resume_iter >= 19 else (resume_iter + 1 if resume_iter > 0 else 1)
    for it in range(start_it, max_iters + 1):
        t_it = time.time()

        # 🧬 Асинхронная кросс-островная миграция элиты
        if it % 10 == 0:
            immigrant_cand = None
            immigrant_geo = -999.0
            immigrant_source = ""
            islands_dir = SAVE_DIR / "islands"
            islands_dir.mkdir(parents=True, exist_ok=True)

            neighbor_id = (island_id - 1) % max(1, num_islands)
            neighbor_p = islands_dir / f"island_{neighbor_id}_best.pt"
            if neighbor_id != island_id and neighbor_p.exists():
                try:
                    n_ck = torch.load(neighbor_p, map_location="cpu", weights_only=False)
                    if "w1" in n_ck and n_ck["w1"].shape[0] == TOTAL_DIM_ORDER3 and n_ck.get("wf_version") == wf_version:
                        n_geo = float(n_ck.get("geo_score_bp", -999.0))
                        if n_geo > best_geo_bp + 0.08:
                            immigrant_cand = n_ck
                            immigrant_geo = n_geo
                            immigrant_source = f"Сосед #{neighbor_id}"
                except Exception:
                    pass

            if sync_path.exists():
                try:
                    g_ck = torch.load(sync_path, map_location="cpu", weights_only=False)
                    if "w1" in g_ck and g_ck["w1"].shape[0] == TOTAL_DIM_ORDER3 and g_ck.get("wf_version") == wf_version:
                        g_geo = float(g_ck.get("geo_score_bp", -999.0))
                        if g_geo > max(best_geo_bp + 0.08, immigrant_geo):
                            immigrant_cand = g_ck
                            immigrant_geo = g_geo
                            immigrant_source = "Абсолютный лидер"
                except Exception:
                    pass

            if immigrant_cand is not None:
                log_msg(f"🧬 [Island #{island_id}] Миграция элиты ({immigrant_source}): ValGeo={immigrant_geo:+.2f} bp vs локального {best_geo_bp:+.2f} bp!")
                prev_best_state = {k: v.clone() for k, v in best_state.items()}
                best_geo_bp = immigrant_geo
                best_clean_ev = float(immigrant_cand.get("clean_ev_pct", 0.0))
                for k in MultimodalMoEPolicyOrder3.PARAM_KEYS:
                    if k in immigrant_cand:
                        best_state[k] = immigrant_cand[k].to(device=device, dtype=policy.dtype).clone()
                policy.update_momentum(prev_best_state, best_state)
                patience = 0
                last_improve_ts = time.time()

        elapsed_hours = (prior_elapsed_sec + (time.time() - t_start)) / 3600.0
        stagnant_min = (time.time() - last_improve_ts) / 60.0
        phase_stag_min = (time.time() - max(last_improve_ts, phase_enter_ts)) / 60.0
        progress = min(1.0, elapsed_hours / max(0.5, time_budget_hours))

        if elapsed_hours >= time_budget_hours:
            log_msg(f"⏱️ Достигнут лимит времени {time_budget_hours:.2f}ч. Завершение.")
            break

        step_down_p1 = max(18.0, early_stop_minutes * 0.65) if early_stop_minutes > 0 else 65.0
        step_down_p2 = max(28.0, early_stop_minutes * 0.85) if early_stop_minutes > 0 else 95.0
        if phase_idx == 0 and (phase_stag_min >= step_down_p1 or progress >= 0.36):
            phase_idx = 1
            phase_enter_ts = time.time()
            patience = 0
            log_msg(f"🔬 [Island #{island_id} | It {it:03d}] AUTO-PHASE -> P2:Pruning (фокусировка вокруг лидера ValGeo={best_geo_bp:+.2f} bp)!")
        elif phase_idx == 1 and (phase_stag_min >= step_down_p2 or progress >= 0.78):
            phase_idx = 2
            phase_enter_ts = time.time()
            patience = 0
            log_msg(f"🎯 [Island #{island_id} | It {it:03d}] AUTO-PHASE -> P3:KellyZoom (тонкая полировка ValGeo={best_geo_bp:+.2f} bp)!")
        elif phase_idx == 2 and early_stop_minutes > 0 and phase_stag_min >= max(20.0, early_stop_minutes * 0.75) and it >= 160:
            log_msg(f"🛑 [EARLY STOPPING] Финальный зум P3 завершен (стагнация {phase_stag_min:.1f} мин). Экономия GPU!")
            break

        if phase_idx == 0:
            stage = "explore" if island_mode == "explore" else "core"
            base_sigma = (0.78 if island_mode == "explore" else 0.55) if is_warm_l43 else (1.20 if island_mode == "explore" else 0.85)
            phase_tag = "P1:WarmExplore" if is_warm_l43 else "P1:Discovery"
        elif phase_idx == 1:
            stage = "explore" if island_mode == "explore" else "balanced"
            base_sigma = (0.52 if island_mode == "explore" else 0.38) if is_warm_l43 else (0.62 if island_mode == "explore" else 0.44)
            phase_tag = "P2:Pruning"
        else:
            stage = "zoom1" if island_mode == "explore" else "zoom2"
            base_sigma = 0.25 if island_mode == "explore" else 0.16
            phase_tag = "P3:KellyZoom"

        if patience >= 24 and len(elite_archive) > 1 and phase_idx < 2:
            hop_idx = int(rng_train.integers(0, len(elite_archive)))
            center = elite_archive[hop_idx][1]
            patience = 0
            log_msg(f"🦘 [Island #{island_id} | It {it:03d} | {phase_tag}] Basin Hopping к элите #{hop_idx} (ValGeo={elite_archive[hop_idx][0]:+.2f} bp)")
        else:
            center = best_state

        sigma_scale = base_sigma * (1.0 + 0.10 * min(3, patience // 6))
        policy.sample_population(center, stage=stage, sigma_scale=sigma_scale, alt_elite_state=best_state)
        cand_list = policy.get_candidate_states()

        t_batch = time.time()
        tr_results = eval_dual_scale(
            cand_list, sample_train_basket(False), sample_train_basket(True), enforce_mi=True, calib_jitter_std=0.045
        )

        tr_geos = np.array([r["geo_score_bp"] for r in tr_results], dtype=np.float64)
        es_cands = policy.build_openai_es_candidates(center, tr_geos, lr_scales=(0.35, 0.75, 1.25, 1.85))
        k_mut_top = max(4, min(32 - len(es_cands), len(cand_list)))
        top_mut_idx = np.argsort(tr_geos)[-k_mut_top:][::-1].tolist()
        val_cands = es_cands + [cand_list[i] for i in top_mut_idx]
        val_results = eval_dual_scale(val_cands, val_macro_basket, val_micro_basket, enforce_mi=True)
        dt_batch = time.time() - t_batch

        val_geos = [r["geo_score_bp"] for r in val_results]
        best_v_local = int(np.argmax(val_geos))
        is_es_win = best_v_local < len(es_cands)
        gen_best_geo = float(val_geos[best_v_local])
        gen_best_res = val_results[best_v_local]

        is_new_champ = gen_best_geo > best_geo_bp + 0.06
        if is_new_champ:
            delta_bp = gen_best_geo - best_geo_bp
            prev_local_state = {k: v.clone() for k, v in best_state.items()}
            best_geo_bp = gen_best_geo
            best_clean_ev = gen_best_res["clean_ev_pct"]
            best_state = {k: val_cands[best_v_local][k].clone() for k in MultimodalMoEPolicyOrder3.PARAM_KEYS}
            best_metrics = dict(gen_best_res)
            policy.update_momentum(prev_local_state, best_state)
            patience = 0
            last_improve_ts = time.time()

            elite_archive.append((best_geo_bp, {k: v.clone() for k, v in best_state.items()}))
            if len(elite_archive) > 8:
                elite_archive.sort(key=lambda x: x[0], reverse=True)
                elite_archive = elite_archive[:8]

            ck_data = {
                "arch_version": "order3_v1",
                "wf_version": wf_version,
                "pure_endogenous": True,
                "k_modes": K_MODES_ORDER3,
                "iteration": it,
                "geo_score_bp": best_geo_bp,
                "macro_geo_bp": best_metrics.get("macro_geo_bp", 0.0),
                "micro_geo_bp": best_metrics.get("micro_geo_bp", 0.0),
                "clean_ev_pct": best_clean_ev,
                "full_ev_pct": best_metrics["full_ev_pct"],
                "cvar25_bp": best_metrics["cvar25_bp"],
                "turn_mult": best_metrics.get("turn_mult", 0.0),
                "fills": best_metrics["fills"],
                "cancels": best_metrics["cancels"],
                "cnl_per_fill": best_metrics["cnl_per_fill"],
                "eta_queue_pct": best_metrics["eta_queue_pct"],
                "mode_probs": best_metrics["mode_probs"],
                "mode_mi_bits": best_metrics["mode_mi_bits"],
                "in_dim": TOTAL_DIM_ORDER3,
                "mkt_dim": MKT_DIM_ORDER3,
                "ag_dim": AG_DIM_ORDER3,
                **best_state,
            }

            islands_dir = SAVE_DIR / "islands"
            islands_dir.mkdir(parents=True, exist_ok=True)
            isl_best_p = islands_dir / f"island_{island_id}_best.pt"
            isl_tmp_p = islands_dir / f"island_{island_id}_best.pt.tmp.{os.getpid()}"
            torch.save(ck_data, isl_tmp_p)
            os.replace(isl_tmp_p, isl_best_p)

            is_new_global = True
            if sync_path.exists():
                try:
                    cur_sync = torch.load(sync_path, map_location="cpu", weights_only=False)
                    if cur_sync.get("wf_version") == wf_version and float(cur_sync.get("geo_score_bp", -999.0)) >= best_geo_bp:
                        is_new_global = False
                except Exception:
                    pass

            if is_new_global:
                torch.save(ck_data, LATEST_CHAMP_PATH)
                torch.save(ck_data, RESUME_CHAMP_PATH)
                try:
                    tmp_sync = SAVE_DIR / f"global_best.pt.tmp.{os.getpid()}"
                    torch.save(ck_data, tmp_sync)
                    os.replace(tmp_sync, sync_path)
                except Exception:
                    pass
                log_msg(f"👑 [GLOBAL K=8 RECORD!] Island #{island_id}: ValJointGeo={best_geo_bp:+.2f} bp (Macro={best_metrics['macro_geo_bp']:+.2f} | Micro={best_metrics['micro_geo_bp']:+.2f} | MI={best_metrics['mode_mi_bits']:.2f}b)!")

            src_tag = f"ES-Grad#{best_v_local}" if is_es_win else "Mutant"
            log_msg(
                f"⭐ [It {it:03d} | +{delta_bp:.2f} bp | {src_tag}] NEW K=8 CHAMPION! "
                f"JointGeo={best_geo_bp:+.2f} bp (Mac={best_metrics['macro_geo_bp']:+.2f}, Mic={best_metrics['micro_geo_bp']:+.2f}) | "
                f"CleanEV={best_clean_ev:+.4f}% | InvEV={best_metrics.get('inv_delta_ev_pct', 0.0):+.4f}% | Turn={best_metrics.get('turn_mult', 0.0):.2f}x | "
                f"Fills/w={best_metrics['fills']/w_cnt_f:.1f} | MI={best_metrics['mode_mi_bits']:.2f}b | Time={time.time() - t_it:.1f}s"
            )
        else:
            patience += 1
            if it % 5 == 0:
                log_msg(
                    f"   [It {it:03d} | {phase_tag}] TrainTop={tr_geos[top_mut_idx[0]]:+.2f} bp | "
                    f"ValTop={gen_best_geo:+.2f} bp | ValChamp={best_geo_bp:+.2f} bp | "
                    f"PhaseStag={phase_stag_min:.1f}m | Batch={dt_batch:.1f}s"
                )

        if is_new_champ or it % 5 == 0:
            try:
                tmp_pr = prog_dir / f"island_{island_id}_progress.pt.tmp.{os.getpid()}"
                ea_serial = [(float(g_e), {k: v.detach().cpu() for k, v in w_e.items()}) for g_e, w_e in elite_archive[:4]]
                torch.save(
                    {
                        "wf_version": wf_version,
                        "iteration": it,
                        "phase_idx": phase_idx,
                        "patience": patience,
                        "elapsed_sec": prior_elapsed_sec + (time.time() - t_start),
                        "adam_t": policy.adam_t,
                        "adam_m": {k: v.detach().cpu() for k, v in policy.adam_m.items()},
                        "adam_v": {k: v.detach().cpu() for k, v in policy.adam_v.items()},
                        "elite_archive": ea_serial,
                    },
                    tmp_pr,
                )
                os.replace(tmp_pr, prog_path)
            except Exception:
                pass

    # 6. Меж-островной Router-Preserving Soup + Слепой OOS-Экзамен!
    for nb_i in range(num_islands):
        nb_p = SAVE_DIR / "islands" / f"island_{nb_i}_best.pt"
        if nb_i != island_id and nb_p.exists():
            try:
                nb_ck = torch.load(nb_p, map_location="cpu", weights_only=False)
                if nb_ck.get("wf_version") == wf_version:
                    nb_w = {k: nb_ck[k].to(device=device, dtype=policy.dtype) for k in MultimodalMoEPolicyOrder3.PARAM_KEYS if k in nb_ck}
                    if len(nb_w) == len(MultimodalMoEPolicyOrder3.PARAM_KEYS):
                        elite_archive.append((float(nb_ck.get("geo_score_bp", -999.0)), nb_w))
            except Exception:
                pass

    elite_archive.sort(key=lambda x: x[0], reverse=True)
    # Оставляем уникальные состояния для честного смешивания:
    uniq_elites = []
    for g_e, w_e in elite_archive:
        if not uniq_elites or float(torch.max(torch.abs(w_e["w1"].detach().cpu().float() - uniq_elites[-1][1]["w1"].detach().cpu().float())).item()) > 1e-5:
            uniq_elites.append((g_e, w_e))

    final_cand = best_state
    if len(uniq_elites) >= 2:
        log_msg("🍲 Сборка Router-Preserving K=8 Soup (0.75 Top1 + 0.25 Top2 без размытия шлюзов w_gate)...")
        gate_keys = {"w_gate1", "b_gate1", "w_gate2", "b_gate2"}
        soup_state = {}
        for k in MultimodalMoEPolicyOrder3.PARAM_KEYS:
            if k in gate_keys:
                soup_state[k] = uniq_elites[0][1][k].detach().to(device=device, dtype=policy.dtype).clone()
            else:
                t0_w = uniq_elites[0][1][k].detach().to(device=device, dtype=torch.float32)
                t1_w = uniq_elites[1][1][k].detach().to(device=device, dtype=torch.float32)
                soup_state[k] = (0.75 * t0_w + 0.25 * t1_w).to(dtype=policy.dtype)
        soup_norm = _normalize_weights_order3(soup_state, policy.dtype)
        soup_val = eval_dual_scale([soup_norm], val_macro_basket, val_micro_basket, enforce_mi=True)[0]
        log_msg(f"🍲 Router-Preserving Soup JointGeo={soup_val['geo_score_bp']:+.2f} bp (vs Top1 {best_geo_bp:+.2f} bp)")
        save_soup = True
        if SOUP_CHAMP_PATH.exists():
            try:
                old_sp = torch.load(SOUP_CHAMP_PATH, map_location="cpu", weights_only=False)
                if old_sp.get("wf_version") == wf_version and float(old_sp.get("geo_score_bp", -999.0)) >= float(soup_val["geo_score_bp"]):
                    save_soup = False
            except Exception:
                pass
        if save_soup:
            torch.save({"arch_version": "order3_v1", "wf_version": wf_version, "pure_endogenous": True, "k_modes": K_MODES_ORDER3, "is_soup": True, "geo_score_bp": soup_val["geo_score_bp"], "clean_ev_pct": soup_val["clean_ev_pct"], "in_dim": TOTAL_DIM_ORDER3, "mkt_dim": MKT_DIM_ORDER3, "ag_dim": AG_DIM_ORDER3, **soup_norm}, SOUP_CHAMP_PATH)
        if float(soup_val["geo_score_bp"]) > best_geo_bp:
            final_cand = soup_norm

    oos_final = eval_dual_scale([final_cand], oos_macro_basket, oos_micro_basket, enforce_mi=False)[0]
    log_msg(
        f"🧪 [BLIND {len(oos_basket)}-WINDOW OOS EXAM ({oos_span/3600:.1f}h)] Baseline OOS={baseline_oos_geo:+.2f} bp -> Final OOS={oos_final['geo_score_bp']:+.2f} bp "
        f"(Macro={oos_final['macro_geo_bp']:+.2f}, Micro={oos_final['micro_geo_bp']:+.2f}) | "
        f"OOS_EV={oos_final['clean_ev_pct']:+.4f}% (Mac={oos_final['macro_ev_pct']:+.4f}%, Mic={oos_final['micro_ev_pct']:+.4f}%) | "
        f"OOS_Turn={oos_final.get('turn_mult', 0.0):.2f}x | OOS_CVaR25={oos_final['cvar25_bp']:+.2f} bp | "
        f"OOS_Fills/w={oos_final['fills']/max(1.0, float(len(oos_basket))):.1f}"
    )
    log_msg(f"🏁 Обучение Order-3 K=8 [{wf_version}] завершено за {(time.time() - t_start)/3600:.2f}ч! Чемпион: {LATEST_CHAMP_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Order-3 RL Champion Walk-Forward v4 (60D Signal Stack v2) Trainer")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g. cuda:0 or cuda:1)")
    parser.add_argument("--groups", type=int, default=160, help="Number of candidate groups")
    parser.add_argument("--iters", type=int, default=800, help="Maximum iterations")
    parser.add_argument("--hours", type=float, default=2.0, help="Time budget in hours")
    parser.add_argument("--win-len", type=int, default=9001, help="Macro window length in ticks")
    parser.add_argument("--val-windows", type=int, default=24, help="Number of windows per basket")
    parser.add_argument("--island-id", type=int, default=0, help="Island ID (0..num_islands-1)")
    parser.add_argument("--num-islands", type=int, default=2, help="Total active islands in ring")
    parser.add_argument("--early-stop-min", type=float, default=32.0, help="Early stopping stagnation minutes (0 to disable)")
    parser.add_argument("--island-mode", type=str, default="exploit", choices=["exploit", "balanced", "explore", "wide"], help="Island Mode")
    parser.add_argument("--log-file", type=str, default=None, help="Custom log file")
    parser.add_argument("--wf-version", type=str, default="pure_wf_v35_l65_wave_rider", help="Lesson / WF version tag")
    parser.add_argument("--oos-hours", type=float, default=22.0, help="Blind OOS tail hours reserved at end of dataset")
    parser.add_argument("--inv-stress", type=float, default=0.35, help="Fraction of windows with pre-loaded inventory stress")
    args = parser.parse_args()

    train_order3(
        max_iters=args.iters,
        n_groups=args.groups,
        win_len=args.win_len,
        n_val_windows=args.val_windows,
        device_str=args.device,
        time_budget_hours=args.hours,
        island_id=args.island_id,
        num_islands=args.num_islands,
        island_mode=args.island_mode,
        early_stop_minutes=args.early_stop_min,
        log_file=args.log_file,
        wf_version=args.wf_version,
        oos_hours=args.oos_hours,
        inv_stress_frac=args.inv_stress,
    )


if __name__ == "__main__":
    main()
