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
    K_MODES_ORDER3,
    EXPERT_OUT_DIM_ORDER3,
    MOE_OUT_DIM_ORDER3,
    ROUTER_IN_DIM_ORDER3,
    ROUTER_H_DIM_ORDER3,
    REG_IN_DIM_ORDER3,
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
    out["b2_moe"] = torch.clamp(state["b2_moe"].float(), -5.0, 5.0).to(dtype)

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

        self.w_rec = torch.empty((n_groups, 256, 32), dtype=self.dtype, device=device)
        self.b_rec = torch.empty((n_groups, 1, 32), dtype=self.dtype, device=device)

        self.w_reg1 = torch.empty((n_groups, REG_IN_DIM_ORDER3, 48), dtype=self.dtype, device=device)
        self.b_reg1_b = torch.empty((n_groups, 1, 48), dtype=self.dtype, device=device)
        self.b_reg1_a = torch.empty((n_groups, 1, 48), dtype=self.dtype, device=device)
        self.w_reg2 = torch.empty((n_groups, 48, 2), dtype=self.dtype, device=device)
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
        is_order3 = ck.get("arch_version") == "order3_v1"

        if (
            is_order3
            and ck["w1"].shape[0] == TOTAL_DIM_ORDER3
            and ck["w_gate1"].shape[0] == ROUTER_IN_DIM_ORDER3
            and ck["w_reg1"].shape[0] == REG_IN_DIM_ORDER3
            and ck["b_gate2"].numel() == K_MODES_ORDER3
        ):
            state = {k: ck[k].to(dtype=self.dtype, device=self.device) for k in self.PARAM_KEYS}
        else:
            # Хирургический перенос Signal Stack v2 + расширение K=4 -> K=8 мод без потери выученных экспертов 0..3:
            gen = torch.Generator().manual_seed(20261146)
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
                wg1_new = torch.randn((ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), generator=gen) * 0.005
                cr = min(ROUTER_IN_DIM_ORDER3, wg1_old.shape[0])
                wg1_new[:cr, :] = wg1_old[:cr, :]
            else:
                wg1_new = wg1_old

            old_k = int(ck["b_gate2"].numel())
            w2m_old = ck["w2_moe"].float().view(256, old_k, EXPERT_OUT_DIM_ORDER3)
            b2m_old = ck["b2_moe"].float().view(1, old_k, EXPERT_OUT_DIM_ORDER3)
            wg2_old = ck["w_gate2"].float()
            bg2_old = ck["b_gate2"].float().view(1, old_k)

            if old_k != K_MODES_ORDER3:
                w2m_new = torch.zeros((256, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32)
                b2m_new = torch.zeros((1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32)
                wg2_new = torch.randn((ROUTER_H_DIM_ORDER3, K_MODES_ORDER3), generator=gen) * 0.03
                bg2_new = torch.zeros((1, K_MODES_ORDER3), dtype=torch.float32)

                ck_copy = min(old_k, K_MODES_ORDER3)
                w2m_new[:, :ck_copy, :] = w2m_old[:, :ck_copy, :]
                b2m_new[:, :ck_copy, :] = b2m_old[:, :ck_copy, :]
                wg2_new[:, :ck_copy] = wg2_old[:, :ck_copy]
                bg2_new[:, :ck_copy] = bg2_old[:, :ck_copy]

                # Инициализация новых мод 4..7 под разные масштабы ВЧ-сбора спреда, карусели и разгрузки:
                ref_m = min(1, old_k - 1)
                for km in range(ck_copy, K_MODES_ORDER3):
                    w2m_new[:, km, :] = w2m_old[:, ref_m, :] + torch.randn((256, EXPERT_OUT_DIM_ORDER3), generator=gen) * 0.025
                    b2m_new[:, km, :] = b2m_old[:, ref_m, :]
                    wg2_new[:, km] = wg2_old[:, km % old_k] + torch.randn((ROUTER_H_DIM_ORDER3,), generator=gen) * 0.06
                bg2_new[0, ck_copy:] = float(bg2_old.mean().item()) - 0.65

                if K_MODES_ORDER3 >= 8 and ck_copy <= 4:
                    # Mode #4: ВЧ-сбор широкого спреда внутри коридора [b1..a1-tick]
                    b2m_new[0, 4, 0] = -1.20; b2m_new[0, 4, 37] = 1.20; b2m_new[0, 4, 38] = -4.00; b2m_new[0, 4, 39] = -3.80; b2m_new[0, 4, 41] = 1.80; b2m_new[0, 4, 42] = 0.55; b2m_new[0, 4, 43] = 0.90
                    # Mode #5: Плотный двусторонний котировщик на лучших уровнях b1 / a1
                    b2m_new[0, 5, 0] = -0.90; b2m_new[0, 5, 37] = -0.50; b2m_new[0, 5, 38] = -4.20; b2m_new[0, 5, 39] = -3.20; b2m_new[0, 5, 41] = 2.20; b2m_new[0, 5, 42] = 0.85; b2m_new[0, 5, 43] = -0.50
                    # Mode #6: Эшелонированная карусель локальных откатов (3-5 слотов)
                    b2m_new[0, 6, 0] = -0.60; b2m_new[0, 6, 37] = -2.00; b2m_new[0, 6, 38] = -3.00; b2m_new[0, 6, 39] = -2.20; b2m_new[0, 6, 41] = 2.00; b2m_new[0, 6, 42] = 0.70; b2m_new[0, 6, 43] = 0.20; b2m_new[0, 6, 44] = -1.20
                    # Mode #7: Быстрая безболезненная разгрузка инвентаря внутри спреда
                    b2m_new[0, 7, 0] = -4.50; b2m_new[0, 7, 41] = -1.50; b2m_new[0, 7, 42] = -0.25; b2m_new[0, 7, 43] = 1.80

                w2m_flat = w2m_new.reshape(256, MOE_OUT_DIM_ORDER3)
                b2m_flat = b2m_new.reshape(1, MOE_OUT_DIM_ORDER3)
            else:
                w2m_flat = w2m_old.reshape(256, MOE_OUT_DIM_ORDER3)
                b2m_flat = b2m_old.reshape(1, MOE_OUT_DIM_ORDER3)
                wg2_new = wg2_old
                bg2_new = bg2_old

            wreg1_old = ck["w_reg1"].float()
            if wreg1_old.shape[0] != REG_IN_DIM_ORDER3:
                wreg1_new = torch.zeros((REG_IN_DIM_ORDER3, 48), dtype=torch.float32)
                if wreg1_old.shape[0] == 26 and REG_IN_DIM_ORDER3 >= 28:
                    wreg1_new[:22, :] = wreg1_old[:22, :]
                    wreg1_new[24:28, :] = wreg1_old[22:26, :]
                else:
                    cr_reg = min(REG_IN_DIM_ORDER3, wreg1_old.shape[0])
                    wreg1_new[:cr_reg, :] = wreg1_old[:cr_reg, :]
                if REG_IN_DIM_ORDER3 > wreg1_old.shape[0]:
                    wreg1_new[wreg1_old.shape[0] :, :] = torch.randn((REG_IN_DIM_ORDER3 - wreg1_old.shape[0], 48), generator=gen) * 0.015
            else:
                wreg1_new = wreg1_old

            brec_new = ck["b_rec"].view(1, 32).float().clone()
            if wreg1_old.shape[0] < 28:
                brec_new[0, 16:24] = brec_new[0, 16:24] + 0.35
                brec_new[0, 24:32] = brec_new[0, 24:32] - 1.20

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
                "w_rec": ck["w_rec"].to(dtype=self.dtype, device=self.device),
                "b_rec": brec_new.to(dtype=self.dtype, device=self.device),
                "w_reg1": wreg1_new.to(dtype=self.dtype, device=self.device),
                "b_reg1_b": ck["b_reg1_b"].view(1, 48).to(dtype=self.dtype, device=self.device),
                "b_reg1_a": ck["b_reg1_a"].view(1, 48).to(dtype=self.dtype, device=self.device),
                "w_reg2": ck["w_reg2"].to(dtype=self.dtype, device=self.device),
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

        def _anti_ring(shape: tuple[int, ...], scale_f: float, param_key: str | None = None) -> torch.Tensor:
            # Строго симметричный нулевой шум +eps / -eps вокруг центра (без искажения антитетического градиента OpenAI-ES!)
            r_view = ring_h.view((half,) + (1,) * len(shape))
            eps = torch.randn((half, *shape), device=self.device, dtype=torch.float32) * (scale_f * r_view)
            return torch.stack([eps, -eps], dim=1).reshape(2 * half, *shape)

        # Мутация w1: для market-среза (0..40) шаг 1.8x, для новых 44D Anti-FOMO каналов (40..44) шаг 3.2x, для agent-среза 0.9x
        dw1 = _anti_ring((TOTAL_DIM_ORDER3, 384), sig_trunk, "w1").to(self.dtype)
        dw1[:, 0:1, :] *= 2.80
        dw1[:, 1:40, :] *= 1.80
        dw1[:, 40:MKT_DIM_ORDER3, :] *= 3.20
        dw1[:, MKT_DIM_ORDER3:, :] *= 0.90

        self.w1[2 : 2 + 2 * half].copy_(base_gpu["w1"].unsqueeze(0) + dw1)
        self.b1[2 : 2 + 2 * half].copy_(base_gpu["b1"].unsqueeze(0) + _anti_ring((1, 384), sig_trunk * 0.6, "b1").to(self.dtype))

        self.w_res1[2 : 2 + 2 * half].copy_(base_gpu["w_res1"].unsqueeze(0) + _anti_ring((256, 128), sig_trunk, "w_res1").to(self.dtype))
        self.b_res1[2 : 2 + 2 * half].copy_(base_gpu["b_res1"].unsqueeze(0) + _anti_ring((1, 128), sig_trunk * 0.5, "b_res1").to(self.dtype))
        self.w_res2[2 : 2 + 2 * half].copy_(base_gpu["w_res2"].unsqueeze(0) + _anti_ring((64, 256), sig_trunk, "w_res2").to(self.dtype))

        moe_sig = 0.0090 * float(sigma_scale)
        dw2m = _anti_ring((256, MOE_OUT_DIM_ORDER3), moe_sig, "w2_moe")
        db2m = _anti_ring((1, MOE_OUT_DIM_ORDER3), moe_sig * 1.35, "b2_moe")
        if K_MODES_ORDER3 >= 8:
            s_new = 4 * EXPERT_OUT_DIM_ORDER3
            dw2m[:, :, :s_new] *= 0.75
            dw2m[:, :, s_new:] *= 1.85
            db2m[:, :, :s_new] *= 0.75
            db2m[:, :, s_new:] *= 2.00
        self.w2_moe[2 : 2 + 2 * half].copy_(base_gpu["w2_moe"].unsqueeze(0) + dw2m.to(self.dtype))
        self.b2_moe[2 : 2 + 2 * half].copy_(base_gpu["b2_moe"].unsqueeze(0) + db2m.to(self.dtype))

        dw_g1 = _anti_ring((ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), moe_sig * 1.85, "w_gate1").to(self.dtype)
        dw_g1[:, 288:, :] *= 2.60
        self.w_gate1[2 : 2 + 2 * half].copy_(base_gpu["w_gate1"].unsqueeze(0) + dw_g1)
        self.b_gate1[2 : 2 + 2 * half].copy_(base_gpu["b_gate1"].unsqueeze(0) + _anti_ring((1, ROUTER_H_DIM_ORDER3), moe_sig * 1.35, "b_gate1").to(self.dtype))
        dw_g2 = _anti_ring((ROUTER_H_DIM_ORDER3, K_MODES_ORDER3), moe_sig * 2.20, "w_gate2")
        db_g2 = _anti_ring((1, K_MODES_ORDER3), moe_sig * 1.60, "b_gate2")
        if K_MODES_ORDER3 >= 8:
            dw_g2[:, :, 4:] *= 1.75
            db_g2[:, :, 4:] *= 1.75
        self.w_gate2[2 : 2 + 2 * half].copy_(base_gpu["w_gate2"].unsqueeze(0) + dw_g2.to(self.dtype))
        self.b_gate2[2 : 2 + 2 * half].copy_(base_gpu["b_gate2"].unsqueeze(0) + db_g2.to(self.dtype))

        self.w_rec[2 : 2 + 2 * half].copy_(base_gpu["w_rec"].unsqueeze(0) + _anti_ring((256, 32), sig_trunk * 0.6, "w_rec").to(self.dtype))
        self.b_rec[2 : 2 + 2 * half].copy_(base_gpu["b_rec"].unsqueeze(0) + _anti_ring((1, 32), sig_trunk * 0.4, "b_rec").to(self.dtype))

        bias_reg_sig = 0.030 * float(sigma_scale)
        dw_reg1 = _anti_ring((REG_IN_DIM_ORDER3, 48), sig_trunk * 0.85, "w_reg1")
        if REG_IN_DIM_ORDER3 >= 32:
            dw_reg1[:, 28:, :] *= 1.90
        self.w_reg1[2 : 2 + 2 * half].copy_(base_gpu["w_reg1"].unsqueeze(0) + dw_reg1.to(self.dtype))
        self.b_reg1_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_b"].unsqueeze(0) + _anti_ring((1, 48), sig_trunk * 0.60, "b_reg1_b").to(self.dtype))
        self.b_reg1_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_a"].unsqueeze(0) + _anti_ring((1, 48), sig_trunk * 0.60, "b_reg1_a").to(self.dtype))
        self.w_reg2[2 : 2 + 2 * half].copy_(base_gpu["w_reg2"].unsqueeze(0) + _anti_ring((48, 2), sig_trunk * 0.85, "w_reg2").to(self.dtype))
        self.b_reg2_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_b"].unsqueeze(0) + _anti_ring((1, 2), bias_reg_sig, "b_reg2_b").to(self.dtype))
        self.b_reg2_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_a"].unsqueeze(0) + _anti_ring((1, 2), bias_reg_sig, "b_reg2_a").to(self.dtype))

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
    n_groups: int = 256,
    win_len: int = 2400,
    n_val_windows: int = 16,
    device_str: str = "cuda:0",
    time_budget_hours: float = 2.5,
    island_id: int = 0,
    num_islands: int = 2,
    island_mode: str = "exploit",
    early_stop_minutes: float = 40.0,
    log_file: str | None = None,
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

    # 1. Загрузка строго каузальных 60D данных Signal Stack v2 и выделение РЕАЛЬНОЙ суб-центовой эпохи SafeTrade ($0.0001 tick!)
    log_msg("📦 Загрузка каузальной 60D матрицы Signal Stack v2 и выделение эпохи суб-центового тика $0.0001...")
    mkt_raw = np.load(BASE_DIR / "data/cache/rl_hybrid_market_obs_60d.npz", mmap_mode="r")
    lob_raw = np.load(BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz", mmap_mode="r")

    b1_full = lob_raw["bid_p"][:, 0]
    subcent_hits = np.where(np.abs(b1_full * 100.0 - np.round(b1_full * 100.0)) > 1e-3)[0]
    subcent_offset = int(subcent_hits[0]) if len(subcent_hits) > 0 else 0
    log_msg(f"🔍 Обнаружена граница перехода биржи на тик $0.0001: idx={subcent_offset:,} (отсечено {subcent_offset:,} старых $0.01 тиков!)")

    mkt_sub = np.asarray(mkt_raw["market_obs_60d"][subcent_offset:], dtype=np.float32)
    total_rows = len(mkt_sub)
    tick_sz = 0.0001
    if win_len > total_rows // 10:
        win_len = 1801

    mkt_gpu = torch.from_numpy(mkt_sub).to(device=device, dtype=torch.bfloat16)
    b10_gpu = torch.from_numpy(np.asarray(lob_raw["bid_p"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    a10_gpu = torch.from_numpy(np.asarray(lob_raw["ask_p"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    bq10_gpu = torch.from_numpy(np.asarray(lob_raw["bid_q"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    aq10_gpu = torch.from_numpy(np.asarray(lob_raw["ask_q"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    sv_gpu = torch.from_numpy(np.asarray(lob_raw["sell_vol"][subcent_offset:], dtype=np.float32)).to(device=device)
    bv_gpu = torch.from_numpy(np.asarray(lob_raw["buy_vol"][subcent_offset:], dtype=np.float32)).to(device=device)

    # 2. Двухмасштабное Purged-разбиение Урока 46 (K=8 мод + 56 Val окон + 56 Blind OOS окон: Macro 100m + Micro 25m ВЧ-флуктуации!):
    oos_start = max(396000, int(total_rows * 0.83))
    macro_win_len = min(win_len, 6001)
    micro_win_len = 1501
    n_bask_w = max(28, n_val_windows)
    oos_end_macro = total_rows - macro_win_len - 100
    oos_end_micro = total_rows - micro_win_len - 100

    # 7 строго очищенных (purged) зон валидации по всей 110-часовой истории до OOS (по 4 окна на зону = 28 Macro + 28 Micro = 56 окон Val!):
    val_zones = [
        (38000, 42500),
        (88000, 92500),
        (142000, 146500),
        (196000, 200500),
        (252000, 256500),
        (308000, 312500),
        (366000, min(370500, oos_start - macro_win_len - 2500)),
    ]
    per_z = max(2, n_bask_w // len(val_zones))
    val_macro_idx = np.concatenate([np.linspace(zlo, zhi, num=per_z, dtype=int) for zlo, zhi in val_zones])[:n_bask_w]
    val_micro_idx = np.concatenate([np.linspace(zlo + 300, zhi + 3600, num=per_z, dtype=int) for zlo, zhi in val_zones])[:n_bask_w]
    val_macro_basket = [(f"VMAC{k_v:02d}", int(idx_v), 0.0, 0.0) for k_v, idx_v in enumerate(val_macro_idx)]
    val_micro_basket = [(f"VMIC{k_v:02d}", int(idx_v), 0.0, 0.0) for k_v, idx_v in enumerate(val_micro_idx)]

    oos_macro_idx = np.linspace(oos_start, oos_end_macro, num=n_bask_w, dtype=int)
    oos_micro_idx = np.linspace(oos_start + 250, oos_end_micro, num=n_bask_w, dtype=int)
    oos_macro_basket = [(f"OMAC{k_o:02d}", int(idx_o), 0.0, 0.0) for k_o, idx_o in enumerate(oos_macro_idx)]
    oos_micro_basket = [(f"OMIC{k_o:02d}", int(idx_o), 0.0, 0.0) for k_o, idx_o in enumerate(oos_micro_idx)]

    # 8 строго изолированных (0 пересечений с Val и OOS!) зон обучения:
    rng_train = np.random.default_rng(20261147 + island_id * 1009)
    tr_segments = [
        (600, 31500),
        (51000, 81500),
        (101000, 135500),
        (155000, 189500),
        (209000, 245500),
        (265000, 301500),
        (321000, 359500),
        (379000, 389500),
    ]
    tr_seg_probs = np.array([0.12, 0.13, 0.14, 0.14, 0.14, 0.14, 0.14, 0.05], dtype=np.float64)

    def sample_train_basket(is_micro: bool = False) -> list[tuple[str, int, float, float]]:
        seg_ids = rng_train.choice(len(tr_segments), size=n_bask_w, p=tr_seg_probs)
        out_tr = []
        extra_hi = 4000 if is_micro else 0
        for k_t, sid in enumerate(seg_ids):
            lo_s, hi_s = tr_segments[int(sid)]
            s_t = int(rng_train.integers(lo_s, hi_s + extra_hi + 1))
            out_tr.append((f"{'TMIC' if is_micro else 'TMAC'}{k_t:02d}", s_t, 0.0, 0.0))
        return out_tr

    def eval_dual_scale(cands: list[dict], mac_bask: list, mic_bask: list, mac_len: int | None = None) -> list[dict]:
        r_mac = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu,
            bq10_all=bq10_gpu, aq10_all=aq10_gpu, sv_all=sv_gpu, bv_all=bv_gpu,
            cand_states=cands, val_windows=mac_bask, win_len=(mac_len or macro_win_len), tick_sz=tick_sz, device=device,
        )
        r_mic = evaluate_order3_basket(
            mkt_39d_all=mkt_gpu, b10_all=b10_gpu, a10_all=a10_gpu,
            bq10_all=bq10_gpu, aq10_all=aq10_gpu, sv_all=sv_gpu, bv_all=bv_gpu,
            cand_states=cands, val_windows=mic_bask, win_len=micro_win_len, tick_sz=tick_sz, device=device,
        )
        merged = []
        for a, b in zip(r_mac, r_mic):
            # Усиливаем вес 25-минутных окон ВЧ-флуктуаций (2.2x на bp в расчете на единицу времени), чтобы микро-маркетмейкинг давал сильный градиент наравне с макро-снайпером!
            joint_geo = 0.45 * float(a["geo_score_bp"]) + 0.55 * (2.20 * float(b["geo_score_bp"]))
            tot_f = float(a["fills"]) + float(b["fills"])
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
                "inv_delta_ev_pct": 0.0,
                "dd_pct": max(float(a["dd_pct"]), float(b["dd_pct"])),
                "fills": tot_f,
                "cancels": tot_c,
                "requeues": float(a["requeues"]) + float(b["requeues"]),
                "cnl_per_fill": tot_c / max(tot_f, 0.5),
                "eta_queue_pct": 0.5 * float(a["eta_queue_pct"]) + 0.5 * float(b["eta_queue_pct"]),
                "mode_probs": mp_join,
                "mode_mi_bits": round(0.5 * float(a["mode_mi_bits"]) + 0.5 * float(b["mode_mi_bits"]), 4),
                "turn_mult": round(float(a.get("turn_mult", 0.0)) + float(b.get("turn_mult", 0.0)), 4),
            })
        return merged

    log_msg(
        f"📊 Purged Multi-Scale K={K_MODES_ORDER3} Split L46 ({total_rows:,} тиков = {total_rows/3600:.1f}ч, 0 эксплоитов!): "
        f"Train=8 purged zones (28 Macro + 28 Micro) | Val={len(val_macro_basket)+len(val_micro_basket)}w (28x{macro_win_len-1}s + 28x{micro_win_len-1}s = 336 траекторий) | "
        f"Blind OOS={len(oos_macro_basket)+len(oos_micro_basket)}w [{oos_start:,}..{oos_end_micro:,}]"
    )

    # 3. Инициализация K=8 политики Урока 46 (перенос Снайпера L45 и L44 в моды 0..3 + 4 новых ВЧ-специалиста в модах 4..7):
    policy = MultimodalMoEPolicyOrder3(n_groups=n_groups, n_budgets=6, device=device)
    champ45_p = SAVE_DIR / "bak_l45_sniper" / "order3_champ_latest.pt"
    champ44_p = SAVE_DIR / "bak_l44_cash" / "order3_champ_latest.pt"
    champ43_p = SAVE_DIR / "bak_l43_v21" / "order3_champ_latest.pt"
    is_warm_l43 = False
    seed_sniper = champ45_p if champ45_p.exists() else LATEST_CHAMP_PATH
    seed_cash = champ44_p if champ44_p.exists() else seed_sniper
    log_msg(f"🧩 Загрузка семян ({seed_sniper.name} + {seed_cash.name}) и расширение до K={K_MODES_ORDER3} мод для 8-часового Урока 46...")

    st_l45_k8 = policy.load_initial_champion(seed_sniper)
    st_l44_raw = policy.load_initial_champion(seed_cash)
    st_l43_raw = policy.load_initial_champion(champ43_p) if champ43_p.exists() else {k: v.clone() for k, v in st_l44_raw.items()}

    def _boost_hf_modes(st_in: dict, gate_boost: float = 0.60, exp_boost: float = 0.80, adapt_legacy_03: bool = False) -> dict:
        st_o = {k: v.clone() for k, v in st_in.items()}
        b2 = st_o["b2_moe"].float().clone().view(1, K_MODES_ORDER3, EXPERT_OUT_DIM_ORDER3)
        bg = st_o["b_gate2"].float().clone().view(1, K_MODES_ORDER3)
        if adapt_legacy_03:
            b2[:, :4, 25] = b2[:, :4, 25] - 1.10
            b2[:, :4, 38] = b2[:, :4, 38] - 2.80
            b2[:, :4, 39] = b2[:, :4, 39] - 2.50
        b2[:, 4:, 0] = b2[:, 4:, 0] + exp_boost
        bg[:, 4:] = bg[:, 4:] + gate_boost
        st_o["b2_moe"] = b2.view_as(st_o["b2_moe"]).to(policy.dtype)
        st_o["b_gate2"] = bg.view_as(st_o["b_gate2"]).to(policy.dtype)
        return _normalize_weights_order3(st_o, policy.dtype)

    st_s0 = _boost_hf_modes(st_l45_k8, gate_boost=0.20, exp_boost=0.30, adapt_legacy_03=False)
    st_s1 = _boost_hf_modes(st_l45_k8, gate_boost=0.95, exp_boost=1.20, adapt_legacy_03=False)
    st_s2 = _boost_hf_modes(st_l44_raw, gate_boost=0.80, exp_boost=1.10, adapt_legacy_03=True)
    st_s3 = _boost_hf_modes(st_l43_raw, gate_boost=1.10, exp_boost=1.50, adapt_legacy_03=True)

    seed_pool = [st_s0, st_s1, st_s2, st_s3] if island_id == 0 else [st_s1, st_s3, st_s2, st_s0]
    seed_val = eval_dual_scale(seed_pool, sample_train_basket(False), sample_train_basket(True))
    best_seed_i = int(np.argmax([r["geo_score_bp"] for r in seed_val]))
    champion_state = seed_pool[best_seed_i]
    val_basket = val_macro_basket + val_micro_basket
    oos_basket = oos_macro_basket + oos_micro_basket
    log_msg(
        f"🌱 Выбрано стартовое семя #{best_seed_i} (TrainJointGeo={seed_val[best_seed_i]['geo_score_bp']:+.2f} bp | "
        f"Macro={seed_val[best_seed_i]['macro_geo_bp']:+.2f} | Micro={seed_val[best_seed_i]['micro_geo_bp']:+.2f} | "
        f"Turn={seed_val[best_seed_i].get('turn_mult', 0.0):.2f}x)"
    )

    # 4. Базовая оценка на 56 окнах Validation и 56 окнах Blind OOS
    log_msg("⚡ Оценка базового K=8 чемпиона на 56 окнах Validation и 56 окнах Blind OOS...")
    t0_eval = time.time()
    init_res = eval_dual_scale([champion_state], val_macro_basket, val_micro_basket)[0]
    init_oos = eval_dual_scale([champion_state], oos_macro_basket, oos_micro_basket)[0]
    dt_eval0 = time.time() - t0_eval

    best_geo_bp = float(init_res["geo_score_bp"])
    best_clean_ev = float(init_res["clean_ev_pct"])
    baseline_oos_geo = float(init_oos["geo_score_bp"])
    best_state = {k: v.clone() for k, v in champion_state.items()}
    best_metrics = dict(init_res)

    log_msg(
        f"🏆 Базовый K=8 результат: ValJointGeo={best_geo_bp:+.2f} bp (Macro={init_res['macro_geo_bp']:+.2f}, Micro={init_res['micro_geo_bp']:+.2f}, EV={best_clean_ev:+.4f}%) | "
        f"OOS_JointGeo={baseline_oos_geo:+.2f} bp (OOS_EV={init_oos['clean_ev_pct']:+.4f}%) | "
        f"Fills={init_res['fills']:.1f} | EvalTime={dt_eval0:.2f}s"
    )

    elite_archive = [(best_geo_bp, {k: v.clone() for k, v in best_state.items()})]

    # 5. Главный цикл оптимизации с Centered-Rank OpenAI-ES + Auto-Phase Step-Down (P1 -> P2 -> P3)
    patience = 0
    phase_idx = 0
    last_improve_ts = time.time()
    phase_enter_ts = time.time()
    sync_path = SAVE_DIR / "global_best.pt"
    w_cnt_f = max(1.0, float(len(val_basket)))

    for it in range(1, max_iters + 1):
        t_it = time.time()

        # 🧬 Асинхронная кросс-островная миграция элиты (исправленная топология num_islands!)
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
                    if "w1" in n_ck and n_ck["w1"].shape[0] == TOTAL_DIM_ORDER3 and n_ck.get("wf_version") == "pure_wf_v16_l46_k8_multibasket":
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
                    if "w1" in g_ck and g_ck["w1"].shape[0] == TOTAL_DIM_ORDER3 and g_ck.get("wf_version") == "pure_wf_v16_l46_k8_multibasket":
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

        # Проверка временного лимита и Auto-Phase Step-Down под 8-часовой горизонт (P1 -> P2 -> P3):
        elapsed_hours = (time.time() - t_start) / 3600.0
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

        # Двухмасштабный проход на 28 Macro (100m) + 28 Micro (25m ВЧ-флуктуации) с переиспользованием 2 единых CUDAGraph(W=28, C=192)!
        t_batch = time.time()
        tr_results = eval_dual_scale(cand_list, sample_train_basket(False), sample_train_basket(True))

        # Шаг Б: Гибрид Centered-Rank OpenAI-ES (4 масштаба градиента) + Топ-28 индивидуальных мутантов -> 32 на 56 окнах Validation!
        tr_geos = np.array([r["geo_score_bp"] for r in tr_results], dtype=np.float64)
        es_cands = policy.build_openai_es_candidates(center, tr_geos, lr_scales=(0.35, 0.75, 1.25, 1.85))
        k_mut_top = max(4, min(32 - len(es_cands), len(cand_list)))
        top_mut_idx = np.argsort(tr_geos)[-k_mut_top:][::-1].tolist()
        val_cands = es_cands + [cand_list[i] for i in top_mut_idx]
        val_results = eval_dual_scale(val_cands, val_macro_basket, val_micro_basket)
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
                "wf_version": "pure_wf_v16_l46_k8_multibasket",
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
                    if cur_sync.get("wf_version") == "pure_wf_v16_l46_k8_multibasket" and float(cur_sync.get("geo_score_bp", -999.0)) >= best_geo_bp:
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
                log_msg(f"👑 [GLOBAL K=8 RECORD!] Island #{island_id}: ValJointGeo={best_geo_bp:+.2f} bp (Macro={best_metrics['macro_geo_bp']:+.2f} | Micro={best_metrics['micro_geo_bp']:+.2f})!")

            src_tag = f"ES-Grad#{best_v_local}" if is_es_win else "Mutant"
            log_msg(
                f"⭐ [It {it:03d} | +{delta_bp:.2f} bp | {src_tag}] NEW K=8 CHAMPION! "
                f"JointGeo={best_geo_bp:+.2f} bp (Mac={best_metrics['macro_geo_bp']:+.2f}, Mic={best_metrics['micro_geo_bp']:+.2f}) | "
                f"EV={best_clean_ev:+.4f}% | Turn={best_metrics.get('turn_mult', 0.0):.2f}x | "
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

    # 6. Router-Preserving Soup (сохраняет острые шлюзы K=8 маршрутизатора топ-1 элиты!) + Слепой 56-оконный OOS-Экзамен!
    elite_archive.sort(key=lambda x: x[0], reverse=True)
    final_cand = best_state
    if len(elite_archive) >= 2:
        log_msg("🍲 Сборка Router-Preserving K=8 Soup (0.80 Top1 + 0.20 Top2 без размытия шлюзов w_gate)...")
        gate_keys = {"w_gate1", "b_gate1", "w_gate2", "b_gate2"}
        soup_state = {}
        for k in MultimodalMoEPolicyOrder3.PARAM_KEYS:
            if k in gate_keys:
                soup_state[k] = elite_archive[0][1][k].detach().to(device=device, dtype=policy.dtype).clone()
            else:
                t0_w = elite_archive[0][1][k].detach().to(device=device, dtype=torch.float32)
                t1_w = elite_archive[1][1][k].detach().to(device=device, dtype=torch.float32)
                soup_state[k] = (0.80 * t0_w + 0.20 * t1_w).to(dtype=policy.dtype)
        soup_norm = _normalize_weights_order3(soup_state, policy.dtype)
        soup_val = eval_dual_scale([soup_norm], val_macro_basket, val_micro_basket)[0]
        log_msg(f"🍲 Router-Preserving Soup JointGeo={soup_val['geo_score_bp']:+.2f} bp (vs Top1 {best_geo_bp:+.2f} bp)")
        save_soup = True
        if SOUP_CHAMP_PATH.exists():
            try:
                old_sp = torch.load(SOUP_CHAMP_PATH, map_location="cpu", weights_only=False)
                if old_sp.get("wf_version") == "pure_wf_v16_l46_k8_multibasket" and float(old_sp.get("geo_score_bp", -999.0)) >= float(soup_val["geo_score_bp"]):
                    save_soup = False
            except Exception:
                pass
        if save_soup:
            torch.save({"arch_version": "order3_v1", "wf_version": "pure_wf_v16_l46_k8_multibasket", "pure_endogenous": True, "k_modes": K_MODES_ORDER3, "is_soup": True, "geo_score_bp": soup_val["geo_score_bp"], "clean_ev_pct": soup_val["clean_ev_pct"], "in_dim": TOTAL_DIM_ORDER3, "mkt_dim": MKT_DIM_ORDER3, "ag_dim": AG_DIM_ORDER3, **soup_norm}, SOUP_CHAMP_PATH)
        if float(soup_val["geo_score_bp"]) > best_geo_bp:
            final_cand = soup_norm

    oos_final = eval_dual_scale([final_cand], oos_macro_basket, oos_micro_basket)[0]
    log_msg(
        f"🧪 [BLIND 56-WINDOW OOS EXAM] Baseline OOS={baseline_oos_geo:+.2f} bp -> Final OOS={oos_final['geo_score_bp']:+.2f} bp "
        f"(Macro={oos_final['macro_geo_bp']:+.2f}, Micro={oos_final['micro_geo_bp']:+.2f}) | "
        f"OOS_EV={oos_final['clean_ev_pct']:+.4f}% (Mac={oos_final['macro_ev_pct']:+.4f}%, Mic={oos_final['micro_ev_pct']:+.4f}%) | "
        f"OOS_Turn={oos_final.get('turn_mult', 0.0):.2f}x | OOS_CVaR25={oos_final['cvar25_bp']:+.2f} bp | "
        f"OOS_Fills/w={oos_final['fills']/max(1.0, float(len(oos_basket))):.1f}"
    )
    log_msg(f"🏁 Обучение Order-3 K=8 Multi-Scale L46 завершено за {(time.time() - t_start)/3600:.2f}ч! Чемпион: {LATEST_CHAMP_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Order-3 RL Champion Walk-Forward v4 (44D Continuous) Trainer")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g. cuda:0 or cuda:1)")
    parser.add_argument("--groups", type=int, default=192, help="Number of candidate groups")
    parser.add_argument("--iters", type=int, default=800, help="Maximum iterations")
    parser.add_argument("--hours", type=float, default=2.5, help="Time budget in hours")
    parser.add_argument("--win-len", type=int, default=4801, help="Window length in ticks (4801 -> 4800 steps = 80m continuous horizon)")
    parser.add_argument("--val-windows", type=int, default=16, help="Number of windows in Train/Val/OOS baskets")
    parser.add_argument("--island-id", type=int, default=0, help="Island ID (0..num_islands-1)")
    parser.add_argument("--num-islands", type=int, default=2, help="Total active islands in ring")
    parser.add_argument("--early-stop-min", type=float, default=40.0, help="Early stopping stagnation minutes (0 to disable)")
    parser.add_argument("--island-mode", type=str, default="exploit", choices=["exploit", "balanced", "explore", "wide"], help="Island Mode")
    parser.add_argument("--log-file", type=str, default=None, help="Custom log file")
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
    )


if __name__ == "__main__":
    main()
