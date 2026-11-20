#!/usr/bin/env python3
"""
_tools/lesson_27_multimodal_moe_trainer.py
=============================================================================
Урок 27: Мультимодальная эндогенная архитектура (`K = 4 Latent MoE` +
Контекстный маршрутизатор `288D -> 64 -> 4D` + Веерное эшелонирование капитала +
`26D -> 48 -> 2D` Регулятор + Token-Bucket `+0.35/с` cap `10.0`) и
8.5-часовой Мультиволновой Поисковый Движок:
  - 4-полюсный Warm-Start экспертов из лучших ортогональных чемпионов Урока 26,
  - Смелые многомасштабные кольца мутаций (0.50x / 1.00x / 1.85x одновременно),
  - Циклические волны: Bold Explore -> Zoom-In (0.42x) -> Micro-Polish (0.20x) ->
    Ортогональный Basin-Hopping рестарт на всём протяжении 8.5-часового бюджета!
=============================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path("/home/restorator/crypto_bot")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_free_multi_order_env import VectorizedFreeMultiOrderEnv
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_18_multistack_lifo_survival_trainer import FusedBF16BackboneRunner
from _tools.lesson_23_pump_capture_penny_l1_trainer import _build_roll_sell_30s
from _tools.lesson_26_endogenous_nonlinear_trainer import (
    SEMANTIC_BLOCKS_L26 as SEMANTIC_BLOCKS_L27,
    N_BLOCKS_L26 as N_BLOCKS_L27,
    expand_block_gates_to_480,
)
from _tools.fused_l27_multimodal_tick import (
    MKT_IDX_L27,
    AG_IDX_L27,
    K_MODES_L27,
    EXPERT_OUT_DIM_L27,
    MOE_OUT_DIM_L27,
    ROUTER_IN_DIM_L27,
    ROUTER_H_DIM_L27,
    REG_IN_DIM_L27,
    TOKEN_BUCKET_CAP_L27,
    precompute_window_market_z370,
    normalize_market_z370_per_block,
    fused_l27_multimodal_tick,
    evaluate_l27_basket,
)

CACHE_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
RELEASE_CHAMP_PATH = BASE_DIR / "models/prl/release_champion_v1.pt"
L26_DIR = BASE_DIR / "models/prl/lesson_26_endogenous_nonlinear"
L26_EXPERT_POLES = (
    L26_DIR / "l26_champ_iter_0027.pt",      # Mode 0: Alpha Sniper (+0.307% OOS)
    L26_DIR / "l26_champ_iter_0013.pt",      # Mode 1: Active Spread / Multi-Fill Maker
    L26_DIR / "l26_champ_iter_0292.pt",      # Mode 2: Crash / Tail Armor (DD=1.64%)
    L26_DIR / "lesson_26_champ_resume.pt",   # Mode 3: Co-Adapted Regulator Specialist (it 520)
)

SAVE_DIR = BASE_DIR / "models/prl/lesson_27_multimodal_moe"
LATEST_CHAMP_PATH = SAVE_DIR / "lesson_27_champ_latest.pt"
RESUME_CHAMP_PATH = SAVE_DIR / "lesson_27_champ_resume.pt"
SOUP_CHAMP_PATH = SAVE_DIR / "l27_soup_top3.pt"
LOG_PATH = BASE_DIR / "data/logs/lesson_27_train.log"

MSK_TZ = timezone(timedelta(hours=3))
VAL_START_TS = datetime(2026, 11, 11, 22, 0, tzinfo=MSK_TZ).timestamp()
BLIND_START_TS = datetime(2026, 11, 12, 20, 30, tzinfo=MSK_TZ).timestamp()


def log_msg(msg: str) -> None:
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts_str}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _normalize_weights_l27(state: dict, dtype: torch.dtype) -> dict:
    out = {}
    w1_f = state["w1"].float()
    w1_parts = []
    for _, s, e, _ in SEMANTIC_BLOCKS_L27:
        wb = w1_f[..., s:e, :]
        cap_b = 6.5 * math.sqrt(float(e - s))
        nb = torch.linalg.vector_norm(wb, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        w1_parts.append(wb * torch.clamp(cap_b / nb, max=1.0))
    out["w1"] = torch.cat(w1_parts, dim=-2).to(dtype)
    out["b1"] = torch.clamp(state["b1"].float(), -2.2, 2.2).to(dtype)

    # Нормализация каждого из 4 экспертов независимо, чтобы один эксперт не забивал норму остальных:
    w2m_f = state["w2_moe"].float()
    w2m_parts = []
    for k_i in range(K_MODES_L27):
        s_k, e_k = k_i * EXPERT_OUT_DIM_L27, (k_i + 1) * EXPERT_OUT_DIM_L27
        wk = w2m_f[..., s_k:e_k]
        nk = torch.linalg.vector_norm(wk, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        w2m_parts.append(wk * torch.clamp(62.0 / nk, max=1.0))
    out["w2_moe"] = torch.cat(w2m_parts, dim=-1).to(dtype)
    out["b2_moe"] = torch.clamp(state["b2_moe"].float(), -3.5, 3.0).to(dtype)

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
    out["b_reg1_b"] = torch.clamp(state["b_reg1_b"].float(), -2.5, 2.5).to(dtype)
    out["b_reg1_a"] = torch.clamp(state["b_reg1_a"].float(), -2.5, 2.5).to(dtype)
    out["b_reg2_b"] = torch.clamp(state["b_reg2_b"].float(), -3.5, 3.0).to(dtype)
    out["b_reg2_a"] = torch.clamp(state["b_reg2_a"].float(), -3.5, 3.0).to(dtype)
    return out


class MultimodalMoEPolicyL27:
    """
    Популяционная 480D политика Урока 27:
      - Общий ствол ResSwiGLU (256D) + рекуррентная память mem_h (16D),
      - Контекстный маршрутизатор (288D -> 64 -> 4D),
      - K = 4 латентных эксперта по 45D (упакованы в [G, 256, 180]),
      - Послотовый Регулятор ценности очереди (26D -> 48 -> 2D).
    """

    PARAM_KEYS = (
        "w1", "b1", "w_res1", "b_res1", "w_res2",
        "w2_moe", "b2_moe",
        "w_gate1", "b_gate1", "w_gate2", "b_gate2",
        "w_rec", "b_rec",
        "w_reg1", "b_reg1_b", "b_reg1_a", "w_reg2", "b_reg2_b", "b_reg2_a",
    )

    def __init__(self, backbone: MultiTaskLOBNet, n_groups: int = 1024, n_budgets: int = 6, device: torch.device = torch.device("cuda")):
        self.backbone = backbone
        self.n_groups = n_groups
        self.n_budgets = n_budgets
        self.pop_size = n_groups * n_budgets
        self.device = device
        self.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        self.w1 = torch.empty((n_groups, 480, 384), dtype=self.dtype, device=device)
        self.b1 = torch.empty((n_groups, 1, 384), dtype=self.dtype, device=device)
        self.w_res1 = torch.empty((n_groups, 256, 128), dtype=self.dtype, device=device)
        self.b_res1 = torch.empty((n_groups, 1, 128), dtype=self.dtype, device=device)
        self.w_res2 = torch.empty((n_groups, 64, 256), dtype=self.dtype, device=device)

        self.w2_moe = torch.empty((n_groups, 256, MOE_OUT_DIM_L27), dtype=self.dtype, device=device)
        self.b2_moe = torch.empty((n_groups, 1, MOE_OUT_DIM_L27), dtype=self.dtype, device=device)
        self.w_gate1 = torch.empty((n_groups, ROUTER_IN_DIM_L27, ROUTER_H_DIM_L27), dtype=self.dtype, device=device)
        self.b_gate1 = torch.empty((n_groups, 1, ROUTER_H_DIM_L27), dtype=self.dtype, device=device)
        self.w_gate2 = torch.empty((n_groups, ROUTER_H_DIM_L27, K_MODES_L27), dtype=self.dtype, device=device)
        self.b_gate2 = torch.empty((n_groups, 1, K_MODES_L27), dtype=self.dtype, device=device)

        self.w_rec = torch.empty((n_groups, 256, 32), dtype=self.dtype, device=device)
        self.b_rec = torch.empty((n_groups, 1, 32), dtype=self.dtype, device=device)

        self.w_reg1 = torch.empty((n_groups, REG_IN_DIM_L27, 48), dtype=self.dtype, device=device)
        self.b_reg1_b = torch.empty((n_groups, 1, 48), dtype=self.dtype, device=device)
        self.b_reg1_a = torch.empty((n_groups, 1, 48), dtype=self.dtype, device=device)
        self.w_reg2 = torch.empty((n_groups, 48, 2), dtype=self.dtype, device=device)
        self.b_reg2_b = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)
        self.b_reg2_a = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)

        self.g_logits = torch.zeros((n_groups, N_BLOCKS_L27), dtype=torch.float32, device=device)
        mut_vec = torch.ones(480, dtype=torch.float32, device=device)
        for _, s, e, mult in SEMANTIC_BLOCKS_L27:
            mut_vec[s:e] = mult
        self.mut_profile = mut_vec.view(1, 480, 1).to(self.dtype)

        # Кольца радиусов для half антитетических пар: 30% Fine (0.50x), 40% Core (1.00x), 30% Bold (1.85x):
        half = max(1, (self.n_groups - 2) // 2)
        n_fine = max(1, int(round(0.30 * half)))
        n_bold = max(1, int(round(0.30 * half)))
        n_core = max(0, half - n_fine - n_bold)
        rings = [0.50] * n_fine + [1.00] * n_core + [1.85] * n_bold
        self.ring_mult_half = torch.tensor(rings[:half], dtype=torch.float32, device=device)

    def load_from_checkpoint(self, ck: dict, expert_pole_paths: tuple[Path, ...] = L26_EXPERT_POLES) -> tuple[dict, torch.Tensor]:
        is_l27 = int(ck.get("lesson", 0)) >= 27 and "w2_moe" in ck
        if is_l27:
            state = {
                k: ck[k].to(dtype=self.dtype, device=self.device)
                for k in self.PARAM_KEYS
            }
            state["b1"] = state["b1"].view(1, 384)
            state["b_res1"] = state["b_res1"].view(1, 128)
            state["b2_moe"] = state["b2_moe"].view(1, MOE_OUT_DIM_L27)
            state["b_gate1"] = state["b_gate1"].view(1, ROUTER_H_DIM_L27)
            state["b_gate2"] = state["b_gate2"].view(1, K_MODES_L27)
            state["b_rec"] = state["b_rec"].view(1, 32)
            state["b_reg1_b"] = state["b_reg1_b"].view(1, 48)
            state["b_reg1_a"] = state["b_reg1_a"].view(1, 48)
            state["b_reg2_b"] = state["b_reg2_b"].view(1, 2)
            state["b_reg2_a"] = state["b_reg2_a"].view(1, 2)
            gl = ck.get("gate_logit", torch.zeros(N_BLOCKS_L27)).float().cpu().clone()
            state = _normalize_weights_l27(state, self.dtype)
            cpu_st = {k: v.detach().cpu().clone() for k, v in state.items()}
            self.sample_population(cpu_st, gl, stage="explore", sigma_scale=1.0)
            return cpu_st, gl

        # 4-Полюсный Warm-Start из ортогональных чемпионов Урока 26!
        gen = torch.Generator(device="cpu").manual_seed(20262701)
        w2_moe_new = torch.zeros((256, MOE_OUT_DIM_L27), dtype=torch.float32)
        b2_moe_new = torch.zeros((1, MOE_OUT_DIM_L27), dtype=torch.float32)
        # Начальные смещения канала #44 (веерность эшелонов): эксперты получают разную начальную ширину веера!
        fan_init_biases = [0.15, -0.55, 0.35, -0.25]

        for k_i in range(K_MODES_L27):
            p_path = expert_pole_paths[k_i] if k_i < len(expert_pole_paths) and expert_pole_paths[k_i].exists() else None
            ck_p = torch.load(p_path, map_location="cpu", weights_only=False) if p_path is not None else ck
            s_k, e_k = k_i * EXPERT_OUT_DIM_L27, (k_i + 1) * EXPERT_OUT_DIM_L27
            w2_p = ck_p["w2"].float()
            b2_p = ck_p["b2"].float().view(-1)
            d_copy = min(44, w2_p.shape[1])
            w2_moe_new[:, s_k : s_k + d_copy] = w2_p[:, :d_copy]
            b2_moe_new[0, s_k : s_k + d_copy] = b2_p[:d_copy]
            w2_moe_new[:, e_k - 1] = torch.randn((256,), generator=gen) * 0.015
            b2_moe_new[0, e_k - 1] = fan_init_biases[k_i]

        # Контекстный маршрутизатор 288D -> 64 -> 4D: мягкий старт с опорой на Альфа-полюса (Mode 0 & Mode 1):
        w_gate1_new = torch.randn((ROUTER_IN_DIM_L27, ROUTER_H_DIM_L27), generator=gen) * 0.035
        # Усиленная чувствительность маршрутизатора к acct_8d (256:264) и queue_8d (264:272) с первого шага:
        w_gate1_new[256:272, :] *= 2.2
        b_gate1_new = torch.zeros((1, ROUTER_H_DIM_L27), dtype=torch.float32)
        w_gate2_new = torch.randn((ROUTER_H_DIM_L27, K_MODES_L27), generator=gen) * 0.045
        b_gate2_new = torch.tensor([[0.45, 0.30, -0.15, -0.10]], dtype=torch.float32)

        # Расширение Регулятора с 22D до 26D (+4D вектор активной моды g_mode):
        w_reg1_old = ck["w_reg1"].float()
        w_reg1_new = torch.zeros((REG_IN_DIM_L27, 48), dtype=torch.float32)
        in_copy = min(REG_IN_DIM_L27, w_reg1_old.shape[0])
        w_reg1_new[:in_copy, :] = w_reg1_old[:in_copy, :]
        w_reg1_new[in_copy:, :] = torch.randn((REG_IN_DIM_L27 - in_copy, 48), generator=gen) * 0.015

        state = {
            "w1": ck["w1"].float().to(dtype=self.dtype, device=self.device),
            "b1": ck["b1"].view(1, 384).to(dtype=self.dtype, device=self.device),
            "w_res1": ck["w_res1"].to(dtype=self.dtype, device=self.device),
            "b_res1": ck["b_res1"].view(1, 128).to(dtype=self.dtype, device=self.device),
            "w_res2": ck["w_res2"].to(dtype=self.dtype, device=self.device),
            "w2_moe": w2_moe_new.to(dtype=self.dtype, device=self.device),
            "b2_moe": b2_moe_new.to(dtype=self.dtype, device=self.device),
            "w_gate1": w_gate1_new.to(dtype=self.dtype, device=self.device),
            "b_gate1": b_gate1_new.to(dtype=self.dtype, device=self.device),
            "w_gate2": w_gate2_new.to(dtype=self.dtype, device=self.device),
            "b_gate2": b_gate2_new.to(dtype=self.dtype, device=self.device),
            "w_rec": ck["w_rec"].to(dtype=self.dtype, device=self.device),
            "b_rec": ck["b_rec"].view(1, 32).to(dtype=self.dtype, device=self.device),
            "w_reg1": w_reg1_new.to(dtype=self.dtype, device=self.device),
            "b_reg1_b": ck["b_reg1_b"].view(1, 48).to(dtype=self.dtype, device=self.device),
            "b_reg1_a": ck["b_reg1_a"].view(1, 48).to(dtype=self.dtype, device=self.device),
            "w_reg2": ck["w_reg2"].to(dtype=self.dtype, device=self.device),
            "b_reg2_b": ck["b_reg2_b"].view(1, 2).to(dtype=self.dtype, device=self.device),
            "b_reg2_a": ck["b_reg2_a"].view(1, 2).to(dtype=self.dtype, device=self.device),
        }
        state = _normalize_weights_l27(state, self.dtype)
        gl = torch.zeros(N_BLOCKS_L27, dtype=torch.float32)
        cpu_st = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.sample_population(cpu_st, gl, stage="wakeup", sigma_scale=1.0)
        return cpu_st, gl

    def effective_group_w1(self) -> torch.Tensor:
        g480 = expand_block_gates_to_480(self.g_logits).to(self.dtype).unsqueeze(-1)
        return self.w1 * g480

    def fold_gate_into_state(self, state: dict, g_logit: torch.Tensor) -> dict:
        out = {k: v.clone() for k, v in state.items()}
        g480 = expand_block_gates_to_480(g_logit.view(1, -1)).squeeze(0).to(out["w1"].dtype).unsqueeze(-1)
        out["w1"] = (out["w1"].to(g480.device) * g480).detach().cpu()
        return out

    def sample_population(
        self,
        center_state: dict,
        center_g_logit: torch.Tensor,
        stage: str,
        sigma_scale: float,
        alt_elite_state: dict | None = None,
        alt_elite_g_logit: torch.Tensor | None = None,
    ) -> None:
        base_gpu = {k: v.to(device=self.device, dtype=self.dtype) for k, v in center_state.items()}
        gl_gpu = center_g_logit.to(device=self.device, dtype=torch.float32)

        # Слот 0 = текущий поисковый центр; Слот 1 = глобальный валидационный чемпион (элитный якорь!):
        alt_gpu = {k: v.to(device=self.device, dtype=self.dtype) for k, v in alt_elite_state.items()} if alt_elite_state is not None else base_gpu
        alt_gl = alt_elite_g_logit.to(device=self.device, dtype=torch.float32) if alt_elite_g_logit is not None else gl_gpu
        for k in self.PARAM_KEYS:
            self.__dict__[k][0].copy_(base_gpu[k])
            self.__dict__[k][1].copy_(alt_gpu[k])
        self.g_logits[0].copy_(gl_gpu)
        self.g_logits[1].copy_(alt_gl)

        half = (self.n_groups - 2) // 2
        is_wakeup = stage == "wakeup"
        is_zoom = stage in ("zoom1", "zoom2")
        # В режиме zoom кольца сжимаются ещё плотнее вокруг лучшего решения (0.45x / 0.85x / 1.25x):
        if is_zoom:
            ring_h = torch.clamp(self.ring_mult_half, max=1.25) * 0.85
        else:
            ring_h = self.ring_mult_half

        sig_trunk = (0.0038 if is_wakeup else 0.0058) * float(sigma_scale)

        def _anti_ring(shape: tuple[int, ...], scale_f: float) -> torch.Tensor:
            r_view = ring_h.view((half,) + (1,) * len(shape))
            eps = torch.randn((half, *shape), device=self.device, dtype=torch.float32) * (scale_f * r_view)
            return torch.stack([eps, -eps], dim=1).reshape(2 * half, *shape)

        dw1 = _anti_ring((480, 384), sig_trunk).to(self.dtype) * self.mut_profile
        if is_wakeup:
            dw1[:, :256, :] *= 0.25
        self.w1[2 : 2 + 2 * half].copy_(base_gpu["w1"].unsqueeze(0) + dw1)
        self.b1[2 : 2 + 2 * half].copy_(base_gpu["b1"].unsqueeze(0) + _anti_ring((1, 384), sig_trunk * 0.6).to(self.dtype))

        res_mult = 0.25 if is_wakeup else 1.0
        self.w_res1[2 : 2 + 2 * half].copy_(base_gpu["w_res1"].unsqueeze(0) + _anti_ring((256, 128), sig_trunk * res_mult).to(self.dtype))
        self.b_res1[2 : 2 + 2 * half].copy_(base_gpu["b_res1"].unsqueeze(0) + _anti_ring((1, 128), sig_trunk * 0.5 * res_mult).to(self.dtype))
        self.w_res2[2 : 2 + 2 * half].copy_(base_gpu["w_res2"].unsqueeze(0) + _anti_ring((64, 256), sig_trunk * res_mult).to(self.dtype))

        # Мутация 4 экспертов (усиленный шаг на каналах нелинейной геометрии 35..44 и веерности #44):
        moe_sig = (0.0115 if is_wakeup else 0.0092) * float(sigma_scale)
        dw2m = _anti_ring((256, MOE_OUT_DIM_L27), moe_sig)
        db2m = _anti_ring((1, MOE_OUT_DIM_L27), moe_sig * 1.35)
        for k_i in range(K_MODES_L27):
            s_g = k_i * EXPERT_OUT_DIM_L27 + 34
            e_g = (k_i + 1) * EXPERT_OUT_DIM_L27
            dw2m[:, :, s_g:e_g] *= 1.65
            db2m[:, :, s_g:e_g] *= 1.95
        self.w2_moe[2 : 2 + 2 * half].copy_(base_gpu["w2_moe"].unsqueeze(0) + dw2m.to(self.dtype))
        self.b2_moe[2 : 2 + 2 * half].copy_(base_gpu["b2_moe"].unsqueeze(0) + db2m.to(self.dtype))

        # Смелая мутация Контекстного Маршрутизатора (особенно в wakeup и explore!):
        gate_sig = (0.026 if is_wakeup else (0.016 if not is_zoom else 0.009)) * float(sigma_scale)
        self.w_gate1[2 : 2 + 2 * half].copy_(base_gpu["w_gate1"].unsqueeze(0) + _anti_ring((ROUTER_IN_DIM_L27, ROUTER_H_DIM_L27), gate_sig).to(self.dtype))
        self.b_gate1[2 : 2 + 2 * half].copy_(base_gpu["b_gate1"].unsqueeze(0) + _anti_ring((1, ROUTER_H_DIM_L27), gate_sig * 0.85).to(self.dtype))
        self.w_gate2[2 : 2 + 2 * half].copy_(base_gpu["w_gate2"].unsqueeze(0) + _anti_ring((ROUTER_H_DIM_L27, K_MODES_L27), gate_sig * 1.15).to(self.dtype))
        self.b_gate2[2 : 2 + 2 * half].copy_(base_gpu["b_gate2"].unsqueeze(0) + _anti_ring((1, K_MODES_L27), gate_sig * 1.35).to(self.dtype))

        self.w_rec[2 : 2 + 2 * half].copy_(base_gpu["w_rec"].unsqueeze(0) + _anti_ring((256, 32), sig_trunk * 1.2).to(self.dtype))
        self.b_rec[2 : 2 + 2 * half].copy_(base_gpu["b_rec"].unsqueeze(0) + _anti_ring((1, 32), sig_trunk * 0.9).to(self.dtype))

        reg_sig = (0.016 if is_wakeup else 0.0115) * float(sigma_scale)
        self.w_reg1[2 : 2 + 2 * half].copy_(base_gpu["w_reg1"].unsqueeze(0) + _anti_ring((REG_IN_DIM_L27, 48), reg_sig).to(self.dtype))
        self.b_reg1_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_b"].unsqueeze(0) + _anti_ring((1, 48), reg_sig * 0.85).to(self.dtype))
        self.b_reg1_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_a"].unsqueeze(0) + _anti_ring((1, 48), reg_sig * 0.85).to(self.dtype))
        self.w_reg2[2 : 2 + 2 * half].copy_(base_gpu["w_reg2"].unsqueeze(0) + _anti_ring((48, 2), reg_sig * 0.90).to(self.dtype))
        self.b_reg2_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_b"].unsqueeze(0) + _anti_ring((1, 2), reg_sig * 1.15).to(self.dtype))
        self.b_reg2_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_a"].unsqueeze(0) + _anti_ring((1, 2), reg_sig * 1.15).to(self.dtype))

        d_gl = _anti_ring((N_BLOCKS_L27,), (0.055 if is_wakeup else 0.070) * float(sigma_scale))
        self.g_logits[2 : 2 + 2 * half].copy_(torch.clamp(gl_gpu.unsqueeze(0) + d_gl, -2.5, 2.0))


def compute_l27_multimodal_pareto_fitness(
    traj_growth_wb_g: torch.Tensor,   # [G, W * n_budgets] лог-рост с поправкой на лог-просадку (в bp)
    clean_ev_g: torch.Tensor,         # [G] средний % прироста на чистых окнах
    inv_ev_g: torch.Tensor,           # [G] средний % прироста на подводных/инвентарных окнах
    val_mse_g: torch.Tensor,          # [G] ошибка самокалибровки V_risk
    mode_mi_g: torch.Tensor,          # [G] взаимная информация маршрутизатора I(State; Mode) в битах
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    6-осевое нелинейное Парето-ранжирование Урока 27:
      1) Asymmetric Geometric Growth (Mean + CVaR25 + Q50 - DownsideSemiDev),
      2) Clean-State Alpha Growth (способность зарабатывать из кэша без пассивного 0-трейд сидения),
      3) Inventory Recovery Delta (способность разгружать инвентарь без паники и без заморозки),
      4) Tail CVaR25 (устойчивость худших 25% траекторий),
      5) Q85 Upside Capture (захват сильных движений),
      6) Регуляризатор латентной специализации: самокалибровка V_risk + взаимная информация мод I(State; Mode).
    """
    G, M = traj_growth_wb_g.shape
    k_cvar = max(1, M // 4)
    cvar25_g = torch.topk(traj_growth_wb_g, k_cvar, dim=1, largest=False).values.mean(dim=1)
    qs = torch.quantile(traj_growth_wb_g, torch.tensor([0.50, 0.85], device=traj_growth_wb_g.device), dim=1)
    q50_g, q85_g = qs[0], qs[1]
    mean_g = traj_growth_wb_g.mean(dim=1)
    down_semi_g = torch.sqrt(torch.mean(torch.clamp(-traj_growth_wb_g, min=0.0) ** 2, dim=1))

    geo_g = 0.55 * mean_g + 0.25 * cvar25_g + 0.20 * q50_g + 35.0 * clean_ev_g + 20.0 * inv_ev_g - 0.18 * down_semi_g
    crit_and_mi_g = -torch.log1p(torch.clamp(val_mse_g * 1e4, min=0.0)) + 1.25 * torch.tanh(mode_mi_g * 2.5)

    def _rank01(v: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.argsort(v)).float() / float(max(1, G - 1))

    R = torch.stack(
        [
            _rank01(geo_g),
            _rank01(clean_ev_g),
            _rank01(inv_ev_g),
            _rank01(cvar25_g),
            _rank01(q85_g),
            _rank01(crit_and_mi_g),
        ],
        dim=1,
    )
    F5 = R[:, :5]
    dom_by_j = (F5.unsqueeze(0) >= F5.unsqueeze(1)).all(dim=-1) & (F5.unsqueeze(0) > F5.unsqueeze(1) + 1e-5).any(dim=-1)
    dom_cnt = dom_by_j.float().sum(dim=1)

    pareto_front_term = -torch.log1p(dom_cnt)
    soft_min_5 = -0.25 * torch.logsumexp(-4.0 * F5, dim=1)
    mean_5 = F5.mean(dim=1)
    fitness = pareto_front_term + 1.45 * soft_min_5 + 1.15 * mean_5 + 0.16 * R[:, 5]
    return fitness, geo_g


def _save_l27_checkpoint(
    path: Path,
    base_ck: dict,
    folded_state: dict,
    gate_logit: torch.Tensor,
    it: int,
    wave: int,
    stage: str,
    release_name: str,
    metrics: dict,
) -> None:
    ck_out = dict(base_ck)
    ck_out.update({
        "lesson": 27,
        "wave": int(wave),
        "stage": str(stage),
        "arch_version": "l27_multimodal_moe_480d",
        "iteration": int(it),
        "n_slots": 24,
        "in_dim": 480,
        "k_modes": K_MODES_L27,
        "expert_out_dim": EXPERT_OUT_DIM_L27,
        "out_dim": MOE_OUT_DIM_L27,
        "router_in_dim": ROUTER_IN_DIM_L27,
        "router_h_dim": ROUTER_H_DIM_L27,
        "reg_in_dim": REG_IN_DIM_L27,
        "reg_h_dim": 48,
        "token_refill_per_sec": 0.35,
        "token_bucket_cap": TOKEN_BUCKET_CAP_L27,
        "use_cross_venue_v1": True,
        "use_bo_swarm_36d": True,
        "use_per_slot_regulator": True,
        "use_sparse_price_matched_lob": True,
        "use_endogenous_nonlinear_grid": True,
        "use_multimodal_latent_moe": True,
        "decoupled_market_backbone": True,
        "release_name": release_name,
        "gate_logit": gate_logit.detach().cpu().clone(),
        "geo_score_bp": float(metrics["geo_score_bp"]),
        "entropic_growth_bp": float(metrics["entropic_growth_bp"]),
        "cvar25_bp": float(metrics["cvar25_bp"]),
        "q15_bp": float(metrics["q15_bp"]),
        "q50_bp": float(metrics["q50_bp"]),
        "q85_bp": float(metrics["q85_bp"]),
        "full_ev_pct": float(metrics["full_ev_pct"]),
        "clean_ev_pct": float(metrics["clean_ev_pct"]),
        "inv_delta_ev_pct": float(metrics["inv_delta_ev_pct"]),
        "dd_pct": float(metrics["dd_pct"]),
        "fills": float(metrics["fills"]),
        "cnl_per_fill": float(metrics["cnl_per_fill"]),
        "eta_queue_pct": float(metrics["eta_queue_pct"]),
        "mode_probs": list(metrics.get("mode_probs", [0.25] * 4)),
        "mode_mi_bits": float(metrics.get("mode_mi_bits", 0.0)),
        "val_ev_pct": float(metrics["full_ev_pct"]),
    })
    for k in MultimodalMoEPolicyL27.PARAM_KEYS:
        ck_out[k] = folded_state[k].detach().cpu().clone()
    # Зеркало первых 44 каналов эксперта #0 в `w2/b2` для безопасной обратной совместимости с диагностикой:
    ck_out["w2"] = folded_state["w2_moe"][:, :44].detach().cpu().clone()
    ck_out["b2"] = folded_state["b2_moe"][:, :44].detach().cpu().clone()
    torch.save(ck_out, path)


def _build_basin_hop_center(
    best_state: dict,
    best_gl: torch.Tensor,
    elite_archive: list[tuple[float, dict, torch.Tensor, list[float]]],
    wave_idx: int,
    dtype: torch.dtype,
) -> tuple[dict, torch.Tensor]:
    """
    Конструирует смелый центр для новой поисковой волны (Basin-Hopping):
    берёт глобального чемпиона, делает шаг вдоль хорды элитного многообразия и добавляет
    ортогональное возмущение маршрутизатору и слабейшему эксперту, чтобы выпрыгнуть из плато!
    """
    gen = torch.Generator(device="cpu").manual_seed(20262700 + 97 * int(wave_idx))
    hop_st = {k: v.float().cpu().clone() for k, v in best_state.items()}
    hop_gl = best_gl.float().cpu().clone()

    if len(elite_archive) >= 2:
        partner_idx = min(len(elite_archive) - 1, 1 + (wave_idx % max(1, len(elite_archive) - 1)))
        _, p_st, p_gl, _ = elite_archive[partner_idx]
        alpha_extrap = 0.22 if (wave_idx % 2 == 0) else -0.18
        for k in MultimodalMoEPolicyL27.PARAM_KEYS:
            hop_st[k] = hop_st[k] + alpha_extrap * (hop_st[k] - p_st[k].float().cpu())
        hop_gl = torch.clamp(hop_gl + alpha_extrap * (hop_gl - p_gl.float().cpu()), -2.2, 1.8)

    # Определяем наименее задействованную моду у чемпиона и даём ей и маршрутизатору свежий импульс:
    mode_p = elite_archive[0][3] if elite_archive else [0.25, 0.25, 0.25, 0.25]
    weak_k = int(np.argmin(np.asarray(mode_p, dtype=np.float32)))
    s_w, e_w = weak_k * EXPERT_OUT_DIM_L27, (weak_k + 1) * EXPERT_OUT_DIM_L27

    hop_st["w_gate1"] += torch.randn(hop_st["w_gate1"].shape, generator=gen) * 0.022
    hop_st["w_gate2"] += torch.randn(hop_st["w_gate2"].shape, generator=gen) * 0.028
    hop_st["b_gate2"][0, weak_k] = float(hop_st["b_gate2"][0, weak_k].item()) + 0.22
    hop_st["w2_moe"][:, s_w:e_w] += torch.randn((256, EXPERT_OUT_DIM_L27), generator=gen) * 0.016
    hop_st["b2_moe"][:, s_w:e_w] += torch.randn((1, EXPERT_OUT_DIM_L27), generator=gen) * 0.024
    hop_st["w_reg1"] += torch.randn(hop_st["w_reg1"].shape, generator=gen) * 0.010

    hop_norm = _normalize_weights_l27(hop_st, dtype)
    return {k: v.detach().cpu().clone() for k, v in hop_norm.items()}, hop_gl


def run_lesson_27(
    max_iters: int = 1600,
    time_budget_hours: float = 8.5,
    wakeup_iters: int = 45,
    explore_max_iters: int = 140,
    explore_patience: int = 40,
    zoom1_iters: int = 34,
    zoom2_iters: int = 28,
    min_delta_bp: float = 0.18,
    n_groups: int = 1024,
    win_len: int = 2400,
    rng_seed: int = 2026270,
    smoke_test: bool = False,
) -> None:
    t_run_start = time.time()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if smoke_test:
        n_groups = 32
        win_len = 240
        max_iters = 3
        time_budget_hours = 0.05
        wakeup_iters = 1
        explore_max_iters = 1
        zoom1_iters = 1
        zoom2_iters = 1

    n_budgets = 6
    pop_size = n_groups * n_budgets

    log_msg("=" * 122)
    log_msg(
        f"🚀 УРОК 27 (MULTIMODAL K=4 LATENT MoE + FANNED ECHELONS + 26D REGULATOR + 8.5H MULTI-WAVE ZOOM/HOP): "
        f"BUDGET={time_budget_hours:.2f}h (MAX_ITERS={max_iters}) | "
        f"12 ОКОН x {pop_size} = {12 * pop_size:,} СРЕД (WIN={win_len}s)"
    )
    log_msg("=" * 122)

    raw = np.load(CACHE_PATH, allow_pickle=True)
    assert "bo_signals_36d" in raw.files, "Отсутствует `bo_signals_36d` в кэше!"
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.01

    slice_len = min(200000 if smoke_test else 1200000, n_total)
    s0 = n_total - slice_len
    ts_sub = np.array(raw["ts"][s0:], dtype=np.float64, copy=True)

    sub = {
        k: np.array(raw[k][s0:], dtype=np.float32, copy=True)
        for k in ["bid_p", "ask_p", "bid_q", "ask_q", "buy_vol", "sell_vol", "y_sweep_12s", "y_bleed_60s", "y_crash_300s", "y_exit_30s"]
    }
    sub["tick_size"] = tick_sz
    v7_raw_t = torch.from_numpy(np.array(raw["v7_signals_58d"][s0:], dtype=np.float32, copy=True)).to(device)
    bo36_t = torch.from_numpy(np.array(raw["bo_signals_36d"][s0:], dtype=np.float32, copy=True)).to(device)
    raw.close()
    del raw
    gc.collect()

    vel_cols = [0, 2, 4, 6, 8, 12, 34, 42]
    v7_vel8 = torch.zeros((len(v7_raw_t), 8), dtype=torch.float32, device=device)
    v7_vel8[5:] = torch.clamp(v7_raw_t[5:, vel_cols] - v7_raw_t[:-5, vel_cols], -3.0, 3.0)

    bp_t = torch.from_numpy(sub["bid_p"][:, :5]).to(device)
    ap_t = torch.from_numpy(sub["ask_p"][:, :5]).to(device)
    bq_t = torch.from_numpy(sub["bid_q"][:, :5]).to(device)
    aq_t = torch.from_numpy(sub["ask_q"][:, :5]).to(device)
    bv_t = torch.from_numpy(sub["buy_vol"]).to(device)
    sv_t = torch.from_numpy(sub["sell_vol"]).to(device)

    lob_vel8 = torch.zeros((len(v7_raw_t), 8), dtype=torch.float32, device=device)
    lob_vel8[5:, 0] = torch.clamp((bp_t[5:, 0] - bp_t[:-5, 0]) / (3.0 * tick_sz), -2.0, 2.0)
    lob_vel8[5:, 1] = torch.clamp((ap_t[5:, 0] - ap_t[:-5, 0]) / (3.0 * tick_sz), -2.0, 2.0)
    lob_vel8[5:, 2] = torch.clamp(torch.log1p(torch.clamp(bq_t[5:, 0], min=0.0)) - torch.log1p(torch.clamp(bq_t[:-5, 0], min=0.0)), -2.0, 2.0)
    lob_vel8[5:, 3] = torch.clamp(torch.log1p(torch.clamp(aq_t[5:, 0], min=0.0)) - torch.log1p(torch.clamp(aq_t[:-5, 0], min=0.0)), -2.0, 2.0)
    lob_vel8[:, 4] = torch.clamp((bq_t[:, 0] - aq_t[:, 0]) / (bq_t[:, 0] + aq_t[:, 0] + 10.0), -1.0, 1.0)
    b_deep = bq_t[:, 1:5].sum(dim=1)
    a_deep = aq_t[:, 1:5].sum(dim=1)
    lob_vel8[:, 5] = torch.clamp((b_deep - a_deep) / (b_deep + a_deep + 50.0), -1.0, 1.0)
    lob_vel8[:, 6] = torch.clamp(torch.log1p(torch.clamp(bv_t, min=0.0)) / 8.0, 0.0, 2.0)
    lob_vel8[:, 7] = torch.clamp(torch.log1p(torch.clamp(sv_t, min=0.0)) / 8.0, 0.0, 2.0)
    del bp_t, ap_t, bq_t, aq_t, bv_t, sv_t

    v7_all = torch.cat([v7_raw_t, v7_vel8, lob_vel8], dim=-1)
    del v7_raw_t, v7_vel8, lob_vel8

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    policy = MultimodalMoEPolicyL27(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, device=device)
    v7_all = v7_all.to(dtype=policy.dtype)
    bo36_all = bo36_t.to(dtype=policy.dtype)
    del bo36_t

    summary_path = SAVE_DIR / "lesson_27_summary.json"
    history = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() and not smoke_test else []
    seed_path = RESUME_CHAMP_PATH if (RESUME_CHAMP_PATH.exists() and not smoke_test) else (LATEST_CHAMP_PATH if (LATEST_CHAMP_PATH.exists() and not smoke_test) else RELEASE_CHAMP_PATH)
    base_ck = torch.load(seed_path, map_location="cpu", weights_only=False)
    if "backbone_state" in base_ck:
        backbone.load_state_dict(base_ck["backbone_state"], strict=False)

    bf16_runner = FusedBF16BackboneRunner(backbone, policy.dtype, device)
    lob_fu_chunks = []
    with torch.no_grad():
        for i0 in range(0, len(lob_t), 100000):
            chunk_gpu = torch.from_numpy(lob_t[i0 : i0 + 100000]).float().to(device)
            lr_ch = backbone.encode_lob_batch(chunk_gpu).to(dtype=policy.dtype)
            lob_fu_chunks.append((lr_ch @ bf16_runner.w_fu_lob + bf16_runner.b_fu).contiguous())
            del chunk_gpu, lr_ch
    del lob_t
    lob_fu_all = torch.cat(lob_fu_chunks, dim=0)
    del lob_fu_chunks

    vec_env = VectorizedFreeMultiOrderEnv(
        bids=torch.from_numpy(sub["bid_p"]).float(),
        asks=torch.from_numpy(sub["ask_p"]).float(),
        bids_q=torch.from_numpy(sub["bid_q"]).float(),
        asks_q=torch.from_numpy(sub["ask_q"]).float(),
        buy_vol=torch.from_numpy(sub["buy_vol"]).float(),
        sell_vol=torch.from_numpy(sub["sell_vol"]).float(),
        lob_tensors=torch.empty((1, 10, 4), dtype=torch.float32),
        flow_market=torch.from_numpy(flow_m).float(),
        targets=torch.from_numpy(targets_np).float(),
        num_envs=18,
        n_side_slots=12,
        tick_size=tick_sz,
        device=device,
        allow_scratch_stop_asks=True,
    )
    roll_sell_30s_all = _build_roll_sell_30s(vec_env.sell_vol)
    del sub, flow_m, targets_np
    gc.collect()
    torch.cuda.empty_cache()

    if smoke_test:
        val_s0 = len(ts_sub) // 2
        blind_s0 = len(ts_sub) - win_len - 200
        train_hi = max(win_len + 500, val_s0 - win_len - 100)
    else:
        val_s0 = int(np.searchsorted(ts_sub, VAL_START_TS, side="left"))
        blind_s0 = int(np.searchsorted(ts_sub, BLIND_START_TS, side="left"))
        embargo_gap = win_len + 600
        train_hi = max(win_len + 1000, val_s0 - embargo_gap)

    train_grid = np.arange(600, train_hi - win_len - 100, 120)
    val_grid = np.arange(val_s0, blind_s0 - win_len - 100, 360)

    hof_state, hof_g_logit = policy.load_from_checkpoint(base_ck, expert_pole_paths=L26_EXPERT_POLES)
    start_iter = (int(base_ck.get("iteration", 0)) + 1) if (int(base_ck.get("lesson", 26)) == 27 and not smoke_test) else 1
    if start_iter > 1 and history:
        history = [r for r in history if int(r.get("iteration", 0)) < start_iter]

    n_val_pts = min(24, len(val_grid))
    val_sel = np.linspace(0, len(val_grid) - 1, num=n_val_pts, dtype=int)
    rho_cycle = [0.0, 0.0, 0.0, 0.25, 0.50, 0.70]
    off_cycle = [0.0, 0.0, 0.0, -0.01, 0.035, 0.08]
    val_basket: list[tuple[str, int, float, float]] = []
    for k_v, idx_v in enumerate(val_sel):
        ws_v = int(val_grid[int(idx_v)])
        r0_v = rho_cycle[k_v % len(rho_cycle)]
        of_v = off_cycle[k_v % len(off_cycle)]
        val_basket.append((f"V{k_v:02d}", ws_v, r0_v, of_v))

    log_msg(
        f"📊 Сетка без ручных ярлыков: TRAIN_WINDOWS={len(train_grid)} (до {datetime.fromtimestamp(ts_sub[train_hi], MSK_TZ).strftime('%m-%d %H:%M')}) | "
        f"VAL_BASKET={len(val_basket)} | BLIND_SEALED={slice_len - blind_s0:,}с (100% ЗАПЕЧАТАН!)"
    )

    base_folded = policy.fold_gate_into_state(hof_state, hof_g_logit)
    t_v0 = time.time()
    init_val_res = evaluate_l27_basket(
        vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
        [base_folded] * (2 if smoke_test else 6), val_basket[:6] if smoke_test else val_basket, win_len, tick_sz, device, true_latency=True,
    )[0]
    dt_v0 = time.time() - t_v0

    best_geo_bp = float(init_val_res["geo_score_bp"])
    best_ent_bp = float(init_val_res["entropic_growth_bp"])
    best_val_state = {k: v.clone() for k, v in hof_state.items()}
    best_val_g_logit = hof_g_logit.clone()
    best_val_folded = {k: v.clone() for k, v in base_folded.items()}
    best_val_metrics = dict(init_val_res)

    # Архив разнообразных элитных вершин для ортогональных Basin-Hopping прыжков и Model Soup:
    elite_archive: list[tuple[float, dict, torch.Tensor, list[float]]] = [
        (best_geo_bp, {k: v.clone() for k, v in best_val_state.items()}, best_val_g_logit.clone(), list(init_val_res["mode_probs"]))
    ]

    log_msg(
        f"⚖️ 4-POLE WARMSTART L27 (compiled {dt_v0:.1f}с): Geo={best_geo_bp:+.2f}bp | Ent={best_ent_bp:+.2f}bp | CVaR25={init_val_res['cvar25_bp']:+.1f}bp | "
        f"CleanEV={init_val_res['clean_ev_pct']:+.4f}% InvEV={init_val_res['inv_delta_ev_pct']:+.4f}% DD={init_val_res['dd_pct']:.2f}% | "
        f"F={init_val_res['fills']:.1f} Cnl/F={init_val_res['cnl_per_fill']:.1f} η_q={init_val_res['eta_queue_pct']:.1f}% | "
        f"Modes={init_val_res['mode_probs']} MI={init_val_res['mode_mi_bits']:.3f}b"
    )
    if start_iter == 1 and not smoke_test:
        _save_l27_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, 0, 1, "wakeup", "L27_4POLE_WARMSTART_0000", init_val_res)

    budget_tiers = torch.tensor([100.0, 300.0, 625.0, 1000.0, 1500.0, 2500.0], device=device)
    env_budgets = budget_tiers.unsqueeze(0).expand(n_groups, n_budgets).reshape(pop_size)
    rng = np.random.default_rng(rng_seed)
    for _ in range(1, start_iter):
        for _k in range(16):
            _ = rng.random()

    win_difficulty = np.ones(len(train_grid), dtype=np.float64)
    W_par = 4 if smoke_test else 12
    N_par = W_par * pop_size
    S_slots = 12
    budget_16 = env_budgets.unsqueeze(0).expand(W_par, pop_size).reshape(N_par).contiguous()
    nes_w = torch.softmax(-torch.arange(min(28, n_groups), dtype=torch.float32, device=device) * 0.24, dim=0)
    mkt_idx_t = torch.tensor(MKT_IDX_L27, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_L27, dtype=torch.long, device=device)
    echelon_k01 = torch.tensor([0.0, 0.0, 0.12, 0.12, 0.26, 0.26, 0.44, 0.44, 0.68, 0.68, 1.0, 1.0], dtype=torch.float32, device=device)
    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    chunk_sz = 100
    ln2 = math.log(2.0)

    # Мультиволновой планировщик шага: Wave 1 (wakeup -> explore -> zoom1 -> zoom2), затем Wave 2..K (explore -> zoom1 -> zoom2)
    wave = int(base_ck.get("wave", 1)) if start_iter > 1 else 1
    stage = "wakeup" if (wave == 1 and start_iter <= wakeup_iters) else "explore"
    stage_it = 0
    stage_best_geo = best_geo_bp
    wave_best_geo = best_geo_bp
    wave_best_folded = {k: v.clone() for k, v in best_val_folded.items()}
    wave_best_gl = best_val_g_logit.clone()
    wave_best_metrics = dict(best_val_metrics)
    last_stage_improve_it = 0

    for it in range(start_iter, max_iters + 1):
        elapsed_h = (time.time() - t_run_start) / 3600.0
        if (not smoke_test) and elapsed_h >= time_budget_hours:
            log_msg(f"⏰ ДОСТИГНУТ ЦЕЛЕВОЙ БЮДЖЕТ ВРЕМЕНИ ({elapsed_h:.2f}h >= {time_budget_hours:.2f}h) на итерации {it - 1} — завершаем поиск!")
            break

        t0 = time.time()
        stage_it += 1

        # Вычисление текущего масштаба шага sigma_scale для активной стадии:
        if stage == "wakeup":
            prog = float(stage_it - 1) / max(1.0, float(wakeup_iters - 1))
            sigma_cur = 0.65 + 0.65 * (0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))  # смелый старт 1.30 -> 0.65!
        elif stage == "explore":
            prog = float(stage_it - 1) / max(1.0, float(explore_max_iters - 1))
            hi_s = 1.18 if wave == 1 else 1.12
            lo_s = 0.48
            sigma_cur = lo_s + (hi_s - lo_s) * (0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))
        elif stage == "zoom1":
            prog = float(stage_it - 1) / max(1.0, float(zoom1_iters - 1))
            sigma_cur = 0.25 + 0.19 * (0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))  # сужение 0.44 -> 0.25 вокруг чемпиона!
        else:  # zoom2 (ювелирный обход малой окрестности)
            prog = float(stage_it - 1) / max(1.0, float(zoom2_iters - 1))
            sigma_cur = 0.10 + 0.11 * (0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))  # микро-полировка 0.21 -> 0.10!

        policy.sample_population(
            hof_state,
            hof_g_logit,
            stage=stage,
            sigma_scale=sigma_cur,
            alt_elite_state=best_val_state,
            alt_elite_g_logit=best_val_g_logit,
        )

        probs_d = win_difficulty / max(1e-9, win_difficulty.sum())
        mix_p = 0.50 * probs_d + 0.50 * (1.0 / len(train_grid))
        mix_p = mix_p / mix_p.sum()
        sel_grid_idx = rng.choice(len(train_grid), size=W_par, replace=(len(train_grid) < W_par), p=mix_p)
        w_starts_np = train_grid[sel_grid_idx]

        rho_batch = np.zeros(W_par, dtype=np.float32)
        lookback_sec = np.zeros(W_par, dtype=np.int64)
        n_clean_w = max(1, W_par * 5 // 12)
        for wi_b in range(n_clean_w, W_par):
            rho_batch[wi_b] = float(rng.uniform(0.12, 0.80))
            lookback_sec[wi_b] = int(rng.integers(30, 540))

        with torch.no_grad():
            w1_g = policy.effective_group_w1()
            w1_mkt_g = w1_g.index_select(1, mkt_idx_t).contiguous()
            w1_ag_g = w1_g.index_select(1, ag_idx_t).contiguous()
            del w1_g

            w_starts = torch.tensor(w_starts_np, dtype=torch.long, device=device)
            t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
            t_steps_cl = torch.clamp(t_steps, max=vec_env.max_steps - 1)

            flow_w_all = vec_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
            lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()
            v7_w_all = v7_all[t_steps_cl[: win_len - 1]].contiguous()
            bo36_w_all = bo36_all[t_steps_cl[: win_len - 1]].contiguous()
            z_mkt_raw = precompute_window_market_z370(flow_w_all, lob_fu_w_all, v7_w_all, bo36_w_all, bf16_runner)
            z_mkt_all, _ = normalize_market_z370_per_block(z_mkt_raw)
            del flow_w_all, lob_fu_w_all, v7_w_all, bo36_w_all, z_mkt_raw

            b10_w_all = vec_env.bids[t_steps_cl].contiguous()
            a10_w_all = vec_env.asks[t_steps_cl].contiguous()
            bq10_w_all = vec_env.bids_q[t_steps_cl].contiguous()
            aq10_w_all = vec_env.asks_q[t_steps_cl].contiguous()
            b1_w_all = b10_w_all[:, :, 0].contiguous()
            a1_w_all = a10_w_all[:, :, 0].contiguous()
            b1q_w_all = bq10_w_all[:, :, 0].contiguous()
            a1q_w_all = aq10_w_all[:, :, 0].contiguous()
            rb_w_all = vec_env.roll_buy_30s[t_steps_cl].contiguous()
            rs_w_all = roll_sell_30s_all[t_steps_cl].contiguous()
            sv_w_all = vec_env.sell_vol[t_steps_cl].contiguous()
            bv_w_all = vec_env.buy_vol[t_steps_cl].contiguous()

            mem_h = torch.zeros((N_par, 16), dtype=policy.dtype, device=device)
            cash = budget_16.clone()
            pos_qty = torch.zeros(N_par, dtype=torch.float32, device=device)
            pos_cost_usd = torch.zeros(N_par, dtype=torch.float32, device=device)
            avg_entry = torch.zeros(N_par, dtype=torch.float32, device=device)
            min_buy_px = torch.zeros(N_par, dtype=torch.float32, device=device)
            low_tranche_qty = torch.zeros(N_par, dtype=torch.float32, device=device)
            mid_buy_px = torch.zeros(N_par, dtype=torch.float32, device=device)
            mid_tranche_qty = torch.zeros(N_par, dtype=torch.float32, device=device)
            entry_s = torch.zeros(N_par, dtype=torch.int16, device=device)

            for wi_b in range(n_clean_w, W_par):
                s0_w, s1_w = wi_b * pop_size, (wi_b + 1) * pop_size
                ws_cur = int(w_starts_np[wi_b])
                past_idx = max(0, ws_cur - int(lookback_sec[wi_b]))
                b1_now = float(max(tick_sz, float(vec_env.bids[ws_cur, 0].item())))
                hist_epx = float(max(tick_sz, float(vec_env.asks[past_idx, 0].item())))
                bud_w = budget_16[s0_w:s1_w]
                iqty = torch.floor((bud_w * float(rho_batch[wi_b]) / b1_now) * 100.0) / 100.0
                vm = iqty >= 2.0
                rcash = torch.clamp(bud_w - iqty * b1_now * (1.0 - vec_env.maker_fee), min=0.0)
                cash[s0_w:s1_w] = torch.where(vm, rcash, bud_w)
                pos_qty[s0_w:s1_w] = torch.where(vm, iqty, torch.zeros_like(iqty))
                avg_entry[s0_w:s1_w] = torch.where(vm, torch.full_like(iqty, hist_epx), torch.zeros_like(iqty))
                pos_cost_usd[s0_w:s1_w] = torch.where(vm, iqty * hist_epx * (1.0 + vec_env.maker_fee), torch.zeros_like(iqty))
                min_buy_px[s0_w:s1_w] = torch.where(vm, torch.full_like(iqty, hist_epx), torch.zeros_like(iqty))
                low_tranche_qty[s0_w:s1_w] = torch.where(vm, iqty * 0.5, torch.zeros_like(iqty))
                mid_buy_px[s0_w:s1_w] = torch.where(vm, torch.full_like(iqty, hist_epx), torch.zeros_like(iqty))
                mid_tranche_qty[s0_w:s1_w] = torch.where(vm, iqty * 0.5, torch.zeros_like(iqty))
                entry_s[s0_w:s1_w] = -int(min(300, lookback_sec[wi_b]))

            init_b1_0 = torch.clamp(b1_w_all[0], min=tick_sz).unsqueeze(1).expand(W_par, pop_size).reshape(N_par)
            start_mtm_eq = torch.clamp(cash + pos_qty * init_b1_0 * (1.0 - vec_env.maker_fee), min=1.0)
            bid_px = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            bid_qty = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            bid_qa = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            bid_qa0 = torch.zeros((N_par, S_slots), dtype=policy.dtype, device=device)
            bid_act_s = torch.zeros((N_par, S_slots), dtype=torch.int16, device=device)
            ask_px = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            ask_qty = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            ask_qa = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            ask_qa0 = torch.zeros((N_par, S_slots), dtype=policy.dtype, device=device)
            ask_act_s = torch.zeros((N_par, S_slots), dtype=torch.int16, device=device)
            api_tokens = torch.full((N_par,), TOKEN_BUCKET_CAP_L27, dtype=torch.float32, device=device)

            peak_mtm_eq = start_mtm_eq.clone()
            prev_mtm_eq = start_mtm_eq.clone()
            prev_val_pred = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_log_dd_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            max_dd_pct_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_mut_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_fqp_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_val_err_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_mode_p_16 = torch.zeros((N_par, K_MODES_L27), dtype=torch.float32, device=device)
            sum_mode_ent_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            n_fills_16 = torch.zeros(N_par, dtype=torch.int16, device=device)
            n_cancels_16 = torch.zeros(N_par, dtype=torch.int16, device=device)
            n_requeues_16 = torch.zeros(N_par, dtype=torch.int16, device=device)

            for c0_t in range(0, win_len - 1, chunk_sz):
                c1_t = min(win_len - 1, c0_t + chunk_sz)
                m_len = c1_t - c0_t
                z_ch = z_mkt_all[c0_t:c1_t].reshape(m_len * W_par, 370).unsqueeze(0).expand(n_groups, -1, -1)
                pre_mkt_ch = (torch.bmm(z_ch, w1_mkt_g) + policy.b1).view(n_groups, m_len, W_par, 384).permute(1, 0, 2, 3).contiguous()
                for j_t in range(m_len):
                    t_step = c0_t + j_t
                    t_nxt = t_step + 1
                    (
                        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                        api_tokens, peak_mtm_eq, sum_log_dd_16, max_dd_pct_16, sum_mut_16,
                        sum_fqp_16, sum_val_err_16, prev_val_pred, prev_mtm_eq,
                        sum_mode_p_16, sum_mode_ent_16,
                        n_fills_16, n_cancels_16, n_requeues_16,
                    ) = fused_l27_multimodal_tick(
                        s_steps_i16[t_step], s_steps_i16[t_nxt], pre_mkt_ch[j_t],
                        b1_w_all[t_step], a1_w_all[t_step], b1q_w_all[t_step], a1q_w_all[t_step],
                        rb_w_all[t_step], rs_w_all[t_step],
                        b10_w_all[t_step], a10_w_all[t_step], bq10_w_all[t_step], aq10_w_all[t_step],
                        b1_w_all[t_nxt], a1_w_all[t_nxt], sv_w_all[t_nxt], bv_w_all[t_nxt],
                        w1_ag_g, policy.w_res1, policy.b_res1, policy.w_res2,
                        policy.w2_moe, policy.b2_moe, policy.w_gate1, policy.b_gate1, policy.w_gate2, policy.b_gate2,
                        policy.w_rec, policy.b_rec,
                        policy.w_reg1, policy.b_reg1_b, policy.b_reg1_a, policy.w_reg2, policy.b_reg2_b, policy.b_reg2_a,
                        echelon_k01, budget_16,
                        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                        api_tokens, peak_mtm_eq, sum_log_dd_16, max_dd_pct_16, sum_mut_16,
                        sum_fqp_16, sum_val_err_16, prev_val_pred, prev_mtm_eq,
                        sum_mode_p_16, sum_mode_ent_16,
                        n_fills_16, n_cancels_16, n_requeues_16,
                        tick_sz, vec_env.maker_fee, vec_env.taker_fee, True,
                    )
                del z_ch, pre_mkt_ch

            end_b1_16 = torch.clamp(b1_w_all[win_len - 1], min=tick_sz).unsqueeze(1).expand(W_par, pop_size).reshape(N_par)
            rem = pos_qty > 1e-6
            end_cash = torch.clamp(torch.where(rem, cash + pos_qty * end_b1_16 * (1.0 - vec_env.maker_fee), cash), min=1.0)
            end_dd_pct = 100.0 * torch.clamp(peak_mtm_eq - end_cash, min=0.0) / budget_16
            max_dd_pct_16 = torch.maximum(max_dd_pct_16, end_dd_pct)

            del (
                w1_mkt_g, w1_ag_g, z_mkt_all, mem_h,
                bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                b10_w_all, a10_w_all, bq10_w_all, aq10_w_all,
            )

            n_steps_f = float(win_len - 1)
            dlog_mat = torch.log(end_cash / start_mtm_eq).view(W_par, n_groups, n_budgets)
            m_log_dd = (sum_log_dd_16 / n_steps_f).view(W_par, n_groups, n_budgets)
            traj_wb_g = 10000.0 * (dlog_mat - 0.55 * m_log_dd).permute(1, 0, 2).reshape(n_groups, W_par * n_budgets)
            mtm_pct_wgb = (100.0 * (end_cash - start_mtm_eq) / budget_16).view(W_par, n_groups, n_budgets)
            clean_ev_g = mtm_pct_wgb[:n_clean_w].mean(dim=(0, 2))
            inv_ev_g = mtm_pct_wgb[n_clean_w:].mean(dim=(0, 2)) if W_par > n_clean_w else torch.zeros_like(clean_ev_g)
            val_mse_g = (sum_val_err_16 / n_steps_f).view(W_par, n_groups, n_budgets).mean(dim=(0, 2))

            mp_wgb = (sum_mode_p_16 / n_steps_f).view(W_par, n_groups, n_budgets, K_MODES_L27)
            mp_bar_g = mp_wgb.mean(dim=(0, 2))  # [G, 4]
            h_marg_g = -(mp_bar_g * torch.log(torch.clamp(mp_bar_g, min=1e-6))).sum(dim=-1)
            h_cond_g = (sum_mode_ent_16 / n_steps_f).view(W_par, n_groups, n_budgets).mean(dim=(0, 2))
            mode_mi_g = torch.clamp((h_marg_g - h_cond_g) / ln2, min=0.0)

            w_med_g = 10000.0 * dlog_mat.mean(dim=2)
            w_iqr = torch.quantile(w_med_g, 0.80, dim=1) - torch.quantile(w_med_g, 0.20, dim=1)
            w_tail = torch.clamp(-torch.quantile(w_med_g, 0.20, dim=1), min=0.0)
            w_diff_obs = torch.clamp(1.0 + 0.08 * w_iqr + 0.04 * w_tail, 0.5, 12.0).detach().cpu().numpy()
            for k_u, gi_u in enumerate(sel_grid_idx):
                win_difficulty[int(gi_u)] = 0.80 * win_difficulty[int(gi_u)] + 0.20 * float(w_diff_obs[k_u])

            pareto_fitness, _ = compute_l27_multimodal_pareto_fitness(traj_wb_g, clean_ev_g, inv_ev_g, val_mse_g, mode_mi_g)
            sorted_g = torch.argsort(pareto_fitness, descending=True)
            best_g = int(sorted_g[0].item())
            top_k = nes_w.numel()
            top_idx = sorted_g[:top_k]

            new_center = {}
            for k in policy.PARAM_KEYS:
                p_top = policy.__dict__[k][top_idx].float()
                w_shape = (-1,) + (1,) * (p_top.ndim - 1)
                new_center[k] = (p_top * nes_w.view(*w_shape)).sum(dim=0)
            new_center = _normalize_weights_l27(new_center, policy.dtype)
            new_gl = torch.clamp((policy.g_logits[top_idx].float() * nes_w.view(-1, 1)).sum(dim=0), -2.5, 2.0).detach().cpu()
            hof_state = {k: v.detach().cpu().clone() for k, v in new_center.items()}
            hof_g_logit = new_gl.clone()

            best_cand_gpu = _normalize_weights_l27({k: policy.__dict__[k][best_g] for k in policy.PARAM_KEYS}, policy.dtype)
            best_cand_cpu = {k: v.detach().cpu().clone() for k, v in best_cand_gpu.items()}
            best_cand_gl = policy.g_logits[best_g].detach().cpu().clone()

            cand_list = [
                policy.fold_gate_into_state(hof_state, hof_g_logit),
                policy.fold_gate_into_state(best_cand_cpu, best_cand_gl),
            ]
            val_pair = evaluate_l27_basket(
                vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
                cand_list if smoke_test else (cand_list + [cand_list[0]] * 4)[:6],
                val_basket[:6] if smoke_test else val_basket, win_len, tick_sz, device, true_latency=True,
            )[:2]
            win_vi = 0 if val_pair[0]["geo_score_bp"] >= val_pair[1]["geo_score_bp"] else 1
            v_res = val_pair[win_vi]
            v_geo = float(v_res["geo_score_bp"])
            win_st_raw = hof_state if win_vi == 0 else best_cand_cpu
            win_gl_raw = hof_g_logit if win_vi == 0 else best_cand_gl
            dt_it = time.time() - t0

            if v_geo > wave_best_geo + 1e-4:
                wave_best_geo = v_geo
                wave_best_folded = {k: v.clone() for k, v in cand_list[win_vi].items()}
                wave_best_gl = win_gl_raw.clone()
                wave_best_metrics = dict(v_res)

            if v_geo >= stage_best_geo + min_delta_bp:
                stage_best_geo = v_geo
                last_stage_improve_it = stage_it

            improved_str = ""
            if v_geo > best_geo_bp + 1e-4:
                best_geo_bp = v_geo
                best_ent_bp = float(v_res["entropic_growth_bp"])
                best_val_state = {k: v.clone() for k, v in win_st_raw.items()}
                best_val_g_logit = win_gl_raw.clone()
                best_val_folded = {k: v.clone() for k, v in cand_list[win_vi].items()}
                best_val_metrics = dict(v_res)
                improved_str = " 🏆 NEW_L27_CHAMP!"

                elite_archive.append((best_geo_bp, {k: v.clone() for k, v in best_val_state.items()}, best_val_g_logit.clone(), list(v_res["mode_probs"])))
                elite_archive.sort(key=lambda x: x[0], reverse=True)
                elite_archive = elite_archive[:8]

                if not smoke_test:
                    tag_c = f"L27_W{wave}_{stage.upper()}_IT{it:04d}"
                    _save_l27_checkpoint(SAVE_DIR / f"l27_champ_iter_{it:04d}.pt", base_ck, best_val_folded, best_val_g_logit, it, wave, stage, tag_c, v_res)
                    _save_l27_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, it, wave, stage, tag_c, v_res)

            elapsed_now_h = (time.time() - t_run_start) / 3600.0
            if not smoke_test:
                _save_l27_checkpoint(RESUME_CHAMP_PATH, base_ck, policy.fold_gate_into_state(hof_state, hof_g_logit), hof_g_logit, it, wave, stage, f"L27_RESUME_{it:04d}", v_res)
                rec = {
                    "iteration": it, "wave": wave, "stage": stage, "sigma_scale": round(sigma_cur, 4),
                    "elapsed_h": round(elapsed_now_h, 3), "dt_sec": round(dt_it, 2),
                    **v_res, "best_geo_score_bp": best_geo_bp,
                }
                history.append(rec)
                summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            log_msg(
                f"[L27 W{wave}:{stage.upper()} σ={sigma_cur:.2f} | it {it:04d} ({elapsed_now_h:.2f}/{time_budget_hours:.1f}h) | {dt_it:.1f}s] "
                f"Geo={v_geo:+.2f}bp (Best={best_geo_bp:+.2f}bp) Ent={v_res['entropic_growth_bp']:+.1f}bp CVaR25={v_res['cvar25_bp']:+.1f} | "
                f"Clean={v_res['clean_ev_pct']:+.4f}% Inv={v_res['inv_delta_ev_pct']:+.4f}% DD={v_res['dd_pct']:.2f}% | "
                f"F={v_res['fills']:.1f} Cnl/F={v_res['cnl_per_fill']:.1f} η_q={v_res['eta_queue_pct']:.1f}% | "
                f"M={[round(x, 2) for x in v_res['mode_probs']]} MI={v_res['mode_mi_bits']:.2f}b{improved_str}"
            )

            # Переходы Мультиволнового Конечного Автомата (Плато -> Возврат к Чемпиону с сужением шага -> Basin-Hopping!):
            if stage == "wakeup" and stage_it >= wakeup_iters:
                stage = "explore"
                stage_it = 0
                stage_best_geo = best_geo_bp
                last_stage_improve_it = 0
                hof_state = {k: v.clone() for k, v in best_val_state.items()}
                hof_g_logit = best_val_g_logit.clone()
                log_msg(f"🔥 [W{wave} -> EXPLORE] Смелая совместная адаптация ствола, маршрутизатора и 4 экспертов из лучшего чемпиона ({best_geo_bp:+.2f}bp)!")
            elif stage == "explore" and (stage_it >= explore_max_iters or (stage_it - last_stage_improve_it) >= explore_patience):
                # Проверяем на лету Soup из Top-2 элит перед входом в Zoom-In:
                if len(elite_archive) >= 2 and not smoke_test:
                    _, e1_st, e1_gl, _ = elite_archive[0]
                    _, e2_st, e2_gl, _ = elite_archive[1]
                    soup_try = _normalize_weights_l27({k: 0.60 * e1_st[k].float() + 0.40 * e2_st[k].float() for k in policy.PARAM_KEYS}, policy.dtype)
                    soup_gl = 0.60 * e1_gl + 0.40 * e2_gl
                    soup_fold = policy.fold_gate_into_state(soup_try, soup_gl)
                    s_res = evaluate_l27_basket(vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all, [soup_fold] * 6, val_basket, win_len, tick_sz, device, true_latency=True)[0]
                    if float(s_res["geo_score_bp"]) > best_geo_bp + 1e-4:
                        best_geo_bp = float(s_res["geo_score_bp"])
                        best_val_state = {k: v.detach().cpu().clone() for k, v in soup_try.items()}
                        best_val_g_logit = soup_gl.detach().cpu().clone()
                        best_val_folded = {k: v.clone() for k, v in soup_fold.items()}
                        best_val_metrics = dict(s_res)
                        _save_l27_checkpoint(SAVE_DIR / f"l27_champ_iter_{it:04d}_soup.pt", base_ck, best_val_folded, best_val_g_logit, it, wave, "soup", f"L27_W{wave}_SOUP_IT{it:04d}", s_res)
                        _save_l27_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, it, wave, "soup", f"L27_W{wave}_SOUP_IT{it:04d}", s_res)
                        log_msg(f"🍲 [W{wave} PRE-ZOOM SOUP BOOST] Элитный ансамбль побил одиночного лидера: Geo={best_geo_bp:+.2f}bp | Clean={s_res['clean_ev_pct']:+.4f}%!")

                stage = "zoom1"
                stage_it = 0
                stage_best_geo = best_geo_bp
                last_stage_improve_it = 0
                hof_state = {k: v.clone() for k, v in best_val_state.items()}
                hof_g_logit = best_val_g_logit.clone()
                log_msg(f"🔍 [W{wave} -> ZOOM_1 (σ=0.44->0.25)] Плато достигнуто! Возврат ровно к лучшему чемпиону ({best_geo_bp:+.2f}bp) и тонкий поиск в окрестности!")
            elif stage == "zoom1" and stage_it >= zoom1_iters:
                stage = "zoom2"
                stage_it = 0
                stage_best_geo = best_geo_bp
                last_stage_improve_it = 0
                hof_state = {k: v.clone() for k, v in best_val_state.items()}
                hof_g_logit = best_val_g_logit.clone()
                log_msg(f"🔬 [W{wave} -> ZOOM_2 (σ=0.21->0.10)] Второй возврат к обновлённому чемпиону ({best_geo_bp:+.2f}bp) — микро-полировка окрестности!")
            elif stage == "zoom2" and stage_it >= zoom2_iters:
                if not smoke_test:
                    _save_l27_checkpoint(SAVE_DIR / f"l27_wave_{wave:02d}_best.pt", base_ck, wave_best_folded, wave_best_gl, it, wave, "wave_best", f"L27_WAVE{wave:02d}_BEST", wave_best_metrics)
                rem_hours = time_budget_hours - elapsed_now_h
                if (not smoke_test) and rem_hours < 0.30:
                    log_msg(f"✅ Волна W{wave} завершена (осталось {rem_hours * 60:.1f} мин < 18 мин) — переходим к финальному синтезу Model Soup!")
                    break
                wave += 1
                stage = "explore"
                stage_it = 0
                stage_best_geo = best_geo_bp
                last_stage_improve_it = 0
                hof_state, hof_g_logit = _build_basin_hop_center(best_val_state, best_val_g_logit, elite_archive, wave, policy.dtype)
                hop_fold = policy.fold_gate_into_state(hof_state, hof_g_logit)
                wave_best_geo = -1e9
                wave_best_folded = hop_fold
                wave_best_gl = hof_g_logit.clone()
                log_msg(f"🚀 [НОВАЯ ВОЛНА W{wave} — BASIN-HOPPING ESCAPE] Прыжок из элитного многообразия в свежую котловину (впереди ещё {rem_hours:.2f}ч вычислений)!")

    if not smoke_test:
        champ_files = sorted(list(SAVE_DIR.glob("l27_champ_iter_*.pt")) + list(SAVE_DIR.glob("l27_wave_*_best.pt")))
        if len(champ_files) >= 2:
            scored = []
            seen_names = set()
            for cf in champ_files:
                ck_i = torch.load(cf, map_location="cpu", weights_only=False)
                rn = str(ck_i.get("release_name", cf.name))
                if rn in seen_names:
                    continue
                seen_names.add(rn)
                scored.append((float(ck_i.get("geo_score_bp", -1e9)), cf, ck_i))
            scored.sort(key=lambda x: x[0], reverse=True)
            k_top = min(3, len(scored))
            ws = [0.50, 0.30, 0.20][:k_top]
            s_w = sum(ws)
            ws = [w / s_w for w in ws]
            soup_st = {
                k: sum(ws[j] * scored[j][2][k].float() for j in range(k_top)).to(scored[0][2][k].dtype)
                for k in MultimodalMoEPolicyL27.PARAM_KEYS
            }
            soup_res = evaluate_l27_basket(
                vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
                [soup_st] * 6, val_basket, win_len, tick_sz, device, true_latency=True,
            )[0]
            log_msg(
                f"🍲 ФИНАЛЬНЫЙ L27 MULTIMODAL SOUP TOP-{k_top}: Geo={soup_res['geo_score_bp']:+.2f}bp | Ent={soup_res['entropic_growth_bp']:+.2f}bp | "
                f"CleanEV={soup_res['clean_ev_pct']:+.4f}% | InvEV={soup_res['inv_delta_ev_pct']:+.4f}% | DD={soup_res['dd_pct']:.2f}%"
            )
            _save_l27_checkpoint(SOUP_CHAMP_PATH, base_ck, soup_st, torch.zeros(N_BLOCKS_L27), max_iters, wave, "final_soup", f"L27_SOUP_TOP{k_top}_MULTIMODAL", soup_res)
    else:
        log_msg("✅ L27 MULTIMODAL SMOKE TEST PASSED SUCCESSFULLY!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=1600)
    ap.add_argument("--time-budget-hours", type=float, default=8.5)
    ap.add_argument("--wakeup-iters", type=int, default=45)
    ap.add_argument("--explore-max-iters", type=int, default=140)
    ap.add_argument("--explore-patience", type=int, default=40)
    ap.add_argument("--zoom1-iters", type=int, default=34)
    ap.add_argument("--zoom2-iters", type=int, default=28)
    ap.add_argument("--min-delta-bp", type=float, default=0.18)
    ap.add_argument("--n-groups", type=int, default=1024)
    ap.add_argument("--win-len", type=int, default=2400)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    run_lesson_27(
        max_iters=args.max_iters,
        time_budget_hours=args.time_budget_hours,
        wakeup_iters=args.wakeup_iters,
        explore_max_iters=args.explore_max_iters,
        explore_patience=args.explore_patience,
        zoom1_iters=args.zoom1_iters,
        zoom2_iters=args.zoom2_iters,
        min_delta_bp=args.min_delta_bp,
        n_groups=args.n_groups,
        win_len=args.win_len,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
