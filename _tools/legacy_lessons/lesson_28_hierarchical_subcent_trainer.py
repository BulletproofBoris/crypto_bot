#!/usr/bin/env python3
"""
_tools/lesson_28_hierarchical_subcent_trainer.py
=============================================================================
Урок 28 (Компонент 4): 8.0-часовой 3-стадийный GPU-марафон коэволюции
иерархического двухплощадочный субцентового Стратега `508D` (`398D Market + 110D Agent`,
0 каналов `90s`), `K = 4 Latent MoE` (`4 x 45D`) и `26D -> 48 -> 2D` Регулятора:
  - Физическое запечатывание `Final SubCent Blind OOS` `[3,076,379 : 3,096,179]` (`5.50h` `$0.0001`),
  - 4-фолдовое Purged CV на `16.70h` реальной субцентовой эры (`50` уровней `$0.0001`) +
    синтетический 50-уровневый `$0.0001` реплей исторических обвалов + макро-якоря `$0.01`,
  - Domain Randomization (`queue_mult ~ U[0.80, 1.35]`, `extra_lat_s in {0, 1}`) и сетка бюджетов
    `[300, 625, 835, 1250, 2500, 6000]` вокруг реального боевого субсчёта `$835`,
  - Стадия A (0.0–1.5ч): Пробуждение 40 новых рецепторов (`1s/2s` Tree + `SubCent Micro 16D`),
  - Стадия B (1.5–6.0ч): Полная мультиволновая коэволюция с ротацией фолдов и Basin-Hopping,
  - Стадия C (6.0–8.0ч): Полировка ортогональных Парето-полюсов (`Alpha`, `SubCent`, `Fortress`)
    и синтез `l28_crosswave_soup.pt` + `l28_best_val.pt`.
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
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_DIR = Path("/home/restorator/crypto_bot")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_free_multi_order_env import VectorizedFreeMultiOrderEnv
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_18_multistack_lifo_survival_trainer import FusedBF16BackboneRunner
from _tools.lesson_23_pump_capture_penny_l1_trainer import _build_roll_sell_30s
from _tools.fused_l26_endogenous_tick import AG_SUBBLOCKS_110
from _tools.build_l28_hierarchical_cache import read_l28_cache_global_slice
from _tools.fused_l28_hierarchical_subcent_tick import (
    MKT_DIM_L28,
    IN_DIM_L28,
    MKT_IDX_L28,
    AG_IDX_L28,
    MKT_SUBBLOCKS_398,
    K_MODES_L28,
    EXPERT_OUT_DIM_L28,
    MOE_OUT_DIM_L28,
    ROUTER_IN_DIM_L28,
    ROUTER_H_DIM_L28,
    REG_IN_DIM_L28,
    TOKEN_BUCKET_CAP_L28,
    warm_start_l27_w1_to_l28,
    build_ext_mkt_138_tensor,
    precompute_window_market_z398,
    normalize_market_z398_per_block,
    synthesize_subcent_50lvl_from_cent10,
    fused_l28_hierarchical_subcent_tick,
    evaluate_l28_dual_basket,
)

LOB_CACHE_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
L28_CACHE_PATH = BASE_DIR / "data/cache/l28_hierarchical_subcent_cache.npz"

L27_DIR = BASE_DIR / "models/prl/lesson_27_multimodal_moe"
L27_SEED_POLES = (
    L27_DIR / "l27_xwave_0722_0700_0171.pt",  # Pole 0: Боевой кросс-волновой чемпион L27
    L27_DIR / "l27_soup_top3.pt",             # Pole 1: Финальный Top-3 ансамбль L27
    L27_DIR / "l27_champ_iter_0722.pt",       # Pole 2: Пиковый чемпион волны 6
    L27_DIR / "l27_champ_iter_0171.pt",       # Pole 3: Ранний ортогональный чемпион волны 1
)

SAVE_DIR = BASE_DIR / "models/prl/lesson_28_hierarchical_subcent"
LATEST_CHAMP_PATH = SAVE_DIR / "lesson_28_champ_latest.pt"
BEST_VAL_PATH = SAVE_DIR / "l28_best_val.pt"
RESUME_CHAMP_PATH = SAVE_DIR / "lesson_28_champ_resume.pt"
SOUP_CHAMP_PATH = SAVE_DIR / "l28_crosswave_soup.pt"
LOG_PATH = BASE_DIR / "data/logs/lesson_28_train.log"

# СТРОГАЯ ГРАНИЦА ФИЗИЧЕСКОГО ЗАПЕЧАТЫВАНИЯ FINAL SUBCENT BLIND OOS (последние 19,800с = 5.50ч):
BLIND_OOS_DURATION_SEC = 19_800
MSK_TZ = timezone(timedelta(hours=3))

# 21 семантический блок 508D (14 рыночных [0:398] + 7 агентских [398:508]) с базовыми множителями мутации:
NEW_MKT_BLOCK_NAMES = {
    "ST_Micro_Tree_1_2s",
    "BO_Micro_Tree_1_2s",
    "SubCent_Micro_Coherence_16D",
}
SEMANTIC_BLOCKS_L28: list[tuple[str, int, int, float]] = []
for _bname, _bs, _be in MKT_SUBBLOCKS_398:
    if _bname == "Z_Backbone_GRU":
        _bmult = 0.70
    elif _bname in NEW_MKT_BLOCK_NAMES:
        _bmult = 1.65
    elif "Tree" in _bname:
        _bmult = 1.25
    else:
        _bmult = 1.05
    SEMANTIC_BLOCKS_L28.append((_bname, _bs, _be, _bmult))
for _aname, _as, _ae in AG_SUBBLOCKS_110:
    SEMANTIC_BLOCKS_L28.append((f"AG_{_aname}", MKT_DIM_L28 + _as, MKT_DIM_L28 + _ae, 1.15))

N_BLOCKS_L28 = len(SEMANTIC_BLOCKS_L28)
assert SEMANTIC_BLOCKS_L28[-1][2] == IN_DIM_L28


def log_msg(msg: str) -> None:
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts_str}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expand_block_gates_to_508(g_logits: torch.Tensor) -> torch.Tensor:
    """Мягкие гейты [0.35 .. 1.65] по 21 семантическому блоку 508D (при g_logit=0 -> ровно 1.0)."""
    gates_21 = 0.35 + 1.30 * torch.sigmoid(g_logits.float())
    parts = [gates_21[..., idx : idx + 1].expand(*gates_21.shape[:-1], e - s) for idx, (_, s, e, _) in enumerate(SEMANTIC_BLOCKS_L28)]
    return torch.cat(parts, dim=-1)


def _normalize_weights_l28(state: dict, dtype: torch.dtype) -> dict:
    out = {}
    w1_f = state["w1"].float()
    w1_parts = []
    for _, s, e, _ in SEMANTIC_BLOCKS_L28:
        wb = w1_f[..., s:e, :]
        cap_b = 6.5 * math.sqrt(float(e - s))
        nb = torch.linalg.vector_norm(wb, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        w1_parts.append(wb * torch.clamp(cap_b / nb, max=1.0))
    out["w1"] = torch.cat(w1_parts, dim=-2).to(dtype)
    out["b1"] = torch.clamp(state["b1"].float(), -2.2, 2.2).to(dtype)

    w2m_f = state["w2_moe"].float()
    w2m_parts = []
    for k_i in range(K_MODES_L28):
        s_k, e_k = k_i * EXPERT_OUT_DIM_L28, (k_i + 1) * EXPERT_OUT_DIM_L28
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


class HierarchicalSubCentPolicyL28:
    """
    Популяционная 508D политика Урока 28 (`398D Market + 110D Agent`, без 90s):
      - Ствол ResSwiGLU (256D) + GRU микропамять `mem_h` (16D),
      - Контекстный маршрутизатор (`288D -> 64 -> 4D`),
      - K = 4 латентных эксперта по 45D (`[G, 256, 180]`),
      - Модально-контекстный Послотовый Регулятор (`26D -> 48 -> 2D`).
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

        self.w1 = torch.empty((n_groups, IN_DIM_L28, 384), dtype=self.dtype, device=device)
        self.b1 = torch.empty((n_groups, 1, 384), dtype=self.dtype, device=device)
        self.w_res1 = torch.empty((n_groups, 256, 128), dtype=self.dtype, device=device)
        self.b_res1 = torch.empty((n_groups, 1, 128), dtype=self.dtype, device=device)
        self.w_res2 = torch.empty((n_groups, 64, 256), dtype=self.dtype, device=device)

        self.w2_moe = torch.empty((n_groups, 256, MOE_OUT_DIM_L28), dtype=self.dtype, device=device)
        self.b2_moe = torch.empty((n_groups, 1, MOE_OUT_DIM_L28), dtype=self.dtype, device=device)
        self.w_gate1 = torch.empty((n_groups, ROUTER_IN_DIM_L28, ROUTER_H_DIM_L28), dtype=self.dtype, device=device)
        self.b_gate1 = torch.empty((n_groups, 1, ROUTER_H_DIM_L28), dtype=self.dtype, device=device)
        self.w_gate2 = torch.empty((n_groups, ROUTER_H_DIM_L28, K_MODES_L28), dtype=self.dtype, device=device)
        self.b_gate2 = torch.empty((n_groups, 1, K_MODES_L28), dtype=self.dtype, device=device)

        self.w_rec = torch.empty((n_groups, 256, 32), dtype=self.dtype, device=device)
        self.b_rec = torch.empty((n_groups, 1, 32), dtype=self.dtype, device=device)

        self.w_reg1 = torch.empty((n_groups, REG_IN_DIM_L28, 48), dtype=self.dtype, device=device)
        self.b_reg1_b = torch.empty((n_groups, 1, 48), dtype=self.dtype, device=device)
        self.b_reg1_a = torch.empty((n_groups, 1, 48), dtype=self.dtype, device=device)
        self.w_reg2 = torch.empty((n_groups, 48, 2), dtype=self.dtype, device=device)
        self.b_reg2_b = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)
        self.b_reg2_a = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)

        self.g_logits = torch.zeros((n_groups, N_BLOCKS_L28), dtype=torch.float32, device=device)

        mut_vec = torch.ones(IN_DIM_L28, dtype=torch.float32, device=device)
        wakeup_vec = torch.full((IN_DIM_L28,), 0.25, dtype=torch.float32, device=device)
        for bname, s, e, mult in SEMANTIC_BLOCKS_L28:
            mut_vec[s:e] = mult
            if bname in NEW_MKT_BLOCK_NAMES:
                wakeup_vec[s:e] = 1.65
            elif bname == "Tree_5s_Velocity_8D":
                wakeup_vec[s:e] = 0.85
        self.mut_profile = mut_vec.view(1, IN_DIM_L28, 1).to(self.dtype)
        self.wakeup_profile = wakeup_vec.view(1, IN_DIM_L28, 1).to(self.dtype)

        half = max(1, (self.n_groups - 2) // 2)
        n_fine = max(1, int(round(0.30 * half)))
        n_bold = max(1, int(round(0.30 * half)))
        n_core = max(0, half - n_fine - n_bold)
        rings = [0.50] * n_fine + [1.00] * n_core + [1.85] * n_bold
        self.ring_mult_half = torch.tensor(rings[:half], dtype=torch.float32, device=device)

    def upgrade_or_extract_state(self, ck: dict, seed: int = 20262801) -> tuple[dict, torch.Tensor]:
        w1_in = ck["w1"].float().cpu()
        if w1_in.shape[0] == 480:
            w1_508 = warm_start_l27_w1_to_l28(w1_in, seed=seed, sigma_new=0.014)
            gl = torch.zeros(N_BLOCKS_L28, dtype=torch.float32)
        else:
            assert w1_in.shape == (IN_DIM_L28, 384), f"Unexpected w1 shape: {w1_in.shape}"
            w1_508 = w1_in
            gl_raw = ck.get("gate_logit", None)
            gl = gl_raw.float().cpu().clone() if (gl_raw is not None and gl_raw.numel() == N_BLOCKS_L28) else torch.zeros(N_BLOCKS_L28, dtype=torch.float32)

        state = {
            "w1": w1_508.to(dtype=self.dtype, device=self.device),
            "b1": ck["b1"].view(1, 384).to(dtype=self.dtype, device=self.device),
            "w_res1": ck["w_res1"].to(dtype=self.dtype, device=self.device),
            "b_res1": ck["b_res1"].view(1, 128).to(dtype=self.dtype, device=self.device),
            "w_res2": ck["w_res2"].to(dtype=self.dtype, device=self.device),
            "w2_moe": ck["w2_moe"].view(256, MOE_OUT_DIM_L28).to(dtype=self.dtype, device=self.device),
            "b2_moe": ck["b2_moe"].view(1, MOE_OUT_DIM_L28).to(dtype=self.dtype, device=self.device),
            "w_gate1": ck["w_gate1"].view(ROUTER_IN_DIM_L28, ROUTER_H_DIM_L28).to(dtype=self.dtype, device=self.device),
            "b_gate1": ck["b_gate1"].view(1, ROUTER_H_DIM_L28).to(dtype=self.dtype, device=self.device),
            "w_gate2": ck["w_gate2"].view(ROUTER_H_DIM_L28, K_MODES_L28).to(dtype=self.dtype, device=self.device),
            "b_gate2": ck["b_gate2"].view(1, K_MODES_L28).to(dtype=self.dtype, device=self.device),
            "w_rec": ck["w_rec"].view(256, 32).to(dtype=self.dtype, device=self.device),
            "b_rec": ck["b_rec"].view(1, 32).to(dtype=self.dtype, device=self.device),
            "w_reg1": ck["w_reg1"].view(REG_IN_DIM_L28, 48).to(dtype=self.dtype, device=self.device),
            "b_reg1_b": ck["b_reg1_b"].view(1, 48).to(dtype=self.dtype, device=self.device),
            "b_reg1_a": ck["b_reg1_a"].view(1, 48).to(dtype=self.dtype, device=self.device),
            "w_reg2": ck["w_reg2"].view(48, 2).to(dtype=self.dtype, device=self.device),
            "b_reg2_b": ck["b_reg2_b"].view(1, 2).to(dtype=self.dtype, device=self.device),
            "b_reg2_a": ck["b_reg2_a"].view(1, 2).to(dtype=self.dtype, device=self.device),
        }
        state = _normalize_weights_l28(state, self.dtype)
        return {k: v.detach().cpu().clone() for k, v in state.items()}, gl

    def effective_group_w1(self) -> torch.Tensor:
        g508 = expand_block_gates_to_508(self.g_logits).to(self.dtype).unsqueeze(-1)
        return self.w1 * g508

    def fold_gate_into_state(self, state: dict, g_logit: torch.Tensor) -> dict:
        out = {k: v.clone() for k, v in state.items()}
        g508 = expand_block_gates_to_508(g_logit.view(1, -1)).squeeze(0).to(out["w1"].dtype).unsqueeze(-1)
        out["w1"] = (out["w1"].to(g508.device) * g508).detach().cpu()
        return out

    def sample_population(
        self,
        center_state: dict,
        center_g_logit: torch.Tensor,
        macro_stage: str,
        wave_stage: str,
        sigma_scale: float,
        alt_elite_state: dict | None = None,
        alt_elite_g_logit: torch.Tensor | None = None,
    ) -> None:
        base_gpu = {k: v.to(device=self.device, dtype=self.dtype) for k, v in center_state.items()}
        gl_gpu = center_g_logit.to(device=self.device, dtype=torch.float32)

        alt_gpu = {k: v.to(device=self.device, dtype=self.dtype) for k, v in alt_elite_state.items()} if alt_elite_state is not None else base_gpu
        alt_gl = alt_elite_g_logit.to(device=self.device, dtype=torch.float32) if alt_elite_g_logit is not None else gl_gpu
        for k in self.PARAM_KEYS:
            self.__dict__[k][0].copy_(base_gpu[k])
            self.__dict__[k][1].copy_(alt_gpu[k])
        self.g_logits[0].copy_(gl_gpu)
        self.g_logits[1].copy_(alt_gl)

        half = (self.n_groups - 2) // 2
        is_stage_a = macro_stage == "A_RECEPTOR_AWAKEN"
        is_stage_c = macro_stage == "C_PARETO_POLISH"
        is_zoom = wave_stage in ("zoom1", "zoom2") or is_stage_c

        if is_zoom:
            ring_h = torch.clamp(self.ring_mult_half, max=1.20) * (0.72 if is_stage_c else 0.85)
        else:
            ring_h = self.ring_mult_half

        sig_trunk = (0.0052 if is_stage_a else (0.0036 if is_stage_c else 0.0058)) * float(sigma_scale)

        def _anti_ring(shape: tuple[int, ...], scale_f: float) -> torch.Tensor:
            r_view = ring_h.view((half,) + (1,) * len(shape))
            eps = torch.randn((half, *shape), device=self.device, dtype=torch.float32) * (scale_f * r_view)
            return torch.stack([eps, -eps], dim=1).reshape(2 * half, *shape)

        prof = self.wakeup_profile if is_stage_a else self.mut_profile
        dw1 = _anti_ring((IN_DIM_L28, 384), sig_trunk).to(self.dtype) * prof
        self.w1[2 : 2 + 2 * half].copy_(base_gpu["w1"].unsqueeze(0) + dw1)
        b1_mult = 0.25 if is_stage_a else 0.60
        self.b1[2 : 2 + 2 * half].copy_(base_gpu["b1"].unsqueeze(0) + _anti_ring((1, 384), sig_trunk * b1_mult).to(self.dtype))

        res_mult = 0.25 if is_stage_a else 1.0
        self.w_res1[2 : 2 + 2 * half].copy_(base_gpu["w_res1"].unsqueeze(0) + _anti_ring((256, 128), sig_trunk * res_mult).to(self.dtype))
        self.b_res1[2 : 2 + 2 * half].copy_(base_gpu["b_res1"].unsqueeze(0) + _anti_ring((1, 128), sig_trunk * 0.5 * res_mult).to(self.dtype))
        self.w_res2[2 : 2 + 2 * half].copy_(base_gpu["w_res2"].unsqueeze(0) + _anti_ring((64, 256), sig_trunk * res_mult).to(self.dtype))

        moe_sig = (0.0065 if is_stage_a else (0.0060 if is_stage_c else 0.0092)) * float(sigma_scale)
        dw2m = _anti_ring((256, MOE_OUT_DIM_L28), moe_sig)
        db2m = _anti_ring((1, MOE_OUT_DIM_L28), moe_sig * 1.30)
        for k_i in range(K_MODES_L28):
            s_g = k_i * EXPERT_OUT_DIM_L28 + 34
            e_g = (k_i + 1) * EXPERT_OUT_DIM_L28
            dw2m[:, :, s_g:e_g] *= 1.55
            db2m[:, :, s_g:e_g] *= 1.80
        self.w2_moe[2 : 2 + 2 * half].copy_(base_gpu["w2_moe"].unsqueeze(0) + dw2m.to(self.dtype))
        self.b2_moe[2 : 2 + 2 * half].copy_(base_gpu["b2_moe"].unsqueeze(0) + db2m.to(self.dtype))

        gate_sig = (0.022 if is_stage_a else (0.015 if not is_zoom else 0.0085)) * float(sigma_scale)
        self.w_gate1[2 : 2 + 2 * half].copy_(base_gpu["w_gate1"].unsqueeze(0) + _anti_ring((ROUTER_IN_DIM_L28, ROUTER_H_DIM_L28), gate_sig).to(self.dtype))
        self.b_gate1[2 : 2 + 2 * half].copy_(base_gpu["b_gate1"].unsqueeze(0) + _anti_ring((1, ROUTER_H_DIM_L28), gate_sig * 0.85).to(self.dtype))
        self.w_gate2[2 : 2 + 2 * half].copy_(base_gpu["w_gate2"].unsqueeze(0) + _anti_ring((ROUTER_H_DIM_L28, K_MODES_L28), gate_sig * 1.15).to(self.dtype))
        self.b_gate2[2 : 2 + 2 * half].copy_(base_gpu["b_gate2"].unsqueeze(0) + _anti_ring((1, K_MODES_L28), gate_sig * 1.35).to(self.dtype))

        rec_mult = 0.40 if is_stage_a else 1.15
        self.w_rec[2 : 2 + 2 * half].copy_(base_gpu["w_rec"].unsqueeze(0) + _anti_ring((256, 32), sig_trunk * rec_mult).to(self.dtype))
        self.b_rec[2 : 2 + 2 * half].copy_(base_gpu["b_rec"].unsqueeze(0) + _anti_ring((1, 32), sig_trunk * 0.8 * rec_mult).to(self.dtype))

        reg_sig = (0.0055 if is_stage_a else (0.0075 if is_stage_c else 0.0115)) * float(sigma_scale)
        self.w_reg1[2 : 2 + 2 * half].copy_(base_gpu["w_reg1"].unsqueeze(0) + _anti_ring((REG_IN_DIM_L28, 48), reg_sig).to(self.dtype))
        self.b_reg1_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_b"].unsqueeze(0) + _anti_ring((1, 48), reg_sig * 0.85).to(self.dtype))
        self.b_reg1_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_a"].unsqueeze(0) + _anti_ring((1, 48), reg_sig * 0.85).to(self.dtype))
        self.w_reg2[2 : 2 + 2 * half].copy_(base_gpu["w_reg2"].unsqueeze(0) + _anti_ring((48, 2), reg_sig * 0.90).to(self.dtype))
        self.b_reg2_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_b"].unsqueeze(0) + _anti_ring((1, 2), reg_sig * 1.15).to(self.dtype))
        self.b_reg2_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_a"].unsqueeze(0) + _anti_ring((1, 2), reg_sig * 1.15).to(self.dtype))

        d_gl = _anti_ring((N_BLOCKS_L28,), (0.050 if is_stage_a else 0.065) * float(sigma_scale))
        self.g_logits[2 : 2 + 2 * half].copy_(torch.clamp(gl_gpu.unsqueeze(0) + d_gl, -2.5, 2.0))


def compute_l28_pareto_fitness(
    traj_growth_wb_g: torch.Tensor,
    clean_ev_g: torch.Tensor,
    inv_ev_g: torch.Tensor,
    subcent_ev_g: torch.Tensor,
    cnl_per_fill_g: torch.Tensor,
    val_mse_g: torch.Tensor,
    mode_mi_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    6-осевое Парето-ранжирование Урока 28:
      1) Asymmetric Geometric Growth (с мягким штрафом за churn Cnl/F > 6.0),
      2) Real + Synthetic SubCent $0.0001 EV,
      3) Clean-State Alpha Growth,
      4) Inventory / Crisis Recovery Delta,
      5) Tail CVaR25,
      6) Регуляризатор V_risk + взаимная информация маршрутизатора I(State; Mode).
    """
    G, M = traj_growth_wb_g.shape
    k_cvar = max(1, M // 4)
    cvar25_g = torch.topk(traj_growth_wb_g, k_cvar, dim=1, largest=False).values.mean(dim=1)
    q50_g = torch.quantile(traj_growth_wb_g, 0.50, dim=1)
    mean_g = traj_growth_wb_g.mean(dim=1)
    down_semi_g = torch.sqrt(torch.mean(torch.clamp(-traj_growth_wb_g, min=0.0) ** 2, dim=1))
    churn_pen_g = torch.clamp(cnl_per_fill_g - 6.0, min=0.0) * 0.55

    geo_g = (
        0.50 * mean_g
        + 0.25 * cvar25_g
        + 0.20 * q50_g
        + 30.0 * clean_ev_g
        + 18.0 * inv_ev_g
        + 28.0 * subcent_ev_g
        - 0.18 * down_semi_g
        - churn_pen_g
    )
    crit_and_mi_g = -torch.log1p(torch.clamp(val_mse_g * 1e4, min=0.0)) + 1.25 * torch.tanh(mode_mi_g * 2.5)

    def _rank01(v: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.argsort(v)).float() / float(max(1, G - 1))

    R = torch.stack(
        [
            _rank01(geo_g),
            _rank01(subcent_ev_g),
            _rank01(clean_ev_g),
            _rank01(inv_ev_g),
            _rank01(cvar25_g),
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


def _save_l28_checkpoint(
    path: Path,
    base_ck: dict,
    folded_state: dict,
    gate_logit: torch.Tensor,
    it: int,
    wave: int,
    macro_stage: str,
    wave_stage: str,
    release_name: str,
    metrics: dict,
) -> None:
    ck_out = dict(base_ck)
    ck_out.update({
        "lesson": 28,
        "wave": int(wave),
        "macro_stage": str(macro_stage),
        "stage": str(wave_stage),
        "arch_version": "l28_hierarchical_subcent_508d_no90s",
        "iteration": int(it),
        "n_slots": 24,
        "in_dim": IN_DIM_L28,
        "mkt_dim": MKT_DIM_L28,
        "ag_dim": 110,
        "horizons": [1, 2, 5, 12, 25, 60, 120],
        "excluded_horizons": [90],
        "k_modes": K_MODES_L28,
        "expert_out_dim": EXPERT_OUT_DIM_L28,
        "out_dim": MOE_OUT_DIM_L28,
        "router_in_dim": ROUTER_IN_DIM_L28,
        "router_h_dim": ROUTER_H_DIM_L28,
        "reg_in_dim": REG_IN_DIM_L28,
        "reg_h_dim": 48,
        "token_refill_per_sec": 0.35,
        "token_bucket_cap": TOKEN_BUCKET_CAP_L28,
        "use_cross_venue_v1": True,
        "use_hierarchical_tree_7h": True,
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
        "subcent_ev_pct": float(metrics.get("subcent_ev_pct", metrics["clean_ev_pct"])),
        "crisis_ev_pct": float(metrics.get("crisis_ev_pct", metrics["inv_delta_ev_pct"])),
        "dd_pct": float(metrics["dd_pct"]),
        "fills": float(metrics["fills"]),
        "cnl_per_fill": float(metrics["cnl_per_fill"]),
        "eta_queue_pct": float(metrics["eta_queue_pct"]),
        "mode_probs": list(metrics.get("mode_probs", [0.25] * 4)),
        "mode_mi_bits": float(metrics.get("mode_mi_bits", 0.0)),
        "val_ev_pct": float(metrics["full_ev_pct"]),
    })
    for k in HierarchicalSubCentPolicyL28.PARAM_KEYS:
        ck_out[k] = folded_state[k].detach().cpu().clone()
    ck_out["w2"] = folded_state["w2_moe"][:, :44].detach().cpu().clone()
    ck_out["b2"] = folded_state["b2_moe"][:, :44].detach().cpu().clone()
    torch.save(ck_out, path)


def _build_l28_basin_hop_center(
    best_state: dict,
    best_gl: torch.Tensor,
    elite_archive: list[tuple[float, dict, torch.Tensor, dict]],
    wave_idx: int,
    dtype: torch.dtype,
) -> tuple[dict, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(20262800 + 101 * int(wave_idx))
    hop_st = {k: v.float().cpu().clone() for k, v in best_state.items()}
    hop_gl = best_gl.float().cpu().clone()

    if len(elite_archive) >= 2:
        partner_idx = min(len(elite_archive) - 1, 1 + (wave_idx % max(1, len(elite_archive) - 1)))
        _, p_st, p_gl, _ = elite_archive[partner_idx]
        alpha_extrap = 0.20 if (wave_idx % 2 == 0) else -0.16
        for k in HierarchicalSubCentPolicyL28.PARAM_KEYS:
            hop_st[k] = hop_st[k] + alpha_extrap * (hop_st[k] - p_st[k].float().cpu())
        hop_gl = torch.clamp(hop_gl + alpha_extrap * (hop_gl - p_gl.float().cpu()), -2.2, 1.8)

    mode_p = elite_archive[0][3].get("mode_probs", [0.25, 0.25, 0.25, 0.25]) if elite_archive else [0.25] * 4
    weak_k = int(np.argmin(np.asarray(mode_p, dtype=np.float32)))
    s_w, e_w = weak_k * EXPERT_OUT_DIM_L28, (weak_k + 1) * EXPERT_OUT_DIM_L28

    hop_st["w_gate1"] += torch.randn(hop_st["w_gate1"].shape, generator=gen) * 0.020
    hop_st["w_gate2"] += torch.randn(hop_st["w_gate2"].shape, generator=gen) * 0.025
    hop_st["b_gate2"][0, weak_k] = float(hop_st["b_gate2"][0, weak_k].item()) + 0.20
    hop_st["w2_moe"][:, s_w:e_w] += torch.randn((256, EXPERT_OUT_DIM_L28), generator=gen) * 0.014
    hop_st["b2_moe"][:, s_w:e_w] += torch.randn((1, EXPERT_OUT_DIM_L28), generator=gen) * 0.020
    # Дополнительный импульс на 40 новых рецепторах 1s/2s/SubCent:
    for bname, bs, be, _ in SEMANTIC_BLOCKS_L28:
        if bname in NEW_MKT_BLOCK_NAMES:
            hop_st["w1"][bs:be, :] += torch.randn((be - bs, 384), generator=gen) * 0.012

    hop_norm = _normalize_weights_l28(hop_st, dtype)
    return {k: v.detach().cpu().clone() for k, v in hop_norm.items()}, hop_gl


@torch.no_grad()
def _run_l28_pop_monolithic(
    vec_env: VectorizedFreeMultiOrderEnv,
    bf16_runner: FusedBF16BackboneRunner,
    lob_fu_all: torch.Tensor,
    ext138_all: torch.Tensor,
    roll_sell_30s_all: torch.Tensor,
    subcent_books: dict[str, torch.Tensor] | None,
    subcent_local_start: int,
    policy: HierarchicalSubCentPolicyL28,
    w1_mkt_g: torch.Tensor,
    w1_ag_g: torch.Tensor,
    starts_real_sub: np.ndarray,
    starts_synth_sub: np.ndarray,
    starts_hist_cent: np.ndarray,
    rho_batch: np.ndarray,
    lookback_sec: np.ndarray,
    win_len: int,
    feat_tick: float,
    synth_seed: int,
    budget_tiers: torch.Tensor,
    echelon_k01: torch.Tensor,
    s_steps_i16: torch.Tensor,
    queue_mult: float,
    extra_lat_s: int,
    chunk_sz: int = 120,
) -> dict[str, torch.Tensor]:
    device = policy.device
    W_real = len(starts_real_sub)
    W_synth = len(starts_synth_sub)
    W_cent = len(starts_hist_cent)
    W_b = W_real + W_synth + W_cent
    pop_size = policy.pop_size
    N_b = W_b * pop_size
    S_slots = 12

    w_starts_np = np.concatenate([starts_real_sub, starts_synth_sub, starts_hist_cent], axis=0)
    et_list = ([0.0001] * (W_real + W_synth)) + ([float(feat_tick)] * W_cent)
    et_1w = torch.tensor(et_list, dtype=torch.float32, device=device)

    env_budgets = budget_tiers.unsqueeze(0).expand(policy.n_groups, policy.n_budgets).reshape(pop_size)
    budget_b = env_budgets.unsqueeze(0).expand(W_b, pop_size).reshape(N_b).contiguous()

    w_starts = torch.tensor(w_starts_np, dtype=torch.long, device=device)
    t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
    t_steps_cl = torch.clamp(t_steps, max=vec_env.max_steps - 1)

    flow_w_all = vec_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
    lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()
    ext_w_all = ext138_all[t_steps_cl[: win_len - 1]].contiguous()
    z_mkt_raw, adv_risk_w_all = precompute_window_market_z398(flow_w_all, lob_fu_w_all, ext_w_all, bf16_runner)
    z_mkt_all, _ = normalize_market_z398_per_block(z_mkt_raw)
    del flow_w_all, lob_fu_w_all, ext_w_all, z_mkt_raw

    b1q_cent_w_all = vec_env.bids_q[t_steps_cl][:, :, 0].contiguous()
    a1q_cent_w_all = vec_env.asks_q[t_steps_cl][:, :, 0].contiguous()

    bL_parts, aL_parts, bqL_parts, aqL_parts = [], [], [], []
    if W_real > 0:
        assert subcent_books is not None
        t_real = t_steps_cl[:, :W_real]
        sub_idx = torch.clamp(t_real - int(subcent_local_start), min=0, max=subcent_books["bid_p50"].shape[0] - 1)
        bL_parts.append(subcent_books["bid_p50"][sub_idx])
        aL_parts.append(subcent_books["ask_p50"][sub_idx])
        bqL_parts.append(subcent_books["bid_q50"][sub_idx])
        aqL_parts.append(subcent_books["ask_q50"][sub_idx])
    if W_synth > 0:
        t_syn = t_steps_cl[:, W_real : W_real + W_synth]
        b_syn, a_syn, bq_syn, aq_syn = synthesize_subcent_50lvl_from_cent10(
            vec_env.bids[t_syn],
            vec_env.asks[t_syn],
            vec_env.bids_q[t_syn],
            vec_env.asks_q[t_syn],
            seed=synth_seed,
        )
        bL_parts.append(b_syn)
        aL_parts.append(a_syn)
        bqL_parts.append(bq_syn)
        aqL_parts.append(aq_syn)
    if W_cent > 0:
        t_cnt = t_steps_cl[:, W_real + W_synth :]
        L_tgt = 50 if (W_real + W_synth) > 0 else vec_env.bids.shape[-1]
        pad_r = L_tgt - vec_env.bids.shape[-1]
        if pad_r > 0:
            bL_parts.append(F.pad(vec_env.bids[t_cnt], (0, pad_r), value=0.0))
            aL_parts.append(F.pad(vec_env.asks[t_cnt], (0, pad_r), value=0.0))
            bqL_parts.append(F.pad(vec_env.bids_q[t_cnt], (0, pad_r), value=0.0))
            aqL_parts.append(F.pad(vec_env.asks_q[t_cnt], (0, pad_r), value=0.0))
        else:
            bL_parts.append(vec_env.bids[t_cnt])
            aL_parts.append(vec_env.asks[t_cnt])
            bqL_parts.append(vec_env.bids_q[t_cnt])
            aqL_parts.append(vec_env.asks_q[t_cnt])

    bL_w_all = torch.cat(bL_parts, dim=1).contiguous()
    aL_w_all = torch.cat(aL_parts, dim=1).contiguous()
    bqL_w_all = torch.cat(bqL_parts, dim=1).contiguous()
    aqL_w_all = torch.cat(aqL_parts, dim=1).contiguous()
    del bL_parts, aL_parts, bqL_parts, aqL_parts

    b1_w_all = bL_w_all[:, :, 0].contiguous()
    a1_w_all = aL_w_all[:, :, 0].contiguous()
    rb_w_all = vec_env.roll_buy_30s[t_steps_cl].contiguous()
    rs_w_all = roll_sell_30s_all[t_steps_cl].contiguous()
    sv_w_all = vec_env.sell_vol[t_steps_cl].contiguous()
    bv_w_all = vec_env.buy_vol[t_steps_cl].contiguous()

    mem_h = torch.zeros((N_b, 16), dtype=policy.dtype, device=device)
    cash = budget_b.clone()
    pos_qty = torch.zeros(N_b, dtype=torch.float32, device=device)
    pos_cost_usd = torch.zeros(N_b, dtype=torch.float32, device=device)
    avg_entry = torch.zeros(N_b, dtype=torch.float32, device=device)
    min_buy_px = torch.zeros(N_b, dtype=torch.float32, device=device)
    low_tranche_qty = torch.zeros(N_b, dtype=torch.float32, device=device)
    mid_buy_px = torch.zeros(N_b, dtype=torch.float32, device=device)
    mid_tranche_qty = torch.zeros(N_b, dtype=torch.float32, device=device)
    entry_s = torch.zeros(N_b, dtype=torch.int16, device=device)

    rho_w = torch.tensor(rho_batch, dtype=torch.float32, device=device)
    lb_w = torch.tensor(lookback_sec, dtype=torch.long, device=device)
    has_inv_w = rho_w > 0.01
    if bool(has_inv_w.any()):
        past_idx_w = torch.clamp(w_starts - lb_w, min=0, max=vec_env.max_steps - 1)
        b1_now_w = torch.maximum(et_1w, b1_w_all[0])
        hist_ask_w = vec_env.asks[past_idx_w, 0]
        hist_epx_w = torch.maximum(et_1w, torch.round(hist_ask_w / et_1w) * et_1w)

        bud_we = budget_b.view(W_b, pop_size)
        iqty_we = torch.floor((bud_we * rho_w.unsqueeze(1) / b1_now_w.unsqueeze(1)) * 100.0) / 100.0
        vm_we = has_inv_w.unsqueeze(1) & (iqty_we >= 2.0)
        rcash_we = torch.clamp(bud_we - iqty_we * b1_now_w.unsqueeze(1) * (1.0 - vec_env.maker_fee), min=0.0)
        epx_we = hist_epx_w.unsqueeze(1).expand(W_b, pop_size)

        cash = torch.where(vm_we, rcash_we, bud_we).reshape(N_b)
        pos_qty = torch.where(vm_we, iqty_we, torch.zeros_like(iqty_we)).reshape(N_b)
        avg_entry = torch.where(vm_we, epx_we, torch.zeros_like(epx_we)).reshape(N_b)
        pos_cost_usd = torch.where(vm_we, iqty_we * epx_we * (1.0 + vec_env.maker_fee), torch.zeros_like(iqty_we)).reshape(N_b)
        min_buy_px = avg_entry.clone()
        low_tranche_qty = (pos_qty * 0.5).clone()
        mid_buy_px = avg_entry.clone()
        mid_tranche_qty = (pos_qty * 0.5).clone()

        ent_w = (-torch.clamp(lb_w, max=300)).to(torch.int16).unsqueeze(1).expand(W_b, pop_size)
        entry_s = torch.where(has_inv_w.unsqueeze(1).expand(W_b, pop_size), ent_w, entry_s.view(W_b, pop_size)).reshape(N_b)

    init_b1_0 = torch.maximum(b1_w_all[0], et_1w).unsqueeze(1).expand(W_b, pop_size).reshape(N_b)
    start_mtm_eq = torch.clamp(cash + pos_qty * init_b1_0 * (1.0 - vec_env.maker_fee), min=1.0)
    bid_px = torch.zeros((N_b, S_slots), dtype=torch.float32, device=device)
    bid_qty = torch.zeros((N_b, S_slots), dtype=torch.float32, device=device)
    bid_qa = torch.zeros((N_b, S_slots), dtype=torch.float32, device=device)
    bid_qa0 = torch.zeros((N_b, S_slots), dtype=policy.dtype, device=device)
    bid_act_s = torch.zeros((N_b, S_slots), dtype=torch.int16, device=device)
    ask_px = torch.zeros((N_b, S_slots), dtype=torch.float32, device=device)
    ask_qty = torch.zeros((N_b, S_slots), dtype=torch.float32, device=device)
    ask_qa = torch.zeros((N_b, S_slots), dtype=torch.float32, device=device)
    ask_qa0 = torch.zeros((N_b, S_slots), dtype=policy.dtype, device=device)
    ask_act_s = torch.zeros((N_b, S_slots), dtype=torch.int16, device=device)
    api_tokens = torch.full((N_b,), TOKEN_BUCKET_CAP_L28, dtype=torch.float32, device=device)

    peak_mtm_eq = start_mtm_eq.clone()
    prev_mtm_eq = start_mtm_eq.clone()
    prev_val_pred = torch.zeros(N_b, dtype=torch.float32, device=device)
    sum_log_dd = torch.zeros(N_b, dtype=torch.float32, device=device)
    max_dd_pct = torch.zeros(N_b, dtype=torch.float32, device=device)
    sum_mut = torch.zeros(N_b, dtype=torch.float32, device=device)
    sum_fqp = torch.zeros(N_b, dtype=torch.float32, device=device)
    sum_val_err = torch.zeros(N_b, dtype=torch.float32, device=device)
    sum_mode_p = torch.zeros((N_b, K_MODES_L28), dtype=torch.float32, device=device)
    sum_mode_ent = torch.zeros(N_b, dtype=torch.float32, device=device)
    n_fills = torch.zeros(N_b, dtype=torch.int16, device=device)
    n_cancels = torch.zeros(N_b, dtype=torch.int16, device=device)
    n_requeues = torch.zeros(N_b, dtype=torch.int16, device=device)

    qm_t = torch.tensor(float(queue_mult), dtype=torch.float32, device=device)
    el_t = torch.tensor(int(extra_lat_s), dtype=torch.int16, device=device)

    for c0_t in range(0, win_len - 1, chunk_sz):
        c1_t = min(win_len - 1, c0_t + chunk_sz)
        m_len = c1_t - c0_t
        z_ch = z_mkt_all[c0_t:c1_t].reshape(m_len * W_b, MKT_DIM_L28).unsqueeze(0).expand(policy.n_groups, -1, -1)
        pre_mkt_ch = (torch.bmm(z_ch, w1_mkt_g) + policy.b1).view(policy.n_groups, m_len, W_b, 384).permute(1, 0, 2, 3).contiguous()
        for j_t in range(m_len):
            t_step = c0_t + j_t
            t_nxt = t_step + 1
            (
                mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
                sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
                sum_mode_p, sum_mode_ent,
                n_fills, n_cancels, n_requeues,
            ) = fused_l28_hierarchical_subcent_tick(
                s_steps_i16[t_step], s_steps_i16[t_nxt], pre_mkt_ch[j_t], adv_risk_w_all[t_step],
                b1_w_all[t_step], a1_w_all[t_step], b1q_cent_w_all[t_step], a1q_cent_w_all[t_step],
                rb_w_all[t_step], rs_w_all[t_step],
                bL_w_all[t_step], aL_w_all[t_step], bqL_w_all[t_step], aqL_w_all[t_step],
                b1_w_all[t_nxt], a1_w_all[t_nxt], sv_w_all[t_nxt], bv_w_all[t_nxt],
                w1_ag_g, policy.w_res1, policy.b_res1, policy.w_res2,
                policy.w2_moe, policy.b2_moe, policy.w_gate1, policy.b_gate1, policy.w_gate2, policy.b_gate2,
                policy.w_rec, policy.b_rec,
                policy.w_reg1, policy.b_reg1_b, policy.b_reg1_a, policy.w_reg2, policy.b_reg2_b, policy.b_reg2_a,
                echelon_k01, budget_b,
                mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
                sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
                sum_mode_p, sum_mode_ent,
                n_fills, n_cancels, n_requeues,
                et_1w, feat_tick, vec_env.maker_fee, vec_env.taker_fee,
                qm_t, el_t, True,
            )
        del z_ch, pre_mkt_ch

    end_b1 = torch.maximum(b1_w_all[win_len - 1], et_1w).unsqueeze(1).expand(W_b, pop_size).reshape(N_b)
    rem = pos_qty > 1e-6
    end_cash = torch.clamp(torch.where(rem, cash + pos_qty * end_b1 * (1.0 - vec_env.maker_fee), cash), min=1.0)
    n_steps_f = float(win_len - 1)
    dlog_mat = torch.log(end_cash / start_mtm_eq).view(W_b, policy.n_groups, policy.n_budgets)
    m_log_dd = (sum_log_dd / n_steps_f).view(W_b, policy.n_groups, policy.n_budgets)
    mtm_pct_wgb = (100.0 * (end_cash - start_mtm_eq) / budget_b).view(W_b, policy.n_groups, policy.n_budgets)

    return {
        "dlog_mat": dlog_mat,
        "traj_wgb": 10000.0 * (dlog_mat - 0.55 * m_log_dd),
        "mtm_pct_wgb": mtm_pct_wgb,
        "val_mse_wgb": (sum_val_err / n_steps_f).view(W_b, policy.n_groups, policy.n_budgets),
        "mode_p_wgb": (sum_mode_p / n_steps_f).view(W_b, policy.n_groups, policy.n_budgets, K_MODES_L28),
        "mode_ent_wgb": (sum_mode_ent / n_steps_f).view(W_b, policy.n_groups, policy.n_budgets),
        "fills_wgb": n_fills.float().view(W_b, policy.n_groups, policy.n_budgets),
        "cancels_wgb": n_cancels.float().view(W_b, policy.n_groups, policy.n_budgets),
    }


def run_lesson_28(
    max_iters: int = 2000,
    target_hours: float = 8.0,
    explore_max_iters: int = 110,
    explore_patience: int = 34,
    zoom1_iters: int = 28,
    zoom2_iters: int = 22,
    min_delta_bp: float = 0.16,
    n_groups: int = 1024,
    win_len: int = 2400,
    rng_seed: int = 2026280,
    smoke_test: bool = False,
) -> None:
    t_run_start = time.time()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if smoke_test:
        n_groups = 32
        win_len = 240
        max_iters = 3
        target_hours = 0.05
        explore_max_iters = 1
        zoom1_iters = 1
        zoom2_iters = 1

    n_budgets = 6
    pop_size = n_groups * n_budgets
    budget_tiers = torch.tensor([300.0, 625.0, 835.0, 1250.0, 2500.0, 6000.0], dtype=torch.float32, device=device)

    log_msg("=" * 124)
    log_msg(
        f"🚀 УРОК 28 (508D HIERARCHICAL TREE 7H [БЕЗ 90s] + 50-LVL $0.0001 SPARSE-LOB + 4-FOLD PURGED CV): "
        f"BUDGET={target_hours:.2f}h | POP={pop_size:,} (TIERS=$300..$6000, LIVE=$835) | WIN={win_len}s"
    )
    log_msg("=" * 124)

    assert L28_CACHE_PATH.exists(), f"Не найден кэш Урока 28: {L28_CACHE_PATH}"
    raw_lob = np.load(LOB_CACHE_PATH, allow_pickle=True)
    raw_l28 = np.load(L28_CACHE_PATH, allow_pickle=True)

    n_total_all = len(raw_lob["bid_p"])
    l28_total = int(raw_l28["total_rows"]) if "total_rows" in raw_l28.files else len(raw_l28["ts"])
    assert l28_total == n_total_all, f"Несовпадение длины кэшей: {l28_total} vs {n_total_all}"
    subcent_global_start = int(raw_l28["subcent_era_start"])
    feat_tick = float(raw_lob["tick_size"]) if "tick_size" in raw_lob.files else 0.01

    # ФИЗИЧЕСКОЕ ЗАПЕЧАТЫВАНИЕ FINAL SUBCENT BLIND OOS [n_total_all - 19,800 : n_total_all]!
    # Эти 19,800 секунд (5.50 часов) самого свежего реального $0.0001 рынка вообще не загружаются в память тренера!
    blind_start_global = n_total_all - BLIND_OOS_DURATION_SEC
    assert blind_start_global > subcent_global_start + 4 * win_len, "Недостаточно субцентовых данных до Blind OOS!"

    slice_span = min(220000 if smoke_test else 1250000, blind_start_global)
    s0 = blind_start_global - slice_span
    s1 = blind_start_global  # НЕ ВКЛЮЧАЯ BLIND OOS!
    subcent_local_start = subcent_global_start - s0
    subcent_trainval_len = s1 - subcent_global_start

    ts_sub = np.array(raw_lob["ts"][s0:s1], dtype=np.float64, copy=True)
    sub = {
        k: np.array(raw_lob[k][s0:s1], dtype=np.float32, copy=True)
        for k in ["bid_p", "ask_p", "bid_q", "ask_q", "buy_vol", "sell_vol", "y_sweep_12s", "y_bleed_60s", "y_crash_300s", "y_exit_30s"]
    }
    sub["tick_size"] = feat_tick

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ SIM-TO-LIVE: на субцентовом участке подставляем агрегированные $0.01 корзины
    # в `sub["bid_p..ask_q"]` для `MultiTaskLOBNet` и `lob_vel8` (1-в-1 с `live_rl_trader.py:2000-2014`!),
    # а 50-уровневый сырой стакан `$0.0001` держим в `subcent_books` для `Sparse-LOB` и Taker-пробоя!
    sub["bid_p"][subcent_local_start:] = np.asarray(raw_l28["cent_bid_p10"][:subcent_trainval_len], dtype=np.float32)
    sub["bid_q"][subcent_local_start:] = np.asarray(raw_l28["cent_bid_q10"][:subcent_trainval_len], dtype=np.float32)
    sub["ask_p"][subcent_local_start:] = np.asarray(raw_l28["cent_ask_p10"][:subcent_trainval_len], dtype=np.float32)
    sub["ask_q"][subcent_local_start:] = np.asarray(raw_l28["cent_ask_q10"][:subcent_trainval_len], dtype=np.float32)

    subcent_books = {
        "bid_p50": torch.from_numpy(np.array(raw_l28["subcent_bid_p50"][:subcent_trainval_len], dtype=np.float32, copy=True)).to(device),
        "bid_q50": torch.from_numpy(np.array(raw_l28["subcent_bid_q50"][:subcent_trainval_len], dtype=np.float32, copy=True)).to(device),
        "ask_p50": torch.from_numpy(np.array(raw_l28["subcent_ask_p50"][:subcent_trainval_len], dtype=np.float32, copy=True)).to(device),
        "ask_q50": torch.from_numpy(np.array(raw_l28["subcent_ask_q50"][:subcent_trainval_len], dtype=np.float32, copy=True)).to(device),
    }

    v7_legacy_t = torch.from_numpy(np.array(raw_lob["v7_signals_58d"][s0:s1], dtype=np.float32, copy=True)).to(device)
    st_tree_t = torch.from_numpy(read_l28_cache_global_slice(raw_l28, "st_tree_42d", s0, s1)).to(device)
    bo_tree_t = torch.from_numpy(read_l28_cache_global_slice(raw_l28, "bo_tree_42d", s0, s1)).to(device)
    sub_micro16_t = torch.from_numpy(read_l28_cache_global_slice(raw_l28, "subcent_micro_16d", s0, s1)).to(device)
    raw_lob.close()
    raw_l28.close()
    del raw_lob, raw_l28
    gc.collect()

    bp_t = torch.from_numpy(sub["bid_p"][:, :5]).to(device)
    ap_t = torch.from_numpy(sub["ask_p"][:, :5]).to(device)
    bq_t = torch.from_numpy(sub["bid_q"][:, :5]).to(device)
    aq_t = torch.from_numpy(sub["ask_q"][:, :5]).to(device)
    bv_t = torch.from_numpy(sub["buy_vol"]).to(device)
    sv_t = torch.from_numpy(sub["sell_vol"]).to(device)

    lob_vel8 = torch.zeros((len(v7_legacy_t), 8), dtype=torch.float32, device=device)
    lob_vel8[5:, 0] = torch.clamp((bp_t[5:, 0] - bp_t[:-5, 0]) / (3.0 * feat_tick), -2.0, 2.0)
    lob_vel8[5:, 1] = torch.clamp((ap_t[5:, 0] - ap_t[:-5, 0]) / (3.0 * feat_tick), -2.0, 2.0)
    lob_vel8[5:, 2] = torch.clamp(torch.log1p(torch.clamp(bq_t[5:, 0], min=0.0)) - torch.log1p(torch.clamp(bq_t[:-5, 0], min=0.0)), -2.0, 2.0)
    lob_vel8[5:, 3] = torch.clamp(torch.log1p(torch.clamp(aq_t[5:, 0], min=0.0)) - torch.log1p(torch.clamp(aq_t[:-5, 0], min=0.0)), -2.0, 2.0)
    lob_vel8[:, 4] = torch.clamp((bq_t[:, 0] - aq_t[:, 0]) / (bq_t[:, 0] + aq_t[:, 0] + 10.0), -1.0, 1.0)
    b_deep = bq_t[:, 1:5].sum(dim=1)
    a_deep = aq_t[:, 1:5].sum(dim=1)
    lob_vel8[:, 5] = torch.clamp((b_deep - a_deep) / (b_deep + a_deep + 50.0), -1.0, 1.0)
    lob_vel8[:, 6] = torch.clamp(torch.log1p(torch.clamp(bv_t, min=0.0)) / 8.0, 0.0, 2.0)
    lob_vel8[:, 7] = torch.clamp(torch.log1p(torch.clamp(sv_t, min=0.0)) / 8.0, 0.0, 2.0)
    del bp_t, ap_t, bq_t, aq_t, bv_t, sv_t

    ext138_f32 = build_ext_mkt_138_tensor(st_tree_t, bo_tree_t, v7_legacy_t, lob_vel8, sub_micro16_t)
    del st_tree_t, bo_tree_t, v7_legacy_t, lob_vel8, sub_micro16_t

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    policy = HierarchicalSubCentPolicyL28(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, device=device)
    ext138_all = ext138_f32.to(dtype=policy.dtype)
    del ext138_f32

    summary_path = SAVE_DIR / "lesson_28_summary.json"
    history = json.loads(summary_path.read_text(encoding="utf-8")) if (summary_path.exists() and not smoke_test) else []

    if BEST_VAL_PATH.exists() and not smoke_test:
        seed_path = BEST_VAL_PATH
    elif RESUME_CHAMP_PATH.exists() and not smoke_test:
        seed_path = RESUME_CHAMP_PATH
    else:
        seed_path = next((p for p in L27_SEED_POLES if p.exists()), BASE_DIR / "models/prl/rl_execution/release_l7_0150_champion.pt")
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

    b1_cent_np = sub["bid_p"][:, 0].copy()
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
        tick_size=feat_tick,
        device=device,
        allow_scratch_stop_asks=True,
    )
    roll_sell_30s_all = _build_roll_sell_30s(vec_env.sell_vol)
    del sub, flow_m, targets_np
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # 4-ФОЛДОВОЕ КАУЗАЛЬНОЕ РАЗБИЕНИЕ СУБЦЕНТОВОЙ ЭРЫ + ИСТОРИЧЕСКИЕ ЯКОРЯ КРИЗИСОВ
    # =========================================================================
    purge_sec = 60 if smoke_test else 300
    hist_end = max(win_len + 1200, subcent_local_start - win_len - purge_sec)
    hist_grid_all = np.arange(600, max(660, hist_end - win_len), 180)

    # Автоматический отбор самых волатильных исторических окон (обвалы и сквизы) для кризисного иммунитета:
    hist_scores = []
    for ws_h in hist_grid_all:
        seg_p = b1_cent_np[int(ws_h) : int(ws_h) + win_len]
        rng_rel = float((np.max(seg_p) - np.min(seg_p)) / max(0.05, float(seg_p[0])))
        hist_scores.append(rng_rel)
    hist_scores_np = np.asarray(hist_scores, dtype=np.float64)
    q70_vol = float(np.quantile(hist_scores_np, 0.70)) if len(hist_scores_np) > 0 else 0.0
    crisis_cand_idx = np.where(hist_scores_np >= q70_vol)[0]
    if len(crisis_cand_idx) < 8:
        crisis_cand_idx = np.arange(len(hist_grid_all))

    # Фиксированная историческая валидационная корзина кризисов (8 окон с purge от train):
    n_crisis_val = min(4 if smoke_test else 8, len(crisis_cand_idx))
    crs_val_sel = crisis_cand_idx[np.linspace(0, len(crisis_cand_idx) - 1, num=n_crisis_val, dtype=int)]
    crisis_val_starts = [int(hist_grid_all[int(i)]) for i in crs_val_sel]
    rho_crs = [0.0, 0.45, 0.65, 0.0, 0.55, 0.30, 0.0, 0.70]
    off_crs = [0.0, 0.04, 0.08, 0.0, 0.05, -0.01, 0.0, 0.09]
    global_crisis_val_windows = [
        (f"CRS{k:02d}", ws_c, rho_crs[k % len(rho_crs)], off_crs[k % len(off_crs)])
        for k, ws_c in enumerate(crisis_val_starts)
    ]

    # Исключаем окрестности валидационных кризисных окон из исторического train-пула:
    hist_train_mask = np.ones(len(hist_grid_all), dtype=bool)
    for ws_c in crisis_val_starts:
        hist_train_mask &= np.abs(hist_grid_all - ws_c) > (win_len + purge_sec)
    hist_train_grid = hist_grid_all[hist_train_mask]
    if len(hist_train_grid) < 8:
        hist_train_grid = hist_grid_all
    hist_crisis_train_grid = hist_train_grid[
        np.argsort(np.asarray([
            (np.max(b1_cent_np[int(w) : int(w) + win_len]) - np.min(b1_cent_np[int(w) : int(w) + win_len])) / max(0.05, float(b1_cent_np[int(w)]))
            for w in hist_train_grid
        ]))[-max(8, len(hist_train_grid) // 3):]
    ]

    # 4 каузальных фолда реальной $0.0001 эры `[subcent_local_start : slice_span]`:
    fold_edges = np.linspace(subcent_local_start, slice_span, num=5, dtype=int)
    subcent_fold_val_windows: list[list[tuple[str, int, float, float]]] = []
    rho_sub = [0.0, 0.0, 0.30, 0.55, 0.70]
    off_sub = [0.0, 0.0, -0.005, 0.015, 0.035]
    for fk in range(4):
        f_lo = int(fold_edges[fk]) + (purge_sec if fk > 0 else 0)
        f_hi = int(fold_edges[fk + 1]) - win_len - (purge_sec if fk < 3 else 30)
        if f_hi <= f_lo + 60:
            f_lo = int(fold_edges[fk])
            f_hi = max(f_lo + 60, int(fold_edges[fk + 1]) - win_len - 10)
        f_starts = np.linspace(f_lo, f_hi, num=(2 if smoke_test else 5), dtype=int)
        f_wins = [
            (f"F{fk}_{j}", int(ws_f), rho_sub[j % len(rho_sub)], off_sub[j % len(off_sub)])
            for j, ws_f in enumerate(f_starts)
        ]
        subcent_fold_val_windows.append(f_wins)

    global_subcent_val_windows = [w for fw in subcent_fold_val_windows for w in fw]
    subcent_all_grid = np.arange(subcent_local_start, slice_span - win_len - 20, 60)

    def _get_subcent_train_grid_for_fold(val_fold_k: int) -> np.ndarray:
        v_lo = int(fold_edges[val_fold_k]) - win_len - purge_sec
        v_hi = int(fold_edges[val_fold_k + 1]) + purge_sec
        keep = (subcent_all_grid < v_lo) | (subcent_all_grid > v_hi)
        g = subcent_all_grid[keep]
        return g if len(g) >= 6 else subcent_all_grid

    log_msg(
        f"🔒 ANTI-LEAK РАЗБИЕНИЕ L28: HIST_TRAIN={len(hist_train_grid)} | SUBCENT_ERA={subcent_trainval_len:,}с (4 Purged Folds, GlobalVal={len(global_subcent_val_windows)}+$0.0001 + {len(global_crisis_val_windows)} Crisis) | "
        f"SEALED FINAL BLIND OOS=[{blind_start_global:,}:{n_total_all:,}] ({n_total_all - blind_start_global:,}с = {(n_total_all - blind_start_global)/3600.0:.2f}ч — 0% ДОСТУПА!)"
    )

    # =========================================================================
    # ХИРУРГИЧЕСКИЙ 4-ПОЛЮСНЫЙ WARM-START 480D -> 508D ИЛИ БЕСШОВНОЕ ПРОДОЛЖЕНИЕ L28
    # =========================================================================
    pole_states: list[tuple[dict, torch.Tensor]] = []
    is_l28_continuation = (int(base_ck.get("lesson", 0)) == 28 and not smoke_test)
    prev_max_iter = int(base_ck.get("iteration", 0)) if is_l28_continuation else 0
    prev_max_wave = int(base_ck.get("wave", 1)) if is_l28_continuation else 1

    if is_l28_continuation:
        cand_paths = []
        for p_fixed in [SOUP_CHAMP_PATH, BEST_VAL_PATH, LATEST_CHAMP_PATH, RESUME_CHAMP_PATH]:
            if p_fixed.exists():
                cand_paths.append(p_fixed)
        cand_paths.extend(sorted(SAVE_DIR.glob("l28_champ_iter_*.pt")))
        cand_paths.extend(sorted(SAVE_DIR.glob("l28_wave_*_best.pt")))
        cand_paths.extend(sorted(SAVE_DIR.glob("l28_wave_*_specialist.pt")))

        uniq_cks = []
        seen_tags = set()
        for cp in cand_paths:
            ck_c = torch.load(cp, map_location="cpu", weights_only=False)
            prev_max_iter = max(prev_max_iter, int(ck_c.get("iteration", 0)))
            prev_max_wave = max(prev_max_wave, int(ck_c.get("wave", 1)))
            rn = str(ck_c.get("release_name", cp.name))
            if rn in seen_tags:
                continue
            seen_tags.add(rn)
            uniq_cks.append((float(ck_c.get("geo_score_bp", -1e9)), ck_c))
        if history:
            prev_max_iter = max(prev_max_iter, max(int(r.get("iteration", 0)) for r in history))
            prev_max_wave = max(prev_max_wave, max(int(r.get("wave", 1)) for r in history))
        uniq_cks.sort(key=lambda x: x[0], reverse=True)
        top_cks = [x[1] for x in uniq_cks[:8]]
        if len(top_cks) % 2 == 1 and len(top_cks) > 1:
            top_cks = top_cks[:-1]  # Чётное число полюсов для прогрева C=2!
        for idx_c, ck_c in enumerate(top_cks):
            pole_states.append(policy.upgrade_or_extract_state(ck_c, seed=20262801 + idx_c))
        if not pole_states:
            pole_states.append(policy.upgrade_or_extract_state(base_ck, seed=20262801))
    else:
        for p_idx, p_path in enumerate(L27_SEED_POLES):
            if p_path.exists():
                ck_p = torch.load(p_path, map_location="cpu", weights_only=False)
                pole_states.append(policy.upgrade_or_extract_state(ck_p, seed=20262801 + p_idx))
        if not pole_states:
            pole_states.append(policy.upgrade_or_extract_state(base_ck, seed=20262801))

    pole_folded_list = [policy.fold_gate_into_state(st_p, gl_p) for st_p, gl_p in pole_states]
    t_v0 = time.time()
    if is_l28_continuation and len(pole_folded_list) >= 2:
        pole_evals = []
        for i_p in range(0, len(pole_folded_list), 2):
            pole_evals.extend(
                evaluate_l28_dual_basket(
                    vec_env, bf16_runner, lob_fu_all, ext138_all, roll_sell_30s_all,
                    subcent_books, subcent_local_start,
                    pole_folded_list[i_p : i_p + 2],
                    global_subcent_val_windows,
                    global_crisis_val_windows,
                    win_len, feat_tick, device, budget_tiers=budget_tiers, true_latency=True,
                )
            )
    else:
        pole_evals = evaluate_l28_dual_basket(
            vec_env, bf16_runner, lob_fu_all, ext138_all, roll_sell_30s_all,
            subcent_books, subcent_local_start,
            pole_folded_list[:2] if smoke_test else pole_folded_list,
            global_subcent_val_windows[:4] if smoke_test else global_subcent_val_windows,
            global_crisis_val_windows[:2] if smoke_test else global_crisis_val_windows,
            win_len, feat_tick, device, budget_tiers=budget_tiers, true_latency=True,
        )
    dt_v0 = time.time() - t_v0

    best_pole_i = int(np.argmax([r["geo_score_bp"] for r in pole_evals]))
    hof_state = {k: v.clone() for k, v in pole_states[best_pole_i][0].items()}
    hof_g_logit = pole_states[best_pole_i][1].clone()
    init_val_res = pole_evals[best_pole_i]

    best_geo_bp = float(init_val_res["geo_score_bp"])
    best_sub_ev = float(init_val_res["subcent_ev_pct"])
    best_crs_ev = float(init_val_res["crisis_ev_pct"])
    best_val_state = {k: v.clone() for k, v in hof_state.items()}
    best_val_g_logit = hof_g_logit.clone()
    best_val_folded = {k: v.clone() for k, v in pole_folded_list[best_pole_i].items()}
    best_val_metrics = dict(init_val_res)

    # Ортогональные Парето-специалисты для Финальной Стадии C и Cross-Wave Soup:
    specialist_subcent = (best_sub_ev, {k: v.clone() for k, v in best_val_state.items()}, best_val_g_logit.clone(), dict(init_val_res))
    specialist_fortress = (best_crs_ev - 0.15 * float(init_val_res["dd_pct"]), {k: v.clone() for k, v in best_val_state.items()}, best_val_g_logit.clone(), dict(init_val_res))

    elite_archive: list[tuple[float, dict, torch.Tensor, dict]] = []
    for idx_e, res_e in enumerate(pole_evals):
        st_e_c = {k: v.clone() for k, v in pole_states[idx_e][0].items()}
        gl_e_c = pole_states[idx_e][1].clone()
        elite_archive.append((float(res_e["geo_score_bp"]), st_e_c, gl_e_c, dict(res_e)))
        if float(res_e["subcent_ev_pct"]) > specialist_subcent[0] + 1e-4 and float(res_e["dd_pct"]) <= best_val_metrics["dd_pct"] + 1.0:
            specialist_subcent = (float(res_e["subcent_ev_pct"]), {k: v.clone() for k, v in st_e_c.items()}, gl_e_c.clone(), dict(res_e))
        f_sc_e = float(res_e["crisis_ev_pct"]) - 0.15 * float(res_e["dd_pct"])
        if f_sc_e > specialist_fortress[0] + 1e-4 and float(res_e["subcent_ev_pct"]) >= -0.02:
            specialist_fortress = (f_sc_e, {k: v.clone() for k, v in st_e_c.items()}, gl_e_c.clone(), dict(res_e))
    elite_archive.sort(key=lambda x: x[0], reverse=True)

    log_msg(
        f"⚖️ {'SEAMLESS L28 CONTINUATION' if is_l28_continuation else 'SURGICAL 508D WARM-START L28'} ({len(pole_evals)} элит, compiled {dt_v0:.1f}с, BestPole=#{best_pole_i}): "
        f"Geo={best_geo_bp:+.2f}bp | SubCentEV={init_val_res['subcent_ev_pct']:+.4f}% | CrisisEV={init_val_res['crisis_ev_pct']:+.4f}% | "
        f"Clean={init_val_res['clean_ev_pct']:+.4f}% Inv={init_val_res['inv_delta_ev_pct']:+.4f}% DD={init_val_res['dd_pct']:.2f}% | "
        f"F={init_val_res['fills']:.1f} Cnl/F={init_val_res['cnl_per_fill']:.1f} η_q={init_val_res['eta_queue_pct']:.1f}%"
    )

    start_iter = (prev_max_iter + 1) if is_l28_continuation else 1
    if start_iter == 1 and not smoke_test:
        _save_l28_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, 0, 1, "A_RECEPTOR_AWAKEN", "warmup", "L28_WARMSTART_0000", init_val_res)
        _save_l28_checkpoint(BEST_VAL_PATH, base_ck, best_val_folded, best_val_g_logit, 0, 1, "A_RECEPTOR_AWAKEN", "warmup", "L28_WARMSTART_0000", init_val_res)

    rng = np.random.default_rng(rng_seed)
    for _ in range(1, start_iter):
        for _k in range(16):
            _ = rng.random()

    W_real_sub = 2 if smoke_test else 8
    W_synth_sub = 1 if smoke_test else 4
    W_hist_cent = 1 if smoke_test else 4
    W_tot = W_real_sub + W_synth_sub + W_hist_cent

    nes_w = torch.softmax(-torch.arange(min(28, n_groups), dtype=torch.float32, device=device) * 0.24, dim=0)
    mkt_idx_t = torch.tensor(MKT_IDX_L28, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_L28, dtype=torch.long, device=device)
    echelon_k01 = torch.tensor([0.0, 0.0, 0.12, 0.12, 0.26, 0.26, 0.44, 0.44, 0.68, 0.68, 1.0, 1.0], dtype=torch.float32, device=device)
    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    ln2 = math.log(2.0)

    if is_l28_continuation:
        stage_a_hours = 0.0                          # Рецепторы уже пробуждены, сразу идём в глубокие волны B -> финальную полировку C!
        stage_b_end_h = target_hours * (4.5 / 6.0)   # 4.5h волн коэволюции + 1.5h финальной полировки при +6.0h
    else:
        stage_a_hours = target_hours * (1.5 / 8.0)   # 1.5h при 8.0h
        stage_b_end_h = target_hours * (6.0 / 8.0)   # 6.0h при 8.0h

    wave = (prev_max_wave + 1) if is_l28_continuation else 1
    wave_stage = "explore"
    stage_it = 0
    stage_best_geo = -1e9
    wave_best_geo = -1e9
    wave_best_folded = {k: v.clone() for k, v in best_val_folded.items()}
    wave_best_gl = best_val_g_logit.clone()
    wave_best_metrics = dict(best_val_metrics)
    last_stage_improve_it = 0
    prev_macro_stage = ""

    for it in range(start_iter, max_iters + 1):
        elapsed_h = (time.time() - t_run_start) / 3600.0
        if (not smoke_test) and elapsed_h >= target_hours:
            log_msg(f"⏰ ДОСТИГНУТ ЦЕЛЕВОЙ 8-ЧАСОВОЙ БЮДЖЕТ ({elapsed_h:.2f}h >= {target_hours:.2f}h) на итерации {it - 1} — переходим к финальному синтезу Cross-Wave Soup!")
            break

        t0 = time.time()
        stage_it += 1

        if elapsed_h < stage_a_hours and not smoke_test:
            macro_stage = "A_RECEPTOR_AWAKEN"
        elif elapsed_h < stage_b_end_h or smoke_test:
            macro_stage = "B_FULL_COEVOLVE"
        else:
            macro_stage = "C_PARETO_POLISH"

        if macro_stage != prev_macro_stage:
            if prev_macro_stage != "":
                hof_state = {k: v.clone() for k, v in best_val_state.items()}
                hof_g_logit = best_val_g_logit.clone()
                wave_stage = "explore" if macro_stage != "C_PARETO_POLISH" else "zoom1"
                stage_it = 1
                stage_best_geo = -1e9
                last_stage_improve_it = 0
                log_msg(f"🌟 [ПЕРЕХОД МАКРО-ФАЗЫ -> {macro_stage} ({elapsed_h:.2f}h/{target_hours:.1f}h)] Возврат к глобальному чемпиону ({best_geo_bp:+.2f}bp)!")
            prev_macro_stage = macro_stage

        if macro_stage == "A_RECEPTOR_AWAKEN":
            prog = min(1.0, elapsed_h / max(1e-4, stage_a_hours))
            sigma_cur = 0.60 + 0.65 * (0.5 * (1.0 + math.cos(math.pi * prog)))
        elif macro_stage == "C_PARETO_POLISH":
            prog = min(1.0, (elapsed_h - stage_b_end_h) / max(1e-4, target_hours - stage_b_end_h))
            sigma_cur = 0.12 + 0.30 * (0.5 * (1.0 + math.cos(math.pi * prog)))
        else:
            if wave_stage == "explore":
                prog = float(stage_it - 1) / max(1.0, float(explore_max_iters - 1))
                sigma_cur = 0.46 + 0.68 * (0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))
            elif wave_stage == "zoom1":
                prog = float(stage_it - 1) / max(1.0, float(zoom1_iters - 1))
                sigma_cur = 0.24 + 0.18 * (0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))
            else:
                prog = float(stage_it - 1) / max(1.0, float(zoom2_iters - 1))
                sigma_cur = 0.10 + 0.11 * (0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

        policy.sample_population(
            hof_state,
            hof_g_logit,
            macro_stage=macro_stage,
            wave_stage=wave_stage,
            sigma_scale=sigma_cur,
            alt_elite_state=best_val_state,
            alt_elite_g_logit=best_val_g_logit,
        )

        # Ротация валидационного субцентового фолда по волнам:
        active_val_fold = (wave - 1) % 4
        sub_train_grid = _get_subcent_train_grid_for_fold(active_val_fold)

        starts_real_sub = rng.choice(sub_train_grid, size=W_real_sub, replace=(len(sub_train_grid) < W_real_sub))
        starts_synth_sub = rng.choice(hist_crisis_train_grid, size=W_synth_sub, replace=(len(hist_crisis_train_grid) < W_synth_sub))
        starts_hist_cent = rng.choice(hist_train_grid, size=W_hist_cent, replace=(len(hist_train_grid) < W_hist_cent))

        def _make_rho_lb(n_w: int, n_clean: int) -> tuple[np.ndarray, np.ndarray]:
            r_arr = np.zeros(n_w, dtype=np.float32)
            lb_arr = np.zeros(n_w, dtype=np.int64)
            for wi in range(n_clean, n_w):
                r_arr[wi] = float(rng.uniform(0.12, 0.80))
                lb_arr[wi] = int(rng.integers(30, 540))
            return r_arr, lb_arr

        rho_rsub, lb_rsub = _make_rho_lb(W_real_sub, max(1, W_real_sub // 2))
        rho_ssub, lb_ssub = _make_rho_lb(W_synth_sub, max(0, W_synth_sub - 1))
        rho_hcnt, lb_hcnt = _make_rho_lb(W_hist_cent, max(1, W_hist_cent // 2))
        rho_all = np.concatenate([rho_rsub, rho_ssub, rho_hcnt], axis=0)
        lb_all = np.concatenate([lb_rsub, lb_ssub, lb_hcnt], axis=0)

        # Domain Randomization очередей и сетевого джиттера в тренировочных роллаутах:
        q_mult_it = float(rng.uniform(0.80, 1.35))
        ex_lat_it = 1 if float(rng.random()) < 0.35 else 0

        with torch.no_grad():
            w1_g = policy.effective_group_w1()
            w1_mkt_g = w1_g.index_select(1, mkt_idx_t).contiguous()
            w1_ag_g = w1_g.index_select(1, ag_idx_t).contiguous()
            del w1_g

            mono_out = _run_l28_pop_monolithic(
                vec_env, bf16_runner, lob_fu_all, ext138_all, roll_sell_30s_all,
                subcent_books, subcent_local_start, policy, w1_mkt_g, w1_ag_g,
                starts_real_sub, starts_synth_sub, starts_hist_cent,
                rho_all, lb_all, win_len, feat_tick, rng_seed + it * 13,
                budget_tiers, echelon_k01, s_steps_i16, q_mult_it, ex_lat_it,
            )
            del w1_mkt_g, w1_ag_g

            traj_wgb = mono_out["traj_wgb"]
            mtm_wgb = mono_out["mtm_pct_wgb"]
            vmse_wgb = mono_out["val_mse_wgb"]
            mp_wgb = mono_out["mode_p_wgb"]
            ment_wgb = mono_out["mode_ent_wgb"]
            fills_wgb = mono_out["fills_wgb"]
            cnls_wgb = mono_out["cancels_wgb"]

            cl_m = [i for i, rv in enumerate(rho_all) if rv <= 0.01]
            iv_m = [i for i, rv in enumerate(rho_all) if rv > 0.01]

            traj_wb_g = traj_wgb.permute(1, 0, 2).reshape(n_groups, W_tot * n_budgets)
            clean_ev_g = mtm_wgb[cl_m].mean(dim=(0, 2)) if cl_m else mtm_wgb.mean(dim=(0, 2))
            inv_ev_g = mtm_wgb[iv_m].mean(dim=(0, 2)) if iv_m else torch.zeros_like(clean_ev_g)
            subcent_ev_g = mtm_wgb[: W_real_sub + W_synth_sub].mean(dim=(0, 2))
            val_mse_g = vmse_wgb.mean(dim=(0, 2))

            tot_f_g = fills_wgb.sum(dim=0).mean(dim=1)
            tot_c_g = cnls_wgb.sum(dim=0).mean(dim=1)
            cnl_pf_g = tot_c_g / torch.clamp(tot_f_g, min=0.5)

            mp_bar_g = mp_wgb.mean(dim=(0, 2))
            h_marg_g = -(mp_bar_g * torch.log(torch.clamp(mp_bar_g, min=1e-6))).sum(dim=-1)
            h_cond_g = ment_wgb.mean(dim=(0, 2))
            mode_mi_g = torch.clamp((h_marg_g - h_cond_g) / ln2, min=0.0)

            pareto_fitness, _ = compute_l28_pareto_fitness(
                traj_wb_g, clean_ev_g, inv_ev_g, subcent_ev_g, cnl_pf_g, val_mse_g, mode_mi_g
            )
            sorted_g = torch.argsort(pareto_fitness, descending=True)
            best_g = int(sorted_g[0].item())
            top_idx = sorted_g[: nes_w.numel()]

            new_center = {}
            for k in policy.PARAM_KEYS:
                p_top = policy.__dict__[k][top_idx].float()
                w_shape = (-1,) + (1,) * (p_top.ndim - 1)
                new_center[k] = (p_top * nes_w.view(*w_shape)).sum(dim=0)
            new_center = _normalize_weights_l28(new_center, policy.dtype)
            new_gl = torch.clamp((policy.g_logits[top_idx].float() * nes_w.view(-1, 1)).sum(dim=0), -2.5, 2.0).detach().cpu()
            hof_state = {k: v.detach().cpu().clone() for k, v in new_center.items()}
            hof_g_logit = new_gl.clone()

            run_fast_val = (wave_stage != "explore") or (it % 2 == 0) or (stage_it == 1) or smoke_test
            improved_str = ""
            if run_fast_val:
                best_cand_gpu = _normalize_weights_l28({k: policy.__dict__[k][best_g] for k in policy.PARAM_KEYS}, policy.dtype)
                best_cand_cpu = {k: v.detach().cpu().clone() for k, v in best_cand_gpu.items()}
                best_cand_gl = policy.g_logits[best_g].detach().cpu().clone()

                cand_pair = [
                    policy.fold_gate_into_state(hof_state, hof_g_logit),
                    policy.fold_gate_into_state(best_cand_cpu, best_cand_gl),
                ]
                # Быстрая монолитная волновая валидация на отложенном фолде `active_val_fold` + 4 кризисных якорях:
                fast_sub_w = subcent_fold_val_windows[active_val_fold]
                fast_crs_w = global_crisis_val_windows[: (2 if smoke_test else 4)]
                fast_pair = evaluate_l28_dual_basket(
                    vec_env, bf16_runner, lob_fu_all, ext138_all, roll_sell_30s_all,
                    subcent_books, subcent_local_start, cand_pair,
                    fast_sub_w, fast_crs_w, win_len, feat_tick, device, budget_tiers=budget_tiers, true_latency=True,
                )
                win_vi = 0 if fast_pair[0]["geo_score_bp"] >= fast_pair[1]["geo_score_bp"] else 1
                v_res = fast_pair[win_vi]
                v_geo_fast = float(v_res["geo_score_bp"])
                win_st_raw = hof_state if win_vi == 0 else best_cand_cpu
                win_gl_raw = hof_g_logit if win_vi == 0 else best_cand_gl
                win_folded = cand_pair[win_vi]

                if v_geo_fast >= stage_best_geo + min_delta_bp:
                    stage_best_geo = v_geo_fast
                    last_stage_improve_it = stage_it

                # Если кандидат перспективен на текущем фолде (или каждые 8 итераций), проверяем пару на ПОЛНОЙ 4-фолдовой глобальной корзине (C=2 без рекомпиляции!):
                need_global_chk = (v_geo_fast > wave_best_geo + 1e-4) or (it % 8 == 0) or smoke_test
                if need_global_chk:
                    g_pair = evaluate_l28_dual_basket(
                        vec_env, bf16_runner, lob_fu_all, ext138_all, roll_sell_30s_all,
                        subcent_books, subcent_local_start, cand_pair,
                        global_subcent_val_windows[:4] if smoke_test else global_subcent_val_windows,
                        global_crisis_val_windows[:2] if smoke_test else global_crisis_val_windows,
                        win_len, feat_tick, device, budget_tiers=budget_tiers, true_latency=True,
                    )
                    g_vi = 0 if g_pair[0]["geo_score_bp"] >= g_pair[1]["geo_score_bp"] else 1
                    g_res = g_pair[g_vi]
                    win_st_raw = hof_state if g_vi == 0 else best_cand_cpu
                    win_gl_raw = hof_g_logit if g_vi == 0 else best_cand_gl
                    win_folded = cand_pair[g_vi]
                    v_res = g_res
                    g_geo = float(g_res["geo_score_bp"])

                    if g_geo > wave_best_geo + 1e-4:
                        wave_best_geo = g_geo
                        wave_best_folded = {k: v.clone() for k, v in win_folded.items()}
                        wave_best_gl = win_gl_raw.clone()
                        wave_best_metrics = dict(g_res)

                    # Обновляем ортогональных Парето-специалистов (SubCent Sniper и Crisis Fortress):
                    if float(g_res["subcent_ev_pct"]) > specialist_subcent[0] + 1e-4 and float(g_res["dd_pct"]) <= best_val_metrics["dd_pct"] + 1.0:
                        specialist_subcent = (float(g_res["subcent_ev_pct"]), {k: v.clone() for k, v in win_st_raw.items()}, win_gl_raw.clone(), dict(g_res))
                    fort_sc = float(g_res["crisis_ev_pct"]) - 0.15 * float(g_res["dd_pct"])
                    if fort_sc > specialist_fortress[0] + 1e-4 and float(g_res["subcent_ev_pct"]) >= -0.02:
                        specialist_fortress = (fort_sc, {k: v.clone() for k, v in win_st_raw.items()}, win_gl_raw.clone(), dict(g_res))

                    if g_geo > best_geo_bp + 1e-4:
                        best_geo_bp = g_geo
                        best_val_state = {k: v.clone() for k, v in win_st_raw.items()}
                        best_val_g_logit = win_gl_raw.clone()
                        best_val_folded = {k: v.clone() for k, v in win_folded.items()}
                        best_val_metrics = dict(g_res)
                        improved_str = " 🏆 NEW_GLOBAL_4FOLD_L28_CHAMP!"

                        elite_archive.append((best_geo_bp, {k: v.clone() for k, v in best_val_state.items()}, best_val_g_logit.clone(), dict(g_res)))
                        elite_archive.sort(key=lambda x: x[0], reverse=True)
                        elite_archive = elite_archive[:10]

                        if not smoke_test:
                            tag_c = f"L28_W{wave}_{macro_stage[:1]}_{wave_stage.upper()}_IT{it:04d}"
                            _save_l28_checkpoint(SAVE_DIR / f"l28_champ_iter_{it:04d}.pt", base_ck, best_val_folded, best_val_g_logit, it, wave, macro_stage, wave_stage, tag_c, g_res)
                            _save_l28_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, it, wave, macro_stage, wave_stage, tag_c, g_res)
                            _save_l28_checkpoint(BEST_VAL_PATH, base_ck, best_val_folded, best_val_g_logit, it, wave, macro_stage, wave_stage, tag_c, g_res)
            else:
                v_res = dict(best_val_metrics)
                improved_str = f" [NES-FAST TrSubEV={float(subcent_ev_g[best_g].item()):+.3f}%]"

            dt_it = time.time() - t0
            elapsed_now_h = (time.time() - t_run_start) / 3600.0
            if not smoke_test:
                if it % 10 == 0:
                    _save_l28_checkpoint(RESUME_CHAMP_PATH, base_ck, policy.fold_gate_into_state(hof_state, hof_g_logit), hof_g_logit, it, wave, macro_stage, wave_stage, f"L28_RESUME_{it:04d}", v_res)
                rec = {
                    "iteration": it, "wave": wave, "fold": active_val_fold, "macro_stage": macro_stage, "stage": wave_stage,
                    "sigma_scale": round(sigma_cur, 4), "queue_mult": round(q_mult_it, 3), "extra_lat_s": ex_lat_it,
                    "elapsed_h": round(elapsed_now_h, 3), "dt_sec": round(dt_it, 2),
                    **v_res, "best_geo_score_bp": best_geo_bp,
                }
                history.append(rec)
                if it % 2 == 0 or bool(improved_str.startswith(" 🏆")):
                    summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            log_msg(
                f"[L28 {macro_stage[:1]}|W{wave}(F{active_val_fold}):{wave_stage.upper()} σ={sigma_cur:.2f} | it {it:04d} ({elapsed_now_h:.2f}/{target_hours:.1f}h) | {dt_it:.1f}s] "
                f"Geo={v_res['geo_score_bp']:+.2f}bp (Best={best_geo_bp:+.2f}bp) | SubEV={v_res['subcent_ev_pct']:+.4f}% CrsEV={v_res['crisis_ev_pct']:+.4f}% | "
                f"DD={v_res['dd_pct']:.2f}% F={v_res['fills']:.1f} Cnl/F={v_res['cnl_per_fill']:.1f} η_q={v_res['eta_queue_pct']:.1f}% | "
                f"M={[round(x, 2) for x in v_res['mode_probs']]} MI={v_res['mode_mi_bits']:.2f}b{improved_str}"
            )

            # Мультиволновые переходы внутри стадий B и C:
            if macro_stage == "B_FULL_COEVOLVE":
                if wave_stage == "explore" and (stage_it >= explore_max_iters or (stage_it - last_stage_improve_it) >= explore_patience):
                    if len(elite_archive) >= 2 and not smoke_test:
                        _, e1_st, e1_gl, _ = elite_archive[0]
                        _, e2_st, e2_gl, _ = elite_archive[1]
                        soup_try = _normalize_weights_l28({k: 0.60 * e1_st[k].float() + 0.40 * e2_st[k].float() for k in policy.PARAM_KEYS}, policy.dtype)
                        soup_gl = 0.60 * e1_gl + 0.40 * e2_gl
                        soup_fold = policy.fold_gate_into_state(soup_try, soup_gl)
                        s_res = evaluate_l28_dual_basket(
                            vec_env, bf16_runner, lob_fu_all, ext138_all, roll_sell_30s_all,
                            subcent_books, subcent_local_start, [soup_fold, best_val_folded],
                            global_subcent_val_windows, global_crisis_val_windows,
                            win_len, feat_tick, device, budget_tiers=budget_tiers, true_latency=True,
                        )[0]
                        if float(s_res["geo_score_bp"]) > best_geo_bp + 1e-4:
                            best_geo_bp = float(s_res["geo_score_bp"])
                            best_val_state = {k: v.detach().cpu().clone() for k, v in soup_try.items()}
                            best_val_g_logit = soup_gl.detach().cpu().clone()
                            best_val_folded = {k: v.clone() for k, v in soup_fold.items()}
                            best_val_metrics = dict(s_res)
                            _save_l28_checkpoint(SAVE_DIR / f"l28_champ_iter_{it:04d}_soup.pt", base_ck, best_val_folded, best_val_g_logit, it, wave, macro_stage, "soup", f"L28_W{wave}_SOUP_IT{it:04d}", s_res)
                            _save_l28_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, it, wave, macro_stage, "soup", f"L28_W{wave}_SOUP_IT{it:04d}", s_res)
                            _save_l28_checkpoint(BEST_VAL_PATH, base_ck, best_val_folded, best_val_g_logit, it, wave, macro_stage, "soup", f"L28_W{wave}_SOUP_IT{it:04d}", s_res)
                            log_msg(f"🍲 [W{wave} PRE-ZOOM SOUP BOOST] Новый глобальный лидер 4-Fold: Geo={best_geo_bp:+.2f}bp | SubEV={s_res['subcent_ev_pct']:+.4f}%!")

                    wave_stage = "zoom1"
                    stage_it = 0
                    stage_best_geo = -1e9
                    last_stage_improve_it = 0
                    hof_state = {k: v.clone() for k, v in best_val_state.items()}
                    hof_g_logit = best_val_g_logit.clone()
                    log_msg(f"🔍 [W{wave} -> ZOOM_1] Возврат к глобальному 4-Fold чемпиону ({best_geo_bp:+.2f}bp) и сужение колец мутации!")
                elif wave_stage == "zoom1" and stage_it >= zoom1_iters:
                    wave_stage = "zoom2"
                    stage_it = 0
                    stage_best_geo = -1e9
                    last_stage_improve_it = 0
                    hof_state = {k: v.clone() for k, v in best_val_state.items()}
                    hof_g_logit = best_val_g_logit.clone()
                    log_msg(f"🔬 [W{wave} -> ZOOM_2] Микро-полировка окрестности чемпиона ({best_geo_bp:+.2f}bp)!")
                elif wave_stage == "zoom2" and stage_it >= zoom2_iters:
                    if not smoke_test:
                        _save_l28_checkpoint(SAVE_DIR / f"l28_wave_{wave:02d}_best.pt", base_ck, wave_best_folded, wave_best_gl, it, wave, macro_stage, "wave_best", f"L28_WAVE{wave:02d}_BEST", wave_best_metrics)
                    wave += 1
                    wave_stage = "explore"
                    stage_it = 0
                    stage_best_geo = -1e9
                    last_stage_improve_it = 0
                    hof_state, hof_g_logit = _build_l28_basin_hop_center(best_val_state, best_val_g_logit, elite_archive, wave, policy.dtype)
                    wave_best_geo = -1e9
                    wave_best_folded = policy.fold_gate_into_state(hof_state, hof_g_logit)
                    wave_best_gl = hof_g_logit.clone()
                    log_msg(f"🚀 [НОВАЯ ВОЛНА W{wave} (Fold F{(wave-1)%4}) — BASIN-HOPPING] Прыжок в новую котловину!")

            elif macro_stage == "C_PARETO_POLISH":
                # В Стадии C каждые 32 итерации чередуем полировку 3 ортогональных полюсов (Global Geo, SubCent Sniper, Crisis Fortress):
                if stage_it >= 32:
                    if not smoke_test:
                        _save_l28_checkpoint(SAVE_DIR / f"l28_wave_{wave:02d}_specialist.pt", base_ck, wave_best_folded, wave_best_gl, it, wave, macro_stage, "specialist", f"L28_SPEC_W{wave:02d}", wave_best_metrics)
                    wave += 1
                    stage_it = 0
                    wave_best_geo = -1e9
                    pole_pick = wave % 3
                    if pole_pick == 1:
                        hof_state = {k: v.clone() for k, v in specialist_subcent[1].items()}
                        hof_g_logit = specialist_subcent[2].clone()
                        p_tag = f"SubCent Sniper ({specialist_subcent[0]:+.4f}%)"
                    elif pole_pick == 2:
                        hof_state = {k: v.clone() for k, v in specialist_fortress[1].items()}
                        hof_g_logit = specialist_fortress[2].clone()
                        p_tag = "Crisis Fortress"
                    else:
                        hof_state = {k: v.clone() for k, v in best_val_state.items()}
                        hof_g_logit = best_val_g_logit.clone()
                        p_tag = f"Global 4-Fold Leader ({best_geo_bp:+.2f}bp)"
                    log_msg(f"💎 [STAGE C PARETO POLISH W{wave}] Переключение ювелирной полировки на полюс: {p_tag}!")

    # =========================================================================
    # ФИНАЛЬНЫЙ СИНТЕЗ И ОТБОР CROSS-WAVE PARETO SOUP (`l28_crosswave_soup.pt`)
    # =========================================================================
    if not smoke_test:
        champ_files = sorted(
            list(SAVE_DIR.glob("l28_champ_iter_*.pt"))
            + list(SAVE_DIR.glob("l28_wave_*_best.pt"))
            + list(SAVE_DIR.glob("l28_wave_*_specialist.pt"))
        )
        scored = []
        seen_rn = set()
        for cf in champ_files:
            ck_i = torch.load(cf, map_location="cpu", weights_only=False)
            rn = str(ck_i.get("release_name", cf.name))
            if rn in seen_rn:
                continue
            seen_rn.add(rn)
            scored.append((float(ck_i.get("geo_score_bp", -1e9)), cf, ck_i))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Строим серию кандидатов Cross-Wave Soup (Top-2, Top-3, Top-4 + ортогональный микс со специалистами):
        soup_candidates: list[tuple[str, dict]] = [("SINGLE_BEST", best_val_folded)]
        if len(scored) >= 2:
            for name_s, weights_s in (
                ("SOUP_TOP2_60_40", [0.60, 0.40]),
                ("SOUP_TOP3_50_30_20", [0.50, 0.30, 0.20]),
                ("SOUP_TOP4_40_25_20_15", [0.40, 0.25, 0.20, 0.15]),
            ):
                k_s = min(len(weights_s), len(scored))
                ws = weights_s[:k_s]
                sw = sum(ws)
                ws = [w / sw for w in ws]
                st_s = _normalize_weights_l28(
                    {k: sum(ws[j] * scored[j][2][k].float() for j in range(k_s)) for k in HierarchicalSubCentPolicyL28.PARAM_KEYS},
                    policy.dtype,
                )
                soup_candidates.append((name_s, {k: v.detach().cpu().clone() for k, v in st_s.items()}))

        spec_sub_fold = policy.fold_gate_into_state(specialist_subcent[1], specialist_subcent[2])
        spec_for_fold = policy.fold_gate_into_state(specialist_fortress[1], specialist_fortress[2])
        tri_pole_st = _normalize_weights_l28(
            {
                k: 0.50 * best_val_folded[k].float() + 0.30 * spec_sub_fold[k].float() + 0.20 * spec_for_fold[k].float()
                for k in HierarchicalSubCentPolicyL28.PARAM_KEYS
            },
            policy.dtype,
        )
        soup_candidates.append(("SOUP_PARETO_TRIPOLE_50_30_20", {k: v.detach().cpu().clone() for k, v in tri_pole_st.items()}))

        soup_evals = evaluate_l28_dual_basket(
            vec_env, bf16_runner, lob_fu_all, ext138_all, roll_sell_30s_all,
            subcent_books, subcent_local_start,
            [x[1] for x in soup_candidates],
            global_subcent_val_windows, global_crisis_val_windows,
            win_len, feat_tick, device, budget_tiers=budget_tiers, true_latency=True,
        )
        for (nm_c, _), ev_c in zip(soup_candidates, soup_evals):
            log_msg(
                f"  🍲 [SOUP CANDIDATE {nm_c:<28}] Geo={ev_c['geo_score_bp']:+.2f}bp | SubEV={ev_c['subcent_ev_pct']:+.4f}% | "
                f"CrsEV={ev_c['crisis_ev_pct']:+.4f}% | DD={ev_c['dd_pct']:.2f}% | Cnl/F={ev_c['cnl_per_fill']:.1f}"
            )

        best_soup_idx = int(np.argmax([e["geo_score_bp"] for e in soup_evals[1:]])) + 1 if len(soup_evals) > 1 else 0
        win_soup_name, win_soup_st = soup_candidates[best_soup_idx]
        win_soup_res = soup_evals[best_soup_idx]
        _save_l28_checkpoint(
            SOUP_CHAMP_PATH, base_ck, win_soup_st, torch.zeros(N_BLOCKS_L28),
            it if "it" in locals() else max_iters, wave, "FINAL_SOUP", "final_soup",
            f"L28_{win_soup_name}", win_soup_res,
        )
        if float(win_soup_res["geo_score_bp"]) > best_geo_bp:
            _save_l28_checkpoint(
                BEST_VAL_PATH, base_ck, win_soup_st, torch.zeros(N_BLOCKS_L28),
                it if "it" in locals() else max_iters, wave, "FINAL_SOUP", "final_soup",
                f"L28_{win_soup_name}", win_soup_res,
            )
        log_msg(
            f"🏆 ФИНАЛЬНЫЙ L28 CROSS-WAVE SOUP ({win_soup_name}) СОХРАНЁН В {SOUP_CHAMP_PATH.name}: "
            f"Geo={win_soup_res['geo_score_bp']:+.2f}bp | SubCentEV={win_soup_res['subcent_ev_pct']:+.4f}% | CrisisEV={win_soup_res['crisis_ev_pct']:+.4f}%!"
        )
    else:
        log_msg("✅ L28 HIERARCHICAL SUBCENT SMOKE TEST PASSED SUCCESSFULLY!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--target-hours", type=float, default=8.0)
    ap.add_argument("--explore-max-iters", type=int, default=110)
    ap.add_argument("--explore-patience", type=int, default=34)
    ap.add_argument("--zoom1-iters", type=int, default=28)
    ap.add_argument("--zoom2-iters", type=int, default=22)
    ap.add_argument("--min-delta-bp", type=float, default=0.16)
    ap.add_argument("--n-groups", type=int, default=1024)
    ap.add_argument("--win-len", type=int, default=2400)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    run_lesson_28(
        max_iters=args.max_iters,
        target_hours=args.target_hours,
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
