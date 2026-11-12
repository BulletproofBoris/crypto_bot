#!/usr/bin/env python3
"""
_tools/lesson_22_p2_slot_queue_444d_trainer.py
=============================================================================
Урок 22 (Фаза 2 — 444D): Индивидуальный прогресс FIFO-очереди по каждому из 24 слотов
+ Честная физика пакетных штормов (Burst > 4 mut/s -> 2-tick RTT) + Косинусные рестарты
+ Zero Gate-Folding Drift (1,000 эпох от живого чемпиона L22_0969, БЕЗ жёстких правил)
-----------------------------------------------------------------------------
1. Вход головы расширен с 420D до 444D (`334D Market + 110D Agent`):
   добавлены 24 канала `slot_queue_progress_24d` (`[420:444]`), показывающие глубину
   продвижения в очереди КАЖДОГО слота (`0.0` = только встал в хвост, `+1.0` = на пороге филла).
   При старте `w1[420:444, :] = 0.0` — бит-в-бит сохраняет живого чемпиона `L22_0969`!
2. Исключён стартовый шумовой шок `0.045/0.080`: поиск сразу работает в продуктивном
   коридоре `sigma = 0.0032..0.0085` с 4 косинусными рестартами (период 250 эпох).
3. Устранён дрейф гейтов (`Zero Gate-Folding Drift`): при каждом обновлении чемпиона
   веса ребейзятся в `best_val_folded` с `g_logit = 0`.
=============================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_free_multi_order_env import VectorizedFreeMultiOrderEnv
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_18_multistack_lifo_survival_trainer import FusedBF16BackboneRunner
from _tools.fused_l22_train_tick import (
    MKT_IDX_LIST,
    AG_IDX_LIST,
    precompute_window_market_z334,
    fused_l22_decoupled_tick,
    evaluate_l22_dual_basket,
)

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
RELEASE_CHAMP_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
SAVE_DIR = BASE_DIR / "models" / "prl" / "lesson_22_p2_slot_queue_444d"
RESUME_CHAMP_PATH = SAVE_DIR / "lesson_22_p2_champ_resume.pt"
LOG_FILE = BASE_DIR / "data" / "logs" / "lesson_22_p2_slot_queue_444d.log"

# 16 семантических входных блоков головы политики (444D = 334D Market + 110D Agent):
SEMANTIC_BLOCKS_L21 = [
    ("Z_Backbone_GRU", 0, 256, 0.45),
    ("Guard_Risk_Heads", 256, 260, 0.70),
    ("Short_Triad_Swarm", 260, 278, 0.65),
    ("CrossVenue_5s_LeadLag", 278, 288, 1.55),
    ("Macro_Swarm_60_120s", 288, 306, 0.40),
    ("Carousel_Hazard_Realism", 306, 312, 0.65),
    ("CrossVenue_FirstOrder_6D", 312, 318, 1.65),
    ("LiqRatio_LIFO_Edge", 318, 324, 1.45),
    ("Active_24Slot_Weights", 324, 348, 2.05),
    ("Swarm_5s_Velocity", 348, 356, 1.05),
    ("Active_24Slot_Distances", 356, 380, 1.15),
    ("Fast_5s_LOB_Impulse", 380, 388, 1.10),
    ("Direct_Acct_State_8D", 388, 396, 2.10),
    ("FIFO_Queue_Ahead_8D", 396, 404, 2.25),
    ("Policy_Micro_Memory_16D", 404, 420, 2.05),
    ("PerSlot_FIFO_Progress_24D", 420, 444, 2.65),
]
N_BLOCKS_L21 = len(SEMANTIC_BLOCKS_L21)


def log_msg(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expand_block_gates_to_420(g_logits: torch.Tensor) -> torch.Tensor:
    gates_b = 2.0 * torch.sigmoid(g_logits.float())
    parts = [gates_b[..., idx : idx + 1].expand(*gates_b.shape[:-1], e - s) for idx, (_, s, e, _) in enumerate(SEMANTIC_BLOCKS_L21)]
    return torch.cat(parts, dim=-1)


def _batch_normalize_weights_l21(
    w1: torch.Tensor,
    b1: torch.Tensor,
    wr1: torch.Tensor,
    br1: torch.Tensor,
    wr2: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w_rec: torch.Tensor,
    b_rec: torch.Tensor,
    dtype: torch.dtype,
):
    squeeze_out = w1.dim() == 2
    if squeeze_out:
        w1 = w1.unsqueeze(0)
        b1 = b1.unsqueeze(0)
        wr1 = wr1.unsqueeze(0)
        br1 = br1.unsqueeze(0)
        wr2 = wr2.unsqueeze(0)
        w2 = w2.unsqueeze(0)
        b2 = b2.unsqueeze(0)
        w_rec = w_rec.unsqueeze(0)
        b_rec = b_rec.unsqueeze(0)

    w1_f = w1.float()
    w1_base = w1_f[:, :, :256]
    n1 = torch.linalg.vector_norm(w1_base, dim=(1, 2), keepdim=True).clamp(min=1e-6)
    w1_base = w1_base * torch.clamp(150.0 / n1, max=1.0)

    w1_gate = w1_f[:, :, 256:384]
    ng = torch.linalg.vector_norm(w1_gate, dim=(1, 2), keepdim=True).clamp(min=1e-6)
    w1_gate = w1_gate * torch.clamp(88.0 / ng, max=1.0)
    w1_f = torch.cat([w1_base, w1_gate], dim=-1)

    wr1_f = wr1.float()
    nr1 = torch.linalg.vector_norm(wr1_f, dim=(1, 2), keepdim=True).clamp(min=1e-6)
    wr1_f = wr1_f * torch.clamp(65.0 / nr1, max=1.0)
    br1_f = torch.clamp(br1.float(), -2.0, 2.0)

    wr2_f = wr2.float()
    nr2 = torch.linalg.vector_norm(wr2_f, dim=(1, 2), keepdim=True).clamp(min=1e-6)
    wr2_f = wr2_f * torch.clamp(40.0 / nr2, max=1.0)

    w2_f = w2.float()
    n2 = torch.linalg.vector_norm(w2_f, dim=(1, 2), keepdim=True).clamp(min=1e-6)
    w2_f = w2_f * torch.clamp(45.0 / n2, max=1.0)

    wrec_f = w_rec.float()
    nrec = torch.linalg.vector_norm(wrec_f, dim=(1, 2), keepdim=True).clamp(min=1e-6)
    wrec_f = wrec_f * torch.clamp(30.0 / nrec, max=1.0)
    brec_f = torch.clamp(b_rec.float(), -2.5, 2.5)

    b1_f = torch.clamp(b1.float(), -2.0, 2.0)
    b2_f = torch.clamp(b2.float(), -3.0, 2.0)

    res = (
        w1_f.to(dtype=dtype),
        b1_f.to(dtype=dtype),
        wr1_f.to(dtype=dtype),
        br1_f.to(dtype=dtype),
        wr2_f.to(dtype=dtype),
        w2_f.to(dtype=dtype),
        b2_f.to(dtype=dtype),
        wrec_f.to(dtype=dtype),
        brec_f.to(dtype=dtype),
    )
    if squeeze_out:
        return tuple(t.squeeze(0) for t in res)
    return res


class CompactDecoupledParetoPolicy:
    """Популяционная 444D политика Урока 22.2 с раздельными рыночными/агентскими каналами и 24D прогрессом слотов."""

    def __init__(self, backbone: MultiTaskLOBNet, n_groups: int = 2048, n_budgets: int = 6, device: torch.device = torch.device("cuda")):
        self.backbone = backbone
        self.n_groups = n_groups
        self.n_budgets = n_budgets
        self.pop_size = n_groups * n_budgets
        self.in_dim = 444
        self.h_dim = 256
        self.h1_full_dim = 384
        self.res_mid_dim = 64
        self.res_full_dim = 128
        self.rec_full_dim = 32
        self.out_dim = 26
        self.device = device
        self.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        self.w1 = torch.empty((n_groups, self.in_dim, self.h1_full_dim), dtype=self.dtype, device=device)
        self.b1 = torch.empty((n_groups, 1, self.h1_full_dim), dtype=self.dtype, device=device)
        self.w_res1 = torch.empty((n_groups, self.h_dim, self.res_full_dim), dtype=self.dtype, device=device)
        self.b_res1 = torch.empty((n_groups, 1, self.res_full_dim), dtype=self.dtype, device=device)
        self.w_res2 = torch.empty((n_groups, self.res_mid_dim, self.h_dim), dtype=self.dtype, device=device)
        self.w2 = torch.empty((n_groups, self.h_dim, self.out_dim), dtype=self.dtype, device=device)
        self.b2 = torch.empty((n_groups, 1, self.out_dim), dtype=self.dtype, device=device)
        self.w_rec = torch.zeros((n_groups, self.h_dim, self.rec_full_dim), dtype=self.dtype, device=device)
        self.b_rec = torch.zeros((n_groups, 1, self.rec_full_dim), dtype=self.dtype, device=device)
        self.g_logits = torch.zeros((n_groups, N_BLOCKS_L21), dtype=torch.float32, device=device)

        mut_profile = torch.ones((1, self.in_dim, 1), dtype=self.dtype, device=device)
        for _, s, e, m_scale in SEMANTIC_BLOCKS_L21:
            mut_profile[:, s:e, :] = float(m_scale)
        self.mut_profile = mut_profile

    def load_champion(self, ckpt_data: dict):
        if "backbone_state" in ckpt_data:
            self.backbone.load_state_dict(ckpt_data["backbone_state"], strict=False)
        w1_raw = ckpt_data["w1"].to(self.device).float()
        if w1_raw.shape[0] < self.in_dim:
            pad_rows = torch.zeros((self.in_dim - w1_raw.shape[0], w1_raw.shape[1]), dtype=torch.float32, device=self.device)
            w1_raw = torch.cat([w1_raw, pad_rows], dim=0)
        wrec_raw = ckpt_data["w_rec"].to(self.device).float() if ("w_rec" in ckpt_data and ckpt_data["w_rec"] is not None) else torch.zeros((self.h_dim, self.rec_full_dim), device=self.device)
        brec_raw = ckpt_data["b_rec"].to(self.device).float().view(1, -1) if ("b_rec" in ckpt_data and ckpt_data["b_rec"] is not None) else torch.zeros((1, self.rec_full_dim), device=self.device)

        w1_n, b1_n, wr1_n, br1_n, wr2_n, w2_n, b2_n, wrec_n, brec_n = _batch_normalize_weights_l21(
            w1_raw,
            ckpt_data["b1"].to(self.device).view(1, -1),
            ckpt_data["w_res1"].to(self.device),
            ckpt_data["b_res1"].to(self.device).view(1, -1),
            ckpt_data["w_res2"].to(self.device),
            ckpt_data["w2"].to(self.device),
            ckpt_data["b2"].to(self.device).view(1, -1),
            wrec_raw,
            brec_raw,
            self.dtype,
        )
        with torch.no_grad():
            self.w1.copy_(w1_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.b1.copy_(b1_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.w_res1.copy_(wr1_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.b_res1.copy_(br1_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.w_res2.copy_(wr2_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.w2.copy_(w2_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.b2.copy_(b2_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.w_rec.copy_(wrec_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.b_rec.copy_(brec_n.unsqueeze(0).expand(self.n_groups, -1, -1))
            self.g_logits.zero_()

            n_mut = self.n_groups - 1
            if n_mut > 0:
                self.w1[1:].add_(torch.randn((n_mut, self.in_dim, self.h1_full_dim), device=self.device, dtype=self.dtype) * 0.004 * self.mut_profile)
                self.w1[1:, 420:444].add_(torch.randn((n_mut, 24, self.h1_full_dim), device=self.device, dtype=self.dtype) * 0.018)
                self.b1[1:].add_(torch.randn((n_mut, 1, self.h1_full_dim), device=self.device, dtype=self.dtype) * 0.004)
                self.w_res1[1:].add_(torch.randn((n_mut, self.h_dim, self.res_full_dim), device=self.device, dtype=self.dtype) * 0.006)
                self.b_res1[1:].add_(torch.randn((n_mut, 1, self.res_full_dim), device=self.device, dtype=self.dtype) * 0.004)
                self.w_res2[1:].add_(torch.randn((n_mut, self.res_mid_dim, self.h_dim), device=self.device, dtype=self.dtype) * 0.007)
                self.w2[1:].add_(torch.randn((n_mut, self.h_dim, self.out_dim), device=self.device, dtype=self.dtype) * 0.006)
                self.b2[1:].add_(torch.randn((n_mut, 1, self.out_dim), device=self.device, dtype=self.dtype) * 0.012)
                self.w_rec[1:].add_(torch.randn((n_mut, self.h_dim, self.rec_full_dim), device=self.device, dtype=self.dtype) * 0.008)
                self.b_rec[1:].add_(torch.randn((n_mut, 1, self.rec_full_dim), device=self.device, dtype=self.dtype) * 0.006)

    def effective_group_w1(self) -> torch.Tensor:
        gate_420 = expand_block_gates_to_420(self.g_logits).unsqueeze(-1)
        return (self.w1.float() * gate_420).to(dtype=self.dtype).contiguous()

    def fold_gate_into_state(self, state: dict, g_logit_1d: torch.Tensor) -> dict:
        gate_420 = expand_block_gates_to_420(g_logit_1d.to(self.device)).unsqueeze(-1)
        w1_folded = state["w1"].to(self.device).float() * gate_420
        w1_n, b1_n, wr1_n, br1_n, wr2_n, w2_n, b2_n, wrec_n, brec_n = _batch_normalize_weights_l21(
            w1_folded,
            state["b1"].to(self.device),
            state["w_res1"].to(self.device),
            state["b_res1"].to(self.device),
            state["w_res2"].to(self.device),
            state["w2"].to(self.device),
            state["b2"].to(self.device),
            state["w_rec"].to(self.device),
            state["b_rec"].to(self.device),
            self.dtype,
        )
        return {
            "w1": w1_n.detach().cpu(),
            "b1": b1_n.detach().cpu(),
            "w_res1": wr1_n.detach().cpu(),
            "b_res1": br1_n.detach().cpu(),
            "w_res2": wr2_n.detach().cpu(),
            "w2": w2_n.detach().cpu(),
            "b2": b2_n.detach().cpu(),
            "w_rec": wrec_n.detach().cpu(),
            "b_rec": brec_n.detach().cpu(),
        }

    def evolve_gated_groups(self, pareto_fitness: torch.Tensor, hof_state: dict, hof_g_logit: torch.Tensor, elite_frac: float = 0.03, progress: float = 0.0):
        n_elites = max(12, int(self.n_groups * elite_frac))
        if (self.n_groups - n_elites) % 2 != 0:
            n_elites += 1
        sorted_g = torch.argsort(pareto_fitness, descending=True)
        elite_g = sorted_g[:n_elites]

        w1_el, b1_el, wr1_el, br1_el, wr2_el, w2_el, b2_el, wrec_el, brec_el = _batch_normalize_weights_l21(
            self.w1[elite_g],
            self.b1[elite_g],
            self.w_res1[elite_g],
            self.b_res1[elite_g],
            self.w_res2[elite_g],
            self.w2[elite_g],
            self.b2[elite_g],
            self.w_rec[elite_g],
            self.b_rec[elite_g],
            self.dtype,
        )
        gl_el = torch.clamp(self.g_logits[elite_g].clone(), -2.5, 2.0)

        w1_el[0].copy_(hof_state["w1"].to(self.device, self.dtype))
        b1_el[0].copy_(hof_state["b1"].to(self.device, self.dtype))
        wr1_el[0].copy_(hof_state["w_res1"].to(self.device, self.dtype))
        br1_el[0].copy_(hof_state["b_res1"].to(self.device, self.dtype))
        wr2_el[0].copy_(hof_state["w_res2"].to(self.device, self.dtype))
        w2_el[0].copy_(hof_state["w2"].to(self.device, self.dtype))
        b2_el[0].copy_(hof_state["b2"].to(self.device, self.dtype))
        wrec_el[0].copy_(hof_state["w_rec"].to(self.device, self.dtype))
        brec_el[0].copy_(hof_state["b_rec"].to(self.device, self.dtype))
        gl_el[0].copy_(hof_g_logit.to(self.device, torch.float32))

        self.w1[:n_elites].copy_(w1_el)
        self.b1[:n_elites].copy_(b1_el)
        self.w_res1[:n_elites].copy_(wr1_el)
        self.b_res1[:n_elites].copy_(br1_el)
        self.w_res2[:n_elites].copy_(wr2_el)
        self.w2[:n_elites].copy_(w2_el)
        self.b2[:n_elites].copy_(b2_el)
        self.w_rec[:n_elites].copy_(wrec_el)
        self.b_rec[:n_elites].copy_(brec_el)
        self.g_logits[:n_elites].copy_(gl_el)

        n_pairs = (self.n_groups - n_elites) // 2
        pair_parents = torch.where(
            torch.rand(n_pairs, device=self.device) < 0.58,
            torch.zeros(n_pairs, dtype=torch.long, device=self.device),
            torch.randint(0, n_elites, (n_pairs,), device=self.device),
        )
        # Fine-scale Cosine Warm Restarts: cos_w in [0.0, 1.0] passed via `progress`
        cos_w = float(progress)
        sig_lo = 0.0032 + 0.0053 * cos_w
        sig_hi = 0.0060 + 0.0090 * cos_w
        sigma_pairs = torch.where(
            torch.rand(n_pairs, 1, 1, device=self.device) < 0.72,
            torch.full((n_pairs, 1, 1), sig_lo, device=self.device, dtype=self.dtype),
            torch.full((n_pairs, 1, 1), sig_hi, device=self.device, dtype=self.dtype),
        )

        dst_gl = self.g_logits[n_elites:].view(n_pairs, 2, N_BLOCKS_L21)
        dst_gl.copy_(gl_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, N_BLOCKS_L21))
        ngl = torch.randn((n_pairs, N_BLOCKS_L21), device=self.device, dtype=torch.float32) * (0.018 + 0.025 * cos_w)
        ngl[:, 0] *= 0.25
        dst_gl[:, 0].add_(ngl).clamp_(-2.5, 2.0)
        dst_gl[:, 1].sub_(ngl).clamp_(-2.5, 2.0)

        dst_w1 = self.w1[n_elites:].view(n_pairs, 2, self.in_dim, self.h1_full_dim)
        dst_w1.copy_(w1_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, self.in_dim, self.h1_full_dim))
        nw1 = torch.randn((n_pairs, self.in_dim, self.h1_full_dim), device=self.device, dtype=self.dtype) * sigma_pairs * self.mut_profile
        nw1[:, 420:444, :].add_(
            torch.randn((n_pairs, 24, self.h1_full_dim), device=self.device, dtype=self.dtype) * (0.0045 + 0.0075 * cos_w)
        )
        dst_w1[:, 0].add_(nw1)
        dst_w1[:, 1].sub_(nw1)
        del nw1

        dst_b1 = self.b1[n_elites:].view(n_pairs, 2, 1, self.h1_full_dim)
        dst_b1.copy_(b1_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, 1, self.h1_full_dim))
        nb1 = torch.randn((n_pairs, 1, self.h1_full_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 0.5)
        dst_b1[:, 0].add_(nb1)
        dst_b1[:, 1].sub_(nb1)

        dst_wr1 = self.w_res1[n_elites:].view(n_pairs, 2, self.h_dim, self.res_full_dim)
        dst_wr1.copy_(wr1_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, self.h_dim, self.res_full_dim))
        nwr1 = torch.randn((n_pairs, self.h_dim, self.res_full_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 1.8)
        dst_wr1[:, 0].add_(nwr1)
        dst_wr1[:, 1].sub_(nwr1)

        dst_br1 = self.b_res1[n_elites:].view(n_pairs, 2, 1, self.res_full_dim)
        dst_br1.copy_(br1_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, 1, self.res_full_dim))
        nbr1 = torch.randn((n_pairs, 1, self.res_full_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 0.8)
        dst_br1[:, 0].add_(nbr1)
        dst_br1[:, 1].sub_(nbr1)

        dst_wr2 = self.w_res2[n_elites:].view(n_pairs, 2, self.res_mid_dim, self.h_dim)
        dst_wr2.copy_(wr2_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, self.res_mid_dim, self.h_dim))
        nwr2 = torch.randn((n_pairs, self.res_mid_dim, self.h_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 2.5)
        dst_wr2[:, 0].add_(nwr2)
        dst_wr2[:, 1].sub_(nwr2)

        dst_w2 = self.w2[n_elites:].view(n_pairs, 2, self.h_dim, self.out_dim)
        dst_w2.copy_(w2_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, self.h_dim, self.out_dim))
        nw2 = torch.randn((n_pairs, self.h_dim, self.out_dim), device=self.device, dtype=self.dtype) * sigma_pairs
        dst_w2[:, 0].add_(nw2)
        dst_w2[:, 1].sub_(nw2)

        dst_b2 = self.b2[n_elites:].view(n_pairs, 2, 1, self.out_dim)
        dst_b2.copy_(b2_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, 1, self.out_dim))
        nb2 = torch.randn((n_pairs, 1, self.out_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 1.5)
        dst_b2[:, 0].add_(nb2)
        dst_b2[:, 1].sub_(nb2)

        dst_wrec = self.w_rec[n_elites:].view(n_pairs, 2, self.h_dim, self.rec_full_dim)
        dst_wrec.copy_(wrec_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, self.h_dim, self.rec_full_dim))
        nwrec = torch.randn((n_pairs, self.h_dim, self.rec_full_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 1.8)
        dst_wrec[:, 0].add_(nwrec)
        dst_wrec[:, 1].sub_(nwrec)

        dst_brec = self.b_rec[n_elites:].view(n_pairs, 2, 1, self.rec_full_dim)
        dst_brec.copy_(brec_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, 1, self.rec_full_dim))
        nbrec = torch.randn((n_pairs, 1, self.rec_full_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 1.2)
        dst_brec[:, 0].add_(nbrec)
        dst_brec[:, 1].sub_(nbrec)


def compute_pareto_group_fitness(
    flat_util_g: torch.Tensor,
    uw_util_g: torch.Tensor,
    np_fills_g: torch.Tensor,
) -> torch.Tensor:
    """
    Двухконтурное лексикографическое Парето-ранжирование 2,048 групп на GPU за ~0.2 мс (Zero Artificial Bonuses):
    1. Лексикографическая допустимость (`feasible_mm`): на tradable окнах `NORM + PUMP` геном обязан совершать
       реальные сделки (`np_fills_g >= 1.0`), тогда как на `DUMP` окнах стояние в кэше (`0` сделок) 100% законно!
       Это полностью закрывает лазейку, когда в турбулентном батче «мертвый молчун» с `0.00 bp` обгонял живых ММ (`-1.5 bp`).
    2. Среди всех живых ММ считает точное число строго доминирующих особей `dom_count[i]` по двум независимым
       чистым экономическим осям `(flat_util_g, uw_util_g)` и упорядочивает по Чебышёвскому балансу рангов.
    """
    g_n = flat_util_g.shape[0]
    feasible = np_fills_g >= 1.0
    if int(feasible.long().sum().item()) < 32:
        feasible = np_fills_g > 0.0

    # Чтобы мёртвые геномы с 0 сделок не создавали ложного доминирования над живыми ММ в турбулентных батчах:
    eff_flat = torch.where(feasible, flat_util_g, flat_util_g - 5000.0)
    eff_uw = torch.where(feasible, uw_util_g, uw_util_g - 5000.0)

    f_col = eff_flat.unsqueeze(1)
    f_row = eff_flat.unsqueeze(0)
    u_col = eff_uw.unsqueeze(1)
    u_row = eff_uw.unsqueeze(0)

    dom_by_j = (f_row >= f_col) & (u_row >= u_col) & ((f_row > f_col + 1e-6) | (u_row > u_col + 1e-6))
    dom_count = dom_by_j.float().sum(dim=1)

    r_flat = torch.argsort(torch.argsort(eff_flat)).float() / float(max(1, g_n - 1))
    r_uw = torch.argsort(torch.argsort(eff_uw)).float() / float(max(1, g_n - 1))

    pareto_layer_score = -torch.log1p(dom_count)
    chebyshev_balance = 0.55 * torch.minimum(r_flat, r_uw) + 0.45 * (0.68 * r_flat + 0.32 * r_uw)
    return 10.0 * feasible.float() + pareto_layer_score + 1.5 * chebyshev_balance


def _build_roll_sell_30s(sell_vol_t: torch.Tensor) -> torch.Tensor:
    sv_1d = torch.clamp(sell_vol_t.float(), min=0.0)
    cs = torch.cat([torch.zeros(1, dtype=torch.float32, device=sv_1d.device), torch.cumsum(sv_1d, dim=0)], dim=0)
    idx_hi = torch.arange(1, len(sv_1d) + 1, device=sv_1d.device)
    idx_lo = torch.clamp(idx_hi - 30, min=0)
    return torch.clamp(cs[idx_hi] - cs[idx_lo], min=10.0)


def run_lesson_22(iterations: int = 1000, n_groups: int = 2048, early_stop: bool = False, rng_seed: int = 2026221):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_budgets = 6
    pop_size = n_groups * n_budgets

    log_msg("=" * 116)
    log_msg(
        f"🚀 УРОК 22.2 (PER-SLOT FIFO 444D + BURST PHYSICS + COSINE RESTARTS): {iterations} ИТЕРАЦИЙ | 10 ОКОН x {pop_size} = {10 * pop_size:,} СРЕД"
    )
    log_msg("=" * 116)

    raw = np.load(CACHE_PATH, allow_pickle=True)
    cv_ver = int(raw["has_cross_venue_v1"]) if "has_cross_venue_v1" in raw.files else 0
    assert cv_ver >= 1, "Сначала обновите кросс-биржевой кэш `_tools/cross_venue_first_order_net.py`!"
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.01
    slice_len = min(960000, n_total)
    s0 = n_total - slice_len

    sub = {
        "bid_p": raw["bid_p"][s0:],
        "ask_p": raw["ask_p"][s0:],
        "bid_q": raw["bid_q"][s0:],
        "ask_q": raw["ask_q"][s0:],
        "buy_vol": raw["buy_vol"][s0:],
        "sell_vol": raw["sell_vol"][s0:],
        "y_sweep_12s": raw["y_sweep_12s"][s0:],
        "y_bleed_60s": raw["y_bleed_60s"][s0:],
        "y_crash_300s": raw["y_crash_300s"][s0:],
        "y_exit_30s": raw["y_exit_30s"][s0:],
        "tick_size": tick_sz,
    }
    v7_raw_t = torch.from_numpy(raw["v7_signals_58d"][s0:].astype(np.float32)).to(device)
    del raw

    vel_cols = [0, 2, 4, 6, 8, 12, 34, 42]
    v7_vel8 = torch.zeros((len(v7_raw_t), 8), dtype=torch.float32, device=device)
    v7_vel8[5:] = torch.clamp(v7_raw_t[5:, vel_cols] - v7_raw_t[:-5, vel_cols], -3.0, 3.0)

    bp_t = torch.from_numpy(sub["bid_p"][:, :5].astype(np.float32)).to(device)
    ap_t = torch.from_numpy(sub["ask_p"][:, :5].astype(np.float32)).to(device)
    bq_t = torch.from_numpy(sub["bid_q"][:, :5].astype(np.float32)).to(device)
    aq_t = torch.from_numpy(sub["ask_q"][:, :5].astype(np.float32)).to(device)
    bv_t = torch.from_numpy(sub["buy_vol"].astype(np.float32)).to(device)
    sv_t = torch.from_numpy(sub["sell_vol"].astype(np.float32)).to(device)

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

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    policy = CompactDecoupledParetoPolicy(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, device=device)
    v7_all = v7_all.to(dtype=policy.dtype)

    summary_path = SAVE_DIR / "lesson_22_p2_summary.json"
    latest_ckpt = SAVE_DIR / "lesson_22_p2_champ_latest.pt"
    history = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []

    seed_path = RESUME_CHAMP_PATH if RESUME_CHAMP_PATH.exists() else (latest_ckpt if latest_ckpt.exists() else RELEASE_CHAMP_PATH)
    ck = torch.load(seed_path, map_location=device, weights_only=False)
    policy.load_champion(ck)
    start_iter = (int(ck.get("iteration", 0)) + 1) if (int(ck.get("lesson", 21)) == 22 and int(ck.get("phase", 1)) == 2) else 1
    if start_iter > 1 and history:
        history = [r for r in history if int(r.get("iteration", 0)) < start_iter]
    log_msg(
        f"🧬 Урок 22.2 (444D Per-Slot FIFO Queue Progress + Burst Physics + Cosine Restarts): старт с it={start_iter} из {seed_path.name} "
        f"({ck.get('release_name', 'L22_0969_queue_kpd_4venue_decoupled')}) на {n_total:,} тиках!"
    )

    hof_state = {
        "w1": policy.w1[0].detach().cpu().clone(),
        "b1": policy.b1[0].detach().cpu().clone(),
        "w_res1": policy.w_res1[0].detach().cpu().clone(),
        "b_res1": policy.b_res1[0].detach().cpu().clone(),
        "w_res2": policy.w_res2[0].detach().cpu().clone(),
        "w2": policy.w2[0].detach().cpu().clone(),
        "b2": policy.b2[0].detach().cpu().clone(),
        "w_rec": policy.w_rec[0].detach().cpu().clone(),
        "b_rec": policy.b_rec[0].detach().cpu().clone(),
    }
    hof_g_logit = torch.zeros(N_BLOCKS_L21, dtype=torch.float32)

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

    vram_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    log_msg(f"⚡ Память GPU после инициализации Decoupled 420D + 4-Venue кэша: {vram_alloc_mb:,.0f} MiB (из 8,151 MiB)")

    budget_tiers = torch.tensor([100.0, 300.0, 625.0, 1000.0, 1500.0, 2500.0], device=device)
    env_budgets = budget_tiers.unsqueeze(0).expand(n_groups, n_budgets).reshape(pop_size)
    rng = np.random.default_rng(rng_seed)
    win_len = 600

    b1_all_np = vec_env.bids[:, 0].detach().cpu().numpy()
    bq_all_np = vec_env.bids_q[:, 0].detach().cpu().numpy()
    aq_all_np = vec_env.asks_q[:, 0].detach().cpu().numpy()
    bv_all_np = vec_env.buy_vol.detach().cpu().numpy()
    sv_all_np = vec_env.sell_vol.detach().cpu().numpy()

    val_tail_len = int(450000 * 0.15)
    split_idx = slice_len - val_tail_len
    train_starts = np.arange(100, split_idx - win_len - 100, 60)
    val_starts = np.arange(split_idx, slice_len - win_len - 100, 60)

    def classify_windows(starts_arr, require_tradable: bool = False):
        norm_s, dump_s, pump_s = [], [], []
        for w0 in starts_arr:
            seg = b1_all_np[w0 : w0 + win_len]
            entry_fillable = True
            if require_tradable:
                s_sv = float(sv_all_np[w0 : w0 + win_len].sum())
                s_bv = float(bv_all_np[w0 : w0 + win_len].sum())
                q_b25 = float(np.percentile(bq_all_np[w0 : w0 + win_len], 25))
                q_a25 = float(np.percentile(aq_all_np[w0 : w0 + win_len], 25))
                if s_sv < q_b25 * 1.5 or s_bv < q_a25 * 1.2:
                    continue
                sv_300 = float(sv_all_np[w0 : w0 + 300].sum())
                bv_420 = float(bv_all_np[w0 + 180 : w0 + win_len].sum())
                entry_fillable = (sv_300 >= max(float(bq_all_np[w0]), q_b25) * 1.35) and (bv_420 >= q_a25 * 1.35)
            end_diff = (seg[-1] - seg[0]) / tick_sz
            max_drop = (np.max(seg[:240]) - np.min(seg[120:])) / tick_sz
            max_rally = (np.max(seg[120:]) - np.min(seg[:240])) / tick_sz
            rng_ticks = (np.max(seg) - np.min(seg)) / tick_sz
            if end_diff <= -1.5 or max_drop >= 2.2:
                dump_s.append(int(w0))
            elif (end_diff >= 1.0 or max_rally >= 2.0) and entry_fillable:
                pump_s.append(int(w0))
            elif rng_ticks >= 1.0 and abs(end_diff) < 0.5 and entry_fillable:
                norm_s.append(int(w0))
        return (
            np.array(norm_s if norm_s else starts_arr, dtype=np.int64),
            np.array(dump_s if dump_s else starts_arr, dtype=np.int64),
            np.array(pump_s if pump_s else starts_arr, dtype=np.int64),
        )

    normal_starts, dump_starts, pump_starts = classify_windows(train_starts, require_tradable=True)
    val_norm, val_dump, val_pump = classify_windows(val_starts, require_tradable=True)

    # Двухконтурная валидационная корзина (22 окна = CLEAN_18 + UW_4):
    val_basket = []
    for k_i in range(6):
        val_basket.append(("NORM", int(val_norm[(k_i * len(val_norm) // 6) % len(val_norm)])))
        val_basket.append(("DUMP", int(val_dump[(k_i * len(val_dump) // 6) % len(val_dump)])))
        val_basket.append(("PUMP", int(val_pump[(k_i * len(val_pump) // 6) % len(val_pump)])))
    val_basket.append(("UW_NORM", int(val_norm[len(val_norm) // 4])))
    val_basket.append(("UW_NORM", int(val_norm[(3 * len(val_norm)) // 4])))
    val_basket.append(("UW_DUMP", int(val_dump[len(val_dump) // 4])))
    val_basket.append(("UW_DUMP", int(val_dump[(3 * len(val_dump)) // 4])))

    base_folded = policy.fold_gate_into_state(hof_state, hof_g_logit)
    t_v0 = time.time()
    init_val_res = evaluate_l22_dual_basket(
        vec_env, bf16_runner, lob_fu_all, v7_all, roll_sell_30s_all, [base_folded] * 8, val_basket, win_len, tick_sz, device, True, True
    )[0]
    dt_v0 = time.time() - t_v0

    best_pareto_util = float(init_val_res["pareto_util_bp"])
    best_clean_util = float(init_val_res["clean_util_bp"])
    best_uw_util = float(init_val_res["uw_util_bp"])
    best_clean_ev = float(init_val_res["clean_ev"])
    best_val_ev = float(init_val_res["realistic_ev"])
    best_eta_queue = float(init_val_res["clean_eta_queue_pct"])
    best_cnl_per_fill = float(init_val_res["clean_cnl_per_fill"])
    best_val_state = {k: v.clone() for k, v in hof_state.items()}
    best_val_g_logit = hof_g_logit.clone()
    best_val_folded = {k: v.clone() for k, v in base_folded.items()}
    last_val_champ_it = max(0, start_iter - 1)

    log_msg(
        f"⚖️ СТАРТОВАЯ БАЗА L22 В ЧЕСТНОЙ ФИЗИКЕ REST+LATENCY (скомпилировано за {dt_v0:.1f}с): "
        f"ParetoUtil={best_pareto_util:+.2f}bp [Clean18={best_clean_util:+.2f}bp (EV={best_clean_ev:+.4f}%, "
        f"Fills={init_val_res['clean_fills']:.1f}, Cancels={init_val_res['clean_cancels']:.1f}, Cnl/Fill={best_cnl_per_fill:.1f}, η_queue={best_eta_queue:.1f}%) | "
        f"UW4={best_uw_util:+.2f}bp (ΔMTM={init_val_res['uw_delta_ev']:+.4f}%)] | "
        f"Clean[N:{init_val_res['clean_norm_pct']:+.3f}% D:{init_val_res['clean_dump_pct']:+.3f}% P:{init_val_res['clean_pump_pct']:+.3f}% DD:{init_val_res['clean_dd_pct']:.2f}%]"
    )

    for _skip in range(1, start_iter):
        for _ in range(10):
            _ = rng.integers(0, 10)

    W_par = 10  # 6 Flat-Start (0..5: 2 NORM + 2 DUMP + 2 PUMP) + 4 Underwater (6..9: 2 UW_NORM + 2 UW_DUMP)
    N_par = W_par * pop_size
    S_slots = int(vec_env.n_side_slots)
    budget_16 = env_budgets.unsqueeze(0).expand(W_par, pop_size).reshape(N_par).contiguous()
    nes_w = torch.softmax(-torch.arange(24, dtype=torch.float32, device=device) * 0.25, dim=0)
    mkt_idx_t = torch.tensor(MKT_IDX_LIST, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_LIST, dtype=torch.long, device=device)
    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    chunk_sz = 100

    for it in range(start_iter, iterations + 1):
        t0 = time.time()
        cycle_prog = float((it - 1) % 250) / 249.0
        cos_w = 0.5 * (1.0 + math.cos(math.pi * cycle_prog))
        cosine_decay = 0.25 + 0.75 * cos_w

        w_batch = [
            ("NORM1", int(rng.choice(normal_starts))),
            ("NORM2", int(rng.choice(normal_starts))),
            ("DUMP1", int(rng.choice(dump_starts))),
            ("DUMP2", int(rng.choice(dump_starts))),
            ("PUMP1", int(rng.choice(pump_starts))),
            ("PUMP2", int(rng.choice(pump_starts))),
            ("UW_NORM1", int(rng.choice(normal_starts))),
            ("UW_NORM2", int(rng.choice(normal_starts))),
            ("UW_DUMP1", int(rng.choice(dump_starts))),
            ("UW_DUMP2", int(rng.choice(dump_starts))),
        ]

        with torch.no_grad():
            w1_g = policy.effective_group_w1()
            w1_mkt_g = w1_g.index_select(1, mkt_idx_t).contiguous()
            w1_ag_g = w1_g.index_select(1, ag_idx_t).contiguous()
            del w1_g
            b1_g = policy.b1
            wr1_g = policy.w_res1
            br1_g = policy.b_res1
            wr2_g = policy.w_res2
            w2_g = policy.w2
            b2_g = policy.b2
            wrec_g = policy.w_rec
            brec_g = policy.b_rec

            w_starts = torch.tensor([ws for _, ws in w_batch], dtype=torch.long, device=device)
            t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
            t_steps_cl = torch.clamp(t_steps, max=vec_env.max_steps - 1)

            flow_w_all = vec_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
            lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()
            v7_w_all = v7_all[t_steps_cl[: win_len - 1]].contiguous()
            z_mkt_all = precompute_window_market_z334(flow_w_all, lob_fu_w_all, v7_w_all, bf16_runner)
            del flow_w_all, lob_fu_w_all, v7_w_all

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

            for wi, (reg_name, ws) in enumerate(w_batch):
                if reg_name.startswith("UW_"):
                    s0_w = wi * pop_size
                    s1_w = (wi + 1) * pop_size
                    b1_w0 = float(max(tick_sz, float(vec_env.bids[ws, 0].item())))
                    epx = round((b1_w0 + 5.0 * tick_sz) / tick_sz) * tick_sz
                    iqty = torch.floor((env_budgets * 0.55 / b1_w0) * 100.0) / 100.0
                    v_mask = iqty >= 2.0
                    rcash = torch.clamp(env_budgets - iqty * b1_w0 * (1.0 - vec_env.maker_fee), min=0.0)
                    cash[s0_w:s1_w] = torch.where(v_mask, rcash, env_budgets)
                    pos_qty[s0_w:s1_w] = torch.where(v_mask, iqty, torch.zeros_like(iqty))
                    avg_entry[s0_w:s1_w] = torch.where(v_mask, torch.full_like(iqty, epx), torch.zeros_like(iqty))
                    pos_cost_usd[s0_w:s1_w] = torch.where(v_mask, iqty * epx * (1.0 + vec_env.maker_fee), torch.zeros_like(iqty))
                    min_buy_px[s0_w:s1_w] = torch.where(v_mask, torch.full_like(iqty, epx), torch.zeros_like(iqty))
                    entry_s[s0_w:s1_w] = -180

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

            init_b1_16 = torch.clamp(b1_w_all[0], min=tick_sz).unsqueeze(1).expand(W_par, pop_size).reshape(N_par)
            start_mtm_eq = cash + pos_qty * init_b1_16 * (1.0 - vec_env.maker_fee)
            peak_mtm_eq = start_mtm_eq.clone()
            sum_sq_dd_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            max_dd_pct_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_mut_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            n_fills_16 = torch.zeros(N_par, dtype=torch.int16, device=device)
            n_cancels_16 = torch.zeros(N_par, dtype=torch.int16, device=device)

            t_total_steps = win_len - 1
            for c0 in range(0, t_total_steps, chunk_sz):
                c1 = min(t_total_steps, c0 + chunk_sz)
                c_len = c1 - c0
                z_ch = z_mkt_all[c0:c1].reshape(c_len * W_par, 334).unsqueeze(0).expand(n_groups, -1, -1)
                pre_mkt_ch = (torch.bmm(z_ch, w1_mkt_g) + b1_g).view(n_groups, c_len, W_par, 384).permute(1, 0, 2, 3).contiguous()
                for sc in range(c_len):
                    s = c0 + sc
                    (
                        mem_h,
                        cash,
                        pos_qty,
                        pos_cost_usd,
                        avg_entry,
                        min_buy_px,
                        low_tranche_qty,
                        mid_buy_px,
                        mid_tranche_qty,
                        entry_s,
                        bid_px,
                        bid_qty,
                        bid_qa,
                        bid_qa0,
                        bid_act_s,
                        ask_px,
                        ask_qty,
                        ask_qa,
                        ask_qa0,
                        ask_act_s,
                        peak_mtm_eq,
                        sum_sq_dd_16,
                        max_dd_pct_16,
                        sum_mut_16,
                        n_fills_16,
                        n_cancels_16,
                    ) = fused_l22_decoupled_tick(
                        s_steps_i16[s],
                        s_steps_i16[s + 1],
                        pre_mkt_ch[sc],
                        b1_w_all[s],
                        a1_w_all[s],
                        b1q_w_all[s],
                        a1q_w_all[s],
                        rb_w_all[s],
                        rs_w_all[s],
                        b10_w_all[s],
                        a10_w_all[s],
                        bq10_w_all[s],
                        aq10_w_all[s],
                        b1_w_all[s + 1],
                        a1_w_all[s + 1],
                        sv_w_all[s + 1],
                        bv_w_all[s + 1],
                        w1_ag_g,
                        wr1_g,
                        br1_g,
                        wr2_g,
                        w2_g,
                        b2_g,
                        wrec_g,
                        brec_g,
                        vec_env.slot_tick_offsets,
                        vec_env.slot_lob_lvl,
                        vec_env.is_penny_slot,
                        budget_16,
                        mem_h,
                        cash,
                        pos_qty,
                        pos_cost_usd,
                        avg_entry,
                        min_buy_px,
                        low_tranche_qty,
                        mid_buy_px,
                        mid_tranche_qty,
                        entry_s,
                        bid_px,
                        bid_qty,
                        bid_qa,
                        bid_qa0,
                        bid_act_s,
                        ask_px,
                        ask_qty,
                        ask_qa,
                        ask_qa0,
                        ask_act_s,
                        peak_mtm_eq,
                        sum_sq_dd_16,
                        max_dd_pct_16,
                        sum_mut_16,
                        n_fills_16,
                        n_cancels_16,
                        tick_sz,
                        vec_env.maker_fee,
                        vec_env.taker_fee,
                        True,
                        True,
                    )
                del pre_mkt_ch

            end_b1_16 = torch.clamp(b1_w_all[win_len - 1], min=tick_sz).unsqueeze(1).expand(W_par, pop_size).reshape(N_par)
            rem = pos_qty > 1e-6
            end_cash = torch.where(rem, cash + pos_qty * end_b1_16 * (1.0 - vec_env.maker_fee), cash)
            end_dd_pct = 100.0 * torch.clamp(peak_mtm_eq - end_cash, min=0.0) / budget_16
            max_dd_pct_16 = torch.maximum(max_dd_pct_16, end_dd_pct)

            del (
                w1_mkt_g,
                w1_ag_g,
                z_mkt_all,
                mem_h,
                bid_px,
                bid_qty,
                bid_qa,
                bid_qa0,
                bid_act_s,
                ask_px,
                ask_qty,
                ask_qa,
                ask_qa0,
                ask_act_s,
                b10_w_all,
                a10_w_all,
                bq10_w_all,
                aq10_w_all,
            )

            # Разделение двух независимых экономических осей:
            # Контур 1: Flat-Start Active MM (окна 0..5 = 2 NORM + 2 DUMP + 2 PUMP)
            # Контур 2: Underwater Recovery (окна 6..9 = 2 UW_NORM + 2 UW_DUMP)
            w_mtm_pct_mat = (100.0 * (end_cash - start_mtm_eq) / budget_16).view(W_par, pop_size)
            w_sq_dd_mat = sum_sq_dd_16.view(W_par, pop_size)
            w_dd_mat = max_dd_pct_16.view(W_par, pop_size)
            w_api_mat = sum_mut_16.view(W_par, pop_size)
            w_fills_mat = n_fills_16.float().view(W_par, pop_size)
            w_cancels_mat = n_cancels_16.float().view(W_par, pop_size)

            flat_steps = 6.0 * float(win_len - 1)
            uw_steps = 4.0 * float(win_len - 1)

            flat_norm_pct = w_mtm_pct_mat[0:2].mean(dim=0)
            flat_dump_pct = w_mtm_pct_mat[2:4].mean(dim=0)
            flat_pump_pct = w_mtm_pct_mat[4:6].mean(dim=0)
            flat_ret_pct = 0.65 * flat_norm_pct + 0.20 * flat_pump_pct + 0.15 * flat_dump_pct
            flat_cvar_pct = torch.topk(w_mtm_pct_mat[0:6], 2, dim=0, largest=False).values.mean(dim=0)
            flat_ulcer_pct = torch.sqrt(w_sq_dd_mat[0:6].sum(dim=0) / flat_steps + 1e-8)
            flat_dd_pct = w_dd_mat[0:6].max(dim=0).values
            flat_api = w_api_mat[0:6].sum(dim=0) / flat_steps
            flat_fills = w_fills_mat[0:6].sum(dim=0)
            flat_cancels = w_cancels_mat[0:6].sum(dim=0)
            np_fills = w_fills_mat[0:2].sum(dim=0) + w_fills_mat[4:6].sum(dim=0)

            flat_util_bp = 100.0 * (
                flat_ret_pct
                + 0.25 * flat_cvar_pct
                - flat_ulcer_pct
                - 0.35 * (flat_dd_pct * flat_dd_pct)
                - 0.25 * flat_api
            )

            uw_ret_pct = w_mtm_pct_mat[6:10].mean(dim=0)
            uw_worst_pct = w_mtm_pct_mat[6:10].min(dim=0).values
            uw_ulcer_pct = torch.sqrt(w_sq_dd_mat[6:10].sum(dim=0) / uw_steps + 1e-8)
            uw_dd_pct = w_dd_mat[6:10].max(dim=0).values
            uw_api = w_api_mat[6:10].sum(dim=0) / uw_steps

            uw_util_bp = 100.0 * (
                uw_ret_pct
                + 0.25 * uw_worst_pct
                - 0.80 * uw_ulcer_pct
                - 0.30 * (uw_dd_pct * uw_dd_pct)
                - 0.25 * uw_api
            )

            g_flat_u_mat = flat_util_bp.view(n_groups, n_budgets)
            g_uw_u_mat = uw_util_bp.view(n_groups, n_budgets)
            g_flat_ret_mat = flat_ret_pct.view(n_groups, n_budgets)
            g_uw_ret_mat = uw_ret_pct.view(n_groups, n_budgets)
            g_norm_mat = flat_norm_pct.view(n_groups, n_budgets)
            g_dump_mat = flat_dump_pct.view(n_groups, n_budgets)
            g_pump_mat = flat_pump_pct.view(n_groups, n_budgets)
            g_dd_mat = flat_dd_pct.view(n_groups, n_budgets)
            g_ulcer_mat = flat_ulcer_pct.view(n_groups, n_budgets)
            g_api_mat = flat_api.view(n_groups, n_budgets)
            g_fills_mat = flat_fills.view(n_groups, n_budgets)
            g_cancels_mat = flat_cancels.view(n_groups, n_budgets)
            g_np_fills_mat = np_fills.view(n_groups, n_budgets)

            flat_score_g = g_flat_u_mat.mean(dim=1) - 0.20 * g_flat_u_mat.std(dim=1)
            uw_score_g = g_uw_u_mat.mean(dim=1) - 0.20 * g_uw_u_mat.std(dim=1)
            flat_ret_g = g_flat_ret_mat.mean(dim=1)
            np_fills_g = g_np_fills_mat.mean(dim=1)

            pareto_fitness = compute_pareto_group_fitness(flat_score_g, uw_score_g, np_fills_g)
            g_prof_buds = (g_flat_ret_mat > 0.0).long().sum(dim=1)

            c0_fit = float(pareto_fitness[0].item())
            c0_flat_ret = float(flat_ret_g[0].item())
            sorted_g = torch.argsort(pareto_fitness, descending=True)
            best_g = int(sorted_g[0].item())

            b_fit = float(pareto_fitness[best_g].item())
            b_flat_u = float(flat_score_g[best_g].item())
            b_uw_u = float(uw_score_g[best_g].item())
            b_flat_ret = float(g_flat_ret_mat[best_g].mean().item())
            b_uw_ret = float(g_uw_ret_mat[best_g].mean().item())
            b_norm_pct = float(g_norm_mat[best_g].mean().item())
            b_dump_pct = float(g_dump_mat[best_g].mean().item())
            b_pump_pct = float(g_pump_mat[best_g].mean().item())
            b_dd_pct = float(g_dd_mat[best_g].mean().item())
            b_ulcer = float(g_ulcer_mat[best_g].mean().item())
            b_api = float(g_api_mat[best_g].mean().item())
            b_fills = float(g_fills_mat[best_g].mean().item())
            b_cancels = float(g_cancels_mat[best_g].mean().item())
            b_cnl_per_fill = b_cancels / max(b_fills, 0.5)
            b_eta_queue = 100.0 / (1.0 + max(0.0, b_cnl_per_fill - 1.5) / 12.0) if b_fills >= 0.5 else (100.0 if b_cancels <= 1.0 else 0.0)
            b_prof_buds = int(g_prof_buds[best_g].item())

            top_g_idx = sorted_g[:24]
            cand_w1 = (policy.w1[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_b1 = (policy.b1[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_wr1 = (policy.w_res1[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_br1 = (policy.b_res1[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_wr2 = (policy.w_res2[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_w2 = (policy.w2[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_b2 = (policy.b2[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_wrec = (policy.w_rec[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_brec = (policy.b_rec[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_gl = (policy.g_logits[top_g_idx].float() * nes_w.view(-1, 1)).sum(dim=0).clamp(-2.5, 2.0)

            cand_w1, cand_b1, cand_wr1, cand_br1, cand_wr2, cand_w2, cand_b2, cand_wrec, cand_brec = _batch_normalize_weights_l21(
                cand_w1, cand_b1, cand_wr1, cand_br1, cand_wr2, cand_w2, cand_b2, cand_wrec, cand_brec, policy.dtype
            )

            polyak_alpha = 0.20 * cosine_decay
            polyak_fast = min(0.45, 0.35 * cosine_decay)
            polyak_new = min(0.65, 0.55 * cosine_decay)
            hof_w1_new = (1.0 - polyak_alpha) * hof_state["w1"].float() + polyak_alpha * cand_w1.detach().cpu().float()
            for idx_s, idx_e in ((278, 288), (312, 318), (324, 380)):
                hof_w1_new[idx_s:idx_e, :] = (1.0 - polyak_fast) * hof_state["w1"][idx_s:idx_e, :].float() + polyak_fast * cand_w1[idx_s:idx_e, :].detach().cpu().float()
            hof_w1_new[388:444, :] = (1.0 - polyak_new) * hof_state["w1"][388:444, :].float() + polyak_new * cand_w1[388:444, :].detach().cpu().float()

            cand_polyak_state = {
                "w1": hof_w1_new.to(hof_state["w1"].dtype),
                "b1": ((1.0 - polyak_alpha) * hof_state["b1"].float() + polyak_alpha * cand_b1.detach().cpu().float()).to(hof_state["b1"].dtype),
                "w_res1": ((1.0 - polyak_alpha) * hof_state["w_res1"].float() + polyak_alpha * cand_wr1.detach().cpu().float()).to(hof_state["w_res1"].dtype),
                "b_res1": ((1.0 - polyak_alpha) * hof_state["b_res1"].float() + polyak_alpha * cand_br1.detach().cpu().float()).to(hof_state["b_res1"].dtype),
                "w_res2": ((1.0 - polyak_alpha) * hof_state["w_res2"].float() + polyak_alpha * cand_wr2.detach().cpu().float()).to(hof_state["w_res2"].dtype),
                "w2": ((1.0 - polyak_alpha) * hof_state["w2"].float() + polyak_alpha * cand_w2.detach().cpu().float()).to(hof_state["w2"].dtype),
                "b2": ((1.0 - polyak_alpha) * hof_state["b2"].float() + polyak_alpha * cand_b2.detach().cpu().float()).to(hof_state["b2"].dtype),
                "w_rec": ((1.0 - polyak_new) * hof_state["w_rec"].float() + polyak_new * cand_wrec.detach().cpu().float()).to(hof_state["w_rec"].dtype),
                "b_rec": ((1.0 - polyak_new) * hof_state["b_rec"].float() + polyak_new * cand_brec.detach().cpu().float()).to(hof_state["b_rec"].dtype),
            }
            cand_polyak_gl = (1.0 - polyak_fast) * hof_g_logit + polyak_fast * cand_gl.detach().cpu()

            if b_fit > c0_fit and b_fills >= 1.0:
                hof_state = cand_polyak_state
                hof_g_logit = cand_polyak_gl

            hof_updated = False
            if it % 3 == 0 or it == iterations or (b_fit > c0_fit + 0.35 and b_prof_buds >= 4):
                ls_alphas = [0.05, 0.12, 0.25, 0.45, 0.70, 1.00]
                ls_raw_states = []
                ls_gls = []
                ls_folded_candidates = []

                c_cpu = {
                    "w1": cand_w1.detach().cpu().float(),
                    "b1": cand_b1.detach().cpu().float(),
                    "w_res1": cand_wr1.detach().cpu().float(),
                    "b_res1": cand_br1.detach().cpu().float(),
                    "w_res2": cand_wr2.detach().cpu().float(),
                    "w2": cand_w2.detach().cpu().float(),
                    "b2": cand_b2.detach().cpu().float(),
                    "w_rec": cand_wrec.detach().cpu().float(),
                    "b_rec": cand_brec.detach().cpu().float(),
                }
                c_gl_cpu = cand_gl.detach().cpu().float()
                for a_ls in ls_alphas:
                    st_a = {k: (1.0 - a_ls) * best_val_state[k].float() + a_ls * c_cpu[k] for k in c_cpu}
                    gl_a = (1.0 - a_ls) * best_val_g_logit.float() + a_ls * c_gl_cpu
                    ls_raw_states.append(st_a)
                    ls_gls.append(gl_a)
                    ls_folded_candidates.append(policy.fold_gate_into_state(st_a, gl_a))

                ls_raw_states.append({k: v.float() for k, v in cand_polyak_state.items()})
                ls_gls.append(cand_polyak_gl.float())
                ls_folded_candidates.append(policy.fold_gate_into_state(cand_polyak_state, cand_polyak_gl))

                raw_best = {
                    "w1": policy.w1[best_g].detach().cpu().float(),
                    "b1": policy.b1[best_g].detach().cpu().float(),
                    "w_res1": policy.w_res1[best_g].detach().cpu().float(),
                    "b_res1": policy.b_res1[best_g].detach().cpu().float(),
                    "w_res2": policy.w_res2[best_g].detach().cpu().float(),
                    "w2": policy.w2[best_g].detach().cpu().float(),
                    "b2": policy.b2[best_g].detach().cpu().float(),
                    "w_rec": policy.w_rec[best_g].detach().cpu().float(),
                    "b_rec": policy.b_rec[best_g].detach().cpu().float(),
                }
                gl_best = policy.g_logits[best_g].detach().cpu().float()
                ls_raw_states.append(raw_best)
                ls_gls.append(gl_best)
                ls_folded_candidates.append(policy.fold_gate_into_state(raw_best, gl_best))

                val_res_list = evaluate_l22_dual_basket(
                    vec_env, bf16_runner, lob_fu_all, v7_all, roll_sell_30s_all, ls_folded_candidates, val_basket, win_len, tick_sz, device, True, True
                )
                for c_idx, v_res in enumerate(val_res_list):
                    c_pu = float(v_res["pareto_util_bp"])
                    c_cu = float(v_res["clean_util_bp"])
                    c_uu = float(v_res["uw_util_bp"])
                    c_cev = float(v_res["clean_ev"])
                    c_fills = float(v_res["clean_fills"])
                    c_act_fr = float(v_res["clean_act_frac"])
                    c_eta_q = float(v_res["clean_eta_queue_pct"])
                    c_cpf = float(v_res["clean_cnl_per_fill"])
                    # Строгий двухконтурный лексикографический Парето-критерий обновления чемпиона:
                    valid_active_mm = (c_cev > 0.003) and (c_fills >= 4.0)
                    strictly_dominates = (c_cu >= best_clean_util) and (c_uu >= best_uw_util) and ((c_cu > best_clean_util + 0.02) or (c_uu > best_uw_util + 0.05))
                    clean_axis_breakthrough = (c_pu > best_pareto_util + 0.03) and (c_cu > best_clean_util + 0.08) and (c_cev >= best_clean_ev - 0.0015) and (c_uu >= best_uw_util - 3.5)
                    if valid_active_mm and (strictly_dominates or clean_axis_breakthrough):
                        best_pareto_util = c_pu
                        best_clean_util = c_cu
                        best_uw_util = c_uu
                        best_clean_ev = c_cev
                        best_val_ev = float(v_res["realistic_ev"])
                        best_eta_queue = c_eta_q
                        best_cnl_per_fill = c_cpf
                        g_vals = (2.0 * torch.sigmoid(ls_gls[c_idx])).tolist()
                        best_val_folded = {k: v.clone() for k, v in ls_folded_candidates[c_idx].items()}
                        # Zero Gate-Folding Drift: ребейз folded-весов в эталон с обнулением g_logit
                        best_val_state = {k: v.to(hof_state[k].dtype).clone() for k, v in best_val_folded.items()}
                        best_val_g_logit = torch.zeros(N_BLOCKS_L21, dtype=torch.float32)
                        hof_state = {k: v.clone() for k, v in best_val_state.items()}
                        hof_g_logit = torch.zeros(N_BLOCKS_L21, dtype=torch.float32)
                        hof_updated = True
                        last_val_champ_it = it
                        ls_tag = f"alpha={ls_alphas[c_idx]:.2f}" if c_idx < len(ls_alphas) else ("polyak_hof" if c_idx == 6 else "best_elite")
                        log_msg(
                            f"  💎 [NEW L22.2 444D CHAMPION it={it} ({ls_tag})] ParetoUtil={c_pu:+.2f}bp "
                            f"[Clean18={c_cu:+.2f}bp (EV={c_cev:+.4f}%, Fills={c_fills:.1f}, Cnl/Fill={c_cpf:.1f}, η_queue={c_eta_q:.1f}%, Act={c_act_fr:.0%}) | "
                            f"UW4={c_uu:+.2f}bp (ΔMTM={v_res['uw_delta_ev']:+.4f}%)] | "
                            f"Clean[N:{v_res['clean_norm_pct']:+.3f}% D:{v_res['clean_dump_pct']:+.3f}% P:{v_res['clean_pump_pct']:+.3f}% DD:{v_res['clean_dd_pct']:.2f}%] | "
                            f"Gates(CV4={g_vals[6]:.2f}, Acct={g_vals[12]:.2f}, Queue={g_vals[13]:.2f}, SlotProg={g_vals[15]:.2f})"
                        )

            if it % 20 == 0 and not hof_updated:
                hof_state = {k: v.clone() for k, v in best_val_state.items()}
                hof_g_logit = best_val_g_logit.clone()

            policy.evolve_gated_groups(
                pareto_fitness,
                hof_state=hof_state,
                hof_g_logit=hof_g_logit,
                elite_frac=0.03,
                progress=cos_w,
            )

            g_dict = {name: round(float(2.0 * torch.sigmoid(best_val_g_logit[bi]).item()), 4) for bi, (name, _, _, _) in enumerate(SEMANTIC_BLOCKS_L21)}
            rec = {
                "iteration": it,
                "hof_updated": hof_updated,
                "best_pareto_util_bp": round(best_pareto_util, 3),
                "best_clean_util_bp": round(best_clean_util, 3),
                "best_uw_util_bp": round(best_uw_util, 3),
                "best_clean_ev_pct": round(best_clean_ev, 4),
                "best_val_ev_pct": round(best_val_ev, 4),
                "best_eta_queue_pct": round(best_eta_queue, 2),
                "best_cnl_per_fill": round(best_cnl_per_fill, 2),
                "batch_flat_ret_pct": round(b_flat_ret, 4),
                "batch_uw_ret_pct": round(b_uw_ret, 4),
                "batch_flat_fills": round(b_fills, 2),
                "batch_flat_cancels": round(b_cancels, 2),
                "batch_eta_queue_pct": round(b_eta_queue, 2),
                "champ_flat_ret_pct": round(c0_flat_ret, 4),
                "ulcer_dd_pct": round(b_ulcer, 4),
                "real_dd_pct": round(b_dd_pct, 4),
                "mean_mut_load": round(b_api, 4),
                "block_gates": g_dict,
            }
            history.append(rec)
            summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            dt_ep = time.time() - t0
            star = "★" if hof_updated else "·"
            log_msg(
                f"🎯 L22.2 [{it}/{iterations}]{star} ({b_prof_buds}/6+) | Flat: {b_flat_ret:+.3f}% ({b_flat_u:+.1f}bp, Fills={b_fills:.1f}, Cnl={b_cancels:.1f}, η_q={b_eta_queue:.0f}%) "
                f"[N:{b_norm_pct:+.2f}% D:{b_dump_pct:+.2f}% P:{b_pump_pct:+.2f}%] | UW: {b_uw_ret:+.3f}% ({b_uw_u:+.1f}bp) | "
                f"DD: {b_dd_pct:.2f}% | Champ EV={best_clean_ev:+.4f}% η_q={best_eta_queue:.1f}% ({dt_ep:.2f}с)"
            )

            if hof_updated or it == iterations or it == start_iter:
                ckpt_payload = {
                    "lesson": 22,
                    "phase": 2,
                    "iteration": it,
                    "release_name": f"L22P2_{it:04d}_slot_queue_444d",
                    "n_slots": 24,
                    "allow_scratch_stop_asks": True,
                    "use_cross_venue_v1": True,
                    "decoupled_market_backbone": True,
                    "selective_front_cancel": True,
                    "pareto_util_bp": best_pareto_util,
                    "clean_util_bp": best_clean_util,
                    "uw_util_bp": best_uw_util,
                    "clean_ev_pct": best_clean_ev,
                    "val_ev_pct": best_val_ev,
                    "eta_queue_pct": best_eta_queue,
                    "cnl_per_fill": best_cnl_per_fill,
                    "block_gates": g_dict,
                    "backbone_state": backbone.state_dict(),
                    "w1": best_val_folded["w1"],
                    "b1": best_val_folded["b1"],
                    "w_res1": best_val_folded["w_res1"],
                    "b_res1": best_val_folded["b_res1"],
                    "w_res2": best_val_folded["w_res2"],
                    "w2": best_val_folded["w2"],
                    "b2": best_val_folded["b2"],
                    "w_rec": best_val_folded["w_rec"],
                    "b_rec": best_val_folded["b_rec"],
                }
                torch.save(ckpt_payload, latest_ckpt)
                torch.save(ckpt_payload, SAVE_DIR / f"lesson_22_p2_champ_iter_{it:04d}.pt")

            if early_stop and (it - last_val_champ_it) >= 350 and it >= 600:
                log_msg(f"🛑 [EARLY STOP L22.2] {it - last_val_champ_it} эпох без Парето-улучшения (лучший на it={last_val_champ_it}).")
                break

    log_msg(
        f"🏆 УРОК 22.2 (444D) ЗАВЕРШЕН! Финальный ParetoUtil={best_pareto_util:+.2f}bp | "
        f"Clean18_Util={best_clean_util:+.2f}bp (CleanEV={best_clean_ev:+.4f}%, Cnl/Fill={best_cnl_per_fill:.1f}, η_queue={best_eta_queue:.1f}%) | "
        f"UW4_Util={best_uw_util:+.2f}bp"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--groups", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=2026222)
    ap.add_argument("--early-stop", action="store_true")
    args = ap.parse_args()
    run_lesson_22(iterations=args.iters, n_groups=args.groups, early_stop=args.early_stop, rng_seed=args.seed)
