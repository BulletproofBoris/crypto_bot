#!/usr/bin/env python3
"""
_tools/lesson_26_endogenous_nonlinear_trainer.py
=============================================================================
Урок 26: Самоадаптивное обучение без эмпирических ограничений:
  - Per-Block Learnable RMSNorm (разблокировка BigONE Swarm 36D наравне со стволом),
  - Эндогенная нелинейная геометрия слотов (до 25% от b1) и свободный якорь Ask,
  - Непрерывный спектр стартовых поз rho_0 in [0, 0.80] + Adversarial Window Sampling,
  - Entropic Log-Wealth Growth + 5-квантильное Парето-доминирование (Q15..Q85).
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
from _tools.fused_l26_endogenous_tick import (
    MKT_IDX_L26,
    AG_IDX_L26,
    precompute_window_market_z370,
    normalize_market_z370_per_block,
    fused_l26_endogenous_tick,
    evaluate_l26_basket,
)

CACHE_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
RELEASE_CHAMP_PATH = BASE_DIR / "models/prl/release_champion_v1.pt"
SAVE_DIR = BASE_DIR / "models/prl/lesson_26_endogenous_nonlinear"
LATEST_CHAMP_PATH = SAVE_DIR / "lesson_26_champ_latest.pt"
RESUME_CHAMP_PATH = SAVE_DIR / "lesson_26_champ_resume.pt"
SOUP_CHAMP_PATH = SAVE_DIR / "l26_soup_top2.pt"
LOG_PATH = BASE_DIR / "data/logs/lesson_26_train.log"

MSK_TZ = timezone(timedelta(hours=3))
VAL_START_TS = datetime(2026, 11, 11, 22, 0, tzinfo=MSK_TZ).timestamp()
BLIND_START_TS = datetime(2026, 11, 12, 20, 30, tzinfo=MSK_TZ).timestamp()

SEMANTIC_BLOCKS_L26 = [
    ("Z_Backbone_GRU", 0, 256, 0.50),
    ("Guard_Risk_Heads", 256, 260, 0.85),
    ("Short_Triad_Swarm_ST", 260, 278, 1.00),
    ("CrossVenue_5s_LeadLag", 278, 288, 1.60),
    ("Macro_Swarm_ST_60_120s", 288, 306, 0.70),
    ("Carousel_Hazard_Realism", 306, 312, 0.85),
    ("CrossVenue_FirstOrder_6D", 312, 318, 1.75),
    ("LiqRatio_LIFO_Edge", 318, 324, 1.45),
    ("Active_24Slot_Weights", 324, 348, 1.65),
    ("Swarm_5s_Velocity", 348, 356, 1.45),
    ("Active_24Slot_Distances", 356, 380, 1.35),
    ("Fast_5s_LOB_Impulse", 380, 388, 1.55),
    ("Direct_Acct_State_8D", 388, 396, 1.85),
    ("FIFO_Queue_Ahead_8D", 396, 404, 1.95),
    ("Policy_Micro_Memory_16D", 404, 420, 1.75),
    ("PerSlot_FIFO_Progress_24D", 420, 444, 2.00),
    ("BigONE_Short_Swarm_18D", 444, 462, 2.20),
    ("BigONE_Macro_Swarm_18D", 462, 480, 1.85),
]
N_BLOCKS_L26 = len(SEMANTIC_BLOCKS_L26)


def log_msg(msg: str) -> None:
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts_str}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expand_block_gates_to_480(g_logits: torch.Tensor) -> torch.Tensor:
    gates_b = 2.0 * torch.sigmoid(g_logits.float())
    parts = [gates_b[..., idx : idx + 1].expand(*gates_b.shape[:-1], e - s) for idx, (_, s, e, _) in enumerate(SEMANTIC_BLOCKS_L26)]
    return torch.cat(parts, dim=-1)


def _normalize_weights_l26(state: dict, dtype: torch.dtype) -> dict:
    out = {}
    w1_f = state["w1"].float()
    # По-блочный мягкий потолок нормы предотвращает подавление новых блоков старым 256D стволом:
    w1_parts = []
    for _, s, e, _ in SEMANTIC_BLOCKS_L26:
        wb = w1_f[..., s:e, :]
        cap_b = 6.5 * math.sqrt(float(e - s))
        nb = torch.linalg.vector_norm(wb, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        w1_parts.append(wb * torch.clamp(cap_b / nb, max=1.0))
    out["w1"] = torch.cat(w1_parts, dim=-2).to(dtype)
    out["b1"] = torch.clamp(state["b1"].float(), -2.2, 2.2).to(dtype)

    for k, cap in (("w_res1", 65.0), ("w_res2", 40.0), ("w2", 62.0), ("w_rec", 30.0), ("w_reg1", 24.0), ("w_reg2", 14.0)):
        wf = state[k].float()
        nw = torch.linalg.vector_norm(wf, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        out[k] = (wf * torch.clamp(cap / nw, max=1.0)).to(dtype)

    out["b_res1"] = torch.clamp(state["b_res1"].float(), -2.0, 2.0).to(dtype)
    out["b2"] = torch.clamp(state["b2"].float(), -3.5, 3.0).to(dtype)
    out["b_rec"] = torch.clamp(state["b_rec"].float(), -2.5, 2.5).to(dtype)
    out["b_reg1_b"] = torch.clamp(state["b_reg1_b"].float(), -2.5, 2.5).to(dtype)
    out["b_reg1_a"] = torch.clamp(state["b_reg1_a"].float(), -2.5, 2.5).to(dtype)
    out["b_reg2_b"] = torch.clamp(state["b_reg2_b"].float(), -3.5, 3.0).to(dtype)
    out["b_reg2_a"] = torch.clamp(state["b_reg2_a"].float(), -3.5, 3.0).to(dtype)
    return out


class EndogenousNonlinearPolicyL26:
    """Популяционная 480D политика Урока 26: Макро-Стратег (44D) + Связанный Нейро-Регулятор (22->48->2)."""

    PARAM_KEYS = (
        "w1", "b1", "w_res1", "b_res1", "w_res2", "w2", "b2",
        "w_rec", "b_rec", "w_reg1", "b_reg1_b", "b_reg1_a", "w_reg2", "b_reg2_b", "b_reg2_a",
    )

    def __init__(self, backbone: MultiTaskLOBNet, n_groups: int = 1024, n_budgets: int = 6, device: torch.device = torch.device("cuda")):
        self.backbone = backbone
        self.n_groups = n_groups
        self.n_budgets = n_budgets
        self.pop_size = n_groups * n_budgets
        self.in_dim = 480
        self.h_dim = 256
        self.h1_full_dim = 384
        self.out_dim = 44
        self.reg_in_dim = 22
        self.reg_h_dim = 48
        self.device = device
        self.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        self.w1 = torch.empty((n_groups, 480, 384), dtype=self.dtype, device=device)
        self.b1 = torch.empty((n_groups, 1, 384), dtype=self.dtype, device=device)
        self.w_res1 = torch.empty((n_groups, 256, 128), dtype=self.dtype, device=device)
        self.b_res1 = torch.empty((n_groups, 1, 128), dtype=self.dtype, device=device)
        self.w_res2 = torch.empty((n_groups, 64, 256), dtype=self.dtype, device=device)
        self.w2 = torch.empty((n_groups, 256, 44), dtype=self.dtype, device=device)
        self.b2 = torch.empty((n_groups, 1, 44), dtype=self.dtype, device=device)
        self.w_rec = torch.empty((n_groups, 256, 32), dtype=self.dtype, device=device)
        self.b_rec = torch.empty((n_groups, 1, 32), dtype=self.dtype, device=device)

        self.w_reg1 = torch.empty((n_groups, 22, 48), dtype=self.dtype, device=device)
        self.b_reg1_b = torch.empty((n_groups, 1, 48), dtype=self.dtype, device=device)
        self.b_reg1_a = torch.empty((n_groups, 1, 48), dtype=self.dtype, device=device)
        self.w_reg2 = torch.empty((n_groups, 48, 2), dtype=self.dtype, device=device)
        self.b_reg2_b = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)
        self.b_reg2_a = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)
        self.g_logits = torch.zeros((n_groups, N_BLOCKS_L26), dtype=torch.float32, device=device)

        mut_profile = torch.ones((1, 480, 1), dtype=self.dtype, device=device)
        for _, s, e, m_scale in SEMANTIC_BLOCKS_L26:
            mut_profile[:, s:e, :] = float(m_scale)
        self.mut_profile = mut_profile

    def load_from_checkpoint(self, ck: dict, emp_block_rms: torch.Tensor | None = None) -> tuple[dict, torch.Tensor]:
        if "backbone_state" in ck:
            self.backbone.load_state_dict(ck["backbone_state"], strict=False)
        is_l26 = int(ck.get("lesson", 0)) >= 26 and int(ck["w2"].shape[-1]) == 44
        if is_l26:
            state = {k: ck[k].to(dtype=self.dtype, device=self.device) for k in self.PARAM_KEYS}
            for k in ("b1", "b_res1", "b2", "b_rec", "b_reg1_b", "b_reg1_a", "b_reg2_b", "b_reg2_a"):
                state[k] = state[k].view(1, -1)
            gl = torch.zeros(N_BLOCKS_L26, dtype=torch.float32)
            cpu_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            self.sample_population(cpu_state, gl, phase=1, sigma_scale=1.0)
            return cpu_state, gl

        # Аналитический RMS-компенсированный перенос из чемпиона Урока 25 (480D -> 35D) в L26 (480D -> 44D):
        w1_old = ck["w1"].float().clone()
        if emp_block_rms is not None and emp_block_rms.numel() == N_BLOCKS_L26:
            for bi, (_, s, e, _) in enumerate(SEMANTIC_BLOCKS_L26):
                w1_old[s:e] = w1_old[s:e] * float(max(0.08, float(emp_block_rms[bi].item())))
        # Поднимаем амплитуду BO_Swarm_36D [444:480] до рабочего паритета (~0.11 RMS) без поломки стартового знака:
        bo_rms = float(torch.sqrt(torch.mean(w1_old[444:480] ** 2) + 1e-8).item())
        if bo_rms < 0.08:
            w1_old[444:480] = w1_old[444:480] * min(4.0, 0.095 / max(bo_rms, 1e-4))

        w2_old = ck["w2"].float()
        b2_old = ck["b2"].float().view(-1)
        w2_new = torch.zeros((256, 44), dtype=torch.float32)
        b2_new = torch.zeros(44, dtype=torch.float32)
        w2_new[:, :35] = w2_old[:, :35]
        b2_new[:35] = b2_old[:35]
        # [35] раздельная температура Ask-слотов наследует tau_slot L25:
        w2_new[:, 35] = w2_old[:, 34]
        b2_new[35] = b2_old[34]
        # [36] предел проскальзывания Taker ~3.5%, [37..40] стартовая кривая ~3.8% (совпадает с 5 тиками при $1.40!):
        b2_new[36] = -1.35
        b2_new[37] = 0.0
        b2_new[38] = -1.20
        b2_new[39] = -2.20
        b2_new[40] = -0.35
        # [41..42] стартовый якорь Ask около avg_entry + +0.45% (плавно разблокируется в ходе эволюции!):
        b2_new[41] = 1.60
        b2_new[42] = 0.12

        torch.manual_seed(2026261)
        w_reg1_new = torch.randn((22, 48), dtype=torch.float32) * 0.035
        w_reg1_new[:16, :32] = ck["w_reg1"].float()
        b_reg1_base = torch.zeros((1, 48), dtype=torch.float32)
        b_reg1_base[:, :32] = ck["b_reg1"].float().view(1, 32)
        w_reg2_new = torch.randn((48, 2), dtype=torch.float32) * 0.025
        w_reg2_new[:32, :] = ck["w_reg2"].float()
        b_reg2_base = ck["b_reg2"].float().view(1, 2)

        state = {
            "w1": w1_old.to(dtype=self.dtype, device=self.device),
            "b1": ck["b1"].view(1, 384).to(dtype=self.dtype, device=self.device),
            "w_res1": ck["w_res1"].to(dtype=self.dtype, device=self.device),
            "b_res1": ck["b_res1"].view(1, 128).to(dtype=self.dtype, device=self.device),
            "w_res2": ck["w_res2"].to(dtype=self.dtype, device=self.device),
            "w2": w2_new.to(dtype=self.dtype, device=self.device),
            "b2": b2_new.view(1, 44).to(dtype=self.dtype, device=self.device),
            "w_rec": ck["w_rec"].to(dtype=self.dtype, device=self.device),
            "b_rec": ck["b_rec"].view(1, 32).to(dtype=self.dtype, device=self.device),
            "w_reg1": w_reg1_new.to(dtype=self.dtype, device=self.device),
            "b_reg1_b": b_reg1_base.clone().to(dtype=self.dtype, device=self.device),
            "b_reg1_a": b_reg1_base.clone().to(dtype=self.dtype, device=self.device),
            "w_reg2": w_reg2_new.to(dtype=self.dtype, device=self.device),
            "b_reg2_b": b_reg2_base.clone().to(dtype=self.dtype, device=self.device),
            "b_reg2_a": b_reg2_base.clone().to(dtype=self.dtype, device=self.device),
        }
        state = _normalize_weights_l26(state, self.dtype)
        gl = torch.zeros(N_BLOCKS_L26, dtype=torch.float32)
        cpu_state = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.sample_population(cpu_state, gl, phase=1, sigma_scale=1.0)
        return cpu_state, gl

    def effective_group_w1(self) -> torch.Tensor:
        g480 = expand_block_gates_to_480(self.g_logits).to(self.dtype).unsqueeze(-1)
        return self.w1 * g480

    def fold_gate_into_state(self, state: dict, g_logit: torch.Tensor) -> dict:
        out = {k: v.clone() for k, v in state.items()}
        g480 = expand_block_gates_to_480(g_logit.view(1, -1)).squeeze(0).to(out["w1"].dtype).unsqueeze(-1)
        out["w1"] = (out["w1"].to(g480.device) * g480).detach().cpu()
        return out

    def sample_population(self, center_state: dict, center_g_logit: torch.Tensor, phase: int, sigma_scale: float) -> None:
        base_gpu = {k: v.to(device=self.device, dtype=self.dtype) for k, v in center_state.items()}
        gl_gpu = center_g_logit.to(device=self.device, dtype=torch.float32)

        for k in self.PARAM_KEYS:
            self.__dict__[k][0].copy_(base_gpu[k])
            self.__dict__[k][1].copy_(base_gpu[k])
        self.g_logits[0].copy_(gl_gpu)
        self.g_logits[1].copy_(gl_gpu)

        half = (self.n_groups - 2) // 2
        sig_base = (0.0042 if phase == 1 else 0.0062) * float(sigma_scale)

        def _anti(shape: tuple[int, ...], scale_f: float) -> torch.Tensor:
            eps = torch.randn((half, *shape), device=self.device, dtype=torch.float32) * scale_f
            return torch.stack([eps, -eps], dim=1).reshape(2 * half, *shape)

        dw1 = _anti((480, 384), sig_base).to(self.dtype) * self.mut_profile
        if phase == 1:
            # В Фазе 1 основное внимание: калибровка BO_Swarm_36D, новой нелинейной геометрии w2[35:44] и 22D Регулятора
            dw1[:, :256, :] *= 0.20
        self.w1[2 : 2 + 2 * half].copy_(base_gpu["w1"].unsqueeze(0) + dw1)
        self.b1[2 : 2 + 2 * half].copy_(base_gpu["b1"].unsqueeze(0) + _anti((1, 384), sig_base * 0.6).to(self.dtype))

        res_mult = 0.15 if phase == 1 else 1.0
        self.w_res1[2 : 2 + 2 * half].copy_(base_gpu["w_res1"].unsqueeze(0) + _anti((256, 128), sig_base * res_mult).to(self.dtype))
        self.b_res1[2 : 2 + 2 * half].copy_(base_gpu["b_res1"].unsqueeze(0) + _anti((1, 128), sig_base * 0.5 * res_mult).to(self.dtype))
        self.w_res2[2 : 2 + 2 * half].copy_(base_gpu["w_res2"].unsqueeze(0) + _anti((64, 256), sig_base * res_mult).to(self.dtype))

        dw2 = _anti((256, 44), sig_base * 1.65)
        db2 = _anti((1, 44), sig_base * 2.10)
        dw2[:, :, 35:44] *= 1.85
        db2[:, :, 35:44] *= 2.20
        self.w2[2 : 2 + 2 * half].copy_(base_gpu["w2"].unsqueeze(0) + dw2.to(self.dtype))
        self.b2[2 : 2 + 2 * half].copy_(base_gpu["b2"].unsqueeze(0) + db2.to(self.dtype))

        self.w_rec[2 : 2 + 2 * half].copy_(base_gpu["w_rec"].unsqueeze(0) + _anti((256, 32), sig_base * 1.2).to(self.dtype))
        self.b_rec[2 : 2 + 2 * half].copy_(base_gpu["b_rec"].unsqueeze(0) + _anti((1, 32), sig_base * 0.9).to(self.dtype))

        reg_sig = (0.018 if phase == 1 else 0.012) * float(sigma_scale)
        self.w_reg1[2 : 2 + 2 * half].copy_(base_gpu["w_reg1"].unsqueeze(0) + _anti((22, 48), reg_sig).to(self.dtype))
        self.b_reg1_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_b"].unsqueeze(0) + _anti((1, 48), reg_sig * 0.85).to(self.dtype))
        self.b_reg1_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg1_a"].unsqueeze(0) + _anti((1, 48), reg_sig * 0.85).to(self.dtype))
        self.w_reg2[2 : 2 + 2 * half].copy_(base_gpu["w_reg2"].unsqueeze(0) + _anti((48, 2), reg_sig * 0.90).to(self.dtype))
        self.b_reg2_b[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_b"].unsqueeze(0) + _anti((1, 2), reg_sig * 1.15).to(self.dtype))
        self.b_reg2_a[2 : 2 + 2 * half].copy_(base_gpu["b_reg2_a"].unsqueeze(0) + _anti((1, 2), reg_sig * 1.15).to(self.dtype))

        d_gl = _anti((N_BLOCKS_L26,), (0.06 if phase == 1 else 0.08) * float(sigma_scale))
        self.g_logits[2 : 2 + 2 * half].copy_(torch.clamp(gl_gpu.unsqueeze(0) + d_gl, -2.5, 2.0))


def compute_endogenous_5d_pareto_fitness(
    traj_growth_wb_g: torch.Tensor,  # [n_groups, W * n_budgets] в базисных пунктах лог-роста
    val_mse_g: torch.Tensor,         # [n_groups] ошибка самокалибровки критика V_risk
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Чисто нелинейное многоквантильное Парето-ранжирование без единой ручной квоты сделок или линейного штрафа:
      1) Entropic Log-Growth (критерий Келли-Пратта с самоадаптивным масштабом дисперсии популяции),
      2) Q15 (устойчивость нижнего хвоста без катастроф),
      3) Q50 (медианный органический рост),
      4) Q85 (способность забирать сильные движения и спреды — естественный антидот против простоя в кэше!),
      5) Отрицательная ошибка внутреннего предсказателя потенциала V_risk (самоорганизация латентного ядра h2).
    """
    G, M = traj_growth_wb_g.shape
    pop_iqr = torch.clamp(
        torch.quantile(traj_growth_wb_g.median(dim=1).values, 0.75) - torch.quantile(traj_growth_wb_g.median(dim=1).values, 0.25),
        min=4.0,
        max=80.0,
    )
    beta_adapt = 1.25 / pop_iqr  # самокалибрующийся коэффициент неприятия риска от масштаба текущего рынка!
    ent_g = (-torch.logsumexp(-beta_adapt * traj_growth_wb_g, dim=1) + math.log(float(M))) / beta_adapt

    qs = torch.quantile(traj_growth_wb_g, torch.tensor([0.15, 0.50, 0.85], device=traj_growth_wb_g.device), dim=1)  # [3, G]
    q15_g, q50_g, q85_g = qs[0], qs[1], qs[2]
    crit_score_g = -torch.log1p(torch.clamp(val_mse_g * 1e4, min=0.0))

    # Нормированные ранги [0..1] по всем 5 нелинейным осям:
    def _rank01(v: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.argsort(v)).float() / float(max(1, G - 1))

    R = torch.stack([_rank01(ent_g), _rank01(q15_g), _rank01(q50_g), _rank01(q85_g), _rank01(crit_score_g)], dim=1)  # [G, 5]
    # 4 основные финансовые оси для счёта строгого Парето-доминирования [G, G]:
    F4 = R[:, :4]
    dom_by_j = (F4.unsqueeze(0) >= F4.unsqueeze(1)).all(dim=-1) & (F4.unsqueeze(0) > F4.unsqueeze(1) + 1e-5).any(dim=-1)
    dom_cnt = dom_by_j.float().sum(dim=1)

    pareto_front_term = -torch.log1p(dom_cnt)
    # Гладкий нелинейный гармонический/Чебышевский агрегатор квантилей + малый вклад самокалибровки критика:
    soft_min_4 = -0.25 * torch.logsumexp(-4.0 * F4, dim=1)
    mean_4 = F4.mean(dim=1)
    fitness = pareto_front_term + 1.40 * soft_min_4 + 1.10 * mean_4 + 0.12 * R[:, 4]
    return fitness, ent_g


def _save_l26_checkpoint(
    path: Path,
    base_ck: dict,
    folded_state: dict,
    gate_logit: torch.Tensor,
    it: int,
    phase: int,
    release_name: str,
    metrics: dict,
) -> None:
    ck_out = dict(base_ck)
    ck_out.update({
        "lesson": 26,
        "phase": int(phase),
        "arch_version": "l26_endogenous_nonlinear_480d",
        "iteration": int(it),
        "n_slots": 24,
        "in_dim": 480,
        "out_dim": 44,
        "reg_in_dim": 22,
        "reg_h_dim": 48,
        "use_cross_venue_v1": True,
        "use_bo_swarm_36d": True,
        "use_per_slot_regulator": True,
        "use_sparse_price_matched_lob": True,
        "use_endogenous_nonlinear_grid": True,
        "decoupled_market_backbone": True,
        "release_name": release_name,
        "gate_logit": gate_logit.detach().cpu().clone(),
        "entropic_growth_bp": float(metrics["entropic_growth_bp"]),
        "q15_bp": float(metrics["q15_bp"]),
        "q50_bp": float(metrics["q50_bp"]),
        "q85_bp": float(metrics["q85_bp"]),
        "full_ev_pct": float(metrics["full_ev_pct"]),
        "clean_ev_pct": float(metrics["clean_ev_pct"]),
        "inv_delta_ev_pct": float(metrics["inv_delta_ev_pct"]),
        "dd_pct": float(metrics["dd_pct"]),
        "cnl_per_fill": float(metrics["cnl_per_fill"]),
        "eta_queue_pct": float(metrics["eta_queue_pct"]),
    })
    for k in EndogenousNonlinearPolicyL26.PARAM_KEYS:
        ck_out[k] = folded_state[k].detach().cpu().clone()
    torch.save(ck_out, path)


def run_lesson_26(
    iterations: int = 560,
    phase1_iters: int = 70,
    phase2_iters: int = 260,
    phase3_iters: int = 440,
    plateau_patience: int = 80,
    plateau_min_delta_bp: float = 0.25,
    n_groups: int = 1024,
    win_len: int = 2400,
    rng_seed: int = 2026260,
    smoke_test: bool = False,
) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if smoke_test:
        n_groups = 32
        win_len = 240
        iterations = 2
        phase1_iters = 1
        phase2_iters = 2
    n_budgets = 6
    pop_size = n_groups * n_budgets

    log_msg("=" * 118)
    log_msg(
        f"🚀 УРОК 26 (ENDOGENOUS NONLINEAR GEOMETRY + SPARSE LOB + ENTROPIC 5D PARETO | WIN={win_len}s): "
        f"MAX_ITERS={iterations} (P1={phase1_iters}, P2={phase2_iters}, P3={phase3_iters}, PLATEAU_PATIENCE={plateau_patience}) | "
        f"12 ОКОН x {pop_size} = {12 * pop_size:,} СРЕД"
    )
    log_msg("=" * 118)

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

    policy = EndogenousNonlinearPolicyL26(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, device=device)
    v7_all = v7_all.to(dtype=policy.dtype)
    bo36_all = bo36_t.to(dtype=policy.dtype)
    del bo36_t

    summary_path = SAVE_DIR / "lesson_26_summary.json"
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

    # Оценка эмпирических RMS 18 блоков строго на срезе TRAIN для точной компенсации при Warm-Start:
    with torch.no_grad():
        cal_idx = torch.linspace(600, max(700, train_hi - 1500), steps=8, dtype=torch.long, device=device)
        cal_t = cal_idx.unsqueeze(0) + torch.arange(min(300, win_len - 1), dtype=torch.long, device=device).unsqueeze(1)
        z_cal_370 = precompute_window_market_z370(
            vec_env.flow_market[cal_t].contiguous(),
            lob_fu_all[cal_t].contiguous(),
            v7_all[cal_t].contiguous(),
            bo36_all[cal_t].contiguous(),
            bf16_runner,
        )
        _, mkt_rms_11 = normalize_market_z370_per_block(z_cal_370)
        del z_cal_370
        emp_rms_18 = torch.ones(N_BLOCKS_L26, dtype=torch.float32)
        mkt_map = [0, 1, 2, 3, 4, 5, 6, 9, 11, 16, 17]
        for j_m, b_i in enumerate(mkt_map):
            emp_rms_18[b_i] = mkt_rms_11[j_m]
        for b_i, r_ag in ((7, 0.45), (8, 0.22), (10, 0.30), (12, 0.55), (13, 0.55), (14, 0.35), (15, 0.85)):
            emp_rms_18[b_i] = r_ag

    hof_state, hof_g_logit = policy.load_from_checkpoint(base_ck, emp_block_rms=emp_rms_18)
    start_iter = (int(base_ck.get("iteration", 0)) + 1) if (int(base_ck.get("lesson", 25)) == 26 and not smoke_test) else 1
    if start_iter > 1 and history:
        history = [r for r in history if int(r.get("iteration", 0)) < start_iter]

    # Неразмеченная хронологическая проверочная корзина (24 точки по всему VAL с непрерывным спектром начальных поз!):
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
    init_val_res = evaluate_l26_basket(
        vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
        [base_folded] * (2 if smoke_test else 6), val_basket[:6] if smoke_test else val_basket, win_len, tick_sz, device, true_latency=True,
    )[0]
    dt_v0 = time.time() - t_v0

    best_ent_bp = float(init_val_res["entropic_growth_bp"])
    best_val_state = {k: v.clone() for k, v in hof_state.items()}
    best_val_g_logit = hof_g_logit.clone()
    best_val_folded = {k: v.clone() for k, v in base_folded.items()}
    last_sig_improve_it = max(0, start_iter - 1)
    sig_anchor_ent = float(best_ent_bp)

    log_msg(
        f"⚖️ СТАРТОВАЯ БАЗА L26 (compiled {dt_v0:.1f}с): EntGrowth={best_ent_bp:+.2f}bp "
        f"[Q15:{init_val_res['q15_bp']:+.1f}bp Q50:{init_val_res['q50_bp']:+.1f}bp Q85:{init_val_res['q85_bp']:+.1f}bp] | "
        f"CleanEV={init_val_res['clean_ev_pct']:+.4f}% InvEV={init_val_res['inv_delta_ev_pct']:+.4f}% DD={init_val_res['dd_pct']:.2f}% | "
        f"Fills={init_val_res['fills']:.1f} Cnl={init_val_res['cancels']:.1f} Cnl/F={init_val_res['cnl_per_fill']:.2f} η_q={init_val_res['eta_queue_pct']:.1f}%"
    )
    if start_iter == 1 and not smoke_test:
        _save_l26_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, 0, 1, "L26_WARMSTART_0000", init_val_res)

    budget_tiers = torch.tensor([100.0, 300.0, 625.0, 1000.0, 1500.0, 2500.0], device=device)
    env_budgets = budget_tiers.unsqueeze(0).expand(n_groups, n_budgets).reshape(pop_size)
    rng = np.random.default_rng(rng_seed)
    for _ in range(1, start_iter):
        for _k in range(16):
            _ = rng.random()

    # Самоадаптивные веса информативности/сложности тренировочных окон (стартуют равномерно, без ручной разметки!):
    win_difficulty = np.ones(len(train_grid), dtype=np.float64)

    W_par = 4 if smoke_test else 12
    N_par = W_par * pop_size
    S_slots = 12
    budget_16 = env_budgets.unsqueeze(0).expand(W_par, pop_size).reshape(N_par).contiguous()
    nes_w = torch.softmax(-torch.arange(min(24, n_groups), dtype=torch.float32, device=device) * 0.25, dim=0)
    mkt_idx_t = torch.tensor(MKT_IDX_L26, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_L26, dtype=torch.long, device=device)
    echelon_k01 = torch.tensor([0.0, 0.0, 0.12, 0.12, 0.26, 0.26, 0.44, 0.44, 0.68, 0.68, 1.0, 1.0], dtype=torch.float32, device=device)
    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    chunk_sz = 100

    for it in range(start_iter, iterations + 1):
        t0 = time.time()
        if it <= phase1_iters:
            phase = 1
            p_prog = float(it - 1) / max(1.0, float(phase1_iters - 1))
            cosine_decay = 0.30 + 0.70 * (0.5 * (1.0 + math.cos(math.pi * p_prog)))
        elif it <= phase2_iters:
            phase = 2
            p_prog = float(it - phase1_iters - 1) / max(1.0, float(phase2_iters - phase1_iters - 1))
            cosine_decay = 0.26 + 0.74 * (0.5 * (1.0 + math.cos(math.pi * p_prog)))
        elif it <= phase3_iters:
            phase = 3
            p_prog = float(it - phase2_iters - 1) / max(1.0, float(phase3_iters - phase2_iters - 1))
            cosine_decay = 0.15 + 0.28 * (0.5 * (1.0 + math.cos(math.pi * p_prog)))
        else:
            phase = 4
            p_prog = float(it - phase3_iters - 1) / max(1.0, float(iterations - phase3_iters - 1))
            cosine_decay = 0.10 + 0.16 * (0.5 * (1.0 + math.cos(math.pi * p_prog)))

        if it in (phase1_iters + 1, phase2_iters + 1, phase3_iters + 1) and not smoke_test:
            log_msg(f"🔓 ПЕРЕХОД К ФАЗЕ {phase}: перезапуск центра популяции из лучшего валидационного чемпиона ({best_ent_bp:+.2f}bp)!")
            hof_state = {k: v.clone() for k, v in best_val_state.items()}
            hof_g_logit = best_val_g_logit.clone()

        policy.sample_population(hof_state, hof_g_logit, phase=phase, sigma_scale=cosine_decay)

        # Сэмплирование W_par окон: 50% по самоадаптивной сложности популяции + 50% равномерный покров истории:
        probs_d = win_difficulty / max(1e-9, win_difficulty.sum())
        mix_p = 0.50 * probs_d + 0.50 * (1.0 / len(train_grid))
        mix_p = mix_p / mix_p.sum()
        sel_grid_idx = rng.choice(len(train_grid), size=W_par, replace=(len(train_grid) < W_par), p=mix_p)
        w_starts_np = train_grid[sel_grid_idx]

        # Непрерывный спектр стартовой загрузки портфеля: 5 чистых (0%) + 7 непрерывных из реальной прошлой истории!
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

            # Инициализация непрерывного спектра инвентаря по РЕАЛЬНЫМ историческим ценам до начала окна:
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
            api_tokens = torch.full((N_par,), 12.0, dtype=torch.float32, device=device)

            peak_mtm_eq = start_mtm_eq.clone()
            prev_mtm_eq = start_mtm_eq.clone()
            prev_val_pred = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_log_dd_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            max_dd_pct_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_mut_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_fqp_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_val_err_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
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
                        n_fills_16, n_cancels_16, n_requeues_16,
                    ) = fused_l26_endogenous_tick(
                        s_steps_i16[t_step], s_steps_i16[t_nxt], pre_mkt_ch[j_t],
                        b1_w_all[t_step], a1_w_all[t_step], b1q_w_all[t_step], a1q_w_all[t_step],
                        rb_w_all[t_step], rs_w_all[t_step],
                        b10_w_all[t_step], a10_w_all[t_step], bq10_w_all[t_step], aq10_w_all[t_step],
                        b1_w_all[t_nxt], a1_w_all[t_nxt], sv_w_all[t_nxt], bv_w_all[t_nxt],
                        w1_ag_g, policy.w_res1, policy.b_res1, policy.w_res2,
                        policy.w2, policy.b2, policy.w_rec, policy.b_rec,
                        policy.w_reg1, policy.b_reg1_b, policy.b_reg1_a, policy.w_reg2, policy.b_reg2_b, policy.b_reg2_a,
                        echelon_k01, budget_16,
                        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                        api_tokens, peak_mtm_eq, sum_log_dd_16, max_dd_pct_16, sum_mut_16,
                        sum_fqp_16, sum_val_err_16, prev_val_pred, prev_mtm_eq,
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

            # Самокалибрующаяся траекторная метрика роста капитала без ручных коэффициентов монеты:
            dlog_mat = torch.log(end_cash / start_mtm_eq).view(W_par, n_groups, n_budgets)
            m_log_dd = (sum_log_dd_16 / float(win_len - 1)).view(W_par, n_groups, n_budgets)
            traj_wb_g = 10000.0 * (dlog_mat - 0.55 * m_log_dd).permute(1, 0, 2).reshape(n_groups, W_par * n_budgets)
            val_mse_g = (sum_val_err_16 / float(win_len - 1)).view(W_par, n_groups, n_budgets).mean(dim=(0, 2))

            # Онлайн-обновление потенциала информативности/сложности окон по разбросу популяции (IQR + просадка):
            w_med_g = (10000.0 * dlog_mat.mean(dim=2))  # [W_par, n_groups]
            w_iqr = torch.quantile(w_med_g, 0.80, dim=1) - torch.quantile(w_med_g, 0.20, dim=1)
            w_tail = torch.clamp(-torch.quantile(w_med_g, 0.20, dim=1), min=0.0)
            w_diff_obs = torch.clamp(1.0 + 0.08 * w_iqr + 0.04 * w_tail, 0.5, 12.0).detach().cpu().numpy()
            for k_u, gi_u in enumerate(sel_grid_idx):
                win_difficulty[int(gi_u)] = 0.80 * win_difficulty[int(gi_u)] + 0.20 * float(w_diff_obs[k_u])

            pareto_fitness, ent_g = compute_endogenous_5d_pareto_fitness(traj_wb_g, val_mse_g)
            sorted_g = torch.argsort(pareto_fitness, descending=True)
            best_g = int(sorted_g[0].item())
            top_k = nes_w.numel()
            top_idx = sorted_g[:top_k]

            new_center = {}
            for k in policy.PARAM_KEYS:
                p_top = policy.__dict__[k][top_idx].float()
                w_shape = (-1,) + (1,) * (p_top.ndim - 1)
                new_center[k] = (p_top * nes_w.view(*w_shape)).sum(dim=0)
            new_center = _normalize_weights_l26(new_center, policy.dtype)
            new_gl = torch.clamp((policy.g_logits[top_idx].float() * nes_w.view(-1, 1)).sum(dim=0), -2.5, 2.0).detach().cpu()
            hof_state = {k: v.detach().cpu().clone() for k, v in new_center.items()}
            hof_g_logit = new_gl.clone()

            best_cand_gpu = _normalize_weights_l26({k: policy.__dict__[k][best_g] for k in policy.PARAM_KEYS}, policy.dtype)
            best_cand_cpu = {k: v.detach().cpu().clone() for k, v in best_cand_gpu.items()}
            best_cand_gl = policy.g_logits[best_g].detach().cpu().clone()

            # Валидация обоих лидеров (NES-центр + лучший индивид) на хронологической проверочной корзине:
            cand_list = [
                policy.fold_gate_into_state(hof_state, hof_g_logit),
                policy.fold_gate_into_state(best_cand_cpu, best_cand_gl),
            ]
            val_pair = evaluate_l26_basket(
                vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
                cand_list if smoke_test else (cand_list + [cand_list[0]] * 4)[:6],
                val_basket[:6] if smoke_test else val_basket, win_len, tick_sz, device, true_latency=True,
            )[:2]
            win_vi = 0 if val_pair[0]["entropic_growth_bp"] >= val_pair[1]["entropic_growth_bp"] else 1
            v_res = val_pair[win_vi]
            v_ent = float(v_res["entropic_growth_bp"])
            dt_it = time.time() - t0

            improved_str = ""
            if v_ent > best_ent_bp + 1e-4:
                best_ent_bp = v_ent
                best_val_state = {k: v.clone() for k, v in (hof_state if win_vi == 0 else best_cand_cpu).items()}
                best_val_g_logit = (hof_g_logit if win_vi == 0 else best_cand_gl).clone()
                best_val_folded = {k: v.clone() for k, v in cand_list[win_vi].items()}
                improved_str = " 🏆 NEW_L26_CHAMP!"
                if not smoke_test:
                    tag_c = f"L26_P{phase}_IT{it:04d}"
                    _save_l26_checkpoint(SAVE_DIR / f"l26_champ_iter_{it:04d}.pt", base_ck, best_val_folded, best_val_g_logit, it, phase, tag_c, v_res)
                    _save_l26_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, it, phase, tag_c, v_res)
                if v_ent >= sig_anchor_ent + plateau_min_delta_bp:
                    sig_anchor_ent = v_ent
                    last_sig_improve_it = it

            if not smoke_test:
                _save_l26_checkpoint(RESUME_CHAMP_PATH, base_ck, policy.fold_gate_into_state(hof_state, hof_g_logit), hof_g_logit, it, phase, f"L26_RESUME_{it:04d}", v_res)
                rec = {"iteration": it, "phase": phase, "dt_sec": round(dt_it, 2), **v_res, "best_entropic_bp": best_ent_bp}
                history.append(rec)
                summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            log_msg(
                f"[L26 P{phase} | it {it:03d}/{iterations} | {dt_it:.1f}s] "
                f"ValEnt={v_ent:+.2f}bp (Best={best_ent_bp:+.2f}bp) [Q15:{v_res['q15_bp']:+.1f} Q50:{v_res['q50_bp']:+.1f} Q85:{v_res['q85_bp']:+.1f}] | "
                f"CleanEV={v_res['clean_ev_pct']:+.4f}% InvEV={v_res['inv_delta_ev_pct']:+.4f}% DD={v_res['dd_pct']:.2f}% | "
                f"F={v_res['fills']:.1f} Cnl/F={v_res['cnl_per_fill']:.1f} η_q={v_res['eta_queue_pct']:.1f}%{improved_str}"
            )

            if (not smoke_test) and phase >= 4 and plateau_patience > 0 and (it - max(phase3_iters, last_sig_improve_it)) >= plateau_patience:
                log_msg(f"🛑 АВТО-ПЛАТО ДОСТИГНУТО на итерации {it}: {plateau_patience} итераций без прироста >= +{plateau_min_delta_bp:.2f}bp!")
                break

    if not smoke_test:
        champ_files = sorted(SAVE_DIR.glob("l26_champ_iter_*.pt"))
        if len(champ_files) >= 2:
            scored = []
            for cf in champ_files:
                ck_i = torch.load(cf, map_location="cpu", weights_only=False)
                scored.append((float(ck_i.get("entropic_growth_bp", -1e9)), cf, ck_i))
            scored.sort(key=lambda x: x[0], reverse=True)
            c1, c2 = scored[0][2], scored[1][2]
            soup_st = {k: (0.5 * c1[k].float() + 0.5 * c2[k].float()).to(c1[k].dtype) for k in EndogenousNonlinearPolicyL26.PARAM_KEYS}
            soup_res = evaluate_l26_basket(
                vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
                [soup_st] * 6, val_basket, win_len, tick_sz, device, true_latency=True,
            )[0]
            log_msg(f"🍲 ФИНАЛЬНЫЙ L26 MODEL SOUP TOP-2: EntGrowth={soup_res['entropic_growth_bp']:+.2f}bp | CleanEV={soup_res['clean_ev_pct']:+.4f}% | InvEV={soup_res['inv_delta_ev_pct']:+.4f}%")
            _save_l26_checkpoint(SOUP_CHAMP_PATH, base_ck, soup_st, torch.zeros(N_BLOCKS_L26), iterations, 4, "L26_SOUP_TOP2_ENDOGENOUS", soup_res)
    else:
        log_msg("✅ L26 SMOKE TEST PASSED SUCCESSFULLY!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=560)
    ap.add_argument("--phase1-iters", type=int, default=70)
    ap.add_argument("--phase2-iters", type=int, default=260)
    ap.add_argument("--phase3-iters", type=int, default=440)
    ap.add_argument("--plateau-patience", type=int, default=80)
    ap.add_argument("--plateau-min-delta-bp", type=float, default=0.25)
    ap.add_argument("--n-groups", type=int, default=1024)
    ap.add_argument("--win-len", type=int, default=2400)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    run_lesson_26(
        iterations=args.iterations,
        phase1_iters=args.phase1_iters,
        phase2_iters=args.phase2_iters,
        phase3_iters=args.phase3_iters,
        plateau_patience=args.plateau_patience,
        plateau_min_delta_bp=args.plateau_min_delta_bp,
        n_groups=args.n_groups,
        win_len=args.win_len,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
