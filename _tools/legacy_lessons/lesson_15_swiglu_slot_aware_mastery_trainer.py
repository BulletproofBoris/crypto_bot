#!/usr/bin/env python3
"""
_tools/lesson_11_tranche_queue_mastery_trainer.py
=============================================================================
Урок 11: Progressive Tranche Scaling, Queue Persistence & End-to-End Execution Mastery (600 итераций)
Устраняет плато насыщения Уроков 6–8 и открывает агенту свободу безубытка/микро-стопа:
1. СНЯТИЕ БАРЬЕРА ВЫХОДА (allow_scratch_stop_asks=True):
   - Ask[0]: Лимитный Micro-Stop (-1 тик от цены входа)
   - Ask[1]: Лимитный Scratch / Breakeven (0 тиков от цены входа)
   - Ask[2..5]: LIFO-слоты фиксации прибыли по нижним траншам (+1..+3 тика)
   - Ask[6..11]: Дальние слоты фиксации прибыли (+1..+5 тиков)
2. ОТКАЗ ОТ ШТРАФА ЗА КОЛИЧЕСТВО УБЫТОЧНЫХ СДЕЛОК (No Loss-Cycle Paralysis):
   - Убран штраф `-loss_cycles * 50`, который заставлял агента пересиживать проливы
     по 1200+ секунд вместо закрытия в 0 / -1 тик.
   - Введен квадратичный штраф за время под водой (Time-Under-Water) и премия за
     своевременный сброс позиции через Ask[0..1] при росте угрозы пролива.
3. МОНОТОННАЯ СХОДИМОСТЬ БЕЗ «ПРОКЛЯТИЯ ПОБЕДИТЕЛЯ» (NES + Per-Step Advantage + Val Guard):
   - Пошаговый градиент преимущества (Trajectory Advantage Gradient) внутри эпизода.
   - Естественный эволюционный градиент (NES / Rank-Weighted Elite Direction) с
     косинусным затуханием радиуса поиска и инерцией Поляка (Polyak Momentum).
   - Фиксированная валидационная корзина из 9 окон (3 Normal + 3 Dump + 3 Pump = 5400с):
     эталонный чемпион `best_val_state` обновляется ТОЛЬКО при строгом росте метрики
     на неизменной многорежимной проверочной выборке!
=============================================================================
"""

import copy
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_free_multi_order_env import VectorizedFreeMultiOrderEnv
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_6_free_multi_order_trainer import BatchedFreeGridPolicyGrouped

class AntitheticFreeGridPolicyGrouped(BatchedFreeGridPolicyGrouped):
    """
    Популяционная политика Урока 15 (356D -> SwiGLU(384->256) -> 26D) с антитетическим (+eps / -eps) сэмплированием:
    - Вход 356D: 324D база + 12D bid_slots + 12D ask_slots + 8D v7_velocity_5s.
    - Скрытый слой SwiGLU (384D -> 256D): 256 базовых нейронов ReLU * 128 мультипликативных вентилей 2*Sigmoid.
    - Бесшовный математический Warm-Start из Чемпиона Урока 14 (324D -> 256D) без потери качества!
    """

    def __init__(
        self,
        backbone: MultiTaskLOBNet,
        n_groups: int = 2048,
        n_budgets: int = 6,
        in_dim: int = 356,
        h_dim: int = 256,
        out_dim: int = 26,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, in_dim=in_dim, h_dim=h_dim, out_dim=out_dim, device=device)
        self.h1_full_dim = 384  # 256 base ReLU + 128 SwiGLU gates
        self.w1 = torch.nn.Parameter(
            torch.randn(self.pop_size, in_dim, self.h1_full_dim, device=device, dtype=self.dtype) * 0.03,
            requires_grad=False,
        )
        self.b1 = torch.nn.Parameter(
            torch.zeros(self.pop_size, 1, self.h1_full_dim, device=device, dtype=self.dtype),
            requires_grad=False,
        )

    def _normalize_single_weights(self, w1_s: torch.Tensor, b1_s: torch.Tensor, w2_s: torch.Tensor, b2_s: torch.Tensor):
        w1_f = w1_s.float()
        w1_base = w1_f[:, :256]
        n1 = float(w1_base.norm().item())
        if n1 > 145.0:
            w1_base = w1_base * (145.0 / n1)
        w1_gate = w1_f[:, 256:384]
        ng = float(w1_gate.norm().item())
        if ng > 85.0:
            w1_gate = w1_gate * (85.0 / ng)
        w1_f = torch.cat([w1_base, w1_gate], dim=-1)

        w2_f = w2_s.float()
        n2 = float(w2_f.norm().item())
        if n2 > 45.0:
            w2_f = w2_f * (45.0 / n2)

        b1_f = torch.clamp(b1_s.float(), -2.0, 2.0)
        b2_f = torch.clamp(b2_s.float(), -3.0, 2.0)

        return (
            w1_f.to(dtype=self.dtype),
            b1_f.to(dtype=self.dtype),
            w2_f.to(dtype=self.dtype),
            b2_f.to(dtype=self.dtype),
        )

    def load_champion_into_population(self, ckpt_data: dict):
        if "backbone_state" in ckpt_data:
            self.backbone.load_state_dict(ckpt_data["backbone_state"], strict=False)
        w1_raw = ckpt_data["w1"].to(device=self.device, dtype=self.dtype)
        b1_raw = ckpt_data["b1"].to(device=self.device, dtype=self.dtype)
        w2_c = ckpt_data["w2"].to(device=self.device, dtype=self.dtype)
        b2_c = ckpt_data["b2"].to(device=self.device, dtype=self.dtype)

        # Бесшовный Zero-Loss Warm-Start из 324x256 Чемпиона Урока 14 в 356x384 SwiGLU:
        w1_c = torch.zeros((self.in_dim, self.h1_full_dim), device=self.device, dtype=self.dtype)
        b1_c = torch.zeros((1, self.h1_full_dim), device=self.device, dtype=self.dtype)
        r_in = min(self.in_dim, w1_raw.shape[0])
        r_h = min(self.h1_full_dim, w1_raw.shape[1])
        w1_c[:r_in, :r_h] = w1_raw[:r_in, :r_h]
        b1_c[:, : min(self.h1_full_dim, b1_raw.shape[-1])] = b1_raw.view(1, -1)[:, : min(self.h1_full_dim, b1_raw.shape[-1])]

        with torch.no_grad():
            w1_n, b1_n, w2_n, b2_n = self._normalize_single_weights(w1_c, b1_c, w2_c, b2_c)
            self.w1.copy_(w1_n.unsqueeze(0).expand(self.pop_size, -1, -1))
            self.b1.copy_(b1_n.unsqueeze(0).expand(self.pop_size, -1, -1))
            self.w2.copy_(w2_n.unsqueeze(0).expand(self.pop_size, -1, -1))
            self.b2.copy_(b2_n.unsqueeze(0).expand(self.pop_size, -1, -1))

            B_buds = self.n_budgets
            n_mut_g = self.n_groups - 1
            if n_mut_g > 0:
                chunk_g = 256
                for g0 in range(1, self.n_groups, chunk_g):
                    g1 = min(self.n_groups, g0 + chunk_g)
                    gc_cnt = g1 - g0
                    nw1 = torch.randn((gc_cnt, 1, self.in_dim, self.h1_full_dim), device=self.device, dtype=self.dtype) * 0.015
                    self.w1[g0 * B_buds : g1 * B_buds].view(gc_cnt, B_buds, self.in_dim, self.h1_full_dim).add_(
                        nw1.expand(gc_cnt, B_buds, self.in_dim, self.h1_full_dim)
                    )
                nb1 = torch.randn((n_mut_g, 1, 1, self.h1_full_dim), device=self.device, dtype=self.dtype) * 0.008
                self.b1[B_buds:].view(n_mut_g, B_buds, 1, self.h1_full_dim).add_(nb1.expand(n_mut_g, B_buds, 1, self.h1_full_dim))
                nw2 = torch.randn((n_mut_g, 1, self.h_dim, self.out_dim), device=self.device, dtype=self.dtype) * 0.015
                self.w2[B_buds:].view(n_mut_g, B_buds, self.h_dim, self.out_dim).add_(nw2.expand(n_mut_g, B_buds, self.h_dim, self.out_dim))
                nb2 = torch.randn((n_mut_g, 1, 1, self.out_dim), device=self.device, dtype=self.dtype) * 0.008
                self.b2[B_buds:].view(n_mut_g, B_buds, 1, self.out_dim).add_(nb2.expand(n_mut_g, B_buds, 1, self.out_dim))

    def evolve_groups(self, group_scores: torch.Tensor, hof_state: dict, elite_frac: float = 0.03, progress: float = 0.0) -> int:
        B_buds = self.n_budgets
        H1_full = self.h1_full_dim
        n_elites = max(12, int(self.n_groups * elite_frac))
        if (self.n_groups - n_elites) % 2 != 0:
            n_elites += 1
        sorted_g = torch.argsort(group_scores, descending=True)
        elite_g = sorted_g[:n_elites]
        best_g = int(sorted_g[0].item())

        elite_env_idx = elite_g * B_buds
        w1_el = self.w1[elite_env_idx].clone()
        b1_el = self.b1[elite_env_idx].clone()
        w2_el = self.w2[elite_env_idx].clone()
        b2_el = self.b2[elite_env_idx].clone()

        for ei in range(n_elites):
            w1_el[ei], b1_el[ei], w2_el[ei], b2_el[ei] = self._normalize_single_weights(
                w1_el[ei], b1_el[ei], w2_el[ei], b2_el[ei]
            )

        w1_el[0].copy_(hof_state["w1"].to(self.device, self.dtype))
        b1_el[0].copy_(hof_state["b1"].to(self.device, self.dtype))
        w2_el[0].copy_(hof_state["w2"].to(self.device, self.dtype))
        b2_el[0].copy_(hof_state["b2"].to(self.device, self.dtype))

        self.w1[: n_elites * B_buds].view(n_elites, B_buds, self.in_dim, H1_full).copy_(
            w1_el.unsqueeze(1).expand(n_elites, B_buds, self.in_dim, H1_full)
        )
        self.b1[: n_elites * B_buds].view(n_elites, B_buds, 1, H1_full).copy_(
            b1_el.unsqueeze(1).expand(n_elites, B_buds, 1, H1_full)
        )
        self.w2[: n_elites * B_buds].view(n_elites, B_buds, self.h_dim, self.out_dim).copy_(
            w2_el.unsqueeze(1).expand(n_elites, B_buds, self.h_dim, self.out_dim)
        )
        self.b2[: n_elites * B_buds].view(n_elites, B_buds, 1, self.out_dim).copy_(
            b2_el.unsqueeze(1).expand(n_elites, B_buds, 1, self.out_dim)
        )

        n_child_g = self.n_groups - n_elites
        n_pairs = n_child_g // 2
        pair_parents = torch.where(
            torch.rand(n_pairs, device=self.device) < 0.55,
            torch.zeros(n_pairs, dtype=torch.long, device=self.device),
            torch.randint(0, n_elites, (n_pairs,), device=self.device),
        )

        decay = max(0.25, 1.0 - 0.68 * float(progress))
        sigma_pairs = torch.where(
            torch.rand(n_pairs, 1, 1, 1, device=self.device) < 0.72,
            torch.full((n_pairs, 1, 1, 1), 0.018 * decay, device=self.device, dtype=self.dtype),
            torch.full((n_pairs, 1, 1, 1), 0.045 * decay, device=self.device, dtype=self.dtype),
        )

        chunk_p = 128
        buf_w1_p = torch.empty((chunk_p, 1, self.in_dim, H1_full), device=self.device, dtype=self.dtype)
        for p0 in range(0, n_pairs, chunk_p):
            p1 = min(n_pairs, p0 + chunk_p)
            np_cur = p1 - p0
            c0 = p0 * 2
            c1 = p1 * 2
            g_start = n_elites + c0
            g_end = n_elites + c1
            sig_p = sigma_pairs[p0:p1]

            dst_w1 = self.w1[g_start * B_buds : g_end * B_buds].view(np_cur, 2, B_buds, self.in_dim, H1_full)
            base_w1 = w1_el[pair_parents[p0:p1]].unsqueeze(1).unsqueeze(2).expand(np_cur, 2, B_buds, self.in_dim, H1_full)
            dst_w1.copy_(base_w1)

            bw = buf_w1_p[:np_cur]
            bw.normal_(0.0, 1.0).mul_(sig_p)
            dst_w1[:, 0].add_(bw.expand(np_cur, B_buds, self.in_dim, H1_full))
            dst_w1[:, 1].sub_(bw.expand(np_cur, B_buds, self.in_dim, H1_full))

        del buf_w1_p, w1_el

        dst_b1 = self.b1[n_elites * B_buds :].view(n_pairs, 2, B_buds, 1, H1_full)
        dst_b1.copy_(b1_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, 1, H1_full))
        nb1 = torch.randn((n_pairs, 1, 1, H1_full), device=self.device, dtype=self.dtype) * (sigma_pairs * 0.5)
        dst_b1[:, 0].add_(nb1.expand(n_pairs, B_buds, 1, H1_full))
        dst_b1[:, 1].sub_(nb1.expand(n_pairs, B_buds, 1, H1_full))

        dst_w2 = self.w2[n_elites * B_buds :].view(n_pairs, 2, B_buds, self.h_dim, self.out_dim)
        dst_w2.copy_(w2_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, self.h_dim, self.out_dim))
        nw2 = torch.randn((n_pairs, 1, self.h_dim, self.out_dim), device=self.device, dtype=self.dtype) * sigma_pairs
        dst_w2[:, 0].add_(nw2.expand(n_pairs, B_buds, self.h_dim, self.out_dim))
        dst_w2[:, 1].sub_(nw2.expand(n_pairs, B_buds, self.h_dim, self.out_dim))

        dst_b2 = self.b2[n_elites * B_buds :].view(n_pairs, 2, B_buds, 1, self.out_dim)
        dst_b2.copy_(b2_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, 1, self.out_dim))
        nb2 = torch.randn((n_pairs, 1, 1, self.out_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 0.5)
        dst_b2[:, 0].add_(nb2.expand(n_pairs, B_buds, 1, self.out_dim))
        dst_b2[:, 1].sub_(nb2.expand(n_pairs, B_buds, 1, self.out_dim))

        return best_g


CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
RELEASE_CHAMP_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
L12_CHAMP_PATH = BASE_DIR / "models" / "prl" / "lesson_14_capital_utilization_mastery" / "lesson_14_champ_latest.pt"
SAVE_DIR = BASE_DIR / "models" / "prl" / "lesson_15_swiglu_slot_aware_mastery"
LOG_FILE = BASE_DIR / "data" / "logs" / "lesson_15_swiglu_slot_aware_mastery.log"


def log_msg(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@torch.jit.script
def _jit_bf16_backbone_step(
    lob_r_1d: torch.Tensor,
    flow_fp32: torch.Tensor,
    h_states: torch.Tensor,
    w_fl: torch.Tensor,
    b_fl: torch.Tensor,
    ln_fl_w: torch.Tensor,
    ln_fl_b: torch.Tensor,
    w_fu_lob: torch.Tensor,
    w_fu_fl: torch.Tensor,
    b_fu: torch.Tensor,
    ln_fu_w: torch.Tensor,
    ln_fu_b: torch.Tensor,
    w_ih: torch.Tensor,
    b_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_hh: torch.Tensor,
    w_h1: torch.Tensor,
    b_h1: torch.Tensor,
    w_h2: torch.Tensor,
    b_h2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = flow_fp32.size(0)
    flow_b = flow_fp32.to(h_states.dtype)
    fl = F.gelu(F.layer_norm(flow_b @ w_fl + b_fl, [64], ln_fl_w, ln_fl_b))
    fu_raw = (lob_r_1d @ w_fu_lob + b_fu) + (fl @ w_fu_fl)
    x = F.gelu(F.layer_norm(fu_raw, [256], ln_fu_w, ln_fu_b))

    gi = x @ w_ih + b_ih
    gh = h_states @ w_hh + b_hh
    i_r, i_z, i_n = gi.chunk(3, 1)
    h_r, h_z, h_n = gh.chunk(3, 1)
    r = torch.sigmoid(i_r + h_r)
    z_gate = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    next_h = (1.0 - z_gate) * n + z_gate * h_states

    hh = F.gelu(next_h @ w_h1 + b_h1).view(B, 4, 64).transpose(0, 1)
    p4 = torch.sigmoid(torch.bmm(hh, w_h2).squeeze(-1).t() + b_h2)
    return next_h, p4, next_h


@torch.jit.script
def _jit_policy_forward(
    z: torch.Tensor,
    p4: torch.Tensor,
    v7_cur: torch.Tensor,
    v7_vel8_cur: torch.Tensor,
    multi_6d: torch.Tensor,
    bid_qty: torch.Tensor,
    ask_qty: torch.Tensor,
    pos_qty: torch.Tensor,
    env_budgets: torch.Tensor,
    b1_t: torch.Tensor,
    w1_g: torch.Tensor,
    b1_g: torch.Tensor,
    w2_g: torch.Tensor,
    b2_g: torch.Tensor,
    n_groups: int,
    n_budgets: int,
    pop_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = z.dtype
    v7_exp = v7_cur.unsqueeze(0).expand(pop_size, -1)
    vel_exp = v7_vel8_cur.unsqueeze(0).expand(pop_size, -1)
    b1_val = torch.clamp(b1_t, min=0.01)
    b_slot_12 = torch.clamp((bid_qty * b1_val) / torch.clamp(env_budgets.unsqueeze(1), min=25.0), 0.0, 1.0).to(dt)
    a_slot_12 = torch.clamp(ask_qty / torch.clamp(pos_qty.unsqueeze(1), min=2.0), 0.0, 1.0).to(dt)
    z_aug_g = torch.cat([z, p4, v7_exp, multi_6d.to(dt), b_slot_12, a_slot_12, vel_exp], dim=-1).view(
        n_groups, n_budgets, 356
    )
    pre_all = torch.bmm(z_aug_g, w1_g) + b1_g
    h_base = torch.relu(pre_all[:, :, :256])
    h_gate = (2.0 * torch.sigmoid(pre_all[:, :, 256:384])).repeat_interleave(2, dim=-1)
    h1_g = h_base * h_gate
    raw_out = torch.sigmoid((torch.bmm(h1_g, w2_g) + b2_g).reshape(pop_size, 26).float())
    return h1_g, raw_out


@torch.jit.script
def _jit_step_metrics(
    cur_pos: torch.Tensor,
    cur_cash: torch.Tensor,
    pos_cost_usd: torch.Tensor,
    avg_entry: torch.Tensor,
    ask_qty: torch.Tensor,
    raw_out: torch.Tensor,
    prev_pos: torch.Tensor,
    env_budgets: torch.Tensor,
    b1_t: torch.Tensor,
    cur_hold: torch.Tensor,
    max_hold_s: torch.Tensor,
    peak_real_eq: torch.Tensor,
    max_real_dd_usd: torch.Tensor,
    peak_paper_eq: torch.Tensor,
    max_paper_dd_usd: torch.Tensor,
    tuw_pen: torch.Tensor,
    scratch_evasion_bonus: torch.Tensor,
    pump_capture_bonus: torch.Tensor,
    tick_sz: float,
    is_bleed: bool,
    is_pump_tick: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    b1 = torch.clamp(b1_t, min=tick_sz)
    in_p = cur_pos > 1e-6

    cur_hold = torch.where(in_p, cur_hold + 1.0, torch.zeros_like(cur_hold))
    max_hold_s = torch.maximum(max_hold_s, cur_hold)

    real_eq = cur_cash + pos_cost_usd
    peak_real_eq = torch.maximum(peak_real_eq, real_eq)
    max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - real_eq)

    paper_eq = cur_cash + cur_pos * b1
    peak_paper_eq = torch.maximum(peak_paper_eq, paper_eq)
    max_paper_dd_usd = torch.maximum(max_paper_dd_usd, peak_paper_eq - paper_eq)

    underwater_ticks = torch.where(in_p, torch.clamp((avg_entry - b1) / tick_sz, min=0.0), torch.zeros_like(cur_pos))
    pos_frac = torch.clamp((cur_pos * b1) / env_budgets, 0.0, 1.0)
    tuw_mult = 0.045 if is_bleed else 0.015
    tuw_pen = tuw_pen + pos_frac * torch.pow(underwater_ticks, 1.6) * tuw_mult

    if is_bleed:
        reduced_pos = (cur_pos < prev_pos - 1e-4).float()
        scratch_active = (torch.maximum(raw_out[:, 12], raw_out[:, 13]) >= 0.15).float()
        scratch_evasion_bonus = scratch_evasion_bonus + reduced_pos * (1.0 + 1.5 * scratch_active)

    if is_pump_tick:
        prof_ticks = torch.clamp((b1 - avg_entry) / tick_sz, min=0.0, max=6.0)
        far_ask_frac = torch.clamp(ask_qty[:, 4:].sum(dim=1) / torch.clamp(cur_pos, min=1e-5), 0.0, 1.0)
        pump_capture_bonus = pump_capture_bonus + in_p.float() * prof_ticks * (0.018 + 0.030 * far_ask_frac)

    return (
        cur_hold,
        max_hold_s,
        peak_real_eq,
        max_real_dd_usd,
        peak_paper_eq,
        max_paper_dd_usd,
        tuw_pen,
        scratch_evasion_bonus,
        pump_capture_bonus,
    )



class FusedBF16BackboneRunner:
    """
    L2-Cache Resident (32 MB) матричный JIT-ускоритель MultiTaskLOBNet в bfloat16/float16:
    - Сливает 15 операций GRUCell + LayerNorm + 4 Risk Heads в единый C++ JIT-граф в L2 кэше GPU.
    """

    def __init__(self, backbone_fp32: MultiTaskLOBNet, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device
        bb = backbone_fp32.backbone

        self.w_fl = bb.flow_encoder[0].weight.detach().to(device=device, dtype=dtype).t().contiguous()
        self.b_fl = bb.flow_encoder[0].bias.detach().to(device=device, dtype=dtype).contiguous()
        self.ln_fl_w = bb.flow_encoder[1].weight.detach().to(device=device, dtype=dtype).contiguous()
        self.ln_fl_b = bb.flow_encoder[1].bias.detach().to(device=device, dtype=dtype).contiguous()

        self.w_fu_lob = bb.fusion[0].weight[:, :64].detach().to(device=device, dtype=dtype).t().contiguous()
        self.w_fu_fl = bb.fusion[0].weight[:, 64:].detach().to(device=device, dtype=dtype).t().contiguous()
        self.b_fu = bb.fusion[0].bias.detach().to(device=device, dtype=dtype).contiguous()
        self.ln_fu_w = bb.fusion[1].weight.detach().to(device=device, dtype=dtype).contiguous()
        self.ln_fu_b = bb.fusion[1].bias.detach().to(device=device, dtype=dtype).contiguous()

        self.w_ih = bb.gru.weight_ih_l0.detach().to(device=device, dtype=dtype).t().contiguous()
        self.b_ih = bb.gru.bias_ih_l0.detach().to(device=device, dtype=dtype).contiguous()
        self.w_hh = bb.gru.weight_hh_l0.detach().to(device=device, dtype=dtype).t().contiguous()
        self.b_hh = bb.gru.bias_hh_l0.detach().to(device=device, dtype=dtype).contiguous()

        self.w_h1 = torch.cat(
            [
                backbone_fp32.head_sweep_12s[0].weight.detach(),
                backbone_fp32.head_bleed_60s[0].weight.detach(),
                backbone_fp32.head_crash_300s[0].weight.detach(),
                backbone_fp32.head_exit_30s[0].weight.detach(),
            ],
            dim=0,
        ).to(device=device, dtype=dtype).t().contiguous()  # [256, 256]
        self.b_h1 = torch.cat(
            [
                backbone_fp32.head_sweep_12s[0].bias.detach(),
                backbone_fp32.head_bleed_60s[0].bias.detach(),
                backbone_fp32.head_crash_300s[0].bias.detach(),
                backbone_fp32.head_exit_30s[0].bias.detach(),
            ],
            dim=0,
        ).to(device=device, dtype=dtype).unsqueeze(0).contiguous()  # [1, 256]
        self.w_h2 = torch.stack(
            [
                backbone_fp32.head_sweep_12s[2].weight.detach(),
                backbone_fp32.head_bleed_60s[2].weight.detach(),
                backbone_fp32.head_crash_300s[2].weight.detach(),
                backbone_fp32.head_exit_30s[2].weight.detach(),
            ],
            dim=0,
        ).to(device=device, dtype=dtype).transpose(1, 2).contiguous()  # [4, 64, 1]
        self.b_h2 = torch.cat(
            [
                backbone_fp32.head_sweep_12s[2].bias.detach(),
                backbone_fp32.head_bleed_60s[2].bias.detach(),
                backbone_fp32.head_crash_300s[2].bias.detach(),
                backbone_fp32.head_exit_30s[2].bias.detach(),
            ],
            dim=0,
        ).to(device=device, dtype=dtype).unsqueeze(0).contiguous()  # [1, 4]

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros((batch_size, 256), device=self.device, dtype=self.dtype)

    def step(self, lob_r_1d: torch.Tensor, flow_fp32: torch.Tensor, h_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _jit_bf16_backbone_step(
            lob_r_1d,
            flow_fp32,
            h_states,
            self.w_fl,
            self.b_fl,
            self.ln_fl_w,
            self.ln_fl_b,
            self.w_fu_lob,
            self.w_fu_fl,
            self.b_fu,
            self.ln_fu_w,
            self.ln_fu_b,
            self.w_ih,
            self.b_ih,
            self.w_hh,
            self.b_hh,
            self.w_h1,
            self.b_h1,
            self.w_h2,
            self.b_h2,
        )


@torch.compile(dynamic=True)
def _multi_win_val_tick(
    t_vec: torch.Tensor,
    next_t_vec: torch.Tensor,
    flow_mkt_w: torch.Tensor,
    b1_w: torch.Tensor,
    a1_w: torch.Tensor,
    b1_q_w: torch.Tensor,
    a1_q_w: torch.Tensor,
    roll_buy_w: torch.Tensor,
    bids_t_w: torch.Tensor,
    asks_t_w: torch.Tensor,
    bids_q_t_w: torch.Tensor,
    asks_q_t_w: torch.Tensor,
    nb1_w: torch.Tensor,
    na1_w: torch.Tensor,
    sv_w: torch.Tensor,
    bv_w: torch.Tensor,
    lob_r_w: torch.Tensor,
    v7_w: torch.Tensor,
    w_fl: torch.Tensor,
    b_fl: torch.Tensor,
    ln_fl_w: torch.Tensor,
    ln_fl_b: torch.Tensor,
    w_fu_lob: torch.Tensor,
    w_fu_fl: torch.Tensor,
    b_fu: torch.Tensor,
    ln_fu_w: torch.Tensor,
    ln_fu_b: torch.Tensor,
    w_ih: torch.Tensor,
    b_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_hh: torch.Tensor,
    w_h1: torch.Tensor,
    b_h1: torch.Tensor,
    w_h2: torch.Tensor,
    b_h2: torch.Tensor,
    w1_c: torch.Tensor,
    b1_c_pol: torch.Tensor,
    w2_c: torch.Tensor,
    b2_c_pol: torch.Tensor,
    slot_tick_offsets: torch.Tensor,
    slot_lob_lvl: torch.Tensor,
    is_penny_slot: torch.Tensor,
    budget: torch.Tensor,
    h_states: torch.Tensor,
    cash: torch.Tensor,
    pos_qty: torch.Tensor,
    pos_cost_usd: torch.Tensor,
    avg_entry: torch.Tensor,
    min_buy_px: torch.Tensor,
    low_tranche_qty: torch.Tensor,
    entry_t: torch.Tensor,
    bid_px: torch.Tensor,
    bid_qty: torch.Tensor,
    bid_qa: torch.Tensor,
    bid_act_t: torch.Tensor,
    ask_px: torch.Tensor,
    ask_qty: torch.Tensor,
    ask_qa: torch.Tensor,
    ask_act_t: torch.Tensor,
    peak_eq: torch.Tensor,
    w_dd_usd: torch.Tensor,
    W: int,
    C: int,
    B_bud: int,
    S: int,
    tick_size: float,
    maker_fee: float,
    taker_fee: float,
    allow_scratch: bool,
):
    N = W * C * B_bud
    b1 = torch.clamp(b1_w, min=tick_size)
    a1 = torch.clamp(a1_w, min=b1 + tick_size)
    b1_q = torch.clamp(b1_q_w, min=10.0)
    a1_q = torch.clamp(a1_q_w, min=10.0)
    l1_bid_usd = torch.clamp(b1 * b1_q_w, min=100.0)
    l1_ask_usd = torch.clamp(a1 * a1_q_w, min=100.0)

    has_pos = (pos_qty > 1e-6).float()
    curr_pos_usd = pos_qty * b1
    pos_ratio = torch.clamp(curr_pos_usd / torch.clamp(budget, min=25.0), 0.0, 1.0)
    reserved_cash_now = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    unreserved_cash_now = torch.clamp(cash - reserved_cash_now, min=0.0)
    free_cash_ratio = torch.clamp(unreserved_cash_now / torch.clamp(budget, min=25.0), 0.0, 1.0)

    eff_pos_scale = torch.clamp(pos_ratio / 0.25, 0.15, 1.0)
    unr_ticks = torch.where(has_pos > 0.5, ((b1 - avg_entry) / tick_size) * eff_pos_scale, torch.zeros_like(has_pos))
    hold_ratio = torch.where(
        has_pos > 0.5,
        torch.clamp((t_vec.float() - entry_t.float()) / 300.0, 0.0, 2.0) * eff_pos_scale,
        torch.zeros_like(has_pos),
    )
    tot_bid_q = bid_qty.sum(dim=1)
    tot_ask_q = ask_qty.sum(dim=1)
    has_bid = (tot_bid_q > 1e-6).float()
    has_ask = (tot_ask_q > 1e-6).float()
    n_bid_slots = (bid_qty > 1e-6).float().sum(dim=1) / float(S)
    n_ask_slots = (ask_qty > 1e-6).float().sum(dim=1) / float(S)

    acct_8d = torch.stack(
        [
            has_pos,
            pos_ratio,
            free_cash_ratio,
            torch.clamp(unr_ticks / 5.0, -2.0, 2.0),
            hold_ratio,
            has_bid,
            has_ask,
            n_bid_slots - n_ask_slots,
        ],
        dim=-1,
    )
    flow = torch.cat([flow_mkt_w[:, :8], acct_8d], dim=-1)

    bud_l1 = torch.log1p(budget / l1_bid_usd) / 3.0
    cash_l1 = torch.log1p(unreserved_cash_now / l1_bid_usd) / 3.0
    pos_l1_ask = torch.clamp((pos_qty * a1) / l1_ask_usd, 0.0, 4.0) / 2.0
    pos_tape = torch.clamp(pos_qty / roll_buy_w, 0.0, 10.0) / 5.0
    low_tr_ratio = torch.where(
        has_pos > 0.5, torch.clamp(low_tranche_qty / torch.clamp(pos_qty, min=1e-6), 0.0, 1.0), torch.zeros_like(has_pos)
    )
    lifo_edge_ticks = torch.where(
        has_pos > 0.5, torch.clamp((b1 - min_buy_px) / (3.0 * tick_size), -2.0, 2.0), torch.zeros_like(has_pos)
    )
    multi_6d = torch.stack([bud_l1, cash_l1, pos_l1_ask, pos_tape, low_tr_ratio, lifo_edge_ticks], dim=-1)

    dt = h_states.dtype
    flow_b = flow.to(dt)
    fl = F.gelu(F.layer_norm(flow_b @ w_fl + b_fl, [64], ln_fl_w, ln_fl_b))
    fu_raw = (lob_r_w @ w_fu_lob + b_fu) + (fl @ w_fu_fl)
    x = F.gelu(F.layer_norm(fu_raw, [256], ln_fu_w, ln_fu_b))
    gi = x @ w_ih + b_ih
    gh = h_states @ w_hh + b_hh
    i_r, i_z, i_n = gi.chunk(3, 1)
    h_r, h_z, h_n = gh.chunk(3, 1)
    r = torch.sigmoid(i_r + h_r)
    z_gate = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    h_states = (1.0 - z_gate) * n + z_gate * h_states
    hh = F.gelu(h_states @ w_h1 + b_h1).view(N, 4, 64).transpose(0, 1)
    p4 = torch.sigmoid(torch.bmm(hh, w_h2).squeeze(-1).t() + b_h2)

    b_slot_12 = torch.clamp((bid_qty * b1.unsqueeze(1)) / torch.clamp(budget.unsqueeze(1), min=25.0), 0.0, 1.0).to(dt)
    a_slot_12 = torch.clamp(ask_qty / torch.clamp(pos_qty.unsqueeze(1), min=2.0), 0.0, 1.0).to(dt)
    z_aug = torch.cat([h_states, p4, v7_w[:, :58], multi_6d.to(dt), b_slot_12, a_slot_12, v7_w[:, 58:66]], dim=-1)
    z_c = z_aug.view(W, C, B_bud, 356).permute(1, 0, 2, 3).reshape(C, W * B_bud, 356)
    pre_all_c = torch.bmm(z_c, w1_c) + b1_c_pol
    h_base_c = torch.relu(pre_all_c[:, :, :256])
    h_gate_c = (2.0 * torch.sigmoid(pre_all_c[:, :, 256:384])).repeat_interleave(2, dim=-1)
    h1_c = h_base_c * h_gate_c
    out_c = torch.sigmoid((torch.bmm(h1_c, w2_c) + b2_c_pol).float())
    raw_outputs = out_c.view(C, W, B_bud, 26).permute(1, 0, 2, 3).reshape(N, 26)

    wide_sp = (a1 - b1) >= 2.0 * tick_size
    is_panic = (raw_outputs[:, 2 * S + 1] > 0.82) & (pos_qty > 1e-6)
    b_p10 = torch.clamp(bids_t_w[:, :10], min=tick_size)
    b_q10 = torch.clamp(bids_q_t_w[:, :10], min=1.0).clone()
    b_q10[:, 9] = 1e9
    cum_q10 = torch.cumsum(b_q10, dim=1)
    prev_q10 = cum_q10 - b_q10
    take_10 = torch.clamp(torch.minimum(pos_qty.unsqueeze(1) - prev_q10, b_q10), min=0.0)
    proceeds = (take_10 * b_p10).sum(dim=1) * (1.0 - taker_fee)

    cash = torch.where(is_panic, cash + proceeds, cash)
    pos_qty = torch.where(is_panic, torch.zeros_like(pos_qty), pos_qty)
    pos_cost_usd = torch.where(is_panic, torch.zeros_like(pos_cost_usd), pos_cost_usd)
    avg_entry = torch.where(is_panic, torch.zeros_like(avg_entry), avg_entry)
    min_buy_px = torch.where(is_panic, torch.zeros_like(min_buy_px), min_buy_px)
    low_tranche_qty = torch.where(is_panic, torch.zeros_like(low_tranche_qty), low_tranche_qty)
    ask_qty = torch.where(is_panic.unsqueeze(1), torch.zeros_like(ask_qty), ask_qty)

    cancel_bids = raw_outputs[:, 2 * S] > 0.75
    cb_mask = cancel_bids.unsqueeze(1)
    b_w = torch.where(cb_mask, torch.zeros_like(raw_outputs[:, :S]), raw_outputs[:, :S])

    penny_add = torch.where(wide_sp, torch.full_like(b1, tick_size), torch.zeros_like(b1)).unsqueeze(1)
    b1_2d = b1.unsqueeze(1)
    a1_2d = a1.unsqueeze(1)
    t_bid_px = torch.where(
        is_penny_slot.unsqueeze(0),
        b1_2d + penny_add,
        torch.clamp(b1_2d - slot_tick_offsets.unsqueeze(0), min=tick_size),
    )

    t_2d = t_vec.unsqueeze(1)
    slot_off_b = cb_mask | (b_w < 0.06)
    grid_shift_b = (bid_qty > 1e-6) & ((t_2d - bid_act_t) >= 10) & (torch.abs(t_bid_px - bid_px) >= 1.9 * tick_size)
    clear_b = slot_off_b | grid_shift_b
    bid_qty = torch.where(clear_b, torch.zeros_like(bid_qty), bid_qty)
    bid_px = torch.where(clear_b, torch.zeros_like(bid_px), bid_px)
    bid_qa = torch.where(clear_b, torch.zeros_like(bid_qa), bid_qa)

    empty_b = bid_qty <= 1e-6
    place_b = empty_b & (b_w >= 0.12)
    b_w_active = torch.where(place_b, b_w, torch.zeros_like(b_w))
    b_total_sum = torch.where(b_w >= 0.08, b_w, torch.zeros_like(b_w)).sum(dim=1)
    target_exp_frac = torch.clamp(b_total_sum, 0.0, 1.0)
    target_exp_usd = budget * 0.996 * target_exp_frac

    b_sum = b_w_active.sum(dim=1, keepdim=True)
    scale_b = torch.where(b_sum > 1e-5, 1.0 / torch.clamp(b_sum, min=1e-6), torch.zeros_like(b_sum))
    b_alloc = b_w_active * scale_b

    lvl_b_q = torch.clamp(bids_q_t_w[:, slot_lob_lvl], min=10.0)
    base_b_qa = torch.where(is_penny_slot.unsqueeze(0) & wide_sp.unsqueeze(1), torch.zeros_like(lvl_b_q), lvl_b_q)

    reserved_cash = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    pos_cost_now = pos_qty * avg_entry
    committed_usd = pos_cost_now + reserved_cash
    avail_exp_usd = torch.clamp(target_exp_usd - committed_usd, min=0.0)
    unreserved_cash = torch.minimum(torch.clamp(cash * 0.996 - reserved_cash, min=0.0), avail_exp_usd).unsqueeze(1)
    t_bid_usd = unreserved_cash * b_alloc
    valid_b = place_b & (t_bid_usd >= 2.0)
    t_bid_qty = torch.where(valid_b, t_bid_usd / t_bid_px, torch.zeros_like(t_bid_usd))

    b_over65 = torch.sigmoid((t_bid_usd - 65.0) / 25.0)
    b_self_impact_qa = base_b_qa + b_over65 * (0.065 * b1_q.unsqueeze(1) + 0.15 * t_bid_qty)

    bid_px = torch.where(valid_b, t_bid_px, bid_px)
    bid_qty = torch.where(valid_b, t_bid_qty, bid_qty)
    bid_qa = torch.where(valid_b, b_self_impact_qa, bid_qa)
    bid_act_t = torch.where(valid_b, t_2d + 1, bid_act_t)

    has_p = (pos_qty > 1e-6) & (~is_panic)
    hp_mask = has_p.unsqueeze(1)
    a_w = torch.where(hp_mask, raw_outputs[:, S : 2 * S], torch.zeros_like(raw_outputs[:, S : 2 * S]))

    has_low_tr = (pos_qty > 1e-6) & (low_tranche_qty >= 1.95) & (min_buy_px < avg_entry - 0.1 * tick_size)
    avg_tp_tick = torch.ceil((avg_entry + 0.55 * tick_size) / tick_size - 1e-5) * tick_size
    lifo_tp_tick = torch.ceil((min_buy_px + 0.55 * tick_size) / tick_size - 1e-5) * tick_size
    base_avg_tp = torch.where(pos_qty > 1e-6, avg_tp_tick, a1)
    base_lifo_tp = torch.where(has_low_tr, lifo_tp_tick, base_avg_tp)

    if allow_scratch:
        avg_round_tick = torch.clamp(torch.round(avg_entry / tick_size) * tick_size, min=tick_size)
        stop_1_px = torch.where(pos_qty > 1e-6, torch.clamp(avg_round_tick - tick_size, min=tick_size), base_avg_tp).unsqueeze(1)
        scratch_0_px = torch.where(pos_qty > 1e-6, avg_round_tick, base_avg_tp).unsqueeze(1)
        slot_base_tp = torch.cat(
            [stop_1_px, scratch_0_px, base_lifo_tp.unsqueeze(1).expand(N, 4), base_avg_tp.unsqueeze(1).expand(N, S - 6)],
            dim=1,
        )
    else:
        slot_base_tp = torch.cat(
            [base_lifo_tp.unsqueeze(1).expand(N, 6), base_avg_tp.unsqueeze(1).expand(N, S - 6)], dim=1
        )

    raw_slot_a_px = torch.where(is_penny_slot.unsqueeze(0), a1_2d - penny_add, a1_2d + slot_tick_offsets.unsqueeze(0))
    t_ask_px = torch.maximum(raw_slot_a_px, slot_base_tp + slot_tick_offsets.unsqueeze(0))

    slot_off_a = (~hp_mask) | ((a_w < 0.04) & ((t_2d - ask_act_t) >= 180))
    under_tp_a = (ask_qty > 1e-6) & (ask_px < slot_base_tp - 0.4 * tick_size)
    grid_shift_a = ((ask_qty > 1e-6) & ((t_2d - ask_act_t) >= 15) & (torch.abs(ask_px - t_ask_px) >= 1.9 * tick_size)) | under_tp_a
    clear_a = slot_off_a | grid_shift_a
    ask_qty = torch.where(clear_a, torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(clear_a, torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(clear_a, torch.zeros_like(ask_qa), ask_qa)

    empty_a = ask_qty <= 1e-6
    place_a = empty_a & (a_w >= 0.12) & hp_mask
    a_w_active = torch.where(place_a, a_w, torch.zeros_like(a_w))
    a_sum = a_w_active.sum(dim=1, keepdim=True)
    scale_a = torch.where(a_sum > 1e-5, 1.0 / torch.clamp(a_sum, min=1e-6), torch.zeros_like(a_sum))
    a_alloc = a_w_active * scale_a

    eff_a_lvl = torch.clamp(torch.round((t_ask_px - a1_2d) / tick_size).long(), 0, 9)
    asks_q_row = torch.clamp(asks_q_t_w, min=10.0)
    lvl_a_q_2d = torch.gather(asks_q_row, 1, eff_a_lvl)
    inside_sp_a = t_ask_px < (a1_2d - 0.1 * tick_size)
    base_a_qa = torch.where(inside_sp_a, torch.zeros_like(lvl_a_q_2d), lvl_a_q_2d)

    unreserved_pos = torch.clamp(pos_qty - ask_qty.sum(dim=1), min=0.0).unsqueeze(1)
    t_ask_qty = unreserved_pos * a_alloc
    need_lifo_cap = has_low_tr & (base_lifo_tp < base_avg_tp - 0.1 * tick_size)
    if allow_scratch:
        lifo_already_placed = ask_qty[:, 2:6].sum(dim=1, keepdim=True)
        rem_low_tr = torch.clamp(low_tranche_qty.unsqueeze(1) - lifo_already_placed, min=0.0)
        want_low_sum = t_ask_qty[:, 2:6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        low_cap_scale = torch.where(
            need_lifo_cap.unsqueeze(1) & (want_low_sum > rem_low_tr), rem_low_tr / want_low_sum, torch.ones_like(want_low_sum)
        )
        t_ask_qty = torch.cat([t_ask_qty[:, :2], t_ask_qty[:, 2:6] * low_cap_scale, t_ask_qty[:, 6:]], dim=1)
    else:
        lifo_already_placed = ask_qty[:, :6].sum(dim=1, keepdim=True)
        rem_low_tr = torch.clamp(low_tranche_qty.unsqueeze(1) - lifo_already_placed, min=0.0)
        want_low_sum = t_ask_qty[:, :6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        low_cap_scale = torch.where(
            need_lifo_cap.unsqueeze(1) & (want_low_sum > rem_low_tr), rem_low_tr / want_low_sum, torch.ones_like(want_low_sum)
        )
        t_ask_qty = torch.cat([t_ask_qty[:, :6] * low_cap_scale, t_ask_qty[:, 6:]], dim=1)

    t_ask_usd = t_ask_qty * t_ask_px
    valid_a = place_a & (t_ask_qty >= 1.0)
    t_ask_qty = torch.where(valid_a, t_ask_qty, torch.zeros_like(t_ask_qty))
    a_over50 = torch.sigmoid((t_ask_usd - 50.0) / 18.0)
    a_self_impact_qa = base_a_qa * (1.0 + 0.75 * a_over50) + a_over50 * (0.45 * t_ask_qty)

    ask_px = torch.where(valid_a, t_ask_px, ask_px)
    ask_qty = torch.where(valid_a, t_ask_qty, ask_qty)
    ask_qa = torch.where(valid_a, a_self_impact_qa, ask_qa)
    ask_act_t = torch.where(valid_a, t_2d + 1, ask_act_t)

    nb1 = torch.clamp(nb1_w, min=tick_size).unsqueeze(1)
    na1 = torch.clamp(na1_w, min=nb1.squeeze(1) + tick_size).unsqueeze(1)
    sv = sv_w.unsqueeze(1)
    bv = bv_w.unsqueeze(1)
    next_t_2d = next_t_vec.unsqueeze(1)

    act_b = (bid_qty > 1e-6) & (next_t_2d >= bid_act_t)
    strict_cross_b = na1 <= (bid_px - 0.5 * tick_size)
    hit_b = act_b & (strict_cross_b | ((nb1 <= bid_px + 1e-6) & (sv > 0.0)))
    any_strict_b = (act_b & strict_cross_b).any(dim=1, keepdim=True)
    tot_flow_b = torch.where(any_strict_b, sv + 0.12 * b1_q.unsqueeze(1), sv)

    even_act_b = act_b[:, 0::2] & (torch.abs(bid_px[:, 1::2] - bid_px[:, 0::2]) < 0.1 * tick_size)
    Incremental_qa_b = bid_qa.clone()
    Incremental_qa_b[:, 1::2] = torch.where(
        even_act_b, torch.clamp(bid_qa[:, 1::2] - bid_qa[:, 0::2], min=0.0), bid_qa[:, 1::2]
    )
    eff_qa_b = torch.where(hit_b, Incremental_qa_b, torch.zeros_like(bid_qa))
    eff_qty_b = torch.where(hit_b, bid_qty, torch.zeros_like(bid_qty))

    slot_need_b = eff_qa_b + eff_qty_b
    cum_need_b = torch.cumsum(slot_need_b, dim=1)
    rem_flow_at_slot_b = torch.clamp(tot_flow_b - (cum_need_b - slot_need_b), min=0.0)

    qa_eaten_b = torch.minimum(eff_qa_b, rem_flow_at_slot_b)
    bid_qa = torch.where(hit_b, torch.clamp(bid_qa - qa_eaten_b, min=0.0), bid_qa)
    bid_qa[:, 1::2] = torch.where(
        act_b[:, 1::2] & even_act_b, torch.clamp(bid_qa[:, 1::2] - qa_eaten_b[:, 0::2], min=0.0), bid_qa[:, 1::2]
    )

    want_b = torch.minimum(eff_qty_b, torch.clamp(rem_flow_at_slot_b - eff_qa_b, min=0.0))
    cum_want_b = torch.cumsum(want_b, dim=1)
    max_afford_b = torch.clamp(cash * 0.998 / (b1 * (1.0 + maker_fee)), min=0.0).unsqueeze(1)
    f_q = torch.minimum(want_b, torch.clamp(max_afford_b - (cum_want_b - want_b), min=0.0))

    has_slot_b = f_q > 1e-5
    tot_fq = f_q.sum(dim=1)
    tot_cost_raw = (f_q * bid_px).sum(dim=1)
    tot_cost = tot_cost_raw * (1.0 + maker_fee)
    has_buy = tot_fq > 1e-5

    old_q = pos_qty
    new_q = old_q + tot_fq
    avg_fill_b_px = tot_cost_raw / torch.clamp(tot_fq, min=1e-6)
    is_dip_buy = (
        has_buy
        & (old_q > 1e-6)
        & (
            (avg_fill_b_px <= avg_entry - 0.15 * tick_size)
            | ((low_tranche_qty > 1e-6) & (avg_fill_b_px <= min_buy_px + 0.15 * tick_size))
        )
    )
    is_higher_buy_from_single = (
        has_buy
        & (old_q >= 1.95)
        & (low_tranche_qty <= 1e-6)
        & (~is_dip_buy)
        & (avg_fill_b_px >= avg_entry + 0.35 * tick_size)
    )
    low_tranche_qty = torch.where(
        has_buy & (old_q <= 1e-6),
        torch.zeros_like(low_tranche_qty),
        torch.where(
            is_dip_buy,
            low_tranche_qty + tot_fq,
            torch.where(is_higher_buy_from_single, old_q, low_tranche_qty),
        ),
    )
    min_buy_px = torch.where(
        has_buy,
        torch.where(
            old_q <= 1e-6,
            avg_fill_b_px,
            torch.where(
                is_dip_buy,
                torch.where(low_tranche_qty > tot_fq + 1e-5, torch.minimum(min_buy_px, avg_fill_b_px), avg_fill_b_px),
                torch.where(is_higher_buy_from_single, avg_entry, min_buy_px),
            ),
        ),
        min_buy_px,
    )
    avg_entry = torch.where(has_buy, (old_q * avg_entry + tot_cost_raw) / torch.clamp(new_q, min=1e-6), avg_entry)
    pos_cost_usd = torch.where(has_buy, pos_cost_usd + tot_cost, pos_cost_usd)
    entry_t = torch.where(has_buy & (old_q <= 1e-6), next_t_vec, entry_t)
    cash = torch.where(has_buy, torch.clamp(cash - tot_cost, min=0.0), cash)
    pos_qty = torch.where(has_buy, new_q, pos_qty)

    rem_ord_b = torch.clamp(bid_qty - f_q, min=0.0)
    bid_qty = torch.where(has_slot_b, torch.where(rem_ord_b >= 1.0, rem_ord_b, torch.zeros_like(rem_ord_b)), bid_qty)

    act_a = (ask_qty > 1e-6) & (next_t_2d >= ask_act_t) & (pos_qty.unsqueeze(1) > 1e-6)
    strict_cross_a = nb1 >= (ask_px + 0.5 * tick_size)
    hit_a = act_a & (strict_cross_a | ((na1 >= ask_px - 1e-6) & (bv > 0.0)))
    any_strict_a = (act_a & strict_cross_a).any(dim=1, keepdim=True)
    tot_flow_a = torch.where(any_strict_a, bv + 0.12 * a1_q.unsqueeze(1), bv)

    even_act_a = act_a[:, 0::2] & (torch.abs(ask_px[:, 1::2] - ask_px[:, 0::2]) < 0.1 * tick_size)
    Incremental_qa_a = ask_qa.clone()
    Incremental_qa_a[:, 1::2] = torch.where(
        even_act_a, torch.clamp(ask_qa[:, 1::2] - ask_qa[:, 0::2], min=0.0), ask_qa[:, 1::2]
    )
    eff_qa_a = torch.where(hit_a, Incremental_qa_a, torch.zeros_like(ask_qa))
    eff_qty_a = torch.where(hit_a, ask_qty, torch.zeros_like(ask_qty))

    slot_need_a = eff_qa_a + eff_qty_a
    cum_need_a = torch.cumsum(slot_need_a, dim=1)
    rem_flow_at_slot_a = torch.clamp(tot_flow_a - (cum_need_a - slot_need_a), min=0.0)

    qa_eaten_a = torch.minimum(eff_qa_a, rem_flow_at_slot_a)
    ask_qa = torch.where(hit_a, torch.clamp(ask_qa - qa_eaten_a, min=0.0), ask_qa)
    ask_qa[:, 1::2] = torch.where(
        act_a[:, 1::2] & even_act_a, torch.clamp(ask_qa[:, 1::2] - qa_eaten_a[:, 0::2], min=0.0), ask_qa[:, 1::2]
    )

    want_a = torch.minimum(eff_qty_a, torch.clamp(rem_flow_at_slot_a - eff_qa_a, min=0.0))
    cum_want_a = torch.cumsum(want_a, dim=1)
    f_qa = torch.minimum(want_a, torch.clamp(pos_qty.unsqueeze(1) - (cum_want_a - want_a), min=0.0))

    has_slot_a = f_qa > 1e-5
    tot_fqa = f_qa.sum(dim=1)
    tot_rev = (f_qa * ask_px).sum(dim=1) * (1.0 - maker_fee)
    has_sell = tot_fqa > 1e-5

    lifo_f_qa = f_qa[:, 2:6].sum(dim=1) if allow_scratch else f_qa[:, :6].sum(dim=1)
    sold_low_q = torch.minimum(lifo_f_qa, low_tranche_qty)
    sold_high_q = torch.clamp(tot_fqa - sold_low_q, min=0.0)
    sold_cost_usd = torch.minimum(
        pos_cost_usd,
        sold_low_q * torch.minimum(min_buy_px, avg_entry) * (1.0 + maker_fee)
        + sold_high_q * avg_entry * (1.0 + maker_fee),
    )

    cash = torch.where(has_sell, cash + tot_rev, cash)
    rem_p = torch.clamp(pos_qty - tot_fqa, min=0.0)
    pos_qty = torch.where(has_sell, rem_p, pos_qty)
    pos_cost_usd = torch.where(
        has_sell,
        torch.where(rem_p > 1e-6, torch.clamp(pos_cost_usd - sold_cost_usd, min=0.0), torch.zeros_like(pos_cost_usd)),
        pos_cost_usd,
    )
    avg_entry = torch.where(
        has_sell,
        torch.where(rem_p > 1e-6, pos_cost_usd / (rem_p * (1.0 + maker_fee)), torch.zeros_like(avg_entry)),
        avg_entry,
    )
    low_tranche_qty = torch.where(
        has_sell,
        torch.where(rem_p > 1e-6, torch.clamp(low_tranche_qty - sold_low_q, min=0.0), torch.zeros_like(low_tranche_qty)),
        low_tranche_qty,
    )
    min_buy_px = torch.where(
        pos_qty > 1e-6, torch.where(low_tranche_qty > 1e-4, min_buy_px, avg_entry), torch.zeros_like(min_buy_px)
    )

    rem_ord_a = torch.clamp(ask_qty - f_qa, min=0.0)
    ask_qty = torch.where(
        has_slot_a,
        torch.where(
            (rem_ord_a >= 1.0) & (pos_qty.unsqueeze(1) >= 1.0),
            torch.minimum(rem_ord_a, pos_qty.unsqueeze(1)),
            torch.zeros_like(rem_ord_a),
        ),
        ask_qty,
    )

    eq_now = cash + pos_qty * b1
    peak_eq = torch.maximum(peak_eq, eq_now)
    w_dd_usd = torch.maximum(w_dd_usd, peak_eq - eq_now)

    return (
        h_states,
        cash,
        pos_qty,
        pos_cost_usd,
        avg_entry,
        min_buy_px,
        low_tranche_qty,
        entry_t,
        bid_px,
        bid_qty,
        bid_qa,
        bid_act_t,
        ask_px,
        ask_qty,
        ask_qa,
        ask_act_t,
        peak_eq,
        w_dd_usd,
    )


@torch.no_grad()
def evaluate_on_fixed_val_basket(
    val_env: VectorizedFreeMultiOrderEnv,
    bf16_runner: FusedBF16BackboneRunner,
    lob_repr_all: torch.Tensor,
    v7_58d: torch.Tensor,
    cand_states: list[dict],
    val_windows: list[tuple[str, int]],
    win_len: int,
    tick_sz: float,
    device: torch.device,
) -> list[dict]:
    """
    100% параллельная оценка всех 9 окон `val_windows` и всех кандидатов `cand_states`
    ЗА ОДИН ПРОХОД В 600 ШАГОВ (вместо 5,400 шагов последовательного цикла!).
    Побитово эквивалентно поштучному прогону окон, живет в L2-кэше (~90 КБ) и работает в 9.5x быстрее (~1.0с)!
    """
    W = len(val_windows)
    C = len(cand_states)
    B_bud = 6
    E = C * B_bud
    N = W * E
    S = val_env.n_side_slots

    budget_tiers = torch.tensor([100.0, 300.0, 625.0, 1000.0, 1500.0, 2500.0], device=device)
    budget = budget_tiers.unsqueeze(0).expand(W * C, B_bud).reshape(N).contiguous()

    dtype = v7_58d.dtype
    w1_c = torch.stack([st["w1"].to(device=device, dtype=dtype) for st in cand_states], dim=0).contiguous()
    b1_c_pol = torch.stack([st["b1"].to(device=device, dtype=dtype) for st in cand_states], dim=0).contiguous()
    w2_c = torch.stack([st["w2"].to(device=device, dtype=dtype) for st in cand_states], dim=0).contiguous()
    b2_c_pol = torch.stack([st["b2"].to(device=device, dtype=dtype) for st in cand_states], dim=0).contiguous()

    w_starts = torch.tensor([int(ws) for _, ws in val_windows], dtype=torch.long, device=device)
    w_starts_rep = w_starts.repeat_interleave(E)  # [N]

    h_states = bf16_runner.init_hidden(batch_size=N)
    cash = budget.clone()
    pos_qty = torch.zeros(N, dtype=torch.float32, device=device)
    pos_cost_usd = torch.zeros(N, dtype=torch.float32, device=device)
    avg_entry = torch.zeros(N, dtype=torch.float32, device=device)
    min_buy_px = torch.zeros(N, dtype=torch.float32, device=device)
    low_tranche_qty = torch.zeros(N, dtype=torch.float32, device=device)
    entry_t = torch.zeros(N, dtype=torch.long, device=device)

    bid_px = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qty = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qa = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_act_t = torch.zeros((N, S), dtype=torch.long, device=device)
    ask_px = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qty = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qa = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_act_t = torch.zeros((N, S), dtype=torch.long, device=device)

    peak_eq = budget.clone()
    w_dd_usd = torch.zeros(N, dtype=torch.float32, device=device)

    for s in range(win_len - 1):
        t_vec = w_starts_rep + s
        next_t_vec = torch.clamp(t_vec + 1, max=val_env.max_steps - 1)

        (
            h_states,
            cash,
            pos_qty,
            pos_cost_usd,
            avg_entry,
            min_buy_px,
            low_tranche_qty,
            entry_t,
            bid_px,
            bid_qty,
            bid_qa,
            bid_act_t,
            ask_px,
            ask_qty,
            ask_qa,
            ask_act_t,
            peak_eq,
            w_dd_usd,
        ) = _multi_win_val_tick(
            t_vec,
            next_t_vec,
            val_env.flow_market[t_vec],
            val_env.bids[t_vec, 0],
            val_env.asks[t_vec, 0],
            val_env.bids_q[t_vec, 0],
            val_env.asks_q[t_vec, 0],
            val_env.roll_buy_30s[t_vec],
            val_env.bids[t_vec],
            val_env.asks[t_vec],
            val_env.bids_q[t_vec],
            val_env.asks_q[t_vec],
            val_env.bids[next_t_vec, 0],
            val_env.asks[next_t_vec, 0],
            val_env.sell_vol[next_t_vec],
            val_env.buy_vol[next_t_vec],
            lob_repr_all[t_vec],
            v7_58d[t_vec],
            bf16_runner.w_fl,
            bf16_runner.b_fl,
            bf16_runner.ln_fl_w,
            bf16_runner.ln_fl_b,
            bf16_runner.w_fu_lob,
            bf16_runner.w_fu_fl,
            bf16_runner.b_fu,
            bf16_runner.ln_fu_w,
            bf16_runner.ln_fu_b,
            bf16_runner.w_ih,
            bf16_runner.b_ih,
            bf16_runner.w_hh,
            bf16_runner.b_hh,
            bf16_runner.w_h1,
            bf16_runner.b_h1,
            bf16_runner.w_h2,
            bf16_runner.b_h2,
            w1_c,
            b1_c_pol,
            w2_c,
            b2_c_pol,
            val_env.slot_tick_offsets,
            val_env.slot_lob_lvl,
            val_env.is_penny_slot,
            budget,
            h_states,
            cash,
            pos_qty,
            pos_cost_usd,
            avg_entry,
            min_buy_px,
            low_tranche_qty,
            entry_t,
            bid_px,
            bid_qty,
            bid_qa,
            bid_act_t,
            ask_px,
            ask_qty,
            ask_qa,
            ask_act_t,
            peak_eq,
            w_dd_usd,
            W,
            C,
            B_bud,
            S,
            tick_sz,
            val_env.maker_fee,
            val_env.taker_fee,
            val_env.allow_scratch_stop_asks,
        )

    end_t_vec = w_starts_rep + (win_len - 1)
    end_b1 = torch.clamp(val_env.bids[end_t_vec, 0], min=tick_sz)
    rem = pos_qty > 1e-6
    # Урок 12: Честная рыночная переоценка остатка по Bid (Mark-to-Bid за вычетом комиссии мейкера) без «бесплатного возврата себестоимости»!
    liq_val = pos_qty * end_b1 * (1.0 - val_env.maker_fee)
    end_cash = torch.where(rem, cash + liq_val, cash)
    w_pnl_pct_mat = (100.0 * (end_cash - budget) / budget).view(W, E)
    w_dd_pct_mat = (100.0 * w_dd_usd / budget).view(W, E)

    n_norm = max(1, sum(1 for r, _ in val_windows if r == "NORM"))
    n_dump = max(1, sum(1 for r, _ in val_windows if r == "DUMP"))
    n_pump = max(1, sum(1 for r, _ in val_windows if r == "PUMP"))

    norm_pct_acc = torch.zeros(E, dtype=torch.float32, device=device)
    dump_pct_acc = torch.zeros(E, dtype=torch.float32, device=device)
    pump_pct_acc = torch.zeros(E, dtype=torch.float32, device=device)
    max_dd_pct_acc = w_dd_pct_mat.max(dim=0).values

    for wi, (reg_name, _) in enumerate(val_windows):
        if reg_name == "NORM":
            norm_pct_acc += w_pnl_pct_mat[wi] / n_norm
        elif reg_name == "DUMP":
            dump_pct_acc += w_pnl_pct_mat[wi] / n_dump
        else:
            pump_pct_acc += w_pnl_pct_mat[wi] / n_pump

    results = []
    for ci in range(C):
        s_idx = ci * B_bud
        e_idx = s_idx + B_bud
        n_p = float(norm_pct_acc[s_idx:e_idx].mean().item())
        d_p = float(dump_pct_acc[s_idx:e_idx].mean().item())
        p_p = float(pump_pct_acc[s_idx:e_idx].mean().item())
        dd_p = float(max_dd_pct_acc[s_idx:e_idx].mean().item())
        realistic_ev = 0.70 * n_p + 0.15 * p_p + 0.15 * d_p
        inactivity_pen = 650.0 if (abs(n_p) + abs(p_p) < 0.015 or (n_p <= 0.002 and p_p <= 0.002)) else 0.0
        val_score = (
            realistic_ev * 2200.0
            + n_p * 950.0
            + p_p * 650.0
            + d_p * 200.0
            - max(0.0, dd_p - 0.25) * 280.0
            - dd_p * 45.0
            - inactivity_pen
        )
        results.append({
            "val_score": val_score,
            "realistic_ev": realistic_ev,
            "norm_pct": n_p,
            "dump_pct": d_p,
            "pump_pct": p_p,
            "dd_pct": dd_p,
        })
    return results


def run_lesson_15(iterations: int = 600, n_groups: int = 2048):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_budgets = 6
    pop_size = n_groups * n_budgets  # 12288 (2048 групп × 6 бюджетов — 20.7 МБ в 32 МБ L2-кэше GPU!)

    log_msg("=" * 106)
    log_msg(
        f"🚀 УРОК 15: DEEP SWIGLU & SLOT-AWARE MASTERY (356D -> SwiGLU(384->256) -> 26D | {pop_size} СРЕД)"
    )
    log_msg("=" * 106)

    raw = np.load(CACHE_PATH)
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.01
    slice_len = min(450000, n_total)
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
    v7_58d = torch.cat([v7_raw_t, v7_vel8], dim=-1)
    del v7_raw_t, v7_vel8

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    policy = AntitheticFreeGridPolicyGrouped(
        backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, in_dim=356, h_dim=256, out_dim=26, device=device
    )
    v7_58d = v7_58d.to(dtype=policy.dtype)

    summary_path = SAVE_DIR / "lesson_15_summary.json"
    history = []
    start_iter = 1
    latest_ckpt = SAVE_DIR / "lesson_15_champ_latest.pt"

    if latest_ckpt.exists() and summary_path.exists():
        try:
            history = json.loads(summary_path.read_text(encoding="utf-8"))
            if history:
                start_iter = int(history[-1]["iteration"]) + 1
                ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
                policy.load_champion_into_population(ck)
                log_msg(f"🔄 Продолжение Урока 15 с итерации {start_iter}/{iterations}...")
        except Exception as e:
            log_msg(f"⚠️ Ошибка чтения чекпоинта Урока 15: {e}")

    if start_iter == 1:
        seed_path = RELEASE_CHAMP_PATH if RELEASE_CHAMP_PATH.exists() else L12_CHAMP_PATH
        if seed_path.exists():
            ck = torch.load(seed_path, map_location=device, weights_only=False)
            policy.load_champion_into_population(ck)
            log_msg(f"🧬 Урок 15: Бесшовный Zero-Loss Warm-Start из Чемпиона Урока 14 ({seed_path.name} -> 356D-SwiGLU) выполнен успешно!")
        else:
            raise FileNotFoundError(f"Не найден стартовый чекпоинт для Урока 15: {seed_path}")

    hof_state = {
        "w1": policy.w1[0].detach().cpu().clone(),
        "b1": policy.b1[0].detach().cpu().clone(),
        "w2": policy.w2[0].detach().cpu().clone(),
        "b2": policy.b2[0].detach().cpu().clone(),
    }

    lob_gpu_tmp = torch.from_numpy(lob_t).float().to(device)
    del lob_t
    with torch.no_grad():
        lob_repr_all = backbone.encode_lob_batch(lob_gpu_tmp).to(dtype=policy.dtype)
    del lob_gpu_tmp

    bf16_runner = FusedBF16BackboneRunner(backbone, policy.dtype, device)

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
        num_envs=pop_size,
        n_side_slots=12,
        tick_size=tick_sz,
        device=device,
        allow_scratch_stop_asks=True,  # Урок 11: Ask[0] = -1 тик, Ask[1] = 0 тиков (безубыток)!
    )
    # Выделенный микро-инстанс val_env на 18 сред (3 кандидата x 6 бюджетов, ~25 КБ в L1/L2 кэше GPU!)
    val_env = VectorizedFreeMultiOrderEnv(
        bids=vec_env.bids,
        asks=vec_env.asks,
        bids_q=vec_env.bids_q,
        asks_q=vec_env.asks_q,
        buy_vol=vec_env.buy_vol,
        sell_vol=vec_env.sell_vol,
        lob_tensors=vec_env.lob_tensors,
        flow_market=vec_env.flow_market,
        targets=vec_env.targets,
        num_envs=18,
        n_side_slots=12,
        tick_size=tick_sz,
        device=device,
        allow_scratch_stop_asks=True,
    )
    del sub, flow_m, targets_np
    gc.collect()
    torch.cuda.empty_cache()

    budget_tiers = torch.tensor([100.0, 300.0, 625.0, 1000.0, 1500.0, 2500.0], device=device)
    env_budgets = budget_tiers.unsqueeze(0).expand(n_groups, n_budgets).reshape(pop_size)

    rng = np.random.default_rng(202609)
    win_len = 600

    # Разделение истории на Train (первые 85%) и фиксированный Holdout Validation Basket (последние 15%)
    b1_all_np = vec_env.bids[:, 0].detach().cpu().numpy()
    bq_all_np = vec_env.bids_q[:, 0].detach().cpu().numpy()
    aq_all_np = vec_env.asks_q[:, 0].detach().cpu().numpy()
    bv_all_np = vec_env.buy_vol.detach().cpu().numpy()
    sv_all_np = vec_env.sell_vol.detach().cpu().numpy()
    split_idx = int(slice_len * 0.85)
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

    normal_starts, dump_starts, pump_starts = classify_windows(train_starts, require_tradable=False)
    val_norm, val_dump, val_pump = classify_windows(val_starts, require_tradable=True)

    # Фиксированная проверочная корзина из 9 окон (3 Normal + 3 Dump + 3 Pump) для защиты от Winner's Curse
    val_basket = []
    for k_i in range(3):
        val_basket.append(("NORM", int(val_norm[(k_i * len(val_norm) // 3) % len(val_norm)])))
        val_basket.append(("DUMP", int(val_dump[(k_i * len(val_dump) // 3) % len(val_dump)])))
        val_basket.append(("PUMP", int(val_pump[(k_i * len(val_pump) // 3) % len(val_pump)])))

    init_val_res = evaluate_on_fixed_val_basket(
        val_env, bf16_runner, lob_repr_all, v7_58d, [hof_state], val_basket, win_len, tick_sz, device
    )[0]
    best_val_score = float(init_val_res["val_score"])
    best_val_ev = float(init_val_res["realistic_ev"])
    best_val_state = {k: v.clone() for k, v in hof_state.items()}
    last_val_champ_it = max(0, start_iter - 1)

    log_msg(
        f"⚖️ УРОК 15 БАЗА: Train=[N:{len(normal_starts)}, D:{len(dump_starts)}, P:{len(pump_starts)}] | "
        f"Fixed ValBasket (9 окон): ValScore={best_val_score:+.2f} | EV={best_val_ev:+.4f}% [N:{init_val_res['norm_pct']:+.2f}% "
        f"D:{init_val_res['dump_pct']:+.2f}% P:{init_val_res['pump_pct']:+.2f}% DD:{init_val_res['dd_pct']:.2f}%]"
    )

    for _skip in range(1, start_iter):
        _ = rng.integers(0, 1000, size=4)

    for it in range(start_iter, iterations + 1):
        t0 = time.time()
        progress = float(it - 1) / float(max(1, iterations - 1))
        # Косинусное затухание радиуса мутации и шага обучения для гарантированной сходимости
        cosine_decay = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress))

        total_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        norm_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        dump_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        pump_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)

        max_real_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_paper_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_hold_s = torch.zeros(pop_size, dtype=torch.float32, device=device)
        cycles = torch.zeros(pop_size, dtype=torch.long, device=device)
        prof_cycles = torch.zeros(pop_size, dtype=torch.long, device=device)
        scratch_evasion_bonus = torch.zeros(pop_size, dtype=torch.float32, device=device)
        tuw_pen = torch.zeros(pop_size, dtype=torch.float32, device=device)
        pump_capture_bonus = torch.zeros(pop_size, dtype=torch.float32, device=device)

        # Аккумуляторы пошагового градиента преимущества (Trajectory Advantage Gradient) для чемпиона (group 0)
        pg_dw2 = torch.zeros((256, 26), dtype=torch.float32, device=device)
        pg_db2 = torch.zeros((1, 26), dtype=torch.float32, device=device)
        pg_steps = 0

        vec_env.n_trades.zero_()
        vec_env.partial_fills.zero_()
        vec_env.multi_orders_used.zero_()
        vec_env.frontrun_hits.zero_()
        vec_env.sell_cycles.zero_()
        vec_env.prof_sell_cycles.zero_()

        # Ультра-интенсифицированный стратифицированный мини-батч из 9 окон (3 Normal + 3 Dump + 3 Pump = 5,400с × 12,288 сред)
        w_batch = [
            ("NORM1", int(rng.choice(normal_starts))),
            ("NORM2", int(rng.choice(normal_starts))),
            ("NORM3", int(rng.choice(normal_starts))),
            ("DUMP1", int(rng.choice(dump_starts))),
            ("DUMP2", int(rng.choice(dump_starts))),
            ("DUMP3", int(rng.choice(dump_starts))),
            ("PUMP1", int(rng.choice(pump_starts))),
            ("PUMP2", int(rng.choice(pump_starts))),
            ("PUMP3", int(rng.choice(pump_starts))),
        ]

        with torch.no_grad():
            # Недублированные веса групп [2048, 356, 384] -> единый Fused Tensor Core BMM!
            w1_g = policy.w1[0::n_budgets].contiguous()
            b1_g = policy.b1[0::n_budgets].contiguous()
            w2_g = policy.w2[0::n_budgets].contiguous()
            b2_g = policy.b2[0::n_budgets].contiguous()

            adv_vec_dump = torch.zeros(26, dtype=torch.float32, device=device)
            adv_vec_dump[0:2] = -0.45   # Запрет входа всей котлетой на первом уровне перед проливом
            adv_vec_dump[4:10] = +0.20  # Эшелонированный подбор на дне пролива (-2..-3 тика)
            adv_vec_dump[12:14] = +0.60 # Ask[0..1]: Scratch/Micro-Stop
            adv_vec_dump[24] = +0.30    # cancel_bids только при реальном проливе

            adv_vec_pump = torch.zeros(26, dtype=torch.float32, device=device)
            adv_vec_pump[0:2] = +0.28   # Умеренный первый транш на лучшем биде
            adv_vec_pump[2:6] = +0.38   # Основная лесенка добора на -1..-2 тика
            adv_vec_pump[12:14] = -0.45
            adv_vec_pump[14:24] = +0.42 # LIFO Ask[2..5] + дальние Ask[6..11]
            adv_vec_pump[24] = -0.50    # Queue Persistence: запрет холостого сдергивания ордеров из очереди

            for reg_name, w_start in w_batch:
                vec_env.reset_with_budgets(env_budgets)
                h_states = bf16_runner.init_hidden(batch_size=pop_size)
                prev_pos = vec_env.pos_qty.clone()
                cur_hold = torch.zeros(pop_size, dtype=torch.float32, device=device)
                peak_real_eq = env_budgets.clone()
                peak_paper_eq = env_budgets.clone()
                start_b1 = float(b1_all_np[w_start])

                for s in range(win_len - 1):
                    cur_t = int(w_start + s)
                    cur_b1_f = float(b1_all_np[cur_t])
                    _, flow, _, multi_6d = vec_env.get_observations(cur_t_scalar=cur_t)
                    z, p4, h_states = bf16_runner.step(lob_repr_all[cur_t : cur_t + 1], flow, h_states)
                    v7_row = v7_58d[cur_t]
                    h1_g, raw_out = _jit_policy_forward(
                        z,
                        p4,
                        v7_row[:58],
                        v7_row[58:66],
                        multi_6d,
                        vec_env.bid_qty,
                        vec_env.ask_qty,
                        vec_env.pos_qty,
                        env_budgets,
                        vec_env.bids[cur_t, 0],
                        w1_g,
                        b1_g,
                        w2_g,
                        b2_g,
                        n_groups,
                        n_budgets,
                        pop_size,
                    )

                    vec_env.step(raw_out, cur_t_scalar=cur_t)

                    is_bleed = bool(vec_env.is_bleed_cpu[cur_t])
                    is_pump_tick = bool(cur_b1_f >= start_b1 + 1.5 * tick_sz)
                    (
                        cur_hold,
                        max_hold_s,
                        peak_real_eq,
                        max_real_dd_usd,
                        peak_paper_eq,
                        max_paper_dd_usd,
                        tuw_pen,
                        scratch_evasion_bonus,
                        pump_capture_bonus,
                    ) = _jit_step_metrics(
                        vec_env.pos_qty,
                        vec_env.cash,
                        vec_env.pos_cost_usd,
                        vec_env.avg_entry,
                        vec_env.ask_qty,
                        raw_out,
                        prev_pos,
                        env_budgets,
                        vec_env.bids[cur_t, 0],
                        cur_hold,
                        max_hold_s,
                        peak_real_eq,
                        max_real_dd_usd,
                        peak_paper_eq,
                        max_paper_dd_usd,
                        tuw_pen,
                        scratch_evasion_bonus,
                        pump_capture_bonus,
                        tick_sz,
                        is_bleed,
                        is_pump_tick,
                    )

                    prev_pos.copy_(vec_env.pos_qty)

                end_t = int(w_start + win_len - 1)
                end_b1 = torch.clamp(vec_env.bids[end_t, 0], min=tick_sz)
                rem = vec_env.pos_qty > 1e-6
                # Урок 12: Строгий Mark-to-Bid за вычетом комиссии мейкера (никаких бесплатных возвратов pos_cost_usd!)
                liq_val = vec_env.pos_qty * end_b1 * (1.0 - vec_env.maker_fee)
                vec_env.cash = torch.where(rem, vec_env.cash + liq_val, vec_env.cash)
                vec_env.pos_qty = torch.where(rem, torch.zeros_like(vec_env.pos_qty), vec_env.pos_qty)
                vec_env.pos_cost_usd = torch.where(rem, torch.zeros_like(vec_env.pos_cost_usd), vec_env.pos_cost_usd)
                max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - vec_env.cash)

                w_pnl = vec_env.cash - env_budgets
                total_pnl_usd += w_pnl * (4.0 / 9.0)
                if reg_name.startswith("NORM"):
                    norm_pnl_usd += w_pnl / 3.0
                elif reg_name.startswith("DUMP"):
                    dump_pnl_usd += w_pnl / 3.0
                else:
                    pump_pnl_usd += w_pnl / 3.0

            cycles.copy_(vec_env.sell_cycles)
            prof_cycles.copy_(vec_env.prof_sell_cycles)

            pnl_pct = 100.0 * total_pnl_usd / env_budgets
            norm_pct = 100.0 * norm_pnl_usd / env_budgets
            dump_pct = 100.0 * dump_pnl_usd / env_budgets
            pump_pct = 100.0 * pump_pnl_usd / env_budgets

            real_dd_pct = 100.0 * max_real_dd_usd / env_budgets
            paper_dd_pct = 100.0 * max_paper_dd_usd / env_budgets

            avg_pnl_per_cyc = torch.where(
                prof_cycles > 0,
                torch.clamp(pnl_pct / torch.clamp(prof_cycles.float(), min=1.0), min=0.0, max=0.50),
                torch.zeros_like(pnl_pct),
            )

            # Целевая функция Урока 13: Front-Queue & Tick-Aligned Spread Mastery:
            tranche_fill_bonus = torch.clamp(vec_env.partial_fills.float(), max=60.0) * 5.0 + torch.clamp(vec_env.multi_orders_used.float() * 0.015, max=40.0)
            env_scores = (
                norm_pct * 3200.0
                + dump_pct * 950.0
                + torch.clamp(pump_pct, min=-1.0, max=4.5) * 1400.0
                + avg_pnl_per_cyc * 1600.0
                + torch.clamp(total_pnl_usd * 10.0, min=-220.0, max=580.0)
                - real_dd_pct * 1750.0
                - torch.clamp(paper_dd_pct - 0.30, min=0.0) * 380.0
                - torch.clamp(tuw_pen, max=260.0) * 3.0
                + torch.clamp(scratch_evasion_bonus, max=26.0) * 16.0
                + torch.clamp(pump_capture_bonus, max=180.0) * 3.5
                + torch.clamp(prof_cycles.float(), max=55.0) * 45.0
                + tranche_fill_bonus
                - torch.clamp(max_hold_s - 165.0, min=0.0) * 0.35
            )
            env_scores = torch.where(cycles == 0, env_scores - 450.0, env_scores)
            env_scores = torch.where(total_pnl_usd <= 0.0, env_scores - 180.0, env_scores)

            g_scores_mat = env_scores.view(n_groups, n_budgets)
            g_pnl_pct_mat = pnl_pct.view(n_groups, n_budgets)
            g_norm_pct_mat = norm_pct.view(n_groups, n_budgets)
            g_dump_pct_mat = dump_pct.view(n_groups, n_budgets)
            g_pump_pct_mat = pump_pct.view(n_groups, n_budgets)
            g_pnl_usd_mat = total_pnl_usd.view(n_groups, n_budgets)
            g_dd_pct_mat = real_dd_pct.view(n_groups, n_budgets)
            g_dd_usd_mat = max_real_dd_usd.view(n_groups, n_budgets)
            g_pdd_pct_mat = paper_dd_pct.view(n_groups, n_budgets)
            g_cyc_mat = cycles.view(n_groups, n_budgets)
            g_pcyc_mat = prof_cycles.view(n_groups, n_budgets)
            g_pf_mat = vec_env.partial_fills.view(n_groups, n_budgets)
            g_mo_mat = vec_env.multi_orders_used.view(n_groups, n_budgets)
            g_hold_mat = max_hold_s.view(n_groups, n_budgets)

            g_prof_buds = (g_pnl_usd_mat > 0.0).long().sum(dim=1)
            group_scores = g_scores_mat.mean(dim=1) + g_prof_buds.float() * 85.0

            c0_pnl_pct = float(g_pnl_pct_mat[0].mean().item())
            c0_pnl_usd = float(g_pnl_usd_mat[0].sum().item())
            c0_dd_pct = float(g_dd_pct_mat[0].mean().item())
            c0_score = float(group_scores[0].item())

            sorted_g = torch.argsort(group_scores, descending=True)
            best_g = int(sorted_g[0].item())

            b_score = float(group_scores[best_g].item())
            b_pnl_pct = float(g_pnl_pct_mat[best_g].mean().item())
            b_norm_pct = float(g_norm_pct_mat[best_g].mean().item())
            b_dump_pct = float(g_dump_pct_mat[best_g].mean().item())
            b_pump_pct = float(g_pump_pct_mat[best_g].mean().item())
            b_pnl_usd = float(g_pnl_usd_mat[best_g].sum().item())
            b_dd_pct = float(g_dd_pct_mat[best_g].mean().item())
            b_dd_usd = float(g_dd_usd_mat[best_g].mean().item())
            b_pdd_pct = float(g_pdd_pct_mat[best_g].mean().item())
            b_prof_buds = int(g_prof_buds[best_g].item())
            b_cyc = int(g_cyc_mat[best_g].float().mean().item())
            b_prof = int(g_pcyc_mat[best_g].float().mean().item())
            b_pf = int(g_pf_mat[best_g].float().mean().item())
            b_mo = int(g_mo_mat[best_g].float().mean().item())
            b_hold = float(g_hold_mat[best_g].mean().item())

            # Формируем кандидатное обновление через NES (Rank-Weighted Top-24 Elites) + пошаговый PG-градиент:
            top_k = 24
            top_g_idx = sorted_g[:top_k]
            top_env_idx = top_g_idx * n_budgets
            ranks = torch.arange(top_k, dtype=torch.float32, device=device)
            nes_w = torch.softmax(-ranks * 0.25, dim=0)

            cand_w1 = (policy.w1[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_b1 = (policy.b1[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_w2 = (policy.w2[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_b2 = (policy.b2[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)

            if pg_steps > 0:
                pg_lr = 0.0035 * cosine_decay / float(pg_steps)
                cand_w2 = cand_w2 + pg_lr * pg_dw2
                cand_b2 = cand_b2 + pg_lr * pg_db2

            cand_w1, cand_b1, cand_w2, cand_b2 = policy._normalize_single_weights(cand_w1, cand_b1, cand_w2, cand_b2)

            # Плавный шаг Поляка (Polyak Momentum) для текущего рабочего центра популяции:
            polyak_alpha = 0.18 * cosine_decay
            hof_w1_new = (1.0 - polyak_alpha) * hof_state["w1"].float() + polyak_alpha * cand_w1.detach().cpu().float()
            hof_b1_new = (1.0 - polyak_alpha) * hof_state["b1"].float() + polyak_alpha * cand_b1.detach().cpu().float()
            hof_w2_new = (1.0 - polyak_alpha) * hof_state["w2"].float() + polyak_alpha * cand_w2.detach().cpu().float()
            hof_b2_new = (1.0 - polyak_alpha) * hof_state["b2"].float() + polyak_alpha * cand_b2.detach().cpu().float()

            cand_polyak_state = {
                "w1": hof_w1_new.to(hof_state["w1"].dtype),
                "b1": hof_b1_new.to(hof_state["b1"].dtype),
                "w2": hof_w2_new.to(hof_state["w2"].dtype),
                "b2": hof_b2_new.to(hof_state["b2"].dtype),
            }
            best_env0 = best_g * n_budgets
            bw1, bb1, bw2, bb2 = policy._normalize_single_weights(
                policy.w1[best_env0], policy.b1[best_env0], policy.w2[best_env0], policy.b2[best_env0]
            )
            cand_best_state = {
                "w1": bw1.detach().cpu(),
                "b1": bb1.detach().cpu(),
                "w2": bw2.detach().cpu(),
                "b2": bb2.detach().cpu(),
            }

            # Обновляем рабочий центр популяции плавным шагом Поляка, если элита лучше group 0
            if b_score > c0_score:
                hof_state = cand_polyak_state

            # 7-точечный параллельный Trust-Region Line Search на фиксированной корзине валидации (9 окон) в L2 кэше:
            hof_updated = False
            if it % 4 == 0 or it == iterations or (b_score > c0_score + 50.0 and b_prof_buds >= 5):
                ls_alphas = [0.03, 0.08, 0.18, 0.35, 0.65, 1.00]
                ls_candidates = []
                cand_cpu_w1 = cand_w1.detach().cpu().float()
                cand_cpu_b1 = cand_b1.detach().cpu().float()
                cand_cpu_w2 = cand_w2.detach().cpu().float()
                cand_cpu_b2 = cand_b2.detach().cpu().float()
                base_st = best_val_state
                for a_ls in ls_alphas:
                    lw1 = (1.0 - a_ls) * base_st["w1"].float() + a_ls * cand_cpu_w1
                    lb1 = (1.0 - a_ls) * base_st["b1"].float() + a_ls * cand_cpu_b1
                    lw2 = (1.0 - a_ls) * base_st["w2"].float() + a_ls * cand_cpu_w2
                    lb2 = (1.0 - a_ls) * base_st["b2"].float() + a_ls * cand_cpu_b2
                    lw1, lb1, lw2, lb2 = policy._normalize_single_weights(lw1, lb1, lw2, lb2)
                    ls_candidates.append({"w1": lw1, "b1": lb1, "w2": lw2, "b2": lb2})
                ls_candidates.append(cand_best_state)

                val_res_list = evaluate_on_fixed_val_basket(
                    val_env,
                    bf16_runner,
                    lob_repr_all,
                    v7_58d,
                    ls_candidates,
                    val_basket,
                    win_len,
                    tick_sz,
                    device,
                )
                for c_idx, v_res in enumerate(val_res_list):
                    if float(v_res["val_score"]) > best_val_score:
                        best_val_score = float(v_res["val_score"])
                        best_val_ev = float(v_res["realistic_ev"])
                        best_val_state = {k: v.clone() for k, v in ls_candidates[c_idx].items()}
                        hof_state = {k: v.clone() for k, v in best_val_state.items()}
                        hof_updated = True
                        last_val_champ_it = it
                        ls_tag = f"alpha={ls_alphas[c_idx]:.2f}" if c_idx < len(ls_alphas) else "best_elite"
                        log_msg(
                            f"  💎 [NEW L15 VAL CHAMPION it={it} ({ls_tag})] ValScore={best_val_score:+.2f} | ValEV={best_val_ev:+.4f}% "
                            f"[N:{v_res['norm_pct']:+.2f}% D:{v_res['dump_pct']:+.2f}% P:{v_res['pump_pct']:+.2f}% DD:{v_res['dd_pct']:.2f}%]"
                        )

            # Если рабочий центр отклонился без улучшения на валидации в течение 25 итераций — возвращаем его к best_val_state
            if it % 25 == 0 and not hof_updated:
                hof_state = {k: v.clone() for k, v in best_val_state.items()}

            policy.evolve_groups(
                group_scores,
                hof_state=hof_state,
                elite_frac=0.03,
                progress=min(0.95, 0.25 + 0.70 * progress),  # Плавное сужение шума для сходимости
            )

            rec = {
                "iteration": it,
                "n_slots": 24,
                "best_idx": best_g,
                "budget": 6025.0,
                "prof_budgets": b_prof_buds,
                "hof_updated": hof_updated,
                "score": round(b_score, 2),
                "pnl_pct": round(b_pnl_pct, 4),
                "norm_pct": round(b_norm_pct, 4),
                "dump_pct": round(b_dump_pct, 4),
                "pump_pct": round(b_pump_pct, 4),
                "pnl_usd": round(b_pnl_usd, 4),
                "champ_oos_pnl_pct": round(c0_pnl_pct, 4),
                "champ_oos_pnl_usd": round(c0_pnl_usd, 4),
                "champ_oos_dd_pct": round(c0_dd_pct, 4),
                "best_val_ev_pct": round(best_val_ev, 4),
                "max_dd_pct": round(b_dd_pct, 4),
                "max_dd_usd": round(b_dd_usd, 4),
                "paper_dd_pct": round(b_pdd_pct, 4),
                "max_hold_s": round(b_hold, 1),
                "cycles": b_cyc,
                "prof_cycles": b_prof,
                "partial_fills": b_pf,
                "multi_order_ticks": b_mo,
            }
            history.append(rec)
            summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            tail20 = history[-20:]
            r20_pnl = float(np.mean([x["pnl_pct"] for x in tail20]))
            r20_oos = float(np.mean([x.get("champ_oos_pnl_pct", x["pnl_pct"]) for x in tail20]))
            r20_dd = float(np.mean([x["max_dd_pct"] for x in tail20]))
            dt = time.time() - t0
            star = "★" if hof_updated else "·"

            log_msg(
                f"🎯 L15 [{it}/{iterations}]{star} ({b_prof_buds}/6+) | "
                f"Batch4: {b_pnl_pct:+.2f}% [N:{b_norm_pct:+.2f}% D:{b_dump_pct:+.2f}% P:{b_pump_pct:+.2f}%] | "
                f"ChampOOS: {c0_pnl_pct:+.2f}% | ValEV: {best_val_ev:+.3f}% | DD: {b_dd_pct:.2f}% | Hold: {b_hold:.0f}s | "
                f"📈Roll20: {r20_pnl:+.2f}% (OOS {r20_oos:+.2f}%, DD {r20_dd:.2f}%) ({dt:.1f}с)"
            )

            # Сохраняем строго монотонный best_val_state как главный чекпоинт!
            ckpt_payload = {
                "iteration": it,
                "release_name": f"L15_{it:04d}",
                "n_slots": 24,
                "allow_scratch_stop_asks": True,
                "score": best_val_score,
                "val_ev_pct": best_val_ev,
                "pnl_pct": b_pnl_pct,
                "pnl_usd": b_pnl_usd,
                "backbone_state": backbone.state_dict(),
                "w1": best_val_state["w1"],
                "b1": best_val_state["b1"],
                "w2": best_val_state["w2"],
                "b2": best_val_state["b2"],
            }
            torch.save(ckpt_payload, latest_ckpt)
            if hof_updated:
                torch.save(ckpt_payload, RELEASE_CHAMP_PATH)
            if it % 50 == 0 or it == iterations or hof_updated:
                torch.save(ckpt_payload, SAVE_DIR / f"lesson_15_champ_iter_{it:04d}.pt")

            if (it - last_val_champ_it) >= 50:
                log_msg(
                    f"🛑 [EARLY STOP L15] {it - last_val_champ_it} эпох без обновления ValScore (последний рекорд на it={last_val_champ_it}). Остановка обучения."
                )
                break

    log_msg("🏆 УРОК 15 (DEEP SWIGLU & SLOT-AWARE MASTERY) УСПЕШНО ЗАВЕРШЕН!")


if __name__ == "__main__":
    run_lesson_15(iterations=600, n_groups=2048)
