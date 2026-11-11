#!/usr/bin/env python3
"""
_tools/lesson_19_cross_venue_channel_gated_trainer.py
=============================================================================
Урок 19: Cross-Venue First-Order Alpha, Learnable Channel Gating & Calibrated NES
1. Вместо вырезанных 16D мертвого балласта в агент (388D) подаются каузальные сигналы 1-го порядка:
   - [278:288] (10D): `CrossVenue_5s_LeadLag` (арбитраж BO vs ST, 5с опережение BigONE, дисбаланс стакана и ленты BO).
   - [312:318]  (6D): `CrossVenueFirstOrderNet` (p_cv_sweep_12s, p_cv_bleed_60s, p_cv_rebound_30s, p_clean_fill_b0/-1t/-2t).
2. Learnable Block Channel Gate (12 семантических блоков -> 388D):
   - Автоматически приглушает шумные блоки (`Macro_Swarm`, `Slot_Distances`) с мягким L1-штрафом
     и усиливает высокоинформативные блоки (`CrossVenue_LeadLag`, `CrossVenueFirstOrder`, `Fast_5s_LOB_Impulse`).
   - При сохранении чекпоинта сворачивается напрямую в `w1_eff = w1 * gate_388[:, None]` (100% совместимость с live!).
3. Zero-Loss Warm-Start из очищенного чемпиона `L18_0065_pruned16` (`release_champion_v1.pt`).
=============================================================================
"""

from __future__ import annotations

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
from _tools.lesson_18_multistack_lifo_survival_trainer import (
    AntitheticFreeGridPolicyGrouped,
    FusedBF16BackboneRunner,
    _jit_policy_forward,
    _jit_step_metrics,
    evaluate_on_fixed_val_basket,
)

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
SEED_CHAMP_PATH = BASE_DIR / "models" / "prl" / "lesson_18_multistack_lifo_survival" / "lesson_18_champ_iter_0065_pruned16.pt"
RELEASE_CHAMP_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
SAVE_DIR = BASE_DIR / "models" / "prl" / "lesson_19_cross_venue_channel_gated"
LOG_FILE = BASE_DIR / "data" / "logs" / "lesson_19_cross_venue_channel_gated.log"

# 12 семантических входных блоков агента (суммарно 388D):
SEMANTIC_BLOCKS = [
    ("Z_Backbone_GRU", 0, 256, 0.85),
    ("Guard_Risk_Heads", 256, 260, 1.20),
    ("Short_Triad_Swarm", 260, 278, 1.15),
    ("CrossVenue_5s_LeadLag", 278, 288, 2.80),
    ("Macro_Swarm_60_120s", 288, 306, 0.55),
    ("Carousel_Hazard_Realism", 306, 312, 1.10),
    ("CrossVenue_FirstOrder_6D", 312, 318, 2.80),
    ("LiqRatio_LIFO_Edge", 318, 324, 1.35),
    ("Active_24Slot_Weights", 324, 348, 1.85),
    ("Swarm_5s_Velocity", 348, 356, 1.85),
    ("Active_24Slot_Distances", 356, 380, 0.65),
    ("Fast_5s_LOB_Impulse", 380, 388, 1.90),
]


def log_msg(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expand_block_gates_to_388(g_logits: torch.Tensor) -> torch.Tensor:
    """Преобразует логиты блоков (..., 12) в мультипликативный гейт (..., 388) в диапазоне [0, 2], где logit=0 -> 1.0."""
    gates_12 = 2.0 * torch.sigmoid(g_logits.float())
    parts = [gates_12[..., idx : idx + 1].expand(*gates_12.shape[:-1], e - s) for idx, (_, s, e, _) in enumerate(SEMANTIC_BLOCKS)]
    return torch.cat(parts, dim=-1)


class ChannelGatedCrossVenuePolicy(AntitheticFreeGridPolicyGrouped):
    """Расширяет популяционную политику обучаемым 12-блочным канальным гейтом и калиброванной шкалой мутаций."""

    def __init__(self, backbone: MultiTaskLOBNet, n_groups: int = 2048, n_budgets: int = 6, device: torch.device = torch.device("cuda")):
        super().__init__(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, in_dim=388, h_dim=256, out_dim=26, device=device)
        self.g_logits = torch.zeros((n_groups, 12), dtype=torch.float32, device=device)
        mut_profile = torch.ones((1, 1, 388, 1), dtype=self.dtype, device=device)
        for _, s, e, m_scale in SEMANTIC_BLOCKS:
            mut_profile[:, :, s:e, :] = float(m_scale)
        self.mut_profile = mut_profile

    def effective_group_w1(self) -> torch.Tensor:
        w1_g = self.w1[0 :: self.n_budgets].float()
        gate_388 = expand_block_gates_to_388(self.g_logits).unsqueeze(-1)
        return (w1_g * gate_388).to(dtype=self.dtype).contiguous()

    def fold_gate_into_state(self, state: dict, g_logit_1d: torch.Tensor) -> dict:
        gate_388 = expand_block_gates_to_388(g_logit_1d.to(self.device)).cpu().float().unsqueeze(-1)
        w1_folded = state["w1"].float() * gate_388
        w1_n, b1_n, wr1_n, br1_n, wr2_n, w2_n, b2_n = self._normalize_single_weights(
            w1_folded.to(self.device),
            state["b1"].to(self.device),
            state["w_res1"].to(self.device),
            state["b_res1"].to(self.device),
            state["w_res2"].to(self.device),
            state["w2"].to(self.device),
            state["b2"].to(self.device),
        )
        return {
            "w1": w1_n.detach().cpu(),
            "b1": b1_n.detach().cpu(),
            "w_res1": wr1_n.detach().cpu(),
            "b_res1": br1_n.detach().cpu(),
            "w_res2": wr2_n.detach().cpu(),
            "w2": w2_n.detach().cpu(),
            "b2": b2_n.detach().cpu(),
        }

    def evolve_gated_groups(self, group_scores: torch.Tensor, hof_state: dict, hof_g_logit: torch.Tensor, elite_frac: float = 0.03, progress: float = 0.0) -> int:
        B_buds = self.n_budgets
        H1_full = self.h1_full_dim
        R_full = self.res_full_dim
        R_mid = self.res_mid_dim
        n_elites = max(12, int(self.n_groups * elite_frac))
        if (self.n_groups - n_elites) % 2 != 0:
            n_elites += 1
        sorted_g = torch.argsort(group_scores, descending=True)
        elite_g = sorted_g[:n_elites]
        best_g = int(sorted_g[0].item())

        elite_env_idx = elite_g * B_buds
        w1_el = self.w1[elite_env_idx].clone()
        b1_el = self.b1[elite_env_idx].clone()
        wr1_el = self.w_res1[elite_env_idx].clone()
        br1_el = self.b_res1[elite_env_idx].clone()
        wr2_el = self.w_res2[elite_env_idx].clone()
        w2_el = self.w2[elite_env_idx].clone()
        b2_el = self.b2[elite_env_idx].clone()
        gl_el = torch.clamp(self.g_logits[elite_g].clone(), -2.5, 2.0)

        for ei in range(n_elites):
            w1_el[ei], b1_el[ei], wr1_el[ei], br1_el[ei], wr2_el[ei], w2_el[ei], b2_el[ei] = self._normalize_single_weights(
                w1_el[ei], b1_el[ei], wr1_el[ei], br1_el[ei], wr2_el[ei], w2_el[ei], b2_el[ei]
            )

        w1_el[0].copy_(hof_state["w1"].to(self.device, self.dtype))
        b1_el[0].copy_(hof_state["b1"].to(self.device, self.dtype))
        wr1_el[0].copy_(hof_state["w_res1"].to(self.device, self.dtype))
        br1_el[0].copy_(hof_state["b_res1"].to(self.device, self.dtype))
        wr2_el[0].copy_(hof_state["w_res2"].to(self.device, self.dtype))
        w2_el[0].copy_(hof_state["w2"].to(self.device, self.dtype))
        b2_el[0].copy_(hof_state["b2"].to(self.device, self.dtype))
        gl_el[0].copy_(hof_g_logit.to(self.device, torch.float32))

        self.w1[: n_elites * B_buds].view(n_elites, B_buds, self.in_dim, H1_full).copy_(w1_el.unsqueeze(1).expand(n_elites, B_buds, self.in_dim, H1_full))
        self.b1[: n_elites * B_buds].view(n_elites, B_buds, 1, H1_full).copy_(b1_el.unsqueeze(1).expand(n_elites, B_buds, 1, H1_full))
        self.w_res1[: n_elites * B_buds].view(n_elites, B_buds, self.h_dim, R_full).copy_(wr1_el.unsqueeze(1).expand(n_elites, B_buds, self.h_dim, R_full))
        self.b_res1[: n_elites * B_buds].view(n_elites, B_buds, 1, R_full).copy_(br1_el.unsqueeze(1).expand(n_elites, B_buds, 1, R_full))
        self.w_res2[: n_elites * B_buds].view(n_elites, B_buds, R_mid, self.h_dim).copy_(wr2_el.unsqueeze(1).expand(n_elites, B_buds, R_mid, self.h_dim))
        self.w2[: n_elites * B_buds].view(n_elites, B_buds, self.h_dim, self.out_dim).copy_(w2_el.unsqueeze(1).expand(n_elites, B_buds, self.h_dim, self.out_dim))
        self.b2[: n_elites * B_buds].view(n_elites, B_buds, 1, self.out_dim).copy_(b2_el.unsqueeze(1).expand(n_elites, B_buds, 1, self.out_dim))
        self.g_logits[:n_elites].copy_(gl_el)

        n_pairs = (self.n_groups - n_elites) // 2
        pair_parents = torch.where(
            torch.rand(n_pairs, device=self.device) < 0.55,
            torch.zeros(n_pairs, dtype=torch.long, device=self.device),
            torch.randint(0, n_elites, (n_pairs,), device=self.device),
        )
        decay = max(0.25, 1.0 - 0.68 * float(progress))
        sigma_pairs = torch.where(
            torch.rand(n_pairs, 1, 1, 1, device=self.device) < 0.72,
            torch.full((n_pairs, 1, 1, 1), 0.016 * decay, device=self.device, dtype=self.dtype),
            torch.full((n_pairs, 1, 1, 1), 0.040 * decay, device=self.device, dtype=self.dtype),
        )

        # Антитетическая мутация канальных гейтов (12 блоков):
        dst_gl = self.g_logits[n_elites:].view(n_pairs, 2, 12)
        dst_gl.copy_(gl_el[pair_parents].unsqueeze(1).expand(n_pairs, 2, 12))
        ngl = torch.randn((n_pairs, 12), device=self.device, dtype=torch.float32) * (0.09 * decay)
        ngl[:, 0] *= 0.25  # Backbone GRU не глушим резко
        dst_gl[:, 0].add_(ngl).clamp_(-2.5, 2.0)
        dst_gl[:, 1].sub_(ngl).clamp_(-2.5, 2.0)

        chunk_p = 128
        buf_w1_p = torch.empty((chunk_p, 1, self.in_dim, H1_full), device=self.device, dtype=self.dtype)
        for p0 in range(0, n_pairs, chunk_p):
            p1 = min(n_pairs, p0 + chunk_p)
            np_cur = p1 - p0
            g_start = n_elites + p0 * 2
            g_end = n_elites + p1 * 2
            sig_p = sigma_pairs[p0:p1]
            dst_w1 = self.w1[g_start * B_buds : g_end * B_buds].view(np_cur, 2, B_buds, self.in_dim, H1_full)
            dst_w1.copy_(w1_el[pair_parents[p0:p1]].unsqueeze(1).unsqueeze(2).expand(np_cur, 2, B_buds, self.in_dim, H1_full))
            bw = buf_w1_p[:np_cur]
            bw.normal_(0.0, 1.0).mul_(sig_p).mul_(self.mut_profile)
            dst_w1[:, 0].add_(bw.expand(np_cur, B_buds, self.in_dim, H1_full))
            dst_w1[:, 1].sub_(bw.expand(np_cur, B_buds, self.in_dim, H1_full))
        del buf_w1_p, w1_el

        dst_b1 = self.b1[n_elites * B_buds :].view(n_pairs, 2, B_buds, 1, H1_full)
        dst_b1.copy_(b1_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, 1, H1_full))
        nb1 = torch.randn((n_pairs, 1, 1, H1_full), device=self.device, dtype=self.dtype) * (sigma_pairs * 0.5)
        dst_b1[:, 0].add_(nb1.expand(n_pairs, B_buds, 1, H1_full))
        dst_b1[:, 1].sub_(nb1.expand(n_pairs, B_buds, 1, H1_full))

        dst_wr1 = self.w_res1[n_elites * B_buds :].view(n_pairs, 2, B_buds, self.h_dim, R_full)
        dst_wr1.copy_(wr1_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, self.h_dim, R_full))
        nwr1 = torch.randn((n_pairs, 1, self.h_dim, R_full), device=self.device, dtype=self.dtype) * (sigma_pairs * 1.8)
        dst_wr1[:, 0].add_(nwr1.expand(n_pairs, B_buds, self.h_dim, R_full))
        dst_wr1[:, 1].sub_(nwr1.expand(n_pairs, B_buds, self.h_dim, R_full))

        dst_br1 = self.b_res1[n_elites * B_buds :].view(n_pairs, 2, B_buds, 1, R_full)
        dst_br1.copy_(br1_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, 1, R_full))
        nbr1 = torch.randn((n_pairs, 1, 1, R_full), device=self.device, dtype=self.dtype) * (sigma_pairs * 0.8)
        dst_br1[:, 0].add_(nbr1.expand(n_pairs, B_buds, 1, R_full))
        dst_br1[:, 1].sub_(nbr1.expand(n_pairs, B_buds, 1, R_full))

        dst_wr2 = self.w_res2[n_elites * B_buds :].view(n_pairs, 2, B_buds, R_mid, self.h_dim)
        dst_wr2.copy_(wr2_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, R_mid, self.h_dim))
        nwr2 = torch.randn((n_pairs, 1, R_mid, self.h_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 2.5)
        dst_wr2[:, 0].add_(nwr2.expand(n_pairs, B_buds, R_mid, self.h_dim))
        dst_wr2[:, 1].sub_(nwr2.expand(n_pairs, B_buds, R_mid, self.h_dim))

        dst_w2 = self.w2[n_elites * B_buds :].view(n_pairs, 2, B_buds, self.h_dim, self.out_dim)
        dst_w2.copy_(w2_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, self.h_dim, self.out_dim))
        nw2 = torch.randn((n_pairs, 1, self.h_dim, self.out_dim), device=self.device, dtype=self.dtype) * sigma_pairs
        dst_w2[:, 0].add_(nw2.expand(n_pairs, B_buds, self.h_dim, self.out_dim))
        dst_w2[:, 1].sub_(nw2.expand(n_pairs, B_buds, self.h_dim, self.out_dim))

        dst_b2 = self.b2[n_elites * B_buds :].view(n_pairs, 2, B_buds, 1, self.out_dim)
        dst_b2.copy_(b2_el[pair_parents].unsqueeze(1).unsqueeze(2).expand(n_pairs, 2, B_buds, 1, self.out_dim))
        nb2 = torch.randn((n_pairs, 1, 1, self.out_dim), device=self.device, dtype=self.dtype) * (sigma_pairs * 1.4)
        dst_b2[:, 0].add_(nb2.expand(n_pairs, B_buds, 1, self.out_dim))
        dst_b2[:, 1].sub_(nb2.expand(n_pairs, B_buds, 1, self.out_dim))
        return best_g


def run_lesson_19(iterations: int = 160, n_groups: int = 2048):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_budgets = 6
    pop_size = n_groups * n_budgets

    log_msg("=" * 108)
    log_msg(f"🚀 УРОК 19: CROSS-VENUE FIRST-ORDER (ST+BO), CHANNEL GATING & CALIBRATED NES ({pop_size} СРЕД)")
    log_msg("=" * 108)

    raw = np.load(CACHE_PATH, allow_pickle=True)
    assert "has_cross_venue_v1" in raw.files, "Сначала запустите `_tools/cross_venue_first_order_net.py` для встраивания ST+BO сигналов 1-го порядка!"
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.01
    slice_len = min(945000, n_total)
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

    v7_58d = torch.cat([v7_raw_t, v7_vel8, lob_vel8], dim=-1)
    del v7_raw_t, v7_vel8, lob_vel8

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    policy = ChannelGatedCrossVenuePolicy(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, device=device)
    v7_58d = v7_58d.to(dtype=policy.dtype)

    summary_path = SAVE_DIR / "lesson_19_summary.json"
    latest_ckpt = SAVE_DIR / "lesson_19_champ_latest.pt"
    history = []
    start_iter = 1

    seed_path = SEED_CHAMP_PATH if SEED_CHAMP_PATH.exists() else RELEASE_CHAMP_PATH
    ck = torch.load(seed_path, map_location=device, weights_only=False)
    ck["lesson"] = 18
    # Убеждаемся, что освобожденные 16 каналов стартуют строго с 0.0 (100% Zero-Loss Warm-Start на новых ST+BO сигналах!):
    ck["w1"] = ck["w1"].clone()
    ck["w1"][278:288, :] = 0.0
    ck["w1"][312:318, :] = 0.0
    policy.load_champion_into_population(ck)
    # Применяем калиброванный профиль стартовых мутаций к группам 1..G-1:
    with torch.no_grad():
        B_buds = n_budgets
        w0 = policy.w1[0].clone()
        for g0 in range(1, n_groups, 256):
            g1 = min(n_groups, g0 + 256)
            gc_cnt = g1 - g0
            nw1 = torch.randn((gc_cnt, 1, 388, policy.h1_full_dim), device=device, dtype=policy.dtype) * 0.020 * policy.mut_profile
            policy.w1[g0 * B_buds : g1 * B_buds].view(gc_cnt, B_buds, 388, policy.h1_full_dim).copy_(
                (w0.unsqueeze(0).unsqueeze(1) + nw1).expand(gc_cnt, B_buds, 388, policy.h1_full_dim)
            )
    log_msg(f"🧬 Урок 19: Zero-Loss Warm-Start из {seed_path.name} + подключены 16D кросс-биржевых сигналов (SafeTrade + BigONE)!")

    hof_state = {
        "w1": policy.w1[0].detach().cpu().clone(),
        "b1": policy.b1[0].detach().cpu().clone(),
        "w_res1": policy.w_res1[0].detach().cpu().clone(),
        "b_res1": policy.b_res1[0].detach().cpu().clone(),
        "w_res2": policy.w_res2[0].detach().cpu().clone(),
        "w2": policy.w2[0].detach().cpu().clone(),
        "b2": policy.b2[0].detach().cpu().clone(),
    }
    hof_g_logit = torch.zeros(12, dtype=torch.float32)

    lob_repr_chunks = []
    with torch.no_grad():
        for i0 in range(0, len(lob_t), 100000):
            chunk_gpu = torch.from_numpy(lob_t[i0 : i0 + 100000]).float().to(device)
            lob_repr_chunks.append(backbone.encode_lob_batch(chunk_gpu).to(dtype=policy.dtype))
            del chunk_gpu
    del lob_t
    lob_repr_all = torch.cat(lob_repr_chunks, dim=0)
    del lob_repr_chunks

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
        allow_scratch_stop_asks=True,
    )
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
    rng = np.random.default_rng(202619)
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

    normal_starts, dump_starts, pump_starts = classify_windows(train_starts, require_tradable=False)
    val_norm, val_dump, val_pump = classify_windows(val_starts, require_tradable=True)

    val_basket = []
    for k_i in range(6):
        val_basket.append(("NORM", int(val_norm[(k_i * len(val_norm) // 6) % len(val_norm)])))
        val_basket.append(("DUMP", int(val_dump[(k_i * len(val_dump) // 6) % len(val_dump)])))
        val_basket.append(("PUMP", int(val_pump[(k_i * len(val_pump) // 6) % len(val_pump)])))
    val_basket.append(("UW_NORM", int(val_norm[len(val_norm) // 2])))
    val_basket.append(("UW_DUMP", int(val_dump[len(val_dump) // 2])))

    base_folded = policy.fold_gate_into_state(hof_state, hof_g_logit)
    init_val_res = evaluate_on_fixed_val_basket(
        val_env, bf16_runner, lob_repr_all, v7_58d, [base_folded] * 7, val_basket, win_len, tick_sz, device
    )[0]
    best_val_score = float(init_val_res["val_score"])
    best_val_ev = float(init_val_res["realistic_ev"])
    best_val_state = {k: v.clone() for k, v in hof_state.items()}
    best_val_g_logit = hof_g_logit.clone()
    best_val_folded = {k: v.clone() for k, v in base_folded.items()}
    last_val_champ_it = 0

    log_msg(
        f"⚖️ УРОК 19 БАЗА (L18_0065_pruned16 на свежих данных): Train=[N:{len(normal_starts)}, D:{len(dump_starts)}, P:{len(pump_starts)}] | "
        f"FULL_20 ValScore={best_val_score:+.2f} | EV={best_val_ev:+.4f}% [N:{init_val_res['norm_pct']:+.2f}% "
        f"D:{init_val_res['dump_pct']:+.2f}% P:{init_val_res['pump_pct']:+.2f}% DD:{init_val_res['dd_pct']:.2f}% | Act N:{init_val_res['n_act']:.0%} P:{init_val_res['p_act']:.0%}]"
    )

    for it in range(start_iter, iterations + 1):
        t0 = time.time()
        progress = float(it - 1) / float(max(1, iterations - 1))
        cosine_decay = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress))

        total_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        norm_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        dump_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        pump_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_real_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_paper_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_hold_s = torch.zeros(pop_size, dtype=torch.float32, device=device)
        tuw_pen = torch.zeros(pop_size, dtype=torch.float32, device=device)
        scratch_evasion_bonus = torch.zeros(pop_size, dtype=torch.float32, device=device)
        pump_capture_bonus = torch.zeros(pop_size, dtype=torch.float32, device=device)

        vec_env.n_trades.zero_()
        vec_env.partial_fills.zero_()
        vec_env.multi_orders_used.zero_()
        vec_env.frontrun_hits.zero_()
        vec_env.sell_cycles.zero_()
        vec_env.prof_sell_cycles.zero_()

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
            w1_g = policy.effective_group_w1()
            b1_g = policy.b1[0::n_budgets].contiguous()
            wr1_g = policy.w_res1[0::n_budgets].contiguous()
            br1_g = policy.b_res1[0::n_budgets].contiguous()
            wr2_g = policy.w_res2[0::n_budgets].contiguous()
            w2_g = policy.w2[0::n_budgets].contiguous()
            b2_g = policy.b2[0::n_budgets].contiguous()

            for reg_name, w_start in w_batch:
                vec_env.reset_with_budgets(env_budgets)
                if reg_name in ("NORM3", "DUMP3"):
                    vec_env.seed_underwater_inventory(w_start, inv_frac=0.55, underwater_ticks=5.0)
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
                    _, raw_out = _jit_policy_forward(
                        z, p4, v7_row[:58], v7_row[58:66], v7_row[66:74], multi_6d,
                        vec_env.bid_qty, vec_env.bid_px, vec_env.ask_qty, vec_env.ask_px,
                        vec_env.pos_qty, env_budgets, vec_env.bids[cur_t, 0], vec_env.asks[cur_t, 0], tick_sz,
                        w1_g, b1_g, wr1_g, br1_g, wr2_g, w2_g, b2_g, n_groups, n_budgets, pop_size,
                    )
                    vec_env.step(raw_out, cur_t_scalar=cur_t)
                    is_bleed = bool(vec_env.is_bleed_cpu[cur_t])
                    is_pump_tick = bool(cur_b1_f >= start_b1 + 1.5 * tick_sz)
                    (
                        cur_hold, max_hold_s, peak_real_eq, max_real_dd_usd,
                        peak_paper_eq, max_paper_dd_usd, tuw_pen, scratch_evasion_bonus, pump_capture_bonus,
                    ) = _jit_step_metrics(
                        vec_env.pos_qty, vec_env.cash, vec_env.pos_cost_usd, vec_env.avg_entry, vec_env.ask_qty,
                        raw_out, prev_pos, env_budgets, vec_env.bids[cur_t, 0],
                        cur_hold, max_hold_s, peak_real_eq, max_real_dd_usd, peak_paper_eq, max_paper_dd_usd,
                        tuw_pen, scratch_evasion_bonus, pump_capture_bonus, tick_sz, is_bleed, is_pump_tick,
                    )
                    prev_pos.copy_(vec_env.pos_qty)

                end_t = int(w_start + win_len - 1)
                end_b1 = torch.clamp(vec_env.bids[end_t, 0], min=tick_sz)
                rem = vec_env.pos_qty > 1e-6
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

            cycles = vec_env.sell_cycles.clone()
            prof_cycles = vec_env.prof_sell_cycles.clone()
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
            tranche_fill_bonus = torch.clamp(vec_env.partial_fills.float(), max=60.0) * 5.0 + torch.clamp(vec_env.multi_orders_used.float() * 0.015, max=40.0)
            env_scores = (
                norm_pct * 3200.0
                + dump_pct * 950.0
                + torch.clamp(pump_pct, min=-1.0, max=4.5) * 1450.0
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

            # Мягкая Information Bottleneck регуляризация на шумные блоки (индексы 4=Macro_Swarm, 10=Slot_Distances):
            gates_12 = 2.0 * torch.sigmoid(policy.g_logits)
            ib_pen = (gates_12[:, 4] + gates_12[:, 10]) * 4.0

            g_scores_mat = env_scores.view(n_groups, n_budgets)
            g_pnl_pct_mat = pnl_pct.view(n_groups, n_budgets)
            g_norm_pct_mat = norm_pct.view(n_groups, n_budgets)
            g_dump_pct_mat = dump_pct.view(n_groups, n_budgets)
            g_pump_pct_mat = pump_pct.view(n_groups, n_budgets)
            g_pnl_usd_mat = total_pnl_usd.view(n_groups, n_budgets)
            g_dd_pct_mat = real_dd_pct.view(n_groups, n_budgets)
            g_hold_mat = max_hold_s.view(n_groups, n_budgets)

            g_prof_buds = (g_pnl_usd_mat > 0.0).long().sum(dim=1)
            group_scores = g_scores_mat.mean(dim=1) + g_prof_buds.float() * 85.0 - ib_pen

            c0_pnl_pct = float(g_pnl_pct_mat[0].mean().item())
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
            b_prof_buds = int(g_prof_buds[best_g].item())
            b_hold = float(g_hold_mat[best_g].mean().item())

            top_k = 24
            top_g_idx = sorted_g[:top_k]
            top_env_idx = top_g_idx * n_budgets
            nes_w = torch.softmax(-torch.arange(top_k, dtype=torch.float32, device=device) * 0.25, dim=0)

            cand_w1 = (policy.w1[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_b1 = (policy.b1[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_wr1 = (policy.w_res1[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_br1 = (policy.b_res1[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_wr2 = (policy.w_res2[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_w2 = (policy.w2[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_b2 = (policy.b2[top_env_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_gl = (policy.g_logits[top_g_idx].float() * nes_w.view(-1, 1)).sum(dim=0).clamp(-2.5, 2.0)

            cand_w1, cand_b1, cand_wr1, cand_br1, cand_wr2, cand_w2, cand_b2 = policy._normalize_single_weights(
                cand_w1, cand_b1, cand_wr1, cand_br1, cand_wr2, cand_w2, cand_b2
            )

            polyak_alpha = 0.20 * cosine_decay
            polyak_new = min(0.55, 0.45 * cosine_decay)
            hof_w1_new = (1.0 - polyak_alpha) * hof_state["w1"].float() + polyak_alpha * cand_w1.detach().cpu().float()
            for idx_s, idx_e in ((278, 288), (312, 318)):
                hof_w1_new[idx_s:idx_e, :] = (1.0 - polyak_new) * hof_state["w1"][idx_s:idx_e, :].float() + polyak_new * cand_w1[idx_s:idx_e, :].detach().cpu().float()

            cand_polyak_state = {
                "w1": hof_w1_new.to(hof_state["w1"].dtype),
                "b1": ((1.0 - polyak_alpha) * hof_state["b1"].float() + polyak_alpha * cand_b1.detach().cpu().float()).to(hof_state["b1"].dtype),
                "w_res1": ((1.0 - polyak_alpha) * hof_state["w_res1"].float() + polyak_alpha * cand_wr1.detach().cpu().float()).to(hof_state["w_res1"].dtype),
                "b_res1": ((1.0 - polyak_alpha) * hof_state["b_res1"].float() + polyak_alpha * cand_br1.detach().cpu().float()).to(hof_state["b_res1"].dtype),
                "w_res2": ((1.0 - polyak_alpha) * hof_state["w_res2"].float() + polyak_alpha * cand_wr2.detach().cpu().float()).to(hof_state["w_res2"].dtype),
                "w2": ((1.0 - polyak_alpha) * hof_state["w2"].float() + polyak_alpha * cand_w2.detach().cpu().float()).to(hof_state["w2"].dtype),
                "b2": ((1.0 - polyak_alpha) * hof_state["b2"].float() + polyak_alpha * cand_b2.detach().cpu().float()).to(hof_state["b2"].dtype),
            }
            cand_polyak_gl = (1.0 - polyak_new) * hof_g_logit + polyak_new * cand_gl.detach().cpu()

            if b_score > c0_score:
                hof_state = cand_polyak_state
                hof_g_logit = cand_polyak_gl

            hof_updated = False
            if it % 3 == 0 or it == iterations or (b_score > c0_score + 45.0 and b_prof_buds >= 5):
                ls_alphas = [0.05, 0.12, 0.25, 0.45, 0.75, 1.00]
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
                }
                c_gl_cpu = cand_gl.detach().cpu().float()
                for a_ls in ls_alphas:
                    st_a = {k: (1.0 - a_ls) * best_val_state[k].float() + a_ls * c_cpu[k] for k in c_cpu}
                    gl_a = (1.0 - a_ls) * best_val_g_logit.float() + a_ls * c_gl_cpu
                    ls_raw_states.append(st_a)
                    ls_gls.append(gl_a)
                    ls_folded_candidates.append(policy.fold_gate_into_state(st_a, gl_a))

                best_env0 = best_g * n_budgets
                raw_best = {
                    "w1": policy.w1[best_env0].detach().cpu().float(),
                    "b1": policy.b1[best_env0].detach().cpu().float(),
                    "w_res1": policy.w_res1[best_env0].detach().cpu().float(),
                    "b_res1": policy.b_res1[best_env0].detach().cpu().float(),
                    "w_res2": policy.w_res2[best_env0].detach().cpu().float(),
                    "w2": policy.w2[best_env0].detach().cpu().float(),
                    "b2": policy.b2[best_env0].detach().cpu().float(),
                }
                gl_best = policy.g_logits[best_g].detach().cpu().float()
                ls_raw_states.append(raw_best)
                ls_gls.append(gl_best)
                ls_folded_candidates.append(policy.fold_gate_into_state(raw_best, gl_best))

                val_res_list = evaluate_on_fixed_val_basket(
                    val_env, bf16_runner, lob_repr_all, v7_58d, ls_folded_candidates, val_basket, win_len, tick_sz, device
                )
                for c_idx, v_res in enumerate(val_res_list):
                    if float(v_res["val_score"]) > best_val_score:
                        best_val_score = float(v_res["val_score"])
                        best_val_ev = float(v_res["realistic_ev"])
                        best_val_state = {k: v.to(hof_state[k].dtype).clone() for k, v in ls_raw_states[c_idx].items()}
                        best_val_g_logit = ls_gls[c_idx].clone()
                        best_val_folded = {k: v.clone() for k, v in ls_folded_candidates[c_idx].items()}
                        hof_state = {k: v.clone() for k, v in best_val_state.items()}
                        hof_g_logit = best_val_g_logit.clone()
                        hof_updated = True
                        last_val_champ_it = it
                        ls_tag = f"alpha={ls_alphas[c_idx]:.2f}" if c_idx < len(ls_alphas) else "best_elite"
                        g_vals = (2.0 * torch.sigmoid(best_val_g_logit)).tolist()
                        log_msg(
                            f"  💎 [NEW L19 VAL CHAMPION it={it} ({ls_tag})] ValScore={best_val_score:+.2f} | ValEV={best_val_ev:+.4f}% "
                            f"[N:{v_res['norm_pct']:+.2f}% D:{v_res['dump_pct']:+.2f}% P:{v_res['pump_pct']:+.2f}% DD:{v_res['dd_pct']:.2f}%] | "
                            f"Gates(CV10={g_vals[3]:.2f}, CV6={g_vals[6]:.2f}, Macro={g_vals[4]:.2f}, Dist={g_vals[10]:.2f}, Fast5s={g_vals[11]:.2f})"
                        )

            if it % 20 == 0 and not hof_updated:
                hof_state = {k: v.clone() for k, v in best_val_state.items()}
                hof_g_logit = best_val_g_logit.clone()

            policy.evolve_gated_groups(
                group_scores,
                hof_state=hof_state,
                hof_g_logit=hof_g_logit,
                elite_frac=0.03,
                progress=min(0.95, 0.20 + 0.75 * progress),
            )

            g_dict = {name: round(float(2.0 * torch.sigmoid(best_val_g_logit[bi]).item()), 4) for bi, (name, _, _, _) in enumerate(SEMANTIC_BLOCKS)}
            rec = {
                "iteration": it,
                "hof_updated": hof_updated,
                "best_val_score": round(best_val_score, 2),
                "best_val_ev_pct": round(best_val_ev, 4),
                "batch_pnl_pct": round(b_pnl_pct, 4),
                "champ_oos_pnl_pct": round(c0_pnl_pct, 4),
                "dd_pct": round(b_dd_pct, 4),
                "block_gates": g_dict,
            }
            history.append(rec)
            summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            dt = time.time() - t0
            star = "★" if hof_updated else "·"
            log_msg(
                f"🎯 L19 [{it}/{iterations}]{star} ({b_prof_buds}/6+) | Batch: {b_pnl_pct:+.2f}% [N:{b_norm_pct:+.2f}% D:{b_dump_pct:+.2f}% P:{b_pump_pct:+.2f}%] | "
                f"ChampOOS: {c0_pnl_pct:+.2f}% | ValScore: {best_val_score:+.2f} (EV {best_val_ev:+.4f}%) | DD: {b_dd_pct:.2f}% | Hold: {b_hold:.0f}s ({dt:.1f}с)"
            )

            ckpt_payload = {
                "lesson": 19,
                "iteration": it,
                "release_name": f"L19_{it:04d}_cv_gated",
                "n_slots": 24,
                "allow_scratch_stop_asks": True,
                "use_cross_venue_v1": True,
                "score": best_val_score,
                "val_ev_pct": best_val_ev,
                "pnl_pct": b_pnl_pct,
                "pnl_usd": b_pnl_usd,
                "block_gates": g_dict,
                "backbone_state": backbone.state_dict(),
                "w1": best_val_folded["w1"],
                "b1": best_val_folded["b1"],
                "w_res1": best_val_folded["w_res1"],
                "b_res1": best_val_folded["b_res1"],
                "w_res2": best_val_folded["w_res2"],
                "w2": best_val_folded["w2"],
                "b2": best_val_folded["b2"],
            }
            torch.save(ckpt_payload, latest_ckpt)
            if hof_updated or it == iterations:
                torch.save(ckpt_payload, SAVE_DIR / f"lesson_19_champ_iter_{it:04d}.pt")

            if (it - last_val_champ_it) >= 75 and it >= 85:
                log_msg(f"🛑 [EARLY STOP L19] {it - last_val_champ_it} эпох без роста ValScore (лучший на it={last_val_champ_it}).")
                break

    log_msg(f"🏆 УРОК 19 ЗАВЕРШЕН! Финальный FULL_20 ValScore={best_val_score:+.2f} | ValEV={best_val_ev:+.4f}%")


if __name__ == "__main__":
    run_lesson_19(iterations=140, n_groups=2048)
