"""
SafeTrade PRL - УРОК 2: ЭКОНОМИКА СПРЕДА И ПРЕОДОЛЕНИЕ КОМИССИЙ (ВЫСОКОСКОРОСТНОЙ ОПТИМУМ: 8,192 МОДЕЛИ, H=256)
8,192 независимые модели параллельно в L2-кэше GPU RTX 5060 Ti (bfloat16, H=256).
250 итераций полного прогона через всю 600-часовую историю (2.19M тиков).
Старт: веса Чемпиона Урока 1 (lesson_1_champ_latest.pt).
Морковка:
  + Net PnL (в долларах с учетом комиссий SafeTrade 0.10% maker)
  +0.20 за вход при спреде >= 3 тиков (гарантия перекрытия комиссии)
  -0.50 за вход в мусорный спред < 2 тиков (где комиссия съедает депозит)
  -0.05 за зависание в позиции без котировки на продажу (моторный рефлекс)
"""

import os
import sys
import time
import json
import copy
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_cuda_l2_env import VectorizedCudaL2Env
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors

CACHE_PATH = Path("/home/restorator/crypto_bot/data/cache/rl_guard_lob_cache_600h.npz")
LESSON_1_CHAMP_PATH = Path("/home/restorator/crypto_bot/models/prl/lesson_1_motorics/lesson_1_champ_latest.pt")
OUT_DIR = Path("/home/restorator/crypto_bot/models/prl/lesson_2_spread")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("/home/restorator/crypto_bot/data/logs/lesson_2_spread_economics.log")


def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


class BatchedPopulationPolicy8K:
    def __init__(
        self,
        backbone: nn.Module,
        pop_size: int = 8192,
        in_dim: int = 260,
        h_dim: int = 256,
        n_actions: int = 7,
        device: torch.device = torch.device("cuda"),
    ):
        self.backbone = backbone
        self.pop_size = pop_size
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.n_actions = n_actions
        self.device = device
        self.dtype = torch.bfloat16

        self.w1 = torch.randn(pop_size, in_dim, h_dim, dtype=self.dtype, device=device) * 0.05
        self.b1 = torch.zeros(pop_size, 1, h_dim, dtype=self.dtype, device=device)
        self.w2 = torch.randn(pop_size, h_dim, n_actions, dtype=self.dtype, device=device) * 0.05
        self.b2 = torch.zeros(pop_size, 1, n_actions, dtype=self.dtype, device=device)

    def load_from_champion_1(self, champ_data: dict):
        w1_c = champ_data["w1"].to(device=self.device, dtype=self.dtype)
        b1_c = champ_data["b1"].to(device=self.device, dtype=self.dtype)
        w2_c = champ_data["w2"].to(device=self.device, dtype=self.dtype)
        b2_c = champ_data["b2"].to(device=self.device, dtype=self.dtype)

        self.w1.copy_(w1_c.unsqueeze(0).expand_as(self.w1))
        self.b1.copy_(b1_c.unsqueeze(0).expand_as(self.b1))
        self.w2.copy_(w2_c.unsqueeze(0).expand_as(self.w2))
        self.b2.copy_(b2_c.unsqueeze(0).expand_as(self.b2))

        # Начальная стратифицированная вариативность
        noise1 = (torch.randn_like(self.w1) * 0.02).to(self.dtype)
        noise2 = (torch.randn_like(self.w2) * 0.02).to(self.dtype)
        self.w1[1:].add_(noise1[1:])
        self.w2[1:].add_(noise2[1:])

    def forward_all(self, lob_b: torch.Tensor, flow_b: torch.Tensor, hidden: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lob_repr = self.backbone.encode_lob_batch(lob_b[:1]).expand(flow_b.size(0), -1)
        z, next_hidden = self.backbone.backbone.forward_step_fast(lob_repr, flow_b, hidden)

        p_sweep = torch.sigmoid(self.backbone.head_sweep_12s(z))
        p_bleed = torch.sigmoid(self.backbone.head_bleed_60s(z))
        p_crash = torch.sigmoid(self.backbone.head_crash_300s(z))
        p_exit = torch.sigmoid(self.backbone.head_exit_30s(z))
        z_aug = torch.cat([z, p_sweep, p_bleed, p_crash, p_exit], dim=-1).unsqueeze(1).to(self.dtype)

        h1 = torch.relu(torch.bmm(z_aug, self.w1) + self.b1)
        logits = (torch.bmm(h1, self.w2) + self.b2).squeeze(1).float()

        logits = logits.masked_fill(~masks, -1e9)

        gumbel_noise = torch.empty_like(logits).exponential_().log_().neg_()
        actions = (logits / 0.8 + gumbel_noise).argmax(dim=-1)
        return actions, next_hidden

    def clone_and_mutate(self, champ_idx: int, top_k: int = 128):
        champ_w1 = self.w1[champ_idx].clone()
        champ_b1 = self.b1[champ_idx].clone()
        champ_w2 = self.w2[champ_idx].clone()
        champ_b2 = self.b2[champ_idx].clone()

        self.w1.copy_(champ_w1.unsqueeze(0).expand_as(self.w1))
        self.b1.copy_(champ_b1.unsqueeze(0).expand_as(self.b1))
        self.w2.copy_(champ_w2.unsqueeze(0).expand_as(self.w2))
        self.b2.copy_(champ_b2.unsqueeze(0).expand_as(self.b2))

        n_fine = int(self.pop_size * 0.50)
        n_mid = int(self.pop_size * 0.35)

        sigmas = torch.zeros(self.pop_size, 1, 1, dtype=self.dtype, device=self.device)
        sigmas[top_k:n_fine] = 0.012
        sigmas[n_fine:n_fine + n_mid] = 0.040
        sigmas[n_fine + n_mid:] = 0.100

        noise1 = (torch.randn_like(self.w1) * sigmas).to(self.dtype)
        noise2 = (torch.randn_like(self.w2) * sigmas).to(self.dtype)

        self.w1.add_(noise1)
        self.w2.add_(noise2)


def run_spread_economics_simulation_8k(
    policy: BatchedPopulationPolicy8K,
    vec_env: VectorizedCudaL2Env,
    total_ticks: int,
    sim_steps: int = 3000,
    device: torch.device = torch.device("cuda"),
) -> Tuple[Dict[str, float], int]:
    pop_size = policy.pop_size
    step_stride = max(1, (total_ticks - 100) // sim_steps)

    vec_env.reset_interval(start_idx=0, end_idx=total_ticks, episode_len=total_ticks)

    completed_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    good_spread_entries = torch.zeros(pop_size, dtype=torch.float32, device=device)
    narrow_spread_penalties = torch.zeros(pop_size, dtype=torch.float32, device=device)
    inventory_lock_steps = torch.zeros(pop_size, dtype=torch.float32, device=device)

    prev_pos = vec_env.pos_qty[:pop_size].clone()
    h_states = policy.backbone.init_hidden(batch_size=pop_size, device=device)

    with torch.no_grad():
        for s in range(sim_steps):
            cur_t = s * step_stride
            if cur_t >= total_ticks - 1:
                break
            vec_env.t[:pop_size] = cur_t

            lob, flow, _ = vec_env.get_observations(cur_t_scalar=cur_t)
            masks = vec_env.get_action_masks(cur_t_scalar=cur_t)
            lob_b = lob
            flow_b = flow[:pop_size]
            mask_b = masks[:pop_size]

            b1 = vec_env.bids[cur_t, 0]
            a1 = vec_env.asks[cur_t, 0]
            spread_ticks = (a1 - b1) / max(1e-4, vec_env.tick_size)

            actions, h_states = policy.forward_all(lob_b, flow_b, h_states, mask_b)

            is_buy = (actions == 1) | (actions == 2)
            narrow_mask = is_buy & (spread_ticks < 2.0)
            narrow_spread_penalties += narrow_mask.float()

            wide_mask = is_buy & (spread_ticks >= 3.0)
            good_spread_entries += wide_mask.float()

            in_pos = vec_env.pos_qty[:pop_size] > 1e-6
            is_hold_in_pos = (actions == 0) & in_pos
            inventory_lock_steps += is_hold_in_pos.float()

            vec_env.step(actions)
            cur_pos = vec_env.pos_qty[:pop_size]

            just_closed = (prev_pos > 1e-6) & (cur_pos < 1e-6)
            completed_cycles += just_closed.float()
            prev_pos = cur_pos.clone()

    pnls = vec_env.prev_mtm[:pop_size]
    scores = (pnls * 100.0) + (good_spread_entries * 0.20) - (narrow_spread_penalties * 0.50) - (inventory_lock_steps * 0.05)
    best_idx = int(scores.argmax().item())
    worst_idx = int(scores.argmin().item())

    champ_dict = {
        "ind_id": best_idx + 1,
        "score": float(scores[best_idx].item()),
        "pnl": float(pnls[best_idx].item()),
        "cycles": int(completed_cycles[best_idx].item()),
        "good_spread_entries": int(good_spread_entries[best_idx].item()),
        "narrow_penalties": int(narrow_spread_penalties[best_idx].item()),
        "inv_lock_steps": int(inventory_lock_steps[best_idx].item()),
        "trades": int(vec_env.n_trades[best_idx].item()),
        "worst_score": float(scores[worst_idx].item()),
    }

    return champ_dict, best_idx


def run_lesson_2(pop_size: int = 8192, total_iterations: int = 250, sim_steps: int = 3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg("=" * 80)
    log_msg(f"🎓 СТАРТ УРОКА 2: ЭКОНОМИКА СПРЕДА И КОМИССИИ ({pop_size:,} МОДЕЛЕЙ, H=256, {total_iterations} ИТЕРАЦИЙ)")
    log_msg(f"Векторизованный мега-батч bfloat16 в L2-кэше {device} (охват всей 600-часовой истории)")
    log_msg("Морковка: Net PnL с комиссией 0.20%, бонус за спред >= 3 тиков, штраф за входы в узкий спред < 2 тиков")
    log_msg("=" * 80)

    log_msg(f"Загрузка кэша L2 датасета: {CACHE_PATH}...")
    raw = np.load(CACHE_PATH)
    n_ticks = len(raw["bid_p"])
    log_msg(f"Всего тиков в датасете: {n_ticks:,} (~{n_ticks / 3600:.1f} часов)")

    lob_tensors, flow_market = build_lob_10_and_flow_tensors(raw)
    targets_tensor = torch.from_numpy(
        np.stack([raw["y_sweep_12s"], raw["y_bleed_60s"], raw["y_crash_300s"], raw["y_exit_30s"]], axis=1)
    ).float()

    log_msg(f"Развертывание {pop_size:,} параллельных сред в VRAM GPU...")
    vec_env = VectorizedCudaL2Env(
        bids=torch.from_numpy(raw["bid_p"]).float(),
        asks=torch.from_numpy(raw["ask_p"]).float(),
        bids_q=torch.from_numpy(raw["bid_q"]).float(),
        asks_q=torch.from_numpy(raw["ask_q"]).float(),
        buy_vol=torch.from_numpy(raw["buy_vol"]).float(),
        sell_vol=torch.from_numpy(raw["sell_vol"]).float(),
        lob_tensors=torch.from_numpy(lob_tensors).float(),
        flow_market=torch.from_numpy(flow_market).float(),
        targets=targets_tensor,
        num_envs=pop_size,
        device=device,
    )

    log_msg(f"Загрузка базового энкодера и Чемпиона Урока 1: {LESSON_1_CHAMP_PATH}...")
    c_data = torch.load(LESSON_1_CHAMP_PATH, map_location=device)

    backbone_model = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone_model.load_state_dict(c_data["backbone_state"], strict=False)
    backbone_model.eval()

    policy = BatchedPopulationPolicy8K(backbone=backbone_model, pop_size=pop_size, h_dim=256, device=device)
    policy.load_from_champion_1(c_data)
    log_msg(f"🌱 Базовые веса Чемпиона Урока 1 (H=256) успешно клонированы на всю популяцию {pop_size:,} особей!")

    vram_used = torch.cuda.memory_allocated() / (1024**3)
    log_msg(f"💾 VRAM занято популяцией {pop_size:,} моделей (H=256): {vram_used:.2f} GB / 8.15 GB")

    history_summary = []

    for it in range(1, total_iterations + 1):
        t0 = time.time()

        champ_dict, champ_idx = run_spread_economics_simulation_8k(
            policy=policy,
            vec_env=vec_env,
            total_ticks=n_ticks,
            sim_steps=sim_steps,
            device=device,
        )

        elapsed = time.time() - t0
        log_msg(
            f"👑 [ИТОГ ИТЕРАЦИИ {it:03d}/{total_iterations}] ЧЕМПИОН #{champ_dict['ind_id']} | "
            f"Score: {champ_dict['score']:.1f} | Net PnL: ${champ_dict['pnl']:+.2f} | "
            f"Входов >=3t: {champ_dict['good_spread_entries']} | Узких входов: {champ_dict['narrow_penalties']} | "
            f"Циклов: {champ_dict['cycles']} | Худший Score: {champ_dict['worst_score']:.1f} ({elapsed:.1f}с | {sim_steps/elapsed:.0f} ш/с)"
        )

        champ_path = OUT_DIR / f"iteration_{it:03d}_champ.pt"
        torch.save(
            {
                "iteration": it,
                "champ_id": champ_dict["ind_id"],
                "score": champ_dict["score"],
                "pnl": champ_dict["pnl"],
                "metrics": champ_dict,
                "w1": policy.w1[champ_idx].cpu(),
                "b1": policy.b1[champ_idx].cpu(),
                "w2": policy.w2[champ_idx].cpu(),
                "b2": policy.b2[champ_idx].cpu(),
                "backbone_state": backbone_model.state_dict(),
            },
            champ_path,
        )

        torch.save(champ_path, OUT_DIR / "lesson_2_champ_latest.pt")

        history_summary.append({
            "iteration": it,
            "champion_id": champ_dict["ind_id"],
            "score": champ_dict["score"],
            "pnl": champ_dict["pnl"],
            "metrics": champ_dict,
            "time_sec": elapsed,
        })
        with open(OUT_DIR / "lesson_2_summary.json", "w") as f:
            json.dump(history_summary, f, indent=2)

        policy.clone_and_mutate(champ_idx=champ_idx, top_k=128)

    log_msg("=" * 80)
    log_msg(f"🎉 УРОК 2 УСПЕШНО ЗАВЕРШЕН ПО ВСЕМ {total_iterations} ИТЕРАЦИЯМ!")
    log_msg(f"Финальный Чемпион сохранен: {OUT_DIR / 'lesson_2_champ_latest.pt'}")
    log_msg("=" * 80)


if __name__ == "__main__":
    run_lesson_2(pop_size=8192, total_iterations=250, sim_steps=3000)
