#!/usr/bin/env python3
"""
_tools/gpu_pbt_swarm_trainer.py
=============================================================================
Sequential Population-Based Training (PBT) v8.0 на RTX 5060 Ti:
- Популяция из 16 конкурирующих моделей (агентов-мутантов) прямо в VRAM GPU;
- Каждая особь обладает уникальным профилем чувствительности к риску и энтропии;
- Все 16 моделей параллельно исследуют рыночные среды (256 сред на GPU);
- На каждом фолде отбирается Топ-Чемпион с максимальным OOS Net PnL и выживаемостью;
- Чемпион размножается на всю популяцию с исследовательскими мутациями;
- Знания накапливаются непрерывно с 1-го по 20-й фолд без сброса опыта!
=============================================================================
"""

import os
import sys
import copy
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.vectorized_cuda_l2_env import (
    VectorizedCudaL2Env,
    collect_vectorized_cuda_rollouts,
    evaluate_blind_oos_cuda,
    evaluate_population_parallel_cuda,
)

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
PRETRAINED_BACKBONE_PATH = BASE_DIR / "models" / "prl" / "multitask_lob" / "pretrained_multitask_backbone.pt"
OUT_DIR = BASE_DIR / "models" / "prl" / "gpu_pbt_swarm_v8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_DIR / "data" / "logs" / "gpu_pbt_swarm_v8.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_msg(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


class Individual:
    def __init__(self, ind_id: int, model: MultiTaskLOBNet, lr: float = 1e-4, entropy_coef: float = 0.05, aux_risk_coef: float = 0.35, spread_boost: float = 0.02):
        self.ind_id = ind_id
        self.model = model
        self.lr = float(lr)
        self.entropy_coef = float(entropy_coef)
        self.aux_risk_coef = float(aux_risk_coef)
        self.spread_boost = float(spread_boost)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.last_oos_pnl = 0.0
        self.last_trades = 0

    def mutate(self, mutation_rate: float = 0.20):
        """Мутация гиперпараметров риска и стратегии."""
        if np.random.rand() < 0.5:
            self.entropy_coef = float(np.clip(self.entropy_coef * np.random.uniform(0.8, 1.25), 0.01, 0.12))
        if np.random.rand() < 0.5:
            self.aux_risk_coef = float(np.clip(self.aux_risk_coef * np.random.uniform(0.85, 1.2), 0.15, 0.60))
        if np.random.rand() < 0.5:
            self.spread_boost = float(np.clip(self.spread_boost * np.random.uniform(0.8, 1.25), 0.005, 0.04))

        # Легкая синаптическая мутация весов Actor (шум 0.5% для выхода из локальных минимумов)
        with torch.no_grad():
            for p in self.model.actor.parameters():
                if np.random.rand() < mutation_rate:
                    noise = torch.randn_like(p) * 0.005
                    p.add_(noise)


def train_pbt_ppo_step(ind: Individual, rollout: dict, device: torch.device, batch_size: int = 128) -> dict:
    ind.model.train()
    lob_s = rollout["lob"].to(device)
    flow_s = rollout["flow"].to(device)
    tgt_s = rollout["targets"].to(device)
    act_s = rollout["actions"].to(device)
    old_lp_s = rollout["old_logprobs"].to(device)
    adv_s = rollout["advantages"].to(device)
    ret_s = rollout["returns"].to(device)
    mask_s = rollout["masks"].to(device)

    n_seqs = lob_s.shape[0]
    adv_flat = adv_s.view(-1)
    adv_norm = (adv_s - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    indices = np.arange(n_seqs)
    np.random.shuffle(indices)

    tot_pol_loss = 0.0
    tot_val_loss = 0.0
    updates = 0

    for b_start in range(0, n_seqs, batch_size):
        b_idx = indices[b_start : b_start + batch_size]
        b_len = len(b_idx)
        if b_len < 4:
            continue

        b_lob, b_flow, b_tgt = lob_s[b_idx], flow_s[b_idx], tgt_s[b_idx]
        b_act, b_old_lp = act_s[b_idx], old_lp_s[b_idx]
        b_adv, b_ret, b_mask = adv_norm[b_idx], ret_s[b_idx], mask_s[b_idx]

        h0 = ind.model.init_hidden(b_len, device=device)
        out = ind.model.forward_sequence(b_lob, b_flow, hidden=h0, mask_seq=b_mask)

        dist = torch.distributions.Categorical(logits=out["logits"])
        new_lp = dist.log_prob(b_act)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_lp - b_old_lp)
        surr1 = ratio * b_adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(out["values"], b_ret)

        # Мультизадачный лосс рисков
        l_sw = F.binary_cross_entropy(out["p_sweep_12s"], b_tgt[:, :, 0])
        l_bl = F.binary_cross_entropy(out["p_bleed_60s"], b_tgt[:, :, 1])
        l_cr = F.binary_cross_entropy(out["p_crash_300s"], b_tgt[:, :, 2])
        l_ex = F.binary_cross_entropy(out["p_exit_30s"], b_tgt[:, :, 3])
        aux_loss = (l_sw + l_bl + l_cr + l_ex) / 4.0

        loss = policy_loss + 0.5 * value_loss + ind.aux_risk_coef * aux_loss - ind.entropy_coef * entropy

        ind.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ind.model.parameters(), max_norm=1.0)
        ind.optimizer.step()

        tot_pol_loss += float(policy_loss.item())
        tot_val_loss += float(value_loss.item())
        updates += 1

    return {"pol_loss": tot_pol_loss / max(1, updates), "val_loss": tot_val_loss / max(1, updates)}


def run_pbt_pipeline(
    pop_size: int = 16,
    total_folds: int = 20,
    train_hours: int = 120,
    test_hours: int = 24,
    embargo_sec: int = 300,
    epochs_per_fold: int = 2,
    resume: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg("=" * 80)
    log_msg(f"🚀 ЗАПУСК SEQUENTIAL PBT (ПОПУЛЯЦИЯ {pop_size} МОДЕЛЕЙ) НА {device}")
    log_msg(f"Вся популяция непрерывно шагает с 1 по {total_folds} фолд без сброса опыта!")
    log_msg("=" * 80)

    raw = np.load(CACHE_PATH)
    n_ticks = len(raw["bid_p"])
    log_msg("Генерация L2 тензоров стакана и потока сделок...")
    lob_tensors, flow_market = build_lob_10_and_flow_tensors(raw)

    targets_tensor = torch.from_numpy(
        np.stack([raw["y_sweep_12s"], raw["y_bleed_60s"], raw["y_crash_300s"], raw["y_exit_30s"]], axis=1)
    ).float()

    log_msg(f"Развертывание 256 параллельных сред биржи в VRAM GPU...")
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
        num_envs=256,
        device=device,
    )

    # Инициализация популяции из pop_size моделей
    population: List[Individual] = []
    latest_ckpt_path = OUT_DIR / "pbt_champion_latest.pt"
    start_fold = 0
    cumulative_champ_pnl = 0.0
    pbt_summary = []

    if resume and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        start_fold = ckpt.get("fold", 0)
        cumulative_champ_pnl = ckpt.get("cumulative_pnl", 0.0)
        champ_state = ckpt["champ_state_dict"]
        log_msg(f"♻️ ВОЗОБНОВЛЕНИЕ PBT с фолда {start_fold + 1}/{total_folds} (предыдущий PnL Чемпиона: ${cumulative_champ_pnl:+.2f})...")

        sum_path = OUT_DIR / "pbt_summary.json"
        if sum_path.exists():
            try:
                with open(sum_path, "r") as f:
                    pbt_summary = json.load(f)
            except Exception:
                pass

        for i in range(pop_size):
            m = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
            m.load_state_dict(champ_state)
            ent = float(0.04 + (i / max(1, pop_size - 1)) * 0.06)
            risk = float(0.20 + (i / max(1, pop_size - 1)) * 0.30)
            ind = Individual(ind_id=i + 1, model=m, entropy_coef=ent, aux_risk_coef=risk)
            if i > 0:
                ind.mutate(mutation_rate=0.25)
            population.append(ind)
    else:
        log_msg(f"Инициализация популяции из {pop_size} агентов на базе Shared Backbone...")
        ckpt = torch.load(PRETRAINED_BACKBONE_PATH, map_location=device)
        base_state = {k: v for k, v in ckpt.get("model_state_dict", ckpt).items() if not k.startswith(("actor.", "critic."))}

        for i in range(pop_size):
            m = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
            m.load_state_dict(base_state, strict=False)
            ent = float(0.04 + (i / max(1, pop_size - 1)) * 0.06)  # 0.04 .. 0.10
            risk = float(0.20 + (i / max(1, pop_size - 1)) * 0.30) # 0.20 .. 0.50
            ind = Individual(ind_id=i + 1, model=m, entropy_coef=ent, aux_risk_coef=risk)
            population.append(ind)

    train_len = train_hours * 3600
    test_len = test_hours * 3600
    step_stride = (n_ticks - train_len - test_len - embargo_sec) // max(1, total_folds)

    for fold in range(start_fold, total_folds):
        f_train_start = fold * step_stride
        f_train_end = f_train_start + train_len
        f_test_start = f_train_end + embargo_sec
        f_test_end = min(n_ticks, f_test_start + test_len)

        log_msg("=" * 80)
        log_msg(f"--- НАЧАЛО ЭПОХИ PBT: ФОЛД {fold + 1}/{total_folds} ---")
        log_msg(f"Train интервал: [{f_train_start}:{f_train_end}] ({train_hours}ч) | Causal Embargo: 300с")
        log_msg(f"Blind OOS Test: [{f_test_start}:{f_test_end}] ({test_hours}ч)")

        # 1. Параллельное обучение всех особей популяции на интервале фолда
        t_f0 = time.time()
        for ep in range(1, epochs_per_fold + 1):
            for ind in population:
                rollout = collect_vectorized_cuda_rollouts(
                    env=vec_env,
                    model=ind.model,
                    start_idx=f_train_start,
                    end_idx=f_train_end,
                    rollout_steps=256,
                    spread_boost=ind.spread_boost,
                    use_teacher=(fold == 0 and ep == 1), # Teacher warm-start только на самом первом шаге
                )
                train_pbt_ppo_step(ind, rollout, device, batch_size=128)

        # 2. Мгновенная параллельная OOS-аттестация всей популяции в едином GPU мега-батче
        log_msg(f"[Фолд {fold + 1}] Параллельная аттестация всех {pop_size} моделей в 1 мега-батче на GPU...")
        t_oos0 = time.time()
        all_metrics = evaluate_population_parallel_cuda(
            bids=vec_env.bids, asks=vec_env.asks, bids_q=vec_env.bids_q, asks_q=vec_env.asks_q,
            buy_vol=vec_env.buy_vol, sell_vol=vec_env.sell_vol,
            lob_tensors=vec_env.lob_tensors, flow_market=vec_env.flow_market, targets=vec_env.targets,
            models=[ind.model for ind in population],
            start_idx=f_test_start, end_idx=f_test_end, device=device,
            slices_per_model=8,
        )
        dt_oos = time.time() - t_oos0

        oos_scores = []
        for ind, metrics in zip(population, all_metrics):
            ind.last_oos_pnl = metrics["net_pnl"]
            ind.last_trades = metrics["trades"]
            oos_scores.append((metrics["net_pnl"], ind))

        # Сортировка по PnL (отбор чемпиона)
        oos_scores.sort(key=lambda x: x[0], reverse=True)
        best_pnl, champion = oos_scores[0]
        worst_pnl, _ = oos_scores[-1]
        cumulative_champ_pnl += best_pnl

        log_msg(
            f"👑 [ИТОГ ФОЛДА {fold + 1}] ЧЕМПИОН: Особь #{champion.ind_id} | "
            f"OOS PnL: ${best_pnl:+.2f} | Сделок: {champion.last_trades} | Худшая особь: ${worst_pnl:+.2f} | "
            f"Кумулятивно: ${cumulative_champ_pnl:+.2f} (OOS {dt_oos:.1f}с | фолд {time.time() - t_f0:.1f}с)"
        )

        pbt_summary.append({
            "fold": fold + 1,
            "champ_id": champion.ind_id,
            "champ_pnl": best_pnl,
            "champ_trades": champion.last_trades,
            "worst_pnl": worst_pnl,
            "cumulative_pnl": cumulative_champ_pnl,
        })

        # 3. ЭВОЛЮЦИЯ: Размножение Чемпиона на всю популяцию + Мутации
        log_msg(f"🧬 ЭВОЛЮЦИЯ: Клонирование Чемпиона #{champion.ind_id} на популяцию с мутациями...")
        champ_weights = copy.deepcopy(champion.model.state_dict())

        # Топ-5 сохраняют оригинальные веса, остальные 43 клонируют чемпиона и мутируют
        top_individuals = [ind for _, ind in oos_scores[:5]]
        for ind in population:
            if ind not in top_individuals:
                ind.model.load_state_dict(champ_weights)
                ind.entropy_coef = champion.entropy_coef
                ind.aux_risk_coef = champion.aux_risk_coef
                ind.spread_boost = champion.spread_boost
                ind.mutate(mutation_rate=0.25)

        # Сохранение текущего чемпиона и прогресса
        torch.save(
            {
                "fold": fold + 1,
                "champ_id": champion.ind_id,
                "champ_state_dict": champion.model.state_dict(),
                "cumulative_pnl": cumulative_champ_pnl,
            },
            OUT_DIR / "pbt_champion_latest.pt"
        )
        with open(OUT_DIR / "pbt_summary.json", "w") as f:
            json.dump(pbt_summary, f, indent=2)

    # Итоговое сохранение финального чемпиона после 20 фолдов
    final_champ_path = OUT_DIR / "pbt_champion_v8_final.pt"
    torch.save(champion.model.state_dict(), final_champ_path)
    with open(OUT_DIR / "pbt_summary.json", "w") as f:
        json.dump(pbt_summary, f, indent=2)

    log_msg("=" * 80)
    log_msg(f"🏆 PBT КУРС ЗАВЕРШЕН ПО ВСЕМ {total_folds} ФОЛДАМ!")
    log_msg(f"Финальный PnL Чемпиона: ${cumulative_champ_pnl:+.2f}")
    log_msg(f"Модель сохранена: {final_champ_path}")
    log_msg("=" * 80)


if __name__ == "__main__":
    run_pbt_pipeline(pop_size=48, total_folds=20, train_hours=120, test_hours=24)
