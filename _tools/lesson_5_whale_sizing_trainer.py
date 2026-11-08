#!/usr/bin/env python3
"""
_tools/lesson_5_whale_sizing_trainer.py
=============================================================================
Урок 5: Масштабирование капитала ($1,000 .. $20,000), Iceberg-транши и
контроль емкости стакана L2 (Whale Execution & Adaptive Sizing).

Архитектура:
- Входное пространство: 321D = 256D (LOB CNN+GRU) + 4D (Guard Heads) + 58D (v7 Walk-Forward)
  + 3D (Физические индикаторы емкости стакана: Budget/L1, Pos/L1_Ask, Pos/Tape_30s).
- Пространство действий: 10 действий (0..6 базовые + 7: BUY_BEST_WHALE,
  8: BUY_LAYERED_ICEBERG, 9: SELL_ASK_ICEBERG_SLICE).
- Тестирование каждой из 8,192 моделей в каждой итерации сразу на 3 корзинах
  бюджета: Средний ($1,000-$3,000), Крупный ($5,000-$10,000) и Китовый ($20,000).
- Оценка и логирование строго в относительных процентах (% Net PnL и % MaxDD).
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_whale_l2_env import VectorizedWhaleL2Env
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
LESSON_4_CHAMP_PATH = BASE_DIR / "models" / "prl" / "lesson_4_inventory" / "lesson_4_champ_latest.pt"
OUT_DIR = BASE_DIR / "models" / "prl" / "lesson_5_whale"
LOG_FILE = BASE_DIR / "data" / "logs" / "lesson_5_whale.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


class BatchedWhalePolicy8K:
    """
    8,192 параллельных голов политики [321 -> 256 -> 10] в bfloat16 на Tensor Cores.
    """

    def __init__(
        self,
        backbone: MultiTaskLOBNet,
        pop_size: int = 8192,
        in_dim: int = 321,
        h_dim: int = 256,
        n_actions: int = 10,
        device: torch.device = torch.device("cuda"),
    ):
        self.backbone = backbone
        self.pop_size = pop_size
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.n_actions = n_actions
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.w1 = torch.zeros(pop_size, in_dim, h_dim, dtype=self.dtype, device=device)
        self.b1 = torch.zeros(pop_size, 1, h_dim, dtype=self.dtype, device=device)
        self.w2 = torch.zeros(pop_size, h_dim, n_actions, dtype=self.dtype, device=device)
        self.b2 = torch.zeros(pop_size, 1, n_actions, dtype=self.dtype, device=device)

    def load_from_champion(self, champ_data: dict):
        w1_c = champ_data["w1"].to(device=self.device, dtype=self.dtype)
        b1_c = champ_data["b1"].to(device=self.device, dtype=self.dtype)
        w2_c = champ_data["w2"].to(device=self.device, dtype=self.dtype)
        b2_c = champ_data["b2"].to(device=self.device, dtype=self.dtype)

        src_in = w1_c.size(0)
        src_act = w2_c.size(-1)
        self.w1.zero_()
        self.w2.zero_()
        self.b2.zero_()

        self.w1[:, :src_in, :].copy_(w1_c.unsqueeze(0).expand(self.pop_size, src_in, self.h_dim))
        self.b1.copy_(b1_c.unsqueeze(0).expand_as(self.b1))
        self.w2[:, :, :src_act].copy_(w2_c.unsqueeze(0).expand(self.pop_size, self.h_dim, src_act))
        self.b2[:, :, :src_act].copy_(b2_c.unsqueeze(0).expand(self.pop_size, 1, src_act))

        # Если переносим веса из 7-действенной модели Урока 4, инициализируем новые действия (7, 8, 9)
        # от базовых действий (1: BuyBest, 2: BuyPass, 3: SellBest)
        if src_act == 7 and self.n_actions == 10:
            self.w2[:, :, 7].copy_(self.w2[:, :, 1])
            self.b2[:, :, 7].copy_(self.b2[:, :, 1])
            self.w2[:, :, 8].copy_(self.w2[:, :, 2])
            self.b2[:, :, 8].copy_(self.b2[:, :, 2])
            self.w2[:, :, 9].copy_(self.w2[:, :, 3])
            self.b2[:, :, 9].copy_(self.b2[:, :, 3])

        noise1 = (torch.randn_like(self.w1) * 0.015).to(self.dtype)
        noise2 = (torch.randn_like(self.w2) * 0.015).to(self.dtype)
        self.w1[1:].add_(noise1[1:])
        self.w2[1:].add_(noise2[1:])
    def forward_all(
        self,
        lob_b: torch.Tensor,
        flow_b: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        v7_row: torch.Tensor,
        liq_3d: torch.Tensor,
    ):
        B = flow_b.size(0)
        lob_repr = self.backbone.encode_lob_batch(lob_b[:1]).expand(B, -1)
        z, next_hidden = self.backbone.backbone.forward_step_fast(lob_repr, flow_b, hidden)

        p_sweep = torch.sigmoid(self.backbone.head_sweep_12s(z))
        p_bleed = torch.sigmoid(self.backbone.head_bleed_60s(z))
        p_crash = torch.sigmoid(self.backbone.head_crash_300s(z))
        p_exit = torch.sigmoid(self.backbone.head_exit_30s(z))
        v7_exp = v7_row.unsqueeze(0).expand(B, -1)
        z_aug = torch.cat([z, p_sweep, p_bleed, p_crash, p_exit, v7_exp, liq_3d], dim=-1).unsqueeze(1).to(self.dtype)

        h1 = torch.relu(torch.bmm(z_aug, self.w1) + self.b1)
        logits = (torch.bmm(h1, self.w2) + self.b2).squeeze(1).float()
        logits = logits.masked_fill(~masks, -1e9)
        actions = logits.argmax(dim=-1)
        return actions, next_hidden

    def clone_and_mutate(self, scores: torch.Tensor, top_k: int = 64, progress: float = 0.0):
        _, top_indices = torch.topk(scores, k=top_k)
        elite_w1 = self.w1[top_indices].clone()
        elite_b1 = self.b1[top_indices].clone()
        elite_w2 = self.w2[top_indices].clone()
        elite_b2 = self.b2[top_indices].clone()

        repeats = self.pop_size // top_k
        self.w1.copy_(elite_w1.repeat(repeats, 1, 1))
        self.b1.copy_(elite_b1.repeat(repeats, 1, 1))
        self.w2.copy_(elite_w2.repeat(repeats, 1, 1))
        self.b2.copy_(elite_b2.repeat(repeats, 1, 1))

        # Плавное затухание шага мутации (Annealing) от 1.0x в начале до 0.25x к 1000-й итерации
        decay = max(0.25, 1.0 - 0.75 * float(progress))
        sigmas = torch.zeros((self.pop_size, 1, 1), dtype=torch.float32, device=self.device)
        n_fine = int(self.pop_size * 0.45)
        n_mid = int(self.pop_size * 0.40)
        sigmas[top_k:n_fine] = 0.010 * decay
        sigmas[n_fine:n_fine + n_mid] = 0.025 * decay
        sigmas[n_fine + n_mid:] = 0.060 * decay

        self.w1.add_((torch.randn_like(self.w1) * sigmas).to(self.dtype))
        self.b1.add_((torch.randn_like(self.b1) * (sigmas * 0.5)).to(self.dtype))
        self.w2.add_((torch.randn_like(self.w2) * sigmas).to(self.dtype))
        self.b2.add_((torch.randn_like(self.b2) * (sigmas * 0.5)).to(self.dtype))


def run_whale_simulation_8k(
    policy: BatchedWhalePolicy8K,
    vec_env: VectorizedWhaleL2Env,
    v7_signals_tensor: torch.Tensor,
    total_ticks: int,
    stress_anchors: np.ndarray,
    sim_steps: int = 3000,
    device: torch.device = torch.device("cuda"),
) -> Tuple[Dict[str, float], int, torch.Tensor]:
    """
    Каждая итерация тестирует все 8,192 модели последовательно на 3 бюджетах:
      Окно 1 (1,000с): Бюджет из [$1,000, $3,000] (Средний капитал)
      Окно 2 (1,000с): Бюджет из [$5,000, $10,000] (Крупный капитал, стресс-участок)
      Окно 3 (1,000с): Бюджет $20,000 (Китовый капитал)
    """
    pop_size = policy.pop_size
    win_len = sim_steps // 3
    recent_floor = max(200, total_ticks - 96 * 3600)

    w1_start = int(np.random.randint(recent_floor, max(recent_floor + 1, total_ticks - win_len - 200)))
    if len(stress_anchors) > 0:
        c = int(np.random.choice(stress_anchors))
        w2_start = int(np.clip(c - np.random.randint(120, 500), 200, total_ticks - win_len - 200))
    else:
        w2_start = int(np.random.randint(200, total_ticks - win_len - 200))
    w3_start = int(np.random.randint(200, total_ticks - win_len - 200))

    tier1_b = float(np.random.choice([1000.0, 3000.0]))
    tier2_b = float(np.random.choice([5000.0, 10000.0]))
    tier3_b = 20000.0
    window_configs = [(w1_start, tier1_b), (w2_start, tier2_b), (w3_start, tier3_b)]
    vec_env.n_trades.zero_()
    vec_env.partial_fills.zero_()
    vec_env.iceberg_unloads.zero_()
    vec_env.slippage_usd.zero_()

    completed_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    profitable_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    bad_narrow_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    shield_cancels = torch.zeros(pop_size, dtype=torch.float32, device=device)
    toxic_buys = torch.zeros(pop_size, dtype=torch.float32, device=device)
    evasion_exits = torch.zeros(pop_size, dtype=torch.float32, device=device)
    stale_hold_steps = torch.zeros(pop_size, dtype=torch.float32, device=device)
    max_hold_steps = torch.zeros(pop_size, dtype=torch.float32, device=device)

    total_pnl_pct = torch.zeros(pop_size, dtype=torch.float32, device=device)
    total_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
    max_dd_pct = torch.zeros(pop_size, dtype=torch.float32, device=device)
    max_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)

    with torch.no_grad():
        for w_start, w_budget in window_configs:
            vec_env.set_uniform_budget(w_budget)
            h_states = policy.backbone.init_hidden(batch_size=pop_size, device=device)
            pos_age = torch.zeros(pop_size, dtype=torch.float32, device=device)
            prev_pos = vec_env.pos_qty[:pop_size].clone()
            prev_cash = vec_env.cash[:pop_size].clone()
            entry_cash_snap = vec_env.cash[:pop_size].clone()
            entered_narrow = torch.zeros(pop_size, dtype=torch.bool, device=device)
            peak_eq = torch.full((pop_size,), w_budget, dtype=torch.float32, device=device)
            win_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)

            for step_in_win in range(win_len):
                cur_t = int(w_start + step_in_win)
                lob, flow, tgt, liq_3d = vec_env.get_observations(cur_t_scalar=cur_t)
                masks = vec_env.get_action_masks(cur_t_scalar=cur_t)

                b1 = vec_env.bids[cur_t, 0]
                a1 = vec_env.asks[cur_t, 0]
                spread_ticks = torch.round((a1 - b1) / max(1e-6, vec_env.tick_size))

                tgt_row = tgt[0]
                is_danger_knife = bool((tgt_row[0] > 0.5) or (tgt_row[2] > 0.5))
                is_bleed_exit = bool((tgt_row[1] > 0.5) or (tgt_row[3] > 0.5))

                had_bid = vec_env.order_side[:pop_size] == 0
                in_pos = prev_pos > 1e-6
                v7_row = v7_signals_tensor[cur_t]

                actions, h_states = policy.forward_all(lob, flow[:pop_size], h_states, masks[:pop_size], v7_row, liq_3d[:pop_size])

                if is_danger_knife:
                    shield_cancels += (had_bid & (actions == 5)).float()

                vec_env.step(actions, cur_t_scalar=cur_t)
                cur_pos = vec_env.pos_qty[:pop_size]
                cur_cash = vec_env.cash[:pop_size]

                # Засчитываем упреждающий выход ТОЛЬКО при реальном сокращении/закрытии позиции!
                if is_bleed_exit:
                    evasion_exits += (in_pos & (cur_pos < (prev_pos - 1e-5))).float()

                still_in_pos = cur_pos > 1e-6
                pos_age = torch.where(still_in_pos, pos_age + 1.0, torch.zeros_like(pos_age))
                max_hold_steps = torch.maximum(max_hold_steps, pos_age)
                stale_hold_steps += (pos_age > 180.0).float()

                cur_eq = cur_cash + cur_pos * torch.clamp(b1, min=vec_env.tick_size)
                peak_eq = torch.maximum(peak_eq, cur_eq)
                cur_dd = torch.clamp(peak_eq - cur_eq, min=0.0)
                win_dd_usd = torch.maximum(win_dd_usd, cur_dd)

                just_opened = (prev_pos <= 1e-6) & still_in_pos
                if just_opened.any():
                    entry_cash_snap = torch.where(just_opened, prev_cash, entry_cash_snap)
                    entered_narrow = torch.where(just_opened, spread_ticks < 2.0, entered_narrow)
                    if is_danger_knife:
                        toxic_buys += just_opened.float()

                just_closed = (prev_pos > 1e-6) & (~still_in_pos)
                if just_closed.any():
                    completed_cycles += just_closed.float()
                    cycle_pnl = cur_cash - entry_cash_snap
                    profitable_cycles += (just_closed & (cycle_pnl > 0.0)).float()
                    bad_narrow_cycles += (just_closed & entered_narrow & (cycle_pnl <= 0.0)).float()

                prev_pos.copy_(cur_pos)
                prev_cash.copy_(cur_cash)

            # Принудительное закрытие остатка позиции в конце окна с реальным пробоем L1..L10
            end_t = int(w_start + win_len - 1)
            rem_mask = vec_env.pos_qty[:pop_size] > 1e-6
            if rem_mask.any():
                rem_q = vec_env.pos_qty[:pop_size][rem_mask].clone()
                proceeds = torch.zeros_like(rem_q)
                for lvl in range(10):
                    lp = torch.clamp(vec_env.bids[end_t, lvl], min=vec_env.tick_size)
                    lq = torch.clamp(vec_env.bids_q[end_t, lvl], min=1.0)
                    tq = torch.minimum(rem_q, lq if lvl < 9 else rem_q)
                    proceeds += tq * lp
                    rem_q = torch.clamp(rem_q - tq, min=0.0)
                vec_env.cash[:pop_size][rem_mask] += proceeds * (1.0 - vec_env.taker_fee)
                vec_env.pos_qty[:pop_size][rem_mask] = 0.0

            win_pnl_usd = vec_env.cash[:pop_size] - w_budget
            win_pnl_pct = 100.0 * win_pnl_usd / w_budget
            win_dd_pct = 100.0 * win_dd_usd / w_budget

            total_pnl_usd += win_pnl_usd
            total_pnl_pct += win_pnl_pct
            max_dd_usd = torch.maximum(max_dd_usd, win_dd_usd)
            max_dd_pct = torch.maximum(max_dd_pct, win_dd_pct)
    # Фитнес-функция Урока 5 без лазеек для накрутки очков (Anti-Reward-Hacking):
    # Главный драйвер: реальный относительный % прибыли и контроль % просадки.
    scores = torch.zeros(pop_size, dtype=torch.float32, device=device)
    scores += total_pnl_pct * 1000.0
    scores -= max_dd_pct * 1200.0
    scores -= stale_hold_steps * 0.35
    scores += profitable_cycles * 25.0
    scores -= (completed_cycles - profitable_cycles) * 35.0
    scores -= bad_narrow_cycles * 35.0
    scores += torch.clamp(shield_cancels, max=20.0) * 1.0
    scores += torch.clamp(evasion_exits, max=10.0) * 8.0
    scores += torch.clamp(vec_env.iceberg_unloads[:pop_size].float(), max=15.0) * 8.0
    scores -= toxic_buys * 25.0
    scores -= (completed_cycles < 2.0).float() * 250.0
    scores -= (total_pnl_pct <= 0.0).float() * 400.0
    scores = torch.nan_to_num(scores, nan=-10000.0, posinf=-10000.0, neginf=-10000.0)

    best_idx = int(torch.argmax(scores).item())
    worst_idx = int(torch.argmin(scores).item())

    champ_dict = {
        "best_idx": best_idx,
        "score": float(scores[best_idx].item()),
        "pnl_pct": float(total_pnl_pct[best_idx].item()),
        "pnl_usd": float(total_pnl_usd[best_idx].item()),
        "max_dd_pct": float(max_dd_pct[best_idx].item()),
        "max_dd_usd": float(max_dd_usd[best_idx].item()),
        "max_hold_s": int(max_hold_steps[best_idx].item()),
        "cycles": int(completed_cycles[best_idx].item()),
        "prof_cycles": int(profitable_cycles[best_idx].item()),
        "iceberg_unloads": int(vec_env.iceberg_unloads[best_idx].item()),
        "partial_fills": int(vec_env.partial_fills[best_idx].item()),
        "slippage_usd": float(vec_env.slippage_usd[best_idx].item()),
        "shield_cancels": int(shield_cancels[best_idx].item()),
        "evasion_exits": int(evasion_exits[best_idx].item()),
        "worst_score": float(scores[worst_idx].item()),
    }
    return champ_dict, best_idx, scores


def load_whale_env_from_cache(pop_size: int, device: torch.device):
    cache_mtime = CACHE_PATH.stat().st_mtime
    raw = np.load(CACHE_PATH)
    n_ticks = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.0001
    lob_tensors, flow_market = build_lob_10_and_flow_tensors(raw)
    targets_np = np.stack([raw["y_sweep_12s"], raw["y_bleed_60s"], raw["y_crash_300s"], raw["y_exit_30s"]], axis=1)
    targets_tensor = torch.from_numpy(targets_np).float()

    stress_mask = (targets_np[:, 1] > 0.5) | (targets_np[:, 2] > 0.5)
    stress_anchors = np.where(stress_mask)[0]
    stress_anchors = stress_anchors[(stress_anchors > 1600) & (stress_anchors < n_ticks - 1600)]
    if len(stress_anchors) > 5000:
        stress_anchors = stress_anchors[:: max(1, len(stress_anchors) // 5000)]

    v7_np = raw["v7_signals_58d"].astype(np.float32)
    v7_signals_tensor = torch.from_numpy(v7_np).to(device=device, dtype=torch.bfloat16)

    vec_env = VectorizedWhaleL2Env(
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
        tick_size=tick_sz,
        device=device,
    )
    log_msg(f"🐋 Загружен кэш Урока 5 (Whale 321D, 10 действий): {n_ticks:,} тиков ({n_ticks / 3600:.1f} ч)")
    return vec_env, v7_signals_tensor, n_ticks, stress_anchors, cache_mtime


def run_lesson_5(pop_size: int = 8192, total_iterations: int = 1000, sim_steps: int = 3000, resume: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg("=" * 80)
    log_msg(f"🐋 СТАРТ УРОКА 5 (WHALE EXECUTION & ADAPTIVE SIZING $1K..$20K, 321D, 10 ДЕЙСТВИЙ): {pop_size:,} МОДЕЛЕЙ, {total_iterations} ИТ")
    log_msg("=" * 80)

    vec_env, v7_signals_tensor, n_ticks, stress_anchors, last_cache_mtime = load_whale_env_from_cache(pop_size, device)

    history_summary = []
    start_it = 1
    latest_pt = OUT_DIR / "lesson_5_champ_latest.pt"
    summary_file = OUT_DIR / "lesson_5_summary.json"

    if resume and latest_pt.exists() and summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                history_summary = json.load(f)
            start_it = len(history_summary) + 1
            c_data = torch.load(latest_pt, map_location=device, weights_only=False)
            log_msg(f"🔄 ВОЗОБНОВЛЕНИЕ УРОКА 5 С ИТЕРАЦИИ {start_it}/{total_iterations}")
        except Exception as e:
            log_msg(f"⚠️ Ошибка чтения чекпоинта Урока 5 ({e}), старт от Чемпиона Урока 4.")
            start_it = 1
            history_summary = []
            c_data = torch.load(LESSON_4_CHAMP_PATH, map_location=device, weights_only=False)
    else:
        c_data = torch.load(LESSON_4_CHAMP_PATH, map_location=device, weights_only=False)

    backbone_model = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone_model.load_state_dict(c_data["backbone_state"], strict=False)
    backbone_model.eval()

    policy = BatchedWhalePolicy8K(backbone=backbone_model, pop_size=pop_size, in_dim=321, h_dim=256, n_actions=10, device=device)
    policy.load_from_champion(c_data)

    for it in range(start_it, total_iterations + 1):
        t0 = time.time()
        champ_dict, champ_idx, scores = run_whale_simulation_8k(
            policy=policy,
            vec_env=vec_env,
            v7_signals_tensor=v7_signals_tensor,
            total_ticks=n_ticks,
            stress_anchors=stress_anchors,
            sim_steps=sim_steps,
            device=device,
        )
        elapsed = time.time() - t0
        log_msg(
            f"🐋 [ИТОГ ИТЕРАЦИИ {it}/{total_iterations}] ЧЕМПИОН #{champ_idx} | Score: {champ_dict['score']:.1f} | "
            f"Net PnL: {champ_dict['pnl_pct']:+.2f}% (${champ_dict['pnl_usd']:+.2f}) | "
            f"MaxDD: {champ_dict['max_dd_pct']:.2f}% (${champ_dict['max_dd_usd']:.2f}) | "
            f"Циклов: {champ_dict['cycles']} (Приб: {champ_dict['prof_cycles']}) | "
            f"Iceberg/Part: {champ_dict['iceberg_unloads']}/{champ_dict['partial_fills']} ({elapsed:.1f}с)"
        )

        save_payload = {
            "iteration": it,
            "champ_idx": champ_idx,
            "in_dim": 321,
            "n_actions": 10,
            "metrics": champ_dict,
            "w1": policy.w1[champ_idx].float().cpu(),
            "b1": policy.b1[champ_idx].float().cpu(),
            "w2": policy.w2[champ_idx].float().cpu(),
            "b2": policy.b2[champ_idx].float().cpu(),
            "backbone_state": backbone_model.state_dict(),
        }
        torch.save(save_payload, latest_pt)
        if it % 50 == 0 or it == total_iterations:
            torch.save(save_payload, OUT_DIR / f"lesson_5_champ_iter_{it:04d}.pt")

        history_summary.append({"iteration": it, **champ_dict})
        with open(summary_file, "w") as f:
            json.dump(history_summary, f, indent=2)

        if it < total_iterations:
            policy.clone_and_mutate(scores, top_k=64, progress=it / float(total_iterations))

    log_msg("🏆 УРОК 5 (WHALE EXECUTION & ADAPTIVE SIZING) УСПЕШНО ЗАВЕРШЕН!")


if __name__ == "__main__":
    run_lesson_5(pop_size=8192, total_iterations=1000, sim_steps=3000)
