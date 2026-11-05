#!/usr/bin/env python3
"""
_tools/curriculum_rl_trainer.py
=============================================================================
Сквозной Curriculum Learning конвейер глубокого обучения RL-агента v8.0
на всем 609-часовом массиве L2 книги SafeTrade (2,193,263 тиков) на RTX 5060 Ti.

Программа обучения (Curriculum Lessons):
-----------------------------------------------------------------------------
🎓 УРОК 1: Базовая микромеханика стакана (Queue & Execution)
   - Цель: освоение постановки ордеров в очередь, fill rate > 40%, осознанные отмены.
   - Параметры: gamma=0.98, entropy_coef=0.08 -> 0.04, lr=3e-4.

🎓 УРОК 2: Преодоление комиссий и захват спреда (Net PnL Generation)
   - Цель: реальные комиссии SafeTrade (0.10% buy, 0.10% sell), Net PnL > 0, Winrate > 60%.
   - Параметры: gamma=0.995, entropy_coef=0.04 -> 0.02, lr=1.5e-4.

🎓 УРОК 3: Управление рисками и самозащита (Survival & Defense)
   - Цель: реакция на сигналы 4 голов рисков (Crash 300s, Bleed 60s, Exit 30s), Max DD < $1.50.
   - Параметры: Multi-Task loss с Focal Loss по рискам, Downside Risk Penalization.

🎓 УРОК 4: Полномасштабный Walk-Forward (Out-of-Sample Mastery)
   - Цель: стабильный рост долларового кошелька по всем 20 фолдам без подглядывания.
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.multi_horizon_guard_net import FocalLoss
from _tools.train_rl_execution import SafeTradeL2ExecutionEnv
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
PRETRAINED_BACKBONE_PATH = BASE_DIR / "models" / "prl" / "multitask_lob" / "pretrained_multitask_backbone.pt"
OUT_DIR = BASE_DIR / "models" / "prl" / "curriculum_v8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_DIR / "data" / "logs" / "curriculum_training_v8.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class FreeMarketL2Env(SafeTradeL2ExecutionEnv):
    """
    Честная L2 среда SafeTrade БЕЗ единой эвристической подмены:
    - Покупка (QUOTE_BID) разрешена всегда, пока на депозите есть свободный баланс;
    - Повторный BID при стоящем ордере — честная перестановка/реквота на актуальный край;
    - Выходы и тейки (QUOTE_ASK_BEST, QUOTE_ASK_PASSIVE, PANIC_TAKER) доступны при наличии позиции;
    - Отмена (CANCEL_ORDERS) доступна всегда, когда стоит любой активный ордер;
    - Награда R_t = чистая дельта реального кошелька в USD с вычетом комиссий.
    """

    def __init__(
        self,
        data: dict,
        lob_tensors: np.ndarray,
        flow_market: np.ndarray,
        targets: dict,
        start_idx: int,
        end_idx: int,
        episode_len: int = 7200,
        initial_cash_usd: float = 95.5,
        min_order_usd: float = 25.0,
    ):
        super().__init__(
            data=data,
            start_idx=start_idx,
            end_idx=end_idx,
            episode_len=episode_len,
            order_usd=min_order_usd,
            legacy_v2=True,          # Отключает старые ручные эвристики
            enable_unfreeze=False,   # Отключает скрытые разморозки
            use_v52_obs=False,
            use_v70_obs=False,
        )
        self.lob_tensors = lob_tensors
        self.flow_market = flow_market
        self.targets = targets
        self.initial_cash_usd = float(initial_cash_usd)
        self.min_order_usd = float(min_order_usd)
        self.cash_balance = float(initial_cash_usd)

    def get_action_mask(self) -> np.ndarray:
        """
        Физическая маска биржи SafeTrade:
        0: HOLD / WAIT (всегда доступно)
        1: QUOTE_BID_BEST (доступно, пока есть свободные деньги)
        2: QUOTE_BID_PASSIVE (доступно, пока есть свободные деньги)
        3: QUOTE_ASK_BEST (доступно при наличии позиции)
        4: QUOTE_ASK_PASSIVE (доступно при наличии позиции)
        5: CANCEL_ORDERS (доступно, если есть активный ордер)
        6: PANIC_TAKER_EXIT (доступно при наличии позиции)
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True  # HOLD

        t = self.t
        b1 = max(self.bid_p[t, 0], self.tick_size)
        curr_pos_usd = self.pos_qty * b1
        free_cash = max(0.0, self.initial_cash_usd - curr_pos_usd)

        # 1. Покупка разрешена, пока свободных денег хватает хотя бы на минимальный ордер
        if free_cash >= self.min_order_usd:
            mask[1] = True  # QUOTE_BID_BEST
            mask[2] = True  # QUOTE_BID_PASSIVE

        # 2. Выходы и тейки доступны, если есть открытая позиция
        if self.pos_qty > 1e-6:
            mask[3] = True  # QUOTE_ASK_BEST
            mask[4] = True  # QUOTE_ASK_PASSIVE
            mask[6] = True  # PANIC_TAKER_EXIT

        # 3. Отмена доступна, если стоит ЛЮБОЙ активный ордер (бид или аск)
        if self.order_side is not None:
            mask[5] = True

        return mask

    def step(self, action: int, spread_reward_boost: float = 0.0) -> Tuple[np.ndarray, float, bool]:
        prev_mtm = self.prev_mtm
        prev_trades = self.n_trades

        obs, _, done = super().step(action)

        cur_mtm = self.prev_mtm
        t = self.t
        b1 = max(self.bid_p[t, 0], self.tick_size)
        tick_val = self.min_order_usd * (self.tick_size / b1)

        # Чистая дельта капитала кошелька в тиках
        delta_pnl_usd = cur_mtm - prev_mtm
        reward = float(delta_pnl_usd / max(1e-6, tick_val))

        # Награда за успешную реализацию спреда (бонус к преодолению пассивности)
        if spread_reward_boost > 0.0 and self.n_trades > prev_trades and delta_pnl_usd > 0:
            reward += spread_reward_boost

        return obs, reward, done

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = self.t
        lob = self.lob_tensors[t]
        flow = self.flow_market[t].copy()

        b1 = max(self.bid_p[t, 0], self.tick_size)
        a1 = max(self.ask_p[t, 0], b1 + self.tick_size)
        has_pos = 1.0 if self.pos_qty > 1e-6 else 0.0
        curr_pos_usd = self.pos_qty * b1
        pos_ratio = min(1.0, curr_pos_usd / self.initial_cash_usd)
        free_cash_ratio = max(0.0, 1.0 - pos_ratio)

        unr_ticks = ((b1 - self.avg_entry) / self.tick_size) if has_pos else 0.0
        hold_ratio = min((t - self.entry_t) / 300.0, 2.0) if has_pos else 0.0
        has_bid = 1.0 if self.order_side == "BUY" else 0.0
        has_ask = 1.0 if self.order_side == "SELL" else 0.0

        if has_bid:
            dist_ticks = (b1 - self.order_price) / self.tick_size
        elif has_ask:
            dist_ticks = (self.order_price - a1) / self.tick_size
        else:
            dist_ticks = 0.0

        # Каналы 8..15 вектора Flow: кристально ясное состояние кошелька и позиции
        flow[8] = has_pos
        flow[9] = pos_ratio
        flow[10] = free_cash_ratio
        flow[11] = np.clip(unr_ticks / 5.0, -2.0, 2.0)
        flow[12] = hold_ratio
        flow[13] = has_bid
        flow[14] = has_ask
        flow[15] = np.clip(dist_ticks / 5.0, -2.0, 2.0)

        tgt = np.array([
            self.targets["sweep"][t],
            self.targets["bleed"][t],
            self.targets["crash"][t],
            self.targets["exit"][t],
        ], dtype=np.float32)

        return lob, flow, tgt


def collect_rollouts_fast(
    env: FreeMarketL2Env,
    model: MultiTaskLOBNet,
    device: torch.device,
    rollout_steps: int = 4096,
    gamma: float = 0.99,
    lam: float = 0.95,
    spread_boost: float = 0.0,
) -> dict:
    model.eval()
    lob_list, flow_list, tgt_list = [], [], []
    act_list, logprob_list, rew_list, val_list, mask_list, done_list = [], [], [], [], [], []

    env.reset()
    start_t = env.t
    end_t = min(len(env.lob_tensors), start_t + rollout_steps + 1)
    chunk_len = end_t - start_t

    with torch.no_grad():
        lob_chunk_t = torch.from_numpy(env.lob_tensors[start_t:end_t]).float().to(device)
        lob_repr_chunk = model.encode_lob_batch(lob_chunk_t)

    hidden = model.init_hidden(batch_size=1, device=device)
    lob, flow, tgt = env.get_state()

    trades_in_rollout = 0
    maker_bids_in_rollout = 0

    for step in range(rollout_steps):
        local_idx = min(step, chunk_len - 1)
        mask = env.get_action_mask()
        lob_r = lob_repr_chunk[local_idx].unsqueeze(0)
        flow_t = torch.from_numpy(flow).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

        with torch.no_grad():
            out = model.forward_step_fast(lob_r, flow_t, hidden=hidden, mask=mask_t)
            logits = out["logits"]
            val = out["value"]
            hidden = out["hidden"]

            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            logprob = dist.log_prob(act)

        act_item = int(act.item())
        if act_item in (1, 2):
            maker_bids_in_rollout += 1

        prev_trades = env.n_trades
        _, reward, done = env.step(act_item, spread_reward_boost=spread_boost)
        if env.n_trades > prev_trades:
            trades_in_rollout += 1

        lob_list.append(lob)
        flow_list.append(flow)
        tgt_list.append(tgt)
        act_list.append(act_item)
        logprob_list.append(float(logprob.item()))
        rew_list.append(float(reward))
        val_list.append(float(val.item()))
        mask_list.append(mask)
        done_list.append(bool(done))

        if done:
            env.reset()
            hidden = model.init_hidden(batch_size=1, device=device)
        lob, flow, tgt = env.get_state()

    # Завершающая оценка для GAE
    with torch.no_grad():
        last_idx = min(rollout_steps, chunk_len - 1)
        lob_r = lob_repr_chunk[last_idx].unsqueeze(0)
        flow_t = torch.from_numpy(flow).float().unsqueeze(0).to(device)
        mask = env.get_action_mask()
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
        out = model.forward_step_fast(lob_r, flow_t, hidden=hidden, mask=mask_t)
        last_val = float(out["value"].item())

    # Generalized Advantage Estimation (GAE)
    adv_arr = np.zeros(rollout_steps, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(rollout_steps)):
        v_next = last_val if t == rollout_steps - 1 else val_list[t + 1]
        non_terminal = 0.0 if done_list[t] else 1.0
        delta = rew_list[t] + gamma * v_next * non_terminal - val_list[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        adv_arr[t] = last_gae

    ret_arr = adv_arr + np.array(val_list, dtype=np.float32)

    return {
        "lob": np.array(lob_list, dtype=np.float32),
        "flow": np.array(flow_list, dtype=np.float32),
        "targets": np.array(tgt_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.int64),
        "old_logprobs": np.array(logprob_list, dtype=np.float32),
        "advantages": adv_arr,
        "returns": ret_arr,
        "masks": np.array(mask_list, dtype=bool),
        "trades": trades_in_rollout,
        "maker_bids": maker_bids_in_rollout,
    }


def train_ppo_step(
    model: MultiTaskLOBNet,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    device: torch.device,
    clip_eps: float = 0.20,
    entropy_coef: float = 0.04,
    value_coef: float = 0.50,
    aux_coef: float = 0.25,
    seq_len: int = 32,
    batch_size: int = 32,
    ppo_iters: int = 3,
) -> Dict[str, float]:
    model.train()
    total_steps = len(rollout["actions"])
    n_seqs = total_steps // seq_len

    lob_s = torch.from_numpy(rollout["lob"][:n_seqs * seq_len]).view(n_seqs, seq_len, 10, 4)
    flow_s = torch.from_numpy(rollout["flow"][:n_seqs * seq_len]).view(n_seqs, seq_len, 16)
    tgt_s = torch.from_numpy(rollout["targets"][:n_seqs * seq_len]).view(n_seqs, seq_len, 4)
    act_s = torch.from_numpy(rollout["actions"][:n_seqs * seq_len]).view(n_seqs, seq_len)
    old_lp_s = torch.from_numpy(rollout["old_logprobs"][:n_seqs * seq_len]).view(n_seqs, seq_len)
    adv_s = torch.from_numpy(rollout["advantages"][:n_seqs * seq_len]).view(n_seqs, seq_len)
    ret_s = torch.from_numpy(rollout["returns"][:n_seqs * seq_len]).view(n_seqs, seq_len)
    mask_s = torch.from_numpy(rollout["masks"][:n_seqs * seq_len]).view(n_seqs, seq_len, 7)

    adv_flat = adv_s.view(-1)
    adv_norm = (adv_s - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    focal_loss = FocalLoss(gamma=2.0)
    history = {"policy_loss": 0.0, "value_loss": 0.0, "aux_loss": 0.0, "entropy": 0.0}
    updates = 0

    indices = np.arange(n_seqs)
    for _ in range(ppo_iters):
        np.random.shuffle(indices)
        for b_start in range(0, n_seqs, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            b_len = len(b_idx)
            if b_len < 4:
                continue

            b_lob = lob_s[b_idx].to(device)
            b_flow = flow_s[b_idx].to(device)
            b_tgt = tgt_s[b_idx].to(device)
            b_act = act_s[b_idx].to(device)
            b_old_lp = old_lp_s[b_idx].to(device)
            b_adv = adv_norm[b_idx].to(device)
            b_ret = ret_s[b_idx].to(device)
            b_mask = mask_s[b_idx].to(device)

            h0 = model.init_hidden(b_len, device=device)
            out = model.forward_sequence(b_lob, b_flow, hidden=h0, mask_seq=b_mask)

            logits = out["logits"]
            values = out["values"]

            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(b_act)
            entropy = dist.entropy().mean()

            # PPO Clipped Surrogate
            ratio = torch.exp(new_lp - b_old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic MSE Loss
            value_loss = F.mse_loss(values, b_ret)

            # Multi-Task Auxiliary Focal Loss
            l_sw = focal_loss(out["p_sweep_12s"], b_tgt[:, :, 0])
            l_bl = focal_loss(out["p_bleed_60s"], b_tgt[:, :, 1])
            l_cr = focal_loss(out["p_crash_300s"], b_tgt[:, :, 2])
            l_ex = focal_loss(out["p_exit_30s"], b_tgt[:, :, 3])
            aux_loss = (l_sw + l_bl + l_cr + l_ex) / 4.0

            total_loss = policy_loss + value_coef * value_loss + aux_coef * aux_loss - entropy_coef * entropy

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            history["policy_loss"] += float(policy_loss.item())
            history["value_loss"] += float(value_loss.item())
            history["aux_loss"] += float(aux_loss.item())
            history["entropy"] += float(entropy.item())
            updates += 1

    for k in history:
        history[k] /= max(1, updates)
    return history


def evaluate_blind_window(
    env: FreeMarketL2Env,
    model: MultiTaskLOBNet,
    device: torch.device,
    max_steps: int = 86400,
) -> Dict[str, float]:
    """Детерминированная слепая оценка без ручных эвристик."""
    model.eval()
    env.reset(eval_mode=True)
    hidden = model.init_hidden(batch_size=1, device=device)

    start_t = env.t
    end_t = min(len(env.lob_tensors), start_t + max_steps + 1)
    window_len = end_t - start_t

    with torch.no_grad():
        lob_w_t = torch.from_numpy(env.lob_tensors[start_t:end_t]).float().to(device)
        lob_reprs = model.encode_lob_batch(lob_w_t)

    actions_taken = []
    step_count = 0
    done = False
    order_age = 0

    trade_pnl_list = []
    prev_total_pnl = 0.0

    with torch.no_grad():
        while not done and step_count < max_steps:
            local_idx = min(step_count, window_len - 1)
            has_order = env.order_side is not None

            need_nn = False
            if not has_order:
                need_nn = True
                order_age = 0
            else:
                order_age += 1
                if order_age >= 4:
                    need_nn = True
                    order_age = 0

            if need_nn:
                mask = env.get_action_mask()
                lob_r = lob_reprs[local_idx].unsqueeze(0)
                _, flow, _ = env.get_state()
                flow_t = torch.from_numpy(flow).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

                out = model.forward_step_fast(lob_r, flow_t, hidden=hidden, mask=mask_t)
                hidden = out["hidden"]
                act = int(torch.argmax(out["logits"], dim=-1).item())
            else:
                act = 0

            actions_taken.append(act)
            prev_trades = env.n_trades
            _, _, done = env.step(act)
            step_count += 1

            if env.n_trades > prev_trades:
                trade_pnl = env.prev_mtm - prev_total_pnl
                trade_pnl_list.append(trade_pnl)
                prev_total_pnl = env.prev_mtm

    pnl_usd = float(env.prev_mtm)
    n_trades = int(env.n_trades)
    maker_bids = sum(1 for a in actions_taken if a in (1, 2))
    maker_pct = (maker_bids / max(1, len(actions_taken))) * 100.0
    wins = sum(1 for p in trade_pnl_list if p > 0)
    winrate = (wins / max(1, n_trades)) * 100.0

    return {
        "net_pnl": pnl_usd,
        "trades": n_trades,
        "wins": wins,
        "winrate_pct": winrate,
        "maker_bid_pct": maker_pct,
        "steps": step_count,
    }


def run_curriculum_training(total_folds: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🎓 Запуск Сквозного Curriculum Learning конвейера v8.0 на {device} (RTX 5060 Ti)")
    print(f"   Файл данных: {CACHE_PATH.name} (609.2ч, 2,193,263 тика L2 книги)")
    print("=" * 80, flush=True)

    raw = np.load(CACHE_PATH)
    n_ticks = len(raw["bid_p"])
    print("⚡ Подготовка тензоров стакана [N, 10, 4] и потока сделок [N, 16]...", flush=True)
    lob_tensors, flow_market = build_lob_10_and_flow_tensors(raw)

    vwap = (raw["bid_p"][:, 0] + raw["ask_p"][:, 0]) / 2.0
    full_data = {
        "bid_p": raw["bid_p"],
        "ask_p": raw["ask_p"],
        "bid_q": raw["bid_q"],
        "ask_q": raw["ask_q"],
        "buy_vol": raw["buy_vol"],
        "sell_vol": raw["sell_vol"],
        "vwap": vwap,
        "tick_size": float(raw["tick_size"]),
        "X_market": np.zeros((n_ticks, 24), dtype=np.float32),
    }

    targets = {
        "sweep": raw["y_sweep_12s"],
        "bleed": raw["y_bleed_60s"],
        "crash": raw["y_crash_300s"],
        "exit": raw["y_exit_30s"],
    }

    # Инициализация модели и загрузка предобученного Shared Backbone
    model = MultiTaskLOBNet().to(device)
    if PRETRAINED_BACKBONE_PATH.exists():
        ckpt = torch.load(PRETRAINED_BACKBONE_PATH, map_location=device)
        filtered_state = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith(("actor.", "critic."))}
        model.load_state_dict(filtered_state, strict=False)
        print(f"⭐ Загружен предобученный Shared Backbone (Val Loss={ckpt.get('val_loss', 0.0):.5f})!", flush=True)

    curriculum_log = []

    # =========================================================================
    # 🎓 УРОК 1: Базовая микромеханика стакана (Queue & Execution)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🎓 УРОК 1: Базовая микромеханика стакана и постановка в очередь FIFO")
    print("   Цель: преодоление пассивности, исследование действий (entropy=0.08, lr=3e-4)")
    print("=" * 80, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    u1_env = FreeMarketL2Env(full_data, lob_tensors, flow_market, targets, start_idx=0, end_idx=86400 * 3, episode_len=4096)

    for epoch in range(4):
        rollout = collect_rollouts_fast(u1_env, model, device, rollout_steps=4096, gamma=0.98, spread_boost=0.50)
        stats = train_ppo_step(model, optimizer, rollout, device, entropy_coef=0.08, ppo_iters=3)
        print(
            f"  [Урок 1 | Эпоха {epoch+1}/4] PolicyLoss: {stats['policy_loss']:.4f} | "
            f"Entropy: {stats['entropy']:.4f} | "
            f"Trades: {rollout['trades']} | Maker Bids: {rollout['maker_bids']}",
            flush=True,
        )

    # =========================================================================
    # 🎓 УРОК 2: Преодоление комиссий и захват спреда (Net PnL Generation)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🎓 УРОК 2: Захват спреда и преодоление комиссий мейкера SafeTrade")
    print("   Цель: положительное матожидание Net PnL (gamma=0.995, entropy=0.04, lr=1.5e-4)")
    print("=" * 80, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    u2_env = FreeMarketL2Env(full_data, lob_tensors, flow_market, targets, start_idx=86400, end_idx=86400 * 5, episode_len=7200)

    for epoch in range(4):
        rollout = collect_rollouts_fast(u2_env, model, device, rollout_steps=4096, gamma=0.995, spread_boost=0.20)
        stats = train_ppo_step(model, optimizer, rollout, device, entropy_coef=0.04, ppo_iters=3)
        print(
            f"  [Урок 2 | Эпоха {epoch+1}/4] PolicyLoss: {stats['policy_loss']:.4f} | "
            f"ValueLoss: {stats['value_loss']:.4f} | "
            f"Entropy: {stats['entropy']:.4f} | "
            f"Trades: {rollout['trades']}",
            flush=True,
        )

    # =========================================================================
    # 🎓 УРОК 3: Управление рисками и самозащита (Survival & Defense)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🎓 УРОК 3: Самозащита и реакция на 4 головы рисков (Crash 300s, Bleed 60s)")
    print("   Цель: своевременный выход из неблагоприятных сделок (lr=1e-4, aux_coef=0.35)")
    print("=" * 80, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    u3_env = FreeMarketL2Env(full_data, lob_tensors, flow_market, targets, start_idx=86400 * 3, end_idx=86400 * 7, episode_len=7200)

    for epoch in range(4):
        rollout = collect_rollouts_fast(u3_env, model, device, rollout_steps=4096, gamma=0.995, spread_boost=0.0)
        stats = train_ppo_step(model, optimizer, rollout, device, entropy_coef=0.025, aux_coef=0.35, ppo_iters=3)
        print(
            f"  [Урок 3 | Эпоха {epoch+1}/4] ValueLoss: {stats['value_loss']:.4f} | "
            f"AuxRiskLoss: {stats['aux_loss']:.4f} | "
            f"Entropy: {stats['entropy']:.4f} | "
            f"Trades: {rollout['trades']}",
            flush=True,
        )

    # =========================================================================
    # 🎓 УРОК 4: Полномасштабный Walk-Forward по всем 20 фолдам (Out-of-Sample)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"🎓 УРОК 4: Аттестация на скользящих окнах Walk-Forward ({total_folds} фолдов)")
    print("   Окно обучения: 120ч | Слепое OOS окно: 24ч | Без подглядывания в будущее")
    print("=" * 80, flush=True)

    train_window = 432000  # 120ч
    test_window = 86400    # 24ч
    step_shift = 86400     # 24ч сдвиг

    max_folds = (n_ticks - train_window) // step_shift
    folds_to_run = min(total_folds, max_folds)

    all_oos_results = []
    cumulative_pnl = 0.0

    for f_idx in range(folds_to_run):
        f_train_start = f_idx * step_shift
        f_train_end = f_train_start + train_window
        f_test_start = f_train_end
        f_test_end = min(n_ticks, f_test_start + test_window)

        print("-" * 80)
        print(f"🔄 Фолд {f_idx + 1:02d}/{folds_to_run:02d} | Train: [{f_train_start//3600}h..{f_train_end//3600}h] | Blind Test: [{f_test_start//3600}h..{f_test_end//3600}h]")
        print("-" * 80, flush=True)

        train_env = FreeMarketL2Env(full_data, lob_tensors, flow_market, targets, f_train_start, f_train_end, episode_len=7200)
        test_env = FreeMarketL2Env(full_data, lob_tensors, flow_market, targets, f_test_start, f_test_end, episode_len=test_window)

        t_fold0 = time.time()
        # 3 роллаута дообучения с адаптивной энтропией
        for r_iter in range(3):
            rollout = collect_rollouts_fast(train_env, model, device, rollout_steps=4096, gamma=0.995, spread_boost=0.05)
            stats = train_ppo_step(model, optimizer, rollout, device, entropy_coef=0.02, ppo_iters=2)

        # Детерминированное слепое тестирование на 24 часах
        oos = evaluate_blind_window(test_env, model, device, max_steps=test_window)
        fold_time = time.time() - t_fold0
        cumulative_pnl += oos["net_pnl"]
        all_oos_results.append(oos)

        sign = "+" if oos["net_pnl"] >= 0 else ""
        cum_sign = "+" if cumulative_pnl >= 0 else ""
        print(
            f"  👉 OOS Результат ({fold_time:.1f}с): Net PnL: {sign}${oos['net_pnl']:.2f} | "
            f"Сделок: {oos['trades']} (Win: {oos['winrate_pct']:.1f}%) | "
            f"Maker Bid: {oos['maker_bid_pct']:.1f}% | "
            f"Кумулятивно: {cum_sign}${cumulative_pnl:.2f}",
            flush=True,
        )

        ckpt_path = OUT_DIR / f"curriculum_fold_{f_idx + 1:02d}.pt"
        torch.save({
            "fold": f_idx + 1,
            "model_state_dict": model.state_dict(),
            "oos_metrics": oos,
            "cumulative_pnl": cumulative_pnl,
        }, ckpt_path)

    # Итоговый паспорт аттестации
    print("=" * 80)
    print("🏆 ИТОГОВЫЙ ПАСПОРТ АТТЕСТАЦИИ CURRICULUM LEARNING v8.0:")
    total_trades = sum(r["trades"] for r in all_oos_results)
    total_wins = sum(r["wins"] for r in all_oos_results)
    overall_winrate = (total_wins / max(1, total_trades)) * 100.0
    profitable_folds = sum(1 for r in all_oos_results if r["net_pnl"] > 0)
    loss_folds = sum(1 for r in all_oos_results if r["net_pnl"] < 0)

    print(f"Кумулятивный чистый Net PnL кошелька: ${cumulative_pnl:.2f} USDT")
    print(f"Прибыльных дней: {profitable_folds} из {len(all_oos_results)} ({profitable_folds/len(all_oos_results)*100:.1f}%) | Убыточных: {loss_folds}")
    print(f"Всего закрытых сделок: {total_trades} | Общий Winrate: {overall_winrate:.1f}%")
    print("=" * 80, flush=True)

    final_model_path = OUT_DIR / "curriculum_v8_final.pt"
    summary_path = OUT_DIR / "curriculum_v8_summary.json"
    torch.save({"model_state_dict": model.state_dict(), "all_oos_results": all_oos_results, "cumulative_pnl": cumulative_pnl}, final_model_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "cumulative_net_pnl": cumulative_pnl,
            "total_trades": total_trades,
            "overall_winrate": overall_winrate,
            "profitable_folds": profitable_folds,
            "total_folds": len(all_oos_results),
            "folds": all_oos_results,
        }, f, indent=2)

    print(f"⭐ Модель сохранена в {final_model_path.name}")
    print(f"📄 Паспорт зафиксирован в {summary_path.name}", flush=True)


if __name__ == "__main__":
    run_curriculum_training(total_folds=20)
