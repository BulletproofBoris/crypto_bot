#!/usr/bin/env python3
"""
_tools/swarm_curriculum_trainer.py
=============================================================================
Сквозной Роевой Curriculum Learning конвейер v8.0 на 8 воркерах CPU + RTX 5060 Ti
на всем 609-часовом датасете L2 микроструктуры SafeTrade (2,193,263 тиков).

Архитектура Роя:
-----------------------------------------------------------------------------
- 8 параллельных воркеров на CPU исследуют 8 разнородных рыночных эпох;
- Центральный мозг MultiTaskLOBNet на RTX 5060 Ti обучается на объединенном опыте;
- Полная каузальная чистота: Causal Window Masking исключает Lookahead Bias;
- Полная свобода действий: покупка разрешена, пока есть свободный баланс;
- Честная L2 FIFO симуляция с Maker 0.10% / Taker 0.20% комиссиями.
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.multi_horizon_guard_net import FocalLoss
from _tools.train_rl_execution import SafeTradeL2ExecutionEnv, teacher_heuristic_action
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.vectorized_cuda_l2_env import (
    VectorizedCudaL2Env,
    collect_vectorized_cuda_rollouts,
    evaluate_blind_oos_cuda,
)

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
PRETRAINED_BACKBONE_PATH = BASE_DIR / "models" / "prl" / "multitask_lob" / "pretrained_multitask_backbone.pt"
OUT_DIR = BASE_DIR / "models" / "prl" / "swarm_curriculum_v8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_DIR / "data" / "logs" / "swarm_curriculum_v8.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class SwarmL2Env(SafeTradeL2ExecutionEnv):
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
        episode_len: int = 4096,
        initial_cash_usd: float = 95.5,
        min_order_usd: float = 25.0,
    ):
        super().__init__(
            data=data,
            start_idx=start_idx,
            end_idx=end_idx,
            episode_len=episode_len,
            order_usd=min_order_usd,
            legacy_v2=True,
            enable_unfreeze=False,
            use_v52_obs=False,
            use_v70_obs=False,
        )
        self.lob_tensors = lob_tensors
        self.flow_market = flow_market
        self.targets = targets
        self.initial_cash_usd = float(initial_cash_usd)
        self.min_order_usd = float(min_order_usd)

    def get_action_mask(self) -> np.ndarray:
        """Биржевая маска: ограничения определяются ТОЛЬКО физикой биржи и свободным балансом."""
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True  # HOLD всегда доступен

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

        # 3. Отмена доступна, если стоит ЛЮБОЙ активный ордер
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

        delta_pnl_usd = cur_mtm - prev_mtm
        reward = float(delta_pnl_usd / max(1e-6, tick_val))

        # Бонус за фиксацию прибыльного спреда мейкером
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


def worker_rollout_task(args: Tuple) -> dict:
    """Один воркер роя: собирает траекторию в своем историческом сегменте."""
    (
        worker_id,
        full_data,
        lob_tensors,
        flow_market,
        targets,
        start_t,
        end_t,
        rollout_steps,
        gamma,
        lam,
        spread_boost,
        model_weights,
        device_str,
        use_teacher,
    ) = args

    device = torch.device(device_str)
    model = MultiTaskLOBNet()
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()

    env = SwarmL2Env(
        full_data,
        lob_tensors,
        flow_market,
        targets,
        start_idx=start_t,
        end_idx=end_t,
        episode_len=rollout_steps,
    )

    env.reset()
    seg_start = env.t
    seg_end = min(len(env.lob_tensors), seg_start + rollout_steps + 1)
    chunk_len = seg_end - seg_start

    with torch.no_grad():
        lob_chunk_t = torch.from_numpy(env.lob_tensors[seg_start:seg_end]).float().to(device)
        lob_repr_chunk = model.encode_lob_batch(lob_chunk_t)

    hidden = model.init_hidden(batch_size=1, device=device)
    lob, flow, tgt = env.get_state()

    lob_list, flow_list, tgt_list = [], [], []
    act_list, teacher_act_list, logprob_list, rew_list, val_list, mask_list, done_list = [], [], [], [], [], [], []
    trades_count = 0
    maker_bids_count = 0

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

        # Если включен режим Учителя (Урок 1), генерируем экспертную подсказку
        t_act_item = act_item
        if use_teacher:
            t_obs = env._get_obs()
            t_act_item = teacher_heuristic_action(t_obs, mask)
            # В фазе жесткой имитации передаем действие учителя
            if np.random.rand() < 0.60:
                act_item = t_act_item

        if act_item in (1, 2):
            maker_bids_count += 1

        prev_trades = env.n_trades
        _, reward, done = env.step(act_item, spread_reward_boost=spread_boost)
        if env.n_trades > prev_trades:
            trades_count += 1

        lob_list.append(lob)
        flow_list.append(flow)
        tgt_list.append(tgt)
        act_list.append(act_item)
        teacher_act_list.append(t_act_item)
        logprob_list.append(float(logprob.item()))
        rew_list.append(float(reward))
        val_list.append(float(val.item()))
        mask_list.append(mask)
        done_list.append(bool(done))

        if done:
            env.reset()
            hidden = model.init_hidden(batch_size=1, device=device)
        lob, flow, tgt = env.get_state()

    # Завершающая оценка ценности
    with torch.no_grad():
        last_idx = min(rollout_steps, chunk_len - 1)
        lob_r = lob_repr_chunk[last_idx].unsqueeze(0)
        flow_t = torch.from_numpy(flow).float().unsqueeze(0).to(device)
        mask = env.get_action_mask()
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
        out = model.forward_step_fast(lob_r, flow_t, hidden=hidden, mask=mask_t)
        last_val = float(out["value"].item())

    # GAE
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
        "worker_id": worker_id,
        "lob": np.array(lob_list, dtype=np.float32),
        "flow": np.array(flow_list, dtype=np.float32),
        "targets": np.array(tgt_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.int64),
        "teacher_actions": np.array(teacher_act_list, dtype=np.int64),
        "old_logprobs": np.array(logprob_list, dtype=np.float32),
        "advantages": adv_arr,
        "returns": ret_arr,
        "masks": np.array(mask_list, dtype=bool),
        "trades": trades_count,
        "maker_bids": maker_bids_count,
        "net_pnl": float(env.prev_mtm),
    }


def collect_swarm_rollouts(
    full_data: dict,
    lob_tensors: np.ndarray,
    flow_market: np.ndarray,
    targets: dict,
    model: MultiTaskLOBNet,
    worker_slices: List[Tuple[int, int]],
    device: torch.device,
    rollout_steps_per_worker: int = 2048,
    gamma: float = 0.995,
    lam: float = 0.95,
    spread_boost: float = 0.0,
    use_teacher: bool = False,
    num_workers: int = 8,
) -> dict:
    """Параллельный сбор опыта роем из 8 воркеров на CPU."""
    model_cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
    device_str = "cpu"  # Воркеры собирают на CPU потоках, чтобы освободить GPU для батч-тренировки

    tasks_args = []
    for w_id in range(num_workers):
        start_t, end_t = worker_slices[w_id % len(worker_slices)]
        args = (
            w_id,
            full_data,
            lob_tensors,
            flow_market,
            targets,
            start_t,
            end_t,
            rollout_steps_per_worker,
            gamma,
            lam,
            spread_boost,
            model_cpu_weights,
            device_str,
            use_teacher,
        )
        tasks_args.append(args)

    # Параллельный запуск на 8 ядрах CPU
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        worker_results = list(pool.map(worker_rollout_task, tasks_args))

    # Слияние опыта всех 8 воркеров в единый батч для GPU
    merged = {
        "lob": np.concatenate([r["lob"] for r in worker_results], axis=0),
        "flow": np.concatenate([r["flow"] for r in worker_results], axis=0),
        "targets": np.concatenate([r["targets"] for r in worker_results], axis=0),
        "actions": np.concatenate([r["actions"] for r in worker_results], axis=0),
        "teacher_actions": np.concatenate([r["teacher_actions"] for r in worker_results], axis=0),
        "old_logprobs": np.concatenate([r["old_logprobs"] for r in worker_results], axis=0),
        "advantages": np.concatenate([r["advantages"] for r in worker_results], axis=0),
        "returns": np.concatenate([r["returns"] for r in worker_results], axis=0),
        "masks": np.concatenate([r["masks"] for r in worker_results], axis=0),
        "total_trades": sum(r["trades"] for r in worker_results),
        "total_maker_bids": sum(r["maker_bids"] for r in worker_results),
        "avg_worker_pnl": float(np.mean([r["net_pnl"] for r in worker_results])),
    }
    return merged


def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(x).to(device)


def train_swarm_ppo_epoch(
    model: MultiTaskLOBNet,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    device: torch.device,
    clip_eps: float = 0.20,
    entropy_coef: float = 0.04,
    value_coef: float = 0.50,
    aux_coef: float = 0.25,
    seq_len: int = 32,
    batch_size: int = 64,  # Увеличенный батч для высокой загрузки тензорных ядер RTX 5060 Ti
    ppo_iters: int = 3,
) -> Dict[str, float]:
    model.train()
    lob_s = _to_tensor(rollout["lob"], device)
    flow_s = _to_tensor(rollout["flow"], device)
    tgt_s = _to_tensor(rollout["targets"], device)
    act_s = _to_tensor(rollout["actions"], device)
    old_lp_s = _to_tensor(rollout["old_logprobs"], device)
    adv_s = _to_tensor(rollout["advantages"], device)
    ret_s = _to_tensor(rollout["returns"], device)
    mask_s = _to_tensor(rollout["masks"], device)

    n_seqs = lob_s.shape[0]
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

            b_lob = lob_s[b_idx]
            b_flow = flow_s[b_idx]
            b_tgt = tgt_s[b_idx]
            b_act = act_s[b_idx]
            b_old_lp = old_lp_s[b_idx]
            b_adv = adv_norm[b_idx]
            b_ret = ret_s[b_idx]
            b_mask = mask_s[b_idx]

            h0 = model.init_hidden(b_len, device=device)
            out = model.forward_sequence(b_lob, b_flow, hidden=h0, mask_seq=b_mask)

            logits = out["logits"]
            values = out["values"]

            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(b_act)
            entropy = dist.entropy().mean()

            # PPO Clipped Objective
            ratio = torch.exp(new_lp - b_old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Loss
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


def train_swarm_distillation_epoch(
    model: MultiTaskLOBNet,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    device: torch.device,
    seq_len: int = 32,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Фаза имитации экспертных действий Учителя (Teacher Behavioral Cloning Warm-Start)."""
    model.train()
    lob_s = _to_tensor(rollout["lob"], device)
    flow_s = _to_tensor(rollout["flow"], device)
    t_act_s = _to_tensor(rollout["teacher_actions"], device)
    mask_s = _to_tensor(rollout["masks"], device)

    n_seqs = lob_s.shape[0]
    ce_loss_fn = nn.CrossEntropyLoss()
    total_ce_loss = 0.0
    accuracy_sum = 0.0
    updates = 0

    indices = np.arange(n_seqs)
    np.random.shuffle(indices)

    for b_start in range(0, n_seqs, batch_size):
        b_idx = indices[b_start : b_start + batch_size]
        b_len = len(b_idx)
        if b_len < 4:
            continue

        b_lob = lob_s[b_idx]
        b_flow = flow_s[b_idx]
        b_tact = t_act_s[b_idx]
        b_mask = mask_s[b_idx]

        h0 = model.init_hidden(b_len, device=device)
        out = model.forward_sequence(b_lob, b_flow, hidden=h0, mask_seq=b_mask)
        logits = out["logits"].view(-1, model.n_actions)
        targets = b_tact.view(-1)

        loss = ce_loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == targets).float().mean()

        total_ce_loss += float(loss.item())
        accuracy_sum += float(acc.item())
        updates += 1

    return {
        "distill_loss": total_ce_loss / max(1, updates),
        "teacher_match_acc": (accuracy_sum / max(1, updates)) * 100.0,
    }


def evaluate_blind_oos(
    env: SwarmL2Env,
    model: MultiTaskLOBNet,
    device: torch.device,
    max_steps: int = 86400,
) -> Dict[str, float]:
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


def log_msg(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_swarm_curriculum(
    total_folds: int = 20,
    num_workers: int = 8,
    train_hours: int = 120,
    test_hours: int = 24,
    embargo_sec: int = 300,
    distill_epochs: int = 3,
    ppo_epochs_l2: int = 3,
    ppo_epochs_l3: int = 3,
    steps_per_worker: int = 2048,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg(f"=== ЗАПУСК РОЕВОГО CURRICULUM ОБУЧЕНИЯ v8.0 НА {device} ===")
    log_msg(f"Воркеров CPU: {num_workers} | Батч роя: {num_workers * steps_per_worker} шагов/эпоху")
    log_msg(f"Загрузка кэша L2 датасета: {CACHE_PATH}...")

    raw = np.load(CACHE_PATH)
    n_ticks = len(raw["bid_p"])
    log_msg(f"Датасет загружен: {n_ticks:,} тиков (~{n_ticks / 3600:.1f} часов L2 стакана)")

    log_msg("Генерация тензоров LOB [10, 4] и Market Flow [16]...")
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
    total_ticks = n_ticks

    targets_tensor = torch.from_numpy(
        np.stack([raw["y_sweep_12s"], raw["y_bleed_60s"], raw["y_crash_300s"], raw["y_exit_30s"]], axis=1)
    ).float()

    log_msg("Создание векторизованной среды 256 параллельных L2 сред прямо в VRAM GPU (65,536 шагов)...")
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

    model = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)

    # Инициализация предобученным бэкбоуном (без actor/critic голов, которые принимают 260D)
    if PRETRAINED_BACKBONE_PATH.exists():
        log_msg(f"Загрузка весов Shared Backbone из {PRETRAINED_BACKBONE_PATH}...")
        ckpt = torch.load(PRETRAINED_BACKBONE_PATH, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        filtered_state = {k: v for k, v in state.items() if not k.startswith(("actor.", "critic."))}
        model.load_state_dict(filtered_state, strict=False)
        log_msg(f"Shared Backbone успешно загружен ({len(filtered_state)} тензоров)")

    train_len = train_hours * 3600
    test_len = test_hours * 3600
    step_stride = (total_ticks - train_len - test_len - embargo_sec) // max(1, total_folds)

    all_fold_results = []
    total_net_pnl = 0.0

    for fold in range(total_folds):
        f_train_start = fold * step_stride
        f_train_end = f_train_start + train_len
        f_test_start = f_train_end + embargo_sec
        f_test_end = min(total_ticks, f_test_start + test_len)

        log_msg("=" * 80)
        log_msg(f"--- НАЧАЛО ФОЛДА {fold + 1}/{total_folds} ---")
        log_msg(f"Train интервал: [{f_train_start}:{f_train_end}] ({train_hours} ч) | Causal Embargo: {embargo_sec}с")
        log_msg(f"Blind OOS Test интервал: [{f_test_start}:{f_test_end}] ({test_hours} ч)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        # -------------------------------------------------------------
        # УРОК 1: Teacher Distillation (Повторение экспертных тактик)
        # -------------------------------------------------------------
        log_msg(f"[Фолд {fold + 1}] >>> УРОК 1: Teacher Distillation (Warm-Start на GPU)...")
        for epoch in range(1, distill_epochs + 1):
            t0 = time.time()
            rollout = collect_vectorized_cuda_rollouts(
                env=vec_env,
                model=model,
                start_idx=f_train_start,
                end_idx=f_train_end,
                rollout_steps=256,
                spread_boost=0.0,
                use_teacher=True,
            )
            d_res = train_swarm_distillation_epoch(model, optimizer, rollout, device, batch_size=64)
            dt = time.time() - t0
            log_msg(
                f"  Урок 1 | Эпоха {epoch}/{distill_epochs} ({dt:.1f}с) | "
                f"Loss: {d_res['distill_loss']:.4f} | Точность имитации: {d_res['teacher_match_acc']:.1f}% | "
                f"Воркеры PnL: ${rollout['avg_worker_pnl']:+.2f} | Сделок: {rollout['total_trades']}"
            )

        # -------------------------------------------------------------
        # УРОК 2: Spread Capture & Net PnL (PPO с честными комиссиями)
        # -------------------------------------------------------------
        log_msg(f"[Фолд {fold + 1}] >>> УРОК 2: Spread Capture & Net PnL (PPO на GPU)...")
        for epoch in range(1, ppo_epochs_l2 + 1):
            t0 = time.time()
            rollout = collect_vectorized_cuda_rollouts(
                env=vec_env,
                model=model,
                start_idx=f_train_start,
                end_idx=f_train_end,
                rollout_steps=256,
                spread_boost=0.02,
                use_teacher=False,
            )
            p_res = train_swarm_ppo_epoch(
                model, optimizer, rollout, device,
                entropy_coef=0.08, value_coef=0.5, aux_coef=0.20,
                batch_size=128, ppo_iters=3
            )
            dt = time.time() - t0
            log_msg(
                f"  Урок 2 | Эпоха {epoch}/{ppo_epochs_l2} ({dt:.1f}с) | "
                f"PolLoss: {p_res['policy_loss']:.4f} | ValLoss: {p_res['value_loss']:.4f} | "
                f"Ent: {p_res['entropy']:.3f} | Воркеры PnL: ${rollout['avg_worker_pnl']:+.2f} | "
                f"Maker Bids: {rollout['total_maker_bids']}"
            )

        # -------------------------------------------------------------
        # УРОК 3: Survival & Risk Defense (Мультизадачный риск-лосс)
        # -------------------------------------------------------------
        log_msg(f"[Фолд {fold + 1}] >>> УРОК 3: Survival & Risk Defense (на GPU)...")
        for epoch in range(1, ppo_epochs_l3 + 1):
            t0 = time.time()
            rollout = collect_vectorized_cuda_rollouts(
                env=vec_env,
                model=model,
                start_idx=f_train_start,
                end_idx=f_train_end,
                rollout_steps=256,
                spread_boost=0.02,
                use_teacher=False,
            )
            p_res = train_swarm_ppo_epoch(
                model, optimizer, rollout, device,
                entropy_coef=0.05, value_coef=0.5, aux_coef=0.35,  # Повышен вес риска
                batch_size=128, ppo_iters=3
            )
            dt = time.time() - t0
            log_msg(
                f"  Урок 3 | Эпоха {epoch}/{ppo_epochs_l3} ({dt:.1f}с) | "
                f"AuxLoss (Risk): {p_res['aux_loss']:.4f} | PolLoss: {p_res['policy_loss']:.4f} | "
                f"Ent: {p_res['entropy']:.3f} | Воркеры PnL: ${rollout['avg_worker_pnl']:+.2f}"
            )

        # -------------------------------------------------------------
        # УРОК 4: Аттестация на слепом OOS (Blind Walk-Forward на GPU)
        # -------------------------------------------------------------
        log_msg(f"[Фолд {fold + 1}] >>> УРОК 4: Blind OOS Аттестация ({test_hours}ч на GPU)...")
        t_oos0 = time.time()
        oos_metrics = evaluate_blind_oos_cuda(
            bids=vec_env.bids,
            asks=vec_env.asks,
            bids_q=vec_env.bids_q,
            asks_q=vec_env.asks_q,
            buy_vol=vec_env.buy_vol,
            sell_vol=vec_env.sell_vol,
            lob_tensors=vec_env.lob_tensors,
            flow_market=vec_env.flow_market,
            targets=vec_env.targets,
            model=model,
            start_idx=f_test_start,
            end_idx=f_test_end,
            device=device,
        )
        dt_oos = time.time() - t_oos0
        total_net_pnl += oos_metrics["net_pnl"]

        fold_stat = {
            "fold": fold + 1,
            "train_hours": train_hours,
            "test_hours": test_hours,
            "net_pnl": oos_metrics["net_pnl"],
            "cumulative_pnl": total_net_pnl,
            "trades": oos_metrics["trades"],
            "winrate_pct": oos_metrics["winrate_pct"],
            "maker_bid_pct": oos_metrics["maker_bid_pct"],
        }
        all_fold_results.append(fold_stat)

        log_msg(
            f"=== [ИТОГ ФОЛДА {fold + 1} ({dt_oos:.1f}с)] OOS PnL: ${oos_metrics['net_pnl']:+.2f} | "
            f"Cumulative: ${total_net_pnl:+.2f} | Trades: {oos_metrics['trades']} | "
            f"Winrate: {oos_metrics['winrate_pct']:.1f}% | Maker Bid%: {oos_metrics['maker_bid_pct']:.1f}% ==="
        )

        # Промежуточный сейв весов
        torch.save(
            {
                "fold": fold + 1,
                "model_state_dict": model.state_dict(),
                "cumulative_pnl": total_net_pnl,
                "fold_results": all_fold_results,
            },
            OUT_DIR / "curriculum_v8_latest.pt"
        )

    # Итоговое сохранение обученной модели
    final_model_path = OUT_DIR / "curriculum_v8_final.pt"
    summary_path = OUT_DIR / "curriculum_v8_summary.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "total_net_pnl": total_net_pnl,
            "fold_results": all_fold_results,
        },
        final_model_path
    )

    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_folds": total_folds,
                "total_net_pnl": total_net_pnl,
                "folds": all_fold_results,
            },
            f,
            indent=2,
        )

    log_msg("=" * 80)
    log_msg(f"КУРРИКУЛУМ ЗАВЕРШЕН ПО ВСЕМ {total_folds} ФОЛДАМ!")
    log_msg(f"Итоговый PnL: ${total_net_pnl:+.2f}")
    log_msg(f"Финальная модель сохранена: {final_model_path}")
    log_msg(f"Сводный отчет: {summary_path}")
    log_msg("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Swarm Curriculum Trainer v8.0")
    parser.add_argument("--folds", type=int, default=20, help="Количество фолдов Walk-Forward (default: 20)")
    parser.add_argument("--workers", type=int, default=8, help="Количество воркеров CPU (default: 8)")
    parser.add_argument("--train-hours", type=int, default=120, help="Часов обучения на фолд (default: 120)")
    parser.add_argument("--test-hours", type=int, default=24, help="Часов слепого OOS теста на фолд (default: 24)")
    args = parser.parse_args()

    run_swarm_curriculum(
        total_folds=args.folds,
        num_workers=args.workers,
        train_hours=args.train_hours,
        test_hours=args.test_hours,
    )

