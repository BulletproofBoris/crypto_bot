#!/usr/bin/env python3
"""
_tools/train_multitask_rl_execution.py
=============================================================================
Сквозное мультизадачное обучение MultiTask Spatial-Temporal Actor-Critic v8.0:
- Вход: тензор стакана L2 [N, 10, 4] + поток ордеров и состояние позиции [N, 16]
- Общий Shared Backbone: Spatial 1D CNN + Causal GRU (hidden=256, W=32)
- Multi-Task Loss:
    L_total = L_PPO(Actor) + 0.5 * L_MSE(Critic) + 0.3 * L_Auxiliary(Focal 4 Guards)
- Чистая среда FIFO без эвристических барьеров с физической маской SafeTrade
- Walk-Forward скользящие окна: 120ч Train / 24ч OOS Test по 609-часовому массиву
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
OUT_DIR = BASE_DIR / "models" / "prl" / "multitask_lob_v8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_DIR / "data" / "logs" / "multitask_ppo_v8.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class MultiTaskL2ExecutionEnv(SafeTradeL2ExecutionEnv):
    """
    Честная L2 среда SafeTrade:
    - Подает точный тензор стакана L2 [10, 4]
    - Подает объединенный вектор [16] (поток сделок + позиция агента)
    - Соблюдает физические правила биржи через action mask
    - Чистая награда Net PnL с вычетом комиссий мейкера (0.10% buy, 0.10% sell)
    """

    def __init__(self, data: dict, lob_tensors: np.ndarray, flow_market: np.ndarray,
                 targets: dict, start_idx: int, end_idx: int, episode_len: int = 7200, order_usd: float = 95.5):
        super().__init__(
            data=data,
            start_idx=start_idx,
            end_idx=end_idx,
            episode_len=episode_len,
            order_usd=order_usd,
            legacy_v2=True,
            enable_unfreeze=False,
            use_v52_obs=False,
            use_v70_obs=False,
        )
        self.lob_tensors = lob_tensors
        self.flow_market = flow_market
        self.targets = targets

    def get_action_mask(self) -> np.ndarray:
        """Физическая маска биржи SafeTrade"""
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True  # HOLD
        has_pos = self.pos_qty > 1e-8
        has_order = self.order_side is not None

        if not has_pos:
            if not has_order:
                mask[1] = True  # QUOTE_BID_BEST
                mask[2] = True  # QUOTE_BID_PASSIVE
            else:
                mask[5] = True  # CANCEL_ORDERS
        else:
            mask[3] = True  # QUOTE_ASK_BEST
            mask[4] = True  # QUOTE_ASK_PASSIVE
            mask[6] = True  # PANIC_TAKER_EXIT
            if has_order:
                mask[5] = True  # CANCEL_ORDERS

        return mask

    def step(self, action: int):
        prev_mtm = self.prev_mtm
        obs, _, done = super().step(action)
        # Чистая награда: строго дельта реального капитала в тиках (с учетом комиссий)
        # БЕЗ искусственных штрафов requote_penalty за бесплатную отмену ордеров SafeTrade!
        cur_mtm = self.prev_mtm
        b1 = max(self.bid_p[self.t, 0], self.tick_size)
        tick_val = self.order_usd * (self.tick_size / b1)
        reward = float((cur_mtm - prev_mtm) / max(1e-6, tick_val))
        return obs, reward, done

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает:
        1. lob: [10, 4]
        2. flow: [16] (рынок + позиция)
        3. target_risks: [4] (sweep, bleed, crash, exit) для вспомогательного лосса
        """
        t = self.t
        lob = self.lob_tensors[t]
        flow = self.flow_market[t].copy()

        # Заполняем каналы 8..15 состоянием позиции и ордеров
        has_pos = 1.0 if self.pos_qty > 1e-8 else 0.0
        b1 = max(self.bid_p[t, 0], self.tick_size)
        a1 = max(self.ask_p[t, 0], b1 + self.tick_size)
        unr_ticks = ((b1 - self.avg_entry) / self.tick_size) if has_pos else 0.0
        hold_ratio = min((t - self.entry_t) / 45.0, 2.0) if has_pos else 0.0
        has_bid = 1.0 if self.order_side == "BUY" else 0.0
        has_ask = 1.0 if self.order_side == "SELL" else 0.0
        if has_bid:
            dist = (b1 - self.order_price) / self.tick_size
        elif has_ask:
            dist = (self.order_price - a1) / self.tick_size
        else:
            dist = 0.0
        q_ratio = self.q_ahead / (self.q_ahead + self.order_qty + 1e-6) if self.order_side else 0.0

        flow[8] = has_pos
        flow[9] = np.clip(unr_ticks * 0.1, -3.0, 3.0)
        flow[10] = hold_ratio
        flow[11] = has_bid
        flow[12] = has_ask
        flow[13] = np.clip(dist * 0.2, -2.0, 2.0)
        flow[14] = q_ratio
        flow[15] = 0.0

        tgt = np.array([
            self.targets["sweep"][t],
            self.targets["bleed"][t],
            self.targets["crash"][t],
            self.targets["exit"][t],
        ], dtype=np.float32)

        return lob, flow, tgt


def collect_multitask_rollouts(
    env: MultiTaskL2ExecutionEnv,
    model: MultiTaskLOBNet,
    device: torch.device,
    rollout_steps: int = 4096,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> dict:
    model.eval()
    lob_list, flow_list, tgt_list = [], [], []
    act_list, logprob_list, rew_list, val_list, mask_list, done_list = [], [], [], [], [], []

    env.reset()
    start_t = env.t
    end_t = min(len(env.lob_tensors), start_t + rollout_steps + 1)
    chunk_len = end_t - start_t

    # Батчевый предрасчет Spatial LOB эмбеддингов на GPU (0.5с на 4096 шагов)
    with torch.no_grad():
        lob_chunk_t = torch.from_numpy(env.lob_tensors[start_t:end_t]).float().to(device)
        lob_repr_chunk = model.encode_lob_batch(lob_chunk_t)

    hidden = model.init_hidden(batch_size=1, device=device)
    lob, flow, tgt = env.get_state()

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
        _, reward, done = env.step(act_item)

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

    # Завершающая оценка ценности для GAE
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
        "lob": np.array(lob_list, dtype=np.float32),
        "flow": np.array(flow_list, dtype=np.float32),
        "targets": np.array(tgt_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.int64),
        "old_logprobs": np.array(logprob_list, dtype=np.float32),
        "advantages": adv_arr,
        "returns": ret_arr,
        "masks": np.array(mask_list, dtype=bool),
    }


def train_multitask_ppo_epoch(
    model: MultiTaskLOBNet,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    device: torch.device,
    clip_eps: float = 0.20,
    entropy_coef: float = 0.01,
    value_coef: float = 0.50,
    aux_coef: float = 0.25,
    seq_len: int = 32,
    batch_size: int = 32,
    ppo_iters: int = 4,
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

    # Нормализация advantages
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

            # PPO Clipped Objective
            ratio = torch.exp(new_lp - b_old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Loss (MSE)
            value_loss = F.mse_loss(values, b_ret)

            # Auxiliary Risk Loss (Focal)
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


def evaluate_oos_fold(
    env: MultiTaskL2ExecutionEnv,
    model: MultiTaskLOBNet,
    device: torch.device,
    max_steps: int = 86400,
) -> Dict[str, float]:
    """
    Честное OOS тестирование на 24-часовом слепом окне (86,400 тиков):
    - Батчевый предрасчет Spatial LOB эмбеддингов на GPU
    - Проверка физики каждого тика стакана SafeTrade
    - Расчет чистого Net PnL (env.prev_mtm) с вычетом комиссий мейкера
    """
    model.eval()
    env.reset(eval_mode=True)
    hidden = model.init_hidden(batch_size=1, device=device)

    start_t = env.t
    end_t = min(len(env.lob_tensors), start_t + max_steps + 1)
    window_len = end_t - start_t

    # Предрасчет LOB-эмбеддингов окна (86,400 тиков) за один батч на GPU
    with torch.no_grad():
        lob_w_t = torch.from_numpy(env.lob_tensors[start_t:end_t]).float().to(device)
        lob_reprs = model.encode_lob_batch(lob_w_t)

    actions_taken = []
    step_count = 0
    done = False
    order_age = 0

    with torch.no_grad():
        while not done and step_count < max_steps:
            local_idx = min(step_count, window_len - 1)
            has_pos = env.pos_qty > 1e-8
            has_order = env.order_side is not None

            # Опрос нейросети: когда нет ордера, либо при исполнении, либо каждые 4 тика
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
                act = 0  # HOLD: ордер стоит в очереди книги заявок биржи

            actions_taken.append(act)
            _, _, done = env.step(act)
            step_count += 1

    pnl_usd = float(env.prev_mtm)
    n_trades = int(env.n_trades)
    maker_bids = sum(1 for a in actions_taken if a in (1, 2))
    maker_pct = (maker_bids / max(1, len(actions_taken))) * 100.0

    return {
        "net_pnl": pnl_usd,
        "trades": n_trades,
        "maker_bid_pct": maker_pct,
        "steps": step_count,
    }


def run_walk_forward_pipeline(folds: int = 19):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🚀 Запуск Walk-Forward сквозного PPO обучения на {device} (RTX 5060 Ti)")
    print(f"   Файл данных: {CACHE_PATH.name} (609.2ч, 2,193,263 тика)")
    print("=" * 80, flush=True)

    raw = np.load(CACHE_PATH)
    n_ticks = len(raw["bid_p"])
    print(f"⚡ Подготовка тензоров стакана [N, 10, 4] и потока сделок [N, 16]...", flush=True)
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

    train_window = 432000  # 120 часов (по 3600 тиков/ч)
    test_window = 86400    # 24 часа OOS
    step_shift = 86400     # 24 часа сдвиг фолда

    # Проверяем, сколько фолдов влезает в датасет
    max_folds = (n_ticks - train_window) // step_shift
    folds_to_run = min(folds, max_folds)
    print(f"📊 Доступно полных фолдов: {max_folds} | Запускаем фолдов: {folds_to_run}", flush=True)

    # Загружаем модель с предобученным весом
    model = MultiTaskLOBNet().to(device)
    if PRETRAINED_BACKBONE_PATH.exists():
        ckpt = torch.load(PRETRAINED_BACKBONE_PATH, map_location=device)
        filtered_state = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith(("actor.", "critic."))}
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"⭐ Загружен предобученный Shared Backbone (Val Loss={ckpt.get('val_loss', 0.0):.5f})!", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    all_oos_results = []
    cumulative_pnl = 0.0

    for f_idx in range(folds_to_run):
        f_train_start = f_idx * step_shift
        f_train_end = f_train_start + train_window
        f_test_start = f_train_end
        f_test_end = min(n_ticks, f_test_start + test_window)

        print("-" * 80)
        print(f"🔄 ФОЛД {f_idx + 1:02d}/{folds_to_run:02d} | Train: [{f_train_start//3600}h..{f_train_end//3600}h] | OOS Test: [{f_test_start//3600}h..{f_test_end//3600}h]")
        print("-" * 80, flush=True)

        train_env = MultiTaskL2ExecutionEnv(
            data=full_data,
            lob_tensors=lob_tensors,
            flow_market=flow_market,
            targets=targets,
            start_idx=f_train_start,
            end_idx=f_train_end,
            episode_len=7200,
            order_usd=95.5,
        )

        test_env = MultiTaskL2ExecutionEnv(
            data=full_data,
            lob_tensors=lob_tensors,
            flow_market=flow_market,
            targets=targets,
            start_idx=f_test_start,
            end_idx=f_test_end,
            episode_len=test_window,
            order_usd=95.5,
        )

        # 4 итерации PPO на фолде с траекториями по 4096 шагов
        t_fold0 = time.time()
        for r_iter in range(4):
            rollout = collect_multitask_rollouts(train_env, model, device, rollout_steps=4096)
            stats = train_multitask_ppo_epoch(model, optimizer, rollout, device, seq_len=32, batch_size=32, ppo_iters=3)
            print(f"   [Iter {r_iter+1}/4] PolicyLoss: {stats['policy_loss']:.4f} | ValueLoss: {stats['value_loss']:.4f} | AuxLoss: {stats['aux_loss']:.4f}", flush=True)

        # Слепое OOS тестирование на следующих 24 часах
        oos = evaluate_oos_fold(test_env, model, device, max_steps=test_window)
        fold_time = time.time() - t_fold0
        cumulative_pnl += oos["net_pnl"]
        all_oos_results.append(oos)

        sign = "+" if oos["net_pnl"] >= 0 else ""
        cum_sign = "+" if cumulative_pnl >= 0 else ""
        print(
            f"  👉 OOS Результат ({fold_time:.1f}с): Net PnL: {sign}${oos['net_pnl']:.2f} | "
            f"Сделок: {oos['trades']} (Maker Bid: {oos['maker_bid_pct']:.1f}%) | "
            f"Кумулятивно: {cum_sign}${cumulative_pnl:.2f}",
            flush=True,
        )

        # Сохранение контрольной точки
        ckpt_path = OUT_DIR / f"multitask_ppo_fold_{f_idx + 1:02d}.pt"
        torch.save({
            "fold": f_idx + 1,
            "model_state_dict": model.state_dict(),
            "oos_metrics": oos,
            "cumulative_pnl": cumulative_pnl,
        }, ckpt_path)

    # Итоговый отчет
    print("=" * 80)
    print("🏁 ИТОГОВЫЙ WALK-FORWARD ОТЧЕТ СКВОЗНОЙ НЕЙРОСЕТИ v8.0:")
    total_trades = sum(r["trades"] for r in all_oos_results)
    profitable_folds = sum(1 for r in all_oos_results if r["net_pnl"] > 0)
    loss_folds = sum(1 for r in all_oos_results if r["net_pnl"] < 0)

    print(f"Кумулятивный чистый Net PnL (за вычетом комиссий SafeTrade): ${cumulative_pnl:.2f}")
    print(f"Прибыльных фолдов: {profitable_folds} из {len(all_oos_results)} ({profitable_folds/len(all_oos_results)*100:.1f}%) | Убыточных фолдов: {loss_folds}")
    print(f"Всего исполненных сделок: {total_trades}")
    print("=" * 80, flush=True)

    # Сохраняем финальную боевую модель и паспорт Walk-Forward
    final_path = OUT_DIR / "multitask_lob_v8_final.pt"
    summary_path = OUT_DIR / "walk_forward_summary_v8.json"
    torch.save({
        "model_state_dict": model.state_dict(),
        "all_oos_results": all_oos_results,
        "cumulative_pnl": cumulative_pnl,
    }, final_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "cumulative_net_pnl": cumulative_pnl,
            "total_trades": total_trades,
            "profitable_folds": profitable_folds,
            "total_folds": len(all_oos_results),
            "folds": all_oos_results,
        }, f, indent=2)

    print(f"⭐ Финальная боевая модель сохранена в {final_path.name}")
    print(f"📄 Паспорт Walk-Forward зафиксирован в {summary_path.name}", flush=True)


if __name__ == "__main__":
    run_walk_forward_pipeline(folds=20)
