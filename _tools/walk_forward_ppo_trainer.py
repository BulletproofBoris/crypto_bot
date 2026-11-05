#!/usr/bin/env python3
"""
_tools/walk_forward_ppo_trainer.py
=============================================================================
Полноценный Walk-Forward конвейер глубокого обучения Causal Recurrent PPO
на всем 609-часовом датасете L2 микроструктуры (2,193,263 тиков) на RTX 5060 Ti:

1. Полный отказ от эвристических границ: агент обучается максимизировать Net Reward
   (чистый PnL с обязательным вычетом 0.20% round-trip комиссий SafeTrade).
2. Скользящие окна Walk-Forward:
   - Длина обучающего окна: 120 часов (432,000 тиков)
   - Длина слепого OOS окна: 24 часа (86,400 тиков)
   - Шаг сдвига окна: 24 часа (86,400 тиков)
   - Всего 19 последовательных Walk-Forward фолдов без единого lookahead bias!
3. Память: Causal GRU (hidden_dim=128, seq_len=32), PPO Clipped Objective + GAE.
4. Выходы Guard-сетей (12s sweep, 60s bleed, 300s crash, 30s exit) подаются прямо
   во входной вектор признаков агента (53D obs) без жестких внешних ограничений.
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.multi_horizon_guard_net import MultiHorizonGuardPredictor, extract_guard_features
from _tools.recurrent_maskable_ppo import RecurrentMaskableActorCritic
from _tools.train_rl_execution import SafeTradeL2ExecutionEnv

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
OUT_DIR = BASE_DIR / "models" / "prl" / "walk_forward_ppo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_DIR / "data" / "logs" / "walk_forward_ppo_train.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class PureNetRewardExecutionEnv(SafeTradeL2ExecutionEnv):
    """
    Среда L2 исполнения БЕЗ искусственных эвристических фильтров.
    Ограничения определяются ТОЛЬКО физикой биржи:
    - Нельзя продать, если нет позиции;
    - Нельзя купить, если уже открыта позиция или лимитный бид;
    - Награда R_t = чистая дельта капитала с вычетом комиссий мейкера (0.10% за вход, 0.10% за выход).
    """

    def __init__(self, data: dict, start_idx: int, end_idx: int, episode_len: int = 7200, order_usd: float = 95.5):
        super().__init__(
            data=data,
            start_idx=start_idx,
            end_idx=end_idx,
            episode_len=episode_len,
            order_usd=order_usd,
            legacy_v2=False,
            use_v52_obs=False,
            use_v70_obs=False,
        )

    def get_action_mask(self) -> np.ndarray:
        """
        Физическая маска биржевых правил (никаких дискретных запретов по спреду или отскоку!):
        0: HOLD / WAIT (всегда доступно)
        1: QUOTE_BID_BEST (доступно, если нет позиции и нет активного бида)
        2: QUOTE_BID_PASSIVE (доступно, если нет позиции и нет активного бида)
        3: QUOTE_ASK_BEST (доступно только если ЕСТЬ позиция)
        4: QUOTE_ASK_PASSIVE (доступно только если ЕСТЬ позиция)
        5: CANCEL_ORDERS (доступно только если ЕСТЬ активная заявка)
        6: PANIC_TAKER_EXIT (доступно только если ЕСТЬ позиция)
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True  # HOLD всегда доступен
        has_pos = self.pos_qty > 1e-8
        has_order = self.order_side is not None

        if not has_pos:
            if not has_order:
                # Входы полностью открыты: PPO САМ решает, когда и по какой цене входить!
                mask[1] = True
                mask[2] = True
            else:
                mask[5] = True  # Можно отменить активный бид
        else:
            # Выходы из позиции
            mask[3] = True
            mask[4] = True
            mask[6] = True
            if has_order:
                mask[5] = True

        return mask

    def step(self, action: int):
        obs, reward, done = super().step(action)
        # Награда формируется строго из чистого Net PnL (с учетом 0.10% Maker / 0.20% Taker комиссий)
        # Дополнительно штрафуем за длительное неэффективное удержание позиции против тренда
        if self.pos_qty > 1e-8:
            unr_ticks = (max(self.bid_p[self.t, 0], self.tick_size) - self.avg_entry) / self.tick_size
            if unr_ticks < -3.0:
                reward -= 0.05 * abs(unr_ticks)  # Штраф за просадку
        return obs, reward, done


def collect_recurrent_rollouts(
    env: PureNetRewardExecutionEnv,
    model: RecurrentMaskableActorCritic,
    device: torch.device,
    rollout_steps: int = 4096,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """
    Сбор траекторий в L2 FIFO среде с сохранением скрытого состояния Causal GRU.
    Вычисляет честный Net Advantage через GAE (Generalized Advantage Estimation).
    """
    model.eval()
    obs_list, act_list, logprob_list, rew_list, val_list, mask_list, done_list = [], [], [], [], [], [], []

    obs = env.reset()
    hidden = model.init_hidden(batch_size=1, device=device)

    for step in range(rollout_steps):
        mask = env.get_action_mask()
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

        with torch.no_grad():
            logits, val, next_hidden = model.forward_step(obs_t, hidden, mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            logprob = dist.log_prob(act)

        act_item = int(act.item())
        next_obs, reward, done = env.step(act_item)

        obs_list.append(obs)
        act_list.append(act_item)
        logprob_list.append(float(logprob.item()))
        rew_list.append(float(reward))
        val_list.append(float(val.item()))
        mask_list.append(mask)
        done_list.append(bool(done))

        obs = next_obs
        hidden = next_hidden
        if done:
            obs = env.reset()
            hidden = model.init_hidden(batch_size=1, device=device)

    # Завершающая оценка ценности для GAE
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        mask = env.get_action_mask()
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
        _, next_val, _ = model.forward_step(obs_t, hidden, mask_t)
        last_val = float(next_val.item())

    # Расчет GAE Advantages и Discounted Returns
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
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.int64),
        "old_logprobs": np.array(logprob_list, dtype=np.float32),
        "advantages": adv_arr,
        "returns": ret_arr,
        "masks": np.array(mask_list, dtype=bool),
    }


def train_ppo_epoch(
    model: RecurrentMaskableActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_data: dict,
    device: torch.device,
    clip_eps: float = 0.20,
    entropy_coef: float = 0.015,
    value_coef: float = 0.50,
    seq_len: int = 32,
    batch_size: int = 32,
    ppo_iters: int = 4,
):
    """
    PPO Clipped Objective с последовательностями траекторий для Causal GRU на GPU.
    """
    model.train()
    total_steps = len(rollout_data["actions"])
    adv = rollout_data["advantages"]
    # Нормализация advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Делим rollout на чанки длины seq_len для рекуррентного обучения
    num_chunks = total_steps // seq_len
    chunk_indices = np.arange(num_chunks)

    total_loss_accum = 0.0

    for _ in range(ppo_iters):
        np.random.shuffle(chunk_indices)
        for i in range(0, num_chunks, batch_size):
            b_chunks = chunk_indices[i : i + batch_size]
            b_size = len(b_chunks)
            if b_size == 0:
                continue

            obs_chunk = np.zeros((b_size, seq_len, model.obs_dim), dtype=np.float32)
            act_chunk = np.zeros((b_size, seq_len), dtype=np.int64)
            mask_chunk = np.zeros((b_size, seq_len, model.n_actions), dtype=bool)
            old_lp_chunk = np.zeros((b_size, seq_len), dtype=np.float32)
            adv_chunk = np.zeros((b_size, seq_len), dtype=np.float32)
            ret_chunk = np.zeros((b_size, seq_len), dtype=np.float32)

            for b_idx, c_idx in enumerate(b_chunks):
                start = c_idx * seq_len
                end = start + seq_len
                obs_chunk[b_idx] = rollout_data["obs"][start:end]
                act_chunk[b_idx] = rollout_data["actions"][start:end]
                mask_chunk[b_idx] = rollout_data["masks"][start:end]
                old_lp_chunk[b_idx] = rollout_data["old_logprobs"][start:end]
                adv_chunk[b_idx] = adv[start:end]
                ret_chunk[b_idx] = rollout_data["returns"][start:end]

            obs_t = torch.from_numpy(obs_chunk).to(device)
            act_t = torch.from_numpy(act_chunk).to(device)
            mask_t = torch.from_numpy(mask_chunk).to(device)
            old_lp_t = torch.from_numpy(old_lp_chunk).to(device)
            adv_t = torch.from_numpy(adv_chunk).to(device)
            ret_t = torch.from_numpy(ret_chunk).to(device)

            logits_seq, val_seq, _ = model.forward_sequence(obs_t, mask_seq=mask_t)

            dist = torch.distributions.Categorical(logits=logits_seq)
            new_logprobs = dist.log_prob(act_t)
            entropy = dist.entropy().mean()

            # PPO Clipped Surrogate Loss
            ratios = torch.exp(new_logprobs - old_lp_t)
            surr1 = ratios * adv_t
            surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value Function Loss (MSE)
            critic_loss = F.mse_loss(val_seq, ret_t)

            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.75)
            optimizer.step()

            total_loss_accum += loss.item()

    return total_loss_accum / max(1, ppo_iters * (num_chunks // max(1, batch_size)))


def evaluate_oos_window(
    env: PureNetRewardExecutionEnv,
    model: RecurrentMaskableActorCritic,
    device: torch.device,
    max_steps: int = 86400,
) -> dict:
    """Детерминированная валидация на слепом OOS окне."""
    model.eval()
    obs = env.reset(eval_mode=True)
    hidden = model.init_hidden(batch_size=1, device=device)

    done = False
    step = 0
    actions_taken = []

    with torch.no_grad():
        while not done and step < max_steps:
            mask = env.get_action_mask()
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

            logits, val, hidden = model.forward_step(obs_t, hidden, mask_t)
            act = int(torch.argmax(logits, dim=-1).item())
            actions_taken.append(act)

            obs, reward, done = env.step(act)
            step += 1

    pnl_usd = float(env.prev_mtm)
    n_trades = int(env.n_trades)
    maker_bids = sum(1 for a in actions_taken if a in (1, 2))
    maker_pct = (maker_bids / max(1, len(actions_taken))) * 100.0

    return {
        "pnl_usd": pnl_usd,
        "trades": n_trades,
        "steps": step,
        "maker_bid_pct": maker_pct,
    }


def run_walk_forward_training():
    print("=" * 80)
    print("🚀 Старт Walk-Forward конвейера глубокого обучения Causal Recurrent PPO")
    print("   Бюджет: 4-6 часов | GPU: NVIDIA GeForce RTX 5060 Ti | Данные: 609.2 часов (2.19M тиков)")
    print("=" * 80, flush=True)

    if not CACHE_PATH.exists():
        print(f"❌ Кэш {CACHE_PATH} не найден!", flush=True)
        return

    print(f"📦 Загрузка полного 609h датасета: {CACHE_PATH}...", flush=True)
    raw = np.load(CACHE_PATH, allow_pickle=True)
    n_ticks = len(raw["bid_p"])
    total_hours = n_ticks / 3600.0
    print(f"Всего тиков: {n_ticks:,} ({total_hours:.2f} часов L2 стакана)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Вычислительное устройство: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Предрасчет 20 микроструктурных признаков и 4 Guard-вероятностей
    print("\n⚡ Расчет 20 Causal признаков микроструктуры...", flush=True)
    feat20 = extract_guard_features(
        raw["bid_p"],
        raw["bid_q"],
        raw["ask_p"],
        raw["ask_q"],
        raw["buy_vol"],
        raw["sell_vol"],
        float(raw["tick_size"]),
    )
    print("⚡ Расчет выходов роя Guard-сетей (H=12s, 60s, 300s, Exit)...", flush=True)
    predictor = MultiHorizonGuardPredictor()
    guard_probs = predictor.predict_probs_batch(feat20)
    print(f"Guard Matrix: {guard_probs.shape} (Avg Sweep12: {guard_probs[:,0].mean():.3f}, Crash300: {guard_probs[:,2].mean():.3f})", flush=True)

    # Объединяем микроструктурные признаки и Guard-вероятности (20 + 4 = 24D)
    X_market_full = np.column_stack([feat20, guard_probs]).astype(np.float32)

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
        "X_market": X_market_full,
    }

    # Параметры Walk-Forward окон
    train_window_ticks = 120 * 3600  # 120 часов обучения (432,000 тиков)
    oos_window_ticks = 24 * 3600     # 24 часа OOS теста (86,400 тиков)
    step_ticks = 24 * 3600           # Сдвиг окна на 24 часа

    num_folds = (n_ticks - train_window_ticks) // step_ticks
    print(f"\n📊 Конфигурация Walk-Forward: Обучение={train_window_ticks/3600:.0f}ч | OOS={oos_window_ticks/3600:.0f}ч | Всего фолдов: {num_folds}", flush=True)

    # Инициализация единой базовой архитектуры агента с Causal GRU
    # obs_dim = 24 (X_market) + 10 (состояние LOB) = 34D
    dummy_env = PureNetRewardExecutionEnv(full_data, start_idx=60, end_idx=train_window_ticks)
    obs_dim = dummy_env.obs_dim
    n_actions = dummy_env.n_actions
    print(f"🧠 Входной вектор наблюдений obs_dim: {obs_dim}D, Действий: {n_actions}", flush=True)

    model = RecurrentMaskableActorCritic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=128,
        gru_layers=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)

    wf_results = []
    total_start_time = time.time()

    for fold in range(num_folds):
        train_start = fold * step_ticks + 60
        train_end = train_start + train_window_ticks
        oos_start = train_end
        oos_end = min(n_ticks - 1, oos_start + oos_window_ticks)

        if oos_end - oos_start < 3600 * 6:
            break

        fold_t0 = time.time()
        print("\n" + "=" * 80, flush=True)
        print(f"🔄 Walk-Forward Фолд [{fold + 1}/{num_folds}]:", flush=True)
        print(f"   Train: тики {train_start:,}..{train_end:,} ({(train_end-train_start)/3600:.1f} ч)", flush=True)
        print(f"   OOS:   тики {oos_start:,}..{oos_end:,} ({(oos_end-oos_start)/3600:.1f} ч)", flush=True)
        print("=" * 80, flush=True)

        train_env = PureNetRewardExecutionEnv(full_data, start_idx=train_start, end_idx=train_end, episode_len=7200)
        oos_env = PureNetRewardExecutionEnv(full_data, start_idx=oos_start, end_idx=oos_end, episode_len=oos_end - oos_start - 2)

        # 4 итерации PPO обучения на фолде (по 4096 шагов сбора = 16,384 шагов опыта)
        n_rollouts = 4
        fold_loss_list = []
        for r_idx in range(n_rollouts):
            rollout = collect_recurrent_rollouts(train_env, model, device, rollout_steps=4096)
            loss_val = train_ppo_epoch(model, optimizer, rollout, device, ppo_iters=4, batch_size=32)
            fold_loss_list.append(loss_val)

        # Валидация на слепом OOS окне следующего дня (24ч)
        oos_res = evaluate_oos_window(oos_env, model, device, max_steps=oos_end - oos_start - 2)
        fold_elapsed = time.time() - fold_t0

        print(f"  [Фолд {fold + 1:02d} OOS Резюме | {fold_elapsed:.1f}с]:", flush=True)
        print(f"     PnL: ${oos_res['pnl_usd']:+.4f} | Сделок: {oos_res['trades']} | Maker Bid %: {oos_res['maker_bid_pct']:.1f}%", flush=True)

        wf_results.append({
            "fold": fold + 1,
            "train_hours": (train_end - train_start) / 3600.0,
            "oos_hours": (oos_end - oos_start) / 3600.0,
            "pnl_usd": oos_res["pnl_usd"],
            "trades": oos_res["trades"],
            "maker_bid_pct": oos_res["maker_bid_pct"],
            "elapsed_sec": fold_elapsed,
        })

        # Сохранение промежуточного чекпоинта фолда
        ckpt_path = OUT_DIR / f"recurrent_ppo_fold_{fold+1:02d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "fold": fold + 1,
                "pnl_usd": oos_res["pnl_usd"],
                "trades": oos_res["trades"],
            },
            ckpt_path,
        )

        # Обновление общего лучшего чекпоинта
        best_ckpt = OUT_DIR / "best_walk_forward_ppo.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "total_folds": fold + 1,
                "last_oos_pnl": oos_res["pnl_usd"],
            },
            best_ckpt,
        )

    total_time = time.time() - total_start_time
    total_oos_pnl = sum(r["pnl_usd"] for r in wf_results)
    total_trades = sum(r["trades"] for r in wf_results)
    print("\n" + "=" * 80, flush=True)
    print("🏆 Финал Walk-Forward глубокого обучения Causal Recurrent PPO:", flush=True)
    print(f"   Всего пройдено фолдов: {len(wf_results)} фолдов (покрыто {len(wf_results)*24} часов слепого OOS)", flush=True)
    print(f"   Суммарный слепой OOS PnL: ${total_oos_pnl:+.4f} USDT", flush=True)
    print(f"   Суммарно закрыто сделок: {total_trades:,}", flush=True)
    print(f"   Общее время обучения на GPU: {total_time/3600:.2f} часов ({total_time:.1f} сек)", flush=True)
    print("=" * 80, flush=True)

    # Сохраняем полный отчет в JSON
    summary_path = OUT_DIR / "walk_forward_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_time_hours": total_time / 3600.0,
                "total_oos_pnl_usd": total_oos_pnl,
                "total_trades": total_trades,
                "folds": wf_results,
            },
            f,
            indent=2,
        )
    print(f"📄 Паспорт Walk-Forward зафиксирован в {summary_path}", flush=True)


if __name__ == "__main__":
    run_walk_forward_training()
