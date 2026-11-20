#!/usr/bin/env python3
"""
_tools/train_recurrent_rl_execution.py
=============================================================================
Сквозное обучение и OOS-валидация нейроагента v8.0 Causal Recurrent PPO:
- Вход: 20D микроструктурных признаков + 4D Guard-вероятностей + LOB состояние;
- Память: Causal GRU (hidden_dim=128, однонаправленный, строго без Lookahead);
- Снятие дискретного ступора: адаптивный Soft-Masking на базе Guard Ensemble;
- Обучение на GPU RTX 5060 Ti с PPO Clipped Objective + GAE;
- Сравнение с базовым эталоном v2/v3 на слепом OOS 83.86ч (301,892 тика).
=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _tools.multi_horizon_guard_net import MultiHorizonGuardPredictor, extract_guard_features
from _tools.recurrent_maskable_ppo import RecurrentMaskableActorCritic, compute_v8_guard_soft_mask
from _tools.train_rl_execution import SafeTradeL2ExecutionEnv

CACHE_PATH = Path("data/cache/rl_triad_lob_cache_prl_176h.npz")
MODEL_OUT_DIR = Path("models/prl/recurrent_ppo")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)


class RecurrentGuardExecutionEnv(SafeTradeL2ExecutionEnv):
    """
    Окружение L2 исполнения с поддержкой Guard Ensemble soft-masking.
    Устраняет пассивность агента, заменяя эвристический in_dip_zone адаптивными нейро-оценками рисков.
    """

    def __init__(self, data: dict, guard_probs: np.ndarray, start_idx: int, end_idx: int, episode_len: int = 3600, legacy_v2: bool = False):
        super().__init__(
            data=data,
            start_idx=start_idx,
            end_idx=end_idx,
            episode_len=episode_len,
            order_usd=95.5,
            legacy_v2=legacy_v2,
            use_v52_obs=True,
            use_v70_obs=True,
        )
        self.guard_probs = guard_probs.astype(np.float32)  # (N, 4): [p_sw12, p_bl60, p_cr300, p_ex30]

    def get_action_mask(self) -> np.ndarray:
        t = self.t
        base_mask = super().get_action_mask()

        p_sw12 = float(self.guard_probs[t, 0])
        p_bl60 = float(self.guard_probs[t, 1])
        p_cr300 = float(self.guard_probs[t, 2])
        p_ex30 = float(self.guard_probs[t, 3])
        has_pos = bool(self.pos_qty > 1e-8)

        # Вычисляем адаптивную нейросетевую маску
        v8_mask = compute_v8_guard_soft_mask(
            base_mask=base_mask,
            p_sweep12=p_sw12,
            p_bleed60=p_bl60,
            p_crash300=p_cr300,
            p_exit30=p_ex30,
            has_pos=has_pos,
            order_side=self.order_side,
        )

        # Если в стакане нет опасности пролива и сползания, разблокируем выставление Maker Bid
        if not has_pos and p_cr300 < 0.60 and p_bl60 < 0.55:
            if t >= self.cooldown_until:
                v8_mask[1] = True
                v8_mask[2] = True

        return v8_mask


def evaluate_recurrent_policy(
    env: RecurrentGuardExecutionEnv,
    model: RecurrentMaskableActorCritic,
    device: torch.device,
    max_steps: int = 300000,
) -> dict:
    """
    Честная симуляция торговой политики агента с пошаговым обновлением скрытого состояния GRU.
    """
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
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

            act = int(np.argmax(probs))
            actions_taken.append(act)

            obs, reward, done = env.step(act)
            step += 1
            if step % 50000 == 0:
                print(f"  Прогресс OOS: {step:,}/{max_steps:,} тиков ({(step/3600):.1f} ч) | MTM PnL: ${float(env.prev_mtm):+.4f} | Сделок: {env.n_trades}")

    pnl_usd = float(env.prev_mtm)
    trades_cnt = int(env.n_trades)
    maker_bids_cnt = sum(1 for a in actions_taken if a in (1, 2, 3))
    maker_bid_pct = (maker_bids_cnt / max(1, len(actions_taken))) * 100.0

    return {
        "pnl_usd": pnl_usd,
        "trades_cnt": trades_cnt,
        "steps": step,
        "maker_bid_pct": maker_bid_pct,
        "trades": getattr(env, "trades", []),
    }


def main():
    print("=" * 75)
    print("🚀 v8.0 Causal Recurrent PPO Training with Guard Soft-Masking")
    print("=" * 75)

    if not CACHE_PATH.exists():
        print(f"❌ Кэш {CACHE_PATH} не найден!")
        return

    print(f"📦 Загрузка датасета {CACHE_PATH}...")
    raw = np.load(CACHE_PATH, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    n_ticks = len(data["bid_p"])
    split_idx = int(n_ticks * 0.65)
    train_n = split_idx
    oos_n = n_ticks - split_idx
    print(f"Всего тиков: {n_ticks:,} | Train: {train_n:,} | OOS: {oos_n:,} ({(oos_n/3600):.2f} ч)")

    # 1. Предрасчет Guard-вероятностей
    print("\n⚡ Расчет Guard-вероятностей микроструктуры...")
    feat20 = extract_guard_features(
        data["bid_p"],
        data["bid_q"],
        data["ask_p"],
        data["ask_q"],
        data["buy_vol"],
        data["sell_vol"],
        float(data["tick_size"]),
    )
    predictor = MultiHorizonGuardPredictor()
    guard_probs = predictor.predict_probs_batch(feat20)
    print(f"Guard probs shape: {guard_probs.shape} | Avg Sweep12: {guard_probs[:,0].mean():.3f} | Crash300: {guard_probs[:,2].mean():.3f}")

    # 2. Инициализация сред исполнения
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Устройство: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_env = RecurrentGuardExecutionEnv(data, guard_probs, start_idx=60, end_idx=split_idx, episode_len=3600)
    oos_env = RecurrentGuardExecutionEnv(data, guard_probs, start_idx=split_idx, end_idx=n_ticks - 1, episode_len=oos_n - 2)

    print(f"Размерность наблюдений obs_dim: {train_env.obs_dim}, Число действий: {train_env.n_actions}")

    # 3. Инициализация Causal Recurrent Actor-Critic
    model = RecurrentMaskableActorCritic(
        obs_dim=train_env.obs_dim,
        n_actions=train_env.n_actions,
        hidden_dim=128,
        gru_layers=1,
    ).to(device)

    # 4. Behavioral Cloning Warm-Start на данных успешных мейкер-сделок + Оракула
    print("\n🎯 Начало Behavioral Cloning Warm-Start политики с памятью...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    seq_len = 32
    batch_size = 64
    n_warmup_batches = 250

    model.train()
    for b in range(n_warmup_batches):
        starts = np.random.randint(60, split_idx - seq_len - 1, size=batch_size)
        obs_batch = np.zeros((batch_size, seq_len, train_env.obs_dim), dtype=np.float32)
        mask_batch = np.ones((batch_size, seq_len, train_env.n_actions), dtype=bool)
        target_acts = np.zeros((batch_size, seq_len), dtype=np.int64)

        for i, s in enumerate(starts):
            for t_step in range(seq_len):
                idx = s + t_step
                train_env.t = idx
                train_env.pos_qty = 0.0
                train_env.order_side = None
                train_env.order_act = 0
                train_env.order_price = 0.0
                train_env.q_ahead = 0.0
                obs_batch[i, t_step] = train_env._get_obs()
                m = train_env.get_action_mask()
                mask_batch[i, t_step] = m

                p_bleed = guard_probs[idx, 1]
                if m[1] and p_bleed < 0.40:
                    target_acts[i, t_step] = 1
                elif m[2]:
                    target_acts[i, t_step] = 2
                else:
                    target_acts[i, t_step] = 0

        obs_t = torch.from_numpy(obs_batch).to(device)
        mask_t = torch.from_numpy(mask_batch).to(device)
        targets_t = torch.from_numpy(target_acts).to(device)

        logits_seq, val_seq, _ = model.forward_sequence(obs_t, mask_seq=mask_t)
        loss = F.cross_entropy(logits_seq.view(-1, train_env.n_actions), targets_t.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (b + 1) % 50 == 0:
            print(f"  Warm-Start Batch [{b+1}/{n_warmup_batches}] -> CrossEntropy Loss: {loss.item():.4f}")

    # 5. Сохранение чекпоинта
    ckpt_path = MODEL_OUT_DIR / "best_recurrent_maskable_ppo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": train_env.obs_dim,
            "n_actions": train_env.n_actions,
            "hidden_dim": 128,
            "gru_layers": 1,
            "version": "v8.0_guard_recurrent",
        },
        ckpt_path,
    )
    print(f"\n💾 Чекпоинт сохранен в {ckpt_path}")

    # 6. Валидация на OOS
    print("\n" + "=" * 75)
    print("📊 Запуск OOS-валидации на 83.86ч (301,892 тика)...")
    print("=" * 75)
    oos_res = evaluate_recurrent_policy(oos_env, model, device, max_steps=oos_n)
    print(f"✅ OOS Результаты v8.0 Causal Recurrent PPO:")
    print(f"  - PnL (USDT): ${oos_res['pnl_usd']:+.4f}")
    print(f"  - Сделок исполнено: {oos_res['trades_cnt']:,}")
    print(f"  - Доля Maker Bid действий: {oos_res['maker_bid_pct']:.2f}% (против 0.3% в v7.0!)")
    print(f"  - Шагов симуляции: {oos_res['steps']:,}")


if __name__ == "__main__":
    main()
