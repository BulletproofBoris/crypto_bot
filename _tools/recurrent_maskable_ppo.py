#!/usr/bin/env python3
"""
_tools/recurrent_maskable_ppo.py
=============================================================================
Архитектура нейроагента v8.0 Causal Recurrent PPO с памятью микроструктуры
и сквозным Soft-Masking на базе роя нейро-ограничителей (Guard Ensemble).

Ключевые особенности:
1. Causal GRU (окно памяти W=32 шагов или скрытое состояние h_t, строго без Lookahead).
2. Динамическое маскирование (Soft Action Masking):
   - Опасность пролива (p_crash_300s > 0.65) / сползания (p_bleed_60s > 0.60): плавное подавление Maker Bid;
   - Защита капитала: Taker Market Exit разрешен только при угрозе зависания в просадке (p_exit_30s > 0.65);
   - Снятие дискретного ступора: при спокойном стакане (p < 0.20) входы разрешены без обязательного отката.
=============================================================================
"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentMaskableActorCritic(nn.Module):
    """
    Actor-Critic архитектура со встроенной рекуррентной памятью Causal GRU
    и поддержкой маскирования недопустимых действий.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 7,
        hidden_dim: int = 128,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        # Входной энкодер признаков
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Однонаправленная каузальная память (строго unidirectional)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Головы политики (Actor) и функции ценности (Critic)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def init_hidden(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Инициализирует нулевое скрытое состояние GRU."""
        return torch.zeros(self.gru_layers, batch_size, self.hidden_dim, device=device)

    def forward_step(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Одиночный шаг инференса (для среды исполнения / лайв-демона):
        obs: (batch_size, obs_dim)
        hidden: (gru_layers, batch_size, hidden_dim)
        mask: (batch_size, n_actions) bool tensor
        Возвращает: logits, value, next_hidden
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        batch_size = obs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)

        feat = self.encoder(obs).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        out, next_hidden = self.gru(feat, hidden)
        h_last = out.squeeze(1)  # (batch_size, hidden_dim)

        logits = self.actor(h_last)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        val = self.critic(h_last).squeeze(-1)
        return logits, val, next_hidden

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        mask_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Последовательный прогон батча траекторий (для PPO обучения на GPU):
        obs_seq: (batch_size, seq_len, obs_dim)
        hidden: (gru_layers, batch_size, hidden_dim)
        mask_seq: (batch_size, seq_len, n_actions) bool
        Возвращает: logits_seq, values_seq, final_hidden
        """
        batch_size, seq_len, _ = obs_seq.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size, obs_seq.device)

        # Пропускаем через энкодер
        flat_obs = obs_seq.contiguous().view(-1, self.obs_dim)
        flat_feat = self.encoder(flat_obs)
        feat_seq = flat_feat.view(batch_size, seq_len, self.hidden_dim)

        # Каузальный GRU по всей длине последовательности
        gru_out, final_hidden = self.gru(feat_seq, hidden)  # (batch_size, seq_len, hidden_dim)

        flat_gru = gru_out.contiguous().view(-1, self.hidden_dim)
        flat_logits = self.actor(flat_gru)
        flat_val = self.critic(flat_gru).squeeze(-1)

        logits_seq = flat_logits.view(batch_size, seq_len, self.n_actions)
        values_seq = flat_val.view(batch_size, seq_len)

        if mask_seq is not None:
            logits_seq = logits_seq.masked_fill(~mask_seq, -1e9)

        return logits_seq, values_seq, final_hidden


def compute_v8_guard_soft_mask(
    base_mask: np.ndarray,
    p_sweep12: float,
    p_bleed60: float,
    p_crash300: float,
    p_exit30: float,
    has_pos: bool,
    order_side: Optional[str] = None,
) -> np.ndarray:
    """
    Вычисляет адаптивную нейросетевую маску допустимости действий v8.0 на основе Guard-вероятностей.
    Заменяет жесткие дискретные барьеры (dd_300 >= 1.0, spread_ticks >= 2.0).

    Действия агента:
    0: HOLD / CANCEL
    1: POST_MAKER_BEST_BID (агрессивный мейкер бид)
    2: POST_MAKER_PASSIVE_BID (пассивный мейкер бид)
    3: POST_MAKER_DEEP_BID (глубокий мейкер бид)
    4: POST_MAKER_BEST_ASK (выход мейкером на лучшем аске)
    5: POST_MAKER_PASSIVE_ASK (выход мейкером пассивно)
    6: EMERGENCY_TAKER_EXIT (экстренный выход тейкером)
    """
    mask = np.copy(base_mask)

    # 1. Защита покупок (Maker Bids: 1, 2, 3) от водопада и сползания:
    # Если вероятность пролива на 300с высока (> 0.65) или сползания на 60с (> 0.60) — покупки блокируются
    danger_down = (p_crash300 > 0.65) or (p_bleed60 > 0.60)
    if danger_down and not has_pos:
        mask[1] = False
        mask[2] = False
        mask[3] = False
    elif not has_pos:
        # Спокойный или умеренный стакан: разрешаем выставление бидов даже без отката!
        # Но если есть микро-угроза сползания (0.40..0.60), запрещаем только агрессивный Best Bid,
        # оставляя пассивный и глубокий бид (2, 3)
        if p_bleed60 > 0.45:
            mask[1] = False

    # 2. Адаптивное управление выходом из позиции (Действие 6: Emergency Taker):
    if has_pos:
        # Разрешаем экстренный тейкер-выход ТОЛЬКО если есть реальная угроза провала (p_exit30 > 0.65 или p_crash300 > 0.70)
        # Это исключает случайный слив на комиссиях 0.10% при спокойном рынке
        threat_detected = (p_exit30 > 0.65) or (p_crash300 > 0.70)
        mask[6] = threat_detected
    else:
        mask[6] = False

    # Всегда гарантируем, что действие 0 (HOLD / CANCEL) доступно
    mask[0] = True

    return mask

