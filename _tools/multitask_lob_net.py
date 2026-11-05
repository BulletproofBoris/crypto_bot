#!/usr/bin/env python3
"""
_tools/multitask_lob_net.py
=============================================================================
Сквозная мультизадачная пространственно-временная нейросеть (Multi-Task LOB Net)
для адаптивного микроструктурного исполнения ордеров (v8.0 Deep Architecture).

Архитектура:
1. Spatial LOB Encoder:
   - Входной тензор книги L2: (batch, seq_len, 10, 4) — 10 уровней Bid/Ask (цена, объем, глубина, дисбаланс).
   - 1D Convolutions по уровням глубины книги: извлечение паттернов плотности, айсбергов и перекосов стакана.
2. Order Flow & Account Encoder:
   - Вектор динамики потока: (batch, seq_len, 16) — CVD, OFI, волатильность, скорости, статус аккаунта и очереди.
3. Spatial-Temporal Causal GRU:
   - Объединяет пространственные признаки стакана и потока ордеров.
   - Однонаправленная рекуррентная память (W=64 тика, hidden_dim=256, без lookahead bias).
   - Формирует общий латентный вектор контекста рынка Z_t (dim=256).
4. Multi-Task Learning Heads (совместное обучение):
   - Head 1: Micro-Sweep 12s Risk (BCE/Focal Loss)
   - Head 2: Staircase Bleed 60s Risk (BCE/Focal Loss)
   - Head 3: Waterfall Crash 300s Risk (BCE/Focal Loss)
   - Head 4: Adverse Hold 30s Risk (BCE/Focal Loss)
   - Head 5: Policy Actor (PPO Clipped Objective, 7 действий)
   - Head 6: Value Critic (MSE Loss / GAE Net Advantage)
=============================================================================
"""

import math
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLOBEncoder(nn.Module):
    """
    Пространственный сверточный энкодер 10-уровневого стакана заявок L2.
    Извлекает локальные микроструктурные паттерны:
    - Стенки и пустоты ликвидности;
    - Асимметрию дисбаланса по глубине L1..L10;
    - Градиенты изменения объемов.
    """

    def __init__(self, in_channels: int = 4, out_features: int = 64):
        super().__init__()
        # in_channels = 4: [bid_price_norm, bid_qty_norm, ask_price_norm, ask_qty_norm]
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.norm1 = nn.LayerNorm(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.norm2 = nn.LayerNorm(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, out_features)

    def forward(self, lob_tensor: torch.Tensor) -> torch.Tensor:
        """
        lob_tensor: (batch_size, 10, in_channels)
        Возвращает: (batch_size, out_features)
        """
        x = lob_tensor.transpose(1, 2)  # (batch, in_channels, 10)
        x = F.gelu(self.conv1(x))       # (batch, 32, 9)
        x = x.transpose(1, 2)
        x = self.norm1(x).transpose(1, 2)

        x = F.gelu(self.conv2(x))       # (batch, 64, 7)
        x = x.transpose(1, 2)
        x = self.norm2(x).transpose(1, 2)

        x = self.pool(x).squeeze(-1)    # (batch, 64)
        out = F.gelu(self.fc(x))        # (batch, out_features)
        return out


class MultiTaskSpatialTemporalBackbone(nn.Module):
    """
    Единый Spatial-Temporal Backbone с памятью Causal GRU:
    Объединяет пространственные свертки LOB и поток ордеров в общее представление Z_t (dim=256).
    """

    def __init__(self, flow_dim: int = 16, latent_dim: int = 256, gru_layers: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru_layers = gru_layers

        # 1. Пространственный энкодер стакана (10 уровней LOB)
        self.lob_encoder = SpatialLOBEncoder(in_channels=4, out_features=64)

        # 2. Энкодер потока ордеров и состояния аккаунта
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 3. Слияние признаков (64 LOB + 64 Flow = 128) -> GRU (latent_dim=256)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,  # Строго каузальная память без lookahead!
        )

    def init_hidden(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch_size, self.latent_dim, device=device)

    def forward_step(
        self,
        lob_10: torch.Tensor,
        flow_feat: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Одиночный шаг для инференса и боевого исполнения:
        lob_10: (batch, 10, 4)
        flow_feat: (batch, flow_dim)
        Возвращает: latent_z (batch, latent_dim), next_hidden
        """
        batch_size = lob_10.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, lob_10.device)

        lob_repr = self.lob_encoder(lob_10)          # (batch, 64)
        flow_repr = self.flow_encoder(flow_feat)     # (batch, 64)

        combined = torch.cat([lob_repr, flow_repr], dim=-1) # (batch, 128)
        fused = self.fusion(combined).unsqueeze(1)          # (batch, 1, latent_dim)

        out, next_hidden = self.gru(fused, hidden)
        z_last = out.squeeze(1)  # (batch, latent_dim)
        return z_last, next_hidden

    def encode_lob_batch(self, lob_batch: torch.Tensor) -> torch.Tensor:
        """Батчевый предрасчет LOB-эмбеддингов на GPU (N, 10, 4) -> (N, 64)"""
        return self.lob_encoder(lob_batch)

    def forward_step_fast(
        self,
        lob_repr: torch.Tensor,
        flow_feat: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ультрабыстрый шаг инференса с готовым LOB-эмбеддингом"""
        batch_size = lob_repr.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, lob_repr.device)

        flow_repr = self.flow_encoder(flow_feat)
        combined = torch.cat([lob_repr, flow_repr], dim=-1)
        fused = self.fusion(combined).unsqueeze(1)

        out, next_hidden = self.gru(fused, hidden)
        return out.squeeze(1), next_hidden

    def forward_sequence(
        self,
        lob_seq: torch.Tensor,
        flow_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Последовательный прогон батча траекторий для сквозного обучения на GPU:
        lob_seq: (batch, seq_len, 10, 4)
        flow_seq: (batch, seq_len, flow_dim)
        Возвращает: latent_seq (batch, seq_len, latent_dim), final_hidden
        """
        batch_size, seq_len, _, _ = lob_seq.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size, lob_seq.device)

        # Сворачиваем batch и time для пространственного энкодера
        flat_lob = lob_seq.contiguous().view(-1, 10, 4)
        flat_lob_repr = self.lob_encoder(flat_lob)
        lob_repr_seq = flat_lob_repr.view(batch_size, seq_len, 64)

        flat_flow = flow_seq.contiguous().view(-1, flow_seq.size(-1))
        flat_flow_repr = self.flow_encoder(flat_flow)
        flow_repr_seq = flat_flow_repr.view(batch_size, seq_len, 64)

        combined_seq = torch.cat([lob_repr_seq, flow_repr_seq], dim=-1) # (batch, seq_len, 128)
        flat_combined = combined_seq.view(-1, 128)
        flat_fused = self.fusion(flat_combined)
        fused_seq = flat_fused.view(batch_size, seq_len, self.latent_dim)

        out_seq, final_hidden = self.gru(fused_seq, hidden)
        return out_seq, final_hidden


class MultiTaskLOBNet(nn.Module):
    """
    Полная мультизадачная нейросетевая модель v8.0:
    Общий Shared Backbone (Spatial 1D CNN + Causal GRU)
    + 4 головы предсказания рисков микроструктуры (Auxiliary Tasks)
    + Голова Политики (Actor) и Ценности (Critic) (RL Task).
    """

    def __init__(
        self,
        flow_dim: int = 16,
        n_actions: int = 7,
        latent_dim: int = 256,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.latent_dim = latent_dim

        # Общий латентный энкодер стакана и потока
        self.backbone = MultiTaskSpatialTemporalBackbone(
            flow_dim=flow_dim,
            latent_dim=latent_dim,
            gru_layers=gru_layers,
        )

        # 4 вспомогательные головы прогнозирования микроструктурных рисков (Shared Representation)
        self.head_sweep_12s = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_bleed_60s = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_crash_300s = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.head_exit_30s = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Основные головы Reinforcement Learning (Actor-Critic с явным Multi-Task Risk Feedback)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim + 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(latent_dim + 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def init_hidden(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return self.backbone.init_hidden(batch_size, device)

    def forward_step(
        self,
        lob_10: torch.Tensor,
        flow_feat: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Одиночный шаг исполнения ордеров:
        Возвращает словарь с логитами политики, оценкой ценности, вероятностями рисков и скрытым состоянием.
        """
        if lob_10.dim() == 2:
            lob_10 = lob_10.unsqueeze(0)
        if flow_feat.dim() == 1:
            flow_feat = flow_feat.unsqueeze(0)

        z, next_hidden = self.backbone.forward_step(lob_10, flow_feat, hidden)

        p_sweep = torch.sigmoid(self.head_sweep_12s(z))
        p_bleed = torch.sigmoid(self.head_bleed_60s(z))
        p_crash = torch.sigmoid(self.head_crash_300s(z))
        p_exit = torch.sigmoid(self.head_exit_30s(z))

        z_aug = torch.cat([z, p_sweep, p_bleed, p_crash, p_exit], dim=-1)

        logits = self.actor(z_aug)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)

        value = self.critic(z_aug).squeeze(-1)

        return {
            "logits": logits,
            "value": value,
            "p_sweep_12s": p_sweep.squeeze(-1),
            "p_bleed_60s": p_bleed.squeeze(-1),
            "p_crash_300s": p_crash.squeeze(-1),
            "p_exit_30s": p_exit.squeeze(-1),
            "hidden": next_hidden,
        }

    def encode_lob_batch(self, lob_batch: torch.Tensor) -> torch.Tensor:
        """Батчевый предрасчет LOB-эмбеддингов (N, 10, 4) -> (N, 64)"""
        return self.backbone.encode_lob_batch(lob_batch)

    def forward_step_fast(
        self,
        lob_repr: torch.Tensor,
        flow_feat: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Ультрабыстрый шаг симуляции OOS с риск-аугментацией"""
        if lob_repr.dim() == 1:
            lob_repr = lob_repr.unsqueeze(0)
        if flow_feat.dim() == 1:
            flow_feat = flow_feat.unsqueeze(0)

        z, next_hidden = self.backbone.forward_step_fast(lob_repr, flow_feat, hidden)

        p_sweep = torch.sigmoid(self.head_sweep_12s(z))
        p_bleed = torch.sigmoid(self.head_bleed_60s(z))
        p_crash = torch.sigmoid(self.head_crash_300s(z))
        p_exit = torch.sigmoid(self.head_exit_30s(z))

        z_aug = torch.cat([z, p_sweep, p_bleed, p_crash, p_exit], dim=-1)

        logits = self.actor(z_aug)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)

        value = self.critic(z_aug).squeeze(-1)
        return {
            "logits": logits,
            "value": value,
            "p_sweep_12s": p_sweep.squeeze(-1),
            "p_bleed_60s": p_bleed.squeeze(-1),
            "p_crash_300s": p_crash.squeeze(-1),
            "p_exit_30s": p_exit.squeeze(-1),
            "hidden": next_hidden,
        }

    def forward_sequence(
        self,
        lob_seq: torch.Tensor,
        flow_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        mask_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Последовательный сквозной прогон для мультизадачного обучения на GPU:
        """
        z_seq, final_hidden = self.backbone.forward_sequence(lob_seq, flow_seq, hidden)
        batch_size, seq_len, _ = z_seq.size()
        flat_z = z_seq.contiguous().view(-1, self.latent_dim)

        # Выходы вспомогательных голов рисков
        p_sw_flat = torch.sigmoid(self.head_sweep_12s(flat_z))
        p_bl_flat = torch.sigmoid(self.head_bleed_60s(flat_z))
        p_cr_flat = torch.sigmoid(self.head_crash_300s(flat_z))
        p_ex_flat = torch.sigmoid(self.head_exit_30s(flat_z))

        flat_z_aug = torch.cat([flat_z, p_sw_flat, p_bl_flat, p_cr_flat, p_ex_flat], dim=-1)

        flat_logits = self.actor(flat_z_aug)
        logits_seq = flat_logits.view(batch_size, seq_len, self.n_actions)
        if mask_seq is not None:
            logits_seq = logits_seq.masked_fill(~mask_seq, -1e9)

        values_seq = self.critic(flat_z_aug).squeeze(-1).view(batch_size, seq_len)

        p_sw_seq = p_sw_flat.squeeze(-1).view(batch_size, seq_len)
        p_bl_seq = p_bl_flat.squeeze(-1).view(batch_size, seq_len)
        p_cr_seq = p_cr_flat.squeeze(-1).view(batch_size, seq_len)
        p_ex_seq = p_ex_flat.squeeze(-1).view(batch_size, seq_len)

        return {
            "logits": logits_seq,
            "values": values_seq,
            "p_sweep_12s": p_sw_seq,
            "p_bleed_60s": p_bl_seq,
            "p_crash_300s": p_cr_seq,
            "p_exit_30s": p_ex_seq,
            "hidden": final_hidden,
            "latent_z": z_seq,
        }
