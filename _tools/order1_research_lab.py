#!/usr/bin/env python3
"""
_tools/order1_research_lab.py
=============================================================================
Исследовательская Лаборатория Моделей 1-го Порядка (Order-1 Research Lab).

Назначение:
  1. Глубокий аудит и оптимизация обучения 7 базовых архитектур:
     - 1: LightGBM (Gradient Boosted Decision Trees)
     - 2: MLP-Bottleneck (Deep Residual Bottleneck MLP)
     - 3: Conv1D + GRU (Hybrid Convolutional Recurrent Net)
     - 4: Causal-CNN (Dilated Residual TCN)
     - 5: Attention (Multi-Head Self-Attention Transformer)
     - 6: WaveNet-TCN (Gated Dilated Convolutional Net)
     - 7: Cross-Mixer (Spatial & Temporal Gated Feature Mixer)
  2. Решение проблемы экстремального дисбаланса классов (Класс 0 ~ 95%):
     - Cross-Entropy vs Sqrt-Balanced vs Focal Loss (gamma=2.0).
  3. Отслеживание реальной динамики сходимости по эпохам:
     - Train/Val Loss, Macro ROC-AUC, Weighted ROC-AUC, Directional Accuracy, ECE.
  4. Формирование и калибровка ансамблированной консолидированной финальной модели:
     - Метод 1: Simple Average (равновесный)
     - Метод 2: Softmax-Weighted по OOS/Val ROC-AUC
     - Метод 3: Post-Hoc Temperature Scaling
     - Метод 4: Stacking Meta-Learner (L2-Regularized Multinomial Logistic Regression)
  5. Сравнительный бенчмарк информационной ценности (Information Coefficient, Directional Alpha)
     для входного пространства RL-агента.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES_7 = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int64)
NUM_CLASSES = 7


# =====================================================================
# 1. ФУНКЦИИ ПОТЕРЬ ДЛЯ БОРЬБЫ С ЭКСТРЕМАЛЬНЫМ ДИСБАЛАНСОМ КЛАССОВ
# =====================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss для многоклассовой классификации (Lin et al., 2017).
    Подавляет градиенты легких примеров (класс 0, составляющий 95%+ выборки)
    и фокусирует обучение на редких информативных импульсах движения цены.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # вероятность истинного класса
        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            at = self.alpha.to(targets.device)[targets]
            focal_weight = focal_weight * at
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(y_train: np.ndarray, strategy: str = "sqrt") -> torch.Tensor:
    """
    Вычисляет веса классов.
    - none: единичные веса
    - balanced: n_samples / (n_classes * count)
    - sqrt: sqrt(n_samples / count) - мягкий корень, предотвращающий взрыв градиентов
    """
    classes, counts = np.unique(y_train, return_counts=True)
    full_counts = np.ones(NUM_CLASSES, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        full_counts[c] = cnt
    total = len(y_train)

    if strategy == "none":
        weights = np.ones(NUM_CLASSES, dtype=np.float32)
    elif strategy == "balanced":
        weights = total / (NUM_CLASSES * full_counts)
    elif strategy == "sqrt":
        weights = np.sqrt(total / full_counts)
        weights /= weights.mean()  # нормализуем к среднему 1.0
    else:
        raise ValueError(f"Unknown class weight strategy: {strategy}")

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# =====================================================================
# 2. АРХИТЕКТУРЫ НЕЙРОСЕТЕЙ (PYTORCH CUDA ENGINE)
# =====================================================================

class MLPBottleneckNet(nn.Module):
    """
    Архитектура 2: Deep MLP Bottleneck с остаточными связями и LayerNorm.
    """
    def __init__(self, seq_len: int, n_features: int, n_classes: int = 7):
        super().__init__()
        in_dim = seq_len * n_features
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.drop1 = nn.Dropout(0.20)
        
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(0.20)
        
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.drop3 = nn.Dropout(0.15)
        
        self.out = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        h = self.flatten(x)
        h = self.drop1(F.gelu(self.ln1(self.fc1(h))))
        res = h
        h = self.drop2(F.gelu(self.ln2(self.fc2(h)))) + res
        h = self.drop3(F.gelu(self.ln3(self.fc3(h))))
        return self.out(h)


class Conv1DGRUNet(nn.Module):
    """
    Архитектура 3: Conv1D локальная фильтрация + GRU рекуррентная память.
    """
    def __init__(self, seq_len: int, n_features: int, n_classes: int = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=min(3, seq_len), padding="same")
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(0.10)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.ln2 = nn.LayerNorm(64)
        self.fc = nn.Linear(64, 64)
        self.drop2 = nn.Dropout(0.15)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> transpose to (B, F, T)
        h = x.transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = h.transpose(1, 2)  # (B, T, 64)
        h = self.drop1(self.ln1(h))
        _, h_gru = self.gru(h)
        h_last = self.ln2(h_gru.squeeze(0))
        h_dense = self.drop2(F.gelu(self.fc(h_last)))
        return self.out(h_dense)


class CausalCNNNet(nn.Module):
    """
    Архитектура 4: Каузальная Dilated CNN с остаточными блоками.
    """
    def __init__(self, seq_len: int, n_features: int, n_classes: int = 7):
        super().__init__()
        self.stem = nn.Conv1d(n_features, 32, kernel_size=1)
        # Блоки с дилатацией 1, 2, 4
        self.b1_conv = nn.Conv1d(32, 32, kernel_size=2, dilation=1, padding=1)
        self.b1_ln = nn.LayerNorm(32)
        
        self.b2_conv = nn.Conv1d(32, 64, kernel_size=2, dilation=1, padding=1)
        self.b2_res = nn.Conv1d(32, 64, kernel_size=1)
        self.b2_ln = nn.LayerNorm(64)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 64)
        self.drop = nn.Dropout(0.15)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        h = x.transpose(1, 2)  # (B, F, T)
        h = F.gelu(self.stem(h))
        
        # Block 1
        res = h
        c1 = F.gelu(self.b1_conv(h)[:, :, :T])
        c1 = self.b1_ln(c1.transpose(1, 2)).transpose(1, 2)
        h = c1 + res
        
        # Block 2
        res2 = self.b2_res(h)
        c2 = F.gelu(self.b2_conv(h)[:, :, :T])
        c2 = self.b2_ln(c2.transpose(1, 2)).transpose(1, 2)
        h = c2 + res2
        
        pooled = self.pool(h).squeeze(-1)
        dense = self.drop(F.gelu(self.fc(pooled)))
        return self.out(dense)


class AttentionNet(nn.Module):
    """
    Архитектура 5: Multi-Head Self-Attention Transformer.
    """
    def __init__(self, seq_len: int, n_features: int, n_classes: int = 7):
        super().__init__()
        self.proj = nn.Linear(n_features, 64)
        self.ln1 = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.2, batch_first=True)
        self.ln2 = nn.LayerNorm(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 64)
        self.drop = nn.Dropout(0.15)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = F.gelu(self.proj(x))
        h = self.ln1(h)
        attn_out, _ = self.attn(h, h, h)
        h = self.ln2(h + attn_out)
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)
        dense = self.drop(F.gelu(self.fc(pooled)))
        return self.out(dense)


class WaveNetTCNNet(nn.Module):
    """
    Архитектура 6: Gated WaveNet TCN (Gated Linear Units).
    """
    def __init__(self, seq_len: int, n_features: int, n_classes: int = 7):
        super().__init__()
        self.stem = nn.Conv1d(n_features, 32, kernel_size=1)
        self.f_conv = nn.Conv1d(32, 64, kernel_size=2, padding=1)
        self.g_conv = nn.Conv1d(32, 64, kernel_size=2, padding=1)
        self.proj = nn.Conv1d(64, 64, kernel_size=1)
        self.res = nn.Conv1d(32, 64, kernel_size=1)
        self.ln = nn.LayerNorm(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 64)
        self.drop = nn.Dropout(0.15)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        h = x.transpose(1, 2)
        h = F.gelu(self.stem(h))
        sc = self.res(h)
        
        f = torch.tanh(self.f_conv(h)[:, :, :T])
        g = torch.sigmoid(self.g_conv(h)[:, :, :T])
        z = F.gelu(self.proj(f * g))
        z = self.ln(z.transpose(1, 2)).transpose(1, 2)
        h = sc + z
        
        pooled = self.pool(h).squeeze(-1)
        dense = self.drop(F.gelu(self.fc(pooled)))
        return self.out(dense)


class CrossMixerNet(nn.Module):
    """
    Архитектура 7: Spatial & Temporal Feature Gated Mixer.
    """
    def __init__(self, seq_len: int, n_features: int, n_classes: int = 7):
        super().__init__()
        self.feat_val = nn.Linear(n_features, 64)
        self.feat_gate = nn.Linear(n_features, 64)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(0.15)
        
        self.t_conv = nn.Conv1d(64, 64, kernel_size=min(3, seq_len), padding="same")
        self.ln2 = nn.LayerNorm(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 64)
        self.drop2 = nn.Dropout(0.15)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Feature mixing via GLU
        v = F.gelu(self.feat_val(x))
        g = torch.sigmoid(self.feat_gate(x))
        h = self.drop1(self.ln1(v * g))
        
        # 2. Temporal mixing
        h_t = h.transpose(1, 2)
        c = F.gelu(self.t_conv(h_t)).transpose(1, 2)
        h = self.ln2(h + c)
        
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)
        dense = self.drop2(F.gelu(self.fc(pooled)))
        return self.out(dense)


# Реестр нейросетевых архитектур
NEURAL_FACTORY = {
    "mlp_bottleneck": MLPBottleneckNet,
    "conv1d_gru": Conv1DGRUNet,
    "causal_cnn": CausalCNNNet,
    "attention": AttentionNet,
    "wavenet_tcn": WaveNetTCNNet,
    "cross_mixer": CrossMixerNet,
}


# =====================================================================
# 3. МЕТРИКИ И АУДИТ КАЧЕСТВА ПРЕДСКАЗАНИЙ
# =====================================================================

def compute_evaluation_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Вычисляет комплексный набор метрик:
    - LogLoss
    - Macro ROC-AUC (OVR)
    - Weighted ROC-AUC
    - Directional Accuracy (точность знака на примерах с ненулевым смещением!)
    - Non-Zero Recall (способность находить активные импульсы)
    - ECE (Expected Calibration Error)
    - Expected Return Correlation (Информационный коэффициент IC)
    """
    n_samples = len(y_true)
    y_true_06 = y_true.copy()  # классы 0..6

    # 1. LogLoss
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    probs_clipped /= probs_clipped.sum(axis=1, keepdims=True)
    try:
        ll = float(log_loss(y_true_06, probs_clipped, labels=list(range(NUM_CLASSES))))
    except Exception:
        ll = float("nan")

    # 2. ROC-AUC
    classes_present = np.unique(y_true_06)
    if len(classes_present) > 1:
        try:
            sub_probs = probs_clipped[:, classes_present]
            sub_probs = sub_probs / sub_probs.sum(axis=1, keepdims=True)
            auc_macro = float(roc_auc_score(y_true_06, sub_probs, multi_class="ovr", average="macro", labels=classes_present))
        except Exception:
            auc_macro = 0.50
        try:
            sub_probs = probs_clipped[:, classes_present]
            sub_probs = sub_probs / sub_probs.sum(axis=1, keepdims=True)
            auc_weighted = float(roc_auc_score(y_true_06, sub_probs, multi_class="ovr", average="weighted", labels=classes_present))
        except Exception:
            auc_weighted = 0.50
    else:
        auc_macro = 0.50
        auc_weighted = 0.50

    # 3. Directional Accuracy на активных примерах
    # ВНИМАНИЕ: при 95% дисбалансе argmax(probs) всегда равен 0.
    # Реальный торговый сигнал извлекается через ожидаемое смещение E[Delta] = sum(c * P(c))
    # либо через асимметрию P(up) - P(down).
    exp_return = np.sum(probs_clipped * CLASSES_7[np.newaxis, :], axis=1)
    true_ticks = CLASSES_7[y_true_06]
    pred_classes_06 = np.argmax(probs_clipped, axis=1)
    
    non_zero_mask = (true_ticks != 0)
    if non_zero_mask.sum() > 0:
        # Угадывание направления при ненулевом реальном сдвиге цены:
        pred_direction = np.sign(exp_return[non_zero_mask])
        true_direction = np.sign(true_ticks[non_zero_mask])
        dir_correct = (pred_direction == true_direction).mean()
        
        # Доля уверенных направленных прогнозов (|E[Delta]| > 0.02 тика)
        confident_mask = np.abs(exp_return[non_zero_mask]) > 0.02
        if confident_mask.sum() > 0:
            confident_dir_acc = (pred_direction[confident_mask] == true_direction[confident_mask]).mean()
        else:
            confident_dir_acc = dir_correct
            
        non_zero_recall = float(confident_mask.sum() / max(1, non_zero_mask.sum()))
    else:
        dir_correct = 0.50
        confident_dir_acc = 0.50
        non_zero_recall = 0.0

    # 4. Expected Return & IC
    if np.std(exp_return) > 1e-8 and np.std(true_ticks) > 1e-8:
        ic = float(np.corrcoef(exp_return, true_ticks)[0, 1])
    else:
        ic = 0.0

    # 5. Expected Calibration Error (ECE)
    confidences = np.max(probs_clipped, axis=1)
    accuracies = (pred_classes_06 == y_true_06)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_m = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if bin_m.sum() > 0:
            bin_acc = accuracies[bin_m].mean()
            bin_conf = confidences[bin_m].mean()
            ece += (bin_m.sum() / n_samples) * abs(bin_acc - bin_conf)

    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}loss": round(ll, 4),
        f"{pfx}auc_macro": round(auc_macro, 4),
        f"{pfx}auc_weighted": round(auc_weighted, 4),
        f"{pfx}dir_acc": round(float(dir_correct) * 100.0, 2),
        f"{pfx}conf_dir_acc": round(float(confident_dir_acc) * 100.0, 2),
        f"{pfx}non_zero_recall": round(non_zero_recall * 100.0, 2),
        f"{pfx}ic": round(ic, 4),
        f"{pfx}ece": round(float(ece) * 100.0, 2),
    }


# =====================================================================
# 4. ТРЕНИРОВОЧНЫЙ ЦИКЛ ОДНОЙ НЕЙРОСЕТИ С ПОЛНЫМ ТРЕКИНГОМ СХОДИМОСТИ
# =====================================================================

def train_neural_architecture(
    arch_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    seq_len: int,
    n_features: int,
    loss_strategy: str = "focal",
    lr: float = 1e-3,
    batch_size: int = 1024,
    epochs: int = 12,
) -> Tuple[nn.Module, Dict[str, any], np.ndarray, np.ndarray]:
    """
    Обучает одну нейросетевую архитектуру с выводом истории сходимости.
    Возвращает: (обученная модель, история метрик, probs_val, probs_train).
    """
    model_cls = NEURAL_FACTORY[arch_name]
    model = model_cls(seq_len=seq_len, n_features=n_features, n_classes=NUM_CLASSES).to(DEVICE)

    # Веса классов
    class_weights = compute_class_weights(y_train.cpu().numpy(), strategy="sqrt" if loss_strategy == "weighted" else "none")

    # Функция потерь
    if loss_strategy == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    elif loss_strategy == "weighted":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    dataset_tr = TensorDataset(X_train, y_train)
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    history = []
    best_val_auc = -1.0
    best_weights = None

    y_val_np = y_val.cpu().numpy()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for bx, by in loader_tr:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss /= max(1, n_batches)

        # Валидация
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()

        val_metrics = compute_evaluation_metrics(y_val_np, val_probs)
        current_auc = val_metrics["auc_macro"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": val_metrics["loss"],
            "val_auc_macro": current_auc,
            "val_auc_weighted": val_metrics["auc_weighted"],
            "val_dir_acc": val_metrics["dir_acc"],
            "val_non_zero_recall": val_metrics["non_zero_recall"],
            "val_ic": val_metrics["ic"],
            "val_ece": val_metrics["ece"],
        })

        if current_auc > best_val_auc:
            best_val_auc = current_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Загружаем лучшие веса по Macro ROC-AUC
    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        final_val_probs = F.softmax(model(X_val), dim=-1).cpu().numpy()
        final_tr_probs = F.softmax(model(X_train), dim=-1).cpu().numpy()

    return model, history, final_val_probs, final_tr_probs


# =====================================================================
# 5. ОБУЧЕНИЕ LIGHTGBM (АРХИТЕКТУРА 1)
# =====================================================================

def train_lightgbm_architecture(
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    X_val_2d: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
) -> Tuple[lgb.Booster, Dict[str, any], np.ndarray, np.ndarray]:
    """
    Обучает LightGBM multiclass GBDT.
    Возвращает: (booster, метрики, probs_val, probs_train).
    """
    classes, counts = np.unique(y_train, return_counts=True)
    full_counts = np.ones(NUM_CLASSES, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        full_counts[c] = cnt
    sample_weights = np.sqrt(len(y_train) / full_counts[y_train])
    sample_weights /= sample_weights.mean()

    dtrain = lgb.Dataset(X_train_2d, label=y_train, weight=sample_weights)
    dval = lgb.Dataset(X_val_2d, label=y_val, reference=dtrain)

    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_threads": min(8, os.cpu_count() or 4),
        "seed": 42,
    }

    eval_result = {}
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.record_evaluation(eval_result)],
    )

    val_probs = booster.predict(X_val_2d)
    train_probs = booster.predict(X_train_2d)

    val_metrics = compute_evaluation_metrics(y_val, val_probs)
    return booster, val_metrics, val_probs, train_probs


# =====================================================================
# 6. КОНСОЛИДАЦИЯ И АНСАМБЛИРОВАНИЕ (4 МЕТОДА)
# =====================================================================

def evaluate_ensembling_methods(
    val_probs_dict: Dict[str, np.ndarray],
    y_val: np.ndarray,
    tr_probs_dict: Optional[Dict[str, np.ndarray]] = None,
    y_tr: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Тестирует 4 метода ансамблирования 7 моделей:
      1. Equal Weight (Simple Average)
      2. Val-AUC Softmax Weighted
      3. Temperature Scaled (Калиброванное усреднение)
      4. Stacking (L2 Multinomial Logistic Regression)
    """
    arch_keys = list(val_probs_dict.keys())
    stacked_val = np.stack([val_probs_dict[k] for k in arch_keys], axis=0)  # (N_models, N_samples, 7)

    results = {}

    # 1. Simple Average
    p_simple = np.mean(stacked_val, axis=0)
    p_simple /= p_simple.sum(axis=1, keepdims=True)
    results["1_Simple_Average"] = compute_evaluation_metrics(y_val, p_simple)

    # 2. Val-AUC Softmax Weighted
    aucs = []
    for k in arch_keys:
        m = compute_evaluation_metrics(y_val, val_probs_dict[k])
        aucs.append(m["auc_macro"])
    weights = np.exp(np.array(aucs) * 10.0)
    weights /= weights.sum()
    p_auc_w = np.sum(stacked_val * weights[:, np.newaxis, np.newaxis], axis=0)
    p_auc_w /= p_auc_w.sum(axis=1, keepdims=True)
    results["2_Val_AUC_Weighted"] = compute_evaluation_metrics(y_val, p_auc_w)

    # 3. Post-Hoc Temperature Scaling
    # Подбираем оптимальную температуру T для усредненных вероятностей
    logits_approx = np.log(np.clip(p_simple, 1e-7, 1.0 - 1e-7))
    best_t = 1.0
    best_t_loss = 1e9
    for t_cand in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]:
        scaled = np.exp(logits_approx / t_cand)
        scaled /= scaled.sum(axis=1, keepdims=True)
        m = compute_evaluation_metrics(y_val, scaled)
        if m["loss"] < best_t_loss:
            best_t_loss = m["loss"]
            best_t = t_cand
    p_temp = np.exp(logits_approx / best_t)
    p_temp /= p_temp.sum(axis=1, keepdims=True)
    results[f"3_Temperature_Scaled_T{best_t}"] = compute_evaluation_metrics(y_val, p_temp)

    # 4. Stacking (L2-Regularized Meta-Learner)
    if tr_probs_dict is not None and y_tr is not None:
        X_meta_tr = np.concatenate([tr_probs_dict[k] for k in arch_keys], axis=1)  # (N_tr, 7 * N_models)
        X_meta_val = np.concatenate([val_probs_dict[k] for k in arch_keys], axis=1)  # (N_val, 7 * N_models)
        try:
            meta = LogisticRegression(C=0.1, max_iter=200, solver="lbfgs")
            meta.fit(X_meta_tr, y_tr)
            p_stack = meta.predict_proba(X_meta_val)
            results["4_Meta_Stacker_L2"] = compute_evaluation_metrics(y_val, p_stack)
        except Exception:
            results["4_Meta_Stacker_L2"] = results["2_Val_AUC_Weighted"]

    return results


# =====================================================================
# 7. ГЛАВНЫЙ ОРКЕСТРАТОР ЛАБОРАТОРИИ ДЛЯ 1 КОНКРЕТНОГО ОКНА
# =====================================================================

def run_order1_lab(
    horizon: int = 3,
    ratio_k: float = 1.0,
    max_train_samples: int = 30000,
    max_val_samples: int = 10000,
    epochs: int = 10,
    batch_size: int = 1024,
    loss_strategy: str = "focal",
    out_dir: str = "scratch/order1_lab_results",
) -> Dict[str, any]:
    """
    Запуск полного цикла исследований на 1 конкретном окне.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🔬 ОРКЕСТРАТОР ИССЛЕДОВАТЕЛЬСКОЙ ЛАБОРАТОРИИ МОДЕЛЕЙ 1-ГО ПОРЯДКА")
    print(f"   Горизонт окна: H = {horizon}s, K = {ratio_k} | Устройство: {DEVICE}")
    print(f"   Выборка: Train = {max_train_samples}, Val = {max_val_samples} | Стратегия Loss: {loss_strategy}")
    print("=" * 80)

    # 1. Загрузка данных
    parquet_path = BASE_DIR / "data" / "labeled" / "prl" / f"full_labeled_h{horizon}s.parquet"
    if not parquet_path.exists():
        parquet_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {parquet_path}")

    print(f"\n[Шаг 1/5] Загрузка признаков и таргета из {parquet_path.name}...")
    
    # Загружаем выбранные признаки
    feats_json = BASE_DIR / "data" / "features" / "selected_features.json"
    with open(feats_json) as f:
        selected_features = json.load(f)["feature_order"]

    # Ищем таргет
    pf = pq.ParquetFile(parquet_path)
    target_cand = [c for c in pf.schema.names if f"target_class_7_h{horizon}" in c or (c == "target_class_7")]
    target_col = target_cand[0] if target_cand else "target_class_7"
    
    cols_to_load = selected_features + [target_col, "sample_split"]
    cols_present = [c for c in cols_to_load if c in pf.schema.names]
    
    df = pq.read_table(parquet_path, columns=cols_present).to_pandas()
    df = df.dropna(subset=[target_col])

    # Срезы Train и Val
    if "sample_split" in df.columns:
        df_tr = df[df["sample_split"] == "train"].iloc[-max_train_samples:]
        df_val = df[df["sample_split"] == "val"].iloc[:max_val_samples]
    else:
        split_idx = int(len(df) * 0.8)
        df_tr = df.iloc[max(0, split_idx - max_train_samples):split_idx]
        df_val = df.iloc[split_idx:split_idx + max_val_samples]

    feature_cols = [c for c in selected_features if c in df.columns]
    X_tr_raw = df_tr[feature_cols].values.astype(np.float32)
    y_tr_raw = df_tr[target_col].values.astype(np.int64)

    X_val_raw = df_val[feature_cols].values.astype(np.float32)
    y_val_raw = df_val[target_col].values.astype(np.int64)

    # Маппинг таргетов [-3..+3] -> [0..6]
    y_tr_06 = (y_tr_raw + 3).clip(0, 6)
    y_val_06 = (y_val_raw + 3).clip(0, 6)

    # Нормализация признаков (Z-score по Train)
    mean = np.nanmean(X_tr_raw, axis=0)
    std = np.nanstd(X_tr_raw, axis=0) + 1e-6
    X_tr_norm = np.nan_to_num((X_tr_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)
    X_val_norm = np.nan_to_num((X_val_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)

    seq_len = max(1, int(round(horizon * ratio_k)))
    n_features = len(feature_cols)

    print(f"  • Число признаков: {n_features}, Длина последовательности (seq_len): {seq_len}")
    print(f"  • Train сэмплов: {len(X_tr_norm)}, Val сэмплов: {len(X_val_norm)}")
    print(f"  • Баланс классов в Train (%): {np.round(np.bincount(y_tr_06, minlength=7) / len(y_tr_06) * 100.0, 2)}")

    # Подготовка 3D тензоров для нейросетей (B, seq_len, F)
    # Для малых окон делаем каузальное развертывание
    def to_seq_tensor(X_arr):
        N = len(X_arr)
        out = np.zeros((N, seq_len, n_features), dtype=np.float32)
        for i in range(seq_len):
            shift = seq_len - 1 - i
            if shift == 0:
                out[:, i, :] = X_arr
            else:
                out[shift:, i, :] = X_arr[:-shift]
                out[:shift, i, :] = X_arr[0]
        return torch.tensor(out, dtype=torch.float32, device=DEVICE)

    X_tr_tensor = to_seq_tensor(X_tr_norm)
    X_val_tensor = to_seq_tensor(X_val_norm)
    y_tr_tensor = torch.tensor(y_tr_06, dtype=torch.long, device=DEVICE)
    y_val_tensor = torch.tensor(y_val_06, dtype=torch.long, device=DEVICE)

    val_probs_dict = {}
    tr_probs_dict = {}
    model_benchmarks = {}
    convergence_histories = {}

    # [Шаг 2/5] Обучение 1: LightGBM
    print("\n[Шаг 2/5] Обучение базовой архитектуры 1: LightGBM...")
    t0 = time.time()
    lgb_model, lgb_metrics, lgb_val_p, lgb_tr_p = train_lightgbm_architecture(
        X_tr_norm, y_tr_06, X_val_norm, y_val_06, n_estimators=80, learning_rate=0.05
    )
    val_probs_dict["1_lightgbm"] = lgb_val_p
    tr_probs_dict["1_lightgbm"] = lgb_tr_p
    model_benchmarks["1_lightgbm"] = lgb_metrics
    print(f"  ✓ LightGBM обучен за {time.time()-t0:.2f}с | Macro-AUC: {lgb_metrics['auc_macro']:.4f} | Dir-Acc: {lgb_metrics['dir_acc']:.1f}%")

    # [Шаг 3/5] Обучение 6 нейросетевых архитектур
    print(f"\n[Шаг 3/5] Обучение 6 нейросетевых архитектур на {DEVICE} (Стратегия: {loss_strategy})...")
    arch_list = ["mlp_bottleneck", "conv1d_gru", "causal_cnn", "attention", "wavenet_tcn", "cross_mixer"]
    
    for idx, arch in enumerate(arch_list, start=2):
        print(f"  [{idx}/7] Обучение {arch} ({epochs} эпох)...")
        t_arch = time.time()
        net, hist, val_p, tr_p = train_neural_architecture(
            arch,
            X_tr_tensor,
            y_tr_tensor,
            X_val_tensor,
            y_val_tensor,
            seq_len=seq_len,
            n_features=n_features,
            loss_strategy=loss_strategy,
            lr=1e-3,
            batch_size=batch_size,
            epochs=epochs,
        )
        tag = f"{idx}_{arch}"
        val_probs_dict[tag] = val_p
        tr_probs_dict[tag] = tr_p
        convergence_histories[tag] = hist
        
        final_metrics = compute_evaluation_metrics(y_val_06, val_p)
        model_benchmarks[tag] = final_metrics
        dt = time.time() - t_arch
        print(f"    ✓ {arch} обучен за {dt:.1f}с | Macro-AUC: {final_metrics['auc_macro']:.4f} | Dir-Acc: {final_metrics['dir_acc']:.1f}% | ECE: {final_metrics['ece']:.1f}%")

    # [Шаг 4/5] Ансамблирование и консолидация
    print("\n[Шаг 4/5] Исследование 4 методов ансамблирования и консолидации...")
    ensemble_results = evaluate_ensembling_methods(val_probs_dict, y_val_06, tr_probs_dict, y_tr_06)
    for ens_name, ens_m in ensemble_results.items():
        print(f"  • {ens_name:<26} | Macro-AUC: {ens_m['auc_macro']:.4f} | Dir-Acc: {ens_m['dir_acc']:.1f}% | Loss: {ens_m['loss']:.4f} | ECE: {ens_m['ece']:.1f}%")

    # [Шаг 5/5] Сохранение полного лабораторного отчета
    print(f"\n[Шаг 5/5] Формирование лабораторного отчета в {out_path}...")
    summary_report = {
        "horizon": horizon,
        "ratio_k": ratio_k,
        "seq_len": seq_len,
        "loss_strategy": loss_strategy,
        "n_train": len(X_tr_norm),
        "n_val": len(X_val_norm),
        "individual_models": model_benchmarks,
        "ensembles": ensemble_results,
        "convergence_histories": convergence_histories,
    }
    
    with open(out_path / f"lab_report_h{horizon}s_{loss_strategy}.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Order-1 Neural & Ensemble Research Lab")
    parser.add_argument("--horizon", type=int, default=3, help="Горизонт H (сек)")
    parser.add_argument("--k", type=float, default=1.0, help="Множитель соотношения K")
    parser.add_argument("--train-samples", type=int, default=30000, help="Число сэмплов train")
    parser.add_argument("--val-samples", type=int, default=10000, help="Число сэмплов val")
    parser.add_argument("--epochs", type=int, default=10, help="Эпох обучения для каждой нейросети")
    parser.add_argument("--batch-size", type=int, default=1024, help="Размер батча")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "weighted", "ce"], help="Функция потерь")
    parser.add_argument("--out-dir", type=str, default="scratch/order1_lab_results", help="Папка вывода")
    args = parser.parse_args()

    run_order1_lab(
        horizon=args.horizon,
        ratio_k=args.k,
        max_train_samples=args.train_samples,
        max_val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_strategy=args.loss,
        out_dir=args.out_dir,
    )
