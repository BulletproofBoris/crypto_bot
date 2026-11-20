#!/usr/bin/env python3
"""
_tools/order1_search_optimal_k.py
=============================================================================
Исследование оптимального соотношения K = L / H для 1-секундного горизонта (H=1s).

Для H = 1s длина входной ретроспективы L = round(H * K) определяет:
  - Сколько секунд предыстории (seq_len) видит модель:
      K=1 -> L=1s (только текущий моментальный срез t0)
      K=2 -> L=2s (t-1, t0)
      K=3 -> L=3s (t-2, t-1, t0 — микро-волна)
      K=4 -> L=4s
      K=5 -> L=5s (t-4..t0 — полный цикл OFI / стакана)
      K=8 -> L=8s
  - Масштаб локальной волатильности и порогов разметки импульса z.

Методология:
  1. Нарезка чистой неразрывной выборки 25k Train / 8k Val из prl_1s_features.parquet.
  2. Разметка таргета 7 классов для каждого K.
  3. Обучение репрезентативных моделей:
     - LightGBM (GBDT)
     - Causal-CNN (Dilated TCN)
     - Conv1D + GRU (Recurrent-Convolutional)
     с Focal Loss (gamma=2.0) на RTX 5060 Ti CUDA.
  4. Сравнение метрик: Macro ROC-AUC, Directional Accuracy, Information Coefficient (IC).
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.data_labeler import compute_single_horizon_labels
from _tools.order1_research_lab import (
    CausalCNNNet,
    Conv1DGRUNet,
    FocalLoss,
    compute_class_weights,
    compute_evaluation_metrics,
    train_lightgbm_architecture,
    DEVICE,
    CLASSES_7,
    NUM_CLASSES,
)


def run_k_search(
    k_list: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0],
    n_train: int = 25000,
    n_val: int = 8000,
    epochs: int = 6,
    batch_size: int = 1024,
    out_dir: str = "scratch/order1_k_search_h1s",
) -> Dict[str, any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(f"🔍 ИССЛЕДОВАНИЕ ОПТИМАЛЬНОГО СООТНОШЕНИЯ K ДЛЯ ГОРИЗОНТА H = 1s")
    print(f"   Сетка кандидатов K: {k_list} | Train = {n_train}, Val = {n_val} | Устройство: {DEVICE}")
    print("=" * 85)

    # 1. Загрузка признаков
    feats_json = BASE_DIR / "data" / "features" / "selected_features.json"
    with open(feats_json) as f:
        selected_features = json.load(f)["feature_order"]

    parquet_path = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
    if not parquet_path.exists():
        parquet_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"

    print(f"\n[1/3] Загрузка данных из {parquet_path.name}...")
    req_cols = [
        "ST_mid_price", "ST_spread", "ST_bid_1_price", "ST_ask_1_price",
        "ST_bid_1_qty", "ST_ask_1_qty", "ST_bids_vol_top5", "ST_asks_vol_top5", "ST_trade_vol"
    ]
    pf = pq.ParquetFile(parquet_path)
    all_needed = list(set(req_cols + selected_features))
    cols_present = [c for c in all_needed if c in pf.schema.names]

    # Берем непрерывный активный хвост
    total_rows = pf.metadata.num_rows
    needed_rows = n_train + n_val + 500  # запас на purge/embargo
    start_row = max(0, total_rows - needed_rows)

    df_full = pq.read_table(parquet_path, columns=cols_present).to_pandas().iloc[start_row:].reset_index(drop=True)
    df_full = df_full.dropna(subset=["ST_mid_price"]).reset_index(drop=True)
    print(f"  • Загружено строк: {len(df_full):,}, число фичей: {len(selected_features)}")

    feature_cols = [c for c in selected_features if c in df_full.columns]
    X_raw_all = df_full[feature_cols].values.astype(np.float32)

    # Индексы train и val
    tr_end = len(df_full) - n_val
    tr_start = max(0, tr_end - n_train)

    df_tr = df_full.iloc[tr_start:tr_end].reset_index(drop=True)
    df_val = df_full.iloc[tr_end:tr_end + n_val].reset_index(drop=True)

    X_tr_raw = X_raw_all[tr_start:tr_end]
    X_val_raw = X_raw_all[tr_end:tr_end + n_val]

    # Нормализация по train
    mean = np.nanmean(X_tr_raw, axis=0)
    std = np.nanstd(X_tr_raw, axis=0) + 1e-6
    X_tr_norm = np.nan_to_num((X_tr_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)
    X_val_norm = np.nan_to_num((X_val_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)

    n_features = len(feature_cols)

    k_results = {}

    print(f"\n[2/3] Тестирование кандидатов K по сетке {k_list}...")
    for k_val in k_list:
        t0 = time.time()
        L = max(1, int(round(1.0 * k_val)))
        print(f"\n--- 🧪 Тест K = {k_val:.1f} (L = {L}s предыстории, seq_len = {L}) ---")

        # 1. Разметка для текущего K
        lbl_tr = compute_single_horizon_labels(df_tr, "prl", H=1, K=k_val, min_vol=50.0, tick_size=0.005)
        lbl_val = compute_single_horizon_labels(df_val, "prl", H=1, K=k_val, min_vol=50.0, tick_size=0.005)

        y_tr_06 = (lbl_tr["class_7"] + 3).clip(0, 6).astype(np.int64)
        y_val_06 = (lbl_val["class_7"] + 3).clip(0, 6).astype(np.int64)

        binc = np.bincount(y_tr_06, minlength=7)
        zero_pct = binc[3] / len(y_tr_06) * 100.0
        active_pct = 100.0 - zero_pct
        print(f"  • Распределение классов Train: 0-класс = {zero_pct:.2f}%, активные движения = {active_pct:.2f}%")

        # 2. Формирование 3D тензоров (N, seq_len, F)
        def to_seq_tensor(X_arr, seq_len):
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

        X_tr_seq = to_seq_tensor(X_tr_norm, L)
        X_val_seq = to_seq_tensor(X_val_norm, L)
        y_tr_t = torch.tensor(y_tr_06, dtype=torch.long, device=DEVICE)
        y_val_t = torch.tensor(y_val_06, dtype=torch.long, device=DEVICE)

        # 3. Обучение Модели A: LightGBM (на текущем срезе t0)
        lgb_mod, lgb_m, lgb_val_p, lgb_tr_p = train_lightgbm_architecture(
            X_tr_norm, y_tr_06, X_val_norm, y_val_06, n_estimators=60, learning_rate=0.05
        )

        # 4. Обучение Модели B: Causal-CNN (на временной последовательности L)
        def train_quick_net(net_cls):
            net = net_cls(seq_len=L, n_features=n_features, n_classes=NUM_CLASSES).to(DEVICE)
            weights = compute_class_weights(y_tr_06, strategy="sqrt")
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            ds = TensorDataset(X_tr_seq, y_tr_t)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ep in range(epochs):
                net.train()
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = net(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            net.eval()
            with torch.no_grad():
                val_p = F.softmax(net(X_val_seq), dim=-1).cpu().numpy()
            return net, compute_evaluation_metrics(y_val_06, val_p), val_p

        causal_net, causal_m, causal_val_p = train_quick_net(CausalCNNNet)
        gru_net, gru_m, gru_val_p = train_quick_net(Conv1DGRUNet)

        # 5. Ансамбль (LightGBM + Causal-CNN + Conv1D-GRU) с Temperature Scaling
        stacked = np.stack([lgb_val_p, causal_val_p, gru_val_p], axis=0)
        p_avg = np.mean(stacked, axis=0)
        logits_approx = np.log(np.clip(p_avg, 1e-7, 1.0 - 1e-7))
        scaled_p = np.exp(logits_approx / 0.5)
        scaled_p /= scaled_p.sum(axis=1, keepdims=True)
        ens_m = compute_evaluation_metrics(y_val_06, scaled_p)

        dt = time.time() - t0
        print(f"  ✓ LightGBM    | Macro-AUC: {lgb_m['auc_macro']:.4f} | Dir-Acc: {lgb_m['dir_acc']:.1f}% | IC: {lgb_m['ic']:+.4f}")
        print(f"  ✓ Causal-CNN  | Macro-AUC: {causal_m['auc_macro']:.4f} | Dir-Acc: {causal_m['dir_acc']:.1f}% | IC: {causal_m['ic']:+.4f}")
        print(f"  ✓ Conv1D-GRU  | Macro-AUC: {gru_m['auc_macro']:.4f} | Dir-Acc: {gru_m['dir_acc']:.1f}% | IC: {gru_m['ic']:+.4f}")
        print(f"  🏆 АНСАМБЛЬ   | Macro-AUC: {ens_m['auc_macro']:.4f} | Conf-Dir: {ens_m['conf_dir_acc']:.1f}% | IC: {ens_m['ic']:+.4f} | ECE: {ens_m['ece']:.2f}% ({dt:.1f}с)")

        k_results[f"K_{k_val:.1f}"] = {
            "K": k_val,
            "seq_len": L,
            "zero_class_pct": round(zero_pct, 2),
            "lgb": lgb_m,
            "causal_cnn": causal_m,
            "conv1d_gru": gru_m,
            "ensemble": ens_m,
            "elapsed_sec": round(dt, 2),
        }

    # [3/3] Сводный отчет
    print("\n" + "=" * 85)
    print("📊 СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ K ДЛЯ ГОРИЗОНТА H = 1s:")
    print("=" * 85)
    print(f"{'K':<6} {'L (seq)':<8} {'Macro-AUC':<12} {'Dir-Acc (%)':<14} {'Conf-Dir (%)':<14} {'IC (Альфа)':<12} {'ECE (%)':<10}")
    print("-" * 85)

    best_k = None
    best_score = -1e9
    for k_tag, res in k_results.items():
        em = res["ensemble"]
        k_val = res["K"]
        L_val = res["seq_len"]
        # Комплексный скор: приоритет IC и Macro-AUC с штрафом за ECE
        score = em["ic"] * 100.0 + (em["auc_macro"] - 0.5) * 50.0 + (em["conf_dir_acc"] - 50.0) * 0.5 - em["ece"]
        ic_str = f"{em['ic']:+.4f}"
        print(f"{k_val:<6.1f} {L_val:<8} {em['auc_macro']:<12.4f} {em['dir_acc']:<14.1f} {em['conf_dir_acc']:<14.1f} {ic_str:<12} {em['ece']:<10.2f}")
        if score > best_score:
            best_score = score
            best_k = k_val

    print("-" * 85)
    print(f"👉 РЕКОМЕНДУЕМЫЙ ОПТИМАЛЬНЫЙ K* ДЛЯ H=1s: K = {best_k:.1f} (L = {int(round(best_k * 1))}s предыстории)")
    print("=" * 85)

    summary_file = out_path / "k_search_h1s_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"best_k": best_k, "results": k_results}, f, indent=2)

    return {"best_k": best_k, "results": k_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimal K search for H=1s")
    parser.add_argument("--k-values", type=str, default="1.0,2.0,3.0,4.0,5.0,8.0", help="Список K через запятую")
    parser.add_argument("--train-samples", type=int, default=25000, help="Train сэмплов")
    parser.add_argument("--val-samples", type=int, default=8000, help="Val сэмплов")
    parser.add_argument("--epochs", type=int, default=6, help="Эпох на нейросеть")
    args = parser.parse_args()

    k_list = [float(x.strip()) for x in args.k_values.split(",") if x.strip()]
    run_k_search(
        k_list=k_list,
        n_train=args.train_samples,
        n_val=args.val_samples,
        epochs=args.epochs,
    )
