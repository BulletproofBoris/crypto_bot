#!/usr/bin/env python3
"""
_tools/order1_multi_horizon_k_explorer.py
=============================================================================
Масштабное исследование оптимального соотношения K = L / H по всей сетке горизонтов.

Согласно пользовательскому регламенту:
  1. Короткие окна (H <= 25s): сетка K расширена до 10.0
     Кандидаты: K in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0}
     Горизонты: H in {1s, 2s, 3s, 5s, 8s, 12s, 25s}
  2. Длительные окна (H > 25s): сетка K ограничена до 5.0
     Кандидаты: K in {1.0, 2.0, 3.0, 4.0, 5.0}
     Горизонты: H in {60s, 90s, 120s}

Методология:
  - Единая выборка признаков из prl_1s_features.parquet (SafeTrade + BigONE + Macro).
  - Для каждой пары (H, K) разметка 7 классов через compute_single_horizon_labels.
  - Обучение репрезентативного триумвирата: LightGBM + Causal-CNN + Conv1D-GRU с Focal Loss.
  - Калиброванное ансамблирование (Temperature Scaling T=0.5).
  - Оценка: Macro ROC-AUC, Confident Directional Accuracy, Information Coefficient (IC), ECE.
  - Автоматическое сохранение результатов после каждого горизонта H.
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


def load_continuous_features(
    n_train: int,
    n_val: int,
    parquet_path: Path,
    feats_json: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Загружает непрерывный срез признаков для train и val."""
    with open(feats_json) as f:
        selected_features = json.load(f)["feature_order"]

    req_cols = [
        "ST_mid_price", "ST_spread", "ST_bid_1_price", "ST_ask_1_price",
        "ST_bid_1_qty", "ST_ask_1_qty", "ST_bids_vol_top5", "ST_asks_vol_top5", "ST_trade_vol"
    ]
    pf = pq.ParquetFile(parquet_path)
    all_needed = list(set(req_cols + selected_features))
    cols_present = [c for c in all_needed if c in pf.schema.names]

    total_rows = pf.metadata.num_rows
    # Берем максимальный запас под буферы и большие горизонты (120s * 10 = 1200с)
    needed_rows = n_train + n_val + 2000
    start_row = max(0, total_rows - needed_rows)

    df_full = pq.read_table(parquet_path, columns=cols_present).to_pandas().iloc[start_row:].reset_index(drop=True)
    df_full = df_full.dropna(subset=["ST_mid_price"]).reset_index(drop=True)

    feature_cols = [c for c in selected_features if c in df_full.columns]
    X_raw_all = df_full[feature_cols].values.astype(np.float32)

    tr_end = len(df_full) - n_val
    tr_start = max(0, tr_end - n_train)

    df_tr = df_full.iloc[tr_start:tr_end].reset_index(drop=True)
    df_val = df_full.iloc[tr_end:tr_end + n_val].reset_index(drop=True)

    X_tr_raw = X_raw_all[tr_start:tr_end]
    X_val_raw = X_raw_all[tr_end:tr_end + n_val]

    # Z-score нормализация
    mean = np.nanmean(X_tr_raw, axis=0)
    std = np.nanstd(X_tr_raw, axis=0) + 1e-6
    X_tr_norm = np.nan_to_num((X_tr_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)
    X_val_norm = np.nan_to_num((X_val_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)

    return df_tr, df_val, X_tr_norm, X_val_norm, feature_cols


def evaluate_single_h_k(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    X_tr_norm: np.ndarray,
    X_val_norm: np.ndarray,
    H: int,
    K: float,
    epochs: int = 5,
    batch_size: int = 1024,
) -> Dict[str, any]:
    """Оценивает производительность моделей и ансамбля для конкретной пары (H, K)."""
    L = max(1, int(round(H * K)))
    n_features = X_tr_norm.shape[1]

    # 1. Разметка
    lbl_tr = compute_single_horizon_labels(df_tr, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)
    lbl_val = compute_single_horizon_labels(df_val, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)

    y_tr_06 = (lbl_tr["class_7"] + 3).clip(0, 6).astype(np.int64)
    y_val_06 = (lbl_val["class_7"] + 3).clip(0, 6).astype(np.int64)

    zero_pct = float((y_tr_06 == 3).mean() * 100.0)

    # 2. Формирование последовательностей (N, L, F)
    torch.cuda.empty_cache()
    def to_seq(X_arr, seq_len):
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

    X_tr_seq = to_seq(X_tr_norm, L)
    X_val_seq = to_seq(X_val_norm, L)
    y_tr_t = torch.tensor(y_tr_06, dtype=torch.long, device=DEVICE)

    # 3. LightGBM
    lgb_mod, lgb_m, lgb_val_p, _ = train_lightgbm_architecture(
        X_tr_norm, y_tr_06, X_val_norm, y_val_06, n_estimators=60, learning_rate=0.05
    )

    # 4. Нейросети
    def train_net(net_cls):
        net = net_cls(seq_len=L, n_features=n_features, n_classes=NUM_CLASSES).to(DEVICE)
        weights = compute_class_weights(y_tr_06, strategy="sqrt")
        crit = FocalLoss(alpha=weights, gamma=2.0)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ds = TensorDataset(X_tr_seq, y_tr_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            net.train()
            for bx, by in loader:
                opt.zero_grad()
                crit(net(bx), by).backward()
                opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            vp = F.softmax(net(X_val_seq), dim=-1).cpu().numpy()
        return compute_evaluation_metrics(y_val_06, vp), vp

    causal_m, causal_p = train_net(CausalCNNNet)
    gru_m, gru_p = train_net(Conv1DGRUNet)

    # 5. Калиброванный ансамбль (Temperature Scaling T=0.5)
    stacked = np.stack([lgb_val_p, causal_p, gru_p], axis=0)
    p_avg = np.mean(stacked, axis=0)
    logits = np.log(np.clip(p_avg, 1e-7, 1.0 - 1e-7))
    p_scaled = np.exp(logits / 0.5)
    p_scaled /= p_scaled.sum(axis=1, keepdims=True)
    ens_m = compute_evaluation_metrics(y_val_06, p_scaled)

    # Комплексный скор оптимальности
    score = (
        ens_m["ic"] * 100.0
        + (ens_m["auc_macro"] - 0.5) * 50.0
        + (ens_m["conf_dir_acc"] - 50.0) * 0.5
        - ens_m["ece"]
    )

    return {
        "H": H,
        "K": K,
        "L": L,
        "zero_class_pct": round(zero_pct, 2),
        "score": round(score, 2),
        "lgb": lgb_m,
        "causal_cnn": causal_m,
        "conv1d_gru": gru_m,
        "ensemble": ens_m,
    }


def run_multi_horizon_exploration(
    horizons: List[int],
    n_train: int = 25000,
    n_val: int = 8000,
    epochs: int = 5,
    out_dir: str = "scratch/multi_horizon_k_results",
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_path = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
    feats_json = BASE_DIR / "data" / "features" / "selected_features.json"

    print("=" * 90)
    print("🌍 МАСШТАБНОЕ ИССЛЕДОВАНИЕ ОПТИМАЛЬНОГО K ПО ВСЕМ ОКНАМ")
    print(f"   Горизонты: {horizons} | Train = {n_train}, Val = {n_val} | Устройство: {DEVICE}")
    print("=" * 90)

    print("\nЗагрузка признаков из базы...")
    df_tr, df_val, X_tr_norm, X_val_norm, feature_cols = load_continuous_features(
        n_train, n_val, parquet_path, feats_json
    )
    print(f"Признаки загружены: {len(feature_cols)} колонок, Train = {len(df_tr)}, Val = {len(df_val)}")

    all_horizon_results = {}
    optimal_k_map = {}

    for H in horizons:
        # Регламент: для H <= 25s сетка K расширена до 10.0; для H > 25s ограничена 5.0
        if H <= 25:
            k_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        else:
            k_grid = [1.0, 2.0, 3.0, 4.0, 5.0]

        print(f"\n{'='*40} ГОРИЗОНТ H = {H}s {'='*40}")
        print(f"Сетка K: {k_grid}")

        h_results = {}
        best_k = None
        best_score = -1e9

        for K in k_grid:
            t0 = time.time()
            res = evaluate_single_h_k(df_tr, df_val, X_tr_norm, X_val_norm, H, K, epochs=epochs)
            dt = time.time() - t0
            em = res["ensemble"]
            ic_str = f"{em['ic']:+.4f}"
            print(
                f"  • K={K:<4.1f} (L={res['L']:<3}s) | Macro-AUC: {em['auc_macro']:.4f} "
                f"| Conf-Dir: {em['conf_dir_acc']:<5.1f}% | IC: {ic_str:<8} | ECE: {em['ece']:<4.2f}% "
                f"| Score: {res['score']:<6.1f} ({dt:.1f}с)"
            )
            h_results[f"K_{K:.1f}"] = res
            if res["score"] > best_score:
                best_score = res["score"]
                best_k = K

        optimal_k_map[f"H_{H}s"] = {
            "optimal_K": best_k,
            "optimal_L": int(round(H * best_k)),
            "best_score": best_score,
            "best_auc": h_results[f"K_{best_k:.1f}"]["ensemble"]["auc_macro"],
            "best_conf_dir": h_results[f"K_{best_k:.1f}"]["ensemble"]["conf_dir_acc"],
            "best_ic": h_results[f"K_{best_k:.1f}"]["ensemble"]["ic"],
        }
        all_horizon_results[f"H_{H}s"] = h_results

        print(f"\n👉 Оптимум для H = {H}s: K* = {best_k:.1f} (L* = {int(round(H * best_k))}s предыстории)")

        # Промежуточное сохранение
        with open(out_path / "multi_horizon_k_optimal_map.json", "w") as f:
            json.dump(optimal_k_map, f, indent=2)
        with open(out_path / f"horizon_{H}s_detailed.json", "w") as f:
            json.dump(h_results, f, indent=2)

    # Финальная сводка
    print("\n" + "=" * 90)
    print("🏆 ИТОГОВАЯ КАРТА ОПТИМАЛЬНЫХ K* ПО ВСЕМ ГОРИЗОНТАМ:")
    print("=" * 90)
    print(f"{'Горизонт H':<12} {'Оптимальный K*':<16} {'История L*':<14} {'Macro-AUC':<12} {'Conf-Dir (%)':<14} {'IC':<10}")
    print("-" * 90)
    for h_tag, opt in optimal_k_map.items():
        h_str = h_tag.replace("H_", "").replace("s", " сек")
        ic_str = f"{opt['best_ic']:+.4f}"
        print(f"{h_str:<12} {opt['optimal_K']:<16.1f} {opt['optimal_L']:<4} сек      {opt['best_auc']:<12.4f} {opt['best_conf_dir']:<14.1f} {ic_str:<10}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Horizon K Explorer")
    parser.add_argument("--horizons", type=str, default="1,2,3,5", help="Список горизонтов H через запятую")
    parser.add_argument("--train-samples", type=int, default=25000, help="Train сэмплов")
    parser.add_argument("--val-samples", type=int, default=8000, help="Val сэмплов")
    parser.add_argument("--epochs", type=int, default=5, help="Эпох на модель")
    args = parser.parse_args()

    h_list = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    run_multi_horizon_exploration(
        horizons=h_list,
        n_train=args.train_samples,
        n_val=args.val_samples,
        epochs=args.epochs,
    )
