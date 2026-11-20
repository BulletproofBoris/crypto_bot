#!/usr/bin/env python3
"""
_tools/rigorous_sample_size_study.py
=============================================================================
Фундаментальное исследование объема обучающей выборки (N_train) и обобщающей
способности моделей (Generalization & Concept Drift).

Методология:
  1. Физическая шкала времени:
     - 12 часов  ( 43 200 сэмплов)
     - 24 часа   ( 86 400 сэмплов, 1 сутки)
     - 48 часов  (172 800 сэмплов, 2 суток)
     - 72 часа   (259 200 сэмплов, 3 суток)
     - 120 часов (432 000 сэмплов, 5 суток)
  2. 3-Fold Walk-Forward Cross-Validation на разных режимах рынка (каждый fold = 20k OOS).
  3. Multi-Seed Averaging для нейросети (Seeds: 42, 100, 777) с фиксацией дисперсии.
  4. Эталонное обучение детерминированного LightGBM + гибридный консенсус.
  5. Опорные горизонты: H in {1s, 5s, 20s, 80s}.
  6. Аппаратное ускорение: RTX 5060 Ti + torch.amp.autocast('cuda', dtype=torch.bfloat16).
=============================================================================
"""

import json
import os
import random
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
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.data_labeler import compute_single_horizon_labels
from _tools.order1_research_lab import (
    Conv1DGRUNet,
    FocalLoss,
    compute_class_weights,
    compute_evaluation_metrics,
    train_lightgbm_architecture,
    DEVICE,
    NUM_CLASSES,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FastSequenceDataset(Dataset):
    def __init__(self, X_arr: np.ndarray, y_arr: np.ndarray, seq_len: int):
        self.X = torch.from_numpy(X_arr).float()
        self.y = torch.from_numpy(y_arr).long()
        self.seq_len = seq_len
        self.N = len(y_arr)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if idx >= self.seq_len - 1:
            seq = self.X[idx - self.seq_len + 1 : idx + 1]
        else:
            pad_len = self.seq_len - 1 - idx
            pad = self.X[0:1].repeat(pad_len, 1)
            seq = torch.cat([pad, self.X[0 : idx + 1]], dim=0)
        return seq, self.y[idx]


def train_eval_conv1d_gru(
    X_tr_norm: np.ndarray,
    y_tr_06: np.ndarray,
    X_val_norm: np.ndarray,
    y_val_06: np.ndarray,
    L: int,
    n_features: int,
    seed: int,
    epochs: int = 8,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    set_seed(seed)
    ds_tr = FastSequenceDataset(X_tr_norm, y_tr_06, L)
    ds_val = FastSequenceDataset(X_val_norm, y_val_06, L)
    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = Conv1DGRUNet(seq_len=L, n_features=n_features, n_classes=NUM_CLASSES).to(DEVICE)
    weights = compute_class_weights(y_tr_06, strategy="sqrt")
    criterion = FocalLoss(alpha=weights, gamma=2.0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(bx)
                loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    dt = time.time() - t0

    model.eval()
    val_probs_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(bx)
                p = F.softmax(logits, dim=-1)
            val_probs_list.append(p.float().cpu().numpy())

    val_probs = np.concatenate(val_probs_list, axis=0)
    metrics = compute_evaluation_metrics(y_val_06, val_probs)

    del model, train_loader, val_loader, ds_tr, ds_val, optimizer
    torch.cuda.empty_cache()
    return val_probs, metrics, dt


def main():
    print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    parquet_path = Path("data/features/prl_1s_features.parquet")
    feats_json = Path("data/features/selected_features.json")
    k_cache_file = Path("scratch/variant_a_pipeline_results/phase1_optimal_k.json")
    out_json = Path("scratch/variant_a_pipeline_results/rigorous_sample_size_study.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(k_cache_file) as f:
        optimal_k_map = {int(k): float(v) for k, v in json.load(f).items()}

    with open(feats_json) as f:
        selected_features = json.load(f)["feature_order"]

    req_cols = [
        "ST_mid_price", "ST_spread", "ST_bid_1_price", "ST_ask_1_price",
        "ST_bid_1_qty", "ST_ask_1_qty", "ST_bids_vol_top5", "ST_asks_vol_top5", "ST_trade_vol"
    ]
    pf = pq.ParquetFile(parquet_path)
    all_needed = list(set(req_cols + selected_features))
    cols_present = [c for c in all_needed if c in pf.schema.names]

    # Сетка физического времени
    time_windows = [
        {"name": "12h", "n_samples": 43_200, "hours": 12.0},
        {"name": "24h", "n_samples": 86_400, "hours": 24.0},
        {"name": "48h", "n_samples": 172_800, "hours": 48.0},
        {"name": "72h", "n_samples": 259_200, "hours": 72.0},
        {"name": "120h", "n_samples": 432_000, "hours": 120.0},
    ]

    # 4 репрезентативных горизонта
    horizons = [1, 5, 20, 80]
    seeds = [42, 100, 777]
    val_size = 20_000

    # Загружаем необходимый объем из parquet для обеспечения 3 фолдов
    # Fold C: конец файла
    # Fold B: отступ назад на 150 000 строк
    # Fold A: отступ назад на 300 000 строк
    # Максимальный трейн 432 000 + 300 000 + буфер = 800 000 строк
    needed_total = 800_000
    total_rows = pf.metadata.num_rows
    start_row = max(0, total_rows - needed_total)

    print(f"Loading continuous slice: {needed_total:,} rows from parquet...")
    df_raw = pq.read_table(parquet_path, columns=cols_present).to_pandas().iloc[start_row:].reset_index(drop=True)
    df_raw = df_raw.dropna(subset=["ST_mid_price"]).reset_index(drop=True)
    feature_cols = [c for c in selected_features if c in df_raw.columns]
    X_raw_all = df_raw[feature_cols].values.astype(np.float32)

    # Z-Score нормализация
    mean = np.nanmean(X_raw_all, axis=0)
    std = np.nanstd(X_raw_all, axis=0) + 1e-6
    X_norm_all = np.nan_to_num((X_raw_all - mean) / std, nan=0.0).clip(-5.0, 5.0)

    total_len = len(df_raw)
    
    # Конфигурация 3 фолдов (Walk-Forward OOS)
    folds = [
        {"name": "Fold_A (Early Regime)", "val_end": total_len - 300_000},
        {"name": "Fold_B (Mid Regime)",   "val_end": total_len - 150_000},
        {"name": "Fold_C (Recent Regime)","val_end": total_len},
    ]

    all_study_results = {}
    if out_json.exists():
        try:
            with open(out_json) as f:
                all_study_results = json.load(f)
        except Exception:
            all_study_results = {}

    print("\n" + "=" * 90)
    print("🔬 ЗАПУСК МАСШТАБНОГО ИССЛЕДОВАНИЯ N_train: 4 ГОРИЗОНТА x 5 ОКОН x 3 ФОЛДА x 3 СИДА")
    print("=" * 90)

    for H in horizons:
        K = optimal_k_map[H]
        L = max(1, int(round(H * K)))
        n_features = len(feature_cols)

        h_key = f"H{H}s"
        if h_key not in all_study_results:
            all_study_results[h_key] = {}

        print(f"\n{'#'*80}")
        print(f"### ГОРИЗОНТ H={H}s (K*={K}, L*={L}s) ###")
        print(f"{'#'*80}")

        for win in time_windows:
            w_name = win["name"]
            n_tr = win["n_samples"]
            w_hours = win["hours"]

            print(f"\n--- ⏳ Окно выборки: {w_name} ({n_tr:,} сэмплов | {w_hours}ч) ---")

            fold_ics_nn = []
            fold_dirs_nn = []
            fold_ics_lgb = []
            fold_ics_ens = []

            for fold in folds:
                f_name = fold["name"]
                val_end = fold["val_end"]
                val_start = val_end - val_size
                tr_end = val_start
                tr_start = tr_end - n_tr

                if tr_start < 0:
                    print(f"  [{f_name}] Пропуск: недостаточно истории для tr_start={tr_start}")
                    continue

                df_tr = df_raw.iloc[tr_start:tr_end].reset_index(drop=True)
                X_tr = X_norm_all[tr_start:tr_end]

                df_val = df_raw.iloc[val_start:val_end].reset_index(drop=True)
                X_val = X_norm_all[val_start:val_end]

                # Разметка
                lbl_tr = compute_single_horizon_labels(df_tr, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)
                y_tr_06 = (lbl_tr["class_7"] + 3).clip(0, 6).astype(np.int64)

                lbl_val = compute_single_horizon_labels(df_val, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)
                y_val_06 = (lbl_val["class_7"] + 3).clip(0, 6).astype(np.int64)

                # 1. Детерминированный LightGBM (быстрый срез)
                _, lgb_m, lgb_p, _ = train_lightgbm_architecture(
                    X_tr, y_tr_06, X_val, y_val_06, n_estimators=60, learning_rate=0.05
                )
                fold_ics_lgb.append(lgb_m["ic"])

                # 2. Нейросеть с 3 сидами
                nn_probs_seeds = []
                seed_ics = []
                seed_dirs = []
                for s in seeds:
                    val_p, m, dt = train_eval_conv1d_gru(
                        X_tr, y_tr_06, X_val, y_val_06, L=L, n_features=n_features, seed=s, epochs=8
                    )
                    nn_probs_seeds.append(val_p)
                    seed_ics.append(m["ic"])
                    seed_dirs.append(m["conf_dir_acc"])

                # Усредненное предсказание нейросети
                nn_p_avg = np.mean(nn_probs_seeds, axis=0)
                nn_avg_metrics = compute_evaluation_metrics(y_val_06, nn_p_avg)
                fold_ics_nn.append(nn_avg_metrics["ic"])
                fold_dirs_nn.append(nn_avg_metrics["conf_dir_acc"])

                # 3. Гибридный ансамбль (50% LightGBM + 50% NN)
                ens_p = 0.5 * lgb_p + 0.5 * nn_p_avg
                ens_p /= ens_p.sum(axis=1, keepdims=True)
                ens_metrics = compute_evaluation_metrics(y_val_06, ens_p)
                fold_ics_ens.append(ens_metrics["ic"])

                print(f"  • {f_name:<24} | NN_IC(avg): {nn_avg_metrics['ic']:+.4f} (std={np.std(seed_ics):.4f}) | "
                      f"LGB_IC: {lgb_m['ic']:+.4f} | ENS_IC: {ens_metrics['ic']:+.4f}")

            # Итоговая статистика по фолдам
            mean_ic_nn = float(np.mean(fold_ics_nn)) if fold_ics_nn else 0.0
            std_ic_nn = float(np.std(fold_ics_nn)) if fold_ics_nn else 0.0
            mean_dir_nn = float(np.mean(fold_dirs_nn)) if fold_dirs_nn else 0.0
            mean_ic_lgb = float(np.mean(fold_ics_lgb)) if fold_ics_lgb else 0.0
            mean_ic_ens = float(np.mean(fold_ics_ens)) if fold_ics_ens else 0.0

            res_entry = {
                "n_samples": n_tr,
                "hours": w_hours,
                "mean_ic_nn": round(mean_ic_nn, 4),
                "std_ic_nn": round(std_ic_nn, 4),
                "conf_dir_acc_nn": round(mean_dir_nn, 3),
                "mean_ic_lgb": round(mean_ic_lgb, 4),
                "mean_ic_ens": round(mean_ic_ens, 4),
            }
            all_study_results[h_key][w_name] = res_entry

            print(f"  ==> [{w_name}] CV-3 Summary: NN_IC = {mean_ic_nn:+.4f} ± {std_ic_nn:.4f} | "
                  f"LGB_IC = {mean_ic_lgb:+.4f} | ENS_IC = {mean_ic_ens:+.4f} | Conf-Dir = {mean_dir_nn*100:.1f}%\n")

            with open(out_json, "w") as f:
                json.dump(all_study_results, f, indent=2)

    print("\n" + "=" * 90)
    print(f"🎉 ФУНДАМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ УСПЕШНО ЗАВЕРШЕНО! Результаты: {out_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()
