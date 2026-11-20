#!/usr/bin/env python3
"""
_tools/comprehensive_c_and_ntrain_study.py
=============================================================================
Масштабное исследование коэффициента C = N_train / L и граничных условий N_train
по ВСЕМ 8 горизонтам Варианта А: H in {1, 2, 5, 10, 20, 40, 80, 160}с.

Сетка значений C:
  C in {500, 1000, 2000, 4000, 6000}
С физическим диапазоном:
  N_train от 10 000 до 600 000 сэмплов.

Использует:
  - RTX 5060 Ti + torch.amp.autocast('cuda', dtype=torch.bfloat16)
  - Архитектура Conv1D-GRU с Focal Loss (gamma=2.0)
  - Фиксированный OOS валидационный срез N_val = 20 000 сэмплов
=============================================================================
"""

import json
import os
import sys
import time
from pathlib import Path

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
    DEVICE,
    NUM_CLASSES,
)


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


def run_full_grid_study():
    print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    parquet_path = Path("data/features/prl_1s_features.parquet")
    feats_json = Path("data/features/selected_features.json")
    k_cache_file = Path("scratch/variant_a_pipeline_results/phase1_optimal_k.json")

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

    # Сетка C
    c_grid = [500, 1000, 2000, 4000, 6000]
    n_val = 20_000
    max_history_needed = 650_000  # до 7.5 суток торгов

    print(f"Loading last {max_history_needed:,} rows from parquet...")
    total_rows = pf.metadata.num_rows
    start_row = max(0, total_rows - max_history_needed)

    df_full = pq.read_table(parquet_path, columns=cols_present).to_pandas().iloc[start_row:].reset_index(drop=True)
    df_full = df_full.dropna(subset=["ST_mid_price"]).reset_index(drop=True)
    feature_cols = [c for c in selected_features if c in df_full.columns]
    X_raw_all = df_full[feature_cols].values.astype(np.float32)

    mean = np.nanmean(X_raw_all, axis=0)
    std = np.nanstd(X_raw_all, axis=0) + 1e-6
    X_norm_all = np.nan_to_num((X_raw_all - mean) / std, nan=0.0).clip(-5.0, 5.0)

    val_start = len(df_full) - n_val
    df_val = df_full.iloc[val_start:].reset_index(drop=True)
    X_val_norm = X_norm_all[val_start:]

    horizons = [1, 2, 5, 10, 20, 40, 80, 160]
    all_horizon_results = {}

    for H in horizons:
        K = optimal_k_map[H]
        L = max(1, int(round(H * K)))
        n_features = len(feature_cols)

        print(f"\n{'='*80}")
        print(f"🚀 ИССЛЕДОВАНИЕ ГОРИЗОНТА H={H}s (K*={K}, L*={L}s) ПО ВСЕЙ СЕТКЕ C")
        print(f"{'='*80}")

        # Разметка фиксированной валидации
        lbl_val = compute_single_horizon_labels(df_val, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)
        y_val_06 = (lbl_val["class_7"] + 3).clip(0, 6).astype(np.int64)
        ds_val = FastSequenceDataset(X_val_norm, y_val_06, L)
        val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False)

        horizon_data = []

        for C in c_grid:
            # Расчет N_train от C
            raw_n_tr = int(C * L)
            # Для малых окон задаем разумный минимум 20 000, для огромных - максимум доступного в слайсе
            n_tr = min(raw_n_tr, val_start - 1000)

            # Пропуск, если n_tr слишком мал (< 10000) или если C не меняет N_train
            if n_tr < 10_000:
                n_tr = 15_000

            tr_start = val_start - n_tr
            if tr_start < 0:
                print(f"  • C={C:<5} (N={n_tr:,}) пропущен: превышает объем истории")
                continue

            df_tr = df_full.iloc[tr_start:val_start].reset_index(drop=True)
            X_tr_norm = X_norm_all[tr_start:val_start]

            # Разметка тренировочного среза
            lbl_tr = compute_single_horizon_labels(df_tr, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)
            y_tr_06 = (lbl_tr["class_7"] + 3).clip(0, 6).astype(np.int64)

            ds_tr = FastSequenceDataset(X_tr_norm, y_tr_06, L)
            train_loader = DataLoader(ds_tr, batch_size=2048, shuffle=True, drop_last=True)

            model = Conv1DGRUNet(seq_len=L, n_features=n_features, n_classes=NUM_CLASSES).to(DEVICE)
            weights = compute_class_weights(y_tr_06, strategy="sqrt")
            criterion = FocalLoss(alpha=weights, gamma=2.0).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            epochs = 8
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
            train_time = time.time() - t0

            # OOS Inference
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

            hours_hist = round(n_tr / 3600.0, 1)
            ic = metrics["ic"]
            conf_dir = metrics["conf_dir_acc"]
            auc = metrics["auc_macro"]

            entry = {
                "H": H,
                "K": K,
                "L": L,
                "C": C,
                "N_train": n_tr,
                "hours_history": hours_hist,
                "ic": round(ic, 4),
                "conf_dir_acc": round(conf_dir, 3),
                "auc_macro": round(auc, 4),
                "train_time_sec": round(train_time, 2)
            }
            horizon_data.append(entry)

            print(f"  • C={C:<5} | N_train={n_tr:<7,} ({hours_hist:>4.1f}ч) | "
                  f"IC: {ic:+.4f} | Conf-Dir: {conf_dir:>5.1f}% | AUC: {auc:.4f} | Time: {train_time:>4.1f}s")

            del model, train_loader, optimizer
            torch.cuda.empty_cache()

        all_horizon_results[str(H)] = horizon_data

        # Промежуточный сброс на диск
        out_file = Path("scratch/variant_a_pipeline_results/comprehensive_c_scaling_study.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(all_horizon_results, f, indent=2)

    print(f"\n🎉 Исследование всех 8 горизонтов завершено! Сохранено в {out_file}")


if __name__ == "__main__":
    run_full_grid_study()
