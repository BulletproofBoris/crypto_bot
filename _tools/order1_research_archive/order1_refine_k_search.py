#!/usr/bin/env python3
"""
_tools/order1_refine_k_search.py
=============================================================================
Глубокое повторное исследование K для всех горизонтов Варианта А:
  Горизонты: H in {1, 2, 5, 10, 20, 80, 160} секунд.

Методология:
  - Сетка K:
      Для H in {1, 2, 5, 10, 20}s : K in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0}
      Для H in {80, 160}s         : K in {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
  - Оценка качества альфы: Information Coefficient (IC) и Confident Directional Accuracy.
  - Чанкованный безопасный инференс без OOM.
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


def load_dataset(n_train: int, n_val: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    parquet_path = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
    feats_json = BASE_DIR / "data" / "features" / "selected_features.json"

    with open(feats_json) as f:
        selected_features = json.load(f)["feature_order"]

    req_cols = [
        "ST_mid_price", "ST_spread", "ST_bid_1_price", "ST_ask_1_price",
        "ST_bid_1_qty", "ST_ask_1_qty", "ST_bids_vol_top5", "ST_asks_vol_top5", "ST_trade_vol"
    ]
    pf = pq.ParquetFile(parquet_path)
    all_needed = list(set(req_cols + selected_features))
    cols_present = [c for c in all_needed if c in pf.schema.names]

    needed_rows = n_train + n_val + 3000
    start_row = max(0, pf.metadata.num_rows - needed_rows)

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

    mean = np.nanmean(X_tr_raw, axis=0)
    std = np.nanstd(X_tr_raw, axis=0) + 1e-6
    X_tr_norm = np.nan_to_num((X_tr_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)
    X_val_norm = np.nan_to_num((X_val_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)

    return df_tr, df_val, X_tr_norm, X_val_norm, feature_cols


def evaluate_candidate(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    X_tr_norm: np.ndarray,
    X_val_norm: np.ndarray,
    H: int,
    K: float,
    epochs: int = 5,
) -> Dict[str, any]:
    L = max(1, int(round(H * K)))
    n_features = X_tr_norm.shape[1]

    # 1. Разметка
    lbl_tr = compute_single_horizon_labels(df_tr, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)
    lbl_val = compute_single_horizon_labels(df_val, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)

    y_tr_06 = (lbl_tr["class_7"] + 3).clip(0, 6).astype(np.int64)
    y_val_06 = (lbl_val["class_7"] + 3).clip(0, 6).astype(np.int64)

    # 2. Тензоры (N, L, F)
    torch.cuda.empty_cache()
    def to_seq(X_arr):
        N = len(X_arr)
        out = np.zeros((N, L, n_features), dtype=np.float32)
        for i in range(L):
            shift = L - 1 - i
            if shift == 0:
                out[:, i, :] = X_arr
            else:
                out[shift:, i, :] = X_arr[:-shift]
                out[:shift, i, :] = X_arr[0]
        return torch.tensor(out, dtype=torch.float32, device=DEVICE)

    X_tr_seq = to_seq(X_tr_norm)
    X_val_seq = to_seq(X_val_norm)
    y_tr_t = torch.tensor(y_tr_06, dtype=torch.long, device=DEVICE)

    # 3. LightGBM
    lgb_mod, lgb_m, lgb_val_p, _ = train_lightgbm_architecture(
        X_tr_norm, y_tr_06, X_val_norm, y_val_06, n_estimators=60, learning_rate=0.05
    )

    # 4. Conv1D-GRU и Causal-CNN
    class_weights = compute_class_weights(y_tr_06, strategy="sqrt")
    batch_size = 512 if L > 120 else 1024
    ds_tr = TensorDataset(X_tr_seq, y_tr_t)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    def train_quick(net_cls):
        net = net_cls(seq_len=L, n_features=n_features, n_classes=NUM_CLASSES).to(DEVICE)
        crit = FocalLoss(alpha=class_weights, gamma=2.0)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            net.train()
            for bx, by in loader_tr:
                opt.zero_grad()
                crit(net(bx), by).backward()
                opt.step()
            sched.step()
        net.eval()
        chunks = []
        with torch.no_grad():
            for vi in range(0, len(X_val_seq), 1024):
                chunks.append(F.softmax(net(X_val_seq[vi:vi + 1024]), dim=-1).cpu().numpy())
        vp = np.concatenate(chunks, axis=0)
        return compute_evaluation_metrics(y_val_06, vp), vp

    causal_m, causal_p = train_quick(CausalCNNNet)
    gru_m, gru_p = train_quick(Conv1DGRUNet)

    # 5. Калиброванный ансамбль (T=0.5)
    stacked = np.stack([lgb_val_p, causal_p, gru_p], axis=0)
    p_avg = np.mean(stacked, axis=0)
    logits = np.log(np.clip(p_avg, 1e-7, 1.0 - 1e-7))
    p_sc = np.exp(logits / 0.5)
    p_sc /= p_sc.sum(axis=1, keepdims=True)
    em = compute_evaluation_metrics(y_val_06, p_sc)

    # Альфа-скор: приоритет IC и точности направления
    alpha_score = em["ic"] * 120.0 + (em["conf_dir_acc"] - 50.0) * 0.6 + (em["auc_macro"] - 0.5) * 40.0 - em["ece"]

    return {
        "H": H, "K": K, "L": L,
        "score": round(alpha_score, 2),
        "lgb": lgb_m,
        "causal_cnn": causal_m,
        "conv1d_gru": gru_m,
        "ensemble": em,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=str, default="1,2,5,10,20")
    parser.add_argument("--train-samples", type=int, default=25000)
    parser.add_argument("--val-samples", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    h_list = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    out_dir = Path("scratch/refined_k_search")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("🔬 ГЛУБОКОЕ ПЕРЕИССЛЕДОВАНИЕ K С ПРИОРИТЕТОМ НА АЛЬФУ (IC)")
    print(f"   Горизонты: {h_list} | Train = {args.train_samples}, Val = {args.val_samples}")
    print("=" * 90)

    df_tr, df_val, X_tr, X_val, cols = load_dataset(args.train_samples, args.val_samples)

    best_k_results = {}

    for H in h_list:
        if H <= 5:
            k_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
        elif H <= 20:
            k_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
        elif H <= 40:
            k_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        else:
            k_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

        print(f"\n{'='*35} ГОРИЗОНТ H = {H}s {'='*35}")
        print(f"Сетка K: {k_grid}")

        best_cand = None
        best_sc = -1e9
        h_cand_records = {}

        for K in k_grid:
            t0 = time.time()
            res = evaluate_candidate(df_tr, df_val, X_tr, X_val, H, K, epochs=args.epochs)
            dt = time.time() - t0
            em = res["ensemble"]
            ic_s = f"{em['ic']:+.4f}"
            print(f"  • K={K:<4.1f} (L={res['L']:<3}s) | IC: {ic_s:<8} | Conf-Dir: {em['conf_dir_acc']:<5.1f}% | AUC: {em['auc_macro']:.4f} | Score: {res['score']:<5.1f} ({dt:.1f}с)")
            h_cand_records[f"K_{K:.1f}"] = res
            if res["score"] > best_sc:
                best_sc = res["score"]
                best_cand = res

        best_em = best_cand["ensemble"]
        best_k_results[H] = {
            "H": H, "best_K": best_cand["K"], "best_L": best_cand["L"],
            "best_IC": best_em["ic"], "best_conf_dir": best_em["conf_dir_acc"],
            "best_AUC": best_em["auc_macro"], "score": best_sc,
        }
        ic_best_s = f"{best_em['ic']:+.4f}"
        print(f"\n🏆 ПИК АЛЬФЫ ДЛЯ H = {H}s: K* = {best_cand['K']:.1f} (L* = {best_cand['L']}s) -> IC = {ic_best_s}, Conf-Dir = {best_em['conf_dir_acc']:.1f}%")

        with open(out_dir / f"h_{H}s_refined.json", "w") as f:
            json.dump(h_cand_records, f, indent=2)

    with open(out_dir / "refined_k_summary.json", "w") as f:
        json.dump(best_k_results, f, indent=2)

    print("\n" + "=" * 90)
    print("🎓 ИТОГОВАЯ ПЕРЕКАЛИБРОВАННАЯ КАРТА K*:")
    print("=" * 90)
    print(f"{'H (сек)':<8} {'Опт. K*':<10} {'История L*':<14} {'IC (Альфа)':<14} {'Conf-Dir (%)':<14} {'Macro-AUC':<12}")
    print("-" * 90)
    for H, item in best_k_results.items():
        ic_s = f"{item['best_IC']:+.4f}"
        print(f"{H:<8} {item['best_K']:<10.1f} {item['best_L']:<4} сек        {ic_s:<14} {item['best_conf_dir']:<14.1f} {item['best_AUC']:<12.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
