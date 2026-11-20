#!/usr/bin/env python3
"""
_tools/order1_h1s_full_lab.py
=============================================================================
Глубокое исследование обучения 7 архитектур на окне H=1s (K=3.0, L=3s):
  1. Детальная сходимость каждой модели по 10 эпохам (Train/Val Loss, AUC, Dir-Acc, IC).
  2. Матрица корреляции предсказаний (Diversity Matrix) 7 архитектур.
  3. Абляционный анализ: нужны ли все 7 моделей или оптимален поднабор из 3-4?
  4. Формирование финальной консолидированной модели с калибровкой.
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
    NEURAL_FACTORY,
    FocalLoss,
    compute_class_weights,
    compute_evaluation_metrics,
    train_lightgbm_architecture,
    DEVICE,
    CLASSES_7,
    NUM_CLASSES,
)


def run_h1s_full_lab(
    n_train: int = 30000,
    n_val: int = 10000,
    epochs: int = 10,
    batch_size: int = 1024,
    out_dir: str = "scratch/h1s_full_lab",
) -> Dict[str, any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    H = 1
    K = 3.0
    L = 3

    print("=" * 85)
    print(f"🔬 ГЛУБОКОЕ ИССЛЕДОВАНИЕ СХОДИМОСТИ И АНСАМБЛЯ: H = {H}s (K = {K:.1f}, L = {L}s)")
    print(f"   Train = {n_train:,}, Val = {n_val:,} | Эпох = {epochs} | Устройство: {DEVICE}")
    print("=" * 85)

    # 1. Загрузка данных
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

    needed_rows = n_train + n_val + 1000
    start_row = max(0, pf.metadata.num_rows - needed_rows)

    df_full = pq.read_table(parquet_path, columns=cols_present).to_pandas().iloc[start_row:].reset_index(drop=True)
    df_full = df_full.dropna(subset=["ST_mid_price"]).reset_index(drop=True)

    feature_cols = [c for c in selected_features if c in df_full.columns]
    n_features = len(feature_cols)

    tr_end = len(df_full) - n_val
    tr_start = max(0, tr_end - n_train)

    df_tr = df_full.iloc[tr_start:tr_end].reset_index(drop=True)
    df_val = df_full.iloc[tr_end:tr_end + n_val].reset_index(drop=True)

    X_tr_raw = df_tr[feature_cols].values.astype(np.float32)
    X_val_raw = df_val[feature_cols].values.astype(np.float32)

    mean = np.nanmean(X_tr_raw, axis=0)
    std = np.nanstd(X_tr_raw, axis=0) + 1e-6
    X_tr_norm = np.nan_to_num((X_tr_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)
    X_val_norm = np.nan_to_num((X_val_raw - mean) / std, nan=0.0).clip(-5.0, 5.0)

    # Разметка H=1s, K=3.0
    lbl_tr = compute_single_horizon_labels(df_tr, "prl", H=1, K=3.0, min_vol=50.0, tick_size=0.005)
    lbl_val = compute_single_horizon_labels(df_val, "prl", H=1, K=3.0, min_vol=50.0, tick_size=0.005)

    y_tr_06 = (lbl_tr["class_7"] + 3).clip(0, 6).astype(np.int64)
    y_val_06 = (lbl_val["class_7"] + 3).clip(0, 6).astype(np.int64)

    print(f"✓ Данные подготовлены: {n_features} признаков, L = {L}s.")
    print(f"  Дисбаланс: Класс 0 = {(y_tr_06 == 3).mean()*100:.2f}%, Активные движения = {(y_tr_06 != 3).mean()*100:.2f}%")

    # 3D последовательности (N, L, F)
    def to_seq_tensor(X_arr):
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

    X_tr_seq = to_seq_tensor(X_tr_norm)
    X_val_seq = to_seq_tensor(X_val_norm)
    y_tr_t = torch.tensor(y_tr_06, dtype=torch.long, device=DEVICE)

    model_predictions = {}
    model_metrics = {}
    convergence_logs = {}

    # [1/4] Обучение Модели 1: LightGBM
    print("\n--- [1/7] LightGBM (Gradient Boosted Trees) ---")
    t0 = time.time()
    lgb_mod, lgb_m, lgb_val_p, lgb_tr_p = train_lightgbm_architecture(
        X_tr_norm, y_tr_06, X_val_norm, y_val_06, n_estimators=80, learning_rate=0.05
    )
    model_predictions["1_LightGBM"] = lgb_val_p
    model_metrics["1_LightGBM"] = lgb_m
    print(f"  ✓ Завершено за {time.time()-t0:.2f}с | Macro-AUC: {lgb_m['auc_macro']:.4f} | Conf-Dir: {lgb_m['conf_dir_acc']:.1f}% | IC: {lgb_m['ic']:+.4f}")

    # [2/4] Обучение 6 нейросетей с пошаговым трекингом эпох
    arch_keys = ["mlp_bottleneck", "conv1d_gru", "causal_cnn", "attention", "wavenet_tcn", "cross_mixer"]
    class_weights = compute_class_weights(y_tr_06, strategy="sqrt")

    ds_tr = TensorDataset(X_tr_seq, y_tr_t)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    for idx, arch in enumerate(arch_keys, start=2):
        name = f"{idx}_{arch}"
        print(f"\n--- [{idx}/7] {arch} ({epochs} эпох на PyTorch CUDA) ---")
        t_net = time.time()
        net = NEURAL_FACTORY[arch](seq_len=L, n_features=n_features, n_classes=NUM_CLASSES).to(DEVICE)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        epoch_log = []
        best_auc = -1.0
        best_weights = None

        for ep in range(1, epochs + 1):
            net.train()
            tr_loss = 0.0
            n_b = 0
            for bx, by in loader_tr:
                optimizer.zero_grad()
                out = net(bx)
                loss = criterion(out, by)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
                tr_loss += loss.item()
                n_b += 1
            scheduler.step()
            tr_loss /= max(1, n_b)

            net.eval()
            with torch.no_grad():
                val_p = F.softmax(net(X_val_seq), dim=-1).cpu().numpy()

            m = compute_evaluation_metrics(y_val_06, val_p)
            epoch_log.append({
                "epoch": ep,
                "train_loss": round(tr_loss, 4),
                "val_loss": m["loss"],
                "auc_macro": m["auc_macro"],
                "conf_dir_acc": m["conf_dir_acc"],
                "ic": m["ic"],
                "ece": m["ece"],
            })

            if m["auc_macro"] > best_auc:
                best_auc = m["auc_macro"]
                best_weights = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            if ep in (1, 3, 5, 8, 10):
                ic_s = f"{m['ic']:+.4f}"
                print(f"    Эпоха {ep:>2}/{epochs}: Loss={tr_loss:.4f} | Val-AUC={m['auc_macro']:.4f} | Conf-Dir={m['conf_dir_acc']:.1f}% | IC={ic_s}")

        # Фиксируем лучшие веса
        if best_weights:
            net.load_state_dict(best_weights)
        net.eval()
        with torch.no_grad():
            final_p = F.softmax(net(X_val_seq), dim=-1).cpu().numpy()

        model_predictions[name] = final_p
        model_metrics[name] = compute_evaluation_metrics(y_val_06, final_p)
        convergence_logs[name] = epoch_log
        print(f"  ✓ {arch} обучен за {time.time()-t_net:.1f}с | Лучший Macro-AUC: {best_auc:.4f}")

    # [3/4] Матрица корреляции предсказаний (Diversity Matrix)
    print("\n" + "=" * 85)
    print("🔍 АНАЛИЗ РАЗНООБРАЗИЯ (DIVERSITY): КОРРЕЛЯЦИЯ ПРЕДСКАЗАНИЙ МОДЕЛЕЙ")
    print("=" * 85)
    all_names = list(model_predictions.keys())
    exp_returns = {}
    for name in all_names:
        p = model_predictions[name]
        exp_returns[name] = np.sum(p * CLASSES_7[np.newaxis, :], axis=1)

    corr_matrix = {}
    for n1 in all_names:
        corr_matrix[n1] = {}
        for n2 in all_names:
            c = float(np.corrcoef(exp_returns[n1], exp_returns[n2])[0, 1])
            corr_matrix[n1][n2] = round(c, 3)

    # Печать матрицы корреляции
    print(f"{'Модель':<16} " + " ".join([f"{n.split('_')[1][:6]:>7}" for n in all_names]))
    for n1 in all_names:
        row = [f"{corr_matrix[n1][n2]:>7.2f}" for n2 in all_names]
        print(f"{n1:<16} " + " ".join(row))

    # [4/4] Жадный абляционный отбор (Greedy Forward Selection)
    print("\n" + "=" * 85)
    print("🎯 АБЛЯЦИОННЫЙ ОТБОР: СКОЛЬКО И КАКИЕ МОДЕЛИ ДАЮТ МАКСИМАЛЬНЫЙ АНСАМБЛЬ?")
    print("=" * 85)

    remaining = list(all_names)
    selected = []
    best_ens_score = -1e9
    ablation_history = []

    def eval_subset(subset_names):
        stacked = np.stack([model_predictions[n] for n in subset_names], axis=0)
        p_avg = np.mean(stacked, axis=0)
        logits = np.log(np.clip(p_avg, 1e-7, 1.0 - 1e-7))
        p_sc = np.exp(logits / 0.5)
        p_sc /= p_sc.sum(axis=1, keepdims=True)
        m = compute_evaluation_metrics(y_val_06, p_sc)
        sc = m["ic"] * 100.0 + (m["auc_macro"] - 0.5) * 50.0 + (m["conf_dir_acc"] - 50.0) * 0.5 - m["ece"]
        return m, sc

    for step in range(1, len(all_names) + 1):
        best_candidate = None
        best_cand_score = -1e9
        best_cand_metrics = None

        for cand in remaining:
            trial_set = selected + [cand]
            m, sc = eval_subset(trial_set)
            if sc > best_cand_score:
                best_cand_score = sc
                best_candidate = cand
                best_cand_metrics = m

        selected.append(best_candidate)
        remaining.remove(best_candidate)

        em = best_cand_metrics
        ic_s = f"{em['ic']:+.4f}"
        print(f"Шаг {step}: Добавлена {best_candidate:<16} | Ансамбль из {step} сетей | AUC: {em['auc_macro']:.4f} | Conf-Dir: {em['conf_dir_acc']:.1f}% | IC: {ic_s} | Score: {best_cand_score:.1f}")

        ablation_history.append({
            "step": step,
            "added_model": best_candidate,
            "ensemble_size": step,
            "models": list(selected),
            "metrics": em,
            "score": round(best_cand_score, 2),
        })

    # Сравнение: Полный ансамбль (все 7) vs Оптимальный поднабор
    m_all, sc_all = eval_subset(all_names)
    best_step_info = max(ablation_history, key=lambda x: x["score"])

    print("\n" + "=" * 85)
    print("🏆 ИТОГИ АБЛЯЦИОННОГО ИССЛЕДОВАНИЯ:")
    print("=" * 85)
    print(f"• Все 7 архитектур вместе:      Macro-AUC = {m_all['auc_macro']:.4f} | Conf-Dir = {m_all['conf_dir_acc']:.1f}% | IC = {m_all['ic']:+.4f} | Score = {sc_all:.1f}")
    print(f"• Оптимальный поднабор ({best_step_info['ensemble_size']} мод.): Macro-AUC = {best_step_info['metrics']['auc_macro']:.4f} | Conf-Dir = {best_step_info['metrics']['conf_dir_acc']:.1f}% | IC = {best_step_info['metrics']['ic']:+.4f} | Score = {best_step_info['score']:.1f}")
    print(f"• Состав оптимального ядра: {', '.join([m.split('_')[1] for m in best_step_info['models']])}")
    print("=" * 85)

    report = {
        "H": H, "K": K, "L": L,
        "individual_models": model_metrics,
        "correlation_matrix": corr_matrix,
        "ablation_history": ablation_history,
        "full_ensemble_metrics": m_all,
        "optimal_ensemble_metrics": best_step_info,
        "convergence_logs": convergence_logs,
    }

    with open(out_path / "h1s_full_lab_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H=1s Full Lab")
    parser.add_argument("--train-samples", type=int, default=30000)
    parser.add_argument("--val-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_h1s_full_lab(
        n_train=args.train_samples,
        n_val=args.val_samples,
        epochs=args.epochs,
    )
