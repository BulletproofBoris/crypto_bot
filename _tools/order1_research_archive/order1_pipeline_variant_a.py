#!/usr/bin/env python3
"""
_tools/order1_pipeline_variant_a.py
=============================================================================
Сквозной конвейер построения моделей 1-го порядка по Варианту А:
  Горизонты: H in {1, 2, 5, 10, 20, 40, 80, 160} секунд (8 уровней).

Двухфазная методология:
  Фаза 1: Определение оптимального K* = L / H для каждого из 8 горизонтов:
          - Для H <= 20s: сетка K in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0}
          - Для H >= 40s: сетка K in {1.0, 2.0, 3.0, 4.0, 5.0}
  Фаза 2: Для каждого горизонта при оптимальном K* — обучение всех 7 архитектур (10 эпох),
          расчет матрицы разнообразия (Diversity) и жадный отбор (Greedy Forward Selection)
          оптимального поднабора моделей для финального ансамбля.
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

VARIANT_A_HORIZONS = [1, 2, 5, 10, 20, 40, 80, 160]


def load_dataset_features(
    n_train: int,
    n_val: int,
    parquet_path: Path,
    feats_json: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
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


def run_phase_1_k_search(
    horizons: List[int],
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    X_tr_norm: np.ndarray,
    X_val_norm: np.ndarray,
    out_dir: Path,
) -> Dict[int, float]:
    """Фаза 1: Поиск K* для каждого горизонта H."""
    from _tools.order1_multi_horizon_k_explorer import evaluate_single_h_k

    print("\n" + "=" * 90)
    print("📍 [ФАЗА 1] ПОИСК ОПТИМАЛЬНОГО K* ДЛЯ ВСЕХ 8 ГОРИЗОНТОВ ВАРИАНТА А")
    print("=" * 90)

    optimal_k_map = {}

    for H in horizons:
        if H <= 20:
            k_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        else:
            k_grid = [1.0, 2.0, 3.0, 4.0, 5.0]

        print(f"\n--- 🔎 Исследование горизонта H = {H}s (Сетка K: {k_grid}) ---")
        best_k = None
        best_score = -1e9
        best_metrics = None

        for K in k_grid:
            t0 = time.time()
            res = evaluate_single_h_k(df_tr, df_val, X_tr_norm, X_val_norm, H, K, epochs=5)
            dt = time.time() - t0
            em = res["ensemble"]
            ic_s = f"{em['ic']:+.4f}"
            print(f"  • K={K:<4.1f} (L={res['L']:<3}s) | Macro-AUC: {em['auc_macro']:.4f} | Conf-Dir: {em['conf_dir_acc']:<5.1f}% | IC: {ic_s} | Score: {res['score']:<5.1f} ({dt:.1f}с)")
            if res["score"] > best_score:
                best_score = res["score"]
                best_k = K
                best_metrics = res

        optimal_k_map[H] = best_k
        print(f"👉 ОПТИМУМ ДЛЯ H = {H}s: K* = {best_k:.1f} (L* = {int(round(H * best_k))}s)")

        # Сохраняем промежуточно
        with open(out_dir / "phase1_optimal_k.json", "w") as f:
            json.dump({str(k): v for k, v in optimal_k_map.items()}, f, indent=2)

    return optimal_k_map


def run_phase_2_architecture_selection(
    horizons: List[int],
    optimal_k_map: Dict[int, float],
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    X_tr_norm: np.ndarray,
    X_val_norm: np.ndarray,
    feature_cols: List[str],
    epochs: int,
    out_dir: Path,
) -> Dict[int, any]:
    """Фаза 2: Обучение 7 архитектур (10 эпох) и жадный отбор ансамбля для каждого H."""
    print("\n" + "=" * 90)
    print("📍 [ФАЗА 2] ОБУЧЕНИЕ 7 АРХИТЕКТУР И АБЛЯЦИОННЫЙ ОТБОР АНСАМБЛЕЙ")
    print("=" * 90)

    n_features = len(feature_cols)
    final_horizon_ensembles = {}

    for H in horizons:
        K = optimal_k_map[H]
        L = max(1, int(round(H * K)))

        print(f"\n{'='*35} ГОРИЗОНТ H = {H}s (K* = {K:.1f}, L* = {L}s) {'='*35}")

        # 1. Разметка
        lbl_tr = compute_single_horizon_labels(df_tr, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)
        lbl_val = compute_single_horizon_labels(df_val, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)

        y_tr_06 = (lbl_tr["class_7"] + 3).clip(0, 6).astype(np.int64)
        y_val_06 = (lbl_val["class_7"] + 3).clip(0, 6).astype(np.int64)

        # 2. 3D тензоры (N, L, F)
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

        model_preds = {}
        model_metrics = {}

        # 3. LightGBM
        t0 = time.time()
        lgb_mod, lgb_m, lgb_val_p, _ = train_lightgbm_architecture(
            X_tr_norm, y_tr_06, X_val_norm, y_val_06, n_estimators=70, learning_rate=0.05
        )
        model_preds["LightGBM"] = lgb_val_p
        model_metrics["LightGBM"] = lgb_m
        print(f"  ✓ [1/7] LightGBM | AUC: {lgb_m['auc_macro']:.4f} | Conf-Dir: {lgb_m['conf_dir_acc']:.1f}% | IC: {lgb_m['ic']:+.4f} ({time.time()-t0:.1f}с)")

        # 4. 6 Нейросетей (PyTorch CUDA)
        class_weights = compute_class_weights(y_tr_06, strategy="sqrt")
        ds_tr = TensorDataset(X_tr_seq, y_tr_t)
        loader_tr = DataLoader(ds_tr, batch_size=512 if L > 100 else 1024, shuffle=True)

        arch_keys = ["mlp_bottleneck", "conv1d_gru", "causal_cnn", "attention", "wavenet_tcn", "cross_mixer"]
        for idx, arch in enumerate(arch_keys, start=2):
            t_net = time.time()
            net = NEURAL_FACTORY[arch](seq_len=L, n_features=n_features, n_classes=NUM_CLASSES).to(DEVICE)
            crit = FocalLoss(alpha=class_weights, gamma=2.0)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

            best_auc = -1.0
            best_w = None

            for ep in range(epochs):
                net.train()
                for bx, by in loader_tr:
                    opt.zero_grad()
                    crit(net(bx), by).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                    opt.step()
                sched.step()

                net.eval()
                val_p_chunks = []
                with torch.no_grad():
                    for vi in range(0, len(X_val_seq), 1024):
                        chunk = X_val_seq[vi:vi + 1024]
                        val_p_chunks.append(F.softmax(net(chunk), dim=-1).cpu().numpy())
                vp = np.concatenate(val_p_chunks, axis=0)
                cur_auc = float(compute_evaluation_metrics(y_val_06, vp)["auc_macro"])
                if cur_auc > best_auc:
                    best_auc = cur_auc
                    best_w = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            if best_w:
                net.load_state_dict(best_w)
            net.eval()
            val_p_chunks = []
            with torch.no_grad():
                for vi in range(0, len(X_val_seq), 1024):
                    chunk = X_val_seq[vi:vi + 1024]
                    val_p_chunks.append(F.softmax(net(chunk), dim=-1).cpu().numpy())
            final_p = np.concatenate(val_p_chunks, axis=0)

            m = compute_evaluation_metrics(y_val_06, final_p)
            model_preds[arch] = final_p
            model_metrics[arch] = m
            print(f"  ✓ [{idx}/7] {arch:<14} | AUC: {m['auc_macro']:.4f} | Conf-Dir: {m['conf_dir_acc']:.1f}% | IC: {m['ic']:+.4f} ({time.time()-t_net:.1f}с)")

        # 5. Жадный абляционный отбор (Greedy Forward Selection)
        all_models = list(model_preds.keys())
        remaining = list(all_models)
        selected = []
        best_score = -1e9
        best_ens_info = None

        def eval_sub(models):
            stacked = np.stack([model_preds[m] for m in models], axis=0)
            p_avg = np.mean(stacked, axis=0)
            logits = np.log(np.clip(p_avg, 1e-7, 1.0 - 1e-7))
            p_sc = np.exp(logits / 0.5)
            p_sc /= p_sc.sum(axis=1, keepdims=True)
            em = compute_evaluation_metrics(y_val_06, p_sc)
            sc = em["ic"] * 100.0 + (em["auc_macro"] - 0.5) * 50.0 + (em["conf_dir_acc"] - 50.0) * 0.5 - em["ece"]
            return em, sc

        print("\n  🎯 Жадный отбор ансамбля:")
        for step in range(1, len(all_models) + 1):
            cand_best = None
            cand_score = -1e9
            cand_m = None

            for cand in remaining:
                trial = selected + [cand]
                em, sc = eval_sub(trial)
                if sc > cand_score:
                    cand_score = sc
                    cand_best = cand
                    cand_m = em

            selected.append(cand_best)
            remaining.remove(cand_best)
            ic_s = f"{cand_m['ic']:+.4f}"
            print(f"    Шаг {step}: +{cand_best:<14} | Ансамбль {step} мод. | AUC: {cand_m['auc_macro']:.4f} | Conf-Dir: {cand_m['conf_dir_acc']:.1f}% | IC: {ic_s} | Score: {cand_score:.1f}")

            if cand_score > best_score:
                best_score = cand_score
                best_ens_info = {
                    "ensemble_size": step,
                    "models": list(selected),
                    "metrics": cand_m,
                    "score": round(cand_score, 2),
                }

        final_horizon_ensembles[H] = {
            "H": H, "K": K, "L": L,
            "best_ensemble": best_ens_info,
            "individual_models": model_metrics,
        }
        print(f"\n  🏆 ИТОГ ДЛЯ H = {H}s: Оптимальное ядро из {best_ens_info['ensemble_size']} моделей: {', '.join(best_ens_info['models'])}")

        with open(out_dir / f"horizon_{H}s_ensemble.json", "w") as f:
            json.dump(final_horizon_ensembles[H], f, indent=2)

    return final_horizon_ensembles


def main():
    parser = argparse.ArgumentParser(description="Variant A Order-1 Pipeline")
    parser.add_argument("--train-samples", type=int, default=25000)
    parser.add_argument("--val-samples", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="scratch/variant_a_pipeline_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
    feats_json = BASE_DIR / "data" / "features" / "selected_features.json"

    print("=" * 95)
    print("🚀 СКВОЗНОЙ КОНВЕЙЕР ПОСТРОЕНИЯ МОДЕЛЕЙ 1-ГО ПОРЯДКА: ВАРИАНТ А (8 ГОРИЗОНТОВ)")
    print(f"   Горизонты: {VARIANT_A_HORIZONS} секунд | Train = {args.train_samples}, Val = {args.val_samples}")
    print("=" * 95)

    df_tr, df_val, X_tr_norm, X_val_norm, feature_cols = load_dataset_features(
        args.train_samples, args.val_samples, parquet_path, feats_json
    )

    # 1. Поиск K* (если не был сохранен)
    p1_cache = out_dir / "phase1_optimal_k.json"
    if p1_cache.exists():
        with open(p1_cache) as f:
            optimal_k_map = {int(k): v for k, v in json.load(f).items()}
        print(f"Загружен кэш Фазы 1 (K*): {optimal_k_map}")
    else:
        optimal_k_map = run_phase_1_k_search(
            VARIANT_A_HORIZONS, df_tr, df_val, X_tr_norm, X_val_norm, out_dir
        )

    # 2. Фаза 2: Обучение 7 моделей и жадный отбор ансамблей
    final_ensembles = run_phase_2_architecture_selection(
        VARIANT_A_HORIZONS, optimal_k_map, df_tr, df_val, X_tr_norm, X_val_norm, feature_cols,
        epochs=args.epochs, out_dir=out_dir
    )

    print("\n" + "=" * 95)
    print("🎓 ИТОГОВАЯ МАТРИЦА ЛУЧШИХ АНСАМБЛЕЙ ВАРИАНТА А:")
    print("=" * 95)
    print(f"{'Горизонт H':<10} {'K*':<6} {'L*':<8} {'Ядро ансамбля':<34} {'Macro-AUC':<12} {'Conf-Dir':<10} {'IC (Альфа)':<12}")
    print("-" * 95)
    for H, item in final_ensembles.items():
        be = item["best_ensemble"]
        em = be["metrics"]
        models_str = "+".join(be["models"])
        ic_s = f"{em['ic']:+.4f}"
        print(f"{H:<4} сек   {item['K']:<6.1f} {item['L']:<4} сек {models_str:<34} {em['auc_macro']:<12.4f} {em['conf_dir_acc']:<10.1f} {ic_s:<12}")
    print("=" * 95)


if __name__ == "__main__":
    main()
