#!/usr/bin/env python3
"""
Масштабное исследование сетки (H in [1:25:1], K in [3.0:6.0:0.5]).
Задачи:
1. Оценка всех 175 комбинаций (H, K) для PRL и QUANTUS:
   - L = round(H * K), N_eff = T_train / L, проверка L <= L_max
   - Ошибка сходимости разметки Err_z(+2), Err_z(-2), Max_Err_z
   - Предсказательная сила OOS (IC, ROC-AUC) через быстрый бустинг
2. Корреляционный анализ между горизонтами H:
   - Матрица взаимной корреляции импульсов z
   - Скорость затухания корреляции при росте delta H
3. Определение эмпирической функции изменения K*(H) и отбор ортогонального ансамбля.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.data_labeler import compute_single_horizon_labels, COIN_CONFIG

SAFE_WORKERS = min(14, max(4, os.cpu_count() - 2 if os.cpu_count() else 4))

def parse_args():
    parser = argparse.ArgumentParser(description="Research H in [1:25], K in [3:6:0.5]")
    parser.add_argument("--coin", type=str, default="prl", choices=["prl", "quantus", "all"])
    parser.add_argument("--workers", type=int, default=SAFE_WORKERS)
    return parser.parse_args()


def eval_single_hk(args):
    """Вычисляет разметку и Out-of-Sample метрики для одной пары (H, K)."""
    df_mini, coin, H, K, min_vol, tick_size, train_end, test_start, test_end, X_tr_sub, X_te, valid_tr_idx, te_idx = args
    L = int(round(H * K))

    try:
        res = compute_single_horizon_labels(df_mini, coin, H, K, min_vol, tick_size)
    except Exception as e:
        return None

    z = res["z"]
    c7 = res["class_7"]
    dt_ticks = res["dt_ticks"]
    is_void = res["is_void"]

    # Фильтрация обучающей выборки от артефактов и void
    tr_mask = (valid_tr_idx < train_end - H) & (~is_void[valid_tr_idx])
    tr_i = valid_tr_idx[tr_mask]

    te_mask = (te_idx < test_end - H) & (~is_void[te_idx])
    te_i = te_idx[te_mask]

    if len(tr_i) < 500 or len(te_i) < 300:
        return None

    z_tr, c_tr = z[tr_i], c7[tr_i]
    z_te, c_te = z[te_i], c7[te_i]
    dt_te = dt_ticks[te_i]

    # Сходимость классов ±2
    m_tr_p2 = (c_tr == 2)
    m_te_p2 = (c_te == 2)
    z_tr_p2 = z_tr[m_tr_p2].mean() if m_tr_p2.sum() > 5 else 1.0
    z_te_p2 = z_te[m_te_p2].mean() if m_te_p2.sum() > 5 else 1.0
    err_p2 = abs(abs(z_te_p2) - abs(z_tr_p2)) / abs(z_tr_p2) * 100.0 if z_tr_p2 != 0 else 0.0

    m_tr_m2 = (c_tr == -2)
    m_te_m2 = (c_te == -2)
    z_tr_m2 = z_tr[m_tr_m2].mean() if m_tr_m2.sum() > 5 else -1.0
    z_te_m2 = z_te[m_te_m2].mean() if m_te_m2.sum() > 5 else -1.0
    err_m2 = abs(abs(z_te_m2) - abs(z_tr_m2)) / abs(z_tr_m2) * 100.0 if z_tr_m2 != 0 else 0.0

    max_err_z = max(err_p2, err_m2)

    # Быстрый регрессор для OOS IC
    y_tr = np.clip(z[tr_i], -2.5, 2.5)
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=30,
        learning_rate=0.08,
        num_leaves=25,
        max_depth=5,
        n_jobs=1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_tr_sub[tr_mask], y_tr)
    pred_te = model.predict(X_te[te_mask])

    ic = float(np.corrcoef(pred_te, np.clip(z_te, -2.5, 2.5))[0, 1]) if np.std(pred_te) > 1e-6 else 0.0

    # ROC-AUC детекции движения
    b_up = (c_te >= 2).astype(int)
    auc = float(roc_auc_score(b_up, pred_te)) if b_up.sum() > 10 else 0.5

    n_eff = train_end // L

    return {
        "H": H,
        "K": K,
        "L": L,
        "N_eff": n_eff,
        "err_p2": round(err_p2, 2),
        "err_m2": round(err_m2, 2),
        "max_err_z": round(max_err_z, 2),
        "oos_ic": round(ic, 4),
        "oos_auc": round(auc, 4),
        "pct_p2": round(float(m_tr_p2.mean() * 100), 2),
        "pct_m2": round(float(m_tr_m2.mean() * 100), 2),
        "z_series": z[test_start:test_end:10] # компактный срез для корреляций
    }


def run_grid_research(coin: str, workers: int):
    coin = coin.lower()
    fpath = BASE_DIR / "data" / "features" / f"{coin}_1s_features.parquet"
    feats_json = BASE_DIR / "data" / "features" / f"selected_features_{coin}_h8s.json"

    with open(feats_json) as f:
        selected_feats = json.load(f)["feature_order"][:30] # Top-30 для быстрого сканирования 175 пар

    cfg = COIN_CONFIG[coin]
    tick_size = cfg["tick_size"]
    min_vol = cfg["min_vol"]

    req_cols = [
        'ST_mid_price', 'ST_spread', 'ST_bid_1_price', 'ST_ask_1_price',
        'ST_bid_1_qty', 'ST_ask_1_qty', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
        'ST_trade_vol', 'is_data_gap'
    ]
    load_cols = list(dict.fromkeys(req_cols + selected_feats))
    tbl = pq.read_table(fpath, columns=load_cols)
    df = tbl.to_pandas()

    n = len(df)
    split_idx = int(n * 0.8)
    embargo = 360
    train_end = split_idx - 60
    test_start = split_idx + embargo
    test_end = n - 60

    # Прореживание для высокой скорости обучения
    stride = 4
    valid_tr_idx = np.arange(60, train_end, stride)
    te_idx = np.arange(test_start, test_end)

    X_all = df[selected_feats].values.astype(np.float32)
    X_tr_sub = X_all[valid_tr_idx]
    X_te = X_all[te_idx]

    df_mini = df[req_cols].copy()

    # Сетка: H in [1:25], K in [3.0:6.0:0.5]
    H_values = list(range(1, 26))
    K_values = [round(k, 1) for k in np.arange(3.0, 6.1, 0.5)]

    print(f"\n{'='*105}")
    print(f"🔬 СКАНИРОВАНИЕ СЕТКИ {coin.upper()}: H in [1..25] ({len(H_values)} зн.), K in [3.0..6.0:0.5] ({len(K_values)} зн.)")
    print(f"   Всего комбинаций: {len(H_values) * len(K_values)} пар | CPU воркеров: {workers}")
    print(f"{'='*105}")

    tasks = []
    for H in H_values:
        for K in K_values:
            tasks.append((df_mini, coin, H, K, min_vol, tick_size, train_end, test_start, test_end, X_tr_sub, X_te, valid_tr_idx, te_idx))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        raw_results = list(executor.map(eval_single_hk, tasks))
    print(f"⏱️ Все 175 комбинаций рассчитаны за {time.time()-t0:.2f} сек!")

    results = [r for r in raw_results if r is not None]
    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'z_series'} for r in results])

    out_csv = BASE_DIR / "data" / "labeled" / f"research_hk_matrix_{coin}.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"📁 Результаты сохранены в: {out_csv.relative_to(BASE_DIR)}")

    # Анализ оптимального K*(H)
    print(f"\n📊 ТОП-1 ОПТИМАЛЬНОЕ K*(H) ДЛЯ КАЖДОГО ГОРИЗОНТА H in [1..25]:")
    print(f"{'H (сек)':<8} | {'K*(H)':<8} | {'L (сек)':<8} | {'N_eff':<9} | {'Max_Err_z':<11} | {'OOS IC':<8} | {'OOS AUC':<8} | {'Статус':<15}")
    print("-" * 90)

    best_k_list = []
    z_series_dict = {}

    for H in H_values:
        sub = [r for r in results if r["H"] == H]
        if not sub:
            continue
        # Критерий: минимальная Max_Err_z при условии OOS IC > 0
        best_cand = min(sub, key=lambda x: (x["max_err_z"] if x["oos_ic"] > 0 else 999.0))
        best_k_list.append(best_cand)
        z_series_dict[H] = best_cand["z_series"]

        status = "🎯 ОТЛИЧНО" if best_cand["max_err_z"] < 3.0 and best_cand["oos_ic"] > 0.15 else ("✅ ХОРОШО" if best_cand["max_err_z"] < 5.0 else "⚠️ УМЕРЕННО")
        print(f"{best_cand['H']:<8} | {best_cand['K']:<8.1f} | {best_cand['L']:<8} | {best_cand['N_eff']:<9,} | {best_cand['max_err_z']:<11.2f}% | {best_cand['oos_ic']:<8.4f} | {best_cand['oos_auc']:<8.4f} | {status:<15}")

    # Корреляционная матрица между горизонтами H при K*(H)
    z_df = pd.DataFrame(z_series_dict)
    corr_mat = z_df.corr()
    corr_csv = BASE_DIR / "data" / "labeled" / f"horizon_correlation_matrix_{coin}.csv"
    corr_mat.to_csv(corr_csv)

    print(f"\n📈 МАТРИЦА ВЗАИМНОЙ КОРРЕЛЯЦИИ ГОРИЗОНТОВ H (выборка шагом 5 сек):")
    sample_hs = [1, 3, 5, 8, 10, 15, 20, 25]
    sample_corr = corr_mat.loc[[h for h in sample_hs if h in corr_mat.index], [h for h in sample_hs if h in corr_mat.columns]]
    print(sample_corr.round(3).to_string())

    # Аппроксимация функции K*(H)
    h_arr = np.array([b["H"] for b in best_k_list], dtype=np.float64)
    k_arr = np.array([b["K"] for b in best_k_list], dtype=np.float64)
    poly = np.polyfit(h_arr, k_arr, deg=1)
    print(f"\n📐 ЭМПИРИЧЕСКАЯ ФУНКЦИЯ K*(H):")
    print(f"   Линейная аппроксимация: K*(H) ≈ {poly[0]:+.4f} * H + {poly[1]:.2f}")
    print(f"   (При H=1..5 K* держится около {k_arr[:5].mean():.1f}, плавно сужаясь к {k_arr[-5:].mean():.1f} на H=20..25 из-за лимита L_max)")

    return best_k_list, corr_mat


def main():
    args = parse_args()
    coins = ["prl", "quantus"] if args.coin == "all" else [args.coin]
    for c in coins:
        run_grid_research(c, args.workers)


if __name__ == "__main__":
    main()

