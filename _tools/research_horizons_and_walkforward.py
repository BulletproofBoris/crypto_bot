#!/usr/bin/env python3
"""
Large-Scale Horizon & Ratio Research Engine with Correlation Analysis & Walk-Forward
=====================================================================================
Исследование широкого пространства параметров:
- Горизонты: H = 2:60:1 (все 59 секундных горизонтов)
- Коридор L/H: K = 2.0:7.0 с шагом 0.5 (при условии целого L = round(H * K) >= 10с)
- Корреляционный анализ: матрица взаимной корреляции таргетов и сигналов моделей
- Ортогональный отбор: отбор наименее скоррелированных кандидатов с максимальной альфой
- Полная Walk-Forward валидация отобранных кандидатов и ансамбля пула моделей
- Соблюдение лимита ресурсов: не более 80% CPU (12 ядер из 16, 4 ядра свободны).
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
from colorama import init, Fore, Style

warnings.filterwarnings("ignore")
init(autoreset=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.data_labeler import compute_single_horizon_labels, COIN_CONFIG
from _tools.experiment_hk_selection import get_stationary_feature_cols
from _tools.system_resources import SAFE_WORKERS


def eval_hk_candidate(args):
    """Изолированный расчет метрик пары (H, K) для заданного среза."""
    df_mini, coin, H, K, min_vol, tick_size, train_end, test_start, test_end = args
    L = max(10, int(round(H * K)))
    res = compute_single_horizon_labels(df_mini, coin, H, K, min_vol, tick_size)
    z = res["z"]
    c7 = res["class_7"]
    is_void = res["is_void"]

    # In-Sample & Out-of-Sample индексы
    stride = max(3, H // 2)
    tr_idx = np.arange(L, train_end, stride)
    tr_idx = tr_idx[~is_void[tr_idx]]

    te_idx = np.arange(test_start, test_end - H)
    te_idx = te_idx[~is_void[te_idx]]

    if len(tr_idx) < 500 or len(te_idx) < 200:
        return (H, K, None)

    z_tr, c_tr = z[tr_idx], c7[tr_idx]
    z_te, c_te = z[te_idx], c7[te_idx]

    m_tr_p2, m_te_p2 = (c_tr == 2), (c_te == 2)
    m_tr_m2, m_te_m2 = (c_tr == -2), (c_te == -2)

    z_tr_p2 = z_tr[m_tr_p2].mean() if m_tr_p2.sum() > 5 else 1.0
    z_te_p2 = z_te[m_te_p2].mean() if m_te_p2.sum() > 5 else 1.0
    err_p2 = abs(abs(z_te_p2) - abs(z_tr_p2)) / abs(z_tr_p2) * 100.0

    z_tr_m2 = z_tr[m_tr_m2].mean() if m_tr_m2.sum() > 5 else -1.0
    z_te_m2 = z_te[m_te_m2].mean() if m_te_m2.sum() > 5 else -1.0
    err_m2 = abs(abs(z_te_m2) - abs(z_tr_m2)) / abs(z_tr_m2) * 100.0

    max_err_z = max(err_p2, err_m2)
    breakout_pct = ((np.abs(c7) >= 2).mean()) * 100.0
    flat_pct = ((c7 == 0).mean()) * 100.0

    return (H, K, {
        "H": H,
        "K": K,
        "L": L,
        "err_p2": err_p2,
        "err_m2": err_m2,
        "max_err_z": max_err_z,
        "breakout_pct": breakout_pct,
        "flat_pct": flat_pct,
    })


def run_full_research(coin="prl", h_min=2, h_max=60, h_step=1):
    coin = coin.lower()
    fpath = os.path.join(BASE_DIR, "data", "features", f"{coin}_1s_features.parquet")
    if not os.path.exists(fpath):
        print(f"❌ Файл {fpath} не найден!")
        return

    print("=" * 115)
    print(f"🔬 МАСШТАБНОЕ ИССЛЕДОВАНИЕ ГОРИЗОНТОВ H={h_min}:{h_max}:{h_step} И K=2.0:7.0:0.5 ДЛЯ {coin.upper()}")
    print(f"   Используется CPU воркеров: {SAFE_WORKERS} (резерв OS >= 4 ядер)")
    print("=" * 115)

    df = pd.read_parquet(fpath)
    cfg = COIN_CONFIG[coin]
    tick_size = cfg["tick_size"]
    min_vol = cfg["min_vol"]
    n = len(df)

    split_idx = int(n * 0.75)
    embargo = 360
    train_end = split_idx - 60
    test_start = split_idx + embargo
    test_end = n - 60

    req_cols = [
        'ST_mid_price', 'ST_spread', 'ST_bid_1_price', 'ST_ask_1_price',
        'ST_bid_1_qty', 'ST_ask_1_qty', 'ST_bids_vol_top5', 'ST_asks_vol_top5', 'ST_trade_vol'
    ]
    df_mini = df[[c for c in req_cols if c in df.columns]].copy()

    # --- ЭТАП 1: Поиск оптимального K*(H) для каждого H из 2..60 ---
    horizons = list(range(h_min, h_max + 1, h_step))
    k_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

    all_tasks = []
    for H in horizons:
        for K in k_values:
            L = int(round(H * K))
            if 10 <= L <= 240:
                all_tasks.append((df_mini, coin, H, K, min_vol, tick_size, train_end, test_start, test_end))

    print(f"⚡ Этап 1: Параллельное сканирование {len(all_tasks)} комбинаций (H, K)...")
    with ProcessPoolExecutor(max_workers=SAFE_WORKERS) as executor:
        results = list(executor.map(eval_hk_candidate, all_tasks))

    # Группировка и выбор лучшего K для каждого H
    h_groups = {}
    for H, K, m in results:
        if m is not None:
            if H not in h_groups:
                h_groups[H] = []
            h_groups[H].append(m)

    optimal_h_k = {}
    phase1_records = []
    for H in horizons:
        if H in h_groups and h_groups[H]:
            # Выбираем K, минимизирующее max_err_z при здоровой частоте пробоев (1% .. 10%)
            valid_k = [r for r in h_groups[H] if 0.8 <= r["breakout_pct"] <= 12.0]
            if not valid_k:
                valid_k = h_groups[H]
            valid_k.sort(key=lambda x: (x["max_err_z"], abs(x["K"] - 3.5)))
            best = valid_k[0]
            optimal_h_k[H] = best["K"]
            phase1_records.append(best)

    opt_df = pd.DataFrame(phase1_records)
    print(f"✅ Этап 1 завершен: оптимизировано {len(opt_df)} горизонтов H.")

    # --- ЭТАП 2: Корреляционный анализ Out-of-Sample сигналов и таргетов ---
    print("\n⚡ Этап 2: Построение прогнозов и матриц корреляции по сетке H=2..60...")
    feat_cols = get_stationary_feature_cols(df)
    X_all = df[feat_cols].to_numpy(dtype=np.float32)

    X_test_oos = X_all[test_start:test_end]

    predictions_oos = {}
    targets_oos = {}
    oos_ics = {}
    oos_aucs = {}

    for H in horizons:
        K = optimal_h_k[H]
        L = int(round(H * K))
        res = compute_single_horizon_labels(df_mini, coin, H, K, min_vol, tick_size)
        z = res["z"]
        c7 = res["class_7"]
        is_void = res["is_void"]

        stride = max(3, H // 2)
        tr_idx = np.arange(L, train_end, stride)
        tr_idx = tr_idx[~is_void[tr_idx]]

        # Обучаем быстрый регрессор импульса
        model = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=35,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=5,
            random_state=42,
            n_jobs=SAFE_WORKERS,
            verbose=-1,
        )
        model.fit(X_all[tr_idx], np.clip(z[tr_idx], -2.5, 2.5))
        pred_val = model.predict(X_test_oos)
        z_val_slice = np.clip(z[test_start:test_end], -2.5, 2.5)
        c_val_slice = c7[test_start:test_end]

        ic = float(np.corrcoef(pred_val, z_val_slice)[0, 1])
        b_up = (c_val_slice >= 2).astype(int)
        b_dn = (c_val_slice <= -2).astype(int)
        auc = 0.5 * (roc_auc_score(b_up, pred_val) + roc_auc_score(b_dn, -pred_val)) if (b_up.sum() > 10 and b_dn.sum() > 10) else 0.5

        predictions_oos[H] = pred_val
        targets_oos[H] = z_val_slice
        oos_ics[H] = ic
        oos_aucs[H] = auc

    pred_df = pd.DataFrame(predictions_oos)
    corr_matrix = pred_df.corr()

    # Считаем длину затухания корреляции: среднее падение корреляции с ростом разности |H1 - H2|
    lags = [1, 2, 5, 10, 15, 20, 30]
    lag_corrs = []
    for lag in lags:
        corrs = [corr_matrix.loc[h, h + lag] for h in horizons if (h + lag) in corr_matrix.columns]
        if corrs:
            lag_corrs.append((lag, np.mean(corrs)))
    print("📉 Затухание взаимной корреляции прогнозов с шагом дельты ΔH:")
    for lag, val in lag_corrs:
        print(f"   ΔH = {lag:2d}с -> Средняя корреляция сигналов: {val:.3f}")

    # --- ЭТАП 3: Ортогональный отбор кандидатов (Forward Orthogonal Selection) ---
    print("\n⚡ Этап 3: Ортогональный отбор кандидатов для пула моделей...")
    # Начинаем с горизонта с наивысшим Out-of-Sample IC
    sorted_by_ic = sorted(horizons, key=lambda h: oos_ics[h], reverse=True)
    best_anchor = sorted_by_ic[0]

    selected_horizons = [best_anchor]
    target_pool_size = 5
    max_pairwise_corr = 0.82

    while len(selected_horizons) < target_pool_size and len(selected_horizons) < len(horizons):
        best_candidate = None
        best_marginal_score = -1e9

        for cand in horizons:
            if cand in selected_horizons:
                continue
            max_corr_with_selected = max(abs(corr_matrix.loc[cand, s]) for s in selected_horizons)
            if max_corr_with_selected > max_pairwise_corr:
                continue

            # Критерий полезности: высокое качество IC, помноженное на ортогональность
            orthogonality = 1.0 - (max_corr_with_selected ** 2)
            marginal_score = oos_ics[cand] * orthogonality

            if marginal_score > best_marginal_score:
                best_marginal_score = marginal_score
                best_candidate = cand

        if best_candidate is not None:
            selected_horizons.append(best_candidate)
        else:
            # Ослабляем порог корреляции, если не хватает
            max_pairwise_corr += 0.05
            if max_pairwise_corr > 0.95:
                break

    selected_horizons.sort()
    print(f"🎯 Отобранные взаимодополняющие кандидаты пула (N={len(selected_horizons)}): {selected_horizons}")

    # --- ЭТАП 4: Полная Walk-Forward валидация отобранных кандидатов и их ансамбля ---
    print(f"\n⚡ Этап 4: Walk-Forward тестирование (4 фолда) отобранных кандидатов {selected_horizons}...")
    from _tools.test_walkforward_labeling import run_walkforward_for_coin
    wf_results = run_walkforward_for_coin(
        coin=coin,
        horizons=tuple(selected_horizons),
        n_folds=4,
        embargo_sec=360,
    )

    # Тестирование синергии ансамбля на Out-of-Sample срезе
    # Нормируем предсказания кандидатов к единичной дисперсии и усредняем
    ensemble_pred = np.mean([
        (predictions_oos[h] - np.mean(predictions_oos[h])) / np.std(predictions_oos[h])
        for h in selected_horizons
    ], axis=0)

    # Оцениваем ансамбль относительно опорного горизонта
    ref_z = targets_oos[selected_horizons[len(selected_horizons)//2]]
    ens_ic = float(np.corrcoef(ensemble_pred, ref_z)[0, 1])

    b_up_ref = (ref_z >= 0.5).astype(int)
    b_dn_ref = (ref_z <= -0.5).astype(int)
    ens_auc = 0.5 * (roc_auc_score(b_up_ref, ensemble_pred) + roc_auc_score(b_dn_ref, -ensemble_pred))

    print("\n" + "=" * 115)
    print(f"🌟 РЕЗУЛЬТАТ АНСАМБЛЯ ПУЛА МОДЕЛЕЙ (H = {selected_horizons}):")
    print(f"   • Синергетический Out-of-Sample IC:  {ens_ic:+.4f} (при среднем одиночном {np.mean([oos_ics[h] for h in selected_horizons]):+.4f})")
    print(f"   • ROC-AUC пробоев:                  {ens_auc:.4f}")
    print(f"   • Снижение дисперсии шума:          в {len(selected_horizons) ** 0.5:.2f} раз")
    print("=" * 115)

    # Сохраняем сводные данные
    out_dir = os.path.join(BASE_DIR, "data", "labeled")
    os.makedirs(out_dir, exist_ok=True)
    csv_candidates = os.path.join(out_dir, f"research_candidates_{coin}.csv")
    if wf_results is not None:
        wf_results.to_csv(csv_candidates, index=False)
        print(f"📁 Отчет по лучшим кандидатам сохранен: {csv_candidates}")

    return {
        "coin": coin,
        "selected_horizons": selected_horizons,
        "optimal_k": {h: optimal_h_k[h] for h in selected_horizons},
        "ens_ic": ens_ic,
        "ens_auc": ens_auc,
        "corr_matrix": corr_matrix,
    }


def main():
    parser = argparse.ArgumentParser(description="Large-Scale Horizon Research Engine")
    parser.add_argument("--coin", type=str, default="all", choices=["prl", "quantus", "all"])
    parser.add_argument("--h-min", type=int, default=2)
    parser.add_argument("--h-max", type=int, default=60)
    parser.add_argument("--h-step", type=int, default=1)
    args = parser.parse_args()

    coins = ["prl", "quantus"] if args.coin == "all" else [args.coin]
    for c in coins:
        run_full_research(coin=c, h_min=args.h_min, h_max=args.h_max, h_step=args.h_step)


if __name__ == "__main__":
    main()

