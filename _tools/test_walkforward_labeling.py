#!/usr/bin/env python3
"""
Walk-Forward Validation Engine for Order-Flow Microstructure Labeling
=====================================================================
Проверяет гипотезу стационарности динамической разметки и качество таргетов
на последовательных непересекающихся Out-of-Sample окнах (Walk-Forward).

Оценивает:
1. Стационарность импульса z: Err_z (+2) и Err_z (-2) между In-Sample и Out-of-Sample.
2. Предсказательную силу: Out-of-Sample Information Coefficient (IC) и ROC-AUC (пробои |class_7| >= 2).
3. Масштабируемость горизонтов H: от микро-импульса (5s) до волны (15-20s) и тренда (30-60s).
4. Защиту от утечек: Purge (H шагов) + Embargo (360с на память банка фильтров W=300s).
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from colorama import init, Fore, Style

warnings.filterwarnings("ignore")
init(autoreset=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.data_labeler import compute_single_horizon_labels, COIN_CONFIG
from _tools.experiment_hk_selection import get_stationary_feature_cols


def run_walkforward_for_coin(
    coin: str,
    horizons=(5, 10, 15, 20, 30, 60),
    n_folds: int = 4,
    embargo_sec: int = 360,
    min_train_pct: float = 0.40,
):
    coin = coin.lower()
    fpath = os.path.join(BASE_DIR, "data", "features", f"{coin}_1s_features.parquet")
    if not os.path.exists(fpath):
        print(f"{Fore.RED}❌ Файл {fpath} не найден!{Style.RESET_ALL}")
        return None

    cfg = COIN_CONFIG.get(coin, COIN_CONFIG["prl"])
    tick_size = cfg["tick_size"]
    min_vol = cfg["min_vol"]

    df = pd.read_parquet(fpath)
    feat_cols = get_stationary_feature_cols(df)
    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    n = len(df)

    # Построение временных срезов Walk-Forward (Expanding Window)
    # От min_train_pct до 1.0 на n_folds шагов
    split_pcts = np.linspace(min_train_pct, 1.0, n_folds + 1)
    folds = []
    for i in range(len(split_pcts) - 1):
        tr_end_idx = int(n * split_pcts[i])
        te_end_idx = int(n * split_pcts[i + 1])
        folds.append((tr_end_idx, te_end_idx))

    print("\n" + "=" * 110)
    print(f"📊 WALK-FORWARD ВАЛИДАЦИЯ РАЗМЕТКИ ДЛЯ {coin.upper()}")
    print(f"   Строк: {n:,} | Период: {df.index.min()} -> {df.index.max()}")
    print(f"   Фолдов: {n_folds} (Expanding Window) | Embargo: {embargo_sec}с | Признаков модели: {len(feat_cols)}")
    print("=" * 110)

    summary_records = []
    horizon_predictions = {H: [] for H in horizons}
    test_actuals = {H: [] for H in horizons}

    for H in horizons:
        # Для коротких горизонтов оптимален K=4, для длинных K=3
        K = 4 if H <= 15 else 3
        L = H * K

        res = compute_single_horizon_labels(
            df, coin=coin, H=H, K=K, min_vol=min_vol, tick_size=tick_size
        )
        z_all = res["z"]
        c7_all = res["class_7"]
        is_void = res["is_void"]
        dt_ticks = res["dt_ticks"]

        fold_ics = []
        fold_aucs = []
        fold_err_p2 = []
        fold_err_m2 = []
        fold_top1_ticks = []

        print(f"\n🔍 Тест горизонта H = {H:2d}с (L = {L:3d}с, K = {K}):")

        for fold_idx, (raw_tr_end, te_end) in enumerate(folds):
            train_end = raw_tr_end - H
            test_start = raw_tr_end + embargo_sec

            if test_start >= te_end - H:
                continue

            stride = max(3, H // 2)
            tr_idx = np.arange(L, train_end, stride)
            tr_idx = tr_idx[~is_void[tr_idx]]

            te_idx = np.arange(test_start, te_end - H)
            te_idx = te_idx[~is_void[te_idx]]

            if len(tr_idx) < 500 or len(te_idx) < 200:
                continue

            z_tr = z_all[tr_idx]
            c_tr = c7_all[tr_idx]
            z_te = z_all[te_idx]
            c_te = c7_all[te_idx]
            dt_te = dt_ticks[te_idx]

            # 1. Стационарность квантилей: Err_z (+2) и Err_z (-2)
            m_tr_p2 = (c_tr == 2)
            m_te_p2 = (c_te == 2)
            z_tr_p2 = z_tr[m_tr_p2].mean() if m_tr_p2.sum() > 5 else 1.0
            z_te_p2 = z_te[m_te_p2].mean() if m_te_p2.sum() > 5 else 1.0
            err_p2 = abs(abs(z_te_p2) - abs(z_tr_p2)) / abs(z_tr_p2) * 100.0

            m_tr_m2 = (c_tr == -2)
            m_te_m2 = (c_te == -2)
            z_tr_m2 = z_tr[m_tr_m2].mean() if m_tr_m2.sum() > 5 else -1.0
            z_te_m2 = z_te[m_te_m2].mean() if m_te_m2.sum() > 5 else -1.0
            err_m2 = abs(abs(z_te_m2) - abs(z_tr_m2)) / abs(z_tr_m2) * 100.0

            fold_err_p2.append(err_p2)
            fold_err_m2.append(err_m2)

            # 2. Обучение быстрой модели LightGBM на Train
            model = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=45,
                learning_rate=0.08,
                num_leaves=31,
                max_depth=5,
                subsample_for_bin=25000,
                random_state=42,
                n_jobs=4,
                verbose=-1,
            )
            model.fit(X_all[tr_idx], np.clip(z_tr, -2.5, 2.5))
            pred_te = model.predict(X_all[te_idx])

            # 3. Метрики Out-of-Sample
            ic = float(np.corrcoef(pred_te, np.clip(z_te, -2.5, 2.5))[0, 1])
            fold_ics.append(ic)

            b_up = (c_te >= 2).astype(int)
            b_dn = (c_te <= -2).astype(int)
            auc_up = roc_auc_score(b_up, pred_te) if b_up.sum() > 10 else 0.5
            auc_dn = roc_auc_score(b_dn, -pred_te) if b_dn.sum() > 10 else 0.5
            mean_auc = 0.5 * (auc_up + auc_dn)
            fold_aucs.append(mean_auc)

            # Доходность на топ-2% экстремальных сигналов модели
            q98 = np.quantile(pred_te, 0.98)
            q02 = np.quantile(pred_te, 0.02)
            top_gain_ticks = 0.5 * (dt_te[pred_te >= q98].mean() - dt_te[pred_te <= q02].mean())
            fold_top1_ticks.append(top_gain_ticks)

            print(
                f"   Fold {fold_idx + 1}/{n_folds} (Train={len(tr_idx):,} -> Test={len(te_idx):,}): "
                f"IC = {ic:+.4f} | AUC = {mean_auc:.4f} | Err_z(+2) = {err_p2:4.1f}% | Err_z(-2) = {err_m2:4.1f}% | Top2% Gain = {top_gain_ticks:+.2f} тиков"
            )

        mean_ic = np.mean(fold_ics)
        mean_auc = np.mean(fold_aucs)
        mean_err_p2 = np.mean(fold_err_p2)
        mean_err_m2 = np.mean(fold_err_m2)
        mean_top_gain = np.mean(fold_top1_ticks)
        max_err = max(mean_err_p2, mean_err_m2)

        # Оценка пригодности в пул моделей
        if mean_ic >= 0.25 and mean_auc >= 0.70 and max_err < 5.0:
            rating = f"{Fore.GREEN}⭐ ИДЕАЛЬНО (Core Pool){Style.RESET_ALL}"
        elif mean_ic >= 0.20 and mean_auc >= 0.68 and max_err < 8.0:
            rating = f"{Fore.CYAN}✅ ОТЛИЧНО (Satellite Pool){Style.RESET_ALL}"
        elif mean_ic >= 0.15 and max_err < 10.0:
            rating = f"{Fore.YELLOW}⚠️ ПРИЕМЛЕМО (Macro Aux){Style.RESET_ALL}"
        else:
            rating = f"{Fore.RED}❌ СЛАБО{Style.RESET_ALL}"

        summary_records.append({
            "coin": coin.upper(),
            "H_sec": H,
            "K_ratio": K,
            "L_sec": L,
            "Mean_OOS_IC": mean_ic,
            "Mean_OOS_AUC": mean_auc,
            "Mean_Err_p2": mean_err_p2,
            "Mean_Err_m2": mean_err_m2,
            "Max_Err_z": max_err,
            "Top2pct_Gain_Ticks": mean_top_gain,
            "Rating": rating,
        })

    sum_df = pd.DataFrame(summary_records)
    print("\n" + "=" * 110)
    print(f"🏆 СВОДКА WALK-FORWARD ТЕСТИРОВАНИЯ ДЛЯ {coin.upper()}")
    print("=" * 110)
    print(
        f"{'H (с)':<6} | {'K':<3} | {'L (с)':<5} | {'Mean OOS IC':<12} | {'Mean ROC-AUC':<12} | "
        f"{'Err_z(+2)':<10} | {'Err_z(-2)':<10} | {'Top2% Gain':<11} | {'Статус для пула'}"
    )
    print("-" * 110)
    for _, r in sum_df.iterrows():
        print(
            f"{r['H_sec']:<6} | {r['K_ratio']:<3} | {r['L_sec']:<5} | {r['Mean_OOS_IC']:+11.4f}  | {r['Mean_OOS_AUC']:11.4f}  | "
            f"{r['Mean_Err_p2']:9.1f}% | {r['Mean_Err_m2']:9.1f}% | {r['Top2pct_Gain_Ticks']:+10.2f}т | {r['Rating']}"
        )
    print("-" * 110)
    return sum_df


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Microstructure Labeling Validation")
    parser.add_argument("--coin", type=str, default="all", choices=["prl", "quantus", "all"])
    parser.add_argument("--horizons", type=str, default="5,10,15,20,30,60")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--embargo", type=int, default=360)
    parser.add_argument("--out-csv", type=str, default="data/labeled/walkforward_summary.csv")
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    coins = ["prl", "quantus"] if args.coin == "all" else [args.coin]

    all_dfs = []
    for c in coins:
        res = run_walkforward_for_coin(
            coin=c,
            horizons=horizons,
            n_folds=args.folds,
            embargo_sec=args.embargo,
        )
        if res is not None:
            all_dfs.append(res)

    if all_dfs:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Убираем цветные ANSI-коды перед сохранением в CSV
        clean_df = final_df.copy()
        clean_df["Rating"] = clean_df["Rating"].astype(str).str.replace(r"\x1b\[[0-9;]*m", "", regex=True)
        clean_df.to_csv(args.out_csv, index=False)
        print(f"\n📁 Результаты Walk-Forward сохранены: {args.out_csv}")


if __name__ == "__main__":
    main()

