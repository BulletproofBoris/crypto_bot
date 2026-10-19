#!/usr/bin/env python3
"""
Экспериментальный поиск оптимальных комбинаций (H, K, L=K*H) и оптимального числа моделей M в ансамбле.
Оценивает каждую пару (H, K) по 4 критериям на отложенной выборке (Validation):
1. Стационарность разметки: Max_Err_z = max(Err_z(+2), Err_z(-2)) между Train и Val.
2. Предсказательная сила (Out-of-Sample IC): корреляция прогноза модели с реальным будущим импульсом на Val.
3. Точность детекции пробоев (Out-of-Sample ROC-AUC для |class_7| >= 2).
4. Информационный вклад в ансамбль при M = 1..6 моделях (Forward Orthogonal Selection).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.data_labeler import COIN_CONFIG


def fast_compute_z_and_classes(df, coin, H, K):
    """Быстрый векторный расчёт z, class_7 и метрик сходимости для пары (H, K)."""
    cfg = COIN_CONFIG[coin]
    tick_size = cfg["tick_size"]
    min_vol = cfg["min_vol"]

    n = len(df)
    L = int(H * K)
    split_idx = int(n * 0.8)

    mid = df["ST_mid_price"].values
    spread_ticks = df["ST_spread"].values / tick_size
    b1_p = df["ST_bid_1_price"].values
    a1_p = df["ST_ask_1_price"].values
    b1_q = df["ST_bid_1_qty"].values
    a1_q = df["ST_ask_1_qty"].values
    b5 = df["ST_bids_vol_top5"].values
    a5 = df["ST_asks_vol_top5"].values
    trade_vol = df["ST_trade_vol"].values

    dt_ticks = np.zeros(n, dtype=np.float32)
    dt_ticks[:-H] = np.round((mid[H:] - mid[:-H]) / tick_size)

    step_1s_ticks = np.zeros(n, dtype=np.float32)
    step_1s_ticks[1:] = (mid[1:] - mid[:-1]) / tick_size
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().fillna(0.0).values

    asks_L = pd.Series(a5).rolling(L, min_periods=L).mean().fillna(min_vol).values
    bids_L = pd.Series(b5).rolling(L, min_periods=L).mean().fillna(min_vol).values
    a1_L = pd.Series(a1_q).rolling(L, min_periods=L).median().fillna(1.0).values
    b1_L = pd.Series(b1_q).rolling(L, min_periods=L).median().fillna(1.0).values

    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 0, (b1_p * a1_q + a1_p * b1_q) / np.maximum(total_q1, 1.0), mid)
    micro_offset = (microprice - mid) / tick_size

    t_wave = max(3, H // 3)
    v_wave = (pd.Series(micro_offset) - pd.Series(micro_offset).shift(t_wave)).fillna(0.0).values / t_wave
    d_vol = (pd.Series(vol_L) - pd.Series(vol_L).shift(t_wave)).fillna(0.0).values / t_wave

    q1_up = np.clip(a1_L / np.maximum(a1_q, 1.0), 0.2, 5.0) ** 0.15
    q1_dn = np.clip(b1_L / np.maximum(b1_q, 1.0), 0.2, 5.0) ** 0.15

    M_up_base = (spread_ticks + vol_L * np.sqrt(H)) * (np.clip(asks_L / np.maximum(a5, min_vol), 0.2, 5.0) ** 0.25) * q1_up
    M_dn_base = (spread_ticks + vol_L * np.sqrt(H)) * (np.clip(bids_L / np.maximum(b5, min_vol), 0.2, 5.0) ** 0.25) * q1_dn

    M_up = np.maximum(0.5, M_up_base * (1.0 + np.maximum(0.0, v_wave)) * (1.0 + np.maximum(0.0, d_vol)))
    M_dn = np.maximum(0.5, M_dn_base * (1.0 + np.maximum(0.0, -v_wave)) * (1.0 + np.maximum(0.0, d_vol)))

    z = np.zeros(n, dtype=np.float32)
    pos = dt_ticks > 0
    neg = dt_ticks < 0
    z[pos] = dt_ticks[pos] / M_up[pos]
    z[neg] = dt_ticks[neg] / M_dn[neg]

    class_7 = np.zeros(n, dtype=np.int8)
    class_7[pos & (z < 0.50)] = 1
    class_7[pos & (z >= 0.50) & (z < 1.00)] = 2
    class_7[pos & (z >= 1.00)] = 3
    class_7[neg & (z > -0.50)] = -1
    class_7[neg & (z <= -0.50) & (z > -0.95)] = -2
    class_7[neg & (z <= -0.95)] = -3

    trade_vol_H = pd.Series(trade_vol).rolling(H, min_periods=1).sum().shift(-H).fillna(0.0).values
    is_void = (np.abs(dt_ticks) >= 5) & (trade_vol_H == 0)

    return z, class_7, dt_ticks, is_void, split_idx


def get_stationary_feature_cols(df):
    """Отбирает стационарные микроструктурные признаки (без абсолютных цен стакана)."""
    drop_patterns = ("_price", "mid_price", "eff_buy_vwap", "eff_sell_vwap", "macro_btc_mid", "macro_btc_last", "macro_near_mid", "macro_near_last", "RHEA_price")
    cols = []
    for c in df.columns:
        if any(p in c for p in drop_patterns):
            continue
        if c.startswith("ST_bid_") or c.startswith("ST_ask_") or c.startswith("BO_bid_") or c.startswith("BO_ask_"):
            if c.endswith("_qty") and not any(f"_{k}_qty" in c for k in (1, 2, 3)):
                continue
        cols.append(c)
    return cols


def run_coin_experiment(coin):
    fpath = os.path.join(BASE_DIR, "data", "features", f"{coin}_1s_features.parquet")
    df = pd.read_parquet(fpath)
    feat_cols = get_stationary_feature_cols(df)
    X_all = df[feat_cols].to_numpy(dtype=np.float32)

    n = len(df)
    split_idx = int(n * 0.8)
    L_max = split_idx // 4000

    print("\n" + "=" * 115)
    print(f"🔬 ЭКСПЕРИМЕНТАЛЬНЫЙ ПОИСК ОПТИМАЛЬНЫХ (H, K, L) И ЧИСЛА МОДЕЛЕЙ ДЛЯ {coin.upper()} (Признаков: {len(feat_cols)}, L_max={L_max}с)")
    print("=" * 115)

    # Расширенная сетка горизонтов H и коэффициентов K = L/H
    H_grid = [3, 5, 8, 10, 12, 15, 20, 25, 30]
    K_grid = [2, 3, 4, 5, 6]

    records = []
    val_preds_dict = {}
    val_targets_dict = {}

    # Индексы валидации (единый срез с отступом 180с от границы Train/Val против утечки)
    val_start = split_idx + 180
    val_end = n - 60
    X_val = X_all[val_start:val_end]

    for H in H_grid:
        for K in K_grid:
            L = H * K
            if L > max(L_max + 15, 90):
                continue

            z, class_7, dt_ticks, is_void, _ = fast_compute_z_and_classes(df, coin, H, K)

            tr_s, tr_e = L, split_idx - H
            z_tr, c_tr = z[tr_s:tr_e], class_7[tr_s:tr_e]
            z_va, c_va = z[val_start:val_end], class_7[val_start:val_end]
            dt_va = dt_ticks[val_start:val_end]

            m_tr_p2, m_va_p2 = (c_tr == 2), (c_va == 2)
            m_tr_m2, m_va_m2 = (c_tr == -2), (c_va == -2)
            if m_tr_p2.sum() < 30 or m_va_p2.sum() < 15 or m_tr_m2.sum() < 30 or m_va_m2.sum() < 15:
                continue

            err_p2 = abs(abs(z_va[m_va_p2].mean()) - abs(z_tr[m_tr_p2].mean())) / abs(z_tr[m_tr_p2].mean()) * 100.0
            err_m2 = abs(abs(z_va[m_va_m2].mean()) - abs(z_tr[m_tr_m2].mean())) / abs(z_tr[m_tr_m2].mean()) * 100.0
            max_err_z = max(err_p2, err_m2)

            # Быстрое обучение регрессора импульса z (шаг прореживания max(2, H//2) убирает автокорреляцию соседей)
            stride = max(3, H // 2)
            valid_tr_idx = np.arange(tr_s, tr_e, stride)
            valid_tr_idx = valid_tr_idx[~is_void[valid_tr_idx]]

            X_tr_sub = X_all[valid_tr_idx]
            y_tr_sub = np.clip(z[valid_tr_idx], -2.5, 2.5)

            model = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=60,
                learning_rate=0.08,
                num_leaves=31,
                max_depth=5,
                subsample_for_bin=30000,
                random_state=42,
                n_jobs=4,
                verbose=-1,
            )
            model.fit(X_tr_sub, y_tr_sub)
            pred_z_val = model.predict(X_val)

            # Метрики на Out-of-Sample Validation
            ic_val = float(np.corrcoef(pred_z_val, np.clip(z_va, -2.5, 2.5))[0, 1])
            breakout_up = (c_va >= 2).astype(int)
            break_dn = (c_va <= -2).astype(int)
            auc_up = roc_auc_score(breakout_up, pred_z_val) if breakout_up.sum() > 10 else 0.5
            auc_dn = roc_auc_score(break_dn, -pred_z_val) if break_dn.sum() > 10 else 0.5
            mean_auc = 0.5 * (auc_up + auc_dn)

            # Средний реальный выигрыш в тиках на Топ-1% сигналов модели на Val
            q99 = np.quantile(pred_z_val, 0.99)
            q01 = np.quantile(pred_z_val, 0.01)
            top1_ticks = 0.5 * (dt_va[pred_z_val >= q99].mean() - dt_va[pred_z_val <= q01].mean())

            # Интегральный индекс качества: предсказуемость (IC + AUC) со штрафом за нестационарность Err_z > 5%
            stability_mult = 1.0 if max_err_z <= 5.0 else np.exp(-0.08 * (max_err_z - 5.0))
            composite_score = (ic_val * 100.0 + (mean_auc - 0.5) * 100.0) * stability_mult

            key = f"H={H}s,K={K}(L={L}s)"
            val_preds_dict[key] = pred_z_val / (np.std(pred_z_val) + 1e-8)
            val_targets_dict[key] = np.clip(z_va, -2.5, 2.5)

            records.append({
                "coin": coin.upper(), "H": H, "K": K, "L": L,
                "in_Lmax": L <= L_max,
                "max_err_z": max_err_z,
                "ic_val": ic_val,
                "auc_val": mean_auc,
                "top1_ticks": top1_ticks,
                "score": composite_score,
                "key": key,
            })

    res_df = pd.DataFrame(records)
    return res_df, val_preds_dict, val_targets_dict


def analyze_and_select_ensemble(res_df, val_preds_dict, val_targets_dict, coin):
    """
    1. Находит лучший K*(H) для каждого горизонта H.
    2. Строит матрицу корреляций прогнозов между горизонтами.
    3. Жадно (Forward Orthogonal Selection) собирает ансамбль из M = 1..6 моделей
       и измеряет прирост Out-of-Sample информативности (Effective Information Gain & Multi-Horizon IC).
    """
    print(f"\n📌 1. ЛУЧШЕЕ СООТНОШЕНИЕ K = L/H ДЛЯ КАЖДОГО ГОРИЗОНТА H ({coin.upper()}):")
    print(f"{'H (с)':<6} | {'Лучший K':<8} | {'L (с)':<6} | {'L<=L_max':<8} | {'Max Err(z)':<10} | {'OOS IC':<8} | {'OOS AUC':<8} | {'Top1% сдвиг':<12} | {'Индекс Score':<10}")
    print("-" * 105)

    best_per_h = []
    for H, grp in res_df.groupby("H"):
        # Предпочитаем кандидатов внутри L_max и с Err_z <= 5%, сортируя по Score
        valid_grp = grp[grp["in_Lmax"] & (grp["max_err_z"] <= 5.5)]
        if valid_grp.empty:
            valid_grp = grp
        best_row = valid_grp.sort_values("score", ascending=False).iloc[0]
        best_per_h.append(best_row)
        l_ok = "ДА" if best_row["in_Lmax"] else "НЕТ(>Lmax)"
        print(
            f"{int(best_row['H']):<6d} | {int(best_row['K']):<8d} | {int(best_row['L']):<6d} | {l_ok:<8} | "
            f"{best_row['max_err_z']:8.2f}%  | {best_row['ic_val']:+7.4f} | {best_row['auc_val']:7.4f}  | "
            f"{best_row['top1_ticks']:+8.2f} тик | {best_row['score']:8.2f}"
        )

    best_h_df = pd.DataFrame(best_per_h).reset_index(drop=True)

    # Матрица корреляций прогнозов между лучшими моделями разных H
    keys = best_h_df["key"].tolist()
    pred_matrix = np.column_stack([val_preds_dict[k] for k in keys])
    corr_mat = np.corrcoef(pred_matrix, rowvar=False)

    # Эталонный композитный будущий импульс по шкале горизонтов (средний нормированный z по всем валидным H <= L_max)
    valid_h_df = best_h_df[best_h_df["in_Lmax"]].copy()
    if valid_h_df.empty:
        valid_h_df = best_h_df.copy()
    valid_keys = valid_h_df["key"].tolist()
    composite_target = np.mean([val_targets_dict[k] for k in valid_keys], axis=0)

    print(f"\n📌 2. ПОШАГОВЫЙ ОТБОР ЧИСЛА МОДЕЛЕЙ В АНСАМБЛЕ M = 1..{min(6, len(valid_h_df))} ({coin.upper()}):")
    print(f"{'Шаг M':<6} | {'Добавлена модель (H, K, L)':<26} | {'Макс. корр с уже взятыми':<25} | {'OOS IC ансамбля':<16} | {'Прирост ΔIC':<12} | {'Эфф. число степеней свободы N_eff_models'}")
    print("-" * 120)

    selected_idx = []
    cand_rows = valid_h_df.to_dict("records")
    prev_ic = 0.0

    for m_step in range(1, min(6, len(cand_rows)) + 1):
        best_gain = -1e9
        best_cand_i = None
        best_ens_ic = 0.0
        best_max_corr = 0.0

        for i in range(len(cand_rows)):
            if i in selected_idx:
                continue
            trial_idx = selected_idx + [i]
            trial_keys = [cand_rows[j]["key"] for j in trial_idx]
            ens_pred = np.mean([val_preds_dict[k] for k in trial_keys], axis=0)
            ens_ic = float(np.corrcoef(ens_pred, composite_target)[0, 1])

            if not selected_idx:
                max_c = 0.0
                # На первом шаге берём модель с максимальным индивидуальным Score и IC
                obj = ens_ic + 0.002 * cand_rows[i]["score"]
            else:
                # Штрафуем за высокую корреляцию (>0.80) с уже выбранными моделями ансамбля
                corrs = [
                    float(np.corrcoef(val_preds_dict[cand_rows[i]["key"]], val_preds_dict[cand_rows[j]["key"]])[0, 1])
                    for j in selected_idx
                ]
                max_c = max(corrs)
                redundancy_penalty = max(0.0, max_c - 0.78) * 0.15
                obj = ens_ic - redundancy_penalty

            if obj > best_gain:
                best_gain = obj
                best_cand_i = i
                best_ens_ic = ens_ic
                best_max_corr = max_c

        selected_idx.append(best_cand_i)
        sel_keys = [cand_rows[j]["key"] for j in selected_idx]
        sub_mat = np.column_stack([val_preds_dict[k] for k in sel_keys])
        sub_corr = np.corrcoef(sub_mat, rowvar=False) if len(sel_keys) > 1 else np.array([[1.0]])
        eigvals = np.linalg.eigvalsh(sub_corr)
        # Effective number of independent models (Participation Ratio / Shannon Effective Rank)
        p_eig = np.maximum(eigvals, 1e-9) / np.sum(np.maximum(eigvals, 1e-9))
        eff_rank = float(np.exp(-np.sum(p_eig * np.log(p_eig))))

        delta_ic = ((best_ens_ic - prev_ic) / abs(prev_ic) * 100.0) if prev_ic > 0 else 100.0
        prev_ic = best_ens_ic
        print(
            f"M = {m_step:<2d} | {cand_rows[best_cand_i]['key']:<26} | {best_max_corr:23.3f}  | "
            f"{best_ens_ic:+15.4f} | {delta_ic:+10.2f}% | {eff_rank:6.2f} из {m_step}"
        )


def main():
    all_dfs = []
    for coin in ("prl", "quantus"):
        res_df, val_preds_dict, val_targets_dict = run_coin_experiment(coin)
        analyze_and_select_ensemble(res_df, val_preds_dict, val_targets_dict, coin)
        all_dfs.append(res_df)

    full_res = pd.concat(all_dfs, ignore_index=True)
    out_csv = os.path.join(BASE_DIR, "data", "labeled", "experiment_hk_oos_results.csv")
    full_res.to_csv(out_csv, index=False)
    print(f"\n✅ Полная таблица эксперимента ({len(full_res)} комбинаций H, K) сохранена в: {out_csv}")


if __name__ == "__main__":
    main()

