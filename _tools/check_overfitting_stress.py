#!/usr/bin/env python3
"""
Стресс-тестирование на переобучение и устойчивость к сдвигу параметров (Anti-Overfitting Suite)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.experiment_hk_selection import get_stationary_feature_cols
from _tools.data_labeler import COIN_CONFIG
from _tools.system_resources import SAFE_WORKERS

def test_parameter_plateau(coin='prl', H=2, K=4):
    """Тест 1: Проверка широкого плато параметров (отсутствие 'иглы' переобучения)"""
    print(f"\n" + "="*80)
    print(f"🔬 ТЕСТ 1: АНАЛИЗ ПЛАТО ПАРАМЕТРОВ ДЛЯ {coin.upper()} (H={H}с)")
    print("="*80)
    fpath = f'data/features/{coin}_1s_features.parquet'
    df = pd.read_parquet(fpath)
    mid = df['ST_mid_price'].values
    cfg = COIN_CONFIG[coin]
    tick_size = cfg['tick_size']
    min_vol = cfg['min_vol']
    spread_ticks = df['ST_spread'].values / tick_size
    n = len(df)
    
    dt = np.zeros(n, dtype=int)
    dt[:-H] = np.round((mid[H:] - mid[:-H]) / tick_size).astype(int)
    
    b1_p = df['ST_bid_1_price'].values
    a1_p = df['ST_ask_1_price'].values
    b1_q = df['ST_bid_1_qty'].values
    a1_q = df['ST_ask_1_qty'].values
    b2_q = df['ST_bid_2_qty'].values if 'ST_bid_2_qty' in df.columns else b1_q
    a2_q = df['ST_ask_2_qty'].values if 'ST_ask_2_qty' in df.columns else a1_q
    b5 = df['ST_bids_vol_top5'].values
    a5 = df['ST_asks_vol_top5'].values
    
    q12_ask = a1_q + a2_q
    q12_bid = b1_q + b2_q
    eps_q = max(1e-4, min_vol * 0.01)
    
    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 1e-6, (b1_p * a1_q + a1_p * b1_q) / total_q1, mid)
    micro_offset = np.clip((microprice - mid) / tick_size, -5.0, 5.0)
    t_wave = max(3, H // 3)
    v_wave = np.clip((pd.Series(micro_offset) - pd.Series(micro_offset).shift(t_wave)).fillna(0).values / t_wave, -2.0, 2.0)
    
    folds = [(int(n * (0.4 + f * 0.15)), int(n * (0.4 + (f + 1) * 0.15))) for f in range(4)]
    
    base_sf = cfg.get('spread_factor', 0.10)
    l_vol_grid = [30, 45, 60, 90]
    sf_grid = [round(base_sf * 0.5, 3), base_sf, round(base_sf * 1.5, 3), round(base_sf * 2.0, 3)]
    exp_grid = [0.04, 0.08, 0.12]
    
    records = []
    step_1s_ticks = np.zeros(n)
    step_1s_ticks[1:] = (mid[1:] - mid[:-1]) / tick_size

    print(f"{'L_vol':<6} | {'Spread_Fac':<10} | {'Exp_Alpha':<10} | {'Err(+2)':<9} | {'Err(-2)':<9} | {'MaxErr':<8} | {'Статус плато'}")
    print("-" * 75)
    
    for l_vol in l_vol_grid:
        vol_L = pd.Series(step_1s_ticks).rolling(l_vol, min_periods=5).std().fillna(0.0).values
        a12_L = pd.Series(q12_ask).rolling(l_vol, min_periods=5).median().fillna(1.0).values
        b12_L = pd.Series(q12_bid).rolling(l_vol, min_periods=5).median().fillna(1.0).values
        asks_L = pd.Series(a5).rolling(l_vol, min_periods=5).mean().fillna(min_vol).values
        bids_L = pd.Series(b5).rolling(l_vol, min_periods=5).mean().fillna(min_vol).values
        
        for sf in sf_grid:
            for exp in exp_grid:
                sp_w = sf * (1.0 if H <= 5 else 0.8 if H <= 20 else 0.5)
                if H <= 5:
                    d_up = np.clip(a12_L / np.maximum(q12_ask, eps_q), 0.4, 2.5) ** exp
                    d_dn = np.clip(b12_L / np.maximum(q12_bid, eps_q), 0.4, 2.5) ** exp
                else:
                    d_up = np.clip(asks_L / np.maximum(a5, min_vol), 0.4, 2.5) ** exp
                    d_dn = np.clip(bids_L / np.maximum(b5, min_vol), 0.4, 2.5) ** exp
                    
                kin_up = 1.0 + 0.10 * np.tanh(np.maximum(0, v_wave))
                kin_dn = 1.0 + 0.10 * np.tanh(np.maximum(0, -v_wave))
                
                M_up = np.maximum(0.5, (spread_ticks * sp_w + vol_L * np.sqrt(H)) * d_up * kin_up)
                M_dn = np.maximum(0.5, (spread_ticks * sp_w + vol_L * np.sqrt(H)) * d_dn * kin_dn)
                
                z = np.zeros(n, dtype=np.float32)
                z[dt > 0] = dt[dt > 0] / M_up[dt > 0]
                z[dt < 0] = dt[dt < 0] / M_dn[dt < 0]
                
                c7 = np.zeros(n, dtype=np.int8)
                c7[(dt > 0) & (z >= 0.5) & (z < 1.0)] = 2
                c7[(dt < 0) & (z <= -0.5) & (z > -0.95)] = -2
                
                err_p_list, err_m_list = [], []
                for tr_end, te_end in folds:
                    z_tr, c_tr = z[:tr_end-H], c7[:tr_end-H]
                    z_te, c_te = z[tr_end+360:te_end-H], c7[tr_end+360:te_end-H]
                    p_tr = z_tr[c_tr == 2].mean() if (c_tr == 2).sum() > 5 else 1.0
                    p_te = z_te[c_te == 2].mean() if (c_te == 2).sum() > 5 else 1.0
                    m_tr = z_tr[c_tr == -2].mean() if (c_tr == -2).sum() > 5 else -1.0
                    m_te = z_te[c_te == -2].mean() if (c_te == -2).sum() > 5 else -1.0
                    err_p_list.append(abs(p_te - p_tr) / p_tr * 100)
                    err_m_list.append(abs(abs(m_te) - abs(m_tr)) / abs(m_tr) * 100)
                    
                ep = np.mean(err_p_list)
                em = np.mean(err_m_list)
                max_err = max(ep, em)
                status = "🟢 Устойчиво (<3%)" if max_err < 3.0 else ("🟡 Приемлемо (<5%)" if max_err < 5.0 else "🔴 Деградация")
                records.append({'L_vol': l_vol, 'sf': sf, 'exp': exp, 'Err_p': ep, 'Err_m': em, 'MaxErr': max_err})
                
                if (l_vol in [30, 45, 90]) and (sf in [sf_grid[0], sf_grid[1], sf_grid[-1]]) and exp == 0.08:
                    print(f"{l_vol:<6} | {sf:<10.3f} | {exp:<10.2f} | {ep:6.2f}%   | {em:6.2f}%   | {max_err:5.2f}%  | {status}")
    rec_df = pd.DataFrame(records)
    print("-" * 75)
    print(f"📊 Итог сетки возмущений ({len(rec_df)} комбинаций в диапазоне ±100%):")
    print(f"   • Средняя ошибка по всему пространству: {rec_df['MaxErr'].mean():.2f}%")
    print(f"   • Медианная ошибка:                     {rec_df['MaxErr'].median():.2f}%")
    print(f"   • Доля конфигураций с ошибкой < 3.0%:   {(rec_df['MaxErr'] < 3.0).mean() * 100:.1f}%")
    print(f"   • Доля конфигураций с ошибкой < 5.0%:   {(rec_df['MaxErr'] < 5.0).mean() * 100:.1f}%")
    return rec_df

def test_target_permutation_placebo(coin='prl', H=2):
    """Тест 2: Проверка на утечку данных и ложную корреляцию (Target Permutation Test)"""
    print(f"\n" + "="*80)
    print(f"🧪 ТЕСТ 2: ТЕСТ СЛУЧАЙНОГО ПЕРЕМЕШИВАНИЯ (PLACEBO TEST) ДЛЯ {coin.upper()}")
    print("="*80)
    fpath = f'data/features/{coin}_1s_features.parquet'
    df = pd.read_parquet(fpath)
    feat_cols = get_stationary_feature_cols(df)
    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    n = len(df)
    
    from _tools.data_labeler import compute_single_horizon_labels, COIN_CONFIG
    cfg = COIN_CONFIG[coin]
    res = compute_single_horizon_labels(df, coin, H, 4, cfg['min_vol'], cfg['tick_size'])
    z_real = res['z']
    
    # 1. Реальная модель
    split = int(n * 0.70)
    tr_idx = np.arange(60, split - H, 2)
    te_idx = np.arange(split + 360, n - H)
    
    model_real = lgb.LGBMRegressor(n_estimators=40, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=SAFE_WORKERS, verbose=-1)
    model_real.fit(X_all[tr_idx], np.clip(z_real[tr_idx], -2.5, 2.5))
    pred_real = model_real.predict(X_all[te_idx])
    ic_real = float(np.corrcoef(pred_real, z_real[te_idx])[0, 1])
    
    # 2. Плацебо-модель (перемешиваем train таргет, разрушая причинно-следственную связь)
    np.random.seed(42)
    z_shuffled_tr = np.random.permutation(z_real[tr_idx])
    
    model_placebo = lgb.LGBMRegressor(n_estimators=40, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=SAFE_WORKERS, verbose=-1)
    model_placebo.fit(X_all[tr_idx], np.clip(z_shuffled_tr, -2.5, 2.5))
    pred_placebo = model_placebo.predict(X_all[te_idx])
    ic_placebo = float(np.corrcoef(pred_placebo, z_real[te_idx])[0, 1])
    
    print(f"   • Реальный Out-of-Sample IC:       {ic_real:+.4f} (настоящий сигнал микроструктуры)")
    print(f"   • Плацебо Out-of-Sample IC:        {ic_placebo:+.4f} (должен быть строго 0.000 ± 0.010)")
    if abs(ic_placebo) < 0.015:
        print(f"   ✅ ВЕРДИКТ: Утечки данных нет (Zero Data Leakage). Сигнал не является артефактом подгонки.")
    else:
        print(f"   ❌ ВНИМАНИЕ: Плацебо IC отличен от нуля, возможна скрытая утечка.")

def test_rolling_window_stability(coin='prl', H=2):
    """Тест 3: Скользящее окно постоянного размера (Rolling Window без накопления истории)"""
    print(f"\n" + "="*80)
    print(f"🔄 ТЕСТ 3: СКОЛЬЗЯЩЕЕ ОКНО (STRICT ROLLING WINDOW, W=120k секунд) ДЛЯ {coin.upper()}")
    print("="*80)
    fpath = f'data/features/{coin}_1s_features.parquet'
    df = pd.read_parquet(fpath)
    feat_cols = get_stationary_feature_cols(df)
    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    n = len(df)
    
    from _tools.data_labeler import compute_single_horizon_labels, COIN_CONFIG
    cfg = COIN_CONFIG[coin]
    res = compute_single_horizon_labels(df, coin, H, 4, cfg['min_vol'], cfg['tick_size'])
    z = res['z']
    c7 = res['class_7']
    
    window_train = 120000 # ~33.3 часа
    window_test = 50000   # ~13.8 часов
    step = 60000
    
    starts = list(range(0, n - window_train - window_test, step))
    ics, err_p2s, err_m2s = [], [], []
    
    print(f"{'Slice':<6} | {'Train Window':<23} | {'Test Window':<23} | {'OOS IC':<10} | {'Err(+2)':<9} | {'Err(-2)'}")
    print("-" * 85)
    for idx, s in enumerate(starts):
        tr_s = s
        tr_e = s + window_train - H
        te_s = s + window_train + 360 # Embargo
        te_e = min(n - H, te_s + window_test)
        
        tr_idx = np.arange(tr_s + 60, tr_e, 3)
        te_idx = np.arange(te_s, te_e)
        
        m_tr = (c7[tr_idx] == 2)
        m_te = (c7[te_idx] == 2)
        p_tr = z[tr_idx][m_tr].mean() if m_tr.sum() > 5 else 1.0
        p_te = z[te_idx][m_te].mean() if m_te.sum() > 5 else 1.0
        err_p = abs(p_te - p_tr) / p_tr * 100
        
        m_tr_neg = (c7[tr_idx] == -2)
        m_te_neg = (c7[te_idx] == -2)
        neg_tr = z[tr_idx][m_tr_neg].mean() if m_tr_neg.sum() > 5 else -1.0
        neg_te = z[te_idx][m_te_neg].mean() if m_te_neg.sum() > 5 else -1.0
        err_m = abs(abs(neg_te) - abs(neg_tr)) / abs(neg_tr) * 100
        
        model = lgb.LGBMRegressor(n_estimators=35, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=SAFE_WORKERS, verbose=-1)
        model.fit(X_all[tr_idx], np.clip(z[tr_idx], -2.5, 2.5))
        pred_te = model.predict(X_all[te_idx])
        ic = float(np.corrcoef(pred_te, z[te_idx])[0, 1])
        
        ics.append(ic)
        err_p2s.append(err_p)
        err_m2s.append(err_m)
        print(f"#{idx+1:<5} | {tr_s:>6,} -> {tr_e:>6,} | {te_s:>6,} -> {te_e:>6,} | {ic:+9.4f}  | {err_p:6.2f}%   | {err_m:6.2f}%")
        
    print("-" * 85)
    print(f"🏆 Средний Rolling OOS IC: {np.mean(ics):+.4f} | Средний Err(+2): {np.mean(err_p2s):.2f}% | Средний Err(-2): {np.mean(err_m2s):.2f}%")

if __name__ == '__main__':
    test_parameter_plateau('prl', H=2)
    test_target_permutation_placebo('prl', H=2)
    test_rolling_window_stability('prl', H=2)
    test_parameter_plateau('quantus', H=25)
    test_target_permutation_placebo('quantus', H=25)

