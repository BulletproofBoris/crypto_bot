import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    path = 'data/features/prl_1s_features.parquet'
    tbl = pq.read_table(path, columns=['ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5', 'ST_delta_vol', 'mid_price_diff'])
    df = tbl.to_pandas()
    n = len(df)
    split_idx = int(n * 0.8)
    TICK_SIZE = 0.005
    H = 5
    L = 20
    
    mid = df['ST_mid_price'].values
    spread = df['ST_spread'].values / TICK_SIZE # in ticks
    bids = df['ST_bids_vol_top5'].values
    asks = df['ST_asks_vol_top5'].values
    delta_vol = df['ST_delta_vol'].values
    
    delta_ticks = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    
    # 1. Local volatility over L (in ticks)
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    # 2. Local spread over L
    spread_mean_L = pd.Series(spread).rolling(L, min_periods=L).mean().values
    
    # 3. Local depth over L
    depth_total = bids + asks
    depth_mean_L = pd.Series(depth_total).rolling(L, min_periods=L).mean().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_L[valid_start:valid_end]
    sp = spread[valid_start:valid_end]
    sp_l = spread_mean_L[valid_start:valid_end]
    dp = depth_total[valid_start:valid_end]
    dp_l = depth_mean_L[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    tr_dt = dt[:train_n]
    va_dt = dt[train_n:]
    tr_vt = vt[:train_n]
    va_vt = vt[train_n:]
    tr_sp = sp[:train_n]
    va_sp = sp[train_n:]
    tr_spl = sp_l[:train_n]
    va_spl = sp_l[train_n:]
    tr_dp = dp[:train_n]
    va_dp = dp[train_n:]
    tr_dpl = dp_l[:train_n]
    va_dpl = dp_l[train_n:]
    
    print("=== АНАЛИЗ КОЭФФИЦИЕНТА K = DELTA / MARKET_STATE НА ОБУЧАЮЩЕЙ ВЫБОРКЕ ===", flush=True)
    
    # Consider only moments with price movement on Train (|dt| > 0)
    move_mask_tr = (np.abs(tr_dt) > 0)
    move_mask_va = (np.abs(va_dt) > 0)
    
    print(f"Всего движений (|dt| >= 1): Train = {np.sum(move_mask_tr):,d} ({np.mean(move_mask_tr)*100:.2f}%) | Val = {np.sum(move_mask_va):,d} ({np.mean(move_mask_va)*100:.2f}%)", flush=True)
    
    # What is the baseline market factor M?
    # In microstructure, expected move scale M is driven by:
    # 1. Spread (minimum possible move is 1 spread / 1 tick)
    # 2. Volatility over window L: sigma_L * sqrt(H)
    # 3. Depth resistance: (depth_L / current_depth)
    
    # Let's test different definitions of M:
    # Model 1: M = max(1.0, vt * sqrt(H))
    # Model 2: M = sp + vt * sqrt(H)
    # Model 3: M = sp * sqrt(dp_L / max(dp, 50))
    # Model 4: M = max(sp, vt * sqrt(H)) * (dp_L / max(dp, 50))^alpha
    
    models = {
        'Vol_only': np.maximum(1.0, tr_vt * np.sqrt(H)),
        'Spread + Vol': tr_sp + np.nan_to_num(tr_vt) * np.sqrt(H),
        'Spread * Depth_Elasticity': tr_sp * np.sqrt(np.clip(tr_dpl / np.maximum(tr_dp, 50.0), 0.2, 5.0)),
        'Spread_L * Vol': np.maximum(tr_spl, 1.0) * (1.0 + np.nan_to_num(tr_vt) * np.sqrt(H)),
    }
    
    models_va = {
        'Vol_only': np.maximum(1.0, va_vt * np.sqrt(H)),
        'Spread + Vol': va_sp + np.nan_to_num(va_vt) * np.sqrt(H),
        'Spread * Depth_Elasticity': va_sp * np.sqrt(np.clip(va_dpl / np.maximum(va_dp, 50.0), 0.2, 5.0)),
        'Spread_L * Vol': np.maximum(va_spl, 1.0) * (1.0 + np.nan_to_num(va_vt) * np.sqrt(H)),
    }
    
    for name, M_tr in models.items():
        M_va = models_va[name]
        
        # When move occurs, what is k = |dt| / M?
        k_tr = np.abs(tr_dt)[move_mask_tr] / M_tr[move_mask_tr]
        k_va = np.abs(va_dt)[move_mask_va] / M_va[move_mask_va]
        
        mean_k_tr = np.mean(k_tr)
        var_k_tr = np.var(k_tr)
        std_k_tr = np.std(k_tr)
        med_k_tr = np.median(k_tr)
        
        mean_k_va = np.mean(k_va)
        var_k_va = np.var(k_va)
        std_k_va = np.std(k_va)
        med_k_va = np.median(k_va)
        
        err_mean = abs(mean_k_va - mean_k_tr) / mean_k_tr * 100
        err_med = abs(med_k_va - med_k_tr) / med_k_tr * 100
        
        print(f"\n--- Модель: {name} ---", flush=True)
        print(f"  Train: E[k] = {mean_k_tr:.4f}, Var[k] = {var_k_tr:.4f}, Std[k] = {std_k_tr:.4f}, Median[k] = {med_k_tr:.4f}", flush=True)
        print(f"  Val:   E[k] = {mean_k_va:.4f}, Var[k] = {var_k_va:.4f}, Std[k] = {std_k_va:.4f}, Median[k] = {med_k_va:.4f}", flush=True)
        print(f"  Расхождение среднего E[k]:  {err_mean:5.2f}%", flush=True)
        print(f"  Расхождение медианы Med[k]: {err_med:5.2f}%", flush=True)

if __name__ == '__main__':
    main()
