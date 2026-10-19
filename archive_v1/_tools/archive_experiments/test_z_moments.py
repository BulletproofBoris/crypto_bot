import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    path = 'data/features/prl_1s_features.parquet'
    tbl = pq.read_table(path, columns=['ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5'])
    df = tbl.to_pandas()
    
    n = len(df)
    split_idx = int(n * 0.8)
    TICK_SIZE = 0.005
    H = 5
    L = 20
    
    mid = df['ST_mid_price'].values
    spread = df['ST_spread'].values / TICK_SIZE
    bids = df['ST_bids_vol_top5'].values
    asks = df['ST_asks_vol_top5'].values
    
    delta_ticks = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    
    # 20s rolling metrics
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    depth_total = bids + asks
    # Fast rolling mean over L
    depth_mean_L = pd.Series(depth_total).rolling(L, min_periods=L).mean().values
    spread_mean_L = pd.Series(spread).rolling(L, min_periods=L).mean().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_L[valid_start:valid_end]
    dp = depth_total[valid_start:valid_end]
    dp_m = depth_mean_L[valid_start:valid_end]
    sp = spread[valid_start:valid_end]
    sp_m = spread_mean_L[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    tr_dt = dt[:train_n]
    va_dt = dt[train_n:]
    tr_vt = vt[:train_n]
    va_vt = vt[train_n:]
    tr_dp = dp[:train_n]
    va_dp = dp[train_n:]
    tr_dpm = dp_m[:train_n]
    va_dpm = dp_m[train_n:]
    
    print("=== АНАЛИЗ СТАТИСТИКИ НОРМИРОВАННОГО Z = DT / SIGMA ===", flush=True)
    
    # What if sigma is local volatility?
    # When local volatility is 0, what is the expected fluctuation? The minimum tick! (1 tick)
    # sigma_eff = sqrt(vt^2 + (spread / 2)^2)? Or sigma_eff = max(vt, 1 tick)?
    # Let's test sigma_eff:
    for base_vol in [0.5, 1.0, 1.5]:
        sig_tr = np.maximum(tr_vt * np.sqrt(H), base_vol)
        sig_va = np.maximum(va_vt * np.sqrt(H), base_vol)
        
        z_tr = tr_dt / sig_tr
        z_va = va_dt / sig_va
        
        # Now let's check: if we classify based on z:
        # Class +2: z >= 2.0 (or quantile)
        # Class +1: 0.5 <= z < 2.0
        # Class 0: -0.5 < z < 0.5
        # Class -1: -2.0 < z <= -0.5
        # Class -2: z <= -2.0
        for th_z in [1.5, 2.0, 2.5]:
            m_tr = np.mean(z_tr[z_tr >= th_z])
            m_va = np.mean(z_va[z_va >= th_z])
            err = abs(m_va - m_tr) / m_tr * 100
            print(f"base_vol={base_vol:.1f}, th_z={th_z:.1f} | Train mean(z) = {m_tr:.3f} | Val mean(z) = {m_va:.3f} | Diff = {err:.1f}%")

if __name__ == '__main__':
    main()
