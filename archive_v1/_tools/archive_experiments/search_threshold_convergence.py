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
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_L[valid_start:valid_end]
    a_cur = asks[valid_start:valid_end]
    b_cur = bids[valid_start:valid_end]
    sp_cur = spread[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    tr_dt = dt[:train_n]
    va_dt = dt[train_n:]
    tr_vt = np.nan_to_num(vt[:train_n])
    va_vt = np.nan_to_num(vt[train_n:])
    tr_a = a_cur[:train_n]
    va_a = a_cur[train_n:]
    tr_sp = sp_cur[:train_n]
    va_sp = sp_cur[train_n:]
    
    results = []
    
    for k_base in [0.0, 0.5, 1.0, 1.5]:
        for k_vol in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            for k_depth in [0.0, 0.2, 0.5, 1.0]:
                for k_sp in [0.0, 0.2, 0.5]:
                    # Raw float threshold
                    tr_th = np.maximum(1, np.round(k_base + k_vol * tr_vt + k_depth * (1000.0 / (tr_a + 500.0)) + k_sp * tr_sp)).astype(int)
                    va_th = np.maximum(1, np.round(k_base + k_vol * va_vt + k_depth * (1000.0 / (va_a + 500.0)) + k_sp * va_sp)).astype(int)
                    
                    tr_mask = (tr_dt >= tr_th)
                    va_mask = (va_dt >= va_th)
                    
                    if np.sum(tr_mask) < 200 or np.sum(va_mask) < 100:
                        continue
                        
                    tr_m = np.mean(tr_dt[tr_mask])
                    va_m = np.mean(va_dt[va_mask])
                    
                    diff_pct = abs(va_m - tr_m) / tr_m * 100
                    results.append((diff_pct, k_base, k_vol, k_depth, k_sp, tr_m, va_m, np.sum(tr_mask), np.sum(va_mask)))
                    
    results.sort(key=lambda x: x[0])
    print("\nТоп-10 конфигураций без фиксации на 2 тика (с минимумом 1 тик):")
    print("Diff (%) | k_base | k_vol | k_depth | k_sp | Train сдвиг | Val сдвиг | Train N | Val N")
    print("-" * 85)
    for r in results[:10]:
        print(f" {r[0]:5.1f}%  | {r[1]:6.1f} | {r[2]:5.1f} | {r[3]:7.1f} | {r[4]:4.1f} | {r[5]:+6.2f} тика | {r[6]:+6.2f} тика| {r[7]:7,d} | {r[8]:5,d}")

if __name__ == '__main__':
    main()
