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
    spread = df['ST_spread'].values / TICK_SIZE # in ticks
    
    delta_ticks = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_L[valid_start:valid_end]
    sp = spread[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    tr_dt = dt[:train_n]
    va_dt = dt[train_n:]
    tr_vt = np.nan_to_num(vt[:train_n])
    va_vt = np.nan_to_num(vt[train_n:])
    tr_sp = sp[:train_n]
    va_sp = sp[train_n:]
    
    # Baseline market scale: M = sp + vt * sqrt(H)
    M_tr = tr_sp + tr_vt * np.sqrt(H)
    M_va = va_sp + va_vt * np.sqrt(H)
    
    # On Train: find the distribution of normalized movement z = dt / M
    z_tr = tr_dt / M_tr
    z_va = va_dt / M_va
    
    # Mathematical moments of z on Train:
    # 1. On all samples:
    mean_z_tr = np.mean(z_tr)
    std_z_tr = np.std(z_tr)
    
    # 2. On non-zero moves:
    move_tr = (tr_dt != 0)
    move_va = (va_dt != 0)
    
    pos_tr = (tr_dt > 0)
    neg_tr = (tr_dt < 0)
    
    print("=== МАТЕМАТИЧЕСКИЕ МОМЕНТЫ НА ОТРЕЗКЕ ОБУЧЕНИЯ ===", flush=True)
    print(f"E[z] (все сэмплы):       {mean_z_tr:+.6f}", flush=True)
    print(f"Std[z] (все сэмплы):     {std_z_tr:.6f}", flush=True)
    print(f"E[|z|] при движении:     {np.mean(np.abs(z_tr[move_tr])):.4f}", flush=True)
    print(f"Std[|z|] при движении:   {np.std(np.abs(z_tr[move_tr])):.4f}", flush=True)
    print(f"E[z|z>0] (рост):         {np.mean(z_tr[pos_tr]):.4f} (Std: {np.std(z_tr[pos_tr]):.4f})", flush=True)
    print(f"E[z|z<0] (спад):         {np.mean(z_tr[neg_tr]):.4f} (Std: {np.std(z_tr[neg_tr]):.4f})", flush=True)
    
    # Now, let's see: on Validation:
    print("\n=== ПРОВЕРКА НА ВАЛИДАЦИОННОМ ОТРЕЗКЕ ===", flush=True)
    print(f"E[z] (все сэмплы):       {np.mean(z_va):+.6f}", flush=True)
    print(f"Std[z] (все сэмплы):     {np.std(z_va):.6f}", flush=True)
    print(f"E[|z|] при движении:     {np.mean(np.abs(z_va[move_va])):.4f}", flush=True)
    print(f"Std[|z|] при движении:   {np.std(np.abs(z_va[move_va])):.4f}", flush=True)
    print(f"E[z|z>0] (рост):         {np.mean(z_va[pos_tr[:len(z_va)]]):.4f}") # wait, pos_va
    pos_va = (va_dt > 0)
    neg_va = (va_dt < 0)
    print(f"E[z|z>0] (рост):         {np.mean(z_va[pos_va]):.4f} (Std: {np.std(z_va[pos_va]):.4f})")
    print(f"E[z|z<0] (спад):         {np.mean(z_va[neg_va]):.4f} (Std: {np.std(z_va[neg_va]):.4f})")
    
    diff_pos = abs(np.mean(z_va[pos_va]) - np.mean(z_tr[pos_tr])) / np.mean(z_tr[pos_tr]) * 100
    diff_neg = abs(abs(np.mean(z_va[neg_va])) - abs(np.mean(z_tr[neg_tr]))) / abs(np.mean(z_tr[neg_tr])) * 100
    print(f"\nОтклонение математического ожидания роста E[z|z>0]: {diff_pos:.2f}%")
    print(f"Отклонение математического ожидания спада E[z|z<0]: {diff_neg:.2f}%")

if __name__ == '__main__':
    main()
