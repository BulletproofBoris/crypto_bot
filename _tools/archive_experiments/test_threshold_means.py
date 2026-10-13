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
    
    # Fast vectorized rolling mean / std
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_L[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    tr_dt = dt[:train_n]
    va_dt = dt[train_n:]
    tr_vt = vt[:train_n]
    va_vt = vt[train_n:]
    
    print("=== ИССЛЕДОВАНИЕ СРЕДНИХ ЗНАЧЕНИЙ ПРИ РАЗНЫХ ПОРОГАХ ===", flush=True)
    print(f"Train samples: {train_n:,} | Val samples: {len(va_dt):,}", flush=True)
    
    print("\n1. ФИКСИРОВАННЫЕ ЦЕЛОЧИСЛЕННЫЕ ПОРОГИ:", flush=True)
    print("Порог  | Train средний сдвиг | Val средний сдвиг | Отклонение (%) | Train N | Val N", flush=True)
    print("-" * 75, flush=True)
    for T in [1, 2, 3, 4, 5]:
        tr_m_up = np.mean(tr_dt[tr_dt >= T]) if np.sum(tr_dt >= T) > 0 else 0
        va_m_up = np.mean(va_dt[va_dt >= T]) if np.sum(va_dt >= T) > 0 else 0
        err_up = abs(va_m_up - tr_m_up) / tr_m_up * 100 if tr_m_up > 0 else 0
        print(f" >= +{T} | {tr_m_up:+6.2f} тика        | {va_m_up:+6.2f} тика       | {err_up:6.1f}%        | {np.sum(tr_dt >= T):7,d} | {np.sum(va_dt >= T):5,d}", flush=True)

    print("-" * 75, flush=True)
    for T in [1, 2, 3, 4, 5]:
        tr_m_dn = np.mean(tr_dt[tr_dt <= -T]) if np.sum(tr_dt <= -T) > 0 else 0
        va_m_dn = np.mean(va_dt[va_dt <= -T]) if np.sum(va_dt <= -T) > 0 else 0
        err_dn = abs(abs(va_m_dn) - abs(tr_m_dn)) / abs(tr_m_dn) * 100 if tr_m_dn < 0 else 0
        print(f" <= -{T} | {tr_m_dn:+6.2f} тика        | {va_m_dn:+6.2f} тика       | {err_dn:6.1f}%        | {np.sum(tr_dt <= -T):7,d} | {np.sum(va_dt <= -T):5,d}", flush=True)

if __name__ == '__main__':
    main()
