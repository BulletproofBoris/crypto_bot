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
    
    # Baseline dynamic scale M = spread + vol_L * sqrt(H)
    # This represents the expected friction + volatility scale in ticks
    M_tr = tr_sp + tr_vt * np.sqrt(H)
    M_va = va_sp + va_vt * np.sqrt(H)
    
    # Normalized movement z = dt / M
    z_tr = tr_dt / M_tr
    z_va = va_dt / M_va
    
    print("=== ПОСТРОЕНИЕ 5 КЛАССОВ НА ОСНОВЕ МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ ===", flush=True)
    
    # On Train:
    # Look at non-zero moves:
    tr_pos_z = z_tr[tr_dt > 0]
    tr_neg_z = z_tr[tr_dt < 0]
    
    # For positive moves:
    # Typical move (Class +1): centered around median / mean
    # Strong move (Class +2): above mean + k * std, or a chosen quantile on Train
    for q_cut in [60, 65, 70, 75, 80]:
        th_pos = np.percentile(tr_pos_z, q_cut)
        th_neg = np.percentile(tr_neg_z, 100 - q_cut) # symmetric quantile
        
        # Labeling function:
        # Class 0: dt == 0
        # Class +1: dt > 0 and z < th_pos
        # Class +2: dt > 0 and z >= th_pos
        # Class -1: dt < 0 and z > th_neg
        # Class -2: dt < 0 and z <= th_neg
        
        tr_lbl = np.zeros(len(tr_dt), dtype=int)
        tr_lbl[(tr_dt > 0) & (z_tr < th_pos)] = 1
        tr_lbl[(tr_dt > 0) & (z_tr >= th_pos)] = 2
        tr_lbl[(tr_dt < 0) & (z_tr > th_neg)] = -1
        tr_lbl[(tr_dt < 0) & (z_tr <= th_neg)] = -2
        
        va_lbl = np.zeros(len(va_dt), dtype=int)
        va_lbl[(va_dt > 0) & (z_va < th_pos)] = 1
        va_lbl[(va_dt > 0) & (z_va >= th_pos)] = 2
        va_lbl[(va_dt < 0) & (z_va > th_neg)] = -1
        va_lbl[(va_dt < 0) & (z_va <= th_neg)] = -2
        
        # Check mean z and mean dt for Class +2 and -2
        m_z_tr_p2 = np.mean(z_tr[tr_lbl == 2])
        m_z_va_p2 = np.mean(z_va[va_lbl == 2])
        err_z_p2 = abs(m_z_va_p2 - m_z_tr_p2) / m_z_tr_p2 * 100
        
        m_dt_tr_p2 = np.mean(tr_dt[tr_lbl == 2])
        m_dt_va_p2 = np.mean(va_dt[va_lbl == 2])
        err_dt_p2 = abs(m_dt_va_p2 - m_dt_tr_p2) / m_dt_tr_p2 * 100
        
        m_z_tr_m2 = np.mean(z_tr[tr_lbl == -2])
        m_z_va_m2 = np.mean(z_va[va_lbl == -2])
        err_z_m2 = abs(abs(m_z_va_m2) - abs(m_z_tr_m2)) / abs(m_z_tr_m2) * 100
        
        m_dt_tr_m2 = np.mean(tr_dt[tr_lbl == -2])
        m_dt_va_m2 = np.mean(va_dt[va_lbl == -2])
        err_dt_m2 = abs(abs(m_dt_va_m2) - abs(m_dt_tr_m2)) / abs(m_dt_tr_m2) * 100
        
        print(f"\nКвантиль q={q_cut}% (Порог +2: z >= {th_pos:.4f}, Порог -2: z <= {th_neg:.4f}):")
        print(f"  Класс +2 (z):  Train E[z] = {m_z_tr_p2:.3f} | Val E[z] = {m_z_va_p2:.3f} | Diff: {err_z_p2:5.2f}%")
        print(f"  Класс +2 (dt): Train E[dt] = {m_dt_tr_p2:+.2f} т | Val E[dt] = {m_dt_va_p2:+.2f} т | Diff: {err_dt_p2:5.2f}%")
        print(f"  Класс -2 (z):  Train E[z] = {m_z_tr_m2:.3f} | Val E[z] = {m_z_va_m2:.3f} | Diff: {err_z_m2:5.2f}%")
        print(f"  Класс -2 (dt): Train E[dt] = {m_dt_tr_m2:+.2f} т | Val E[dt] = {m_dt_va_m2:+.2f} т | Diff: {err_dt_m2:5.2f}%")

if __name__ == '__main__':
    main()
