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
    bids = df['ST_bids_vol_top5'].values
    asks = df['ST_asks_vol_top5'].values
    
    delta_ticks = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    asks_L = pd.Series(asks).rolling(L, min_periods=L).mean().values
    bids_L = pd.Series(bids).rolling(L, min_periods=L).mean().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = np.nan_to_num(vol_L[valid_start:valid_end])
    sp = spread[valid_start:valid_end]
    a_cur = asks[valid_start:valid_end]
    b_cur = bids[valid_start:valid_end]
    a_L = asks_L[valid_start:valid_end]
    b_L = bids_L[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    tr_dt, va_dt = dt[:train_n], dt[train_n:]
    tr_vt, va_vt = vt[:train_n], vt[train_n:]
    tr_sp, va_sp = sp[:train_n], sp[train_n:]
    tr_a, va_a = a_cur[:train_n], a_cur[train_n:]
    tr_b, va_b = b_cur[:train_n], b_cur[train_n:]
    tr_aL, va_aL = a_L[:train_n], a_L[train_n:]
    tr_bL, va_bL = b_L[:train_n], b_L[train_n:]
    
    # Asymmetric M with p_depth = 0.3
    M_up_tr = (tr_sp + tr_vt * np.sqrt(H)) * (np.clip(tr_aL / np.maximum(tr_a, 50.0), 0.2, 5.0) ** 0.3)
    M_up_va = (va_sp + va_vt * np.sqrt(H)) * (np.clip(va_aL / np.maximum(va_a, 50.0), 0.2, 5.0) ** 0.3)
    
    M_dn_tr = (tr_sp + tr_vt * np.sqrt(H)) * (np.clip(tr_bL / np.maximum(tr_b, 50.0), 0.2, 5.0) ** 0.3)
    M_dn_va = (va_sp + va_vt * np.sqrt(H)) * (np.clip(va_bL / np.maximum(va_b, 50.0), 0.2, 5.0) ** 0.3)
    
    # Normalized z
    z_tr = np.zeros(len(tr_dt))
    z_tr[tr_dt > 0] = tr_dt[tr_dt > 0] / M_up_tr[tr_dt > 0]
    z_tr[tr_dt < 0] = tr_dt[tr_dt < 0] / M_dn_tr[tr_dt < 0]
    
    z_va = np.zeros(len(va_dt))
    z_va[va_dt > 0] = va_dt[va_dt > 0] / M_up_va[va_dt > 0]
    z_va[va_dt < 0] = va_dt[va_dt < 0] / M_dn_va[va_dt < 0]
    
    # Let's check percentiles of non-zero movements on Train:
    tr_pos = z_tr[tr_dt > 0]
    tr_neg = z_tr[tr_dt < 0]
    
    print("=== РАСПРЕДЕЛЕНИЕ Z НА TRAIN ПРИ ДВИЖЕНИИ ===", flush=True)
    print(f"Рост: E[z] = {np.mean(tr_pos):.4f}, Std = {np.std(tr_pos):.4f}, 50% = {np.median(tr_pos):.4f}, 75% = {np.percentile(tr_pos, 75):.4f}, 90% = {np.percentile(tr_pos, 90):.4f}")
    print(f"Спад: E[z] = {np.mean(tr_neg):.4f}, Std = {np.std(tr_neg):.4f}, 50% = {np.median(tr_neg):.4f}, 25% = {np.percentile(tr_neg, 25):.4f}, 10% = {np.percentile(tr_neg, 10):.4f}")
    
    # Now let's test using the 75th percentile on Train as threshold for Class +2 / -2:
    th_p2 = np.percentile(tr_pos, 75)
    th_m2 = np.percentile(tr_neg, 25)
    
    def get_labels(z_arr, dt_arr, th_p, th_m):
        lbl = np.zeros(len(dt_arr), dtype=int)
        lbl[(dt_arr > 0) & (z_arr < th_p)] = 1
        lbl[(dt_arr > 0) & (z_arr >= th_p)] = 2
        lbl[(dt_arr < 0) & (z_arr > th_m)] = -1
        lbl[(dt_arr < 0) & (z_arr <= th_m)] = -2
        return lbl
        
    tr_lbl = get_labels(z_tr, tr_dt, th_p2, th_m2)
    va_lbl = get_labels(z_va, va_dt, th_p2, th_m2)
    
    print("\n" + "="*80)
    print(f"📊 СРАВНЕНИЕ 5 КЛАССОВ (ПОРОГ +2: z >= {th_p2:.4f}, ПОРОГ -2: z <= {th_m2:.4f})")
    print("="*80)
    print("Класс | Train % (N)        | Val % (N)         | Train E[z] | Val E[z]  | Err(z) | Train E[dt] | Val E[dt]  | Err(dt)")
    print("-" * 105)
    for c in [-2, -1, 0, 1, 2]:
        tr_m = (tr_lbl == c)
        va_m = (va_lbl == c)
        
        tr_cnt = np.sum(tr_m)
        va_cnt = np.sum(va_m)
        
        tr_pct = tr_cnt / len(tr_lbl) * 100
        va_pct = va_cnt / len(va_lbl) * 100
        
        tr_ez = np.mean(z_tr[tr_m]) if tr_cnt > 0 else 0
        va_ez = np.mean(z_va[va_m]) if va_cnt > 0 else 0
        err_z = abs(abs(va_ez) - abs(tr_ez)) / abs(tr_ez) * 100 if tr_ez != 0 else 0
        
        tr_edt = np.mean(tr_dt[tr_m]) if tr_cnt > 0 else 0
        va_edt = np.mean(va_dt[va_m]) if va_cnt > 0 else 0
        err_dt = abs(abs(va_edt) - abs(tr_edt)) / abs(tr_edt) * 100 if tr_edt != 0 else 0
        
        print(f" {c:+2d}   | {tr_pct:5.2f}% ({tr_cnt:6,d}) | {va_pct:5.2f}% ({va_cnt:6,d}) | {tr_ez:+9.4f}  | {va_ez:+9.4f} | {err_z:5.1f}% | {tr_edt:+7.2f} т   | {va_edt:+7.2f} т  | {err_dt:5.1f}%")
    print("-" * 105)

if __name__ == '__main__':
    main()
