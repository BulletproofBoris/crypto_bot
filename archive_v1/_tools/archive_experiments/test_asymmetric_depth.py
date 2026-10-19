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
    
    # Dynamic rolling mean depth over L
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
    
    # Train / Val arrays
    tr_dt, va_dt = dt[:train_n], dt[train_n:]
    tr_vt, va_vt = vt[:train_n], vt[train_n:]
    tr_sp, va_sp = sp[:train_n], sp[train_n:]
    tr_a, va_a = a_cur[:train_n], a_cur[train_n:]
    tr_b, va_b = b_cur[:train_n], b_cur[train_n:]
    tr_aL, va_aL = a_L[:train_n], a_L[train_n:]
    tr_bL, va_bL = b_L[:train_n], b_L[train_n:]
    
    print("=== ТЕСТИРОВАНИЕ АСИММЕТРИЧНОЙ МОДЕЛИ С УЧЕТОМ ГЛУБИНЫ АСКОВ И БИДОВ ===", flush=True)
    
    # Asymmetric depth factor:
    # Upward resistance: if current ask depth is low relative to window L, price moves easier -> M_up increases!
    # Downward resistance: if current bid depth is low relative to window L, price moves easier -> M_dn increases!
    for p_depth in [0.1, 0.2, 0.3, 0.5]:
        tr_res_up = np.clip(tr_aL / np.maximum(tr_a, 50.0), 0.2, 5.0) ** p_depth
        va_res_up = np.clip(va_aL / np.maximum(va_a, 50.0), 0.2, 5.0) ** p_depth
        
        tr_res_dn = np.clip(tr_bL / np.maximum(tr_b, 50.0), 0.2, 5.0) ** p_depth
        va_res_dn = np.clip(va_bL / np.maximum(va_b, 50.0), 0.2, 5.0) ** p_depth
        
        M_up_tr = (tr_sp + tr_vt * np.sqrt(H)) * tr_res_up
        M_up_va = (va_sp + va_vt * np.sqrt(H)) * va_res_up
        
        M_dn_tr = (tr_sp + tr_vt * np.sqrt(H)) * tr_res_dn
        M_dn_va = (va_sp + va_vt * np.sqrt(H)) * va_res_dn
        
        # Normalized returns:
        z_up_tr = tr_dt[tr_dt > 0] / M_up_tr[tr_dt > 0]
        z_up_va = va_dt[va_dt > 0] / M_up_va[va_dt > 0]
        
        z_dn_tr = np.abs(tr_dt[tr_dt < 0]) / M_dn_tr[tr_dt < 0]
        z_dn_va = np.abs(va_dt[va_dt < 0]) / M_dn_va[va_dt < 0]
        
        diff_up = abs(np.mean(z_up_va) - np.mean(z_up_tr)) / np.mean(z_up_tr) * 100
        diff_dn = abs(np.mean(z_dn_va) - np.mean(z_dn_tr)) / np.mean(z_dn_tr) * 100
        
        print(f"Степень p_depth={p_depth:.1f}:")
        print(f"  Рост: Train E[z] = {np.mean(z_up_tr):.4f} | Val E[z] = {np.mean(z_up_va):.4f} | Diff: {diff_up:5.2f}%")
        print(f"  Спад: Train E[z] = {np.mean(z_dn_tr):.4f} | Val E[z] = {np.mean(z_dn_va):.4f} | Diff: {diff_dn:5.2f}%")

if __name__ == '__main__':
    main()
