import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    path = 'data/features/prl_1s_features.parquet'
    cols = [
        'ST_bid_1_price', 'ST_ask_1_price', 'ST_bid_1_qty', 'ST_ask_1_qty',
        'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
        'ST_delta_vol'
    ]
    tbl = pq.read_table(path, columns=cols)
    df = tbl.to_pandas()
    n = len(df)
    split_idx = int(n * 0.8)
    TICK_SIZE = 0.005
    H = 5
    L = 20
    
    mid = df['ST_mid_price'].values
    spread = df['ST_spread'].values / TICK_SIZE
    b5 = df['ST_bids_vol_top5'].values
    a5 = df['ST_asks_vol_top5'].values
    
    step_1s = (mid[1:] - mid[:-1]) / TICK_SIZE
    vol_L = pd.Series(step_1s).rolling(L, min_periods=L).std().values
    
    asks_L = pd.Series(a5).rolling(L, min_periods=L).mean().values
    bids_L = pd.Series(b5).rolling(L, min_periods=L).mean().values
    
    valid_start = L
    valid_end = n - H
    
    dt = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)[valid_start:valid_end]
    vt = np.nan_to_num(vol_L[valid_start:valid_end])
    sp = spread[valid_start:valid_end]
    a_cur = a5[valid_start:valid_end]
    b_cur = b5[valid_start:valid_end]
    a_m = asks_L[valid_start:valid_end]
    b_m = bids_L[valid_start:valid_end]
    
    # Асимметричный масштаб сопротивления
    M_up = (sp + vt * np.sqrt(H)) * (np.clip(a_m / np.maximum(a_cur, 50.0), 0.2, 5.0) ** 0.3)
    M_dn = (sp + vt * np.sqrt(H)) * (np.clip(b_m / np.maximum(b_cur, 50.0), 0.2, 5.0) ** 0.3)
    
    z = np.zeros(len(dt))
    z[dt > 0] = dt[dt > 0] / M_up[dt > 0]
    z[dt < 0] = dt[dt < 0] / M_dn[dt < 0]
    
    train_n = split_idx - valid_start
    
    z_tr, z_va = z[:train_n], z[train_n:]
    dt_tr, dt_va = dt[:train_n], dt[train_n:]
    
    # Калибровка порогов на Train (устойчивые плато):
    # Рост:
    # 75% квантиль роста = 0.50
    # 90% квантиль роста = 1.00
    # Спад:
    # 75% квантиль спада = -0.50
    # 90% квантиль спада = -0.95
    
    th_p1_p2 = 0.50 # граница умеренного и регулярного роста
    th_p2_p3 = 1.00 # граница регулярного роста и супер-импульса китов
    
    th_m1_m2 = -0.50 # граница умеренного и регулярного спада
    th_m2_m3 = -0.95 # граница регулярного спада и паники / дампа
    
    def get_7_labels(z_arr, dt_arr):
        lbl = np.zeros(len(dt_arr), dtype=int)
        # Рост:
        lbl[(dt_arr > 0) & (z_arr < th_p1_p2)] = 1
        lbl[(dt_arr > 0) & (z_arr >= th_p1_p2) & (z_arr < th_p2_p3)] = 2
        lbl[(dt_arr > 0) & (z_arr >= th_p2_p3)] = 3
        # Спад:
        lbl[(dt_arr < 0) & (z_arr > th_m1_m2)] = -1
        lbl[(dt_arr < 0) & (z_arr <= th_m1_m2) & (z_arr > th_m2_m3)] = -2
        lbl[(dt_arr < 0) & (z_arr <= th_m2_m3)] = -3
        return lbl
        
    lbl_tr = get_7_labels(z_tr, dt_tr)
    lbl_va = get_7_labels(z_va, dt_va)
    
    print("=== АНАЛИЗ 7 АСИММЕТРИЧНЫХ КЛАССОВ (TRAIN VS VAL) ===", flush=True)
    print("Класс | Описание               | Train % (N)       | Val % (N)        | Train E[dt]| Val E[dt]  | Err(dt) | Train E[z] | Val E[z]  | Err(z)")
    print("-" * 125)
    
    class_desc = {
        -3: "Панический дамп (киты)  ",
        -2: "Регулярный спад         ",
        -1: "Микросдвиг вниз (1 тик) ",
         0: "Флэт / шум              ",
         1: "Микросдвиг вверх (1 тик)",
         2: "Регулярный пробой       ",
         3: "Супер-памп (киты)       "
    }
    
    for c in [-3, -2, -1, 0, 1, 2, 3]:
        m_tr = (lbl_tr == c)
        m_va = (lbl_va == c)
        
        c_tr = np.sum(m_tr)
        c_va = np.sum(m_va)
        
        pct_tr = c_tr / len(lbl_tr) * 100
        pct_va = c_va / len(lbl_va) * 100
        
        m_dt_tr = np.mean(dt_tr[m_tr]) if c_tr > 0 else 0
        m_dt_va = np.mean(dt_va[m_va]) if c_va > 0 else 0
        err_dt = abs(abs(m_dt_va) - abs(m_dt_tr)) / abs(m_dt_tr) * 100 if m_dt_tr != 0 else 0
        
        m_z_tr = np.mean(z_tr[m_tr]) if c_tr > 0 else 0
        m_z_va = np.mean(z_va[m_va]) if c_va > 0 else 0
        err_z = abs(abs(m_z_va) - abs(m_z_tr)) / abs(m_z_tr) * 100 if m_z_tr != 0 else 0
        
        desc = class_desc[c]
        print(f" {c:+2d}   | {desc} | {pct_tr:5.2f}% ({c_tr:6,d}) | {pct_va:5.2f}% ({c_va:5,d}) | {m_dt_tr:+6.2f} т  | {m_dt_va:+6.2f} т  | {err_dt:5.1f}%  | {m_z_tr:+8.4f}  | {m_z_va:+8.4f}  | {err_z:5.1f}%")
    print("-" * 125)

if __name__ == '__main__':
    main()
