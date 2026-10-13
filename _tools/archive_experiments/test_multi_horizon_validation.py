import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def run_multi_horizon_validation():
    path = 'data/features/prl_1s_features.parquet'
    cols = [
        'ST_bid_1_price', 'ST_ask_1_price', 'ST_bid_1_qty', 'ST_ask_1_qty',
        'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5'
    ]
    tbl = pq.read_table(path, columns=cols)
    df = tbl.to_pandas()
    n = len(df)
    split_idx = int(n * 0.8)
    TICK_SIZE = 0.005
    
    mid = df['ST_mid_price'].values
    spread = df['ST_spread'].values / TICK_SIZE # в тиках
    b1_p = df['ST_bid_1_price'].values
    a1_p = df['ST_ask_1_price'].values
    b1_q = df['ST_bid_1_qty'].values
    a1_q = df['ST_ask_1_qty'].values
    b5 = df['ST_bids_vol_top5'].values
    a5 = df['ST_asks_vol_top5'].values
    
    # 1-second price steps
    step_1s = (mid[1:] - mid[:-1]) / TICK_SIZE
    s_step = pd.Series(step_1s)
    
    # Микроцена
    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 0, (b1_p * a1_q + a1_p * b1_q) / np.maximum(total_q1, 1.0), mid)
    micro_offset = (microprice - mid) / TICK_SIZE
    s_micro = pd.Series(micro_offset)
    
    horizons = [5, 15, 30, 60]
    
    print("=" * 110, flush=True)
    print("🚀 ВАЛИДАЦИЯ ДИНАМИЧЕСКОГО МАСШТАБИРОВАНИЯ НА МУЛЬТИГОРИЗОНТАХ (H = 5s, 15s, 30s, 60s, L = 4*H)", flush=True)
    print("=" * 110, flush=True)
    
    all_horizon_results = {}
    
    for H in horizons:
        L = 4 * H # отношение L/H = 4 сохраняется неизменным!
        
        # 1. Локальная волатильность за окно L
        vol_L = s_step.rolling(L, min_periods=L).std().values
        
        # 2. Локальная средняя глубина стакана за окно L
        asks_L = pd.Series(a5).rolling(L, min_periods=L).mean().values
        bids_L = pd.Series(b5).rolling(L, min_periods=L).mean().values
        
        # 3. Волновой накат за время t_wave = max(3, H // 3)
        t_w = max(3, H // 3)
        v_wave = (s_micro - s_micro.shift(t_w)).fillna(0).values / t_w
        d_vol = (pd.Series(vol_L) - pd.Series(vol_L).shift(t_w)).fillna(0).values / t_w
        
        # Сдвиг цены за горизонт H
        dt_full = np.zeros(n)
        dt_full[:-H] = np.round((mid[H:] - mid[:-H]) / TICK_SIZE)
        
        valid_start = L
        valid_end = n - max(horizons)
        
        dt = dt_full[valid_start:valid_end].astype(int)
        vt = np.nan_to_num(vol_L[valid_start:valid_end])
        sp = spread[valid_start:valid_end]
        a_cur = a5[valid_start:valid_end]
        b_cur = b5[valid_start:valid_end]
        a_m = asks_L[valid_start:valid_end]
        b_m = bids_L[valid_start:valid_end]
        vw = v_wave[valid_start:valid_end]
        dv = d_vol[valid_start:valid_end]
        
        # Асимметричный динамический масштаб M_up и M_dn
        M_up_base = (sp + vt * np.sqrt(H)) * (np.clip(a_m / np.maximum(a_cur, 50.0), 0.2, 5.0) ** 0.3)
        M_dn_base = (sp + vt * np.sqrt(H)) * (np.clip(b_m / np.maximum(b_cur, 50.0), 0.2, 5.0) ** 0.3)
        
        M_up = M_up_base * (1.0 + np.maximum(0, vw)) * (1.0 + np.maximum(0, dv))
        M_dn = M_dn_base * (1.0 + np.maximum(0, -vw)) * (1.0 + np.maximum(0, dv))
        
        z = np.zeros(len(dt))
        z[dt > 0] = dt[dt > 0] / M_up[dt > 0]
        z[dt < 0] = dt[dt < 0] / M_dn[dt < 0]
        
        train_n = split_idx - valid_start
        
        z_tr, z_va = z[:train_n], z[train_n:]
        dt_tr, dt_va = dt[:train_n], dt[train_n:]
        
        # Пороги, калиброванные на Train:
        th_p1_p2 = 0.50 # устойчивое плато регулярного пробоя вверх
        th_p2_p3 = 1.00 # граница китового супер-пампа
        th_m1_m2 = -0.50 # устойчивое плато регулярного спада вниз
        th_m2_m3 = -0.95 # граница панического сброса
        
        def assign_labels(z_arr, dt_arr):
            lbl = np.zeros(len(dt_arr), dtype=int)
            lbl[(dt_arr > 0) & (z_arr < th_p1_p2)] = 1
            lbl[(dt_arr > 0) & (z_arr >= th_p1_p2) & (z_arr < th_p2_p3)] = 2
            lbl[(dt_arr > 0) & (z_arr >= th_p2_p3)] = 3
            lbl[(dt_arr < 0) & (z_arr > th_m1_m2)] = -1
            lbl[(dt_arr < 0) & (z_arr <= th_m1_m2) & (z_arr > th_m2_m3)] = -2
            lbl[(dt_arr < 0) & (z_arr <= th_m2_m3)] = -3
            return lbl
            
        lbl_tr = assign_labels(z_tr, dt_tr)
        lbl_va = assign_labels(z_va, dt_va)
        
        # Сбор метрик по классам
        res_classes = {}
        for c in [-3, -2, -1, 0, 1, 2, 3]:
            m_tr = (lbl_tr == c)
            m_va = (lbl_va == c)
            
            c_tr = np.sum(m_tr)
            c_va = np.sum(m_va)
            
            dt_m_tr = np.mean(dt_tr[m_tr]) if c_tr > 0 else 0
            dt_m_va = np.mean(dt_va[m_va]) if c_va > 0 else 0
            err_dt = abs(abs(dt_m_va) - abs(dt_m_tr)) / abs(dt_m_tr) * 100 if dt_m_tr != 0 else 0
            
            z_m_tr = np.mean(z_tr[m_tr]) if c_tr > 0 else 0
            z_m_va = np.mean(z_va[m_va]) if c_va > 0 else 0
            err_z = abs(abs(z_m_va) - abs(z_m_tr)) / abs(z_m_tr) * 100 if z_m_tr != 0 else 0
            
            pct_tr = c_tr / len(lbl_tr) * 100
            pct_va = c_va / len(lbl_va) * 100
            
            res_classes[c] = {
                'pct_tr': pct_tr, 'pct_va': pct_va,
                'dt_tr': dt_m_tr, 'dt_va': dt_m_va, 'err_dt': err_dt,
                'z_tr': z_m_tr, 'z_va': z_m_va, 'err_z': err_z,
                'count_tr': c_tr, 'count_va': c_va
            }
            
        all_horizon_results[H] = res_classes
        
        print(f"\n⏱️ ГОРИЗОНТ H = {H}с (ОКНО L = {L}с):", flush=True)
        print(f"Train samples: {train_n:,d} | Val samples: {len(lbl_va):,d}", flush=True)
        print("Класс | Описание               | Train % (N)       | Val % (N)        | Train E[dt]| Val E[dt]  | Err(dt) | Train E[z] | Val E[z]  | Err(z)")
        print("-" * 125, flush=True)
        
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
            r = res_classes[c]
            desc = class_desc[c]
            print(f" {c:+2d}   | {desc} | {r['pct_tr']:5.2f}% ({r['count_tr']:6,d}) | {r['pct_va']:5.2f}% ({r['count_va']:5,d}) | {r['dt_tr']:+6.2f} т  | {r['dt_va']:+6.2f} т  | {r['err_dt']:5.1f}%  | {r['z_tr']:+8.4f}  | {r['z_va']:+8.4f}  | {r['err_z']:5.1f}%", flush=True)
        print("-" * 125, flush=True)

if __name__ == '__main__':
    run_multi_horizon_validation()
