import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def run_lh_grid_test():
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
    
    step_1s = (mid[1:] - mid[:-1]) / TICK_SIZE
    s_step = pd.Series(step_1s)
    
    # Микроцена
    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 0, (b1_p * a1_q + a1_p * b1_q) / np.maximum(total_q1, 1.0), mid)
    micro_offset = (microprice - mid) / TICK_SIZE
    s_micro = pd.Series(micro_offset)
    
    ratios = [3, 4, 5]
    horizons = [5, 15, 30, 60]
    
    # Предрассчитаем сдвиги цен для всех H
    dt_dict = {}
    for H in horizons:
        d = np.zeros(n)
        d[:-H] = np.round((mid[H:] - mid[:-H]) / TICK_SIZE)
        dt_dict[H] = d
        
    max_L = 5 * max(horizons) # 300 сек
    valid_end = n - max(horizons)
    
    results = []
    
    # Калибровочные устойчивые пороги
    th_p1_p2 = 0.50
    th_p2_p3 = 1.00
    th_m1_m2 = -0.50
    th_m2_m3 = -0.95
    
    print("Запуск расчетов по сетке L/H = [3, 4, 5] и H = [5, 15, 30, 60]...", flush=True)
    
    for ratio in ratios:
        for H in horizons:
            L = ratio * H
            
            # Локальные метрики окна L
            vol_L = s_step.rolling(L, min_periods=L).std().values
            asks_L = pd.Series(a5).rolling(L, min_periods=L).mean().values
            bids_L = pd.Series(b5).rolling(L, min_periods=L).mean().values
            
            t_w = max(3, H // 3)
            v_wave = (s_micro - s_micro.shift(t_w)).fillna(0).values / t_w
            d_vol = (pd.Series(vol_L) - pd.Series(vol_L).shift(t_w)).fillna(0).values / t_w
            
            valid_start = L
            dt = dt_dict[H][valid_start:valid_end].astype(int)
            vt = np.nan_to_num(vol_L[valid_start:valid_end])
            sp = spread[valid_start:valid_end]
            a_cur = a5[valid_start:valid_end]
            b_cur = b5[valid_start:valid_end]
            a_m = asks_L[valid_start:valid_end]
            b_m = bids_L[valid_start:valid_end]
            vw = v_wave[valid_start:valid_end]
            dv = d_vol[valid_start:valid_end]
            
            # Асимметричный динамический масштаб
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
            
            # Извлечем ключевые классы: +2 (пробой вверх), -2 (спад вниз), +1, -1, 0
            def get_cls_stat(c):
                m_tr = (lbl_tr == c)
                m_va = (lbl_va == c)
                c_tr, c_va = np.sum(m_tr), np.sum(m_va)
                
                m_dt_tr = np.mean(dt_tr[m_tr]) if c_tr > 0 else 0
                m_dt_va = np.mean(dt_va[m_va]) if c_va > 0 else 0
                err_dt = abs(abs(m_dt_va) - abs(m_dt_tr)) / abs(m_dt_tr) * 100 if m_dt_tr != 0 else 0
                
                m_z_tr = np.mean(z_tr[m_tr]) if c_tr > 0 else 0
                m_z_va = np.mean(z_va[m_va]) if c_va > 0 else 0
                err_z = abs(abs(m_z_va) - abs(m_z_tr)) / abs(m_z_tr) * 100 if m_z_tr != 0 else 0
                
                pct_tr = c_tr / len(lbl_tr) * 100
                pct_va = c_va / len(lbl_va) * 100
                return {
                    'err_z': err_z, 'err_dt': err_dt,
                    'dt_tr': m_dt_tr, 'dt_va': m_dt_va,
                    'pct_tr': pct_tr, 'pct_va': pct_va,
                    'n_tr': c_tr, 'n_va': c_va
                }
                
            s_p2 = get_cls_stat(2)
            s_m2 = get_cls_stat(-2)
            s_p1 = get_cls_stat(1)
            s_m1 = get_cls_stat(-1)
            s_p3 = get_cls_stat(3)
            s_m3 = get_cls_stat(-3)
            s_0 = get_cls_stat(0)
            
            results.append({
                'ratio': ratio, 'H': H, 'L': L,
                'p2': s_p2, 'm2': s_m2, 'p1': s_p1, 'm1': s_m1,
                'p3': s_p3, 'm3': s_m3, 'flat': s_0
            })
            
    print("\n" + "=" * 125, flush=True)
    print("📊 СВОДНЫЕ РЕЗУЛЬТАТЫ СРАВНЕНИЯ L/H = [3, 4, 5] ДЛЯ ГОРИЗОНТОВ H = 5, 15, 30, 60 СЕКУНД", flush=True)
    print("=" * 125, flush=True)
    print("L/H | H(сек) | L(сек) | Класс +2 Err(z) | Класс -2 Err(z) | Класс +1 Err(z) | Класс -1 Err(z) | Флэт % (Train/Val) | Статус")
    print("-" * 125, flush=True)
    
    for r in results:
        ratio = r['ratio']
        H = r['H']
        L = r['L']
        err_p2 = r['p2']['err_z']
        err_m2 = r['m2']['err_z']
        err_p1 = r['p1']['err_z']
        err_m1 = r['m1']['err_z']
        pct_0_tr = r['flat']['pct_tr']
        pct_0_va = r['flat']['pct_va']
        
        status = "ИДЕАЛЬНО (< 10%)" if max(err_p2, err_m2, err_p1, err_m1) <= 10.0 else "ОТКЛОНЕНИЕ"
        print(f" {ratio:1d}  |  {H:2d}с  |  {L:3d}с |      {err_p2:5.2f}%    |      {err_m2:5.2f}%    |      {err_p1:5.2f}%    |      {err_m1:5.2f}%    |  {pct_0_tr:5.1f}% / {pct_0_va:5.1f}%  | {status}")
    print("-" * 125, flush=True)

    # Детальная таблица по рабочим классам (+2 и -2):
    print("\n" + "=" * 125, flush=True)
    print("🔍 ДЕТАЛИЗАЦИЯ ОСНОВНЫХ ТОРГОВЫХ КЛАССОВ (+2 ПРОБОЙ И -2 СПАД): СДВИГ В ТИКАХ И ДОЛИ", flush=True)
    print("=" * 125, flush=True)
    print("L/H | H(сек) | Класс +2 Train сдвиг | Класс +2 Val сдвиг | Err(dt) | Класс -2 Train сдвиг | Класс -2 Val сдвиг | Err(dt)")
    print("-" * 125, flush=True)
    for r in results:
        ratio = r['ratio']
        H = r['H']
        p2 = r['p2']
        m2 = r['m2']
        print(f" {ratio:1d}  |  {H:2d}с  |      {p2['dt_tr']:+5.2f} тика      |     {p2['dt_va']:+5.2f} тика     |  {p2['err_dt']:5.1f}% |      {m2['dt_tr']:+5.2f} тика      |     {m2['dt_va']:+5.2f} тика     |  {m2['err_dt']:5.1f}%")
    print("-" * 125, flush=True)

if __name__ == '__main__':
    run_lh_grid_test()
