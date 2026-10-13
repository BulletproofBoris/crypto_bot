import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    path = 'data/features/prl_1s_features.parquet'
    cols = [
        'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
        'ST_imbalance_top5', 'ST_slippage_buy_100_pct', 'ST_slippage_sell_100_pct',
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
    spread = df['ST_spread'].values / TICK_SIZE # в тиках
    bids5 = df['ST_bids_vol_top5'].values
    asks5 = df['ST_asks_vol_top5'].values
    imb5 = df['ST_imbalance_top5'].values
    slip_buy = df['ST_slippage_buy_100_pct'].values
    slip_sell = df['ST_slippage_sell_100_pct'].values
    delta_vol = df['ST_delta_vol'].values
    
    delta_ticks = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    
    # Локальная волатильность за окно L (в тиках)
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    # Производная волатильности (ускорение)
    vol_series = pd.Series(vol_L)
    d_vol = (vol_series - vol_series.shift(3)).fillna(0).values
    
    # Скользящие средние характеристик стакана за окно L
    asks_L = pd.Series(asks5).rolling(L, min_periods=L).mean().values
    bids_L = pd.Series(bids5).rolling(L, min_periods=L).mean().values
    slip_buy_L = pd.Series(slip_buy).rolling(L, min_periods=L).mean().values
    slip_sell_L = pd.Series(slip_sell).rolling(L, min_periods=L).mean().values
    
    # Волновое смещение дисбаланса d(Imbalance)/dt
    imb_series = pd.Series(imb5)
    d_imb_3s = (imb_series - imb_series.shift(3)).fillna(0).values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = np.nan_to_num(vol_L[valid_start:valid_end])
    dvt = d_vol[valid_start:valid_end]
    sp = spread[valid_start:valid_end]
    a_cur = asks5[valid_start:valid_end]
    b_cur = bids5[valid_start:valid_end]
    a_L = asks_L[valid_start:valid_end]
    b_L = bids_L[valid_start:valid_end]
    sb_cur = slip_buy[valid_start:valid_end]
    sb_L = slip_buy_L[valid_start:valid_end]
    ss_cur = slip_sell[valid_start:valid_end]
    ss_L = slip_sell_L[valid_start:valid_end]
    d_imb = d_imb_3s[valid_start:valid_end]
    cur_imb = imb5[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    # Разделение Train / Val
    tr_dt, va_dt = dt[:train_n], dt[train_n:]
    tr_vt, va_vt = vt[:train_n], vt[train_n:]
    tr_dvt, va_dvt = dvt[:train_n], dvt[train_n:]
    tr_sp, va_sp = sp[:train_n], sp[train_n:]
    tr_a, va_a = a_cur[:train_n], a_cur[train_n:]
    tr_b, va_b = b_cur[:train_n], b_cur[train_n:]
    tr_aL, va_aL = a_L[:train_n], a_L[train_n:]
    tr_bL, va_bL = b_L[:train_n], b_L[train_n:]
    tr_sb, va_sb = sb_cur[:train_n], sb_cur[train_n:]
    tr_sbL, va_sbL = sb_L[:train_n], sb_L[train_n:]
    tr_dimb, va_dimb = d_imb[:train_n], d_imb[train_n:]
    tr_imb, va_imb = cur_imb[:train_n], cur_imb[train_n:]
    
    print("=== ПОИСК ГЕОМЕТРИЧЕСКОГО И ВОЛНОВОГО МАСШТАБА ДЛЯ СБЛИЖЕНИЯ КЛАССА +2 ===", flush=True)
    
    # Исследуем модели масштабирования M_up:
    # Базовый масштаб: M_base = sp + vt * sqrt(H)
    # 1. Геометрия наклона стакана: geom_factor = (sb_cur / max(sb_L, 0.01))^p_geom
    #    (крутизна асков)
    # 2. Динамический коэффициент усиления: gain_vol = 1.0 + k_gain * max(0, dvt)
    # 3. Волновой накат: wave_factor = 1.0 + k_wave * max(0, d_imb)
    
    results = []
    
    for p_geom in [0.0, 0.1, 0.2, 0.3]:
        for k_gain in [0.0, 0.5, 1.0, 1.5]:
            for k_wave in [0.0, 0.5, 1.0]:
                # M_up
                tr_geom = np.clip(tr_sb / np.maximum(tr_sbL, 0.01), 0.2, 5.0) ** p_geom
                va_geom = np.clip(va_sb / np.maximum(va_sbL, 0.01), 0.2, 5.0) ** p_geom
                
                tr_gain = 1.0 + k_gain * np.maximum(0, tr_dvt)
                va_gain = 1.0 + k_gain * np.maximum(0, va_dvt)
                
                tr_wave = 1.0 + k_wave * np.maximum(0, tr_dimb)
                va_wave = 1.0 + k_wave * np.maximum(0, va_dimb)
                
                tr_depth_res = np.clip(tr_aL / np.maximum(tr_a, 50.0), 0.2, 5.0) ** 0.3
                va_depth_res = np.clip(va_aL / np.maximum(va_a, 50.0), 0.2, 5.0) ** 0.3
                
                M_up_tr = (tr_sp + tr_vt * np.sqrt(H)) * tr_depth_res * tr_geom * tr_gain * tr_wave
                M_up_va = (va_sp + va_vt * np.sqrt(H)) * va_depth_res * va_geom * va_gain * va_wave
                
                pos_tr = (tr_dt > 0)
                pos_va = (va_dt > 0)
                
                z_tr = tr_dt[pos_tr] / M_up_tr[pos_tr]
                z_va = va_dt[pos_va] / M_up_va[pos_va]
                
                # Check mean z on all positive moves
                m_z_tr = np.mean(z_tr)
                m_z_va = np.mean(z_va)
                err_all_pos = abs(m_z_va - m_z_tr) / m_z_tr * 100
                
                # Check Class +2: top 25% of positive moves on Train
                th_p2 = np.percentile(z_tr, 75)
                
                cl2_tr = (z_tr >= th_p2)
                cl2_va = (z_va >= th_p2)
                
                m_cl2_tr = np.mean(z_tr[cl2_tr])
                m_cl2_va = np.mean(z_va[cl2_va])
                err_cl2_z = abs(m_cl2_va - m_cl2_tr) / m_cl2_tr * 100
                
                # Check raw dt for Class +2
                m_dt_tr = np.mean(tr_dt[pos_tr][cl2_tr])
                m_dt_va = np.mean(va_dt[pos_va][cl2_va])
                err_cl2_dt = abs(m_dt_va - m_dt_tr) / m_dt_tr * 100
                
                results.append((err_cl2_z, err_all_pos, err_cl2_dt, p_geom, k_gain, k_wave, m_cl2_tr, m_cl2_va, m_dt_tr, m_dt_va))
                
    results.sort(key=lambda x: x[0])
    
    print("Топ-10 конфигураций по сходимости нормированного импульса Класса +2:")
    print("Err(z_cl2) | Err(z_all) | Err(dt) | p_geom | k_gain | k_wave | Train E[z] | Val E[z] | Train dt | Val dt")
    print("-" * 105)
    for r in results[:10]:
        print(f"  {r[0]:5.2f}%   |  {r[1]:5.2f}%   | {r[2]:5.2f}% |  {r[3]:4.1f}  |  {r[4]:4.1f}  |  {r[5]:4.1f}  |  {r[6]:.4f}    |  {r[7]:.4f}  |  {r[8]:+.2f} т | {r[9]:+.2f} т")

if __name__ == '__main__':
    main()
