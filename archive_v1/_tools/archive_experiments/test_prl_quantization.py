import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def test_prl_quantization():
    path = 'data/features/prl_1s_features.parquet'
    cols = [
        'ST_bid_1_price', 'ST_ask_1_price', 'ST_bid_1_qty', 'ST_ask_1_qty',
        'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
        'ST_trade_vol', 'mid_price_diff'
    ]
    tbl = pq.read_table(path, columns=cols)
    df = tbl.to_pandas()
    n = len(df)
    split_idx = int(n * 0.8)
    TICK_SIZE = 0.005
    
    mid = df['ST_mid_price'].values
    spread = df['ST_spread'].values / TICK_SIZE # в тиках
    b1_q = df['ST_bid_1_qty'].values
    a1_q = df['ST_ask_1_qty'].values
    b5 = df['ST_bids_vol_top5'].values
    a5 = df['ST_asks_vol_top5'].values
    b1_p = df['ST_bid_1_price'].values
    a1_p = df['ST_ask_1_price'].values
    arb_diff = df['mid_price_diff'].values / TICK_SIZE
    
    step_1s = (mid[1:] - mid[:-1]) / TICK_SIZE
    s_step = pd.Series(step_1s)
    
    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 0, (b1_p * a1_q + a1_p * b1_q) / np.maximum(total_q1, 1.0), mid)
    micro_offset = (microprice - mid) / TICK_SIZE
    s_micro = pd.Series(micro_offset)
    
    horizons = [5, 15, 30, 60]
    
    # Квант первого уровня в PRL: сколько токенов отделяет цену от следующего тика
    # Медиана сделки ~ 17.3 PRL. Нормируем объем 1-го уровня относительно типичной сделки (17.3 PRL):
    TYPICAL_TRADE_PRL = 17.3
    
    # Легкость пробития первого уровня:
    # Если на первом аске лежит всего 5 PRL, легкость пробития велика!
    # Если лежит 5000 PRL, пробить тяжело.
    ease_up = np.clip(TYPICAL_TRADE_PRL / np.maximum(a1_q, 1.0), 0.1, 5.0)
    ease_dn = np.clip(TYPICAL_TRADE_PRL / np.maximum(b1_q, 1.0), 0.1, 5.0)
    
    print("=" * 110, flush=True)
    print("🔬 ТЕСТИРОВАНИЕ КВАНТОВАНИЯ PRL (УРОВЕНЬ 1 И ЛОТЫ) ДЛЯ ДОСТИЖЕНИЯ ОШИБКИ < 5%", flush=True)
    print("=" * 110, flush=True)
    
    for H in horizons:
        L = 4 * H
        
        vol_L = s_step.rolling(L, min_periods=L).std().values
        asks_L = pd.Series(a5).rolling(L, min_periods=L).mean().values
        bids_L = pd.Series(b5).rolling(L, min_periods=L).mean().values
        a1_L = pd.Series(a1_q).rolling(L, min_periods=L).median().values
        b1_L = pd.Series(b1_q).rolling(L, min_periods=L).median().values
        
        t_w = max(3, H // 3)
        v_wave = (s_micro - s_micro.shift(t_w)).fillna(0).values / t_w
        d_vol = (pd.Series(vol_L) - pd.Series(vol_L).shift(t_w)).fillna(0).values / t_w
        
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
        
        # Дискретный квант 1-го уровня PRL: отношение базового объема 1-го уровня к текущему
        q1_factor_up = np.clip(a1_L[valid_start:valid_end] / np.maximum(a1_q[valid_start:valid_end], 1.0), 0.2, 5.0) ** 0.15
        q1_factor_dn = np.clip(b1_L[valid_start:valid_end] / np.maximum(b1_q[valid_start:valid_end], 1.0), 0.2, 5.0) ** 0.15
        
        # Полный динамический масштаб с учетом квантования PRL
        M_up = (sp + vt * np.sqrt(H)) * (np.clip(a_m / np.maximum(a_cur, 50.0), 0.2, 5.0) ** 0.25) * q1_factor_up * (1.0 + np.maximum(0, vw)) * (1.0 + np.maximum(0, dv))
        M_dn = (sp + vt * np.sqrt(H)) * (np.clip(b_m / np.maximum(b_cur, 50.0), 0.2, 5.0) ** 0.25) * q1_factor_dn * (1.0 + np.maximum(0, -vw)) * (1.0 + np.maximum(0, dv))
        
        z = np.zeros(len(dt))
        z[dt > 0] = dt[dt > 0] / M_up[dt > 0]
        z[dt < 0] = dt[dt < 0] / M_dn[dt < 0]
        
        train_n = split_idx - valid_start
        z_tr, z_va = z[:train_n], z[train_n:]
        dt_tr, dt_va = dt[:train_n], dt[train_n:]
        
        th_p1_p2 = 0.50
        th_p2_p3 = 1.00
        th_m1_m2 = -0.50
        th_m2_m3 = -0.95
        
        def assign(z_a, d_a):
            lbl = np.zeros(len(d_a), dtype=int)
            lbl[(d_a > 0) & (z_a < th_p1_p2)] = 1
            lbl[(d_a > 0) & (z_a >= th_p1_p2) & (z_a < th_p2_p3)] = 2
            lbl[(d_a > 0) & (z_a >= th_p2_p3)] = 3
            lbl[(d_a < 0) & (z_a > th_m1_m2)] = -1
            lbl[(d_a < 0) & (z_a <= th_m1_m2) & (z_a > th_m2_m3)] = -2
            lbl[(d_a < 0) & (z_a <= th_m2_m3)] = -3
            return lbl
            
        l_tr = assign(z_tr, dt_tr)
        l_va = assign(z_va, dt_va)
        
        print(f"\n⏱️ ГОРИЗОНТ H = {H}с (L = {L}с) С УЧЕТОМ КВАНТОВАНИЯ PRL (1-Й УРОВЕНЬ):", flush=True)
        print("Класс | Описание               | Train % (N)       | Val % (N)        | Train E[dt]| Val E[dt]  | Err(dt) | Train E[z] | Val E[z]  | Err(z) | Статус")
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
            m_tr = (l_tr == c)
            m_va = (l_va == c)
            c_tr, c_va = np.sum(m_tr), np.sum(m_va)
            
            dt_m_tr = np.mean(dt_tr[m_tr]) if c_tr > 0 else 0
            dt_m_va = np.mean(dt_va[m_va]) if c_va > 0 else 0
            err_dt = abs(abs(dt_m_va) - abs(dt_m_tr)) / abs(dt_m_tr) * 100 if dt_m_tr != 0 else 0
            
            z_m_tr = np.mean(z_tr[m_tr]) if c_tr > 0 else 0
            z_m_va = np.mean(z_va[m_va]) if c_va > 0 else 0
            err_z = abs(abs(z_m_va) - abs(z_m_tr)) / abs(z_m_tr) * 100 if z_m_tr != 0 else 0
            
            pct_tr = c_tr / len(l_tr) * 100
            pct_va = c_va / len(l_va) * 100
            
            flag = "🎯 < 5%" if err_z < 5.0 else ("✅ < 10%" if err_z < 10.0 else "⚠️ Хвост")
            desc = class_desc[c]
            print(f" {c:+2d}   | {desc} | {pct_tr:5.2f}% ({c_tr:6,d}) | {pct_va:5.2f}% ({c_va:5,d}) | {dt_m_tr:+6.2f} т  | {dt_m_va:+6.2f} т  | {err_dt:5.1f}%  | {z_m_tr:+8.4f}  | {z_m_va:+8.4f}  | {err_z:5.1f}% | {flag}", flush=True)

if __name__ == '__main__':
    test_prl_quantization()
