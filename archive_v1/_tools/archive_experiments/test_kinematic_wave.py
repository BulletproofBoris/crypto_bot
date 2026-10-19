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
        'ST_slippage_buy_100_pct', 'ST_slippage_sell_100_pct', 'ST_delta_vol'
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
    b1_p = df['ST_bid_1_price'].values
    a1_p = df['ST_ask_1_price'].values
    b1_q = df['ST_bid_1_qty'].values
    a1_q = df['ST_ask_1_qty'].values
    
    b5 = df['ST_bids_vol_top5'].values
    a5 = df['ST_asks_vol_top5'].values
    slip_buy = df['ST_slippage_buy_100_pct'].values
    slip_sell = df['ST_slippage_sell_100_pct'].values
    
    # 1. Микроцена (центр тяжести спреда):
    # Microprice = (bid_p * ask_q + ask_p * bid_q) / (bid_q + ask_q)
    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 0, (b1_p * a1_q + a1_p * b1_q) / np.maximum(total_q1, 1.0), mid)
    micro_offset = (microprice - mid) / TICK_SIZE # смещение микроцены относительно mid в тиках (-0.5 .. +0.5)
    
    # 2. Волновое смещение центра масс (скорость набегающей волны за 3 секунды):
    s_offset = pd.Series(micro_offset)
    v_wave = (s_offset - s_offset.shift(3)).fillna(0).values # тиков / 3 сек
    
    # 3. Скорость таяния асков (Ask depletion rate):
    s_a5 = pd.Series(a5)
    d_a5 = (s_a5.shift(3) - s_a5).fillna(0).values # если > 0, аски тают!
    a5_L = s_a5.rolling(L, min_periods=L).mean().values
    depletion_rate = np.clip(d_a5 / np.maximum(a5_L, 50.0), -2.0, 2.0)
    
    # 4. Геометрический наклон профиля стакана:
    # Отношение крутизны асков к крутизне бидов
    geom_skew = np.clip(slip_buy / np.maximum(slip_sell, 0.01), 0.2, 5.0)
    
    # 5. Локальная волатильность за окно L
    step_1s = (mid[1:] - mid[:-1]) / TICK_SIZE
    vol_L = pd.Series(step_1s).rolling(L, min_periods=L).std().values
    
    valid_start = L
    valid_end = n - H
    
    dt = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)[valid_start:valid_end]
    vt = np.nan_to_num(vol_L[valid_start:valid_end])
    sp = spread[valid_start:valid_end]
    m_off = micro_offset[valid_start:valid_end]
    v_w = v_wave[valid_start:valid_end]
    dep = depletion_rate[valid_start:valid_end]
    skew = geom_skew[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    print("=== АНАЛИЗ ВОЛНОВОГО СМЕЩЕНИЯ И ГЕОМЕТРИИ СТАКАНА ПЕРЕД ИМПУЛЬСАМИ ===", flush=True)
    
    # Сравним среднее значение параметров в моменты сильного роста (dt >= 2) против спокойного рынка (dt == 0):
    tr_p2 = (dt[:train_n] >= 2)
    tr_flat = (dt[:train_n] == 0)
    
    print("На обучающей выборке (Train):")
    print(f"  Смещение микроцены (тиков):   при dt>=2: {np.mean(m_off[:train_n][tr_p2]):+.4f} | при флэте: {np.mean(m_off[:train_n][tr_flat]):+.4f}")
    print(f"  Волновой накат v_wave (тиков):при dt>=2: {np.mean(v_w[:train_n][tr_p2]):+.4f} | при флэте: {np.mean(v_w[:train_n][tr_flat]):+.4f}")
    print(f"  Скорость таяния асков:       при dt>=2: {np.mean(dep[:train_n][tr_p2]):+.4f} | при флэте: {np.mean(dep[:train_n][tr_flat]):+.4f}")
    print(f"  Геометрический перекос слип.:при dt>=2: {np.mean(skew[:train_n][tr_p2]):.4f} | при флэте: {np.mean(skew[:train_n][tr_flat]):.4f}")
    
    va_p2 = (dt[train_n:] >= 2)
    va_flat = (dt[train_n:] == 0)
    print("\nНа проверочной выборке (Val):")
    print(f"  Смещение микроцены (тиков):   при dt>=2: {np.mean(m_off[train_n:][va_p2]):+.4f} | при флэте: {np.mean(m_off[train_n:][va_flat]):+.4f}")
    print(f"  Волновой накат v_wave (тиков):при dt>=2: {np.mean(v_w[train_n:][va_p2]):+.4f} | при флэте: {np.mean(v_w[train_n:][va_flat]):+.4f}")
    print(f"  Скорость таяния асков:       при dt>=2: {np.mean(dep[train_n:][va_p2]):+.4f} | при флэте: {np.mean(dep[train_n:][va_flat]):+.4f}")
    print(f"  Геометрический перекос слип.:при dt>=2: {np.mean(skew[train_n:][va_p2]):.4f} | при флэте: {np.mean(skew[train_n:][va_flat]):.4f}")

if __name__ == '__main__':
    main()
