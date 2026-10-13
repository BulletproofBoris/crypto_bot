import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def analyze():
    path = 'data/features/prl_1s_features.parquet'
    cols = [
        'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
        'ST_delta_vol', 'mid_price_diff'
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
    bids = df['ST_bids_vol_top5'].values
    asks = df['ST_asks_vol_top5'].values
    delta_vol = df['ST_delta_vol'].values
    arb_diff = df['mid_price_diff'].values / TICK_SIZE
    
    delta_usd = mid[H:] - mid[:-H]
    delta_ticks = np.round(delta_usd / TICK_SIZE)
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    
    # Let's check different volatility definitions over window L:
    # 1. Rolling std of 1s returns
    vol_std_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    # 2. Rolling Parkinson or High-Low range over L? (mid_max - mid_min over window L)
    s_mid = pd.Series(mid / TICK_SIZE)
    range_L = (s_mid.rolling(L, min_periods=L).max() - s_mid.rolling(L, min_periods=L).min()).values
    
    # 3. Rolling mean absolute return (MAD) over L
    vol_mad_L = pd.Series(np.abs(step_1s_ticks)).rolling(L, min_periods=L).mean().values
    
    # 4. Total volume and depth over L
    depth_total = bids + asks
    depth_base_L = pd.Series(depth_total).rolling(L, min_periods=L).median().values
    spread_base_L = pd.Series(spread).rolling(L, min_periods=L).median().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    v_std = vol_std_L[valid_start:valid_end]
    v_rng = range_L[valid_start:valid_end]
    v_mad = vol_mad_L[valid_start:valid_end]
    
    dp_cur = depth_total[valid_start:valid_end]
    dp_base = depth_base_L[valid_start:valid_end]
    sp_cur = spread[valid_start:valid_end]
    sp_base = spread_base_L[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    print("=== АНАЛИЗ МЕТРИК НА ОКНЕ L=20 ДЛЯ РАЗНЫХ РЕЖИМОВ ===", flush=True)
    print(f"Train samples: {train_n:,} | Val samples: {len(dt) - train_n:,}", flush=True)
    
    print("\n1. Волатильность (Rolling Range L=20):", flush=True)
    print(f"   Train range median: {np.median(v_rng[:train_n]):.2f} тиков | mean: {np.mean(v_rng[:train_n]):.2f} тиков", flush=True)
    print(f"   Val range median:   {np.median(v_rng[train_n:]):.2f} тиков | mean: {np.mean(v_rng[train_n:]):.2f} тиков", flush=True)
    
    print("\n2. Волатильность (Rolling MAD L=20):", flush=True)
    print(f"   Train MAD median: {np.median(v_mad[:train_n]):.4f} тиков | mean: {np.mean(v_mad[:train_n]):.4f} тиков", flush=True)
    print(f"   Val MAD median:   {np.median(v_mad[train_n:]):.4f} тиков | mean: {np.mean(v_mad[train_n:]):.4f} тиков", flush=True)
    
    print("\n3. Отношение глубины стакана (cur / base_L):", flush=True)
    dp_ratio = dp_cur / np.maximum(dp_base, 50.0)
    print(f"   Train dp_ratio median: {np.median(dp_ratio[:train_n]):.2f} | mean: {np.mean(dp_ratio[:train_n]):.2f}", flush=True)
    print(f"   Val dp_ratio median:   {np.median(dp_ratio[train_n:]):.2f} | mean: {np.mean(dp_ratio[train_n:]):.2f}", flush=True)
    
    print("\n4. Отношение спреда (cur / base_L):", flush=True)
    sp_ratio = sp_cur / np.maximum(sp_base, 0.5)
    print(f"   Train sp_ratio median: {np.median(sp_ratio[:train_n]):.2f} | mean: {np.mean(sp_ratio[:train_n]):.2f}", flush=True)
    print(f"   Val sp_ratio median:   {np.median(sp_ratio[train_n:]):.2f} | mean: {np.mean(sp_ratio[train_n:]):.2f}", flush=True)

if __name__ == '__main__':
    analyze()
