import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    path = 'data/features/prl_1s_features.parquet'
    cols = ['ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5']
    tbl = pq.read_table(path, columns=cols)
    df = tbl.to_pandas()
    
    n = len(df)
    split_idx = int(n * 0.8)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print('=== СРАВНЕНИЕ РЫНОЧНЫХ УСЛОВИЙ: TRAIN VS VALIDATION ===', flush=True)
    print('1. Средняя ширина спреда в USD:', flush=True)
    print(f'   Train: {train_df["ST_spread"].mean():.5f} USD ({train_df["ST_spread"].mean()/0.005:.2f} тиков)', flush=True)
    print(f'   Val:   {val_df["ST_spread"].mean():.5f} USD ({val_df["ST_spread"].mean()/0.005:.2f} тиков)', flush=True)
    
    print('\n2. Средняя глубина стакана (Bids + Asks top-5):', flush=True)
    train_depth = train_df['ST_bids_vol_top5'] + train_df['ST_asks_vol_top5']
    val_depth = val_df['ST_bids_vol_top5'] + val_df['ST_asks_vol_top5']
    print(f'   Train: {train_depth.mean():.1f} PRL (Медиана: {train_depth.median():.1f} PRL)', flush=True)
    print(f'   Val:   {val_depth.mean():.1f} PRL (Медиана: {val_depth.median():.1f} PRL)', flush=True)
    
    mid = df['ST_mid_price'].values
    ret_1s = (mid[1:] - mid[:-1]) / mid[:-1] * 100.0
    r1_series = pd.Series(ret_1s)
    vol_20s = r1_series.rolling(20, min_periods=20).std().values
    
    train_vol = vol_20s[:split_idx]
    val_vol = vol_20s[split_idx:]
    print('\n3. Волатильность (20s rolling std, %):', flush=True)
    print(f'   Train: {np.nanmean(train_vol):.4f}% (Std: {np.nanstd(train_vol):.4f}%)', flush=True)
    print(f'   Val:   {np.nanmean(val_vol):.4f}% (Std: {np.nanstd(val_vol):.4f}%)', flush=True)

if __name__ == '__main__':
    main()
