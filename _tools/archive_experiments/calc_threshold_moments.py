import sys
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    print("Loading data from prl_1s_features.parquet...", flush=True)
    # Load mid price from prl_1s_features directly (already 1-second continuous!)
    cols = ['ST_mid_price', 'ST_spread_pct']
    path = 'data/features/prl_1s_features.parquet'
    if not os.path.exists(path):
        path = 'data/processed/prl_1s_features.parquet'
    tbl = pq.read_table(path, columns=cols)
    df = tbl.to_pandas()
    
    mid = df['ST_mid_price'].values
    spread_pct = df['ST_spread_pct'].values
    n = len(df)
    
    print(f"Total 1s ticks: {n:,}", flush=True)
    
    # Train / Val Split: 80% train, 20% validation
    split_idx = int(n * 0.8)
    
    train_mid = mid[:split_idx]
    train_spread = spread_pct[:split_idx]
    
    print(f"Train split size: {split_idx:,} (80%) | Val split size: {n - split_idx:,} (20%)", flush=True)
    
    # 5s future returns: (mid[t+5] - mid[t]) / mid[t] * 100%
    ret_5s = (train_mid[5:] - train_mid[:-5]) / train_mid[:-5] * 100.0
    
    # 20s rolling volatility of 1s returns: std over rolling 20 ticks
    ret_1s = (train_mid[1:] - train_mid[:-1]) / train_mid[:-1] * 100.0
    
    # Efficient rolling std over 20 ticks using numpy stride/convolution
    window = 20
    # Rolling std with numpy
    r1_series = pd.Series(ret_1s)
    vol_20s = r1_series.rolling(window, min_periods=window).std().values
    
    # Align indices
    # ret_5s has length split_idx - 5
    # vol_20s has length split_idx - 1
    valid_start = window
    valid_end = split_idx - 5
    
    r5 = ret_5s[valid_start:valid_end]
    v20 = vol_20s[valid_start:valid_end]
    spr = train_spread[valid_start:valid_end]
    
    # Filter valid non-zero volatility
    mask = ~np.isnan(v20) & (v20 > 1e-6)
    r5 = r5[mask]
    v20 = v20[mask]
    spr = spr[mask]
    
    # Normalized movement coefficient: z = ret_5s / vol_20s
    z = r5 / v20
    
    print("\n" + "="*60, flush=True)
    print("📊 СТАТИСТИЧЕСКИЕ ПАРАМЕТРЫ НА ОТРЕЗКЕ ОБУЧЕНИЯ (80% данных)", flush=True)
    print("="*60, flush=True)
    
    print("\n1. Доходность за 5 секунд (Return 5s, %):", flush=True)
    print(f"   Математическое ожидание: {np.mean(r5):+.5f}%", flush=True)
    print(f"   Дисперсия:              {np.var(r5):.6f}", flush=True)
    print(f"   Стандартное отклонение:  {np.std(r5):.5f}%", flush=True)
    print(f"   Медиана:                {np.percentile(r5, 50):+.5f}%", flush=True)
    print(f"   Размах (10% ... 90%):    {np.percentile(r5, 10):+.4f}% ... {np.percentile(r5, 90):+.4f}%", flush=True)
    
    print("\n2. Локальная волатильность за 20 секунд (Vol 20s, %):", flush=True)
    print(f"   Математическое ожидание: {np.mean(v20):.5f}%", flush=True)
    print(f"   Дисперсия:              {np.var(v20):.6f}", flush=True)
    print(f"   Стандартное отклонение:  {np.std(v20):.5f}%", flush=True)
    print(f"   Медиана:                {np.percentile(v20, 50):.5f}%", flush=True)
    
    print("\n3. Спред стакана SafeTrade (Spread %):", flush=True)
    print(f"   Математическое ожидание: {np.mean(spr):.4f}%", flush=True)
    print(f"   Медиана спреда:         {np.percentile(spr, 50):.4f}%", flush=True)
    print(f"   75% перцентиль:         {np.percentile(spr, 75):.4f}%", flush=True)
    
    print("\n4. Нормированный коэффициент динамического порога (z = Return_5s / Vol_20s):", flush=True)
    print(f"   Математическое ожидание (E[z]): {np.mean(z):+.4f}", flush=True)
    print(f"   Дисперсия (Var[z]):            {np.var(z):.4f}", flush=True)
    print(f"   Стандартное отклонение (Std[z]):{np.std(z):.4f}", flush=True)
    
    print("\n5. Границы квантилей для 5 сбалансированных классов (по 20% в каждом):", flush=True)
    q20 = np.percentile(z, 20)
    q40 = np.percentile(z, 40)
    q60 = np.percentile(z, 60)
    q80 = np.percentile(z, 80)
    
    print(f"   Порог Класс -2 (Сильный спад, нижние 20%):   z < {q20:+.3f}", flush=True)
    print(f"   Порог Класс -1 (Умеренный спад, 20% - 40%):   {q20:+.3f} <= z < {q40:+.3f}", flush=True)
    print(f"   Порог Класс  0 (Шум / Флэт, 40% - 60%):       {q40:+.3f} <= z < {q60:+.3f}", flush=True)
    print(f"   Порог Класс +1 (Умеренный рост, 60% - 80%):   {q60:+.3f} <= z < {q80:+.3f}", flush=True)
    print(f"   Порог Класс +2 (Сильный рост, верхние 20%):   z >= {q80:+.3f}", flush=True)

if __name__ == '__main__':
    main()
