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
    TICK_SIZE = 0.005
    H = 5
    L = 20
    
    # Extract series
    mid = df['ST_mid_price'].values
    spread = df['ST_spread'].values
    depth = (df['ST_bids_vol_top5'] + df['ST_asks_vol_top5']).values
    
    # 5s price delta in ticks
    delta_usd = mid[H:] - mid[:-H]
    delta_ticks = np.round(delta_usd / TICK_SIZE)
    
    # 20s rolling volatility in ticks
    step_1s_usd = mid[1:] - mid[:-1]
    step_1s_ticks = step_1s_usd / TICK_SIZE
    vol_ticks = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_ticks[valid_start:valid_end]
    sp = spread[valid_start:valid_end] / TICK_SIZE # spread in ticks
    dp = depth[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    # Train slice
    tr_dt = dt[:train_n]
    tr_vt = vt[:train_n]
    tr_sp = sp[:train_n]
    tr_dp = dp[:train_n]
    
    # Val slice
    va_dt = dt[train_n:]
    va_vt = vt[train_n:]
    va_sp = sp[train_n:]
    va_dp = dp[train_n:]
    
    # Baseline statistics from Train ONLY (no lookahead!)
    MEDIAN_DEPTH = np.median(tr_dp[tr_dp > 0])
    MEDIAN_SPREAD = np.median(tr_sp[tr_sp > 0])
    
    print(f"Базовые медианы (строго с Обучения):", flush=True)
    print(f"  Медианная глубина: {MEDIAN_DEPTH:.1f} PRL", flush=True)
    print(f"  Медианный спред:   {MEDIAN_SPREAD:.2f} тиков", flush=True)
    
    # Multi-factor dynamic threshold
    # When depth is thin (dp < MEDIAN_DEPTH), ratio > 1 -> threshold increases to demand bigger move
    # When spread is wide (sp > MEDIAN_SPREAD), ratio > 1 -> threshold increases
    def calc_threshold(vt_arr, sp_arr, dp_arr, c_vol=2.5, c_depth=0.3, c_spread=0.2):
        v_safe = np.nan_to_num(vt_arr, nan=0.1)
        
        # Depth elasticity factor: if liquidity is half, threshold expands
        depth_safe = np.maximum(dp_arr, 1000.0)
        depth_factor = (MEDIAN_DEPTH / depth_safe) ** c_depth
        depth_factor = np.clip(depth_factor, 0.5, 3.0)
        
        # Spread elasticity factor
        spread_safe = np.maximum(sp_arr, 1.0)
        spread_factor = (spread_safe / MEDIAN_SPREAD) ** c_spread
        spread_factor = np.clip(spread_factor, 0.5, 3.0)
        
        # Combined dynamic threshold
        dyn_th = np.maximum(2, np.round(c_vol * v_safe * depth_factor * spread_factor)).astype(int)
        return dyn_th

    # Test baseline (pure vol) vs Multifactor
    tr_th = calc_threshold(tr_vt, tr_sp, tr_dp)
    va_th = calc_threshold(va_vt, va_sp, va_dp)
    
    print("\n" + "="*65, flush=True)
    print("📊 СРАВНЕНИЕ МНОГОФАКТОРНОГО ПОРОГА НА TRAIN И VAL", flush=True)
    print("="*65, flush=True)
    print(f"Средний порог на Train:      {np.mean(tr_th):.2f} тика | мин={np.min(tr_th)} | макс={np.max(tr_th)}", flush=True)
    print(f"Средний порог на Validation: {np.mean(va_th):.2f} тика | мин={np.min(va_th)} | макс={np.max(va_th)}", flush=True)
    
    def evaluate(dt_arr, th_arr):
        labels = np.zeros(len(dt_arr), dtype=int)
        labels[(dt_arr > 0) & (dt_arr < th_arr)] = 1
        labels[dt_arr >= th_arr] = 2
        labels[(dt_arr < 0) & (dt_arr > -th_arr)] = -1
        labels[dt_arr <= -th_arr] = -2
        return labels

    tr_lbl = evaluate(tr_dt, tr_th)
    va_lbl = evaluate(va_dt, va_th)
    
    print("\nРаспределение классов с учетом глубины стакана и спреда:")
    print("Класс | Train % (Сэмплов)  | Val % (Сэмплов)   | Train сдвиг | Val сдвиг", flush=True)
    print("-" * 65, flush=True)
    for c in [-2, -1, 0, 1, 2]:
        tr_cnt = np.sum(tr_lbl == c)
        tr_pct = tr_cnt / len(tr_lbl) * 100
        tr_m = np.mean(tr_dt[tr_lbl == c]) if tr_cnt > 0 else 0
        
        va_cnt = np.sum(va_lbl == c)
        va_pct = va_cnt / len(va_lbl) * 100
        va_m = np.mean(va_dt[va_lbl == c]) if va_cnt > 0 else 0
        
        print(f" {c:+2d}   | {tr_pct:5.2f}% ({tr_cnt:6,}) | {va_pct:5.2f}% ({va_cnt:6,}) | {tr_m:+.2f} тика | {va_m:+.2f} тика", flush=True)
    print("-" * 65, flush=True)

if __name__ == '__main__':
    main()
