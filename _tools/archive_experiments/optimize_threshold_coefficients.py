import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize

def main():
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
    spread = df['ST_spread'].values / TICK_SIZE # in ticks
    bids = df['ST_bids_vol_top5'].values
    asks = df['ST_asks_vol_top5'].values
    
    # 5s price delta in ticks
    delta_usd = mid[H:] - mid[:-H]
    delta_ticks = np.round(delta_usd / TICK_SIZE)
    
    # 1s tick return
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    
    # Dynamic rolling metrics over window L
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    depth_total = bids + asks
    depth_base_L = pd.Series(depth_total).rolling(L, min_periods=L).median().values
    spread_base_L = pd.Series(spread).rolling(L, min_periods=L).median().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_L[valid_start:valid_end]
    cur_dp = depth_total[valid_start:valid_end]
    base_dp = depth_base_L[valid_start:valid_end]
    cur_sp = spread[valid_start:valid_end]
    base_sp = spread_base_L[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    tr_dt = dt[:train_n]
    tr_vt = vt[:train_n]
    tr_cur_dp = cur_dp[:train_n]
    tr_base_dp = base_dp[:train_n]
    tr_cur_sp = cur_sp[:train_n]
    tr_base_sp = base_sp[:train_n]
    
    va_dt = dt[train_n:]
    va_vt = vt[train_n:]
    va_cur_dp = cur_dp[train_n:]
    va_base_dp = base_dp[train_n:]
    va_cur_sp = cur_sp[train_n:]
    va_base_sp = base_sp[train_n:]
    
    print("=== ПОИСК ТОЧНЫХ ПОПРАВОЧНЫХ КОЭФФИЦИЕНТОВ ЧЕРЕЗ ОПТИМИЗАЦИЮ ===", flush=True)
    
    # Let's inspect the correlation of delta_ticks with features when a move occurs (|dt| >= 1)
    move_mask = (np.abs(tr_dt) >= 1)
    abs_dt = np.abs(tr_dt)[move_mask]
    v_m = tr_vt[move_mask]
    dp_ratio_m = tr_base_dp[move_mask] / np.maximum(tr_cur_dp[move_mask], 50.0)
    sp_ratio_m = tr_cur_sp[move_mask] / np.maximum(tr_base_sp[move_mask], 0.5)
    
    # Fit a power-law model for expected move magnitude on Train:
    # Expected_Move = c * (vol ** p_vol) * (dp_ratio ** p_depth) * (sp_ratio ** p_spread)
    # Using log-linear regression:
    log_y = np.log(abs_dt)
    log_v = np.log(np.maximum(v_m, 0.01))
    log_dp = np.log(np.clip(dp_ratio_m, 0.1, 10.0))
    log_sp = np.log(np.clip(sp_ratio_m, 0.1, 10.0))
    
    X = np.column_stack([np.ones_like(log_v), log_v, log_dp, log_sp])
    
    # OLS estimation on Train
    betas, residuals, rank, s = np.linalg.lstsq(X, log_y, rcond=None)
    
    c_mult = np.exp(betas[0])
    p_vol = betas[1]
    p_depth = betas[2]
    p_spread = betas[3]
    
    print("Результаты регрессии физических законов на Train:", flush=True)
    print(f"  Базовый множитель c:     {c_mult:.4f}", flush=True)
    print(f"  Степень волатильности:   {p_vol:.4f}", flush=True)
    print(f"  Степень истощения стакана:{p_depth:.4f}", flush=True)
    print(f"  Степень спреда:          {p_spread:.4f}", flush=True)
    
    # Function to calculate threshold based on fitted physics
    def calc_opt_thresh(vt_arr, b_dp, c_dp, b_sp, c_sp, scale=1.8):
        v_cl = np.maximum(np.nan_to_num(vt_arr, nan=0.1), 0.01)
        dp_r = np.clip(b_dp / np.maximum(c_dp, 50.0), 0.2, 5.0)
        sp_r = np.clip(c_sp / np.maximum(b_sp, 0.5), 0.2, 5.0)
        
        # Expected baseline magnitude of move
        pred_mag = c_mult * (v_cl ** p_vol) * (dp_r ** p_depth) * (sp_r ** p_spread)
        
        # Threshold for Class +2 / -2: scale * predicted magnitude, minimum 2 ticks
        th = np.maximum(2, np.round(scale * pred_mag)).astype(int)
        return th

    # Let's grid search scale parameter on Train to balance classes
    best_scale = 1.6
    
    tr_th = calc_opt_thresh(tr_vt, tr_base_dp, tr_cur_dp, tr_base_sp, tr_cur_sp, scale=best_scale)
    va_th = calc_opt_thresh(va_vt, va_base_dp, va_cur_dp, va_base_sp, va_cur_sp, scale=best_scale)
    
    def evaluate(dt_arr, th_arr):
        labels = np.zeros(len(dt_arr), dtype=int)
        labels[(dt_arr > 0) & (dt_arr < th_arr)] = 1
        labels[dt_arr >= th_arr] = 2
        labels[(dt_arr < 0) & (dt_arr > -th_arr)] = -1
        labels[dt_arr <= -th_arr] = -2
        return labels

    tr_lbl = evaluate(tr_dt, tr_th)
    va_lbl = evaluate(va_dt, va_th)
    
    print("\n" + "="*70, flush=True)
    print("📊 СРАВНЕНИЕ С МОДЕЛЬЮ, ОБУЧЕННОЙ ПО ФИЗИКЕ РЫНКА (TRAIN VS VAL)", flush=True)
    print("="*70, flush=True)
    print("Класс | Train % (Сэмплов)  | Val % (Сэмплов)   | Train сдвиг | Val сдвиг | Ошибка (%)", flush=True)
    print("-" * 75, flush=True)
    for c in [-2, -1, 0, 1, 2]:
        tr_cnt = np.sum(tr_lbl == c)
        tr_pct = tr_cnt / len(tr_lbl) * 100
        tr_m = np.mean(tr_dt[tr_lbl == c]) if tr_cnt > 0 else 0
        
        va_cnt = np.sum(va_lbl == c)
        va_pct = va_cnt / len(va_lbl) * 100
        va_m = np.mean(va_dt[va_lbl == c]) if va_cnt > 0 else 0
        
        err = abs(abs(va_m) - abs(tr_m)) / abs(tr_m) * 100 if tr_m != 0 else 0
        print(f" {c:+2d}   | {tr_pct:5.2f}% ({tr_cnt:6,}) | {va_pct:5.2f}% ({va_cnt:6,}) | {tr_m:+.2f} тика | {va_m:+.2f} тика | {err:5.1f}%", flush=True)
    print("-" * 75, flush=True)

if __name__ == '__main__':
    main()
