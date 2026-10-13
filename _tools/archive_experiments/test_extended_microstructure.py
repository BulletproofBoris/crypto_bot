import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    path = 'data/features/prl_1s_features.parquet'
    cols = [
        'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
        'ST_imbalance_top5', 'ST_delta_vol', 'mid_price_diff'
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
    imbalance = df['ST_imbalance_top5'].values # from -1.0 to +1.0
    delta_vol = df['ST_delta_vol'].values
    arb_diff_ticks = df['mid_price_diff'].values / TICK_SIZE # in ticks
    
    # 5s price delta in ticks
    delta_usd = mid[H:] - mid[:-H]
    delta_ticks = np.round(delta_usd / TICK_SIZE)
    
    # 1s tick return
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    
    # 1. Local dynamic volatility over window L
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    # 2. Local rolling orderbook depth (bids and asks separately) over window L
    asks_base_L = pd.Series(asks).rolling(L, min_periods=L).median().values
    bids_base_L = pd.Series(bids).rolling(L, min_periods=L).median().values
    
    # 3. Local cumulative taker delta (CVD) over window L
    cvd_L = pd.Series(delta_vol).rolling(L, min_periods=L).sum().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_L[valid_start:valid_end]
    
    a_cur = asks[valid_start:valid_end]
    a_base = asks_base_L[valid_start:valid_end]
    b_cur = bids[valid_start:valid_end]
    b_base = bids_base_L[valid_start:valid_end]
    
    imb = imbalance[valid_start:valid_end]
    cvd = cvd_L[valid_start:valid_end]
    arb = arb_diff_ticks[valid_start:valid_end]
    
    # Resistance ratio (Asks resist UP moves, Bids resist DOWN moves)
    up_res = np.clip(a_base / np.maximum(a_cur, 50.0), 0.5, 3.0)
    dn_res = np.clip(b_base / np.maximum(b_cur, 50.0), 0.5, 3.0)
    
    # Flow pressure: if CVD > 0, pushes UP; if CVD < 0, pushes DOWN
    # Imbalance factor: if imb > 0, more bids -> supports UP, resists DOWN
    # Arbitrage factor: if BigONE is higher (arb > 0), gravity pushes UP
    
    # Let's compute directional dynamic thresholds
    c_base = 2.4
    v_safe = np.nan_to_num(vt, nan=0.1)
    
    # Directional threshold for UP moves:
    # If taker flow is buying (cvd > 0) or BigONE is higher (arb > 0) or imbalance is positive (imb > 0),
    # the threshold needed to trigger +2 is LOWERED or normalized!
    # Conversely, if resistance is high, threshold expands.
    
    # Asymmetric threshold formulas:
    up_thresh = np.maximum(2, np.round(c_base * v_safe * (up_res ** 0.35) * (1.0 - 0.15 * np.clip(imb, -1, 1)))).astype(int)
    dn_thresh = np.maximum(2, np.round(c_base * v_safe * (dn_res ** 0.35) * (1.0 + 0.15 * np.clip(imb, -1, 1)))).astype(int)
    
    train_n = split_idx - valid_start
    
    tr_dt = dt[:train_n]
    tr_up = up_thresh[:train_n]
    tr_dn = dn_thresh[:train_n]
    
    va_dt = dt[train_n:]
    va_up = up_thresh[train_n:]
    va_dn = dn_thresh[train_n:]
    
    def evaluate(dt_arr, up_th, dn_th):
        labels = np.zeros(len(dt_arr), dtype=int)
        labels[(dt_arr > 0) & (dt_arr < up_th)] = 1
        labels[dt_arr >= up_th] = 2
        labels[(dt_arr < 0) & (dt_arr > -dn_th)] = -1
        labels[dt_arr <= -dn_th] = -2
        return labels

    tr_lbl = evaluate(tr_dt, tr_up, tr_dn)
    va_lbl = evaluate(va_dt, va_up, va_dn)
    
    print('=== РЕЗУЛЬТАТЫ С АСИММЕТРИЕЙ СТОРОН И ДИСБАЛАНСОМ СТАКАНА ===', flush=True)
    print('Класс | Train % (Сэмплов)  | Val % (Сэмплов)   | Train сдвиг | Val сдвиг', flush=True)
    print('-' * 65, flush=True)
    for c in [-2, -1, 0, 1, 2]:
        tr_cnt = np.sum(tr_lbl == c)
        tr_pct = tr_cnt / len(tr_lbl) * 100
        tr_m = np.mean(tr_dt[tr_lbl == c]) if tr_cnt > 0 else 0
        
        va_cnt = np.sum(va_lbl == c)
        va_pct = va_cnt / len(va_lbl) * 100
        va_m = np.mean(va_dt[va_lbl == c]) if va_cnt > 0 else 0
        
        print(f' {c:+2d}   | {tr_pct:5.2f}% ({tr_cnt:6,}) | {va_pct:5.2f}% ({va_cnt:6,}) | {tr_m:+.2f} тика | {va_m:+.2f} тика', flush=True)
    print('-' * 65, flush=True)

if __name__ == '__main__':
    main()
