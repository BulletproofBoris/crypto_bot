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
        'ST_cvd_1m', 'ST_delta_vol', 'mid_price_diff'
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
    slip_buy = df['ST_slippage_buy_100_pct'].values # слиппедж покупки на 100$
    slip_sell = df['ST_slippage_sell_100_pct'].values # слиппедж продажи на 100$
    cvd1m = df['ST_cvd_1m'].values
    delta_vol = df['ST_delta_vol'].values
    arb_diff = df['mid_price_diff'].values / TICK_SIZE # спред BigONE vs SafeTrade в тиках
    
    delta_ticks = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    
    # 1. Локальная волатильность за окно L (в тиках)
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    # 2. Производная волатильности (ускорение волатильности за последние 3-5 сек)
    # d(Vol)/dt: волна начинает закручиваться
    vol_series = pd.Series(vol_L)
    d_vol = (vol_series - vol_series.shift(3)).fillna(0).values
    
    # 3. Геометрия стакана:
    # Отношение слиппеджа покупки к среднему за окно L
    # Слиппедж покупки прямо отражает крутизну (наклон) стены асков!
    slip_buy_L = pd.Series(slip_buy).rolling(L, min_periods=L).mean().values
    # Относительная кривизна / крутизна асков:
    ask_slope_factor = np.clip(slip_buy / np.maximum(slip_buy_L, 0.01), 0.2, 5.0)
    
    # 4. Волновой кинематический фактор смещения центра масс (Order Flow / Imbalance drift):
    # Скорость смещения дисбаланса за последние 3-5 секунд:
    # d(Imbalance)/dt: накат волны ликвидности в сторону покупателей
    imb_series = pd.Series(imb5)
    d_imb_3s = (imb_series - imb_series.shift(3)).fillna(0).values
    
    # 5. Скорость притока сделок (производная CVD / торговая инерция)
    cvd_momentum = pd.Series(delta_vol).rolling(5, min_periods=5).sum().values
    
    # Базовые скользящие средние глубины
    asks_L = pd.Series(asks5).rolling(L, min_periods=L).mean().values
    bids_L = pd.Series(bids5).rolling(L, min_periods=L).mean().values
    
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
    
    geom_ask = ask_slope_factor[valid_start:valid_end]
    wave_imb = d_imb_3s[valid_start:valid_end]
    wave_cvd = cvd_momentum[valid_start:valid_end]
    wave_arb = arb_diff[valid_start:valid_end]
    
    train_n = split_idx - valid_start
    
    print("=== ИССЛЕДОВАНИЕ ГЕОМЕТРИИ СТАКАНА И КИНЕМАТИЧЕСКОЙ ВОЛНЫ ===", flush=True)
    
    # Посмотрим, как вели себя геометрические и волновые метрики на Train и Val перед сильным ростом (dt >= 2):
    tr_dt, va_dt = dt[:train_n], dt[train_n:]
    tr_geom, va_geom = geom_ask[:train_n], geom_ask[train_n:]
    tr_w_imb, va_w_imb = wave_imb[:train_n], wave_imb[train_n:]
    tr_w_cvd, va_w_cvd = wave_cvd[:train_n], wave_cvd[train_n:]
    tr_w_arb, va_w_arb = wave_arb[:train_n], wave_arb[train_n:]
    tr_dvt, va_dvt = dvt[:train_n], dvt[train_n:]
    
    tr_mask_p2 = (tr_dt >= 2)
    va_mask_p2 = (va_dt >= 2)
    
    print(f"\n1. Геометрический фактор наклона асков (ST_slippage_buy / base_L) при dt >= 2:")
    print(f"   Train: mean = {np.mean(tr_geom[tr_mask_p2]):.3f} | median = {np.median(tr_geom[tr_mask_p2]):.3f}")
    print(f"   Val:   mean = {np.mean(va_geom[va_mask_p2]):.3f} | median = {np.median(va_geom[va_mask_p2]):.3f}")
    
    print(f"\n2. Волновое смещение дисбаланса d(Imbalance)/dt_3s при dt >= 2:")
    print(f"   Train: mean = {np.mean(tr_w_imb[tr_mask_p2]):+.4f} | median = {np.median(tr_w_imb[tr_mask_p2]):+.4f}")
    print(f"   Val:   mean = {np.mean(va_w_imb[va_mask_p2]):+.4f} | median = {np.median(va_w_imb[va_mask_p2]):+.4f}")
    
    print(f"\n3. Ускорение волатильности d(Vol)/dt_3s при dt >= 2:")
    print(f"   Train: mean = {np.mean(tr_dvt[tr_mask_p2]):+.4f} | median = {np.median(tr_dvt[tr_mask_p2]):+.4f}")
    print(f"   Val:   mean = {np.mean(va_dvt[va_mask_p2]):+.4f} | median = {np.median(va_dvt[va_mask_p2]):+.4f}")
    
    print(f"\n4. Кросс-биржевое опережение BigONE vs SafeTrade (тиков) при dt >= 2:")
    print(f"   Train: mean = {np.mean(tr_w_arb[tr_mask_p2]):+.3f} | median = {np.median(tr_w_arb[tr_mask_p2]):+.3f}")
    print(f"   Val:   mean = {np.mean(va_w_arb[va_mask_p2]):+.3f} | median = {np.median(va_w_arb[va_mask_p2]):+.3f}")

if __name__ == '__main__':
    main()
