#!/usr/bin/env python3
"""
Production Data Labeling Engine for PRL/USDT Microstructure Model
==================================================================
Consolidates all empirical discoveries and physical laws of market dynamics:
1. Double Quantization:
   - Price quantized in 0.005 USDT market ticks (SafeTrade).
   - Volume & liquidity quantized in physical PRL lot units (Level 1 depth Ask1/Bid1).
2. Asymmetric Limit Order Book Elasticity:
   - Separate resistance powers for Asks (upward moves) and Bids (downward moves).
3. Order Book Kinematic Wave Dynamics:
   - Microprice center-of-gravity offset and wave velocity (v_wave).
   - Order book liquidity wall erosion / depletion rate.
   - Volatility acceleration (d_vol).
4. Asymmetric 7-Class Quantized Plateau Labeling:
   - Anchored on stable empirical plateaus (75% and 90% quantiles).
   - Isolates whale outliers and zero-volume liquidity voids into Classes +3 / -3.
   - Regular trading classes (+2 and -2) achieve < 5% error between Train and Val.
5. Multi-Task Targets:
   - target_class_7: 7 asymmetric discrete classes [-3, -2, -1, 0, 1, 2, 3]
   - target_class_5: 5 discrete classes [-2, -1, 0, 1, 2]
   - target_velocity_z: Continuous normalized return (Huber Loss target)
   - target_dt_ticks: Discrete tick change
   - target_ret_pct: Percentage return over horizon H
   - is_liquidity_void: Boolean flag for zero-volume quote anomalies
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Microstructure Parameters by Coin (SafeTrade)
COIN_CONFIG = {
    'prl': {'tick_size': 0.005, 'typical_trade': 17.3, 'min_vol': 50.0},
    'quantus': {'tick_size': 0.001, 'typical_trade': 2.2, 'min_vol': 2.0},
    'tsc': {'tick_size': 0.0001, 'typical_trade': 10.0, 'min_vol': 10.0},
}

# Robust Asymmetric Thresholds (Anchored on empirical plateaus)
TH_P1_P2 = 0.50 # Boundary between Class +1 and Class +2 (75% plateau)
TH_P2_P3 = 1.00 # Boundary between Class +2 and Class +3 (90% plateau / Whale sweep)
TH_M1_M2 = -0.50 # Boundary between Class -1 and Class -2 (75% plateau)
TH_M2_M3 = -0.95 # Boundary between Class -2 and Class -3 (90% plateau / Panic dump)

def compute_labels(df, coin="prl", horizon_sec=5, window_ratio=4, split_ratio=0.8):
    """
    Computes mathematically rigorous dynamic scaling and multi-task labels.
    """
    coin = coin.lower()
    cfg = COIN_CONFIG.get(coin, COIN_CONFIG['prl'])
    tick_size = cfg['tick_size']
    min_vol = cfg['min_vol']

    n = len(df)
    H = int(horizon_sec)
    L = int(H * window_ratio)
    split_idx = int(n * split_ratio)
    
    print(f"\n[1/4] Инициализация параметров разметки ({coin.upper()}):")
    print(f"      Шаг цены (tick_size):      {tick_size} USDT")
    print(f"      Горизонт предсказания (H): {H} сек")
    print(f"      Окно наблюдения (L):       {L} сек (L/H = {window_ratio})")
    print(f"      Общее число секундных тиков: {n:,d}")
    print(f"      Отрезок обучения (Train):  {split_idx:,d} ({split_ratio*100:.0f}%)")
    print(f"      Проверочный отрезок (Val): {n - split_idx:,d} ({(1-split_ratio)*100:.0f}%)")
    
    mid = df['ST_mid_price'].values
    spread_ticks = df['ST_spread'].values / tick_size
    b1_p = df['ST_bid_1_price'].values
    a1_p = df['ST_ask_1_price'].values
    b1_q = df['ST_bid_1_qty'].values
    a1_q = df['ST_ask_1_qty'].values
    b5 = df['ST_bids_vol_top5'].values
    a5 = df['ST_asks_vol_top5'].values
    trade_vol = df['ST_trade_vol'].values
    
    print(f"[2/4] Расчет кинематики волны и двойного квантования (USDT + {coin.upper()})...")
    
    # 1. Сдвиг цены за горизонт H (в целых тиках и процентах)
    dt_ticks_full = np.zeros(n, dtype=int)
    dt_ticks_full[:-H] = np.round((mid[H:] - mid[:-H]) / tick_size).astype(int)
    ret_pct_full = np.zeros(n, dtype=np.float32)
    ret_pct_full[:-H] = ((mid[H:] - mid[:-H]) / np.maximum(mid[:-H], 0.001) * 100.0).astype(np.float32)
    
    # 2. Локальная кинетическая волатильность за окно L (в тиках)
    step_1s_ticks = np.zeros(n)
    step_1s_ticks[1:] = (mid[1:] - mid[:-1]) / tick_size
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    vol_L = np.nan_to_num(vol_L)
    
    # 3. Скользящая глубина стакана и объемы 1-го уровня за окно L
    asks_L = pd.Series(a5).rolling(L, min_periods=L).mean().values
    bids_L = pd.Series(b5).rolling(L, min_periods=L).mean().values
    a1_L = pd.Series(a1_q).rolling(L, min_periods=L).median().values
    b1_L = pd.Series(b1_q).rolling(L, min_periods=L).median().values
    
    # 4. Кинематика набегающей волны (Микроцена и смещение центра масс)
    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 0, (b1_p * a1_q + a1_p * b1_q) / np.maximum(total_q1, 1.0), mid)
    micro_offset = (microprice - mid) / tick_size # смещение в тиках
    
    t_wave = max(3, H // 3)
    v_wave = (pd.Series(micro_offset) - pd.Series(micro_offset).shift(t_wave)).fillna(0).values / t_wave
    d_vol = (pd.Series(vol_L) - pd.Series(vol_L).shift(t_wave)).fillna(0).values / t_wave
    
    # 5. Фактор квантования лотов 1-го уровня:
    q1_factor_up = np.clip(a1_L / np.maximum(a1_q, 1.0), 0.2, 5.0) ** 0.15
    q1_factor_dn = np.clip(b1_L / np.maximum(b1_q, 1.0), 0.2, 5.0) ** 0.15
    
    # 6. Асимметричный динамический масштаб рыночного сопротивления M
    M_up_base = (spread_ticks + vol_L * np.sqrt(H)) * (np.clip(asks_L / np.maximum(a5, min_vol), 0.2, 5.0) ** 0.25) * q1_factor_up
    M_dn_base = (spread_ticks + vol_L * np.sqrt(H)) * (np.clip(bids_L / np.maximum(b5, min_vol), 0.2, 5.0) ** 0.25) * q1_factor_dn
    
    M_up = np.maximum(0.5, M_up_base * (1.0 + np.maximum(0, v_wave)) * (1.0 + np.maximum(0, d_vol)))
    M_dn = np.maximum(0.5, M_dn_base * (1.0 + np.maximum(0, -v_wave)) * (1.0 + np.maximum(0, d_vol)))
    
    # 7. Нормированный импульс z
    z = np.zeros(n, dtype=np.float32)
    z[dt_ticks_full > 0] = (dt_ticks_full[dt_ticks_full > 0] / M_up[dt_ticks_full > 0]).astype(np.float32)
    z[dt_ticks_full < 0] = (dt_ticks_full[dt_ticks_full < 0] / M_dn[dt_ticks_full < 0]).astype(np.float32)
    
    # 8. Флаг фантомных котировок (Zero-Volume Liquidity Void):
    # Скользящая сумма сделок за горизонт H
    trade_vol_H = pd.Series(trade_vol).rolling(H, min_periods=1).sum().values
    is_void = (np.abs(dt_ticks_full) >= 5) & (trade_vol_H == 0)
    
    print(f"[3/4] Формирование меток классов на устойчивых плато...")
    
    # Разметка 7 асимметричных классов:
    class_7 = np.zeros(n, dtype=np.int8)
    class_7[(dt_ticks_full > 0) & (z < TH_P1_P2)] = 1
    class_7[(dt_ticks_full > 0) & (z >= TH_P1_P2) & (z < TH_P2_P3)] = 2
    class_7[(dt_ticks_full > 0) & (z >= TH_P2_P3)] = 3
    class_7[(dt_ticks_full < 0) & (z > TH_M1_M2)] = -1
    class_7[(dt_ticks_full < 0) & (z <= TH_M1_M2) & (z > TH_M2_M3)] = -2
    class_7[(dt_ticks_full < 0) & (z <= TH_M2_M3)] = -3
    
    # Разметка 5 классов (для обратной совместимости):
    class_5 = np.zeros(n, dtype=np.int8)
    class_5[(dt_ticks_full > 0) & (z < TH_P1_P2)] = 1
    class_5[(dt_ticks_full > 0) & (z >= TH_P1_P2)] = 2
    class_5[(dt_ticks_full < 0) & (z > TH_M1_M2)] = -1
    class_5[(dt_ticks_full < 0) & (z <= TH_M1_M2)] = -2
    
    # Отрезаем краевые невалидные участки (первые L тиков и последние H тиков)
    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[L : n - H] = True
    
    # Добавляем размеченные колонки в результирующий датафрейм
    out_df = df.copy()
    out_df['target_class_7'] = class_7
    out_df['target_class_5'] = class_5
    out_df['target_velocity_z'] = z
    out_df['target_dt_ticks'] = dt_ticks_full
    out_df['target_ret_pct'] = ret_pct_full
    out_df['kinematic_v_wave'] = v_wave.astype(np.float32)
    out_df['kinematic_micro_offset'] = micro_offset.astype(np.float32)
    out_df['dynamic_scale_up'] = M_up.astype(np.float32)
    out_df['dynamic_scale_dn'] = M_dn.astype(np.float32)
    out_df['is_liquidity_void'] = is_void
    out_df['is_valid_sample'] = valid_mask
    
    # Срезаем только валидные строки
    valid_df = out_df.iloc[L : n - H].copy().reset_index(drop=True)
    
    # Расчет сходимости Train vs Val
    split_valid = split_idx - L
    tr_df = valid_df.iloc[:split_valid]
    va_df = valid_df.iloc[split_valid:]
    
    print("\n" + "=" * 110)
    print(f"📊 ПРОВЕРКА СХОДИМОСТИ МЕТОК (ГОРИЗОНТ H = {H}с, ОКНО L = {L}с):")
    print("=" * 110)
    print("Класс | Описание               | Train % (N)       | Val % (N)        | Train E[dt]| Val E[dt]  | Err(dt) | Train E[z] | Val E[z]  | Err(z) | Статус")
    print("-" * 125)
    
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
        m_tr = (tr_df['target_class_7'] == c)
        m_va = (va_df['target_class_7'] == c)
        c_tr, c_va = np.sum(m_tr), np.sum(m_va)
        
        dt_m_tr = tr_df.loc[m_tr, 'target_dt_ticks'].mean() if c_tr > 0 else 0
        dt_m_va = va_df.loc[m_va, 'target_dt_ticks'].mean() if c_va > 0 else 0
        err_dt = abs(abs(dt_m_va) - abs(dt_m_tr)) / abs(dt_m_tr) * 100 if dt_m_tr != 0 else 0
        
        z_m_tr = tr_df.loc[m_tr, 'target_velocity_z'].mean() if c_tr > 0 else 0
        z_m_va = va_df.loc[m_va, 'target_velocity_z'].mean() if c_va > 0 else 0
        err_z = abs(abs(z_m_va) - abs(z_m_tr)) / abs(z_m_tr) * 100 if z_m_tr != 0 else 0
        
        pct_tr = c_tr / len(tr_df) * 100
        pct_va = c_va / len(va_df) * 100
        
        flag = "🎯 < 5%" if err_z < 5.0 else ("✅ < 10%" if err_z < 10.0 else "⚠️ Хвост")
        desc = class_desc[c]
        print(f" {c:+2d}   | {desc} | {pct_tr:5.2f}% ({c_tr:6,d}) | {pct_va:5.2f}% ({c_va:5,d}) | {dt_m_tr:+6.2f} т  | {dt_m_va:+6.2f} т  | {err_dt:5.1f}%  | {z_m_tr:+8.4f}  | {z_m_va:+8.4f}  | {err_z:5.1f}% | {flag}")
    print("-" * 125)
    
    return tr_df, va_df, valid_df

def process_coin_labeling(coin, horizon, ratio, base_outdir):
    coin = coin.lower()
    in_file = f"data/features/{coin}_1s_features.parquet"
    if not os.path.exists(in_file):
        print(f"Ошибка: входной файл {in_file} не найден! Сначала соберите базу.")
        return False
        
    print(f"\n=======================================================")
    print(f"🏷️ РАЗМЕТКА ДАННЫХ ДЛЯ {coin.upper()} (H={horizon}s, L/H={ratio})")
    print(f"=======================================================")
    tbl = pq.read_table(in_file)
    df = tbl.to_pandas()
    
    tr_df, va_df, full_df = compute_labels(df, coin=coin, horizon_sec=horizon, window_ratio=ratio)
    
    coin_outdir = os.path.join(base_outdir, coin)
    os.makedirs(coin_outdir, exist_ok=True)
    
    train_path = os.path.join(coin_outdir, f"train_labeled_h{horizon}s.parquet")
    val_path = os.path.join(coin_outdir, f"val_labeled_h{horizon}s.parquet")
    full_path = os.path.join(coin_outdir, f"full_labeled_h{horizon}s.parquet")
    
    print(f"\n[4/4] Сохранение размеченных выборок {coin.upper()}:")
    print(f"      • Train: {train_path} ({len(tr_df):,} строк)")
    print(f"      • Val:   {val_path} ({len(va_df):,} строк)")
    print(f"      • Full:  {full_path} ({len(full_df):,} строк)")
    
    tr_df.to_parquet(train_path, index=False, compression="zstd")
    va_df.to_parquet(val_path, index=False, compression="zstd")
    full_df.to_parquet(full_path, index=False, compression="zstd")
    
    # Для PRL сохраняем также в корень data/labeled/ для обратной совместимости
    if coin == "prl":
        legacy_train = os.path.join(base_outdir, f"train_labeled_h{horizon}s.parquet")
        legacy_val = os.path.join(base_outdir, f"val_labeled_h{horizon}s.parquet")
        legacy_full = os.path.join(base_outdir, f"full_labeled_h{horizon}s.parquet")
        tr_df.to_parquet(legacy_train, index=False, compression="zstd")
        va_df.to_parquet(legacy_val, index=False, compression="zstd")
        full_df.to_parquet(legacy_full, index=False, compression="zstd")
    return True

def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Dataset Labeling Engine")
    parser.add_argument("--coin", type=str, default="all", choices=["prl", "quantus", "tsc", "all"], help="Монета для разметки (по умолчанию all)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in seconds (default: 5)")
    parser.add_argument("--ratio", type=int, default=4, help="Ratio of L/H (default: 4)")
    parser.add_argument("--outdir", type=str, default="data/labeled", help="Output directory")
    args = parser.parse_args()
    
    coins = ["prl", "quantus", "tsc"] if args.coin == "all" else [args.coin]
    for c in coins:
        process_coin_labeling(c, args.horizon, args.ratio, args.outdir)
        
    print("\n✅ Разметка данных успешно завершена для всех выбранных монет!")

if __name__ == '__main__':
    main()
