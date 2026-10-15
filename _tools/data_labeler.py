#!/usr/bin/env python3
"""
Production Data Labeling Engine for PRL/USDT and QUANTUS
========================================================
Поддерживает:
1. Двойное квантование цены и ликвидности.
2. Асимметричную эластичность стакана и кинематику волны.
3. Разметку 7 и 5 квантильных классов.
4. Защиту от утечки данных (Purging H шагов + Embargo gap).
5. Сетки параметров:
   - horizon = "5:Lmax:5", "[5,10,15]", "5"
   - ratio   = "[3,4,5,6]", "3:6", "4"
   - Динамический L_max = T_train // N (по умолчанию N = 5000).
6. Объединение в единые Parquet-датасеты:
   - data/labeled/{coin}_labeled.parquet (все таргеты и sample_split)
   - data/labeled/all_coins_labeled.parquet (объединенный датасет всех монет)
   - data/labeled/grid_validation_results.csv (сводка сходимости)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Микроструктурные параметры монет (SafeTrade)
COIN_CONFIG = {
    'prl': {'tick_size': 0.005, 'typical_trade': 17.3, 'min_vol': 50.0},
    'quantus': {'tick_size': 0.05, 'typical_trade': 2.2, 'min_vol': 2.0},
}

# Пороги квантильных плато
TH_P1_P2 = 0.50
TH_P2_P3 = 1.00
TH_M1_M2 = -0.50
TH_M2_M3 = -0.95

def parse_ratio_spec(ratio_input):
    """Парсит соотношение K = L/H (число, список или срез 3:6)."""
    if isinstance(ratio_input, (int, float)):
        return [int(ratio_input)]
    s = str(ratio_input).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    if ':' in s:
        parts = [int(p.strip()) for p in s.split(':')]
        if len(parts) == 2:
            return list(range(parts[0], parts[1] + 1))
        elif len(parts) == 3:
            return list(range(parts[0], parts[1] + 1, parts[2]))
    if ',' in s:
        return [int(p.strip()) for p in s.split(',') if p.strip()]
    return [int(s)]

def parse_horizon_spec(horizon_input, L_max, K):
    """
    Парсит горизонт H для заданного K.
    Поддерживает: 5, "5,10,15", "[5,10,15]", "5:30:5", "5:Lmax:5".
    """
    if isinstance(horizon_input, (int, float)):
        return [int(horizon_input)]
    s = str(horizon_input).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    if ':' in s:
        parts = [p.strip() for p in s.split(':')]
        start = int(parts[0])
        step = int(parts[2]) if len(parts) >= 3 else 5
        end_str = parts[1].lower()
        if end_str in ('lmax', 'hmax', 'auto', 'max'):
            H_max = max(1, L_max // K)
        else:
            H_max = int(end_str)
        h_vals = list(range(start, H_max + 1, step))
        if not h_vals:
            h_vals = [max(1, H_max)]
        return h_vals
    if ',' in s:
        return [int(p.strip()) for p in s.split(',') if p.strip()]
    if s.lower() in ('lmax', 'hmax', 'auto', 'max'):
        return [max(1, L_max // K)]
    return [int(s)]

def compute_single_horizon_labels(df, coin, H, K, min_vol, tick_size):
    """Рассчитывает микроструктурные метрики и таргеты для пары (H, K)."""
    n = len(df)
    L = int(H * K)
    mid = df['ST_mid_price'].values
    spread_ticks = df['ST_spread'].values / tick_size
    b1_p = df['ST_bid_1_price'].values
    a1_p = df['ST_ask_1_price'].values
    b1_q = df['ST_bid_1_qty'].values
    a1_q = df['ST_ask_1_qty'].values
    b5 = df['ST_bids_vol_top5'].values
    a5 = df['ST_asks_vol_top5'].values
    trade_vol = df['ST_trade_vol'].values
    
    # 1. Сдвиг цены в тиках и процентах
    dt_ticks_full = np.zeros(n, dtype=int)
    dt_ticks_full[:-H] = np.round((mid[H:] - mid[:-H]) / tick_size).astype(int)
    ret_pct_full = np.zeros(n, dtype=np.float32)
    ret_pct_full[:-H] = ((mid[H:] - mid[:-H]) / np.maximum(mid[:-H], 0.001) * 100.0).astype(np.float32)
    
    # 2. Локальная волатильность за окно L (в тиках)
    step_1s_ticks = np.zeros(n)
    step_1s_ticks[1:] = (mid[1:] - mid[:-1]) / tick_size
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    vol_L = np.nan_to_num(vol_L)
    
    # 3. Скользящие объемы стакана
    asks_L = pd.Series(a5).rolling(L, min_periods=L).mean().values
    bids_L = pd.Series(b5).rolling(L, min_periods=L).mean().values
    a1_L = pd.Series(a1_q).rolling(L, min_periods=L).median().values
    b1_L = pd.Series(b1_q).rolling(L, min_periods=L).median().values
    
    # 4. Кинематика микроцены
    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 0, (b1_p * a1_q + a1_p * b1_q) / np.maximum(total_q1, 1.0), mid)
    micro_offset = (microprice - mid) / tick_size
    
    t_wave = max(3, H // 3)
    v_wave = (pd.Series(micro_offset) - pd.Series(micro_offset).shift(t_wave)).fillna(0).values / t_wave
    d_vol = (pd.Series(vol_L) - pd.Series(vol_L).shift(t_wave)).fillna(0).values / t_wave
    
    # 5. Факторы лотов 1-го уровня
    q1_factor_up = np.clip(a1_L / np.maximum(a1_q, 1.0), 0.2, 5.0) ** 0.15
    q1_factor_dn = np.clip(b1_L / np.maximum(b1_q, 1.0), 0.2, 5.0) ** 0.15
    
    # 6. Динамическое сопротивление M
    M_up_base = (spread_ticks + vol_L * np.sqrt(H)) * (np.clip(asks_L / np.maximum(a5, min_vol), 0.2, 5.0) ** 0.25) * q1_factor_up
    M_dn_base = (spread_ticks + vol_L * np.sqrt(H)) * (np.clip(bids_L / np.maximum(b5, min_vol), 0.2, 5.0) ** 0.25) * q1_factor_dn
    
    M_up = np.maximum(0.5, M_up_base * (1.0 + np.maximum(0, v_wave)) * (1.0 + np.maximum(0, d_vol)))
    M_dn = np.maximum(0.5, M_dn_base * (1.0 + np.maximum(0, -v_wave)) * (1.0 + np.maximum(0, d_vol)))
    
    # 7. Нормированный импульс z
    z = np.zeros(n, dtype=np.float32)
    z[dt_ticks_full > 0] = (dt_ticks_full[dt_ticks_full > 0] / M_up[dt_ticks_full > 0]).astype(np.float32)
    z[dt_ticks_full < 0] = (dt_ticks_full[dt_ticks_full < 0] / M_dn[dt_ticks_full < 0]).astype(np.float32)
    
    # 8. Флаг фантомных котировок (Zero-Volume Liquidity Void):
    # Сумма сделок вперед за горизонт (t, t+H]
    trade_cumsum = np.pad(np.cumsum(trade_vol), (1, 0))
    trade_vol_future_H = np.zeros(n, dtype=np.float32)
    trade_vol_future_H[:-H] = trade_cumsum[1 + H : n + 1] - trade_cumsum[1 : n - H + 1]
    is_void = (np.abs(dt_ticks_full) >= 5) & (trade_vol_future_H == 0)
    
    # Разметка 7 классов
    class_7 = np.zeros(n, dtype=np.int8)
    class_7[(dt_ticks_full > 0) & (z < TH_P1_P2)] = 1
    class_7[(dt_ticks_full > 0) & (z >= TH_P1_P2) & (z < TH_P2_P3)] = 2
    class_7[(dt_ticks_full > 0) & (z >= TH_P2_P3)] = 3
    class_7[(dt_ticks_full < 0) & (z > TH_M1_M2)] = -1
    class_7[(dt_ticks_full < 0) & (z <= TH_M1_M2) & (z > TH_M2_M3)] = -2
    class_7[(dt_ticks_full < 0) & (z <= TH_M2_M3)] = -3
    
    # Разметка 5 классов
    class_5 = np.zeros(n, dtype=np.int8)
    class_5[(dt_ticks_full > 0) & (z < TH_P1_P2)] = 1
    class_5[(dt_ticks_full > 0) & (z >= TH_P1_P2)] = 2
    class_5[(dt_ticks_full < 0) & (z > TH_M1_M2)] = -1
    class_5[(dt_ticks_full < 0) & (z <= TH_M1_M2)] = -2
    
    return {
        'class_7': class_7,
        'class_5': class_5,
        'z': z,
        'dt_ticks': dt_ticks_full,
        'ret_pct': ret_pct_full,
        'v_wave': v_wave.astype(np.float32),
        'micro_offset': micro_offset.astype(np.float32),
        'M_up': M_up.astype(np.float32),
        'M_dn': M_dn.astype(np.float32),
        'future_trade_vol_H': trade_vol_future_H,
        'is_void': is_void,
    }

def compute_labels(df, coin="prl", horizon_sec=5, window_ratio=4, split_ratio=0.8, purge_sec=None, embargo_sec=0):
    """
    Сохраняет совместимость с одиночными вызовами (run_grid_validation.py).
    Возвращает (tr_df, va_df, valid_df).
    """
    coin = coin.lower()
    cfg = COIN_CONFIG.get(coin, COIN_CONFIG['prl'])
    tick_size = cfg['tick_size']
    min_vol = cfg['min_vol']

    n = len(df)
    H = int(horizon_sec)
    L = int(H * window_ratio)
    
    purge_steps = int(purge_sec if purge_sec is not None else H)
    embargo_steps = int(embargo_sec)
    gap_steps = purge_steps + embargo_steps
    
    split_idx = int(n * split_ratio)
    train_end = split_idx - gap_steps
    val_start = split_idx
    
    res = compute_single_horizon_labels(df, coin, H, window_ratio, min_vol, tick_size)
    
    sample_split = np.full(n, 'unknown', dtype=object)
    sample_split[:L] = 'warmup'
    sample_split[L:train_end] = 'train'
    sample_split[train_end:val_start] = 'purged' if embargo_steps == 0 else 'purged_embargo'
    sample_split[val_start:n - H] = 'val'
    sample_split[n - H:] = 'tail'
    
    out_df = df.copy()
    out_df['target_class_7'] = res['class_7']
    out_df['target_class_5'] = res['class_5']
    out_df['target_velocity_z'] = res['z']
    out_df['target_dt_ticks'] = res['dt_ticks']
    out_df['target_ret_pct'] = res['ret_pct']
    out_df['kinematic_v_wave'] = res['v_wave']
    out_df['kinematic_micro_offset'] = res['micro_offset']
    out_df['dynamic_scale_up'] = res['M_up']
    out_df['dynamic_scale_dn'] = res['M_dn']
    out_df['future_trade_vol_H'] = res['future_trade_vol_H']
    out_df['is_liquidity_void'] = res['is_void']
    out_df['sample_split'] = sample_split
    out_df['is_valid_sample'] = (sample_split == 'train') | (sample_split == 'val')
    
    if isinstance(out_df.index, pd.DatetimeIndex) or out_df.index.name == 'timestamp':
        out_df = out_df.reset_index()

    tr_df = out_df.iloc[L:train_end].copy().reset_index(drop=True)
    va_df = out_df.iloc[val_start:n - H].copy().reset_index(drop=True)
    valid_df = pd.concat([tr_df, va_df], ignore_index=True)
    return tr_df, va_df, valid_df

def process_coin_labeling(
    coin,
    horizon_spec="5",
    ratio_spec="4",
    base_outdir="data/labeled",
    split_ratio=0.8,
    n_target=5000,
    purge_sec=None,
    embargo_sec=0,
    save_legacy_splits=True
):
    """
    Размечает датасет для монеты (одиночный или по сетке H и K).
    Сохраняет объединенный датасет в один Parquet-файл.
    """
    coin = coin.lower()
    in_file = f"data/features/{coin}_1s_features.parquet"
    if not os.path.exists(in_file):
        print(f"Ошибка: входной файл {in_file} не найден!")
        return None, []
        
    cfg = COIN_CONFIG.get(coin, COIN_CONFIG['prl'])
    tick_size = cfg['tick_size']
    min_vol = cfg['min_vol']
    
    tbl = pq.read_table(in_file)
    df = tbl.to_pandas()
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name == 'timestamp':
        df = df.reset_index()
    total_rows = len(df)
    train_80 = int(total_rows * split_ratio)
    L_max = max(10, train_80 // n_target)
    
    ratios = parse_ratio_spec(ratio_spec)
    pairs = []
    for K in ratios:
        h_vals = parse_horizon_spec(horizon_spec, L_max, K)
        for H in h_vals:
            pairs.append((H, K))
            
    if not pairs:
        pairs = [(5, 4)]
        
    max_H = max(p[0] for p in pairs)
    max_L = max(p[0] * p[1] for p in pairs)
    purge_steps = int(purge_sec if purge_sec is not None else max_H)
    embargo_steps = int(embargo_sec)
    gap_steps = purge_steps + embargo_steps
    
    split_idx = int(total_rows * split_ratio)
    train_end = split_idx - gap_steps
    val_start = split_idx
    
    print("\n" + "=" * 80)
    print(f"🏷️ РАЗМЕТКА {coin.upper()}: Строк={total_rows:,} | Train(80%)={train_80:,} | L_max(N={n_target})={L_max}с")
    print(f"   Сетка пар (H, K): {len(pairs)} комбинаций")
    print(f"   Защита от утечки: Purge={purge_steps}с, Embargo={embargo_steps}с (Буфер={gap_steps}с)")
    print("=" * 80)
    
    sample_split = np.full(total_rows, 'unknown', dtype=object)
    sample_split[:max_L] = 'warmup'
    sample_split[max_L:train_end] = 'train'
    sample_split[train_end:val_start] = 'purged' if embargo_steps == 0 else 'purged_embargo'
    sample_split[val_start:total_rows - max_H] = 'val'
    sample_split[total_rows - max_H:] = 'tail'
    
    out_df = df.copy()
    out_df['coin'] = coin.upper()
    out_df['sample_split'] = sample_split
    out_df['is_valid_sample'] = (sample_split == 'train') | (sample_split == 'val')
    
    primary_pair = (5, 4) if (5, 4) in pairs else pairs[0]
    grid_records = []
    
    tr_mask = (sample_split == 'train')
    va_mask = (sample_split == 'val')
    extra_cols = {}
    
    for H, K in pairs:
        L = H * K
        res = compute_single_horizon_labels(df, coin, H, K, min_vol, tick_size)
        
        # Добавляем таргеты с суффиксами в общий датасет
        suffix = f"_h{H}_k{K}"
        extra_cols[f'target_class_7{suffix}'] = res['class_7']
        extra_cols[f'target_class_5{suffix}'] = res['class_5']
        extra_cols[f'target_velocity_z{suffix}'] = res['z']
        extra_cols[f'target_dt_ticks{suffix}'] = res['dt_ticks']
        extra_cols[f'target_ret_pct{suffix}'] = res['ret_pct']
        extra_cols[f'kinematic_v_wave{suffix}'] = res['v_wave']
        extra_cols[f'dynamic_scale_up{suffix}'] = res['M_up']
        extra_cols[f'dynamic_scale_dn{suffix}'] = res['M_dn']
        extra_cols[f'is_liquidity_void_h{H}'] = res['is_void']
        
        # Для базовой пары устанавливаем стандартные имена без суффиксов
        if (H, K) == primary_pair:
            extra_cols['target_class_7'] = res['class_7']
            extra_cols['target_class_5'] = res['class_5']
            extra_cols['target_velocity_z'] = res['z']
            extra_cols['target_dt_ticks'] = res['dt_ticks']
            extra_cols['target_ret_pct'] = res['ret_pct']
            extra_cols['kinematic_v_wave'] = res['v_wave']
            extra_cols['kinematic_micro_offset'] = res['micro_offset']
            extra_cols['dynamic_scale_up'] = res['M_up']
            extra_cols['dynamic_scale_dn'] = res['M_dn']
            extra_cols['future_trade_vol_H'] = res['future_trade_vol_H']
            extra_cols['is_liquidity_void'] = res['is_void']
            
        # Метрики сходимости
        z_col = res['z']
        dt_col = res['dt_ticks']
        c7_col = res['class_7']
        
        m_tr_p2 = tr_mask & (c7_col == 2)
        m_va_p2 = va_mask & (c7_col == 2)
        z_tr_p2 = z_col[m_tr_p2].mean() if np.any(m_tr_p2) else 0.0
        z_va_p2 = z_col[m_va_p2].mean() if np.any(m_va_p2) else 0.0
        err_z_p2 = abs(abs(z_va_p2) - abs(z_tr_p2)) / abs(z_tr_p2) * 100.0 if z_tr_p2 != 0 else 0.0
        
        m_tr_m2 = tr_mask & (c7_col == -2)
        m_va_m2 = va_mask & (c7_col == -2)
        z_tr_m2 = z_col[m_tr_m2].mean() if np.any(m_tr_m2) else 0.0
        z_va_m2 = z_col[m_va_m2].mean() if np.any(m_va_m2) else 0.0
        err_z_m2 = abs(abs(z_va_m2) - abs(z_tr_m2)) / abs(z_tr_m2) * 100.0 if z_tr_m2 != 0 else 0.0
        
        grid_records.append({
            'coin': coin.upper(),
            'K': K,
            'H': H,
            'L': L,
            'L_max': L_max,
            'z_tr_p2': z_tr_p2,
            'z_va_p2': z_va_p2,
            'err_z_p2': err_z_p2,
            'z_tr_m2': z_tr_m2,
            'z_va_m2': z_va_m2,
            'err_z_m2': err_z_m2,
            'pct_flat': (c7_col[tr_mask] == 0).mean() * 100.0
        })

    out_df = pd.concat([out_df, pd.DataFrame(extra_cols, index=out_df.index)], axis=1)

    # Вывод результатов
    print(f"\n📊 СХОДИМОСТЬ СЕТКИ ДЛЯ {coin.upper()}:")
    print(f"{'K':<3} | {'H':<4} | {'L':<4} | {'E[z] Class +2':<15} | {'Err(+2)':<8} | {'E[z] Class -2':<15} | {'Err(-2)':<8} | {'Статус'}")
    print("-" * 85)
    for r in grid_records:
        max_err = max(r['err_z_p2'], r['err_z_m2'])
        st = "🎯 < 5%" if max_err < 5.0 else ("✅ < 10%" if max_err < 10.0 else "⚠️ Хвост")
        print(f"{r['K']:<3} | {r['H']:<4} | {r['L']:<4} | {r['z_tr_p2']:+.4f} -> {r['z_va_p2']:+.4f} | {r['err_z_p2']:5.1f}%  | {r['z_tr_m2']:+.4f} -> {r['z_va_m2']:+.4f} | {r['err_z_m2']:5.1f}%  | {st}")
    print("-" * 85)

    # Сохраняем единый датасет для монеты
    os.makedirs(base_outdir, exist_ok=True)
    coin_outdir = os.path.join(base_outdir, coin)
    os.makedirs(coin_outdir, exist_ok=True)
    
    unified_coin_path = os.path.join(coin_outdir, f"{coin}_labeled.parquet")
    out_df.to_parquet(unified_coin_path, index=False, compression="zstd")
    print(f"💾 Единый размеченный датасет {coin.upper()}: {unified_coin_path} ({len(out_df):,} строк, {len(out_df.columns)} колонок)")
    
    # Обратная совместимость для сплитов базового горизонта
    if save_legacy_splits:
        pH, pK = primary_pair
        tr_df = out_df[out_df['sample_split'] == 'train'].reset_index(drop=True)
        va_df = out_df[out_df['sample_split'] == 'val'].reset_index(drop=True)
        
        tr_path = os.path.join(coin_outdir, f"train_labeled_h{pH}s.parquet")
        va_path = os.path.join(coin_outdir, f"val_labeled_h{pH}s.parquet")
        full_path = os.path.join(coin_outdir, f"full_labeled_h{pH}s.parquet")
        
        tr_df.to_parquet(tr_path, index=False, compression="zstd")
        va_df.to_parquet(va_path, index=False, compression="zstd")
        out_df[out_df['is_valid_sample']].to_parquet(full_path, index=False, compression="zstd")
        
        if coin == "prl":
            legacy_tr = os.path.join(base_outdir, f"train_labeled_h{pH}s.parquet")
            legacy_va = os.path.join(base_outdir, f"val_labeled_h{pH}s.parquet")
            legacy_full = os.path.join(base_outdir, f"full_labeled_h{pH}s.parquet")
            tr_df.to_parquet(legacy_tr, index=False, compression="zstd")
            va_df.to_parquet(legacy_va, index=False, compression="zstd")
            out_df[out_df['is_valid_sample']].to_parquet(legacy_full, index=False, compression="zstd")
            
    return out_df, grid_records

def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Dynamic Dataset Labeling Engine")
    parser.add_argument("--coin", type=str, default="all", choices=["prl", "quantus", "all"], help="Монета для разметки (по умолчанию all)")
    parser.add_argument("--horizon", type=str, default="5", help="Горизонт H или интервал (например: 5, '5,10,15', '5:Lmax:5', '5:30:5')")
    parser.add_argument("--ratio", type=str, default="4", help="Соотношение K=L/H или интервал (например: 4, '[3,4,5,6]', '3,4,5', '3:6')")
    parser.add_argument("--outdir", type=str, default="data/labeled", help="Выходная директория")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Доля обучающей выборки Train (по умолчанию 0.8)")
    parser.add_argument("--n-target", type=int, default=5000, help="Минимум независимых отсчетов N для расчета L_max = T_train // N (по умолчанию 5000)")
    parser.add_argument("--purge-sec", type=int, default=None, help="Зазор Purging в секундах (по умолчанию равен max H)")
    parser.add_argument("--embargo-sec", type=int, default=0, help="Зазор Embargo в секундах (по умолчанию 0)")
    parser.add_argument("--no-legacy-splits", action="store_true", help="Не сохранять отдельные train/val файлы")
    args = parser.parse_args()
    
    coins = ["prl", "quantus"] if args.coin == "all" else [args.coin]
    all_coin_dfs = []
    all_records = []
    for c in coins:
        cdf, recs = process_coin_labeling(
            coin=c,
            horizon_spec=args.horizon,
            ratio_spec=args.ratio,
            base_outdir=args.outdir,
            split_ratio=args.split_ratio,
            n_target=args.n_target,
            purge_sec=args.purge_sec,
            embargo_sec=args.embargo_sec,
            save_legacy_splits=not args.no_legacy_splits
        )
        if cdf is not None:
            all_coin_dfs.append(cdf)
            all_records.extend(recs)
            
    # Объединение всех монет в единый мастер-файл
    if args.coin == "all" and all_coin_dfs:
        merged_master_path = os.path.join(args.outdir, "all_coins_labeled.parquet")
        master_df = pd.concat(all_coin_dfs, ignore_index=True)
        master_df.to_parquet(merged_master_path, index=False, compression="zstd")
        print("\n" + "=" * 80)
        print(f"🏆 ВСЕ МОНЕТЫ ОБЪЕДИНЕНЫ В ЕДИНЫЙ ФАЙЛ: {merged_master_path}")
        print(f"   Всего строк: {len(master_df):,} | Колонок: {len(master_df.columns)}")
        print("=" * 80)
        
    # Сохранение сводной CSV-таблицы валидации сетки
    if all_records:
        csv_path = os.path.join(args.outdir, "grid_validation_results.csv")
        pd.DataFrame(all_records).to_csv(csv_path, index=False)
        print(f"📈 Сводная таблица сходимости сетки сохранена: {csv_path}")

    print("\n✅ Разметка данных успешно завершена!")

if __name__ == '__main__':
    main()
