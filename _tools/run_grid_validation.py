#!/usr/bin/env python3
"""
Скрипт валидации сетки параметров (H, K = L/H) для всех монет (PRL, QUAN) при N = 4000.
Рассчитывает L_max для каждой монеты:
L_max = train_dataset (80%) // 4000.
Тестирует H = 5:Max_L/K:5 для K in [3, 4, 5] и оценивает сходимость меток Train vs Val.
"""
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Добавляем пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.data_labeler import COIN_CONFIG, compute_labels

def evaluate_run(df, coin, H, K):
    """Выполняет разметку и возвращает метрики сходимости."""
    tr_df, va_df, _ = compute_labels(df, coin=coin, horizon_sec=H, window_ratio=K, split_ratio=0.8)
    
    # Метрики для класса +2
    m_tr_p2 = (tr_df['target_class_7'] == 2)
    m_va_p2 = (va_df['target_class_7'] == 2)
    c_tr_p2, c_va_p2 = m_tr_p2.sum(), m_va_p2.sum()
    
    z_tr_p2 = tr_df.loc[m_tr_p2, 'target_velocity_z'].mean() if c_tr_p2 > 0 else 0.0
    z_va_p2 = va_df.loc[m_va_p2, 'target_velocity_z'].mean() if c_va_p2 > 0 else 0.0
    err_z_p2 = abs(abs(z_va_p2) - abs(z_tr_p2)) / abs(z_tr_p2) * 100.0 if z_tr_p2 != 0 else 0.0
    
    dt_tr_p2 = tr_df.loc[m_tr_p2, 'target_dt_ticks'].mean() if c_tr_p2 > 0 else 0.0
    dt_va_p2 = va_df.loc[m_va_p2, 'target_dt_ticks'].mean() if c_va_p2 > 0 else 0.0
    err_dt_p2 = abs(abs(dt_va_p2) - abs(dt_tr_p2)) / abs(dt_tr_p2) * 100.0 if dt_tr_p2 != 0 else 0.0

    # Метрики для класса -2
    m_tr_m2 = (tr_df['target_class_7'] == -2)
    m_va_m2 = (va_df['target_class_7'] == -2)
    c_tr_m2, c_va_m2 = m_tr_m2.sum(), m_va_m2.sum()
    
    z_tr_m2 = tr_df.loc[m_tr_m2, 'target_velocity_z'].mean() if c_tr_m2 > 0 else 0.0
    z_va_m2 = va_df.loc[m_va_m2, 'target_velocity_z'].mean() if c_va_m2 > 0 else 0.0
    err_z_m2 = abs(abs(z_va_m2) - abs(z_tr_m2)) / abs(z_tr_m2) * 100.0 if z_tr_m2 != 0 else 0.0

    dt_tr_m2 = tr_df.loc[m_tr_m2, 'target_dt_ticks'].mean() if c_tr_m2 > 0 else 0.0
    dt_va_m2 = va_df.loc[m_va_m2, 'target_dt_ticks'].mean() if c_va_m2 > 0 else 0.0
    err_dt_m2 = abs(abs(dt_va_m2) - abs(dt_tr_m2)) / abs(dt_tr_m2) * 100.0 if dt_tr_m2 != 0 else 0.0

    # Доля класса 0 (флэт)
    pct_flat_tr = (tr_df['target_class_7'] == 0).mean() * 100.0
    pct_flat_va = (va_df['target_class_7'] == 0).mean() * 100.0

    return {
        'coin': coin.upper(),
        'K': K,
        'H': H,
        'L': K * H,
        'N_train': len(tr_df),
        'N_val': len(va_df),
        'pct_flat_tr': pct_flat_tr,
        'pct_p2_tr': c_tr_p2 / len(tr_df) * 100.0,
        'pct_m2_tr': c_tr_m2 / len(tr_df) * 100.0,
        'z_tr_p2': z_tr_p2,
        'z_va_p2': z_va_p2,
        'err_z_p2': err_z_p2,
        'z_tr_m2': z_tr_m2,
        'z_va_m2': z_va_m2,
        'err_z_m2': err_z_m2,
        'dt_tr_p2': dt_tr_p2,
        'dt_va_p2': dt_va_p2,
        'err_dt_p2': err_dt_p2,
        'dt_tr_m2': dt_tr_m2,
        'dt_va_m2': dt_va_m2,
        'err_dt_m2': err_dt_m2,
    }

def main():
    N_TARGET = 4000
    coins = ['prl', 'quantus']
    results = []
    
    print("================================================================================")
    print(f"🚀 ВАЛИДАЦИЯ ДИНАМИЧЕСКОЙ РАЗМЕТКИ ДЛЯ ВСЕХ ПАР (N = {N_TARGET})")
    print("================================================================================\n")
    
    coin_lmax = {}
    for coin in coins:
        fpath = f"data/features/{coin}_1s_features.parquet"
        if not os.path.exists(fpath):
            print(f"Файл {fpath} не найден!")
            continue
        tbl = pq.read_table(fpath)
        total_rows = len(tbl)
        train_80 = int(total_rows * 0.8)
        L_max = train_80 // N_TARGET
        coin_lmax[coin] = (total_rows, train_80, L_max)
        print(f"📊 {coin.upper()}: Всего = {total_rows:,} с | Train (80%) = {train_80:,} с | L_max (N={N_TARGET}) = {L_max} с")

    print("\n" + "="*80)
    print("ЗАПУСК ПЕРЕБОРА СЕТКИ (H = 5:Max_L/K:5 для K = 3, 4, 5, 6)")
    print("="*80)

    for coin in coins:
        if coin not in coin_lmax:
            continue
        fpath = f"data/features/{coin}_1s_features.parquet"
        tbl = pq.read_table(fpath)
        df = tbl.to_pandas()
        total_rows, train_80, L_max = coin_lmax[coin]
        
        for K in [3, 4, 5, 6]:
            H_max = L_max // K
            # Сетка горизонтов: от 5 с шагом 5 до H_max
            H_values = list(range(5, H_max + 1, 5))
            if not H_values and H_max >= 5:
                H_values = [5]
            elif not H_values and H_max < 5:
                H_values = [max(1, H_max)]
                
            for H in H_values:
                print(f"\nТест: Монета={coin.upper()}, K={K}, H={H}с (L={K*H}с, L_max={L_max}с)...")
                res = evaluate_run(df, coin, H, K)
                res['L_max'] = L_max
                results.append(res)

    res_df = pd.DataFrame(results)
    csv_out = "data/labeled/grid_validation_results_n4000.csv"
    res_df.to_csv(csv_out, index=False)
    print(f"\n✅ Результаты сохранены в: {csv_out}")

    print("\n" + "="*120)
    print("ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА СХОДИМОСТИ МЕТОДА")
    print("="*120)
    print(f"{'Монета':<8} | {'K (L/H)':<7} | {'H (с)':<5} | {'L (с)':<5} | {'L_max':<5} | {'E[z] Class +2':<15} | {'Err_z(+2)':<9} | {'E[z] Class -2':<15} | {'Err_z(-2)':<9} | {'Статус надежности':<15}")
    print("-" * 120)
    for _, r in res_df.iterrows():
        err_p2 = r['err_z_p2']
        err_m2 = r['err_z_m2']
        status = "🎯 Отлично (<5%)" if max(err_p2, err_m2) < 5.0 else ("✅ Хорошо (<10%)" if max(err_p2, err_m2) < 10.0 else "⚠️ Умеренно")
        print(f"{r['coin']:<8} | {r['K']:<7} | {r['H']:<5} | {r['L']:<5} | {r['L_max']:<5} | {r['z_tr_p2']:+.4f} -> {r['z_va_p2']:+.4f} | {err_p2:6.1f}%   | {r['z_tr_m2']:+.4f} -> {r['z_va_m2']:+.4f} | {err_m2:6.1f}%   | {status:<15}")
    print("-" * 120)

if __name__ == '__main__':
    main()
