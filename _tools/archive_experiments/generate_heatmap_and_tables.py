import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap_and_tables():
    path = 'data/features/prl_1s_features.parquet'
    cols = [
        'ST_bid_1_price', 'ST_ask_1_price', 'ST_bid_1_qty', 'ST_ask_1_qty',
        'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
        'ST_slippage_buy_100_pct', 'ST_slippage_sell_100_pct', 'ST_delta_vol'
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
    b1_p = df['ST_bid_1_price'].values
    a1_p = df['ST_ask_1_price'].values
    b1_q = df['ST_bid_1_qty'].values
    a1_q = df['ST_ask_1_qty'].values
    b5 = df['ST_bids_vol_top5'].values
    a5 = df['ST_asks_vol_top5'].values
    
    # Микроцена и волновой накат
    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 0, (b1_p * a1_q + a1_p * b1_q) / np.maximum(total_q1, 1.0), mid)
    micro_offset = (microprice - mid) / TICK_SIZE
    v_wave = (pd.Series(micro_offset) - pd.Series(micro_offset).shift(3)).fillna(0).values
    
    step_1s = (mid[1:] - mid[:-1]) / TICK_SIZE
    vol_L = pd.Series(step_1s).rolling(L, min_periods=L).std().values
    d_vol = (pd.Series(vol_L) - pd.Series(vol_L).shift(3)).fillna(0).values
    
    asks_L = pd.Series(a5).rolling(L, min_periods=L).mean().values
    bids_L = pd.Series(b5).rolling(L, min_periods=L).mean().values
    
    # Разные горизонты для анализа
    horizons = [1, 2, 3, 5, 10]
    dt_dict = {}
    for h in horizons:
        d = np.zeros(n)
        d[:-h] = np.round((mid[h:] - mid[:-h]) / TICK_SIZE)
        dt_dict[h] = d
        
    valid_start = L
    valid_end = n - 10
    
    train_n = split_idx - valid_start
    
    # Расчет нормированного z для H = 5s
    vt = np.nan_to_num(vol_L[valid_start:valid_end])
    sp = spread[valid_start:valid_end]
    a_cur = a5[valid_start:valid_end]
    b_cur = b5[valid_start:valid_end]
    a_m = asks_L[valid_start:valid_end]
    b_m = bids_L[valid_start:valid_end]
    vw = v_wave[valid_start:valid_end]
    dv = d_vol[valid_start:valid_end]
    
    # Базовые масштабы M
    M_up_base = (sp + vt * np.sqrt(H)) * (np.clip(a_m / np.maximum(a_cur, 50.0), 0.2, 5.0) ** 0.3)
    M_dn_base = (sp + vt * np.sqrt(H)) * (np.clip(b_m / np.maximum(b_cur, 50.0), 0.2, 5.0) ** 0.3)
    
    # Волновые масштабы M с учетом кинетики
    M_up_wave = M_up_base * (1.0 + np.maximum(0, vw)) * (1.0 + np.maximum(0, dv))
    M_dn_wave = M_dn_base * (1.0 + np.maximum(0, -vw)) * (1.0 + np.maximum(0, dv))
    
    dt_5 = dt_dict[5][valid_start:valid_end]
    
    # 1. Построим сетку перцентилей от 1% до 99%
    percentiles = np.arange(1, 100, 1) # 1, 2, ..., 99
    
    # Режимы для строк тепловой карты:
    regimes = [
        "Рост (базовый масштаб)",
        "Рост (с волновым накатом)",
        "Спад (базовый масштаб)",
        "Спад (с волновым накатом)",
        "Рост: тонкий стакан",
        "Рост: плотный стакан",
        "Горизонт H=1 сек",
        "Горизонт H=3 сек",
        "Горизонт H=10 сек",
    ]
    
    heatmap_matrix = np.zeros((len(regimes), len(percentiles)))
    
    # Расчет для каждого режима:
    # Режим 0: Рост базовый
    tr_pos = dt_5[:train_n] > 0
    va_pos = dt_5[train_n:] > 0
    z_up_tr_base = dt_5[:train_n][tr_pos] / M_up_base[:train_n][tr_pos]
    z_up_va_base = dt_5[train_n:][va_pos] / M_up_base[train_n:][va_pos]
    
    # Режим 1: Рост с волновым накатом
    z_up_tr_wave = dt_5[:train_n][tr_pos] / M_up_wave[:train_n][tr_pos]
    z_up_va_wave = dt_5[train_n:][va_pos] / M_up_wave[train_n:][va_pos]
    
    # Режим 2: Спад базовый
    tr_neg = dt_5[:train_n] < 0
    va_neg = dt_5[train_n:] < 0
    z_dn_tr_base = np.abs(dt_5[:train_n][tr_neg]) / M_dn_base[:train_n][tr_neg]
    z_dn_va_base = np.abs(dt_5[train_n:][va_neg]) / M_dn_base[train_n:][va_neg]
    
    # Режим 3: Спад с волновым накатом
    z_dn_tr_wave = np.abs(dt_5[:train_n][tr_neg]) / M_dn_wave[:train_n][tr_neg]
    z_dn_va_wave = np.abs(dt_5[train_n:][va_neg]) / M_dn_wave[train_n:][va_neg]
    
    # Режим 4: Рост тонкий стакан (asks < median)
    a_med = np.median(a_cur)
    tr_thin = tr_pos & (a_cur[:train_n] < a_med)
    va_thin = va_pos & (a_cur[train_n:] < a_med)
    z_thin_tr = dt_5[:train_n][tr_thin] / M_up_wave[:train_n][tr_thin]
    z_thin_va = dt_5[train_n:][va_thin] / M_up_wave[train_n:][va_thin]
    
    # Режим 5: Рост плотный стакан (asks >= median)
    tr_thick = tr_pos & (a_cur[:train_n] >= a_med)
    va_thick = va_pos & (a_cur[train_n:] >= a_med)
    z_thick_tr = dt_5[:train_n][tr_thick] / M_up_wave[:train_n][tr_thick]
    z_thick_va = dt_5[train_n:][va_thick] / M_up_wave[train_n:][va_thick]
    
    # Режимы 6, 7, 8: Горизонты 1s, 3s, 10s
    def get_h_z(h):
        dt_h = dt_dict[h][valid_start:valid_end]
        tr_p = dt_h[:train_n] > 0
        va_p = dt_h[train_n:] > 0
        M_tr = (sp[:train_n] + vt[:train_n] * np.sqrt(h))
        M_va = (sp[train_n:] + vt[train_n:] * np.sqrt(h))
        return dt_h[:train_n][tr_p] / M_tr[tr_p], dt_h[train_n:][va_p] / M_va[va_p]
        
    z_h1_tr, z_h1_va = get_h_z(1)
    z_h3_tr, z_h3_va = get_h_z(3)
    z_h10_tr, z_h10_va = get_h_z(10)
    
    series_pairs = [
        (z_up_tr_base, z_up_va_base),
        (z_up_tr_wave, z_up_va_wave),
        (z_dn_tr_base, z_dn_va_base),
        (z_dn_tr_wave, z_dn_va_wave),
        (z_thin_tr, z_thin_va),
        (z_thick_tr, z_thick_va),
        (z_h1_tr, z_h1_va),
        (z_h3_tr, z_h3_va),
        (z_h10_tr, z_h10_va),
    ]
    
    for row_idx, (tr_arr, va_arr) in enumerate(series_pairs):
        for col_idx, p in enumerate(percentiles):
            q_tr = np.percentile(tr_arr, p)
            q_va = np.percentile(va_arr, p)
            err = abs(q_va - q_tr) / q_tr * 100 if q_tr > 0 else 0
            heatmap_matrix[row_idx, col_idx] = err
            
    # Визуализация красивого графика и тепловой карты
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)
    
    # Панель 1: Тепловая карта ошибки
    ax1 = fig.add_subplot(gs[0])
    # Клипуем ошибку до 25% для четкости отображения градиента
    matrix_capped = np.clip(heatmap_matrix, 0, 25.0)
    
    sns.heatmap(
        matrix_capped,
        cmap='YlGnBu_r', # Зеленый/синий — низкая ошибка, желтый/красный — высокая
        vmin=0.0,
        vmax=15.0,
        cbar_kws={'label': 'Ошибка расхождения Train vs Val (%)'},
        yticklabels=regimes,
        xticklabels=10, # каждые 10 перцентилей
        ax=ax1
    )
    ax1.set_title('ТЕПЛОВАЯ КАРТА ОШИБКИ ДИНАМИЧЕСКОГО МАСШТАБИРОВАНИЯ (0% - 100% КВАНТИЛИ)', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel('Квантиль распределения движения цены (Percentile, %)', fontsize=12)
    ax1.set_ylabel('Рыночный режим / Фактор', fontsize=12)
    
    # Добавим вертикальную линию на 98%
    ax1.axvline(x=97, color='red', linestyle='--', linewidth=2, label='98% (Граница редких экстремумов)')
    ax1.legend(loc='upper left')
    
    # Панель 2: Профиль ошибки для базового vs волнового роста и спада
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(percentiles, heatmap_matrix[0], label='Рост: Базовый масштаб (Err %)', color='#e67e22', linewidth=2)
    ax2.plot(percentiles, heatmap_matrix[1], label='Рост: С волновым накатом (Err %)', color='#27ae60', linewidth=2.5)
    ax2.plot(percentiles, heatmap_matrix[3], label='Спад: С волновым накатом (Err %)', color='#2980b9', linewidth=2)
    
    # Зеленая зона точности до 10%
    ax2.axhspan(0, 10, color='#2ecc71', alpha=0.15, label='Целевой диапазон точности (< 10%)')
    ax2.axvline(x=98, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='98-й перцентиль')
    
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 25)
    ax2.set_title('ПРОФИЛЬ ОШИБКИ ПО ВСЕМ КВАНТИЛЯМ ОТ 0% ДО 100%', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Квантиль движения цены (%)', fontsize=12)
    ax2.set_ylabel('Ошибка расхождения Train vs Val (%)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', framealpha=0.9)
    
    out_img = 'data_error_heatmap_0_100.png'
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"График сохранен в {out_img}", flush=True)
    
    # Выведем сводную таблицу перцентилей с шагом 5%
    print("\n" + "="*80)
    print("📊 СВОДНАЯ ТАБЛИЦА ПЕРЦЕНТИЛЕЙ Z И ОШИБКИ (TRAIN VS VAL)")
    print("="*80)
    print("Квантиль | Рост Train z | Рост Val z | Ошибка Роста | Спад Train z | Спад Val z | Ошибка Спада")
    print("-" * 80)
    for p in [5, 10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 98, 99]:
        q_up_tr = np.percentile(z_up_tr_wave, p)
        q_up_va = np.percentile(z_up_va_wave, p)
        err_up = abs(q_up_va - q_up_tr) / q_up_tr * 100
        
        q_dn_tr = np.percentile(z_dn_tr_wave, p)
        q_dn_va = np.percentile(z_dn_va_wave, p)
        err_dn = abs(q_dn_va - q_dn_tr) / q_dn_tr * 100
        
        flag = "✅ < 10%" if (err_up <= 10.0 and err_dn <= 10.0) else "⚠️ хвост"
        print(f"  {p:2d}%    |    {q_up_tr:.4f}    |   {q_up_va:.4f}   |    {err_up:5.2f}%   |    {q_dn_tr:.4f}    |   {q_dn_va:.4f}   |    {err_dn:5.2f}%   | {flag}")
    print("-" * 80)

if __name__ == '__main__':
    generate_heatmap_and_tables()
