import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def main():
    path = 'data/features/prl_1s_features.parquet'
    if not os.path.exists(path):
        path = 'data/processed/prl_1s_features.parquet'
        
    print(f"Загрузка данных из {path}...", flush=True)
    cols = [
        'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
        'ST_delta_vol', 'mid_price_diff'
    ]
    tbl = pq.read_table(path, columns=cols)
    df = tbl.to_pandas()
    
    n = len(df)
    split_idx = int(n * 0.8)
    TICK_SIZE = 0.005 # 0.005 USD
    H = 5 # Горизонт 5 секунд
    L = 20 # Окно наблюдения 20 секунд
    
    mid = df['ST_mid_price'].values
    spread = df['ST_spread'].values / TICK_SIZE # в тиках
    bids = df['ST_bids_vol_top5'].values
    asks = df['ST_asks_vol_top5'].values
    delta_vol = df['ST_delta_vol'].values
    arb_diff = df['mid_price_diff'].values / TICK_SIZE # спред BigONE vs SafeTrade в тиках
    
    # 5s price delta в тиках
    delta_usd = mid[H:] - mid[:-H]
    delta_ticks = np.round(delta_usd / TICK_SIZE)
    
    # 1s tick return
    step_1s_ticks = (mid[1:] - mid[:-1]) / TICK_SIZE
    
    # ДИНАМИЧЕСКИЕ МЕТРИКИ СТРОГО НА ОКНЕ L = 20 сек
    # 1. Локальная волатильность за окно L
    vol_L = pd.Series(step_1s_ticks).rolling(L, min_periods=L).std().values
    
    # 2. Локальная медианная глубина асков (сопротивление росту)
    asks_base_L = pd.Series(asks).rolling(L, min_periods=L).median().values
    
    # 3. Локальная медианная глубина бидов (сопротивление спаду)
    bids_base_L = pd.Series(bids).rolling(L, min_periods=L).median().values
    
    # 4. Локальный кумулятивный поток сделок CVD за окно L (в штуках PRL)
    cvd_L = pd.Series(delta_vol).rolling(L, min_periods=L).sum().values
    
    # 5. Локальный средний спред за окно L
    spread_base_L = pd.Series(spread).rolling(L, min_periods=L).median().values
    
    valid_start = L
    valid_end = n - H
    
    dt = delta_ticks[valid_start:valid_end]
    vt = vol_L[valid_start:valid_end]
    
    a_cur = asks[valid_start:valid_end]
    a_base = asks_base_L[valid_start:valid_end]
    b_cur = bids[valid_start:valid_end]
    b_base = bids_base_L[valid_start:valid_end]
    
    sp_cur = spread[valid_start:valid_end]
    sp_base = spread_base_L[valid_start:valid_end]
    
    cvd = cvd_L[valid_start:valid_end]
    arb = arb_diff[valid_start:valid_end]
    
    # Расчет локальных динамических коэффициентов:
    # 1. Асимметричное сопротивление сторон
    up_res = np.clip(a_base / np.maximum(a_cur, 50.0), 0.5, 3.0)
    dn_res = np.clip(b_base / np.maximum(b_cur, 50.0), 0.5, 3.0)
    
    # 2. Спредовый фактор за окно L
    sp_factor = np.clip(sp_cur / np.maximum(sp_base, 0.5), 0.5, 3.0)
    
    # 3. Фактор кинетической инерции сделок (CVD за 20с)
    # Нормируем поток относительно текущей глубины
    cvd_norm_up = np.clip(cvd / np.maximum(a_cur, 100.0), -2.0, 2.0)
    cvd_norm_dn = np.clip(-cvd / np.maximum(b_cur, 100.0), -2.0, 2.0)
    
    # 4. Фактор кросс-биржевого давления (BigONE vs SafeTrade)
    arb_up = np.clip(arb / 2.0, -1.0, 1.0)
    arb_dn = np.clip(-arb / 2.0, -1.0, 1.0)
    
    # Полный многофакторный динамический порог вверх и вниз
    c_base = 2.4
    v_safe = np.nan_to_num(vt, nan=0.1)
    
    # Порог для движения вверх:
    # Ослабляется, если сильный поток покупок (cvd > 0) или BigONE выше (arb > 0)
    # Ужесточается, если аски истощены или спред широк
    up_th_raw = c_base * v_safe * (up_res ** 0.3) * (sp_factor ** 0.2) * (1.0 - 0.15 * cvd_norm_up - 0.10 * arb_up)
    up_thresh = np.maximum(2, np.round(up_th_raw)).astype(int)
    
    # Порог для движения вниз:
    dn_th_raw = c_base * v_safe * (dn_res ** 0.3) * (sp_factor ** 0.2) * (1.0 - 0.15 * cvd_norm_dn - 0.10 * arb_dn)
    dn_thresh = np.maximum(2, np.round(dn_th_raw)).astype(int)
    
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
    
    print("\n" + "="*70, flush=True)
    print("📊 РЕЗУЛЬТАТЫ МНОГОФАКТОРНОГО ДИНАМИЧЕСКОГО ТЕСТА (L=20s, H=5s)", flush=True)
    print("="*70, flush=True)
    print(f"Отрезок обучения: {len(tr_lbl):,} (80%) | Проверочный: {len(va_lbl):,} (20%)", flush=True)
    
    print("\n" + "-"*75, flush=True)
    print("Класс | Описание       | Train % (Сэмплов)  | Val % (Сэмплов)   | Train сдвиг | Val сдвиг", flush=True)
    print("-"*75, flush=True)
    
    class_names = {
        -2: "Сильный дамп",
        -1: "Сдвиг 1 тик ",
         0: "Флэт (0 тиков)",
         1: "Сдвиг 1 тик ",
         2: "Сильный памп"
    }
    
    for c in [-2, -1, 0, 1, 2]:
        tr_mask = (tr_lbl == c)
        va_mask = (va_lbl == c)
        
        tr_cnt = np.sum(tr_mask)
        tr_pct = tr_cnt / len(tr_lbl) * 100
        tr_m = np.mean(tr_dt[tr_mask]) if tr_cnt > 0 else 0
        tr_med = np.median(tr_dt[tr_mask]) if tr_cnt > 0 else 0
        
        va_cnt = np.sum(va_mask)
        va_pct = va_cnt / len(va_lbl) * 100
        va_m = np.mean(va_dt[va_mask]) if va_cnt > 0 else 0
        va_med = np.median(va_dt[va_mask]) if va_cnt > 0 else 0
        
        name = class_names[c]
        print(f" {c:+2d}   | {name} | {tr_pct:5.2f}% ({tr_cnt:6,}) | {va_pct:5.2f}% ({va_cnt:6,}) | {tr_m:+.2f} т [мед:{tr_med:+.1f}] | {va_m:+.2f} т [мед:{va_med:+.1f}]", flush=True)
        
    print("-"*75, flush=True)
    
    # Клиппинг аномальных спайков (> 4 тиков)
    print("\nСравнение с фильтрацией аномальных редких спайков (> 4 тиков):", flush=True)
    tr_p2_clip = np.clip(tr_dt[tr_lbl == 2], 0, 4)
    va_p2_clip = np.clip(va_dt[va_lbl == 2], 0, 4)
    tr_m2_clip = np.clip(tr_dt[tr_lbl == -2], -4, 0)
    va_m2_clip = np.clip(va_dt[va_lbl == -2], -4, 0)
    
    print(f"  • Класс +2 (Срез до 4 тиков): Train = {np.mean(tr_p2_clip):+.2f} тика | Val = {np.mean(va_p2_clip):+.2f} тика (Ошибка: {abs(np.mean(tr_p2_clip) - np.mean(va_p2_clip)):.2f} тика)", flush=True)
    print(f"  • Класс -2 (Срез до 4 тиков): Train = {np.mean(tr_m2_clip):+.2f} тика | Val = {np.mean(va_m2_clip):+.2f} тика (Ошибка: {abs(np.mean(tr_m2_clip) - np.mean(va_m2_clip)):.2f} тика)", flush=True)

if __name__ == '__main__':
    main()
