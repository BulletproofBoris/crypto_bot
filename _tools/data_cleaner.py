import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

KNOWN_MOEX_SPLITS = {
    'TRNFP@MISX': [('2024-02-21', 0.01)],
    'SBER@MISX':  [('2007-07-18', 0.001)],
    'SBERP@MISX': [('2007-07-18', 0.001)],
}
STANDARD_RATIOS = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]

def apply_corporate_actions(df, ticker):
    g = df.sort_values('datetime').copy()
    g.reset_index(drop=True, inplace=True)
    
    dates = g['datetime'].dt.strftime('%Y-%m-%d').values
    closes = g['close'].values.astype(float)
    opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float)
    lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)
    
    n = len(closes)
    if n < 5: return g
    is_macro = ticker.startswith('MACRO_')

    # Фильтр "шпилек" (1 день выброса с возвратом)
    for i in range(1, n - 1):
        prev_c, curr_c, next_c = closes[i-1], closes[i], closes[i+1]
        if prev_c <= 0 or curr_c <= 0: continue
        r1, r2 = curr_c / prev_c, next_c / curr_c
        if (r1 > 1.4 or r1 < 0.7) and (0.85 < r1 * r2 < 1.15):
            fix_val = (prev_c + next_c) / 2.0
            closes[i] = opens[i] = highs[i] = lows[i] = fix_val

    adj_factor = 1.0
    adj_closes, adj_opens, adj_highs, adj_lows, adj_vols = np.copy(closes), np.copy(opens), np.copy(highs), np.copy(lows), np.copy(volumes)
    known_splits = KNOWN_MOEX_SPLITS.get(ticker, [])
    known_split_dates = {date for date, ratio in known_splits}

    # Корректировка сплитов
    for i in range(n - 1, 0, -1):
        curr_c, prev_c, curr_date = closes[i], closes[i-1], dates[i]
        if prev_c <= 0: continue
        ret = curr_c / prev_c
        ratio_to_apply = None
        
        if not is_macro:
            if curr_date in known_split_dates:
                for d, r in known_splits:
                    if d == curr_date:
                        ratio_to_apply, _ = r, print(f"  📌 [KNOWN SPLIT] {ticker} на {curr_date}: Коэфф {r}")
                        break
            # 🚨 ФИКС №2: Жесткие рамки для эвристики (чтобы не удалить крахи рынка как в феврале 2022)
            # Падение на 45% (ret=0.55) - это крах, а не сплит! Сплит 2:1 это ret=0.5
            elif ret > 1.8 or ret < 0.55: 
                for sr in STANDARD_RATIOS:
                    if 0.95 <= ret / sr <= 1.05:
                        ratio_to_apply, _ = sr, print(f"  🕵️‍♂️ [HEURISTIC SPLIT] {ticker} на {curr_date}: Коэфф {sr}")
                        break
        
        if ratio_to_apply is not None: adj_factor *= ratio_to_apply
        if adj_factor != 1.0:
            adj_closes[i-1] = prev_c * adj_factor
            adj_opens[i-1]  = opens[i-1] * adj_factor
            adj_highs[i-1]  = highs[i-1] * adj_factor
            adj_lows[i-1]   = lows[i-1] * adj_factor
            adj_vols[i-1]   = volumes[i-1] / adj_factor if volumes[i-1] > 0 else 0.0

    # 🚨 ФИКС №1: УБРАНО center=True. Заглядывание в будущее недопустимо!
    # Берем медиану только по ПРЕДЫДУЩИМ 31 дням.
    med_values = pd.Series(adj_closes).rolling(window=31, min_periods=1).median().values
    med_values = np.where(med_values == 0, 1e-8, med_values)
    
    # Сравниваем с медианой прошлого
    deviations = np.abs(adj_closes - med_values) / med_values
    
    bad_idx = deviations > 0.6
    if np.any(bad_idx):
        adj_closes[bad_idx] = adj_opens[bad_idx] = adj_highs[bad_idx] = adj_lows[bad_idx] = med_values[bad_idx]

    g['close'], g['open'], g['high'], g['low'], g['volume'] = adj_closes, adj_opens, adj_highs, adj_lows, adj_vols
    return g

def clean_and_adjust_data(df):
    print("\n🧹 Запуск модуля очистки данных (Сплиты, Иглы, Выбросы без заглядывания в будущее)...")
    df.columns = [c.lower() for c in df.columns]
    
    cleaned_dfs = []
    for ticker in tqdm(df['ticker'].unique(), desc="Корректировка сплитов", ncols=100, mininterval=2.0):
        cleaned_dfs.append(apply_corporate_actions(df[df['ticker'] == ticker], ticker))
        
    final_df = pd.concat(cleaned_dfs, ignore_index=True)
    final_df = final_df[final_df['close'] > 0]
    print("✅ Очистка завершена!")
    return final_df