import argparse
import os
import pandas as pd
import numpy as np
from numba import njit
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

@njit
def _get_max_min_returns(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, horizon: int):
    n = len(closes)
    valid_len = max(0, n - horizon)
    max_rets = np.zeros(valid_len)
    min_rets = np.zeros(valid_len)
    for i in range(valid_len):
        c = closes[i]
        if c == 0 or np.isnan(c): continue
        max_rets[i] = (np.max(highs[i+1 : i+horizon+1]) - c) / c
        min_rets[i] = (np.min(lows[i+1 : i+horizon+1]) - c) / c
    return max_rets, min_rets

@njit
def _apply_triple_barrier(prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, horizon: int, tp_factor: float, sl_factor: float) -> np.ndarray:
    n_rows = len(prices)
    labels = np.full(n_rows, 0.0) 
    for i in range(n_rows - horizon):
        entry_price = prices[i]
        if entry_price == 0 or np.isnan(entry_price): continue
        
        for j in range(1, horizon + 1):
            # 🚨 ФИКС №3: Обработка коллизий внутри одного дня (High-Low Logic)
            hit_sl = lows[i + j] <= entry_price * (1 - sl_factor)
            hit_tp = highs[i + j] >= entry_price * (1 + tp_factor)
            
            if hit_sl and hit_tp:
                # ПЕССИМИСТИЧНЫЙ СЦЕНАРИЙ: 
                # Если в течение дня пробиты ОБА барьера (волатильная свеча),
                # мы обязаны считать, что сначала выбило стоп-лосс. 
                # Это единственное правило, защищающее депозит в бэктестах на D1.
                labels[i] = -1.0
                break
            elif hit_sl:
                labels[i] = -1.0
                break
            elif hit_tp:
                labels[i] = 1.0
                break
    return labels

def process_ticker(ticker_name, df_ticker, horizon, tp, sl):
    df_ticker = df_ticker.sort_values('datetime').copy()
    c, h, l = df_ticker['close'].values, df_ticker['high'].values, df_ticker['low'].values
    
    labels = _apply_triple_barrier(c, h, l, horizon, tp / 100.0, sl / 100.0)
    df_ticker['target_return'] = df_ticker['close'].pct_change(horizon).shift(-horizon)
    df_ticker['label'] = labels
    df_ticker['target_tp'] = (labels == 1.0).astype(int)
    df_ticker['target_sl'] = (labels == -1.0).astype(int)
    return df_ticker.iloc[:-horizon]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--timeframe', type=str, default="1d")
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--tp', type=float, default=5.0)
    parser.add_argument('--sl', type=float, default=3.0)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--percentile', type=int, default=75)
    parser.add_argument('--workers', type=int, default=os.cpu_count())

    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df['datetime'] = pd.to_datetime(df['datetime'])

    final_tp, final_sl = args.tp, args.sl
    
    if args.auto:
        valid_max_all, valid_min_all = [], []
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values('datetime')
            c, h, l = group['close'].values, group['high'].values, group['low'].values
            max_r, min_r = _get_max_min_returns(c, h, l, args.horizon)
            if len(max_r[max_r > 0]) > 0: valid_max_all.append(max_r[max_r > 0])
            if len(min_r[min_r < 0]) > 0: valid_min_all.append(min_r[min_r < 0])
            
        if valid_max_all:
            final_tp = np.percentile(np.concatenate(valid_max_all), args.percentile) * 100
        if valid_min_all:
            final_sl = np.abs(np.percentile(np.concatenate(valid_min_all), 100 - args.percentile)) * 100
        print(f"✅ Авто-уровни: TP={final_tp:.2f}%, SL={final_sl:.2f}%")

    processed_dfs = []
    grouped_data = [group for _, group in df.groupby('ticker')]
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_ticker, group['ticker'].iloc[0], group, args.horizon, final_tp, final_sl): i for i, group in enumerate(grouped_data)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Разметка (Пессимистичный Тройной Барьер)", ncols=100, mininterval=2.0):
            try: processed_dfs.append(future.result())
            except Exception as e: print(f"❌ Ошибка: {e}")

    if not processed_dfs: return
    df_labeled = pd.concat(processed_dfs, ignore_index=True).sort_values(by=['datetime', 'ticker'])
    df_labeled.to_csv(args.output, index=False)
    print(f"🎉 Размеченный датасет сохранен: {Path(args.output).name}")

if __name__ == "__main__":
    main()