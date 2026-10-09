#!/usr/bin/env python3
"""
Генерация базового 1-секундного датасета с первичными фичами.
Работает напрямую с объединенным Parquet хранилищем через get_market_data.
Вычисляет:
- Внутрибиржевые спреды и mid-price
- Дисбаланс стакана (Orderbook Imbalance) по топ-5 уровням
- Агрегированные объемы покупок и продаж (1s)
- Межбиржевой арбитражный спред (SafeTrade vs BigONE)
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import numpy as np
import pandas as pd
from colorama import init, Fore, Style

from _tools.sync_all_data import get_market_data

init(autoreset=True)

def process_exchange_data(depth_df, trades_df, prefix):
    print(f"[{prefix}] Обработка данных...")

    if depth_df is None or depth_df.empty:
        raise ValueError(f"Стакан для {prefix} пуст!")

    df_d = depth_df.copy()
    df_t = trades_df.copy() if trades_df is not None and not trades_df.empty else pd.DataFrame()

    # 1. Приведение к московскому времени для читабельности
    if df_d['timestamp'].dt.tz is None:
        df_d['timestamp'] = df_d['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
    else:
        df_d['timestamp'] = df_d['timestamp'].dt.tz_convert('Europe/Moscow')

    df_d = df_d.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    df_d.set_index('timestamp', inplace=True)

    # 2. Ресемплинг стакана (1s, ffill)
    print(f"[{prefix}] Ресемплинг стаканов до 1 секунды...")
    depth_cols = [c for c in df_d.columns if 'bid' in c or 'ask' in c]
    depth_1s = df_d[depth_cols].resample('1s').ffill()

    # 3. Вычисление базовых фичей стакана
    print(f"[{prefix}] Расчет фичей стакана (спред, mid-price, дисбаланс)...")
    best_bid = depth_1s['bid_1_price'].astype(float)
    best_ask = depth_1s['ask_1_price'].astype(float)
    mid_price = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    spread_pct = spread / mid_price

    # Дисбаланс стакана (топ-5 уровней)
    bids_vol_top5 = sum(depth_1s[f'bid_{i}_qty'].astype(float) for i in range(1, 6))
    asks_vol_top5 = sum(depth_1s[f'ask_{i}_qty'].astype(float) for i in range(1, 6))
    total_vol_top5 = bids_vol_top5 + asks_vol_top5
    ob_imbalance = (bids_vol_top5 - asks_vol_top5) / total_vol_top5.replace(0, np.nan)

    depth_1s['mid_price'] = mid_price
    depth_1s['spread'] = spread
    depth_1s['spread_pct'] = spread_pct
    depth_1s['imbalance_top5'] = ob_imbalance.fillna(0)

    # 4. Ресемплинг сделок (1s, Aggregation)
    if not df_t.empty:
        print(f"[{prefix}] Агрегация сделок до 1 секунды...")
        if df_t['timestamp'].dt.tz is None:
            df_t['timestamp'] = df_t['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        else:
            df_t['timestamp'] = df_t['timestamp'].dt.tz_convert('Europe/Moscow')

        df_t['amount'] = df_t['amount'].astype(float)
        # Определяем колонку стороны сделки (side для BigONE, type для SafeTrade)
        side_col = 'side' if 'side' in df_t.columns else ('type' if 'type' in df_t.columns else None)
        if side_col:
            df_t['side_clean'] = df_t[side_col].astype(str).str.upper()
            df_t['buy_vol'] = np.where(df_t['side_clean'].isin(['ASK', 'BUY']), df_t['amount'], 0.0)
            df_t['sell_vol'] = np.where(df_t['side_clean'].isin(['BID', 'SELL']), df_t['amount'], 0.0)
        else:
            df_t['buy_vol'] = 0.0
            df_t['sell_vol'] = 0.0

        df_t = df_t.sort_values('timestamp')
        df_t.set_index('timestamp', inplace=True)

        id_col = 'id' if 'id' in df_t.columns else 'amount'
        trades_1s = df_t.resample('1s').agg({
            'amount': 'sum',
            'buy_vol': 'sum',
            'sell_vol': 'sum',
            id_col: 'count'
        }).rename(columns={'amount': 'trade_vol', id_col: 'trade_count'})
        trades_1s = trades_1s.fillna(0)
    else:
        trades_1s = pd.DataFrame(index=depth_1s.index, data={
            'trade_vol': 0.0,
            'buy_vol': 0.0,
            'sell_vol': 0.0,
            'trade_count': 0
        })

    # 5. Объединение стаканов и сделок
    df_1s = pd.merge(depth_1s, trades_1s, left_index=True, right_index=True, how='left')
    df_1s[['trade_vol', 'buy_vol', 'sell_vol', 'trade_count']] = df_1s[['trade_vol', 'buy_vol', 'sell_vol', 'trade_count']].fillna(0)

    # Добавляем префикс биржи ко всем колонкам
    df_1s.columns = [f"{prefix}_{c}" for c in df_1s.columns]
    print(f"{Fore.GREEN}✅ [{prefix}] Обработка завершена. Строк: {len(df_1s):,}{Style.RESET_ALL}\n")
    return df_1s

def main():
    parser = argparse.ArgumentParser(description="Генерация 1-секундного датасета признаков.")
    parser.add_argument("--sync", action="store_true", help="Сначала выполнить синхронизацию с Google Drive и VPS.")
    args = parser.parse_args()

    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}⚙️ ГЕНЕРАЦИЯ БАЗОВОГО 1-СЕКУНДНОГО ДАТАСЕТА{Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}\n")

    st_depth, st_trades, bo_depth, bo_trades = get_market_data(force_sync=args.sync)

    df_bo = process_exchange_data(bo_depth, bo_trades, "BO")
    df_st = process_exchange_data(st_depth, st_trades, "ST")

    # Слияние данных двух бирж по точному времени
    print("Слияние данных BigONE и SafeTrade в единый таймлайн...")
    merged_df = pd.merge(df_bo, df_st, left_index=True, right_index=True, how='inner')
    merged_df = merged_df.dropna()

    # Межбиржевые арбитражные фичи
    print("Расчет межбиржевых арбитражных спредов...")
    # Покупка на SafeTrade, продажа на BigONE:
    merged_df['arb_st_buy_bo_sell'] = merged_df['BO_bid_1_price'].astype(float) - merged_df['ST_ask_1_price'].astype(float)
    # Покупка на BigONE, продажа на SafeTrade:
    merged_df['arb_bo_buy_st_sell'] = merged_df['ST_bid_1_price'].astype(float) - merged_df['BO_ask_1_price'].astype(float)

    # Разница средних цен (Mid-price difference):
    merged_df['mid_price_diff'] = merged_df['BO_mid_price'] - merged_df['ST_mid_price']

    # Сохранение результатов
    output_dir = os.path.join(BASE_DIR, 'data', 'features')
    os.makedirs(output_dir, exist_ok=True)
    parquet_out = os.path.join(output_dir, 'merged_1s_base.parquet')

    merged_df.to_parquet(parquet_out)
    parquet_mb = os.path.getsize(parquet_out) / (1024 * 1024)

    print(f"{Fore.GREEN}==================================================={Style.RESET_ALL}")
    print(f"🎉 Итоговый датасет признаков успешно создан!")
    print(f"📁 Parquet: {parquet_out} ({parquet_mb:.2f} MB)")
    print(f"📊 Размер: {len(merged_df):,} строк (секунд) x {len(merged_df.columns)} признаков")
    print(f"⏰ Период: {merged_df.index.min()} -> {merged_df.index.max()}")
    print(f"{Fore.GREEN}==================================================={Style.RESET_ALL}")

if __name__ == "__main__":
    main()
