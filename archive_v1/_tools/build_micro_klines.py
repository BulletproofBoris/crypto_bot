import os
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRADES = BASE_DIR / "data" / "realtime" / "prlusdt" / "trades.csv"

def build_klines(trades_path: str, timeframe: str, output_path: str):
    print(f"📥 Загрузка сделок из {trades_path}...")
    try:
        df = pd.read_csv(trades_path)
    except FileNotFoundError:
        print(f"❌ Файл не найден: {trades_path}")
        return

    if df.empty:
        print("⚠️ Файл со сделками пуст.")
        return

    # Конвертируем метку времени. Peatio created_at может быть строкой или unix timestamp.
    # Используем `created_at` (время создания сделки на бирже), а не `timestamp` (время парсинга)
    # Если created_at это unix timestamp:
    # df['datetime'] = pd.to_datetime(df['created_at'], unit='s', utc=True) 
    # Если ISO-8601:
    df['datetime'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)

    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    print(f"📊 Агрегация свечей (Таймфрейм: {timeframe})...")
    
    # Ресемплирование
    # Цена: open, high, low, close
    # Объем: sum
    ohlc_dict = {
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum',
        'funds': 'sum'
    }
    
    klines = df.resample(timeframe).agg(ohlc_dict)
    
    # Убираем мультииндекс столбцов
    klines.columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    
    # Заполняем пропуски в периодах, когда не было сделок:
    # Объем = 0, Цена копируется с предыдущего close
    klines['volume'] = klines['volume'].fillna(0)
    klines['quote_volume'] = klines['quote_volume'].fillna(0)
    klines['close'] = klines['close'].ffill()
    klines['open'] = klines['open'].fillna(klines['close'])
    klines['high'] = klines['high'].fillna(klines['close'])
    klines['low'] = klines['low'].fillna(klines['close'])

    # Удаляем пустые строки в самом начале (до первой сделки)
    klines.dropna(subset=['close'], inplace=True)
    
    klines.reset_index(inplace=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    klines.to_csv(out_file, index=False)
    
    print(f"✅ Успешно сгенерировано {len(klines)} свечей таймфрейма {timeframe}.")
    print(f"💾 Сохранено в {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка микро-свечей (5s, 15s, 1m) из сырых сделок")
    parser.add_argument("--input", type=str, default=str(DEFAULT_TRADES), help="Путь к trades.csv")
    parser.add_argument("--timeframe", type=str, default="5s", help="Pandas timeframe (5s, 15s, 1min, 5min)")
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "data" / "raw" / "5s" / "PRL_USDT_5S_MAX.csv"))
    
    args = parser.parse_args()
    build_klines(args.input, args.timeframe, args.output)
