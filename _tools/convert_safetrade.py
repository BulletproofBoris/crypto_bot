import json
import pandas as pd
from datetime import datetime, timezone
import argparse
from pathlib import Path

def convert_safetrade_json_to_csv(json_path: str, output_csv: str):
    print(f"Читаем файл: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # SafeTrade / Peatio K-Line format:
    # [timestamp_sec, open, high, low, close, volume]
    
    rows = []
    for item in data:
        # Индекс 0: время в секундах
        ts_sec = int(item[0])
        dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        
        rows.append({
            "Date": dt,
            "Open": float(item[1]),
            "High": float(item[2]),
            "Low": float(item[3]),
            "Close": float(item[4]),
            "Volume": float(item[5])
        })
        
    df_new = pd.DataFrame(rows)
    print(f"Загружено {len(df_new)} свечей из JSON (с {df_new['Date'].min()} по {df_new['Date'].max()})")
    
    output_path = Path(output_csv)
    
    # Если уже есть CSV, сливаем данные, чтобы не потерять то, что было (или то, что скачалось с CoinEx)
    if output_path.exists():
        print(f"Найден существующий CSV: {output_csv}, объединяем...")
        df_old = pd.read_csv(output_path)
        df_old["Date"] = pd.to_datetime(df_old["Date"], utc=True)
        
        # Объединяем и удаляем дубликаты по Date
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    else:
        df_combined = df_new.sort_values("Date").reset_index(drop=True)
        
    # Форматируем цены с 3 знаками после запятой
    for col in ["Open", "High", "Low", "Close"]:
        df_combined[col] = df_combined[col].apply(lambda x: f"{x:.3f}")
        
    df_combined.to_csv(output_path, index=False)
    print(f"✅ Успешно сохранено {len(df_combined)} строк в {output_csv}")
    print(f"   Финальный диапазон: {df_combined['Date'].iloc[0]} -> {df_combined['Date'].iloc[-1]}")
    
    # Обновляем мета-файл
    meta_file = output_path.with_name(output_path.stem.replace("_MAX", "_meta") + ".json")
    meta = {
        "first_trade_date": df_combined["Date"].iloc[0].strftime("%Y-%m-%d"),
        "last_downloaded_date": df_combined["Date"].iloc[-1].strftime("%Y-%m-%d")
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"✅ Мета-файл обновлен: {meta_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SafeTrade JSON to standard OHLCV CSV")
    parser.add_argument("json_file", help="Path to input JSON file")
    parser.add_argument("--output", default="data/raw/5m/PRL_USDT_5M_MAX.csv", help="Path to output CSV")
    args = parser.parse_args()
    
    convert_safetrade_json_to_csv(args.json_file, args.output)
