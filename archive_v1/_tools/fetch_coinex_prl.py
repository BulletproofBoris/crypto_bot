"""
Специальный загрузчик исторических данных PRL/USDT с CoinEx.
CoinEx API v2 поддерживает параметр end_time для пагинации назад.
"""
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "5m" / "PRL_USDT_5M_MAX.csv"
META_FILE  = BASE_DIR / "data" / "raw" / "5m" / "PRL_USDT_5M_meta.json"

MARKET   = "PRLUSDT"
PERIOD   = "5min"
LIMIT    = 1000
START_DT = datetime(2026, 7, 12, tzinfo=timezone.utc)  # дата листинга PRL на CoinEx

def fetch_batch(end_time_sec: int) -> list:
    """Запрос LIMIT свечей, заканчивающихся не позже end_time_sec."""
    url = (
        f"https://api.coinex.com/v2/spot/kline"
        f"?market={MARKET}&period={PERIOD}&limit={LIMIT}"
        f"&price_type=latest_price&end_time={end_time_sec}"
    )
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "CryptoBot/1.0"})
        data = r.json()
        return data.get("data", [])
    except Exception as e:
        print(f"  ⚠️  Ошибка запроса: {e}")
        return []

def main():
    start_ts = int(START_DT.timestamp())
    now_ts   = int(datetime.now(timezone.utc).timestamp())
    end_ts   = now_ts

    all_rows = []
    batch_num = 0

    print(f"⬇️  Скачивание PRL/USDT 5m с CoinEx (с {START_DT.date()} по сегодня)")
    print(f"   Пагинация назад по end_time...")

    while end_ts > start_ts:
        batch = fetch_batch(end_ts)
        if not batch:
            print("  ⚠️  Пустой ответ, прерываем")
            break

        batch_num += 1
        # CoinEx возвращает [created_at, open, close, high, low, volume, value]
        rows = []
        for item in batch:
            ts = int(item.get("created_at", 0)) // 1000  # ms → sec
            if ts < start_ts:
                continue
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            rows.append({
                "Date":   dt,
                "Open":   float(item["open"]),
                "High":   float(item["high"]),
                "Low":    float(item["low"]),
                "Close":  float(item["close"]),
                "Volume": float(item["volume"]),
            })

        all_rows.extend(rows)
        oldest_ts = min(int(item["created_at"]) // 1000 for item in batch)
        oldest_dt = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
        print(f"  Батч {batch_num:3d}: {len(rows):4d} строк | старейшая свеча: {oldest_dt}")

        if oldest_ts <= start_ts:
            print("  ✅ Достигли начала истории")
            break
        if oldest_ts >= end_ts:
            print("  ⚠️  Нет продвижения в пагинации, прерываем")
            break

        end_ts = oldest_ts - 1  # сдвигаемся назад
        time.sleep(0.3)  # rate-limit

    if not all_rows:
        print("❌ Данные не получены")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Объединяем с существующими данными (если есть более свежие)
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        existing["Date"] = pd.to_datetime(existing["Date"], utc=True)
        df = pd.concat([df, existing], ignore_index=True)
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Сохранено {len(df)} строк → {OUTPUT_FILE.name}")
    print(f"   Диапазон: {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")

    # Обновляем мета-файл
    import json
    meta = {
        "first_trade_date": df["Date"].iloc[0].strftime("%Y-%m-%d"),
        "last_downloaded_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"   Мета-файл обновлён")

if __name__ == "__main__":
    main()
