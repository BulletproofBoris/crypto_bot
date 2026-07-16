import os
import asyncio
import random
import json
import pandas as pd
from typing import Optional, List
from pathlib import Path
from datetime import datetime, timezone
import ccxt.async_support as ccxt
from tqdm.auto import tqdm

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import RAW_DIR  

DATE_CANDIDATES: List[str] = ["datetime", "date", "TRADEDATE", "DATE", "Datetime"]

# Карта перевода наших MLOps-таймфреймов в формат ccxt
TIMEFRAME_MAPPING = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}

# Длительность одной свечи в миллисекундах (для расчёта прогресс-бара)
TF_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
    "1w":  604_800_000,
}

# =====================================================================
# ЧАСТЬ 1: РАБОТА С ЛОКАЛЬНЫМИ CSV 
# =====================================================================

def _resolve_path(ticker_or_path: str) -> Optional[str]:
    if os.path.exists(ticker_or_path):
        return ticker_or_path
        
    ticker = str(ticker_or_path).upper().strip()
    safe_symbol = ticker.replace('/', '_').replace('@', '_')
    
    # Ищем файл в новых папках таймфреймов (начиная с 1d)
    for tf in ['1d', '1h', '30m', '15m', '5m', '1m']:
        tf_dir = RAW_DIR / tf
        candidates = [
            f"{safe_symbol}_{tf.upper()}_MAX.csv", 
            f"{ticker}_{tf.upper()}_MAX.csv",
            f"{safe_symbol}.csv", 
            f"{ticker}.csv"
        ]
        for fname in candidates:
            p = tf_dir / fname
            if p.exists():
                return str(p)
    return None

def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    cols = {c.lower() for c in df.columns}
    for candidate in DATE_CANDIDATES:
        if candidate.lower() in cols:
            for original_col in df.columns:
                if original_col.lower() == candidate.lower():
                    return original_col
    return None

def load_data(ticker_or_path: str) -> Optional[pd.DataFrame]:
    file_path = _resolve_path(ticker_or_path)
    if not file_path:
        return None
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception as e:
        print(f"[load_data] ❌ Ошибка чтения {file_path}: {e}")
        return None

    if df is None or df.empty:
        return None
        
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    df.columns = [str(col).lower() for col in df.columns]

    date_col = _detect_date_column(df)
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
            df = df.rename(columns={date_col: "datetime"})
            df.dropna(subset=['datetime'], inplace=True)
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f"[load_data] ❌ Ошибка обработки даты в {file_path}: {e}")
            return None
            
    return df

# =====================================================================
# ЧАСТЬ 2: СКАЧИВАНИЕ ДАННЫХ ИЗ CCXT (Криптобиржи)
# =====================================================================

class CryptoClient:
    def __init__(self, exchange_id='binance'):
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True,
        })

    async def disconnect(self):
        await self.exchange.close()

    def _update_meta(self, df: pd.DataFrame, meta_file: Path):
        if df is not None and not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            first_date_str = df['Date'].min().strftime('%Y-%m-%d')
            last_date_str = df['Date'].max().strftime('%Y-%m-%d')
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump({"first_trade_date": first_date_str, "last_downloaded_date": last_date_str}, f, indent=4)

    async def fetch_history(self, symbol: str, start_year: int = 2017, timeframe: str = "1d"):
        """
        Умное инкрементальное выкачивание с криптобиржи.
        """
        current_time = datetime.now(timezone.utc)
        today = current_time.date()
        all_data = []

        ccxt_tf = TIMEFRAME_MAPPING.get(timeframe.lower(), "1d")
        
        target_dir = RAW_DIR / timeframe.lower()
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_symbol = symbol.replace('/', '_').replace('@', '_')
        file_path = target_dir / f"{safe_symbol}_{timeframe.upper()}_MAX.csv"
        meta_file = target_dir / f"{safe_symbol}_{timeframe.upper()}_meta.json"
        
        existing_df = pd.DataFrame()
        
        start_ts = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        # --- 1. ЧТЕНИЕ МЕТАДАННЫХ ---
        if meta_file.exists() and file_path.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                last_date = pd.to_datetime(meta["last_downloaded_date"]).date()
                if last_date >= today:
                    print(f"  ✅ {symbol} | Актуально на {today}. Обновление не требуется.")
                    try:
                        existing_df = pd.read_csv(file_path)
                    except:
                        pass
                    return existing_df
                else:
                    # Начинаем качать с последней скачанной даты
                    start_ts = int(pd.to_datetime(meta["last_downloaded_date"]).timestamp() * 1000)
                existing_df = pd.read_csv(file_path)
            except Exception as e:
                print(f"  ⚠️ Ошибка чтения метафайла {symbol}: {e}")
        elif file_path.exists():
            try:
                existing_df = pd.read_csv(file_path)
                dates = pd.to_datetime(existing_df['Date'])
                last_date = max(dates)
                if last_date.date() >= today:
                    print(f"  ✅ {symbol} | Актуально на {today}. Обновление не требуется.")
                    return existing_df
                start_ts = int(last_date.timestamp() * 1000)
            except:
                pass

        now_ts = int(current_time.timestamp() * 1000)
        tf_ms = TF_MS.get(timeframe.lower(), 3_600_000)
        estimated_total = max(1, (now_ts - start_ts) // tf_ms)
        start_date_str = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d')

        # --- 2. ЗАГРУЗКА ДАННЫХ ЧЕРЕЗ CCXT ---
        limit = 1000
        current_ts = start_ts
        retries = 0

        with tqdm(
            total=estimated_total,
            desc=f"📥 {symbol} ({timeframe}) c {start_date_str}",
            unit=" св",
            dynamic_ncols=True,
            colour="cyan",
            leave=True,
        ) as pbar:
            while True:
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, ccxt_tf, since=current_ts, limit=limit)
                    if not ohlcv:
                        break

                    for b in ohlcv:
                        dt = datetime.fromtimestamp(b[0] / 1000.0, tz=timezone.utc)
                        all_data.append({
                            'Date': dt.date() if timeframe.lower() == '1d' else dt,
                            'Open': float(b[1]),
                            'High': float(b[2]),
                            'Low': float(b[3]),
                            'Close': float(b[4]),
                            'Volume': float(b[5])
                        })

                    batch_size = len(ohlcv)
                    last_ts = ohlcv[-1][0]
                    last_date_str = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                    pbar.set_postfix_str(f"по {last_date_str}", refresh=False)
                    pbar.update(batch_size)


                    if last_ts <= current_ts:
                        break

                    current_ts = last_ts + 1
                    retries = 0  # Сбрасываем счётчик после успешного запроса

                    if batch_size < limit:
                        break  # Достигли конца доступных данных

                except Exception as e:
                    retries += 1
                    tqdm.write(f"  ⏳ {symbol} | Ошибка CCXT: {e}. Попытка {retries}/3...")
                    if retries >= 3:
                        tqdm.write(f"  ❌ {symbol} | Превышено число попыток. Прерываем загрузку.")
                        break
                    await asyncio.sleep(5)

        if all_data:
            tqdm.write(f"  ✅ {symbol} | Получено новых свечей: {len(all_data)}")

        # --- 3. СЛИЯНИЕ ДАННЫХ И ОБНОВЛЕНИЕ МЕТАФАЙЛА ---
        if all_data:
            new_df = pd.DataFrame(all_data)
            df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
            df.to_csv(file_path, index=False)
        else:
            df = existing_df if not existing_df.empty else None

        self._update_meta(df, meta_file)
        return df