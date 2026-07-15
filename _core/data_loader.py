import os

# ОТКЛЮЧАЕМ системное хранилище паролей, чтобы потоки не блокировали друг друга
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"

import asyncio
import random
import json
import pandas as pd
from typing import Optional, List
from pathlib import Path
from datetime import datetime, timezone
from google.protobuf.json_format import ParseDict

# --- Импорты Финама ---
from FinamPy import FinamPy
from FinamPy.grpc.marketdata_service_pb2 import BarsRequest

# --- Настройки проекта ---
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import FINAM_TOKEN, RAW_DIR  

DATE_CANDIDATES: List[str] = ["datetime", "date", "TRADEDATE", "DATE", "Datetime"]

# Карта перевода наших MLOps-таймфреймов в формат Finam API
TIMEFRAME_MAPPING = {
    "1m": "TIME_FRAME_M1",
    "5m": "TIME_FRAME_M5",
    "15m": "TIME_FRAME_M15",
    "30m": "TIME_FRAME_M30",
    "1h": "TIME_FRAME_H1",
    "4h": "TIME_FRAME_H4",
    "1d": "TIME_FRAME_D",
    "1w": "TIME_FRAME_W",
}

# =====================================================================
# ЧАСТЬ 1: РАБОТА С ЛОКАЛЬНЫМИ CSV 
# =====================================================================

def _resolve_path(ticker_or_path: str) -> Optional[str]:
    if os.path.exists(ticker_or_path):
        return ticker_or_path
        
    ticker = str(ticker_or_path).upper().strip()
    safe_symbol = ticker.replace('@', '_')
    
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
# ЧАСТЬ 2: СКАЧИВАНИЕ ДАННЫХ ИЗ FINAM (gRPC клиент)
# =====================================================================

class FinamClient:
    def __init__(self, token: str = FINAM_TOKEN):
        self.token = token
        self.fp = None

    def connect(self):
        if not self.fp:
            self.fp = FinamPy(self.token)

    def disconnect(self):
        if self.fp:
            self.fp.close_channel()
            self.fp = None

    def _parse_price(self, decimal_obj) -> float:
        if hasattr(decimal_obj, 'value'):
            return float(decimal_obj.value)
        elif hasattr(decimal_obj, 'num'):
            return decimal_obj.num * (10 ** -decimal_obj.scale)
        return float(decimal_obj)

    def _update_meta(self, df: pd.DataFrame, meta_file: Path):
        if df is not None and not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            first_date_str = df['Date'].min().strftime('%Y-%m-%d')
            last_date_str = df['Date'].max().strftime('%Y-%m-%d')
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump({"first_trade_date": first_date_str, "last_downloaded_date": last_date_str}, f, indent=4)

    async def fetch_history(self, symbol: str, start_year: int = 2010, timeframe: str = "1d"):
        """
        Умное инкрементальное выкачивание.
        timeframe: короткий формат (1d, 1h, 30m)
        """
        self.connect()
        current_time = datetime.now(timezone.utc)
        current_year = current_time.year
        today = current_time.date()
        all_data = []

        # Конвертируем MLOps-таймфрейм в формат Finam
        finam_tf_enum = TIMEFRAME_MAPPING.get(timeframe.lower(), "TIME_FRAME_D")
        
        # Динамически определяем целевую папку
        target_dir = RAW_DIR / timeframe.lower()
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_symbol = symbol.replace('@', '_')
        file_path = target_dir / f"{safe_symbol}_{timeframe.upper()}_MAX.csv"
        meta_file = target_dir / f"{safe_symbol}_{timeframe.upper()}_meta.json"
        
        existing_df = pd.DataFrame()
        years_to_fetch = list(range(start_year, current_year + 1))
        
        # --- 1. ЧТЕНИЕ МЕТАДАННЫХ ---
        if meta_file.exists() and file_path.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                last_date = pd.to_datetime(meta["last_downloaded_date"]).date()
                if last_date >= today:
                    years_to_fetch = []
                else:
                    years_to_fetch = list(range(last_date.year, current_year + 1))
                existing_df = pd.read_csv(file_path)
            except Exception as e:
                print(f"  ⚠️ Ошибка чтения метафайла {symbol}: {e}")
        elif file_path.exists():
            try:
                existing_df = pd.read_csv(file_path)
                dates = pd.to_datetime(existing_df['Date']).dt.date
                min_y, max_y = min(dates).year, max(dates).year
                last_date = max(dates)
                missing_early = [y for y in range(start_year, min_y)]
                missing_late = [] if last_date >= today else [y for y in range(max_y, current_year + 1)]
                years_to_fetch = sorted(list(set(missing_early + missing_late)))
            except:
                pass

        if not years_to_fetch:
            print(f"  ✅ {symbol} | Актуально на {today}. Обновление не требуется.")
            self._update_meta(existing_df, meta_file)
            return existing_df

        print(f"🚀 {symbol} ({timeframe}) | Докачиваем года: {years_to_fetch}")

        try:
            # --- 2. ЗАГРУЗКА НЕДОСТАЮЩИХ ЛЕТ ---
            for year in years_to_fetch:
                year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
                year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                if year_end > current_time:
                    year_end = current_time

                request_data = {
                    "symbol": symbol,
                    "timeframe": finam_tf_enum, # <-- ПЕРЕДАЕМ ПРАВИЛЬНЫЙ ENUM ФИНАМА!
                    "interval": {
                        "startTime": year_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "endTime": year_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                }
                
                request = BarsRequest()
                ParseDict(request_data, request)

                max_retries = 6  
                success = False
                
                for attempt in range(max_retries):
                    response = self.fp.call_function(self.fp.marketdata_stub.Bars, request)
                    
                    if response is not None:
                        if response.bars and len(response.bars) > 0:
                            print(f"  ✅ {symbol} | {year} год: Получено свечей - {len(response.bars)}")
                            for b in response.bars:
                                dt = datetime.fromtimestamp(b.timestamp.seconds, tz=timezone.utc)
                                all_data.append({
                                    'Date': dt.date() if timeframe.lower() == '1d' else dt, # Если часы, сохраняем и время
                                    'Open': self._parse_price(b.open),
                                    'High': self._parse_price(b.high),
                                    'Low': self._parse_price(b.low),
                                    'Close': self._parse_price(b.close),
                                    'Volume': self._parse_price(b.volume)
                                })
                        else:
                            print(f"  ℹ️ {symbol} | {year} год: Данных нет")
                        success = True
                        break  
                    else:
                        delay = (2 ** attempt) + random.uniform(0, 1.5)  
                        print(f"  ⏳ {symbol} | {year}: Сбой API. Ждем {delay:.1f} сек...")
                        await asyncio.sleep(delay)
                
                if not success:
                    print(f"  ❌ {symbol} | {year}: Пропущен")

                await asyncio.sleep(1.0) 

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

        finally:
            self.disconnect()