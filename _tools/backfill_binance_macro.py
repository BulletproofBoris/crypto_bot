#!/usr/bin/env python3
"""
Ретроспективная выгрузка секундных макро-поводырей Binance:
- BTC/USDT (мировой локомотив крипторынка)
- NEAR/USDT (поводырь экосистемы NEAR и Quantus)

Диапазон: с 3 октября 2026 00:00:00 UTC (за неделю до старта торгов)
по 11 октября 2026 18:00:00 UTC (стыковка с активным онлайн-сборщиком).

Особенности:
- Запрос 1s klines через Binance REST API пакетами по 1000 секунд
- Контроль веса запросов через заголовок x-mbx-used-weight-1m
- Посуточная компоновка и сохранение в Parquet Data Lake и суточные архивы Google Диска
"""

import os
import sys
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.sync_all_data import get_drive_service, FOLDER_ID, CACHE_GDRIVE_DIR, PROCESSED_DIR
from _tools.unify_gdrive_data import get_or_create_folder, upload_or_replace_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("backfill_macro")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
START_DT = datetime(2026, 10, 3, 0, 0, 0, tzinfo=timezone.utc)
END_DT = datetime(2026, 10, 11, 18, 0, 0, tzinfo=timezone.utc)

session = requests.Session()

def fetch_chunk(symbol, start_ms, limit=1000):
    url = f"{BINANCE_KLINES_URL}?symbol={symbol}&interval=1s&startTime={start_ms}&limit={limit}"
    for attempt in range(4):
        try:
            r = session.get(url, timeout=8)
            used_weight = int(r.headers.get("x-mbx-used-weight-1m", 0))
            if used_weight > 1050:
                time.sleep(2.0)
            elif used_weight > 900:
                time.sleep(0.5)

            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 10))
                logger.warning(f"⚠️ 429 от Binance. Пауза {retry_after} сек...")
                time.sleep(retry_after)
            else:
                time.sleep(1.0)
        except Exception as e:
            time.sleep(1.0 * (attempt + 1))
    return []

def fetch_day_candles(symbol, day_dt):
    start_ms = int(day_dt.timestamp() * 1000)
    end_ms = int((day_dt + timedelta(days=1)).timestamp() * 1000)
    
    # Ограничение по END_DT
    max_end_ms = int(END_DT.timestamp() * 1000)
    end_ms = min(end_ms, max_end_ms)
    
    chunk_starts = []
    curr = start_ms
    while curr < end_ms:
        chunk_starts.append(curr)
        curr += 1000 * 1000  # 1000 секунд
        
    candles = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ts = {executor.submit(fetch_chunk, symbol, ts): ts for ts in chunk_starts}
        for future in as_completed(future_to_ts):
            data = future.result()
            for k in data:
                ot = k[0]
                h = float(k[2])
                l = float(k[3])
                c = float(k[4])
                candles[ot] = {
                    "mid": (h + l) / 2.0,
                    "last": c
                }
    return candles

def process_all_days():
    curr_day = START_DT
    daily_dfs = []
    
    total_days = (END_DT.date() - START_DT.date()).days + 1
    day_idx = 0
    
    drive_service = None
    try:
        drive_service = get_drive_service()
    except Exception as e:
        logger.warning(f"Не удалось подключить Google Drive API: {e}")

    while curr_day < END_DT:
        day_str = curr_day.strftime("%Y-%m-%d")
        day_idx += 1
        t0 = time.time()
        logger.info(f"[{day_idx}/{total_days}] Загрузка 1s макро-данных за {day_str}...")

        # Параллельная загрузка BTC и NEAR
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_btc = ex.submit(fetch_day_candles, "BTCUSDT", curr_day)
            f_near = ex.submit(fetch_day_candles, "NEARUSDT", curr_day)
            btc_candles = f_btc.result()
            near_candles = f_near.result()

        common_ts = sorted(set(btc_candles.keys()) & set(near_candles.keys()))
        if not common_ts:
            logger.warning(f"Нет общих временных меток за {day_str}")
            curr_day += timedelta(days=1)
            continue

        rows = []
        for ts in common_ts:
            btc = btc_candles[ts]
            near = near_candles[ts]
            rows.append({
                "timestamp": pd.to_datetime(ts, unit='ms', utc=True),
                "btc_bid": btc["last"] - 0.01,
                "btc_ask": btc["last"] + 0.01,
                "btc_mid": btc["mid"],
                "btc_last": btc["last"],
                "near_bid": near["last"] - 0.0005,
                "near_ask": near["last"] + 0.0005,
                "near_mid": near["mid"],
                "near_last": near["last"]
            })

        df_day = pd.DataFrame(rows)
        df_day = df_day.sort_values("timestamp").reset_index(drop=True)
        daily_dfs.append(df_day)
        
        elapsed = time.time() - t0
        logger.info(f"✅ {day_str}: получено {len(df_day):,} секундных срезов за {elapsed:.1f}с. (BTC: {df_day['btc_mid'].iloc[0]:.1f} -> {df_day['btc_mid'].iloc[-1]:.1f})")

        # Сохранение суточного архива для Google Диска
        out_name = f"binance_macro_{day_str}.csv.gz"
        out_path = os.path.join(CACHE_GDRIVE_DIR, out_name)
        os.makedirs(CACHE_GDRIVE_DIR, exist_ok=True)
        df_day.to_csv(out_path, index=False, compression="gzip")
        
        # Загрузка на Google Диск
        if drive_service and day_str in ["2026-10-10", "2026-10-11", "2026-10-12"]:
            try:
                folder_id = get_or_create_folder(drive_service, day_str, FOLDER_ID)
                upload_or_replace_file(drive_service, out_path, folder_id)
            except Exception as ed:
                logger.warning(f"Ошибка загрузки на Google Диск: {ed}")

        curr_day += timedelta(days=1)

    # Объединение всей ретроспективной истории
    if not daily_dfs:
        logger.error("Нет загруженных данных!")
        return

    df_hist = pd.concat(daily_dfs, ignore_index=True)
    
    # Мердж с существующим файлом (онлайн-сегмент с 11 по 12 октября)
    parquet_path = os.path.join(PROCESSED_DIR, "binance_macro.parquet")
    if os.path.exists(parquet_path):
        try:
            df_existing = pd.read_parquet(parquet_path)
            df_all = pd.concat([df_hist, df_existing], ignore_index=True)
            df_all = df_all.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        except Exception as ee:
            logger.warning(f"Ошибка слияния с существующим parquet: {ee}")
            df_all = df_hist
    else:
        df_all = df_hist

    # Сохраняем итоговые Parquet датасеты
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_all.to_parquet(parquet_path, index=False)
    
    macro_dir = os.path.join(PROCESSED_DIR, "macro")
    os.makedirs(macro_dir, exist_ok=True)
    bbo_path = os.path.join(macro_dir, "binance_bbo.parquet")
    df_all.to_parquet(bbo_path, index=False)
    
    logger.info(f"🎉 Полный макро-датасет сохранен:")
    logger.info(f"   Файлы: {parquet_path} и {bbo_path}")
    logger.info(f"   Всего строк: {len(df_all):,} | Диапазон: {df_all['timestamp'].min()} -> {df_all['timestamp'].max()}")

if __name__ == "__main__":
    process_all_days()
