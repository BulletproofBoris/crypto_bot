#!/usr/bin/env python3
"""
Сборщик рыночных данных Quantus (QTC) с Rhea Finance / Near Intents (DEX).
Периодически опрашивает ончейн-пул QTC/USDC (контракт qtc.omft.near на v2.ref-finance.near).
Сохраняет:
- timestamp (UTC ISO)
- price_usd
- price_native
- liquidity_usd
- base_reserve (QTC)
- quote_reserve (USDC)
- volume_5m
- buys_5m
- sells_5m
"""

import asyncio
import csv
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from datetime import datetime
import traceback
from curl_cffi import requests

DATA_DIR = "/opt/crypto_bot/data/realtime/quantus_rhea"
PRICE_FILE = os.path.join(DATA_DIR, "price_rhea.csv")
LOG_FILE = os.path.join(DATA_DIR, "collection_rhea.log")

PAIR_ID = "refv2-17208628f84f5d6ad33f0da3bbbeb27ffcb398eac501a31bd6ad2011e36133a1:qtc.omft.near:2000"
API_URL = f"https://api.dexscreener.com/latest/dex/pairs/near/{PAIR_ID}"
POLL_INTERVAL = 1.0  # секунды (соответствует времени блока NEAR ~1.0-1.2 сек)

os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=2),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rhea_collector")

RHEA_HEADERS = [
    "timestamp", "price_usd", "price_native", "liquidity_usd",
    "base_reserve", "quote_reserve", "volume_5m", "buys_5m", "sells_5m",
    "volume_1h", "volume_24h", "buys_1h", "sells_1h", "buys_24h", "sells_24h",
    "price_change_5m", "price_change_1h", "price_change_24h"
]

def init_csv():
    exists = os.path.exists(PRICE_FILE)
    if not exists or os.path.getsize(PRICE_FILE) == 0:
        with open(PRICE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(RHEA_HEADERS)
    else:
        try:
            with open(PRICE_FILE, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                if first_line and len(first_line.split(",")) < len(RHEA_HEADERS):
                    rest = f.read()
                    tmp_path = PRICE_FILE + ".hdr_tmp"
                    with open(tmp_path, "w", encoding="utf-8", newline="") as out_f:
                        out_f.write(",".join(RHEA_HEADERS) + "\n")
                        out_f.write(rest)
                    os.replace(tmp_path, PRICE_FILE)
                    logger.info(f"Обновлен заголовок {PRICE_FILE} до {len(RHEA_HEADERS)} колонок")
        except Exception as e:
            logger.warning(f"Не удалось обновить заголовок {PRICE_FILE}: {e}")

async def poll_rhea():
    session = requests.AsyncSession(impersonate="chrome110")
    logger.info(f"Запуск опроса Rhea Finance / Near Intents для QTC ({PAIR_ID})...")

    while True:
        start_t = time.time()
        try:
            resp = await session.get(API_URL, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get('pair', {})
                if data:
                    ts = datetime.utcnow().isoformat() + "Z"
                    price_usd = data.get('priceUsd', '')
                    price_native = data.get('priceNative', '')
                    liq = data.get('liquidity', {})
                    liq_usd = liq.get('usd', '')
                    base_res = liq.get('base', '')
                    quote_res = liq.get('quote', '')
                    vol = data.get('volume', {})
                    vol_5m = vol.get('m5', '')
                    vol_1h = vol.get('h1', '')
                    vol_24h = vol.get('h24', '')
                    txns = data.get('txns', {})
                    txns_5m = txns.get('m5', {})
                    buys_5m = txns_5m.get('buys', '')
                    sells_5m = txns_5m.get('sells', '')
                    txns_1h = txns.get('h1', {})
                    buys_1h = txns_1h.get('buys', '')
                    sells_1h = txns_1h.get('sells', '')
                    txns_24h = txns.get('h24', {})
                    buys_24h = txns_24h.get('buys', '')
                    sells_24h = txns_24h.get('sells', '')
                    p_chg = data.get('priceChange', {})
                    chg_5m = p_chg.get('m5', '')
                    chg_1h = p_chg.get('h1', '')
                    chg_24h = p_chg.get('h24', '')

                    row = [
                        ts, price_usd, price_native, liq_usd,
                        base_res, quote_res, vol_5m, buys_5m, sells_5m,
                        vol_1h, vol_24h, buys_1h, sells_1h, buys_24h, sells_24h,
                        chg_5m, chg_1h, chg_24h
                    ]

                    with open(PRICE_FILE, "a", newline="") as f:
                        csv.writer(f).writerow(row)
            else:
                logger.warning(f"Rhea API HTTP {resp.status_code}")

        except Exception as e:
            logger.error(f"Ошибка опроса Rhea Finance: {e}")
            await asyncio.sleep(2)

        elapsed = time.time() - start_t
        sleep_time = max(0.2, POLL_INTERVAL - elapsed)
        await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    init_csv()
    try:
        asyncio.run(poll_rhea())
    except KeyboardInterrupt:
        logger.info("Rhea collector остановлен пользователем.")
