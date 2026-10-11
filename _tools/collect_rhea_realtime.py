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
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rhea_collector")

def init_csv():
    exists = os.path.exists(PRICE_FILE)
    with open(PRICE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp", "price_usd", "price_native", "liquidity_usd",
                "base_reserve", "quote_reserve", "volume_5m", "buys_5m", "sells_5m"
            ])

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
                    vol_5m = data.get('volume', {}).get('m5', '')
                    txns_5m = data.get('txns', {}).get('m5', {})
                    buys_5m = txns_5m.get('buys', '')
                    sells_5m = txns_5m.get('sells', '')

                    row = [
                        ts, price_usd, price_native, liq_usd,
                        base_res, quote_res, vol_5m, buys_5m, sells_5m
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
