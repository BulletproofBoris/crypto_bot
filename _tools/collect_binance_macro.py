#!/usr/bin/env python3
"""
Сборщик макро-поводырей с Binance WebSocket в реальном времени.
Слушает потоки bookTicker и trade для:
- BTC/USDT (мировой локомотив крипторынка)
- NEAR/USDT (поводырь экосистемы NEAR и Quantus)

Каждую 1.0 секунду атомарно фиксирует срез в CSV:
timestamp,btc_bid,btc_ask,btc_mid,btc_last,near_bid,near_ask,near_mid,near_last
"""

import asyncio
import json
import csv
import logging
import os
import time
from datetime import datetime
from curl_cffi.requests import AsyncSession

DATA_DIR = "/opt/crypto_bot/data/realtime/macro_binance"
CSV_FILE = os.path.join(DATA_DIR, "binance_macro.csv")
LOG_FILE = os.path.join(DATA_DIR, "binance_macro.log")

os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("binance_collector")

WS_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@bookTicker/nearusdt@bookTicker/btcusdt@trade/nearusdt@trade"

state = {
    "btc": {"bid": None, "ask": None, "last": None},
    "near": {"bid": None, "ask": None, "last": None}
}

def init_csv():
    exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp",
                "btc_bid", "btc_ask", "btc_mid", "btc_last",
                "near_bid", "near_ask", "near_mid", "near_last"
            ])

async def writer_loop():
    while True:
        try:
            # Синхронизация с началом каждой секунды
            now = time.time()
            sleep_time = max(0.05, 1.0 - (now % 1.0))
            await asyncio.sleep(sleep_time)

            ts = datetime.utcnow().isoformat() + "Z"

            btc = state["btc"]
            near = state["near"]

            if btc["bid"] is not None and near["bid"] is not None:
                btc_mid = (btc["bid"] + btc["ask"]) / 2.0 if btc["bid"] and btc["ask"] else ""
                near_mid = (near["bid"] + near["ask"]) / 2.0 if near["bid"] and near["ask"] else ""

                row = [
                    ts,
                    f"{btc['bid']:.2f}" if btc['bid'] else "",
                    f"{btc['ask']:.2f}" if btc['ask'] else "",
                    f"{btc_mid:.2f}" if btc_mid != "" else "",
                    f"{btc['last']:.2f}" if btc['last'] else "",
                    f"{near['bid']:.4f}" if near['bid'] else "",
                    f"{near['ask']:.4f}" if near['ask'] else "",
                    f"{near_mid:.4f}" if near_mid != "" else "",
                    f"{near['last']:.4f}" if near['last'] else ""
                ]

                with open(CSV_FILE, "a", newline="") as f:
                    csv.writer(f).writerow(row)

        except Exception as e:
            logger.error(f"Ошибка в writer_loop: {e}")
            await asyncio.sleep(1)

async def ws_loop():
    while True:
        try:
            logger.info("Подключение к Binance WebSocket...")
            async with AsyncSession() as session:
                async with session.ws_connect(WS_URL) as ws:
                    logger.info("✅ Подключено к Binance WebSocket (BTCUSDT + NEARUSDT)!")

                    while True:
                        msg, *_ = await ws.recv()
                        payload = json.loads(msg)
                        stream = payload.get("stream", "")
                        data = payload.get("data", {})

                        if "btcusdt@bookTicker" in stream:
                            state["btc"]["bid"] = float(data.get("b", 0))
                            state["btc"]["ask"] = float(data.get("a", 0))
                        elif "nearusdt@bookTicker" in stream:
                            state["near"]["bid"] = float(data.get("b", 0))
                            state["near"]["ask"] = float(data.get("a", 0))
                        elif "btcusdt@trade" in stream:
                            state["btc"]["last"] = float(data.get("p", 0))
                        elif "nearusdt@trade" in stream:
                            state["near"]["last"] = float(data.get("p", 0))

        except Exception as e:
            logger.error(f"Разрыв соединения с Binance WebSocket: {e}. Переподключение через 5 секунд...")
            await asyncio.sleep(5)

async def main():
    init_csv()
    await asyncio.gather(
        ws_loop(),
        writer_loop()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Сборщик Binance остановлен.")
