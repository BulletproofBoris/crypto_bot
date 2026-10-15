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

MACRO_HEADERS = [
    "timestamp",
    "btc_bid", "btc_ask", "btc_mid", "btc_last",
    "near_bid", "near_ask", "near_mid", "near_last",
    "btc_bid_qty", "btc_ask_qty", "btc_buy_vol", "btc_sell_vol",
    "near_bid_qty", "near_ask_qty", "near_buy_vol", "near_sell_vol"
]

state = {
    "btc": {"bid": None, "ask": None, "bid_qty": 0.0, "ask_qty": 0.0, "last": None, "buy_vol": 0.0, "sell_vol": 0.0},
    "near": {"bid": None, "ask": None, "bid_qty": 0.0, "ask_qty": 0.0, "last": None, "buy_vol": 0.0, "sell_vol": 0.0}
}

def init_csv():
    exists = os.path.exists(CSV_FILE)
    if not exists or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(MACRO_HEADERS)
    else:
        try:
            with open(CSV_FILE, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                if first_line and len(first_line.split(",")) < len(MACRO_HEADERS):
                    rest = f.read()
                    tmp_path = CSV_FILE + ".hdr_tmp"
                    with open(tmp_path, "w", encoding="utf-8", newline="") as out_f:
                        out_f.write(",".join(MACRO_HEADERS) + "\n")
                        out_f.write(rest)
                    os.replace(tmp_path, CSV_FILE)
                    logger.info(f"Обновлен заголовок {CSV_FILE} до {len(MACRO_HEADERS)} колонок")
        except Exception as e:
            logger.warning(f"Не удалось обновить заголовок {CSV_FILE}: {e}")

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

                btc_buy, btc_sell = btc["buy_vol"], btc["sell_vol"]
                near_buy, near_sell = near["buy_vol"], near["sell_vol"]
                btc["buy_vol"] = 0.0
                btc["sell_vol"] = 0.0
                near["buy_vol"] = 0.0
                near["sell_vol"] = 0.0

                row = [
                    ts,
                    f"{btc['bid']:.2f}" if btc['bid'] else "",
                    f"{btc['ask']:.2f}" if btc['ask'] else "",
                    f"{btc_mid:.2f}" if btc_mid != "" else "",
                    f"{btc['last']:.2f}" if btc['last'] else "",
                    f"{near['bid']:.4f}" if near['bid'] else "",
                    f"{near['ask']:.4f}" if near['ask'] else "",
                    f"{near_mid:.4f}" if near_mid != "" else "",
                    f"{near['last']:.4f}" if near['last'] else "",
                    f"{btc['bid_qty']:.5f}",
                    f"{btc['ask_qty']:.5f}",
                    f"{btc_buy:.5f}",
                    f"{btc_sell:.5f}",
                    f"{near['bid_qty']:.2f}",
                    f"{near['ask_qty']:.2f}",
                    f"{near_buy:.2f}",
                    f"{near_sell:.2f}"
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
                            state["btc"]["bid_qty"] = float(data.get("B", 0))
                            state["btc"]["ask_qty"] = float(data.get("A", 0))
                        elif "nearusdt@bookTicker" in stream:
                            state["near"]["bid"] = float(data.get("b", 0))
                            state["near"]["ask"] = float(data.get("a", 0))
                            state["near"]["bid_qty"] = float(data.get("B", 0))
                            state["near"]["ask_qty"] = float(data.get("A", 0))
                        elif "btcusdt@trade" in stream:
                            state["btc"]["last"] = float(data.get("p", 0))
                            qty = float(data.get("q", 0))
                            if data.get("m", False):
                                state["btc"]["sell_vol"] += qty
                            else:
                                state["btc"]["buy_vol"] += qty
                        elif "nearusdt@trade" in stream:
                            state["near"]["last"] = float(data.get("p", 0))
                            qty = float(data.get("q", 0))
                            if data.get("m", False):
                                state["near"]["sell_vol"] += qty
                            else:
                                state["near"]["buy_vol"] += qty

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
