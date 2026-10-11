import asyncio
import json
import csv
import logging
import os
import time
from datetime import datetime
from curl_cffi.requests import AsyncSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MARKETS = [
    {
        "id": "prlusdt",
        "data_dir": "/opt/crypto_bot/data/realtime/prlusdt"
    },
    {
        "id": "quantususdt",
        "data_dir": "/opt/crypto_bot/data/realtime/quantususdt"
    },
    {
        "id": "tscusdt",
        "data_dir": "/opt/crypto_bot/data/realtime/tscusdt"
    }
]

WS_URL = "wss://safe.trade/api/v2/websocket/public"
PROXY = "http://edtilhmt:iblj7uuixsmy@31.59.20.176:6754"

# Создаем директории
for m in MARKETS:
    os.makedirs(m["data_dir"], exist_ok=True)

orderbooks = {m["id"]: {"bids": {}, "asks": {}} for m in MARKETS}

def write_depth(market_id, timestamp):
    m_info = next((m for m in MARKETS if m["id"] == market_id), None)
    if not m_info:
        return

    ob = orderbooks[market_id]
    sorted_bids = sorted(ob["bids"].items(), key=lambda x: float(x[0]), reverse=True)[:50]
    sorted_asks = sorted(ob["asks"].items(), key=lambda x: float(x[0]))[:50]

    file_path = os.path.join(m_info["data_dir"], "depth.csv")
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = ["timestamp"]
            for i in range(1, 51): headers.extend([f"bid_{i}_price", f"bid_{i}_qty"])
            for i in range(1, 51): headers.extend([f"ask_{i}_price", f"ask_{i}_qty"])
            writer.writerow(headers)

        row = [timestamp]
        for i in range(50):
            if i < len(sorted_bids):
                row.extend([sorted_bids[i][0], sorted_bids[i][1]])
            else:
                row.extend(["", ""])

        for i in range(50):
            if i < len(sorted_asks):
                row.extend([sorted_asks[i][0], sorted_asks[i][1]])
            else:
                row.extend(["", ""])
        writer.writerow(row)

async def fetch_snapshot(session, market_id):
    logger.info(f"📥 [{market_id}] Скачиваем начальный снимок стакана...")
    rest_url = f"https://safe.trade/api/v2/trade/public/markets/{market_id}/depth"
    try:
        proxies = {"http": PROXY, "https": PROXY}
        r = await session.get(rest_url, proxies=proxies, timeout=15)
        if r.status_code != 200:
            logger.error(f"[{market_id}] Snapshot HTTP {r.status_code}: {r.text[:100]}")
            return False

        data = r.json()
        orderbooks[market_id]["bids"] = {price: qty for price, qty in data.get("bids", [])}
        orderbooks[market_id]["asks"] = {price: qty for price, qty in data.get("asks", [])}
        logger.info(f"✅ [{market_id}] Снимок загружен: {len(orderbooks[market_id]['bids'])} bids, {len(orderbooks[market_id]['asks'])} asks.")
        return True
    except Exception as e:
        logger.error(f"⚠️ [{market_id}] Ошибка при скачивании снимка: {e}")
        return False

def update_orderbook(market_id, updates):
    ob = orderbooks[market_id]
    for price, qty in updates.get("bids", []):
        if float(qty) == 0:
            ob["bids"].pop(price, None)
        else:
            ob["bids"][price] = qty

    for price, qty in updates.get("asks", []):
        if float(qty) == 0:
            ob["asks"].pop(price, None)
        else:
            ob["asks"][price] = qty

async def main():
    market_ids = [m["id"] for m in MARKETS]
    while True:
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                for mid in market_ids:
                    success = await fetch_snapshot(session, mid)
                    if not success:
                        logger.warning(f"⚠️ [{mid}] Не удалось получить снимок стакана! Собираем с нуля.")

                logger.info(f"🚀 Запуск WebSocket сбора данных для {market_ids} с {WS_URL} (через прокси)")

                ws_proxies = {"http": PROXY, "https": PROXY}
                streams = []
                for mid in market_ids:
                    streams.extend([f"{mid}.depth", f"{mid}.trades"])

                async with session.ws_connect(WS_URL, proxies=ws_proxies) as ws:
                    subscribe_msg = {
                        "event": "subscribe",
                        "streams": streams
                    }
                    await ws.send(json.dumps(subscribe_msg).encode())

                    while True:
                        msg, *_ = await ws.recv()
                        data = json.loads(msg)

                        if data.get("event") == "ping":
                            pass

                        for k, stream_data in data.items():
                            if k.endswith(".depth"):
                                mid = k.split(".")[0]
                                if mid in orderbooks:
                                    update_orderbook(mid, stream_data)
                                    write_depth(mid, int(time.time() * 1000))
                            elif k.endswith(".trades"):
                                mid = k.split(".")[0]
                                m_info = next((m for m in MARKETS if m["id"] == mid), None)
                                if m_info:
                                    file_path = os.path.join(m_info["data_dir"], "trades.csv")
                                    file_exists = os.path.exists(file_path)
                                    with open(file_path, "a", newline="") as f:
                                        writer = csv.writer(f)
                                        if not file_exists:
                                            writer.writerow(["id", "price", "amount", "type", "timestamp"])
                                        for trade in stream_data:
                                            writer.writerow([
                                                trade.get("id", ""),
                                                trade.get("price", ""),
                                                trade.get("amount", ""),
                                                trade.get("type", trade.get("taker_type", "")),
                                                trade.get("date", trade.get("created_at", ""))
                                            ])

        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}. Переподключение через 5 секунд...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
