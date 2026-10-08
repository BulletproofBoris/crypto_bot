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

MARKET = "prlusdt"
REST_URL = f"https://safe.trade/api/v2/trade/public/markets/{MARKET}/depth"
WS_URL = "wss://safe.trade/api/v2/websocket/public"
DATA_DIR = f"/opt/crypto_bot/data/realtime/{MARKET}"

PROXY = "http://edtilhmt:iblj7uuixsmy@31.59.20.176:6754"

os.makedirs(DATA_DIR, exist_ok=True)

orderbook = {"bids": {}, "asks": {}}

def write_depth(timestamp):
    sorted_bids = sorted(orderbook["bids"].items(), key=lambda x: float(x[0]), reverse=True)[:50]
    sorted_asks = sorted(orderbook["asks"].items(), key=lambda x: float(x[0]))[:50]
    
    file_path = os.path.join(DATA_DIR, "depth.csv")
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

async def fetch_snapshot(session):
    logger.info("📥 Скачиваем начальный снимок стакана...")
    try:
        proxies = {"http": PROXY, "https": PROXY}
        r = await session.get(REST_URL, proxies=proxies, timeout=15)
        if r.status_code != 200:
            logger.error(f"Snapshot HTTP {r.status_code}: {r.text[:100]}")
            return False
            
        data = r.json()
        orderbook["bids"] = {price: qty for price, qty in data.get("bids", [])}
        orderbook["asks"] = {price: qty for price, qty in data.get("asks", [])}
        logger.info(f"✅ Снимок загружен: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks.")
        return True
    except Exception as e:
        logger.error(f"⚠️ Ошибка при скачивании снимка: {e}")
        return False

def update_orderbook(updates):
    for price, qty in updates.get("bids", []):
        if float(qty) == 0:
            orderbook["bids"].pop(price, None)
        else:
            orderbook["bids"][price] = qty

    for price, qty in updates.get("asks", []):
        if float(qty) == 0:
            orderbook["asks"].pop(price, None)
        else:
            orderbook["asks"][price] = qty

async def main():
    while True:
        try:
            async with AsyncSession(impersonate="chrome110") as session:
                success = await fetch_snapshot(session)
                if not success:
                    logger.warning("⚠️ Не удалось получить снимок стакана! Собираем с нуля.")
                
                logger.info(f"🚀 Запуск WebSocket сбора данных для {MARKET.upper()} с {WS_URL} (через прокси)")
                
                # curl_cffi doesn't directly take `proxies` dict in ws_connect like it does in get(), wait, let's verify. 
                # According to curl_cffi docs, ws_connect takes `proxies={"http": ...}` just like get, or we can just pass `proxy`.
                ws_proxies = {"http": PROXY, "https": PROXY}
                
                async with session.ws_connect(WS_URL, proxies=ws_proxies) as ws:
                    subscribe_msg = {
                        "event": "subscribe",
                        "streams": [f"{MARKET}.depth", f"{MARKET}.trades"]
                    }
                    await ws.send(json.dumps(subscribe_msg).encode())
                    
                    while True:
                        msg, *_ = await ws.recv()
                        data = json.loads(msg)
                        
                        if data.get("event") == "ping":
                            # No standard ping in curl_cffi WS yet, but server might send string ping
                            pass 
                            
                        for k, stream_data in data.items():
                            if k == f"{MARKET}.depth":
                                update_orderbook(stream_data)
                                write_depth(int(time.time() * 1000))
                            elif k == f"{MARKET}.trades":
                                file_path = os.path.join(DATA_DIR, "trades.csv")
                                file_exists = os.path.exists(file_path)
                                with open(file_path, "a", newline="") as f:
                                    writer = csv.writer(f)
                                    if not file_exists:
                                        writer.writerow(["id", "price", "amount", "type", "timestamp"])
                                    for trade in stream_data:
                                        writer.writerow([trade.get("id", ""), trade.get("price", ""), trade.get("amount", ""), trade.get("type", trade.get("taker_type", "")), trade.get("date", trade.get("created_at", ""))])
                                        
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}. Переподключение через 5 секунд...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
