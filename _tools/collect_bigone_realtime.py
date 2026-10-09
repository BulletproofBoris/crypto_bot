import asyncio
import csv
import logging
import os
import time
from datetime import datetime
import traceback

from curl_cffi import requests

# Конфигурация
SYMBOL = "PRL-USDT"
DATA_DIR = "/opt/crypto_bot/data/realtime/prlusdt_bigone"
DEPTH_FILE = os.path.join(DATA_DIR, "depth_bigone.csv")
TRADES_FILE = os.path.join(DATA_DIR, "trades_bigone.csv")
LOG_FILE = os.path.join(DATA_DIR, "collection_bigone.log")

DEPTH_LEVELS = 50
POLL_INTERVAL = 1.0  # seconds

# Настройка логирования
os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Прокси (если нужен, берем из окружения или хардкод)
PROXIES = None

def init_csv_files():
    # Инициализация файла стакана
    depth_exists = os.path.exists(DEPTH_FILE)
    with open(DEPTH_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not depth_exists:
            headers = ["timestamp"]
            for i in range(1, DEPTH_LEVELS + 1):
                headers.extend([f"bid_{i}_price", f"bid_{i}_qty"])
            for i in range(1, DEPTH_LEVELS + 1):
                headers.extend([f"ask_{i}_price", f"ask_{i}_qty"])
            writer.writerow(headers)

    # Инициализация файла сделок
    trades_exists = os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not trades_exists:
            writer.writerow(["id", "timestamp", "price", "amount", "side"])

last_trade_id = None

async def poll_data():
    global last_trade_id
    
    session = requests.AsyncSession(impersonate="chrome110", proxies=PROXIES)
    
    while True:
        try:
            start_time = time.time()
            
            # Fetch Depth
            depth_resp = await session.get(f"https://big.one/api/v3/asset_pairs/{SYMBOL}/depth?limit={DEPTH_LEVELS}", timeout=5)
            if depth_resp.status_code == 200:
                depth_data = depth_resp.json().get('data', {})
                bids = depth_data.get('bids', [])
                asks = depth_data.get('asks', [])
                
                if bids and asks:
                    ts = datetime.utcnow().isoformat() + "Z"
                    row = [ts]
                    
                    # Заполняем bids
                    for i in range(DEPTH_LEVELS):
                        if i < len(bids):
                            row.extend([bids[i]['price'], bids[i]['quantity']])
                        else:
                            row.extend(["", ""])
                            
                    # Заполняем asks
                    for i in range(DEPTH_LEVELS):
                        if i < len(asks):
                            row.extend([asks[i]['price'], asks[i]['quantity']])
                        else:
                            row.extend(["", ""])
                            
                    with open(DEPTH_FILE, "a", newline="") as f:
                        csv.writer(f).writerow(row)
            
            # Fetch Trades
            trades_resp = await session.get(f"https://big.one/api/v3/asset_pairs/{SYMBOL}/trades?limit=50", timeout=5)
            if trades_resp.status_code == 200:
                trades_data = trades_resp.json().get('data', [])
                
                if trades_data:
                    # Trades are returned descending (newest first). Let's sort them ascending to write in chronological order.
                    trades_data.sort(key=lambda x: x['id'])
                    
                    new_trades = []
                    for t in trades_data:
                        t_id = int(t['id'])
                        if last_trade_id is None or t_id > last_trade_id:
                            new_trades.append(t)
                    
                    if new_trades:
                        with open(TRADES_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            for t in new_trades:
                                writer.writerow([
                                    t['id'],
                                    t['created_at'],
                                    t['price'],
                                    t['amount'],
                                    t['taker_side']
                                ])
                        last_trade_id = int(new_trades[-1]['id'])

            # Sleep to maintain 1 sec interval
            elapsed = time.time() - start_time
            sleep_time = max(0.1, POLL_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)

        except Exception as e:
            logging.error(f"Error fetching data: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(2) # Backoff on error

if __name__ == "__main__":
    init_csv_files()
    logging.info(f"Starting BigONE polling for {SYMBOL}...")
    try:
        asyncio.run(poll_data())
    except KeyboardInterrupt:
        logging.info("Polling stopped by user")
