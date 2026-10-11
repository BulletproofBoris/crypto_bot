import asyncio
import csv
import logging
import os
import time
from datetime import datetime
import traceback
from curl_cffi import requests

PAIRS = [
    {
        "symbol": "PRL-USDT",
        "dir": "/opt/crypto_bot/data/realtime/prlusdt_bigone"
    },
    {
        "symbol": "QTC-USDT",
        "dir": "/opt/crypto_bot/data/realtime/quantus_bigone"
    }
]

DEPTH_LEVELS = 50
POLL_INTERVAL = 1.0  # секунды
LOG_FILE = "/opt/crypto_bot/data/realtime/collection_bigone.log"

for p in PAIRS:
    os.makedirs(p["dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bigone_collector")

PROXIES = None

def init_csv_files():
    for p in PAIRS:
        depth_file = os.path.join(p["dir"], "depth_bigone.csv")
        trades_file = os.path.join(p["dir"], "trades_bigone.csv")

        depth_exists = os.path.exists(depth_file)
        with open(depth_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not depth_exists:
                headers = ["timestamp"]
                for i in range(1, DEPTH_LEVELS + 1):
                    headers.extend([f"bid_{i}_price", f"bid_{i}_qty"])
                for i in range(1, DEPTH_LEVELS + 1):
                    headers.extend([f"ask_{i}_price", f"ask_{i}_qty"])
                writer.writerow(headers)

        trades_exists = os.path.exists(trades_file)
        with open(trades_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not trades_exists:
                writer.writerow(["id", "timestamp", "price", "amount", "side"])

last_trade_ids = {p["symbol"]: None for p in PAIRS}

async def fetch_pair(session, pair_info):
    symbol = pair_info["symbol"]
    p_dir = pair_info["dir"]
    depth_file = os.path.join(p_dir, "depth_bigone.csv")
    trades_file = os.path.join(p_dir, "trades_bigone.csv")

    try:
        # 1. Fetch Depth
        depth_resp = await session.get(f"https://big.one/api/v3/asset_pairs/{symbol}/depth?limit={DEPTH_LEVELS}", timeout=5)
        if depth_resp.status_code == 200:
            depth_data = depth_resp.json().get('data', {})
            bids = depth_data.get('bids', [])
            asks = depth_data.get('asks', [])

            if bids and asks:
                ts = datetime.utcnow().isoformat() + "Z"
                row = [ts]
                for i in range(DEPTH_LEVELS):
                    if i < len(bids):
                        row.extend([bids[i]['price'], bids[i]['quantity']])
                    else:
                        row.extend(["", ""])
                for i in range(DEPTH_LEVELS):
                    if i < len(asks):
                        row.extend([asks[i]['price'], asks[i]['quantity']])
                    else:
                        row.extend(["", ""])

                with open(depth_file, "a", newline="") as f:
                    csv.writer(f).writerow(row)

        # 2. Fetch Trades
        trades_resp = await session.get(f"https://big.one/api/v3/asset_pairs/{symbol}/trades?limit=50", timeout=5)
        if trades_resp.status_code == 200:
            trades_data = trades_resp.json().get('data', [])
            if trades_data:
                trades_data.sort(key=lambda x: int(x['id']))
                last_id = last_trade_ids[symbol]
                new_trades = []
                for t in trades_data:
                    t_id = int(t['id'])
                    if last_id is None or t_id > last_id:
                        new_trades.append(t)

                if new_trades:
                    with open(trades_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        for t in new_trades:
                            writer.writerow([
                                t['id'],
                                t['created_at'],
                                t['price'],
                                t['amount'],
                                t['taker_side']
                            ])
                    last_trade_ids[symbol] = int(new_trades[-1]['id'])

    except Exception as e:
        logger.error(f"[{symbol}] Ошибка получения данных: {e}")

async def poll_data():
    session = requests.AsyncSession(impersonate="chrome110", proxies=PROXIES)
    logger.info(f"Запуск BigONE polling для пар: {[p['symbol'] for p in PAIRS]}...")

    while True:
        start_time = time.time()
        for p in PAIRS:
            await fetch_pair(session, p)

        elapsed = time.time() - start_time
        sleep_time = max(0.1, POLL_INTERVAL - elapsed)
        await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    init_csv_files()
    try:
        asyncio.run(poll_data())
    except KeyboardInterrupt:
        logger.info("BigONE polling остановлен пользователем")
