import asyncio
import csv
import logging
from logging.handlers import RotatingFileHandler
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
        RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=2),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bigone_collector")

PROXIES = None
last_trade_ids = {p["symbol"]: None for p in PAIRS}

DEPTH_TAIL_COLS = [
    "deep_bid_vol", "deep_ask_vol",
    "max_bid_wall_price", "max_bid_wall_qty",
    "max_ask_wall_price", "max_ask_wall_qty"
]

def build_depth_headers():
    headers = ["timestamp"]
    for i in range(1, DEPTH_LEVELS + 1):
        headers.extend([f"bid_{i}_price", f"bid_{i}_qty"])
    for i in range(1, DEPTH_LEVELS + 1):
        headers.extend([f"ask_{i}_price", f"ask_{i}_qty"])
    headers.extend(DEPTH_TAIL_COLS)
    return headers

DEPTH_HEADERS = build_depth_headers()


def upgrade_csv_header_if_needed(file_path, expected_headers):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
            if not first_line:
                return
            current_cols = first_line.split(",")
            if len(current_cols) >= len(expected_headers):
                return
            rest = f.read()
        tmp_path = file_path + ".hdr_tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(expected_headers) + "\n")
            f.write(rest)
        os.replace(tmp_path, file_path)
        logger.info(f"Обновлен заголовок {file_path}: {len(current_cols)} -> {len(expected_headers)} колонок")
    except Exception as e:
        logger.warning(f"Не удалось обновить заголовок {file_path}: {e}")


def init_csv_files():
    for p in PAIRS:
        depth_file = os.path.join(p["dir"], "depth_bigone.csv")
        trades_file = os.path.join(p["dir"], "trades_bigone.csv")

        depth_exists = os.path.exists(depth_file)
        if not depth_exists:
            with open(depth_file, "a", newline="") as f:
                csv.writer(f).writerow(DEPTH_HEADERS)
        else:
            upgrade_csv_header_if_needed(depth_file, DEPTH_HEADERS)

        trades_exists = os.path.exists(trades_file)
        with open(trades_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not trades_exists:
                writer.writerow(["id", "timestamp", "price", "amount", "side"])

        if trades_exists and os.path.getsize(trades_file) > 0:
            try:
                with open(trades_file, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(0, size - 2048))
                    lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                    if lines:
                        last_col = lines[-1].split(",")[0].strip()
                        if last_col.isdigit():
                            last_trade_ids[p["symbol"]] = int(last_col)
            except Exception:
                pass

async def fetch_pair(session, pair_info):
    symbol = pair_info["symbol"]
    p_dir = pair_info["dir"]
    depth_file = os.path.join(p_dir, "depth_bigone.csv")
    trades_file = os.path.join(p_dir, "trades_bigone.csv")

    try:
        depth_url = f"https://big.one/api/v3/asset_pairs/{symbol}/depth?limit=200"
        trades_url = f"https://big.one/api/v3/asset_pairs/{symbol}/trades?limit=50"
        depth_resp, trades_resp = await asyncio.gather(
            session.get(depth_url, timeout=5),
            session.get(trades_url, timeout=5),
            return_exceptions=True
        )

        # 1. Process Depth
        if not isinstance(depth_resp, Exception) and depth_resp.status_code == 200:
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

                deep_bid_vol = sum(float(b.get('quantity', 0) or 0) for b in bids[DEPTH_LEVELS:])
                deep_ask_vol = sum(float(a.get('quantity', 0) or 0) for a in asks[DEPTH_LEVELS:])
                max_bid_wall = max(bids, key=lambda b: float(b.get('quantity', 0) or 0))
                max_ask_wall = max(asks, key=lambda a: float(a.get('quantity', 0) or 0))
                row.extend([
                    f"{deep_bid_vol:.4f}",
                    f"{deep_ask_vol:.4f}",
                    max_bid_wall.get('price', ''),
                    max_bid_wall.get('quantity', ''),
                    max_ask_wall.get('price', ''),
                    max_ask_wall.get('quantity', '')
                ])

                with open(depth_file, "a", newline="") as f:
                    csv.writer(f).writerow(row)


        # 2. Process Trades
        if not isinstance(trades_resp, Exception) and trades_resp.status_code == 200:
            trades_data = trades_resp.json().get('data', [])
            if trades_data:
                trades_data.sort(key=lambda x: int(x['id']))
                last_id = last_trade_ids[symbol]
                new_trades = [t for t in trades_data if last_id is None or int(t['id']) > last_id]

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
    session = requests.AsyncSession(impersonate="chrome120", proxies=PROXIES)
    logger.info(f"Запуск параллельного HTTP/2 BigONE polling для пар: {[p['symbol'] for p in PAIRS]}...")

    while True:
        start_time = time.time()
        await asyncio.gather(*(fetch_pair(session, p) for p in PAIRS), return_exceptions=True)

        elapsed = time.time() - start_time
        sleep_time = max(0.05, POLL_INTERVAL - elapsed)
        await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    init_csv_files()
    try:
        asyncio.run(poll_data())
    except KeyboardInterrupt:
        logger.info("BigONE polling остановлен пользователем")
