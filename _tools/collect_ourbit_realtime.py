#!/usr/bin/env python3
"""
Реалтайм-сборщик стакана (Top-50 + глубокая ликвидность до 200 уровней) и сделок PEARL/USDT с биржи Ourbit (Spot API v3).
Примечание:
- На Ourbit токен Pearl (отображаемый на бирже как PEARL/USDT) в системном API имеет символ PRLUSDT.
- В публичном эндпоинте /api/v3/trades поле "id" равно null, поэтому формируется детерминированный
  уникальный синтетический ключ "{time_ms}_{side}_{price}_{qty}_{occ_idx}" и O(1) буфер дедупликации.
"""

import asyncio
from collections import Counter, deque
import csv
import logging
from logging.handlers import RotatingFileHandler
import os
import socket
import time
from datetime import datetime, timezone
from curl_cffi import requests

# Очистка невалидного локального прокси при локальном запуске
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    if "127.0.0.1:2080" in os.environ.get(_var, ""):
        del os.environ[_var]


def notify_systemd(state: str = "WATCHDOG=1") -> None:
    """Отправляет сигнал сторожевому таймеру systemd (WatchdogSec)."""
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    try:
        if sock_path.startswith("@"):
            sock_path = "\0" + sock_path[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(sock_path)
            sock.sendall(state.encode())
    except Exception:
        pass


DEFAULT_BASE_DATA = (
    "/opt/crypto_bot/data/realtime"
    if os.path.exists("/opt/crypto_bot")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "realtime")
)
REALTIME_ROOT = os.environ.get("REALTIME_DATA_DIR", DEFAULT_BASE_DATA)

PAIRS = [
    {
        "symbol": "PRLUSDT",  # Внутренний тикер PEARL/USDT на Ourbit Spot API v3
        "display": "PEARL/USDT",
        "dir": os.path.join(REALTIME_ROOT, "prlusdt_ourbit"),
    }
]

DEPTH_LEVELS = 50
ORDERBOOK_FETCH_LIMIT = 200
TRADES_FETCH_LIMIT = 100
POLL_INTERVAL = 1.0  # секунды
SEEN_TRADES_CAP = 5000
LOG_FILE = os.path.join(REALTIME_ROOT, "collection_ourbit.log")

os.makedirs(REALTIME_ROOT, exist_ok=True)
for p in PAIRS:
    os.makedirs(p["dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=2),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ourbit_collector")

seen_trade_queues = {p["symbol"]: deque() for p in PAIRS}
seen_trade_sets = {p["symbol"]: set() for p in PAIRS}
last_trade_ms_watermark = {p["symbol"]: 0 for p in PAIRS}

DEPTH_TAIL_COLS = [
    "deep_bid_vol", "deep_ask_vol",
    "max_bid_wall_price", "max_bid_wall_qty",
    "max_ask_wall_price", "max_ask_wall_qty",
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


def _remember_trade_id(symbol: str, tid: str) -> None:
    s = seen_trade_sets[symbol]
    if tid in s:
        return
    q = seen_trade_queues[symbol]
    if len(q) >= SEEN_TRADES_CAP:
        old = q.popleft()
        s.discard(old)
    q.append(tid)
    s.add(tid)


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
        sym = p["symbol"]
        depth_file = os.path.join(p["dir"], "depth_ourbit.csv")
        trades_file = os.path.join(p["dir"], "trades_ourbit.csv")

        if not os.path.exists(depth_file) or os.path.getsize(depth_file) == 0:
            with open(depth_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(DEPTH_HEADERS)
        else:
            upgrade_csv_header_if_needed(depth_file, DEPTH_HEADERS)

        trades_exists = os.path.exists(trades_file) and os.path.getsize(trades_file) > 0
        if not trades_exists:
            with open(trades_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["id", "timestamp", "price", "amount", "side"])
        else:
            try:
                with open(trades_file, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(0, size - 32768))
                    lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                    for line in lines:
                        parts = line.split(",")
                        if not parts or parts[0] == "id":
                            continue
                        tid = parts[0].strip()
                        if tid:
                            _remember_trade_id(sym, tid)
                            ms_prefix = tid.split("_")[0]
                            if ms_prefix.isdigit():
                                last_trade_ms_watermark[sym] = max(last_trade_ms_watermark[sym], int(ms_prefix))
            except Exception:
                pass


def _extract_ourbit_side(t_obj: dict) -> str:
    # На Ourbit: isBuyerMaker=True (tradeType="ASK") => тейкер продал в бид ("sell");
    # isBuyerMaker=False (tradeType="BID") => тейкер купил с аска ("buy").
    if "isBuyerMaker" in t_obj and t_obj["isBuyerMaker"] is not None:
        return "sell" if bool(t_obj["isBuyerMaker"]) else "buy"
    tt = str(t_obj.get("tradeType", "")).upper()
    if tt == "ASK":
        return "sell"
    if tt == "BID":
        return "buy"
    return "buy"


async def fetch_pair(session: requests.AsyncSession, pair_info: dict) -> None:
    symbol = pair_info["symbol"]
    p_dir = pair_info["dir"]
    depth_file = os.path.join(p_dir, "depth_ourbit.csv")
    trades_file = os.path.join(p_dir, "trades_ourbit.csv")

    try:
        depth_url = f"https://api.ourbit.com/api/v3/depth?symbol={symbol}&limit={ORDERBOOK_FETCH_LIMIT}"
        trades_url = f"https://api.ourbit.com/api/v3/trades?symbol={symbol}&limit={TRADES_FETCH_LIMIT}"
        depth_resp, trades_resp = await asyncio.gather(
            session.get(depth_url, timeout=5),
            session.get(trades_url, timeout=5),
            return_exceptions=True,
        )

        # 1. Обработка стакана (bids по убыванию цены, asks по возрастанию цены: [[price_str, qty_str], ...])
        if isinstance(depth_resp, Exception):
            logger.warning(f"[{symbol}] Ошибка запроса стакана Ourbit: {depth_resp}")
        elif depth_resp.status_code != 200:
            logger.warning(f"[{symbol}] HTTP {depth_resp.status_code} стакана Ourbit: {depth_resp.text[:120]}")
        else:
            depth_data = depth_resp.json()
            bids = depth_data.get("bids") or []
            asks = depth_data.get("asks") or []

            if bids and asks:
                ts_ms = depth_data.get("timestamp")
                if ts_ms:
                    ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()
                else:
                    ts = datetime.now(timezone.utc).isoformat()

                row = [ts]
                for i in range(DEPTH_LEVELS):
                    if i < len(bids) and len(bids[i]) >= 2:
                        row.extend([bids[i][0], bids[i][1]])
                    else:
                        row.extend(["", ""])
                for i in range(DEPTH_LEVELS):
                    if i < len(asks) and len(asks[i]) >= 2:
                        row.extend([asks[i][0], asks[i][1]])
                    else:
                        row.extend(["", ""])

                deep_bid_vol = sum(float(b[1] or 0) for b in bids[DEPTH_LEVELS:] if len(b) >= 2)
                deep_ask_vol = sum(float(a[1] or 0) for a in asks[DEPTH_LEVELS:] if len(a) >= 2)
                max_bid_wall = max(bids, key=lambda b: float(b[1] or 0) if len(b) >= 2 else 0.0)
                max_ask_wall = max(asks, key=lambda a: float(a[1] or 0) if len(a) >= 2 else 0.0)
                row.extend([
                    f"{deep_bid_vol:.4f}",
                    f"{deep_ask_vol:.4f}",
                    max_bid_wall[0] if len(max_bid_wall) >= 2 else "",
                    max_bid_wall[1] if len(max_bid_wall) >= 2 else "",
                    max_ask_wall[0] if len(max_ask_wall) >= 2 else "",
                    max_ask_wall[1] if len(max_ask_wall) >= 2 else "",
                ])

                with open(depth_file, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)

        # 2. Обработка сделок с детерминированным уникальным синтетическим ID
        if isinstance(trades_resp, Exception):
            logger.debug(f"[{symbol}] Ошибка запроса сделок Ourbit: {trades_resp}")
        elif trades_resp.status_code == 200:
            trades_data = trades_resp.json()
            if isinstance(trades_data, list) and trades_data:
                # API отдает от новых к старым; разворачиваем в хронологический порядок с сохранением порядка внутри 1 мс
                chron_trades = list(reversed(trades_data))
                chron_trades.sort(key=lambda x: int(x.get("time", 0)))

                occ_counter = Counter()
                new_rows = []
                watermark = last_trade_ms_watermark[symbol]

                for t in chron_trades:
                    t_ms = int(t.get("time") or 0)
                    if t_ms <= 0:
                        continue
                    price = str(t.get("price", "")).strip()
                    qty = str(t.get("qty", "")).strip()
                    side = _extract_ourbit_side(t)

                    raw_id = t.get("id")
                    if raw_id is not None and str(raw_id).strip() != "":
                        tid = str(raw_id).strip()
                    else:
                        sig = (t_ms, side[0], price, qty)
                        occ_idx = occ_counter[sig]
                        occ_counter[sig] += 1
                        tid = f"{t_ms}_{side[0]}_{price}_{qty}_{occ_idx}"

                    # Отсекаем заведомо старые сделки глубже 5 секунд от watermark + проверяем O(1) set
                    if watermark > 0 and t_ms < watermark - 5000:
                        continue
                    if tid in seen_trade_sets[symbol]:
                        continue

                    ts_iso = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc).isoformat()
                    new_rows.append((tid, ts_iso, price, qty, side, t_ms))

                if new_rows:
                    with open(trades_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for tid, ts_iso, price, qty, side, t_ms in new_rows:
                            writer.writerow([tid, ts_iso, price, qty, side])
                            _remember_trade_id(symbol, tid)
                            if t_ms > last_trade_ms_watermark[symbol]:
                                last_trade_ms_watermark[symbol] = t_ms

    except Exception as e:
        logger.error(f"[{symbol}] Ошибка сбора Ourbit: {e}")


async def poll_data() -> None:
    init_csv_files()
    async with requests.AsyncSession(impersonate="chrome120") as session:
        logger.info(
            f"🚀 Запуск HTTP/2 Ourbit коллектора для пар: "
            f"{[p.get('display', p['symbol']) + ' (' + p['symbol'] + ')' for p in PAIRS]}..."
        )
        notify_systemd("READY=1")
        last_watchdog_ping = 0.0

        while True:
            start_time = time.time()
            await asyncio.gather(*(fetch_pair(session, p) for p in PAIRS), return_exceptions=True)

            now = time.time()
            if now - last_watchdog_ping >= 5.0:
                notify_systemd("WATCHDOG=1")
                last_watchdog_ping = now

            elapsed = time.time() - start_time
            await asyncio.sleep(max(0.05, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        asyncio.run(poll_data())
    except KeyboardInterrupt:
        logger.info("Ourbit коллектор остановлен пользователем")
