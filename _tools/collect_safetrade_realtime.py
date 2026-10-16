import asyncio
from collections import OrderedDict
from datetime import datetime
import json
import csv
import logging
import os
import socket
import time
from curl_cffi.requests import AsyncSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def notify_systemd(state="WATCHDOG=1"):
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

MARKETS = [
    {
        "id": "prlusdt",
        "data_dir": "/opt/crypto_bot/data/realtime/prlusdt"
    },
    {
        "id": "quantususdt",
        "data_dir": "/opt/crypto_bot/data/realtime/quantususdt"
    }
]

WS_URL = "wss://safe.trade/api/v2/websocket/public"
try:
    from _tools.vps_config import FALLBACK_PROXY
except ImportError:
    try:
        from vps_config import FALLBACK_PROXY
    except ImportError:
        FALLBACK_PROXY = os.environ.get("SAFETRADE_PROXY", "")

# Периодичность фоновой ре-синхронизации стакана через REST (сек) и heartbeat-записи (сек)
SNAPSHOT_RESYNC_INTERVAL = 900.0
HEARTBEAT_WRITE_INTERVAL = 10.0

for m in MARKETS:
    os.makedirs(m["data_dir"], exist_ok=True)

# Внутреннее состояние стаканов: ключи — float(price), значения — (price_str, qty_str)
orderbooks = {m["id"]: {"bids": {}, "asks": {}} for m in MARKETS}
snapshot_ready = {m["id"]: False for m in MARKETS}
last_write_time = {m["id"]: 0.0 for m in MARKETS}
last_sequence = {m["id"]: None for m in MARKETS}
seen_trade_ids = {m["id"]: OrderedDict() for m in MARKETS}
last_ws_msg_time = 0.0
last_market_depth_time = {m["id"]: 0.0 for m in MARKETS}


DEPTH_TAIL_COLS = [
    "deep_bid_vol", "deep_ask_vol",
    "max_bid_wall_price", "max_bid_wall_qty",
    "max_ask_wall_price", "max_ask_wall_qty"
]

def build_depth_headers():
    headers = ["timestamp"]
    for i in range(1, 51):
        headers.extend([f"bid_{i}_price", f"bid_{i}_qty"])
    for i in range(1, 51):
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


def remember_trade_id(mid, tid):
    if not tid:
        return
    seen_trade_ids[mid][tid] = None
    while len(seen_trade_ids[mid]) > 2000:
        seen_trade_ids[mid].popitem(last=False)


def init_seen_trades():
    for m in MARKETS:
        mid = m["id"]
        d_path = os.path.join(m["data_dir"], "depth.csv")
        upgrade_csv_header_if_needed(d_path, DEPTH_HEADERS)
        t_path = os.path.join(m["data_dir"], "trades.csv")
        if os.path.exists(t_path):
            try:
                with open(t_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()[-500:]
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if parts and parts[0]:
                        remember_trade_id(mid, str(parts[0]))
            except Exception as e:
                logger.warning(f"[{mid}] Не удалось считать историю сделок: {e}")


def parse_float(val, default=0.0):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def write_depth(market_id, timestamp_ms):
    if not snapshot_ready.get(market_id, False):
        return

    m_info = next((m for m in MARKETS if m["id"] == market_id), None)
    if not m_info:
        return

    ob = orderbooks[market_id]
    if not ob["bids"] or not ob["asks"]:
        return

    all_bids = sorted(ob["bids"].items(), key=lambda x: x[0], reverse=True)
    all_asks = sorted(ob["asks"].items(), key=lambda x: x[0])
    sorted_bids = all_bids[:50]
    sorted_asks = all_asks[:50]

    # Защита от записи скрещенного стакана
    best_bid_p = sorted_bids[0][0]
    best_ask_p = sorted_asks[0][0]
    if best_ask_p <= best_bid_p:
        logger.warning(f"⚠️ [{market_id}] Пропуск записи скрещенного стакана (bid={best_bid_p} >= ask={best_ask_p})")
        return

    deep_bid_vol = sum(parse_float(x[1][1]) for x in all_bids[50:])
    deep_ask_vol = sum(parse_float(x[1][1]) for x in all_asks[50:])
    max_bid_wall = max(all_bids, key=lambda x: parse_float(x[1][1]))
    max_ask_wall = max(all_asks, key=lambda x: parse_float(x[1][1]))

    file_path = os.path.join(m_info["data_dir"], "depth.csv")
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(DEPTH_HEADERS)

        row = [timestamp_ms]
        for i in range(50):
            if i < len(sorted_bids):
                p_str, q_str = sorted_bids[i][1]
                row.extend([p_str, q_str])
            else:
                row.extend(["", ""])

        for i in range(50):
            if i < len(sorted_asks):
                p_str, q_str = sorted_asks[i][1]
                row.extend([p_str, q_str])
            else:
                row.extend(["", ""])

        row.extend([
            f"{deep_bid_vol:.4f}",
            f"{deep_ask_vol:.4f}",
            max_bid_wall[1][0],
            max_bid_wall[1][1],
            max_ask_wall[1][0],
            max_ask_wall[1][1]
        ])
        writer.writerow(row)

    last_write_time[market_id] = time.time()


async def fetch_snapshot(session, market_id):
    logger.info(f"📥 [{market_id}] Скачиваем снимок стакана...")
    rest_url = f"https://safe.trade/api/v2/trade/public/markets/{market_id}/depth?limit=300"

    # Сначала пробуем прямое подключение (работает за <100мс с VPS), при сбое — резервный прокси
    for attempt_name, proxies in [("DIRECT", None), ("PROXY", {"http": FALLBACK_PROXY, "https": FALLBACK_PROXY})]:
        try:
            r = await session.get(rest_url, proxies=proxies, timeout=8)
            if r.status_code == 200:
                data = r.json()
                new_bids = {}
                for price, qty in data.get("bids", []):
                    p_f = parse_float(price)
                    q_f = parse_float(qty)
                    if p_f > 0 and q_f > 0:
                        new_bids[p_f] = (str(price), str(qty))

                new_asks = {}
                for price, qty in data.get("asks", []):
                    p_f = parse_float(price)
                    q_f = parse_float(qty)
                    if p_f > 0 and q_f > 0:
                        new_asks[p_f] = (str(price), str(qty))

                if new_bids and new_asks:
                    # Очищаем возможное пересечение на границе спреда
                    max_bid = max(new_bids.keys())
                    new_asks = {k: v for k, v in new_asks.items() if k > max_bid}

                if new_bids and new_asks:
                    orderbooks[market_id]["bids"] = new_bids
                    orderbooks[market_id]["asks"] = new_asks
                    snapshot_ready[market_id] = True
                    last_market_depth_time[market_id] = time.time()
                    logger.info(
                        f"✅ [{market_id}] Снимок загружен ({attempt_name}): "
                        f"{len(new_bids)} bids, {len(new_asks)} asks."
                    )
                    write_depth(market_id, int(time.time() * 1000))
                    return True
            else:
                logger.warning(f"[{market_id}] Snapshot ({attempt_name}) HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e:
            logger.warning(f"⚠️ [{market_id}] Ошибка снимка ({attempt_name}): {e}")

    return False


def update_orderbook(market_id, updates):
    ob = orderbooks[market_id]

    for item in updates.get("bids", []):
        if len(item) < 2:
            continue
        price, qty = item[0], item[1]
        p_f = parse_float(price)
        if p_f <= 0:
            continue
        q_f = parse_float(qty)
        if q_f <= 0:
            ob["bids"].pop(p_f, None)
        else:
            ob["bids"][p_f] = (str(price), str(qty))
            # Вытесняем устаревшие asks, которые оказались <= нового bid
            crossed_asks = [ak for ak in ob["asks"] if ak <= p_f]
            for ak in crossed_asks:
                ob["asks"].pop(ak, None)

    for item in updates.get("asks", []):
        if len(item) < 2:
            continue
        price, qty = item[0], item[1]
        p_f = parse_float(price)
        if p_f <= 0:
            continue
        q_f = parse_float(qty)
        if q_f <= 0:
            ob["asks"].pop(p_f, None)
        else:
            ob["asks"][p_f] = (str(price), str(qty))
            # Вытесняем устаревшие bids, которые оказались >= нового ask
            crossed_bids = [bk for bk in ob["bids"] if bk >= p_f]
            for bk in crossed_bids:
                ob["bids"].pop(bk, None)


def infer_trade_side(market_id, price_str, raw_side):
    if raw_side and str(raw_side).strip().lower() not in ("", "nan", "none"):
        return str(raw_side).strip().lower()
    p_f = parse_float(price_str)
    ob = orderbooks.get(market_id, {})
    bids = ob.get("bids", {})
    asks = ob.get("asks", {})
    if p_f > 0 and bids and asks:
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid = (best_bid + best_ask) / 2.0
        return "buy" if p_f >= mid else "sell"
    return ""


def append_trades_to_csv(mid, trade_list):
    m_info = next((m for m in MARKETS if m["id"] == mid), None)
    if not m_info or not trade_list:
        return
    file_path = os.path.join(m_info["data_dir"], "trades.csv")
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "price", "amount", "type", "timestamp"])
        for trade in trade_list:
            tid = str(trade.get("id", ""))
            if tid and tid in seen_trade_ids[mid]:
                continue
            remember_trade_id(mid, tid)
            raw_side = (
                trade.get("side")
                or trade.get("type")
                or trade.get("taker_type")
                or ""
            )
            clean_side = infer_trade_side(mid, trade.get("price"), raw_side)
            raw_ts = trade.get("date", trade.get("created_at", ""))
            if isinstance(raw_ts, str) and "T" in raw_ts:
                try:
                    raw_ts = int(datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp())
                except Exception:
                    pass
            writer.writerow([
                tid,
                trade.get("price", ""),
                trade.get("amount", ""),
                clean_side,
                raw_ts
            ])


async def fetch_recent_trades(session, market_id):
    rest_url = f"https://safe.trade/api/v2/trade/public/markets/{market_id}/trades?limit=50"
    for attempt_name, proxies in [("DIRECT", None), ("PROXY", {"http": FALLBACK_PROXY, "https": FALLBACK_PROXY})]:
        try:
            r = await session.get(rest_url, proxies=proxies, timeout=8)
            if r.status_code == 200:
                trades = r.json()
                if isinstance(trades, list) and trades:
                    trades_sorted = sorted(trades, key=lambda x: int(x.get("id", 0) or 0))
                    append_trades_to_csv(market_id, trades_sorted)
                return True
        except Exception as e:
            logger.debug(f"[{market_id}] Ошибка догрузки REST сделок ({attempt_name}): {e}")
    return False


async def periodic_maintenance(session, ws, market_ids, subscribe_msg):
    """Фоновая задача: проверка живости WS, watchdog-пинг, heartbeat-запись и ре-синхронизация."""
    last_resync = time.time()
    last_watchdog_ping = 0.0

    while True:
        await asyncio.sleep(2.0)
        now = time.time()
        ws_silence = now - last_ws_msg_time

        # 1. Пинг сторожевого таймера systemd (WatchdogSec) каждые 5-10 сек
        if now - last_watchdog_ping >= 5.0:
            notify_systemd("WATCHDOG=1")
            last_watchdog_ping = now

        # 2. Если WebSocket молчит дольше 45 секунд во всех потоках — принудительно разрываем соединение
        if ws_silence >= 45.0:
            logger.warning(f"⚠️ WebSocket молчит {ws_silence:.1f} сек (порог: 45с). Инициируем переподключение...")
            try:
                await ws.close()
            except Exception:
                pass
            return

        # Проактивный пинг подпиской при тишине в сокете > 20 сек (Ranger отвечает JSON-кадром)
        elif ws_silence >= 20.0:
            try:
                await ws.send(json.dumps(subscribe_msg).encode())
            except Exception as e:
                logger.warning(f"⚠️ Ошибка отправки WS-проверки: {e}")

        # 3. Порыночный контроль свежести стакана и Heartbeat-запись
        for mid in market_ids:
            time_since_last_depth = now - last_market_depth_time.get(mid, 0.0)
            time_since_last_write = now - last_write_time.get(mid, 0.0)

            # Если для конкретного рынка не поступало обновлений стакана > 60 сек
            # (затишье в тонком стакане или подвисший поток), принудительно обновляем через REST
            if time_since_last_depth >= 60.0 and time_since_last_write >= HEARTBEAT_WRITE_INTERVAL:
                logger.info(f"ℹ️ [{mid}] Стакан не обновлялся {time_since_last_depth:.1f} сек. Синхронизируем через REST...")
                fetch_ok = await fetch_snapshot(session, mid)
                if fetch_ok:
                    last_market_depth_time[mid] = now

            # Heartbeat пишем, если прошло HEARTBEAT_WRITE_INTERVAL с последней записи и сокет живой
            elif time_since_last_write >= HEARTBEAT_WRITE_INTERVAL and ws_silence <= 45.0:
                if not snapshot_ready.get(mid, False):
                    await fetch_snapshot(session, mid)
                else:
                    write_depth(mid, int(now * 1000))

        # 4. Периодическая сверка полного снимка стакана и сделок каждые 15 минут
        if now - last_resync >= SNAPSHOT_RESYNC_INTERVAL:
            for mid in market_ids:
                await fetch_snapshot(session, mid)
                await fetch_recent_trades(session, mid)
            last_resync = time.time()


async def main():
    global last_ws_msg_time
    init_seen_trades()
    notify_systemd("READY=1")
    market_ids = [m["id"] for m in MARKETS]
    while True:
        maint_task = None
        try:
            async with AsyncSession(impersonate="chrome120") as session:
                for mid in market_ids:
                    snapshot_ready[mid] = False
                    last_sequence[mid] = None
                    await fetch_snapshot(session, mid)
                    await fetch_recent_trades(session, mid)

                logger.info(f"🚀 Запуск WebSocket сбора данных для {market_ids} с {WS_URL}")

                streams = []
                for mid in market_ids:
                    streams.extend([f"{mid}.depth", f"{mid}.trades"])

                async with session.ws_connect(WS_URL) as ws:
                    last_ws_msg_time = time.time()
                    subscribe_msg = {
                        "event": "subscribe",
                        "streams": streams
                    }
                    await ws.send(json.dumps(subscribe_msg).encode())

                    # Повторная быстрая синхронизация снимка и сделок сразу после подписки на WS (закрывает окно гонки)
                    for mid in market_ids:
                        await fetch_snapshot(session, mid)
                        await fetch_recent_trades(session, mid)

                    maint_task = asyncio.create_task(periodic_maintenance(session, ws, market_ids, subscribe_msg))

                    while True:
                        if maint_task.done():
                            exc = maint_task.exception()
                            if exc:
                                raise exc
                            break

                        msg, *_ = await asyncio.wait_for(ws.recv(), timeout=45.0)
                        last_ws_msg_time = time.time()
                        notify_systemd("WATCHDOG=1")
                        data = json.loads(msg)

                        for k, stream_data in data.items():
                            if k.endswith(".depth") and isinstance(stream_data, dict):
                                mid = k.split(".")[0]
                                if mid in orderbooks:
                                    last_market_depth_time[mid] = time.time()
                                    seq = stream_data.get("sequence")
                                    prev_seq = last_sequence.get(mid)
                                    if seq is not None and prev_seq is not None and seq > prev_seq + 1:
                                        logger.warning(
                                            f"⚠️ [{mid}] Пропуск WS sequence ({prev_seq} -> {seq}), обновляем REST-снимок..."
                                        )
                                        await fetch_snapshot(session, mid)
                                    else:
                                        update_orderbook(mid, stream_data)
                                        write_depth(mid, int(time.time() * 1000))
                                    if seq is not None:
                                        last_sequence[mid] = seq

                            elif k.endswith(".trades") and isinstance(stream_data, list):
                                mid = k.split(".")[0]
                                append_trades_to_csv(mid, stream_data)

        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}. Переподключение через 5 секунд...")
            for _ in range(5):
                notify_systemd("WATCHDOG=1")
                await asyncio.sleep(1)
        finally:
            if maint_task:
                maint_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
