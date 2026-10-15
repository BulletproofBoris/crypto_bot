#!/usr/bin/env python3
"""
Сборщик ончейн-метрик (Pearl, Quantus) и торговых правил/статусов бирж (SafeTrade, BigONE).
Частота опроса:
- Ончейн-метрики (PRL mempool/height, Quantus stats/miners): каждые 30 секунд.
- Торговые правила и статусы шлюзов (SafeTrade & BigONE): каждый 1 час (и сразу при старте).
"""

import os
import sys
import time
import json
import csv
import logging
from datetime import datetime

# Для обхода Cloudflare на SafeTrade используем curl_cffi если доступно, иначе requests
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False
import requests

DEFAULT_DATA_DIR = "/opt/crypto_bot/data" if os.path.exists("/opt/crypto_bot") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_DIR = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)
ONCHAIN_DIR = os.path.join(DATA_DIR, "realtime", "onchain")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
LOG_DIR = os.path.join(DATA_DIR, "logs")

os.makedirs(ONCHAIN_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "onchain_and_rules.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("onchain_collector")

SAFETRADE_PROXY = "http://edtilhmt:iblj7uuixsmy@31.59.20.176:6754"

PRL_ONCHAIN_CSV = os.path.join(ONCHAIN_DIR, "prl_onchain.csv")
QUANTUS_ONCHAIN_CSV = os.path.join(ONCHAIN_DIR, "quantus_onchain.csv")
RULES_JSON = os.path.join(METADATA_DIR, "exchange_rules.json")
RULES_HISTORY_CSV = os.path.join(METADATA_DIR, "exchange_rules_history.csv")

PRL_ONCHAIN_COLS = [
    "timestamp", "block_height", "mempool_size", "indexer_lag", "difficulty",
    "reward_grains", "reward_prl", "tx_count", "transfer_count",
    "transfer_volume_prl", "max_tx_prl", "whale_tx_count_5k",
    "whale_volume_5k", "whale_tx_count_25k", "whale_volume_25k",
    "unique_receivers", "miner_pool", "miner_address"
]

QUANTUS_ONCHAIN_COLS = [
    "timestamp", "block_height", "circulating_supply", "total_supply",
    "total_miners", "total_accounts", "total_transfers",
    "recent_tx_count_5m", "recent_vol_qtc_5m", "recent_whale_vol_qtc_5m"
]

def init_csv_files():
    if not os.path.exists(PRL_ONCHAIN_CSV):
        with open(PRL_ONCHAIN_CSV, "w", newline="") as f:
            csv.writer(f).writerow(PRL_ONCHAIN_COLS)
    else:
        try:
            with open(PRL_ONCHAIN_CSV, "r", newline="") as f:
                reader = list(csv.reader(f))
            if reader and reader[0] != PRL_ONCHAIN_COLS:
                old_header = reader[0]
                upgraded_rows = [PRL_ONCHAIN_COLS]
                for r in reader[1:]:
                    row_map = {old_header[i]: r[i] for i in range(min(len(old_header), len(r)))}
                    if not row_map.get("reward_prl") and row_map.get("reward_grains"):
                        try:
                            row_map["reward_prl"] = f"{float(row_map['reward_grains']) / 1e8:.4f}"
                        except Exception:
                            pass
                    upgraded_rows.append([row_map.get(c, "") for c in PRL_ONCHAIN_COLS])
                with open(PRL_ONCHAIN_CSV, "w", newline="") as f:
                    csv.writer(f).writerows(upgraded_rows)
        except Exception as e:
            logger.warning(f"Не удалось обновить заголовок {PRL_ONCHAIN_CSV}: {e}")
            
    if not os.path.exists(QUANTUS_ONCHAIN_CSV):
        with open(QUANTUS_ONCHAIN_CSV, "w", newline="") as f:
            csv.writer(f).writerow(QUANTUS_ONCHAIN_COLS)
    else:
        try:
            with open(QUANTUS_ONCHAIN_CSV, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                if first_line and len(first_line.split(",")) < len(QUANTUS_ONCHAIN_COLS):
                    rest = f.read()
                    tmp_path = QUANTUS_ONCHAIN_CSV + ".hdr_tmp"
                    with open(tmp_path, "w", encoding="utf-8", newline="") as out_f:
                        out_f.write(",".join(QUANTUS_ONCHAIN_COLS) + "\n")
                        out_f.write(rest)
                    os.replace(tmp_path, QUANTUS_ONCHAIN_CSV)
        except Exception as e:
            logger.warning(f"Не удалось обновить заголовок {QUANTUS_ONCHAIN_CSV}: {e}")
            
    if not os.path.exists(RULES_HISTORY_CSV):
        with open(RULES_HISTORY_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "exchange", "market", "min_amount", "min_value",
                "price_precision", "amount_precision", "deposit_enabled", "withdrawal_enabled", "withdrawal_fee"
            ])

def fetch_safetrade_rules():
    """Получает торговые правила SafeTrade для prlusdt, quantususdt."""
    logger.info("Запрос торговых правил SafeTrade...")
    markets_data = []
    target_ids = {"prlusdt": "PRL", "quantususdt": "QUANTUS"}
    rules = {}
    
    for proxies in [None, {"http": SAFETRADE_PROXY, "https": SAFETRADE_PROXY}]:
        markets_data = []
        try:
            for page in range(1, 6):
                url = f"https://safe.trade/api/v2/peatio/public/markets?page={page}&limit=100"
                if HAS_CURL_CFFI:
                    r = cffi_requests.get(url, proxies=proxies, impersonate="chrome120", timeout=10)
                else:
                    r = requests.get(url, proxies=proxies, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if not data:
                        break
                    markets_data.extend(data)
                    found = sum(1 for m in markets_data if m.get("id") in target_ids)
                    if found == len(target_ids):
                        break
                else:
                    break
            if any(m.get("id") in target_ids for m in markets_data):
                break
        except Exception as e:
            logger.warning(f"Ошибка запроса SafeTrade markets (proxies={bool(proxies)}): {e}")

    for m in markets_data:
        mid = m.get("id")
        if mid in target_ids:
            rules[mid] = {
                "coin": target_ids[mid],
                "min_price": float(m.get("min_price", 0)),
                "max_price": float(m.get("max_price", 0)),
                "min_amount": float(m.get("min_amount", 0)),
                "price_precision": int(m.get("price_precision", 0)),
                "amount_precision": int(m.get("amount_precision", 0)),
                "state": m.get("state", "enabled"),
                "deposit_enabled": True,
                "withdrawal_enabled": True,
                "withdrawal_fee": None
            }
    return rules

def fetch_bigone_rules():
    """Получает торговые правила и статусы кошельков BigONE."""
    logger.info("Запрос торговых правил и статусов BigONE...")
    rules = {}
    try:
        # 1. Asset pairs (торговые фильтры)
        r_pairs = requests.get("https://big.one/api/v3/asset_pairs", timeout=10)
        pairs_data = r_pairs.json().get("data", []) if r_pairs.status_code == 200 else []

        target_pairs = {"PRL-USDT": "PRL", "QTC-USDT": "QUANTUS"}
        for p in pairs_data:
            name = p.get("name")
            if name in target_pairs:
                rules[name] = {
                    "coin": target_pairs[name],
                    "min_quote_value": float(p.get("min_quote_value", 0)),
                    "max_quote_value": float(p.get("max_quote_value", 0)),
                    "quote_scale": int(p.get("quote_scale", 0)),
                    "base_scale": int(p.get("base_scale", 0)),
                }

        # 2. Assets (шлюзы и комиссии)
        r_assets = requests.get("https://big.one/api/v3/assets", timeout=10)
        assets_data = r_assets.json().get("data", []) if r_assets.status_code == 200 else []

        for a in assets_data:
            sym = a.get("symbol")
            pair_key = "PRL-USDT" if sym == "PRL" else ("QTC-USDT" if sym == "QTC" else None)
            if pair_key and pair_key in rules:
                gateways = a.get("binding_gateways", [])
                if gateways:
                    gw = gateways[0]
                    rules[pair_key]["deposit_enabled"] = gw.get("is_deposit_enabled", True)
                    rules[pair_key]["withdrawal_enabled"] = gw.get("is_withdrawal_enabled", True)
                    rules[pair_key]["withdrawal_fee"] = float(gw.get("withdrawal_fee", 0))
                    rules[pair_key]["min_withdrawal_amount"] = float(gw.get("min_withdrawal_amount", 0))

    except Exception as e:
        logger.error(f"Ошибка запроса BigONE: {e}")

    return rules

def update_exchange_rules():
    """Обновляет и сохраняет актуальные правила бирж."""
    try:
        prev_data = {}
        if os.path.exists(RULES_JSON):
            try:
                with open(RULES_JSON, "r") as f:
                    prev_data = json.load(f)
            except Exception:
                prev_data = {}

        st_rules = fetch_safetrade_rules() or prev_data.get("safetrade", {})
        bo_rules = fetch_bigone_rules() or prev_data.get("bigone", {})
        st_rules.pop("tscusdt", None)

        ts = datetime.utcnow().isoformat() + "Z"
        combined = {
            "updated_at": ts,
            "safetrade": st_rules,
            "bigone": bo_rules
        }

        # Атомарная запись JSON
        tmp_file = RULES_JSON + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(combined, f, indent=2)
        os.replace(tmp_file, RULES_JSON)
        logger.info(f"✅ Правила бирж сохранены в {RULES_JSON}")

        # Запись в историю CSV
        with open(RULES_HISTORY_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            for m, d in st_rules.items():
                writer.writerow([
                    ts, "safetrade", m, d.get("min_amount"), "", d.get("price_precision"),
                    d.get("amount_precision"), d.get("deposit_enabled"), d.get("withdrawal_enabled"), d.get("withdrawal_fee")
                ])
            for m, d in bo_rules.items():
                writer.writerow([
                    ts, "bigone", m, "", d.get("min_quote_value"), d.get("quote_scale"),
                    d.get("base_scale"), d.get("deposit_enabled"), d.get("withdrawal_enabled"), d.get("withdrawal_fee")
                ])

    except Exception as e:
        logger.error(f"Ошибка обновления правил бирж: {e}")

_prl_block_cache = {}

def collect_prl_onchain():
    """Опрашивает статус сети Pearl (PRL) через api.prlscan.com: высота, мемпул, сложность, награда и переводы китов."""
    try:
        r = requests.get("https://api.prlscan.com/v1/status", timeout=8)
        if r.status_code == 200:
            d = r.json()
            ts = datetime.utcnow().isoformat() + "Z"
            height = d.get("tip_height")
            mempool = d.get("mempool_size")
            lag = d.get("indexer_lag")

            if height is not None and height not in _prl_block_cache:
                try:
                    rb = requests.get(f"https://api.prlscan.com/v1/blocks/{height}", timeout=8)
                    if rb.status_code == 200:
                        bdata = rb.json()
                        blk = bdata.get("block", {})
                        txs = bdata.get("transactions", [])
                        r_grains = blk.get("reward_grains")
                        r_prl = round((r_grains or 0) / 1e8, 4) if r_grains is not None else ""
                        non_cb = [t for t in txs if not t.get("is_coinbase")]
                        transfer_vols = [t.get("output_grains", 0) / 1e8 for t in non_cb]
                        whale_vols = [v for v in transfer_vols if v >= 5000.0]
                        whale_25k = [v for v in transfer_vols if v >= 25000.0]
                        receivers = set()
                        for t in non_cb:
                            for out_item in t.get("outputs", []) or []:
                                addr = out_item.get("address") if isinstance(out_item, dict) else None
                                if addr:
                                    receivers.add(addr)
                            if t.get("to_address"):
                                receivers.add(t.get("to_address"))
                        uniq_recv = len(receivers) if receivers else len(non_cb)
                        _prl_block_cache.clear()
                        _prl_block_cache[height] = {
                            "difficulty": blk.get("difficulty", ""),
                            "reward_grains": r_grains if r_grains is not None else "",
                            "reward_prl": r_prl,
                            "tx_count": blk.get("tx_count", 0),
                            "transfer_count": len(non_cb),
                            "transfer_volume_prl": round(sum(transfer_vols), 4),
                            "max_tx_prl": round(max(transfer_vols), 4) if transfer_vols else 0.0,
                            "whale_tx_count_5k": len(whale_vols),
                            "whale_volume_5k": round(sum(whale_vols), 4),
                            "whale_tx_count_25k": len(whale_25k),
                            "whale_volume_25k": round(sum(whale_25k), 4),
                            "unique_receivers": uniq_recv,
                            "miner_pool": blk.get("miner_label", "Unknown"),
                            "miner_address": blk.get("miner_address", "")
                        }
                except Exception as eb:
                    logger.debug(f"Не удалось получить детали блока Pearl {height}: {eb}")

            binfo = _prl_block_cache.get(height, {})
            if height is not None:
                row = [
                    ts,
                    height,
                    mempool,
                    lag,
                    binfo.get("difficulty", ""),
                    binfo.get("reward_grains", ""),
                    binfo.get("reward_prl", ""),
                    binfo.get("tx_count", ""),
                    binfo.get("transfer_count", ""),
                    binfo.get("transfer_volume_prl", ""),
                    binfo.get("max_tx_prl", ""),
                    binfo.get("whale_tx_count_5k", ""),
                    binfo.get("whale_volume_5k", ""),
                    binfo.get("whale_tx_count_25k", ""),
                    binfo.get("whale_volume_25k", ""),
                    binfo.get("unique_receivers", ""),
                    binfo.get("miner_pool", ""),
                    binfo.get("miner_address", "")
                ]
                with open(PRL_ONCHAIN_CSV, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                logger.debug(f"Pearl onchain: height={height}, diff={binfo.get('difficulty')}, mempool={mempool}")
                return True
    except Exception as e:
        logger.error(f"Ошибка запроса Pearl onchain: {e}")
    return False

def collect_quantus_onchain():
    """Опрашивает метрики сети Quantus и поток переводов через GraphQL sqm.quantus.com."""
    try:
        from datetime import timedelta
        since_5m = (datetime.utcnow() - timedelta(minutes=5)).isoformat() + "+00:00"
        query = f"""
        query {{
          chain_stats(limit: 1, order_by: {{id: desc}}) {{
            block_height
            circulating_supply
            total_supply
            total_miners
            total_accounts
            total_executed_transfers
          }}
          transfer(where: {{timestamp: {{_gte: "{since_5m}"}}}}, limit: 200, order_by: {{timestamp: desc}}) {{
            amount
          }}
        }}
        """
        r = requests.post("https://sqm.quantus.com/v1/graphql", json={"query": query}, timeout=8)
        if r.status_code == 200:
            data = r.json().get("data", {})
            stats_list = data.get("chain_stats", [])
            txs_5m = data.get("transfer", []) or []
            vols_qtc = []
            for t in txs_5m:
                try:
                    vols_qtc.append(float(t.get("amount", 0)) / 1e8)
                except Exception:
                    pass
            recent_tx_cnt = len(vols_qtc)
            recent_vol = round(sum(vols_qtc), 4)
            recent_whale_vol = round(sum(v for v in vols_qtc if v >= 10000.0), 4)
            if stats_list:
                s = stats_list[0]
                ts = datetime.utcnow().isoformat() + "Z"
                row = [
                    ts,
                    s.get("block_height"),
                    s.get("circulating_supply"),
                    s.get("total_supply"),
                    s.get("total_miners"),
                    s.get("total_accounts"),
                    s.get("total_executed_transfers"),
                    recent_tx_cnt,
                    recent_vol,
                    recent_whale_vol
                ]
                with open(QUANTUS_ONCHAIN_CSV, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                logger.debug(f"Quantus onchain: height={s.get('block_height')}, tx_5m={recent_tx_cnt}")
                return True
    except Exception as e:
        logger.error(f"Ошибка запроса Quantus onchain: {e}")
    return False


def main_loop():
    init_csv_files()
    logger.info("🚀 Запуск сборщика ончейн-метрик и торговых правил...")

    # Сразу при старте обновляем правила бирж
    update_exchange_rules()
    last_rules_update = time.time()

    while True:
        try:
            # 1. Ончейн-опрос (каждые 30 секунд)
            collect_prl_onchain()
            collect_quantus_onchain()

            # 2. Обновление правил бирж (каждый 1 час = 3600 сек)
            now = time.time()
            if now - last_rules_update >= 3600:
                update_exchange_rules()
                last_rules_update = now

        except Exception as e:
            logger.error(f"Непредвиденная ошибка в основном цикле: {e}")

        time.sleep(30)

if __name__ == "__main__":
    main_loop()
