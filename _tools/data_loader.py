#!/usr/bin/env python3
"""
Единый фасад (Data Loader & API) для доступа к консолидированным рыночным данным.
Предоставляет простой и интуитивный доступ к данным в Jupyter Notebook:

Примеры использования:
-----------------------
from _tools.data_loader import load_coin, load_macro, load_rules, sync_data

# 1. Загрузить все данные по конкретной монете:
prl = load_coin('prl')
# prl.safetrade_depth  -> DataFrame стакана SafeTrade
# prl.bigone_depth     -> DataFrame стакана BigONE
# prl.onchain          -> DataFrame ончейн-метрик (высота блока, мемпул)

quantus = load_coin('quantus')
# quantus.safetrade_depth
# quantus.bigone_depth
# quantus.rhea_price
# quantus.onchain      -> DataFrame ончейна (блок, майнеры, циркуляция)

# 2. Загрузить макро-поводырей (BTC + NEAR 1s BBO):
macro = load_macro()

# 3. Загрузить точные торговые правила и статусы бирж:
rules = load_rules('prl')
# rules['safetrade']['min_amount'] -> 2.0
# rules['bigone']['quote_scale']   -> 4
"""

import os
import json
import logging
import pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger("data_loader")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")

class CoinData:
    """Контейнер датасетов для конкретной монеты с доступом через точку или словарь."""
    def __init__(self, coin: str, datasets: Dict[str, pd.DataFrame]):
        self.coin = coin.lower()
        self._datasets = datasets
        for name, df in datasets.items():
            setattr(self, name, df)

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self._datasets.get(key, pd.DataFrame())

    def get(self, key: str, default=None):
        return self._datasets.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._datasets

    def keys(self):
        return self._datasets.keys()

    def __repr__(self):
        info = []
        for k, v in self._datasets.items():
            info.append(f"{k}: {len(v):,} rows")
        return f"<CoinData [{self.coin.upper()}]: " + ", ".join(info) + ">"

def load_coin(coin: str) -> CoinData:
    """
    Загружает все датасеты по монете из структурированной папки data/processed/{coin}/.
    Поддерживаемые монеты: 'prl', 'quantus', 'tsc'.
    """
    coin = coin.lower()
    coin_dir = os.path.join(PROCESSED_DIR, coin)
    datasets = {}

    if os.path.exists(coin_dir):
        for fname in sorted(os.listdir(coin_dir)):
            if fname.endswith(".parquet"):
                key = fname.replace(".parquet", "")
                fpath = os.path.join(coin_dir, fname)
                try:
                    datasets[key] = pd.read_parquet(fpath)
                except Exception as e:
                    logger.warning(f"Не удалось прочитать {fpath}: {e}")
                    datasets[key] = pd.DataFrame()
    else:
        logger.warning(f"Директория для монеты {coin} не найдена: {coin_dir}")

    return CoinData(coin, datasets)

def load_macro() -> pd.DataFrame:
    """Загружает макро-поводырей (BTC/USDT + NEAR/USDT 1s BBO) из data/processed/macro/binance_bbo.parquet."""
    path = os.path.join(PROCESSED_DIR, "macro", "binance_bbo.parquet")
    if not os.path.exists(path):
        # Fallback to flat path
        path = os.path.join(PROCESSED_DIR, "binance_macro.parquet")

    if os.path.exists(path):
        return pd.read_parquet(path)
    logger.warning("Датасет макро-поводырей не найден.")
    return pd.DataFrame()

def load_rules(coin: Optional[str] = None) -> Dict[str, Any]:
    """
    Загружает актуальные торговые правила и статусы бирж из data/metadata/exchange_rules.json.
    Если указан coin ('prl', 'quantus', 'tsc'), фильтрует правила для этой монеты.
    """
    rules_file = os.path.join(METADATA_DIR, "exchange_rules.json")
    if not os.path.exists(rules_file):
        logger.warning(f"Файл правил {rules_file} не найден.")
        return {}

    with open(rules_file, "r") as f:
        data = json.load(f)

    if not coin:
        return data

    coin = coin.lower()
    filtered = {
        "updated_at": data.get("updated_at"),
        "safetrade": {},
        "bigone": {}
    }

    # SafeTrade
    for mid, mdata in data.get("safetrade", {}).items():
        if mdata.get("coin", "").lower() == coin or coin in mid:
            filtered["safetrade"][mid] = mdata

    # BigONE
    for pid, pdata in data.get("bigone", {}).items():
        if pdata.get("coin", "").lower() == coin or coin in pid.lower():
            filtered["bigone"][pid] = pdata

    return filtered

def load_rules_history() -> pd.DataFrame:
    """Загружает историю изменений правил и статусов из data/metadata/exchange_rules_history.csv."""
    path = os.path.join(METADATA_DIR, "exchange_rules_history.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def sync_data(include_vps: bool = True):
    """Выполняет полную синхронизацию данных с Google Drive и VPS."""
    from _tools.sync_all_data import sync_and_process_all
    return sync_and_process_all(include_vps=include_vps)
