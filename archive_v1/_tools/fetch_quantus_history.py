#!/usr/bin/env python3
"""
Ретроспективная выгрузка ончейн-данных сети Quantus (QNTUS/QTC):
1. Выгрузка ежедневной ончейн-статистики (daily_chain_stats):
   - active_accounts, blocks_count, transferred_amount, tx_count
2. Выгрузка транзакций китов (whale transfers) с 3 октября 2026:
   - крупные переводы (>= 10,000,000,000,000 grains / 100k+ QTC)
3. Формирование обогащенного датасета Quantus ончейн.
4. Синхронизация с Parquet Data Lake и Google Диском.
"""

import os
import sys
import gzip
import json
import logging
import requests
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.sync_all_data import get_drive_service, FOLDER_ID, CACHE_GDRIVE_DIR, PROCESSED_DIR
from _tools.unify_gdrive_data import get_or_create_folder, upload_or_replace_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("quantus_history")

GRAPHQL_URL = "https://sqm.quantus.com/v1/graphql"
WHALE_THRESHOLD_GRAINS = 10_000_000_000_000  # 100,000 QTC (единицы 1e8)

def fetch_daily_stats():
    """Запрашивает историческую суточную статистику Quantus."""
    logger.info("Запрос daily_chain_stats Quantus...")
    query = """
    query {
      daily_chain_stats(order_by: {date: asc}) {
        id
        date
        active_accounts
        blocks_count
        transferred_amount
        tx_count
      }
    }
    """
    r = requests.post(GRAPHQL_URL, json={"query": query}, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", {}).get("daily_chain_stats", [])
    logger.info(f"Получено {len(data)} суточных записей Quantus.")
    return data

def fetch_whale_transfers(start_date="2026-10-03T00:00:00+00:00"):
    """Запрашивает все транзакции китов Quantus с заданной даты с пагинацией offset."""
    logger.info(f"Запрос всех транзакций китов Quantus с {start_date} (порог {WHALE_THRESHOLD_GRAINS})...")
    all_transfers = []
    offset = 0
    limit = 1000
    
    while True:
        query = f"""
        query {{
          transfer(
            where: {{
              timestamp: {{_gte: "{start_date}"}},
              amount: {{_gte: "{WHALE_THRESHOLD_GRAINS}"}}
            }},
            order_by: {{timestamp: asc}},
            limit: {limit},
            offset: {offset}
          ) {{
            id
            amount
            block_height
            from_id
            to_id
            timestamp
          }}
        }}
        """
        r = requests.post(GRAPHQL_URL, json={"query": query}, timeout=20)
        r.raise_for_status()
        batch = r.json().get("data", {}).get("transfer", [])
        if not batch:
            break
        all_transfers.extend(batch)
        offset += len(batch)
        if len(batch) < limit:
            break
            
    logger.info(f"Всего получено {len(all_transfers)} транзакций китов Quantus.")
    return all_transfers

def build_quantus_onchain_dataset():
    daily_stats = fetch_daily_stats()
    whale_txs = fetch_whale_transfers()
    
    # Формируем DataFrame китов
    df_whales = pd.DataFrame(whale_txs)
    if not df_whales.empty:
        df_whales['timestamp'] = pd.to_datetime(df_whales['timestamp'], format='ISO8601', utc=True)
        df_whales['amount_qtc'] = df_whales['amount'].astype(float) / 1e8
        df_whales['date'] = df_whales['timestamp'].dt.strftime('%Y-%m-%d')
        
        whale_agg = df_whales.groupby('date').agg(
            whale_tx_count=('id', 'count'),
            whale_volume_qtc=('amount_qtc', 'sum'),
            max_whale_tx_qtc=('amount_qtc', 'max')
        ).reset_index()
    else:
        whale_agg = pd.DataFrame(columns=['date', 'whale_tx_count', 'whale_volume_qtc', 'max_whale_tx_qtc'])

    # Формируем DataFrame суточной статистики
    df_daily = pd.DataFrame(daily_stats)
    df_daily = df_daily[df_daily['id'] != '1970-01-01'].copy()
    df_daily['timestamp'] = pd.to_datetime(df_daily['date'], utc=True)
    df_daily['date'] = df_daily['timestamp'].dt.strftime('%Y-%m-%d')
    df_daily['transferred_volume_qtc'] = df_daily['transferred_amount'].astype(float) / 1e8
    
    # Мерджим китов с суточной статистикой
    df_merged = pd.merge(df_daily, whale_agg, on='date', how='left')
    df_merged['whale_tx_count'] = df_merged['whale_tx_count'].fillna(0).astype(int)
    df_merged['whale_volume_qtc'] = df_merged['whale_volume_qtc'].fillna(0.0)
    df_merged['max_whale_tx_qtc'] = df_merged['max_whale_tx_qtc'].fillna(0.0)
    
    # Оставляем только нужные колонки
    cols = [
        'timestamp', 'date', 'active_accounts', 'blocks_count',
        'tx_count', 'transferred_volume_qtc', 'whale_tx_count',
        'whale_volume_qtc', 'max_whale_tx_qtc'
    ]
    df_merged = df_merged[cols].sort_values('timestamp').reset_index(drop=True)
    
    # Фильтруем с 2026-10-03 (за неделю до старта торгов)
    df_merged = df_merged[df_merged['timestamp'] >= '2026-10-03T00:00:00Z']
    
    logger.info(f"📊 Итоговый Quantus ончейн датасет: {len(df_merged)} дней.")
    for _, r in df_merged.iterrows():
        logger.info(f"  {r['date']} | Активных аккаунтов: {r['active_accounts']} | ТХ: {r['tx_count']} | Китов: {r['whale_tx_count']} ({r['whale_volume_qtc']:,.0f} QTC)")

    # 1. Сохранение в Parquet
    qntus_dir = os.path.join(PROCESSED_DIR, "quantus")
    os.makedirs(qntus_dir, exist_ok=True)
    
    daily_parquet = os.path.join(qntus_dir, "onchain_daily.parquet")
    df_merged.to_parquet(daily_parquet, index=False)
    
    hier_parquet = os.path.join(qntus_dir, "onchain.parquet")
    df_merged.to_parquet(hier_parquet, index=False)
    
    flat_parquet = os.path.join(PROCESSED_DIR, "quantus_onchain.parquet")
    df_merged.to_parquet(flat_parquet, index=False)
    logger.info(f"✅ Сохранены {daily_parquet}, {hier_parquet} и {flat_parquet}")
    
    # 2. Выгрузка в суточные архивы для Google Диска
    service = get_drive_service()
    for _, row in df_merged.iterrows():
        d = row['date']
        if d in ["2026-10-10", "2026-10-11", "2026-10-12"]:
            # Создаем или дополняем архив
            out_name = f"quantus_onchain_{d}.csv.gz"
            out_path = os.path.join(CACHE_GDRIVE_DIR, out_name)
            
            # Сохраняем суточный файл
            row_df = pd.DataFrame([row])
            row_df.to_csv(out_path, index=False, compression='gzip')
            logger.info(f"  📦 Сформирован суточный архив {out_name}")
            
            folder_id = get_or_create_folder(service, d, FOLDER_ID)
            upload_or_replace_file(service, out_path, folder_id)

    return df_merged

if __name__ == "__main__":
    build_quantus_onchain_dataset()
