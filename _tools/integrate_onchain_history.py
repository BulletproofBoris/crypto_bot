#!/usr/bin/env python3
"""
Интеграция полной ретроспективной ончейн-истории Pearl:
1. Выгружает pearl_historical_blocks.csv с VPS.
2. Формирует суточные архивы prl_onchain_YYYY-MM-DD.csv.gz для Google Диска.
3. Обновляет Parquet Data Lake (data/processed/prl/onchain.parquet).
4. Загружает суточные ончейн-архивы на Google Диск в папки соответствующих дат.
"""

import os
import sys
import shutil
import logging
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.sync_all_data import get_drive_service, FOLDER_ID, CACHE_GDRIVE_DIR, PROCESSED_DIR
from _tools.unify_gdrive_data import get_or_create_folder, upload_or_replace_file
from _tools.vps_config import VPS_HOST, VPS_PORT, VPS_USER, VPS_PASS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("integrate_onchain")

LOCAL_HIST_CSV = os.path.join(BASE_DIR, "data", "metadata", "pearl_historical_blocks.csv")

def download_from_vps():
    try:
        from _tools.vps_config import create_ssh_client
    except ImportError:
        from vps_config import create_ssh_client
    ssh = create_ssh_client(timeout=10)
    sftp = ssh.open_sftp()
    remote_path = "/opt/crypto_bot/data/realtime/onchain/pearl_historical_blocks.csv"
    os.makedirs(os.path.dirname(LOCAL_HIST_CSV), exist_ok=True)
    
    sftp.get(remote_path, LOCAL_HIST_CSV)
    sftp.close()
    ssh.close()
    logger.info(f"✅ Файл успешно скачан: {LOCAL_HIST_CSV} ({os.path.getsize(LOCAL_HIST_CSV)/(1024*1024):.2f} MB)")

def process_and_distribute():
    if not os.path.exists(LOCAL_HIST_CSV):
        raise FileNotFoundError(f"Файл не найден: {LOCAL_HIST_CSV}")

    df = pd.read_csv(LOCAL_HIST_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
    df = df.sort_values('timestamp').drop_duplicates(subset=['block_height'])
    
    logger.info(f"📊 Всего загружено блоков: {len(df):,} | Диапазон: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    
    # 1. Сохранение полного исторического датасета в Parquet
    prl_dir = os.path.join(PROCESSED_DIR, "prl")
    os.makedirs(prl_dir, exist_ok=True)
    parquet_path = os.path.join(prl_dir, "onchain.parquet")
    df.to_parquet(parquet_path, index=False)
    
    flat_parquet_path = os.path.join(PROCESSED_DIR, "prl_onchain.parquet")
    df.to_parquet(flat_parquet_path, index=False)
    logger.info(f"✅ Сохранен полный ончейн Parquet: {parquet_path} и {flat_parquet_path}")
    
    # 2. Формирование суточных архивов для Google Диска
    service = get_drive_service()
    dates = df['timestamp'].dt.strftime('%Y-%m-%d').unique()
    
    for d in dates:
        df_day = df[df['timestamp'].dt.strftime('%Y-%m-%d') == d]
        out_name = f"prl_onchain_{d}.csv.gz"
        out_path = os.path.join(CACHE_GDRIVE_DIR, out_name)
        
        # Если есть существующие ончейн-данные (например с мемпулом), мерджим
        if os.path.exists(out_path):
            try:
                df_old = pd.read_csv(out_path)
                df_old['timestamp'] = pd.to_datetime(df_old['timestamp'], utc=True)
                df_combined = pd.concat([df_old, df_day], ignore_index=True)
                df_combined = df_combined.sort_values('timestamp').drop_duplicates(subset=['block_height'], keep='last')
                df_combined.to_csv(out_path, index=False, compression='gzip')
            except Exception:
                df_day.to_csv(out_path, index=False, compression='gzip')
        else:
            df_day.to_csv(out_path, index=False, compression='gzip')
            
        logger.info(f"  📦 Сформирован архив {out_name} ({len(df_day)} блоков)")
        
        # Загрузка на Google Диск в суточную папку (если папка есть или актуальна)
        if d in ["2026-10-10", "2026-10-11", "2026-10-12"]:
            folder_id = get_or_create_folder(service, d, FOLDER_ID)
            upload_or_replace_file(service, out_path, folder_id)

    logger.info("🎉 Ончейн-история Pearl полностью интегрирована в локальный Data Lake и Google Диск!")

if __name__ == "__main__":
    download_from_vps()
    process_and_distribute()
