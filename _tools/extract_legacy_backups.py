#!/usr/bin/env python3
"""
Скрипт для извлечения и интеграции исторических архивов (legacy .zip) с Google Drive в локальный кэш data/cache_gdrive/*.csv.gz.
Позволяет бесшовно устранить разрыв в данных PRL между 10 и 12 октября 2026.
"""

import os
import io
import sys
import gzip
import shutil
import zipfile
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.sync_all_data import get_drive_service, FOLDER_ID, CACHE_GDRIVE_DIR
from googleapiclient.http import MediaIoBaseDownload

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("legacy_extractor")

MAPPING = {
    'depth.csv': 'safetrade_depth',
    'depth.csv.bak': 'safetrade_depth',
    'trades.csv': 'safetrade_trades',
    'trades.csv.bak': 'safetrade_trades',
    'depth_bigone.csv': 'bigone_depth',
    'depth_bigone.csv.bak': 'bigone_depth',
    'trades_bigone.csv': 'bigone_trades',
    'trades_bigone.csv.bak': 'bigone_trades',
}

def process_legacy_backups():
    os.makedirs(CACHE_GDRIVE_DIR, exist_ok=True)
    service = get_drive_service()

    query = f"'{FOLDER_ID}' in parents and trashed = false"
    items = service.files().list(q=query, fields='files(id, name, size)').execute().get('files', [])

    legacy_zips = [it for it in items if it['name'].startswith('prlusdt_backup_') and it['name'].endswith('.zip')]
    logger.info(f"Найдено legacy zip-архивов на Google Drive: {len(legacy_zips)}")

    for item in legacy_zips:
        fname = item['name']
        fid = item['id']
        fsize = int(item.get('size', 0))

        # Извлекаем дату (например prlusdt_backup_20261011_220001.zip -> 2026-10-11)
        parts = fname.split('_')
        if len(parts) >= 3 and len(parts[2]) >= 8:
            raw_d = parts[2][:8]
            date_str = f"{raw_d[:4]}-{raw_d[4:6]}-{raw_d[6:8]}"
        else:
            date_str = "legacy"

        logger.info(f"--- Обработка архива: {fname} (дата: {date_str}, размер: {fsize / (1024*1024):.2f} MB) ---")

        # Проверяем, извлечены ли уже файлы
        check_file = os.path.join(CACHE_GDRIVE_DIR, f"safetrade_depth_{date_str}_legacy.csv.gz")
        if os.path.exists(check_file):
            logger.info(f"Файлы для даты {date_str} уже существуют в {CACHE_GDRIVE_DIR}, пропускаем.")
            continue

        # Скачиваем zip во временный буфер в памяти
        logger.info(f"Загрузка архива {fname} с Google Drive...")
        req = service.files().get_media(fileId=fid)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                logger.info(f"Прогресс скачивания: {int(status.progress() * 100)}%")

        buf.seek(0)
        logger.info("Архив загружен. Распаковка и потоковая конвертация в .csv.gz...")

        with zipfile.ZipFile(buf) as z:
            for member in z.namelist():
                base_member = os.path.basename(member)
                if base_member in MAPPING:
                    target_prefix = MAPPING[base_member]
                    target_name = f"{target_prefix}_{date_str}_legacy.csv.gz"
                    target_path = os.path.join(CACHE_GDRIVE_DIR, target_name)

                    logger.info(f"  Извлечение: {member} ({z.getinfo(member).file_size / (1024*1024):.1f} MB) -> {target_name}...")
                    with z.open(member) as src, gzip.open(target_path, 'wb', compresslevel=6) as dst:
                        shutil.copyfileobj(src, dst, length=1024*1024)
                    logger.info(f"  ✅ Сохранен: {target_name} ({os.path.getsize(target_path) / (1024*1024):.1f} MB)")

    logger.info("Все legacy архивы успешно обработаны и добавлены в data/cache_gdrive/!")

if __name__ == "__main__":
    process_legacy_backups()
