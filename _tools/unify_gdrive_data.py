#!/usr/bin/env python3
"""
Скрипт стандартизации хранилища Google Drive:
1. Объединяет кусочки данных за 10 и 11 октября в стандартные суточные архивы (*_YYYY-MM-DD.csv.gz).
2. Создает на Google Диске папку 2026-10-10/ и загружает туда полные данные.
3. Обновляет папку 2026-10-11/ полными объединенными данными без разрывов.
4. Удаляет старые zip-архивы из корня Google Диска.
5. Удаляет промежуточные *_legacy.csv.gz файлы.
"""

import os
import sys
import gzip
import shutil
import logging
import pandas as pd
from googleapiclient.http import MediaFileUpload

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.sync_all_data import get_drive_service, FOLDER_ID, CACHE_GDRIVE_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("unify_gdrive")

DATASETS = [
    ("safetrade_depth", "timestamp"),
    ("safetrade_trades", "id"),
    ("bigone_depth", "timestamp"),
    ("bigone_trades", "id"),
]

def merge_and_standardize_local():
    logger.info("🛠️ Локальное слияние и стандартизация суточных архивов за 10 и 11 октября...")
    
    # 1. 2026-10-10: просто переименовываем legacy в стандартный формат
    for ds, _ in DATASETS:
        legacy_f = os.path.join(CACHE_GDRIVE_DIR, f"{ds}_2026-10-10_legacy.csv.gz")
        target_f = os.path.join(CACHE_GDRIVE_DIR, f"{ds}_2026-10-10.csv.gz")
        if os.path.exists(legacy_f):
            if not os.path.exists(target_f) or os.path.getsize(target_f) == 0:
                shutil.copy2(legacy_f, target_f)
                logger.info(f"  ✅ Создан стандартный файл: {os.path.basename(target_f)}")

    # 2. 2026-10-11: объединяем кусочек №1 и кусочек №2 (legacy)
    for ds, dedup_col in DATASETS:
        part1 = os.path.join(CACHE_GDRIVE_DIR, f"{ds}_2026-10-11.csv.gz")
        part2 = os.path.join(CACHE_GDRIVE_DIR, f"{ds}_2026-10-11_legacy.csv.gz")
        
        dfs = []
        if os.path.exists(part1):
            dfs.append(pd.read_csv(part1))
        if os.path.exists(part2):
            dfs.append(pd.read_csv(part2))
            
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            if dedup_col in merged_df.columns:
                merged_df = merged_df.drop_duplicates(subset=[dedup_col])
            merged_df = merged_df.sort_values("timestamp")
            
            merged_df.to_csv(part1, index=False, compression="gzip")
            logger.info(f"  ✅ Объединен суточный архив: {ds}_2026-10-11.csv.gz ({len(merged_df):,} строк, {os.path.getsize(part1)/(1024*1024):.2f} MB)")

def get_or_create_folder(service, folder_name, parent_id):
    query = f"name = '{folder_name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    items = service.files().list(q=query, fields='files(id, name)').execute().get('files', [])
    if items:
        return items[0]['id']
    
    # Создаем папку
    metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(body=metadata, fields='id').execute()
    logger.info(f"📁 Создана новая папка на Google Диске: {folder_name} (id={folder['id']})")
    return folder['id']

def upload_or_replace_file(service, local_path, parent_id):
    fname = os.path.basename(local_path)
    query = f"name = '{fname}' and '{parent_id}' in parents and trashed = false"
    items = service.files().list(q=query, fields='files(id, name, size)').execute().get('files', [])
    
    media = MediaFileUpload(local_path, mimetype='application/gzip', resumable=True)
    
    if items:
        file_id = items[0]['id']
        logger.info(f"  🔄 Обновление файла {fname} на Google Диске...")
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        logger.info(f"  ⬆️ Загрузка нового файла {fname} на Google Диск...")
        metadata = {'name': fname, 'parents': [parent_id]}
        service.files().create(body=metadata, media_body=media).execute()

def sync_to_gdrive():
    service = get_drive_service()
    
    # 1. Папка 2026-10-10
    folder_10_id = get_or_create_folder(service, "2026-10-10", FOLDER_ID)
    for ds, _ in DATASETS:
        fpath = os.path.join(CACHE_GDRIVE_DIR, f"{ds}_2026-10-10.csv.gz")
        if os.path.exists(fpath):
            upload_or_replace_file(service, fpath, folder_10_id)

    # 2. Папка 2026-10-11
    folder_11_id = get_or_create_folder(service, "2026-10-11", FOLDER_ID)
    for ds, _ in DATASETS:
        fpath = os.path.join(CACHE_GDRIVE_DIR, f"{ds}_2026-10-11.csv.gz")
        if os.path.exists(fpath):
            upload_or_replace_file(service, fpath, folder_11_id)

    # 3. Удаление legacy zip-файлов из корня Google Диска
    logger.info("🗑️ Проверка и удаление устаревших .zip бэкапов из корня Google Диска...")
    query = f"'{FOLDER_ID}' in parents and mimeType = 'application/zip' and trashed = false"
    zips = service.files().list(q=query, fields='files(id, name)').execute().get('files', [])
    for z in zips:
        logger.info(f"  ❌ Удаление устаревшего архива: {z['name']} (id={z['id']})")
        service.files().delete(fileId=z['id']).execute()

    # 4. Удаление локальных legacy файлов
    logger.info("🧹 Очистка локальных временных файлов *_legacy.csv.gz...")
    for fname in os.listdir(CACHE_GDRIVE_DIR):
        if "_legacy.csv.gz" in fname:
            os.remove(os.path.join(CACHE_GDRIVE_DIR, fname))
            logger.info(f"  Удален локальный временный файл: {fname}")

    logger.info("🎉 Google Диск и локальный кэш полностью приведены к единому стандарту!")

if __name__ == "__main__":
    merge_and_standardize_local()
    sync_to_gdrive()
