#!/usr/bin/env python3
"""
Синхронизация конфиденциальных параметров и секретов проекта через Google Drive.
Позволяет безопасно хранить пароли, API-ключи и конфигурацию VPS в облаке
без утечки в публичные/приватные Git-репозитории.
"""

import os
import json
import logging
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

try:
    from _tools.sync_all_data import get_drive_service, FOLDER_ID
except ImportError:
    from sync_all_data import get_drive_service, FOLDER_ID

logger = logging.getLogger("sync_secrets")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_LOCAL_PATH = os.path.join(CURRENT_DIR, "project_secrets.json")
SECRETS_REMOTE_NAME = "project_secrets.json"

def upload_secrets_to_gdrive():
    """Загружает локальный project_secrets.json на Google Drive в защищённую папку."""
    if not os.path.exists(SECRETS_LOCAL_PATH):
        logger.error(f"Локальный файл секретов не найден: {SECRETS_LOCAL_PATH}")
        return False

    service = get_drive_service()
    query = f"'{FOLDER_ID}' in parents and name = '{SECRETS_REMOTE_NAME}' and trashed = false"
    resp = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = resp.get('files', [])

    media = MediaFileUpload(SECRETS_LOCAL_PATH, mimetype='application/json', resumable=False)
    if files:
        file_id = files[0]['id']
        logger.info(f"Обновление существующего файла секретов на Google Диске (ID: {file_id})...")
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        meta = {'name': SECRETS_REMOTE_NAME, 'parents': [FOLDER_ID]}
        logger.info("Создание файла секретов на Google Диске...")
        res = service.files().create(body=meta, media_body=media, fields='id').execute()
        file_id = res.get('id')

    logger.info(f"✅ Секреты успешно синхронизированы с Google Drive (ID: {file_id})")
    return True

def download_secrets_from_gdrive():
    """Скачивает project_secrets.json с Google Drive, если его нет локально."""
    service = get_drive_service()
    query = f"'{FOLDER_ID}' in parents and name = '{SECRETS_REMOTE_NAME}' and trashed = false"
    resp = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = resp.get('files', [])

    if not files:
        logger.warning("Файл секретов на Google Диске не найден.")
        return False

    file_id = files[0]['id']
    logger.info(f"📥 Скачивание секретов с Google Drive (ID: {file_id})...")
    req = service.files().get_media(fileId=file_id)
    with io.FileIO(SECRETS_LOCAL_PATH, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    logger.info(f"✅ Файл секретов сохранён локально: {SECRETS_LOCAL_PATH}")
    return True

if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        download_secrets_from_gdrive()
    else:
        upload_secrets_to_gdrive()

