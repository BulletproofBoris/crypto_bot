#!/usr/bin/env python3
"""
Скрипт безопасной суточной ротации данных и выгрузки на Google Drive.
Принцип работы (Zero Data Loss):
1. Для каждого горячего файла данных фиксируется точка среза (строки до момента ротации).
2. Срез сжимается в .csv.gz (сжатие ~85-90%).
3. Архив выгружается на Google Drive в папку с текущей датой (YYYY-MM-DD).
4. Только при подтвержденном успехе выгрузки горячий файл атомарно обрезается:
   в нем остается CSV-заголовок и любые новые строки, поступившие во время архивации и выгрузки.
5. Поддерживается регулярный запуск (cron раз в сутки) и аварийный запуск по порогу свободного места на диске.
"""

import os
import sys
import gzip
import shutil
import logging
import argparse
from datetime import datetime, timezone
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("rotate_and_upload")

# Конфигурация Google Drive
FOLDER_ID = '1-qWy2q2nYoZtSCiMDZ1DiKZ5roBqXx-c'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token.json')

# Порог свободного места для аварийной ротации (в гигабайтах)
MIN_FREE_DISK_GB = 2.5

# Список файлов для ротации на VPS
FILES_TO_ROTATE = [
    {
        "prefix": "safetrade_depth",
        "path": "/opt/crypto_bot/data/realtime/prlusdt/depth.csv",
    },
    {
        "prefix": "safetrade_trades",
        "path": "/opt/crypto_bot/data/realtime/prlusdt/trades.csv",
    },
    {
        "prefix": "bigone_depth",
        "path": "/opt/crypto_bot/data/realtime/prlusdt_bigone/depth_bigone.csv",
    },
    {
        "prefix": "bigone_trades",
        "path": "/opt/crypto_bot/data/realtime/prlusdt_bigone/trades_bigone.csv",
    },
]

def get_free_disk_gb(path="/"):
    """Возвращает свободное место на диске в гигабайтах."""
    stat = os.statvfs(path)
    free_bytes = stat.f_bavail * stat.f_frsize
    return free_bytes / (1024 ** 3)

def get_drive_service():
    """Инициализирует и возвращает сервис Google Drive API."""
    if not os.path.exists(TOKEN_PATH):
        raise FileNotFoundError(f"Файл токена авторизации не найден: {TOKEN_PATH}. Запустите auth_gdrive.py.")
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    return build('drive', 'v3', credentials=creds)

def get_or_create_date_folder(service, parent_id, date_str):
    """Находит или создает подпапку с датой (YYYY-MM-DD) в указанной папке Google Drive."""
    query = f"'{parent_id}' in parents and name = '{date_str}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    resp = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = resp.get('files', [])
    if files:
        logger.info(f"Найдена существующая папка на Диске для даты {date_str} (ID: {files[0]['id']})")
        return files[0]['id']
    
    meta = {
        'name': date_str,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(body=meta, fields='id').execute()
    folder_id = folder.get('id')
    logger.info(f"Создана новая папка на Диске для даты {date_str} (ID: {folder_id})")
    return folder_id

def upload_file_to_drive(service, file_path, folder_id, file_name):
    """Загружает файл в указанную папку Google Drive."""
    media = MediaFileUpload(file_path, mimetype='application/gzip', resumable=True)
    meta = {
        'name': file_name,
        'parents': [folder_id]
    }
    logger.info(f"Загрузка {file_name} ({os.path.getsize(file_path) / (1024 * 1024):.2f} MB) на Google Drive...")
    uploaded = service.files().create(body=meta, media_body=media, fields='id, name, size').execute()
    file_id = uploaded.get('id')
    logger.info(f"✅ Файл успешно загружен на Диск: {file_name} (ID: {file_id})")
    return file_id

def get_unique_archive_name(service, folder_id, prefix, date_str, timestamp_suffix=""):
    """Определяет уникальное имя для архива на Google Диске, добавляя _part2, _part3 при наличии дубликатов."""
    if timestamp_suffix:
        return f"{prefix}_{date_str}_{timestamp_suffix}.csv.gz"

    query = f"'{folder_id}' in parents and trashed = false"
    resp = service.files().list(q=query, spaces='drive', fields='files(name)').execute()
    existing_names = {f['name'] for f in resp.get('files', [])}

    base_name = f"{prefix}_{date_str}"
    candidate = f"{base_name}.csv.gz"
    if candidate not in existing_names:
        return candidate

    part = 2
    while f"{base_name}_part{part}.csv.gz" in existing_names:
        part += 1
    return f"{base_name}_part{part}.csv.gz"

def process_single_file(service, folder_id, file_info, date_str, timestamp_suffix=""):
    """
    Выполняет безопасную ротацию для одного файла:
    1. Читает и архивирует строки до текущего момента.
    2. Загружает архив на Google Drive.
    3. Атомарно перезаписывает горячий файл (заголовок + строки, поступившие во время загрузки).
    """
    file_path = file_info["path"]
    prefix = file_info["prefix"]

    if not os.path.exists(file_path):
        logger.warning(f"Файл {file_path} не найден на диске. Пропуск.")
        return False

    archive_name = get_unique_archive_name(service, folder_id, prefix, date_str, timestamp_suffix)
    tmp_archive_path = os.path.join("/tmp", archive_name)

    total_lines = 0
    header_line = None
    last_archived_line = None

    # Шаг 1. Читаем файл потоково и пишем в gzip архив
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         gzip.open(tmp_archive_path, "wt", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if idx == 0:
                header_line = line
            last_archived_line = line
            f_out.write(line)
            total_lines += 1

    # Если в файле только заголовок или он пуст — не архивируем
    if total_lines <= 1:
        logger.info(f"Файл {file_path} содержит только заголовок ({total_lines} строк). Архивировать нечего.")
        if os.path.exists(tmp_archive_path):
            os.remove(tmp_archive_path)
        return True

    archive_size_mb = os.path.getsize(tmp_archive_path) / (1024 * 1024)
    orig_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"Архивировано {total_lines} строк из {file_path} ({orig_size_mb:.2f} MB -> {archive_size_mb:.2f} MB сжато).")

    # Шаг 2. Выгрузка на Google Drive
    try:
        drive_file_id = upload_file_to_drive(service, tmp_archive_path, folder_id, archive_name)
        if not drive_file_id:
            raise RuntimeError(f"Не удалось получить ID загруженного файла {archive_name}")
    except Exception as e:
        logger.error(f"❌ Ошибка выгрузки {archive_name} на Google Drive: {e}")
        logger.error("Горячий файл НЕ будет обрезан во избежание потери данных!")
        if os.path.exists(tmp_archive_path):
            os.remove(tmp_archive_path)
        return False

    # Шаг 3. Атомарная обрезка горячего файла
    tmp_hot_path = file_path + ".tmp"
    lines_preserved = 0
    skipped_count = 0

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f_src, \
             open(tmp_hot_path, "w", encoding="utf-8") as f_dst:
            # Записываем заголовок
            src_header = f_src.readline()
            f_dst.write(header_line if header_line else src_header)

            # Пропускаем архивированные строки данных (total_lines - 1)
            for _ in range(total_lines - 1):
                line = f_src.readline()
                if not line:
                    break
                skipped_count += 1

            # Записываем все новые строки, которые сборщик успел добавить во время выгрузки!
            for line in f_src:
                f_dst.write(line)
                lines_preserved += 1

        # Атомарная замена файла
        os.replace(tmp_hot_path, file_path)
        logger.info(f"✂️ Файл {file_path} успешно обрезан: удалено {skipped_count} заархивированных строк, сохранено {lines_preserved} новых строк.")
    except Exception as e:
        logger.error(f"❌ Ошибка при обрезке файла {file_path}: {e}")
        if os.path.exists(tmp_hot_path):
            os.remove(tmp_hot_path)
        return False
    finally:
        # Шаг 4. Очистка временного локального архива
        if os.path.exists(tmp_archive_path):
            os.remove(tmp_archive_path)

    return True

def rotate_all(force=False, check_disk=False):
    """Основная процедура ротации всех файлов."""
    free_gb = get_free_disk_gb("/")
    logger.info(f"Проверка свободного места: доступно {free_gb:.2f} GB (порог: {MIN_FREE_DISK_GB} GB).")

    if check_disk and not force:
        if free_gb >= MIN_FREE_DISK_GB:
            logger.info(f"Свободного места достаточно ({free_gb:.2f} GB >= {MIN_FREE_DISK_GB} GB). Аварийная ротация не требуется.")
            return

    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    timestamp_suffix = now_utc.strftime("%H%M%S") if check_disk else ""

    logger.info(f"🚀 Запуск ротации данных для даты {date_str}...")

    try:
        service = get_drive_service()
        folder_id = get_or_create_date_folder(service, FOLDER_ID, date_str)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка подключения к Google Drive: {e}")
        return

    success_count = 0
    for file_info in FILES_TO_ROTATE:
        success = process_single_file(service, folder_id, file_info, date_str, timestamp_suffix)
        if success:
            success_count += 1

    after_free_gb = get_free_disk_gb("/")
    logger.info(f"🎉 Ротация завершена. Успешно обработано: {success_count}/{len(FILES_TO_ROTATE)} файлов. Свободно на диске: {after_free_gb:.2f} GB (освобождено {after_free_gb - free_gb:+.2f} GB).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Суточная и аварийная ротация данных криптовалютного бота.")
    parser.add_argument("--force", action="store_true", help="Принудительно выполнить ротацию прямо сейчас.")
    parser.add_argument("--check-disk", action="store_true", help="Выполнить ротацию только если свободного места меньше MIN_FREE_DISK_GB.")
    args = parser.parse_args()

    # По умолчанию без флагов запускается штатная суточная ротация
    rotate_all(force=args.force or (not args.check_disk), check_disk=args.check_disk)
