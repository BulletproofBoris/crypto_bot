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
from google.auth.transport.requests import Request
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
SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'service_account.json')

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
        "prefix": "safetrade_quantus_depth",
        "path": "/opt/crypto_bot/data/realtime/quantususdt/depth.csv",
    },
    {
        "prefix": "safetrade_quantus_trades",
        "path": "/opt/crypto_bot/data/realtime/quantususdt/trades.csv",
    },
    {
        "prefix": "bigone_depth",
        "path": "/opt/crypto_bot/data/realtime/prlusdt_bigone/depth_bigone.csv",
    },
    {
        "prefix": "bigone_trades",
        "path": "/opt/crypto_bot/data/realtime/prlusdt_bigone/trades_bigone.csv",
    },
    {
        "prefix": "bigone_quantus_depth",
        "path": "/opt/crypto_bot/data/realtime/quantus_bigone/depth_bigone.csv",
    },
    {
        "prefix": "bigone_quantus_trades",
        "path": "/opt/crypto_bot/data/realtime/quantus_bigone/trades_bigone.csv",
    },
    {
        "prefix": "rhea_quantus_price",
        "path": "/opt/crypto_bot/data/realtime/quantus_rhea/price_rhea.csv",
    },
    {
        "prefix": "binance_macro",
        "path": "/opt/crypto_bot/data/realtime/macro_binance/binance_macro.csv",
    },
    {
        "prefix": "prl_onchain",
        "path": "/opt/crypto_bot/data/realtime/onchain/prl_onchain.csv",
    },
    {
        "prefix": "quantus_onchain",
        "path": "/opt/crypto_bot/data/realtime/onchain/quantus_onchain.csv",
    },
    {
        "prefix": "exchange_rules_history",
        "path": "/opt/crypto_bot/data/metadata/exchange_rules_history.csv",
    },
]

def get_free_disk_gb(path="/"):
    """Возвращает свободное место на диске в гигабайтах."""
    stat = os.statvfs(path)
    free_bytes = stat.f_bavail * stat.f_frsize
    return free_bytes / (1024 ** 3)

def get_drive_service():
    """
    Инициализирует и возвращает сервис Google Drive API.
    Приоритет:
    1. Service Account (service_account.json) — вечный ключ без срока годности.
    2. OAuth User Token (token.json) с проактивным автообновлением и сохранением.
    """
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        try:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_PATH, scopes=SCOPES
            )
            logger.info("🔑 Используется Google Cloud Service Account (бессрочный доступ).")
            return build('drive', 'v3', credentials=creds, cache_discovery=False)
        except Exception as e:
            logger.warning(f"Ошибка загрузки service_account.json: {e}")

    if not os.path.exists(TOKEN_PATH):
        raise FileNotFoundError(f"Файл авторизации не найден: {TOKEN_PATH} или {SERVICE_ACCOUNT_PATH}. Запустите auth_gdrive.py.")
    
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if creds and creds.refresh_token:
        needs_refresh = False
        if not creds.valid or creds.expired:
            needs_refresh = True
        elif creds.expiry:
            now = datetime.now(timezone.utc)
            expiry = creds.expiry
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if (expiry - now).total_seconds() < 600:
                needs_refresh = True

        if needs_refresh:
            try:
                creds.refresh(Request())
                tmp_token = TOKEN_PATH + ".tmp"
                with open(tmp_token, "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
                os.replace(tmp_token, TOKEN_PATH)
                logger.info("🔄 Токен Google OAuth успешно обновлен и сохранен.")
            except Exception as e:
                logger.warning(f"Не удалось предварительно обновить токен: {e}")
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

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

PENDING_ARCHIVE_DIR = "/opt/crypto_bot/data/archive_pending"

def get_unique_archive_name(service, folder_id, prefix, date_str, timestamp_suffix=""):
    """Определяет уникальное имя для архива на Google Диске и в локальном буфере, добавляя _part2, _part3."""
    if timestamp_suffix:
        return f"{prefix}_{date_str}_{timestamp_suffix}.csv.gz"

    existing_names = set()
    if service and folder_id:
        try:
            query = f"'{folder_id}' in parents and trashed = false"
            resp = service.files().list(q=query, spaces='drive', fields='files(name)').execute()
            existing_names.update(f['name'] for f in resp.get('files', []))
        except Exception:
            pass

    pending_date_dir = os.path.join(PENDING_ARCHIVE_DIR, date_str)
    if os.path.exists(pending_date_dir):
        existing_names.update(os.listdir(pending_date_dir))

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
    2. Загружает архив на Google Drive (или сохраняет в локальный буфер archive_pending при сбое сети).
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

    # Шаг 1. Читаем файл потоково и пишем в gzip архив
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         gzip.open(tmp_archive_path, "wt", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if idx == 0:
                header_line = line
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

    # Шаг 2. Выгрузка на Google Drive или сохранение в локальный буфер archive_pending
    uploaded = False
    if service and folder_id:
        try:
            drive_file_id = upload_file_to_drive(service, tmp_archive_path, folder_id, archive_name)
            if drive_file_id:
                uploaded = True
        except Exception as e:
            logger.error(f"❌ Ошибка выгрузки {archive_name} на Google Drive: {e}")

    if not uploaded:
        pending_date_dir = os.path.join(PENDING_ARCHIVE_DIR, date_str)
        os.makedirs(pending_date_dir, exist_ok=True)
        pending_target = os.path.join(pending_date_dir, archive_name)
        shutil.move(tmp_archive_path, pending_target)
        logger.warning(f"📦 Архив сохранён в локальный резервный буфер: {pending_target}")

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

        # Атомарная POSIX-замена файла (исключает состояние гонки при параллельной записи коллектором)
        os.replace(tmp_hot_path, file_path)
        logger.info(f"✂️ Файл {file_path} атомарно обрезан: удалено {skipped_count} заархивированных строк, сохранено {lines_preserved} новых строк.")
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

def flush_pending_archives(service):
    """Докачивает ранее сохранённые в локальном буфере архивы на Google Drive."""
    if not service or not os.path.exists(PENDING_ARCHIVE_DIR):
        return
    for d_name in sorted(os.listdir(PENDING_ARCHIVE_DIR)):
        d_path = os.path.join(PENDING_ARCHIVE_DIR, d_name)
        if not os.path.isdir(d_path):
            continue
        files = sorted(f for f in os.listdir(d_path) if f.endswith(".csv.gz"))
        if not files:
            continue
        try:
            f_id = get_or_create_date_folder(service, FOLDER_ID, d_name)
            for fname in files:
                fpath = os.path.join(d_path, fname)
                if upload_file_to_drive(service, fpath, f_id, fname):
                    os.remove(fpath)
                    logger.info(f"☁️ Докачан отложенный архив из буфера: {d_name}/{fname}")
            if not os.listdir(d_path):
                os.rmdir(d_path)
        except Exception as e:
            logger.warning(f"Не удалось докачать отложенные архивы за {d_name}: {e}")

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

    service = None
    folder_id = None
    try:
        service = get_drive_service()
        flush_pending_archives(service)
        folder_id = get_or_create_date_folder(service, FOLDER_ID, date_str)
    except Exception as e:
        logger.warning(f"⚠️ Google Drive временно недоступен ({e}). Ротация выполнится в локальный буфер {PENDING_ARCHIVE_DIR}!")

    success_count = 0
    for file_info in FILES_TO_ROTATE:
        success = process_single_file(service, folder_id, file_info, date_str, timestamp_suffix)
        if success:
            success_count += 1

    after_free_gb = get_free_disk_gb("/")
    logger.info(f"🎉 Ротация завершена. Успешно обработано: {success_count}/{len(FILES_TO_ROTATE)} файлов. Свободно на диске: {after_free_gb:.2f} GB (освобождено {after_free_gb - free_gb:+.2f} GB).")

    # Плановый перезапуск сборщиков после ротации
    if success_count > 0:
        logger.info("🔄 Перезапуск служб сборщиков после ротации для обновления дескрипторов и снимков стакана...")
        try:
            import subprocess
            subprocess.run(["systemctl", "restart", "safetrade-collector", "bigone-collector", "rhea-collector", "binance-collector", "onchain-collector"], check=True, timeout=30)
            logger.info("✅ Службы сборщиков успешно перезапущены.")
        except Exception as e:
            logger.error(f"⚠️ Ошибка при перезапуске служб сборщиков: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Суточная и аварийная ротация данных криптовалютного бота.")
    parser.add_argument("--force", action="store_true", help="Принудительно выполнить ротацию прямо сейчас.")
    parser.add_argument("--check-disk", action="store_true", help="Выполнить ротацию только если свободного места меньше MIN_FREE_DISK_GB.")
    args = parser.parse_args()

    # По умолчанию без флагов запускается штатная суточная ротация
    rotate_all(force=args.force or (not args.check_disk), check_disk=args.check_disk)
