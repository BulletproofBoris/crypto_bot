#!/usr/bin/env python3
"""
Единый инструмент синхронизации и консолидации рыночных данных.
Архитектура (Lambda-хранилище):
1. Холодный слой: инкрементальное скачивание исторических архивов (.csv.gz) с Google Drive в data/cache_gdrive/.
2. Горячий слой: получение оперативного хвоста текущих данных с VPS через SFTP в data/vps_hot/.
3. Склейка и стандартизация:
   - Приведение всех временных меток к единому UTC pd.to_datetime.
   - Удаление дубликатов (drop_duplicates) и хронологическая сортировка.
   - Сохранение в ультрабыстрый формат Parquet (data/processed/*.parquet).
4. Простой Python API для импорта в Jupyter:
   from _tools.sync_all_data import get_market_data
   depth_safe, trades_safe, depth_bigone, trades_bigone = get_market_data()
"""

import os
import io
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("sync_all_data")

# Директории проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_GDRIVE_DIR = os.path.join(BASE_DIR, "data", "cache_gdrive")
HOT_VPS_DIR = os.path.join(BASE_DIR, "data", "vps_hot")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")

# Google Drive настройки
FOLDER_ID = '1-qWy2q2nYoZtSCiMDZ1DiKZ5roBqXx-c'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token.json')

# Импорт конфигурации VPS
try:
    from _tools.vps_config import VPS_HOST, VPS_PORT, VPS_USER, VPS_PASS
except ImportError:
    # Попытка относительного импорта
    from vps_config import VPS_HOST, VPS_PORT, VPS_USER, VPS_PASS

REMOTE_FILES = [
    # PRL/USDT
    {
        "remote": "/opt/crypto_bot/data/realtime/prlusdt/depth.csv",
        "local": os.path.join(HOT_VPS_DIR, "safetrade_depth_hot.csv"),
        "name": "SafeTrade Depth Hot (PRL)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/prlusdt/trades.csv",
        "local": os.path.join(HOT_VPS_DIR, "safetrade_trades_hot.csv"),
        "name": "SafeTrade Trades Hot (PRL)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/prlusdt_bigone/depth_bigone.csv",
        "local": os.path.join(HOT_VPS_DIR, "bigone_depth_hot.csv"),
        "name": "BigONE Depth Hot (PRL)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/prlusdt_bigone/trades_bigone.csv",
        "local": os.path.join(HOT_VPS_DIR, "bigone_trades_hot.csv"),
        "name": "BigONE Trades Hot (PRL)"
    },
    # QUANTUS / QTC
    {
        "remote": "/opt/crypto_bot/data/realtime/quantususdt/depth.csv",
        "local": os.path.join(HOT_VPS_DIR, "safetrade_quantus_depth_hot.csv"),
        "name": "SafeTrade Depth Hot (Quantus)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/quantususdt/trades.csv",
        "local": os.path.join(HOT_VPS_DIR, "safetrade_quantus_trades_hot.csv"),
        "name": "SafeTrade Trades Hot (Quantus)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/quantus_bigone/depth_bigone.csv",
        "local": os.path.join(HOT_VPS_DIR, "bigone_quantus_depth_hot.csv"),
        "name": "BigONE Depth Hot (Quantus)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/quantus_bigone/trades_bigone.csv",
        "local": os.path.join(HOT_VPS_DIR, "bigone_quantus_trades_hot.csv"),
        "name": "BigONE Trades Hot (Quantus)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/quantus_rhea/price_rhea.csv",
        "local": os.path.join(HOT_VPS_DIR, "rhea_quantus_price_hot.csv"),
        "name": "Rhea Finance Price Hot (Quantus)"
    },
    # TSC/USDT (SafeTrade)
    {
        "remote": "/opt/crypto_bot/data/realtime/tscusdt/depth.csv",
        "local": os.path.join(HOT_VPS_DIR, "safetrade_tsc_depth_hot.csv"),
        "name": "SafeTrade Depth Hot (TSC)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/tscusdt/trades.csv",
        "local": os.path.join(HOT_VPS_DIR, "safetrade_tsc_trades_hot.csv"),
        "name": "SafeTrade Trades Hot (TSC)"
    },
    # Binance Macro Leading Indicators (BTC + NEAR)
    {
        "remote": "/opt/crypto_bot/data/realtime/macro_binance/binance_macro.csv",
        "local": os.path.join(HOT_VPS_DIR, "binance_macro_hot.csv"),
        "name": "Binance Macro Hot (BTC + NEAR)"
    },
    # On-chain Data
    {
        "remote": "/opt/crypto_bot/data/realtime/onchain/prl_onchain.csv",
        "local": os.path.join(HOT_VPS_DIR, "prl_onchain_hot.csv"),
        "name": "Pearl On-chain Hot (PRL)"
    },
    {
        "remote": "/opt/crypto_bot/data/realtime/onchain/quantus_onchain.csv",
        "local": os.path.join(HOT_VPS_DIR, "quantus_onchain_hot.csv"),
        "name": "Quantus On-chain Hot (QTC)"
    },
    # Metadata & Rules
    {
        "remote": "/opt/crypto_bot/data/metadata/exchange_rules.json",
        "local": os.path.join(METADATA_DIR, "exchange_rules.json"),
        "name": "Exchange Rules JSON"
    },
    {
        "remote": "/opt/crypto_bot/data/metadata/exchange_rules_history.csv",
        "local": os.path.join(METADATA_DIR, "exchange_rules_history.csv"),
        "name": "Exchange Rules History CSV"
    },
]

DATASET_CONFIGS = [
    # PRL Datasets
    {
        "key": "safetrade_depth",
        "coin": "prl",
        "cache_pattern": "safetrade_depth_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "safetrade_depth_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "safetrade_depth.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "prl", "safetrade_depth.parquet"),
        "dedup_key": "timestamp",
    },
    {
        "key": "safetrade_trades",
        "coin": "prl",
        "cache_pattern": "safetrade_trades_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "safetrade_trades_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "safetrade_trades.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "prl", "safetrade_trades.parquet"),
        "dedup_key": "id",
    },
    {
        "key": "bigone_depth",
        "coin": "prl",
        "cache_pattern": "bigone_depth_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "bigone_depth_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "bigone_depth.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "prl", "bigone_depth.parquet"),
        "dedup_key": "timestamp",
    },
    {
        "key": "bigone_trades",
        "coin": "prl",
        "cache_pattern": "bigone_trades_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "bigone_trades_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "bigone_trades.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "prl", "bigone_trades.parquet"),
        "dedup_key": "id",
    },
    {
        "key": "prl_onchain",
        "coin": "prl",
        "cache_pattern": "prl_onchain_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "prl_onchain_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "prl_onchain.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "prl", "onchain.parquet"),
        "dedup_key": "timestamp",
    },
    # Quantus Datasets
    {
        "key": "safetrade_quantus_depth",
        "coin": "quantus",
        "cache_pattern": "safetrade_quantus_depth_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "safetrade_quantus_depth_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "safetrade_quantus_depth.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "quantus", "safetrade_depth.parquet"),
        "dedup_key": "timestamp",
    },
    {
        "key": "safetrade_quantus_trades",
        "coin": "quantus",
        "cache_pattern": "safetrade_quantus_trades_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "safetrade_quantus_trades_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "safetrade_quantus_trades.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "quantus", "safetrade_trades.parquet"),
        "dedup_key": "id",
    },
    {
        "key": "bigone_quantus_depth",
        "coin": "quantus",
        "cache_pattern": "bigone_quantus_depth_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "bigone_quantus_depth_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "bigone_quantus_depth.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "quantus", "bigone_depth.parquet"),
        "dedup_key": "timestamp",
    },
    {
        "key": "bigone_quantus_trades",
        "coin": "quantus",
        "cache_pattern": "bigone_quantus_trades_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "bigone_quantus_trades_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "bigone_quantus_trades.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "quantus", "bigone_trades.parquet"),
        "dedup_key": "id",
    },
    {
        "key": "rhea_quantus_price",
        "coin": "quantus",
        "cache_pattern": "rhea_quantus_price_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "rhea_quantus_price_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "rhea_quantus_price.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "quantus", "rhea_price.parquet"),
        "dedup_key": "timestamp",
    },
    {
        "key": "quantus_onchain",
        "coin": "quantus",
        "cache_pattern": "quantus_onchain_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "quantus_onchain_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "quantus_onchain.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "quantus", "onchain.parquet"),
        "dedup_key": "timestamp",
    },
    # TSC Datasets
    {
        "key": "safetrade_tsc_depth",
        "coin": "tsc",
        "cache_pattern": "safetrade_tsc_depth_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "safetrade_tsc_depth_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "safetrade_tsc_depth.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "tsc", "safetrade_depth.parquet"),
        "dedup_key": "timestamp",
    },
    {
        "key": "safetrade_tsc_trades",
        "coin": "tsc",
        "cache_pattern": "safetrade_tsc_trades_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "safetrade_tsc_trades_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "safetrade_tsc_trades.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "tsc", "safetrade_trades.parquet"),
        "dedup_key": "id",
    },
    # Binance Macro Dataset
    {
        "key": "binance_macro",
        "coin": "macro",
        "cache_pattern": "binance_macro_*.csv.gz",
        "hot_file": os.path.join(HOT_VPS_DIR, "binance_macro_hot.csv"),
        "parquet_file": os.path.join(PROCESSED_DIR, "binance_macro.parquet"),
        "hierarchical_file": os.path.join(PROCESSED_DIR, "macro", "binance_bbo.parquet"),
        "dedup_key": "timestamp",
    },
]

def ensure_directories():
    os.makedirs(CACHE_GDRIVE_DIR, exist_ok=True)
    os.makedirs(HOT_VPS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    for c in ["prl", "quantus", "tsc", "macro"]:
        os.makedirs(os.path.join(PROCESSED_DIR, c), exist_ok=True)

# ==============================================================================
# 1. СИНХРОНИЗАЦИЯ С GOOGLE DRIVE
# ==============================================================================

def get_drive_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    import socket
    socket.setdefaulttimeout(30.0)
    if not os.path.exists(TOKEN_PATH):
        raise FileNotFoundError(f"Файл токена не найден: {TOKEN_PATH}. Запустите auth_gdrive.py.")
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

def sync_gdrive():
    """Скачивает все недостающие архивы .csv.gz из Google Drive в локальный кэш с повторными попытками."""
    from googleapiclient.http import MediaIoBaseDownload
    import socket
    socket.setdefaulttimeout(30.0)
    logger.info("☁️ Проверка обновлений на Google Drive...")
    ensure_directories()
    
    service = None
    items = []
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            service = get_drive_service()
            query = f"'{FOLDER_ID}' in parents and trashed = false"
            items = service.files().list(q=query, fields='files(id, name, mimeType, size)').execute().get('files', [])
            break
        except Exception as e:
            if attempt == max_retries:
                raise
            logger.warning(f"Попытка {attempt}/{max_retries} подключения к Google Drive завершилась с ошибкой: {e}. Повтор через 3с...")
            time.sleep(3)

    files_to_download = []

    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # Сканируем содержимое папки даты
            sub_query = f"'{item['id']}' in parents and trashed = false"
            try:
                sub_files = service.files().list(q=sub_query, fields='files(id, name, mimeType, size)').execute().get('files', [])
                for sf in sub_files:
                    if sf['name'].endswith('.csv.gz'):
                        files_to_download.append(sf)
            except Exception as e:
                logger.warning(f"Ошибка при чтении папки {item.get('name')}: {e}")
        elif item['name'].endswith('.csv.gz'):
            files_to_download.append(item)

    logger.info(f"Найдено архивов на Google Drive: {len(files_to_download)}")
    downloaded_count = 0

    for f in files_to_download:
        fname = f['name']
        local_path = os.path.join(CACHE_GDRIVE_DIR, fname)
        remote_size = int(f.get('size', 0))

        # Проверяем кэш: если файл уже есть и совпадает по размеру, пропускаем
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            if local_size == remote_size:
                continue

        logger.info(f"📥 Скачивание архива: {fname} ({remote_size / 1024:.1f} KB)...")
        success = False
        for dl_attempt in range(1, 4):
            try:
                request = service.files().get_media(fileId=f['id'])
                with io.FileIO(local_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                success = True
                downloaded_count += 1
                break
            except Exception as e:
                logger.warning(f"Ошибка при скачивании {fname} (попытка {dl_attempt}/3): {e}")
                time.sleep(2)
        if not success:
            logger.error(f"Не удалось скачать {fname} после 3 попыток.")

    # Проверяем и подтягиваем legacy zip-бэкапы (PRL за 10-11 октября)
    try:
        try:
            from _tools.extract_legacy_backups import process_legacy_backups
        except ImportError:
            from extract_legacy_backups import process_legacy_backups
        process_legacy_backups()
    except Exception as e:
        logger.warning(f"Ошибка при обработке legacy бэкапов: {e}")

    logger.info(f"✅ Синхронизация Google Drive завершена (скачано новых архивов: {downloaded_count}).")

# ==============================================================================
# 2. СИНХРОНИЗАЦИЯ С VPS (ГОРЯЧИЙ СРЕЗ)
# ==============================================================================

def print_progress(transferred, total):
    if total > 0:
        progress = transferred / float(total)
        bar_len = 35
        block = int(round(bar_len * progress))
        text = f"\r⏳ [{('=' * block) + (' ' * (bar_len - block))}] {progress*100:5.1f}% ({transferred/(1024*1024):.1f}/{total/(1024*1024):.1f} MB)"
        sys.stdout.write(text)
        sys.stdout.flush()

def sync_vps_hot():
    """Скачивает текущие горячие файлы с сервера VPS через SFTP с таймаутом."""
    import paramiko
    import socket
    logger.info(f"⚡ Подключение к VPS ({VPS_HOST}:{VPS_PORT}) для выгрузки горячего хвоста...")
    ensure_directories()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    try:
        sock.connect((VPS_HOST, VPS_PORT))
    except Exception as e:
        logger.warning(f"Не удалось подключиться к VPS {VPS_HOST}:{VPS_PORT} (таймаут/сеть): {e}")
        sock.close()
        return

    transport = paramiko.Transport(sock)
    transport.banner_timeout = 10.0
    try:
        transport.connect(username=VPS_USER, password=VPS_PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)

        for item in REMOTE_FILES:
            remote_path = item["remote"]
            local_path = item["local"]
            name = item["name"]

            try:
                r_stat = sftp.stat(remote_path)
                r_size_mb = r_stat.st_size / (1024 * 1024)
                sys.stdout.write(f"\nЗагрузка {name} ({r_size_mb:.2f} MB):\n")
                sftp.get(remote_path, local_path, callback=print_progress)
                sys.stdout.write("\n")
            except Exception as e:
                logger.warning(f"Не удалось получить {remote_path} с VPS: {e}")

        sftp.close()
    except Exception as e:
        logger.warning(f"Ошибка во время передачи данных с VPS: {e}")
    finally:
        transport.close()

    logger.info("✅ Горячие данные с VPS успешно обновлены.")

# ==============================================================================
# 3. ОБЪЕДИНЕНИЕ, ОЧИСТКА И ПРЕОБРАЗОВАНИЕ В PARQUET
# ==============================================================================

def standardize_timestamps(df):
    """Приводит колонку timestamp к единому объекту UTC datetime."""
    if 'timestamp' not in df.columns:
        return df

    if df['timestamp'].dtype in [np.float64, np.int64]:
        # Миллисекунды SafeTrade
        if df['timestamp'].max() > 20000000000:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    else:
        # Строковые метки (ISO8601)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='mixed')

    return df

def build_dataset_for_config(cfg, include_vps=True):
    """Объединяет все архивы из кэша и горячий файл VPS в единый датасет."""
    key = cfg["key"]
    pattern = os.path.join(CACHE_GDRIVE_DIR, cfg["cache_pattern"])
    archive_files = sorted(glob.glob(pattern))

    dfs = []

    # 1. Читаем все архивы из кэша
    for arch_file in archive_files:
        try:
            df_part = pd.read_csv(arch_file)
            if not df_part.empty:
                dfs.append(df_part)
        except Exception as e:
            logger.warning(f"Ошибка чтения архива {arch_file}: {e}")

    # 2. Читаем горячий файл с VPS
    if include_vps and os.path.exists(cfg["hot_file"]):
        try:
            df_hot = pd.read_csv(cfg["hot_file"])
            if not df_hot.empty:
                dfs.append(df_hot)
        except Exception as e:
            logger.warning(f"Ошибка чтения горячего файла {cfg['hot_file']}: {e}")

    if not dfs:
        logger.warning(f"Нет данных для {key}.")
        return pd.DataFrame()

    # 3. Конкатенация
    full_df = pd.concat(dfs, ignore_index=True)

    # 4. Приведение временных меток
    full_df = standardize_timestamps(full_df)

    # 5. Дедупликация
    dedup_key = cfg["dedup_key"]
    if dedup_key in full_df.columns:
        full_df = full_df.drop_duplicates(subset=[dedup_key], keep='last')
    elif 'timestamp' in full_df.columns:
        full_df = full_df.drop_duplicates(subset=['timestamp'], keep='last')

    # 6. Хронологическая сортировка
    if 'timestamp' in full_df.columns:
        full_df = full_df.sort_values('timestamp').reset_index(drop=True)

    # 7. Сохранение в Parquet
    parquet_path = cfg["parquet_file"]
    full_df.to_parquet(parquet_path, index=False)
    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)

    # Сохраняем в структурированную иерархическую папку (data/processed/{coin}/...)
    if "hierarchical_file" in cfg:
        h_path = cfg["hierarchical_file"]
        os.makedirs(os.path.dirname(h_path), exist_ok=True)
        full_df.to_parquet(h_path, index=False)

    logger.info(f"📊 {key}: {len(full_df):,} записей сохранены в {os.path.basename(parquet_path)} ({file_size_mb:.2f} MB)")
    if 'timestamp' in full_df.columns and not full_df.empty:
        t_min = full_df['timestamp'].min()
        t_max = full_df['timestamp'].max()
        logger.info(f"   Диапазон: {t_min} -> {t_max}")

    return full_df

def sync_and_process_all(include_vps=True):
    """Выполняет полный пайплайн синхронизации."""
    # 1. Синхронизируем Google Drive
    try:
        sync_gdrive()
    except Exception as e:
        logger.error(f"Ошибка при обращении к Google Drive: {e}")

    # 2. Синхронизируем VPS (если включено)
    if include_vps:
        try:
            sync_vps_hot()
        except Exception as e:
            logger.error(f"Ошибка при скачивании данных с VPS: {e}")

    # 3. Склеиваем и создаем Parquet-датасеты
    logger.info("⚙️ Сборка и стандартизация датасетов...")
    results = {}
    for cfg in DATASET_CONFIGS:
        results[cfg["key"]] = build_dataset_for_config(cfg, include_vps=include_vps)

    return (
        results.get("safetrade_depth", pd.DataFrame()),
        results.get("safetrade_trades", pd.DataFrame()),
        results.get("bigone_depth", pd.DataFrame()),
        results.get("bigone_trades", pd.DataFrame()),
    )

# ==============================================================================
# ПУБЛИЧНЫЙ API ДЛЯ JUPYTER НОУТБУКА
# ==============================================================================

def load_dataset(key):
    """
    Быстрая загрузка любого датасета по ключу из data/processed/{key}.parquet.
    Ключи:
      - 'safetrade_depth', 'safetrade_trades' (PRL)
      - 'bigone_depth', 'bigone_trades' (PRL)
      - 'safetrade_quantus_depth', 'safetrade_quantus_trades' (Quantus)
      - 'bigone_quantus_depth', 'bigone_quantus_trades' (Quantus / QTC)
      - 'rhea_quantus_price' (Quantus on Rhea Finance / Near Intents)
      - 'safetrade_tsc_depth', 'safetrade_tsc_trades' (TSC)
      - 'binance_macro' (BTC/USDT + NEAR/USDT 1s snapshots)
    """
    path = os.path.join(PROCESSED_DIR, f"{key}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    logger.warning(f"Датасет {key} не найден по пути {path}")
    return pd.DataFrame()

def get_all_market_data():
    """Возвращает словарь всех доступных DataFrames из Parquet кэша."""
    out = {}
    for cfg in DATASET_CONFIGS:
        out[cfg["key"]] = load_dataset(cfg["key"])
    return out

def get_market_data(force_sync=True, include_vps=True):
    """
    Основная функция для ноутбука (PRL).
    Возвращает 4 чистых DataFrame:
    (depth_safetrade, trades_safetrade, depth_bigone, trades_bigone)
    
    Если force_sync=False и parquet файлы уже существуют, быстро загружает их из кэша.
    """
    parquet_files = [cfg["parquet_file"] for cfg in DATASET_CONFIGS[:4]]
    all_exist = all(os.path.exists(p) for p in parquet_files)

    if not force_sync and all_exist:
        logger.info("⚡ Быстрая загрузка датасетов из локального Parquet кэша...")
        return (
            pd.read_parquet(DATASET_CONFIGS[0]["parquet_file"]),
            pd.read_parquet(DATASET_CONFIGS[1]["parquet_file"]),
            pd.read_parquet(DATASET_CONFIGS[2]["parquet_file"]),
            pd.read_parquet(DATASET_CONFIGS[3]["parquet_file"]),
        )

    return sync_and_process_all(include_vps=include_vps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Синхронизация и консолидация данных с Google Drive и VPS.")
    parser.add_argument("--no-vps", action="store_true", help="Не скачивать горячие данные с VPS (только Google Drive).")
    args = parser.parse_args()

    sync_and_process_all(include_vps=not args.no_vps)
