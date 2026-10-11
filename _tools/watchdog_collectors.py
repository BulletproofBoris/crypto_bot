#!/usr/bin/env python3
"""
Сторожевой таймер (Watchdog / Healthcheck) для сборщиков рыночных данных.
Проверяет:
1. Статус служб systemd (safetrade-collector, bigone-collector).
2. Свежесть файлов стаканов (mtime): если данные не поступали более 90 секунд,
   выполняет принудительный перезапуск службы.
3. Защита от частого перезапуска (cooldown / anti-flapping).
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("watchdog")

# Максимальный допустимый возраст данных в секундах
MAX_STALE_SECONDS = 90

# Защита от бесконечного перезапуска: макс. 3 перезапуска за 15 минут
MAX_RESTARTS = 3
WINDOW_SECONDS = 900
STATE_FILE = "/tmp/crypto_watchdog_state.json"

TARGETS = [
    {
        "service": "safetrade-collector",
        "file": "/opt/crypto_bot/data/realtime/prlusdt/depth.csv",
        "name": "SafeTrade PRL"
    },
    {
        "service": "safetrade-collector",
        "file": "/opt/crypto_bot/data/realtime/quantususdt/depth.csv",
        "name": "SafeTrade Quantus"
    },
    {
        "service": "safetrade-collector",
        "file": "/opt/crypto_bot/data/realtime/tscusdt/depth.csv",
        "name": "SafeTrade TSC"
    },
    {
        "service": "bigone-collector",
        "file": "/opt/crypto_bot/data/realtime/prlusdt_bigone/depth_bigone.csv",
        "name": "BigONE PRL"
    },
    {
        "service": "bigone-collector",
        "file": "/opt/crypto_bot/data/realtime/quantus_bigone/depth_bigone.csv",
        "name": "BigONE Quantus"
    },
    {
        "service": "rhea-collector",
        "file": "/opt/crypto_bot/data/realtime/quantus_rhea/price_rhea.csv",
        "name": "Rhea Finance Quantus"
    },
    {
        "service": "binance-collector",
        "file": "/opt/crypto_bot/data/realtime/macro_binance/binance_macro.csv",
        "name": "Binance Macro"
    },
    {
        "service": "onchain-collector",
        "file": "/opt/crypto_bot/data/realtime/onchain/prl_onchain.csv",
        "name": "Pearl On-chain"
    },
    {
        "service": "onchain-collector",
        "file": "/opt/crypto_bot/data/realtime/onchain/quantus_onchain.csv",
        "name": "Quantus On-chain"
    }
]

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Не удалось сохранить состояние watchdog: {e}")

def can_restart(service, state):
    now = time.time()
    history = state.get(service, [])
    # Очищаем устаревшие записи
    history = [t for t in history if now - t < WINDOW_SECONDS]
    state[service] = history
    return len(history) < MAX_RESTARTS

def record_restart(service, state):
    state.setdefault(service, []).append(time.time())
    save_state(state)

def is_service_active(service):
    res = subprocess.run(["systemctl", "is-active", service], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.stdout.strip() == "active"

def restart_service(service):
    logger.warning(f"🚨 ПРИНУДИТЕЛЬНЫЙ ПЕРЕЗАПУСК СЛУЖБЫ: {service}...")
    res = subprocess.run(["systemctl", "restart", service], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        logger.info(f"✅ Служба {service} успешно перезапущена.")
        return True
    else:
        logger.error(f"❌ Ошибка перезапуска {service}: {res.stderr.strip()}")
        return False

def check_target(target, state, verbose=False):
    service = target["service"]
    fpath = target["file"]
    name = target["name"]
    now = time.time()

    active = is_service_active(service)
    if not active:
        logger.warning(f"⚠️ {name}: служба {service} НЕ активна (is-active != active)!")
        if can_restart(service, state):
            if restart_service(service):
                record_restart(service, state)
        else:
            logger.error(f"🛑 {name}: превышен лимит перезапусков ({MAX_RESTARTS} за 15 мин). Cooldown.")
        return False

    if not os.path.exists(fpath):
        logger.warning(f"⚠️ {name}: файл данных {fpath} не существует!")
        if can_restart(service, state):
            if restart_service(service):
                record_restart(service, state)
        return False

    try:
        mtime = os.path.getmtime(fpath)
        age_sec = now - mtime
        file_size_kb = os.path.getsize(fpath) / 1024.0

        if verbose:
            logger.info(f"ℹ️ {name}: служба активна, файл {fpath} ({file_size_kb:.1f} KB), возраст данных: {age_sec:.1f} сек.")

        if age_sec > MAX_STALE_SECONDS:
            logger.warning(f"⚠️ {name}: данные зависли! Файл не обновлялся {age_sec:.1f} сек (порог: {MAX_STALE_SECONDS} сек).")
            if can_restart(service, state):
                if restart_service(service):
                    record_restart(service, state)
            else:
                logger.error(f"🛑 {name}: превышен лимит перезапусков ({MAX_RESTARTS} за 15 мин). Cooldown.")
            return False

        return True
    except Exception as e:
        logger.error(f"❌ Ошибка проверки {name}: {e}")
        return False

def main():
    verbose = "--status" in sys.argv or "-v" in sys.argv
    state = load_state()

    all_ok = True
    for target in TARGETS:
        ok = check_target(target, state, verbose=verbose)
        if not ok:
            all_ok = False

    if verbose and all_ok:
        logger.info("🟢 Все службы и потоки данных работают в штатном режиме.")

if __name__ == "__main__":
    main()
