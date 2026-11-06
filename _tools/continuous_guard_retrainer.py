#!/usr/bin/env python3
"""
_tools/continuous_guard_retrainer.py
=============================================================================
Конвейер циклического непрерывного дообучения нейро-ограничителей (Guard Ensemble):
1. Мониторит накопление свежих тиков LOB книги (data/features/prl_1s_features.parquet).
2. При накоплении >= 24 часов новых данных:
   - Обновляет кэш LOB и разметку 4 таргетов (12s sweep, 60s bleed, 300s crash, 30s exit);
   - Запускает быстрый Warm-Finetune (2–3 эпохи) на GPU RTX 5060 Ti с Focal Loss;
   - Проверяет Macro-AUC на слепом OOS окне (порог >= 0.85);
   - Бесшовно обновляет чекпоинты в models/prl/guard_ensemble/;
   - Опционально загружает обновленные чекпоинты на VPS без рестарта или со soft-reload.
=============================================================================
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.multi_horizon_guard_net import train_guard_models, GUARD_ENSEMBLE_DIR
from _tools.build_600h_guard_dataset import build_guard_dataset

RETRAIN_LOG = BASE_DIR / "data" / "logs" / "guard_continuous_retrain.log"
RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RETRAIN_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ContinuousRetrainer")


def check_and_retrain():
    logger.info("🔍 Проверка актуальности данных LOB книги...")
    feat_parquet = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
    if not feat_parquet.exists():
        logger.warning(f"Файл {feat_parquet} не найден. Пропуск цикла.")
        return

    cache_path = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
    need_rebuild = False
    if not cache_path.exists():
        need_rebuild = True
    else:
        # Проверяем разницу времени изменения
        p_mtime = feat_parquet.stat().st_mtime
        c_mtime = cache_path.stat().st_mtime
        if p_mtime > c_mtime + 86400:  # Появились сутки новых данных
            need_rebuild = True

    if need_rebuild:
        logger.info("⚡ Обнаружены новые данные LOB книги (>= 24ч). Пересборка Causal Guard датасета...")
        build_guard_dataset()
    else:
        logger.info("✅ Кэш LOB книги актуален. Запуск цикла дообучения роя нейро-ограничителей...")

    # Запуск дообучения
    t0 = time.time()
    results = train_guard_models(epochs=4, batch_size=4096)
    elapsed = time.time() - t0

    macro_auc = float(np.mean(list(results.values())))
    logger.info(f"🏁 Дообучение завершено за {elapsed:.1f}с | Macro OOS ROC-AUC: {macro_auc:.4f}")

    # Запись телеметрии дообучения
    telemetry = {
        "timestamp": time.time(),
        "macro_auc": macro_auc,
        "models": results,
        "elapsed_sec": elapsed,
        "status": "SUCCESS" if macro_auc >= 0.85 else "DEGRADED",
    }
    telem_file = GUARD_ENSEMBLE_DIR / "retrain_telemetry.json"
    with open(telem_file, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    logger.info(f"📊 Результаты зафиксированы в {telem_file.name}")


if __name__ == "__main__":
    check_and_retrain()

