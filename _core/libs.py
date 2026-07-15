# _core/libs.py
# Импортируй этот модуль ТОЛЬКО там, где реально нужны тяжёлые библиотеки
# (обучение, отбор фич, отчёты). Это ускоряет импорты в лёгких модулях.

# Стандартные
import os, json, math, time, gc, itertools, argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from functools import partial
import multiprocessing
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass
from datetime import date as date_cls
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import lightgbm as lgb



# Научный стек
import numpy as np
import pandas as pd

# Веб/IO
import requests

# TA
try:
    import talib  # системные либы должны стоять в Dockerfile
except Exception:
    talib = None
import pandas_ta as ta  # лёгкая альтернатива TA-Lib

# ML/DS
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
import joblib



# Прогресс-бар
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ---------- УСТОЙЧИВЫЙ HTTP-КЛИЕНТ ДЛЯ ЗАГРУЗОК ----------
def make_http_session() -> requests.Session:
    """
    Создаёт Session с ретраями и 'Connection: close', что сильно снижает
    шанс SSL/ChunkedEncoding ошибок при длинных выгрузках.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Connection": "close"})

    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def http_get_text(session: requests.Session, url: str, *, params=None, timeout: float = 30.0,
                  max_attempts: int = 5, sleep_base: float = 0.5) -> str:
    """
    GET с повторными попытками при сетевых сбоях чтения (SSL/Chunked/Conn).
    Возвращает .text. При неудаче кидает RuntimeError.
    """
    import time
    import requests
    exc = None
    for i in range(max_attempts):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            # если non-200: пусть Retry адаптера уже отработает,
            # здесь просто проверим и, если что, ещё круг
            if resp.status_code == 200:
                # попытка прочитать тело; иногда ошибки вылетают именно здесь
                _ = resp.content  # прогреем чтение, чтобы ловить исключения на этом шаге
                return resp.text
        except (requests.exceptions.SSLError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            exc = e
        # экспоненциальная пауза и "свежая" сессия/соединение
        time.sleep(sleep_base * (2 ** i))
        try:
            session.close()
        except Exception:
            pass
        session = make_http_session()
    raise RuntimeError(f"HTTP GET failed after retries for {url}: {exc}")
# ----------------------------------------------------------
