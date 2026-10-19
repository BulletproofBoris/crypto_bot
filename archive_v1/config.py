import os
from pathlib import Path
from dotenv import load_dotenv

# Базовые пути
BASE_DIR = Path(__file__).parent

# --- НОВЫЕ ГЛОБАЛЬНЫЕ ПУТИ ---
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Загрузка переменных окружения
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Секреты и API
# Здесь можно добавить API ключи для криптобирж (Binance, Bybit и т.д.)
# CCXT_API_KEY = os.getenv("CCXT_API_KEY", "")
# CCXT_SECRET = os.getenv("CCXT_SECRET", "")

# Настройки по умолчанию
DEFAULT_TIMEFRAME = "1d" # Упростили формат