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
FINAM_TOKEN = os.getenv("FINAM_API_TOKEN", "").strip('"\'')

if not FINAM_TOKEN:
    print("⚠️ ВНИМАНИЕ: Токен Финама не найден в .env!")

# Настройки по умолчанию
DEFAULT_TIMEFRAME = "1d" # Упростили формат