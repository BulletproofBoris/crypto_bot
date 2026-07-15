from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# config_manager.py

PARAMS_FILENAME = "best_params.json"

def save_params(params_dict):
    """Сохраняет словарь с параметрами в JSON файл."""
    try:
        # Преобразуем numpy типы в стандартные типы Python для сохранения
        for key, value in params_dict.items():
            if hasattr(value, 'item'):
                params_dict[key] = value.item()
        
        with open(PARAMS_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, ensure_ascii=False, indent=4)
        print(f"✅ Лучшие параметры успешно сохранены в файл {PARAMS_FILENAME}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении параметров: {e}")

def load_params(default_params):
    """
    Загружает параметры из JSON файла.
    Если файл не найден или пуст, возвращает параметры по умолчанию.
    """
    if os.path.exists(PARAMS_FILENAME):
        try:
            with open(PARAMS_FILENAME, 'r', encoding='utf-8') as f:
                # Проверяем, что файл не пустой
                content = f.read()
                if content:
                    params = json.loads(content)
                    print(f"✅ Параметры успешно загружены из файла {PARAMS_FILENAME}")
                    return params
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ Не удалось прочитать файл параметров ({e}). Использую параметры по умолчанию.")
            return default_params
    
    print(f"⚠️ Файл {PARAMS_FILENAME} не найден. Использую параметры по умолчанию.")
    return default_params
