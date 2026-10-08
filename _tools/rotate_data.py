import os
import subprocess
import sys
from datetime import datetime

DATA_DIR = "/opt/crypto_bot/data/realtime/prlusdt"
TOOLS_DIR = "/opt/crypto_bot/_tools"
MAX_SIZE_MB = 100  # Порог в 100 МБ для срабатывания архивации

def get_size_mb(filepath):
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)

def rotate_and_upload():
    depth_file = os.path.join(DATA_DIR, "depth.csv")
    trades_file = os.path.join(DATA_DIR, "trades.csv")
    
    # Проверяем, достиг ли файл лимита
    current_size = get_size_mb(depth_file)
    if current_size < MAX_SIZE_MB:
        print(f"Текущий размер {current_size:.1f}MB меньше лимита ({MAX_SIZE_MB}MB). Ожидаем.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"prlusdt_backup_{timestamp}.zip"
    archive_path = os.path.join(DATA_DIR, archive_name)
    
    # 1. Останавливаем сборщик на пару секунд
    print("Останавливаем сборщик...")
    subprocess.run(["systemctl", "stop", "safetrade-collector"])
    
    # 2. Переименовываем файлы
    depth_backup = depth_file + ".bak"
    trades_backup = trades_file + ".bak"
    if os.path.exists(depth_file):
        os.rename(depth_file, depth_backup)
    if os.path.exists(trades_file):
        os.rename(trades_file, trades_backup)
        
    # 3. Запускаем сборщик обратно (он создаст новые чистые файлы с заголовками)
    print("Запускаем сборщик...")
    subprocess.run(["systemctl", "start", "safetrade-collector"])
    
    # 4. Архивируем бэкапы
    print(f"Архивируем данные в {archive_name}...")
    subprocess.run(["zip", "-j", archive_path, depth_backup, trades_backup])
    
    # 5. Загружаем архив в Google Drive
    print("Загружаем в Google Drive...")
    uploader_script = os.path.join(TOOLS_DIR, "upload_to_gdrive.py")
    result = subprocess.run(["/opt/crypto_bot/venv/bin/python", uploader_script, archive_path], cwd="/opt/crypto_bot")
    
    # 6. Удаляем локальные архивы в случае успеха
    if result.returncode == 0:
        print("Загрузка успешна! Очистка локальных бэкапов...")
        os.remove(archive_path)
        os.remove(depth_backup)
        os.remove(trades_backup)
    else:
        print("Ошибка загрузки. Локальные бэкапы сохранены на диске.")

if __name__ == "__main__":
    rotate_and_upload()
