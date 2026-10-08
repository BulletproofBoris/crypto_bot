import os
import subprocess
import sys
from datetime import datetime

DATA_DIR_SAFETRADE = "/opt/crypto_bot/data/realtime/prlusdt"
DATA_DIR_BIGONE = "/opt/crypto_bot/data/realtime/prlusdt_bigone"
TOOLS_DIR = "/opt/crypto_bot/_tools"
MAX_SIZE_MB = 100  # Порог в 100 МБ для срабатывания архивации

def get_size_mb(filepath):
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)

def rotate_and_upload():
    depth_safe = os.path.join(DATA_DIR_SAFETRADE, "depth.csv")
    trades_safe = os.path.join(DATA_DIR_SAFETRADE, "trades.csv")
    
    depth_bigone = os.path.join(DATA_DIR_BIGONE, "depth_bigone.csv")
    trades_bigone = os.path.join(DATA_DIR_BIGONE, "trades_bigone.csv")
    
    # Проверяем, достиг ли хоть один из файлов лимита
    size_safe = get_size_mb(depth_safe)
    size_bigone = get_size_mb(depth_bigone)
    
    if size_safe < MAX_SIZE_MB and size_bigone < MAX_SIZE_MB:
        print(f"Текущий размер (SafeTrade: {size_safe:.1f}MB, BigONE: {size_bigone:.1f}MB) меньше лимита ({MAX_SIZE_MB}MB). Ожидаем.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"prlusdt_backup_{timestamp}.zip"
    archive_path = os.path.join("/tmp", archive_name)
    
    # 1. Останавливаем сборщики
    print("Останавливаем сборщики...")
    subprocess.run(["systemctl", "stop", "safetrade-collector", "bigone-collector"])
    
    # 2. Переименовываем файлы
    files_to_backup = [(depth_safe, depth_safe + ".bak"), 
                       (trades_safe, trades_safe + ".bak"),
                       (depth_bigone, depth_bigone + ".bak"), 
                       (trades_bigone, trades_bigone + ".bak")]
                       
    actual_backups = []
    for orig, bak in files_to_backup:
        if os.path.exists(orig):
            os.rename(orig, bak)
            actual_backups.append(bak)
            
    # 3. Запускаем сборщики обратно
    print("Запускаем сборщики...")
    subprocess.run(["systemctl", "start", "safetrade-collector", "bigone-collector"])
    
    # 4. Архивируем бэкапы
    if not actual_backups:
        print("Нет файлов для архивации.")
        return
        
    print(f"Архивируем данные в {archive_name}...")
    subprocess.run(["zip", "-j", archive_path] + actual_backups)
    
    # 5. Загружаем архив в Google Drive
    print("Загружаем в Google Drive...")
    uploader_script = os.path.join(TOOLS_DIR, "upload_to_gdrive.py")
    result = subprocess.run(["/opt/crypto_bot/venv/bin/python", uploader_script, archive_path], cwd="/opt/crypto_bot")
    
    # 6. Удаляем локальные архивы в случае успеха
    if result.returncode == 0:
        print("Загрузка успешна! Очистка локальных бэкапов...")
        os.remove(archive_path)
        for bak in actual_backups:
            os.remove(bak)
    else:
        print("Ошибка загрузки. Локальные бэкапы сохранены на диске.")

if __name__ == "__main__":
    rotate_and_upload()
