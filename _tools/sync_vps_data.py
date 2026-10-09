import paramiko
import os
import sys

import sys
import os
sys.path.append(os.path.dirname(__file__))
import vps_config

HOST = vps_config.VPS_HOST
PORT = vps_config.VPS_PORT
USER = vps_config.VPS_USER
PASS = vps_config.VPS_PASS

REMOTE_DIR = "/opt/crypto_bot/data/realtime/prlusdt"
LOCAL_DIR = "/home/restorator/crypto_bot/data/realtime/prlusdt"

FILES = ["trades.csv", "depth.csv", "collection.log"]

def sync_data():
    print(f"🔗 Подключение к {HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=10)
        sftp = ssh.open_sftp()
        
        # Directories
        dirs_to_sync = [
            {
                "remote": "/opt/crypto_bot/data/realtime/prlusdt",
                "local": "/home/restorator/crypto_bot/data/realtime/prlusdt",
                "files": ["trades.csv", "depth.csv", "collection.log"]
            },
            {
                "remote": "/opt/crypto_bot/data/realtime/prlusdt_bigone",
                "local": "/home/restorator/crypto_bot/data/realtime/prlusdt_bigone",
                "files": ["trades_bigone.csv", "depth_bigone.csv", "collection_bigone.log"]
            }
        ]
        
        def print_progress(transferred, total):
            if total > 0:
                progress = transferred / float(total)
                bar_length = 40
                block = int(round(bar_length * progress))
                text = f"\r⏳ Прогресс: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.1f}% ({transferred/1024/1024:.2f} MB / {total/1024/1024:.2f} MB)"
                sys.stdout.write(text)
                sys.stdout.flush()

        for d in dirs_to_sync:
            os.makedirs(d["local"], exist_ok=True)
            for file in d["files"]:
                remote_path = f"{d['remote']}/{file}"
                local_path = f"{d['local']}/{file.replace('.csv', '_remote.csv')}"
                
                try:
                    print(f"\n📥 Скачивание {file} -> {os.path.basename(local_path)}...")
                    sftp.get(remote_path, local_path, callback=print_progress)
                    print(f"\n✅ Успешно скачан: {file}")
                except FileNotFoundError:
                    print(f"\n⚠️ Файл {file} пока не существует на сервере.")
                except Exception as e:
                    print(f"\n❌ Ошибка при скачивании {file}: {e}")
                
        sftp.close()
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
    finally:
        ssh.close()
        print("🔌 Соединение закрыто.")

if __name__ == "__main__":
    sync_data()
