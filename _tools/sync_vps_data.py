import paramiko
import os
import sys

HOST = "81.19.137.18"
USER = "root"
PASS = "cbqeoR6392Lr"

REMOTE_DIR = "/opt/crypto_bot/data/realtime/prlusdt"
LOCAL_DIR = "/home/restorator/crypto_bot/data/realtime/prlusdt"

FILES = ["trades.csv", "depth.csv", "collection.log"]

def sync_data():
    print(f"🔗 Подключение к {HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(HOST, username=USER, password=PASS, timeout=10)
        sftp = ssh.open_sftp()
        
        os.makedirs(LOCAL_DIR, exist_ok=True)
        
        for file in FILES:
            remote_path = f"{REMOTE_DIR}/{file}"
            local_path = f"{LOCAL_DIR}/{file.replace('.csv', '_remote.csv')}"
            
            try:
                print(f"📥 Скачивание {file} -> {os.path.basename(local_path)}...")
                sftp.get(remote_path, local_path)
                print(f"✅ Успешно скачан: {file} ({os.path.getsize(local_path) / 1024:.2f} KB)")
            except FileNotFoundError:
                print(f"⚠️ Файл {file} пока не существует на сервере.")
            except Exception as e:
                print(f"❌ Ошибка при скачивании {file}: {e}")
                
        sftp.close()
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
    finally:
        ssh.close()
        print("🔌 Соединение закрыто.")

if __name__ == "__main__":
    sync_data()
