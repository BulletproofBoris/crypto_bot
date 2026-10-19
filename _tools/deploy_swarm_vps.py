#!/usr/bin/env python3
"""
Скрипт развертывания обновленной микроструктурной триады и обученных моделей на VPS.
"""

import os
import sys
import tarfile
import subprocess
from pathlib import Path
import paramiko

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "_tools"))
import vps_config

def main():
    print("🚀 Подготовка архива моделей триады для загрузки...")
    tar_path = Path("/tmp/swarm_triad_models.tar.gz")
    
    model_dirs = [
        "models/prl/h5s_k3",
        "models/prl/h12s_k3.5",
        "models/prl/h25s_k3",
        "models/quantus/h5s_k3",
        "models/quantus/h12s_k3.5",
        "models/quantus/h25s_k3"
    ]
    
    with tarfile.open(tar_path, "w:gz") as tar:
        for md in model_dirs:
            p = BASE_DIR / md
            if p.exists():
                print(f"  + Добавление {md} в архив...")
                tar.add(p, arcname=md)
            else:
                print(f"  ⚠️ Директория не найдена: {p}")

    print(f"📦 Размер архива моделей: {tar_path.stat().st_size / (1024*1024):.2f} MB")

    print(f"🌐 Подключение к VPS {vps_config.VPS_HOST}:{vps_config.VPS_PORT}...")
    ssh = vps_config.create_ssh_client(timeout=15)
    sftp = ssh.open_sftp()

    print("📁 Загрузка кода микроструктуры и торгового демона...")
    code_files = [
        "_tools/microstructure_features.py",
        "_tools/swarm_router.py",
        "_tools/triad_dispatcher.py",
        "_tools/live_paper_trader.py",
        "_tools/plot_live_operations.py",
    ]
    
    for cf in code_files:
        local_f = BASE_DIR / cf
        remote_f = f"/opt/crypto_bot/{cf}"
        print(f"  ⬆️ Upload {cf} -> {remote_f}")
        sftp.put(str(local_f), remote_f)

    print("📁 Загрузка архива моделей на VPS...")
    sftp.put(str(tar_path), "/opt/crypto_bot/swarm_triad_models.tar.gz")
    sftp.close()

    print("📦 Распаковка моделей на VPS...")
    stdin, stdout, stderr = ssh.exec_command("cd /opt/crypto_bot && tar -xzf swarm_triad_models.tar.gz && rm swarm_triad_models.tar.gz")
    stdout.channel.recv_exit_status()
    print("✅ Модели распакованы!")

    print("🔄 Перезапуск службы crypto-paper-trader...")
    stdin, stdout, stderr = ssh.exec_command("systemctl restart crypto-paper-trader.service")
    stdout.channel.recv_exit_status()

    import time
    time.sleep(3)

    print("📋 Проверка статуса службы:")
    stdin, stdout, stderr = ssh.exec_command("systemctl status crypto-paper-trader.service --no-pager")
    status_out = stdout.read().decode()
    print(status_out)

    print("📜 Последние логи crypto-paper-trader:")
    stdin, stdout, stderr = ssh.exec_command("journalctl -u crypto-paper-trader.service -n 25 --no-pager")
    log_out = stdout.read().decode()
    print(log_out)

    ssh.close()
    if tar_path.exists():
        tar_path.unlink()
    print("🎉 Развертывание на VPS успешно завершено!")

if __name__ == "__main__":
    main()

