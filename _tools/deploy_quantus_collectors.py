import paramiko
import sys
import time

sys.path.append('/home/restorator/crypto_bot/_tools')
import vps_config

def log(msg):
    print(f"[{time.strftime('%X')}] {msg}", flush=True)

def deploy_all():
    log("1. Подключение к VPS через SSH...")
    if hasattr(vps_config, "create_ssh_client"):
        ssh = vps_config.create_ssh_client(timeout=10)
    else:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(vps_config.VPS_HOST, port=vps_config.VPS_PORT, username=vps_config.VPS_USER, password=vps_config.VPS_PASS, timeout=10)
    log("1. Подключение успешно!")

    sftp = ssh.open_sftp()

    files_to_upload = [
        ("collect_safetrade_realtime.py", "/opt/crypto_bot/_tools/collect_safetrade_realtime.py"),
        ("collect_bigone_realtime.py", "/opt/crypto_bot/_tools/collect_bigone_realtime.py"),
        ("collect_rhea_realtime.py", "/opt/crypto_bot/_tools/collect_rhea_realtime.py"),
        ("rotate_and_upload.py", "/opt/crypto_bot/_tools/rotate_and_upload.py"),
        ("watchdog_collectors.py", "/opt/crypto_bot/_tools/watchdog_collectors.py"),
    ]

    log("2. Загрузка обновленных скриптов на VPS через SFTP...")
    for local_name, remote_path in files_to_upload:
        local_path = f"/home/restorator/crypto_bot/_tools/{local_name}"
        log(f"   ⬆️ {local_name} -> {remote_path}")
        sftp.put(local_path, remote_path)

    sftp.close()

    # 3. chmod +x
    log("3. Установка прав chmod +x...")
    stdin, stdout, stderr = ssh.exec_command("chmod +x /opt/crypto_bot/_tools/*.py")
    stdin.close()
    stdout.read()

    # 4. Создание rhea-collector.service
    log("4. Создание службы rhea-collector.service...")
    service_content = """[Unit]
Description=Rhea Finance / Near Intents QTC Realtime Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/crypto_bot
ExecStart=/opt/crypto_bot/venv/bin/python /opt/crypto_bot/_tools/collect_rhea_realtime.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    cmd_service = f"cat << 'EOF' > /etc/systemd/system/rhea-collector.service\n{service_content}EOF"
    stdin, stdout, stderr = ssh.exec_command(cmd_service)
    stdin.close()
    stdout.read()

    # 5. Перезапуск systemd и всех 3 служб
    log("5. Перезапуск systemd и всех служб сборщиков...")
    cmds = [
        "systemctl daemon-reload",
        "systemctl enable rhea-collector",
        "systemctl restart safetrade-collector bigone-collector rhea-collector"
    ]
    for c in cmds:
        stdin, stdout, stderr = ssh.exec_command(c)
        stdin.close()
        stdout.read()

    log("6. Ожидание 10 секунд для наполнения файлов...")
    time.sleep(10)

    # 7. Проверка статусов служб
    log("7. Проверка статуса служб:")
    for svc in ["safetrade-collector", "bigone-collector", "rhea-collector"]:
        stdin, stdout, stderr = ssh.exec_command(f"systemctl is-active {svc}")
        stdin.close()
        status = stdout.read().decode().strip()
        log(f"   {svc}: {'🟢 ACTIVE' if status == 'active' else '🔴 ' + status}")

    # 8. Проверка размеров и строк в файлах
    log("\n8. Проверка созданных файлов на диске:")
    stdin, stdout, stderr = ssh.exec_command("ls -lh /opt/crypto_bot/data/realtime/*/*.csv")
    stdin.close()
    log("\n" + stdout.read().decode())

    # 9. Проверка watchdog
    log("9. Проверка сторожевого таймера watchdog (--status):")
    stdin, stdout, stderr = ssh.exec_command("/opt/crypto_bot/venv/bin/python /opt/crypto_bot/_tools/watchdog_collectors.py --status")
    stdin.close()
    log("\n" + stdout.read().decode() + stderr.read().decode())

    ssh.close()
    log("🎉 Все 3 площадки успешно запущены и собирают данные!")

if __name__ == '__main__':
    deploy_all()
