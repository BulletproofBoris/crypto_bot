import paramiko
import time

import sys
import os
sys.path.append(os.path.dirname(__file__))
import vps_config

host = vps_config.VPS_HOST
port = vps_config.VPS_PORT
user = vps_config.VPS_USER
secret = vps_config.VPS_PASS

print(f"Connecting to {host}:{port} as {user}...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=host, port=port, username=user, password=secret)

def run(cmd):
    print(f"Running: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out: print(out.strip())
    if err: print("ERROR:", err.strip())
    return out

# Wait for apt lock just in case it's a fresh server updating automatically
run("while fuser /var/lib/dpkg/lock >/dev/null 2>&1; do sleep 1; done")

print("1. Installing python and pip...")
run("apt-get update && apt-get install -y python3 python3-pip python3-venv zip")

print("2. Creating directories...")
run("mkdir -p /opt/crypto_bot/_tools /opt/crypto_bot/data/realtime/prlusdt")

print("3. Setting up virtual environment...")
run("python3 -m venv /opt/crypto_bot/venv")
run("/opt/crypto_bot/venv/bin/pip install curl_cffi google-api-python-client google-auth-httplib2 google-auth-oauthlib")

print("4. Uploading script...")
import os
sftp = client.open_sftp()
files = [
    "collect_safetrade_realtime.py", 
    "collect_bigone_realtime.py",
    "upload_to_gdrive.py", 
    "rotate_data.py", 
    "token.json"
]
for f in files:
    src = f"/home/restorator/crypto_bot/_tools/{f}"
    if os.path.exists(src):
        print(f"Uploading {f}...")
        sftp.put(src, f"/opt/crypto_bot/_tools/{f}")
sftp.close()

print("4.5 Setting up cron for automatic rotation...")
run("echo '0 * * * * root /opt/crypto_bot/venv/bin/python /opt/crypto_bot/_tools/rotate_data.py >> /var/log/crypto_rotate.log 2>&1' > /etc/cron.d/crypto_rotate")

print("5. Creating systemd services...")
service_file_safetrade = """[Unit]
Description=SafeTrade Realtime Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/crypto_bot
ExecStart=/opt/crypto_bot/venv/bin/python /opt/crypto_bot/_tools/collect_safetrade_realtime.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
run(f"cat << 'EOF' > /etc/systemd/system/safetrade-collector.service\n{service_file_safetrade}EOF")

service_file_bigone = """[Unit]
Description=BigONE Realtime Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/crypto_bot
ExecStart=/opt/crypto_bot/venv/bin/python /opt/crypto_bot/_tools/collect_bigone_realtime.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
run(f"cat << 'EOF' > /etc/systemd/system/bigone-collector.service\n{service_file_bigone}EOF")

print("6. Starting services...")
run("systemctl daemon-reload")
run("systemctl enable safetrade-collector bigone-collector")
run("systemctl restart safetrade-collector bigone-collector")

time.sleep(2)
print("Deployment complete! Checking status...")
run("systemctl status safetrade-collector --no-pager")
run("systemctl status bigone-collector --no-pager")

client.close()
