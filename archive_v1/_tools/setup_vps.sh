#!/bin/bash
set -e

echo "1. Installing python, pip, git, zip..."
apt-get update && apt-get install -y python3 python3-pip python3-venv git zip curl

echo "2. Creating directories and cloning repository..."
rm -rf /opt/crypto_bot
git clone https://github.com/BulletproofBoris/crypto_bot.git /opt/crypto_bot
cd /opt/crypto_bot

echo "3. Fetching credentials..."
curl -s https://termbin.com/3ebjo > _tools/token.json
curl -s https://termbin.com/5l1c > _tools/client_secret.json

echo "4. Setting up virtual environment..."
python3 -m venv venv
venv/bin/pip install curl_cffi google-api-python-client google-auth-httplib2 google-auth-oauthlib

echo "5. Setting up cron job..."
echo '0 * * * * root /opt/crypto_bot/venv/bin/python /opt/crypto_bot/_tools/rotate_data.py >> /var/log/crypto_rotate.log 2>&1' > /etc/cron.d/crypto_rotate

echo "6. Creating systemd services..."
cat << 'EOF' > /etc/systemd/system/safetrade-collector.service
[Unit]
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
EOF

cat << 'EOF' > /etc/systemd/system/bigone-collector.service
[Unit]
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
EOF

echo "7. Starting services..."
systemctl daemon-reload
systemctl enable safetrade-collector bigone-collector
systemctl restart safetrade-collector bigone-collector

sleep 2
echo "Deployment complete! Checking status..."
systemctl status safetrade-collector --no-pager
systemctl status bigone-collector --no-pager
