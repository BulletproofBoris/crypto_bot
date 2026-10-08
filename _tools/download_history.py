import subprocess
import json
import csv
from pathlib import Path

MARKET = "prlusdt"
HOST_IP = "104.26.12.76"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "historical" / MARKET
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_with_curl(endpoint, params_str):
    url = f"https://safe.trade/api/v2/trade/public/markets/{MARKET}/{endpoint}?{params_str}"
    
    # curl --resolve позволяет подменить IP для домена, но при этом отправить правильный SNI заголовок для Cloudflare
    cmd = [
        "curl", "-s",
        "--resolve", f"safe.trade:443:{HOST_IP}",
        "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Ошибка curl: {result.stderr}")
        return None
        
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"❌ Ошибка парсинга JSON: {result.stdout[:200]}")
        return None

def download_klines():
    print(f"📊 Скачиваем исторические свечи (5 минут) для {MARKET}...")
    data = fetch_with_curl("k-line", "period=5&limit=1000")
    if not data: return
    
    print(f"✅ Получено {len(data)} свечей.")
    file_path = DATA_DIR / "klines_5m.csv"
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in data:
            writer.writerow(row)
    print(f"💾 Свечи сохранены в {file_path}")

def download_trades():
    print(f"\n💸 Скачиваем последние сделки для {MARKET}...")
    data = fetch_with_curl("trades", "limit=1000")
    if not data: return
    
    print(f"✅ Получено {len(data)} сделок.")
    file_path = DATA_DIR / "recent_trades.csv"
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "price", "amount", "total", "side", "created_at"])
        for t in data:
            writer.writerow([t.get('id'), t.get('price'), t.get('amount'), t.get('total'), t.get('side'), t.get('created_at')])
    print(f"💾 Сделки сохранены в {file_path}")

if __name__ == "__main__":
    download_klines()
    download_trades()
