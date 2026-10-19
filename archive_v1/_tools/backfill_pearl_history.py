#!/usr/bin/env python3
"""
Ретроспективный сбор ончейн-данных сети Pearl (PRL):
- Сложность (difficulty)
- Награда майнеров (reward_prl)
- Пулы майнинга (miner_pool, miner_address)
- Переводы и транзакции китов (transfer_volume_prl, whale_tx_count_5k, whale_volume_5k)
Период: с 3 октября 2026 (блок 122400) по текущий момент (~блок 126610).
"""

import os
import sys
import csv
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

START_HEIGHT = 122400
OUT_CSV = "/opt/crypto_bot/data/realtime/onchain/pearl_historical_blocks.csv"

HEADERS = [
    "timestamp",
    "block_height",
    "difficulty",
    "reward_prl",
    "tx_count",
    "transfer_count",
    "transfer_volume_prl",
    "max_tx_prl",
    "whale_tx_count_5k",
    "whale_volume_5k",
    "miner_pool",
    "miner_address"
]

def fetch_block(height):
    url = f"https://api.prlscan.com/v1/blocks/{height}"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                data = r.json()
                blk = data.get("block", {})
                txs = data.get("transactions", [])
                
                ts = blk.get("time")
                diff = blk.get("difficulty")
                reward = (blk.get("reward_grains") or 0) / 1e8
                tx_count = blk.get("tx_count", 0)
                miner_pool = blk.get("miner_label", "Unknown")
                miner_addr = blk.get("miner_address", "")
                
                # Анализ транзакций
                non_cb = [t for t in txs if not t.get("is_coinbase")]
                transfer_count = len(non_cb)
                transfer_vols = [t.get("output_grains", 0) / 1e8 for t in non_cb]
                
                transfer_volume = sum(transfer_vols)
                max_tx = max(transfer_vols) if transfer_vols else 0.0
                
                whale_vols = [v for v in transfer_vols if v >= 5000.0]
                whale_count = len(whale_vols)
                whale_vol = sum(whale_vols)
                
                return {
                    "timestamp": ts,
                    "block_height": height,
                    "difficulty": diff,
                    "reward_prl": round(reward, 4),
                    "tx_count": tx_count,
                    "transfer_count": transfer_count,
                    "transfer_volume_prl": round(transfer_volume, 4),
                    "max_tx_prl": round(max_tx, 4),
                    "whale_tx_count_5k": whale_count,
                    "whale_volume_5k": round(whale_vol, 4),
                    "miner_pool": miner_pool,
                    "miner_address": miner_addr
                }
            elif r.status_code == 404:
                return None
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None

def main():
    # 1. Получаем текущую высоту сети
    try:
        r_status = requests.get("https://api.prlscan.com/v1/status", timeout=6)
        tip_height = r_status.json().get("tip_height", 126610)
    except Exception:
        tip_height = 126610
        
    print(f"=== Запуск ретроспективной выгрузки Pearl ончейн: блоки {START_HEIGHT} -> {tip_height} ===", flush=True)
    total_blocks = tip_height - START_HEIGHT + 1
    print(f"Всего блоков для загрузки: {total_blocks}", flush=True)

    heights = list(range(START_HEIGHT, tip_height + 1))
    results = {}
    
    t0 = time.time()
    completed = 0
    
    # Пакетная параллельная загрузка
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_h = {executor.submit(fetch_block, h): h for h in heights}
        for future in as_completed(future_to_h):
            h = future_to_h[future]
            try:
                res = future.result()
                if res:
                    results[h] = res
            except Exception as e:
                pass
            completed += 1
            if completed % 250 == 0 or completed == total_blocks:
                pct = completed / total_blocks * 100
                speed = completed / (time.time() - t0 + 1e-5)
                print(f"Прогресс: {completed}/{total_blocks} ({pct:.1f}%) | Скорость: {speed:.1f} блоков/сек", flush=True)

    print(f"Загрузка завершена за {time.time() - t0:.1f}с. Успешно получено: {len(results)} блоков.", flush=True)
    
    # Запись в хронологическом порядке
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for h in sorted(results.keys()):
            writer.writerow(results[h])
            
    print(f"✅ Данные сохранены в {OUT_CSV} ({os.path.getsize(OUT_CSV) / (1024*1024):.2f} MB)", flush=True)

if __name__ == "__main__":
    main()
