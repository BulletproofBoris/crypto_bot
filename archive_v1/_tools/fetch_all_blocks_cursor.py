#!/usr/bin/env python3
"""
Постраничная выгрузка блоков Pearl через курсор (limit=10):
- Надежно
- Без превышения лимитов запросов (429)
- Захватывает полную историю с 3 октября (блок 122400) по текущий момент
"""

import os
import sys
import csv
import time
import requests

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

def load_existing():
    existing = {}
    if os.path.exists(OUT_CSV):
        try:
            with open(OUT_CSV, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    h = int(r["block_height"])
                    existing[h] = r
            print(f"Загружено {len(existing)} существующих блоков из кэша {OUT_CSV}", flush=True)
        except Exception as e:
            print(f"Предупреждение при чтении {OUT_CSV}: {e}", flush=True)
    return existing

def fetch_block_tx_details(height):
    url = f"https://api.prlscan.com/v1/blocks/{height}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                txs = data.get("transactions", [])
                non_cb = [t for t in txs if not t.get("is_coinbase")]
                transfer_count = len(non_cb)
                vols = [t.get("output_grains", 0) / 1e8 for t in non_cb]
                vol_sum = sum(vols)
                max_v = max(vols) if vols else 0.0
                whales = [v for v in vols if v >= 5000.0]
                return transfer_count, round(vol_sum, 4), round(max_v, 4), len(whales), round(sum(whales), 4)
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(1)
    return None

def main():
    print(f"=== Запуск постраничной выгрузки Pearl блоков (до высоты {START_HEIGHT}) ===", flush=True)
    existing = load_existing()
    blocks_dict = dict(existing)
    cursor = None
    page = 0
    consecutive_errors = 0
    
    t0 = time.time()
    
    # 1. Быстро выгружаем все заголовки блоков (limit=100)
    print("Этап 1: Получение сводки блоков через постраничный курсор (limit=100)...", flush=True)
    summaries = []
    while True:
        url = "https://api.prlscan.com/v1/blocks?limit=100"
        if cursor:
            url += f"&cursor={cursor}"
            
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                consecutive_errors = 0
                d = r.json()
                items = d.get("items", [])
                cursor = d.get("next_cursor")
                
                if not items:
                    print("Нет больше блоков, завершение постраничного обхода.", flush=True)
                    break
                    
                summaries.extend(items)
                page += 1
                min_h = items[-1].get("height")
                speed = len(summaries) / (time.time() - t0 + 1e-5)
                print(f"Страница {page:>3} | Блоков: {len(summaries):>5} | Текущая высота: {min_h} | Скорость: {speed:.1f} бл/с", flush=True)
                    
                if min_h <= START_HEIGHT:
                    print(f"Достигнута целевая высота {START_HEIGHT}!", flush=True)
                    break
                    
                time.sleep(0.1)
            elif r.status_code == 429:
                print("⚠️ Получен 429 (Rate Limit). Пауза 5 секунд...", flush=True)
                time.sleep(5)
            else:
                print(f"HTTP {r.status_code}. Повтор через 2с...", flush=True)
                time.sleep(2)
        except Exception as e:
            consecutive_errors += 1
            print(f"Ошибка запроса: {e}. Повтор через 2с...", flush=True)
            time.sleep(2)
            if consecutive_errors > 6:
                time.sleep(10)
                
    # 2. Обработка блоков
    print(f"Этап 2: Обработка {len(summaries)} блоков...", flush=True)
    blocks_to_inspect = []
    for it in summaries:
        h = it.get("height")
        if h < START_HEIGHT:
            continue
        ts = it.get("time")
        diff = it.get("difficulty")
        reward = round((it.get("reward_grains") or 0) / 1e8, 4)
        tx_count = it.get("tx_count", 0)
        pool = it.get("miner_label", "Unknown")
        addr = it.get("miner_address", "")
        
        if h in existing and "whale_tx_count_5k" in existing[h]:
            # Уже есть полные данные
            continue
            
        if tx_count <= 1:
            # Только coinbase транзакция
            blocks_dict[h] = {
                "timestamp": ts,
                "block_height": h,
                "difficulty": diff,
                "reward_prl": reward,
                "tx_count": tx_count,
                "transfer_count": 0,
                "transfer_volume_prl": 0.0,
                "max_tx_prl": 0.0,
                "whale_tx_count_5k": 0,
                "whale_volume_5k": 0.0,
                "miner_pool": pool,
                "miner_address": addr
            }
        else:
            # Требуется проверка транзакций
            blocks_to_inspect.append((h, ts, diff, reward, tx_count, pool, addr))

    print(f"Блоков без переводов/уже обработанных: {len(blocks_dict)}. Блоков с переводами для анализа: {len(blocks_to_inspect)}", flush=True)
    
    # 3. Дозагрузка деталей для блоков с переводами (с мягким рейтом)
    inspected = 0
    t_det = time.time()
    for item in blocks_to_inspect:
        h, ts, diff, reward, tx_count, pool, addr = item
        res = fetch_block_tx_details(h)
        if res:
            t_cnt, t_vol, max_v, w_cnt, w_vol = res
        else:
            t_cnt = max(0, tx_count - 1)
            t_vol, max_v, w_cnt, w_vol = 0.0, 0.0, 0, 0.0
            
        blocks_dict[h] = {
            "timestamp": ts,
            "block_height": h,
            "difficulty": diff,
            "reward_prl": reward,
            "tx_count": tx_count,
            "transfer_count": t_cnt,
            "transfer_volume_prl": t_vol,
            "max_tx_prl": max_v,
            "whale_tx_count_5k": w_cnt,
            "whale_volume_5k": w_vol,
            "miner_pool": pool,
            "miner_address": addr
        }
        inspected += 1
        if inspected % 100 == 0 or inspected == len(blocks_to_inspect):
            speed = inspected / (time.time() - t_det + 1e-5)
            print(f"Анализ транзакций: {inspected}/{len(blocks_to_inspect)} | Скорость: {speed:.1f} бл/с", flush=True)
            # Периодически сохраняем чекпоинт
            with open(OUT_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=HEADERS)
                writer.writeheader()
                for bh in sorted(blocks_dict.keys()):
                    writer.writerow(blocks_dict[bh])
        time.sleep(0.08)

    # Итоговое сохранение в CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for bh in sorted(blocks_dict.keys()):
            writer.writerow(blocks_dict[bh])
            
    print(f"✅ Успешно сохранено {len(blocks_dict)} блоков в {OUT_CSV} ({os.path.getsize(OUT_CSV)/(1024*1024):.2f} MB)", flush=True)

if __name__ == "__main__":
    main()
