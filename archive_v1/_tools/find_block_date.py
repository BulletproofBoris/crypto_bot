#!/usr/bin/env python3
import requests
import sys

def find_block_for_target_time(target_iso="2026-10-03T00:00:00Z"):
    # Binary search or probe
    # Known:
    # 125600 -> 2026-10-09T23:38:06Z
    # 120000 -> 2026-09-2...
    low = 120000
    high = 125600
    
    print(f"Поиск блока для времени {target_iso} между {low} и {high}...", flush=True)
    
    for h in [122000, 122500, 123000, 123500, 124000]:
        try:
            r = requests.get(f"https://api.prlscan.com/v1/blocks/{h}", timeout=6)
            b = r.json().get('block', {})
            print(f"Высота {h}: {b.get('time')}", flush=True)
        except Exception as e:
            print(f"Ошибка {h}: {e}", flush=True)

if __name__ == "__main__":
    find_block_for_target_time()
