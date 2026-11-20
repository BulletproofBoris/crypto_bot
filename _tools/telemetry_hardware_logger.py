#!/usr/bin/env python3
"""
_tools/telemetry_hardware_logger.py
=============================================================================
Автономный регистратор телеметрии оборудования во время RL обучения:
- Загрузка GPU чипа (%) и VRAM (MiB / %)
- Энергопотребление GPU (Вт) и температура (C)
- Нагрузка CPU (% всех ядер и Load Average)
- Использование оперативной памяти RAM (GiB / %)
Запись в data/logs/hardware_telemetry.csv каждые 2 секунды.
=============================================================================
"""

import os
import sys
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = LOG_DIR / "hardware_telemetry.csv"


def get_mem_info():
    """Чтение /proc/meminfo без внешних зависимостей."""
    mem_total, mem_avail = 0, 0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) * 1024
        mem_used = mem_total - mem_avail
        return mem_used / (1024**3), mem_total / (1024**3), (mem_used / max(1, mem_total)) * 100.0
    except Exception:
        return 0.0, 0.0, 0.0


def get_gpu_info():
    """Опрос nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        parts = [p.strip() for p in out.split(",")]
        gpu_util = float(parts[0])
        vram_used = float(parts[1])
        vram_total = float(parts[2])
        vram_pct = (vram_used / max(1.0, vram_total)) * 100.0
        temp_c = float(parts[3])
        power_w = float(parts[4])
        return gpu_util, vram_used, vram_total, vram_pct, temp_c, power_w
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def main():
    if not CSV_FILE.exists():
        with open(CSV_FILE, "w") as f:
            f.write(
                "timestamp,gpu_util_pct,vram_used_mib,vram_total_mib,vram_pct,gpu_temp_c,gpu_power_w,"
                "cpu_load_1m,cpu_load_5m,ram_used_gib,ram_total_gib,ram_pct\n"
            )

    print(f"📡 Запуск регистрации телеметрии железа в {CSV_FILE}...")
    while True:
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            gpu_util, vram_used, vram_total, vram_pct, temp_c, power_w = get_gpu_info()
            load1, load5, _ = os.getloadavg()
            ram_used, ram_total, ram_pct = get_mem_info()

            row = (
                f"{ts},{gpu_util:.1f},{vram_used:.1f},{vram_total:.1f},{vram_pct:.1f},{temp_c:.1f},{power_w:.1f},"
                f"{load1:.2f},{load5:.2f},{ram_used:.2f},{ram_total:.2f},{ram_pct:.1f}\n"
            )

            with open(CSV_FILE, "a") as f:
                f.write(row)

            time.sleep(2.0)
        except KeyboardInterrupt:
            break
        except Exception as e:
            time.sleep(2.0)


if __name__ == "__main__":
    main()

