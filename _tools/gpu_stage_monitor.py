#!/usr/bin/env python3
"""
_tools/gpu_stage_monitor.py
=============================================================================
Live GPU & Training Stage Telemetry Logger.
Tracks GPU 0 / GPU 1 utilization, VRAM, temperature, power, and the exact
currently running training stage from the worker logs.
=============================================================================
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "scratch" / "wf_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_LOG = LOGS_DIR / "gpu_stage_telemetry.log"
GPU0_LOG = LOGS_DIR / "gpu0_worker.log"
GPU1_LOG = LOGS_DIR / "gpu1_worker.log"


def get_latest_stage(log_path: Path) -> str:
    if not log_path.exists():
        return "Idle/Not Started"
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Processing Window") or "Processing Window" in line:
                return line
            if "H=" in line and ("XGBoost" in line or "mlp" in line or "wave" in line or "cnn" in line):
                return line
            if "Ensemble" in line:
                return line
            if "Completed in" in line:
                return line
        return "Initializing..."
    except Exception as e:
        return f"Error reading log: {e}"


def get_gpu_metrics():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.check_output(cmd, text=True).strip().splitlines()
        metrics = {}
        for line in res:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                idx = parts[0]
                metrics[idx] = {
                    "util": f"{parts[2]}%",
                    "mem": f"{parts[3]}/{parts[4]}MB",
                    "temp": f"{parts[5]}°C",
                    "power": f"{parts[6]}W"
                }
        return metrics
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 80)
    print("  🚀 Starting Live GPU & Stage Telemetry Monitor")
    print(f"  Telemetry log target: {TELEMETRY_LOG}")
    print("=" * 80, flush=True)

    header = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TELEMETRY MONITOR INITIALIZED\n"
    with open(TELEMETRY_LOG, "a") as f:
        f.write(header)

    try:
        while True:
            ts = datetime.now().strftime("%H:%M:%S")
            gpus = get_gpu_metrics()
            stage0 = get_latest_stage(GPU0_LOG)
            stage1 = get_latest_stage(GPU1_LOG)

            g0 = gpus.get("0", {})
            g1 = gpus.get("1", {})

            entry0 = f"[{ts}][GPU 0] Util: {g0.get('util', 'N/A'):>4} | Mem: {g0.get('mem', 'N/A'):>14} | Temp: {g0.get('temp', 'N/A'):>4} | Pwr: {g0.get('power', 'N/A'):>6} || Stage: {stage0}"
            entry1 = f"[{ts}][GPU 1] Util: {g1.get('util', 'N/A'):>4} | Mem: {g1.get('mem', 'N/A'):>14} | Temp: {g1.get('temp', 'N/A'):>4} | Pwr: {g1.get('power', 'N/A'):>6} || Stage: {stage1}"

            with open(TELEMETRY_LOG, "a") as f:
                f.write(entry0 + "\n")
                f.write(entry1 + "\n")

            print(entry0, flush=True)
            print(entry1, flush=True)
            print("-" * 80, flush=True)

            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
