#!/usr/bin/env python3
"""
_tools/launch_dual_wf.py
=============================================================================
Dual-Process Master Launcher for Walk-Forward Training.

Runs 2 independent OS subprocesses:
  - Process 0 on GPU 0 (cuda:0): Odd windows (15, 17, 19, ...)
  - Process 1 on GPU 1 (cuda:1): Even windows (16, 18, 20, ...)
Complete OS-level process isolation guarantees 0% chance of OpenMP segfaults!
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "models" / "prl" / "variant_a_order1"
LOGS_DIR = BASE_DIR / "scratch" / "wf_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Dual-Process Walk-Forward Master Launcher")
    parser.add_argument("--start-win", type=int, default=0, help="Starting window index (default: 0)")
    parser.add_argument("--end-win", type=int, default=68, help="Ending window index (default: 68)")
    parser.add_argument("--force", action="store_true", help="Force re-training completed windows")
    args = parser.parse_args()

    from _tools.walkforward_trainer import HORIZONS
    completed = set()
    if not args.force:
        # Окно считается полностью готовым только при наличии всех 12 горизонтов (1s..1800s):
        for w in range(args.start_win, args.end_win + 1):
            w_dir = OUTPUT_DIR / f"win_{w:02d}"
            mf = w_dir / f"meta_win_{w:02d}.json"
            if mf.exists() and all((w_dir / f"xgb_h{H}s.ubj").exists() for H in HORIZONS):
                try:
                    with open(mf) as f:
                        if json.load(f).get("completed", False):
                            completed.add(w)
                except Exception:
                    pass

    remaining = [w for w in range(args.start_win, args.end_win + 1) if w not in completed]
    print("=" * 70)
    print("  DUAL-PROCESS WALK-FORWARD MASTER LAUNCHER (XGBoost GPU + PyTorch NNs)")
    print(f"  Range: [{args.start_win}..{args.end_win}] | Force: {args.force}")
    print(f"  Total remaining windows: {len(remaining)}")
    print(f"  Already completed: {len(completed)}")
    print("=" * 70, flush=True)

    if not remaining:
        print("  🎉 All windows are already completed!")
        return

    # Split into GPU 0 and GPU 1
    # Odd indices to GPU 0, Even indices to GPU 1
    wins_gpu0 = [w for w in remaining if w % 2 != 0]
    wins_gpu1 = [w for w in remaining if w % 2 == 0]

    print(f"  GPU 0 ({len(wins_gpu0)} windows): {wins_gpu0[:8]}...")
    print(f"  GPU 1 ({len(wins_gpu1)} windows): {wins_gpu1[:8]}...", flush=True)

    py_bin = sys.executable
    trainer_script = str(BASE_DIR / "_tools" / "walkforward_trainer.py")

    log_gpu0 = open(LOGS_DIR / "gpu0_worker.log", "w")
    log_gpu1 = open(LOGS_DIR / "gpu1_worker.log", "w")

    env0 = os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = "0"
    env1 = os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = "1"

    p0, p1 = None, None
    if wins_gpu0:
        cmd0 = [py_bin, trainer_script, "--windows", ",".join(map(str, wins_gpu0)), "--device", "cuda:0"]
        if args.force:
            cmd0.append("--force")
        print("\n  🚀 Starting Process 0 on GPU 0 (CUDA_VISIBLE_DEVICES=0)...")
        p0 = subprocess.Popen(cmd0, stdout=log_gpu0, stderr=subprocess.STDOUT, env=env0)
        print(f"  Process 0 PID: {p0.pid} (Logs: scratch/wf_logs/gpu0_worker.log)")

    if wins_gpu1:
        cmd1 = [py_bin, trainer_script, "--windows", ",".join(map(str, wins_gpu1)), "--device", "cuda:0"]
        if args.force:
            cmd1.append("--force")
        print("  🚀 Starting Process 1 on GPU 1 (CUDA_VISIBLE_DEVICES=1)...\n")
        p1 = subprocess.Popen(cmd1, stdout=log_gpu1, stderr=subprocess.STDOUT, env=env1)
        print(f"  Process 1 PID: {p1.pid} (Logs: scratch/wf_logs/gpu1_worker.log)")

    print("  Workers are running with complete OS-level process isolation.\n", flush=True)

    # Monitor loop
    try:
        while True:
            r0 = p0.poll() if p0 is not None else 0
            r1 = p1.poll() if p1 is not None else 0
            if r0 is not None and r1 is not None:
                print(f"\n  ✅ All workers finished! (P0 rc={r0}, P1 rc={r1})")
                break
            time.sleep(10)
    finally:
        log_gpu0.close()
        log_gpu1.close()


if __name__ == "__main__":
    main()
