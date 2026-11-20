#!/usr/bin/env python3
"""
_tools/launch_dual_order3_island_trainer.py
=============================================================================
DUAL-GPU ISLAND PBT MASTER LAUNCHER (2x NVIDIA GeForce RTX 5060 Ti)
=============================================================================
- GPU 0 (cuda:0): Island #0 [EXPLOITATION / KELLY ZOOM]
- GPU 1 (cuda:1): Island #1 [EXPLORATION / TAIL ARMOR]
- Непрерывная генетическая миграция через shared global_best.pt каждые 10 шагов.
- 100% насыщение обеих видеокарт (80%+ SM утилизация).
=============================================================================
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "scratch" / "order3_dual_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Dual-GPU Order-3 Walk-Forward v2 Island Trainer")
    parser.add_argument("--groups", type=int, default=192, help="Groups per island (POPSIZE = groups * 6, default=192 -> 1152/island)")
    parser.add_argument("--iters", type=int, default=800, help="Iterations per island")
    parser.add_argument("--hours", type=float, default=2.5, help="Time limit in hours (default=2.5h)")
    parser.add_argument("--win-len", type=int, default=4801, help="Window length in ticks (4801 -> 4800 steps = 80m continuous horizon)")
    parser.add_argument("--val-windows", type=int, default=16, help="Number of windows in Train/Val/OOS baskets")
    parser.add_argument("--early-stop-min", type=float, default=40.0, help="Early stopping stagnation minutes")
    parser.add_argument("--islands", type=int, default=2, choices=[2, 4], help="Total islands (2 = 1 per GPU dedicated, 4 = 2 per GPU interleaved)")
    args = parser.parse_args()

    # Очистка директории островной миграции для чистого старта
    islands_dir = BASE_DIR / "models" / "prl" / "order3_champion" / "islands"
    islands_dir.mkdir(parents=True, exist_ok=True)
    for old_isl in islands_dir.glob("island_*_best.pt*"):
        try:
            old_isl.unlink()
        except OSError:
            pass

    print("=" * 80)
    print(f"  🚀 DUAL-GPU ORDER-3 SWARM MASTER LAUNCHER (2x RTX 5060 Ti)")
    print(f"  Configuration: {args.islands} Islands | {args.groups} groups/island (Pop={args.groups*6}) | Total Pop={args.groups*6*args.islands} | {args.val_windows} Val Windows | Limit: {args.hours}h")
    if args.islands == 2:
        print("  GPU 0: Island #0 (Exploit / Kelly Zoom)")
        print("  GPU 1: Island #1 (Explore / Tail Armor)")
    else:
        print("  GPU 0: Island #0 (Exploit) + Island #1 (Balanced) [Interleaved]")
        print("  GPU 1: Island #2 (Explore) + Island #3 (Wide)    [Interleaved]")
    print("=" * 80, flush=True)

    def make_cmd(island_id: int, mode: str, log_path: Path):
        return [
            sys.executable,
            str(BASE_DIR / "_tools" / "train_order3_champion.py"),
            "--device", "cuda:0",
            "--groups", str(args.groups),
            "--iters", str(args.iters),
            "--hours", str(args.hours),
            "--win-len", str(args.win_len),
            "--val-windows", str(args.val_windows),
            "--island-id", str(island_id),
            "--num-islands", str(args.islands),
            "--early-stop-min", str(args.early_stop_min),
            "--island-mode", mode,
            "--log-file", str(log_path),
        ]

    env0 = os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = "0"
    env1 = os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = "1"

    workers = []
    files = []
    log_paths = []

    if args.islands == 2:
        log0 = LOG_DIR / "island0_gpu0_exploit.log"
        log1 = LOG_DIR / "island1_gpu1_explore.log"
        
        print("  🚀 Launching Island #0 on GPU 0 (Exploit / Momentum Zoom)...")
        f0 = open(log0, "w", encoding="utf-8")
        p0 = subprocess.Popen(make_cmd(0, "exploit", log0), stdout=f0, stderr=subprocess.STDOUT, env=env0)
        workers.append(p0); files.append(f0); log_paths.append(log0)
        time.sleep(4)

        print("  🚀 Launching Island #1 on GPU 1 (Explore / Tail Armor)...")
        f1 = open(log1, "w", encoding="utf-8")
        p1 = subprocess.Popen(make_cmd(1, "explore", log1), stdout=f1, stderr=subprocess.STDOUT, env=env1)
        workers.append(p1); files.append(f1); log_paths.append(log1)
    else:
        log0 = LOG_DIR / "island0_gpu0_exploit.log"
        log1 = LOG_DIR / "island1_gpu0_balanced.log"
        log2 = LOG_DIR / "island2_gpu1_explore.log"
        log3 = LOG_DIR / "island3_gpu1_wide.log"

        print("  🚀 Launching Island #0 on GPU 0 (Exploit)...")
        f0 = open(log0, "w", encoding="utf-8")
        workers.append(subprocess.Popen(make_cmd(0, "exploit", log0), stdout=f0, stderr=subprocess.STDOUT, env=env0))
        files.append(f0); log_paths.append(log0)
        time.sleep(4)

        print("  🚀 Launching Island #1 on GPU 0 (Balanced)...")
        f1 = open(log1, "w", encoding="utf-8")
        workers.append(subprocess.Popen(make_cmd(1, "balanced", log1), stdout=f1, stderr=subprocess.STDOUT, env=env0))
        files.append(f1); log_paths.append(log1)
        time.sleep(4)

        print("  🚀 Launching Island #2 on GPU 1 (Explore)...")
        f2 = open(log2, "w", encoding="utf-8")
        workers.append(subprocess.Popen(make_cmd(2, "explore", log2), stdout=f2, stderr=subprocess.STDOUT, env=env1))
        files.append(f2); log_paths.append(log2)
        time.sleep(4)

        print("  🚀 Launching Island #3 on GPU 1 (Wide)...")
        f3 = open(log3, "w", encoding="utf-8")
        workers.append(subprocess.Popen(make_cmd(3, "wide", log3), stdout=f3, stderr=subprocess.STDOUT, env=env1))
        files.append(f3); log_paths.append(log3)

    print(f"  ✅ All {args.islands} islands running with dedicated GPU saturation and shared migration!")
    print("  Monitoring active workers...", flush=True)

    try:
        last_print = 0
        while True:
            alive = [p.poll() is None for p in workers]
            if not any(alive):
                break
            time.sleep(5)
            now = time.time()
            if now - last_print >= 25:
                last_print = now
                for i, (f_w, lp) in enumerate(zip(files, log_paths)):
                    try:
                        f_w.flush()
                        lines = lp.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                        last_line = lines[-1] if lines else "waiting..."
                        print(f"  [Island #{i}]: {last_line[:95]}", flush=True)
                    except Exception:
                        pass
    finally:
        for f in files:
            f.close()

    rcs = [p.poll() for p in workers]
    print("\n" + "=" * 80)
    print(f"  🏁 All islands finished! Exit codes: {rcs}")
    print("=" * 80, flush=True)

    if any(rc != 0 for rc in rcs):
        print("⚠️ Warning: One of the islands exited with non-zero code!")
        sys.exit(1)


if __name__ == "__main__":
    main()
