#!/usr/bin/env python3
"""
_tools/launch_dual_order2_wf.py
=============================================================================
Dual-GPU Master Launcher for Order-2 Walk-Forward Risk Library:
  - GPU 0 (cuda:0): Odd windows
  - GPU 1 (cuda:1): Even windows
Complete OS-level process isolation (CUDA_VISIBLE_DEVICES).

After all windows finish, compiles the continuous 11D OOS Risk Matrix:
  data/cache/order2_risk_matrix_11d.npz
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
OUTPUT_DIR = BASE_DIR / "models" / "prl" / "order2_wf"
LOGS_DIR = BASE_DIR / "scratch" / "wf_order2_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def compile_continuous_oos_matrix():
    print("\n" + "=" * 70)
    print("  COMPILING 100% CAUSAL CONTINUOUS 11D OOS RISK MATRIX...")
    print("=" * 70, flush=True)

    import numpy as np
    import xgboost as xgb
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from _tools.walkforward_order2_trainer import (
        CORE_TARGET_NAMES,
        TARGET_NAMES,
        build_66d_features,
        GUARD_LOB_PATH,
        ALPHA_PATH,
        WF_SUMMARY_PATH,
    )

    lob = np.load(GUARD_LOB_PATH, allow_pickle=True)
    alpha = np.load(ALPHA_PATH, allow_pickle=True)
    total_rows = len(lob["bid_p"])

    with open(WF_SUMMARY_PATH) as f:
        summary_order1 = json.load(f)

    X_all, feature_names = build_66d_features(lob, alpha)
    oos_matrix_16d = np.zeros((total_rows, len(TARGET_NAMES)), dtype=np.float32)

    # Compile OOS slices dynamically from all available Walk-Forward windows!
    win_keys = sorted([k for k in summary_order1.keys() if k.startswith("win_")], key=lambda x: int(x.split("_")[1]))
    w0_end = summary_order1[win_keys[0]]["val_range"][1]
    slices = [(int(win_keys[0].split("_")[1]), 0, w0_end)]

    for wk in win_keys[1:-1]:
        w_i = int(wk.split("_")[1])
        v_s, v_e = summary_order1[wk]["val_range"]
        slices.append((w_i, v_s, v_e))

    last_wk = win_keys[-1]
    last_wi = int(last_wk.split("_")[1])
    w_last_start = summary_order1[last_wk]["val_range"][0]
    slices.append((last_wi, w_last_start, total_rows))

    for win_idx, s_idx, e_idx in slices:
        win_dir = OUTPUT_DIR / f"win_{win_idx:02d}"
        X_slice = X_all[s_idx:e_idx]
        d_slice = xgb.DMatrix(X_slice, feature_names=feature_names)

        for col_idx, name in enumerate(TARGET_NAMES):
            model_file = win_dir / f"{name}.ubj"
            if model_file.exists():
                bst = xgb.Booster()
                bst.load_model(str(model_file))
                oos_matrix_16d[s_idx:e_idx, col_idx] = bst.predict(d_slice)

    out_npz = BASE_DIR / "data" / "cache" / "order2_risk_matrix_11d.npz"
    tmp_npz = out_npz.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_npz,
        risk_matrix_11d=oos_matrix_16d[:, :len(CORE_TARGET_NAMES)],
        target_names=np.array(CORE_TARGET_NAMES),
        total_rows=total_rows,
    )
    tmp_npz.replace(out_npz)

    out_ext_npz = BASE_DIR / "data" / "cache" / "order2_risk_matrix_16d.npz"
    tmp_ext_npz = out_ext_npz.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_ext_npz,
        risk_matrix_16d=oos_matrix_16d,
        target_names=np.array(TARGET_NAMES),
        total_rows=total_rows,
    )
    tmp_ext_npz.replace(out_ext_npz)
    sz_mb = out_npz.stat().st_size / (1024 * 1024)
    sz_ext = out_ext_npz.stat().st_size / (1024 * 1024)
    print(f"✅ Continuous 11D Core OOS Risk Matrix saved: {out_npz} ({sz_mb:.1f} MB)")
    print(f"✅ Continuous 16D Extended OOS Risk Matrix saved: {out_ext_npz} ({sz_ext:.1f} MB, {total_rows:,} rows)")


def main():
    parser = argparse.ArgumentParser(description="Dual-GPU Order-2 Walk-Forward Launcher")
    parser.add_argument("--start-win", type=int, default=0, help="Start window")
    parser.add_argument("--end-win", type=int, default=71, help="End window")
    parser.add_argument("--force", action="store_true", help="Force re-training")
    args = parser.parse_args()

    from _tools.walkforward_order2_trainer import TARGET_NAMES
    completed = set()
    if not args.force:
        for w in range(args.start_win, args.end_win + 1):
            w_dir = OUTPUT_DIR / f"win_{w:02d}"
            mf = w_dir / f"meta_order2_win_{w:02d}.json"
            if mf.exists() and all((w_dir / f"{nm}.ubj").exists() for nm in TARGET_NAMES):
                try:
                    with open(mf) as f:
                        if json.load(f).get("completed", False):
                            completed.add(w)
                except Exception:
                    pass

    remaining = [w for w in range(args.start_win, args.end_win + 1) if w not in completed]
    print("=" * 70)
    print("  DUAL-GPU ORDER-2 WALK-FORWARD MASTER LAUNCHER (XGBOOST GPU)")
    print(f"  Range: [{args.start_win}..{args.end_win}] | Remaining: {len(remaining)}/{args.end_win - args.start_win + 1}")
    print("=" * 70, flush=True)

    if not remaining:
        print("  🎉 All Order-2 windows are already completed!")
        compile_continuous_oos_matrix()
        return

    wins_gpu0 = [w for w in remaining if w % 2 != 0]
    wins_gpu1 = [w for w in remaining if w % 2 == 0]

    py_bin = sys.executable
    trainer_script = str(BASE_DIR / "_tools" / "walkforward_order2_trainer.py")

    log_gpu0 = open(LOGS_DIR / "gpu0_order2.log", "w")
    log_gpu1 = open(LOGS_DIR / "gpu1_order2.log", "w")

    env0 = os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = "0"
    env1 = os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = "1"

    p0, p1 = None, None
    if wins_gpu0:
        cmd0 = [py_bin, trainer_script, "--windows", ",".join(map(str, wins_gpu0)), "--device", "cuda:0"]
        if args.force:
            cmd0.append("--force")
        print(f"  🚀 Starting Process 0 on GPU 0 ({len(wins_gpu0)} windows)...")
        p0 = subprocess.Popen(cmd0, stdout=log_gpu0, stderr=subprocess.STDOUT, env=env0)

    if wins_gpu1:
        cmd1 = [py_bin, trainer_script, "--windows", ",".join(map(str, wins_gpu1)), "--device", "cuda:0"]
        if args.force:
            cmd1.append("--force")
        print(f"  🚀 Starting Process 1 on GPU 1 ({len(wins_gpu1)} windows)...")
        p1 = subprocess.Popen(cmd1, stdout=log_gpu1, stderr=subprocess.STDOUT, env=env1)

    print("  Workers are running with complete OS-level process isolation.\n", flush=True)

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

    compile_continuous_oos_matrix()


if __name__ == "__main__":
    main()
