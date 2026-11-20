#!/usr/bin/env python3
"""
_tools/convert_models_to_binary.py
=============================================================================
Converts all XGBoost models in models/prl/variant_a_order1/ from verbose JSON
to high-speed compact binary format (UBJSON / .ubj).
Verifies binary integrity before removing JSON.
=============================================================================
"""

import os
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "prl" / "variant_a_order1"


def convert_single_file(json_path_str: str) -> tuple[bool, int, int]:
    json_path = Path(json_path_str)
    ubj_path = json_path.with_suffix(".ubj")
    try:
        sz_before = json_path.stat().st_size
        bst = xgb.Booster()
        bst.load_model(str(json_path))
        bst.save_model(str(ubj_path))
        sz_after = ubj_path.stat().st_size

        # Quick validation
        bst_check = xgb.Booster()
        bst_check.load_model(str(ubj_path))
        assert sz_after > 0

        # Remove original JSON file
        json_path.unlink()
        return True, sz_before, sz_after
    except Exception as e:
        print(f"Error converting {json_path}: {e}")
        if ubj_path.exists():
            ubj_path.unlink()
        return False, 0, 0


def main():
    json_files = sorted(glob.glob(str(MODELS_DIR / "win_*" / "*.json")))
    json_models = [f for f in json_files if not os.path.basename(f).startswith("meta_")]

    total_models = len(json_models)
    print("=" * 70)
    print(f"  CONVERTING {total_models} XGBOOST MODELS TO NATIVE BINARY (.UBJ)")
    print(f"  Target directory: {MODELS_DIR}")
    print("=" * 70, flush=True)

    if not json_models:
        print("  No JSON models found to convert!")
        return

    t0 = time.time()
    total_before = 0
    total_after = 0
    successful = 0

    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        results = list(executor.map(convert_single_file, json_models))

    for ok, b4, aft in results:
        if ok:
            successful += 1
            total_before += b4
            total_after += aft

    elapsed = time.time() - t0
    mb_before = total_before / (1024 * 1024)
    mb_after = total_after / (1024 * 1024)
    savings = (1.0 - (total_after / total_before)) * 100 if total_before else 0

    print("=" * 70)
    print(f"  ✅ Successfully converted: {successful}/{total_models} models in {elapsed:.1f}s")
    print(f"  Original JSON size: {mb_before:.1f} MB")
    print(f"  Binary UBJ size:    {mb_after:.1f} MB")
    print(f"  Space saved:        {mb_before - mb_after:.1f} MB (-{savings:.1f}%)")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
