#!/usr/bin/env python3
"""
_tools/build_order1_alpha_matrix.py
=============================================================================
Builds a continuous, 100% causal Walk-Forward Out-Of-Sample (OOS) Alpha Matrix
across all 3,306,887 rows (38 days) using the 72 Order-1 XGBoost windows (.ubj).

Outputs:
  - data/cache/order1_alpha_matrix_56d.npz:
      * probs_56d: float16 array [3306887, 56] (7 probabilities x 8 horizons)
      * expected_z_8h: float32 array [3306887, 8]
      * confidence_8h: float32 array [3306887, 8]
      * bias_8h: float32 array [3306887, 8]
=============================================================================
"""

import os
import sys
import json
import time
import gc
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
FEATS_PATH = BASE_DIR / "data" / "features" / "selected_features.json"
MODELS_DIR = BASE_DIR / "models" / "prl" / "variant_a_order1"
SUMMARY_PATH = MODELS_DIR / "walkforward_summary.json"
OUT_CACHE_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_56d.npz"
OUT_EXT_CACHE_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_84d.npz"

HORIZONS = [1, 2, 5, 10, 20, 40, 80, 160]
EXTENDED_HORIZONS = [320, 600, 900, 1800]
ALL_HORIZONS = HORIZONS + EXTENDED_HORIZONS  # 12 horizons up to 30m (1800s) statistical ceiling
CLASS_VALS = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)


def main():
    print("=" * 70)
    print("  BUILDING 100% CAUSAL WALK-FORWARD OOS ORDER-1 ALPHA MATRIX (8 HORIZONS)")
    print("=" * 70, flush=True)

    with open(FEATS_PATH) as f:
        num_cols = json.load(f)["feature_order"]

    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    pf = pq.ParquetFile(str(PARQUET_PATH))
    total_rows = pf.metadata.num_rows
    print(f"Total rows in dataset: {total_rows:,} (918.6 hours)")
    print(f"Total features per row: {len(num_cols)}")

    # Pre-allocate output arrays (56D backward-compatible + 84D extended 12-horizon pool)
    print("Pre-allocating Alpha memory arrays (56D core + 84D extended)...")
    probs_84d = np.zeros((total_rows, 7 * len(ALL_HORIZONS)), dtype=np.float16)
    expected_z_12h = np.zeros((total_rows, len(ALL_HORIZONS)), dtype=np.float32)
    confidence_12h = np.zeros((total_rows, len(ALL_HORIZONS)), dtype=np.float32)
    bias_12h = np.zeros((total_rows, len(ALL_HORIZONS)), dtype=np.float32)

    # Determine window prediction slices dynamically for any number of Walk-Forward windows!
    win_keys = sorted([k for k in summary.keys() if k.startswith("win_")], key=lambda x: int(x.split("_")[1]))
    slices = []
    w0_end = summary[win_keys[0]]["val_range"][1]
    slices.append((int(win_keys[0].split("_")[1]), 0, w0_end))

    for wk in win_keys[1:-1]:
        w_i = int(wk.split("_")[1])
        v_s, v_e = summary[wk]["val_range"]
        slices.append((w_i, v_s, v_e))

    last_wk = win_keys[-1]
    last_wi = int(last_wk.split("_")[1])
    w_last_start = summary[last_wk]["val_range"][0]
    slices.append((last_wi, w_last_start, total_rows))

    print(f"Total Walk-Forward slices: {len(slices)}")
    t0 = time.time()

    # Инкрементальное переиспользование неизменных прошлых окон из существующего кэша (экономия ~12 минут!)
    cached_valid_up_to = 0
    if OUT_EXT_CACHE_PATH.exists():
        try:
            old_c = np.load(OUT_EXT_CACHE_PATH, mmap_mode="r")
            old_len = len(old_c["expected_z_12h"])
            safe_cut = max(0, old_len - 90000)
            for _, s_i, e_i in slices:
                if e_i <= safe_cut:
                    cached_valid_up_to = e_i
            if cached_valid_up_to > 0:
                print(f"  ⚡ Переиспользуем готовый 84D каузальный префикс [0..{cached_valid_up_to:,}] из существующего кэша!")
                probs_84d[:cached_valid_up_to] = old_c["probs_84d"][:cached_valid_up_to]
                expected_z_12h[:cached_valid_up_to] = old_c["expected_z_12h"][:cached_valid_up_to]
                confidence_12h[:cached_valid_up_to] = old_c["confidence_12h"][:cached_valid_up_to]
                bias_12h[:cached_valid_up_to] = old_c["bias_12h"][:cached_valid_up_to]
        except Exception as e_inc:
            print(f"  ⚠️ Не удалось прочитать 84D кэш ({e_inc}), полный пересчет...")
            cached_valid_up_to = 0

    # Read row groups incrementally using pyarrow
    rg_meta = [pf.metadata.row_group(i) for i in range(pf.metadata.num_row_groups)]
    rg_offsets = [0]
    for m in rg_meta[:-1]:
        rg_offsets.append(rg_offsets[-1] + m.num_rows)

    for step_i, (win_idx, s_idx, e_idx) in enumerate(slices):
        if e_idx <= cached_valid_up_to:
            continue
        t_slice = time.time()
        win_dir = MODELS_DIR / f"win_{win_idx:02d}"
        n_rows = e_idx - s_idx

        # Find row groups overlapping [s_idx, e_idx]
        needed_rgs = []
        for i, (offset, m) in enumerate(zip(rg_offsets, rg_meta)):
            rg_end = offset + m.num_rows
            if not (e_idx <= offset or s_idx >= rg_end):
                needed_rgs.append(i)

        tables = [pf.read_row_group(rg_i, columns=num_cols) for rg_i in needed_rgs]
        combined = tables[0] if len(tables) == 1 else tables[0] if len(tables) == 1 else None
        if len(tables) == 1:
            combined = tables[0]
        else:
            import pyarrow as pa
            combined = pa.concat_tables(tables)

        first_rg_offset = rg_offsets[needed_rgs[0]]
        rel_start = s_idx - first_rg_offset
        window_table = combined.slice(rel_start, n_rows)
        del tables, combined

        X_slice = np.column_stack([window_table[col].to_numpy(zero_copy_only=False) for col in num_cols]).astype(np.float32)
        X_slice = np.nan_to_num(X_slice, nan=0.0, posinf=0.0, neginf=0.0)
        del window_table

        d_slice = xgb.DMatrix(X_slice)
        del X_slice

        # Run predict for all 12 horizons (gracefully zero-filling extended horizons if not yet trained for old windows)
        for h_i, H in enumerate(ALL_HORIZONS):
            model_file = win_dir / f"xgb_h{H}s.ubj"
            if not model_file.exists():
                model_file = win_dir / f"xgb_h{H}s.json"
            if not model_file.exists():
                continue
            bst = xgb.Booster()
            bst.load_model(str(model_file))
            p_h = bst.predict(d_slice)  # [n_rows, 7]

            c_start = h_i * 7
            probs_84d[s_idx:e_idx, c_start : c_start + 7] = p_h.astype(np.float16)
            expected_z_12h[s_idx:e_idx, h_i] = np.dot(p_h, CLASS_VALS)
            confidence_12h[s_idx:e_idx, h_i] = 1.0 - p_h[:, 3]
            bias_12h[s_idx:e_idx, h_i] = np.sum(p_h[:, 4:], axis=1) - np.sum(p_h[:, :3], axis=1)

        del d_slice
        gc.collect()

        elapsed = time.time() - t_slice
        if (step_i + 1) % 5 == 0 or step_i == len(slices) - 1:
            print(f"  [{step_i+1:2d}/{len(slices)}] Win {win_idx:02d} [{s_idx:,}..{e_idx:,}] ({n_rows:,} rows) processed in {elapsed:.1f}s | Total: {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 70)
    print("  SAVING COMPRESSED ORDER-1 ALPHA CACHES (56D Core + 84D Extended)...")
    OUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = OUT_CACHE_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_out,
        probs_56d=probs_84d[:, :56],
        expected_z_8h=expected_z_12h[:, :8],
        confidence_8h=confidence_12h[:, :8],
        bias_8h=bias_12h[:, :8],
        horizons=np.array(HORIZONS, dtype=np.int32),
    )
    tmp_out.replace(OUT_CACHE_PATH)

    tmp_ext = OUT_EXT_CACHE_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_ext,
        probs_84d=probs_84d,
        expected_z_12h=expected_z_12h,
        confidence_12h=confidence_12h,
        bias_12h=bias_12h,
        horizons=np.array(ALL_HORIZONS, dtype=np.int32),
    )
    tmp_ext.replace(OUT_EXT_CACHE_PATH)
    sz_mb = OUT_CACHE_PATH.stat().st_size / (1024 * 1024)
    sz_ext = OUT_EXT_CACHE_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Order-1 Core Cache saved to {OUT_CACHE_PATH} ({sz_mb:.1f} MB, 56D)")
    print(f"✅ Order-1 Extended Cache saved to {OUT_EXT_CACHE_PATH} ({sz_ext:.1f} MB, 84D / 12 horizons)")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
