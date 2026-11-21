#!/usr/bin/env python3
"""
_tools/walkforward_trainer.py
=============================================================================
Process-Isolated Walk-Forward Library Trainer for Order-1 Models (Variant A).

Key Properties:
  - Process Isolation: Designed to run as independent OS processes per GPU,
    completely eliminating OpenMP C++ thread contention / segfaults.
  - VRAM Efficiency: BATCH_SIZE = 4096, 20 epochs with early validation.
  - Checkpoint Safety: Atomic file replacement for meta_win_XX.json and
    walkforward_summary.json.
  - Gap Masking: ST_sec_since_tick > 60 | BO_sec_since_tick > 60.
=============================================================================
"""

import os
os.environ["CUPY_CACHE_DIR"] = str((os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch", ".cupy_cache"))))
import sys
import json
import time
import argparse
import warnings
import traceback
import gc
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from order1_research_lab import (
    NUM_CLASSES, compute_evaluation_metrics
)
from data_labeler import compute_single_horizon_labels

# ─── Configuration ───
HORIZONS = [1, 2, 5, 10, 20, 40, 80, 160]
K_STAR = {1: 3.0, 2: 3.0, 5: 3.0, 10: 3.0, 20: 3.0, 40: 5.0, 80: 2.5, 160: 2.0}
TICK_SIZE = 0.0001
VENUE = "ST"

WIN_TRAIN = 172800  # 48 hours
WIN_STEP  = 43200   # 12 hours
EPOCHS_NN = 20
BATCH_SIZE = 4096

PARQUET_PATH = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
OUTPUT_DIR   = BASE_DIR / "models" / "prl" / "variant_a_order1"
FEATS_PATH   = BASE_DIR / "data" / "features" / "selected_features.json"


def get_seq_len(H: int) -> int:
    return min(max(1, int(round(H * K_STAR[H]))), MAX_SEQ_LEN)


def get_window_bounds(total_rows: int, win_idx: int) -> tuple[int, int, int, int]:
    train_start = win_idx * WIN_STEP
    train_end = train_start + WIN_TRAIN
    val_start = train_end
    val_end = min(total_rows, val_start + WIN_STEP)
    if val_end <= val_start:
        val_end = total_rows
        val_start = total_rows - WIN_STEP
        train_end = val_start
        train_start = train_end - WIN_TRAIN
    return train_start, train_end, val_start, val_end


def load_window_data(pf: pq.ParquetFile, num_cols: list[str],
                     train_start: int, train_end: int,
                     val_start: int, val_end: int):
    label_cols = [
        'ST_mid_price', 'ST_spread', 'ST_bid_1_price', 'ST_ask_1_price',
        'ST_bid_1_qty', 'ST_ask_1_qty', 'ST_bid_2_qty', 'ST_ask_2_qty',
        'ST_bids_vol_top5', 'ST_asks_vol_top5', 'ST_trade_vol',
        'ST_sec_since_tick', 'BO_sec_since_tick', 'arb_spread_zscore_20s'
    ]
    all_schema_names = pf.schema.names
    actual_num_cols = [c for c in num_cols if c in all_schema_names]
    actual_lbl_cols = [c for c in label_cols if c in all_schema_names]
    all_req_cols = list(set(actual_num_cols + actual_lbl_cols))

    rg_meta = [pf.metadata.row_group(i) for i in range(pf.metadata.num_row_groups)]
    rg_offsets = [0]
    for m in rg_meta[:-1]:
        rg_offsets.append(rg_offsets[-1] + m.num_rows)

    needed_rgs = []
    for i, (offset, m) in enumerate(zip(rg_offsets, rg_meta)):
        rg_end = offset + m.num_rows
        if not (val_end <= offset or train_start >= rg_end):
            needed_rgs.append(i)

    tables = [pf.read_row_group(rg_i, columns=all_req_cols) for rg_i in needed_rgs]
    combined = pa.concat_tables(tables)

    first_rg_offset = rg_offsets[needed_rgs[0]]
    slice_start = train_start - first_rg_offset
    total_len = val_end - train_start
    window_table = combined.slice(slice_start, total_len)

    # 1. Fast label extraction (only 11 columns to pandas, ~0.05s)
    df_lbl = window_table.select(actual_lbl_cols).to_pandas()
    n_tr = train_end - train_start
    n_val = val_end - val_start

    gap = (df_lbl['ST_sec_since_tick'] > 60) | (df_lbl['BO_sec_since_tick'] > 60)
    tr_mask = ~gap.iloc[:n_tr].values
    val_mask = ~gap.iloc[n_tr:].values

    df_tr_lbl = df_lbl.iloc[:n_tr].reset_index(drop=True)
    df_val_lbl = df_lbl.iloc[n_tr:].reset_index(drop=True)
    del df_lbl

    # 2. Fast numeric feature extraction directly from Arrow chunks (no 956-col DataFrame)
    X_full = np.column_stack([window_table[col].to_numpy(zero_copy_only=False) for col in actual_num_cols]).astype(np.float32)
    del tables, combined, window_table
    gc.collect()

    X_tr = X_full[:n_tr][tr_mask]
    X_val = X_full[n_tr:][val_mask]
    del X_full

    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    gc.collect()
    return X_tr, X_val, df_tr_lbl, df_val_lbl, tr_mask, val_mask


def train_xgboost_gpu_model(X_tr: np.ndarray, y_tr: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            save_path: Path, device: torch.device) -> tuple[np.ndarray, dict]:
    dev_str = str(device)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'objective': 'multi:softprob',
        'num_class': NUM_CLASSES,
        'tree_method': 'hist',
        'device': dev_str,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0,
        'seed': 42
    }
    model = xgb.train(
        params, dtrain, num_boost_round=70,
        evals=[(dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    model.save_model(str(save_path))
    probs = model.predict(dval)
    metrics = compute_evaluation_metrics(y_val, probs)
    del model, dtrain, dval
    gc.collect()
    return probs, metrics


def train_single_window(win_idx: int, num_cols: list[str],
                        total_rows: int, assigned_device: torch.device,
                        force: bool = False):
    win_dir = OUTPUT_DIR / f"win_{win_idx:02d}"
    win_dir.mkdir(parents=True, exist_ok=True)
    meta_path = win_dir / f"meta_win_{win_idx:02d}.json"

    if meta_path.exists() and not force:
        try:
            with open(meta_path) as f:
                existing_meta = json.load(f)
            if existing_meta.get("completed", False):
                print(f"  ⏭ Window {win_idx:02d} already completed. Skipping.", flush=True)
                return existing_meta
        except Exception:
            pass

    t_start = time.time()
    t_s, t_e, v_s, v_e = get_window_bounds(total_rows, win_idx)
    dev_str = str(assigned_device)
    print(f"\n{'='*70}\n  🚀 [{dev_str}] Processing Window {win_idx:02d}/68: Train [{t_s:,}..{t_e:,}], Val [{v_s:,}..{v_e:,}]\n{'='*70}", flush=True)

    pf = pq.ParquetFile(str(PARQUET_PATH))
    X_tr, X_val, df_tr_lbl, df_val_lbl, tr_mask, val_mask = load_window_data(
        pf, num_cols, t_s, t_e, v_s, v_e
    )
    print(f"  [{dev_str} | Win {win_idx:02d}] Valid Rows: Train {len(X_tr):,} | Val {len(X_val):,}", flush=True)

    labels_tr, labels_val = {}, {}
    for H in HORIZONS:
        K = K_STAR[H]
        l_tr = compute_single_horizon_labels(df_tr_lbl, "prl", H, K, min_vol=0, tick_size=TICK_SIZE, venue=VENUE)
        l_val = compute_single_horizon_labels(df_val_lbl, "prl", H, K, min_vol=0, tick_size=TICK_SIZE, venue=VENUE)
        labels_tr[H] = (l_tr['class_7'] + 3).clip(0, 6).astype(np.int64)[tr_mask]
        labels_val[H] = (l_val['class_7'] + 3).clip(0, 6).astype(np.int64)[val_mask]

    del df_tr_lbl, df_val_lbl
    gc.collect()

    horizon_results = {}

    # Concurrent XGBoost GPU for all 8 horizons (2 parallel workers on assigned_device)
    from concurrent.futures import ThreadPoolExecutor

    def _train_single_xgb_h(H):
        t0 = time.time()
        y_tr, y_val = labels_tr[H], labels_val[H]
        xgb_file = win_dir / f"xgb_h{H}s.ubj"
        probs, metrics = train_xgboost_gpu_model(X_tr, y_tr, X_val, y_val, xgb_file, assigned_device)
        return H, probs, metrics, time.time() - t0

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_train_single_xgb_h, H) for H in HORIZONS]
        for f in futures:
            H, probs, metrics, elapsed = f.result()
            horizon_results[f"H{H}s"] = {
                "model": "XGBoost",
                "auc_macro": round(float(metrics["auc_macro"]), 5),
                "conf_dir_acc": round(float(metrics["conf_dir_acc"]), 2),
                "ic": round(float(metrics["ic"]), 5),
                "time_sec": round(elapsed, 1)
            }
            print(f"  [{dev_str} | Win {win_idx:02d}] H={H:3d}s XGBoost | AUC={metrics['auc_macro']:.4f} | Dir={metrics['conf_dir_acc']:.1f}% | IC={metrics['ic']:+.4f} | {elapsed:.1f}s", flush=True)

    del X_tr, X_val, labels_tr, labels_val
    gc.collect()

    elapsed_total = round(time.time() - t_start, 1)
    meta = {
        "window_id": f"win_{win_idx:02d}",
        "win_idx": win_idx,
        "completed": True,
        "train_range": [int(t_s), int(t_e)],
        "val_range": [int(v_s), int(v_e)],
        "valid_train_samples": int(tr_mask.sum()),
        "valid_val_samples": int(val_mask.sum()),
        "elapsed_sec": elapsed_total,
        "horizons": horizon_results
    }
    # Atomic write of meta
    tmp_meta = meta_path.with_name(f"{meta_path.name}.tmp.{os.getpid()}")
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, meta_path)

    # Safe multi-process summary update
    summary_path = OUTPUT_DIR / "walkforward_summary.json"
    try:
        cur_summary = {}
        if summary_path.exists():
            try:
                with open(summary_path) as sf:
                    cur_summary = json.load(sf)
            except Exception:
                cur_summary = {}
        cur_summary[f"win_{win_idx:02d}"] = meta
        tmp_sum = summary_path.with_name(f"{summary_path.name}.tmp.{os.getpid()}")
        with open(tmp_sum, "w") as sf:
            json.dump(cur_summary, sf, indent=2)
        os.replace(tmp_sum, summary_path)
    except Exception as e_sum:
        print(f"  ⚠️ Warning updating summary: {e_sum}", flush=True)

    print(f"  ✅ [{dev_str}] Window {win_idx:02d} Completed in {elapsed_total}s ({elapsed_total/60:.1f}m)", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Process-Isolated Walk-Forward Order-1 Model Trainer")
    parser.add_argument("--windows", type=str, required=True, help="Comma-separated list of window indices (e.g. 15,17,19)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (e.g. cuda:0 or cuda:1)")
    parser.add_argument("--force", action="store_true", help="Force re-training even if window already exists")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assigned_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    win_list = [int(w.strip()) for w in args.windows.split(",") if w.strip().isdigit()]

    print("=" * 70)
    print(f"  PROCESS WORKER [{assigned_device}] — PID: {os.getpid()}")
    print(f"  Assigned Windows ({len(win_list)}): {win_list[:10]}...")
    print(f"  Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS_NN} | Device: {assigned_device} | Force: {args.force}")
    print("=" * 70, flush=True)

    with open(FEATS_PATH) as f:
        num_cols = json.load(f)["feature_order"]

    pf = pq.ParquetFile(str(PARQUET_PATH))
    total_rows = pf.metadata.num_rows

    t_all = time.time()
    for w in win_list:
        train_single_window(w, num_cols, total_rows, assigned_device, force=args.force)

    total_time = round(time.time() - t_all, 1)
    print(f"\n🎉 [{assigned_device}] All assigned windows completed in {total_time}s ({total_time/60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
