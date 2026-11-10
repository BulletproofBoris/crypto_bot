#!/usr/bin/env python3
"""
_tools/run_local_rebuild_and_train_windows.py
=============================================================================
Локальный конвейер сборки data/_mmap_cache (с экономией RAM) и инкрементального
дообучения новых скользящих окон (LightGBM + 6 нейросетей) на свежих 2,615,713 тиках.
=============================================================================
"""

import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.v7_retrain_pipeline import HORIZON_PAIRS, run_cmd, step2_train_rolling_windows_6h
from _tools.train_rolling_swarm import find_clean_segments, generate_rolling_windows_reverse


def rebuild_mmap_cache_low_ram(coin: str = "prl"):
    t0 = time.time()
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    feats_json = BASE_DIR / "data" / "features" / f"selected_features_{coin}_h8s.json"
    with open(feats_json) as f:
        feature_cols = json.load(f)["feature_order"]

    pf = pq.ParquetFile(labeled_path)
    n_rows = pf.metadata.num_rows
    print("=" * 95, flush=True)
    print(f"⚙️ [1/2] НИЗКОУРОВНЕВАЯ СБОРКА data/_mmap_cache ИЗ {labeled_path.name} ({n_rows:,} СТРОК)", flush=True)
    print("=" * 95, flush=True)

    cache_dir = BASE_DIR / "data" / "_mmap_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old_f in cache_dir.glob("*_idx_*.npy"):
        old_f.unlink(missing_ok=True)

    # 1. Заполняем X_all (n_rows, 299) чанками по 50 колонок без двойного копирования в RAM
    n_feats = len(feature_cols)
    X_all = np.empty((n_rows, n_feats), dtype=np.float32)
    chunk_c = 50
    for c0 in range(0, n_feats, chunk_c):
        c1 = min(n_feats, c0 + chunk_c)
        cols_sub = feature_cols[c0:c1]
        tbl = pq.read_table(labeled_path, columns=cols_sub)
        for j, col_name in enumerate(cols_sub):
            X_all[:, c0 + j] = tbl.column(col_name).to_numpy(zero_copy_only=False).astype(np.float32)
        del tbl
        gc.collect()

    schema_cols = set(pf.schema.names)
    if "BO_sec_since_tick" in schema_cols:
        bo_sec = pq.read_table(labeled_path, columns=["BO_sec_since_tick"]).column(0).to_numpy(zero_copy_only=False)
        bo_stale = np.nan_to_num(bo_sec, nan=0.0) > 60.0
        if np.any(bo_stale):
            zero_prefixes = ("LL_", "BO_ofi_", "BO_cvd_", "BO_imb_", "BO_obi_", "BO_micro_", "BO_ret_", "BO_vol_intensity_")
            zero_cols = [i_c for i_c, fn in enumerate(feature_cols) if fn.startswith(zero_prefixes)]
            for zc in zero_cols:
                X_all[bo_stale, zc] = 0.0
        del bo_sec, bo_stale
        gc.collect()

    np.save(cache_dir / "X_all.npy", X_all)
    del X_all
    gc.collect()
    print("✅ Матрица X_all.npy сохранена на диск, память освобождена.", flush=True)

    # 2. Читаем только легкие мета-колонки и таргеты (~150 МБ вместо 4 ГБ)
    meta_cols = [
        "timestamp", "sample_split", "is_data_gap", "ST_mid_price", "ST_trade_vol",
        "target_class_7_h5_k3", "target_dt_ticks_h5_k3",
        "target_class_7_h12_k3.5", "target_dt_ticks_h12_k3.5",
        "target_class_7_h25_k3", "target_dt_ticks_h25_k3",
        "target_class_7_h60_k3", "target_dt_ticks_h60_k3",
        "target_class_7_h90_k3", "target_dt_ticks_h90_k3",
        "target_class_7_h120_k3", "target_dt_ticks_h120_k3",
    ]
    load_meta = [c for c in meta_cols if c in schema_cols]
    df = pq.read_table(labeled_path, columns=load_meta).to_pandas()
    if "timestamp" not in df.columns:
        df = df.reset_index()
        if "timestamp" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})

    segs = find_clean_segments(df)
    wins_map = {}
    for h, k in HORIZON_PAIRS:
        w_list, _, _, _, _ = generate_rolling_windows_reverse(segs, H=h, K=k)
        wins_map[f"h{h}"] = w_list
    print(
        "⏱️ Сетка скользящих окон: "
        + ", ".join(f"H={h}s ({len(wins_map[f'h{h}'])})" for h, _ in HORIZON_PAIRS),
        flush=True,
    )

    arrays = {
        "ts_all": df["timestamp"].astype(str).values,
        "y_all_h5": (df["target_class_7_h5_k3"].fillna(0).values + 3).astype(np.int8),
        "y_all_h12": (df["target_class_7_h12_k3.5"].fillna(0).values + 3).astype(np.int8),
        "y_all_h25": (df["target_class_7_h25_k3"].fillna(0).values + 3).astype(np.int8),
        "y_all_h60": (df["target_class_7_h60_k3"].fillna(0).values + 3).astype(np.int8),
        "y_all_h90": (df["target_class_7_h90_k3"].fillna(0).values + 3).astype(np.int8),
        "y_all_h120": (df["target_class_7_h120_k3"].fillna(0).values + 3).astype(np.int8),
        "dt_all_h5": df["target_dt_ticks_h5_k3"].fillna(0).values.astype(np.float16),
        "dt_all_h12": df["target_dt_ticks_h12_k3.5"].fillna(0).values.astype(np.float16),
        "dt_all_h25": df["target_dt_ticks_h25_k3"].fillna(0).values.astype(np.float16),
        "dt_all_h60": df["target_dt_ticks_h60_k3"].fillna(0).values.astype(np.float16),
        "dt_all_h90": df["target_dt_ticks_h90_k3"].fillna(0).values.astype(np.float16),
        "dt_all_h120": df["target_dt_ticks_h120_k3"].fillna(0).values.astype(np.float16),
        "features": np.array(feature_cols),
        "n_windows": np.int32(len(wins_map["h12"])),
    }
    for h_key, w_list in wins_map.items():
        arrays[f"{h_key}_n_windows"] = np.int32(len(w_list))
        for w_i, w_obj in enumerate(w_list):
            arrays[f"{h_key}_tr_idx_{w_i:02d}"] = w_obj["tr_idx"].astype(np.int32)
            arrays[f"{h_key}_te_idx_{w_i:02d}"] = w_obj["te_idx"].astype(np.int32)

    for k_name, v_arr in arrays.items():
        np.save(cache_dir / f"{k_name}.npy", v_arr)
    (cache_dir / "ready.txt").write_text("ok", encoding="utf-8")
    del df, arrays
    gc.collect()
    print(f"🎉 Бинарный кэш data/_mmap_cache обновлён за {time.time() - t0:.1f}с!", flush=True)


def main():
    rebuild_mmap_cache_low_ram(coin="prl")
    step2_train_rolling_windows_6h(coin="prl", epochs=10, runs=15)


if __name__ == "__main__":
    main()
