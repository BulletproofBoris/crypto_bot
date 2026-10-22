#!/usr/bin/env python3
"""
Экспортер данных микроструктуры PRL для обучения нейросетей в Google Colab на GPU.
Выделяет строго идентичные временные интервалы, что и у 1-й модели (LightGBM):
- Train win_00: 2026-10-15 23:01:18 -> 2026-10-18 09:21:18 (210,000 строк)
- OOS Test win_00: 2026-10-18 09:27:30 -> 2026-10-18 15:13:48 (20,754 строки)
Формирует ультра-компактный архив (.npz ~27 МБ) для моментальной загрузки в Colab.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def export_prl_colab_dataset(upload_gdrive=False):
    print("=" * 80)
    print("📦 ЭКСПОРТ ДАТАСЕТА PRL ДЛЯ ОБУЧЕНИЯ В GOOGLE COLAB (NVIDIA T4)")
    print("=" * 80)

    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    feats_json = BASE_DIR / "data" / "features" / "selected_features_prl_h8s.json"
    h12_manifest = BASE_DIR / "models" / "prl" / "h12s_k3.5" / "manifest.json"

    if not labeled_path.exists() or not feats_json.exists():
        print("❌ Ошибка: Не найдены исходные файлы датасета или признаков!")
        return

    with open(feats_json) as f:
        feature_cols = json.load(f)["feature_order"]

    meta_cols = [
        'timestamp', 'sample_split', 'is_data_gap', 'ST_mid_price', 'ST_trade_vol', 'BO_sec_since_tick',
        'target_class_7_h5_k3', 'target_dt_ticks_h5_k3',
        'target_class_7_h12_k3.5', 'target_dt_ticks_h12_k3.5',
        'target_class_7_h25_k3', 'target_dt_ticks_h25_k3',
        'target_class_7_h60_k2.5', 'target_dt_ticks_h60_k2.5'
    ]

    print(f"📥 Чтение Parquet ({len(feature_cols)} признаков + мета-колонки)...")
    schema_cols = set(pq.read_schema(labeled_path).names)
    load_cols = list(dict.fromkeys([c for c in feature_cols + meta_cols if c in schema_cols]))
    tbl = pq.read_table(labeled_path, columns=load_cols)
    df = tbl.to_pandas()
    print(f"✅ Загружено {len(df):,} строк.")

    # Очистка зависших осцилляторов BigONE (BO_sec_since_tick > 60s)
    X_all = df[feature_cols].values.astype(np.float32).copy()
    if 'BO_sec_since_tick' in df.columns:
        bo_stale = (df['BO_sec_since_tick'].fillna(0).values > 60.0)
        if np.any(bo_stale):
            zero_prefixes = ('LL_', 'BO_ofi_', 'BO_cvd_', 'BO_imb_', 'BO_obi_', 'BO_micro_', 'BO_ret_', 'BO_vol_intensity_')
            zero_cols = [i_c for i_c, fn in enumerate(feature_cols) if fn.startswith(zero_prefixes)]
            if zero_cols:
                X_all[np.ix_(bo_stale, zero_cols)] = 0.0

    # Временные границы всех асинхронных окон Walk-Forward (H=5: 12 окон, H=12: 5 окон, H=25: 2 окна)
    from _tools.train_rolling_swarm import find_clean_segments, generate_rolling_windows_reverse
    segs = find_clean_segments(df)
    wins_h5, _, _, _, _ = generate_rolling_windows_reverse(segs, H=5, K=3.0)
    wins_h12, _, _, _, _ = generate_rolling_windows_reverse(segs, H=12, K=3.5)
    wins_h25, _, _, _, _ = generate_rolling_windows_reverse(segs, H=25, K=3.0)
    wins_h60, _, _, _, _ = generate_rolling_windows_reverse(segs, H=60, K=2.5)
    w0 = wins_h12[0]
    tr_idx = w0['tr_idx']
    te_idx = w0['te_idx']

    print(f"⏱️ Сетка асинхронных окон: H=5 ({len(wins_h5)} окон по 75k), H=12 ({len(wins_h12)} окон по 210k), H=25 ({len(wins_h25)} окон по 375k), H=60 ({len(wins_h60)} окон по 750k)")
    print(f"⏱️ Train win_00 (H=12) TS: {df['timestamp'].iloc[tr_idx[0]]} ➔ {df['timestamp'].iloc[tr_idx[-1]]} ({len(tr_idx):,} строк)")
    print(f"⏱️ Test  win_00 (OOS)  TS: {df['timestamp'].iloc[te_idx[0]]} ➔ {df['timestamp'].iloc[te_idx[-1]]} ({len(te_idx):,} строк)")

    windows_spec = {}
    for h_key, w_list in [("h5", wins_h5), ("h12", wins_h12), ("h25", wins_h25), ("h60", wins_h60)]:
        windows_spec[f"{h_key}_n_windows"] = np.int32(len(w_list))
        for w_i, w_obj in enumerate(w_list):
            windows_spec[f"{h_key}_tr_idx_{w_i:02d}"] = w_obj['tr_idx'].astype(np.int32)
            windows_spec[f"{h_key}_te_idx_{w_i:02d}"] = w_obj['te_idx'].astype(np.int32)

    out_file = BASE_DIR / "data" / "prl_neural_colab_data.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Упаковка в сжатый NPZ: {out_file.relative_to(BASE_DIR)}...")
    np.savez_compressed(
        out_file,
        X_all=X_all.astype(np.float32),
        ts_all=df['timestamp'].astype(str).values,
        y_all_h5=(df['target_class_7_h5_k3'].fillna(0).values + 3).astype(np.int8),
        y_all_h12=(df['target_class_7_h12_k3.5'].fillna(0).values + 3).astype(np.int8),
        y_all_h25=(df['target_class_7_h25_k3'].fillna(0).values + 3).astype(np.int8),
        y_all_h60=(df['target_class_7_h60_k2.5'].fillna(0).values + 3).astype(np.int8) if 'target_class_7_h60_k2.5' in df.columns else np.zeros(len(df), dtype=np.int8),
        dt_all_h5=df['target_dt_ticks_h5_k3'].fillna(0).values.astype(np.float16),
        dt_all_h12=df['target_dt_ticks_h12_k3.5'].fillna(0).values.astype(np.float16),
        dt_all_h25=df['target_dt_ticks_h25_k3'].fillna(0).values.astype(np.float16),
        dt_all_h60=df['target_dt_ticks_h60_k2.5'].fillna(0).values.astype(np.float16) if 'target_dt_ticks_h60_k2.5' in df.columns else np.zeros(len(df), dtype=np.float16),
        features=np.array(feature_cols),
        n_windows=np.int32(len(wins_h12)),
        **windows_spec
    )

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"🎉 Экспорт завершен! Размер сжатого файла: {size_mb:.2f} МБ")

    if upload_gdrive:
        token_path = BASE_DIR / "_tools" / "token.json"
        if token_path.exists():
            print("\n☁️ Загрузка на Google Drive...")
            try:
                from _tools.upload_to_gdrive import get_drive_service, upload_file, FOLDER_ID
                svc = get_drive_service(str(token_path))
                fid = upload_file(svc, str(out_file), FOLDER_ID)
                print(f"✅ Файл на Google Drive готов: ID {fid}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки на Google Drive: {e}")
        else:
            print("ℹ️ Google Drive токен не найден, файл сохранен локально.")

if __name__ == '__main__':
    upload = "--upload" in sys.argv
    export_prl_colab_dataset(upload_gdrive=upload)

