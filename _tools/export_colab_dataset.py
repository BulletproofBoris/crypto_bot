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
        'timestamp',
        'target_class_7_h5_k3', 'target_dt_ticks_h5_k3',
        'target_class_7_h12_k3.5', 'target_dt_ticks_h12_k3.5'
    ]

    print(f"📥 Чтение Parquet ({len(feature_cols)} признаков + мета-колонки)...")
    tbl = pq.read_table(labeled_path, columns=feature_cols + meta_cols)
    df = tbl.to_pandas()
    print(f"✅ Загружено {len(df):,} строк.")

    # Временные границы по референсной модели h12s_k3.5
    with open(h12_manifest) as f:
        ref_meta = json.load(f)[0] # win_00

    ts_series = pd.to_datetime(df['timestamp'])
    tr_s_ts = pd.to_datetime(ref_meta['train_ts_range'][0])
    tr_e_ts = pd.to_datetime(ref_meta['train_ts_range'][1])
    te_s_ts = pd.to_datetime(ref_meta['test_ts_range'][0])
    te_e_ts = pd.to_datetime(df['timestamp'].iloc[-1])

    if ts_series.dt.tz is not None:
        tr_s_ts = tr_s_ts.tz_localize(ts_series.dt.tz) if tr_s_ts.tzinfo is None else tr_s_ts.tz_convert(ts_series.dt.tz)
        tr_e_ts = tr_e_ts.tz_localize(ts_series.dt.tz) if tr_e_ts.tzinfo is None else tr_e_ts.tz_convert(ts_series.dt.tz)
        te_s_ts = te_s_ts.tz_localize(ts_series.dt.tz) if te_s_ts.tzinfo is None else te_s_ts.tz_convert(ts_series.dt.tz)
        te_e_ts = te_e_ts.tz_localize(ts_series.dt.tz) if te_e_ts.tzinfo is None else te_e_ts.tz_convert(ts_series.dt.tz)

    tr_s_idx = int(ts_series.searchsorted(tr_s_ts))
    tr_e_idx = int(ts_series.searchsorted(tr_e_ts))
    te_s_idx = int(ts_series.searchsorted(te_s_ts))
    te_e_idx = int(ts_series.searchsorted(te_e_ts))

    print(f"⏱️ Train TS: {df['timestamp'].iloc[tr_s_idx]} ➔ {df['timestamp'].iloc[tr_e_idx]} ({tr_e_idx-tr_s_idx:,} строк)")
    print(f"⏱️ Test TS:  {df['timestamp'].iloc[te_s_idx]} ➔ {df['timestamp'].iloc[min(te_e_idx, len(df)-1)]} ({te_e_idx-te_s_idx:,} строк)")

    # Выгрузка числовых массивов
    X_train = df[feature_cols].iloc[tr_s_idx:tr_e_idx].values.astype(np.float32)
    y_train_h5 = (df['target_class_7_h5_k3'].iloc[tr_s_idx:tr_e_idx].values + 3).astype(np.int32)
    y_train_h12 = (df['target_class_7_h12_k3.5'].iloc[tr_s_idx:tr_e_idx].values + 3).astype(np.int32)

    X_test = df[feature_cols].iloc[te_s_idx:te_e_idx].values.astype(np.float32)
    y_test_h5 = (df['target_class_7_h5_k3'].iloc[te_s_idx:te_e_idx].values + 3).astype(np.int32)
    y_test_h12 = (df['target_class_7_h12_k3.5'].iloc[te_s_idx:te_e_idx].values + 3).astype(np.int32)
    dt_test_h5 = df['target_dt_ticks_h5_k3'].iloc[te_s_idx:te_e_idx].values.astype(np.float32)
    dt_test_h12 = df['target_dt_ticks_h12_k3.5'].iloc[te_s_idx:te_e_idx].values.astype(np.float32)

    ts_train = df['timestamp'].iloc[tr_s_idx:tr_e_idx].astype(str).values
    ts_test = df['timestamp'].iloc[te_s_idx:te_e_idx].astype(str).values

    out_file = BASE_DIR / "data" / "prl_neural_colab_data.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Упаковка в сжатый NPZ: {out_file.relative_to(BASE_DIR)}...")
    np.savez_compressed(
        out_file,
        X_train=X_train,
        y_train_h5=y_train_h5,
        y_train_h12=y_train_h12,
        X_test=X_test,
        y_test_h5=y_test_h5,
        y_test_h12=y_test_h12,
        dt_test_h5=dt_test_h5,
        dt_test_h12=dt_test_h12,
        ts_train=ts_train,
        ts_test=ts_test,
        features=np.array(feature_cols)
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

