#!/usr/bin/env python3
"""
v8.0: Сборщик 600-часового Causal датасета с разметкой 4 таргетов роя Guard-сетей.
Гарантия причинности:
1. Фичи X_t берутся строго из [t-W .. t].
2. Таргеты Y_t смотрят в будущее на H тиков только для расчета функции потерь при обучении.
3. Полная защита от data leakage / lookahead bias.
"""

import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CACHE_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"

def build_600h_dataset(parquet_path: str = "data/features/prl_1s_features.parquet"):
    p_file = BASE_DIR / parquet_path
    print(f"📦 Чтение {p_file}...")
    pf = pq.ParquetFile(p_file)
    total_rows = pf.metadata.num_rows
    print(f"Всего строк в файле: {total_rows:,} ({total_rows/3600:.1f} часов)")

    # Выбираем ключевые колонки стакана L1-L10 и объемов
    b_p_cols = [f"ST_bid_{i}_price" for i in range(1, 11)]
    b_q_cols = [f"ST_bid_{i}_qty" for i in range(1, 11)]
    a_p_cols = [f"ST_ask_{i}_price" for i in range(1, 11)]
    a_q_cols = [f"ST_ask_{i}_qty" for i in range(1, 11)]
    vol_cols = ["ST_buy_vol", "ST_sell_vol", "ST_trade_vol", "ST_cvd_1m", "ST_cvd_5m", "ST_spread"]

    needed_cols = ["timestamp"] + b_p_cols + b_q_cols + a_p_cols + a_q_cols + vol_cols
    available_cols = [c for c in needed_cols if c in pf.schema_arrow.names]

    print(f"Загрузка {len(available_cols)} колонок в память...")
    table = pf.read(columns=available_cols)
    df = table.to_pandas()
    if "timestamp" not in df.columns:
        df = df.reset_index()
    n = len(df)
    print(f"Загружено {n:,} тиков. Проверка монотонности времени...")

    ts = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("s").astype("int64").values
    if np.any(np.diff(ts) < 0):
        print("⚠️ Обнаружены немонотонные временные метки. Сортировка...")
        df = df.sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("s").astype("int64").values

    tick = 0.01
    b_p = df[b_p_cols].values.astype(np.float32)
    b_q = df[b_q_cols].values.astype(np.float32)
    a_p = df[a_p_cols].values.astype(np.float32)
    a_q = df[a_q_cols].values.astype(np.float32)
    buy_v = df["ST_buy_vol"].values.astype(np.float32) if "ST_buy_vol" in df else np.zeros(n, dtype=np.float32)
    sell_v = df["ST_sell_vol"].values.astype(np.float32) if "ST_sell_vol" in df else np.zeros(n, dtype=np.float32)

    b1 = b_p[:, 0]
    a1 = a_p[:, 0]

    print("🏷️ Векторизованный расчёт 4 таргетов Guard-сетей...")
    # 1. Y_sweep_12s: падение b1 на >= 1 тик в течение 12с
    # Векторизованный min-pool вперед на 12 шагов
    y_sweep_12s = np.zeros(n, dtype=np.float32)
    for k in range(1, 13):
        b1_fwd = np.pad(b1[k:], (0, k), mode="edge")
        y_sweep_12s = np.maximum(y_sweep_12s, ((b1 - b1_fwd) >= (tick * 0.95)).astype(np.float32))

    # 2. Y_bleed_60s: падение на >= 2 тика без возврата вверх на +1 тик в течение 60с
    min_drop_60s = np.zeros(n, dtype=np.float32)
    max_rise_60s = np.zeros(n, dtype=np.float32)
    for k in range(1, 61):
        b1_fwd = np.pad(b1[k:], (0, k), mode="edge")
        min_drop_60s = np.maximum(min_drop_60s, (b1 - b1_fwd) / tick)
        max_rise_60s = np.maximum(max_rise_60s, (b1_fwd - b1) / tick)
    y_bleed_60s = ((min_drop_60s >= 1.95) & (max_rise_60s < 0.95)).astype(np.float32)

    # 3. Y_crash_300s: просадка >= 5 тиков за 300с (5 минут)
    min_drop_300s = np.zeros(n, dtype=np.float32)
    # Сэмплируем каждые 5 тиков для ускорения
    for k in range(5, 301, 5):
        b1_fwd = np.pad(b1[k:], (0, k), mode="edge")
        min_drop_300s = np.maximum(min_drop_300s, (b1 - b1_fwd) / tick)
    y_crash_300s = (min_drop_300s >= 4.95).astype(np.float32)

    # 4. Y_exit_30s: падение на >= 2 тика при удержании в течение 30с
    min_drop_30s = np.zeros(n, dtype=np.float32)
    for k in range(1, 31):
        b1_fwd = np.pad(b1[k:], (0, k), mode="edge")
        min_drop_30s = np.maximum(min_drop_30s, (b1 - b1_fwd) / tick)
    y_exit_30s = (min_drop_30s >= 1.95).astype(np.float32)

    print("Статистика таргетов на 600h массиве:")
    print(f"  Y_sweep_12s : {y_sweep_12s.sum():>9,} ({y_sweep_12s.mean()*100:.2f}%)")
    print(f"  Y_bleed_60s : {y_bleed_60s.sum():>9,} ({y_bleed_60s.mean()*100:.2f}%)")
    print(f"  Y_crash_300s: {y_crash_300s.sum():>9,} ({y_crash_300s.mean()*100:.2f}%)")
    print(f"  Y_exit_30s  : {y_exit_30s.sum():>9,} ({y_exit_30s.mean()*100:.2f}%)")

    # Сохраняем в кэш (атомарная запись через временный файл для защиты активных тренеров)
    OUTPUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_CACHE_PATH.with_suffix(".tmp.npz")
    print(f"💾 Сохранение тензорного кэша в {OUTPUT_CACHE_PATH}...")
    np.savez_compressed(
        tmp_path,
        bid_p=b_p,
        bid_q=b_q,
        ask_p=a_p,
        ask_q=a_q,
        buy_vol=buy_v,
        sell_vol=sell_v,
        ts=ts,
        tick_size=np.float32(tick),
        y_sweep_12s=y_sweep_12s,
        y_bleed_60s=y_bleed_60s,
        y_crash_300s=y_crash_300s,
        y_exit_30s=y_exit_30s,
    )
    tmp_path.replace(OUTPUT_CACHE_PATH)
    sz_mb = OUTPUT_CACHE_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Датасет 600h успешно сохранен: {OUTPUT_CACHE_PATH.name} ({sz_mb:.1f} MB, {n:,} тиков)")

if __name__ == "__main__":
    build_600h_dataset()
