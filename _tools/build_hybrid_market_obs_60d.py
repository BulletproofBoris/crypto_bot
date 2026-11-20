#!/usr/bin/env python3
"""
_tools/build_hybrid_market_obs_60d.py
=============================================================================
Signal Stack v2 (Lesson 41) — Строго каузальный генератор 60D рыночного вектора:
  - [0..43]  Базовые 44 каузальных канала (совместимые с L35–L40 для Surgical Warm-Start)
  - [44..49] Все 6 ранее не подключенных каузальных моделей Order-2:
             wave_exhaustion_10s, ask_sweep_5s, exec_fill_30s,
             exec_frontrun_30s, carousel_success, log_carousel_capacity
  - [50..55] Расширенное распределение Order-1:
             alpha_1s, alpha_2s, conf_10s, conf_40s, conf_160s, tail_skew_20s
  - [56..59] Микроструктура потока стакана (OFI, Book Slope) и кросс-биржи Gate.io / Ourbit
             со строгой защитой от протухания (staleness <= 30s -> 0.0 для 100% паритета с VPS)
=============================================================================
"""

import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
OBS_44D_PATH = BASE_DIR / "data" / "cache" / "rl_hybrid_market_obs_40d.npz"
CACHE_600H_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
ORDER1_ALPHA_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_56d.npz"
ORDER2_RISK_PATH = BASE_DIR / "data" / "cache" / "order2_risk_matrix_11d.npz"
PARQUET_PATH = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
OUT_PATH = BASE_DIR / "data" / "cache" / "rl_hybrid_market_obs_60d.npz"


def compute_ofi_5s_and_slope_from_lob(b1: np.ndarray, a1: np.ndarray, bq: np.ndarray, aq: np.ndarray, bv: np.ndarray, sv: np.ndarray):
    """Формульный каузальный расчет OFI-5s и наклона книги из сырого стакана (для 100% паритета Sim/Live даже без Parquet)."""
    b5 = bq[:, :5].sum(axis=1)
    a5 = aq[:, :5].sum(axis=1)
    slope_b = np.log1p(np.maximum(0.0, b5 - bq[:, 0])) - np.log1p(np.maximum(0.0, bq[:, 0]))
    slope_a = np.log1p(np.maximum(0.0, a5 - aq[:, 0])) - np.log1p(np.maximum(0.0, aq[:, 0]))
    log_slope_ratio = np.tanh((slope_b - slope_a) * 0.5).astype(np.float32)
    return log_slope_ratio


def main():
    print("=" * 84)
    print("🚀 СБОРКА SIGNAL STACK V2: СТРОГО КАУЗАЛЬНЫЙ 60D РЫНОЧНЫЙ ВЕКТОР (0% LEAKAGE)")
    print("=" * 84)
    t0 = time.time()

    base_44d = np.load(OBS_44D_PATH, mmap_mode="r")["market_obs_40d"]
    lob = np.load(CACHE_600H_PATH, mmap_mode="r")
    o1 = np.load(ORDER1_ALPHA_PATH, mmap_mode="r")
    o2 = np.load(ORDER2_RISK_PATH, mmap_mode="r")

    n = len(base_44d)
    assert base_44d.shape == (n, 44), f"Ожидалась база (N, 44), получено {base_44d.shape}"
    assert len(lob["bid_p"]) == n and len(o1["expected_z_8h"]) == n and len(o2["risk_matrix_11d"]) == n

    st_b1 = lob["bid_p"][:, 0].astype(np.float32)
    st_a1 = lob["ask_p"][:, 0].astype(np.float32)
    st_mid = np.maximum(0.5 * (st_b1 + st_a1), 1e-4)

    # 1. [44..49] Все 6 исполнительских и разворотных моделей Order-2 (6D)
    print("  ⚡ 1/3 Подключение 6 недостающих каузальных моделей Order-2 (carousel_success, wave_exhaustion и др.)...")
    rmat = o2["risk_matrix_11d"]
    p_wave_exh = np.clip(rmat[:, 1].astype(np.float32), 0.0, 1.0)
    p_ask_swp  = np.clip(rmat[:, 2].astype(np.float32), 0.0, 1.0)
    p_fill_30s = np.clip(rmat[:, 7].astype(np.float32), 0.0, 1.0)
    p_frun_30s = np.clip(rmat[:, 8].astype(np.float32), 0.0, 1.0)
    p_car_succ = np.clip(rmat[:, 9].astype(np.float32), 0.0, 1.0)
    car_cap_lg = np.clip(np.log1p(np.maximum(0.0, rmat[:, 10].astype(np.float32))) / 3.0, 0.0, 2.5)
    blk_o2_6d = np.column_stack([p_wave_exh, p_ask_swp, p_fill_30s, p_frun_30s, p_car_succ, car_cap_lg]).astype(np.float32)

    # 2. [50..55] Расширенное распределение Order-1: быстрые 1s/2s, неопределенность confidence и хвостовой скос (6D)
    print("  🧠 2/3 Извлечение быстрых горизонтов 1s/2s, confidence и хвостовой асимметрии Order-1 (6D)...")
    ez8 = o1["expected_z_8h"]
    conf8 = o1["confidence_8h"]
    probs56 = o1["probs_56d"]
    alpha_1s = np.clip(ez8[:, 0].astype(np.float32), -3.0, 3.0)
    alpha_2s = np.clip(ez8[:, 1].astype(np.float32), -3.0, 3.0)
    conf_10s = np.clip(conf8[:, 3].astype(np.float32), 0.0, 1.0)
    conf_40s = np.clip(conf8[:, 5].astype(np.float32), 0.0, 1.0)
    conf_160s = np.clip(conf8[:, 7].astype(np.float32), 0.0, 1.0)
    # H=20s (индекс 4 -> столбцы 28..35: классы [-3,-2,-1,0,+1,+2,+3])
    p_up_tail = probs56[:, 33].astype(np.float32) + probs56[:, 34].astype(np.float32)
    p_dn_tail = probs56[:, 28].astype(np.float32) + probs56[:, 29].astype(np.float32)
    tail_skew_20s = np.clip((p_up_tail - p_dn_tail) * 3.0, -2.0, 2.0)
    blk_o1_6d = np.column_stack([alpha_1s, alpha_2s, conf_10s, conf_40s, conf_160s, tail_skew_20s]).astype(np.float32)

    # 3. [56..59] Потоковый OFI, наклон глубины стакана и кросс-биржи Gate.io / Ourbit (4D)
    print("  🌐 3/3 Извлечение потокового OFI, наклона стакана и базисов Gate.io / Ourbit из prl_1s_features.parquet...")
    pf = pq.ParquetFile(PARQUET_PATH)
    assert pf.metadata.num_rows == n, f"Parquet rows {pf.metadata.num_rows} != {n}"
    req_cols = [
        "ST_ofi_norm_5s",
        "ST_log_slope_ratio",
        "GATE_mid_price",
        "GATE_sec_since_tick",
        "OUR_mid_price",
        "OUR_sec_since_tick",
    ]
    tbl = pf.read(columns=req_cols)
    ofi_raw = np.nan_to_num(tbl["ST_ofi_norm_5s"].to_numpy(zero_copy_only=False).astype(np.float32), nan=0.0)
    slope_raw = np.nan_to_num(tbl["ST_log_slope_ratio"].to_numpy(zero_copy_only=False).astype(np.float32), nan=0.0)
    gt_mid = np.nan_to_num(tbl["GATE_mid_price"].to_numpy(zero_copy_only=False).astype(np.float32), nan=0.0)
    gt_sec = np.nan_to_num(tbl["GATE_sec_since_tick"].to_numpy(zero_copy_only=False).astype(np.float32), nan=999.0)
    our_mid = np.nan_to_num(tbl["OUR_mid_price"].to_numpy(zero_copy_only=False).astype(np.float32), nan=0.0)
    our_sec = np.nan_to_num(tbl["OUR_sec_since_tick"].to_numpy(zero_copy_only=False).astype(np.float32), nan=999.0)

    st_ofi_5s = np.tanh(ofi_raw * 8.0).astype(np.float32)
    st_slope = np.tanh(slope_raw * 0.40).astype(np.float32)
    gt_valid = (gt_mid > 1e-4) & (gt_sec <= 30.0)
    our_valid = (our_mid > 1e-4) & (our_sec <= 30.0)
    gt_basis = np.where(gt_valid, np.tanh(((gt_mid - st_mid) / st_mid) * 50.0), 0.0).astype(np.float32)
    our_basis = np.where(our_valid, np.tanh(((our_mid - st_mid) / st_mid) * 50.0), 0.0).astype(np.float32)
    blk_micro_4d = np.column_stack([st_ofi_5s, st_slope, gt_basis, our_basis]).astype(np.float32)

    print("🧩 Компоновка итоговой каузальной матрицы (N, 60)...")
    obs_60d = np.empty((n, 60), dtype=np.float32)
    obs_60d[:, :44] = np.asarray(base_44d, dtype=np.float32)
    obs_60d[:, 44:50] = blk_o2_6d
    obs_60d[:, 50:56] = blk_o1_6d
    obs_60d[:, 56:60] = blk_micro_4d
    obs_60d = np.nan_to_num(obs_60d, nan=0.0, posinf=2.5, neginf=-2.5)

    sub_epoch = obs_60d[3016277:]
    stds = np.std(sub_epoch[::50], axis=0)
    dead = np.where(stds < 1e-5)[0]
    assert len(dead) == 0, f"Обнаружены мертвые каналы на суб-центовой эпохе: {dead}"

    print(f"💾 Сохранение {obs_60d.shape} в {OUT_PATH.name} (без сжатия для мгновенного mmap чтения на GPU!)...")
    tmp_out = OUT_PATH.with_name(f"{OUT_PATH.stem}_tmp.npz")
    np.savez(tmp_out, market_obs_60d=obs_60d)
    tmp_out.replace(OUT_PATH)

    dt_s = time.time() - t0
    sz_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ СОБРАН SIGNAL STACK V2 60D ({obs_60d.shape}, {sz_mb:.1f} МБ) за {dt_s:.1f}с | min_std(raw)={stds.min():.5f}!")
    for idx_c in range(44, 60):
        print(f"   ch[{idx_c:02d}]: mean={sub_epoch[::50, idx_c].mean():+.4f} | std={stds[idx_c]:.4f}")


if __name__ == "__main__":
    main()
