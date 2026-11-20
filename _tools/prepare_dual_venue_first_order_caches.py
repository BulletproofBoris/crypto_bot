#!/usr/bin/env python3
"""
_tools/prepare_dual_venue_first_order_caches.py
=============================================================================
100% CPU-пайплайн подготовки данных и бинарных кэшей для моделей 1-го порядка
SafeTrade (`prl` -> `data/_mmap_cache`) и BigONE (`prl_bo` -> `data/_mmap_cache_bo`)
БЕЗ единого обращения к GPU (безопасно работает параллельно с обучением RL-агента):
  1. Сборка свежих 1-секундных признаков (`_tools.prepare_1s_dataset --coin prl`).
  2. Разметка всех 6 пар `(H, K) = [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]`
     для SafeTrade (`_tools.data_labeler`).
  3. Симметричная разметка тех же 6 пар `(H, K)` для BigONE (`target_bo_class_7_h{H}_k{K}`,
     `target_bo_dt_ticks_h{H}_k{K}`) в единой экономической шкале шага цены `$0.005`
     с микроструктурным динамическим сопротивлением книги BigONE `M_bo(t, H)`.
  4. Отбор `299` стационарных признаков `selected_features_prl_bo_h8s.json` строго на
     историческом Train-срезе (без малейшей утечки в OOS) с приоритетом книги и потока BigONE.
  5. Низкоуровневая low-RAM сборка обоих кэшей `data/_mmap_cache` (ST) и `data/_mmap_cache_bo` (BO).
=============================================================================
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.data_labeler import TH_P1_P2, TH_P2_P3, TH_M1_M2, TH_M2_M3
from _tools.train_rolling_swarm import find_clean_segments, generate_rolling_windows_reverse

HORIZON_PAIRS = [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]


def run_cpu_cmd(cmd: list[str]):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "8"
    print(f"▶ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(BASE_DIR), env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Команда завершилась с кодом {res.returncode}: {' '.join(cmd)}")


def compute_bigone_single_horizon_labels(df_bo: pd.DataFrame, H: int, K: float, eff_tick: float = 0.005, min_vol: float = 50.0) -> dict:
    """
    Симметричная разметка 7 квантильных классов (-3..+3) для стакана и потока BigONE
    в единой экономической единице шага цены `eff_tick = $0.005` (сопоставимой с SafeTrade).
    """
    n = len(df_bo)
    L = int(round(H * K))
    mid = np.nan_to_num(df_bo["BO_mid_price"].values.astype(np.float64), nan=0.0)
    b1_p = np.nan_to_num(df_bo["BO_bid_1_price"].values.astype(np.float64), nan=0.0)
    a1_p = np.nan_to_num(df_bo["BO_ask_1_price"].values.astype(np.float64), nan=0.0)
    spread_ticks = np.maximum(0.2, np.nan_to_num(df_bo["BO_spread"].values.astype(np.float64), nan=eff_tick) / eff_tick)
    b1_q = np.maximum(0.0, np.nan_to_num(df_bo["BO_bid_1_qty"].values.astype(np.float64), nan=0.0))
    a1_q = np.maximum(0.0, np.nan_to_num(df_bo["BO_ask_1_qty"].values.astype(np.float64), nan=0.0))
    b2_q = np.maximum(0.0, np.nan_to_num(df_bo["BO_bid_2_qty"].values.astype(np.float64), nan=0.0)) if "BO_bid_2_qty" in df_bo.columns else b1_q
    a2_q = np.maximum(0.0, np.nan_to_num(df_bo["BO_ask_2_qty"].values.astype(np.float64), nan=0.0)) if "BO_ask_2_qty" in df_bo.columns else a1_q
    b5 = np.maximum(0.0, np.nan_to_num(df_bo["BO_bids_vol_top5"].values.astype(np.float64), nan=0.0))
    a5 = np.maximum(0.0, np.nan_to_num(df_bo["BO_asks_vol_top5"].values.astype(np.float64), nan=0.0))
    trade_vol = np.maximum(0.0, np.nan_to_num(df_bo["BO_trade_vol"].values.astype(np.float64), nan=0.0))
    sec_stale = np.nan_to_num(df_bo["BO_sec_since_tick"].values.astype(np.float64), nan=999.0)

    # Непрерывный экономический сдвиг в единицах $0.005 (без грубого округления микро-шага $0.0001 до нуля, но с защитой от шума < 0.25 eff_tick)
    dt_ticks_float = np.zeros(n, dtype=np.float32)
    dt_ticks_float[:-H] = ((mid[H:] - mid[:-H]) / eff_tick).astype(np.float32)
    valid_now_fut = (mid > eff_tick) & (np.pad(mid[H:], (0, H), mode="edge") > eff_tick) & (sec_stale <= 60.0)
    dt_ticks_float[~valid_now_fut] = 0.0

    L_vol = max(45, L)
    step_1s_ticks = np.zeros(n, dtype=np.float64)
    step_1s_ticks[1:] = np.clip((mid[1:] - mid[:-1]) / eff_tick, -20.0, 20.0)
    vol_L = pd.Series(step_1s_ticks).rolling(L_vol, min_periods=max(5, L_vol // 2)).std().fillna(0.0).values
    vol_L = np.nan_to_num(vol_L, nan=0.0)

    q12_ask = a1_q + a2_q
    q12_bid = b1_q + b2_q
    eps_q = max(1e-4, min_vol * 0.01)

    a12_L = pd.Series(q12_ask).rolling(L_vol, min_periods=max(5, L_vol // 2)).median().fillna(1.0).values
    b12_L = pd.Series(q12_bid).rolling(L_vol, min_periods=max(5, L_vol // 2)).median().fillna(1.0).values
    asks_L = pd.Series(a5).rolling(L_vol, min_periods=max(5, L_vol // 2)).mean().fillna(min_vol).values
    bids_L = pd.Series(b5).rolling(L_vol, min_periods=max(5, L_vol // 2)).mean().fillna(min_vol).values

    total_q1 = b1_q + a1_q
    microprice = np.where(total_q1 > 1e-6, (b1_p * a1_q + a1_p * b1_q) / total_q1, mid)
    micro_offset = np.clip((microprice - mid) / eff_tick, -5.0, 5.0)

    t_wave = max(3, H // 3)
    v_wave = (pd.Series(micro_offset) - pd.Series(micro_offset).shift(t_wave)).fillna(0).values / t_wave
    v_wave = np.clip(v_wave, -2.0, 2.0)

    med_spread = pd.Series(spread_ticks).rolling(300, min_periods=30).median().bfill().values
    med_spread = np.maximum(1.0, np.nan_to_num(med_spread, nan=1.0))
    spread_ratio = np.clip(spread_ticks / med_spread, 0.2, 5.0)

    bid_dist = np.maximum(0.5, (mid - b1_p) / eff_tick + 0.5)
    ask_dist = np.maximum(0.5, (a1_p - mid) / eff_tick + 0.5)
    slope_bid = b5 / bid_dist
    slope_ask = a5 / ask_dist
    slope_ratio = np.clip(slope_bid / np.maximum(slope_ask, 1e-4), 0.3, 3.0)

    h_scale = (5.0 / (H + 4.0)) ** 0.5
    dyn_sp_up = 0.22 * h_scale * (spread_ratio ** (-0.25)) * (slope_ratio ** 0.15)
    dyn_sp_dn = 0.22 * h_scale * (spread_ratio ** (-0.25)) * (slope_ratio ** (-0.15))

    depth_w12 = (5.0 / (H + 5.0)) * 0.08
    depth_w5 = (H / (H + 5.0)) * 0.08
    depth_up = (np.clip(asks_L / np.maximum(a5, min_vol), 0.4, 2.5) ** depth_w5) * (np.clip(a12_L / np.maximum(q12_ask, eps_q), 0.5, 2.0) ** depth_w12)
    depth_dn = (np.clip(bids_L / np.maximum(b5, min_vol), 0.4, 2.5) ** depth_w5) * (np.clip(b12_L / np.maximum(q12_bid, eps_q), 0.5, 2.0) ** depth_w12)

    kin_coef = 0.10 + 0.05 * (H / (H + 20.0))
    kin_up = 1.0 + kin_coef * np.tanh(np.maximum(0.0, v_wave))
    kin_dn = 1.0 + kin_coef * np.tanh(np.maximum(0.0, -v_wave))

    M_up = np.maximum(0.5, np.nan_to_num((spread_ticks * dyn_sp_up + vol_L * np.sqrt(H)) * depth_up * kin_up, nan=1.0))
    M_dn = np.maximum(0.5, np.nan_to_num((spread_ticks * dyn_sp_dn + vol_L * np.sqrt(H)) * depth_dn * kin_dn, nan=1.0))

    z = np.zeros(n, dtype=np.float32)
    # На BigONE минимальный осмысленный экономический импульс >= 0.35 * eff_tick ($0.00175), чтобы отсечь шум $0.0001
    pos_mask = dt_ticks_float >= 0.35
    neg_mask = dt_ticks_float <= -0.35
    z[pos_mask] = (dt_ticks_float[pos_mask] / M_up[pos_mask]).astype(np.float32)
    z[neg_mask] = (dt_ticks_float[neg_mask] / M_dn[neg_mask]).astype(np.float32)
    z = np.nan_to_num(z, nan=0.0)

    # Фантомными считаем только скачки >= 10 экон. тиков ($0.05) при полном отсутствии сделок и устаревшем стакане > 30с
    trade_cumsum = np.pad(np.cumsum(trade_vol), (1, 0))
    trade_vol_fut = np.zeros(n, dtype=np.float32)
    trade_vol_fut[:-H] = trade_cumsum[1 + H : n + 1] - trade_cumsum[1 : n - H + 1]
    fut_stale = np.pad(sec_stale[H:], (0, H), mode="edge")
    is_void = (np.abs(dt_ticks_float) >= 10.0) & (trade_vol_fut == 0.0) & (fut_stale > 30.0)

    class_7 = np.zeros(n, dtype=np.int8)
    class_7[pos_mask & (z < TH_P1_P2)] = 1
    class_7[pos_mask & (z >= TH_P1_P2) & (z < TH_P2_P3)] = 2
    class_7[pos_mask & (z >= TH_P2_P3)] = 3
    class_7[neg_mask & (z > TH_M1_M2)] = -1
    class_7[neg_mask & (z <= TH_M1_M2) & (z > TH_M2_M3)] = -2
    class_7[neg_mask & (z <= TH_M2_M3)] = -3
    class_7[is_void] = 0
    dt_clean = np.where(is_void, 0.0, dt_ticks_float).astype(np.float16)

    return {"class_7": class_7, "dt_ticks": dt_clean}


def label_bigone_targets_into_parquet(labeled_path: Path):
    print("\n" + "=" * 95, flush=True)
    print("🎯 [3/5] СИММЕТРИЧНАЯ РАЗМЕТКА BIGONE (6 ГОРИЗОНТОВ H=5,12,25,60,90,120s, eff_tick=$0.005)", flush=True)
    print("=" * 95, flush=True)
    t0 = time.time()
    bo_req_cols = [
        "BO_mid_price", "BO_bid_1_price", "BO_ask_1_price", "BO_spread",
        "BO_bid_1_qty", "BO_ask_1_qty", "BO_bid_2_qty", "BO_ask_2_qty",
        "BO_bids_vol_top5", "BO_asks_vol_top5", "BO_trade_vol", "BO_sec_since_tick",
    ]
    schema_names = set(pq.read_schema(labeled_path).names)
    load_cols = [c for c in bo_req_cols if c in schema_names]
    df_bo = pq.read_table(labeled_path, columns=load_cols).to_pandas()

    extra_bo_cols = {}
    for h, k in HORIZON_PAIRS:
        k_str = f"{k:g}"
        res = compute_bigone_single_horizon_labels(df_bo, H=h, K=k, eff_tick=0.005, min_vol=50.0)
        c7_name = f"target_bo_class_7_h{h}_k{k_str}"
        dt_name = f"target_bo_dt_ticks_h{h}_k{k_str}"
        extra_bo_cols[c7_name] = res["class_7"]
        extra_bo_cols[dt_name] = res["dt_ticks"]
        pct_non_zero = float((res["class_7"] != 0).mean() * 100.0)
        print(f"  ✅ BigONE H={h:3d}s K={k_str:<3s}: active_impulse={pct_non_zero:5.1f}% | classes={np.unique(res['class_7'], return_counts=True)[1].tolist()}", flush=True)

    del df_bo
    gc.collect()

    base_tbl = pq.read_table(labeled_path)
    existing = set(base_tbl.schema.names)
    for col_name, col_arr in extra_bo_cols.items():
        pa_arr = pa.array(col_arr)
        if col_name in existing:
            idx_c = base_tbl.schema.get_field_index(col_name)
            base_tbl = base_tbl.set_column(idx_c, col_name, pa_arr)
        else:
            base_tbl = base_tbl.append_column(col_name, pa_arr)
            existing.add(col_name)
    del extra_bo_cols
    gc.collect()

    tmp_p = str(labeled_path) + ".bo_tmp"
    pq.write_table(base_tbl, tmp_p, compression="zstd")
    os.replace(tmp_p, labeled_path)
    print(f"💾 Таргеты BigONE записаны в {labeled_path.name} ({base_tbl.num_rows:,} строк, {base_tbl.num_columns} колонок) за {time.time() - t0:.1f}с!", flush=True)
    del base_tbl
    gc.collect()


def select_bigone_299_features(labeled_path: Path):
    """
    Формирует `data/features/selected_features_prl_bo_h8s.json` (ровно 299 стационарных признаков,
    совместимых с архитектурами .keras):
      - Исключает любые сырые уровни цен (`*_price`, `*_vwap_*`) и таргеты (`target_*`, `sample_split`, `timestamp`);
      - Оценивает важность всех стационарных признаков на исторических 70% выборки (0 утечки в OOS!)
        на паре LightGBM (короткий H=12s + макро H=60s таргеты BigONE) + зеркальное ядро `ST <-> BO`.
    """
    import lightgbm as lgb

    out_json = BASE_DIR / "data" / "features" / "selected_features_prl_bo_h8s.json"
    st_json = BASE_DIR / "data" / "features" / "selected_features_prl_h8s.json"
    st_feats = json.loads(st_json.read_text())["feature_order"]
    n_req = len(st_feats)  # 299

    pf = pq.ParquetFile(labeled_path)
    n_rows = pf.metadata.num_rows
    arrow_schema = pf.schema_arrow
    causal_feat_names = set(pq.read_schema(BASE_DIR / "data" / "features" / "prl_1s_features.parquet").names)
    all_cols = [c for c in arrow_schema.names if c in causal_feat_names]

    banned_sub = ("target_", "future_", "timestamp", "sample_split", "is_data_gap", "_price", "_vwap_", "block_height")
    allowed_exact = {"ST_microprice_offset_ticks", "BO_microprice_offset_ticks", "mid_price_diff"}
    cand_cols = []
    for c in all_cols:
        if c in allowed_exact:
            cand_cols.append(c)
            continue
        if any(b in c for b in banned_sub):
            continue
        f_type = str(arrow_schema.field(c).type)
        if any(t in f_type for t in ("float", "double", "int")):
            cand_cols.append(c)

    print("\n" + "=" * 95, flush=True)
    print(f"🔍 [4/5] ОТБОР {n_req} СТАЦИОНАРНЫХ ПРИЗНАКОВ ДЛЯ BIGONE SWARM (из {len(cand_cols)} кандидатов, строго на 70% Train)", flush=True)
    print("=" * 95, flush=True)

    # Берем равномерный шаг по первым 70% истории (строго до OOS и свежих окон!)
    tr_end = int(n_rows * 0.70)
    sub_step = max(1, tr_end // 180000)
    idx_sub = np.arange(1000, tr_end, sub_step, dtype=np.int64)

    X_sub = np.empty((len(idx_sub), len(cand_cols)), dtype=np.float32)
    for c0 in range(0, len(cand_cols), 60):
        c1 = min(len(cand_cols), c0 + 60)
        sub_names = cand_cols[c0:c1]
        tbl = pq.read_table(labeled_path, columns=sub_names)
        for j, cn in enumerate(sub_names):
            arr = tbl.column(cn).to_numpy(zero_copy_only=False)
            X_sub[:, c0 + j] = np.nan_to_num(arr[idx_sub].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        del tbl
        gc.collect()

    t_tbl = pq.read_table(labeled_path, columns=["target_bo_class_7_h12_k3.5", "target_bo_class_7_h60_k3", "BO_sec_since_tick"])
    y12 = t_tbl.column("target_bo_class_7_h12_k3.5").to_numpy(zero_copy_only=False)[idx_sub].astype(np.int32) + 3
    y60 = t_tbl.column("target_bo_class_7_h60_k3").to_numpy(zero_copy_only=False)[idx_sub].astype(np.int32) + 3
    bo_sec = np.nan_to_num(t_tbl.column("BO_sec_since_tick").to_numpy(zero_copy_only=False)[idx_sub].astype(np.float32), nan=999.0)
    del t_tbl
    valid_m = bo_sec <= 60.0
    X_sub = X_sub[valid_m]
    y12 = y12[valid_m]
    y60 = y60[valid_m]

    gains = np.zeros(len(cand_cols), dtype=np.float64)
    for y_t in (y12, y60):
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=7, n_estimators=90, learning_rate=0.07,
            num_leaves=31, subsample=0.8, colsample_bytree=0.7, n_jobs=8, verbose=-1, random_state=42
        )
        clf.fit(X_sub, y_t)
        g = clf.booster_.feature_importance(importance_type="gain").astype(np.float64)
        gains += g / (g.sum() + 1e-12)
    del X_sub
    gc.collect()

    # Небольшой приоритет нативным BO_* и межбиржевым признакам при равном информационном выигрыше
    for i_c, cn in enumerate(cand_cols):
        if cn.startswith(("BO_", "leadlag_", "arb_spread_", "LL_")):
            gains[i_c] *= 1.25

    order_idx = np.argsort(-gains)[:n_req]
    selected_bo = [cand_cols[i] for i in order_idx]
    assert len(selected_bo) == n_req, f"Expected {n_req} features, got {len(selected_bo)}"

    payload = {
        "coin": "prl_bo",
        "venue": "bigone",
        "n_features": len(selected_bo),
        "feature_order": selected_bo,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    n_bo = sum(1 for c in selected_bo if c.startswith("BO_"))
    n_st = sum(1 for c in selected_bo if c.startswith("ST_"))
    n_other = len(selected_bo) - n_bo - n_st
    print(f"  ✅ Сохранен {out_json.name}: {len(selected_bo)} признаков (BO_*={n_bo}, ST_*={n_st}, Cross/Macro/OnChain={n_other})", flush=True)
    print(f"     Топ-12 признаков BigONE: {selected_bo[:12]}", flush=True)


def build_mmap_cache_for_venue(labeled_path: Path, feats_json: Path, cache_subdir: str, is_bo: bool):
    t0 = time.time()
    feature_cols = json.loads(feats_json.read_text())["feature_order"]
    pf = pq.ParquetFile(labeled_path)
    n_rows = pf.metadata.num_rows
    cache_dir = BASE_DIR / "data" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old_f in cache_dir.glob("*_idx_*.npy"):
        old_f.unlink(missing_ok=True)

    tag = "BIGONE (prl_bo)" if is_bo else "SAFETRADE (prl)"
    print("\n" + "=" * 95, flush=True)
    print(f"⚙️ [5/5] СБОРКА БИНАРНОГО КЭША data/{cache_subdir} ДЛЯ {tag} ({n_rows:,} СТРОК x {len(feature_cols)} ФИЧ)", flush=True)
    print("=" * 95, flush=True)

    X_all = np.empty((n_rows, len(feature_cols)), dtype=np.float32)
    for c0 in range(0, len(feature_cols), 50):
        c1 = min(len(feature_cols), c0 + 50)
        cols_sub = feature_cols[c0:c1]
        tbl = pq.read_table(labeled_path, columns=cols_sub)
        for j, col_name in enumerate(cols_sub):
            X_all[:, c0 + j] = np.nan_to_num(tbl.column(col_name).to_numpy(zero_copy_only=False).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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

    p_pref = "target_bo" if is_bo else "target"
    meta_cols = [
        "timestamp", "sample_split", "is_data_gap", "ST_mid_price", "ST_trade_vol", "BO_mid_price", "BO_trade_vol",
        f"{p_pref}_class_7_h5_k3", f"{p_pref}_dt_ticks_h5_k3",
        f"{p_pref}_class_7_h12_k3.5", f"{p_pref}_dt_ticks_h12_k3.5",
        f"{p_pref}_class_7_h25_k3", f"{p_pref}_dt_ticks_h25_k3",
        f"{p_pref}_class_7_h60_k3", f"{p_pref}_dt_ticks_h60_k3",
        f"{p_pref}_class_7_h90_k3", f"{p_pref}_dt_ticks_h90_k3",
        f"{p_pref}_class_7_h120_k3", f"{p_pref}_dt_ticks_h120_k3",
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
    print(f"  ⏱️ Сетка окон {tag}: " + ", ".join(f"H={h}s ({len(wins_map[f'h{h}'])})" for h, _ in HORIZON_PAIRS), flush=True)

    arrays = {
        "ts_all": df["timestamp"].astype(str).values,
        "y_all_h5": (df[f"{p_pref}_class_7_h5_k3"].fillna(0).values + 3).astype(np.int8),
        "y_all_h12": (df[f"{p_pref}_class_7_h12_k3.5"].fillna(0).values + 3).astype(np.int8),
        "y_all_h25": (df[f"{p_pref}_class_7_h25_k3"].fillna(0).values + 3).astype(np.int8),
        "y_all_h60": (df[f"{p_pref}_class_7_h60_k3"].fillna(0).values + 3).astype(np.int8),
        "y_all_h90": (df[f"{p_pref}_class_7_h90_k3"].fillna(0).values + 3).astype(np.int8),
        "y_all_h120": (df[f"{p_pref}_class_7_h120_k3"].fillna(0).values + 3).astype(np.int8),
        "dt_all_h5": df[f"{p_pref}_dt_ticks_h5_k3"].fillna(0).values.astype(np.float16),
        "dt_all_h12": df[f"{p_pref}_dt_ticks_h12_k3.5"].fillna(0).values.astype(np.float16),
        "dt_all_h25": df[f"{p_pref}_dt_ticks_h25_k3"].fillna(0).values.astype(np.float16),
        "dt_all_h60": df[f"{p_pref}_dt_ticks_h60_k3"].fillna(0).values.astype(np.float16),
        "dt_all_h90": df[f"{p_pref}_dt_ticks_h90_k3"].fillna(0).values.astype(np.float16),
        "dt_all_h120": df[f"{p_pref}_dt_ticks_h120_k3"].fillna(0).values.astype(np.float16),
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
    print(f"  🎉 Кэш data/{cache_subdir} готов за {time.time() - t0:.1f}с!", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-labeled", action="store_true", help="Пропустить шаги 1..3, если prl_labeled.parquet (807 колонок) уже собран")
    ap.add_argument("--only-bo-cache", action="store_true", help="Пересобрать только признаки и кэш BigONE (_mmap_cache_bo)")
    args = ap.parse_args()

    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    if not args.skip_labeled:
        py = str(BASE_DIR / ".venv" / "bin" / "python")
        print("=" * 95, flush=True)
        print("🚀 [1/5] ПЕРЕСБОРКА СВЕЖЕГО 1-СЕКУНДНОГО ДАТАСЕТА ПРИЗНАКОВ (CPU ONLY)", flush=True)
        print("=" * 95, flush=True)
        run_cpu_cmd([py, "-u", "-m", "_tools.prepare_1s_dataset", "--coin", "prl"])

        print("\n" + "=" * 95, flush=True)
        print("🎯 [2/5] РАЗМЕТКА ВСЕХ 6 ПАР (H, K) ДЛЯ SAFETRADE (CPU ONLY)", flush=True)
        print("=" * 95, flush=True)
        run_cpu_cmd([py, "-u", "-m", "_tools.data_labeler", "--coin", "prl", "--pairs", "5:3,12:3.5,25:3,60:3,90:3,120:3"])

        label_bigone_targets_into_parquet(labeled_path)

    select_bigone_299_features(labeled_path)

    if not args.only_bo_cache:
        build_mmap_cache_for_venue(
            labeled_path=labeled_path,
            feats_json=BASE_DIR / "data" / "features" / "selected_features_prl_h8s.json",
            cache_subdir="_mmap_cache",
            is_bo=False,
        )
    build_mmap_cache_for_venue(
        labeled_path=labeled_path,
        feats_json=BASE_DIR / "data" / "features" / "selected_features_prl_bo_h8s.json",
        cache_subdir="_mmap_cache_bo",
        is_bo=True,
    )
    print("\n🏆 ВСЕ ДАННЫЕ, РАЗМЕТКИ (ST + BO) И ОБА MMAP-КЭША ПОЛНОСТЬЮ ГОТОВЫ К ОБУЧЕНИЮ МОДЕЛЕЙ 1-ГО ПОРЯДКА!", flush=True)


if __name__ == "__main__":
    main()
