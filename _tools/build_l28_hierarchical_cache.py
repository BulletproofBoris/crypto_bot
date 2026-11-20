#!/usr/bin/env python3
"""
_tools/build_l28_hierarchical_cache.py
=============================================================================
Модернизированный турбо-инкрементальный сборщик кэша Урока 28 (БЕЗ 90s):
  1) 100% каузальные веса и EMA иерархического дерева (`prl` и `prl_bo`) на всей
     истории `[0 : N]` по чистой сетке `[1s, 2s, 5s, 12s, 25s, 60s, 120s]`;
  2) Компактное хранение активного окна `[eval_start_row : N]` в чистом `float32`
     (без 1.75M строк мёртвого нулевого префикса и без шума квантования `float16`);
  3) Исправленная `compute_tree_coherence_4d` (строго по активным горизонтам без
     искажений от выключенных `ST 1s/2s` на эпохе `$0.01`);
  4) Исправленная агрегация 50 субцентовых уровней `$0.0001` в 10 центовых корзин
     `$0.01` (`0 <= k < 10`, без искусственной «фальшивой стенки» на 10-м центе!);
  5) Полноценный режим `--incremental` (`append-only` с точным переносом каузального
     `EMA`-состояния роутеров — досчёт свежих суток с VPS за ~10–20 секунд!).
=============================================================================
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path("/home/restorator/crypto_bot")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.hierarchical_regime_tree import (
    L28_HORIZONS_ASC,
    TREE_LEVELS_DESC,
    HierarchicalRegimeTreeRouter,
    build_or_load_tree_index,
    compute_tree_coherence_4d,
    extract_venue_invariant_states,
    required_router_state_columns,
)

SUBCENT_ERA_START_DEFAULT = 3_016_277
FEAT_TICK_PRL = 0.01
ACTIVE_EVAL_SPAN_ROWS = 1_350_000


def compute_subcent_micro_8d_single(
    bp_arr: np.ndarray,
    bq_arr: np.ndarray,
    ap_arr: np.ndarray,
    aq_arr: np.ndarray,
    buy_v1: float,
    sell_v1: float,
    exec_tick: float,
) -> np.ndarray:
    """
    Живой 1-строчный расчёт первых 8 физических каналов `subcent_micro_16d[:8]` (100% паритет с батчем!).
    """
    b1 = float(bp_arr[0]) if len(bp_arr) > 0 and bp_arr[0] > 0 else 1.50
    a1 = float(ap_arr[0]) if len(ap_arr) > 0 and ap_arr[0] > b1 else b1 + max(exec_tick, 1e-4)
    mid = max(0.5 * (b1 + a1), 0.05)

    b1_q = max(0.0, float(bq_arr[0])) if len(bq_arr) > 0 else 0.0
    a1_q = max(0.0, float(aq_arr[0])) if len(aq_arr) > 0 else 0.0

    sp_bps = min(max((a1 - b1) / mid * 1e4, 0.5), 400.0)
    log_sp = math.log1p(sp_bps) / 4.0

    mp = (b1 * a1_q + a1 * b1_q) / max(b1_q + a1_q, 1e-4) if (b1_q + a1_q) > 1e-4 else mid
    mp_bps = min(max(((mp - mid) / mid * 1e4) / 20.0, -2.5), 2.5)

    bp64 = np.asarray(bp_arr, dtype=np.float64)
    ap64 = np.asarray(ap_arr, dtype=np.float64)
    bq64 = np.maximum(np.asarray(bq_arr, dtype=np.float64), 0.0)
    aq64 = np.maximum(np.asarray(aq_arr, dtype=np.float64), 0.0)

    qb_10 = float(np.sum(bq64[(bp64 > 0.0) & (bp64 >= b1 * (1.0 - 0.0010) - 1e-7)]))
    qa_10 = float(np.sum(aq64[(ap64 > 0.0) & (ap64 <= a1 * (1.0 + 0.0010) + 1e-7)]))
    imb_10 = (qb_10 - qa_10) / (qb_10 + qa_10 + 10.0)

    qb_50 = float(np.sum(bq64[(bp64 > 0.0) & (bp64 >= b1 * (1.0 - 0.0050) - 1e-7)]))
    qa_50 = float(np.sum(aq64[(ap64 > 0.0) & (ap64 <= a1 * (1.0 + 0.0050) + 1e-7)]))
    ld_b50 = min(max(math.log1p(max(0.0, qb_50 * mid)) / 10.0, 0.0), 2.5)
    ld_a50 = min(max(math.log1p(max(0.0, qa_50 * mid)) / 10.0, 0.0), 2.5)

    qb_100 = float(np.sum(bq64[(bp64 > 0.0) & (bp64 >= b1 * (1.0 - 0.0100) - 1e-7)]))
    qa_100 = float(np.sum(aq64[(ap64 > 0.0) & (ap64 <= a1 * (1.0 + 0.0100) + 1e-7)]))
    imb_100 = (qb_100 - qa_100) / (qb_100 + qa_100 + 25.0)

    bv = max(0.0, float(buy_v1))
    sv = max(0.0, float(sell_v1))
    fast_tape = min(max(((bv - sv) / (bv + sv + 25.0)) * 2.0, -2.0), 2.0)
    is_sub = 1.0 if exec_tick <= 0.001 else 0.0

    return np.array([log_sp, mp_bps, imb_10, imb_100, ld_b50, ld_a50, fast_tape, is_sub], dtype=np.float32)


def compute_subcent_micro_8d_batch(
    bp_mat: np.ndarray,
    bq_mat: np.ndarray,
    ap_mat: np.ndarray,
    aq_mat: np.ndarray,
    buy_v: np.ndarray,
    sell_v: np.ndarray,
    is_subcent_flag: float,
) -> np.ndarray:
    """Векторизованный расчёт первых 8 каналов `subcent_micro_16d[:8]` на произвольной матрице [M, L]."""
    bp64 = np.asarray(bp_mat, dtype=np.float64)
    ap64 = np.asarray(ap_mat, dtype=np.float64)
    bq64 = np.maximum(np.asarray(bq_mat, dtype=np.float64), 0.0)
    aq64 = np.maximum(np.asarray(aq_mat, dtype=np.float64), 0.0)

    b1 = np.maximum(bp64[:, 0], 0.01)
    a1 = np.maximum(ap64[:, 0], b1 + (0.0001 if is_subcent_flag > 0.5 else 0.01))
    mid = np.maximum(0.5 * (b1 + a1), 0.05)

    b1_q = bq64[:, 0]
    a1_q = aq64[:, 0]

    sp_bps = np.clip((a1 - b1) / mid * 1e4, 0.5, 400.0)
    log_sp = np.log1p(sp_bps) / 4.0

    sum_q1 = b1_q + a1_q
    mp = np.where(sum_q1 > 1e-4, (b1 * a1_q + a1 * b1_q) / np.maximum(sum_q1, 1e-4), mid)
    mp_bps = np.clip(((mp - mid) / mid * 1e4) / 20.0, -2.5, 2.5)

    b_pos = bp64 > 0.0
    a_pos = ap64 > 0.0

    qb_10 = np.sum(bq64 * (b_pos & (bp64 >= b1[:, None] * (1.0 - 0.0010) - 1e-7)), axis=1)
    qa_10 = np.sum(aq64 * (a_pos & (ap64 <= a1[:, None] * (1.0 + 0.0010) + 1e-7)), axis=1)
    imb_10 = (qb_10 - qa_10) / (qb_10 + qa_10 + 10.0)

    qb_50 = np.sum(bq64 * (b_pos & (bp64 >= b1[:, None] * (1.0 - 0.0050) - 1e-7)), axis=1)
    qa_50 = np.sum(aq64 * (a_pos & (ap64 <= a1[:, None] * (1.0 + 0.0050) + 1e-7)), axis=1)
    ld_b50 = np.clip(np.log1p(np.maximum(0.0, qb_50 * mid)) / 10.0, 0.0, 2.5)
    ld_a50 = np.clip(np.log1p(np.maximum(0.0, qa_50 * mid)) / 10.0, 0.0, 2.5)

    qb_100 = np.sum(bq64 * (b_pos & (bp64 >= b1[:, None] * (1.0 - 0.0100) - 1e-7)), axis=1)
    qa_100 = np.sum(aq64 * (a_pos & (ap64 <= a1[:, None] * (1.0 + 0.0100) + 1e-7)), axis=1)
    imb_100 = (qb_100 - qa_100) / (qb_100 + qa_100 + 25.0)

    bv = np.maximum(0.0, np.asarray(buy_v, dtype=np.float64))
    sv = np.maximum(0.0, np.asarray(sell_v, dtype=np.float64))
    fast_tape = np.clip(((bv - sv) / (bv + sv + 25.0)) * 2.0, -2.0, 2.0)
    is_sub = np.full(len(b1), float(is_subcent_flag), dtype=np.float32)

    return np.column_stack([log_sp, mp_bps, imb_10, imb_100, ld_b50, ld_a50, fast_tape, is_sub]).astype(np.float32)


def aggregate_subcent_50_to_cent_10(
    bp50: np.ndarray,
    bq50: np.ndarray,
    ap50: np.ndarray,
    aq50: np.ndarray,
    feat_tick: float = FEAT_TICK_PRL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Точная векторизованная копия `live_rl_trader.py:2075-2089` (БЕЗ фальшивой стенки на 10-м центе!):
    Агрегирует субцентовые ($0.0001) уровни строго внутри реального 10-центового окна `0 <= k < 10`.
    """
    m = len(bp50)
    b1 = np.maximum(bp50[:, 0], 0.0001)
    a1 = np.maximum(ap50[:, 0], b1 + 0.0001)

    b_cent_top = np.floor((b1 + 1e-7) / feat_tick) * feat_tick
    a_cent_top = np.maximum(b_cent_top + feat_tick, np.ceil((a1 - 1e-7) / feat_tick) * feat_tick)

    ks = np.arange(10, dtype=np.float64)[None, :]
    cent_bp10 = np.round(np.maximum(feat_tick, b_cent_top[:, None] - ks * feat_tick), 6).astype(np.float32)
    cent_ap10 = np.round(a_cent_top[:, None] + ks * feat_tick, 6).astype(np.float32)

    cent_bq10 = np.zeros((m, 10), dtype=np.float32)
    cent_aq10 = np.zeros((m, 10), dtype=np.float32)

    rows = np.arange(m)
    for j in range(bp50.shape[1]):
        pj = bp50[:, j]
        qj = bq50[:, j]
        kb = np.floor((b_cent_top - pj + 1e-7) / feat_tick).astype(np.int64)
        valid_b = (pj > 0.0) & (qj > 0.0) & (kb >= 0) & (kb < 10)
        cent_bq10[rows[valid_b], kb[valid_b]] += qj[valid_b]

        paj = ap50[:, j]
        qaj = aq50[:, j]
        ka = np.floor((paj - a_cent_top + 1e-7) / feat_tick).astype(np.int64)
        valid_a = (paj > 0.0) & (qaj > 0.0) & (ka >= 0) & (ka < 10)
        cent_aq10[rows[valid_a], ka[valid_a]] += qaj[valid_a]

    return cent_bp10, cent_bq10, cent_ap10, cent_aq10


def detect_subcent_era_start(bp1_series: np.ndarray) -> int:
    """Определяет точный индекс начала субцентовой эры ($0.0001), где дробная часть цен < $0.01."""
    frac_cents = np.abs(bp1_series * 100.0 - np.round(bp1_series * 100.0))
    hits = np.where(frac_cents > 1e-3)[0]
    if len(hits) == 0:
        return SUBCENT_ERA_START_DEFAULT
    tail_hits = hits[hits >= int(len(bp1_series) * 0.80)]
    return int(tail_hits[0]) if len(tail_hits) > 0 else int(hits[0])


def read_l28_cache_global_slice(
    raw_l28,
    key: str,
    s0_global: int,
    s1_global: int,
    dtype=np.float32,
) -> np.ndarray:
    """
    Универсальное чтение глобального среза `[s0_global : s1_global]` из `l28_hierarchical_subcent_cache.npz`
    (прозрачно поддерживает как компактный формат с `eval_start_row`, так и полноразмерные массивы `[0 : N]`).
    """
    eval_s0 = int(raw_l28["eval_start_row"]) if "eval_start_row" in raw_l28.files else 0
    arr = raw_l28[key]
    n_total = int(raw_l28["total_rows"]) if "total_rows" in raw_l28.files else len(raw_l28["ts"])
    if len(arr) == n_total:
        return np.array(arr[s0_global:s1_global], dtype=dtype, copy=True)
    assert s0_global >= eval_s0, f"Запрошен глобальный индекс {s0_global:,} левее eval_start_row={eval_s0:,}!"
    return np.array(arr[(s0_global - eval_s0) : (s1_global - eval_s0)], dtype=dtype, copy=True)


def _extract_sub50_arrays(labeled_path: Path, start_row: int, length: int, bp10_fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lvl50_cols = []
    for k in range(1, 51):
        lvl50_cols.extend([f"ST_bid_{k}_price", f"ST_bid_{k}_qty", f"ST_ask_{k}_price", f"ST_ask_{k}_qty"])
    df_sub50 = pq.read_table(labeled_path, columns=lvl50_cols).slice(start_row, length).to_pandas()

    sub_bp50 = np.zeros((length, 50), dtype=np.float32)
    sub_bq50 = np.zeros((length, 50), dtype=np.float32)
    sub_ap50 = np.zeros((length, 50), dtype=np.float32)
    sub_aq50 = np.zeros((length, 50), dtype=np.float32)
    for k in range(50):
        lv = k + 1
        sub_bp50[:, k] = np.nan_to_num(df_sub50[f"ST_bid_{lv}_price"].to_numpy(dtype=np.float32), nan=0.0)
        sub_bq50[:, k] = np.maximum(0.0, np.nan_to_num(df_sub50[f"ST_bid_{lv}_qty"].to_numpy(dtype=np.float32), nan=0.0))
        sub_ap50[:, k] = np.nan_to_num(df_sub50[f"ST_ask_{lv}_price"].to_numpy(dtype=np.float32), nan=0.0)
        sub_aq50[:, k] = np.maximum(0.0, np.nan_to_num(df_sub50[f"ST_ask_{lv}_qty"].to_numpy(dtype=np.float32), nan=0.0))
    del df_sub50
    gc.collect()

    sub_bp50[:, 0] = np.where(sub_bp50[:, 0] > 0.0, sub_bp50[:, 0], bp10_fallback)
    sub_ap50[:, 0] = np.where(sub_ap50[:, 0] > sub_bp50[:, 0], sub_ap50[:, 0], sub_bp50[:, 0] + 0.0001)
    return sub_bp50, sub_bq50, sub_ap50, sub_aq50


def _pack_ema_dict(prefix: str, ema_by_h: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_ema_h{int(h)}": np.asarray(v, dtype=np.float32) for h, v in ema_by_h.items()}


def _unpack_ema_dict(npz_obj, prefix: str) -> dict[int, np.ndarray]:
    out = {}
    for H, _ in TREE_LEVELS_DESC:
        k = f"{prefix}_ema_h{int(H)}"
        if k in npz_obj.files:
            out[int(H)] = np.asarray(npz_obj[k], dtype=np.float32)
    return out


def run_incremental_append(
    out_path: Path,
    labeled_path: Path,
    lob_cache_path: Path,
) -> bool:
    """
    Быстрое `Append-Only` дописывание только новых секунд `[old_n : new_n]` (~10–20с на сутки новых данных!).
    Возвращает `True`, если инкрементальное обновление успешно выполнено.
    """
    if not out_path.exists():
        return False
    prev = np.load(out_path, allow_pickle=False)
    if "eval_start_row" not in prev.files or "total_rows" not in prev.files:
        prev.close()
        return False

    old_n = int(prev["total_rows"])
    eval_s0 = int(prev["eval_start_row"])
    era_start = int(prev["subcent_era_start"])
    st_ema_init = _unpack_ema_dict(prev, "st")
    bo_ema_init = _unpack_ema_dict(prev, "bo")
    if not st_ema_init or not bo_ema_init:
        prev.close()
        return False

    pf = pq.ParquetFile(labeled_path)
    new_n = pf.metadata.num_rows
    if new_n == old_n:
        print(f"✅ Инкрементальный кэш уже актуален (`N = {new_n:,}` строк) — пересчёт не требуется!", flush=True)
        prev.close()
        return True
    if new_n < old_n:
        prev.close()
        return False

    delta_n = new_n - old_n
    t0 = time.time()
    print(f"⚡ [INCREMENTAL APPEND] Дописываем только ΔN = {delta_n:,} новых секунд ({old_n:,} -> {new_n:,})...", flush=True)

    all_schema_cols = set(pf.schema_arrow.names)
    state_cols = list(dict.fromkeys([c for c in required_router_state_columns() + ["BO_sec_since_tick"] if c in all_schema_cols]))
    df_tail = pq.read_table(labeled_path, columns=state_cols).slice(old_n, delta_n).to_pandas()
    raw_ts = df_tail["timestamp"] if "timestamp" in df_tail.columns else df_tail.index
    ts_tail = pd.to_datetime(np.asarray(raw_ts), utc=True).as_unit("s").astype("int64").to_numpy()
    bo_sec_col = df_tail["BO_sec_since_tick"] if "BO_sec_since_tick" in df_tail.columns else None
    if bo_sec_col is not None and getattr(bo_sec_col, "ndim", 1) > 1:
        bo_sec_col = bo_sec_col.iloc[:, 0]
    bo_stale_tail = (
        (np.nan_to_num(bo_sec_col.to_numpy(dtype=np.float32), nan=999.0) > 120.0).reshape(-1)
        if bo_sec_col is not None
        else np.zeros(delta_n, dtype=bool)
    )

    st_router = HierarchicalRegimeTreeRouter("prl", use_hybrid_nn=True)
    X_st = np.load(BASE_DIR / "data" / "_mmap_cache" / "X_all.npy", mmap_mode="r")
    st_tree_tail, _, st_coh_tail = st_router.predict_all_7_horizons_batch(
        X_st,
        extract_venue_invariant_states(df_tail, "prl"),
        ts_tail,
        stale_mask=None,
        min_eval_row=0,
        ema_init_by_h=st_ema_init,
        x_global_offset=old_n,
    )
    st_ema_last = dict(st_router._last_batch_ema_by_h)
    del st_router

    bo_router = HierarchicalRegimeTreeRouter("prl_bo", use_hybrid_nn=True)
    X_bo = np.load(BASE_DIR / "data" / "_mmap_cache_bo" / "X_all.npy", mmap_mode="r")
    bo_tree_tail, _, bo_coh_tail = bo_router.predict_all_7_horizons_batch(
        X_bo,
        extract_venue_invariant_states(df_tail, "prl_bo"),
        ts_tail,
        stale_mask=bo_stale_tail,
        min_eval_row=0,
        ema_init_by_h=bo_ema_init,
        x_global_offset=old_n,
    )
    bo_ema_last = dict(bo_router._last_batch_ema_by_h)
    del bo_router, df_tail
    gc.collect()

    lob_npz = np.load(lob_cache_path, mmap_mode="r")
    bp10_tail = np.asarray(lob_npz["bid_p"][old_n:new_n, 0], dtype=np.float32)
    buy_v_tail = np.asarray(lob_npz["buy_vol"][old_n:new_n], dtype=np.float32)
    sell_v_tail = np.asarray(lob_npz["sell_vol"][old_n:new_n], dtype=np.float32)

    sub_bp50_t, sub_bq50_t, sub_ap50_t, sub_aq50_t = _extract_sub50_arrays(labeled_path, old_n, delta_n, bp10_tail)
    cent_bp10_t, cent_bq10_t, cent_ap10_t, cent_aq10_t = aggregate_subcent_50_to_cent_10(
        sub_bp50_t, sub_bq50_t, sub_ap50_t, sub_aq50_t, feat_tick=FEAT_TICK_PRL
    )

    m8_tail = compute_subcent_micro_8d_batch(
        sub_bp50_t, sub_bq50_t, sub_ap50_t, sub_aq50_t, buy_v_tail, sell_v_tail, is_subcent_flag=1.0
    )
    st_1s2s_open_t = (np.sum(np.abs(st_tree_tail[:, :12]), axis=1) > 1e-5).astype(np.float32)
    m8_tail[:, 7] = 0.5 * m8_tail[:, 7] + 0.5 * st_1s2s_open_t
    micro_tail = np.column_stack([m8_tail, st_coh_tail, bo_coh_tail]).astype(np.float32)

    save_dict = {
        "ts": np.concatenate([prev["ts"], ts_tail], axis=0),
        "total_rows": np.int64(new_n),
        "eval_start_row": np.int64(eval_s0),
        "subcent_era_start": np.int64(era_start),
        "st_tree_42d": np.concatenate([prev["st_tree_42d"].astype(np.float32), st_tree_tail], axis=0),
        "bo_tree_42d": np.concatenate([prev["bo_tree_42d"].astype(np.float32), bo_tree_tail], axis=0),
        "subcent_micro_16d": np.concatenate([prev["subcent_micro_16d"].astype(np.float32), micro_tail], axis=0),
        "cent_bid_p10": np.concatenate([prev["cent_bid_p10"], cent_bp10_t], axis=0),
        "cent_bid_q10": np.concatenate([prev["cent_bid_q10"], cent_bq10_t], axis=0),
        "cent_ask_p10": np.concatenate([prev["cent_ask_p10"], cent_ap10_t], axis=0),
        "cent_ask_q10": np.concatenate([prev["cent_ask_q10"], cent_aq10_t], axis=0),
        "subcent_bid_p50": np.concatenate([prev["subcent_bid_p50"], sub_bp50_t], axis=0),
        "subcent_bid_q50": np.concatenate([prev["subcent_bid_q50"], sub_bq50_t], axis=0),
        "subcent_ask_p50": np.concatenate([prev["subcent_ask_p50"], sub_ap50_t], axis=0),
        "subcent_ask_q50": np.concatenate([prev["subcent_ask_q50"], sub_aq50_t], axis=0),
        **_pack_ema_dict("st", st_ema_last),
        **_pack_ema_dict("bo", bo_ema_last),
    }
    prev.close()
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **save_dict)
    tmp_path.replace(out_path)
    sz_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(f"✅ [INCREMENTAL APPEND] Дописано {delta_n:,} строк за {time.time() - t0:.1f}с -> {out_path.name} ({sz_mb:.1f} МБ)!", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental", action="store_true", help="Быстрое инкрементальное дописывание только новых строк")
    parser.add_argument("--force-full", action="store_true", help="Игнорировать промежуточные _tmp_l28_* стейджи и пересчитать с нуля")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 108, flush=True)
    print("🚀 СБОРКА МОДЕРНИЗИРОВАННОГО ИЕРАРХИЧЕСКОГО И СУБЦЕНТОВОГО КЭША УРОКА 28 (БЕЗ 90s, FLOAT32)", flush=True)
    print("=" * 108, flush=True)

    for c_name in ("prl", "prl_bo"):
        build_or_load_tree_index(c_name, force_rebuild=False)

    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    lob_cache_path = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
    out_path = BASE_DIR / "data" / "cache" / "l28_hierarchical_subcent_cache.npz"

    if args.incremental and not args.force_full:
        if run_incremental_append(out_path, labeled_path, lob_cache_path):
            return

    pf = pq.ParquetFile(labeled_path)
    n_rows = pf.metadata.num_rows
    all_schema_cols = set(pf.schema_arrow.names)

    state_cols = list(dict.fromkeys([c for c in required_router_state_columns() + ["BO_sec_since_tick"] if c in all_schema_cols]))
    print(f"📥 Загрузка {len(state_cols)} колонок состояния на {n_rows:,} секунд...", flush=True)
    df_state = pq.read_table(labeled_path, columns=state_cols).to_pandas()
    raw_ts = df_state["timestamp"] if "timestamp" in df_state.columns else df_state.index
    ts_sec = pd.to_datetime(np.asarray(raw_ts), utc=True).as_unit("s").astype("int64").to_numpy()
    bo_sec_col = df_state["BO_sec_since_tick"] if "BO_sec_since_tick" in df_state.columns else None
    if bo_sec_col is not None and getattr(bo_sec_col, "ndim", 1) > 1:
        bo_sec_col = bo_sec_col.iloc[:, 0]
    bo_stale = (
        (np.nan_to_num(bo_sec_col.to_numpy(dtype=np.float32), nan=999.0) > 120.0).reshape(-1)
        if bo_sec_col is not None
        else np.zeros(n_rows, dtype=bool)
    )
    assert bo_stale.shape == (n_rows,), f"Unexpected bo_stale shape: {bo_stale.shape}"

    min_eval_row = max(0, n_rows - ACTIVE_EVAL_SPAN_ROWS)
    st_stage_path = BASE_DIR / "data" / "cache" / "_tmp_l28_st_stage.npz"
    bo_stage_path = BASE_DIR / "data" / "cache" / "_tmp_l28_bo_stage.npz"

    st_router = HierarchicalRegimeTreeRouter("prl", use_hybrid_nn=True)
    st_states = extract_venue_invariant_states(df_state, "prl")
    if st_stage_path.exists() and not args.force_full:
        print(f"♻️ [1/4] Восстановление матрицы `st_tree_42` из {st_stage_path.name} + пересчёт чистой `coherence_4d` и EMA...", flush=True)
        d_st = np.load(st_stage_path)
        st_raw_tree = d_st["tree"]
        st_tree_act = np.array(st_raw_tree[min_eval_row:n_rows] if len(st_raw_tree) == n_rows else st_raw_tree, dtype=np.float32, copy=True)
        d_st.close()
        st_router.compute_hierarchical_weights_batch(st_states, ts_sec)
        st_ema_last = dict(st_router._last_batch_ema_by_h)
    else:
        print(f"🌲 [1/4] Турбо-прогон иерархического дерева SafeTrade (PRL: 1s..120s, eval_from={min_eval_row:,})...", flush=True)
        X_st = np.load(BASE_DIR / "data" / "_mmap_cache" / "X_all.npy", mmap_mode="r")
        st_full_42, _, _ = st_router.predict_all_7_horizons_batch(X_st, st_states, ts_sec, stale_mask=None, min_eval_row=min_eval_row)
        st_tree_act = np.array(st_full_42[min_eval_row:n_rows], dtype=np.float32, copy=True)
        st_ema_last = dict(st_router._last_batch_ema_by_h)
        np.savez(st_stage_path, tree=st_tree_act)
        del st_full_42
    st_coh_act = compute_tree_coherence_4d(st_tree_act)
    del st_router, st_states
    gc.collect()

    bo_router = HierarchicalRegimeTreeRouter("prl_bo", use_hybrid_nn=True)
    bo_states = extract_venue_invariant_states(df_state, "prl_bo")
    if bo_stage_path.exists() and not args.force_full:
        print(f"♻️ [2/4] Восстановление матрицы `bo_tree_42` из {bo_stage_path.name} + пересчёт чистой `coherence_4d` и EMA...", flush=True)
        d_bo = np.load(bo_stage_path)
        bo_raw_tree = d_bo["tree"]
        bo_tree_act = np.array(bo_raw_tree[min_eval_row:n_rows] if len(bo_raw_tree) == n_rows else bo_raw_tree, dtype=np.float32, copy=True)
        d_bo.close()
        bo_router.compute_hierarchical_weights_batch(bo_states, ts_sec)
        bo_ema_last = dict(bo_router._last_batch_ema_by_h)
    else:
        print(f"🌲 [2/4] Турбо-прогон иерархического дерева BigONE (PRL_BO: 1s..120s, eval_from={min_eval_row:,})...", flush=True)
        X_bo = np.load(BASE_DIR / "data" / "_mmap_cache_bo" / "X_all.npy", mmap_mode="r")
        bo_full_42, _, _ = bo_router.predict_all_7_horizons_batch(X_bo, bo_states, ts_sec, stale_mask=bo_stale, min_eval_row=min_eval_row)
        bo_tree_act = np.array(bo_full_42[min_eval_row:n_rows], dtype=np.float32, copy=True)
        bo_ema_last = dict(bo_router._last_batch_ema_by_h)
        np.savez(bo_stage_path, tree=bo_tree_act)
        del bo_full_42
    bo_stale_act = bo_stale[min_eval_row:n_rows]
    bo_tree_act[bo_stale_act] = 0.0
    bo_coh_act = compute_tree_coherence_4d(bo_tree_act)
    bo_coh_act[bo_stale_act] = 0.0
    del bo_router, bo_states, df_state
    gc.collect()

    print("📚 [3/4] Построение 50-уровневой книги $0.0001, чистых $0.01 корзин (без bucket-9 wall) и float32 16D микро-блока...", flush=True)
    lob_npz = np.load(lob_cache_path, mmap_mode="r")
    bp10_all = lob_npz["bid_p"]
    bq10_all = lob_npz["bid_q"]
    ap10_all = lob_npz["ask_p"]
    aq10_all = lob_npz["ask_q"]
    buy_v_all = np.asarray(lob_npz["buy_vol"], dtype=np.float32)
    sell_v_all = np.asarray(lob_npz["sell_vol"], dtype=np.float32)

    era_start = detect_subcent_era_start(np.asarray(bp10_all[:, 0], dtype=np.float64))
    sub_len = n_rows - era_start
    print(f"  🔎 Субцентовая эра $0.0001: era_start={era_start:,} .. {n_rows:,} ({sub_len:,} с = {sub_len / 3600.0:.2f} ч)", flush=True)

    sub_bp50, sub_bq50, sub_ap50, sub_aq50 = _extract_sub50_arrays(
        labeled_path, era_start, sub_len, np.asarray(bp10_all[era_start:, 0], dtype=np.float32)
    )
    cent_bp10, cent_bq10, cent_ap10, cent_aq10 = aggregate_subcent_50_to_cent_10(
        sub_bp50, sub_bq50, sub_ap50, sub_aq50, feat_tick=FEAT_TICK_PRL
    )

    act_len = n_rows - min_eval_row
    sub_local_start = era_start - min_eval_row
    micro_16d_act = np.zeros((act_len, 16), dtype=np.float32)

    chunk_m = 250000
    for c0 in range(min_eval_row, era_start, chunk_m):
        c1 = min(era_start, c0 + chunk_m)
        m8 = compute_subcent_micro_8d_batch(
            np.asarray(bp10_all[c0:c1], dtype=np.float32),
            np.asarray(bq10_all[c0:c1], dtype=np.float32),
            np.asarray(ap10_all[c0:c1], dtype=np.float32),
            np.asarray(aq10_all[c0:c1], dtype=np.float32),
            buy_v_all[c0:c1],
            sell_v_all[c0:c1],
            is_subcent_flag=0.0,
        )
        st_open_c = (np.sum(np.abs(st_tree_act[(c0 - min_eval_row) : (c1 - min_eval_row), :12]), axis=1) > 1e-5).astype(np.float32)
        m8[:, 7] = 0.5 * m8[:, 7] + 0.5 * st_open_c
        micro_16d_act[(c0 - min_eval_row) : (c1 - min_eval_row), :8] = m8

    m8_sub = compute_subcent_micro_8d_batch(
        sub_bp50, sub_bq50, sub_ap50, sub_aq50,
        buy_v_all[era_start:], sell_v_all[era_start:],
        is_subcent_flag=1.0,
    )
    st_open_sub = (np.sum(np.abs(st_tree_act[sub_local_start:, :12]), axis=1) > 1e-5).astype(np.float32)
    m8_sub[:, 7] = 0.5 * m8_sub[:, 7] + 0.5 * st_open_sub
    micro_16d_act[sub_local_start:, :8] = m8_sub
    micro_16d_act[:, 8:12] = st_coh_act
    micro_16d_act[:, 12:16] = bo_coh_act

    # Строгая самопроверка 1-в-1 паритета batch vs live single на последней строке субцентовой эры:
    m8_single = compute_subcent_micro_8d_single(
        sub_bp50[-1], sub_bq50[-1], sub_ap50[-1], sub_aq50[-1],
        float(buy_v_all[-1]), float(sell_v_all[-1]), exec_tick=0.0001,
    )
    m8_single[7] = 0.5 * m8_single[7] + 0.5 * (1.0 if float(np.sum(np.abs(st_tree_act[-1, :12]))) > 1e-5 else 0.0)
    diff_m8 = float(np.max(np.abs(m8_single - micro_16d_act[-1, :8])))
    assert diff_m8 < 1e-5, f"Micro-8D parity mismatch: {diff_m8}"

    print(f"💾 [4/4] Запись чистого float32 кэша {out_path} (eval_start_row={min_eval_row:,}, rows={act_len:,})...", flush=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        ts=ts_sec,
        total_rows=np.int64(n_rows),
        eval_start_row=np.int64(min_eval_row),
        subcent_era_start=np.int64(era_start),
        st_tree_42d=st_tree_act,
        bo_tree_42d=bo_tree_act,
        subcent_micro_16d=micro_16d_act,
        cent_bid_p10=cent_bp10,
        cent_bid_q10=cent_bq10,
        cent_ask_p10=cent_ap10,
        cent_ask_q10=cent_aq10,
        subcent_bid_p50=sub_bp50,
        subcent_bid_q50=sub_bq50,
        subcent_ask_p50=sub_ap50,
        subcent_ask_q50=sub_aq50,
        **_pack_ema_dict("st", st_ema_last),
        **_pack_ema_dict("bo", bo_ema_last),
    )
    tmp_path.replace(out_path)
    sz_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(f"✅ Готово за {time.time() - t0:.1f}с! Файл {out_path.name} ({sz_mb:.1f} МБ, float32), micro_8d parity diff={diff_m8:.2e}", flush=True)


if __name__ == "__main__":
    main()
