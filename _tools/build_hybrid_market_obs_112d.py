#!/usr/bin/env python3
"""
_tools/build_hybrid_market_obs_112d.py
=============================================================================
Signal Stack v8 (Lesson 93: 112D Cross-Venue Lead-Lag, Macro BTC/NEAR & Deep L50/L500 Walls):
  - Repairs ch[58] (Gate.io basis) across full history using the 1Hz-reconstructed `gate_depth.parquet` (99%+ coverage instead of 64%!)
  - Appends 16 strictly causal channels ch[96..111] (0% lookahead, 100% Sim/Live parity):
      [96]  gate_ret_15s             [-1..+1]
      [97]  gate_imb_top5            [-1..+1]
      [98]  gate_cvd_60s             [-1..+1]
      [99]  gate_cvd_300s            [-1..+1]
      [100] ourbit_ret_15s           [-1..+1]
      [101] ourbit_cvd_60s           [-1..+1]
      [102] ourbit_cvd_300s          [-1..+1]
      [103] cross_lead_consensus     [-1..+1]
      [104] btc_ret_60s              [-1..+1]
      [105] btc_ret_900s             [-1..+1]
      [106] near_ret_60s             [-1..+1]
      [107] near_ret_900s            [-1..+1]
      [108] st_deep_imb_2pct         [-1..+1]
      [109] st_deep_imb_5pct         [-1..+1]
      [110] st_work_bid_wall_dist    [0..1.5]
      [111] st_work_bid_wall_str     [0..1.0]
  - Also saves `data/cache/rl_deep_lob_profile.npz` (per-second work_bid_wall_px, cum_bid_2pct, cum_bid_4pct)
    for realistic L50/L500 queue simulation and Wall-Anchored Placement in `fused_order3_clean_tick.py`!
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
CACHE_DIR = BASE_DIR / "data" / "cache"


def causal_rolling_sum(arr: np.ndarray, win: int) -> np.ndarray:
    cs = np.empty(len(arr) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(arr, dtype=np.float64, out=cs[1:])
    idx_hi = np.arange(1, len(arr) + 1, dtype=np.int64)
    idx_lo = np.maximum(0, idx_hi - win)
    return (cs[idx_hi] - cs[idx_lo]).astype(np.float32)


def _to_unix_s(ser) -> np.ndarray:
    return pd.to_datetime(ser, utc=True).astype("datetime64[s, UTC]").astype("int64").to_numpy()


def _align_asof_1s(ts_grid_s: np.ndarray, src_s: np.ndarray, vals: np.ndarray, max_stale_s: float = 45.0) -> tuple[np.ndarray, np.ndarray]:
    """Strictly causal backward searchsorted alignment (t_src <= t_grid, 0% lookahead)."""
    src_s = src_s.astype(np.int64)
    order = np.argsort(src_s, kind="mergesort")
    src_s = src_s[order]
    vals = vals[order]
    pos = np.searchsorted(src_s, ts_grid_s, side="right") - 1
    valid = pos >= 0
    pos_clip = np.maximum(pos, 0)
    age = np.where(valid, ts_grid_s - src_s[pos_clip], 999999)
    fresh = valid & (age <= max_stale_s)
    out = vals[pos_clip].copy()
    if out.ndim == 1:
        out[~fresh] = 0.0
    else:
        out[~fresh, :] = 0.0
    return out, fresh


def _bin_trades_1s(ts_grid_s: np.ndarray, parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Causal 1-second buy/sell volume binning aligned to ts_grid_s."""
    n = len(ts_grid_s)
    buy_1s = np.zeros(n, dtype=np.float32)
    sell_1s = np.zeros(n, dtype=np.float32)
    if not parquet_path.exists():
        return buy_1s, sell_1s
    tbl = pq.read_table(parquet_path, columns=["timestamp", "amount", "side"])
    ts_s = _to_unix_s(tbl["timestamp"].to_pandas())
    amt = np.nan_to_num(pd.to_numeric(tbl["amount"].to_pandas(), errors="coerce").to_numpy(dtype=np.float32), nan=0.0)
    side = tbl["side"].to_pandas().astype(str).str.lower().to_numpy()
    # Map exact second to grid index via searchsorted
    idx = np.searchsorted(ts_grid_s, ts_s, side="left")
    in_bounds = (idx >= 0) & (idx < n) & (ts_grid_s[np.minimum(idx, n - 1)] == ts_s)
    idx_v = idx[in_bounds]
    amt_v = np.maximum(0.0, amt[in_bounds])
    is_buy = np.isin(side[in_bounds], ["buy", "bid", "b"])
    np.add.at(buy_1s, idx_v[is_buy], amt_v[is_buy])
    np.add.at(sell_1s, idx_v[~is_buy], amt_v[~is_buy])
    return buy_1s, sell_1s


def _extract_st_deep_profile(ts_grid_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts causal L50/L500 SafeTrade deep liquidity & protective wall in chunks to keep RAM low:
      returns (cum_b_2p, cum_a_2p, cum_b_5p, cum_a_5p, work_wall_px, work_wall_qty) aligned to ts_grid_s.
    """
    pf = pq.ParquetFile(PROC_DIR / "safetrade_depth.parquet")
    schema_names = set(pf.schema.names)
    bp_cols = [f"bid_{i}_price" for i in range(1, 51) if f"bid_{i}_price" in schema_names]
    bq_cols = [f"bid_{i}_qty" for i in range(1, 51) if f"bid_{i}_qty" in schema_names]
    ap_cols = [f"ask_{i}_price" for i in range(1, 51) if f"ask_{i}_price" in schema_names]
    aq_cols = [f"ask_{i}_qty" for i in range(1, 51) if f"ask_{i}_qty" in schema_names]
    opt_cols = [c for c in ("cum_bid_2pct", "cum_bid_4pct", "cum_bid_6pct", "cum_ask_2pct", "cum_ask_5pct", "work_bid_wall_price", "work_bid_wall_qty", "max_bid_wall_price", "max_bid_wall_qty") if c in schema_names]
    read_cols = ["timestamp"] + bp_cols + bq_cols + ap_cols + aq_cols + opt_cols

    # Only process post-Nov-09 rows where PRL active history lives (earlier rows default cleanly)
    min_grid_s = int(ts_grid_s[max(0, len(ts_grid_s) - 1_600_000)]) - 120
    ts_chunks = []
    val_chunks = []
    for rg in range(pf.num_row_groups):
        t_rg = pf.read_row_group(rg, columns=["timestamp"])
        ts_s_rg = _to_unix_s(t_rg["timestamp"].to_pandas())
        if len(ts_s_rg) == 0 or ts_s_rg[-1] < min_grid_s:
            continue
        tb = pf.read_row_group(rg, columns=read_cols)
        df = tb.to_pandas()
        bp = np.nan_to_num(df[bp_cols].to_numpy(dtype=np.float32), nan=0.0)
        bq = np.nan_to_num(df[bq_cols].to_numpy(dtype=np.float32), nan=0.0)
        ap = np.nan_to_num(df[ap_cols].to_numpy(dtype=np.float32), nan=0.0)
        aq = np.nan_to_num(df[aq_cols].to_numpy(dtype=np.float32), nan=0.0)
        b1 = np.maximum(bp[:, 0:1], 1e-4)
        a1 = np.maximum(ap[:, 0:1], 1e-4)

        cb2 = (bq * (bp >= b1 * 0.98)).sum(axis=1)
        ca2 = (aq * ((ap > 0) & (ap <= a1 * 1.02))).sum(axis=1)
        cb5 = (bq * (bp >= b1 * 0.945)).sum(axis=1)
        ca5 = (aq * ((ap > 0) & (ap <= a1 * 1.05))).sum(axis=1)

        # Work-zone bid wall in [-0.3% .. -6.5%] across L1..L50, upgraded by L500 column when present
        in_wz = (bp >= b1 * 0.935) & (bp <= b1 * 0.997)
        wz_q = np.where(in_wz, bq, 0.0)
        wz_idx = np.argmax(wz_q, axis=1)
        rows_ar = np.arange(len(df))
        ww_px = bp[rows_ar, wz_idx]
        ww_qt = wz_q[rows_ar, wz_idx]

        if "max_bid_wall_price" in df.columns and "max_bid_wall_qty" in df.columns:
            mw_p = np.nan_to_num(df["max_bid_wall_price"].to_numpy(dtype=np.float32), nan=0.0)
            mw_q = np.nan_to_num(df["max_bid_wall_qty"].to_numpy(dtype=np.float32), nan=0.0)
            mw_ok = (mw_p >= b1[:, 0] * 0.935) & (mw_p <= b1[:, 0] * 0.997) & (mw_q > ww_qt)
            ww_px = np.where(mw_ok, mw_p, ww_px)
            ww_qt = np.where(mw_ok, mw_q, ww_qt)

        if "work_bid_wall_price" in df.columns and "work_bid_wall_qty" in df.columns:
            wbp = np.nan_to_num(df["work_bid_wall_price"].to_numpy(dtype=np.float32), nan=0.0)
            wbq = np.nan_to_num(df["work_bid_wall_qty"].to_numpy(dtype=np.float32), nan=0.0)
            has_wb = (wbp > 0.0) & (wbq > 0.0)
            ww_px = np.where(has_wb, wbp, ww_px)
            ww_qt = np.where(has_wb, wbq, ww_qt)
        if "cum_bid_2pct" in df.columns:
            c2 = np.nan_to_num(df["cum_bid_2pct"].to_numpy(dtype=np.float32), nan=-1.0)
            cb2 = np.where(c2 >= 0.0, c2, cb2)
        if "cum_bid_6pct" in df.columns:
            c6 = np.nan_to_num(df["cum_bid_6pct"].to_numpy(dtype=np.float32), nan=-1.0)
            cb5 = np.where(c6 >= 0.0, c6, cb5)
        if "cum_ask_2pct" in df.columns:
            a2 = np.nan_to_num(df["cum_ask_2pct"].to_numpy(dtype=np.float32), nan=-1.0)
            ca2 = np.where(a2 >= 0.0, a2, ca2)
        if "cum_ask_5pct" in df.columns:
            a5 = np.nan_to_num(df["cum_ask_5pct"].to_numpy(dtype=np.float32), nan=-1.0)
            ca5 = np.where(a5 >= 0.0, a5, ca5)

        ts_chunks.append(ts_s_rg)
        val_chunks.append(np.column_stack([cb2, ca2, cb5, ca5, ww_px, ww_qt]).astype(np.float32))

    all_ts = np.concatenate(ts_chunks)
    all_v = np.concatenate(val_chunks, axis=0)
    aligned, _ = _align_asof_1s(ts_grid_s, all_ts, all_v, max_stale_s=30.0)
    return aligned[:, 0], aligned[:, 1], aligned[:, 2], aligned[:, 3], aligned[:, 4], aligned[:, 5]


def main() -> None:
    t0 = time.time()
    print("=" * 88)
    print("🚀 BUILDING SIGNAL STACK V8: CAUSAL 112D MARKET MATRIX + DEEP L50/L500 PROFILE (DEC + RAW)")
    print("=" * 88, flush=True)

    lob_dec = np.load(CACHE_DIR / "rl_guard_lob_cache_600h.npz", mmap_mode="r")
    ts_grid = np.asarray(lob_dec["ts"], dtype=np.int64)
    n = len(ts_grid)
    idx_ar = np.arange(n, dtype=np.int64)

    # 1. Gate.io 1Hz repaired book + trades
    print("  1/4 Aligning repaired 1Hz Gate.io orderbook & trades...", flush=True)
    gt_cols = ["timestamp"] + [f"bid_{i}_price" for i in range(1, 6)] + [f"bid_{i}_qty" for i in range(1, 6)] + [f"ask_{i}_price" for i in range(1, 6)] + [f"ask_{i}_qty" for i in range(1, 6)]
    gt_tb = pq.read_table(PROC_DIR / "gate_depth.parquet", columns=gt_cols).to_pandas()
    gt_ts = _to_unix_s(gt_tb["timestamp"])
    gt_b1 = np.nan_to_num(gt_tb["bid_1_price"].to_numpy(dtype=np.float32), nan=0.0)
    gt_a1 = np.nan_to_num(gt_tb["ask_1_price"].to_numpy(dtype=np.float32), nan=0.0)
    gt_mid_raw = 0.5 * (gt_b1 + gt_a1)
    gt_bq5 = np.nan_to_num(gt_tb[[f"bid_{i}_qty" for i in range(1, 6)]].to_numpy(dtype=np.float32), nan=0.0).sum(axis=1)
    gt_aq5 = np.nan_to_num(gt_tb[[f"ask_{i}_qty" for i in range(1, 6)]].to_numpy(dtype=np.float32), nan=0.0).sum(axis=1)
    gt_imb5_raw = (gt_bq5 - gt_aq5) / (gt_bq5 + gt_aq5 + 50.0)
    gt_mat, gt_fresh = _align_asof_1s(ts_grid, gt_ts, np.column_stack([gt_mid_raw, gt_imb5_raw]).astype(np.float32), max_stale_s=30.0)
    gt_mid = gt_mat[:, 0]
    gt_imb5 = np.clip(gt_mat[:, 1], -1.0, 1.0)
    gt_prev15 = gt_mid[np.maximum(0, idx_ar - 15)]
    gt_ret_15s = np.where(gt_fresh & (gt_prev15 > 1e-4), np.tanh((gt_mid / np.maximum(gt_prev15, 1e-4) - 1.0) * 40.0), 0.0).astype(np.float32)

    gt_bv, gt_sv = _bin_trades_1s(ts_grid, PROC_DIR / "gate_trades.parquet")
    gt_b60, gt_s60 = causal_rolling_sum(gt_bv, 60), causal_rolling_sum(gt_sv, 60)
    gt_b300, gt_s300 = causal_rolling_sum(gt_bv, 300), causal_rolling_sum(gt_sv, 300)
    gt_cvd_60s = np.clip((gt_b60 - gt_s60) / (gt_b60 + gt_s60 + 200.0), -1.0, 1.0)
    gt_cvd_300s = np.clip((gt_b300 - gt_s300) / (gt_b300 + gt_s300 + 800.0), -1.0, 1.0)

    # 2. Ourbit orderbook & trades
    print("  2/4 Aligning Ourbit orderbook & trades...", flush=True)
    ob_tb = pq.read_table(PROC_DIR / "ourbit_depth.parquet", columns=["timestamp", "bid_1_price", "ask_1_price"]).to_pandas()
    ob_ts = _to_unix_s(ob_tb["timestamp"])
    ob_mid_raw = 0.5 * (np.nan_to_num(ob_tb["bid_1_price"].to_numpy(dtype=np.float32), nan=0.0) + np.nan_to_num(ob_tb["ask_1_price"].to_numpy(dtype=np.float32), nan=0.0))
    ob_mid, ob_fresh = _align_asof_1s(ts_grid, ob_ts, ob_mid_raw, max_stale_s=30.0)
    ob_prev15 = ob_mid[np.maximum(0, idx_ar - 15)]
    ob_ret_15s = np.where(ob_fresh & (ob_prev15 > 1e-4), np.tanh((ob_mid / np.maximum(ob_prev15, 1e-4) - 1.0) * 40.0), 0.0).astype(np.float32)

    ob_bv, ob_sv = _bin_trades_1s(ts_grid, PROC_DIR / "ourbit_trades.parquet")
    ob_b60, ob_s60 = causal_rolling_sum(ob_bv, 60), causal_rolling_sum(ob_sv, 60)
    ob_b300, ob_s300 = causal_rolling_sum(ob_bv, 300), causal_rolling_sum(ob_sv, 300)
    ob_cvd_60s = np.clip((ob_b60 - ob_s60) / (ob_b60 + ob_s60 + 200.0), -1.0, 1.0)
    ob_cvd_300s = np.clip((ob_b300 - ob_s300) / (ob_b300 + ob_s300 + 800.0), -1.0, 1.0)

    # 3. Binance Macro (BTC + NEAR)
    print("  3/4 Aligning Binance Macro (BTC + NEAR)...", flush=True)
    bm_tb = pq.read_table(PROC_DIR / "binance_macro.parquet", columns=["timestamp", "btc_mid", "near_mid"]).to_pandas()
    bm_ts = _to_unix_s(bm_tb["timestamp"])
    bm_raw = np.column_stack([
        np.nan_to_num(bm_tb["btc_mid"].to_numpy(dtype=np.float32), nan=0.0),
        np.nan_to_num(bm_tb["near_mid"].to_numpy(dtype=np.float32), nan=0.0),
    ])
    bm_mat, bm_fresh = _align_asof_1s(ts_grid, bm_ts, bm_raw, max_stale_s=30.0)
    btc_m, near_m = bm_mat[:, 0], bm_mat[:, 1]
    btc_p60, btc_p900 = btc_m[np.maximum(0, idx_ar - 60)], btc_m[np.maximum(0, idx_ar - 900)]
    near_p60, near_p900 = near_m[np.maximum(0, idx_ar - 60)], near_m[np.maximum(0, idx_ar - 900)]
    btc_ret_60s = np.where(bm_fresh & (btc_p60 > 1.0), np.tanh((btc_m / np.maximum(btc_p60, 1.0) - 1.0) * 120.0), 0.0).astype(np.float32)
    btc_ret_900s = np.where(bm_fresh & (btc_p900 > 1.0), np.tanh((btc_m / np.maximum(btc_p900, 1.0) - 1.0) * 50.0), 0.0).astype(np.float32)
    near_ret_60s = np.where(bm_fresh & (near_p60 > 1e-3), np.tanh((near_m / np.maximum(near_p60, 1e-3) - 1.0) * 80.0), 0.0).astype(np.float32)
    near_ret_900s = np.where(bm_fresh & (near_p900 > 1e-3), np.tanh((near_m / np.maximum(near_p900, 1e-3) - 1.0) * 35.0), 0.0).astype(np.float32)

    # 4. SafeTrade L50/L500 Deep Profile & Work-Zone Protective Wall
    print("  4/4 Extracting SafeTrade L50/L500 deep liquidity & protective walls...", flush=True)
    cb2, ca2, cb5, ca5, ww_px, ww_qt = _extract_st_deep_profile(ts_grid)

    prof_tmp = CACHE_DIR / "rl_deep_lob_profile_tmp.npz"
    np.savez(
        prof_tmp,
        ts=ts_grid,
        cum_bid_2pct=cb2.astype(np.float32),
        cum_ask_2pct=ca2.astype(np.float32),
        cum_bid_5pct=cb5.astype(np.float32),
        cum_ask_5pct=ca5.astype(np.float32),
        work_bid_wall_px=ww_px.astype(np.float32),
        work_bid_wall_qty=ww_qt.astype(np.float32),
    )
    prof_tmp.replace(CACHE_DIR / "rl_deep_lob_profile.npz")

    for tag, in_96, in_lob, out_112 in (
        ("DEC", CACHE_DIR / "rl_hybrid_market_obs_96d.npz", CACHE_DIR / "rl_guard_lob_cache_600h.npz", CACHE_DIR / "rl_hybrid_market_obs_112d.npz"),
        ("RAW", CACHE_DIR / "rl_hybrid_market_obs_96d_raw_with_self.npz", CACHE_DIR / "rl_guard_lob_cache_600h_raw_with_self.npz", CACHE_DIR / "rl_hybrid_market_obs_112d_raw_with_self.npz"),
    ):
        if not in_96.exists() or not in_lob.exists():
            continue
        obs96 = np.load(in_96, mmap_mode="r")["market_obs_96d"]
        lob = np.load(in_lob, mmap_mode="r")
        st_b1 = np.maximum(np.asarray(lob["bid_p"][:, 0], dtype=np.float32), 1e-4)
        st_a1 = np.maximum(np.asarray(lob["ask_p"][:, 0], dtype=np.float32), 1e-4)
        st_mid = 0.5 * (st_b1 + st_a1)

        # Repaired ch[58] Gate basis!
        gt_basis_rep = np.where(gt_fresh & (gt_mid > 1e-4), np.tanh(((gt_mid - st_mid) / st_mid) * 50.0), 0.0).astype(np.float32)
        both_ok = gt_fresh & ob_fresh & (gt_mid > 1e-4) & (ob_mid > 1e-4)
        lead_mid = np.where(both_ok, 0.5 * (gt_mid + ob_mid), np.where(gt_fresh & (gt_mid > 1e-4), gt_mid, np.where(ob_fresh & (ob_mid > 1e-4), ob_mid, st_mid)))
        cross_cons = np.tanh(((lead_mid - st_mid) / st_mid) * 35.0).astype(np.float32)

        st_imb_2p = np.clip((cb2 - ca2) / (cb2 + ca2 + 500.0), -1.0, 1.0).astype(np.float32)
        st_imb_5p = np.clip((cb5 - ca5) / (cb5 + ca5 + 1000.0), -1.0, 1.0).astype(np.float32)
        has_ww = (ww_px > 1e-4) & (ww_px < st_b1) & (ww_qt > 1.0)
        ww_dist = np.where(has_ww, np.clip((st_b1 - ww_px) / (st_b1 * 0.06), 0.0, 1.5), 1.0).astype(np.float32)
        ww_str = np.where(has_ww, np.tanh(np.log1p(ww_px * ww_qt / 1000.0) / 4.0), 0.0).astype(np.float32)

        obs112 = np.empty((n, 112), dtype=np.float32)
        obs112[:, :96] = obs96
        obs112[:, 58] = gt_basis_rep
        obs112[:, 96] = gt_ret_15s
        obs112[:, 97] = gt_imb5
        obs112[:, 98] = gt_cvd_60s
        obs112[:, 99] = gt_cvd_300s
        obs112[:, 100] = ob_ret_15s
        obs112[:, 101] = ob_cvd_60s
        obs112[:, 102] = ob_cvd_300s
        obs112[:, 103] = cross_cons
        obs112[:, 104] = btc_ret_60s
        obs112[:, 105] = btc_ret_900s
        obs112[:, 106] = near_ret_60s
        obs112[:, 107] = near_ret_900s
        obs112[:, 108] = st_imb_2p
        obs112[:, 109] = st_imb_5p
        obs112[:, 110] = ww_dist
        obs112[:, 111] = ww_str
        obs112 = np.nan_to_num(obs112, nan=0.0, posinf=3.0, neginf=-3.0)

        sub = obs112[3016277::50]
        stds = np.std(sub, axis=0)
        assert np.all(stds > 1e-4), f"Dead channels in {tag} 112D: {np.where(stds <= 1e-4)[0]}"
        gt_nz = (np.abs(sub[:, 58]) > 1e-6).mean() * 100.0

        tmp_p = out_112.with_name(out_112.stem + "_tmp.npz")
        np.savez(tmp_p, market_obs_112d=obs112)
        tmp_p.replace(out_112)
        print(f"✅ [{tag}] Built {out_112.name} ({obs112.shape}, {out_112.stat().st_size/1e6:.1f} MB) | ch[58] Gate coverage={gt_nz:.1f}% | min_std={stds[96:].min():.4f}")

    print(f"🏁 Signal Stack v8 (112D) + Deep Profile completed in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
