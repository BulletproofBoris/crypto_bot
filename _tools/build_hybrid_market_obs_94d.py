#!/usr/bin/env python3
"""
_tools/build_hybrid_market_obs_94d.py
=============================================================================
Signal Stack v6 (94D Macro-Structure & Tape CVD Stack, 0% Leakage, 100% Sim/Live Parity):
  - [0..87]  Existing 88 causal channels (Signal Stack v5: LOB + BigONE + O1 + O2 + Wave + O2.5)
  - [88..93] 6 causal medium-term (5m / 15m / 30m) trend structure & tape delta channels:
      [88] ret_30m         = tanh((mid / mid_30m - 1.0) * 12.0)
      [89] twap_dev_15m    = tanh(((mid - mean_mid_15m) / mid) * 25.0)
      [90] lower_highs_30m = tanh(((max_prev_15m - max_cur_15m) / mid) * 25.0)  (>0 on stair-step bear!)
      [91] cvd_norm_5m     = (buy_5m - sell_5m) / (buy_5m + sell_5m + 50.0)
      [92] cvd_norm_15m    = (buy_15m - sell_15m) / (buy_15m + sell_15m + 150.0)
      [93] cvd_norm_30m    = (buy_30m - sell_30m) / (buy_30m + sell_30m + 300.0)

Builds BOTH worlds automatically:
  1) `data/cache/rl_hybrid_market_obs_94d.npz` (Decontaminated Interactive World `DEC`)
  2) `data/cache/rl_hybrid_market_obs_94d_raw_with_self.npz` (Untouched Raw World `RAW`)
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d

BASE_DIR = Path(__file__).resolve().parent.parent


def causal_rolling_max(arr: np.ndarray, win: int) -> np.ndarray:
    """Strictly causal max over [max(0, t - win + 1) .. t] inclusive (0 lookahead)."""
    assert win % 2 == 0, "win must be even for exact causal alignment"
    pad = np.concatenate([np.full(win - 1, arr[0], dtype=arr.dtype), arr])
    mx_full = maximum_filter1d(pad, size=win, mode="nearest")
    return mx_full[np.arange(len(arr), dtype=np.int64) + (win // 2)]


def causal_rolling_sum(arr: np.ndarray, win: int) -> np.ndarray:
    """Strictly causal sum over [max(0, t - win + 1) .. t] in float64 to avoid precision drift."""
    cs = np.empty(len(arr) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(arr, dtype=np.float64, out=cs[1:])
    idx_hi = np.arange(1, len(arr) + 1, dtype=np.int64)
    idx_lo = np.maximum(0, idx_hi - win)
    return cs[idx_hi] - cs[idx_lo]


def causal_rolling_mean_padded(arr: np.ndarray, win: int) -> np.ndarray:
    """Strictly causal mean over [t - win + 1 .. t] with left-padding by arr[0]."""
    pad = np.concatenate([np.full(win - 1, arr[0], dtype=np.float64), arr.astype(np.float64)])
    cs = np.empty(len(pad) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(pad, dtype=np.float64, out=cs[1:])
    idx = np.arange(len(arr), dtype=np.int64)
    return ((cs[idx + win] - cs[idx]) / float(win)).astype(np.float32)


def build_one_world(tag: str, in_88d: Path, in_lob: Path, out_94d: Path) -> None:
    t0 = time.time()
    obs88 = np.load(in_88d, mmap_mode="r")["market_obs_88d"]
    lob = np.load(in_lob, mmap_mode="r")
    n = len(obs88)
    assert len(lob["bid_p"]) == n, f"Length mismatch: obs88={n} vs lob={len(lob['bid_p'])}"

    b1 = np.asarray(lob["bid_p"][:, 0], dtype=np.float32)
    a1 = np.asarray(lob["ask_p"][:, 0], dtype=np.float32)
    mid = np.maximum(0.5 * (b1 + a1), 1e-4)
    bv = np.maximum(0.0, np.asarray(lob["buy_vol"], dtype=np.float32))
    sv = np.maximum(0.0, np.asarray(lob["sell_vol"], dtype=np.float32))

    idx_ar = np.arange(n, dtype=np.int64)
    mid_30m = mid[np.maximum(0, idx_ar - 1800)]
    ret_30m = np.tanh((mid / mid_30m - 1.0) * 12.0).astype(np.float32)

    mean_15m = causal_rolling_mean_padded(mid, 900)
    twap_dev_15m = np.tanh(((mid - mean_15m) / mid) * 25.0).astype(np.float32)

    mx_cur_15m = causal_rolling_max(mid, 900)
    mx_prev_15m = mx_cur_15m[np.maximum(0, idx_ar - 900)]
    # Self-check causality at tail
    for tt in (3100000, n - 1):
        ref_cur = float(mid[tt - 899 : tt + 1].max())
        ref_prv = float(mid[tt - 1799 : tt - 899].max())
        assert abs(float(mx_cur_15m[tt]) - ref_cur) < 1e-6 and abs(float(mx_prev_15m[tt]) - ref_prv) < 1e-6
    lower_highs_30m = np.tanh(((mx_prev_15m - mx_cur_15m) / mid) * 25.0).astype(np.float32)

    b_5m, s_5m = causal_rolling_sum(bv, 300), causal_rolling_sum(sv, 300)
    b_15m, s_15m = causal_rolling_sum(bv, 900), causal_rolling_sum(sv, 900)
    b_30m, s_30m = causal_rolling_sum(bv, 1800), causal_rolling_sum(sv, 1800)

    cvd_5m = np.clip((b_5m - s_5m) / (b_5m + s_5m + 50.0), -1.0, 1.0).astype(np.float32)
    cvd_15m = np.clip((b_15m - s_15m) / (b_15m + s_15m + 150.0), -1.0, 1.0).astype(np.float32)
    cvd_30m = np.clip((b_30m - s_30m) / (b_30m + s_30m + 300.0), -1.0, 1.0).astype(np.float32)

    obs94 = np.empty((n, 94), dtype=np.float32)
    obs94[:, :88] = obs88
    obs94[:, 88] = ret_30m
    obs94[:, 89] = twap_dev_15m
    obs94[:, 90] = lower_highs_30m
    obs94[:, 91] = cvd_5m
    obs94[:, 92] = cvd_15m
    obs94[:, 93] = cvd_30m
    obs94 = np.nan_to_num(obs94, nan=0.0, posinf=2.5, neginf=-2.5)

    sub = obs94[3016277::50]
    stds = np.std(sub, axis=0)
    assert np.all(stds > 1e-5), f"[{tag}] Dead channels in 94D: {np.where(stds <= 1e-5)[0]}"

    tmp_p = out_94d.with_name(out_94d.stem + "_tmp.npz")
    np.savez(tmp_p, market_obs_94d=obs94)
    tmp_p.replace(out_94d)
    print(f"✅ [{tag}] Built 94D Signal Stack v6 ({obs94.shape}, {out_94d.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s!", flush=True)
    for c_i in range(88, 94):
        print(f"   [{tag}] ch[{c_i:02d}]: mean={sub[:, c_i].mean():+.4f} | std={stds[c_i]:.4f}", flush=True)


def main():
    print("=" * 88)
    print("🌊 BUILDING SIGNAL STACK V6: CAUSAL 94D MACRO-STRUCTURE & TAPE CVD (DEC + RAW)")
    print("=" * 88, flush=True)
    build_one_world(
        "DEC",
        BASE_DIR / "data/cache/rl_hybrid_market_obs_88d.npz",
        BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz",
        BASE_DIR / "data/cache/rl_hybrid_market_obs_94d.npz",
    )
    build_one_world(
        "RAW",
        BASE_DIR / "data/cache/rl_hybrid_market_obs_88d_raw_with_self.npz",
        BASE_DIR / "data/cache/rl_guard_lob_cache_600h_raw_with_self.npz",
        BASE_DIR / "data/cache/rl_hybrid_market_obs_94d_raw_with_self.npz",
    )


if __name__ == "__main__":
    main()
