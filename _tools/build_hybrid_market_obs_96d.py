#!/usr/bin/env python3
"""
_tools/build_hybrid_market_obs_96d.py
=============================================================================
Signal Stack v7 (96D Macro-Structure, Tape CVD & 2h Blowoff Stack, 0% Leakage):
Builds `96D` directly from `88D` (`train_order25_impact_and_regret.py`) + LOB
in a single pass (without intermediate 94D 3GB disk caches):
  - [0..87]  88 causal channels (LOB + BigONE + O1 + O2 + Wave + O2.5)
  - [88]     ret_30m         = tanh((mid / mid_30m - 1.0) * 12.0)
  - [89]     twap_dev_15m    = tanh(((mid - mean_mid_15m) / mid) * 25.0)
  - [90]     lower_highs_30m = tanh(((max_prev_15m - max_cur_15m) / mid) * 25.0)
  - [91]     cvd_norm_5m     = (buy_5m - sell_5m) / (buy_5m + sell_5m + 50.0)
  - [92]     cvd_norm_15m    = (buy_15m - sell_15m) / (buy_15m + sell_15m + 150.0)
  - [93]     cvd_norm_30m    = (buy_30m - sell_30m) / (buy_30m + sell_30m + 300.0)
  - [94]     ret_120m        = tanh((mid / mid_120m - 1.0) * 8.0)
  - [95]     cvd_norm_120m   = (buy_120m - sell_120m) / (buy_120m + sell_120m + 1000.0)

Builds BOTH worlds automatically:
  1) `data/cache/rl_hybrid_market_obs_96d.npz` (`DEC`)
  2) `data/cache/rl_hybrid_market_obs_96d_raw_with_self.npz` (`RAW`)
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
    """Strictly causal sum over [max(0, t - win + 1) .. t] in float64."""
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


def build_one_world(tag: str, in_88d: Path, in_lob: Path, out_96d: Path) -> None:
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
    ret_30m = np.tanh((mid / mid[np.maximum(0, idx_ar - 1800)] - 1.0) * 12.0).astype(np.float32)
    mean_15m = causal_rolling_mean_padded(mid, 900)
    twap_dev_15m = np.tanh(((mid - mean_15m) / mid) * 25.0).astype(np.float32)
    mx_cur_15m = causal_rolling_max(mid, 900)
    mx_prev_15m = mx_cur_15m[np.maximum(0, idx_ar - 900)]
    lower_highs_30m = np.tanh(((mx_prev_15m - mx_cur_15m) / mid) * 25.0).astype(np.float32)

    b_5m, s_5m = causal_rolling_sum(bv, 300), causal_rolling_sum(sv, 300)
    b_15m, s_15m = causal_rolling_sum(bv, 900), causal_rolling_sum(sv, 900)
    b_30m, s_30m = causal_rolling_sum(bv, 1800), causal_rolling_sum(sv, 1800)
    b_120m, s_120m = causal_rolling_sum(bv, 7200), causal_rolling_sum(sv, 7200)

    cvd_5m = np.clip((b_5m - s_5m) / (b_5m + s_5m + 50.0), -1.0, 1.0).astype(np.float32)
    cvd_15m = np.clip((b_15m - s_15m) / (b_15m + s_15m + 150.0), -1.0, 1.0).astype(np.float32)
    cvd_30m = np.clip((b_30m - s_30m) / (b_30m + s_30m + 300.0), -1.0, 1.0).astype(np.float32)
    ret_120m = np.tanh((mid / mid[np.maximum(0, idx_ar - 7200)] - 1.0) * 8.0).astype(np.float32)
    cvd_120m = np.clip((b_120m - s_120m) / (b_120m + s_120m + 1000.0), -1.0, 1.0).astype(np.float32)

    obs96 = np.empty((n, 96), dtype=np.float32)
    obs96[:, :88] = obs88
    obs96[:, 88] = ret_30m
    obs96[:, 89] = twap_dev_15m
    obs96[:, 90] = lower_highs_30m
    obs96[:, 91] = cvd_5m
    obs96[:, 92] = cvd_15m
    obs96[:, 93] = cvd_30m
    obs96[:, 94] = ret_120m
    obs96[:, 95] = cvd_120m
    obs96 = np.nan_to_num(obs96, nan=0.0, posinf=3.0, neginf=-3.0)

    stds = np.std(obs96[3016277::50], axis=0)
    assert np.all(stds > 1e-5), f"Dead channels in {tag} 96D: {np.where(stds <= 1e-5)[0]}"

    tmp_p = out_96d.with_name(out_96d.stem + "_tmp.npz")
    np.savez(tmp_p, market_obs_96d=obs96)
    tmp_p.replace(out_96d)
    print(f"✅ [{tag}] Built {out_96d.name} ({obs96.shape}, {out_96d.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s!", flush=True)


def main() -> None:
    cdir = BASE_DIR / "data" / "cache"
    print("=" * 86)
    print("🌊 BUILDING SIGNAL STACK V7: CAUSAL 96D MARKET MATRIX (DEC + RAW, DIRECT FROM 88D)")
    print("=" * 86, flush=True)
    build_one_world("DEC", cdir / "rl_hybrid_market_obs_88d.npz", cdir / "rl_guard_lob_cache_600h.npz", cdir / "rl_hybrid_market_obs_96d.npz")
    raw_88 = cdir / "rl_hybrid_market_obs_88d_raw_with_self.npz"
    raw_lob = cdir / "rl_guard_lob_cache_600h_raw_with_self.npz"
    if raw_88.exists() and raw_lob.exists():
        build_one_world("RAW", raw_88, raw_lob, cdir / "rl_hybrid_market_obs_96d_raw_with_self.npz")


if __name__ == "__main__":
    main()
