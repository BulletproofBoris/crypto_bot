#!/usr/bin/env python3
"""
_tools/build_hybrid_market_obs_96d.py
=============================================================================
Signal Stack v7 (96D Macro-2h Exhaustion/Capitulation & Tape CVD Stack, 0% Leakage, 100% Sim/Live Parity):
  - [0..93]  Existing 94 causal channels (Signal Stack v6)
  - [94..95] 2 causal 2-hour (7200s) horizon channels for Regime H (Blowoff / Capitulation):
      [94] ret_120m      = tanh((mid / mid_120m - 1.0) * 8.0)
      [95] cvd_norm_120m = (buy_120m - sell_120m) / (buy_120m + sell_120m + 1000.0)

Builds BOTH worlds automatically:
  1) `data/cache/rl_hybrid_market_obs_96d.npz` (Decontaminated Interactive World `DEC`)
  2) `data/cache/rl_hybrid_market_obs_96d_raw_with_self.npz` (Untouched Raw World `RAW`)
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def causal_rolling_sum(arr: np.ndarray, win: int) -> np.ndarray:
    """Strictly causal sum over [max(0, t - win + 1) .. t] in float64."""
    cs = np.empty(len(arr) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(arr, dtype=np.float64, out=cs[1:])
    idx_hi = np.arange(1, len(arr) + 1, dtype=np.int64)
    idx_lo = np.maximum(0, idx_hi - win)
    return cs[idx_hi] - cs[idx_lo]


def build_one_world(tag: str, in_94d: Path, in_lob: Path, out_96d: Path) -> None:
    t0 = time.time()
    obs94 = np.load(in_94d, mmap_mode="r")["market_obs_94d"]
    lob = np.load(in_lob, mmap_mode="r")
    n = len(obs94)
    assert len(lob["bid_p"]) == n, f"Length mismatch: obs94={n} vs lob={len(lob['bid_p'])}"

    b1 = np.asarray(lob["bid_p"][:, 0], dtype=np.float32)
    a1 = np.asarray(lob["ask_p"][:, 0], dtype=np.float32)
    mid = np.maximum(0.5 * (b1 + a1), 1e-4)
    bv = np.maximum(0.0, np.asarray(lob["buy_vol"], dtype=np.float32))
    sv = np.maximum(0.0, np.asarray(lob["sell_vol"], dtype=np.float32))

    idx_ar = np.arange(n, dtype=np.int64)
    mid_120m = mid[np.maximum(0, idx_ar - 7200)]
    ret_120m = np.tanh((mid / mid_120m - 1.0) * 8.0).astype(np.float32)

    b_120m = causal_rolling_sum(bv, 7200)
    s_120m = causal_rolling_sum(sv, 7200)
    cvd_120m = np.clip((b_120m - s_120m) / (b_120m + s_120m + 1000.0), -1.0, 1.0).astype(np.float32)

    # Causality check at tail
    for tt in (3100000, n - 1):
        ref_r = float(np.tanh((mid[tt] / mid[tt - 7200] - 1.0) * 8.0))
        assert abs(float(ret_120m[tt]) - ref_r) < 1e-6

    obs96 = np.empty((n, 96), dtype=np.float32)
    obs96[:, :94] = obs94
    obs96[:, 94] = ret_120m
    obs96[:, 95] = cvd_120m

    sub = obs96[3016277:]
    assert not np.isnan(sub).any() and not np.isinf(sub).any(), f"NaN/Inf in {tag}!"
    stds = sub[::50].std(axis=0)
    dead = np.where(stds < 1e-6)[0]
    assert len(dead) == 0, f"Dead channels in {tag}: {dead}"

    tmp_p = out_96d.with_suffix(".tmp.npz")
    np.savez(tmp_p, market_obs_96d=obs96)
    tmp_p.replace(out_96d)
    for c_i in range(94, 96):
        print(f"   [{tag}] ch[{c_i:02d}]: mean={sub[:, c_i].mean():+.4f} | std={stds[c_i]:.4f}", flush=True)
    print(f"✅ [{tag}] Built {out_96d.name}: shape={obs96.shape} in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    print("=" * 88)
    print("🏗️ BUILDING SIGNAL STACK V7: 96D MACRO-2H EXHAUSTION & CVD STACK (DEC + RAW)")
    print("=" * 88)
    build_one_world(
        "DEC",
        BASE_DIR / "data/cache/rl_hybrid_market_obs_94d.npz",
        BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz",
        BASE_DIR / "data/cache/rl_hybrid_market_obs_96d.npz",
    )
    build_one_world(
        "RAW",
        BASE_DIR / "data/cache/rl_hybrid_market_obs_94d_raw_with_self.npz",
        BASE_DIR / "data/cache/rl_guard_lob_cache_600h_raw_with_self.npz",
        BASE_DIR / "data/cache/rl_hybrid_market_obs_96d_raw_with_self.npz",
    )


if __name__ == "__main__":
    main()
