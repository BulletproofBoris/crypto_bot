#!/usr/bin/env python3
"""
_tools/build_hybrid_market_obs_84d.py
=============================================================================
Signal Stack v4 (84D Full Order-1 & Order-2 Pool, 0% Leakage, 100% Sim/Live Parity):
  - [0..71]  Базовые 72 каузальных канала Signal Stack v3 (100% совместимость весов L70!)
  - [72..75] 4 новых каузальных горизонта Order-1 Expected Z (320s, 600s, 900s, 1800s) [-3..+3]
  - [76..78] 3 канала уверенности и асимметрии средних волн Order-1:
             conf_600s [0..1], conf_1800s [0..1], bias_900s [-1..+1]
  - [79..83] 5 новых каузальных физических моделей Order-2 (риски до 15m и очередь до 4m):
             guard_bleed_180s, guard_crash_600s, guard_crash_900s,
             exec_fill_120s, carousel_success_240s [0..1]
=============================================================================
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OBS_72D_PATH = BASE_DIR / "data" / "cache" / "rl_hybrid_market_obs_72d.npz"
ALPHA_84D_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_84d.npz"
RISK_16D_PATH = BASE_DIR / "data" / "cache" / "order2_risk_matrix_16d.npz"
OUT_84D_PATH = BASE_DIR / "data" / "cache" / "rl_hybrid_market_obs_84d.npz"


def main():
    t0 = time.time()
    print("=" * 86)
    print("🌊 СБОРКА SIGNAL STACK V4: СТРОГО КАУЗАЛЬНЫЙ 84D СТЕК (12H ORDER-1 + 16D ORDER-2)")
    print("=" * 86, flush=True)

    obs72 = np.load(OBS_72D_PATH, mmap_mode="r")["market_obs_72d"]
    a84 = np.load(ALPHA_84D_PATH, mmap_mode="r")
    r16 = np.load(RISK_16D_PATH, mmap_mode="r")["risk_matrix_16d"]

    n = len(obs72)
    ez12 = a84["expected_z_12h"]
    cf12 = a84["confidence_12h"]
    bi12 = a84["bias_12h"]
    assert len(ez12) == n and len(r16) == n, f"Length mismatch: obs72={n}, ez12={len(ez12)}, r16={len(r16)}"

    obs_84d = np.empty((n, 84), dtype=np.float32)
    obs_84d[:, :72] = obs72
    obs_84d[:, 72:76] = np.clip(ez12[:, 8:12], -3.0, 3.0)
    obs_84d[:, 76] = np.clip(cf12[:, 9], 0.0, 1.0)
    obs_84d[:, 77] = np.clip(cf12[:, 11], 0.0, 1.0)
    obs_84d[:, 78] = np.clip(bi12[:, 10], -1.0, 1.0)
    obs_84d[:, 79:84] = np.clip(r16[:, 11:16], 0.0, 1.0)
    obs_84d = np.nan_to_num(obs_84d, nan=0.0, posinf=3.0, neginf=-3.0)

    sub = obs_84d[3016277:]
    stds = np.std(sub[::50], axis=0)
    assert np.all(stds > 1e-4), f"Dead channels in 84D: {np.where(stds <= 1e-4)[0]}"

    tmp_p = OUT_84D_PATH.with_name("rl_hybrid_market_obs_84d_tmp.npz")
    np.savez(tmp_p, market_obs_84d=obs_84d)
    tmp_p.replace(OUT_84D_PATH)
    sz_mb = OUT_84D_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ СОБРАН SIGNAL STACK V4 84D ({obs_84d.shape}, {sz_mb:.1f} МБ) за {time.time()-t0:.1f}с!")
    ch_names = [
        "ez_320s", "ez_600s", "ez_900s", "ez_1800s",
        "conf_600s", "conf_1800s", "bias_900s",
        "guard_bleed_180s", "guard_crash_600s", "guard_crash_900s",
        "exec_fill_120s", "carousel_success_240s",
    ]
    for off_i, nm in enumerate(ch_names):
        c_i = 72 + off_i
        print(f"   ch[{c_i:02d}] ({nm:<22}): mean={sub[::50, c_i].mean():+.4f} | std={stds[c_i]:.4f}", flush=True)


if __name__ == "__main__":
    main()
