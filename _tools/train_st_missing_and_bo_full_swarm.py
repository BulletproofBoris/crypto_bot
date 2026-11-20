#!/usr/bin/env python3
"""
_tools/train_st_missing_and_bo_full_swarm.py
=============================================================================
Единый оркестратор обучения моделей 1-го порядка (после завершения обучения RL-агента):
  Этап 1: Дообучение всех недостающих и сдвинувшихся окон SafeTrade (`coin="prl"`,
          `data/_mmap_cache` -> `models/prl/`) по всем 6 парам `(H, K)`:
          [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)].
  Этап 2: Полное обучение всех моделей 1-го порядка (LightGBM + 6 нейросетевых архитектур:
          mlp_bottleneck, wavenet_tcn, cross_mixer, causal_cnn, attention, conv1d+gru)
          на данных BigONE (`coin="prl_bo"`, `data/_mmap_cache_bo` -> `models/prl_bo/`)
          для тех же самых горизонтов H и коэффициентов K.
  Этап 3: Сводная таблица OOS-метрик (IC, OVR-AUC, Top-2% импульс и число активных экспертов в альянсах).
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.v7_retrain_pipeline import HORIZON_PAIRS, NEURAL_ARCHS, step2_train_rolling_windows_6h


def print_swarm_summary(coin: str, cache_subdir: str):
    coin_l = coin.lower()
    cache_dir = BASE_DIR / "data" / cache_subdir
    print("\n" + "=" * 112, flush=True)
    print(f"📊 ИТОГОВАЯ OOS-СВОДКА МОДЕЛЕЙ 1-ГО ПОРЯДКА: {coin_l.upper()} (models/{coin_l}/)", flush=True)
    print("=" * 112, flush=True)
    print(f"{'Горизонт':<12} | {'Окон':>4} | {'LGB Mean IC':>11} | {'LGB Mean AUC':>12} | {'NN Best IC':>10} | {'NN Mean AUC':>11} | {'Экспертов в альянсе':>20}", flush=True)
    print("-" * 112, flush=True)

    for h, k in HORIZON_PAIRS:
        n_w = int(np.load(cache_dir / f"h{h}_n_windows.npy"))
        lgb_dir = BASE_DIR / "models" / coin_l / f"h{h}s_k{k:g}"
        lgb_ics, lgb_aucs = [], []
        for w_i in range(n_w):
            mp = lgb_dir / f"meta_win_{w_i:02d}.json"
            if mp.exists():
                mo = json.loads(mp.read_text()).get("metrics_oos", {})
                lgb_ics.append(float(mo.get("ic_dt_ticks", 0.0)))
                lgb_aucs.append(float(mo.get("roc_auc_ovr", 0.5)))

        nn_ics, nn_aucs = [], []
        for arch in NEURAL_ARCHS:
            a_dir = BASE_DIR / "models" / coin_l / f"{arch.replace('+', '_')}_h{h}s"
            for w_i in range(n_w):
                mp = a_dir / f"meta_win_{w_i:02d}.json"
                if mp.exists():
                    mo = json.loads(mp.read_text()).get("metrics_oos", {})
                    nn_ics.append(float(mo.get("ic_dt_ticks", 0.0)))
                    nn_aucs.append(float(mo.get("roc_auc_ovr", 0.5)))

        al_p = BASE_DIR / "models" / coin_l / f"ensemble_alliance_live_h{h}s.json"
        n_sel = len(json.loads(al_p.read_text()).get("selected_models", [])) if al_p.exists() else 0
        print(
            f"H={h:3d}s K={k:<3g} | {n_w:4d} | {np.mean(lgb_ics) if lgb_ics else 0.0:+11.4f} | "
            f"{np.mean(lgb_aucs) if lgb_aucs else 0.5:12.4f} | {np.max(nn_ics) if nn_ics else 0.0:+10.4f} | "
            f"{np.mean(nn_aucs) if nn_aucs else 0.5:11.4f} | {n_sel:20d}",
            flush=True,
        )
    print("=" * 112, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--st-runs", type=int, default=15)
    ap.add_argument("--bo-runs", type=int, default=8, help="Число мультистартов на окно для полного пула BigONE (120+ окон x 6 NN)")
    ap.add_argument("--only", choices=["all", "st", "bo"], default="all")
    ap.add_argument("--phase", choices=["all", "lgb", "nn"], default="all")
    ap.add_argument("--lgb-workers", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.only in ("all", "st"):
        step2_train_rolling_windows_6h(
            coin="prl", epochs=args.epochs, runs=args.st_runs, cache_subdir="_mmap_cache", phase=args.phase, lgb_workers=args.lgb_workers
        )
        print_swarm_summary("prl", "_mmap_cache")

    if args.only in ("all", "bo"):
        step2_train_rolling_windows_6h(
            coin="prl_bo", epochs=args.epochs, runs=args.bo_runs, cache_subdir="_mmap_cache_bo", phase=args.phase, lgb_workers=args.lgb_workers
        )
        print_swarm_summary("prl_bo", "_mmap_cache_bo")

    print(f"\n🎉 Полный цикл обучения моделей 1-го порядка завершен за {(time.time() - t0) / 60.0:.1f} мин!", flush=True)


if __name__ == "__main__":
    main()
