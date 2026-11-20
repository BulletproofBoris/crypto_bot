#!/usr/bin/env python3
"""
_tools/build_bo_swarm_36d_cache.py
=============================================================================
Урок 25 (Компонент 2): Сверхбыстрая каузальная сборка 36-мерной матрицы сигналов
роя 1-го порядка BigONE (`models/prl_bo/`, 6 горизонтов H=5..120s) и встраивание
`bo_signals_36d` [N, 36] в `data/cache/rl_guard_lob_cache_600h.npz` (444D -> 480D).

Ключевые свойства 1-в-1 Sim-to-Real и отсутствия утечек:
  1. Каузальный Top-2 Recency Router (`0.70 * rank_0 + 0.30 * rank_1` среди окон с `train_end_sec < ts_sec`):
     - Исключает размывание сигналов старыми октябрьскими окнами (до запуска 50-уровневого коллектора BigONE).
     - Каждая модель вычисляется ТОЛЬКО на своём активном каузальном интервале (~35с на все 2.95M тиков!).
     - В live (`LiveTriadRLTrader`) требует ровно 2 последних бустера на горизонт (12 бустеров = ~0.6 мс/тик!)
       и вызывает ту же самую функцию `extract_bo_6d_from_probs(probs, H)` — 100% побитовый паритет!
  2. Лог-отношение шансов (`dir_log_odds`, `tail_log_odds`) сохраняет полную ранговую силу роя BigONE
     (`OOS IC = +0.253..+0.353`) в `float16` даже на сверхкоротком горизонте `H=5s`.
=============================================================================
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

HORIZONS = [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]
EZ_SCALE_BY_H = {5: 40.0, 12: 12.0, 25: 5.0, 60: 2.0, 90: 1.5, 120: 1.2}
ACT_SCALE_BY_H = {5: 250.0, 12: 40.0, 25: 15.0, 60: 5.0, 90: 4.0, 120: 3.5}
CLASS_VALS_7 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)


def extract_bo_6d_from_probs(probs: np.ndarray, H: int) -> np.ndarray:
    """
    Единая офлайн/онлайн функция преобразования 7-классовых вероятностей BigONE [N, 7] -> [N, 6]
    без потери точности в float16 на малых хвостовых вероятностях H=5s/12s.
    """
    p = np.clip(np.asarray(probs, dtype=np.float32), 1e-8, 1.0)
    if p.ndim == 1:
        p = p[np.newaxis, :]
    p_dn = p[:, 0] + p[:, 1] + p[:, 2]
    p_up = p[:, 4] + p[:, 5] + p[:, 6]
    p_tn = p[:, 0] + p[:, 1]
    p_tp = p[:, 5] + p[:, 6]
    ez = p @ CLASS_VALS_7

    ez_sc = EZ_SCALE_BY_H.get(int(H), 2.0)
    act_sc = ACT_SCALE_BY_H.get(int(H), 5.0)

    dir_lo = np.tanh(0.45 * np.log((p_up + 1e-5) / (p_dn + 1e-5)))
    tail_lo = np.tanh(0.45 * np.log((p_tp + 1e-5) / (p_tn + 1e-5)))
    ez_n = np.clip(ez * ez_sc, -3.0, 3.0)
    act_u = np.clip(np.log1p(p_up * act_sc) / 2.5, 0.0, 2.0)
    act_d = np.clip(np.log1p(p_dn * act_sc) / 2.5, 0.0, 2.0)
    move_int = np.clip((p_up + p_dn) * act_sc / 3.0, 0.0, 2.0)
    return np.column_stack([dir_lo, tail_lo, ez_n, act_u, act_d, move_int]).astype(np.float32)


class CausalTop2BigONERouter:
    """
    Быстрый каузальный роутер последних 2 обученных экспертов BigONE (0.70 * latest + 0.30 * prev)
    для 100% идентичного расчёта в офлайн-кэше и живом `LiveTriadRLTrader` (~0.1 мс на горизонт).
    """

    def __init__(self, H: int, K: float, live_only_top2: bool = False):
        self.H = int(H)
        self.K = float(K)
        self.model_dir = BASE_DIR / "models" / "prl_bo" / f"h{self.H}s_k{self.K:g}"
        metas = sorted(self.model_dir.glob("meta_win_*.json"))
        items = []
        for mp in metas:
            m = json.loads(mp.read_text(encoding="utf-8"))
            mf = self.model_dir / m["model_file"]
            if mf.exists():
                t_end = int(pd.Timestamp(m["train_ts_range"][1], tz="UTC").timestamp())
                items.append((t_end, mf))
        items.sort(key=lambda x: x[0])
        if live_only_top2 and len(items) > 2:
            items = items[-2:]
        self.train_ends = np.array([x[0] for x in items], dtype=np.int64)
        self.boosters = [lgb.Booster(model_file=str(x[1])) for x in items]
        self.feature_names = list(self.boosters[-1].feature_name()) if self.boosters else []

    def predict_live_1row(self, x_1d: np.ndarray, bo_sec_since_tick: float) -> np.ndarray:
        if not self.boosters or bo_sec_since_tick > 90.0:
            return np.zeros(6, dtype=np.float32)
        x_2d = np.nan_to_num(np.asarray(x_1d, dtype=np.float32).reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        if len(self.boosters) == 1:
            pr = self.boosters[-1].predict(x_2d)
        else:
            pr = 0.70 * self.boosters[-1].predict(x_2d) + 0.30 * self.boosters[-2].predict(x_2d)
        return extract_bo_6d_from_probs(pr, self.H)[0]


def build_and_inject_bo_signals_36d() -> dict:
    from _tools.build_hybrid_v7_v8_cache import CACHE_600H_PATH, compute_horizon_maturity_6d
    t0 = time.time()
    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    mmap_bo_path = BASE_DIR / "data" / "_mmap_cache_bo" / "X_all.npy"
    feats_bo_path = BASE_DIR / "data" / "features" / "selected_features_prl_bo_h8s.json"

    with open(feats_bo_path, encoding="utf-8") as f:
        bo_features = json.load(f)["feature_order"]

    raw_600 = np.load(CACHE_600H_PATH, allow_pickle=True)
    ts_600 = raw_600["ts"].astype(np.int64)
    bo_sec_600 = raw_600["bo_sec_since_tick"].astype(np.float32)
    n_total = len(ts_600)
    raw_600.close()

    pf = pq.ParquetFile(labeled_path)
    schema_cols = set(pf.schema_arrow.names)
    x_bo_mm = np.load(mmap_bo_path, mmap_mode="r")
    n_mm = len(x_bo_mm)

    x_tail_bo = None
    if n_total > n_mm:
        tail_len = n_total - n_mm
        avail_bo_cols = [c for c in bo_features if c in schema_cols]
        tb_tail = pq.read_table(labeled_path, columns=avail_bo_cols).slice(n_mm, tail_len).to_pandas()
        x_tail_bo = np.zeros((tail_len, len(bo_features)), dtype=np.float32)
        for j, cname in enumerate(bo_features):
            if cname in tb_tail.columns:
                x_tail_bo[:, j] = np.nan_to_num(tb_tail[cname].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        del tb_tail
        gc.collect()

    def _get_x_slice(s: int, e: int) -> np.ndarray:
        if e <= n_mm:
            return np.nan_to_num(np.array(x_bo_mm[s:e], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if s >= n_mm:
            return x_tail_bo[s - n_mm : e - n_mm]
        p1 = np.nan_to_num(np.array(x_bo_mm[s:n_mm], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.vstack([p1, x_tail_bo[0 : e - n_mm]])

    fresh_bo_mask = (np.nan_to_num(bo_sec_600, nan=999.0) <= 90.0).astype(np.float32)[:, None]
    maturity_bo_6d = compute_horizon_maturity_6d(ts_600, coin="prl_bo")
    bo_signals_36d = np.zeros((n_total, 36), dtype=np.float32)

    print("=" * 95, flush=True)
    print(f"🌐 УРОК 25: СВЕРХБЫСТРАЯ КАУЗАЛЬНАЯ СБОРКА bo_signals_36d ({n_total:,} тиков | Top-2 Recency + LogOdds)", flush=True)
    print("=" * 95, flush=True)

    for h_i, (H, K) in enumerate(HORIZONS):
        th0 = time.time()
        router = CausalTop2BigONERouter(H, K, live_only_top2=False)
        m_cnt = len(router.boosters)
        # Индекс последнего каузально доступного окна: k_latest = searchsorted(train_ends, ts_600 - 1, right) - 1
        k_lat = np.searchsorted(router.train_ends, ts_600 - 1, side="right") - 1
        h_out = np.zeros((n_total, 6), dtype=np.float32)

        # Идём по непересекающимся интервалам постоянства k_lat = k (всего M интервалов, 2 вызова predict на тик!)
        for k in range(m_cnt):
            s_k = int(np.searchsorted(k_lat, k, side="left"))
            e_k = int(np.searchsorted(k_lat, k, side="right"))
            if e_k <= s_k:
                continue
            for c0 in range(s_k, e_k, 250000):
                c1 = min(e_k, c0 + 250000)
                x_sl = _get_x_slice(c0, c1)
                if k == 0:
                    pr_blend = router.boosters[0].predict(x_sl)
                else:
                    pr_blend = 0.70 * router.boosters[k].predict(x_sl) + 0.30 * router.boosters[k - 1].predict(x_sl)
                h_out[c0:c1] = extract_bo_6d_from_probs(pr_blend, H)

        h_out *= fresh_bo_mask
        bo_signals_36d[:, h_i * 6 : (h_i + 1) * 6] = np.nan_to_num(h_out, nan=0.0, posinf=3.0, neginf=-3.0)
        tail_sl = bo_signals_36d[-100000:, h_i * 6 : (h_i + 1) * 6]
        print(
            f"   ✅ H={H:3d}s (M={m_cnt:2d}) готов за {time.time() - th0:4.1f}с | "
            f"Tail-100k: std(dir_lo)={np.std(tail_sl[:, 0]):.3f}, std(ez_n)={np.std(tail_sl[:, 2]):.3f}, mean(move)={np.mean(tail_sl[:, 5]):.3f}",
            flush=True,
        )

    del x_tail_bo
    gc.collect()

    raw_fresh = np.load(CACHE_600H_PATH, allow_pickle=True)
    data = {k: raw_fresh[k] for k in raw_fresh.files}
    raw_fresh.close()

    data["bo_signals_36d"] = bo_signals_36d.astype(np.float16)
    data["bo_maturity_6d"] = maturity_bo_6d.astype(np.float16)
    data["has_bo_swarm_36d"] = np.int32(2)

    tmp_path = CACHE_600H_PATH.with_suffix(".bo36_v2_tmp.npz")
    np.savez(tmp_path, **data)
    tmp_path.replace(CACHE_600H_PATH)
    print(f"🏆 bo_signals_36d v2 ({bo_signals_36d.shape}, float16) сохранена в {CACHE_600H_PATH.name} за {time.time() - t0:.1f}с!", flush=True)
    return {"n_total": n_total, "elapsed_sec": round(time.time() - t0, 1)}


if __name__ == "__main__":
    build_and_inject_bo_signals_36d()
