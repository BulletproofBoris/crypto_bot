#!/usr/bin/env python3
"""
_tools/train_order1_wave_wf.py
=============================================================================
Stage 1 (Wave Stack v3): Строго каузальный Walk-Forward тренер 6 дальнобойных
волновых бустеров XGBoost на 2x GPU (win_64..win_82 -> покрытие всей суб-центовой
эпохи [2,937,600 .. 3,736,745] с ЧЕСТНЫМ КАУЗАЛЬНЫМ PURGE = H + 60 сек!):

Модели в каждом окне models/prl/order1_wave_wf/win_XX/:
  1. wave_ret_h300s.ubj   (reg:pseudohubererror): ожидаемый ход волны за 5 мин (%)
  2. wave_ret_h900s.ubj   (reg:pseudohubererror): ожидаемый ход волны за 15 мин (%)
  3. wave_ret_h1800s.ubj  (reg:pseudohubererror): ожидаемый ход волны за 30 мин (%)
  4. wave_ret_h3600s.ubj  (reg:pseudohubererror): ожидаемый ход волны за 60 мин (%)
  5. wave_launch_30m.ubj  (binary:logistic): P(макс Bid1 >= +2.5% раньше стопа -1.2% за 30 мин)
  6. wave_top_15m.ubj     (binary:logistic): P(мин Bid1 <= -2.0% раньше перехая +1.2% за 15 мин)

После обучения собирает непрерывный 100% OOS кэш data/cache/order1_wave_matrix_6d.npz!
=============================================================================
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
LOB_CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
FEATS_PATH = BASE_DIR / "data" / "features" / "selected_features.json"
OUT_MODELS_DIR = BASE_DIR / "models" / "prl" / "order1_wave_wf"
OUT_CACHE_PATH = BASE_DIR / "data" / "cache" / "order1_wave_matrix_6d.npz"

WAVE_MODEL_NAMES = [
    "wave_ret_h300s",
    "wave_ret_h900s",
    "wave_ret_h1800s",
    "wave_ret_h3600s",
    "wave_launch_30m",
    "wave_top_15m",
]


def build_wf_grid(total_rows: int) -> list[tuple[int, int, int]]:
    """Равномерная 12-часовая сетка окон win_64..win_82 (шаг 43,200 сек), покрывающая всю эпоху >= 2,937,600."""
    wins = []
    # win_k имеет val_start = 172800 + k * 43200 (1-в-1 с основной сеткой Order-1/Order-2!)
    k = 64
    while True:
        v_s = 172800 + k * 43200
        if v_s >= total_rows - 3600:
            break
        v_e = min(total_rows, v_s + 43200)
        if total_rows - v_e < 7200:
            v_e = total_rows
        wins.append((k, v_s, v_e))
        if v_e >= total_rows:
            break
        k += 1
    return wins


def precompute_wave_targets(b1: np.ndarray, a1: np.ndarray, s_lo: int, s_hi: int) -> dict[str, np.ndarray]:
    """
    Быстрый векторизованный расчет 6 волновых таргетов на срезе [s_lo .. s_hi]
    с использованием будущего [s_lo .. s_hi + 3605] (вызывается ТОЛЬКО с tr_end <= val_start - H - 60!).
    """
    mid = 0.5 * (b1 + a1)
    n_tot = len(mid)
    idx = np.arange(s_lo, s_hi, dtype=np.int64)
    m0 = np.maximum(mid[idx], 1e-4)
    a0 = np.maximum(a1[idx], 1e-4)
    b0 = np.maximum(b1[idx], 1e-4)

    # Интегральная средняя цена второй половины горизонта через cumsum (гладкий истинный вектор волны без 1с шума!):
    csum = np.concatenate([[0.0], np.cumsum(mid, dtype=np.float64)])
    targets = {}
    for H in (300, 900, 1800, 3600):
        h_half = H // 2
        i1 = np.minimum(idx + h_half, n_tot - 1)
        i2 = np.minimum(idx + H, n_tot)
        cnt = np.maximum(1, i2 - i1)
        fwd_mean = (csum[i2] - csum[i1]) / cnt
        ret_pct = np.clip((fwd_mean / m0 - 1.0) * 100.0, -12.0, 12.0).astype(np.float32)
        targets[f"wave_ret_h{H}s"] = ret_pct

    # Барьерные таргеты First-Passage Time (шаг 5 сек -> 360 точек на 30м, векторно за <0.3с!):
    # 1) wave_launch_30m: P(Bid1 вырастет >= Ask0 * 1.025 РАНЬШЕ, чем Bid1 упадет <= Ask0 * 0.988 за 1800с)
    m_len = len(idx)
    t_hit_up = np.full(m_len, 9999, dtype=np.int32)
    t_hit_dn = np.full(m_len, 9999, dtype=np.int32)
    up_lvl = a0 * 1.025
    dn_lvl = a0 * 0.988
    for step_k, dt_s in enumerate(range(5, 1805, 5)):
        j = np.minimum(idx + dt_s, n_tot - 1)
        bj = b1[j]
        t_hit_up = np.where((t_hit_up == 9999) & (bj >= up_lvl), step_k, t_hit_up)
        t_hit_dn = np.where((t_hit_dn == 9999) & (bj <= dn_lvl), step_k, t_hit_dn)
    targets["wave_launch_30m"] = ((t_hit_up < 9999) & (t_hit_up < t_hit_dn)).astype(np.float32)

    # 2) wave_top_15m: P(Bid1 упадет <= Bid0 * 0.980 РАНЬШЕ, чем Ask1 вырастет >= Bid0 * 1.012 за 900с)
    t_top_dn = np.full(m_len, 9999, dtype=np.int32)
    t_top_up = np.full(m_len, 9999, dtype=np.int32)
    drop_lvl = b0 * 0.980
    new_hi_lvl = b0 * 1.012
    for step_k, dt_s in enumerate(range(5, 905, 5)):
        j = np.minimum(idx + dt_s, n_tot - 1)
        t_top_dn = np.where((t_top_dn == 9999) & (b1[j] <= drop_lvl), step_k, t_top_dn)
        t_top_up = np.where((t_top_up == 9999) & (a1[j] >= new_hi_lvl), step_k, t_top_up)
    targets["wave_top_15m"] = ((t_top_dn < 9999) & (t_top_dn < t_top_up)).astype(np.float32)
    return targets


def read_parquet_slice(pf: pq.ParquetFile, num_cols: list[str], rg_offsets: list[int], rg_meta: list, s_idx: int, e_idx: int, stride: int = 1) -> np.ndarray:
    needed_rgs = [i for i, (off, m) in enumerate(zip(rg_offsets, rg_meta)) if not (e_idx <= off or s_idx >= off + m.num_rows)]
    tables = [pf.read_row_group(rg_i, columns=num_cols) for rg_i in needed_rgs]
    comb = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    rel_s = s_idx - rg_offsets[needed_rgs[0]]
    w_tbl = comb.slice(rel_s, e_idx - s_idx)
    del tables, comb
    if stride > 1:
        w_tbl = w_tbl.take(pa.array(np.arange(0, len(w_tbl), stride, dtype=np.int64)))
    X = np.column_stack([w_tbl[c].to_numpy(zero_copy_only=False) for c in num_cols]).astype(np.float32)
    del w_tbl
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def worker_train(gpu_id: int, mod_parity: int):
    device_str = f"cuda:{gpu_id}"
    num_cols = json.loads(FEATS_PATH.read_text())["feature_order"]
    pf = pq.ParquetFile(str(PARQUET_PATH))
    total_rows = pf.metadata.num_rows
    rg_meta = [pf.metadata.row_group(i) for i in range(pf.metadata.num_row_groups)]
    rg_offsets = [0]
    for m in rg_meta[:-1]:
        rg_offsets.append(rg_offsets[-1] + m.num_rows)

    lob = np.load(LOB_CACHE_PATH, mmap_mode="r")
    b1 = np.asarray(lob["bid_p"][:, 0], dtype=np.float32)
    a1 = np.asarray(lob["ask_p"][:, 0], dtype=np.float32)

    wins = build_wf_grid(total_rows)
    my_wins = [w for i, w in enumerate(wins) if (i % 2) == mod_parity]
    print(f"[GPU {gpu_id}] Старт обучения {len(my_wins)} каузальных окон Wave Stack v3: {[w[0] for w in my_wins]}", flush=True)

    train_span = 259200  # 72 часа истории
    stride_tr = 5        # шаг 5с на 72ч -> 51,840 независимых волновых срезов!

    for w_k, v_s, v_e in my_wins:
        t0 = time.time()
        w_dir = OUT_MODELS_DIR / f"win_{w_k:02d}"
        w_dir.mkdir(parents=True, exist_ok=True)
        meta_p = w_dir / "metrics.json"
        if meta_p.exists() and all((w_dir / f"{nm}.ubj").exists() for nm in WAVE_MODEL_NAMES):
            # Если win_82 расширился, пересчитываем только последнее окно
            old_m = json.loads(meta_p.read_text())
            if old_m.get("val_range") == [v_s, v_e]:
                print(f"[GPU {gpu_id}] win_{w_k:02d} уже готов, пропуск.", flush=True)
                continue

        # Берем tr_lo..v_s - 360 (минимальный purge 6м; для каждого горизонта H отрезаем хвост v_s - H - 60 -> 0% утечки!)
        max_tr_hi = v_s - 360
        tr_lo = max(0, max_tr_hi - train_span)
        X_tr_all = read_parquet_slice(pf, num_cols, rg_offsets, rg_meta, tr_lo, max_tr_hi, stride=stride_tr)
        tr_indices_abs = np.arange(tr_lo, max_tr_hi, stride_tr, dtype=np.int64)
        y_tr_dict_full = precompute_wave_targets(b1, a1, tr_lo, max_tr_hi)

        # Валидационный срез для честной OOS-проверки IC / AUC (шаг 10с, без хвоста последних H сек)
        val_eval_hi = max(v_s + 600, min(v_e, total_rows - 3610))
        X_val = read_parquet_slice(pf, num_cols, rg_offsets, rg_meta, v_s, val_eval_hi, stride=10)
        y_val_full = precompute_wave_targets(b1, a1, v_s, val_eval_hi)

        metrics = {"win_idx": w_k, "val_range": [v_s, v_e], "models": {}}
        horizon_map = {
            "wave_ret_h300s": (300, "reg"),
            "wave_ret_h900s": (900, "reg"),
            "wave_ret_h1800s": (1800, "reg"),
            "wave_ret_h3600s": (3600, "reg"),
            "wave_launch_30m": (1800, "bin"),
            "wave_top_15m": (900, "bin"),
        }

        dval = xgb.DMatrix(X_val)
        for m_name, (H_purge, m_kind) in horizon_map.items():
            # Строгий каузальный Purge: ни одна обучающая метка не пересекает границу v_s!
            valid_mask = (tr_indices_abs + H_purge + 60) <= v_s
            X_m = X_tr_all[valid_mask]
            y_m = y_tr_dict_full[m_name][::stride_tr][:len(tr_indices_abs)][valid_mask]
            y_v = y_val_full[m_name][::10][:len(X_val)]

            # Разделим X_m на 88% fit + 12% internal early-stopping (с внутренним purge H_purge!)
            cut_es = max(1000, int(len(X_m) * 0.86))
            gap_es = max(1, H_purge // stride_tr)
            dtr = xgb.DMatrix(X_m[:cut_es], label=y_m[:cut_es])
            des = xgb.DMatrix(X_m[min(len(X_m)-100, cut_es + gap_es):], label=y_m[min(len(X_m)-100, cut_es + gap_es):])

            if m_kind == "reg":
                params = {
                    "tree_method": "hist",
                    "device": device_str,
                    "objective": "reg:pseudohubererror",
                    "huber_slope": 1.5,
                    "max_depth": 5,
                    "learning_rate": 0.035,
                    "subsample": 0.75,
                    "colsample_bytree": 0.35,
                    "min_child_weight": 80.0,
                    "reg_lambda": 8.0,
                    "reg_alpha": 2.0,
                    "eval_metric": "mae",
                }
            else:
                pos_rate = float(np.clip(np.mean(y_m[:cut_es]), 0.02, 0.98))
                spw = float(np.clip(np.sqrt((1.0 - pos_rate) / pos_rate), 1.0, 5.0))
                params = {
                    "tree_method": "hist",
                    "device": device_str,
                    "objective": "binary:logistic",
                    "scale_pos_weight": spw,
                    "max_depth": 5,
                    "learning_rate": 0.035,
                    "subsample": 0.75,
                    "colsample_bytree": 0.35,
                    "min_child_weight": 80.0,
                    "reg_lambda": 8.0,
                    "reg_alpha": 2.0,
                    "eval_metric": "logloss",
                }

            bst = xgb.train(
                params,
                dtr,
                num_boost_round=320,
                evals=[(des, "es")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
            bst.save_model(str(w_dir / f"{m_name}.ubj"))
            pred_v = bst.predict(dval)
            if m_kind == "reg":
                ic = float(spearmanr(pred_v, y_v).statistic) if np.std(pred_v) > 1e-6 and np.std(y_v) > 1e-6 else 0.0
                metrics["models"][m_name] = {"oos_ic": round(ic, 4), "best_ntree": int(bst.best_iteration + 1)}
            else:
                auc = float(roc_auc_score(y_v > 0.5, pred_v)) if len(np.unique(y_v > 0.5)) > 1 else 0.5
                metrics["models"][m_name] = {"oos_auc": round(auc, 4), "pos_rate": round(float(np.mean(y_v)), 4), "best_ntree": int(bst.best_iteration + 1)}
            del dtr, des, bst

        del X_tr_all, X_val, dval
        gc.collect()
        meta_p.write_text(json.dumps(metrics, indent=2))
        ic30 = metrics["models"]["wave_ret_h1800s"]["oos_ic"]
        ic60 = metrics["models"]["wave_ret_h3600s"]["oos_ic"]
        auc_l = metrics["models"]["wave_launch_30m"]["oos_auc"]
        auc_t = metrics["models"]["wave_top_15m"]["oos_auc"]
        print(f"[GPU {gpu_id}] ✅ win_{w_k:02d} [{v_s:,}..{v_e:,}] за {time.time()-t0:.1f}с | OOS IC_30m={ic30:+.4f}, IC_60m={ic60:+.4f} | AUC_launch={auc_l:.4f}, AUC_top={auc_t:.4f}", flush=True)


def build_wave_cache():
    print("\n======================================================================")
    print("🌊 СБОРКА 100% КАУЗАЛЬНОЙ МАТРИЦЫ ORDER-1 WAVE STACK (6D)")
    print("======================================================================", flush=True)
    num_cols = json.loads(FEATS_PATH.read_text())["feature_order"]
    pf = pq.ParquetFile(str(PARQUET_PATH))
    total_rows = pf.metadata.num_rows
    rg_meta = [pf.metadata.row_group(i) for i in range(pf.metadata.num_row_groups)]
    rg_offsets = [0]
    for m in rg_meta[:-1]:
        rg_offsets.append(rg_offsets[-1] + m.num_rows)

    wins = build_wf_grid(total_rows)
    wave_mat = np.zeros((total_rows, 6), dtype=np.float32)

    t0 = time.time()
    for step_i, (w_k, v_s, v_e) in enumerate(wins):
        t_s = time.time()
        w_dir = OUT_MODELS_DIR / f"win_{w_k:02d}"
        X_sl = read_parquet_slice(pf, num_cols, rg_offsets, rg_meta, v_s, v_e, stride=1)
        for m_i, nm in enumerate(WAVE_MODEL_NAMES):
            bst = xgb.Booster()
            bst.load_model(str(w_dir / f"{nm}.ubj"))
            bst.set_param({"device": "cuda:0"})
            wave_mat[v_s:v_e, m_i] = bst.predict(xgb.DMatrix(X_sl)).astype(np.float32)
            del bst
        del X_sl
        gc.collect()
        print(f"  [{step_i+1:2d}/{len(wins)}] Слайс win_{w_k:02d} [{v_s:,}..{v_e:,}] собран за {time.time()-t_s:.1f}с", flush=True)

    summary = {f"win_{w_k:02d}": json.loads((OUT_MODELS_DIR / f"win_{w_k:02d}/metrics.json").read_text()) for w_k, _, _ in wins}
    (OUT_MODELS_DIR / "walkforward_wave_summary.json").write_text(json.dumps(summary, indent=2))

    tmp_p = OUT_CACHE_PATH.with_name("order1_wave_matrix_6d_tmp.npz")
    np.savez(tmp_p, wave_matrix_6d=wave_mat, model_names=np.array(WAVE_MODEL_NAMES))
    tmp_p.replace(OUT_CACHE_PATH)
    sub = wave_mat[3016277:]
    print(f"✅ Сохранен {OUT_CACHE_PATH.name} ({wave_mat.shape}) за {time.time()-t0:.1f}с!")
    for i, nm in enumerate(WAVE_MODEL_NAMES):
        print(f"   [{i}] {nm:<18}: mean={sub[::20, i].mean():+.4f} | std={sub[::20, i].std():.4f} | p10={np.percentile(sub[::20, i], 10):+.4f} | p90={np.percentile(sub[::20, i], 90):+.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-gpu", type=int, default=-1)
    parser.add_argument("--mod", type=int, default=0)
    args = parser.parse_args()

    OUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.worker_gpu >= 0:
        worker_train(args.worker_gpu, args.mod)
        return

    print("🚀 ЗАПУСК DUAL-GPU WALK-FORWARD ОБУЧЕНИЯ WAVE STACK V3 (GPU 0 + GPU 1)...", flush=True)
    p0 = subprocess.Popen([sys.executable, __file__, "--worker-gpu", "0", "--mod", "0"])
    p1 = subprocess.Popen([sys.executable, __file__, "--worker-gpu", "1", "--mod", "1"])
    rc0, rc1 = p0.wait(), p1.wait()
    if rc0 != 0 or rc1 != 0:
        raise RuntimeError(f"Ошибка воркеров Wave Stack: rc0={rc0}, rc1={rc1}")
    build_wave_cache()


if __name__ == "__main__":
    main()
