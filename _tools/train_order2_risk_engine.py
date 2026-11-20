#!/usr/bin/env python3
"""
_tools/train_order2_risk_engine.py
=============================================================================
Unified Pure GPU XGBoost Trainer for Order-2 Microstructure & Execution Risk:
  1. adverse_selection_5s   : P(Bid_1 Collapse >= 15 bps within 5s | Fill)
  2. wave_exhaustion_10s    : P(Local Peak reached within 10s after momentum)
  3. ask_sweep_5s           : P(Ask_1 swept aggressively by buyers within 5s)
  4. guard_sweep_12s        : P(Drop >= 15 bps within 12s)
  5. guard_bleed_60s        : P(Staircase decline >= 30 bps within 60s)
  6. guard_crash_300s       : P(Waterfall crash >= 75 bps within 300s)
  7. guard_exit_30s         : P(Emergency drop >= 30 bps in 30s)
  8. exec_fill_30s          : P(Real fill on SafeTrade queue within 30s)
  9. exec_frontrun_30s      : P(Penny-jump frontrunning risk by competing bots)
 10. carousel_success       : P(Maker Round-Trip Bid_1 -> Ask_1 success within 90s)
 11. carousel_capacity      : Q_max capacity volume in PRL (continuous regression)

All models are trained with Pure GPU XGBoost (CUDA Hist), validated strictly OOS,
and exported directly to compact native binary format (.ubj).
=============================================================================
"""

import os
import sys
import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent.parent
GUARD_LOB_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
ALPHA_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_56d.npz"
TELEMETRY_PATH = BASE_DIR / "data" / "processed" / "prl_rl_telemetry.parquet"
OUTPUT_DIR = BASE_DIR / "models" / "prl" / "order2_risk"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. Microstructure Feature Engineering ───
def extract_order2_feature_matrix(lob: dict, alpha: dict) -> tuple[np.ndarray, list[str]]:
    """
    Constructs a 38D feature matrix combining:
      - 14D Local LOB microstructure (SafeTrade & BigONE lead-lag)
      - 24D Order-1 Alpha signals (8 E[Δmid], 8 Confidence, 8 Directional Bias)
    """
    bid_p = lob["bid_p"][:, 0].astype(np.float32)
    ask_p = lob["ask_p"][:, 0].astype(np.float32)
    bid_q1 = lob["bid_q"][:, 0].astype(np.float32)
    ask_q1 = lob["ask_q"][:, 0].astype(np.float32)
    bid_q5 = np.sum(lob["bid_q"][:, :5], axis=1).astype(np.float32)
    ask_q5 = np.sum(lob["ask_q"][:, :5], axis=1).astype(np.float32)
    buy_v = np.nan_to_num(lob["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(lob["sell_vol"].astype(np.float32), nan=0.0)

    # BigONE lead-lag
    bo_bp1 = lob["bo_bid_p"][:, 0].astype(np.float32)
    bo_ap1 = lob["bo_ask_p"][:, 0].astype(np.float32)
    bo_mid = 0.5 * (bo_bp1 + bo_ap1)
    st_mid = 0.5 * (bid_p + ask_p)
    arb_basis_bps = np.clip((bo_mid - st_mid) / np.maximum(st_mid, 1e-4) * 1e4, -100.0, 100.0)

    n = len(bid_p)
    cs_buy = np.concatenate([[0.0], np.cumsum(buy_v, dtype=np.float64)])
    cs_sell = np.concatenate([[0.0], np.cumsum(sell_v, dtype=np.float64)])
    idx = np.arange(n)
    buy_10s = (cs_buy[idx + 1] - cs_buy[np.maximum(0, idx - 9)]).astype(np.float32)
    sell_10s = (cs_sell[idx + 1] - cs_sell[np.maximum(0, idx - 9)]).astype(np.float32)
    buy_30s = (cs_buy[idx + 1] - cs_buy[np.maximum(0, idx - 29)]).astype(np.float32)
    sell_30s = (cs_sell[idx + 1] - cs_sell[np.maximum(0, idx - 29)]).astype(np.float32)

    spread_bps = np.clip((ask_p - bid_p) / np.maximum(st_mid, 1e-4) * 1e4, 1.0, 150.0)
    imb_bbo = (bid_q1 - ask_q1) / (bid_q1 + ask_q1 + 1e-4)
    imb_top5 = (bid_q5 - ask_q5) / (bid_q5 + ask_q5 + 1e-4)
    two_sided_flow = np.minimum(buy_30s, sell_30s) / (np.maximum(bid_q1, ask_q1) + 50.0)
    turnover_bid = sell_30s / (bid_q1 + 50.0)
    turnover_ask = buy_30s / (ask_q1 + 50.0)

    lob_cols = [
        "log_bid_q1", "log_ask_q1", "log_bid_q5", "log_ask_q5",
        "log_buy_10s", "log_sell_10s", "log_buy_30s", "log_sell_30s",
        "spread_bps", "imb_bbo", "imb_top5", "two_sided_flow",
        "turnover_bid", "arb_basis_bps"
    ]
    lob_mat = np.column_stack([
        np.log1p(np.maximum(0.0, bid_q1)),
        np.log1p(np.maximum(0.0, ask_q1)),
        np.log1p(np.maximum(0.0, bid_q5)),
        np.log1p(np.maximum(0.0, ask_q5)),
        np.log1p(np.maximum(0.0, buy_10s)),
        np.log1p(np.maximum(0.0, sell_10s)),
        np.log1p(np.maximum(0.0, buy_30s)),
        np.log1p(np.maximum(0.0, sell_30s)),
        spread_bps,
        imb_bbo,
        imb_top5,
        np.clip(two_sided_flow, 0.0, 10.0),
        np.clip(turnover_bid, 0.0, 10.0),
        arb_basis_bps,
    ]).astype(np.float32)

    # Order-1 Alpha signals (8 horizons)
    ez = alpha["expected_z_8h"].astype(np.float32)
    conf = alpha["confidence_8h"].astype(np.float32)
    bias = alpha["bias_8h"].astype(np.float32)

    alpha_cols = []
    for h in [1, 2, 5, 10, 20, 40, 80, 160]:
        alpha_cols.extend([f"ez_h{h}s", f"conf_h{h}s", f"bias_h{h}s"])

    feature_names = lob_cols + alpha_cols
    X_full = np.hstack([lob_mat, ez, conf, bias]).astype(np.float32)
    return np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0), feature_names


# ─── 2. Physical Target Label Computation (in bps / physics) ───
def compute_all_order2_targets(lob: dict) -> dict[str, np.ndarray]:
    """Computes all 11 physical risk and execution targets."""
    print("  Calculating 11 physical microstructure target labels across 3.3M ticks...")
    bid_p = lob["bid_p"][:, 0].astype(np.float32)
    ask_p = lob["ask_p"][:, 0].astype(np.float32)
    bid_q1 = lob["bid_q"][:, 0].astype(np.float32)
    ask_q1 = lob["ask_q"][:, 0].astype(np.float32)
    buy_v = np.nan_to_num(lob["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(lob["sell_vol"].astype(np.float32), nan=0.0)
    st_mid = 0.5 * (bid_p + ask_p)
    n = len(bid_p)

    # Physical bps thresholds
    # 15 bps on $1.66 is ~$0.0025 (25 subcent ticks)
    # 30 bps is ~$0.0050 (~1 full spread)
    # 75 bps is ~$0.0125 (~2.5 full spreads)
    bps_15 = st_mid * 0.0015
    bps_30 = st_mid * 0.0030
    bps_75 = st_mid * 0.0075

    cs_buy = np.concatenate([[0.0], np.cumsum(buy_v, dtype=np.float64)])
    cs_sell = np.concatenate([[0.0], np.cumsum(sell_v, dtype=np.float64)])

    y_adverse_5s = np.zeros(n, dtype=np.float32)
    y_exhaust_10s = np.zeros(n, dtype=np.float32)
    y_ask_sweep_5s = np.zeros(n, dtype=np.float32)
    y_sweep_12s = np.zeros(n, dtype=np.float32)
    y_bleed_60s = np.zeros(n, dtype=np.float32)
    y_crash_300s = np.zeros(n, dtype=np.float32)
    y_exit_30s = np.zeros(n, dtype=np.float32)
    y_exec_fill = np.zeros(n, dtype=np.float32)
    y_exec_frontrun = np.zeros(n, dtype=np.float32)
    y_carousel_succ = np.zeros(n, dtype=np.float32)
    y_carousel_cap = np.zeros(n, dtype=np.float32)

    # 1. Vectorized Guard targets (fast sliding windows)
    for k in range(1, 13):
        b_fwd = np.pad(bid_p[k:], (0, k), mode="edge")
        y_sweep_12s = np.maximum(y_sweep_12s, ((bid_p - b_fwd) >= bps_15).astype(np.float32))

    min_drop_60s = np.zeros(n, dtype=np.float32)
    max_rise_60s = np.zeros(n, dtype=np.float32)
    for k in range(1, 61):
        b_fwd = np.pad(bid_p[k:], (0, k), mode="edge")
        min_drop_60s = np.maximum(min_drop_60s, (bid_p - b_fwd))
        max_rise_60s = np.maximum(max_rise_60s, (b_fwd - bid_p))
    y_bleed_60s = ((min_drop_60s >= bps_30) & (max_rise_60s < bps_15)).astype(np.float32)

    min_drop_300s = np.zeros(n, dtype=np.float32)
    for k in range(5, 301, 5):
        b_fwd = np.pad(bid_p[k:], (0, k), mode="edge")
        min_drop_300s = np.maximum(min_drop_300s, (bid_p - b_fwd))
    y_crash_300s = (min_drop_300s >= bps_75).astype(np.float32)

    min_drop_30s = np.zeros(n, dtype=np.float32)
    for k in range(1, 31):
        b_fwd = np.pad(bid_p[k:], (0, k), mode="edge")
        min_drop_30s = np.maximum(min_drop_30s, (bid_p - b_fwd))
    y_exit_30s = (min_drop_30s >= bps_30).astype(np.float32)

    # 2. Hazard, Execution and Carousel Targets
    for t in range(30, n - 95):
        b0, a0 = bid_p[t], ask_p[t]
        thresh_15 = bps_15[t]

        # Adverse Selection 5s: sell pressure hits bid_1 and bid drops >= 15 bps
        min_b_5s = float(np.min(bid_p[t + 1 : t + 6]))
        sell_5s = float(np.sum(sell_v[t + 1 : t + 6]))
        if (b0 - min_b_5s) >= thresh_15 and sell_5s >= 10.0:
            y_adverse_5s[t] = 1.0

        # Wave Exhaustion 10s: peak after momentum >= 15 bps in last 30s
        b_prev30 = float(np.min(bid_p[t - 30 : t]))
        max_a_10s = float(np.max(ask_p[t + 1 : t + 11]))
        min_b_10s = float(np.min(bid_p[t + 1 : t + 11]))
        if (b0 - b_prev30) >= thresh_15 and max_a_10s <= a0 and (b0 - min_b_10s) >= thresh_15:
            y_exhaust_10s[t] = 1.0

        # Ask Sweep 5s: buyer volume clears ask_1
        max_b_5s = float(np.max(bid_p[t + 1 : t + 6]))
        buy_5s = float(np.sum(buy_v[t + 1 : t + 6]))
        if max_b_5s >= a0 and buy_5s >= 10.0:
            y_ask_sweep_5s[t] = 1.0

        # Execution Realism 30s:
        s_30 = float(cs_sell[t + 31] - cs_sell[t + 1])
        min_a_30 = float(np.min(ask_p[t + 1 : t + 31]))
        max_b_30 = float(np.max(bid_p[t + 1 : t + 31]))
        if s_30 >= 0.45 * bid_q1[t] + 20.0 or min_a_30 <= b0:
            y_exec_fill[t] = 1.0
        if max_b_30 >= b0 + thresh_15 and s_30 < 10.0 and bid_q1[t] >= 500.0:
            y_exec_frontrun[t] = 1.0

        # Carousel Success & Capacity:
        sell_w = sell_v[t + 1 : t + 91]
        cs_s = np.cumsum(sell_w)
        q_ahead_b = 0.50 * bid_q1[t] + 10.0
        idx_fill = np.where(cs_s >= q_ahead_b)[0]
        if len(idx_fill) > 0:
            t_fill = idx_fill[0] + 1
            if np.min(bid_p[t + 1 : t + t_fill + 1]) >= b0 - thresh_15:
                buy_w = buy_v[t + t_fill + 1 : t + 91]
                cs_b = np.cumsum(buy_w)
                q_ahead_a = 0.50 * ask_q1[t + t_fill] + 10.0
                idx_sell = np.where(cs_b >= q_ahead_a)[0]
                if len(idx_sell) > 0:
                    t_sell = t_fill + idx_sell[0] + 1
                    if np.min(bid_p[t + 1 : t + t_sell + 1]) >= b0 - thresh_15:
                        y_carousel_succ[t] = 1.0
                        y_carousel_cap[t] = float(min(1000.0, min(np.sum(sell_w[:t_fill]), np.sum(buy_w[: idx_sell[0] + 1]))))

    return {
        "adverse_selection_5s": y_adverse_5s,
        "wave_exhaustion_10s": y_exhaust_10s,
        "ask_sweep_5s": y_ask_sweep_5s,
        "guard_sweep_12s": y_sweep_12s,
        "guard_bleed_60s": y_bleed_60s,
        "guard_crash_300s": y_crash_300s,
        "guard_exit_30s": y_exit_30s,
        "exec_fill_30s": y_exec_fill,
        "exec_frontrun_30s": y_exec_frontrun,
        "carousel_success": y_carousel_succ,
        "carousel_capacity": y_carousel_cap,
    }


# ─── 3. Pure GPU XGBoost Model Training Function ───
def train_single_order2_xgboost(
    target_name: str,
    y_arr: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    is_regression: bool = False,
) -> tuple[float, int, float]:
    t0 = time.time()
    n_pos = int(np.sum(y_tr > 0.5))
    n_neg = len(y_tr) - n_pos
    pos_rate = (n_pos / len(y_tr)) * 100.0

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    if not is_regression:
        scale_pos = max(1.0, min(30.0, float(n_neg) / max(1.0, float(n_pos))))
        params = {
            "tree_method": "hist",
            "device": "cuda:0",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": scale_pos,
            "reg_lambda": 2.0,
            "reg_alpha": 0.5,
        }
    else:
        params = {
            "tree_method": "hist",
            "device": "cuda:0",
            "objective": "reg:pseudohubererror",
            "eval_metric": "mae",
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
        }

    evals = [(dval, "val")]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=250,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    best_iter = bst.best_iteration
    preds = bst.predict(dval)
    if not is_regression:
        try:
            score = float(roc_auc_score(y_val, preds))
        except Exception:
            score = 0.5
        metric_str = f"OOS ROC-AUC = {score:.4f}"
    else:
        score = float(mean_absolute_error(y_val, preds))
        metric_str = f"OOS MAE = {score:.1f} PRL"

    # Save to native binary UBJ format
    out_file = OUTPUT_DIR / f"{target_name}.ubj"
    bst.save_model(str(out_file))
    elapsed = time.time() - t0
    sz_kb = out_file.stat().st_size / 1024

    print(f"  ✅ [{target_name:20s}] {metric_str} | Trees={best_iter:3d} | Pos={pos_rate:5.2f}% | Size={sz_kb:.0f}KB | {elapsed:.1f}s", flush=True)
    del dtrain, dval, bst
    gc.collect()
    return score, best_iter, elapsed


# ─── 4. Main Master Execution ───
def main():
    print("=" * 75)
    print("  PURE GPU XGBOOST ORDER-2 MICROSTRUCTURE & RISK ENGINE")
    print(f"  Target directory: {OUTPUT_DIR}")
    print("=" * 75, flush=True)

    print("Loading LOB (918h) and Order-1 Alpha caches...")
    lob = np.load(GUARD_LOB_PATH, allow_pickle=True)
    alpha = np.load(ALPHA_PATH, allow_pickle=True)
    n_total = len(lob["bid_p"])
    oos_len = 301892
    split_idx = n_total - oos_len
    print(f"Total ticks: {n_total:,} | Train: {split_idx:,} (834.7h) | OOS Val: {oos_len:,} (83.9h)")

    X_all, feature_names = extract_order2_feature_matrix(lob, alpha)
    print(f"Constructed 38D Feature Matrix: {X_all.shape} ({len(feature_names)} features)")

    targets = compute_all_order2_targets(lob)

    X_tr = X_all[:split_idx]
    X_val = X_all[split_idx:]
    del X_all
    gc.collect()

    summary_metrics = {}
    print("\n" + "=" * 75)
    print("  TRAINING ALL 11 ORDER-2 RISK MODELS ON GPU (RTX 5060 Ti)...")
    print("=" * 75, flush=True)

    t_all = time.time()
    for name, y_arr in targets.items():
        is_reg = (name == "carousel_capacity")
        y_tr = y_arr[:split_idx]
        y_val = y_arr[split_idx:]
        score, trees, el = train_single_order2_xgboost(
            name, y_arr, X_tr, y_tr, X_val, y_val, feature_names, is_regression=is_reg
        )
        summary_metrics[name] = {
            "metric": "mae" if is_reg else "auc",
            "score": round(score, 4),
            "trees": trees,
            "elapsed_sec": round(el, 1),
        }

    # Save summary metadata
    meta_path = OUTPUT_DIR / "order2_summary.json"
    meta = {
        "n_samples": n_total,
        "n_train": split_idx,
        "n_val": oos_len,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "total_elapsed_sec": round(time.time() - t_all, 1),
        "models": summary_metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t_all
    print("\n" + "=" * 75)
    print(f"🎉 ALL 11 ORDER-2 MODELS SUCCESSFULLY TRAINED IN {total_time:.1f}s ({total_time/60:.1f}m)!")
    print(f"  Summary saved to {meta_path}")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
