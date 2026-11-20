#!/usr/bin/env python3
"""
_tools/walkforward_order2_trainer.py
=============================================================================
Process-Isolated High-Capacity Walk-Forward Trainer for Order-2 Risk Models.

Trains 11 models per window on GPU (max_depth=8, 600 estimators, lr=0.035):
  1. adverse_selection_5s
  2. wave_exhaustion_10s
  3. ask_sweep_5s
  4. guard_sweep_12s
  5. guard_bleed_60s
  6. guard_crash_300s
  7. guard_exit_30s
  8. exec_fill_30s
  9. exec_frontrun_30s
 10. carousel_success
 11. carousel_capacity

Features: 66D (18 LOB microstructure + 48 Order-1 Alpha signals).
Saves models to models/prl/order2_wf/win_{i:02d}/{target_name}.ubj
=============================================================================
"""

import os
import sys
import json
import time
import argparse
import gc
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_absolute_error
from numba import njit, prange

BASE_DIR = Path(__file__).resolve().parent.parent
GUARD_LOB_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
ALPHA_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_56d.npz"
WF_SUMMARY_PATH = BASE_DIR / "models" / "prl" / "variant_a_order1" / "walkforward_summary.json"
OUTPUT_DIR = BASE_DIR / "models" / "prl" / "order2_wf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = [
    "adverse_selection_5s",
    "wave_exhaustion_10s",
    "ask_sweep_5s",
    "guard_sweep_12s",
    "guard_bleed_60s",
    "guard_crash_300s",
    "guard_exit_30s",
    "exec_fill_30s",
    "exec_frontrun_30s",
    "carousel_success",
    "carousel_capacity",
]


# ─── Numba Parallel Target Computation ───
@njit(parallel=True, fastmath=True)
def compute_numba_targets(bid_p, ask_p, bid_q1, ask_q1, sell_v, buy_v, bps_15, bps_30, bps_75, n):
    y_adv = np.zeros(n, dtype=np.float32)
    y_exh = np.zeros(n, dtype=np.float32)
    y_swp = np.zeros(n, dtype=np.float32)
    y_sw12 = np.zeros(n, dtype=np.float32)
    y_bleed = np.zeros(n, dtype=np.float32)
    y_crash = np.zeros(n, dtype=np.float32)
    y_exit = np.zeros(n, dtype=np.float32)
    y_fill = np.zeros(n, dtype=np.float32)
    y_front = np.zeros(n, dtype=np.float32)
    y_car_succ = np.zeros(n, dtype=np.float32)
    y_car_cap = np.zeros(n, dtype=np.float32)

    for t in prange(30, n - 95):
        b0 = bid_p[t]
        a0 = ask_p[t]
        th15 = bps_15[t]
        th30 = bps_30[t]
        th75 = bps_75[t]

        # 1. Sweep 12s
        min_b12 = bid_p[t + 1]
        for k in range(2, 13):
            if bid_p[t + k] < min_b12:
                min_b12 = bid_p[t + k]
        if (b0 - min_b12) >= th15:
            y_sw12[t] = 1.0

        # 2. Bleed 60s
        min_b60 = bid_p[t + 1]
        max_b60 = bid_p[t + 1]
        for k in range(2, 61):
            if bid_p[t + k] < min_b60:
                min_b60 = bid_p[t + k]
            if bid_p[t + k] > max_b60:
                max_b60 = bid_p[t + k]
        if (b0 - min_b60) >= th30 and (max_b60 - b0) < th15:
            y_bleed[t] = 1.0

        # 3. Crash 300s
        min_b300 = bid_p[t + 5]
        for k in range(10, 301, 5):
            if bid_p[t + k] < min_b300:
                min_b300 = bid_p[t + k]
        if (b0 - min_b300) >= th75:
            y_crash[t] = 1.0

        # 4. Exit 30s
        min_b30 = bid_p[t + 1]
        for k in range(2, 31):
            if bid_p[t + k] < min_b30:
                min_b30 = bid_p[t + k]
        if (b0 - min_b30) >= th30:
            y_exit[t] = 1.0

        # 5. Adverse Selection 5s
        min_b5 = bid_p[t + 1]
        s5 = sell_v[t + 1]
        for k in range(2, 6):
            if bid_p[t + k] < min_b5:
                min_b5 = bid_p[t + k]
            s5 += sell_v[t + k]
        if (b0 - min_b5) >= th15 and s5 >= 10.0:
            y_adv[t] = 1.0

        # 6. Wave Exhaustion 10s
        min_b_past = bid_p[t - 30]
        for k in range(1, 30):
            if bid_p[t - k] < min_b_past:
                min_b_past = bid_p[t - k]
        max_a10 = ask_p[t + 1]
        min_b10 = bid_p[t + 1]
        for k in range(2, 11):
            if ask_p[t + k] > max_a10:
                max_a10 = ask_p[t + k]
            if bid_p[t + k] < min_b10:
                min_b10 = bid_p[t + k]
        if (b0 - min_b_past) >= th15 and max_a10 <= a0 and (b0 - min_b10) >= th15:
            y_exh[t] = 1.0

        # 7. Ask Sweep 5s
        max_b5 = bid_p[t + 1]
        b5_vol = buy_v[t + 1]
        for k in range(2, 6):
            if bid_p[t + k] > max_b5:
                max_b5 = bid_p[t + k]
            b5_vol += buy_v[t + k]
        if max_b5 >= a0 and b5_vol >= 10.0:
            y_swp[t] = 1.0

        # 8. Execution Realism (30s)
        s30_vol = 0.0
        min_a30 = ask_p[t + 1]
        max_b30 = bid_p[t + 1]
        for k in range(1, 31):
            s30_vol += sell_v[t + k]
            if ask_p[t + k] < min_a30:
                min_a30 = ask_p[t + k]
            if bid_p[t + k] > max_b30:
                max_b30 = bid_p[t + k]
        if s30_vol >= (0.45 * bid_q1[t] + 20.0) or min_a30 <= b0:
            y_fill[t] = 1.0
        if max_b30 >= (b0 + th15) and s30_vol < 10.0 and bid_q1[t] >= 500.0:
            y_front[t] = 1.0

        # 9. Carousel Round-Trip
        s90_vol = 0.0
        q_ahead_b = 0.50 * bid_q1[t] + 10.0
        t_fill = -1
        for k in range(1, 91):
            s90_vol += sell_v[t + k]
            if s90_vol >= q_ahead_b:
                t_fill = k
                break
        if t_fill > 0:
            min_b_during = bid_p[t + 1]
            for k in range(2, t_fill + 1):
                if bid_p[t + k] < min_b_during:
                    min_b_during = bid_p[t + k]
            if min_b_during >= (b0 - th15):
                b90_vol = 0.0
                q_ahead_a = 0.50 * ask_q1[t + t_fill] + 10.0
                t_sell = -1
                for k in range(t_fill + 1, 91):
                    b90_vol += buy_v[t + k]
                    if b90_vol >= q_ahead_a:
                        t_sell = k
                        break
                if t_sell > 0:
                    min_b_cycle = bid_p[t + t_fill + 1]
                    for k in range(t_fill + 2, t_sell + 1):
                        if bid_p[t + k] < min_b_cycle:
                            min_b_cycle = bid_p[t + k]
                    if min_b_cycle >= (b0 - th15):
                        y_car_succ[t] = 1.0
                        cap_val = s90_vol if s90_vol < b90_vol else b90_vol
                        y_car_cap[t] = cap_val if cap_val < 1000.0 else 1000.0

    return (
        y_adv, y_exh, y_swp, y_sw12, y_bleed, y_crash,
        y_exit, y_fill, y_front, y_car_succ, y_car_cap
    )


# ─── 66D Feature Extraction ───
def build_66d_features(lob: dict, alpha: dict) -> tuple[np.ndarray, list[str]]:
    bid_p = lob["bid_p"][:, 0].astype(np.float32)
    ask_p = lob["ask_p"][:, 0].astype(np.float32)
    bid_q1 = lob["bid_q"][:, 0].astype(np.float32)
    ask_q1 = lob["ask_q"][:, 0].astype(np.float32)
    bid_q5 = np.sum(lob["bid_q"][:, :5], axis=1).astype(np.float32)
    ask_q5 = np.sum(lob["ask_q"][:, :5], axis=1).astype(np.float32)
    buy_v = np.nan_to_num(lob["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(lob["sell_vol"].astype(np.float32), nan=0.0)

    st_mid = 0.5 * (bid_p + ask_p)
    bo_bp1 = lob["bo_bid_p"][:, 0].astype(np.float32)
    bo_ap1 = lob["bo_ask_p"][:, 0].astype(np.float32)
    bo_mid = 0.5 * (bo_bp1 + bo_ap1)
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
    spread_past = np.pad(spread_bps[:-15], (15, 0), mode="edge")
    spread_chg_15s = spread_bps - spread_past

    bbo_share_bid = bid_q1 / (bid_q5 + 1e-4)
    bbo_share_ask = ask_q1 / (ask_q5 + 1e-4)
    imb_bbo = (bid_q1 - ask_q1) / (bid_q1 + ask_q1 + 1e-4)
    imb_top5 = (bid_q5 - ask_q5) / (bid_q5 + ask_q5 + 1e-4)
    two_sided = np.minimum(buy_30s, sell_30s) / (np.maximum(bid_q1, ask_q1) + 50.0)
    turn_b = sell_30s / (bid_q1 + 50.0)
    turn_a = buy_30s / (ask_q1 + 50.0)

    lob_cols = [
        "log_bid_q1", "log_ask_q1", "log_bid_q5", "log_ask_q5",
        "bbo_share_bid", "bbo_share_ask", "spread_bps", "spread_chg_15s",
        "log_buy_10s", "log_sell_10s", "log_buy_30s", "log_sell_30s",
        "imb_bbo", "imb_top5", "two_sided_flow", "turnover_bid", "turnover_ask",
        "arb_basis_bps"
    ]
    lob_mat = np.column_stack([
        np.log1p(np.maximum(0.0, bid_q1)),
        np.log1p(np.maximum(0.0, ask_q1)),
        np.log1p(np.maximum(0.0, bid_q5)),
        np.log1p(np.maximum(0.0, ask_q5)),
        bbo_share_bid,
        bbo_share_ask,
        spread_bps,
        spread_chg_15s,
        np.log1p(np.maximum(0.0, buy_10s)),
        np.log1p(np.maximum(0.0, sell_10s)),
        np.log1p(np.maximum(0.0, buy_30s)),
        np.log1p(np.maximum(0.0, sell_30s)),
        imb_bbo,
        imb_top5,
        np.clip(two_sided, 0.0, 10.0),
        np.clip(turn_b, 0.0, 10.0),
        np.clip(turn_a, 0.0, 10.0),
        arb_basis_bps,
    ]).astype(np.float32)

    # 48D Order-1 Alpha signals (6 features x 8 horizons)
    # ez, conf, bias, p_dump, p_pump, entropy
    probs_56 = alpha["probs_56d"].astype(np.float32)  # [N, 56]
    ez = alpha["expected_z_8h"].astype(np.float32)
    conf = alpha["confidence_8h"].astype(np.float32)
    bias = alpha["bias_8h"].astype(np.float32)

    alpha_cols = []
    alpha_features = []
    for h_i, h in enumerate([1, 2, 5, 10, 20, 40, 80, 160]):
        p_c = probs_56[:, h_i * 7 : (h_i + 1) * 7]
        p_dump = p_c[:, 0]  # P(C = -3)
        p_pump = p_c[:, 6]  # P(C = +3)
        ent = -np.sum(p_c * np.log(np.maximum(p_c, 1e-6)), axis=1)

        alpha_features.extend([
            ez[:, h_i], conf[:, h_i], bias[:, h_i],
            p_dump, p_pump, ent
        ])
        alpha_cols.extend([
            f"ez_h{h}s", f"conf_h{h}s", f"bias_h{h}s",
            f"pdump_h{h}s", f"ppump_h{h}s", f"entropy_h{h}s"
        ])

    alpha_mat = np.column_stack(alpha_features).astype(np.float32)
    all_cols = lob_cols + alpha_cols
    X_full = np.hstack([lob_mat, alpha_mat]).astype(np.float32)
    return np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0), all_cols


# ─── Single Window Trainer ───
def train_window_order2(
    win_idx: int,
    X_all: np.ndarray,
    targets_dict: dict[str, np.ndarray],
    feature_names: list[str],
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    device_str: str,
    force: bool = False,
) -> dict:
    win_dir = OUTPUT_DIR / f"win_{win_idx:02d}"
    win_dir.mkdir(parents=True, exist_ok=True)
    meta_path = win_dir / f"meta_order2_win_{win_idx:02d}.json"

    if meta_path.exists() and not force:
        try:
            with open(meta_path) as f:
                d = json.load(f)
            if d.get("completed", False):
                return d
        except Exception:
            pass

    t0_win = time.time()
    X_tr = X_all[train_start:train_end]
    X_val = X_all[val_start:val_end]
    n_tr = len(X_tr)
    n_val = len(X_val)

    dval = xgb.DMatrix(X_val, feature_names=feature_names)
    win_metrics = {}

    for name in TARGET_NAMES:
        t0_m = time.time()
        y_all = targets_dict[name]
        y_tr = y_all[train_start:train_end]
        y_val = y_all[val_start:val_end]

        is_reg = (name == "carousel_capacity")
        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
        dval.set_info(label=y_val)

        if not is_reg:
            n_pos = int(np.sum(y_tr > 0.5))
            n_neg = n_tr - n_pos
            scale_pos = max(1.0, min(25.0, float(n_neg) / max(1.0, float(n_pos))))
            params = {
                "tree_method": "hist",
                "device": device_str,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 8,            # Capacity upgrade: depth 8
                "learning_rate": 0.035,     # Smooth shrinkage
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "scale_pos_weight": scale_pos,
                "reg_lambda": 3.0,
                "reg_alpha": 1.0,
            }
        else:
            params = {
                "tree_method": "hist",
                "device": device_str,
                "objective": "reg:pseudohubererror",
                "eval_metric": "mae",
                "max_depth": 8,
                "learning_rate": 0.035,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
            }

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=600,            # Capacity upgrade: 600 trees
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        preds = bst.predict(dval)
        if not is_reg:
            try:
                score = float(roc_auc_score(y_val, preds))
            except Exception:
                score = 0.5
        else:
            score = float(mean_absolute_error(y_val, preds))

        out_file = win_dir / f"{name}.ubj"
        bst.save_model(str(out_file))
        el_m = time.time() - t0_m
        sz_kb = out_file.stat().st_size / 1024

        win_metrics[name] = {
            "score": round(score, 4),
            "trees": bst.best_iteration,
            "size_kb": round(sz_kb, 1),
            "time_sec": round(el_m, 1),
        }
        del dtr, bst
        gc.collect()

    del dval
    gc.collect()

    elapsed_win = time.time() - t0_win
    mean_auc = np.mean([v["score"] for k, v in win_metrics.items() if k != "carousel_capacity"])
    meta = {
        "window_id": f"win_{win_idx:02d}",
        "win_idx": win_idx,
        "completed": True,
        "train_range": [train_start, train_end],
        "val_range": [val_start, val_end],
        "mean_auc": round(float(mean_auc), 4),
        "elapsed_sec": round(elapsed_win, 1),
        "models": win_metrics,
    }

    tmp_meta = meta_path.with_name(f"{meta_path.name}.tmp.{os.getpid()}")
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.replace(meta_path)

    print(f"  ✅ [{device_str}] Win {win_idx:02d} [{train_start:,}..{val_end:,}] Completed in {elapsed_win:.1f}s | Mean OOS AUC: {mean_auc:.4f}", flush=True)
    return meta


# ─── Main Worker CLI ───
def main():
    parser = argparse.ArgumentParser(description="Order-2 Walk-Forward Worker")
    parser.add_argument("--windows", type=str, required=True, help="Comma-separated list of window indices")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device string")
    parser.add_argument("--force", action="store_true", help="Force re-training")
    args = parser.parse_args()

    win_list = [int(w.strip()) for w in args.windows.split(",") if w.strip().isdigit()]
    print("=" * 70)
    print(f"  ORDER-2 WALK-FORWARD WORKER [{args.device}] — Windows: {win_list[:8]}... ({len(win_list)} total)")
    print("=" * 70, flush=True)

    print("Loading LOB & Order-1 Alpha caches...")
    lob = np.load(GUARD_LOB_PATH, allow_pickle=True)
    alpha = np.load(ALPHA_PATH, allow_pickle=True)
    n_total = len(lob["bid_p"])

    print("Building 66D Microstructure + Order-1 Feature Matrix...")
    X_all, feature_names = build_66d_features(lob, alpha)
    print(f"66D Feature Matrix ready: {X_all.shape}")

    print("Computing Numba accelerated physical target labels...")
    bid_p = lob["bid_p"][:, 0].astype(np.float32)
    ask_p = lob["ask_p"][:, 0].astype(np.float32)
    bid_q1 = lob["bid_q"][:, 0].astype(np.float32)
    ask_q1 = lob["ask_q"][:, 0].astype(np.float32)
    buy_v = np.nan_to_num(lob["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(lob["sell_vol"].astype(np.float32), nan=0.0)
    st_mid = 0.5 * (bid_p + ask_p)

    bps_15 = st_mid * 0.0015
    bps_30 = st_mid * 0.0030
    bps_75 = st_mid * 0.0075

    t0_nb = time.time()
    t_arrays = compute_numba_targets(bid_p, ask_p, bid_q1, ask_q1, sell_v, buy_v, bps_15, bps_30, bps_75, n_total)
    targets_dict = {name: t_arrays[i] for i, name in enumerate(TARGET_NAMES)}
    print(f"Numba physical targets computed in {time.time() - t0_nb:.2f}s!")

    with open(WF_SUMMARY_PATH) as f:
        summary_order1 = json.load(f)

    # Walk-Forward parameters
    # Train: 72 hours (259,200 ticks), Val: 12 hours (43,200 ticks)
    # Causal shift: window i trains on [val_start_i - 259200 .. val_start_i]
    WIN_TRAIN_72H = 259200
    WIN_STEP_12H = 43200

    t_all = time.time()
    for w_i in win_list:
        key = f"win_{w_i:02d}"
        if key in summary_order1:
            val_start, val_end = summary_order1[key]["val_range"]
        else:
            val_start = w_i * WIN_STEP_12H + 172800
            val_end = val_start + WIN_STEP_12H

        train_start = max(0, val_start - WIN_TRAIN_72H)
        train_end = val_start

        train_window_order2(
            w_i, X_all, targets_dict, feature_names,
            train_start, train_end, val_start, val_end,
            args.device, force=args.force
        )

    print(f"\n🎉 [{args.device}] All assigned windows completed in {time.time() - t_all:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
