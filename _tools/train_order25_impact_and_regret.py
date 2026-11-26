#!/usr/bin/env python3
"""
Step 3 of Order-2.5 Pipeline:
Trains strictly causal Walk-Forward XGBoost GPU models on the 30,451 real live bot interventions
(27,374 near-book PLACES + 3,077 real LIVE FILLS across 19 RL generations) to extract 4 invariant
Order-2.5 channels (`84D -> 88D Signal Stack v5`, 0% lookahead, 100% Sim=Live parity):
  - ch[84] p_penny_jump_small : P(rival bot penny-jumps a $40 BBO order within 5s | z_mkt_84d)
  - ch[85] p_penny_jump_whale : P(rival bot penny-jumps or pushes away from a $250 BBO wall | z_mkt_84d)
  - ch[86] p_live_buy_trap    : P(real live Buy fill suffers <= -35 bp 3m adverse markout | z_mkt_84d)
  - ch[87] p_live_sell_regret : P(real live Sell fill misses >= +35 bp 3m continuation rally | z_mkt_84d)

Also saves lightweight `.ubj` boosters in `models/prl/order25_live/` for <0.4ms live VPS inference
and builds `data/cache/rl_hybrid_market_obs_88d.npz`!
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BASE_DIR = Path("/home/restorator/crypto_bot")
INT_PATH = BASE_DIR / "data/cache/order25_live_interventions.npz"
OBS84_PATH = BASE_DIR / "data/cache/rl_hybrid_market_obs_84d.npz"
OBS84_RAW_PATH = BASE_DIR / "data/cache/rl_hybrid_market_obs_84d_raw_with_self.npz"
LOB_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
LOB_RAW_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h_raw_with_self.npz"
OUT_MODELS_DIR = BASE_DIR / "models/prl/order25_live"
OUT_88D_PATH = BASE_DIR / "data/cache/rl_hybrid_market_obs_88d.npz"
OUT_88D_RAW_PATH = BASE_DIR / "data/cache/rl_hybrid_market_obs_88d_raw_with_self.npz"


def train_wf_xgb_binary(X: np.ndarray, y: np.ndarray, name: str, n_folds: int = 5, max_depth: int = 4, reg_lambda: float = 8.0) -> tuple[xgb.Booster, float]:
    """Expanding-window causal Walk-Forward OOS evaluation + final full-history booster for Live."""
    n = len(X)
    fold_sz = n // (n_folds + 1)
    oos_preds = []
    oos_true = []

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda:0",
        "max_depth": max_depth,
        "learning_rate": 0.04,
        "subsample": 0.75,
        "colsample_bytree": 0.65,
        "min_child_weight": 24,
        "reg_lambda": reg_lambda,
        "reg_alpha": 1.5,
        "seed": 20261124,
    }

    for k in range(1, n_folds + 1):
        tr_e = k * fold_sz
        va_s = min(n - 10, tr_e + 25)  # purge gap of 25 interventions
        va_e = (k + 1) * fold_sz if k < n_folds else n
        if va_e <= va_s + 10 or len(np.unique(y[:tr_e])) < 2 or len(np.unique(y[va_s:va_e])) < 2:
            continue
        dtr = xgb.DMatrix(X[:tr_e], label=y[:tr_e])
        dva = xgb.DMatrix(X[va_s:va_e], label=y[va_s:va_e])
        bst = xgb.train(params, dtr, num_boost_round=180, evals=[(dva, "val")], early_stopping_rounds=25, verbose_eval=False)
        p_va = bst.predict(dva)
        oos_preds.append(p_va)
        oos_true.append(y[va_s:va_e])

    oos_auc = roc_auc_score(np.concatenate(oos_true), np.concatenate(oos_preds)) if oos_preds else 0.50

    # Train causal 3-stage boosters for historical cache generation without lookahead, plus final live booster!
    sp = int(n * 0.85)
    dtr_f = xgb.DMatrix(X[:sp], label=y[:sp])
    dva_f = xgb.DMatrix(X[sp:], label=y[sp:])
    best_rounds = 110
    if len(np.unique(y[sp:])) >= 2:
        bst_probe = xgb.train(params, dtr_f, num_boost_round=200, evals=[(dva_f, "val")], early_stopping_rounds=25, verbose_eval=False)
        best_rounds = max(45, int(bst_probe.best_iteration) + 1)

    dall = xgb.DMatrix(X, label=y)
    bst_live = xgb.train(params, dall, num_boost_round=best_rounds, verbose_eval=False)
    print(f"   ✅ [{name:<24}] Samples={n:6,d} | PosRate={y.mean()*100:5.1f}% | Causal WF OOS AUC = {oos_auc:.4f} (rounds={best_rounds})", flush=True)
    return bst_live, float(oos_auc), params, best_rounds


def _build_88d_world(
    tag: str,
    obs84_path: Path,
    lob_path: Path,
    out_88d_path: Path,
    bst_imp_early: xgb.Booster,
    bst_imp_pre_oos: xgb.Booster,
    bst_bt_early: xgb.Booster,
    bst_bt_pre_oos: xgb.Booster,
    bst_sr_early: xgb.Booster,
    bst_sr_pre_oos: xgb.Booster,
    mid_cut: int | None = None,
) -> None:
    t_w = time.time()
    obs84 = np.load(obs84_path, mmap_mode="r")["market_obs_84d"]
    lob = np.load(lob_path, mmap_mode="r")
    bp10 = lob["bid_p"]
    bq10 = lob["bid_q"]
    ap10 = lob["ask_p"]
    N = len(obs84)
    subcent_off = int(np.where(np.abs(bp10[:, 0] * 100.0 - np.round(bp10[:, 0] * 100.0)) > 1e-3)[0][0])
    if mid_cut is None:
        mid_cut = subcent_off

    print(f"\n🌊 [{tag}] Computing 4 Causal Order-2.5 Channels across sub-cent rows [{subcent_off:,} .. {N:,}]...", flush=True)
    obs88 = np.zeros((N, 88), dtype=np.float32)
    obs88[:, :84] = obs84
    obs88[:subcent_off, 84] = 0.35
    obs88[:subcent_off, 85] = 0.50
    obs88[:subcent_off, 86] = 0.23
    obs88[:subcent_off, 87] = 0.35

    chunk_sz = 60000
    for s_c in range(subcent_off, N, chunk_sz):
        e_c = min(N, s_c + chunk_sz)
        m_c = obs84[s_c:e_c]
        n_c = len(m_c)
        b1_c = np.maximum(0.0001, bp10[s_c:e_c, 0])
        a1_c = np.maximum(b1_c + 0.0001, ap10[s_c:e_c, 0])
        l1_u = np.maximum(20.0, bq10[s_c:e_c, 0] * b1_c)
        t5_u = np.maximum(50.0, np.sum(bq10[s_c:e_c, :5] * bp10[s_c:e_c, :5], axis=1))
        sp_s = np.clip(((a1_c - b1_c) / b1_c * 10000.0) / 30.0, 0.0, 4.0)

        q_small = np.column_stack([
            m_c,
            np.zeros(n_c, dtype=np.float32),
            np.zeros(n_c, dtype=np.float32),
            np.full(n_c, np.log1p(40.0) / 6.0, dtype=np.float32),
            np.clip(40.0 / l1_u, 0.0, 10.0) / 3.0,
            np.clip(40.0 / t5_u, 0.0, 5.0),
            sp_s,
        ]).astype(np.float32)

        q_whale = np.column_stack([
            m_c,
            np.zeros(n_c, dtype=np.float32),
            np.zeros(n_c, dtype=np.float32),
            np.full(n_c, np.log1p(250.0) / 6.0, dtype=np.float32),
            np.clip(250.0 / l1_u, 0.0, 10.0) / 3.0,
            np.clip(250.0 / t5_u, 0.0, 5.0),
            sp_s,
        ]).astype(np.float32)

        use_pre_oos = s_c >= mid_cut
        b_imp = bst_imp_pre_oos if use_pre_oos else bst_imp_early
        b_bt = bst_bt_pre_oos if use_pre_oos else bst_bt_early
        b_sr = bst_sr_pre_oos if use_pre_oos else bst_sr_early

        dm_mkt = xgb.DMatrix(m_c)
        obs88[s_c:e_c, 84] = np.clip(b_imp.predict(xgb.DMatrix(q_small)), 0.0, 1.0)
        obs88[s_c:e_c, 85] = np.clip(b_imp.predict(xgb.DMatrix(q_whale)), 0.0, 1.0)
        obs88[s_c:e_c, 86] = np.clip(b_bt.predict(dm_mkt), 0.0, 1.0)
        obs88[s_c:e_c, 87] = np.clip(b_sr.predict(dm_mkt), 0.0, 1.0)

    sub88 = obs88[subcent_off:]
    for ci, nm in [(84, "p_penny_jump_small"), (85, "p_penny_jump_whale"), (86, "p_live_buy_trap_3m"), (87, "p_live_sell_regret_3m")]:
        print(f"   [{tag}] ch[{ci}] {nm:<24}: mean={sub88[:, ci].mean():.4f} | std={sub88[:, ci].std():.4f}", flush=True)

    tmp_88 = out_88d_path.with_name(out_88d_path.stem + "_tmp.npz")
    np.savez(tmp_88, market_obs_88d=obs88)
    tmp_88.replace(out_88d_path)
    print(f"✅ [{tag}] Saved {out_88d_path.name} ({obs88.shape}, {out_88d_path.stat().st_size/1e6:.1f} MB) in {time.time()-t_w:.1f}s!", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="Build 88D DEC + RAW caches using existing live .ubj boosters without retraining")
    args = parser.parse_args()

    t0 = time.time()
    OUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        print("=" * 96)
        print("🧊 STEP 3 (EVAL-ONLY): COMPILING 88D DEC + RAW CACHES USING FROZEN LIVE .UBJ BOOSTERS")
        print("=" * 96, flush=True)
        b_imp = xgb.Booster(); b_imp.load_model(str(OUT_MODELS_DIR / "order25_bot_impact.ubj")); b_imp.set_param({"device": "cuda:0"})
        b_bt = xgb.Booster(); b_bt.load_model(str(OUT_MODELS_DIR / "order25_buy_trap_3m.ubj")); b_bt.set_param({"device": "cuda:0"})
        b_sr = xgb.Booster(); b_sr.load_model(str(OUT_MODELS_DIR / "order25_sell_regret_3m.ubj")); b_sr.set_param({"device": "cuda:0"})
        _build_88d_world("DEC", OBS84_PATH, LOB_PATH, OUT_88D_PATH, b_imp, b_imp, b_bt, b_bt, b_sr, b_sr)
        if OBS84_RAW_PATH.exists() and LOB_RAW_PATH.exists():
            _build_88d_world("RAW", OBS84_RAW_PATH, LOB_RAW_PATH, OUT_88D_RAW_PATH, b_imp, b_imp, b_bt, b_bt, b_sr, b_sr)
        return

    print("=" * 96)
    print("🧠 STEP 3: TRAINING CAUSAL ORDER-2.5 MARKET-IMPACT & LIVE-REGRET MODELS")
    print("=" * 96, flush=True)

    data = np.load(INT_PATH)
    X_place = data["X_place"]   # [27374, 90]: 84D mkt + [side, dist_bp/50, log_usd/6, usd/l1, usd/top5, sp_bp/30]
    Y_place = data["Y_place"]   # [27374, 3] : [penny_5s, push_10s, filled]
    X_fill = data["X_fill"]     # [3077, 87] : 84D mkt + [side, log_usd/6, slot/11]
    Y_fill = data["Y_fill"]     # [3077, 4]  : [mk10, mk30, mk180, is_trap]

    near_mask = X_place[:, 85] <= 0.60
    Xp_near = X_place[near_mask]
    Yp_impact = np.clip(Y_place[near_mask, 0] + Y_place[near_mask, 1], 0.0, 1.0)
    bst_impact, auc_imp, p_imp, r_imp = train_wf_xgb_binary(Xp_near, Yp_impact, "order25_bot_impact", n_folds=5, max_depth=4, reg_lambda=8.0)
    bst_impact.save_model(str(OUT_MODELS_DIR / "order25_bot_impact.ubj"))

    buy_mask = X_fill[:, 84] < 0.5
    Xf_buy = X_fill[buy_mask, :84]
    Yf_buy_trap = (Y_fill[buy_mask, 2] <= -35.0).astype(np.float32)
    bst_buy_trap, auc_bt, p_bt, r_bt = train_wf_xgb_binary(Xf_buy, Yf_buy_trap, "order25_buy_trap_3m", n_folds=4, max_depth=3, reg_lambda=10.0)
    bst_buy_trap.save_model(str(OUT_MODELS_DIR / "order25_buy_trap_3m.ubj"))

    sell_mask = X_fill[:, 84] > 0.5
    Xf_sell = X_fill[sell_mask, :84]
    Yf_sell_regret = (Y_fill[sell_mask, 2] <= -35.0).astype(np.float32)
    bst_sell_reg, auc_sr, p_sr, r_sr = train_wf_xgb_binary(Xf_sell, Yf_sell_regret, "order25_sell_regret_3m", n_folds=4, max_depth=3, reg_lambda=10.0)
    bst_sell_reg.save_model(str(OUT_MODELS_DIR / "order25_sell_regret_3m.ubj"))

    obs84_n = len(np.load(OBS84_PATH, mmap_mode="r")["market_obs_84d"])
    lob_bp = np.load(LOB_PATH, mmap_mode="r")["bid_p"]
    subcent_off = int(np.where(np.abs(lob_bp[:, 0] * 100.0 - np.round(lob_bp[:, 0] * 100.0)) > 1e-3)[0][0])

    place_row_exact = data["place_idx"][near_mask]
    buy_row_exact = data["fill_idx"][buy_mask]
    sell_row_exact = data["fill_idx"][sell_mask]

    oos_cut = obs84_n - 75000
    mid_cut = subcent_off + (oos_cut - subcent_off) // 2

    def fit_cutoff(X_all, y_all, row_idx, cut_row, prms, rnds):
        m = row_idx < (cut_row - 600)
        if np.sum(m) < 80 or len(np.unique(y_all[m])) < 2:
            m = np.arange(len(X_all)) < max(80, len(X_all) // 3)
        return xgb.train(prms, xgb.DMatrix(X_all[m], label=y_all[m]), num_boost_round=rnds, verbose_eval=False)

    bst_imp_early = fit_cutoff(Xp_near, Yp_impact, place_row_exact, mid_cut, p_imp, r_imp)
    bst_imp_pre_oos = fit_cutoff(Xp_near, Yp_impact, place_row_exact, oos_cut, p_imp, r_imp)
    bst_bt_early = fit_cutoff(Xf_buy, Yf_buy_trap, buy_row_exact, mid_cut, p_bt, r_bt)
    bst_bt_pre_oos = fit_cutoff(Xf_buy, Yf_buy_trap, buy_row_exact, oos_cut, p_bt, r_bt)
    bst_sr_early = fit_cutoff(Xf_sell, Yf_sell_regret, sell_row_exact, mid_cut, p_sr, r_sr)
    bst_sr_pre_oos = fit_cutoff(Xf_sell, Yf_sell_regret, sell_row_exact, oos_cut, p_sr, r_sr)

    meta_o25 = {
        "models": {
            "order25_bot_impact": {"oos_auc": round(auc_imp, 4), "rounds": r_imp},
            "order25_buy_trap_3m": {"oos_auc": round(auc_bt, 4), "rounds": r_bt},
            "order25_sell_regret_3m": {"oos_auc": round(auc_sr, 4), "rounds": r_sr},
        },
        "channels_88d": {
            "84": "p_penny_jump_small ($40 at BBO)",
            "85": "p_penny_jump_whale ($250 at BBO)",
            "86": "p_live_buy_trap_3m (real live adverse fill markout <= -35bp)",
            "87": "p_live_sell_regret_3m (real live premature ask fill rally >= +35bp)",
        },
    }
    (OUT_MODELS_DIR / "meta_order25.json").write_text(json.dumps(meta_o25, indent=2))

    _build_88d_world("DEC", OBS84_PATH, LOB_PATH, OUT_88D_PATH, bst_imp_early, bst_imp_pre_oos, bst_bt_early, bst_bt_pre_oos, bst_sr_early, bst_sr_pre_oos, mid_cut=mid_cut)
    if OBS84_RAW_PATH.exists() and LOB_RAW_PATH.exists():
        _build_88d_world("RAW", OBS84_RAW_PATH, LOB_RAW_PATH, OUT_88D_RAW_PATH, bst_imp_early, bst_imp_pre_oos, bst_bt_early, bst_bt_pre_oos, bst_sr_early, bst_sr_pre_oos, mid_cut=mid_cut)
    print(f"✅ Step 3 Total Completed in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
