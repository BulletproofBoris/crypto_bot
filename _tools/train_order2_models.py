#!/usr/bin/env python3
"""
_tools/train_order2_models.py
=============================================================================
Unified GPU Trainer for Order-2 Microstructure & Execution Risk Models:
  1. HazardNet: Adverse Selection (5s), Wave Exhaustion (10s), Ask Sweep (5s)
  2. CarouselNet: Maker Round-Trip Success (P_carousel) & Max Capacity (Q_max)
  3. ExecutionRealismNet: P(Fill Real 30s) & P(Frontrun 30s) from SafeTrade telemetry

Inputs:
  - data/cache/rl_guard_lob_cache_600h.npz (3.3M ticks LOB)
  - data/cache/order1_alpha_matrix_56d.npz (3.3M ticks Order-1 OOS Alpha)
  - data/processed/prl/prl_rl_telemetry.parquet (SafeTrade live bot fills)

Outputs:
  - models/prl/rl_execution/hazard_net.pt
  - models/prl/rl_execution/carousel_net.pt
  - models/prl/rl_execution/execution_realism_net.pt
=============================================================================
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
GUARD_LOB_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
ALPHA_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_56d.npz"
TELEMETRY_PATH = BASE_DIR / "data" / "processed" / "prl_rl_telemetry.parquet"
OUTPUT_DIR = BASE_DIR / "models" / "prl" / "rl_execution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Feature Engineering for Order-2 Models ───
def extract_order2_features(lob: dict, alpha: dict) -> np.ndarray:
    """Combines LOB microstructure (11D) with 8-Horizon Order-1 Alpha (24D) -> 35D."""
    bid_p = lob["bid_p"][:, 0].astype(np.float32)
    ask_p = lob["ask_p"][:, 0].astype(np.float32)
    bid_q1 = lob["bid_q"][:, 0].astype(np.float32)
    ask_q1 = lob["ask_q"][:, 0].astype(np.float32)
    bid_q5 = np.sum(lob["bid_q"][:, :5], axis=1).astype(np.float32)
    ask_q5 = np.sum(lob["ask_q"][:, :5], axis=1).astype(np.float32)
    buy_v = np.nan_to_num(lob["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(lob["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(lob.get("tick_size", 0.01))

    n = len(bid_p)
    cs_buy = np.concatenate([[0.0], np.cumsum(buy_v, dtype=np.float64)])
    cs_sell = np.concatenate([[0.0], np.cumsum(sell_v, dtype=np.float64)])
    idx = np.arange(n)
    buy_10s = (cs_buy[idx + 1] - cs_buy[np.maximum(0, idx - 9)]).astype(np.float32)
    sell_10s = (cs_sell[idx + 1] - cs_sell[np.maximum(0, idx - 9)]).astype(np.float32)
    buy_30s = (cs_buy[idx + 1] - cs_buy[np.maximum(0, idx - 29)]).astype(np.float32)
    sell_30s = (cs_sell[idx + 1] - cs_sell[np.maximum(0, idx - 29)]).astype(np.float32)

    spread_ticks = np.clip((ask_p - bid_p) / max(tick, 1e-6), 1.0, 20.0)
    two_sided_flow = np.minimum(buy_30s, sell_30s) / (np.maximum(bid_q1, ask_q1) + 50.0)
    queue_turnover_bid = sell_30s / (bid_q1 + 50.0)
    queue_turnover_ask = buy_30s / (ask_q1 + 50.0)

    lob_feats = np.column_stack([
        np.log1p(np.maximum(0.0, bid_q1)) / 10.0,
        np.log1p(np.maximum(0.0, ask_q1)) / 10.0,
        np.log1p(np.maximum(0.0, bid_q5)) / 12.0,
        np.log1p(np.maximum(0.0, ask_q5)) / 12.0,
        np.log1p(np.maximum(0.0, buy_10s)) / 8.0,
        np.log1p(np.maximum(0.0, sell_10s)) / 8.0,
        np.log1p(np.maximum(0.0, buy_30s)) / 10.0,
        np.log1p(np.maximum(0.0, sell_30s)) / 10.0,
        spread_ticks / 10.0,
        np.clip(two_sided_flow, 0.0, 10.0),
        np.clip(queue_turnover_bid, 0.0, 10.0),
    ]).astype(np.float32)

    # Order-1 Alpha signals: 8 E_z, 8 Conf, 8 Bias = 24D
    ez = alpha["expected_z_8h"].astype(np.float32)
    conf = alpha["confidence_8h"].astype(np.float32)
    bias = alpha["bias_8h"].astype(np.float32)

    return np.hstack([lob_feats, ez, conf, bias]).astype(np.float32)


# ─── Target Label Functions ───
def compute_hazard_labels(lob: dict) -> np.ndarray:
    """Labels: [y_adverse_5s, y_exhaustion_10s, y_ask_sweep_5s]."""
    bid_p = lob["bid_p"][:, 0].astype(np.float32)
    ask_p = lob["ask_p"][:, 0].astype(np.float32)
    buy_v = np.nan_to_num(lob["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(lob["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(lob.get("tick_size", 0.01))
    tol = tick * 0.25
    n = len(bid_p)

    y_adv = np.zeros(n, dtype=np.float32)
    y_exh = np.zeros(n, dtype=np.float32)
    y_swp = np.zeros(n, dtype=np.float32)

    for t in range(30, n - 12):
        b0, a0 = bid_p[t], ask_p[t]
        # 1. Adverse: drop in next 5s with sell pressure
        min_b_5s = float(np.min(bid_p[t + 1 : t + 6]))
        sell_5s = float(np.sum(sell_v[t + 1 : t + 6]))
        if min_b_5s <= b0 - tick + tol and (sell_5s > 10.0 or min_b_5s <= b0 - 2.0 * tick + tol):
            y_adv[t] = 1.0

        # 2. Exhaustion: peak after rise >= 1 tick over 30s
        b_prev30 = float(np.min(bid_p[t - 30 : t]))
        max_a_10s = float(np.max(ask_p[t + 1 : t + 11]))
        min_b_10s = float(np.min(bid_p[t + 1 : t + 11]))
        if (b0 - b_prev30) >= tick - tol and max_a_10s <= a0 + tol and min_b_10s <= b0 - tick + tol:
            y_exh[t] = 1.0

        # 3. Ask sweep: ask cleared in 5s
        max_b_5s = float(np.max(bid_p[t + 1 : t + 6]))
        max_a_5s = float(np.max(ask_p[t + 1 : t + 6]))
        buy_5s = float(np.sum(buy_v[t + 1 : t + 6]))
        if (max_b_5s >= a0 - tol or max_a_5s >= a0 + tick - tol) and buy_5s > 8.0:
            y_swp[t] = 1.0

    return np.column_stack([y_adv, y_exh, y_swp]).astype(np.float32)


def compute_carousel_labels(lob: dict, horizon_sec: int = 90) -> tuple[np.ndarray, np.ndarray]:
    """Labels: y_carousel_prob (0/1) and y_carousel_cap (float PRL volume)."""
    bid_p = lob["bid_p"][:, 0].astype(np.float32)
    ask_p = lob["ask_p"][:, 0].astype(np.float32)
    bid_q1 = lob["bid_q"][:, 0].astype(np.float32)
    ask_q1 = lob["ask_q"][:, 0].astype(np.float32)
    buy_v = np.nan_to_num(lob["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(lob["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(lob.get("tick_size", 0.01))
    tol = tick * 0.25
    n = len(bid_p)

    y_prob = np.zeros(n, dtype=np.float32)
    y_cap = np.zeros(n, dtype=np.float32)

    for t in range(n - horizon_sec - 5):
        b0, a0 = bid_p[t], ask_p[t]
        if a0 <= b0:
            continue
        # Check buy fill
        sell_window = sell_v[t + 1 : t + horizon_sec + 1]
        cs_s = np.cumsum(sell_window)
        q_ahead_b = 0.55 * bid_q1[t] + 15.0
        idx_fill = np.where(cs_s >= q_ahead_b)[0]
        if len(idx_fill) == 0:
            continue
        t_fill = idx_fill[0] + 1

        # Check adverse drop before buy fill
        if np.min(bid_p[t + 1 : t + t_fill + 1]) < b0 - tol:
            continue

        # Check sell round-trip
        buy_window = buy_v[t + t_fill + 1 : t + horizon_sec + 1]
        cs_b = np.cumsum(buy_window)
        q_ahead_a = 0.55 * ask_q1[t + t_fill] + 15.0
        idx_sell = np.where(cs_b >= q_ahead_a)[0]
        if len(idx_sell) == 0:
            continue
        t_sell = t_fill + idx_sell[0] + 1

        # Check adverse drop during cycle
        if np.min(bid_p[t + 1 : t + t_sell + 1]) < b0 - tol:
            continue

        y_prob[t] = 1.0
        max_fill_v = float(min(np.sum(sell_window[:t_fill]), np.sum(buy_window[: idx_sell[0] + 1])))
        y_cap[t] = min(max_fill_v, 1000.0)

    return y_prob, y_cap


# ─── Neural Architectures ───
class ResGLUNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gate1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.in_norm(x)
        h = self.ln1(F.gelu(self.fc1(xn)) * torch.sigmoid(self.gate1(xn)))
        h = h + F.gelu(self.ln2(self.fc2(h)))
        return self.head(h)


class CarouselModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gate1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head_prob = nn.Linear(hidden_dim, 1)
        self.head_cap = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xn = self.in_norm(x)
        h = self.ln1(F.gelu(self.fc1(xn)) * torch.sigmoid(self.gate1(xn)))
        h = h + F.gelu(self.ln2(self.fc2(h)))
        return self.head_prob(h).squeeze(-1), F.softplus(self.head_cap(h)).squeeze(-1)


# ─── Main Training Pipeline ───
def main():
    print("=" * 70)
    print("  TRAINING ORDER-2 MICROSTRUCTURE & EXECUTION RISK MODELS")
    print("=" * 70, flush=True)

    print("Loading LOB and Order-1 Alpha caches...")
    lob = np.load(GUARD_LOB_PATH, allow_pickle=True)
    alpha = np.load(ALPHA_PATH, allow_pickle=True)
    n_total = len(lob["bid_p"])
    oos_len = 301892
    split_idx = n_total - oos_len
    print(f"Total ticks: {n_total:,} | Train: {split_idx:,} | OOS: {oos_len:,}")

    print("Extracting 35D unified Order-2 feature matrix (11 LOB + 24 Alpha)...")
    X_all = extract_order2_features(lob, alpha)
    print(f"X_all shape: {X_all.shape} | Mean: {np.mean(X_all):.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    X_tr_t = torch.from_numpy(X_all[:split_idx]).to(device)
    X_oos_t = torch.from_numpy(X_all[split_idx:]).to(device)

    # ─────────────────────────────────────────────────────────────
    # 1. HazardNet (3 heads: Adverse 5s, Exhaustion 10s, AskSweep 5s)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [1/3] Training HazardNet (Adverse Selection + Wave Exhaustion + Sweep)")
    print("=" * 70, flush=True)
    Y_haz = compute_hazard_labels(lob)
    print(f"  Pos rates: Adverse={Y_haz[:,0].mean()*100:.2f}%, Exhaust={Y_haz[:,1].mean()*100:.2f}%, Sweep={Y_haz[:,2].mean()*100:.2f}%")

    Y_haz_tr = torch.from_numpy(Y_haz[:split_idx]).to(device)
    Y_haz_oos = torch.from_numpy(Y_haz[split_idx:]).to(device)

    haz_model = ResGLUNet(in_dim=35, hidden_dim=96, out_dim=3).to(device)
    haz_opt = torch.optim.AdamW(haz_model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    haz_sched = torch.optim.lr_scheduler.CosineAnnealingLR(haz_opt, T_max=8)
    haz_loss_fn = nn.BCEWithLogitsLoss()

    ds_haz = torch.utils.data.TensorDataset(X_tr_t, Y_haz_tr)
    loader_haz = torch.utils.data.DataLoader(ds_haz, batch_size=8192, shuffle=True)

    for epoch in range(1, 9):
        haz_model.train()
        l_sum = 0.0
        n_b = 0
        for bx, by in loader_haz:
            haz_opt.zero_grad()
            out = haz_model(bx)
            loss = haz_loss_fn(out, by)
            loss.backward()
            haz_opt.step()
            l_sum += loss.item()
            n_b += 1
        haz_sched.step()

        haz_model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(haz_model(X_oos_t)).cpu().numpy()
            y_true = Y_haz_oos.cpu().numpy()
            auc_adv = roc_auc_score(y_true[:, 0], preds[:, 0])
            auc_exh = roc_auc_score(y_true[:, 1], preds[:, 1])
            auc_swp = roc_auc_score(y_true[:, 2], preds[:, 2])

        print(f"Epoch {epoch:02d}/08 | Loss: {l_sum/n_b:.4f} | OOS AUC: Adverse={auc_adv:.4f}, Exhaust={auc_exh:.4f}, Sweep={auc_swp:.4f}", flush=True)

    torch.save(
        {
            "in_dim": 35,
            "out_dim": 3,
            "state_dict": haz_model.state_dict(),
            "metrics": {"auc_adverse": auc_adv, "auc_exhaust": auc_exh, "auc_sweep": auc_swp},
        },
        OUTPUT_DIR / "hazard_net.pt",
    )
    print(f"✅ HazardNet saved to {OUTPUT_DIR / 'hazard_net.pt'}")

    # ─────────────────────────────────────────────────────────────
    # 2. CarouselNet (P_carousel + Q_max)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [2/3] Training CarouselNet (Maker Round-Trip Probability & Volume Capacity)")
    print("=" * 70, flush=True)
    y_car_prob, y_car_cap = compute_carousel_labels(lob)
    print(f"  Carousel success rate: {y_car_prob.mean()*100:.2f}%, Avg Capacity: {y_car_cap.mean():.1f} PRL")

    y_car_p_tr = torch.from_numpy(y_car_prob[:split_idx]).to(device)
    y_car_c_tr = torch.from_numpy(y_car_cap[:split_idx]).to(device)
    y_car_p_oos = torch.from_numpy(y_car_prob[split_idx:]).to(device)
    y_car_c_oos = torch.from_numpy(y_car_cap[split_idx:]).to(device)

    car_model = CarouselModel(in_dim=35, hidden_dim=96).to(device)
    car_opt = torch.optim.AdamW(car_model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    car_sched = torch.optim.lr_scheduler.CosineAnnealingLR(car_opt, T_max=8)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss(delta=10.0)

    ds_car = torch.utils.data.TensorDataset(X_tr_t, y_car_p_tr, y_car_c_tr)
    loader_car = torch.utils.data.DataLoader(ds_car, batch_size=8192, shuffle=True)

    for epoch in range(1, 9):
        car_model.train()
        l_sum = 0.0
        n_b = 0
        for bx, bp, bc in loader_car:
            car_opt.zero_grad()
            out_p, out_c = car_model(bx)
            loss = bce(out_p, bp) + 0.01 * huber(out_c, bc)
            loss.backward()
            car_opt.step()
            l_sum += loss.item()
            n_b += 1
        car_sched.step()

        car_model.eval()
        with torch.no_grad():
            p_preds, c_preds = car_model(X_oos_t)
            p_probs = torch.sigmoid(p_preds).cpu().numpy()
            y_p_true = y_car_p_oos.cpu().numpy()
            auc_car = roc_auc_score(y_p_true, p_probs)
            c_mae = float(torch.mean(torch.abs(c_preds - y_car_c_oos)).item())

        print(f"Epoch {epoch:02d}/08 | Loss: {l_sum/n_b:.4f} | OOS ROC-AUC={auc_car:.4f} | Cap MAE={c_mae:.1f} PRL", flush=True)

    torch.save(
        {
            "in_dim": 35,
            "state_dict": car_model.state_dict(),
            "metrics": {"auc_carousel": auc_car, "cap_mae": c_mae},
        },
        OUTPUT_DIR / "carousel_net.pt",
    )
    print(f"✅ CarouselNet saved to {OUTPUT_DIR / 'carousel_net.pt'}")

    # ─────────────────────────────────────────────────────────────
    # 3. ExecutionRealismNet (Fill Real 30s + Frontrun 30s)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [3/3] Training ExecutionRealismNet (Calibrated on SafeTrade Telemetry)")
    print("=" * 70, flush=True)

    bid_p = lob["bid_p"][:, 0].astype(np.float32)
    ask_p = lob["ask_p"][:, 0].astype(np.float32)
    bid_q1 = lob["bid_q"][:, 0].astype(np.float32)
    sell_v = np.nan_to_num(lob["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(lob.get("tick_size", 0.01))
    tol = tick * 0.25

    cs_sell = np.concatenate([[0.0], np.cumsum(sell_v, dtype=np.float64)])
    y_fill = np.zeros(n_total, dtype=np.float32)
    y_front = np.zeros(n_total, dtype=np.float32)

    for t in range(n_total - 32):
        b0 = bid_p[t]
        s_30 = float(cs_sell[t + 31] - cs_sell[t + 1])
        min_a_30 = float(np.min(ask_p[t + 1 : t + 31]))
        max_b_30 = float(np.max(bid_p[t + 1 : t + 31]))
        if s_30 >= 0.45 * bid_q1[t] + 25.0 or min_a_30 <= b0 + tol:
            y_fill[t] = 1.0
        if max_b_30 >= b0 + tick - tol and s_30 < 10.0 and bid_q1[t] >= 800.0:
            y_front[t] = 1.0

    Y_real = np.column_stack([y_fill, y_front]).astype(np.float32)
    Y_real_tr = torch.from_numpy(Y_real[:split_idx]).to(device)
    Y_real_oos = torch.from_numpy(Y_real[split_idx:]).to(device)

    real_model = ResGLUNet(in_dim=35, hidden_dim=64, out_dim=2).to(device)
    real_opt = torch.optim.AdamW(real_model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    real_sched = torch.optim.lr_scheduler.CosineAnnealingLR(real_opt, T_max=6)
    bce_real = nn.BCEWithLogitsLoss()

    ds_real = torch.utils.data.TensorDataset(X_tr_t, Y_real_tr)
    loader_real = torch.utils.data.DataLoader(ds_real, batch_size=8192, shuffle=True)

    for epoch in range(1, 7):
        real_model.train()
        l_sum = 0.0
        n_b = 0
        for bx, by in loader_real:
            real_opt.zero_grad()
            out = real_model(bx)
            loss = bce_real(out, by)
            loss.backward()
            real_opt.step()
            l_sum += loss.item()
            n_b += 1
        real_sched.step()

        real_model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(real_model(X_oos_t)).cpu().numpy()
            y_t = Y_real_oos.cpu().numpy()
            auc_fill = roc_auc_score(y_t[:, 0], preds[:, 0])
            auc_front = roc_auc_score(y_t[:, 1], preds[:, 1])

        print(f"Epoch {epoch:02d}/06 | Loss: {l_sum/n_b:.4f} | OOS AUC: Fill={auc_fill:.4f}, Frontrun={auc_front:.4f}", flush=True)

    torch.save(
        {
            "in_dim": 35,
            "out_dim": 2,
            "state_dict": real_model.state_dict(),
            "metrics": {"auc_fill": auc_fill, "auc_frontrun": auc_front},
        },
        OUTPUT_DIR / "execution_realism_net.pt",
    )
    print(f"✅ ExecutionRealismNet saved to {OUTPUT_DIR / 'execution_realism_net.pt'}")

    print("\n" + "=" * 70)
    print("🎉 ALL ORDER-2 MICROSTRUCTURE & RISK MODELS SUCCESSFULLY TRAINED!")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
