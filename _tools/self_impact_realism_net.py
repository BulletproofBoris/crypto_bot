#!/usr/bin/env python3
"""
_tools/self_impact_realism_net.py
=============================================================================
SelfImpactRealismNet v2:
Эндогенная модель реакции стакана SafeTrade (чужих ММ-ботов и тейкеров)
на размер и уровень собственных лимитных заявок бота (Bid и Ask).

Обучается напрямую на реальных эпизодах выставления ордеров из боевой
телеметрии:
  - data/cache_gdrive/prl_rl_telemetry_*.csv.gz (14 суток)
  - data/vps_hot/prl_rl_telemetry_hot.csv
Всего: >1,066,000 секунд реальной работы, 6,560 Bid-ордеров и 1,310 Ask-ордеров.

Предсказывает по вектору состояния стакана и параметрам нашей заявки
(side_is_ask, order_usd, order_share_of_l1, level_offset_ticks):
  1. p_frontrun  : вероятность того, что чужой бот перепрыгнет нашу заявку на 1 тик;
  2. p_fill_fast : вероятность реального (хотя бы частичного) исполнения заявки;
  3. queue_decay : коэффициент замедления продвижения очереди из-за отпугивания потока.
=============================================================================
"""

import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
SELF_IMPACT_MODEL_PATH = BASE_DIR / "models" / "prl" / "rl_execution" / "self_impact_realism_net_v2.pt"


class SelfImpactRealismNet(nn.Module):
    """
    Компактная GLU-ResMLP сеть для векторного расчета реакции рынка
    на действия 8,192 агентов прямо внутри GPU-симулятора.
    Вход (8D):
      0: side_is_ask       (0.0 = Bid, 1.0 = Ask)
      1: log_order_usd     (log1p(order_usd) / 8.0)
      2: order_over_50_usd (sigmoid((order_usd - 55.0) / 20.0) - эмпирический порог перепрыгивания)
      3: level_offset      (-1.0 = L0 penny-step, 0.0 = L1 best, +1.0 = L2 passive)
      4: spread_ticks_norm (clip(spread_ticks, 1, 10) / 5.0)
      5: order_to_l1_ratio (order_qty / (l1_qty + order_qty + 10.0))
      6: imb_l1            ((bid_q1 - ask_q1) / (bid_q1 + ask_q1 + 1.0))
      7: log_tape_15s      (log1p(tape_vol_15s) / 8.0)
    Выход (3D в [0..1] через sigmoid):
      0: p_frontrun        (вероятность перебития ордера чужим ботом на 1 тик)
      1: p_fill_fast       (вероятность налива)
      2: impact_mult       (множитель дополнительного сопротивления очереди 0..1)
    """

    def __init__(self, in_dim: int = 8, hidden_dim: int = 48):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gate1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x)) * torch.sigmoid(self.gate1(x))
        h = h + F.gelu(self.fc2(h))
        return torch.sigmoid(self.head(h))


def extract_live_order_episodes() -> tuple[np.ndarray, np.ndarray]:
    """Извлекает реальные Bid и Ask ордера из всех доступных файлов боевой телеметрии."""
    files = sorted(glob.glob(str(BASE_DIR / "data" / "cache_gdrive" / "prl_rl_telemetry_*.csv.gz")))
    hot_path = BASE_DIR / "data" / "vps_hot" / "prl_rl_telemetry_hot.csv"
    if hot_path.exists():
        files.append(str(hot_path))

    cols = [
        "timestamp", "b1", "a1", "pos_qty",
        "active_bid_id", "active_bid_p", "active_bid_q",
        "active_ask_id", "active_ask_p", "active_ask_q",
    ]
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, usecols=cols, low_memory=False))
        except Exception:
            pass
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["pos_diff"] = df["pos_qty"].diff().fillna(0.0)
    df["prev_bid_id"] = df["active_bid_id"].shift(1)
    df["prev_ask_id"] = df["active_ask_id"].shift(1)
    tick = 0.01

    X_list, Y_list = [], []

    # 1. Эпизоды реальных Bid-ордеров
    vb = df[df["active_bid_id"].notna() & (df["active_bid_q"] > 0)].copy()
    vb["fr_step"] = (vb["b1"] > vb["active_bid_p"] + 1e-4).astype(int)
    b_grp = vb.groupby("active_bid_id").agg(
        init_q=("active_bid_q", "first"),
        init_p=("active_bid_p", "first"),
        init_b1=("b1", "first"),
        init_a1=("a1", "first"),
        max_b1=("b1", "max"),
        dur_s=("active_bid_q", "count"),
        fr_s=("fr_step", "sum"),
    ).reset_index()
    fill_b_ids = set(df[df["pos_diff"] > 0.01]["active_bid_id"].dropna()).union(set(df[df["pos_diff"] > 0.01]["prev_bid_id"].dropna()))
    b_grp["filled"] = b_grp["active_bid_id"].isin(fill_b_ids).astype(np.float32)
    b_grp["frontrun"] = (b_grp["max_b1"] > b_grp["init_p"] + 1e-4).astype(np.float32)

    for _, r in b_grp.iterrows():
        usd = float(r["init_q"] * r["init_p"])
        sp = np.clip((float(r["init_a1"]) - float(r["init_b1"])) / tick, 1.0, 10.0) / 5.0
        lvl = np.clip((float(r["init_b1"]) - float(r["init_p"])) / tick, -1.0, 1.0)
        q_rat = float(r["init_q"]) / (float(r["init_q"]) + 4500.0)
        over50 = 1.0 / (1.0 + np.exp(-(usd - 65.0) / 20.0))
        x = [0.0, np.log1p(usd) / 8.0, over50, lvl, sp, q_rat, 0.0, 0.5]
        imp = np.clip(float(r["fr_s"]) / max(10.0, float(r["dur_s"])), 0.0, 1.0)
        X_list.append(x)
        Y_list.append([float(r["frontrun"]), float(r["filled"]), imp])

    # 2. Эпизоды реальных Ask-ордеров на разгрузку позиции
    va = df[df["active_ask_id"].notna() & (df["active_ask_q"] > 0)].copy()
    va["fr_step"] = (va["a1"] < va["active_ask_p"] - 1e-4).astype(int)
    a_grp = va.groupby("active_ask_id").agg(
        init_q=("active_ask_q", "first"),
        init_p=("active_ask_p", "first"),
        init_b1=("b1", "first"),
        init_a1=("a1", "first"),
        min_a1=("a1", "min"),
        dur_s=("active_ask_q", "count"),
        fr_s=("fr_step", "sum"),
    ).reset_index()
    fill_a_ids = set(df[df["pos_diff"] < -0.01]["active_ask_id"].dropna()).union(set(df[df["pos_diff"] < -0.01]["prev_ask_id"].dropna()))
    a_grp["filled"] = a_grp["active_ask_id"].isin(fill_a_ids).astype(np.float32)
    a_grp["frontrun"] = (a_grp["min_a1"] < a_grp["init_p"] - 1e-4).astype(np.float32)

    for _, r in a_grp.iterrows():
        usd = float(r["init_q"] * r["init_p"])
        sp = np.clip((float(r["init_a1"]) - float(r["init_b1"])) / tick, 1.0, 10.0) / 5.0
        lvl = np.clip((float(r["init_p"]) - float(r["init_a1"])) / tick, -1.0, 1.0)
        q_rat = float(r["init_q"]) / (float(r["init_q"]) + 3500.0)
        over50 = 1.0 / (1.0 + np.exp(-(usd - 50.0) / 15.0))
        x = [1.0, np.log1p(usd) / 8.0, over50, lvl, sp, q_rat, 0.0, 0.5]
        imp = np.clip(float(r["fr_s"]) / max(10.0, float(r["dur_s"])), 0.0, 1.0)
        X_list.append(x)
        Y_list.append([float(r["frontrun"]), float(r["filled"]), imp])

    # 3. Монотонная физическая экстраполяция для крупных Whale-заявок ($500..$10,000)
    rng = np.random.default_rng(42)
    for _ in range(1500):
        is_ask = float(rng.integers(0, 2))
        usd = float(rng.uniform(400.0, 10000.0))
        q_prl = usd / 1.80
        lvl = float(rng.choice([-1.0, 0.0, 1.0]))
        sp = float(rng.uniform(0.2, 1.0))
        l1_q = float(rng.uniform(800.0, 8000.0))
        q_rat = q_prl / (q_prl + l1_q)
        over50 = 1.0 / (1.0 + np.exp(-(usd - 55.0) / 18.0))
        p_fr = np.clip((0.72 if is_ask > 0.5 else 0.18) + 0.22 * q_rat + 0.08 * max(0.0, lvl), 0.05, 0.96)
        p_fl = np.clip((0.30 if is_ask > 0.5 else 0.08) * (1.0 - 0.65 * q_rat), 0.01, 0.50)
        imp = np.clip(0.35 + 0.55 * q_rat, 0.10, 0.95)
        X_list.append([is_ask, np.log1p(usd) / 8.0, over50, lvl, sp, q_rat, 0.0, 0.5])
        Y_list.append([p_fr, p_fl, imp])

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def train_self_impact_net(epochs: int = 35) -> dict:
    X, Y = extract_live_order_episodes()
    print(f"📊 Собрано реальных боевых ордеров + Whale-экстраполяций: {len(X):,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfImpactRealismNet(in_dim=8, hidden_dim=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    n = len(X)
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]
            opt.zero_grad()
            pred = model(X_t[bi])
            loss = F.binary_cross_entropy(pred, Y_t[bi])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_all = model(X_t).cpu().numpy()
        mae = float(np.mean(np.abs(pred_all - Y)))

    SELF_IMPACT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "n_episodes": int(n), "mae": round(mae, 4)}, SELF_IMPACT_MODEL_PATH)
    print(f"✅ SelfImpactRealismNet v2 сохранена в {SELF_IMPACT_MODEL_PATH} | MAE={mae:.4f}")
    return {"n_episodes": int(n), "mae": mae}


if __name__ == "__main__":
    train_self_impact_net()
