#!/usr/bin/env python3
"""
ExecutionRealismNet (v7.0 Stage 2):
Калибратор вероятности реального исполнения лимитного ордера (p_fill_real)
и риска фронт-раннинга чужими ботами биржи SafeTrade (p_frontrun)
на основе 695,737 секунд боевой телеметрии prl_rl_telemetry.parquet + L2 стакана.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

EXEC_REALISM_MODEL_PATH = Path("models/prl/rl_execution/execution_realism_net.pt")


def build_exec_realism_features_from_cache(data: dict, order_qty_prl: float = 150.0) -> np.ndarray:
    """Формирует 16D вектор микроструктурных признаков для ExecutionRealismNet из L2 кэша."""
    X_m = np.nan_to_num(data["X_market"].astype(np.float32), nan=0.0)
    X_60 = np.nan_to_num(data.get("X_h60", np.zeros((len(X_m), 6), dtype=np.float32)).astype(np.float32), nan=0.0)
    X_90 = np.nan_to_num(data.get("X_h90", X_60).astype(np.float32), nan=0.0)
    X_120 = np.nan_to_num(data.get("X_h120", X_60).astype(np.float32), nan=0.0)
    bid_p = data["bid_p"][:, 0].astype(np.float32)
    ask_p = data["ask_p"][:, 0].astype(np.float32)
    bid_q1 = np.maximum(0.0, data["bid_q"][:, 0].astype(np.float32))
    ask_q1 = np.maximum(0.0, data["ask_q"][:, 0].astype(np.float32))
    bid_q5 = np.maximum(0.0, np.sum(data["bid_q"][:, :5], axis=1).astype(np.float32))
    ask_q5 = np.maximum(0.0, np.sum(data["ask_q"][:, :5], axis=1).astype(np.float32))
    buy_v = np.nan_to_num(data["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(data["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(data.get("tick_size", 0.01))

    n = len(bid_p)
    cs_buy = np.concatenate([[0.0], np.cumsum(buy_v, dtype=np.float64)])
    cs_sell = np.concatenate([[0.0], np.cumsum(sell_v, dtype=np.float64)])
    idx = np.arange(n)
    buy_15s = (cs_buy[idx + 1] - cs_buy[np.maximum(0, idx - 14)]).astype(np.float32)
    sell_15s = (cs_sell[idx + 1] - cs_sell[np.maximum(0, idx - 14)]).astype(np.float32)

    spread_ticks = np.clip((ask_p - bid_p) / max(tick, 1e-6), 1.0, 20.0)
    imb1 = (bid_q1 - ask_q1) / (bid_q1 + ask_q1 + 1e-5)
    imb5 = (bid_q5 - ask_q5) / (bid_q5 + ask_q5 + 1e-5)
    q_share_bid = float(order_qty_prl) / (bid_q1 + float(order_qty_prl) + 10.0)
    q_share_ask = float(order_qty_prl) / (ask_q1 + float(order_qty_prl) + 10.0)

    return np.column_stack([
        spread_ticks,
        imb1,
        imb5,
        np.log1p(bid_q1) / 10.0,
        np.log1p(ask_q1) / 10.0,
        np.log1p(bid_q5) / 12.0,
        np.log1p(ask_q5) / 12.0,
        np.log1p(buy_15s) / 8.0,
        np.log1p(sell_15s) / 8.0,
        q_share_bid,
        q_share_ask,
        X_m[:, 0],   # E_z_5s
        X_m[:, 6],   # E_z_12s
        X_m[:, 12],  # E_z_25s
        X_90[:, 0],  # E_z_90s
        X_120[:, 0], # E_z_120s
    ]).astype(np.float32)


class ExecutionRealismNet(nn.Module):
    """Двухголовая GLU-ResMLP сеть: [logit_fill_real_30s, logit_frontrun_30s]."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gate1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.in_norm(x)
        h = self.ln1(F.gelu(self.fc1(xn)) * torch.sigmoid(self.gate1(xn)))
        h = h + F.gelu(self.ln2(self.fc2(h)))
        return self.head(h)


class ExecutionRealismPredictor:
    """Быстрый инференс ExecutionRealismNet для симулятора и боевого демона."""

    def __init__(self, ckpt_path: Path = EXEC_REALISM_MODEL_PATH):
        self.ready = False
        self.mean = None
        self.std = None
        self.model = None
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            in_dim = int(ckpt.get("in_dim", 16))
            self.model = ExecutionRealismNet(in_dim=in_dim)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.mean = np.array(ckpt["mean"], dtype=np.float32)
            self.std = np.array(ckpt["std"], dtype=np.float32)
            self.ready = True

    def predict_batch(self, X_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready:
            return np.full(len(X_feat), 0.35, dtype=np.float32), np.full(len(X_feat), 0.15, dtype=np.float32)
        in_d = len(self.mean)
        X_in = X_feat[:, :in_d] if X_feat.shape[1] >= in_d else np.pad(X_feat, ((0, 0), (0, in_d - X_feat.shape[1])))
        X_norm = np.clip((X_in - self.mean) / self.std, -5.0, 5.0).astype(np.float32)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.from_numpy(X_norm))).numpy()
        return probs[:, 0], probs[:, 1]

    def predict_single(self, feat_1d: np.ndarray) -> tuple[float, float]:
        if not self.ready:
            return 0.35, 0.15
        in_d = len(self.mean)
        x_in = feat_1d[:in_d] if len(feat_1d) >= in_d else np.pad(feat_1d, (0, in_d - len(feat_1d)))
        x_norm = np.clip((x_in - self.mean) / self.std, -5.0, 5.0).astype(np.float32)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.from_numpy(x_norm[None, :])))[0].numpy()
        return float(probs[0]), float(probs[1])


def train_execution_realism_net(
    cache_path: str = "data/cache/rl_triad_lob_cache_prl_176h.npz",
    telemetry_path: str = "data/processed/prl_rl_telemetry.parquet",
    epochs: int = 12,
) -> dict:
    """
    Обучает ExecutionRealismNet на пересечении боевой телеметрии (695,737 сек) и L2 кэша:
      - y_fill_real: реальный филл нашего ордера в течение 30с ИЛИ проторговка очереди > 0.65 * q_top,
      - y_frontrun: перебитие нашего крупного лимитного ордера на 1 тик вперёд (penny-jumping) чужим ботом в течение 30с.
    """
    from sklearn.metrics import roc_auc_score

    raw = np.load(cache_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.keys()}
    X_all = build_exec_realism_features_from_cache(data, order_qty_prl=150.0)
    n = len(X_all)

    bid_p = data["bid_p"][:, 0].astype(np.float32)
    ask_p = data["ask_p"][:, 0].astype(np.float32)
    bid_q1 = np.maximum(0.0, data["bid_q"][:, 0].astype(np.float32))
    ask_q1 = np.maximum(0.0, data["ask_q"][:, 0].astype(np.float32))
    buy_v = np.nan_to_num(data["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(data["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(data.get("tick_size", 0.01))
    tol = tick * 0.25

    # Синхронизируем реальные события боевой телеметрии по секундным меткам
    tel_fill_mask = np.zeros(n, dtype=np.float32)
    tel_frontrun_mask = np.zeros(n, dtype=np.float32)
    tel_weight = np.ones(n, dtype=np.float32)
    t_path = Path(telemetry_path)
    if t_path.exists():
        try:
            df_t = pq.read_table(
                t_path,
                columns=["timestamp", "b1", "a1", "pos_qty", "active_bid_p", "active_bid_q", "active_ask_p", "active_ask_q"],
            ).to_pandas()
            ts_cache = pd.DatetimeIndex(pd.to_datetime(data["ts"], utc=True)).floor("s").asi8
            ts_tel = pd.DatetimeIndex(pd.to_datetime(df_t["timestamp"], utc=True)).floor("s").asi8
            ts_map = {int(ts_v): i for i, ts_v in enumerate(ts_cache)}
            d_pos = np.abs(df_t["pos_qty"].diff().fillna(0.0).values)
            act_bid_p = df_t["active_bid_p"].fillna(0.0).values
            act_bid_q = df_t["active_bid_q"].fillna(0.0).values
            b1_t = df_t["b1"].fillna(0.0).values
            for r_i, ts_v in enumerate(ts_tel):
                c_idx = ts_map.get(int(ts_v), -1)
                if c_idx >= 0:
                    if act_bid_q[r_i] > 1.0 or df_t["active_ask_q"].iloc[r_i] > 1.0:
                        tel_weight[c_idx] = 2.5  # Повышенный вес реальных боевых секунд с активным ордером
                    if d_pos[r_i] > 0.5:
                        tel_fill_mask[max(0, c_idx - 30) : c_idx + 1] = 1.0
                    if act_bid_p[r_i] > 0.0 and b1_t[r_i] >= act_bid_p[r_i] + tick - tol and d_pos[r_i] < 0.1:
                        tel_frontrun_mask[max(0, c_idx - 30) : c_idx + 1] = 1.0
        except Exception as e:
            print(f"⚠️ Telemetry alignment note: {e}")

    cs_buy = np.concatenate([[0.0], np.cumsum(buy_v, dtype=np.float64)])
    cs_sell = np.concatenate([[0.0], np.cumsum(sell_v, dtype=np.float64)])
    y_fill = np.zeros(n, dtype=np.float32)
    y_frontrun = np.zeros(n, dtype=np.float32)

    for t in range(n - 32):
        b0, a0 = bid_p[t], ask_p[t]
        sell_30s = float(cs_sell[t + 31] - cs_sell[t + 1])
        buy_30s = float(cs_buy[t + 31] - cs_buy[t + 1])
        min_a_30s = float(np.min(ask_p[t + 1 : t + 31]))
        max_b_30s = float(np.max(bid_p[t + 1 : t + 31]))
        # Филл за 30с: либо реальный боевой филл из телеметрии, либо прохождение очереди стакана
        if tel_fill_mask[t] > 0.5 or sell_30s >= 0.45 * bid_q1[t] + 25.0 or min_a_30s <= b0 + tol:
            y_fill[t] = 1.0
        # Penny-jump / Frontrun: наш ордер на b0 остаётся позади, потому что чужой бот шагает на b0 + tick без налива нам
        if tel_frontrun_mask[t] > 0.5 or (max_b_30s >= b0 + tick - tol and sell_30s < 10.0 and bid_q1[t] >= 800.0):
            y_frontrun[t] = 1.0

    Y_all = np.column_stack([y_fill, y_frontrun]).astype(np.float32)
    oos_len = min(int(n * 0.45), 310000)
    split_idx = n - oos_len
    tr_end = split_idx - 60

    X_tr, Y_tr, W_tr = X_all[:tr_end], Y_all[:tr_end], tel_weight[:tr_end]
    X_te, Y_te = X_all[split_idx:-35], Y_all[split_idx:-35]

    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0) + 1e-5
    X_tr_n = np.clip((X_tr - mean) / std, -5.0, 5.0).astype(np.float32)
    X_te_n = np.clip((X_te - mean) / std, -5.0, 5.0).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ExecutionRealismNet(in_dim=X_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    X_tr_g = torch.from_numpy(X_tr_n).to(device)
    Y_tr_g = torch.from_numpy(Y_tr).to(device)
    W_tr_g = torch.from_numpy(W_tr[:, None]).to(device)

    batch_size = 8192
    n_tr = len(X_tr_n)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, batch_size):
            bi = perm[s : s + batch_size]
            opt.zero_grad()
            logits = model(X_tr_g[bi])
            loss = (F.binary_cross_entropy_with_logits(logits, Y_tr_g[bi], reduction="none") * W_tr_g[bi]).mean()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(torch.from_numpy(X_te_n).to(device))).cpu().numpy()

    auc_fill = float(roc_auc_score(Y_te[:, 0], p_te[:, 0])) if len(np.unique(Y_te[:, 0])) > 1 else 0.5
    auc_fr = float(roc_auc_score(Y_te[:, 1], p_te[:, 1])) if len(np.unique(Y_te[:, 1])) > 1 else 0.5

    EXEC_REALISM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "in_dim": int(X_all.shape[1]),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "state_dict": model.cpu().state_dict(),
            "metrics_oos": {"auc_fill_real": round(auc_fill, 4), "auc_frontrun": round(auc_fr, 4)},
        },
        EXEC_REALISM_MODEL_PATH,
    )
    print(f"✅ ExecutionRealismNet обучена и сохранена в {EXEC_REALISM_MODEL_PATH} | OOS AUC Fill={auc_fill:.4f}, Frontrun={auc_fr:.4f}")
    return {"auc_fill_real": auc_fill, "auc_frontrun": auc_fr}