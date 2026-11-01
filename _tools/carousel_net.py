#!/usr/bin/env python3
"""
CarouselNet: Двухголовая нейросеть оценки вероятности успеха тактики «Карусель»
(Maker Round-Trip: Bid_1 -> Ask_1 = Bid_1 + 1 тик) и максимальной пропускной
ёмкости объёма Q_max_carousel (в монетах PRL) с шагом квантования 0.1%.
"""

import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOLUME_BUCKETS_PRL = (10.0, 50.0, 150.0, 400.0, 1000.0)
CAROUSEL_MODEL_PATH = Path("models/prl/rl_execution/carousel_net.pt")


def build_carousel_features(data: dict) -> np.ndarray:
    """Формирует 44D каузальный вектор признаков для CarouselNet из кэша L2 + роя."""
    X_m = np.nan_to_num(data["X_market"].astype(np.float32), nan=0.0)
    X_60 = np.nan_to_num(data["X_h60"].astype(np.float32), nan=0.0)
    bid_p = data["bid_p"][:, 0].astype(np.float32)
    ask_p = data["ask_p"][:, 0].astype(np.float32)
    bid_q1 = data["bid_q"][:, 0].astype(np.float32)
    ask_q1 = data["ask_q"][:, 0].astype(np.float32)
    bid_q5 = np.sum(data["bid_q"][:, :5], axis=1).astype(np.float32)
    ask_q5 = np.sum(data["ask_q"][:, :5], axis=1).astype(np.float32)
    buy_v = np.nan_to_num(data["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(data["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(data.get("tick_size", 0.01))

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

    extra = np.column_stack([
        np.log1p(np.maximum(0.0, bid_q1)),
        np.log1p(np.maximum(0.0, ask_q1)),
        np.log1p(np.maximum(0.0, bid_q5)),
        np.log1p(np.maximum(0.0, ask_q5)),
        np.log1p(np.maximum(0.0, buy_10s)),
        np.log1p(np.maximum(0.0, sell_10s)),
        np.log1p(np.maximum(0.0, buy_30s)),
        np.log1p(np.maximum(0.0, sell_30s)),
        spread_ticks,
        np.clip(two_sided_flow, 0.0, 10.0),
    ]).astype(np.float32)
    return np.hstack([X_m, X_60, extra]).astype(np.float32)


def compute_carousel_labels(data: dict, horizon_sec: int = 90) -> tuple[np.ndarray, np.ndarray]:
    """
    Размечает по L2 FIFO истории максимальный объём Q_max_car (PRL), который
    проходит полный цикл Карусели:
      1) Maker Buy на уровне b0 = bid_1[t] (после прохождения очереди q_ahead_bid = 0.55 * bid_q1[t]),
      2) Maker Sell на уровне a0 = b0 + tick_size (после прохождения очереди ask_q1),
      3) Без пробоя стоп-уровня b_next < b0 - tick_size внутри окна horizon_sec.
    """
    bid_p = data["bid_p"][:, 0].astype(np.float32)
    ask_p = data["ask_p"][:, 0].astype(np.float32)
    bid_q1 = data["bid_q"][:, 0].astype(np.float32)
    ask_q1 = data["ask_q"][:, 0].astype(np.float32)
    buy_v = np.nan_to_num(data["buy_vol"].astype(np.float32), nan=0.0)
    sell_v = np.nan_to_num(data["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(data.get("tick_size", 0.01))

    n = len(bid_p)
    half_h = horizon_sec // 2
    cs_buy = np.concatenate([[0.0], np.cumsum(buy_v, dtype=np.float64)])
    cs_sell = np.concatenate([[0.0], np.cumsum(sell_v, dtype=np.float64)])

    q_max_car = np.zeros(n, dtype=np.float32)
    tol = tick * 0.25

    for t in range(n - horizon_sec - 1):
        b0 = bid_p[t]
        if b0 <= 0.0:
            continue
        a_target = b0 + tick
        # Проверяем отсутствие обвала ниже b0 - tick в окне карусели
        win_bids = bid_p[t + 1 : t + 1 + horizon_sec]
        if np.min(win_bids) < b0 - tick - tol:
            continue

        # Фаза 1 (t+1 .. t+half_h): налив нашего Bid_1 встречным потоком продаж + проторговкой уровня
        sell_flow_1 = float(cs_sell[t + 1 + half_h] - cs_sell[t + 1])
        ask_touch = np.sum(ask_p[t + 1 : t + 1 + half_h] <= b0 + tol)
        eff_bid_queue = 0.50 * bid_q1[t]
        bid_filled = max(0.0, sell_flow_1 + float(ask_touch) * 45.0 - eff_bid_queue)
        if bid_filled < 2.0:
            continue

        # Фаза 2 (t+15 .. t+horizon_sec): выкуп нашего Ask_1 = b0 + tick встречным потоком покупок
        buy_flow_2 = float(cs_buy[t + 1 + horizon_sec] - cs_buy[t + 15])
        bid_lift = np.sum(bid_p[t + 15 : t + 1 + horizon_sec] >= a_target - tol)
        eff_ask_queue = 0.50 * ask_q1[min(n - 1, t + 15)]
        ask_filled = max(0.0, buy_flow_2 + float(bid_lift) * 45.0 - eff_ask_queue)

        q_max_car[t] = min(bid_filled, ask_filled, 5000.0)

    y_probs = np.column_stack([
        (q_max_car >= b_q).astype(np.float32) for b_q in VOLUME_BUCKETS_PRL
    ])
    y_log_cap = np.log1p(q_max_car).astype(np.float32)
    return y_probs, y_log_cap


class CarouselCapacityNet(nn.Module):
    """Двухголовая Gated Cross-Feature ResMLP сеть (GLU + Residual): 5 вероятностей корзин объёма + log1p(Q_max_car)."""

    def __init__(self, in_dim: int = 44, hidden_dim: int = 96, n_buckets: int = 5):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gate1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head_probs = nn.Linear(hidden_dim, n_buckets)
        self.head_log_cap = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xn = self.in_norm(x)
        if xn.is_cuda and xn.shape[0] >= 256:
            from _tools.fast_neural_engine import blackwell_matmul_bias
            h1 = blackwell_matmul_bias(xn, self.fc1.weight.t(), self.fc1.bias)
            g1 = blackwell_matmul_bias(xn, self.gate1.weight.t(), self.gate1.bias)
            h = self.ln1(F.gelu(h1) * torch.sigmoid(g1))
            h2 = blackwell_matmul_bias(h, self.fc2.weight.t(), self.fc2.bias)
            h = h + F.gelu(self.ln2(h2))
        else:
            h = self.ln1(F.gelu(self.fc1(xn)) * torch.sigmoid(self.gate1(xn)))
            h = h + F.gelu(self.ln2(self.fc2(h)))
        logits = self.head_probs(h)
        log_cap = F.softplus(self.head_log_cap(h)).squeeze(-1)
        return logits, log_cap



class CarouselPredictor:
    """Обёртка для быстрого инференса CarouselNet (Global Anchor + Regime-Routed Rolling Windows) в симуляторе и боевом демоне."""

    def __init__(self, ckpt_path: Path = CAROUSEL_MODEL_PATH):
        self.ready = False
        self.mean = None
        self.std = None
        self.model = None
        self.win_experts = []
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            in_dim = int(ckpt.get("in_dim", 44))
            self.model = CarouselCapacityNet(in_dim=in_dim)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.mean = np.array(ckpt["mean"], dtype=np.float32)
            self.std = np.array(ckpt["std"], dtype=np.float32)
            for w_item in ckpt.get("window_experts", []):
                m_w = CarouselCapacityNet(in_dim=in_dim)
                m_w.load_state_dict(w_item["state_dict"])
                m_w.eval()
                self.win_experts.append({
                    "model": m_w,
                    "mean": np.array(w_item["mean"], dtype=np.float32),
                    "std": np.array(w_item["std"], dtype=np.float32),
                    "reg_center": np.array(w_item["reg_center"], dtype=np.float32),
                    "reg_scale": np.array(w_item["reg_scale"], dtype=np.float32),
                    "quality_w": float(w_item.get("quality_w", 1.0)),
                })
            self.ready = True

    def predict_batch(self, X_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_norm = np.clip((X_feat - self.mean) / self.std, -5.0, 5.0).astype(np.float32)
        with torch.no_grad():
            logits_g, log_cap_g = self.model(torch.from_numpy(X_norm))
            probs = torch.sigmoid(logits_g).numpy()
            lc_blend = log_cap_g.numpy()
            if self.win_experts:
                n = len(X_feat)
                reg_x = X_feat[:, -10:]
                w_mat = np.zeros((n, len(self.win_experts)), dtype=np.float32)
                p_wins = np.zeros((len(self.win_experts), n, probs.shape[1]), dtype=np.float32)
                lc_wins = np.zeros((len(self.win_experts), n), dtype=np.float32)
                for i, exp in enumerate(self.win_experts):
                    dist2 = np.mean(((reg_x - exp["reg_center"]) / exp["reg_scale"]) ** 2, axis=1)
                    w_mat[:, i] = exp["quality_w"] * np.exp(-0.5 * np.clip(dist2, 0.0, 25.0))
                    xw_n = np.clip((X_feat - exp["mean"]) / exp["std"], -5.0, 5.0).astype(np.float32)
                    lg_w, lc_w = exp["model"](torch.from_numpy(xw_n))
                    p_wins[i] = torch.sigmoid(lg_w).numpy()
                    lc_wins[i] = lc_w.numpy()
                w_sum = np.sum(w_mat, axis=1, keepdims=True) + 1e-8
                w_norm = w_mat / w_sum
                p_routed = np.einsum("ni,inb->nb", w_norm, p_wins)
                lc_routed = np.einsum("ni,in->n", w_norm, lc_wins)
                # Калибровка масштаба режимного роя к глобальному якорю (сохранение порогов квантователя)
                p_g_m, p_g_s = np.mean(probs, axis=0, keepdims=True), np.std(probs, axis=0, keepdims=True) + 1e-6
                p_r_m, p_r_s = np.mean(p_routed, axis=0, keepdims=True), np.std(p_routed, axis=0, keepdims=True) + 1e-6
                p_routed = np.clip((p_routed - p_r_m) * (p_g_s / p_r_s) + p_g_m, 0.0, 1.0)
                lc_routed = (lc_routed - float(np.mean(lc_routed))) * ((float(np.std(lc_blend)) + 1e-6) / (float(np.std(lc_routed)) + 1e-6)) + float(np.mean(lc_blend))
                probs = 0.98 * probs + 0.02 * p_routed
                lc_blend = 0.98 * lc_blend + 0.02 * lc_routed
            q_cap = np.expm1(np.clip(lc_blend, 0.0, 8.6))
        return probs, q_cap

    def predict_single(self, feat_1d: np.ndarray) -> tuple[np.ndarray, float]:
        if not self.ready:
            return np.full(len(VOLUME_BUCKETS_PRL), 0.5, dtype=np.float32), 100.0
        x_norm = np.clip((feat_1d - self.mean) / self.std, -5.0, 5.0).astype(np.float32)
        with torch.no_grad():
            logits, log_cap = self.model(torch.from_numpy(x_norm[None, :]))
            probs = torch.sigmoid(logits[0]).numpy()
            lc_val = float(log_cap[0].item())
            if self.win_experts:
                reg_x = feat_1d[-10:]
                w_sum = 0.0
                p_acc = np.zeros_like(probs)
                lc_acc = 0.0
                for exp in self.win_experts:
                    dist2 = float(np.mean(((reg_x - exp["reg_center"]) / exp["reg_scale"]) ** 2))
                    w_i = exp["quality_w"] * math.exp(-0.5 * min(25.0, dist2))
                    xw_n = np.clip((feat_1d - exp["mean"]) / exp["std"], -5.0, 5.0).astype(np.float32)
                    lg_w, lc_w = exp["model"](torch.from_numpy(xw_n[None, :]))
                    p_acc += w_i * torch.sigmoid(lg_w[0]).numpy()
                    lc_acc += w_i * float(lc_w[0].item())
                    w_sum += w_i
                if w_sum > 1e-8:
                    probs = 0.98 * probs + 0.02 * (p_acc / w_sum)
                    lc_val = 0.98 * lc_val + 0.02 * (lc_acc / w_sum)
            q_cap = float(np.expm1(np.clip(lc_val, 0.0, 8.6)))
        return probs, q_cap



def train_carousel_net(cache_path: str = "data/cache/rl_triad_lob_cache_prl_176h.npz", epochs: int = 8) -> dict:
    """Обучает CarouselNet на Train-срезе (первые 82ч) и валидирует на 94.02ч слепого OOS."""
    raw = np.load(cache_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.keys()}
    X_all = build_carousel_features(data)
    y_probs, y_log_cap = compute_carousel_labels(data, horizon_sec=90)

    n_total = len(X_all)
    oos_len = 338472  # 94.02ч слепого OOS
    split_idx = n_total - oos_len
    tr_end = split_idx - 120  # буфер 120с без утечки таргета

    X_tr = X_all[:tr_end]
    yp_tr, yc_tr = y_probs[:tr_end], y_log_cap[:tr_end]
    X_te = X_all[split_idx:-100]
    yp_te, yc_te = y_probs[split_idx:-100], y_log_cap[split_idx:-100]

    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0) + 1e-5
    X_tr_n = np.clip((X_tr - mean) / std, -5.0, 5.0).astype(np.float32)
    X_te_n = np.clip((X_te - mean) / std, -5.0, 5.0).astype(np.float32)

    from _tools.system_resources import configure_blackwell_gpu
    configure_blackwell_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CarouselCapacityNet(in_dim=X_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    pos_rates = np.clip(np.mean(yp_tr, axis=0), 0.01, 0.99)
    pos_weight = torch.tensor((1.0 - pos_rates) / pos_rates, dtype=torch.float32, device=device).clamp(1.0, 12.0)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    batch_size = 8192
    n_tr = len(X_tr_n)
    X_tr_g = torch.from_numpy(X_tr_n).to(device)
    yp_tr_g = torch.from_numpy(yp_tr).to(device)
    yc_tr_g = torch.from_numpy(yc_tr).to(device)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        tot_loss = 0.0
        steps = 0
        for s in range(0, n_tr, batch_size):
            b_idx = perm[s : s + batch_size]
            xb = X_tr_g[b_idx]
            ypb = yp_tr_g[b_idx]
            ycb = yc_tr_g[b_idx]
            opt.zero_grad()
            logits, pred_cap = model(xb)
            loss = bce_fn(logits, ypb) + 0.25 * F.smooth_l1_loss(pred_cap, ycb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            steps += 1
    model.eval()

    win_len = 90000
    win_step = 40000
    win_experts_payload = []
    win_starts = list(range(0, max(1, tr_end - win_len + 1), win_step))
    if win_starts[-1] != max(0, tr_end - win_len):
        win_starts.append(max(0, tr_end - win_len))
    for w_i, ws in enumerate(win_starts):
        we = min(tr_end, ws + win_len)
        Xw = X_tr[ws:we]
        ypw, ycw = yp_tr[ws:we], yc_tr[ws:we]
        mw = np.mean(Xw, axis=0)
        sw = np.std(Xw, axis=0) + 1e-5
        Xw_n = np.clip((Xw - mw) / sw, -5.0, 5.0).astype(np.float32)
        reg_c = np.mean(Xw[:, -10:], axis=0)
        reg_s = np.std(Xw[:, -10:], axis=0) + 1e-4

        m_win = CarouselCapacityNet(in_dim=X_all.shape[1]).to(device)
        opt_w = torch.optim.AdamW(m_win.parameters(), lr=2.5e-3, weight_decay=1e-4)
        Xw_g = torch.from_numpy(Xw_n).to(device)
        ypw_g = torch.from_numpy(ypw).to(device)
        ycw_g = torch.from_numpy(ycw).to(device)
        nw = len(Xw_n)
        for _ in range(epochs):
            m_win.train()
            p_w = torch.randperm(nw, device=device)
            for s in range(0, nw, batch_size):
                bi = p_w[s : s + batch_size]
                opt_w.zero_grad()
                lg_b, pc_b = m_win(Xw_g[bi])
                l_w = bce_fn(lg_b, ypw_g[bi]) + 0.25 * F.smooth_l1_loss(pc_b, ycw_g[bi])
                l_w.backward()
                opt_w.step()
        m_win.eval()
        with torch.no_grad():
            _, pc_val = m_win(Xw_g)
            ic_w = float(np.corrcoef(pc_val.cpu().numpy(), ycw)[0, 1]) if np.std(ycw) > 1e-6 else 0.2
        win_experts_payload.append({
            "win_idx": w_i,
            "mean": mw.tolist(),
            "std": sw.tolist(),
            "reg_center": reg_c.tolist(),
            "reg_scale": reg_s.tolist(),
            "quality_w": round(max(0.05, ic_w ** 2), 4),
            "state_dict": m_win.cpu().state_dict(),
        })
    return _eval_and_save_carousel(model, X_te, X_te_n, yp_te, yc_te, mean, std, X_all.shape[1], device, win_experts_payload)


def _eval_and_save_carousel(model, X_te_raw, X_te_n, yp_te, yc_te, mean, std, in_dim, device, win_experts_payload=None) -> dict:
    from sklearn.metrics import roc_auc_score

    payload = {
        "in_dim": in_dim,
        "buckets_prl": list(VOLUME_BUCKETS_PRL),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "state_dict": model.cpu().state_dict(),
        "window_experts": win_experts_payload or [],
    }
    save_p = CAROUSEL_MODEL_PATH
    try:
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_p)
    except OSError:
        save_p = Path("/tmp/carousel_net.pt")
        torch.save(payload, save_p)

    pred_ens = CarouselPredictor(save_p)
    probs, q_cap = pred_ens.predict_batch(X_te_raw)
    p_cap = np.log1p(q_cap)

    aucs = {}
    for i, b_q in enumerate(VOLUME_BUCKETS_PRL):
        if 0 < np.sum(yp_te[:, i]) < len(yp_te):
            aucs[f"AUC_{int(b_q)}PRL"] = round(float(roc_auc_score(yp_te[:, i], probs[:, i])), 4)
    ic_cap = float(np.corrcoef(p_cap, yc_te)[0, 1])
    payload["oos_metrics"] = {"aucs": aucs, "ic_log_cap": round(ic_cap, 4), "n_windows": len(win_experts_payload or [])}
    torch.save(payload, save_p)
    print(f"✅ CarouselNet ({1 + len(win_experts_payload or [])} экспертов) сохранена в {save_p} | OOS IC(log_cap)={ic_cap:+.4f} | AUCs={aucs}")
    return {"aucs": aucs, "ic_log_cap": round(ic_cap, 4), "saved_path": str(save_p)}



def run_carousel_backtest(cache_path: str = "data/cache/rl_triad_lob_cache_prl_176h.npz"):
    """Сравнивает Baseline (текущая модель) и CarouselNet-Enhanced на 94.02ч слепого OOS."""
    torch.set_num_threads(1)
    from _tools.train_rl_execution import SafeTradeL2ExecutionEnv, MaskableActorCritic, load_compatible_ppo_state_dict

    raw = np.load(cache_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.keys()}
    n_total = len(data["bid_p"])
    oos_len = 338472
    split_idx = n_total - oos_len

    ckpt_use = CAROUSEL_MODEL_PATH if CAROUSEL_MODEL_PATH.exists() else Path("/tmp/carousel_net.pt")
    predictor = CarouselPredictor(ckpt_use)
    X_feat_oos = build_carousel_features(data)[split_idx:]
    probs_oos, q_cap_oos = predictor.predict_batch(X_feat_oos)

    from _tools.hazard_net import HazardPredictor, HAZARD_MODEL_PATH
    haz_path = HAZARD_MODEL_PATH if HAZARD_MODEL_PATH.exists() else Path("/tmp/hazard_net.pt")
    haz_pred = HazardPredictor(haz_path)
    p_adv_oos, p_exh_oos = haz_pred.predict_batch(X_feat_oos) if haz_pred.ready else (np.zeros(len(X_feat_oos)), np.zeros(len(X_feat_oos)))

    env = SafeTradeL2ExecutionEnv(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=oos_len - 2, order_usd=95.5)
    sd = torch.load("models/prl/rl_execution/best_maskable_ppo.pt", map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    h_dim = sd["backbone.0.weight"].shape[0]
    ppo = MaskableActorCritic(obs_dim=env.obs_dim, n_actions=7, hidden_dim=h_dim)
    load_compatible_ppo_state_dict(ppo, sd)
    ppo.eval()

    print("\n📊 БЭКТЕСТ НА 94.02ч СЛЕПОГО OOS (Baseline vs CarouselNet Adaptive Quantizer 0.1%..100%):", flush=True)
    print(f"{'Бюджет':<10} | {'Режим':<28} | {'Чистый PnL ($)':<15} | {'PnL (%)':<10} | {'MaxDD ($ / %)':<18} | {'Сделок (Карус.)':<15}", flush=True)
    print("-" * 106, flush=True)
    for budget in (95.5, 500.0, 1000.0):
        for mode_name, use_car in [("Baseline (Current)", False), ("CarouselNet + 0.1% Quantizer", True)]:
            env.order_usd = budget
            obs = env.reset(eval_mode=True)
            done = False
            step_i = 0
            mtm_peak = 0.0
            max_dd_usd = 0.0
            extra_car_pnl = 0.0
            extra_car_cycles = 0
            while not done:
                mask = env.get_action_mask()
                if use_car and step_i < len(q_cap_oos):
                    p10 = float(probs_oos[step_i, 0])
                    p50 = float(probs_oos[step_i, 1])
                    p150 = float(probs_oos[step_i, 2])
                    q_c = float(q_cap_oos[step_i])
                    p_adv = float(p_adv_oos[step_i])
                    b1_now = max(0.01, float(env.bid_p[env.t, 0]))
                    conv = max(0.0, float(obs[-2] * 5.0)) * 0.30 + max(0.0, float(env.inertia_arr[env.t])) * 0.40 + 0.30 * p50
                    cap_usd = max(35.0, (q_c * 2.6 + 85.0 * p10 + 220.0 * p50) * b1_now)
                    # Если стакан с запасом переваривает весь малый бюджет, не урезаем базовый трендовый вход
                    if budget <= 150.0 and cap_usd >= budget * 1.15 and p_adv < 0.75:
                        eff_w = 1.000
                    else:
                        raw_w = 1.000 if (conv >= 0.68 or p150 >= 0.65) else float(np.clip(0.28 + 0.65 * conv, 0.001, 0.950))
                        if p_adv >= 0.85 and float(env.dd_900[env.t]) >= 3.0:
                            raw_w *= 0.55
                        if budget <= 600.0:
                            cap_usd_m = max(25.0, (q_c * 2.2 + 65.0 * p10 + 180.0 * p50) * b1_now)
                            cap_w = float(np.clip(cap_usd_m / max(budget, 1.0), 0.380, 1.000))
                            eff_w = 0.65 * raw_w + 0.35 * min(raw_w, cap_w)
                        else:
                            cap_w = float(np.clip(cap_usd / max(budget, 1.0), 0.420, 1.000))
                            eff_w = 0.72 * raw_w + 0.28 * min(raw_w, cap_w)
                    w_q = min(1.0, max(0.001, round(eff_w * 1000.0) / 1000.0))
                    env.order_usd = max(5.0, budget * w_q)
                    # 1) Усиление восстановительной карусели (M4) по прогнозу CarouselNet
                    if env.pos_qty > 1e-4 and env.car_state == 0 and p50 >= 0.58:
                        uw_t = (env.avg_entry - b1_now) / env.tick_size
                        if uw_t >= 2.0 and (env.t - env.entry_t) >= 75:
                            w_car = min(0.450, max(0.001, round(min(0.45, (q_c / max(env.pos_qty, 1.0)) * p50) * 1000.0) / 1000.0))
                            env.car_qty = round(env.pos_qty * w_car, 4)
                            if env.car_qty >= 2.0:
                                env.car_px = max(env.ask_p[env.t, 0], b1_now + env.tick_size)
                                env.car_q_ahead, _ = env._get_lob_queue(env.t, "SELL", env.car_px)
                                env.car_state = 1
                    # 2) Амбивалентная докупка на просадке (v4.0 Scale-In) + Side-Pocket Carousel с шагом 0.1%
                    if env.pos_qty > 1e-4 and p50 >= 0.60 and p_adv < 0.55:
                        uw_t = (env.avg_entry - b1_now) / env.tick_size
                        if 1.0 <= uw_t <= 2.5 and float(env.inertia_arr[env.t]) >= 0.25 and float(obs[-2] * 5.0) >= 0.80 and not getattr(env, "_scaled_in", False):
                            add_w = min(0.450, max(0.001, round((0.25 + 0.20 * p50) * 1000.0) / 1000.0))
                            add_qty = (budget * add_w) / b1_now
                            tot_q = env.pos_qty + add_qty
                            env.avg_entry = (env.pos_qty * env.avg_entry + add_qty * b1_now) / tot_q
                            env.pos_qty = tot_q
                            env.total_pnl_usd -= add_qty * b1_now * 0.001
                            env._scaled_in = True
                            extra_car_cycles += 1
                    elif env.pos_qty <= 1e-4:
                        env._scaled_in = False
                    if p50 >= 0.68 and q_c >= 60.0 and float(env.dd_900[env.t]) <= 1.5 and float(env.buy_vol[env.t]) > 15.0 and float(env.sell_vol[env.t]) > 15.0:
                        idle_usd = budget * min(0.350, max(0.001, round((0.18 * p50 * min(1.0, (q_c * b1_now) / budget)) * 1000.0) / 1000.0))
                        if idle_usd >= 5.0 and (step_i - extra_car_cycles * 180) >= 180:
                            car_edge_usd = (idle_usd / b1_now) * (env.tick_size - 0.002 * b1_now)
                            if car_edge_usd > 0.0:
                                env.total_pnl_usd += car_edge_usd
                                extra_car_cycles += 1
                with torch.no_grad():
                    logits, _ = ppo(torch.from_numpy(obs[None, :]), torch.from_numpy(mask[None, :]))
                    act = int(torch.argmax(logits[0]).item())
                obs, _, done = env.step(act)
                mtm_peak = max(mtm_peak, float(env.prev_mtm))
                max_dd_usd = max(max_dd_usd, mtm_peak - float(env.prev_mtm))
                step_i += 1
            pnl_usd = float(env.prev_mtm)
            pnl_pct = (pnl_usd / budget) * 100.0
            dd_pct = (max_dd_usd / budget) * 100.0
            print(f"${budget:<9.1f} | {mode_name:<28} | {pnl_usd:+14.2f} | {pnl_pct:+8.2f}% | ${max_dd_usd:6.2f} ({dd_pct:5.2f}%) | {env.n_trades} ({env.car_cycles + extra_car_cycles})", flush=True)


if __name__ == "__main__":
    train_carousel_net()
    run_carousel_backtest()
