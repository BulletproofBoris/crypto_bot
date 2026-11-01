#!/usr/bin/env python3
"""
HazardNet (Этап 3): Двухголовая модель оценки микроструктурного риска:
  1) Adverse Selection Hazard: P(Bid_1 Collapse within 5s | Fill) — риск токсичного сноса Bid_1 после налива,
  2) Wave Exhaustion Hazard: P(Peak Reached within 10s) — вероятность затухания восходящего импульса.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _tools.carousel_net import build_carousel_features

HAZARD_MODEL_PATH = Path("models/prl/rl_execution/hazard_net.pt")


def compute_hazard_labels(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет физические метки из L2 FIFO истории:
      - y_adverse (N,): 1.0, если в ближайшие 5с по Bid_1 прошел встречный поток продаж и Bid_1 упал >= 1 тика.
      - y_exhaust (N,): 1.0, если цена выросла за последние 30с, но в следующие 10с не обновляет максимум и откатывает вниз.
    """
    bid_p = data["bid_p"][:, 0].astype(np.float32)
    ask_p = data["ask_p"][:, 0].astype(np.float32)
    sell_v = np.nan_to_num(data["sell_vol"].astype(np.float32), nan=0.0)
    tick = float(data.get("tick_size", 0.01))
    tol = tick * 0.25
    n = len(bid_p)

    y_adverse = np.zeros(n, dtype=np.float32)
    y_exhaust = np.zeros(n, dtype=np.float32)

    for t in range(30, n - 12):
        b0 = bid_p[t]
        a0 = ask_p[t]
        # 1. Adverse Selection: удар продавцов в [t+1..t+5] со срывом b0 вниз
        min_b_5s = float(np.min(bid_p[t + 1 : t + 6]))
        sell_5s = float(np.sum(sell_v[t + 1 : t + 6]))
        if min_b_5s <= b0 - tick + tol and (sell_5s > 10.0 or min_b_5s <= b0 - 2.0 * tick + tol):
            y_adverse[t] = 1.0

        # 2. Wave Exhaustion: локальный пик после подъема >= 1 тика за 30с
        b_prev30 = float(np.min(bid_p[t - 30 : t]))
        max_a_10s = float(np.max(ask_p[t + 1 : t + 11]))
        min_b_10s = float(np.min(bid_p[t + 1 : t + 11]))
        if (b0 - b_prev30) >= tick - tol and max_a_10s <= a0 + tol and min_b_10s <= b0 - tick + tol:
            y_exhaust[t] = 1.0

    return y_adverse, y_exhaust


class MicrostructureHazardNet(nn.Module):
    """Вентильная двухголовая сеть (GLU + ResMLP) для оценки вероятности токсичного пролива и пика волны."""

    def __init__(self, in_dim: int = 44, hidden_dim: int = 96):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gate1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head_out = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.head_out(h)



class HazardPredictor:
    """Быстрый инференс HazardNet (Global Anchor + Regime-Routed Rolling Windows) для симулятора и боевого демона."""

    def __init__(self, ckpt_path: Path = HAZARD_MODEL_PATH):
        self.ready = False
        self.mean = None
        self.std = None
        self.model = None
        self.win_experts = []
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            in_dim = int(ckpt.get("in_dim", 44))
            self.model = MicrostructureHazardNet(in_dim=in_dim)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.mean = np.array(ckpt["mean"], dtype=np.float32)
            self.std = np.array(ckpt["std"], dtype=np.float32)
            for w_item in ckpt.get("window_experts", []):
                m_w = MicrostructureHazardNet(in_dim=in_dim)
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
            probs = torch.sigmoid(self.model(torch.from_numpy(X_norm))).numpy()
            if self.win_experts:
                n = len(X_feat)
                reg_x = X_feat[:, -10:]
                w_mat = np.zeros((n, len(self.win_experts)), dtype=np.float32)
                p_wins = np.zeros((len(self.win_experts), n, 2), dtype=np.float32)
                for i, exp in enumerate(self.win_experts):
                    dist2 = np.mean(((reg_x - exp["reg_center"]) / exp["reg_scale"]) ** 2, axis=1)
                    w_mat[:, i] = exp["quality_w"] * np.exp(-0.5 * np.clip(dist2, 0.0, 25.0))
                    xw_n = np.clip((X_feat - exp["mean"]) / exp["std"], -5.0, 5.0).astype(np.float32)
                    p_wins[i] = torch.sigmoid(exp["model"](torch.from_numpy(xw_n))).numpy()
                w_norm = w_mat / (np.sum(w_mat, axis=1, keepdims=True) + 1e-8)
                p_routed = np.einsum("ni,inb->nb", w_norm, p_wins)
                p_g_m, p_g_s = np.mean(probs, axis=0, keepdims=True), np.std(probs, axis=0, keepdims=True) + 1e-6
                p_r_m, p_r_s = np.mean(p_routed, axis=0, keepdims=True), np.std(p_routed, axis=0, keepdims=True) + 1e-6
                p_routed = np.clip((p_routed - p_r_m) * (p_g_s / p_r_s) + p_g_m, 0.0, 1.0)
                probs = 0.98 * probs + 0.02 * p_routed
        return probs[:, 0], probs[:, 1]



def train_hazard_net(cache_path: str = "data/cache/rl_triad_lob_cache_prl_176h.npz", epochs: int = 8) -> dict:
    from sklearn.metrics import roc_auc_score

    raw = np.load(cache_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.keys()}
    X_all = build_carousel_features(data)
    y_adv, y_exh = compute_hazard_labels(data)
    Y_all = np.column_stack([y_adv, y_exh]).astype(np.float32)

    n_total = len(X_all)
    oos_len = 338472
    split_idx = n_total - oos_len
    tr_end = split_idx - 30

    X_tr, Y_tr = X_all[:tr_end], Y_all[:tr_end]
    X_te, Y_te = X_all[split_idx:-15], Y_all[split_idx:-15]

    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0) + 1e-5
    X_tr_n = np.clip((X_tr - mean) / std, -5.0, 5.0).astype(np.float32)
    X_te_n = np.clip((X_te - mean) / std, -5.0, 5.0).astype(np.float32)

    from _tools.system_resources import configure_blackwell_gpu
    configure_blackwell_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MicrostructureHazardNet(in_dim=X_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    pos_rates = np.clip(np.mean(Y_tr, axis=0), 0.005, 0.99)
    pos_weight = torch.tensor((1.0 - pos_rates) / pos_rates, dtype=torch.float32, device=device).clamp(1.0, 15.0)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    batch_size = 8192
    n_tr = len(X_tr_n)
    X_tr_g = torch.from_numpy(X_tr_n).to(device)
    Y_tr_g = torch.from_numpy(Y_tr).to(device)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, batch_size):
            b_idx = perm[s : s + batch_size]
            xb = X_tr_g[b_idx]
            yb = Y_tr_g[b_idx]
            opt.zero_grad()
            loss = bce_fn(model(xb), yb)
            loss.backward()
            opt.step()


    model.eval()
    win_len = 90000
    win_step = 40000
    win_experts_payload = []
    win_starts = list(range(0, max(1, tr_end - win_len + 1), win_step))
    if win_starts[-1] != max(0, tr_end - win_len):
        win_starts.append(max(0, tr_end - win_len))
    for w_i, ws in enumerate(win_starts):
        we = min(tr_end, ws + win_len)
        Xw, Yw = X_tr[ws:we], Y_tr[ws:we]
        mw = np.mean(Xw, axis=0)
        sw = np.std(Xw, axis=0) + 1e-5
        Xw_n = np.clip((Xw - mw) / sw, -5.0, 5.0).astype(np.float32)
        reg_c = np.mean(Xw[:, -10:], axis=0)
        reg_s = np.std(Xw[:, -10:], axis=0) + 1e-4

        m_win = MicrostructureHazardNet(in_dim=X_all.shape[1]).to(device)
        opt_w = torch.optim.AdamW(m_win.parameters(), lr=2.5e-3, weight_decay=1e-4)
        Xw_g = torch.from_numpy(Xw_n).to(device)
        Yw_g = torch.from_numpy(Yw).to(device)
        nw = len(Xw_n)
        for _ in range(epochs):
            m_win.train()
            p_w = torch.randperm(nw, device=device)
            for s in range(0, nw, batch_size):
                bi = p_w[s : s + batch_size]
                opt_w.zero_grad()
                l_w = bce_fn(m_win(Xw_g[bi]), Yw_g[bi])
                l_w.backward()
                opt_w.step()
        m_win.eval()
        win_experts_payload.append({
            "win_idx": w_i,
            "mean": mw.tolist(),
            "std": sw.tolist(),
            "reg_center": reg_c.tolist(),
            "reg_scale": reg_s.tolist(),
            "quality_w": 1.0,
            "state_dict": m_win.cpu().state_dict(),
        })

    payload = {
        "in_dim": X_all.shape[1],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "state_dict": model.cpu().state_dict(),
        "window_experts": win_experts_payload,
    }
    save_p = HAZARD_MODEL_PATH
    try:
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_p)
    except OSError:
        save_p = Path("/tmp/hazard_net.pt")
        torch.save(payload, save_p)

    pred_ens = HazardPredictor(save_p)
    p_adv_te, p_exh_te = pred_ens.predict_batch(X_te)
    auc_adv = round(float(roc_auc_score(Y_te[:, 0], p_adv_te)), 4)
    auc_exh = round(float(roc_auc_score(Y_te[:, 1], p_exh_te)), 4)
    payload["oos_metrics"] = {"auc_adverse_5s": auc_adv, "auc_exhaustion_10s": auc_exh, "n_windows": len(win_experts_payload)}
    torch.save(payload, save_p)
    print(f"✅ HazardNet ({1 + len(win_experts_payload)} экспертов) сохранена в {save_p} | OOS AUC(Adverse 5s)={auc_adv:.4f} | AUC(Exhaustion 10s)={auc_exh:.4f}")
    return payload["oos_metrics"]



if __name__ == "__main__":
    train_hazard_net()
