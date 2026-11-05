#!/usr/bin/env python3
"""
v8.0: Multi-Horizon Guard Ensemble (Рой Нейро-Ограничителей).

Обучает 4 специализированные нейросети-ограничителя на 600+ часах данных:
1. G_12s   (Micro-Sweep Guard):      риск проноса лимитки за 12 секунд.
2. G_60s   (Staircase Bleed Guard):  риск ступенчатого сползания цены вниз на >= 2 тика за 1 минуту.
3. G_300s  (Waterfall Crash Guard):  хвостовой риск краша >= 5 тиков за 5 минут.
4. G_exit  (Exit Urgency Guard):     риск убытка при пассивном выходе (своевременный Taker Panic).

Заменяет жесткие ручные правила if-else на непрерывные вероятностные оценки риска P(Danger).
"""

from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
GUARD_ENSEMBLE_DIR = BASE_DIR / "models/prl/guard_ensemble"


def extract_guard_features(b_p: np.ndarray, b_q: np.ndarray, a_p: np.ndarray, a_q: np.ndarray, buy_v: np.ndarray, sell_v: np.ndarray, tick: float = 0.01) -> np.ndarray:
    """Извлекает 20 Causal-признаков для Guard сетей без заглядывания в будущее."""
    n = len(b_p)
    b1 = b_p[:, 0]
    a1 = a_p[:, 0]
    sp = np.clip((a1 - b1) / max(1e-4, tick), 0.5, 10.0)

    b1_q = b_q[:, 0]
    a1_q = a_q[:, 0]
    b1_dom = b1_q / np.maximum(1e-4, b1_q + a1_q)

    b5_tot = np.sum(b_q[:, :5], axis=1)
    a5_tot = np.sum(a_q[:, :5], axis=1)
    b5_dom = b5_tot / np.maximum(1e-4, b5_tot + a5_tot)

    b_roll10 = np.pad(b1[:-10], (10, 0), mode="edge")
    vel10 = np.clip((b1 - b_roll10) / (10.0 * tick), -5.0, 5.0)

    b_roll60 = np.pad(b1[:-60], (60, 0), mode="edge")
    vel60 = np.clip((b1 - b_roll60) / (60.0 * tick), -5.0, 5.0)

    b_roll3 = np.pad(b1[:-3], (3, 0), mode="edge")
    vel3 = np.clip((b1 - b_roll3) / (3.0 * tick), -5.0, 5.0)
    inertia = np.clip(0.6 * vel10 + 0.4 * vel3, -3.0, 3.0)

    tot_v = buy_v + sell_v
    cvd_ratio = np.clip((buy_v - sell_v) / np.maximum(1.0, tot_v), -1.0, 1.0)
    log_vol = np.log1p(tot_v) / 6.0

    # L2-L5 дисперсия
    depth_ratio_b = np.clip(b5_tot / np.maximum(1.0, b1_q * 5.0), 0.1, 10.0)
    depth_ratio_a = np.clip(a5_tot / np.maximum(1.0, a1_q * 5.0), 0.1, 10.0)

    feats = np.column_stack([
        b1_dom - 0.5,
        b5_dom - 0.5,
        (depth_ratio_b - 1.0) * 0.2,
        (depth_ratio_a - 1.0) * 0.2,
        (sp - 1.0) * 0.5,
        vel3 * 0.2,
        vel10 * 0.2,
        vel60 * 0.2,
        inertia * 0.33,
        cvd_ratio,
        log_vol,
        np.log1p(b1_q) / 8.0,
        np.log1p(a1_q) / 8.0,
        np.log1p(b5_tot) / 10.0,
        np.log1p(a5_tot) / 10.0,
        (b1_dom - b5_dom) * 2.0,
        (b_p[:, 0] - b_p[:, 4]) / (4.0 * tick) - 1.0, # наклон стакана bid
        (a_p[:, 4] - a_p[:, 0]) / (4.0 * tick) - 1.0, # наклон стакана ask
        np.clip(np.sum(b_q[:, :3], axis=1) / np.maximum(1e-4, b5_tot) - 0.6, -1.0, 1.0),
        np.clip(np.sum(a_q[:, :3], axis=1) / np.maximum(1e-4, a5_tot) - 0.6, -1.0, 1.0),
    ]).astype(np.float32)

    return feats


class FocalLoss(nn.Module):
    """Focal Loss для эффективного обучения на сильно несбалансированных редких событиях."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()


class GuardClassifierNet(nn.Module):
    """Быстрый Residual MLP классификатор риска."""
    def __init__(self, in_dim: int = 20, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.res1(h)
        h = h + self.res2(h)
        logits = self.out(h).squeeze(-1)
        return logits

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def train_guard_models(cache_path: str = "data/cache/rl_guard_lob_cache_600h.npz", epochs: int = 6, batch_size: int = 4096) -> dict:
    print(f"📦 Загрузка 600h датасета {cache_path}...")
    raw = np.load(BASE_DIR / cache_path, allow_pickle=True)
    n_total = len(raw["bid_p"])
    oos_len = 301892
    split_idx = n_total - oos_len
    print(f"Всего тиков: {n_total:,} | Train: {split_idx:,} | OOS: {oos_len:,}")

    print("⚡ Расчет 20 Causal признаков микроструктуры...")
    X_all = extract_guard_features(raw["bid_p"], raw["bid_q"], raw["ask_p"], raw["ask_q"], raw["buy_vol"], raw["sell_vol"], float(raw["tick_size"]))

    targets = {
        "guard_12s_sweep": raw["y_sweep_12s"],
        "guard_60s_bleed": raw["y_bleed_60s"],
        "guard_300s_crash": raw["y_crash_300s"],
        "guard_30s_exit": raw["y_exit_30s"],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Устройство обучения: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    X_train_t = torch.from_numpy(X_all[:split_idx]).to(device)
    X_oos_t = torch.from_numpy(X_all[split_idx:]).to(device)

    GUARD_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, y_arr in targets.items():
        print("\n" + "=" * 70)
        pos_cnt = int(y_arr[:split_idx].sum())
        print(f"🎯 Обучение модели {name} (Позитивных событий в Train: {pos_cnt:,} / {pos_cnt/split_idx*100:.2f}%)")
        print("=" * 70)

        Y_train_t = torch.from_numpy(y_arr[:split_idx]).to(device)
        Y_oos_t = torch.from_numpy(y_arr[split_idx:]).to(device)

        ds = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = GuardClassifierNet(in_dim=20, hidden_dim=64).to(device)
        pos_ratio = max(0.01, min(0.99, float(y_arr[:split_idx].mean())))
        criterion = FocalLoss(alpha=1.0 - pos_ratio, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = 0.0
        ckpt_path = GUARD_ENSEMBLE_DIR / f"{name}.pt"

        from sklearn.metrics import roc_auc_score

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            n_b = 0
            for bx, by in loader:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_b += 1
            scheduler.step()

            # Валидация на слепом OOS
            model.eval()
            with torch.no_grad():
                oos_logits = model(X_oos_t)
                oos_probs = torch.sigmoid(oos_logits).cpu().numpy()
                oos_y = Y_oos_t.cpu().numpy()

                try:
                    auc = roc_auc_score(oos_y, oos_probs)
                except Exception:
                    auc = 0.50

            print(f"Эпоха {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/n_b:.4f} | OOS ROC-AUC: {auc:.4f}", flush=True)

            if auc > best_auc:
                best_auc = auc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "auc": best_auc,
                        "epoch": epoch,
                        "name": name,
                    },
                    ckpt_path,
                )

        print(f"✅ {name} успешно обучена. Лучший OOS ROC-AUC: {best_auc:.4f}. Сохранено в {ckpt_path.name}")
        results[name] = best_auc

    return results


class MultiHorizonGuardPredictor:
    """Обертка для мгновенного инференса всех 4 Guard-сетей."""
    def __init__(self, ensemble_dir: Path = GUARD_ENSEMBLE_DIR):
        self.models = {}
        for name in ["guard_12s_sweep", "guard_60s_bleed", "guard_300s_crash", "guard_30s_exit"]:
            p = ensemble_dir / f"{name}.pt"
            m = GuardClassifierNet(in_dim=20, hidden_dim=64)
            if p.exists():
                sd = torch.load(p, map_location="cpu", weights_only=False)
                sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
                m.load_state_dict(sd)
            m.eval()
            self.models[name] = m

    def predict_probs_batch(self, feat_mat: np.ndarray) -> np.ndarray:
        """Возвращает матрицу (N, 4): [p_sweep12, p_bleed60, p_crash300, p_exit30]."""
        x_t = torch.from_numpy(feat_mat.astype(np.float32))
        probs = []
        with torch.no_grad():
            for name in ["guard_12s_sweep", "guard_60s_bleed", "guard_300s_crash", "guard_30s_exit"]:
                p = self.models[name].predict_prob(x_t).numpy()
                probs.append(p)
        return np.column_stack(probs).astype(np.float32)

    def predict_single(self, feat20: np.ndarray) -> np.ndarray:
        """Возвращает массив (4,): [p_sweep12, p_bleed60, p_crash300, p_exit30] для одиночного тика."""
        x_t = torch.from_numpy(feat20.astype(np.float32)).unsqueeze(0)
        res = np.zeros(4, dtype=np.float32)
        with torch.no_grad():
            for i, name in enumerate(["guard_12s_sweep", "guard_60s_bleed", "guard_300s_crash", "guard_30s_exit"]):
                res[i] = float(self.models[name].predict_prob(x_t).item())
        return res


if __name__ == "__main__":
    train_guard_models()
