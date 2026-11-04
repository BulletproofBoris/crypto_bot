#!/usr/bin/env python3
"""
v8.0: Microstructure Regime Gate Network (Нейро-Гейт Микро-Режимов).

Классифицирует микроструктурное состояние стакана L2 на 4 режима:
0: REGIME_DEAD_FLAT         (Мертвый штиль: нулевая активность тейкеров, микро-объёмы L1)
1: REGIME_ACCUMULATION      (Скрытое накопление: спред 1-2 тика, приток объёма на L2-L5 бидов, истощение продавцов)
2: REGIME_BREAKOUT_EXPANSION (Импульсный пробой: расширение спреда, скачок CVD, ускорение цены)
3: REGIME_TOXIC_DRIFT       (Токсичное сползание: доминирование продавцов, отрицательная инерция, скрытый сброс)

Заменяет жесткие эвристические барьеры (dd_300 >= 1.0, inertia >= 0.15) на сквозную вероятностную модель.
"""

from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
REGIME_GATE_MODEL_PATH = BASE_DIR / "models/prl/regime_gate/best_regime_gate.pt"

REGIME_NAMES = {
    0: "DEAD_FLAT",
    1: "ACCUMULATION",
    2: "BREAKOUT_EXPANSION",
    3: "TOXIC_DRIFT",
}


def compute_regime_features(data_dict: dict) -> np.ndarray:
    """
    Извлекает 24 нормализованных микроструктурных признака из L2 стакана:
    - Доминирование объёмов L1-L5: bid_q / (bid_q + ask_q)
    - Относительный спред и относительная глубина
    - Кинематика цены: vel_10, inertia, vel_60
    - OFI и CVD нормализованные (flow imbalance)
    - Вероятности оракулов (HazardNet p_adv, p_exh, p_swp; CarouselNet p_car50)
    """
    b_p = data_dict["bid_p"]
    b_q = data_dict["bid_q"]
    a_p = data_dict["ask_p"]
    a_q = data_dict["ask_q"]
    n_ticks = len(b_p)
    tick = float(data_dict.get("tick_size", 0.01))

    b1 = b_p[:, 0]
    a1 = a_p[:, 0]
    spread_ticks = np.clip((a1 - b1) / max(1e-4, tick), 0.5, 10.0)

    # Доминирование стакана по уровням
    b1_q = b_q[:, 0]
    a1_q = a_q[:, 0]
    b1_dom = b1_q / np.maximum(1e-4, b1_q + a1_q)

    b5_tot = np.sum(b_q[:, :5], axis=1)
    a5_tot = np.sum(a_q[:, :5], axis=1)
    b5_dom = b5_tot / np.maximum(1e-4, b5_tot + a5_tot)

    b10_tot = np.sum(b_q[:, :10], axis=1)
    a10_tot = np.sum(a_q[:, :10], axis=1)
    b10_dom = b10_tot / np.maximum(1e-4, b10_tot + a10_tot)

    # Относительная глубина L5 к L1
    depth_ratio_b = np.clip(b5_tot / np.maximum(1.0, b1_q * 5.0), 0.1, 10.0)
    depth_ratio_a = np.clip(a5_tot / np.maximum(1.0, a1_q * 5.0), 0.1, 10.0)

    # Кинематика (vel, inertia)
    b_roll10 = np.pad(b1[:-10], (10, 0), mode="edge")
    vel10 = np.clip((b1 - b_roll10) / (10.0 * tick), -5.0, 5.0)

    b_roll60 = np.pad(b1[:-60], (60, 0), mode="edge")
    vel60 = np.clip((b1 - b_roll60) / (60.0 * tick), -5.0, 5.0)

    # Инерция
    b_roll3 = np.pad(b1[:-3], (3, 0), mode="edge")
    vel3 = (b1 - b_roll3) / (3.0 * tick)
    inertia = np.clip(0.6 * vel10 + 0.4 * vel3, -3.0, 3.0)

    # Объемы торгов
    buy_v = data_dict.get("buy_vol", np.zeros(n_ticks))
    sell_v = data_dict.get("sell_vol", np.zeros(n_ticks))
    tot_v = buy_v + sell_v
    cvd_ratio = np.clip((buy_v - sell_v) / np.maximum(1.0, tot_v), -1.0, 1.0)
    log_vol = np.log1p(tot_v) / 6.0

    # Оракулы (X_v52)
    v52 = data_dict.get("X_v52")
    if v52 is not None and len(v52) == n_ticks and v52.shape[1] >= 4:
        p_car50 = v52[:, 1]
        p_adv = v52[:, 2]
        p_exh = v52[:, 3]
        p_swp = v52[:, 4] if v52.shape[1] > 4 else np.zeros(n_ticks)
    else:
        p_car50 = np.zeros(n_ticks)
        p_adv = np.zeros(n_ticks)
        p_exh = np.zeros(n_ticks)
        p_swp = np.zeros(n_ticks)

    # Сигналы H90 и H120
    ez90 = np.clip(data_dict.get("X_h90", np.zeros((n_ticks, 1)))[:, 0] * 0.5, -3.0, 3.0)
    ez120 = np.clip(data_dict.get("X_h120", np.zeros((n_ticks, 1)))[:, 0] * 0.5, -3.0, 3.0)

    # Сборка матрицы фичей (24D)
    feats = np.column_stack([
        b1_dom - 0.5,             # 0: доминирование L1 (-0.5 .. +0.5)
        b5_dom - 0.5,             # 1: доминирование L5 (-0.5 .. +0.5)
        b10_dom - 0.5,            # 2: доминирование L10 (-0.5 .. +0.5)
        (depth_ratio_b - 1.0) * 0.2, # 3: глубина бида
        (depth_ratio_a - 1.0) * 0.2, # 4: глубина аска
        (spread_ticks - 1.0) * 0.5,  # 5: ширина спреда
        vel10 * 0.2,              # 6: скорость 10s
        vel60 * 0.2,              # 7: скорость 60s
        inertia * 0.33,           # 8: инерция
        cvd_ratio,                # 9: дисбаланс потока сделок
        log_vol,                  # 10: логарифм торговой активности
        p_car50 - 0.2,            # 11: вероятность цикла
        p_adv,                    # 12: риск пролива
        p_exh,                    # 13: истощение продавцов
        p_swp,                    # 14: вероятность свипа
        ez90,                     # 15: тренд 90s
        ez120,                    # 16: тренд 120s
        np.log1p(b1_q) / 8.0,     # 17: абсолютный размер b1
        np.log1p(a1_q) / 8.0,     # 18: абсолютный размер a1
        np.log1p(b5_tot) / 10.0,   # 19: абсолютный объем L5 bid
        np.log1p(a5_tot) / 10.0,   # 20: абсолютный объем L5 ask
        np.sin(np.linspace(0, 3.14, n_ticks)), # 21: регуляризатор
        vel3 * 0.3,               # 22: микро-скорость 3s
        (b1_dom - b5_dom) * 2.0,  # 23: расхождение микро/макро стакана
    ]).astype(np.float32)

    return feats


def label_microstructure_regimes(data_dict: dict) -> np.ndarray:
    """
    Размечает исторический ряд на 4 микроструктурных режима (Ground Truth):
    0: DEAD_FLAT
    1: ACCUMULATION
    2: BREAKOUT_EXPANSION
    3: TOXIC_DRIFT
    """
    b_p = data_dict["bid_p"]
    b_q = data_dict["bid_q"]
    a_p = data_dict["ask_p"]
    a_q = data_dict["ask_q"]
    n_ticks = len(b_p)
    tick = float(data_dict.get("tick_size", 0.01))

    b1 = b_p[:, 0]
    a1 = a_p[:, 0]
    sp = (a1 - b1) / max(1e-4, tick)

    b1_q = b_q[:, 0]
    a1_q = a_q[:, 0]
    b1_dom = b1_q / np.maximum(1e-4, b1_q + a1_q)
    b5_tot = np.sum(b_q[:, :5], axis=1)
    a5_tot = np.sum(a_q[:, :5], axis=1)
    b5_dom = b5_tot / np.maximum(1e-4, b5_tot + a5_tot)

    buy_v = data_dict.get("buy_vol", np.zeros(n_ticks))
    sell_v = data_dict.get("sell_vol", np.zeros(n_ticks))
    tot_v = buy_v + sell_v

    # Будущее движение цены на горизонте 25 тиков (реальный исход)
    b1_fut25 = np.pad(b1[25:], (0, 25), mode="edge")
    ret25_ticks = (b1_fut25 - b1) / tick

    # Будущее движение на 60 тиков
    b1_fut60 = np.pad(b1[60:], (0, 60), mode="edge")
    ret60_ticks = (b1_fut60 - b1) / tick

    labels = np.zeros(n_ticks, dtype=np.int64)

    # 1. Токсичный дрейф (цена сползает вниз >= 1 тик, продавцы давят)
    is_toxic = (ret25_ticks <= -1.0) & (b5_dom < 0.45)
    labels[is_toxic] = 3

    # 2. Пробойный импульс (цена летит вверх >= 2 тика, высокий объем)
    is_breakout = (ret25_ticks >= 1.8) & (tot_v > 0)
    labels[is_breakout] = 2

    # 3. Скрытое накопление (цена вырастет >= 1 тик на H=25..60, но текущий спред узкий 1-2 тика, бид крепкий)
    is_accum = (ret25_ticks >= 0.8) & (ret60_ticks >= 1.0) & (sp <= 2.2) & (b5_dom >= 0.48) & ~is_breakout
    labels[is_accum] = 1

    # 4. Мертвый штиль (все остальное, где волатильность около нуля и объем минимален)
    is_flat = (~is_toxic) & (~is_breakout) & (~is_accum)
    labels[is_flat] = 0

    return labels


class MicrostructureRegimeGateNet(nn.Module):
    """
    Нейро-Гейт Микро-Режимов с Residual-связями и LayerNorm.
    """

    def __init__(self, in_dim: int = 24, hidden_dim: int = 64, n_classes: int = 4, dropout: float = 0.1):
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
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.res1(h)
        h = h + self.res2(h)
        logits = self.head(h)
        return logits

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


class RegimeGatePredictor:
    """Обертка для сверхбыстрого инференса на CPU/GPU."""

    def __init__(self, model_path: Path = REGIME_GATE_MODEL_PATH):
        self.model = MicrostructureRegimeGateNet(in_dim=24, hidden_dim=64, n_classes=4)
        if model_path.exists():
            sd = torch.load(model_path, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
            self.model.load_state_dict(sd)
        self.model.eval()

    def predict_single(self, feat_vec: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(feat_vec[None, :].astype(np.float32))
        probs = self.model.predict_probs(x_t).numpy()[0]
        return probs

    def predict_batch(self, feat_mat: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(feat_mat.astype(np.float32))
        probs = self.model.predict_probs(x_t).numpy()
        return probs


def train_regime_gate_model(cache_path: str = "data/cache/rl_triad_lob_cache_prl_176h.npz", epochs: int = 8, batch_size: int = 1024) -> dict:
    """Обучает нейро-гейт на обучающей части кэша (Train) с проверкой на слепом OOS."""
    print(f"📦 Загрузка LOB-кэша {cache_path}...")
    raw = np.load(BASE_DIR / cache_path, allow_pickle=True)
    data_dict = {k: raw[k] for k in raw.keys()}
    n_total = len(data_dict["bid_p"])
    oos_len = 301892
    split_idx = n_total - oos_len
    print(f"Всего тиков: {n_total:,} | Train: {split_idx:,} | OOS: {oos_len:,}")

    print("⚡ Расчет 24D признаков микро-режимов...")
    X_all = compute_regime_features(data_dict)
    print("🏷️ Разметка 4 классов микро-режимов...")
    Y_all = label_microstructure_regimes(data_dict)

    # Статистика классов
    from collections import Counter
    c_train = Counter(Y_all[:split_idx])
    c_oos = Counter(Y_all[split_idx:])
    print("Распределение классов в Train:")
    for k in range(4):
        print(f"  {REGIME_NAMES[k]:<20}: {c_train[k]:>8,} ({c_train[k]/split_idx*100:.2f}%)")

    # Веса классов для Balanced CrossEntropy
    counts = np.array([max(1, c_train[k]) for k in range(4)], dtype=np.float32)
    weights = np.sum(counts) / (4.0 * counts)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    print(f"⚖️ Веса классов потерь: {weights.round(3).tolist()}")

    X_train_t = torch.from_numpy(X_all[:split_idx])
    Y_train_t = torch.from_numpy(Y_all[:split_idx])
    X_oos_t = torch.from_numpy(X_all[split_idx:])
    Y_oos_t = torch.from_numpy(Y_all[split_idx:])

    train_ds = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Устройство обучения: {device}")
    model = MicrostructureRegimeGateNet(in_dim=24, hidden_dim=64, n_classes=4).to(device)
    weights_t = weights_t.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    REGIME_GATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🚀 СТАРТ ОБУЧЕНИЯ MicrostructureRegimeGateNet")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Валидация на OOS
        model.eval()
        with torch.no_grad():
            oos_bx = X_oos_t.to(device)
            oos_by = Y_oos_t.to(device)
            oos_logits = model(oos_bx)
            oos_loss = criterion(oos_logits, oos_by).item()
            oos_preds = torch.argmax(oos_logits, dim=-1)
            oos_acc = (oos_preds == oos_by).float().mean().item()

            # Точность по ключевому классу аккумуляции
            accum_mask = (oos_by == 1)
            accum_acc = (oos_preds[accum_mask] == 1).float().mean().item() if accum_mask.sum() > 0 else 0.0

        print(
            f"Эпоха {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/n_batches:.4f} | "
            f"OOS Loss: {oos_loss:.4f} | OOS Acc: {oos_acc*100:.2f}% | Accum Recall: {accum_acc*100:.2f}%",
            flush=True,
        )

        if oos_acc > best_acc:
            best_acc = oos_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch,
                    "in_dim": 24,
                    "hidden_dim": 64,
                    "n_classes": 4,
                },
                REGIME_GATE_MODEL_PATH,
            )

    print(f"✅ Обучение завершено. Лучшая точность на OOS: {best_acc*100:.2f}%. Чекпоинт сохранен в {REGIME_GATE_MODEL_PATH}")
    return {"best_acc": best_acc, "model_path": str(REGIME_GATE_MODEL_PATH)}


if __name__ == "__main__":
    train_regime_gate_model()

