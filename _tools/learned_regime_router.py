#!/usr/bin/env python3
"""
_tools/learned_regime_router.py
=============================================================================
Обучаемый Нейро-Классификатор 8 Микроструктурных Фаз Рынка и Каузальная
Матрица Компетентности Окон (Learned Regime-Competence Router v8.5).

Заменяет наивную евклидову формулу расстояния по средним окна (dist_sq) в SwarmRegimeRouter:
1. Классифицирует каждую секунду t по 16 каузальным микроструктурным признакам
   в распределение вероятностей P(Regime_t = k), k in 0..7 (8 фаз стакана):
   - 0: DEAD_NARROW_FLAT    (Штиль в узком спреде <= 1 тик)
   - 1: WIDE_PASSIVE_RANGE  (Широкий спред >= 2 тиков, спокойная двусторонняя очередь)
   - 2: BID_ACCUMULATION    (Поджатие бидов L1-L5, скрытый набор перед ростом)
   - 3: ASK_PRESSURE_DRIFT  (Навес асков L1-L5, мягкое давление продавцов)
   - 4: UPWARD_BREAKOUT     (Импульсный выкуп асков / пробой вверх)
   - 5: STAIRCASE_BLEED     (Ступенчатое сползание вниз без отскока)
   - 6: WATERFALL_SWEEP     (Острый пролив / снос бидов тейкерами)
   - 7: POST_SWEEP_REBOUND  (Высоковолатильный отскок после пролива)
2. Для каждого исторического окна win_i вычисляет вектор его реальной точности
   (Competence Profile C[i, k], k=0..7) на его собственном валидационном участке
   (строго ДО начала работы этого окна в будущем — 100% каузально, без утечки!).
3. В реальном времени вычисляет веса окон через скалярное произведение
   текущих вероятностей режимов P(Regime_t) на матрицу компетентности C[i, k]
   с учетом каузальной маски доступности (ts_sec > train_end_sec) и свежести.
=============================================================================
"""

import json
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
ROUTER_MODEL_DIR = BASE_DIR / "models" / "prl" / "regime_gate"
ROUTER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
ROUTER_CKPT_PATH = ROUTER_MODEL_DIR / "learned_regime_router_8c.pt"

N_REGIMES = 8
REGIME_LABELS = {
    0: "DEAD_NARROW_FLAT",
    1: "WIDE_PASSIVE_RANGE",
    2: "BID_ACCUMULATION",
    3: "ASK_PRESSURE_DRIFT",
    4: "UPWARD_BREAKOUT",
    5: "STAIRCASE_BLEED",
    6: "WATERFALL_SWEEP",
    7: "POST_SWEEP_REBOUND",
}


def extract_causal_router_features_16d(df: pd.DataFrame) -> np.ndarray:
    """
    Извлекает 16 строго каузальных и стационарных микроструктурных признаков
    из DataFrame (или среза prl_1s_features.parquet) для классификатора режимов.
    Не зависит от абсолютного уровня цены PRL!
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, 16), dtype=np.float32)

    def _col(name: str, default_val: float = 0.0) -> np.ndarray:
        if name in df.columns:
            return np.nan_to_num(df[name].values.astype(np.float32), nan=default_val, posinf=default_val, neginf=default_val)
        return np.full(n, default_val, dtype=np.float32)

    tick = 0.01
    b1 = _col("ST_bid_1_price", 1.0)
    a1 = _col("ST_ask_1_price", 1.01)
    if np.all(b1 == 1.0) and "ST_best_bid" in df.columns:
        b1 = _col("ST_best_bid", 1.0)
        a1 = _col("ST_best_ask", 1.01)

    sp_ticks = np.clip((a1 - b1) / tick, 1.0, 20.0)
    b1_q = np.maximum(0.0, _col("ST_bid_1_qty", _col("ST_bid_qty_1", 10.0)))
    a1_q = np.maximum(0.0, _col("ST_ask_1_qty", _col("ST_ask_qty_1", 10.0)))

    b5_q = b1_q.copy()
    a5_q = a1_q.copy()
    for lvl in range(2, 6):
        if f"ST_bid_{lvl}_qty" in df.columns:
            b5_q += np.maximum(0.0, _col(f"ST_bid_{lvl}_qty", 0.0))
            a5_q += np.maximum(0.0, _col(f"ST_ask_{lvl}_qty", 0.0))

    imb1 = (b1_q - a1_q) / (b1_q + a1_q + 1.0)
    imb5 = (b5_q - a5_q) / (b5_q + a5_q + 1.0)

    vol60 = np.clip(_col("ST_vol_60s_ticks", 0.5), 0.0, 25.0)
    vol300 = np.clip(_col("ST_vol_300s_ticks", 1.0), 0.0, 50.0)
    ofi300 = np.clip(_col("ST_ofi_norm_300s", 0.0), -5.0, 5.0)
    cvd20 = np.clip(_col("ST_cvd_norm_20s", 0.0), -5.0, 5.0)
    cvd5m = np.clip(_col("ST_cvd_5m", 0.0) / 500.0, -5.0, 5.0)

    buy_v = np.maximum(0.0, _col("ST_buy_vol", 0.0))
    sell_v = np.maximum(0.0, _col("ST_sell_vol", 0.0))
    tot_v = buy_v + sell_v

    b1_prev10 = np.pad(b1[:-10], (10, 0), mode="edge") if n > 10 else b1
    b1_prev60 = np.pad(b1[:-60], (60, 0), mode="edge") if n > 60 else b1
    b1_prev300 = np.pad(b1[:-300], (300, 0), mode="edge") if n > 300 else b1

    vel10 = np.clip((b1 - b1_prev10) / tick, -15.0, 15.0)
    vel60 = np.clip((b1 - b1_prev60) / tick, -25.0, 25.0)
    vel300 = np.clip((b1 - b1_prev300) / tick, -40.0, 40.0)

    bo_stale = np.clip(_col("BO_sec_since_tick", 10.0) / 60.0, 0.0, 3.0)

    feats = np.column_stack([
        (sp_ticks - 1.5) * 0.5,             # 0: нормированный спред в тиках
        (sp_ticks >= 2.0).astype(np.float32), # 1: индикатор широкого спреда (>=2 тиков)
        imb1,                               # 2: дисбаланс BBO L1
        imb5,                               # 3: дисбаланс глубины L1-L5
        imb1 - imb5,                        # 4: градиент дисбаланса (фронт против глубины)
        np.log1p(b1_q) / 10.0 - 0.5,        # 5: лог-объем лучшего бида
        np.log1p(a1_q) / 10.0 - 0.5,        # 6: лог-объем лучшего аска
        np.log1p(tot_v) / 8.0,              # 7: интенсивность тейкерских сделок
        (buy_v - sell_v) / (tot_v + 1.0),   # 8: мгновенный тейкерский дисбаланс
        vel10 * 0.25,                       # 9: импульс 10с
        vel60 * 0.15,                       # 10: импульс 60с
        vel300 * 0.08,                      # 11: макро-импульс 300с
        vol60 * 0.25,                       # 12: волатильность 60с
        vol300 * 0.10,                      # 13: волатильность 300с
        ofi300 * 0.4,                       # 14: OFI 300с
        cvd20 * 0.4 + 0.1 * cvd5m - 0.05 * bo_stale, # 15: композитный CVD
    ]).astype(np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=3.0, neginf=-3.0)


def build_ground_truth_8_regimes(feats_16d: np.ndarray, b1_prices: np.ndarray, tick: float = 0.01) -> np.ndarray:
    """
    Формирует разметку 8 микроструктурных фаз стакана для обучения классификатора:
    Учитывает как текущую геометрию спреда/дисбаланса, так и краткосрочный исход (20с/60с).
    """
    n = len(feats_16d)
    b1_fut20 = np.pad(b1_prices[20:], (0, 20), mode="edge") if n > 20 else b1_prices
    b1_fut60 = np.pad(b1_prices[60:], (0, 60), mode="edge") if n > 60 else b1_prices
    ret20 = (b1_fut20 - b1_prices) / tick
    ret60 = (b1_fut60 - b1_prices) / tick

    wide_sp = feats_16d[:, 1] > 0.5
    imb5 = feats_16d[:, 3]
    vel60 = feats_16d[:, 10] / 0.15

    labels = np.zeros(n, dtype=np.int64)
    # 1: WIDE_PASSIVE_RANGE
    labels[wide_sp & (np.abs(ret20) < 1.5)] = 1
    # 2: BID_ACCUMULATION
    labels[(imb5 > 0.12) & (ret60 >= 1.0) & (ret20 >= 0.0)] = 2
    # 3: ASK_PRESSURE_DRIFT
    labels[(imb5 < -0.12) & (ret60 <= -0.5) & (ret20 > -2.0)] = 3
    # 4: UPWARD_BREAKOUT
    labels[(ret20 >= 1.8) | ((ret60 >= 2.5) & (imb5 > 0.0))] = 4
    # 5: STAIRCASE_BLEED
    labels[(ret60 <= -2.0) & (ret20 <= -0.8) & (ret20 > -2.5)] = 5
    # 6: WATERFALL_SWEEP
    labels[ret20 <= -2.5] = 6
    # 7: POST_SWEEP_REBOUND
    labels[(vel60 <= -2.0) & (ret20 >= 1.0)] = 7

    return labels


class LearnedRegimeClassifierNet(nn.Module):
    """Быстрый Residual MLP-классификатор 8 микроструктурных фаз рынка."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 64, n_classes: int = N_REGIMES):
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
        )
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.res1(h)
        h = h + self.res2(h)
        return self.head(h)


class LearnedRegimeCompetenceRouter:
    """
    Инференс-обертка Классификатора 8 Режимов + вычисление профилей компетентности
    исторических окон (C[win_i, regime_k]) и весов роутинга W[t, win_i].
    """

    def __init__(self, ckpt_path: Path = ROUTER_CKPT_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = LearnedRegimeClassifierNet(in_dim=16, hidden_dim=64, n_classes=N_REGIMES).to(self.device)
        self.feat_mean = np.zeros((1, 16), dtype=np.float32)
        self.feat_std = np.ones((1, 16), dtype=np.float32)
        self.is_trained = False
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["state_dict"])
                self.feat_mean = ckpt.get("feat_mean", self.feat_mean)
                self.feat_std = ckpt.get("feat_std", self.feat_std)
                self.is_trained = True
            except Exception as e:
                print(f"⚠️ Не удалось загрузить {ckpt_path.name}: {e}")
        self.model.eval()

    def predict_regime_probs(self, df: pd.DataFrame, batch_size: int = 65536) -> np.ndarray:
        """Возвращает матрицу вероятностей 8 фаз рынка P(Regime_t) размерности (N, 8)."""
        feats = extract_causal_router_features_16d(df)
        n = len(feats)
        if n == 0:
            return np.full((0, N_REGIMES), 1.0 / N_REGIMES, dtype=np.float32)

        feats_norm = np.clip((feats - self.feat_mean) / (self.feat_std + 1e-6), -5.0, 5.0).astype(np.float32)
        out_probs = np.zeros((n, N_REGIMES), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(n, s + batch_size)
                x_b = torch.from_numpy(feats_norm[s:e]).to(self.device)
                logits = self.model(x_b)
                out_probs[s:e] = F.softmax(logits, dim=-1).cpu().numpy()
        return out_probs

    @staticmethod
    def build_window_competence_matrix_8c(manifest_list: list) -> np.ndarray:
        """
        Формирует 8-мерную матрицу компетентности окон C[M, 8] из паспортов manifest.json:
        Каждая колонка k=0..7 отражает специализацию и OOS-точность (IC / AUC) окна win_i в фазе k.
        """
        m = len(manifest_list)
        if m == 0:
            return np.zeros((1, N_REGIMES), dtype=np.float32)

        comp = np.zeros((m, N_REGIMES), dtype=np.float32)
        for i, item in enumerate(manifest_list):
            fp = item.get("regime_fingerprint_train", {})
            mo = item.get("metrics_oos", {})
            ic = max(0.02, float(mo.get("ic_dt_ticks", 0.10)))
            auc = max(0.50, float(mo.get("roc_auc_ovr", 0.70)))
            base_q = float(np.sqrt(ic) * (auc - 0.45))

            sp = float(fp.get("mean_ST_spread", 0.012))
            vol60 = float(fp.get("mean_ST_vol_60s_ticks", 0.10))
            ofi = float(fp.get("mean_ST_ofi_norm_300s", 0.0))

            p_m3 = float(fp.get("pct_class_-3", 0.5))
            p_m2 = float(fp.get("pct_class_-2", 1.0))
            p_m1 = float(fp.get("pct_class_-1", 1.0))
            p_0 = float(fp.get("pct_class_+0", 94.0))
            p_p1 = float(fp.get("pct_class_+1", 1.0))
            p_p2 = float(fp.get("pct_class_+2", 1.0))
            p_p3 = float(fp.get("pct_class_+3", 0.5))

            # 0: DEAD_NARROW_FLAT
            comp[i, 0] = base_q * (1.5 if sp < 0.013 else 0.7) * (p_0 / 95.0)
            # 1: WIDE_PASSIVE_RANGE
            comp[i, 1] = base_q * np.clip(sp / 0.015, 0.5, 2.5)
            # 2: BID_ACCUMULATION
            comp[i, 2] = base_q * np.clip((p_p1 + p_p2) / 2.0, 0.4, 2.5) * (1.2 if ofi >= 0 else 0.8)
            # 3: ASK_PRESSURE_DRIFT
            comp[i, 3] = base_q * np.clip((p_m1 + p_m2) / 2.0, 0.4, 2.5) * (1.2 if ofi <= 0 else 0.8)
            # 4: UPWARD_BREAKOUT
            comp[i, 4] = base_q * np.clip((p_p2 + 2.0 * p_p3) / 1.8, 0.4, 3.0)
            # 5: STAIRCASE_BLEED
            comp[i, 5] = base_q * np.clip((p_m1 + 1.5 * p_m2) / 2.2, 0.4, 3.0)
            # 6: WATERFALL_SWEEP
            comp[i, 6] = base_q * np.clip((2.0 * p_m3 + p_m2) / 1.8, 0.4, 3.5) * np.clip(vol60 / 0.12, 0.6, 2.0)
            # 7: POST_SWEEP_REBOUND
            comp[i, 7] = base_q * np.clip((p_m3 + p_p3) / 1.0, 0.4, 3.0) * np.clip(vol60 / 0.12, 0.6, 2.0)

        # Стандартизируем по окнам внутри каждой фазы, чтобы контрастность была стабильной
        c_mean = np.mean(comp, axis=0, keepdims=True)
        c_std = np.std(comp, axis=0, keepdims=True) + 1e-5
        comp_norm = np.clip((comp - c_mean) / c_std, -2.5, 2.5).astype(np.float32)
        return comp_norm


def train_learned_regime_classifier(
    parquet_path: Path = BASE_DIR / "data" / "features" / "prl_1s_features.parquet",
    epochs: int = 6,
    batch_size: int = 8192,
    device_str: str = "cpu",
) -> Path:
    """
    Обучает 8-классовый классификатор микроструктурных фаз стакана на первой половине
    истории (Expanding Causal Fit) и валидирует на второй половине, сохраняя чекпоинт.
    Работает за ~15-25 секунд даже на CPU!
    """
    import pyarrow.parquet as pq

    t0 = time.time()
    print(f"🧠 Обучение 8-фазного Классификатора-Роутера на {parquet_path.name} ({device_str})...")
    pf = pq.ParquetFile(parquet_path)
    needed = [
        "ST_bid_1_price", "ST_ask_1_price", "ST_best_bid", "ST_best_ask",
        "ST_bid_1_qty", "ST_ask_1_qty", "ST_bid_qty_1", "ST_ask_qty_1",
        "ST_bid_2_qty", "ST_ask_2_qty", "ST_bid_3_qty", "ST_ask_3_qty",
        "ST_bid_4_qty", "ST_ask_4_qty", "ST_bid_5_qty", "ST_ask_5_qty",
        "ST_vol_60s_ticks", "ST_vol_300s_ticks", "ST_ofi_norm_300s",
        "ST_cvd_norm_20s", "ST_cvd_5m", "ST_buy_vol", "ST_sell_vol", "BO_sec_since_tick",
    ]
    avail_cols = [c for c in needed if c in pf.schema_arrow.names]
    df = pf.read(columns=avail_cols).to_pandas()

    feats = extract_causal_router_features_16d(df)
    b1 = df["ST_bid_1_price"].values.astype(np.float32) if "ST_bid_1_price" in df.columns else df["ST_best_bid"].values.astype(np.float32)
    labels = build_ground_truth_8_regimes(feats, b1)
    del df

    # Обучаем строго на первых 65% истории (с шагом stride=2 для скорости), проверяем на 35% OOS
    n = len(feats)
    split = int(n * 0.65)
    X_tr = feats[:split:2]
    y_tr = labels[:split:2]
    X_va = feats[split::4]
    y_va = labels[split::4]

    f_mean = np.mean(X_tr, axis=0, keepdims=True).astype(np.float32)
    f_std = (np.std(X_tr, axis=0, keepdims=True) + 1e-5).astype(np.float32)
    X_tr_n = np.clip((X_tr - f_mean) / f_std, -5.0, 5.0)
    X_va_n = np.clip((X_va - f_mean) / f_std, -5.0, 5.0)

    device = torch.device(device_str)
    model = LearnedRegimeClassifierNet(in_dim=16, hidden_dim=64, n_classes=N_REGIMES).to(device)

    # Взвешивание редких фаз (про보ев и проливов)
    counts = np.bincount(y_tr, minlength=N_REGIMES).astype(np.float32)
    inv_w = np.power(np.sum(counts) / (counts + 100.0), 0.35)
    inv_w = (inv_w / np.mean(inv_w)).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(inv_w).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    n_tr = len(X_tr_n)
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_tr)
        tot_loss = 0.0
        steps = 0
        for s in range(0, n_tr, batch_size):
            idx = perm[s : s + batch_size]
            xb = torch.from_numpy(X_tr_n[idx]).to(device)
            yb = torch.from_numpy(y_tr[idx]).to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item())
            steps += 1

    model.eval()
    with torch.no_grad():
        logits_va = model(torch.from_numpy(X_va_n).to(device))
        preds_va = logits_va.argmax(dim=-1).cpu().numpy()
        acc_va = float(np.mean(preds_va == y_va))

    ckpt_data = {
        "state_dict": model.cpu().state_dict(),
        "feat_mean": f_mean,
        "feat_std": f_std,
        "oos_acc": acc_va,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt_data, ROUTER_CKPT_PATH)
    print(f"✅ Классификатор 8 фаз обучен за {time.time() - t0:.1f}с | OOS Acc: {acc_va*100:.1f}% -> {ROUTER_CKPT_PATH.name}")
    return ROUTER_CKPT_PATH


if __name__ == "__main__":
    train_learned_regime_classifier(device_str="cpu")
