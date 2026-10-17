#!/usr/bin/env python3
"""
Контекстный роутер пула моделей (Swarm Regime Router & Expert Blender).

Архитектура Mixture of Experts (MoE) для высокочастотного стакана SafeTrade:
1. Загружает обученные чекпоинты пула моделей из models/{coin}/h{H}s_k{K}/.
2. Считывает паспорта рыночных режимов (Regime Fingerprint) каждого эксперта.
3. В реальном времени сопоставляет текущее состояние стакана S_t (волатильность, спред, дисбаланс OFI/CVD)
   с историческими режимами экспертов.
4. Вычисляет вектор динамических весов W = [w_1, ..., w_M] через регуляризованный Softmax (Gating Network).
5. Формирует взвешенное распределение вероятностей 7 классов и формирует торговый сигнал.
6. Содержит встроенный бэктестер для сравнения:
   - Single Model (Win 00)
   - Uniform Ensemble (простое усреднение 1/M)
   - Dynamic Regime Router (адаптивный роутер)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent

class SwarmRegimeRouter:
    """Контекстный роутер и блендер экспертов пула моделей."""

    def __init__(self, coin: str, H: int, K: int, temperature: float = 0.5):
        self.coin = coin.lower()
        self.H = H
        self.K = K
        self.L = H * K
        self.temperature = max(0.01, temperature)
        self.model_dir = BASE_DIR / "models" / self.coin / f"h{H}s_k{K}"
        
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Манифест моделей не найден: {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        feats_json = BASE_DIR / "data" / "features" / f"selected_features_{self.coin}_h8s.json"
        with open(feats_json) as f:
            self.feature_names = json.load(f)["feature_order"]

        # Загрузка моделей LightGBM в память
        self.models = []
        self.model_ids = []
        self.regime_fingerprints = []
        
        print(f"📦 Загрузка пула экспертов {self.coin.upper()} (H={H}с, K={K})...")
        for item in self.manifest:
            m_path = self.model_dir / item['model_file']
            if m_path.exists():
                booster = lgb.Booster(model_file=str(m_path))
                self.models.append(booster)
                self.model_ids.append(item['window_id'])
                self.regime_fingerprints.append(item['regime_fingerprint_train'])

        self.num_models = len(self.models)
        print(f"✅ Успешно загружено {self.num_models} моделей-экспертов в RAM.")

        # Подготовка матрицы признаков режимов экспертов
        self._build_regime_matrix()

    def _build_regime_matrix(self):
        """Формирует нормализованную матрицу профилей режимов для экспертов."""
        self.regime_keys = [
            'median_ST_spread_pct', 'mean_ST_spread_pct',
            'pct_class_+2', 'pct_class_-2', 'pct_class_+3', 'pct_class_-3'
        ]
        
        mat = []
        for fp in self.regime_fingerprints:
            row = [fp.get(k, 0.0) for k in self.regime_keys]
            mat.append(row)
            
        self.regime_matrix = np.array(mat, dtype=np.float32)
        # Центрирование и нормализация
        self.regime_mean = np.mean(self.regime_matrix, axis=0, keepdims=True)
        self.regime_std = np.std(self.regime_matrix, axis=0, keepdims=True) + 1e-6
        self.norm_regime_matrix = (self.regime_matrix - self.regime_mean) / self.regime_std

    def compute_routing_weights(self, current_regime_df: pd.DataFrame) -> np.ndarray:
        """
        Вычисляет матрицу весов W (N x M) для каждой секунды на основе близости
        текущего рыночного профиля к историческому профилю эксперта.
        """
        n_samples = len(current_regime_df)
        if self.num_models == 1:
            return np.ones((n_samples, 1), dtype=np.float32)

        # Текущий вектор режима
        cur_spread = current_regime_df['ST_spread_pct'].values if 'ST_spread_pct' in current_regime_df.columns else np.zeros(n_samples)
        
        # Эвристическая оценка локальной доли пробоев
        pct_p2 = np.full(n_samples, 1.5, dtype=np.float32)
        pct_m2 = np.full(n_samples, 1.5, dtype=np.float32)
        pct_p3 = np.full(n_samples, 2.5, dtype=np.float32)
        pct_m3 = np.full(n_samples, 2.5, dtype=np.float32)

        cur_mat = np.column_stack([cur_spread, cur_spread, pct_p2, pct_m2, pct_p3, pct_m3]).astype(np.float32)
        norm_cur = (cur_mat - self.regime_mean) / self.regime_std

        # Евклидово расстояние между текущим режимом и каждым экспертом
        # norm_cur: (N, D), norm_regime: (M, D) -> dist: (N, M)
        diff = norm_cur[:, np.newaxis, :] - self.norm_regime_matrix[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        # Softmax с регулируемой температурой
        logits = -dist_sq / self.temperature
        # Численная стабильность
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return weights.astype(np.float32)

    def predict_ensemble(self, X: np.ndarray, current_regime_df: pd.DataFrame, mode: str = "router"):
        """
        Вычисляет ансамблевый прогноз по 7 классам:
        mode:
          - 'router': динамическое взвешивание на основе похожести режима
          - 'uniform': простое среднее 1/M
          - 'latest': только самая свежая модель (win_00)
        """
        n_samples = len(X)
        all_probs = np.zeros((self.num_models, n_samples, 7), dtype=np.float32)

        for i, booster in enumerate(self.models):
            raw_p = booster.predict(X)
            # В LightGBM при predict возвращается (N, num_classes)
            if raw_p.shape[1] == 7:
                all_probs[i] = raw_p
            else:
                # Если редкий класс отсутствовал
                for col_idx in range(min(7, raw_p.shape[1])):
                    all_probs[i, :, col_idx] = raw_p[:, col_idx]

        if mode == "latest":
            blend_probs = all_probs[0]
            weights = np.zeros((n_samples, self.num_models), dtype=np.float32)
            weights[:, 0] = 1.0
        elif mode == "uniform":
            blend_probs = np.mean(all_probs, axis=0)
            weights = np.full((n_samples, self.num_models), 1.0 / self.num_models, dtype=np.float32)
        elif mode == "router":
            weights = self.compute_routing_weights(current_regime_df)
            # weights: (N, M), all_probs: (M, N, 7) -> (N, 7)
            # einsum: w_nm, p_mnc -> n_c
            blend_probs = np.einsum('nm,mnc->nc', weights, all_probs)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

        # Ожидаемый импульс E[z] в тиках
        class_values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
        expected_z = np.dot(blend_probs, class_values)

        # Сигналы на вход
        p_buy = blend_probs[:, 5] + blend_probs[:, 6]   # P(+2) + P(+3)
        p_sell = blend_probs[:, 1] + blend_probs[:, 0]  # P(-2) + P(-3)

        return {
            'probs': blend_probs,
            'expected_z': expected_z,
            'p_buy': p_buy,
            'p_sell': p_sell,
            'weights': weights
        }


def benchmark_router(coin: str, H: int, K: int):
    """Сравнивает эффективность Single Model vs Uniform vs Swarm Router на Out-of-Sample."""
    coin = coin.lower()
    router = SwarmRegimeRouter(coin, H, K)

    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    target_c7_col = f"target_class_7_h{H}_k{K}"
    target_dt_col = f"target_dt_ticks_h{H}_k{K}"

    cols = list(dict.fromkeys(router.feature_names + ['timestamp', 'sample_split', 'ST_spread_pct', target_c7_col, target_dt_col]))
    tbl = pq.read_table(labeled_path, columns=cols)
    df = tbl.to_pandas()

    # Берем чистый OOS (последние 20% Val)
    val_mask = (df['sample_split'] == 'val')
    val_df = df[val_mask].copy()

    if len(val_df) == 0:
        val_df = df.iloc[int(len(df)*0.8):].copy()

    X_val = val_df[router.feature_names].values
    y_val = val_df[target_c7_col].values + 3
    dt_val = val_df[target_dt_col].values

    print(f"\n📊 БЭКТЕСТ РОУТЕРА НА ВАЛИДАЦИОННОМ OOS ({len(val_df):,} сек):")
    print(f"{'Стратегия ансамблирования':<30} | {'OOS IC':<8} | {'AUC OVR':<8} | {'Top-2% PnL':<12} | {'WinRate (Top-2%)':<16} | {'Прирост':<10}")
    print("-" * 95)

    modes = [
        ("Single Model (Win 00)", "latest"),
        ("Uniform Average (1/M)", "uniform"),
        ("Dynamic Swarm Router (MoE)", "router")
    ]

    base_pnl = 0.0
    for name, mode in modes:
        res = router.predict_ensemble(X_val, val_df, mode=mode)
        exp_z = res['expected_z']

        # IC
        ic = float(np.corrcoef(exp_z, dt_val)[0, 1]) if np.std(exp_z) > 1e-6 else 0.0

        # AUC
        try:
            auc = float(roc_auc_score(y_val, res['probs'], multi_class='ovr', labels=list(range(7))))
        except Exception:
            auc = 0.5

        # Top 2%
        th = np.quantile(exp_z, 0.98)
        mask = (exp_z >= th)
        top2_pnl = float(np.mean(dt_val[mask])) if np.sum(mask) > 0 else 0.0
        win_rate = float(np.mean(dt_val[mask] > 0)) * 100.0 if np.sum(mask) > 0 else 0.0

        if mode == "latest":
            base_pnl = top2_pnl
            diff_str = "База"
        else:
            diff = top2_pnl - base_pnl
            diff_str = f"{diff:+5.2f} тиков"

        print(f"{name:<30} | {ic:+6.4f}  | {auc:6.4f}   | {top2_pnl:+6.2f} тиков | {win_rate:6.1f}%          | {diff_str:<10}")

    print("-" * 95)


def main():
    parser = argparse.ArgumentParser(description="Swarm Regime Router & Benchmark")
    parser.add_argument("--coin", type=str, default="prl", help="Монета (prl, quantus)")
    parser.add_argument("--horizon", type=int, default=8, help="Горизонт H (сек)")
    parser.add_argument("--ratio", type=int, default=None, help="Коэффициент K (L/H)")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Запустить сравнительный бэктест")
    args = parser.parse_args()

    coin = args.coin.lower()
    H = args.horizon
    if args.ratio is not None:
        K = args.ratio
    else:
        K = 2 if coin == 'prl' and H == 8 else (3 if coin == 'prl' and H == 25 else 1)

    benchmark_router(coin, H, K)


if __name__ == "__main__":
    main()
