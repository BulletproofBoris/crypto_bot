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
sys.path.insert(0, str(BASE_DIR))

from _tools.data_labeler import compute_single_horizon_labels, COIN_CONFIG

class SwarmRegimeRouter:
    """Контекстный роутер и блендер экспертов пула моделей."""

    def __init__(self, coin: str, H: int, K: float = 3.0, temperature: float = 0.5, use_neural: bool = True, neural_weight: float = 0.15):
        self.coin = coin.lower()
        self.H = H
        self.K = float(K)
        self.L = int(round(H * self.K))
        self.temperature = max(0.01, temperature)
        self.use_neural = use_neural
        self.neural_weight = float(neural_weight)
        k_str = f"{self.K:g}"
        self.model_dir = BASE_DIR / "models" / self.coin / f"h{H}s_k{k_str}"
        
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Манифест моделей не найден: {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

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
        if self.models and hasattr(self.models[0], 'feature_name'):
            self.feature_names = list(self.models[0].feature_name())
        else:
            feats_json = BASE_DIR / "data" / "features" / f"selected_features_{self.coin}_h8s.json"
            with open(feats_json) as f:
                self.feature_names = json.load(f)["feature_order"]
        print(f"✅ Успешно загружено {self.num_models} моделей-экспертов LightGBM в RAM ({len(self.feature_names)} признаков).")

        # Загрузка метаданных мультиархитектурных нейросетевых экспертов (Conv1D+GRU, Causal CNN, Attention, MLP)
        # Сами графы Keras подгружаются лениво (только при отсутствии готового .npy кэша предсказаний)
        self.neural_experts = []
        self.alliance_cfg = None
        alliance_path = BASE_DIR / "models" / self.coin / f"ensemble_alliance_h{H}s.json"
        if self.use_neural and self.neural_weight > 0.0 and alliance_path.exists():
            try:
                with open(alliance_path) as f:
                    self.alliance_cfg = json.load(f)
                for arch_dir_name, arch_key in [
                    ("conv1d_gru", "conv1d+gru"),
                    ("causal_cnn", "causal_cnn"),
                    ("mlp_bottleneck", "mlp_bottleneck"),
                    ("attention", "attention")
                ]:
                    n_dir = BASE_DIR / "models" / self.coin / f"{arch_dir_name}_h{H}s"
                    if not n_dir.exists():
                        continue
                    for k_file in sorted(n_dir.glob("model_win_*.keras")):
                        win_tag = k_file.stem.replace("model_", "")  # win_00, win_01...
                        win_idx = int(win_tag.split("_")[-1]) if win_tag.split("_")[-1].isdigit() else 0
                        m_file = n_dir / f"meta_{win_tag}.json"
                        if m_file.exists():
                            with open(m_file) as mf:
                                n_meta = json.load(mf)
                            ic_val = float(n_meta.get("metrics_oos", {}).get("ic_dt_ticks", 0.1))
                            if ic_val >= 0.10:
                                self.neural_experts.append({
                                    "arch": arch_key if win_idx == 0 else f"{arch_key}_{win_tag}",
                                    "base_arch": arch_key,
                                    "win_idx": win_idx,
                                    "k_file": str(k_file),
                                    "model": None,
                                    "seq_len": int(n_meta["seq_len"]),
                                    "mean": np.array(n_meta["feature_norm"]["mean"], dtype=np.float32),
                                    "std": np.array(n_meta["feature_norm"]["std"], dtype=np.float32),
                                    "ic": ic_val,
                                    "train_ts_range": n_meta.get("train_ts_range", [])
                                })
                if self.neural_experts:
                    archs_str = ", ".join(f"{e['arch']}(IC={e['ic']:+.3f})" for e in self.neural_experts)
                    print(f"🧬 Подключен нейросетевой альянс H={H}с ({len(self.neural_experts)} арх.): {archs_str}")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить нейросети для H={H}с: {e}")

        # Подготовка матрицы признаков режимов экспертов
        self._build_regime_matrix()

    def _build_regime_matrix(self):
        """Формирует нормализованную матрицу профилей режимов для экспертов."""
        self.live_feature_map = [
            ('mean_ST_spread_pct', 'std_ST_spread_pct', 'ST_spread_pct'),
            ('mean_ST_vol_300s_ticks', 'std_ST_vol_300s_ticks', 'ST_vol_300s_ticks'),
            ('mean_ST_ofi_norm_300s', 'std_ST_ofi_norm_300s', 'ST_ofi_norm_300s'),
            ('mean_ST_cvd_5m', 'std_ST_cvd_5m', 'ST_cvd_5m'),
        ]
        means = []
        stds = []
        for fp in self.regime_fingerprints:
            means.append([fp.get(m_k, 0.0) for m_k, _, _ in self.live_feature_map])
            stds.append([max(fp.get(s_k, 1.0), 1e-6) for _, s_k, _ in self.live_feature_map])

        self.regime_matrix = np.array(means, dtype=np.float32)
        self.regime_stds = np.array(stds, dtype=np.float32)
        self.regime_mean = np.mean(self.regime_matrix, axis=0, keepdims=True)
        self.regime_std = np.mean(self.regime_stds, axis=0, keepdims=True) + 1e-6
        self.norm_regime_matrix = (self.regime_matrix - self.regime_mean) / self.regime_std

    def compute_routing_weights(self, current_regime_df: pd.DataFrame) -> np.ndarray:
        """
        Вычисляет матрицу весов W (N x M) для каждой секунды на основе близости
        текущего рыночного профиля к историческому профилю эксперта + приора свежести.
        """
        n_samples = len(current_regime_df)
        if self.num_models == 1:
            return np.ones((n_samples, 1), dtype=np.float32)

        cols = []
        for idx, (m_k, s_k, col_name) in enumerate(self.live_feature_map):
            if col_name in current_regime_df.columns:
                cols.append(np.nan_to_num(current_regime_df[col_name].values, nan=float(self.regime_mean[0, idx])).astype(np.float32))
            else:
                cols.append(np.full(n_samples, float(self.regime_mean[0, idx]), dtype=np.float32))

        cur_mat = np.column_stack(cols).astype(np.float32)
        diff = (cur_mat[:, np.newaxis, :] - self.regime_matrix[np.newaxis, :, :]) / self.regime_stds[np.newaxis, :, :]
        diff = np.clip(diff, -3.0, 3.0)
        dist_sq = np.mean(diff ** 2, axis=-1)

        # Приор свежести: win_00 получает приоритет при сопоставимом рыночном режиме
        recency_bonus = -1.0 * np.arange(self.num_models, dtype=np.float32)[np.newaxis, :]
        logits = (-dist_sq / max(0.5, self.temperature)) + recency_bonus
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return weights.astype(np.float32)

    def _compute_slice_hash(self, X: np.ndarray) -> str:
        import hashlib
        n = len(X)
        if n == 0:
            return "empty"
        sample_pts = np.asarray([
            n,
            float(X[0, 0]),
            float(X[-1, 0]),
            float(X[n // 2, 0]),
            float(np.nansum(X[:, 0])),
            float(np.nansum(X[:, min(12, X.shape[1] - 1)]))
        ], dtype=np.float64)
        return hashlib.md5(sample_pts.tobytes()).hexdigest()[:10]

    def _sanitize_stale_bo_features(self, X: np.ndarray, current_regime_df: pd.DataFrame) -> np.ndarray:
        """Обнуляет направленные осцилляторы BigONE / Lead-Lag на секундах зависания коллектора (BO_sec_since_tick > 60s)."""
        if 'BO_sec_since_tick' not in current_regime_df.columns:
            return X
        bo_stale = (current_regime_df['BO_sec_since_tick'].values > 60.0)
        if not np.any(bo_stale):
            return X
        X_clean = X.copy()
        zero_prefixes = ('LL_', 'BO_ofi_', 'BO_cvd_', 'BO_imb_', 'BO_obi_', 'BO_micro_', 'BO_ret_', 'BO_vol_intensity_')
        zero_cols = [idx for idx, fname in enumerate(self.feature_names) if fname.startswith(zero_prefixes)]
        if zero_cols:
            X_clean[np.ix_(bo_stale, zero_cols)] = 0.0
        return X_clean

    def predict_ensemble(self, X: np.ndarray, current_regime_df: pd.DataFrame, mode: str = "router"):
        """
        Вычисляет ансамблевый прогноз по 7 классам:
        mode:
          - 'router': динамическое взвешивание на основе похожести режима
          - 'uniform': простое среднее 1/M
          - 'latest': только самая свежая модель (win_00)
        """
        n_samples = len(X)
        slice_hash = self._compute_slice_hash(X)
        cache_dir = BASE_DIR / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        manifest_mtime = int((self.model_dir / "manifest.json").stat().st_mtime)
        lgb_cache_file = cache_dir / f"lgb_probs_{self.coin}_h{self.H}s_k{self.K:g}_{n_samples}_{slice_hash}_{manifest_mtime}.npy"

        if getattr(self, "_lgb_cache_sig", None) == (slice_hash, manifest_mtime):
            all_probs = self._lgb_cache_probs
        elif lgb_cache_file.exists():
            all_probs = np.load(lgb_cache_file)
            self._lgb_cache_sig = (slice_hash, manifest_mtime)
            self._lgb_cache_probs = all_probs
        else:
            X_inf = self._sanitize_stale_bo_features(X, current_regime_df)
            all_probs = np.zeros((self.num_models, n_samples, 7), dtype=np.float32)
            for i, booster in enumerate(self.models):
                raw_p = booster.predict(X_inf)
                if raw_p.shape[1] == 7:
                    all_probs[i] = raw_p
                else:
                    for col_idx in range(min(7, raw_p.shape[1])):
                        all_probs[i, :, col_idx] = raw_p[:, col_idx]
            self._lgb_cache_sig = (slice_hash, manifest_mtime)
            self._lgb_cache_probs = all_probs
            try:
                np.save(lgb_cache_file, all_probs)
            except Exception:
                pass

        if mode == "latest":
            blend_probs = all_probs[0].copy()
            weights = np.zeros((n_samples, self.num_models), dtype=np.float32)
            weights[:, 0] = 1.0
        elif mode == "uniform":
            blend_probs = np.mean(all_probs, axis=0)
            weights = np.full((n_samples, self.num_models), 1.0 / self.num_models, dtype=np.float32)
        elif mode == "router":
            weights = self.compute_routing_weights(current_regime_df)
            blend_probs = np.einsum('nm,mnc->nc', weights, all_probs)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

        # Ожидаемый импульс E[z] и векторизованная калибровка исторических окон к эталонному окну win_00
        class_values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
        ez_all = np.matmul(all_probs, class_values)  # (M, N)
        pb_all = all_probs[:, :, 5] + all_probs[:, :, 6]
        ps_all = all_probs[:, :, 0] + all_probs[:, :, 1]

        ref_mean = float(np.mean(ez_all[0]))
        ref_std = float(np.std(ez_all[0])) + 1e-6
        ref_pb_mean = float(np.mean(pb_all[0])) + 1e-6
        ref_ps_mean = float(np.mean(ps_all[0])) + 1e-6

        if self.num_models > 1:
            ez_means = np.mean(ez_all, axis=1, keepdims=True)
            ez_stds = np.std(ez_all, axis=1, keepdims=True) + 1e-6
            cal_ez_all = (ez_all - ez_means) * (ref_std / ez_stds) + ref_mean
            cal_ez_all[0] = ez_all[0]

            pb_means = np.mean(pb_all, axis=1, keepdims=True) + 1e-6
            ps_means = np.mean(ps_all, axis=1, keepdims=True) + 1e-6
            cal_pb_all = np.clip(pb_all * (ref_pb_mean / pb_means), 0.0, 1.0)
            cal_ps_all = np.clip(ps_all * (ref_ps_mean / ps_means), 0.0, 1.0)
            cal_pb_all[0] = pb_all[0]
            cal_ps_all[0] = ps_all[0]
        else:
            cal_ez_all = ez_all
            cal_pb_all = pb_all
            cal_ps_all = ps_all

        expected_z = np.einsum('nm,mn->n', weights, cal_ez_all)
        expected_z = (expected_z - float(np.mean(expected_z))) * (ref_std / (float(np.std(expected_z)) + 1e-6)) + ref_mean
        p_buy = np.einsum('nm,mn->n', weights, cal_pb_all)
        p_sell = np.einsum('nm,mn->n', weights, cal_ps_all)

        # Если подключены нейросетевые эксперты (Conv1D+GRU, Causal CNN, MLP, Attention),
        # формируем IC-взвешенный альянс нейросетей с байесовской калибровкой априорных весов классов
        if self.use_neural and self.neural_experts and self.neural_weight > 0.0:
            cw = np.array(self.alliance_cfg.get("class_weights", [1.0]*7), dtype=np.float32) if self.alliance_cfg else np.ones(7, dtype=np.float32)
            cw_pow = np.power(np.maximum(cw, 1e-3), 0.85)[np.newaxis, :]

            nn_ez_num = np.zeros(n_samples, dtype=np.float32)
            nn_probs_num = np.zeros((n_samples, 7), dtype=np.float32)
            nn_w_sum = np.zeros(n_samples, dtype=np.float32)
            lgb_ez_std = float(np.std(expected_z)) + 1e-6
            multi_win = len(set(e.get("win_idx", 0) for e in self.neural_experts)) > 1
            X_nn = None

            for exp in self.neural_experts:
                w_idx = min(exp.get("win_idx", 0), weights.shape[1] - 1)
                reg_w = weights[:, w_idx] if multi_win else 1.0
                sl = exp["seq_len"]
                arch_clean = exp["arch"].replace("+", "_")
                k_mtime = int(Path(exp["k_file"]).stat().st_mtime)
                cache_file = cache_dir / f"nn_preds_{self.coin}_h{self.H}s_{arch_clean}_{n_samples}_{slice_hash}_{k_mtime}.npy"
                legacy_cache = cache_dir / f"nn_preds_{self.coin}_h{self.H}s_{arch_clean}_{n_samples}_{slice_hash}.npy"
                if not cache_file.exists() and legacy_cache.exists() and int(legacy_cache.stat().st_mtime) >= k_mtime:
                    cache_file = legacy_cache

                if multi_win and not cache_file.exists() and float(np.mean(reg_w)) < 0.02:
                    continue

                if cache_file.exists():
                    p_raw = np.load(cache_file)
                else:
                    if X_nn is None:
                        X_nn = self._sanitize_stale_bo_features(X, current_regime_df)
                    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
                    import tensorflow as tf
                    print(f"  ⚡ [H={self.H}s] Инференс нейросети {exp['arch']} (mean_w={float(np.mean(reg_w)):.3f})...", flush=True)
                    model = tf.keras.models.load_model(exp["k_file"], compile=False)
                    x_norm = np.clip((X_nn - exp["mean"]) / exp["std"], -5.0, 5.0).astype(np.float32)
                    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=5.0, neginf=-5.0)
                    x_pad = np.vstack([np.repeat(x_norm[:1], sl - 1, axis=0), x_norm])
                    x_3d_view = np.lib.stride_tricks.sliding_window_view(x_pad, (sl, x_norm.shape[1]))[:, 0, :, :]

                    # Пакетный инференс чанками по 16384 строк
                    p_chunks = []
                    chunk_sz = 16384
                    for start_i in range(0, n_samples, chunk_sz):
                        end_i = min(n_samples, start_i + chunk_sz)
                        x_batch = np.ascontiguousarray(x_3d_view[start_i:end_i], dtype=np.float32)
                        p_batch = model.predict_on_batch(x_batch)
                        p_chunks.append(np.asarray(p_batch, dtype=np.float32))
                    p_raw = np.vstack(p_chunks)
                    del model, x_norm, x_pad, x_3d_view, p_chunks
                    tf.keras.backend.clear_session()
                    try:
                        np.save(cache_file, p_raw)
                    except Exception:
                        pass

                p_raw = np.clip(p_raw, 1e-7, 1.0 - 1e-7)

                # 1. Сохраняем максимальный IC сырого нейросетевого прогноза, приводя дисперсию к шкале роутера
                ez_raw = np.dot(p_raw, class_values)
                ez_std = float(np.std(ez_raw)) + 1e-6
                ez_scaled = (ez_raw - float(np.mean(ez_raw))) * (lgb_ez_std / ez_std) + float(np.mean(expected_z))

                # 2. Байесовская калибровка вероятностей для совместимости с порогами p_buy / p_sell
                p_cal = p_raw / cw_pow
                p_cal = p_cal / np.sum(p_cal, axis=1, keepdims=True)

                w_idx = min(exp.get("win_idx", 0), weights.shape[1] - 1)
                reg_w = weights[:, w_idx] if multi_win else 1.0
                w_exp = np.broadcast_to(np.asarray(max(0.01, exp["ic"] ** 2) * reg_w, dtype=np.float32), (n_samples,))
                nn_ez_num += w_exp * ez_scaled
                nn_probs_num += w_exp[:, np.newaxis] * p_cal
                nn_w_sum += w_exp
                del p_raw, ez_raw, ez_scaled, p_cal

            if np.any(nn_w_sum > 0):
                inv_w = 1.0 / np.maximum(nn_w_sum, 1e-7)
                nn_ez_blend = nn_ez_num * inv_w
                nn_probs_blend = nn_probs_num * inv_w[:, np.newaxis]

                w_nn = np.clip(self.neural_weight, 0.0, 1.0)
                lgb_mean = float(np.mean(expected_z))
                raw_blend_ez = (1.0 - w_nn) * expected_z + w_nn * nn_ez_blend
                blend_std = float(np.std(raw_blend_ez)) + 1e-6
                # Сохраняем амплитуду дисперсии сигнала, чтобы пороги диспетчера (trend_gate, threshold_core) не размывались
                expected_z = (raw_blend_ez - float(np.mean(raw_blend_ez))) * (lgb_ez_std / blend_std) + lgb_mean
                blend_probs = (1.0 - w_nn) * blend_probs + w_nn * nn_probs_blend
                nn_pb = nn_probs_blend[:, 5] + nn_probs_blend[:, 6]
                nn_ps = nn_probs_blend[:, 0] + nn_probs_blend[:, 1]
                nn_pb = np.clip(nn_pb * (ref_pb_mean / (float(np.mean(nn_pb)) + 1e-6)), 0.0, 1.0)
                nn_ps = np.clip(nn_ps * (ref_ps_mean / (float(np.mean(nn_ps)) + 1e-6)), 0.0, 1.0)
                p_buy = (1.0 - w_nn) * p_buy + w_nn * nn_pb
                p_sell = (1.0 - w_nn) * p_sell + w_nn * nn_ps

        # Защита ML-инференса от зависания коллектора второй биржи (BO_sec_since_tick > 60s):
        # когда 135 признаков BO_*/LL_* замирают, дисперсия прогноза сжимается на 30-35% и возникает ложный сдвиг среднего.
        # Выравниваем масштаб (mean/std) и вероятности хвостов на интервалах заморозки BO к живым участкам SafeTrade.
        if 'BO_sec_since_tick' in current_regime_df.columns:
            bo_stale = (current_regime_df['BO_sec_since_tick'].values > 60.0)
            n_stale = int(np.sum(bo_stale))
            n_fresh = int(len(bo_stale) - n_stale)
            if n_stale > 60 and n_fresh > 60:
                f_mean = float(np.mean(expected_z[~bo_stale]))
                f_std = float(np.std(expected_z[~bo_stale])) + 1e-6
                s_mean = float(np.mean(expected_z[bo_stale]))
                s_std = float(np.std(expected_z[bo_stale])) + 1e-6
                expected_z = expected_z.copy()
                expected_z[bo_stale] = (expected_z[bo_stale] - s_mean) * (f_std / s_std) + f_mean

                f_pb = float(np.mean(p_buy[~bo_stale])) + 1e-6
                s_pb = float(np.mean(p_buy[bo_stale])) + 1e-6
                p_buy = p_buy.copy()
                p_buy[bo_stale] = np.clip(p_buy[bo_stale] * (f_pb / s_pb), 0.0, 1.0)

                f_ps = float(np.mean(p_sell[~bo_stale])) + 1e-6
                s_ps = float(np.mean(p_sell[bo_stale])) + 1e-6
                p_sell = p_sell.copy()
                p_sell[bo_stale] = np.clip(p_sell[bo_stale] * (f_ps / s_ps), 0.0, 1.0)

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
    k_str = f"{K:g}"
    target_c7_col = f"target_class_7_h{H}_k{k_str}"
    target_dt_col = f"target_dt_ticks_h{H}_k{k_str}"

    schema_cols = set(pq.read_schema(labeled_path).names)
    need_labeling = (target_c7_col not in schema_cols) or (target_dt_col not in schema_cols)

    extra_cols = ['timestamp', 'sample_split', 'ST_spread_pct']
    if need_labeling:
        extra_cols += [
            'ST_mid_price', 'ST_spread', 'ST_bid_1_price', 'ST_ask_1_price',
            'ST_bid_1_qty', 'ST_ask_1_qty', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
            'ST_trade_vol'
        ]
    else:
        extra_cols += [target_c7_col, target_dt_col]

    cols = list(dict.fromkeys([c for c in router.feature_names + extra_cols if c in schema_cols]))
    tbl = pq.read_table(labeled_path, columns=cols)
    df = tbl.to_pandas()

    if need_labeling:
        cfg = COIN_CONFIG[coin]
        labels = compute_single_horizon_labels(df, coin, H, K, cfg['min_vol'], cfg['tick_size'])
        df['target_class'] = labels['class_7']
        df['target_dt'] = labels['dt_ticks']
    else:
        df['target_class'] = df[target_c7_col]
        df['target_dt'] = df[target_dt_col]

    # Берем чистый OOS (последние 20% Val)
    val_mask = (df['sample_split'] == 'val')
    val_df = df[val_mask].copy()

    if len(val_df) == 0:
        val_df = df.iloc[int(len(df)*0.8):].copy()

    X_val = val_df[router.feature_names].values
    y_val = val_df['target_class'].values + 3
    dt_val = val_df['target_dt'].values

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
    parser.add_argument("--horizon", type=int, default=5, help="Горизонт H (сек)")
    parser.add_argument("--ratio", type=float, default=None, help="Коэффициент K (L/H)")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Запустить сравнительный бэктест")
    args = parser.parse_args()

    coin = args.coin.lower()
    H = args.horizon
    if args.ratio is not None:
        K = args.ratio
    else:
        if H == 5:
            K = 3.0
        elif H in (12, 13, 14, 15):
            K = 3.5
        elif H >= 18:
            K = 3.0
        else:
            K = 3.0

    benchmark_router(coin, H, K)


if __name__ == "__main__":
    main()
