#!/usr/bin/env python3
"""
_tools/build_rl_observation_matrix_48d.py
=============================================================================
Сборщик единой, свободной от эвристик матрицы рыночных наблюдений (39D Market Obs):
  - 16D Order-1 Alpha: bias_8h (8D) + expected_z_8h (8D)
  - 11D Order-2 Risk: 11 калиброванных OOS вероятностей (sweep, bleed, adverse, fill, etc.)
  - 12D LOB Microstructure & Cross-Venue Basis (ST, BigONE, Gate.io)
Все 39 признаков строго непрерывны, масштабно-инвариантны (относительные % и log-величины).
Агент добавляет 9D динамических признаков инвентаря на GPU, формируя итоговый 48D вектор.
=============================================================================
"""

import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
ALPHA_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_56d.npz"
RISK_PATH = BASE_DIR / "data" / "cache" / "order2_risk_matrix_11d.npz"
LOB_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
PARQUET_PATH = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
OUT_PATH = BASE_DIR / "data" / "cache" / "rl_market_obs_39d.npz"


import numba

@numba.njit(fastmath=True)
def _fast_ema_flow(buy_v: np.ndarray, sell_v: np.ndarray, alpha_ema: float):
    n = len(buy_v)
    flow = np.zeros(n, dtype=np.float32)
    b_curr, s_curr = 0.0, 0.0
    for i in range(n):
        b_curr = (1.0 - alpha_ema) * b_curr + alpha_ema * buy_v[i]
        s_curr = (1.0 - alpha_ema) * s_curr + alpha_ema * sell_v[i]
        flow[i] = (b_curr - s_curr) / (b_curr + s_curr + 1e-5)
    return flow


def main():
    print("=" * 70)
    print("  BUILDING 39D DATA-DRIVEN MARKET OBSERVATION MATRIX (NO EMPIRICAL RULES)")
    print("=" * 70, flush=True)

    t0 = time.time()

    # 1. Загрузка Order-1 Alpha сигналов
    print("📦 1. Загрузка Order-1 Alpha (8 горизонтов)...", flush=True)
    alpha = np.load(ALPHA_PATH, allow_pickle=True)
    bias_8h = alpha["bias_8h"].astype(np.float32)          # (N, 8) in [-1, +1]
    expected_z_8h = alpha["expected_z_8h"].astype(np.float32)  # (N, 8) in ticks
    # Нормализация expected_z в диапазон [-3, +3] делением на 3.0
    expected_z_8h_norm = np.clip(expected_z_8h / 3.0, -2.0, 2.0)
    n = len(bias_8h)

    # 2. Загрузка Order-2 Risk сигналов
    print("📦 2. Загрузка Order-2 Microstructure & Risk (11 таргетов)...", flush=True)
    risk = np.load(RISK_PATH, allow_pickle=True)
    risk_11d = risk["risk_matrix_11d"].astype(np.float32)  # (N, 11) in [0, 1]
    risk_names = [str(x) for x in risk["target_names"]]
    # Для carousel_capacity делаем log1p нормализацию:
    cap_idx = risk_names.index("carousel_capacity")
    risk_11d[:, cap_idx] = np.log1p(np.maximum(0.0, risk_11d[:, cap_idx])) / 5.0

    # 3. Загрузка SafeTrade и BigONE LOB кэша
    print("📦 3. Загрузка SafeTrade L1-L10 и BigONE L1-L5 стаканов...", flush=True)
    lob = np.load(LOB_PATH, allow_pickle=True)
    st_b1 = lob["bid_p"][:, 0].astype(np.float32)
    st_a1 = lob["ask_p"][:, 0].astype(np.float32)
    st_mid = 0.5 * (st_b1 + st_a1)
    st_mid_safe = np.maximum(st_mid, 1e-4)

    # Top-3 объемы SafeTrade
    st_bq_top3 = lob["bid_q"][:, :3].sum(axis=1).astype(np.float32)
    st_aq_top3 = lob["ask_q"][:, :3].sum(axis=1).astype(np.float32)
    st_spread_rel = np.clip((st_a1 - st_b1) / st_mid_safe, 0.0, 0.05)
    st_imbalance_top3 = (st_bq_top3 - st_aq_top3) / (st_bq_top3 + st_aq_top3 + 1e-5)
    st_log_bid_vol_top3 = np.log1p(st_bq_top3) / 10.0
    st_log_ask_vol_top3 = np.log1p(st_aq_top3) / 10.0

    # Торговый поток SafeTrade (rolling volume 30s)
    st_buy_vol = lob["buy_vol"].astype(np.float32)
    st_sell_vol = lob["sell_vol"].astype(np.float32)
    st_trade_flow_30s = _fast_ema_flow(st_buy_vol, st_sell_vol, 2.0 / 31.0)

    # BigONE микроструктура
    bo_b1 = lob["bo_bid_p"][:, 0].astype(np.float32)
    bo_a1 = lob["bo_ask_p"][:, 0].astype(np.float32)
    bo_mid = 0.5 * (bo_b1 + bo_a1)
    bo_mid_safe = np.maximum(bo_mid, 1e-4)
    bo_has_data = (bo_mid > 1e-4).astype(np.float32)

    bo_basis_rel = np.where(bo_has_data > 0.5, (bo_mid - st_mid) / st_mid_safe, 0.0).astype(np.float32)
    bo_basis_rel = np.clip(bo_basis_rel * 10.0, -2.0, 2.0)  # ~20% shift cap
    bo_spread_rel = np.where(bo_has_data > 0.5, (bo_a1 - bo_b1) / bo_mid_safe, 0.0).astype(np.float32)
    bo_spread_rel = np.clip(bo_spread_rel, 0.0, 0.05)

    bo_bq_top3 = lob["bo_bid_q"][:, :3].sum(axis=1).astype(np.float32)
    bo_aq_top3 = lob["bo_ask_q"][:, :3].sum(axis=1).astype(np.float32)
    bo_imbalance_top3 = np.where(bo_has_data > 0.5, (bo_bq_top3 - bo_aq_top3) / (bo_bq_top3 + bo_aq_top3 + 1e-5), 0.0).astype(np.float32)
    bo_sec_col = "bo_sec_since_tick" if "bo_sec_since_tick" in lob else "bo_sec"
    bo_sec_norm = np.clip(lob[bo_sec_col].astype(np.float32) / 60.0, 0.0, 1.0)

    # 4. Загрузка Gate.io данных
    print("📦 4. Загрузка Gate.io микроструктуры из Parquet...", flush=True)
    gate_table = pq.read_table(str(PARQUET_PATH), columns=["GATE_mid_price", "GATE_bids_vol_top5", "GATE_asks_vol_top5"])
    gate_mid = gate_table["GATE_mid_price"].to_numpy().astype(np.float32)
    gate_bvol = gate_table["GATE_bids_vol_top5"].to_numpy().astype(np.float32)
    gate_avol = gate_table["GATE_asks_vol_top5"].to_numpy().astype(np.float32)
    del gate_table

    gate_has_data = (~np.isnan(gate_mid) & (gate_mid > 1e-4)).astype(np.float32)
    gate_basis_rel = np.where(gate_has_data > 0.5, (gate_mid - st_mid) / st_mid_safe, 0.0).astype(np.float32)
    gate_basis_rel = np.clip(gate_basis_rel * 10.0, -2.0, 2.0)
    gate_imb = np.where(gate_has_data > 0.5, (gate_bvol - gate_avol) / (gate_bvol + gate_avol + 1e-5), 0.0).astype(np.float32)
    gate_imb = np.clip(gate_imb, -1.0, 1.0)

    # Нелинейный арбитражный импульс: sign(basis) * sqrt(|basis|)
    cross_arb_pressure = np.sign(bo_basis_rel) * np.sqrt(np.abs(bo_basis_rel))

    # Сборка 12D блока LOB & Cross-Venue
    lob_cross_12d = np.column_stack([
        st_spread_rel,
        st_imbalance_top3,
        st_log_bid_vol_top3,
        st_log_ask_vol_top3,
        st_trade_flow_30s,
        bo_basis_rel,
        bo_spread_rel,
        bo_imbalance_top3,
        bo_sec_norm,
        gate_basis_rel,
        gate_imb,
        cross_arb_pressure,
    ]).astype(np.float32)

    # 5. Объединение в 39D матрицу: 16D + 11D + 12D
    print("🧩 5. Объединение блоков 16D Alpha + 11D Risk + 12D LOB/Basis...", flush=True)
    obs_39d = np.hstack([
        bias_8h,            # 8D: alpha skew [-1, +1]
        expected_z_8h_norm, # 8D: expected shift [-2, +2]
        risk_11d,           # 11D: risk probabilities [0, 1]
        lob_cross_12d,      # 12D: LOB microstructure & basis
    ]).astype(np.float32)

    # Защита от NaN / Inf
    n_nans = np.isnan(obs_39d).sum()
    n_infs = np.isinf(obs_39d).sum()
    if n_nans > 0 or n_infs > 0:
        print(f"⚠️ Найдено {n_nans} NaNs и {n_infs} Infs. Замена нулями...")
        obs_39d = np.nan_to_num(obs_39d, nan=0.0, posinf=2.0, neginf=-2.0)

    feature_names = (
        [f"alpha_bias_h{h}s" for h in [1, 2, 5, 10, 20, 40, 80, 160]] +
        [f"alpha_z_h{h}s" for h in [1, 2, 5, 10, 20, 40, 80, 160]] +
        [f"risk_{name}" for name in risk_names] +
        [
            "st_spread_rel", "st_imbalance_top3", "st_log_bid_vol_top3", "st_log_ask_vol_top3", "st_trade_flow_30s",
            "bo_basis_rel", "bo_spread_rel", "bo_imbalance_top3", "bo_sec_norm",
            "gate_basis_rel", "gate_imb", "cross_arb_pressure",
        ]
    )

    print(f"💾 6. Сохранение итогового кэша: {OUT_PATH}...")
    tmp_path = OUT_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        market_obs_39d=obs_39d,
        feature_names=np.array(feature_names),
        total_rows=n,
    )
    tmp_path.replace(OUT_PATH)

    sz_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    dt = time.time() - t0
    print(f"✅ Готово за {dt:.1f}с! Размер: {sz_mb:.1f} MB, {n:,} строк × {obs_39d.shape[1]} признаков.")
    print("=" * 70)


if __name__ == "__main__":
    main()
