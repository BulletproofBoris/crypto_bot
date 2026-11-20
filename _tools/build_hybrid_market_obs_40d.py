#!/usr/bin/env python3
"""
_tools/build_hybrid_market_obs_40d.py
=============================================================================
Единый строго каузальный генератор 40D рыночного вектора (0% утечки будущего!):
  - [0..11]  12D Raw SafeTrade LOB & Trade Tape (спреды, дисбалансы 1/3/5/10, лог-объемы, EMA flow 15/30/60s)
  - [12..21] 10D Raw BigONE LOB & Cross-Venue Basis (базис BO/ST, спред, дисбаланс, возраст тика, OFI, ret30s)
  - [22..33] 12D Causal Momentum & Order-2 Walk-Forward Risk Signals (ret 5..120s, vol10s, inertia,
             causal guard_sweep_12s, guard_bleed_60s, guard_crash_300s, guard_exit_30s, adverse_selection_5s)
             * НИКАКИХ бинарных будущих меток y_*! Только честные каузальные прогнозы Order-2!
  - [34..39] 6D Multi-Horizon Causal Order-1 Alpha (H = 5..160s expected_z)
=============================================================================
"""

import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_600H_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
ORDER1_ALPHA_PATH = BASE_DIR / "data" / "cache" / "order1_alpha_matrix_56d.npz"
ORDER2_RISK_PATH = BASE_DIR / "data" / "cache" / "order2_risk_matrix_11d.npz"
OUT_PATH = BASE_DIR / "data" / "cache" / "rl_hybrid_market_obs_40d.npz"


def fast_ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    cur = float(arr[0])
    out[0] = cur
    one_m_a = 1.0 - alpha
    for i in range(1, n):
        cur = one_m_a * cur + alpha * float(arr[i])
        out[i] = cur
    return out


def main():
    print("=" * 80)
    print("🚀 СБОРКА СТРОГО КАУЗАЛЬНОГО 40D РЫНОЧНОГО ВЕКТОРА (0% ORACLE LEAKAGE)")
    print("=" * 80)

    t0 = time.time()
    print("📦 Загрузка rl_guard_lob_cache_600h.npz...")
    lob = np.load(CACHE_600H_PATH, mmap_mode="r")
    n = len(lob["bid_p"])
    print(f"  Всего тиков: {n:,}")

    # 1. SafeTrade Raw LOB & Tape (12D)
    print("  ⚡ 1/4 Извлечение SafeTrade сырого стакана и ленты (12D)...")
    st_b1 = lob["bid_p"][:, 0].astype(np.float32)
    st_a1 = lob["ask_p"][:, 0].astype(np.float32)
    st_mid = np.maximum(0.5 * (st_b1 + st_a1), 1e-4)
    st_spread_rel = np.clip((st_a1 - st_b1) / st_mid, 0.0, 0.05)

    bq = lob["bid_q"].astype(np.float32)
    aq = lob["ask_q"].astype(np.float32)
    st_imb1 = np.clip((bq[:, 0] - aq[:, 0]) / (bq[:, 0] + aq[:, 0] + 1e-5), -1.0, 1.0)
    st_imb3 = np.clip((bq[:, :3].sum(axis=1) - aq[:, :3].sum(axis=1)) / (bq[:, :3].sum(axis=1) + aq[:, :3].sum(axis=1) + 1e-5), -1.0, 1.0)
    st_imb5 = np.clip((bq[:, :5].sum(axis=1) - aq[:, :5].sum(axis=1)) / (bq[:, :5].sum(axis=1) + aq[:, :5].sum(axis=1) + 1e-5), -1.0, 1.0)
    st_imb10 = np.clip((bq[:, :10].sum(axis=1) - aq[:, :10].sum(axis=1)) / (bq[:, :10].sum(axis=1) + aq[:, :10].sum(axis=1) + 1e-5), -1.0, 1.0)

    st_log_b1_q = np.log1p(np.maximum(0.0, bq[:, 0])) / 10.0
    st_log_a1_q = np.log1p(np.maximum(0.0, aq[:, 0])) / 10.0
    st_log_bq_top3 = np.log1p(np.maximum(0.0, bq[:, :3].sum(axis=1))) / 10.0
    st_log_aq_top3 = np.log1p(np.maximum(0.0, aq[:, :3].sum(axis=1))) / 10.0

    st_bv = lob["buy_vol"].astype(np.float32)
    st_sv = lob["sell_vol"].astype(np.float32)

    ema_b15, ema_s15 = fast_ema(st_bv, 2.0 / 16.0), fast_ema(st_sv, 2.0 / 16.0)
    st_trade_flow_15s = np.clip((ema_b15 - ema_s15) / (ema_b15 + ema_s15 + 1e-5), -1.0, 1.0)

    ema_b30, ema_s30 = fast_ema(st_bv, 2.0 / 31.0), fast_ema(st_sv, 2.0 / 31.0)
    st_trade_flow_30s = np.clip((ema_b30 - ema_s30) / (ema_b30 + ema_s30 + 1e-5), -1.0, 1.0)

    ema_b60, ema_s60 = fast_ema(st_bv, 2.0 / 61.0), fast_ema(st_sv, 2.0 / 61.0)
    st_trade_flow_60s = np.clip((ema_b60 - ema_s60) / (ema_b60 + ema_s60 + 1e-5), -1.0, 1.0)

    block1_st_12d = np.column_stack([
        st_spread_rel, st_imb1, st_imb3, st_imb5, st_imb10,
        st_log_b1_q, st_log_a1_q, st_log_bq_top3, st_log_aq_top3,
        st_trade_flow_15s, st_trade_flow_30s, st_trade_flow_60s,
    ])

    # 2. BigONE Raw LOB & Cross-Venue Basis (10D)
    print("  🌐 2/4 Извлечение BigONE сырого стакана и кросс-базиса (10D)...")
    bo_b1 = lob["bo_bid_p"][:, 0].astype(np.float32)
    bo_a1 = lob["bo_ask_p"][:, 0].astype(np.float32)
    bo_mid = np.maximum(0.5 * (bo_b1 + bo_a1), 1e-4)
    bo_valid = (bo_b1 > 1e-4).astype(np.float32)

    bo_basis_rel = np.where(bo_valid > 0.5, np.clip(((bo_mid - st_mid) / st_mid) * 10.0, -2.0, 2.0), 0.0).astype(np.float32)
    bo_spread_rel = np.where(bo_valid > 0.5, np.clip((bo_a1 - bo_b1) / bo_mid, 0.0, 0.05), 0.0).astype(np.float32)

    bo_bq = lob["bo_bid_q"].astype(np.float32)
    bo_aq = lob["bo_ask_q"].astype(np.float32)
    bo_imb1 = np.where(bo_valid > 0.5, np.clip((bo_bq[:, 0] - bo_aq[:, 0]) / (bo_bq[:, 0] + bo_aq[:, 0] + 1e-5), -1.0, 1.0), 0.0).astype(np.float32)
    bo_imb3 = np.where(bo_valid > 0.5, np.clip((bo_bq[:, :3].sum(axis=1) - bo_aq[:, :3].sum(axis=1)) / (bo_bq[:, :3].sum(axis=1) + bo_aq[:, :3].sum(axis=1) + 1e-5), -1.0, 1.0), 0.0).astype(np.float32)

    bo_log_b1_q = np.where(bo_valid > 0.5, np.log1p(np.maximum(0.0, bo_bq[:, 0])) / 10.0, 0.0).astype(np.float32)
    bo_log_a1_q = np.where(bo_valid > 0.5, np.log1p(np.maximum(0.0, bo_aq[:, 0])) / 10.0, 0.0).astype(np.float32)

    bo_bv = lob["bo_buy_vol"].astype(np.float32)
    bo_sv = lob["bo_sell_vol"].astype(np.float32)
    bo_ema_b30, bo_ema_s30 = fast_ema(bo_bv, 2.0 / 31.0), fast_ema(bo_sv, 2.0 / 31.0)
    bo_trade_flow_30s = np.where(bo_valid > 0.5, np.clip((bo_ema_b30 - bo_ema_s30) / (bo_ema_b30 + bo_ema_s30 + 1e-5), -1.0, 1.0), 0.0).astype(np.float32)

    bo_sec_col = "bo_sec_since_tick" if "bo_sec_since_tick" in lob.files else "bo_sec"
    bo_sec_norm = np.clip(lob[bo_sec_col].astype(np.float32) / 60.0, 0.0, 1.0)
    cross_arb_pressure = np.sign(bo_basis_rel) * np.sqrt(np.abs(bo_basis_rel))

    bo_mid_roll30 = fast_ema(bo_mid, 2.0 / 31.0)
    bo_mid_ret_30s = np.where(bo_valid > 0.5, np.clip(((bo_mid - bo_mid_roll30) / bo_mid) * 50.0, -2.0, 2.0), 0.0).astype(np.float32)

    block2_bo_10d = np.column_stack([
        bo_basis_rel, bo_spread_rel, bo_imb1, bo_imb3,
        bo_log_b1_q, bo_log_a1_q, bo_trade_flow_30s, bo_sec_norm,
        cross_arb_pressure, bo_mid_ret_30s,
    ])

    # 3. Causal Momentum & Walk-Forward Order-2 Risk Signals (12D — БЕЗ УТЕЧКИ БУДУЩЕГО!)
    print("  🛡️ 3/4 Загрузка строго каузальных рисков Order-2 (order2_risk_matrix_11d.npz)...")
    ret_5s = np.clip((st_mid - fast_ema(st_mid, 2.0 / 6.0)) / st_mid * 50.0, -2.0, 2.0)
    ret_15s = np.clip((st_mid - fast_ema(st_mid, 2.0 / 16.0)) / st_mid * 40.0, -2.0, 2.0)
    ret_30s = np.clip((st_mid - fast_ema(st_mid, 2.0 / 31.0)) / st_mid * 30.0, -2.0, 2.0)
    ret_60s = np.clip((st_mid - fast_ema(st_mid, 2.0 / 61.0)) / st_mid * 20.0, -2.0, 2.0)
    ret_120s = np.clip((st_mid - fast_ema(st_mid, 2.0 / 121.0)) / st_mid * 15.0, -2.0, 2.0)
    vol_10s = np.clip(np.abs(ret_5s) * 2.0, 0.0, 2.0)
    inertia_30s = np.clip(ret_30s - ret_120s, -2.0, 2.0)

    if not ORDER2_RISK_PATH.exists():
        raise FileNotFoundError(f"Отсутствует каузальная матрица рисков {ORDER2_RISK_PATH}! Запрещено использовать y_* метки!")
    o2 = np.load(ORDER2_RISK_PATH, mmap_mode="r")
    rmat = o2["risk_matrix_11d"]
    assert len(rmat) == n, f"Несовпадение длины order2_risk_matrix_11d ({len(rmat)}) и LOB кэша ({n})!"

    p_adverse = np.clip(rmat[:, 0].astype(np.float32), 0.0, 1.0)  # adverse_selection_5s
    p_sweep   = np.clip(rmat[:, 3].astype(np.float32), 0.0, 1.0)  # guard_sweep_12s (causal!)
    p_bleed   = np.clip(rmat[:, 4].astype(np.float32), 0.0, 1.0)  # guard_bleed_60s (causal!)
    p_crash   = np.clip(rmat[:, 5].astype(np.float32), 0.0, 1.0)  # guard_crash_300s (causal!)
    p_exit    = np.clip(rmat[:, 6].astype(np.float32), 0.0, 1.0)  # guard_exit_30s (causal!)

    block3_macro_12d = np.column_stack([
        ret_5s, ret_15s, ret_30s, ret_60s, ret_120s,
        vol_10s, inertia_30s,
        p_sweep, p_bleed, p_crash, p_exit, p_adverse,
    ])

    # 4. Multi-Horizon Causal Alpha Summary (6D из order1_alpha_matrix_56d.npz: H = 5, 10, 20, 40, 80, 160s)
    print("  🧠 4/4 Загрузка каузальной многогоризонтной альфы Order-1 (6D)...")
    if not ORDER1_ALPHA_PATH.exists():
        raise FileNotFoundError(f"Отсутствует каузальная матрица альфы {ORDER1_ALPHA_PATH}!")
    o1 = np.load(ORDER1_ALPHA_PATH, mmap_mode="r")
    ez8 = o1["expected_z_8h"]
    assert len(ez8) == n, f"Несовпадение длины order1_alpha_matrix_56d ({len(ez8)}) и LOB кэша ({n})!"
    alpha_h6 = np.clip(ez8[:, 2:8].astype(np.float32), -3.0, 3.0)

    # 5. Macro Anti-FOMO Regime Channels (4D: ret_15m, ret_1h, range_pos_4h [-1..+1], range_amp_4h [0..2.5])
    print("  🔭 5/5 Расчет каузальных макро-каналов 15m / 1h / 4h против покупки хая пампа (4D)...")
    import pandas as pd
    ret_15m = np.clip(((st_mid - fast_ema(st_mid, 2.0 / 901.0)) / st_mid) * 10.0, -2.5, 2.5)
    ret_1h = np.clip(((st_mid - fast_ema(st_mid, 2.0 / 3601.0)) / st_mid) * 6.0, -2.5, 2.5)
    b1_ser = pd.Series(st_b1)
    h_14400 = b1_ser.rolling(14400, min_periods=1).max().values.astype(np.float32)
    l_14400 = b1_ser.rolling(14400, min_periods=1).min().values.astype(np.float32)
    pos_14400 = (st_b1 - l_14400) / np.maximum(h_14400 - l_14400, 1e-4)
    range_pos_4h = np.clip(2.0 * pos_14400 - 1.0, -1.0, 1.0).astype(np.float32)
    range_amp_4h = np.clip(((h_14400 - l_14400) / np.maximum(l_14400, 1e-4)) * 4.0, 0.0, 2.5).astype(np.float32)
    block5_macro_4d = np.column_stack([ret_15m, ret_1h, range_pos_4h, range_amp_4h]).astype(np.float32)

    print("🧩 Итоговая компоновка каузального 44D вектора...")
    hybrid_40d = np.hstack([block1_st_12d, block2_bo_10d, block3_macro_12d, alpha_h6, block5_macro_4d]).astype(np.float32)
    hybrid_40d = np.nan_to_num(hybrid_40d, nan=0.0, posinf=2.5, neginf=-2.5)

    print(f"💾 Атомарное сохранение {hybrid_40d.shape} в {OUT_PATH.name}...")
    tmp_out = OUT_PATH.with_name(f"{OUT_PATH.stem}_tmp.npz")
    np.savez_compressed(tmp_out, market_obs_40d=hybrid_40d)
    tmp_out.replace(OUT_PATH)

    dt_s = time.time() - t0
    sz_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print("=" * 80)
    print(f"✅ СОБРАН ЧИСТЫЙ КАУЗАЛЬНЫЙ 44D ВЕКТОР ({hybrid_40d.shape}, {sz_mb:.1f} МБ) за {dt_s:.1f}с (0% lookahead)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
