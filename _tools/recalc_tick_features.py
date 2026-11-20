#!/usr/bin/env python3
"""
Пересчитывает все tick-зависимые колонки в prl_1s_features.parquet
после смены TICK_SIZES с 0.005 -> 0.0001.

Вместо полной перегенерации (долго) — читаем сырые цены из parquet,
пересчитываем только зависимые от tick колонки и заменяем их.
"""
import sys, os, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PARQUET = os.path.join(BASE_DIR, "data", "features", "prl_1s_features.parquet")
OLD_TICK = 0.005
NEW_TICK = 0.0001
RATIO = OLD_TICK / NEW_TICK  # 50.0

def main():
    t0 = time.time()
    print(f"Loading parquet: {PARQUET}")
    df = pq.read_table(PARQUET).to_pandas()
    print(f"  Loaded {len(df)} rows, {len(df.columns)} cols in {time.time()-t0:.1f}s")

    # ====== Стратегия: все _ticks колонки были = value/OLD_TICK ======
    # Правильное значение = value/NEW_TICK = (value/OLD_TICK) * (OLD_TICK/NEW_TICK) = old_col * RATIO
    # Но для производных (ratios, slopes) нужен пересчёт через сырые цены.

    # --- Группа 1: Прямые тиковые колонки (просто умножить на RATIO) ---
    direct_tick_cols = []
    for prefix in ("ST", "BO"):
        direct_tick_cols += [
            f"{prefix}_microprice_offset_ticks",
            f"{prefix}_spread_ticks",
        ]
        for k in range(2, 6):
            direct_tick_cols.append(f"{prefix}_bid_{k}_dist_ticks")
            direct_tick_cols.append(f"{prefix}_ask_{k}_dist_ticks")
        for w in (5, 20, 60, 300):
            direct_tick_cols.append(f"{prefix}_ret_{w}s_ticks")
            # vol_*_ticks = std(ret_1s_ticks) → ret_1s_ticks *= RATIO → std *= RATIO
            direct_tick_cols.append(f"{prefix}_vol_{w}s_ticks")

    # Cross-exchange тиковые колонки
    direct_tick_cols.append("arb_mid_diff_ticks")
    for w in (5, 20, 60, 300):
        direct_tick_cols.append(f"leadlag_bo_minus_st_{w}s_ticks")

    rescaled = 0
    for col in direct_tick_cols:
        if col in df.columns:
            df[col] = df[col] * RATIO
            rescaled += 1
        else:
            print(f"  [WARN] Column {col} not found")

    print(f"  Rescaled {rescaled} direct tick columns (×{RATIO})")

    # --- Группа 2: Производные, зависящие от tick через ratio ---
    for prefix in ("ST", "BO"):
        # wave_vel_* = diff(microprice_offset_ticks) → уже пересчитан выше
        # Но wave_vel пересчитываем из пересчитанного offset
        mp = df[f"{prefix}_microprice_offset_ticks"]
        df[f"{prefix}_wave_vel_2s"] = (mp - mp.shift(2)).fillna(0.0)
        df[f"{prefix}_wave_vel_5s"] = ((mp - mp.shift(5)) / 5.0).fillna(0.0)
        df[f"{prefix}_wave_vel_10s"] = ((mp - mp.shift(10)) / 10.0).fillna(0.0)
        df[f"{prefix}_wave_accel_5s"] = (df[f"{prefix}_wave_vel_2s"] - df[f"{prefix}_wave_vel_2s"].shift(3)).fillna(0.0)
        df[f"{prefix}_wave_to_spread_ratio"] = (df[f"{prefix}_wave_vel_5s"] / (df[f"{prefix}_spread_ticks"] + 0.5)).fillna(0.0)

        # book_slope_bid/ask: vol / (dist_in_ticks) → dist *= RATIO → slope /= RATIO
        mid = df[f"{prefix}_mid_price"].astype(float)
        b1_p = df[f"{prefix}_bid_1_price"].astype(float)
        a1_p = df[f"{prefix}_ask_1_price"].astype(float)
        b5_q = df[f"{prefix}_bids_vol_top5"].astype(float)
        a5_q = df[f"{prefix}_asks_vol_top5"].astype(float)
        bp5_col = f"{prefix}_bid_5_price"
        ap5_col = f"{prefix}_ask_5_price"
        bp_5 = df[bp5_col].astype(float) if bp5_col in df.columns else b1_p
        ap_5 = df[ap5_col].astype(float) if ap5_col in df.columns else a1_p
        bid_dist_5 = np.maximum(0.5, (mid - bp_5) / NEW_TICK)
        ask_dist_5 = np.maximum(0.5, (ap_5 - mid) / NEW_TICK)
        bid_slope = b5_q / bid_dist_5
        ask_slope = a5_q / ask_dist_5
        df[f"{prefix}_book_slope_bid"] = bid_slope.fillna(0.0)
        df[f"{prefix}_book_slope_ask"] = ask_slope.fillna(0.0)
        df[f"{prefix}_log_slope_ratio"] = np.log((bid_slope + 1.0) / (ask_slope + 1.0)).fillna(0.0)

        # range_to_spread_20s: range_ticks / (spread_ticks + 0.5)
        range_20s = (mid.rolling(20, min_periods=1).max() - mid.rolling(20, min_periods=1).min()) / NEW_TICK
        df[f"{prefix}_range_to_spread_20s"] = (range_20s / (df[f"{prefix}_spread_ticks"] + 0.5)).fillna(0.0)

        # upside/downside semi vol и rolls_spread_proxy зависят от ret_1s_ticks
        ret_1s_ticks = mid.diff().fillna(0.0) / NEW_TICK
        pos_ret = np.maximum(0.0, ret_1s_ticks)
        neg_ret = np.maximum(0.0, -ret_1s_ticks)
        upside_semi = np.sqrt(pd.Series(pos_ret ** 2, index=df.index).rolling(20, min_periods=1).mean()).fillna(0.0)
        downside_semi = np.sqrt(pd.Series(neg_ret ** 2, index=df.index).rolling(20, min_periods=1).mean()).fillna(0.0)
        df[f"{prefix}_upside_semi_vol_20s"] = upside_semi
        df[f"{prefix}_downside_semi_vol_20s"] = downside_semi
        df[f"{prefix}_volatility_skewness_20s"] = ((upside_semi - downside_semi) / (upside_semi + downside_semi + 0.01)).fillna(0.0)

        dp_prev = ret_1s_ticks.shift(1).fillna(0.0)
        cov_60s = (ret_1s_ticks * dp_prev).rolling(60, min_periods=5).mean() - \
                  ret_1s_ticks.rolling(60, min_periods=5).mean() * dp_prev.rolling(60, min_periods=5).mean()
        rolls_spread = 2.0 * np.sqrt(np.maximum(0.0, -cov_60s.fillna(0.0)))
        df[f"{prefix}_rolls_spread_proxy"] = rolls_spread

        # vol_ratio и noise_to_signal зависят от vol_*_ticks (уже пересчитаны)
        df[f"{prefix}_vol_ratio_5_60"] = df[f"{prefix}_vol_5s_ticks"] / (df[f"{prefix}_vol_60s_ticks"] + 0.05)
        df[f"{prefix}_vol_ratio_20_300"] = df[f"{prefix}_vol_20s_ticks"] / (df[f"{prefix}_vol_300s_ticks"] + 0.05)
        df[f"{prefix}_noise_to_signal_bounce_ratio"] = (rolls_spread / (df[f"{prefix}_vol_60s_ticks"] + 0.05)).fillna(0.0)

    print(f"  Recalculated derived columns for ST and BO")

    # --- Группа 3: Cross-exchange производные ---
    if "arb_mid_diff_ticks" in df.columns:
        diff_ticks = df["arb_mid_diff_ticks"]  # already rescaled
        for w in (5, 20, 60, 300):
            col = f"leadlag_bo_minus_st_{w}s_ticks"
            # Уже пересчитаны в Группе 1

    # --- Сохраняем ---
    backup = PARQUET + ".bak_tick005"
    if not os.path.exists(backup):
        os.rename(PARQUET, backup)
        print(f"  Backup saved: {backup}")
    else:
        print(f"  Backup already exists: {backup}")

    print(f"  Writing updated parquet...")
    t1 = time.time()
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, PARQUET, compression="zstd")
    print(f"  Written in {time.time()-t1:.1f}s")

    # --- Верификация ---
    df2 = pq.read_table(PARQUET, columns=["ST_spread_ticks", "BO_spread_ticks", "ST_microprice_offset_ticks", "ST_ret_5s_ticks", "arb_mid_diff_ticks"]).to_pandas()
    print(f"\n=== ВЕРИФИКАЦИЯ (tick={NEW_TICK}) ===")
    for c in df2.columns:
        print(f"  {c}: mean={df2[c].mean():.2f}, std={df2[c].std():.2f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("DONE")

if __name__ == "__main__":
    main()
