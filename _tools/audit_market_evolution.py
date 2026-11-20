#!/usr/bin/env python3
"""
_tools/audit_market_evolution.py
=============================================================================
Глубокий исторический аудит эволюции характеристик торговли на площадках (SafeTrade, BigONE, Binance):
  1. Динамика ценовых уровней и шага тика (Tick Size / Price Levels).
  2. Динамика среднего спреда и волатильности по неделям.
  3. Плотность книги заявок (Top-5 / Top-50 Depth) и размеры стен.
  4. Активность сделок (Trade volume & frequency).
  5. Сдвиг лидерства (Lead-Lag & Arbitrage spread).
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def run_evolution_audit():
    parquet_path = Path("data/features/prl_1s_features.parquet")
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"Total dataset rows: {total_rows:,} (~{total_rows/86400:.1f} days)")

    # Читаем ключевые колонки
    cols_to_check = [
        "ST_mid_price", "ST_spread", "ST_bid_1_price", "ST_ask_1_price",
        "ST_bids_vol_top5", "ST_asks_vol_top5", "ST_trade_vol",
        "BO_mid_price", "BO_spread", "BO_bid_1_price", "BO_ask_1_price",
        "BO_bids_vol_top5", "BO_asks_vol_top5", "BO_trade_vol",
        "macro_btc_return_5m", "macro_btc_volatility_5m",
        "arb_spread_st_bo"
    ]
    cols_present = [c for c in cols_to_check if c in pf.schema.names]

    # Анализируем по 5 временным эпохам (неделям)
    chunk_size = total_rows // 5
    epochs = [
        ("Неделя 1 (10-17 Окт)", 0, chunk_size),
        ("Неделя 2 (17-24 Окт)", chunk_size, 2 * chunk_size),
        ("Неделя 3 (24-31 Окт)", 2 * chunk_size, 3 * chunk_size),
        ("Неделя 4 (31 Окт-07 Ноя)", 3 * chunk_size, 4 * chunk_size),
        ("Неделя 5 (07-15 Ноя, Текущая)", 4 * chunk_size, total_rows),
    ]

    print("\n" + "=" * 95)
    print("📊 ИСТОРИЧЕСКИЙ АУДИТ ЭВОЛЮЦИИ РЫНОЧНЫХ СВОЙСТВ (SafeTrade vs BigONE)")
    print("=" * 95)

    stats = []

    for name, start, end in epochs:
        tbl = pf.read_row_group(0) # test
        # Читаем срез через slice
        df = pq.read_table(parquet_path, columns=cols_present).to_pandas().iloc[start:end]
        
        st_mid = df["ST_mid_price"].dropna()
        bo_mid = df["BO_mid_price"].dropna() if "BO_mid_price" in df else None
        st_spr = df["ST_spread"].dropna()
        bo_spr = df["BO_spread"].dropna() if "BO_spread" in df else None
        st_vol = df["ST_bids_vol_top5"].dropna()
        bo_vol = df["BO_bids_vol_top5"].dropna() if "BO_bids_vol_top5" in df else None
        st_tr_vol = df["ST_trade_vol"].dropna()
        arb_spr = df["arb_spread_st_bo"].dropna() if "arb_spread_st_bo" in df else None

        row = {
            "epoch": name,
            "st_price_mean": float(st_mid.mean()),
            "st_price_min": float(st_mid.min()),
            "st_price_max": float(st_mid.max()),
            "st_spread_mean": float(st_spr.mean()),
            "st_spread_p95": float(st_spr.quantile(0.95)),
            "bo_spread_mean": float(bo_spr.mean()) if bo_spr is not None else 0.0,
            "st_depth_top5": float(st_vol.mean()),
            "bo_depth_top5": float(bo_vol.mean()) if bo_vol is not None else 0.0,
            "st_trades_sum": float(st_tr_vol.sum()),
            "arb_spread_mean": float(arb_spr.mean()) if arb_spr is not None else 0.0,
            "arb_spread_abs": float(arb_spr.abs().mean()) if arb_spr is not None else 0.0,
        }
        stats.append(row)

        print(f"\n📍 {name}:")
        print(f"  • SafeTrade Mid Price: avg={row['st_price_mean']:.4f} (диапазон: {row['st_price_min']:.4f} - {row['st_price_max']:.4f})")
        print(f"  • SafeTrade Spread: avg={row['st_spread_mean']:.5f} | 95%-квантиль={row['st_spread_p95']:.5f}")
        print(f"  • BigONE Spread: avg={row['bo_spread_mean']:.5f}")
        print(f"  • Глубина Top-5 Bids: SafeTrade={row['st_depth_top5']:.1f} PRL | BigONE={row['bo_depth_top5']:.1f} PRL")
        print(f"  • Межбиржевой арбитражный спред |ST - BO|: {row['arb_spread_abs']:.5f}")
        print(f"  • Объем рыночных сделок SafeTrade за неделю: {row['st_trades_sum']:,.0f} PRL")

    df_res = pd.DataFrame(stats)
    df_res.to_csv("scratch/variant_a_pipeline_results/market_evolution_stats.csv", index=False)
    print("\n✅ Сводка сохранена в scratch/variant_a_pipeline_results/market_evolution_stats.csv")

if __name__ == "__main__":
    run_evolution_audit()
