#!/usr/bin/env python3
"""
Test and compare class balance under tick_size=0.005 vs tick_size=0.0001
for SafeTrade (ST) and BigONE (BO) across H in [1, 2, 5, 20]s.
"""

import os
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def test_label_balance():
    parquet_path = "data/features/prl_1s_features.parquet"
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    
    # Берем последние 100 000 строк (свежие данные)
    needed = 100_000
    cols = [
        "ST_mid_price", "ST_spread", "ST_bid_1_price", "ST_ask_1_price",
        "ST_bid_1_qty", "ST_ask_1_qty", "ST_bids_vol_top5", "ST_asks_vol_top5", "ST_trade_vol",
        "BO_mid_price", "BO_spread", "BO_bid_1_price", "BO_ask_1_price",
        "BO_bid_1_qty", "BO_ask_1_qty", "BO_bids_vol_top5", "BO_asks_vol_top5", "BO_trade_vol"
    ]
    df = pq.read_table(parquet_path, columns=cols).to_pandas().iloc[-needed:].reset_index(drop=True)
    
    from _tools.data_labeler import compute_single_horizon_labels

    print("=" * 80)
    print("⚖️ СРАВНИТЕЛЬНЫЙ АНАЛИЗ БАЛАНСА 7 КЛАССОВ (СТАРЫЙ 0.005 vs НОВЫЙ 0.0001)")
    print("=" * 80)

    for H in [1, 2, 5, 20]:
        K = 3.0
        print(f"\n--- ГОРИЗОНТ H = {H}s ---")
        
        # 1. Старый tick_size = 0.005 (SafeTrade)
        lbl_old = compute_single_horizon_labels(df, "prl", H=H, K=K, min_vol=50.0, tick_size=0.005)
        c7_old = lbl_old["class_7"]
        cnt_old = pd.Series(c7_old).value_counts(normalize=True).sort_index() * 100.0

        # 2. Новый субцентовый tick_size = 0.0001 (SafeTrade)
        lbl_new_st = compute_single_horizon_labels(df, "prl", H=H, K=K, min_vol=50.0, tick_size=0.0001)
        c7_new_st = lbl_new_st["class_7"]
        cnt_new_st = pd.Series(c7_new_st).value_counts(normalize=True).sort_index() * 100.0

        print(f"  • SafeTrade (Старый 0.005$): Класс 0 = {cnt_old.get(0, 0):.1f}% | Классы ±1 = {cnt_old.get(1,0)+cnt_old.get(-1,0):.1f}% | Классы ±2,±3 = {cnt_old.get(2,0)+cnt_old.get(3,0)+cnt_old.get(-2,0)+cnt_old.get(-3,0):.2f}%")
        print(f"  • SafeTrade (Новый 0.0001$):  Класс 0 = {cnt_new_st.get(0, 0):.1f}% | Классы ±1 = {cnt_new_st.get(1,0)+cnt_new_st.get(-1,0):.1f}% | Классы ±2,±3 = {cnt_new_st.get(2,0)+cnt_new_st.get(3,0)+cnt_new_st.get(-2,0)+cnt_new_st.get(-3,0):.2f}%")

if __name__ == "__main__":
    test_label_balance()
