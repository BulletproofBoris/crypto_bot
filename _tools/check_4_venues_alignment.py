#!/usr/bin/env python3
"""Check recent Gate and Ourbit characteristics vs SafeTrade and BigONE."""
import pandas as pd
import gzip

def check_all_4_venues():
    print("Reading hot files...")
    st = pd.read_csv("data/vps_hot/safetrade_depth_hot.csv").tail(5000)
    bo = pd.read_csv("data/vps_hot/bigone_depth_hot.csv").tail(5000)
    gt = pd.read_csv("data/vps_hot/gate_depth_hot.csv").tail(5000)
    ob = pd.read_csv("data/vps_hot/ourbit_depth_hot.csv").tail(5000)

    for name, df in [("SafeTrade", st), ("BigONE", bo), ("Gate.io", gt), ("Ourbit", ob)]:
        # колонки bid_1_price, ask_1_price
        b1 = df["bid_1_price"].astype(float)
        a1 = df["ask_1_price"].astype(float)
        mid = (b1 + a1) / 2.0
        spr = a1 - b1
        print(f"\n📍 {name} (Последние 5000 баров):")
        print(f"  • Mid Price: {mid.mean():.4f} (диапазон: {mid.min():.4f} - {mid.max():.4f})")
        print(f"  • Spread: {spr.mean():.5f} (min={spr.min():.5f}, max={spr.max():.5f})")
        if "bid_1_qty" in df.columns:
            print(f"  • Best Bid Qty: {df['bid_1_qty'].astype(float).mean():.1f}")

if __name__ == "__main__":
    check_all_4_venues()
