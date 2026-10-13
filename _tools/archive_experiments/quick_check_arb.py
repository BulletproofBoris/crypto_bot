import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    feat_path = "data/features/prl_1s_features.parquet"
    if not os.path.exists(feat_path):
        feat_path = "data/processed/prl_1s_features.parquet"
        
    print(f"Loading features from: {feat_path}")
    df = pd.read_parquet(feat_path)
    print(f"Data shape: {df.shape}")
    print(f"Index: {df.index.min()} to {df.index.max()}")
    
    # Check key columns
    cols = [
        'ST_bid_1_price', 'ST_ask_1_price', 'ST_eff_buy_vwap_100', 'ST_eff_sell_vwap_100',
        'BO_bid_1_price', 'BO_ask_1_price', 'BO_eff_buy_vwap_100', 'BO_eff_sell_vwap_100',
        'ST_bids_usd_top5', 'ST_asks_usd_top5', 'BO_bids_usd_top5', 'BO_asks_usd_top5'
    ]
    avail_cols = [c for c in cols if c in df.columns]
    print(f"Available orderbook columns: {avail_cols}")
    
    # Check price spread
    if 'ST_bid_1_price' in df.columns and 'BO_ask_1_price' in df.columns:
        # Opportunity 1: Buy on BigONE (at BO_ask), Sell on SafeTrade (at ST_bid)
        df['spread_buy_bo_sell_st'] = (df['ST_bid_1_price'] - df['BO_ask_1_price']) / df['BO_ask_1_price']
        # Opportunity 2: Buy on SafeTrade (at ST_ask), Sell on BigONE (at BO_bid)
        df['spread_buy_st_sell_bo'] = (df['BO_bid_1_price'] - df['ST_ask_1_price']) / df['ST_ask_1_price']
        
        # Effective spread taking into account top orderbook slippage on $100
        if 'ST_eff_sell_vwap_100' in df.columns and 'BO_eff_buy_vwap_100' in df.columns:
            df['eff_spread_buy_bo_sell_st'] = (df['ST_eff_sell_vwap_100'] - df['BO_eff_buy_vwap_100']) / df['BO_eff_buy_vwap_100']
            df['eff_spread_buy_st_sell_bo'] = (df['BO_eff_sell_vwap_100'] - df['ST_eff_buy_vwap_100']) / df['ST_eff_buy_vwap_100']
        
        # Fees: SafeTrade ~0.1% (maker) / 0.2% (taker), BigONE ~0.1% / 0.2%
        # If taker on both: roundtrip fee = 0.2% + 0.2% = 0.4%
        taker_fee_total = 0.004
        
        opp1 = (df['spread_buy_bo_sell_st'] > taker_fee_total).sum()
        opp2 = (df['spread_buy_st_sell_bo'] > taker_fee_total).sum()
        
        print("\n--- RAW BBO ARBITRAGE OPPORTUNITIES (Spread > 0.4% fees) ---")
        print(f"Total 1-second ticks: {len(df)}")
        print(f"Ticks Buy BigONE & Sell SafeTrade: {opp1} ({opp1 / len(df) * 100:.2f}%)")
        print(f"Ticks Buy SafeTrade & Sell BigONE: {opp2} ({opp2 / len(df) * 100:.2f}%)")
        print(f"Total arb ticks: {opp1 + opp2} ({(opp1 + opp2) / len(df) * 100:.2f}%)")
        
        if 'eff_spread_buy_bo_sell_st' in df.columns:
            eff_opp1 = (df['eff_spread_buy_bo_sell_st'] > taker_fee_total).sum()
            eff_opp2 = (df['eff_spread_buy_st_sell_bo'] > taker_fee_total).sum()
            print("\n--- EFFECTIVE VWAP $100 ARBITRAGE (Depth-adjusted) ---")
            print(f"Ticks Buy BigONE & Sell SafeTrade: {eff_opp1} ({eff_opp1 / len(df) * 100:.2f}%)")
            print(f"Ticks Buy SafeTrade & Sell BigONE: {eff_opp2} ({eff_opp2 / len(df) * 100:.2f}%)")
            print(f"Total effective arb ticks: {eff_opp1 + eff_opp2} ({(eff_opp1 + eff_opp2) / len(df) * 100:.2f}%)")
            
        # Summary statistics of spreads
        print("\nSpread distribution (Buy BO, Sell ST):")
        print(df['spread_buy_bo_sell_st'].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
        print("\nSpread distribution (Buy ST, Sell BO):")
        print(df['spread_buy_st_sell_bo'].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

if __name__ == '__main__':
    main()
