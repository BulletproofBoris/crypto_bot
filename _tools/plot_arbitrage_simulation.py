import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_arb_plot():
    st_path = "data/processed/prl/safetrade_depth.parquet"
    bo_path = "data/processed/prl/bigone_depth.parquet"
    
    st_cols = ['timestamp', 'bid_1_price', 'ask_1_price']
    bo_cols = ['timestamp', 'bid_1_price', 'ask_1_price']
    
    df_st = pd.read_parquet(st_path, columns=st_cols)
    df_bo = pd.read_parquet(bo_path, columns=bo_cols)
    
    df_st['timestamp'] = pd.to_datetime(df_st['timestamp']).dt.floor('s')
    df_bo['timestamp'] = pd.to_datetime(df_bo['timestamp']).dt.floor('s')
    
    df_st = df_st.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    df_bo = df_bo.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    
    start_time = max(df_st.index.min(), df_bo.index.min())
    end_time = min(df_st.index.max(), df_bo.index.max())
    
    # 10s downsampling for clean, fast plotting
    common_idx = pd.date_range(start_time, end_time, freq='10s')
    df_st_10s = df_st.reindex(common_idx).ffill()
    df_bo_10s = df_bo.reindex(common_idx).ffill()
    
    st_mid = (df_st_10s['bid_1_price'] + df_st_10s['ask_1_price']) / 2.0
    bo_mid = (df_bo_10s['bid_1_price'] + df_bo_10s['ask_1_price']) / 2.0
    
    spread_pct = (bo_mid - st_mid) / st_mid * 100.0
    
    # Run rapid simulation on 10s
    fee_pct = 0.4 # 0.4% total fee
    
    pnl = []
    cum_pnl = 0.0
    inv_st_prl = 5000.0
    inv_bo_prl = 5000.0
    inv_history = []
    
    for t, row_st, row_bo in zip(common_idx, df_st_10s.itertuples(), df_bo_10s.itertuples()):
        st_ask = row_st.ask_1_price
        st_bid = row_st.bid_1_price
        bo_ask = row_bo.ask_1_price
        bo_bid = row_bo.bid_1_price
        
        # Buy BO, Sell ST
        sp_bo_to_st = (st_bid - bo_ask) / bo_ask * 100.0
        # Buy ST, Sell BO
        sp_st_to_bo = (bo_bid - st_ask) / st_ask * 100.0
        
        trade_usd = 50.0
        if sp_bo_to_st > fee_pct + 0.3 and inv_st_prl > 50:
            gain = trade_usd * ((sp_bo_to_st - fee_pct) / 100.0)
            cum_pnl += gain
            inv_bo_prl += trade_usd / bo_ask
            inv_st_prl -= trade_usd / st_bid
        elif sp_st_to_bo > fee_pct + 0.3 and inv_bo_prl > 50:
            gain = trade_usd * ((sp_st_to_bo - fee_pct) / 100.0)
            cum_pnl += gain
            inv_st_prl += trade_usd / st_ask
            inv_bo_prl -= trade_usd / bo_bid
            
        pnl.append(cum_pnl)
        inv_history.append((inv_st_prl, inv_bo_prl))
        
    inv_df = pd.DataFrame(inv_history, index=common_idx, columns=['SafeTrade_PRL', 'BigONE_PRL'])
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    
    # Subplot 1: Prices
    axes[0].plot(common_idx, st_mid, label='SafeTrade Mid ($)', color='#00d2ff', alpha=0.9, lw=1.2)
    axes[0].plot(common_idx, bo_mid, label='BigONE Mid ($)', color='#ff9900', alpha=0.9, lw=1.2)
    axes[0].set_title('PRL/USDT: SafeTrade vs BigONE Prices (3 Days)', fontsize=13, fontweight='bold', color='white')
    axes[0].set_ylabel('USD Price')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.2)
    
    # Subplot 2: Spread % and Fee Barriers
    axes[1].plot(common_idx, spread_pct, label='Mid Difference (BigONE - SafeTrade) %', color='#bb86fc', lw=0.8)
    axes[1].axhline(fee_pct, color='#00ff88', linestyle='--', lw=1.2, label=f'+{fee_pct}% Taker Fee Barrier')
    axes[1].axhline(-fee_pct, color='#ff3366', linestyle='--', lw=1.2, label=f'-{fee_pct}% Taker Fee Barrier')
    axes[1].fill_between(common_idx, fee_pct, spread_pct, where=(spread_pct > fee_pct), color='#00ff88', alpha=0.3)
    axes[1].fill_between(common_idx, -fee_pct, spread_pct, where=(spread_pct < -fee_pct), color='#ff3366', alpha=0.3)
    axes[1].set_title('Cross-Exchange Spread & Profit Windows', fontsize=12, fontweight='bold', color='white')
    axes[1].set_ylabel('Spread (%)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.2)
    
    # Subplot 3: Cumulative PnL & Inventory
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    p1 = ax3.plot(common_idx, pnl, label='Simultaneous Arbitrage PnL ($)', color='#00ff88', lw=1.8)
    p2 = ax3_twin.plot(common_idx, inv_df['SafeTrade_PRL'], label='SafeTrade PRL Inventory', color='#00d2ff', linestyle=':', lw=1.2)
    p3 = ax3_twin.plot(common_idx, inv_df['BigONE_PRL'], label='BigONE PRL Inventory', color='#ff9900', linestyle=':', lw=1.2)
    
    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    ax3.set_title('Simultaneous Execution: Cumulative Profit vs Inventory Skew', fontsize=12, fontweight='bold', color='white')
    ax3.set_ylabel('Cumulative Profit (USDT)', color='#00ff88')
    ax3_twin.set_ylabel('Exchange PRL Balance', color='#ffaa00')
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out_path = "data_arbitrage_simulation.png"
    plt.savefig(out_path, dpi=130)
    print(f"Chart saved to: {out_path}")

if __name__ == '__main__':
    generate_arb_plot()
