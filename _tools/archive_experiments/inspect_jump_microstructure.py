import numpy as np, pandas as pd, pyarrow.parquet as pq

cols = [
    'ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
    'ST_slippage_buy_100_pct', 'ST_imbalance_top5', 'BO_mid_price', 'mid_price_diff',
    'ST_trade_vol', 'ST_delta_vol', 'ST_trade_count'
]
tbl = pq.read_table('data/features/prl_1s_features.parquet', columns=cols)
df = tbl.to_pandas()

# Row 267292 was the big jump
sub = df.iloc[267280:267300].copy()
sub['index'] = range(267280, 267300)
sub['dt_5s'] = np.round((df['ST_mid_price'].iloc[267285:267305].values - sub['ST_mid_price'].values) / 0.005)

print(sub[['index', 'ST_mid_price', 'dt_5s', 'ST_spread', 'ST_asks_vol_top5', 'ST_slippage_buy_100_pct', 'BO_mid_price', 'mid_price_diff', 'ST_trade_vol']])
