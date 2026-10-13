import numpy as np, pandas as pd, pyarrow.parquet as pq

cols = ['ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5', 'ST_delta_vol']
tbl = pq.read_table('data/features/prl_1s_features.parquet', columns=cols)
df = tbl.to_pandas()
n = len(df)
split_idx = int(n * 0.8)
TICK_SIZE = 0.005
H = 5
L = 20

mid = df['ST_mid_price'].values
dt = np.zeros(n, dtype=int)
dt[:-H] = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)

step_1s = np.zeros(n)
step_1s[1:] = (mid[1:] - mid[:-1]) / TICK_SIZE
vol_L = pd.Series(step_1s).rolling(L, min_periods=L).std().values

df['dt'] = dt
df['vol_L'] = vol_L

df_val = df.iloc[split_idx:].copy()

big_moves = df_val[df_val['dt'] >= 10]
print(f"Total points with dt >= 10 on Val: {len(big_moves)}")
print(big_moves[['ST_mid_price', 'dt', 'vol_L', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5']].head(20))
