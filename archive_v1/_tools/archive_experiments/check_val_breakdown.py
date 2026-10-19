import numpy as np, pandas as pd, pyarrow.parquet as pq

tbl = pq.read_table('data/features/prl_1s_features.parquet', columns=['ST_mid_price'])
df = tbl.to_pandas()
mid = df['ST_mid_price'].values
TICK_SIZE = 0.005
H = 5
dt = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)
split_idx = int(len(df) * 0.8)
tr_dt = dt[:split_idx - H]
va_dt = dt[split_idx - H:]

print("Train sum of dt >= 2:", np.sum(tr_dt[tr_dt >= 2]), "count:", len(tr_dt[tr_dt >= 2]), "mean:", np.mean(tr_dt[tr_dt >= 2]))
print("Val sum of dt >= 2:", np.sum(va_dt[va_dt >= 2]), "count:", len(va_dt[va_dt >= 2]), "mean:", np.mean(va_dt[va_dt >= 2]))

# Let's inspect what happens to the mean if we exclude or if we change threshold
val_ge2 = va_dt[va_dt >= 2]
print("Val distribution of dt >= 2:")
print(pd.Series(val_ge2).value_counts().sort_index())
