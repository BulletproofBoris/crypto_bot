import numpy as np, pandas as pd, pyarrow.parquet as pq

cols = ['ST_mid_price', 'ST_spread', 'ST_bids_vol_top5', 'ST_asks_vol_top5']
tbl = pq.read_table('data/features/prl_1s_features.parquet', columns=cols)
df = tbl.to_pandas()
n = len(df)
split_idx = int(n * 0.8)
TICK_SIZE = 0.005
H = 5
L = 20
mid = df['ST_mid_price'].values
spread = df['ST_spread'].values / TICK_SIZE
asks5 = df['ST_asks_vol_top5'].values
step_1s = (mid[1:] - mid[:-1]) / TICK_SIZE
dt = np.round((mid[H:] - mid[:-H]) / TICK_SIZE).astype(int)
vol_L = pd.Series(step_1s).rolling(L, min_periods=L).std().values
asks_L = pd.Series(asks5).rolling(L, min_periods=L).mean().values

dt = dt[L:n-H]
vt = np.nan_to_num(vol_L[L:n-H])
sp = spread[L:n-H]
a_cur = asks5[L:n-H]
a_L = asks_L[L:n-H]

train_n = split_idx - L
M_up_tr = (sp[:train_n] + vt[:train_n] * np.sqrt(H)) * (np.clip(a_L[:train_n] / np.maximum(a_cur[:train_n], 50.0), 0.2, 5.0) ** 0.3)
M_up_va = (sp[train_n:] + vt[train_n:] * np.sqrt(H)) * (np.clip(a_L[train_n:] / np.maximum(a_cur[train_n:], 50.0), 0.2, 5.0) ** 0.3)

dt_tr = dt[:train_n][dt[:train_n] > 0]
dt_va = dt[train_n:][dt[train_n:] > 0]

z_tr = dt_tr / M_up_tr[dt[:train_n] > 0]
z_va = dt_va / M_up_va[dt[train_n:] > 0]

# Check mean z inside window 75% to 90%:
mask_tr = (z_tr >= np.percentile(z_tr, 75)) & (z_tr <= np.percentile(z_tr, 90))
mask_va = (z_va >= np.percentile(z_va, 75)) & (z_va <= np.percentile(z_va, 90))
m_tr = np.mean(z_tr[mask_tr])
m_va = np.mean(z_va[mask_va])
print(f"Среднее E[z] в интервале 75%..90%: Train = {m_tr:.4f} | Val = {m_va:.4f} | Diff = {abs(m_va-m_tr)/m_tr*100:.2f}%")

# Check mean raw dt inside window 75% to 90%:
m_dt_tr = np.mean(dt_tr[mask_tr])
m_dt_va = np.mean(dt_va[mask_va])
print(f"Среднее E[dt] в интервале 75%..90%: Train = {m_dt_tr:+.2f} тика | Val = {m_dt_va:+.2f} тика | Diff = {abs(m_dt_va-m_dt_tr)/m_dt_tr*100:.2f}%")
