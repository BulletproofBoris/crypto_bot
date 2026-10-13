import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

valid_start = L
valid_end = n - H

dt = dt[valid_start:valid_end]
vt = np.nan_to_num(vol_L[valid_start:valid_end])
sp = spread[valid_start:valid_end]
a_cur = asks5[valid_start:valid_end]
a_L = asks_L[valid_start:valid_end]

train_n = split_idx - valid_start
tr_dt, va_dt = dt[:train_n], dt[train_n:]
M_up_tr = (sp[:train_n] + vt[:train_n] * np.sqrt(H)) * (np.clip(a_L[:train_n] / np.maximum(a_cur[:train_n], 50.0), 0.2, 5.0) ** 0.3)
M_up_va = (sp[train_n:] + vt[train_n:] * np.sqrt(H)) * (np.clip(a_L[train_n:] / np.maximum(a_cur[train_n:], 50.0), 0.2, 5.0) ** 0.3)

z_tr = tr_dt[tr_dt > 0] / M_up_tr[tr_dt > 0]
z_va = va_dt[va_dt > 0] / M_up_va[va_dt > 0]

print("=== ПЕРЦЕНТИЛИ Z ДЛЯ ПОЛОЖИТЕЛЬНЫХ ДВИЖЕНИЙ ===")
pcts = [10, 25, 50, 60, 70, 75, 80, 85, 90, 95, 98, 99]
print("Percentile | Train z | Val z   | Diff (%)")
print("-" * 45)
for p in pcts:
    pz_tr = np.percentile(z_tr, p)
    pz_va = np.percentile(z_va, p)
    err = abs(pz_va - pz_tr) / pz_tr * 100
    print(f"   {p:2d}%      | {pz_tr:.4f}  | {pz_va:.4f}  | {err:5.2f}%")
