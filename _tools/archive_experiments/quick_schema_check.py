import os, sys, pyarrow.parquet as pq, pandas as pd

paths = [
    'data/processed/safetrade_depth.parquet',
    'data/processed/safetrade_trades.parquet',
    'data/processed/bigone_depth.parquet',
    'data/processed/bigone_trades.parquet',
    'data/processed/binance_macro.parquet',
    'data/processed/prl_onchain.parquet',
    'data/metadata/pearl_historical_blocks.csv',
    'data/features/prl_1s_features.parquet'
]

for p in paths:
    if not os.path.exists(p):
        print(f"NOT FOUND: {p}")
        continue
    if p.endswith('.parquet'):
        schema = pq.read_schema(p)
        t_cols = [c for c in schema.names if any(w in c.lower() for w in ['time', 'date', 'ts', 'sec', 'utc'])]
        print(f"{p}: rows={pq.ParquetFile(p).metadata.num_rows}, cols={len(schema.names)}, time_cols={t_cols}")
    else:
        df = pd.read_csv(p, nrows=2)
        t_cols = [c for c in df.columns if any(w in c.lower() for w in ['time', 'date', 'ts', 'sec', 'utc'])]
        print(f"{p}: cols={len(df.columns)}, time_cols={t_cols}")
