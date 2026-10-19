import os, sys, pyarrow.parquet as pq, pandas as pd

paths = [
    ('SafeTrade L2 Depth (PRL)', 'data/processed/safetrade_depth.parquet'),
    ('SafeTrade Trades (PRL)', 'data/processed/safetrade_trades.parquet'),
    ('BigONE L2 Depth (PRL)', 'data/processed/bigone_depth.parquet'),
    ('BigONE Trades (PRL)', 'data/processed/bigone_trades.parquet'),
    ('Binance Macro 1s (BTC/NEAR)', 'data/processed/binance_macro.parquet'),
    ('Pearl On-Chain Metrics', 'data/processed/prl_onchain.parquet'),
    ('Pearl Historical Blocks', 'data/metadata/pearl_historical_blocks.csv'),
    ('Unified 1s Feature Lake', 'data/features/prl_1s_features.parquet')
]

print("=" * 115)
print("📊 ТОЧНЫЙ АУДИТ ДАТ И ВРЕМЕННЫХ ИНТЕРВАЛОВ ПО ВСЕМ ИСТОЧНИКАМ")
print("=" * 115)

for name, p in paths:
    if p.endswith('.parquet'):
        tbl = pq.read_table(p, columns=['timestamp'])
        s = tbl.column('timestamp').to_pandas()
        rows = len(s)
    else:
        df = pd.read_csv(p, usecols=['timestamp'])
        s = df['timestamp']
        rows = len(s)
        
    # Convert s to datetime
    if pd.api.types.is_numeric_dtype(s):
        if s.max() > 1e11:
            dt = pd.to_datetime(s, unit='ms', utc=True)
        else:
            dt = pd.to_datetime(s, unit='s', utc=True)
    else:
        dt = pd.to_datetime(s, utc=True)
        
    t_min = dt.min()
    t_max = dt.max()
    dur_sec = (t_max - t_min).total_seconds()
    dur_h = dur_sec / 3600.0
    dur_d = dur_h / 24.0
    
    print(f"\nИсточник: {name}")
    print(f"  Файл:         {p}")
    print(f"  Записей:      {rows:,d} строк")
    print(f"  Начало:       {t_min.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Конец:        {t_max.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Длительность: {dur_sec:,.0f} сек (~{dur_h:.1f} ч / ~{dur_d:.2f} дней)")
