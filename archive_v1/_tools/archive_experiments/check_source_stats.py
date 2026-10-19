import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def analyze_sources():
    sources = {
        'SafeTrade PRL L2 Depth': 'data/processed/safetrade_depth.parquet',
        'SafeTrade PRL Trades': 'data/processed/safetrade_trades.parquet',
        'BigONE PRL L2 Depth': 'data/processed/bigone_depth.parquet',
        'BigONE PRL Trades': 'data/processed/bigone_trades.parquet',
        'Binance Macro (BTC/NEAR)': 'data/processed/binance_macro.parquet',
        'Pearl On-Chain Metrics': 'data/processed/prl_onchain.parquet',
        'Pearl Historical Blocks': 'data/metadata/pearl_historical_blocks.csv',
        'Merged Unified 1s Matrix': 'data/features/prl_1s_features.parquet',
    }
    
    print("=" * 115)
    print("📊 ДЕТАЛЬНЫЙ АУДИТ ДАННЫХ ИЗ ВСЕХ ИСТОЧНИКОВ И МАКСИМАЛЬНЫХ ОКОН L")
    print("=" * 115)
    
    records = []
    
    for name, path in sources.items():
        if not os.path.exists(path):
            print(f"Файл {path} не найден!")
            continue
            
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        
        if path.endswith('.parquet'):
            pf = pq.ParquetFile(path)
            num_rows = pf.metadata.num_rows
            num_cols = pf.metadata.num_columns
            
            time_cols = [c for c in pf.schema.names if 'time' in c.lower() or 'date' in c.lower() or 'ts' in c.lower()]
            if time_cols:
                tbl = pq.read_table(path, columns=[time_cols[0]])
                s = tbl.to_pandas()[time_cols[0]]
                if pd.api.types.is_datetime64_any_dtype(s):
                    dt_series = s
                elif pd.api.types.is_numeric_dtype(s):
                    if s.max() > 1e11:
                        dt_series = pd.to_datetime(s, unit='ms', utc=True)
                    else:
                        dt_series = pd.to_datetime(s, unit='s', utc=True)
                else:
                    dt_series = pd.to_datetime(s, utc=True)
                    
                t_min = dt_series.min()
                t_max = dt_series.max()
                duration = t_max - t_min
                duration_sec = duration.total_seconds()
                duration_hours = duration_sec / 3600.0
                duration_days = duration_hours / 24.0
            else:
                t_min = t_max = "N/A"
                duration_sec = duration_hours = duration_days = 0
                
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            num_rows = len(df)
            num_cols = len(df.columns)
            time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'ts' in c.lower()]
            if time_cols:
                s = df[time_cols[0]]
                if pd.api.types.is_numeric_dtype(s):
                    if s.max() > 1e11:
                        dt_series = pd.to_datetime(s, unit='ms', utc=True)
                    else:
                        dt_series = pd.to_datetime(s, unit='s', utc=True)
                else:
                    dt_series = pd.to_datetime(s, utc=True)
                t_min = dt_series.min()
                t_max = dt_series.max()
                duration = t_max - t_min
                duration_sec = duration.total_seconds()
                duration_hours = duration_sec / 3600.0
                duration_days = duration_hours / 24.0
            else:
                t_min = t_max = "N/A"
                duration_sec = duration_hours = duration_days = 0
                
        records.append({
            'name': name,
            'rows': num_rows,
            'cols': num_cols,
            'size_mb': file_size_mb,
            't_min': str(t_min)[:19],
            't_max': str(t_max)[:19],
            'sec': duration_sec,
            'hours': duration_hours,
            'days': duration_days
        })
        
    for r in records:
        print(f"\nИсточник: {r['name']}")
        print(f"  • Количество записей: {r['rows']:,d} строк, {r['cols']} колонок ({r['size_mb']:.2f} MB)")
        print(f"  • Диапазон дат:       {r['t_min']}  -->  {r['t_max']} UTC")
        print(f"  • Общая длительность: {r['sec']:,.0f} сек (~{r['hours']:.1f} ч / ~{r['days']:.2f} дней)")

if __name__ == '__main__':
    analyze_sources()
