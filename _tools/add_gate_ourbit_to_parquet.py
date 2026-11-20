#!/usr/bin/env python3
"""
Добавляет Gate.io и Ourbit каналы к существующему prl_1s_features.parquet.
Вместо полной пересборки (OOM) — инкрементальное расширение.
"""
import sys, os, gc, time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PARQUET = os.path.join(BASE_DIR, "data", "features", "prl_1s_features.parquet")

def process_venue_to_1s(depth_path, trades_path, prefix):
    """Обрабатывает depth+trades одной биржи в 1s DataFrame с prefix."""
    import pyarrow.parquet as pq
    
    print(f"  Loading {prefix} depth...")
    depth = pd.read_parquet(depth_path)
    depth['timestamp'] = pd.to_datetime(depth['timestamp'], utc=True)
    depth = depth.set_index('timestamp').sort_index()
    depth = depth.tz_convert('Europe/Moscow')
    
    # Resample to 1s
    depth_1s = depth.resample('1s').ffill()
    
    # Compute mid_price, spread, etc.
    b1p = depth_1s['bid_1_price'].astype(float)
    a1p = depth_1s['ask_1_price'].astype(float)
    b1q = depth_1s['bid_1_qty'].astype(float)
    a1q = depth_1s['ask_1_qty'].astype(float)
    
    result = pd.DataFrame(index=depth_1s.index)
    result[f'{prefix}_mid_price'] = (b1p + a1p) / 2.0
    result[f'{prefix}_spread'] = a1p - b1p
    result[f'{prefix}_spread_pct'] = result[f'{prefix}_spread'] / result[f'{prefix}_mid_price'] * 100
    result[f'{prefix}_bid_1_price'] = b1p
    result[f'{prefix}_ask_1_price'] = a1p
    result[f'{prefix}_bid_1_qty'] = b1q
    result[f'{prefix}_ask_1_qty'] = a1q
    
    # Top-5 volumes
    for side in ('bid', 'ask'):
        vol5 = sum(depth_1s[f'{side}_{i}_qty'].astype(float).fillna(0) for i in range(1, 6))
        result[f'{prefix}_{side}s_vol_top5'] = vol5
        usd5 = sum(depth_1s[f'{side}_{i}_price'].astype(float) * depth_1s[f'{side}_{i}_qty'].astype(float).fillna(0) for i in range(1, 6))
        result[f'{prefix}_{side}s_usd_top5'] = usd5
    
    # Imbalance top5
    result[f'{prefix}_imbalance_top5'] = (
        (result[f'{prefix}_bids_vol_top5'] - result[f'{prefix}_asks_vol_top5']) /
        (result[f'{prefix}_bids_vol_top5'] + result[f'{prefix}_asks_vol_top5'] + 0.01)
    )
    
    # Level 2-5 prices/qtys
    for i in range(2, 6):
        for side in ('bid', 'ask'):
            pc = f'{side}_{i}_price'
            qc = f'{side}_{i}_qty'
            if pc in depth_1s.columns:
                result[f'{prefix}_{pc}'] = depth_1s[pc].astype(float)
            if qc in depth_1s.columns:
                result[f'{prefix}_{qc}'] = depth_1s[qc].astype(float)
    
    # Level 5 price for book slope
    if 'bid_5_price' in depth_1s.columns:
        result[f'{prefix}_bid_5_price'] = depth_1s['bid_5_price'].astype(float)
    if 'ask_5_price' in depth_1s.columns:
        result[f'{prefix}_ask_5_price'] = depth_1s['ask_5_price'].astype(float)
    
    # Top-10 volumes (for convexity)
    for side in ('bid', 'ask'):
        vol10 = sum(depth_1s[f'{side}_{i}_qty'].astype(float).fillna(0) 
                     for i in range(1, 11) if f'{side}_{i}_qty' in depth_1s.columns)
        # Store as vol_11_50 placeholder (just top6-10 minus top5)
        result[f'{prefix}_{side}s_vol_11_50'] = 0.0  # no deep data needed
    
    del depth, depth_1s
    gc.collect()
    
    # Trades
    if os.path.exists(trades_path):
        print(f"  Loading {prefix} trades...")
        trades = pd.read_parquet(trades_path)
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], utc=True)
        trades = trades.set_index('timestamp').sort_index()
        trades = trades.tz_convert('Europe/Moscow')
        
        price = trades['price'].astype(float)
        amount = trades['amount'].astype(float)
        side_col = trades['side'].astype(str).str.upper() if 'side' in trades.columns else trades.get('type', pd.Series('', index=trades.index)).astype(str).str.upper()
        is_buy = side_col.isin(['BUY', 'ASK'])
        
        trades_agg = pd.DataFrame(index=trades.index)
        trades_agg['buy_vol'] = np.where(is_buy, amount, 0.0)
        trades_agg['sell_vol'] = np.where(~is_buy, amount, 0.0)
        trades_agg['trade_vol'] = amount
        trades_agg['trade_count'] = 1.0
        trades_agg['delta_vol'] = trades_agg['buy_vol'] - trades_agg['sell_vol']
        
        tr_1s = trades_agg.resample('1s').sum().fillna(0.0)
        # Shift by 1s for causality
        tr_1s = tr_1s.shift(1).fillna(0.0)
        
        for c in tr_1s.columns:
            result[f'{prefix}_{c}'] = tr_1s[c].reindex(result.index, fill_value=0.0)
        
        del trades, trades_agg, tr_1s
        gc.collect()
    else:
        for c in ['buy_vol', 'sell_vol', 'trade_vol', 'trade_count', 'delta_vol']:
            result[f'{prefix}_{c}'] = 0.0
    
    # sec_since_tick
    has_update = result[f'{prefix}_mid_price'].diff().abs() > 1e-8
    grp = has_update.cumsum()
    result[f'{prefix}_sec_since_tick'] = grp.groupby(grp).cumcount().astype(float)
    
    print(f"  {prefix}: {len(result)} rows, {len(result.columns)} cols, range {result.index[0]} - {result.index[-1]}")
    return result.astype(np.float32, errors='ignore')


def main():
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    t0 = time.time()
    
    # 1. Load existing parquet index (timestamps only)
    print("Loading existing parquet timestamps...")
    existing = pq.read_table(PARQUET, columns=['ST_mid_price']).to_pandas()
    existing_idx = existing.index
    print(f"  Existing: {len(existing)} rows")
    del existing
    gc.collect()
    
    # 2. Process Gate and Ourbit
    processed = os.path.join(BASE_DIR, "data", "processed", "prl")
    venues = []
    
    for prefix, depth_name, trades_name in [
        ('GATE', 'gate_depth.parquet', 'gate_trades.parquet'),
        ('OUR', 'ourbit_depth.parquet', 'ourbit_trades.parquet'),
    ]:
        depth_path = os.path.join(processed, depth_name)
        trades_path = os.path.join(processed, trades_name)
        if os.path.exists(depth_path):
            venue_df = process_venue_to_1s(depth_path, trades_path, prefix)
            venues.append((prefix, venue_df))
            gc.collect()
    
    if not venues:
        print("No Gate/Ourbit data found!")
        return
    
    # 3. Load existing parquet in chunks and merge
    print("Loading full parquet for merge...")
    df = pq.read_table(PARQUET).to_pandas()
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} cols")
    
    # Восстановим DatetimeIndex из backup (потерян при recalc_tick_features.py)
    if not isinstance(df.index, pd.DatetimeIndex):
        BACKUP = PARQUET + ".bak_tick005"
        if os.path.exists(BACKUP):
            print("  Restoring DatetimeIndex from backup...")
            ts_idx = pd.read_parquet(BACKUP, columns=[]).index
            if len(ts_idx) == len(df):
                df.index = ts_idx
            else:
                print(f"  WARNING: backup has {len(ts_idx)} rows vs {len(df)}, skipping restore")
        else:
            print("  WARNING: no backup found and no DatetimeIndex!")
    print(f"  Index type: {type(df.index).__name__}")
    
    # Remove old Gate/Ourbit columns if any
    old_cols = [c for c in df.columns if c.startswith('GATE_') or c.startswith('OUR_') 
                or c.startswith('arb_gate') or c.startswith('arb_our')
                or c.startswith('leadlag_gate') or c.startswith('leadlag_our')
                or c.startswith('gate_') or c.startswith('our_')]
    if old_cols:
        df = df.drop(columns=old_cols)
        print(f"  Removed {len(old_cols)} old Gate/Ourbit columns")
    
    # 4. Merge venue data
    for prefix, venue_df in venues:
        print(f"  Merging {prefix}...")
        # Align to existing index
        venue_aligned = venue_df.reindex(df.index)
        # ffill within reasonable limit (60s for Gate, 10s for Ourbit)
        max_ffill = 60 if prefix == 'GATE' else 10
        venue_aligned = venue_aligned.ffill(limit=max_ffill).fillna(0.0)
        
        for c in venue_aligned.columns:
            df[c] = venue_aligned[c].values
        
        del venue_df, venue_aligned
        gc.collect()
    
    # 5. Compute filter bank for new venues
    print("Computing filter banks for GATE/OUR...")
    from _tools.microstructure_features import add_exchange_filter_bank, add_cross_exchange_and_macro_filters
    
    for prefix, _ in venues:
        if f'{prefix}_mid_price' in df.columns and df[f'{prefix}_mid_price'].sum() > 0:
            print(f"  Filter bank for {prefix}...")
            df = add_exchange_filter_bank(df, prefix, coin='prl')
            gc.collect()
    
    # 6. Recompute cross-exchange features (includes new Gate/Ourbit arb signals)
    print("Recomputing cross-exchange features...")
    # Remove old cross-exchange cols to recompute
    cross_cols = [c for c in df.columns if any(c.startswith(p) for p in 
                  ['arb_', 'leadlag_', 'bo_volume', 'gate_volume', 'our_volume',
                   'cross_', 'macro_', 'hour_', 'market_activity', 'ST_time_since',
                   'ST_flow_regime', 'ST_trades_burst', 'onchain_'])]
    df = df.drop(columns=[c for c in cross_cols if c in df.columns], errors='ignore')
    df = add_cross_exchange_and_macro_filters(df, coin='prl')
    df = df.fillna(0.0)
    gc.collect()
    
    # 7. Float32 optimization
    keep_f64 = {
        'ST_mid_price', 'ST_spread', 'ST_bid_1_price', 'ST_ask_1_price',
        'BO_mid_price', 'BO_spread', 'BO_bid_1_price', 'BO_ask_1_price',
        'GATE_mid_price', 'GATE_spread', 'GATE_bid_1_price', 'GATE_ask_1_price',
        'OUR_mid_price', 'OUR_spread', 'OUR_bid_1_price', 'OUR_ask_1_price',
    }
    f64_cols = [c for c in df.columns if df[c].dtype == np.float64 and c not in keep_f64]
    if f64_cols:
        df[f64_cols] = df[f64_cols].astype(np.float32)
    
    # 8. Save
    print(f"Saving {len(df)} rows, {len(df.columns)} cols...")
    tbl = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(tbl, PARQUET, compression='zstd')
    size_mb = os.path.getsize(PARQUET) / (1024**2)
    
    # Verify
    gate_nonzero = (df.get('GATE_mid_price', pd.Series(0)) > 0).sum()
    our_nonzero = (df.get('OUR_mid_price', pd.Series(0)) > 0).sum()
    
    print(f"\n=== DONE in {time.time()-t0:.0f}s ===")
    print(f"  File: {size_mb:.1f} MB, {len(df)} rows, {len(df.columns)} cols")
    print(f"  GATE coverage: {gate_nonzero:,} rows ({gate_nonzero/len(df)*100:.1f}%)")
    print(f"  OUR coverage: {our_nonzero:,} rows ({our_nonzero/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
