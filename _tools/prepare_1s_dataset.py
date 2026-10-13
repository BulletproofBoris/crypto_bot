#!/usr/bin/env python3
"""
Генерация 1-секундного датасета признаков (Features Engineering).
Объединяет в синхронный 1-секундный таймлайн:
- Стаканы L2 и сделки бирж (SafeTrade, BigONE, Rhea Finance)
- Внутрибиржевые спреды, дисбаланс объемов (Imbalance top-5) и объемы торгов
- Межбиржевые арбитражные спреды (Cross-Exchange Arbitrage)
- Ончейн-метрики (высота блока, размер мемпула, количество майнеров)
- Мировые макро-поводыри Binance (BTC/USDT, NEAR/USDT)

Поддерживаемые монеты: PRL, QUANTUS, TSC (или --coin all).
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from colorama import init, Fore, Style

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.data_loader import load_coin, load_macro, sync_data

init(autoreset=True)

def calc_effective_vwap(df, side='ask', target_usd=100.0, max_levels=10):
    """
    Рассчитывает средневзвешенную цену исполнения (VWAP) для заданного объема в USD.
    side='ask' -> покупка монет (по аскам снизу вверх)
    side='bid' -> продажа монет (по бидам сверху вниз)
    """
    prices = [df[f'{side}_{i}_price'].values.astype(float) for i in range(1, max_levels+1) if f'{side}_{i}_price' in df.columns]
    qtys = [df[f'{side}_{i}_qty'].values.astype(float) for i in range(1, max_levels+1) if f'{side}_{i}_qty' in df.columns]

    if not prices or not qtys:
        return np.full(len(df), np.nan)

    n_rows = len(df)
    filled_usd = np.zeros(n_rows, dtype=float)
    filled_qty = np.zeros(n_rows, dtype=float)

    for p, q in zip(prices, qtys):
        level_usd = p * q
        needed_usd = target_usd - filled_usd
        take_usd = np.clip(needed_usd, 0.0, level_usd)
        with np.errstate(divide='ignore', invalid='ignore'):
            take_qty = np.where(p > 0, take_usd / p, 0.0)

        filled_usd += take_usd
        filled_qty += take_qty

    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = np.where(filled_qty > 0, target_usd / filled_qty, np.nan)
    return vwap

def process_exchange_l2(depth_df, trades_df, prefix):
    """Преобразует стакан L2 и сделки в 1-секундный поток признаков биржи."""
    print(f"  [{prefix}] Ресемплинг стакана и сделок до 1 секунды...")

    if depth_df is None or depth_df.empty:
        raise ValueError(f"Стакан для {prefix} пуст!")

    df_d = depth_df.copy()
    df_t = trades_df.copy() if trades_df is not None and not trades_df.empty else pd.DataFrame()

    # 1. Приведение к московскому времени
    if df_d['timestamp'].dt.tz is None:
        df_d['timestamp'] = df_d['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
    else:
        df_d['timestamp'] = df_d['timestamp'].dt.tz_convert('Europe/Moscow')

    df_d = df_d.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    df_d.set_index('timestamp', inplace=True)

    # 2. Ресемплинг стакана (1s, ffill)
    depth_cols = [c for c in df_d.columns if 'bid' in c or 'ask' in c]
    depth_1s = df_d[depth_cols].resample('1s').ffill()

    # 3. Базовые фичи стакана
    best_bid = depth_1s['bid_1_price'].astype(float)
    best_ask = depth_1s['ask_1_price'].astype(float)
    mid_price = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    spread_pct = spread / mid_price.replace(0, np.nan)

    # Дисбаланс стакана (топ-5 и топ-10 уровней)
    bids_vol_top5 = sum(depth_1s[f'bid_{i}_qty'].astype(float) for i in range(1, 6) if f'bid_{i}_qty' in depth_1s.columns)
    asks_vol_top5 = sum(depth_1s[f'ask_{i}_qty'].astype(float) for i in range(1, 6) if f'ask_{i}_qty' in depth_1s.columns)
    total_vol_top5 = bids_vol_top5 + asks_vol_top5
    ob_imbalance_top5 = (bids_vol_top5 - asks_vol_top5) / total_vol_top5.replace(0, np.nan)

    bids_vol_top10 = sum(depth_1s[f'bid_{i}_qty'].astype(float) for i in range(1, 11) if f'bid_{i}_qty' in depth_1s.columns)
    asks_vol_top10 = sum(depth_1s[f'ask_{i}_qty'].astype(float) for i in range(1, 11) if f'ask_{i}_qty' in depth_1s.columns)
    total_vol_top10 = bids_vol_top10 + asks_vol_top10
    ob_imbalance_top10 = (bids_vol_top10 - asks_vol_top10) / total_vol_top10.replace(0, np.nan)

    # Объемы в USD на топ-5
    bids_usd_top5 = sum(depth_1s[f'bid_{i}_price'].astype(float) * depth_1s[f'bid_{i}_qty'].astype(float) for i in range(1, 6) if f'bid_{i}_qty' in depth_1s.columns)
    asks_usd_top5 = sum(depth_1s[f'ask_{i}_price'].astype(float) * depth_1s[f'ask_{i}_qty'].astype(float) for i in range(1, 6) if f'ask_{i}_qty' in depth_1s.columns)

    depth_1s['mid_price'] = mid_price
    depth_1s['spread'] = spread
    depth_1s['spread_pct'] = spread_pct
    depth_1s['bids_vol_top5'] = bids_vol_top5
    depth_1s['asks_vol_top5'] = asks_vol_top5
    depth_1s['bids_usd_top5'] = bids_usd_top5
    depth_1s['asks_usd_top5'] = asks_usd_top5
    depth_1s['imbalance_top5'] = ob_imbalance_top5.fillna(0)
    depth_1s['imbalance_top10'] = ob_imbalance_top10.fillna(0)

    # Эффективная цена исполнения и проскальзывание (на $100 и $300)
    depth_1s['eff_buy_vwap_100'] = calc_effective_vwap(depth_1s, 'ask', 100.0)
    depth_1s['eff_sell_vwap_100'] = calc_effective_vwap(depth_1s, 'bid', 100.0)
    depth_1s['slippage_buy_100_pct'] = (depth_1s['eff_buy_vwap_100'] - best_ask) / best_ask.replace(0, np.nan)
    depth_1s['slippage_sell_100_pct'] = (best_bid - depth_1s['eff_sell_vwap_100']) / best_bid.replace(0, np.nan)

    depth_1s['eff_buy_vwap_300'] = calc_effective_vwap(depth_1s, 'ask', 300.0)
    depth_1s['eff_sell_vwap_300'] = calc_effective_vwap(depth_1s, 'bid', 300.0)
    depth_1s['slippage_buy_300_pct'] = (depth_1s['eff_buy_vwap_300'] - best_ask) / best_ask.replace(0, np.nan)
    depth_1s['slippage_sell_300_pct'] = (best_bid - depth_1s['eff_sell_vwap_300']) / best_bid.replace(0, np.nan)

    # 4. Ресемплинг сделок (1s)
    if not df_t.empty and 'timestamp' in df_t.columns:
        if df_t['timestamp'].dt.tz is None:
            df_t['timestamp'] = df_t['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        else:
            df_t['timestamp'] = df_t['timestamp'].dt.tz_convert('Europe/Moscow')

        df_t['amount'] = df_t['amount'].astype(float)
        side_col = 'side' if 'side' in df_t.columns else ('type' if 'type' in df_t.columns else None)
        if side_col:
            df_t['side_clean'] = df_t[side_col].astype(str).str.upper()
            df_t['buy_vol'] = np.where(df_t['side_clean'].isin(['ASK', 'BUY']), df_t['amount'], 0.0)
            df_t['sell_vol'] = np.where(df_t['side_clean'].isin(['BID', 'SELL']), df_t['amount'], 0.0)
        else:
            df_t['buy_vol'] = 0.0
            df_t['sell_vol'] = 0.0

        df_t = df_t.sort_values('timestamp')
        df_t.set_index('timestamp', inplace=True)

        id_col = 'id' if 'id' in df_t.columns else 'amount'
        trades_1s = df_t.resample('1s').agg({
            'amount': 'sum',
            'buy_vol': 'sum',
            'sell_vol': 'sum',
            id_col: 'count'
        }).rename(columns={'amount': 'trade_vol', id_col: 'trade_count'}).fillna(0)

        # Метрики потока сделок и CVD (Cumulative Volume Delta)
        trades_1s['delta_vol'] = trades_1s['buy_vol'] - trades_1s['sell_vol']
        trades_1s['cvd_1m'] = trades_1s['delta_vol'].rolling(60, min_periods=1).sum()
        trades_1s['cvd_5m'] = trades_1s['delta_vol'].rolling(300, min_periods=1).sum()
        trades_1s['cvd_15m'] = trades_1s['delta_vol'].rolling(900, min_periods=1).sum()
        trades_1s['cvd_total'] = trades_1s['delta_vol'].cumsum()
        vol_1m = trades_1s['trade_vol'].rolling(60, min_periods=1).sum()
        trades_1s['trade_imbalance_1m'] = (trades_1s['cvd_1m'] / vol_1m.replace(0, np.nan)).fillna(0)
    else:
        trades_1s = pd.DataFrame(index=depth_1s.index, data={
            'trade_vol': 0.0,
            'buy_vol': 0.0,
            'sell_vol': 0.0,
            'trade_count': 0,
            'delta_vol': 0.0,
            'cvd_1m': 0.0,
            'cvd_5m': 0.0,
            'cvd_15m': 0.0,
            'cvd_total': 0.0,
            'trade_imbalance_1m': 0.0
        })

    # Объединяем стакан и сделки
    df_1s = pd.merge(depth_1s, trades_1s, left_index=True, right_index=True, how='left')
    trade_metrics = ['trade_vol', 'buy_vol', 'sell_vol', 'trade_count', 'delta_vol', 'cvd_1m', 'cvd_5m', 'cvd_15m', 'cvd_total', 'trade_imbalance_1m']
    df_1s[trade_metrics] = df_1s[trade_metrics].fillna(0)
    df_1s.columns = [f"{prefix}_{c}" for c in df_1s.columns]
    return df_1s

def build_features_for_coin(coin_name, macro_df=None):
    coin = coin_name.lower()
    print(f"\n{Fore.CYAN}--- Генерация признаков для {coin.upper()} ---{Style.RESET_ALL}")
    cdata = load_coin(coin)

    parts = []

    # 1. SafeTrade L2
    if 'safetrade_depth' in cdata and not cdata.safetrade_depth.empty:
        df_st = process_exchange_l2(cdata.safetrade_depth, cdata.get('safetrade_trades'), 'ST')
        parts.append(('ST', df_st))

    # 2. BigONE L2
    if 'bigone_depth' in cdata and not cdata.bigone_depth.empty:
        df_bo = process_exchange_l2(cdata.bigone_depth, cdata.get('bigone_trades'), 'BO')
        parts.append(('BO', df_bo))

    # 3. Rhea Finance (Quantus)
    if 'rhea_price' in cdata and not cdata.rhea_price.empty:
        df_rh = cdata.rhea_price.copy()
        if df_rh['timestamp'].dt.tz is None:
            df_rh['timestamp'] = df_rh['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        else:
            df_rh['timestamp'] = df_rh['timestamp'].dt.tz_convert('Europe/Moscow')
        df_rh = df_rh.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
        rhea_cols = [c for c in ['price_usd', 'price_native', 'liquidity_usd', 'volume_5m'] if c in df_rh.columns]
        rhea_1s = df_rh[rhea_cols].apply(pd.to_numeric, errors='coerce').resample('1s').ffill()
        rhea_1s.columns = [f"RHEA_{c}" for c in rhea_1s.columns]
        parts.append(('RHEA', rhea_1s))

    if not parts:
        print(f"{Fore.RED}❌ Нет рыночных данных для монеты {coin}!{Style.RESET_ALL}")
        return None

    # Объединение по времени
    main_prefix, merged = parts[0]
    for pfx, part_df in parts[1:]:
        print(f"  Слияние {main_prefix} и {pfx}...")
        merged = pd.merge(merged, part_df, left_index=True, right_index=True, how='inner')

    # Арбитражные фичи (если есть и ST, и BO)
    if 'BO_bid_1_price' in merged.columns and 'ST_ask_1_price' in merged.columns:
        print("  Расчет межбиржевых спредов SafeTrade vs BigONE...")
        # 1. BBO арбитраж (по 1 лоту)
        merged['arb_st_buy_bo_sell'] = merged['BO_bid_1_price'].astype(float) - merged['ST_ask_1_price'].astype(float)
        merged['arb_bo_buy_st_sell'] = merged['ST_bid_1_price'].astype(float) - merged['BO_ask_1_price'].astype(float)
        merged['mid_price_diff'] = merged['BO_mid_price'] - merged['ST_mid_price']

        # 2. Глубокий арбитраж с учетом проскальзывания на $100 и $300 (Effective Arbitrage)
        if 'ST_eff_sell_vwap_100' in merged.columns and 'BO_eff_buy_vwap_100' in merged.columns:
            merged['arb_eff_100_bo_buy_st_sell'] = merged['ST_eff_sell_vwap_100'] - merged['BO_eff_buy_vwap_100']
            merged['arb_eff_100_pct'] = merged['arb_eff_100_bo_buy_st_sell'] / merged['BO_eff_buy_vwap_100'].replace(0, np.nan)

            merged['arb_eff_300_bo_buy_st_sell'] = merged['ST_eff_sell_vwap_300'] - merged['BO_eff_buy_vwap_300']
            merged['arb_eff_300_pct'] = merged['arb_eff_300_bo_buy_st_sell'] / merged['BO_eff_buy_vwap_300'].replace(0, np.nan)

            # 3. Чистый прирост в монетах PRL на сделку в $100 с учетом комиссий
            # При покупке на BigONE (0.2% fee) и продаже на SafeTrade (0.1% fee)
            q_bought = (100.0 * 0.998) / merged['BO_eff_buy_vwap_100'].replace(0, np.nan)
            q_sold = 100.0 / (merged['ST_eff_sell_vwap_100'].replace(0, np.nan) * 0.999)
            merged['q_prl_accum_gain_100usd'] = (q_bought - q_sold).fillna(0)

    # Арбитражные фичи для Rhea (DEX vs CEX)
    if 'RHEA_price_usd' in merged.columns:
        if 'ST_mid_price' in merged.columns:
            merged['arb_rhea_diff_st'] = merged['RHEA_price_usd'] - merged['ST_mid_price']
        if 'BO_mid_price' in merged.columns:
            merged['arb_rhea_diff_bo'] = merged['RHEA_price_usd'] - merged['BO_mid_price']

    # 4. Ончейн фичи
    if 'onchain' in cdata and not cdata.onchain.empty:
        print("  Добавление ончейн-метрик...")
        df_on = cdata.onchain.copy()
        if df_on['timestamp'].dt.tz is None:
            df_on['timestamp'] = df_on['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        else:
            df_on['timestamp'] = df_on['timestamp'].dt.tz_convert('Europe/Moscow')
        df_on = df_on.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
        
        # Индикаторы майнинг-пулов
        if 'miner_pool' in df_on.columns:
            df_on['pool_kryptex'] = (df_on['miner_pool'].astype(str).str.lower() == 'kryptex').astype(int)
            df_on['pool_pearlhash'] = (df_on['miner_pool'].astype(str).str.lower() == 'pearlhash').astype(int)
            df_on['pool_pearlfortune'] = (df_on['miner_pool'].astype(str).str.lower().str.contains('fortune')).astype(int)
            df_on['pool_luckypool'] = (df_on['miner_pool'].astype(str).str.lower().str.contains('lucky')).astype(int)
        
        onchain_cols = [c for c in df_on.columns if c not in ['timestamp', 'miner_pool', 'miner_label', 'miner_address']]
        on_1s = df_on[onchain_cols].apply(pd.to_numeric, errors='coerce').resample('1s').ffill()
        on_1s.columns = [f"onchain_{c}" for c in on_1s.columns]
        
        # Динамика сложности и объемы китов
        if 'onchain_difficulty' in on_1s.columns:
            on_1s['onchain_diff_pct_change_1h'] = on_1s['onchain_difficulty'].pct_change(3600).fillna(0)
        if 'onchain_whale_volume_5k' in on_1s.columns:
            on_1s['onchain_whale_vol_sum_1h'] = on_1s['onchain_whale_volume_5k'].rolling(3600, min_periods=1).sum().fillna(0)
            on_1s['onchain_whale_tx_sum_1h'] = on_1s.get('onchain_whale_tx_count_5k', pd.Series(0, index=on_1s.index)).rolling(3600, min_periods=1).sum().fillna(0)
            
        merged = pd.merge(merged, on_1s, left_index=True, right_index=True, how='left')
        merged[[c for c in on_1s.columns]] = merged[[c for c in on_1s.columns]].ffill().bfill().fillna(0)

    # 5. Макро-поводыри Binance
    if macro_df is not None and not macro_df.empty:
        print("  Добавление макро-поводырей (Binance BTC & NEAR)...")
        df_m = macro_df.copy()
        if df_m['timestamp'].dt.tz is None:
            df_m['timestamp'] = df_m['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        else:
            df_m['timestamp'] = df_m['timestamp'].dt.tz_convert('Europe/Moscow')
        df_m = df_m.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
        
        m_cols = ['btc_mid', 'btc_last', 'near_mid', 'near_last']
        avail_m_cols = [c for c in m_cols if c in df_m.columns]
        m_1s = df_m[avail_m_cols].astype(float).resample('1s').ffill()
        m_1s.columns = [f"macro_{c}" for c in m_1s.columns]
        
        # Доходность за 1с
        if 'macro_btc_mid' in m_1s.columns:
            m_1s['macro_btc_ret_1s'] = m_1s['macro_btc_mid'].pct_change().fillna(0)
        if 'macro_near_mid' in m_1s.columns:
            m_1s['macro_near_ret_1s'] = m_1s['macro_near_mid'].pct_change().fillna(0)

        merged = pd.merge(merged, m_1s, left_index=True, right_index=True, how='left')
        for c in m_1s.columns:
            merged[c] = merged[c].ffill().bfill().fillna(0)

    # Очистка: удаляем только строки, где нет mid_price ключевой биржи, а пустые глубокие уровни стакана заполняем нулями
    key_cols = [c for c in merged.columns if 'mid_price' in c or 'bid_1_price' in c]
    merged = merged.dropna(subset=key_cols)
    merged = merged.fillna(0)

    # Сохранение в data/features/{coin}_1s_features.parquet
    out_dir = os.path.join(BASE_DIR, 'data', 'features')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{coin}_1s_features.parquet")
    merged.to_parquet(out_file)
    size_mb = os.path.getsize(out_file) / (1024 * 1024)

    print(f"{Fore.GREEN}✅ [{coin.upper()}] Датасет признаков сохранен: {os.path.basename(out_file)} ({size_mb:.2f} MB)")
    print(f"   Строк: {len(merged):,} | Признаков: {len(merged.columns)} | Период: {merged.index.min()} -> {merged.index.max()}{Style.RESET_ALL}")

    # Для PRL сохраняем также merged_1s_base.parquet для обратной совместимости
    if coin == 'prl':
        legacy_out = os.path.join(out_dir, "merged_1s_base.parquet")
        merged.to_parquet(legacy_out)
        print(f"   (Также сохранен legacy файл: {os.path.basename(legacy_out)})")

    return merged

def main():
    parser = argparse.ArgumentParser(description="Синхронизация и построение 1-секундного базового датасета (PRL, QUANTUS, TSC).")
    parser.add_argument("--coin", type=str, default="all", choices=["prl", "quantus", "tsc", "all"], help="Монета для генерации (по умолчанию all).")
    parser.add_argument("--sync", action="store_true", help="Сначала выполнить синхронизацию с Google Drive и VPS.")
    args = parser.parse_args()

    if args.sync:
        print(f"{Fore.CYAN}🔄 Выполняется синхронизация данных...{Style.RESET_ALL}")
        sync_data()

    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}⚙️ ПОСТРОЕНИЕ 1-СЕКУНДНЫХ БАЗОВЫХ ДАТАСЕТОВ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")

    macro_df = load_macro()
    coins = ["prl", "quantus", "tsc"] if args.coin == "all" else [args.coin]

    for c in coins:
        build_features_for_coin(c, macro_df=macro_df)

    print(f"\n{Fore.GREEN}🎉 Построение 1-секундных датасетов успешно завершено!{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
