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
from _tools.system_resources import SAFE_WORKERS

init(autoreset=True)

def calc_effective_vwap(df, side='ask', target_usd=100.0, max_levels=50):
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
    last_valid_price = prices[0].copy()

    for p, q in zip(prices, qtys):
        active = filled_usd < target_usd
        if not np.any(active):
            break
        valid_mask = active & np.isfinite(p) & (p > 0) & np.isfinite(q) & (q > 0)
        if not np.any(valid_mask):
            continue
        last_valid_price[valid_mask] = p[valid_mask]
        p_act = p[valid_mask]
        q_act = q[valid_mask]
        level_usd = p_act * q_act
        needed_usd = target_usd - filled_usd[valid_mask]
        take_usd = np.minimum(needed_usd, level_usd)
        with np.errstate(divide='ignore', invalid='ignore'):
            take_qty = np.where(p_act > 0, take_usd / p_act, 0.0)

        filled_usd[valid_mask] += take_usd
        filled_qty[valid_mask] += take_qty

    rem_usd = np.maximum(0.0, target_usd - filled_usd)
    penalty_mult = 1.005 if side == 'ask' else 0.995
    worst_p = last_valid_price * penalty_mult
    with np.errstate(divide='ignore', invalid='ignore'):
        rem_qty = np.where(worst_p > 0, rem_usd / worst_p, 0.0)
        total_qty = filled_qty + rem_qty
        vwap = np.where(total_qty > 0, target_usd / total_qty, np.nan)
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

    # 2. Ресемплинг стакана (1s, ffill) и время с последнего реального тика
    depth_cols = [c for c in df_d.columns if 'bid' in c or 'ask' in c]
    raw_tick_ts = pd.Series(df_d.index, index=df_d.index).resample('1s').ffill()
    depth_1s = df_d[depth_cols].resample('1s').ffill()
    depth_1s['sec_since_tick'] = (depth_1s.index - raw_tick_ts).dt.total_seconds().fillna(0.0)

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

    # Глубокий стакан (уровни 11..50 и топ-50)
    bids_vol_11_50 = sum(depth_1s[f'bid_{i}_qty'].astype(float).fillna(0) for i in range(11, 51) if f'bid_{i}_qty' in depth_1s.columns)
    asks_vol_11_50 = sum(depth_1s[f'ask_{i}_qty'].astype(float).fillna(0) for i in range(11, 51) if f'ask_{i}_qty' in depth_1s.columns)
    bids_vol_top50 = bids_vol_top10.fillna(0) + bids_vol_11_50
    asks_vol_top50 = asks_vol_top10.fillna(0) + asks_vol_11_50
    ob_imbalance_11_50 = (bids_vol_11_50 - asks_vol_11_50) / (bids_vol_11_50 + asks_vol_11_50).replace(0, np.nan)
    ob_imbalance_top50 = (bids_vol_top50 - asks_vol_top50) / (bids_vol_top50 + asks_vol_top50).replace(0, np.nan)

    # Поиск крупнейшей лимитной стены (Whale Wall) в стакане
    bid_p_cols = [f'bid_{i}_price' for i in range(1, 51) if f'bid_{i}_price' in depth_1s.columns]
    bid_q_cols = [f'bid_{i}_qty' for i in range(1, 51) if f'bid_{i}_qty' in depth_1s.columns]
    ask_p_cols = [f'ask_{i}_price' for i in range(1, 51) if f'ask_{i}_price' in depth_1s.columns]
    ask_q_cols = [f'ask_{i}_qty' for i in range(1, 51) if f'ask_{i}_qty' in depth_1s.columns]

    if bid_q_cols and ask_q_cols:
        bq_mat = np.nan_to_num(depth_1s[bid_q_cols].to_numpy(dtype=float), nan=0.0)
        bp_mat = np.nan_to_num(depth_1s[bid_p_cols].to_numpy(dtype=float), nan=0.0)
        aq_mat = np.nan_to_num(depth_1s[ask_q_cols].to_numpy(dtype=float), nan=0.0)
        ap_mat = np.nan_to_num(depth_1s[ask_p_cols].to_numpy(dtype=float), nan=0.0)
        b_idx = np.argmax(bq_mat, axis=1)
        a_idx = np.argmax(aq_mat, axis=1)
        rows_idx = np.arange(len(depth_1s))
        calc_bid_wall_q = bq_mat[rows_idx, b_idx]
        calc_bid_wall_p = bp_mat[rows_idx, b_idx]
        calc_ask_wall_q = aq_mat[rows_idx, a_idx]
        calc_ask_wall_p = ap_mat[rows_idx, a_idx]
    else:
        calc_bid_wall_q = np.zeros(len(depth_1s))
        calc_bid_wall_p = best_bid.to_numpy()
        calc_ask_wall_q = np.zeros(len(depth_1s))
        calc_ask_wall_p = best_ask.to_numpy()

    if 'max_bid_wall_qty' in depth_1s.columns:
        depth_1s['max_bid_wall_qty'] = pd.to_numeric(depth_1s['max_bid_wall_qty'], errors='coerce').fillna(pd.Series(calc_bid_wall_q, index=depth_1s.index))
        depth_1s['max_bid_wall_price'] = pd.to_numeric(depth_1s['max_bid_wall_price'], errors='coerce').fillna(pd.Series(calc_bid_wall_p, index=depth_1s.index))
        depth_1s['max_ask_wall_qty'] = pd.to_numeric(depth_1s['max_ask_wall_qty'], errors='coerce').fillna(pd.Series(calc_ask_wall_q, index=depth_1s.index))
        depth_1s['max_ask_wall_price'] = pd.to_numeric(depth_1s['max_ask_wall_price'], errors='coerce').fillna(pd.Series(calc_ask_wall_p, index=depth_1s.index))
    else:
        depth_1s['max_bid_wall_qty'] = calc_bid_wall_q
        depth_1s['max_bid_wall_price'] = calc_bid_wall_p
        depth_1s['max_ask_wall_qty'] = calc_ask_wall_q
        depth_1s['max_ask_wall_price'] = calc_ask_wall_p

    if 'deep_bid_vol' in depth_1s.columns:
        depth_1s['deep_bid_vol'] = pd.to_numeric(depth_1s['deep_bid_vol'], errors='coerce').fillna(0.0)
        depth_1s['deep_ask_vol'] = pd.to_numeric(depth_1s['deep_ask_vol'], errors='coerce').fillna(0.0)
    else:
        depth_1s['deep_bid_vol'] = 0.0
        depth_1s['deep_ask_vol'] = 0.0

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
    depth_1s['bids_vol_11_50'] = bids_vol_11_50
    depth_1s['asks_vol_11_50'] = asks_vol_11_50
    depth_1s['imbalance_11_50'] = ob_imbalance_11_50.fillna(0)
    depth_1s['imbalance_top50'] = ob_imbalance_top50.fillna(0)
    depth_1s['bid_wall_dist_bps'] = ((mid_price - depth_1s['max_bid_wall_price']) / mid_price.replace(0, np.nan) * 1e4).clip(0, 5000).fillna(0)
    depth_1s['ask_wall_dist_bps'] = ((depth_1s['max_ask_wall_price'] - mid_price) / mid_price.replace(0, np.nan) * 1e4).clip(0, 5000).fillna(0)
    wall_tot = (depth_1s['max_bid_wall_qty'] + depth_1s['max_ask_wall_qty']).replace(0, np.nan)
    depth_1s['wall_vol_imbalance'] = ((depth_1s['max_bid_wall_qty'] - depth_1s['max_ask_wall_qty']) / wall_tot).fillna(0)

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
            df_t['side_clean'] = df_t[side_col].astype(str).str.strip().str.upper()
            if prefix == 'BO':
                buy_tokens = ['BID', 'BUY']
                sell_tokens = ['ASK', 'SELL']
            else:
                buy_tokens = ['BUY', 'ASK']
                sell_tokens = ['SELL', 'BID']
            df_t['buy_vol'] = np.where(df_t['side_clean'].isin(buy_tokens), df_t['amount'], 0.0)
            df_t['sell_vol'] = np.where(df_t['side_clean'].isin(sell_tokens), df_t['amount'], 0.0)
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

        # -------------------------------------------------------------
        # 🛡️ ПРИНЦИП СТРОГОЙ ПРИЧИННОСТИ (STRICT CAUSALITY / ZERO-LOOKAHEAD):
        # -------------------------------------------------------------
        # Ресемплинг '1s' агрегирует сделки за интервал [t.000, t.999].
        # Стакан L2 (depth_1s) зафиксирован на НАЧАЛО секунды t.000.
        # Чтобы в момент t не знать сделки, которые произойдут в течение секунды t,
        # поток сделок ОБЯЗАН быть сдвинут на 1 секунду в прошлое: trades_1s.shift(1).
        # В строке t будут находиться только сделки, завершившиеся в [t-1, t).
        trades_1s = trades_1s.shift(1).fillna(0.0)
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
        rhea_cols = [c for c in [
            'price_usd', 'price_native', 'liquidity_usd', 'base_reserve', 'quote_reserve',
            'volume_5m', 'buys_5m', 'sells_5m', 'volume_1h', 'volume_24h',
            'buys_1h', 'sells_1h', 'buys_24h', 'sells_24h',
            'price_change_5m', 'price_change_1h', 'price_change_24h'
        ] if c in df_rh.columns]
        rhea_1s = df_rh[rhea_cols].apply(pd.to_numeric, errors='coerce').resample('1s').ffill()
        if 'buys_5m' in rhea_1s.columns and 'sells_5m' in rhea_1s.columns:
            tot_tx5 = (rhea_1s['buys_5m'] + rhea_1s['sells_5m']).replace(0, np.nan)
            rhea_1s['tx_imbalance_5m'] = ((rhea_1s['buys_5m'] - rhea_1s['sells_5m']) / tot_tx5).fillna(0)
        if 'base_reserve' in rhea_1s.columns:
            rhea_1s['reserve_flow_60s_bps'] = (-rhea_1s['base_reserve'].pct_change(60) * 1e4).fillna(0)
            rhea_1s['reserve_flow_300s_bps'] = (-rhea_1s['base_reserve'].pct_change(300) * 1e4).fillna(0)
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

        if 'miner_pool' in df_on.columns:
            df_on['pool_kryptex'] = (df_on['miner_pool'].astype(str).str.lower() == 'kryptex').astype(int)
            df_on['pool_pearlhash'] = (df_on['miner_pool'].astype(str).str.lower() == 'pearlhash').astype(int)
            df_on['pool_pearlfortune'] = (df_on['miner_pool'].astype(str).str.lower().str.contains('fortune')).astype(int)
            df_on['pool_luckypool'] = (df_on['miner_pool'].astype(str).str.lower().str.contains('lucky')).astype(int)

        # По-блочные 1-часовые суммы китов (до 1s ffill, чтобы 1 блок не умножался в 3600 раз)
        if 'block_height' in df_on.columns:
            blocks = df_on[~df_on['block_height'].duplicated(keep='first')].copy()
            for w_col, out_col in [
                ('transfer_volume_prl', 'transfer_vol_sum_1h'),
                ('whale_volume_5k', 'whale_vol_sum_1h'),
                ('whale_tx_count_5k', 'whale_tx_sum_1h'),
                ('whale_volume_25k', 'whale_vol_25k_sum_1h'),
            ]:
                if w_col in blocks.columns:
                    b_sum = pd.to_numeric(blocks[w_col], errors='coerce').fillna(0).rolling('1h').sum()
                    df_on[out_col] = df_on['block_height'].map(dict(zip(blocks['block_height'], b_sum)))

        drop_on = {
            'timestamp', 'date', 'miner_pool', 'miner_label', 'miner_address', 'block_hash', 'status',
            'circulating_supply', 'total_supply', 'active_validators', 'era_reward',
            'daily_txs', 'daily_Active_Accounts', 'block_reward', 'accounts_count'
        }
        onchain_cols = [c for c in df_on.columns if c not in drop_on]
        on_1s = df_on[onchain_cols].apply(pd.to_numeric, errors='coerce').resample('1s').ffill()
        on_1s.columns = [f"onchain_{c}" for c in on_1s.columns]

        if 'onchain_difficulty' in on_1s.columns:
            on_1s['onchain_diff_pct_change_1h'] = on_1s['onchain_difficulty'].pct_change(3600).fillna(0)
        if 'onchain_mempool_size' in on_1s.columns:
            on_1s['onchain_mempool_delta_5m'] = on_1s['onchain_mempool_size'].diff(300).fillna(0)

        merged = pd.merge(merged, on_1s, left_index=True, right_index=True, how='left')
        merged[[c for c in on_1s.columns]] = merged[[c for c in on_1s.columns]].ffill().fillna(0)

    # 5. Макро-поводыри Binance
    if macro_df is not None and not macro_df.empty:
        print("  Добавление макро-поводырей (Binance BTC & NEAR)...")
        df_m = macro_df.copy()
        if df_m['timestamp'].dt.tz is None:
            df_m['timestamp'] = df_m['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        else:
            df_m['timestamp'] = df_m['timestamp'].dt.tz_convert('Europe/Moscow')
        df_m = df_m.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')

        m_cols = [
            'btc_bid', 'btc_ask', 'btc_mid', 'btc_last',
            'near_bid', 'near_ask', 'near_mid', 'near_last',
            'btc_bid_qty', 'btc_ask_qty', 'btc_buy_vol', 'btc_sell_vol',
            'near_bid_qty', 'near_ask_qty', 'near_buy_vol', 'near_sell_vol'
        ]
        avail_m_cols = [c for c in m_cols if c in df_m.columns]
        m_1s = df_m[avail_m_cols].apply(pd.to_numeric, errors='coerce').resample('1s').ffill()
        m_1s.columns = [f"macro_{c}" for c in m_1s.columns]

        if 'macro_btc_mid' in m_1s.columns:
            m_1s['macro_btc_ret_1s'] = m_1s['macro_btc_mid'].pct_change().fillna(0)
        if 'macro_near_mid' in m_1s.columns:
            m_1s['macro_near_ret_1s'] = m_1s['macro_near_mid'].pct_change().fillna(0)
        if 'macro_btc_bid_qty' in m_1s.columns and 'macro_btc_ask_qty' in m_1s.columns:
            bt_tot = (m_1s['macro_btc_bid_qty'] + m_1s['macro_btc_ask_qty']).replace(0, np.nan)
            m_1s['macro_btc_bbo_imb'] = ((m_1s['macro_btc_bid_qty'] - m_1s['macro_btc_ask_qty']) / bt_tot).fillna(0)
        if 'macro_near_bid_qty' in m_1s.columns and 'macro_near_ask_qty' in m_1s.columns:
            nr_tot = (m_1s['macro_near_bid_qty'] + m_1s['macro_near_ask_qty']).replace(0, np.nan)
            m_1s['macro_near_bbo_imb'] = ((m_1s['macro_near_bid_qty'] - m_1s['macro_near_ask_qty']) / nr_tot).fillna(0)

        # Сдвиг макро-объемов сделок на 1с для строгой причинности
        for mc in ['macro_btc_buy_vol', 'macro_btc_sell_vol', 'macro_near_buy_vol', 'macro_near_sell_vol']:
            if mc in m_1s.columns:
                m_1s[mc] = m_1s[mc].shift(1).fillna(0.0)

        merged = pd.merge(merged, m_1s, left_index=True, right_index=True, how='left')
        for c in m_1s.columns:
            merged[c] = merged[c].ffill().fillna(0)

    # Очистка: удаляем строки без ключевых BBO/mid_price и приводим сетку к строгому шагу 1 секунда без дыр
    key_cols = [c for c in merged.columns if 'mid_price' in c or 'bid_1_price' in c or 'ask_1_price' in c]
    merged = merged.dropna(subset=key_cols)
    orig_idx = merged.index
    merged = merged.resample('1s').ffill()
    gap_mask = ~merged.index.isin(orig_idx)
    if gap_mask.any():
        flow_cols = [c for c in merged.columns if any(k in c for k in ['trade_vol', 'buy_vol', 'sell_vol', 'trade_count', 'delta_vol'])]
        merged.loc[gap_mask, flow_cols] = 0.0

    # Флаг технического разрыва связи (когда одновременно все биржи молчат > 180 сек)
    st_silent = merged['ST_sec_since_tick'] > 180.0 if 'ST_sec_since_tick' in merged.columns else pd.Series(False, index=merged.index)
    bo_silent = merged['BO_sec_since_tick'] > 180.0 if 'BO_sec_since_tick' in merged.columns else pd.Series(True, index=merged.index)
    merged['is_data_gap'] = (st_silent & bo_silent).astype(np.int8)
    if 'ST_sec_since_tick' in merged.columns:
        merged['is_st_gap_60s'] = (merged['ST_sec_since_tick'] > 60.0).astype(np.int8)
    merged = merged.fillna(0)

    # 6. Многочастотный стационарный банк фильтров W = [5, 20, 60, 300]с и полосовые разности (Band-Pass)
    print("  Расчет стационарного банка фильтров W=[5, 20, 60, 300]с (OFI, Lead-Lag, Band-Pass)...")
    from _tools.microstructure_features import add_exchange_filter_bank, add_cross_exchange_and_macro_filters
    if 'ST_mid_price' in merged.columns:
        merged = add_exchange_filter_bank(merged, 'ST', coin=coin)
    if 'BO_mid_price' in merged.columns:
        merged = add_exchange_filter_bank(merged, 'BO', coin=coin)
    merged = add_cross_exchange_and_macro_filters(merged, coin=coin)
    merged = merged.fillna(0)

    # Оптимизация RAM: конвертируем производные признаки в float32, сохраняя float64 для опорных цен
    keep_f64 = {
        'ST_mid_price', 'ST_micro_price', 'ST_spread', 'ST_bid_1_price', 'ST_ask_1_price',
        'BO_mid_price', 'BO_micro_price', 'BO_spread', 'BO_bid_1_price', 'BO_ask_1_price',
        'RHEA_price_usd', 'RHEA_price_native', 'macro_btc_mid', 'macro_near_mid',
        'ONCHAIN_block_height'
    }
    f64_to_f32 = [
        c for c in merged.columns
        if merged[c].dtype == np.float64 and c not in keep_f64
    ]
    if f64_to_f32:
        merged[f64_to_f32] = merged[f64_to_f32].astype(np.float32)

    # Сохранение в data/features/{coin}_1s_features.parquet (zstd компрессия)
    out_dir = os.path.join(BASE_DIR, 'data', 'features')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{coin}_1s_features.parquet")
    merged.to_parquet(out_file, compression='zstd')
    size_mb = os.path.getsize(out_file) / (1024 * 1024)

    print(f"{Fore.GREEN}✅ [{coin.upper()}] Датасет признаков сохранен: {os.path.basename(out_file)} ({size_mb:.2f} MB)")
    print(f"   Строк: {len(merged):,} | Признаков: {len(merged.columns)} | Период: {merged.index.min()} -> {merged.index.max()}{Style.RESET_ALL}")

    # Для PRL сохраняем также merged_1s_base.parquet для обратной совместимости (через symlink)
    if coin == 'prl':
        legacy_out = os.path.join(out_dir, "merged_1s_base.parquet")
        try:
            if os.path.islink(legacy_out) or os.path.exists(legacy_out):
                os.remove(legacy_out)
            os.symlink(os.path.basename(out_file), legacy_out)
            print(f"   (Создана быстрая ссылка: {os.path.basename(legacy_out)} -> {os.path.basename(out_file)})")
        except Exception:
            merged.to_parquet(legacy_out, compression='zstd')
            print(f"   (Также сохранен legacy файл: {os.path.basename(legacy_out)})")

    return merged

def main():
    parser = argparse.ArgumentParser(description="Синхронизация и построение 1-секундного базового датасета (PRL, QUANTUS).")
    parser.add_argument("--coin", type=str, default="all", choices=["prl", "quantus", "all"], help="Монета для генерации (по умолчанию all).")
    parser.add_argument("--sync", action="store_true", help="Сначала выполнить синхронизацию с Google Drive и VPS.")
    args = parser.parse_args()

    if args.sync:
        print(f"{Fore.CYAN}🔄 Выполняется синхронизация данных...{Style.RESET_ALL}")
        sync_data()

    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}⚙️ ПОСТРОЕНИЕ 1-СЕКУНДНЫХ БАЗОВЫХ ДАТАСЕТОВ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")

    macro_df = load_macro()
    coins = ["prl", "quantus"] if args.coin == "all" else [args.coin]

    from concurrent.futures import ProcessPoolExecutor
    workers = min(len(coins), SAFE_WORKERS)
    if workers > 1:
        print(f"⚡ Мультипроцессорная сборка датасетов: {workers} параллельных процессов (лимит CPU <= 80%, {os.cpu_count() - SAFE_WORKERS} ядер свободно для ОС)")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(build_features_for_coin, c, macro_df) for c in coins]
            for f in futures:
                f.result()
    else:
        for c in coins:
            build_features_for_coin(c, macro_df=macro_df)

    print(f"\n{Fore.GREEN}🎉 Построение 1-секундных датасетов успешно завершено!{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
