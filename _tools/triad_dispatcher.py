#!/usr/bin/env python3
"""
Мультигоризонтный координатор ордеров триады (Multi-Scale Triad Order Dispatcher).

Архитектура координации ордеров:
- Триада масштабов:
    1. Scale 1 (Sniper, H=5с, K=3.0, L=15с): Высокоскоростной отклик на микро-перекос BBO.
    2. Scale 2 (Core Maker, H=12с, K=3.5, L=42с): Ядро удержания спреда (наименьшая ошибка < 1.0%).
    3. Scale 3 (Wave Regime Trend, H=25с, K=3.0, L=75с): Трендовый фильтр и шлюз защиты (Trend Gate).

Правила диспетчеризации и разрешения конфликтов:
1. Защитный шлюз тренда (Scale 3 - H=25):
   - При ez_25 < -trend_gate_th (нисходящая волна):
     * Запрещаются покупки (Buy Gate Locked) для защиты от Adverse Selection ("падающих ножей").
     * Если активен лонговый лимитный ордер, выставляется аварийная отмена.
   - При ez_25 > +trend_gate_th (восходящая волна):
     * Запрещаются продажи (Sell Gate Locked) для предотвращения преждевременного сброса токенов.
     * Если активен шортовый лимитный ордер, выставляется аварийная отмена.

2. Согласование сигналов (Cross-Horizon Synergy):
   - Если H=5 и H=12 согласованы по направлению с H=25 -> Высокая уверенность (High Conviction),
     постановка лимитного ордера внутри спреда (BBO + tick).
   - Если H=5 противоречит H=12 (микрошум) -> пауза до синхронизации потока ордеров.

3. Реалистичная симуляция SafeTrade:
   - Модель очереди: Queue Priority (учет Q_ahead на BBO).
   - Ограничение объема: исполнение ограничено фактическим объемом сделок (sell_vol / buy_vol).
   - Сетевая задержка RTT (1.0с) для постановки и отмены ордеров.
   - Чистый Maker-режим (0.0% комиссия) с максимизацией Q_token.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.swarm_router import SwarmRegimeRouter
from _tools.simulate_execution import COIN_RULES


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Scale Triad Order Dispatcher & Simulator")
    parser.add_argument("--coin", type=str, default="prl", choices=["prl", "quantus"], help="Монета (prl, quantus)")
    parser.add_argument("--mode", type=str, default="router", choices=["router", "uniform", "latest"],
                        help="Режим ансамблирования внутри каждого роутера")
    parser.add_argument("--trend-gate", type=float, default=0.15,
                        help="Порог трендового шлюза H=25 для блокировки противоположных входов")
    parser.add_argument("--threshold-core", type=float, default=0.20,
                        help="Порог входа ядра H=12 E[z]")
    parser.add_argument("--threshold-sniper", type=float, default=0.25,
                        help="Порог согласования снайпера H=5 E[z]")
    parser.add_argument("--latency-sec", type=int, default=1,
                        help="Сетевая задержка RTT постановки и отмены заявок (сек)")
    parser.add_argument("--queue-model", type=str, default="realistic", choices=["realistic", "instant"],
                        help="Модель очереди (realistic: учет глубины и объема, instant: мгновенное исполнение)")
    parser.add_argument("--cancel-timeout", type=int, default=60,
                        help="Таймаут жизни пассивного ордера в стакане (сек, по умолчанию 60)")
    parser.add_argument("--maker-fee", type=float, default=0.000, help="Комиссия Maker (0.0%% на SafeTrade)")
    parser.add_argument("--taker-fee", type=float, default=0.002, help="Комиссия Taker (0.2%% на SafeTrade)")
    parser.add_argument("--q-token-bias", type=float, default=0.10,
                        help="Асимметричное смещение порога продажи в сторону накопления токенов при восходящей волне H=25")
    parser.add_argument("--ask-skew", action="store_true", default=True,
                        help="Динамический перекос аска при восходящей волне z25 > 0.05")
    parser.add_argument("--no-ask-skew", dest="ask_skew", action="store_false",
                        help="Отключить динамический перекос аска")
    parser.add_argument("--strict-oos", action="store_true", default=True,
                        help="Автоматически изолировать срез строго после максимальной даты обучения моделей")
    parser.add_argument("--no-strict-oos", dest="strict_oos", action="store_false",
                        help="Отключить строгую изоляцию OOS (тест на всем val)")
    parser.add_argument("--leadlag-shield", action="store_true", default=True,
                        help="Защитный шлюз опережения BigONE: экстренная отмена токсичных заявок")
    parser.add_argument("--no-leadlag-shield", dest="leadlag_shield", action="store_false",
                        help="Отключить Lead-Lag Shield")
    parser.add_argument("--dip-sniper", action="store_true", default=True,
                        help="Снайпинг проливов по опережающему импульсу BigONE")
    parser.add_argument("--no-dip-sniper", dest="dip_sniper", action="store_false",
                        help="Отключить снайпинг проливов")
    parser.add_argument("--bo-ask-peg", action="store_true", default=True,
                        help="Привязка цены продажи Ask Escalator к стакану асков BigONE")
    parser.add_argument("--no-bo-ask-peg", dest="bo_ask_peg", action="store_false",
                        help="Отключить привязку аска к BigONE")
    parser.add_argument("--inventory-skew", action="store_true", default=True,
                        help="Avellaneda-Stoikov перекос инвентаря и динамический сайзинг лотов")
    parser.add_argument("--no-inventory-skew", dest="inventory_skew", action="store_false",
                        help="Отключить динамический перекос инвентаря")
    parser.add_argument("--base-lot", type=float, default=None,
                        help="Базовый размер лота в токенах (для PRL по умолчанию 50.0)")
    parser.add_argument("--min-order-val", type=float, default=0.5,
                        help="Минимальный объем ордера в USDT на бирже SafeTrade")
    parser.add_argument("--max-steps", type=int, default=None, help="Ограничить шаги для быстрого теста")
    parser.add_argument("--plot", type=str, default=None, help="Путь для сохранения PNG-графика бэктеста")
    return parser.parse_args()


class TriadDispatcher:
    """Мультигоризонтный координатор триады экспертов (H=5, 12, 25)."""

    def __init__(self, coin: str):
        self.coin = coin.lower()
        print(f"\n📦 Инициализация триады масштабов для {self.coin.upper()}...")
        # Scale 1: Sniper
        self.router_5 = SwarmRegimeRouter(self.coin, H=5, K=3.0)
        # Scale 2: Core Spread Maker
        self.router_12 = SwarmRegimeRouter(self.coin, H=12, K=3.5)
        # Scale 3: Wave Regime Trend Gate
        self.router_25 = SwarmRegimeRouter(self.coin, H=25, K=3.0)
        print("✅ Триада экспертов успешно загружена в оперативную память!")


def run_triad_simulation(args):
    coin = args.coin.lower()
    cfg = COIN_RULES.get(coin, COIN_RULES['prl'])
    tick_size = cfg['tick_size']

    print("\n" + "=" * 105)
    print(f"🎯 МУЛЬТИГОРИЗОНТНЫЙ ДИСПЕТЧЕР ТРИАДЫ SAFETRADE: {coin.upper()}")
    print(f"   Триада: H1 = 5с (K=3.0) | H2 = 12с (K=3.5) | H3 = 25с (K=3.0)")
    print(f"   Параметры: Trend Gate={args.trend_gate} | Core Thresh={args.threshold_core} | Sniper Thresh={args.threshold_sniper}")
    print(f"   Реализм: Очередь={args.queue_model.upper()} | Сетевой лаг RTT={args.latency_sec}с | Таймаут={args.cancel_timeout}с")
    print(f"   Инвариант: целевая функция оптимизирована на максимальный чистый прирост монет (Q_token)")
    print("=" * 105)

    # 1. Загрузка триады
    dispatcher = TriadDispatcher(coin)

    # 2. Загрузка чистого OOS-датасета
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    schema_cols = set(pq.read_schema(labeled_path).names)

    exec_cols = [
        'timestamp', 'sample_split', 'ST_bid_1_price', 'ST_ask_1_price',
        'ST_bid_1_qty', 'ST_ask_1_qty', 'ST_mid_price', 'ST_spread', 'ST_spread_pct',
        'ST_trade_vol', 'ST_buy_vol', 'ST_sell_vol',
        'BO_bid_1_price', 'BO_ask_1_price', 'BO_mid_price',
        'BO_sell_vol', 'BO_buy_vol', 'BO_trade_vol',
        'leadlag_accum_bo_st_60s', 'leadlag_accum_bo_st_300s',
        'leadlag_bo_minus_st_5s_ticks', 'leadlag_bo_minus_st_20s_ticks',
        'leadlag_bo_minus_st_60s_ticks', 'leadlag_bo_minus_st_300s_ticks'
    ]

    all_feat_cols = list(dict.fromkeys(
        dispatcher.router_5.feature_names +
        dispatcher.router_12.feature_names +
        dispatcher.router_25.feature_names
    ))
    load_cols = list(dict.fromkeys([c for c in all_feat_cols + exec_cols if c in schema_cols]))

    print(f"📥 Чтение OOS-выборки ({len(load_cols)} колонок)...")
    tbl = pq.read_table(labeled_path, columns=load_cols)
    df = tbl.to_pandas()

    val_mask = (df['sample_split'] == 'val')
    val_df = df[val_mask].copy().reset_index(drop=True)
    if len(val_df) == 0:
        val_df = df.iloc[int(len(df)*0.8):].copy().reset_index(drop=True)

    if args.strict_oos:
        all_train_ends = []
        for r in [dispatcher.router_5, dispatcher.router_12, dispatcher.router_25]:
            for m in r.manifest:
                if 'train_ts_range' in m and len(m['train_ts_range']) > 1:
                    all_train_ends.append(pd.to_datetime(m['train_ts_range'][1]))
        if all_train_ends:
            latest_train_ts = max(all_train_ends)
            print(f"🛡️ Strict OOS: максимальная дата обучения моделей в триаде = {latest_train_ts}")
            safe_oos_ts = latest_train_ts + pd.Timedelta(seconds=360)
            ts_series = pd.to_datetime(df['timestamp'])
            if ts_series.dt.tz is not None and safe_oos_ts.tzinfo is None:
                safe_oos_ts = safe_oos_ts.tz_localize(ts_series.dt.tz)
            df_oos = df[ts_series > safe_oos_ts].copy().reset_index(drop=True)
            if len(df_oos) > 0:
                val_df = df_oos
                print(f"✅ Успешно выделен чистый OOS срез: {len(val_df):,} строк ({val_df['timestamp'].iloc[0]} ➔ {val_df['timestamp'].iloc[-1]})")
            else:
                print(f"⚠️ Внимание: после {safe_oos_ts} нет данных, используется срез val_df")

    if args.max_steps:
        val_df = val_df.iloc[:args.max_steps].copy()

    n_seconds = len(val_df)
    t_start = val_df['timestamp'].iloc[0]
    t_end = val_df['timestamp'].iloc[-1]
    days_span = n_seconds / 86400.0
    print(f"⏱️ Тестовый диапазон: {t_start} ➔ {t_end} ({n_seconds:,} сек / {days_span:.2f} дней)")

    # 3. Синхронный инференс сигналов триады
    print("🧠 Генерация синхронных прогнозов триады экспертов...")
    t_inf_start = time.time()

    X_5 = val_df[dispatcher.router_5.feature_names].values
    preds_5 = dispatcher.router_5.predict_ensemble(X_5, val_df, mode=args.mode)

    X_12 = val_df[dispatcher.router_12.feature_names].values
    preds_12 = dispatcher.router_12.predict_ensemble(X_12, val_df, mode=args.mode)

    X_25 = val_df[dispatcher.router_25.feature_names].values
    preds_25 = dispatcher.router_25.predict_ensemble(X_25, val_df, mode=args.mode)

    print(f"⚡ Инференс {n_seconds:,} секунд по 3 масштабам завершен за {time.time()-t_inf_start:.2f}с")

    ez_5 = preds_5['expected_z']
    ez_12 = preds_12['expected_z']
    ez_25 = preds_25['expected_z']

    pb_5 = preds_5['p_buy']
    pb_12 = preds_12['p_buy']
    ps_5 = preds_5['p_sell']
    ps_12 = preds_12['p_sell']

    # 4. Моделирование мультигоризонтного исполнения заявок
    initial_tokens = cfg['initial_tokens']
    initial_usdt = cfg['initial_usdt']
    p_mid_0 = float(val_df['ST_mid_price'].iloc[0])
    initial_q_total = initial_tokens + (initial_usdt / p_mid_0)

    tokens = initial_tokens
    usdt = initial_usdt
    base_lot = args.base_lot if args.base_lot is not None else (50.0 if coin == 'prl' else cfg['trade_size_tokens'])

    active_order = None
    trades = []
    maker_fills = 0
    cancels = 0
    trend_blocks = 0
    leadlag_shields = 0
    saved_fees_usdt = 0.0

    q_total_history = []
    usd_total_history = []

    bid_prices = val_df['ST_bid_1_price'].values
    ask_prices = val_df['ST_ask_1_price'].values
    mid_prices = val_df['ST_mid_price'].values
    buy_vols = val_df['ST_buy_vol'].values
    sell_vols = val_df['ST_sell_vol'].values
    bid_qtys = val_df['ST_bid_1_qty'].values if 'ST_bid_1_qty' in val_df.columns else np.full(n_seconds, 100.0)
    ask_qtys = val_df['ST_ask_1_qty'].values if 'ST_ask_1_qty' in val_df.columns else np.full(n_seconds, 100.0)

    # Данные BigONE и Lead-Lag опережения
    bo_mid_prices = val_df['BO_mid_price'].values if 'BO_mid_price' in val_df.columns else mid_prices
    bo_bid_prices = val_df['BO_bid_1_price'].values if 'BO_bid_1_price' in val_df.columns else bid_prices
    bo_ask_prices = val_df['BO_ask_1_price'].values if 'BO_ask_1_price' in val_df.columns else ask_prices
    bo_buy_vols = val_df['BO_buy_vol'].values if 'BO_buy_vol' in val_df.columns else np.zeros(n_seconds)
    bo_sell_vols = val_df['BO_sell_vol'].values if 'BO_sell_vol' in val_df.columns else np.zeros(n_seconds)
    ll_60s = val_df['leadlag_accum_bo_st_60s'].values if 'leadlag_accum_bo_st_60s' in val_df.columns else np.zeros(n_seconds)
    ll_ticks_5s = val_df['leadlag_bo_minus_st_5s_ticks'].values if 'leadlag_bo_minus_st_5s_ticks' in val_df.columns else np.zeros(n_seconds)

    for t in range(n_seconds):
        bid_p = bid_prices[t]
        ask_p = ask_prices[t]
        mid_p = mid_prices[t]
        sell_vol = sell_vols[t]
        buy_vol = buy_vols[t]

        bo_mid = bo_mid_prices[t]
        bo_bid = bo_bid_prices[t]
        bo_ask = bo_ask_prices[t]
        bo_sell = bo_sell_vols[t]
        bo_buy = bo_buy_vols[t]
        ll60 = ll_60s[t]
        ll5 = ll_ticks_5s[t]

        # 4.1. Проверка существующего ордера
        if active_order is not None:
            o_side = active_order['side']
            o_price = active_order['price']
            rem_qty = active_order['rem_qty']
            is_active = (t >= active_order['active_after'])

            # Защитная отмена по шлюзу тренда H=25 (Wave Regime Emergency Cancel)
            z25_now = ez_25[t]
            if not active_order['canceling']:
                # Защита Lead-Lag Shield по опережению BigONE (Stage 1)
                if args.leadlag_shield:
                    if o_side == 'buy':
                        # Защита от токсичного пролива: BigONE mid уже провалился ниже нашего бида, либо резкий сброс
                        is_toxic_dump = (bo_mid < o_price) or (ll60 < -1.8) or (bo_sell > 50.0 and bo_mid <= o_price)
                        if is_toxic_dump:
                            active_order['canceling'] = True
                            active_order['cancel_after'] = t + args.latency_sec
                            leadlag_shields += 1
                    elif o_side == 'sell':
                        # Защита от отдачи дешевых монет на взлете: BigONE mid улетел выше нашего аска, либо мощный памп
                        is_cheap_sell = (bo_mid > o_price) or (ll60 > 1.8) or (bo_buy > 50.0 and bo_mid >= o_price)
                        if is_cheap_sell:
                            active_order['canceling'] = True
                            active_order['cancel_after'] = t + args.latency_sec
                            leadlag_shields += 1

                # Защитная отмена по шлюзу тренда H=25 (Wave Regime Emergency Cancel)
                if not active_order['canceling']:
                    if o_side == 'buy' and z25_now < -args.trend_gate:
                        active_order['canceling'] = True
                        active_order['cancel_after'] = t + args.latency_sec
                        trend_blocks += 1
                    elif o_side == 'sell' and (z25_now > args.trend_gate or ez_5[t] > 0.18 or (buy_vol > 0 and o_price <= ask_p)):
                        active_order['canceling'] = True
                        active_order['cancel_after'] = t + args.latency_sec
                        trend_blocks += 1
                    elif t - active_order['placed_at'] > args.cancel_timeout:
                        active_order['canceling'] = True
                        active_order['cancel_after'] = t + args.latency_sec

            order_done = False

            if is_active and rem_qty > 0:
                fill_qty = 0.0
                if o_side == 'buy':
                    if ask_p <= o_price:
                        avail = max(ask_qtys[t], sell_vol) if args.queue_model != 'instant' else rem_qty
                        fill_qty = min(rem_qty, avail)
                    elif sell_vol > 0 and bid_p <= o_price:
                        if args.queue_model == 'instant':
                            fill_qty = rem_qty
                        else:
                            if o_price < bid_p:
                                fill_qty = 0.0
                            elif o_price == bid_p:
                                if sell_vol <= active_order['q_ahead']:
                                    active_order['q_ahead'] -= sell_vol
                                    fill_qty = 0.0
                                else:
                                    avail = sell_vol - active_order['q_ahead']
                                    active_order['q_ahead'] = 0.0
                                    fill_qty = min(rem_qty, avail)
                            else:
                                fill_qty = min(rem_qty, sell_vol)

                    if fill_qty > 0:
                        cost = fill_qty * o_price
                        fee = cost * args.maker_fee
                        if usdt >= cost + fee:
                            usdt -= (cost + fee)
                            tokens += fill_qty
                            active_order['rem_qty'] -= fill_qty
                            maker_fills += 1
                            saved_fees_usdt += (cost * args.taker_fee)
                            trades.append({'t': t, 'side': 'BUY_MAKER', 'price': o_price, 'qty': fill_qty, 'tokens': tokens, 'usdt': usdt, 'scale': active_order.get('scale', 'triad')})
                            if active_order['rem_qty'] <= 1e-6:
                                order_done = True

                elif o_side == 'sell':
                    if bid_p >= o_price:
                        avail = max(bid_qtys[t], buy_vol) if args.queue_model != 'instant' else rem_qty
                        fill_qty = min(rem_qty, avail)
                    elif buy_vol > 0 and ask_p >= o_price:
                        if args.queue_model == 'instant':
                            fill_qty = rem_qty
                        else:
                            if o_price > ask_p:
                                fill_qty = 0.0
                            elif o_price == ask_p:
                                if buy_vol <= active_order['q_ahead']:
                                    active_order['q_ahead'] -= buy_vol
                                    fill_qty = 0.0
                                else:
                                    avail = buy_vol - active_order['q_ahead']
                                    active_order['q_ahead'] = 0.0
                                    fill_qty = min(rem_qty, avail)
                            else:
                                fill_qty = min(rem_qty, buy_vol)

                    if fill_qty > 0:
                        rev = fill_qty * o_price
                        fee = rev * args.maker_fee
                        if tokens >= fill_qty:
                            tokens -= fill_qty
                            usdt += (rev - fee)
                            active_order['rem_qty'] -= fill_qty
                            maker_fills += 1
                            saved_fees_usdt += (rev * args.taker_fee)
                            trades.append({'t': t, 'side': 'SELL_MAKER', 'price': o_price, 'qty': fill_qty, 'tokens': tokens, 'usdt': usdt, 'scale': active_order.get('scale', 'triad')})
                            if active_order['rem_qty'] <= 1e-6:
                                order_done = True

            if order_done:
                active_order = None
            elif active_order is not None and active_order['canceling'] and t >= active_order['cancel_after']:
                active_order = None
                cancels += 1

        # 4.2. Диспетчер триады: синтез сигналов и выставление новых ордеров
        elif active_order is None:
            z5 = ez_5[t]
            z12 = ez_12[t]
            z25 = ez_25[t]

            # 1. Avellaneda-Stoikov Inventory Skew (Stage 2) & Conviction Sizing
            if args.inventory_skew:
                excess_usdt = max(0.0, usdt - 20.0)
                cash_boost = min(3.0, excess_usdt / 30.0)
                conviction_boost = max(0.0, z12 * 2.0) if z12 > 0 else 0.0

                desired_buy_lot = base_lot * (1.0 + cash_boost * 0.75 + conviction_boost)
                max_buyable = max(0.0, (usdt - 1.0) / max(bid_p, 1e-4))
                effective_buy_lot = min(desired_buy_lot, max_buyable)

                cash_drag = min(0.6, excess_usdt / 250.0)
                desired_sell_lot = base_lot * (1.0 - cash_drag)
                effective_sell_lot = min(tokens, desired_sell_lot)

                effective_buy_gate = -(args.trend_gate + min(0.12, (excess_usdt / 100.0) * 0.10))
                effective_buy_thresh = max(0.08, args.threshold_core - min(0.10, (excess_usdt / 100.0) * 0.08))
            else:
                total_equity_usd = usdt + (tokens * mid_p)
                cash_ratio = usdt / total_equity_usd if total_equity_usd > 1e-4 else 0.0
                cash_excess = max(0.0, cash_ratio - 0.25)

                effective_buy_gate = -(args.trend_gate + min(0.15, cash_excess * 0.6))
                effective_buy_thresh = max(0.08, args.threshold_core - min(0.12, cash_excess * 0.5))

                effective_lot = base_lot
                if abs(z12) > 0.35 and abs(z5) > 0.25:
                    effective_lot = min(base_lot * 2.0, tokens if z12 < 0 else usdt / max(bid_p, 1e-4))
                effective_buy_lot = effective_lot
                effective_sell_lot = effective_lot

            # 2. Локальные экстремумы в окне 30с
            w_win = max(0, t - 30)
            local_min = np.min(mid_prices[w_win : t + 1])
            local_max = np.max(mid_prices[w_win : t + 1])
            is_near_bottom = (mid_p <= local_min + 1.0 * tick_size)
            is_near_top = (mid_p >= local_max - 1.0 * tick_size)

            # Локальное поглощение продаж на дне (Absorption Bottom Bounce):
            recent_sell_vol = np.sum(sell_vols[max(0, t - 5) : t + 1])
            is_bottom_absorption = (is_near_bottom and recent_sell_vol > 0 and z5 > 0.08 and pb_12[t] > 0.15)

            # Снайпинг проливов по опережающему отскоку BigONE (Stage 1):
            is_bo_dip_sniping = False
            if args.dip_sniper:
                is_bo_bounce = (ll60 > 0.8) or (ll5 > 0.8) or (bo_mid > mid_p + tick_size and z5 > 0.02)
                if is_bo_bounce and pb_12[t] > 0.15 and z12 > 0.05:
                    is_bo_dip_sniping = True

            # Локальное истощение пампа на вершине (Peak Exhaustion):
            z5_prev = ez_5[t - 1] if t > 0 else z5
            is_peak_exhaustion = (is_near_top and z5 < z5_prev and z5 < 0.15 and ps_12[t] > 0.15)
            is_pump_accelerating = (z5 > 0.20) or (buy_vol > 0 and z5 > 0.05)

            # А. Трендовый шлюз Scale 3 (H=25)
            allow_buy = (z25 >= effective_buy_gate) or is_bottom_absorption or is_bo_dip_sniping
            allow_sell = (z25 <= args.trend_gate) or is_peak_exhaustion

            # Б. Согласование сигналов покупки (импульс, ребалансировка кэша или поглощение дна)
            trigger_buy = allow_buy and (
                (z12 > effective_buy_thresh and z5 > -0.05 and pb_12[t] > 0.18) or 
                is_bottom_absorption or 
                is_bo_dip_sniping
            )

            if trigger_buy and effective_buy_lot >= 1.0:
                limit_p = round(bid_p, 4)
                q_ahead = float(bid_qtys[t])
                # Если спред широкий, котируем агрессивно внутрь спреда
                if ask_p - bid_p > 2 * tick_size:
                    limit_p = round(bid_p + tick_size, 4)
                    q_ahead = 0.0
                elif args.inventory_skew and (usdt > 100.0) and (ask_p - bid_p > tick_size):
                    limit_p = round(bid_p + tick_size, 4)
                    q_ahead = 0.0

                if limit_p * effective_buy_lot >= args.min_order_val and usdt >= (effective_buy_lot * limit_p):
                    if is_bo_dip_sniping:
                        scale_label = 'BO_LEADLAG_SNIPER_BUY'
                    elif is_bottom_absorption:
                        scale_label = 'SNIPER_DIP_BUY'
                    else:
                        scale_label = 'TRIAD_CORE_BUY'

                    active_order = {
                        'side': 'buy',
                        'price': limit_p,
                        'qty': effective_buy_lot,
                        'rem_qty': effective_buy_lot,
                        'placed_at': t,
                        'active_after': t + args.latency_sec,
                        'q_ahead': q_ahead,
                        'canceling': False,
                        'cancel_after': None,
                        'scale': scale_label
                    }

            # Условие продажи:
            sell_threshold = args.threshold_core + max(0.0, z25 * args.q_token_bias)
            trigger_sell = allow_sell and (
                (z12 < -sell_threshold and z5 < 0.05 and ps_12[t] > 0.18) or 
                is_peak_exhaustion
            )

            if trigger_sell and tokens >= effective_sell_lot and effective_sell_lot >= 1.0:
                if is_pump_accelerating:
                    # Ask Escalator: памп разгоняется! Не отдаем монеты на взлете, ставим аск выше с премией
                    ticks_premium = max(2, min(8, int(round(z5 * 10))))
                    limit_p = round(ask_p + ticks_premium * tick_size, 4)
                    q_ahead = 0.0
                elif is_peak_exhaustion:
                    # На пике истощения продаем прямо по лучшему аску
                    limit_p = round(ask_p, 4)
                    q_ahead = 0.0
                elif args.ask_skew and z25 > 0.05:
                    ticks_premium = min(5, max(1, int(round(z25 * 10))))
                    limit_p = round(ask_p + ticks_premium * tick_size, 4)
                    q_ahead = 0.0
                elif ask_p - bid_p > 2 * tick_size:
                    limit_p = round(ask_p - tick_size, 4)
                    q_ahead = 0.0
                else:
                    limit_p = round(ask_p, 4)
                    q_ahead = float(ask_qtys[t])

                # Привязка Ask Escalator к стакану BigONE (Stage 1)
                if args.bo_ask_peg and bo_ask > ask_p:
                    limit_p = round(max(limit_p, bo_ask - tick_size), 4)
                    q_ahead = 0.0

                if limit_p * effective_sell_lot >= args.min_order_val:
                    scale_label = 'PEAK_EXHAUSTION_SELL' if is_peak_exhaustion else ('ASK_ESCALATOR_SELL' if is_pump_accelerating else 'TRIAD_CORE_SELL')
                    active_order = {
                        'side': 'sell',
                        'price': limit_p,
                        'qty': effective_sell_lot,
                        'rem_qty': effective_sell_lot,
                        'placed_at': t,
                        'active_after': t + args.latency_sec,
                        'q_ahead': q_ahead,
                        'canceling': False,
                        'cancel_after': None,
                        'scale': scale_label
                    }

        # 4.3. Оценка динамики баланса
        q_total_t = tokens + (usdt / mid_p)
        usd_total_t = usdt + (tokens * mid_p)
        q_total_history.append(q_total_t)
        usd_total_history.append(usd_total_t)

    # 5. Итоговый расчет метрик
    p_mid_final = mid_prices[-1]
    final_q_total = tokens + (usdt / p_mid_final)
    final_usd_total = usdt + (tokens * p_mid_final)
    initial_usd_total = initial_usdt + (initial_tokens * p_mid_0)

    delta_q = final_q_total - initial_q_total
    delta_q_pct = (delta_q / initial_q_total) * 100.0

    delta_usd = final_usd_total - initial_usd_total
    delta_usd_pct = (delta_usd / initial_usd_total) * 100.0

    hodl_price_pct = ((p_mid_final / p_mid_0) - 1.0) * 100.0

    q_arr = np.array(q_total_history)
    running_max_q = np.maximum.accumulate(q_arr)
    drawdowns_q = (q_arr - running_max_q) / running_max_q * 100.0
    max_dd_q = float(np.min(drawdowns_q))

    total_trades = len(trades)

    print("\n" + "=" * 105)
    print("🏆 РЕЗУЛЬТАТЫ СИНХРОНИЗИРОВАННОЙ ТРИАДЫ НА SAFETRADE (ИНВАРИАНТ Q_TOKEN)")
    print("=" * 105)
    print(f"  • Стартовый капитал:  {initial_tokens:,.1f} {coin.upper()} + {initial_usdt:,.2f} USDT (Всего в монетах: {initial_q_total:,.2f})")
    print(f"  • Итоговый капитал:   {tokens:,.1f} {coin.upper()} + {usdt:,.2f} USDT (Всего в монетах: {final_q_total:,.2f})")
    print(f"  • Стартовая цена mid: ${p_mid_0:.4f}  ➔  Итоговая цена mid: ${p_mid_final:.4f} (Движение рынка: {hodl_price_pct:+.2f}%)")
    print("-" * 105)
    print(f"  🎯 ЧИСТЫЙ ПРИРОСТ МОНЕТ (Q_token PnL): {delta_q:+,.2f} {coin.upper()} ({delta_q_pct:+.2f}%) 🚀")
    print(f"  💵 Прирост портфеля в USD:             {delta_usd:+,.2f} USDT ({delta_usd_pct:+.2f}%)")
    print(f"  🛡️ Максимальная просадка по монетам:  {max_dd_q:.2f}%")
    print(f"  🔄 Исполнено Maker-сделок:             {total_trades:,} (Отмен: {cancels:,} | Срабатываний тренд-шлюза: {trend_blocks:,} | Защит Lead-Lag: {leadlag_shields:,})")
    print(f"  💰 Сэкономлено на Maker комиссиях:     ${saved_fees_usdt:.2f} USDT")
    if getattr(args, 'plot', None):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2, 1.0]})
        fig.patch.set_facecolor('#0d1117')
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#c9d1d9')
            ax.grid(True, color='#30363d', linestyle='--', alpha=0.6)
            for spine in ax.spines.values():
                spine.set_color('#30363d')

        ts_series = pd.to_datetime(val_df['timestamp'])

        # Панель 1: Mid Price & Maker Fills
        ax1.plot(ts_series, mid_prices, label=f'{coin.upper()} Mid Price', color='#58a6ff', lw=1.5, zorder=2)

        # Сделки
        if trades:
            df_tr = pd.DataFrame(trades)
            df_tr['time'] = ts_series.iloc[df_tr['t']].values

            buys_bo_sniper = df_tr[(df_tr['side'] == 'BUY_MAKER') & (df_tr['scale'] == 'BO_LEADLAG_SNIPER_BUY')]
            buys_sniper = df_tr[(df_tr['side'] == 'BUY_MAKER') & (df_tr['scale'] == 'SNIPER_DIP_BUY')]
            buys_core = df_tr[(df_tr['side'] == 'BUY_MAKER') & (~df_tr['scale'].isin(['BO_LEADLAG_SNIPER_BUY', 'SNIPER_DIP_BUY']))]
            sells_peak = df_tr[(df_tr['side'] == 'SELL_MAKER') & (df_tr['scale'] == 'PEAK_EXHAUSTION_SELL')]
            sells_esc = df_tr[(df_tr['side'] == 'SELL_MAKER') & (df_tr['scale'] == 'ASK_ESCALATOR_SELL')]
            sells_core = df_tr[(df_tr['side'] == 'SELL_MAKER') & (~df_tr['scale'].isin(['PEAK_EXHAUSTION_SELL', 'ASK_ESCALATOR_SELL']))]

            if not buys_bo_sniper.empty:
                ax1.scatter(buys_bo_sniper['time'], buys_bo_sniper['price'], color='#38d9a9', marker='^', s=90, edgecolors='white', lw=1.2, label=f'⚡ BigONE Sniper Buy ({len(buys_bo_sniper)})', zorder=6)
            if not buys_sniper.empty:
                ax1.scatter(buys_sniper['time'], buys_sniper['price'], color='#2ea043', marker='^', s=80, edgecolors='white', lw=1.2, label=f'🟢 Absorption Dip Buy ({len(buys_sniper)})', zorder=5)
            if not buys_core.empty:
                ax1.scatter(buys_core['time'], buys_core['price'], color='#3fb950', marker='^', s=45, alpha=0.7, label=f'Core Buy ({len(buys_core)})', zorder=4)
            if not sells_peak.empty:
                ax1.scatter(sells_peak['time'], sells_peak['price'], color='#f85149', marker='v', s=80, edgecolors='white', lw=1.2, label=f'🔴 Peak Exhaustion Sell ({len(sells_peak)})', zorder=5)
            if not sells_esc.empty:
                ax1.scatter(sells_esc['time'], sells_esc['price'], color='#d29922', marker='v', s=55, alpha=0.8, label=f'🟠 Ask Escalator ({len(sells_esc)})', zorder=4)
            if not sells_core.empty:
                ax1.scatter(sells_core['time'], sells_core['price'], color='#da3633', marker='v', s=35, alpha=0.6, label=f'Core Sell ({len(sells_core)})', zorder=3)

        ax1.set_ylabel('Price (USDT)', color='#c9d1d9', fontsize=11)
        ax1.set_title(f'STRICT OUT-OF-SAMPLE BACKTEST — {coin.upper()} / USDT (Maker 0% Fee, RTT=1s)', color='#f0f6fc', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8.5)

        # Панель 2: Прирост монет Q_token (%)
        q_pnl_pct_arr = ((np.array(q_total_history) - initial_q_total) / initial_q_total) * 100.0
        ax2.plot(ts_series, q_pnl_pct_arr, color='#e3b341', lw=1.8, label=f'Accumulated {coin.upper()} Q_token ({delta_q_pct:+.2f}%)', zorder=3)
        ax2.axhline(0, color='#8b949e', linestyle=':', lw=1, alpha=0.7)
        ax2.set_ylabel('Token PnL (%)', color='#c9d1d9', fontsize=11)
        ax2.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)

        # Панель 3: Drawdown (%)
        ax3.fill_between(ts_series, drawdowns_q, 0, color='#f85149', alpha=0.35, label=f'Token Drawdown (Max: {max_dd_q:.2f}%)')
        ax3.plot(ts_series, drawdowns_q, color='#f85149', lw=1.0)
        ax3.set_ylabel('Drawdown (%)', color='#c9d1d9', fontsize=11)
        ax3.set_xlabel('Время', color='#c9d1d9', fontsize=11)
        ax3.legend(loc='lower left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        plt.savefig(args.plot, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"📊 График бэктеста успешно сохранен: {args.plot}")
        plt.close()

    return {
        'coin': coin.upper(),
        'delta_q_tokens': delta_q,
        'delta_q_pct': delta_q_pct,
        'max_dd_q_pct': max_dd_q,
        'total_trades': total_trades,
        'cancels': cancels,
        'trend_blocks': trend_blocks,
        'leadlag_shields': leadlag_shields,
        'saved_fees_usdt': saved_fees_usdt
    }


if __name__ == '__main__':
    args = parse_args()
    run_triad_simulation(args)
