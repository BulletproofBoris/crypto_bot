#!/usr/bin/env python3
"""
Симулятор микроструктурного исполнения сделок на бирже SafeTrade с интеграцией SwarmRegimeRouter.

Ключевой инвариант проекта:
- Целевая функция: максимальный физический прирост монет (Q_token), а не нестабильный USD PnL.
- Реалистичное исполнение лимитных ордеров (Maker, 0.0% комиссия) внутри спреда и на BBO.
- Защита от фантомных сделок: ордер считается исполненным только при реальном торговом потоке
  (ST_sell_vol для Bid, ST_buy_vol для Ask) либо при физическом пересечении спреда.
- Срочные выходы (Taker, 0.2% комиссия) по сигналам экстремальных выбросов P(±3).
- Двойное квантование цены (tick_size: 0.005 для PRL, 0.05 для Quantus).
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

# Импортируем наш SwarmRegimeRouter
sys.path.insert(0, str(BASE_DIR))
from _tools.swarm_router import SwarmRegimeRouter
from _tools.data_labeler import compute_single_horizon_labels, COIN_CONFIG

COIN_RULES = {
    'prl': {
        'tick_size': 0.005,
        'min_vol': 1.0,
        'initial_tokens': 10000.0,
        'initial_usdt': 100.0,
        'trade_size_tokens': 200.0,
    },
    'quantus': {
        'tick_size': 0.05,
        'min_vol': 1.0,
        'initial_tokens': 1000.0,
        'initial_usdt': 100.0,
        'trade_size_tokens': 20.0,
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="SafeTrade Execution Simulator with Swarm Router")
    parser.add_argument("--coin", type=str, default="prl", help="Монета (prl, quantus)")
    parser.add_argument("--horizon", type=int, default=5, help="Горизонт модели H (сек)")
    parser.add_argument("--ratio", type=float, default=None, help="Соотношение K (L/H)")
    parser.add_argument("--mode", type=str, default="router", choices=["router", "uniform", "latest"],
                        help="Режим ансамбля (router, uniform, latest)")
    parser.add_argument("--maker-fee", type=float, default=0.000, help="Комиссия Maker (0.0%% на SafeTrade)")
    parser.add_argument("--taker-fee", type=float, default=0.002, help="Комиссия Taker (0.2%% на SafeTrade)")
    parser.add_argument("--threshold-maker", type=float, default=0.25, help="Порог входа лимитным ордером E[z]")
    parser.add_argument("--threshold-taker", type=float, default=0.35, help="Порог экстренного маркет-ордера P(±3)")
    parser.add_argument("--max-steps", type=int, default=None, help="Ограничить количество секунд для быстрого теста")
    return parser.parse_args()


def run_execution_simulation(args):
    coin = args.coin.lower()
    cfg = COIN_RULES.get(coin, COIN_RULES['prl'])
    tick_size = cfg['tick_size']

    H = args.horizon
    if args.ratio is not None:
        K = args.ratio
    else:
        if H == 5:
            K = 3.0
        elif H in (12, 13, 14, 15):
            K = 3.5
        elif H >= 18:
            K = 3.0
        else:
            K = 3.0

    k_str = f"{K:g}"

    print("\n" + "=" * 105)
    print(f"🚀 СИМУЛЯЦИЯ ИСПОЛНЕНИЯ SAFETRADE: {coin.upper()} | H = {H}с, K = {k_str} | Режим: {args.mode.upper()}")
    print(f"   Шаг цены (tick_size): {tick_size} USDT | Комиссии: Maker={args.maker_fee*100:.1f}%, Taker={args.taker_fee*100:.1f}%")
    print(f"   Инвариант: целевая функция оптимизирована на чистый прирост монет (Q_token)")
    print("=" * 105)

    # 1. Загрузка SwarmRouter
    router = SwarmRegimeRouter(coin, H, K)

    # 2. Загрузка данных тестирования (чистый OOS)
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    target_c7_col = f"target_class_7_h{H}_k{k_str}"
    target_dt_col = f"target_dt_ticks_h{H}_k{k_str}"

    schema_cols = set(pq.read_schema(labeled_path).names)
    need_labeling = (target_c7_col not in schema_cols) or (target_dt_col not in schema_cols)

    exec_cols = [
        'timestamp', 'sample_split', 'ST_bid_1_price', 'ST_ask_1_price',
        'ST_bid_1_qty', 'ST_ask_1_qty', 'ST_mid_price', 'ST_spread', 'ST_spread_pct',
        'ST_trade_vol', 'ST_buy_vol', 'ST_sell_vol',
        'ST_bids_vol_top5', 'ST_asks_vol_top5'
    ]
    if not need_labeling:
        exec_cols += [target_c7_col, target_dt_col]

    load_cols = list(dict.fromkeys([c for c in router.feature_names + exec_cols if c in schema_cols]))
    
    print("📥 Чтение тестового OOS-датасета...")
    tbl = pq.read_table(labeled_path, columns=load_cols)
    df = tbl.to_pandas()

    if need_labeling:
        cfg_label = COIN_CONFIG[coin]
        labels = compute_single_horizon_labels(df, coin, H, K, cfg_label['min_vol'], cfg_label['tick_size'])
        df['target_class'] = labels['class_7']
        df['target_dt'] = labels['dt_ticks']
    else:
        df['target_class'] = df[target_c7_col]
        df['target_dt'] = df[target_dt_col]

    val_mask = (df['sample_split'] == 'val')
    val_df = df[val_mask].copy().reset_index(drop=True)
    if len(val_df) == 0:
        val_df = df.iloc[int(len(df)*0.8):].copy().reset_index(drop=True)

    if args.max_steps:
        val_df = val_df.iloc[:args.max_steps].copy()

    n_seconds = len(val_df)
    t_start = val_df['timestamp'].iloc[0]
    t_end = val_df['timestamp'].iloc[-1]
    days_span = n_seconds / 86400.0
    print(f"⏱️ Тестовый диапазон: {t_start} ➔ {t_end} ({n_seconds:,} сек / {days_span:.2f} дней)")

    # 3. Инференс сигналов через роутер
    print("🧠 Генерация сигналов через SwarmRegimeRouter...")
    t_inf_start = time.time()
    X_val = val_df[router.feature_names].values
    preds = router.predict_ensemble(X_val, val_df, mode=args.mode)
    expected_z = preds['expected_z']
    p_buy = preds['p_buy']
    p_sell = preds['p_sell']
    probs = preds['probs']
    print(f"⚡ Инференс {n_seconds:,} тиков завершен за {time.time()-t_inf_start:.2f}с (скорость: {n_seconds/(time.time()-t_inf_start):,.0f} оп/сек)")

    # 4. Моделирование исполнения SafeTrade Orderbook
    # Портфель
    initial_tokens = cfg['initial_tokens']
    initial_usdt = cfg['initial_usdt']
    p_mid_0 = float(val_df['ST_mid_price'].iloc[0])
    initial_q_total = initial_tokens + (initial_usdt / p_mid_0)

    tokens = initial_tokens
    usdt = initial_usdt
    trade_size = cfg['trade_size_tokens']

    # Состояние активных ордеров
    active_order = None  # {'side': 'buy'/'sell', 'price': float, 'qty': float, 'placed_at': int}
    
    # Статистика
    trades = []
    maker_fills = 0
    taker_fills = 0
    cancels = 0
    saved_fees_usdt = 0.0

    q_total_history = []
    usd_total_history = []

    bid_prices = val_df['ST_bid_1_price'].values
    ask_prices = val_df['ST_ask_1_price'].values
    mid_prices = val_df['ST_mid_price'].values
    buy_vols = val_df['ST_buy_vol'].values
    sell_vols = val_df['ST_sell_vol'].values

    for t in range(n_seconds):
        bid_p = bid_prices[t]
        ask_p = ask_prices[t]
        mid_p = mid_prices[t]
        sell_vol = sell_vols[t]
        buy_vol = buy_vols[t]

        # 4.1. Проверка исполнения существующего ордера
        if active_order is not None:
            o_side = active_order['side']
            o_price = active_order['price']
            o_qty = active_order['qty']

            filled = False
            # Maker BUY: исполняется если пришли рыночные продажи (sell_vol > 0) на уровне <= o_price
            # или если спред опустился до цены ордера
            if o_side == 'buy':
                if (sell_vol > 0 and bid_p <= o_price) or (ask_p <= o_price):
                    cost = o_qty * o_price
                    fee = cost * args.maker_fee
                    if usdt >= cost + fee:
                        usdt -= (cost + fee)
                        tokens += o_qty
                        filled = True
                        maker_fills += 1
                        saved_fees_usdt += (cost * args.taker_fee)  # сэкономили разницу с Taker
                        trades.append({'t': t, 'side': 'BUY_MAKER', 'price': o_price, 'qty': o_qty, 'tokens': tokens, 'usdt': usdt})

            # Maker SELL: исполняется если пришли рыночные покупки (buy_vol > 0) на уровне >= o_price
            # или если бид поднялся до цены ордера
            elif o_side == 'sell':
                if (buy_vol > 0 and ask_p >= o_price) or (bid_p >= o_price):
                    rev = o_qty * o_price
                    fee = rev * args.maker_fee
                    if tokens >= o_qty:
                        tokens -= o_qty
                        usdt += (rev - fee)
                        filled = True
                        maker_fills += 1
                        saved_fees_usdt += (rev * args.taker_fee)
                        trades.append({'t': t, 'side': 'SELL_MAKER', 'price': o_price, 'qty': o_qty, 'tokens': tokens, 'usdt': usdt})

            if filled:
                active_order = None
            elif t - active_order['placed_at'] > 30:  # Ордер завис дольше 30 сек без исполнения -> отмена
                active_order = None
                cancels += 1

        # 4.2. Анализ сигналов SwarmRouter для новых действий
        ez = expected_z[t]
        pb = p_buy[t]
        ps = p_sell[t]
        p_p3 = probs[t, 6]  # Class +3 (Экстремальный памп)
        p_m3 = probs[t, 0]  # Class -3 (Экстремальный дамп)

        # А. Экстренный TAKER (импульс кита / защита от сквиза)
        if p_p3 > args.threshold_taker and usdt >= (trade_size * ask_p * 1.003):
            # Taker BUY
            exec_p = ask_p
            cost = trade_size * exec_p
            fee = cost * args.taker_fee
            usdt -= (cost + fee)
            tokens += trade_size
            taker_fills += 1
            active_order = None
            trades.append({'t': t, 'side': 'BUY_TAKER', 'price': exec_p, 'qty': trade_size, 'tokens': tokens, 'usdt': usdt})

        elif p_m3 > args.threshold_taker and tokens >= trade_size:
            # Taker SELL
            exec_p = bid_p
            rev = trade_size * exec_p
            fee = rev * args.taker_fee
            tokens -= trade_size
            usdt += (rev - fee)
            taker_fills += 1
            active_order = None
            trades.append({'t': t, 'side': 'SELL_TAKER', 'price': exec_p, 'qty': trade_size, 'tokens': tokens, 'usdt': usdt})

        # Б. Регулярный MAKER (вход в спред по сигналу роутера)
        elif active_order is None:
            # Сигнал на покупку: ожидаемый импульс вверх и вероятность роста > порога
            if ez > args.threshold_maker and pb > 0.20:
                if usdt >= (trade_size * bid_p):
                    # Ставим лимитный ордер на лучший Bid (или внутри спреда, если спред широкий)
                    limit_p = round(bid_p, 4)
                    if ask_p - bid_p > 2 * tick_size:
                        limit_p = round(bid_p + tick_size, 4)
                    active_order = {'side': 'buy', 'price': limit_p, 'qty': trade_size, 'placed_at': t}

            # Сигнал на продажу: ожидаемый импульс вниз и вероятность спада > порога
            elif ez < -args.threshold_maker and ps > 0.20:
                if tokens >= trade_size:
                    limit_p = round(ask_p, 4)
                    if ask_p - bid_p > 2 * tick_size:
                        limit_p = round(ask_p - tick_size, 4)
                    active_order = {'side': 'sell', 'price': limit_p, 'qty': trade_size, 'placed_at': t}

        # 4.3. Оценка баланса портфеля в монетах Q_total(t)
        q_total_t = tokens + (usdt / mid_p)
        usd_total_t = usdt + (tokens * mid_p)
        q_total_history.append(q_total_t)
        usd_total_history.append(usd_total_t)

    # 5. Итоговый расчет результатов
    p_mid_final = mid_prices[-1]
    final_q_total = tokens + (usdt / p_mid_final)
    final_usd_total = usdt + (tokens * p_mid_final)
    initial_usd_total = initial_usdt + (initial_tokens * p_mid_0)

    # Прирост монет (главная метрика!)
    delta_q = final_q_total - initial_q_total
    delta_q_pct = (delta_q / initial_q_total) * 100.0

    # Прирост USD
    delta_usd = final_usd_total - initial_usd_total
    delta_usd_pct = (delta_usd / initial_usd_total) * 100.0

    # Buy & Hold прирост цены актива
    hodl_price_pct = ((p_mid_final / p_mid_0) - 1.0) * 100.0

    # Просадка по монетам (MaxDD Q_token)
    q_arr = np.array(q_total_history)
    running_max_q = np.maximum.accumulate(q_arr)
    drawdowns_q = (q_arr - running_max_q) / running_max_q * 100.0
    max_dd_q = float(np.min(drawdowns_q))

    # Статистика сделок
    total_trades = len(trades)
    print("\n" + "=" * 105)
    print("📈 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ НА SAFETRADE (ИНВАРИАНТ Q_TOKEN)")
    print("=" * 105)
    print(f"  • Стартовый капитал:  {initial_tokens:,.1f} {coin.upper()} + {initial_usdt:,.2f} USDT (Всего в монетах: {initial_q_total:,.2f})")
    print(f"  • Итоговый капитал:   {tokens:,.1f} {coin.upper()} + {usdt:,.2f} USDT (Всего в монетах: {final_q_total:,.2f})")
    print(f"  • Стартовая цена mid: ${p_mid_0:.4f}  ➔  Итоговая цена mid: ${p_mid_final:.4f} (Движение рынка: {hodl_price_pct:+.2f}%)")
    print("-" * 105)
    print(f"  🎯 ЧИСТЫЙ ПРИРОСТ МОНЕТ (Q_token PnL): {delta_q:+,.2f} {coin.upper()} ({delta_q_pct:+.2f}%) 🚀")
    print(f"  💵 Прирост портфеля в USD:             {delta_usd:+,.2f} USDT ({delta_usd_pct:+.2f}%)")
    print(f"  🛡️ Максимальная просадка по монетам:  {max_dd_q:.2f}%")
    print(f"  🔄 Всего исполнено сделок:              {total_trades:,} (Maker: {maker_fills:,} | Taker: {taker_fills:,} | Отмен: {cancels:,})")
    print(f"  💰 Сэкономлено на Maker комиссиях:     ${saved_fees_usdt:.2f} USDT")
    print("=" * 105)

    return {
        'coin': coin.upper(),
        'horizon': H,
        'mode': args.mode,
        'initial_q': initial_q_total,
        'final_q': final_q_total,
        'delta_q_tokens': delta_q,
        'delta_q_pct': delta_q_pct,
        'max_dd_q_pct': max_dd_q,
        'total_trades': total_trades,
        'maker_fills': maker_fills,
        'taker_fills': taker_fills,
        'saved_fees_usdt': saved_fees_usdt
    }


if __name__ == '__main__':
    args = parse_args()
    run_execution_simulation(args)
