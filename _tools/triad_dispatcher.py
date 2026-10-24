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
    parser.add_argument("--neural-weight", type=float, default=None,
                        help="Единый вес нейросетей (если задан, переопределяет w-nn-5/12/25)")
    parser.add_argument("--w-nn-5", type=float, default=0.12,
                        help="Вес нейросетевого альянса для H=5с (по умолчанию 0.12)")
    parser.add_argument("--w-nn-12", type=float, default=0.15,
                        help="Вес нейросетевого альянса для H=12с (по умолчанию 0.15)")
    parser.add_argument("--w-nn-25", type=float, default=0.15,
                        help="Вес нейросетевого альянса для H=25с (по умолчанию 0.15)")
    parser.add_argument("--no-neural", dest="use_neural", action="store_false", default=True,
                        help="Отключить нейросетевой альянс (только LightGBM)")
    parser.add_argument("--target-token-ratio", type=float, default=0.55,
                        help="Целевая нейтральная доля портфеля в монетах (по умолчанию 0.55 = 55%% PRL / 45%% USDT)")
    parser.add_argument("--penny-jump-penalty", type=float, default=0.10,
                        help="Штраф конкурентной очереди при шаге внутрь спреда (доля от BBO, по умолчанию 0.10)")
    parser.add_argument("--max-participation", type=float, default=0.50,
                        help="Максимальная доля рыночного объема секунды, которую может забрать один наш ордер (по умолчанию 0.50)")
    parser.add_argument("--stale-timeout", type=int, default=180,
                        help="Таймаут заморозки стакана SafeTrade (сек без сделок и сдвигов BBO) для защитной паузы")
    parser.add_argument("--initial-usdt", type=float, default=None, help="Стартовый баланс USDT (по умолчанию из COIN_RULES: 8000 для PRL)")
    parser.add_argument("--initial-tokens", type=float, default=None, help="Стартовый баланс токенов (по умолчанию из COIN_RULES: 0 для PRL)")
    parser.add_argument("--max-steps", type=int, default=None, help="Ограничить шаги для быстрого теста")
    parser.add_argument("--plot", type=str, default=None, help="Путь для сохранения PNG-графика бэктеста")
    return parser.parse_args()


class TriadDispatcher:
    """Мультигоризонтный координатор триады экспертов (H=5, 12, 25)."""

    def __init__(
        self,
        coin: str,
        use_neural: bool = True,
        neural_weight: float = None,
        w_nn_5: float = 0.12,
        w_nn_12: float = 0.15,
        w_nn_25: float = 0.15,
    ):
        self.coin = coin.lower()
        if neural_weight is not None:
            w5 = w12 = w25 = float(neural_weight)
        else:
            w5, w12, w25 = float(w_nn_5), float(w_nn_12), float(w_nn_25)
        print(
            f"\n📦 Инициализация триады масштабов для {self.coin.upper()} "
            f"(use_neural={use_neural}, w_nn=[H5:{w5:.2f}, H12:{w12:.2f}, H25:{w25:.2f}])..."
        )
        self.router_5 = SwarmRegimeRouter(self.coin, H=5, K=3.0, use_neural=use_neural, neural_weight=w5)
        self.router_12 = SwarmRegimeRouter(self.coin, H=12, K=3.5, use_neural=use_neural, neural_weight=w12)
        self.router_25 = SwarmRegimeRouter(self.coin, H=25, K=3.0, use_neural=use_neural, neural_weight=w25)
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
    dispatcher = TriadDispatcher(
        coin,
        use_neural=args.use_neural,
        neural_weight=args.neural_weight,
        w_nn_5=args.w_nn_5,
        w_nn_12=args.w_nn_12,
        w_nn_25=args.w_nn_25,
    )

    from _tools.triad_utils import load_strict_oos_slice
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    val_df = load_strict_oos_slice(
        labeled_path,
        dispatcher,
        strict_oos=args.strict_oos,
        keep_outage=getattr(args, 'keep_outage', False),
        max_steps=args.max_steps,
    )

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

    X_12 = X_5 if dispatcher.router_12.feature_names == dispatcher.router_5.feature_names else val_df[dispatcher.router_12.feature_names].values
    preds_12 = dispatcher.router_12.predict_ensemble(X_12, val_df, mode=args.mode)

    X_25 = X_5 if dispatcher.router_25.feature_names == dispatcher.router_5.feature_names else val_df[dispatcher.router_25.feature_names].values
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
    initial_tokens = args.initial_tokens if getattr(args, 'initial_tokens', None) is not None else cfg['initial_tokens']
    initial_usdt = args.initial_usdt if getattr(args, 'initial_usdt', None) is not None else cfg['initial_usdt']
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
    stale_skips = 0
    rebuy_blocks = 0
    trend_holds = 0
    saved_fees_usdt = 0.0
    avg_buy_price = p_mid_0
    last_sell_anchor = 0.0
    last_sell_t = -99999
    stale_run = 0
    warmup_left = 0

    q_total_history = []
    usd_total_history = []

    bid_prices = val_df['ST_bid_1_price'].values
    ask_prices = val_df['ST_ask_1_price'].values
    mid_prices = val_df['ST_mid_price'].values
    ts_gaps = pd.to_datetime(val_df['timestamp']).diff().dt.total_seconds().fillna(1.0).values
    seam_indices = np.where(ts_gaps > 600.0)[0]
    ema_30m = np.empty_like(mid_prices, dtype=np.float64)
    seg_starts = [0] + list(seam_indices) + [len(mid_prices)]
    for s_i in range(len(seg_starts) - 1):
        a, b = seg_starts[s_i], seg_starts[s_i + 1]
        ema_30m[a:b] = pd.Series(mid_prices[a:b]).ewm(span=1800, min_periods=1).mean().values
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
    bo_sec_since = val_df['BO_sec_since_tick'].values if 'BO_sec_since_tick' in val_df.columns else np.zeros(n_seconds)
    ll_60s = val_df['leadlag_accum_bo_st_60s'].values if 'leadlag_accum_bo_st_60s' in val_df.columns else np.zeros(n_seconds)
    ll_300s = val_df['leadlag_accum_bo_st_300s'].values if 'leadlag_accum_bo_st_300s' in val_df.columns else np.zeros(n_seconds)
    ll_ticks_5s = val_df['leadlag_bo_minus_st_5s_ticks'].values if 'leadlag_bo_minus_st_5s_ticks' in val_df.columns else np.zeros(n_seconds)
    ll_ticks_20s = val_df['leadlag_bo_minus_st_20s_ticks'].values if 'leadlag_bo_minus_st_20s_ticks' in val_df.columns else np.zeros(n_seconds)

    ts_gaps = pd.to_datetime(val_df['timestamp']).diff().dt.total_seconds().fillna(1.0).values

    for t in range(n_seconds):
        bid_p = bid_prices[t]
        ask_p = ask_prices[t]
        mid_p = mid_prices[t]
        sell_vol = sell_vols[t]
        buy_vol = buy_vols[t]

        if ts_gaps[t] > 60.0:
            warmup_left = 75
            stale_run = 0

        # 3. Защита от «мертвых полок» ffill (Stale Book Guard)
        if t > 0 and sell_vol == 0 and buy_vol == 0 and bid_p == bid_prices[t - 1] and ask_p == ask_prices[t - 1] and bid_qtys[t] == bid_qtys[t - 1]:
            stale_run += 1
        else:
            if stale_run >= args.stale_timeout:
                warmup_left = 75  # Прогрев окон L=75с после восстановления связи с биржей
            stale_run = 0

        if stale_run >= args.stale_timeout or warmup_left > 0:
            if warmup_left > 0 and stale_run == 0:
                warmup_left -= 1
            if active_order is not None:
                active_order = None
                cancels += 1
            stale_skips += 1
            q_total_history.append(tokens + (usdt / mid_p))
            usd_total_history.append(usdt + (tokens * mid_p))
            continue

        bo_fresh = (bo_sec_since[t] <= 60.0)
        bo_mid = bo_mid_prices[t] if bo_fresh else mid_p
        bo_bid = bo_bid_prices[t] if bo_fresh else bid_p
        bo_ask = bo_ask_prices[t] if bo_fresh else ask_p
        bo_sell = bo_sell_vols[t] if bo_fresh else 0.0
        bo_buy = bo_buy_vols[t] if bo_fresh else 0.0
        ll60 = ll_60s[t] if bo_fresh else 0.0
        ll5 = ll_ticks_5s[t] if bo_fresh else 0.0

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
                if args.leadlag_shield and bo_fresh:
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
                    elif o_side == 'sell' and (z25_now > max(0.55, args.trend_gate) or ez_5[t] > 0.65):
                        active_order['canceling'] = True
                        active_order['cancel_after'] = t + args.latency_sec
                        trend_blocks += 1
                    elif t - active_order['placed_at'] > args.cancel_timeout:
                        active_order['canceling'] = True
                        active_order['cancel_after'] = t + args.latency_sec

            order_done = False

            if is_active and rem_qty > 0:
                fill_qty = 0.0
                max_part = float(args.max_participation)
                if o_side == 'buy':
                    if ask_p <= o_price:
                        avail = max(ask_qtys[t] * max_part, sell_vol * max_part) if args.queue_model != 'instant' else rem_qty
                        fill_qty = min(rem_qty, avail)
                    elif sell_vol > 0 and bid_p <= o_price:
                        if args.queue_model == 'instant':
                            fill_qty = rem_qty
                        else:
                            if sell_vol <= active_order['q_ahead']:
                                active_order['q_ahead'] -= sell_vol
                                fill_qty = 0.0
                            else:
                                avail = (sell_vol - active_order['q_ahead']) * max_part
                                active_order['q_ahead'] = 0.0
                                fill_qty = min(rem_qty, avail)

                    if fill_qty > 0:
                        cost = fill_qty * o_price
                        fee = cost * args.maker_fee
                        if usdt >= cost + fee:
                            avg_buy_price = ((tokens * avg_buy_price) + (fill_qty * o_price)) / max(tokens + fill_qty, 1e-6)
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
                        avail = max(bid_qtys[t] * max_part, buy_vol * max_part) if args.queue_model != 'instant' else rem_qty
                        fill_qty = min(rem_qty, avail)
                    elif buy_vol > 0 and ask_p >= o_price:
                        if args.queue_model == 'instant':
                            fill_qty = rem_qty
                        else:
                            if buy_vol <= active_order['q_ahead']:
                                active_order['q_ahead'] -= buy_vol
                                fill_qty = 0.0
                            else:
                                avail = (buy_vol - active_order['q_ahead']) * max_part
                                active_order['q_ahead'] = 0.0
                                fill_qty = min(rem_qty, avail)

                    if fill_qty > 0:
                        rev = fill_qty * o_price
                        fee = rev * args.maker_fee
                        if tokens >= fill_qty:
                            tokens -= fill_qty
                            usdt += (rev - fee)
                            active_order['rem_qty'] -= fill_qty
                            maker_fills += 1
                            saved_fees_usdt += (rev * args.taker_fee)
                            last_sell_anchor = o_price if last_sell_anchor <= 0 else (0.7 * o_price + 0.3 * last_sell_anchor)
                            last_sell_t = t
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
            ll300 = ll_300s[t] if bo_fresh else 0.0
            ll20 = ll_ticks_20s[t] if bo_fresh else 0.0

            # 1. Режимно-адаптивный таргетинг инвентаря (Regime-Adaptive Target Ratio)
            total_equity_usd = usdt + (tokens * mid_p)
            token_ratio = (tokens * mid_p) / max(total_equity_usd, 1e-4)
            ema_val = ema_30m[t]
            rel_dev = (mid_p - ema_val) / max(ema_val, 1e-4)

            # На просадке (rel_dev < 0) или бычьем накоплении BigONE повышаем целевую долю PRL до 82-85%,
            # а на сильном перегреве (> +2.5% над EMA_30m при угасании импульса) снижаем до 45% для фиксации профита
            base_target = float(args.target_token_ratio)
            dip_bonus = np.clip(-rel_dev * 12.0, -0.18, 0.28)
            trend_bonus = 0.15 if (z25 > 0.25 or ll300 > 0.5 or (bo_fresh and bo_mid > mid_p)) else 0.0
            target_ratio = float(np.clip(base_target + 0.15 + dip_bonus + trend_bonus, 0.45, 0.85))
            in_profit = (tokens > 10.0) and (mid_p >= avg_buy_price + 3.0 * tick_size)
            deep_in_profit = (tokens > 500.0) and (token_ratio >= 0.52) and (mid_p >= avg_buy_price + 6.0 * tick_size)

            if args.inventory_skew:
                # Избыток кэша относительно целевого коридора
                underweight_tokens = max(0.0, target_ratio - token_ratio)
                overweight_tokens = max(0.0, token_ratio - target_ratio)

                cash_boost = min(2.8, underweight_tokens * 4.5)
                conviction_boost = max(0.0, min(1.5, z12 * 1.2)) if z12 > 0 else 0.0
                # Тормозим докупку при заполнении портфеля монетами свыше 78%
                top_brake = 0.25 if (token_ratio > 0.78 and (in_profit or rel_dev > 0.0)) else (0.50 if token_ratio > 0.82 else 1.0)

                desired_buy_lot = base_lot * (1.0 + cash_boost + conviction_boost) * top_brake
                max_buyable = max(0.0, (usdt - 1.0) / max(bid_p, 1e-4))
                effective_buy_lot = min(desired_buy_lot, max_buyable)

                # На стороне продаж: если монет много (> target_ratio) и мы в плюсе, увеличиваем лот фиксации прибыли в USDT
                sell_boost = min(2.5, overweight_tokens * 4.5) if in_profit else 0.0
                cash_drag = min(0.5, underweight_tokens * 1.2)
                desired_sell_lot = base_lot * max(0.4, 1.0 - cash_drag + sell_boost)
                effective_sell_lot = min(tokens, desired_sell_lot)

                effective_buy_gate = -(args.trend_gate + min(0.10, underweight_tokens * 0.20))
                effective_buy_thresh = max(0.09, args.threshold_core - min(0.09, underweight_tokens * 0.18) + (0.08 if (token_ratio > 0.75 and in_profit) else 0.0))
            else:
                cash_ratio = 1.0 - token_ratio
                cash_excess = max(0.0, cash_ratio - 0.25)
                effective_buy_gate = -(args.trend_gate + min(0.15, cash_excess * 0.6))
                effective_buy_thresh = max(0.08, args.threshold_core - min(0.12, cash_excess * 0.5))
                effective_buy_lot = base_lot
                effective_sell_lot = base_lot

            # 2. Локальные экстремумы в окне 30с
            w_win = max(0, t - 30)
            local_min = np.min(mid_prices[w_win : t + 1])
            local_max = np.max(mid_prices[w_win : t + 1])
            is_near_bottom = (mid_p <= local_min + 1.0 * tick_size)
            is_near_top = (mid_p >= local_max - 1.0 * tick_size)

            # Локальное поглощение продаж на дне (Absorption Bottom Bounce):
            recent_sell_vol = np.sum(sell_vols[max(0, t - 5) : t + 1])
            is_bottom_absorption = (is_near_bottom and recent_sell_vol > 0 and z5 > 0.08 and pb_12[t] > 0.15)

            # Снайпинг проливов по опережающему отскоку BigONE (Stage 1) — только ниже перегрева (rel_dev <= +1.0%):
            is_bo_dip_sniping = False
            if args.dip_sniper and bo_fresh and (rel_dev <= 0.010):
                is_bo_bounce = (ll60 > 0.8) or (ll5 > 0.8) or (bo_mid > mid_p + tick_size and z5 > 0.02)
                if is_bo_bounce and pb_12[t] > 0.15 and z12 > 0.05:
                    is_bo_dip_sniping = True

            # 3. Тренд-фильтр удержания прибыли (Lead-Lag & H3 Trend Hold)
            # Запрещаем ранний сброс монет на старте ралли, пока старший горизонт H=25с (топ-20% импульсов) или поводырь BigONE тянут цену вверх
            strong_bull_hold = (z25 > 0.55) or (bo_fresh and ((ll20 >= 1.0) or (ll60 > 0.6) or (bo_mid >= ask_p)))
            if strong_bull_hold:
                trend_holds += 1

            # Локальное истощение пампа на вершине (Peak Exhaustion) — активно ТОЛЬКО в плюсе (in_profit) выше EMA_30m (+1.2%):
            z5_prev = ez_5[t - 1] if t > 0 else z5
            is_peak_exhaustion = (
                is_near_top and in_profit and (rel_dev > 0.012) and not strong_bull_hold and
                z5 < z5_prev and z5 < 0.25 and ps_12[t] > 0.16
            )
            is_pump_accelerating = (z5 > 0.45) or (buy_vol > 0 and z5 > 0.20)

            # Откуп откатов после фиксации прибыли или при недоборе позиции на дне:
            is_pullback_rebuy = (
                is_near_bottom and (z5 > 0.03) and (pb_12[t] > 0.14) and (z12 > 0.02) and
                ((last_sell_anchor > 0.0 and mid_p <= last_sell_anchor - 5.0 * tick_size) or (token_ratio < 0.65 and rel_dev <= -0.002))
            )

            # А. Трендовый шлюз Scale 3 (H=25)
            allow_buy = (z25 >= effective_buy_gate) or is_bottom_absorption or is_bo_dip_sniping or is_pullback_rebuy
            allow_sell = ((z25 <= args.trend_gate) and not strong_bull_hold and (in_profit or rel_dev > 0.008 or z12 < -0.26)) or is_peak_exhaustion

            # 2. Защита от «откупа дороже продажи» (High-Water Mark Rebuy Guard)
            # Если недавно продали часть позиции, не откупаем обратно дороже цены продажи (кроме арбитражного пробоя BigONE)
            rebuy_ok = True
            if last_sell_anchor > 0.0 and (t - last_sell_t) < 3600 and token_ratio > 0.45:
                if bid_p > last_sell_anchor - 2.0 * tick_size and not (is_bo_dip_sniping and bo_mid > ask_p + tick_size):
                    rebuy_ok = False
                    rebuy_blocks += 1

            # Б. Согласование сигналов покупки (с жестким резервом кэша >= 14% при token_ratio > 0.86)
            can_buy_more = rebuy_ok and (token_ratio < 0.86) and (not deep_in_profit or is_pullback_rebuy) and not (token_ratio > 0.72 and in_profit and z12 < 0.35 and not is_pullback_rebuy)
            trigger_buy = allow_buy and can_buy_more and (
                (z12 > effective_buy_thresh and z5 > -0.05 and pb_12[t] > 0.18) or 
                is_bottom_absorption or 
                is_bo_dip_sniping or
                is_pullback_rebuy
            )

            pj_pen = float(args.penny_jump_penalty)
            if trigger_buy and effective_buy_lot >= 1.0:
                limit_p = round(bid_p, 4)
                q_ahead = float(bid_qtys[t])
                # Если спред широкий, котируем внутрь спреда, но с учетом штрафа конкуренции (Penny-Jumping)
                if ask_p - bid_p > 2 * tick_size:
                    limit_p = round(bid_p + tick_size, 4)
                    q_ahead = float(bid_qtys[t]) * pj_pen
                elif args.inventory_skew and (token_ratio < target_ratio) and (ask_p - bid_p > tick_size):
                    limit_p = round(bid_p + tick_size, 4)
                    q_ahead = float(bid_qtys[t]) * pj_pen

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

            # Условие продажи (включая ребалансировку в USDT только на настоящем перегреве > +2% к EMA_30m):
            is_profit_rebalance = (
                args.inventory_skew and in_profit and (rel_dev > 0.018) and not strong_bull_hold and
                (token_ratio > target_ratio + 0.06) and (z5 < 0.04) and (z12 < 0.02) and (ll5 < 0.3)
            )
            sell_threshold = max(0.08, args.threshold_core + max(0.0, z25 * args.q_token_bias) - (0.06 if (in_profit and rel_dev > 0.02 and token_ratio > target_ratio) else 0.0))
            trigger_sell = (allow_sell or is_profit_rebalance) and (
                (z12 < -sell_threshold and z5 < 0.05 and ps_12[t] > 0.18 and not strong_bull_hold) or 
                is_peak_exhaustion or
                is_profit_rebalance
            )

            if trigger_sell and tokens >= effective_sell_lot and effective_sell_lot >= 1.0:
                if is_pump_accelerating:
                    # Ask Escalator: памп разгоняется! Не отдаем монеты на взлете, ставим аск выше с премией
                    ticks_premium = max(2, min(8, int(round(z5 * 10))))
                    limit_p = round(ask_p + ticks_premium * tick_size, 4)
                    q_ahead = float(ask_qtys[t]) * pj_pen
                elif is_peak_exhaustion:
                    # На пике истощения продаем по лучшему аску (с учетом очереди/конкуренции)
                    limit_p = round(ask_p, 4)
                    q_ahead = float(ask_qtys[t]) * pj_pen
                elif args.ask_skew and z25 > 0.05:
                    ticks_premium = min(5, max(1, int(round(z25 * 10))))
                    limit_p = round(ask_p + ticks_premium * tick_size, 4)
                    q_ahead = float(ask_qtys[t]) * pj_pen
                elif ask_p - bid_p > 2 * tick_size:
                    limit_p = round(ask_p - tick_size, 4)
                    q_ahead = float(ask_qtys[t]) * pj_pen
                else:
                    limit_p = round(ask_p, 4)
                    q_ahead = float(ask_qtys[t])

                # Привязка Ask Escalator к стакану BigONE (Stage 1)
                if args.bo_ask_peg and bo_fresh and bo_ask > ask_p:
                    limit_p = round(max(limit_p, bo_ask - tick_size), 4)
                    q_ahead = float(ask_qtys[t]) * pj_pen

                if limit_p * effective_sell_lot >= args.min_order_val:
                    scale_label = 'PEAK_EXHAUSTION_SELL' if (is_peak_exhaustion or is_profit_rebalance) else ('ASK_ESCALATOR_SELL' if is_pump_accelerating else 'TRIAD_CORE_SELL')
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
    bot_buy_tokens = sum(tr['qty'] for tr in trades if 'BUY' in tr['side'])
    bot_sell_tokens = sum(tr['qty'] for tr in trades if 'SELL' in tr['side'])
    bot_total_tokens = bot_buy_tokens + bot_sell_tokens
    bot_total_usdt = sum(tr['qty'] * tr['price'] for tr in trades)
    st_market_vol = float(np.sum(buy_vols) + np.sum(sell_vols))
    st_market_usdt = float(np.sum((buy_vols + sell_vols) * mid_prices))
    market_share_pct = (bot_total_tokens / st_market_vol * 100.0) if st_market_vol > 0 else 0.0

    print("\n" + "=" * 105)
    print("🏆 РЕЗУЛЬТАТЫ СИНХРОНИЗИРОВАННОЙ ТРИАДЫ НА SAFETRADE (ИНВАРИАНТ Q_TOKEN)")
    print("=" * 105)
    final_token_ratio_pct = ((tokens * p_mid_final) / final_usd_total * 100.0) if final_usd_total > 0 else 0.0
    print(f"  • Стартовый капитал:  {initial_tokens:,.1f} {coin.upper()} + {initial_usdt:,.2f} USDT (Всего в монетах: {initial_q_total:,.2f})")
    print(f"  • Итоговый капитал:   {tokens:,.1f} {coin.upper()} + {usdt:,.2f} USDT (Доля монеты: {final_token_ratio_pct:.1f}% | Всего в монетах: {final_q_total:,.2f})")
    print(f"  • Стартовая цена mid: ${p_mid_0:.4f}  ➔  Итоговая цена mid: ${p_mid_final:.4f} (Движение рынка: {hodl_price_pct:+.2f}% | Средняя цена закупа: ${avg_buy_price:.4f})")
    print("-" * 105)
    print(f"  🎯 ЧИСТЫЙ ПРИРОСТ МОНЕТ (Q_token PnL): {delta_q:+,.2f} {coin.upper()} ({delta_q_pct:+.2f}%) 🚀")
    print(f"  💵 Прирост портфеля в USD:             {delta_usd:+,.2f} USDT ({delta_usd_pct:+.2f}%)")
    print(f"  🛡️ Максимальная просадка по монетам:  {max_dd_q:.2f}%")
    print(f"  📊 Оборот биржи SafeTrade за период:   {st_market_vol:,.1f} {coin.upper()} (~${st_market_usdt:,.0f} USDT | Buy: {np.sum(buy_vols):,.1f} | Sell: {np.sum(sell_vols):,.1f})")
    print(f"  🔄 Оборот нашего бота за период:       {bot_total_tokens:,.1f} {coin.upper()} (~${bot_total_usdt:,.0f} USDT | Куплено: {bot_buy_tokens:,.1f} | Продано: {bot_sell_tokens:,.1f} | Доля рынка: {market_share_pct:.2f}%)")
    print(f"  ⚡ Исполнено Maker-сделок:             {total_trades:,} (Отмен: {cancels:,} | Rebuy-защит: {rebuy_blocks:,} | Trend-Hold: {trend_holds:,} | Lead-Lag: {leadlag_shields:,})")
    print(f"  💰 Сэкономлено на Maker комиссиях:     ${saved_fees_usdt:.2f} USDT")
    if getattr(args, 'plot', None):
        from _tools.triad_utils import save_backtest_plot
        save_backtest_plot(
            args.plot, coin, val_df, mid_prices, ema_30m, trades,
            q_total_history, usd_total_history, initial_q_total, initial_usd_total,
            delta_q_pct, delta_usd_pct, drawdowns_q, max_dd_q
        )

    return {
        'coin': coin.upper(),
        'delta_q_tokens': delta_q,
        'delta_q_pct': delta_q_pct,
        'max_dd_q_pct': max_dd_q,
        'total_trades': total_trades,
        'cancels': cancels,
        'trend_blocks': trend_blocks,
        'leadlag_shields': leadlag_shields,
        'saved_fees_usdt': saved_fees_usdt,
        'trades': trades
    }


if __name__ == '__main__':
    args = parse_args()
    run_triad_simulation(args)
