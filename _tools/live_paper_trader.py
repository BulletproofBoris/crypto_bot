#!/usr/bin/env python3
"""
Онлайн Paper-Trading демон (Canary Live Trader) для биржи SafeTrade.
Работает круглосуточно на VPS, обрабатывает реальные секундные срезы стакана и ленты сделок,
рассчитывает микроструктурный банк фильтров в RAM и исполняет виртуальные Maker-заявки
с полным учетом очереди (Q_ahead), проскальзывания, задержки RTT и трендовых шлюзов триады.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import deque
import numpy as np
import pandas as pd

# Добавляем корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.swarm_router import SwarmRegimeRouter
from _tools.microstructure_features import add_exchange_filter_bank, add_cross_exchange_and_macro_filters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [LIVE-TRADER] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("live_paper_trader")

MSK = timezone(timedelta(hours=3))

COIN_CONFIG = {
    'prl': {
        'tick_size': 0.005,
        'trade_size': 10.0,
        'initial_tokens': 10000.0,
        'initial_usdt': 100.0,
        'st_dir': 'prlusdt',
        'bo_dir': 'prlusdt_bigone',
    },
    'quantus': {
        'tick_size': 0.05,
        'trade_size': 1.0,
        'initial_tokens': 1000.0,
        'initial_usdt': 100.0,
        'st_dir': 'quantususdt',
        'bo_dir': 'quantus_bigone',
    }
}


class LiveCoinPaperTrader:
    def __init__(self, coin: str, base_data_dir: Path):
        self.coin = coin.lower()
        self.cfg = COIN_CONFIG[self.coin]
        self.tick_size = self.cfg['tick_size']
        self.trade_size = self.cfg['trade_size']
        self.base_data_dir = base_data_dir

        logger.info(f"Инициализация триады масштабов для {self.coin.upper()}...")
        self.router_5 = SwarmRegimeRouter(self.coin, H=5, K=3.0)
        self.router_12 = SwarmRegimeRouter(self.coin, H=12, K=3.5)
        self.router_25 = SwarmRegimeRouter(self.coin, H=25, K=3.0)

        # Портфель
        self.tokens = self.cfg['initial_tokens']
        self.usdt = self.cfg['initial_usdt']
        self.initial_q_total = None
        self.max_q_total = None
        self.max_drawdown = 0.0

        # Активный виртуальный ордер
        self.active_order = None
        self.maker_fills = 0
        self.cancels = 0
        self.trend_blocks = 0
        self.saved_fees_usdt = 0.0
        self.last_trade_id = None

        # Кольцевой буфер секундных баров для расчета 300с фильтров
        self.history_buffer = deque(maxlen=360)
        self.last_z5 = 0.0

    def parse_last_depth(self, filepath: Path):
        """Быстрое чтение последней валидной строки стакана через seek с конца файла."""
        if not filepath.exists() or filepath.stat().st_size == 0:
            return None
        try:
            with open(filepath, 'rb') as f:
                f.seek(max(0, filepath.stat().st_size - 4096))
                lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
                if not lines:
                    return None
                last_line = lines[-1].split(',')
                if len(last_line) < 103:
                    return None

                raw_ts = last_line[0]
                if len(raw_ts) == 13 and raw_ts.isdigit():
                    ts = datetime.fromtimestamp(int(raw_ts) / 1000.0, tz=timezone.utc).astimezone(MSK)
                else:
                    ts = pd.to_datetime(raw_ts).tz_convert(MSK)

                d = {'timestamp': ts}
                # Bids 1..10
                for i in range(1, 11):
                    d[f'bid_{i}_price'] = float(last_line[1 + (i - 1) * 2])
                    d[f'bid_{i}_qty'] = float(last_line[2 + (i - 1) * 2])
                # Asks 1..10
                for i in range(1, 11):
                    d[f'ask_{i}_price'] = float(last_line[101 + (i - 1) * 2])
                    d[f'ask_{i}_qty'] = float(last_line[102 + (i - 1) * 2])

                b1_p = d['bid_1_price']
                b1_q = d['bid_1_qty']
                a1_p = d['ask_1_price']
                a1_q = d['ask_1_qty']
                mid_p = (b1_p + a1_p) / 2.0

                d['mid_price'] = mid_p
                d['spread'] = a1_p - b1_p
                d['spread_pct'] = (a1_p - b1_p) / mid_p if mid_p > 0 else 0.0
                d['bids_vol_top5'] = sum(d[f'bid_{i}_qty'] for i in range(1, 6))
                d['asks_vol_top5'] = sum(d[f'ask_{i}_qty'] for i in range(1, 6))
                d['bids_usd_top5'] = sum(d[f'bid_{i}_price'] * d[f'bid_{i}_qty'] for i in range(1, 6))
                d['asks_usd_top5'] = sum(d[f'ask_{i}_price'] * d[f'ask_{i}_qty'] for i in range(1, 6))
                tot5 = d['bids_vol_top5'] + d['asks_vol_top5']
                d['imbalance_top5'] = (d['bids_vol_top5'] - d['asks_vol_top5']) / tot5 if tot5 > 0 else 0.0
                d['bids_vol_11_50'] = 0.0

                return d
        except Exception as e:
            logger.debug(f"Ошибка парсинга depth {filepath.name}: {e}")
            return None

    def parse_recent_trades(self, filepath: Path, since_id=None):
        """Чтение свежих сделок SafeTrade с момента since_id."""
        if not filepath.exists() or filepath.stat().st_size == 0:
            return 0.0, 0.0, 0.0, since_id
        try:
            with open(filepath, 'rb') as f:
                f.seek(max(0, filepath.stat().st_size - 8192))
                lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
                if not lines:
                    return 0.0, 0.0, 0.0, since_id

                buy_vol = 0.0
                sell_vol = 0.0
                new_latest_id = since_id

                for line in lines[-50:]:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            tid = int(parts[0]) if parts[0].isdigit() else parts[0]
                            if since_id is not None and str(tid) <= str(since_id):
                                continue
                            price = float(parts[1])
                            amount = float(parts[2])
                            side = str(parts[3]).strip().lower()
                            if side in ('buy', 'ask'):
                                buy_vol += amount
                            else:
                                sell_vol += amount
                            new_latest_id = tid
                        except Exception:
                            continue

                trade_vol = buy_vol + sell_vol
                return trade_vol, buy_vol, sell_vol, trade_count, new_latest_id
        except Exception as e:
            logger.debug(f"Ошибка парсинга trades: {e}")
            return 0.0, 0.0, 0.0, 0, since_id

    def parse_last_macro(self, filepath: Path):
        """Чтение последнего среза макро-поводырей Binance (BTC, NEAR)."""
        if not filepath.exists() or filepath.stat().st_size == 0:
            return None
        try:
            with open(filepath, 'rb') as f:
                f.seek(max(0, filepath.stat().st_size - 1024))
                lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
                if not lines:
                    return None
                parts = lines[-1].split(',')
                if len(parts) >= 17:
                    return {
                        'macro_btc_mid': float(parts[3]),
                        'macro_near_mid': float(parts[7]),
                        'macro_btc_bid_qty': float(parts[9]),
                        'macro_btc_ask_qty': float(parts[10]),
                        'macro_btc_buy_vol': float(parts[11]),
                        'macro_btc_sell_vol': float(parts[12]),
                        'macro_near_bid_qty': float(parts[13]),
                        'macro_near_ask_qty': float(parts[14]),
                        'macro_near_buy_vol': float(parts[15]),
                        'macro_near_sell_vol': float(parts[16]),
                    }
        except Exception:
            return None

    def step(self):
        """Один такт (1 секунда) онлайн-инференса и котирования."""
        now_ts = datetime.now(tz=MSK)

        st_depth_f = self.base_data_dir / "realtime" / self.cfg['st_dir'] / "depth.csv"
        st_trades_f = self.base_data_dir / "realtime" / self.cfg['st_dir'] / "trades.csv"
        bo_depth_f = self.base_data_dir / "realtime" / self.cfg['bo_dir'] / "depth_bigone.csv"
        macro_f = self.base_data_dir / "realtime" / "macro_binance" / "binance_macro.csv"

        st_data = self.parse_last_depth(st_depth_f)
        if not st_data:
            return None

        trade_vol, buy_vol, sell_vol, trade_count, self.last_trade_id = self.parse_recent_trades(st_trades_f, self.last_trade_id)
        bo_data = self.parse_last_depth(bo_depth_f)
        macro_data = self.parse_last_macro(macro_f)

        row = {'timestamp': now_ts}
        for k, v in st_data.items():
            if k != 'timestamp':
                row[f"ST_{k}"] = v

        row['ST_trade_vol'] = trade_vol
        row['ST_trade_count'] = float(trade_count)
        row['ST_buy_vol'] = buy_vol
        row['ST_sell_vol'] = sell_vol
        row['ST_delta_vol'] = buy_vol - sell_vol

        if bo_data:
            for k, v in bo_data.items():
                if k != 'timestamp':
                    row[f"BO_{k}"] = v
            row['BO_trade_vol'] = 0.0
            row['BO_trade_count'] = 0.0
            row['BO_buy_vol'] = 0.0
            row['BO_sell_vol'] = 0.0
            row['BO_delta_vol'] = 0.0

        if macro_data:
            for k, v in macro_data.items():
                row[k] = v

        self.history_buffer.append(row)
        if len(self.history_buffer) < 10:
            return None

        # Формируем DataFrame скользящего окна
        df_win = pd.DataFrame(list(self.history_buffer))
        df_win.set_index('timestamp', inplace=True)
        df_win = add_exchange_filter_bank(df_win, 'ST', coin=self.coin)
        if bo_data:
            df_win = add_exchange_filter_bank(df_win, 'BO', coin=self.coin)
        df_win = add_cross_exchange_and_macro_filters(df_win, coin=self.coin)
        df_win = df_win.fillna(0.0)

        # Инференс триады
        last_df = df_win.iloc[[-1]]

        def get_pred(router):
            X = np.nan_to_num(last_df.reindex(columns=router.feature_names, fill_value=0.0).values, 0.0)
            return router.predict_ensemble(X, last_df, mode='router')

        pred_5 = get_pred(self.router_5)
        pred_12 = get_pred(self.router_12)
        pred_25 = get_pred(self.router_25)

        z5 = float(pred_5['expected_z'][0])
        z12 = float(pred_12['expected_z'][0])
        z25 = float(pred_25['expected_z'][0])
        pb_12 = float(pred_12['p_buy'][0])
        ps_12 = float(pred_12['p_sell'][0])

        # Симуляция исполнения
        bid_p = st_data['bid_1_price']
        ask_p = st_data['ask_1_price']
        mid_p = st_data['mid_price']
        tick_size = self.tick_size
        trade_size = self.trade_size
        now_sec = time.time()

        if self.initial_q_total is None:
            self.initial_q_total = self.tokens + (self.usdt / mid_p)
            self.max_q_total = self.initial_q_total

        # 3. Локальные экстремумы в окне 30с
        recent_mids = [r['ST_mid_price'] for r in list(self.history_buffer)[-30:] if 'ST_mid_price' in r]
        local_min = min(recent_mids) if recent_mids else mid_p
        local_max = max(recent_mids) if recent_mids else mid_p
        is_near_bottom = (mid_p <= local_min + 1.0 * tick_size)
        is_near_top = (mid_p >= local_max - 1.0 * tick_size)

        # Локальное поглощение продаж на дне (Absorption Bottom Bounce):
        recent_sell_vol = sum(r.get('ST_sell_vol', 0.0) for r in list(self.history_buffer)[-5:])
        is_bottom_absorption = (is_near_bottom and recent_sell_vol > 0 and z5 > 0.08 and pb_12 > 0.15)

        # Локальное истощение пампа на вершине (Peak Exhaustion):
        is_peak_exhaustion = (is_near_top and z5 < self.last_z5 and z5 < 0.15 and ps_12 > 0.15)
        is_pump_accelerating = (z5 > 0.20) or (buy_vol > 0 and z5 > 0.05)

        # 1. Проверка существующего ордера
        if self.active_order is not None:
            o = self.active_order
            is_active = (now_sec >= o['active_after'])

            if not o['canceling']:
                if o['side'] == 'buy' and z25 < -0.15 and not is_bottom_absorption:
                    o['canceling'] = True
                    o['cancel_after'] = now_sec + 1.0
                    self.trend_blocks += 1
                elif o['side'] == 'sell' and (z25 > 0.15 or z5 > 0.18 or (buy_vol > 0 and o['price'] <= ask_p)):
                    o['canceling'] = True
                    o['cancel_after'] = now_sec + 1.0
                    self.trend_blocks += 1
                elif now_sec - o['placed_at'] > 60.0:
                    o['canceling'] = True
                    o['cancel_after'] = now_sec + 1.0

            order_done = False
            if is_active and o['rem_qty'] > 0:
                fill_qty = 0.0
                if o['side'] == 'buy':
                    if ask_p <= o['price']:
                        avail = max(st_data['ask_1_qty'], sell_vol)
                        fill_qty = min(o['rem_qty'], avail)
                    elif sell_vol > 0 and bid_p <= o['price']:
                        if o['price'] == bid_p:
                            if sell_vol <= o['q_ahead']:
                                o['q_ahead'] -= sell_vol
                            else:
                                avail = sell_vol - o['q_ahead']
                                o['q_ahead'] = 0.0
                                fill_qty = min(o['rem_qty'], avail)
                        elif o['price'] > bid_p:
                            fill_qty = min(o['rem_qty'], sell_vol)

                    if fill_qty > 0:
                        cost = fill_qty * o['price']
                        if self.usdt >= cost:
                            self.usdt -= cost
                            self.tokens += fill_qty
                            o['rem_qty'] -= fill_qty
                            self.maker_fills += 1
                            self.saved_fees_usdt += (cost * 0.002)
                            logger.info(f"🟢 [FILL BUY] {self.coin.upper()}: {fill_qty:.2f} @ {o['price']} (Tokens: {self.tokens:.2f}, USDT: {self.usdt:.2f})")
                            if o['rem_qty'] <= 1e-5:
                                order_done = True

                elif o['side'] == 'sell':
                    if bid_p >= o['price']:
                        avail = max(st_data['bid_1_qty'], buy_vol)
                        fill_qty = min(o['rem_qty'], avail)
                    elif buy_vol > 0 and ask_p >= o['price']:
                        if o['price'] == ask_p:
                            if buy_vol <= o['q_ahead']:
                                o['q_ahead'] -= buy_vol
                            else:
                                avail = buy_vol - o['q_ahead']
                                o['q_ahead'] = 0.0
                                fill_qty = min(o['rem_qty'], avail)
                        elif o['price'] < ask_p:
                            fill_qty = min(o['rem_qty'], buy_vol)

                    if fill_qty > 0:
                        rev = fill_qty * o['price']
                        if self.tokens >= fill_qty:
                            self.tokens -= fill_qty
                            self.usdt += rev
                            o['rem_qty'] -= fill_qty
                            self.maker_fills += 1
                            self.saved_fees_usdt += (rev * 0.002)
                            logger.info(f"🔴 [FILL SELL] {self.coin.upper()}: {fill_qty:.2f} @ {o['price']} (Tokens: {self.tokens:.2f}, USDT: {self.usdt:.2f})")
                            if o['rem_qty'] <= 1e-5:
                                order_done = True

            if order_done:
                self.active_order = None
            elif o['canceling'] and now_sec >= o['cancel_after']:
                self.active_order = None
                self.cancels += 1

        # 2. Выставление новых ордеров
        elif self.active_order is None:
            # 1. Inventory Rebalancing Pull:
            total_equity_usd = self.usdt + (self.tokens * mid_p)
            cash_ratio = self.usdt / total_equity_usd if total_equity_usd > 1e-4 else 0.0
            cash_excess = max(0.0, cash_ratio - 0.25)

            effective_buy_gate = -(0.15 + min(0.15, cash_excess * 0.6))
            effective_buy_thresh = max(0.08, 0.20 - min(0.12, cash_excess * 0.5))

            # 2. Conviction Sizing:
            effective_lot = trade_size
            if abs(z12) > 0.35 and abs(z5) > 0.25:
                effective_lot = min(trade_size * 2.0, self.tokens if z12 < 0 else self.usdt / max(bid_p, 1e-4))

            # А. Трендовый шлюз Scale 3 (H=25)
            allow_buy = (z25 >= effective_buy_gate) or is_bottom_absorption
            allow_sell = (z25 <= 0.15) or is_peak_exhaustion

            # Б. Согласование сигналов покупки
            trigger_buy = allow_buy and ((z12 > effective_buy_thresh and z5 > -0.05 and pb_12 > 0.18) or is_bottom_absorption)

            if trigger_buy:
                limit_p = round(bid_p, 4)
                q_ahead = float(st_data['bid_1_qty'])
                if ask_p - bid_p > 2 * tick_size:
                    limit_p = round(bid_p + tick_size, 4)
                    q_ahead = 0.0

                if limit_p * effective_lot >= 0.5 and self.usdt >= (effective_lot * limit_p):
                    scale_label = 'SNIPER_DIP_BUY' if is_bottom_absorption else 'TRIAD_CORE_BUY'
                    self.active_order = {
                        'side': 'buy',
                        'price': limit_p,
                        'qty': effective_lot,
                        'rem_qty': effective_lot,
                        'placed_at': now_sec,
                        'active_after': now_sec + 1.0,
                        'q_ahead': q_ahead,
                        'canceling': False,
                        'cancel_after': None,
                        'scale': scale_label
                    }

            # Условие продажи:
            sell_thresh = 0.20 + max(0.0, z25 * 0.10)
            trigger_sell = allow_sell and ((z12 < -sell_thresh and z5 < 0.05 and ps_12 > 0.18) or is_peak_exhaustion)

            if trigger_sell and self.tokens >= effective_lot:
                if is_pump_accelerating:
                    # Ask Escalator: памп разгоняется! Не отдаем монеты на взлете, ставим аск выше с премией
                    ticks_premium = max(2, min(8, int(round(z5 * 10))))
                    limit_p = round(ask_p + ticks_premium * tick_size, 4)
                    q_ahead = 0.0
                elif is_peak_exhaustion:
                    # На пике истощения продаем прямо по лучшему аску
                    limit_p = round(ask_p, 4)
                    q_ahead = 0.0
                elif z25 > 0.05:
                    ticks_premium = min(5, max(1, int(round(z25 * 10))))
                    limit_p = round(ask_p + ticks_premium * tick_size, 4)
                    q_ahead = 0.0
                elif ask_p - bid_p > 2 * tick_size:
                    limit_p = round(ask_p - tick_size, 4)
                    q_ahead = 0.0
                else:
                    limit_p = round(ask_p, 4)
                    q_ahead = float(st_data['ask_1_qty'])

                if limit_p * effective_lot >= 0.5:
                    scale_label = 'PEAK_EXHAUSTION_SELL' if is_peak_exhaustion else ('ASK_ESCALATOR_SELL' if is_pump_accelerating else 'TRIAD_CORE_SELL')
                    self.active_order = {
                        'side': 'sell',
                        'price': limit_p,
                        'qty': effective_lot,
                        'rem_qty': effective_lot,
                        'placed_at': now_sec,
                        'active_after': now_sec + 1.0,
                        'q_ahead': q_ahead,
                        'canceling': False,
                        'cancel_after': None,
                        'scale': scale_label
                    }

        self.last_z5 = z5

        # Метрики
        q_total = self.tokens + (self.usdt / mid_p)
        if q_total > self.max_q_total:
            self.max_q_total = q_total
        dd = ((q_total - self.max_q_total) / self.max_q_total) * 100.0
        if dd < self.max_drawdown:
            self.max_drawdown = dd
        pnl_pct = ((q_total - self.initial_q_total) / self.initial_q_total) * 100.0

        return {
            'coin': self.coin.upper(),
            'timestamp': now_ts.strftime('%Y-%m-%d %H:%M:%S'),
            'mid_price': round(mid_p, 4),
            'spread': round(st_data['spread'], 4),
            'z5': round(z5, 3),
            'z12': round(z12, 3),
            'z25': round(z25, 3),
            'active_order': f"{self.active_order['side'].upper()} {self.active_order['rem_qty']} @ {self.active_order['price']}" if self.active_order else "IDLE",
            'tokens': round(self.tokens, 2),
            'usdt': round(self.usdt, 2),
            'q_total': round(q_total, 2),
            'pnl_pct': round(pnl_pct, 3),
            'max_dd_pct': round(self.max_drawdown, 2),
            'fills': self.maker_fills,
            'cancels': self.cancels,
            'trend_blocks': self.trend_blocks,
            'saved_fees': round(self.saved_fees_usdt, 2)
        }


def run_daemon():
    logger.info("🚀 Запуск Online Paper-Trading Демона SafeTrade (PRL + QUANTUS)...")
    base_data = Path("/opt/crypto_bot/data") if Path("/opt/crypto_bot/data").exists() else BASE_DIR / "data"
    status_file = base_data / "paper_trading_status.json"

    trader_prl = LiveCoinPaperTrader('prl', base_data)
    trader_quantus = LiveCoinPaperTrader('quantus', base_data)

    loop_count = 0
    while True:
        try:
            t0 = time.time()
            res_prl = trader_prl.step()
            res_qnt = trader_quantus.step()

            loop_count += 1
            if loop_count % 5 == 0 and res_prl and res_qnt:
                status_payload = {
                    'updated_at': datetime.now(tz=MSK).strftime('%Y-%m-%d %H:%M:%S'),
                    'prl': res_prl,
                    'quantus': res_qnt
                }
                with open(status_file, 'w') as f:
                    json.dump(status_payload, f, indent=2)

                logger.info(
                    f"📊 [PRL: ${res_prl['mid_price']} | Q_pnl: {res_prl['pnl_pct']:+5.2f}% | Fills: {res_prl['fills']} | Order: {res_prl['active_order']}] "
                    f"[QNT: ${res_qnt['mid_price']} | Q_pnl: {res_qnt['pnl_pct']:+5.2f}% | Fills: {res_qnt['fills']} | Order: {res_qnt['active_order']}]"
                )

            elapsed = time.time() - t0
            time.sleep(max(0.1, 1.0 - elapsed))
        except KeyboardInterrupt:
            logger.info("Остановка Paper-Trading демона пользователем.")
            break
        except Exception as e:
            logger.error(f"Непредвиденная ошибка в цикле демона: {e}", exc_info=True)
            time.sleep(1.0)


if __name__ == '__main__':
    run_daemon()
