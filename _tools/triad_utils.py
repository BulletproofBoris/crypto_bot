#!/usr/bin/env python3
"""
Вспомогательные утилиты для мультигоризонтного диспетчера триады (triad_dispatcher.py):
- фильтрация технических обрывов связи на OOS-срезе;
- расчёт и печать сводных метрик Q_token / USD / просадки;
- отрисовка графика бэктеста.
"""

import os
import numpy as np
import pandas as pd


def filter_outage_intervals(val_df: pd.DataFrame, max_flat_sec: int = 1800) -> pd.DataFrame:
    """Исключает неактуальные интервалы обрыва связи (неизменная цена > max_flat_sec)."""
    active_mask = (
        (val_df['ST_buy_vol'].fillna(0) > 0) |
        (val_df['ST_sell_vol'].fillna(0) > 0) |
        (val_df['ST_bid_1_price'].diff().abs().fillna(0) > 1e-8) |
        (val_df['ST_ask_1_price'].diff().abs().fillna(0) > 1e-8)
    )
    active_idx = np.where(active_mask)[0]
    if len(active_idx) <= 1:
        return val_df

    keep_mask = np.ones(len(val_df), dtype=bool)
    gaps = np.diff(active_idx)
    outage_pos = np.where(gaps > max_flat_sec)[0]
    outage_ranges = []
    for pos in outage_pos:
        g_start = int(active_idx[pos]) + 1
        g_end = int(active_idx[pos + 1])
        keep_mask[g_start:g_end] = False
        outage_ranges.append(
            f"{val_df['timestamp'].iloc[g_start]} ➔ {val_df['timestamp'].iloc[g_end - 1]} ({g_end - g_start:,} с)"
        )
    if len(val_df) - 1 - int(active_idx[-1]) > max_flat_sec:
        g_start = int(active_idx[-1]) + 1
        keep_mask[g_start:] = False
        outage_ranges.append(
            f"{val_df['timestamp'].iloc[g_start]} ➔ {val_df['timestamp'].iloc[-1]} ({len(val_df) - g_start:,} с)"
        )
    if outage_ranges:
        removed_rows = int(np.sum(~keep_mask))
        val_df = val_df[keep_mask].copy().reset_index(drop=True)
        print(
            f"✂️ Исключены отрезки обрыва связи: удалено {removed_rows:,} с "
            f"({'; '.join(outage_ranges)}), оставлено {len(val_df):,} с живого рынка"
        )
    return val_df


def save_backtest_plot(
    plot_path: str,
    coin: str,
    val_df: pd.DataFrame,
    mid_prices: np.ndarray,
    ema_30m: np.ndarray,
    trades: list,
    q_total_history: list,
    usd_total_history: list,
    initial_q_total: float,
    initial_usd_total: float,
    delta_q_pct: float,
    delta_usd_pct: float,
    drawdowns_q: np.ndarray,
    max_dd_q: float,
):
    """Сохраняет 3-панельный PNG-график бэктеста триады."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2, 1.0]}
    )
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.grid(True, color='#30363d', linestyle='--', alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    ts_series = pd.to_datetime(val_df['timestamp'])
    ax1.plot(ts_series, mid_prices, label=f'{coin.upper()} Mid Price', color='#58a6ff', lw=1.5, zorder=2)
    ax1.plot(ts_series, ema_30m, label='EMA 30m (Regime Anchor)', color='#a371f7', linestyle='--', lw=1.2, alpha=0.85, zorder=2)

    if trades:
        df_tr = pd.DataFrame(trades)
        df_tr['time'] = ts_series.iloc[df_tr['t']].values
        groups = [
            ('BUY_MAKER', ['BO_LEADLAG_SNIPER_BUY'], True, '#38d9a9', '^', 90, 1.0, '⚡ BigONE Sniper Buy', 6),
            ('BUY_MAKER', ['SNIPER_DIP_BUY'], True, '#2ea043', '^', 80, 1.0, 'Absorption Dip Buy', 5),
            ('BUY_MAKER', ['BO_LEADLAG_SNIPER_BUY', 'SNIPER_DIP_BUY'], False, '#3fb950', '^', 45, 0.7, 'Core Buy', 4),
            ('SELL_MAKER', ['PEAK_EXHAUSTION_SELL'], True, '#f85149', 'v', 80, 1.0, 'Peak Exhaustion Sell', 5),
            ('SELL_MAKER', ['ASK_ESCALATOR_SELL'], True, '#d29922', 'v', 55, 0.8, 'Ask Escalator', 4),
            ('SELL_MAKER', ['PEAK_EXHAUSTION_SELL', 'ASK_ESCALATOR_SELL'], False, '#da3633', 'v', 35, 0.6, 'Core Sell', 3),
        ]
        for side, scales, include, color, marker, sz, alpha, lbl, zord in groups:
            mask = (df_tr['side'] == side) & (df_tr['scale'].isin(scales) if include else ~df_tr['scale'].isin(scales))
            sub = df_tr[mask]
            if not sub.empty:
                ec = 'white' if alpha == 1.0 else 'none'
                ax1.scatter(sub['time'], sub['price'], color=color, marker=marker, s=sz, alpha=alpha, edgecolors=ec, lw=1.0, label=f'{lbl} ({len(sub)})', zorder=zord)

    ax1.set_ylabel('Price (USDT)', color='#c9d1d9', fontsize=11)
    ax1.set_title(f'STRICT OUT-OF-SAMPLE BACKTEST — {coin.upper()} / USDT (Maker 0% Fee, RTT=1s)', color='#f0f6fc', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8.5)

    q_pnl_pct_arr = ((np.array(q_total_history) - initial_q_total) / initial_q_total) * 100.0
    usd_pnl_pct_arr = ((np.array(usd_total_history) - initial_usd_total) / initial_usd_total) * 100.0
    ax2.plot(ts_series, usd_pnl_pct_arr, color='#3fb950', lw=1.8, label=f'USD Portfolio Equity ({delta_usd_pct:+.2f}%)', zorder=4)
    ax2.plot(ts_series, q_pnl_pct_arr, color='#e3b341', lw=1.6, label=f'Accumulated {coin.upper()} Q_token ({delta_q_pct:+.2f}%)', zorder=3)
    ax2.axhline(0, color='#8b949e', linestyle=':', lw=1, alpha=0.7)
    ax2.set_ylabel('PnL (%)', color='#c9d1d9', fontsize=11)
    ax2.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)

    ax3.fill_between(ts_series, drawdowns_q, 0, color='#f85149', alpha=0.35, label=f'Token Drawdown (Max: {max_dd_q:.2f}%)')
    ax3.plot(ts_series, drawdowns_q, color='#f85149', lw=1.0)
    ax3.set_ylabel('Drawdown (%)', color='#c9d1d9', fontsize=11)
    ax3.set_xlabel('Время', color='#c9d1d9', fontsize=11)
    ax3.legend(loc='lower left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.savefig(plot_path, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"📊 График бэктеста успешно сохранен: {plot_path}")
    plt.close()


def load_strict_oos_slice(labeled_path, dispatcher, exec_cols: list = None, strict_oos: bool = True, keep_outage: bool = False, max_steps: int = None) -> pd.DataFrame:
    """Читает Parquet-файл и выделяет строгий OOS-срез после максимальной даты обучения всех экспертов триады."""
    import pyarrow.parquet as pq

    if exec_cols is None:
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

    schema_cols = set(pq.read_schema(labeled_path).names)
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
        val_df = df.iloc[int(len(df) * 0.8):].copy().reset_index(drop=True)

    if strict_oos:
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

    if not keep_outage:
        val_df = filter_outage_intervals(val_df)

    if max_steps:
        val_df = val_df.iloc[:max_steps].copy()

    return val_df
