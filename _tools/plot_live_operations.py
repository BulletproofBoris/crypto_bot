#!/usr/bin/env python3
"""
Визуализация операций торгового бота в реальном времени.
Скрипт загружает актуальный журнал paper_trader.log с VPS-сервера,
парсит секундные срезы, активные ордера и исполненные сделки (Fills),
и строит графики динамики цены, зон котирования и накопления токенов.
"""

import os
import sys
import re
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Настройка путей
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)

import vps_config

def fetch_telemetry_from_vps():
    """Загружает paper_trader.log и paper_trading_status.json с сервера."""
    print("📡 Подключение к VPS (81.19.137.18) для загрузки логов онлайн-котирования...")
    try:
        ssh = vps_config.create_ssh_client(timeout=10.0)
        
        # 1. Загрузка логов
        stdin, stdout, stderr = ssh.exec_command("cat /opt/crypto_bot/data/logs/paper_trader.log 2>/dev/null")
        log_content = stdout.read().decode("utf-8", errors="replace")
        
        # 2. Загрузка текущего JSON статуса
        stdin, stdout, stderr = ssh.exec_command("cat /opt/crypto_bot/data/paper_trading_status.json 2>/dev/null")
        status_content = stdout.read().decode("utf-8", errors="replace")
        
        ssh.close()
        print(f"✅ Лог загружен: {len(log_content.splitlines())} строк.")
        return log_content, status_content
    except Exception as e:
        print(f"⚠️ Не удалось загрузить логи с VPS ({e}). Торговый демон на сервере остановлен или отсутствует связь.")
        return "", ""

def parse_telemetry(log_text):
    """
    Парсит журнал на временные ряды котировок и сделок.
    """
    tick_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[INFO\] \[LIVE-TRADER\] 📊 \[PRL: \$([\d\.]+) \| Q_pnl: ([+\-]?[\d\.]+)% \| Fills: (\d+) \| Order: ([^\]]+)\] \[QNT: \$([\d\.]+) \| Q_pnl: ([+\-]?[\d\.]+)% \| Fills: (\d+) \| Order: ([^\]]+)\]"
    )
    fill_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[INFO\] \[LIVE-TRADER\] ([🟢🔴]) \[FILL (BUY|SELL)\] (PRL|QUANTUS): ([\d\.]+) @ ([\d\.]+) \(Tokens: ([\d\.]+), USDT: ([\d\.]+)\)"
    )

    ticks = []
    fills = []

    for line in log_text.splitlines():
        tm = tick_re.match(line)
        if tm:
            t, p_p, p_pnl, p_f, p_o, q_p, q_pnl, q_f, q_o = tm.groups()
            ticks.append({
                "timestamp": pd.to_datetime(t),
                "prl_price": float(p_p),
                "prl_pnl": float(p_pnl),
                "prl_fills": int(p_f),
                "prl_order": p_o.strip(),
                "qnt_price": float(q_p),
                "qnt_pnl": float(q_pnl),
                "qnt_fills": int(q_f),
                "qnt_order": q_o.strip(),
            })
        fm = fill_re.match(line)
        if fm:
            t, icon, side, coin, qty, price, tokens, usdt = fm.groups()
            fills.append({
                "timestamp": pd.to_datetime(t),
                "coin": coin.upper(),
                "side": side.upper(),
                "qty": float(qty),
                "price": float(price),
                "tokens": float(tokens),
                "usdt": float(usdt),
                "icon": icon
            })

    df_ticks = pd.DataFrame(ticks)
    df_fills = pd.DataFrame(fills)
    return df_ticks, df_fills

def parse_order_price(order_str):
    """Извлекает цену и сторону из строки вида 'BUY 10.0 @ 1.865'."""
    if not order_str or order_str == "IDLE":
        return None, None
    parts = order_str.split("@")
    if len(parts) == 2:
        try:
            side = parts[0].strip().split()[0]
            price = float(parts[1].strip())
            return side, price
        except Exception:
            return None, None
    return None, None

def plot_matplotlib(df_ticks, df_fills, coin='prl', output_png=None):
    """Строит двухпанельный график в Matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    coin_key = coin.lower()
    coin_name = "PRL" if coin_key == "prl" else "QUANTUS"
    price_col = f"{'prl' if coin_key == 'prl' else 'qnt'}_price"
    pnl_col = f"{'prl' if coin_key == 'prl' else 'qnt'}_pnl"
    order_col = f"{'prl' if coin_key == 'prl' else 'qnt'}_order"

    coin_fills = df_fills[df_fills["coin"] == coin_name].copy() if not df_fills.empty else pd.DataFrame()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.grid(True, color='#30363d', linestyle='--', alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Панель 1: Динамика цены и ордеров
    ax1.plot(df_ticks['timestamp'], df_ticks[price_col], label=f'{coin_name} Mid Price', color='#58a6ff', lw=1.5, zorder=2)
    
    # Активные ордера в виде точек/линий
    buy_orders = []
    sell_orders = []
    for idx, row in df_ticks.iterrows():
        side, o_p = parse_order_price(row[order_col])
        if side == 'BUY':
            buy_orders.append((row['timestamp'], o_p))
        elif side == 'SELL':
            sell_orders.append((row['timestamp'], o_p))
            
    if buy_orders:
        b_times, b_prices = zip(*buy_orders)
        ax1.scatter(b_times, b_prices, color='#2ea043', s=4, alpha=0.35, label='Active BUY Maker Quote', zorder=3)
    if sell_orders:
        s_times, s_prices = zip(*sell_orders)
        ax1.scatter(s_times, s_prices, color='#f85149', s=4, alpha=0.35, label='Active SELL Maker Quote', zorder=3)

    # Исполненные сделки (Fills)
    if not coin_fills.empty:
        buys = coin_fills[coin_fills['side'] == 'BUY']
        sells = coin_fills[coin_fills['side'] == 'SELL']
        
        if not buys.empty:
            ax1.scatter(buys['timestamp'], buys['price'], color='#3fb950', s=140, marker='^', edgecolors='white', lw=1.5, label='BUY Fill', zorder=5)
            for _, b in buys.iterrows():
                ax1.annotate(f"BUY {b['qty']:.1f}@{b['price']}", (b['timestamp'], b['price']),
                             xytext=(0, -22), textcoords='offset points', ha='center', fontsize=8,
                             color='#3fb950', bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', edgecolor='#3fb950', alpha=0.85))
        if not sells.empty:
            ax1.scatter(sells['timestamp'], sells['price'], color='#f85149', s=140, marker='v', edgecolors='white', lw=1.5, label='SELL Fill', zorder=5)
            for _, s in sells.iterrows():
                ax1.annotate(f"SELL {s['qty']:.1f}@{s['price']}", (s['timestamp'], s['price']),
                             xytext=(0, 14), textcoords='offset points', ha='center', fontsize=8,
                             color='#f85149', bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', edgecolor='#f85149', alpha=0.85))

    ax1.set_ylabel(f'Price (USDT)', color='#c9d1d9', fontsize=11)
    ax1.set_title(f'LIVE TRADING OPERATIONS — {coin_name} / USDT (Pure Maker 0.0% Fee)', color='#f0f6fc', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)

    # Панель 2: Накопление токенов (Q_pnl %)
    ax2.plot(df_ticks['timestamp'], df_ticks[pnl_col], label='Token Accumulation $Q_{pnl}$ (%)', color='#e3b341', lw=1.8, zorder=2)
    ax2.axhline(0, color='#8b949e', linestyle=':', lw=1, alpha=0.7)
    ax2.set_ylabel('Token PnL (%)', color='#c9d1d9', fontsize=11)
    ax2.set_xlabel('Время (MSK)', color='#c9d1d9', fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9)

    plt.tight_layout()
    if output_png:
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png, dpi=160, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"📊 График успешно сохранен: {output_png}")
    plt.close()

def plot_plotly_html(df_ticks, df_fills, output_html):
    """Генерирует интерактивный Plotly HTML с переключением между PRL и QUANTUS."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Рыночная цена и операции бота (SafeTrade)", "Чистый прирост запаса монет (Q_pnl %)"),
        row_heights=[0.7, 0.3]
    )

    # PRL Price
    fig.add_trace(go.Scatter(
        x=df_ticks['timestamp'], y=df_ticks['prl_price'],
        mode='lines', name='PRL Mid Price',
        line=dict(color='#58a6ff', width=1.5)
    ), row=1, col=1)

    # PRL Fills
    prl_buys = df_fills[(df_fills['coin'] == 'PRL') & (df_fills['side'] == 'BUY')]
    prl_sells = df_fills[(df_fills['coin'] == 'PRL') & (df_fills['side'] == 'SELL')]

    if not prl_buys.empty:
        fig.add_trace(go.Scatter(
            x=prl_buys['timestamp'], y=prl_buys['price'],
            mode='markers+text', name='PRL 🟢 BUY Fill',
            marker=dict(symbol='triangle-up', size=14, color='#3fb950', line=dict(width=1.5, color='white')),
            text=[f"BUY {q}@{p}" for q, p in zip(prl_buys['qty'], prl_buys['price'])],
            textposition="bottom center"
        ), row=1, col=1)

    if not prl_sells.empty:
        fig.add_trace(go.Scatter(
            x=prl_sells['timestamp'], y=prl_sells['price'],
            mode='markers+text', name='PRL 🔴 SELL Fill',
            marker=dict(symbol='triangle-down', size=14, color='#f85149', line=dict(width=1.5, color='white')),
            text=[f"SELL {q}@{p}" for q, p in zip(prl_sells['qty'], prl_sells['price'])],
            textposition="top center"
        ), row=1, col=1)

    # PRL PnL
    fig.add_trace(go.Scatter(
        x=df_ticks['timestamp'], y=df_ticks['prl_pnl'],
        mode='lines', name='PRL Q_PnL (%)',
        line=dict(color='#e3b341', width=2)
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        title="<b>🚀 LIVE REAL-TIME TRADING TELEMETRY (SafeTrade Pure Maker)</b>",
        hovermode='x unified',
        height=750,
        margin=dict(l=50, r=40, t=70, b=40)
    )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"🌐 Интерактивный дашборд сохранен: {output_html}")

def main():
    parser = argparse.ArgumentParser(description="Визуализация операций торгового бота онлайн.")
    parser.add_argument("--coin", choices=["prl", "quantus", "all"], default="all", help="Инструмент для отображения")
    parser.add_argument("--local-log", type=str, default=None, help="Путь к локальному логу (если не нужно скачивать с VPS)")
    args = parser.parse_args()

    if args.local_log and os.path.exists(args.local_log):
        with open(args.local_log, "r", encoding="utf-8") as f:
            log_text = f.read()
    else:
        log_text, _ = fetch_telemetry_from_vps()

    df_ticks, df_fills = parse_telemetry(log_text)
    if df_ticks.empty:
        print("❌ Не найдено записей котирования в логе.")
        return

    # Сохраняем PNG и HTML в папку data/
    out_dir = os.path.join(PROJECT_ROOT, "data", "plots")
    os.makedirs(out_dir, exist_ok=True)

    if args.coin in ["prl", "all"]:
        plot_matplotlib(df_ticks, df_fills, coin='prl', output_png=os.path.join(out_dir, "live_operations_prl.png"))
    if args.coin in ["quantus", "all"]:
        plot_matplotlib(df_ticks, df_fills, coin='quantus', output_png=os.path.join(out_dir, "live_operations_quantus.png"))

    plot_plotly_html(df_ticks, df_fills, os.path.join(out_dir, "live_operations_interactive.html"))

if __name__ == "__main__":
    main()
