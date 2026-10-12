#!/usr/bin/env python3
"""
Скрипт комплексной проверки качества, плотности и свежести рыночных данных.
Поддерживает новую иерархическую структуру данных и любые монеты:
- PRL (SafeTrade, BigONE, On-chain Pearl)
- QUANTUS (SafeTrade, BigONE, Rhea DEX, On-chain Quantus)
- TSC (SafeTrade)
- Macro (Binance BTC & NEAR)
- Правила и статусы бирж (SafeTrade, BigONE)
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

# Добавляем корень проекта в sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.data_loader import load_coin, load_macro, load_rules, sync_data

init(autoreset=True)

def analyze_dataset(df, name, is_trades=False, max_gap_sec=5.0):
    if df is None or df.empty:
        print(f"  {Fore.RED}❌ {name}: пуст или отсутствует!{Style.RESET_ALL}")
        return None

    if 'timestamp' not in df.columns:
        print(f"  {Fore.YELLOW}⚠️ {name}: колонка timestamp не найдена.{Style.RESET_ALL}")
        return None

    ts_col = df['timestamp']
    if ts_col.dt.tz is None:
        ts_msk = ts_col.dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
    else:
        ts_msk = ts_col.dt.tz_convert('Europe/Moscow')

    total_records = len(df)
    t_min = ts_msk.min().strftime('%Y-%m-%d %H:%M:%S')
    t_max = ts_msk.max().strftime('%Y-%m-%d %H:%M:%S')

    print(f"  {Fore.CYAN}• {name:<30}{Style.RESET_ALL} {total_records:>8,} строк | {t_min} -> {t_max}", end="")

    # Анализ разрывов (для стаканов и макро)
    if not is_trades:
        deltas = ts_msk.diff().dt.total_seconds()
        gaps = deltas[deltas > max_gap_sec]
        if len(gaps) > 0:
            print(f" | Разрывов >{max_gap_sec:.0f}с: {Fore.YELLOW}{len(gaps)}{Style.RESET_ALL} (макс {gaps.max():.1f}с)")
        else:
            print(f" | {Fore.GREEN}Разрывов нет (непрерывно){Style.RESET_ALL}")
    else:
        print(f" | {Fore.GREEN}Тики сделок{Style.RESET_ALL}")

    return ts_msk

def check_coin(coin_name):
    coin_name = coin_name.lower()
    print(f"\n{Fore.GREEN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.GREEN}🪙 АНАЛИЗ ДАННЫХ МОНЕТЫ: {coin_name.upper()}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}==================================================={Style.RESET_ALL}")

    cdata = load_coin(coin_name)
    rules = load_rules(coin_name)

    # 1. Правила бирж
    if rules:
        print(f"\n📋 {Fore.YELLOW}Торговые правила и статусы шлюзов:{Style.RESET_ALL}")
        for ex, ex_dict in rules.items():
            if ex in ["safetrade", "bigone"] and ex_dict:
                for pair, pinfo in ex_dict.items():
                    step = pinfo.get('price_precision') or pinfo.get('quote_scale')
                    lot = pinfo.get('amount_precision') or pinfo.get('base_scale')
                    min_ord = pinfo.get('min_amount') or pinfo.get('min_quote_value')
                    dep = pinfo.get('deposit_enabled', True)
                    wdr = pinfo.get('withdrawal_enabled', True)
                    fee = pinfo.get('withdrawal_fee')
                    fee_str = f", Вывод fee: {fee}" if fee is not None else ""
                    print(f"  - [{ex.upper()}] {pair}: шаг цены={step}, шаг лота={lot}, мин={min_ord} | Депозит: {dep}, Вывод: {wdr}{fee_str}")

    # 2. Датасеты
    print(f"\n📊 {Fore.YELLOW}Потоки данных:{Style.RESET_ALL}")
    ts_dict = {}
    for key in sorted(cdata.keys()):
        df = cdata[key]
        is_tr = 'trades' in key
        max_gap = 60.0 if 'onchain' in key else 5.0
        label = f"{coin_name.upper()} {key}"
        ts = analyze_dataset(df, label, is_trades=is_tr, max_gap_sec=max_gap)
        if ts is not None:
            ts_dict[key] = ts

    return ts_dict

def check_macro():
    print(f"\n{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}🌍 МАКРО-ПОВОДЫРИ (BINANCE WEBSOCKET){Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")

    df_macro = load_macro()
    if df_macro is None or df_macro.empty:
        print(f"  {Fore.RED}❌ Данные макро-поводырей отсутствуют!{Style.RESET_ALL}")
        return None

    analyze_dataset(df_macro, "Binance BTC & NEAR (1s)", is_trades=False, max_gap_sec=5.0)

    last_row = df_macro.iloc[-1]
    btc_p = last_row.get('btc_last') or last_row.get('btc_mid')
    near_p = last_row.get('near_last') or last_row.get('near_mid')
    print(f"  Текущие котировки Binance: BTC = ${float(btc_p):,.2f} | NEAR = ${float(near_p):,.4f}")
    return df_macro

def main():
    parser = argparse.ArgumentParser(description="Анализ качества и плотности собранных рыночных данных.")
    parser.add_argument("--coin", type=str, default="all", choices=["prl", "quantus", "tsc", "all"], help="Монета для анализа (по умолчанию all).")
    parser.add_argument("--sync", action="store_true", help="Сначала выполнить синхронизацию с Google Drive и VPS.")
    args = parser.parse_args()

    if args.sync:
        print(f"{Fore.CYAN}🔄 Выполняется синхронизация данных...{Style.RESET_ALL}")
        sync_data()

    coins_to_check = ["prl", "quantus", "tsc"] if args.coin == "all" else [args.coin]

    all_ts = {}
    for c in coins_to_check:
        ts_dict = check_coin(c)
        for k, v in ts_dict.items():
            all_ts[f"{c}_{k}"] = v

    check_macro()

    # Генерация графика плотности
    plot_keys = [k for k in all_ts if 'depth' in k or 'price' in k]
    if plot_keys:
        print(f"\n{Fore.CYAN}📈 Генерация графика непрерывности потоков...{Style.RESET_ALL}")
        plt.figure(figsize=(14, max(4, len(plot_keys) * 0.8)))
        
        y_pos = 1
        for key in plot_keys:
            ts = all_ts[key]
            if not ts.empty:
                plt.plot(ts, np.ones(len(ts)) * y_pos, '|', label=key, alpha=0.6, markersize=14)
                y_pos += 1

        plt.yticks(range(1, y_pos), plot_keys)
        plt.title("Непрерывность рыночных потоков (Realtime Data Streams)")
        plt.xlabel("Время (MSK)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(BASE_DIR, "data_density_plot.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"{Fore.GREEN}✅ График плотности успешно сохранен в: {os.path.basename(plot_path)}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
