#!/usr/bin/env python3
"""
Визуализация и валидация 1-секундных датасетов признаков.
Поддерживает монеты: PRL, QUANTUS, TSC (аргумент --coin).
Отображает:
- Цены Mid/Bid/Ask по всем биржам (SafeTrade, BigONE, Rhea)
- Межбиржевые арбитражные спреды и профитные окна
- Объемы сделок и дисбаланс стаканов
- Ончейн-динамику и макро-поводырей (BTC / NEAR)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

init(autoreset=True)

def visualize_coin(coin_name):
    coin = coin_name.lower()
    print(f"\n{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}🔍 ВИЗУАЛИЗАЦИЯ ДАТАСЕТА ПРИЗНАКОВ: {coin.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}\n")

    features_dir = os.path.join(BASE_DIR, 'data', 'features')
    coin_file = os.path.join(features_dir, f"{coin}_1s_features.parquet")
    legacy_file = os.path.join(features_dir, "merged_1s_base.parquet")

    if os.path.exists(coin_file):
        df = pd.read_parquet(coin_file)
    elif coin == 'prl' and os.path.exists(legacy_file):
        df = pd.read_parquet(legacy_file)
    else:
        print(f"{Fore.RED}Файл {coin_file} не найден! Запустите generate_features.py --coin {coin}.{Style.RESET_ALL}")
        return

    print(f"{Fore.GREEN}--- Основная статистика ---{Style.RESET_ALL}")
    print(f"Размер таблицы: {df.shape[0]:,} строк, {df.shape[1]} колонок")
    print(f"Диапазон времени (MSK): {df.index.min()} -> {df.index.max()}")

    nan_count = df.isna().sum().sum()
    if nan_count == 0:
        print(f"{Fore.GREEN}Пропусков (NaN): 0 (Данные идеально чистые!){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Внимание: найдено {nan_count} пропусков (NaN).{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}--- Спреды и арбитраж ---{Style.RESET_ALL}")
    if 'BO_spread' in df.columns:
        print(f"Средний спред BigONE:    {df['BO_spread'].mean():.5f} USDT")
    if 'ST_spread' in df.columns:
        print(f"Средний спред SafeTrade: {df['ST_spread'].mean():.5f} USDT")

    if 'arb_st_buy_bo_sell' in df.columns:
        arb1 = df['arb_st_buy_bo_sell']
        arb2 = df['arb_bo_buy_st_sell']
        pos1 = (arb1 > 0).sum()
        pos2 = (arb2 > 0).sum()
        print(f"Арбитраж (Купить ST -> Продать BO): средний = {arb1.mean():+.5f} USDT | Окон с прибылью > 0: {pos1:,} сек ({pos1/len(df)*100:.1f}%)")
        print(f"Арбитраж (Купить BO -> Продать ST): средний = {arb2.mean():+.5f} USDT | Окон с прибылью > 0: {pos2:,} сек ({pos2/len(df)*100:.1f}%)")

    if 'arb_rhea_diff_st' in df.columns:
        diff_rhea = df['arb_rhea_diff_st']
        print(f"Разница Rhea DEX vs SafeTrade: средняя = {diff_rhea.mean():+.4f} USDT (max = {diff_rhea.max():+.4f})")

    # Построение графиков
    print(f"\n{Fore.CYAN}Генерация комплексного графика...{Style.RESET_ALL}")
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # 1. Цены
    ax1 = axes[0]
    if 'ST_mid_price' in df.columns:
        ax1.plot(df.index, df['ST_mid_price'], label='SafeTrade Mid', color='orange', alpha=0.9, lw=1.2)
    if 'BO_mid_price' in df.columns:
        ax1.plot(df.index, df['BO_mid_price'], label='BigONE Mid', color='blue', alpha=0.9, lw=1.2)
    if 'RHEA_price_usd' in df.columns:
        ax1.plot(df.index, df['RHEA_price_usd'], label='Rhea DEX Price', color='purple', alpha=0.8, lw=1.2)
    ax1.set_title(f"1. Динамика котировок {coin.upper()} на площадках")
    ax1.set_ylabel("Цена (USDT)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Арбитражный спред
    ax2 = axes[1]
    if 'arb_st_buy_bo_sell' in df.columns:
        ax2.plot(df.index, df['arb_st_buy_bo_sell'], label='Купить ST -> Продать BO', color='green', alpha=0.7, lw=1)
        ax2.plot(df.index, df['arb_bo_buy_st_sell'], label='Купить BO -> Продать ST', color='crimson', alpha=0.7, lw=1)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title(f"2. Межбиржевой арбитражный спред (Arbitrage Spread)")
        ax2.set_ylabel("Спред (USDT)")
        ax2.legend(loc='upper left')
    elif 'ST_spread' in df.columns:
        ax2.plot(df.index, df['ST_spread'], label='Внутрибиржевой спред SafeTrade', color='darkorange', lw=1)
        ax2.set_title("2. Внутрибиржевой спред SafeTrade")
        ax2.set_ylabel("Спред (USDT)")
        ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. Объемы сделок или ончейн
    ax3 = axes[2]
    vol_cols = [c for c in df.columns if 'trade_vol' in c]
    if vol_cols:
        for vc in vol_cols:
            ax3.plot(df.index, df[vc], label=vc, alpha=0.6, lw=1)
        ax3.set_title("3. Секундные объемы сделок (Trade Volume 1s)")
        ax3.set_ylabel("Объем")
        ax3.legend(loc='upper left')
    elif 'onchain_mempool_size' in df.columns:
        ax3.plot(df.index, df['onchain_mempool_size'], label='Pearl Mempool Size', color='brown', lw=1)
        ax3.set_title("3. Размер ончейн-мемпула (Mempool Transactions)")
        ax3.set_ylabel("Транзакций")
        ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 4. Макро-поводыри
    ax4 = axes[3]
    if 'macro_btc_mid' in df.columns:
        color = 'tab:orange'
        ax4.set_ylabel('BTC Price ($)', color=color)
        ax4.plot(df.index, df['macro_btc_mid'], color=color, label='Binance BTC/USDT', lw=1.2)
        ax4.tick_params(axis='y', labelcolor=color)

        if 'macro_near_mid' in df.columns:
            ax4_twin = ax4.twinx()
            color2 = 'tab:cyan'
            ax4_twin.set_ylabel('NEAR Price ($)', color=color2)
            ax4_twin.plot(df.index, df['macro_near_mid'], color=color2, label='Binance NEAR/USDT', lw=1.2, linestyle=':')
            ax4_twin.tick_params(axis='y', labelcolor=color2)

        ax4.set_title("4. Мировые макро-поводыри Binance (BTC & NEAR)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_plot = os.path.join(BASE_DIR, f"data_overview_{coin}.png")
    plt.savefig(out_plot, dpi=120)
    plt.close()
    print(f"{Fore.GREEN}✅ График успешно сохранен в: {os.path.basename(out_plot)}{Style.RESET_ALL}")

    if coin == 'prl':
        legacy_plot = os.path.join(BASE_DIR, "data_overview_plot.png")
        import shutil
        shutil.copy(out_plot, legacy_plot)

def main():
    parser = argparse.ArgumentParser(description="Визуализация и аудит 1-секундных признаков.")
    parser.add_argument("--coin", type=str, default="prl", choices=["prl", "quantus", "tsc", "all"], help="Монета для визуализации (по умолчанию prl).")
    args = parser.parse_args()

    coins = ["prl", "quantus", "tsc"] if args.coin == "all" else [args.coin]
    for c in coins:
        visualize_coin(c)

if __name__ == "__main__":
    main()
