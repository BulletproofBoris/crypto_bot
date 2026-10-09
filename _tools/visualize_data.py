#!/usr/bin/env python3
"""
Проверка и визуализация сводного 1-секундного датасета.
Работает напрямую с Parquet (merged_1s_base.parquet) для мгновенной загрузки.
Отображает:
- Лучшие цены Bid/Ask по обеим биржам
- Внутрибиржевые и межбиржевые (арбитражные) спреды
- Объемы сделок
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

init(autoreset=True)

def main():
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}🔍 ПРОВЕРКА И ВИЗУАЛИЗАЦИЯ СВОДНОЙ ТАБЛИЦЫ (1s){Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}\n")

    parquet_path = os.path.join(BASE_DIR, 'data', 'features', 'merged_1s_base.parquet')
    legacy_parquet = os.path.join(BASE_DIR, 'data', 'processed', 'merged_1s_base.parquet')

    if os.path.exists(parquet_path):
        print("⚡ Быстрая загрузка данных из Parquet (data/features/)...")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(legacy_parquet):
        print("⚡ Быстрая загрузка данных из Parquet (data/processed/)...")
        df = pd.read_parquet(legacy_parquet)
    else:
        print(f"{Fore.RED}Файл признаков не найден в data/features/! Запустите generate_features.py.{Style.RESET_ALL}")
        return

    print(f"\n{Fore.GREEN}--- Основная статистика ---{Style.RESET_ALL}")
    print(f"Размер таблицы: {df.shape[0]:,} строк, {df.shape[1]} колонок")
    print(f"Диапазон времени: {df.index.min()} -> {df.index.max()}")

    nan_count = df.isna().sum().sum()
    if nan_count == 0:
        print(f"{Fore.GREEN}Пропусков (NaN): 0 (Данные идеально чистые!){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Внимание: найдено {nan_count} пропусков (NaN).{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}--- Спреды и цены ---{Style.RESET_ALL}")
    bo_spread = (df['BO_ask_1_price'] - df['BO_bid_1_price']).mean()
    st_spread = (df['ST_ask_1_price'] - df['ST_bid_1_price']).mean()
    print(f"Средний спред BigONE:   {bo_spread:.5f} USDT")
    print(f"Средний спред SafeTrade: {st_spread:.5f} USDT")

    if 'arb_st_buy_bo_sell' in df.columns:
        arb_1 = df['arb_st_buy_bo_sell']
        arb_2 = df['arb_bo_buy_st_sell']
        print(f"Межбиржевой спред (Купить SafeTrade -> Продать BigONE): макс = {arb_1.max():.5f}, средний = {arb_1.mean():.5f}")
        print(f"Межбиржевой спред (Купить BigONE -> Продать SafeTrade): макс = {arb_2.max():.5f}, средний = {arb_2.mean():.5f}")

    # Визуализация (3 панели)
    print("\nГенерация графиков...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # 1. Лучшие цены Bid / Ask
    axes[0].plot(df.index, df['BO_bid_1_price'], label='BigONE Best Bid', color='blue', alpha=0.7)
    axes[0].plot(df.index, df['BO_ask_1_price'], label='BigONE Best Ask', color='cyan', alpha=0.7)
    axes[0].plot(df.index, df['ST_bid_1_price'], label='SafeTrade Best Bid', color='red', alpha=0.7)
    axes[0].plot(df.index, df['ST_ask_1_price'], label='SafeTrade Best Ask', color='orange', alpha=0.7)
    axes[0].set_title('Лучшие цены спроса и предложения (Best Bid & Ask)')
    axes[0].set_ylabel('Цена (USDT)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Межбиржевой арбитражный спред
    if 'arb_st_buy_bo_sell' in df.columns:
        axes[1].plot(df.index, df['arb_st_buy_bo_sell'], label='SafeTrade Buy -> BigONE Sell', color='purple', alpha=0.8)
        axes[1].plot(df.index, df['arb_bo_buy_st_sell'], label='BigONE Buy -> SafeTrade Sell', color='green', alpha=0.8)
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.7)
        axes[1].set_title('Межбиржевой арбитражный спред (Положительный = Вилка)')
        axes[1].set_ylabel('Дельта (USDT)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Объемы торгов
    axes[2].bar(df.index, df['BO_trade_vol'], label='BigONE Trade Volume', color='blue', alpha=0.6, width=0.0005)
    axes[2].bar(df.index, df['ST_trade_vol'], label='SafeTrade Trade Volume', color='red', alpha=0.6, width=0.0005)
    axes[2].set_title('Проторгованный объем (Сделки)')
    axes[2].set_ylabel('Объем (PRL)')
    axes[2].set_xlabel('Время (MSK)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_file = 'data_overview_plot.png'
    plt.savefig(plot_file, dpi=150)
    print(f"{Fore.GREEN}✅ График успешно сохранен в: {plot_file}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
