#!/usr/bin/env python3
"""
Скрипт проверки качества и плотности рыночных данных реального времени.
Работает напрямую с объединенным хранилищем Parquet (через get_market_data).
"""

import sys
import os

# Добавляем корень проекта в sys.path для поддержки запуска из любого каталога
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

from _tools.sync_all_data import get_market_data

init(autoreset=True)

def analyze_dataset(df, name, is_trades=False):
    if df is None or df.empty:
        print(f"{Fore.RED}❌ Датасет {name} пуст или отсутствует!{Style.RESET_ALL}\n")
        return None

    # Приводим к московскому времени для наглядного отчета
    ts_col = df['timestamp']
    if ts_col.dt.tz is None:
        ts_msk = ts_col.dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
    else:
        ts_msk = ts_col.dt.tz_convert('Europe/Moscow')

    total_records = len(df)
    t_min = ts_msk.min().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    t_max = ts_msk.max().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    print(f"{Fore.CYAN}--- Отчет по: {name} ---{Style.RESET_ALL}")
    print(f"Всего записей: {total_records:,}")
    print(f"Период (MSK): {t_min} -> {t_max}")

    # Анализ разрывов (только для стаканов)
    if not is_trades:
        deltas = ts_msk.diff().dt.total_seconds()
        gaps = deltas[deltas > 5.0]
        print(f"Найдено разрывов связи (> 5 сек): {Fore.YELLOW}{len(gaps)}{Style.RESET_ALL}")

        if len(gaps) > 0:
            print(f"Максимальный разрыв: {Fore.RED}{gaps.max():.1f} сек{Style.RESET_ALL}")
            print(f"Средний интервал обновления: {deltas.mean():.2f} сек")
        else:
            print(f"{Fore.GREEN}✅ Разрывов больше 5 секунд нет! Поток непрерывен.{Style.RESET_ALL}")

    print("")
    return ts_msk

def main():
    parser = argparse.ArgumentParser(description="Анализ качества и плотности собранных рыночных данных.")
    parser.add_argument("--sync", action="store_true", help="Сначала выполнить синхронизацию с Google Drive и VPS.")
    args = parser.parse_args()

    print(f"{Fore.GREEN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.GREEN}🩺 АНАЛИЗ РЫНОЧНЫХ ДАННЫХ РЕАЛЬНОГО ВРЕМЕНИ{Style.RESET_ALL}")
    print(f"{Fore.GREEN}==================================================={Style.RESET_ALL}\n")

    st_depth, st_trades, bo_depth, bo_trades = get_market_data(force_sync=args.sync)

    bo_depth_ts = analyze_dataset(bo_depth, "BigONE Стаканы")
    bo_trades_ts = analyze_dataset(bo_trades, "BigONE Сделки", is_trades=True)
    st_depth_ts = analyze_dataset(st_depth, "SafeTrade Стаканы")
    st_trades_ts = analyze_dataset(st_trades, "SafeTrade Сделки", is_trades=True)

    # Построение графика плотности данных
    print(f"{Fore.CYAN}Генерация графика плотности данных...{Style.RESET_ALL}")
    plt.figure(figsize=(14, 5))

    if bo_depth_ts is not None and not bo_depth_ts.empty:
        plt.plot(bo_depth_ts, np.ones(len(bo_depth_ts)), '|', color='blue', label='BigONE Depth (Стаканы)', alpha=0.5, markersize=20)

    if st_depth_ts is not None and not st_depth_ts.empty:
        plt.plot(st_depth_ts, np.ones(len(st_depth_ts)) * 1.1, '|', color='orange', label='SafeTrade Depth (Стаканы)', alpha=0.5, markersize=20)

    plt.yticks([1, 1.1], ['BigONE', 'SafeTrade'])
    plt.ylim(0.9, 1.2)
    plt.title('Плотность поступления данных со стаканов (MSK)')
    plt.xlabel('Время (MSK)')
    plt.legend(loc='upper right')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_file = 'data_density_plot.png'
    plt.savefig(plot_file, dpi=150)
    print(f"{Fore.GREEN}✅ График плотности успешно сохранен в: {plot_file}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
