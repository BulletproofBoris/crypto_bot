#!/usr/bin/env python3
"""
Калькулятор максимальной длины окна наблюдения (L_max) и эффективного размера выборки (N_eff).

Математическая основа:
- При частоте дискретизации 1 Гц соседние секунды автокоррелированы.
- Независимые траектории наблюдения определяются длиной окна L = K * H (сек).
- Эффективное число независимых сэмплов в обучающей выборке:
    N_eff = T_train / L = T_train / (K * H)
- При заданном целевом числе независимых событий N_target максимальное допустимое окно:
    L_max = floor(T_train / N_target)
    H_max(K) = floor(L_max / K)
- Требуемый объем непрерывных данных (в днях) для заданного L при целевом N_target:
    Days_required = (N_target * L) / (split_ratio * 86400)
"""

import os
import sys
import argparse
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="Расчет L_max, N_eff и достаточности выборки для торговых инструментов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n", "--n-target",
        type=str,
        default="5000",
        help="Целевой эффективный размер выборки (можно список через запятую, например: 4000,5000)"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Доля обучающей выборки (Train ratio)"
    )
    parser.add_argument(
        "--coins",
        type=str,
        default="prl,quantus",
        help="Список монет через запятую или 'all'"
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="3,5,8,15,25,30,60",
        help="Список горизонтов прогноза H (сек) для матричного анализа"
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="1,2,3,4,5",
        help="Список коэффициентов окна K = L / H для матричного анализа"
    )
    parser.add_argument(
        "--show-classes",
        action="store_true",
        help="Показать фактическое распределение всех 7 классов в размеченных данных"
    )
    return parser.parse_args()


def load_coin_metadata(coin: str):
    """Считывает метаданные по объему данных для монеты."""
    feat_path = BASE_DIR / "data" / "features" / f"{coin.lower()}_1s_features.parquet"
    if not feat_path.exists():
        return None
    
    meta = pq.read_metadata(feat_path)
    total_rows = meta.num_rows
    num_cols = meta.num_columns
    
    # Пытаемся считать временной диапазон (первый и последний timestamp)
    start_ts, end_ts = None, None
    try:
        t = pq.read_table(feat_path, columns=['timestamp'])
        df_ts = t.to_pandas()
        start_ts = pd.to_datetime(df_ts['timestamp'].iloc[0])
        end_ts = pd.to_datetime(df_ts['timestamp'].iloc[-1])
    except Exception:
        pass
    
    return {
        'coin': coin.upper(),
        'file_path': feat_path,
        'total_rows': total_rows,
        'num_cols': num_cols,
        'start_ts': start_ts,
        'end_ts': end_ts,
    }


def analyze_lmax_for_coin(meta: dict, n_targets: list, split_ratio: float, horizons: list, ratios: list):
    """Выполняет расчет L_max и матрицу совместимости."""
    total_rows = meta['total_rows']
    t_train = int(total_rows * split_ratio)
    t_val = total_rows - t_train
    days_total = total_rows / 86400.0
    days_train = t_train / 86400.0

    print("=" * 105)
    print(f"🪙  ИНСТРУМЕНТ: {meta['coin']}  |  Файл: {meta['file_path'].name}")
    print(f"   📊 Общий объем: {total_rows:,} сек ({days_total:.2f} дней / {days_total*24:.1f} ч)")
    if meta['start_ts'] and meta['end_ts']:
        print(f"   ⏱️  Диапазон:   {meta['start_ts']} ➔ {meta['end_ts']}")
    print(f"   🎯 Обучение ({split_ratio*100:.0f}% Train): {t_train:,} сек ({days_train:.2f} дней)")
    print(f"   🧪 Валидация ({(1-split_ratio)*100:.0f}% Val):   {t_val:,} сек ({t_val/86400:.2f} дней)")
    print("=" * 105)

    # 1. Сводка L_max для каждого заданного N_target
    print(f"\n📌 РАСЧЕТ L_max И ГРАНИЧНЫХ ГОРИЗОНТОВ H_max:")
    headers = [
        "N_target", "L_max (сек)", "L_max (Train)", "H_max (K=1)", "H_max (K=2)", 
        "H_max (K=3)", "H_max (K=4)", "H_max (K=5)"
    ]
    print(f"{headers[0]:<10} | {headers[1]:<12} | {headers[2]:<13} | {headers[3]:<11} | {headers[4]:<11} | {headers[5]:<11} | {headers[6]:<11} | {headers[7]:<11}")
    print("-" * 105)

    for n in n_targets:
        l_max_train = t_train // n
        l_max_total = total_rows // n
        h1 = l_max_train // 1
        h2 = l_max_train // 2
        h3 = l_max_train // 3
        h4 = l_max_train // 4
        h5 = l_max_train // 5
        print(f"{n:<10} | {l_max_total:>9} с  | {l_max_train:>10} с | {h1:>8} с  | {h2:>8} с  | {h3:>8} с  | {h4:>8} с  | {h5:>8} с")
    print("-" * 105)

    # 2. Матрица тактических горизонтов для основного N_target (первого в списке)
    main_n = n_targets[0]
    main_lmax = t_train // main_n
    print(f"\n📋 МАТРИЦА СОВМЕСТИМОСТИ ТАКТИЧЕСКИХ СЕТОК (H, K) ДЛЯ N_target = {main_n:,} (L_max = {main_lmax} с):")
    m_headers = ["H (сек)", "K (L/H)", "L = K*H", "N_eff (Train)", "Запас к цели", "Статус", "Нужно дней истории"]
    print(f"{m_headers[0]:<8} | {m_headers[1]:<8} | {m_headers[2]:<8} | {m_headers[3]:<13} | {m_headers[4]:<13} | {m_headers[5]:<16} | {m_headers[6]:<18}")
    print("-" * 105)

    for h in horizons:
        for k in ratios:
            L = h * k
            n_eff = t_train // L
            margin = n_eff / main_n
            req_days = (main_n * L) / (split_ratio * 86400.0)
            
            if L <= main_lmax:
                status = "✅ ДОСТУПЕН"
            else:
                deficit_days = req_days - days_total
                status = f"❌ НЕ ХВАТАЕТ (+{deficit_days:.1f}д)"
            
            margin_str = f"{margin:.2f}x"
            print(f"{h:<8} | {k:<8} | {L:>5} с  | {n_eff:>10,}  | {margin_str:>11}  | {status:<16} | {req_days:>15.2f} дн.")
    print("-" * 105)


def print_class_distribution(coin: str):
    """Выводит фактическое распределение всех 7 классов в размеченном датасете."""
    labeled_path = BASE_DIR / "data" / "labeled" / coin.lower() / f"{coin.lower()}_labeled.parquet"
    if not labeled_path.exists():
        print(f"\n⚠️ Файл разметки {labeled_path} не найден.")
        return

    print(f"\n📊 ФАКТИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ 7 КЛАССОВ ДЛЯ {coin.upper()}:")
    try:
        tbl = pq.read_table(labeled_path)
        cols = [c for c in tbl.column_names if c.startswith("target_class_7")]
        if not cols:
            print("   Нет колонок target_class_7")
            return

        total_n = len(tbl)
        n_tr = int(total_n * 0.8)
        n_va = total_n - n_tr

        # Берем до 3 ключевых горизонтов
        sample_cols = [c for c in ["target_class_7_h3_k1", "target_class_7_h8_k2", "target_class_7_h25_k2", "target_class_7"] if c in cols]
        if not sample_cols:
            sample_cols = cols[:3]

        for col in sample_cols:
            s_tr = tbl[col].slice(0, n_tr).to_pandas().value_counts().sort_index()
            s_va = tbl[col].slice(n_tr, n_va).to_pandas().value_counts().sort_index()

            print(f"\n  ➤ Таргет: {col} (Train 80%: {n_tr:,} с | Val 20%: {n_va:,} с)")
            print(f"    {'Класс':<8} | {'Смысл класса':<38} | {'Train (N)':<10} | {'Train (%)':<10} | {'Val (N)':<10} | {'Val (%)':<10}")
            print("    " + "-" * 98)

            class_names = {
                -3: "Экстремальный дамп / кит (M2-M3)",
                -2: "Рабочий спад / пробой вниз (M1-M2)",
                -1: "Слабый спад (шум)",
                0: "Флэт / нейтральная зона",
                1: "Слабый рост (шум)",
                2: "Рабочий рост / пробой вверх (P1-P2)",
                3: "Экстремальный памп / кит (P2-P3)",
            }

            for cls in range(-3, 4):
                c_tr = s_tr.get(cls, 0)
                p_tr = c_tr / n_tr * 100.0 if n_tr > 0 else 0.0
                c_va = s_va.get(cls, 0)
                p_va = c_va / n_va * 100.0 if n_va > 0 else 0.0
                name = class_names.get(cls, "Неизвестно")
                print(f"    {cls:+2d}{'':<5} | {name:<38} | {c_tr:>10,} | {p_tr:>8.2f}%  | {c_va:>10,} | {p_va:>8.2f}%")
            print("    " + "-" * 98)

    except Exception as e:
        print(f"   Ошибка чтения классов: {e}")


def main():
    args = parse_args()

    # Парсим N_target
    n_targets = [int(x.strip()) for x in args.n_target.split(",") if x.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    ratios = [int(x.strip()) for x in args.ratios.split(",") if x.strip()]

    # Парсим монеты
    if args.coins.lower() == 'all':
        feat_dir = BASE_DIR / "data" / "features"
        coins = []
        if feat_dir.exists():
            for f in feat_dir.glob("*_1s_features.parquet"):
                cname = f.name.replace("_1s_features.parquet", "")
                if cname not in ['macro', 'all']:
                    coins.append(cname)
    else:
        coins = [c.strip().lower() for c in args.coins.split(",") if c.strip()]

    print("\n" + "=" * 105)
    print("🚀 КАЛЬКУЛЯТОР L_max, N_eff И МАКСИМАЛЬНЫХ ГОРИЗОНТОВ МОДЕЛЕЙ")
    print(f"   Параметры: N_targets = {n_targets}  |  Train Split = {args.split*100:.0f}%  |  Монеты: {', '.join([c.upper() for c in coins])}")
    print("=" * 105 + "\n")

    found_any = False
    for coin in coins:
        meta = load_coin_metadata(coin)
        if not meta:
            print(f"❌ Данные для монеты {coin.upper()} не найдены в data/features/{coin}_1s_features.parquet")
            continue
        found_any = True
        analyze_lmax_for_coin(meta, n_targets, args.split, horizons, ratios)
        if args.show_classes:
            print_class_distribution(coin)
        print("\n")

    if not found_any:
        print("Не найдено ни одного файла фич в data/features/!")
        sys.exit(1)


if __name__ == "__main__":
    main()
