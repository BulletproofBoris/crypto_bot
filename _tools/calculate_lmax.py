#!/usr/bin/env python3
"""
Калькулятор максимальной длины окна наблюдения (L_max) и эффективного размера выборки (N_eff)
с учетом разрывов связи (is_data_gap), зон прогрева (warmup) и буферов очистки (purge/embargo).

Физическая и математическая основа:
- При дискретизации 1 Гц соседние секунды автокоррелированы.
- Не вся последовательность пригодна для обучения:
    1. Начальная зона нечувствительности / прогрева фильтров: W_max = 300 с (5 мин).
    2. Разрывы данных (is_data_gap > 0): потеря сетевых пакетов, простой серверов или биржи.
    3. Защитный буфер перед гэпом (pre-gap purge = max_H): предотвращает заглядывание вперед в пустоту.
    4. Защитный буфер после гэпа (post-gap warmup = 300 с): время релаксации скользящих средних.
    5. Межсплитовый зазор (purge + embargo = 360-385 с): исключение эффекта памяти на границе Train/Val.
- Чистый пригодный объем обучения: T_train_usable (только валидные секунды 'train').
- Максимальное допустимое окно при целевом N_target:
    L_max = floor(T_train_usable / N_target)
    H_max(K) = floor(L_max / K)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="Расчет L_max и N_eff с учетом разрывов связи, зон прогрева и эмбарго",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n", "--n-target",
        type=str,
        default="5000",
        help="Целевой эффективный размер выборки (можно список: 4000,5000)"
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


def load_coin_data(coin: str, split_ratio: float = 0.8):
    """Считывает данные монеты и вычисляет пригодные маски."""
    c = coin.lower()
    labeled_path = BASE_DIR / "data" / "labeled" / c / f"{c}_labeled.parquet"
    feat_path = BASE_DIR / "data" / "features" / f"{c}_1s_features.parquet"

    if labeled_path.exists():
        # Считываем из размеченного файла
        cols_to_load = ['timestamp', 'sample_split', 'is_data_gap']
        tbl = pq.read_table(labeled_path, columns=cols_to_load)
        df = tbl.to_pandas()
        source_type = "labeled"
        source_file = labeled_path
    elif feat_path.exists():
        # Считываем из фичей и эмулируем разбиение
        cols_to_load = ['timestamp', 'is_data_gap']
        tbl = pq.read_table(feat_path, columns=cols_to_load)
        df = tbl.to_pandas()
        source_type = "features"
        source_file = feat_path
        
        # Эмуляция логики data_labeler.py
        n = len(df)
        split_idx = int(n * split_ratio)
        max_L, max_H = 75, 25
        purge_steps = max_H
        embargo_steps = 360
        gap_steps = purge_steps + embargo_steps
        train_end = split_idx - gap_steps

        sample_split = np.full(n, 'unknown', dtype=object)
        sample_split[:max_L] = 'warmup'
        sample_split[max_L:train_end] = 'train'
        sample_split[train_end:split_idx] = 'purged_embargo'
        sample_split[split_idx:n - max_H] = 'val'
        sample_split[n - max_H:] = 'tail'

        if 'is_data_gap' in df.columns:
            raw_gap = (df['is_data_gap'].values > 0)
            if np.any(raw_gap):
                post_gap_warmup = min(int(max_L), 300)
                pre_gap_purge = int(max_H)
                gap_series = pd.Series(raw_gap.astype(np.int32))
                fwd_mask = gap_series.rolling(post_gap_warmup + 1, min_periods=1).max().values > 0
                bwd_mask = gap_series.iloc[::-1].rolling(pre_gap_purge + 1, min_periods=1).max().iloc[::-1].values > 0
                full_gap_mask = (raw_gap | fwd_mask | bwd_mask) & ((sample_split == 'train') | (sample_split == 'val'))
                sample_split[full_gap_mask] = 'purged_gap'
        df['sample_split'] = sample_split
    else:
        return None

    # Временные метки
    df['dt'] = pd.to_datetime(df['timestamp'])
    start_ts = df['dt'].iloc[0]
    end_ts = df['dt'].iloc[-1]
    total_rows = len(df)

    # Детальная статистика по категориям сплита
    split_counts = df['sample_split'].value_counts().to_dict()
    n_train_usable = split_counts.get('train', 0)
    n_val_usable = split_counts.get('val', 0)
    n_purged_gap = split_counts.get('purged_gap', 0)
    n_warmup = split_counts.get('warmup', 0)
    n_purged_embargo = split_counts.get('purged_embargo', 0)
    n_tail = split_counts.get('tail', 0)
    raw_gaps = int((df['is_data_gap'] > 0).sum()) if 'is_data_gap' in df.columns else 0

    # Анализ непрерывных блоков для Train
    train_mask = (df['sample_split'] == 'train')
    blocks = (~train_mask).cumsum()[train_mask]
    block_sizes = blocks.value_counts().sort_index()

    train_blocks = []
    for b_id, size in block_sizes.items():
        sub_df = df.loc[blocks[blocks == b_id].index]
        t0 = sub_df['dt'].iloc[0]
        t1 = sub_df['dt'].iloc[-1]
        train_blocks.append({
            'size': size,
            'hours': size / 3600.0,
            'days': size / 86400.0,
            'start': t0,
            'end': t1
        })

    return {
        'coin': coin.upper(),
        'source_type': source_type,
        'source_file': source_file,
        'total_rows': total_rows,
        'start_ts': start_ts,
        'end_ts': end_ts,
        'raw_gaps': raw_gaps,
        'n_train_usable': n_train_usable,
        'n_val_usable': n_val_usable,
        'n_purged_gap': n_purged_gap,
        'n_warmup': n_warmup,
        'n_purged_embargo': n_purged_embargo,
        'n_tail': n_tail,
        'train_blocks': train_blocks,
        'df': df
    }


def analyze_lmax_for_coin(data: dict, n_targets: list, horizons: list, ratios: list):
    """Выполняет расчет L_max и матрицу совместимости с учетом всех исключений."""
    total_rows = data['total_rows']
    n_train_usable = data['n_train_usable']
    n_val_usable = data['n_val_usable']
    days_total = total_rows / 86400.0
    days_train_usable = n_train_usable / 86400.0
    days_val_usable = n_val_usable / 86400.0

    # Сравнение валового (сырого) 80% Train и чистого Train
    gross_train = int(total_rows * 0.8)
    loss_train = gross_train - n_train_usable
    loss_pct = (loss_train / gross_train) * 100.0 if gross_train > 0 else 0.0

    print("=" * 112)
    print(f"🪙  ИНСТРУМЕНТ: {data['coin']}  |  Источник: {data['source_file'].name} ({data['source_type']})")
    print(f"   ⏱️  Период истории:   {data['start_ts']} ➔ {data['end_ts']}")
    print(f"   📊 Валовый объем:    {total_rows:,} сек ({days_total:.2f} дней / {days_total*24:.1f} ч)")
    print(f"   ⚠️  Сырые гэпы (is_data_gap): {data['raw_gaps']:,} сек ({data['raw_gaps']/3600:.2f} ч)")
    print(f"   🧹 Исключено из обучения (Гэпы + Purge + Warmup): {data['n_purged_gap']:,} сек ({data['n_purged_gap']/3600:.2f} ч)")
    print(f"   🛡️  Начальный прогрев (Warmup): {data['n_warmup']:,} сек  |  Буфер Embargo: {data['n_purged_embargo']:,} сек")
    print(f"   ---------------------------------------------------------------------------------------------")
    print(f"   ✅ ЧИСТЫЙ ОБУЧАЮЩИЙ ОБЪЕМ (Net Usable Train): {n_train_usable:,} сек ({days_train_usable:.2f} дней)")
    print(f"      (Валовый 80% Train был: {gross_train:,} с ➔ Убыль из-за гэпов/буферов: -{loss_train:,} с / -{loss_pct:.1f}%)")
    print(f"   ✅ ЧИСТЫЙ ВАЛИДАЦИОННЫЙ ОБЪЕМ (Net Usable Val): {n_val_usable:,} сек ({days_val_usable:.2f} дней)")
    
    # Непрерывные блоки
    print(f"\n   🧱 Непрерывные сегменты в обучающей выборке (Train blocks): {len(data['train_blocks'])}")
    for i, b in enumerate(data['train_blocks']):
        print(f"      • Блок {i+1}: {b['size']:,} сек ({b['hours']:.2f} ч / {b['days']:.2f} дн.) с {b['start']} по {b['end']}")
    print("=" * 112)

    # 1. Сводка L_max (Сравнение Gross vs Net Usable)
    print(f"\n📌 РАСЧЕТ СКОРРЕКТИРОВАННОГО L_max И ГРАНИЧНЫХ ГОРИЗОНТОВ H_max:")
    headers = [
        "N_target", "Сырой L_max", "ЧИСТЫЙ L_max", "H_max(K=1)", "H_max(K=2)", 
        "H_max(K=3)", "H_max(K=4)", "H_max(K=5)", "Поправка гэпа"
    ]
    print(f"{headers[0]:<9} | {headers[1]:<12} | {headers[2]:<13} | {headers[3]:<10} | {headers[4]:<10} | {headers[5]:<10} | {headers[6]:<10} | {headers[7]:<10} | {headers[8]:<14}")
    print("-" * 112)

    for n in n_targets:
        l_max_gross = gross_train // n
        l_max_net = n_train_usable // n
        h1 = l_max_net // 1
        h2 = l_max_net // 2
        h3 = l_max_net // 3
        h4 = l_max_net // 4
        h5 = l_max_net // 5
        delta_l = l_max_net - l_max_gross
        print(f"{n:<9} | {l_max_gross:>10} с | {l_max_net:>11} с | {h1:>8} с | {h2:>8} с | {h3:>8} с | {h4:>8} с | {h5:>8} с | {delta_l:>+12} с")
    print("-" * 112)

    # 2. Матрица тактических горизонтов для основного N_target
    main_n = n_targets[0]
    main_lmax = n_train_usable // main_n
    print(f"\n📋 МАТРИЦА СОВМЕСТИМОСТИ ТАКТИЧЕСКИХ СЕТОК (H, K) ДЛЯ N_target = {main_n:,} (Чистый L_max = {main_lmax} с):")
    m_headers = ["H (сек)", "K (L/H)", "L = K*H", "N_eff (Чистый)", "Запас к цели", "Статус", "Нужно чистых дней"]
    print(f"{m_headers[0]:<8} | {m_headers[1]:<8} | {m_headers[2]:<8} | {m_headers[3]:<14} | {m_headers[4]:<13} | {m_headers[5]:<18} | {m_headers[6]:<18}")
    print("-" * 112)

    for h in horizons:
        for k in ratios:
            L = h * k
            n_eff = n_train_usable // L
            margin = n_eff / main_n
            req_days_clean = (main_n * L) / 86400.0
            
            if L <= main_lmax:
                status = "✅ ДОСТУПЕН"
            else:
                deficit_days = req_days_clean - days_train_usable
                status = f"❌ НЕ ХВАТАЕТ (+{deficit_days:.1f}д)"
            
            margin_str = f"{margin:.2f}x"
            print(f"{h:<8} | {k:<8} | {L:>5} с  | {n_eff:>11,}  | {margin_str:>11}  | {status:<18} | {req_days_clean:>15.2f} дн.")
    print("-" * 112)


def print_class_distribution(coin: str):
    """Выводит фактическое распределение всех 7 классов в размеченном датасете."""
    labeled_path = BASE_DIR / "data" / "labeled" / coin.lower() / f"{coin.lower()}_labeled.parquet"
    if not labeled_path.exists():
        print(f"\n⚠️ Файл разметки {labeled_path} не найден.")
        return

    print(f"\n📊 ФАКТИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ 7 КЛАССОВ ДЛЯ {coin.upper()} (Только чистые выборки Train и Val):")
    try:
        tbl = pq.read_table(labeled_path)
        df = tbl.to_pandas()
        
        tr_df = df[df['sample_split'] == 'train']
        va_df = df[df['sample_split'] == 'val']
        n_tr = len(tr_df)
        n_va = len(va_df)

        cols = [c for c in df.columns if c.startswith("target_class_7")]
        sample_cols = [c for c in ["target_class_7_h3_k1", "target_class_7_h8_k2", "target_class_7_h25_k2", "target_class_7"] if c in cols]
        if not sample_cols:
            sample_cols = cols[:3]

        for col in sample_cols:
            s_tr = tr_df[col].value_counts().sort_index()
            s_va = va_df[col].value_counts().sort_index()

            print(f"\n  ➤ Таргет: {col} (Чистый Train: {n_tr:,} с | Чистый Val: {n_va:,} с)")
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

    n_targets = [int(x.strip()) for x in args.n_target.split(",") if x.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    ratios = [int(x.strip()) for x in args.ratios.split(",") if x.strip()]

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

    print("\n" + "=" * 112)
    print("🚀 КАЛЬКУЛЯТОР L_max, N_eff С УЧЕТОМ РАЗРЫВОВ СВЯЗИ (is_data_gap), WARM-UP И БУФЕРОВ")
    print(f"   Параметры: N_targets = {n_targets}  |  Train Split = {args.split*100:.0f}%  |  Монеты: {', '.join([c.upper() for c in coins])}")
    print("=" * 112 + "\n")

    found_any = False
    for coin in coins:
        data = load_coin_data(coin, split_ratio=args.split)
        if not data:
            print(f"❌ Данные для монеты {coin.upper()} не найдены!")
            continue
        found_any = True
        analyze_lmax_for_coin(data, n_targets, horizons, ratios)
        if args.show_classes:
            print_class_distribution(coin)
        print("\n")

    if not found_any:
        print("Не найдено данных для анализа!")
        sys.exit(1)


if __name__ == "__main__":
    main()
