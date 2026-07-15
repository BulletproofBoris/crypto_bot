import os
import json
import argparse
import pandas as pd
from pathlib import Path
from colorama import init, Fore, Style

# Инициализация цветов для консоли
init(autoreset=True)

# Ожидаемые файлы для каждого полностью готового фолда (в порядке конвейера)
PIPELINE_FILES = [
    ("data/train/dataset.csv", "csv", "Train Build"),
    ("data/train/labels.csv", "csv", "Train Labels"),
    ("data/train/ml_data.parquet", "parquet", "Train Features"),
    ("artifacts/feature_cols.json", "json", "Feature Freezing"),
    ("artifacts/quantiles_winsor.json", "json", "Winsorization Boundaries"),
    ("data/val/dataset.csv", "csv", "Val Build"),
    ("data/val/labels.csv", "csv", "Val Labels"),
    ("data/val/ml_data.parquet", "parquet", "Val Features"),
    ("artifacts/features_selected.json", "json", "Feature Selection"),
    ("data/train/data.tfrecord", "tfrecord", "Train TFRecords"),
    ("data/val/data.tfrecord", "tfrecord", "Val TFRecords")
]

def check_file_integrity(filepath, file_type):
    """Возвращает (is_ok, reason, can_be_healed)"""
    if not filepath.exists():
        return False, "Отсутствует", False
    
    if filepath.stat().st_size == 0:
        return False, "0 байт (Зомби-файл)", True

    # Пытаемся прочитать структуру тяжелых файлов
    try:
        if file_type == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
        elif file_type == "parquet":
            # Читаем только метаданные (быстро), чтобы убедиться, что файл не оборван
            pd.read_parquet(filepath, columns=[])
    except Exception as e:
        return False, f"Битая структура ({type(e).__name__})", True

    return True, "OK", False

def main(auto_heal=False):
    base_dir = Path("data/processed")
    if not base_dir.exists():
        print(f"{Fore.RED}❌ Папка {base_dir} не найдена!{Style.RESET_ALL}")
        return

    configs = sorted([d for d in base_dir.iterdir() if d.is_dir() and "1d" in d.name])
    
    total_folds = 0
    completed_folds = 0
    corrupted_files_to_heal = []

    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}🩺 АУДИТ И ВОССТАНОВЛЕНИЕ КОНВЕЙЕРА ДАННЫХ{Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")

    for config_dir in configs:
        print(f"\n{Fore.YELLOW}📂 Конфигурация: {config_dir.name}{Style.RESET_ALL}")
        
        # Проверка базового кэша
        raw_cache = config_dir / "raw_combined.csv"
        ok, reason, healable = check_file_integrity(raw_cache, "csv")
        if not ok:
            print(f"  ⚠️ Базовый кэш (raw_combined.csv): {Fore.RED}{reason}{Style.RESET_ALL}")
            if healable: corrupted_files_to_heal.append(raw_cache)

        folds = sorted([d for d in config_dir.glob("fold_*") if d.is_dir()])
        
        for fold in folds:
            total_folds += 1
            fold_is_perfect = True
            missing_steps = []

            for rel_path, ftype, step_name in PIPELINE_FILES:
                full_path = fold / rel_path
                ok, reason, healable = check_file_integrity(full_path, ftype)
                
                if not ok:
                    fold_is_perfect = False
                    missing_steps.append((step_name, reason, full_path, healable))

            if fold_is_perfect:
                completed_folds += 1
                # Раскомментируйте строку ниже, если хотите видеть логи идеальных фолдов
                # print(f"  ✅ {fold.name}: Полностью готов")
            else:
                print(f"  ⚠️ {fold.name}: {Fore.RED}Не завершен{Style.RESET_ALL}")
                for step_name, reason, full_path, healable in missing_steps:
                    color = Fore.RED if healable else Fore.MAGENTA
                    print(f"      - {step_name} ({full_path.name}): {color}{reason}{Style.RESET_ALL}")
                    if healable:
                        corrupted_files_to_heal.append(full_path)

    print(f"\n{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"📊 ИТОГИ АУДИТА:")
    print(f"   Всего фолдов найдено: {total_folds}")
    print(f"   Полностью готово:     {Fore.GREEN}{completed_folds}{Style.RESET_ALL}")
    print(f"   Требуют доработки:    {Fore.YELLOW}{total_folds - completed_folds}{Style.RESET_ALL}")
    print(f"   Поврежденных файлов:  {Fore.RED}{len(corrupted_files_to_heal)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")

    if corrupted_files_to_heal:
        if auto_heal:
            do_heal = 'y'
        else:
            print("\nНайдены поврежденные (0 байт или битые) файлы. Их необходимо удалить,")
            print("чтобы скрипт подготовки смог пересоздать их заново.")
            do_heal = input(f"{Fore.YELLOW}Удалить битые файлы сейчас? (y/n): {Style.RESET_ALL}").strip().lower()

        if do_heal == 'y':
            healed = 0
            for f in corrupted_files_to_heal:
                try:
                    f.unlink()
                    healed += 1
                except Exception as e:
                    print(f"Не удалось удалить {f}: {e}")
            print(f"\n{Fore.GREEN}✅ Успешно удалено битых файлов: {healed}.{Style.RESET_ALL}")
            print("Теперь вы можете запустить ./start_data_swarm.sh (БЕЗ флага --force), и пайплайн продолжится с места обрыва!")
    else:
        if total_folds > completed_folds:
            print(f"\n{Fore.GREEN}✅ Битых файлов нет!{Style.RESET_ALL} Просто запустите ./start_data_swarm.sh (БЕЗ флага --force),")
            print("чтобы скрипт дособрал недостающие файлы.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto_heal", action="store_true", help="Автоматически удалять битые файлы без запроса")
    args = parser.parse_args()
    main(args.auto_heal)