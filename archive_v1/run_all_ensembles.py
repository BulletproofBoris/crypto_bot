import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    # 1. Настраиваем прием аргументов из командной строки
    parser = argparse.ArgumentParser(description="Массовый запуск ансамблирования по всем фолдам.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Путь к директории конфигурации (например, data/processed/2000_2026_1d_10_1)")
    parser.add_argument("--max_k", type=int, default=20, help="Макс. количество моделей в ансамбле (по умолчанию: 20)")
    
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    max_k = args.max_k

    # 2. Проверяем директорию и ищем фолды
    path_obj = Path(dataset_dir)
    if not path_obj.exists():
        print(f"❌ Ошибка: Директория {dataset_dir} не найдена!")
        sys.exit(1)

    folds = sorted([d.name for d in path_obj.glob("fold_*") if d.is_dir()])
    
    if not folds:
        print(f"⚠️ Фолды (вида fold_*) не найдены внутри {dataset_dir}")
        sys.exit(1)

    print(f"🔍 Найдено фолдов: {len(folds)} в конфигурации {dataset_dir}\n")

    # 3. Итерируемся по фолдам и запускаем процесс
    for fold in folds:
        print("\n" + "="*80)
        print(f"🚀 ЗАПУСК АНСАМБЛИРОВАНИЯ: {fold}")
        print("="*80)
        
        # Собираем команду для запуска
        # sys.executable гарантирует, что будет использован Python из текущего venv
        command = [
            sys.executable, "-m", "_tools.ensemble_predictor",
            "--dataset_dir", dataset_dir,
            "--fold", fold,
            "--max_k", str(max_k)
        ]
        
        # Выполняем скрипт, вывод будет идти прямо в терминал
        subprocess.run(command)

if __name__ == "__main__":
    main()