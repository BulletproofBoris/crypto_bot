import os
# Отключаем GPU для аудита, чтобы скрипт не отбирал память у запущенного процесса обучения
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tensorflow as tf

def audit_dataset(config_name=None):
    base_path = Path("data/processed")
    
    if not base_path.exists():
        print(f"❌ Базовая директория {base_path} не найдена.")
        return

    dataset_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    if not dataset_dirs:
        print("❌ Данные не найдены в", base_path)
        return
    
    # Фильтрация по конкретной конфигурации, если она указана
    if config_name:
        dataset_dirs = [d for d in dataset_dirs if d.name == config_name]
        if not dataset_dirs:
            print(f"❌ Конфигурация '{config_name}' не найдена.")
            print("Доступные конфигурации:")
            for d in sorted([d for d in base_path.iterdir() if d.is_dir()]):
                print(f"  - {d.name}")
            return
            
    report_path = Path("data_audit_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as report:
        for dataset_dir in dataset_dirs:
            report.write(f"{'='*60}\nАУДИТ ДАТАСЕТА: {dataset_dir.name}\n{'='*60}\n")

            folds = sorted([d for d in dataset_dir.glob("fold_*") if d.is_dir()])
            
            if not folds:
                report.write("  ⚠️ Фолды не найдены.\n\n")
                continue
            
            for fold in folds:
                report.write(f"\n📁 ФОЛД: {fold.name}\n{'-'*30}\n")
                
                # --- Отобранные признаки ---
                features_json_path = fold / "artifacts" / "features_selected.json"
                selected_features = []
                if features_json_path.exists():
                    with open(features_json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        selected_features = data.get("feature_order", [])
                    
                    report.write(f"  🎯 ОТОБРАННЫЕ ПРИЗНАКИ: {len(selected_features)} шт.\n")
                else:
                    report.write(f"  ⚠️ ВНИМАНИЕ: features_selected.json не найден!\n")
                
                train_dates = set()
                
                for phase in ["train", "val"]:
                    phase_dir = fold / "data" / phase
                    parquet_path = phase_dir / "ml_data.parquet"
                    tfrecord_path = phase_dir / "data.tfrecord"
                    
                    if not parquet_path.exists(): continue
                    
                    df = pd.read_parquet(parquet_path)
                    if phase == "train":
                        train_dates = set(df['datetime'].unique())
                    
                    report.write(f"  🔹 {phase.upper()} (Parquet: {len(df)} строк):\n")
                    
                    # Баланс классов в Parquet
                    counts = df['label'].value_counts(normalize=True).sort_index()
                    for lbl, pct in counts.items():
                        name = {1.0: "TP (+1)", -1.0: "SL (-1)", 0.0: "Hold (0)"}.get(lbl, "Unknown")
                        report.write(f"    {name:<10}: {pct*100:>6.2f}%\n")
                    
                    # --- АУДИТ TFRECORDS ---
                    if tfrecord_path.exists():
                        report.write(f"    📦 Проверка TFRecord ({tfrecord_path.name}):\n")
                        try:
                            raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
                            
                            # Описательная структура для парсинга
                            feature_description = {
                                'sequence': tf.io.FixedLenFeature([], tf.string),
                                'target': tf.io.FixedLenFeature([], tf.int64),
                            }

                            for raw_record in raw_dataset.take(1):
                                example = tf.io.parse_single_example(raw_record, feature_description)
                                seq = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
                                
                                report.write(f"      ✅ Читаемость: OK\n")
                                report.write(f"      📐 Shape тензора: {seq.shape} (Lookback x Features)\n")
                                
                                # Проверка на соответствие количества признаков
                                if selected_features and seq.shape[1] != len(selected_features):
                                    report.write(f"      ❌ ОШИБКА: Кол-во признаков в тензоре ({seq.shape[1]}) != отобранным ({len(selected_features)})\n")

                            # Быстрый подсчет примеров
                            example_count = sum(1 for _ in raw_dataset)
                            report.write(f"      🔢 Всего примеров (окон): {example_count}\n")
                            
                        except Exception as e:
                            report.write(f"      ❌ ОШИБКА ЧТЕНИЯ: {str(e)}\n")
                    else:
                        report.write(f"    ⚠️ TFRecord файл отсутствует.\n")

                    # Статистика фичей
                    if selected_features and not df.empty:
                        f_mean = df[selected_features].mean().mean()
                        f_std = df[selected_features].std().mean()
                        report.write(f"    📊 Физика фичей: Mean={f_mean:.4f}, Std={f_std:.4f}\n")

                    # Утечка данных
                    if phase == "val":
                        val_dates = set(df['datetime'].unique())
                        overlap = train_dates.intersection(val_dates)
                        if overlap:
                            report.write(f"    ❌ УТЕЧКА БУДУЩЕГО: {len(overlap)} общих дней!\n")
                        else:
                            report.write(f"    ✅ УТЕЧКА БУДУЩЕГО: Чистый тест\n")
            
            report.write("\n") # Отступ между конфигурациями

    print(f"✅ Расширенный аудит завершен: {report_path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инструмент аудита собранных датасетов")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Имя конкретной папки (например: 2000_2026_1d_60_10). Если не указано, проверяются все.", 
        default=None
    )
    args = parser.parse_args()
    
    audit_dataset(args.config)