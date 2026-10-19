import argparse
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def process_and_write(df, lookback, output_path, feature_cols):
    writer = tf.io.TFRecordWriter(str(output_path))
    samples_count = 0
    
    # Группируем по тикерам, чтобы случайно не склеить конец Сбербанка с началом Газпрома
    for ticker, group in tqdm(df.groupby('ticker'), desc=f"Запись {Path(output_path).name}", ncols=100, mininterval=2.0):
        group = group.sort_values('datetime')
        features = group[feature_cols].values.astype(np.float32)
        
        # Сдвигаем метки из [-1, 0, 1] в [0, 1, 2] для CrossEntropy loss
        labels = (group['label'].values + 1).astype(int)
        
        # Нарезаем "окна"
        for i in range(len(features) - lookback):
            seq = features[i : i + lookback]
            target = labels[i + lookback - 1] # Берем метку последнего дня окна
            
            # Сериализуем тензор
            seq_bytes = tf.io.serialize_tensor(tf.constant(seq, dtype=tf.float32)).numpy()
            
            feature = {
                'sequence': _bytes_feature(seq_bytes),
                'target': _int64_feature(target)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            samples_count += 1
            
    writer.close()
    return samples_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Путь к ml_data.parquet")
    parser.add_argument('--output', type=str, required=True, help="Путь к выходному .tfrecord")
    parser.add_argument('--artifacts_dir', type=str, required=True, help="Папка artifacts фолда")
    parser.add_argument('--phase', type=str, required=True)
    
    # Заглушки от оркестратора
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--timeframe', type=str)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--tp', type=float)
    parser.add_argument('--sl', type=float)
    parser.add_argument('--percentile', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--cum_threshold', type=float)
    parser.add_argument('--corr_threshold', type=float)
    
    args = parser.parse_args()

    artifacts_path = Path(args.artifacts_dir)
    features_json = artifacts_path / "features_selected.json"
    
    if not features_json.exists():
        print(f"❌ Файл {features_json.name} не найден! Отбор признаков не выполнен.")
        return

    with open(features_json, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f).get("feature_order", [])

    if not feature_cols:
        print(f"❌ Список отобранных признаков пуст!")
        return

    print(f"\n📦 [TFRECORDS] Упаковка {args.phase.upper()} выборки...")
    print(f"   Окно (lookback): {args.lookback}")
    print(f"   Отобрано признаков: {len(feature_cols)}")

    df = pd.read_parquet(args.input)
    
    # Защита: если вдруг колонка потерялась, заполняем нулями, чтобы не ломать тензор
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for c in missing: df[c] = 0.0
            
    samples_count = process_and_write(df, args.lookback, args.output, feature_cols)
    
    print(f"✅ Сохранено {samples_count} 3D-тензоров в {Path(args.output).name}")

if __name__ == "__main__":
    main()