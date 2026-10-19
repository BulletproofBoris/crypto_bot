import tensorflow as tf
import numpy as np
from collections import Counter

def parse_tfrecord_fn(example, lookback, n_features):
    """
    Парсинг сырого бинарного примера из TFRecord обратно в тензоры.
    Входные данные строго приводятся к float32 для стабильности Keras (BFLOAT16 включится позже внутри слоев).
    """
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string), 
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    
    # Декодируем последовательность
    sequence = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
    sequence.set_shape([lookback, n_features])
    
    # One-hot кодирование таргета (Вниз, Флэт, Вверх)
    label = tf.one_hot(example['target'], depth=3)
    label.set_shape([3])
    
    return sequence, label

def load_tfrecord_dataset(filepath, batch_size, seq_len, n_features, is_training=True):
    """
    Создает высокопроизводительный пайплайн данных.
    """
    # 1. Многопоточное чтение с диска
    dataset = tf.data.TFRecordDataset(filepath, num_parallel_reads=tf.data.AUTOTUNE)
    
    # 2. Асинхронный парсинг
    # deterministic=False - разрешаем отдавать готовые элементы в любом порядке (ускоряет работу на 20-30%)
    dataset = dataset.map(
        lambda x: parse_tfrecord_fn(x, seq_len, n_features),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    if is_training:
        # 3. Кешируем распарсенные тензоры в ОЗУ, чтобы не дергать диск каждую эпоху
        dataset = dataset.cache() 
        # 4. Перемешиваем только ПОСЛЕ кэширования для максимальной скорости
        dataset = dataset.shuffle(buffer_size=8192, reshuffle_each_iteration=True)
    else:
        # Для валидации перемешивание не нужно, но кэш обязателен
        dataset = dataset.cache()
        
    # 5. Разбивка на батчи (drop_remainder=True защищает от плавающего размера батча на последнем шаге)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    
    # 6. Prefetch: Процессор готовит следующий батч в фоновом режиме, пока GPU считает текущий
    dataset = dataset.prefetch(tf.data.AUTOTUNE) 
    
    return dataset

def count_tfrecord_samples(filepath):
    """
    Быстрый подсчет количества записей в TFRecord.
    Нужен для математического расчета идеального размера батча.
    """
    c = 0
    for _ in tf.data.TFRecordDataset(filepath):
        c += 1
    return c

def compute_class_weights_fast(filepath):
    """
    Сканирует TFRecord, подсчитывает баланс классов и вычисляет веса 
    для компенсации несбалансированности рынка (например, если Флэт встречается чаще).
    """
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Читаем ТОЛЬКО таргеты, чтобы не тратить время на парсинг тяжелых матриц
    feature_description = {
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    
    def parse_target(example):
        return tf.io.parse_single_example(example, feature_description)['target']
        
    targets_dataset = dataset.map(parse_target, num_parallel_calls=tf.data.AUTOTUNE)
    
    class_counts = Counter()
    for target in targets_dataset.as_numpy_iterator():
        class_counts[target] += 1
        
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for cls, count in class_counts.items():
        # Формула взвешивания: (Total) / (Num_Classes * Count)
        # Если класс редкий, его вес будет больше 1.0. Если частый - меньше 1.0.
        class_weights[cls] = float(total) / (num_classes * count)
        
    return class_weights