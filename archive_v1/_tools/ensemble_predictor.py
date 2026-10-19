import os
import argparse
import json
import numpy as np
import pandas as pd
import itertools
import multiprocessing
from pathlib import Path
from sklearn.metrics import classification_report
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Отключаем спам TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def create_sequences(df, feature_cols, lookback):
    X, y, dates, tickers = [], [], [], []
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('datetime')
        features = group[feature_cols].values.astype(np.float32)
        labels = (group['label'].values + 1).astype(int)
        dt = group['datetime'].values
        tk = group['ticker'].values
        
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback - 1])
            dates.append(dt[i + lookback - 1])
            tickers.append(tk[i + lookback - 1])
            
    return np.array(X), np.array(y), np.array(dates), np.array(tickers)

def load_top_models(models_dir, top_n=10):
    models_info = []
    for json_file in Path(models_dir).glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                model_name = meta.get("model_name")
                val_loss = meta.get("metrics", {}).get("val_loss", float('inf'))
                
                arch = meta.get("arch", model_name.split('_')[0] if model_name else "legacy")
                
                model_path = Path(models_dir) / model_name
                if model_path.exists():
                    models_info.append({"path": str(model_path), "loss": val_loss, "meta": meta, "arch": arch})
        except Exception as e:
            pass
            
    models_info.sort(key=lambda x: x["loss"])
    return models_info[:top_n]

# === НОВОЕ: Изолированная функция для пула процессов ===
def evaluate_combinations_chunk(combos, all_probs, y_val, y_val_smoothed):
    results = []
    for combo in combos:
        # Ускоренное усреднение:
        combo_probs = np.mean([all_probs[i] for i in combo], axis=0)
        
        # Ускоренный расчет Accuracy (через чистый numpy)
        ensemble_preds = np.argmax(combo_probs, axis=1)
        acc = np.mean(y_val == ensemble_preds)
        
        # Ускоренный расчет Smoothed Loss (без участия TensorFlow)
        # Ограничиваем пределы (clip), чтобы логарифм от 0 не выдал NaN
        y_pred = np.clip(combo_probs, 1e-7, 1.0 - 1e-7)
        loss = -np.sum(y_val_smoothed * np.log(y_pred)) / y_pred.shape[0]
        
        combo_names = "+".join([str(i+1) for i in combo])
        
        results.append({
            "combo": combo_names,
            "k": len(combo),
            "accuracy": float(acc),
            "log_loss": float(loss),
            "indices": combo 
            # 🛑 ПАМЯТЬ СПАСЕНА: Мы больше не сохраняем массив preds для всех 32 000 связок!
        })
    return results

def main(args):
    dataset_dir = Path(args.dataset_dir)
    fold_dir = dataset_dir / args.fold
    models_dir = fold_dir / "models"
    artifacts_dir = fold_dir / "artifacts"
    val_parquet = fold_dir / "data" / "val" / "ml_data.parquet"
    
    print(f"🚀 Запуск СУПЕР-АНАЛИЗАТОРА Ансамблей ({args.fold})")
    
    with open(dataset_dir / "metadata.json", 'r', encoding='utf-8') as f:
        lookback = json.load(f)["parameters"]["lookback"]
        
    with open(artifacts_dir / "features_selected.json", 'r', encoding='utf-8') as f:
        feature_cols = json.load(f).get("feature_order", [])
        
    print(f"📦 Загрузка валидационных данных...")
    df_val = pd.read_parquet(val_parquet)
    X_val, y_val, dates_val, tickers_val = create_sequences(df_val, feature_cols, lookback)
    
    # === НОВОЕ: Готовим математику для быстрого Smoothed Loss ===
    num_classes = 3
    alpha = 0.1
    y_val_one_hot = np.eye(num_classes)[y_val]
    # Формула сглаживания Keras: y_smooth = y_one_hot * (1 - alpha) + alpha / num_classes
    y_val_smoothed = y_val_one_hot * (1.0 - alpha) + (alpha / num_classes)

    max_k = min(args.max_k, 30)
    top_models_info = load_top_models(models_dir, top_n=max_k)
    actual_k = len(top_models_info)
    
    if actual_k == 0:
        print("❌ Модели не найдены!")
        return
        
    print(f"\n🏆 Найдено {actual_k} элитных моделей. Генерация предиктов...")
    
    all_probs = []
    for i, info in enumerate(top_models_info, 1):
        arch_label = info['arch'].upper()
        print(f"  ⏳ Модель {i:<2} [{arch_label:<10}] (Indiv. Smoothed Loss: {info['loss']:.4f})...")
        model = tf.keras.models.load_model(info['path'], compile=False)
        probs = model.predict(X_val, batch_size=2048, verbose=0)
        all_probs.append(probs)
        del model
        tf.keras.backend.clear_session()
        
    # Генерируем список всех комбинаций индексов
    all_combos = []
    for k in range(1, actual_k + 1):
        all_combos.extend(list(itertools.combinations(range(actual_k), k)))
        
    total_combinations = len(all_combos)
    print(f"\n🤝 Просчет всех связок ({total_combinations} комбинаций)...")
    
    # Нарезаем комбинации на чанки (по пакетам), чтобы не передавать их по одной в пулы
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, total_combinations // (num_cores * 4))
    combo_chunks = [all_combos[i:i + chunk_size] for i in range(0, total_combinations, chunk_size)]

    results = []
    
    # === НОВОЕ: Многопроцессорный параллелизм ===
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Отправляем чанки в пулы воркеров
        futures = {
            executor.submit(evaluate_combinations_chunk, chunk, all_probs, y_val, y_val_smoothed): chunk 
            for chunk in combo_chunks
        }
        
        with tqdm(total=total_combinations, desc="Анализ ансамблей", unit="комб", ncols=80) as pbar:
            for future in as_completed(futures):
                chunk_res = future.result()
                results.extend(chunk_res)
                pbar.update(len(chunk_res))

    # Сортируем результаты по Сглаженному Loss, затем тай-брейк по Accuracy (твой фикс!)
    results.sort(key=lambda x: (round(x['log_loss'], 4), -x['accuracy']))
    
    print("\n" + "="*70)
    print(f"{'Комбинация моделей':<30} | {'Кол-во (K)':<12} | {'Acc (%)':<10} | {'Smoothed Loss':<10}")
    print("-" * 70)
    
    for res in results[:30]:
        marker = "⭐ БЕСТСЕЛЛЕР" if res == results[0] else ""
        print(f"[{res['combo']:<28}] | K={res['k']:<10} | {res['accuracy']*100:<10.2f} | {res['log_loss']:<10.4f} {marker}")
    print("="*70)
    
    best_result = results[0]
    print(f"\n✅ Оптимальный альянс: Модели [{best_result['combo']}] (Smoothed Loss: {best_result['log_loss']:.4f})")
    
    alliance_archs = [top_models_info[i]['arch'].upper() for i in best_result['indices']]
    arch_summary = {}
    for a in alliance_archs:
        arch_summary[a] = arch_summary.get(a, 0) + 1
    summary_str = ", ".join([f"{count}x {arch}" for arch, count in arch_summary.items()])
    print(f"🧬 Состав команды: {summary_str}")
    
    # === НОВОЕ: Восстанавливаем предикты ТОЛЬКО для лучшего ансамбля ===
    best_combo_probs = np.mean([all_probs[i] for i in best_result['indices']], axis=0)
    best_preds = np.argmax(best_combo_probs, axis=1)
    
    print("\nДетальный отчет (по argmax, чисто для справки):")
    print(classification_report(y_val, best_preds, target_names=['SL (-1)', 'Hold (0)', 'TP (+1)']))

    optimal_models_files = [top_models_info[i]['path'] for i in best_result['indices']]
    
    alliance_config = {
        "fold": args.fold,
        "optimal_k": best_result['k'],
        "ensemble_smoothed_loss": float(best_result['log_loss']),
        "composition": arch_summary,
        "models": optimal_models_files
    }
    
    alliance_file = artifacts_dir / "optimal_alliance.json"
    with open(alliance_file, 'w', encoding='utf-8') as f:
        json.dump(alliance_config, f, indent=4, ensure_ascii=False)
        
    print(f"\n💾 Состав оптимального альянса сохранен в: {alliance_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Многопоточный Анализатор лучших комбинаций ансамбля")
    parser.add_argument("--dataset_dir", type=str, default="data/processed/2000_2026_1d_6_1")
    parser.add_argument("--fold", type=str, default="fold_2010")
    parser.add_argument("--max_k", type=int, default=20, help="Сколько топ-моделей взять для перебора (Макс 30)")
    args = parser.parse_args()
    
    main(args)