import sys
import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def reduce_multicollinearity(df, feature_cols, threshold=0.85):
    print(f"📉 Удаление мультиколлинеарности (порог {threshold})...")
    sample_df = df.sample(n=min(len(df), 50000), random_state=42)
    
    # 🚀 СУПЕР-УСКОРЕНИЕ (в 20-50 раз):
    # Pandas считает Spearman в 1 ядро. Математически Spearman = Pearson от рангов.
    # Ранжируем данные (мгновенно) и считаем Пирсона (многопоточно через C/BLAS).
    print("  ⚡ Запуск многопоточной матрицы корреляций (Rank+Pearson)...")
    ranked_df = sample_df[feature_cols].astype(np.float32).rank()
    corr_matrix = ranked_df.corr(method='pearson').abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    retained = [c for c in feature_cols if c not in to_drop]
    
    print(f"  ✅ Осталось признаков: {len(retained)} из {len(feature_cols)}")
    return retained

def get_dynamic_feature_importance(df, feature_cols, target_col='label', cumulative_threshold=0.95):
    print(f"🧠 LightGBM оценивает важность {len(feature_cols)} признаков...")
    sample_df = df.sample(n=min(len(df), 100000), random_state=42)
    
    # 🚀 УСКОРЕНИЕ ПАМЯТИ: Снижаем точность до float32. 
    # Это разгружает шину памяти и ускоряет градиентный бустинг.
    X = sample_df[feature_cols].astype(np.float32)
    y = (sample_df[target_col] + 1).astype(int)
    
    n_estimators = 100
    
    # Агрессивные параметры для скорости (n_jobs=-1 задействует все ядра CPU)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=64,
        max_bin=255,       # Оптимизация построения гистограмм для скорости
        random_state=42, 
        n_jobs=4,         #-1 Включаем все доступные потоки
        verbose=-1
    )
    
    # Обертка tqdm для визуализации процесса обучения (100 деревьев)
    with tqdm(total=n_estimators, desc="  🌲 Построение деревьев", unit=" итер", ncols=100, mininterval=1.0) as pbar:
        # Передаем callback, который будет обновлять полосу после каждого дерева
        model.fit(X, y, callbacks=[lambda env: pbar.update(1)])
    
    importance = model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False)
    
    imp_df['cumulative_importance'] = imp_df['importance'].cumsum() / imp_df['importance'].sum()
    selected_features = imp_df[imp_df['cumulative_importance'] <= cumulative_threshold]['feature'].tolist()
    
    if not selected_features:
        selected_features = imp_df['feature'].tolist()[:10]
        
    return selected_features, imp_df

def plot_importance(imp_df, top_n, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=imp_df.head(top_n))
    plt.title(f'Топ-{top_n} признаков (LightGBM)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    train_path = Path(args.input)
    if not train_path.exists():
        print(f"❌ Датасет не найден: {train_path}"); return

    print(f"🚀 Динамический отбор признаков...")
    df = pd.read_parquet(train_path)
    
    exclude_cols = {'datetime', 'ticker', 'target_tp', 'target_sl', 'target_return', 'label', 'open', 'high', 'low', 'close', 'volume'}
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]
    
    if not args.skip_corr:
        feature_cols = reduce_multicollinearity(df, feature_cols, threshold=args.corr_threshold)
        
    top_features, imp_df = get_dynamic_feature_importance(df, feature_cols, cumulative_threshold=args.cum_threshold)
    
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"feature_order": top_features}, f, indent=2)
        
    if args.out_plot:
        out_plot = Path(args.out_plot)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_importance(imp_df, len(top_features), out_plot)
        
    print(f"✅ Отбор завершен! Выбрано: {len(top_features)} признаков. Сохранено в {out_json.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Путь к ml_data.parquet (Train)")
    parser.add_argument("--out_json", type=str, required=True, help="Путь сохранения features_selected.json")
    parser.add_argument("--out_plot", type=str, help="Путь сохранения графика")
    parser.add_argument("--cum_threshold", type=float, default=0.95)
    parser.add_argument("--corr_threshold", type=float, default=0.85)
    parser.add_argument("--skip_corr", action="store_true")
    args = parser.parse_args()
    main(args)