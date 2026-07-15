import os
import json
import shutil
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Опционально: отключаем предупреждения TF
tf.get_logger().setLevel('ERROR')

def create_sequences(df, feature_cols, lookback):
    """Генерация 3D тензоров для подачи в модели."""
    X, dates, tickers = [], [], []
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('datetime')
        features = group[feature_cols].values.astype(np.float32)
        dt = group['datetime'].values
        tk = group['ticker'].values
        
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            # Метка времени привязывается к ПОСЛЕДНЕМУ дню окна
            dates.append(dt[i + lookback - 1])
            tickers.append(tk[i + lookback - 1])
            
    return np.array(X), np.array(dates), np.array(tickers)

def main():
    PROCESSED_DIR = Path("data/processed")
    
    # 🌟 АКТУАЛЬНЫЕ КОНФИГУРАЦИИ (ПРЕДИКТОРЫ) 🌟
    CONFIGS = {
        "c3": "2000_2026_1d_3_1",
        "c5": "2000_2026_1d_5_10",
        "c6": "2000_2026_1d_6_1",
        "c10": "2000_2026_1d_10_1",
        "c18": "2000_2026_1d_18_3"
    }
    
    # Базовая конфигурация (откуда возьмем сырые фичи для RL-агента)
    BASE_CONFIG = "c18"
    
    RL_ENV_DIR = PROCESSED_DIR / "2000_2026_1d" / "rl_env"
    CHAMPIONS_DIR = RL_ENV_DIR / "champions"
    OUTPUT_FILE = RL_ENV_DIR / "environment_data.parquet"
    
    if CHAMPIONS_DIR.exists():
        shutil.rmtree(CHAMPIONS_DIR)
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    base_config_dir = PROCESSED_DIR / CONFIGS[BASE_CONFIG]
    if not base_config_dir.exists():
        print(f"❌ Базовая папка {base_config_dir} не найдена!")
        return
        
    folds = sorted([d.name for d in base_config_dir.glob("fold_*") if d.is_dir()])
    all_rl_data = []

    print(f"🚀 СТАРТ: Сборка Песочницы (Strict 1-Year Walk-Forward)")
    print(f"🧩 Интеграция ансамблей: {list(CONFIGS.values())}")
    
    for fold_name in folds:
        fold_year = int(fold_name.split('_')[1])
        print(f"\n=================================================")
        print(f"🗓️ ОБРАБОТКА ФОЛДА: {fold_name} (Таргетный год: {fold_year})")
        print(f"=================================================")
        
        list_of_conf_dfs = []
        skip_fold = False

        for conf_prefix, conf_folder in CONFIGS.items():
            conf_dir = PROCESSED_DIR / conf_folder
            fold_dir = conf_dir / fold_name
            
            # 1. Читаем метаданные конфигурации (Lookback)
            meta_path = conf_dir / "metadata.json"
            if not meta_path.exists(): 
                skip_fold = True; break
            with open(meta_path, 'r') as f:
                lookback = json.load(f)["parameters"]["lookback"]

            # 2. Получаем фичи и данные
            features_json = fold_dir / "artifacts" / "features_selected.json"
            val_parquet = fold_dir / "data" / "val" / "ml_data.parquet"
            alliance_json = fold_dir / "artifacts" / "optimal_alliance.json"
            
            if not (features_json.exists() and val_parquet.exists() and alliance_json.exists()):
                print(f"  ⚠️ [{conf_prefix}] Нет данных или ансамбля. Фолд пропускается.")
                skip_fold = True; break
                
            with open(features_json, 'r') as f:
                feature_cols = json.load(f)["feature_order"]
                
            with open(alliance_json, 'r') as f:
                alliance = json.load(f)

            print(f"  🧠 [{conf_prefix}] Сборка ансамбля (Моделей: {len(alliance['models'])}, Lookback: {lookback})")

            val_df = pd.read_parquet(val_parquet)
            val_df['datetime'] = pd.to_datetime(val_df['datetime'])
            
            # 3. ГЕНЕРАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ИЗ ПОЛНОГО ДАТАСЕТА (ЧТОБЫ НЕ ПОТЕРЯТЬ ЯНВАРЬ)
            X_val, dates_val, tickers_val = create_sequences(val_df, feature_cols, lookback)
            
            if len(X_val) == 0:
                skip_fold = True; break

            # 4. ФИЛЬТРАЦИЯ СТРОГО ПО ГОДУ ФОЛДА ПОСЛЕ ГЕНЕРАЦИИ
            years = pd.DatetimeIndex(dates_val).year
            mask = (years == fold_year)
            
            X_val = X_val[mask]
            dates_val = dates_val[mask]
            tickers_val = tickers_val[mask]
            
            if len(X_val) == 0:
                print(f"  ⚠️ [{conf_prefix}] В году {fold_year} нет валидных данных после сдвига.")
                skip_fold = True; break

            # 5. Прогоняем модели из альянса
            all_probs = []
            for m_path_str in alliance["models"]:
                m_name = Path(m_path_str).name
                m_path = fold_dir / "models" / m_name
                
                if not m_path.exists():
                    print(f"    ❌ Ошибка: Модель ансамбля {m_name} не найдена!")
                    continue
                
                # Копируем модель в архив чемпионов
                shutil.copy2(m_path, CHAMPIONS_DIR / f"{fold_name}_{conf_prefix}_{m_name}")
                
                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(m_path, compile=False)
                
                # Защита размерностей
                expected_features = model.input_shape[2]
                current_features = X_val.shape[2]
                if expected_features > current_features:
                    zeros = np.zeros((X_val.shape[0], X_val.shape[1], expected_features - current_features), dtype=np.float32)
                    X_model = np.concatenate([X_val, zeros], axis=2)
                elif expected_features < current_features:
                    X_model = X_val[:, :, :expected_features]
                else:
                    X_model = X_val
                    
                probs = model.predict(X_model, batch_size=2048, verbose=0)
                all_probs.append(probs)
                
                del model
                gc.collect()

            if not all_probs:
                skip_fold = True; break
                
            # 6. УСРЕДНЯЕМ ВЕРОЯТНОСТИ (Ensemble Consensus)
            ensemble_probs = np.mean(all_probs, axis=0)
            
            conf_df = pd.DataFrame({
                'datetime': dates_val,
                'ticker': tickers_val,
                f'{conf_prefix}_p0': ensemble_probs[:, 0], # Вероятность SL (Шорт/Вниз)
                f'{conf_prefix}_p1': ensemble_probs[:, 1], # Вероятность Hold (Флэт)
                f'{conf_prefix}_p2': ensemble_probs[:, 2]  # Вероятность TP (Лонг/Вверх)
            })
            list_of_conf_dfs.append(conf_df)
            
        if skip_fold or len(list_of_conf_dfs) < len(CONFIGS):
            print(f"⚠️ Фолд {fold_name} пропущен (отсутствует часть конфигураций).")
            continue

        # =========================================================
        # 7. ФОРМИРОВАНИЕ ИТОГОВОГО ВЕКТОРА СОСТОЯНИЙ (STATE SPACE)
        # =========================================================
        print(f"  🔗 Слияние векторов и признаков...")
        
        base_fold_dir = PROCESSED_DIR / CONFIGS[BASE_CONFIG] / fold_name
        
        # Загружаем базовые признаки
        base_features_df = pd.read_parquet(base_fold_dir / "data" / "val" / "ml_data.parquet")
        base_features_df['datetime'] = pd.to_datetime(base_features_df['datetime'])
        base_features_df = base_features_df[base_features_df['datetime'].dt.year == fold_year]
        
        # Удаляем таргеты (утечка будущего)
        leak_cols = ['label', 'target_return', 'target_tp', 'target_sl']
        base_features_df = base_features_df.drop(columns=[c for c in leak_cols if c in base_features_df.columns])
        
        # Загружаем базовые OHLCV
        base_ohlcv_df = pd.read_csv(base_fold_dir / "data" / "val" / "dataset.csv")[['datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        base_ohlcv_df['datetime'] = pd.to_datetime(base_ohlcv_df['datetime'])
        base_ohlcv_df = base_ohlcv_df[base_ohlcv_df['datetime'].dt.year == fold_year]
        
        # Аккуратно клеим всё вместе (удаляя дублирующиеся колонки через суффиксы)
        fold_rl_df = pd.merge(base_ohlcv_df, base_features_df, on=['datetime', 'ticker'], suffixes=('', '_drop'))
        fold_rl_df = fold_rl_df.loc[:, ~fold_rl_df.columns.str.endswith('_drop')]
        
        for conf_df in list_of_conf_dfs:
            fold_rl_df = pd.merge(fold_rl_df, conf_df, on=['datetime', 'ticker'], how='inner')
            
        all_rl_data.append(fold_rl_df)
        print(f"  ✅ Фолд {fold_name} успешно упакован! Строк: {len(fold_rl_df)}")

    if not all_rl_data:
        print("\n❌ Критическая ошибка: Не удалось собрать данные ни для одного фолда.")
        return

    print("\n🧩 Склеиваем весь Walk-Forward в финальную симуляцию...")
    final_env_df = pd.concat(all_rl_data, ignore_index=True)
    final_env_df = final_env_df.sort_values(['ticker', 'datetime']).reset_index(drop=True)
    
    # Сохраняем результат
    final_env_df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n🎉 ПЕСОЧНИЦА АГЕНТА УСПЕШНО СОБРАНА!")
    print(f"   Файл: {OUTPUT_FILE}")
    print(f"   Количество строк (Торговых дней): {len(final_env_df)}")
    print(f"   Размерность State Space: {len(final_env_df.columns)} колонок")

if __name__ == "__main__":
    main()