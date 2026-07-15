import os
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from _core.feature_generator import create_individual_features, create_cross_sectional_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--phase', type=str, required=True, choices=['train', 'val'])
    parser.add_argument('--artifacts_dir', type=str, required=True)
    parser.add_argument('--start_date', type=str, help="Реальная дата начала фазы (без буфера)")
    
    parser.add_argument('--timeframe', type=str)
    parser.add_argument('--lookback', type=int)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--tp', type=float)
    parser.add_argument('--sl', type=float)
    parser.add_argument('--percentile', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--auto', action='store_true')
    
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"\n⚙️ [{args.phase.upper()}] Инициализация расчета...")

    macro_mapping = {
        'MACRO_USDRUB': 'usdrub_close', 'MACRO_BRENT': 'brent_close',
        'MACRO_SP500': 'sp500_close', 'MACRO_IMOEX': 'imoex_close', 'MACRO_VIX': 'vix_close'
    }
    
    external_data = pd.DataFrame()
    for m_ticker, col_name in macro_mapping.items():
        m_df = df[df['ticker'] == m_ticker][['datetime', 'close']].copy()
        if not m_df.empty:
            m_df.rename(columns={'close': col_name}, inplace=True)
            m_df.set_index('datetime', inplace=True)
            if external_data.empty: external_data = m_df
            else: external_data = external_data.join(m_df, how='outer')

    df_stocks = df[~df['ticker'].str.startswith('MACRO_')].copy()

    all_data_dict = {}
    for ticker, group in df_stocks.groupby('ticker'):
        if len(group) < 260:
            continue
        all_data_dict[ticker] = group.set_index('datetime')
        
    if not all_data_dict:
        print("❌ Ошибка: Нет тикеров с достаточной историей.")
        return

    cs_features = create_cross_sectional_features(all_data_dict)
    processed_dfs = []
    
    for ticker, group_idx in tqdm(all_data_dict.items(), desc="Индивидуальные фичи", ncols=100, mininterval=2.0):
        res_ticker, df_with_features = create_individual_features((ticker, group_idx), external_data, cs_features)
        if 'datetime' not in df_with_features.columns:
            df_with_features.reset_index(inplace=True)
        df_with_features['ticker'] = res_ticker
        processed_dfs.append(df_with_features)
        
    df_features = pd.concat(processed_dfs, ignore_index=True)
    
    # =========================================================
    # ФИКС: ЗАМОРОЗКА КОЛОНОК (Feature Freezing)
    # =========================================================
    exclude_cols = ['datetime', 'ticker', 'target_tp', 'target_sl', 'target_return', 'label', 'open', 'high', 'low', 'close', 'volume']
    features_list_file = Path(args.artifacts_dir) / "feature_cols.json"
    
    if args.phase == 'train':
        # В Train мы определяем и сохраняем эталонный список колонок (по алфавиту)
        feature_cols = sorted([c for c in df_features.columns if c not in exclude_cols])
        with open(features_list_file, 'w', encoding='utf-8') as f:
            json.dump(feature_cols, f)
    elif args.phase == 'val':
        # В Val мы строго загружаем эталонный список
        if not features_list_file.exists():
            raise FileNotFoundError("feature_cols.json не найден! Прогони Train.")
        with open(features_list_file, 'r', encoding='utf-8') as f:
            feature_cols = json.load(f)
            
        # Восстанавливаем недостающие колонки нулями (если индикатор упал в Val)
        missing_cols = set(feature_cols) - set(df_features.columns)
        if missing_cols:
            print(f"  ⚠️ Восстановление {len(missing_cols)} пропущенных колонок (заполняем нулями).")
            for c in missing_cols:
                df_features[c] = 0.0

    # Оставляем только нужные колонки для нормализации
    print(f"✂️ Winsorization (0.1% - 99.9%) для {len(feature_cols)} признаков...")
    quantiles_file = Path(args.artifacts_dir) / "quantiles_winsor.json"

    if args.phase == 'train':
        quantiles_dict = {}
        for col in feature_cols:
            lower = float(df_features[col].quantile(0.001))
            upper = float(df_features[col].quantile(0.999))
            quantiles_dict[col] = {"lower": lower, "upper": upper}
            df_features[col] = df_features[col].clip(lower=lower, upper=upper)
        
        # Сохраняем границы, выученные на Train
        with open(quantiles_file, 'w', encoding='utf-8') as f:
            json.dump(quantiles_dict, f, indent=4)
            
    elif args.phase == 'val':
        if not quantiles_file.exists():
            raise FileNotFoundError("quantiles_winsor.json не найден! Прогони Train.")
            
        with open(quantiles_file, 'r', encoding='utf-8') as f:
            quantiles_dict = json.load(f)
            
        for col in feature_cols:
            if col in quantiles_dict:
                # 🛡️ ПРИМЕНЯЕМ: Обрезаем Val строго по границам Train!
                df_features[col] = df_features[col].clip(
                    lower=quantiles_dict[col]["lower"], 
                    upper=quantiles_dict[col]["upper"]
                )

    scaler_path = Path(args.artifacts_dir) / "scaler_features.pkl"
    scaler = StandardScaler()

    if args.phase == 'train':
        print(f"📈 Обучение Scaler...")
        df_features[feature_cols] = scaler.fit_transform(df_features[feature_cols])
        joblib.dump(scaler, scaler_path)
    elif args.phase == 'val':
        print(f"📉 Применение Scaler...")
        loaded_scaler = joblib.load(scaler_path)
        # Гарантируем строгий порядок колонок (как в Train) при передаче в Scaler
        df_features[feature_cols] = loaded_scaler.transform(df_features[feature_cols])
        
        if args.start_date:
            original_len = len(df_features)
            df_features = df_features[df_features['datetime'] >= pd.to_datetime(args.start_date)]
            print(f"🛡️ Удалено {original_len - len(df_features)} строк буфера (Защита от утечки)")

    for col in feature_cols:
        df_features[col] = df_features[col].astype(np.float32)
        
    df_features.to_parquet(args.output, engine='pyarrow', index=False)
    print(f"💾 Сохранено в {Path(args.output).name}")

if __name__ == "__main__":
    main()