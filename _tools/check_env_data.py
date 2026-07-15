import pandas as pd
import numpy as np
from pathlib import Path

def check_parquet():
    file_path = Path("data/processed/2000_2026_1d/rl_env/environment_data.parquet")
    
    if not file_path.exists():
        print("❌ Файл не найден!")
        return

    print("⏳ Загрузка данных...")
    df = pd.read_parquet(file_path)
    
    print("\n" + "="*50)
    print("📊 ОТЧЕТ О ПЕСОЧНИЦЕ RL")
    print("="*50)
    print(f"🔹 Размер датасета: {df.shape[0]:,} строк, {df.shape[1]} колонок")
    print(f"🔹 Количество тикеров: {df['ticker'].nunique()}")
    print(f"🔹 Период данных: с {df['datetime'].min().date()} по {df['datetime'].max().date()}")
    
    # Считаем колонки с вероятностями
    prob_cols = [c for c in df.columns if c.endswith('_p0') or c.endswith('_p1') or c.endswith('_p2')]
    print(f"🔹 Найдено колонок с вероятностями: {len(prob_cols)} (Должно быть 63 = 21 модель * 3 класса)")
    
    # Проверка на NaN
    total_nans = df.isna().sum().sum()
    if total_nans == 0:
        print("✅ Пропусков (NaN) нет. Данные кристально чистые!")
    else:
        print(f"⚠️ ВНИМАНИЕ: Найдено {total_nans:,} пропусков (NaN)!")
        print(df.isna().sum()[df.isna().sum() > 0])

    # Проверка суммы вероятностей (p0 + p1 + p2 должно быть = 1.0)
    # Возьмем одну модель для примера: c90_rank1
    if 'c90_rank1_trading_bot_best_acc_70.38_run_5c3b834f.keras_p0' in df.columns:
        # Имена колонок могут отличаться, найдем первый набор
        p0_col = prob_cols[0]
        prefix = p0_col[:-3] # убираем _p0
        
        sums = df[f'{prefix}_p0'] + df[f'{prefix}_p1'] + df[f'{prefix}_p2']
        if np.allclose(sums, 1.0, atol=1e-3):
            print("✅ Математика Softmax в норме (сумма вероятностей = 1.0)")
        else:
            print("❌ ОШИБКА: Вероятности не складываются в 1.0!")
            
    print("="*50)

if __name__ == "__main__":
    check_parquet()