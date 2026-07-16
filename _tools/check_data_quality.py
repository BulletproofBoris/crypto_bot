import argparse
import pandas as pd
from pathlib import Path
from colorama import init, Fore, Style

init(autoreset=True)

def check_ohlcv_quality(timeframe):
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data" / "raw" / timeframe
    if not data_dir.exists():
        print(f"{Fore.RED}❌ Директория {data_dir} не найдена!{Style.RESET_ALL}")
        return

    csv_files = list(data_dir.glob("*_MAX.csv"))
    if not csv_files:
        print(f"{Fore.YELLOW}⚠️ В {data_dir} нет файлов *_MAX.csv{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}🩺 КОМПЛЕКСНАЯ ПРОВЕРКА OHLCV ДАННЫХ ({timeframe}){Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")

    total_files = len(csv_files)
    passed_files = 0

    expected_delta = pd.Timedelta(hours=1) if timeframe == '1h' else pd.Timedelta(days=1)
    if timeframe == '15m':
        expected_delta = pd.Timedelta(minutes=15)

    for file_path in sorted(csv_files):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"{Fore.RED}🔴 {file_path.name}: Файл пуст!{Style.RESET_ALL}")
                continue

            # Проверка наличия колонок
            required_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                print(f"{Fore.RED}🔴 {file_path.name}: Отсутствуют колонки {missing_cols}{Style.RESET_ALL}")
                continue

            # Проверка NaN
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                print(f"{Fore.YELLOW}⚠️ {file_path.name}: Найдено {nan_count} пропусков (NaN)!{Style.RESET_ALL}")
            
            # Проверка логики OHLC
            errors = []
            if (df['High'] < df['Low']).any():
                errors.append("High < Low")
            if (df['High'] < df['Open']).any() or (df['High'] < df['Close']).any():
                errors.append("High меньше Open/Close")
            if (df['Low'] > df['Open']).any() or (df['Low'] > df['Close']).any():
                errors.append("Low больше Open/Close")
            if (df['Volume'] < 0).any():
                errors.append("Отрицательный объем")
            if (df['Close'] <= 0).any():
                errors.append("Нулевой или отрицательный Close")

            if errors:
                print(f"{Fore.RED}🔴 {file_path.name}: Ошибки в данных: {', '.join(errors)}{Style.RESET_ALL}")
                continue

            # Проверка на пропуски во времени (гэпы)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            deltas = df['Date'].diff().dropna()
            gaps = deltas[deltas > expected_delta]
            
            if len(gaps) > 0:
                max_gap = gaps.max()
                print(f"{Fore.YELLOW}⚠️ {file_path.name}: Найдено {len(gaps)} пропусков (гэпов) > {expected_delta}. Макс. гэп: {max_gap}{Style.RESET_ALL}")
                # We do not count gaps as hard failures unless they are critical, but we report them.
            else:
                pass
                
            if not errors and nan_count == 0:
                # Если всё ок
                passed_files += 1

        except Exception as e:
            print(f"{Fore.RED}🔴 {file_path.name}: Ошибка при чтении/проверке - {e}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}==================================================={Style.RESET_ALL}")
    print(f"✅ Проверка завершена. Полностью чистых файлов: {passed_files} / {total_files}")
    if passed_files < total_files:
        print(f"{Fore.YELLOW}⚠️ В некоторых файлах найдены проблемы (см. лог выше).{Style.RESET_ALL}")
    print(f"{Fore.CYAN}==================================================={Style.RESET_ALL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Комплексная проверка OHLCV данных")
    parser.add_argument("--timeframe", type=str, default="1d", help="Таймфрейм (1d, 1h, 15m)")
    args = parser.parse_args()

    check_ohlcv_quality(args.timeframe)
