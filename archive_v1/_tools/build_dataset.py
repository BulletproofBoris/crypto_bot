import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--timeframe', type=str)
    parser.add_argument('--workers', type=int)

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"❌ Ошибка: Входной файл {args.input} не найден")
        return

    df = pd.read_csv(args.input)
    df['datetime'] = pd.to_datetime(df['datetime'])
    mask = (df['datetime'] >= args.start_date) & (df['datetime'] <= args.end_date)
    
    df.loc[mask].sort_values(['datetime', 'ticker']).to_csv(args.output, index=False)
    print(f"✅ Готово: сохранено в {args.output}")

if __name__ == "__main__":
    main()