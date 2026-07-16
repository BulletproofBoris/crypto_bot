import os
import sys
import json
import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# Подключаем корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import RAW_DIR

# Словарь: {Тикер Yahoo Finance : Название нашего файла}
MACRO_SYMBOLS = {
    "BTC-USD":  "BTC",    # Bitcoin (поводырь крипты)
    "ETH-USD":  "ETH",    # Ethereum (поводырь альткоинов)
    "DX-Y.NYB": "DXY",    # Индекс Доллара США (обратная корреляция)
    "^GSPC":    "SP500",  # S&P 500 (общий рыночный сентимент)
    "^VIX":     "VIX",    # Индекс страха (волатильность)
}


def _update_meta(df: pd.DataFrame, meta_file: Path):
    """Обновляет метафайл с датами первой и последней записи."""
    if df is not None and not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        first_date_str = df["Date"].min().strftime("%Y-%m-%d")
        last_date_str  = df["Date"].max().strftime("%Y-%m-%d")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {"first_trade_date": first_date_str, "last_downloaded_date": last_date_str},
                f, indent=4,
            )


def process_macro(name: str, ticker: str, start_year: int, raw_dir: Path) -> tuple[str, bool, str]:
    """Скачивает один макро-инструмент через yfinance (thread-safe)."""
    try:
        meta_file = raw_dir / f"MACRO_{name}_meta.json"
        csv_file  = raw_dir / f"MACRO_{name}.csv"
        start_date = f"{start_year}-01-01"

        # Инкрементальная загрузка: берём только новые данные
        if csv_file.exists() and meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
            last_date = meta.get("last_downloaded_date")
            if last_date:
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if last_date >= today_str:
                    return name, True, "Уже актуально"
                start_date = last_date

        ticker_obj = yf.Ticker(ticker)
        df_new = ticker_obj.history(start=start_date, auto_adjust=False)

        if df_new.empty:
            return name, True, "Нет новых данных"

        df_new.reset_index(inplace=True)
        if "Date" not in df_new.columns or "Close" not in df_new.columns:
            return name, False, "Отсутствуют нужные колонки в ответе Yahoo"

        df_new = df_new[["Date", "Close"]]
        df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.tz_localize(None)

        if csv_file.exists():
            df_old = pd.read_csv(csv_file)
            df_old["Date"] = pd.to_datetime(df_old["Date"])
            df_combined = (
                pd.concat([df_old, df_new])
                .drop_duplicates(subset=["Date"], keep="last")
                .sort_values("Date")
            )
            df_combined.to_csv(csv_file, index=False)
            _update_meta(df_combined, meta_file)
            return name, True, f"Докачано {len(df_new)} новых записей"
        else:
            df_new.to_csv(csv_file, index=False)
            _update_meta(df_new, meta_file)
            return name, True, f"Загружено с нуля ({len(df_new)} записей)"

    except Exception as e:
        return name, False, str(e)


def update_all_macro(start_year: int, workers: int, timeframe: str):
    raw_tf_dir = RAW_DIR / timeframe
    raw_tf_dir.mkdir(parents=True, exist_ok=True)

    total = len(MACRO_SYMBOLS)
    print(f"🚀 Загрузка макро-маркеров | Папка: {raw_tf_dir.relative_to(BASE_DIR)} | Потоков: {workers}")
    print("-" * 50)

    success_count = 0
    failed_symbols: list[str] = []

    pbar = tqdm(
        total=total,
        desc="📡 Макро",
        unit=" тик",
        dynamic_ncols=True,
        colour="magenta",
        leave=True,
    )

    # yfinance не поддерживает asyncio → используем ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_macro, name, ticker, start_year, raw_tf_dir): name
            for ticker, name in MACRO_SYMBOLS.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                res_name, success, msg = future.result()
                if success:
                    success_count += 1
                    tqdm.write(f"🟢 [{success_count}/{total}] {res_name}: {msg}")
                else:
                    failed_symbols.append(res_name)
                    tqdm.write(f"🔴 {res_name}: Ошибка — {msg}")
            except Exception as e:
                failed_symbols.append(name)
                tqdm.write(f"💥 {name}: Критический сбой — {e}")
            finally:
                pbar.set_postfix_str(name, refresh=False)
                pbar.update(1)

    pbar.close()

    print("\n" + "=" * 50)
    print("📈 ОБНОВЛЕНИЕ МАКРОДАННЫХ ЗАВЕРШЕНО")
    print(f"Успешно обработано: {success_count} / {total}")
    if failed_symbols:
        print(f"⚠️ Ошибки: {', '.join(failed_symbols)}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Многопоточный загрузчик макроданных (yfinance)")
    parser.add_argument("--start-year", type=int, default=2017, help="Год начала загрузки (по умолчанию: 2017)")
    parser.add_argument("--workers",    type=int, default=4,    help="Число параллельных потоков (по умолчанию: 4)")
    parser.add_argument("--timeframe",  type=str, default="1d", help="Целевой таймфрейм (1d, 1h, 15m)")
    args = parser.parse_args()

    update_all_macro(args.start_year, args.workers, args.timeframe)