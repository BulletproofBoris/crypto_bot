import os
import sys
import asyncio
import argparse
import random
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Подключаем корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Импортируем из нашего ядра
from _core.data_loader import FinamClient

UNIVERSE_FILE = BASE_DIR / "universe.csv"

def process_ticker(ticker: str, start_year: int, timeframe: str) -> tuple[str, bool, str]:
    """
    Изолированная функция-воркер для одного процесса.
    Запускает асинхронную загрузку внутри синхронного пула процессов.
    """
    async def run_fetch():
        # ⏳ РАЗМАЗЫВАЕМ СТАРТ ПОТОКОВ (Jitter)
        delay = random.uniform(0, 5.0)
        await asyncio.sleep(delay)
        
        client = FinamClient()
        try:
            # Передаем параметр timeframe напрямую в умный клиент
            df = await client.fetch_history(ticker, start_year=start_year, timeframe=timeframe)
            
            if df is not None and not df.empty:
                return ticker, True, ""
            else:
                return ticker, False, "Данные не получены (пустой ответ или ошибка)"
        except Exception as e:
            return ticker, False, str(e)
    
    return asyncio.run(run_fetch())

def update_all_data(start_year: int, max_workers: int, timeframe: str):
    print(f"📊 Чтение списка инструментов из {UNIVERSE_FILE.name}...")
    
    if not UNIVERSE_FILE.exists():
        print(f"❌ Файл {UNIVERSE_FILE.name} не найден! Создайте его перед запуском.")
        return

    universe_df = pd.read_csv(UNIVERSE_FILE)
    if 'Ticker' not in universe_df.columns:
        print("❌ В universe.csv нет колонки 'Ticker'!")
        return

    tickers = universe_df['Ticker'].tolist()
    print(f"✅ Найдено инструментов: {len(tickers)}")
    print(f"🚀 Запуск: Потоков = {max_workers} | Таймфрейм = {timeframe} | Год старта = {start_year}")
    print("-" * 50)

    success_count = 0
    failed_tickers = []

    # Запускаем пул изолированных процессов
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_ticker, ticker, start_year, timeframe): ticker 
            for ticker in tickers
        }
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ticker_name, success, error_msg = future.result()
                if success:
                    success_count += 1
                    print(f"🟢 [{success_count}/{len(tickers)}] {ticker_name}: Успешно загружен")
                else:
                    failed_tickers.append(ticker_name)
                    print(f"🔴 {ticker_name}: Ошибка - {error_msg}")
            except Exception as e:
                failed_tickers.append(ticker)
                print(f"💥 {ticker}: Критический сбой процесса - {e}")

    # Итоговый отчет
    print("\n" + "=" * 50)
    print(f"📈 ПАРАЛЛЕЛЬНОЕ ОБНОВЛЕНИЕ ({timeframe}) ЗАВЕРШЕНО")
    print(f"Успешно загружено: {success_count} / {len(tickers)}")
    if failed_tickers:
        print(f"⚠️ Ошибки при загрузке: {', '.join(failed_tickers)}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инкрементальный загрузчик котировок с Finam")
    parser.add_argument("--start-year", type=int, default=2000, help="Год начала загрузки данных (по умолчанию: 2000)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Количество параллельных процессов")
    parser.add_argument("--timeframe", type=str, default="1d", help="Целевой MLOps таймфрейм (1d, 1h, 15m)")
    args = parser.parse_args()

    update_all_data(start_year=args.start_year, max_workers=args.workers, timeframe=args.timeframe)