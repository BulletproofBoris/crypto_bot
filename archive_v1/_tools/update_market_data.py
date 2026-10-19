import os
import sys
import asyncio
import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime, timezone

# Подключаем корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Импортируем из нашего ядра
from _core.data_loader import CryptoClient, TF_MS

UNIVERSE_FILE = BASE_DIR / "universe.csv"

# Биржи с ограниченной глубиной истории и их лимит (кол-во свечей)
EXCHANGE_CANDLE_LIMIT = {
    "gate": 10_000,
}


def _effective_start_year(exchange_id: str, start_year: int, timeframe: str) -> int:
    """Для бирж с лимитом глубины обрезает стартовый год до допустимого."""
    candle_limit = EXCHANGE_CANDLE_LIMIT.get(exchange_id)
    if not candle_limit:
        return start_year
    tf_ms = TF_MS.get(timeframe.lower(), 3_600_000)
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    earliest_ts = now_ts - candle_limit * tf_ms
    earliest_year = datetime.fromtimestamp(earliest_ts / 1000, tz=timezone.utc).year
    return max(start_year, earliest_year)


async def fetch_one(
    ticker: str,
    exchange_id: str,
    client: CryptoClient,
    start_year: int,
    timeframe: str,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    results: list,
    counter: list,
    total: int,
):
    """Скачивает один тикер под защитой семафора."""
    async with semaphore:
        try:
            df = await client.fetch_history(ticker, start_year=start_year, timeframe=timeframe)
            if df is not None and not df.empty:
                counter[0] += 1
                tqdm.write(f"🟢 [{counter[0]}/{total}] {ticker} ({exchange_id}): Успешно загружен")
                results.append((ticker, True))
            else:
                tqdm.write(f"🔴 {ticker}: Ошибка — данные не получены")
                results.append((ticker, False))
        except Exception as e:
            tqdm.write(f"💥 {ticker}: Сбой загрузки — {e}")
            results.append((ticker, False))
        finally:
            pbar.set_postfix_str(f"{ticker} [{exchange_id}]", refresh=False)
            pbar.update(1)


async def process_tickers_async(
    ticker_exchange_pairs: list,
    start_year: int,
    timeframe: str,
    workers: int,
):
    """
    Параллельная загрузка через asyncio.Semaphore.
    Один CryptoClient на биржу, все тикеры — конкурентно.
    """
    total = len(ticker_exchange_pairs)
    results: list = []
    counter = [0]  # mutable counter для корутин

    # Группируем тикеры по бирже
    groups: dict[str, list[str]] = {}
    for ticker, exchange in ticker_exchange_pairs:
        groups.setdefault(exchange, []).append(ticker)

    # Предупреждения об обрезке глубины истории
    for exchange_id in groups:
        eff = _effective_start_year(exchange_id, start_year, timeframe)
        if eff > start_year:
            tqdm.write(
                f"⚠️  [{exchange_id}] Лимит {EXCHANGE_CANDLE_LIMIT[exchange_id]} свечей → "
                f"старт с {eff} вместо {start_year}"
            )

    semaphore = asyncio.Semaphore(workers)

    pbar = tqdm(
        total=total,
        desc="📊 Инструменты",
        unit=" тик",
        dynamic_ncols=True,
        colour="green",
        leave=True,
    )

    # Создаём клиентов и запускаем все задачи разом
    clients: dict[str, CryptoClient] = {
        ex: CryptoClient(exchange_id=ex) for ex in groups
    }

    try:
        tasks = []
        for exchange_id, tickers in groups.items():
            eff_year = _effective_start_year(exchange_id, start_year, timeframe)
            client = clients[exchange_id]
            for ticker in tickers:
                tasks.append(
                    fetch_one(
                        ticker, exchange_id, client,
                        eff_year, timeframe,
                        semaphore, pbar, results, counter, total,
                    )
                )
        await asyncio.gather(*tasks)
    finally:
        pbar.close()
        for client in clients.values():
            await client.disconnect()

    success = sum(1 for _, ok in results if ok)
    failed = [t for t, ok in results if not ok]
    return success, failed


def update_all_data(start_year: int, timeframe: str, workers: int):
    print(f"📊 Чтение списка инструментов из {UNIVERSE_FILE.name}...")

    if not UNIVERSE_FILE.exists():
        print(f"❌ Файл {UNIVERSE_FILE.name} не найден!")
        return

    universe_df = pd.read_csv(UNIVERSE_FILE)
    if "Ticker" not in universe_df.columns:
        print("❌ В universe.csv нет колонки 'Ticker'!")
        return

    if "Exchange" not in universe_df.columns:
        universe_df["Exchange"] = "binance"

    ticker_exchange_pairs = list(zip(universe_df["Ticker"], universe_df["Exchange"]))
    tickers = universe_df["Ticker"].tolist()

    exchange_summary = universe_df.groupby("Exchange")["Ticker"].count().to_dict()
    summary_str = ", ".join(f"{ex}: {cnt}" for ex, cnt in exchange_summary.items())
    print(f"✅ Найдено инструментов: {len(tickers)} ({summary_str})")
    print(f"🚀 Запуск (CCXT Async) | Таймфрейм = {timeframe} | Год старта = {start_year} | Потоков = {workers}")
    print("-" * 50)

    success_count, failed_tickers = asyncio.run(
        process_tickers_async(ticker_exchange_pairs, start_year, timeframe, workers)
    )

    print("\n" + "=" * 50)
    print(f"📈 ОБНОВЛЕНИЕ ({timeframe}) ЗАВЕРШЕНО")
    print(f"Успешно загружено: {success_count} / {len(tickers)}")
    if failed_tickers:
        print(f"⚠️ Ошибки при загрузке: {', '.join(failed_tickers)}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Многопоточный загрузчик котировок с криптобирж")
    parser.add_argument("--start-year", type=int, default=2017, help="Год начала загрузки (по умолчанию: 2017)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Таймфрейм (1d, 1h, 15m, 5m)")
    parser.add_argument("--workers", type=int, default=4, help="Число параллельных загрузок (по умолчанию: 4)")
    args = parser.parse_args()

    update_all_data(start_year=args.start_year, timeframe=args.timeframe, workers=args.workers)