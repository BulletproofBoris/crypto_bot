# Контекст проекта и передача дел для ИИ-агента (Agent Handover & Context)

**Дата актуализации:** 15 октября 2026 г.
**Рабочая директория:** `/home/restorator/crypto_bot`
**Python-окружение:** `/home/restorator/crypto_bot/.venv/bin/python`

---

## 1. Главная цель и активные торговые пары

Разработка высокочастотной и среднесрочной алгоритмической торговой системы (маркет-мейкинг + статистический арбитраж + направленный микроструктурный импульс) для пар:
1. **`PRL/USDT` (Pearl):**
   - Целевая биржа исполнения: **SafeTrade** (`prlusdt`, шаг цены `tick_size = 0.005 USDT`).
   - Внешний ликвидный ориентир (CEX): **BigONE** (`PRL-USDT`).
   - Ончейн-сеть: **Pearl (`api.prlscan.com`)** — высота блока, размер мемпула (`mempool_size`), сложность (`difficulty`), награда (`reward_prl`), переводы китов (`whale_volume_5k`, `whale_volume_25k`), майнинг-пулы.
2. **`QUANTUS/USDT` (`QTC-USDT`):**
   - Целевая биржа исполнения: **SafeTrade** (`quantususdt`, эффективный рыночный шаг котирования `tick_size = 0.05 USDT`).
   - Внешний ликвидный ориентир (CEX): **BigONE** (`QTC-USDT`).
   - Децентрализованный пул ликвидности (DEX): **Rhea Finance / Near Intents** (`QTC/USDC` через `api.dexscreener.com`).
   - Ончейн-сеть: **Quantus (`sqm.quantus.com/v1/graphql`)** — высота блока, майнеры, аккаунты, 5-минутный поток трансферов и объёмов китов.
3. **Глобальные макро-поводыри (Binance WebSocket):**
   - `BTC/USDT` и `NEAR/USDT` (секундные срезы BBO `bid/ask/mid/last`, объёмы на лучших уровнях `bid_qty/ask_qty` и секундные агрегаты рыночных покупок/продаж `buy_vol/sell_vol`).

> **Важно:** Монета `TSC` (`tscusdt`) была полностью выведена из проекта и удалена со всех уровней (VPS, Google Drive, локальные `data/` и все скрипты пайплайна) из-за мёртвой ликвидности и отсутствия связи с `PRL` и `QUANTUS`.

---

## 2. Инфраструктура сбора данных (Lambda Data Lake)

### 2.1. Удалённый VPS-сервер (`81.19.137.18:2222`)
- Конфигурация подключения находится в `_tools/vps_config.py` (не коммитится в Git).
- На сервере в `/opt/crypto_bot` работают 5 системных служб (`systemd`), управляемых сторожем `_tools/watchdog_collectors.py` и проверяемых через `_tools/monitor_vps.py --once`:
  1. `safetrade-collector` (`_tools/collect_safetrade_realtime.py`) — прямое подключение с `curl_cffi` (`impersonate="chrome120"`) и автоматическим fallback на прокси. Собирает топ-50 уровней стакана (`bid_1..50`, `ask_1..50` из снапшота `limit=300`) + 6 агрегированных признаков глубокого стакана (`deep_bid_vol`, `deep_ask_vol`, `max_bid_wall_price`, `max_bid_wall_qty`, `max_ask_wall_price`, `max_ask_wall_qty`), а также сделки с полем `side`.
  2. `bigone-collector` (`_tools/collect_bigone_realtime.py`) — параллельный асинхронный опрос (`asyncio.gather`, цикл `<200 мс`) стаканов (`limit=200`, запись топ-50 + 6 глубоких признаков) и сделок для `PRL-USDT` и `QTC-USDT`.
  3. `rhea-collector` (`_tools/collect_rhea_realtime.py`) — ежесекундный опрос пула `QTC/USDC` на Rhea Finance (`price_usd`, `liquidity_usd`, `base_reserve`, `quote_reserve`, `volume_5m/1h/24h`, `buys/sells_5m/1h/24h`, `price_change_5m/1h/24h`).
  4. `binance-collector` (`_tools/collect_binance_macro.py`) — WebSocket `bookTicker` + `trade` для `BTCUSDT` и `NEARUSDT` с записью цен, объёмов BBO и секундных рыночных объёмов `buy_vol` / `sell_vol`.
  5. `onchain-collector` (`_tools/collect_onchain_and_rules.py`) — опрос ончейн-метрик Pearl и Quantus каждые 30 с + ежечасное обновление правил и статусов ввода/вывода бирж в `data/metadata/exchange_rules.json`.
- Ротация (`_tools/rotate_and_upload.py`): ежесуточно сжимает горячие CSV в `.csv.gz`, выгружает на Google Drive (`FOLDER_ID = 1-qWy2q2nYoZtSCiMDZ1DiKZ5roBqXx-c`) и имеет локальный буфер `/opt/crypto_bot/data/archive_pending/` на случай сбоев сети.

---

## 3. Локальный пайплайн данных и инженерия признаков

Порядок запуска конвейера обновления данных на локальной машине:
1. **Синхронизация и санитарная очистка (`_tools/sync_all_data.py` + `_tools/data_sanitizer.py`):**
   - Скачивает архивы `.csv.gz` с Google Drive в `data/cache_gdrive/` и горячие CSV с VPS в `data/vps_hot/`.
   - Модуль `_tools/data_sanitizer.py`:
     - Удаляет пересечённые уровни стакана (`ask_1_price <= bid_1_price`) и аномалии холодного старта (`0` скрещенных строк в итоговых Parquet).
     - Восстанавливает пропущенные стороны сделок (`buy/sell`) для исторических записей SafeTrade по `mid_price` стакана и правилу тика (`100%` заполненность).
     - Безопасно склеивает исторические и новые схемы `prl_onchain` (4, 8, 12, 15 и 18 столбцов) и `quantus_onchain` без потери строк.
   - Сохраняет очищенные таблицы в `data/processed/{prl,quantus,macro}/*.parquet`.

2. **Генерация 1-секундного датасета и стационарного банка фильтров (`_tools/prepare_1s_dataset.py` + `_tools/microstructure_features.py`):**
   - Команда: `.venv/bin/python _tools/prepare_1s_dataset.py --coin all`
   - Приводит все потоки к строгой равномерной сетке `1.0s` без дыр (пропуски цен заполняются `ffill`, а потоковые объёмы `trade_vol/buy_vol/sell_vol` на искусственных секундах зануляются).
   - Рассчитывает глубокие метрики стакана по 50 уровням (`calc_effective_vwap(..., max_levels=50)`, `imbalance_11_50`, `imbalance_top50`, дистанции до крупнейших лимитных стен `bid_wall_dist_bps`, `ask_wall_dist_bps`, `wall_vol_imbalance`).
   - Строит **4-полосный геометрический стационарный банк фильтров** с окнами $W \in [5, 20, 60, 300]\text{ с}$ (множитель $\approx 3.5\times\text{–}4\times$):
     - Приращения в тиках (`ret_{w}s_ticks`), Order Flow Imbalance (`ofi_norm_{w}s` по Cont–Kukanov–Stoikov), нормализованный CVD (`cvd_norm_{w}s`), интенсивность торгов (`vol_intensity_{w}s`), реализованная волатильность (`rvol_{w}s_ticks`).
     - Ортогональные **полосовые разности (Band-Pass Filters)**: `imb_band_5_20`, `imb_band_20_60`, `imb_band_60_300` и `ofi_band_*` (снимают мультиколлинеарность между окнами).
     - Межбиржевые **Lead-Lag** расхождения (`leadlag_bo_st_*`, `leadlag_rhea_st_*`) и адаптивные Z-score спредов (`arb_spread_zscore_*`, `arb_rhea_zscore_*`).
   - Выходные файлы:
     - `data/features/prl_1s_features.parquet` (`447 210` строк, `642` признака, `0` NaN).
     - `data/features/quantus_1s_features.parquet` (`303 983` строки, `659` признаков, `0` NaN).

---

## 4. Разметка (7 классов) и экспериментальный выбор сетки `(H, K)`

1. **Физическая разметка (`_tools/data_labeler.py`):**
   - Использует двойное квантование (шаг цены `tick_size` и объём 1-го уровня стакана), асимметричную эластичность глубины топ-5 уровней в степени `0.25` и кинематику волны (`v_wave`, `d_vol`).
   - Формирует 7 классов (`-3..+3`) на устойчивых квантильных плато (`75%` и `90%`). Сходимость ошибки `Err(z)` между Train (80%) и Val (20%) на рабочих классах `±2` составляет **менее 5%** по всей сетке (`_tools/run_grid_validation.py`).

2. **Эмпирический отбор горизонтов `H` и коэффициентов `K = L/H` (Актуализация 17.10.2026):**
   - На полной сетке из 175 комбинаций ($H \in [1..25], K \in [3.0..6.0]$) доказано, что значения $K < 3$ недопустимы (шум глубины стакана).
   - Зафиксирована физическая граница $K \le \lfloor L_{\max} / H \rfloor$ при $N_{\text{eff}} \ge 5000$.
   - **Утвержденная ортогональная триада масштабов ($K \ge 3.0$):**
     1. **Fast BBO Sniper:** $H_1 = 5\,\text{с}, K_1 = 3.0 \implies L_1 = 15\,\text{с}$ (высокая скорость отклика на перекос BBO).
     2. **Core Spread Maker:** $H_2 = 12\,\text{с}, K_2 = 3.5 \implies L_2 = 42\,\text{с}$ (ядро удержания спреда, ошибка разметки $< 1.0\%$).
     3. **Wave Regime Trend:** $H_3 = 25\,\text{с}, K_3 = 3.0 \implies L_3 = 75\,\text{с}$ (волновой фильтр затяжного тренда).
   - Обучен зоопарк моделей с разметкой с конца (`_tools/train_rolling_swarm.py`).
   - На симуляторе SafeTrade (`_tools/simulate_execution.py`) чистый Maker-пайплайн показал прирост **$+1.80\% \dots +2.42\%$ монет PRL** во время рыночного дампа на $-9.07\%$.

3. **Отбор признаков (`_tools/feature_selector.py`):**
   - Актуальные списки сохранены в `data/features/selected_features_prl_h8s.json` (`331` признак) и `data/features/selected_features_quantus_h8s.json` (`288` признаков).

---

## 5. ⚠️ ОБЯЗАТЕЛЬНО К ИСПОЛНЕНИЮ СЛЕДУЮЩИМ АГЕНТОМ: КРИТИЧЕСКИЙ АНАЛИЗ

Подробный отчет о тестировании зафиксирован в документе **[`docs/RESEARCH_AND_TESTING_RESULTS.md`](file:///home/restorator/crypto_bot/docs/RESEARCH_AND_TESTING_RESULTS.md)**.
**Следующий агент не должен слепо доверять полученным цифрам симуляции (+1.8% .. +2.4% прироста монет).**

Необходимо провести жесткий критический анализ следующих допущений симулятора:
1. **Задержка исполнения и отмены ордеров (0ms vs 200–800ms):**
   В реальности отмена лимитки на SafeTrade через прокси занимает сотни миллисекунд. Проверить сценарии, когда заявку «прошивают» до отмены.
2. **Моделирование очереди (Queue Priority & Winner's Curse):**
   В симуляторе ордер исполняется при любом ненулевом `trade_vol`. В реальности лимитка внутри спреда ловит токсичный поток информированных трейдеров.
3. **Репрезентативность выборки:**
   Тест проведен на 1.4-дневном непрерывном отрезке. Требуется стресс-тестирование на участках мертвого флэта со сжатым спредом и резких каскадных сквизах.
4. **Координация ордеров триады:**
   Спроектировать мультигоризонтный диспетчер заявок, чтобы модели $H=5, 12, 25$ не конкурировали за один торговый баланс.

