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

2. **Экспериментальный отбор горизонтов `H` и коэффициентов `K = L/H` (`_tools/experiment_hk_selection.py`):**
   - Результаты 74 прогонов Out-of-Sample LightGBM сохранены в `data/labeled/experiment_hk_oos_results.csv`.
   - Выявлена фундаментальная обратная зависимость $K^*(H)$: на малых $H$ оптимален $K=1$, на средних $K=2$, на длинных $K=3$ (при $L > 75\text{ с}$ начинается деградация из-за снижения $N_{\text{eff}} = T_{\text{total}} / L < 3500$).
   - **Утверждённая конфигурация тактического 3-горизонтного ансамбля:**
     - **PRL/USDT:**
       1. Быстрый фильтр (Taker/Sniper): `H = 3s, K = 1 (L = 3s)` — ROC-AUC `0.7931`, сигнал срабатывает, когда оба `H=3s` и `H=25s` согласованы.
       2. Рабочий фильтр (Core Maker): `H = 8s, K = 2 (L = 16s)` — ROC-AUC `0.7136`, максимальный статистический запас.
       3. Структурный фильтр волны: `H = 25s, K = 3 (L = 75s)` — максимальный F1-macro `0.2784` и Lift `3.59x`.
     - **QUANTUS/USDT:**
       1. Быстрый фильтр: `H = 3s, K = 1 (L = 3s)` — ROC-AUC `0.8588`, Lift `11.17x`.
       2. Рабочий фильтр: `H = 8s, K = 1 (L = 8s)` — ROC-AUC `0.7905`, Lift `6.78x`.
       3. Структурный фильтр волны: `H = 25s, K = 2 (L = 50s)` — F1-macro `0.2683`, Lift `4.25x`.

3. **Отбор признаков (`_tools/feature_selector.py`):**
   - Отсекает мультиколлинеарные признаки ($|\rho| > 0.85$) и отбирает 95% кумулятивной важности LightGBM.
   - Актуальные списки сохранены в `data/features/selected_features_prl_h8s.json` (`290` признаков) и `data/features/selected_features_quantus_h8s.json` (`243` признака).

---

## 5. Архитектура долгосрочного планирования и следующие шаги (Next Steps)

Поскольку на 1-секундных стаканах за 4–5 дней нельзя напрямую обучать бустинг на часовых горизонтах ($N_{\text{eff}} = T / 3600 \approx 108 \ll 4000$), система строится как **двухуровневая иерархия**:
1. **Верхний уровень (Strategic Regime & Inventory Controller, масштаб 15m – 24h):**
   - Малопараметрическая аналитическая модель (по типу Avellaneda–Stoikov): отслеживает коинтеграционный базис между SafeTrade, BigONE и Rhea DEX, целевой баланс инвентаря $q_{\text{target}}(t)$, часовые/суточные метрики `RHEA_volume_1h/24h`, тренд `BTC/NEAR` и ончейн-притоки китов/мемпула (`onchain_mempool_delta_5m`, `onchain_whale_vol_sum_1h`).
   - Для калибровки верхнего уровня можно использовать 30–90-дневную историю минутных/15-минутных свечей (`_tools/fetch_coinex_prl.py`, REST klines BigONE/Binance) и полную базу блоков Pearl.
2. **Нижний уровень (Tactical 3-Horizon Ensemble, `H = 3s, 8s, 25s`):**
   - Обучение и объединение трёх моделей тактического горизонта (`H=3s`, `H=8s`, `H=25s`), которые принимают решение о моменте выставления/снятия ордера внутри коридора, заданного стратегическим контроллером.
