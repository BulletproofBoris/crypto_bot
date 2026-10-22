# Архитектура проекта и каталог активных инструментов

**Актуализация:** 20 октября 2026 г.  
**Рабочая директория:** `/home/restorator/crypto_bot`

---

## 1. Концепция и целевой инвариант

Автономная высокочастотная и среднесрочная алгоритмическая система котирования для биржи **SafeTrade** по монетам:
- **`PRL/USDT` (`pearlusdt`, Pearl):** `tick_size = 0.005 USDT`. Поводырь: **BigONE (`PRL-USDT`)** + ончейн **Pearl (`api.prlscan.com`)**.
- **`QUANTUS/USDT` (`quantususdt`, Quantus):** `tick_size = 0.05 USDT`. Поводыри: **BigONE (`QTC-USDT`)**, **Rhea Finance DEX (`QTC/USDC`)** + ончейн **Quantus**.
- **Макро-поводыри:** **Binance (`BTC/USDT`, `NEAR/USDT`)** 1s BBO и потоки сделок.

**Ключевой инвариант:** целевая функция оптимизирована на максимальный физический прирост монет ($Q_{\text{token}}$) в режиме чистого **Maker (`0.0%` комиссия)** с защитой от токсичного потока (Adverse Selection) и учётом приоритета очереди $Q_{\text{ahead}}$ и задержки сети $\text{RTT} = 1.0\,\text{с}$.

---

## 2. Каталог активного ядра (`_tools/`)

После рефакторинга и очистки устаревших скриптов проект содержит только боевые модули конвейера:

### 2.1. Сбор данных на VPS (`81.19.137.18:2222`) и ротация
- [_tools/collect_safetrade_realtime.py](file:///home/restorator/crypto_bot/_tools/collect_safetrade_realtime.py) — сбор стакана L2 (топ-50 + глубокие стены) и сделок SafeTrade (`pearlusdt`, `quantususdt`).
- [_tools/collect_bigone_realtime.py](file:///home/restorator/crypto_bot/_tools/collect_bigone_realtime.py) — асинхронный сбор стакана и сделок BigONE (`PRL-USDT`, `QTC-USDT`).
- [_tools/collect_rhea_realtime.py](file:///home/restorator/crypto_bot/_tools/collect_rhea_realtime.py) — сбор пула ликвидности `QTC/USDC` на Rhea Finance.
- [_tools/collect_binance_macro.py](file:///home/restorator/crypto_bot/_tools/collect_binance_macro.py) — WebSocket `bookTicker` + `trade` для `BTCUSDT` и `NEARUSDT`.
- [_tools/collect_onchain_and_rules.py](file:///home/restorator/crypto_bot/_tools/collect_onchain_and_rules.py) — ончейн-метрики Pearl / Quantus и правила бирж (`exchange_rules.json`).
- [_tools/watchdog_collectors.py](file:///home/restorator/crypto_bot/_tools/watchdog_collectors.py), [_tools/rotate_and_upload.py](file:///home/restorator/crypto_bot/_tools/rotate_and_upload.py), [_tools/monitor_vps.py](file:///home/restorator/crypto_bot/_tools/monitor_vps.py) — мониторинг 5 служб и суточная выгрузка `.csv.gz` на Google Drive.

### 2.2. Синхронизация, инженерия признаков и разметка
- [_tools/sync_all_data.py](file:///home/restorator/crypto_bot/_tools/sync_all_data.py) + [_tools/data_sanitizer.py](file:///home/restorator/crypto_bot/_tools/data_sanitizer.py) — синхронизация архивов Google Drive и горячих CSV с VPS, очистка скрещенных стаканов и сборка `data/processed/{prl,quantus,macro}/*.parquet`.
- [_tools/data_loader.py](file:///home/restorator/crypto_bot/_tools/data_loader.py) + [_tools/check_realtime_data.py](file:///home/restorator/crypto_bot/_tools/check_realtime_data.py) — фасад чтения данных и аудит непрерывности рядов.
- [_tools/prepare_1s_dataset.py](file:///home/restorator/crypto_bot/_tools/prepare_1s_dataset.py) + [_tools/microstructure_features.py](file:///home/restorator/crypto_bot/_tools/microstructure_features.py) — генерация равномерной 1-секундной сетки (со строгим сдвигом сделок `shift(1)` без заглядывания вперёд) и стационарного банка фильтров $W \in [5, 20, 60, 300]\,\text{с}$ (OFI, CVD, Band-Pass, Lead-Lag).
- [_tools/data_labeler.py](file:///home/restorator/crypto_bot/_tools/data_labeler.py) + [_tools/calculate_lmax.py](file:///home/restorator/crypto_bot/_tools/calculate_lmax.py) + [_tools/feature_selector.py](file:///home/restorator/crypto_bot/_tools/feature_selector.py) — физическая разметка на 7 классов (`-3..+3`) с двойным квантованием и отбор информативных признаков.

### 2.3. Мультигоризонтная триада, роутер режимов и симуляция исполнения
- [_tools/train_rolling_swarm.py](file:///home/restorator/crypto_bot/_tools/train_rolling_swarm.py) — обучение синхронизированного пула экспертов LightGBM с конца временного ряда (`win_00`, `win_01`, `win_02` по `210,000` с) для ортогональной триады масштабов:
  1. **Scale 1 (Sniper):** $H_1 = 5\,\text{с}, K_1 = 3.0 \implies L_1 = 15\,\text{с}$
  2. **Scale 2 (Core Maker):** $H_2 = 12\,\text{с}, K_2 = 3.5 \implies L_2 = 42\,\text{с}$
  3. **Scale 3 (Wave Regime Trend):** $H_3 = 25\,\text{с}, K_3 = 3.0 \implies L_3 = 75\,\text{с}$
- [_tools/swarm_router.py](file:///home/restorator/crypto_bot/_tools/swarm_router.py) — контекстный роутер экспертов LightGBM + блендер нейросетевого альянса (`conv1d+gru`, `causal_cnn`, `mlp_bottleneck`, `attention`) с быстрым дисковым кэшированием по MD5-отпечатку среза (`data/cache/`).
- [_tools/triad_dispatcher.py](file:///home/restorator/crypto_bot/_tools/triad_dispatcher.py) + [_tools/triad_utils.py](file:///home/restorator/crypto_bot/_tools/triad_utils.py) — мультигоризонтный координатор ордеров и реалистичный симулятор SafeTrade с поддержкой раздельных весов нейросетей (`--w-nn-5`, `--w-nn-12`, `--w-nn-25`), Lead-Lag Shield по BigONE, адаптивным управлением инвентарём и строгой OOS-изоляцией.
- [_tools/simulate_execution.py](file:///home/restorator/crypto_bot/_tools/simulate_execution.py) — базовый одномасштабный симулятор исполнения.
- [_tools/export_colab_dataset.py](file:///home/restorator/crypto_bot/_tools/export_colab_dataset.py) — экспорт сжатого `.npz` пакета для обучения нейросетевых экспертов в Google Colab.

- `_tools/train_rl_execution.py` — предрасчёт матрицы триады (`H=5s, 12s, 25s`), 10-уровневая L2 FIFO-среда исполнения (`SafeTradeL2ExecutionEnv`) и обучение RL-агента (`Maskable PPO` + `Behavioral Cloning Warm-Start`).

---

## 5. Итоги итерации от 2026-10-22 (Полный рой триады + L2 FIFO + RL Execution v1 + Горизонт H=60s)

### 5.1. Завершение обучения мультиархитектурного Walk-Forward роя (`92/92` нейросетей + `24` LightGBM)
Полностью обучены и синхронизированы в `models/prl/` все 4 нейросетевые архитектуры (`conv1d+gru`, `causal_cnn`, `attention`, `mlp_bottleneck`) и бустинги `LightGBM` на асинхронной сетке окон PRL:
- **`H=5s, K=3.0` (`L=15s`, `Train=75,000s`):** `14/14` окон (`56` нейросетей + `14` LightGBM). Средний OOS `IC`: `conv1d_gru = +0.1258` (`AUC = 0.7500`), `mlp_bottleneck = +0.1243` (`AUC = 0.7699`), `causal_cnn = +0.1144`, `attention = +0.1117`.
- **`H=12s, K=3.5` (`L=42s`, `Train=210,000s`):** `6/6` окон (`24` нейросети + `6` LightGBM). Средний OOS `IC`: `mlp_bottleneck = +0.1672` (`AUC = 0.7818`), `conv1d_gru = +0.1626` (`AUC = 0.7585`), `causal_cnn = +0.1394`, `attention = +0.1290`.
- **`H=25s, K=3.0` (`L=75s`, `Train=375,000s`):** `3/3` окон (`12` нейросетей + `3` LightGBM). Лучшие метрики OOS: `win_00 conv1d_gru IC = +0.2182`, `win_01 conv1d_gru IC = +0.1779`, `win_02 mlp_bottleneck IC = +0.1956 (AUC = 0.8164)`, `win_02 conv1d_gru IC = +0.1728 (Loss = 1.2621, Top-2% = +0.22 тиков)`.

### 5.2. Исследование и добавление 4-го макро-горизонта (`H=60s, K=2.5`)
Проведено эмпирическое исследование коэффициента памяти $K \in [2.0, 4.0]$ для горизонта $H=60\text{s}$ на расширенном 12-дневном датасете (`1,036,797` строк до `2026-10-22 03:05`):
- Оптимум достигнут при **$K = 2.5$ (`L = 150s`, `Train = 750,000s`, $N_{\text{eff}} = 5000$)**: чистый профит лучших сигналов `Top-2% PnL` вырос на `+53%` относительно $K=2.0$ (`+0.87` тика против `+0.57` тика).
- Обучен эталонный бустинг `models/prl/h60s_k2.5/` (`win_00`: `Train 2026-10-11 01:50 ➔ 2026-10-21 03:11`, `OOS Test 2026-10-21 03:18 ➔ 2026-10-22 03:05`): **`OOS IC = +0.2568`, `ROC-AUC = 0.7508`, `Top-2% PnL = +0.79 тиков`**.

### 5.3. Реалистичный 10-уровневый L2 FIFO-симулятор и RL-агент (`Maskable PPO`)
1. В `_tools/simulate_execution.py` и `_tools/train_rl_execution.py` реализовано честное сведение лимитных и рыночных ордеров по 10 уровням стакана SafeTrade (`ST_bid_1..10`, `ST_ask_1..10`) с учётом приоритета очереди `q_ahead`, сжатия очереди при отменах и задержки сети `latency = 1s`.
2. Реализован RL-агент управления ордерами (`MaskableActorCritic`, 7 действий с динамической маской `get_action_mask`):
   - **Целеполагание:** максимизация чистого баланса в `USDT` при консервативной оценке позиции по ликвидационной цене `best_bid`.
   - **Количественная оценка выгоды:** в вектор состояния подаются явные оценки ожидаемого чистого профита в тиках (`exp_entry_edge`, `exp_hold_edge`), сопоставляемые с внутренней оценкой головы Критика $V(S_t)$.
   - **Защита от ошибок моделей и проливов:** добавлены детектор ошибки прогноза в реальном времени (`model_err_12`), блокировка агрессивных покупок при провале глубины стакана (`imb_l5 <= -0.75`), 30-секундный `Cooldown` после стоп-выхода и асимметричный штраф `×1.75` за просадку.
