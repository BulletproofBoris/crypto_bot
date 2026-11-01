# 🚀 PRL/USDT & QUANTUS/USDT Autonomous Trading System (`crypto_bot`)

Автономная высокочастотная и среднесрочная система алгоритмического маркет-мейкинга и микроструктурного исполнения для биржи **SafeTrade** (`PRL/USDT`, `QUANTUS/USDT`).

---

## 🏗️ Единый 7-этапный пайплайн разработки и эксплуатации

Весь рабочий цикл сосредоточен в [`Trading bot notebook.ipynb`](file:///home/restorator/crypto_bot/Trading%20bot%20notebook.ipynb), [`RL_Execution_Lab.ipynb`](file:///home/restorator/crypto_bot/RL_Execution_Lab.ipynb) и [`Live_RL_Monitor.ipynb`](file:///home/restorator/crypto_bot/Live_RL_Monitor.ipynb):

| Этап | Модуль | Команда / Назначение |
| :--- | :--- | :--- |
| **1. Сбор данных 24/7 (VPS)** | `_tools/monitor_vps.py` | Контроль 5 системных служб на VPS (`81.19.137.18:2222`): стаканы L2 топ-50 и сделки SafeTrade + BigONE, DEX Rhea, макро Binance (`BTC`, `NEAR`), ончейн Pearl/Quantus. |
| **2. Синхронизация Data Lake** | `_tools/sync_all_data.py` | Скачивание архивов `.csv.gz` с Google Drive и горячих CSV + боевой телеметрии (`rl_telemetry.csv`) с VPS, санитарная очистка и сборка 13 таблиц в `data/processed/`. |
| **3. Инженерия 1s-признаков и разметка** | `_tools/prepare_1s_dataset.py`<br>`_tools/data_labeler.py` | Строгая 1-секундная сетка (`642` признака, стационарный фильтр-банк `W=[5,20,60,300]s` без утечки из будущего `shift(1)`) и 7-классовая разметка по 4 ортогональным масштабам **`5:3, 12:3.5, 25:3, 60:3`**. |
| **4. Планирование окон Walk-Forward** | `_tools/plan_rolling_windows.py` | Хронологическая сетка окон от начала датасета (`win_00..win_N`): проверка накопления данных для новых окон по горизонтам `H = 5, 12, 25, 60s`. |
| **5. Обучение роя экспертов (Swarm)** | `_tools/train_rolling_swarm.py`<br>`_tools/colab_neural_trainer.py` | Обучение пула деревьев **LightGBM** (`70` моделей по 4 горизонтам) и 4 нейросетевых архитектур (**`Conv1D+GRU`**, **`Causal CNN`**, **`Attention`**, **`MLP Bottleneck`** — `280` моделей). |
| **6. Обучение RL-агента исполнения** | `_tools/train_rl_execution.py`<br>`_tools/rl_gpu_population.py` | Предрасчёт кэша триады + макро `H=60s` (`data/cache/rl_triad_lob_cache_prl_176h.npz`), 10-уровневая L2 FIFO-среда `SafeTradeL2ExecutionEnv`, `Behavioral Cloning Warm-Start` и обучение **`Maskable PPO v3` + CEM GPU Population Search**. |
| **7. Шлюз регрессии и боевой демон** | `_tools/regression_gate.py`<br>`_tools/live_rl_trader.py` | Проверка чекпоинта `best_maskable_ppo.pt` на эталонных рыночных режимах (`Non-Regression Gate`) и деплой в боевой демон `LiveTriadRLTrader` (`2.8 Гц`, чистый PyTorch `FastKerasModel` без TensorFlow). |

---

## ⚡ Быстрый запуск из терминала

```bash
# 1-3. Обновить данные с VPS/GDrive, пересобрать 1s-признаки и разметку 4 горизонтов:
.venv/bin/python -m _tools.sync_all_data
.venv/bin/python -m _tools.prepare_1s_dataset --coin prl
.venv/bin/python -m _tools.data_labeler --coin prl --pairs "5:3,12:3.5,25:3,60:3"

# 4-5. Проверить доступность новых окон и дообучить пул экспертов:
.venv/bin/python -m _tools.plan_rolling_windows --coin prl --horizons 5,12,25,60
.venv/bin/python _tools/train_rolling_swarm.py --coins prl --horizons 5,12,25,60

# 6-7. Обновить RL-кэш, обучить Maskable PPO v3 и проверить статус боевого бота:
.venv/bin/python _tools/train_rl_execution.py --coin prl --tail-hours 176 --epochs 30
.venv/bin/python _tools/regression_gate.py
```
