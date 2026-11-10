# Текущий контекст проекта (SafeTrade RL Market Making v2 + Гибрид v7/v8 + Уроки 1..18 Завершены + Боевой Деплой `v18.1 + L18_0065`)

- **Статус обучения:**
  - **ПОЛНОСТЬЮ ЗАВЕРШЕНЫ УРОКИ 1..18:**
    - `L17 Dynamic Slot Co-Evolution`: чемпион `L17_0261` (`ValScore = +208.61` на 18 чистых окнах, `ValEV = +0.0608%`).
    - **`L18 3-Tier LIFO Stack & Underwater Survival` (`_tools/lesson_18_multistack_lifo_survival_trainer.py`, завершен на `it=225`, пик `L18_0065`):** на жесткой 20-оконной корзине с подводными краш-тестами (`UW_NORM` + `UW_DUMP`, 55% инвентаря на `-5 тиков` и честный 3-уровневый LIFO-стек) чемпион **`L18_0065`** поднял `ValScore` с **`-166.01` (база L17)** до **`+28.66`** (`+194.67 п.`), а в прямом очном бенчмарке против `L17_0261` выиграл `FULL_20` (`Score -144.59` vs `-364.36`, `EV +0.0161%` vs `+0.0153%`, `DD 0.315%` vs `0.687%`) и сократил просадку на подводных краш-тестах **в 4.6 раза** (`-0.150%` vs `-0.687%`).
- **Боевой сервер VPS:**
  - АКТИВЕН В БОЕВОМ РЕЖИМЕ (`live-rl-prl-l9.service`, `--mode live`, ядро **`v18.1`**).
  - Развернут релизный чемпион Урока 18 **`models/prl/release_champion_v1.pt` (`L18_0065`, `388D-2StageResSwiGLU(384)`, `allow_scratch_stop_asks=True`)** с реконструированным 3-уровневым LIFO-стеком (`[tr_low, tr_mid, tr_high]`).
  - Развернуты и активны 24/7 новые реалтайм-коллекторы **Gate.io (`PEARL_USDT`, `gate-collector.service`)** и **Ourbit (`PRLUSDT` / `PEARL/USDT`, `ourbit-collector.service`)** с полной интеграцией в Watchdog, суточную GDrive-ротацию и Parquet-хранилище (`sync_all_data.py`).
- **Рефакторинг и безопасность (v18.1):**
  - Вычищены захардкоженные секреты/прокси из `_tools/vps_config.py` (перенесены в `.env` / `_tools/project_secrets.json`).
  - Устранено Sim-to-Real расхождение `flow_16[13:15]` (`has_bid` / `has_ask`) и `is_ask_penny` для `Ask[2..3]` в `_tools/live_rl_trader.py`.
  - Внедрен 3-уровневый LIFO-стек с жесткой сверкой инварианта `tr_low + tr_mid + tr_high == pos_qty` и защита от `HTTP 422 insufficient_balance` на SafeTrade.
  - Оптимизированы потоковый буфер `_tools/live_feature_updater.py` (`compact_rows` + `O(1)` дедупликация сделок) и единый расчет `LearnedRegimeCompetenceRouter` с живыми скоростями `(vel10, vel60, vel300)` в `_tools/swarm_router.py`.
- **Важное правило:** НЕ читай `.log` файлы целиком! Используй только `tail -n 20` или `manage_task`.
