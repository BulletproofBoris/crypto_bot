# Текущий контекст проекта (SafeTrade RL Market Making v2 + Гибрид v7/v8 + Уроки 1..17 Завершены + Боевой Деплой `L17_0261` + Урок 18 в работе)

- **Статус обучения:**
  - **ПОЛНОСТЬЮ ЗАВЕРШЕНЫ УРОКИ 1..17:** включая `L15 SwiGLU Slot-Aware`, `L16 2-Stage Residual SwiGLU Geometry (388D)` и **`L17 Dynamic Slot Co-Evolution` (чемпион `L17_0261`: `ValScore = +208.61`, `ValEV = +0.0608%`, `[N:+0.09% D:-0.03% P:+0.04% DD:0.27%]`)**.
  - **В РАБОТЕ — Урок 18 (`_tools/lesson_18_multistack_lifo_survival_trainer.py`, `600` итераций):** честный 3-уровневый LIFO-стек (`[low_tranche, mid_tranche, core_pos]`), стресс-сценарии `Underwater Warm-Reset` (`55%` инвентаря на `-5 тиков`), 20 валидационных окон (`6 NORM + 6 DUMP + 6 PUMP + UW_NORM + UW_DUMP`) и ко-эволюция быстрых слотов `Bid[0..3]`.
- **Боевой сервер VPS:**
  - АКТИВЕН В БОЕВОМ РЕЖИМЕ (`live-rl-prl-l9.service`, `--mode live`, `Equity ~ $703.40`).
  - На лету через Hot-Reload активирован чемпион Урока 17 **`models/prl/release_champion_v1.pt` (`L17_0261`, `388D-2StageResSwiGLU(384)`, `allow_scratch_stop_asks=True`)**.
  - Развернуты и активны 24/7 новые реалтайм-коллекторы **Gate.io (`PEARL_USDT`, `gate-collector.service`)** и **Ourbit (`PRLUSDT` / `PEARL/USDT`, `ourbit-collector.service`)** с полной интеграцией в Watchdog, суточную GDrive-ротацию и Parquet-хранилище (`sync_all_data.py`).
- **Рефакторинг и безопасность (v18.1):**
  - Вычищены захардкоженные секреты/прокси из `_tools/vps_config.py` (перенесены в `.env` / `_tools/project_secrets.json`).
  - Устранено Sim-to-Real расхождение `flow_16[13:15]` (`has_bid` / `has_ask`) и `is_ask_penny` для `Ask[2..3]` в `_tools/live_rl_trader.py`.
  - Внедрен 3-уровневый LIFO-стек с жесткой сверкой инварианта `tr_low + tr_mid + tr_high == pos_qty` и защита от `HTTP 422 insufficient_balance` на SafeTrade.
  - Оптимизированы потоковый буфер `_tools/live_feature_updater.py` (`compact_rows` + `O(1)` дедупликация сделок) и единый расчет `LearnedRegimeCompetenceRouter` с живыми скоростями `(vel10, vel60, vel300)` в `_tools/swarm_router.py`.
- **Важное правило:** НЕ читай `.log` файлы целиком! Используй только `tail -n 20` или `manage_task`.
