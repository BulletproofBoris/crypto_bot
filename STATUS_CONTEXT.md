# Текущий контекст проекта (SafeTrade RL Market Making v2 + Гибрид v7/v8 + Урок 6 Free 24-Slot Multi-Order Grid)

- **Статус:** АКТИВЕН УРОК 6 v3 (`task-2078`, `~655/1000` итераций, 24 независимых слота `12 Bid + 12 Ask` на 6 ценовых уровнях, синхронная оценка на 6 бюджетах `$95.5 .. $20,000`, оптимизация по `Realized DD` + `LIFO Sub-Ledger` скальпинг нижних траншей + `SelfImpactRealismNet v2`).
- **Уроки 1..5:** ПОЛНОСТЬЮ ЗАВЕРШЕНЫ (`L1: 600/600`, `L2: 1000/1000`, `L3: 1000/1000`, `Bridge v7+v8`, `L4: 1000/1000`, `L5: 1000/1000`).
- **`SelfImpactRealismNet v2` (`_tools/self_impact_realism_net.py`):** Обучена на `9,370` реальных боевых ордерах из `1,066,941` секунд живой работы бота (`models/prl/rl_execution/self_impact_realism_net_v2.pt`, `MAE = 0.0891`).
- **Урок 6 (`_tools/lesson_6_free_multi_order_trainer.py`, `324D -> 256 -> 26 continuous sigmoid outputs`, `_tools/vectorized_free_multi_order_env.py`):**
  - Векторная среда без точек синхронизации CPU-GPU (`VRAM = 3.34 GB` при `8190` средах и `24` слотах).
  - `1365` кандидатов × `6` бюджетов (`$95.5, $1k, $3k, $5k, $10k, $20k`, суммарный портфель `$39,095.50`).
  - Защита Чемпиона `Hall-of-Fame` (`Group #0`) + глобальное ограничение нормы Фробениуса (`max_norm_w1 = 145.0, max_norm_w2 = 45.0`).
  - Раздельный учёт реализованной просадки по закрытым операциям (`RealDD`) и плавающей бумажной просадки (`PaperDD`) с `LIFO Sub-Ledger` разгрузкой.
- **Оптимальная стратегия обучения 2.0 (3-фазный пайплайн на `1500` итераций вместо 6 уроков):** Задокументирована в разделе 8 [`docs/PROJECT_ARCHITECTURE_AND_ROADMAP.md`](file:///home/restorator/crypto_bot/docs/PROJECT_ARCHITECTURE_AND_ROADMAP.md).
- **Важное правило:** НЕ читай `.log` файлы целиком! Используй только `tail -n 20`.
