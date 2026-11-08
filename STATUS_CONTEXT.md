# Текущий контекст проекта (SafeTrade RL Market Making v2 + Гибрид v7/v8 + Урок 6 v4.1 Честная Физика)

- **Статус:** АКТИВЕН УРОК 6 v4.1 «Честная Физика» (`task-2197`, `~720/1000` итераций, 24 независимых слота `12 Bid + 12 Ask` на 6 ценовых уровнях, синхронная оценка на 6 бюджетах `$95.5 .. $20,000`, строгий последовательный водопад очереди и объёма `Sequential Queue + Fill Waterfall`, честная рыночная переоценка на конце окна по `vec_env.avg_entry`, двойная телеметрия `BestMut` + слепой `BlindChamp OOS`).
- **Уроки 1..5:** ПОЛНОСТЬЮ ЗАВЕРШЕНЫ (`L1: 600/600`, `L2: 1000/1000`, `L3: 1000/1000`, `Bridge v7+v8`, `L4: 1000/1000`, `L5: 1000/1000`).
- **`SelfImpactRealismNet v2` (`_tools/self_impact_realism_net.py`):** Обучена на `9,370` реальных боевых ордерах из `1,066,941` секунд живой работы бота (`models/prl/rl_execution/self_impact_realism_net_v2.pt`, `MAE = 0.0891`).
- **Урок 6 v4.1 (`_tools/lesson_6_free_multi_order_trainer.py`, `324D -> 256 -> 26 continuous sigmoid outputs`, `_tools/vectorized_free_multi_order_env.py`):**
  - Закрыты все 4 физических эксплойта симуляции (раздел 8.4 [`docs/PROJECT_ARCHITECTURE_AND_ROADMAP.md`](file:///home/restorator/crypto_bot/docs/PROJECT_ARCHITECTURE_AND_ROADMAP.md)):
    1. Никакого списания убытка через `min_buy_px` на конце окна — проверка строго по истинному `vec_env.avg_entry - 1.05 * tick_sz`.
    2. Убрана фантомная ликвидность `+l1_bid_q`/`+l1_ask_q` при касании спреда — налив идёт строго от реального потока сделок `sv`/`bv`.
    3. Последовательный водопад `cumsum(eff_qa + eff_qty)` — объём сделок расходуется сверху вниз без параллельного списания очередей.
    4. Честный бухучёт `LIFO` и логирование слепого Чемпиона `BlindChamp` (`Group #0` OOS: `+0.05%..+0.26%` за 30 мин при `RealDD = 0.00%` и `PaperDD = 0.33%..0.68%`).
- **Оптимальная стратегия обучения 2.0 (3-фазный пайплайн на `1500` итераций вместо 6 уроков):** Задокументирована в разделе 8.3 [`docs/PROJECT_ARCHITECTURE_AND_ROADMAP.md`](file:///home/restorator/crypto_bot/docs/PROJECT_ARCHITECTURE_AND_ROADMAP.md).
- **Важное правило:** НЕ читай `.log` файлы целиком! Используй только `tail -n 20`.
