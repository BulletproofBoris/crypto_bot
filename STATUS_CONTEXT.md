# Текущий контекст проекта (SafeTrade RL Market Making v2 + Гибрид v7/v8 + Уроки 1..22.2 Полностью Завершены `1000/1000` + Боевой Деплой `v22.2 + L22P2_SOUP_TOP2_iters_1_2_slot_queue_444d`)

- **Актуальная база данных и двухбиржевые кэши моделей 1-го порядка (обновлено 2026-11-12):**
  - Синхронизировано **`2,837,058` секундных тиков (`788.1ч` до `2026-11-11 23:09 MSK`)** (`safetrade_depth`: `2,500,094`, `bigone_depth`: `2,667,560`, `gate_trades`: `92,220`, `ourbit_trades`: `23,515`, `prl_rl_telemetry`).
  - Размечены 6 горизонтов `[(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]` одновременно для **SafeTrade** (`data/_mmap_cache/`, `models/prl/`) и **BigONE** (`data/_mmap_cache_bo/`, `models/prl_bo/`, `299` каузальных признаков `selected_features_prl_bo_h8s.json` в едином шаге `$0.005`).
  - Активна **4-биржевая модель 1-го порядка `CrossVenueFirstOrderNet v2`** (`SafeTrade + BigONE + Gate.io + Ourbit + BTC/NEAR`, `in_dim=44`, `models/prl/cross_venue_first_order/cross_venue_first_order_v1.pt`, слепой OOS Mean AUC **`0.84674`**).

- **Статус обучения RL-агента (ПОЛНОСТЬЮ ЗАВЕРШЕНЫ УРОКИ 1..22 Фаза 2, `1,000 / 1,000` эпох):**
  - **`L22.2 Per-Slot FIFO Queue-Progress (`444D`), Burst Rate-Limit Physics & Zero Gate-Folding Drift` (`_tools/lesson_22_p2_slot_queue_444d_trainer.py` + `_tools/fused_l22_train_tick.py`):**
    - Вход агента расширен с `420D` до **`444D`** (блоки `[420:444]` = индивидуальный прогресс продвижения очереди `1 - q_ahead / q_init` по каждому из `24` слотов `12 Bid + 12 Ask`, 1-в-1 с `_tools/live_rl_trader.py`), добавлена физика Burst Rate-Limit (`>4 mut/s -> 2-tick RTT`) и устранён дрейф свёртки гейтов (`Zero Gate-Folding Drift`), без единого жёсткого правила или таймера удержания.
    - **Победа `L22P2_SOUP_TOP2_iters_1_2_slot_queue_444d` (`models/prl/release_champion_v1.pt`) на слепых OOS-корзинах в полной физике Burst+REST+Latency:**
      - `CLEAN_18` (`18` окон с нуля): `CleanEV = +0.0279%`, `Pareto = +1.16 bp`, `MaxDD = 0.172%`, `Cnl/Fill = 51.8` (лучше `L22_0969 = 54.3` и `L21_0262 = 108.4`), **`η_queue = 19.3%`** (`L21_0262 = 10.1%`).
      - `FULL_22` (`18` чистых + `4` подводных окна): **`Pareto = -1.81 bp`** (лучше `L22_0969 = -2.03 bp` и `L21_0262 = -8.08 bp`), **`MaxDD = 0.353%`** (против `0.380%` у `L22_0969`).
      - **`RECENT_HOLDOUT_28w` (`28` свежих слепых OOS-окон):** **`CleanEV = +0.0014%`** (`3.5x` выше `L22_0969 = +0.0004%`, против `-0.0015%` у `L21_0262`), **`Pareto = -8.74 bp`** (`+0.96 bp` к `L22_0969 = -9.70 bp` и **`+5.93 bp`** к `L21_0262 = -14.67 bp`), `Fills = 38.2`, **`Cnl/Fill = 38.9`**, **`η_queue = 24.3%`**, **`MaxDD = 0.997%`** (`< 1.0%`!).
      - `BLIND_PRE_76w` (`76` исторических окон): `CleanEV = +0.0101%`, рекордный минимум `Cnl/Fill = 26.5` и рекордный **`η_queue = 32.4%`**.

- **Боевой сервер VPS, аудит КПД (`_tools/vps_kpd_audit.py`) и Sim-to-Real паритет (`v22.2`):**
  - Работа ведётся **строго на изолированном субаккаунте `...6C9` (`fp=220a65fa6912`)**; основной счёт `...6A4` криптографически заблокирован.
  - В `_tools/live_rl_trader.py` активен 100% честный паритет `444D` (`slot_qprog_24`) + `l7_selective_front_cancel = True` без искусственных таймеров удержания ордеров.

- **Важное правило:** НЕ читай `.log` файлы целиком! Используй только `tail -n 20` или `manage_task`.
