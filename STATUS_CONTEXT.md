# Текущий контекст проекта (SafeTrade RL Market Making v2 + Гибрид v7/v8 + Уроки 1..20 Завершены + Боевой Деплой `v20.1 + L20_0025_soup22_pure_econ_reward`)

- **Актуальная база данных и рой экспертов 1-го порядка (обновлено 2026-11-11):**
  - Синхронизировано **`2,779,660` секундных тиков (`772.1ч`)** до `2026-11-11 07:12:48 MSK` (`safetrade_depth`: `2,429,378`, `bigone_depth`: `2,610,329`, `gate_trades`: `43,515`, `ourbit_trades`: `10,569`, `prl_rl_telemetry`: `1,344,690`).
  - Пересобраны и дообучены все свежие плавающие окна роя LightGBM + 6 нейросетевых архитектур: `H=5s` (`win_68`, 69 окон), `H=12s` (`win_23`, 24 окна, `AUC=0.7651`), `H=25s` (`win_12`, 13 окон, 78 экспертов) + обновлены альянсы `H=60/90/120s`.
  - Дообучена кросс-биржевая модель 1-го порядка **`CrossVenueFirstOrderNet` (`SafeTrade + BigONE`, `_tools/cross_venue_first_order_net.py`)**: слепой OOS Mean AUC вырос с `0.7966` (`ST-only`) до **`0.8226` (`ΔAUC = +0.0260`)** (`sweep_12s = 0.875`, `rebound_30s = 0.870`).
  - **Внедрены 4 lossless-оптимизации обучения сетей 1-го порядка:**
    1. `_tools/pretrain_lob_backbone.py`: векторизация 10-уровневого стакана + предзагрузка в VRAM (`BF16`, `batch_size=1024`, `@torch.compile`, `AdamW(fused=True)`, `FocalLoss` в `FP32`).
    2. `_tools/cross_venue_first_order_net.py` + `_tools/build_hybrid_v7_v8_cache.py`: побитово эквивалентный GPU `F.max_pool1d` вместо 90-шагового `np.pad` цикла, полный VRAM-тренинг (`batch_size=16384`), GPU `ROC-AUC` (`torch.argsort`) и быстрый `np.savez`.
    3. `_tools/fast_neural_engine.py`: отказ от грубого `FP8` на узких слоях в пользу `BF16` Tensor Cores + `FP32` loss, замена `@torch.jit.script` цикла GRU на нативный C++/cuDNN `torch._VF.gru` (`max_diff = 2.08e-07`), `F.layer_norm` и параллелизм `runs` через `CUDA Streams`.
    4. `_tools/train_rolling_swarm.py`: единый `shared_cache` DataFrame между горизонтами `H` и параллельное обучение окон LightGBM через `ThreadPoolExecutor(max_workers=3)`.

- **Статус обучения RL-агента (ПОЛНОСТЬЮ ЗАВЕРШЕНЫ УРОКИ 1..20, `400 / 400` эпох):**
  - **`L20 Pure Economic Utility & 9-Window Parallel Triton GPU` (`_tools/lesson_20_sim2real_noscratch_champion_trainer.py` + `_tools/fused_l20_train_tick.py`):**
    - Полный отказ от искусственных таймеров, запретов и суррогатных бонусов суеты в пользу 100% чистой экономической полезности счёта (`NetReturn_% - UlcerIndex_% - 0.35 * TailDD_%^2 - 0.04 * MeanAPILoad`).
    - **Базовая скорость RL-обучения зафиксирована на `5.7–7.2 с/эпоху`** (`16.5x` быстрее исходных `115 с/эпоху`): 9-оконный параллельный Triton-проход (`599` шагов × `110,592` сред вместо `5,391` шагов) + `O(1)` префиксный стакан (`100% GPU-Util`, `4,973 MiB / 8,151 MiB` VRAM, 0% PCIe swap). Резерв дальнейшего ускорения (`~2.2–2.5 с/эпоху`) сохранён в бэклоге Урока 21.
    - Абсолютный активный маркет-мейкер чемпион **`L20_0025_soup22_pure_econ_reward` (`models/prl/release_champion_v1.pt`, Model Soup итераций `22 + 25`)** уверенно победил `L18_0065_pruned16` и `L19_0055_cv_gated` на всех слепых OOS-корзинах:
      - `CLEAN_18`: **`EconUtil = +2.94bp`, `EV = +0.0224%`** (`NORM:+0.019%`, `DUMP:+0.013%`, `PUMP:+0.041%`, `MaxDD = 0.290%`) против `L19` (`+1.01bp`, `EV = +0.0123%`, `DUMP:-0.027%`, `MaxDD = 0.345%`) и `L18` (`-1.35bp`, `EV = +0.0063%`, `MaxDD = 0.705%`).
      - `FULL_20`: **`EconUtil = +1.76bp`, `EV = +0.0141%`** (в **`2.82x`** выше `L19` `+0.0050%` и в **`10.1x`** выше `L18` `+0.0014%`, все 3 режима строго в плюсе).
      - `RECENT_HOLDOUT_ALL` (`35` свежих OOS-окон = `18.7ч`): **`EV = +0.0243%`** (`NORM:+0.011%`, `DUMP:+0.080%`, `PUMP:+0.027%`) против отрицательного `EV = -0.0013%` (`DUMP:-0.052%`) у `L19`.
      - `BLIND_55H_PRE_TRAIN` (`29` слепых исторических окон за `55ч` до трейна): `EV = -0.0200%` (`DUMP:-0.105%`) лучше `L19` (`EV = -0.0232%`, `DUMP:-0.168%`).

- **Аудит боевого сервера VPS и Sim-to-Real укрепление (`v20.1`):**
  - Работа ведётся **строго на изолированном субаккаунте `...6C9` (`fp=220a65fa6912`)**; основной счёт `...6A4` криптографически заблокирован.
  - В `_tools/live_rl_trader.py` активны все 4 Sim-to-Real защиты: фикс гонки отмены прострельных исполнений (`_sync_24slot_orders`), `Cold-Start Warmup Shield` (`15` тиков прогрева GRU), лимитированный пул ордеров (`max_workers=2` + пейсинг `45 мс` + backoff на `HTTP 429`) и бесшовный Hot-Reload чемпиона.

- **Важное правило:** НЕ читай `.log` файлы целиком! Используй только `tail -n 20` или `manage_task`.
