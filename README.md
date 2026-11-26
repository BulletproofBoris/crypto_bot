# 🚀 SafeTrade Order-3 Autonomous Multi-Regime Trading System (`crypto_bot`)

Автономная каузальная система алгоритмического маркет-мейкинга и селективного исполнения для рынка **`PRL/USDT` (SafeTrade)** с кросс-биржевым радаром (`BigONE`, `Gate.io`, `Ourbit`, `Binance`) и математическим паритетом симуляции и боя **`0.00e+00`**.

---

## 🏗️ Актуальная архитектура (`Signal Stack v7`: `96D` → `206D` → `K=8 Liquid MoE × 58D`)

| Уровень | Ключевые модули | Назначение |
| :--- | :--- | :--- |
| **1. Сбор данных 24/7 (VPS)** | `_tools/watchdog_collectors.py`<br>`_tools/collect_*_realtime.py` | Непрерывный сбор стаканов и ленты сделок `SafeTrade`, `BigONE`, `Gate.io`, `Ourbit`, макро `Binance` и ончейн-метрик с авто-ротацией в облако (`_tools/rotate_and_upload.py`). |
| **2. Каузальный конвейер данных (`96D`)** | `_tools/update_data_pipeline_96d.py`<br>`_tools/build_hybrid_market_obs_96d.py` | Синхронизация данных, декантаминация собственных сделок (`RAW → DEC`), дообучение Walk-Forward бустеров (`Order-1`, `Order-2`, `Wave`, `Order-2.5`) и сборка каузальных матриц `96D` для обоих миров (`0%` утечки будущего). |
| **3. Тензорное ядро симуляции** | `_tools/fused_order3_clean_tick.py` | 10-уровневый FIFO-симулятор 24 слотов с физикой `DEC / RAW × Calib / Harsh`, якорем рециклинга всплесков `ch[56]`, эндогенным гистерезисом отмен `ch[57]` и порежимной атрибуцией (`UP / DN / FL`). |
| **4. Многостадийное обучение и модульные специалисты** | `_tools/train_order3_champion.py`<br>`_tools/train_l83_l84_specialists_and_router.py` | Обучение полного ансамбля (`L80..L85`), а также изолированная «Школа узких специалистов» (`Mode #0..#3`) + отдельный «Урок Чистого Оркестратора» (`L84_Router`, `370D` Liquid Gate) на двух GPU. |
| **5. Боевой демон, паритет и канарейка** | `_tools/live_rl_trader.py`<br>`_tools/order3_live_bridge.py`<br>`_tools/canary_scalp_probe.py` | Боевое исполнение 24 слотов на VPS, проверка побитового паритета (`_tools/test_order3_sim_live_parity.py`), изолированный канареечный зонд широкого спреда (`8 PRL`, `_tools/analyze_canary_scalp.py`) и деплой одной командой (`_tools/deploy_vps.py`). |

---

## ⚡ Основные команды эксплуатации

```bash
# 1. Полное обновление данных с VPS, дообучение WF-бустеров и сборка 96D кэша (DEC + RAW):
CUDA_VISIBLE_DEVICES=1 .venv/bin/python _tools/update_data_pipeline_96d.py --device cuda:0

# 2. Проверка математического паритета Sim/Live (0.00e+00) и латентности CPU:
.venv/bin/python _tools/test_order3_sim_live_parity.py models/prl/release_champion_v1.pt

# 3. Изолированное дообучение отдельной моды-специалиста и Оркестратора (на обеих GPU):
CUDA_VISIBLE_DEVICES=0 .venv/bin/python _tools/train_l83_l84_specialists_and_router.py --stage mode0_scalp --world-mode raw --island-id 0
CUDA_VISIBLE_DEVICES=1 .venv/bin/python _tools/train_l83_l84_specialists_and_router.py --stage mode0_scalp --world-mode dec --island-id 1
.venv/bin/python _tools/train_l83_l84_specialists_and_router.py --stage assemble
CUDA_VISIBLE_DEVICES=0 .venv/bin/python _tools/train_l83_l84_specialists_and_router.py --stage router --world-mode raw --island-id 0

# 4. Анализ живой телеметрии канареечного теста широкого спреда:
.venv/bin/python _tools/analyze_canary_scalp.py

# 5. Единая выгрузка обновлений/чемпиона на боевой VPS + авто-синхронизация с Git:
.venv/bin/python _tools/deploy_vps.py
```
