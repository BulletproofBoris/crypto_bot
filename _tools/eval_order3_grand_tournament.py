#!/usr/bin/env python3
"""
_tools/eval_order3_grand_tournament.py
=============================================================================
Grand Tournament: Order-3 Champion vs Order-3 Model Soup vs Lesson 27 Champion
Слепое кросс-секционное тестирование на 24 независимых окнах истории (38.3 суток):
- Чистые старты (Clean EV)
- Подводные старты (Underwater Recovery EV)
- Памп октября (Oct 17-24)
- Откат октября (Oct 24-31)
- Ноябрьское сжатие спредов (Nov 10-17)
=============================================================================
"""

import time
import json
import sys
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.fused_order3_clean_tick import evaluate_order3_basket

MKT_PATH = BASE_DIR / "data/cache/rl_market_obs_39d.npz"
LOB_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"

CHAMP_O3_PATH = BASE_DIR / "models/prl/order3_champion/order3_champ_latest.pt"
SOUP_O3_PATH = BASE_DIR / "models/prl/order3_champion/order3_soup_top3.pt"
CHAMP_L27_PATH = BASE_DIR / "models/prl/release_champion_v1_backup_l27.pt"

REPORT_PATH = BASE_DIR / "data/reports/order3_grand_tournament_report.json"


def run_grand_tournament(device_str: str = "cuda:0", win_len: int = 3600):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🏆 GRAND TOURNAMENT: ORDER-3 VS LESSON-27 ON {device}")
    print("=" * 80)

    # 1. Загрузка данных
    print("📦 Загрузка 39D матрицы наблюдений и LOB кэша...", flush=True)
    mkt_raw = np.load(MKT_PATH)
    lob_raw = np.load(LOB_PATH)
    total_rows = len(mkt_raw["market_obs_39d"])
    tick_sz = 0.0001

    mkt_gpu = torch.from_numpy(mkt_raw["market_obs_39d"]).to(device=device, dtype=torch.bfloat16)
    b10_gpu = torch.from_numpy(lob_raw["bid_p"][:, :10]).to(device=device, dtype=torch.float32)
    a10_gpu = torch.from_numpy(lob_raw["ask_p"][:, :10]).to(device=device, dtype=torch.float32)
    bq10_gpu = torch.from_numpy(lob_raw["bid_q"][:, :10]).to(device=device, dtype=torch.float32)
    aq10_gpu = torch.from_numpy(lob_raw["ask_q"][:, :10]).to(device=device, dtype=torch.float32)
    sv_gpu = torch.from_numpy(lob_raw["sell_vol"]).to(device=device, dtype=torch.float32)
    bv_gpu = torch.from_numpy(lob_raw["buy_vol"]).to(device=device, dtype=torch.float32)

    # 2. Формирование корзины из 24 окон (длина = 3600с = 1 час каждое)
    val_indices = np.linspace(60000, total_rows - win_len - 1000, num=24, dtype=int)
    rho_cycle = [0.0, 0.0, 0.0, 0.25, 0.50, 0.70]
    off_cycle = [0.0, 0.0, 0.0, -0.015, 0.025, 0.050]

    val_basket: list[tuple[str, int, float, float]] = []
    for k_v, idx_v in enumerate(val_indices):
        r0_v = rho_cycle[k_v % len(rho_cycle)]
        of_v = off_cycle[k_v % len(off_cycle)]
        val_basket.append((f"Tourn_W{k_v:02d}", int(idx_v), r0_v, of_v))

    print(f"📊 Корзина турнира: {len(val_basket)} окон по {win_len}с каждое.")

    # 3. Загрузка кандидатов
    models_to_test = []

    # А. Order-3 Champion
    if CHAMP_O3_PATH.exists():
        ck_o3 = torch.load(CHAMP_O3_PATH, map_location="cpu", weights_only=False)
        models_to_test.append(("Order-3 Champion", ck_o3))

    # Б. Order-3 Model Soup
    if SOUP_O3_PATH.exists():
        ck_soup = torch.load(SOUP_O3_PATH, map_location="cpu", weights_only=False)
        models_to_test.append(("Order-3 Model Soup", ck_soup))

    # В. Урок 27 Champion (спроецированный в 149D)
    if CHAMP_L27_PATH.exists():
        ck_l27 = torch.load(CHAMP_L27_PATH, map_location="cpu", weights_only=False)
        gen = torch.Generator().manual_seed(20261117)
        w1_mkt = torch.randn((39, 384), generator=gen) * (1.0 / (39 ** 0.5))
        w1_ag = ck_l27["w1"][370:, :].clone()
        ck_l27_proj = dict(ck_l27)
        ck_l27_proj["w1"] = torch.cat([w1_mkt, w1_ag], dim=0)
        models_to_test.append(("Lesson-27 Champion (Legacy Baseline)", ck_l27_proj))

    print(f"🤖 Тестируемых моделей: {len(models_to_test)}")

    # 4. Запуск оценки
    cand_states = [m[1] for m in models_to_test]
    t0 = time.time()
    results = evaluate_order3_basket(
        mkt_39d_all=mkt_gpu,
        b10_all=b10_gpu,
        a10_all=a10_gpu,
        bq10_all=bq10_gpu,
        aq10_all=aq10_gpu,
        sv_all=sv_gpu,
        bv_all=bv_gpu,
        cand_states=cand_states,
        val_windows=val_basket,
        win_len=win_len,
        tick_sz=tick_sz,
        device=device,
    )
    dt = time.time() - t0
    print(f"⚡ Оценка всех моделей завершена за {dt:.2f}с!\n")

    # 5. Вывод сводной таблицы
    print("=" * 115)
    print(f"{'Модель':<32} | {'GeoScore':<10} | {'CleanEV':<10} | {'FullEV':<10} | {'CVaR25':<10} | {'Fills':<8} | {'Cnl/Fill':<8}")
    print("=" * 115)

    summary_data = []
    for (name, _), res in zip(models_to_test, results):
        print(
            f"{name:<32} | "
            f"{res['geo_score_bp']:+7.2f} bp | "
            f"{res['clean_ev_pct']:+7.4f}% | "
            f"{res['full_ev_pct']:+7.4f}% | "
            f"{res['cvar25_bp']:+7.2f} bp | "
            f"{res['fills']:7.1f} | "
            f"{res['cnl_per_fill']:7.2f}"
        )
        summary_data.append({"model": name, **res})
    print("=" * 115)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"📁 Отчет турнира сохранен в {REPORT_PATH}")
    return summary_data


if __name__ == "__main__":
    run_grand_tournament()
