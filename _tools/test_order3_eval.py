#!/usr/bin/env python3
"""
_tools/test_order3_eval.py
=============================================================================
Быстрый smoke-test для Order-3:
- Загружает 39D рыночные наблюдения и LOB кэш
- Инициализирует модель с хирургическим warm-start из L27
- Запускает evaluate_order3_basket на 2 тестовых окнах по 2400 тиков на GPU
=============================================================================
"""

import time
import numpy as np
import torch
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.fused_order3_clean_tick import (
    evaluate_order3_basket,
    MKT_DIM_ORDER3,
    AG_DIM_ORDER3,
    TOTAL_DIM_ORDER3,
    K_MODES_ORDER3,
    EXPERT_OUT_DIM_ORDER3,
    MOE_OUT_DIM_ORDER3,
    ROUTER_IN_DIM_ORDER3,
    ROUTER_H_DIM_ORDER3,
    REG_IN_DIM_ORDER3,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Smoke-test Order-3 Evaluator on {device}...")

    # 1. Загрузка кэшей
    print("📦 Загрузка 39D рыночной матрицы и LOB кэша...")
    t0 = time.time()
    mkt_raw = np.load(BASE_DIR / "data/cache/rl_market_obs_39d.npz")
    lob_raw = np.load(BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz")

    # Берем тестовый срез (последние 200,000 тиков)
    s0 = len(mkt_raw["market_obs_39d"]) - 200000
    mkt_gpu = torch.from_numpy(mkt_raw["market_obs_39d"][s0:]).to(device=device, dtype=torch.bfloat16)
    b10_gpu = torch.from_numpy(lob_raw["bid_p"][s0:, :10]).to(device=device, dtype=torch.float32)
    a10_gpu = torch.from_numpy(lob_raw["ask_p"][s0:, :10]).to(device=device, dtype=torch.float32)
    bq10_gpu = torch.from_numpy(lob_raw["bid_q"][s0:, :10]).to(device=device, dtype=torch.float32)
    aq10_gpu = torch.from_numpy(lob_raw["ask_q"][s0:, :10]).to(device=device, dtype=torch.float32)
    sv_gpu = torch.from_numpy(lob_raw["sell_vol"][s0:]).to(device=device, dtype=torch.float32)
    bv_gpu = torch.from_numpy(lob_raw["buy_vol"][s0:]).to(device=device, dtype=torch.float32)
    tick_sz = 0.0001
    print(f"   Загружено за {time.time() - t0:.2f}с")

    # 2. Загрузка и трансформация весов L27
    print("🧩 Инициализация весов с хирургическим переносом L27...")
    ck = torch.load(BASE_DIR / "models/prl/release_champion_v1_backup_l27.pt", weights_only=False, map_location="cpu")
    dt = torch.bfloat16

    gen = torch.Generator().manual_seed(20261117)
    w1_mkt = torch.randn((MKT_DIM_ORDER3, 384), generator=gen) * (1.0 / (MKT_DIM_ORDER3 ** 0.5))
    w1_ag = ck["w1"][370:, :].clone()  # 110 строк
    w1_full = torch.cat([w1_mkt, w1_ag], dim=0)

    cand_state = {
        "w1": w1_full,
        "b1": ck["b1"].view(1, 384),
        "w_res1": ck["w_res1"],
        "b_res1": ck["b_res1"].view(1, 128),
        "w_res2": ck["w_res2"],
        "w2_moe": ck["w2_moe"],
        "b2_moe": ck["b2_moe"].view(1, MOE_OUT_DIM_ORDER3),
        "w_gate1": ck["w_gate1"],
        "b_gate1": ck["b_gate1"].view(1, ROUTER_H_DIM_ORDER3),
        "w_gate2": ck["w_gate2"],
        "b_gate2": ck["b_gate2"].view(1, K_MODES_ORDER3),
        "w_rec": ck["w_rec"],
        "b_rec": ck["b_rec"].view(1, 32),
        "w_reg1": ck["w_reg1"],
        "b_reg1_b": ck["b_reg1_b"].view(1, 48),
        "b_reg1_a": ck["b_reg1_a"].view(1, 48),
        "w_reg2": ck["w_reg2"],
        "b_reg2_b": ck["b_reg2_b"].view(1, 2),
        "b_reg2_a": ck["b_reg2_a"].view(1, 2),
    }

    # 3. Запуск оценки на 2 окнах (чистый старт и подводный старт)
    val_windows = [
        ("clean_win1", 10000, 0.0, 0.0),
        ("underwater_win2", 40000, 0.35, -0.015),
    ]

    print("⚡ Выполнение evaluate_order3_basket на GPU (2 окна x 6 бюджетов = 12 сессий)...")
    t1 = time.time()
    res = evaluate_order3_basket(
        mkt_39d_all=mkt_gpu,
        b10_all=b10_gpu,
        a10_all=a10_gpu,
        bq10_all=bq10_gpu,
        aq10_all=aq10_gpu,
        sv_all=sv_gpu,
        bv_all=bv_gpu,
        cand_states=[cand_state],
        val_windows=val_windows,
        win_len=2400,
        tick_sz=tick_sz,
        device=device,
    )
    dt = time.time() - t1
    print(f"✅ Успешно завершено за {dt:.2f}с!")
    r = res[0]
    print(f"📊 Результаты:")
    print(f"   Clean EV:       {r['clean_ev_pct']:+.4f}%")
    print(f"   Full EV:        {r['full_ev_pct']:+.4f}%")
    print(f"   Geo Score:      {r['geo_score_bp']:+.2f} bp")
    print(f"   CVaR25:         {r['cvar25_bp']:+.2f} bp")
    print(f"   Fills:          {r['fills']:.1f}")
    print(f"   Cancels:        {r['cancels']:.1f}")
    print(f"   Cnl/Fill Ratio: {r['cnl_per_fill']:.2f}")
    print(f"   Mode Probs:     {r['mode_probs']}")
    print(f"   Router MI:      {r['mode_mi_bits']:.4f} bits")


if __name__ == "__main__":
    main()
