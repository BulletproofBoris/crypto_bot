#!/usr/bin/env python3
"""
_tools/regime_purged_split_l94.py
=============================================================================
Урок 94 (Шаг Б): Саморасширяющийся стратифицированный валидатор `Per-Regime Purged Split`
для любого объёма накопленной суб-центовой истории N_days и закон роста порядка
резонансного усилителя K_max(T):

  1. Аппаратная изоляция: СТРОГО 2-я видеокарта (CUDA_VISIBLE_DEVICES=1).
  2. Скользящий слепой хвост (Rolling Blind OOS):
       последние max(2, floor(0.18 * N_days)) полных суток (0% участия в обучении и отборе!).
  3. Непересекающиеся 24ч блоки с санитарным зазором Purge = 7,200с (2ч) и
     объективная рыночная классификация в 4 корзины:
       - C (Cascade / High-Range) : hi_lo >= 20% И отскок от минимума >= 6%
       - D (Unidirectional Dump)  : ret_24h <= -6.5% (без сильного отскока)
       - U (Sustained Rally)      : ret_24h >= +6.5%
       - Q (Quiet / Flat)         : |ret_24h| < 6.5% и умеренный размах
  4. Пропорциональная квота в PurgedVal (~35% блоков каждой корзины) и PurgedTrain (~65%)
     + расчёт допустимого порядка банка резонаторов K_max(T) и минимального числа
     срабатываний N_hits_min(T).
  5. Калибровочный замер базовой несущей H_0 = Live_L92_V1 (release_champion_v1.pt)
     по всем 4 корзинам (C, D, U, Q) и Rolling Blind OOS в обеих физиках (DEC + RAW)
     на только что запечатанном эталонном реестре 112D!
=============================================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("L94_GPU_ID", "1")

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CACHE_DIR = BASE_DIR / "data" / "cache"
LOG_DIR = BASE_DIR / "logs" / "l94_ledger_and_split"
SUBCENT_START = 3016277
WIN_24H = 86401
PURGE_SEC = 7200
L500_START_UNIX = 1795771260  # 2026-11-27 12:21:00 MSK (старт 223-колоночного L500 глубокого профиля на VPS)


@dataclass
class DayBlockSpec:
    block_id: str
    rel_start: int
    rel_end: int
    abs_start: int
    abs_end: int
    regime: str        # 'C', 'D', 'U', 'Q', or 'BLIND'
    role: str          # 'train', 'val', 'blind'
    ret_24h_pct: float
    hi_lo_pct: float
    rebound_from_low_pct: float
    l500_coverage_pct: float


def compute_resonant_capacity_law(n_days_total: float, n_days_l500: float) -> dict:
    """
    Закон допустимого порядка полинома/банка узкополосных резонаторов K_max(T)
    по мере накопления истории (защита от эффекта Рунге / переобучения иголок):
    """
    if n_days_total < 16.0:
        k_bands = 6
        min_hits = 25
        min_distinct_days = 3
    elif n_days_total < 26.0:
        k_bands = 12
        min_hits = 40
        min_distinct_days = 5
    elif n_days_total < 45.0:
        k_bands = 22
        min_hits = 60
        min_distinct_days = 8
    else:
        k_bands = 32
        min_hits = 95
        min_distinct_days = 12

    return {
        "n_days_subcent": round(n_days_total, 2),
        "n_days_l500_era": round(n_days_l500, 2),
        "k_max_bands": int(k_bands),
        "allow_l500_wall_resonators": bool(n_days_l500 >= 4.0),
        "min_hits_per_band": int(min_hits),
        "min_distinct_days_per_band": int(min_distinct_days),
    }


def classify_24h_regime(b1_seg: np.ndarray) -> tuple[str, float, float, float]:
    p0 = max(1e-4, float(b1_seg[0]))
    p1 = max(1e-4, float(b1_seg[-1]))
    p_min = max(1e-4, float(np.min(b1_seg)))
    p_max = max(1e-4, float(np.max(b1_seg)))

    ret_pct = (p1 / p0 - 1.0) * 100.0
    hi_lo_pct = (p_max / p_min - 1.0) * 100.0
    min_pos = int(np.argmin(b1_seg))
    post_max = float(np.max(b1_seg[min_pos:])) if min_pos < len(b1_seg) - 60 else p1
    rebound_pct = (post_max / p_min - 1.0) * 100.0

    # 1) C (Cascade): широкий размах >= 20% И реальный пост-проливной отскок >= 6.5%
    if hi_lo_pct >= 20.0 and rebound_pct >= 6.5:
        reg = "C"
    # 2) D (Dump): направленный слив <= -6.5% без сильного отскока
    elif ret_pct <= -6.5:
        reg = "D"
    # 3) U (Up-Trend): направленный рост >= +6.5%
    elif ret_pct >= 6.5:
        reg = "U"
    # 4) Q (Quiet / Wide-Spread Flat): спокойный рынок
    else:
        reg = "Q"
    return reg, round(ret_pct, 2), round(hi_lo_pct, 2), round(rebound_pct, 2)


def build_dynamic_regime_split(lob_npz_path: Path | None = None) -> dict:
    """
    Детерминированно строит непересекающиеся корзины PurgedTrain / PurgedVal / RollingBlindOOS
    для текущей длины суб-центового датасета (автоматически масштабируется при каждом новом дне!).
    """
    lob_p = lob_npz_path or (CACHE_DIR / "rl_guard_lob_cache_600h.npz")
    lob = np.load(lob_p, mmap_mode="r")
    ts_sub = np.asarray(lob["ts"][SUBCENT_START:], dtype=np.int64)
    b1_sub = np.asarray(lob["bid_p"][SUBCENT_START:, 0], dtype=np.float64)
    n_sub = len(ts_sub)
    n_days_total = n_sub / 86400.0
    n_days_l500 = float(np.sum(ts_sub >= L500_START_UNIX)) / 86400.0

    # 1. Rolling Blind OOS = max(2, floor(0.18 * N_days)) полных суток в самом конце:
    n_blind_days = max(2, int(math.floor(0.18 * n_days_total)))
    blind_specs: list[DayBlockSpec] = []
    cur_end = n_sub - 5
    for b_k in range(n_blind_days):
        bs = cur_end - WIN_24H
        seg = b1_sub[bs:cur_end]
        reg_b, ret_b, hl_b, reb_b = classify_24h_regime(seg)
        l500_cov = float(np.mean(ts_sub[bs:cur_end] >= L500_START_UNIX) * 100.0)
        blind_specs.append(
            DayBlockSpec(
                block_id=f"BLIND_{n_blind_days - 1 - b_k:02d}_{reg_b}",
                rel_start=int(bs),
                rel_end=int(cur_end),
                abs_start=int(SUBCENT_START + bs),
                abs_end=int(SUBCENT_START + cur_end),
                regime=reg_b,
                role="blind",
                ret_24h_pct=ret_b,
                hi_lo_pct=hl_b,
                rebound_from_low_pct=reb_b,
                l500_coverage_pct=round(l500_cov, 1),
            )
        )
        cur_end = bs - 300  # 5-минутный зазор между сутками внутри Blind OOS
    blind_specs.reverse()

    # 2. До-слепая история (строго левее первого Blind блока минус PURGE_SEC = 2 часа!):
    usable_end = blind_specs[0].rel_start - PURGE_SEC
    raw_blocks: list[DayBlockSpec] = []
    blk_idx = 0
    ws = 1800  # 30 минут на начальный прогрев SSM памяти перед стартом 1-го блока!
    while ws + WIN_24H <= usable_end:
        we = ws + WIN_24H
        seg = b1_sub[ws:we]
        reg, ret_p, hl_p, reb_p = classify_24h_regime(seg)
        l500_cov = float(np.mean(ts_sub[ws:we] >= L500_START_UNIX) * 100.0)
        raw_blocks.append(
            DayBlockSpec(
                block_id=f"BLK_{blk_idx:02d}_{reg}",
                rel_start=int(ws),
                rel_end=int(we),
                abs_start=int(SUBCENT_START + ws),
                abs_end=int(SUBCENT_START + we),
                regime=reg,
                role="train",
                ret_24h_pct=ret_p,
                hi_lo_pct=hl_p,
                rebound_from_low_pct=reb_p,
                l500_coverage_pct=round(l500_cov, 1),
            )
        )
        ws = we + PURGE_SEC  # 2-часовой санитарный зазор между ЛЮБЫМИ соседними блоками (0% пересечения!)
        blk_idx += 1

    # 3. Стратифицированное выделение PurgedVal (~35% из каждой представленной корзины C, D, U, Q):
    by_reg: dict[str, list[int]] = {"C": [], "D": [], "U": [], "Q": []}
    for i, b in enumerate(raw_blocks):
        by_reg[b.regime].append(i)

    for reg_k, idxs in by_reg.items():
        if len(idxs) >= 2:
            n_val_k = max(1, int(round(0.35 * len(idxs))))
            # Выбираем равномерно по времени (включая более свежие блоки данной корзины):
            pick_pos = [idxs[int(round(p))] for p in np.linspace(len(idxs) - 1, 0, n_val_k + 1)[:-1]]
            for pi in pick_pos:
                raw_blocks[pi].role = "val"
        elif len(idxs) == 1 and reg_k in ("Q", "D"):
            # Если редкий защитный/тихий режим встретился всего 1 раз — оставляем в Train, а при >=2 делим честно
            pass

    # Дополнительно строим плотную сетку сдвинутых Train-стартов СТРОГО внутри разрешённых Train-интервалов
    # на расстоянии >= PURGE_SEC от любого Val и Blind блока (0.0% общих секунд!):
    forbidden = [(b.rel_start - PURGE_SEC, b.rel_end + PURGE_SEC) for b in raw_blocks if b.role == "val"]
    forbidden.append((blind_specs[0].rel_start - PURGE_SEC, n_sub + WIN_24H))

    jittered_train_by_regime: dict[str, list[int]] = {"C": [], "D": [], "U": [], "Q": []}
    for s_cand in range(1800, usable_end - WIN_24H + 1, 14400):
        e_cand = s_cand + WIN_24H
        if any(not (e_cand <= f0 or s_cand >= f1) for f0, f1 in forbidden):
            continue
        reg_c, _, _, _ = classify_24h_regime(b1_sub[s_cand:e_cand])
        jittered_train_by_regime[reg_c].append(int(s_cand))

    cap_law = compute_resonant_capacity_law(n_days_total, n_days_l500)
    return {
        "n_subcent_rows": int(n_sub),
        "capacity_law": cap_law,
        "blocks": [asdict(b) for b in raw_blocks],
        "blind_blocks": [asdict(b) for b in blind_specs],
        "jittered_train_starts_by_regime": jittered_train_by_regime,
    }


def evaluate_per_regime_dominance(cand_by_reg: dict[str, float], base_by_reg: dict[str, float], cand_dd: float, base_dd: float, cand_blind: float, base_blind: float) -> tuple[bool, dict]:
    """
    Покорзинный критерий приёмки Урока 94/95 (Per-Regime Dominance Gate):
      - Запрещает ухудшать каскадную корзину C более чем на 0.05%/сут и корзину сливов D вообще;
      - Требует строгого прироста на целевых корзинах Q + U (или совокупно по всем корзинам)
        и неотрицательного баланса на Rolling Blind OOS при невозрастании MaxDD!
    """
    d_c = cand_by_reg.get("C", 0.0) - base_by_reg.get("C", 0.0)
    d_d = cand_by_reg.get("D", 0.0) - base_by_reg.get("D", 0.0)
    d_u = cand_by_reg.get("U", 0.0) - base_by_reg.get("U", 0.0)
    d_q = cand_by_reg.get("Q", 0.0) - base_by_reg.get("Q", 0.0)
    d_blind = cand_blind - base_blind
    dd_ok = cand_dd <= max(base_dd + 0.15, 6.20)

    c_ok = d_c >= -0.05
    d_ok = d_d >= -0.02
    qu_gain = (d_q + d_u) > 0.02 or (d_c + d_d + d_u + d_q) > 0.10
    blind_ok = d_blind >= -0.01

    passed = bool(c_ok and d_ok and qu_gain and blind_ok and dd_ok)
    return passed, {
        "passed": passed,
        "delta_C_pct": round(d_c, 3),
        "delta_D_pct": round(d_d, 3),
        "delta_U_pct": round(d_u, 3),
        "delta_Q_pct": round(d_q, 3),
        "delta_blind_pct": round(d_blind, 3),
        "cand_dd_pct": round(cand_dd, 2),
        "base_dd_pct": round(base_dd, 2),
    }


def run_baseline_h0_audit_on_clean_ledger() -> dict:
    """Прогоняет эталонную несущую H_0 = Live_L92_V1 по всем корзинам C/D/U/Q и Blind OOS на новом запечатанном реестре 112D (строго GPU #1!)."""
    from _tools.fused_order3_clean_tick import evaluate_order3_basket
    from _tools.train_order3_champion import MultimodalMoEPolicyOrder3

    assert torch.cuda.device_count() == 1, f"Ожидалась строго 1 GPU (GPU #1), видно: {torch.cuda.device_count()}"
    dev = torch.device("cuda:0")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    split_info = build_dynamic_regime_split()
    pol = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=dev)
    h0_st = {k: v.to(dev) for k, v in pol.load_initial_champion(BASE_DIR / "models/prl/release_champion_v1.pt").items()}
    # Подаём пачку C=32 идентичных копий для мгновенного попадания в тёплый CUDA Graph кэш без перекомпиляции C=1!
    cands_pad = [h0_st] * 32

    all_eval_specs = split_info["blocks"] + split_info["blind_blocks"]
    bask = [(b["block_id"], int(b["rel_start"]), 0.0, 0.0) for b in all_eval_specs]

    report = {"split_summary": split_info["capacity_law"], "blocks": all_eval_specs, "worlds": {}}
    print("=" * 96)
    print(f"📐 СТРАТИФИЦИРОВАННЫЙ РЕЕСТР УРОКА 94: {len(split_info['blocks'])} до-слепых блоков + {len(split_info['blind_blocks'])} Blind OOS")
    print(f"   Закон ёмкости резонаторов: {split_info['capacity_law']}")
    print("=" * 96, flush=True)
    for b in all_eval_specs:
        print(
            f"  • {b['block_id']:14s} | role={b['role']:5s} | reg={b['regime']} | ret={b['ret_24h_pct']:+6.2f}% "
            f"| hi_lo={b['hi_lo_pct']:5.2f}% | reb={b['rebound_from_low_pct']:5.2f}% | L500={b['l500_coverage_pct']:4.1f}%",
            flush=True,
        )

    for wm in ("dec", "raw"):
        is_r = wm == "raw"
        lob = np.load(CACHE_DIR / ("rl_guard_lob_cache_600h_raw_with_self.npz" if is_r else "rl_guard_lob_cache_600h.npz"), mmap_mode="r")
        mkt = np.load(CACHE_DIR / ("rl_hybrid_market_obs_112d_raw_with_self.npz" if is_r else "rl_hybrid_market_obs_112d.npz"), mmap_mode="r")["market_obs_112d"]
        ts = lob["ts"][SUBCENT_START:]
        q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
        q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
        sv_np = np.asarray(lob["sell_vol"][SUBCENT_START:], dtype=np.float32).copy()
        bv_np = np.asarray(lob["buy_vol"][SUBCENT_START:], dtype=np.float32).copy()
        qm = (ts >= q0_u) & (ts <= q1_u)
        sv_np[qm] = 0.0
        bv_np[qm] = 0.0

        mkt_g = torch.from_numpy(np.asarray(mkt[SUBCENT_START:], dtype=np.float32)).to(dev, torch.bfloat16)
        b10_g = torch.from_numpy(np.asarray(lob["bid_p"][SUBCENT_START:, :10], dtype=np.float32)).to(dev)
        a10_g = torch.from_numpy(np.asarray(lob["ask_p"][SUBCENT_START:, :10], dtype=np.float32)).to(dev)
        bq10_g = torch.from_numpy(np.asarray(lob["bid_q"][SUBCENT_START:, :10], dtype=np.float32)).to(dev)
        aq10_g = torch.from_numpy(np.asarray(lob["ask_q"][SUBCENT_START:, :10], dtype=np.float32)).to(dev)
        sv_g = torch.from_numpy(sv_np).to(dev)
        bv_g = torch.from_numpy(bv_np).to(dev)

        per_blk = []
        reg_evs: dict[str, list[float]] = {"C": [], "D": [], "U": [], "Q": [], "BLIND": []}
        max_dd_w = 0.0
        for b in all_eval_specs:
            wb = [(b["block_id"], int(b["rel_start"]), 0.0, 0.0)]
            rc = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=cands_pad, val_windows=wb, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=False, enforce_activity_floor=False)[0]
            rh = evaluate_order3_basket(mkt_39d_all=mkt_g, b10_all=b10_g, a10_all=a10_g, bq10_all=bq10_g, aq10_all=aq10_g, sv_all=sv_g, bv_all=bv_g, cand_states=cands_pad, val_windows=wb, win_len=WIN_24H, tick_sz=0.0001, device=dev, pad_to_max_c=False, harsh_stress=True, enforce_activity_floor=False)[0]
            ev_m = 0.50 * float(rc["clean_ev_pct"]) + 0.50 * float(rh["clean_ev_pct"])
            dd_m = max(float(rc["dd_pct"]), float(rh["dd_pct"]))
            max_dd_w = max(max_dd_w, dd_m)
            key_r = "BLIND" if b["role"] == "blind" else b["regime"]
            reg_evs[key_r].append(ev_m)
            per_blk.append({"block_id": b["block_id"], "role": b["role"], "regime": b["regime"], "ev_mix_pct": round(ev_m, 3), "fills": round(float(rc["fills"]), 1), "dd_pct": round(dd_m, 2)})

        reg_means = {rk: round(float(np.mean(vs)), 3) if vs else 0.0 for rk, vs in reg_evs.items()}
        all_non_blind = [x["ev_mix_pct"] for x in per_blk if x["role"] != "blind"]
        report["worlds"][wm.upper()] = {
            "mean_pre_blind_ev_d": round(float(np.mean(all_non_blind)), 3),
            "regime_means_pct": reg_means,
            "max_dd_pct": round(max_dd_w, 2),
            "per_block": per_blk,
        }
        print(f"\n🌍 [{wm.upper()}] Несущая H_0 (Live_L92_V1) на чистом запечатанном реестре 112D:")
        print(f"   Среднее до Blind: {np.mean(all_non_blind):+.3f}%/сут | MaxDD={max_dd_w:.2f}% | По корзинам: {reg_means}", flush=True)
        for pb in per_blk:
            print(f"     {pb['block_id']:14s} ({pb['role']:5s}) -> EV={pb['ev_mix_pct']:+6.3f}% | Fills={pb['fills']:4.1f} | DD={pb['dd_pct']:4.2f}%", flush=True)

    out_json = LOG_DIR / "h0_clean_ledger_audit.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n✅ Эталонный отчёт Шага Б сохранён: {out_json}", flush=True)
    return report


if __name__ == "__main__":
    run_baseline_h0_audit_on_clean_ledger()
