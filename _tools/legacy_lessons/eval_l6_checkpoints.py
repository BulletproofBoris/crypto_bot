#!/usr/bin/env python3
"""
_tools/eval_l6_checkpoints.py
=============================================================================
Синхронный мульти-режимный и мульти-бюджетный стресс-экзамен чекпоинтов Урока 6
на ЧЕСТНОЙ ФИЗИКЕ v4.2 (Sequential Queue + Fill Waterfall, без фантомной ликвидности):

Оценивает одновременно все ключевые чекпоинты (iter_0100 .. iter_1200) на 6 бюджетах
[$95.5, $1000, $3000, $5000, $10000, $20000] по 3 архетипам рынка:
  1. NORMAL / RANGE SCALP (10 окон × 900с = 2.5ч флэта/коридора)
  2. STRESS / DUMP DEFENSE (10 окон × 900с = 2.5ч обвалов >= 2..3 тиков)
  3. PUMP / BREAKOUT SURGE (10 окон × 900с = 2.5ч пробоев/взлетов >= +2..3 тиков)
=============================================================================
"""

import json
import time
from pathlib import Path
import numpy as np
import torch

from _tools.lesson_1_motorics_trainer import MultiTaskLOBNet, build_lob_10_and_flow_tensors
from _tools.vectorized_free_multi_order_env import VectorizedFreeMultiOrderEnv

ROOT = Path("/home/restorator/crypto_bot")
CACHE_PATH = ROOT / "data/cache/rl_guard_lob_cache_600h.npz"
CKPT_DIR = ROOT / "models/prl/lesson_6_free_grid"


def run_benchmark():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    l6_dir = ROOT / "models/prl/lesson_6_free_grid"
    l7_dir = ROOT / "models/prl/lesson_7_triad_mastery"
    l8_dir = ROOT / "models/prl/lesson_8_marathon"

    labeled_paths = []
    for it in [1050, 1200]:
        p = l6_dir / f"lesson_6_champ_iter_{it:04d}.pt"
        if p.exists():
            labeled_paths.append((f"L6_{it:04d}", p))
    for it in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 525]:
        p = l7_dir / f"lesson_7_champ_iter_{it:04d}.pt"
        if p.exists():
            labeled_paths.append((f"L7_{it:04d}", p))
    for it in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 525]:
        p = l8_dir / f"lesson_8_champ_iter_{it:04d}.pt"
        if p.exists():
            labeled_paths.append((f"L8_{it:04d}", p))

    n_ckpts = len(labeled_paths)
    budget_tiers = [95.5, 1000.0, 3000.0, 5000.0, 10000.0, 20000.0]
    n_budgets = len(budget_tiers)
    pop_size = n_ckpts * n_budgets

    print("=" * 108)
    print(f"🔬 МУЛЬТИ-РЕЖИМНЫЙ ТЕСТ {n_ckpts} ЧЕКПОИНТОВ (L6 vs L7 vs L8) × {n_budgets} БЮДЖЕТОВ ({pop_size} СРЕД)")
    print("=" * 108)

    raw = np.load(CACHE_PATH)
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.01
    slice_len = min(400000, n_total)
    s0 = n_total - slice_len

    sub = {
        "bid_p": raw["bid_p"][s0:],
        "ask_p": raw["ask_p"][s0:],
        "bid_q": raw["bid_q"][s0:],
        "ask_q": raw["ask_q"][s0:],
        "buy_vol": raw["buy_vol"][s0:],
        "sell_vol": raw["sell_vol"][s0:],
        "y_sweep_12s": raw["y_sweep_12s"][s0:],
        "y_bleed_60s": raw["y_bleed_60s"][s0:],
        "y_crash_300s": raw["y_crash_300s"][s0:],
        "y_exit_30s": raw["y_exit_30s"][s0:],
        "tick_size": tick_sz,
    }
    v7_58d = torch.from_numpy(raw["v7_signals_58d"][s0:].astype(np.float32)).to(device)
    del raw

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone.eval()

    # Загружаем веса всех чекпоинтов в единый батч [pop_size, ...]
    ck0 = torch.load(labeled_paths[-1][1], map_location=device, weights_only=False)
    backbone.load_state_dict(ck0["backbone_state"], strict=False)

    w1_list, b1_list, w2_list, b2_list = [], [], [], []
    names = []
    for lbl, p in labeled_paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        names.append(lbl)
        w1 = ck["w1"].to(device=device, dtype=torch.float16).unsqueeze(0).expand(n_budgets, -1, -1)
        b1 = ck["b1"].to(device=device, dtype=torch.float16).unsqueeze(0).expand(n_budgets, -1, -1)
        w2 = ck["w2"].to(device=device, dtype=torch.float16).unsqueeze(0).expand(n_budgets, -1, -1)
        b2 = ck["b2"].to(device=device, dtype=torch.float16).unsqueeze(0).expand(n_budgets, -1, -1)
        w1_list.append(w1)
        b1_list.append(b1)
        w2_list.append(w2)
        b2_list.append(b2)

    W1 = torch.cat(w1_list, dim=0)  # [pop_size, 324, 256]
    B1 = torch.cat(b1_list, dim=0)  # [pop_size, 1, 256]
    W2 = torch.cat(w2_list, dim=0)  # [pop_size, 256, 26]
    B2 = torch.cat(b2_list, dim=0)  # [pop_size, 1, 26]
    v7_58d = v7_58d.to(dtype=torch.float16)

    lob_gpu_tmp = torch.from_numpy(lob_t).float().to(device)
    del lob_t
    with torch.no_grad():
        lob_repr_all = backbone.encode_lob_batch(lob_gpu_tmp)
    del lob_gpu_tmp

    vec_env = VectorizedFreeMultiOrderEnv(
        bids=torch.from_numpy(sub["bid_p"]).float(),
        asks=torch.from_numpy(sub["ask_p"]).float(),
        bids_q=torch.from_numpy(sub["bid_q"]).float(),
        asks_q=torch.from_numpy(sub["ask_q"]).float(),
        buy_vol=torch.from_numpy(sub["buy_vol"]).float(),
        sell_vol=torch.from_numpy(sub["sell_vol"]).float(),
        lob_tensors=torch.empty((1, 10, 4), dtype=torch.float32),
        flow_market=torch.from_numpy(flow_m).float(),
        targets=torch.from_numpy(targets_np).float(),
        num_envs=pop_size,
        n_side_slots=12,
        tick_size=tick_sz,
        device=device,
    )

    env_budgets = torch.tensor(budget_tiers, device=device).unsqueeze(0).expand(n_ckpts, n_budgets).reshape(pop_size)

    # Классифицируем окна на 3 режима: Normal (коридор), Dump (обвал), Pump (пробой вверх)
    win_len = 900
    b1_all_np = vec_env.bids[:, 0].detach().cpu().numpy()
    cand_starts = np.arange(500, slice_len - win_len - 500, 450)
    normal_w, dump_w, pump_w = [], [], []
    for w0 in cand_starts:
        seg = b1_all_np[w0 : w0 + win_len]
        end_diff = (seg[-1] - seg[0]) / tick_sz
        rng_ticks = (np.max(seg) - np.min(seg)) / tick_sz
        if end_diff <= -2.0:
            dump_w.append(int(w0))
        elif end_diff >= 2.0:
            pump_w.append(int(w0))
        elif rng_ticks >= 1.0 and abs(end_diff) <= 1.0:
            normal_w.append(int(w0))

    rng = np.random.default_rng(20261108)
    n_per_regime = 8
    regimes = {
        "NORMAL": rng.choice(normal_w, size=min(n_per_regime, len(normal_w)), replace=False).tolist(),
        "DUMP": rng.choice(dump_w, size=min(n_per_regime, len(dump_w)), replace=False).tolist(),
        "PUMP": rng.choice(pump_w, size=min(n_per_regime, len(pump_w)), replace=False).tolist(),
    }

    results_by_regime = {}
    with torch.no_grad():
        for reg_name, w_list in regimes.items():
            tot_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
            max_real_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
            max_paper_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
            vec_env.sell_cycles.zero_()
            vec_env.prof_sell_cycles.zero_()
            vec_env.partial_fills.zero_()

            for w_start in w_list:
                vec_env.reset_with_budgets(env_budgets)
                h_states = backbone.init_hidden(batch_size=pop_size, device=device)
                peak_real_eq = env_budgets.clone()
                peak_paper_eq = env_budgets.clone()

                for s in range(win_len - 1):
                    cur_t = int(w_start + s)
                    _, flow, _, multi_6d = vec_env.get_observations(cur_t_scalar=cur_t)
                    lob_r = lob_repr_all[cur_t : cur_t + 1].expand(pop_size, -1)
                    z, h_states = backbone.backbone.forward_step_fast(lob_r, flow, h_states)
                    ps = torch.sigmoid(backbone.head_sweep_12s(z))
                    pb = torch.sigmoid(backbone.head_bleed_60s(z))
                    pc = torch.sigmoid(backbone.head_crash_300s(z))
                    pe = torch.sigmoid(backbone.head_exit_30s(z))
                    v7_exp = v7_58d[cur_t].unsqueeze(0).expand(pop_size, -1)
                    z_aug = torch.cat([z, ps, pb, pc, pe, v7_exp, multi_6d], dim=-1).unsqueeze(1).to(torch.float16)

                    h1 = torch.relu(torch.bmm(z_aug, W1) + B1)
                    raw_out = torch.sigmoid((torch.bmm(h1, W2) + B2).squeeze(1).float())
                    vec_env.step(raw_out, cur_t_scalar=cur_t)

                    b1 = torch.clamp(vec_env.bids[cur_t, 0], min=tick_sz)
                    real_eq = vec_env.cash + vec_env.pos_cost_usd
                    peak_real_eq = torch.maximum(peak_real_eq, real_eq)
                    max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - real_eq)

                    paper_eq = vec_env.cash + vec_env.pos_qty * b1
                    peak_paper_eq = torch.maximum(peak_paper_eq, paper_eq)
                    max_paper_dd_usd = torch.maximum(max_paper_dd_usd, peak_paper_eq - paper_eq)

                end_t = int(w_start + win_len - 1)
                end_b1 = torch.clamp(vec_env.bids[end_t, 0], min=tick_sz)
                rem = vec_env.pos_qty > 1e-6
                underwater_vs_avg = rem & (end_b1 < (vec_env.avg_entry - 1.05 * tick_sz))
                in_profit_vs_avg = rem & (end_b1 > (vec_env.avg_entry + 1.05 * tick_sz))
                liq_val = torch.where(
                    underwater_vs_avg | in_profit_vs_avg,
                    vec_env.pos_qty * end_b1 * (1.0 - vec_env.maker_fee),
                    vec_env.pos_cost_usd,
                )
                vec_env.cash = torch.where(rem, vec_env.cash + liq_val, vec_env.cash)
                vec_env.pos_qty.zero_()
                vec_env.pos_cost_usd.zero_()
                max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - vec_env.cash)
                tot_pnl_usd += vec_env.cash - env_budgets

            pnl_pct = (100.0 * tot_pnl_usd / env_budgets).view(n_ckpts, n_budgets).cpu().numpy()
            pnl_usd = tot_pnl_usd.view(n_ckpts, n_budgets).cpu().numpy()
            rdd_pct = (100.0 * max_real_dd_usd / env_budgets).view(n_ckpts, n_budgets).cpu().numpy()
            pdd_pct = (100.0 * max_paper_dd_usd / env_budgets).view(n_ckpts, n_budgets).cpu().numpy()
            cyc = vec_env.sell_cycles.view(n_ckpts, n_budgets).cpu().numpy()
            pcyc = vec_env.prof_sell_cycles.view(n_ckpts, n_budgets).cpu().numpy()
            part = vec_env.partial_fills.view(n_ckpts, n_budgets).cpu().numpy()
            results_by_regime[reg_name] = {
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "rdd_pct": rdd_pct,
                "pdd_pct": pdd_pct,
                "cyc": cyc,
                "pcyc": pcyc,
                "part": part,
            }

    print(
        f"{'Чекпоинт':<11} | {'NORMAL PnL% ($)':<20} | {'DUMP PnL% (RealDD%)':<22} | "
        f"{'PUMP PnL% ($)':<20} | {'ИТОГО PnL% (Sum $)':<22} | {'Cyc/Prof (Part)'}"
    )
    print("-" * 118)

    out_records = []
    for i, name in enumerate(names):
        n_pct = float(results_by_regime["NORMAL"]["pnl_pct"][i].mean())
        n_usd = float(results_by_regime["NORMAL"]["pnl_usd"][i].sum())
        d_pct = float(results_by_regime["DUMP"]["pnl_pct"][i].mean())
        d_rdd = float(results_by_regime["DUMP"]["rdd_pct"][i].mean())
        d_usd = float(results_by_regime["DUMP"]["pnl_usd"][i].sum())
        p_pct = float(results_by_regime["PUMP"]["pnl_pct"][i].mean())
        p_usd = float(results_by_regime["PUMP"]["pnl_usd"][i].sum())

        tot_pct = n_pct + d_pct + p_pct
        tot_usd = n_usd + d_usd + p_usd
        tot_c = int(
            results_by_regime["NORMAL"]["cyc"][i].mean()
            + results_by_regime["DUMP"]["cyc"][i].mean()
            + results_by_regime["PUMP"]["cyc"][i].mean()
        )
        tot_pc = int(
            results_by_regime["NORMAL"]["pcyc"][i].mean()
            + results_by_regime["DUMP"]["pcyc"][i].mean()
            + results_by_regime["PUMP"]["pcyc"][i].mean()
        )
        tot_pf = int(
            results_by_regime["NORMAL"]["part"][i].mean()
            + results_by_regime["DUMP"]["part"][i].mean()
            + results_by_regime["PUMP"]["part"][i].mean()
        )
        print(
            f"{name:<11} | {n_pct:+6.2f}% (${n_usd:+7.1f}) | {d_pct:+6.2f}% (DD {d_rdd:5.2f}%) | "
            f"{p_pct:+6.2f}% (${p_usd:+7.1f}) | {tot_pct:+6.2f}% (${tot_usd:+8.1f}) | {tot_c}/{tot_pc} ({tot_pf})"
        )
        out_records.append(
            {
                "checkpoint": name,
                "normal_pnl_pct": round(n_pct, 3),
                "normal_pnl_usd": round(n_usd, 2),
                "dump_pnl_pct": round(d_pct, 3),
                "dump_real_dd_pct": round(d_rdd, 3),
                "dump_pnl_usd": round(d_usd, 2),
                "pump_pnl_pct": round(p_pct, 3),
                "pump_pnl_usd": round(p_usd, 2),
                "total_pnl_pct": round(tot_pct, 3),
                "total_pnl_usd": round(tot_usd, 2),
                "cycles": tot_c,
                "prof_cycles": tot_pc,
                "partial_fills": tot_pf,
            }
        )

    out_path = CKPT_DIR / "eval_l6_checkpoints_v42.json"
    out_path.write_text(json.dumps(out_records, indent=2), encoding="utf-8")
    print("=" * 108)
    print(f"✅ Готово за {time.time() - t0:.1f}с! Отчет сохранен в {out_path}")


if __name__ == "__main__":
    run_benchmark()
