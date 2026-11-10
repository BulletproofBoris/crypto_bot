#!/usr/bin/env python3
"""
_tools/test_model_soups.py
=============================================================================
Тестирование ансамблей (Model Soup / Weight Averaging) лучших чекпоинтов
Уроков 7 и 8 против одиночных лидеров и сохранение готового боевого чекпоинта
`models/prl/release_champion_v1.pt` для запуска на сервере!
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
L7_DIR = ROOT / "models/prl/lesson_7_triad_mastery"
L8_DIR = ROOT / "models/prl/lesson_8_marathon"
RELEASE_PATH = ROOT / "models/prl/release_champion_v1.pt"


def make_soup(ckpts_and_weights):
    w1_acc, b1_acc, w2_acc, b2_acc = 0.0, 0.0, 0.0, 0.0
    w_sum = sum(w for _, w in ckpts_and_weights)
    base_ck = ckpts_and_weights[0][0]
    for ck, w in ckpts_and_weights:
        alpha = float(w) / float(w_sum)
        w1_acc = w1_acc + ck["w1"].float() * alpha
        b1_acc = b1_acc + ck["b1"].float() * alpha
        w2_acc = w2_acc + ck["w2"].float() * alpha
        b2_acc = b2_acc + ck["b2"].float() * alpha
    return {
        "iteration": 1000,
        "n_slots": 24,
        "backbone_state": base_ck["backbone_state"],
        "w1": w1_acc.to(base_ck["w1"].dtype),
        "b1": b1_acc.to(base_ck["b1"].dtype),
        "w2": w2_acc.to(base_ck["w2"].dtype),
        "b2": b2_acc.to(base_ck["b2"].dtype),
    }


def run_soup_eval():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck_l7_150 = torch.load(L7_DIR / "lesson_7_champ_iter_0150.pt", map_location="cpu", weights_only=False)
    ck_l7_200 = torch.load(L7_DIR / "lesson_7_champ_iter_0200.pt", map_location="cpu", weights_only=False)
    ck_l7_300 = torch.load(L7_DIR / "lesson_7_champ_iter_0300.pt", map_location="cpu", weights_only=False)
    ck_l7_350 = torch.load(L7_DIR / "lesson_7_champ_iter_0350.pt", map_location="cpu", weights_only=False)
    ck_l7_500 = torch.load(L7_DIR / "lesson_7_champ_iter_0500.pt", map_location="cpu", weights_only=False)
    ck_l8_100 = torch.load(L8_DIR / "lesson_8_champ_iter_0100.pt", map_location="cpu", weights_only=False)
    ck_l8_150 = torch.load(L8_DIR / "lesson_8_champ_iter_0150.pt", map_location="cpu", weights_only=False)
    ck_l8_450 = torch.load(L8_DIR / "lesson_8_champ_iter_0450.pt", map_location="cpu", weights_only=False)
    ck_l8_500 = torch.load(L8_DIR / "lesson_8_champ_iter_0500.pt", map_location="cpu", weights_only=False)

    candidates = [
        ("L7_0150", ck_l7_150),
        ("L7_0300", ck_l7_300),
        ("L7_0350", ck_l7_350),
        ("L7_0500", ck_l7_500),
        ("L8_0150", ck_l8_150),
        ("L8_0500", ck_l8_500),
        # Ансамбль 1: Сбалансированное трио (Универсал L7_0150 + Защитник L7_0300 + Марафонец L8_0150)
        ("SOUP_Trio_Bal", make_soup([(ck_l7_150, 0.40), (ck_l7_300, 0.25), (ck_l8_150, 0.35)])),
        # Ансамбль 2: Дуэт лучших универсалов (L7_0150 + L8_0150)
        ("SOUP_Best2_150", make_soup([(ck_l7_150, 0.50), (ck_l8_150, 0.50)])),
        # Ансамбль 3: Защитно-активный квартет (L7_0150 + L7_0200 + L7_0300 + L8_0150)
        ("SOUP_Shield4", make_soup([(ck_l7_150, 0.30), (ck_l7_200, 0.20), (ck_l7_300, 0.25), (ck_l8_150, 0.25)])),
        # Ансамбль 4: Памп-ускоренный ансамбль (L7_0150 + L7_0350 + L7_0500 + L8_0150 + L8_0500)
        ("SOUP_Alpha5", make_soup([(ck_l7_150, 0.30), (ck_l7_350, 0.15), (ck_l7_500, 0.15), (ck_l8_150, 0.25), (ck_l8_500, 0.15)])),
        # Ансамбль 5: Усиленный L7_0150 + 30% L7_0300 (Щит)
        ("SOUP_L7_150_300", make_soup([(ck_l7_150, 0.70), (ck_l7_300, 0.30)])),
        # Ансамбль 6: Полный элитный ансамбль (Top-6)
        ("SOUP_Elite6", make_soup([(ck_l7_150, 0.25), (ck_l7_300, 0.20), (ck_l7_350, 0.10), (ck_l7_500, 0.10), (ck_l8_150, 0.25), (ck_l8_450, 0.10)])),
    ]

    n_ckpts = len(candidates)
    budget_tiers = [95.5, 1000.0, 3000.0, 5000.0, 10000.0, 20000.0]
    n_budgets = len(budget_tiers)
    pop_size = n_ckpts * n_budgets

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
    v7_58d = torch.from_numpy(raw["v7_signals_58d"][s0:].astype(np.float32)).to(device, dtype=torch.float16)
    del raw

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone.eval()
    backbone.load_state_dict(ck_l7_150["backbone_state"], strict=False)

    w1_list, b1_list, w2_list, b2_list, names = [], [], [], [], []
    for name, ck in candidates:
        names.append(name)
        w1_list.append(ck["w1"].to(device=device, dtype=torch.float16).unsqueeze(0).expand(n_budgets, -1, -1))
        b1_list.append(ck["b1"].to(device=device, dtype=torch.float16).unsqueeze(0).expand(n_budgets, -1, -1))
        w2_list.append(ck["w2"].to(device=device, dtype=torch.float16).unsqueeze(0).expand(n_budgets, -1, -1))
        b2_list.append(ck["b2"].to(device=device, dtype=torch.float16).unsqueeze(0).expand(n_budgets, -1, -1))

    W1 = torch.cat(w1_list, dim=0)
    B1 = torch.cat(b1_list, dim=0)
    W2 = torch.cat(w2_list, dim=0)
    B2 = torch.cat(b2_list, dim=0)

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
            vec_env.sell_cycles.zero_()
            vec_env.prof_sell_cycles.zero_()

            for w_start in w_list:
                vec_env.reset_with_budgets(env_budgets)
                h_states = backbone.init_hidden(batch_size=pop_size, device=device)
                peak_real_eq = env_budgets.clone()

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

                    real_eq = vec_env.cash + vec_env.pos_cost_usd
                    peak_real_eq = torch.maximum(peak_real_eq, real_eq)
                    max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - real_eq)

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

            results_by_regime[reg_name] = {
                "pnl_pct": (100.0 * tot_pnl_usd / env_budgets).view(n_ckpts, n_budgets).cpu().numpy(),
                "pnl_usd": tot_pnl_usd.view(n_ckpts, n_budgets).cpu().numpy(),
                "rdd_pct": (100.0 * max_real_dd_usd / env_budgets).view(n_ckpts, n_budgets).cpu().numpy(),
                "cyc": vec_env.sell_cycles.view(n_ckpts, n_budgets).cpu().numpy(),
                "pcyc": vec_env.prof_sell_cycles.view(n_ckpts, n_budgets).cpu().numpy(),
            }

    print("=" * 124)
    print(
        f"{'Кандидат / Soup':<16} | {'NORMAL PnL% ($)':<19} | {'DUMP PnL% (DD%)':<20} | "
        f"{'PUMP PnL% ($)':<19} | {'1:1:1 Sum ($)':<19} | {'RealMix 70/15/15'} | {'Cyc/Prof'}"
    )
    print("-" * 124)

    best_score = -1e9
    best_idx = 0
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
        real_mix = 0.70 * n_pct + 0.15 * p_pct + 0.15 * d_pct
        real_mix_usd = 0.70 * n_usd + 0.15 * p_usd + 0.15 * d_usd

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

        # Комбинированный рейтинг для выбора релизного чемпиона
        score = real_mix * 100.0 + tot_pct * 10.0
        if score > best_score:
            best_score = score
            best_idx = i

        print(
            f"{name:<16} | {n_pct:+6.2f}% (${n_usd:+7.1f}) | {d_pct:+6.2f}% (DD {d_rdd:4.2f}%) | "
            f"{p_pct:+6.2f}% (${p_usd:+7.1f}) | {tot_pct:+6.2f}% (${tot_usd:+7.1f}) | "
            f"{real_mix:+6.3f}% (${real_mix_usd:+6.1f}) | {tot_c}/{tot_pc}"
        )

    best_name, best_ck = candidates[best_idx]
    best_ck["release_name"] = best_name
    torch.save(best_ck, RELEASE_PATH)
    print("=" * 124)
    print(f"🏆 ЛУЧШИЙ БОЕВОЙ РЕЛИЗ: {best_name} -> сохранен в {RELEASE_PATH} (за {time.time() - t0:.1f}с)")


if __name__ == "__main__":
    run_soup_eval()
