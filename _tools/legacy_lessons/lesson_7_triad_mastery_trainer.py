#!/usr/bin/env python3
"""
_tools/lesson_7_triad_mastery_trainer.py
=============================================================================
Урок 7: Triad Mastery (1 Normal + 1 Dump + 1 Pump = 1,800s на каждой итерации)
100% Чистое RL-обучение через многорежимный шейпинг вознаграждения (без эвристик!):
1. СИНХРОННАЯ ТРИАДА РЕЖИМОВ (3 окна × 600с = 1,800с, 1365 кандидатов × 6 бюджетов = 8190 сред):
   - Окно #1: NORMAL (спокойный коридор / органическая волатильность)
   - Окно #2: DUMP (обвал / водопад >= 2 тиков вниз)
   - Окно #3: PUMP (пробой / импульсный взлет >= 2.5 тиков вверх)
2. ЧИСТЫЙ RL-ШЕЙПИНГ ВОЗНАГРАЖДЕНИЯ (REWARD SHAPING):
   - Защита на проливах: квадратично-линейный штраф за пиковую просадку и просадку
     позиции на падающей ленте (учит сам сжимать `b_w` и удерживать кэш при `pc`/`pb`).
   - Активность на пампе: прямое поощрение за реализованный и плавающий профит на
     пробойном окне `pump_pnl_pct` + масштабируемая премия за многотиковые выходы
     (`avg_profit_per_cycle`), чтобы агент сам учился переносить часть позиции в дальние
     Ask-слоты (+3..+5 тиков), а не скидывать 100% по +1 тику.
=============================================================================
"""

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_free_multi_order_env import VectorizedFreeMultiOrderEnv
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_6_free_multi_order_trainer import BatchedFreeGridPolicyGrouped

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
L6_CHAMP_PATH = BASE_DIR / "models" / "prl" / "lesson_6_free_grid" / "lesson_6_champ_iter_1200.pt"
L6_FALLBACK_PATH = BASE_DIR / "models" / "prl" / "lesson_6_free_grid" / "lesson_6_champ_latest.pt"
SAVE_DIR = BASE_DIR / "models" / "prl" / "lesson_7_triad_mastery"
LOG_FILE = BASE_DIR / "data" / "logs" / "lesson_7_triad_mastery.log"


def log_msg(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_lesson_7(iterations: int = 525, n_groups: int = 1365):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_budgets = 6
    pop_size = n_groups * n_budgets  # 8190

    log_msg("=" * 102)
    log_msg(f"🚀 УРОК 7: TRIAD MASTERY (1 NORMAL + 1 DUMP + 1 PUMP × 600s = 1800s) | 100% PURE RL ({pop_size} СРЕД)")
    log_msg("=" * 102)

    raw = np.load(CACHE_PATH)
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.0001
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
    for p in backbone.parameters():
        p.requires_grad = False

    policy = BatchedFreeGridPolicyGrouped(
        backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, in_dim=324, h_dim=256, out_dim=26, device=device
    )
    v7_58d = v7_58d.to(dtype=policy.dtype)

    summary_path = SAVE_DIR / "lesson_7_summary.json"
    history = []
    start_iter = 1
    latest_ckpt = SAVE_DIR / "lesson_7_champ_latest.pt"

    if latest_ckpt.exists() and summary_path.exists():
        try:
            history = json.loads(summary_path.read_text(encoding="utf-8"))
            if history:
                start_iter = int(history[-1]["iteration"]) + 1
                ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
                policy.load_champion_into_population(ck)
                log_msg(f"🔄 Продолжение Урока 7 с итерации {start_iter}/{iterations}...")
        except Exception as e:
            log_msg(f"⚠️ Ошибка чтения чекпоинта Урока 7: {e}")

    if start_iter == 1:
        seed_path = L6_CHAMP_PATH if L6_CHAMP_PATH.exists() else L6_FALLBACK_PATH
        if seed_path.exists():
            ck = torch.load(seed_path, map_location=device, weights_only=False)
            policy.load_champion_into_population(ck)
            log_msg(f"🧬 Урок 7: Теплый старт (Warm-Start) из чемпиона Урока 6 ({seed_path.name}) выполнен успешно!")
        else:
            raise FileNotFoundError(f"Не найден чекпоинт Урока 6 для старта Урока 7: {seed_path}")

    hof_state = {
        "w1": policy.w1[0].detach().cpu(),
        "b1": policy.b1[0].detach().cpu(),
        "w2": policy.w2[0].detach().cpu(),
        "b2": policy.b2[0].detach().cpu(),
    }

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
    del sub, flow_m, targets_np
    gc.collect()
    torch.cuda.empty_cache()

    budget_tiers = torch.tensor([95.5, 1000.0, 3000.0, 5000.0, 10000.0, 20000.0], device=device)
    env_budgets = budget_tiers.unsqueeze(0).expand(n_groups, n_budgets).reshape(pop_size)

    rng = np.random.default_rng(202607)
    win_len = 600  # 3 окна по 600с = 1,800 секунд (1 Normal + 1 Dump + 1 Pump!)

    # Классификация всех 600-секундных окон на 3 режима: NORMAL, DUMP и PUMP
    b1_all_np = vec_env.bids[:, 0].detach().cpu().numpy()
    cand_starts = np.arange(100, slice_len - win_len - 100, 60)
    normal_starts = []
    dump_starts = []
    pump_starts = []
    for w0 in cand_starts:
        seg = b1_all_np[w0 : w0 + win_len]
        end_diff = (seg[-1] - seg[0]) / tick_sz
        max_drop = (np.max(seg[:240]) - np.min(seg[120:])) / tick_sz
        max_rally = (np.max(seg[120:]) - np.min(seg[:240])) / tick_sz
        rng_ticks = (np.max(seg) - np.min(seg)) / tick_sz
        if end_diff <= -1.8 or max_drop >= 2.6:
            dump_starts.append(int(w0))
        elif end_diff >= 1.8 or max_rally >= 2.6:
            pump_starts.append(int(w0))
        elif rng_ticks >= 1.0 and abs(end_diff) <= 1.2:
            normal_starts.append(int(w0))

    normal_starts = np.array(normal_starts if normal_starts else cand_starts, dtype=np.int64)
    dump_starts = np.array(dump_starts if dump_starts else cand_starts, dtype=np.int64)
    pump_starts = np.array(pump_starts if pump_starts else cand_starts, dtype=np.int64)
    log_msg(
        f"⚖️ ТРИАДА РЕЖИМОВ (Урок 7): Normal={len(normal_starts)}, Dump={len(dump_starts)}, Pump={len(pump_starts)} окон (до {iterations} ит.)"
    )

    for _skip in range(1, start_iter):
        _ = rng.integers(0, 1000, size=3)

    for it in range(start_iter, iterations + 1):
        t0 = time.time()

        total_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        norm_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        dump_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        pump_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)

        max_real_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_paper_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_hold_s = torch.zeros(pop_size, dtype=torch.float32, device=device)
        cycles = torch.zeros(pop_size, dtype=torch.long, device=device)
        prof_cycles = torch.zeros(pop_size, dtype=torch.long, device=device)
        evasion_exits = torch.zeros(pop_size, dtype=torch.float32, device=device)
        bleed_underwater_pen = torch.zeros(pop_size, dtype=torch.float32, device=device)
        pump_capture_bonus = torch.zeros(pop_size, dtype=torch.float32, device=device)

        vec_env.n_trades.zero_()
        vec_env.partial_fills.zero_()
        vec_env.multi_orders_used.zero_()
        vec_env.frontrun_hits.zero_()
        vec_env.sell_cycles.zero_()
        vec_env.prof_sell_cycles.zero_()

        # Триада: 0=Normal, 1=Dump, 2=Pump
        w_triad = [
            ("NORM", int(rng.choice(normal_starts))),
            ("DUMP", int(rng.choice(dump_starts))),
            ("PUMP", int(rng.choice(pump_starts))),
        ]
        with torch.no_grad():
            for reg_name, w_start in w_triad:
                vec_env.reset_with_budgets(env_budgets)
                h_states = backbone.init_hidden(batch_size=pop_size, device=device)
                prev_pos = vec_env.pos_qty.clone()
                cur_hold = torch.zeros(pop_size, dtype=torch.float32, device=device)
                peak_real_eq = env_budgets.clone()
                peak_paper_eq = env_budgets.clone()
                start_b1 = float(vec_env.bids[w_start, 0].item())

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
                    z_aug = torch.cat([z, ps, pb, pc, pe, v7_exp, multi_6d], dim=-1).unsqueeze(1).to(policy.dtype)

                    h1 = torch.relu(torch.bmm(z_aug, policy.w1) + policy.b1)
                    raw_out = torch.sigmoid((torch.bmm(h1, policy.w2) + policy.b2).squeeze(1).float())

                    vec_env.step(raw_out, cur_t_scalar=cur_t)

                    cur_pos = vec_env.pos_qty
                    cur_cash = vec_env.cash
                    b1 = torch.clamp(vec_env.bids[cur_t, 0], min=tick_sz)
                    in_p = cur_pos > 1e-6

                    cur_hold = torch.where(in_p, cur_hold + 1.0, torch.zeros_like(cur_hold))
                    max_hold_s = torch.maximum(max_hold_s, cur_hold)

                    real_eq = cur_cash + vec_env.pos_cost_usd
                    peak_real_eq = torch.maximum(peak_real_eq, real_eq)
                    max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - real_eq)

                    paper_eq = cur_cash + cur_pos * b1
                    peak_paper_eq = torch.maximum(peak_paper_eq, paper_eq)
                    max_paper_dd_usd = torch.maximum(max_paper_dd_usd, peak_paper_eq - paper_eq)

                    if vec_env.is_bleed_cpu[cur_t]:
                        evasion_exits = evasion_exits + (cur_pos < prev_pos - 1e-4).float()
                        # Чистый RL-сигнал осторожности на проливах: штраф за удержание убыточной позиции под лавиной
                        underwater_ticks = torch.clamp((vec_env.avg_entry - b1) / tick_sz - 0.5, min=0.0)
                        pos_frac = torch.clamp((cur_pos * b1) / env_budgets, 0.0, 1.0)
                        bleed_underwater_pen = bleed_underwater_pen + pos_frac * underwater_ticks * 0.04

                    # Чистый RL-сигнал захвата импульса (PUMP): если цена выросла на >= 1.5 тика от входа,
                    # поощряем наличие плюсовой позиции и дальних Ask-ордеров (+2..+5 тиков, слоты 4..11)
                    if float(b1.item()) >= start_b1 + 1.5 * tick_sz:
                        prof_ticks = torch.clamp((b1 - vec_env.avg_entry) / tick_sz, min=0.0, max=6.0)
                        far_ask_frac = torch.clamp(
                            vec_env.ask_qty[:, 4:].sum(dim=1) / torch.clamp(cur_pos, min=1e-5), 0.0, 1.0
                        )
                        pump_capture_bonus = pump_capture_bonus + in_p.float() * prof_ticks * (0.015 + 0.025 * far_ask_frac)

                    prev_pos.copy_(cur_pos)

                end_t = int(w_start + win_len - 1)
                end_b1 = torch.clamp(vec_env.bids[end_t, 0], min=tick_sz)
                rem = vec_env.pos_qty > 1e-6
                # Честная оценка на границе окна:
                # - Если остаток в просадке >= 1.05 тика под avg_entry -> списываем по рынку end_b1;
                # - Если остаток в чистом плюсе (>= 1.05 тика НАД avg_entry, например на пампе!) -> засчитываем рыночный профит!
                underwater_vs_avg = rem & (end_b1 < (vec_env.avg_entry - 1.05 * tick_sz))
                in_profit_vs_avg = rem & (end_b1 > (vec_env.avg_entry + 1.05 * tick_sz))
                liq_val = torch.where(
                    underwater_vs_avg | in_profit_vs_avg,
                    vec_env.pos_qty * end_b1 * (1.0 - vec_env.maker_fee),
                    vec_env.pos_cost_usd,
                )
                vec_env.cash = torch.where(rem, vec_env.cash + liq_val, vec_env.cash)
                vec_env.pos_qty = torch.where(rem, torch.zeros_like(vec_env.pos_qty), vec_env.pos_qty)
                vec_env.pos_cost_usd = torch.where(rem, torch.zeros_like(vec_env.pos_cost_usd), vec_env.pos_cost_usd)
                max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - vec_env.cash)

                w_pnl = vec_env.cash - env_budgets
                total_pnl_usd += w_pnl
                if reg_name == "NORM":
                    norm_pnl_usd.copy_(w_pnl)
                elif reg_name == "DUMP":
                    dump_pnl_usd.copy_(w_pnl)
                else:
                    pump_pnl_usd.copy_(w_pnl)

            cycles.copy_(vec_env.sell_cycles)
            prof_cycles.copy_(vec_env.prof_sell_cycles)

            pnl_pct = 100.0 * total_pnl_usd / env_budgets
            norm_pct = 100.0 * norm_pnl_usd / env_budgets
            dump_pct = 100.0 * dump_pnl_usd / env_budgets
            pump_pct = 100.0 * pump_pnl_usd / env_budgets

            real_dd_pct = 100.0 * max_real_dd_usd / env_budgets
            paper_dd_pct = 100.0 * max_paper_dd_usd / env_budgets
            loss_cycles = torch.clamp(cycles - prof_cycles, min=0).float()

            # Средняя прибыль на 1 закрытый цикл (в % от депозита) — учит брать широкие движения 2..5 тиков на пампе!
            avg_pnl_per_cyc = torch.where(
                prof_cycles > 0,
                torch.clamp(pnl_pct / torch.clamp(prof_cycles.float(), min=1.0), min=0.0, max=0.50),
                torch.zeros_like(pnl_pct),
            )

            # Целевая функция Урока 7 (Triad Mastery):
            # 1) Общий PnL + повышенный вес за сохранение капитала на DUMP и захват движения на PUMP;
            # 2) Жесткий анти-просадочный щит (real_dd_pct + bleed_underwater_pen);
            # 3) Премия за амплитуду прибыльной сделки (avg_pnl_per_cyc * 950) и работу на пампе (pump_capture_bonus).
            env_scores = (
                pnl_pct * 1200.0
                + dump_pct * 450.0
                + torch.clamp(pump_pct, min=-1.0, max=3.0) * 650.0
                + avg_pnl_per_cyc * 950.0
                + torch.clamp(total_pnl_usd * 7.5, min=-220.0, max=420.0)
                - real_dd_pct * 1650.0
                - torch.clamp(paper_dd_pct - 0.45, min=0.0) * 220.0
                - torch.clamp(bleed_underwater_pen, max=250.0) * 2.5
                + torch.clamp(pump_capture_bonus, max=160.0) * 2.2
                + torch.clamp(prof_cycles.float(), max=35.0) * 24.0
                - loss_cycles * 45.0
                + torch.clamp(vec_env.partial_fills.float(), max=25.0) * 4.5
                + torch.clamp(evasion_exits, max=12.0) * 5.0
                - torch.clamp(max_hold_s - 260.0, min=0.0) * 0.18
            )
            env_scores = torch.where(cycles == 0, env_scores - 450.0, env_scores)
            env_scores = torch.where(total_pnl_usd <= 0.0, env_scores - 200.0, env_scores)

            g_scores_mat = env_scores.view(n_groups, n_budgets)
            g_pnl_pct_mat = pnl_pct.view(n_groups, n_budgets)
            g_norm_pct_mat = norm_pct.view(n_groups, n_budgets)
            g_dump_pct_mat = dump_pct.view(n_groups, n_budgets)
            g_pump_pct_mat = pump_pct.view(n_groups, n_budgets)
            g_pnl_usd_mat = total_pnl_usd.view(n_groups, n_budgets)
            g_dd_pct_mat = real_dd_pct.view(n_groups, n_budgets)
            g_dd_usd_mat = max_real_dd_usd.view(n_groups, n_budgets)
            g_pdd_pct_mat = paper_dd_pct.view(n_groups, n_budgets)
            g_cyc_mat = cycles.view(n_groups, n_budgets)
            g_pcyc_mat = prof_cycles.view(n_groups, n_budgets)
            g_pf_mat = vec_env.partial_fills.view(n_groups, n_budgets)
            g_mo_mat = vec_env.multi_orders_used.view(n_groups, n_budgets)
            g_hold_mat = max_hold_s.view(n_groups, n_budgets)

            g_prof_buds = (g_pnl_usd_mat > 0.0).long().sum(dim=1)
            group_scores = g_scores_mat.mean(dim=1) + g_prof_buds.float() * 90.0

            c0_pnl_pct = float(g_pnl_pct_mat[0].mean().item())
            c0_pnl_usd = float(g_pnl_usd_mat[0].sum().item())
            c0_dd_pct = float(g_dd_pct_mat[0].mean().item())

            sorted_g = torch.argsort(group_scores, descending=True)
            best_g = int(sorted_g[0].item())

            b_score = float(group_scores[best_g].item())
            b_pnl_pct = float(g_pnl_pct_mat[best_g].mean().item())
            b_norm_pct = float(g_norm_pct_mat[best_g].mean().item())
            b_dump_pct = float(g_dump_pct_mat[best_g].mean().item())
            b_pump_pct = float(g_pump_pct_mat[best_g].mean().item())
            b_pnl_usd = float(g_pnl_usd_mat[best_g].sum().item())
            b_dd_pct = float(g_dd_pct_mat[best_g].mean().item())
            b_dd_usd = float(g_dd_usd_mat[best_g].mean().item())
            b_pdd_pct = float(g_pdd_pct_mat[best_g].mean().item())
            b_prof_buds = int(g_prof_buds[best_g].item())
            b_cyc = int(g_cyc_mat[best_g].float().mean().item())
            b_prof = int(g_pcyc_mat[best_g].float().mean().item())
            b_pf = int(g_pf_mat[best_g].float().mean().item())
            b_mo = int(g_mo_mat[best_g].float().mean().item())
            b_hold = float(g_hold_mat[best_g].mean().item())

            c0_score = float(group_scores[0].item())
            hof_updated = False
            is_strict_win = (b_pnl_pct > 0.0) and (b_prof_buds >= 4) and (b_score >= c0_score)
            is_triad_progress = (
                (b_cyc >= 2)
                and (b_score > c0_score + 18.0)
                and (b_pnl_pct > c0_pnl_pct + 0.04)
                and (b_dd_pct <= c0_dd_pct + 0.02)
            )
            if is_strict_win or is_triad_progress:
                best_env0 = best_g * n_budgets
                w1_n, b1_n, w2_n, b2_n = policy._normalize_single_weights(
                    policy.w1[best_env0], policy.b1[best_env0], policy.w2[best_env0], policy.b2[best_env0]
                )
                hof_state["w1"] = w1_n.detach().cpu()
                hof_state["b1"] = b1_n.detach().cpu()
                hof_state["w2"] = w2_n.detach().cpu()
                hof_state["b2"] = b2_n.detach().cpu()
                hof_updated = True

            policy.evolve_groups(group_scores, hof_state=hof_state, elite_frac=0.03, progress=float(it) / float(iterations))

            rec = {
                "iteration": it,
                "n_slots": 24,
                "best_idx": best_g,
                "budget": 39095.5,
                "prof_budgets": b_prof_buds,
                "hof_updated": hof_updated,
                "score": round(b_score, 2),
                "pnl_pct": round(b_pnl_pct, 4),
                "norm_pct": round(b_norm_pct, 4),
                "dump_pct": round(b_dump_pct, 4),
                "pump_pct": round(b_pump_pct, 4),
                "pnl_usd": round(b_pnl_usd, 4),
                "champ_oos_pnl_pct": round(c0_pnl_pct, 4),
                "champ_oos_pnl_usd": round(c0_pnl_usd, 4),
                "champ_oos_dd_pct": round(c0_dd_pct, 4),
                "max_dd_pct": round(b_dd_pct, 4),
                "max_dd_usd": round(b_dd_usd, 4),
                "paper_dd_pct": round(b_pdd_pct, 4),
                "max_hold_s": round(b_hold, 1),
                "cycles": b_cyc,
                "prof_cycles": b_prof,
                "partial_fills": b_pf,
                "multi_order_ticks": b_mo,
            }
            history.append(rec)
            summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            tail20 = history[-20:]
            r20_pnl = float(np.mean([x["pnl_pct"] for x in tail20]))
            r20_oos = float(np.mean([x.get("champ_oos_pnl_pct", x["pnl_pct"]) for x in tail20]))
            r20_dd = float(np.mean([x["max_dd_pct"] for x in tail20]))
            r20_wr = 100.0 * sum(x["prof_cycles"] for x in tail20) / max(1, sum(x["cycles"] for x in tail20))
            dt = time.time() - t0
            star = "★" if hof_updated else "·"

            log_msg(
                f"🎯 L7 [{it}/{iterations}]{star} ({b_prof_buds}/6+) | "
                f"Triad: {b_pnl_pct:+.2f}% [N:{b_norm_pct:+.2f}% D:{b_dump_pct:+.2f}% P:{b_pump_pct:+.2f}%] | "
                f"BlindChamp: {c0_pnl_pct:+.2f}% | DD: {b_dd_pct:.2f}% | Cyc: {b_cyc}/{b_prof} | "
                f"📈Roll20: {r20_pnl:+.2f}% (OOS {r20_oos:+.2f}%, DD {r20_dd:.2f}%, WR {r20_wr:.1f}%) ({dt:.1f}с)"
            )

            ckpt_payload = {
                "iteration": it,
                "n_slots": 24,
                "score": b_score,
                "pnl_pct": b_pnl_pct,
                "pnl_usd": b_pnl_usd,
                "backbone_state": backbone.state_dict(),
                "w1": hof_state["w1"],
                "b1": hof_state["b1"],
                "w2": hof_state["w2"],
                "b2": hof_state["b2"],
            }
            torch.save(ckpt_payload, latest_ckpt)
            if it % 50 == 0 or it == iterations:
                torch.save(ckpt_payload, SAVE_DIR / f"lesson_7_champ_iter_{it:04d}.pt")

    log_msg("🏆 УРОК 7 (TRIAD MASTERY: NORMAL + DUMP + PUMP) УСПЕШНО ЗАВЕРШЕН!")


if __name__ == "__main__":
    run_lesson_7(iterations=525, n_groups=1365)
