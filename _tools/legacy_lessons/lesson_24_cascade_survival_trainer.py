#!/usr/bin/env python3
"""
Урок 24 (Cascade Survival & No-Scratch Holding, 444D):
1. Физическая честность 1-в-1 с live_rl_trader.py:
   - allow_scratch_stop_asks = False (полный запрет ASK #00/#01 гарантированного убытка avg-1t / avg).
   - panic_thresh = 0.85 (единый с продовым порогом PANIC_TAKER_EXIT).
   - selective_front_cancel = True, true_latency = True.
2. Длинный 40-минутный горизонт эпизодов (win_len = 2400s):
   Агент проживает полные многоволновые каскады (-5%..-18%) и V-развороты,
   обучаясь дозированно расходовать кэш и не паниковать перед отскоком.
3. Строгое 3-стороннее разделение без утечки:
   - TRAIN <= 2026-11-11 22:00 MSK (- 3000s эмбарго)
   - VAL_CASCADE [2026-11-11 22:00 -> 2026-11-12 20:30 MSK] (включая краш 12.11 $1.40->$1.12->$1.32)
   - BLIND_OVERNIGHT >= 2026-11-12 20:30 MSK (полностью неприкасаемый финальный тест)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path("/home/restorator/crypto_bot")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_free_multi_order_env import VectorizedFreeMultiOrderEnv
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors
from _tools.lesson_18_multistack_lifo_survival_trainer import FusedBF16BackboneRunner
from _tools.fused_l22_train_tick import (
    MKT_IDX_LIST,
    AG_IDX_LIST,
    precompute_window_market_z334,
    fused_l22_decoupled_tick,
    evaluate_l22_dual_basket,
)
from _tools.lesson_23_pump_capture_penny_l1_trainer import (
    N_BLOCKS_L21,
    CompactDecoupledParetoPolicy,
    _batch_normalize_weights_l21,
    _build_roll_sell_30s,
    compute_pareto_group_fitness,
)

CACHE_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
RELEASE_CHAMP_PATH = BASE_DIR / "models/prl/release_champion_v1.pt"
SAVE_DIR = BASE_DIR / "models/prl/lesson_24_cascade_survival"
LATEST_CHAMP_PATH = SAVE_DIR / "lesson_24_champ_latest.pt"
RESUME_CHAMP_PATH = SAVE_DIR / "lesson_24_champ_resume.pt"
LOG_PATH = BASE_DIR / "data/logs/lesson_24_train.log"

MSK_TZ = timezone(timedelta(hours=3))
VAL_START_TS = datetime(2026, 11, 11, 22, 0, tzinfo=MSK_TZ).timestamp()
BLIND_START_TS = datetime(2026, 11, 12, 20, 30, tzinfo=MSK_TZ).timestamp()


def log_msg(msg: str) -> None:
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts_str}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _classify_2400s_windows(
    starts_arr: np.ndarray,
    b1_np: np.ndarray,
    bq_np: np.ndarray,
    aq_np: np.ndarray,
    bv_np: np.ndarray,
    sv_np: np.ndarray,
    win_len: int,
    tick_sz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Разделяет длинные окна win_len на NORM, DUMP, CASCADE и PUMP с проверкой ликвидности."""
    norm_s, dump_s, casc_s, pump_s = [], [], [], []
    for w0 in starts_arr:
        w0 = int(w0)
        seg = b1_np[w0 : w0 + win_len]
        s_sv = float(sv_np[w0 : w0 + win_len].sum())
        s_bv = float(bv_np[w0 : w0 + win_len].sum())
        q_b25 = float(np.percentile(bq_np[w0 : w0 + win_len], 25))
        q_a25 = float(np.percentile(aq_np[w0 : w0 + win_len], 25))
        if s_sv < max(250.0, q_b25 * 2.0) or s_bv < max(250.0, q_a25 * 1.5):
            continue
        sv_early = float(sv_np[w0 : w0 + min(600, win_len // 2)].sum())
        bv_later = float(bv_np[w0 + min(300, win_len // 4) : w0 + win_len].sum())
        entry_fillable = (sv_early >= max(float(bq_np[w0]), q_b25) * 1.35) and (bv_later >= q_a25 * 1.35)
        ret_pct = 100.0 * (seg[-1] - seg[0]) / max(seg[0], tick_sz)
        dd_seg_pct = 100.0 * (np.max(seg) - np.min(seg)) / max(seg[0], tick_sz)
        end_ticks = (seg[-1] - seg[0]) / tick_sz
        if ret_pct <= -3.0 or (dd_seg_pct >= 6.0 and ret_pct < -0.5):
            casc_s.append(w0)
        elif ret_pct <= -0.8 or end_ticks <= -1.5:
            dump_s.append(w0)
        elif (ret_pct >= 1.5 or end_ticks >= 2.0) and entry_fillable:
            pump_s.append(w0)
        elif dd_seg_pct >= 0.7 and entry_fillable:
            norm_s.append(w0)

    def _to_arr(lst):
        return np.array(lst if lst else starts_arr[: min(32, len(starts_arr))], dtype=np.int64)

    return _to_arr(norm_s), _to_arr(dump_s), _to_arr(casc_s), _to_arr(pump_s)


def _save_l24_checkpoint(
    path: Path,
    base_ck: dict,
    folded_state: dict,
    gate_logit: torch.Tensor,
    it: int,
    release_name: str,
    metrics: dict,
) -> None:
    ck_out = dict(base_ck)
    ck_out.update({
        "lesson": 24,
        "phase": 1,
        "iteration": int(it),
        "n_slots": 24,
        "use_cross_venue_v1": True,
        "decoupled_market_backbone": True,
        "selective_front_cancel": True,
        "allow_scratch_stop_asks": False,
        "panic_thresh": 0.85,
        "release_name": release_name,
        "gate_logit": gate_logit.clone(),
        "pareto_util_bp": float(metrics["pareto_util_bp"]),
        "clean_util_bp": float(metrics["clean_util_bp"]),
        "uw_util_bp": float(metrics["uw_util_bp"]),
        "clean_ev": float(metrics["clean_ev"]),
        "cascade_pct": float(metrics.get("cascade_pct", metrics.get("clean_dump_pct", 0.0))),
        "w1": folded_state["w1"].cpu().clone(),
        "b1": folded_state["b1"].cpu().clone(),
        "w_res1": folded_state["w_res1"].cpu().clone(),
        "b_res1": folded_state["b_res1"].cpu().clone(),
        "w_res2": folded_state["w_res2"].cpu().clone(),
        "w2": folded_state["w2"].cpu().clone(),
        "b2": folded_state["b2"].cpu().clone(),
        "w_rec": folded_state["w_rec"].cpu().clone(),
        "b_rec": folded_state["b_rec"].cpu().clone(),
    })
    torch.save(ck_out, path)


def run_lesson_24(
    iterations: int = 800,
    n_groups: int = 1024,
    win_len: int = 2400,
    rng_seed: int = 2026241,
    allow_scratch_stop_asks: bool = False,
    panic_thresh: float = 0.85,
) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_budgets = 6
    pop_size = n_groups * n_budgets

    log_msg("=" * 116)
    log_msg(
        f"🚀 УРОК 24 (CASCADE SURVIVAL 444D | NO-SCRATCH={not allow_scratch_stop_asks} | PANIC={panic_thresh:.2f} | "
        f"WIN={win_len}s): {iterations} ИТЕР | 10 ОКОН x {pop_size} = {10 * pop_size:,} СРЕД"
    )
    log_msg("=" * 116)

    raw = np.load(CACHE_PATH, allow_pickle=True)
    cv_ver = int(raw["has_cross_venue_v1"]) if "has_cross_venue_v1" in raw.files else 0
    assert cv_ver >= 1, "Сначала обновите кросс-биржевой кэш `_tools/cross_venue_first_order_net.py`!"
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.01

    # Берём ~1,200,000 последних тиков (прим. 14 дней), где точно присутствуют старые каскады, краш 12.11 и слепая ночь
    slice_len = min(1200000, n_total)
    s0 = n_total - slice_len
    ts_sub = np.array(raw["ts"][s0:], dtype=np.float64, copy=True)

    sub = {
        k: np.array(raw[k][s0:], dtype=np.float32, copy=True)
        for k in ["bid_p", "ask_p", "bid_q", "ask_q", "buy_vol", "sell_vol", "y_sweep_12s", "y_bleed_60s", "y_crash_300s", "y_exit_30s"]
    }
    sub["tick_size"] = tick_sz
    v7_raw_t = torch.from_numpy(np.array(raw["v7_signals_58d"][s0:], dtype=np.float32, copy=True)).to(device)
    raw.close()
    del raw
    gc.collect()

    vel_cols = [0, 2, 4, 6, 8, 12, 34, 42]
    v7_vel8 = torch.zeros((len(v7_raw_t), 8), dtype=torch.float32, device=device)
    v7_vel8[5:] = torch.clamp(v7_raw_t[5:, vel_cols] - v7_raw_t[:-5, vel_cols], -3.0, 3.0)

    bp_t = torch.from_numpy(sub["bid_p"][:, :5]).to(device)
    ap_t = torch.from_numpy(sub["ask_p"][:, :5]).to(device)
    bq_t = torch.from_numpy(sub["bid_q"][:, :5]).to(device)
    aq_t = torch.from_numpy(sub["ask_q"][:, :5]).to(device)
    bv_t = torch.from_numpy(sub["buy_vol"]).to(device)
    sv_t = torch.from_numpy(sub["sell_vol"]).to(device)

    lob_vel8 = torch.zeros((len(v7_raw_t), 8), dtype=torch.float32, device=device)
    lob_vel8[5:, 0] = torch.clamp((bp_t[5:, 0] - bp_t[:-5, 0]) / (3.0 * tick_sz), -2.0, 2.0)
    lob_vel8[5:, 1] = torch.clamp((ap_t[5:, 0] - ap_t[:-5, 0]) / (3.0 * tick_sz), -2.0, 2.0)
    lob_vel8[5:, 2] = torch.clamp(torch.log1p(torch.clamp(bq_t[5:, 0], min=0.0)) - torch.log1p(torch.clamp(bq_t[:-5, 0], min=0.0)), -2.0, 2.0)
    lob_vel8[5:, 3] = torch.clamp(torch.log1p(torch.clamp(aq_t[5:, 0], min=0.0)) - torch.log1p(torch.clamp(aq_t[:-5, 0], min=0.0)), -2.0, 2.0)
    lob_vel8[:, 4] = torch.clamp((bq_t[:, 0] - aq_t[:, 0]) / (bq_t[:, 0] + aq_t[:, 0] + 10.0), -1.0, 1.0)
    b_deep = bq_t[:, 1:5].sum(dim=1)
    a_deep = aq_t[:, 1:5].sum(dim=1)
    lob_vel8[:, 5] = torch.clamp((b_deep - a_deep) / (b_deep + a_deep + 50.0), -1.0, 1.0)
    lob_vel8[:, 6] = torch.clamp(torch.log1p(torch.clamp(bv_t, min=0.0)) / 8.0, 0.0, 2.0)
    lob_vel8[:, 7] = torch.clamp(torch.log1p(torch.clamp(sv_t, min=0.0)) / 8.0, 0.0, 2.0)
    del bp_t, ap_t, bq_t, aq_t, bv_t, sv_t

    v7_all = torch.cat([v7_raw_t, v7_vel8, lob_vel8], dim=-1)
    del v7_raw_t, v7_vel8, lob_vel8

    lob_t, flow_m = build_lob_10_and_flow_tensors(sub)
    targets_np = np.stack([sub["y_sweep_12s"], sub["y_bleed_60s"], sub["y_crash_300s"], sub["y_exit_30s"]], axis=1)

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    policy = CompactDecoupledParetoPolicy(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, device=device)
    v7_all = v7_all.to(dtype=policy.dtype)

    summary_path = SAVE_DIR / "lesson_24_summary.json"
    history = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    seed_path = LATEST_CHAMP_PATH if LATEST_CHAMP_PATH.exists() else RELEASE_CHAMP_PATH
    ck = torch.load(seed_path, map_location=device, weights_only=False)
    policy.load_champion(ck)

    if RESUME_CHAMP_PATH.exists():
        ck_res = torch.load(RESUME_CHAMP_PATH, map_location="cpu", weights_only=False)
        start_iter = (int(ck_res.get("iteration", 0)) + 1) if int(ck_res.get("lesson", 23)) == 24 else 1
    else:
        start_iter = (int(ck.get("iteration", 0)) + 1) if int(ck.get("lesson", 23)) == 24 else 1
    if start_iter > 1 and history:
        history = [r for r in history if int(r.get("iteration", 0)) < start_iter]

    hof_state = {
        k: getattr(policy, k)[0].detach().cpu().clone()
        for k in ("w1", "b1", "w_res1", "b_res1", "w_res2", "w2", "b2", "w_rec", "b_rec")
    }
    hof_g_logit = torch.zeros(N_BLOCKS_L21, dtype=torch.float32)

    bf16_runner = FusedBF16BackboneRunner(backbone, policy.dtype, device)
    lob_fu_chunks = []
    with torch.no_grad():
        for i0 in range(0, len(lob_t), 100000):
            chunk_gpu = torch.from_numpy(lob_t[i0 : i0 + 100000]).float().to(device)
            lr_ch = backbone.encode_lob_batch(chunk_gpu).to(dtype=policy.dtype)
            lob_fu_chunks.append((lr_ch @ bf16_runner.w_fu_lob + bf16_runner.b_fu).contiguous())
            del chunk_gpu, lr_ch
    del lob_t
    lob_fu_all = torch.cat(lob_fu_chunks, dim=0)
    del lob_fu_chunks

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
        num_envs=18,
        n_side_slots=12,
        tick_size=tick_sz,
        device=device,
        allow_scratch_stop_asks=allow_scratch_stop_asks,
    )
    roll_sell_30s_all = _build_roll_sell_30s(vec_env.sell_vol)
    del sub, flow_m, targets_np
    gc.collect()
    torch.cuda.empty_cache()

    b1_all_np = vec_env.bids[:, 0].detach().cpu().numpy()
    bq_all_np = vec_env.bids_q[:, 0].detach().cpu().numpy()
    aq_all_np = vec_env.asks_q[:, 0].detach().cpu().numpy()
    bv_all_np = vec_env.buy_vol.detach().cpu().numpy()
    sv_all_np = vec_env.sell_vol.detach().cpu().numpy()

    val_s0 = int(np.searchsorted(ts_sub, VAL_START_TS, side="left"))
    blind_s0 = int(np.searchsorted(ts_sub, BLIND_START_TS, side="left"))
    embargo_gap = win_len + 600
    train_hi = max(win_len + 1000, val_s0 - embargo_gap)

    train_grid = np.arange(120, train_hi - win_len - 100, 120)
    val_grid = np.arange(val_s0, blind_s0 - win_len - 100, 300)

    train_norm, train_dump, train_casc, train_pump = _classify_2400s_windows(
        train_grid, b1_all_np, bq_all_np, aq_all_np, bv_all_np, sv_all_np, win_len, tick_sz
    )
    val_norm, val_dump, val_casc, val_pump = _classify_2400s_windows(
        val_grid, b1_all_np, bq_all_np, aq_all_np, bv_all_np, sv_all_np, win_len, tick_sz
    )

    log_msg(
        f"📊 Разметка 40-мин окон (win_len={win_len}s): "
        f"TRAIN[N={len(train_norm)}, D={len(train_dump)}, CASC={len(train_casc)}, P={len(train_pump)}] | "
        f"VAL[N={len(val_norm)}, D={len(val_dump)}, CASC={len(val_casc)}, P={len(val_pump)}] | "
        f"BLIND_OVERNIGHT={slice_len - blind_s0:,}с ЗАКРЫТ ОТ ТРЕЙНА И ВАЛИДАЦИИ"
    )

    # 24-оконная сбалансированная валидационная корзина (20 CLEAN: 6 NORM + 3 DUMP + 5 CASCADE + 6 PUMP; + 4 UW):
    val_basket = []
    for k_i in range(6):
        val_basket.append(("NORM", int(val_norm[(k_i * len(val_norm) // 6) % len(val_norm)])))
        val_basket.append(("PUMP", int(val_pump[(k_i * len(val_pump) // 6) % len(val_pump)])))
    for k_i in range(5):
        val_basket.append(("DUMP", int(val_casc[(k_i * len(val_casc) // 5) % len(val_casc)])))
    for k_i in range(3):
        val_basket.append(("DUMP", int(val_dump[(k_i * len(val_dump) // 3) % len(val_dump)])))
    val_basket.append(("UW_NORM", int(val_norm[len(val_norm) // 4])))
    val_basket.append(("UW_NORM", int(val_norm[(3 * len(val_norm)) // 4])))
    val_basket.append(("UW_DUMP", int(val_casc[len(val_casc) // 4])))
    val_basket.append(("UW_DUMP", int(val_casc[(3 * len(val_casc)) // 4])))

    base_folded = policy.fold_gate_into_state(hof_state, hof_g_logit)
    t_v0 = time.time()
    init_val_res = evaluate_l22_dual_basket(
        vec_env, bf16_runner, lob_fu_all, v7_all, roll_sell_30s_all,
        [base_folded] * 8, val_basket, win_len, tick_sz, device,
        selective_front_cancel=True, true_latency=True,
        allow_scratch_stop_asks=allow_scratch_stop_asks, panic_thresh=panic_thresh,
    )[0]
    dt_v0 = time.time() - t_v0

    best_pareto_util = float(init_val_res["pareto_util_bp"])
    best_clean_util = float(init_val_res["clean_util_bp"])
    best_uw_util = float(init_val_res["uw_util_bp"])
    best_clean_ev = float(init_val_res["clean_ev"])
    best_val_ev = float(init_val_res["realistic_ev"])
    best_val_state = {k: v.clone() for k, v in hof_state.items()}
    best_val_g_logit = hof_g_logit.clone()
    best_val_folded = {k: v.clone() for k, v in base_folded.items()}
    last_val_champ_it = max(0, start_iter - 1)

    log_msg(
        f"⚖️ СТАРТОВАЯ БАЗА В НОВОЙ ФИЗИКЕ (compiled {dt_v0:.1f}с): "
        f"ParetoUtil={best_pareto_util:+.2f}bp [Clean20={best_clean_util:+.2f}bp (EV={best_clean_ev:+.4f}%, "
        f"Fills={init_val_res['clean_fills']:.1f}, Cancels={init_val_res['clean_cancels']:.1f}, "
        f"Cnl/Fill={init_val_res['clean_cnl_per_fill']:.1f}, η_queue={init_val_res['clean_eta_queue_pct']:.1f}%) | "
        f"UW4={best_uw_util:+.2f}bp (ΔMTM={init_val_res['uw_delta_ev']:+.4f}%)] | "
        f"Clean[N:{init_val_res['clean_norm_pct']:+.3f}% D+Casc:{init_val_res['clean_dump_pct']:+.3f}% "
        f"P:{init_val_res['clean_pump_pct']:+.3f}% CVaR25:{init_val_res['clean_cvar25_pct']:+.2f}% DD:{init_val_res['clean_dd_pct']:.2f}%]"
    )

    budget_tiers = torch.tensor([100.0, 300.0, 625.0, 1000.0, 1500.0, 2500.0], device=device)
    env_budgets = budget_tiers.unsqueeze(0).expand(n_groups, n_budgets).reshape(pop_size)
    rng = np.random.default_rng(rng_seed)
    for _ in range(1, start_iter):
        for _k in range(10):
            _ = rng.integers(0, 10)

    W_par = 10  # 2 NORM + 1 DUMP + 1 CASCADE + 2 PUMP (CLEAN) + 2 UW_NORM + 1 UW_DUMP + 1 UW_CASCADE (UW)
    N_par = W_par * pop_size
    S_slots = int(vec_env.n_side_slots)
    budget_16 = env_budgets.unsqueeze(0).expand(W_par, pop_size).reshape(N_par).contiguous()
    nes_w = torch.softmax(-torch.arange(24, dtype=torch.float32, device=device) * 0.25, dim=0)
    mkt_idx_t = torch.tensor(MKT_IDX_LIST, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_LIST, dtype=torch.long, device=device)
    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    chunk_sz = 100

    for it in range(start_iter, iterations + 1):
        t0 = time.time()
        cycle_prog = float((it - 1) % 200) / 199.0
        cos_w = 0.5 * (1.0 + math.cos(math.pi * cycle_prog))
        cosine_decay = 0.25 + 0.75 * cos_w

        w_batch = [
            ("NORM1", int(rng.choice(train_norm))),
            ("NORM2", int(rng.choice(train_norm))),
            ("DUMP1", int(rng.choice(train_dump))),
            ("CASC1", int(rng.choice(train_casc))),
            ("PUMP1", int(rng.choice(train_pump))),
            ("PUMP2", int(rng.choice(train_pump))),
            ("UW_NORM1", int(rng.choice(train_norm))),
            ("UW_NORM2", int(rng.choice(train_norm))),
            ("UW_DUMP1", int(rng.choice(train_dump))),
            ("UW_CASC1", int(rng.choice(train_casc))),
        ]

        with torch.no_grad():
            w1_g = policy.effective_group_w1()
            w1_mkt_g = w1_g.index_select(1, mkt_idx_t).contiguous()
            w1_ag_g = w1_g.index_select(1, ag_idx_t).contiguous()
            del w1_g
            b1_g, wr1_g, br1_g, wr2_g = policy.b1, policy.w_res1, policy.b_res1, policy.w_res2
            w2_g, b2_g, wrec_g, brec_g = policy.w2, policy.b2, policy.w_rec, policy.b_rec

            w_starts = torch.tensor([ws for _, ws in w_batch], dtype=torch.long, device=device)
            t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
            t_steps_cl = torch.clamp(t_steps, max=vec_env.max_steps - 1)

            flow_w_all = vec_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
            lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()
            v7_w_all = v7_all[t_steps_cl[: win_len - 1]].contiguous()
            z_mkt_all = precompute_window_market_z334(flow_w_all, lob_fu_w_all, v7_w_all, bf16_runner)
            del flow_w_all, lob_fu_w_all, v7_w_all

            b10_w_all = vec_env.bids[t_steps_cl].contiguous()
            a10_w_all = vec_env.asks[t_steps_cl].contiguous()
            bq10_w_all = vec_env.bids_q[t_steps_cl].contiguous()
            aq10_w_all = vec_env.asks_q[t_steps_cl].contiguous()
            b1_w_all = b10_w_all[:, :, 0].contiguous()
            a1_w_all = a10_w_all[:, :, 0].contiguous()
            b1q_w_all = bq10_w_all[:, :, 0].contiguous()
            a1q_w_all = aq10_w_all[:, :, 0].contiguous()
            rb_w_all = vec_env.roll_buy_30s[t_steps_cl].contiguous()
            rs_w_all = roll_sell_30s_all[t_steps_cl].contiguous()
            sv_w_all = vec_env.sell_vol[t_steps_cl].contiguous()
            bv_w_all = vec_env.buy_vol[t_steps_cl].contiguous()

            mem_h = torch.zeros((N_par, 16), dtype=policy.dtype, device=device)
            cash = budget_16.clone()
            pos_qty = torch.zeros(N_par, dtype=torch.float32, device=device)
            pos_cost_usd = torch.zeros(N_par, dtype=torch.float32, device=device)
            avg_entry = torch.zeros(N_par, dtype=torch.float32, device=device)
            min_buy_px = torch.zeros(N_par, dtype=torch.float32, device=device)
            low_tranche_qty = torch.zeros(N_par, dtype=torch.float32, device=device)
            mid_buy_px = torch.zeros(N_par, dtype=torch.float32, device=device)
            mid_tranche_qty = torch.zeros(N_par, dtype=torch.float32, device=device)
            entry_s = torch.zeros(N_par, dtype=torch.int16, device=device)

            init_b1_0 = torch.clamp(b1_w_all[0], min=tick_sz).unsqueeze(1).expand(W_par, pop_size).reshape(N_par)
            is_uw_w = torch.zeros(W_par, dtype=torch.bool, device=device)
            is_uw_w[6:] = True
            uw_mask = is_uw_w.unsqueeze(1).expand(W_par, pop_size).reshape(N_par)
            uw_depth = torch.zeros(N_par, dtype=torch.float32, device=device)
            uw_depth[6 * pop_size : 7 * pop_size] = 0.04
            uw_depth[7 * pop_size : 8 * pop_size] = 0.09
            uw_depth[8 * pop_size : 9 * pop_size] = 0.04
            uw_depth[9 * pop_size : 10 * pop_size] = 0.12

            uw_usd = budget_16 * 0.60
            uw_avg = torch.round((init_b1_0 * (1.0 + uw_depth)) / tick_sz) * tick_sz
            uw_qty = uw_usd / torch.clamp(uw_avg, min=tick_sz)
            cash = torch.where(uw_mask, budget_16 - uw_usd, cash)
            pos_qty = torch.where(uw_mask, uw_qty, pos_qty)
            avg_entry = torch.where(uw_mask, uw_avg, avg_entry)
            pos_cost_usd = torch.where(uw_mask, uw_qty * uw_avg * (1.0 + vec_env.maker_fee), pos_cost_usd)
            min_buy_px = torch.where(uw_mask, uw_avg, min_buy_px)
            low_tranche_qty = torch.where(uw_mask, uw_qty * 0.5, low_tranche_qty)
            mid_buy_px = torch.where(uw_mask, uw_avg, mid_buy_px)
            mid_tranche_qty = torch.where(uw_mask, uw_qty * 0.5, mid_tranche_qty)

            start_mtm_eq = cash + pos_qty * init_b1_0 * (1.0 - vec_env.maker_fee)
            bid_px = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            bid_qty = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            bid_qa = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            bid_qa0 = torch.zeros((N_par, S_slots), dtype=policy.dtype, device=device)
            bid_act_s = torch.zeros((N_par, S_slots), dtype=torch.int16, device=device)
            ask_px = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            ask_qty = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            ask_qa = torch.zeros((N_par, S_slots), dtype=torch.float32, device=device)
            ask_qa0 = torch.zeros((N_par, S_slots), dtype=policy.dtype, device=device)
            ask_act_s = torch.zeros((N_par, S_slots), dtype=torch.int16, device=device)

            peak_mtm_eq = start_mtm_eq.clone()
            sum_sq_dd_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            max_dd_pct_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            sum_mut_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            n_fills_16 = torch.zeros(N_par, dtype=torch.int16, device=device)
            n_cancels_16 = torch.zeros(N_par, dtype=torch.int16, device=device)

            for c0_t in range(0, win_len - 1, chunk_sz):
                c1_t = min(win_len - 1, c0_t + chunk_sz)
                m_len = c1_t - c0_t
                z_ch = z_mkt_all[c0_t:c1_t].reshape(m_len * W_par, 334).unsqueeze(0).expand(n_groups, -1, -1)
                pre_mkt_ch = (torch.bmm(z_ch, w1_mkt_g) + b1_g).view(n_groups, m_len, W_par, 384).permute(1, 0, 2, 3).contiguous()

                for j_t in range(m_len):
                    t_step = c0_t + j_t
                    t_nxt = t_step + 1
                    (
                        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                        peak_mtm_eq, sum_sq_dd_16, max_dd_pct_16, sum_mut_16, n_fills_16, n_cancels_16,
                    ) = fused_l22_decoupled_tick(
                        s_steps_i16[t_step], s_steps_i16[t_nxt], pre_mkt_ch[j_t],
                        b1_w_all[t_step], a1_w_all[t_step], b1q_w_all[t_step], a1q_w_all[t_step],
                        rb_w_all[t_step], rs_w_all[t_step],
                        b10_w_all[t_step], a10_w_all[t_step], bq10_w_all[t_step], aq10_w_all[t_step],
                        b1_w_all[t_nxt], a1_w_all[t_nxt], sv_w_all[t_nxt], bv_w_all[t_nxt],
                        w1_ag_g, wr1_g, br1_g, wr2_g, w2_g, b2_g, wrec_g, brec_g,
                        vec_env.slot_tick_offsets, vec_env.slot_lob_lvl, vec_env.is_penny_slot, budget_16,
                        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                        peak_mtm_eq, sum_sq_dd_16, max_dd_pct_16, sum_mut_16, n_fills_16, n_cancels_16,
                        tick_sz, vec_env.maker_fee, vec_env.taker_fee,
                        True, True, allow_scratch_stop_asks, panic_thresh,
                    )
                del z_ch, pre_mkt_ch

            init_b1_w = torch.clamp(b1_w_all[0], min=tick_sz)
            end_b1_w = torch.clamp(b1_w_all[win_len - 1], min=tick_sz)
            end_b1_16 = end_b1_w.unsqueeze(1).expand(W_par, pop_size).reshape(N_par)
            rem = pos_qty > 1e-6
            end_cash = torch.where(rem, cash + pos_qty * end_b1_16 * (1.0 - vec_env.maker_fee), cash)
            end_dd_pct = 100.0 * torch.clamp(peak_mtm_eq - end_cash, min=0.0) / budget_16
            max_dd_pct_16 = torch.maximum(max_dd_pct_16, end_dd_pct)

            del (
                w1_mkt_g, w1_ag_g, z_mkt_all, mem_h,
                bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                b10_w_all, a10_w_all, bq10_w_all, aq10_w_all,
            )

            w_mtm_pct_mat = (100.0 * (end_cash - start_mtm_eq) / budget_16).view(W_par, pop_size)
            w_sq_dd_mat = sum_sq_dd_16.view(W_par, pop_size)
            w_dd_mat = max_dd_pct_16.view(W_par, pop_size)
            w_api_mat = sum_mut_16.view(W_par, pop_size)
            w_fills_mat = n_fills_16.float().view(W_par, pop_size)
            w_cancels_mat = n_cancels_16.float().view(W_par, pop_size)

            flat_steps = 6.0 * float(win_len - 1)
            uw_steps = 4.0 * float(win_len - 1)

            flat_norm_pct = w_mtm_pct_mat[0:2].mean(dim=0)
            flat_dump_pct = w_mtm_pct_mat[2]
            flat_casc_pct = w_mtm_pct_mat[3]
            flat_pump_pct = w_mtm_pct_mat[4:6].mean(dim=0)
            mkt_pump_gain_pct = 100.0 * torch.clamp((end_b1_w[4:6] - init_b1_w[4:6]) / init_b1_w[4:6], min=0.0).unsqueeze(1)
            pump_shortfall_pct = torch.clamp(0.28 * mkt_pump_gain_pct - w_mtm_pct_mat[4:6], min=0.0).mean(dim=0)

            flat_ret_pct = 0.45 * flat_norm_pct + 0.25 * flat_pump_pct + 0.15 * flat_dump_pct + 0.15 * flat_casc_pct
            flat_cvar_pct = torch.topk(w_mtm_pct_mat[0:6], 2, dim=0, largest=False).values.mean(dim=0)
            flat_ulcer_pct = torch.sqrt(w_sq_dd_mat[0:6].sum(dim=0) / flat_steps + 1e-8)
            flat_dd_pct = w_dd_mat[0:6].max(dim=0).values
            flat_api = w_api_mat[0:6].sum(dim=0) / flat_steps
            flat_fills = w_fills_mat[0:6].sum(dim=0)
            flat_cancels = w_cancels_mat[0:6].sum(dim=0)
            norm_fills = w_fills_mat[0:2].sum(dim=0)
            pump_fills = w_fills_mat[4:6].sum(dim=0)

            flat_util_bp = 100.0 * (
                1.8 * flat_ret_pct
                + 0.25 * flat_cvar_pct
                - 0.45 * pump_shortfall_pct
                - flat_ulcer_pct
                - 0.06 * (flat_dd_pct * flat_dd_pct)
                - 0.25 * flat_api
            )

            uw_ret_pct = 0.45 * w_mtm_pct_mat[6:8].mean(dim=0) + 0.25 * w_mtm_pct_mat[8] + 0.30 * w_mtm_pct_mat[9]
            uw_worst_pct = w_mtm_pct_mat[6:10].min(dim=0).values
            uw_ulcer_pct = torch.sqrt(w_sq_dd_mat[6:10].sum(dim=0) / uw_steps + 1e-8)
            uw_dd_pct = w_dd_mat[6:10].max(dim=0).values
            uw_api = w_api_mat[6:10].sum(dim=0) / uw_steps

            uw_util_bp = 100.0 * (
                1.4 * uw_ret_pct
                + 0.25 * uw_worst_pct
                - 0.80 * uw_ulcer_pct
                - 0.06 * (uw_dd_pct * uw_dd_pct)
                - 0.25 * uw_api
            )

            g_flat_u_mat = flat_util_bp.view(n_groups, n_budgets)
            g_uw_u_mat = uw_util_bp.view(n_groups, n_budgets)
            g_flat_ret_mat = flat_ret_pct.view(n_groups, n_budgets)
            g_uw_ret_mat = uw_ret_pct.view(n_groups, n_budgets)
            g_norm_mat = flat_norm_pct.view(n_groups, n_budgets)
            g_dump_mat = flat_dump_pct.view(n_groups, n_budgets)
            g_casc_mat = flat_casc_pct.view(n_groups, n_budgets)
            g_pump_mat = flat_pump_pct.view(n_groups, n_budgets)
            g_dd_mat = flat_dd_pct.view(n_groups, n_budgets)
            g_ulcer_mat = flat_ulcer_pct.view(n_groups, n_budgets)
            g_api_mat = flat_api.view(n_groups, n_budgets)
            g_fills_mat = flat_fills.view(n_groups, n_budgets)
            g_cancels_mat = flat_cancels.view(n_groups, n_budgets)

            flat_score_g = g_flat_u_mat.mean(dim=1) - 0.20 * g_flat_u_mat.std(dim=1)
            uw_score_g = g_uw_u_mat.mean(dim=1) - 0.20 * g_uw_u_mat.std(dim=1)
            norm_fills_g = norm_fills.view(n_groups, n_budgets).mean(dim=1)
            pump_fills_g = pump_fills.view(n_groups, n_budgets).mean(dim=1)

            # Жёсткий фильтр активности на длинных 40-мин окнах (закрывает пассивный простой):
            act_2400_ok = ((norm_fills_g + pump_fills_g) >= 8.0) & (pump_fills_g >= 3.0)
            if int(act_2400_ok.long().sum().item()) < 64:
                act_2400_ok = (norm_fills_g + pump_fills_g) >= 3.0
            eff_flat_in = torch.where(act_2400_ok, flat_score_g, flat_score_g - 5000.0)
            eff_uw_in = torch.where(act_2400_ok, uw_score_g, uw_score_g - 5000.0)
            pareto_fitness = compute_pareto_group_fitness(eff_flat_in, eff_uw_in, norm_fills_g, pump_fills_g)
            g_prof_buds = (g_flat_ret_mat > 0.0).long().sum(dim=1)

            c0_fit = float(pareto_fitness[0].item())
            sorted_g = torch.argsort(pareto_fitness, descending=True)
            best_g = int(sorted_g[0].item())

            b_fit = float(pareto_fitness[best_g].item())
            b_flat_ret = float(g_flat_ret_mat[best_g].mean().item())
            b_uw_ret = float(g_uw_ret_mat[best_g].mean().item())
            b_norm_pct = float(g_norm_mat[best_g].mean().item())
            b_dump_pct = float(g_dump_mat[best_g].mean().item())
            b_casc_pct = float(g_casc_mat[best_g].mean().item())
            b_pump_pct = float(g_pump_mat[best_g].mean().item())
            b_dd_pct = float(g_dd_mat[best_g].mean().item())
            b_ulcer = float(g_ulcer_mat[best_g].mean().item())
            b_api = float(g_api_mat[best_g].mean().item())
            b_fills = float(g_fills_mat[best_g].mean().item())
            b_cancels = float(g_cancels_mat[best_g].mean().item())
            b_cnl_per_fill = b_cancels / max(b_fills, 0.5)
            b_eta_queue = 100.0 / (1.0 + max(0.0, b_cnl_per_fill - 1.5) / 12.0) if b_fills >= 0.5 else 0.0
            b_prof_buds = int(g_prof_buds[best_g].item())

            top_g_idx = sorted_g[:24]
            cand_w1 = (policy.w1[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_b1 = (policy.b1[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_wr1 = (policy.w_res1[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_br1 = (policy.b_res1[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_wr2 = (policy.w_res2[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_w2 = (policy.w2[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_b2 = (policy.b2[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_wrec = (policy.w_rec[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_brec = (policy.b_rec[top_g_idx].float() * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_gl = (policy.g_logits[top_g_idx].float() * nes_w.view(-1, 1)).sum(dim=0).clamp(-2.5, 2.0)

            cand_w1, cand_b1, cand_wr1, cand_br1, cand_wr2, cand_w2, cand_b2, cand_wrec, cand_brec = _batch_normalize_weights_l21(
                cand_w1, cand_b1, cand_wr1, cand_br1, cand_wr2, cand_w2, cand_b2, cand_wrec, cand_brec, policy.dtype
            )

            polyak_alpha = 0.22 * cosine_decay
            polyak_fast = min(0.45, 0.35 * cosine_decay)
            polyak_new = min(0.65, 0.55 * cosine_decay)
            hof_w1_new = (1.0 - polyak_alpha) * hof_state["w1"].float() + polyak_alpha * cand_w1.detach().cpu().float()
            for idx_s, idx_e in ((278, 288), (312, 318), (324, 380)):
                hof_w1_new[idx_s:idx_e, :] = (1.0 - polyak_fast) * hof_state["w1"][idx_s:idx_e, :].float() + polyak_fast * cand_w1[idx_s:idx_e, :].detach().cpu().float()
            hof_w1_new[388:444, :] = (1.0 - polyak_new) * hof_state["w1"][388:444, :].float() + polyak_new * cand_w1[388:444, :].detach().cpu().float()

            cand_polyak_state = {
                "w1": hof_w1_new.to(hof_state["w1"].dtype),
                "b1": ((1.0 - polyak_alpha) * hof_state["b1"].float() + polyak_alpha * cand_b1.detach().cpu().float()).to(hof_state["b1"].dtype),
                "w_res1": ((1.0 - polyak_alpha) * hof_state["w_res1"].float() + polyak_alpha * cand_wr1.detach().cpu().float()).to(hof_state["w_res1"].dtype),
                "b_res1": ((1.0 - polyak_alpha) * hof_state["b_res1"].float() + polyak_alpha * cand_br1.detach().cpu().float()).to(hof_state["b_res1"].dtype),
                "w_res2": ((1.0 - polyak_alpha) * hof_state["w_res2"].float() + polyak_alpha * cand_wr2.detach().cpu().float()).to(hof_state["w_res2"].dtype),
                "w2": ((1.0 - polyak_alpha) * hof_state["w2"].float() + polyak_alpha * cand_w2.detach().cpu().float()).to(hof_state["w2"].dtype),
                "b2": ((1.0 - polyak_alpha) * hof_state["b2"].float() + polyak_alpha * cand_b2.detach().cpu().float()).to(hof_state["b2"].dtype),
                "w_rec": ((1.0 - polyak_new) * hof_state["w_rec"].float() + polyak_new * cand_wrec.detach().cpu().float()).to(hof_state["w_rec"].dtype),
                "b_rec": ((1.0 - polyak_new) * hof_state["b_rec"].float() + polyak_new * cand_brec.detach().cpu().float()).to(hof_state["b_rec"].dtype),
            }
            cand_polyak_gl = (1.0 - polyak_fast) * hof_g_logit + polyak_fast * cand_gl.detach().cpu()

            if b_fit > c0_fit and b_fills >= 1.0:
                hof_state = cand_polyak_state
                hof_g_logit = cand_polyak_gl

            hof_updated = False
            if it % 4 == 0 or it == iterations or (b_fit > c0_fit + 0.30 and b_prof_buds >= 4):
                ls_alphas = [0.05, 0.15, 0.35, 0.70, 1.00]
                ls_raw_states, ls_gls, ls_folded_candidates = [], [], []
                c_cpu = {
                    "w1": cand_w1.detach().cpu().float(),
                    "b1": cand_b1.detach().cpu().float(),
                    "w_res1": cand_wr1.detach().cpu().float(),
                    "b_res1": cand_br1.detach().cpu().float(),
                    "w_res2": cand_wr2.detach().cpu().float(),
                    "w2": cand_w2.detach().cpu().float(),
                    "b2": cand_b2.detach().cpu().float(),
                    "w_rec": cand_wrec.detach().cpu().float(),
                    "b_rec": cand_brec.detach().cpu().float(),
                }
                c_gl_cpu = cand_gl.detach().cpu().float()
                for a_ls in ls_alphas:
                    st_a = {k: (1.0 - a_ls) * best_val_state[k].float() + a_ls * c_cpu[k] for k in c_cpu}
                    gl_a = (1.0 - a_ls) * best_val_g_logit.float() + a_ls * c_gl_cpu
                    ls_raw_states.append(st_a)
                    ls_gls.append(gl_a)
                    ls_folded_candidates.append(policy.fold_gate_into_state(st_a, gl_a))

                ls_raw_states.append({k: v.float() for k, v in cand_polyak_state.items()})
                ls_gls.append(cand_polyak_gl.float())
                ls_folded_candidates.append(policy.fold_gate_into_state(cand_polyak_state, cand_polyak_gl))

                for el_rank in (0, 1):
                    el_g = int(sorted_g[el_rank].item())
                    raw_el = {k: getattr(policy, k)[el_g].detach().cpu().float() for k in c_cpu}
                    gl_el_c = policy.g_logits[el_g].detach().cpu().float()
                    ls_raw_states.append(raw_el)
                    ls_gls.append(gl_el_c)
                    ls_folded_candidates.append(policy.fold_gate_into_state(raw_el, gl_el_c))

                val_res_list = evaluate_l22_dual_basket(
                    vec_env, bf16_runner, lob_fu_all, v7_all, roll_sell_30s_all,
                    ls_folded_candidates, val_basket, win_len, tick_sz, device,
                    selective_front_cancel=True, true_latency=True,
                    allow_scratch_stop_asks=allow_scratch_stop_asks, panic_thresh=panic_thresh,
                )
                for c_idx, v_res in enumerate(val_res_list):
                    c_pu = float(v_res["pareto_util_bp"])
                    c_cu = float(v_res["clean_util_bp"])
                    c_uu = float(v_res["uw_util_bp"])
                    c_cev = float(v_res["clean_ev"])
                    c_fills = float(v_res["clean_fills"])
                    c_pump_ev = float(v_res["clean_pump_pct"])
                    c_dump_ev = float(v_res["clean_dump_pct"])
                    valid_active = (
                        (c_fills >= max(90.0, 0.72 * float(init_val_res["clean_fills"])))
                        and (c_cev >= max(0.080, 0.55 * float(init_val_res["clean_ev"])))
                        and (c_pump_ev >= max(0.75, 0.55 * float(init_val_res["clean_pump_pct"])))
                    )
                    strict_dom = (c_cu >= best_clean_util - 0.1) and (c_uu >= best_uw_util - 0.1) and (c_pu > best_pareto_util + 0.05)
                    pareto_gain = (c_pu > best_pareto_util + 0.10) and (c_cu >= best_clean_util - 1.0) and (c_uu >= best_uw_util - 2.0) and (c_dump_ev >= float(init_val_res["clean_dump_pct"]) - 0.05)
                    if valid_active and (strict_dom or pareto_gain):
                        best_pareto_util = c_pu
                        best_clean_util = c_cu
                        best_uw_util = c_uu
                        best_clean_ev = c_cev
                        best_val_ev = float(v_res["realistic_ev"])
                        best_val_folded = {k: v.clone() for k, v in ls_folded_candidates[c_idx].items()}
                        best_val_state = {k: v.to(hof_state[k].dtype).clone() for k, v in best_val_folded.items()}
                        best_val_g_logit = torch.zeros(N_BLOCKS_L21, dtype=torch.float32)
                        hof_state = {k: v.clone() for k, v in best_val_state.items()}
                        hof_g_logit = torch.zeros(N_BLOCKS_L21, dtype=torch.float32)
                        hof_updated = True
                        last_val_champ_it = it
                        ls_tag = f"alpha={ls_alphas[c_idx]:.2f}" if c_idx < len(ls_alphas) else ("polyak_hof" if c_idx == 5 else f"elite_{c_idx - 5}")
                        log_msg(
                            f"  🏆 НОВЫЙ ЧЕМПИОН УРОКА 24 it={it:04d} [{ls_tag}] | "
                            f"ParetoUtil={best_pareto_util:+.2f}bp [Clean24={best_clean_util:+.2f}bp (EV={best_clean_ev:+.4f}%, "
                            f"Fills={c_fills:.1f}, Cnl/Fill={v_res['clean_cnl_per_fill']:.1f}, η={v_res['clean_eta_queue_pct']:.1f}%) | "
                            f"UW4={best_uw_util:+.2f}bp (ΔMTM={v_res['uw_delta_ev']:+.4f}%)] | "
                            f"N={v_res['clean_norm_pct']:+.3f}% D+C={c_dump_ev:+.3f}% P={c_pump_ev:+.3f}% CVaR25={v_res['clean_cvar25_pct']:+.2f}% DD={v_res['clean_dd_pct']:.2f}%"
                        )
                        _save_l24_checkpoint(
                            SAVE_DIR / f"lesson_24_champ_iter_{it:04d}.pt", ck, best_val_folded, best_val_g_logit,
                            it, f"l24_cascade_survival_it{it:04d}", v_res,
                        )
                        _save_l24_checkpoint(
                            LATEST_CHAMP_PATH, ck, best_val_folded, best_val_g_logit,
                            it, f"l24_cascade_survival_it{it:04d}", v_res,
                        )

            if (it - last_val_champ_it) > 0 and (it - last_val_champ_it) % 100 == 0 and not hof_updated:
                hof_state = {
                    k: (0.65 * best_val_state[k].float() + 0.35 * hof_state[k].float()).to(best_val_state[k].dtype)
                    for k in best_val_state
                }
                hof_g_logit = 0.65 * best_val_g_logit + 0.35 * hof_g_logit

            policy.evolve_gated_groups(
                pareto_fitness,
                hof_state=hof_state,
                hof_g_logit=hof_g_logit,
                elite_frac=0.03,
                progress=cos_w,
            )
            dt_it = time.time() - t0
            history.append({
                "iteration": it,
                "best_pareto_util": best_pareto_util,
                "best_clean_util": best_clean_util,
                "best_uw_util": best_uw_util,
                "b_flat_ret": b_flat_ret,
                "b_casc_pct": b_casc_pct,
                "b_dd_pct": b_dd_pct,
                "dt": dt_it,
            })

            if it <= 5 or it % 10 == 0 or hof_updated or it == iterations:
                log_msg(
                    f"[У24 Ит {it:04d}/{iterations}] ValPareto={best_pareto_util:+.1f}bp | CleanEV={best_clean_ev:+.4f}% | "
                    f"Batch(N={b_norm_pct:+.2f}%, D={b_dump_pct:+.2f}%, Casc={b_casc_pct:+.2f}%, P={b_pump_pct:+.2f}%, "
                    f"UW={b_uw_ret:+.2f}%, DD={b_dd_pct:.2f}%, Ulc={b_ulcer:.2f}%, F={b_fills:.1f}, η={b_eta_queue:.0f}%, API={b_api:.2f}) "
                    f"| dt={dt_it:.2f}с"
                )
                summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
                _save_l24_checkpoint(
                    RESUME_CHAMP_PATH, ck, policy.fold_gate_into_state(hof_state, hof_g_logit), hof_g_logit,
                    it, f"l24_resume_it{it:04d}", init_val_res,
                )

    log_msg(f"🏁 Урок 24 завершён: {iterations} итераций | Лучший ValPareto={best_pareto_util:+.2f}bp (it={last_val_champ_it})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--n-groups", type=int, default=1024)
    ap.add_argument("--win-len", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=2026241)
    args = ap.parse_args()
    run_lesson_24(iterations=args.iterations, n_groups=args.n_groups, win_len=args.win_len, rng_seed=args.seed)
