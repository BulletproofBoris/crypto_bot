"""
SafeTrade PRL - УРОК 4 (v2): УПРАВЛЕНИЕ ИНВЕНТАРЕМ (AVELLANEDA-STOIKOV SKEW) И КОНТРОЛЬ ПРОСАДКИ MAX DD
- Длинные непрерывные окна: 2 окна по 1,500с (25 минут каждое, суммарно 3,000с).
- Контроль возраста позиции (Inventory Aging): штраф за заморозку капитала > 180с и бонус за Skew-разгрузку.
- Траекторный контроль риска: отслеживание пикового Equity на каждом тике и жесткий штраф за Max Drawdown.
- Сохранение всех защитных навыков Урока 3 (preserve_queue=True, Pre-Trade Shield, In-Trade Evasion).
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_cuda_l2_env import VectorizedCudaL2Env
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors

CACHE_PATH = Path("/home/restorator/crypto_bot/data/cache/rl_guard_lob_cache_600h.npz")
LESSON_3_CHAMP_PATH = Path("/home/restorator/crypto_bot/models/prl/lesson_3_defense/lesson_3_champ_latest.pt")
OUT_DIR = Path("/home/restorator/crypto_bot/models/prl/lesson_4_inventory")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("/home/restorator/crypto_bot/data/logs/lesson_4_inventory.log")


def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


class BatchedPopulationPolicy8K:
    def __init__(
        self,
        backbone: nn.Module,
        pop_size: int = 8192,
        in_dim: int = 318,
        h_dim: int = 256,
        n_actions: int = 7,
        device: torch.device = torch.device("cuda"),
    ):
        self.backbone = backbone
        self.pop_size = pop_size
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.n_actions = n_actions
        self.device = device
        self.dtype = torch.bfloat16

        self.w1 = torch.zeros(pop_size, in_dim, h_dim, dtype=self.dtype, device=device)
        self.b1 = torch.zeros(pop_size, 1, h_dim, dtype=self.dtype, device=device)
        self.w2 = torch.randn(pop_size, h_dim, n_actions, dtype=self.dtype, device=device) * 0.05
        self.b2 = torch.zeros(pop_size, 1, n_actions, dtype=self.dtype, device=device)

    def load_from_champion(self, champ_data: dict):
        w1_c = champ_data["w1"].to(device=self.device, dtype=self.dtype)
        b1_c = champ_data["b1"].to(device=self.device, dtype=self.dtype)
        w2_c = champ_data["w2"].to(device=self.device, dtype=self.dtype)
        b2_c = champ_data["b2"].to(device=self.device, dtype=self.dtype)

        src_in_dim = w1_c.size(0)
        self.w1.zero_()
        self.w1[:, :src_in_dim, :].copy_(w1_c.unsqueeze(0).expand(self.pop_size, src_in_dim, self.h_dim))
        self.b1.copy_(b1_c.unsqueeze(0).expand_as(self.b1))
        self.w2.copy_(w2_c.unsqueeze(0).expand_as(self.w2))
        self.b2.copy_(b2_c.unsqueeze(0).expand_as(self.b2))

        noise1 = (torch.randn_like(self.w1) * 0.015).to(self.dtype)
        noise2 = (torch.randn_like(self.w2) * 0.015).to(self.dtype)
        self.w1[1:].add_(noise1[1:])
        self.w2[1:].add_(noise2[1:])

    def forward_all(
        self,
        lob_b: torch.Tensor,
        flow_b: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        v7_row: torch.Tensor,
    ):
        B = flow_b.size(0)
        lob_repr = self.backbone.encode_lob_batch(lob_b[:1]).expand(B, -1)
        z, next_hidden = self.backbone.backbone.forward_step_fast(lob_repr, flow_b, hidden)

        p_sweep = torch.sigmoid(self.backbone.head_sweep_12s(z))
        p_bleed = torch.sigmoid(self.backbone.head_bleed_60s(z))
        p_crash = torch.sigmoid(self.backbone.head_crash_300s(z))
        p_exit = torch.sigmoid(self.backbone.head_exit_30s(z))
        v7_exp = v7_row.unsqueeze(0).expand(B, -1)
        z_aug = torch.cat([z, p_sweep, p_bleed, p_crash, p_exit, v7_exp], dim=-1).unsqueeze(1).to(self.dtype)

        h1 = torch.relu(torch.bmm(z_aug, self.w1) + self.b1)
        logits = (torch.bmm(h1, self.w2) + self.b2).squeeze(1).float()
        logits = logits.masked_fill(~masks, -1e9)

        actions = logits.argmax(dim=-1)
        return actions, next_hidden

    def clone_and_mutate(self, scores: torch.Tensor, top_k: int = 64):
        _, top_indices = torch.topk(scores, k=top_k)

        elite_w1 = self.w1[top_indices].clone()
        elite_b1 = self.b1[top_indices].clone()
        elite_w2 = self.w2[top_indices].clone()
        elite_b2 = self.b2[top_indices].clone()

        repeats = self.pop_size // top_k
        self.w1.copy_(elite_w1.repeat(repeats, 1, 1))
        self.b1.copy_(elite_b1.repeat(repeats, 1, 1))
        self.w2.copy_(elite_w2.repeat(repeats, 1, 1))
        self.b2.copy_(elite_b2.repeat(repeats, 1, 1))

        n_fine = int(self.pop_size * 0.55)
        n_mid = int(self.pop_size * 0.30)

        sigmas = torch.zeros(self.pop_size, 1, 1, dtype=self.dtype, device=self.device)
        sigmas[top_k:n_fine] = 0.010
        sigmas[n_fine:n_fine + n_mid] = 0.035
        sigmas[n_fine + n_mid:] = 0.085

        self.w1.add_((torch.randn_like(self.w1) * sigmas).to(self.dtype))
        self.b1.add_((torch.randn_like(self.b1) * (sigmas * 0.5)).to(self.dtype))
        self.w2.add_((torch.randn_like(self.w2) * sigmas).to(self.dtype))
        self.b2.add_((torch.randn_like(self.b2) * (sigmas * 0.5)).to(self.dtype))


def run_inventory_skew_simulation_8k(
    policy: BatchedPopulationPolicy8K,
    vec_env: VectorizedCudaL2Env,
    v7_signals_tensor: torch.Tensor,
    total_ticks: int,
    stress_anchors: np.ndarray,
    sim_steps: int = 3000,
    n_windows: int = 2,
    device: torch.device = torch.device("cuda"),
) -> Tuple[Dict[str, float], int, torch.Tensor]:
    pop_size = policy.pop_size
    win_len = sim_steps // n_windows
    n_reg = n_windows // 2
    n_stress = n_windows - n_reg

    recent_floor = max(200, total_ticks - 72 * 3600)
    if np.random.rand() < 0.50:
        reg_starts = [int(np.random.randint(recent_floor, max(recent_floor + 1, total_ticks - win_len - 200))) for _ in range(n_reg)]
    else:
        seg_len = (total_ticks - win_len - 500) // max(1, n_reg)
        reg_starts = [
            200 + i * seg_len + int(np.random.randint(0, max(1, seg_len - win_len)))
            for i in range(n_reg)
        ]

    if len(stress_anchors) >= n_stress:
        recent_stress = stress_anchors[stress_anchors >= recent_floor]
        pool = recent_stress if (len(recent_stress) >= n_stress and np.random.rand() < 0.50) else stress_anchors
        chosen = np.random.choice(pool, size=n_stress, replace=False)
        stress_starts = [
            int(np.clip(c - np.random.randint(120, 600), 200, total_ticks - win_len - 200))
            for c in chosen
        ]
    else:
        stress_starts = [int(np.random.randint(200, total_ticks - win_len - 200)) for _ in range(n_stress)]

    win_starts = sorted(reg_starts + stress_starts)
    vec_env.reset_interval(start_idx=0, end_idx=total_ticks, episode_len=total_ticks)

    completed_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    profitable_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    bad_narrow_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    shield_cancels = torch.zeros(pop_size, dtype=torch.float32, device=device)
    toxic_buys = torch.zeros(pop_size, dtype=torch.float32, device=device)
    evasion_exits = torch.zeros(pop_size, dtype=torch.float32, device=device)
    inventory_lock_steps = torch.zeros(pop_size, dtype=torch.float32, device=device)

    pos_age = torch.zeros(pop_size, dtype=torch.float32, device=device)
    max_hold_steps = torch.zeros(pop_size, dtype=torch.float32, device=device)
    stale_hold_steps = torch.zeros(pop_size, dtype=torch.float32, device=device)
    skew_unloads = torch.zeros(pop_size, dtype=torch.float32, device=device)
    max_drawdown = torch.zeros(pop_size, dtype=torch.float32, device=device)

    prev_pos = vec_env.pos_qty[:pop_size].clone()
    prev_cash = vec_env.cash[:pop_size].clone()
    entry_cash_snap = vec_env.cash[:pop_size].clone()
    entered_narrow = torch.zeros(pop_size, dtype=torch.bool, device=device)
    peak_equity = vec_env.cash[:pop_size].clone()

    with torch.no_grad():
        for w_start in win_starts:
            h_states = policy.backbone.init_hidden(batch_size=pop_size, device=device)
            pos_age.zero_()
            for step_in_win in range(win_len):
                cur_t = int(w_start + step_in_win)
                vec_env.t[:pop_size] = cur_t

                lob, flow, tgt = vec_env.get_observations(cur_t_scalar=cur_t)
                masks = vec_env.get_action_masks(cur_t_scalar=cur_t)

                b1 = vec_env.bids[cur_t, 0]
                a1 = vec_env.asks[cur_t, 0]
                spread_ticks = torch.round((a1 - b1) / max(1e-4, vec_env.tick_size))

                tgt_row = tgt[0]
                is_danger_knife = bool((tgt_row[0] > 0.5) or (tgt_row[2] > 0.5))
                is_bleed_exit = bool((tgt_row[1] > 0.5) or (tgt_row[3] > 0.5))

                had_bid = vec_env.order_side[:pop_size] == 0
                in_pos = prev_pos > 1e-6

                v7_row = v7_signals_tensor[cur_t]
                actions, h_states = policy.forward_all(lob, flow[:pop_size], h_states, masks[:pop_size], v7_row)

                if is_danger_knife:
                    shield_cancels += (had_bid & (actions == 5)).float()

                if is_bleed_exit:
                    evasion_exits += (in_pos & ((actions == 3) | (actions == 6))).float()

                is_aging_pos = in_pos & (pos_age >= 90.0) & (pos_age <= 240.0)
                if is_aging_pos.any():
                    skew_unloads += (is_aging_pos & ((actions == 3) | (actions == 4))).float() * 0.1

                vec_env.step(actions)
                cur_pos = vec_env.pos_qty[:pop_size]
                cur_cash = vec_env.cash[:pop_size]

                still_in_pos = cur_pos > 1e-6
                pos_age = torch.where(still_in_pos, pos_age + 1.0, torch.zeros_like(pos_age))
                max_hold_steps = torch.maximum(max_hold_steps, pos_age)
                stale_hold_steps += (pos_age > 180.0).float()

                no_sell_order = vec_env.order_side[:pop_size] != 1
                inventory_lock_steps += (still_in_pos & no_sell_order).float()

                cur_equity = cur_cash + cur_pos * torch.clamp(b1, min=vec_env.tick_size)
                peak_equity = torch.maximum(peak_equity, cur_equity)
                cur_dd = torch.clamp(peak_equity - cur_equity, min=0.0)
                max_drawdown = torch.maximum(max_drawdown, cur_dd)

                just_bought = (prev_pos < 1e-6) & still_in_pos
                if just_bought.any():
                    entry_cash_snap = torch.where(just_bought, prev_cash, entry_cash_snap)
                    entered_narrow = torch.where(just_bought, spread_ticks < 2.0, entered_narrow)
                    if is_danger_knife:
                        toxic_buys += just_bought.float()

                just_closed = (prev_pos > 1e-6) & (~still_in_pos)
                if just_closed.any():
                    completed_cycles += just_closed.float()
                    cycle_pnl = cur_cash - entry_cash_snap
                    is_prof = just_closed & (cycle_pnl > 0.0)
                    is_bad_nar = just_closed & (cycle_pnl <= 0.0) & entered_narrow
                    profitable_cycles += is_prof.float()
                    bad_narrow_cycles += is_bad_nar.float()

                prev_pos = cur_pos.clone()
                prev_cash = cur_cash.clone()

            last_t = int(w_start + win_len - 1)
            b1_last = torch.clamp(vec_env.bids[last_t, 0], min=vec_env.tick_size)
            has_rem_pos = vec_env.pos_qty[:pop_size] > 1e-6
            if has_rem_pos.any():
                rem_qty = vec_env.pos_qty[:pop_size][has_rem_pos]
                fee = rem_qty * b1_last * vec_env.taker_fee
                vec_env.cash[:pop_size][has_rem_pos] += (rem_qty * b1_last - fee)
                vec_env.pos_qty[:pop_size][has_rem_pos] = 0.0
                vec_env.avg_entry[:pop_size][has_rem_pos] = 0.0
                vec_env.n_trades[:pop_size][has_rem_pos] += 1

            cur_equity_end = vec_env.cash[:pop_size]
            peak_equity = torch.maximum(peak_equity, cur_equity_end)
            max_drawdown = torch.maximum(max_drawdown, torch.clamp(peak_equity - cur_equity_end, min=0.0))

            vec_env.prev_mtm[:pop_size] = vec_env.cash[:pop_size] - vec_env.initial_cash
            vec_env.order_side[:pop_size].fill_(-1)
            vec_env.order_price[:pop_size].zero_()
            vec_env.order_qty[:pop_size].zero_()
            vec_env.q_ahead[:pop_size].zero_()
            vec_env.active_after[:pop_size].zero_()
            prev_pos.zero_()
            prev_cash = vec_env.cash[:pop_size].clone()
            entered_narrow.zero_()

    pnls = vec_env.prev_mtm[:pop_size]
    cycle_bonus = torch.clamp(profitable_cycles, max=50.0) * 6.0
    inactivity_pen = torch.clamp(12.0 - completed_cycles, min=0.0) * 20.0
    scores = (
        (pnls * 100.0)
        + cycle_bonus
        - inactivity_pen
        + (shield_cancels * 2.0)
        + (evasion_exits * 3.0)
        + torch.clamp(skew_unloads, max=25.0)
        - (max_drawdown * 150.0)
        - (stale_hold_steps * 0.08)
        - (toxic_buys * 20.0)
        - (bad_narrow_cycles * 15.0)
        - (inventory_lock_steps * 0.05)
    )
    best_idx = int(scores.argmax().item())
    worst_idx = int(scores.argmin().item())

    init_cap = float(vec_env.initial_cash)
    pnl_val = float(pnls[best_idx].item())
    dd_val = float(max_drawdown[best_idx].item())
    champ_dict = {
        "ind_id": best_idx + 1,
        "score": float(scores[best_idx].item()),
        "pnl": pnl_val,
        "pnl_pct": round(100.0 * pnl_val / init_cap, 2),
        "max_dd": dd_val,
        "max_dd_pct": round(100.0 * dd_val / init_cap, 2),
        "max_hold_s": int(max_hold_steps[best_idx].item()),
        "stale_steps": int(stale_hold_steps[best_idx].item()),
        "cycles": int(completed_cycles[best_idx].item()),
        "prof_cycles": int(profitable_cycles[best_idx].item()),
        "bad_narrow": int(bad_narrow_cycles[best_idx].item()),
        "shield_cancels": int(shield_cancels[best_idx].item()),
        "toxic_buys": int(toxic_buys[best_idx].item()),
        "evasion_exits": int(evasion_exits[best_idx].item()),
        "inv_lock_steps": int(inventory_lock_steps[best_idx].item()),
        "trades": int(vec_env.n_trades[best_idx].item()),
        "worst_score": float(scores[worst_idx].item()),
    }

    return champ_dict, best_idx, scores


def load_env_from_cache(pop_size: int, device: torch.device):
    cache_mtime = CACHE_PATH.stat().st_mtime
    raw = np.load(CACHE_PATH)
    n_ticks = len(raw["bid_p"])
    lob_tensors, flow_market = build_lob_10_and_flow_tensors(raw)
    targets_np = np.stack([raw["y_sweep_12s"], raw["y_bleed_60s"], raw["y_crash_300s"], raw["y_exit_30s"]], axis=1)
    targets_tensor = torch.from_numpy(targets_np).float()

    stress_mask = (targets_np[:, 1] > 0.5) | (targets_np[:, 2] > 0.5)
    stress_anchors = np.where(stress_mask)[0]
    stress_anchors = stress_anchors[(stress_anchors > 1600) & (stress_anchors < n_ticks - 1600)]
    if len(stress_anchors) > 5000:
        stress_anchors = stress_anchors[:: max(1, len(stress_anchors) // 5000)]

    if "v7_signals_58d" in raw.files and len(raw["v7_signals_58d"]) == n_ticks:
        v7_np = raw["v7_signals_58d"].astype(np.float32)
    else:
        log_msg("⚠️ В кэше отсутствует v7_signals_58d, запуск сборки гибридного кэша v7+v8...")
        from _tools.build_hybrid_v7_v8_cache import build_and_inject_v7_signals_58d
        build_and_inject_v7_signals_58d(coin="prl", force_recompute=False)
        raw = np.load(CACHE_PATH)
        cache_mtime = CACHE_PATH.stat().st_mtime
        v7_np = raw["v7_signals_58d"].astype(np.float32)

    v7_signals_tensor = torch.from_numpy(v7_np).to(device=device, dtype=torch.bfloat16)

    vec_env = VectorizedCudaL2Env(
        bids=torch.from_numpy(raw["bid_p"]).float(),
        asks=torch.from_numpy(raw["ask_p"]).float(),
        bids_q=torch.from_numpy(raw["bid_q"]).float(),
        asks_q=torch.from_numpy(raw["ask_q"]).float(),
        buy_vol=torch.from_numpy(raw["buy_vol"]).float(),
        sell_vol=torch.from_numpy(raw["sell_vol"]).float(),
        lob_tensors=torch.from_numpy(lob_tensors).float(),
        flow_market=torch.from_numpy(flow_market).float(),
        targets=targets_tensor,
        num_envs=pop_size,
        device=device,
        preserve_queue=True,
    )
    log_msg(f"📦 Загружен гибридный кэш v7+v8: {n_ticks:,} тиков ({n_ticks / 3600:.1f} ч) | v7(58D) + v8(260D) = 318D")
    return vec_env, v7_signals_tensor, n_ticks, stress_anchors, cache_mtime


def run_lesson_4(pop_size: int = 8192, total_iterations: int = 1000, sim_steps: int = 3000, resume: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg("=" * 80)
    log_msg(f"⚖️ СТАРТ УРОКА 4 (v2 ГИБРИД v7+v8 318D): INVENTORY SKEW И КОНТРОЛЬ MAX DD ({pop_size:,} МОДЕЛЕЙ, {total_iterations} ИТ)")
    log_msg("=" * 80)

    vec_env, v7_signals_tensor, n_ticks, stress_anchors, last_cache_mtime = load_env_from_cache(pop_size, device)

    history_summary = []
    start_it = 1
    latest_pt = OUT_DIR / "lesson_4_champ_latest.pt"
    summary_file = OUT_DIR / "lesson_4_summary.json"

    if resume and latest_pt.exists() and summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                history_summary = json.load(f)
            start_it = len(history_summary) + 1
            c_data = torch.load(latest_pt, map_location=device, weights_only=False)
            log_msg(f"🔄 ВОЗОБНОВЛЕНИЕ УРОКА 4 С ИТЕРАЦИИ {start_it}/{total_iterations} (загружен {latest_pt})")
        except Exception as e:
            log_msg(f"⚠️ Не удалось возобновить чекпоинт ({e}), старт с Чемпиона Урока 3.")
            start_it = 1
            history_summary = []
            c_data = torch.load(LESSON_3_CHAMP_PATH, map_location=device, weights_only=False)
    else:
        c_data = torch.load(LESSON_3_CHAMP_PATH, map_location=device, weights_only=False)

    backbone_model = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone_model.load_state_dict(c_data["backbone_state"], strict=False)
    backbone_model.eval()

    policy = BatchedPopulationPolicy8K(backbone=backbone_model, pop_size=pop_size, in_dim=318, h_dim=256, device=device)
    policy.load_from_champion(c_data)

    for it in range(start_it, total_iterations + 1):
        try:
            cur_mtime = CACHE_PATH.stat().st_mtime
            if cur_mtime > last_cache_mtime + 1.0:
                log_msg("⚡ [HOT RELOAD] Обнаружено обновление файла кэша на диске! Перезагрузка тензоров в GPU...")
                del vec_env, v7_signals_tensor
                torch.cuda.empty_cache()
                vec_env, v7_signals_tensor, n_ticks, stress_anchors, last_cache_mtime = load_env_from_cache(pop_size, device)
        except Exception as e:
            log_msg(f"⚠️ Ошибка проверки обновления кэша: {e}")

        t0 = time.time()
        champ_dict, champ_idx, scores = run_inventory_skew_simulation_8k(
            policy=policy,
            vec_env=vec_env,
            v7_signals_tensor=v7_signals_tensor,
            total_ticks=n_ticks,
            stress_anchors=stress_anchors,
            sim_steps=sim_steps,
            device=device,
        )
        elapsed = time.time() - t0
        pnl_pct = 100.0 * champ_dict["pnl"] / 95.5
        dd_pct = 100.0 * champ_dict["max_dd"] / 95.5
        log_msg(
            f"👑 [ИТОГ ИТЕРАЦИИ {it:03d}/{total_iterations}] ЧЕМПИОН #{champ_dict['ind_id']} | "
            f"Score: {champ_dict['score']:.1f} | Net PnL: {pnl_pct:+.2f}% (${champ_dict['pnl']:+.2f}) | "
            f"MaxDD: {dd_pct:.2f}% (${champ_dict['max_dd']:.2f}) | MaxHold: {champ_dict['max_hold_s']}с | "
            f"Циклов: {champ_dict['cycles']} (Приб: {champ_dict['prof_cycles']}) | "
            f"Щит/Эвак: {champ_dict['shield_cancels']}/{champ_dict['evasion_exits']} | "
            f"Худший: {champ_dict['worst_score']:.1f} ({elapsed:.1f}с)"
        )

        champ_dict_save = {
            "iteration": it,
            "champ_id": champ_dict["ind_id"],
            "score": champ_dict["score"],
            "pnl": champ_dict["pnl"],
            "metrics": champ_dict,
            "w1": policy.w1[champ_idx].cpu(),
            "b1": policy.b1[champ_idx].cpu(),
            "w2": policy.w2[champ_idx].cpu(),
            "b2": policy.b2[champ_idx].cpu(),
            "backbone_state": backbone_model.state_dict(),
        }
        torch.save(champ_dict_save, OUT_DIR / f"iteration_{it:03d}_champ.pt")
        torch.save(champ_dict_save, OUT_DIR / "lesson_4_champ_latest.pt")

        history_summary.append({"iteration": it, "champion_id": champ_dict["ind_id"], "score": champ_dict["score"], "pnl": champ_dict["pnl"], "metrics": champ_dict, "time_sec": elapsed})
        with open(OUT_DIR / "lesson_4_summary.json", "w") as f:
            json.dump(history_summary, f, indent=2)

        policy.clone_and_mutate(scores=scores, top_k=64)


if __name__ == "__main__":
    run_lesson_4(pop_size=8192, total_iterations=1000, sim_steps=3000)
