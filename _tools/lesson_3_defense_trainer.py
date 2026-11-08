"""
SafeTrade PRL - УРОК 3 (v2): ЗАЩИТА ОТ ТОКСИЧНОГО ПОТОКА И ДИСЦИПЛИНА ОЧЕРЕДИ FIFO (8,192 МОДЕЛИ, H=256)
- 10 окон по 300с (5 стратифицированных рыночных + 5 стресс-окон с реальными проливами sweep/crash).
- Честное сохранение очереди FIFO (preserve_queue=True) и Penny Jump (q_ahead=0).
- Защита от падающих ножей (Pre-Trade Shield) и экстренная эвакуация (In-Trade Evasion).
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
LESSON_2_CHAMP_PATH = Path("/home/restorator/crypto_bot/models/prl/lesson_2_spread/lesson_2_champ_latest.pt")
OUT_DIR = Path("/home/restorator/crypto_bot/models/prl/lesson_3_defense")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("/home/restorator/crypto_bot/data/logs/lesson_3_defense.log")


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
        in_dim: int = 260,
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

        self.w1 = torch.randn(pop_size, in_dim, h_dim, dtype=self.dtype, device=device) * 0.05
        self.b1 = torch.zeros(pop_size, 1, h_dim, dtype=self.dtype, device=device)
        self.w2 = torch.randn(pop_size, h_dim, n_actions, dtype=self.dtype, device=device) * 0.05
        self.b2 = torch.zeros(pop_size, 1, n_actions, dtype=self.dtype, device=device)

    def load_from_champion(self, champ_data: dict):
        w1_c = champ_data["w1"].to(device=self.device, dtype=self.dtype)
        b1_c = champ_data["b1"].to(device=self.device, dtype=self.dtype)
        w2_c = champ_data["w2"].to(device=self.device, dtype=self.dtype)
        b2_c = champ_data["b2"].to(device=self.device, dtype=self.dtype)

        self.w1.copy_(w1_c.unsqueeze(0).expand_as(self.w1))
        self.b1.copy_(b1_c.unsqueeze(0).expand_as(self.b1))
        self.w2.copy_(w2_c.unsqueeze(0).expand_as(self.w2))
        self.b2.copy_(b2_c.unsqueeze(0).expand_as(self.b2))

        noise1 = (torch.randn_like(self.w1) * 0.02).to(self.dtype)
        noise2 = (torch.randn_like(self.w2) * 0.02).to(self.dtype)
        self.w1[1:].add_(noise1[1:])
        self.w2[1:].add_(noise2[1:])

    def forward_all(self, lob_b: torch.Tensor, flow_b: torch.Tensor, hidden: torch.Tensor, masks: torch.Tensor):
        lob_repr = self.backbone.encode_lob_batch(lob_b[:1]).expand(flow_b.size(0), -1)
        z, next_hidden = self.backbone.backbone.forward_step_fast(lob_repr, flow_b, hidden)

        p_sweep = torch.sigmoid(self.backbone.head_sweep_12s(z))
        p_bleed = torch.sigmoid(self.backbone.head_bleed_60s(z))
        p_crash = torch.sigmoid(self.backbone.head_crash_300s(z))
        p_exit = torch.sigmoid(self.backbone.head_exit_30s(z))
        z_aug = torch.cat([z, p_sweep, p_bleed, p_crash, p_exit], dim=-1).unsqueeze(1).to(self.dtype)

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

        n_fine = int(self.pop_size * 0.50)
        n_mid = int(self.pop_size * 0.35)

        sigmas = torch.zeros(self.pop_size, 1, 1, dtype=self.dtype, device=self.device)
        sigmas[top_k:n_fine] = 0.012
        sigmas[n_fine:n_fine + n_mid] = 0.040
        sigmas[n_fine + n_mid:] = 0.100

        self.w1.add_((torch.randn_like(self.w1) * sigmas).to(self.dtype))
        self.b1.add_((torch.randn_like(self.b1) * (sigmas * 0.5)).to(self.dtype))
        self.w2.add_((torch.randn_like(self.w2) * sigmas).to(self.dtype))
        self.b2.add_((torch.randn_like(self.b2) * (sigmas * 0.5)).to(self.dtype))


def run_defense_simulation_8k(
    policy: BatchedPopulationPolicy8K,
    vec_env: VectorizedCudaL2Env,
    total_ticks: int,
    stress_anchors: np.ndarray,
    sim_steps: int = 3000,
    n_windows: int = 10,
    device: torch.device = torch.device("cuda"),
) -> Tuple[Dict[str, float], int, torch.Tensor]:
    pop_size = policy.pop_size
    win_len = sim_steps // n_windows
    n_reg = n_windows // 2
    n_stress = n_windows - n_reg

    seg_len = (total_ticks - win_len - 200) // n_reg
    reg_starts = [
        100 + i * seg_len + int(np.random.randint(0, max(1, seg_len - win_len)))
        for i in range(n_reg - 1)
    ]
    recent_floor = max(100, total_ticks - 72 * 3600)
    reg_starts.append(int(np.random.randint(recent_floor, max(recent_floor + 1, total_ticks - win_len - 100))))

    if len(stress_anchors) >= n_stress:
        recent_stress = stress_anchors[stress_anchors >= recent_floor]
        if len(recent_stress) >= 1 and n_stress >= 2:
            c_old = np.random.choice(stress_anchors, size=n_stress - 1, replace=False)
            c_rec = np.random.choice(recent_stress, size=1, replace=False)
            chosen = np.concatenate([c_old, c_rec])
        else:
            chosen = np.random.choice(stress_anchors, size=n_stress, replace=False)
        stress_starts = [int(np.clip(c - np.random.randint(30, 120), 100, total_ticks - win_len - 100)) for c in chosen]
    else:
        stress_starts = [int(np.random.randint(100, total_ticks - win_len - 100)) for _ in range(n_stress)]

    win_starts = sorted(reg_starts + stress_starts)
    vec_env.reset_interval(start_idx=0, end_idx=total_ticks, episode_len=total_ticks)

    completed_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    profitable_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    bad_narrow_cycles = torch.zeros(pop_size, dtype=torch.float32, device=device)
    shield_cancels = torch.zeros(pop_size, dtype=torch.float32, device=device)
    toxic_buys = torch.zeros(pop_size, dtype=torch.float32, device=device)
    evasion_exits = torch.zeros(pop_size, dtype=torch.float32, device=device)
    inventory_lock_steps = torch.zeros(pop_size, dtype=torch.float32, device=device)

    prev_pos = vec_env.pos_qty[:pop_size].clone()
    prev_cash = vec_env.cash[:pop_size].clone()
    entry_cash_snap = vec_env.cash[:pop_size].clone()
    entered_narrow = torch.zeros(pop_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for w_start in win_starts:
            h_states = policy.backbone.init_hidden(batch_size=pop_size, device=device)
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

                actions, h_states = policy.forward_all(lob, flow[:pop_size], h_states, masks[:pop_size])

                if is_danger_knife:
                    shield_cancels += (had_bid & (actions == 5)).float()

                if is_bleed_exit:
                    evasion_exits += (in_pos & ((actions == 3) | (actions == 6))).float()

                vec_env.step(actions)
                cur_pos = vec_env.pos_qty[:pop_size]
                cur_cash = vec_env.cash[:pop_size]

                no_sell_order = vec_env.order_side[:pop_size] != 1
                inventory_lock_steps += ((cur_pos > 1e-6) & no_sell_order).float()

                just_bought = (prev_pos < 1e-6) & (cur_pos > 1e-6)
                if just_bought.any():
                    entry_cash_snap = torch.where(just_bought, prev_cash, entry_cash_snap)
                    entered_narrow = torch.where(just_bought, spread_ticks < 2.0, entered_narrow)
                    if is_danger_knife:
                        toxic_buys += just_bought.float()

                just_closed = (prev_pos > 1e-6) & (cur_pos < 1e-6)
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
        - (toxic_buys * 20.0)
        - (bad_narrow_cycles * 15.0)
        - (inventory_lock_steps * 0.05)
    )
    best_idx = int(scores.argmax().item())
    worst_idx = int(scores.argmin().item())

    champ_dict = {
        "ind_id": best_idx + 1,
        "score": float(scores[best_idx].item()),
        "pnl": float(pnls[best_idx].item()),
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

    stress_mask = (targets_np[:, 0] > 0.5) | (targets_np[:, 2] > 0.5)
    stress_anchors = np.where(stress_mask)[0]
    stress_anchors = stress_anchors[(stress_anchors > 500) & (stress_anchors < n_ticks - 500)]
    if len(stress_anchors) > 5000:
        stress_anchors = stress_anchors[:: max(1, len(stress_anchors) // 5000)]

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
    log_msg(f"📦 Загружен кэш LOB: {n_ticks:,} тиков ({n_ticks / 3600:.1f} ч) | Стресс-точек: {len(stress_anchors):,}")
    return vec_env, n_ticks, stress_anchors, cache_mtime


def run_lesson_3(pop_size: int = 8192, total_iterations: int = 1000, sim_steps: int = 3000, resume: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg("=" * 80)
    log_msg(f"🛡️ СТАРТ УРОКА 3 (v2): ЗАЩИТА ОТ ПРОЛИВОВ И ОЧЕРЕДЬ FIFO ({pop_size:,} МОДЕЛЕЙ, H=256, {total_iterations} ИТ)")
    log_msg("=" * 80)

    vec_env, n_ticks, stress_anchors, last_cache_mtime = load_env_from_cache(pop_size, device)

    history_summary = []
    start_it = 1
    latest_pt = OUT_DIR / "lesson_3_champ_latest.pt"
    summary_file = OUT_DIR / "lesson_3_summary.json"

    if resume and latest_pt.exists() and summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                history_summary = json.load(f)
            start_it = len(history_summary) + 1
            c_data = torch.load(latest_pt, map_location=device, weights_only=False)
            log_msg(f"🔄 ВОЗОБНОВЛЕНИЕ УРОКА 3 С ИТЕРАЦИИ {start_it}/{total_iterations} (загружен {latest_pt})")
        except Exception as e:
            log_msg(f"⚠️ Не удалось возобновить чекпоинт ({e}), старт с Чемпиона Урока 2.")
            start_it = 1
            history_summary = []
            c_data = torch.load(LESSON_2_CHAMP_PATH, map_location=device, weights_only=False)
    else:
        c_data = torch.load(LESSON_2_CHAMP_PATH, map_location=device, weights_only=False)

    backbone_model = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone_model.load_state_dict(c_data["backbone_state"], strict=False)
    backbone_model.eval()

    policy = BatchedPopulationPolicy8K(backbone=backbone_model, pop_size=pop_size, h_dim=256, device=device)
    policy.load_from_champion(c_data)

    for it in range(start_it, total_iterations + 1):
        try:
            cur_mtime = CACHE_PATH.stat().st_mtime
            if cur_mtime > last_cache_mtime + 1.0:
                log_msg("⚡ [HOT RELOAD] Обнаружено обновление файла кэша на диске! Перезагрузка тензоров в GPU...")
                del vec_env
                torch.cuda.empty_cache()
                vec_env, n_ticks, stress_anchors, last_cache_mtime = load_env_from_cache(pop_size, device)
        except Exception as e:
            log_msg(f"⚠️ Ошибка проверки обновления кэша: {e}")

        t0 = time.time()
        champ_dict, champ_idx, scores = run_defense_simulation_8k(
            policy=policy, vec_env=vec_env, total_ticks=n_ticks, stress_anchors=stress_anchors, sim_steps=sim_steps, device=device
        )
        elapsed = time.time() - t0
        log_msg(
            f"👑 [ИТОГ ИТЕРАЦИИ {it:03d}/{total_iterations}] ЧЕМПИОН #{champ_dict['ind_id']} | "
            f"Score: {champ_dict['score']:.1f} | Net PnL: ${champ_dict['pnl']:+.2f} | "
            f"Циклов: {champ_dict['cycles']} (Приб: {champ_dict['prof_cycles']}) | "
            f"Щит/Отмен: {champ_dict['shield_cancels']} | Ножей: {champ_dict['toxic_buys']} | "
            f"Эвак: {champ_dict['evasion_exits']} | Худший: {champ_dict['worst_score']:.1f} ({elapsed:.1f}с)"
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
        torch.save(champ_dict_save, OUT_DIR / "lesson_3_champ_latest.pt")

        history_summary.append({"iteration": it, "champion_id": champ_dict["ind_id"], "score": champ_dict["score"], "pnl": champ_dict["pnl"], "metrics": champ_dict, "time_sec": elapsed})
        with open(OUT_DIR / "lesson_3_summary.json", "w") as f:
            json.dump(history_summary, f, indent=2)

        policy.clone_and_mutate(scores=scores, top_k=64)


if __name__ == "__main__":
    run_lesson_3(pop_size=8192, total_iterations=1000, sim_steps=3000)
