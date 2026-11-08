#!/usr/bin/env python3
"""
_tools/lesson_6_free_multi_order_trainer.py
=============================================================================
Урок 6 v3: 24 Одновременных Независимых Ордера (12 Bid + 12 Ask Слотов),
Мульти-Бюджетная Синхронная Оценка (1365 Кандидатов × 6 Бюджетов = 8190 Сред),
Hall-of-Fame Элитизм + Защита от Вырождения Весов (Weight Norm Clamping)

Что исправлено для устранения разброса и стабилизации обучения:
1. СИНХРОННЫЙ 6-БЮДЖЕТНЫЙ ЭКЗАМЕН ДЛЯ КАЖДОГО МУТАНТА (1365 × 6 = 8190):
   - Больше никакой лотереи одиночного бюджета! Каждый из 1,365 кандидатов
     одновременно тестируется сразу на ВСЕХ 6 бюджетах ($95.5, $1k, $3k, $5k, $10k, $20k).
   - В элиту попадают только те политики, которые дают высокий средний результат
     по всем 6 бюджетам сразу.
2. HALL-OF-FAME ЗАЩИТА ЧЕМПИОНА НА МЕРТВЫХ ОКНАХ:
   - Группа #0 (среды 0..5) всегда хранит действующего Чемпиона Зала Славы.
   - Если случайное окно рынка оказалось пустым/мертвым, Чемпион НЕ затирается
     случайным мутантом, а обновляется только при чистом плюсе (>= 4/6 бюджетов в плюс)
     и победе над Группой #0 на этом же окне.
3. WEIGHT NORM CLAMPING (ЗАЩИТА ОТ НАСЫЩЕНИЯ SIGMOID):
   - Нормы столбцов `w1` и `w2` жестко удерживаются в чувствительном коридоре,
     чтобы `sigmoid` всегда выбирал плавные доли объема (10%..100%), а не скакал в 0/1.
4. ПРОЗРАЧНЫЙ ЛОГ СКОЛЬЗЯЩЕГО ТРЕНДА (Roll20):
   - Каждая строка показывает средний результат по всем 6 бюджетам и скользящее среднее
     за последние 20 итераций (`Roll20`), чтобы динамика обучения была видна как на ладони.
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

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
L5_CHAMP_PATH = BASE_DIR / "models" / "prl" / "lesson_5_whale" / "lesson_5_champ_latest.pt"
SAVE_DIR = BASE_DIR / "models" / "prl" / "lesson_6_free_grid"
LOG_FILE = BASE_DIR / "data" / "logs" / "lesson_6_free_grid.log"


def log_msg(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class BatchedFreeGridPolicyGrouped(nn.Module):
    """
    Популяция из `n_groups = 1365` кандидатов × `n_budgets = 6` бюджетов = `8,190` параллельных сред.
    Каждая группа `g` имеет идентичные веса на всех 6 бюджетах и оценивается как единый портфель.
    """

    def __init__(
        self,
        backbone: MultiTaskLOBNet,
        n_groups: int = 1365,
        n_budgets: int = 6,
        in_dim: int = 324,
        h_dim: int = 256,
        out_dim: int = 26,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.backbone = backbone
        self.n_groups = n_groups
        self.n_budgets = n_budgets
        self.pop_size = n_groups * n_budgets  # 8190
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.w1 = nn.Parameter(torch.randn(self.pop_size, in_dim, h_dim, device=device, dtype=self.dtype) * 0.04, requires_grad=False)
        self.b1 = nn.Parameter(torch.zeros(self.pop_size, 1, h_dim, device=device, dtype=self.dtype), requires_grad=False)
        self.w2 = nn.Parameter(torch.randn(self.pop_size, h_dim, out_dim, device=device, dtype=self.dtype) * 0.04, requires_grad=False)
        self.b2 = nn.Parameter(torch.zeros(self.pop_size, 1, out_dim, device=device, dtype=self.dtype), requires_grad=False)

    def _normalize_single_weights(self, w1_s: torch.Tensor, b1_s: torch.Tensor, w2_s: torch.Tensor, b2_s: torch.Tensor):
        """Мягкое скалярное ограничение общей нормы Фробениуса (100% сохранение направлений всех 256 нейронов!)."""
        w1_f = w1_s.float()
        n1 = float(w1_f.norm().item())
        if n1 > 145.0:
            w1_f = w1_f * (145.0 / n1)

        w2_f = w2_s.float()
        n2 = float(w2_f.norm().item())
        if n2 > 45.0:
            w2_f = w2_f * (45.0 / n2)

        b1_f = torch.clamp(b1_s.float(), -2.0, 2.0)
        b2_f = torch.clamp(b2_s.float(), -3.0, 2.0)

        return (
            w1_f.to(dtype=self.dtype),
            b1_f.to(dtype=self.dtype),
            w2_f.to(dtype=self.dtype),
            b2_f.to(dtype=self.dtype),
        )

    def load_champion_into_population(self, ckpt_data: dict):
        if "backbone_state" in ckpt_data:
            self.backbone.load_state_dict(ckpt_data["backbone_state"], strict=False)
        w1_c = ckpt_data["w1"].to(device=self.device, dtype=self.dtype)
        b1_c = ckpt_data["b1"].to(device=self.device, dtype=self.dtype)
        w2_c = ckpt_data["w2"].to(device=self.device, dtype=self.dtype)
        b2_c = ckpt_data["b2"].to(device=self.device, dtype=self.dtype)

        if w2_c.shape[-1] == 8:
            col_map = [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7]
            w2_c = w2_c[:, col_map].clone()
            b2_c = b2_c[:, col_map].clone()

        with torch.no_grad():
            w1_n, b1_n, w2_n, b2_n = self._normalize_single_weights(w1_c, b1_c, w2_c, b2_c)
            self.w1.copy_(w1_n.unsqueeze(0).expand(self.pop_size, -1, -1))
            self.b1.copy_(b1_n.unsqueeze(0).expand(self.pop_size, -1, -1))
            self.w2.copy_(w2_n.unsqueeze(0).expand(self.pop_size, -1, -1))
            self.b2.copy_(b2_n.unsqueeze(0).expand(self.pop_size, -1, -1))
            # Группа 0 (среды 0..5) остается чистым эталоном, группы 1..1364 получают синхронные по 6 бюджетам мутации
            self._mutate_groups_inplace(1, self.n_groups, 0.030)

    def _mutate_groups_inplace(self, g0: int, g1: int, scale: float):
        """Мутирует группы кандидатов [g0..g1], назначая одинаковый шум всем 6 бюджетам внутри каждой группы."""
        B_buds = self.n_budgets
        chunk_g = 195  # 195 * 6 = 1170 сред за шаг
        noise_w1 = torch.empty((chunk_g, 1, self.in_dim, self.h_dim), device=self.device, dtype=self.dtype)
        for s_g in range(g0, g1, chunk_g):
            e_g = min(g1, s_g + chunk_g)
            ng = e_g - s_g
            nw = noise_w1[:ng]
            nw.normal_(0.0, scale)
            dst_w1 = self.w1[s_g * B_buds : e_g * B_buds].view(ng, B_buds, self.in_dim, self.h_dim)
            dst_w1.add_(nw.expand(ng, B_buds, self.in_dim, self.h_dim))
        del noise_w1

        ng_tot = g1 - g0
        nb1 = torch.randn((ng_tot, 1, 1, self.h_dim), device=self.device, dtype=self.dtype) * (scale * 0.5)
        self.b1[g0 * B_buds : g1 * B_buds].view(ng_tot, B_buds, 1, self.h_dim).add_(nb1.expand(ng_tot, B_buds, 1, self.h_dim))

        nw2 = torch.randn((ng_tot, 1, self.h_dim, self.out_dim), device=self.device, dtype=self.dtype) * scale
        self.w2[g0 * B_buds : g1 * B_buds].view(ng_tot, B_buds, self.h_dim, self.out_dim).add_(nw2.expand(ng_tot, B_buds, self.h_dim, self.out_dim))

        nb2 = torch.randn((ng_tot, 1, 1, self.out_dim), device=self.device, dtype=self.dtype) * (scale * 0.5)
        self.b2[g0 * B_buds : g1 * B_buds].view(ng_tot, B_buds, 1, self.out_dim).add_(nb2.expand(ng_tot, B_buds, 1, self.out_dim))

    def evolve_groups(self, group_scores: torch.Tensor, hof_state: dict, elite_frac: float = 0.03, progress: float = 0.0) -> int:
        """
        Эволюция на уровне портфельных групп (каждая группа оценена сразу на всех 6 бюджетах!).
        Группа #0 всегда получает веса Чемпиона Зала Славы (`hof_state`), чтобы элита никогда не деградировала.
        """
        B_buds = self.n_budgets
        n_elites = max(12, int(self.n_groups * elite_frac))  # ~41 элитная группа
        sorted_g = torch.argsort(group_scores, descending=True)
        elite_g = sorted_g[:n_elites]
        best_g = int(sorted_g[0].item())

        # Извлекаем веса элитных групп (берем 0-й бюджет каждой элитной группы, т.к. все 6 внутри группы идентичны)
        elite_env_idx = elite_g * B_buds
        w1_el = self.w1[elite_env_idx].clone()  # [n_elites, 324, 256]
        b1_el = self.b1[elite_env_idx].clone()  # [n_elites, 1, 256]
        w2_el = self.w2[elite_env_idx].clone()  # [n_elites, 256, 26]
        b2_el = self.b2[elite_env_idx].clone()  # [n_elites, 1, 26]

        # Защищаем каждую элиту от раздувания норм весов (Weight Norm Clamping)
        for ei in range(n_elites):
            w1_el[ei], b1_el[ei], w2_el[ei], b2_el[ei] = self._normalize_single_weights(
                w1_el[ei], b1_el[ei], w2_el[ei], b2_el[ei]
            )

        # Группа 0 всегда хранит актуального Чемпиона Зала Славы (Hall-of-Fame)
        w1_el[0].copy_(hof_state["w1"].to(self.device, self.dtype))
        b1_el[0].copy_(hof_state["b1"].to(self.device, self.dtype))
        w2_el[0].copy_(hof_state["w2"].to(self.device, self.dtype))
        b2_el[0].copy_(hof_state["b2"].to(self.device, self.dtype))

        # Записываем элитные группы без мутаций в первые n_elites групп
        self.w1[: n_elites * B_buds].view(n_elites, B_buds, self.in_dim, self.h_dim).copy_(
            w1_el.unsqueeze(1).expand(n_elites, B_buds, self.in_dim, self.h_dim)
        )
        self.b1[: n_elites * B_buds].view(n_elites, B_buds, 1, self.h_dim).copy_(
            b1_el.unsqueeze(1).expand(n_elites, B_buds, 1, self.h_dim)
        )
        self.w2[: n_elites * B_buds].view(n_elites, B_buds, self.h_dim, self.out_dim).copy_(
            w2_el.unsqueeze(1).expand(n_elites, B_buds, self.h_dim, self.out_dim)
        )
        self.b2[: n_elites * B_buds].view(n_elites, B_buds, 1, self.out_dim).copy_(
            b2_el.unsqueeze(1).expand(n_elites, B_buds, 1, self.out_dim)
        )

        # Генерируем дочерние группы (n_elites .. n_groups) с синхронным шумом внутри каждой 6-бюджетной группы
        n_child_g = self.n_groups - n_elites
        parent_g = torch.randint(0, n_elites, (n_child_g,), device=self.device)
        decay = max(0.30, 1.0 - 0.65 * float(progress))
        sigma_g = torch.where(
            torch.rand(n_child_g, 1, 1, 1, device=self.device) < 0.70,
            torch.full((n_child_g, 1, 1, 1), 0.022 * decay, device=self.device, dtype=self.dtype),
            torch.full((n_child_g, 1, 1, 1), 0.055 * decay, device=self.device, dtype=self.dtype),
        )

        chunk_g = 195
        buf_w1 = torch.empty((chunk_g, 1, self.in_dim, self.h_dim), device=self.device, dtype=self.dtype)
        for c0 in range(0, n_child_g, chunk_g):
            c1 = min(n_child_g, c0 + chunk_g)
            ng = c1 - c0
            g_start = n_elites + c0
            g_end = n_elites + c1
            p_idx = parent_g[c0:c1]
            sig_c = sigma_g[c0:c1]

            dst_w1 = self.w1[g_start * B_buds : g_end * B_buds].view(ng, B_buds, self.in_dim, self.h_dim)
            dst_w1.copy_(w1_el[p_idx].unsqueeze(1).expand(ng, B_buds, self.in_dim, self.h_dim))
            bw = buf_w1[:ng]
            bw.normal_(0.0, 1.0).mul_(sig_c)
            dst_w1.add_(bw.expand(ng, B_buds, self.in_dim, self.h_dim))

        del buf_w1, w1_el

        dst_b1 = self.b1[n_elites * B_buds :].view(n_child_g, B_buds, 1, self.h_dim)
        dst_b1.copy_(b1_el[parent_g].unsqueeze(1).expand(n_child_g, B_buds, 1, self.h_dim))
        nb1 = torch.randn((n_child_g, 1, 1, self.h_dim), device=self.device, dtype=self.dtype) * (sigma_g * 0.5)
        dst_b1.add_(nb1.expand(n_child_g, B_buds, 1, self.h_dim))

        dst_w2 = self.w2[n_elites * B_buds :].view(n_child_g, B_buds, self.h_dim, self.out_dim)
        dst_w2.copy_(w2_el[parent_g].unsqueeze(1).expand(n_child_g, B_buds, self.h_dim, self.out_dim))
        nw2 = torch.randn((n_child_g, 1, self.h_dim, self.out_dim), device=self.device, dtype=self.dtype) * sigma_g
        dst_w2.add_(nw2.expand(n_child_g, B_buds, self.h_dim, self.out_dim))

        dst_b2 = self.b2[n_elites * B_buds :].view(n_child_g, B_buds, 1, self.out_dim)
        dst_b2.copy_(b2_el[parent_g].unsqueeze(1).expand(n_child_g, B_buds, 1, self.out_dim))
        nb2 = torch.randn((n_child_g, 1, 1, self.out_dim), device=self.device, dtype=self.dtype) * (sigma_g * 0.5)
        dst_b2.add_(nb2.expand(n_child_g, B_buds, 1, self.out_dim))

        return best_g


def run_lesson_6(iterations: int = 1000, n_groups: int = 1365):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_budgets = 6
    pop_size = n_groups * n_budgets  # 8190

    log_msg("=" * 98)
    log_msg(f"🚀 УРОК 6 v3: СИНХРОННЫЙ 6-БЮДЖЕТНЫЙ ЭКЗАМЕН (1365 ГРУПП × 6 БЮДЖЕТОВ = {pop_size}) + NORM CLAMP")
    log_msg("=" * 98)

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

    summary_path = SAVE_DIR / "lesson_6_summary.json"
    history = []
    start_iter = 1
    latest_ckpt = SAVE_DIR / "lesson_6_champ_latest.pt"
    # Предпочитаем стабильный чекпоинт iter_0100.pt / latest для инициализации Hall-of-Fame
    if latest_ckpt.exists() and summary_path.exists():
        try:
            history = json.loads(summary_path.read_text(encoding="utf-8"))
            if history:
                start_iter = int(history[-1]["iteration"]) + 1
                ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
                policy.load_champion_into_population(ck)
                log_msg(f"🔄 Продолжение Урока 6 v3 (Синхронный 6-Бюджетный Отбор + Norm Clamp) с итерации {start_iter}/{iterations}...")
        except Exception as e:
            log_msg(f"⚠️ Ошибка чтения чекпоинта Урока 6: {e}")

    # Инициализируем Hall-of-Fame из нормализованной Группы #0
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
    # Каждая группа g (0..1364) получает ровно все 6 бюджетов [95.5, 1000, 3000, 5000, 10000, 20000]!
    env_budgets = budget_tiers.unsqueeze(0).expand(n_groups, n_budgets).reshape(pop_size)

    rng = np.random.default_rng(202606)
    win_len = 900  # 2 окна по 900с = 1,800 секунд (30 минут реального рынка)
    for _skip in range(1, start_iter):
        _ = rng.integers(100, slice_len - win_len - 100, size=2)

    for it in range(start_iter, iterations + 1):
        t0 = time.time()

        total_pnl_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_real_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_paper_dd_usd = torch.zeros(pop_size, dtype=torch.float32, device=device)
        max_hold_s = torch.zeros(pop_size, dtype=torch.float32, device=device)
        cycles = torch.zeros(pop_size, dtype=torch.long, device=device)
        prof_cycles = torch.zeros(pop_size, dtype=torch.long, device=device)
        evasion_exits = torch.zeros(pop_size, dtype=torch.float32, device=device)

        vec_env.n_trades.zero_()
        vec_env.partial_fills.zero_()
        vec_env.multi_orders_used.zero_()
        vec_env.frontrun_hits.zero_()
        vec_env.sell_cycles.zero_()
        vec_env.prof_sell_cycles.zero_()

        w_starts = rng.integers(100, slice_len - win_len - 100, size=2)
        with torch.no_grad():
            for w_start in w_starts:
                vec_env.reset_with_budgets(env_budgets)
                h_states = backbone.init_hidden(batch_size=pop_size, device=device)
                prev_pos = vec_env.pos_qty.clone()
                cur_hold = torch.zeros(pop_size, dtype=torch.float32, device=device)
                peak_real_eq = env_budgets.clone()
                peak_paper_eq = env_budgets.clone()

                for s in range(win_len - 1):
                    cur_t = int(w_start + s)
                    _, flow, _, multi_6d = vec_env.get_observations(cur_t_scalar=cur_t)
                    lob_r = lob_repr_all[cur_t:cur_t + 1].expand(pop_size, -1)
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

                    # 1. Фактический капитал по закрытым операциям (Realized Equity = Cash + PosCostUSD)
                    real_eq = cur_cash + vec_env.pos_cost_usd
                    peak_real_eq = torch.maximum(peak_real_eq, real_eq)
                    max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - real_eq)

                    # 2. Бумажный (плавающий) капитал (Paper Equity = Cash + PosQty * b1)
                    paper_eq = cur_cash + cur_pos * b1
                    peak_paper_eq = torch.maximum(peak_paper_eq, paper_eq)
                    max_paper_dd_usd = torch.maximum(max_paper_dd_usd, peak_paper_eq - paper_eq)

                    if vec_env.is_bleed_cpu[cur_t]:
                        evasion_exits = evasion_exits + (cur_pos < prev_pos - 1e-4).float()

                    prev_pos.copy_(cur_pos)

                end_t = int(w_start + win_len - 1)
                end_b1 = torch.clamp(vec_env.bids[end_t, 0], min=tick_sz)
                rem = vec_env.pos_qty > 1e-6
                # Если на границе окна позиция в рабочей сетке (end_b1 не провалился глубже min_buy_px - 2 тика),
                # не штрафуем бота искусственным рыночным сбросом на таймере, а возвращаем себестоимость рабочего остатка;
                # если же бот допустил глубокий провал > 2 тиков под нижний транш — оцениваем остаток по рынку end_b1!
                deep_underwater = rem & (end_b1 < (vec_env.min_buy_px - 2.0 * tick_sz))
                liq_val = torch.where(
                    deep_underwater,
                    vec_env.pos_qty * end_b1 * (1.0 - vec_env.maker_fee),
                    vec_env.pos_cost_usd,
                )
                vec_env.cash = torch.where(rem, vec_env.cash + liq_val, vec_env.cash)
                vec_env.pos_qty = torch.where(rem, torch.zeros_like(vec_env.pos_qty), vec_env.pos_qty)
                vec_env.pos_cost_usd = torch.where(rem, torch.zeros_like(vec_env.pos_cost_usd), vec_env.pos_cost_usd)
                max_real_dd_usd = torch.maximum(max_real_dd_usd, peak_real_eq - vec_env.cash)

                total_pnl_usd += (vec_env.cash - env_budgets)

            cycles.copy_(vec_env.sell_cycles)
            prof_cycles.copy_(vec_env.prof_sell_cycles)

            pnl_pct = 100.0 * total_pnl_usd / env_budgets
            real_dd_pct = 100.0 * max_real_dd_usd / env_budgets
            paper_dd_pct = 100.0 * max_paper_dd_usd / env_budgets
            loss_cycles = torch.clamp(cycles - prof_cycles, min=0).float()

            # Целевая функция: максимизация реализованной прибыли при минимальной ФАКТИЧЕСКОЙ просадке (real_dd_pct)!
            # Бумажная просадка (paper_dd_pct) внутри рабочей сетки (до 0.50%) не штрафуется вообще!
            env_scores = (
                pnl_pct * 1150.0
                + torch.clamp(total_pnl_usd * 7.5, min=-200.0, max=380.0)
                - real_dd_pct * 1500.0
                - torch.clamp(paper_dd_pct - 0.50, min=0.0) * 180.0
                + torch.clamp(prof_cycles.float(), max=35.0) * 22.0
                - loss_cycles * 40.0
                + torch.clamp(vec_env.partial_fills.float(), max=25.0) * 4.5
                + torch.clamp(evasion_exits, max=10.0) * 4.0
                - torch.clamp(max_hold_s - 300.0, min=0.0) * 0.15
            )
            env_scores = torch.where(cycles == 0, env_scores - 450.0, env_scores)
            env_scores = torch.where(total_pnl_usd <= 0.0, env_scores - 200.0, env_scores)

            # Агрегируем метрики по каждой группе g (по всем 6 бюджетам сразу!)
            g_scores_mat = env_scores.view(n_groups, n_budgets)
            g_pnl_pct_mat = pnl_pct.view(n_groups, n_budgets)
            g_pnl_usd_mat = total_pnl_usd.view(n_groups, n_budgets)
            g_dd_pct_mat = real_dd_pct.view(n_groups, n_budgets)
            g_dd_usd_mat = max_real_dd_usd.view(n_groups, n_budgets)
            g_pdd_pct_mat = paper_dd_pct.view(n_groups, n_budgets)
            g_cyc_mat = cycles.view(n_groups, n_budgets)
            g_pcyc_mat = prof_cycles.view(n_groups, n_budgets)
            g_pf_mat = vec_env.partial_fills.view(n_groups, n_budgets)
            g_mo_mat = vec_env.multi_orders_used.view(n_groups, n_budgets)
            g_hold_mat = max_hold_s.view(n_groups, n_budgets)

            g_prof_buds = (g_pnl_usd_mat > 0.0).long().sum(dim=1)  # сколько из 6 бюджетов закрыты в плюс!
            group_scores = g_scores_mat.mean(dim=1) + g_prof_buds.float() * 85.0

            sorted_g = torch.argsort(group_scores, descending=True)
            best_g = int(sorted_g[0].item())

            b_score = float(group_scores[best_g].item())
            b_pnl_pct = float(g_pnl_pct_mat[best_g].mean().item())
            b_pnl_usd = float(g_pnl_usd_mat[best_g].sum().item())  # суммарный $ PnL по портфелю из 6 бюджетов
            b_dd_pct = float(g_dd_pct_mat[best_g].mean().item())
            b_dd_usd = float(g_dd_usd_mat[best_g].mean().item())
            b_pdd_pct = float(g_pdd_pct_mat[best_g].mean().item())
            b_prof_buds = int(g_prof_buds[best_g].item())
            b_cyc = int(g_cyc_mat[best_g].float().mean().item())
            b_prof = int(g_pcyc_mat[best_g].float().mean().item())
            b_pf = int(g_pf_mat[best_g].float().mean().item())
            b_mo = int(g_mo_mat[best_g].float().mean().item())
            b_hold = float(g_hold_mat[best_g].mean().item())

            # Обновляем Hall-of-Fame только если лучшая группа показала чистый плюс минимум на 4 из 6 бюджетов
            # и превзошла действующего чемпиона (Группу #0) на этом же окне
            hof_updated = False
            if b_pnl_pct > 0.0 and b_prof_buds >= 4 and b_score >= float(group_scores[0].item()):
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
                "pnl_usd": round(b_pnl_usd, 4),
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

            # Скользящее среднее за последние 20 итераций (для прозрачной оценки тренда)
            tail20 = history[-20:]
            r20_pnl = float(np.mean([x["pnl_pct"] for x in tail20]))
            r20_dd = float(np.mean([x["max_dd_pct"] for x in tail20]))
            r20_wr = 100.0 * sum(x["prof_cycles"] for x in tail20) / max(1, sum(x["cycles"] for x in tail20))
            dt = time.time() - t0
            star = "★" if hof_updated else "·"

            log_msg(
                f"🌐 [{it}/{iterations}]{star} 6-Buds({b_prof_buds}/6+) | "
                f"MeanPnL: {b_pnl_pct:+.2f}% (Sum ${b_pnl_usd:+.2f}) | RealDD: {b_dd_pct:.2f}% (Paper:{b_pdd_pct:.2f}%) | "
                f"Cyc: {b_cyc}/{b_prof} (Part:{b_pf}) | 📈Roll20: {r20_pnl:+.2f}% (RealDD {r20_dd:.2f}%, WR {r20_wr:.1f}%) ({dt:.1f}с)"
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
                torch.save(ckpt_payload, SAVE_DIR / f"lesson_6_champ_iter_{it:04d}.pt")

    log_msg("🏆 УРОК 6 v3 (24-SLOT MULTI-BUDGET SYNCHRONOUS GRID & LIVE SELF-IMPACT) УСПЕШНО ЗАВЕРШЕН!")


if __name__ == "__main__":
    run_lesson_6(iterations=1000, n_groups=1365)
