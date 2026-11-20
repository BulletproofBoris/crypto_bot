#!/usr/bin/env python3
"""
_tools/lesson_25_dual_swarm_regulator_trainer.py
=============================================================================
Урок 25: Двухроевой Стратег (480D = SafeTrade Swarm + BigONE Swarm) +
Послотовый Нейро-Регулятор Очереди (`Shared Per-Slot Micro-Regulator`).

Двухфазный протокол обучения без утечки данных:
  - Строгий сплит (1-в-1 с Уроком 24):
      * TRAIN <= 2026-11-11 22:00 MSK (- 3000s эмбарго)
      * VAL_CASCADE [2026-11-11 22:00 -> 2026-11-12 20:30 MSK] (краш $1.40->$1.12->$1.32)
      * BLIND_OVERNIGHT >= 2026-11-12 20:30 MSK (100% запечатанный слепой тест!)
  - Warm-Start из живого чемпиона `L24_SOUP_11_47` (`release_champion_v1.pt`):
      * Ствол 444D переносится 1-в-1, новые +36D каналы роя BigONE `[444:480]` инициализируются
        малым шумом, голова 35D наследует пирамиду Bid/Ask/Panic из L24, а Регулятор (16->32->2)
        инициализируется в спокойном режиме сохранения очереди `HOLD_QUEUE`.
  - Фаза 1 (`1 .. phase1_iters`, по умолч. `60`): заморожены `ResSwiGLU` блоки ствола (`wr1, wr2`),
    быстро адаптируются новые +36D каналы `bo_signals_36d`, развязанная голова `E_bid + Softmax(π)`
    и `Shared Per-Slot Micro-Regulator` для устранения «тремора» заявок.
  - Фаза 2 (`phase1_iters+1 .. iterations`, по умолч. `61 .. 240`): полная совместная коэволюция
    всех слоёв 480D Стратега + Регулятора на 40-минутных окнах (`win_len = 2400s`) с Pareto-отбором
    и автоматической сборкой `Model Soup` топ-чекпоинтов.
=============================================================================
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
from _tools.lesson_23_pump_capture_penny_l1_trainer import (
    _build_roll_sell_30s,
    compute_pareto_group_fitness,
)
from _tools.lesson_24_cascade_survival_trainer import _classify_2400s_windows
from _tools.fused_l25_regulator_tick import (
    MKT_IDX_L25,
    AG_IDX_L25,
    precompute_window_market_z370,
    fused_l25_regulator_tick,
    evaluate_l25_dual_basket,
)

CACHE_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
RELEASE_CHAMP_PATH = BASE_DIR / "models/prl/release_champion_v1.pt"
SAVE_DIR = BASE_DIR / "models/prl/lesson_25_dual_swarm_regulator"
LATEST_CHAMP_PATH = SAVE_DIR / "lesson_25_champ_latest.pt"
RESUME_CHAMP_PATH = SAVE_DIR / "lesson_25_champ_resume.pt"
SOUP_CHAMP_PATH = SAVE_DIR / "l25_soup_top2.pt"
LOG_PATH = BASE_DIR / "data/logs/lesson_25_train.log"

MSK_TZ = timezone(timedelta(hours=3))
VAL_START_TS = datetime(2026, 11, 11, 22, 0, tzinfo=MSK_TZ).timestamp()
BLIND_START_TS = datetime(2026, 11, 12, 20, 30, tzinfo=MSK_TZ).timestamp()

# 18 семантических входных блоков 480D (16 блоков L24 + 2 новых блока роя 1-го порядка BigONE):
SEMANTIC_BLOCKS_L25 = [
    ("Z_Backbone_GRU", 0, 256, 0.45),
    ("Guard_Risk_Heads", 256, 260, 0.75),
    ("Short_Triad_Swarm_ST", 260, 278, 0.95),
    ("CrossVenue_5s_LeadLag", 278, 288, 1.95),
    ("Macro_Swarm_ST_60_120s", 288, 306, 0.55),
    ("Carousel_Hazard_Realism", 306, 312, 0.75),
    ("CrossVenue_FirstOrder_6D", 312, 318, 2.05),
    ("LiqRatio_LIFO_Edge", 318, 324, 1.55),
    ("Active_24Slot_Weights", 324, 348, 2.15),
    ("Swarm_5s_Velocity", 348, 356, 1.65),
    ("Active_24Slot_Distances", 356, 380, 1.25),
    ("Fast_5s_LOB_Impulse", 380, 388, 1.75),
    ("Direct_Acct_State_8D", 388, 396, 2.10),
    ("FIFO_Queue_Ahead_8D", 396, 404, 2.25),
    ("Policy_Micro_Memory_16D", 404, 420, 2.05),
    ("PerSlot_FIFO_Progress_24D", 420, 444, 2.45),
    ("BigONE_Short_Swarm_18D", 444, 462, 2.10),
    ("BigONE_Macro_Swarm_18D", 462, 480, 1.65),
]
N_BLOCKS_L25 = len(SEMANTIC_BLOCKS_L25)


def log_msg(msg: str) -> None:
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts_str}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expand_block_gates_to_480(g_logits: torch.Tensor) -> torch.Tensor:
    gates_b = 2.0 * torch.sigmoid(g_logits.float())
    parts = [gates_b[..., idx : idx + 1].expand(*gates_b.shape[:-1], e - s) for idx, (_, s, e, _) in enumerate(SEMANTIC_BLOCKS_L25)]
    return torch.cat(parts, dim=-1)


def _normalize_weights_l25(state: dict, dtype: torch.dtype) -> dict:
    out = {}
    w1_f = state["w1"].float()
    w1_base = w1_f[..., :256]
    n1 = torch.linalg.vector_norm(w1_base, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    w1_base = w1_base * torch.clamp(155.0 / n1, max=1.0)
    w1_gate = w1_f[..., 256:384]
    ng = torch.linalg.vector_norm(w1_gate, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    w1_gate = w1_gate * torch.clamp(92.0 / ng, max=1.0)
    out["w1"] = torch.cat([w1_base, w1_gate], dim=-1).to(dtype)
    out["b1"] = torch.clamp(state["b1"].float(), -2.0, 2.0).to(dtype)

    for k, cap in (("w_res1", 65.0), ("w_res2", 40.0), ("w2", 52.0), ("w_rec", 30.0), ("w_reg1", 18.0), ("w_reg2", 12.0)):
        wf = state[k].float()
        nw = torch.linalg.vector_norm(wf, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        out[k] = (wf * torch.clamp(cap / nw, max=1.0)).to(dtype)

    out["b_res1"] = torch.clamp(state["b_res1"].float(), -2.0, 2.0).to(dtype)
    out["b2"] = torch.clamp(state["b2"].float(), -3.5, 2.5).to(dtype)
    out["b_rec"] = torch.clamp(state["b_rec"].float(), -2.5, 2.5).to(dtype)
    out["b_reg1"] = torch.clamp(state["b_reg1"].float(), -2.5, 2.5).to(dtype)
    out["b_reg2"] = torch.clamp(state["b_reg2"].float(), -3.5, 3.0).to(dtype)
    return out


class DualSwarmRegulatorPolicyL25:
    """Популяционная 480D политика Урока 25: Макро-Стратег (35D) + Послотовый Нейро-Регулятор (16->32->2)."""

    PARAM_KEYS = (
        "w1", "b1", "w_res1", "b_res1", "w_res2", "w2", "b2",
        "w_rec", "b_rec", "w_reg1", "b_reg1", "w_reg2", "b_reg2",
    )

    def __init__(self, backbone: MultiTaskLOBNet, n_groups: int = 1024, n_budgets: int = 6, device: torch.device = torch.device("cuda")):
        self.backbone = backbone
        self.n_groups = n_groups
        self.n_budgets = n_budgets
        self.pop_size = n_groups * n_budgets
        self.in_dim = 480
        self.h_dim = 256
        self.h1_full_dim = 384
        self.out_dim = 35
        self.reg_in_dim = 16
        self.reg_h_dim = 32
        self.device = device
        self.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        self.w1 = torch.empty((n_groups, 480, 384), dtype=self.dtype, device=device)
        self.b1 = torch.empty((n_groups, 1, 384), dtype=self.dtype, device=device)
        self.w_res1 = torch.empty((n_groups, 256, 128), dtype=self.dtype, device=device)
        self.b_res1 = torch.empty((n_groups, 1, 128), dtype=self.dtype, device=device)
        self.w_res2 = torch.empty((n_groups, 64, 256), dtype=self.dtype, device=device)
        self.w2 = torch.empty((n_groups, 256, 35), dtype=self.dtype, device=device)
        self.b2 = torch.empty((n_groups, 1, 35), dtype=self.dtype, device=device)
        self.w_rec = torch.empty((n_groups, 256, 32), dtype=self.dtype, device=device)
        self.b_rec = torch.empty((n_groups, 1, 32), dtype=self.dtype, device=device)

        self.w_reg1 = torch.empty((n_groups, 16, 32), dtype=self.dtype, device=device)
        self.b_reg1 = torch.empty((n_groups, 1, 32), dtype=self.dtype, device=device)
        self.w_reg2 = torch.empty((n_groups, 32, 2), dtype=self.dtype, device=device)
        self.b_reg2 = torch.empty((n_groups, 1, 2), dtype=self.dtype, device=device)
        self.g_logits = torch.zeros((n_groups, N_BLOCKS_L25), dtype=torch.float32, device=device)

    @staticmethod
    def build_calm_initial_regulator(dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """
        Детерминированный спокойный Warm-Start Послотового Микро-Регулятора (16 -> 32 -> 2):
        - Гейт постановки `g_place` открыт при `targ_val > 0.025` и блокируется при высокой токсичности на переднем крае.
        - Гейт отмены `g_cancel` закрыт по умолчанию (`b_cancel = -1.35`), открывается ТОЛЬКО при:
            1) обнулении целевого объёма (`targ_val < 0.01`),
            2) уходе цены на >= 2 тика (`|drift| >= 0.40`),
            3) всплеске токсичности Стратега на переднем слоте,
          и сильно тормозится накопленным прогрессом очереди `qprog_s` (`feat[6]`)!
        """
        torch.manual_seed(202625)
        w_reg1 = torch.randn((16, 32), dtype=torch.float32) * 0.05
        b_reg1 = torch.zeros((1, 32), dtype=torch.float32)
        w_reg2 = torch.randn((32, 2), dtype=torch.float32) * 0.04
        b_reg2 = torch.tensor([[0.65, -1.35]], dtype=torch.float32)

        # Юнит 0: целевой объём слота есть (`targ_val = feat[3]`) -> разрешить постановку, не отменять
        w_reg1[3, 0] = 6.0
        b_reg1[0, 0] = -0.10
        w_reg2[0, 0] = 2.2
        w_reg2[0, 1] = -2.2

        # Юнит 1: слот активен (`feat[1]=1`), но целевой объём обнулён (`feat[3]=0`) -> снять слот
        w_reg1[1, 1] = 3.0
        w_reg1[3, 1] = -10.0
        b_reg1[0, 1] = -0.50
        w_reg2[1, 1] = 2.6

        # Юниты 2 и 3: сдвиг цены вверх/вниз >= 2 тиков (`|drift| = |feat[5]| >= 0.38`) при зрелом ордере -> переставить
        w_reg1[5, 2] = 5.0
        w_reg1[8, 2] = 0.8
        b_reg1[0, 2] = -1.85
        w_reg2[2, 1] = 2.8

        w_reg1[5, 3] = -5.0
        w_reg1[8, 3] = 0.8
        b_reg1[0, 3] = -1.85
        w_reg2[3, 1] = 2.8

        # Юнит 4: защита накопленного FIFO-прогресса (`qprog = feat[6]`) -> удерживать очередь!
        w_reg1[6, 4] = 4.0
        b_reg1[0, 4] = -0.20
        w_reg2[4, 1] = -1.8

        # Юнит 5: сигнал токсичности Стратега (`ctx_0 = feat[12]`) на передних слотах (`is_front = feat[10]`) -> снять передний край
        w_reg1[12, 5] = 3.2
        w_reg1[10, 5] = 1.6
        b_reg1[0, 5] = -2.60
        w_reg2[5, 0] = -2.2
        w_reg2[5, 1] = 2.8

        # Юнит 6: крупный недобор объёма (`rel_delta = feat[4] > 0.75`) ТОЛЬКО при пустой очереди (`qprog` мал) и наличии свободного кэша (`feat[11]`)
        w_reg1[4, 6] = 2.4
        w_reg1[11, 6] = 1.8
        w_reg1[6, 6] = -3.5
        w_reg1[8, 6] = 0.9
        b_reg1[0, 6] = -3.40
        w_reg2[6, 1] = 1.9

        return {
            "w_reg1": w_reg1.to(dtype),
            "b_reg1": b_reg1.to(dtype),
            "w_reg2": w_reg2.to(dtype),
            "b_reg2": b_reg2.to(dtype),
        }

    def load_from_checkpoint(self, ck: dict) -> tuple[dict, torch.Tensor]:
        if "backbone_state" in ck:
            self.backbone.load_state_dict(ck["backbone_state"], strict=False)
        is_l25 = int(ck.get("lesson", 0)) >= 25 and int(ck["w2"].shape[-1]) == 35
        if is_l25:
            state = {
                "w1": ck["w1"].to(dtype=self.dtype, device=self.device),
                "b1": ck["b1"].view(1, 384).to(dtype=self.dtype, device=self.device),
                "w_res1": ck["w_res1"].to(dtype=self.dtype, device=self.device),
                "b_res1": ck["b_res1"].view(1, 128).to(dtype=self.dtype, device=self.device),
                "w_res2": ck["w_res2"].to(dtype=self.dtype, device=self.device),
                "w2": ck["w2"].to(dtype=self.dtype, device=self.device),
                "b2": ck["b2"].view(1, 35).to(dtype=self.dtype, device=self.device),
                "w_rec": ck["w_rec"].to(dtype=self.dtype, device=self.device),
                "b_rec": ck["b_rec"].view(1, 32).to(dtype=self.dtype, device=self.device),
                "w_reg1": ck["w_reg1"].to(dtype=self.dtype, device=self.device),
                "b_reg1": ck["b_reg1"].view(1, 32).to(dtype=self.dtype, device=self.device),
                "w_reg2": ck["w_reg2"].to(dtype=self.dtype, device=self.device),
                "b_reg2": ck["b_reg2"].view(1, 2).to(dtype=self.dtype, device=self.device),
            }
            # В сохранённом чекпоинте L25 матрица `w1` УЖЕ содержит вплетённые гейты `2*sigmoid(gate_logit)`:
            # стартуем новый виток с нейтрального `gl = 0` (`2*sigmoid(0) == 1.0`) и БЕЗ повторного клиппинга `w1`!
            gl = torch.zeros(N_BLOCKS_L25, dtype=torch.float32)
            cpu_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            self.sample_population(cpu_state, gl, phase=1, sigma_scale=1.0)
            return cpu_state, gl
        else:
            # Warm-Start из чемпиона Урока 24 (444D -> 26D) в 480D -> 35D + Per-Slot Regulator:
            w1_old = ck["w1"].float()
            torch.manual_seed(2026251)
            w1_bo36 = torch.randn((36, 384), dtype=torch.float32) * 0.008
            w1_480 = torch.cat([w1_old, w1_bo36], dim=0)

            w2_old = ck["w2"].float()  # [256, 26]
            b2_old = ck["b2"].float().view(-1)  # [26]
            w2_new = torch.zeros((256, 35), dtype=torch.float32)
            b2_new = torch.zeros(35, dtype=torch.float32)

            # [0] E_bid: наследует суммарное желание набирать глубокую пирамиду минус сигнал отмены [24]
            w2_new[:, 0] = 0.85 * w2_old[:, 2:10].mean(dim=1) - 0.55 * w2_old[:, 24]
            b2_new[0] = 0.85 + 0.50 * float(b2_old[2:10].mean().item()) - 0.35 * float(b2_old[24].item())
            # [1:13] логиты Softmax(π_bid) и [13:25] логиты Softmax(π_ask)
            w2_new[:, 1:13] = w2_old[:, 0:12] * 0.85
            b2_new[1:13] = b2_old[0:12] * 0.85
            w2_new[:, 13:25] = w2_old[:, 12:24] * 0.85
            b2_new[13:25] = b2_old[12:24] * 0.85
            # [25] Panic Taker Exit 1-в-1 из L24
            w2_new[:, 25] = w2_old[:, 25]
            b2_new[25] = b2_old[25]
            # [26] Контекст токсичности переднего края для Микро-Регулятора <- голова #24 L24
            w2_new[:, 26] = w2_old[:, 24]
            b2_new[26] = b2_old[24] - 0.80
            # [34] Температура Softmax τ ≈ 0.85 на старте (позволяет и концентрировать, и строить лестницу)
            b2_new[34] = -0.75

            reg_init = self.build_calm_initial_regulator(self.dtype)
            state = {
                "w1": w1_480.to(dtype=self.dtype, device=self.device),
                "b1": ck["b1"].view(1, 384).to(dtype=self.dtype, device=self.device),
                "w_res1": ck["w_res1"].to(dtype=self.dtype, device=self.device),
                "b_res1": ck["b_res1"].view(1, 128).to(dtype=self.dtype, device=self.device),
                "w_res2": ck["w_res2"].to(dtype=self.dtype, device=self.device),
                "w2": w2_new.to(dtype=self.dtype, device=self.device),
                "b2": b2_new.view(1, 35).to(dtype=self.dtype, device=self.device),
                "w_rec": ck["w_rec"].to(dtype=self.dtype, device=self.device),
                "b_rec": ck["b_rec"].view(1, 32).to(dtype=self.dtype, device=self.device),
                "w_reg1": reg_init["w_reg1"].to(self.device),
                "b_reg1": reg_init["b_reg1"].to(self.device),
                "w_reg2": reg_init["w_reg2"].to(self.device),
                "b_reg2": reg_init["b_reg2"].to(self.device),
            }
            gl = torch.zeros(N_BLOCKS_L25, dtype=torch.float32)

        state = _normalize_weights_l25(state, self.dtype)
        cpu_state = {k: v.detach().cpu().clone() for k, v in state.items()}
        self.sample_population(cpu_state, gl, phase=1, sigma_scale=1.0)
        return cpu_state, gl

    def fold_gate_into_state(self, cpu_state: dict, g_logit: torch.Tensor) -> dict:
        g480 = expand_block_gates_to_480(g_logit.view(1, -1)).view(480, 1)
        out = {k: v.clone() for k, v in cpu_state.items()}
        out["w1"] = (cpu_state["w1"].float() * g480).to(self.dtype)
        return out

    def effective_group_w1(self) -> torch.Tensor:
        g480 = expand_block_gates_to_480(self.g_logits).unsqueeze(-1).to(self.dtype)
        return (self.w1 * g480).contiguous()

    @torch.no_grad()
    def sample_population(self, center_cpu: dict, center_gl: torch.Tensor, phase: int, sigma_scale: float) -> None:
        """Антитетическое сэмплирование популяции с учётом Фазы 1 (заморожен ствол) и Фазы 2 (полная коэволюция)."""
        half = (self.n_groups - 2) // 2
        c_gpu = {k: v.to(device=self.device, dtype=torch.float32) for k, v in center_cpu.items()}
        gl_gpu = center_gl.to(device=self.device, dtype=torch.float32)

        # Группа 0 и 1: чистый центр без шума
        for k in self.PARAM_KEYS:
            getattr(self, k)[0:2].copy_(c_gpu[k].to(self.dtype).unsqueeze(0).expand(2, *c_gpu[k].shape))
        self.g_logits[0:2].copy_(gl_gpu.unsqueeze(0).expand(2, -1))

        # Масштабы шума по слоям: в Фазе 1 учим только bo36 [444:480], голову w2/b2 и Микро-Регулятор!
        s_trunk = 0.0 if phase == 1 else (0.014 * sigma_scale)
        s_bo36 = 0.032 * sigma_scale
        s_head = 0.030 * sigma_scale
        s_reg = 0.042 * sigma_scale
        s_rec = (0.010 if phase == 1 else 0.022) * sigma_scale

        def _antithetic_pair(shape, std_val):
            if std_val <= 1e-8:
                z = torch.zeros((half, *shape), dtype=torch.float32, device=self.device)
                return torch.cat([z, z], dim=0)
            eps = torch.randn((half, *shape), dtype=torch.float32, device=self.device) * std_val
            return torch.cat([eps, -eps], dim=0)

        dw1 = torch.zeros((2 * half, 480, 384), dtype=torch.float32, device=self.device)
        if s_trunk > 0.0:
            dw1[:, :444, :] = _antithetic_pair((444, 384), s_trunk)
        dw1[:, 444:480, :] = _antithetic_pair((36, 384), s_bo36)

        pert = {
            "w1": c_gpu["w1"].unsqueeze(0) + dw1,
            "b1": c_gpu["b1"].unsqueeze(0) + _antithetic_pair((1, 384), s_trunk),
            "w_res1": c_gpu["w_res1"].unsqueeze(0) + _antithetic_pair((256, 128), s_trunk),
            "b_res1": c_gpu["b_res1"].unsqueeze(0) + _antithetic_pair((1, 128), s_trunk),
            "w_res2": c_gpu["w_res2"].unsqueeze(0) + _antithetic_pair((64, 256), s_trunk),
            "w2": c_gpu["w2"].unsqueeze(0) + _antithetic_pair((256, 35), s_head),
            "b2": c_gpu["b2"].unsqueeze(0) + _antithetic_pair((1, 35), s_head * 1.2),
            "w_rec": c_gpu["w_rec"].unsqueeze(0) + _antithetic_pair((256, 32), s_rec),
            "b_rec": c_gpu["b_rec"].unsqueeze(0) + _antithetic_pair((1, 32), s_rec),
            "w_reg1": c_gpu["w_reg1"].unsqueeze(0) + _antithetic_pair((16, 32), s_reg),
            "b_reg1": c_gpu["b_reg1"].unsqueeze(0) + _antithetic_pair((1, 32), s_reg),
            "w_reg2": c_gpu["w_reg2"].unsqueeze(0) + _antithetic_pair((32, 2), s_reg),
            "b_reg2": c_gpu["b_reg2"].unsqueeze(0) + _antithetic_pair((1, 2), s_reg * 1.2),
        }
        pert_norm = _normalize_weights_l25(pert, self.dtype)
        for k in self.PARAM_KEYS:
            getattr(self, k)[2 : 2 + 2 * half].copy_(pert_norm[k])

        d_gl = _antithetic_pair((N_BLOCKS_L25,), (0.04 if phase == 1 else 0.08) * sigma_scale)
        if phase == 1:
            d_gl[:, :16] = 0.0
        self.g_logits[2 : 2 + 2 * half].copy_(torch.clamp(gl_gpu.unsqueeze(0) + d_gl, -2.5, 2.0))


def _save_l25_checkpoint(
    path: Path,
    base_ck: dict,
    folded_state: dict,
    gate_logit: torch.Tensor,
    it: int,
    phase: int,
    release_name: str,
    metrics: dict,
) -> None:
    ck_out = dict(base_ck)
    ck_out.update({
        "lesson": 25,
        "phase": int(phase),
        "arch_version": "l25_dual_swarm_regulator_480d",
        "iteration": int(it),
        "n_slots": 24,
        "in_dim": 480,
        "out_dim": 35,
        "use_cross_venue_v1": True,
        "use_bo_swarm_36d": True,
        "use_per_slot_regulator": True,
        "decoupled_market_backbone": True,
        "selective_front_cancel": True,
        "allow_scratch_stop_asks": False,
        "panic_thresh": 0.85,
        "release_name": release_name,
        "gate_logit": gate_logit.detach().cpu().clone(),
        "pareto_util_bp": float(metrics["pareto_util_bp"]),
        "clean_util_bp": float(metrics["clean_util_bp"]),
        "uw_util_bp": float(metrics["uw_util_bp"]),
        "clean_ev": float(metrics["clean_ev"]),
        "clean_cnl_per_fill": float(metrics.get("clean_cnl_per_fill", 0.0)),
        "clean_eta_queue_pct": float(metrics.get("clean_eta_queue_pct", 0.0)),
        "cascade_pct": float(metrics.get("clean_dump_pct", 0.0)),
    })
    for k in DualSwarmRegulatorPolicyL25.PARAM_KEYS:
        ck_out[k] = folded_state[k].detach().cpu().clone()
    torch.save(ck_out, path)


def run_lesson_25(
    iterations: int = 220,
    phase1_iters: int = 55,
    phase2_iters: int = 220,
    phase3_iters: int = 500,
    plateau_patience: int = 0,
    plateau_min_delta_bp: float = 0.30,
    n_groups: int = 1024,
    win_len: int = 2400,
    rng_seed: int = 2026250,
    allow_scratch_stop_asks: bool = False,
    panic_thresh: float = 0.85,
) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_budgets = 6
    pop_size = n_groups * n_budgets

    log_msg("=" * 118)
    p3_iters = max(0, min(iterations, phase3_iters) - phase2_iters)
    p4_iters = max(0, iterations - phase3_iters)
    log_msg(
        f"🚀 УРОК 25 (480D DUAL-SWARM STRATEGIST + PER-SLOT MICRO-REGULATOR | WIN={win_len}s): "
        f"{iterations} ИТЕР (P1={phase1_iters}, P2={max(0, min(iterations, phase2_iters) - phase1_iters)}, "
        f"P3={p3_iters}, P4_PLATEAU={p4_iters} [patience={plateau_patience}, Δ>={plateau_min_delta_bp:.2f}bp]) | "
        f"10 ОКОН x {pop_size} = {10 * pop_size:,} СРЕД"
    )
    log_msg("=" * 118)

    raw = np.load(CACHE_PATH, allow_pickle=True)
    assert "bo_signals_36d" in raw.files, "Сначала соберите `bo_signals_36d` через `_tools/build_bo_swarm_36d_cache.py`!"
    n_total = len(raw["bid_p"])
    tick_sz = float(raw["tick_size"]) if "tick_size" in raw.files else 0.01

    slice_len = min(1200000, n_total)
    s0 = n_total - slice_len
    ts_sub = np.array(raw["ts"][s0:], dtype=np.float64, copy=True)

    sub = {
        k: np.array(raw[k][s0:], dtype=np.float32, copy=True)
        for k in ["bid_p", "ask_p", "bid_q", "ask_q", "buy_vol", "sell_vol", "y_sweep_12s", "y_bleed_60s", "y_crash_300s", "y_exit_30s"]
    }
    sub["tick_size"] = tick_sz
    v7_raw_t = torch.from_numpy(np.array(raw["v7_signals_58d"][s0:], dtype=np.float32, copy=True)).to(device)
    bo36_t = torch.from_numpy(np.array(raw["bo_signals_36d"][s0:], dtype=np.float32, copy=True)).to(device)
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

    backbone = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    policy = DualSwarmRegulatorPolicyL25(backbone=backbone, n_groups=n_groups, n_budgets=n_budgets, device=device)
    v7_all = v7_all.to(dtype=policy.dtype)
    bo36_all = bo36_t.to(dtype=policy.dtype)
    del bo36_t

    summary_path = SAVE_DIR / "lesson_25_summary.json"
    history = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    seed_path = RESUME_CHAMP_PATH if RESUME_CHAMP_PATH.exists() else (LATEST_CHAMP_PATH if LATEST_CHAMP_PATH.exists() else RELEASE_CHAMP_PATH)
    base_ck = torch.load(seed_path, map_location="cpu", weights_only=False)
    hof_state, hof_g_logit = policy.load_from_checkpoint(base_ck)

    start_iter = (int(base_ck.get("iteration", 0)) + 1) if int(base_ck.get("lesson", 24)) == 25 else 1
    if start_iter > 1 and history:
        history = [r for r in history if int(r.get("iteration", 0)) < start_iter]

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
        f"📊 Сплит 40-мин окон (win_len={win_len}s): "
        f"TRAIN[N={len(train_norm)}, D={len(train_dump)}, CASC={len(train_casc)}, P={len(train_pump)}] | "
        f"VAL[N={len(val_norm)}, D={len(val_dump)}, CASC={len(val_casc)}, P={len(val_pump)}] | "
        f"BLIND_OVERNIGHT={slice_len - blind_s0:,}с (100% ЗАПЕЧАТАН!)"
    )

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
    init_val_res = evaluate_l25_dual_basket(
        vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
        [base_folded] * 6, val_basket, win_len, tick_sz, device,
        true_latency=True, allow_scratch_stop_asks=allow_scratch_stop_asks, panic_thresh=panic_thresh,
    )[0]
    dt_v0 = time.time() - t_v0

    best_pareto_util = float(init_val_res["pareto_util_bp"])
    best_clean_ev = float(init_val_res["clean_ev"])
    best_val_state = {k: v.clone() for k, v in hof_state.items()}
    best_val_g_logit = hof_g_logit.clone()
    best_val_folded = {k: v.clone() for k, v in base_folded.items()}
    last_val_champ_it = max(0, start_iter - 1)

    log_msg(
        f"⚖️ СТАРТОВАЯ БАЗА L25 WARM-START (compiled {dt_v0:.1f}с): "
        f"ParetoUtil={best_pareto_util:+.2f}bp [CleanEV={best_clean_ev:+.4f}%, "
        f"Fills={init_val_res['clean_fills']:.1f}, Cancels={init_val_res['clean_cancels']:.1f}, "
        f"Requeue={init_val_res['clean_requeues']:.1f}, Cnl/Fill={init_val_res['clean_cnl_per_fill']:.2f}, "
        f"η_queue={init_val_res['clean_eta_queue_pct']:.1f}% | UW={init_val_res['uw_delta_ev']:+.4f}%] | "
        f"Clean[N:{init_val_res['clean_norm_pct']:+.3f}% D+C:{init_val_res['clean_dump_pct']:+.3f}% "
        f"P:{init_val_res['clean_pump_pct']:+.3f}% CVaR25:{init_val_res['clean_cvar25_pct']:+.2f}% DD:{init_val_res['clean_dd_pct']:.2f}%]"
    )

    if start_iter == 1:
        _save_l25_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, 0, 1, "L25_WARMSTART_0000", init_val_res)

    budget_tiers = torch.tensor([100.0, 300.0, 625.0, 1000.0, 1500.0, 2500.0], device=device)
    env_budgets = budget_tiers.unsqueeze(0).expand(n_groups, n_budgets).reshape(pop_size)
    rng = np.random.default_rng(rng_seed)
    for _ in range(1, start_iter):
        for _k in range(10):
            _ = rng.integers(0, 10)

    W_par = 10
    N_par = W_par * pop_size
    S_slots = int(vec_env.n_side_slots)
    budget_16 = env_budgets.unsqueeze(0).expand(W_par, pop_size).reshape(N_par).contiguous()
    nes_w = torch.softmax(-torch.arange(24, dtype=torch.float32, device=device) * 0.25, dim=0)
    mkt_idx_t = torch.tensor(MKT_IDX_L25, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_L25, dtype=torch.long, device=device)
    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    chunk_sz = 100
    last_sig_improve_it = max(0, start_iter - 1)
    sig_anchor_pareto = float(best_pareto_util)

    for it in range(start_iter, iterations + 1):
        t0 = time.time()
        if it <= phase1_iters:
            phase = 1
            p_prog = float(it - 1) / max(1.0, float(phase1_iters - 1))
            cosine_decay = 0.28 + 0.72 * (0.5 * (1.0 + math.cos(math.pi * p_prog)))
        elif it <= phase2_iters:
            phase = 2
            p_prog = float(it - phase1_iters - 1) / max(1.0, float(phase2_iters - phase1_iters - 1))
            cosine_decay = 0.28 + 0.72 * (0.5 * (1.0 + math.cos(math.pi * p_prog)))
        elif it <= phase3_iters:
            phase = 3
            p_prog = float(it - phase2_iters - 1) / max(1.0, float(phase3_iters - phase2_iters - 1))
            cosine_decay = 0.16 + 0.24 * (0.5 * (1.0 + math.cos(math.pi * p_prog)))
        else:
            phase = 4
            p_prog = float(it - phase3_iters - 1) / max(1.0, float(iterations - phase3_iters - 1))
            cosine_decay = 0.10 + 0.16 * (0.5 * (1.0 + math.cos(math.pi * p_prog)))

        if it == phase1_iters + 1:
            log_msg("🔓 ПЕРЕХОД К ФАЗЕ 2: Разморозка полного 480D ствола и совместная сквозная коэволюция Стратега + Микро-Регулятора!")
            hof_state = {k: v.clone() for k, v in best_val_state.items()}
            hof_g_logit = best_val_g_logit.clone()
        elif it == phase2_iters + 1:
            log_msg("💎 ПЕРЕХОД К ФАЗЕ 3: Тонкая косинусная полировка всех 480D + Микро-Регулятора (sigma=0.40->0.16) из лучшего чемпиона!")
            hof_state = {k: v.clone() for k, v in best_val_state.items()}
            hof_g_logit = best_val_g_logit.clone()
        elif it == phase3_iters + 1:
            log_msg(
                f"🎯 ПЕРЕХОД К ФАЗЕ 4 (АВТО-ПЛАТО): Микро-полировка (sigma=0.26->0.10) до сходимости "
                f"(patience={plateau_patience} итераций без прироста >= +{plateau_min_delta_bp:.2f}bp)!"
            )
            hof_state = {k: v.clone() for k, v in best_val_state.items()}
            hof_g_logit = best_val_g_logit.clone()

        policy.sample_population(hof_state, hof_g_logit, phase=phase, sigma_scale=cosine_decay)

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

            w_starts = torch.tensor([ws for _, ws in w_batch], dtype=torch.long, device=device)
            t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
            t_steps_cl = torch.clamp(t_steps, max=vec_env.max_steps - 1)

            flow_w_all = vec_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
            lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()
            v7_w_all = v7_all[t_steps_cl[: win_len - 1]].contiguous()
            bo36_w_all = bo36_all[t_steps_cl[: win_len - 1]].contiguous()
            z_mkt_all = precompute_window_market_z370(flow_w_all, lob_fu_w_all, v7_w_all, bo36_w_all, bf16_runner)
            del flow_w_all, lob_fu_w_all, v7_w_all, bo36_w_all

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
            sum_fqp_16 = torch.zeros(N_par, dtype=torch.float32, device=device)
            n_fills_16 = torch.zeros(N_par, dtype=torch.int16, device=device)
            n_cancels_16 = torch.zeros(N_par, dtype=torch.int16, device=device)
            n_requeues_16 = torch.zeros(N_par, dtype=torch.int16, device=device)

            for c0_t in range(0, win_len - 1, chunk_sz):
                c1_t = min(win_len - 1, c0_t + chunk_sz)
                m_len = c1_t - c0_t
                z_ch = z_mkt_all[c0_t:c1_t].reshape(m_len * W_par, 370).unsqueeze(0).expand(n_groups, -1, -1)
                pre_mkt_ch = (torch.bmm(z_ch, w1_mkt_g) + policy.b1).view(n_groups, m_len, W_par, 384).permute(1, 0, 2, 3).contiguous()

                for j_t in range(m_len):
                    t_step = c0_t + j_t
                    t_nxt = t_step + 1
                    (
                        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                        peak_mtm_eq, sum_sq_dd_16, max_dd_pct_16, sum_mut_16, sum_fqp_16,
                        n_fills_16, n_cancels_16, n_requeues_16,
                    ) = fused_l25_regulator_tick(
                        s_steps_i16[t_step], s_steps_i16[t_nxt], pre_mkt_ch[j_t],
                        b1_w_all[t_step], a1_w_all[t_step], b1q_w_all[t_step], a1q_w_all[t_step],
                        rb_w_all[t_step], rs_w_all[t_step],
                        b10_w_all[t_step], a10_w_all[t_step], bq10_w_all[t_step], aq10_w_all[t_step],
                        b1_w_all[t_nxt], a1_w_all[t_nxt], sv_w_all[t_nxt], bv_w_all[t_nxt],
                        w1_ag_g, policy.w_res1, policy.b_res1, policy.w_res2,
                        policy.w2, policy.b2, policy.w_rec, policy.b_rec,
                        policy.w_reg1, policy.b_reg1, policy.w_reg2, policy.b_reg2,
                        vec_env.slot_tick_offsets, vec_env.slot_lob_lvl, vec_env.is_penny_slot, budget_16,
                        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
                        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
                        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
                        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
                        peak_mtm_eq, sum_sq_dd_16, max_dd_pct_16, sum_mut_16, sum_fqp_16,
                        n_fills_16, n_cancels_16, n_requeues_16,
                        tick_sz, vec_env.maker_fee, vec_env.taker_fee,
                        True, allow_scratch_stop_asks, panic_thresh,
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
            w_requeues_mat = n_requeues_16.float().view(W_par, pop_size)
            w_fqp_mat = sum_fqp_16.view(W_par, pop_size)

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
            flat_requeues = w_requeues_mat[0:6].sum(dim=0)
            flat_fqp = w_fqp_mat[0:6].sum(dim=0)
            norm_fills = w_fills_mat[0:2].sum(dim=0)
            pump_fills = w_fills_mat[4:6].sum(dim=0)

            cnl_per_fill_vec = flat_cancels / torch.clamp(flat_fills, min=1.0)
            tremor_pen_vec = torch.clamp(cnl_per_fill_vec - 12.0, min=0.0) * 0.010

            flat_util_bp = 100.0 * (
                1.8 * flat_ret_pct
                + 0.25 * flat_cvar_pct
                - 0.45 * pump_shortfall_pct
                - flat_ulcer_pct
                - 0.06 * (flat_dd_pct * flat_dd_pct)
                - 0.28 * flat_api
                - tremor_pen_vec
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
                - 0.28 * uw_api
            )

            g_flat_u_mat = flat_util_bp.view(n_groups, n_budgets)
            g_uw_u_mat = uw_util_bp.view(n_groups, n_budgets)
            g_flat_ret_mat = flat_ret_pct.view(n_groups, n_budgets)
            g_uw_ret_mat = uw_ret_pct.view(n_groups, n_budgets)
            g_dd_mat = flat_dd_pct.view(n_groups, n_budgets)
            g_fills_mat = flat_fills.view(n_groups, n_budgets)
            g_cancels_mat = flat_cancels.view(n_groups, n_budgets)
            g_requeues_mat = flat_requeues.view(n_groups, n_budgets)
            g_fqp_mat = flat_fqp.view(n_groups, n_budgets)

            flat_score_g = g_flat_u_mat.mean(dim=1) - 0.20 * g_flat_u_mat.std(dim=1)
            uw_score_g = g_uw_u_mat.mean(dim=1) - 0.20 * g_uw_u_mat.std(dim=1)
            norm_fills_g = norm_fills.view(n_groups, n_budgets).mean(dim=1)
            pump_fills_g = pump_fills.view(n_groups, n_budgets).mean(dim=1)

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
            b_dd_pct = float(g_dd_mat[best_g].mean().item())
            b_fills = float(g_fills_mat[best_g].mean().item())
            b_cancels = float(g_cancels_mat[best_g].mean().item())
            b_requeues = float(g_requeues_mat[best_g].mean().item())
            b_fqp = float(g_fqp_mat[best_g].mean().item())
            b_cnl_per_fill = b_cancels / max(b_fills, 0.5)
            b_eta_q = 100.0 * min(1.0, b_fqp / max(b_fills, 1.0)) if b_fills >= 0.5 else 0.0
            b_prof_buds = int(g_prof_buds[best_g].item())

            top_g_idx = sorted_g[:24]
            cand_raw = {}
            for k in DualSwarmRegulatorPolicyL25.PARAM_KEYS:
                p_top = getattr(policy, k)[top_g_idx].float()
                cand_raw[k] = (p_top * nes_w.view(-1, 1, 1)).sum(dim=0)
            cand_gl = (policy.g_logits[top_g_idx].float() * nes_w.view(-1, 1)).sum(dim=0).clamp(-2.5, 2.0)
            cand_norm = _normalize_weights_l25(cand_raw, policy.dtype)

            polyak_trunk = 0.0 if phase == 1 else (0.20 * cosine_decay)
            polyak_fast = min(0.55, 0.42 * cosine_decay)

            w1_old_f = hof_state["w1"].float()
            w1_cand_f = cand_norm["w1"].detach().cpu().float()
            w1_new_f = w1_old_f.clone()
            if polyak_trunk > 0.0:
                w1_new_f[:444, :] = (1.0 - polyak_trunk) * w1_old_f[:444, :] + polyak_trunk * w1_cand_f[:444, :]
            w1_new_f[444:480, :] = (1.0 - polyak_fast) * w1_old_f[444:480, :] + polyak_fast * w1_cand_f[444:480, :]

            cand_polyak_state = {"w1": w1_new_f.to(policy.dtype)}
            for k in ("b1", "w_res1", "b_res1", "w_res2"):
                a_k = polyak_trunk
                cand_polyak_state[k] = ((1.0 - a_k) * hof_state[k].float() + a_k * cand_norm[k].detach().cpu().float()).to(policy.dtype)
            for k in ("w2", "b2", "w_rec", "b_rec", "w_reg1", "b_reg1", "w_reg2", "b_reg2"):
                a_k = polyak_fast
                cand_polyak_state[k] = ((1.0 - a_k) * hof_state[k].float() + a_k * cand_norm[k].detach().cpu().float()).to(policy.dtype)

            cand_polyak_gl = hof_g_logit.clone()
            cand_gl_cpu = cand_gl.detach().cpu()
            if phase == 1:
                cand_polyak_gl[16:] = (1.0 - polyak_fast) * hof_g_logit[16:] + polyak_fast * cand_gl_cpu[16:]
            else:
                cand_polyak_gl = (1.0 - polyak_fast) * hof_g_logit + polyak_fast * cand_gl_cpu

            if b_fit > c0_fit and b_fills >= 1.0:
                hof_state = cand_polyak_state
                hof_g_logit = cand_polyak_gl

            val_tag = ""
            if it % 4 == 0 or it == phase1_iters or it == iterations or (b_fit > c0_fit + 0.35 and b_prof_buds >= 4):
                ls_alphas = [0.05, 0.15, 0.35, 0.70, 1.00]
                ls_raw_states, ls_gls, ls_folded = [], [], []
                c_cpu = {k: cand_norm[k].detach().cpu().float() for k in DualSwarmRegulatorPolicyL25.PARAM_KEYS}
                for a_ls in ls_alphas:
                    st_ls = {}
                    for k in DualSwarmRegulatorPolicyL25.PARAM_KEYS:
                        if phase == 1 and k in ("b1", "w_res1", "b_res1", "w_res2"):
                            st_ls[k] = best_val_state[k].clone()
                        elif phase == 1 and k == "w1":
                            w_m = best_val_state["w1"].float().clone()
                            w_m[444:480, :] = (1.0 - a_ls) * w_m[444:480, :] + a_ls * c_cpu["w1"][444:480, :]
                            st_ls[k] = w_m.to(policy.dtype)
                        else:
                            st_ls[k] = ((1.0 - a_ls) * best_val_state[k].float() + a_ls * c_cpu[k]).to(policy.dtype)
                    st_ls = _normalize_weights_l25(st_ls, policy.dtype)
                    gl_ls = best_val_g_logit.clone()
                    if phase == 1:
                        gl_ls[16:] = (1.0 - a_ls) * best_val_g_logit[16:] + a_ls * cand_gl_cpu[16:]
                    else:
                        gl_ls = (1.0 - a_ls) * best_val_g_logit + a_ls * cand_gl_cpu
                    ls_raw_states.append(st_ls)
                    ls_gls.append(gl_ls)
                    ls_folded.append(policy.fold_gate_into_state(st_ls, gl_ls))

                ls_raw_states.append({k: v.clone() for k, v in hof_state.items()})
                ls_gls.append(hof_g_logit.clone())
                ls_folded.append(policy.fold_gate_into_state(hof_state, hof_g_logit))

                val_res_list = evaluate_l25_dual_basket(
                    vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
                    ls_folded, val_basket, win_len, tick_sz, device,
                    true_latency=True, allow_scratch_stop_asks=allow_scratch_stop_asks, panic_thresh=panic_thresh,
                )
                best_ls_i = int(np.argmax([r["pareto_util_bp"] if r["clean_fills"] >= 8.0 else -1e9 for r in val_res_list]))
                vr = val_res_list[best_ls_i]
                v_pu = float(vr["pareto_util_bp"])

                if v_pu > best_pareto_util and vr["clean_fills"] >= 8.0:
                    if v_pu >= sig_anchor_pareto + plateau_min_delta_bp:
                        sig_anchor_pareto = v_pu
                        last_sig_improve_it = it
                    best_pareto_util = v_pu
                    best_clean_ev = float(vr["clean_ev"])
                    best_val_state = {k: v.clone() for k, v in ls_raw_states[best_ls_i].items()}
                    best_val_g_logit = ls_gls[best_ls_i].clone()
                    best_val_folded = {k: v.clone() for k, v in ls_folded[best_ls_i].items()}
                    hof_state = {k: v.clone() for k, v in best_val_state.items()}
                    hof_g_logit = best_val_g_logit.clone()
                    last_val_champ_it = it
                    rel_nm = f"L25_P{phase}_IT{it:04d}_DUAL_SWARM_REGULATOR"
                    _save_l25_checkpoint(LATEST_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, it, phase, rel_nm, vr)
                    _save_l25_checkpoint(SAVE_DIR / f"l25_champ_iter_{it:04d}.pt", base_ck, best_val_folded, best_val_g_logit, it, phase, rel_nm, vr)
                    val_tag = (
                        f" 🏆 NEW L25 CHAMP! Pareto={v_pu:+.2f}bp [EV={vr['clean_ev']:+.4f}%, "
                        f"D+C={vr['clean_dump_pct']:+.3f}%, P={vr['clean_pump_pct']:+.3f}%, DD={vr['clean_dd_pct']:.2f}%, "
                        f"Cnl/Fill={vr['clean_cnl_per_fill']:.2f}, η_q={vr['clean_eta_queue_pct']:.1f}%, UW={vr['uw_delta_ev']:+.3f}%]"
                    )
                else:
                    val_tag = f" | Val={v_pu:+.2f}bp (best={best_pareto_util:+.2f}bp @ it={last_val_champ_it})"

                anchor_patience = 14 if phase >= 3 else 24
                if it - last_val_champ_it >= anchor_patience:
                    hof_state = {k: v.clone() for k, v in best_val_state.items()}
                    hof_g_logit = best_val_g_logit.clone()

            dt_it = time.time() - t0
            rec = {
                "iteration": it,
                "phase": phase,
                "train_best_fit": round(b_fit, 4),
                "train_flat_ret": round(b_flat_ret, 4),
                "train_uw_ret": round(b_uw_ret, 4),
                "train_cnl_per_fill": round(b_cnl_per_fill, 2),
                "train_eta_queue": round(b_eta_q, 1),
                "best_val_pareto_bp": round(best_pareto_util, 2),
                "best_val_clean_ev": round(best_clean_ev, 4),
                "elapsed_s": round(dt_it, 2),
            }
            history.append(rec)
            summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            _save_l25_checkpoint(
                RESUME_CHAMP_PATH, base_ck, best_val_folded, best_val_g_logit, it, phase,
                f"L25_RESUME_IT{it:04d}", {"pareto_util_bp": best_pareto_util, "clean_util_bp": 0.0, "uw_util_bp": 0.0, "clean_ev": best_clean_ev},
            )
            log_msg(
                f"[L25 P{phase} it={it:04d}/{iterations} | {dt_it:4.1f}s] "
                f"TrFit={b_fit:+.3f} (c0={c0_fit:+.3f}) | EV={b_flat_ret:+.3f}% UW={b_uw_ret:+.3f}% DD={b_dd_pct:.2f}% | "
                f"Fills={b_fills:.1f} Cnl={b_cancels:.1f} Req={b_requeues:.1f} Cnl/F={b_cnl_per_fill:.2f} η_q={b_eta_q:.1f}%{val_tag}"
            )
            if phase >= 4 and plateau_patience > 0 and (it - last_sig_improve_it) >= plateau_patience:
                log_msg(
                    f"🛑 ДОСТИГНУТО АВТО-ПЛАТО на it={it}: за последние {it - last_sig_improve_it} итераций "
                    f"не было прироста >= +{plateau_min_delta_bp:.2f}bp (лучший рекорд {best_pareto_util:+.2f}bp @ it={last_val_champ_it}). "
                    f"Завершаем обучение и собираем Model Soup!"
                )
                break

    # Автоматическая сборка Model Soup из топ-2 различных чекпоинтов Урока 25 на VAL_CASCADE (БЕЗ взгляда в BLIND_OVERNIGHT!):
    champ_files = sorted(SAVE_DIR.glob("l25_champ_iter_*.pt"))
    if len(champ_files) >= 2:
        cks = [torch.load(p, map_location="cpu", weights_only=False) for p in champ_files]
        cks.sort(key=lambda c: float(c.get("pareto_util_bp", -1e9)), reverse=True)
        c_a, c_b = cks[0], cks[1]
        soup_cands = []
        alphas_soup = [0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.0]
        for a_s in alphas_soup:
            st_s = {}
            for k in DualSwarmRegulatorPolicyL25.PARAM_KEYS:
                st_s[k] = ((1.0 - a_s) * c_a[k].float() + a_s * c_b[k].float()).to(policy.dtype)
            soup_cands.append(st_s)
        soup_res = evaluate_l25_dual_basket(
            vec_env, bf16_runner, lob_fu_all, v7_all, bo36_all, roll_sell_30s_all,
            soup_cands, val_basket, win_len, tick_sz, device,
            true_latency=True, allow_scratch_stop_asks=allow_scratch_stop_asks, panic_thresh=panic_thresh,
        )
        best_si = int(np.argmax([r["pareto_util_bp"] for r in soup_res]))
        sr = soup_res[best_si]
        soup_name = f"L25_SOUP_{c_a['iteration']:04d}_{c_b['iteration']:04d}_a{int(alphas_soup[best_si]*100):02d}"
        _save_l25_checkpoint(SOUP_CHAMP_PATH, base_ck, soup_cands[best_si], best_val_g_logit, iterations, 2, soup_name, sr)
        log_msg(
            f"🍲 СОБРАН MODEL SOUP {soup_name}: Pareto={sr['pareto_util_bp']:+.2f}bp | CleanEV={sr['clean_ev']:+.4f}% | "
            f"Cnl/Fill={sr['clean_cnl_per_fill']:.2f} | η_q={sr['clean_eta_queue_pct']:.1f}%!"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=220)
    ap.add_argument("--phase1-iters", type=int, default=55)
    ap.add_argument("--phase2-iters", type=int, default=220)
    ap.add_argument("--phase3-iters", type=int, default=500)
    ap.add_argument("--plateau-patience", type=int, default=0)
    ap.add_argument("--plateau-min-delta-bp", type=float, default=0.30)
    ap.add_argument("--n-groups", type=int, default=1024)
    ap.add_argument("--win-len", type=int, default=2400)
    args = ap.parse_args()
    run_lesson_25(
        iterations=args.iterations,
        phase1_iters=args.phase1_iters,
        phase2_iters=args.phase2_iters,
        phase3_iters=args.phase3_iters,
        plateau_patience=args.plateau_patience,
        plateau_min_delta_bp=args.plateau_min_delta_bp,
        n_groups=args.n_groups,
        win_len=args.win_len,
    )
