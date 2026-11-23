#!/usr/bin/env python3
"""
_tools/order3_live_bridge.py
=============================================================================
Order-3 Continuous RL Live Bridge (100% Data-Driven & Scale-Invariant)
=============================================================================
- Вход: 149D = 39D Market (16D O1 Alpha + 11D O2 Risk + 12D LOB) + 110D Agent
- Архитектура: K=4 Latent MoE + ResSwiGLU + Context Router + Per-Slot Regulator
- Token-Bucket API Rate Limiter: +0.35/s, burst cap 10.0
- 0 жестких порогов, 0 ручных весов бирж, 100% непрерывная шкала.
=============================================================================
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from _tools.fused_order3_clean_tick import (
    MKT_DIM_ORDER3,
    AG_DIM_ORDER3,
    TOTAL_DIM_ORDER3,
    MEM_DIM_ORDER3,
    K_MODES_ORDER3,
    EXPERT_OUT_DIM_ORDER3,
    MOE_OUT_DIM_ORDER3,
    ROUTER_IN_DIM_ORDER3,
    ROUTER_H_DIM_ORDER3,
    REG_IN_DIM_ORDER3,
    REG_H_DIM_ORDER3,
    TOKEN_REFILL_PER_SEC_ORDER3,
    TOKEN_BUCKET_CAP_ORDER3,
    _fanned_slot_softmax,
    _normalize_agent_z110_per_block,
    _scale_mkt_44d,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHAMP_PATH = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
LIVE_TOKEN_REFILL_PER_SEC = TOKEN_REFILL_PER_SEC_ORDER3
LIVE_TOKEN_BUCKET_CAP = TOKEN_BUCKET_CAP_ORDER3


class Order3LiveInferenceEngine:
    """
    Высокопроизводительный инференс-движок Order-3 для live_rl_trader.
    Обеспечивает 100% побитовый паритет с симулятором fused_order3_clean_tick.py.
    """

    def __init__(self, checkpoint_path: Optional[Union[Path, str]] = None, device: str = "cpu"):
        self.device = torch.device(device)
        p = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHAMP_PATH
        if not p.exists():
            for cand in [
                BASE_DIR / "models/prl/release_champion_v1.pt",
                BASE_DIR / "models/prl/release_champion_v2.pt",
                BASE_DIR / "models/prl/order3_champion/global_best.pt",
                BASE_DIR / "models/prl/order3_champion/order3_champ_latest.pt",
            ]:
                if cand.exists():
                    p = cand
                    break

        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.pure_endogenous = bool(ck.get("pure_endogenous", True))

        # Веса 154D -> 384 (с бесшовной обратной совместимостью 149D / 150D -> 154D)
        w1_full = ck["w1"].to(dtype=torch.float32)
        if w1_full.shape[0] != TOTAL_DIM_ORDER3:
            old_mkt_d = w1_full.shape[0] - AG_DIM_ORDER3
            w1_adapted = torch.zeros((TOTAL_DIM_ORDER3, 384), dtype=torch.float32)
            copy_m = min(MKT_DIM_ORDER3, max(0, old_mkt_d))
            w1_adapted[:copy_m, :] = w1_full[:copy_m, :]
            w1_adapted[MKT_DIM_ORDER3:, :] = w1_full[old_mkt_d:, :]
            w1_full = w1_adapted

        self.w1_mkt = w1_full[:MKT_DIM_ORDER3, :].to(device=self.device)  # [44, 384]
        self.w1_ag = w1_full[MKT_DIM_ORDER3:, :].to(device=self.device)   # [110, 384]
        self.b1 = ck["b1"].view(1, 384).to(device=self.device, dtype=torch.float32)

        # ResSwiGLU блок
        self.w_res1 = ck["w_res1"].to(device=self.device, dtype=torch.float32)
        self.b_res1 = ck["b_res1"].view(1, 128).to(device=self.device, dtype=torch.float32)
        self.w_res2 = ck["w_res2"].to(device=self.device, dtype=torch.float32)

        # Латентная многомасштабная память (16D -> 32D с бесшовной совместимостью L46/L48)
        wrec_raw = ck["w_rec"].to(dtype=torch.float32)
        brec_raw = ck["b_rec"].to(dtype=torch.float32).view(1, -1)
        if brec_raw.numel() != 2 * MEM_DIM_ORDER3:
            wrec_pad = torch.zeros((256, 2 * MEM_DIM_ORDER3), dtype=torch.float32)
            brec_pad = torch.zeros((1, 2 * MEM_DIM_ORDER3), dtype=torch.float32)
            wrec_pad[:, :16] = wrec_raw[:, :16]
            wrec_pad[:, 32:48] = wrec_raw[:, 16:32]
            brec_pad[:, :16] = brec_raw[:, :16]
            brec_pad[:, 32:48] = brec_raw[:, 16:32]
            wrec_raw, brec_raw = wrec_pad, brec_pad
        self.w_rec = wrec_raw.to(device=self.device)
        self.b_rec = brec_raw.view(1, 2 * MEM_DIM_ORDER3).to(device=self.device)

        # Авто-определение числа мод K (поддержка как K=4, так и расширенных K=8 чекпоинтов!)
        self.k_modes = int(ck["b_gate2"].numel())
        self.moe_out_dim = self.k_modes * EXPERT_OUT_DIM_ORDER3
        self.reg_in_dim = 28 + self.k_modes

        # Контекстный маршрутизатор (с бесшовной обратной совместимостью 288D / 295D -> 311D)
        wg1_raw = ck["w_gate1"].to(dtype=torch.float32)
        if wg1_raw.shape[0] != ROUTER_IN_DIM_ORDER3:
            wg1_adap = torch.zeros((ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), dtype=torch.float32)
            if wg1_raw.shape[0] == 295 and ROUTER_IN_DIM_ORDER3 >= 311:
                wg1_adap[:288, :] = wg1_raw[:288, :]
                wg1_adap[304:311, :] = wg1_raw[288:295, :]
            else:
                copy_r = min(ROUTER_IN_DIM_ORDER3, wg1_raw.shape[0])
                wg1_adap[:copy_r, :] = wg1_raw[:copy_r, :]
            wg1_raw = wg1_adap
        self.w_gate1 = wg1_raw.to(device=self.device)
        self.b_gate1 = ck["b_gate1"].view(1, ROUTER_H_DIM_ORDER3).to(device=self.device, dtype=torch.float32)
        self.w_gate2 = ck["w_gate2"].to(device=self.device, dtype=torch.float32)
        self.b_gate2 = ck["b_gate2"].view(1, self.k_modes).to(device=self.device, dtype=torch.float32)

        # MoE эксперты (с бесшовной обратной совместимостью 45D -> 48D на моду)
        w2m_raw = ck["w2_moe"].to(dtype=torch.float32)
        b2m_raw = ck["b2_moe"].to(dtype=torch.float32).view(1, -1)
        if b2m_raw.numel() != self.moe_out_dim:
            old_ed = b2m_raw.numel() // self.k_modes
            w2m_old = w2m_raw.view(256, self.k_modes, old_ed)
            b2m_old = b2m_raw.view(1, self.k_modes, old_ed)
            w2m_pad = torch.zeros((256, self.k_modes, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32)
            b2m_pad = torch.zeros((1, self.k_modes, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32)
            c_ed = min(old_ed, EXPERT_OUT_DIM_ORDER3)
            w2m_pad[:, :, :c_ed] = w2m_old[:, :, :c_ed]
            b2m_pad[:, :, :c_ed] = b2m_old[:, :, :c_ed]
            if old_ed <= 45 and EXPERT_OUT_DIM_ORDER3 >= 48:
                b2m_pad[:, :, 47] = 6.0
            if old_ed <= 48 and EXPERT_OUT_DIM_ORDER3 >= 50:
                b2m_pad[:, :, 48:50] = -12.0
            w2m_raw = w2m_pad.reshape(256, self.moe_out_dim)
            b2m_raw = b2m_pad.reshape(1, self.moe_out_dim)
        self.w2_moe = w2m_raw.to(device=self.device)
        self.b2_moe = b2m_raw.view(1, self.moe_out_dim).to(device=self.device)

        # Регулятор слотов (с бесшовной обратной совместимостью 26D / 32D->48 -> 36D->64)
        wreg1_raw = ck["w_reg1"].to(dtype=torch.float32)
        breg1b_raw = ck["b_reg1_b"].to(dtype=torch.float32).view(1, -1)
        breg1a_raw = ck["b_reg1_a"].to(dtype=torch.float32).view(1, -1)
        wreg2_raw = ck["w_reg2"].to(dtype=torch.float32)
        if wreg1_raw.shape != (self.reg_in_dim, REG_H_DIM_ORDER3):
            old_ri, old_rh = wreg1_raw.shape
            wreg1_adap = torch.zeros((self.reg_in_dim, REG_H_DIM_ORDER3), dtype=torch.float32)
            breg1b_adap = torch.zeros((1, REG_H_DIM_ORDER3), dtype=torch.float32)
            breg1a_adap = torch.zeros((1, REG_H_DIM_ORDER3), dtype=torch.float32)
            wreg2_adap = torch.zeros((REG_H_DIM_ORDER3, 2), dtype=torch.float32)
            cr_h = min(old_rh, REG_H_DIM_ORDER3)
            if old_ri == 24 + self.k_modes and self.reg_in_dim == 28 + self.k_modes:
                wreg1_adap[:24, :cr_h] = wreg1_raw[:24, :cr_h]
                wreg1_adap[28 : 28 + self.k_modes, :cr_h] = wreg1_raw[24 : 24 + self.k_modes, :cr_h]
            elif old_ri == 26:
                wreg1_adap[:22, :cr_h] = wreg1_raw[:22, :cr_h]
                wreg1_adap[28:32, :cr_h] = wreg1_raw[22:26, :cr_h]
            else:
                cr_reg = min(self.reg_in_dim, old_ri)
                wreg1_adap[:cr_reg, :cr_h] = wreg1_raw[:cr_reg, :cr_h]
            breg1b_adap[:, :cr_h] = breg1b_raw[:, :cr_h]
            breg1a_adap[:, :cr_h] = breg1a_raw[:, :cr_h]
            wreg2_adap[:cr_h, :] = wreg2_raw[:cr_h, :]
            wreg1_raw, breg1b_raw, breg1a_raw, wreg2_raw = wreg1_adap, breg1b_adap, breg1a_adap, wreg2_adap
        self.w_reg1 = wreg1_raw.to(device=self.device)
        self.b_reg1_b = breg1b_raw.view(1, REG_H_DIM_ORDER3).to(device=self.device)
        self.b_reg1_a = breg1a_raw.view(1, REG_H_DIM_ORDER3).to(device=self.device)
        self.w_reg2 = wreg2_raw.to(device=self.device)
        self.b_reg2_b = ck["b_reg2_b"].view(1, 2).to(device=self.device, dtype=torch.float32)
        self.b_reg2_a = ck["b_reg2_a"].view(1, 2).to(device=self.device, dtype=torch.float32)

        # Внутреннее состояние
        self.mem_h = torch.zeros((1, MEM_DIM_ORDER3), dtype=torch.float32, device=self.device)
        self.api_tokens = float(LIVE_TOKEN_BUCKET_CAP)
        self.last_ts = time.time()
        self.echelon_k01 = torch.linspace(0.0, 1.0, 12, dtype=torch.float32, device=self.device)

    def refill_tokens(self, now_ts: float) -> float:
        dt = max(0.0, min(10.0, now_ts - self.last_ts))
        self.last_ts = now_ts
        self.api_tokens = min(LIVE_TOKEN_BUCKET_CAP, self.api_tokens + LIVE_TOKEN_REFILL_PER_SEC * dt)
        return self.api_tokens

    @torch.no_grad()
    def forward_tick(
        self,
        z_mkt_39d: np.ndarray,
        z_ag_110d: np.ndarray,
        b1: float,
        a1: float,
        budget: float,
        pos_qty: float,
        cash: float,
        bid_px_12: np.ndarray,
        bid_qty_12: np.ndarray,
        ask_px_12: np.ndarray,
        ask_qty_12: np.ndarray,
        tick_sz: float = 0.0001,
        maker_fee: float = 0.0010,
        trader_entry: float = 0.0,
        bid_act_12: Optional[np.ndarray] = None,
        ask_act_12: Optional[np.ndarray] = None,
        now_ts: float = 0.0,
        bid_qa_12: Optional[np.ndarray] = None,
        bid_qa0_12: Optional[np.ndarray] = None,
        ask_qa_12: Optional[np.ndarray] = None,
        ask_qa0_12: Optional[np.ndarray] = None,
        bp_10: Optional[np.ndarray] = None,
        bq_10: Optional[np.ndarray] = None,
        ap_10: Optional[np.ndarray] = None,
        aq_10: Optional[np.ndarray] = None,
        roll_sell_1s: float = 10.0,
        roll_buy_1s: float = 10.0,
        hold_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Полный шаг инференса Order-3 (L46/L48 100% паритет с fused_order3_clean_tick.py).
        """
        # Тензоры
        zm = _scale_mkt_44d(torch.from_numpy(z_mkt_39d).to(device=self.device, dtype=torch.float32).view(1, MKT_DIM_ORDER3), torch.float32)
        za = torch.from_numpy(z_ag_110d).to(device=self.device, dtype=torch.float32).view(1, AG_DIM_ORDER3)
        za_norm = _normalize_agent_z110_per_block(za, torch.float32)

        # 1. 170D -> 384
        pre_all = (zm @ self.w1_mkt + self.b1) + (za_norm @ self.w1_ag)
        h_base = torch.relu(pre_all[:, :256])
        h_gate = (2.0 * torch.sigmoid(pre_all[:, 256:384])).repeat_interleave(2, dim=-1)
        h1 = h_base * h_gate

        # ResSwiGLU
        res_pre = h1 @ self.w_res1 + self.b_res1
        res_mid = torch.relu(res_pre[:, :64]) * (2.0 * torch.sigmoid(res_pre[:, 64:128]))
        h2 = h1 + res_mid @ self.w_res2

        # 4-горизонтная латентная память 32D (~3.7s, ~45s, ~180s, ~735s)
        rec_raw = h2 @ self.w_rec + self.b_rec
        rec_u = torch.tanh(rec_raw[:, :MEM_DIM_ORDER3])
        rec_gate = torch.cat(
            [
                torch.sigmoid(rec_raw[:, 32:40] + 1.0),
                torch.sigmoid(rec_raw[:, 40:48] + 3.8),
                torch.sigmoid(rec_raw[:, 48:56] + 5.2),
                torch.sigmoid(rec_raw[:, 56:64] + 6.6),
            ],
            dim=-1,
        )
        self.mem_h = rec_gate * self.mem_h + (1.0 - rec_gate) * rec_u

        # Контекстный маршрутизатор (326D -> 64 -> K, с бесшовной обратной совместимостью 288D / 295D / 320D -> 326D)
        acct_8d = za[:, 54:62]
        queue_8d = za[:, 62:70]
        r_extra_22d = torch.cat(
            [
                zm[:, 0:1], zm[:, 40:44], zm[:, 44:45], zm[:, 48:49],
                zm[:, 29:34], zm[:, 36:37], zm[:, 39:40], zm[:, 55:56], zm[:, 12:13],
                zm[:, 60:66],
            ],
            dim=-1,
        )
        router_in = torch.cat([h2, acct_8d, queue_8d, self.mem_h, r_extra_22d], dim=-1)
        r_hid = F.silu(router_in @ self.w_gate1 + self.b_gate1)
        g_mode = torch.softmax((r_hid @ self.w_gate2 + self.b_gate2).float(), dim=-1)

        # K Латентных экспертов (EXPERT_OUT_DIM_ORDER3 = 50)
        moe_all = (h2 @ self.w2_moe + self.b2_moe).float().view(1, self.k_modes, EXPERT_OUT_DIM_ORDER3)
        strat_raw = (g_mode.unsqueeze(-1) * moe_all).sum(dim=1).squeeze(0)  # [50]

        # 1C. 100% Эндогенная экспозиция [0..1], распределение долей и резерв каскадной постановки renorm_blend:
        exp_bid_frac = float(torch.sigmoid(strat_raw[0]).item())
        renorm_blend = float(torch.sigmoid(strat_raw[47]).item())

        tau_b_t = 0.08 + 2.42 * torch.sigmoid(strat_raw[34:35])
        tau_a_t = 0.08 + 2.42 * torch.sigmoid(strat_raw[35:36])
        max_share_b_t = 0.08 + 0.92 * torch.sigmoid(strat_raw[44:45])
        max_share_a_t = 0.08 + 0.92 * torch.sigmoid(strat_raw[44:45] + 0.25)

        pi_b_t = _fanned_slot_softmax(strat_raw[1:13].unsqueeze(0), tau_b_t, max_share_b_t, floor=0.005).squeeze(0)
        pi_a_t = _fanned_slot_softmax(strat_raw[13:25].unsqueeze(0), tau_a_t, max_share_a_t, floor=0.005).squeeze(0)

        # 2. Непрерывные эндогенные Taker-вентили (Exit ch 25 + Impulse Wave Buy ch 48):
        u_taker = float(torch.tanh(F.relu(strat_raw[25])).item())
        u_taker_buy = float(torch.tanh(F.relu(strat_raw[48] - 1.2)).item())
        max_slip_pct = 0.005 + 0.145 * float(torch.sigmoid(strat_raw[36]).item())

        # 3. Независимая полно-спредовая геометрия 24 слотов + гладкий дифференцируемый Wave-Hold Ask Target (+0..+10%, ch 49):
        spread_room = max(0.0, a1 - b1 - tick_sz)
        inside_b = spread_room * float(torch.sigmoid(strat_raw[37]).item())
        retreat_b = b1 * 0.04 * float(torch.sigmoid(strat_raw[38]).item())
        span_b_usd = b1 * (0.0005 + 0.2495 * float(torch.sigmoid(strat_raw[39]).item()))
        gamma_b = 0.40 + 2.10 * float(torch.sigmoid(strat_raw[40]).item())

        curve_b = torch.pow(self.echelon_k01, gamma_b)
        raw_t_bid = (b1 + inside_b - retreat_b) - span_b_usd * curve_b
        max_maker_bid = max(tick_sz, a1 - tick_sz)
        t_bid_px = torch.clamp(torch.round(torch.clamp(raw_t_bid, max=max_maker_bid) / tick_sz) * tick_sz, min=tick_sz)

        ref_entry = float(trader_entry) * (1.0 + 2.0 * maker_fee) if (pos_qty >= 2.0 and trader_entry > 1e-4) else a1
        skew_blend = float(torch.sigmoid(strat_raw[41]).item())
        z42_t = strat_raw[42]
        t42_s = float(torch.tanh(z42_t * 0.25).item())
        wave_hold_pct = 0.10 * float(torch.sigmoid(strat_raw[49] - 1.2).item())
        ask_edge_pct = 0.0035 * float(torch.tanh(z42_t).item()) + 0.0765 * (t42_s * t42_s * t42_s) + wave_hold_pct
        span_a_usd = a1 * (0.0005 + 0.2495 * float(torch.sigmoid(strat_raw[39] + 0.5 * z42_t + strat_raw[45]).item()))
        gamma_a = 0.40 + 2.10 * float(torch.sigmoid(strat_raw[40] + strat_raw[46]).item())

        inside_a = spread_room * float(torch.sigmoid(strat_raw[43]).item())
        anchor_base_a = a1 + skew_blend * (ref_entry - a1) + a1 * ask_edge_pct - inside_a
        min_maker_ask = b1 + tick_sz
        curve_a = torch.pow(self.echelon_k01, gamma_a)
        raw_t_ask = anchor_base_a + span_a_usd * curve_a
        t_ask_px = torch.maximum(torch.full_like(raw_t_ask, min_maker_ask), torch.round(raw_t_ask / tick_sz) * tick_sz)

        tot_bud_eff = max(budget, cash + pos_qty * b1)
        target_bid_pool_usd = max(0.0, tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1)
        q_target_b = (target_bid_pool_usd * pi_b_t) / (t_bid_px * (1.0 + maker_fee))
        q_target_b = torch.where(q_target_b >= 2.0, torch.floor(q_target_b * 10000.0) / 10000.0, torch.zeros_like(q_target_b))
        q_target_a = torch.where(torch.full_like(pi_a_t, pos_qty >= 2.0, dtype=torch.bool), torch.floor((pos_qty * pi_a_t) * 10000.0) / 10000.0, torch.zeros_like(pi_a_t))
        q_target_a = torch.where(q_target_a >= 2.0, q_target_a, torch.zeros_like(q_target_a))

        # 4. Модально-контекстный Послотовый Нейро-Регулятор (26D -> 48 -> 2D)
        cur_b_px = torch.from_numpy(bid_px_12).to(device=self.device, dtype=torch.float32)
        cur_b_qty = torch.from_numpy(bid_qty_12).to(device=self.device, dtype=torch.float32)
        cur_a_px = torch.from_numpy(ask_px_12).to(device=self.device, dtype=torch.float32)
        cur_a_qty = torch.from_numpy(ask_qty_12).to(device=self.device, dtype=torch.float32)

        ctx_4d_b = torch.tanh(strat_raw[26:30])
        ctx_4d_a = torch.tanh(strat_raw[30:34])

        had_b_pre = cur_b_qty > 1e-6
        had_a_pre = cur_a_qty > 1e-6

        if bid_act_12 is not None and now_ts > 0:
            b_act_f64 = np.asarray(bid_act_12, dtype=np.float64)
            b_dt_f64 = np.where(b_act_f64 > 0.0, np.maximum(0.0, float(now_ts) - b_act_f64), 0.0)
            b_dt_t = torch.from_numpy(b_dt_f64.astype(np.float32)).to(device=self.device)
            age_b_norm = torch.where(had_b_pre, torch.log1p(b_dt_t) / 6.0, torch.zeros_like(cur_b_qty))
        else:
            b_dt_t = None
            age_b_norm = torch.zeros(12, dtype=torch.float32, device=self.device)

        if ask_act_12 is not None and now_ts > 0:
            a_act_f64 = np.asarray(ask_act_12, dtype=np.float64)
            a_dt_f64 = np.where(a_act_f64 > 0.0, np.maximum(0.0, float(now_ts) - a_act_f64), 0.0)
            a_dt_t = torch.from_numpy(a_dt_f64.astype(np.float32)).to(device=self.device)
            age_a_norm = torch.where(had_a_pre, torch.log1p(a_dt_t) / 6.0, torch.zeros_like(cur_a_qty))
        else:
            a_dt_t = None
            age_a_norm = torch.zeros(12, dtype=torch.float32, device=self.device)

        bud_1d = max(25.0, tot_bud_eff)
        pos_1d = max(2.0, pos_qty)
        cur_val_b = torch.clamp((cur_b_qty * cur_b_px) / bud_1d, 0.0, 1.5)
        targ_val_b = torch.clamp((q_target_b * t_bid_px) / bud_1d, 0.0, 1.5)
        cur_val_a = torch.clamp(cur_a_qty / pos_1d, 0.0, 1.5)
        targ_val_a = torch.clamp(q_target_a / pos_1d, 0.0, 1.5)

        rel_delta_b = torch.clamp((q_target_b - cur_b_qty) / torch.maximum(torch.maximum(cur_b_qty, q_target_b), torch.full_like(cur_b_qty, 2.0)), -2.0, 2.0)
        rel_delta_a = torch.clamp((q_target_a - cur_a_qty) / torch.maximum(torch.maximum(cur_a_qty, q_target_a), torch.full_like(cur_a_qty, 2.0)), -2.0, 2.0)

        drift_b = torch.where(had_b_pre, torch.clamp(((t_bid_px - cur_b_px) / b1) * 25.0, -2.0, 2.0), torch.zeros_like(cur_b_qty))
        drift_a = torch.where(had_a_pre, torch.clamp(((cur_a_px - t_ask_px) / a1) * 25.0, -2.0, 2.0), torch.zeros_like(cur_a_qty))

        cur_dist_b = torch.where(had_b_pre, torch.clamp(((b1 - cur_b_px) / b1) * 12.0, -0.5, 3.0), torch.zeros_like(cur_b_qty))
        cur_dist_a = torch.where(had_a_pre, torch.clamp(((cur_a_px - a1) / a1) * 12.0, -0.5, 3.0), torch.zeros_like(cur_a_qty))
        if b_dt_t is not None:
            b_age_s = torch.where(had_b_pre, torch.clamp(b_dt_t / 60.0, 0.0, 2.0), torch.zeros_like(cur_b_qty))
        else:
            b_age_s = torch.zeros(12, dtype=torch.float32, device=self.device)

        if a_dt_t is not None:
            a_age_s = torch.where(had_a_pre, torch.clamp(a_dt_t / 60.0, 0.0, 2.0), torch.zeros_like(cur_a_qty))
        else:
            a_age_s = torch.zeros(12, dtype=torch.float32, device=self.device)

        depth_b_norm = torch.clamp(((b1 - t_bid_px) / b1) * 8.0, -0.2, 2.5)
        depth_a_norm = torch.clamp(((t_ask_px - a1) / a1) * 8.0, -0.2, 2.5)
        front_b_fl = (t_bid_px >= b1 - 0.5 * tick_sz).float()
        front_a_fl = (t_ask_px <= a1 + 0.5 * tick_sz).float()
        unres_cash_1d = max(0.0, cash - float((cur_b_qty * cur_b_px * (1.0 + maker_fee)).sum().item()))
        free_b_2d = torch.full((12,), float(min(1.0, unres_cash_1d / max(25.0, budget))), device=self.device)
        free_a_2d = torch.full((12,), float(torch.clamp((pos_qty - cur_a_qty.sum()) / max(2.0, pos_qty), 0.0, 1.0).item()), device=self.device)

        tot_side_b = torch.full((12,), float(cur_val_b.sum().item()), device=self.device)
        tot_side_a = torch.full((12,), float(cur_val_a.sum().item()), device=self.device)
        eye_12 = torch.eye(12, dtype=torch.bool, device=self.device)
        near_b_m = (torch.abs(cur_b_px.unsqueeze(1) - t_bid_px.unsqueeze(0)) <= 1.1 * tick_sz) & had_b_pre.unsqueeze(0) & (~eye_12)
        near_a_m = (torch.abs(cur_a_px.unsqueeze(1) - t_ask_px.unsqueeze(0)) <= 1.1 * tick_sz) & had_a_pre.unsqueeze(0) & (~eye_12)
        neigh_val_b = torch.clamp(near_b_m.float() @ cur_val_b, 0.0, 1.5)
        neigh_val_a = torch.clamp(near_a_m.float() @ cur_val_a, 0.0, 1.5)
        had_b_cnt = had_b_pre.float().sum()
        mean_age_b = torch.full((12,), float(torch.where(had_b_cnt > 0.5, (b_age_s * had_b_pre.float()).sum() / max(1.0, float(had_b_cnt.item())), torch.tensor(0.0, device=self.device)).item()), device=self.device)
        had_a_cnt = had_a_pre.float().sum()
        mean_age_a = torch.full((12,), float(torch.where(had_a_cnt > 0.5, (a_age_s * had_a_pre.float()).sum() / max(1.0, float(had_a_cnt.item())), torch.tensor(0.0, device=self.device)).item()), device=self.device)
        api_ratio_2d = torch.full((12,), float(min(1.0, self.api_tokens / TOKEN_BUCKET_CAP_ORDER3)), device=self.device)
        ech_2d = self.echelon_k01
        spread = max(tick_sz, a1 - b1)
        sp_rel_2d = torch.full((12,), float(min(3.0, (spread / b1) * 40.0)), device=self.device)
        g_mode_12 = g_mode.expand(12, self.k_modes)
        o1_slot_imp = torch.tanh(0.35 * (zm[:, 34:35] + zm[:, 36:37]) + 0.40 * (zm[:, 61:62] + zm[:, 62:63])).squeeze(0)
        slot_edge_b = torch.clamp((((a1 - t_bid_px) / torch.clamp(t_bid_px, min=tick_sz)) - 2.0 * maker_fee) * 40.0, -2.0, 3.0)
        slot_exec_b = (0.65 * zm[:, 48:49].squeeze(0) + 0.55 * o1_slot_imp).expand(12)
        if pos_qty >= 2.0 and trader_entry > 1e-4:
            slot_edge_a = torch.clamp((((t_ask_px - float(trader_entry)) / max(tick_sz, float(trader_entry))) - 2.0 * maker_fee) * 40.0, -2.5, 3.0)
        else:
            slot_edge_a = torch.zeros_like(t_ask_px)
        slot_exec_a = (0.65 * zm[:, 45:46].squeeze(0) - 0.55 * o1_slot_imp).expand(12)

        # 4D датчики реальной FIFO-очереди и долгого удержания (1-в-1 с fused_order3_clean_tick.py):
        bqa_t = torch.from_numpy(bid_qa_12).to(device=self.device, dtype=torch.float32) if bid_qa_12 is not None else torch.zeros_like(cur_b_qty)
        bqa0_t = torch.from_numpy(bid_qa0_12).to(device=self.device, dtype=torch.float32) if bid_qa0_12 is not None else torch.zeros_like(cur_b_qty)
        aqa_t = torch.from_numpy(ask_qa_12).to(device=self.device, dtype=torch.float32) if ask_qa_12 is not None else torch.zeros_like(cur_a_qty)
        aqa0_t = torch.from_numpy(ask_qa0_12).to(device=self.device, dtype=torch.float32) if ask_qa0_12 is not None else torch.zeros_like(cur_a_qty)
        qprog_b = torch.where(had_b_pre, torch.clamp(1.0 - bqa_t / (bqa0_t + 10.0), 0.0, 1.0), torch.zeros_like(cur_b_qty))
        qprog_a = torch.where(had_a_pre, torch.clamp(1.0 - aqa_t / (aqa0_t + 10.0), 0.0, 1.0), torch.zeros_like(cur_a_qty))
        qa_eta_b = torch.where(had_b_pre, torch.clamp(torch.log1p(bqa_t / max(10.0, float(roll_sell_1s))) / 4.0, 0.0, 2.5), torch.zeros_like(cur_b_qty))
        qa_eta_a = torch.where(had_a_pre, torch.clamp(torch.log1p(aqa_t / max(10.0, float(roll_buy_1s))) / 4.0, 0.0, 2.5), torch.zeros_like(cur_a_qty))

        if bp_10 is not None and bq_10 is not None:
            bp10_t = torch.as_tensor(bp_10, dtype=torch.float32, device=self.device).view(10)
            bq10_t = torch.as_tensor(bq_10, dtype=torch.float32, device=self.device).view(10)
            diff_b = torch.abs(bp10_t.unsqueeze(0) - t_bid_px.unsqueeze(1)) < (0.45 * tick_sz)
            t_vis_bq = torch.where(diff_b, bq10_t.unsqueeze(0), torch.zeros_like(bq10_t.unsqueeze(0))).sum(dim=1)
        else:
            t_vis_bq = torch.zeros_like(t_bid_px)
        if ap_10 is not None and aq_10 is not None:
            ap10_t = torch.as_tensor(ap_10, dtype=torch.float32, device=self.device).view(10)
            aq10_t = torch.as_tensor(aq_10, dtype=torch.float32, device=self.device).view(10)
            diff_a = torch.abs(ap10_t.unsqueeze(0) - t_ask_px.unsqueeze(1)) < (0.45 * tick_sz)
            t_vis_aq = torch.where(diff_a, aq10_t.unsqueeze(0), torch.zeros_like(aq10_t.unsqueeze(0))).sum(dim=1)
        else:
            t_vis_aq = torch.zeros_like(t_ask_px)

        wall_b_log = torch.clamp(torch.log1p((t_vis_bq * t_bid_px) / bud_1d) / 2.0, 0.0, 2.5)
        wall_a_log = torch.clamp(torch.log1p((t_vis_aq * t_ask_px) / bud_1d) / 2.0, 0.0, 2.5)
        hl_val = min(2.0, math.log1p(max(0.0, float(hold_sec)) / 60.0) / 4.0) if pos_qty >= 2.0 else 0.0
        hold_long_2d = torch.full((12,), hl_val, dtype=torch.float32, device=self.device)

        feat_b_36 = torch.cat(
            [
                torch.stack(
                    [
                        torch.zeros_like(cur_b_qty), had_b_pre.float(), cur_val_b, targ_val_b, rel_delta_b, drift_b,
                        cur_dist_b, b_age_s, age_b_norm, depth_b_norm, front_b_fl, free_b_2d,
                    ],
                    dim=-1,
                ),
                ctx_4d_b.unsqueeze(0).expand(12, 4),
                torch.stack([tot_side_b, neigh_val_b, mean_age_b, api_ratio_2d, ech_2d, sp_rel_2d, slot_edge_b, slot_exec_b, qprog_b, qa_eta_b, wall_b_log, hold_long_2d], dim=-1),
                g_mode_12,
            ],
            dim=-1,
        )

        feat_a_36 = torch.cat(
            [
                torch.stack(
                    [
                        torch.ones_like(cur_a_qty), had_a_pre.float(), cur_val_a, targ_val_a, rel_delta_a, drift_a,
                        cur_dist_a, a_age_s, age_a_norm, depth_a_norm, front_a_fl, free_a_2d,
                    ],
                    dim=-1,
                ),
                ctx_4d_a.unsqueeze(0).expand(12, 4),
                torch.stack([tot_side_a, neigh_val_a, mean_age_a, api_ratio_2d, ech_2d, sp_rel_2d, slot_edge_a, slot_exec_a, qprog_a, qa_eta_a, wall_a_log, hold_long_2d], dim=-1),
                g_mode_12,
            ],
            dim=-1,
        )

        rb_h = F.silu(feat_b_36 @ self.w_reg1 + self.b_reg1_b)
        ra_h = F.silu(feat_a_36 @ self.w_reg1 + self.b_reg1_a)
        rb_out = rb_h @ self.w_reg2 + self.b_reg2_b  # [12, 2]
        ra_out = ra_h @ self.w_reg2 + self.b_reg2_a

        g_plc_b = (rb_out[:, 0] > 0.0).cpu().numpy()
        g_cnl_b = (rb_out[:, 1] > 0.0).cpu().numpy()
        g_plc_a = (ra_out[:, 0] > 0.0).cpu().numpy()
        g_cnl_a = (ra_out[:, 1] > 0.0).cpu().numpy()

        return {
            "exp_bid_frac": exp_bid_frac,
            "renorm_blend": renorm_blend,
            "u_taker": u_taker,
            "u_taker_buy": u_taker_buy,
            "wave_hold_pct": wave_hold_pct,
            "max_slip_pct": max_slip_pct,
            "pi_bid": pi_b_t.cpu().numpy(),
            "pi_ask": pi_a_t.cpu().numpy(),
            "t_bid_px": t_bid_px.cpu().numpy(),
            "t_ask_px": t_ask_px.cpu().numpy(),
            "q_target_b": q_target_b.cpu().numpy(),
            "q_target_a": q_target_a.cpu().numpy(),
            "g_plc_b": g_plc_b,
            "g_cnl_b": g_cnl_b,
            "g_plc_a": g_plc_a,
            "g_cnl_a": g_cnl_a,
            "cnl_logit_b": rb_out[:, 1].cpu().numpy(),
            "cnl_logit_a": ra_out[:, 1].cpu().numpy(),
            "t_vis_bq": t_vis_bq.cpu().numpy(),
            "t_vis_aq": t_vis_aq.cpu().numpy(),
            "g_mode": g_mode.squeeze(0).cpu().numpy(),
            "api_tokens": self.api_tokens,
        }


_LIVE_WF_CACHE: Dict[str, Any] = {"ready": False}


def _eval_latest_wf_ubj_live(trader, b1: float, a1: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Прямой субмиллисекундный инференс свежайшего окна win_XX (.ubj) для 100% паритета каналов 29..39 на VPS."""
    raw_feats = getattr(trader, "last_features", None)
    if not raw_feats or not isinstance(raw_feats, dict):
        return None, None
    try:
        if not _LIVE_WF_CACHE.get("ready"):
            import json
            import xgboost as xgb

            feats_p = BASE_DIR / "data/features/selected_features.json"
            o1_dir = BASE_DIR / "models/prl/variant_a_order1"
            o2_dir = BASE_DIR / "models/prl/order2_wf"
            if not feats_p.exists() or not o1_dir.exists() or not o2_dir.exists():
                _LIVE_WF_CACHE["ready"] = "unavailable"
                return None, None
            num_cols = json.loads(feats_p.read_text())["feature_order"]
            o1_wins = sorted([d for d in o1_dir.glob("win_*") if d.is_dir()], key=lambda p: int(p.name.split("_")[1]))
            o2_wins = sorted([d for d in o2_dir.glob("win_*") if d.is_dir()], key=lambda p: int(p.name.split("_")[1]))
            if not o1_wins or not o2_wins:
                _LIVE_WF_CACHE["ready"] = "unavailable"
                return None, None
            w1_latest, w2_latest = o1_wins[-1], o2_wins[-1]
            o1_models = []
            for h in (1, 2, 5, 10, 20, 40, 80, 160):
                mf = w1_latest / f"xgb_h{h}s.ubj"
                bst = xgb.Booster()
                bst.load_model(str(mf))
                bst.feature_names = None
                bst.set_param({"nthread": 1})
                o1_models.append(bst)
            o2_names = [
                "adverse_selection_5s", "wave_exhaustion_10s", "ask_sweep_5s",
                "guard_sweep_12s", "guard_bleed_60s", "guard_crash_300s",
                "guard_exit_30s", "exec_fill_30s", "exec_frontrun_30s",
                "carousel_success", "carousel_capacity",
            ]
            o2_models = []
            for nm in o2_names:
                mf2 = w2_latest / f"{nm}.ubj"
                bst2 = xgb.Booster()
                bst2.load_model(str(mf2))
                bst2.feature_names = None
                bst2.set_param({"nthread": 1})
                o2_models.append(bst2)
            ow_dir = BASE_DIR / "models/prl/order1_wave_wf"
            ow_models = []
            if ow_dir.exists():
                ow_wins = sorted([d for d in ow_dir.glob("win_*") if d.is_dir()], key=lambda p: int(p.name.split("_")[1]))
                if ow_wins:
                    ww_latest = ow_wins[-1]
                    for wnm in ("wave_ret_h300s", "wave_ret_h900s", "wave_ret_h1800s", "wave_ret_h3600s", "wave_launch_30m", "wave_top_15m"):
                        mfw = ww_latest / f"{wnm}.ubj"
                        if mfw.exists():
                            bstw = xgb.Booster()
                            bstw.load_model(str(mfw))
                            bstw.feature_names = None
                            bstw.set_param({"nthread": 1})
                            ow_models.append(bstw)
            _LIVE_WF_CACHE.update({
                "ready": "ok",
                "xgb": xgb,
                "num_cols": num_cols,
                "x_buf": np.zeros((1, len(num_cols)), dtype=np.float32),
                "o1_models": o1_models,
                "o2_models": o2_models,
                "ow_models": ow_models,
                "c_vals": np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            })
        if _LIVE_WF_CACHE.get("ready") != "ok":
            return None, None

        x_row = _LIVE_WF_CACHE["x_buf"]
        for ci, c in enumerate(_LIVE_WF_CACHE["num_cols"]):
            v_c = float(raw_feats.get(c, 0.0) or 0.0)
            x_row[0, ci] = v_c if math.isfinite(v_c) else 0.0
        c_vals = _LIVE_WF_CACHE["c_vals"]
        ez8 = np.zeros(8, dtype=np.float32)
        conf8 = np.zeros(8, dtype=np.float32)
        tail_skew_20s = 0.0
        alpha_48 = []
        for hi, bst in enumerate(_LIVE_WF_CACHE["o1_models"]):
            p_c = bst.inplace_predict(x_row)[0]
            ez_h = float(np.dot(p_c, c_vals))
            conf_h = float(1.0 - p_c[3])
            bias_h = float(np.sum(p_c[4:]) - np.sum(p_c[:3]))
            ent_h = float(-np.sum(p_c * np.log(np.maximum(p_c, 1e-6))))
            ez8[hi] = ez_h
            conf8[hi] = conf_h
            if hi == 4:
                tail_skew_20s = float(np.clip(((p_c[5] + p_c[6]) - (p_c[0] + p_c[1])) * 3.0, -2.0, 2.0))
            alpha_48.extend([ez_h, conf_h, bias_h, float(p_c[0]), float(p_c[6]), ent_h])

        bq10 = getattr(trader, "last_clean_bq_10", np.full(10, 10.0, dtype=np.float32))
        aq10 = getattr(trader, "last_clean_aq_10", np.full(10, 10.0, dtype=np.float32))
        bq1, aq1 = max(0.0, float(bq10[0])), max(0.0, float(aq10[0]))
        bq5, aq5 = max(0.0, float(np.sum(bq10[:5]))), max(0.0, float(np.sum(aq10[:5])))
        now_wf = time.time()
        fh = list(getattr(trader, "flow_hist_10s", []))
        fh_ts = list(getattr(trader, "flow_ts_hist_30s", []))
        buy10 = max(0.0, float(getattr(trader, "last_buy_vol_10s", 0.0)))
        sell10 = max(0.0, float(sum(x[3] for x in fh[-10:]))) if fh else 0.0

        # Точный 30-секундный буфер потока и 15-секундный сдвиг спреда (1-в-1 с walkforward_order2_trainer.py!):
        fh30 = _LIVE_WF_CACHE.setdefault("flow_hist_30s", [])
        last_fh_ts = float(_LIVE_WF_CACHE.get("last_fh_ts", 0.0))
        if fh_ts:
            for item_t in fh_ts:
                ts_f = float(item_t[0])
                if ts_f > last_fh_ts + 1e-6:
                    fh30.append((ts_f, max(0.0, float(item_t[1])), max(0.0, float(item_t[2]))))
                    last_fh_ts = ts_f
            _LIVE_WF_CACHE["last_fh_ts"] = last_fh_ts
        elif fh:
            for item_f in fh:
                ts_f = float(item_f[0])
                if ts_f > 1e9 and ts_f > last_fh_ts + 1e-6:
                    fh30.append((ts_f, max(0.0, float(item_f[2])), max(0.0, float(item_f[3]))))
                    last_fh_ts = ts_f
            _LIVE_WF_CACHE["last_fh_ts"] = last_fh_ts
        fh30[:] = [x for x in fh30 if (now_wf - x[0]) <= 30.5]
        if len(fh30) >= 3:
            buy30 = max(buy10, float(sum(x[1] for x in fh30)))
            sell30 = max(sell10, float(sum(x[2] for x in fh30)))
        else:
            buy30, sell30 = buy10 * 3.0, sell10 * 3.0

        st_mid = max(1e-4, 0.5 * (b1 + a1))
        bo_b1 = float(getattr(trader, "last_bo_b1", 0.0))
        bo_a1 = float(getattr(trader, "last_bo_a1", 0.0))
        bo_mid = 0.5 * (bo_b1 + bo_a1) if (bo_b1 > 1e-4 and bo_a1 > 1e-4) else st_mid
        arb_bps = float(np.clip((bo_mid - st_mid) / st_mid * 1e4, -100.0, 100.0))
        sp_bps = float(np.clip((a1 - b1) / st_mid * 1e4, 1.0, 150.0))
        sp_hist = _LIVE_WF_CACHE.setdefault("sp_hist_15s", [])
        sp_hist.append((now_wf, sp_bps))
        sp_hist[:] = [x for x in sp_hist if (now_wf - x[0]) <= 20.0]
        past_sp = next((x[1] for x in sp_hist if (now_wf - x[0]) >= 14.0), sp_hist[0][1])
        lob_18 = [
            math.log1p(bq1), math.log1p(aq1), math.log1p(bq5), math.log1p(aq5),
            bq1 / (bq5 + 1e-4), aq1 / (aq5 + 1e-4), sp_bps, sp_bps - past_sp,
            math.log1p(buy10), math.log1p(sell10), math.log1p(buy30), math.log1p(sell30),
            (bq1 - aq1) / (bq1 + aq1 + 1e-4), (bq5 - aq5) / (bq5 + aq5 + 1e-4),
            min(10.0, min(buy30, sell30) / (max(bq1, aq1) + 50.0)),
            min(10.0, sell30 / (bq1 + 50.0)), min(10.0, buy30 / (aq1 + 50.0)), arb_bps,
        ]
        x66 = np.array([lob_18 + alpha_48], dtype=np.float32)
        r11 = np.array([float(m.inplace_predict(x66)[0]) for m in _LIVE_WF_CACHE["o2_models"]], dtype=np.float32)
        _LIVE_WF_CACHE["last_conf8"] = conf8
        _LIVE_WF_CACHE["last_tail_skew_20s"] = tail_skew_20s
        ow_mods = _LIVE_WF_CACHE.get("ow_models") or []
        if len(ow_mods) == 6:
            _LIVE_WF_CACHE["last_wave6"] = np.array([float(m.inplace_predict(x_row)[0]) for m in ow_mods], dtype=np.float32)
        trader.last_p_adverse = float(np.clip(r11[0], 0.0, 1.0))
        trader.last_p_exhaust = float(np.clip(r11[1], 0.0, 1.0))
        trader.last_p_ask_sweep = float(np.clip(r11[2], 0.0, 1.0))
        trader.last_guard_probs = np.clip(r11[3:7], 0.0, 1.0)
        trader.last_ps, trader.last_pb, trader.last_pc, trader.last_pe = [float(v) for v in trader.last_guard_probs]
        trader.last_p_fill_real = float(np.clip(r11[7], 0.0, 1.0))
        trader.last_p_frontrun = float(np.clip(r11[8], 0.0, 1.0))
        car_p = float(np.clip(r11[9], 0.0, 1.0))
        trader.last_car_probs = [car_p, car_p]
        trader.last_car_cap_prl = max(0.0, float(r11[10]))
        return ez8, r11
    except Exception:
        return None, None


def build_live_hybrid_market_obs_40d(trader, b1: float, a1: float) -> np.ndarray:
    """
    100% побитовый расчет 44D гибридного вектора рынка (Lesson 35):
    12D SafeTrade LOB/Tape + 10D BigONE LOB/Basis + 12D Risks + 6D O1 Alpha + 4D Anti-FOMO Macro
    """
    z = np.zeros(MKT_DIM_ORDER3, dtype=np.float32)
    st_mid = max(1e-4, 0.5 * (a1 + b1))

    # 1. SafeTrade Raw LOB & Tape (12D)
    b1_q = float(getattr(trader, "last_b1_q", 10.0))
    a1_q = float(getattr(trader, "last_a1_q", 10.0))
    bids_top = float(getattr(trader, "last_bids_top3_q", b1_q))
    asks_top = float(getattr(trader, "last_asks_top3_q", a1_q))

    z[0] = min(0.05, max(0.0, (a1 - b1) / st_mid))
    z[1] = np.clip((b1_q - a1_q) / (b1_q + a1_q + 1e-5), -1.0, 1.0)
    z[2] = np.clip(float(getattr(trader, "last_imb3", (bids_top - asks_top) / (bids_top + asks_top + 1e-5))), -1.0, 1.0)
    z[3] = np.clip(float(getattr(trader, "last_imb5_exec", getattr(trader, "last_imb5", 0.0))), -1.0, 1.0)
    z[4] = np.clip(float(getattr(trader, "last_imb10", z[3])), -1.0, 1.0)
    z[5] = math.log1p(max(0.0, b1_q)) / 10.0
    z[6] = math.log1p(max(0.0, a1_q)) / 10.0
    z[7] = math.log1p(max(0.0, bids_top)) / 10.0
    z[8] = math.log1p(max(0.0, asks_top)) / 10.0

    fl30 = float(getattr(trader, "last_flow_30s", getattr(trader, "roll_flow_30", 0.0)))
    z[9] = float(getattr(trader, "last_flow_15s", fl30))
    z[10] = fl30
    z[11] = float(getattr(trader, "last_flow_60s", fl30))

    # 2. BigONE Raw LOB & Cross-Venue Basis (10D)
    bo_b1 = float(getattr(trader, "last_bo_b1", 0.0))
    bo_a1 = float(getattr(trader, "last_bo_a1", 0.0))
    bo_sec_age = float(getattr(trader, "last_bo_age_sec", getattr(trader, "bo_sec_age", 99.0)))
    if bo_b1 > 1e-4 and bo_a1 > 1e-4:
        bo_mid = 0.5 * (bo_b1 + bo_a1)
        bo_basis = np.clip(((bo_mid - st_mid) / st_mid) * 10.0, -2.0, 2.0)
        z[12] = bo_basis
        z[13] = min(0.05, max(0.0, (bo_a1 - bo_b1) / bo_mid))
        z[14] = np.clip(float(getattr(trader, "last_bo_imb1", 0.0)), -1.0, 1.0)
        z[15] = np.clip(float(getattr(trader, "last_bo_imb3", z[14])), -1.0, 1.0)
        z[16] = math.log1p(float(getattr(trader, "last_bo_b1_q", 10.0))) / 10.0
        z[17] = math.log1p(float(getattr(trader, "last_bo_a1_q", 10.0))) / 10.0
        z[18] = float(getattr(trader, "last_bo_flow_30s", 0.0))
        z[19] = min(1.0, max(0.0, bo_sec_age / 60.0))
        z[20] = np.sign(bo_basis) * np.sqrt(np.abs(bo_basis))
        z[21] = float(getattr(trader, "last_bo_ret_30s", 0.0))

    # 3. Strategic Multi-Horizon Signals (12D — 100% формульный паритет с build_hybrid_market_obs_40d.py)
    z[22] = float(getattr(trader, "last_ret_5s", 0.0))
    z[23] = float(getattr(trader, "last_ret_15s", 0.0))
    z[24] = float(getattr(trader, "last_ret_30s", 0.0))
    z[25] = float(getattr(trader, "last_ret_60s", 0.0))
    z[26] = float(getattr(trader, "last_ret_120s", 0.0))
    z[27] = float(getattr(trader, "last_vol_10s", 0.0))
    z[28] = float(np.clip(z[24] - z[26], -2.0, 2.0))

    p_sw = float(getattr(trader, "last_ps", 0.0))
    p_bl = float(getattr(trader, "last_pb", 0.0))
    p_cr = float(getattr(trader, "last_pc", 0.0))
    p_ex = float(getattr(trader, "last_pe", 0.0))
    z[29] = p_sw
    z[30] = p_bl
    z[31] = p_cr
    z[32] = p_ex
    z[33] = float(getattr(trader, "last_p_adverse", np.clip(0.4 * p_sw + 0.3 * p_bl + 0.3 * p_cr, 0.0, 1.0)))

    # 4. Multi-Horizon Causal Alpha Summary (6D) — прямой инференс последнего окна variant_a_order1 / order2_wf (.ubj) при наличии фич!
    wf_o1_ez, wf_o2_r = _eval_latest_wf_ubj_live(trader, b1, a1)
    if wf_o2_r is not None:
        z[29] = float(np.clip(wf_o2_r[3], 0.0, 1.0))  # guard_sweep_12s
        z[30] = float(np.clip(wf_o2_r[4], 0.0, 1.0))  # guard_bleed_60s
        z[31] = float(np.clip(wf_o2_r[5], 0.0, 1.0))  # guard_crash_300s
        z[32] = float(np.clip(wf_o2_r[6], 0.0, 1.0))  # guard_exit_30s
        z[33] = float(np.clip(wf_o2_r[0], 0.0, 1.0))  # adverse_selection_5s

    if wf_o1_ez is not None:
        z[34:40] = np.clip(wf_o1_ez[2:8], -3.0, 3.0)
    else:
        ez_5 = float(getattr(trader, "last_ez5", 0.0))
        ez_12 = float(getattr(trader, "last_ez12", 0.0))
        ez_25 = float(getattr(trader, "last_ez25", 0.0))
        ez_60 = float(getattr(trader, "last_ez60", ez_25))
        ez_90 = float(getattr(trader, "last_ez90", ez_60))
        ez_120 = float(getattr(trader, "last_ez120", ez_90))
        z[34] = float(np.clip(ez_5, -3.0, 3.0))
        z[35] = float(np.clip(ez_12, -3.0, 3.0))
        z[36] = float(np.clip(ez_25, -3.0, 3.0))
        z[37] = float(np.clip(ez_60, -3.0, 3.0))
        z[38] = float(np.clip(ez_90, -3.0, 3.0))
        z[39] = float(np.clip(ez_120, -3.0, 3.0))

    # 5. Anti-FOMO Macro Regime Channels (4D: ret_15m, ret_1h, range_pos_4h [-1..+1], range_amp_4h [0..2.5])
    if MKT_DIM_ORDER3 >= 44:
        z[40] = float(np.clip(getattr(trader, "last_ret_15m", z[26]), -2.5, 2.5))
        z[41] = float(np.clip(getattr(trader, "last_ret_1h", z[40]), -2.5, 2.5))
        pos_4h = float(getattr(trader, "last_pos14400", 0.5))
        z[42] = float(np.clip(getattr(trader, "last_range_pos_4h", 2.0 * pos_4h - 1.0), -1.0, 1.0))
        z[43] = float(np.clip(getattr(trader, "last_range_amp_4h", 0.05), 0.0, 2.5))

    # 6. Signal Stack v2 (Lesson 41: [44..49] 6D Order-2 Exec/Carousel + [50..55] 6D Order-1 Dist + [56..59] 4D Micro/Cross)
    if MKT_DIM_ORDER3 >= 60:
        if wf_o2_r is not None:
            z[44] = float(np.clip(wf_o2_r[1], 0.0, 1.0))  # wave_exhaustion_10s
            z[45] = float(np.clip(wf_o2_r[2], 0.0, 1.0))  # ask_sweep_5s
            z[46] = float(np.clip(wf_o2_r[7], 0.0, 1.0))  # exec_fill_30s
            z[47] = float(np.clip(wf_o2_r[8], 0.0, 1.0))  # exec_frontrun_30s
            z[48] = float(np.clip(wf_o2_r[9], 0.0, 1.0))  # carousel_success
            z[49] = float(np.clip(math.log1p(max(0.0, float(wf_o2_r[10]))) / 3.0, 0.0, 2.5))
        else:
            z[44] = float(np.clip(getattr(trader, "last_p_exhaust", 0.08), 0.0, 1.0))
            z[45] = float(np.clip(getattr(trader, "last_p_ask_sweep", 0.12), 0.0, 1.0))
            z[46] = float(np.clip(getattr(trader, "last_p_fill_real", 0.50), 0.0, 1.0))
            z[47] = float(np.clip(getattr(trader, "last_p_frontrun", 0.08), 0.0, 1.0))
            car_p = getattr(trader, "last_car_probs", [0.47])
            z[48] = float(np.clip(car_p[0] if len(car_p) > 0 else 0.47, 0.0, 1.0))
            z[49] = float(np.clip(math.log1p(max(0.0, float(getattr(trader, "last_car_cap_prl", 2.0)))) / 3.0, 0.0, 2.5))

        conf8_l = _LIVE_WF_CACHE.get("last_conf8")
        if wf_o1_ez is not None and conf8_l is not None:
            z[50] = float(np.clip(wf_o1_ez[0], -3.0, 3.0))
            z[51] = float(np.clip(wf_o1_ez[1], -3.0, 3.0))
            z[52] = float(np.clip(conf8_l[3], 0.0, 1.0))
            z[53] = float(np.clip(conf8_l[5], 0.0, 1.0))
            z[54] = float(np.clip(conf8_l[7], 0.0, 1.0))
            z[55] = float(_LIVE_WF_CACHE.get("last_tail_skew_20s", 0.0))
        else:
            z[50] = float(np.clip(z[34] * 0.5, -3.0, 3.0))
            z[51] = float(np.clip(z[34] * 0.75, -3.0, 3.0))
            z[52] = 0.45
            z[53] = 0.69
            z[54] = 0.80
            z[55] = 0.0

        rf = getattr(trader, "last_features", None) or {}
        ofi_r = float(rf.get("ST_ofi_norm_5s", 0.0) or 0.0)
        slp_r = float(rf.get("ST_log_slope_ratio", 0.0) or 0.0)
        gt_m = float(rf.get("GATE_mid_price", 0.0) or 0.0)
        gt_s = float(rf.get("GATE_sec_since_tick", 999.0) or 999.0)
        ou_m = float(rf.get("OUR_mid_price", 0.0) or 0.0)
        ou_s = float(rf.get("OUR_sec_since_tick", 999.0) or 999.0)
        z[56] = float(math.tanh(ofi_r * 8.0))
        z[57] = float(math.tanh(slp_r * 0.40))
        z[58] = float(math.tanh(((gt_m - st_mid) / st_mid) * 50.0)) if (gt_m > 1e-4 and gt_s <= 30.0) else 0.0
        z[59] = float(math.tanh(((ou_m - st_mid) / st_mid) * 50.0)) if (ou_m > 1e-4 and ou_s <= 30.0) else 0.0

    if MKT_DIM_ORDER3 >= 72:
        w6_live = _LIVE_WF_CACHE.get("last_wave6")
        if w6_live is not None and len(w6_live) == 6:
            z[60] = float(math.tanh(float(w6_live[0]) * 1.5))
            z[61] = float(math.tanh(float(w6_live[1]) * 0.85))
            z[62] = float(math.tanh(float(w6_live[2]) * 0.50))
            z[63] = float(math.tanh(float(w6_live[3]) * 0.30))
            z[64] = float(np.clip(w6_live[4], 0.0, 1.0))
            z[65] = float(np.clip(w6_live[5], 0.0, 1.0))
        else:
            z[64] = 0.25
            z[65] = 0.25
        now_m = time.time()
        mh = _LIVE_WF_CACHE.setdefault("mid_hist_3600s", [])
        if not mh:
            bh_warm = list(getattr(trader, "bid_hist_1h", []))[-3600:]
            if len(bh_warm) >= 10:
                sp_half = max(0.0, 0.5 * (a1 - b1))
                n_bh = len(bh_warm)
                mh.extend((now_m - float(n_bh - i), float(v) + sp_half) for i, v in enumerate(bh_warm))
        mh.append((now_m, st_mid))
        if len(mh) > 4000 or (mh and (now_m - mh[0][0]) > 3660.0):
            mh[:] = [x for x in mh if (now_m - x[0]) <= 3605.0]
        m_5m = next((x[1] for x in reversed(mh) if (now_m - x[0]) >= 295.0), mh[0][1])
        m_15m = next((x[1] for x in reversed(mh) if (now_m - x[0]) >= 895.0), mh[0][1])
        m_60m = mh[0][1]
        vals_30m = [x[1] for x in mh if (now_m - x[0]) <= 1805.0] or [st_mid]
        vals_60m = [x[1] for x in mh] or [st_mid]
        mn_1800, mx_1800 = min(vals_30m), max(vals_30m)
        mn_3600, mx_3600 = min(vals_60m), max(vals_60m)
        rng_1h = max(mx_3600 - mn_3600, st_mid * 0.005)
        z[66] = float(math.tanh((st_mid / max(m_5m, 1e-4) - 1.0) * 25.0))
        z[67] = float(math.tanh((st_mid / max(m_15m, 1e-4) - 1.0) * 16.0))
        z[68] = float(math.tanh((st_mid / max(m_60m, 1e-4) - 1.0) * 10.0))
        z[69] = float(np.clip(((st_mid - mn_3600) / rng_1h) * 2.0 - 1.0, -1.0, 1.0))
        z[70] = float(math.tanh(((st_mid - mn_1800) / st_mid) * 25.0))
        z[71] = float(math.tanh(((mx_1800 - st_mid) / st_mid) * 25.0))

    return z


def execute_live_24slot_order3(trader, b1: float, a1: float) -> None:
    """
    Боевое исполнение 24 слотов с непрерывным Order-3 Champion:
    - 0 жестких порогов, непрерывный Token-Bucket лимитер (+0.35/s, cap 10.0)
    - Послотовый нейро-регулятор (12 Bid / 12 Ask)
    - Поддержка субцентового тика $0.0001
    """
    if getattr(trader, "order3_engine", None) is None:
        trader.order3_engine = Order3LiveInferenceEngine(device="cpu")

    now_ts = time.time()
    trader.order3_engine.refill_tokens(now_ts)

    # 1. Извлечение 39D вектора рынка и 110D вектора агента из трейдера
    tot_eq = max(float(getattr(trader, "initial_budget_usd", 1022.54)), float(getattr(trader, "bot_cash_usd", 0.0)) + float(getattr(trader, "pos_qty", 0.0)) * b1)

    # Подготовка текущих массивов слотов
    b_px = np.array([float(s.get("px", 0.0)) for s in trader.bid_slots], dtype=np.float32)
    b_qty = np.array([float(s.get("qty", 0.0)) for s in trader.bid_slots], dtype=np.float32)
    a_px = np.array([float(s.get("px", 0.0)) for s in trader.ask_slots], dtype=np.float32)
    a_qty = np.array([float(s.get("qty", 0.0)) for s in trader.ask_slots], dtype=np.float32)

    z_mkt = build_live_hybrid_market_obs_40d(trader, b1, a1)
    if not getattr(trader.order3_engine, "_mem_warmed", False):
        with torch.no_grad():
            eng_w = trader.order3_engine
            zm_w = _scale_mkt_44d(torch.from_numpy(z_mkt).to(device=eng_w.device, dtype=torch.float32).view(1, MKT_DIM_ORDER3), torch.float32)
            pw = zm_w @ eng_w.w1_mkt + eng_w.b1
            h1w = torch.relu(pw[:, :256]) * (2.0 * torch.sigmoid(pw[:, 256:384])).repeat_interleave(2, dim=-1)
            rpw = h1w @ eng_w.w_res1 + eng_w.b_res1
            h2w = h1w + (torch.relu(rpw[:, :64]) * (2.0 * torch.sigmoid(rpw[:, 64:128]))) @ eng_w.w_res2
            rrw = h2w @ eng_w.w_rec + eng_w.b_rec
            eng_w.mem_h.copy_(torch.tanh(rrw[:, :MEM_DIM_ORDER3]))
            eng_w._mem_warmed = True

    z_ag = getattr(trader, "last_order3_ag_110d", None)
    if z_ag is None or len(z_ag) != AG_DIM_ORDER3:
        z_ag = np.zeros(AG_DIM_ORDER3, dtype=np.float32)
        b1_q_live = max(0.0, float(getattr(trader, "last_b1_q", 10.0)))
        a1_q_live = max(0.0, float(getattr(trader, "last_a1_q", 10.0)))
        l1_bid_usd = max(100.0, b1 * b1_q_live)
        l1_ask_usd = max(100.0, a1 * a1_q_live)
        res_b_usd_ag = float(np.sum(b_qty * b_px * 1.001))
        unres_cash_ag = max(0.0, float(trader.bot_cash_usd) - res_b_usd_ag)
        roll_buy_1s = max(10.0, float(getattr(trader, "last_buy_vol_10s", 100.0)) / 10.0)
        low_tr_q = float(getattr(trader, "tr_low_qty", getattr(trader, "low_tranche_qty", 0.0)))
        raw_min_p = float(getattr(trader, "tr_low_px", getattr(trader, "min_buy_px", 0.0)))
        min_buy_p = raw_min_p if raw_min_p > 1e-4 else float(trader.avg_entry)

        t_sz_ag = max(1e-6, float(getattr(trader, "exec_tick_size", trader.tick_size)))
        m_fee_ag = 0.0010
        has_p_ag = float(trader.pos_qty) > 1e-6
        pos_rat_ag = float(np.clip((float(trader.pos_qty) * b1) / max(tot_eq, 1.0), 0.0, 1.5))
        eff_ps_ag = float(np.clip(pos_rat_ag / 0.25, 0.15, 1.0))

        z_ag[0] = math.log1p(tot_eq / l1_bid_usd) / 3.0
        z_ag[1] = math.log1p(unres_cash_ag / l1_bid_usd) / 3.0
        z_ag[2] = min(4.0, (trader.pos_qty * a1) / l1_ask_usd) / 2.0
        z_ag[3] = min(10.0, trader.pos_qty / roll_buy_1s) / 5.0
        z_ag[4] = float(np.clip(low_tr_q / max(trader.pos_qty, 1e-6), 0.0, 1.0)) if has_p_ag else 0.0
        z_ag[5] = float(np.clip((((b1 - min_buy_p) / max(min_buy_p, t_sz_ag)) - 2.0 * m_fee_ag) * 30.0, -2.5, 2.5)) if (has_p_ag and min_buy_p > 0) else 0.0
        z_ag[6:18] = np.clip((b_qty * b1) / max(tot_eq, 25.0), 0.0, 1.0)
        z_ag[18:30] = np.clip(a_qty / max(trader.pos_qty, 2.0), 0.0, 1.0)
        z_ag[30:42] = np.where(b_qty > 1e-6, np.log1p(np.maximum(0.0, (b1 - b_px) / t_sz_ag)) / 3.5, 0.0)
        z_ag[42:54] = np.where(a_qty > 1e-6, np.log1p(np.maximum(0.0, (a_px - a1) / t_sz_ag)) / 3.5, 0.0)
        z_ag[54] = 1.0 if has_p_ag else 0.0
        z_ag[55] = pos_rat_ag
        z_ag[56] = float(np.clip(unres_cash_ag / max(tot_eq, 1.0), 0.0, 1.0))
        z_ag[57] = float(np.clip(((((b1 - trader.avg_entry) / max(t_sz_ag, trader.avg_entry)) - 2.0 * m_fee_ag) * 25.0 * eff_ps_ag), -2.5, 2.5)) if (has_p_ag and trader.avg_entry > 0) else 0.0
        z_ag[58] = (min(2.0, max(0.0, trader.hold_sec / 300.0)) * eff_ps_ag) if has_p_ag else 0.0
        z_ag[59] = 1.0 if np.any(b_qty > 1e-6) else 0.0
        z_ag[60] = 1.0 if np.any(a_qty > 1e-6) else 0.0
        z_ag[61] = (np.sum(b_qty > 1e-6) - np.sum(a_qty > 1e-6)) / 12.0
        b_act_ts = np.array([float(s.get("act_ts", 0.0)) for s in trader.bid_slots], dtype=np.float64)
        a_act_ts = np.array([float(s.get("act_ts", 0.0)) for s in trader.ask_slots], dtype=np.float64)

        has_b = np.any(b_qty > 1e-6)
        has_a = np.any(a_qty > 1e-6)
        best_b_idx = int(np.argmin(np.where(b_qty > 1e-6, b1 - b_px, 999.0))) if has_b else 0
        best_a_idx = int(np.argmin(np.where(a_qty > 1e-6, a_px - a1, 999.0))) if has_a else 0
        best_b_act = float(b_act_ts[best_b_idx])
        best_a_act = float(a_act_ts[best_a_idx])
        best_b_age = min(2.0, max(0.0, (now_ts - best_b_act) / 60.0)) if (has_b and best_b_act > 0) else 0.0
        best_a_age = min(2.0, max(0.0, (now_ts - best_a_act) / 60.0)) if (has_a and best_a_act > 0) else 0.0

        zm_sc = _scale_mkt_44d(torch.from_numpy(z_mkt).float(), torch.float32).numpy()
        z_ag[62] = best_b_age
        z_ag[63] = best_a_age
        z_ag[64] = float(zm_sc[12]) if len(zm_sc) > 12 else 0.0
        z_ag[65] = float(zm_sc[13]) if len(zm_sc) > 13 else 0.0
        z_ag[66] = float(zm_sc[10]) if len(zm_sc) > 10 else 0.0
        z_ag[67] = float(zm_sc[3]) if len(zm_sc) > 3 else 0.0
        z_ag[68] = float(zm_sc[44]) if len(zm_sc) > 44 else 0.0
        z_ag[69] = float(zm_sc[48]) if len(zm_sc) > 48 else 0.0

        z_ag[70:86] = trader.order3_engine.mem_h[:, :16].squeeze(0).cpu().numpy()
        z_ag[86:98] = np.where(b_qty > 1e-6, np.clip(np.where(b_act_ts > 0.0, (float(now_ts) - b_act_ts) / 60.0, 0.0), 0.0, 2.0), -1.0).astype(np.float32)
        z_ag[98:110] = np.where(a_qty > 1e-6, np.clip(np.where(a_act_ts > 0.0, (float(now_ts) - a_act_ts) / 60.0, 0.0), 0.0, 2.0), -1.0).astype(np.float32)

    else:
        b_act_ts = np.array([float(s.get("act_ts", 0.0)) for s in trader.bid_slots], dtype=np.float64)
        a_act_ts = np.array([float(s.get("act_ts", 0.0)) for s in trader.ask_slots], dtype=np.float64)

    # Синхронизация живой FIFO-очереди слотов (qa / qa0) по чистому 10-уровневому стакану (1-в-1 с fused_order3_clean_tick.py):
    bp10_live = np.asarray(getattr(trader, "last_clean_bp_10", [b1] * 10), dtype=np.float32)[:10]
    ap10_live = np.asarray(getattr(trader, "last_clean_ap_10", [a1] * 10), dtype=np.float32)[:10]
    bq10_live = np.asarray(getattr(trader, "last_clean_bq_10", np.zeros(10, dtype=np.float32)), dtype=np.float32)[:10]
    aq10_live = np.asarray(getattr(trader, "last_clean_aq_10", np.zeros(10, dtype=np.float32)), dtype=np.float32)[:10]
    t_sz = float(getattr(trader, "exec_tick_size", trader.tick_size))
    fh_live = list(getattr(trader, "flow_hist_10s", []))
    fh_ts_live = list(getattr(trader, "flow_ts_hist_30s", []))
    rs_1s = max(10.0, float(sum(x[3] for x in fh_live[-10:])) / 10.0) if fh_live else 10.0
    rb_1s = max(10.0, float(getattr(trader, "last_buy_vol_10s", 100.0)) / 10.0)
    last_qa_step_ts = float(getattr(trader, "_last_qa_drain_ts", 0.0))
    if fh_ts_live:
        sv_since = float(sum(max(0.0, float(x[2])) for x in fh_ts_live if float(x[0]) > last_qa_step_ts + 1e-6)) if last_qa_step_ts > 0.0 else 0.0
        bv_since = float(sum(max(0.0, float(x[1])) for x in fh_ts_live if float(x[0]) > last_qa_step_ts + 1e-6)) if last_qa_step_ts > 0.0 else 0.0
        trader._last_qa_drain_ts = float(fh_ts_live[-1][0])
    elif fh_live and float(fh_live[-1][0]) > 1e9:
        sv_since = float(sum(max(0.0, float(x[3])) for x in fh_live if float(x[0]) > last_qa_step_ts + 1e-6)) if last_qa_step_ts > 0.0 else 0.0
        bv_since = float(sum(max(0.0, float(x[2])) for x in fh_live if float(x[0]) > last_qa_step_ts + 1e-6)) if last_qa_step_ts > 0.0 else 0.0
        trader._last_qa_drain_ts = float(fh_live[-1][0])
    else:
        sv_since = max(0.0, float(fh_live[-1][3])) if fh_live else 0.0
        bv_since = max(0.0, float(fh_live[-1][2])) if fh_live else 0.0

    b_qa_arr = np.zeros(12, dtype=np.float32)
    b_qa0_arr = np.zeros(12, dtype=np.float32)
    a_qa_arr = np.zeros(12, dtype=np.float32)
    a_qa0_arr = np.zeros(12, dtype=np.float32)
    for si in range(12):
        sb = trader.bid_slots[si]
        if sb.get("id") is not None and b_qty[si] > 1e-6:
            qa_cur = float(sb.get("qa", 0.0))
            qa0_cur = float(sb.get("qa0", qa_cur))
            if float(sb.get("acc_fill", 0.0)) > 1e-4:
                qa_cur = 0.0
            elif b_px[si] <= b1 + 0.1 * t_sz:
                if sv_since > 0.0 and b1 <= b_px[si] + 0.1 * t_sz:
                    above_bq = float(np.sum(bq10_live[bp10_live > (b_px[si] + 0.5 * t_sz)]))
                    qa_cur = max(0.0, qa_cur - max(0.0, sv_since - above_bq))
                m_b = np.abs(bp10_live - b_px[si]) < (0.45 * t_sz)
                vis_bq = float(np.sum(bq10_live[m_b])) if np.any(m_b) else 0.0
                qa_cur = min(qa_cur, vis_bq)
            sb["qa"] = qa_cur
            b_qa_arr[si], b_qa0_arr[si] = qa_cur, qa0_cur
        sa = trader.ask_slots[si]
        if sa.get("id") is not None and a_qty[si] > 1e-6:
            qa_cur = float(sa.get("qa", 0.0))
            qa0_cur = float(sa.get("qa0", qa_cur))
            if float(sa.get("acc_fill", 0.0)) > 1e-4:
                qa_cur = 0.0
            elif a_px[si] >= a1 - 0.1 * t_sz:
                if bv_since > 0.0 and a1 >= a_px[si] - 0.1 * t_sz:
                    below_aq = float(np.sum(aq10_live[ap10_live < (a_px[si] - 0.5 * t_sz)]))
                    qa_cur = max(0.0, qa_cur - max(0.0, bv_since - below_aq))
                m_a = np.abs(ap10_live - a_px[si]) < (0.45 * t_sz)
                vis_aq = float(np.sum(aq10_live[m_a])) if np.any(m_a) else 0.0
                qa_cur = min(qa_cur, vis_aq)
            sa["qa"] = qa_cur
            a_qa_arr[si], a_qa0_arr[si] = qa_cur, qa0_cur

    plan = trader.order3_engine.forward_tick(
        z_mkt_39d=z_mkt,
        z_ag_110d=z_ag,
        b1=b1,
        a1=a1,
        budget=tot_eq,
        pos_qty=float(trader.pos_qty),
        cash=float(trader.bot_cash_usd),
        bid_px_12=b_px,
        bid_qty_12=b_qty,
        ask_px_12=a_px,
        ask_qty_12=a_qty,
        tick_sz=t_sz,
        maker_fee=0.0010,
        trader_entry=float(trader.avg_entry),
        bid_act_12=b_act_ts,
        ask_act_12=a_act_ts,
        now_ts=now_ts,
        bid_qa_12=b_qa_arr,
        bid_qa0_12=b_qa0_arr,
        ask_qa_12=a_qa_arr,
        ask_qa0_12=a_qa0_arr,
        bp_10=bp10_live,
        bq_10=bq10_live,
        ap_10=ap10_live,
        aq_10=aq10_live,
        roll_sell_1s=rs_1s,
        roll_buy_1s=rb_1s,
        hold_sec=float(getattr(trader, "hold_sec", 0.0)),
    )

    p_prec = int(getattr(trader, "price_precision", 4))
    a_prec = int(getattr(trader, "amount_precision", 4))
    min_amt = float(getattr(trader, "min_order_amount", 2.0))

    u_taker = float(plan.get("u_taker", 0.0))
    u_taker_buy = float(plan.get("u_taker_buy", 0.0))
    max_slip_pct = float(plan.get("max_slip_pct", 0.02))
    renorm_b = float(plan.get("renorm_blend", 1.0))

    # 1B. Импульсный Taker-Buy вентиль на старте волны (1-в-1 с fused_order3_clean_tick.py):
    if u_taker_buy > 1e-4 and u_taker <= 1e-4 and trader.order3_engine.api_tokens >= 1.0:
        res_b_pre = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) * 1.001 for s in trader.bid_slots if s.get("id") is not None)
        free_c_tb = max(0.0, min(float(trader.bot_cash_usd) * 0.996 - res_b_pre, float(trader.ex_usdt_free)))
        rem_e_tb = max(0.0, tot_eq * 0.996 * float(plan["exp_bid_frac"]) - float(trader.pos_qty) * b1 - res_b_pre)
        tb_usd = min(free_c_tb, rem_e_tb) * u_taker_buy
        lim_buy_px = round(math.ceil((a1 * (1.0 + min(0.03, max_slip_pct))) / t_sz) * t_sz, p_prec)
        buy_q_tb = math.floor((tb_usd / max(1e-6, lim_buy_px * 1.002)) * 10000.0) / 10000.0
        if buy_q_tb >= min_amt:
            try:
                res_tb = trader.client.place_limit_order(trader.market, "buy", buy_q_tb, lim_buy_px, price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt)
                trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                if res_tb and res_tb.get("id") is not None:
                    free_sb = next((si for si in range(12) if trader.bid_slots[si].get("id") is None), 0)
                    trader.bid_slots[free_sb] = {"slot": free_sb, "id": res_tb.get("id"), "px": float(res_tb.get("price", lim_buy_px)), "qty": float(res_tb.get("origin_amount", buy_q_tb)), "act_ts": now_ts, "acc_fill": 0.0}
                    print(f"  🚀 [ENDOGENOUS WAVE TAKER BUY ({u_taker_buy:.0%})] id={res_tb.get('id')} | {buy_q_tb:.4f} PRL @ <=${lim_buy_px:.{p_prec}f}")
                trader._sync_24slot_orders(b1, count_hold=False)
                trader._save_wallet()
            except Exception as e_tb:
                print(f"  ⚠️ [ENDOGENOUS TAKER BUY WARN]: {e_tb}")
            return

    # 1. Эндогенный Taker-вентиль БЕЗ карательного сноса обеспеченных Ask-слотов при частичном хеджировании:
    if (float(trader.pos_qty) * u_taker) >= min_amt and trader.order3_engine.api_tokens >= 1.0:
        raw_sq = float(trader.pos_qty) * u_taker
        sell_q = float(trader.pos_qty) if (float(trader.pos_qty) - raw_sq) < min_amt else raw_sq
        rem_pos_est = max(0.0, float(trader.pos_qty) - sell_q)
        cum_aq = 0.0
        drop_ask_oids = []
        drop_ask_slots = []
        for s in range(12):
            if trader.ask_slots[s].get("id") is not None:
                sq_s = float(trader.ask_slots[s].get("qty", 0.0))
                cum_aq += sq_s
                keep_s = (rem_pos_est >= min_amt) and (cum_aq <= rem_pos_est + 1e-5) and (sq_s >= min_amt)
                if not keep_s:
                    drop_ask_oids.append(trader.ask_slots[s]["id"])
                    drop_ask_slots.append(s)
        if drop_ask_oids:
            n_open_all_a = sum(1 for s in trader.ask_slots if s.get("id") is not None)
            if len(drop_ask_oids) >= 2 and len(drop_ask_oids) == n_open_all_a:
                try:
                    raw_all_a = trader.client.cancel_all_orders(trader.market, side="sell")
                    c_m = {o.get("id"): o for o in (raw_all_a if isinstance(raw_all_a, list) else []) if isinstance(o, dict) and o.get("id") is not None}
                    trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                except Exception:
                    raw_c = list(trader._io_pool.map(trader._safe_cancel_oid, drop_ask_oids))
                    c_m = {oid: rc for oid, rc in zip(drop_ask_oids, raw_c) if isinstance(rc, dict)}
                    trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - float(len(drop_ask_oids)))
            else:
                raw_c = list(trader._io_pool.map(trader._safe_cancel_oid, drop_ask_oids))
                c_m = {oid: rc for oid, rc in zip(drop_ask_oids, raw_c) if isinstance(rc, dict)}
                trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - float(len(drop_ask_oids)))
            trader._sync_24slot_orders(b1, count_hold=False, pre_fetched_orders=c_m)
            for s in drop_ask_slots:
                if trader.ask_slots[s].get("id") in drop_ask_oids:
                    trader.ask_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
        sell_q = math.floor(min(sell_q, float(getattr(trader, "ex_coin_free", trader.pos_qty)) or float(trader.pos_qty)) * 10000.0) / 10000.0
        if sell_q >= min_amt:
            lim_cross_px = round(max(t_sz, math.floor((b1 * (1.0 - max_slip_pct)) / t_sz) * t_sz), p_prec)
            try:
                res_t = trader.client.place_limit_order(trader.market, "sell", sell_q, lim_cross_px, price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt)
                trader.order3_engine.api_tokens = max(0.0, trader.order3_engine.api_tokens - 1.0)
                if res_t and res_t.get("id") is not None:
                    free_s = next((si for si in range(12) if trader.ask_slots[si].get("id") is None), 0)
                    trader.ask_slots[free_s] = {"slot": free_s, "id": res_t.get("id"), "px": float(res_t.get("price", lim_cross_px)), "qty": float(res_t.get("origin_amount", sell_q)), "act_ts": now_ts, "acc_fill": 0.0}
                    print(f"  ⚡ [ENDOGENOUS TAKER EXIT ({u_taker:.0%})] id={res_t.get('id')} | {sell_q:.4f} PRL @ >=${lim_cross_px:.{p_prec}f}")
                trader._sync_24slot_orders(b1, count_hold=False)
                trader._save_wallet()
            except Exception as e_tk:
                print(f"  ⚠️ [ENDOGENOUS TAKER WARN]: {e_tk}")
            return

    # 2. Ранжированное по срочности снятие ордеров (1-в-1 с fused_order3_clean_tick.py):
    to_cnl_b, to_cnl_a = [], []
    cand_cb = []
    for s in range(12):
        if trader.bid_slots[s].get("id") is not None:
            cur_p = float(trader.bid_slots[s].get("px", 0.0))
            crossed_ask = cur_p >= (a1 - 0.1 * t_sz)
            if crossed_ask or bool(plan["g_cnl_b"][s]):
                prio = float(plan["cnl_logit_b"][s]) + (100.0 if crossed_ask else 0.0)
                cand_cb.append((prio, s, trader.bid_slots[s]["id"], crossed_ask))
    cand_cb.sort(key=lambda x: x[0], reverse=True)
    for _, s, oid, crossed_ask in cand_cb:
        if trader.order3_engine.api_tokens >= 1.0:
            to_cnl_b.append((s, oid, crossed_ask))
            trader.order3_engine.api_tokens -= 1.0

    hp_now = float(trader.pos_qty) >= min_amt
    cand_ca = []
    for s in range(12):
        if trader.ask_slots[s].get("id") is not None:
            cur_p = float(trader.ask_slots[s].get("px", 0.0))
            crossed_bid = cur_p <= (b1 + 0.1 * t_sz)
            if (not hp_now) or crossed_bid or bool(plan["g_cnl_a"][s]):
                prio = float(plan["cnl_logit_a"][s]) + (100.0 if (crossed_bid or not hp_now) else 0.0)
                cand_ca.append((prio, s, trader.ask_slots[s]["id"], crossed_bid or (not hp_now)))
    cand_ca.sort(key=lambda x: x[0], reverse=True)
    for _, s, oid, is_cr in cand_ca:
        if (not hp_now) or trader.order3_engine.api_tokens >= 1.0:
            to_cnl_a.append((s, oid, is_cr))
            if hp_now:
                trader.order3_engine.api_tokens -= 1.0

    if to_cnl_b or to_cnl_a:
        n_open_b = sum(1 for s in trader.bid_slots if s.get("id") is not None)
        n_open_a = sum(1 for s in trader.ask_slots if s.get("id") is not None)
        bulk_b = (len(to_cnl_b) >= 2) and (len(to_cnl_b) == n_open_b)
        bulk_a = (len(to_cnl_a) >= 2) and (len(to_cnl_a) == n_open_a)
        c_map = {}
        indiv_oids = []
        bulk_sides = []
        if bulk_b:
            bulk_sides.append(("buy", [x[1] for x in to_cnl_b]))
            trader.order3_engine.api_tokens = min(LIVE_TOKEN_BUCKET_CAP, trader.order3_engine.api_tokens + float(len(to_cnl_b) - 1))
        else:
            indiv_oids.extend([x[1] for x in to_cnl_b])
        if bulk_a:
            bulk_sides.append(("sell", [x[1] for x in to_cnl_a]))
            if hp_now:
                trader.order3_engine.api_tokens = min(LIVE_TOKEN_BUCKET_CAP, trader.order3_engine.api_tokens + float(len(to_cnl_a) - 1))
        else:
            indiv_oids.extend([x[1] for x in to_cnl_a])

        def _exec_cancel_op(op):
            kind, arg = op
            if kind == "bulk":
                side_s, oids_s = arg
                try:
                    res_b = trader.client.cancel_all_orders(trader.market, side=side_s)
                    return ("bulk", oids_s, res_b)
                except Exception:
                    return ("fallback", oids_s, [trader._safe_cancel_oid(x) for x in oids_s])
            return ("single", arg, trader._safe_cancel_oid(arg))

        cancel_ops = [("bulk", bs) for bs in bulk_sides] + [("single", oid) for oid in indiv_oids]
        for r_kind, r_key, r_val in trader._io_pool.map(_exec_cancel_op, cancel_ops):
            if r_kind == "single" and isinstance(r_val, dict):
                c_map[r_key] = r_val
            elif r_kind == "bulk":
                if isinstance(r_val, list):
                    for item_o in r_val:
                        if isinstance(item_o, dict) and item_o.get("id") is not None:
                            c_map[item_o["id"]] = item_o
                for oid_b in r_key:
                    c_map.setdefault(oid_b, {"id": oid_b, "state": "cancel", "filled_amount": float(next((sl.get("acc_fill", 0.0) for sl in (trader.bid_slots + trader.ask_slots) if sl.get("id") == oid_b), 0.0))})
            elif r_kind == "fallback":
                for oid_f, rc_f in zip(r_key, r_val):
                    if isinstance(rc_f, dict):
                        c_map[oid_f] = rc_f

        need_fill_sync = False

        for s, oid, is_cross in to_cnl_b:
            c_info = c_map.get(oid)
            if c_info is None:
                try:
                    c_info = trader.client.get_order(oid)
                    if isinstance(c_info, dict):
                        c_map[oid] = c_info
                except Exception:
                    c_info = None
            if isinstance(c_info, dict):
                st_c = str(c_info.get("state", "")).lower()
                prev_acc = float(trader.bid_slots[s].get("acc_fill", 0.0))
                if float(c_info.get("filled_amount", 0.0)) - prev_acc > 1e-4:
                    need_fill_sync = True
                elif st_c in ("cancel", "done", ""):
                    s_px = float(trader.bid_slots[s].get("px", 0.0))
                    trader.ex_usdt_free += float(trader.bid_slots[s].get("qty", 0.0)) * s_px
                    trader.bid_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
                    print(f"  ✖️ [NEURAL CANCEL BID #{s:02d} ({'crossed' if is_cross else 'reg'})] id={oid}")

        for s, oid, is_cross in to_cnl_a:
            c_info = c_map.get(oid)
            if c_info is None:
                try:
                    c_info = trader.client.get_order(oid)
                    if isinstance(c_info, dict):
                        c_map[oid] = c_info
                except Exception:
                    c_info = None
            if isinstance(c_info, dict):
                st_c = str(c_info.get("state", "")).lower()
                prev_acc = float(trader.ask_slots[s].get("acc_fill", 0.0))
                if float(c_info.get("filled_amount", 0.0)) - prev_acc > 1e-4:
                    need_fill_sync = True
                elif st_c in ("cancel", "done", ""):
                    trader.ex_coin_free = float(getattr(trader, "ex_coin_free", 0.0)) + float(trader.ask_slots[s].get("qty", 0.0))
                    trader.ask_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
                    print(f"  ✖️ [NEURAL CANCEL ASK #{s:02d} ({'crossed' if is_cross else 'reg'})] id={oid}")

        if need_fill_sync:
            trader._sync_24slot_orders(b1, count_hold=False, pre_fetched_orders=c_map)

    # 3. Постановка Bid-слотов 1-в-1 с fused_order3_clean_tick.py (сортировка по pi_bid, шлюз g_plc_b):
    exp_b = float(plan["exp_bid_frac"])
    res_b_usd = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) * 1.001 for s in trader.bid_slots if s.get("id") is not None)
    free_cash_b = max(0.0, min(trader.bot_cash_usd * 0.998 - res_b_usd, trader.ex_usdt_free))
    rem_exp_b = max(0.0, tot_eq * 0.996 * exp_b - float(trader.pos_qty) * b1 - res_b_usd)
    avail_b_usd = min(free_cash_b, rem_exp_b)

    cand_b_slots = [
        s for s in sorted(range(12), key=lambda i: float(plan["pi_bid"][i]), reverse=True)
        if trader.bid_slots[s].get("id") is None and bool(plan["g_plc_b"][s]) and float(plan["pi_bid"][s]) > 1e-4 and (avail_b_usd / max(1e-6, float(plan["t_bid_px"][s]) * 1.001)) >= min_amt
    ]
    sel_b = []
    for s in cand_b_slots:
        if trader.order3_engine.api_tokens >= 1.0:
            sel_b.append(s)
            trader.order3_engine.api_tokens -= 1.0

    sum_pi_b = sum(float(plan["pi_bid"][s]) + 1e-5 for s in sel_b)
    b_tasks = []
    for s in sel_b:
        px_s = round(float(plan["t_bid_px"][s]), p_prec)
        full_sh_b = (float(plan["pi_bid"][s]) + 1e-5) / max(1e-6, sum_pi_b)
        raw_sh_b = min(float(plan["pi_bid"][s]), full_sh_b)
        share_s = renorm_b * full_sh_b + (1.0 - renorm_b) * raw_sh_b
        q_s = math.floor(((avail_b_usd * share_s * 0.999) / max(1e-6, px_s * 1.001)) * 10000.0) / 10000.0
        if q_s >= min_amt and px_s > 0:
            b_tasks.append((s, q_s, px_s))
        else:
            trader.order3_engine.api_tokens = min(LIVE_TOKEN_BUCKET_CAP, trader.order3_engine.api_tokens + 1.0)

    # 4. Постановка Ask-слотов 1-в-1 с fused_order3_clean_tick.py (эндогенный резерв renorm_blend):
    res_a_qty = sum(float(s.get("qty", 0.0)) for s in trader.ask_slots if s.get("id") is not None)
    avail_a_qty = max(0.0, min(float(trader.pos_qty) - res_a_qty, float(getattr(trader, "ex_coin_free", trader.pos_qty))))

    cand_a_slots = [
        s for s in sorted(range(12), key=lambda i: float(plan["pi_ask"][i]), reverse=True)
        if float(trader.pos_qty) >= min_amt and avail_a_qty >= min_amt and trader.ask_slots[s].get("id") is None and bool(plan["g_plc_a"][s]) and float(plan["pi_ask"][s]) > 1e-4
    ]
    sel_a = []
    for s in cand_a_slots:
        if trader.order3_engine.api_tokens >= 1.0:
            sel_a.append(s)
            trader.order3_engine.api_tokens -= 1.0

    sum_pi_a = sum(float(plan["pi_ask"][s]) + 1e-5 for s in sel_a)
    a_tasks = []
    for s in sel_a:
        px_s = round(max(b1 + t_sz, float(plan["t_ask_px"][s])), p_prec)
        full_sh_a = (float(plan["pi_ask"][s]) + 1e-5) / max(1e-6, sum_pi_a)
        raw_sh_a = min(float(plan["pi_ask"][s]), full_sh_a)
        share_a = renorm_b * full_sh_a + (1.0 - renorm_b) * raw_sh_a
        q_s = math.floor((avail_a_qty * share_a) * 10000.0) / 10000.0
        if q_s >= min_amt:
            a_tasks.append((s, q_s, px_s))
        else:
            trader.order3_engine.api_tokens = min(LIVE_TOKEN_BUCKET_CAP, trader.order3_engine.api_tokens + 1.0)

    # 5. Единый параллельный сетевой залп Buy + Sell в 6 потоков (USDT и PRL независимы — 1 RTT вместо 2!):
    m_idx = int(np.argmax(plan["g_mode"]))
    m_prob = float(plan["g_mode"][m_idx])
    joint_tasks = [("buy", s, q_s, px_s) for s, q_s, px_s in b_tasks] + [("sell", s, q_s, px_s) for s, q_s, px_s in a_tasks]
    if joint_tasks:
        joint_res = list(
            trader._io_pool.map(lambda it: (it[0], *trader._place_slot_order(it[0], it[1], it[2], it[3])), joint_tasks)
        )
        for side_s, s, res_o, q_req, p_req in joint_res:
            if res_o and res_o.get("id") is not None:
                oid_n = res_o.get("id")
                q_act = float(res_o.get("origin_amount", q_req))
                p_act = float(res_o.get("price", p_req))
                if side_s == "buy":
                    init_qa_b = 0.0 if p_act > (b1 + 0.1 * t_sz) else float(plan["t_vis_bq"][s])
                    trader.ex_usdt_free = max(0.0, trader.ex_usdt_free - q_act * p_act * 1.001)
                    trader.bid_slots[s] = {"slot": s, "id": oid_n, "px": p_act, "qty": q_act, "act_ts": now_ts, "acc_fill": 0.0, "qa": init_qa_b, "qa0": init_qa_b}
                    print(f"  📥 [NEURAL PLACE BID #{s:02d}] id={oid_n} | {q_act:.4f} PRL @ ${p_act:.{p_prec}f} (Mode#{m_idx}={m_prob:.0%} | E={exp_b:.0%})")
                else:
                    init_qa_a = 0.0 if p_act < (a1 - 0.1 * t_sz) else float(plan["t_vis_aq"][s])
                    trader.ex_coin_free = max(0.0, float(getattr(trader, "ex_coin_free", 0.0)) - q_act)
                    trader.ask_slots[s] = {"slot": s, "id": oid_n, "px": p_act, "qty": q_act, "act_ts": now_ts, "acc_fill": 0.0, "qa": init_qa_a, "qa0": init_qa_a}
                    print(f"  📤 [NEURAL PLACE ASK #{s:02d}] id={oid_n} | {q_act:.4f} PRL @ ${p_act:.{p_prec}f} (Mode#{m_idx}={m_prob:.0%})")

    if b_tasks or a_tasks or to_cnl_b or to_cnl_a:
        trader._save_wallet()
