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
    MOE_MLP_H_ORDER3,
    MOE_MLP_UP_DIM_ORDER3,
    ROUTER_EXTRA_DIM_ORDER3,
    ROUTER_IN_DIM_ORDER3,
    ROUTER_H_DIM_ORDER3,
    ROUTER_RES_H_ORDER3,
    ROUTER_OUT_DIM_ORDER3,
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

        # Латентная многомасштабная память (16D / 32D -> 48D + прямая связь в ствол/экспертов w_mem2h)
        wrec_raw = ck["w_rec"].to(dtype=torch.float32)
        brec_raw = ck["b_rec"].to(dtype=torch.float32).view(1, -1)
        if brec_raw.numel() != 2 * MEM_DIM_ORDER3:
            old_m = brec_raw.numel() // 2
            wrec_pad = torch.zeros((256, 2 * MEM_DIM_ORDER3), dtype=torch.float32)
            brec_pad = torch.zeros((1, 2 * MEM_DIM_ORDER3), dtype=torch.float32)
            cm_r = min(old_m, MEM_DIM_ORDER3)
            wrec_pad[:, :cm_r] = wrec_raw[:, :cm_r]
            wrec_pad[:, MEM_DIM_ORDER3 : MEM_DIM_ORDER3 + cm_r] = wrec_raw[:, old_m : old_m + cm_r]
            brec_pad[:, :cm_r] = brec_raw[:, :cm_r]
            brec_pad[:, MEM_DIM_ORDER3 : MEM_DIM_ORDER3 + cm_r] = brec_raw[:, old_m : old_m + cm_r]
            wrec_raw, brec_raw = wrec_pad, brec_pad
        self.w_rec = wrec_raw.to(device=self.device)
        self.b_rec = brec_raw.view(1, 2 * MEM_DIM_ORDER3).to(device=self.device)
        if "w_mem2h" in ck:
            wm2h_raw = ck["w_mem2h"].to(dtype=torch.float32)
            if wm2h_raw.shape == (MEM_DIM_ORDER3, 256):
                self.w_mem2h = wm2h_raw.to(device=self.device)
            else:
                wm2h_pad = torch.zeros((MEM_DIM_ORDER3, 256), dtype=torch.float32)
                cm2 = min(wm2h_raw.shape[0], MEM_DIM_ORDER3)
                wm2h_pad[:cm2, :] = wm2h_raw[:cm2, :]
                self.w_mem2h = wm2h_pad.to(device=self.device)
        else:
            self.w_mem2h = torch.zeros((MEM_DIM_ORDER3, 256), dtype=torch.float32, device=self.device)

        # Авто-определение числа мод K (поддержка как K=8 с 16D Liquid-выходами, так и старых 8D выходов!)
        bg2_numel = int(ck["b_gate2"].numel())
        self.k_modes = int(ck.get("k_modes", K_MODES_ORDER3 if bg2_numel in (K_MODES_ORDER3, 2 * K_MODES_ORDER3) else bg2_numel))
        self.moe_out_dim = self.k_modes * EXPERT_OUT_DIM_ORDER3
        self.reg_in_dim = 28 + self.k_modes

        # Двухкаскадный Вентильный рекуррентный маршрутизатор (386D -> 128 + ResSwiGLU -> 16)
        wg1_raw = ck["w_gate1"].to(dtype=torch.float32)
        bg1_raw = ck["b_gate1"].to(dtype=torch.float32).view(1, -1)
        wg2_raw = ck["w_gate2"].to(dtype=torch.float32)
        bg2_raw = ck["b_gate2"].to(dtype=torch.float32).view(1, -1)
        if wg1_raw.shape != (ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3):
            old_rin, old_rh = wg1_raw.shape
            wg1_adap = torch.zeros((ROUTER_IN_DIM_ORDER3, ROUTER_H_DIM_ORDER3), dtype=torch.float32)
            bg1_adap = torch.zeros((1, ROUTER_H_DIM_ORDER3), dtype=torch.float32)
            ch_r = min(old_rh, ROUTER_H_DIM_ORDER3)
            bg1_adap[:, :ch_r] = bg1_raw[:, :ch_r]
            if old_rin == 370 and ROUTER_IN_DIM_ORDER3 == 386:
                wg1_adap[:320, :ch_r] = wg1_raw[:320, :ch_r]
                wg1_adap[336:386, :ch_r] = wg1_raw[320:370, :ch_r]
            elif old_rin == 368 and ROUTER_IN_DIM_ORDER3 == 386:
                wg1_adap[:320, :ch_r] = wg1_raw[:320, :ch_r]
                wg1_adap[336:376, :ch_r] = wg1_raw[320:360, :ch_r]
                wg1_adap[378:386, :ch_r] = wg1_raw[360:368, :ch_r]
            elif old_rin == 368 and ROUTER_IN_DIM_ORDER3 == 370:
                wg1_adap[:360, :ch_r] = wg1_raw[:360, :ch_r]
                wg1_adap[362:370, :ch_r] = wg1_raw[360:368, :ch_r]
            elif old_rin == 362 and ROUTER_IN_DIM_ORDER3 >= 368:
                wg1_adap[:320, :ch_r] = wg1_raw[:320, :ch_r]
                wg1_adap[336:370, :ch_r] = wg1_raw[320:354, :ch_r]
                wg1_adap[ROUTER_IN_DIM_ORDER3 - 8 : ROUTER_IN_DIM_ORDER3, :ch_r] = wg1_raw[354:362, :ch_r]
            elif old_rin == 358 and ROUTER_IN_DIM_ORDER3 >= 362:
                wg1_adap[:320, :ch_r] = wg1_raw[:320, :ch_r]
                wg1_adap[336:366, :ch_r] = wg1_raw[320:350, :ch_r]
                wg1_adap[ROUTER_IN_DIM_ORDER3 - 8 : ROUTER_IN_DIM_ORDER3, :ch_r] = wg1_raw[350:358, :ch_r]
            elif old_rin == 326 and ROUTER_IN_DIM_ORDER3 >= 358:
                wg1_adap[:304, :ch_r] = wg1_raw[:304, :ch_r]
                wg1_adap[336:358, :ch_r] = wg1_raw[304:326, :ch_r]
            elif old_rin == 295:
                wg1_adap[:288, :ch_r] = wg1_raw[:288, :ch_r]
                wg1_adap[336:343, :ch_r] = wg1_raw[288:295, :ch_r]
            else:
                copy_r = min(ROUTER_IN_DIM_ORDER3, old_rin)
                wg1_adap[:copy_r, :ch_r] = wg1_raw[:copy_r, :ch_r]
            wg1_raw, bg1_raw = wg1_adap, bg1_adap
        if wg2_raw.shape != (ROUTER_H_DIM_ORDER3, 2 * self.k_modes):
            old_rh2, old_ro = wg2_raw.shape
            wg2_adap = torch.zeros((ROUTER_H_DIM_ORDER3, 2 * self.k_modes), dtype=torch.float32)
            bg2_adap = torch.zeros((1, 2 * self.k_modes), dtype=torch.float32)
            wg2_adap[:min(old_rh2, ROUTER_H_DIM_ORDER3), :min(old_ro, 2 * self.k_modes)] = wg2_raw[:min(old_rh2, ROUTER_H_DIM_ORDER3), :min(old_ro, 2 * self.k_modes)]
            bg2_adap[:, :min(old_ro, 2 * self.k_modes)] = bg2_raw[:, :min(old_ro, 2 * self.k_modes)]
            if old_ro <= self.k_modes:
                bg2_adap[:, self.k_modes:] = 12.0  # beta_gate=1.0 для 0.00e+00 паритета старых чекпоинтов!
            wg2_raw, bg2_raw = wg2_adap, bg2_adap
        self.w_gate1 = wg1_raw.to(device=self.device)
        self.b_gate1 = bg1_raw.view(1, ROUTER_H_DIM_ORDER3).to(device=self.device)
        if "w_gate_res1" in ck and ck["w_gate_res1"].shape == (ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3):
            self.w_gate_res1 = ck["w_gate_res1"].to(device=self.device, dtype=torch.float32)
            self.b_gate_res = ck["b_gate_res"].view(1, 2 * ROUTER_RES_H_ORDER3).to(device=self.device, dtype=torch.float32)
            self.w_gate_res2 = ck["w_gate_res2"].to(device=self.device, dtype=torch.float32)
        else:
            self.w_gate_res1 = torch.zeros((ROUTER_H_DIM_ORDER3, 2 * ROUTER_RES_H_ORDER3), dtype=torch.float32, device=self.device)
            self.b_gate_res = torch.zeros((1, 2 * ROUTER_RES_H_ORDER3), dtype=torch.float32, device=self.device)
            self.w_gate_res2 = torch.zeros((ROUTER_RES_H_ORDER3, ROUTER_H_DIM_ORDER3), dtype=torch.float32, device=self.device)
        self.w_gate2 = wg2_raw.to(device=self.device)
        self.b_gate2 = bg2_raw.view(1, 2 * self.k_modes).to(device=self.device)

        # MoE эксперты (линейный путь + нелинейная Residual SwiGLU ветка на каждую моду)
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
            if old_ed <= 50 and EXPERT_OUT_DIM_ORDER3 >= 52:
                b2m_pad[:, :, 50:52] = 12.0
            if old_ed <= 52 and EXPERT_OUT_DIM_ORDER3 >= 54:
                b2m_pad[:, :, 52:54] = -12.0
            if old_ed <= 54 and EXPERT_OUT_DIM_ORDER3 >= 55:
                b2m_pad[:, :, 54] = -12.0
            if old_ed <= 55 and EXPERT_OUT_DIM_ORDER3 >= 56:
                b2m_pad[:, :, 55] = -12.0
            if old_ed <= 56 and EXPERT_OUT_DIM_ORDER3 >= 58:
                b2m_pad[:, :, 56:58] = -12.0
            w2m_raw = w2m_pad.reshape(256, self.moe_out_dim)
            b2m_raw = b2m_pad.reshape(1, self.moe_out_dim)
        self.w2_moe = w2m_raw.to(device=self.device)
        self.b2_moe = b2m_raw.view(1, self.moe_out_dim).to(device=self.device)
        if "w_moe_up" in ck and ck["w_moe_up"].shape == (256, MOE_MLP_UP_DIM_ORDER3):
            self.w_moe_up = ck["w_moe_up"].to(device=self.device, dtype=torch.float32)
            self.b_moe_up = ck["b_moe_up"].view(1, MOE_MLP_UP_DIM_ORDER3).to(device=self.device, dtype=torch.float32)
            self.w_moe_dn = ck["w_moe_dn"].view(self.k_modes, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3).to(device=self.device, dtype=torch.float32)
        else:
            self.w_moe_up = torch.zeros((256, MOE_MLP_UP_DIM_ORDER3), dtype=torch.float32, device=self.device)
            self.b_moe_up = torch.zeros((1, MOE_MLP_UP_DIM_ORDER3), dtype=torch.float32, device=self.device)
            self.w_moe_dn = torch.zeros((self.k_modes, MOE_MLP_H_ORDER3, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32, device=self.device)

        # Двухкаскадный ResSwiGLU регулятор слотов (36D -> 64 -> ResSwiGLU -> 2)
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
        if "w_reg_res1" in ck and ck["w_reg_res1"].shape == (REG_H_DIM_ORDER3, 64):
            self.w_reg_res1 = ck["w_reg_res1"].to(device=self.device, dtype=torch.float32)
            self.b_reg_res = ck["b_reg_res"].view(1, 64).to(device=self.device, dtype=torch.float32)
            self.w_reg_res2 = ck["w_reg_res2"].to(device=self.device, dtype=torch.float32)
        else:
            self.w_reg_res1 = torch.zeros((REG_H_DIM_ORDER3, 64), dtype=torch.float32, device=self.device)
            self.b_reg_res = torch.zeros((1, 64), dtype=torch.float32, device=self.device)
            self.w_reg_res2 = torch.zeros((32, REG_H_DIM_ORDER3), dtype=torch.float32, device=self.device)
        self.w_reg2 = wreg2_raw.to(device=self.device)
        self.b_reg2_b = ck["b_reg2_b"].view(1, 2).to(device=self.device, dtype=torch.float32)
        self.b_reg2_a = ck["b_reg2_a"].view(1, 2).to(device=self.device, dtype=torch.float32)

        # Внутреннее состояние (48D память + Liquid-вентили роутера и плана!)
        self.mem_h = torch.zeros((1, MEM_DIM_ORDER3), dtype=torch.float32, device=self.device)
        self.gate_state = torch.zeros((1, self.k_modes), dtype=torch.float32, device=self.device)
        self.strat_state = torch.zeros((1, EXPERT_OUT_DIM_ORDER3), dtype=torch.float32, device=self.device)
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
        min_buy_px: float = 0.0,
        low_tranche_qty: float = 0.0,
        last_sell_px: float = 0.0,
        last_sell_age_s: float = 99999.0,
    ) -> Dict[str, Any]:
        """
        Полный шаг инференса Order-3 (L46/L48 100% паритет с fused_order3_clean_tick.py).
        """
        # Тензоры
        zm_in = torch.from_numpy(z_mkt_39d).to(device=self.device, dtype=torch.float32).view(1, -1)
        if zm_in.shape[-1] < MKT_DIM_ORDER3:
            zm_in = torch.cat([zm_in, torch.zeros((1, MKT_DIM_ORDER3 - zm_in.shape[-1]), dtype=torch.float32, device=self.device)], dim=-1)
        zm = _scale_mkt_44d(zm_in[:, :MKT_DIM_ORDER3], torch.float32)
        za = torch.from_numpy(z_ag_110d).to(device=self.device, dtype=torch.float32).view(1, AG_DIM_ORDER3)
        za_norm = _normalize_agent_z110_per_block(za, torch.float32)

        # 1. 194D -> 384
        pre_all = (zm @ self.w1_mkt + self.b1) + (za_norm @ self.w1_ag)
        h_base = torch.relu(pre_all[:, :256])
        h_gate = (2.0 * torch.sigmoid(pre_all[:, 256:384])).repeat_interleave(2, dim=-1)
        h1 = h_base * h_gate

        # ResSwiGLU
        res_pre = h1 @ self.w_res1 + self.b_res1
        res_mid = torch.relu(res_pre[:, :64]) * (2.0 * torch.sigmoid(res_pre[:, 64:128]))
        h2 = h1 + res_mid @ self.w_res2

        # 8-горизонтная латентная память 64D + обратная проекция в экспертов и роутер h2_ctx
        rec_raw = h2 @ self.w_rec + self.b_rec
        rec_u = torch.tanh(rec_raw[:, :MEM_DIM_ORDER3])
        rec_gate = torch.cat(
            [
                torch.sigmoid(rec_raw[:, 64:72] + 1.0),
                torch.sigmoid(rec_raw[:, 72:80] + 3.8),
                torch.sigmoid(rec_raw[:, 80:88] + 5.2),
                torch.sigmoid(rec_raw[:, 88:96] + 6.6),
                torch.sigmoid(rec_raw[:, 96:104] + 2.2),
                torch.sigmoid(rec_raw[:, 104:112] + 4.5),
                torch.sigmoid(rec_raw[:, 112:120] + 7.5),
                torch.sigmoid(rec_raw[:, 120:128] + 8.9),
            ],
            dim=-1,
        )
        self.mem_h = rec_gate * self.mem_h + (1.0 - rec_gate) * rec_u
        h2_ctx = h2 + self.mem_h @ self.w_mem2h

        # Двухкаскадный Вентильный рекуррентный маршрутизатор (386D -> 128 SiLU + ResSwiGLU -> 16)
        acct_8d = za[:, 54:62]
        queue_8d = za[:, 62:70]
        r_extra_30d = torch.cat(
            [
                zm[:, 0:1], zm[:, 40:44], zm[:, 44:45], zm[:, 48:49],
                zm[:, 29:34], zm[:, 36:37], zm[:, 39:40], zm[:, 55:56], zm[:, 12:13],
                zm[:, 60:66],
                zm[:, 72:76], zm[:, 79:82], zm[:, 83:84],
                zm[:, 84:88],
                zm[:, 88:96],
            ],
            dim=-1,
        )
        prev_g_mode = torch.softmax(self.gate_state, dim=-1)
        router_in = torch.cat([h2_ctx, acct_8d, queue_8d, self.mem_h, r_extra_30d, prev_g_mode], dim=-1)
        r_hid1 = F.silu(router_in @ self.w_gate1 + self.b_gate1)
        r_res_pre = r_hid1 @ self.w_gate_res1 + self.b_gate_res
        r_res_act = F.silu(r_res_pre[:, :ROUTER_RES_H_ORDER3]) * (2.0 * torch.sigmoid(r_res_pre[:, ROUTER_RES_H_ORDER3:]))
        r_hid = r_hid1 + r_res_act @ self.w_gate_res2
        gate_out = (r_hid @ self.w_gate2 + self.b_gate2).float()
        raw_logits = gate_out[:, :self.k_modes]
        beta_gate = torch.sigmoid(gate_out[:, self.k_modes : 2 * self.k_modes])
        self.gate_state = (1.0 - beta_gate) * self.gate_state + beta_gate * raw_logits
        g_mode = torch.softmax(self.gate_state, dim=-1)

        # K Нелинейных экспертов (Linear 58D + Residual SwiGLU 256 -> 96 [48 gated] -> 58D)
        moe_lin = (h2_ctx @ self.w2_moe + self.b2_moe).float().view(1, self.k_modes, EXPERT_OUT_DIM_ORDER3)
        moe_up = (h2_ctx @ self.w_moe_up + self.b_moe_up).view(1, self.k_modes, 2 * MOE_MLP_H_ORDER3)
        moe_act = F.silu(moe_up[:, :, :MOE_MLP_H_ORDER3]) * (2.0 * torch.sigmoid(moe_up[:, :, MOE_MLP_H_ORDER3:]))
        moe_mlp = torch.matmul(moe_act.permute(1, 0, 2), self.w_moe_dn).permute(1, 0, 2).float()
        moe_all = moe_lin + moe_mlp
        strat_inst = (g_mode.unsqueeze(-1) * moe_all).sum(dim=1)  # [1, 58]
        alpha_bid = torch.sigmoid(strat_inst[:, 50:51])
        alpha_ask = torch.sigmoid(strat_inst[:, 51:52])
        bid_blend = (1.0 - alpha_bid) * self.strat_state + alpha_bid * strat_inst
        ask_blend = (1.0 - alpha_ask) * self.strat_state + alpha_ask * strat_inst
        strat_merged = torch.cat(
            [
                bid_blend[:, 0:13], ask_blend[:, 13:25], strat_inst[:, 25:26],
                bid_blend[:, 26:35], ask_blend[:, 35:36], strat_inst[:, 36:37],
                bid_blend[:, 37:41], ask_blend[:, 41:44], bid_blend[:, 44:45],
                ask_blend[:, 45:47], bid_blend[:, 47:48], strat_inst[:, 48:49],
                ask_blend[:, 49:50], strat_inst[:, 50:52],
                ask_blend[:, 52:53], bid_blend[:, 53:54],
                ask_blend[:, 54:55], bid_blend[:, 55:58],
            ],
            dim=-1,
        )
        self.strat_state = strat_merged
        strat_raw = strat_merged.squeeze(0)  # [56]

        # 1C. 100% Эндогенная экспозиция [0..1] + L79 Wide-Spread Micro-Sniper [55]:
        rel_sp_now = (a1 - b1) / max(tick_sz, float(b1))
        wide_sp_gate = min(1.0, max(0.0, (rel_sp_now - 0.0035) / 0.0045))
        eff_micro_s = float(torch.sigmoid(strat_raw[55]).item()) * wide_sp_gate
        exp_bid_frac = max(float(torch.sigmoid(strat_raw[0]).item()), 0.065 * eff_micro_s)
        renorm_blend = float(torch.sigmoid(strat_raw[47]).item())

        tau_b_t = 0.08 + 2.42 * torch.sigmoid(strat_raw[34:35])
        tau_a_t = 0.08 + 2.42 * torch.sigmoid(strat_raw[35:36])
        max_share_b_t = 0.08 + 0.92 * torch.sigmoid(strat_raw[44:45])
        max_share_a_t = 0.08 + 0.92 * torch.sigmoid(strat_raw[44:45] + 0.25)

        logits_b_12 = strat_raw[1:13].clone()
        logits_b_12[0] = logits_b_12[0] + 5.0 * eff_micro_s
        pi_b_t = _fanned_slot_softmax(logits_b_12.unsqueeze(0), tau_b_t, max_share_b_t, floor=0.005).squeeze(0)
        pi_a_t = _fanned_slot_softmax(strat_raw[13:25].unsqueeze(0), tau_a_t, max_share_a_t, floor=0.005).squeeze(0)

        # 2. Непрерывные эндогенные Taker-вентили (Exit ch 25 + Impulse Wave Buy ch 48):
        u_taker = float(torch.tanh(F.relu(strat_raw[25])).item())
        u_taker_buy = float(torch.tanh(F.relu(strat_raw[48] - 1.2)).item())
        max_slip_pct = 0.005 + 0.145 * float(torch.sigmoid(strat_raw[36]).item())

        # 3. Полно-спредовая геометрия 24 слотов + L72..L79 Micro-Sniper [55] (1-в-1 с fused_order3_clean_tick.py):
        spread_room = max(0.0, a1 - b1 - tick_sz)
        inside_b = spread_room * float(torch.sigmoid(strat_raw[37]).item())
        sniper_front = float(torch.sigmoid(strat_raw[53]).item())
        retreat_b = b1 * 0.04 * float(torch.sigmoid(strat_raw[38]).item()) * (1.0 - sniper_front)
        span_b_usd = b1 * (0.0005 + 0.2495 * float(torch.sigmoid(strat_raw[39]).item()))
        gamma_b = 0.40 + 2.10 * float(torch.sigmoid(strat_raw[40]).item())

        curve_b = torch.pow(self.echelon_k01, gamma_b)
        ech0_m = (self.echelon_k01 < 0.01).float()
        eff_retreat_k = retreat_b * (1.0 - ech0_m * eff_micro_s)
        micro_step_in = min(spread_room, 2.0 * tick_sz) * eff_micro_s
        eff_inside_k = torch.clamp(inside_b + ech0_m * micro_step_in, max=spread_room)
        raw_t_bid = (b1 + eff_inside_k - eff_retreat_k) - span_b_usd * curve_b
        # T0.6: Якорь рециклинга [56] — потолок откупа ниже last_sell_px (1-в-1 с fused_order3_clean_tick.py):
        s_age_f = max(0.0, float(last_sell_age_s))
        if float(last_sell_px) > 1e-4 and s_age_f < 3600.0:
            rec_w_s = float(torch.sigmoid(strat_raw[56]).item()) * math.exp(-s_age_f / 1800.0)
            rec_w_k = rec_w_s * (1.0 - ech0_m * eff_micro_s)
            rec_disc = 0.0040 + 0.0160 * float(torch.sigmoid(strat_raw[56] - 1.0).item())
            rec_cap_px = float(last_sell_px) * (1.0 - rec_disc)
            raw_t_bid = (1.0 - rec_w_k) * raw_t_bid + rec_w_k * torch.clamp(raw_t_bid, max=rec_cap_px)
        max_maker_bid = max(tick_sz, a1 - tick_sz)
        t_bid_px = torch.clamp(torch.round(torch.clamp(raw_t_bid, max=max_maker_bid) / tick_sz) * tick_sz, min=tick_sz)

        hp_now_b = pos_qty >= 2.0
        ref_entry = float(trader_entry) * (1.0 + 2.0 * maker_fee) if (hp_now_b and trader_entry > 1e-4) else a1
        has_low_tr = hp_now_b and (float(low_tranche_qty) >= 2.0) and (float(min_buy_px) > 1e-4)
        low_ref_entry = (float(min_buy_px) * (1.0 + 2.0 * maker_fee)) if has_low_tr else ref_entry
        tranche_s = float(torch.sigmoid(strat_raw[52]).item()) if has_low_tr else 0.0
        ech_front_sq = torch.pow(1.0 - self.echelon_k01, 2.0)
        tranche_w_k = tranche_s * ech_front_sq
        ref_entry_k = ref_entry * (1.0 - tranche_w_k) + low_ref_entry * tranche_w_k
        unwind_s = float(torch.sigmoid(strat_raw[54]).item())
        unwind_w_k = (unwind_s * ech_front_sq) * (ref_entry_k > a1).float()
        eff_ref_entry_k = ref_entry_k * (1.0 - unwind_w_k) + a1 * unwind_w_k

        skew_blend = float(torch.sigmoid(strat_raw[41]).item())
        z42_t = strat_raw[42]
        t42_s = float(torch.tanh(z42_t * 0.25).item())
        wave_hold_pct = 0.10 * float(torch.sigmoid(strat_raw[49] - 1.2).item())
        ask_edge_pct = 0.0035 * float(torch.tanh(z42_t).item()) + 0.0765 * (t42_s * t42_s * t42_s) + wave_hold_pct
        span_a_usd = a1 * (0.0005 + 0.2495 * float(torch.sigmoid(strat_raw[39] + 0.5 * z42_t + strat_raw[45]).item()))
        gamma_a = 0.40 + 2.10 * float(torch.sigmoid(strat_raw[40] + strat_raw[46]).item())

        inside_a = spread_room * float(torch.sigmoid(strat_raw[43]).item())
        anchor_base_a_k = a1 + skew_blend * (eff_ref_entry_k - a1) + (a1 * ask_edge_pct - inside_a)
        min_maker_ask = b1 + tick_sz
        curve_a = torch.pow(self.echelon_k01, gamma_a)
        raw_t_ask = anchor_base_a_k + span_a_usd * curve_a
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
        o1_slot_imp = torch.tanh(
            0.30 * (zm[:, 34:35] + zm[:, 36:37])
            + 0.35 * (zm[:, 61:62] + zm[:, 62:63])
            + 0.25 * (zm[:, 72:73] + zm[:, 74:75])
        ).squeeze(0)
        slot_edge_b = torch.clamp((((a1 - t_bid_px) / torch.clamp(t_bid_px, min=tick_sz)) - 2.0 * maker_fee) * 40.0, -2.0, 3.0)
        slot_exec_b = (0.50 * zm[:, 48:49].squeeze(0) + 0.25 * zm[:, 82:83].squeeze(0) + 0.25 * zm[:, 83:84].squeeze(0) + 0.55 * o1_slot_imp).expand(12)
        if pos_qty >= 2.0 and trader_entry > 1e-4:
            slot_edge_a = torch.clamp((((t_ask_px - float(trader_entry)) / max(tick_sz, float(trader_entry))) - 2.0 * maker_fee) * 40.0, -2.5, 3.0)
        else:
            slot_edge_a = torch.zeros_like(t_ask_px)
        slot_exec_a = (0.55 * zm[:, 45:46].squeeze(0) + 0.25 * zm[:, 80:81].squeeze(0) - 0.55 * o1_slot_imp).expand(12)

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

        rb_h0 = F.silu(feat_b_36 @ self.w_reg1 + self.b_reg1_b)
        ra_h0 = F.silu(feat_a_36 @ self.w_reg1 + self.b_reg1_a)
        rb_rp = rb_h0 @ self.w_reg_res1 + self.b_reg_res
        ra_rp = ra_h0 @ self.w_reg_res1 + self.b_reg_res
        rb_h = rb_h0 + (torch.relu(rb_rp[:, :32]) * (2.0 * torch.sigmoid(rb_rp[:, 32:64]))) @ self.w_reg_res2
        ra_h = ra_h0 + (torch.relu(ra_rp[:, :32]) * (2.0 * torch.sigmoid(ra_rp[:, 32:64]))) @ self.w_reg_res2
        rb_out = rb_h @ self.w_reg2 + self.b_reg2_b  # [12, 2]
        ra_out = ra_h @ self.w_reg2 + self.b_reg2_a
        # T0.4: Эндогенный гистерезис перестановки глубоких заявок [57] (1-в-1 с fused_order3_clean_tick.py):
        hyst_band = 0.0080 * float(torch.sigmoid(strat_raw[57]).item())
        abs_drift_b = torch.abs(t_bid_px - cur_b_px) / max(tick_sz, float(b1))
        deep_rest_b = (had_b_pre & (cur_b_px < float(b1) * 0.992) & (hyst_band > 2e-5)).float()
        hyst_damp_b = 8.0 * torch.sigmoid((hyst_band - abs_drift_b) * 1200.0) * deep_rest_b
        rb_cnl_adj = rb_out[:, 1] - hyst_damp_b

        g_plc_b = (rb_out[:, 0] > 0.0).cpu().numpy()
        g_cnl_b = (rb_cnl_adj > 0.0).cpu().numpy()
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
            "cnl_logit_b": rb_cnl_adj.cpu().numpy(),
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
            o1_ext_models = []
            for h_ext in (320, 600, 900, 1800):
                mfe = w1_latest / f"xgb_h{h_ext}s.ubj"
                if mfe.exists():
                    bste = xgb.Booster()
                    bste.load_model(str(mfe))
                    bste.feature_names = None
                    bste.set_param({"nthread": 1})
                    o1_ext_models.append(bste)
            o2_names = [
                "adverse_selection_5s", "wave_exhaustion_10s", "ask_sweep_5s",
                "guard_sweep_12s", "guard_bleed_60s", "guard_crash_300s",
                "guard_exit_30s", "exec_fill_30s", "exec_frontrun_30s",
                "carousel_success", "carousel_capacity",
            ]
            o2_ext_names = [
                "guard_bleed_180s", "guard_crash_600s", "guard_crash_900s",
                "exec_fill_120s", "carousel_success_240s",
            ]
            o2_models = []
            for nm in o2_names:
                mf2 = w2_latest / f"{nm}.ubj"
                bst2 = xgb.Booster()
                bst2.load_model(str(mf2))
                bst2.feature_names = None
                bst2.set_param({"nthread": 1})
                o2_models.append(bst2)
            o2_ext_models = []
            for nme in o2_ext_names:
                mf2e = w2_latest / f"{nme}.ubj"
                if mf2e.exists():
                    bst2e = xgb.Booster()
                    bst2e.load_model(str(mf2e))
                    bst2e.feature_names = None
                    bst2e.set_param({"nthread": 1})
                    o2_ext_models.append(bst2e)
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
            o25_dir = BASE_DIR / "models/prl/order25_live"
            o25_models = {}
            if o25_dir.exists():
                for o25_nm in ("order25_bot_impact", "order25_buy_trap_3m", "order25_sell_regret_3m"):
                    mf25 = o25_dir / f"{o25_nm}.ubj"
                    if mf25.exists():
                        bst25 = xgb.Booster()
                        bst25.load_model(str(mf25))
                        bst25.feature_names = None
                        bst25.set_param({"nthread": 1})
                        o25_models[o25_nm] = bst25
            _LIVE_WF_CACHE.update({
                "ready": "ok",
                "xgb": xgb,
                "num_cols": num_cols,
                "x_buf": np.zeros((1, len(num_cols)), dtype=np.float32),
                "o1_models": o1_models,
                "o1_ext_models": o1_ext_models,
                "o2_models": o2_models,
                "o2_ext_models": o2_ext_models,
                "ow_models": ow_models,
                "o25_models": o25_models,
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
        o1_ext_m = _LIVE_WF_CACHE.get("o1_ext_models") or []
        if len(o1_ext_m) == 4:
            ez_e, cf_e, bi_e = [], [], []
            for bste in o1_ext_m:
                pe = bste.inplace_predict(x_row)[0]
                ez_e.append(float(np.dot(pe, c_vals)))
                cf_e.append(float(1.0 - pe[3]))
                bi_e.append(float(np.sum(pe[4:]) - np.sum(pe[:3])))
            _LIVE_WF_CACHE["last_o1_ext7"] = np.array([ez_e[0], ez_e[1], ez_e[2], ez_e[3], cf_e[1], cf_e[3], bi_e[2]], dtype=np.float32)
        o2_ext_m = _LIVE_WF_CACHE.get("o2_ext_models") or []
        if len(o2_ext_m) == 5:
            _LIVE_WF_CACHE["last_o2_ext5"] = np.array([float(m.inplace_predict(x66)[0]) for m in o2_ext_m], dtype=np.float32)
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
        if len(mh) > 8000 or (mh and (now_m - mh[0][0]) > 7260.0):
            mh[:] = [x for x in mh if (now_m - x[0]) <= 7205.0]
        m_5m = next((x[1] for x in reversed(mh) if (now_m - x[0]) >= 299.5), mh[0][1])
        m_15m = next((x[1] for x in reversed(mh) if (now_m - x[0]) >= 899.5), mh[0][1])
        m_60m = next((x[1] for x in reversed(mh) if (now_m - x[0]) >= 3599.5), mh[0][1])
        vals_30m = [x[1] for x in mh if (now_m - x[0]) <= 1799.5] or [st_mid]
        vals_60m = [x[1] for x in mh if (now_m - x[0]) <= 3599.5] or [st_mid]
        mn_1800, mx_1800 = min(vals_30m), max(vals_30m)
        mn_3600, mx_3600 = min(vals_60m), max(vals_60m)
        rng_1h = max(mx_3600 - mn_3600, st_mid * 0.005)
        z[66] = float(math.tanh((st_mid / max(m_5m, 1e-4) - 1.0) * 25.0))
        z[67] = float(math.tanh((st_mid / max(m_15m, 1e-4) - 1.0) * 16.0))
        z[68] = float(math.tanh((st_mid / max(m_60m, 1e-4) - 1.0) * 10.0))
        z[69] = float(np.clip(((st_mid - mn_3600) / rng_1h) * 2.0 - 1.0, -1.0, 1.0))
        z[70] = float(math.tanh(((st_mid - mn_1800) / st_mid) * 25.0))
        z[71] = float(math.tanh(((mx_1800 - st_mid) / st_mid) * 25.0))

    if MKT_DIM_ORDER3 >= 84:
        o1_e7 = _LIVE_WF_CACHE.get("last_o1_ext7")
        if o1_e7 is not None and len(o1_e7) == 7:
            z[72:76] = np.clip(o1_e7[0:4], -3.0, 3.0)
            z[76] = float(np.clip(o1_e7[4], 0.0, 1.0))
            z[77] = float(np.clip(o1_e7[5], 0.0, 1.0))
            z[78] = float(np.clip(o1_e7[6], -1.0, 1.0))
        else:
            z[76:78] = 0.95
        o2_e5 = _LIVE_WF_CACHE.get("last_o2_ext5")
        if o2_e5 is not None and len(o2_e5) == 5:
            z[79:84] = np.clip(o2_e5, 0.0, 1.0)
        else:
            z[79:84] = 0.40

    if MKT_DIM_ORDER3 >= 88:
        o25_m = _LIVE_WF_CACHE.get("o25_models") or {}
        xgb_mod = _LIVE_WF_CACHE.get("xgb")
        if xgb_mod is not None and len(o25_m) == 3:
            try:
                bp10_l = np.asarray(getattr(trader, "last_clean_bp_10", [b1] * 10), dtype=np.float32)[:5]
                bq10_l = np.asarray(getattr(trader, "last_clean_bq_10", np.zeros(10, dtype=np.float32)), dtype=np.float32)[:5]
                b1_c = max(0.0001, float(b1))
                a1_c = max(b1_c + 0.0001, float(a1))
                l1_u = max(20.0, float(bq10_l[0]) * b1_c if len(bq10_l) > 0 else 20.0)
                t5_u = max(50.0, float(np.sum(bq10_l * bp10_l)) if len(bq10_l) == len(bp10_l) else l1_u)
                sp_s = float(np.clip(((a1_c - b1_c) / b1_c * 10000.0) / 30.0, 0.0, 4.0))
                z84_row = z[:84].reshape(1, 84)
                q_2row = np.zeros((2, 90), dtype=np.float32)
                q_2row[0, :84] = z[:84]
                q_2row[0, 86] = math.log1p(40.0) / 6.0
                q_2row[0, 87] = min(10.0, 40.0 / l1_u) / 3.0
                q_2row[0, 88] = min(5.0, 40.0 / t5_u)
                q_2row[0, 89] = sp_s
                q_2row[1, :84] = z[:84]
                q_2row[1, 86] = math.log1p(250.0) / 6.0
                q_2row[1, 87] = min(10.0, 250.0 / l1_u) / 3.0
                q_2row[1, 88] = min(5.0, 250.0 / t5_u)
                q_2row[1, 89] = sp_s
                p_imp2 = o25_m["order25_bot_impact"].predict(xgb_mod.DMatrix(q_2row))
                dm84 = xgb_mod.DMatrix(z84_row)
                z[84] = float(np.clip(p_imp2[0], 0.0, 1.0))
                z[85] = float(np.clip(p_imp2[1], 0.0, 1.0))
                z[86] = float(np.clip(o25_m["order25_buy_trap_3m"].predict(dm84)[0], 0.0, 1.0))
                z[87] = float(np.clip(o25_m["order25_sell_regret_3m"].predict(dm84)[0], 0.0, 1.0))
            except Exception:
                z[84], z[85], z[86], z[87] = 0.35, 0.36, 0.15, 0.33
        else:
            z[84], z[85], z[86], z[87] = 0.35, 0.36, 0.15, 0.33

    if MKT_DIM_ORDER3 >= 94:
        now_m94 = time.time()
        mh94 = _LIVE_WF_CACHE.get("mid_hist_3600s") or [(now_m94, st_mid)]
        m_30m = next((x[1] for x in reversed(mh94) if (now_m94 - x[0]) >= 1799.5), mh94[0][1])
        vals_cur_15m = [x[1] for x in mh94 if (now_m94 - x[0]) <= 899.5] or [st_mid]
        vals_prv_15m = [x[1] for x in mh94 if 899.5 < (now_m94 - x[0]) <= 1799.5] or vals_cur_15m
        mean_15m = sum(vals_cur_15m) / float(len(vals_cur_15m))
        mx_cur_15m = max(vals_cur_15m)
        mx_prv_15m = max(vals_prv_15m)
        z[88] = float(math.tanh((st_mid / max(m_30m, 1e-4) - 1.0) * 12.0))
        z[89] = float(math.tanh(((st_mid - mean_15m) / st_mid) * 25.0))
        z[90] = float(math.tanh(((mx_prv_15m - mx_cur_15m) / st_mid) * 25.0))

        fh1800 = _LIVE_WF_CACHE.setdefault("flow_hist_1800s", [])  # прогревается при старте демона из trades.csv (live_rl_trader.py)
        last_ts1800 = float(_LIVE_WF_CACHE.get("last_fh1800_ts", 0.0))
        fh_ts_live = list(getattr(trader, "flow_ts_hist_30s", []))
        fh_old_live = list(getattr(trader, "flow_hist_10s", []))
        if fh_ts_live:
            for it_f in fh_ts_live:
                ts_v = float(it_f[0])
                if ts_v > last_ts1800 + 1e-6:
                    fh1800.append((ts_v, max(0.0, float(it_f[1])), max(0.0, float(it_f[2]))))
                    last_ts1800 = ts_v
            _LIVE_WF_CACHE["last_fh1800_ts"] = last_ts1800
        elif fh_old_live:
            for it_f in fh_old_live:
                ts_v = float(it_f[0])
                if ts_v > 1e9 and ts_v > last_ts1800 + 1e-6:
                    fh1800.append((ts_v, max(0.0, float(it_f[2])), max(0.0, float(it_f[3]))))
                    last_ts1800 = ts_v
            _LIVE_WF_CACHE["last_fh1800_ts"] = last_ts1800
        if len(fh1800) > 8000 or (fh1800 and (now_m94 - fh1800[0][0]) > 7260.0):
            fh1800[:] = [x for x in fh1800 if (now_m94 - x[0]) <= 7205.0]
        b5 = sum(x[1] for x in fh1800 if (now_m94 - x[0]) <= 299.5)
        s5 = sum(x[2] for x in fh1800 if (now_m94 - x[0]) <= 299.5)
        b15 = sum(x[1] for x in fh1800 if (now_m94 - x[0]) <= 899.5)
        s15 = sum(x[2] for x in fh1800 if (now_m94 - x[0]) <= 899.5)
        b30 = sum(x[1] for x in fh1800 if (now_m94 - x[0]) <= 1799.5)
        s30 = sum(x[2] for x in fh1800 if (now_m94 - x[0]) <= 1799.5)
        z[91] = float(np.clip((b5 - s5) / (b5 + s5 + 50.0), -1.0, 1.0))
        z[92] = float(np.clip((b15 - s15) / (b15 + s15 + 150.0), -1.0, 1.0))
        z[93] = float(np.clip((b30 - s30) / (b30 + s30 + 300.0), -1.0, 1.0))
        if MKT_DIM_ORDER3 >= 96:
            m_120m = next((x[1] for x in reversed(mh94) if (now_m94 - x[0]) >= 7199.5), mh94[0][1])
            b120 = sum(x[1] for x in fh1800 if (now_m94 - x[0]) <= 7199.5)
            s120 = sum(x[2] for x in fh1800 if (now_m94 - x[0]) <= 7199.5)
            z[94] = float(math.tanh((st_mid / max(m_120m, 1e-4) - 1.0) * 8.0))
            z[95] = float(np.clip((b120 - s120) / (b120 + s120 + 1000.0), -1.0, 1.0))

    return z


def _msk_ms() -> str:
    """Точная метка времени MSK с миллисекундами для строк постановки/отмены (сверка live с симулятором по секундам)."""
    t_now = time.time()
    return time.strftime("%H:%M:%S", time.gmtime(t_now + 3 * 3600)) + f".{int((t_now % 1) * 1000):03d}"


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

    # Подготовка текущих массивов слотов и чистого стакана
    b_px = np.array([float(s.get("px", 0.0)) for s in trader.bid_slots], dtype=np.float32)
    b_qty = np.array([float(s.get("qty", 0.0)) for s in trader.bid_slots], dtype=np.float32)
    a_px = np.array([float(s.get("px", 0.0)) for s in trader.ask_slots], dtype=np.float32)
    a_qty = np.array([float(s.get("qty", 0.0)) for s in trader.ask_slots], dtype=np.float32)

    z_mkt = build_live_hybrid_market_obs_40d(trader, b1, a1)

    # Изолированный канареечный прибор широкого спреда (8 PRL, авто-стоп -$2.50 / 80 кругов):
    bp10_pre = np.asarray(getattr(trader, "last_clean_bp_10", [b1] * 10), dtype=np.float32)[:10]
    ap10_pre = np.asarray(getattr(trader, "last_clean_ap_10", [a1] * 10), dtype=np.float32)[:10]
    bq10_pre = np.asarray(getattr(trader, "last_clean_bq_10", np.zeros(10, dtype=np.float32)), dtype=np.float32)[:10]
    aq10_pre = np.asarray(getattr(trader, "last_clean_aq_10", np.zeros(10, dtype=np.float32)), dtype=np.float32)[:10]
    fh_pre = list(getattr(trader, "flow_hist_10s", []))
    rs_1s_pre = float(fh_pre[-1][3]) if fh_pre else 0.0
    try:
        from _tools.canary_scalp_probe import step_canary_scalp
        can_info = step_canary_scalp(
            trader=trader, b1=b1, a1=a1, z_mkt=z_mkt,
            bp10_live=bp10_pre, bq10_live=bq10_pre, ap10_live=ap10_pre, aq10_live=aq10_pre,
            roll_sell_1s=rs_1s_pre, now_ts=now_ts,
        )
    except Exception as e_can:
        print(f"  ⚠️ [CANARY PROBE WARN]: {e_can}")
        can_info = {"canary_pos_qty": 0.0, "canary_cash_reserve": 0.0, "canary_bid_oid": None, "canary_ask_oid": None}

    can_pos_q = max(0.0, float(can_info.get("canary_pos_qty", 0.0)))
    can_cash_res = max(0.0, float(can_info.get("canary_cash_reserve", 0.0)))
    core_pos_qty = max(0.0, float(getattr(trader, "pos_qty", 0.0)) - can_pos_q)
    core_cash_usd = max(0.0, float(getattr(trader, "bot_cash_usd", 0.0)) - (can_cash_res if can_pos_q < 1.95 else 0.0))
    tot_eq = max(float(getattr(trader, "initial_budget_usd", 1022.54)) - can_cash_res, core_cash_usd + core_pos_qty * b1)
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
            h2w_ctx = h2w + eng_w.mem_h @ eng_w.w_mem2h
            rex_w = torch.cat(
                [
                    zm_w[:, 0:1], zm_w[:, 40:44], zm_w[:, 44:45], zm_w[:, 48:49],
                    zm_w[:, 29:34], zm_w[:, 36:37], zm_w[:, 39:40], zm_w[:, 55:56], zm_w[:, 12:13],
                    zm_w[:, 60:66], zm_w[:, 72:76], zm_w[:, 79:82], zm_w[:, 83:84],
                    zm_w[:, 84:88], zm_w[:, 88:96],
                ],
                dim=-1,
            )
            z16_w = torch.zeros((1, 16), dtype=torch.float32, device=eng_w.device)
            p0_m = torch.full((1, eng_w.k_modes), 1.0 / float(eng_w.k_modes), dtype=torch.float32, device=eng_w.device)
            rin_w = torch.cat([h2w_ctx, z16_w, eng_w.mem_h, rex_w, p0_m], dim=-1)
            rh1_w = F.silu(rin_w @ eng_w.w_gate1 + eng_w.b_gate1)
            rres_pw = rh1_w @ eng_w.w_gate_res1 + eng_w.b_gate_res
            rh_w = rh1_w + (F.silu(rres_pw[:, :ROUTER_RES_H_ORDER3]) * (2.0 * torch.sigmoid(rres_pw[:, ROUTER_RES_H_ORDER3:]))) @ eng_w.w_gate_res2
            go_w = (rh_w @ eng_w.w_gate2 + eng_w.b_gate2)[:, :eng_w.k_modes]
            gm_w = torch.softmax(go_w, dim=-1)
            moe_lin_w = (h2w_ctx @ eng_w.w2_moe + eng_w.b2_moe).view(1, eng_w.k_modes, EXPERT_OUT_DIM_ORDER3)
            moe_up_w = (h2w_ctx @ eng_w.w_moe_up + eng_w.b_moe_up).view(1, eng_w.k_modes, 2 * MOE_MLP_H_ORDER3)
            moe_act_w = F.silu(moe_up_w[:, :, :MOE_MLP_H_ORDER3]) * (2.0 * torch.sigmoid(moe_up_w[:, :, MOE_MLP_H_ORDER3:]))
            moe_w = moe_lin_w + torch.matmul(moe_act_w.permute(1, 0, 2), eng_w.w_moe_dn).permute(1, 0, 2)
            eng_w.gate_state.copy_(go_w)
            eng_w.strat_state.copy_((gm_w.unsqueeze(-1) * moe_w).sum(dim=1))
            eng_w._mem_warmed = True

    z_ag = getattr(trader, "last_order3_ag_110d", None)
    if z_ag is None or len(z_ag) != AG_DIM_ORDER3:
        z_ag = np.zeros(AG_DIM_ORDER3, dtype=np.float32)
        b1_q_live = max(0.0, float(getattr(trader, "last_b1_q", 10.0)))
        a1_q_live = max(0.0, float(getattr(trader, "last_a1_q", 10.0)))
        l1_bid_usd = max(100.0, b1 * b1_q_live)
        l1_ask_usd = max(100.0, a1 * a1_q_live)
        res_b_usd_ag = float(np.sum(b_qty * b_px * 1.001))
        unres_cash_ag = max(0.0, core_cash_usd - res_b_usd_ag)
        roll_buy_1s = max(10.0, float(getattr(trader, "last_buy_vol_10s", 100.0)) / 10.0)
        low_tr_q = float(getattr(trader, "tr_low_qty", getattr(trader, "low_tranche_qty", 0.0)))
        raw_min_p = float(getattr(trader, "tr_low_px", getattr(trader, "min_buy_px", 0.0)))
        min_buy_p = raw_min_p if raw_min_p > 1e-4 else float(trader.avg_entry)

        t_sz_ag = max(1e-6, float(getattr(trader, "exec_tick_size", trader.tick_size)))
        m_fee_ag = 0.0010
        has_p_ag = core_pos_qty > 1e-6
        pos_rat_ag = float(np.clip((core_pos_qty * b1) / max(tot_eq, 1.0), 0.0, 1.5))
        eff_ps_ag = float(np.clip(pos_rat_ag / 0.25, 0.15, 1.0))

        z_ag[0] = math.log1p(tot_eq / l1_bid_usd) / 3.0
        z_ag[1] = math.log1p(unres_cash_ag / l1_bid_usd) / 3.0
        z_ag[2] = min(4.0, (core_pos_qty * a1) / l1_ask_usd) / 2.0
        z_ag[3] = min(10.0, core_pos_qty / roll_buy_1s) / 5.0
        z_ag[4] = float(np.clip(low_tr_q / max(core_pos_qty, 1e-6), 0.0, 1.0)) if has_p_ag else 0.0
        z_ag[5] = float(np.clip((((b1 - min_buy_p) / max(min_buy_p, t_sz_ag)) - 2.0 * m_fee_ag) * 30.0, -2.5, 2.5)) if (has_p_ag and min_buy_p > 0) else 0.0
        z_ag[6:18] = np.clip((b_qty * b1) / max(tot_eq, 25.0), 0.0, 1.0)
        z_ag[18:30] = np.clip(a_qty / max(core_pos_qty, 2.0), 0.0, 1.0)
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
        pos_qty=core_pos_qty,
        cash=core_cash_usd,
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
        min_buy_px=float(getattr(trader, "tr_low_px", getattr(trader, "min_buy_px", 0.0))),
        low_tranche_qty=float(getattr(trader, "tr_low_qty", getattr(trader, "low_tranche_qty", 0.0))),
        last_sell_px=float(getattr(trader, "last_exit_price", getattr(trader, "last_sell_px", 0.0))),
        last_sell_age_s=(now_ts - float(getattr(trader, "last_exit_ts", getattr(trader, "last_sell_ts", 0.0)))) if float(getattr(trader, "last_exit_ts", getattr(trader, "last_sell_ts", 0.0))) > 0.0 else 99999.0,
    )

    # Посекундная запись того, что реально видела нейросеть на сервере (94D рынок + 110D агент + режимы/экспозиция):
    # позволяет прогонять симулятор на ровно тех входах, что были в бою, и отделить расхождение признаков от остального.
    try:
        import struct as _st
        obs_f = getattr(trader, "_o3_obs_log_f", None)
        if obs_f is None:
            obs_p = BASE_DIR / "data" / "logs" / "order3_live_obs.bin"
            obs_p.parent.mkdir(parents=True, exist_ok=True)
            obs_f = open(obs_p, "ab", buffering=1 << 16)
            trader._o3_obs_log_f = obs_f
        g_m = np.asarray(plan.get("g_mode", np.zeros(K_MODES_ORDER3)), dtype=np.float32).reshape(-1)[:K_MODES_ORDER3]
        rec_v = np.concatenate([
            np.asarray(z_mkt, dtype=np.float32).reshape(-1)[:MKT_DIM_ORDER3],
            np.asarray(z_ag, dtype=np.float32).reshape(-1)[:AG_DIM_ORDER3],
            np.pad(g_m, (0, K_MODES_ORDER3 - len(g_m))),
            np.array([float(plan.get("exp_bid_frac", 0.0)), float(trader.order3_engine.api_tokens), core_pos_qty, core_cash_usd, b1, a1], dtype=np.float32),
        ])
        obs_f.write(_st.pack("<d", float(now_ts)) + rec_v.tobytes())
    except Exception:
        pass

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
        free_c_tb = max(0.0, min(core_cash_usd * 0.996 - res_b_pre, float(trader.ex_usdt_free)))
        rem_e_tb = max(0.0, tot_eq * 0.996 * float(plan["exp_bid_frac"]) - core_pos_qty * b1 - res_b_pre)
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
    if (core_pos_qty * u_taker) >= min_amt and trader.order3_engine.api_tokens >= 1.0:
        raw_sq = core_pos_qty * u_taker
        sell_q = core_pos_qty if (core_pos_qty - raw_sq) < min_amt else raw_sq
        rem_pos_est = max(0.0, core_pos_qty - sell_q)
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
            if len(drop_ask_oids) >= 2 and len(drop_ask_oids) == n_open_all_a and can_info.get("canary_ask_oid") is None:
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

    hp_now = core_pos_qty >= min_amt
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
        bulk_b = (len(to_cnl_b) >= 2) and (len(to_cnl_b) == n_open_b) and (can_info.get("canary_bid_oid") is None)
        bulk_a = (len(to_cnl_a) >= 2) and (len(to_cnl_a) == n_open_a) and (can_info.get("canary_ask_oid") is None)
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

        # Peatio v2 асинхронная очередь отмен: если cancel вернул state="wait", даём движку 120 мс и проверяем исчезновение из open_orders!
        has_wait_cnl = any(isinstance(v, dict) and str(v.get("state", "")).lower() == "wait" for v in c_map.values())
        if has_wait_cnl:
            time.sleep(0.12)
            try:
                still_open_ids = {o_it.get("id") for o_it in (trader.client.get_open_orders(trader.market) or []) if isinstance(o_it, dict)}
                for oid_w, info_w in list(c_map.items()):
                    if isinstance(info_w, dict) and str(info_w.get("state", "")).lower() == "wait" and oid_w not in still_open_ids:
                        info_w["state"] = "cancel"
            except Exception:
                pass

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
                    print(f"  ✖️ [NEURAL CANCEL BID #{s:02d} ({'crossed' if is_cross else 'reg'})] id={oid} | t={_msk_ms()}")

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
                    print(f"  ✖️ [NEURAL CANCEL ASK #{s:02d} ({'crossed' if is_cross else 'reg'})] id={oid} | t={_msk_ms()}")

        if need_fill_sync:
            trader._sync_24slot_orders(b1, count_hold=False, pre_fetched_orders=c_map)

    # 3. Постановка Bid-слотов 1-в-1 с fused_order3_clean_tick.py (сортировка по pi_bid, шлюз g_plc_b):
    # Важно: Peatio v2 (SafeTrade) блокирует под лимитный BUY qty * px * (1 + taker_fee = 1.002), поэтому резервируем 1.0025 + буфер 0.990 от ex_usdt_free, чтобы исключить HTTP 422 на дальнем слоте #11!
    exp_b = float(plan["exp_bid_frac"])
    res_b_usd = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) * 1.002 for s in trader.bid_slots if s.get("id") is not None)
    ex_free_net = max(0.0, float(trader.ex_usdt_free) - (can_cash_res if can_pos_q < 1.95 and can_info.get("canary_bid_oid") is None else 0.0)) * 0.994
    free_cash_b = max(0.0, min(core_cash_usd * 0.992 - res_b_usd, ex_free_net))
    rem_exp_b = max(0.0, tot_eq * 0.990 * exp_b - core_pos_qty * b1 - res_b_usd)
    avail_b_usd = min(free_cash_b, rem_exp_b)

    cand_b_slots = [
        s for s in sorted(range(12), key=lambda i: float(plan["pi_bid"][i]), reverse=True)
        if trader.bid_slots[s].get("id") is None and bool(plan["g_plc_b"][s]) and float(plan["pi_bid"][s]) > 1e-4 and (avail_b_usd / max(1e-6, float(plan["t_bid_px"][s]) * 1.0025)) >= min_amt
    ]
    sel_b = []
    for s in cand_b_slots:
        if trader.order3_engine.api_tokens >= 1.0:
            sel_b.append(s)
            trader.order3_engine.api_tokens -= 1.0

    # "Легко в бою": транспортный 2-тиковый зазор от встречного BBO при широком спреде (>=3 тиков) защищает пассивный Maker от случайного Taker-перехлёста за 206 мс полёта HTTP!
    flight_guard_ticks = 2.0 if (a1 - b1) >= 2.9 * t_sz else 1.0
    max_safe_bid = max(t_sz, a1 - flight_guard_ticks * t_sz)
    min_safe_ask = b1 + flight_guard_ticks * t_sz

    sum_pi_b = sum(float(plan["pi_bid"][s]) + 1e-5 for s in sel_b)
    b_tasks = []
    rem_batch_usd = ex_free_net
    for s in sel_b:
        px_s = round(min(max_safe_bid, float(plan["t_bid_px"][s])), p_prec)
        full_sh_b = (float(plan["pi_bid"][s]) + 1e-5) / max(1e-6, sum_pi_b)
        raw_sh_b = min(float(plan["pi_bid"][s]), full_sh_b)
        share_s = renorm_b * full_sh_b + (1.0 - renorm_b) * raw_sh_b
        alloc_s_usd = min(avail_b_usd * share_s * 0.998, rem_batch_usd)
        q_s = math.floor((alloc_s_usd / max(1e-6, px_s * 1.0025)) * 10000.0) / 10000.0
        if q_s >= min_amt and px_s > 0:
            b_tasks.append((s, q_s, px_s))
            rem_batch_usd = max(0.0, rem_batch_usd - q_s * px_s * 1.0025)
        else:
            trader.order3_engine.api_tokens = min(LIVE_TOKEN_BUCKET_CAP, trader.order3_engine.api_tokens + 1.0)

    # 4. Постановка Ask-слотов 1-в-1 с fused_order3_clean_tick.py (эндогенный резерв renorm_blend):
    res_a_qty = sum(float(s.get("qty", 0.0)) for s in trader.ask_slots if s.get("id") is not None)
    avail_a_qty = max(0.0, min(core_pos_qty - res_a_qty, float(getattr(trader, "ex_coin_free", core_pos_qty))))

    cand_a_slots = [
        s for s in sorted(range(12), key=lambda i: float(plan["pi_ask"][i]), reverse=True)
        if core_pos_qty >= min_amt and avail_a_qty >= min_amt and trader.ask_slots[s].get("id") is None and bool(plan["g_plc_a"][s]) and float(plan["pi_ask"][s]) > 1e-4
    ]
    sel_a = []
    for s in cand_a_slots:
        if trader.order3_engine.api_tokens >= 1.0:
            sel_a.append(s)
            trader.order3_engine.api_tokens -= 1.0

    sum_pi_a = sum(float(plan["pi_ask"][s]) + 1e-5 for s in sel_a)
    a_tasks = []
    for s in sel_a:
        px_s = round(max(min_safe_ask, float(plan["t_ask_px"][s])), p_prec)
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
                    print(f"  📥 [NEURAL PLACE BID #{s:02d}] id={oid_n} | {q_act:.4f} PRL @ ${p_act:.{p_prec}f} (Mode#{m_idx}={m_prob:.0%} | E={exp_b:.0%}) | t={_msk_ms()}")
                else:
                    init_qa_a = 0.0 if p_act < (a1 - 0.1 * t_sz) else float(plan["t_vis_aq"][s])
                    trader.ex_coin_free = max(0.0, float(getattr(trader, "ex_coin_free", 0.0)) - q_act)
                    trader.ask_slots[s] = {"slot": s, "id": oid_n, "px": p_act, "qty": q_act, "act_ts": now_ts, "acc_fill": 0.0, "qa": init_qa_a, "qa0": init_qa_a}
                    print(f"  📤 [NEURAL PLACE ASK #{s:02d}] id={oid_n} | {q_act:.4f} PRL @ ${p_act:.{p_prec}f} (Mode#{m_idx}={m_prob:.0%}) | t={_msk_ms()}")

    if b_tasks or a_tasks or to_cnl_b or to_cnl_a:
        trader._save_wallet()
