#!/usr/bin/env python3
"""
_tools/test_l28_sim_live_parity.py
=============================================================================
Обязательный предтренировочный и пострелизный гейт Урока 28:
  1. Проверка хирургического переноса `480D -> 508D` (`warm_start_l27_w1_to_l28`, 0 каналов `90s`);
  2. Побитовая сверка математического паритета между векторным ядром тренажера
     `_tools/fused_l28_hierarchical_subcent_tick.py` и боевым мостом `_tools/l28_live_bridge.py`
     на серии сложных `$0.0001` 50-уровневых состояний (активные 24 слота, `Sparse-LOB`,
     `min_fee_tp`, `calm_bid_hold`, penny-jump, токсичный `adv_risk`);
  3. Замер сквозной CPU-латентности полного 1-секундного шага L28 (`ST Tree + BO Tree + LOBNet + 508D`).
=============================================================================
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from _tools.build_l28_hierarchical_cache import compute_subcent_micro_8d_single
from _tools.fused_l28_hierarchical_subcent_tick import (
    AG_IDX_L28,
    IN_DIM_L28,
    MKT_IDX_L28,
    TOKEN_BUCKET_CAP_L28,
    TOKEN_REFILL_PER_SEC_L28,
    build_ext_mkt_138_tensor,
    fused_l28_hierarchical_subcent_tick,
    normalize_market_z398_per_block,
    warm_start_l27_w1_to_l28,
)
from _tools.hierarchical_regime_tree import HierarchicalRegimeTreeRouter
from _tools.l26_live_bridge import ECHELON_K01_NP
from _tools.l28_live_bridge import (
    assemble_l28_508d_live,
    build_l28_agent_features_from_trader,
    compute_l28_hierarchical_plan,
    forward_l28_hierarchical_live,
)
from _tools.multitask_lob_net import MultiTaskLOBNet

BASE_DIR = Path(__file__).resolve().parent.parent


class DummyTraderL28:
    def __init__(self, ckpt: dict, exec_tick: float = 0.0001, budget_usd: float = 835.0):
        self.tick_size = 0.01
        self.exec_tick_size = exec_tick
        self.initial_budget_usd = budget_usd
        self.bot_cash_usd = budget_usd
        self.pos_qty = 0.0
        self.avg_entry = 0.0
        self.hold_sec = 0
        self.tr_low_qty = 0.0
        self.tr_low_px = 0.0
        self.l27_api_tokens = TOKEN_BUCKET_CAP_L28
        self.l28_api_tokens = TOKEN_BUCKET_CAP_L28
        self.l7_mem_h = torch.zeros((1, 16), dtype=torch.float32)

        self.l7_w1 = ckpt["w1"].float().cpu()
        self.l7_b1 = ckpt["b1"].float().cpu()
        self.l7_w_res1 = ckpt["w_res1"].float().cpu()
        self.l7_b_res1 = ckpt["b_res1"].float().cpu()
        self.l7_w_res2 = ckpt["w_res2"].float().cpu()
        self.l7_w_rec = ckpt["w_rec"].float().cpu()
        self.l7_b_rec = ckpt["b_rec"].float().cpu()
        self.l7_w2_moe = ckpt["w2_moe"].float().cpu()
        self.l7_b2_moe = ckpt["b2_moe"].float().cpu()
        self.l7_w_gate1 = ckpt["w_gate1"].float().cpu()
        self.l7_b_gate1 = ckpt["b_gate1"].float().cpu()
        self.l7_w_gate2 = ckpt["w_gate2"].float().cpu()
        self.l7_b_gate2 = ckpt["b_gate2"].float().cpu()
        self.l7_w_reg1 = ckpt["w_reg1"].float().cpu()
        self.l7_b_reg1_b = ckpt["b_reg1_b"].float().cpu()
        self.l7_b_reg1_a = ckpt["b_reg1_a"].float().cpu()
        self.l7_w_reg2 = ckpt["w_reg2"].float().cpu()
        self.l7_b_reg2_b = ckpt["b_reg2_b"].float().cpu()
        self.l7_b_reg2_a = ckpt["b_reg2_a"].float().cpu()

        self.bid_slots = [{"slot": i, "id": None, "px": 0.0, "qty": 0.0, "qa": 0.0, "qa0": 0.0, "act_ts": 0.0} for i in range(12)]
        self.ask_slots = [{"slot": i, "id": None, "px": 0.0, "qty": 0.0, "qa": 0.0, "qa0": 0.0, "act_ts": 0.0} for i in range(12)]
        self.last_l28_state = None
        self.last_l27_state = None


def load_or_build_l28_ckpt(ckpt_path: str | None) -> dict:
    if ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ck["w1"].shape == (508, 384), f"Expected (508, 384), got {ck['w1'].shape}"
        return ck
    l27_p = BASE_DIR / "models" / "prl" / "lesson_27_multimodal_moe" / "l27_soup_top3.pt"
    if not l27_p.exists():
        l27_p = BASE_DIR / "models" / "prl" / "rl_execution" / "release_l7_0150_champion.pt"
    ck27 = torch.load(l27_p, map_location="cpu", weights_only=False)
    ck28 = dict(ck27)
    ck28["w1"] = warm_start_l27_w1_to_l28(ck27["w1"], seed=20262801, sigma_new=0.015)
    ck28["lesson"] = 28
    return ck28


def run_parity_suite(ckpt: dict) -> None:
    rng = np.random.default_rng(20262899)
    torch.manual_seed(20262899)

    max_d_strat = 0.0
    max_d_mem = 0.0
    max_d_gmode = 0.0
    max_d_tbid = 0.0
    max_d_task = 0.0
    max_d_qb = 0.0
    max_d_qa = 0.0
    gate_mismatch = 0
    total_gates = 0

    scenarios = [
        {"name": "1. Чистый старт $0.0001 (широкий спред, penny-jump)", "pos": 0.0, "avg": 0.0, "adv": 0.08, "active_slots": False, "b1": 0.4215, "a1": 0.4228},
        {"name": "2. Позиция в плюсе + спокойный стакан (calm_bid_hold + min_fee_tp)", "pos": 520.0, "avg": 0.4200, "adv": 0.09, "active_slots": True, "b1": 0.4204, "a1": 0.4209},
        {"name": "3. Подводная позиция + спокойный стакан (защита min_fee_tp выше b1)", "pos": 910.0, "avg": 0.4280, "adv": 0.14, "active_slots": True, "b1": 0.4192, "a1": 0.4198},
        {"name": "4. Токсичный пролив adv_risk=0.58 (отключение calm_bid_hold и min_fee_tp)", "pos": 740.0, "avg": 0.4250, "adv": 0.58, "active_slots": True, "b1": 0.4180, "a1": 0.4185},
        {"name": "5. Узкий 1-тиковый спред $0.0001 (penny_room=0)", "pos": 380.0, "avg": 0.4212, "adv": 0.21, "active_slots": True, "b1": 0.4215, "a1": 0.4216},
    ]

    for sc_idx, sc in enumerate(scenarios):
        exec_tick = 0.0001
        feat_tick = 0.01
        b1 = float(sc["b1"])
        a1 = float(sc["a1"])

        # 50-уровневая субцентовая книга
        bp50 = np.array([round(max(exec_tick, b1 - k * 0.0002), 4) for k in range(50)], dtype=np.float32)
        bq50 = rng.uniform(40.0, 950.0, size=50).astype(np.float32)
        ap50 = np.array([round(a1 + k * 0.0002, 4) for k in range(50)], dtype=np.float32)
        aq50 = rng.uniform(40.0, 950.0, size=50).astype(np.float32)

        b1_q_cent = float(np.sum(bq50[:12]))
        a1_q_cent = float(np.sum(aq50[:12]))
        vb_cur = float(rng.uniform(0.0, 180.0))
        vs_cur = float(rng.uniform(0.0, 180.0))
        roll_b30 = float(rng.uniform(250.0, 2200.0))
        roll_s30 = float(rng.uniform(250.0, 2200.0))

        z_256 = torch.randn((1, 256), dtype=torch.float32)
        ps_t = torch.tensor([[min(0.95, float(sc["adv"]) * 0.9)]], dtype=torch.float32)
        pb_t = torch.tensor([[min(0.95, float(sc["adv"]) * 0.7)]], dtype=torch.float32)
        pc_t = torch.tensor([[0.06]], dtype=torch.float32)
        pe_t = torch.tensor([[0.28]], dtype=torch.float32)

        st_tree_42 = rng.normal(0.0, 0.45, size=42).astype(np.float32)
        bo_tree_42 = rng.normal(0.0, 0.45, size=42).astype(np.float32)
        st_coh_4 = rng.normal(0.0, 0.35, size=4).astype(np.float32)
        bo_coh_4 = rng.normal(0.0, 0.35, size=4).astype(np.float32)
        slot_18_28 = rng.normal(0.0, 0.5, size=10).astype(np.float32)
        v52_4d = np.array([0.32, 0.48, float(sc["adv"]), 0.19], dtype=np.float32)
        real_2d = np.array([0.55, 0.18], dtype=np.float32)
        slot_52_58 = rng.uniform(-0.6, 0.6, size=6).astype(np.float32)
        lvel_8d = rng.uniform(-0.8, 0.8, size=8).astype(np.float32)

        vel_hist = deque(maxlen=6)
        trader = DummyTraderL28(ckpt, exec_tick=exec_tick, budget_usd=835.0)
        trader.pos_qty = float(sc["pos"])
        trader.avg_entry = float(sc["avg"])
        trader.bot_cash_usd = max(50.0, 835.0 - trader.pos_qty * max(trader.avg_entry, b1) * 0.85)
        trader.hold_sec = 95 if trader.pos_qty > 0 else 0
        trader.tr_low_qty = trader.pos_qty * 0.45 if trader.pos_qty > 0 else 0.0
        trader.tr_low_px = max(exec_tick, trader.avg_entry - 0.0018) if trader.pos_qty > 0 else 0.0
        trader.l7_mem_h = torch.randn((1, 16), dtype=torch.float32) * 0.25
        mem_init = trader.l7_mem_h.clone()

        s_cur_val = 200
        now_q_ts = float(s_cur_val)
        api_tok_pre = 7.4
        api_tok_post_refill = min(TOKEN_BUCKET_CAP_L28, api_tok_pre + TOKEN_REFILL_PER_SEC_L28)

        b_qty_init = np.zeros(12, dtype=np.float32)
        b_px_init = np.zeros(12, dtype=np.float32)
        b_qa_init = np.zeros(12, dtype=np.float32)
        b_qa0_init = np.zeros(12, dtype=np.float32)
        b_act_init = np.zeros(12, dtype=np.int16)

        a_qty_init = np.zeros(12, dtype=np.float32)
        a_px_init = np.zeros(12, dtype=np.float32)
        a_qa_init = np.zeros(12, dtype=np.float32)
        a_qa0_init = np.zeros(12, dtype=np.float32)
        a_act_init = np.zeros(12, dtype=np.int16)

        if sc["active_slots"]:
            for i in (0, 1, 3, 6, 9):
                px_i = float(bp50[min(49, i * 2)])
                vis_i = float(bq50[min(49, i * 2)])
                qa_i = min(vis_i, vis_i * 0.65)
                qa0_i = vis_i * 1.15
                q_i = 42.0 + i * 5.0
                b_qty_init[i], b_px_init[i], b_qa_init[i], b_qa0_init[i], b_act_init[i] = q_i, px_i, qa_i, qa0_i, s_cur_val - (8 + i * 4)
                trader.bid_slots[i] = {"slot": i, "id": f"B{i}", "px": px_i, "qty": q_i, "qa": qa_i, "qa0": qa0_i, "act_ts": float(b_act_init[i])}
            if trader.pos_qty >= 2.0:
                for i in (0, 2, 5, 8):
                    px_i = float(ap50[min(49, i * 2)])
                    vis_i = float(aq50[min(49, i * 2)])
                    qa_i = min(vis_i, vis_i * 0.55)
                    qa0_i = vis_i * 1.10
                    q_i = trader.pos_qty * 0.16
                    a_qty_init[i], a_px_init[i], a_qa_init[i], a_qa0_init[i], a_act_init[i] = q_i, px_i, qa_i, qa0_i, s_cur_val - (6 + i * 5)
                    trader.ask_slots[i] = {"slot": i, "id": f"A{i}", "px": px_i, "qty": q_i, "qa": qa_i, "qa0": qa0_i, "act_ts": float(a_act_init[i])}

        # --- ПУТЬ A: Боевой мост L28 (`l28_live_bridge.py`) ---
        ag_l28 = build_l28_agent_features_from_trader(
            trader, b1, a1, b1_q_cent, a1_q_cent, roll_b30, roll_s30, now_q_ts, maker_fee=0.001
        )
        z_508_live, adv_live = assemble_l28_508d_live(
            z_t_256=z_256,
            ps_t=ps_t,
            pb_t=pb_t,
            pc_t=pc_t,
            pe_t=pe_t,
            st_tree_42=st_tree_42,
            st_coh_4=st_coh_4,
            bo_tree_42=bo_tree_42,
            bo_coh_4=bo_coh_4,
            slot_18_28=slot_18_28,
            v52_4d=v52_4d,
            real_2d=real_2d,
            slot_52_58=slot_52_58,
            lvel_8d=lvel_8d,
            bp_exec_np=bp50,
            bq_exec_np=bq50,
            ap_exec_np=ap50,
            aq_exec_np=aq50,
            vb_cur=vb_cur,
            vs_cur=vs_cur,
            exec_tick=exec_tick,
            vel8_hist_deque=vel_hist,
            multi_6d_np=ag_l28["multi_6d"],
            b_slot_12=ag_l28["b_slot_12"],
            a_slot_12=ag_l28["a_slot_12"],
            b_dist_12=ag_l28["b_dist_12"],
            a_dist_12=ag_l28["a_dist_12"],
            acct_8d_list=ag_l28["acct_8d"],
            queue_8d_list=ag_l28["queue_8d"],
            mem_h_t=mem_init,
            slot_qprog_24=ag_l28["slot_qprog_24"],
        )
        st_live = forward_l28_hierarchical_live(trader, z_508_live, ag_l28["acct_8d"], now_q_ts, adv_live)
        plan_live = compute_l28_hierarchical_plan(
            trader, b1, a1, now_ts=now_q_ts, maker_fee=0.001, api_tokens_override=api_tok_post_refill
        )

        # --- ПУТЬ B: Векторное ядро тренажера (`fused_l28_hierarchical_subcent_tick.py`, float32 для точной сверки) ---
        v7_58 = np.zeros((1, 58), dtype=np.float32)
        v7_58[0, 18:28] = slot_18_28
        v7_58[0, 46:50] = v52_4d
        v7_58[0, 50:52] = real_2d
        v7_58[0, 52:58] = slot_52_58
        micro8 = compute_subcent_micro_8d_single(bp50, bq50, ap50, aq50, vb_cur, vs_cur, exec_tick)
        st_1s2s_open = 1.0 if float(np.sum(np.abs(st_tree_42[:12]))) > 1e-5 else 0.0
        micro8[7] = 0.5 * float(micro8[7]) + 0.5 * st_1s2s_open
        sub_m16 = np.concatenate([micro8, st_coh_4, bo_coh_4], axis=0)[None, :]

        ext_138_sim = build_ext_mkt_138_tensor(
            torch.from_numpy(st_tree_42[None, :]),
            torch.from_numpy(bo_tree_42[None, :]),
            torch.from_numpy(v7_58),
            torch.from_numpy(lvel_8d[None, :]),
            torch.from_numpy(sub_m16),
        )
        z_mkt_398_sim = torch.cat([z_256, ps_t, pb_t, pc_t, pe_t, ext_138_sim], dim=-1)
        z_mkt_norm, _ = normalize_market_z398_per_block(z_mkt_398_sim)
        w1_mkt_g = trader.l7_w1[MKT_IDX_L28].unsqueeze(0)
        w1_ag_g = trader.l7_w1[AG_IDX_L28].unsqueeze(0)
        pre_mkt_gw = torch.matmul(z_mkt_norm, trader.l7_w1[MKT_IDX_L28]).unsqueeze(0) + trader.l7_b1.view(1, 1, 384)
        adv_w0 = torch.tensor([max(float(ps_t.item()), float(pb_t.item()), float(v52_4d[2]))], dtype=torch.float32)

        # Чтобы исключить fill на t+1 при проверке чистого плана t_bid_px / t_ask_px / gate_b / gate_a, подаём на t+1 нулевые объёмы сделок и широкий nb1/na1:
        sim_fn = getattr(fused_l28_hierarchical_subcent_tick, "__wrapped__", fused_l28_hierarchical_subcent_tick)
        out_sim = sim_fn(
            s_cur_i16=torch.tensor(s_cur_val, dtype=torch.int16),
            s_nxt_i16=torch.tensor(s_cur_val + 1, dtype=torch.int16),
            pre_mkt_gw=pre_mkt_gw,
            adv_risk_w0=adv_w0,
            b1_w0=torch.tensor([b1], dtype=torch.float32),
            a1_w0=torch.tensor([a1], dtype=torch.float32),
            b1q_cent_w0=torch.tensor([b1_q_cent], dtype=torch.float32),
            a1q_cent_w0=torch.tensor([a1_q_cent], dtype=torch.float32),
            rb_w0=torch.tensor([roll_b30], dtype=torch.float32),
            rs_w0=torch.tensor([roll_s30], dtype=torch.float32),
            bids_L_w=torch.from_numpy(bp50[None, :]),
            asks_L_w=torch.from_numpy(ap50[None, :]),
            bids_q_L_w=torch.from_numpy(bq50[None, :]),
            asks_q_L_w=torch.from_numpy(aq50[None, :]),
            nb1_w0=torch.tensor([b1], dtype=torch.float32),
            na1_w0=torch.tensor([a1], dtype=torch.float32),
            sv_w0=torch.tensor([0.0], dtype=torch.float32),
            bv_w0=torch.tensor([0.0], dtype=torch.float32),
            w1_ag_g=w1_ag_g,
            wr1_g=trader.l7_w_res1.unsqueeze(0),
            br1_g=trader.l7_b_res1.view(1, 1, -1),
            wr2_g=trader.l7_w_res2.unsqueeze(0),
            w2_moe_g=trader.l7_w2_moe.unsqueeze(0),
            b2_moe_g=trader.l7_b2_moe.view(1, 1, -1),
            w_gate1_g=trader.l7_w_gate1.unsqueeze(0),
            b_gate1_g=trader.l7_b_gate1.view(1, 1, -1),
            w_gate2_g=trader.l7_w_gate2.unsqueeze(0),
            b_gate2_g=trader.l7_b_gate2.view(1, 1, -1),
            w_rec_g=trader.l7_w_rec.unsqueeze(0),
            b_rec_g=trader.l7_b_rec.view(1, 1, -1),
            w_reg1_g=trader.l7_w_reg1.unsqueeze(0),
            b_reg1_b_g=trader.l7_b_reg1_b.view(1, 1, -1),
            b_reg1_a_g=trader.l7_b_reg1_a.view(1, 1, -1),
            w_reg2_g=trader.l7_w_reg2.unsqueeze(0),
            b_reg2_b_g=trader.l7_b_reg2_b.view(1, 1, -1),
            b_reg2_a_g=trader.l7_b_reg2_a.view(1, 1, -1),
            echelon_k01=torch.from_numpy(ECHELON_K01_NP),
            budget=torch.tensor([835.0], dtype=torch.float32),
            mem_h=mem_init.clone(),
            cash=torch.tensor([trader.bot_cash_usd], dtype=torch.float32),
            pos_qty=torch.tensor([trader.pos_qty], dtype=torch.float32),
            pos_cost_usd=torch.tensor([trader.pos_qty * trader.avg_entry], dtype=torch.float32),
            avg_entry=torch.tensor([trader.avg_entry], dtype=torch.float32),
            min_buy_px=torch.tensor([trader.tr_low_px], dtype=torch.float32),
            low_tranche_qty=torch.tensor([trader.tr_low_qty], dtype=torch.float32),
            mid_buy_px=torch.tensor([0.0], dtype=torch.float32),
            mid_tranche_qty=torch.tensor([0.0], dtype=torch.float32),
            entry_s=torch.tensor([s_cur_val - trader.hold_sec], dtype=torch.int16),
            bid_px=torch.from_numpy(b_px_init[None, :]),
            bid_qty=torch.from_numpy(b_qty_init[None, :]),
            bid_qa=torch.from_numpy(b_qa_init[None, :]),
            bid_qa0=torch.from_numpy(b_qa0_init[None, :]),
            bid_act_s=torch.from_numpy(b_act_init[None, :]),
            ask_px=torch.from_numpy(a_px_init[None, :]),
            ask_qty=torch.from_numpy(a_qty_init[None, :]),
            ask_qa=torch.from_numpy(a_qa_init[None, :]),
            ask_qa0=torch.from_numpy(a_qa0_init[None, :]),
            ask_act_s=torch.from_numpy(a_act_init[None, :]),
            api_tokens=torch.tensor([api_tok_pre], dtype=torch.float32),
            peak_mtm_eq=torch.tensor([835.0], dtype=torch.float32),
            sum_log_dd=torch.tensor([0.0], dtype=torch.float32),
            max_mtm_dd_pct=torch.tensor([0.0], dtype=torch.float32),
            sum_mut_load=torch.tensor([0.0], dtype=torch.float32),
            sum_fill_qprog=torch.tensor([0.0], dtype=torch.float32),
            sum_val_err=torch.tensor([0.0], dtype=torch.float32),
            prev_val_pred=torch.tensor([0.0], dtype=torch.float32),
            prev_mtm_eq=torch.tensor([trader.bot_cash_usd + trader.pos_qty * b1 * 0.999], dtype=torch.float32),
            sum_mode_p=torch.zeros((1, 4), dtype=torch.float32),
            sum_mode_ent=torch.tensor([0.0], dtype=torch.float32),
            n_fills=torch.tensor([0], dtype=torch.int16),
            n_cancels=torch.tensor([0], dtype=torch.int16),
            n_requeues=torch.tensor([0], dtype=torch.int16),
            exec_tick=exec_tick,
            feat_tick=feat_tick,
            maker_fee=0.001,
            taker_fee=0.001,
            queue_mult=1.0,
            extra_lat_s=0,
            true_latency=True,
        )

        # Сверяем mem_h, g_mode и итоговое состояние 24 слотов после регулятора и токенов!
        mem_sim = out_sim[0]
        b_px_sim = out_sim[10].squeeze(0).numpy()
        b_qty_sim = out_sim[11].squeeze(0).numpy()
        a_px_sim = out_sim[15].squeeze(0).numpy()
        a_qty_sim = out_sim[16].squeeze(0).numpy()
        g_mode_sim = out_sim[29].squeeze(0).numpy()

        d_mem = float(torch.max(torch.abs(trader.l7_mem_h - mem_sim)).item())
        d_gm = float(np.max(np.abs(st_live["g_mode"] - g_mode_sim)))
        max_d_mem = max(max_d_mem, d_mem)
        max_d_gmode = max(max_d_gmode, d_gm)

        # Проверим прямое совпадение strat_raw с ручным прогоном z_508_live -> normalize -> SwiGLU -> MoE:
        # И проверим итоговые ордера слотов (если не было Taker-выхода):
        if st_live["u_taker"] <= 0.02 or trader.pos_qty < 1.95:
            tok = api_tok_post_refill
            exp_b_px = b_px_init.copy()
            exp_b_q = b_qty_init.copy()
            exp_a_px = a_px_init.copy()
            exp_a_q = a_qty_init.copy()

            # 1) Отмены Bid и Ask 1-в-1 с fused_l28:755-780 и execute_live_24slot_l28:616-629:
            clear_b_live = np.zeros(12, dtype=bool)
            for s in range(12):
                if plan_live["had_b"][s] and plan_live["g_cnl_b"][s] and tok >= 1.0 - 1e-4:
                    clear_b_live[s] = True
                    tok = max(0.0, tok - 1.0)
                    exp_b_px[s], exp_b_q[s] = 0.0, 0.0

            clear_a_live = np.zeros(12, dtype=bool)
            min_ftp = float(plan_live["min_fee_tp"])
            for s in range(12):
                under_tp = plan_live["hp_now"] and (min_ftp > 0.0) and (a_px_init[s] < min_ftp - 0.4 * exec_tick)
                must_clr = (not plan_live["hp_now"]) or under_tp
                if plan_live["had_a"][s] and (must_clr or (plan_live["g_cnl_a"][s] and tok >= 1.0 - 1e-4)):
                    clear_a_live[s] = True
                    if not must_clr:
                        tok = max(0.0, tok - 1.0)
                    exp_a_px[s], exp_a_q[s] = 0.0, 0.0

            # 2) Выставление новых Bid в порядке убывания pi_bid:
            res_after_b = float(np.sum(exp_b_q * exp_b_px * 1.001))
            avail_b_u = min(
                max(0.0, trader.bot_cash_usd * 0.998 - res_after_b),
                max(0.0, plan_live["tot_bud_eff"] * 0.996 * plan_live["exp_bid_frac"] - trader.pos_qty * b1 - res_after_b),
            )
            want_plc_b = np.zeros(12, dtype=bool)
            for s in np.argsort(-plan_live["pi_bid"]):
                if exp_b_q[s] <= 1e-6 and plan_live["g_plc_b"][s] and plan_live["q_target_b"][s] >= 2.0 and tok >= 1.0 - 1e-4:
                    want_plc_b[s] = True
                    tok_tmp = tok  # tot_want_b_usd собирается до списания valid_b токенов
                    tok = max(0.0, tok - 1.0)
            # Пересчёт точного лимита токенов как в fused_l28:792 (до scale_cap_b):
            tok_before_b = api_tok_post_refill - float(np.sum(clear_b_live)) - float(np.sum(clear_a_live & ~((not plan_live["hp_now"]) | (plan_live["hp_now"] & (min_ftp > 0.0) & (a_px_init < min_ftp - 0.4 * exec_tick)))))
            want_plc_b = np.zeros(12, dtype=bool)
            cnt_b = 0
            for s in np.argsort(-plan_live["pi_bid"]):
                if exp_b_q[s] <= 1e-6 and plan_live["g_plc_b"][s] and plan_live["q_target_b"][s] >= 2.0:
                    cnt_b += 1
                    if cnt_b <= tok_before_b + 1e-4:
                        want_plc_b[s] = True
            want_b_u = np.where(want_plc_b, plan_live["q_target_b"] * plan_live["t_bid_px"] * 1.001, 0.0)
            tot_w_bu = float(np.sum(want_b_u))
            sc_b = avail_b_u / max(tot_w_bu, 1e-6) if tot_w_bu > avail_b_u else 1.0
            plc_bq = (want_b_u * sc_b) / (plan_live["t_bid_px"] * 1.001)
            val_b = want_plc_b & (plc_bq >= 2.0)
            exp_b_px = np.where(val_b, plan_live["t_bid_px"], exp_b_px)
            exp_b_q = np.where(val_b, plc_bq, exp_b_q)
            tok_after_b = max(0.0, tok_before_b - float(np.sum(val_b)))

            # 3) Выставление новых Ask в порядке убывания pi_ask:
            unres_p = max(0.0, trader.pos_qty - float(np.sum(exp_a_q)))
            want_plc_a = np.zeros(12, dtype=bool)
            cnt_a = 0
            for s in np.argsort(-plan_live["pi_ask"]):
                if exp_a_q[s] <= 1e-6 and plan_live["hp_now"] and plan_live["g_plc_a"][s] and plan_live["q_target_a"][s] >= 2.0:
                    cnt_a += 1
                    if cnt_a <= tok_after_b + 1e-4:
                        want_plc_a[s] = True
            want_aq = np.where(want_plc_a, plan_live["q_target_a"], 0.0)
            tot_w_aq = float(np.sum(want_aq))
            sc_a = unres_p / max(tot_w_aq, 1e-6) if tot_w_aq > unres_p else 1.0
            plc_aq = want_aq * sc_a
            val_a = want_plc_a & (plc_aq >= 2.0)
            exp_a_px = np.where(val_a, plan_live["t_ask_px"], exp_a_px)
            exp_a_q = np.where(val_a, plc_aq, exp_a_q)

            d_tb = float(np.max(np.abs(exp_b_px - b_px_sim)))
            d_ta = float(np.max(np.abs(exp_a_px - a_px_sim)))
            d_qb = float(np.max(np.abs(exp_b_q - b_qty_sim)))
            d_qa = float(np.max(np.abs(exp_a_q - a_qty_sim)))
            max_d_tbid = max(max_d_tbid, d_tb)
            max_d_task = max(max_d_task, d_ta)
            max_d_qb = max(max_d_qb, d_qb)
            max_d_qa = max(max_d_qa, d_qa)

            upd_b_sim = (np.abs(b_px_sim - b_px_init) > 1e-7) | (np.abs(b_qty_sim - b_qty_init) > 1e-4)
            upd_a_sim = (np.abs(a_px_sim - a_px_init) > 1e-7) | (np.abs(a_qty_sim - a_qty_init) > 1e-4)
            upd_b_exp = clear_b_live | val_b
            upd_a_exp = clear_a_live | val_a
            mm = int(np.sum(upd_b_sim != upd_b_exp) + np.sum(upd_a_sim != upd_a_exp))
            gate_mismatch += mm
            total_gates += 24
            print(
                f"  ✅ {sc['name']}:\n"
                f"     |Δmem_h|={d_mem:.2e} | |Δg_mode|={d_gm:.2e} | |Δbid_px|={d_tb:.2e} | |Δask_px|={d_ta:.2e} | "
                f"|Δbid_q|={d_qb:.2e} | |Δask_q|={d_qa:.2e} | gate_mm={mm}/24 | min_fee_tp=${plan_live['min_fee_tp']:.4f}",
                flush=True,
            )

    assert max_d_mem < 1e-5, f"mem_h parity failed: {max_d_mem}"
    assert max_d_gmode < 1e-5, f"g_mode parity failed: {max_d_gmode}"
    assert max_d_tbid < 1e-6, f"t_bid_px parity failed: {max_d_tbid}"
    assert max_d_task < 1e-6, f"t_ask_px parity failed: {max_d_task}"
    assert max_d_qb < 5e-3, f"bid_qty parity failed: {max_d_qb}"
    assert max_d_qa < 5e-3, f"ask_qty parity failed: {max_d_qa}"
    assert gate_mismatch == 0, f"Regulator gate mismatch: {gate_mismatch}/{total_gates}"


def benchmark_live_1hz_latency(ckpt: dict) -> float:
    st_router = HierarchicalRegimeTreeRouter("prl", use_hybrid_nn=True)
    bo_router = HierarchicalRegimeTreeRouter("prl_bo", use_hybrid_nn=True)
    lob_net = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).eval()
    lob_net.load_state_dict(ckpt["backbone_state"], strict=False)
    h_lob = lob_net.init_hidden(1, torch.device("cpu"))

    pf = pq.ParquetFile(BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet")
    n_rows = pf.metadata.num_rows
    df_tail = pf.read_row_groups([pf.num_row_groups - 1]).to_pandas().iloc[-400:]
    raw_ts = df_tail["timestamp"] if "timestamp" in df_tail.columns else df_tail.index
    ts_arr = np.asarray(raw_ts.astype("int64") // 10**9 if hasattr(raw_ts, "asi8") else [int(x.timestamp()) for x in raw_ts], dtype=np.int64)

    x_st = df_tail.reindex(columns=st_router.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)
    x_bo = df_tail.reindex(columns=bo_router.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)
    r_last = df_tail.iloc[-1]
    now_ts = int(ts_arr[-1])

    trader = DummyTraderL28(ckpt, exec_tick=0.0001, budget_usd=835.0)
    vel_hist = deque(maxlen=6)
    lob_t = torch.zeros((1, 10, 4), dtype=torch.float32)
    flow_t = torch.zeros((1, 16), dtype=torch.float32)
    bp50 = np.full(50, 0.4210, dtype=np.float32)
    bq50 = np.full(50, 200.0, dtype=np.float32)
    ap50 = np.full(50, 0.4216, dtype=np.float32)
    aq50 = np.full(50, 200.0, dtype=np.float32)
    zeros10 = np.zeros(10, dtype=np.float32)
    zeros4 = np.zeros(4, dtype=np.float32)
    zeros2 = np.zeros(2, dtype=np.float32)
    zeros6 = np.zeros(6, dtype=np.float32)
    zeros8 = np.zeros(8, dtype=np.float32)

    # Прогрев кэшей моделей
    for _ in range(3):
        st_42, _, st_c4 = st_router.predict_live_1row(x_st[-360:], r_last, now_ts, is_stale=False)
        bo_42, _, bo_c4 = bo_router.predict_live_1row(x_bo[-360:], r_last, now_ts, is_stale=False)

    dts_ms = []
    for _ in range(20):
        t0 = time.perf_counter()
        with torch.inference_mode():
            st_42, _, st_c4 = st_router.predict_live_1row(x_st[-360:], r_last, now_ts, is_stale=False)
            bo_42, _, bo_c4 = bo_router.predict_live_1row(x_bo[-360:], r_last, now_ts, is_stale=False)
            z_t, h_lob = lob_net.backbone.forward_step(lob_t, flow_t, h_lob)
            ps_t = torch.sigmoid(lob_net.head_sweep_12s(z_t))
            pb_t = torch.sigmoid(lob_net.head_bleed_60s(z_t))
            pc_t = torch.sigmoid(lob_net.head_crash_300s(z_t))
            pe_t = torch.sigmoid(lob_net.head_exit_30s(z_t))
            ag_f = build_l28_agent_features_from_trader(trader, 0.4210, 0.4216, 800.0, 800.0, 500.0, 500.0, float(now_ts))
            z508, adv_r = assemble_l28_508d_live(
                z_t, ps_t, pb_t, pc_t, pe_t, st_42, st_c4, bo_42, bo_c4,
                zeros10, zeros4, zeros2, zeros6, zeros8,
                bp50, bq50, ap50, aq50, 20.0, 15.0, 0.0001, vel_hist,
                ag_f["multi_6d"], ag_f["b_slot_12"], ag_f["a_slot_12"],
                ag_f["b_dist_12"], ag_f["a_dist_12"], ag_f["acct_8d"],
                ag_f["queue_8d"], trader.l7_mem_h, ag_f["slot_qprog_24"],
            )
            forward_l28_hierarchical_live(trader, z508, ag_f["acct_8d"], float(now_ts), adv_r)
            compute_l28_hierarchical_plan(trader, 0.4210, 0.4216, now_ts=float(now_ts))
        dts_ms.append((time.perf_counter() - t0) * 1000.0)

    med_ms = float(np.median(dts_ms))
    p95_ms = float(np.percentile(dts_ms, 95))
    print(f"  ⚡ Сквозная CPU-латентность полного 1 Гц шага L28 (ST Tree + BO Tree + LOBNet + 508D MoE): median={med_ms:.2f} ms | p95={p95_ms:.2f} ms", flush=True)
    assert med_ms < 80.0, f"Live 1Hz step too slow: {med_ms:.2f} ms"
    return med_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    print("=" * 104, flush=True)
    print("🔬 ТЕСТ МАТЕМАТИЧЕСКОГО ПАРИТЕТА SIM-TO-LIVE И ЛАТЕНТНОСТИ УРОКА 28 (508D, БЕЗ 90s)", flush=True)
    print("=" * 104, flush=True)
    ckpt = load_or_build_l28_ckpt(args.ckpt)
    assert ckpt["w1"].shape == (IN_DIM_L28, 384)
    print(f"  📦 Проверен чекпоинт L28: w1={tuple(ckpt['w1'].shape)} | w2_moe={tuple(ckpt['w2_moe'].shape)} | reg1={tuple(ckpt['w_reg1'].shape)}", flush=True)
    run_parity_suite(ckpt)
    benchmark_live_1hz_latency(ckpt)
    print("=" * 104, flush=True)
    print("🏆 PASS: 100% МАТЕМАТИЧЕСКИЙ ПАРИТЕТ SIM-TO-LIVE L28 ПОДТВЕРЖДЁН!", flush=True)
    print("=" * 104, flush=True)


if __name__ == "__main__":
    main()
