#!/usr/bin/env python3
"""
_tools/fused_l28_hierarchical_subcent_tick.py
=============================================================================
Урок 28 (Компонент 3a): Иерархический двухплощадочный субцентовый Стратег `508D`
(`398D Market + 110D Agent`, 0 каналов `90s`), 50-уровневый `Sparse-LOB` `$0.0001`,
честный пробой 50 уровней стакана, встроенный round-trip `min_fee_tp`, `calm_bid_hold`
и коэволюция с `26D -> 48 -> 2D` Регулятором и `K = 4` латентными экспертами.
=============================================================================
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor_cache")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch._dynamo
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 256

from _tools.fused_l25_regulator_tick import (
    MKT_IDX_L25 as MKT_IDX_L27,
    AG_IDX_L25 as AG_IDX_L27,
)
from _tools.fused_l26_endogenous_tick import (
    AG_SUBBLOCKS_110,
    _normalize_agent_z110_per_block,
)
from _tools.fused_l27_multimodal_tick import (
    K_MODES_L27 as K_MODES_L28,
    EXPERT_OUT_DIM_L27 as EXPERT_OUT_DIM_L28,
    MOE_OUT_DIM_L27 as MOE_OUT_DIM_L28,
    ROUTER_IN_DIM_L27 as ROUTER_IN_DIM_L28,
    ROUTER_H_DIM_L27 as ROUTER_H_DIM_L28,
    REG_IN_DIM_L27 as REG_IN_DIM_L28,
    TOKEN_REFILL_PER_SEC_L27 as TOKEN_REFILL_PER_SEC_L28,
    TOKEN_BUCKET_CAP_L27 as TOKEN_BUCKET_CAP_L28,
    _fanned_slot_softmax_l27 as _fanned_slot_softmax_l28,
)

# В Уроке 28 шина 508D упорядочена прозрачно: [0:398] = 398D Рынок, [398:508] = 110D Агент!
MKT_DIM_L28 = 398
AG_DIM_L28 = 110
IN_DIM_L28 = MKT_DIM_L28 + AG_DIM_L28  # 508

MKT_IDX_L28 = list(range(0, MKT_DIM_L28))
AG_IDX_L28 = list(range(MKT_DIM_L28, IN_DIM_L28))

# 14 семантических рыночных подблоков в 398D (полностью исключён коллинеарный горизонт 90s!):
MKT_SUBBLOCKS_398 = [
    ("Z_Backbone_GRU", 0, 256),               # 256D ($0.01 cent-invariant LOB GRU)
    ("Guard_Risk_Heads", 256, 260),           #   4D (p_sweep_12s, p_bleed_60s, p_crash_300s, p_exit_30s)
    ("ST_Micro_Tree_1_2s", 260, 272),         #  12D (NEW: SafeTrade Tree 1s[6] + 2s[6])
    ("ST_Short_Tree_5_12_25s", 272, 290),     #  18D (SafeTrade Tree 5s[6] + 12s[6] + 25s[6])
    ("CrossVenue_5s_LeadLag", 290, 300),      #  10D (Cross-Venue 5s Lead-Lag)
    ("ST_Macro_Tree_60_120s", 300, 312),      #  12D (SafeTrade Tree 60s[6] + 120s[6], БЕЗ 90s!)
    ("Carousel_Hazard_Realism", 312, 318),    #   6D (v52_4d + real_2d; idx 314 = p_adverse)
    ("CrossVenue_FirstOrder_6D", 318, 324),   #   6D (Cross-Venue 1st-Order probs)
    ("Tree_5s_Velocity_8D", 324, 332),        #   8D (5s diff of 7 ST vel + BO 5s vel, БЕЗ 90s!)
    ("Fast_5s_LOB_Impulse", 332, 340),        #   8D ($0.01 cent-invariant 5s LOB impulse)
    ("BO_Micro_Tree_1_2s", 340, 352),         #  12D (NEW: BigONE Tree 1s[6] + 2s[6])
    ("BO_Short_Tree_5_12_25s", 352, 370),     #  18D (BigONE Tree 5s[6] + 12s[6] + 25s[6])
    ("BO_Macro_Tree_60_120s", 370, 382),      #  12D (BigONE Tree 60s[6] + 120s[6], БЕЗ 90s!)
    ("SubCent_Micro_Coherence_16D", 382, 398),#  16D (NEW: 8D SubCent bps depth + 4D ST coh + 4D BO coh)
]
assert MKT_SUBBLOCKS_398[-1][2] == MKT_DIM_L28

# Индексы 8 скоростных каналов внутри `ext_mkt_138` (`[260:398]` -> локальные смещения `idx - 260`) для 5s дифференциала:
# ST vel channels: 1s(2), 2s(8), 5s(14), 12s(20), 25s(26), 60s(42), 120s(48) + BO 5s vel(94)
TREE_VEL8_LOCAL_OFFSETS = (2, 8, 14, 20, 26, 42, 48, 94)


def warm_start_l27_w1_to_l28(w1_480: torch.Tensor, seed: int = 20262801, sigma_new: float = 0.015) -> torch.Tensor:
    """
    Хирургический перенос входной матрицы `w1` из Урока 27 `[480, 384]` в Урок 28 `[508, 384]`:
      - вырезает ровно 12 коллинеарных каналов `90s` (`ST 90s` и `BO 90s`),
      - 1-в-1 переносит все 358 унаследованных рыночных и 110 агентских каналов,
      - мягко инициализирует 40 новых рецепторов (`ST 1s/2s`, `BO 1s/2s`, `SubCent Micro 16D`) с `sigma_new`.
    """
    w480 = w1_480.detach().float().cpu()
    assert w480.shape == (480, 384), f"Expected (480, 384), got {tuple(w480.shape)}"
    mkt_370 = w480[MKT_IDX_L27]  # [370, 384]
    ag_110 = w480[AG_IDX_L27]    # [110, 384]

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    mkt_398 = torch.randn((MKT_DIM_L28, 384), generator=gen, dtype=torch.float32) * float(sigma_new)

    mkt_398[0:260] = mkt_370[0:260]          # Z_Backbone_GRU(256) + Guard_Risk_Heads(4)
    # [260:272] = NEW ST_Micro_Tree_1_2s (12D) -> N(0, sigma_new^2)
    mkt_398[272:290] = mkt_370[260:278]      # ST 5s, 12s, 25s (18D)
    mkt_398[290:300] = mkt_370[278:288]      # CrossVenue_5s_LeadLag (10D)
    mkt_398[300:306] = mkt_370[288:294]      # ST 60s (6D) — mkt_370[294:300] (ST 90s) УДАЛЁН!
    mkt_398[306:312] = mkt_370[300:306]      # ST 120s (6D)
    mkt_398[312:318] = mkt_370[306:312]      # Carousel_Hazard_Realism (6D)
    mkt_398[318:324] = mkt_370[312:318]      # CrossVenue_FirstOrder_6D (6D)
    mkt_398[324:332] = mkt_370[318:326]      # 5s Velocity 8D
    mkt_398[332:340] = mkt_370[326:334]      # Fast_5s_LOB_Impulse (8D)
    # [340:352] = NEW BO_Micro_Tree_1_2s (12D) -> N(0, sigma_new^2)
    mkt_398[352:370] = mkt_370[334:352]      # BO 5s, 12s, 25s (18D)
    mkt_398[370:376] = mkt_370[352:358]      # BO 60s (6D) — mkt_370[358:364] (BO 90s) УДАЛЁН!
    mkt_398[376:382] = mkt_370[364:370]      # BO 120s (6D)
    # [382:398] = NEW SubCent_Micro_Coherence_16D (16D) -> N(0, sigma_new^2)

    return torch.cat([mkt_398, ag_110], dim=0).contiguous()


def build_ext_mkt_138_tensor(
    st_tree_42: torch.Tensor,
    bo_tree_42: torch.Tensor,
    v7_58_legacy: torch.Tensor,
    lob_vel_8: torch.Tensor,
    subcent_micro_16: torch.Tensor,
) -> torch.Tensor:
    """
    Собирает 138 внешних рыночных каналов `[260:398]` на всём срезе `[N, 138]` без единого канала `90s`:
      - [  0: 12] ST Tree 1s, 2s (12)
      - [ 12: 30] ST Tree 5s, 12s, 25s (18)
      - [ 30: 40] CrossVenue 5s LeadLag (10, из v7_58[:, 18:28])
      - [ 40: 52] ST Tree 60s, 120s (12)
      - [ 52: 58] Carousel + Realism (6, из v7_58[:, 46:52])
      - [ 58: 64] CrossVenue 1stOrder 6D (6, из v7_58[:, 52:58])
      - [ 64: 72] Tree 5s Velocity 8D (8, 5s diff по 7 горизонтам ST + BO 5s)
      - [ 72: 80] Fast 5s LOB Impulse 8D (8, $0.01-инвариантный)
      - [ 80: 92] BO Tree 1s, 2s (12)
      - [ 92:110] BO Tree 5s, 12s, 25s (18)
      - [110:122] BO Tree 60s, 120s (12)
      - [122:138] SubCent Micro + Coherence 16D (16)
    """
    n = st_tree_42.shape[0]
    dev = st_tree_42.device
    st_f = st_tree_42.float()
    bo_f = bo_tree_42.float()
    v7_f = v7_58_legacy.float()

    st_1_2 = st_f[:, 0:12]
    st_5_25 = st_f[:, 12:30]
    cv_10 = v7_f[:, 18:28]
    st_60_120 = st_f[:, 30:42]
    car_real_6 = v7_f[:, 46:52]
    cv_prob_6 = v7_f[:, 52:58]

    bo_1_2 = bo_f[:, 0:12]
    bo_5_25 = bo_f[:, 12:30]
    bo_60_120 = bo_f[:, 30:42]

    # 8 осевых скоростей дерева для 5-секундного ускорения (7 горизонтов ST + BO 5s):
    vel_src_8 = torch.stack(
        [
            st_f[:, 2],   # ST 1s vel
            st_f[:, 8],   # ST 2s vel
            st_f[:, 14],  # ST 5s vel
            st_f[:, 20],  # ST 12s vel
            st_f[:, 26],  # ST 25s vel
            st_f[:, 32],  # ST 60s vel
            st_f[:, 38],  # ST 120s vel
            bo_f[:, 14],  # BO 5s vel
        ],
        dim=-1,
    )
    tree_vel_8 = torch.zeros((n, 8), dtype=torch.float32, device=dev)
    if n > 5:
        tree_vel_8[5:] = torch.clamp(vel_src_8[5:] - vel_src_8[:-5], -3.0, 3.0)

    ext_138 = torch.cat(
        [
            st_1_2,
            st_5_25,
            cv_10,
            st_60_120,
            car_real_6,
            cv_prob_6,
            tree_vel_8,
            lob_vel_8.float(),
            bo_1_2,
            bo_5_25,
            bo_60_120,
            subcent_micro_16.float(),
        ],
        dim=-1,
    )
    assert ext_138.shape == (n, 138)
    return ext_138.contiguous()


@torch.no_grad()
def precompute_window_market_z398(
    flow_mkt_w_all: torch.Tensor,
    lob_fu_w_all: torch.Tensor,
    ext138_w_all: torch.Tensor,
    bf16_runner,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Предвычисляет 398D рыночный ствол `[T_steps, W, 398]` + скалярный риск `adv_risk_w [T_steps, W]`
    (`max(p_sweep_12s, p_bleed_60s, p_adverse)` — 1-в-1 с `l28_live_bridge.py`).
    """
    t_steps, w_cnt, _ = flow_mkt_w_all.shape
    dt = lob_fu_w_all.dtype
    dev = lob_fu_w_all.device

    flow_16_all = torch.zeros((t_steps, w_cnt, 16), dtype=dt, device=dev)
    flow_16_all[:, :, :8] = flow_mkt_w_all[:, :, :8].to(dt)

    fl_all = F.gelu(F.layer_norm(flow_16_all @ bf16_runner.w_fl + bf16_runner.b_fl, [64], bf16_runner.ln_fl_w, bf16_runner.ln_fl_b))
    fu_raw_all = lob_fu_w_all + (fl_all @ bf16_runner.w_fu_fl)
    x_all = F.gelu(F.layer_norm(fu_raw_all, [256], bf16_runner.ln_fu_w, bf16_runner.ln_fu_b))
    gi_all = x_all @ bf16_runner.w_ih + bf16_runner.b_ih

    h = bf16_runner.init_hidden(batch_size=w_cnt)
    h_list = []
    for s in range(t_steps):
        gh = h @ bf16_runner.w_hh + bf16_runner.b_hh
        i_r, i_z, i_n = gi_all[s].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = torch.sigmoid(i_r + h_r)
        z_gate = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1.0 - z_gate) * n + z_gate * h
        h_list.append(h)

    h_stack = torch.stack(h_list, dim=0)  # [T, W, 256]
    n_tw = t_steps * w_cnt
    hh = F.gelu(h_stack.view(n_tw, 256) @ bf16_runner.w_h1 + bf16_runner.b_h1).view(n_tw, 4, 64).transpose(0, 1)
    p4 = torch.sigmoid(torch.bmm(hh, bf16_runner.w_h2).squeeze(-1).t() + bf16_runner.b_h2).view(t_steps, w_cnt, 4)

    ext_t = ext138_w_all[:t_steps, :, :138].to(dt)
    # p_adverse хранится в ext_138[:, :, 54] (Carousel_Hazard_Realism[2] -> глобальный индекс 314 в 398D)
    p_adv = ext_t[:, :, 54].float()
    adv_risk_w = torch.maximum(torch.maximum(p4[:, :, 0].float(), p4[:, :, 1].float()), p_adv).contiguous()

    z398 = torch.cat([h_stack, p4, ext_t], dim=-1).contiguous()
    return z398, adv_risk_w


def normalize_market_z398_per_block(z_mkt_398: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Каузальная по-блочная RMS-нормировка 14 семантических рыночных блоков 398D."""
    parts = []
    rms_stats = []
    zf = z_mkt_398.float()
    for _, s, e in MKT_SUBBLOCKS_398:
        blk = zf[..., s:e]
        rms = torch.sqrt(torch.mean(blk * blk, dim=-1, keepdim=True) + eps)
        rms_stats.append(rms.mean().item())
        parts.append((blk / torch.clamp(rms, min=0.05)).to(z_mkt_398.dtype))
    return torch.cat(parts, dim=-1).contiguous(), torch.tensor(rms_stats, dtype=torch.float32)


def normalize_full_508d_live(z_508: torch.Tensor) -> torch.Tensor:
    """Точная 1-строчная по-блочная RMS-нормировка 508D (`398D Market + 110D Agent`) для `live_rl_trader.py`."""
    zf = z_508.float()
    mkt_norm, _ = normalize_market_z398_per_block(zf[:, :MKT_DIM_L28])
    ag_norm = _normalize_agent_z110_per_block(zf[:, MKT_DIM_L28:], torch.float32)
    return torch.cat([mkt_norm, ag_norm], dim=-1)


def _lookup_sparse_bid_lob_q_l28(
    b_pL_1w: torch.Tensor,
    b_qL_1w: torch.Tensor,
    px_ns: torch.Tensor,
    W: int,
    E: int,
    S: int,
    et_w1: torch.Tensor,
) -> torch.Tensor:
    """Точный поиск объёма на уровне `px_ns` в разреженном `L`-уровневом (`L=50` на $0.0001) Bid-стакане."""
    max_i = b_pL_1w.shape[1] - 1
    px_wes = px_ns.view(W, E * S)
    idx = torch.clamp(torch.searchsorted(-b_pL_1w, -px_wes, right=False), 0, max_i)
    mp = torch.gather(b_pL_1w, 1, idx)
    mq = torch.gather(b_qL_1w, 1, idx)
    matched = torch.abs(mp - px_wes) < (0.45 * et_w1)
    return torch.where(matched, mq, torch.zeros_like(mq)).view(-1, S)


def _lookup_sparse_ask_lob_q_l28(
    a_pL_1w: torch.Tensor,
    a_qL_1w: torch.Tensor,
    px_ns: torch.Tensor,
    W: int,
    E: int,
    S: int,
    et_w1: torch.Tensor,
) -> torch.Tensor:
    """Точный поиск объёма на уровне `px_ns` в разреженном `L`-уровневом (`L=50` на $0.0001) Ask-стакане."""
    max_i = a_pL_1w.shape[1] - 1
    px_wes = px_ns.view(W, E * S)
    idx = torch.clamp(torch.searchsorted(a_pL_1w, px_wes, right=False), 0, max_i)
    mp = torch.gather(a_pL_1w, 1, idx)
    mq = torch.gather(a_qL_1w, 1, idx)
    matched = torch.abs(mp - px_wes) < (0.45 * et_w1)
    return torch.where(matched, mq, torch.zeros_like(mq)).view(-1, S)


@torch.compile(dynamic=False)
def fused_l28_hierarchical_subcent_tick(
    s_cur_i16: torch.Tensor,
    s_nxt_i16: torch.Tensor,
    pre_mkt_gw: torch.Tensor,
    adv_risk_w0: torch.Tensor,
    b1_w0: torch.Tensor,
    a1_w0: torch.Tensor,
    b1q_cent_w0: torch.Tensor,
    a1q_cent_w0: torch.Tensor,
    rb_w0: torch.Tensor,
    rs_w0: torch.Tensor,
    bids_L_w: torch.Tensor,
    asks_L_w: torch.Tensor,
    bids_q_L_w: torch.Tensor,
    asks_q_L_w: torch.Tensor,
    nb1_w0: torch.Tensor,
    na1_w0: torch.Tensor,
    sv_w0: torch.Tensor,
    bv_w0: torch.Tensor,
    w1_ag_g: torch.Tensor,
    wr1_g: torch.Tensor,
    br1_g: torch.Tensor,
    wr2_g: torch.Tensor,
    w2_moe_g: torch.Tensor,
    b2_moe_g: torch.Tensor,
    w_gate1_g: torch.Tensor,
    b_gate1_g: torch.Tensor,
    w_gate2_g: torch.Tensor,
    b_gate2_g: torch.Tensor,
    w_rec_g: torch.Tensor,
    b_rec_g: torch.Tensor,
    w_reg1_g: torch.Tensor,
    b_reg1_b_g: torch.Tensor,
    b_reg1_a_g: torch.Tensor,
    w_reg2_g: torch.Tensor,
    b_reg2_b_g: torch.Tensor,
    b_reg2_a_g: torch.Tensor,
    echelon_k01: torch.Tensor,
    budget: torch.Tensor,
    mem_h: torch.Tensor,
    cash: torch.Tensor,
    pos_qty: torch.Tensor,
    pos_cost_usd: torch.Tensor,
    avg_entry: torch.Tensor,
    min_buy_px: torch.Tensor,
    low_tranche_qty: torch.Tensor,
    mid_buy_px: torch.Tensor,
    mid_tranche_qty: torch.Tensor,
    entry_s: torch.Tensor,
    bid_px: torch.Tensor,
    bid_qty: torch.Tensor,
    bid_qa: torch.Tensor,
    bid_qa0: torch.Tensor,
    bid_act_s: torch.Tensor,
    ask_px: torch.Tensor,
    ask_qty: torch.Tensor,
    ask_qa: torch.Tensor,
    ask_qa0: torch.Tensor,
    ask_act_s: torch.Tensor,
    api_tokens: torch.Tensor,
    peak_mtm_eq: torch.Tensor,
    sum_log_dd: torch.Tensor,
    max_mtm_dd_pct: torch.Tensor,
    sum_mut_load: torch.Tensor,
    sum_fill_qprog: torch.Tensor,
    sum_val_err: torch.Tensor,
    prev_val_pred: torch.Tensor,
    prev_mtm_eq: torch.Tensor,
    sum_mode_p: torch.Tensor,
    sum_mode_ent: torch.Tensor,
    n_fills: torch.Tensor,
    n_cancels: torch.Tensor,
    n_requeues: torch.Tensor,
    exec_tick: float | torch.Tensor,
    feat_tick: float,
    maker_fee: float,
    taker_fee: float,
    queue_mult: float | torch.Tensor = 1.0,
    extra_lat_s: int | torch.Tensor = 0,
    true_latency: bool = True,
):
    W = b1_w0.shape[0]
    G = w1_ag_g.shape[0]
    N = cash.shape[0]
    E = N // W
    B = E // G
    M = W * B
    S = echelon_k01.shape[0]
    L_depth = bids_L_w.shape[1]
    dt = mem_h.dtype

    if isinstance(exec_tick, torch.Tensor):
        et_1w = exec_tick
    else:
        et_1w = torch.full((W,), float(exec_tick), dtype=torch.float32, device=b1_w0.device)
    qm_t = queue_mult if isinstance(queue_mult, torch.Tensor) else torch.tensor(float(queue_mult), dtype=torch.float32, device=b1_w0.device)
    el_i16 = extra_lat_s if isinstance(extra_lat_s, torch.Tensor) else torch.tensor(int(extra_lat_s), dtype=torch.int16, device=b1_w0.device)
    et_w1 = et_1w.unsqueeze(1)
    et_n = et_1w.unsqueeze(1).expand(W, E).reshape(N)
    et_n1 = et_n.unsqueeze(1)
    is_subcent_n = et_n < (feat_tick * 0.5)
    neigh_tol_n11 = torch.clamp(1.1 * et_n, min=1.1 * feat_tick).view(N, 1, 1)

    b1_1w = torch.maximum(b1_w0, et_1w)
    a1_1w = torch.maximum(a1_w0, b1_1w + et_1w)
    # Для масштабно-инвариантных `l1_bid_usd` / `l1_ask_usd` в `multi_6d` используем центовую корзину L1 (1-в-1 с live_rl_trader.py:2384!):
    b1_q_cent_1w = torch.clamp(b1q_cent_w0, min=0.0)
    a1_q_cent_1w = torch.clamp(a1q_cent_w0, min=0.0)
    l1_bid_usd_1w = torch.clamp(b1_1w * b1_q_cent_1w, min=100.0)
    l1_ask_usd_1w = torch.clamp(a1_1w * a1_q_cent_1w, min=100.0)
    roll_buy_1w = torch.clamp(rb_w0, min=10.0)
    roll_sell_1w = torch.clamp(rs_w0, min=10.0)

    b_pL_1w = torch.clamp(bids_L_w, min=0.0)
    b_qL_raw_1w = torch.clamp(bids_q_L_w, min=0.0)
    a_pL_1w = torch.where(asks_L_w > 0.0, torch.maximum(asks_L_w, b1_1w.unsqueeze(1) + et_w1), torch.full_like(asks_L_w, 999.0))
    a_qL_raw_1w = torch.clamp(asks_q_L_w, min=0.0)

    b1 = b1_1w.unsqueeze(1).expand(W, E).reshape(N)
    a1 = a1_1w.unsqueeze(1).expand(W, E).reshape(N)
    b1_q_raw = b_qL_raw_1w[:, 0].unsqueeze(1).expand(W, E).reshape(N)
    a1_q_raw = a_qL_raw_1w[:, 0].unsqueeze(1).expand(W, E).reshape(N)
    l1_bid_usd = l1_bid_usd_1w.unsqueeze(1).expand(W, E).reshape(N)
    l1_ask_usd = l1_ask_usd_1w.unsqueeze(1).expand(W, E).reshape(N)
    l1_bid_raw_usd = torch.clamp(b1 * b1_q_raw, min=100.0)
    l1_ask_raw_usd = torch.clamp(a1 * a1_q_raw, min=100.0)
    roll_buy_n = roll_buy_1w.unsqueeze(1).expand(W, E).reshape(N)
    roll_sell_n = roll_sell_1w.unsqueeze(1).expand(W, E).reshape(N)
    adv_risk = adv_risk_w0.unsqueeze(1).expand(W, E).reshape(N)

    api_tokens = torch.clamp(api_tokens + TOKEN_REFILL_PER_SEC_L28, 0.0, TOKEN_BUCKET_CAP_L28)

    s_cur_f = s_cur_i16.float()
    has_p = pos_qty > 1e-6
    has_pos = has_p.float()
    pos_val = pos_qty * b1
    tot_eq = torch.clamp(cash + pos_val, min=1.0)
    pos_ratio = torch.clamp(pos_val / tot_eq, 0.0, 1.5)
    res_cash_now = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    unreserved_cash_now = torch.clamp(cash - res_cash_now, min=0.0)
    free_cash_ratio = torch.clamp(unreserved_cash_now / tot_eq, 0.0, 1.0)

    eff_pos_scale = torch.clamp(pos_ratio / 0.25, 0.15, 1.0)
    unr_pct = torch.where(has_p, ((b1 - avg_entry) / torch.clamp(avg_entry, min=feat_tick)) * 25.0 * eff_pos_scale, torch.zeros_like(has_pos))
    hold_ratio = torch.where(has_p, torch.clamp((s_cur_f - entry_s.float()) / 300.0, 0.0, 2.0) * eff_pos_scale, torch.zeros_like(has_pos))

    vis_b_lvl_q0 = _lookup_sparse_bid_lob_q_l28(b_pL_1w, b_qL_raw_1w, bid_px, W, E, S, et_w1) * qm_t
    in_book_b0 = (bid_qty > 1e-6) & (bid_px <= b1.unsqueeze(1) + 0.1 * et_n1)
    bid_qa = torch.where(in_book_b0, torch.minimum(bid_qa, vis_b_lvl_q0), bid_qa)

    vis_a_lvl_q0 = _lookup_sparse_ask_lob_q_l28(a_pL_1w, a_qL_raw_1w, ask_px, W, E, S, et_w1) * qm_t
    in_book_a0 = (ask_qty > 1e-6) & (ask_px >= a1.unsqueeze(1) - 0.1 * et_n1)
    ask_qa = torch.where(in_book_a0, torch.minimum(ask_qa, vis_a_lvl_q0), ask_qa)

    act_b_mask = bid_qty > 1e-6
    act_a_mask = ask_qty > 1e-6
    has_bid = act_b_mask.any(dim=1).float()
    has_ask = act_a_mask.any(dim=1).float()
    n_bid_slots = act_b_mask.float().sum(dim=1) / 12.0
    n_ask_slots = act_a_mask.float().sum(dim=1) / 12.0

    acct_8d = torch.stack(
        [
            has_pos,
            pos_ratio,
            free_cash_ratio,
            torch.clamp(unr_pct, -2.5, 2.5),
            hold_ratio,
            has_bid,
            has_ask,
            n_bid_slots - n_ask_slots,
        ],
        dim=-1,
    ).to(dt)

    b_qa_rat_all = torch.where(act_b_mask, torch.clamp(bid_qa / (bid_qa0.float() + 10.0), 0.0, 1.5), torch.full_like(bid_qa, 2.0))
    best_b_idx = torch.argmin(torch.where(act_b_mask, b1.unsqueeze(1) - bid_px, torch.full_like(bid_px, 999.0)), dim=1, keepdim=True)
    best_b_qa = torch.gather(bid_qa, 1, best_b_idx).squeeze(1)
    best_b_qa0 = torch.gather(bid_qa0.float(), 1, best_b_idx).squeeze(1)
    best_b_act = torch.gather(bid_act_s, 1, best_b_idx).squeeze(1).float()
    best_b_rat = torch.where(has_bid > 0.5, torch.clamp(best_b_qa / (best_b_qa0 + 10.0), 0.0, 1.5), torch.ones_like(has_bid))
    best_b_tape = torch.where(has_bid > 0.5, torch.clamp(torch.log1p(best_b_qa / roll_sell_n) / 3.0, 0.0, 2.0), torch.zeros_like(has_bid))
    best_b_age = torch.where(has_bid > 0.5, torch.clamp((s_cur_f - best_b_act) / 120.0, 0.0, 2.0), torch.zeros_like(has_bid))
    deep_b_act = act_b_mask[:, 4:]
    deep_b_cnt = deep_b_act.float().sum(dim=1)
    deep_b_rat = torch.where(
        deep_b_cnt > 0.5,
        torch.where(deep_b_act, b_qa_rat_all[:, 4:], torch.zeros_like(b_qa_rat_all[:, 4:])).sum(dim=1) / torch.clamp(deep_b_cnt, min=1.0),
        torch.ones_like(has_bid),
    )

    a_qa_rat_all = torch.where(act_a_mask, torch.clamp(ask_qa / (ask_qa0.float() + 10.0), 0.0, 1.5), torch.full_like(ask_qa, 2.0))
    best_a_idx = torch.argmin(torch.where(act_a_mask, ask_px - a1.unsqueeze(1), torch.full_like(ask_px, 999.0)), dim=1, keepdim=True)
    best_a_qa = torch.gather(ask_qa, 1, best_a_idx).squeeze(1)
    best_a_qa0 = torch.gather(ask_qa0.float(), 1, best_a_idx).squeeze(1)
    best_a_act = torch.gather(ask_act_s, 1, best_a_idx).squeeze(1).float()
    best_a_rat = torch.where(has_ask > 0.5, torch.clamp(best_a_qa / (best_a_qa0 + 10.0), 0.0, 1.5), torch.ones_like(has_ask))
    best_a_tape = torch.where(has_ask > 0.5, torch.clamp(torch.log1p(best_a_qa / roll_buy_n) / 3.0, 0.0, 2.0), torch.zeros_like(has_ask))
    best_a_age = torch.where(has_ask > 0.5, torch.clamp((s_cur_f - best_a_act) / 120.0, 0.0, 2.0), torch.zeros_like(has_ask))
    q_imb_adv = (1.0 - torch.clamp(best_a_rat, 0.0, 1.0)) * has_ask - (1.0 - torch.clamp(best_b_rat, 0.0, 1.0)) * has_bid

    queue_8d = torch.stack([best_b_rat, best_b_tape, best_b_age, deep_b_rat, best_a_rat, best_a_tape, best_a_age, q_imb_adv], dim=-1).to(dt)

    bud_l1 = torch.log1p(budget / l1_bid_usd) / 3.0
    cash_l1 = torch.log1p(unreserved_cash_now / l1_bid_usd) / 3.0
    pos_l1_ask = torch.clamp((pos_qty * a1) / l1_ask_usd, 0.0, 4.0) / 2.0
    pos_tape = torch.clamp(pos_qty / roll_buy_n, 0.0, 10.0) / 5.0
    low_tr_ratio = torch.where(has_p, torch.clamp(low_tranche_qty / torch.clamp(pos_qty, min=1e-6), 0.0, 1.0), torch.zeros_like(has_pos))
    lifo_ref = torch.where((low_tranche_qty > 1e-4) & (min_buy_px > 0.0), min_buy_px, avg_entry)
    lifo_edge_pct = torch.where(has_p, torch.clamp(((b1 - lifo_ref) / torch.clamp(lifo_ref, min=feat_tick)) * 30.0, -2.5, 2.5), torch.zeros_like(has_pos))
    multi_6d = torch.stack([bud_l1, cash_l1, pos_l1_ask, pos_tape, low_tr_ratio, lifo_edge_pct], dim=-1).to(dt)

    b_slot_12 = torch.clamp((bid_qty * b1.unsqueeze(1)) / torch.clamp(budget.unsqueeze(1), min=25.0), 0.0, 1.0).to(dt)
    a_slot_12 = torch.clamp(ask_qty / torch.clamp(pos_qty.unsqueeze(1), min=2.0), 0.0, 1.0).to(dt)
    b_dist_12 = torch.where(act_b_mask, torch.clamp(((b1.unsqueeze(1) - bid_px) / torch.clamp(b1.unsqueeze(1), min=feat_tick)) * 12.0, -0.5, 3.0), torch.zeros_like(bid_qty)).to(dt)
    a_dist_12 = torch.where(act_a_mask, torch.clamp(((ask_px - a1.unsqueeze(1)) / torch.clamp(a1.unsqueeze(1), min=feat_tick)) * 12.0, -0.5, 3.0), torch.zeros_like(ask_qty)).to(dt)

    b_qprog_12_f = torch.where(act_b_mask, torch.clamp(1.0 - b_qa_rat_all, -0.5, 1.0), torch.full_like(bid_qty, -1.0))
    a_qprog_12_f = torch.where(act_a_mask, torch.clamp(1.0 - a_qa_rat_all, -0.5, 1.0), torch.full_like(ask_qty, -1.0))

    # 1. Прямой проход 508D ствола L28 + GRU-16 микропамять:
    z_ag_raw = torch.cat(
        [multi_6d, b_slot_12, a_slot_12, b_dist_12, a_dist_12, acct_8d, queue_8d, mem_h, b_qprog_12_f.to(dt), a_qprog_12_f.to(dt)],
        dim=-1,
    )
    z_ag = _normalize_agent_z110_per_block(z_ag_raw, dt)
    z_ag_g = z_ag.view(W, G, B, 110).permute(1, 0, 2, 3).reshape(G, M, 110)
    pre_ag = torch.bmm(z_ag_g, w1_ag_g).view(G, W, B, 384)
    pre_all = (pre_ag + pre_mkt_gw.unsqueeze(2)).reshape(G, M, 384)

    h_base = torch.relu(pre_all[:, :, :256])
    h_gate = (2.0 * torch.sigmoid(pre_all[:, :, 256:384])).repeat_interleave(2, dim=-1)
    h1 = h_base * h_gate

    res_pre = torch.bmm(h1, wr1_g) + br1_g
    res_mid = torch.relu(res_pre[:, :, :64]) * (2.0 * torch.sigmoid(res_pre[:, :, 64:128]))
    h2 = h1 + torch.bmm(res_mid, wr2_g)

    rec_raw = torch.bmm(h2, w_rec_g) + b_rec_g
    rec_u = torch.tanh(rec_raw[:, :, :16])
    rec_gate = torch.sigmoid(rec_raw[:, :, 16:32] + 1.5)
    mem_prev_g = mem_h.view(W, G, B, 16).permute(1, 0, 2, 3).reshape(G, M, 16)
    mem_next_g = rec_gate * mem_prev_g + (1.0 - rec_gate) * rec_u
    mem_h = mem_next_g.view(G, W, B, 16).permute(1, 0, 2, 3).reshape(N, 16)

    # 1B. Контекстный Маршрутизатор (288D -> 64 -> 4D) и 4 Латентных Эксперта (K = 4 x 45D = 180D):
    acct_g = acct_8d.view(W, G, B, 8).permute(1, 0, 2, 3).reshape(G, M, 8)
    queue_g = queue_8d.view(W, G, B, 8).permute(1, 0, 2, 3).reshape(G, M, 8)
    router_in_g = torch.cat([h2, acct_g, queue_g, mem_next_g], dim=-1)
    r_hid_g = F.silu(torch.bmm(router_in_g, w_gate1_g) + b_gate1_g)
    gate_logits_g = (torch.bmm(r_hid_g, w_gate2_g) + b_gate2_g).float()
    g_mode_g = torch.softmax(gate_logits_g, dim=-1)

    moe_all_g = (torch.bmm(h2, w2_moe_g) + b2_moe_g).float().view(G, M, K_MODES_L28, EXPERT_OUT_DIM_L28)
    strat_raw_g = (g_mode_g.unsqueeze(-1) * moe_all_g).sum(dim=2)
    strat_raw = strat_raw_g.view(G, W, B, EXPERT_OUT_DIM_L28).permute(1, 0, 2, 3).reshape(N, EXPERT_OUT_DIM_L28)
    g_mode = g_mode_g.view(G, W, B, K_MODES_L28).permute(1, 0, 2, 3).reshape(N, K_MODES_L28)

    sum_mode_p = sum_mode_p + g_mode
    mode_ent_step = -(g_mode * torch.log(torch.clamp(g_mode, min=1e-6))).sum(dim=-1)
    sum_mode_ent = sum_mode_ent + mode_ent_step

    raw_ebid = torch.sigmoid(strat_raw[:, 0])
    exp_bid_frac = torch.clamp((raw_ebid - 0.03) / 0.97, 0.0, 1.0)

    tau_b = 0.30 + 2.00 * torch.sigmoid(strat_raw[:, 34:35])
    tau_a = 0.30 + 2.00 * torch.sigmoid(strat_raw[:, 35:36])
    max_share_b = 0.18 + 0.70 * torch.sigmoid(strat_raw[:, 44:45] - 0.65)
    max_share_a = 0.22 + 0.72 * torch.sigmoid(strat_raw[:, 44:45] - 0.25)
    pi_bid = _fanned_slot_softmax_l28(strat_raw[:, 1:13], tau_b, max_share_b, floor=0.008)
    pi_ask = _fanned_slot_softmax_l28(strat_raw[:, 13:25], tau_a, max_share_a, floor=0.008)

    # 2. Честный Taker-пробой всех `L` (`50` на $0.0001) уровней стакана в пределах `max_slip_pct`:
    u_taker = torch.clamp(torch.tanh(torch.relu(strat_raw[:, 25] - 1.10)), 0.0, 1.0)
    max_slip_pct = 0.005 + 0.145 * torch.sigmoid(strat_raw[:, 36])
    is_taker = (u_taker > 0.02) & (pos_qty >= 1.95) & (api_tokens >= 1.0)
    raw_sell_q = pos_qty * u_taker
    sell_q_req = torch.where((pos_qty - raw_sell_q) < 2.0, pos_qty, raw_sell_q)

    b_pL = b_pL_1w.unsqueeze(1).expand(W, E, L_depth).reshape(N, L_depth)
    b_qL_honest = b_qL_raw_1w.unsqueeze(1).expand(W, E, L_depth).reshape(N, L_depth)
    min_cross_px = torch.maximum(b1 * (1.0 - max_slip_pct), et_n).unsqueeze(1)
    elig_qL = torch.where((b_pL > 0.0) & (b_pL >= min_cross_px - 0.1 * et_n1), b_qL_honest, torch.zeros_like(b_qL_honest))
    cum_qL = torch.cumsum(elig_qL, dim=1)
    prev_qL = cum_qL - elig_qL
    take_L = torch.clamp(torch.minimum(sell_q_req.unsqueeze(1) - prev_qL, elig_qL), min=0.0)
    actual_taker_q = take_L.sum(dim=1)
    exec_taker = is_taker & (actual_taker_q >= 1e-4)

    proceeds = (take_L * b_pL).sum(dim=1) * (1.0 - taker_fee)
    rem_pos = torch.clamp(pos_qty - actual_taker_q, min=0.0)
    rem_frac = torch.where(pos_qty > 1e-6, rem_pos / torch.clamp(pos_qty, min=1e-6), torch.zeros_like(pos_qty))
    full_exit = rem_pos <= 1e-5

    cash = torch.where(exec_taker, cash + proceeds, cash)
    pos_qty = torch.where(exec_taker, rem_pos, pos_qty)
    pos_cost_usd = torch.where(exec_taker, pos_cost_usd * rem_frac, pos_cost_usd)
    avg_entry = torch.where(exec_taker & full_exit, torch.zeros_like(avg_entry), avg_entry)
    min_buy_px = torch.where(exec_taker & full_exit, torch.zeros_like(min_buy_px), min_buy_px)
    low_tranche_qty = torch.where(exec_taker, low_tranche_qty * rem_frac, low_tranche_qty)
    mid_buy_px = torch.where(exec_taker & full_exit, torch.zeros_like(mid_buy_px), mid_buy_px)
    mid_tranche_qty = torch.where(exec_taker, mid_tranche_qty * rem_frac, mid_tranche_qty)
    had_ask_taker_cnt = torch.where(exec_taker, (ask_qty > 1e-6).long().sum(dim=1), torch.zeros(N, dtype=torch.long, device=ask_qty.device))
    ask_qty = torch.where(exec_taker.unsqueeze(1), torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(exec_taker.unsqueeze(1), torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(exec_taker.unsqueeze(1), torch.zeros_like(ask_qa), ask_qa)
    ask_qa0 = torch.where(exec_taker.unsqueeze(1), torch.zeros_like(ask_qa0), ask_qa0)
    api_tokens = torch.where(exec_taker, torch.clamp(api_tokens - 1.0 - had_ask_taker_cnt.float(), min=0.0), api_tokens)
    n_fills = torch.clamp(n_fills.long() + exec_taker.long(), max=30000).to(torch.int16)

    # 3. Нативная субцентовая геометрия 24 слотов ($0.0001 penny-jump) + встроенный `min_fee_tp`:
    wide_sp = (a1 - b1) >= (2.0 * et_n - 1e-7)
    penny_room_b = torch.where(wide_sp, et_n, torch.zeros_like(b1))

    front_b_shift = -penny_room_b * torch.sigmoid(strat_raw[:, 37]) + b1 * 0.04 * F.softplus(strat_raw[:, 38] - 1.8) * 0.35
    span_b_usd = b1 * (0.015 + 0.235 * torch.sigmoid(strat_raw[:, 39]))
    gamma_b = 0.65 + 1.65 * torch.sigmoid(strat_raw[:, 40])

    curve_b = torch.pow(echelon_k01.unsqueeze(0), gamma_b.unsqueeze(1))
    raw_t_bid = b1.unsqueeze(1) - front_b_shift.unsqueeze(1) - span_b_usd.unsqueeze(1) * curve_b
    max_penny_b = torch.maximum(a1 - et_n, et_n).unsqueeze(1)
    t_bid_px = torch.maximum(torch.round(torch.minimum(raw_t_bid, max_penny_b) / et_n1) * et_n1, et_n1)

    hp_now = pos_qty >= 1.95
    hp_now_mask = hp_now.unsqueeze(1)
    ref_entry = torch.where(hp_now & (avg_entry > 0.0), avg_entry, a1)
    skew_blend = torch.sigmoid(strat_raw[:, 41] + 1.2)
    ask_edge_pct = 0.04 * torch.tanh(strat_raw[:, 42])
    span_a_usd = a1 * (0.012 + 0.238 * torch.sigmoid(strat_raw[:, 39] + 0.5 * strat_raw[:, 42]))
    gamma_a = 0.65 + 1.65 * torch.sigmoid(strat_raw[:, 40])

    anchor_base_a = a1 + skew_blend * (ref_entry - a1) + a1 * ask_edge_pct - penny_room_b * torch.sigmoid(-strat_raw[:, 41])

    roundtrip_fee_mult = (1.0 + maker_fee) / max(1e-6, 1.0 - maker_fee) + 0.0004
    use_fee_floor = hp_now & (avg_entry > 0.0) & (u_taker <= 0.02) & (adv_risk < 0.30) & is_subcent_n
    min_fee_tp = torch.where(
        use_fee_floor,
        torch.ceil((ref_entry * roundtrip_fee_mult) / et_n - 1e-6) * et_n,
        torch.zeros_like(ref_entry),
    )
    anchor_base_a = torch.where(use_fee_floor, torch.maximum(anchor_base_a, min_fee_tp), anchor_base_a)
    floor_ask = torch.maximum(b1 + et_n, min_fee_tp).unsqueeze(1)

    curve_a = torch.pow(echelon_k01.unsqueeze(0), gamma_a.unsqueeze(1))
    raw_t_ask = anchor_base_a.unsqueeze(1) + span_a_usd.unsqueeze(1) * curve_a
    t_ask_px = torch.maximum(floor_ask, torch.round(raw_t_ask / et_n1) * et_n1)

    val_pred_now = 0.05 * torch.tanh(strat_raw[:, 43])

    tot_bud_eff = torch.maximum(budget, cash + pos_qty * b1)
    target_bid_pool_usd = torch.clamp(tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1, min=0.0)
    q_target_b = (target_bid_pool_usd.unsqueeze(1) * pi_bid) / (t_bid_px * (1.0 + maker_fee))
    pair_small_b = ((q_target_b[:, 0::2] > 0.05) & (q_target_b[:, 0::2] < 2.0)) | ((q_target_b[:, 1::2] > 0.05) & (q_target_b[:, 1::2] < 2.0))
    pair_comb_b = q_target_b[:, 0::2] + q_target_b[:, 1::2]
    even_win_b = pi_bid[:, 0::2] >= pi_bid[:, 1::2]
    qb_ev = torch.where(pair_small_b, torch.where(even_win_b, pair_comb_b, torch.zeros_like(pair_comb_b)), q_target_b[:, 0::2])
    qb_od = torch.where(pair_small_b, torch.where(even_win_b, torch.zeros_like(pair_comb_b), pair_comb_b), q_target_b[:, 1::2])
    q_target_b = torch.stack([qb_ev, qb_od], dim=2).reshape(N, S)
    q_target_b = torch.where(q_target_b >= 2.0, q_target_b, torch.zeros_like(q_target_b))

    q_target_a = torch.where(hp_now_mask, pos_qty.unsqueeze(1) * pi_ask, torch.zeros_like(pi_ask))
    dust_a_mask = (q_target_a > 0.0) & (q_target_a < 2.0)
    dust_a_sum = torch.where(dust_a_mask, q_target_a, torch.zeros_like(q_target_a)).sum(dim=1, keepdim=True)
    q_target_a = torch.where(dust_a_mask, torch.zeros_like(q_target_a), q_target_a)
    best_a_t_idx = torch.argmax(pi_ask, dim=1, keepdim=True)
    best_a_oh = torch.zeros_like(q_target_a).scatter_(1, best_a_t_idx, 1.0)
    q_target_a = torch.where(hp_now_mask & (dust_a_sum >= 0.01) & (best_a_oh > 0.5), torch.minimum(pos_qty.unsqueeze(1), q_target_a + dust_a_sum), q_target_a)
    q_target_a = torch.where(q_target_a >= 2.0, q_target_a, torch.zeros_like(q_target_a))

    # 4. Модально-контекстный Послотовый Нейро-Регулятор (26D -> 48 -> 2D):
    ctx_4d_b = torch.tanh(strat_raw[:, 26:30])
    ctx_4d_a = torch.tanh(strat_raw[:, 30:34])
    s_2d = s_cur_i16.view(1, 1)
    next_s_2d = s_nxt_i16.view(1, 1)
    act_lat_2d = (next_s_2d + 1 + el_i16) if true_latency else next_s_2d

    had_b_pre = bid_qty > 1e-6
    had_a_pre = ask_qty > 1e-6
    age_b_norm = torch.where(had_b_pre, torch.log1p(torch.clamp((s_2d - bid_act_s).float(), min=0.0)) / 6.0, torch.zeros_like(bid_qty))
    age_a_norm = torch.where(had_a_pre, torch.log1p(torch.clamp((s_2d - ask_act_s).float(), min=0.0)) / 6.0, torch.zeros_like(ask_qty))

    bud_2d = torch.clamp(tot_bud_eff.unsqueeze(1), min=25.0)
    pos_2d = torch.clamp(pos_qty.unsqueeze(1), min=2.0)
    cur_val_b = torch.clamp((bid_qty * bid_px) / bud_2d, 0.0, 1.5)
    targ_val_b = torch.clamp((q_target_b * t_bid_px) / bud_2d, 0.0, 1.5)
    cur_val_a = torch.clamp(ask_qty / pos_2d, 0.0, 1.5)
    targ_val_a = torch.clamp(q_target_a / pos_2d, 0.0, 1.5)

    rel_delta_b = torch.clamp((q_target_b - bid_qty) / torch.maximum(torch.maximum(bid_qty, q_target_b), torch.full_like(bid_qty, 2.0)), -2.0, 2.0)
    rel_delta_a = torch.clamp((q_target_a - ask_qty) / torch.maximum(torch.maximum(ask_qty, q_target_a), torch.full_like(ask_qty, 2.0)), -2.0, 2.0)

    drift_b = torch.where(had_b_pre, torch.clamp(((t_bid_px - bid_px) / b1.unsqueeze(1)) * 25.0, -2.0, 2.0), torch.zeros_like(bid_qty))
    drift_a = torch.where(had_a_pre, torch.clamp(((ask_px - t_ask_px) / a1.unsqueeze(1)) * 25.0, -2.0, 2.0), torch.zeros_like(ask_qty))

    qprog_b_01 = torch.where(had_b_pre, torch.clamp(1.0 - b_qa_rat_all, 0.0, 1.0), torch.zeros_like(bid_qty))
    qprog_a_01 = torch.where(had_a_pre, torch.clamp(1.0 - a_qa_rat_all, 0.0, 1.0), torch.zeros_like(ask_qty))
    log_qa_b = torch.where(had_b_pre, torch.log1p(torch.clamp(bid_qa, min=0.0)) / 10.0, torch.zeros_like(bid_qty))
    log_qa_a = torch.where(had_a_pre, torch.log1p(torch.clamp(ask_qa, min=0.0)) / 10.0, torch.zeros_like(ask_qty))

    depth_b_norm = torch.clamp(((b1.unsqueeze(1) - t_bid_px) / b1.unsqueeze(1)) * 8.0, -0.2, 2.5)
    depth_a_norm = torch.clamp(((t_ask_px - a1.unsqueeze(1)) / a1.unsqueeze(1)) * 8.0, -0.2, 2.5)
    front_b_fl = (t_bid_px >= b1.unsqueeze(1) - 0.5 * et_n1).float()
    front_a_fl = (t_ask_px <= a1.unsqueeze(1) + 0.5 * et_n1).float()
    free_b_2d = free_cash_ratio.unsqueeze(1).expand(N, S)
    free_a_2d = torch.clamp((pos_qty - ask_qty.sum(dim=1)) / torch.clamp(pos_qty, min=2.0), 0.0, 1.0).unsqueeze(1).expand(N, S)

    tot_side_b = cur_val_b.sum(dim=1, keepdim=True).expand(N, S)
    tot_side_a = cur_val_a.sum(dim=1, keepdim=True).expand(N, S)
    near_b_mask = (torch.abs(bid_px.unsqueeze(2) - t_bid_px.unsqueeze(1)) <= neigh_tol_n11) & had_b_pre.unsqueeze(1)
    near_a_mask = (torch.abs(ask_px.unsqueeze(2) - t_ask_px.unsqueeze(1)) <= neigh_tol_n11) & had_a_pre.unsqueeze(1)
    eye_s = torch.eye(S, dtype=torch.bool, device=bid_px.device).unsqueeze(0)
    neigh_val_b = torch.clamp((near_b_mask & (~eye_s)).float().bmm(cur_val_b.unsqueeze(2)).squeeze(2), 0.0, 1.5)
    neigh_val_a = torch.clamp((near_a_mask & (~eye_s)).float().bmm(cur_val_a.unsqueeze(2)).squeeze(2), 0.0, 1.5)
    mean_qp_b = ((qprog_b_01 * cur_val_b).sum(dim=1, keepdim=True) / torch.clamp(cur_val_b.sum(dim=1, keepdim=True), min=0.05)).expand(N, S)
    mean_qp_a = ((qprog_a_01 * cur_val_a).sum(dim=1, keepdim=True) / torch.clamp(cur_val_a.sum(dim=1, keepdim=True), min=0.05)).expand(N, S)
    api_ratio_2d = (api_tokens / TOKEN_BUCKET_CAP_L28).unsqueeze(1).expand(N, S)
    ech_2d = echelon_k01.unsqueeze(0).expand(N, S)
    sp_rel_2d = torch.clamp(((a1 - b1) / b1) * 40.0, 0.0, 3.0).unsqueeze(1).expand(N, S)
    g_mode_s = g_mode.unsqueeze(1).expand(N, S, K_MODES_L28)

    feat_b_26 = torch.cat(
        [
            torch.stack(
                [
                    torch.zeros_like(bid_qty), had_b_pre.float(), cur_val_b, targ_val_b, rel_delta_b, drift_b,
                    qprog_b_01, log_qa_b, age_b_norm, depth_b_norm, front_b_fl, free_b_2d,
                ],
                dim=-1,
            ),
            ctx_4d_b.unsqueeze(1).expand(N, S, 4),
            torch.stack([tot_side_b, neigh_val_b, mean_qp_b, api_ratio_2d, ech_2d, sp_rel_2d], dim=-1),
            g_mode_s,
        ],
        dim=-1,
    ).to(dt)
    feat_a_26 = torch.cat(
        [
            torch.stack(
                [
                    torch.ones_like(ask_qty), had_a_pre.float(), cur_val_a, targ_val_a, rel_delta_a, drift_a,
                    qprog_a_01, log_qa_a, age_a_norm, depth_a_norm, front_a_fl, free_a_2d,
                ],
                dim=-1,
            ),
            ctx_4d_a.unsqueeze(1).expand(N, S, 4),
            torch.stack([tot_side_a, neigh_val_a, mean_qp_a, api_ratio_2d, ech_2d, sp_rel_2d], dim=-1),
            g_mode_s,
        ],
        dim=-1,
    ).to(dt)

    fb_g = feat_b_26.view(W, G, B, S, REG_IN_DIM_L28).permute(1, 0, 2, 3, 4).reshape(G, M * S, REG_IN_DIM_L28)
    fa_g = feat_a_26.view(W, G, B, S, REG_IN_DIM_L28).permute(1, 0, 2, 3, 4).reshape(G, M * S, REG_IN_DIM_L28)
    rb_h = F.silu(torch.bmm(fb_g, w_reg1_g) + b_reg1_b_g)
    ra_h = F.silu(torch.bmm(fa_g, w_reg1_g) + b_reg1_a_g)
    rb_out = (torch.bmm(rb_h, w_reg2_g) + b_reg2_b_g).float().view(G, W, B, S, 2).permute(1, 0, 2, 3, 4).reshape(N, S, 2)
    ra_out = (torch.bmm(ra_h, w_reg2_g) + b_reg2_a_g).float().view(G, W, B, S, 2).permute(1, 0, 2, 3, 4).reshape(N, S, 2)
    g_plc_b = rb_out[:, :, 0] > 0.0
    g_cnl_b = rb_out[:, :, 1] > 0.0
    g_plc_a = ra_out[:, :, 0] > 0.0
    g_cnl_a = ra_out[:, :, 1] > 0.0

    # 5. Token-Bucket API + встроенные `calm_bid_hold` и `under_fee_tp` (1-в-1 с `l28_live_bridge.py`!):
    calm_bid_hold = (exp_bid_frac > 0.08) & (adv_risk < 0.12) & (u_taker <= 0.02) & is_subcent_n
    slot_ge2 = torch.arange(S, device=bid_px.device).unsqueeze(0) >= 2
    hold_mask_b = calm_bid_hold.unsqueeze(1) & slot_ge2 & (torch.abs(t_bid_px - bid_px) < 0.0020) & (q_target_b >= 0.65 * bid_qty)

    want_cnl_b = had_b_pre & g_cnl_b & (~hold_mask_b)
    under_fee_tp = hp_now_mask & (min_fee_tp.unsqueeze(1) > 0.0) & (ask_px < min_fee_tp.unsqueeze(1) - 0.4 * et_n1)
    must_clear_a = (~hp_now_mask) | under_fee_tp
    want_cnl_a = had_a_pre & (must_clear_a | g_cnl_a)

    cum_cnl = want_cnl_b.float().cumsum(dim=1)
    clear_b = want_cnl_b & (cum_cnl <= api_tokens.unsqueeze(1) + 1e-4)
    api_tokens = torch.clamp(api_tokens - clear_b.float().sum(dim=1), min=0.0)

    paid_cnl_a = want_cnl_a & (~must_clear_a)
    cum_cnl_a = paid_cnl_a.float().cumsum(dim=1)
    clear_a = had_a_pre & (must_clear_a | (paid_cnl_a & (cum_cnl_a <= api_tokens.unsqueeze(1) + 1e-4)))
    api_tokens = torch.clamp(api_tokens - (clear_a & (~must_clear_a)).float().sum(dim=1), min=0.0)

    n_cnl_b_step = clear_b.long().sum(dim=1)
    n_cnl_a_step = clear_a.long().sum(dim=1) + had_ask_taker_cnt

    bid_qty = torch.where(clear_b, torch.zeros_like(bid_qty), bid_qty)
    bid_px = torch.where(clear_b, torch.zeros_like(bid_px), bid_px)
    bid_qa = torch.where(clear_b, torch.zeros_like(bid_qa), bid_qa)
    bid_qa0 = torch.where(clear_b, torch.zeros_like(bid_qa0), bid_qa0)

    ask_qty = torch.where(clear_a, torch.zeros_like(ask_qty), ask_qty)
    ask_px = torch.where(clear_a, torch.zeros_like(ask_px), ask_px)
    ask_qa = torch.where(clear_a, torch.zeros_like(ask_qa), ask_qa)
    ask_qa0 = torch.where(clear_a, torch.zeros_like(ask_qa0), ask_qa0)

    res_cash_after_cnl = (bid_qty * bid_px * (1.0 + maker_fee)).sum(dim=1)
    free_cash_for_b = torch.clamp(cash * 0.998 - res_cash_after_cnl, min=0.0)
    rem_exp_for_b = torch.clamp(tot_bud_eff * 0.996 * exp_bid_frac - pos_qty * b1 - res_cash_after_cnl, min=0.0)
    avail_b_usd = torch.minimum(free_cash_for_b, rem_exp_for_b)

    empty_b_now = bid_qty <= 1e-6
    cand_plc_b = empty_b_now & g_plc_b & (q_target_b >= 2.0)
    ord_b_idx = torch.argsort(torch.where(cand_plc_b, pi_bid, torch.full_like(pi_bid, -1.0)), dim=1, descending=True)
    inv_b_idx = torch.argsort(ord_b_idx, dim=1)
    cand_b_sorted = torch.gather(cand_plc_b.float(), 1, ord_b_idx)
    allow_b_tok_sorted = (cand_b_sorted.cumsum(dim=1) <= api_tokens.unsqueeze(1) + 1e-4) & (cand_b_sorted > 0.5)
    want_place_b = torch.gather(allow_b_tok_sorted, 1, inv_b_idx)

    want_b_usd = torch.where(want_place_b, q_target_b * t_bid_px * (1.0 + maker_fee), torch.zeros_like(q_target_b))
    tot_want_b_usd = want_b_usd.sum(dim=1, keepdim=True)
    scale_cap_b = torch.where(tot_want_b_usd > avail_b_usd.unsqueeze(1), avail_b_usd.unsqueeze(1) / torch.clamp(tot_want_b_usd, min=1e-6), torch.ones_like(tot_want_b_usd))
    plc_b_qty = (want_b_usd * scale_cap_b) / (t_bid_px * (1.0 + maker_fee))
    valid_b = want_place_b & (plc_b_qty >= 2.0)
    plc_b_qty = torch.where(valid_b, plc_b_qty, torch.zeros_like(plc_b_qty))
    n_plc_b_step = valid_b.long().sum(dim=1)
    api_tokens = torch.clamp(api_tokens - valid_b.float().sum(dim=1), min=0.0)
    n_req_b_step = (clear_b & valid_b).long().sum(dim=1)

    exact_b_q_2d = _lookup_sparse_bid_lob_q_l28(b_pL_1w, b_qL_raw_1w, t_bid_px, W, E, S, et_w1) * qm_t
    inside_sp_b = t_bid_px > (b1.unsqueeze(1) + 0.1 * et_n1)
    base_b_qa = torch.where(inside_sp_b, torch.zeros_like(exact_b_q_2d), exact_b_q_2d)
    plc_b_usd = plc_b_qty * t_bid_px
    b_over50 = torch.sigmoid((plc_b_usd - 50.0) / 18.0)
    b_depth_sat = F.softplus(plc_b_usd / l1_bid_raw_usd.unsqueeze(1) - 0.85)
    b_self_impact_qa = base_b_qa * (1.0 + 0.75 * b_over50) + b_over50 * (0.45 * plc_b_qty) + 0.35 * b_depth_sat * plc_b_qty

    bid_px = torch.where(valid_b, t_bid_px, bid_px)
    bid_qty = torch.where(valid_b, plc_b_qty, bid_qty)
    bid_qa = torch.where(valid_b, b_self_impact_qa, bid_qa)
    bid_qa0 = torch.where(valid_b, b_self_impact_qa.to(dt), bid_qa0)
    bid_act_s = torch.where(valid_b, act_lat_2d.expand_as(bid_act_s), bid_act_s)

    unreserved_pos = torch.clamp(pos_qty - ask_qty.sum(dim=1), min=0.0).unsqueeze(1)
    empty_a_now = ask_qty <= 1e-6
    cand_plc_a = empty_a_now & hp_now_mask & g_plc_a & (q_target_a >= 2.0)
    ord_a_idx = torch.argsort(torch.where(cand_plc_a, pi_ask, torch.full_like(pi_ask, -1.0)), dim=1, descending=True)
    inv_a_idx = torch.argsort(ord_a_idx, dim=1)
    cand_a_sorted = torch.gather(cand_plc_a.float(), 1, ord_a_idx)
    allow_a_tok_sorted = (cand_a_sorted.cumsum(dim=1) <= api_tokens.unsqueeze(1) + 1e-4) & (cand_a_sorted > 0.5)
    want_place_a = torch.gather(allow_a_tok_sorted, 1, inv_a_idx)

    want_a_q = torch.where(want_place_a, q_target_a, torch.zeros_like(q_target_a))
    tot_want_a_q = want_a_q.sum(dim=1, keepdim=True)
    scale_cap_a = torch.where(tot_want_a_q > unreserved_pos, unreserved_pos / torch.clamp(tot_want_a_q, min=1e-6), torch.ones_like(tot_want_a_q))
    plc_a_qty = want_a_q * scale_cap_a
    valid_a = want_place_a & (plc_a_qty >= 2.0)
    plc_a_qty = torch.where(valid_a, plc_a_qty, torch.zeros_like(plc_a_qty))
    n_plc_a_step = valid_a.long().sum(dim=1)
    api_tokens = torch.clamp(api_tokens - valid_a.float().sum(dim=1), min=0.0)
    n_req_a_step = (clear_a & valid_a).long().sum(dim=1)

    exact_a_q_2d = _lookup_sparse_ask_lob_q_l28(a_pL_1w, a_qL_raw_1w, t_ask_px, W, E, S, et_w1) * qm_t
    inside_sp_a = t_ask_px < (a1.unsqueeze(1) - 0.1 * et_n1)
    base_a_qa = torch.where(inside_sp_a, torch.zeros_like(exact_a_q_2d), exact_a_q_2d)
    plc_a_usd = plc_a_qty * t_ask_px
    a_over50 = torch.sigmoid((plc_a_usd - 50.0) / 18.0)
    a_depth_sat = F.softplus(plc_a_usd / l1_ask_raw_usd.unsqueeze(1) - 0.85)
    a_self_impact_qa = base_a_qa * (1.0 + 0.75 * a_over50) + a_over50 * (0.45 * plc_a_qty) + 0.35 * a_depth_sat * plc_a_qty

    ask_px = torch.where(valid_a, t_ask_px, ask_px)
    ask_qty = torch.where(valid_a, plc_a_qty, ask_qty)
    ask_qa = torch.where(valid_a, a_self_impact_qa, ask_qa)
    ask_qa0 = torch.where(valid_a, a_self_impact_qa.to(dt), ask_qa0)
    ask_act_s = torch.where(valid_a, act_lat_2d.expand_as(ask_act_s), ask_act_s)

    n_mut_tot = n_cnl_b_step + n_cnl_a_step + n_plc_b_step + n_plc_a_step
    if true_latency:
        burst_delay = ((n_mut_tot > 4) | (api_tokens < 2.0)).unsqueeze(1)
        extra_s = torch.where((api_tokens < 1.0).unsqueeze(1), (next_s_2d + 3 + el_i16).expand_as(bid_act_s), (next_s_2d + 2 + el_i16).expand_as(bid_act_s))
        bid_act_s = torch.where(valid_b & burst_delay, extra_s, bid_act_s)
        ask_act_s = torch.where(valid_a & burst_delay, extra_s, ask_act_s)

    # 6. Исполнение FIFO-очередей по полной `L`-уровневой книге (`L=50` на $0.0001) + 3-траншевый LIFO:
    nb1_1w = torch.maximum(nb1_w0, et_1w)
    na1_1w = torch.maximum(na1_w0, nb1_1w + et_1w)
    nb1 = nb1_1w.unsqueeze(1).expand(W, E).reshape(N, 1)
    na1 = na1_1w.unsqueeze(1).expand(W, E).reshape(N, 1)
    sv = sv_w0.unsqueeze(1).expand(W, E).reshape(N, 1)
    bv = bv_w0.unsqueeze(1).expand(W, E).reshape(N, 1)

    act_b = (bid_qty > 1e-6) & (next_s_2d >= bid_act_s)
    strict_cross_b = na1 <= (bid_px - 0.5 * et_n1)
    hit_b = act_b & (strict_cross_b | ((nb1 <= bid_px + 1e-6) & (sv > 0.0)))
    any_strict_b = (act_b & strict_cross_b).any(dim=1, keepdim=True)
    tot_flow_b = torch.where(any_strict_b, sv + 0.12 * b1_q_raw.unsqueeze(1), sv)

    even_act_b = act_b[:, 0::2] & (torch.abs(bid_px[:, 1::2] - bid_px[:, 0::2]) < 0.1 * et_n1)
    inc_qa_b = bid_qa.clone()
    inc_qa_b[:, 1::2] = torch.where(even_act_b, torch.clamp(bid_qa[:, 1::2] - bid_qa[:, 0::2], min=0.0), bid_qa[:, 1::2])
    eff_qa_b = torch.where(hit_b, inc_qa_b, torch.zeros_like(bid_qa))
    eff_qty_b = torch.where(hit_b, bid_qty, torch.zeros_like(bid_qty))

    slot_need_b = eff_qa_b + eff_qty_b
    cum_need_b = torch.cumsum(slot_need_b, dim=1)
    flow_before_slot_b = cum_need_b - slot_need_b

    cum_bq_1w = torch.cat([torch.zeros((W, 1), dtype=torch.float32, device=bid_px.device), torch.cumsum(b_qL_raw_1w * qm_t, dim=1)], dim=1)
    cnt_above_b = torch.searchsorted(-b_pL_1w, -(bid_px.view(W, E * S) + 0.5 * et_w1), right=False)
    mkt_above_b = torch.gather(cum_bq_1w, 1, cnt_above_b).view(N, S)
    rem_flow_at_slot_b = torch.clamp(tot_flow_b - torch.maximum(flow_before_slot_b, mkt_above_b), min=0.0)

    pre_fill_qp_b = torch.clamp(1.0 - bid_qa / (bid_qa0.float() + 10.0), 0.0, 1.0)
    qa_eaten_b = torch.minimum(eff_qa_b, rem_flow_at_slot_b)
    bid_qa = torch.where(hit_b, torch.clamp(bid_qa - qa_eaten_b, min=0.0), bid_qa)
    bid_qa[:, 1::2] = torch.where(act_b[:, 1::2] & even_act_b, torch.clamp(bid_qa[:, 1::2] - qa_eaten_b[:, 0::2], min=0.0), bid_qa[:, 1::2])

    want_b = torch.minimum(eff_qty_b, torch.clamp(rem_flow_at_slot_b - eff_qa_b, min=0.0))
    cum_want_b = torch.cumsum(want_b, dim=1)
    max_afford_b = torch.clamp(cash * 0.998 / (b1 * (1.0 + maker_fee)), min=0.0).unsqueeze(1)
    f_q = torch.minimum(want_b, torch.clamp(max_afford_b - (cum_want_b - want_b), min=0.0))

    has_slot_b = f_q > 1e-5
    tot_fq = f_q.sum(dim=1)
    tot_cost_raw = (f_q * bid_px).sum(dim=1)
    tot_cost = tot_cost_raw * (1.0 + maker_fee)
    has_buy = tot_fq > 1e-5
    n_fills = torch.clamp(n_fills.long() + has_buy.long(), max=30000).to(torch.int16)
    fill_qp_b_step = torch.where(has_buy, (f_q * pre_fill_qp_b).sum(dim=1) / torch.clamp(tot_fq, min=1e-6), torch.zeros_like(tot_fq))
    old_q = pos_qty
    new_q = old_q + tot_fq
    avg_fill_b_px = tot_cost_raw / torch.clamp(tot_fq, min=1e-6)
    first_buy = has_buy & (old_q <= 1e-6)
    no_low_yet = has_buy & (old_q > 1e-6) & (low_tranche_qty <= 1e-6)
    has_low_already = has_buy & (old_q > 1e-6) & (low_tranche_qty > 1e-6)

    dip_tol = 0.35 * feat_tick
    init_dip = no_low_yet & (avg_fill_b_px <= avg_entry - dip_tol)
    init_higher = no_low_yet & (~init_dip) & (old_q >= 1.95) & (avg_fill_b_px >= avg_entry + dip_tol)
    deeper_dip = has_low_already & (avg_fill_b_px < min_buy_px - dip_tol)
    same_low = has_low_already & (~deeper_dip) & (torch.abs(avg_fill_b_px - min_buy_px) <= dip_tol)
    mid_dip = has_low_already & (~deeper_dip) & (~same_low) & (avg_fill_b_px <= avg_entry - dip_tol)

    pushed_mid_q = mid_tranche_qty + low_tranche_qty
    pushed_mid_px = (mid_tranche_qty * mid_buy_px + low_tranche_qty * min_buy_px) / torch.clamp(pushed_mid_q, min=1e-6)
    added_mid_q = mid_tranche_qty + tot_fq
    added_mid_px = (mid_tranche_qty * mid_buy_px + tot_fq * avg_fill_b_px) / torch.clamp(added_mid_q, min=1e-6)

    new_mid_q = torch.where(first_buy, torch.zeros_like(mid_tranche_qty), torch.where(deeper_dip, pushed_mid_q, torch.where(mid_dip, added_mid_q, mid_tranche_qty)))
    new_mid_px = torch.where(first_buy, torch.zeros_like(mid_buy_px), torch.where(deeper_dip, pushed_mid_px, torch.where(mid_dip, added_mid_px, mid_buy_px)))
    same_low_q = low_tranche_qty + tot_fq
    same_low_px = (low_tranche_qty * min_buy_px + tot_fq * avg_fill_b_px) / torch.clamp(same_low_q, min=1e-6)

    low_tranche_qty = torch.where(first_buy, torch.zeros_like(low_tranche_qty), torch.where(init_dip | deeper_dip, tot_fq, torch.where(init_higher, old_q, torch.where(same_low, same_low_q, low_tranche_qty))))
    min_buy_px = torch.where(first_buy | init_dip | deeper_dip, avg_fill_b_px, torch.where(init_higher, avg_entry, torch.where(same_low, same_low_px, min_buy_px)))
    mid_tranche_qty = new_mid_q
    mid_buy_px = new_mid_px
    avg_entry = torch.where(has_buy, (old_q * avg_entry + tot_cost_raw) / torch.clamp(new_q, min=1e-6), avg_entry)
    pos_cost_usd = torch.where(has_buy, pos_cost_usd + tot_cost, pos_cost_usd)
    entry_s = torch.where(has_buy & (old_q <= 1e-6), s_nxt_i16.expand_as(entry_s), entry_s)
    cash = torch.where(has_buy, torch.clamp(cash - tot_cost, min=0.0), cash)
    pos_qty = torch.where(has_buy, new_q, pos_qty)

    rem_ord_b = torch.clamp(bid_qty - f_q, min=0.0)
    bid_qty = torch.where(has_slot_b, torch.where(rem_ord_b >= 1.0, rem_ord_b, torch.zeros_like(rem_ord_b)), bid_qty)

    act_a = (ask_qty > 1e-6) & (next_s_2d >= ask_act_s) & (pos_qty.unsqueeze(1) > 1e-6)
    strict_cross_a = nb1 >= (ask_px + 0.5 * et_n1)
    hit_a = act_a & (strict_cross_a | ((na1 >= ask_px - 1e-6) & (bv > 0.0)))
    any_strict_a = (act_a & strict_cross_a).any(dim=1, keepdim=True)
    tot_flow_a = torch.where(any_strict_a, bv + 0.12 * a1_q_raw.unsqueeze(1), bv)

    even_act_a = act_a[:, 0::2] & (torch.abs(ask_px[:, 1::2] - ask_px[:, 0::2]) < 0.1 * et_n1)
    inc_qa_a = ask_qa.clone()
    inc_qa_a[:, 1::2] = torch.where(even_act_a, torch.clamp(ask_qa[:, 1::2] - ask_qa[:, 0::2], min=0.0), ask_qa[:, 1::2])
    eff_qa_a = torch.where(hit_a, inc_qa_a, torch.zeros_like(ask_qa))
    eff_qty_a = torch.where(hit_a, ask_qty, torch.zeros_like(ask_qty))

    slot_need_a = eff_qa_a + eff_qty_a
    cum_need_a = torch.cumsum(slot_need_a, dim=1)
    flow_before_slot_a = cum_need_a - slot_need_a

    cum_aq_1w = torch.cat([torch.zeros((W, 1), dtype=torch.float32, device=ask_px.device), torch.cumsum(a_qL_raw_1w * qm_t, dim=1)], dim=1)
    cnt_below_a = torch.searchsorted(a_pL_1w, ask_px.view(W, E * S) - 0.5 * et_w1, right=False)
    mkt_below_a = torch.gather(cum_aq_1w, 1, cnt_below_a).view(N, S)
    rem_flow_at_slot_a = torch.clamp(tot_flow_a - torch.maximum(flow_before_slot_a, mkt_below_a), min=0.0)

    pre_fill_qp_a = torch.clamp(1.0 - ask_qa / (ask_qa0.float() + 10.0), 0.0, 1.0)
    qa_eaten_a = torch.minimum(eff_qa_a, rem_flow_at_slot_a)
    ask_qa = torch.where(hit_a, torch.clamp(ask_qa - qa_eaten_a, min=0.0), ask_qa)
    ask_qa[:, 1::2] = torch.where(act_a[:, 1::2] & even_act_a, torch.clamp(ask_qa[:, 1::2] - qa_eaten_a[:, 0::2], min=0.0), ask_qa[:, 1::2])

    want_a = torch.minimum(eff_qty_a, torch.clamp(rem_flow_at_slot_a - eff_qa_a, min=0.0))
    cum_want_a = torch.cumsum(want_a, dim=1)
    f_qa = torch.minimum(want_a, torch.clamp(pos_qty.unsqueeze(1) - (cum_want_a - want_a), min=0.0))

    has_slot_a = f_qa > 1e-5
    tot_fqa = f_qa.sum(dim=1)
    tot_rev = (f_qa * ask_px).sum(dim=1) * (1.0 - maker_fee)
    has_sell = tot_fqa > 1e-5
    n_fills = torch.clamp(n_fills.long() + has_sell.long(), max=30000).to(torch.int16)
    fill_qp_a_step = torch.where(has_sell, (f_qa * pre_fill_qp_a).sum(dim=1) / torch.clamp(tot_fqa, min=1e-6), torch.zeros_like(tot_fqa))
    sum_fill_qprog = sum_fill_qprog + fill_qp_b_step + fill_qp_a_step

    lifo_f_qa = f_qa[:, :6].sum(dim=1)
    sold_low_q = torch.minimum(lifo_f_qa, low_tranche_qty)
    rem_lifo_qa = torch.clamp(lifo_f_qa - sold_low_q, min=0.0)
    sold_mid_q = torch.minimum(rem_lifo_qa, mid_tranche_qty)
    sold_high_q = torch.clamp(tot_fqa - sold_low_q - sold_mid_q, min=0.0)

    core_qty_pre = torch.clamp(pos_qty - low_tranche_qty - mid_tranche_qty, min=0.0)
    low_mid_cost_pre = (low_tranche_qty * min_buy_px + mid_tranche_qty * mid_buy_px) * (1.0 + maker_fee)
    core_cost_pre = torch.clamp(pos_cost_usd - low_mid_cost_pre, min=0.0)
    core_unit_cost = torch.where(core_qty_pre > 1e-5, core_cost_pre / torch.clamp(core_qty_pre, min=1e-6), avg_entry * (1.0 + maker_fee))
    sold_cost_usd = torch.minimum(
        pos_cost_usd,
        sold_low_q * min_buy_px * (1.0 + maker_fee) + sold_mid_q * mid_buy_px * (1.0 + maker_fee) + sold_high_q * core_unit_cost,
    )

    cash = torch.where(has_sell, cash + tot_rev, cash)
    rem_p = torch.clamp(pos_qty - tot_fqa, min=0.0)
    pos_qty = torch.where(has_sell, rem_p, pos_qty)
    pos_cost_usd = torch.where(has_sell, torch.where(rem_p > 1e-6, torch.clamp(pos_cost_usd - sold_cost_usd, min=0.0), torch.zeros_like(pos_cost_usd)), pos_cost_usd)
    avg_entry = torch.where(has_sell, torch.where(rem_p > 1e-6, pos_cost_usd / (rem_p * (1.0 + maker_fee)), torch.zeros_like(avg_entry)), avg_entry)

    rem_low_q = torch.where(has_sell, torch.where(rem_p > 1e-6, torch.clamp(low_tranche_qty - sold_low_q, min=0.0), torch.zeros_like(low_tranche_qty)), low_tranche_qty)
    rem_mid_q = torch.where(has_sell, torch.where(rem_p > 1e-6, torch.clamp(mid_tranche_qty - sold_mid_q, min=0.0), torch.zeros_like(mid_tranche_qty)), mid_tranche_qty)
    promote_mid = has_sell & (rem_p > 1e-6) & (rem_low_q < 1.95) & (rem_mid_q >= 1.95)
    prom_q = rem_low_q + rem_mid_q
    prom_px = (rem_low_q * min_buy_px + rem_mid_q * mid_buy_px) / torch.clamp(prom_q, min=1e-6)

    low_tranche_qty = torch.where(promote_mid, prom_q, rem_low_q)
    min_buy_px = torch.where(pos_qty > 1e-6, torch.where(promote_mid, prom_px, torch.where(low_tranche_qty > 1e-4, min_buy_px, avg_entry)), torch.zeros_like(min_buy_px))
    mid_tranche_qty = torch.where(promote_mid | (pos_qty <= 1e-6), torch.zeros_like(rem_mid_q), rem_mid_q)
    mid_buy_px = torch.where((pos_qty > 1e-6) & (mid_tranche_qty > 1e-4), mid_buy_px, torch.zeros_like(mid_buy_px))

    rem_ord_a = torch.clamp(ask_qty - f_qa, min=0.0)
    ask_qty = torch.where(has_slot_a, torch.where((rem_ord_a >= 1.0) & (pos_qty.unsqueeze(1) >= 1.0), torch.minimum(rem_ord_a, pos_qty.unsqueeze(1)), torch.zeros_like(rem_ord_a)), ask_qty)

    # 7. Логарифмическая траекторная просадка и самокалибровка критика V_risk:
    mtm_eq = torch.clamp(cash + pos_qty * nb1.squeeze(1) * (1.0 - maker_fee), min=1e-2)
    peak_mtm_eq = torch.maximum(peak_mtm_eq, mtm_eq)
    log_dd_step = torch.clamp(torch.log(peak_mtm_eq / mtm_eq), min=0.0)
    sum_log_dd = sum_log_dd + log_dd_step
    dd_pct = 100.0 * torch.clamp(peak_mtm_eq - mtm_eq, min=0.0) / budget
    max_mtm_dd_pct = torch.maximum(max_mtm_dd_pct, dd_pct)

    realized_dlog = torch.log(mtm_eq / torch.clamp(prev_mtm_eq, min=1e-2))
    td_diff = realized_dlog + 0.985 * val_pred_now - prev_val_pred
    sum_val_err = sum_val_err + td_diff * td_diff
    prev_val_pred = val_pred_now
    prev_mtm_eq = mtm_eq

    n_req_step = n_req_b_step + n_req_a_step
    sum_mut_load = sum_mut_load + n_mut_tot.float()
    n_cancels = torch.clamp(n_cancels.long() + n_cnl_b_step + n_cnl_a_step, max=30000).to(torch.int16)
    n_requeues = torch.clamp(n_requeues.long() + n_req_step, max=30000).to(torch.int16)

    return (
        mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
        min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
        bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
        ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
        api_tokens, peak_mtm_eq, sum_log_dd, max_mtm_dd_pct, sum_mut_load,
        sum_fill_qprog, sum_val_err, prev_val_pred, prev_mtm_eq,
        sum_mode_p, sum_mode_ent,
        n_fills, n_cancels, n_requeues,
    )


@torch.no_grad()
def synthesize_subcent_50lvl_from_cent10(
    bp10_tw: torch.Tensor,
    ap10_tw: torch.Tensor,
    bq10_tw: torch.Tensor,
    aq10_tw: torch.Tensor,
    seed: int = 202628,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Детерминированно интерполирует исторический 10-уровневый стакан `$0.01` `[T, W, 10]`
    в реалистичный 50-уровневый субцентовый стакан `$0.0001` `[T, W, 50]` (по 5 подуровней на цент
    с убывающим профилем глубины и микро-дрожанием внутри тика), чтобы обучать политику L28
    субцентовой механике `penny-jump`, `min_fee_tp` и 50-уровневому Taker-выходу на исторических обвалах!
    """
    dev = bp10_tw.device
    T, W, _ = bp10_tw.shape
    gen = torch.Generator(device=dev).manual_seed(int(seed))

    # 5 подуровней внутри каждого центового шага ($0.0000, -$0.0018, -$0.0039, -$0.0063, -$0.0086):
    sub_offs = torch.tensor([0.0000, 0.0018, 0.0039, 0.0063, 0.0086], dtype=torch.float32, device=dev)
    sub_w = torch.tensor([0.36, 0.24, 0.18, 0.13, 0.09], dtype=torch.float32, device=dev)

    jit_b = torch.randint(-2, 3, (T, W, 10, 1), generator=gen, device=dev, dtype=torch.int64).float() * 0.0001
    jit_a = torch.randint(-2, 3, (T, W, 10, 1), generator=gen, device=dev, dtype=torch.int64).float() * 0.0001
    jit_b[:, :, 0, :] = torch.clamp(jit_b[:, :, 0, :], max=0.0)
    jit_a[:, :, 0, :] = torch.clamp(jit_a[:, :, 0, :], min=0.0)

    bp50 = bp10_tw.unsqueeze(-1) - sub_offs.view(1, 1, 1, 5) + jit_b
    ap50 = ap10_tw.unsqueeze(-1) + sub_offs.view(1, 1, 1, 5) + jit_a
    bp50 = torch.clamp(torch.round(bp50 / 0.0001) * 0.0001, min=0.0001).reshape(T, W, 50)
    ap50 = torch.clamp(torch.round(ap50 / 0.0001) * 0.0001, min=0.0002).reshape(T, W, 50)

    # Гарантируем монотонность убывания Bid и возрастания Ask и отсутствие скрещивания b1 < a1:
    bp50, _ = torch.cummin(bp50, dim=-1)
    ap50 = torch.maximum(ap50, bp50[:, :, :1] + 0.0002)
    ap50, _ = torch.cummax(ap50, dim=-1)

    bq50 = (bq10_tw.unsqueeze(-1) * sub_w.view(1, 1, 1, 5)).reshape(T, W, 50).contiguous()
    aq50 = (aq10_tw.unsqueeze(-1) * sub_w.view(1, 1, 1, 5)).reshape(T, W, 50).contiguous()
    return bp50.contiguous(), ap50.contiguous(), bq50, aq50


@torch.no_grad()
def _run_l28_eval_fused(
    val_env,
    bf16_runner,
    lob_fu_all: torch.Tensor,
    ext138_all: torch.Tensor,
    roll_sell_30s_all: torch.Tensor,
    subcent_books: dict[str, torch.Tensor] | None,
    subcent_local_start: int,
    stacked_weights: tuple[torch.Tensor, ...],
    subcent_windows: list[tuple[str, int, float, float]],
    crisis_windows: list[tuple[str, int, float, float]],
    win_len: int,
    feat_tick: float,
    budget_tiers: torch.Tensor,
    device: torch.device,
    true_latency: bool = True,
) -> dict[str, torch.Tensor]:
    (
        w1_mkt_c, w1_ag_c, b1_c, wr1_c, br1_c, wr2_c, w2m_c, b2m_c,
        wg1_c, bg1_c, wg2_c, bg2_c, wrec_c, brec_c,
        wreg1_c, breg1b_c, breg1a_c, wreg2_c, breg2b_c, breg2a_c,
    ) = stacked_weights
    n_sub = len(subcent_windows)
    n_crs = len(crisis_windows)
    windows = list(subcent_windows) + list(crisis_windows)
    W = len(windows)
    C = w1_mkt_c.shape[0]
    B_bud = budget_tiers.numel()
    E = C * B_bud
    N = W * E
    S = 12
    dt = lob_fu_all.dtype
    echelon_k01 = torch.tensor([0.0, 0.0, 0.12, 0.12, 0.26, 0.26, 0.44, 0.44, 0.68, 0.68, 1.0, 1.0], dtype=torch.float32, device=device)

    et_list = ([0.0001] * n_sub) + ([float(feat_tick)] * n_crs)
    et_1w = torch.tensor(et_list, dtype=torch.float32, device=device)

    w_starts = torch.tensor([int(ws) for _, ws, _, _ in windows], dtype=torch.long, device=device)
    t_steps = torch.arange(win_len, dtype=torch.long, device=device).unsqueeze(1) + w_starts.unsqueeze(0)
    t_steps_cl = torch.clamp(t_steps, max=val_env.max_steps - 1)

    flow_w_all = val_env.flow_market[t_steps_cl[: win_len - 1]].contiguous()
    lob_fu_w_all = lob_fu_all[t_steps_cl[: win_len - 1]].contiguous()
    ext_w_all = ext138_all[t_steps_cl[: win_len - 1]].contiguous()

    z_mkt_raw, adv_risk_w_all = precompute_window_market_z398(flow_w_all, lob_fu_w_all, ext_w_all, bf16_runner)
    z_mkt_norm, _ = normalize_market_z398_per_block(z_mkt_raw)
    z_mkt_c_in = z_mkt_norm.view((win_len - 1) * W, MKT_DIM_L28).unsqueeze(0).expand(C, -1, -1)
    pre_mkt_all = (torch.bmm(z_mkt_c_in, w1_mkt_c) + b1_c).view(C, win_len - 1, W, 384).permute(1, 0, 2, 3).contiguous()

    b1q_cent_w_all = val_env.bids_q[t_steps_cl][:, :, 0].contiguous()
    a1q_cent_w_all = val_env.asks_q[t_steps_cl][:, :, 0].contiguous()

    if n_sub > 0 and subcent_books is not None:
        sub_t = t_steps_cl[:, :n_sub]
        sub_idx = torch.clamp(sub_t - int(subcent_local_start), min=0, max=subcent_books["bid_p50"].shape[0] - 1)
        b_sub = subcent_books["bid_p50"][sub_idx]
        a_sub = subcent_books["ask_p50"][sub_idx]
        bq_sub = subcent_books["bid_q50"][sub_idx]
        aq_sub = subcent_books["ask_q50"][sub_idx]
        if n_crs > 0:
            crs_t = t_steps_cl[:, n_sub:]
            L_tgt = b_sub.shape[-1]
            pad_r = L_tgt - val_env.bids.shape[-1]
            b_crs = F.pad(val_env.bids[crs_t], (0, pad_r), value=0.0)
            a_crs = F.pad(val_env.asks[crs_t], (0, pad_r), value=0.0)
            bq_crs = F.pad(val_env.bids_q[crs_t], (0, pad_r), value=0.0)
            aq_crs = F.pad(val_env.asks_q[crs_t], (0, pad_r), value=0.0)
            bL_w_all = torch.cat([b_sub, b_crs], dim=1).contiguous()
            aL_w_all = torch.cat([a_sub, a_crs], dim=1).contiguous()
            bqL_w_all = torch.cat([bq_sub, bq_crs], dim=1).contiguous()
            aqL_w_all = torch.cat([aq_sub, aq_crs], dim=1).contiguous()
        else:
            bL_w_all = b_sub.contiguous()
            aL_w_all = a_sub.contiguous()
            bqL_w_all = bq_sub.contiguous()
            aqL_w_all = aq_sub.contiguous()
    else:
        bL_w_all = val_env.bids[t_steps_cl].contiguous()
        aL_w_all = val_env.asks[t_steps_cl].contiguous()
        bqL_w_all = val_env.bids_q[t_steps_cl].contiguous()
        aqL_w_all = val_env.asks_q[t_steps_cl].contiguous()

    b1_w_all = bL_w_all[:, :, 0].contiguous()
    a1_w_all = aL_w_all[:, :, 0].contiguous()
    rb_w_all = val_env.roll_buy_30s[t_steps_cl].contiguous()
    rs_w_all = roll_sell_30s_all[t_steps_cl].contiguous()
    sv_w_all = val_env.sell_vol[t_steps_cl].contiguous()
    bv_w_all = val_env.buy_vol[t_steps_cl].contiguous()

    env_budgets_e = budget_tiers.unsqueeze(0).expand(C, B_bud).reshape(E)
    budget = env_budgets_e.unsqueeze(0).expand(W, E).reshape(N).contiguous()

    mem_h = torch.zeros((N, 16), dtype=dt, device=device)
    cash = budget.clone()
    pos_qty = torch.zeros(N, dtype=torch.float32, device=device)
    pos_cost_usd = torch.zeros(N, dtype=torch.float32, device=device)
    avg_entry = torch.zeros(N, dtype=torch.float32, device=device)
    min_buy_px = torch.zeros(N, dtype=torch.float32, device=device)
    low_tranche_qty = torch.zeros(N, dtype=torch.float32, device=device)
    mid_buy_px = torch.zeros(N, dtype=torch.float32, device=device)
    mid_tranche_qty = torch.zeros(N, dtype=torch.float32, device=device)
    entry_s = torch.zeros(N, dtype=torch.int16, device=device)

    rho_w = torch.tensor([float(r0) for _, _, r0, _ in windows], dtype=torch.float32, device=device)
    roff_w = torch.tensor([float(ro) for _, _, _, ro in windows], dtype=torch.float32, device=device)
    has_inv_w = rho_w > 0.01
    if bool(has_inv_w.any()):
        b1_0_w = torch.maximum(et_1w, b1_w_all[0])
        epx_w = torch.maximum(et_1w, torch.round((b1_0_w * (1.0 + roff_w)) / et_1w) * et_1w)
        bud_we = budget.view(W, E)
        iqty_we = torch.floor((bud_we * rho_w.unsqueeze(1) / b1_0_w.unsqueeze(1)) * 100.0) / 100.0
        vm_we = has_inv_w.unsqueeze(1) & (iqty_we >= 2.0)
        rcash_we = torch.clamp(bud_we - iqty_we * b1_0_w.unsqueeze(1) * (1.0 - val_env.maker_fee), min=0.0)
        epx_we = epx_w.unsqueeze(1).expand(W, E)

        cash = torch.where(vm_we, rcash_we, bud_we).reshape(N)
        pos_qty = torch.where(vm_we, iqty_we, torch.zeros_like(iqty_we)).reshape(N)
        avg_entry = torch.where(vm_we, epx_we, torch.zeros_like(epx_we)).reshape(N)
        pos_cost_usd = torch.where(vm_we, iqty_we * epx_we * (1.0 + val_env.maker_fee), torch.zeros_like(iqty_we)).reshape(N)
        min_buy_px = avg_entry.clone()
        low_tranche_qty = (pos_qty * 0.5).clone()
        mid_buy_px = avg_entry.clone()
        mid_tranche_qty = (pos_qty * 0.5).clone()
        entry_s = torch.where(has_inv_w.unsqueeze(1).expand(W, E).reshape(N), torch.full_like(entry_s, -120), entry_s)

    bid_px = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qty = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qa = torch.zeros((N, S), dtype=torch.float32, device=device)
    bid_qa0 = torch.zeros((N, S), dtype=dt, device=device)
    bid_act_s = torch.zeros((N, S), dtype=torch.int16, device=device)
    ask_px = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qty = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qa = torch.zeros((N, S), dtype=torch.float32, device=device)
    ask_qa0 = torch.zeros((N, S), dtype=dt, device=device)
    ask_act_s = torch.zeros((N, S), dtype=torch.int16, device=device)
    api_tokens = torch.full((N,), TOKEN_BUCKET_CAP_L28, dtype=torch.float32, device=device)

    init_b1 = torch.maximum(b1_w_all[0], et_1w).unsqueeze(1).expand(W, E).reshape(N)
    start_mtm_eq = torch.clamp(cash + pos_qty * init_b1 * (1.0 - val_env.maker_fee), min=1.0)
    peak_mtm_eq = start_mtm_eq.clone()
    prev_mtm_eq = start_mtm_eq.clone()
    prev_val_pred = torch.zeros(N, dtype=torch.float32, device=device)
    sum_log_dd = torch.zeros(N, dtype=torch.float32, device=device)
    max_dd_pct = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mut = torch.zeros(N, dtype=torch.float32, device=device)
    sum_fqp = torch.zeros(N, dtype=torch.float32, device=device)
    sum_val_err = torch.zeros(N, dtype=torch.float32, device=device)
    sum_mode_p = torch.zeros((N, K_MODES_L28), dtype=torch.float32, device=device)
    sum_mode_ent = torch.zeros(N, dtype=torch.float32, device=device)
    n_fills = torch.zeros(N, dtype=torch.int16, device=device)
    n_cancels = torch.zeros(N, dtype=torch.int16, device=device)
    n_requeues = torch.zeros(N, dtype=torch.int16, device=device)

    qm_eval = torch.tensor(1.0, dtype=torch.float32, device=device)
    el_eval = torch.tensor(0, dtype=torch.int16, device=device)
    s_steps_i16 = torch.arange(win_len, dtype=torch.int16, device=device)
    for s in range(win_len - 1):
        (
            mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
            min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
            bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
            ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
            api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
            sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
            sum_mode_p, sum_mode_ent,
            n_fills, n_cancels, n_requeues,
        ) = fused_l28_hierarchical_subcent_tick(
            s_steps_i16[s], s_steps_i16[s + 1], pre_mkt_all[s], adv_risk_w_all[s],
            b1_w_all[s], a1_w_all[s], b1q_cent_w_all[s], a1q_cent_w_all[s],
            rb_w_all[s], rs_w_all[s],
            bL_w_all[s], aL_w_all[s], bqL_w_all[s], aqL_w_all[s],
            b1_w_all[s + 1], a1_w_all[s + 1], sv_w_all[s + 1], bv_w_all[s + 1],
            w1_ag_c, wr1_c, br1_c, wr2_c, w2m_c, b2m_c, wg1_c, bg1_c, wg2_c, bg2_c, wrec_c, brec_c,
            wreg1_c, breg1b_c, breg1a_c, wreg2_c, breg2b_c, breg2a_c,
            echelon_k01, budget,
            mem_h, cash, pos_qty, pos_cost_usd, avg_entry,
            min_buy_px, low_tranche_qty, mid_buy_px, mid_tranche_qty, entry_s,
            bid_px, bid_qty, bid_qa, bid_qa0, bid_act_s,
            ask_px, ask_qty, ask_qa, ask_qa0, ask_act_s,
            api_tokens, peak_mtm_eq, sum_log_dd, max_dd_pct, sum_mut,
            sum_fqp, sum_val_err, prev_val_pred, prev_mtm_eq,
            sum_mode_p, sum_mode_ent,
            n_fills, n_cancels, n_requeues,
            et_1w, feat_tick, val_env.maker_fee, val_env.taker_fee,
            qm_eval, el_eval, true_latency,
        )

    end_b1 = torch.maximum(b1_w_all[win_len - 1], et_1w).unsqueeze(1).expand(W, E).reshape(N)
    rem = pos_qty > 1e-6
    end_cash = torch.clamp(torch.where(rem, cash + pos_qty * end_b1 * (1.0 - val_env.maker_fee), cash), min=1.0)
    end_dd_pct = 100.0 * torch.clamp(peak_mtm_eq - end_cash, min=0.0) / budget
    max_dd_pct = torch.maximum(max_dd_pct, end_dd_pct)

    n_steps_f = float(win_len - 1)
    dlog_eq = torch.log(end_cash / start_mtm_eq).view(W, E)
    mean_log_dd = (sum_log_dd / n_steps_f).view(W, E)
    return {
        "traj_growth_bp": 10000.0 * (dlog_eq - 0.55 * mean_log_dd),
        "mtm_delta_pct": (100.0 * (end_cash - start_mtm_eq) / budget).view(W, E),
        "max_dd_pct": max_dd_pct.view(W, E),
        "fills": n_fills.float().view(W, E),
        "cancels": n_cancels.float().view(W, E),
        "requeues": n_requeues.float().view(W, E),
        "fqp": sum_fqp.view(W, E),
        "mode_p": (sum_mode_p / n_steps_f).view(W, E, K_MODES_L28),
        "mode_ent": (sum_mode_ent / n_steps_f).view(W, E),
    }


@torch.no_grad()
def evaluate_l28_dual_basket(
    val_env,
    bf16_runner,
    lob_fu_all: torch.Tensor,
    ext138_all: torch.Tensor,
    roll_sell_30s_all: torch.Tensor,
    subcent_books: dict[str, torch.Tensor] | None,
    subcent_local_start: int,
    cand_states: list[dict],
    subcent_windows: list[tuple[str, int, float, float]],
    crisis_windows: list[tuple[str, int, float, float]],
    win_len: int,
    feat_tick: float,
    device: torch.device,
    budget_tiers: torch.Tensor | None = None,
    true_latency: bool = True,
) -> list[dict]:
    """
    Двухрежимная честная оценка кандидатов Урока 28 за ОДИН монолитный проход одновременно на:
      1) Реальных субцентовых окнах `$0.0001` с 50-уровневым стаканом (`subcent_windows`),
      2) Исторических макро-кризисах/сквизах `$0.01` с 10-уровневым стаканом (`crisis_windows`, zero-padded до L=50).
    Гарантирует отсутствие переобучения под спокойный боковик и сохранение кризисной брони!
    """
    dt = lob_fu_all.dtype
    if budget_tiers is None:
        budget_tiers = torch.tensor([300.0, 625.0, 835.0, 1250.0, 2500.0, 6000.0], dtype=torch.float32, device=device)
    B_bud = budget_tiers.numel()
    C = len(cand_states)

    mkt_idx_t = torch.tensor(MKT_IDX_L28, dtype=torch.long, device=device)
    ag_idx_t = torch.tensor(AG_IDX_L28, dtype=torch.long, device=device)

    w1_l, b1_l, wr1_l, br1_l, wr2_l, w2m_l, b2m_l = [], [], [], [], [], [], []
    wg1_l, bg1_l, wg2_l, bg2_l, wrec_l, brec_l = [], [], [], [], [], []
    wreg1_l, breg1b_l, breg1a_l, wreg2_l, breg2b_l, breg2a_l = [], [], [], [], [], []
    for st in cand_states:
        w1_l.append(st["w1"].to(device=device, dtype=dt))
        b1_l.append(st["b1"].to(device=device, dtype=dt).view(1, 384))
        wr1_l.append(st["w_res1"].to(device=device, dtype=dt))
        br1_l.append(st["b_res1"].to(device=device, dtype=dt).view(1, 128))
        wr2_l.append(st["w_res2"].to(device=device, dtype=dt))
        w2m_l.append(st["w2_moe"].to(device=device, dtype=dt))
        b2m_l.append(st["b2_moe"].to(device=device, dtype=dt).view(1, MOE_OUT_DIM_L28))
        wg1_l.append(st["w_gate1"].to(device=device, dtype=dt))
        bg1_l.append(st["b_gate1"].to(device=device, dtype=dt).view(1, ROUTER_H_DIM_L28))
        wg2_l.append(st["w_gate2"].to(device=device, dtype=dt))
        bg2_l.append(st["b_gate2"].to(device=device, dtype=dt).view(1, K_MODES_L28))
        wrec_l.append(st["w_rec"].to(device=device, dtype=dt))
        brec_l.append(st["b_rec"].to(device=device, dtype=dt).view(1, 32))
        wreg1_l.append(st["w_reg1"].to(device=device, dtype=dt))
        breg1b_l.append(st["b_reg1_b"].to(device=device, dtype=dt).view(1, 48))
        breg1a_l.append(st["b_reg1_a"].to(device=device, dtype=dt).view(1, 48))
        wreg2_l.append(st["w_reg2"].to(device=device, dtype=dt))
        breg2b_l.append(st["b_reg2_b"].to(device=device, dtype=dt).view(1, 2))
        breg2a_l.append(st["b_reg2_a"].to(device=device, dtype=dt).view(1, 2))

    w1_c = torch.stack(w1_l, dim=0).contiguous()
    stacked_w = (
        w1_c.index_select(1, mkt_idx_t).contiguous(),
        w1_c.index_select(1, ag_idx_t).contiguous(),
        torch.stack(b1_l, dim=0).contiguous(),
        torch.stack(wr1_l, dim=0).contiguous(),
        torch.stack(br1_l, dim=0).contiguous(),
        torch.stack(wr2_l, dim=0).contiguous(),
        torch.stack(w2m_l, dim=0).contiguous(),
        torch.stack(b2m_l, dim=0).contiguous(),
        torch.stack(wg1_l, dim=0).contiguous(),
        torch.stack(bg1_l, dim=0).contiguous(),
        torch.stack(wg2_l, dim=0).contiguous(),
        torch.stack(bg2_l, dim=0).contiguous(),
        torch.stack(wrec_l, dim=0).contiguous(),
        torch.stack(brec_l, dim=0).contiguous(),
        torch.stack(wreg1_l, dim=0).contiguous(),
        torch.stack(breg1b_l, dim=0).contiguous(),
        torch.stack(breg1a_l, dim=0).contiguous(),
        torch.stack(wreg2_l, dim=0).contiguous(),
        torch.stack(breg2b_l, dim=0).contiguous(),
        torch.stack(breg2a_l, dim=0).contiguous(),
    )

    n_sub_w = len(subcent_windows)
    all_wins = list(subcent_windows) + list(crisis_windows)
    fused_out = _run_l28_eval_fused(
        val_env, bf16_runner, lob_fu_all, ext138_all, roll_sell_30s_all,
        subcent_books, subcent_local_start, stacked_w, subcent_windows, crisis_windows,
        win_len, feat_tick, budget_tiers, device, true_latency,
    )

    traj_growth_bp = fused_out["traj_growth_bp"]
    mtm_delta_pct = fused_out["mtm_delta_pct"]
    w_dd_mat = fused_out["max_dd_pct"]
    w_fills_mat = fused_out["fills"]
    w_cancels_mat = fused_out["cancels"]
    w_requeues_mat = fused_out["requeues"]
    w_fqp_mat = fused_out["fqp"]
    mode_p_mat = fused_out["mode_p"]
    mode_ent_mat = fused_out["mode_ent"]

    clean_idx = [wi for wi, (_, _, r0, _) in enumerate(all_wins) if r0 <= 0.01]
    inv_idx = [wi for wi, (_, _, r0, _) in enumerate(all_wins) if r0 > 0.01]
    sub_idx_list = list(range(0, n_sub_w))
    crs_idx_list = list(range(n_sub_w, len(all_wins)))

    results = []
    beta_risk = 0.06
    ln2 = math.log(2.0)
    for ci in range(C):
        s_i, e_i = ci * B_bud, (ci + 1) * B_bud
        g_vec = traj_growth_bp[:, s_i:e_i].reshape(-1)
        ent_bp = float((-torch.logsumexp(-beta_risk * g_vec, dim=0) + torch.log(torch.tensor(float(g_vec.numel()), device=device))).item() / beta_risk)
        q_vals = torch.quantile(g_vec, torch.tensor([0.15, 0.50, 0.85], device=device))

        k_cvar = max(1, g_vec.numel() // 4)
        cvar25_bp = float(torch.topk(g_vec, k_cvar, largest=False).values.mean().item())
        mean_g_bp = float(g_vec.mean().item())
        downside_semi = float(torch.sqrt(torch.mean(torch.clamp(-g_vec, min=0.0) ** 2)).item())

        cl_ev = float(mtm_delta_pct[clean_idx, s_i:e_i].mean().item()) if clean_idx else 0.0
        inv_ev = float(mtm_delta_pct[inv_idx, s_i:e_i].mean().item()) if inv_idx else 0.0
        sub_ev = float(mtm_delta_pct[sub_idx_list, s_i:e_i].mean().item()) if sub_idx_list else cl_ev
        crs_ev = float(mtm_delta_pct[crs_idx_list, s_i:e_i].mean().item()) if crs_idx_list else inv_ev
        full_ev = float(mtm_delta_pct[:, s_i:e_i].mean().item())
        dd_p = float(w_dd_mat[:, s_i:e_i].max(dim=0).values.mean().item())
        fills = float(w_fills_mat[:, s_i:e_i].sum(dim=0).mean().item())
        cancels = float(w_cancels_mat[:, s_i:e_i].sum(dim=0).mean().item())
        requeues = float(w_requeues_mat[:, s_i:e_i].sum(dim=0).mean().item())
        fqp_s = float(w_fqp_mat[:, s_i:e_i].sum(dim=0).mean().item())
        cnl_pf = cancels / max(fills, 0.5)

        # Мягкий штраф за избыточный churn отмен (Cnl/F > 6.0) + премия за реальный SubCent EV и Crisis защиту:
        churn_pen_bp = max(0.0, cnl_pf - 6.0) * 0.55
        geo_score_bp = (
            0.50 * mean_g_bp
            + 0.25 * cvar25_bp
            + 0.20 * float(q_vals[1].item())
            + 30.0 * cl_ev
            + 18.0 * inv_ev
            + 28.0 * sub_ev
            + 14.0 * crs_ev
            - 0.18 * downside_semi
            - churn_pen_bp
        )

        mp_bar = mode_p_mat[:, s_i:e_i, :].reshape(-1, K_MODES_L28).mean(dim=0)
        h_marginal = float(-(mp_bar * torch.log(torch.clamp(mp_bar, min=1e-6))).sum().item())
        h_cond = float(mode_ent_mat[:, s_i:e_i].mean().item())
        mi_bits = max(0.0, (h_marginal - h_cond) / ln2)

        results.append({
            "geo_score_bp": float(geo_score_bp),
            "entropic_growth_bp": ent_bp,
            "cvar25_bp": cvar25_bp,
            "q15_bp": float(q_vals[0].item()),
            "q50_bp": float(q_vals[1].item()),
            "q85_bp": float(q_vals[2].item()),
            "full_ev_pct": full_ev,
            "clean_ev_pct": cl_ev,
            "inv_delta_ev_pct": inv_ev,
            "subcent_ev_pct": sub_ev,
            "crisis_ev_pct": crs_ev,
            "dd_pct": dd_p,
            "fills": fills,
            "cancels": cancels,
            "requeues": requeues,
            "cnl_per_fill": cnl_pf,
            "eta_queue_pct": 100.0 * min(1.0, fqp_s / max(fills, 1.0)) if fills >= 0.5 else 100.0,
            "mode_probs": [round(float(x), 4) for x in mp_bar.tolist()],
            "mode_mi_bits": round(mi_bits, 4),
        })
    return results
