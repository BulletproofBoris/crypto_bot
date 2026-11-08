#!/usr/bin/env python3
"""
_tools/vectorized_free_multi_order_env.py
=============================================================================
VectorizedFreeMultiOrderEnv (Урок 6 v2 — 24 Независимых Слота [12 Bid + 12 Ask],
Полная Свобода Объема Заявок + Безветвевая GPU-Векторизация + SelfImpactRealismNet v2):

Ключевые особенности:
1. 24 ОДНОВРЕМЕННЫХ НЕЗАВИСИМЫХ ОРДЕРА В КНИГЕ (4x расширение: 12 Bid + 12 Ask):
   - 6 ценовых эшелонов × 2 независимых транша (айсберга) на каждом эшелоне:
     * Слоты 0, 1 : Penny-Jump (`b1 + tick` / `a1 - tick` при широком спреде, иначе L1)
     * Слоты 2, 3 : Best L1 (`b1` / `a1`)
     * Слоты 4, 5 : Passive L2 (`b1 - 1 tick` / `a1 + 1 tick`)
     * Слоты 6, 7 : Deep L3 (`b1 - 2 ticks` / `a1 + 2 ticks`)
     * Слоты 8, 9 : Deep L4 (`b1 - 3 ticks` / `a1 + 3 ticks`)
     * Слоты 10,11: Knife/Pump L5 (`b1 - 5 ticks` / `a1 + 5 ticks`)
2. 100% БЕЗВЕТВЕВАЯ МАТРИЧНАЯ ВЕКТОРИЗАЦИЯ (ZERO CPU-GPU SYNC):
   - Полностью устранены вызовы `.any()`, `bool()` и `torch.nonzero()` внутри шага `step()`.
   - Каскадный налив по всем 12 слотам Bid и 12 слотам Ask вычисляется точно через
     матричный водопад `torch.cumsum(..., dim=1)` за один проход ядра CUDA без потери точности.
=============================================================================
"""

import torch
from typing import Tuple


class VectorizedFreeMultiOrderEnv:
    def __init__(
        self,
        bids: torch.Tensor,
        asks: torch.Tensor,
        bids_q: torch.Tensor,
        asks_q: torch.Tensor,
        buy_vol: torch.Tensor,
        sell_vol: torch.Tensor,
        lob_tensors: torch.Tensor,
        flow_market: torch.Tensor,
        targets: torch.Tensor,
        num_envs: int = 8192,
        n_side_slots: int = 12,
        tick_size: float = 0.01,
        maker_fee: float = 0.0010,
        taker_fee: float = 0.0020,
        init_budget: float = 1000.0,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.num_envs = num_envs
        self.n_side_slots = n_side_slots  # 12 слотов Bid + 12 слотов Ask = 24 слота одновременно
        self.out_dim = 2 * n_side_slots + 2  # 12 Bid + 12 Ask + Cancel Bid gate + Panic Sweep gate = 26
        self.tick_size = float(tick_size)
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)

        self.bids = bids.to(device)
        self.asks = asks.to(device)
        self.bids_q = bids_q.to(device)
        self.asks_q = asks_q.to(device)
        self.buy_vol = buy_vol.to(device)
        self.sell_vol = sell_vol.to(device)
        self.lob_tensors = lob_tensors.to(device)
        self.flow_market = flow_market.to(device)
        self.targets = targets.to(device)
        # CPU-копия флагов кровотечения/обвала, чтобы не вызывать CPU-GPU синхронизацию на каждом тике
        t_cpu = targets.detach().cpu()
        self.is_bleed_cpu = ((t_cpu[:, 1] > 0.5) | (t_cpu[:, 2] > 0.5)).numpy()
        self.max_steps = len(self.bids)

        # Скользящий 30-секундный поток для оценки емкости
        bv_cpu = buy_vol.cpu().float()
        sv_cpu = sell_vol.cpu().float()
        cs_b = torch.cat([torch.zeros(1), torch.cumsum(bv_cpu, dim=0)])
        cs_s = torch.cat([torch.zeros(1), torch.cumsum(sv_cpu, dim=0)])
        idx = torch.arange(self.max_steps)
        self.roll_buy_30s = torch.clamp(cs_b[idx + 1] - cs_b[torch.clamp(idx - 29, min=0)], min=10.0).to(device)
        self.roll_sell_30s = torch.clamp(cs_s[idx + 1] - cs_s[torch.clamp(idx - 29, min=0)], min=10.0).to(device)

        self.budget = torch.full((num_envs,), float(init_budget), dtype=torch.float32, device=device)
        self.cash = torch.full((num_envs,), float(init_budget), dtype=torch.float32, device=device)
        self.pos_qty = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.pos_cost_usd = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.avg_entry = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.min_buy_px = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.low_tranche_qty = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.entry_t = torch.zeros(num_envs, dtype=torch.long, device=device)

        # 12 независимых слотов Bid [B, 12] и 12 независимых слотов Ask [B, 12]
        self.bid_px = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.bid_qty = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.bid_qa = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.bid_act_t = torch.zeros((num_envs, n_side_slots), dtype=torch.long, device=device)

        self.ask_px = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.ask_qty = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.ask_qa = torch.zeros((num_envs, n_side_slots), dtype=torch.float32, device=device)
        self.ask_act_t = torch.zeros((num_envs, n_side_slots), dtype=torch.long, device=device)

        # Предвычисленные смещения в тиках и индексы уровней стакана для 12 слотов (6 эшелонов по 2 транша)
        # Эшелоны: [0: Penny (+1/-1), 1: L1 (0), 2: L2 (-1/+1), 3: L3 (-2/+2), 4: L4 (-3/+3), 5: L5 (-5/+5)]
        self.slot_tick_offsets = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0],
            dtype=torch.float32,
            device=device,
        ) * self.tick_size
        self.slot_lob_lvl = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.long, device=device)
        self.is_penny_slot = torch.tensor(
            [True, True, False, False, False, False, False, False, False, False, False, False],
            dtype=torch.bool,
            device=device,
        )

        # Метрики
        self.n_trades = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.partial_fills = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.multi_orders_used = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.frontrun_hits = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.sell_cycles = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.prof_sell_cycles = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset_with_budgets(self, budgets_t: torch.Tensor):
        self.budget.copy_(budgets_t)
        self.cash.copy_(budgets_t)
        self.pos_qty.zero_()
        self.pos_cost_usd.zero_()
        self.avg_entry.zero_()
        self.min_buy_px.zero_()
        self.low_tranche_qty.zero_()
        self.entry_t.zero_()
        self.bid_px.zero_()
        self.bid_qty.zero_()
        self.bid_qa.zero_()
        self.bid_act_t.zero_()
        self.ask_px.zero_()
        self.ask_qty.zero_()
        self.ask_qa.zero_()
        self.ask_act_t.zero_()

    def get_observations(self, cur_t_scalar: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает LOB [1, 10, 4], Flow [B, 16], Targets [1, 4], MultiSlotSignals [B, 6]."""
        lob = self.lob_tensors[cur_t_scalar:cur_t_scalar + 1]
        flow = self.flow_market[cur_t_scalar:cur_t_scalar + 1].expand(self.num_envs, -1).clone()
        tgt = self.targets[cur_t_scalar:cur_t_scalar + 1]

        b1 = torch.clamp(self.bids[cur_t_scalar, 0], min=self.tick_size)
        a1 = torch.clamp(self.asks[cur_t_scalar, 0], min=b1 + self.tick_size)
        l1_bid_usd = torch.clamp(b1 * self.bids_q[cur_t_scalar, 0], min=100.0)
        l1_ask_usd = torch.clamp(a1 * self.asks_q[cur_t_scalar, 0], min=100.0)

        has_pos = (self.pos_qty > 1e-6).float()
        curr_pos_usd = self.pos_qty * b1
        pos_ratio = torch.clamp(curr_pos_usd / torch.clamp(self.budget, min=25.0), 0.0, 1.0)
        free_cash_ratio = torch.clamp(self.cash / torch.clamp(self.budget, min=25.0), 0.0, 1.0)

        unr_ticks = torch.where(has_pos > 0.5, (b1 - self.avg_entry) / self.tick_size, torch.zeros_like(has_pos))
        hold_ratio = torch.where(
            has_pos > 0.5,
            torch.clamp((cur_t_scalar - self.entry_t).float() / 300.0, 0.0, 2.0),
            torch.zeros_like(has_pos),
        )

        tot_bid_q = self.bid_qty.sum(dim=1)
        tot_ask_q = self.ask_qty.sum(dim=1)
        has_bid = (tot_bid_q > 1e-6).float()
        has_ask = (tot_ask_q > 1e-6).float()
        n_bid_slots = (self.bid_qty > 1e-6).float().sum(dim=1) / float(self.n_side_slots)
        n_ask_slots = (self.ask_qty > 1e-6).float().sum(dim=1) / float(self.n_side_slots)

        flow[:, 8] = has_pos
        flow[:, 9] = pos_ratio
        flow[:, 10] = free_cash_ratio
        flow[:, 11] = torch.clamp(unr_ticks / 5.0, -2.0, 2.0)
        flow[:, 12] = hold_ratio
        flow[:, 13] = has_bid
        flow[:, 14] = has_ask
        flow[:, 15] = n_bid_slots - n_ask_slots

        bud_l1 = torch.log1p(self.budget / l1_bid_usd) / 3.0
        cash_l1 = torch.log1p(torch.clamp(self.cash, min=0.0) / l1_bid_usd) / 3.0
        pos_l1_ask = torch.clamp((self.pos_qty * a1) / l1_ask_usd, 0.0, 4.0) / 2.0
        pos_tape = torch.clamp(self.pos_qty / self.roll_buy_30s[cur_t_scalar], 0.0, 10.0) / 5.0
        multi_6d = torch.stack([bud_l1, cash_l1, pos_l1_ask, pos_tape, n_bid_slots, n_ask_slots], dim=-1)

        return lob, flow, tgt, multi_6d

    def step(self, raw_outputs: torch.Tensor, cur_t_scalar: int):
        """
        100% Безветвевой GPU-шаг для 24 слотов (12 Bid + 12 Ask):
        raw_outputs: [B, 26]:
          0..11 : веса свободного кэша для 12 Bid-слотов
          12..23: веса свободной позиции для 12 Ask-слотов
          24    : cancel_all_bids_gate (> 0.65)
          25    : panic_sweep_gate (> 0.82)
        """
        S = self.n_side_slots
        b1 = torch.clamp(self.bids[cur_t_scalar, 0], min=self.tick_size)
        a1 = torch.clamp(self.asks[cur_t_scalar, 0], min=b1 + self.tick_size)
        l1_bid_q = torch.clamp(self.bids_q[cur_t_scalar, 0], min=10.0)
        l1_ask_q = torch.clamp(self.asks_q[cur_t_scalar, 0], min=10.0)
        wide_sp = (a1 - b1) >= 2.0 * self.tick_size

        # 1. Экстренный рыночный выход по L1..L10 (без синхронизации .any())
        is_panic = (raw_outputs[:, 2 * S + 1] > 0.82) & (self.pos_qty > 1e-6)
        b_p10 = torch.clamp(self.bids[cur_t_scalar, :10], min=self.tick_size)  # [10]
        b_q10 = torch.clamp(self.bids_q[cur_t_scalar, :10], min=1.0).clone()  # [10]
        b_q10[9] = 1e9  # 10-й уровень поглощает остаток
        cum_q10 = torch.cumsum(b_q10, dim=0)  # [10]
        prev_q10 = cum_q10 - b_q10  # [10]
        take_10 = torch.clamp(
            torch.minimum(self.pos_qty.unsqueeze(1) - prev_q10.unsqueeze(0), b_q10.unsqueeze(0)),
            min=0.0,
        )  # [B, 10]
        proceeds = (take_10 * b_p10.unsqueeze(0)).sum(dim=1) * (1.0 - self.taker_fee)

        self.cash = torch.where(is_panic, self.cash + proceeds, self.cash)
        self.pos_qty = torch.where(is_panic, torch.zeros_like(self.pos_qty), self.pos_qty)
        self.pos_cost_usd = torch.where(is_panic, torch.zeros_like(self.pos_cost_usd), self.pos_cost_usd)
        self.avg_entry = torch.where(is_panic, torch.zeros_like(self.avg_entry), self.avg_entry)
        self.min_buy_px = torch.where(is_panic, torch.zeros_like(self.min_buy_px), self.min_buy_px)
        self.low_tranche_qty = torch.where(is_panic, torch.zeros_like(self.low_tranche_qty), self.low_tranche_qty)
        self.ask_qty = torch.where(is_panic.unsqueeze(1), torch.zeros_like(self.ask_qty), self.ask_qty)
        self.n_trades = self.n_trades + is_panic.long()

        # 2. Управление 12 независимыми персистентными слотами Bid (без .any())
        cancel_bids = raw_outputs[:, 2 * S] > 0.78
        cb_mask = cancel_bids.unsqueeze(1)
        self.bid_qty = torch.where(cb_mask, torch.zeros_like(self.bid_qty), self.bid_qty)
        self.bid_px = torch.where(cb_mask, torch.zeros_like(self.bid_px), self.bid_px)
        self.bid_qa = torch.where(cb_mask, torch.zeros_like(self.bid_qa), self.bid_qa)

        penny_add = torch.where(wide_sp, torch.tensor(self.tick_size, device=self.device), torch.tensor(0.0, device=self.device))
        slot_b_px_1d = torch.where(
            self.is_penny_slot,
            b1 + penny_add,
            torch.clamp(b1 - self.slot_tick_offsets, min=self.tick_size),
        )  # [12]
        t_bid_px = slot_b_px_1d.unsqueeze(0).expand(self.num_envs, S)

        # Авто-подтяжка отставших слотов: если рынок ушел вверх на >= 2 тика от цены слота и слот висит >= 15с,
        # освобождаем слот для перестановки на актуальную цену (но сохраняем очередь, если цена в пределах 1 тика!)
        stale_b = (
            (self.bid_qty > 1e-6)
            & ((cur_t_scalar - self.bid_act_t) >= 15)
            & (torch.abs(t_bid_px - self.bid_px) >= 1.9 * self.tick_size)
        )
        self.bid_qty = torch.where(stale_b, torch.zeros_like(self.bid_qty), self.bid_qty)
        self.bid_px = torch.where(stale_b, torch.zeros_like(self.bid_px), self.bid_px)
        self.bid_qa = torch.where(stale_b, torch.zeros_like(self.bid_qa), self.bid_qa)

        empty_b = self.bid_qty <= 1e-6
        b_w = torch.where(cb_mask, torch.zeros_like(raw_outputs[:, :S]), raw_outputs[:, :S])
        place_b = empty_b & (b_w >= 0.12)
        b_w_active = torch.where(place_b, b_w, torch.zeros_like(b_w))
        b_sum = b_w_active.sum(dim=1, keepdim=True)
        scale_b = torch.where(
            b_sum > 1.0,
            1.0 / torch.clamp(b_sum, min=1e-6),
            torch.where(b_sum >= 0.75, 1.0 / torch.clamp(b_sum, min=1e-6), torch.ones_like(b_sum)),
        )
        b_alloc = b_w_active * scale_b  # [B, 12]

        lvl_b_q = torch.clamp(self.bids_q[cur_t_scalar, self.slot_lob_lvl], min=10.0)  # [12]
        base_b_qa_1d = torch.where(self.is_penny_slot & wide_sp, torch.zeros_like(lvl_b_q), lvl_b_q)
        base_b_qa = base_b_qa_1d.unsqueeze(0).expand(self.num_envs, S)

        reserved_cash = (self.bid_qty * self.bid_px * (1.0 + self.maker_fee)).sum(dim=1)
        unreserved_cash = torch.clamp(self.cash * 0.996 - reserved_cash, min=0.0).unsqueeze(1)
        t_bid_usd = unreserved_cash * b_alloc
        valid_b = place_b & (t_bid_usd >= 2.0)
        t_bid_qty = torch.where(valid_b, t_bid_usd / t_bid_px, torch.zeros_like(t_bid_usd))

        b_over65 = torch.sigmoid((t_bid_usd - 65.0) / 25.0)
        b_self_impact_qa = base_b_qa + b_over65 * (0.065 * l1_bid_q + 0.15 * t_bid_qty)

        self.bid_px = torch.where(valid_b, t_bid_px, self.bid_px)
        self.bid_qty = torch.where(valid_b, t_bid_qty, self.bid_qty)
        self.bid_qa = torch.where(valid_b, b_self_impact_qa, self.bid_qa)
        self.bid_act_t = torch.where(valid_b, torch.full_like(self.bid_act_t, cur_t_scalar + 1), self.bid_act_t)

        # 3. Управление 12 независимыми персистентными слотами Ask (без .any())
        has_p = (self.pos_qty > 1e-6) & (~is_panic)
        hp_mask = has_p.unsqueeze(1)
        self.ask_qty = torch.where(hp_mask, self.ask_qty, torch.zeros_like(self.ask_qty))
        self.ask_px = torch.where(hp_mask, self.ask_px, torch.zeros_like(self.ask_px))
        self.ask_qa = torch.where(hp_mask, self.ask_qa, torch.zeros_like(self.ask_qa))

        # Потраншевый учет LIFO (Знание №2):
        # - Если у нас есть докупленный на дне пролива транш (low_tranche_qty >= 1.0 по цене min_buy_px < avg_entry),
        #   то младшие слоты Ask 0..5 могут сразу разгружать этот нижний транш от (min_buy_px + 1 тик),
        #   зарабатывая внутри пролива и снижая безубыток оставшейся верхней позиции!
        # - Старшие слоты Ask 6..11 всегда стоят от (avg_entry + 1 тик).
        has_low_tr = (self.pos_qty > 1e-6) & (self.low_tranche_qty >= 1.0) & (self.min_buy_px < self.avg_entry - 0.1 * self.tick_size)
        base_avg_tp = torch.where(self.pos_qty > 1e-6, self.avg_entry + self.tick_size, a1)
        base_lifo_tp = torch.where(has_low_tr, torch.maximum(self.min_buy_px + self.tick_size, base_avg_tp - 3.0 * self.tick_size), base_avg_tp)

        slot_base_tp = torch.cat([
            base_lifo_tp.unsqueeze(1).expand(self.num_envs, 6),
            base_avg_tp.unsqueeze(1).expand(self.num_envs, S - 6),
        ], dim=1)  # [B, 12]

        raw_slot_a_px = torch.where(
            self.is_penny_slot,
            a1 - penny_add,
            a1 + self.slot_tick_offsets,
        ).unsqueeze(0).expand(self.num_envs, S)
        t_ask_px = torch.maximum(raw_slot_a_px, slot_base_tp + self.slot_tick_offsets.unsqueeze(0))  # [B, 12]

        # Авто-подтяжка отставших Ask-слотов, если рынок опустился ближе к нашему тейку (но не ниже t_ask_px)
        stale_a = (
            (self.ask_qty > 1e-6)
            & ((cur_t_scalar - self.ask_act_t) >= 20)
            & ((self.ask_px - t_ask_px) >= 1.9 * self.tick_size)
        )
        self.ask_qty = torch.where(stale_a, torch.zeros_like(self.ask_qty), self.ask_qty)
        self.ask_px = torch.where(stale_a, torch.zeros_like(self.ask_px), self.ask_px)
        self.ask_qa = torch.where(stale_a, torch.zeros_like(self.ask_qa), self.ask_qa)

        empty_a = self.ask_qty <= 1e-6
        a_w = torch.where(hp_mask, raw_outputs[:, S:2 * S], torch.zeros_like(raw_outputs[:, S:2 * S]))
        place_a = empty_a & (a_w >= 0.12) & hp_mask
        a_w_active = torch.where(place_a, a_w, torch.zeros_like(a_w))
        a_sum = a_w_active.sum(dim=1, keepdim=True)
        scale_a = torch.where(
            a_sum > 1.0,
            1.0 / torch.clamp(a_sum, min=1e-6),
            torch.where(a_sum >= 0.75, 1.0 / torch.clamp(a_sum, min=1e-6), torch.ones_like(a_sum)),
        )
        a_alloc = a_w_active * scale_a  # [B, 12]

        lvl_a_q = torch.clamp(self.asks_q[cur_t_scalar, self.slot_lob_lvl], min=10.0)
        base_a_qa_1d = torch.where(self.is_penny_slot & wide_sp, torch.zeros_like(lvl_a_q), lvl_a_q)
        base_a_qa = base_a_qa_1d.unsqueeze(0).expand(self.num_envs, S)

        unreserved_pos = torch.clamp(self.pos_qty - self.ask_qty.sum(dim=1), min=0.0).unsqueeze(1)
        t_ask_qty = unreserved_pos * a_alloc
        # Если активен нижний LIFO-транш, слоты 0..5 (стоящие ниже avg_entry) могут продать не более объема нижнего транша!
        lifo_already_placed = self.ask_qty[:, :6].sum(dim=1, keepdim=True)
        rem_low_tr = torch.clamp(self.low_tranche_qty.unsqueeze(1) - lifo_already_placed, min=0.0)
        want_low_sum = t_ask_qty[:, :6].sum(dim=1, keepdim=True).clamp(min=1e-6)
        low_cap_scale = torch.where(
            has_low_tr.unsqueeze(1) & (want_low_sum > rem_low_tr),
            rem_low_tr / want_low_sum,
            torch.ones_like(want_low_sum),
        )
        t_ask_qty = torch.cat([t_ask_qty[:, :6] * low_cap_scale, t_ask_qty[:, 6:]], dim=1)

        t_ask_usd = t_ask_qty * t_ask_px
        valid_a = place_a & (t_ask_qty >= 1.0)
        t_ask_qty = torch.where(valid_a, t_ask_qty, torch.zeros_like(t_ask_qty))

        a_over50 = torch.sigmoid((t_ask_usd - 50.0) / 18.0)
        a_self_impact_qa = base_a_qa * (1.0 + 0.75 * a_over50) + a_over50 * (0.45 * t_ask_qty)
        self.frontrun_hits = self.frontrun_hits + ((valid_a & (t_ask_usd > 60.0)).any(dim=1)).long()

        self.ask_px = torch.where(valid_a, t_ask_px, self.ask_px)
        self.ask_qty = torch.where(valid_a, t_ask_qty, self.ask_qty)
        self.ask_qa = torch.where(valid_a, a_self_impact_qa, self.ask_qa)
        self.ask_act_t = torch.where(valid_a, torch.full_like(self.ask_act_t, cur_t_scalar + 1), self.ask_act_t)

        tot_active_orders = (self.bid_qty > 1e-6).long().sum(dim=1) + (self.ask_qty > 1e-6).long().sum(dim=1)
        self.multi_orders_used = self.multi_orders_used + (tot_active_orders >= 2).long()

        # 4. Безветвевое ЧЕСТНОЕ ПОСЛЕДОВАТЕЛЬНОЕ исполнение на тике t + 1 по 12 Bid и 12 Ask слотам!
        # Устранены эксплойты:
        #  - Никакой фантомной ликвидности (+l1_bid_q / +l1_ask_q) при касании спреда: исполнение идет строго
        #    от реального ленточного потока сделок sv / bv (плюс 12% l1_q только при строгом пробое уровня на >= 1 тик).
        #  - Честный последовательный водопад (Sequential Queue + Fill Waterfall): поток сделок расходуется
        #    последовательно сверху вниз сначала на очередь (bid_qa), затем на сам ордер (bid_qty), и только
        #    остаток переходит к следующему слоту!
        next_t = min(cur_t_scalar + 1, self.max_steps - 1)
        nb1 = torch.clamp(self.bids[next_t, 0], min=self.tick_size)
        na1 = torch.clamp(self.asks[next_t, 0], min=nb1 + self.tick_size)
        sv = self.sell_vol[next_t]
        bv = self.buy_vol[next_t]

        # 4a. Строгий последовательный водопад по 12 слотам Bid [B, 12]
        act_b = (self.bid_qty > 1e-6) & (next_t >= self.bid_act_t)
        strict_cross_b = na1 <= (self.bid_px - 0.5 * self.tick_size)
        hit_b = act_b & (strict_cross_b | ((nb1 <= self.bid_px + 1e-6) & (sv > 0.0)))

        # Общий доступный поток продавцов на тике next_t для каждого агента [B, 1]
        any_strict_b = (act_b & strict_cross_b).any(dim=1, keepdim=True)
        tot_flow_b = torch.where(any_strict_b, sv + 0.12 * l1_bid_q, sv)  # [B, 1]

        # Для нечетных слотов (второй транш того же ценового эшелона) очередь уже учтена в четном слоте,
        # если четный слот активен на той же цене; иначе слот проходит свою очередь сам.
        even_act_b = act_b[:, 0::2]
        Incremental_qa_b = self.bid_qa.clone()
        Incremental_qa_b[:, 1::2] = torch.where(
            even_act_b,
            torch.clamp(self.bid_qa[:, 1::2] - self.bid_qa[:, 0::2], min=0.0),
            self.bid_qa[:, 1::2],
        )
        eff_qa_b = torch.where(hit_b, Incremental_qa_b, torch.zeros_like(self.bid_qa))
        eff_qty_b = torch.where(hit_b, self.bid_qty, torch.zeros_like(self.bid_qty))

        # Кумулятивный расход потока продавцов сверху вниз: каждый слот требует сначала пробить eff_qa_b, затем eff_qty_b
        slot_need_b = eff_qa_b + eff_qty_b  # [B, 12]
        cum_need_b = torch.cumsum(slot_need_b, dim=1)
        flow_before_slot_b = cum_need_b - slot_need_b
        rem_flow_at_slot_b = torch.clamp(tot_flow_b - flow_before_slot_b, min=0.0)

        qa_eaten_b = torch.minimum(eff_qa_b, rem_flow_at_slot_b)
        self.bid_qa = torch.where(hit_b, torch.clamp(self.bid_qa - qa_eaten_b, min=0.0), self.bid_qa)
        # Синхронизируем очередь второго транша, если первый транш стоял перед ним
        self.bid_qa[:, 1::2] = torch.where(
            act_b[:, 1::2] & even_act_b,
            torch.clamp(self.bid_qa[:, 1::2] - qa_eaten_b[:, 0::2], min=0.0),
            self.bid_qa[:, 1::2],
        )

        want_b = torch.minimum(eff_qty_b, torch.clamp(rem_flow_at_slot_b - eff_qa_b, min=0.0))  # [B, 12]
        cum_want_b = torch.cumsum(want_b, dim=1)
        prev_cum_b = cum_want_b - want_b
        max_afford_b = torch.clamp(self.cash * 0.998 / (b1 * (1.0 + self.maker_fee)), min=0.0).unsqueeze(1)
        f_q = torch.minimum(want_b, torch.clamp(max_afford_b - prev_cum_b, min=0.0))  # [B, 12]

        has_slot_b = f_q > 1e-5
        tot_fq = f_q.sum(dim=1)
        tot_cost_raw = (f_q * self.bid_px).sum(dim=1)
        tot_cost = tot_cost_raw * (1.0 + self.maker_fee)
        has_buy = tot_fq > 1e-5

        old_q = self.pos_qty
        new_q = old_q + tot_fq
        avg_fill_b_px = tot_cost_raw / torch.clamp(tot_fq, min=1e-6)
        min_fill_b_px = torch.where(has_slot_b, self.bid_px, torch.full_like(self.bid_px, 1e9)).min(dim=1).values

        # Обновляем LIFO нижний транш только если докупка существенно (минимум на 1 полный тик) дешевле средней цены
        is_dip_buy = has_buy & (old_q > 1e-6) & (avg_fill_b_px <= self.avg_entry - 0.95 * self.tick_size)
        self.low_tranche_qty = torch.where(
            has_buy & (old_q <= 1e-6),
            torch.zeros_like(self.low_tranche_qty),
            torch.where(is_dip_buy, self.low_tranche_qty + tot_fq, self.low_tranche_qty),
        )
        self.min_buy_px = torch.where(
            has_buy,
            torch.where(
                old_q <= 1e-6,
                avg_fill_b_px,
                torch.where(is_dip_buy, avg_fill_b_px, self.min_buy_px),
            ),
            self.min_buy_px,
        )

        self.avg_entry = torch.where(
            has_buy,
            (old_q * self.avg_entry + tot_cost_raw) / torch.clamp(new_q, min=1e-6),
            self.avg_entry,
        )
        self.pos_cost_usd = torch.where(has_buy, self.pos_cost_usd + tot_cost, self.pos_cost_usd)
        self.entry_t = torch.where(has_buy & (old_q <= 1e-6), torch.full_like(self.entry_t, next_t), self.entry_t)
        self.cash = torch.where(has_buy, torch.clamp(self.cash - tot_cost, min=0.0), self.cash)
        self.pos_qty = torch.where(has_buy, new_q, self.pos_qty)

        rem_ord_b = torch.clamp(self.bid_qty - f_q, min=0.0)
        self.partial_fills = self.partial_fills + (has_slot_b & (rem_ord_b > 1e-4)).long().sum(dim=1)
        self.bid_qty = torch.where(
            has_slot_b,
            torch.where(rem_ord_b >= 1.0, rem_ord_b, torch.zeros_like(rem_ord_b)),
            self.bid_qty,
        )
        self.n_trades = self.n_trades + has_slot_b.long().sum(dim=1)

        # 4b. Строгий последовательный водопад по 12 слотам Ask [B, 12]
        act_a = (self.ask_qty > 1e-6) & (next_t >= self.ask_act_t) & (self.pos_qty.unsqueeze(1) > 1e-6)
        strict_cross_a = nb1 >= (self.ask_px + 0.5 * self.tick_size)
        hit_a = act_a & (strict_cross_a | ((na1 >= self.ask_px - 1e-6) & (bv > 0.0)))

        any_strict_a = (act_a & strict_cross_a).any(dim=1, keepdim=True)
        tot_flow_a = torch.where(any_strict_a, bv + 0.12 * l1_ask_q, bv)  # [B, 1]

        even_act_a = act_a[:, 0::2]
        Incremental_qa_a = self.ask_qa.clone()
        Incremental_qa_a[:, 1::2] = torch.where(
            even_act_a,
            torch.clamp(self.ask_qa[:, 1::2] - self.ask_qa[:, 0::2], min=0.0),
            self.ask_qa[:, 1::2],
        )
        eff_qa_a = torch.where(hit_a, Incremental_qa_a, torch.zeros_like(self.ask_qa))
        eff_qty_a = torch.where(hit_a, self.ask_qty, torch.zeros_like(self.ask_qty))

        slot_need_a = eff_qa_a + eff_qty_a  # [B, 12]
        cum_need_a = torch.cumsum(slot_need_a, dim=1)
        flow_before_slot_a = cum_need_a - slot_need_a
        rem_flow_at_slot_a = torch.clamp(tot_flow_a - flow_before_slot_a, min=0.0)

        qa_eaten_a = torch.minimum(eff_qa_a, rem_flow_at_slot_a)
        self.ask_qa = torch.where(hit_a, torch.clamp(self.ask_qa - qa_eaten_a, min=0.0), self.ask_qa)
        self.ask_qa[:, 1::2] = torch.where(
            act_a[:, 1::2] & even_act_a,
            torch.clamp(self.ask_qa[:, 1::2] - qa_eaten_a[:, 0::2], min=0.0),
            self.ask_qa[:, 1::2],
        )

        want_a = torch.minimum(eff_qty_a, torch.clamp(rem_flow_at_slot_a - eff_qa_a, min=0.0))  # [B, 12]
        cum_want_a = torch.cumsum(want_a, dim=1)
        prev_cum_a = cum_want_a - want_a
        cap_a = self.pos_qty.unsqueeze(1)
        f_qa = torch.minimum(want_a, torch.clamp(cap_a - prev_cum_a, min=0.0))  # [B, 12]

        has_slot_a = f_qa > 1e-5
        tot_fqa = f_qa.sum(dim=1)
        tot_rev = (f_qa * self.ask_px).sum(dim=1) * (1.0 - self.maker_fee)
        has_sell = tot_fqa > 1e-5

        # Строгий бухгалтерский учет LIFO:
        # Сколько реально стоили проданные монеты (нижний транш по min_buy_px, верхний по avg_entry):
        sold_low_q = torch.minimum(f_qa[:, :6].sum(dim=1), self.low_tranche_qty)
        sold_high_q = torch.clamp(tot_fqa - sold_low_q, min=0.0)
        sold_cost_usd = torch.minimum(
            self.pos_cost_usd,
            sold_low_q * torch.minimum(self.min_buy_px, self.avg_entry) * (1.0 + self.maker_fee)
            + sold_high_q * self.avg_entry * (1.0 + self.maker_fee),
        )
        # Цикл считается прибыльным ТОЛЬКО если фактическая чистая выручка превысила фактическую себестоимость проданных монет!
        is_net_prof_sell = has_sell & (tot_rev > sold_cost_usd + 1e-5)
        self.sell_cycles = self.sell_cycles + has_sell.long()
        self.prof_sell_cycles = self.prof_sell_cycles + is_net_prof_sell.long()

        self.cash = torch.where(has_sell, self.cash + tot_rev, self.cash)
        rem_p = torch.clamp(self.pos_qty - tot_fqa, min=0.0)
        self.pos_qty = torch.where(has_sell, rem_p, self.pos_qty)
        # Честный остаток себестоимости без двойного учета скидки:
        self.pos_cost_usd = torch.where(
            has_sell,
            torch.where(rem_p > 1e-6, torch.clamp(self.pos_cost_usd - sold_cost_usd, min=0.0), torch.zeros_like(self.pos_cost_usd)),
            self.pos_cost_usd,
        )
        self.avg_entry = torch.where(
            has_sell,
            torch.where(rem_p > 1e-6, self.pos_cost_usd / (rem_p * (1.0 + self.maker_fee)), torch.zeros_like(self.avg_entry)),
            self.avg_entry,
        )
        self.low_tranche_qty = torch.where(
            has_sell,
            torch.where(rem_p > 1e-6, torch.clamp(self.low_tranche_qty - sold_low_q, min=0.0), torch.zeros_like(self.low_tranche_qty)),
            self.low_tranche_qty,
        )
        self.min_buy_px = torch.where(
            self.pos_qty > 1e-6,
            torch.where(self.low_tranche_qty > 1e-4, self.min_buy_px, self.avg_entry),
            torch.zeros_like(self.min_buy_px),
        )

        rem_ord_a = torch.clamp(self.ask_qty - f_qa, min=0.0)
        self.partial_fills = self.partial_fills + (has_slot_a & (rem_ord_a > 1e-4)).long().sum(dim=1)
        self.ask_qty = torch.where(
            has_slot_a,
            torch.where(
                (rem_ord_a >= 1.0) & (self.pos_qty.unsqueeze(1) >= 1.0),
                torch.minimum(rem_ord_a, self.pos_qty.unsqueeze(1)),
                torch.zeros_like(rem_ord_a),
            ),
            self.ask_qty,
        )
        self.n_trades = self.n_trades + has_slot_a.long().sum(dim=1)
