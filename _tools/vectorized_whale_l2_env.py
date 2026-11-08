#!/usr/bin/env python3
"""
_tools/vectorized_whale_l2_env.py
=============================================================================
Векторизованная L2 FIFO Среда для Урока 5 (Whale Execution & Adaptive Sizing)
на бюджетах $95.5 .. $20,000 прямо в VRAM GPU (PyTorch CUDA).

Ключевые физические отличия от базовой среды Уроков 1-4:
1. ЧЕСТНЫЕ ЧАСТИЧНЫЕ НАЛИВЫ (True Partial Fills):
   Исполненный за 1 секунду объем строго ограничен реальным остатком встречного
   потока ленты (sell_vol / buy_vol за вычетом очереди q_ahead). Неисполненный
   остаток ордера остается стоять во главе очереди (q_ahead = 0).
2. МНОГОТРАНШЕВЫЙ НАБОР И РАЗГРУЗКА ПОЗИЦИИ (Iceberg / Tranche Pyramiding):
   Агент может набирать позицию несколькими траншами (с пересчетом VWAP avg_entry)
   и разгружать крупную позицию частями (SELL_ASK_ICEBERG_SLICE), не выставляя
   пугающую "плиту" в первый уровень Ask.
3. РЕАЛЬНЫЙ ПРОБОЙ СТАКАНА L1..L10 ПРИ PANIC_TAKER:
   Экстренный рыночный выход продает объем последовательно по уровням L1 -> L10,
   наказывая агента реальным Market Impact за паник-сейл крупной позиции.
4. 10 ДЕЙСТВИЙ И 3 СИГНАЛА ЕМКОСТИ ЛИКВИДНОСТИ (Liquidity Ratio 3D).
=============================================================================
"""

from typing import Tuple
import torch


class VectorizedWhaleL2Env:
    def __init__(
        self,
        bids: torch.Tensor,        # [N, 10] float32
        asks: torch.Tensor,        # [N, 10] float32
        bids_q: torch.Tensor,      # [N, 10] float32
        asks_q: torch.Tensor,      # [N, 10] float32
        buy_vol: torch.Tensor,     # [N] float32
        sell_vol: torch.Tensor,    # [N] float32
        lob_tensors: torch.Tensor, # [N, 10, 4] float32
        flow_market: torch.Tensor, # [N, 16] float32
        targets: torch.Tensor,     # [N, 4] float32
        num_envs: int = 8192,
        tick_size: float = 0.0001,
        maker_fee: float = 0.0010,
        taker_fee: float = 0.0020,
        initial_cash: float = 5000.0,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.num_envs = num_envs
        self.tick_size = tick_size
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.n_actions = 10

        self.bids = bids.to(device)
        self.asks = asks.to(device)
        self.bids_q = bids_q.to(device)
        self.asks_q = asks_q.to(device)
        self.buy_vol = buy_vol.to(device)
        self.sell_vol = sell_vol.to(device)
        self.lob_tensors = lob_tensors.to(device)
        self.flow_market = flow_market.to(device)
        self.targets = targets.to(device)
        self.n_ticks = len(self.bids)

        # Предвычисляем 30-секундную скользящую скорость покупок для оценки времени разгрузки
        bv_cpu = buy_vol.cpu().float()
        csum = torch.cumsum(bv_cpu, dim=0)
        roll30 = csum.clone()
        roll30[30:] = csum[30:] - csum[:-30]
        self.roll_buy_vol_30s = torch.clamp(roll30.to(device), min=1.0)

        self.budget = torch.full((num_envs,), float(initial_cash), dtype=torch.float32, device=device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_t = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pos_qty = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.avg_entry = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.entry_t = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.order_side = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.order_act = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.order_price = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.order_qty = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.q_ahead = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.active_after = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.cash = torch.full((num_envs,), float(initial_cash), dtype=torch.float32, device=device)
        self.n_trades = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.partial_fills = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.iceberg_unloads = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.slippage_usd = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def set_uniform_budget(self, budget_usd: float):
        """Устанавливает единый бюджет для всех сред перед началом нового окна."""
        self.budget.fill_(float(budget_usd))
        self.pos_qty.zero_()
        self.avg_entry.zero_()
        self.entry_t.zero_()
        self.order_side.fill_(-1)
        self.order_act.zero_()
        self.order_price.zero_()
        self.order_qty.zero_()
        self.q_ahead.zero_()
        self.active_after.zero_()
        self.cash.fill_(float(budget_usd))
    def get_action_masks(self, cur_t_scalar: int) -> torch.Tensor:
        """
        Маски 10 действий [B, 10]:
          0: HOLD
          1: BUY_BEST_MICRO        (до $220 / 3% L1)
          2: BUY_PASSIVE_MICRO     (до $260 / 4% L1 на b1 - tick)
          3: SELL_ASK_BEST_FULL    (выставить всю позицию на a1)
          4: SELL_ASK_PASSIVE_FULL (выставить всю позицию на a1 + tick)
          5: CANCEL_ORDERS
          6: PANIC_TAKER_SWEEP     (рыночный выход по уровням L1..L10)
          7: BUY_BEST_WHALE        (до $650 / 7% L1 на b1)
          8: BUY_LAYERED_ICEBERG   (до $1,100 / 11% L1 на b1 - tick)
          9: SELL_ASK_ICEBERG_SLICE(частичная разгрузка траншем <= $450 / 6% L1 Ask)
        """
        masks = torch.zeros((self.num_envs, self.n_actions), dtype=torch.bool, device=self.device)
        masks[:, 0] = True

        b1 = torch.clamp(self.bids[cur_t_scalar, 0], min=self.tick_size)
        curr_pos_usd = self.pos_qty * b1
        free_cash = torch.clamp(self.cash, min=0.0)
        can_accumulate = curr_pos_usd < (0.55 * self.budget)

        can_micro = can_accumulate & (free_cash >= 25.0)
        masks[can_micro, 1] = True
        masks[can_micro, 2] = True

        can_whale = can_accumulate & (free_cash >= 350.0)
        masks[can_whale, 7] = True
        masks[can_whale, 8] = True

        has_pos = self.pos_qty > 1e-6
        masks[has_pos, 3] = True
        masks[has_pos, 4] = True
        masks[has_pos, 6] = True
        masks[has_pos & (curr_pos_usd >= 300.0), 9] = True

        has_order = self.order_side >= 0
        masks[has_order, 5] = True
        return masks

    def get_observations(self, cur_t_scalar: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает LOB [1, 10, 4], Flow [B, 16], Targets [1, 4], LiquiditySignals [B, 3]."""
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

        has_bid = (self.order_side == 0).float()
        has_ask = (self.order_side == 1).float()
        dist_bid = (b1 - self.order_price) / self.tick_size
        dist_ask = (self.order_price - a1) / self.tick_size
        dist_ticks = torch.where(has_bid > 0.5, dist_bid, torch.where(has_ask > 0.5, dist_ask, torch.zeros_like(has_pos)))

        flow[:, 8] = has_pos
        flow[:, 9] = pos_ratio
        flow[:, 10] = free_cash_ratio
        flow[:, 11] = torch.clamp(unr_ticks / 5.0, -2.0, 2.0)
        flow[:, 12] = hold_ratio
        flow[:, 13] = has_bid
        flow[:, 14] = has_ask
        flow[:, 15] = torch.clamp(dist_ticks / 5.0, -2.0, 2.0)

        # 3 физических сигнала емкости стакана для управления крупным бюджетом
        bud_l1 = torch.log1p(self.budget / l1_bid_usd) / 3.0
        pos_l1_ask = torch.clamp((self.pos_qty * a1) / l1_ask_usd, 0.0, 4.0) / 2.0
        pos_tape = torch.clamp(self.pos_qty / self.roll_buy_vol_30s[cur_t_scalar], 0.0, 10.0) / 5.0
        liq_3d = torch.stack([bud_l1, pos_l1_ask, pos_tape], dim=-1)

        return lob, flow, tgt, liq_3d
    def step(self, actions: torch.Tensor, cur_t_scalar: int):
        """
        Один шаг всех B сред с честной физикой частичных наливов и пробоем L1..L10.
        """
        b1 = torch.clamp(self.bids[cur_t_scalar, 0], min=self.tick_size)
        a1 = torch.clamp(self.asks[cur_t_scalar, 0], min=b1 + self.tick_size)
        l1_bid_q = torch.clamp(self.bids_q[cur_t_scalar, 0], min=10.0)
        l1_ask_q = torch.clamp(self.asks_q[cur_t_scalar, 0], min=10.0)

        # 1. ПОСТАНОВКА ЛИМИТНЫХ ПОКУПОК (1, 2, 7, 8)
        is_b_micro_best = actions == 1
        is_b_micro_pass = actions == 2
        is_b_whale_best = actions == 7
        is_b_whale_pass = actions == 8
        is_buy = is_b_micro_best | is_b_micro_pass | is_b_whale_best | is_b_whale_pass

        if is_buy.any():
            is_best = is_b_micro_best | is_b_whale_best
            px_best = torch.where((a1 - b1) >= 2.0 * self.tick_size, b1 + self.tick_size, b1)
            px_pass = b1 - self.tick_size
            target_px = torch.where(is_best, px_best, px_pass)

            # Адаптивный расчет транша от бюджета и глубины L1
            usd_m_best = torch.minimum(torch.clamp(self.budget * 0.22, min=25.0, max=220.0), l1_bid_q * b1 * 0.035)
            usd_m_pass = torch.minimum(torch.clamp(self.budget * 0.25, min=25.0, max=260.0), l1_bid_q * b1 * 0.045)
            usd_w_best = torch.minimum(torch.clamp(self.budget * 0.22, min=250.0, max=650.0), l1_bid_q * b1 * 0.075)
            usd_w_pass = torch.minimum(torch.clamp(self.budget * 0.26, min=300.0, max=1100.0), l1_bid_q * b1 * 0.110)

            tranche_usd = torch.where(
                is_b_micro_best, usd_m_best,
                torch.where(is_b_micro_pass, usd_m_pass, torch.where(is_b_whale_best, usd_w_best, usd_w_pass))
            )
            tranche_usd = torch.minimum(tranche_usd, torch.clamp(self.cash * 0.98, min=0.0))

            same_buy = is_buy & (self.order_side == 0) & (torch.abs(self.order_price - target_px) < 1e-6)
            upd_buy = is_buy & (~same_buy) & (tranche_usd >= 20.0)
            if upd_buy.any():
                self.order_side[upd_buy] = 0
                self.order_act[upd_buy] = actions[upd_buy]
                self.order_price[upd_buy] = target_px[upd_buy]
                self.order_qty[upd_buy] = tranche_usd[upd_buy] / torch.clamp(target_px[upd_buy], min=self.tick_size)
                is_penny = target_px[upd_buy] > (b1 + 0.5 * self.tick_size)
                self.q_ahead[upd_buy] = torch.where(is_penny, torch.zeros_like(self.q_ahead[upd_buy]), l1_bid_q)
                self.active_after[upd_buy] = cur_t_scalar + 1

        # 2. ПОСТАНОВКА ЛИМИТНЫХ ПРОДАЖ (3, 4, 9 - Iceberg Slice)
        is_s_best = (actions == 3) & (self.pos_qty > 1e-6)
        is_s_pass = (actions == 4) & (self.pos_qty > 1e-6)
        is_s_ice = (actions == 9) & (self.pos_qty > 1e-6)
        is_sell = is_s_best | is_s_pass | is_s_ice

        if is_sell.any():
            target_ask = torch.where(is_s_pass, a1 + self.tick_size, a1)
            ice_max_q = torch.minimum(torch.full_like(self.pos_qty, 450.0) / a1, l1_ask_q * 0.065)
            sell_q = torch.where(is_s_ice, torch.minimum(self.pos_qty, ice_max_q), self.pos_qty)

            same_sell = is_sell & (self.order_side == 1) & (torch.abs(self.order_price - target_ask) < 1e-6)
            upd_sell = is_sell & (~same_sell) & (sell_q > 1e-6)
            if upd_sell.any():
                self.order_side[upd_sell] = 1
                self.order_act[upd_sell] = actions[upd_sell]
                self.order_price[upd_sell] = target_ask[upd_sell]
                self.order_qty[upd_sell] = sell_q[upd_sell]
                self.q_ahead[upd_sell] = l1_ask_q
                self.active_after[upd_sell] = cur_t_scalar + 1

        # 3. ОТМЕНА ОРДЕРА (5)
        is_cancel = actions == 5
        if is_cancel.any():
            self.order_side[is_cancel] = -1

        # 4. ЭКСТРЕННЫЙ РЫНОЧНЫЙ ВЫХОД С ПРОБОЕМ СТАКАНА L1..L10 (6)
        is_panic = (actions == 6) & (self.pos_qty > 1e-6)
        if is_panic.any():
            rem_q = self.pos_qty[is_panic].clone()
            proceeds = torch.zeros_like(rem_q)
            for lvl in range(10):
                lvl_p = torch.clamp(self.bids[cur_t_scalar, lvl], min=self.tick_size)
                lvl_q = torch.clamp(self.bids_q[cur_t_scalar, lvl], min=1.0)
                take_q = torch.minimum(rem_q, lvl_q if lvl < 9 else rem_q)
                proceeds += take_q * lvl_p
                rem_q = torch.clamp(rem_q - take_q, min=0.0)
            ideal_proceeds = self.pos_qty[is_panic] * b1
            slip = torch.clamp(ideal_proceeds - proceeds, min=0.0)
            fee = proceeds * self.taker_fee
            self.cash[is_panic] += proceeds - fee
            self.slippage_usd[is_panic] += slip
            self.pos_qty[is_panic] = 0.0
            self.avg_entry[is_panic] = 0.0
            self.order_side[is_panic] = -1
            self.n_trades[is_panic] += 1

        # 5. ПЕРЕХОД НА СЛЕДУЮЩИЙ ТИК nt И ИСПОЛНЕНИЕ ЧАСТИЧНЫХ НАЛИВОВ (FIFO + Tape Cap)
        nt = min(cur_t_scalar + 1, self.n_ticks - 1)
        nb1 = torch.clamp(self.bids[nt, 0], min=self.tick_size)
        na1 = torch.clamp(self.asks[nt, 0], min=nb1 + self.tick_size)
        tol = self.tick_size * 0.25
        active_mask = (self.order_side >= 0) & (nt >= self.active_after)

        buy_active = active_mask & (self.order_side == 0)
        if buy_active.any():
            crossed_b = na1 <= self.order_price + tol
            sv = self.sell_vol[nt]
            avail_b = torch.clamp(sv - self.q_ahead, min=0.0) + torch.where(crossed_b, self.asks_q[nt, 0] * 0.35, 0.0)
            exec_b = torch.minimum(self.order_qty, avail_b)
            can_fill_b = buy_active & (exec_b > 1e-4)

            if can_fill_b.any():
                fq = exec_b[can_fill_b]
                fpx = self.order_price[can_fill_b]
                cost = fq * fpx * (1.0 + self.maker_fee)
                old_q = self.pos_qty[can_fill_b]
                new_q = old_q + fq
                self.avg_entry[can_fill_b] = torch.where(
                    old_q > 1e-6, (old_q * self.avg_entry[can_fill_b] + fq * fpx) / new_q, fpx
                )
                self.entry_t[can_fill_b] = torch.where(old_q > 1e-6, self.entry_t[can_fill_b], nt)
                self.pos_qty[can_fill_b] = new_q
                self.cash[can_fill_b] -= cost
                self.order_qty[can_fill_b] -= fq
                self.q_ahead[can_fill_b] = 0.0
                self.n_trades[can_fill_b] += 1

                done_b = self.order_qty[can_fill_b] <= 1e-4
                part_b = ~done_b
                if part_b.any():
                    idx_part = torch.where(can_fill_b)[0][part_b]
                    self.partial_fills[idx_part] += 1
                if done_b.any():
                    idx_done = torch.where(can_fill_b)[0][done_b]
                    self.order_side[idx_done] = -1

            unfilled_b = buy_active & (~can_fill_b)
            if unfilled_b.any():
                self.q_ahead[unfilled_b] = torch.clamp(self.q_ahead[unfilled_b] - sv, min=0.0)

        sell_active = active_mask & (self.order_side == 1)
        if sell_active.any():
            crossed_s = nb1 >= self.order_price - tol
            bv = self.buy_vol[nt]
            avail_s = torch.clamp(bv - self.q_ahead, min=0.0) + torch.where(crossed_s, self.bids_q[nt, 0] * 0.35, 0.0)
            exec_s = torch.minimum(self.order_qty, torch.minimum(self.pos_qty, avail_s))
            can_fill_s = sell_active & (exec_s > 1e-4)

            if can_fill_s.any():
                fq = exec_s[can_fill_s]
                fpx = self.order_price[can_fill_s]
                rev = fq * fpx * (1.0 - self.maker_fee)
                self.cash[can_fill_s] += rev
                self.pos_qty[can_fill_s] = torch.clamp(self.pos_qty[can_fill_s] - fq, min=0.0)
                self.order_qty[can_fill_s] = torch.clamp(self.order_qty[can_fill_s] - fq, min=0.0)
                self.q_ahead[can_fill_s] = 0.0
                self.n_trades[can_fill_s] += 1

                was_ice = self.order_act[can_fill_s] == 9
                if was_ice.any():
                    self.iceberg_unloads[torch.where(can_fill_s)[0][was_ice]] += 1

                done_s = (self.order_qty[can_fill_s] <= 1e-4) | (self.pos_qty[can_fill_s] <= 1e-4)
                if done_s.any():
                    idx_done_s = torch.where(can_fill_s)[0][done_s]
                    self.order_side[idx_done_s] = -1
                    flat_s = self.pos_qty[idx_done_s] <= 1e-4
                    if flat_s.any():
                        self.avg_entry[idx_done_s[flat_s]] = 0.0

            unfilled_s = sell_active & (~can_fill_s)
            if unfilled_s.any():
                self.q_ahead[unfilled_s] = torch.clamp(self.q_ahead[unfilled_s] - bv, min=0.0)
