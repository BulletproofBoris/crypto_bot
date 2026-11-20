#!/usr/bin/env python3
"""
_tools/test_subcent_0001_precision.py
=============================================================================
Верификация масштабно-инвариантной поддержки субцентового шага $0.0001
(price_precision = 4 на SafeTrade pearlusdt) с чемпионом Урока 27:
  1) Форматирование REST-заявок SafeTradeClient (`price_precision=4` -> `1.5405`).
  2) Независимость `exec_tick_size = 0.0001` от эталонного признакового `tick_size = 0.01`.
  3) Субцентовый penny-jump (`1.5404 -> 1.5405`), сетка 24 эшелонов в шаге `$0.0001`
     и точное разделение соседних уровней очереди `lookup_sparse_lob_qty_1d`.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _tools.live_rl_trader import SafeTradePrivateClient as SafeTradeClient
from _tools.l26_live_bridge import lookup_sparse_lob_qty_1d
from _tools.l27_live_bridge import forward_l27_multimodal_live, compute_l27_multimodal_plan


class DummySubcentTrader:
    def __init__(self, ckpt_path: Path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.coin = "prl"
        self.market = "pearlusdt"
        self.tick_size = 0.01          # Эталонный центовый масштаб признаков нейросети
        self.price_precision = 4       # Реальная точность SafeTrade pearlusdt
        self.amount_precision = 4
        self.min_order_amount = 2.0
        self.exec_tick_size = 0.0001   # Реальный шаг стакана $0.0001

        self.initial_budget_usd = 801.34
        self.bot_cash_usd = 620.00
        self.pos_qty = 115.0
        self.avg_entry = 1.5432

        self.l7_w1 = ck["w1"].float()
        self.l7_b1 = ck["b1"].float()
        self.l7_w_res1 = ck["w_res1"].float()
        self.l7_b_res1 = ck["b_res1"].float()
        self.l7_w_res2 = ck["w_res2"].float()
        self.l7_w_rec = ck["w_rec"].float()
        self.l7_b_rec = ck["b_rec"].float()
        self.l7_mem_h = torch.zeros((1, 16), dtype=torch.float32)

        self.l7_w_gate1 = ck["w_gate1"].float()
        self.l7_b_gate1 = ck["b_gate1"].float()
        self.l7_w_gate2 = ck["w_gate2"].float()
        self.l7_b_gate2 = ck["b_gate2"].float()
        self.l7_w2_moe = ck["w2_moe"].float()
        self.l7_b2_moe = ck["b2_moe"].float()

        self.l7_w_reg1 = ck["w_reg1"].float()
        self.l7_b_reg1_b = ck["b_reg1_b"].float()
        self.l7_b_reg1_a = ck["b_reg1_a"].float()
        self.l7_w_reg2 = ck["w_reg2"].float()
        self.l7_b_reg2_b = ck["b_reg2_b"].float()
        self.l7_b_reg2_a = ck["b_reg2_a"].float()

        self.l27_api_tokens = 10.0
        self.bid_slots = [{"slot": i, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "qa0": 0.0, "qa": 0.0} for i in range(12)]
        self.ask_slots = [{"slot": i, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "qa0": 0.0, "qa": 0.0} for i in range(12)]


def main() -> None:
    # 1. Проверяем REST-форматирование SafeTradeClient без сетевого вызова:
    captured = {}

    class MockClient(SafeTradeClient):
        def __init__(self):
            pass

        def _private_req(self, method: str, path: str, payload: dict = None):
            captured["payload"] = payload
            return {"id": 999001, "price": payload["price"], "origin_amount": payload["amount"]}

    mc = MockClient()
    res = mc.place_limit_order("pearlusdt", "buy", 24.56789, 1.54051, price_precision=4, amount_precision=4, min_amount=2.0)
    assert captured["payload"]["price"] == "1.5405", f"Expected '1.5405', got {captured['payload']['price']}"
    assert captured["payload"]["amount"] == "24.5679", f"Expected '24.5679', got {captured['payload']['amount']}"

    # 2. Проверяем точное разделение соседних субцентовых уровней $1.5404 и $1.5403:
    bp_10 = np.array([1.5404, 1.5403, 1.5401, 1.5385, 1.5350, 1.5300, 1.5250, 1.5200, 1.5150, 1.5100], dtype=np.float32)
    bq_10 = np.array([120.0, 850.0, 410.0, 1500.0, 2200.0, 9400.0, 500.0, 300.0, 200.0, 100.0], dtype=np.float32)
    q_at_5404 = lookup_sparse_lob_qty_1d(bp_10, bq_10, 1.5404, 0.0001)
    q_at_5403 = lookup_sparse_lob_qty_1d(bp_10, bq_10, 1.5403, 0.0001)
    q_at_5405 = lookup_sparse_lob_qty_1d(bp_10, bq_10, 1.5405, 0.0001)
    assert abs(q_at_5404 - 120.0) < 1e-3, f"Expected 120.0 at 1.5404, got {q_at_5404}"
    assert abs(q_at_5403 - 850.0) < 1e-3, f"Expected 850.0 at 1.5403, got {q_at_5403}"
    assert abs(q_at_5405 - 0.0) < 1e-3, f"Expected 0.0 at empty penny-jump 1.5405, got {q_at_5405}"

    # 3. Проверяем L27 Multimodal Plan с чемпионом Волны 6 на реальном субцентовом спреде ($0.0093):
    ckpt = ROOT / "models" / "prl" / "lesson_27_multimodal_moe" / "lesson_27_champ_latest.pt"
    tr = DummySubcentTrader(ckpt)
    torch.manual_seed(7)
    z_480 = torch.randn((1, 480), dtype=torch.float32) * 0.35
    acct_8 = [1.0, 0.22, 0.77, -0.05, 0.30, 0.0, 0.0, 0.0]
    forward_l27_multimodal_live(tr, z_480, acct_8, now_q_ts=1000.0)

    b1_live, a1_live = 1.5404, 1.5497  # Спред внутри 1 цента: $0.0093 = 93 микротика!
    plan = compute_l27_multimodal_plan(tr, b1=b1_live, a1=a1_live, now_ts=1000.0)
    t_bid = plan["t_bid_px"]
    t_ask = plan["t_ask_px"]

    # Все цены кратны $0.0001 (с точностью float32 ~1e-6), maker-safe (bid < a1, ask > b1), и содержат субцентовую детализацию!
    assert np.all(t_bid <= a1_live - 0.0001 + 1e-6), f"Bid crossed ask: {t_bid}"
    assert np.all(t_ask >= b1_live + 0.0001 - 1e-6), f"Ask crossed bid: {t_ask}"
    subcent_bids = sum(1 for p in t_bid if abs(round(float(p), 4) - round(float(p), 2)) > 1e-5)
    subcent_asks = sum(1 for p in t_ask if abs(round(float(p), 4) - round(float(p), 2)) > 1e-5)
    assert subcent_bids >= 8 and subcent_asks >= 8, f"Expected true 4-decimal grid, got {t_bid} / {t_ask}"

    print(
        f"SUBCENT_0001_OK | payload_px={res['price']} | "
        f"qa(1.5404)={q_at_5404:.0f}, qa(1.5403)={q_at_5403:.0f}, qa(1.5405)={q_at_5405:.0f} | "
        f"b1={b1_live:.4f}, a1={a1_live:.4f} | "
        f"top3_bids={[round(float(x), 4) for x in t_bid[:3]]} | "
        f"top3_asks={[round(float(x), 4) for x in t_ask[:3]]} | "
        f"g_mode={[round(float(x), 3) for x in plan['g_mode']]}"
    )


if __name__ == "__main__":
    main()
