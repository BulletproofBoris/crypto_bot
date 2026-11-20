#!/usr/bin/env python3
"""
_tools/test_order_orchestrator.py
Проверка инвариантов Оркестратора v2 (Break-Even Floor Guard, Inventory Skew, Monotonicity, Anti-Dust).
"""
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.order_orchestrator import orchestrate_grid_orders, compute_breakeven_floor_px
import _tools.order3_live_bridge as bridge
import py_compile


def main():
    py_compile.compile(str(BASE_DIR / "_tools/live_rl_trader.py"), doraise=True)
    py_compile.compile(str(BASE_DIR / "_tools/order3_live_bridge.py"), doraise=True)
    py_compile.compile(str(BASE_DIR / "_tools/order_orchestrator.py"), doraise=True)
    print("✅ Синтаксическая компиляция live_rl_trader / order3_live_bridge / order_orchestrator пройдена.")

    # 1. Нормальный рынок без позиции
    r1 = orchestrate_grid_orders(
        budget_usd=1024.0, cash_usd=1024.0, pos_qty=0.0,
        b1=1.8400, a1=1.8450, capital_exp=0.75, grid_mode_idx=1, spread_tight=0.7
    )
    act_b = r1["bid_px"][r1["bid_qty"] > 0]
    assert len(act_b) == 4, f"Expected 4 active bids, got {len(act_b)}"
    assert np.all(np.diff(act_b) < 0), f"Bids not strictly decreasing: {act_b}"
    assert np.all(r1["bid_px"][r1["bid_qty"] > 0] * r1["bid_qty"][r1["bid_qty"] > 0] >= 24.0), "Dust order detected!"
    print(f"✅ Тест 1 (Чистый кэш): {len(act_b)} слотов, цены={np.round(act_b, 4)}, сумма=${r1['total_bid_usd']:.2f}")

    # 2. Просадка рынка ниже avg_entry ($1.8440 -> рынок упал на $1.8000): НИ ОДНОЙ продажи в минус!
    be_fl = compute_breakeven_floor_px(1.8440, tick_sz=0.0001, maker_fee=0.0010)
    r2 = orchestrate_grid_orders(
        budget_usd=1024.0, cash_usd=650.0, pos_qty=200.0,
        b1=1.7980, a1=1.8010, capital_exp=0.75, grid_mode_idx=2,
        avg_entry=1.8440, low_tranche_px=1.8100, low_tranche_qty=40.0,
    )
    act_a = r2["ask_px"][r2["ask_qty"] > 0]
    lifo_fl = compute_breakeven_floor_px(1.8100, tick_sz=0.0001, maker_fee=0.0010)
    assert act_a[0] >= lifo_fl - 1e-7, f"LIFO slot below lifo break-even: {act_a[0]} < {lifo_fl}"
    assert np.all(act_a[1:] >= be_fl - 1e-7), f"Core ask slots below core break-even {be_fl}: {act_a}"
    assert np.all(np.diff(act_a) > 0), f"Asks not strictly increasing: {act_a}"
    print(f"✅ Тест 2 (Break-Even Guard на проливе $1.80 при входе $1.8440): Ask={np.round(act_a, 4)} (LIFO_BE=${lifo_fl:.4f}, Core_BE=${be_fl:.4f})")

    # 3. Перегруз инвентаря >= 65%: полная блокировка новых покупок (Inventory Brake)
    r3 = orchestrate_grid_orders(
        budget_usd=1024.0, cash_usd=300.0, pos_qty=400.0,
        b1=1.8400, a1=1.8440, capital_exp=0.85, grid_mode_idx=2, avg_entry=1.8400
    )
    assert np.sum(r3["bid_qty"]) == 0.0, f"Expected 0 bids at 72% inventory, got {r3['bid_qty']}"
    print(f"✅ Тест 3 (Inventory Skew Brake при pos_ratio={r3['pos_ratio']:.1%}): новые покупки заблокированы (bid_usd=${r3['total_bid_usd']:.2f}).")

    # 4. Токсичный шторм (toxicity_score = 0.80): полная блокировка новых покупок
    r4 = orchestrate_grid_orders(
        budget_usd=1024.0, cash_usd=900.0, pos_qty=50.0,
        b1=1.8400, a1=1.8440, capital_exp=0.80, toxicity_score=0.80, avg_entry=1.8350
    )
    assert np.sum(r4["bid_qty"]) == 0.0, "Expected 0 bids during toxic storm!"
    print("✅ Тест 4 (Toxicity Storm Guard): покупки мгновенно заморожены.")
    print("🎉 ВСЕ ТЕСТЫ STAGE 1 УСПЕШНО ПРОЙДЕНЫ!")


if __name__ == "__main__":
    main()
