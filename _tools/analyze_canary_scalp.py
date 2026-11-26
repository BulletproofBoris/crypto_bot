#!/usr/bin/env python3
"""
_tools/analyze_canary_scalp.py
=============================================================================
Сводный отчёт по живому канареечному тесту широкого спреда (Режим B):
читает локальный или скачанный с VPS файл data/logs/canary_scalp_telemetry.jsonl
и выдаёт 3 калибровочные константы для симулятора fused_order3_clean_tick.py:
  1. Доля перешагиваний (penny_jump_rate) и медианное время penny_jump_dt (с).
  2. Реальный коэффициент перехвата ленты (capture_ratio).
  3. Markouts через 10s / 60s / 180s и чистый реализованный EV круга (bp и $).
=============================================================================
"""

import json
from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOG = BASE_DIR / "data" / "logs" / "canary_scalp_telemetry.jsonl"


def main():
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    if not p.exists():
        print(f"Файл телеметрии ещё не создан (ожидание первого широкого спреда >=50 bp): {p}")
        return

    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    places = [r for r in rows if r.get("event") == "canary_bid_place"]
    jumps = [r for r in rows if r.get("event") == "penny_jump_detected"]
    bfills = [r for r in rows if r.get("event") == "canary_bid_fill"]
    rounds = [r for r in rows if r.get("event") == "canary_round_complete"]
    mouts = [r for r in rows if r.get("event") == "entry_markouts"]

    print("=" * 76)
    print(f"🐤 ОТЧЁТ КАНАРЕЕЧНОГО ТЕСТА МИКРО-СКАЛЬПА ({p.name}, событий={len(rows)})")
    print("=" * 76)
    print(f"  • Постановок Bid (>=50 bp):   {len(places)}")
    if jumps:
        j_dts = [float(r["penny_jump_dt"]) for r in jumps]
        print(f"  • Перешагиваний чужими ботами: {len(jumps)} ({len(jumps)/max(1,len(places)):.1%}) | медиана dt={np.median(j_dts):.2f}s (p25={np.percentile(j_dts,25):.2f}s)")
    else:
        print("  • Перешагиваний чужими ботами: 0")

    if bfills:
        caps = [float(r["capture_ratio"]) for r in bfills]
        rests = [float(r["rest_sec"]) for r in bfills]
        print(f"  • Исполнений входа (Bid):     {len(bfills)} | медиана ожидания={np.median(rests):.1f}s | capture_ratio={np.mean(caps):.1%} (мед. {np.median(caps):.1%})")

    if mouts:
        m10 = [float(r["markout_10s_bp"]) for r in mouts if r.get("markout_10s_bp") is not None]
        m60 = [float(r["markout_60s_bp"]) for r in mouts if r.get("markout_60s_bp") is not None]
        m180 = [float(r["markout_180s_bp"]) for r in mouts if r.get("markout_180s_bp") is not None]
        print(f"  • Markouts (n={len(mouts)}):         10s={np.mean(m10):+.1f}bp | 60s={np.mean(m60):+.1f}bp | 180s={np.mean(m180):+.1f}bp")

    if rounds:
        bps = [float(r["net_pnl_bp"]) for r in rounds]
        usds = [float(r["net_pnl_usd"]) for r in rounds]
        holds = [float(r["hold_sec"]) for r in rounds]
        win_r = sum(1 for x in bps if x > 0) / len(bps)
        print(
            f"  • Завершённых кругов:         {len(rounds)} | WinRate={win_r:.1%} | "
            f"Mean EV={np.mean(bps):+.2f} bp/круг (p5={np.percentile(bps,5):+.1f} bp) | "
            f"Sum PnL=${np.sum(usds):+.4f} | мед. удержание={np.median(holds):.0f}s"
        )
    else:
        print("  • Завершённых кругов:         0 (ожидание исполнения)")
    print("=" * 76)


if __name__ == "__main__":
    main()
