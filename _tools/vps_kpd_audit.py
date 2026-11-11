#!/usr/bin/env python3
"""
_tools/vps_kpd_audit.py
=============================================================================
Инженерно-экономический аудит КПД (Efficiency, η_MM ∈ [0..100%]) боевого
RL маркет-мейкера на VPS (субаккаунт ...6C9).

Вычисляет 4 фундаментальных контура КПД без единого искусственного костыля:
  1. η_spread (Edge Capture): захват чистого спреда на прокрученный объём
     относительно теоретического мейкер-потолка PRL (~60 bp на полный круг).
  2. η_shield (Toxicity & Drawdown Shield): удержание минимальной просадки
     относительно амплитуды рынка + доля времени без подставленных бидов в TOX.
  3. η_work   (Safe-Regime Duty Cycle): доля времени работы сетки ордеров
     в безопасных рыночных режимах (FLAT / ACC / BRK).
  4. η_queue  (FIFO Queue Retention): отношение реальных исполнений (Fills)
     к числу отмен (Cancels), вскрывающее потерю приоритета очереди.

Может вызываться как самостоятельный CLI:
    /home/restorator/crypto_bot/.venv/bin/python _tools/vps_kpd_audit.py
так и напрямую из _tools/deploy_vps.py через print_vps_kpd_summary(ssh).
=============================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.vps_config import create_ssh_client

REMOTE_AUDIT_PY = r'''
import json, re
from pathlib import Path

root = Path("/opt/crypto_bot")
log_files = []
if (root / "data/logs/live_rl_trader.log.1").exists():
    log_files.append(root / "data/logs/live_rl_trader.log.1")
if (root / "data/logs/live_rl_trader.log").exists():
    log_files.append(root / "data/logs/live_rl_trader.log")

re_tick = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) MSK \| #(\d+)\] "
    r"Bid=([\d.]+) Ask=([\d.]+) \| Act=(\S+)\s*\| Slots=\[B:(\d+)/12 A:(\d+)/12\] \| "
    r"Reg=(\S+) .*?Equity=\$([\d.]+) \(([+-][\d.]+)\$\) \| ([\d.]+) ms"
)
re_fill_b = re.compile(r"\[24-SLOT FILL BID #(\d+)\] \+([\d.]+) PRL @ \$([\d.]+)")
re_fill_a = re.compile(r"\[24-SLOT FILL ASK #(\d+)\] -([\d.]+) PRL @ \$([\d.]+) \| PnL=([+-][\d.]+)\$")
re_place_b = re.compile(r"\[24-SLOT PLACE BID #(\d+)\]")
re_place_a = re.compile(r"\[24-SLOT PLACE ASK #(\d+)\]")
re_cancel_b = re.compile(r"\[24-SLOT CANCEL BID #(\d+)")
re_cancel_a = re.compile(r"\[24-SLOT CANCEL ASK #(\d+)")
re_champ = re.compile(r"(Подключен боевой 24-слотовый Чемпион|HOT-RELOAD CHAMPION).*?\(([^)]+)\)")
re_sub = re.compile(r"Изолированный субсчет бота: кэш=\$([\d.]+) USDT, поз=([\d.]+) PRL")

def new_seg(name="unknown"):
    return {
        "champ": name,
        "ticks": 0,
        "first_ts": None,
        "last_ts": None,
        "b1_first": None,
        "b1_last": None,
        "b1_min": 999999.0,
        "b1_max": 0.0,
        "eq_first": None,
        "eq_last": None,
        "eq_min": 1e12,
        "eq_max": 0.0,
        "peak_eq": 0.0,
        "max_dd_pct": 0.0,
        "ticks_non_tox": 0,
        "ticks_work_non_tox": 0,
        "ticks_tox": 0,
        "ticks_tox_no_bid": 0,
        "place_b": 0,
        "place_a": 0,
        "cancel_b": 0,
        "cancel_a": 0,
        "fill_b_cnt": 0,
        "fill_b_prl": 0.0,
        "fill_b_usd": 0.0,
        "fill_a_cnt": 0,
        "fill_a_prl": 0.0,
        "fill_a_usd": 0.0,
        "realized_pnl": 0.0,
        "http_429": 0,
        "lat_ms_sum": 0.0,
        "lat_ms_max": 0.0,
    }

segments = []
cur = new_seg()

for lf in log_files:
    with open(lf, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mc = re_champ.search(line)
            if mc:
                if cur["ticks"] > 0 or cur["fill_b_cnt"] > 0 or cur["fill_a_cnt"] > 0:
                    segments.append(cur)
                cur = new_seg(mc.group(2).split(",")[0].strip())
                continue
            if "429" in line and "WARN" in line:
                cur["http_429"] += 1
            if re_place_b.search(line):
                cur["place_b"] += 1
            elif re_place_a.search(line):
                cur["place_a"] += 1
            elif re_cancel_b.search(line):
                cur["cancel_b"] += 1
            elif re_cancel_a.search(line):
                cur["cancel_a"] += 1
            else:
                mfb = re_fill_b.search(line)
                if mfb:
                    q, px = float(mfb.group(2)), float(mfb.group(3))
                    cur["fill_b_cnt"] += 1
                    cur["fill_b_prl"] += q
                    cur["fill_b_usd"] += q * px
                    continue
                mfa = re_fill_a.search(line)
                if mfa:
                    q, px, pnl = float(mfa.group(2)), float(mfa.group(3)), float(mfa.group(4))
                    cur["fill_a_cnt"] += 1
                    cur["fill_a_prl"] += q
                    cur["fill_a_usd"] += q * px
                    cur["realized_pnl"] += pnl
                    continue
                mt = re_tick.search(line)
                if mt:
                    ts_s = mt.group(1)
                    b1 = float(mt.group(3))
                    nb, na = int(mt.group(6)), int(mt.group(7))
                    reg = mt.group(8)
                    eq = float(mt.group(9))
                    lat = float(mt.group(11))
                    cur["ticks"] += 1
                    if cur["first_ts"] is None:
                        cur["first_ts"] = ts_s
                        cur["b1_first"] = b1
                        cur["eq_first"] = eq
                    cur["last_ts"] = ts_s
                    cur["b1_last"] = b1
                    cur["eq_last"] = eq
                    cur["b1_min"] = min(cur["b1_min"], b1)
                    cur["b1_max"] = max(cur["b1_max"], b1)
                    cur["eq_min"] = min(cur["eq_min"], eq)
                    cur["eq_max"] = max(cur["eq_max"], eq)
                    cur["peak_eq"] = max(cur["peak_eq"], eq)
                    if cur["peak_eq"] > 0:
                        dd = (cur["peak_eq"] - eq) / cur["peak_eq"] * 100.0
                        if dd > cur["max_dd_pct"]:
                            cur["max_dd_pct"] = dd
                    if reg == "TOX":
                        cur["ticks_tox"] += 1
                        if nb == 0:
                            cur["ticks_tox_no_bid"] += 1
                    else:
                        cur["ticks_non_tox"] += 1
                        if nb > 0 or na > 0:
                            cur["ticks_work_non_tox"] += 1
                    cur["lat_ms_sum"] += lat
                    cur["lat_ms_max"] = max(cur["lat_ms_max"], lat)

if cur["ticks"] > 0 or cur["fill_b_cnt"] > 0 or cur["fill_a_cnt"] > 0:
    segments.append(cur)

# Выделяем эру изолированного субаккаунта ...6C9 (Equity < $2000 после последнего переключения счёта)
clean_idx = 0
for i, s in enumerate(segments):
    if s["eq_first"] is not None and s["eq_first"] > 2000.0:
        clean_idx = i + 1
sub_segs = [s for s in segments[clean_idx:] if s["eq_first"] is not None and s["eq_first"] < 2000.0]
# Пропускаем переходный сегмент первичной разгрузки, если L18_0065_pruned16 и новее идут следом
for j, s in enumerate(sub_segs):
    if s["champ"].startswith("L18_0065_pruned") or s["champ"].startswith("L19_") or s["champ"].startswith("L20_") or s["champ"].startswith("L21_"):
        sub_segs = sub_segs[j:]
        break

print(json.dumps({"subaccount_segments": sub_segs}, ensure_ascii=False))
'''


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_kpd_metrics(segs: list[dict]) -> dict:
    if not segs:
        return {}
    eq_start = float(segs[0]["eq_first"])
    eq_end = float(segs[-1]["eq_last"])
    net_eq_delta = eq_end - eq_start
    net_roi_pct = (net_eq_delta / max(eq_start, 1.0)) * 100.0

    b1_min = min(float(s["b1_min"]) for s in segs if s["b1_min"] < 9999.0)
    b1_max = max(float(s["b1_max"]) for s in segs)
    mkt_swing_pct = ((b1_max - b1_min) / max(b1_min, 0.01)) * 100.0

    peak_eq = 0.0
    max_dd_pct = 0.0
    for s in segs:
        peak_eq = max(peak_eq, float(s["eq_max"]))
        if peak_eq > 0.0:
            dd_seg = (peak_eq - float(s["eq_min"])) / peak_eq * 100.0
            max_dd_pct = max(max_dd_pct, dd_seg, float(s.get("max_dd_pct", 0.0)))

    buy_usd = sum(float(s["fill_b_usd"]) for s in segs)
    sell_usd = sum(float(s["fill_a_usd"]) for s in segs)
    round_trip_usd = 0.5 * (buy_usd + sell_usd)

    fills_tot = sum(int(s["fill_b_cnt"]) + int(s["fill_a_cnt"]) for s in segs)
    cancels_tot = sum(int(s["cancel_b"]) + int(s["cancel_a"]) for s in segs)
    places_tot = sum(int(s["place_b"]) + int(s["place_a"]) for s in segs)

    t_non_tox = sum(int(s["ticks_non_tox"]) for s in segs)
    t_work = sum(int(s["ticks_work_non_tox"]) for s in segs)
    t_tox = sum(int(s["ticks_tox"]) for s in segs)
    t_tox_prot = sum(int(s["ticks_tox_no_bid"]) for s in segs)

    ticks_tot = sum(int(s["ticks"]) for s in segs)
    lat_avg = sum(float(s["lat_ms_sum"]) for s in segs) / max(ticks_tot, 1)
    http_429 = sum(int(s["http_429"]) for s in segs)

    # 1. η_spread: захват спреда относительно теоретического чистого потолка 60 bp (0.60%) на полный круг
    if round_trip_usd >= 25.0:
        realized_edge_bp = (net_eq_delta / round_trip_usd) * 10000.0
        eta_spread = _clip(realized_edge_bp / 60.0)
    else:
        realized_edge_bp = 0.0
        eta_spread = 0.0 if ticks_tot >= 300 else (0.50 if abs(net_eq_delta) < 0.05 else (1.0 if net_eq_delta > 0 else 0.0))

    # 2. η_shield: защита от просадки и токсичного потока
    dd_protection = _clip(1.0 - max_dd_pct / max(mkt_swing_pct, 1.5))
    tox_avoidance = (t_tox_prot / t_tox) if t_tox > 0 else 1.0
    eta_shield = _clip(0.55 * dd_protection + 0.45 * tox_avoidance)

    # 3. η_work: коэффициент полезной занятости сетки в безопасном рынке
    eta_work = _clip(t_work / max(t_non_tox, 1))

    # 4. η_queue: КПД удержания FIFO-очереди (эталон 6-слотовой сетки: >= 1 fill на 15 снятий = 100%)
    eta_queue = _clip((fills_tot * 15.0) / max(cancels_tot + fills_tot, 1))

    # Интегральный КПД маркет-мейкера (взвешенное среднее 4 контуров)
    eta_total = 0.30 * eta_spread + 0.25 * eta_shield + 0.25 * eta_work + 0.20 * eta_queue

    return {
        "first_ts": segs[0]["first_ts"],
        "last_ts": segs[-1]["last_ts"],
        "eq_start": eq_start,
        "eq_end": eq_end,
        "net_eq_delta": net_eq_delta,
        "net_roi_pct": net_roi_pct,
        "max_dd_pct": max_dd_pct,
        "b1_min": b1_min,
        "b1_max": b1_max,
        "mkt_swing_pct": mkt_swing_pct,
        "round_trip_usd": round_trip_usd,
        "realized_edge_bp": realized_edge_bp,
        "fills_tot": fills_tot,
        "places_tot": places_tot,
        "cancels_tot": cancels_tot,
        "cancel_to_fill": cancels_tot / max(fills_tot, 1),
        "http_429": http_429,
        "lat_avg_ms": lat_avg,
        "eta_spread": eta_spread * 100.0,
        "eta_shield": eta_shield * 100.0,
        "eta_work": eta_work * 100.0,
        "eta_queue": eta_queue * 100.0,
        "eta_total": eta_total * 100.0,
    }


def fetch_vps_segments(ssh) -> list[dict]:
    stdin, stdout, _ = ssh.exec_command("python3 -")
    stdin.write(REMOTE_AUDIT_PY)
    stdin.channel.shutdown_write()
    raw = stdout.read().decode("utf-8", errors="ignore").strip()
    if not raw:
        return []
    data = json.loads(raw[raw.index("{") :])
    return data.get("subaccount_segments", [])


def format_kpd_report(segs: list[dict]) -> str:
    if not segs:
        return "⚠️ [KPD AUDIT] Нет данных сегментов в логе VPS."
    ov = compute_kpd_metrics(segs)
    cur_champ_name = segs[-1]["champ"]
    cur_champ_segs = [s for s in segs if s["champ"] == cur_champ_name]
    cur_m = compute_kpd_metrics(cur_champ_segs)

    # Группируем по чемпионам для наглядной динамики поколений
    grouped: dict[str, list[dict]] = {}
    for s in segs:
        grouped.setdefault(s["champ"], []).append(s)

    lines = [
        "=== ⚡ АУДИТ КПД РОБОТА НА VPS (субаккаунт ...6C9) ===",
        (
            f"📊 Сводка эпохи ({ov['first_ts'][:16]} -> {ov['last_ts'][11:19]} MSK): "
            f"Equity ${ov['eq_start']:.2f} -> ${ov['eq_end']:.2f} ({ov['net_eq_delta']:+.2f}$ / {ov['net_roi_pct']:+.2f}%) | "
            f"MaxDD={ov['max_dd_pct']:.2f}% (рынок ${ov['b1_min']:.2f}..${ov['b1_max']:.2f}, размах {ov['mkt_swing_pct']:.1f}%) | "
            f"Оборот=${ov['round_trip_usd']:,.0f}"
        ),
        (
            f"🏆 ИНТЕГРАЛЬНЫЙ КПД (η_MM): {ov['eta_total']:.1f}%  "
            f"[η_spread={ov['eta_spread']:.0f}% ({ov['realized_edge_bp']:+.1f}bp/круг) | "
            f"η_shield={ov['eta_shield']:.0f}% | η_work={ov['eta_work']:.0f}% | "
            f"η_queue={ov['eta_queue']:.0f}% (Cancel/Fill={ov['cancel_to_fill']:.1f}:1)]"
        ),
        (
            f"🎯 Текущий активный чемпион [{cur_champ_name}]: "
            f"КПД={cur_m['eta_total']:.1f}% [η_work={cur_m['eta_work']:.0f}% | η_shield={cur_m['eta_shield']:.0f}% | "
            f"η_queue={cur_m['eta_queue']:.0f}% (Plc={cur_m['places_tot']}/Cnl={cur_m['cancels_tot']}/Fill={cur_m['fills_tot']})] | "
            f"429={cur_m['http_429']} | AvgLat={cur_m['lat_avg_ms']:.0f}ms"
        ),
        "📈 Сравнение КПД по чемпионам на VPS:",
    ]
    for ch_name, ch_list in grouped.items():
        m = compute_kpd_metrics(ch_list)
        lines.append(
            f"  • {ch_name:34s} | КПД={m['eta_total']:5.1f}% "
            f"[Spr:{m['eta_spread']:3.0f}% Shld:{m['eta_shield']:3.0f}% Wrk:{m['eta_work']:3.0f}% Que:{m['eta_queue']:3.0f}%] | "
            f"ΔEq={m['net_eq_delta']:+6.2f}$ | Fills={m['fills_tot']:3d} | Cnl={m['cancels_tot']:4d} | 429={m['http_429']:2d}"
        )
    return "\n".join(lines)


def print_vps_kpd_summary(ssh) -> None:
    try:
        segs = fetch_vps_segments(ssh)
        print(format_kpd_report(segs))
    except Exception as e_kpd:
        print(f"⚠️ [VPS KPD AUDIT WARN]: {e_kpd}")


def main():
    ssh = create_ssh_client(timeout=15.0)
    try:
        print_vps_kpd_summary(ssh)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
