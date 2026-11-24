#!/usr/bin/env python3
"""
Step 1 of Order-2.5 Pipeline:
Parses the full 119 MB VPS live history (2026-11-10 -> 2026-11-24), aligns all PLACE / CANCEL / FILL
interventions of the 19 past RL generations with the 1s L2 Orderbook & 84D Signal Stack,
computes exact empirical Market-Impact / Penny-Jump / Book-Push / Post-Fill Markout statistics,
and saves the clean intervention dataset + active bot-order timeline for Step 2 (De-contamination) & Step 3 (Order-2.5 Training)!
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path("/home/restorator/crypto_bot")
HIST_LOG = BASE_DIR / "data/vps_live_history/full_live_rl_trader_history.log"
PARQUET_PATH = BASE_DIR / "data/features/prl_1s_features.parquet"
LOB_CACHE_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
OBS_84D_PATH = BASE_DIR / "data/cache/rl_hybrid_market_obs_84d.npz"
OUT_DATASET_PATH = BASE_DIR / "data/cache/order25_live_interventions.npz"

RE_TICK = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) MSK \| #(\d+)\] Bid=([\d.]+) Ask=([\d.]+).*?Equity=\$([\d.]+)"
)
RE_START = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) MSK\] Старт боевого цикла LIVE")
RE_PLACE = re.compile(
    r"\[(?:24-SLOT|ORCH|NEURAL) PLACE (BID|ASK) #(\d+)\] id=(\S+) \| ([\d.]+) PRL @ \$([\d.]+)"
)
RE_CANCEL = re.compile(
    r"\[(?:24-SLOT|ORCH|NEURAL) CANCEL (BID|ASK) #(\d+)[^\]]*\] id=(\S+)"
)
RE_FILL = re.compile(
    r"\[24-SLOT FILL (BID|ASK) #(\d+)\] [+-]([\d.]+) PRL @ \$([\d.]+)(?: \| PnL=([+-]?[\d.]+)\$)?"
)
MSK_TZ = timezone(timedelta(hours=3))


def parse_msk_to_utc_sec(ts_str: str) -> int:
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=MSK_TZ)
    return int(round(dt.timestamp()))


def main():
    t0 = time.time()
    print("=" * 96)
    print("🔬 STEP 1: EXTRACTING 14-DAY LIVE AGENT INTERVENTIONS & MEASURING MARKET REACTION")
    print("=" * 96, flush=True)

    raw_text = HIST_LOG.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()
    print(f"Loaded {len(lines):,} raw log lines ({len(raw_text)/1e6:.2f} MB) in {time.time()-t0:.1f}s")

    # Detect concatenation boundary where .log (Nov 18..24) transitions back to .log.1 (Nov 10..18)
    split_line = None
    prev_sec = 0
    for idx, ln in enumerate(lines):
        m = RE_TICK.search(ln) or RE_START.search(ln)
        if m:
            s_utc = parse_msk_to_utc_sec(m.group(1))
            if prev_sec > 0 and s_utc < prev_sec - 86400 * 2:
                split_line = idx
                break
            prev_sec = s_utc

    if split_line is not None:
        print(f"Reordering rotated log segments at line {split_line:,} -> strict chronological order (Nov 10 -> Nov 24)!")
        lines = lines[split_line:] + lines[:split_line]

    # Pass 1: associate pending events before a tick summary line with that tick's timestamp!
    places = []   # (utc_sec, side_int[0=bid,1=ask], slot, oid, qty, px, b1_tick, a1_tick)
    cancels = []  # (utc_sec, side_int, slot, oid)
    fills = []    # (utc_sec, side_int, slot, qty, px, pnl_usd, b1_tick, a1_tick)
    pending = []
    cur_sec = None
    cur_b1, cur_a1 = 0.0, 0.0
    cycle_resets = []

    for ln in lines:
        ms = RE_START.search(ln)
        if ms:
            s_u = parse_msk_to_utc_sec(ms.group(1))
            cur_sec = s_u
            cycle_resets.append(s_u)
            continue

        mt = RE_TICK.search(ln)
        if mt:
            s_u = parse_msk_to_utc_sec(mt.group(1))
            b1_v = float(mt.group(3))
            a1_v = float(mt.group(4))
            cur_sec, cur_b1, cur_a1 = s_u, b1_v, a1_v
            for ev in pending:
                kind = ev[0]
                if kind == "P":
                    places.append((s_u, ev[1], ev[2], ev[3], ev[4], ev[5], b1_v, a1_v))
                elif kind == "C":
                    cancels.append((s_u, ev[1], ev[2], ev[3]))
                elif kind == "F":
                    fills.append((s_u, ev[1], ev[2], ev[3], ev[4], ev[5], b1_v, a1_v))
            pending.clear()
            continue

        mp = RE_PLACE.search(ln)
        if mp:
            side_i = 0 if mp.group(1) == "BID" else 1
            pending.append(("P", side_i, int(mp.group(2)), mp.group(3), float(mp.group(4)), float(mp.group(5))))
            continue

        mc = RE_CANCEL.search(ln)
        if mc:
            side_i = 0 if mc.group(1) == "BID" else 1
            pending.append(("C", side_i, int(mc.group(2)), mc.group(3)))
            continue

        mf = RE_FILL.search(ln)
        if mf:
            side_i = 0 if mf.group(1) == "BID" else 1
            pnl_v = float(mf.group(5)) if mf.group(5) is not None else 0.0
            pending.append(("F", side_i, int(mf.group(2)), float(mf.group(3)), float(mf.group(4)), pnl_v))
            continue

    print(f"Extracted Live Events: PLACES={len(places):,} | CANCELS={len(cancels):,} | FILLS={len(fills):,}")

    # Load LOB 600h cache (contains exact int64 UTC timestamps 'timestamps' / 'ts') to align to row indices [0 .. N-1]
    lob = np.load(LOB_CACHE_PATH, mmap_mode="r")
    if "timestamps" in lob.files:
        ts_arr = np.asarray(lob["timestamps"], dtype=np.int64)
    elif "ts" in lob.files:
        ts_arr = np.asarray(lob["ts"], dtype=np.int64)
    else:
        pf = pq.ParquetFile(str(PARQUET_PATH))
        df_ts = pf.read(columns=[]).to_pandas()
        ts_arr = pd.to_datetime(df_ts.index, utc=True).as_unit("s").astype("int64").values
    N = len(ts_arr)
    bp10 = np.asarray(lob["bid_p"][:, :10], dtype=np.float32)
    ap10 = np.asarray(lob["ask_p"][:, :10], dtype=np.float32)
    bq10 = np.asarray(lob["bid_q"][:, :10], dtype=np.float32)
    aq10 = np.asarray(lob["ask_q"][:, :10], dtype=np.float32)
    b1_all = bp10[:, 0]
    a1_all = ap10[:, 0]
    subcent_off = int(np.where(np.abs(b1_all * 100.0 - np.round(b1_all * 100.0)) > 1e-3)[0][0])
    subcent_start_ts = int(ts_arr[subcent_off])
    print(f"Dataset N={N:,} rows | Sub-cent ($0.0001) starts at idx={subcent_off:,} ({datetime.fromtimestamp(subcent_start_ts, tz=timezone.utc)})")

    obs84 = np.load(OBS_84D_PATH, mmap_mode="r")["market_obs_84d"]

    # Map UTC sec -> exact dataset index via searchsorted (accepting match within <= 2s)
    def map_sec_to_idx(sec_u: int) -> int:
        pos = int(np.searchsorted(ts_arr, sec_u))
        if pos < N and abs(int(ts_arr[pos]) - sec_u) <= 2:
            return pos
        if pos > 0 and abs(int(ts_arr[pos - 1]) - sec_u) <= 2:
            return pos - 1
        return -1

    # Reconstruct exact order lifetimes (place_idx -> end_idx via cancel, slot overwrite, or fill)
    # Track active order per (side, slot) & oid -> index in places_clean
    places_records = []
    oid_to_rec = {}
    slot_active_rec = {}

    for s_u, side_i, slot, oid, qty, px, b1_t, a1_t in places:
        idx = map_sec_to_idx(s_u)
        if idx < 100 or idx >= N - 185:
            continue
        # If previous order in same slot was still open, close it at idx
        prev_r = slot_active_rec.get((side_i, slot))
        if prev_r is not None and prev_r["end_idx"] is None:
            prev_r["end_idx"] = max(prev_r["start_idx"] + 1, idx)
        rec = {
            "start_idx": idx,
            "end_idx": None,
            "filled": 0,
            "side": side_i,
            "slot": slot,
            "oid": oid,
            "qty": qty,
            "px": px,
            "usd": qty * px,
            "b1_log": b1_t,
            "a1_log": a1_t,
        }
        places_records.append(rec)
        oid_to_rec[oid] = rec
        slot_active_rec[(side_i, slot)] = rec

    for s_u, side_i, slot, oid in cancels:
        idx = map_sec_to_idx(s_u)
        if oid in oid_to_rec and oid_to_rec[oid]["end_idx"] is None:
            r = oid_to_rec[oid]
            r["end_idx"] = max(r["start_idx"] + 1, idx if idx > 0 else r["start_idx"] + 5)

    fill_records = []
    for s_u, side_i, slot, qty, px, pnl_u, b1_t, a1_t in fills:
        idx = map_sec_to_idx(s_u)
        if idx < 100 or idx >= N - 185:
            continue
        fill_records.append((idx, side_i, slot, qty, px, qty * px, pnl_u, b1_t, a1_t))
        prev_r = slot_active_rec.get((side_i, slot))
        if prev_r is not None:
            prev_r["filled"] = 1
            prev_r["rem_qty"] = max(0.0, prev_r.get("rem_qty", prev_r["qty"]) - qty)
            if prev_r["rem_qty"] <= 1.5 and prev_r["end_idx"] is None:
                prev_r["end_idx"] = max(prev_r["start_idx"] + 1, idx)

    # Cap any dangling open orders (e.g. across service crash) to max 600s if another reset happened
    reset_idxs = sorted([i for i in (map_sec_to_idx(s) for s in cycle_resets) if i >= 0])
    for r in places_records:
        if r["end_idx"] is None:
            pos_r = int(np.searchsorted(reset_idxs, r["start_idx"] + 1))
            if pos_r < len(reset_idxs):
                r["end_idx"] = min(reset_idxs[pos_r], r["start_idx"] + 1800)
            else:
                r["end_idx"] = min(N - 1, r["start_idx"] + 600)

    print(f"Aligned Valid Interventions in Dataset: PLACES={len(places_records):,} | FILLS={len(fill_records):,}")

    # =========================================================================
    # AUDIT 1: PENNY-JUMPING (FRONT-RUNNING) & BOOK-PUSH ON LIVE PLACES
    # =========================================================================
    # Focus on near-touch Maker placements (dist <= 60 bp from BBO) in sub-cent era (and overall)
    pj_rows = []
    X_place_list = []
    Y_place_list = []  # [penny_jump_5s, book_push_away_10s, realized_fill_30s]
    place_idx_list = []

    for r in places_records:
        i0 = r["start_idx"]
        # Use pre-action snapshot i_pre = i0 - 1 so our own newly placed order is NOT yet in the book!
        ipre = i0 - 1
        b1_0 = float(b1_all[ipre])
        a1_0 = float(a1_all[ipre])
        if b1_0 <= 0.01 or a1_0 <= b1_0:
            continue
        tick = 0.0001 if i0 >= subcent_off else 0.01
        px = r["px"]
        usd = r["usd"]
        side = r["side"]
        sp_bp = (a1_0 - b1_0) / b1_0 * 10000.0
        l1_usd = max(20.0, float(bq10[ipre, 0] * b1_0) if side == 0 else float(aq10[ipre, 0] * a1_0))
        top5_usd = max(50.0, float(np.sum(bq10[ipre, :5] * bp10[ipre, :5])) if side == 0 else float(np.sum(aq10[ipre, :5] * ap10[ipre, :5])))

        if side == 0:  # BID
            dist_bp = (b1_0 - px) / b1_0 * 10000.0  # negative = inside spread!
            # Penny-jump within 5s: best bid moves strictly ABOVE our placed price px + 0.5*tick while Ask didn't get lifted first, OR someone steps ahead of us!
            b1_max_5s = float(np.max(b1_all[i0 + 1 : i0 + 6]))
            a1_min_5s = float(np.min(a1_all[i0 + 1 : i0 + 6]))
            penny_5s = 1.0 if (px >= b1_0 - 25.0 * tick and b1_max_5s > px + 0.5 * tick and a1_min_5s > px + 0.5 * tick) else 0.0
            # Book push away within 10s: Ask1 runs upward >= +8 bp without filling our Bid!
            a1_10s = float(np.max(a1_all[i0 + 1 : i0 + 11]))
            b1_min_10s = float(np.min(b1_all[i0 + 1 : i0 + 11]))
            push_10s = 1.0 if (b1_min_10s > px + 0.2 * tick and (a1_10s - a1_0) / a1_0 * 10000.0 >= 8.0) else 0.0
        else:  # ASK
            dist_bp = (px - a1_0) / a1_0 * 10000.0  # negative = inside spread!
            a1_min_5s = float(np.min(a1_all[i0 + 1 : i0 + 6]))
            b1_max_5s = float(np.max(b1_all[i0 + 1 : i0 + 6]))
            penny_5s = 1.0 if (px <= a1_0 + 25.0 * tick and a1_min_5s < px - 0.5 * tick and b1_max_5s < px - 0.5 * tick) else 0.0
            b1_10s = float(np.min(b1_all[i0 + 1 : i0 + 11]))
            a1_max_10s = float(np.max(a1_all[i0 + 1 : i0 + 11]))
            push_10s = 1.0 if (a1_max_10s < px - 0.2 * tick and (b1_0 - b1_10s) / b1_0 * 10000.0 >= 8.0) else 0.0

        is_subcent = 1 if i0 >= subcent_off else 0
        pj_rows.append((is_subcent, side, dist_bp, usd, usd / l1_usd, sp_bp, penny_5s, push_10s, float(r["filled"])))

        if -50.0 <= dist_bp <= 250.0:
            act_feat = np.array([
                float(side),
                np.clip(dist_bp / 50.0, -1.0, 5.0),
                np.log1p(usd) / 6.0,
                np.clip(usd / l1_usd, 0.0, 10.0) / 3.0,
                np.clip(usd / top5_usd, 0.0, 5.0),
                np.clip(sp_bp / 30.0, 0.0, 4.0),
            ], dtype=np.float32)
            X_place_list.append(np.concatenate([obs84[ipre], act_feat]))
            Y_place_list.append([penny_5s, push_10s, float(r["filled"])])
            place_idx_list.append(i0)

    df_pj = pd.DataFrame(pj_rows, columns=["subcent", "side", "dist_bp", "usd", "l1_ratio", "sp_bp", "penny_5s", "push_10s", "filled"])
    near_df = df_pj[(df_pj["subcent"] == 1) & (df_pj["dist_bp"] <= 25.0)]
    print(f"\n📊 [AUDIT 1A] SUB-CENT ($0.0001) NEAR-TOUCH ORDERS (dist <= 25 bp, N={len(near_df):,}):")
    for lo, hi, lbl in [(0, 35, "< $35 (Small)"), (35, 90, "$35..$90 (Med)"), (90, 220, "$90..$220 (Large)"), (220, 10000, "> $220 (Whale Wall)")]:
        sub = near_df[(near_df["usd"] >= lo) & (near_df["usd"] < hi)]
        if len(sub) > 0:
            print(
                f"   Size {lbl:<20} (N={len(sub):5d}): Penny-Jumped(5s) = {sub['penny_5s'].mean()*100:5.1f}% | "
                f"Price-Pushed-Away(10s) = {sub['push_10s'].mean()*100:5.1f}% | Actual Fill Rate = {sub['filled'].mean()*100:5.1f}%"
            )

    print(f"\n📊 [AUDIT 1B] BY DEPTH ZONE (Sub-Cent Era, All Sizes):")
    for dlo, dhi, dlbl in [(-60, -0.5, "Inside Spread (<0 bp)"), (-0.5, 2.0, "At BBO (0..2 bp)"), (2.0, 25.0, "Near Book (2..25 bp)"), (25.0, 150.0, "Deep Cushion (25..150 bp)")]:
        sub = df_pj[(df_pj["subcent"] == 1) & (df_pj["dist_bp"] >= dlo) & (df_pj["dist_bp"] < dhi)]
        if len(sub) > 0:
            print(
                f"   Zone {dlbl:<26} (N={len(sub):5d}): Penny-Jumped(5s) = {sub['penny_5s'].mean()*100:5.1f}% | "
                f"Pushed-Away(10s) = {sub['push_10s'].mean()*100:5.1f}% | Fill Rate = {sub['filled'].mean()*100:5.1f}%"
            )

    # =========================================================================
    # AUDIT 2: REAL LIVE FILLS — POST-FILL MARKOUT & LIVE TRAP REGRET
    # =========================================================================
    X_fill_list = []
    Y_fill_list = []  # [markout_10s_bp, markout_30s_bp, markout_180s_bp, is_toxic_trap_180s]
    fill_idx_list = []
    mk_rows = []

    for idx, side, slot, qty, px, usd, pnl_u, b1_t, a1_t in fill_records:
        ipre = idx - 1
        mid_0 = 0.5 * float(b1_all[idx] + a1_all[idx])
        mid_10 = 0.5 * float(b1_all[idx + 10] + a1_all[idx + 10])
        mid_30 = 0.5 * float(b1_all[idx + 30] + a1_all[idx + 30])
        mid_180 = 0.5 * float(b1_all[idx + 180] + a1_all[idx + 180])
        if px <= 0.01:
            continue
        sgn = 1.0 if side == 0 else -1.0  # +1 for Buy, -1 for Sell
        # Gross markout relative to actual execution price px, minus 10 bp maker fee!
        mk10 = sgn * ((mid_10 - px) / px * 10000.0) - 10.0
        mk30 = sgn * ((mid_30 - px) / px * 10000.0) - 10.0
        mk180 = sgn * ((mid_180 - px) / px * 10000.0) - 10.0
        # Live Toxic Trap: 3m net markout <= -35 bp (adversely run over in live!)
        is_trap = 1.0 if mk180 <= -35.0 else 0.0
        mk_rows.append((1 if idx >= subcent_off else 0, side, usd, mk10, mk30, mk180, is_trap))

        f_feat = np.array([float(side), np.log1p(usd) / 6.0, float(slot) / 11.0], dtype=np.float32)
        X_fill_list.append(np.concatenate([obs84[ipre], f_feat]))
        Y_fill_list.append([mk10, mk30, mk180, is_trap])
        fill_idx_list.append(idx)

    df_mk = pd.DataFrame(mk_rows, columns=["subcent", "side", "usd", "mk10", "mk30", "mk180", "trap"])
    print(f"\n📊 [AUDIT 2] REAL LIVE FILLS POST-TRADE MARKOUT (Net of 10bp Maker Fee, N={len(df_mk):,}):")
    for s_val, s_nm in [(0, "LIVE BUY FILLS "), (1, "LIVE SELL FILLS")]:
        sub = df_mk[df_mk["side"] == s_val]
        print(
            f"   {s_nm} (N={len(sub):4d}): Net Markout 10s={sub['mk10'].mean():+6.1f} bp | "
            f"30s={sub['mk30'].mean():+6.1f} bp | 180s={sub['mk180'].mean():+6.1f} bp | Toxic Trap(<-35bp @ 3m)={sub['trap'].mean()*100:4.1f}%"
        )

    # =========================================================================
    # AUDIT 3: SELF-ORDER CONTAMINATION OF HISTORICAL SAFETRADE ORDERBOOK
    # =========================================================================
    # Measure on how many seconds in sub-cent era our own bot orders sat inside Top-10 SafeTrade levels!
    bot_bid_usd_1s = np.zeros(N, dtype=np.float32)
    bot_ask_usd_1s = np.zeros(N, dtype=np.float32)
    events_span = np.zeros((len(places_records), 5), dtype=np.float32)  # [start_idx, end_idx, side, px, qty]
    for k_i, r in enumerate(places_records):
        s_i, e_i = int(r["start_idx"]), int(r["end_idx"])
        events_span[k_i] = [s_i, e_i, r["side"], r["px"], r["qty"]]
        if r["side"] == 0:
            bot_bid_usd_1s[s_i:e_i] += r["usd"]
        else:
            bot_ask_usd_1s[s_i:e_i] += r["usd"]

    sub_b_usd = bot_bid_usd_1s[subcent_off:]
    sub_a_usd = bot_ask_usd_1s[subcent_off:]
    st_top5_b_usd = np.sum(bp10[subcent_off:, :5] * bq10[subcent_off:, :5], axis=1) + 1.0
    active_bot_mask = (sub_b_usd + sub_a_usd) > 10.0
    share_when_active = np.minimum(1.0, sub_b_usd[active_bot_mask] / st_top5_b_usd[active_bot_mask]) if np.any(active_bot_mask) else np.array([0.0])

    print(f"\n📊 [AUDIT 3] HISTORICAL BOOK SELF-CONTAMINATION (Sub-Cent Era = {len(sub_b_usd)/3600:.1f}h):")
    print(f"   Seconds w/ active past-bot orders in SafeTrade book: {np.sum(active_bot_mask):,}s ({np.mean(active_bot_mask)*100:.1f}% of sub-cent dataset!)")
    print(f"   Mean past-bot Bid USD in book when active: ${sub_b_usd[active_bot_mask].mean():,.1f} (p90=${np.percentile(sub_b_usd[active_bot_mask], 90):,.1f})")
    print(f"   Past-bot share of SafeTrade Top-5 Bid Book when active: mean={share_when_active.mean()*100:.1f}% | p90={np.percentile(share_when_active, 90)*100:.1f}%!")

    np.savez_compressed(
        OUT_DATASET_PATH,
        X_place=np.asarray(X_place_list, dtype=np.float32),
        Y_place=np.asarray(Y_place_list, dtype=np.float32),
        place_idx=np.asarray(place_idx_list, dtype=np.int64),
        X_fill=np.asarray(X_fill_list, dtype=np.float32),
        Y_fill=np.asarray(Y_fill_list, dtype=np.float32),
        fill_idx=np.asarray(fill_idx_list, dtype=np.int64),
        events_span=events_span,
    )
    print(f"\n✅ Saved Order-2.5 intervention dataset -> {OUT_DATASET_PATH} (X_place={len(X_place_list):,}, X_fill={len(X_fill_list):,}) in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
