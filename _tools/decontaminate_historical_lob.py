#!/usr/bin/env python3
"""
Step 2 of Order-2.5 Pipeline: Historical Self-Order De-Contamination!
Subtracts the exact active limit orders of all 19 past live RL generations (40,547 order spans)
from SafeTrade's 10-level historical orderbook (`rl_guard_lob_cache_600h.npz`) and updates
the SafeTrade LOB channels `z[0..8]` in `rl_hybrid_market_obs_84d.npz` so that:
  1) Sim FIFO queues no longer force new agents to queue behind $800 of our own historical bot orders,
  2) Sim LOB imbalance features `z[0..8]` match Live's self-stripped `last_clean_bp_10 / bq_10` 1-to-1!
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/restorator/crypto_bot")
LOB_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
LOB_RAW_BAK = BASE_DIR / "data/cache/rl_guard_lob_cache_600h_raw_with_self.npz"
OBS84_PATH = BASE_DIR / "data/cache/rl_hybrid_market_obs_84d.npz"
OBS84_RAW_BAK = BASE_DIR / "data/cache/rl_hybrid_market_obs_84d_raw_with_self.npz"
INT_PATH = BASE_DIR / "data/cache/order25_live_interventions.npz"


def compact_levels_chunk(p_arr: np.ndarray, q_arr: np.ndarray, is_bid: bool) -> tuple[np.ndarray, np.ndarray]:
    """Shifts remaining positive-qty levels left when our own order was the entire L1/L2 level (1-to-1 with live_rl_trader)."""
    # Vectorized stable compaction across 10 columns
    N_c, K = p_arr.shape
    valid = q_arr > 0.05
    all_first_valid = valid[:, 0]
    # Only rows where at least one level became empty and a deeper level exists need compaction
    needs_fix = ~np.all(valid, axis=1)
    idxs = np.where(needs_fix)[0]
    if len(idxs) == 0:
        return p_arr, q_arr

    p_out = p_arr.copy()
    q_out = q_arr.copy()
    sub_p = p_out[idxs]
    sub_q = q_out[idxs]
    sub_v = valid[idxs]

    # Sort valid=True before valid=False preserving level order via argsort of (~sub_v) with kind='mergesort'
    order = np.argsort(~sub_v, axis=1, kind="mergesort")
    row_i = np.arange(len(idxs))[:, None]
    p_comp = sub_p[row_i, order]
    q_comp = np.where(sub_v[row_i, order], sub_q[row_i, order], 0.0)

    # Fill trailing empty price slots monotonically from last valid price (or original fallback if all 10 levels were ours)
    any_v = np.any(sub_v, axis=1)
    for col in range(1, K):
        empty_c = q_comp[:, col] <= 0.05
        step = -0.0001 if is_bid else +0.0001
        p_comp[empty_c, col] = np.maximum(0.0001, p_comp[empty_c, col - 1] + step)
    # If all 10 levels were somehow empty on a row, keep original L1 price with tiny 1.0 qty fallback
    no_v = ~any_v
    if np.any(no_v):
        p_comp[no_v] = sub_p[no_v]
        q_comp[no_v, 0] = 1.0

    p_out[idxs] = p_comp
    q_out[idxs] = q_comp
    return p_out, q_out


def main():
    t0 = time.time()
    print("=" * 96)
    print("🧹 STEP 2: HISTORICAL ORDERBOOK SELF-DECONTAMINATION (MATCHING LIVE SELF-STRIPPING)")
    print("=" * 96, flush=True)

    if not LOB_RAW_BAK.exists():
        shutil.copy2(LOB_PATH, LOB_RAW_BAK)
        print(f"Backed up raw LOB cache -> {LOB_RAW_BAK.name}")
    if not OBS84_RAW_BAK.exists():
        shutil.copy2(OBS84_PATH, OBS84_RAW_BAK)
        print(f"Backed up raw 84D cache -> {OBS84_RAW_BAK.name}")

    # Always read from the clean untouched raw backup so operation is 100% idempotent!
    raw_lob = np.load(LOB_RAW_BAK)
    lob_dict = {k: raw_lob[k].copy() for k in raw_lob.files}

    bp = lob_dict["bid_p"]
    bq = lob_dict["bid_q"]
    ap = lob_dict["ask_p"]
    aq = lob_dict["ask_q"]
    N = len(bp)

    subcent_off = int(np.where(np.abs(bp[:, 0] * 100.0 - np.round(bp[:, 0] * 100.0)) > 1e-3)[0][0])
    spans = np.load(INT_PATH)["events_span"]  # [M, 5]: start_idx, end_idx, side, px, qty

    min_active_idx = int(spans[:, 0].min())
    max_active_idx = int(spans[:, 1].max())
    print(f"Applying {len(spans):,} past-bot order spans across rows [{min_active_idx:,} .. {max_active_idx:,}]...")

    removed_bid_prl = 0.0
    removed_ask_prl = 0.0
    matched_spans = 0

    for s_f, e_f, side_f, px, qty in spans:
        s_i = max(0, int(s_f))
        e_i = min(N, int(e_f))
        if e_i <= s_i or qty <= 0.0:
            continue
        tol = 0.00006 if s_i >= subcent_off else 0.006
        if int(side_f) == 0:  # BID
            p_sl = bp[s_i:e_i]
            q_sl = bq[s_i:e_i]
            m = np.abs(p_sl - px) <= tol
            if np.any(m):
                matched_spans += 1
                sub_amt = np.minimum(q_sl, np.where(m, qty, 0.0))
                removed_bid_prl += float(sub_amt.sum())
                bq[s_i:e_i] = np.maximum(0.0, q_sl - sub_amt)
        else:  # ASK
            p_sl = ap[s_i:e_i]
            q_sl = aq[s_i:e_i]
            m = np.abs(p_sl - px) <= tol
            if np.any(m):
                matched_spans += 1
                sub_amt = np.minimum(q_sl, np.where(m, qty, 0.0))
                removed_ask_prl += float(sub_amt.sum())
                aq[s_i:e_i] = np.maximum(0.0, q_sl - sub_amt)

    print(f"Matched {matched_spans:,}/{len(spans):,} order spans in Top-10 book!")
    print(f"Removed phantom self-volume (order-seconds): Bid={removed_bid_prl:,.0f} PRL*s | Ask={removed_ask_prl:,.0f} PRL*s")

    # Compact shifted empty levels on [min_active_idx : max_active_idx]
    bp_sub, bq_sub = compact_levels_chunk(bp[min_active_idx:max_active_idx], bq[min_active_idx:max_active_idx], is_bid=True)
    ap_sub, aq_sub = compact_levels_chunk(ap[min_active_idx:max_active_idx], aq[min_active_idx:max_active_idx], is_bid=False)
    # Ensure strict non-crossing a1 >= b1 + tick
    ticks_vec = np.where(np.arange(min_active_idx, max_active_idx) >= subcent_off, 0.0001, 0.01).astype(np.float32)
    ap_sub[:, 0] = np.maximum(ap_sub[:, 0], bp_sub[:, 0] + ticks_vec)

    b1_changed_pct = float(np.mean(np.abs(bp_sub[:, 0] - bp[min_active_idx:max_active_idx, 0]) > 1e-5) * 100.0)
    a1_changed_pct = float(np.mean(np.abs(ap_sub[:, 0] - ap[min_active_idx:max_active_idx, 0]) > 1e-5) * 100.0)
    old_top5_b = float(np.mean(np.sum(lob_dict["bid_q"][subcent_off:, :5] * lob_dict["bid_p"][subcent_off:, :5], axis=1)))

    bp[min_active_idx:max_active_idx] = bp_sub
    bq[min_active_idx:max_active_idx] = bq_sub
    ap[min_active_idx:max_active_idx] = ap_sub
    aq[min_active_idx:max_active_idx] = aq_sub
    new_top5_b = float(np.mean(np.sum(bq[subcent_off:, :5] * bp[subcent_off:, :5], axis=1)))

    print(f"Sub-Cent Top-5 Bid Book USD: BEFORE=${old_top5_b:,.1f} -> AFTER DECONTAM=${new_top5_b:,.1f} (-{(1 - new_top5_b/old_top5_b)*100:.1f}% self-wall illusion removed!)")
    print(f"Ticks where our past bot WAS the sole L1 BBO: Bid1 shifted on {b1_changed_pct:.2f}% ticks | Ask1 shifted on {a1_changed_pct:.2f}% ticks!")

    tmp_lob = LOB_PATH.with_name("rl_guard_lob_cache_600h_decontam_tmp.npz")
    np.savez(tmp_lob, **lob_dict)
    tmp_lob.replace(LOB_PATH)

    # Update z[0..8] in rl_hybrid_market_obs_84d.npz on [min_active_idx:max_active_idx] so LOB features match decontaminated book!
    obs84 = np.load(OBS84_RAW_BAK)["market_obs_84d"].copy()
    sl = slice(min_active_idx, max_active_idx)
    b1 = bp[sl, 0]
    a1 = ap[sl, 0]
    mid = np.maximum(1e-4, 0.5 * (b1 + a1))
    b1q = np.maximum(0.0, bq[sl, 0])
    a1q = np.maximum(0.0, aq[sl, 0])
    b3q = np.sum(np.maximum(0.0, bq[sl, :3]), axis=1)
    a3q = np.sum(np.maximum(0.0, aq[sl, :3]), axis=1)
    b5q = np.sum(np.maximum(0.0, bq[sl, :5]), axis=1)
    a5q = np.sum(np.maximum(0.0, aq[sl, :5]), axis=1)
    b10q = np.sum(np.maximum(0.0, bq[sl, :10]), axis=1)
    a10q = np.sum(np.maximum(0.0, aq[sl, :10]), axis=1)

    obs84[sl, 0] = np.clip((a1 - b1) / mid, 0.0, 0.05)
    obs84[sl, 1] = np.clip((b1q - a1q) / (b1q + a1q + 1e-5), -1.0, 1.0)
    obs84[sl, 2] = np.clip((b3q - a3q) / (b3q + a3q + 1e-5), -1.0, 1.0)
    obs84[sl, 3] = np.clip((b5q - a5q) / (b5q + a5q + 1e-5), -1.0, 1.0)
    obs84[sl, 4] = np.clip((b10q - a10q) / (b10q + a10q + 1e-5), -1.0, 1.0)
    obs84[sl, 5] = np.log1p(b1q) / 10.0
    obs84[sl, 6] = np.log1p(a1q) / 10.0
    obs84[sl, 7] = np.log1p(b3q) / 10.0
    obs84[sl, 8] = np.log1p(a3q) / 10.0

    tmp_obs = OBS84_PATH.with_name("rl_hybrid_market_obs_84d_decontam_tmp.npz")
    np.savez(tmp_obs, market_obs_84d=obs84)
    tmp_obs.replace(OBS84_PATH)

    print(f"✅ Step 2 Completed in {time.time()-t0:.1f}s! Clean LOB & 84D caches saved atomically.")


if __name__ == "__main__":
    main()
