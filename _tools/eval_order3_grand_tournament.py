#!/usr/bin/env python3
"""
_tools/eval_order3_grand_tournament.py
=============================================================================
Multi-Generation Blind OOS Tournament — Order-3 K=8 Signal Stack v2
Строгое слепое сравнение поколений чемпионов (L46 Live, L48, L49, L50, L51, L52)
на изолированном Blind OOS и 100% новых живых данных с VPS после обучения!
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

SAVE_CACHE_ROOT = BASE_DIR / "scratch"
os.environ.setdefault("TRITON_CACHE_DIR", str(SAVE_CACHE_ROOT / ".triton_cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(SAVE_CACHE_ROOT / ".inductor_cache"))

from _tools.fused_order3_clean_tick import evaluate_order3_basket, TOTAL_DIM_ORDER3, K_MODES_ORDER3

MKT_60D_PATH = BASE_DIR / "data/cache/rl_hybrid_market_obs_60d.npz"
LOB_PATH = BASE_DIR / "data/cache/rl_guard_lob_cache_600h.npz"
REPORT_PATH = BASE_DIR / "data/reports/order3_fresh_blind_oos_report.json"


def _regime_summary(b1_arr: np.ndarray, a1_arr: np.ndarray, s_lo: int, s_hi: int) -> dict:
    b_s = b1_arr[s_lo:s_hi]
    a_s = a1_arr[s_lo:s_hi]
    mid = 0.5 * (b_s + a_s)
    spr_bp = ((a_s - b_s) / np.maximum(mid, 1e-4)) * 10000.0
    ret_pct = (float(mid[-1]) / max(float(mid[0]), 1e-6) - 1.0) * 100.0
    return {
        "hours": round((s_hi - s_lo) / 3600.0, 2),
        "start_mid": round(float(mid[0]), 4),
        "end_mid": round(float(mid[-1]), 4),
        "min_mid": round(float(np.min(mid)), 4),
        "max_mid": round(float(np.max(mid)), 4),
        "hodl_ret_pct": round(ret_pct, 3),
        "median_spread_bp": round(float(np.median(spr_bp)), 2),
    }


def _make_basket(idx_arr: np.ndarray, prefix: str, inv_frac: float = 0.0) -> list[tuple[str, int, float, float]]:
    inv_profiles = [(0.25, 0.004), (0.45, 0.008), (0.65, 0.014)]
    n_inv = int(round(len(idx_arr) * inv_frac))
    inv_slots = set(np.linspace(1, len(idx_arr) - 1, num=n_inv, dtype=int).tolist()) if n_inv > 0 else set()
    out, ic = [], 0
    for k, i in enumerate(idx_arr):
        if k in inv_slots:
            r0, roff = inv_profiles[ic % len(inv_profiles)]
            ic += 1
            out.append((f"{prefix}{k:02d}", int(i), float(r0), float(roff)))
        else:
            out.append((f"{prefix}{k:02d}", int(i), 0.0, 0.0))
    return out


def _eval_dual_scale(
    mkt_t: torch.Tensor,
    b10_t: torch.Tensor,
    a10_t: torch.Tensor,
    bq10_t: torch.Tensor,
    aq10_t: torch.Tensor,
    sv_t: torch.Tensor,
    bv_t: torch.Tensor,
    cands: list[dict],
    s_lo: int,
    s_hi: int,
    n_pairs: int,
    macro_win_len: int,
    micro_win_len: int,
    tick_sz: float,
    device: torch.device,
    inv_frac: float = 0.0,
) -> list[dict]:
    span_len = s_hi - s_lo
    eff_mac_len = min(macro_win_len, max(1801, int(span_len * 0.55) | 1))
    eff_mic_len = min(micro_win_len, max(901, int(span_len * 0.25) | 1))
    mac_end = max(s_lo + 10, s_hi - eff_mac_len - 30)
    mic_end = max(s_lo + 120, s_hi - eff_mic_len - 30)
    mac_idx = np.linspace(s_lo, mac_end, num=n_pairs, dtype=int)
    mic_idx = np.linspace(min(s_lo + 120, mic_end), mic_end, num=n_pairs, dtype=int)
    mac_bask = _make_basket(mac_idx, "OMAC", inv_frac=inv_frac)
    mic_bask = _make_basket(mic_idx, "OMIC", inv_frac=inv_frac)

    def _run_chunked(bask: list, w_len: int, chunk_w: int = 28) -> list[dict]:
        if len(bask) <= chunk_w:
            return evaluate_order3_basket(
                mkt_39d_all=mkt_t, b10_all=b10_t, a10_all=a10_t,
                bq10_all=bq10_t, aq10_all=aq10_t, sv_all=sv_t, bv_all=bv_t,
                cand_states=cands, val_windows=bask, win_len=w_len, tick_sz=tick_sz,
                device=device, pad_to_max_c=False,
            )
        chunks_res, weights = [], []
        for s_c in range(0, len(bask), chunk_w):
            sub_b = bask[s_c : s_c + chunk_w]
            pad_n = chunk_w - len(sub_b)
            sub_padded = sub_b + [sub_b[-1]] * pad_n if pad_n > 0 else sub_b
            r_sub = evaluate_order3_basket(
                mkt_39d_all=mkt_t, b10_all=b10_t, a10_all=a10_t,
                bq10_all=bq10_t, aq10_all=aq10_t, sv_all=sv_t, bv_all=bv_t,
                cand_states=cands, val_windows=sub_padded, win_len=w_len, tick_sz=tick_sz,
                device=device, pad_to_max_c=False,
            )
            chunks_res.append(r_sub)
            weights.append(float(len(sub_b)))
        tot_w = sum(weights)
        out_m = []
        for c_i in range(len(cands)):
            d = {}
            for k in chunks_res[0][c_i]:
                v0 = chunks_res[0][c_i][k]
                if k in ("fills", "cancels", "requeues", "clean_ev_pct", "full_ev_pct", "inv_delta_ev_pct", "turn_mult"):
                    d[k] = sum(ch[c_i][k] * (w / float(chunk_w)) for ch, w in zip(chunks_res, weights))
                elif k == "dd_pct":
                    d[k] = max(float(ch[c_i][k]) for ch in chunks_res)
                elif isinstance(v0, list):
                    d[k] = [sum(float(ch[c_i][k][j]) * w for ch, w in zip(chunks_res, weights)) / tot_w for j in range(len(v0))]
                else:
                    d[k] = sum(float(ch[c_i][k]) * w for ch, w in zip(chunks_res, weights)) / tot_w
            out_m.append(d)
        return out_m

    r_mac = _run_chunked(mac_bask, eff_mac_len)
    r_mic = _run_chunked(mic_bask, eff_mic_len)

    merged = []
    for a, b in zip(r_mac, r_mic):
        joint_geo = 0.50 * float(a["geo_score_bp"]) + 0.50 * (1.55 * float(b["geo_score_bp"]))
        tot_f = float(a["fills"]) + float(b["fills"])
        tot_c = float(a["cancels"]) + float(b["cancels"])
        merged.append({
            "geo_score_bp": round(joint_geo, 4),
            "macro_geo_bp": round(float(a["geo_score_bp"]), 4),
            "micro_geo_bp": round(float(b["geo_score_bp"]), 4),
            "clean_ev_pct": round(float(a["clean_ev_pct"]) + float(b["clean_ev_pct"]), 5),
            "inv_ev_pct": round(float(a.get("inv_delta_ev_pct", 0.0)) + float(b.get("inv_delta_ev_pct", 0.0)), 5),
            "full_ev_pct": round(float(a["full_ev_pct"]) + float(b["full_ev_pct"]), 5),
            "cvar25_bp": round(0.5 * float(a["cvar25_bp"]) + 0.5 * float(b["cvar25_bp"]), 4),
            "dd_pct": round(max(float(a["dd_pct"]), float(b["dd_pct"])), 4),
            "fills": round(tot_f, 2),
            "fills_per_w": round(tot_f / float(2 * n_pairs), 3),
            "cnl_per_fill": round(tot_c / max(tot_f, 0.5), 3),
            "eta_queue_pct": round(0.5 * float(a.get("eta_queue_pct", 0.0)) + 0.5 * float(b.get("eta_queue_pct", 0.0)), 2),
            "turn_mult": round(float(a.get("turn_mult", 0.0)) + float(b.get("turn_mult", 0.0)), 4),
            "mode_mi_bits": round(0.5 * float(a["mode_mi_bits"]) + 0.5 * float(b["mode_mi_bits"]), 4),
        })
    return merged


def run_extended_oos(
    device_str: str = "cuda:0",
    oos_cut: int = 537042,
    fresh_cut: int = 600042,
    macro_win_len: int = 6001,
    micro_win_len: int = 1501,
) -> dict:
    device = torch.device(device_str)
    mkt_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print("=" * 134)
    print(f"🧪 MULTI-GENERATION BLIND OOS TOURNAMENT (60D Signal Stack v2 | K={K_MODES_ORDER3} | Honest L2 Liquidation) ON {device}")
    print("=" * 134, flush=True)

    mkt_raw = np.load(MKT_60D_PATH, mmap_mode="r")
    lob_raw = np.load(LOB_PATH, mmap_mode="r")
    b1_full = lob_raw["bid_p"][:, 0]
    subcent_hits = np.where(np.abs(b1_full * 100.0 - np.round(b1_full * 100.0)) > 1e-3)[0]
    subcent_offset = int(subcent_hits[0]) if len(subcent_hits) > 0 else 0

    mkt_sub = np.asarray(mkt_raw["market_obs_60d"][subcent_offset:], dtype=np.float32)
    b10_np = np.asarray(lob_raw["bid_p"][subcent_offset:, :10], dtype=np.float32)
    a10_np = np.asarray(lob_raw["ask_p"][subcent_offset:, :10], dtype=np.float32)
    total_rows = len(mkt_sub)
    tick_sz = 0.0001
    brand_new_rows = max(0, total_rows - fresh_cut)
    print(
        f"📦 Эпоха $0.0001: {total_rows:,} сек ({total_rows/3600:.2f}ч) | "
        f"Blind OOS >= {oos_cut:,} ({(total_rows-oos_cut)/3600:.2f}ч) | "
        f"100% Новые живые данные после старта урока [>={fresh_cut:,}]: +{brand_new_rows:,} сек (+{brand_new_rows/3600:.2f}ч)",
        flush=True,
    )

    mkt_t = torch.from_numpy(mkt_sub).to(device=device, dtype=mkt_dtype)
    b10_t = torch.from_numpy(b10_np).to(device=device)
    a10_t = torch.from_numpy(a10_np).to(device=device)
    bq10_t = torch.from_numpy(np.asarray(lob_raw["bid_q"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    aq10_t = torch.from_numpy(np.asarray(lob_raw["ask_q"][subcent_offset:, :10], dtype=np.float32)).to(device=device)
    sv_t = torch.from_numpy(np.asarray(lob_raw["sell_vol"][subcent_offset:], dtype=np.float32)).to(device=device)
    bv_t = torch.from_numpy(np.asarray(lob_raw["buy_vol"][subcent_offset:], dtype=np.float32)).to(device=device)

    champ_dir = BASE_DIR / "models/prl/order3_champion"
    candidates_spec = [
        ("★ Current Champion v1", BASE_DIR / "models/prl/release_champion_v1.pt"),
        ("Prev Live Champion (L46)", BASE_DIR / "models/prl/release_champion_v1_bak_l46.pt"),
        ("L48 Island-0 Best", champ_dir / "bak_l48_qmoe/island_0_best.pt"),
        ("L48 Global Best", champ_dir / "bak_l48_qmoe/global_best.pt"),
        ("L49 Global Best", champ_dir / "bak_l49_dual_shield/global_best.pt"),
        ("L49 Island-1 Best", champ_dir / "bak_l49_dual_shield/island_1_best.pt"),
        ("L49 Soup Top3", champ_dir / "bak_l49_dual_shield/order3_soup_top3.pt"),
        ("L50 Global Best", champ_dir / "bak_l50/global_best.pt"),
        ("L51 Island-0 Best", champ_dir / "bak_l51/island_0_best.pt"),
        ("L51 Global Best", champ_dir / "bak_l51/global_best.pt"),
        ("L51 Soup Top3", champ_dir / "bak_l51/order3_soup_top3.pt"),
        ("L52 Island-0 Best", champ_dir / "bak_l52/island_0_best.pt"),
        ("L52 Island-1 Best", champ_dir / "bak_l52/island_1_best.pt"),
        ("L52 Global Best", champ_dir / "bak_l52/global_best.pt"),
        ("L52 Soup Top3", champ_dir / "bak_l52/order3_soup_top3.pt"),
        ("L53 Island-0 Best", champ_dir / "bak_l53/island_0_best.pt"),
        ("L53 Island-1 Best", champ_dir / "bak_l53/island_1_best.pt"),
        ("L53 Global Best", champ_dir / "bak_l53/global_best.pt"),
        ("L53 Soup Isl0", champ_dir / "bak_l53/soup_isl0.pt"),
        ("L53 Soup Top3", champ_dir / "bak_l53/order3_soup_top3.pt"),
        ("L54 Island-0 Best", champ_dir / "bak_l54/island_0_best.pt"),
        ("L54 Island-1 Best", champ_dir / "bak_l54/island_1_best.pt"),
        ("L54 Global Best", champ_dir / "bak_l54/global_best.pt"),
        ("L54 Soup Isl0", champ_dir / "bak_l54/soup_isl0.pt"),
        ("L54 Soup Top3", champ_dir / "bak_l54/order3_soup_top3.pt"),
        ("Chimera(R54+B53)", champ_dir / "bak_l55/chimera_54_53.pt"),
        ("Chimera(R54+B50)", champ_dir / "bak_l55/chimera_54_50.pt"),
        ("L55 Global Best", champ_dir / "bak_l55/global_best.pt"),
        ("L55 Soup Top3", champ_dir / "bak_l55/order3_soup_top3.pt"),
        ("L56 Island-0 Best", champ_dir / "bak_l56/island_0_best.pt"),
        ("L56 Island-1 Best", champ_dir / "bak_l56/island_1_best.pt"),
        ("L56 Global Best", champ_dir / "bak_l56/global_best.pt"),
        ("L56 Soup Top3", champ_dir / "bak_l56/order3_soup_top3.pt"),
        ("Current Latest", champ_dir / "global_best.pt"),
        ("Current Soup", champ_dir / "order3_soup_top3.pt"),
    ]
    models: list[tuple[str, dict]] = []
    seen_sigs = set()
    for label, path in candidates_spec:
        if path.exists():
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if "w1" in ck and ck["w1"].shape[0] == TOTAL_DIM_ORDER3 and int(ck.get("k_modes", K_MODES_ORDER3)) == K_MODES_ORDER3:
                wf_v = str(ck.get("wf_version", "?"))
                sig = (round(float(torch.sum(ck["w1"].float()).item()), 4), round(float(torch.sum(ck["w2_moe"].float()).item()), 4))
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                it_s = "soup" if ck.get("is_soup") else f"it={ck.get('iteration', '?')}"
                tag = f"{label} ({wf_v[-12:]} {it_s})"
                models.append((tag, {k: (v.to(dtype=mkt_dtype) if hasattr(v, "shape") else v) for k, v in ck.items()}))

    baskets_cfg = []
    if brand_new_rows >= 1800:
        baskets_cfg.append((f"1_BRAND_NEW_LIVE_{brand_new_rows/3600:.2f}h [{fresh_cut//1000}k..{total_rows//1000}k] (Clean)", fresh_cut, total_rows, 28, 0.0))
        baskets_cfg.append((f"2_BRAND_NEW_LIVE_{brand_new_rows/3600:.2f}h [{fresh_cut//1000}k..{total_rows//1000}k] (25% Inv)", fresh_cut, total_rows, 28, 0.25))
    if total_rows - oos_cut >= macro_win_len + 1000:
        baskets_cfg.append((f"3_FULL_BLIND_OOS_{(total_rows-oos_cut)/3600:.1f}h [{oos_cut//1000}k..{total_rows//1000}k] (Clean)", oos_cut, total_rows, 56, 0.0))
        baskets_cfg.append((f"4_FULL_BLIND_OOS_{(total_rows-oos_cut)/3600:.1f}h [{oos_cut//1000}k..{total_rows//1000}k] (25% Inv)", oos_cut, total_rows, 56, 0.25))
    if total_rows - 600042 >= macro_win_len + 1000 and oos_cut != 600042:
        baskets_cfg.append((f"5_MARATHON_LIVE_WINDOW_{(total_rows-600042)/3600:.1f}h [600k..{total_rows//1000}k] (Clean)", 600042, total_rows, 56, 0.0))
        baskets_cfg.append((f"6_MARATHON_LIVE_WINDOW_{(total_rows-600042)/3600:.1f}h [600k..{total_rows//1000}k] (25% Inv)", 600042, total_rows, 56, 0.25))
    if total_rows - 640588 >= macro_win_len + 1000:
        baskets_cfg.append((f"7_DAY_MARATHON_LIVE_{(total_rows-640588)/3600:.1f}h [640k..{total_rows//1000}k] (Clean)", 640588, total_rows, 56, 0.0))
        baskets_cfg.append((f"8_DAY_MARATHON_LIVE_{(total_rows-640588)/3600:.1f}h [640k..{total_rows//1000}k] (25% Inv)", 640588, total_rows, 56, 0.25))

    all_reports = {}
    regime_reports = {}
    cand_states = [m[1] for m in models]
    for b_name, s_lo, s_hi, n_pairs, inv_f in baskets_cfg:
        reg = _regime_summary(b10_np[:, 0], a10_np[:, 0], s_lo, s_hi)
        regime_reports[b_name] = reg
        t0 = time.time()
        res_list = _eval_dual_scale(
            mkt_t, b10_t, a10_t, bq10_t, aq10_t, sv_t, bv_t,
            cand_states, s_lo, s_hi, n_pairs, macro_win_len, micro_win_len, tick_sz, device, inv_frac=inv_f,
        )
        dt = time.time() - t0
        print(
            f"\n📊 Корзина: {b_name} | 2x{n_pairs}={2*n_pairs} окон | Время: {dt:.2f}с\n"
            f"   Режим рынка ({reg['hours']}ч): Mid ${reg['start_mid']:.4f} -> ${reg['end_mid']:.4f} "
            f"(HODL {reg['hodl_ret_pct']:+.2f}%, диапазон ${reg['min_mid']:.4f}..${reg['max_mid']:.4f}, мед. спред {reg['median_spread_bp']:.1f} bp)"
        )
        print("-" * 134)
        print(f"{'Модель':<45} | {'JointGeo':<10} | {'MacGeo':<9} | {'MicGeo':<9} | {'CleanEV':<9} | {'FullEV':<9} | {'CVaR25':<9} | {'Fills/w':<7} | {'Cnl/F':<6} | {'MI':<5}")
        print("-" * 134)
        all_reports[b_name] = []
        for (m_name, _), r in zip(models, res_list):
            print(
                f"{m_name:<45} | "
                f"{r['geo_score_bp']:+7.2f} bp | "
                f"{r['macro_geo_bp']:+6.2f} bp | "
                f"{r['micro_geo_bp']:+6.2f} bp | "
                f"{r['clean_ev_pct']:+8.4f}% | "
                f"{r['full_ev_pct']:+8.4f}% | "
                f"{r['cvar25_bp']:+6.2f} bp | "
                f"{r['fills_per_w']:7.2f} | "
                f"{r['cnl_per_fill']:5.2f}x | "
                f"{r['mode_mi_bits']:4.2f}b"
            )
            all_reports[b_name].append({"model": m_name, **r})

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "subcent_rows": total_rows,
                "oos_cut": oos_cut,
                "fresh_cut": fresh_cut,
                "brand_new_hours": round(brand_new_rows / 3600.0, 2),
                "regimes": regime_reports,
                "baskets": all_reports,
            },
            f,
            indent=2,
        )
    print(f"\n📁 Полный отчет сохранен в {REPORT_PATH}")
    return all_reports


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--oos-cut", type=int, default=537042)
    ap.add_argument("--fresh-cut", type=int, default=600042)
    args = ap.parse_args()
    run_extended_oos(device_str=args.device, oos_cut=args.oos_cut, fresh_cut=args.fresh_cut)
