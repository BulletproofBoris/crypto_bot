#!/usr/bin/env python3
"""
Потоковый онлайн-вычислитель всех 299 микроструктурных, кросс-биржевых и макро-признаков
в реальном времени (1 Гц) для LiveTriadRLTrader.

Поддерживает:
- Быстрое дочитывание (tail seek) живых логов коллекторов на VPS (/opt/crypto_bot/data/realtime/):
  1) prlusdt/depth.csv и trades.csv (SafeTrade 50 уровней + сделки)
  2) prlusdt_bigone/depth_bigone.csv и trades_bigone.csv (BigONE 50 уровней + сделки)
  3) macro_binance/binance_macro.csv (Binance BTCUSDT + NEARUSDT)
- Векторный пересчет базовых L2-метрик (VWAP, slippage, стены, дисбалансы, CVD 1m/5m/15m)
  и полного банка стационарных фильтров W = [5, 20, 60, 300]с (OFI, Band-Pass, Lead-Lag, Arb Z-score)
  на кольцевом буфере 310 секунд за ~3-6 мс.
"""

import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from _tools.microstructure_features import (
    add_exchange_filter_bank,
    add_cross_exchange_and_macro_filters,
)


def _calc_vwap_1row(prices: np.ndarray, qtys: np.ndarray, side: str, target_usd: float) -> float:
    filled_usd = 0.0
    filled_qty = 0.0
    last_p = float(prices[0]) if len(prices) > 0 and prices[0] > 0 else 1.0
    for p, q in zip(prices, qtys):
        if p <= 0 or q <= 0 or not np.isfinite(p) or not np.isfinite(q):
            continue
        last_p = float(p)
        lvl_usd = float(p * q)
        take_usd = min(target_usd - filled_usd, lvl_usd)
        filled_usd += take_usd
        filled_qty += take_usd / float(p)
        if filled_usd >= target_usd:
            break
    rem_usd = max(0.0, target_usd - filled_usd)
    if rem_usd > 0:
        worst_p = last_p * (1.005 if side == "ask" else 0.995)
        if worst_p > 0:
            filled_qty += rem_usd / worst_p
    return float(target_usd / filled_qty) if filled_qty > 0 else last_p


class LiveFeatureStreamer:
    """Поддерживает 310-секундное скользящее окно сырых 1с-данных и вычисляет полный вектор признаков Триады."""

    def __init__(self, seed_df: pd.DataFrame, feature_names: list[str], coin: str = "prl", rt_root: str = "/opt/crypto_bot/data/realtime"):
        self.coin = coin.lower()
        self.feature_names = list(feature_names)
        self.rt_dir = Path(rt_root)
        self.st_depth_path = self.rt_dir / "prlusdt" / "depth.csv"
        self.st_trades_path = self.rt_dir / "prlusdt" / "trades.csv"
        self.bo_depth_path = self.rt_dir / "prlusdt_bigone" / "depth_bigone.csv"
        self.bo_trades_path = self.rt_dir / "prlusdt_bigone" / "trades_bigone.csv"
        self.macro_path = self.rt_dir / "macro_binance" / "binance_macro.csv"
        self.onchain_path = self.rt_dir / "onchain" / "prl_onchain.csv"

        self.seed_template = seed_df.iloc[-1:].copy().reset_index(drop=True)
        self.buf_len = 310
        self.raw_rows = deque(maxlen=self.buf_len)
        self.st_delta_hist = deque([0.0] * 900, maxlen=900)
        self.st_vol_hist = deque([0.0] * 60, maxlen=60)
        self.bo_delta_hist = deque([0.0] * 900, maxlen=900)
        self.bo_vol_hist = deque([0.0] * 60, maxlen=60)

        self._st_trade_pos = 0
        self._bo_trade_pos = 0
        self._seen_st_trade_ids = deque(maxlen=500)
        self._seen_bo_trade_ids = deque(maxlen=500)
        self._last_block_height = -1
        self._sec_since_block = 10.0
        self._diff_1h_ago = 0.0

        self._warmup_from_seed_and_rt(seed_df)

    def _tail_lines(self, path: Path, n: int = 350) -> list[str]:
        if not path.exists():
            return []
        try:
            if n == 1:
                sz = path.stat().st_size
                with open(path, "rb") as f:
                    f.seek(max(0, sz - 4096))
                    chunk = f.read().decode("utf-8", errors="ignore")
                lines = [ln.strip() for ln in chunk.splitlines() if ln.strip() and not ln.startswith("timestamp") and not ln.startswith("id,")]
                return lines[-1:] if lines else []
            out = subprocess.run(["tail", "-n", str(n), str(path)], capture_output=True, text=True, timeout=2).stdout
            return [ln.strip() for ln in out.splitlines() if ln.strip() and not ln.startswith("timestamp") and not ln.startswith("id,")]
        except Exception:
            return []

    def _parse_50lvl_line(self, line: str, prefix: str, target_dict: dict):
        parts = line.split(",")
        if len(parts) < 201:
            return False
        bp = np.zeros(50, dtype=np.float64)
        bq = np.zeros(50, dtype=np.float64)
        ap = np.zeros(50, dtype=np.float64)
        aq = np.zeros(50, dtype=np.float64)
        for i in range(50):
            try:
                bp[i] = float(parts[1 + 2 * i]) if parts[1 + 2 * i] else 0.0
                bq[i] = float(parts[2 + 2 * i]) if parts[2 + 2 * i] else 0.0
            except ValueError:
                pass
            try:
                ap[i] = float(parts[101 + 2 * i]) if parts[101 + 2 * i] else 0.0
                aq[i] = float(parts[102 + 2 * i]) if parts[102 + 2 * i] else 0.0
            except ValueError:
                pass
        if bp[0] <= 0 or ap[0] <= 0:
            return False
        for i in range(50):
            target_dict[f"{prefix}_bid_{i+1}_price"] = bp[i]
            target_dict[f"{prefix}_bid_{i+1}_qty"] = bq[i]
            target_dict[f"{prefix}_ask_{i+1}_price"] = ap[i]
            target_dict[f"{prefix}_ask_{i+1}_qty"] = aq[i]
        if len(parts) >= 207:
            try:
                target_dict[f"{prefix}_deep_bid_vol"] = float(parts[201] or 0.0)
                target_dict[f"{prefix}_deep_ask_vol"] = float(parts[202] or 0.0)
                target_dict[f"{prefix}_max_bid_wall_price"] = float(parts[203] or bp[0])
                target_dict[f"{prefix}_max_bid_wall_qty"] = float(parts[204] or 0.0)
                target_dict[f"{prefix}_max_ask_wall_price"] = float(parts[205] or ap[0])
                target_dict[f"{prefix}_max_ask_wall_qty"] = float(parts[206] or 0.0)
            except ValueError:
                pass
        self._recalc_l2_Book_scalars(target_dict, prefix, bp, bq, ap, aq)
        return True

    def _recalc_l2_Book_scalars(self, d: dict, prefix: str, bp: np.ndarray, bq: np.ndarray, ap: np.ndarray, aq: np.ndarray):
        b1, a1 = float(bp[0]), float(ap[0])
        mid = 0.5 * (b1 + a1)
        spread = max(0.0, a1 - b1)
        d[f"{prefix}_best_bid"] = b1
        d[f"{prefix}_best_ask"] = a1
        d[f"{prefix}_mid_price"] = mid
        d[f"{prefix}_spread"] = spread
        d[f"{prefix}_spread_pct"] = spread / mid if mid > 0 else 0.0

        b5_q = float(np.sum(bq[:5]))
        a5_q = float(np.sum(aq[:5]))
        b10_q = float(np.sum(bq[:10]))
        a10_q = float(np.sum(aq[:10]))
        b11_50_q = float(np.sum(bq[10:50]))
        a11_50_q = float(np.sum(aq[10:50]))
        b50_q = b10_q + b11_50_q
        a50_q = a10_q + a11_50_q

        d[f"{prefix}_bids_vol_top5"] = b5_q
        d[f"{prefix}_asks_vol_top5"] = a5_q
        d[f"{prefix}_bids_usd_top5"] = float(np.sum(bp[:5] * bq[:5]))
        d[f"{prefix}_asks_usd_top5"] = float(np.sum(ap[:5] * aq[:5]))
        d[f"{prefix}_imbalance_top5"] = (b5_q - a5_q) / (b5_q + a5_q) if (b5_q + a5_q) > 0 else 0.0
        d[f"{prefix}_imbalance_top10"] = (b10_q - a10_q) / (b10_q + a10_q) if (b10_q + a10_q) > 0 else 0.0
        d[f"{prefix}_bids_vol_11_50"] = b11_50_q
        d[f"{prefix}_asks_vol_11_50"] = a11_50_q
        d[f"{prefix}_imbalance_11_50"] = (b11_50_q - a11_50_q) / (b11_50_q + a11_50_q) if (b11_50_q + a11_50_q) > 0 else 0.0
        d[f"{prefix}_imbalance_top50"] = (b50_q - a50_q) / (b50_q + a50_q) if (b50_q + a50_q) > 0 else 0.0

        bw_i = int(np.argmax(bq)) if np.max(bq) > 0 else 0
        aw_i = int(np.argmax(aq)) if np.max(aq) > 0 else 0
        bw_p = float(d.get(f"{prefix}_max_bid_wall_price", bp[bw_i]) or bp[bw_i])
        bw_q = float(d.get(f"{prefix}_max_bid_wall_qty", bq[bw_i]) or bq[bw_i])
        aw_p = float(d.get(f"{prefix}_max_ask_wall_price", ap[aw_i]) or ap[aw_i])
        aw_q = float(d.get(f"{prefix}_max_ask_wall_qty", aq[aw_i]) or aq[aw_i])
        d[f"{prefix}_max_bid_wall_price"] = bw_p
        d[f"{prefix}_max_bid_wall_qty"] = bw_q
        d[f"{prefix}_max_ask_wall_price"] = aw_p
        d[f"{prefix}_max_ask_wall_qty"] = aw_q
        d[f"{prefix}_bid_wall_dist_bps"] = float(np.clip((mid - bw_p) / mid * 1e4, 0.0, 5000.0)) if mid > 0 else 0.0
        d[f"{prefix}_ask_wall_dist_bps"] = float(np.clip((aw_p - mid) / mid * 1e4, 0.0, 5000.0)) if mid > 0 else 0.0
        d[f"{prefix}_wall_vol_imbalance"] = (bw_q - aw_q) / (bw_q + aw_q) if (bw_q + aw_q) > 0 else 0.0

        vwap_b100 = _calc_vwap_1row(ap, aq, "ask", 100.0)
        vwap_s100 = _calc_vwap_1row(bp, bq, "bid", 100.0)
        vwap_b300 = _calc_vwap_1row(ap, aq, "ask", 300.0)
        vwap_s300 = _calc_vwap_1row(bp, bq, "bid", 300.0)
        d[f"{prefix}_eff_buy_vwap_100"] = vwap_b100
        d[f"{prefix}_eff_sell_vwap_100"] = vwap_s100
        d[f"{prefix}_slippage_buy_100_pct"] = (vwap_b100 - a1) / a1 if a1 > 0 else 0.0
        d[f"{prefix}_slippage_sell_100_pct"] = (b1 - vwap_s100) / b1 if b1 > 0 else 0.0
        d[f"{prefix}_eff_buy_vwap_300"] = vwap_b300
        d[f"{prefix}_eff_sell_vwap_300"] = vwap_s300
        d[f"{prefix}_slippage_buy_300_pct"] = (vwap_b300 - a1) / a1 if a1 > 0 else 0.0
        d[f"{prefix}_slippage_sell_300_pct"] = (b1 - vwap_s300) / b1 if b1 > 0 else 0.0

    def _parse_macro_line(self, line: str, d: dict):
        parts = line.split(",")
        if len(parts) < 17:
            return
        try:
            prev_btc = float(d.get("macro_btc_mid", 0.0) or 0.0)
            prev_near = float(d.get("macro_near_mid", 0.0) or 0.0)
            btc_b, btc_a, btc_m, btc_l = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            near_b, near_a, near_m, near_l = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            btc_bq, btc_aq, btc_bv, btc_sv = float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12])
            near_bq, near_aq, near_bv, near_sv = float(parts[13]), float(parts[14]), float(parts[15]), float(parts[16])
            d["macro_btc_bid"], d["macro_btc_ask"], d["macro_btc_mid"], d["macro_btc_last"] = btc_b, btc_a, btc_m, btc_l
            d["macro_near_bid"], d["macro_near_ask"], d["macro_near_mid"], d["macro_near_last"] = near_b, near_a, near_m, near_l
            d["macro_btc_bid_qty"], d["macro_btc_ask_qty"], d["macro_btc_buy_vol"], d["macro_btc_sell_vol"] = btc_bq, btc_aq, btc_bv, btc_sv
            d["macro_near_bid_qty"], d["macro_near_ask_qty"], d["macro_near_buy_vol"], d["macro_near_sell_vol"] = near_bq, near_aq, near_bv, near_sv
            d["macro_btc_ret_1s"] = (btc_m / prev_btc - 1.0) if prev_btc > 0 else 0.0
            d["macro_near_ret_1s"] = (near_m / prev_near - 1.0) if prev_near > 0 else 0.0
            d["macro_btc_bbo_imb"] = (btc_bq - btc_aq) / (btc_bq + btc_aq) if (btc_bq + btc_aq) > 0 else 0.0
            d["macro_near_bbo_imb"] = (near_bq - near_aq) / (near_bq + near_aq) if (near_bq + near_aq) > 0 else 0.0
        except ValueError:
            pass

    def _parse_onchain_line(self, line: str, d: dict, advance_timer: bool = True):
        parts = line.split(",")
        if len(parts) < 5:
            return
        try:
            bh = int(float(parts[1]))
            mp_sz = float(parts[2])
            diff = float(parts[4])
            whale_5k = float(parts[12]) if len(parts) > 12 and parts[12] else 0.0
            if self._last_block_height < 0:
                self._last_block_height = bh
            elif bh != self._last_block_height:
                self._last_block_height = bh
                self._sec_since_block = 0.0
            elif advance_timer:
                self._sec_since_block = min(600.0, self._sec_since_block + 1.0)
            if self._diff_1h_ago <= 0:
                self._diff_1h_ago = diff
            d["onchain_block_height"] = float(bh)
            d["onchain_seconds_since_block"] = float(self._sec_since_block)
            d["onchain_mempool_size"] = mp_sz
            d["onchain_difficulty"] = diff
            d["onchain_whale_volume_5k"] = whale_5k
            d["onchain_diff_pct_change_1h"] = (diff / self._diff_1h_ago - 1.0) if self._diff_1h_ago > 0 else 0.0
        except ValueError:
            pass

    @staticmethod
    def _extract_ts_sec(ts_str: str) -> int:
        ts_str = ts_str.strip()
        if not ts_str:
            return 0
        try:
            v = float(ts_str)
            return int(v // 1000) if v > 1e11 else int(v)
        except ValueError:
            try:
                return int(pd.Timestamp(ts_str).timestamp())
            except Exception:
                return 0

    def _poll_trades(self, path: Path, prefix: str, d: dict, pos_attr: str, seen_ids: deque, delta_hist: deque, vol_hist: deque):
        buy_v, sell_v, cnt = 0.0, 0.0, 0
        if path.exists():
            try:
                sz = path.stat().st_size
                cur_pos = getattr(self, pos_attr)
                if cur_pos == 0 or cur_pos > sz:
                    cur_pos = max(0, sz - 4096)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(cur_pos)
                    lines = f.readlines()
                    setattr(self, pos_attr, f.tell())
                for ln in lines:
                    parts = ln.strip().split(",")
                    if len(parts) < 5 or parts[0] == "id":
                        continue
                    tid = parts[0]
                    if tid in seen_ids:
                        continue
                    seen_ids.append(tid)
                    try:
                        if prefix == "ST":
                            amt = float(parts[2])
                            side = parts[3].strip().upper()
                        else:
                            amt = float(parts[3])
                            side = parts[4].strip().upper()
                        cnt += 1
                        if side in ("BUY", "ASK") if prefix == "ST" else side in ("BID", "BUY"):
                            buy_v += amt
                        else:
                            sell_v += amt
                    except ValueError:
                        pass
            except Exception:
                pass
        tot_v = buy_v + sell_v
        dv = buy_v - sell_v
        delta_hist.append(dv)
        vol_hist.append(tot_v)
        d_arr = list(delta_hist)
        v_arr = list(vol_hist)
        cvd_1m = float(sum(d_arr[-60:]))
        cvd_5m = float(sum(d_arr[-300:]))
        cvd_15m = float(sum(d_arr))
        vol_1m = float(sum(v_arr))
        d[f"{prefix}_trade_vol"] = tot_v
        d[f"{prefix}_buy_vol"] = buy_v
        d[f"{prefix}_sell_vol"] = sell_v
        d[f"{prefix}_trade_count"] = float(cnt)
        d[f"{prefix}_delta_vol"] = dv
        d[f"{prefix}_cvd_1m"] = cvd_1m
        d[f"{prefix}_cvd_5m"] = cvd_5m
        d[f"{prefix}_cvd_15m"] = cvd_15m
        d[f"{prefix}_trade_imbalance_1m"] = cvd_1m / vol_1m if vol_1m > 0 else 0.0

    def _bucket_trades(self, lines: list[str], prefix: str, seen_ids: deque) -> dict[int, tuple[float, float, int]]:
        buckets: dict[int, tuple[float, float, int]] = {}
        for ln in lines:
            parts = ln.strip().split(",")
            if len(parts) < 5 or parts[0] == "id":
                continue
            tid = parts[0]
            seen_ids.append(tid)
            try:
                if prefix == "ST":
                    amt = float(parts[2])
                    side = parts[3].strip().upper()
                    sec = self._extract_ts_sec(parts[4]) + 1
                    is_buy = side in ("BUY", "ASK")
                else:
                    sec = self._extract_ts_sec(parts[1]) + 1
                    amt = float(parts[3])
                    side = parts[4].strip().upper()
                    is_buy = side in ("BID", "BUY")
                if sec <= 1000:
                    continue
                bv, sv, c = buckets.get(sec, (0.0, 0.0, 0))
                if is_buy:
                    bv += amt
                else:
                    sv += amt
                buckets[sec] = (bv, sv, c + 1)
            except ValueError:
                pass
        return buckets

    def _warmup_from_seed_and_rt(self, seed_df: pd.DataFrame):
        base_dict = seed_df.iloc[-1].to_dict()
        for _, r in seed_df.tail(self.buf_len).iterrows():
            d = r.to_dict()
            d["_ts"] = pd.Timestamp.now(tz="UTC")
            self.raw_rows.append(d)

        st_lines = self._tail_lines(self.st_depth_path, n=800)
        bo_lines = self._tail_lines(self.bo_depth_path, n=400)
        m_lines = self._tail_lines(self.macro_path, n=400)
        on_lines = self._tail_lines(self.onchain_path, n=200)
        st_tr_Map = self._bucket_trades(self._tail_lines(self.st_trades_path, n=1500), "ST", self._seen_st_trade_ids)
        bo_tr_Map = self._bucket_trades(self._tail_lines(self.bo_trades_path, n=1500), "BO", self._seen_bo_trade_ids)

        if st_lines:
            self.raw_rows.clear()
            now_sec = self._extract_ts_sec(st_lines[-1].split(",", 1)[0]) or int(datetime.now(timezone.utc).timestamp())
            if bo_lines:
                now_sec = max(now_sec, self._extract_ts_sec(bo_lines[-1].split(",", 1)[0]))

            # Прогреваем 1-часовую сложность и возраст последнего блока по истории prl_onchain.csv
            if on_lines:
                try:
                    self._diff_1h_ago = float(on_lines[0].split(",")[4])
                except Exception:
                    pass
                last_bh = -1
                last_bh_sec = now_sec - 15
                for oln in on_lines:
                    op = oln.split(",")
                    if len(op) >= 5:
                        try:
                            osec = self._extract_ts_sec(op[0])
                            obh = int(float(op[1]))
                            if obh != last_bh:
                                last_bh = obh
                                last_bh_sec = osec
                        except ValueError:
                            pass
                if last_bh > 0:
                    self._last_block_height = last_bh
                    self._sec_since_block = float(np.clip(now_sec - self.buf_len - last_bh_sec, 0.0, 600.0))

            # Прогреваем 900с историю сделок до начала окна [now_sec - 1200 .. now_sec - buf_len]
            for s in range(now_sec - 1200, now_sec - self.buf_len + 1):
                sb, ss, _ = st_tr_Map.get(s, (0.0, 0.0, 0))
                self.st_delta_hist.append(sb - ss)
                self.st_vol_hist.append(sb + ss)
                bb, bs, _ = bo_tr_Map.get(s, (0.0, 0.0, 0))
                self.bo_delta_hist.append(bb - bs)
                self.bo_vol_hist.append(bb + bs)

            def _index_by_sec(lns: list[str]) -> list[tuple[int, str]]:
                out_l = []
                for ln in lns:
                    sc = self._extract_ts_sec(ln.split(",", 1)[0])
                    if sc > 0:
                        out_l.append((sc, ln))
                return out_l

            st_seq = _index_by_sec(st_lines)
            bo_seq = _index_by_sec(bo_lines)
            m_seq = _index_by_sec(m_lines)
            on_seq = _index_by_sec(on_lines)
            p_st = p_bo = p_m = p_on = 0

            d = dict(base_dict)
            if st_seq:
                self._parse_50lvl_line(st_seq[0][1], "ST", d)
            if bo_seq:
                self._parse_50lvl_line(bo_seq[0][1], "BO", d)
            if m_seq:
                self._parse_macro_line(m_seq[0][1], d)
            if on_seq:
                self._parse_onchain_line(on_seq[0][1], d, advance_timer=False)

            for s in range(now_sec - self.buf_len + 1, now_sec + 1):
                d = dict(d)
                while p_st < len(st_seq) and st_seq[p_st][0] <= s:
                    self._parse_50lvl_line(st_seq[p_st][1], "ST", d)
                    p_st += 1
                while p_bo < len(bo_seq) and bo_seq[p_bo][0] <= s:
                    self._parse_50lvl_line(bo_seq[p_bo][1], "BO", d)
                    p_bo += 1
                while p_m < len(m_seq) and m_seq[p_m][0] <= s:
                    self._parse_macro_line(m_seq[p_m][1], d)
                    p_m += 1
                on_updated = False
                while p_on < len(on_seq) and on_seq[p_on][0] <= s:
                    self._parse_onchain_line(on_seq[p_on][1], d, advance_timer=not on_updated)
                    on_updated = True
                    p_on += 1
                if not on_updated:
                    self._sec_since_block = min(600.0, self._sec_since_block + 1.0)
                    d["onchain_seconds_since_block"] = float(self._sec_since_block)

                for pfx, tmap, dhist, vhist in (
                    ("ST", st_tr_Map, self.st_delta_hist, self.st_vol_hist),
                    ("BO", bo_tr_Map, self.bo_delta_hist, self.bo_vol_hist),
                ):
                    bv, sv, cnt = tmap.get(s, (0.0, 0.0, 0))
                    tv = bv + sv
                    dv = bv - sv
                    dhist.append(dv)
                    vhist.append(tv)
                    da = list(dhist)
                    va = list(vhist)
                    c1 = float(sum(da[-60:]))
                    c5 = float(sum(da[-300:]))
                    c15 = float(sum(da))
                    v1 = float(sum(va))
                    d[f"{pfx}_trade_vol"] = tv
                    d[f"{pfx}_buy_vol"] = bv
                    d[f"{pfx}_sell_vol"] = sv
                    d[f"{pfx}_trade_count"] = float(cnt)
                    d[f"{pfx}_delta_vol"] = dv
                    d[f"{pfx}_cvd_1m"] = c1
                    d[f"{pfx}_cvd_5m"] = c5
                    d[f"{pfx}_cvd_15m"] = c15
                    d[f"{pfx}_trade_imbalance_1m"] = c1 / v1 if v1 > 0 else 0.0

                if "BO_bid_1_price" in d and "ST_ask_1_price" in d:
                    d["arb_st_buy_bo_sell"] = float(d["BO_bid_1_price"]) - float(d["ST_ask_1_price"])
                    d["arb_bo_buy_st_sell"] = float(d["ST_bid_1_price"]) - float(d["BO_ask_1_price"])
                    d["mid_price_diff"] = float(d.get("BO_mid_price", 0.0)) - float(d.get("ST_mid_price", 0.0))
                d["_ts"] = pd.Timestamp(s, unit="s", tz="UTC")
                self.raw_rows.append(d)

        for p, attr in ((self.st_trades_path, "_st_trade_pos"), (self.bo_trades_path, "_bo_trade_pos")):
            if p.exists():
                try:
                    setattr(self, attr, p.stat().st_size)
                except Exception:
                    pass

    def _strip_own_l2(self, d: dict, own_orders: dict):
        tol = float(own_orders.get("tick_size", 0.01)) * 0.25
        bid_p = float(own_orders.get("bid_p", 0.0) or 0.0)
        bid_q = float(own_orders.get("bid_q", 0.0) or 0.0)
        asks_own = own_orders.get("asks", [])
        if bid_q <= 0.0 and not any(q > 0.0 for _, q in asks_own):
            return
        bp = [float(d.get(f"ST_bid_{i}_price", 0.0)) for i in range(1, 51)]
        bq = [float(d.get(f"ST_bid_{i}_qty", 0.0)) for i in range(1, 51)]
        ap = [float(d.get(f"ST_ask_{i}_price", 0.0)) for i in range(1, 51)]
        aq = [float(d.get(f"ST_ask_{i}_qty", 0.0)) for i in range(1, 51)]
        b_clean = []
        for p, q in zip(bp, bq):
            if p <= 0:
                continue
            if bid_q > 0 and abs(p - bid_p) <= tol:
                q = max(0.0, q - bid_q)
            if q > 1e-4:
                b_clean.append((p, q))
        a_clean = []
        for p, q in zip(ap, aq):
            if p <= 0:
                continue
            for op, oq in asks_own:
                if oq > 0 and abs(p - op) <= tol:
                    q = max(0.0, q - oq)
            if q > 1e-4:
                a_clean.append((p, q))
        if not b_clean or not a_clean:
            return
        bp_arr = np.zeros(50, dtype=np.float64)
        bq_arr = np.zeros(50, dtype=np.float64)
        ap_arr = np.zeros(50, dtype=np.float64)
        aq_arr = np.zeros(50, dtype=np.float64)
        for i, (p, q) in enumerate(b_clean[:50]):
            bp_arr[i], bq_arr[i] = p, q
        for i, (p, q) in enumerate(a_clean[:50]):
            ap_arr[i], aq_arr[i] = p, q
        for i in range(50):
            d[f"ST_bid_{i+1}_price"], d[f"ST_bid_{i+1}_qty"] = bp_arr[i], bq_arr[i]
            d[f"ST_ask_{i+1}_price"], d[f"ST_ask_{i+1}_qty"] = ap_arr[i], aq_arr[i]
        self._recalc_l2_Book_scalars(d, "ST", bp_arr, bq_arr, ap_arr, aq_arr)
        d["_l2_cleaned"] = 1.0

    def step(self, rest_depth: dict, now_utc: datetime, own_orders: dict = None) -> pd.DataFrame:
        d = dict(self.raw_rows[-1])
        d["_ts"] = pd.Timestamp(now_utc)
        d["_l2_cleaned"] = 0.0

        # 1. Полный 50-уровневый стакан ST из горячего CSV + свежий BBO из REST
        st_tail = self._tail_lines(self.st_depth_path, n=1)
        if st_tail:
            self._parse_50lvl_line(st_tail[-1], "ST", d)

        bids = rest_depth.get("bids", []) if rest_depth else []
        asks = rest_depth.get("asks", []) if rest_depth else []
        if bids and asks:
            for lvl in range(1, min(51, max(len(bids), len(asks)) + 1)):
                if lvl <= len(bids):
                    d[f"ST_bid_{lvl}_price"] = float(bids[lvl - 1][0])
                    d[f"ST_bid_{lvl}_qty"] = float(bids[lvl - 1][1])
                if lvl <= len(asks):
                    d[f"ST_ask_{lvl}_price"] = float(asks[lvl - 1][0])
                    d[f"ST_ask_{lvl}_qty"] = float(asks[lvl - 1][1])
            bp = np.array([float(d.get(f"ST_bid_{i}_price", 0.0)) for i in range(1, 51)], dtype=np.float64)
            bq = np.array([float(d.get(f"ST_bid_{i}_qty", 0.0)) for i in range(1, 51)], dtype=np.float64)
            ap = np.array([float(d.get(f"ST_ask_{i}_price", 0.0)) for i in range(1, 51)], dtype=np.float64)
            aq = np.array([float(d.get(f"ST_ask_{i}_qty", 0.0)) for i in range(1, 51)], dtype=np.float64)
            self._recalc_l2_Book_scalars(d, "ST", bp, bq, ap, aq)
        if own_orders:
            self._strip_own_l2(d, own_orders)

        # 2. Горячий стакан BigONE + макро Binance + живые сделки
        bo_tail = self._tail_lines(self.bo_depth_path, n=1)
        if bo_tail:
            self._parse_50lvl_line(bo_tail[-1], "BO", d)
            bo_sec = self._extract_ts_sec(bo_tail[-1].split(",", 1)[0])
            now_s = int(now_utc.timestamp())
            d["BO_sec_since_tick"] = float(max(0, now_s - bo_sec)) if bo_sec > 0 else 0.0
        m_tail = self._tail_lines(self.macro_path, n=1)
        if m_tail:
            self._parse_macro_line(m_tail[-1], d)
        on_tail = self._tail_lines(self.onchain_path, n=1)
        if on_tail:
            self._parse_onchain_line(on_tail[-1], d, advance_timer=True)
        else:
            self._sec_since_block = min(600.0, self._sec_since_block + 1.0)
            d["onchain_seconds_since_block"] = float(self._sec_since_block)

        self._poll_trades(self.st_trades_path, "ST", d, "_st_trade_pos", self._seen_st_trade_ids, self.st_delta_hist, self.st_vol_hist)
        self._poll_trades(self.bo_trades_path, "BO", d, "_bo_trade_pos", self._seen_bo_trade_ids, self.bo_delta_hist, self.bo_vol_hist)

        # Кросс-биржевые мгновенные спреды
        if "BO_bid_1_price" in d and "ST_ask_1_price" in d:
            d["arb_st_buy_bo_sell"] = float(d["BO_bid_1_price"]) - float(d["ST_ask_1_price"])
            d["arb_bo_buy_st_sell"] = float(d["ST_bid_1_price"]) - float(d["BO_ask_1_price"])
            d["mid_price_diff"] = float(d.get("BO_mid_price", 0.0)) - float(d.get("ST_mid_price", 0.0))
            if float(d.get("BO_eff_buy_vwap_100", 0.0)) > 0 and float(d.get("ST_eff_sell_vwap_100", 0.0)) > 0:
                d["arb_eff_100_bo_buy_st_sell"] = float(d["ST_eff_sell_vwap_100"]) - float(d["BO_eff_buy_vwap_100"])
                d["arb_eff_100_pct"] = d["arb_eff_100_bo_buy_st_sell"] / float(d["BO_eff_buy_vwap_100"])
                d["arb_eff_300_bo_buy_st_sell"] = float(d["ST_eff_sell_vwap_300"]) - float(d["BO_eff_buy_vwap_300"])
                d["arb_eff_300_pct"] = d["arb_eff_300_bo_buy_st_sell"] / float(d["BO_eff_buy_vwap_300"])
                q_b = (100.0 * 0.998) / float(d["BO_eff_buy_vwap_100"])
                q_s = 100.0 / (float(d["ST_eff_sell_vwap_100"]) * 0.999)
                d["q_prl_accum_gain_100usd"] = q_b - q_s

        self.raw_rows.append(d)

        # 3. Векторный расчет банка фильтров W=[5, 20, 60, 300]с на окне буфера
        df_win = self._compute_window_features()
        return df_win.iloc[-1:].fillna(0.0).reset_index(drop=True)

    def _compute_window_features(self) -> pd.DataFrame:
        df_win = pd.DataFrame(list(self.raw_rows))
        df_win = df_win.set_index("_ts")
        ssb_saved = df_win["onchain_seconds_since_block"].copy() if "onchain_seconds_since_block" in df_win.columns else None
        if "ST_mid_price" in df_win.columns:
            df_win = add_exchange_filter_bank(df_win, "ST", coin=self.coin)
        if "BO_mid_price" in df_win.columns:
            df_win = add_exchange_filter_bank(df_win, "BO", coin=self.coin)
        df_win = add_cross_exchange_and_macro_filters(df_win, coin=self.coin)
        if ssb_saved is not None:
            df_win["onchain_seconds_since_block"] = ssb_saved
        if "onchain_mempool_size" in df_win.columns:
            mp = df_win["onchain_mempool_size"].astype(float)
            df_win["onchain_mempool_delta_5m"] = (mp - mp.shift(min(300, len(mp) - 1))).fillna(0.0)
        if "ST_microprice_offset_ticks" in df_win.columns:
            mo = df_win["ST_microprice_offset_ticks"].astype(float).clip(-5.0, 5.0)
            df_win["kinematic_v_wave"] = ((mo - mo.shift(3)).fillna(0.0) / 3.0).clip(-2.0, 2.0)
        return df_win

    def get_recent_window(self, n: int = 60) -> pd.DataFrame:
        """Возвращает последние n строк с полным набором рассчитанных признаков для инициализации X_buf."""
        df_win = self._compute_window_features()
        return df_win.iloc[-n:].fillna(0.0).reset_index(drop=True)

    def peek_fast_bbo(self, rest_depth: dict = None) -> dict:
        """Субсекундное чтение (<0.3 мс) лучших цен/объемов ST и BO для внутрисекундного сторожа очереди."""
        last_d = self.raw_rows[-1] if self.raw_rows else {}
        b1 = float(last_d.get("ST_bid_1_price", last_d.get("ST_best_bid", 0.0)))
        b2 = float(last_d.get("ST_bid_2_price", 0.0))
        a1 = float(last_d.get("ST_ask_1_price", last_d.get("ST_best_ask", 0.0)))
        b1_q = float(last_d.get("ST_bid_1_qty", 0.0))
        b2_q = float(last_d.get("ST_bid_2_qty", 0.0))
        a1_q = float(last_d.get("ST_ask_1_qty", 0.0))
        bo_b1 = float(last_d.get("BO_bid_1_price", 0.0))

        st_tail = self._tail_lines(self.st_depth_path, n=1)
        if st_tail:
            parts = st_tail[-1].split(",")
            if len(parts) > 103:
                try:
                    b1 = float(parts[1] or b1)
                    b1_q = float(parts[2] or b1_q)
                    b2 = float(parts[3] or b2)
                    b2_q = float(parts[4] or b2_q)
                    a1 = float(parts[101] or a1)
                    a1_q = float(parts[102] or a1_q)
                except ValueError:
                    pass
        if rest_depth:
            bids = rest_depth.get("bids", [])
            asks = rest_depth.get("asks", [])
            if bids and asks:
                try:
                    b1, b1_q = float(bids[0][0]), float(bids[0][1])
                    if len(bids) > 1:
                        b2, b2_q = float(bids[1][0]), float(bids[1][1])
                    a1, a1_q = float(asks[0][0]), float(asks[0][1])
                except (ValueError, IndexError):
                    pass
        bo_tail = self._tail_lines(self.bo_depth_path, n=1)
        if bo_tail:
            b_parts = bo_tail[-1].split(",")
            if len(b_parts) > 3:
                try:
                    bo_b1 = float(b_parts[1] or bo_b1)
                except ValueError:
                    pass
        return {"b1": b1, "b2": b2, "a1": a1, "b1_q": b1_q, "b2_q": b2_q, "a1_q": a1_q, "bo_b1": bo_b1}



