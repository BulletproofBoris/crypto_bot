#!/usr/bin/env python3
"""
Этап 1–4: Предрасчёт матрицы триады (H=5s, 12s, 25s) и обучение RL-агента
микроструктурного исполнения (Maskable PPO + Behavioral Cloning Warm-Start)
поверх 10-уровневого L2 FIFO-стакана SafeTrade.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.swarm_router import SwarmRegimeRouter
from _tools.data_labeler import COIN_CONFIG


def build_or_load_rl_cache(coin: str = "prl", tail_hours: float = 115.0, force_rebuild: bool = False) -> dict:
    """
    Этап 1: Предрасчёт и кэширование сигналов триады ансамблей (H=5s, 12s, 25s)
    и 10 уровней стакана L2 для мгновенной симуляции RL-среды.
    """
    coin = coin.lower()
    cache_dir = BASE_DIR / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rl_cache_path = cache_dir / f"rl_triad_lob_cache_{coin}_{int(tail_hours)}h.npz"

    if rl_cache_path.exists() and not force_rebuild:
        print(f"⚡ Загрузка готового кэша RL-полигона: {rl_cache_path.name}")
        data = np.load(rl_cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    parquet_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    if not parquet_path.exists():
        parquet_path = BASE_DIR / "data" / "features" / f"{coin}_1s_features.parquet"

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    n_rows = min(total_rows, int(tail_hours * 3600))
    start_row = max(0, total_rows - n_rows)

    print(f"📂 Чтение последних {n_rows:,} сек ({tail_hours:.1f} ч) из {parquet_path.name}...")
    table = pf.read()
    df = table.slice(start_row, n_rows).to_pandas()
    del table

    horizons = [(5, 3.0), (12, 3.5), (25, 3.0)]
    triad_feats = []

    for H, K in horizons:
        print(f"\n🔮 Расчёт ансамбля H={H}s (K={K})...")
        router = SwarmRegimeRouter(coin, H, K, temperature=0.5, use_neural=True, neural_weight=0.15)
        X_df = pd.DataFrame(index=df.index)
        for col in router.feature_names:
            X_df[col] = df[col].values if col in df.columns else 0.0
        X = np.nan_to_num(X_df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        del X_df

        probs, ez, p_buy, p_sell, _ = router.predict_ensemble(X, df, mode="router")
        del X, router

        p_up = probs[:, 4] + probs[:, 5] + probs[:, 6]
        p_down = probs[:, 0] + probs[:, 1] + probs[:, 2]
        p_tail_pos = probs[:, 5] + probs[:, 6]
        p_tail_neg = probs[:, 0] + probs[:, 1]
        entropy = -np.sum(np.clip(probs, 1e-7, 1.0) * np.log(np.clip(probs, 1e-7, 1.0)), axis=1) / np.log(7.0)

        triad_feats.append(np.column_stack([ez, p_up, p_down, p_tail_pos, p_tail_neg, entropy]).astype(np.float32))

    # 18 признаков триады (3 горизонта x 6 метрик)
    X_triad = np.hstack(triad_feats).astype(np.float32)

    # 10 уровней стакана SafeTrade
    tick_size = float(COIN_CONFIG[coin]["tick_size"])
    bid_p = np.zeros((len(df), 10), dtype=np.float64)
    bid_q = np.zeros((len(df), 10), dtype=np.float64)
    ask_p = np.zeros((len(df), 10), dtype=np.float64)
    ask_q = np.zeros((len(df), 10), dtype=np.float64)

    for lvl in range(1, 11):
        bp_col, bq_col = f"ST_bid_{lvl}_price", f"ST_bid_{lvl}_qty"
        ap_col, aq_col = f"ST_ask_{lvl}_price", f"ST_ask_{lvl}_qty"
        if lvl == 1:
            bid_p[:, 0] = df[bp_col].values if bp_col in df.columns else df["ST_best_bid"].values
            bid_q[:, 0] = df[bq_col].values if bq_col in df.columns else df["ST_bid_qty_1"].values
            ask_p[:, 0] = df[ap_col].values if ap_col in df.columns else df["ST_best_ask"].values
            ask_q[:, 0] = df[aq_col].values if aq_col in df.columns else df["ST_ask_qty_1"].values
        else:
            if bp_col in df.columns:
                bid_p[:, lvl - 1] = np.nan_to_num(df[bp_col].values, nan=0.0)
                bid_q[:, lvl - 1] = np.nan_to_num(df[bq_col].values, nan=0.0)
            if ap_col in df.columns:
                ask_p[:, lvl - 1] = np.nan_to_num(df[ap_col].values, nan=0.0)
                ask_q[:, lvl - 1] = np.nan_to_num(df[aq_col].values, nan=0.0)

    # 10 микроструктурных признаков стакана
    spread_ticks = np.clip((ask_p[:, 0] - bid_p[:, 0]) / tick_size, 0.0, 50.0)
    b1_q = np.maximum(bid_q[:, 0], 0.0)
    a1_q = np.maximum(ask_q[:, 0], 0.0)
    imb_l1 = (b1_q - a1_q) / (b1_q + a1_q + 1e-6)
    b5_q = np.sum(np.maximum(bid_q[:, :5], 0.0), axis=1)
    a5_q = np.sum(np.maximum(ask_q[:, :5], 0.0), axis=1)
    imb_l5 = (b5_q - a5_q) / (b5_q + a5_q + 1e-6)
    b10_q = np.sum(np.maximum(bid_q, 0.0), axis=1)
    a10_q = np.sum(np.maximum(ask_q, 0.0), axis=1)
    imb_l10 = (b10_q - a10_q) / (b10_q + a10_q + 1e-6)

    log_b1 = np.log1p(b1_q) / 10.0
    log_a1 = np.log1p(a1_q) / 10.0
    log_b5 = np.log1p(b5_q) / 12.0
    log_a5 = np.log1p(a5_q) / 12.0
    vol_300 = np.nan_to_num(df["ST_vol_300s_ticks"].values if "ST_vol_300s_ticks" in df.columns else np.ones(len(df)), nan=1.0)
    ofi_300 = np.nan_to_num(df["ST_ofi_norm_300s"].values if "ST_ofi_norm_300s" in df.columns else np.zeros(len(df)), nan=0.0)

    X_lob = np.column_stack([
        spread_ticks / 10.0, imb_l1, imb_l5, imb_l10,
        log_b1, log_a1, log_b5, log_a5,
        np.clip(vol_300 / 5.0, 0.0, 5.0), np.clip(ofi_300, -3.0, 3.0)
    ]).astype(np.float32)

    X_market = np.nan_to_num(np.hstack([X_triad, X_lob]), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)

    trade_vol = np.nan_to_num(df["ST_trade_vol_1s"].values if "ST_trade_vol_1s" in df.columns else np.zeros(len(df)), nan=0.0)
    buy_vol = np.nan_to_num(df["ST_buy_vol_1s"].values if "ST_buy_vol_1s" in df.columns else np.zeros(len(df)), nan=0.0)
    sell_vol = np.maximum(0.0, trade_vol - buy_vol)
    vwap = np.nan_to_num(df["ST_vwap_1s"].values if "ST_vwap_1s" in df.columns else (bid_p[:, 0] + ask_p[:, 0]) * 0.5, nan=0.0)
    ts_arr = df["timestamp"].astype(str).values

    payload = {
        "X_market": X_market,
        "bid_p": bid_p,
        "bid_q": bid_q,
        "ask_p": ask_p,
        "ask_q": ask_q,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "vwap": vwap,
        "ts": ts_arr,
        "tick_size": np.array(tick_size, dtype=np.float64),
    }
    np.savez_compressed(rl_cache_path, **payload)
    print(f"✅ Кэш RL-полигона сохранён в {rl_cache_path} ({X_market.shape[0]:,} строк x {X_market.shape[1]} признаков)")
    return payload



class SafeTradeL2ExecutionEnv:
    """
    Быстрая 10-уровневая L2 FIFO среда исполнения заявок для обучения RL-агента.
    Действия (7 шт.):
      0: HOLD / WAIT (сохранять текущую позицию и место q_ahead в очереди FIFO)
      1: QUOTE_BID_BEST (лимитка на покупку на best_bid или +1 тик внутри спреда)
      2: QUOTE_BID_PASSIVE (лимитка на покупку на best_bid - 1 тик)
      3: QUOTE_ASK_BEST (лимитный выход на best_ask)
      4: QUOTE_ASK_AGGRESSIVE (лимитный выход на best_ask - 1 тик)
      5: CANCEL_ORDERS (снять активную лимитку)
      6: PANIC_TAKER_EXIT (мгновенный выход по рынку через 10 уровней стакана)
    """

    def __init__(self, data: dict, start_idx: int, end_idx: int, episode_len: int = 3600, order_usd: float = 15.0):
        self.X_market = data["X_market"]
        self.bid_p = data["bid_p"]
        self.bid_q = data["bid_q"]
        self.ask_p = data["ask_p"]
        self.ask_q = data["ask_q"]
        self.buy_vol = data["buy_vol"]
        self.sell_vol = data["sell_vol"]
        self.vwap = data["vwap"]
        self.tick_size = float(data["tick_size"])

        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.episode_len = int(episode_len)
        self.order_usd = float(order_usd)
        self.obs_dim = self.X_market.shape[1] + 10
        self.n_actions = 7
        self.reset()

    def _get_lob_queue(self, t: int, side: str, price: float):
        tol = self.tick_size * 0.25
        q_at, q_better = 0.0, 0.0
        if side == "BUY":
            for lvl in range(10):
                p, q = self.bid_p[t, lvl], self.bid_q[t, lvl]
                if p <= 0 or q <= 0:
                    continue
                if abs(p - price) <= tol:
                    q_at += q
                elif p > price + tol:
                    q_better += q
        else:
            for lvl in range(10):
                p, q = self.ask_p[t, lvl], self.ask_q[t, lvl]
                if p <= 0 or q <= 0:
                    continue
                if abs(p - price) <= tol:
                    q_at += q
                elif p < price - tol:
                    q_better += q
        return q_at, q_better

    def _sweep_taker(self, t: int, qty: float):
        rem = qty
        proceeds = 0.0
        filled = 0.0
        for lvl in range(10):
            p, q = self.bid_p[t, lvl], self.bid_q[t, lvl]
            if p <= 0 or q <= 0:
                continue
            take = min(rem, q)
            proceeds += take * p
            filled += take
            rem -= take
            if rem <= 1e-8:
                break
        if rem > 1e-8:
            fb_p = max(self.tick_size, self.bid_p[t, 0] - 10 * self.tick_size)
            proceeds += rem * fb_p
            filled += rem
        return proceeds / max(filled, 1e-8)

    def reset(self, eval_mode: bool = False):
        if eval_mode or (self.end_idx - self.start_idx <= self.episode_len + 10):
            self.t = self.start_idx
            self.max_t = self.end_idx - 1
        else:
            self.t = int(np.random.randint(self.start_idx, self.end_idx - self.episode_len - 1))
            self.max_t = self.t + self.episode_len
        self.pos_qty = 0.0
        self.avg_entry = 0.0
        self.entry_t = 0
        self.order_side = None
        self.order_price = 0.0
        self.order_qty = 0.0
        self.q_ahead = 0.0
        self.active_after = 0
        self.prev_mtm = 0.0
        self.total_pnl_usd = 0.0
        self.n_trades = 0
        self.cooldown_until = 0
        return self._get_obs()

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True  # HOLD всегда разрешён
        if self.pos_qty <= 1e-8:
            in_cooldown = self.t < self.cooldown_until
            imb_l5 = float(self.X_market[self.t, 20])
            if not in_cooldown:
                if imb_l5 > -0.75:
                    mask[1] = True  # QUOTE_BID_BEST (блокируется при обвале стакана imb_l5 <= -0.75)
                mask[2] = True  # QUOTE_BID_PASSIVE
            if self.order_side is not None:
                mask[5] = True  # CANCEL_ORDERS
        else:
            mask[3] = True  # QUOTE_ASK_BEST
            mask[4] = True  # QUOTE_ASK_AGGRESSIVE
            mask[6] = True  # PANIC_TAKER_EXIT
            if self.order_side is not None:
                mask[5] = True
        return mask

    def _get_obs(self) -> np.ndarray:
        b1 = max(self.bid_p[self.t, 0], self.tick_size)
        a1 = max(self.ask_p[self.t, 0], b1 + self.tick_size)
        has_pos = 1.0 if self.pos_qty > 1e-8 else 0.0
        unr_ticks = ((b1 - self.avg_entry) / self.tick_size) if has_pos else 0.0
        hold_ratio = min((self.t - self.entry_t) / 45.0, 2.0) if has_pos else 0.0
        has_bid = 1.0 if self.order_side == "BUY" else 0.0
        has_ask = 1.0 if self.order_side == "SELL" else 0.0
        if has_bid:
            dist = (b1 - self.order_price) / self.tick_size
        elif has_ask:
            dist = (self.order_price - a1) / self.tick_size
        else:
            dist = 0.0
        q_ratio = self.q_ahead / (self.q_ahead + self.order_qty + 1e-6) if self.order_side else 0.0
        xm = self.X_market[self.t]
        ez_5, p_tn_5 = xm[0], xm[4]
        ez_12, p_dn_12 = xm[6], xm[8]
        ez_25 = xm[12]
        spread_ticks = xm[18] * 10.0

        # Индикатор ошибки прогноза ансамбля в реальном времени (Surprise / Model Forecast Error за последние 12с)
        t_prev = max(0, self.t - 12)
        b1_prev = max(self.bid_p[t_prev, 0], self.tick_size)
        actual_dt_12 = (b1 - b1_prev) / self.tick_size
        pred_dt_12 = float(self.X_market[t_prev, 6]) * 4.0
        model_err_12 = np.clip(actual_dt_12 - pred_dt_12, -10.0, 10.0)
        surprise_pen = max(0.0, -model_err_12) * 0.6

        # Количественная оценка ожидаемой будущей выгоды с поправкой на ошибку моделей
        exp_entry_edge = (0.5 * ez_25 + 0.35 * ez_12 + 0.15 * ez_5) * 4.0 + 0.5 * spread_ticks - p_tn_5 * 3.0 - surprise_pen
        exp_hold_edge = (0.6 * ez_12 + 0.4 * ez_5) * 4.0 - p_dn_12 * spread_ticks - surprise_pen * 0.8
        state_vec = np.array([
            has_pos, np.clip(unr_ticks / 10.0, -3.0, 3.0), hold_ratio,
            has_bid, has_ask, np.clip(dist / 5.0, -2.0, 2.0), q_ratio,
            np.clip(model_err_12 / 5.0, -3.0, 3.0),
            np.clip(exp_entry_edge / 5.0, -3.0, 3.0), np.clip(exp_hold_edge / 5.0, -3.0, 3.0)
        ], dtype=np.float32)
        return np.concatenate([xm, state_vec])

    def step(self, action: int):
        t = self.t
        b1 = max(self.bid_p[t, 0], self.tick_size)
        a1 = max(self.ask_p[t, 0], b1 + self.tick_size)
        tol = self.tick_size * 0.25
        requote_penalty = 0.0

        # 1. Обработка выбранного действия агента (с задержкой сети latency=1s)
        if action in (1, 2) and self.pos_qty <= 1e-8:
            target_p = (b1 + self.tick_size if (a1 - b1) >= 2.0 * self.tick_size else b1) if action == 1 else max(self.tick_size, b1 - self.tick_size)
            if self.order_side != "BUY" or abs(self.order_price - target_p) > tol:
                if self.order_side is not None:
                    requote_penalty = 0.02
                self.order_side = "BUY"
                self.order_price = round(target_p / self.tick_size) * self.tick_size
                self.order_qty = self.order_usd / self.order_price
                self.q_ahead = -1.0
                self.active_after = t + 1
        elif action in (3, 4) and self.pos_qty > 1e-8:
            target_p = a1 if action == 3 else max(b1 + self.tick_size, a1 - self.tick_size)
            if self.order_side != "SELL" or abs(self.order_price - target_p) > tol:
                if self.order_side is not None:
                    requote_penalty = 0.02
                self.order_side = "SELL"
                self.order_price = round(target_p / self.tick_size) * self.tick_size
                self.order_qty = self.pos_qty
                self.q_ahead = -1.0
                self.active_after = t + 1
        elif action == 5:
            if self.order_side is not None:
                self.order_side = None
                requote_penalty = 0.01
        elif action == 6 and self.pos_qty > 1e-8:
            exec_p = self._sweep_taker(t, self.pos_qty)
            net_pnl = self.pos_qty * (exec_p * (1.0 - 0.001) - self.avg_entry)
            self.total_pnl_usd += net_pnl
            self.pos_qty = 0.0
            self.avg_entry = 0.0
            self.order_side = None
            if net_pnl < 0:
                self.cooldown_until = t + 30  # 30с защита от повторного входа на падающем ноже

        # 2. Переход к следующей секунде t+1 и симуляция очереди L2 FIFO
        self.t += 1
        nt = self.t
        nb1 = max(self.bid_p[nt, 0], self.tick_size)
        na1 = max(self.ask_p[nt, 0], nb1 + self.tick_size)

        if self.order_side is not None and nt >= self.active_after:
            q_at, q_better = self._get_lob_queue(nt, self.order_side, self.order_price)
            if self.q_ahead < 0.0:
                self.q_ahead = q_at
            else:
                self.q_ahead = min(self.q_ahead, q_at)
            eff_q = self.q_ahead + q_better

            if self.order_side == "BUY":
                crossed = (na1 <= self.order_price + tol) or (self.vwap[nt] > 0 and self.vwap[nt] < self.order_price - tol)
                vol_hit = self.sell_vol[nt] if abs(nb1 - self.order_price) <= tol else 0.0
                if crossed or vol_hit > eff_q:
                    fill_q = self.order_qty if crossed else min(self.order_qty, max(0.0, vol_hit - eff_q))
                    if fill_q >= self.order_qty * 0.25:
                        self.pos_qty = self.order_qty
                        self.avg_entry = self.order_price
                        self.entry_t = nt
                        self.order_side = None
                        self.n_trades += 1
                elif vol_hit > 0:
                    self.q_ahead = max(0.0, self.q_ahead - max(0.0, vol_hit - q_better))
            elif self.order_side == "SELL":
                crossed = (nb1 >= self.order_price - tol) or (self.vwap[nt] > 0 and self.vwap[nt] > self.order_price + tol)
                vol_hit = self.buy_vol[nt] if abs(na1 - self.order_price) <= tol else 0.0
                if crossed or vol_hit > eff_q:
                    net_pnl = self.pos_qty * (self.order_price - self.avg_entry)
                    self.total_pnl_usd += net_pnl
                    self.pos_qty = 0.0
                    self.avg_entry = 0.0
                    self.order_side = None
                elif vol_hit > 0:
                    self.q_ahead = max(0.0, self.q_ahead - max(0.0, vol_hit - q_better))

        cur_mtm = self.total_pnl_usd + (self.pos_qty * (nb1 - self.avg_entry) if self.pos_qty > 1e-8 else 0.0)
        delta_pnl_ticks = (cur_mtm - self.prev_mtm) / (self.order_usd * (self.tick_size / nb1))
        self.prev_mtm = cur_mtm

        # Асимметричная неприязнь к риску (Downside Risk Aversion): глубокий минус штрафуется в 1.75x сильнее
        shaped_pnl = delta_pnl_ticks * 1.75 if delta_pnl_ticks < -0.5 else delta_pnl_ticks
        hold_sec = (nt - self.entry_t) if self.pos_qty > 1e-8 else 0
        inv_pen = 0.003 * (1.0 if self.pos_qty > 1e-8 else 0.0) + (0.01 if hold_sec > 35 else 0.0)
        reward = float(shaped_pnl - inv_pen - requote_penalty)
        done = bool(nt >= self.max_t)
        return self._get_obs(), reward, done


class MaskableActorCritic(nn.Module):
    """Нейросеть политики и ценности (Actor-Critic) с маскированием недопустимых действий."""

    def __init__(self, obs_dim: int, n_actions: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        h = self.backbone(obs)
        logits = self.actor(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        val = self.critic(h).squeeze(-1)
        return logits, val


def teacher_heuristic_action(obs: np.ndarray, mask: np.ndarray) -> int:
    """Эвристическая политика для Behavioral Cloning Warm-Start перед запуском PPO."""
    ez_5, p_up_5, p_dn_5, p_tp_5, p_tn_5 = obs[0], obs[1], obs[2], obs[3], obs[4]
    ez_12, p_up_12, p_dn_12 = obs[6], obs[7], obs[8]
    ez_25 = obs[12]
    has_pos = obs[-10] > 0.5
    unr_ticks = obs[-9] * 10.0
    hold_ratio = obs[-8]
    has_bid = obs[-7] > 0.5
    has_ask = obs[-6] > 0.5
    model_err_12 = obs[-3] * 5.0
    exp_entry_edge = obs[-2] * 5.0
    exp_hold_edge = obs[-1] * 5.0

    if not has_pos:
        strong_buy = (exp_entry_edge > 0.65 and model_err_12 > -2.5 and ez_12 > 0.08 and ez_5 > 0.04 and p_tn_5 < 0.25)
        dip_buy = (exp_entry_edge > 0.55 and model_err_12 > -2.5 and ez_25 > 0.10 and ez_12 > 0.06 and ez_5 < 0.02)
        if strong_buy and mask[1]:
            return 0 if has_bid else 1
        if dip_buy and mask[2]:
            return 0 if has_bid else 2
        if has_bid and (exp_entry_edge < 0.10 or model_err_12 < -2.0 or ez_12 < -0.04 or p_tn_5 > 0.35) and mask[5]:
            return 5
        return 0
    else:
        if (unr_ticks <= -3.0 or model_err_12 < -3.5 or (ez_12 < -0.15 and ez_5 < -0.15)) and mask[6]:
            return 6
        if (hold_ratio > 0.8 or exp_hold_edge < -0.25 or ez_12 < -0.05) and mask[4]:
            return 4
        if mask[3]:
            return 0 if has_ask else 3
        return 0


def evaluate_policy(env: SafeTradeL2ExecutionEnv, model: MaskableActorCritic = None, device: str = "cpu") -> dict:
    """Прогоняет детерминированный бэктест на всём интервале среды [start_idx, end_idx]."""
    obs = env.reset(eval_mode=True)
    done = False
    steps = 0
    entry_edges = []
    entry_v_preds = []
    while not done:
        mask = env.get_action_mask()
        v_pred = 0.0
        if model is None:
            act = teacher_heuristic_action(obs, mask)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                logits, v_t = model(obs_t, mask_t)
                act = int(torch.argmax(logits, dim=-1).item())
                v_pred = float(v_t.item())
        if act in (1, 2) and env.pos_qty <= 1e-8:
            entry_edges.append(float(obs[-2] * 5.0))
            entry_v_preds.append(v_pred)
        obs, _, done = env.step(act)
        steps += 1
    avg_real_pnl_usd = float(env.prev_mtm / max(1, env.n_trades))
    return {
        "pnl_usd": round(float(env.prev_mtm), 4),
        "pnl_pct": round(float(env.prev_mtm / env.order_usd * 100.0), 2),
        "trades": int(env.n_trades),
        "hours": round(steps / 3600.0, 2),
        "exp_edge_ticks": round(float(np.mean(entry_edges)) if entry_edges else 0.0, 2),
        "critic_v_ticks": round(float(np.mean(entry_v_preds)) if entry_v_preds else 0.0, 2),
        "real_pnl_per_trade_usd": round(avg_real_pnl_usd, 4),
    }


def train_rl_agent(coin: str = "prl", tail_hours: float = 115.0, ppo_epochs: int = 25, steps_per_epoch: int = 16384):
    data = build_or_load_rl_cache(coin=coin, tail_hours=tail_hours)
    n_total = len(data["X_market"])
    split_idx = int(n_total * 0.65)  # 65% (~75 ч) RL-Train, 35% (~40 ч) Blind OOS RL-Test

    train_env = SafeTradeL2ExecutionEnv(data, start_idx=60, end_idx=split_idx, episode_len=3600)
    test_env = SafeTradeL2ExecutionEnv(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=n_total - split_idx - 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskableActorCritic(obs_dim=train_env.obs_dim, n_actions=train_env.n_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    base_res = evaluate_policy(test_env, model=None, device=device)
    print(f"\n📊 Базовая эвристика на слепом OOS ({base_res['hours']} ч): PnL = ${base_res['pnl_usd']:+.4f} ({base_res['pnl_pct']:+.2f}%) | Сделок: {base_res['trades']}")

    # 1. Behavioral Cloning Warm-Start (инициализация от учителя)
    print("🎓 Behavioral Cloning Warm-Start (сбор 30,000 демонстраций учителя)...")
    bc_obs, bc_masks, bc_acts = [], [], []
    obs = train_env.reset()
    for _ in range(30000):
        mask = train_env.get_action_mask()
        act = teacher_heuristic_action(obs, mask)
        bc_obs.append(obs)
        bc_masks.append(mask)
        bc_acts.append(act)
        obs, _, done = train_env.step(act)
        if done:
            obs = train_env.reset()

    t_obs = torch.as_tensor(np.array(bc_obs), dtype=torch.float32, device=device)
    t_masks = torch.as_tensor(np.array(bc_masks), dtype=torch.bool, device=device)
    t_acts = torch.as_tensor(np.array(bc_acts), dtype=torch.long, device=device)
    for _ in range(8):
        perm = torch.randperm(len(t_obs), device=device)
        for i in range(0, len(t_obs), 1024):
            idx = perm[i:i + 1024]
            logits, _ = model(t_obs[idx], t_masks[idx])
            loss = F.cross_entropy(logits, t_acts[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 2. Обучение Maskable PPO в L2 FIFO среде
    print(f"🚀 Запуск Maskable PPO ({ppo_epochs} эпох по {steps_per_epoch:,} шагов)...")
    best_pnl = -1e9
    out_dir = BASE_DIR / "models" / coin.lower() / "rl_execution"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, ppo_epochs + 1):
        b_obs, b_masks, b_acts, b_logp, b_rews, b_vals, b_dones = [], [], [], [], [], [], []
        obs = train_env.reset()
        for _ in range(steps_per_epoch):
            mask = train_env.get_action_mask()
            with torch.no_grad():
                o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                logits, val = model(o_t, m_t)
                dist = Categorical(logits=logits)
                act_t = dist.sample()
                logp_t = dist.log_prob(act_t)
            act = int(act_t.item())
            next_obs, rew, done = train_env.step(act)
            b_obs.append(obs); b_masks.append(mask); b_acts.append(act)
            b_logp.append(float(logp_t.item())); b_rews.append(rew); b_vals.append(float(val.item())); b_dones.append(done)
            obs = train_env.reset() if done else next_obs

        # GAE (Generalized Advantage Estimation)
        advs = np.zeros(steps_per_epoch, dtype=np.float32)
        last_gae = 0.0
        for t_i in reversed(range(steps_per_epoch)):
            next_val = 0.0 if (t_i == steps_per_epoch - 1 or b_dones[t_i]) else b_vals[t_i + 1]
            delta = b_rews[t_i] + 0.995 * next_val - b_vals[t_i]
            last_gae = delta + 0.995 * 0.95 * (0.0 if b_dones[t_i] else last_gae)
            advs[t_i] = last_gae
        rets = advs + np.array(b_vals, dtype=np.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        T_obs = torch.as_tensor(np.array(b_obs), dtype=torch.float32, device=device)
        T_masks = torch.as_tensor(np.array(b_masks), dtype=torch.bool, device=device)
        T_acts = torch.as_tensor(np.array(b_acts), dtype=torch.long, device=device)
        T_old_logp = torch.as_tensor(np.array(b_logp), dtype=torch.float32, device=device)
        T_advs = torch.as_tensor(advs, dtype=torch.float32, device=device)
        T_rets = torch.as_tensor(rets, dtype=torch.float32, device=device)

        for _ in range(4):
            perm = torch.randperm(steps_per_epoch, device=device)
            for i in range(0, steps_per_epoch, 1024):
                idx = perm[i:i + 1024]
                logits, vals = model(T_obs[idx], T_masks[idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(T_acts[idx])
                ratio = torch.exp(new_logp - T_old_logp[idx])
                surr1 = ratio * T_advs[idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * T_advs[idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(vals, T_rets[idx]) - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        oos_res = evaluate_policy(test_env, model=model, device=device)
        is_best = oos_res["pnl_usd"] > best_pnl
        if is_best:
            best_pnl = oos_res["pnl_usd"]
            torch.save(model.state_dict(), out_dir / "best_maskable_ppo.pt")
        mark = "🏆 РЕКОРД" if is_best else "✅ OK"
        print(
            f"  [PPO Ep {ep:02d}/{ppo_epochs}] {mark} -> Blind OOS PnL: ${oos_res['pnl_usd']:+.4f} ({oos_res['pnl_pct']:+.2f}%) | "
            f"Сделок: {oos_res['trades']} | Ожид.Edge: {oos_res['exp_edge_ticks']:+.2f}т (V={oos_res['critic_v_ticks']:+.2f}) -> "
            f"Факт/сделку: ${oos_res['real_pnl_per_trade_usd']:+.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="prl")
    parser.add_argument("--tail-hours", type=float, default=115.0)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    train_rl_agent(coin=args.coin, tail_hours=args.tail_hours, ppo_epochs=args.epochs)
