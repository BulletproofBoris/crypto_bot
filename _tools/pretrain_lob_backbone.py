#!/usr/bin/env python3
"""
_tools/pretrain_lob_backbone.py
=============================================================================
Этап 1 глубокого мультизадачного обучения v8.0:
Сквозное предобучение (Pretraining) Spatial-Temporal Backbone на всех 2,193,263 тиках
(609.2 часов) L2 книги стакана на NVIDIA GeForce RTX 5060 Ti.

Задачи предобучения (Auxiliary Multi-Task Learning):
1. Head Sweep 12s: прогнозирование микро-выноса ликвидности (Focal Loss, gamma=2.0);
2. Head Bleed 60s: прогнозирование ступенчатого сползания (Focal Loss, gamma=2.0);
3. Head Crash 300s: прогнозирование водопадного обвала стакана (Focal Loss, gamma=2.0);
4. Head Adverse 30s: прогнозирование зависания в просадке (Focal Loss, gamma=2.0).

Результат:
Spatial 1D CNN и Causal GRU (latent_dim=256) формируют глубокие нелинейные представления
о форме стакана LOB и потоке сделок Z_t, готовые для устойчивого RL-исполнения.
=============================================================================
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.multi_horizon_guard_net import FocalLoss

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
CKPT_DIR = BASE_DIR / "models" / "prl" / "multitask_lob"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class SequenceLOBDataset(Dataset):
    """
    Датасет последовательностей L2 стакана и потока сделок (seq_len=32)
    для обучения рекуррентного Causal GRU на всем 609h массиве.
    """

    def __init__(self, lob_10: np.ndarray, flow_16: np.ndarray, targets: dict, seq_len: int = 32):
        self.lob_10 = torch.from_numpy(lob_10.astype(np.float32))
        self.flow_16 = torch.from_numpy(flow_16.astype(np.float32))
        self.y_sw = torch.from_numpy(targets["sweep"].astype(np.float32))
        self.y_bl = torch.from_numpy(targets["bleed"].astype(np.float32))
        self.y_cr = torch.from_numpy(targets["crash"].astype(np.float32))
        self.y_ex = torch.from_numpy(targets["exit"].astype(np.float32))
        self.seq_len = seq_len
        self.n_samples = len(self.lob_10) - seq_len - 1

    def __len__(self):
        return self.n_samples // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return (
            self.lob_10[start:end],
            self.flow_16[start:end],
            self.y_sw[start:end],
            self.y_bl[start:end],
            self.y_cr[start:end],
            self.y_ex[start:end],
        )


def build_lob_10_and_flow_tensors(raw: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формирует тензор L2 стакана (N, 10, 4) и тензор потока ордеров (N, 16):
    LOB: [bid_p_diff_ticks, log_bid_q, ask_p_diff_ticks, log_ask_q] для уровней 1..10.
    Flow: [spread_ticks, spread_pct, ofi, cvd, vel10, vel3, vel180, dd300, pos3600, rng14400, v_buy, v_sell, ...]
    """
    b_p = raw["bid_p"]  # (N, 10)
    b_q = raw["bid_q"]  # (N, 10)
    a_p = raw["ask_p"]  # (N, 10)
    a_q = raw["ask_q"]  # (N, 10)
    tick = float(raw["tick_size"])
    n = len(b_p)

    # 1. Формируем LOB (N, 10, 4)
    mid_p = (b_p[:, 0] + a_p[:, 0]) / 2.0
    lob = np.zeros((n, 10, 4), dtype=np.float32)
    for lvl in range(10):
        lob[:, lvl, 0] = np.clip((mid_p - b_p[:, lvl]) / tick, 0.0, 50.0) / 10.0
        lob[:, lvl, 1] = np.log1p(np.maximum(0.0, b_q[:, lvl])) / 10.0
        lob[:, lvl, 2] = np.clip((a_p[:, lvl] - mid_p) / tick, 0.0, 50.0) / 10.0
        lob[:, lvl, 3] = np.log1p(np.maximum(0.0, a_q[:, lvl])) / 10.0

    # 2. Формируем Flow & Macro (N, 16)
    v_buy = raw["buy_vol"]
    v_sell = raw["sell_vol"]
    s_b1 = b_p[:, 0]
    high_300 = np.maximum.accumulate(s_b1) # rolling max proxy
    dd_300 = np.clip((high_300 - s_b1) / tick, 0.0, 20.0) / 10.0

    flow = np.zeros((n, 16), dtype=np.float32)
    flow[:, 0] = np.clip((a_p[:, 0] - b_p[:, 0]) / tick, 1.0, 10.0) / 5.0
    flow[:, 1] = (b_q[:, 0] - a_q[:, 0]) / np.maximum(1.0, b_q[:, 0] + a_q[:, 0])
    flow[:, 2] = np.log1p(v_buy) / 8.0
    flow[:, 3] = np.log1p(v_sell) / 8.0
    flow[:, 4] = (v_buy - v_sell) / np.maximum(1.0, v_buy + v_sell)
    flow[:, 5] = dd_300
    flow[:, 6] = np.log1p(np.sum(b_q, axis=1)) / 12.0
    flow[:, 7] = np.log1p(np.sum(a_q, axis=1)) / 12.0
    # Остальные каналы заполняем нормированными производными
    flow[:, 8:16] = 0.0

    return lob, flow


def pretrain_backbone(epochs: int = 5, batch_size: int = 64, seq_len: int = 32):
    print("=" * 80)
    print("🧠 Предварительное обучение MultiTask Shared Backbone (Spatial CNN + Causal GRU)")
    print(f"   Данные: {CACHE_PATH.name} (609.2ч, 2.19M тиков) | Устройство: NVIDIA RTX 5060 Ti")
    print("=" * 80, flush=True)

    raw = np.load(CACHE_PATH)
    n_ticks = len(raw["bid_p"])
    split_idx = int(n_ticks * 0.85)

    print("⚡ Подготовка тензоров стакана L2 [N, 10, 4] и потока сделок [N, 16]...", flush=True)
    lob, flow = build_lob_10_and_flow_tensors(raw)

    targets = {
        "sweep": raw["y_sweep_12s"],
        "bleed": raw["y_bleed_60s"],
        "crash": raw["y_crash_300s"],
        "exit": raw["y_exit_30s"],
    }

    train_ds = SequenceLOBDataset(lob[:split_idx], flow[:split_idx], {k: v[:split_idx] for k, v in targets.items()}, seq_len=seq_len)
    val_ds = SequenceLOBDataset(lob[split_idx:], flow[split_idx:], {k: v[split_idx:] for k, v in targets.items()}, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Запуск на устройстве: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"📦 Train батчей: {len(train_loader):,} | Val батчей: {len(val_loader):,}", flush=True)

    model = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256, gru_layers=1).to(device)

    crit_sw = FocalLoss(alpha=0.95, gamma=2.0)
    crit_bl = FocalLoss(alpha=0.98, gamma=2.0)
    crit_cr = FocalLoss(alpha=0.99, gamma=2.0)
    crit_ex = FocalLoss(alpha=0.98, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    ckpt_path = CKPT_DIR / "pretrained_multitask_backbone.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_b = 0

        for b_lob, b_flow, y_sw, y_bl, y_cr, y_ex in train_loader:
            b_lob = b_lob.to(device, non_blocking=True)
            b_flow = b_flow.to(device, non_blocking=True)
            y_sw = y_sw.to(device, non_blocking=True)
            y_bl = y_bl.to(device, non_blocking=True)
            y_cr = y_cr.to(device, non_blocking=True)
            y_ex = y_ex.to(device, non_blocking=True)

            out = model.forward_sequence(b_lob, b_flow)

            # Мультизадачный совокупный лосс по всем горизонтам микроструктуры
            l_sw = crit_sw(out["p_sweep_12s"].view(-1), y_sw.view(-1))
            l_bl = crit_bl(out["p_bleed_60s"].view(-1), y_bl.view(-1))
            l_cr = crit_cr(out["p_crash_300s"].view(-1), y_cr.view(-1))
            l_ex = crit_ex(out["p_exit_30s"].view(-1), y_ex.view(-1))

            total_l = l_sw + 1.5 * l_bl + 2.0 * l_cr + 1.2 * l_ex

            optimizer.zero_grad()
            total_l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_l.item()
            n_b += 1

        scheduler.step()
        train_loss /= max(1, n_b)

        # Валидация на слепом OOS
        model.eval()
        val_loss = 0.0
        n_v = 0
        with torch.no_grad():
            for b_lob, b_flow, y_sw, y_bl, y_cr, y_ex in val_loader:
                b_lob = b_lob.to(device, non_blocking=True)
                b_flow = b_flow.to(device, non_blocking=True)
                y_sw = y_sw.to(device, non_blocking=True)
                y_bl = y_bl.to(device, non_blocking=True)
                y_cr = y_cr.to(device, non_blocking=True)
                y_ex = y_ex.to(device, non_blocking=True)

                out = model.forward_sequence(b_lob, b_flow)
                l_sw = crit_sw(out["p_sweep_12s"].view(-1), y_sw.view(-1))
                l_bl = crit_bl(out["p_bleed_60s"].view(-1), y_bl.view(-1))
                l_cr = crit_cr(out["p_crash_300s"].view(-1), y_cr.view(-1))
                l_ex = crit_ex(out["p_exit_30s"].view(-1), y_ex.view(-1))
                val_loss += (l_sw + 1.5 * l_bl + 2.0 * l_cr + 1.2 * l_ex).item()
                n_v += 1
        val_loss /= max(1, n_v)
        dt = time.time() - t0

        print(f"Эпоха {epoch:02d}/{epochs:02d} [{dt:.1f}с] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "latent_dim": 256,
                    "flow_dim": 16,
                },
                ckpt_path,
            )
            print(f"  ⭐ Сохранен лучший Shared Backbone в {ckpt_path.name}", flush=True)

    print("\n✅ Сквозное предобучение MultiTask Backbone успешно завершено!", flush=True)
    return ckpt_path


if __name__ == "__main__":
    pretrain_backbone()

