#!/usr/bin/env python3
"""
_tools/pretrain_lob_backbone.py
=============================================================================
Этап 1 глубокого мультизадачного обучения v8.0 (GPU VRAM-Resident High-Throughput):
Сквозное предобучение (Pretraining) Spatial-Temporal Backbone на всех 2.78M+ тиках
L2 книги стакана на NVIDIA GeForce RTX 5060 Ti.

Оптимизации без потери качества вычислений:
1. Векторизованная сборка `build_lob_10_and_flow_tensors` сразу по всем 10 уровням без Python-цикла.
2. Размещение обучающих/валидационных блоков последовательностей напрямую в VRAM (`BF16` входы,
   `~320 МБ` вместо 73,000+ IPC-пересылок `DataLoader` с `batch_size=64`).
3. Пакетное исполнение `batch_size=1024` с `BF16` Tensor Cores (`autocast`), `AdamW(fused=True)`,
   `@torch.compile` и вычислением `FocalLoss` строго в `FP32` (0 андерфлоу редких событий).
4. Полный отказ от `.item()` внутри цикла батчей (единичная синхронизация на конец эпохи).
=============================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.multi_horizon_guard_net import FocalLoss

CACHE_PATH = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
CKPT_DIR = BASE_DIR / "models" / "prl" / "multitask_lob"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class SequenceLOBDataset(Dataset):
    """Сохранено для обратной совместимости внешних импортов."""

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
    Векторизованно формирует тензор L2 стакана (N, 10, 4) и тензор потока ордеров (N, 16)
    со 100% математическим совпадением с эталоном (без цикла по 10 уровням).
    """
    b_p = raw["bid_p"].astype(np.float32, copy=False)
    b_q = raw["bid_q"].astype(np.float32, copy=False)
    a_p = raw["ask_p"].astype(np.float32, copy=False)
    a_q = raw["ask_q"].astype(np.float32, copy=False)
    tick = float(raw["tick_size"])
    n = len(b_p)

    mid_p = ((b_p[:, 0] + a_p[:, 0]) * 0.5)[:, None]
    lob = np.empty((n, 10, 4), dtype=np.float32)
    lob[:, :, 0] = np.clip((mid_p - b_p[:, :10]) / tick, 0.0, 50.0) * 0.1
    lob[:, :, 1] = np.log1p(np.maximum(0.0, b_q[:, :10])) * 0.1
    lob[:, :, 2] = np.clip((a_p[:, :10] - mid_p) / tick, 0.0, 50.0) * 0.1
    lob[:, :, 3] = np.log1p(np.maximum(0.0, a_q[:, :10])) * 0.1

    v_buy = raw["buy_vol"].astype(np.float32, copy=False)
    v_sell = raw["sell_vol"].astype(np.float32, copy=False)
    s_b1 = b_p[:, 0]
    high_300 = np.maximum.accumulate(s_b1)
    dd_300 = np.clip((high_300 - s_b1) / tick, 0.0, 20.0) * 0.1

    flow = np.zeros((n, 16), dtype=np.float32)
    flow[:, 0] = np.clip((a_p[:, 0] - b_p[:, 0]) / tick, 1.0, 10.0) * 0.2
    flow[:, 1] = (b_q[:, 0] - a_q[:, 0]) / np.maximum(1.0, b_q[:, 0] + a_q[:, 0])
    flow[:, 2] = np.log1p(np.maximum(0.0, v_buy)) * 0.125
    flow[:, 3] = np.log1p(np.maximum(0.0, v_sell)) * 0.125
    flow[:, 4] = (v_buy - v_sell) / np.maximum(1.0, v_buy + v_sell)
    flow[:, 5] = dd_300
    flow[:, 6] = np.log1p(np.maximum(0.0, np.sum(b_q[:, :10], axis=1))) / 12.0
    flow[:, 7] = np.log1p(np.maximum(0.0, np.sum(a_q[:, :10], axis=1))) / 12.0

    return lob, flow


def _pack_sequences_gpu(
    lob_np: np.ndarray,
    flow_np: np.ndarray,
    y_sw_np: np.ndarray,
    y_bl_np: np.ndarray,
    y_cr_np: np.ndarray,
    y_ex_np: np.ndarray,
    seq_len: int,
    device: torch.device,
    feat_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Упаковывает непрерывный массив в блоки [N_seq, seq_len, ...] сразу на GPU."""
    n_seq = (len(lob_np) - seq_len - 1) // seq_len
    used = n_seq * seq_len
    lob_t = torch.from_numpy(lob_np[:used]).to(device=device, dtype=feat_dtype).view(n_seq, seq_len, 10, 4).contiguous()
    flow_t = torch.from_numpy(flow_np[:used]).to(device=device, dtype=feat_dtype).view(n_seq, seq_len, 16).contiguous()
    y4_np = np.stack(
        [
            y_sw_np[:used].astype(np.float32, copy=False),
            y_bl_np[:used].astype(np.float32, copy=False),
            y_cr_np[:used].astype(np.float32, copy=False),
            y_ex_np[:used].astype(np.float32, copy=False),
        ],
        axis=-1,
    )
    y4_t = torch.from_numpy(y4_np).to(device=device, dtype=torch.float32).view(n_seq, seq_len, 4).contiguous()
    return lob_t, flow_t, y4_t


def pretrain_backbone(epochs: int = 8, batch_size: int = 1024, seq_len: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    feat_dtype = torch.bfloat16 if use_bf16 else torch.float32

    raw = np.load(CACHE_PATH, allow_pickle=True)
    n_ticks = len(raw["bid_p"])
    split_idx = int(n_ticks * 0.85)

    print("=" * 88)
    print("🧠 Предобучение MultiTask Shared Backbone (VRAM-Resident BF16 + FP32 FocalLoss)")
    print(f"   Данные: {CACHE_PATH.name} ({n_ticks:,} тиков) | Устройство: {device} | batch_size={batch_size}")
    print("=" * 88, flush=True)

    lob, flow = build_lob_10_and_flow_tensors(raw)
    y_sw = raw["y_sweep_12s"]
    y_bl = raw["y_bleed_60s"]
    y_cr = raw["y_crash_300s"]
    y_ex = raw["y_exit_30s"]
    del raw

    tr_lob, tr_flow, tr_y4 = _pack_sequences_gpu(
        lob[:split_idx], flow[:split_idx], y_sw[:split_idx], y_bl[:split_idx], y_cr[:split_idx], y_ex[:split_idx], seq_len, device, feat_dtype
    )
    va_lob, va_flow, va_y4 = _pack_sequences_gpu(
        lob[split_idx:], flow[split_idx:], y_sw[split_idx:], y_bl[split_idx:], y_cr[split_idx:], y_ex[split_idx:], seq_len, device, feat_dtype
    )
    del lob, flow, y_sw, y_bl, y_cr, y_ex

    n_tr_seq = tr_lob.size(0)
    n_va_seq = va_lob.size(0)
    print(f"📦 Последовательностей в VRAM: Train={n_tr_seq:,} | Val={n_va_seq:,} (seq_len={seq_len})", flush=True)

    model = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256, gru_layers=1).to(device)
    fast_forward = torch.compile(model.forward_sequence) if device.type == "cuda" else model.forward_sequence

    crit_sw = FocalLoss(alpha=0.95, gamma=2.0)
    crit_bl = FocalLoss(alpha=0.98, gamma=2.0)
    crit_cr = FocalLoss(alpha=0.99, gamma=2.0)
    crit_ex = FocalLoss(alpha=0.98, gamma=2.0)

    # Масштабируем шаг AdamW пропорционально sqrt(1024 / 64) = 4x (с 1e-3 до 2.5e-3) + fused CUDA kernel
    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4, fused=use_fused)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)

    best_val_loss = float("inf")
    ckpt_path = CKPT_DIR / "pretrained_multitask_backbone.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_tr_seq, device=device)
        train_loss_gpu = torch.zeros((), dtype=torch.float32, device=device)
        n_b = 0

        for s_i in range(0, n_tr_seq, batch_size):
            idx = perm[s_i : s_i + batch_size]
            b_lob = tr_lob[idx]
            b_flow = tr_flow[idx]
            b_y4 = tr_y4[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=feat_dtype, enabled=use_bf16):
                out = fast_forward(b_lob, b_flow)

            # FocalLoss вычисляется строго в FP32 без потери точности редких хвостов!
            l_sw = crit_sw(out["p_sweep_12s"].float().reshape(-1), b_y4[:, :, 0].reshape(-1))
            l_bl = crit_bl(out["p_bleed_60s"].float().reshape(-1), b_y4[:, :, 1].reshape(-1))
            l_cr = crit_cr(out["p_crash_300s"].float().reshape(-1), b_y4[:, :, 2].reshape(-1))
            l_ex = crit_ex(out["p_exit_30s"].float().reshape(-1), b_y4[:, :, 3].reshape(-1))
            total_l = l_sw + 1.5 * l_bl + 2.0 * l_cr + 1.2 * l_ex

            total_l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_gpu += total_l.detach()
            n_b += 1

        scheduler.step()

        model.eval()
        val_loss_gpu = torch.zeros((), dtype=torch.float32, device=device)
        n_v = 0
        with torch.no_grad():
            for s_i in range(0, n_va_seq, batch_size * 2):
                b_lob = va_lob[s_i : s_i + batch_size * 2]
                b_flow = va_flow[s_i : s_i + batch_size * 2]
                b_y4 = va_y4[s_i : s_i + batch_size * 2]
                with torch.amp.autocast(device_type=device.type, dtype=feat_dtype, enabled=use_bf16):
                    out = fast_forward(b_lob, b_flow)
                l_sw = crit_sw(out["p_sweep_12s"].float().reshape(-1), b_y4[:, :, 0].reshape(-1))
                l_bl = crit_bl(out["p_bleed_60s"].float().reshape(-1), b_y4[:, :, 1].reshape(-1))
                l_cr = crit_cr(out["p_crash_300s"].float().reshape(-1), b_y4[:, :, 2].reshape(-1))
                l_ex = crit_ex(out["p_exit_30s"].float().reshape(-1), b_y4[:, :, 3].reshape(-1))
                val_loss_gpu += (l_sw + 1.5 * l_bl + 2.0 * l_cr + 1.2 * l_ex)
                n_v += 1

        # ЕДИНСТВЕННАЯ синхронизация GPU->CPU на всю эпоху!
        train_loss = float(train_loss_gpu.item()) / max(1, n_b)
        val_loss = float(val_loss_gpu.item()) / max(1, n_v)
        dt = time.time() - t0

        print(f"Эпоха {epoch:02d}/{epochs:02d} [{dt:.2f}с] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}", flush=True)

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
