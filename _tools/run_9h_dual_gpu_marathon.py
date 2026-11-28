#!/usr/bin/env python3
"""
_tools/run_9h_dual_gpu_marathon.py
================================================================================
9-Часовая Автономная Программа Исследования и Развития Резонансной Архитектуры
на 2 видеокартах (GPU 0 [DEC] || GPU 1 [RAW]) с 3 циклами живого обновления
данных с бирж и реальных логов боевого робота с VPS:

  • Чекпоинт данных #1 (Старт, T+0.0ч):
      Синхронизация логов робота с VPS (+52ч реальных PLACE/CANCEL/FILL) + горячих
      L500 стаканов бирж -> инкрементальное дообучение Order-1, Wave, Order-2, Order-2.5 -> 112D.
  • Волна 1 / L95-A (~2.5ч на 2 GPU):
      Подпространственные (Subspace-Sparse, 4-5 активных частот на моду) High-Q
      Гауссовы резонаторы в нуль-пространстве H_0 (OMP + Sep-CMA-ES + Joint Polish).
  • Чекпоинт данных #2 (~T+2.8ч):
      Повторная синхронизация бирж + свежих сделок робота с VPS -> запечатывание новой
      полной 24ч эпохи Write-Once реестра + пересчёт Order-1/2/2.5 и скользящего Blind OOS.
  • Волна 2 / L95-B (~2.5ч на 2 GPU):
      Связанные дипольные пары резонаторов Вход+Выход (Entry Agile Pod + Profit/Unwind Dipole)
      и альтернативные спектральные архетипы + совместная настройка всех полос.
  • Чекпоинт данных #3 (~T+5.7ч):
      Утренняя синхронизация L500 стаканов + логов робота с VPS -> дообучение боевых окон
      win_live (Order-1/Wave/Order-2/Order-2.5) и расширение реестра 112D.
  • Волна 3 / L95-C (~2.3ч на 2 GPU):
      Полиномиальные резонансные фильтры Баттерворта порядка p_k in [1.2 .. 4.5]
      с плоской вершиной полосы пропускания (Flat-Top Passband) + Multi-Seed Resonant Soup.
  • Финальный экзамен и безопасный деплой (~T+8.4ч .. 9.0ч):
      Честный турнир H_0 vs L95-A vs L95-B vs L95-C vs Soup на всех блоках C/D/U/Q + Blind OOS,
      проверка паритета Sim/Live = 0.00e+00 и синхронизация на боевой VPS.
================================================================================
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CACHE_DIR = BASE_DIR / "data" / "cache"
MARATHON_DIR = BASE_DIR / "logs" / "l95_9h_marathon"
STATUS_JSON = MARATHON_DIR / "live_status.json"
FINAL_JSON = MARATHON_DIR / "final_9h_report.json"
CHAMP_V1 = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _save_status(state: dict) -> None:
    MARATHON_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_JSON.with_name("live_status.json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS_JSON)


# ==============================================================================
# 1. ИЗОЛИРОВАННЫЙ GPU-ВОРКЕР ОДНОГО МИРА (GPU 0 = DEC, GPU 1 = RAW)
# ==============================================================================
def _gpu_world_worker(gpu_id: int, world_name: str, subcent_start: int, cmd_q: mp.Queue, res_q: mp.Queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TRITON_CACHE_DIR"] = f"/home/restorator/crypto_bot/scratch/.triton_g{gpu_id}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/home/restorator/crypto_bot/scratch/.inductor_g{gpu_id}"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    import torch as th
    from _tools.fused_order3_clean_tick import evaluate_order3_basket

    dev = th.device("cuda:0")
    is_r = world_name == "RAW"
    lob = np.load(
        CACHE_DIR / ("rl_guard_lob_cache_600h_raw_with_self.npz" if is_r else "rl_guard_lob_cache_600h.npz"),
        mmap_mode="r",
    )
    mkt = np.load(
        CACHE_DIR / ("rl_hybrid_market_obs_112d_raw_with_self.npz" if is_r else "rl_hybrid_market_obs_112d.npz"),
        mmap_mode="r",
    )["market_obs_112d"]

    ts = lob["ts"][subcent_start:]
    q0_u = int(np.datetime64("2026-11-24T17:05:00", "s").astype(np.int64))
    q1_u = int(np.datetime64("2026-11-24T18:10:00", "s").astype(np.int64))
    sv_np = np.asarray(lob["sell_vol"][subcent_start:], dtype=np.float32).copy()
    bv_np = np.asarray(lob["buy_vol"][subcent_start:], dtype=np.float32).copy()
    qm = (ts >= q0_u) & (ts <= q1_u)
    sv_np[qm] = 0.0
    bv_np[qm] = 0.0

    mkt_g = th.from_numpy(np.asarray(mkt[subcent_start:], dtype=np.float32)).to(dev, th.bfloat16)
    b10_g = th.from_numpy(np.asarray(lob["bid_p"][subcent_start:, :10], dtype=np.float32)).to(dev)
    a10_g = th.from_numpy(np.asarray(lob["ask_p"][subcent_start:, :10], dtype=np.float32)).to(dev)
    bq10_g = th.from_numpy(np.asarray(lob["bid_q"][subcent_start:, :10], dtype=np.float32)).to(dev)
    aq10_g = th.from_numpy(np.asarray(lob["ask_q"][subcent_start:, :10], dtype=np.float32)).to(dev)
    sv_g = th.from_numpy(sv_np).to(dev)
    bv_g = th.from_numpy(bv_np).to(dev)

    base_state_gpu: dict[str, th.Tensor] = {}
    res_q.put(("READY", world_name, int(len(mkt_g)), th.cuda.get_device_name(0)))

    while True:
        msg = cmd_q.get()
        if msg is None or msg[0] == "STOP":
            break
        cmd = msg[0]
        if cmd == "SET_CARRIER":
            cpu_st = msg[1]
            base_state_gpu = {k: v.to(dev) for k, v in cpu_st.items()}
            res_q.put(("OK_CARRIER", world_name))
        elif cmd == "EVAL_RES_POP":
            # msg = ("EVAL_RES_POP", res_banks_cpu_list, windows, win_len)
            res_banks, windows, win_len = msg[1], msg[2], int(msg[3])
            cand_states = []
            for rb in res_banks:
                st_c = dict(base_state_gpu)
                st_c["res_amp"] = th.from_numpy(rb["res_amp"]).to(dev)
                st_c["res_center"] = th.from_numpy(rb["res_center"]).to(dev)
                st_c["res_q"] = th.from_numpy(rb["res_q"]).to(dev)
                st_c["res_du"] = th.from_numpy(rb["res_du"]).to(dev)
                st_c["res_p"] = th.from_numpy(rb["res_p"]).to(dev)
                cand_states.append(st_c)
            out = evaluate_order3_basket(
                mkt_39d_all=mkt_g,
                b10_all=b10_g,
                a10_all=a10_g,
                bq10_all=bq10_g,
                aq10_all=aq10_g,
                sv_all=sv_g,
                bv_all=bv_g,
                cand_states=cand_states,
                val_windows=windows,
                win_len=win_len,
                tick_sz=0.0001,
                device=dev,
                pad_to_max_c=False,
                harsh_stress=True,
                enforce_activity_floor=False,
            )
            res_q.put(("EVAL_DONE", world_name, out))


class DualGpuWorldPool:
    """Параллельный пул оценки DEC (GPU 0) || RAW (GPU 1) с двукратным ускорением на шаг."""

    def __init__(self, subcent_start: int, carrier_cpu: dict[str, torch.Tensor]):
        ctx = mp.get_context("spawn")
        self.q_dec_in, self.q_dec_out = ctx.Queue(), ctx.Queue()
        self.q_raw_in, self.q_raw_out = ctx.Queue(), ctx.Queue()
        self.p_dec = ctx.Process(target=_gpu_world_worker, args=(0, "DEC", subcent_start, self.q_dec_in, self.q_dec_out), daemon=True)
        self.p_raw = ctx.Process(target=_gpu_world_worker, args=(1, "RAW", subcent_start, self.q_raw_in, self.q_raw_out), daemon=True)
        self.p_dec.start()
        self.p_raw.start()
        r0 = self.q_dec_out.get(timeout=180)
        r1 = self.q_raw_out.get(timeout=180)
        _log(f"🟢 Dual-GPU Pool активен: GPU 0 [{r0[1]} rows={r0[2]:,} {r0[3]}] || GPU 1 [{r1[1]} rows={r1[2]:,} {r1[3]}]")
        self.q_dec_in.put(("SET_CARRIER", carrier_cpu))
        self.q_raw_in.put(("SET_CARRIER", carrier_cpu))
        self.q_dec_out.get(timeout=60)
        self.q_raw_out.get(timeout=60)

    def eval_pop(self, res_banks: list[dict[str, np.ndarray]], windows: list[tuple[str, int, float, float]], win_len: int = 86400) -> list[dict]:
        # Отправляем одновременно в оба процесса -> GPU 0 и GPU 1 считают параллельно!
        self.q_dec_in.put(("EVAL_RES_POP", res_banks, windows, win_len))
        self.q_raw_in.put(("EVAL_RES_POP", res_banks, windows, win_len))
        _, _, out_dec = self.q_dec_out.get()
        _, _, out_raw = self.q_raw_out.get()

        merged = []
        for rd, rr in zip(out_dec, out_raw):
            ev_dec = 0.50 * float(rd["calib_clean_ev_pct"]) + 0.50 * float(rd["clean_ev_pct"])
            ev_raw = 0.50 * float(rr["calib_clean_ev_pct"]) + 0.50 * float(rr["clean_ev_pct"])
            ev_4w = 0.50 * ev_dec + 0.50 * ev_raw
            dd_max = max(float(rd["dd_pct"]), float(rd["calib_dd_pct"]), float(rr["dd_pct"]), float(rr["calib_dd_pct"]))
            cvar_4w = 0.50 * float(rd["cvar25_bp"]) + 0.50 * float(rr["cvar25_bp"])
            fills_w = 0.50 * float(rd["fills"]) + 0.50 * float(rr["fills"])
            merged.append(
                {
                    "ev_4w_pct": ev_4w,
                    "ev_dec_pct": ev_dec,
                    "ev_raw_pct": ev_raw,
                    "dd_max_pct": dd_max,
                    "dd_dec_pct": max(float(rd["dd_pct"]), float(rd["calib_dd_pct"])),
                    "dd_raw_pct": max(float(rr["dd_pct"]), float(rr["calib_dd_pct"])),
                    "cvar_bp": cvar_4w,
                    "fills": fills_w,
                    "fills_dec": float(rd["fills"]),
                    "fills_raw": float(rr["fills"]),
                }
            )
        return merged

    def close(self) -> None:
        for q_in in (self.q_dec_in, self.q_raw_in):
            try:
                q_in.put(("STOP",))
            except Exception:
                pass
        for p in (self.p_dec, self.p_raw):
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        _log("🧹 Dual-GPU Pool остановлен, VRAM GPU 0 + GPU 1 полностью освобождена.")


# ==============================================================================
# 2. ИЗВЛЕЧЕНИЕ КАУЗАЛЬНОГО 16D БАЗИСА НА TRAIN И ПОДПРОСТРАНСТВЕННЫЕ СЕМЕНА
# ==============================================================================
def empty_res_bank() -> dict[str, np.ndarray]:
    return {
        "res_amp": np.zeros((1, 32), dtype=np.float32),
        "res_center": np.zeros((32, 16), dtype=np.float32),
        "res_q": np.zeros((32, 16), dtype=np.float32),
        "res_du": np.zeros((32, 60), dtype=np.float32),
        "res_p": np.zeros((1, 32), dtype=np.float32),
    }


def clone_bank(b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in b.items()}


@torch.no_grad()
def extract_train_basis_cpu_fast(carrier_cpu: dict[str, torch.Tensor], subcent_start: int, train_blocks: list[dict]) -> dict[str, np.ndarray]:
    """Каузально извлекает 16D базис phi(t) строго на Train-блоках (шаг 15с -> ~23,000 точек за ~1.5с)."""
    from _tools.fused_order3_clean_tick import MEM_DIM_ORDER3, MKT_DIM_ORDER3, _extract_resonant_phi_16d, _scale_mkt_44d

    lob = np.load(CACHE_DIR / "rl_guard_lob_cache_600h.npz", mmap_mode="r")
    mkt = np.load(CACHE_DIR / "rl_hybrid_market_obs_112d.npz", mmap_mode="r")["market_obs_112d"]
    b1_sub = np.asarray(lob["bid_p"][subcent_start:, 0], dtype=np.float32)

    w1_mkt = carrier_cpu["w1"][:MKT_DIM_ORDER3, :].float()
    b1_v = carrier_cpu["b1"].float().view(1, 384)
    wr1 = carrier_cpu["w_res1"].float()
    br1 = carrier_cpu["b_res1"].float().view(1, 128)
    wr2 = carrier_cpu["w_res2"].float()
    wrec = carrier_cpu["w_rec"].float()
    brec = carrier_cpu["b_rec"].float().view(1, 2 * MEM_DIM_ORDER3)

    phi_l, day_l, reg_l, fwd_l = [], [], [], []
    reg_map = {"Q": 0, "U": 1, "C": 2, "D": 3}

    for di, tb in enumerate(train_blocks):
        ws, we = int(tb["rel_start"]), int(tb["rel_end"])
        idx_rel = np.arange(max(0, ws - 1800), we, 15, dtype=np.int64)
        zm_seq = _scale_mkt_44d(torch.from_numpy(np.asarray(mkt[subcent_start + idx_rel], dtype=np.float32)), torch.float32)
        pre = zm_seq @ w1_mkt + b1_v
        h1 = torch.relu(pre[:, :256]) * (2.0 * torch.sigmoid(pre[:, 256:384])).repeat_interleave(2, dim=-1)
        rp = h1 @ wr1 + br1
        h2 = h1 + (torch.relu(rp[:, :64]) * (2.0 * torch.sigmoid(rp[:, 64:128]))) @ wr2
        rr = h2 @ wrec + brec
        ru = torch.tanh(rr[:, :MEM_DIM_ORDER3])
        rg_1s = torch.cat([
            torch.sigmoid(rr[:, 64:72] + 1.0), torch.sigmoid(rr[:, 72:80] + 3.8),
            torch.sigmoid(rr[:, 80:88] + 5.2), torch.sigmoid(rr[:, 88:96] + 6.6),
            torch.sigmoid(rr[:, 96:104] + 2.2), torch.sigmoid(rr[:, 104:112] + 4.5),
            torch.sigmoid(rr[:, 112:120] + 7.5), torch.sigmoid(rr[:, 120:128] + 8.9),
        ], dim=-1)
        rg_step = torch.pow(rg_1s, 15.0)
        mem = torch.zeros((MEM_DIM_ORDER3,), dtype=torch.float32)
        mems = []
        for ti in range(len(idx_rel)):
            mem = rg_step[ti] * mem + (1.0 - rg_step[ti]) * ru[ti]
            mems.append(mem)
        mem_mat = torch.stack(mems, dim=0)
        keep = idx_rel >= ws
        ik = idx_rel[keep]
        phi_0 = _extract_resonant_phi_16d(mem_mat[keep], zm_seq[keep], torch.zeros(len(ik))).numpy()
        ik_f = np.minimum(we - 1, ik + 900)
        ret_15m = (b1_sub[ik_f] - b1_sub[ik]) / np.maximum(1e-4, b1_sub[ik])

        phi_l.append(phi_0)
        day_l.append(np.full(len(ik), di, dtype=np.int32))
        reg_l.append(np.full(len(ik), reg_map[tb["regime"]], dtype=np.int32))
        fwd_l.append(ret_15m.astype(np.float32))

    return {
        "phi": np.concatenate(phi_l, axis=0),
        "day": np.concatenate(day_l, axis=0),
        "reg": np.concatenate(reg_l, axis=0),
        "fwd15m": np.concatenate(fwd_l, axis=0),
    }


def check_nyquist_subspace(
    amp_arr: np.ndarray,   # [C]
    cen_arr: np.ndarray,   # [C, 16]
    q_arr: np.ndarray,     # [C, 16]
    p_arr: np.ndarray,     # [C] (0 = Гаусс, >0 = Баттерворт)
    tr_basis: dict[str, np.ndarray],
    min_hits: int = 25,
    min_days: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Быстрая векторизованная проверка закона Найквиста и узкополосной селективности:
      - шаг 60с (каждая 4-я точка из 15с сетки),
      - >= min_hits независимых эпизодов на >= min_days различных днях Train,
      - duty_cycle строго в узком окне [0.6% .. 15.0%],
      - срабатывание на медвежьем обвале D <= 1.2%!
    """
    phi_s = tr_basis["phi"][::4]  # [L, 16] (шаг 60с)
    day_s = tr_basis["day"][::4]
    reg_s = tr_basis["reg"][::4]

    C = len(amp_arr)
    phi_c = np.broadcast_to(phi_s[None, :, :], (C, len(phi_s), 16)).copy()
    inv_probe = np.where(cen_arr[:, 15:16] > 0.04, cen_arr[:, 15:16], 0.0)
    phi_c[:, :, 15] = inv_probe

    diff = phi_c - cen_arr[:, None, :]
    q_eff = np.clip(q_arr, 0.0, 12.0)[:, None, :]
    d2 = np.clip(0.5 * np.sum((diff * q_eff) ** 2, axis=-1), 0.0, 40.0)  # [C, L]
    env_g = np.exp(-d2)
    p_pos = np.clip(p_arr, 0.5, 6.0)[:, None]
    env_b = 1.0 / (1.0 + np.power(d2 + 1e-8, p_pos))
    env = np.where((p_arr > 0.05)[:, None], env_b, env_g)
    gamma = np.clip(amp_arr, 0.0, 1.5)[:, None] * env

    active = gamma > 0.22
    ep_start = active.copy()
    ep_start[:, 1:] = active[:, 1:] & (~active[:, :-1])
    n_ep = ep_start.sum(axis=1)

    n_dy = np.zeros(C, dtype=np.int32)
    for d_id in np.unique(day_s):
        dm = day_s == d_id
        n_dy += (ep_start[:, dm].sum(axis=1) >= 2).astype(np.int32)

    dump_m = reg_s == 3
    dump_rate = active[:, dump_m].mean(axis=1) if np.any(dump_m) else np.zeros(C, dtype=np.float32)
    duty = active.mean(axis=1)

    valid = (n_ep >= min_hits) & (n_dy >= min_days) & (dump_rate <= 0.012) & (duty >= 0.005) & (duty <= 0.15)
    return valid, n_ep, n_dy, duty


def build_subspace_archetypes(tr_basis: dict[str, np.ndarray], wave_tag: str) -> list[dict]:
    """
    Формирует физически специализированные подпространственные архетипы (ровно 4-5 активных осей Q>0,
    остальные оси прозрачны Q=0.0 -> 0% проклятия размерности!) с автоматической калибровкой начального Q
    в рабочую зону Найквиста (40..120 эпизодов на Train).
    """
    phi = tr_basis["phi"]
    reg = tr_basis["reg"]
    fwd = tr_basis["fwd15m"]
    safe = reg != 3
    std_all = np.maximum(0.12, phi[safe, :15].std(axis=0))

    specs = [
        # (name, mask_cond, active_axes, inv_center, kind)
        (
            "LeadLag_UpWave_Pod",
            safe & (phi[:, 11] > np.quantile(phi[:, 11], 0.78)) & (phi[:, 9] > np.quantile(phi[:, 9], 0.72)) & (phi[:, 14] < 0.0),
            [4, 9, 11, 12, 14, 15],
            0.00,
            "maker_up",
        ),
        (
            "WideSpread_Calm_Sniper",
            safe & (phi[:, 8] > np.quantile(phi[:, 8], 0.72)) & (np.abs(phi[:, 13]) < np.quantile(np.abs(phi[:, 13]), 0.50)) & (phi[:, 14] < -0.05),
            [2, 8, 9, 13, 14, 15],
            0.00,
            "quiet_sniper",
        ),
        (
            "Macro30m_Pullback_Bounce",
            safe & (phi[:, 10] > np.quantile(phi[:, 10], 0.68)) & (phi[:, 3] < np.quantile(phi[:, 3], 0.32)) & (phi[:, 9] > 0.0) & (phi[:, 14] < 0.05),
            [3, 9, 10, 13, 14, 15],
            0.00,
            "dip_pod",
        ),
        (
            "CrossVenue_CVD_Breakout",
            safe & (phi[:, 12] > np.quantile(phi[:, 12], 0.80)) & (phi[:, 11] > np.quantile(phi[:, 11], 0.65)) & (phi[:, 13] > 0.0) & (phi[:, 14] < 0.0),
            [5, 11, 12, 13, 14, 15],
            0.00,
            "trend_pod",
        ),
        (
            "MultiScale_SSM_Resonance",
            safe & (fwd > np.quantile(fwd, 0.84)) & (phi[:, 9] > 0.05) & (phi[:, 14] < 0.0),
            [1, 4, 6, 9, 14, 15],
            0.00,
            "rebound_pod",
        ),
        (
            "AgilePod_Harvest_Dipole",
            safe & (phi[:, 9] < np.quantile(phi[:, 9], 0.30)) & (phi[:, 11] < -0.05),
            [3, 9, 11, 14, 15],
            0.14,
            "ask_harvest",
        ),
    ]
    if "L95B_Dipole" in wave_tag:
        # Во Волне 2 ставим дипольный выход сразу 2-м номером в пару к лучшим входным полосам!
        specs = [specs[0], specs[5], specs[1], specs[2], specs[3], specs[4]]
    elif "L95C_Butterworth" in wave_tag:
        # Во Волне 3 (Butterworth Flat-Top) включаем ось 13 (30m тренд > 0) во все входные полосы против пред-обвальных входов на BLK_06_C!
        new_specs = []
        for nm, mc, ax_l, ic, kd in specs:
            if kd != "ask_harvest" and 13 not in ax_l:
                ax_l = sorted(set(ax_l + [13]))
                mc = mc & (phi[:, 13] > np.quantile(phi[:, 13], 0.62))
            new_specs.append((nm, mc, ax_l, ic, kd))
        specs = new_specs

    archetypes = []
    for name_s, m_cond, axes, inv_c, kind_s in specs:
        pts = phi[m_cond] if int(m_cond.sum()) >= 30 else phi[safe]
        c_v = np.median(pts, axis=0).astype(np.float32)
        c_v[15] = float(inv_c)
        # По оси 14 (bleed+crash+trap щит) центрируем в безопасной отрицательной зоне:
        c_v[14] = min(float(c_v[14]), -0.25)

        q_v = np.zeros(16, dtype=np.float32)
        for ax in axes:
            if ax < 14:
                q_v[ax] = float(1.15 / std_all[ax])
        q_v[14] = max(2.20, float(1.80 / std_all[14]))
        q_v[15] = 4.20 if kind_s == "ask_harvest" else 3.00

        # Авто-калибровка масштаба профильных осей (кроме 14 и 15) в целевое окно 35..110 эпизодов:
        p_init = 2.20 if wave_tag == "L95C_Butterworth" else 0.0
        for _ in range(12):
            _, nep, ndy, dt_c = check_nyquist_subspace(
                np.array([0.85], dtype=np.float32),
                c_v[None, :],
                q_v[None, :],
                np.array([p_init], dtype=np.float32),
                tr_basis,
            )
            if nep[0] < 35 or dt_c[0] < 0.010:
                for ax in axes:
                    if ax < 14:
                        q_v[ax] *= 0.84
                q_v[14] = max(1.65, q_v[14] * 0.92)
            elif dt_c[0] > 0.11:
                for ax in axes:
                    if ax < 14:
                        q_v[ax] *= 1.18
            else:
                break

        du_v = np.zeros(60, dtype=np.float32)
        if kind_s in ("maker_up", "trend_pod"):
            du_v[0] = +14.2; du_v[59] = +18.5; du_v[1:4] = +3.2; du_v[37] = +2.5
            du_v[38] = -3.0; du_v[39] = -2.8; du_v[53] = +10.5; du_v[52] = +11.5
            du_v[54] = +8.5; du_v[42] = +0.6; du_v[50:52] = +4.0
        elif kind_s == "quiet_sniper":
            du_v[0] = +13.6; du_v[59] = +18.5; du_v[55] = +6.5; du_v[1:3] = +3.5
            du_v[37] = +2.2; du_v[39] = -3.2; du_v[42] = -0.9; du_v[52] = +12.0
            du_v[54] = +9.0; du_v[50:52] = +4.0
        elif kind_s in ("dip_pod", "rebound_pod"):
            du_v[0] = +14.0; du_v[59] = +18.5; du_v[2:7] = +2.8; du_v[39] = -1.6
            du_v[52] = +12.0; du_v[54] = +8.5; du_v[56] = +14.0; du_v[50:52] = +4.0
        elif kind_s == "ask_harvest":
            du_v[0] = -6.0; du_v[13:17] = +3.6; du_v[43] = +3.0; du_v[52] = +12.5
            du_v[54] = +10.0; du_v[51] = +4.5

        archetypes.append(
            {
                "name": f"{wave_tag}_{name_s}",
                "kind": kind_s,
                "axes": axes,
                "c": c_v,
                "q": q_v,
                "du": du_v,
                "p": p_init,
            }
        )
    return archetypes


# ==============================================================================
# 3. ОПТИМИЗАТОР ОДНОЙ ВОЛНЫ (OMP ПО ПОЛОСАМ + СОВМЕСТНАЯ ПОЛИРОВКА НА 2 GPU)
# ==============================================================================
def run_evolution_wave(
    wave_tag: str,
    start_bank: dict[str, np.ndarray],
    pool: DualGpuWorldPool,
    tr_basis: dict[str, np.ndarray],
    train_wins: list[tuple[str, int, float, float]],
    val_wins: list[tuple[str, int, float, float]],
    k_max: int,
    min_hits: int,
    min_days: int,
    n_gens_band: int,
    n_gens_polish: int,
    allow_butterworth: bool,
    seed_rng: int,
    deadline_ts: float,
    status_dict: dict,
) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    rng = np.random.default_rng(seed_rng)
    cur_bank = clone_bank(start_bank)
    archetypes = build_subspace_archetypes(tr_basis, wave_tag)

    pop_sz = 32
    mu_el = 8
    w_rec = np.array([math.log(mu_el + 0.5) - math.log(i + 1.0) for i in range(mu_el)], dtype=np.float32)
    w_rec /= w_rec.sum()

    base_tr = pool.eval_pop([cur_bank] * pop_sz, train_wins)[0]
    base_val = pool.eval_pop([cur_bank] * pop_sz, val_wins)[0]
    _log(
        f"🌊 [{wave_tag}] Старт волны: Train_4W={base_tr['ev_4w_pct']:+.3f}%/d (DD={base_tr['dd_max_pct']:.2f}%) | "
        f"PurgedVal_4W={base_val['ev_4w_pct']:+.3f}%/d (DD={base_val['dd_max_pct']:.2f}%)"
    )

    locked_info = []
    # Слоты 0..29 — узкополосные входы/выходы; слоты 30..31 зарезервированы под сопряжённый Баттерворт-контроллер сопровождения и стоп-щит:
    free_slots = [idx_s for idx_s in range(30) if cur_bank["res_amp"][0, idx_s] < 1e-5]
    target_slots = free_slots[: min(k_max, len(archetypes))]

    for b_ord, (slot_k, arch) in enumerate(zip(target_slots, archetypes)):
        if time.time() > deadline_ts - 600:
            _log(f"⏰ [{wave_tag}] Достигнут бюджет времени волны перед полосой #{b_ord + 1}.")
            break

        axes = arch["axes"]
        n_ax = len(axes)
        is_coupled_dipole = ("L95B_Dipole" in wave_tag) and (arch["kind"] != "ask_harvest")
        # Компактный вектор параметров подпространственной полосы (+6 параметров сопряжённого диполя сопровождения/стопа во Волнах 2-3):
        # [raw_amp (1), c_axes (n_ax), log_q_axes (n_ax), raw_p (1), du (60), life_6 (6)]
        d_th = 1 + n_ax + n_ax + 1 + 60 + (6 if is_coupled_dipole else 0)
        mean_th = np.zeros(d_th, dtype=np.float32)
        mean_th[0] = 0.35
        mean_th[1 : 1 + n_ax] = arch["c"][axes]
        mean_th[1 + n_ax : 1 + 2 * n_ax] = np.log(np.maximum(0.25, arch["q"][axes]))
        mean_th[1 + 2 * n_ax] = arch["p"] if allow_butterworth else 0.0
        mean_th[2 + 2 * n_ax : 62 + 2 * n_ax] = arch["du"]
        if is_coupled_dipole:
            # [ask_edge_z42, inside_a_z43, unwind_z54, crash_c14, crash_q14, mom_c9]
            mean_th[62 + 2 * n_ax :] = np.array([-0.40, +3.20, +14.0, +0.45, +2.10, -0.85], dtype=np.float32)

        sig_th = np.ones(d_th, dtype=np.float32)
        sig_th[0] = 0.45
        sig_th[1 : 1 + n_ax] = 0.25
        sig_th[1 + n_ax : 1 + 2 * n_ax] = 0.22
        sig_th[1 + 2 * n_ax] = 0.45 if allow_butterworth else 0.0
        sig_th[2 + 2 * n_ax : 62 + 2 * n_ax] = 1.15
        if is_coupled_dipole:
            sig_th[62 + 2 * n_ax :] = np.array([0.65, 0.70, 1.20, 0.25, 0.30, 0.30], dtype=np.float32)

        best_cands_tr: list[tuple[float, dict[str, np.ndarray], int, int, float]] = []
        _log(f"🎛️ [{wave_tag} Полоса {b_ord + 1}/{len(target_slots)} -> slot#{slot_k}] {arch['name']} (оси={axes}, coupled_3pole={is_coupled_dipole})")

        for gen in range(n_gens_band):
            if time.time() > deadline_ts - 360:
                break
            t_g0 = time.time()
            half = (pop_sz - 2) // 2
            eps = rng.standard_normal((half, d_th)).astype(np.float32)
            th_pop = np.zeros((pop_sz, d_th), dtype=np.float32)
            th_pop[0] = mean_th
            th_pop[1] = mean_th
            th_pop[2 : 2 + half] = mean_th[None, :] + eps * sig_th[None, :]
            th_pop[2 + half :] = mean_th[None, :] - eps * sig_th[None, :]

            amp_v = 1.35 / (1.0 + np.exp(-np.clip(th_pop[:, 0], -8.0, 8.0)))
            amp_v[0] = 0.0  # слот 0 — неизменная текущая база!
            cen_v = np.repeat(arch["c"][None, :], pop_sz, axis=0)
            cen_v[:, axes] = th_pop[:, 1 : 1 + n_ax]
            cen_v[:, 14] = np.minimum(cen_v[:, 14], -0.20)
            if is_coupled_dipole or (allow_butterworth and arch["kind"] != "ask_harvest"):
                # Входной полюс строго при пустом портфеле (не наслаивает второй транш!):
                cen_v[:, 15] = 0.0
            if allow_butterworth and arch["kind"] != "ask_harvest":
                # Строгий фильтр подтверждённого восходящего 30m тренда против входов перед проливом BLK_06_C / BLK_02_Q:
                cen_v[:, 13] = np.maximum(cen_v[:, 13], +0.35)

            q_v = np.zeros((pop_sz, 16), dtype=np.float32)
            q_v[:, axes] = np.clip(np.exp(np.clip(th_pop[:, 1 + n_ax : 1 + 2 * n_ax], -1.5, 2.4)), 0.25, 11.0)
            q_v[:, 14] = np.maximum(q_v[:, 14], 1.85)
            q_v[:, 15] = np.maximum(q_v[:, 15], 8.00 if (is_coupled_dipole or allow_butterworth) else 2.50)
            if allow_butterworth and arch["kind"] != "ask_harvest":
                q_v[:, 13] = np.maximum(q_v[:, 13], 1.35)

            if allow_butterworth:
                p_v = np.clip(1.0 + 3.5 / (1.0 + np.exp(-np.clip(th_pop[:, 1 + 2 * n_ax], -6.0, 6.0))), 1.0, 4.5)
            else:
                p_v = np.zeros(pop_sz, dtype=np.float32)

            du_v = np.clip(th_pop[:, 2 + 2 * n_ax : 62 + 2 * n_ax], -18.0, 20.0)
            if arch["kind"] != "ask_harvest":
                du_v[:, 59] = np.clip(du_v[:, 59], 16.5, 20.0)
                du_v[:, 25] = np.clip(du_v[:, 25], -5.0, 0.4)

            valid_nyq, n_ep, n_dy, duty_v = check_nyquist_subspace(
                amp_v, cen_v, q_v, p_v, tr_basis, min_hits=min_hits, min_days=min_days
            )
            valid_nyq[0] = True

            banks_pop = []
            for ci in range(pop_sz):
                bk = clone_bank(cur_bank)
                if valid_nyq[ci] and ci > 0:
                    bk["res_amp"][0, slot_k] = amp_v[ci]
                    bk["res_center"][slot_k] = cen_v[ci]
                    bk["res_q"][slot_k] = q_v[ci]
                    bk["res_du"][slot_k] = du_v[ci]
                    bk["res_p"][0, slot_k] = p_v[ci]
                    if is_coupled_dipole:
                        lf = th_pop[ci, 62 + 2 * n_ax :]
                        # Слот #30: Прямоугольный Баттерворт-фильтр (p=6.0) непрерывного сопровождения открытого Agile Pod (inv in [0.07..0.23], 0.0004 при inv=0!):
                        bk["res_amp"][0, 30] = 1.00
                        bk["res_p"][0, 30] = 6.00
                        bk["res_center"][30] = 0.0
                        bk["res_center"][30, 15] = 0.15
                        bk["res_q"][30] = 0.0
                        bk["res_q"][30, 15] = 18.0
                        du30 = np.zeros(60, dtype=np.float32)
                        du30[0] = -8.0                         # Блокирует долив поверх уже взятого Agile Pod
                        du30[59] = +18.5                       # Удерживает 20% лимит
                        du30[13:17] = +4.2                     # Концентрирует мейкер-тейк в передних слотах Ask #00..#03
                        du30[42] = float(np.clip(lf[0], -2.5, 2.5))   # Целевой спред тейк-профита
                        du30[43] = float(np.clip(lf[1], 0.0, 6.0))    # Шаг Ask внутрь спреда
                        du30[52] = +14.0                       # Карусель нижнего транша ВКЛ на 100% жизни позиции!
                        du30[54] = float(np.clip(lf[2], 6.0, 18.0))   # Мейкер stop-unwind при уходе цены ниже входа!
                        du30[51] = +5.0                        # Быстрая реакция Ask-плана
                        bk["res_du"][30] = du30

                        # Слот #31: Пред-обвальный Taker-предохранитель (срезает 20% Agile Pod в 100% кэш ДО обвала BLK_06_C, 0.00001 при inv=0!):
                        bk["res_amp"][0, 31] = 1.15
                        bk["res_p"][0, 31] = 0.0
                        bk["res_center"][31] = 0.0
                        bk["res_center"][31, 14] = float(np.clip(lf[3], 0.10, 1.40))
                        bk["res_center"][31, 9] = float(np.clip(lf[5], -2.20, -0.25))
                        bk["res_center"][31, 15] = 0.15
                        bk["res_q"][31] = 0.0
                        bk["res_q"][31, 14] = float(np.clip(lf[4], 1.0, 4.0))
                        bk["res_q"][31, 9] = 1.10
                        bk["res_q"][31, 15] = 15.0
                        du31 = np.zeros(60, dtype=np.float32)
                        du31[0] = -10.0
                        du31[25] = +5.5                        # Мгновенный Taker-выход u_taker -> 1.0 при пробое щита!
                        du31[36] = +2.0
                        bk["res_du"][31] = du31
                else:
                    bk["res_amp"][0, slot_k] = 0.0
                banks_pop.append(bk)

            tr_res = pool.eval_pop(banks_pop, train_wins)
            c0 = tr_res[0]
            scores = np.full(pop_sz, -1e6, dtype=np.float32)

            for ci in range(1, pop_sz):
                if not valid_nyq[ci]:
                    continue
                rc = tr_res[ci]
                d_ev = rc["ev_4w_pct"] - c0["ev_4w_pct"]
                d_worst = min(rc["ev_dec_pct"] - c0["ev_dec_pct"], rc["ev_raw_pct"] - c0["ev_raw_pct"])
                dd_pen = max(0.0, rc["dd_max_pct"] - max(c0["dd_max_pct"] + 0.08, 6.25))
                sc = d_ev + 0.40 * d_worst - 3.0 * dd_pen
                scores[ci] = sc
                if d_ev > 0.008 and d_worst >= -0.005 and dd_pen <= 0.05:
                    best_cands_tr.append((float(sc), clone_bank(banks_pop[ci]), int(n_ep[ci]), int(n_dy[ci]), float(duty_v[ci])))

            # Оставляем топ-4 различных кандидата с Train для проверки консистентности:
            best_cands_tr.sort(key=lambda x: x[0], reverse=True)
            best_cands_tr = best_cands_tr[:4]

            elites = [i for i in np.argsort(-scores)[:mu_el] if scores[i] > -1e5]
            if elites:
                wu = w_rec[: len(elites)]
                wu /= wu.sum()
                el_m = th_pop[elites]
                var_m = (wu[:, None] * ((el_m - mean_th[None, :]) ** 2)).sum(axis=0)
                sig_th = np.clip(0.75 * sig_th + 0.25 * np.sqrt(var_m + 1e-6), 0.04, 1.80)
                if not allow_butterworth:
                    sig_th[1 + 2 * n_ax] = 0.0
                mean_th = (wu[:, None] * el_m).sum(axis=0)

            best_g = max((tr_res[i]["ev_4w_pct"] for i in range(1, pop_sz) if valid_nyq[i]), default=c0["ev_4w_pct"])
            _log(
                f"  Gen {gen + 1:02d}/{n_gens_band} ({time.time()-t_g0:.1f}s) | nyquist={int(valid_nyq[1:].sum()):2d}/31 | "
                f"Train_4W {c0['ev_4w_pct']:+.3f}% -> {best_g:+.3f}%"
            )
            status_dict["last_gen"] = f"{wave_tag} band {b_ord + 1} gen {gen + 1}/{n_gens_band}"
            status_dict["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_status(status_dict)

        if not best_cands_tr:
            _log(f"  ⚪ Полоса {arch['name']} не дала устойчивого прироста на Train — пропуск.")
            continue

        # Проверяем лучший Train-кандидат (строго 1 топ-кандидат с Train -> 0 selection bias на PurgedVal!):
        top_sc, top_bk, ep_c, dy_c, dt_c = best_cands_tr[0]
        val_chk = pool.eval_pop([cur_bank, top_bk] + [cur_bank] * (pop_sz - 2), val_wins)
        v0, v1 = val_chk[0], val_chk[1]
        dv_4w = v1["ev_4w_pct"] - v0["ev_4w_pct"]
        dv_dec = v1["ev_dec_pct"] - v0["ev_dec_pct"]
        dv_raw = v1["ev_raw_pct"] - v0["ev_raw_pct"]
        dd_ok = v1["dd_max_pct"] <= max(v0["dd_max_pct"] + 0.15, 5.50)

        _log(
            f"  🔎 PurgedVal полосы {arch['name']} (ep={ep_c}, days={dy_c}, duty={100*dt_c:.1f}%, p={top_bk['res_p'][0, slot_k]:.2f}): "
            f"{v0['ev_4w_pct']:+.3f}% -> {v1['ev_4w_pct']:+.3f}% (Δ4W={dv_4w:+.3f}%, ΔDEC={dv_dec:+.3f}%, ΔRAW={dv_raw:+.3f}%, DD={v1['dd_max_pct']:.2f}%)"
        )
        if dv_4w > 0.005 and dv_dec >= -0.008 and dv_raw >= -0.008 and dd_ok:
            cur_bank = top_bk
            locked_info.append(
                {
                    "slot": int(slot_k),
                    "name": arch["name"],
                    "amp": round(float(cur_bank["res_amp"][0, slot_k]), 4),
                    "order_p": round(float(cur_bank["res_p"][0, slot_k]), 3),
                    "episodes": ep_c,
                    "days": dy_c,
                    "duty_pct": round(100.0 * dt_c, 2),
                    "delta_val_4w_pct": round(dv_4w, 4),
                    "val_4w_pct": round(v1["ev_4w_pct"], 4),
                }
            )
            _log(f"  ✅ ПРИНЯТА ПОЛОСА slot#{slot_k} ({arch['name']}): PurgedVal_4W = {v1['ev_4w_pct']:+.3f}%/сут!")
        else:
            _log(f"  🛡️ Полоса {arch['name']} отсеяна на PurgedVal.")

    # Совместная тонкая полировка (Joint Multi-Band Polish) всех активных полос одновременно:
    active_slots = [s for s in range(32) if cur_bank["res_amp"][0, s] > 1e-4]
    if len(active_slots) >= 1 and n_gens_polish > 0 and time.time() < deadline_ts - 420:
        _log(f"✨ [{wave_tag}] Совместная полировка {len(active_slots)} активных полос ({n_gens_polish} поколений на 2 GPU)...")
        best_pol_bk = clone_bank(cur_bank)
        best_pol_sc = 0.0
        for gp in range(n_gens_polish):
            if time.time() > deadline_ts - 240:
                break
            half = (pop_sz - 2) // 2
            bks = [clone_bank(best_pol_bk), clone_bank(best_pol_bk)]
            scale_p = 0.12 * (0.88 ** (gp / 5.0))
            for _ in range(half):
                b_plus = clone_bank(best_pol_bk)
                b_minus = clone_bank(best_pol_bk)
                for sl in active_slots:
                    q_mask = (best_pol_bk["res_q"][sl] > 1e-4).astype(np.float32)
                    dc = rng.standard_normal(16).astype(np.float32) * scale_p * q_mask
                    dq = np.exp(rng.standard_normal(16).astype(np.float32) * (0.6 * scale_p) * q_mask)
                    ddu = rng.standard_normal(60).astype(np.float32) * (3.0 * scale_p)
                    da = float(rng.standard_normal() * (0.5 * scale_p))

                    b_plus["res_amp"][0, sl] = np.clip(best_pol_bk["res_amp"][0, sl] + da, 0.05, 1.40)
                    b_minus["res_amp"][0, sl] = np.clip(best_pol_bk["res_amp"][0, sl] - da, 0.05, 1.40)
                    b_plus["res_center"][sl] += dc
                    b_minus["res_center"][sl] -= dc
                    b_plus["res_q"][sl] = np.clip(best_pol_bk["res_q"][sl] * dq, 0.0, 12.0) * q_mask
                    b_minus["res_q"][sl] = np.clip(best_pol_bk["res_q"][sl] / dq, 0.0, 12.0) * q_mask
                    b_plus["res_du"][sl] = np.clip(best_pol_bk["res_du"][sl] + ddu, -18.0, 20.0)
                    b_minus["res_du"][sl] = np.clip(best_pol_bk["res_du"][sl] - ddu, -18.0, 20.0)
                    if best_pol_bk["res_du"][sl, 59] > 10.0:
                        b_plus["res_du"][sl, 59] = np.clip(b_plus["res_du"][sl, 59], 16.5, 20.0)
                        b_minus["res_du"][sl, 59] = np.clip(b_minus["res_du"][sl, 59], 16.5, 20.0)
                    if allow_butterworth and best_pol_bk["res_p"][0, sl] > 0.05:
                        dp = float(rng.standard_normal() * (1.2 * scale_p))
                        b_plus["res_p"][0, sl] = np.clip(best_pol_bk["res_p"][0, sl] + dp, 0.8, 5.0)
                        b_minus["res_p"][0, sl] = np.clip(best_pol_bk["res_p"][0, sl] - dp, 0.8, 5.0)
                bks.extend([b_plus, b_minus])

            pr = pool.eval_pop(bks, train_wins)
            p0 = pr[0]
            for ci in range(2, len(bks)):
                d_ev = pr[ci]["ev_4w_pct"] - p0["ev_4w_pct"]
                dd_ok = pr[ci]["dd_max_pct"] <= p0["dd_max_pct"] + 0.05
                if d_ev > 0.004 and dd_ok:
                    # Проверяем совместный кандидат на PurgedVal только при улучшении Train:
                    if d_ev > best_pol_sc:
                        best_pol_sc = d_ev
                        cand_pol = clone_bank(bks[ci])
                        vc = pool.eval_pop([best_pol_bk, cand_pol] + [best_pol_bk] * (pop_sz - 2), val_wins)
                        if vc[1]["ev_4w_pct"] > vc[0]["ev_4w_pct"] + 0.002 and vc[1]["dd_max_pct"] <= vc[0]["dd_max_pct"] + 0.10:
                            best_pol_bk = cand_pol
                            _log(f"  ✨ Polish Gen {gp + 1:02d}: PurgedVal_4W улучшен до {vc[1]['ev_4w_pct']:+.3f}%/сут!")
                            best_pol_sc = 0.0
        cur_bank = best_pol_bk

    end_val = pool.eval_pop([cur_bank] * pop_sz, val_wins)[0]
    return cur_bank, locked_info, end_val


def run_full_block_exam(bank: dict[str, np.ndarray], pool: DualGpuWorldPool, all_blocks: list[dict]) -> dict:
    """Честный поблочный экзамен по всем суточным корзинам C / D / U / Q и скрытому Rolling Blind OOS."""
    pad_bks = [bank] * 32
    worlds_rep: dict[str, dict] = {"DEC": {"per_block": []}, "RAW": {"per_block": []}}
    by_reg_d = {"C": [], "D": [], "U": [], "Q": [], "BLIND": []}
    by_reg_r = {"C": [], "D": [], "U": [], "Q": [], "BLIND": []}
    max_dd_d, max_dd_r = 0.0, 0.0

    for b in all_blocks:
        wb = [(b["block_id"], int(b["rel_start"]), 0.0, 0.0)]
        r0 = pool.eval_pop(pad_bks, wb)[0]
        rk = "BLIND" if b["role"] == "blind" else b["regime"]
        by_reg_d[rk].append(r0["ev_dec_pct"])
        by_reg_r[rk].append(r0["ev_raw_pct"])
        max_dd_d = max(max_dd_d, r0["dd_dec_pct"])
        max_dd_r = max(max_dd_r, r0["dd_raw_pct"])
        worlds_rep["DEC"]["per_block"].append(
            {"block_id": b["block_id"], "role": b["role"], "regime": b["regime"], "ev_mix_pct": round(r0["ev_dec_pct"], 3), "fills": round(r0["fills_dec"], 1), "dd_pct": round(r0["dd_dec_pct"], 2)}
        )
        worlds_rep["RAW"]["per_block"].append(
            {"block_id": b["block_id"], "role": b["role"], "regime": b["regime"], "ev_mix_pct": round(r0["ev_raw_pct"], 3), "fills": round(r0["fills_raw"], 1), "dd_pct": round(r0["dd_raw_pct"], 2)}
        )

    for wm, by_r, mdd in (("DEC", by_reg_d, max_dd_d), ("RAW", by_reg_r, max_dd_r)):
        pb = worlds_rep[wm]["per_block"]
        pre_b = [x["ev_mix_pct"] for x in pb if x["role"] != "blind"]
        val_b = [x["ev_mix_pct"] for x in pb if x["role"] == "val"]
        worlds_rep[wm]["mean_pre_blind_ev_d"] = round(float(np.mean(pre_b)), 3)
        worlds_rep[wm]["mean_val_ev_d"] = round(float(np.mean(val_b)), 3) if val_b else 0.0
        worlds_rep[wm]["regime_means_pct"] = {k: round(float(np.mean(v)), 3) if v else 0.0 for k, v in by_r.items()}
        worlds_rep[wm]["max_dd_pct"] = round(mdd, 2)
    return worlds_rep


def save_candidate_checkpoint(path: Path, carrier_raw_ck: dict, bank: dict[str, np.ndarray], tag: str, meta: dict) -> None:
    ck = dict(carrier_raw_ck)
    for k, v in bank.items():
        ck[k] = torch.from_numpy(v.copy()).float()
    ck["lesson"] = tag
    ck["l95_meta"] = meta
    torch.save(ck, path)


# ==============================================================================
# 4. ГЛАВНЫЙ 9-ЧАСОВОЙ КОНТУР (3 СИНХРОНИЗАЦИИ ДАННЫХ + 3 ВОЛНЫ АРХИТЕКТУРЫ + ФИНАЛ)
# ==============================================================================
def _load_bank_from_pt(pt_path: Path) -> dict[str, np.ndarray] | None:
    if not pt_path.exists():
        return None
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "res_amp" not in ck:
        return None
    bk = empty_res_bank()
    for k in bk:
        if k in ck:
            bk[k] = ck[k].detach().cpu().float().numpy().reshape(bk[k].shape)
    return bk


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-wave", type=int, choices=[1, 2, 3], default=None, help="Запустить строго одну волну (1, 2 или 3) в песочнице на 2 GPU")
    ap.add_argument("--hours", type=float, default=2.55, help="Бюджет времени для одиночной волны (в часах)")
    ap.add_argument("--n-gens-band", type=int, default=0, help="Переопределить число поколений на полосу (0 = по умолчанию)")
    ap.add_argument("--final-exam-only", action="store_true", help="Запустить только финальный турнир по сохранённым чекпоинтам волн (без сети)")
    args = ap.parse_args()

    t_marathon_start = time.time()
    total_budget_sec = (args.hours if args.single_wave else 9.0) * 3600.0
    t_marathon_end = t_marathon_start + total_budget_sec
    MARATHON_DIR.mkdir(parents=True, exist_ok=True)

    from _tools.regime_purged_split_l94 import SUBCENT_START, build_dynamic_regime_split, evaluate_per_regime_dominance
    from _tools.sync_live_exchange_and_bot_ledger import run_incremental_exchange_and_ledger_refresh
    from _tools.train_order3_champion import MultimodalMoEPolicyOrder3

    pol_cpu = MultimodalMoEPolicyOrder3(n_groups=4, n_budgets=6, device=torch.device("cpu"))
    carrier_cpu = pol_cpu.load_initial_champion(CHAMP_V1)
    carrier_raw_ck = torch.load(CHAMP_V1, map_location="cpu", weights_only=False)

    status = json.loads(STATUS_JSON.read_text(encoding="utf-8")) if STATUS_JSON.exists() else {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_end_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_marathon_start + 9.0 * 3600.0)),
        "phase": "init",
        "sync_checkpoints": [],
        "waves": {},
    }
    _save_status(status)

    hall_of_fame: dict[str, dict[str, np.ndarray]] = {"H0_Baseline": empty_res_bank()}
    best_overall_bank = empty_res_bank()
    for t_prev in ("L95A_Gauss", "L95B_Dipole", "L95C_Butterworth"):
        bk_prev = _load_bank_from_pt(BASE_DIR / "models" / "prl" / f"{t_prev.lower()}_best.pt")
        if bk_prev is not None:
            hall_of_fame[t_prev] = bk_prev
            best_overall_bank = clone_bank(bk_prev)

    if args.final_exam_only:
        wave_plan = []
    elif args.single_wave is not None:
        w_i = args.single_wave - 1
        gb_u = int(args.n_gens_band) if int(args.n_gens_band) > 0 else 20
        all_specs = [
            (0, "Wave1_SubspaceGauss", False, gb_u, 18, 95101, t_marathon_start + args.hours * 3600.0),
            (1, "Wave2_CoupledDipole", False, gb_u, 12, 95202, t_marathon_start + args.hours * 3600.0),
            (2, "Wave3_ButterworthPoly", True, gb_u, 12, 95303, t_marathon_start + args.hours * 3600.0),
        ]
        wave_plan = [all_specs[w_i]]
    else:
        wave_plan = [
            (0, "Sync1_Wave1_SubspaceGauss", False, 20, 18, 95101, t_marathon_start + 2.85 * 3600.0),
            (1, "Sync2_Wave2_CoupledDipole", False, 20, 22, 95202, t_marathon_start + 5.75 * 3600.0),
            (2, "Sync3_Wave3_ButterworthPoly", True, 20, 24, 95303, t_marathon_start + 8.35 * 3600.0),
        ]

    for w_idx, w_name, use_butt, n_gb, n_gp, seed_w, w_deadline in wave_plan:
        if args.single_wave is None:
            _log("=" * 96)
            _log(f"🔄 [ЧЕКПОИНТ ДАННЫХ #{w_idx + 1}/3] Выгрузка свежих стаканов L500 с бирж + реальных логов робота с VPS...")
            status["phase"] = f"data_sync_{w_idx + 1}"
            _save_status(status)
            try:
                sync_res = run_incremental_exchange_and_ledger_refresh(gpu_id_for_xgb="1")
                status["sync_checkpoints"].append(sync_res)
                _log(f"✅ Чекпоинт данных #{w_idx + 1} завершён: rows={sync_res['total_rows']:,} | O1_sealed={sync_res['o1_sealed_up_to']:,} | {sync_res['bot_log_sync']}")
                rc_aux = subprocess.run([str(BASE_DIR / ".venv/bin/python"), "_tools/deploy_vps.py"], cwd=str(BASE_DIR), check=False).returncode
                _log(f"📡 Свежие бустеры 1/2/2.5 порядков синхронизированы на боевой VPS (rc={rc_aux}).")
            except Exception as e_sync:
                _log(f"⚠️ Предупреждение синхронизации #{w_idx + 1} (продолжаем на последнем консистентном реестре): {e_sync}")
                status["sync_checkpoints"].append({"error": str(e_sync), "ts": time.strftime("%H:%M:%S")})
            _save_status(status)

        split_info = build_dynamic_regime_split()
        cap_law = split_info["capacity_law"]
        k_max = int(cap_law["k_max_bands"])
        min_hits = int(cap_law["min_hits_per_band"])
        min_days = int(cap_law["min_distinct_days_per_band"])

        train_blocks = [b for b in split_info["blocks"] if b["role"] == "train"]
        val_blocks = [b for b in split_info["blocks"] if b["role"] == "val"]
        train_wins = [(b["block_id"], int(b["rel_start"]), 0.0, 0.0) for b in train_blocks]
        jit_qd = list(split_info["jittered_train_starts_by_regime"].get("Q", [])) + list(split_info["jittered_train_starts_by_regime"].get("D", []))
        used_s = {int(b["rel_start"]) for b in train_blocks}
        for js in jit_qd:
            if all(abs(int(js) - bs) >= 14400 for bs in used_s):
                train_wins.append((f"JIT_QD_{js}", int(js), 0.0, 0.0))
                used_s.add(int(js))
                if len(train_wins) >= len(train_blocks) + 2:
                    break
        val_wins = [(b["block_id"], int(b["rel_start"]), 0.0, 0.0) for b in val_blocks]

        tr_basis = extract_train_basis_cpu_fast(carrier_cpu, SUBCENT_START, train_blocks)
        status["phase"] = w_name
        _save_status(status)

        pool = DualGpuWorldPool(SUBCENT_START, carrier_cpu)
        try:
            tag_short = "L95A_Gauss" if w_idx == 0 else ("L95B_Dipole" if w_idx == 1 else "L95C_Butterworth")
            # На каждой волне проводим мульти-стартовые прогоны пока не исчерпан бюджет времени волны!
            sub_run = 0
            wave_best_bk = clone_bank(best_overall_bank)
            wave_best_val = pool.eval_pop([wave_best_bk] * 32, val_wins)[0]["ev_4w_pct"]
            wave_locks_all = []

            while time.time() < w_deadline - 720:
                sub_run += 1
                # Чередуем продолжение от лучшего банка и чистый старт с новым зерном для разнообразия мод:
                init_bk = clone_bank(wave_best_bk) if (sub_run == 1 and w_idx > 0) else empty_res_bank()
                if use_butt and sub_run == 1 and np.any(wave_best_bk["res_amp"] > 1e-4):
                    init_bk = clone_bank(wave_best_bk)
                    active_sl = init_bk["res_amp"][0] > 1e-4
                    init_bk["res_p"][0, active_sl] = 2.20

                cand_bk, locks, v_metrics = run_evolution_wave(
                    wave_tag=f"{tag_short}_R{sub_run}",
                    start_bank=init_bk,
                    pool=pool,
                    tr_basis=tr_basis,
                    train_wins=train_wins,
                    val_wins=val_wins,
                    k_max=k_max,
                    min_hits=min_hits,
                    min_days=min_days,
                    n_gens_band=n_gb,
                    n_gens_polish=n_gp,
                    allow_butterworth=use_butt,
                    seed_rng=seed_w + sub_run * 97,
                    deadline_ts=w_deadline,
                    status_dict=status,
                )
                if v_metrics["ev_4w_pct"] > wave_best_val + 0.002:
                    wave_best_val = v_metrics["ev_4w_pct"]
                    wave_best_bk = clone_bank(cand_bk)
                    wave_locks_all = locks
                    best_overall_bank = clone_bank(cand_bk)

            ck_w_path = BASE_DIR / "models" / "prl" / f"{tag_short.lower()}_best.pt"
            save_candidate_checkpoint(ck_w_path, carrier_raw_ck, wave_best_bk, tag_short, {"val_4w_pct": wave_best_val, "locks": wave_locks_all})
            hall_of_fame[tag_short] = clone_bank(wave_best_bk)
            status["waves"][tag_short] = {"val_4w_pct": round(wave_best_val, 4), "sub_runs": sub_run, "locks": wave_locks_all}
            _save_status(status)
        finally:
            pool.close()

    if args.single_wave is not None:
        _log(f"✅ Одиночная волна #{args.single_wave} полностью завершена, чекпоинт сохранён!")
        return

    # Формируем ансамблевый банк Resonant Soup (объединение лучших ортогональных полос из волн A, B, C, если есть):
    elite_keys = [k for k in ("L95A_Gauss", "L95B_Dipole", "L95C_Butterworth") if k in hall_of_fame and np.any(hall_of_fame[k]["res_amp"] > 1e-4)]
    if len(elite_keys) >= 2:
        soup_bk = empty_res_bank()
        slot_ptr = 0
        w_share = 1.0 / float(len(elite_keys))
        for ek in elite_keys:
            eb = hall_of_fame[ek]
            for s_i in range(32):
                if eb["res_amp"][0, s_i] > 1e-4 and slot_ptr < 32:
                    soup_bk["res_amp"][0, slot_ptr] = eb["res_amp"][0, s_i] * w_share
                    soup_bk["res_center"][slot_ptr] = eb["res_center"][s_i]
                    soup_bk["res_q"][slot_ptr] = eb["res_q"][s_i]
                    soup_bk["res_du"][slot_ptr] = eb["res_du"][s_i]
                    soup_bk["res_p"][0, slot_ptr] = eb["res_p"][0, s_i]
                    slot_ptr += 1
        hall_of_fame["L95_ResonantSoup"] = soup_bk

    # ФИНАЛЬНЫЙ ЭКЗАМЕН НА СВЕЖЕМ УТРЕННЕМ РЕЕСТРЕ (ВСЕ БЛОКИ C/D/U/Q + СВЕЖИЙ ROLLING BLIND OOS!):
    _log("=" * 96)
    _log("🏁 ФИНАЛЬНЫЙ ВЫПУСКНОЙ ТУРНИР (H_0 vs L95A vs L95B vs L95C vs Soup) НА СВЕЖЕМ УТРЕННЕМ РЕЕСТРЕ")
    status["phase"] = "final_exam"
    _save_status(status)

    split_final = build_dynamic_regime_split()
    all_final_blocks = split_final["blocks"] + split_final["blind_blocks"]
    exam_results = {}

    pool_fin = DualGpuWorldPool(SUBCENT_START, carrier_cpu)
    try:
        for cand_name, cand_bk in hall_of_fame.items():
            _log(f"📊 Поблочный аудит {cand_name} на {len(all_final_blocks)} суточных окнах (DEC || RAW)...")
            exam_results[cand_name] = run_full_block_exam(cand_bk, pool_fin, all_final_blocks)
    finally:
        pool_fin.close()

    base_ex = exam_results["H0_Baseline"]
    winner_name = "H0_Baseline"
    winner_score = -1e9
    dominance_table = {}

    for cand_name, ex in exam_results.items():
        if cand_name == "H0_Baseline":
            continue
        p_dec, g_dec = evaluate_per_regime_dominance(
            ex["DEC"]["regime_means_pct"], base_ex["DEC"]["regime_means_pct"],
            ex["DEC"]["max_dd_pct"], base_ex["DEC"]["max_dd_pct"],
            ex["DEC"]["regime_means_pct"]["BLIND"], base_ex["DEC"]["regime_means_pct"]["BLIND"],
        )
        p_raw, g_raw = evaluate_per_regime_dominance(
            ex["RAW"]["regime_means_pct"], base_ex["RAW"]["regime_means_pct"],
            ex["RAW"]["max_dd_pct"], base_ex["RAW"]["max_dd_pct"],
            ex["RAW"]["regime_means_pct"]["BLIND"], base_ex["RAW"]["regime_means_pct"]["BLIND"],
        )
        both_ok = bool(p_dec and p_raw)
        gain_4w = 0.5 * (ex["DEC"]["mean_pre_blind_ev_d"] - base_ex["DEC"]["mean_pre_blind_ev_d"]) + 0.5 * (ex["RAW"]["mean_pre_blind_ev_d"] - base_ex["RAW"]["mean_pre_blind_ev_d"]) + 0.5 * (g_dec["delta_blind_pct"] + g_raw["delta_blind_pct"])
        dominance_table[cand_name] = {"both_passed": both_ok, "DEC": g_dec, "RAW": g_raw, "composite_gain_pct": round(gain_4w, 4)}
        _log(f"🔎 [{cand_name}] Gate={both_ok} | gain={gain_4w:+.3f}% | DEC={g_dec} | RAW={g_raw}")
        if both_ok and gain_4w > winner_score:
            winner_score = gain_4w
            winner_name = cand_name

    py = str(BASE_DIR / ".venv" / "bin" / "python")
    promoted = False
    if winner_name != "H0_Baseline":
        win_path = BASE_DIR / "models" / "prl" / "l95_resonant_candidate.pt"
        save_candidate_checkpoint(win_path, carrier_raw_ck, hall_of_fame[winner_name], winner_name, dominance_table[winner_name])
        rc_par = subprocess.run([py, "_tools/test_order3_sim_live_parity.py", str(win_path)], cwd=str(BASE_DIR), check=False).returncode
        if rc_par == 0:
            bak_p = BASE_DIR / "models" / "prl" / "release_champion_v1_pre_l95_9h_bak.pt"
            if not bak_p.exists():
                shutil.copy2(CHAMP_V1, bak_p)
            shutil.copy2(win_path, CHAMP_V1)
            promoted = True
            _log(f"🏆 ПОБЕДИТЕЛЬ 9-ЧАСОВОГО МАРАФОНА [{winner_name}] ПРОШЁЛ ВСЕ ВОРОТА И ПАРИТЕТ 0.00e+00 -> ОБНОВЛЁН release_champion_v1.pt!")
        else:
            _log(f"⚠️ Сбой теста паритета у {winner_name} (rc={rc_par}) — release_champion_v1.pt НЕ изменён!")
    else:
        _log("🛡️ Ни один кандидат не превзошёл строгий порог доминирования в обоих мирах DEC+RAW+Blind — сохранён эталон H_0, обновляются свежие бустеры 1/2/2.5 порядков на VPS.")

    rc_dep = -1
    if not args.final_exam_only:
        # Синхронизация свежих бустеров Order-1/Wave/Order-2/Order-2.5, кода моста и (если promoted) нового Чемпиона на боевой VPS!
        _log("🚀 Запуск deploy_vps.py для выкатки обновлённых бустеров 1/2/2.5 порядков и актуального моста на VPS...")
        rc_dep = subprocess.run([py, "_tools/deploy_vps.py"], cwd=str(BASE_DIR), check=False).returncode
        _log(f"✅ deploy_vps.py завершён с кодом rc={rc_dep}")

    final_rep = {
        "total_elapsed_hours": round((time.time() - t_marathon_start) / 3600.0, 2),
        "winner": winner_name,
        "promoted_to_live_champion": promoted,
        "deploy_vps_rc": rc_dep,
        "sync_checkpoints": status["sync_checkpoints"],
        "waves": status["waves"],
        "dominance_table": dominance_table,
        "exam_results": exam_results,
    }
    FINAL_JSON.write_text(json.dumps(final_rep, indent=2, ensure_ascii=False), encoding="utf-8")
    status["phase"] = "completed"
    status["winner"] = winner_name
    status["promoted"] = promoted
    _save_status(status)
    _log(f"🎉 9-ЧАСОВОЙ МАРАФОН НА 2 GPU УСПЕШНО ЗАВЕРШЁН! Итоговый отчёт: {FINAL_JSON}")


if __name__ == "__main__":
    main()
