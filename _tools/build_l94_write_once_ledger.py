#!/usr/bin/env python3
"""
_tools/build_l94_write_once_ledger.py
=============================================================================
Урок 94 (Шаг А): Вложенный каузальный реестр признаков порядков 1 -> 2 -> 2.5
с правилом неизменяемой записи (Write-Once OOS Feature Ledger) и суточным шагом 24ч:

  1. Аппаратная изоляция: СТРОГО 2-я видеокарта (CUDA_VISIBLE_DEVICES=1 -> cuda:0 внутри процесса).
  2. Трёхзонное обучение каждого суточного окна E_k = [t_k, t_k + 86400]:
       - Train     : [t_k - 5.5д .. t_k - 12ч - H_max]
       - InnerStop : [t_k - 12ч - H_max .. t_k - H_max]  (ранняя остановка СТРОГО ДО t_k!)
       - OOS Slice : [t_k .. t_k + 86400]                (применяется вслепую 1 раз и запечатывается!)
  3. Строгая вложенность порядков (Stacking Hygiene):
       - Order-1 (12 горизонтов 1s..1800s) + Wave (6 моделей до 3600s) запечатываются с эпохи 2,844,000
         (за 2 суток до старта суб-центовой эры 3,016,277).
       - Order-2 (16 моделей до 900s) обучается ИСКЛЮЧИТЕЛЬНО на уже запечатанных OOS-выходах Order-1!
       - Order-2.5 (4 канала ch[84..87]) обучается посуточно строго на интервенциях бота < t_k - 900с
         (устраняет in-sample перезапись прошлого боевым бустером!).
  4. O(1) дисковый пул моделей:
       - До-субцентовые и устаревшие окна win_00..XX очищаются; на диске хранятся ТОЛЬКО последние
         3 суточных окна (72ч буфер отката + актуальное live-окно для VPS).
       - Запечатанные исторические эпохи фиксируются в `data/cache/l94_ledger_manifest.json` и при
         последующих запусках переиспользуются за 0 секунд (досчитывается только новый суточный хвост!).
=============================================================================
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# ЖЁСТКАЯ ИЗОЛЯЦИЯ НА 2-Ю ВИДЕОКАРТУ ДО ИМПОРТА TORCH / XGBOOST / NUMBA!
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("L94_GPU_ID", "1")
os.environ["CUPY_CACHE_DIR"] = "/home/restorator/crypto_bot/scratch/.cupy_cache"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "_tools") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "_tools"))

from data_labeler import compute_single_horizon_labels
from train_order1_wave_wf import WAVE_MODEL_NAMES, precompute_wave_targets
from walkforward_order2_trainer import (
    CORE_TARGET_NAMES as O2_CORE_NAMES,
    TARGET_HORIZONS_SEC as O2_HORIZONS_SEC,
    TARGET_NAMES as O2_ALL_NAMES,
    build_66d_features,
    compute_numba_targets,
)
from walkforward_trainer import CORE_HORIZONS, EXTENDED_HORIZONS, HORIZONS as O1_HORIZONS, K_STAR

PARQUET_PATH = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
FEATS_PATH = BASE_DIR / "data" / "features" / "selected_features.json"
CACHE_DIR = BASE_DIR / "data" / "cache"
MANIFEST_PATH = CACHE_DIR / "l94_ledger_manifest.json"

O1_DIR = BASE_DIR / "models" / "prl" / "variant_a_order1"
OW_DIR = BASE_DIR / "models" / "prl" / "order1_wave_wf"
O2_DIR = BASE_DIR / "models" / "prl" / "order2_wf"
O25_DIR = BASE_DIR / "models" / "prl" / "order25_live"

SUBCENT_START = 3016277
O1_LEDGER_START = 2844000   # За 2 полных дня (172,277с) до SUBCENT_START, чтобы у Order-2 уже был запечатанный O1 OOS!
O2_LEDGER_START = 3016800   # Суб-центовая эра Order-2 (обучается на запечатанном O1 OOS >= 2,844,000!)
EPOCH_STEP = 86400          # Единый синхронный шаг реестра = ровно 24 часа!
INNER_STOP_LEN = 43200      # 12 часов пред-окно для early_stopping строго левее t_k - H_max!
TRAIN_LOOKBACK = 432000     # 5 суток чистого обучающего окна левее InnerStop!
KEEP_RECENT_WINS = 3        # Храним на диске ровно 3 последних окна (.ubj) -> O(1) размер папки models/!

CLASS_VALS = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_24h_epochs(start_row: int, total_rows: int) -> list[tuple[int, int, int]]:
    """Строит список суточных эпох (epoch_id, s_idx, e_idx) с шагом 86,400с до конца датасета."""
    epochs: list[tuple[int, int, int]] = []
    ep_id = 0
    cur = start_row
    while cur < total_rows:
        nxt = min(total_rows, cur + EPOCH_STEP)
        # Если остаточный хвостик меньше 3 часов (<10,800с), присоединяем к предыдущей эпохе, иначе отдельная неполная текущая эпоха
        if epochs and (nxt - cur) < 10800:
            prev_id, prev_s, _ = epochs[-1]
            epochs[-1] = (prev_id, prev_s, nxt)
            break
        epochs.append((ep_id, cur, nxt))
        cur = nxt
        ep_id += 1
    return epochs


def _read_parquet_window(
    pf: pq.ParquetFile,
    rg_offsets: list[int],
    rg_meta: list,
    num_cols: list[str],
    lbl_cols: list[str],
    lo: int,
    hi: int,
) -> tuple[np.ndarray, "pd.DataFrame"]:
    all_cols = list(dict.fromkeys(num_cols + lbl_cols))
    needed = [i for i, (off, m) in enumerate(zip(rg_offsets, rg_meta)) if not (hi <= off or lo >= off + m.num_rows)]
    tbl = pa.concat_tables([pf.read_row_group(i, columns=all_cols) for i in needed])
    sub_tbl = tbl.slice(lo - rg_offsets[needed[0]], hi - lo)
    df_lbl = sub_tbl.select(lbl_cols).to_pandas()
    X_np = np.column_stack([sub_tbl[c].to_numpy(zero_copy_only=False) for c in num_cols]).astype(np.float32)
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
    del tbl, sub_tbl
    return X_np, df_lbl


def _prune_old_windows(mod_dir: Path, keep_ids: set[int]) -> int:
    removed = 0
    if not mod_dir.exists():
        return 0
    for d in list(mod_dir.glob("win_*")):
        if not d.is_dir():
            continue
        try:
            w_num = int(d.name.split("_")[1])
        except Exception:
            continue
        if w_num not in keep_ids:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def _train_o1_booster(X_tr: np.ndarray, y_tr: np.ndarray, X_es: np.ndarray, y_es: np.ndarray, H: int) -> xgb.Booster:
    if H >= 900:
        max_depth, reg_lam, min_cw, sub_s = 4, 6.0, 64.0, 0.70
    elif H >= 320:
        max_depth, reg_lam, min_cw, sub_s = 5, 3.0, 32.0, 0.75
    else:
        max_depth, reg_lam, min_cw, sub_s = 6, 1.0, 1.0, 0.80
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    des = xgb.DMatrix(X_es, label=y_es)
    params = {
        "max_depth": max_depth,
        "learning_rate": 0.05,
        "objective": "multi:softprob",
        "num_class": 7,
        "tree_method": "hist",
        "device": "cuda:0",
        "subsample": sub_s,
        "colsample_bytree": 0.80,
        "reg_lambda": reg_lam,
        "min_child_weight": min_cw,
        "verbosity": 0,
        "seed": 42,
    }
    bst = xgb.train(params, dtr, num_boost_round=75, evals=[(des, "es")], early_stopping_rounds=12, verbose_eval=False)
    del dtr, des
    return bst


def _train_wave_booster(X_tr: np.ndarray, y_tr: np.ndarray, X_es: np.ndarray, y_es: np.ndarray, m_name: str) -> xgb.Booster:
    is_cls = m_name in ("wave_launch_30m", "wave_top_15m")
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    des = xgb.DMatrix(X_es, label=y_es)
    params = {
        "max_depth": 4,
        "learning_rate": 0.04,
        "tree_method": "hist",
        "device": "cuda:0",
        "subsample": 0.72,
        "colsample_bytree": 0.70,
        "reg_lambda": 8.0,
        "reg_alpha": 1.5,
        "min_child_weight": 64.0,
        "verbosity": 0,
        "seed": 202611,
    }
    if is_cls:
        pos_n = float(np.sum(y_tr > 0.5))
        neg_n = float(len(y_tr)) - pos_n
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "auc"
        params["scale_pos_weight"] = max(1.0, min(15.0, neg_n / max(1.0, pos_n)))
        if len(np.unique(y_es > 0.5)) < 2:
            bst = xgb.train(params, dtr, num_boost_round=65, verbose_eval=False)
            del dtr, des
            return bst
    else:
        params["objective"] = "reg:pseudohubererror"
        params["eval_metric"] = "mae"
    bst = xgb.train(params, dtr, num_boost_round=160, evals=[(des, "es")], early_stopping_rounds=20, verbose_eval=False)
    del dtr, des
    return bst


def run_step1_order1_and_wave(manifest: dict, force_rebuild: bool = False) -> None:
    _log("=" * 88)
    _log("🔷 ШАГ А.1: ТРЁХЗОННЫЙ 24h WALK-FORWARD РЕЕСТР ORDER-1 (12H) + WAVE (6D) НА GPU #1")
    _log("=" * 88)

    pf = pq.ParquetFile(str(PARQUET_PATH))
    total_rows = pf.metadata.num_rows
    schema_set = set(pf.schema.names)
    raw_num_cols = json.loads(FEATS_PATH.read_text())["feature_order"]
    num_cols = [c for c in raw_num_cols if c in schema_set]
    lbl_cols = [
        c for c in (
            "ST_mid_price", "ST_spread", "ST_bid_1_price", "ST_ask_1_price",
            "ST_bid_1_qty", "ST_ask_1_qty", "ST_bid_2_qty", "ST_ask_2_qty",
            "ST_bids_vol_top5", "ST_asks_vol_top5", "ST_trade_vol",
            "ST_sec_since_tick", "BO_sec_since_tick", "arb_spread_zscore_20s",
        ) if c in schema_set
    ]
    rg_meta = [pf.metadata.row_group(i) for i in range(pf.metadata.num_row_groups)]
    rg_offsets = [0]
    for m in rg_meta[:-1]:
        rg_offsets.append(rg_offsets[-1] + m.num_rows)

    lob_raw = np.load(CACHE_DIR / "rl_guard_lob_cache_600h_raw_with_self.npz", mmap_mode="r")
    b1_all = np.asarray(lob_raw["bid_p"][:, 0], dtype=np.float32)
    a1_all = np.asarray(lob_raw["ask_p"][:, 0], dtype=np.float32)

    o1_56_p = CACHE_DIR / "order1_alpha_matrix_56d.npz"
    o1_84_p = CACHE_DIR / "order1_alpha_matrix_84d.npz"
    ow_6_p = CACHE_DIR / "order1_wave_matrix_6d.npz"

    probs_84d = np.zeros((total_rows, 84), dtype=np.float16)
    probs_84d[:, 3::7] = 1.0  # нейтральный априор для до-субцентового префикса < O1_LEDGER_START
    ez_12h = np.zeros((total_rows, 12), dtype=np.float32)
    conf_12h = np.zeros((total_rows, 12), dtype=np.float32)
    bias_12h = np.zeros((total_rows, 12), dtype=np.float32)
    wave_6d = np.zeros((total_rows, 6), dtype=np.float32)

    sealed_o1_up_to = int(manifest.get("o1_sealed_up_to", 0)) if not force_rebuild else 0
    if sealed_o1_up_to > O1_LEDGER_START and o1_84_p.exists() and ow_6_p.exists():
        old84 = np.load(o1_84_p, mmap_mode="r")
        oldw6 = np.load(ow_6_p, mmap_mode="r")
        cut = min(sealed_o1_up_to, len(old84["expected_z_12h"]), len(oldw6["wave_matrix_6d"]))
        probs_84d[:cut] = old84["probs_84d"][:cut]
        ez_12h[:cut] = old84["expected_z_12h"][:cut]
        conf_12h[:cut] = old84["confidence_12h"][:cut]
        bias_12h[:cut] = old84["bias_12h"][:cut]
        wave_6d[:cut] = oldw6["wave_matrix_6d"][:cut]
        sealed_o1_up_to = cut
        _log(f"⚡ Переиспользован запечатанный Write-Once префикс O1+Wave [0 .. {sealed_o1_up_to:,}]!")
    else:
        sealed_o1_up_to = 0

    epochs = build_24h_epochs(O1_LEDGER_START, total_rows)
    # Также в конце обучим win_live (id = 900 + len(epochs)) вплоть до N для живого VPS!
    O1_DIR.mkdir(parents=True, exist_ok=True)
    OW_DIR.mkdir(parents=True, exist_ok=True)
    keep_win_ids: set[int] = set()
    recent_ep_ids = {ep[0] + 100 for ep in epochs[-KEEP_RECENT_WINS:]}
    live_win_id = 100 + len(epochs)
    keep_win_ids.update(recent_ep_ids)
    keep_win_ids.add(live_win_id)

    for ep_id, s_idx, e_idx in epochs:
        win_num = 100 + ep_id
        is_full_24h = (e_idx - s_idx) >= EPOCH_STEP
        if e_idx <= sealed_o1_up_to and (win_num not in keep_win_ids or all((O1_DIR / f"win_{win_num:03d}/xgb_h{H}s.ubj").exists() for H in O1_HORIZONS)):
            continue

        t0_ep = time.time()
        # Строгая 3-зонная граница: максимальный горизонт разметки H_max = 3600с + 60с буфер = 3660с ДО s_idx!
        purge_h = 3660
        es_end = s_idx - purge_h
        es_start = es_end - INNER_STOP_LEN
        tr_end = es_start - purge_h
        tr_start = max(172800, tr_end - TRAIN_LOOKBACK)

        X_hist, df_lbl_hist = _read_parquet_window(pf, rg_offsets, rg_meta, num_cols, lbl_cols, tr_start, es_end)
        gap_h = ((df_lbl_hist["ST_sec_since_tick"] > 60) | (df_lbl_hist["BO_sec_since_tick"] > 60)).to_numpy()

        n_tr_raw = tr_end - tr_start
        es_off = es_start - tr_start
        # Подвыборка с шагом stride=2 на 5-суточном окне даёт 216k чистых некоррелированных секунд и ускоряет обучение в 2 раза!
        tr_sel = np.arange(0, n_tr_raw, 2, dtype=np.int64)
        tr_sel = tr_sel[~gap_h[tr_sel]]
        es_sel = np.arange(es_off, len(X_hist), 2, dtype=np.int64)
        es_sel = es_sel[~gap_h[es_sel]]

        X_tr = X_hist[tr_sel]
        X_es = X_hist[es_sel]

        X_oos, _ = _read_parquet_window(pf, rg_offsets, rg_meta, num_cols, [], s_idx, e_idx)
        d_oos = xgb.DMatrix(X_oos)

        save_this_win = win_num in keep_win_ids
        w1_dir = O1_DIR / f"win_{win_num:03d}"
        ww_dir = OW_DIR / f"win_{win_num:03d}"
        if save_this_win:
            w1_dir.mkdir(parents=True, exist_ok=True)
            ww_dir.mkdir(parents=True, exist_ok=True)

        tick_ep = 0.0001 if tr_end >= SUBCENT_START else 0.01
        for h_i, H in enumerate(O1_HORIZONS):
            K = K_STAR[H]
            lbl_all = compute_single_horizon_labels(df_lbl_hist, "prl", H, K, min_vol=0, tick_size=tick_ep, venue="ST")
            y_cls = (np.asarray(lbl_all["class_7"]) + 3).clip(0, 6).astype(np.int64)
            bst = _train_o1_booster(X_tr, y_cls[tr_sel], X_es, y_cls[es_sel], H)
            if save_this_win:
                bst.save_model(str(w1_dir / f"xgb_h{H}s.ubj"))
            p7 = bst.predict(d_oos).astype(np.float32)
            probs_84d[s_idx:e_idx, h_i * 7 : (h_i + 1) * 7] = p7.astype(np.float16)
            ez_12h[s_idx:e_idx, h_i] = p7 @ CLASS_VALS
            conf_12h[s_idx:e_idx, h_i] = 1.0 - p7[:, 3]
            bias_12h[s_idx:e_idx, h_i] = (p7[:, 4] + p7[:, 5] + p7[:, 6]) - (p7[:, 0] + p7[:, 1] + p7[:, 2])
            del bst, p7

        # Wave 6D (строго по b1_all / a1_all <= s_idx!):
        w_targs_tr = precompute_wave_targets(b1_all[:s_idx], a1_all[:s_idx], tr_start, es_end)
        for m_i, m_nm in enumerate(WAVE_MODEL_NAMES):
            y_w = w_targs_tr[m_nm]
            bst_w = _train_wave_booster(X_tr, y_w[tr_sel], X_es, y_w[es_sel], m_nm)
            if save_this_win:
                bst_w.save_model(str(ww_dir / f"{m_nm}.ubj"))
            wave_6d[s_idx:e_idx, m_i] = bst_w.predict(d_oos).astype(np.float32)
            del bst_w

        if save_this_win:
            (w1_dir / f"meta_win_{win_num:03d}.json").write_text(json.dumps({"completed": True, "oos_range": [s_idx, e_idx]}, indent=2))
            (ww_dir / "metrics.json").write_text(json.dumps({"completed": True, "oos_range": [s_idx, e_idx]}, indent=2))

        if is_full_24h:
            sealed_o1_up_to = max(sealed_o1_up_to, e_idx)

        del X_hist, df_lbl_hist, X_tr, X_es, X_oos, d_oos, w_targs_tr
        gc.collect()
        _log(f"  ✅ [O1+Wave Ep {ep_id:02d}/{len(epochs)-1}] OOS [{s_idx:,}..{e_idx:,}] запечатан за {time.time()-t0_ep:.1f}с (Train<{tr_end:,}, ES<{es_end:,})")

    # Обучаем свежее боевое окно win_live вплоть до самого конца датасета N (для 0ч свежести на VPS!):
    t0_lv = time.time()
    lv_es_end = total_rows - 3660
    lv_es_start = lv_es_end - INNER_STOP_LEN
    lv_tr_end = lv_es_start - 3660
    lv_tr_start = max(SUBCENT_START, lv_tr_end - TRAIN_LOOKBACK)
    X_hl, df_hl = _read_parquet_window(pf, rg_offsets, rg_meta, num_cols, lbl_cols, lv_tr_start, lv_es_end)
    gap_l = ((df_hl["ST_sec_since_tick"] > 60) | (df_hl["BO_sec_since_tick"] > 60)).to_numpy()
    tr_l = np.arange(0, lv_tr_end - lv_tr_start, 2, dtype=np.int64); tr_l = tr_l[~gap_l[tr_l]]
    es_l = np.arange(lv_es_start - lv_tr_start, len(X_hl), 2, dtype=np.int64); es_l = es_l[~gap_l[es_l]]
    w1_lv = O1_DIR / f"win_{live_win_id:03d}"; w1_lv.mkdir(parents=True, exist_ok=True)
    ww_lv = OW_DIR / f"win_{live_win_id:03d}"; ww_lv.mkdir(parents=True, exist_ok=True)
    for H in O1_HORIZONS:
        lbl_l = compute_single_horizon_labels(df_hl, "prl", H, K_STAR[H], min_vol=0, tick_size=0.0001, venue="ST")
        yc_l = (np.asarray(lbl_l["class_7"]) + 3).clip(0, 6).astype(np.int64)
        bst_l = _train_o1_booster(X_hl[tr_l], yc_l[tr_l], X_hl[es_l], yc_l[es_l], H)
        bst_l.save_model(str(w1_lv / f"xgb_h{H}s.ubj"))
        del bst_l
    wt_l = precompute_wave_targets(b1_all, a1_all, lv_tr_start, lv_es_end)
    for m_nm in WAVE_MODEL_NAMES:
        bst_wl = _train_wave_booster(X_hl[tr_l], wt_l[m_nm][tr_l], X_hl[es_l], wt_l[m_nm][es_l], m_nm)
        bst_wl.save_model(str(ww_lv / f"{m_nm}.ubj"))
        del bst_wl
    (w1_lv / f"meta_win_{live_win_id:03d}.json").write_text(json.dumps({"completed": True, "live_head_rows": total_rows}, indent=2))
    (ww_lv / "metrics.json").write_text(json.dumps({"completed": True, "live_head_rows": total_rows}, indent=2))
    del X_hl, df_hl, wt_l
    gc.collect()
    _log(f"  🟢 Свежее боевое окно win_{live_win_id:03d} (O1+Wave до строки {total_rows:,}) готово за {time.time()-t0_lv:.1f}с!")

    # Сохраняем запечатанные O1 + Wave кэши атомарно:
    t_56 = o1_56_p.with_name("order1_alpha_matrix_56d_tmp.npz")
    np.savez(
        t_56,
        probs_56d=probs_84d[:, :56],
        expected_z_8h=ez_12h[:, :8],
        confidence_8h=conf_12h[:, :8],
        bias_8h=bias_12h[:, :8],
    )
    t_56.replace(o1_56_p)

    t_84 = o1_84_p.with_name("order1_alpha_matrix_84d_tmp.npz")
    np.savez(
        t_84,
        probs_84d=probs_84d,
        expected_z_12h=ez_12h,
        confidence_12h=conf_12h,
        bias_12h=bias_12h,
    )
    t_84.replace(o1_84_p)

    t_w6 = ow_6_p.with_name("order1_wave_matrix_6d_tmp.npz")
    np.savez(t_w6, wave_matrix_6d=wave_6d, model_names=np.array(WAVE_MODEL_NAMES))
    t_w6.replace(ow_6_p)

    # Обновляем walkforward_summary.json и удаляем старые win_00..89 (оставляя только последние 3 + live):
    wf_sum = {f"win_{ep_id+100:03d}": {"val_range": [s_i, e_i]} for ep_id, s_i, e_i in epochs[-KEEP_RECENT_WINS:]}
    wf_sum[f"win_{live_win_id:03d}"] = {"val_range": [epochs[-1][1], total_rows]}
    (O1_DIR / "walkforward_summary.json").write_text(json.dumps(wf_sum, indent=2))

    rm_o1 = _prune_old_windows(O1_DIR, keep_win_ids)
    rm_ow = _prune_old_windows(OW_DIR, keep_win_ids)
    _log(f"🧹 Очистка O(1) пула: удалено {rm_o1} старых папок Order-1 и {rm_ow} папок Wave (оставлены {sorted(keep_win_ids)})!")
    manifest["o1_sealed_up_to"] = int(sealed_o1_up_to)


def _train_o2_booster(X_tr: np.ndarray, y_tr: np.ndarray, X_es: np.ndarray, y_es: np.ndarray, name: str, feat_names: list[str]) -> xgb.Booster:
    h_sec = int(O2_HORIZONS_SEC.get(name, 60))
    is_reg = name == "carousel_capacity"
    if h_sec >= 600:
        m_depth, r_lam, sub_s = 5, 6.0, 0.75
    elif h_sec >= 180:
        m_depth, r_lam, sub_s = 6, 4.0, 0.80
    else:
        m_depth, r_lam, sub_s = 7, 3.5, 0.82

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    des = xgb.DMatrix(X_es, label=y_es, feature_names=feat_names)
    if not is_reg:
        n_pos = float(np.sum(y_tr > 0.5))
        n_neg = float(len(y_tr)) - n_pos
        params = {
            "tree_method": "hist",
            "device": "cuda:0",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": m_depth,
            "learning_rate": 0.04,
            "subsample": sub_s,
            "colsample_bytree": 0.82,
            "scale_pos_weight": max(1.0, min(25.0, n_neg / max(1.0, n_pos))),
            "reg_lambda": r_lam,
            "reg_alpha": 1.0,
            "verbosity": 0,
            "seed": 42,
        }
        if len(np.unique(y_es > 0.5)) < 2:
            bst = xgb.train(params, dtr, num_boost_round=85, verbose_eval=False)
            del dtr, des
            return bst
    else:
        params = {
            "tree_method": "hist",
            "device": "cuda:0",
            "objective": "reg:pseudohubererror",
            "eval_metric": "mae",
            "max_depth": m_depth,
            "learning_rate": 0.04,
            "subsample": sub_s,
            "colsample_bytree": 0.82,
            "reg_lambda": r_lam,
            "verbosity": 0,
            "seed": 42,
        }
    bst = xgb.train(params, dtr, num_boost_round=240, evals=[(des, "es")], early_stopping_rounds=22, verbose_eval=False)
    del dtr, des
    return bst


def run_step2_order2_on_sealed_o1(manifest: dict, force_rebuild: bool = False) -> None:
    _log("=" * 88)
    _log("🔶 ШАГ А.2: ТРЁХЗОННЫЙ 24h WALK-FORWARD ORDER-2 (16D) СТРОГО НА ЗАПЕЧАТАННОМ O1 OOS")
    _log("=" * 88)

    lob_raw = np.load(CACHE_DIR / "rl_guard_lob_cache_600h_raw_with_self.npz", allow_pickle=True)
    alpha56 = np.load(CACHE_DIR / "order1_alpha_matrix_56d.npz", allow_pickle=True)
    total_rows = len(lob_raw["bid_p"])
    X_66, feat_names = build_66d_features(lob_raw, alpha56)

    o2_11_p = CACHE_DIR / "order2_risk_matrix_11d.npz"
    o2_16_p = CACHE_DIR / "order2_risk_matrix_16d.npz"
    oos_16d = np.zeros((total_rows, len(O2_ALL_NAMES)), dtype=np.float32)

    sealed_o2_up_to = int(manifest.get("o2_sealed_up_to", 0)) if not force_rebuild else 0
    if sealed_o2_up_to > O2_LEDGER_START and o2_16_p.exists():
        old16 = np.load(o2_16_p, mmap_mode="r")
        cut = min(sealed_o2_up_to, len(old16["risk_matrix_16d"]))
        oos_16d[:cut] = old16["risk_matrix_16d"][:cut]
        sealed_o2_up_to = cut
        _log(f"⚡ Переиспользован запечатанный Write-Once префикс Order-2 [0 .. {sealed_o2_up_to:,}]!")
    else:
        sealed_o2_up_to = 0

    epochs = build_24h_epochs(O2_LEDGER_START, total_rows)
    O2_DIR.mkdir(parents=True, exist_ok=True)
    live_win_id = 100 + len(build_24h_epochs(O1_LEDGER_START, total_rows))
    keep_win_ids = {102 + ep[0] for ep in epochs[-KEEP_RECENT_WINS:]} | {live_win_id}

    bp1 = lob_raw["bid_p"][:, 0].astype(np.float32)
    ap1 = lob_raw["ask_p"][:, 0].astype(np.float32)
    bq1 = lob_raw["bid_q"][:, 0].astype(np.float32)
    aq1 = lob_raw["ask_q"][:, 0].astype(np.float32)
    bv = np.nan_to_num(lob_raw["buy_vol"].astype(np.float32), nan=0.0)
    sv = np.nan_to_num(lob_raw["sell_vol"].astype(np.float32), nan=0.0)
    mid = 0.5 * (bp1 + ap1)
    # Вычисляем физические таргеты один раз (но при обучении окна k берём срез строго < s_idx - 960!):
    t_arrays = compute_numba_targets(bp1, ap1, bq1, aq1, sv, bv, mid * 0.0015, mid * 0.0030, mid * 0.0075, total_rows)
    t_dict = {nm: t_arrays[i] for i, nm in enumerate(O2_ALL_NAMES)}

    for ep_id, s_idx, e_idx in epochs:
        win_num = 102 + ep_id  # синхронная нумерация с O1 (где 0,1 = два пред-субцентовых дня!)
        is_full_24h = (e_idx - s_idx) >= EPOCH_STEP
        if e_idx <= sealed_o2_up_to and (win_num not in keep_win_ids or all((O2_DIR / f"win_{win_num:03d}/{nm}.ubj").exists() for nm in O2_ALL_NAMES)):
            continue

        t0_ep = time.time()
        purge_o2 = 960  # > 900s макс. горизонта Order-2!
        es_end = s_idx - purge_o2
        es_start = es_end - INNER_STOP_LEN
        tr_end = es_start - purge_o2
        # ВАЖНО: tr_start >= O1_LEDGER_START (2,844,000), поэтому Order-2 учится ТОЛЬКО на запечатанном O1 OOS!
        tr_start = max(O1_LEDGER_START, tr_end - TRAIN_LOOKBACK)

        tr_idx = np.arange(tr_start, tr_end, 2, dtype=np.int64)
        es_idx = np.arange(es_start, es_end, 2, dtype=np.int64)
        X_tr = X_66[tr_idx]
        X_es = X_66[es_idx]
        d_oos = xgb.DMatrix(X_66[s_idx:e_idx], feature_names=feat_names)

        save_this = win_num in keep_win_ids
        w2_dir = O2_DIR / f"win_{win_num:03d}"
        if save_this:
            w2_dir.mkdir(parents=True, exist_ok=True)

        for c_i, nm in enumerate(O2_ALL_NAMES):
            y_full = t_dict[nm]
            bst2 = _train_o2_booster(X_tr, y_full[tr_idx], X_es, y_full[es_idx], nm, feat_names)
            if save_this:
                bst2.save_model(str(w2_dir / f"{nm}.ubj"))
            oos_16d[s_idx:e_idx, c_i] = bst2.predict(d_oos).astype(np.float32)
            del bst2

        if save_this:
            (w2_dir / f"meta_order2_win_{win_num:03d}.json").write_text(json.dumps({"completed": True, "oos_range": [s_idx, e_idx]}, indent=2))
        if is_full_24h:
            sealed_o2_up_to = max(sealed_o2_up_to, e_idx)
        del X_tr, X_es, d_oos
        gc.collect()
        _log(f"  ✅ [Order-2 Ep {ep_id:02d}/{len(epochs)-1}] OOS [{s_idx:,}..{e_idx:,}] запечатан за {time.time()-t0_ep:.1f}с (O1_OOS Train>={tr_start:,})")

    # Свежее боевое окно win_live для VPS:
    t0_lv = time.time()
    lv_es_e = total_rows - 960
    lv_es_s = lv_es_e - INNER_STOP_LEN
    lv_tr_e = lv_es_s - 960
    lv_tr_s = max(SUBCENT_START, lv_tr_e - TRAIN_LOOKBACK)
    tr_l = np.arange(lv_tr_s, lv_tr_e, 2, dtype=np.int64)
    es_l = np.arange(lv_es_s, lv_es_e, 2, dtype=np.int64)
    w2_lv = O2_DIR / f"win_{live_win_id:03d}"; w2_lv.mkdir(parents=True, exist_ok=True)
    for nm in O2_ALL_NAMES:
        bst2_l = _train_o2_booster(X_66[tr_l], t_dict[nm][tr_l], X_66[es_l], t_dict[nm][es_l], nm, feat_names)
        bst2_l.save_model(str(w2_lv / f"{nm}.ubj"))
        del bst2_l
    (w2_lv / f"meta_order2_win_{live_win_id:03d}.json").write_text(json.dumps({"completed": True, "live_head_rows": total_rows}, indent=2))
    _log(f"  🟢 Свежее боевое окно Order-2 win_{live_win_id:03d} готово за {time.time()-t0_lv:.1f}с!")

    # Мягкое заполнение пред-субцентового старта [SUBCENT_START .. O2_LEDGER_START] (523 строки = 8.7 мин) первым окном:
    if O2_LEDGER_START > SUBCENT_START:
        oos_16d[SUBCENT_START:O2_LEDGER_START] = oos_16d[O2_LEDGER_START : O2_LEDGER_START + 1]

    t11 = o2_11_p.with_name("order2_risk_matrix_11d_tmp.npz")
    np.savez_compressed(t11, risk_matrix_11d=oos_16d[:, :len(O2_CORE_NAMES)], target_names=np.array(O2_CORE_NAMES), total_rows=total_rows)
    t11.replace(o2_11_p)

    t16 = o2_16_p.with_name("order2_risk_matrix_16d_tmp.npz")
    np.savez_compressed(t16, risk_matrix_16d=oos_16d, target_names=np.array(O2_ALL_NAMES), total_rows=total_rows)
    t16.replace(o2_16_p)

    rm_o2 = _prune_old_windows(O2_DIR, keep_win_ids)
    _log(f"🧹 Очистка O(1) пула Order-2: удалено {rm_o2} старых папок (оставлены {sorted(keep_win_ids)})!")
    manifest["o2_sealed_up_to"] = int(sealed_o2_up_to)
    del X_66, t_arrays, t_dict
    gc.collect()


def _fit_o25_causal_pair(X_sub: np.ndarray, y_sub: np.ndarray, max_depth: int, reg_lam: float) -> xgb.Booster | None:
    if len(X_sub) < 60 or len(np.unique(y_sub)) < 2:
        return None
    sp = max(40, int(len(X_sub) * 0.80))
    if len(np.unique(y_sub[:sp])) < 2:
        return None
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda:0",
        "max_depth": max_depth,
        "learning_rate": 0.04,
        "subsample": 0.75,
        "colsample_bytree": 0.65,
        "min_child_weight": 20,
        "reg_lambda": reg_lam,
        "reg_alpha": 1.5,
        "verbosity": 0,
        "seed": 20261124,
    }
    dtr = xgb.DMatrix(X_sub[:sp], label=y_sub[:sp])
    if len(X_sub) - sp >= 15 and len(np.unique(y_sub[sp:])) >= 2:
        des = xgb.DMatrix(X_sub[sp:], label=y_sub[sp:])
        return xgb.train(params, dtr, num_boost_round=160, evals=[(des, "es")], early_stopping_rounds=22, verbose_eval=False)
    return xgb.train(params, dtr, num_boost_round=65, verbose_eval=False)


def run_step3_assemble_112d_and_wf_o25(manifest: dict) -> None:
    _log("=" * 88)
    _log("🟣 ШАГ А.3: СБОРКА 84D (RAW+DEC) -> ПОСУТОЧНЫЙ WF ORDER-2.5 (БЕЗ IN-SAMPLE УТЕЧКИ!) -> 112D")
    _log("=" * 88)

    # 1. Гарантируем целостность чистого RAW-бэкапа стакана при обоих сценариях (после build_600h_guard_dataset или при повторе):
    lob_dec_p = CACHE_DIR / "rl_guard_lob_cache_600h.npz"
    lob_raw_p = CACHE_DIR / "rl_guard_lob_cache_600h_raw_with_self.npz"
    cur_n = len(np.load(lob_dec_p, mmap_mode="r")["ts"])
    bak_n = len(np.load(lob_raw_p, mmap_mode="r")["ts"]) if lob_raw_p.exists() else -1
    if not lob_raw_p.exists() or cur_n > bak_n:
        shutil.copy2(lob_dec_p, lob_raw_p)
        _log(f"  📦 Обновлён эталонный RAW LOB бэкап ({cur_n:,} строк) -> {lob_raw_p.name}")
    else:
        shutil.copy2(lob_raw_p, lob_dec_p)
    (CACHE_DIR / "rl_hybrid_market_obs_84d_raw_with_self.npz").unlink(missing_ok=True)

    py = str(BASE_DIR / ".venv" / "bin" / "python")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE_DIR)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("L94_GPU_ID", "1")

    for script_nm in (
        "_tools/build_hybrid_market_obs_40d.py",
        "_tools/build_hybrid_market_obs_60d.py",
        "_tools/build_hybrid_market_obs_72d.py",
        "_tools/build_hybrid_market_obs_84d.py",
        "_tools/audit_and_extract_live_impact.py",
        "_tools/decontaminate_historical_lob.py",
    ):
        t_s = time.time()
        rc = subprocess.run([py, script_nm], cwd=str(BASE_DIR), env=env, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"Ошибка при выполнении {script_nm} (rc={rc})")
        _log(f"  ✅ Выполнен {script_nm} за {time.time()-t_s:.1f}с")

    # 2. Посуточный Walk-Forward Order-2.5 (ch[84..87]) БЕЗ in-sample утечки!
    int_data = np.load(CACHE_DIR / "order25_live_interventions.npz")
    X_place = int_data["X_place"]
    Y_place = int_data["Y_place"]
    X_fill = int_data["X_fill"]
    Y_fill = int_data["Y_fill"]

    near_m = X_place[:, 85] <= 0.60
    Xp_near = X_place[near_m]
    Yp_imp = np.clip(Y_place[near_m, 0] + Y_place[near_m, 1], 0.0, 1.0)
    p_rows = int_data["place_idx"][near_m]

    buy_m = X_fill[:, 84] < 0.5
    Xf_buy = X_fill[buy_m, :84]
    Yf_bt = (Y_fill[buy_m, 2] <= -35.0).astype(np.float32)
    b_rows = int_data["fill_idx"][buy_m]

    sell_m = X_fill[:, 84] > 0.5
    Xf_sell = X_fill[sell_m, :84]
    Yf_sr = (Y_fill[sell_m, 2] <= -35.0).astype(np.float32)
    s_rows = int_data["fill_idx"][sell_m]

    obs84_dec = np.load(CACHE_DIR / "rl_hybrid_market_obs_84d.npz", mmap_mode="r")["market_obs_84d"]
    total_rows = len(obs84_dec)
    epochs_o25 = build_24h_epochs(SUBCENT_START, total_rows)

    # Для каждой суточной эпохи E_k обучаем тройку бустеров строго на событиях row_idx < s_idx - 600:
    ep_models: list[tuple[int, int, xgb.Booster | None, xgb.Booster | None, xgb.Booster | None]] = []
    for ep_id, s_i, e_i in epochs_o25:
        cut_r = s_i - 600
        b_i = _fit_o25_causal_pair(Xp_near[p_rows < cut_r], Yp_imp[p_rows < cut_r], max_depth=4, reg_lam=8.0)
        b_t = _fit_o25_causal_pair(Xf_buy[b_rows < cut_r], Yf_bt[b_rows < cut_r], max_depth=3, reg_lam=10.0)
        b_r = _fit_o25_causal_pair(Xf_sell[s_rows < cut_r], Yf_sr[s_rows < cut_r], max_depth=3, reg_lam=10.0)
        ep_models.append((s_i, e_i, b_i, b_t, b_r))
    _log(f"  ✅ Обучено {len(ep_models)} строго каузальных суточных срезов Order-2.5 (0% in-sample наложения!)")

    # И отдельно сохраняем актуальный боевой комплект order25_live/*.ubj для VPS (на всей доступной истории < N - 200):
    O25_DIR.mkdir(parents=True, exist_ok=True)
    b_i_lv = _fit_o25_causal_pair(Xp_near, Yp_imp, max_depth=4, reg_lam=8.0)
    b_t_lv = _fit_o25_causal_pair(Xf_buy, Yf_bt, max_depth=3, reg_lam=10.0)
    b_r_lv = _fit_o25_causal_pair(Xf_sell, Yf_sr, max_depth=3, reg_lam=10.0)
    if b_i_lv is not None:
        b_i_lv.save_model(str(O25_DIR / "order25_bot_impact.ubj"))
    if b_t_lv is not None:
        b_t_lv.save_model(str(O25_DIR / "order25_buy_trap_3m.ubj"))
    if b_r_lv is not None:
        b_r_lv.save_model(str(O25_DIR / "order25_sell_regret_3m.ubj"))

    for tag, in_84_p, in_lob_p, out_88_p in (
        ("DEC", CACHE_DIR / "rl_hybrid_market_obs_84d.npz", CACHE_DIR / "rl_guard_lob_cache_600h.npz", CACHE_DIR / "rl_hybrid_market_obs_88d.npz"),
        ("RAW", CACHE_DIR / "rl_hybrid_market_obs_84d_raw_with_self.npz", CACHE_DIR / "rl_guard_lob_cache_600h_raw_with_self.npz", CACHE_DIR / "rl_hybrid_market_obs_88d_raw_with_self.npz"),
    ):
        o84 = np.load(in_84_p, mmap_mode="r")["market_obs_84d"]
        lob = np.load(in_lob_p, mmap_mode="r")
        bp10 = lob["bid_p"]
        bq10 = lob["bid_q"]
        ap10 = lob["ask_p"]

        obs88 = np.zeros((total_rows, 88), dtype=np.float32)
        obs88[:, :84] = o84
        # До первых реальных интервенций бота (ранний холодный старт) — нейтральные априоры:
        obs88[:, 84] = 0.35
        obs88[:, 85] = 0.50
        obs88[:, 86] = 0.23
        obs88[:, 87] = 0.35

        for s_i, e_i, bi, bt, br in ep_models:
            if bi is None and bt is None and br is None:
                continue
            m_c = o84[s_i:e_i]
            n_c = len(m_c)
            b1_c = np.maximum(0.0001, bp10[s_i:e_i, 0])
            a1_c = np.maximum(b1_c + 0.0001, ap10[s_i:e_i, 0])
            l1_usd = np.maximum(1.0, bq10[s_i:e_i, 0] * b1_c)
            t5_usd = np.maximum(1.0, np.sum(bq10[s_i:e_i, :5] * bp10[s_i:e_i, :5], axis=1))
            sp_n = ((a1_c - b1_c) / (0.5 * (a1_c + b1_c)) * 1e4) / 30.0
            zeros_c = np.zeros(n_c, dtype=np.float32)

            if bi is not None:
                c_sm = np.column_stack([zeros_c, zeros_c, np.full(n_c, np.log1p(40.0) / 6.0, dtype=np.float32), np.minimum(5.0, 40.0 / l1_usd), np.minimum(5.0, 40.0 / t5_usd), sp_n])
                c_wh = np.column_stack([zeros_c, zeros_c, np.full(n_c, np.log1p(250.0) / 6.0, dtype=np.float32), np.minimum(5.0, 250.0 / l1_usd), np.minimum(5.0, 250.0 / t5_usd), sp_n])
                obs88[s_i:e_i, 84] = np.clip(bi.predict(xgb.DMatrix(np.hstack([m_c, c_sm]))), 0.0, 1.0)
                obs88[s_i:e_i, 85] = np.clip(bi.predict(xgb.DMatrix(np.hstack([m_c, c_wh]))), 0.0, 1.0)
            dm_m = xgb.DMatrix(m_c)
            if bt is not None:
                obs88[s_i:e_i, 86] = np.clip(bt.predict(dm_m), 0.0, 1.0)
            if br is not None:
                obs88[s_i:e_i, 87] = np.clip(br.predict(dm_m), 0.0, 1.0)

        t88 = out_88_p.with_name(out_88_p.stem + "_tmp.npz")
        np.savez(t88, market_obs_88d=obs88)
        t88.replace(out_88_p)
        _log(f"  ✅ [{tag}] Запечатан честный посуточный Order-2.5 кэш {out_88_p.name}!")

    # 3. Финальная сборка 96D и 112D + очистка промежуточных 40d..88d:
    for script_fin in ("_tools/build_hybrid_market_obs_96d.py", "_tools/build_hybrid_market_obs_112d.py"):
        t_s = time.time()
        rc = subprocess.run([py, script_fin], cwd=str(BASE_DIR), env=env, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"Ошибка при выполнении {script_fin} (rc={rc})")
        _log(f"  ✅ Выполнен {script_fin} за {time.time()-t_s:.1f}с")

    for pat in ("rl_hybrid_market_obs_40d*.npz", "rl_hybrid_market_obs_60d*.npz", "rl_hybrid_market_obs_72d*.npz", "rl_hybrid_market_obs_84d*.npz", "rl_hybrid_market_obs_88d*.npz"):
        for f in CACHE_DIR.glob(pat):
            f.unlink(missing_ok=True)

    # Контрольная сумма запечатанного префикса 112D для детекции неизменности истории:
    obs112_chk = np.load(CACHE_DIR / "rl_hybrid_market_obs_112d.npz", mmap_mode="r")["market_obs_112d"]
    seal_n = int(manifest.get("o2_sealed_up_to", total_rows))
    sha = hashlib.sha256(np.ascontiguousarray(obs112_chk[SUBCENT_START:seal_n:500]).tobytes()).hexdigest()[:16]
    manifest["total_rows"] = int(total_rows)
    manifest["subcent_start"] = int(SUBCENT_START)
    manifest["sealed_112d_sha16"] = sha
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    _log(f"🔒 Манифест Write-Once Ledger сохранён: {MANIFEST_PATH.name} (rows={total_rows:,}, sha16={sha})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-rebuild", action="store_true", help="Полная пересборка всех суточных эпох с нуля (сброс старого leaky-кэша)")
    args = ap.parse_args()

    t0 = time.time()
    assert torch.cuda.is_available(), "CUDA недоступна!"
    assert torch.cuda.device_count() == 1, f"Ожидалась строго 1 изолированная GPU (GPU #1), видно: {torch.cuda.device_count()}"
    _log(f"🛡️ Аппаратный контроль пройден: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} -> {torch.cuda.get_device_name(0)}")

    # Синхронизируем длину RAW-бэкапа стакана ДО запуска Шага А.1 и А.2 (если только что прошёл build_600h_guard_dataset):
    lob_dec_p = CACHE_DIR / "rl_guard_lob_cache_600h.npz"
    lob_raw_p = CACHE_DIR / "rl_guard_lob_cache_600h_raw_with_self.npz"
    cur_n = len(np.load(lob_dec_p, mmap_mode="r")["ts"])
    bak_n = len(np.load(lob_raw_p, mmap_mode="r")["ts"]) if lob_raw_p.exists() else -1
    if not lob_raw_p.exists() or cur_n > bak_n:
        shutil.copy2(lob_dec_p, lob_raw_p)
        _log(f"📦 Синхронизирован свежий RAW LOB кэш ({cur_n:,} строк) -> {lob_raw_p.name}")

    manifest = json.loads(MANIFEST_PATH.read_text()) if (MANIFEST_PATH.exists() and not args.force_rebuild) else {}
    run_step1_order1_and_wave(manifest, force_rebuild=args.force_rebuild)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    run_step2_order2_on_sealed_o1(manifest, force_rebuild=args.force_rebuild)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    run_step3_assemble_112d_and_wf_o25(manifest)
    _log(f"🏁 ШАГ А ПОЛНОСТЬЮ ЗАВЕРШЁН ЗА {(time.time() - t0) / 60:.1f} МИНУТ!")


if __name__ == "__main__":
    main()
