#!/usr/bin/env python3
"""
_tools/bridge_l3_to_l4_upgrade.py
=============================================================================
Единый межстадийный мост (Между Уроком 3 и Уроком 4):
1. Синхронизация свежих данных (VPS/локальный горячий буфер) -> prl_1s_features.parquet -> rl_guard_lob_cache_600h.npz.
2. Разметка 6 горизонтов (H=5, 12, 25, 60, 90, 120s) и пересборка сетки окон в data/_mmap_cache.
3. Дообучение всех новых и сдвинувшихся окон LightGBM + 6 нейросетевых архитектур (conv1d_gru, causal_cnn,
   wavenet_tcn, cross_mixer, attention, mlp_bottleneck) и обновление альянсов.
4. Обучение 8-фазного Классификатора-Роутера (LearnedRegimeClassifierNet) и матриц компетентности окон.
5. Дообучение вспомогательных сетей (ExecutionRealismNet, CarouselNet v3, HazardNet v3).
6. Сборка полной 58D каузальной матрицы сигналов v7_signals_58d на всей глубине истории (681+ ч)
   и инъекция в data/cache/rl_guard_lob_cache_600h.npz перед стартом Гибридного Урока 4 (318D).
=============================================================================
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

BRIDGE_DONE_FLAG = BASE_DIR / "models" / "prl" / "lesson_4_inventory" / "bridge_v7_v8_ready.json"
LOG_FILE = BASE_DIR / "data" / "logs" / "bridge_l3_to_l4.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_bridge(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [BRIDGE v7+v8] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_bridge_upgrade(force: bool = False):
    if BRIDGE_DONE_FLAG.exists() and not force:
        log_bridge(f"✅ Межстадийный апгрейд v7+v8 уже выполнен ({BRIDGE_DONE_FLAG.name}).")
        return

    t_start = time.time()
    py = str(BASE_DIR / ".venv" / "bin" / "python")
    log_bridge("=" * 85)
    log_bridge("🌉 СТАРТ МЕЖСТАДИЙНОГО ОБНОВЛЕНИЯ ДАННЫХ, ОКОН v7, КЛАССИФИКАТОРА И СЛИЯНИЯ v7+v8")
    log_bridge("=" * 85)

    # Шаг 1. Синхронизация и пересборка 1s признаков и базового 600h кэша
    log_bridge("📥 Шаг 1/6: Обновление Parquet-таблиц, 1s-признаков и базового L2-кэша...")
    vps_updated = False
    try:
        from _tools.sync_all_data import sync_vps_hot, DATASET_CONFIGS, build_dataset_for_config
        try:
            res = sync_vps_hot()
            if res and res.get("downloaded", 0) > 0:
                vps_updated = True
        except Exception as e:
            log_bridge(f"ℹ️ VPS SFTP недоступен в текущем окружении ({e}), используем локальный буфер data/vps_hot.")
        if vps_updated:
            for cfg in DATASET_CONFIGS:
                if "quantus" not in cfg["key"]:
                    build_dataset_for_config(cfg, include_vps=True)
    except Exception as e:
        log_bridge(f"⚠️ Предупреждение на этапе склейки Parquet: {e}")

    feats_1s_path = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
    cache_600h_path = BASE_DIR / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
    proc_rl_path = BASE_DIR / "data" / "processed" / "prl_rl_telemetry.parquet"
    need_rebuild_1s = vps_updated or not feats_1s_path.exists() or not cache_600h_path.exists()
    if not need_rebuild_1s and proc_rl_path.exists():
        if proc_rl_path.stat().st_mtime > feats_1s_path.stat().st_mtime + 10:
            need_rebuild_1s = True

    if need_rebuild_1s:
        subprocess.run([py, "-u", "-m", "_tools.prepare_1s_dataset", "--coin", "prl"], cwd=str(BASE_DIR), check=True)
        subprocess.run([py, "-u", "-m", "_tools.build_600h_guard_dataset"], cwd=str(BASE_DIR), check=True)
    else:
        log_bridge("   ℹ️ 1s-признаки и базовый 600h-кэш уже синхронизированы с последним срезом данных.")

    # Шаг 2. Разметка 6 горизонтов и обновление сетки окон в data/_mmap_cache
    log_bridge("🏷️ Шаг 2/6: Разметка 6 горизонтов (H=5..120s) и нарезка новых окон Walk-Forward в data/_mmap_cache...")
    import pyarrow.parquet as pq
    feats_1s_path = BASE_DIR / "data" / "features" / "prl_1s_features.parquet"
    labeled_path = BASE_DIR / "data" / "labeled" / "prl" / "prl_labeled.parquet"
    need_relabel = True
    if labeled_path.exists() and feats_1s_path.exists():
        try:
            n_1s = pq.ParquetFile(feats_1s_path).metadata.num_rows
            n_lbl = pq.ParquetFile(labeled_path).metadata.num_rows
            if n_lbl == n_1s and labeled_path.stat().st_mtime >= feats_1s_path.stat().st_mtime - 60:
                log_bridge(f"   ℹ️ Размеченный датасет {labeled_path.name} ({n_lbl:,} строк) уже актуален, пропуск повторной разметки.")
                need_relabel = False
        except Exception:
            need_relabel = True

    if need_relabel:
        subprocess.run(
            [py, "-u", "-m", "_tools.data_labeler", "--coin", "prl", "--pairs", "5:3,12:3.5,25:3,60:3,90:3,120:3", "--no-legacy-splits"],
            cwd=str(BASE_DIR),
            check=True,
        )

    from _tools.train_rolling_swarm import find_clean_segments, generate_rolling_windows_reverse
    from _tools.v7_retrain_pipeline import HORIZON_PAIRS

    cache_dir = BASE_DIR / "data" / "_mmap_cache"
    x_all_file = cache_dir / "X_all.npy"
    h5_nw_file = cache_dir / "h5_n_windows.npy"
    need_mmap_rebuild = need_relabel or not x_all_file.exists() or not h5_nw_file.exists()
    if not need_mmap_rebuild:
        try:
            x_mm = np.load(x_all_file, mmap_mode="r")
            n_lbl = pq.ParquetFile(labeled_path).metadata.num_rows
            if x_mm.shape[0] == n_lbl and int(np.load(h5_nw_file)) >= 60:
                log_bridge(f"   ℹ️ Кэш data/_mmap_cache ({x_mm.shape[0]:,} строк, H=5s: {int(np.load(h5_nw_file))} окон) уже актуален.")
            else:
                need_mmap_rebuild = True
        except Exception:
            need_mmap_rebuild = True

    if need_mmap_rebuild:
        feats_json = BASE_DIR / "data" / "features" / "selected_features_prl_h8s.json"
        feature_cols = json.loads(feats_json.read_text(encoding="utf-8"))["feature_order"]
        need_cols = list(dict.fromkeys(
            feature_cols
            + ["timestamp", "sample_split", "is_data_gap", "ST_mid_price", "ST_trade_vol", "BO_sec_since_tick"]
            + [f"target_class_7_h{h}_k{k:g}" for h, k in HORIZON_PAIRS]
            + [f"target_dt_ticks_h{h}_k{k:g}" for h, k in HORIZON_PAIRS]
        ))
        avail_cols = [c for c in need_cols if c in pq.read_schema(labeled_path).names]
        df = pq.read_table(labeled_path, columns=avail_cols).to_pandas()
        segments = find_clean_segments(df)
        wins_map = {}
        for h, k in HORIZON_PAIRS:
            wins, *_ = generate_rolling_windows_reverse(segments, h, k)
            wins_map[f"h{h}"] = wins
            log_bridge(f"   Горизонт H={h:3d}s: доступно {len(wins):2d} валидных окон Walk-Forward")

        X_all = np.nan_to_num(df[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if "BO_sec_since_tick" in df.columns:
            bo_stale = df["BO_sec_since_tick"].fillna(0).values > 60.0
            zero_prefixes = ("LL_", "BO_ofi_", "BO_cvd_", "BO_imb_", "BO_obi_", "BO_micro_", "BO_ret_", "BO_vol_intensity_")
            zero_cols = [i_c for i_c, fn in enumerate(feature_cols) if fn.startswith(zero_prefixes)]
            if np.any(bo_stale) and zero_cols:
                X_all[np.ix_(bo_stale, zero_cols)] = 0.0

        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "X_all": X_all,
            "ts_all": df["timestamp"].astype(str).values,
            "y_all_h5": (df["target_class_7_h5_k3"].fillna(0).values + 3).astype(np.int8),
            "y_all_h12": (df["target_class_7_h12_k3.5"].fillna(0).values + 3).astype(np.int8),
            "y_all_h25": (df["target_class_7_h25_k3"].fillna(0).values + 3).astype(np.int8),
            "y_all_h60": (df["target_class_7_h60_k3"].fillna(0).values + 3).astype(np.int8),
            "y_all_h90": (df["target_class_7_h90_k3"].fillna(0).values + 3).astype(np.int8),
            "y_all_h120": (df["target_class_7_h120_k3"].fillna(0).values + 3).astype(np.int8),
            "dt_all_h5": df["target_dt_ticks_h5_k3"].fillna(0).values.astype(np.float16),
            "dt_all_h12": df["target_dt_ticks_h12_k3.5"].fillna(0).values.astype(np.float16),
            "dt_all_h25": df["target_dt_ticks_h25_k3"].fillna(0).values.astype(np.float16),
            "dt_all_h60": df["target_dt_ticks_h60_k3"].fillna(0).values.astype(np.float16),
            "dt_all_h90": df["target_dt_ticks_h90_k3"].fillna(0).values.astype(np.float16),
            "dt_all_h120": df["target_dt_ticks_h120_k3"].fillna(0).values.astype(np.float16),
            "features": np.array(feature_cols),
            "n_windows": np.int32(len(wins_map["h12"])),
        }
        for h_key, w_list in wins_map.items():
            arrays[f"{h_key}_n_windows"] = np.int32(len(w_list))
            for w_i, w_obj in enumerate(w_list):
                arrays[f"{h_key}_tr_idx_{w_i:02d}"] = w_obj["tr_idx"].astype(np.int32)
                arrays[f"{h_key}_te_idx_{w_i:02d}"] = w_obj["te_idx"].astype(np.int32)
        for k_name, v_arr in arrays.items():
            np.save(cache_dir / f"{k_name}.npy", v_arr)
        (cache_dir / "ready.txt").write_text("ok", encoding="utf-8")
        del df, X_all, arrays

    # Шаг 3. Обучение новых и сдвинувшихся окон LightGBM + 6 нейросетей (H=5..120s)
    log_bridge("🧬 Шаг 3/6: Дообучение новых окон LightGBM + 6 нейросетевых архитектур (H=5..120s)...")
    from _tools.v7_retrain_pipeline import step2_train_rolling_windows_6h, step3_update_rl_cache_and_aux_nets
    step2_train_rolling_windows_6h(coin="prl", epochs=8, runs=5)

    # Шаг 4. Обучение 8-фазного Классификатора-Роутера
    log_bridge("🧠 Шаг 4/6: Обучение 8-фазного Классификатора-Роутера на полном массиве...")
    from _tools.learned_regime_router import train_learned_regime_classifier
    train_learned_regime_classifier(device_str="cuda")

    # Шаг 5. Обновление вспомогательных сетей CarouselNet v3, HazardNet v3, ExecutionRealismNet
    log_bridge("🛡️ Шаг 5/6: Актуализация CarouselNet v3, HazardNet v3 и ExecutionRealismNet...")
    step3_update_rl_cache_and_aux_nets(coin="prl", tail_hours=176.0, epochs=8)

    # Шаг 6. Сборка полной 58D каузальной матрицы v7_signals_58d и инъекция в 600h кэш
    log_bridge("🔗 Шаг 6/6: Сборка полной 58D каузальной матрицы сигналов v7 и слияние с кэшем v8...")
    from _tools.build_hybrid_v7_v8_cache import build_and_inject_v7_signals_58d
    build_and_inject_v7_signals_58d(coin="prl", force_recompute=True)

    BRIDGE_DONE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    BRIDGE_DONE_FLAG.write_text(
        json.dumps({"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_sec": round(time.time() - t_start, 1)}, indent=2),
        encoding="utf-8",
    )
    log_bridge(f"🎉 МЕЖСТАДИЙНЫЙ АПГРЕЙД v7+v8 ПОЛНОСТЬЮ ЗАВЕРШЕН за {(time.time() - t_start)/60:.1f} мин! Готов к старту Урока 4 (318D).")


if __name__ == "__main__":
    run_bridge_upgrade(force="--force" in sys.argv)
