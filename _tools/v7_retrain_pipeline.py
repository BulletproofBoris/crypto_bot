#!/usr/bin/env python3
"""
Единый пайплайн полной актуализации робота PRL/USDT v7.0 на свежих данных:
  Шаг 1. step1_sync_and_rebuild_dataset()     — скачивание горячих CSV с VPS, пересборка 7 таблиц Parquet, 1s-признаков, разметки 6 горизонтов и mmap-кэша.
  Шаг 2. step2_train_rolling_windows_6h()     — инкрементальное дообучение новых окон LightGBM + 6 нейросетей (H=5, 12, 25, 60, 90, 120s) и сборка альянсов.
  Шаг 3. step3_update_rl_cache_and_aux_nets() — пересборка 6-горизонтного L2-кэша (176ч) и дообучение ExecutionRealismNet, CarouselNet v3 (47D), HazardNet v3 (47D).
  Шаг 4. step4_retrain_ppo_v70()              — дообучение нативного 10-слотового RL-агента MaskablePPO v7.0 (49D).
  Шаг 5. step5_run_validation_gate()          — слепой OOS-бенчмарк (v7.0 vs v6.1/v6.0/Baseline) + проверка латентности (< 350 мс).
  Шаг 6. step6_deploy_to_vps_and_verify()     — безопасная выкатка кода и весов на боевой VPS (81.19.137.18:2222) с сохранением кошелька и перезапуском демона.
"""

import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/restorator/crypto_bot")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

HORIZON_PAIRS = [(5, 3.0), (12, 3.5), (25, 3.0), (60, 3.0), (90, 3.0), (120, 3.0)]
NEURAL_ARCHS = ["mlp_bottleneck", "wavenet_tcn", "cross_mixer", "causal_cnn", "attention", "conv1d+gru"]


def run_cmd(cmd: list[str]):
    print(f"▶ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    if res.returncode != 0:
        raise RuntimeError(f"Команда завершилась с кодом {res.returncode}: {' '.join(cmd)}")


def step1_sync_and_rebuild_dataset(coin: str = "prl"):
    """Шаг 1: Подтягивание горячих данных и телеметрии с VPS, пересборка Parquet, 1s-признаков, 6 горизонтов разметки и mmap-кэша."""
    import pyarrow.parquet as pq
    from _tools.train_rolling_swarm import find_clean_segments, generate_rolling_windows_reverse

    py = str(BASE_DIR / ".venv" / "bin" / "python")
    print("=" * 95)
    print("📥 [ШАГ 1/6] СИНХРОНИЗАЦИЯ СВЕЖИХ ДАННЫХ С VPS, ПЕРЕСБОРКА PARQUET И РАЗМЕТКА 6 ГОРИЗОНТОВ")
    print("=" * 95)
    run_cmd([py, "-u", "-m", "_tools.sync_all_data"])
    run_cmd([py, "-u", "-m", "_tools.prepare_1s_dataset", "--coin", coin])
    run_cmd([py, "-u", "-m", "_tools.data_labeler", "--coin", coin, "--pairs", "5:3,12:3.5,25:3,60:3,90:3,120:3"])

    t0 = time.time()
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    feats_json = BASE_DIR / "data" / "features" / f"selected_features_{coin}_h8s.json"
    with open(feats_json) as f:
        feature_cols = json.load(f)["feature_order"]

    meta_cols = [
        "timestamp", "sample_split", "is_data_gap", "ST_mid_price", "ST_trade_vol", "BO_sec_since_tick",
        "target_class_7_h5_k3", "target_dt_ticks_h5_k3",
        "target_class_7_h12_k3.5", "target_dt_ticks_h12_k3.5",
        "target_class_7_h25_k3", "target_dt_ticks_h25_k3",
        "target_class_7_h60_k3", "target_dt_ticks_h60_k3",
        "target_class_7_h90_k3", "target_dt_ticks_h90_k3",
        "target_class_7_h120_k3", "target_dt_ticks_h120_k3",
    ]
    schema_cols = set(pq.read_schema(labeled_path).names)
    load_cols = list(dict.fromkeys([c for c in feature_cols + meta_cols if c in schema_cols]))
    df = pq.read_table(labeled_path, columns=load_cols).to_pandas()
    print(f"✅ Загружено {len(df):,} строк из {labeled_path.name}", flush=True)

    X_all = df[feature_cols].values.astype(np.float32).copy()
    if "BO_sec_since_tick" in df.columns:
        bo_stale = (df["BO_sec_since_tick"].fillna(0).values > 60.0)
        if np.any(bo_stale):
            zero_prefixes = ("LL_", "BO_ofi_", "BO_cvd_", "BO_imb_", "BO_obi_", "BO_micro_", "BO_ret_", "BO_vol_intensity_")
            zero_cols = [i_c for i_c, fn in enumerate(feature_cols) if fn.startswith(zero_prefixes)]
            if zero_cols:
                X_all[np.ix_(bo_stale, zero_cols)] = 0.0

    segs = find_clean_segments(df)
    wins_map = {}
    for h, k in HORIZON_PAIRS:
        w_list, _, _, _, _ = generate_rolling_windows_reverse(segs, H=h, K=k)
        wins_map[f"h{h}"] = w_list
    print(
        "⏱️ Сетка скользящих окон: "
        + ", ".join(f"H={h}s ({len(wins_map[f'h{h}'])})" for h, _ in HORIZON_PAIRS),
        flush=True,
    )

    cache_dir = BASE_DIR / "data" / "_mmap_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old_f in cache_dir.glob("*_idx_*.npy"):
        old_f.unlink(missing_ok=True)

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
    print(f"🎉 Бинарный кэш data/_mmap_cache обновлён за {time.time() - t0:.1f}с!", flush=True)


def step2_train_rolling_windows_6h(coin: str = "prl", epochs: int = 10, runs: int = 15, cache_subdir: str | None = None):
    """Шаг 2: Инкрементальное дообучение скользящих окон LightGBM + 6 нейросетей (H=5..120s, runs=15) и обновление альянсов."""
    from _tools.train_rolling_swarm import train_coin_horizon
    from _tools.fast_neural_engine import train_pytorch_window_fast
    from _tools.system_resources import SAFE_WORKERS

    coin_l = coin.lower()
    if cache_subdir is None:
        cache_subdir = "_mmap_cache_bo" if coin_l.endswith("_bo") else "_mmap_cache"

    print("=" * 95)
    print(f"🧬 [ШАГ 2/6] ДООБУЧЕНИЕ СКОЛЬЗЯЩИХ ОКОН РОЯ {coin_l.upper()} (LIGHTGBM + 6 НЕЙРОСЕТЕЙ, H=5..120s, runs={runs})")
    print("=" * 95)
    cache_dir = BASE_DIR / "data" / cache_subdir
    ts_all_arr = np.load(cache_dir / "ts_all.npy", allow_pickle=True)

    for h, k in HORIZON_PAIRS:
        n_w = int(np.load(cache_dir / f"h{h}_n_windows.npy"))
        lgb_dir = BASE_DIR / "models" / coin_l / f"h{h}s_k{k:g}"
        need_lgb = False
        for w_chk in range(n_w):
            mp_chk = lgb_dir / f"meta_win_{w_chk:02d}.json"
            if not mp_chk.exists():
                need_lgb = True
                break
            if w_chk >= max(0, n_w - 2):
                tr_idx_chk = np.load(cache_dir / f"h{h}_tr_idx_{w_chk:02d}.npy")
                exp_tr_range = [str(ts_all_arr[tr_idx_chk[0]])[:19], str(ts_all_arr[tr_idx_chk[-1]])[:19]]
                if json.loads(mp_chk.read_text()).get("train_ts_range") != exp_tr_range:
                    need_lgb = True
                    break
        if need_lgb:
            print(f"\n🔹 [LightGBM {coin_l.upper()} H={h}s, K={k}] Проверка и дообучение скользящих окон...")
            train_coin_horizon(coin=coin_l, H=h, K=k, n_estimators=450, n_jobs=SAFE_WORKERS)
        else:
            print(f"\n🔹 [LightGBM {coin_l.upper()} H={h}s, K={k}] Все {n_w} окон актуальны по временным меткам.")
        for arch in NEURAL_ARCHS:
            ep = (6 if h >= 90 else 8) if arch == "conv1d+gru" else (8 if h >= 90 else epochs)
            bsz = 4096 if h >= 60 else 2048
            arch_dir = BASE_DIR / "models" / coin_l / f"{arch.replace('+', '_')}_h{h}s"
            for w_i in range(n_w):
                mp = arch_dir / f"meta_win_{w_i:02d}.json"
                tr_idx_w = np.load(cache_dir / f"h{h}_tr_idx_{w_i:02d}.npy")
                exp_tr_range = [str(ts_all_arr[tr_idx_w[0]])[:19], str(ts_all_arr[tr_idx_w[-1]])[:19]]
                if mp.exists():
                    try:
                        open(mp, "r+").close()
                    except OSError:
                        continue
                    meta_obj = json.loads(mp.read_text())
                    age_sec = time.time() - mp.stat().st_mtime
                    cur_runs = int(meta_obj.get("runs", 3))
                    ts_mismatch = (w_i >= max(0, n_w - 2)) and (meta_obj.get("train_ts_range") != exp_tr_range)
                    if not ts_mismatch and not ((age_sec < 43200 or w_i == n_w - 1) and cur_runs < runs):
                        continue
                print(f"  🧠 [{coin_l.upper()} {arch.upper()} H={h}s] Обучение окна win_{w_i:02d}/{n_w-1} (runs={runs}, epochs={ep})...", flush=True)
                train_pytorch_window_fast(
                    cache_dir=str(cache_dir),
                    horizon=h,
                    win_idx=w_i,
                    arch=arch,
                    runs=runs,
                    epochs=ep,
                    batch_size=bsz,
                    lr=2.2e-3,
                    force=True,
                    coin=coin_l,
                )
            manifest = []
            for w_i in range(n_w):
                mp = arch_dir / f"meta_win_{w_i:02d}.json"
                if mp.exists():
                    manifest.append(json.loads(mp.read_text()))
            if manifest:
                try:
                    (arch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                except OSError:
                    pass

        # Пересборка мета-ансамбля ensemble_alliance_h{h}s.json
        al_path = BASE_DIR / "models" / coin_l / f"ensemble_alliance_h{h}s.json"
        al_live_path = BASE_DIR / "models" / coin_l / f"ensemble_alliance_live_h{h}s.json"
        al_path.parent.mkdir(parents=True, exist_ok=True)
        selected_models = []
        weights = {}
        for a2 in NEURAL_ARCHS:
            a2_dir = BASE_DIR / "models" / coin_l / f"{a2.replace('+', '_')}_h{h}s"
            for w_i in range(n_w):
                mp = a2_dir / f"meta_win_{w_i:02d}.json"
                if mp.exists():
                    mo = json.loads(mp.read_text()).get("metrics_oos", {})
                    ic = float(mo.get("ic_dt_ticks", 0.0))
                    if ic > 0.02:
                        key = f"{a2}_win_{w_i:02d}"
                        selected_models.append(key)
                        weights[key] = round(max(0.001, ic ** 2), 6)
        w_sum = sum(weights.values()) + 1e-12
        weights = {k_m: round(v_m / w_sum, 6) for k_m, v_m in weights.items()}
        payload = {
            "coin": coin.upper(),
            "horizon_sec": h,
            "n_windows": n_w,
            "selected_models": selected_models,
            "weights": weights,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            al_live_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
        try:
            al_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            Path(f"/tmp/{al_path.name}").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  🤝 Альянс {al_path.name} обновлён: {len(selected_models)} активных экспертов ({n_w} окон)")
    print("\n✅ Все 6 горизонтов (LightGBM + 6 нейросетевых архитектур + альянсы) синхронизированы!")


def step3_update_rl_cache_and_aux_nets(coin: str = "prl", tail_hours: float = 176.0, epochs: int = 10):
    """Шаг 3: Пересборка 6-горизонтного L2 кэша (176ч) и дообучение ExecutionRealismNet, CarouselNet v3 (47D), HazardNet v3 (47D)."""
    from _tools.train_rl_execution import build_or_load_rl_cache, _compute_v52_signals_matrix, _compute_v70_realism_matrix
    from _tools.execution_realism_net import train_execution_realism_net
    from _tools.carousel_net import train_carousel_net
    from _tools.hazard_net import train_hazard_net

    print("=" * 95)
    print("🧠 [ШАГ 3/6] ОБНОВЛЕНИЕ 6-ГОРИЗОНТНОГО L2-КЭША И ДООБУЧЕНИЕ REALISM / CAROUSEL v3 / HAZARD v3")
    print("=" * 95)
    for tmp_al in Path("/tmp").glob("ensemble_alliance_h*.json"):
        dst_al = BASE_DIR / "models" / coin / tmp_al.name
        try:
            dst_al.write_text(tmp_al.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass
    data_dict = build_or_load_rl_cache(coin=coin, tail_hours=tail_hours, force_rebuild=True)
    cache_path = str(BASE_DIR / "data" / "cache" / f"rl_triad_lob_cache_{coin}_{int(tail_hours)}h.npz")

    train_execution_realism_net(cache_path=cache_path, epochs=epochs)
    train_carousel_net(cache_path=cache_path, epochs=epochs)
    train_hazard_net(cache_path=cache_path, epochs=epochs)

    data_dict["X_v52"] = _compute_v52_signals_matrix(data_dict, coin=coin)
    data_dict["X_v70_realism"] = _compute_v70_realism_matrix(data_dict, coin=coin)
    np.savez_compressed(cache_path, **data_dict)
    print(f"✅ Кэш {cache_path} полностью обновлен (X_v52 + X_v70_realism)!")


def step4_retrain_ppo_v70(coin: str = "prl", tail_hours: float = 176.0, ppo_epochs: int = 12, steps_per_epoch: int = 32768):
    """Шаг 4: Дообучение нативного 10-слотового RL-агента MaskablePPO v7.0 (49D)."""
    from _tools.train_rl_execution import train_rl_agent

    print("=" * 95)
    print("🤖 [ШАГ 4/6] ДООБУЧЕНИЕ НАТИВНОГО 10-СЛОТОВОГО RL-АГЕНТА MASKABLE PPO v7.0 (49D)")
    print("=" * 95)
    return train_rl_agent(
        coin=coin,
        tail_hours=tail_hours,
        ppo_epochs=ppo_epochs,
        steps_per_epoch=steps_per_epoch,
        pop_size=8,
        force_rebuild=False,
    )


def step5_run_validation_gate(
    coin: str = "prl",
    replay_steps: int = 10,
    scaling_budgets: tuple = (8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0),
):
    """Шаг 5: Проверка латентности боевого шага (< 350 мс), слепой OOS-экзамен и стресс-тест масштабирования капитала ($8k..$20k с шагом $2k)."""
    from _tools.live_rl_trader import run_replay_test
    from _tools.train_rl_execution import inspect_and_plot_rl_oos
    from _tools.multi_slot_benchmark import run_capital_scaling_grid

    print("=" * 95)
    print("🛡️ [ШАГ 5/6] ВАЛИДАЦИЯ ЛАТЕНТНОСТИ (< 350 МС), СЛЕПОЙ OOS И ТЕСТ МАСШТАБИРОВАНИЯ КАПИТАЛА ($8k..$20k)")
    print("=" * 95)
    run_replay_test(coin=coin, steps=replay_steps, use_neural=True)
    df_ppo, df_base = inspect_and_plot_rl_oos(coin=coin, tail_hours=176.0)
    df_scale = run_capital_scaling_grid(coin=coin, budgets=scaling_budgets)
    return df_ppo, df_base, df_scale


def step6_deploy_to_vps_and_verify(coin: str = "prl"):
    """Шаг 6: Безопасная выкатка обновленного кода и весов моделей на VPS (81.19.137.18:2222) с сохранением кошелька."""
    from _tools.vps_config import create_ssh_client
    from _tools.live_rl_trader import show_live_status

    print("=" * 95)
    print("🚀 [ШАГ 6/6] ДЕПЛОЙ ОБНОВЛЕННЫХ ВЕСОВ v7.0 НА БОЕВОЙ VPS И ПЕРЕЗАПУСК ДЕМОНА")
    print("=" * 95)
    bundle_path = Path("/tmp/v70_deploy_bundle.tar.gz")
    files_to_pack = [
        "_tools/live_rl_trader.py",
        "_tools/execution_realism_net.py",
        "_tools/carousel_net.py",
        "_tools/hazard_net.py",
        "_tools/train_rl_execution.py",
        "_tools/swarm_router.py",
        "_tools/fast_neural_engine.py",
        "_tools/multi_slot_benchmark.py",
        "_tools/v7_retrain_pipeline.py",
        f"models/{coin}/rl_execution/best_maskable_ppo.pt",
        f"models/{coin}/rl_execution/carousel_net.pt",
        f"models/{coin}/rl_execution/hazard_net.pt",
        f"models/{coin}/rl_execution/execution_realism_net.pt",
    ]
    for f in (BASE_DIR / "models" / coin).glob("ensemble_alliance_h*.json"):
        files_to_pack.append(str(f.relative_to(BASE_DIR)))
    for d in (BASE_DIR / "models" / coin).glob("h*s_k*"):
        if d.is_dir():
            for f in sorted(d.glob("*")):
                if f.is_file():
                    files_to_pack.append(str(f.relative_to(BASE_DIR)))
    for arch_d in (BASE_DIR / "models" / coin).glob("*_h*s"):
        if arch_d.is_dir():
            for f in sorted(arch_d.glob("*")):
                if f.is_file():
                    files_to_pack.append(str(f.relative_to(BASE_DIR)))

    with tarfile.open(bundle_path, "w:gz") as tar:
        for rel in files_to_pack:
            full_p = BASE_DIR / rel
            if full_p.exists():
                tar.add(full_p, arcname=rel)
    print(f"📦 Упаковано {len(files_to_pack)} файлов ({bundle_path.stat().st_size / (1024*1024):.2f} МБ)")

    ssh = create_ssh_client(timeout=20.0)
    _, out_w, _ = ssh.exec_command(f"cat /opt/crypto_bot/models/{coin}/rl_execution/live_wallet_state.json 2>/dev/null || echo '{{}}'")
    wallet_dict = json.loads(out_w.read().decode("utf-8", errors="ignore").strip() or "{}")
    budget_usd = float(wallet_dict.get("initial_budget_usd", 305.99))
    print(f"💼 Сохранено состояние кошелька VPS: budget=${budget_usd:.2f}, cash=${wallet_dict.get('bot_cash_usd', 0):.2f}, pos={wallet_dict.get('pos_qty', 0):.4f} PRL")

    sftp = ssh.open_sftp()
    sftp.put(str(bundle_path), "/tmp/v70_deploy_bundle.tar.gz")
    sftp.close()
    _, out_ex, _ = ssh.exec_command("cd /opt/crypto_bot && tar -xzf /tmp/v70_deploy_bundle.tar.gz && rm -f /tmp/v70_deploy_bundle.tar.gz && echo 'UNPACK_OK'")
    print("📤 Статус распаковки на VPS:", out_ex.read().decode("utf-8", errors="ignore").strip())

    restart_cmd = (
        "OLD_PID=$(cat /opt/crypto_bot/data/logs/live_rl_trader.pid 2>/dev/null); "
        "if [ -n \"$OLD_PID\" ] && kill -0 \"$OLD_PID\" 2>/dev/null; then kill -TERM \"$OLD_PID\" 2>/dev/null; sleep 1; kill -9 \"$OLD_PID\" 2>/dev/null || true; fi; "
        "pkill -9 -f '[_]tools/live_rl_trader.py' 2>/dev/null || true; sleep 1; "
        "PY_BIN=/opt/crypto_bot/venv/bin/python; [ -x /opt/crypto_bot/.venv/bin/python ] && PY_BIN=/opt/crypto_bot/.venv/bin/python; "
        f"cd /opt/crypto_bot && nohup $PY_BIN -u _tools/live_rl_trader.py --coin {coin} --mode live --steps 864000 --order-usd {budget_usd:.2f} --budget-usd {budget_usd:.2f} > /opt/crypto_bot/data/logs/live_rl_trader.log 2>&1 & "
        "NEW_PID=$!; echo $NEW_PID > /opt/crypto_bot/data/logs/live_rl_trader.pid; echo \"STARTED_PID:$NEW_PID\""
    )
    _, out_r, _ = ssh.exec_command(restart_cmd)
    print("🚀 Перезапуск демона:", out_r.read().decode("utf-8", errors="ignore").strip())
    ssh.close()

    time.sleep(15)
    show_live_status(tail_lines=25)
