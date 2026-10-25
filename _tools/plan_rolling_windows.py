#!/usr/bin/env python3
"""
Калькулятор и планировщик новых окон Walk-Forward (Учитель LightGBM + Ученики Нейросети).

Определяет по актуальному датасету (prl_labeled.parquet / prl_1s_features.parquet):
1. Объем чистых данных (n_clean после Seam-Purged Stitching без гэпов и заморозок).
2. Точные позиции (tr_end_pos в clean_idx и временные границы) уже обученных окон win_00..win_N.
3. Сколько новых секунд/часов накопилось после последнего обученного окна win_00.
4. Возможные диапазоны для НОВЫХ окон (как с полным шагом Step = T_train // 2, так и скользящих с минимальным OOS-хвостом),
   а также сколько еще данных нужно собрать до следующего шага по каждому горизонту H=5, 12, 25с.
5. Готовый словарь fixed_tr_ends_map для train_rolling_swarm.py и команду сдвига старых окон (win_00 -> win_01...),
   чтобы дообучать только новые окна спереди без переобучения старых.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.train_rolling_swarm import find_clean_segments, get_default_k


def build_clean_index(df: pd.DataFrame, seam_purge: int = 360) -> np.ndarray:
    segments = find_clean_segments(df)
    chron_segs = sorted(segments, key=lambda x: x[0])
    clean_blocks = []
    for s_idx, (s, e) in enumerate(chron_segs):
        s_clean = s + (seam_purge if s_idx > 0 else 0)
        e_clean = e - (seam_purge if s_idx < len(chron_segs) - 1 else 0)
        if e_clean - s_clean > 1000:
            clean_blocks.append(np.arange(s_clean, e_clean, dtype=np.int64))
    if not clean_blocks:
        return np.array([], dtype=np.int64)
    return np.concatenate(clean_blocks)


def analyze_horizon_windows(
    df: pd.DataFrame,
    clean_idx: np.ndarray,
    ts_clean: pd.Series,
    coin: str,
    H: int,
    K: float,
    n_target: int = 5000,
    min_test_sec: int = 21600,
) -> dict:
    L = int(round(H * K))
    t_train = int(n_target * L)
    step = t_train // 2
    buffer = H + 360
    n_clean = len(clean_idx)
    k_str = f"{K:g}"

    manifest_path = BASE_DIR / "models" / coin / f"h{H}s_k{k_str}" / "manifest.json"
    existing_wins = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_wins = json.load(f)

    # Сопоставляем уже обученные окна с позициями в текущем clean_idx
    mapped_existing = []
    for w in existing_wins:
        tr_s_str, tr_e_str = w["train_ts_range"]
        te_s_str, te_e_str = w["test_ts_range"]
        tr_s_ts = pd.to_datetime(tr_s_str, utc=True)
        tr_e_ts = pd.to_datetime(tr_e_str, utc=True)
        tr_s_pos = int(ts_clean.searchsorted(tr_s_ts))
        tr_e_pos = int(ts_clean.searchsorted(tr_e_ts, side="right"))
        mapped_existing.append({
            "window_id": w["window_id"],
            "tr_s_pos": tr_s_pos,
            "tr_e_pos": tr_e_pos,
            "tr_ts": (tr_s_str, tr_e_str),
            "te_ts": (te_s_str, te_e_str),
            "n_train": w.get("n_train", tr_e_pos - tr_s_pos),
            "n_test": w.get("n_test", 0),
            "ic": w.get("metrics_oos", {}).get("ic_dt_ticks", 0.0),
        })

    # Проверяем, какие окна уже обучены у нейросетей (Учеников)
    neural_trained = {}
    for arch_dir in ["conv1d_gru", "causal_cnn", "mlp_bottleneck", "attention"]:
        ndir = BASE_DIR / "models" / coin / f"{arch_dir}_h{H}s"
        if ndir.exists():
            wins = sorted([p.stem.replace("model_", "") for p in ndir.glob("model_win_*.keras")])
            neural_trained[arch_dir] = wins

    latest_tr_e_pos = mapped_existing[0]["tr_e_pos"] if mapped_existing else (n_clean - step - buffer)
    avail_after_latest_tr = n_clean - latest_tr_e_pos

    # 1. Новые стандартные окна с шагом step = T_train // 2 вперед от последнего win_00
    # Для каждого нового шага k >= 1: tr_e_new = latest_tr_e_pos + k * step
    # Чтобы окно имело валидный OOS-тест >= min_test_sec, нужно: tr_e_new + buffer + min_test_sec <= n_clean
    new_step_windows = []
    k = 1
    while True:
        cand_tr_e = latest_tr_e_pos + k * step
        max_te_len = n_clean - (cand_tr_e + buffer)
        if max_te_len < min_test_sec:
            break
        cand_tr_s = cand_tr_e - t_train
        cand_te_s = cand_tr_e + buffer
        cand_te_e = min(n_clean, cand_te_s + step)
        new_step_windows.append({
            "type": "STANDARD_STEP",
            "step_num": k,
            "tr_s_pos": cand_tr_s,
            "tr_e_pos": cand_tr_e,
            "te_s_pos": cand_te_s,
            "te_e_pos": cand_te_e,
            "tr_ts": (str(df["timestamp"].iloc[clean_idx[cand_tr_s]])[:19], str(df["timestamp"].iloc[clean_idx[cand_tr_e - 1]])[:19]),
            "te_ts": (str(df["timestamp"].iloc[clean_idx[cand_te_s]])[:19], str(df["timestamp"].iloc[clean_idx[cand_te_e - 1]])[:19]),
            "n_train": t_train,
            "n_test": cand_te_e - cand_te_s,
            "pct_full_test": round(100.0 * (cand_te_e - cand_te_s) / step, 1),
        })
        k += 1

    # 2. Максимально свежее "плавающее" (Edge) окно прямо от текущего края данных (если стандартный шаг еще не накопился)
    edge_tr_e = n_clean - buffer - min(step, max(min_test_sec, avail_after_latest_tr - buffer))
    edge_window = None
    if edge_tr_e > latest_tr_e_pos + 3600 and edge_tr_e - t_train >= 0:
        edge_tr_s = edge_tr_e - t_train
        edge_te_s = edge_tr_e + buffer
        edge_te_e = n_clean
        edge_window = {
            "type": "EDGE_FRESH",
            "shift_from_win00_sec": edge_tr_e - latest_tr_e_pos,
            "tr_s_pos": edge_tr_s,
            "tr_e_pos": edge_tr_e,
            "te_s_pos": edge_te_s,
            "te_e_pos": edge_te_e,
            "tr_ts": (str(df["timestamp"].iloc[clean_idx[edge_tr_s]])[:19], str(df["timestamp"].iloc[clean_idx[edge_tr_e - 1]])[:19]),
            "te_ts": (str(df["timestamp"].iloc[clean_idx[edge_te_s]])[:19], str(df["timestamp"].iloc[clean_idx[edge_te_e - 1]])[:19]),
            "n_train": t_train,
            "n_test": edge_te_e - edge_te_s,
        }

    # Сколько секунд не хватает до следующего полного шага (100% Step + Buffer + 100% OOS Test = 2*step + buffer от latest_tr_e_pos)
    next_k = len(new_step_windows) + 1
    next_step_tr_e = latest_tr_e_pos + next_k * step
    need_for_min_oos = max(0, (next_step_tr_e + buffer + min_test_sec) - n_clean)
    need_for_full_oos = max(0, (next_step_tr_e + buffer + step) - n_clean)

    new_tr_ends = [w["tr_e_pos"] for w in reversed(new_step_windows)] + [w["tr_e_pos"] for w in mapped_existing]

    return {
        "H": H,
        "K": K,
        "L": L,
        "t_train": t_train,
        "step": step,
        "buffer": buffer,
        "n_clean": n_clean,
        "latest_tr_e_pos": latest_tr_e_pos,
        "avail_after_latest_tr": avail_after_latest_tr,
        "mapped_existing": mapped_existing,
        "neural_trained": neural_trained,
        "new_step_windows": new_step_windows,
        "edge_window": edge_window,
        "need_for_min_oos_sec": need_for_min_oos,
        "need_for_full_oos_sec": need_for_full_oos,
        "recommended_tr_ends": new_tr_ends,
    }


def print_window_plan(coin: str = "prl", horizons: tuple = (5, 12, 25), min_test_sec: int = 21600):
    coin = coin.lower()
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    features_path = BASE_DIR / "data" / "features" / f"{coin}_1s_features.parquet"
    raw_depth_path = BASE_DIR / "data" / "processed" / "safetrade_depth.parquet" if coin == "prl" else BASE_DIR / "data" / "processed" / f"safetrade_{coin}_depth.parquet"

    candidates = [p for p in (labeled_path, features_path) if p.exists()]
    if not candidates:
        print(f"❌ Датасет не найден: {labeled_path}")
        return {}

    # Выбираем самый свежий по числу строк/времени модификации датасет 1с
    active_path = max(candidates, key=lambda p: pq.read_metadata(p).num_rows)
    cols = ["timestamp", "sample_split", "is_data_gap", "ST_mid_price", "ST_trade_vol"]
    schema_cols = set(pq.read_schema(active_path).names)
    df = pq.read_table(active_path, columns=[c for c in cols if c in schema_cols]).to_pandas()
    if "timestamp" not in df.columns:
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})

    # Если в raw_depth_path уже синхронизированы более свежие данные (после sync_all_data, но до prepare_1s_dataset),
    # достраиваем виртуальную 1-секундную сетку до конца raw_depth_path, чтобы показать реальный потенциал новых окон!
    raw_extra_sec = 0
    if raw_depth_path.exists():
        try:
            raw_ts = pq.read_table(raw_depth_path, columns=["timestamp"]).to_pandas()["timestamp"]
            raw_max_ts = pd.to_datetime(raw_ts.max(), utc=True)
            df_max_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
            if raw_max_ts > df_max_ts + pd.Timedelta(seconds=60):
                raw_extra_sec = int((raw_max_ts - df_max_ts).total_seconds())
                extra_idx = pd.date_range(df_max_ts + pd.Timedelta(seconds=1), raw_max_ts, freq="1s", tz="UTC")
                extra_df = pd.DataFrame({
                    "timestamp": extra_idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                    "is_data_gap": 0,
                    "ST_mid_price": np.linspace(float(df["ST_mid_price"].iloc[-1]), float(df["ST_mid_price"].iloc[-1]) + 0.01, len(extra_idx)),
                    "ST_trade_vol": 1.0,
                })
                df = pd.concat([df, extra_df], ignore_index=True)
        except Exception:
            pass

    clean_idx = build_clean_index(df)
    n_clean = len(clean_idx)
    ts_clean = pd.to_datetime(df["timestamp"].iloc[clean_idx], utc=True).reset_index(drop=True)

    ts_first = str(df["timestamp"].iloc[clean_idx[0]])[:19]
    ts_last = str(df["timestamp"].iloc[clean_idx[-1]])[:19]

    print("=" * 105)
    print(f"📐 КАЛЬКУЛЯТОР ДИАПАЗОНОВ ОКОН (УЧИТЕЛЬ LightGBM + УЧЕНИКИ Нейросети) | {coin.upper()}")
    print("=" * 105)
    print(f"📂 Источник 1с-сетки      : {active_path.relative_to(BASE_DIR)}" + (f" (+{raw_extra_sec:,}с из {raw_depth_path.name}, требуется пересборка признаков/меток)" if raw_extra_sec > 0 else ""))
    print(f"📊 Всего строк в таймлайне: {len(df):,} сек")
    print(f"🧼 Чистый ряд (n_clean)   : {n_clean:,} сек ({n_clean/3600:.1f} ч / {n_clean/86400:.2f} сут)")
    print(f"⏱️ Диапазон чистых данных : {ts_first} UTC ➔ {ts_last} UTC")
    print("=" * 105)

    results = {}
    for H in horizons:
        K = get_default_k(coin, H)
        res = analyze_horizon_windows(df, clean_idx, ts_clean, coin, H, K, min_test_sec=min_test_sec)
        results[H] = res
        existing = res["mapped_existing"]
        n_ex = len(existing)
        print(f"\n🔹 ГОРИЗОНТ H = {H}с (K = {K:g}, L = {res['L']}с)")
        print(
            f"   Параметры сетки: Train = {res['t_train']:,}с ({res['t_train']/3600:.1f}ч) | "
            f"Шаг (Step = Train/2) = {res['step']:,}с ({res['step']/3600:.1f}ч) | Buffer = {res['buffer']}с"
        )
        if existing:
            w0 = existing[0]
            w_last = existing[-1]
            print(
                f"   Обучено у Учителя (LightGBM): {n_ex} окон (win_00..win_{n_ex-1:02d}) | "
                f"Свежее win_00: Train [{w0['tr_ts'][0]} ➔ {w0['tr_ts'][1]}] (tr_e_pos={w0['tr_e_pos']:,}), "
                f"Test [{w0['te_ts'][0]} ➔ {w0['te_ts'][1]}] ({w0['n_test']:,}с)"
            )
            print(
                f"   Накоплено чистых данных после Train win_00: +{res['avail_after_latest_tr']:,}с "
                f"({res['avail_after_latest_tr']/3600:.1f}ч)"
            )
        else:
            print("   Обученных окон в манифесте пока нет.")

        # Сравнение покрытия Учитель vs Ученики (какие окна уже обучены у LightGBM, но еще не обучены у нейросетей)
        teacher_win_ids = [w["window_id"] for w in existing]
        missing_students = {}
        for arch, s_wins in res["neural_trained"].items():
            s_set = set(s_wins)
            miss = [wid for wid in teacher_win_ids if wid not in s_set]
            if miss:
                missing_students[arch] = miss
        if missing_students:
            print("   🎓 Недообученные окна Учеников (уже есть у Учителя LightGBM, можно дообучить в Colab прямо сейчас):")
            for arch, miss in missing_students.items():
                print(f"      - {arch:<15}: не хватает {len(miss)} окон -> {', '.join(miss[:8])}{'...' if len(miss) > 8 else ''}")
        else:
            print("   🎓 Все текущие окна Учителя полностью закрыты моделями-Учениками!")

        # Новые окна спереди (по стандартному шагу Step = Train // 2)
        if res["new_step_windows"]:
            print(f"   🟢 ДОСТУПНЫ НОВЫЕ СТАНДАРТНЫЕ ОКНА СПЕРЕДИ ({len(res['new_step_windows'])} шт. с шагом {res['step']:,}с):")
            for nw in res["new_step_windows"]:
                print(
                    f"      + Шаг #{nw['step_num']}: tr_e_pos={nw['tr_e_pos']:,} | "
                    f"Train [{nw['tr_ts'][0]} ➔ {nw['tr_ts'][1]}] ({nw['n_train']:,}с) | "
                    f"Test [{nw['te_ts'][0]} ➔ {nw['te_ts'][1]}] ({nw['n_test']:,}с, {nw['pct_full_test']}% от шага)"
                )
        else:
            print(
                f"   ⏳ До следующего стандартного окна (шаг +{res['step']:,}с + мин. тест {min_test_sec:,}с) "
                f"осталось накопить: {res['need_for_min_oos_sec']:,}с ({res['need_for_min_oos_sec']/3600:.1f}ч) | "
                f"до 100% полного теста ({res['step']:,}с): {res['need_for_full_oos_sec']:,}с ({res['need_for_full_oos_sec']/3600:.1f}ч)"
            )

        if res["edge_window"] and not res["new_step_windows"]:
            ew = res["edge_window"]
            print(
                f"   💡 Возможен внеочередной (Edge) сдвиг на +{ew['shift_from_win00_sec']:,}с ({ew['shift_from_win00_sec']/3600:.1f}ч): "
                f"tr_e_pos={ew['tr_e_pos']:,} | Train [{ew['tr_ts'][0]} ➔ {ew['tr_ts'][1]}] | "
                f"Test [{ew['te_ts'][0]} ➔ {ew['te_ts'][1]}] ({ew['n_test']:,}с)"
            )

    print("\n" + "=" * 105)
    print("📋 РЕКОМЕНДУЕМЫЙ FIXED_TR_ENDS_MAP ДЛЯ _tools/train_rolling_swarm.py:")
    print("-" * 105)
    print("    fixed_tr_ends_map = {")
    for H in horizons:
        r_list = results[H]["recommended_tr_ends"]
        print(f"        {H}: {r_list},")
    print("    }")
    print("=" * 105)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Калькулятор диапазонов новых окон Walk-Forward (Учитель + Ученик)")
    parser.add_argument("--coin", type=str, default="prl", help="Монета (prl или quantus)")
    parser.add_argument("--horizons", type=str, default="5,12,25", help="Горизонты через запятую")
    parser.add_argument("--min-test-sec", type=int, default=21600, help="Минимальный OOS-тест для нового окна (по умолч. 21600с = 6ч)")
    args = parser.parse_args()
    h_tuple = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    print_window_plan(coin=args.coin, horizons=h_tuple, min_test_sec=args.min_test_sec)
