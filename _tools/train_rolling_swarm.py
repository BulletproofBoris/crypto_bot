#!/usr/bin/env python3
"""
Обучение пула моделей (Rolling Swarm) с разметкой окон с конца временного ряда.
Формула окна пользователя:
- T_train = 5000 * L
- T_test = int(5000 * L / 0.8) = 6250 * L
- Buffer = H + 360 с (Purge + Embargo)
- W = T_train + Buffer + T_test
- Step = W // 2 (50% перекрытие окон)
- Направление: с конца последовательности (наиболее свежие данные -> окно 00).
- Старый хвост в начале отбрасывается.
- Для каждого окна сохраняется чекпоинт LightGBM (7 классов) + JSON-паспорт режима для RL-роутера.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

BASE_DIR = Path(__file__).resolve().parent.parent

SAFE_WORKERS = min(14, max(4, os.cpu_count() - 2 if os.cpu_count() else 4))

def parse_args():
    parser = argparse.ArgumentParser(description="Rolling Swarm Trainer")
    parser.add_argument("--coins", type=str, default="prl,quantus", help="Монеты через запятую")
    parser.add_argument("--horizons", type=str, default="3,8", help="Горизонты H (сек)")
    parser.add_argument("--n-estimators", type=int, default=70, help="Количество деревьев LightGBM")
    parser.add_argument("--n-jobs", type=int, default=SAFE_WORKERS, help="Количество потоков CPU")
    return parser.parse_args()


def get_default_k(coin: str, H: int) -> int:
    """Возвращает утвержденное оптимальное соотношение K = L / H."""
    c = coin.lower()
    if H == 3:
        return 1
    elif H == 8:
        return 2 if c == 'prl' else 1
    elif H == 25:
        return 3 if c == 'prl' else 2
    return 1


def find_clean_segments(df: pd.DataFrame):
    """
    Разбивает датасет на непрерывные валидные сегменты, исключая разрывы связи (is_data_gap).
    Возвращает список диапазонов (start_idx, end_idx) в обратном хронологическом порядке (с конца).
    """
    is_valid = np.ones(len(df), dtype=bool)
    if 'sample_split' in df.columns:
        is_valid &= (df['sample_split'] != 'purged_gap') & (df['sample_split'] != 'warmup') & (df['sample_split'] != 'tail')
    if 'is_data_gap' in df.columns:
        is_valid &= (df['is_data_gap'] == 0)

    # Ищем непрерывные блоки True
    change_points = np.diff(np.pad(is_valid.astype(np.int8), (1, 1), 'constant'))
    starts = np.where(change_points == 1)[0]
    ends = np.where(change_points == -1)[0]

    segments = []
    for s, e in zip(starts, ends):
        if e - s > 1000:
            segments.append((s, e))

    # Сортируем с конца (самый свежий сегмент первым)
    segments.sort(key=lambda x: x[0], reverse=True)
    return segments


def generate_rolling_windows_reverse(segments: list, H: int, K: int, n_target: int = 5000):
    """
    Генерирует окна строго с конца временного ряда:
    T_train = 5000 * L
    T_test = int(5000 * L / 0.8)
    W = T_train + buffer + T_test
    Step = W // 2
    """
    L = H * K
    t_train = n_target * L
    t_test = int(n_target * L / 0.8)
    buffer = H + 360
    W = t_train + buffer + t_test
    step = W // 2

    windows = []
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        curr_end = seg_end
        while curr_end - W >= seg_start:
            w_start = curr_end - W
            tr_start = w_start
            tr_end = tr_start + t_train
            te_start = tr_end + buffer
            te_end = curr_end

            windows.append({
                'segment_id': seg_idx,
                'tr_start': tr_start,
                'tr_end': tr_end,
                'te_start': te_start,
                'te_end': te_end,
                't_train': t_train,
                't_test': te_end - te_start,
                'W': W,
                'L': L,
                'H': H,
                'K': K
            })
            curr_end -= step

    return windows, W, step, t_train, t_test


def compute_regime_fingerprint(df_slice: pd.DataFrame):
    """Считывает макро- и микроструктурный паспорт рынка для RL-роутера."""
    fp = {}
    for col in ['rvol_300s_ticks', 'rvol_60s_ticks', 'ST_spread', 'ST_spread_pct', 'ofi_norm_300s', 'cvd_norm_300s']:
        if col in df_slice.columns:
            fp[f"median_{col}"] = float(np.nanmedian(df_slice[col].values))
            fp[f"mean_{col}"] = float(np.nanmean(df_slice[col].values))
            fp[f"std_{col}"] = float(np.nanstd(df_slice[col].values))

    # Доля классов
    if 'target_class' in df_slice.columns:
        counts = df_slice['target_class'].value_counts(normalize=True).to_dict()
        for cls in range(-3, 4):
            fp[f"pct_class_{cls:+d}"] = float(counts.get(cls, 0.0) * 100.0)

    return fp


def train_coin_horizon(coin: str, H: int, K: int, n_estimators: int, n_jobs: int):
    """Обучает полный пул Rolling-моделей для монеты и горизонта H."""
    coin = coin.lower()
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    feats_json = BASE_DIR / "data" / "features" / f"selected_features_{coin}_h8s.json"

    if not labeled_path.exists() or not feats_json.exists():
        print(f"❌ Файлы для {coin.upper()} не найдены!")
        return

    with open(feats_json) as f:
        feature_cols = json.load(f)["feature_order"]

    target_c7_col = f"target_class_7_h{H}_k{K}"
    target_dt_col = f"target_dt_ticks_h{H}_k{K}"

    cols_to_load = list(set(feature_cols + [
        'timestamp', 'sample_split', 'is_data_gap',
        target_c7_col, target_dt_col,
        'rvol_300s_ticks', 'ST_spread', 'ST_spread_pct', 'ofi_norm_300s', 'cvd_norm_300s'
    ]))
    cols_to_load = [c for c in cols_to_load if c in pq.read_schema(labeled_path).names]

    print(f"\n{'='*95}")
    print(f"🚀 ОБУЧЕНИЕ ROLLING SWARM: {coin.upper()} | H = {H}с, K = {K} (L = {H*K}с)")
    print(f"   Признаков: {len(feature_cols)} | Потоков CPU: {n_jobs} | Целевой таргет: {target_c7_col}")
    print(f"{'='*95}")

    t0 = time.time()
    tbl = pq.read_table(labeled_path, columns=cols_to_load)
    df = tbl.to_pandas()
    df['target_class'] = df[target_c7_col]
    df['target_dt'] = df[target_dt_col]
    print(f"⏱️ Данные загружены за {time.time()-t0:.2f}с ({len(df):,} строк)")

    # Поиск непрерывных сегментов (без гэпов)
    segments = find_clean_segments(df)
    windows, W, step, t_train, t_test = generate_rolling_windows_reverse(segments, H, K)

    print(f"📐 Формула окна W = Train ({t_train:,}с) + Buffer + Test ({t_test:,}с) = {W:,}с ({W/3600:.1f}ч)")
    print(f"   Шаг сдвига W/2 = {step:,}с ({step/3600:.1f}ч) | Всего валидных окон: {len(windows)}")

    if not windows:
        print("⚠️ Нет непрерывных сегментов достаточной длины для данного горизонта!")
        return

    out_dir = BASE_DIR / "models" / coin / f"h{H}s_k{K}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    print(f"\n{'Win':<6} | {'Train TS':<33} | {'Test TS':<33} | {'OOS IC':<8} | {'AUC OVR':<8} | {'Top-2% PnL':<11} | {'Статус':<10}")
    print("-" * 125)

    X_all = df[feature_cols].values

    for idx, w in enumerate(windows):
        win_id = f"win_{idx:02d}"
        tr_s, tr_e = w['tr_start'], w['tr_end']
        te_s, te_e = w['te_start'], w['te_end']

        ts_tr_start = str(df['timestamp'].iloc[tr_s])[:19]
        ts_tr_end = str(df['timestamp'].iloc[tr_e])[:19]
        ts_te_start = str(df['timestamp'].iloc[te_s])[:19]
        ts_te_end = str(df['timestamp'].iloc[te_e])[:19]

        X_train = X_all[tr_s:tr_e]
        y_train = df['target_class'].iloc[tr_s:tr_e].values + 3  # Сдвиг [-3..3] -> [0..6]

        X_test = X_all[te_s:te_e]
        y_test = df['target_class'].iloc[te_s:te_e].values + 3
        dt_test = df['target_dt'].iloc[te_s:te_e].values

        # Обучение LightGBM
        df_train = pd.DataFrame(X_train, columns=feature_cols)
        df_test = pd.DataFrame(X_test, columns=feature_cols)

        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=7,
            class_weight="balanced",
            n_estimators=n_estimators,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42 + idx,
            n_jobs=n_jobs,
            verbose=-1
        )
        clf.fit(df_train, y_train)

        # Оценка на чистом Out-of-Sample
        probs_test = clf.predict_proba(df_test)
        full_probs = np.zeros((len(df_test), 7), dtype=np.float32)
        for col_i, cls_val in enumerate(clf.classes_):
            if 0 <= cls_val < 7:
                full_probs[:, cls_val] = probs_test[:, col_i]
        
        # Ожидаемый импульс E[z] = sum P(c) * c
        classes_values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)
        expected_z = np.dot(full_probs, classes_values)

        # Метрики OOS
        ic = float(np.corrcoef(expected_z, dt_test)[0, 1]) if np.std(expected_z) > 1e-6 else 0.0
        
        try:
            auc = float(roc_auc_score(y_test, full_probs, multi_class='ovr', labels=list(range(7))))
        except Exception:
            auc = 0.5

        # Доходность Top-2% сигналов (покупка по экстремальному ожидаемому z)
        top2_threshold = np.quantile(expected_z, 0.98)
        top2_mask = expected_z >= top2_threshold
        top2_pnl = float(np.mean(dt_test[top2_mask])) if np.sum(top2_mask) > 0 else 0.0

        # Сохранение модели
        model_path = out_dir / f"model_{win_id}.txt"
        clf.booster_.save_model(str(model_path))

        # Паспорт режима (Regime Fingerprint)
        fp_train = compute_regime_fingerprint(df.iloc[tr_s:tr_e])
        fp_test = compute_regime_fingerprint(df.iloc[te_s:te_e])

        meta = {
            'window_id': win_id,
            'coin': coin.upper(),
            'H': H,
            'K': K,
            'L': H * K,
            'train_ts_range': [ts_tr_start, ts_tr_end],
            'test_ts_range': [ts_te_start, ts_te_end],
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'metrics_oos': {
                'ic_dt_ticks': round(ic, 4),
                'roc_auc_ovr': round(auc, 4),
                'top2pct_mean_ticks': round(top2_pnl, 2),
            },
            'regime_fingerprint_train': fp_train,
            'regime_fingerprint_test': fp_test,
            'model_file': model_path.name
        }

        meta_path = out_dir / f"meta_{win_id}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        manifest.append(meta)

        status = "🎯 ОТЛИЧНО" if ic > 0.15 and top2_pnl > 0.5 else ("✅ ХОРОШО" if ic > 0.05 else "⚠️ СЛАБО")
        print(f"{win_id:<6} | {ts_tr_start} -> {ts_tr_end} | {ts_te_start} -> {ts_te_end} | {ic:+6.4f}  | {auc:6.4f}   | {top2_pnl:+6.2f} тиков | {status:<10}")

    with open(out_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print("-" * 125)
    print(f"✅ Пул из {len(manifest)} моделей успешно сохранен в: {out_dir.relative_to(BASE_DIR)}")


def main():
    args = parse_args()
    coins = [c.strip().lower() for c in args.coins.split(',') if c.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(',') if h.strip()]

    print(f"\n{'='*80}")
    print(f"🤖 ЗАПУСК ОБУЧЕНИЯ ПУЛА ЭКСПЕРТОВ (ROLLING SWARM) С РАЗМЕТКОЙ С КОНЦА")
    print(f"   Монеты: {coins} | Горизонты: {horizons} | Потоков CPU: {args.n_jobs}")
    print(f"{'='*80}")

    for coin in coins:
        for H in horizons:
            K = get_default_k(coin, H)
            train_coin_horizon(coin, H, K, args.n_estimators, args.n_jobs)


if __name__ == '__main__':
    main()

