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
sys.path.insert(0, str(BASE_DIR))

from _tools.data_labeler import compute_single_horizon_labels, COIN_CONFIG

SAFE_WORKERS = min(14, max(4, os.cpu_count() - 2 if os.cpu_count() else 4))

def parse_args():
    parser = argparse.ArgumentParser(description="Rolling Swarm Trainer")
    parser.add_argument("--coins", type=str, default="prl,quantus", help="Монеты через запятую")
    parser.add_argument("--horizons", type=str, default="5,12,25", help="Горизонты H (сек)")
    parser.add_argument("--k-values", type=str, default="", help="Значения K через запятую (например 3.0,3.5,3.0)")
    parser.add_argument("--n-estimators", type=int, default=70, help="Количество деревьев LightGBM")
    parser.add_argument("--n-jobs", type=int, default=SAFE_WORKERS, help="Количество потоков CPU")
    parser.add_argument("--align-horizon", type=int, default=None, help="Выравнивать временные окна под референсный горизонт H (например 12)")
    return parser.parse_args()


def get_default_k(coin: str, H: int) -> float:
    """
    Возвращает эмпирически подтвержденное оптимальное соотношение K = L / H (K >= 3.0).
    - H in [1..11]: K* = 3.0 (высокая скорость реакции BBO, L <= 33s)
    - H in [12..17]: K* = 3.5 (ядро удержания спреда, L in [42..60]s, погрешность < 1.0%)
    - H in [18..25]: K* = 3.0 (ограничение N_eff >= 5000, L <= 75s)
    """
    if H == 5:
        return 3.0
    elif H in (12, 13, 14, 15):
        return 3.5
    elif H >= 18:
        return 3.0
    elif H < 11:
        return 3.0
    else:
        return 3.5


def find_clean_segments(df: pd.DataFrame):
    """
    Разбивает датасет на непрерывные валидные сегменты, исключая разрывы связи (is_data_gap)
    и длительные заморозки цены (> 1800 с).
    Возвращает список диапазонов (start_idx, end_idx) в обратном хронологическом порядке (с конца).
    """
    is_valid = np.ones(len(df), dtype=bool)
    if 'sample_split' in df.columns:
        is_valid &= (df['sample_split'] != 'purged_gap') & (df['sample_split'] != 'warmup') & (df['sample_split'] != 'tail')
    if 'is_data_gap' in df.columns:
        is_valid &= (df['is_data_gap'] == 0)

    if 'ST_mid_price' in df.columns:
        mid_diff = df['ST_mid_price'].diff().abs().fillna(0).values
        tr_vol = df['ST_trade_vol'].fillna(0).values if 'ST_trade_vol' in df.columns else np.zeros(len(df))
        active_idx = np.where((mid_diff > 1e-8) | (tr_vol > 0))[0]
        if len(active_idx) > 1:
            gaps = np.diff(active_idx)
            for pos in np.where(gaps > 1800)[0]:
                is_valid[int(active_idx[pos]) + 1 : int(active_idx[pos + 1]) + 75] = False

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


def generate_rolling_windows_reverse(segments: list, H: int, K: float, n_target: int = 5000, oos_holdout_sec: int = 85600):
    """
    Генерирует окна строго с конца временного ряда по сшитому массиву чистых индексов (Seam-Purged Stitching).
    - Разрывы связи полностью вырезаются с защитным буфером seam_purge = 360с на краях каждого сегмента;
    - Самое свежее окно win_00 оставляет последние oos_holdout_sec (по умолчанию 43,200с = 12ч)
      пост-разрывного сегмента под строгий неприкосновенный OOS-тест, а в обучение win_00 включает
      первую половину пост-разрывного сегмента + чистую историю до разрыва.
    """
    L = int(round(H * K))
    t_train = int(n_target * L)
    t_test = int(t_train / 0.8)
    buffer = H + 360
    W = t_train + buffer + t_test
    step = t_train // 2

    seam_purge = 360
    chron_segs = sorted(segments, key=lambda x: x[0])
    clean_blocks = []
    for s_idx, (s, e) in enumerate(chron_segs):
        s_clean = s + (seam_purge if s_idx > 0 else 0)
        e_clean = e - (seam_purge if s_idx < len(chron_segs) - 1 else 0)
        if e_clean - s_clean > 1000:
            clean_blocks.append(np.arange(s_clean, e_clean, dtype=np.int64))

    if not clean_blocks:
        return [], W, step, t_train, t_test

    clean_idx = np.concatenate(clean_blocks)
    n_clean = len(clean_idx)
    windows = []

    te0_len = min(oos_holdout_sec, len(clean_blocks[-1]))
    te0_end_pos = n_clean
    te0_start_pos = te0_end_pos - te0_len
    tr0_end_pos = te0_start_pos - buffer
    tr0_start_pos = tr0_end_pos - t_train

    if tr0_start_pos >= 0:
        windows.append({
            'segment_id': 0,
            'tr_idx': clean_idx[tr0_start_pos:tr0_end_pos],
            'te_idx': clean_idx[te0_start_pos:te0_end_pos],
            'tr_start': int(clean_idx[tr0_start_pos]),
            'tr_end': int(clean_idx[tr0_end_pos - 1]) + 1,
            'te_start': int(clean_idx[te0_start_pos]),
            'te_end': int(clean_idx[te0_end_pos - 1]) + 1,
            't_train': t_train,
            't_test': te0_len,
            'W': t_train + buffer + te0_len,
            'L': L,
            'H': H,
            'K': K
        })
    curr_tr_end = tr0_end_pos - step
    while curr_tr_end - t_train >= 0 and len(windows) < 12:
        tr_e_pos = curr_tr_end
        tr_s_pos = tr_e_pos - t_train
        te_s_pos = tr_e_pos + buffer
        te_e_pos = min(n_clean, te_s_pos + step)
        if te_e_pos - te_s_pos < 3600:
            break
        windows.append({
            'segment_id': len(windows),
            'tr_idx': clean_idx[tr_s_pos:tr_e_pos],
            'te_idx': clean_idx[te_s_pos:te_e_pos],
            'tr_start': int(clean_idx[tr_s_pos]),
            'tr_end': int(clean_idx[tr_e_pos - 1]) + 1,
            'te_start': int(clean_idx[te_s_pos]),
            'te_end': int(clean_idx[te_e_pos - 1]) + 1,
            't_train': t_train,
            't_test': te_e_pos - te_s_pos,
            'W': t_train + buffer + (te_e_pos - te_s_pos),
            'L': L,
            'H': H,
            'K': K
        })
        curr_tr_end -= step

    return windows, W, step, t_train, t_test


def compute_regime_fingerprint(df_slice: pd.DataFrame):
    """Считывает макро- и микроструктурный паспорт рынка для RL-роутера."""
    fp = {}
    for col in [
        'ST_vol_300s_ticks', 'ST_vol_60s_ticks', 'ST_spread', 'ST_spread_pct',
        'ST_ofi_norm_300s', 'ST_cvd_5m', 'ST_cvd_norm_20s',
        'rvol_300s_ticks', 'ofi_norm_300s', 'cvd_norm_300s'
    ]:
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


def train_coin_horizon(coin: str, H: int, K: float, n_estimators: int, n_jobs: int, align_horizon: int = None):
    """Обучает полный пул Rolling-моделей для монеты и горизонта H."""
    coin = coin.lower()
    labeled_path = BASE_DIR / "data" / "labeled" / coin / f"{coin}_labeled.parquet"
    feats_json = BASE_DIR / "data" / "features" / f"selected_features_{coin}_h8s.json"

    if not labeled_path.exists() or not feats_json.exists():
        print(f"❌ Файлы для {coin.upper()} не найдены!")
        return

    with open(feats_json) as f:
        feature_cols = json.load(f)["feature_order"]

    k_str = f"{K:g}"
    target_c7_col = f"target_class_7_h{H}_k{k_str}"
    target_dt_col = f"target_dt_ticks_h{H}_k{k_str}"

    schema_cols = set(pq.read_schema(labeled_path).names)
    need_labeling = (target_c7_col not in schema_cols) or (target_dt_col not in schema_cols)

    base_cols = [
        'timestamp', 'sample_split', 'is_data_gap', 'ST_mid_price', 'ST_trade_vol', 'BO_sec_since_tick',
        'ST_vol_300s_ticks', 'ST_vol_60s_ticks', 'ST_spread', 'ST_spread_pct',
        'ST_ofi_norm_300s', 'ST_cvd_5m', 'ST_cvd_norm_20s'
    ]
    if need_labeling:
        orderbook_cols = [
            'ST_mid_price', 'ST_spread', 'ST_bid_1_price', 'ST_ask_1_price',
            'ST_bid_1_qty', 'ST_ask_1_qty', 'ST_bids_vol_top5', 'ST_asks_vol_top5',
            'ST_trade_vol'
        ]
        base_cols += orderbook_cols
    else:
        base_cols += [target_c7_col, target_dt_col]

    cols_to_load = list(dict.fromkeys([c for c in feature_cols + base_cols if c in schema_cols]))

    L = int(round(H * K))
    print(f"\n{'='*95}")
    print(f"🚀 ОБУЧЕНИЕ ROLLING SWARM: {coin.upper()} | H = {H}с, K = {k_str} (L = {L}с)")
    print(f"   Признаков: {len(feature_cols)} | Потоков CPU: {n_jobs} | Целевой таргет: {target_c7_col}")
    print(f"{'='*95}")

    t0 = time.time()
    tbl = pq.read_table(labeled_path, columns=cols_to_load)
    df = tbl.to_pandas()
    print(f"⏱️ Данные загружены за {time.time()-t0:.2f}с ({len(df):,} строк)")

    if need_labeling:
        print(f"⚙️ Динамический расчет микроструктурных меток для H = {H}с, K = {k_str}...")
        cfg = COIN_CONFIG[coin]
        labels = compute_single_horizon_labels(df, coin, H, K, cfg['min_vol'], cfg['tick_size'])
        df['target_class'] = labels['class_7']
        df['target_dt'] = labels['dt_ticks']
        df['is_void'] = labels['is_void']
    else:
        df['target_class'] = df[target_c7_col]
        df['target_dt'] = df[target_dt_col]
        df['is_void'] = False

    windows = []
    if align_horizon is not None:
        ref_k = get_default_k(coin, align_horizon)
        ref_manifest_path = BASE_DIR / "models" / coin / f"h{align_horizon}s_k{ref_k:g}" / "manifest.json"
        if ref_manifest_path.exists():
            print(f"🔗 Синхронизация временных окон с эталоном H={align_horizon}с ({ref_manifest_path.relative_to(BASE_DIR)})")
            with open(ref_manifest_path) as f:
                ref_manifest = json.load(f)

            ts_series = pd.to_datetime(df['timestamp'])
            for item in ref_manifest:
                tr_s_ts = pd.to_datetime(item['train_ts_range'][0])
                tr_e_ts = pd.to_datetime(item['train_ts_range'][1])
                te_s_ts = pd.to_datetime(item['test_ts_range'][0])
                te_e_ts = pd.to_datetime(item['test_ts_range'][1])

                if ts_series.dt.tz is not None:
                    tr_s_ts = tr_s_ts.tz_localize(ts_series.dt.tz) if tr_s_ts.tzinfo is None else tr_s_ts.tz_convert(ts_series.dt.tz)
                    tr_e_ts = tr_e_ts.tz_localize(ts_series.dt.tz) if tr_e_ts.tzinfo is None else tr_e_ts.tz_convert(ts_series.dt.tz)
                    te_s_ts = te_s_ts.tz_localize(ts_series.dt.tz) if te_s_ts.tzinfo is None else te_s_ts.tz_convert(ts_series.dt.tz)
                    te_e_ts = te_e_ts.tz_localize(ts_series.dt.tz) if te_e_ts.tzinfo is None else te_e_ts.tz_convert(ts_series.dt.tz)

                tr_s_idx = int(ts_series.searchsorted(tr_s_ts))
                tr_e_idx = int(ts_series.searchsorted(tr_e_ts))
                te_s_idx = int(ts_series.searchsorted(te_s_ts))
                te_e_idx = int(ts_series.searchsorted(te_e_ts))

                windows.append({
                    'segment_id': 0,
                    'tr_start': tr_s_idx,
                    'tr_end': tr_e_idx,
                    'te_start': te_s_idx,
                    'te_end': te_e_idx,
                    't_train': tr_e_idx - tr_s_idx,
                    't_test': te_e_idx - te_s_idx,
                    'W': te_e_idx - tr_s_idx,
                    'L': int(round(H * K)),
                    'H': H,
                    'K': K
                })
            print(f"✅ Успешно сформировано {len(windows)} окон строго по референсной сетке H={align_horizon}с!")
        else:
            print(f"⚠️ Референсный манифест {ref_manifest_path} не найден! Переход к стандартной нарезке.")

    if not windows:
        # Поиск непрерывных сегментов (без гэпов)
        segments = find_clean_segments(df)
        windows, W, step, t_train, t_test = generate_rolling_windows_reverse(segments, H, K)

        print(f"📐 Формула окна W = Train ({t_train:,}с) + Buffer + Test ({t_test:,}с) = {W:,}с ({W/3600:.1f}ч)")
        print(f"   Шаг сдвига W/2 = {step:,}с ({step/3600:.1f}ч) | Всего валидных окон: {len(windows)}")

    if not windows:
        print("⚠️ Нет непрерывных сегментов достаточной длины для данного горизонта!")
        return

    out_dir = BASE_DIR / "models" / coin / f"h{H}s_k{k_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    print(f"\n{'Win':<6} | {'Train TS':<33} | {'Test TS':<33} | {'OOS IC':<8} | {'AUC OVR':<8} | {'Top-2% PnL':<11} | {'Статус':<10}")
    print("-" * 125)

    X_all = df[feature_cols].values.copy()
    if 'BO_sec_since_tick' in df.columns:
        bo_stale = (df['BO_sec_since_tick'].fillna(0).values > 60.0)
        if np.any(bo_stale):
            zero_prefixes = ('LL_', 'BO_ofi_', 'BO_cvd_', 'BO_imb_', 'BO_obi_', 'BO_micro_', 'BO_ret_', 'BO_vol_intensity_')
            zero_cols = [i_c for i_c, fn in enumerate(feature_cols) if fn.startswith(zero_prefixes)]
            if zero_cols:
                X_all[np.ix_(bo_stale, zero_cols)] = 0.0

    for idx, w in enumerate(windows):
        win_id = f"win_{idx:02d}"
        tr_s, tr_e = w['tr_start'], w['tr_end']
        te_s, te_e = w['te_start'], w['te_end']
        tr_idx = w.get('tr_idx', np.arange(tr_s, tr_e))
        te_idx = w.get('te_idx', np.arange(te_s, te_e))

        ts_tr_start = str(df['timestamp'].iloc[min(tr_s, len(df)-1)])[:19]
        ts_tr_end = str(df['timestamp'].iloc[min(tr_e - 1, len(df)-1)])[:19]
        ts_te_start = str(df['timestamp'].iloc[min(te_s, len(df)-1)])[:19]
        ts_te_end = str(df['timestamp'].iloc[min(te_e - 1, len(df)-1)])[:19]

        X_train = X_all[tr_idx]
        y_train = df['target_class'].iloc[tr_idx].values + 3  # Сдвиг [-3..3] -> [0..6]

        X_test = X_all[te_idx]
        y_test = df['target_class'].iloc[te_idx].values + 3
        dt_test = df['target_dt'].iloc[te_idx].values

        # Обучение LightGBM
        df_train = pd.DataFrame(X_train, columns=feature_cols)
        df_test = pd.DataFrame(X_test, columns=feature_cols)

        # Асимметричные веса классов для защиты от False Positive Buy на проливах
        from sklearn.utils.class_weight import compute_class_weight
        base_weights = compute_class_weight('balanced', classes=np.arange(7), y=y_train)
        custom_weights = dict(enumerate(base_weights))
        custom_weights[0] *= 1.35  # Class -3 (защита от ножей)
        custom_weights[1] *= 1.25  # Class -2
        custom_weights[2] *= 1.10  # Class -1

        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=7,
            class_weight=custom_weights,
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
            present_classes = [c for c in range(7) if np.sum(y_test == c) > 0]
            if len(present_classes) > 1:
                # Нормализация вероятностей для присутствующих классов
                p_sub = full_probs[:, present_classes]
                p_sub = p_sub / np.maximum(p_sub.sum(axis=1, keepdims=True), 1e-7)
                auc = float(roc_auc_score(y_test, p_sub, multi_class='ovr', labels=present_classes))
            else:
                auc = 0.5
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
        fp_train = compute_regime_fingerprint(df.iloc[tr_idx])
        fp_test = compute_regime_fingerprint(df.iloc[te_idx])

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
    k_vals = [float(k.strip()) for k in args.k_values.split(',') if k.strip()] if args.k_values else None

    print(f"\n{'='*80}")
    print(f"🤖 ЗАПУСК ОБУЧЕНИЯ ПУЛА ЭКСПЕРТОВ (ROLLING SWARM) С РАЗМЕТКОЙ С КОНЦА")
    print(f"   Монеты: {coins} | Горизонты: {horizons} | Потоков CPU: {args.n_jobs}")
    if k_vals:
        print(f"   Пользовательские K: {k_vals}")
    print(f"{'='*80}")

    for coin in coins:
        for idx, H in enumerate(horizons):
            if k_vals and idx < len(k_vals):
                K = k_vals[idx]
            else:
                K = get_default_k(coin, H)
            train_coin_horizon(coin, H, K, args.n_estimators, args.n_jobs, align_horizon=args.align_horizon)


if __name__ == '__main__':
    main()

