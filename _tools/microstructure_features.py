"""
Модуль генерации стационарных микроструктурных признаков и многомасштабного банка фильтров.
Использует геометрическую сетку окон W = [5, 20, 60, 300] секунд (шаг 3x-5x между полосами частот),
обеспечивающую статистическую достаточность (N_eff >= 4000 на микроструктурных окнах <= 60с)
и ортогональность полосовых фильтров (Band-Pass).
"""
import numpy as np
import pandas as pd

# Геометрическая сетка частотных полос (в секундах):
# 5s  = Горизонт исполнения H (Impulse band)
# 20s = Окно микроструктурной волны L = 4*H и время блока (Wave band)
# 60s = Локальное равновесие потока ордеров (Equilibrium band, N_eff ~ 4000..5500)
# 300s = Медленный контекст режима и волатильности для Z-нормировки (Regime band)
FILTER_WINDOWS = (5, 20, 60, 300)

TICK_SIZES = {
    "prl": 0.005,
    "quantus": 0.05,
}


def compute_ofi_1s(df_1s, prefix):
    """
    Рассчитывает 1-секундный Order Flow Imbalance (Cont-Kukanov-Stoikov OFI) на лучших уровнях BBO.
    Учитывает как сделки, так и подстановку/снятие лимитных заявок.
    """
    b_p = df_1s[f"{prefix}_bid_1_price"].astype(float)
    b_q = df_1s[f"{prefix}_bid_1_qty"].astype(float)
    a_p = df_1s[f"{prefix}_ask_1_price"].astype(float)
    a_q = df_1s[f"{prefix}_ask_1_qty"].astype(float)

    b_p_prev = b_p.shift(1)
    b_q_prev = b_q.shift(1)
    a_p_prev = a_p.shift(1)
    a_q_prev = a_q.shift(1)

    e_bid = np.where(b_p >= b_p_prev, b_q, 0.0) - np.where(b_p <= b_p_prev, b_q_prev, 0.0)
    e_ask = np.where(a_p <= a_p_prev, a_q, 0.0) - np.where(a_p >= a_p_prev, a_q_prev, 0.0)

    ofi_raw = pd.Series(e_bid - e_ask, index=df_1s.index).fillna(0.0)
    return ofi_raw


def compute_multilevel_ofi_1s(df_1s, prefix, max_levels=5):
    """
    Рассчитывает многоуровневый кумулятивный OFI (Cont-Kukanov-Stoikov) по топ-5 уровням стакана
    с коэффициентами затухания w_k = [1.0, 0.7, 0.5, 0.35, 0.25].
    Учитывает скрытые намерения и спуфинг маркет-мейкеров на глубине L2..L5.
    """
    weights = [1.0, 0.7, 0.5, 0.35, 0.25]
    cum_ofi = pd.Series(0.0, index=df_1s.index)

    for k in range(1, max_levels + 1):
        bp_col = f"{prefix}_bid_{k}_price"
        bq_col = f"{prefix}_bid_{k}_qty"
        ap_col = f"{prefix}_ask_{k}_price"
        aq_col = f"{prefix}_ask_{k}_qty"
        if bp_col not in df_1s.columns or bq_col not in df_1s.columns or ap_col not in df_1s.columns or aq_col not in df_1s.columns:
            continue

        bp = df_1s[bp_col].astype(float)
        bq = df_1s[bq_col].astype(float)
        ap = df_1s[ap_col].astype(float)
        aq = df_1s[aq_col].astype(float)

        bp_prev = bp.shift(1)
        bq_prev = bq.shift(1)
        ap_prev = ap.shift(1)
        aq_prev = aq.shift(1)

        e_bid = np.where(bp > bp_prev, bq, np.where(bp == bp_prev, bq - bq_prev, -bq_prev))
        e_ask = np.where(ap < ap_prev, aq, np.where(ap == ap_prev, aq - aq_prev, -aq_prev))

        ofi_k = pd.Series(e_bid - e_ask, index=df_1s.index).fillna(0.0)
        w = weights[k - 1] if (k - 1) < len(weights) else 1.0 / np.sqrt(k)
        cum_ofi += w * ofi_k

    return cum_ofi


def add_exchange_filter_bank(df_1s, prefix, coin="prl"):
    """
    Добавляет стационарные признаки геометрии стакана и многочастотный банк фильтров
    W = [5, 20, 60, 300] с полосовыми разностями (Band-Pass) для биржи (ST или BO).
    """
    tick = TICK_SIZES.get(coin.lower(), 0.005)
    new_cols = {}

    mid = df_1s[f"{prefix}_mid_price"].astype(float)
    b1_p = df_1s[f"{prefix}_bid_1_price"].astype(float)
    a1_p = df_1s[f"{prefix}_ask_1_price"].astype(float)
    b1_q = df_1s[f"{prefix}_bid_1_qty"].astype(float)
    a1_q = df_1s[f"{prefix}_ask_1_qty"].astype(float)
    b5_q = df_1s[f"{prefix}_bids_vol_top5"].astype(float)
    a5_q = df_1s[f"{prefix}_asks_vol_top5"].astype(float)

    # 1. Стационарная геометрия стакана (в тиках и долях)
    tot_q1 = (b1_q + a1_q).replace(0, np.nan)
    microprice = (b1_p * a1_q + a1_p * b1_q) / tot_q1
    new_cols[f"{prefix}_microprice_offset_ticks"] = ((microprice - mid) / tick).fillna(0.0)
    new_cols[f"{prefix}_spread_ticks"] = ((a1_p - b1_p) / tick).fillna(0.0)
    new_cols[f"{prefix}_bbo_imbalance"] = ((b1_q - a1_q) / tot_q1).fillna(0.0)
    new_cols[f"{prefix}_bid1_dominance"] = (b1_q / b5_q.replace(0, np.nan)).fillna(0.0)
    new_cols[f"{prefix}_ask1_dominance"] = (a1_q / a5_q.replace(0, np.nan)).fillna(0.0)
    new_cols[f"{prefix}_log_book_pressure_top5"] = np.log((df_1s[f"{prefix}_bids_usd_top5"] + 1.0) / (df_1s[f"{prefix}_asks_usd_top5"] + 1.0))

    # Кинематика набегания волны микроцены (скорость, ускорение, отношение к спреду)
    mp_offset = new_cols[f"{prefix}_microprice_offset_ticks"]
    new_cols[f"{prefix}_wave_vel_2s"] = (mp_offset - mp_offset.shift(2)).fillna(0.0)
    new_cols[f"{prefix}_wave_vel_5s"] = ((mp_offset - mp_offset.shift(5)) / 5.0).fillna(0.0)
    new_cols[f"{prefix}_wave_vel_10s"] = ((mp_offset - mp_offset.shift(10)) / 10.0).fillna(0.0)
    new_cols[f"{prefix}_wave_accel_5s"] = (new_cols[f"{prefix}_wave_vel_2s"] - new_cols[f"{prefix}_wave_vel_2s"].shift(3)).fillna(0.0)
    new_cols[f"{prefix}_wave_to_spread_ratio"] = (new_cols[f"{prefix}_wave_vel_5s"] / (new_cols[f"{prefix}_spread_ticks"] + 0.5)).fillna(0.0)

    # Относительные расстояния уровней 2..5 от mid_price (в тиках)
    for k in range(2, 6):
        if f"{prefix}_bid_{k}_price" in df_1s.columns:
            bp_k = df_1s[f"{prefix}_bid_{k}_price"].astype(float)
            new_cols[f"{prefix}_bid_{k}_dist_ticks"] = np.where(bp_k > 0, (mid - bp_k) / tick, np.nan)
        if f"{prefix}_ask_{k}_price" in df_1s.columns:
            ap_k = df_1s[f"{prefix}_ask_{k}_price"].astype(float)
            new_cols[f"{prefix}_ask_{k}_dist_ticks"] = np.where(ap_k > 0, (ap_k - mid) / tick, np.nan)

    # Крутизна (Book Slope / Kyle's lambda) и выпуклость книги заявок (Curvature)
    bp_5 = df_1s[f"{prefix}_bid_5_price"].astype(float) if f"{prefix}_bid_5_price" in df_1s.columns else b1_p
    ap_5 = df_1s[f"{prefix}_ask_5_price"].astype(float) if f"{prefix}_ask_5_price" in df_1s.columns else a1_p
    bid_dist_5 = np.maximum(0.5, (mid - bp_5) / tick)
    ask_dist_5 = np.maximum(0.5, (ap_5 - mid) / tick)
    bid_slope = b5_q / bid_dist_5
    ask_slope = a5_q / ask_dist_5
    new_cols[f"{prefix}_book_slope_bid"] = bid_slope.fillna(0.0)
    new_cols[f"{prefix}_book_slope_ask"] = ask_slope.fillna(0.0)
    new_cols[f"{prefix}_log_slope_ratio"] = (np.log((bid_slope + 1.0) / (ask_slope + 1.0))).fillna(0.0)

    # Выпуклость стакана (Convexity топ-10) и затухание плотности вглубь
    b10_q = sum(df_1s[f"{prefix}_bid_{k}_qty"].astype(float) for k in range(1, 11) if f"{prefix}_bid_{k}_qty" in df_1s.columns)
    a10_q = sum(df_1s[f"{prefix}_ask_{k}_qty"].astype(float) for k in range(1, 11) if f"{prefix}_ask_{k}_qty" in df_1s.columns)
    b6_10_q = np.maximum(0.0, b10_q - b5_q)
    a6_10_q = np.maximum(0.0, a10_q - a5_q)
    new_cols[f"{prefix}_bid_convexity_top10"] = ((b6_10_q - b5_q) / (b10_q + 1.0)).fillna(0.0)
    new_cols[f"{prefix}_ask_convexity_top10"] = ((a6_10_q - a5_q) / (a10_q + 1.0)).fillna(0.0)
    b11_50_q = df_1s[f"{prefix}_bids_vol_11_50"].astype(float) if f"{prefix}_bids_vol_11_50" in df_1s.columns else pd.Series(0.0, index=df_1s.index)
    new_cols[f"{prefix}_liquidity_density_decay"] = (b5_q / (b11_50_q + 1.0)).fillna(0.0)

    # 2. Многомасштабный банк фильтров по W = [5, 20, 60, 300]
    ofi_1s = compute_ofi_1s(df_1s, prefix)
    ofi_top5_1s = compute_multilevel_ofi_1s(df_1s, prefix)
    delta_vol = df_1s[f"{prefix}_delta_vol"].astype(float)
    trade_vol = df_1s[f"{prefix}_trade_vol"].astype(float)
    trade_cnt = df_1s[f"{prefix}_trade_count"].astype(float)
    buy_vol = df_1s[f"{prefix}_buy_vol"].astype(float)
    sell_vol = df_1s[f"{prefix}_sell_vol"].astype(float)
    depth_scale = (b5_q + a5_q).rolling(60, min_periods=1).median().replace(0, np.nan).ffill().fillna(1.0)
    ret_1s_ticks = mid.diff().fillna(0.0) / tick

    # Истощение очередей BBO и доля отмен (Cancel vs Trade)
    b_drop = np.where((b1_p == b1_p.shift(1)) & (b1_q < b1_q.shift(1)), b1_q.shift(1) - b1_q, 0.0)
    a_drop = np.where((a1_p == a1_p.shift(1)) & (a1_q < a1_q.shift(1)), a1_q.shift(1) - a1_q, 0.0)
    new_cols[f"{prefix}_queue_depletion_bid_5s"] = (pd.Series(b_drop, index=df_1s.index).rolling(5, min_periods=1).mean() / depth_scale).fillna(0.0)
    new_cols[f"{prefix}_queue_depletion_ask_5s"] = (pd.Series(a_drop, index=df_1s.index).rolling(5, min_periods=1).mean() / depth_scale).fillna(0.0)
    b_cancel = np.maximum(0.0, b_drop - sell_vol)
    a_cancel = np.maximum(0.0, a_drop - buy_vol)
    tot_cancel_5s = pd.Series(b_cancel + a_cancel, index=df_1s.index).rolling(5, min_periods=1).sum()
    tot_trade_5s = trade_vol.rolling(5, min_periods=1).sum()
    new_cols[f"{prefix}_cancel_to_trade_ratio_5s"] = (tot_cancel_5s / (tot_trade_5s + 0.1)).fillna(0.0).clip(0.0, 50.0)

    # Hawkes самовозбуждение сделок, проедание BBO и сметание стакана (Whale Sweep)
    n_2s = trade_cnt.rolling(2, min_periods=1).sum()
    n_20s = trade_cnt.rolling(20, min_periods=1).sum()
    new_cols[f"{prefix}_trade_arrival_accel"] = ((n_2s / 2.0) / (n_20s / 20.0 + 0.05)).fillna(0.0).clip(0.0, 20.0)
    new_cols[f"{prefix}_absorption_ask_1s"] = (buy_vol / (a1_q + 0.1)).fillna(0.0).clip(0.0, 20.0)
    new_cols[f"{prefix}_absorption_bid_1s"] = (sell_vol / (b1_q + 0.1)).fillna(0.0).clip(0.0, 20.0)
    sweep_buy = np.maximum(0.0, buy_vol - a1_q.shift(1))
    sweep_sell = np.maximum(0.0, sell_vol - b1_q.shift(1))
    sweep_vol_5s = pd.Series(sweep_buy + sweep_sell, index=df_1s.index).rolling(5, min_periods=1).sum()
    new_cols[f"{prefix}_whale_sweep_ratio_5s"] = (sweep_vol_5s / (tot_trade_5s + 0.1)).fillna(0.0).clip(0.0, 1.0)

    # Асимметричная полуволатильность и скачки диапазона
    pos_ret = np.maximum(0.0, ret_1s_ticks)
    neg_ret = np.maximum(0.0, -ret_1s_ticks)
    upside_semi = np.sqrt(pd.Series(pos_ret ** 2, index=df_1s.index).rolling(20, min_periods=1).mean()).fillna(0.0)
    downside_semi = np.sqrt(pd.Series(neg_ret ** 2, index=df_1s.index).rolling(20, min_periods=1).mean()).fillna(0.0)
    new_cols[f"{prefix}_upside_semi_vol_20s"] = upside_semi
    new_cols[f"{prefix}_downside_semi_vol_20s"] = downside_semi
    new_cols[f"{prefix}_volatility_skewness_20s"] = ((upside_semi - downside_semi) / (upside_semi + downside_semi + 0.01)).fillna(0.0)
    range_20s = (mid.rolling(20, min_periods=1).max() - mid.rolling(20, min_periods=1).min()) / tick
    new_cols[f"{prefix}_range_to_spread_20s"] = (range_20s / (new_cols[f"{prefix}_spread_ticks"] + 0.5)).fillna(0.0)

    # Скрытый эффективный спред модели Ролла и шум микроколебаний
    dp_prev = ret_1s_ticks.shift(1).fillna(0.0)
    cov_60s = (ret_1s_ticks * dp_prev).rolling(60, min_periods=5).mean() - ret_1s_ticks.rolling(60, min_periods=5).mean() * dp_prev.rolling(60, min_periods=5).mean()
    rolls_spread = 2.0 * np.sqrt(np.maximum(0.0, -cov_60s.fillna(0.0)))
    new_cols[f"{prefix}_rolls_spread_proxy"] = rolls_spread

    for w in FILTER_WINDOWS:
        # Нормированная скорость изменения цены (тиков в секунду за окно W) и импульс в тиках
        new_cols[f"{prefix}_ret_{w}s_ticks"] = (mid - mid.shift(w)).fillna(0.0) / tick
        new_cols[f"{prefix}_vol_{w}s_ticks"] = ret_1s_ticks.rolling(w, min_periods=1).std().fillna(0.0)
        # Нормированный поток ордеров OFI (уровень 1 и многоуровневый) и CVD
        new_cols[f"{prefix}_ofi_norm_{w}s"] = (ofi_1s.rolling(w, min_periods=1).mean() / depth_scale).fillna(0.0)
        new_cols[f"{prefix}_ofi_top5_norm_{w}s"] = (ofi_top5_1s.rolling(w, min_periods=1).mean() / depth_scale).fillna(0.0)
        new_cols[f"{prefix}_cvd_norm_{w}s"] = (delta_vol.rolling(w, min_periods=1).mean() / depth_scale).fillna(0.0)
        new_cols[f"{prefix}_vol_intensity_{w}s"] = (trade_vol.rolling(w, min_periods=1).mean() / depth_scale).fillna(0.0)
        new_cols[f"{prefix}_imbalance_ema_{w}s"] = df_1s[f"{prefix}_imbalance_top5"].ewm(span=w, adjust=False).mean()

    # 3. Ортогональные полосовые фильтры (Band-Pass разности между соседними частотами)
    new_cols[f"{prefix}_ofi_band_5_20"] = new_cols[f"{prefix}_ofi_norm_5s"] - new_cols[f"{prefix}_ofi_norm_20s"]
    new_cols[f"{prefix}_ofi_band_20_60"] = new_cols[f"{prefix}_ofi_norm_20s"] - new_cols[f"{prefix}_ofi_norm_60s"]
    new_cols[f"{prefix}_ofi_band_60_300"] = new_cols[f"{prefix}_ofi_norm_60s"] - new_cols[f"{prefix}_ofi_norm_300s"]
    new_cols[f"{prefix}_ofi_top5_band_5_20"] = new_cols[f"{prefix}_ofi_top5_norm_5s"] - new_cols[f"{prefix}_ofi_top5_norm_20s"]
    new_cols[f"{prefix}_ofi_top5_band_20_60"] = new_cols[f"{prefix}_ofi_top5_norm_20s"] - new_cols[f"{prefix}_ofi_top5_norm_60s"]

    new_cols[f"{prefix}_imb_band_5_20"] = new_cols[f"{prefix}_imbalance_ema_5s"] - new_cols[f"{prefix}_imbalance_ema_20s"]
    new_cols[f"{prefix}_imb_band_20_60"] = new_cols[f"{prefix}_imbalance_ema_20s"] - new_cols[f"{prefix}_imbalance_ema_60s"]
    new_cols[f"{prefix}_vol_ratio_5_60"] = new_cols[f"{prefix}_vol_5s_ticks"] / (new_cols[f"{prefix}_vol_60s_ticks"] + 0.05)
    new_cols[f"{prefix}_vol_ratio_20_300"] = new_cols[f"{prefix}_vol_20s_ticks"] / (new_cols[f"{prefix}_vol_300s_ticks"] + 0.05)
    new_cols[f"{prefix}_noise_to_signal_bounce_ratio"] = (rolls_spread / (new_cols[f"{prefix}_vol_60s_ticks"] + 0.05)).fillna(0.0)

    ext_df = pd.DataFrame(new_cols, index=df_1s.index).ffill().fillna(0.0)
    overlap = [c for c in ext_df.columns if c in df_1s.columns]
    if overlap:
        df_1s = df_1s.drop(columns=overlap)
    return pd.concat([df_1s, ext_df], axis=1)


def add_cross_exchange_and_macro_filters(merged, coin="prl"):
    """
    Добавляет кросс-биржевые опережающие фильтры (Lead-Lag BigONE/Rhea -> SafeTrade),
    адаптивные Z-score межбиржевого спреда и многомасштабные доходности поводырей BTC/NEAR.
    Удаляет нестационарные накопительные счётчики.
    """
    tick = TICK_SIZES.get(coin.lower(), 0.005)
    new_cols = {}

    if "BO_mid_price" in merged.columns and "ST_mid_price" in merged.columns:
        diff_ticks = (merged["BO_mid_price"] - merged["ST_mid_price"]) / tick
        new_cols["arb_mid_diff_ticks"] = diff_ticks

        for w in (20, 60, 300):
            ema_w = diff_ticks.ewm(span=w, adjust=False).mean()
            std_w = diff_ticks.rolling(w, min_periods=5).std().fillna(0.0)
            new_cols[f"arb_spread_zscore_{w}s"] = (diff_ticks - ema_w) / (std_w + 0.25)

        for w in (5, 20, 60, 300):
            if f"BO_ret_{w}s_ticks" in merged.columns and f"ST_ret_{w}s_ticks" in merged.columns:
                new_cols[f"leadlag_bo_minus_st_{w}s_ticks"] = (
                    merged[f"BO_ret_{w}s_ticks"] - merged[f"ST_ret_{w}s_ticks"]
                )
            if f"BO_ofi_norm_{w}s" in merged.columns and f"ST_ofi_norm_{w}s" in merged.columns:
                new_cols[f"leadlag_bo_minus_st_ofi_{w}s"] = (
                    merged[f"BO_ofi_norm_{w}s"] - merged[f"ST_ofi_norm_{w}s"]
                )

        if "BO_ret_60s_ticks" in merged.columns and "ST_ret_60s_ticks" in merged.columns:
            new_cols["leadlag_accum_bo_st_60s"] = (merged["BO_ret_60s_ticks"] - merged["ST_ret_60s_ticks"]).rolling(60, min_periods=5).mean().fillna(0.0)
        if "BO_ret_300s_ticks" in merged.columns and "ST_ret_300s_ticks" in merged.columns:
            new_cols["leadlag_accum_bo_st_300s"] = (merged["BO_ret_300s_ticks"] - merged["ST_ret_300s_ticks"]).rolling(300, min_periods=10).mean().fillna(0.0)

        # Кросс-биржевые триггеры импульса с BigONE и токсичность потока
        bo_vol = merged["BO_trade_vol"].astype(float) if "BO_trade_vol" in merged.columns else pd.Series(0.0, index=merged.index)
        bo_vol_5s = bo_vol.rolling(5, min_periods=1).mean()
        bo_vol_60s = bo_vol.rolling(60, min_periods=1).mean()
        new_cols["bo_volume_surge_5_60"] = (bo_vol_5s / (bo_vol_60s + 0.05)).fillna(0.0)

        if "BO_cvd_norm_20s" in merged.columns and "ST_cvd_norm_20s" in merged.columns:
            new_cols["cross_cvd_delta_20s"] = merged["BO_cvd_norm_20s"] - merged["ST_cvd_norm_20s"]

        if "BO_cvd_norm_5s" in merged.columns and "ST_spread_ticks" in merged.columns:
            new_cols["cross_flow_toxicity_5s"] = merged["BO_cvd_norm_5s"] * merged["ST_spread_ticks"]

        if "BO_imbalance_top5" in merged.columns and "ST_imbalance_top5" in merged.columns:
            new_cols["cross_orderbook_skew_diff"] = merged["BO_imbalance_top5"] - merged["ST_imbalance_top5"]

        if "BO_bid_1_price" in merged.columns and "ST_ask_1_price" in merged.columns and "ST_asks_usd_top5" in merged.columns:
            arb_spread = np.maximum(0.0, merged["BO_bid_1_price"].astype(float) - merged["ST_ask_1_price"].astype(float))
            new_cols["arb_exhaustion_depth_usd"] = np.where(arb_spread > 0, merged["ST_asks_usd_top5"].astype(float), 0.0)

    if "RHEA_price_usd" in merged.columns and "ST_mid_price" in merged.columns:
        rhea_diff_ticks = (merged["RHEA_price_usd"] - merged["ST_mid_price"]) / tick
        new_cols["arb_rhea_diff_ticks"] = rhea_diff_ticks
        for w in (5, 20, 60, 300):
            new_cols[f"RHEA_ret_{w}s_ticks"] = (
                merged["RHEA_price_usd"] - merged["RHEA_price_usd"].shift(w)
            ).fillna(0.0) / tick
        for w in (60, 300):
            ema_r = rhea_diff_ticks.ewm(span=w, adjust=False).mean()
            std_r = rhea_diff_ticks.rolling(w, min_periods=5).std().fillna(0.0)
            new_cols[f"arb_rhea_zscore_{w}s"] = (rhea_diff_ticks - ema_r) / (std_r + 0.25)

    # Многомасштабные доходности макро-поводырей (в базисных пунктах bps = 1e4 * pct)
    for m_asset in ("btc", "near"):
        col = f"macro_{m_asset}_mid"
        if col in merged.columns:
            p_m = merged[col].astype(float).replace(0, np.nan)
            for w in FILTER_WINDOWS:
                new_cols[f"macro_{m_asset}_ret_{w}s_bps"] = ((p_m / p_m.shift(w) - 1.0) * 10000.0).fillna(0.0)
            new_cols[f"macro_{m_asset}_band_5_20_bps"] = (
                new_cols[f"macro_{m_asset}_ret_5s_bps"] - new_cols[f"macro_{m_asset}_ret_20s_bps"] / 4.0
            )

    # Направленное согласие с макро-поводырем (Directional Alignment с BTC / NEAR)
    if "ST_ret_20s_ticks" in merged.columns:
        st_ret = merged["ST_ret_20s_ticks"]
        if isinstance(st_ret, pd.DataFrame):
            st_ret = st_ret.iloc[:, 0]
        if "macro_btc_ret_20s_bps" in new_cols:
            new_cols["macro_btc_alignment_20s"] = (np.sign(st_ret) * np.sign(new_cols["macro_btc_ret_20s_bps"])).fillna(0.0)
        elif "macro_btc_ret_20s_bps" in merged.columns:
            m_ret = merged["macro_btc_ret_20s_bps"]
            if isinstance(m_ret, pd.DataFrame):
                m_ret = m_ret.iloc[:, 0]
            new_cols["macro_btc_alignment_20s"] = (np.sign(st_ret) * np.sign(m_ret)).fillna(0.0)

        if "macro_near_ret_20s_bps" in new_cols:
            new_cols["macro_near_alignment_20s"] = (np.sign(st_ret) * np.sign(new_cols["macro_near_ret_20s_bps"])).fillna(0.0)
        elif "macro_near_ret_20s_bps" in merged.columns:
            m_ret = merged["macro_near_ret_20s_bps"]
            if isinstance(m_ret, pd.DataFrame):
                m_ret = m_ret.iloc[:, 0]
            new_cols["macro_near_alignment_20s"] = (np.sign(st_ret) * np.sign(m_ret)).fillna(0.0)

    # Внутрисуточная цикличность ликвидности (Intraday Cyclics)
    hour = pd.Series(merged.index.hour + merged.index.minute / 60.0, index=merged.index)
    new_cols["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    new_cols["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # Непрерывный композитный индекс рыночного режима (Market Activity Regime)
    if "ST_vol_intensity_60s" in merged.columns and "ST_spread_ticks" in merged.columns:
        vol_int = merged["ST_vol_intensity_60s"]
        if isinstance(vol_int, pd.DataFrame):
            vol_int = vol_int.iloc[:, 0]
        sp_ticks = merged["ST_spread_ticks"]
        if isinstance(sp_ticks, pd.DataFrame):
            sp_ticks = sp_ticks.iloc[:, 0]
        act_raw = (vol_int.astype(float) / (sp_ticks.astype(float) + 1.0)).clip(0.0, 10.0)
        new_cols["market_activity_regime"] = 2.0 / (1.0 + np.exp(-0.5 * act_raw)) - 1.0

    # Динамика потока сделок SafeTrade (Flow Regime & Stale Book Detection)
    if "ST_trade_count" in merged.columns or "ST_trade_vol" in merged.columns:
        tc = merged["ST_trade_count"].astype(float) if "ST_trade_count" in merged.columns else pd.Series(0.0, index=merged.index)
        tv = merged["ST_trade_vol"].astype(float) if "ST_trade_vol" in merged.columns else pd.Series(0.0, index=merged.index)
        has_trade = (tc > 0) | (tv > 0)
        trade_grp = has_trade.cumsum()
        tslt = merged.groupby(trade_grp).cumcount().astype(float).clip(upper=600.0)
        new_cols["ST_time_since_last_trade"] = tslt
        new_cols["ST_flow_regime"] = np.where(tslt <= 30.0, 1.0, 0.0)
        new_cols["ST_trades_burst_10s"] = has_trade.astype(float).rolling(10, min_periods=1).sum().fillna(0.0)

    # Макро-согласие на старшем горизонте 300с
    if "ST_ret_300s_ticks" in merged.columns and "macro_btc_ret_300s_bps" in new_cols:
        st_ret_300 = merged["ST_ret_300s_ticks"]
        if isinstance(st_ret_300, pd.DataFrame):
            st_ret_300 = st_ret_300.iloc[:, 0]
        new_cols["macro_btc_alignment_300s"] = (np.sign(st_ret_300) * np.sign(new_cols["macro_btc_ret_300s_bps"])).fillna(0.0)

    # Стационарные ончейн-признаки: время с момента последнего блока (в секундах)
    if "onchain_block_height" in merged.columns:
        bh = merged["onchain_block_height"]
        if isinstance(bh, pd.DataFrame):
            bh = bh.iloc[:, 0]
        block_changed = (bh != bh.shift(1)).cumsum()
        new_cols["onchain_seconds_since_block"] = merged.groupby(block_changed).cumcount().astype(float)

    whale_col = "onchain_whale_volume_5k" if "onchain_whale_volume_5k" in merged.columns else (
        "onchain_whale_transfers_5m" if "onchain_whale_transfers_5m" in merged.columns else None
    )
    if whale_col:
        wv = merged[whale_col].astype(float)
        new_cols["onchain_whale_surge_300s"] = (wv - wv.shift(300)).clip(lower=0.0).fillna(0.0)

    mp_col = "onchain_mempool_size" if "onchain_mempool_size" in merged.columns else (
        "onchain_mempool_pending_count" if "onchain_mempool_pending_count" in merged.columns else None
    )
    if mp_col:
        mp = merged[mp_col]
        if isinstance(mp, pd.DataFrame):
            mp = mp.iloc[:, 0]
        mp = mp.astype(float)
        new_cols["onchain_mempool_delta_20s"] = (mp - mp.shift(20)).fillna(0.0)
        new_cols["onchain_mempool_delta_60s"] = (mp - mp.shift(60)).fillna(0.0)

    if "onchain_total_transfers" in merged.columns:
        tt = merged["onchain_total_transfers"]
        if isinstance(tt, pd.DataFrame):
            tt = tt.iloc[:, 0]
        tt = tt.astype(float)
        new_cols["onchain_transfers_delta_300s"] = (tt - tt.shift(300)).fillna(0.0)

    if "onchain_total_miners" in merged.columns:
        tm = merged["onchain_total_miners"]
        if isinstance(tm, pd.DataFrame):
            tm = tm.iloc[:, 0]
        tm = tm.astype(float)
        new_cols["onchain_miners_delta_300s"] = (tm - tm.shift(300)).fillna(0.0)

    ext_df = pd.DataFrame(new_cols, index=merged.index).ffill().fillna(0.0)
    overlap = [c for c in ext_df.columns if c in merged.columns]
    if overlap:
        merged = merged.drop(columns=overlap)
    out = pd.concat([merged, ext_df], axis=1)

    # Удаляем нестационарные монотонно растущие счётчики (защита от переобучения на номера блоков и даты)
    non_stationary = [
        c for c in [
            "ST_cvd_total", "BO_cvd_total", "onchain_total_accounts", "onchain_total_transactions",
            "onchain_total_transfers", "onchain_block_height", "onchain_total_miners"
        ]
        if c in out.columns
    ]
    if non_stationary:
        out = out.drop(columns=non_stationary)

    return out

