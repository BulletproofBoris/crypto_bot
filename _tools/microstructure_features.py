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

    # Относительные расстояния уровней 2..5 от mid_price (в тиках) и их наклон (book slope)
    for k in range(2, 6):
        if f"{prefix}_bid_{k}_price" in df_1s.columns:
            bp_k = df_1s[f"{prefix}_bid_{k}_price"].astype(float)
            new_cols[f"{prefix}_bid_{k}_dist_ticks"] = np.where(bp_k > 0, (mid - bp_k) / tick, np.nan)
        if f"{prefix}_ask_{k}_price" in df_1s.columns:
            ap_k = df_1s[f"{prefix}_ask_{k}_price"].astype(float)
            new_cols[f"{prefix}_ask_{k}_dist_ticks"] = np.where(ap_k > 0, (ap_k - mid) / tick, np.nan)

    # 2. Многомасштабный банк фильтров по W = [5, 20, 60, 300]
    ofi_1s = compute_ofi_1s(df_1s, prefix)
    delta_vol = df_1s[f"{prefix}_delta_vol"].astype(float)
    trade_vol = df_1s[f"{prefix}_trade_vol"].astype(float)
    depth_scale = (b5_q + a5_q).rolling(60, min_periods=1).median().replace(0, np.nan).ffill().fillna(1.0)
    ret_1s_ticks = mid.diff().fillna(0.0) / tick

    for w in FILTER_WINDOWS:
        # Нормированная скорость изменения цены (тиков в секунду за окно W) и импульс в тиках
        new_cols[f"{prefix}_ret_{w}s_ticks"] = (mid - mid.shift(w)).fillna(0.0) / tick
        new_cols[f"{prefix}_vol_{w}s_ticks"] = ret_1s_ticks.rolling(w, min_periods=1).std().fillna(0.0)
        # Нормированный поток ордеров OFI и CVD (в долях глубины стакана в секунду)
        new_cols[f"{prefix}_ofi_norm_{w}s"] = (ofi_1s.rolling(w, min_periods=1).mean() / depth_scale).fillna(0.0)
        new_cols[f"{prefix}_cvd_norm_{w}s"] = (delta_vol.rolling(w, min_periods=1).mean() / depth_scale).fillna(0.0)
        new_cols[f"{prefix}_vol_intensity_{w}s"] = (trade_vol.rolling(w, min_periods=1).mean() / depth_scale).fillna(0.0)
        new_cols[f"{prefix}_imbalance_ema_{w}s"] = df_1s[f"{prefix}_imbalance_top5"].ewm(span=w, adjust=False).mean()

    # 3. Ортогональные полосовые фильтры (Band-Pass разности между соседними частотами)
    new_cols[f"{prefix}_ofi_band_5_20"] = new_cols[f"{prefix}_ofi_norm_5s"] - new_cols[f"{prefix}_ofi_norm_20s"]
    new_cols[f"{prefix}_ofi_band_20_60"] = new_cols[f"{prefix}_ofi_norm_20s"] - new_cols[f"{prefix}_ofi_norm_60s"]
    new_cols[f"{prefix}_ofi_band_60_300"] = new_cols[f"{prefix}_ofi_norm_60s"] - new_cols[f"{prefix}_ofi_norm_300s"]

    new_cols[f"{prefix}_imb_band_5_20"] = new_cols[f"{prefix}_imbalance_ema_5s"] - new_cols[f"{prefix}_imbalance_ema_20s"]
    new_cols[f"{prefix}_imb_band_20_60"] = new_cols[f"{prefix}_imbalance_ema_20s"] - new_cols[f"{prefix}_imbalance_ema_60s"]
    new_cols[f"{prefix}_vol_ratio_5_60"] = new_cols[f"{prefix}_vol_5s_ticks"] / (new_cols[f"{prefix}_vol_60s_ticks"] + 0.05)
    new_cols[f"{prefix}_vol_ratio_20_300"] = new_cols[f"{prefix}_vol_20s_ticks"] / (new_cols[f"{prefix}_vol_300s_ticks"] + 0.05)

    ext_df = pd.DataFrame(new_cols, index=df_1s.index).ffill().fillna(0.0)
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

        for w in (5, 20, 60):
            if f"BO_ret_{w}s_ticks" in merged.columns and f"ST_ret_{w}s_ticks" in merged.columns:
                new_cols[f"leadlag_bo_minus_st_{w}s_ticks"] = (
                    merged[f"BO_ret_{w}s_ticks"] - merged[f"ST_ret_{w}s_ticks"]
                )
            if f"BO_ofi_norm_{w}s" in merged.columns and f"ST_ofi_norm_{w}s" in merged.columns:
                new_cols[f"leadlag_bo_minus_st_ofi_{w}s"] = (
                    merged[f"BO_ofi_norm_{w}s"] - merged[f"ST_ofi_norm_{w}s"]
                )

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

    # Стационарные ончейн-признаки: время с момента последнего блока (в секундах)
    if "onchain_block_height" in merged.columns:
        bh = merged["onchain_block_height"]
        block_changed = (bh != bh.shift(1)).cumsum()
        new_cols["onchain_seconds_since_block"] = merged.groupby(block_changed).cumcount().astype(float)

    mp_col = "onchain_mempool_size" if "onchain_mempool_size" in merged.columns else (
        "onchain_mempool_pending_count" if "onchain_mempool_pending_count" in merged.columns else None
    )
    if mp_col:
        mp = merged[mp_col].astype(float)
        new_cols["onchain_mempool_delta_20s"] = (mp - mp.shift(20)).fillna(0.0)
        new_cols["onchain_mempool_delta_60s"] = (mp - mp.shift(60)).fillna(0.0)

    if "onchain_total_transfers" in merged.columns:
        tt = merged["onchain_total_transfers"].astype(float)
        new_cols["onchain_transfers_delta_300s"] = (tt - tt.shift(300)).fillna(0.0)

    if "onchain_total_miners" in merged.columns:
        tm = merged["onchain_total_miners"].astype(float)
        new_cols["onchain_miners_delta_300s"] = (tm - tm.shift(300)).fillna(0.0)

    ext_df = pd.DataFrame(new_cols, index=merged.index).ffill().fillna(0.0)
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

