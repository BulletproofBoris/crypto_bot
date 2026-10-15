"""
Модуль генерации стационарных микроструктурных признаков и многомасштабного банка фильтров (6 блоков).
Окна банка фильтров: W = [3, 8, 20, 60, 300] секунд (синхронизированы с тактическими горизонтами H = 3, 8, 25с).
"""
import numpy as np
import pandas as pd

FILTER_WINDOWS = (3, 8, 20, 60, 300)

TICK_SIZES = {
    "prl": 0.005,
    "quantus": 0.05,
}


def compute_ofi_1s(df_1s, prefix):
    """
    Рассчитывает 1-секундный Order Flow Imbalance (Cont-Kukanov-Stoikov OFI):
    - ofi_bbo: по 1-му уровню (bid_1, ask_1)
    - ofi_deep5: взвешенный по уровням 1..5 с весами exp(-0.5*(k-1))
    """
    weights = [1.0, 0.6065, 0.3679, 0.2231, 0.1353]
    ofi_bbo = None
    ofi_deep = np.zeros(len(df_1s), dtype=np.float32)

    for k in range(1, 6):
        bp_col, bq_col = f"{prefix}_bid_{k}_price", f"{prefix}_bid_{k}_qty"
        ap_col, aq_col = f"{prefix}_ask_{k}_price", f"{prefix}_ask_{k}_qty"
        if bp_col not in df_1s.columns or ap_col not in df_1s.columns:
            continue

        b_p = df_1s[bp_col].to_numpy(dtype=np.float32)
        b_q = np.nan_to_num(df_1s[bq_col].to_numpy(dtype=np.float32), nan=0.0)
        a_p = df_1s[ap_col].to_numpy(dtype=np.float32)
        a_q = np.nan_to_num(df_1s[aq_col].to_numpy(dtype=np.float32), nan=0.0)

        b_p_prev = np.roll(b_p, 1); b_p_prev[0] = b_p[0]
        b_q_prev = np.roll(b_q, 1); b_q_prev[0] = b_q[0]
        a_p_prev = np.roll(a_p, 1); a_p_prev[0] = a_p[0]
        a_q_prev = np.roll(a_q, 1); a_q_prev[0] = a_q[0]

        e_bid = np.where(b_p >= b_p_prev, b_q, 0.0) - np.where(b_p <= b_p_prev, b_q_prev, 0.0)
        e_ask = np.where(a_p <= a_p_prev, a_q, 0.0) - np.where(a_p >= a_p_prev, a_q_prev, 0.0)
        ofi_k = (e_bid - e_ask).astype(np.float32)

        if k == 1:
            ofi_bbo = pd.Series(ofi_k, index=df_1s.index)
        ofi_deep += weights[k - 1] * ofi_k

    if ofi_bbo is None:
        ofi_bbo = pd.Series(0.0, index=df_1s.index, dtype=np.float32)
    return ofi_bbo, pd.Series(ofi_deep, index=df_1s.index)


def compute_price_bucket_depth(df_1s, prefix, tick):
    """
    Блок 1: Плотность ликвидности в фиксированных ценовых коридорах (2, 5, 15, 50 тиков от mid_price)
    и центр масс топ-10 уровней стакана.
    """
    mid = df_1s[f"{prefix}_mid_price"].to_numpy(dtype=np.float32)[:, None]
    max_lvl = 20
    bp_cols = [f"{prefix}_bid_{i}_price" for i in range(1, max_lvl + 1) if f"{prefix}_bid_{i}_price" in df_1s.columns]
    bq_cols = [f"{prefix}_bid_{i}_qty" for i in range(1, max_lvl + 1) if f"{prefix}_bid_{i}_qty" in df_1s.columns]
    ap_cols = [f"{prefix}_ask_{i}_price" for i in range(1, max_lvl + 1) if f"{prefix}_ask_{i}_price" in df_1s.columns]
    aq_cols = [f"{prefix}_ask_{i}_qty" for i in range(1, max_lvl + 1) if f"{prefix}_ask_{i}_qty" in df_1s.columns]

    out = {}
    if not bp_cols or not ap_cols:
        return out

    bp = np.nan_to_num(df_1s[bp_cols].to_numpy(dtype=np.float32), nan=0.0)
    bq = np.nan_to_num(df_1s[bq_cols].to_numpy(dtype=np.float32), nan=0.0)
    ap = np.nan_to_num(df_1s[ap_cols].to_numpy(dtype=np.float32), nan=0.0)
    aq = np.nan_to_num(df_1s[aq_cols].to_numpy(dtype=np.float32), nan=0.0)

    bid_dist_ticks = np.where(bp > 0, (mid - bp) / tick, 9999.0)
    ask_dist_ticks = np.where(ap > 0, (ap - mid) / tick, 9999.0)

    b5 = df_1s[f"{prefix}_bids_vol_top5"].astype(np.float32)
    a5 = df_1s[f"{prefix}_asks_vol_top5"].astype(np.float32)
    norm_scale = (b5 + a5).rolling(60, min_periods=1).median().replace(0, np.nan).ffill().fillna(1.0).to_numpy(dtype=np.float32)

    for rad in (2, 5, 15, 50):
        b_vol = np.where(bid_dist_ticks <= rad, bq, 0.0).sum(axis=1)
        a_vol = np.where(ask_dist_ticks <= rad, aq, 0.0).sum(axis=1)
        tot = b_vol + a_vol
        out[f"{prefix}_bucket_bid_norm_{rad}t"] = b_vol / norm_scale
        out[f"{prefix}_bucket_ask_norm_{rad}t"] = a_vol / norm_scale
        out[f"{prefix}_bucket_imb_{rad}t"] = np.where(tot > 0, (b_vol - a_vol) / tot, 0.0)

    # Центр масс топ-10 уровней (в тиках от mid) и наклон книги (Book Slope)
    k10 = min(10, bp.shape[1])
    b10_sum = np.maximum(bq[:, :k10].sum(axis=1), 1e-6)
    a10_sum = np.maximum(aq[:, :k10].sum(axis=1), 1e-6)
    com_bid = (np.clip(bid_dist_ticks[:, :k10], 0, 500) * bq[:, :k10]).sum(axis=1) / b10_sum
    com_ask = (np.clip(ask_dist_ticks[:, :k10], 0, 500) * aq[:, :k10]).sum(axis=1) / a10_sum

    out[f"{prefix}_com_bid_top10_ticks"] = com_bid
    out[f"{prefix}_com_ask_top10_ticks"] = com_ask
    out[f"{prefix}_com_asymmetry_ticks"] = com_ask - com_bid

    slope_bid = b10_sum / np.maximum(com_bid, 0.5) / norm_scale
    slope_ask = a10_sum / np.maximum(com_ask, 0.5) / norm_scale
    out[f"{prefix}_book_slope_bid"] = slope_bid
    out[f"{prefix}_book_slope_ask"] = slope_ask
    out[f"{prefix}_book_slope_imb"] = (slope_bid - slope_ask) / np.maximum(slope_bid + slope_ask, 1e-6)
    return out
