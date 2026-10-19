"""
Модуль очистки и нормализации сырых рыночных и ончейн-данных при сборке Parquet.
"""
import os
import csv
import gzip
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("sync_all_data")

PRL_ONCHAIN_COLS = [
    "timestamp", "block_height", "mempool_size", "indexer_lag", "difficulty",
    "reward_grains", "reward_prl", "tx_count", "transfer_count",
    "transfer_volume_prl", "max_tx_prl", "whale_tx_count_5k",
    "whale_volume_5k", "whale_tx_count_25k", "whale_volume_25k",
    "unique_receivers", "miner_pool", "miner_address"
]

PRL_SCHEMA_12 = [
    "timestamp", "block_height", "difficulty", "reward_prl", "tx_count",
    "transfer_count", "transfer_volume_prl", "max_tx_prl", "whale_tx_count_5k",
    "whale_volume_5k", "miner_pool", "miner_address"
]

PRL_SCHEMA_8 = [
    "timestamp", "block_height", "mempool_size", "indexer_lag",
    "difficulty", "reward_grains", "tx_count", "miner_pool"
]

PRL_SCHEMA_4 = [
    "timestamp", "block_height", "mempool_size", "indexer_lag"
]

PRL_SCHEMA_15 = [
    "timestamp", "block_height", "mempool_size", "indexer_lag", "difficulty",
    "reward_grains", "reward_prl", "tx_count", "transfer_count",
    "transfer_volume_prl", "max_tx_prl", "whale_tx_count_5k",
    "whale_volume_5k", "miner_pool", "miner_address"
]


def read_csv_flexible(filepath, key):
    """Безопасное чтение CSV/CSV.GZ с поддержкой эволюции схем колонок без потери строк."""
    if key != "prl_onchain":
        try:
            return pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)
        except Exception:
            pass

    opener = gzip.open if filepath.endswith(".gz") else open
    if key != "prl_onchain":
        with opener(filepath, "rt", encoding="utf-8", errors="ignore") as f:
            reader = list(csv.reader(f))
        if not reader:
            return pd.DataFrame()
        header = reader[0]
        max_len = max(len(r) for r in reader)
        if max_len > len(header):
            header = header + [f"extra_{i}" for i in range(len(header), max_len)]
        rows = [r + [""] * (len(header) - len(r)) for r in reader[1:] if r and r[0] != "timestamp"]
        return pd.DataFrame(rows, columns=header)

    rows = []
    with opener(filepath, "rt", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0] == "timestamp":
                header = row
                continue
            n = len(row)
            if n == 18:
                schema = PRL_ONCHAIN_COLS
            elif n == 15:
                schema = PRL_SCHEMA_15
            elif n == 12:
                schema = PRL_SCHEMA_12
            elif n == 8:
                schema = PRL_SCHEMA_8
            elif n == 4:
                schema = PRL_SCHEMA_4
            elif header and len(header) == n:
                schema = header
            else:
                schema = PRL_ONCHAIN_COLS[:n]
            row_dict = {schema[i]: row[i] for i in range(min(len(schema), n))}
            rows.append([row_dict.get(c, "") for c in PRL_ONCHAIN_COLS])

    if not rows:
        return pd.DataFrame(columns=PRL_ONCHAIN_COLS)
    df = pd.DataFrame(rows, columns=PRL_ONCHAIN_COLS)
    for col in PRL_ONCHAIN_COLS:
        if col not in ("timestamp", "miner_pool", "miner_address"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def clean_orderbook_df(df, key):
    """Удаляет строки без BBO, аномалии холодного старта и сдвигает залипшие пересечённые уровни."""
    if df.empty or "bid_1_price" not in df.columns or "ask_1_price" not in df.columns:
        return df

    n_before = len(df)
    df = df.dropna(subset=["bid_1_price", "ask_1_price"]).copy()
    df = df[(df["bid_1_price"] > 0) & (df["ask_1_price"] > 0)].copy()

    if "bid_2_price" in df.columns and "ask_2_price" in df.columns:
        rel_spread = (df["ask_1_price"] - df["bid_1_price"]) / df["bid_1_price"]
        cold_spike = (df["bid_2_price"].isna() | df["ask_2_price"].isna()) & (rel_spread > 0.25)
        df = df[~cold_spike].copy()

    crossed_mask = df["ask_1_price"] <= df["bid_1_price"]
    if crossed_mask.any():
        ask_lvl_indices = [
            int(c.split("_")[1]) for c in df.columns
            if c.startswith("ask_") and c.endswith("_price") and c.split("_")[1].isdigit()
        ]
        max_lvl = max(ask_lvl_indices) if ask_lvl_indices else 50
        ask_p_cols = [f"ask_{i}_price" for i in range(1, max_lvl + 1) if f"ask_{i}_price" in df.columns]
        ask_q_cols = [f"ask_{i}_qty" for i in range(1, max_lvl + 1) if f"ask_{i}_qty" in df.columns]

        crossed_idx = df.index[crossed_mask]
        ask_p_vals = df.loc[crossed_idx, ask_p_cols].to_numpy(dtype=float)
        ask_q_vals = df.loc[crossed_idx, ask_q_cols].to_numpy(dtype=float)
        bid1_vals = df.loc[crossed_idx, "bid_1_price"].to_numpy(dtype=float)

        new_p = np.full_like(ask_p_vals, np.nan)
        new_q = np.full_like(ask_q_vals, np.nan)
        m_levels = len(ask_p_cols)

        for r_i in range(len(crossed_idx)):
            b1 = bid1_vals[r_i]
            valid = np.where(ask_p_vals[r_i] > b1)[0]
            if len(valid) > 0:
                k = valid[0]
                rem = m_levels - k
                new_p[r_i, :rem] = ask_p_vals[r_i, k:]
                new_q[r_i, :rem] = ask_q_vals[r_i, k:]

        df.loc[crossed_idx, ask_p_cols] = new_p
        df.loc[crossed_idx, ask_q_cols] = new_q
        df = df.dropna(subset=["bid_1_price", "ask_1_price"]).copy()
        df = df[df["ask_1_price"] > df["bid_1_price"]].copy()

    n_after = len(df)
    if n_before != n_after:
        logger.info(f"🧹 {key}: очищено {n_before - n_after} некорректных/пересечённых строк стакана.")
    return df.reset_index(drop=True)


def clean_trades_df(df, key, processed_dir):
    """Восстанавливает пропущенное направление сделок (type: buy/sell) для SafeTrade по стакану и тик-правилу."""
    if df.empty or "type" not in df.columns:
        return df

    df["type"] = df["type"].astype(str).str.strip().str.lower()
    missing_mask = ~df["type"].isin(["buy", "sell", "bid", "ask"])
    if not missing_mask.any():
        return df

    depth_key = "safetrade_depth" if key == "safetrade_trades" else "safetrade_quantus_depth"
    depth_path = os.path.join(processed_dir, f"{depth_key}.parquet")

    inferred = pd.Series(index=df.index, dtype="object")
    if os.path.exists(depth_path):
        try:
            d_df = pd.read_parquet(depth_path, columns=["timestamp", "bid_1_price", "ask_1_price"]).sort_values("timestamp")
            d_df["timestamp"] = d_df["timestamp"].astype("datetime64[us, UTC]")
            t_sub = df[["timestamp", "price"]].copy()
            t_sub["timestamp"] = t_sub["timestamp"].astype("datetime64[us, UTC]")
            t_sub = t_sub.sort_values("timestamp")
            merged = pd.merge_asof(
                t_sub,
                d_df,
                on="timestamp",
                direction="backward"
            )
            mid = 0.5 * (merged["bid_1_price"] + merged["ask_1_price"])
            side = np.where(merged["price"] > mid, "buy", np.where(merged["price"] < mid, "sell", None))
            inferred.loc[merged.index] = side
        except Exception as e:
            logger.warning(f"Не удалось восстановить type по стакану для {key}: {e}")

    # Fallback: tick rule
    price_diff = df["price"].diff()
    tick_dir = np.where(price_diff > 0, "buy", np.where(price_diff < 0, "sell", None))
    tick_series = pd.Series(tick_dir, index=df.index).ffill().bfill().fillna("buy")

    fill_vals = inferred.combine_first(tick_series)
    df.loc[missing_mask, "type"] = fill_vals.loc[missing_mask]
    logger.info(f"🧹 {key}: восстановлено направление (buy/sell) для {missing_mask.sum()} сделок.")
    return df


def clean_onchain_df(df, key):
    """Унифицирует схему ончейн-датасетов PRL и Quantus и заполняет пропуски при смене версий."""
    if df.empty:
        return df

    if key == "quantus_onchain":
        if "accounts_count" in df.columns:
            if "total_accounts" in df.columns:
                df["total_accounts"] = df["total_accounts"].combine_first(df["accounts_count"])
            else:
                df["total_accounts"] = df["accounts_count"]
            df = df.drop(columns=["accounts_count"])
        for c in df.columns:
            if c not in ("timestamp", "date"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        state_cols = [c for c in ["block_height", "circulating_supply", "total_supply", "total_miners", "total_accounts", "total_transfers"] if c in df.columns]
        if state_cols:
            df[state_cols] = df[state_cols].ffill().bfill()
        return df

    if key == "prl_onchain":
        for c in PRL_ONCHAIN_COLS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[PRL_ONCHAIN_COLS].copy()
        for c in PRL_ONCHAIN_COLS:
            if c not in ("timestamp", "miner_pool", "miner_address"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Взаимное восстановление reward_prl и reward_grains
        mask_r_prl = df["reward_prl"].isna() & df["reward_grains"].notna()
        df.loc[mask_r_prl, "reward_prl"] = (df.loc[mask_r_prl, "reward_grains"] / 1e8).round(4)
        mask_r_gr = df["reward_grains"].isna() & df["reward_prl"].notna()
        df.loc[mask_r_gr, "reward_grains"] = (df.loc[mask_r_gr, "reward_prl"] * 1e8).round(0)

        block_cols = [
            "difficulty", "reward_grains", "reward_prl", "tx_count", "transfer_count",
            "transfer_volume_prl", "max_tx_prl", "whale_tx_count_5k", "whale_volume_5k",
            "whale_tx_count_25k", "whale_volume_25k", "unique_receivers", "miner_pool", "miner_address"
        ]
        if "block_height" in df.columns:
            df["block_height"] = df["block_height"].ffill().bfill()
            df[block_cols] = df.groupby("block_height")[block_cols].transform(lambda s: s.ffill().bfill())

        flow_cols = [
            "tx_count", "transfer_count", "transfer_volume_prl", "max_tx_prl",
            "whale_tx_count_5k", "whale_volume_5k", "whale_tx_count_25k", "whale_volume_25k", "unique_receivers"
        ]
        for c in flow_cols:
            df[c] = df[c].fillna(0.0)
        for c in ["mempool_size", "indexer_lag", "difficulty", "reward_grains", "reward_prl"]:
            df[c] = df[c].ffill().bfill().fillna(0.0)
        df["miner_pool"] = df["miner_pool"].replace("", np.nan).ffill().bfill().fillna("Unknown")
        df["miner_address"] = df["miner_address"].replace("", np.nan).ffill().bfill().fillna("")
    return df



def sanitize_dataset(df, key, processed_dir):
    if key in ("safetrade_depth", "safetrade_quantus_depth", "bigone_depth", "bigone_quantus_depth"):
        return clean_orderbook_df(df, key)
    if key in ("safetrade_trades", "safetrade_quantus_trades"):
        return clean_trades_df(df, key, processed_dir)
    if key in ("prl_onchain", "quantus_onchain"):
        return clean_onchain_df(df, key)
    return df
