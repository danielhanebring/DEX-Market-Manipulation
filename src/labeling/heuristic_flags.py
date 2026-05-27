from __future__ import annotations

import pandas as pd


def add_heuristic_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple rule flags used for weak labels.
    """
    df = dataframe.copy()

    df["flag_high_gas_context"] = _flag_high_gas_context(df)
    df["flag_same_block_pattern"] = _flag_same_block_pattern(df)
    df["flag_repeated_address_pattern"] = _flag_repeated_address_pattern(df)
    df["flag_large_price_movement"] = _flag_large_price_movement(df)
    df["flag_large_trade_size"] = _flag_large_trade_size(df)
    df["flag_burst_activity"] = _flag_burst_activity(df)

    return df


def _flag_high_gas_context(dataframe: pd.DataFrame) -> pd.Series:
    if "gas_price_relative_to_local_median" not in dataframe.columns:
        return pd.Series(0, index=dataframe.index, dtype="int64")

    return (dataframe["gas_price_relative_to_local_median"].fillna(1.0) >= 1.5).astype(int)


def _flag_same_block_pattern(dataframe: pd.DataFrame) -> pd.Series:
    if "same_block_event_count" not in dataframe.columns:
        return pd.Series(0, index=dataframe.index, dtype="int64")

    return (dataframe["same_block_event_count"].fillna(1) >= 3).astype(int)


def _flag_repeated_address_pattern(dataframe: pd.DataFrame) -> pd.Series:
    sender_count = dataframe.get("same_sender_recent_count_20")
    recipient_count = dataframe.get("same_recipient_recent_count_20")
    pair_count = dataframe.get("sender_recipient_pair_recent_count_20")

    if sender_count is None and recipient_count is None and pair_count is None:
        return pd.Series(0, index=dataframe.index, dtype="int64")

    sender_signal = sender_count.fillna(0) >= 3 if sender_count is not None else False
    recipient_signal = recipient_count.fillna(0) >= 3 if recipient_count is not None else False
    pair_signal = pair_count.fillna(0) >= 2 if pair_count is not None else False

    return (sender_signal | recipient_signal | pair_signal).astype(int)


def _flag_large_price_movement(dataframe: pd.DataFrame) -> pd.Series:
    if "abs_tick_change" not in dataframe.columns:
        return pd.Series(0, index=dataframe.index, dtype="int64")

    threshold = dataframe["abs_tick_change"].quantile(0.99)
    if pd.isna(threshold):
        threshold = 0

    return (dataframe["abs_tick_change"].fillna(0) >= threshold).astype(int)


def _flag_large_trade_size(dataframe: pd.DataFrame) -> pd.Series:
    size_col = None
    if "swap_size_token0" in dataframe.columns:
        size_col = "swap_size_token0"
    elif "abs_amount0" in dataframe.columns:
        size_col = "abs_amount0"

    if size_col is None:
        return pd.Series(0, index=dataframe.index, dtype="int64")

    threshold = dataframe[size_col].quantile(0.99)
    if pd.isna(threshold):
        threshold = 0

    return (dataframe[size_col].fillna(0) >= threshold).astype(int)


def _flag_burst_activity(dataframe: pd.DataFrame) -> pd.Series:
    if "local_event_density_10" not in dataframe.columns:
        return pd.Series(0, index=dataframe.index, dtype="int64")

    threshold = dataframe["local_event_density_10"].quantile(0.99)
    if pd.isna(threshold):
        threshold = 0

    return (dataframe["local_event_density_10"].fillna(0) >= threshold).astype(int)
