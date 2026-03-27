from __future__ import annotations

import pandas as pd

from src.features.address_features import add_address_features
from src.features.gas_features import add_gas_features
from src.features.timing_features import add_timing_features


def build_event_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Event features from swaps
    """
    df = dataframe.copy()

    df = df.sort_values(["pool_address", "block_number", "log_index", "swap_id"]).reset_index(drop=True)

    df = _add_trade_magnitude_features(df)
    df = _add_price_state_features(df)
    df = add_timing_features(df)
    df = add_gas_features(df)
    df = add_address_features(df)
    df = _add_rule_support_features(df)

    return df


def _add_trade_magnitude_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Basic trade magnitude and direction features.
    """
    df = dataframe.copy()

    if "amount0" in df.columns:
        df["swap_size_token0"] = df["amount0"].abs()

    if "amount1" in df.columns:
        df["swap_size_token1"] = df["amount1"].abs()

    if "amount0" in df.columns and "amount1" in df.columns:
        df["trade_direction"] = df.apply(_derive_trade_direction, axis=1)

    return df


def _add_price_state_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-state and local movement proxy features.
    """
    df = dataframe.copy()

    if "tick" in df.columns:
        df["tick_change_from_previous"] = (
            df.groupby("pool_address")["tick"].diff()
        )
        df["abs_tick_change"] = df["tick_change_from_previous"].abs()

    if "sqrt_price_x96_raw" in df.columns:
        sqrt_price_numeric = pd.to_numeric(df["sqrt_price_x96_raw"], errors="coerce")
        df["sqrt_price_x96"] = sqrt_price_numeric

        df["sqrt_price_change_from_previous"] = (
            df.groupby("pool_address")["sqrt_price_x96"].diff()
        )

        df["relative_sqrt_price_change"] = (
            df["sqrt_price_change_from_previous"] /
            df.groupby("pool_address")["sqrt_price_x96"].shift(1)
        )

    return df


def _add_rule_support_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple rule support flags that may later help weak labeling.
    """
    df = dataframe.copy()

    if "same_block_event_count" in df.columns and "abs_tick_change" in df.columns:
        df["burst_activity_flag"] = (
            (df["same_block_event_count"].fillna(1) > 2) |
            (df["abs_tick_change"].fillna(0) > 0)
        ).astype(int)

    if "gas_spike_flag" not in df.columns:
        df["gas_spike_flag"] = 0

    if "same_block_pattern_flag" not in df.columns:
        df["same_block_pattern_flag"] = 0

    return df


def _derive_trade_direction(row: pd.Series) -> str:
    """
    Derive a interpretable trade direction label.
    """
    amount0 = row.get("amount0")
    amount1 = row.get("amount1")

    if pd.isna(amount0) or pd.isna(amount1):
        return "other"

    if amount0 > 0 and amount1 < 0:
        return "token0_in_token1_out"

    if amount0 < 0 and amount1 > 0:
        return "token1_in_token0_out"

    return "other"