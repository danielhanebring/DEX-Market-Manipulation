from __future__ import annotations

import pandas as pd

from src.features.address_features import add_address_features
from src.features.gas_features import add_gas_features
from src.features.sandwich_features import add_sandwich_features
from src.features.timing_features import add_timing_features


def build_event_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from processed swaps.
    - Trade magnitude
    - Local movement
    - TIming
    - Gas context
    - Adress repetition
    - Sandwich features
    """
    event_dataframe = dataframe.copy()

    event_dataframe = event_dataframe.sort_values(
        ["pool_address", "block_number", "log_index", "swap_id"]
    ).reset_index(drop=True)

    event_dataframe = _add_trade_magnitude_features(event_dataframe)
    event_dataframe = _add_price_state_features(event_dataframe)
    event_dataframe = add_timing_features(event_dataframe)
    event_dataframe = add_gas_features(event_dataframe)
    event_dataframe = add_address_features(event_dataframe)
    event_dataframe = _add_rule_support_features(event_dataframe)
    event_dataframe = add_sandwich_features(event_dataframe)

    return event_dataframe


def _add_trade_magnitude_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic trade magnitude and direction features.
    """
    event_dataframe = dataframe.copy()

    if "amount0" in event_dataframe.columns:
        event_dataframe["swap_size_token0"] = event_dataframe["amount0"].abs()

    if "amount1" in event_dataframe.columns:
        event_dataframe["swap_size_token1"] = event_dataframe["amount1"].abs()

    if "amount0" in event_dataframe.columns and "amount1" in event_dataframe.columns:
        event_dataframe["trade_direction"] = event_dataframe.apply(
            _derive_trade_direction,
            axis=1,
        )

    return event_dataframe


def _add_price_state_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    event_dataframe = dataframe.copy()

    if "tick" in event_dataframe.columns:
        event_dataframe["tick_change_from_previous"] = (
            event_dataframe.groupby("pool_address")["tick"].diff()
        )
        event_dataframe["abs_tick_change"] = event_dataframe["tick_change_from_previous"].abs()

    if "sqrt_price_x96_raw" in event_dataframe.columns:
        sqrt_price_numeric = pd.to_numeric(
            event_dataframe["sqrt_price_x96_raw"],
            errors="coerce",
        )
        event_dataframe["sqrt_price_x96"] = sqrt_price_numeric

        event_dataframe["sqrt_price_change_from_previous"] = (
            event_dataframe.groupby("pool_address")["sqrt_price_x96"].diff()
        )

        event_dataframe["relative_sqrt_price_change"] = (
            event_dataframe["sqrt_price_change_from_previous"]
            / event_dataframe.groupby("pool_address")["sqrt_price_x96"].shift(1)
        )

    return event_dataframe


def _add_rule_support_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    event_dataframe = dataframe.copy()

    if "same_block_event_count" in event_dataframe.columns and "abs_tick_change" in event_dataframe.columns:
        event_dataframe["burst_activity_flag"] = (
            (event_dataframe["same_block_event_count"].fillna(1) > 2)
            | (event_dataframe["abs_tick_change"].fillna(0) > 0)
        ).astype(int)

    if "gas_spike_flag" not in event_dataframe.columns:
        event_dataframe["gas_spike_flag"] = 0

    if "same_block_pattern_flag" not in event_dataframe.columns:
        event_dataframe["same_block_pattern_flag"] = 0

    return event_dataframe


def _derive_trade_direction(row: pd.Series) -> str:
    amount0 = row.get("amount0")
    amount1 = row.get("amount1")

    if pd.isna(amount0) or pd.isna(amount1):
        return "other"

    if amount0 > 0 and amount1 < 0:
        return "token0_in_token1_out"

    if amount0 < 0 and amount1 > 0:
        return "token1_in_token0_out"

    return "other"