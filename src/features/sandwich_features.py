from __future__ import annotations

from typing import Any

import pandas as pd


def add_sandwich_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create sandwich-related features.
    """
    required_columns = [
        "swap_id",
        "pool_address",
        "block_number",
        "timestamp",
        "sender_address",
        "tick",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for sandwich feature engineering: {missing_columns}"
        )

    event_dataframe = dataframe.copy()
    event_dataframe = _sort_for_block_pattern_features(event_dataframe)

    event_dataframe = _add_position_features(event_dataframe)
    event_dataframe = _add_three_event_structure_features(event_dataframe)
    event_dataframe = _add_block_context_features(event_dataframe)
    event_dataframe = _add_relative_trade_size_features(event_dataframe)
    event_dataframe = _add_block_gas_features(event_dataframe)
    event_dataframe = _add_sequence_pattern_flags(event_dataframe)

    return event_dataframe


def _sort_for_block_pattern_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Sort events for same-block window logic.
    """
    sorted_dataframe = dataframe.copy()

    numeric_columns = [
        "block_number",
        "timestamp",
        "log_index",
        "tick",
        "swap_size_token0",
        "swap_size_token1",
        "gas_price_gwei",
    ]
    for column in numeric_columns:
        if column in sorted_dataframe.columns:
            sorted_dataframe[column] = pd.to_numeric(
                sorted_dataframe[column],
                errors="coerce",
            )

    sort_columns = [
        column
        for column in [
            "pool_address",
            "block_number",
            "timestamp",
            "log_index",
            "swap_id",
        ]
        if column in sorted_dataframe.columns
    ]

    return (
        sorted_dataframe
        .sort_values(sort_columns, ascending=True)
        .reset_index(drop=True)
    )


def _add_position_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add event position inside each pool/block group.
    """
    event_dataframe = dataframe.copy()

    event_dataframe["position_in_block"] = (
        event_dataframe.groupby(["pool_address", "block_number"])
        .cumcount()
        .astype("Int64")
    )

    event_dataframe["block_event_count"] = (
        event_dataframe.groupby(["pool_address", "block_number"])["swap_id"]
        .transform("count")
        .astype("Int64")
    )

    return event_dataframe


def _add_three_event_structure_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add neighbor aware features for strict three-event sandwich candidates.

    Each event looks at previous event and the next even within the same pool/block

    So we treat the current row as potential victim
    """
    event_dataframe = dataframe.copy()

    grouped = event_dataframe.groupby(["pool_address", "block_number"], sort=False)

    event_dataframe["prev_sender_address"] = grouped["sender_address"].shift(1)
    event_dataframe["next_sender_address"] = grouped["sender_address"].shift(-1)

    if "origin_address" in event_dataframe.columns:
        event_dataframe["prev_origin_address"] = grouped["origin_address"].shift(1)
        event_dataframe["next_origin_address"] = grouped["origin_address"].shift(-1)
    else:
        event_dataframe["prev_origin_address"] = pd.NA
        event_dataframe["next_origin_address"] = pd.NA

    event_dataframe["prev_tick"] = grouped["tick"].shift(1)
    event_dataframe["next_tick"] = grouped["tick"].shift(-1)

    if "swap_size_token0" in event_dataframe.columns:
        event_dataframe["prev_swap_size_token0"] = grouped["swap_size_token0"].shift(1)
        event_dataframe["next_swap_size_token0"] = grouped["swap_size_token0"].shift(-1)

    if "swap_size_token1" in event_dataframe.columns:
        event_dataframe["prev_swap_size_token1"] = grouped["swap_size_token1"].shift(1)
        event_dataframe["next_swap_size_token1"] = grouped["swap_size_token1"].shift(-1)

    if "gas_price_gwei" in event_dataframe.columns:
        event_dataframe["prev_gas_price_gwei"] = grouped["gas_price_gwei"].shift(1)
        event_dataframe["next_gas_price_gwei"] = grouped["gas_price_gwei"].shift(-1)

        same_sender_before_after_condition = (
        event_dataframe["prev_sender_address"].notna()
        & event_dataframe["next_sender_address"].notna()
        & (event_dataframe["prev_sender_address"] == event_dataframe["next_sender_address"])
    )

    event_dataframe["same_sender_before_after_flag"] = (
        same_sender_before_after_condition.fillna(False).astype(int)
    )

    different_middle_sender_condition = (
        event_dataframe["sender_address"].notna()
        & event_dataframe["prev_sender_address"].notna()
        & event_dataframe["next_sender_address"].notna()
        & (event_dataframe["sender_address"] != event_dataframe["prev_sender_address"])
        & (event_dataframe["sender_address"] != event_dataframe["next_sender_address"])
    )

    event_dataframe["different_middle_sender_from_neighbors_flag"] = (
        different_middle_sender_condition.fillna(False).astype(int)
    )

    same_origin_before_after_condition = (
        event_dataframe["prev_origin_address"].notna()
        & event_dataframe["next_origin_address"].notna()
        & (event_dataframe["prev_origin_address"] == event_dataframe["next_origin_address"])
    )

    event_dataframe["same_origin_before_after_flag"] = (
        same_origin_before_after_condition.fillna(False).astype(int)
    )

    event_dataframe["tick_change_before"] = (
        event_dataframe["tick"] - event_dataframe["prev_tick"]
    )

    event_dataframe["tick_change_after"] = (
        event_dataframe["next_tick"] - event_dataframe["tick"]
    )

    reversal_pattern_condition = (
        (
            (event_dataframe["tick_change_before"] > 0)
            & (event_dataframe["tick_change_after"] < 0)
        )
        |
        (
            (event_dataframe["tick_change_before"] < 0)
            & (event_dataframe["tick_change_after"] > 0)
        )
    )

    event_dataframe["reversal_pattern_flag"] = (
        reversal_pattern_condition.fillna(False).astype(int)
    )

    three_event_pattern_condition = (
        (event_dataframe["same_sender_before_after_flag"] == 1)
        & (event_dataframe["different_middle_sender_from_neighbors_flag"] == 1)
        & (event_dataframe["reversal_pattern_flag"] == 1)
    )

    event_dataframe["three_event_pattern_indicator"] = (
        three_event_pattern_condition.fillna(False).astype(int)
    )

    return event_dataframe


def _add_block_context_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add block contextual features.
    """
    event_dataframe = dataframe.copy()

    if "swap_size_token0" in event_dataframe.columns:
        block_mean_token0 = (
            event_dataframe.groupby(["pool_address", "block_number"])["swap_size_token0"]
            .transform("mean")
        )
        event_dataframe["swap_size_token0_relative_to_block_mean"] = _safe_divide_series(
            event_dataframe["swap_size_token0"],
            block_mean_token0,
        )

    if "swap_size_token1" in event_dataframe.columns:
        block_mean_token1 = (
            event_dataframe.groupby(["pool_address", "block_number"])["swap_size_token1"]
            .transform("mean")
        )
        event_dataframe["swap_size_token1_relative_to_block_mean"] = _safe_divide_series(
            event_dataframe["swap_size_token1"],
            block_mean_token1,
        )

    if "tick_change_before" in event_dataframe.columns and "tick_change_after" in event_dataframe.columns:
        event_dataframe["abs_tick_change_before"] = event_dataframe["tick_change_before"].abs()
        event_dataframe["abs_tick_change_after"] = event_dataframe["tick_change_after"].abs()
        event_dataframe["combined_reversal_magnitude"] = (
            event_dataframe["abs_tick_change_before"].fillna(0.0)
            + event_dataframe["abs_tick_change_after"].fillna(0.0)
        )

    return event_dataframe


def _add_relative_trade_size_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    event_dataframe = dataframe.copy()

    if "prev_swap_size_token0" in event_dataframe.columns and "next_swap_size_token0" in event_dataframe.columns:
        event_dataframe["attacker_average_size_token0"] = (
            event_dataframe[["prev_swap_size_token0", "next_swap_size_token0"]]
            .mean(axis=1)
        )
        event_dataframe["relative_trade_size_token0"] = _safe_divide_series(
            event_dataframe["swap_size_token0"],
            event_dataframe["attacker_average_size_token0"],
        )
        event_dataframe["attacker_vs_victim_size_ratio_token0"] = _safe_divide_series(
            event_dataframe["attacker_average_size_token0"],
            event_dataframe["swap_size_token0"],
        )

    if "prev_swap_size_token1" in event_dataframe.columns and "next_swap_size_token1" in event_dataframe.columns:
        event_dataframe["attacker_average_size_token1"] = (
            event_dataframe[["prev_swap_size_token1", "next_swap_size_token1"]]
            .mean(axis=1)
        )
        event_dataframe["relative_trade_size_token1"] = _safe_divide_series(
            event_dataframe["swap_size_token1"],
            event_dataframe["attacker_average_size_token1"],
        )
        event_dataframe["attacker_vs_victim_size_ratio_token1"] = _safe_divide_series(
            event_dataframe["attacker_average_size_token1"],
            event_dataframe["swap_size_token1"],
        )

    return event_dataframe


def _add_block_gas_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    event_dataframe = dataframe.copy()

    if "gas_price_gwei" not in event_dataframe.columns:
        event_dataframe["gas_price_relative_to_block_mean"] = 0.0
        event_dataframe["gas_price_relative_to_block_median"] = 0.0
        event_dataframe["gas_price_relative_to_neighbors_mean"] = 0.0
        return event_dataframe

    block_mean_gas = (
        event_dataframe.groupby(["pool_address", "block_number"])["gas_price_gwei"]
        .transform("mean")
    )
    block_median_gas = (
        event_dataframe.groupby(["pool_address", "block_number"])["gas_price_gwei"]
        .transform("median")
    )

    event_dataframe["gas_price_relative_to_block_mean"] = _safe_divide_series(
        event_dataframe["gas_price_gwei"],
        block_mean_gas,
    )
    event_dataframe["gas_price_relative_to_block_median"] = _safe_divide_series(
        event_dataframe["gas_price_gwei"],
        block_median_gas,
    )

    if "prev_gas_price_gwei" in event_dataframe.columns and "next_gas_price_gwei" in event_dataframe.columns:
        neighbor_gas_mean = (
            event_dataframe[["prev_gas_price_gwei", "next_gas_price_gwei"]]
            .mean(axis=1)
        )
        event_dataframe["gas_price_relative_to_neighbors_mean"] = _safe_divide_series(
            event_dataframe["gas_price_gwei"],
            neighbor_gas_mean,
        )

    return event_dataframe


def _add_sequence_pattern_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add final helper flags that combine sandwich-specific signals.
    """
    event_dataframe = dataframe.copy()

    gas_block_median = event_dataframe.get(
        "gas_price_relative_to_block_median",
        pd.Series(0.0, index=event_dataframe.index, dtype="float64"),
    ).fillna(0.0)

    attacker_vs_victim_ratio = event_dataframe.get(
        "attacker_vs_victim_size_ratio_token0",
        pd.Series(0.0, index=event_dataframe.index, dtype="float64"),
    ).fillna(0.0)

    same_sender_flag = event_dataframe.get(
        "same_sender_before_after_flag",
        pd.Series(0, index=event_dataframe.index, dtype="int64"),
    ).fillna(0).astype(int)

    different_middle_flag = event_dataframe.get(
        "different_middle_sender_from_neighbors_flag",
        pd.Series(0, index=event_dataframe.index, dtype="int64"),
    ).fillna(0).astype(int)

    reversal_flag = event_dataframe.get(
        "reversal_pattern_flag",
        pd.Series(0, index=event_dataframe.index, dtype="int64"),
    ).fillna(0).astype(int)

    event_dataframe["high_block_gas_context_flag"] = (
        (gas_block_median >= 1.2).fillna(False).astype(int)
    )

    event_dataframe["high_relative_trade_size_flag"] = (
        (attacker_vs_victim_ratio >= 1.0).fillna(False).astype(int)
    )

    event_dataframe["strict_sandwich_support_flag"] = (
        (
            (same_sender_flag == 1)
            & (different_middle_flag == 1)
            & (reversal_flag == 1)
        )
        .fillna(False)
        .astype(int)
    )

    event_dataframe["sandwich_support_score"] = (
        same_sender_flag
        + different_middle_flag
        + reversal_flag
        + event_dataframe["high_block_gas_context_flag"]
        + event_dataframe["high_relative_trade_size_flag"]
    )

    return event_dataframe


def _safe_divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, pd.NA)
    result = pd.to_numeric(result, errors="coerce").fillna(0.0)
    return result


def _safe_float(value: Any) -> float | None:
    """
    Convert a value to float where possible.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
