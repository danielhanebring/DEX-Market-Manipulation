from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SEQUENCE_FEATURE_COLUMNS = [
    "swap_size_token0",
    "swap_size_token1",
    "interarrival_seconds",
    "same_block_event_count",
    "local_event_density_10",
    "gas_price_gwei",
    "gas_price_relative_to_local_median",
    "same_sender_recent_count_20",
    "same_recipient_recent_count_20",
    "sender_recipient_pair_recent_count_20",
    "tick_change_from_previous",
    "abs_tick_change",
]


def build_sliding_window_sequences(
    dataframe: pd.DataFrame,
    sequence_length: int = 20,
    feature_columns: list[str] | None = None,
    step_size: int = 1,
) -> pd.DataFrame:
    """
    Build sliding-window sequences in each pool.
    Each row in the output represents one sequence.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_SEQUENCE_FEATURE_COLUMNS.copy()

    available_feature_columns = [
        column for column in feature_columns if column in dataframe.columns
    ]
    if not available_feature_columns:
        raise ValueError("No requested feature columns were found in the dataframe.")

    required_columns = [
        "swap_id",
        "pool_address",
        "block_number",
        "timestamp",
    ]
    missing_required = [column for column in required_columns if column not in dataframe.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for sequence construction: {missing_required}")

    df = dataframe.copy()
    df = df.sort_values(["pool_address", "block_number", "log_index", "swap_id"]).reset_index(drop=True)

    for column in available_feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[available_feature_columns] = df[available_feature_columns].fillna(0.0)

    sequence_rows: list[dict[str, Any]] = []

    for pool_address, pool_df in df.groupby("pool_address", sort=False):
        pool_df = pool_df.reset_index(drop=True)

        if len(pool_df) < sequence_length:
            continue

        for start_index in range(0, len(pool_df) - sequence_length + 1, step_size):
            end_index = start_index + sequence_length
            window_df = pool_df.iloc[start_index:end_index]

            sequence_id = _build_sequence_id(
                pool_address=pool_address,
                start_block=int(window_df["block_number"].iloc[0]),
                end_block=int(window_df["block_number"].iloc[-1]),
                start_swap_id=str(window_df["swap_id"].iloc[0]),
                end_swap_id=str(window_df["swap_id"].iloc[-1]),
            )

            sequence_matrix = window_df[available_feature_columns].to_numpy(dtype=np.float32)

            rule_score_max = (
                float(window_df["rule_score"].max())
                if "rule_score" in window_df.columns
                else None
            )

            contains_weak_anomaly = (
                int((window_df["label_class"] == "weak_anomaly").any())
                if "label_class" in window_df.columns
                else None
            )

            contains_suspicious = (
                int(window_df["label_class"].isin(["suspicious", "weak_anomaly"]).any())
                if "label_class" in window_df.columns
                else None
            )

            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "pool_address": pool_address,
                    "window_start_block": int(window_df["block_number"].iloc[0]),
                    "window_end_block": int(window_df["block_number"].iloc[-1]),
                    "window_start_time": int(window_df["timestamp"].iloc[0]),
                    "window_end_time": int(window_df["timestamp"].iloc[-1]),
                    "event_count": int(len(window_df)),
                    "sequence_length": int(sequence_length),
                    "feature_columns": available_feature_columns,
                    "sequence_features": sequence_matrix.tolist(),
                    "source_event_ids": window_df["swap_id"].tolist(),
                    "target_contains_weak_anomaly": contains_weak_anomaly,
                    "target_contains_suspicious_or_anomaly": contains_suspicious,
                    "rule_score_max": rule_score_max,
                }
            )

    return pd.DataFrame(sequence_rows)


def _build_sequence_id(
    pool_address: str,
    start_block: int,
    end_block: int,
    start_swap_id: str,
    end_swap_id: str,
) -> str:
    return f"{pool_address}|{start_block}|{end_block}|{start_swap_id}|{end_swap_id}"