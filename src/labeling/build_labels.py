from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import ensure_directory
from src.labeling.heuristic_flags import add_heuristic_flags
from src.labeling.label_mapping import add_binary_model_target
from src.labeling.rule_labels import assign_rule_based_labels


def build_event_labels(features_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Build event level labels from event features.
    """
    df = features_dataframe.copy()
    df = add_heuristic_flags(df)
    df = assign_rule_based_labels(df)
    df = add_binary_model_target(df)

    return df


def extract_event_label_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a clean label table for storage in data/labels/.
    """
    preferred_columns = [
        "swap_id",
        "pool_address",
        "transaction_hash",
        "block_number",
        "timestamp",
        "label_class",
        "label_source",
        "label_confidence",
        "rule_score",
        "binary_target",
        "label_notes",
    ]

    heuristic_columns = [
        "flag_high_gas_context",
        "flag_same_block_pattern",
        "flag_repeated_address_pattern",
        "flag_large_price_movement",
        "flag_large_trade_size",
        "flag_burst_activity",
    ]

    keep_columns = [
        column for column in preferred_columns + heuristic_columns
        if column in dataframe.columns
    ]

    return dataframe[keep_columns].copy()


def save_event_labels(dataframe: pd.DataFrame, output_file: str | Path) -> Path:
    output_path = Path(output_file)
    ensure_directory(output_path.parent)
    dataframe.to_parquet(output_path, index=False)
    return output_path