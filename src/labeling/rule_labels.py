from __future__ import annotations

import pandas as pd


def assign_rule_based_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Assign weak labels from rule flags.
    Examples:
    - normal
    - suspicious
    - weak_anomaly
    - unlabeled

    """
    df = dataframe.copy()

    flag_columns = [
        "flag_high_gas_context",
        "flag_same_block_pattern",
        "flag_repeated_address_pattern",
        "flag_large_price_movement",
        "flag_large_trade_size",
        "flag_burst_activity",
    ]

    existing_flag_columns = [column for column in flag_columns if column in df.columns]

    if not existing_flag_columns:
        df["label_class"] = "unlabeled"
        df["label_confidence"] = "low"
        df["label_source"] = "rule_based"
        df["rule_score"] = 0
        return df

    df["rule_score"] = df[existing_flag_columns].fillna(0).sum(axis=1)

    df["label_class"] = df["rule_score"].apply(_map_score_to_label)
    df["label_confidence"] = df["rule_score"].apply(_map_score_to_confidence)
    df["label_source"] = "rule_based"

    df["label_notes"] = df.apply(_build_label_notes, axis=1)

    return df


def _map_score_to_label(score: int | float) -> str:
    if score >= 3:
        return "weak_anomaly"
    if score >= 1:
        return "suspicious"
    return "normal"


def _map_score_to_confidence(score: int | float) -> str:
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _build_label_notes(row: pd.Series) -> str:
    active_flags: list[str] = []

    flag_name_mapping = {
        "flag_high_gas_context": "high_gas_context",
        "flag_same_block_pattern": "same_block_pattern",
        "flag_repeated_address_pattern": "repeated_address_pattern",
        "flag_large_price_movement": "large_price_movement",
        "flag_large_trade_size": "large_trade_size",
        "flag_burst_activity": "burst_activity",
    }

    for column, label in flag_name_mapping.items():
        if column in row.index and row[column] == 1:
            active_flags.append(label)

    if not active_flags:
        return "no_rule_triggered"

    return ",".join(active_flags)
