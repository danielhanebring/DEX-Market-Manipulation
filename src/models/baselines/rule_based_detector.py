from __future__ import annotations

import pandas as pd


def build_rule_based_predictions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Simple baseline prediction table from heuristic rule scores
    """
    df = dataframe.copy()

    if "rule_score" not in df.columns:
        raise ValueError("Expected 'rule_score' column for rule-based baseline.")

    prediction_df = pd.DataFrame({
        "swap_id": df.get("swap_id"),
        "rule_score": df["rule_score"],
        "predicted_suspicious_flag": (df["rule_score"] >= 1).astype(int),
        "predicted_weak_anomaly_flag": (df["rule_score"] >= 3).astype(int),
    })

    return prediction_df