from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.time_split import time_based_split


def evaluate_rule_based_model(dataframe: pd.DataFrame) -> dict:
    """
    Evaluate rule-based baseline against weak labels.
    """

    df = dataframe.dropna(subset=["binary_target"]).copy()

    if df.empty:
        return {"error": "No labeled data available"}

    train_df, test_df = time_based_split(df)

    y_true = test_df["binary_target"]
    y_pred = test_df["predicted_weak_anomaly_flag"]
    y_score = test_df["rule_score"]

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )

    return {
        "model": "rule_based",
        "metrics": metrics,
        "num_train": len(train_df),
        "num_test": len(test_df),
    }