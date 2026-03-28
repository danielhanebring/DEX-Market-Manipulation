from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series | None = None,
) -> dict:
    """
    Compute basic classification metrics.
    """

    metrics = {}

    # Remove NA
    mask = y_true.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"error": "No valid samples"}

    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if y_score is not None:
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_score)
        except Exception:
            metrics["pr_auc"] = None

    metrics["support"] = len(y_true)

    return metrics