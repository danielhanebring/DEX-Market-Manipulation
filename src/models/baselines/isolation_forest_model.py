from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.models.baselines.random_forest_model import (
    DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS,
    prepare_random_forest_dataset,
    time_split_for_random_forest,
)


@dataclass
class IsolationForestArtifacts:
    """
    Stores model and evaluation for the Isolation Forest run.
    """

    model: IsolationForest
    feature_columns: list[str]
    train_size: int
    validation_size: int
    test_size: int
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


def prepare_isolation_forest_dataset(
    event_features_df: pd.DataFrame,
    event_labels_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:

    return prepare_random_forest_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
        feature_columns=feature_columns or DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS,
    )


def train_isolation_forest_model(
    dataset_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
    contamination: float = 0.01,
) -> IsolationForestArtifacts:
    """
    Train Isolation Forest on normal training events only.

    """
    selected_feature_columns = feature_columns or [
        column
        for column in DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS
        if column in dataset_df.columns
    ]

    train_df, validation_df, test_df = time_split_for_random_forest(dataset_df)

    normal_train_df = train_df[train_df["binary_target"] == 0].copy()
    if normal_train_df.empty:
        raise ValueError("No normal training rows available for Isolation Forest.")

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(normal_train_df[selected_feature_columns])

    validation_metrics = evaluate_isolation_forest_model(
        model=model,
        dataframe=validation_df,
        feature_columns=selected_feature_columns,
    )

    test_metrics = evaluate_isolation_forest_model(
        model=model,
        dataframe=test_df,
        feature_columns=selected_feature_columns,
    )

    return IsolationForestArtifacts(
        model=model,
        feature_columns=selected_feature_columns,
        train_size=len(normal_train_df),
        validation_size=len(validation_df),
        test_size=len(test_df),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def evaluate_isolation_forest_model(
    model: IsolationForest,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    if dataframe.empty:
        return {"error": "Empty dataset split."}

    x_values = dataframe[feature_columns]
    y_true = dataframe["binary_target"]

    raw_predictions = model.predict(x_values)
    anomaly_flags = (raw_predictions == -1).astype(int)

    # More abnormal = lower decision score, so invert for PR-AUC
    decision_scores = -model.decision_function(x_values)

    metrics = {
        "precision": precision_score(y_true, anomaly_flags, zero_division=0),
        "recall": recall_score(y_true, anomaly_flags, zero_division=0),
        "f1": f1_score(y_true, anomaly_flags, zero_division=0),
        "pr_auc": average_precision_score(y_true, decision_scores) if len(set(y_true)) > 1 else None,
        "support": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) > 0 else None,
        "predicted_positive_rate": float(anomaly_flags.mean()) if len(anomaly_flags) > 0 else None,
    }

    return metrics


def generate_isolation_forest_predictions(
    model: IsolationForest,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Generate anomaly predictions for a dataframe split.
    """
    prediction_df = dataframe[["swap_id", "timestamp", "label_class", "binary_target"]].copy()

    raw_predictions = model.predict(dataframe[feature_columns])
    anomaly_flags = (raw_predictions == -1).astype(int)
    anomaly_scores = -model.decision_function(dataframe[feature_columns])

    prediction_df["if_anomaly_score"] = anomaly_scores
    prediction_df["if_predicted_flag"] = anomaly_flags

    return prediction_df