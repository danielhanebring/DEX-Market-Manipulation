from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


@dataclass
class RandomForestArtifacts:
    """
    Store data from run
    """

    model: RandomForestClassifier
    feature_columns: list[str]
    train_size: int
    validation_size: int
    test_size: int
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    feature_importance_table: pd.DataFrame


DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS = [
    "swap_size_token0",
    "swap_size_token1",
    "interarrival_seconds",
    "same_block_event_count",
    "same_block_pattern_flag",
    "local_event_density_10",
    "gas_price_gwei",
    "gas_price_relative_to_local_mean",
    "gas_price_relative_to_local_median",
    "gas_spike_flag",
    "same_sender_recent_count_20",
    "same_recipient_recent_count_20",
    "sender_recipient_pair_recent_count_20",
    "same_sender_same_block_count",
    "same_recipient_same_block_count",
    "tick_change_from_previous",
    "abs_tick_change",
    "relative_sqrt_price_change",
    "burst_activity_flag",
]


def prepare_random_forest_dataset(
    event_features_df: pd.DataFrame,
    event_labels_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge even features and labels and turn into binary
    """
    selected_feature_columns = feature_columns or DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS

    required_merge_columns = ["swap_id", "timestamp"]
    missing_event_columns = [column for column in required_merge_columns if column not in event_features_df.columns]
    if missing_event_columns:
        raise ValueError(
            f"Missing required event feature columns: {missing_event_columns}"
        )

    if "swap_id" not in event_labels_df.columns or "label_class" not in event_labels_df.columns:
        raise ValueError("Event labels must include 'swap_id' and 'label_class'.")

    merged_df = event_features_df.merge(
        event_labels_df[["swap_id", "label_class"]],
        on="swap_id",
        how="inner",
    )

    merged_df = merged_df[merged_df["label_class"].isin(["normal", "weak_anomaly"])].copy()
    merged_df["binary_target"] = merged_df["label_class"].map(
        {
            "normal": 0,
            "weak_anomaly": 1,
        }
    )

    available_feature_columns = [
        column for column in selected_feature_columns
        if column in merged_df.columns
    ]

    if not available_feature_columns:
        raise ValueError("No selected feature columns were found in the merged dataset.")

    for column in available_feature_columns:
        merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce")

    merged_df[available_feature_columns] = merged_df[available_feature_columns].fillna(0.0)
    merged_df["binary_target"] = pd.to_numeric(merged_df["binary_target"], errors="coerce")

    keep_columns = ["swap_id", "timestamp", "label_class", "binary_target"] + available_feature_columns
    dataset_df = merged_df[keep_columns].copy()

    return dataset_df.sort_values("timestamp").reset_index(drop=True)


def time_split_for_random_forest(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split for event-level modeling.
    """
    df = dataframe.sort_values("timestamp").reset_index(drop=True)

    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def train_random_forest_model(
    dataset_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
) -> RandomForestArtifacts:
    """
    Train and evaluate a Random Forest classifier on event-level sandwich labels.
    """
    selected_feature_columns = feature_columns or [
        column
        for column in DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS
        if column in dataset_df.columns
    ]

    train_df, validation_df, test_df = time_split_for_random_forest(dataset_df)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        train_df[selected_feature_columns],
        train_df["binary_target"],
    )

    validation_metrics = evaluate_random_forest_model(
        model=model,
        dataframe=validation_df,
        feature_columns=selected_feature_columns,
    )

    test_metrics = evaluate_random_forest_model(
        model=model,
        dataframe=test_df,
        feature_columns=selected_feature_columns,
    )

    feature_importance_table = pd.DataFrame(
        {
            "feature_name": selected_feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return RandomForestArtifacts(
        model=model,
        feature_columns=selected_feature_columns,
        train_size=len(train_df),
        validation_size=len(validation_df),
        test_size=len(test_df),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        feature_importance_table=feature_importance_table,
    )


def evaluate_random_forest_model(
    model: RandomForestClassifier,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate Random Forest predictions
    """
    if dataframe.empty:
        return {"error": "Empty dataset split."}

    x_values = dataframe[feature_columns]
    y_true = dataframe["binary_target"]

    y_score = model.predict_proba(x_values)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else None,
        "support": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) > 0 else None,
        "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) > 0 else None,
    }

    return metrics


def generate_random_forest_predictions(
    model: RandomForestClassifier,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate prediction table for a given split.
    """
    prediction_df = dataframe[["swap_id", "timestamp", "label_class", "binary_target"]].copy()

    probabilities = model.predict_proba(dataframe[feature_columns])[:, 1]
    predicted_flags = (probabilities >= threshold).astype(int)

    prediction_df["rf_probability"] = probabilities
    prediction_df["rf_predicted_flag"] = predicted_flags

    return prediction_df