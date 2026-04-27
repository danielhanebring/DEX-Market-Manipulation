from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.models.baselines.random_forest_model import (
    DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS,
    prepare_random_forest_dataset,
    time_split_for_random_forest,
)


@dataclass
class XGBoostArtifacts:
    """
    Stores model and evaluation artifacts for the XGBoost baseline.
    """

    model: Any  #
    feature_columns: list[str]
    train_size: int
    validation_size: int
    test_size: int
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    feature_importance_table: pd.DataFrame


def prepare_xgboost_dataset(
    event_features_df: pd.DataFrame,
    event_labels_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reuse the same event-level dataset builder as RandomForest.
    """
    return prepare_random_forest_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
        feature_columns=feature_columns or DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS,
    )


def train_xgboost_model(
    dataset_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    n_estimators: int = 2000,
    early_stopping_rounds: int = 50,
) -> XGBoostArtifacts:
    """
    Train an XGBoost classifier on event-level weak labels.

    """
    from xgboost import XGBClassifier 

    selected_feature_columns = feature_columns or [
        column
        for column in DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS
        if column in dataset_df.columns
    ]

    if not selected_feature_columns:
        raise ValueError("No feature columns available for XGBoost training.")

    train_df, validation_df, test_df = time_split_for_random_forest(dataset_df)

    pos = float(train_df["binary_target"].sum())
    neg = float(len(train_df) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=int(random_state),
        scale_pos_weight=float(scale_pos_weight),
        early_stopping_rounds=int(early_stopping_rounds),
    )

    model.fit(
        train_df[selected_feature_columns],
        train_df["binary_target"],
        eval_set=[(validation_df[selected_feature_columns], validation_df["binary_target"])],
        verbose=False,
    )

    best_threshold, threshold_table = select_best_threshold(
        model=model,
        dataframe=validation_df,
        feature_columns=selected_feature_columns,
    )

    validation_metrics = evaluate_xgboost_model(
        model=model,
        dataframe=validation_df,
        feature_columns=selected_feature_columns,
        threshold=best_threshold,
    )
    test_metrics = evaluate_xgboost_model(
        model=model,
        dataframe=test_df,
        feature_columns=selected_feature_columns,
        threshold=best_threshold,
    )

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = [0.0 for _ in selected_feature_columns]

    feature_importance_table = pd.DataFrame(
        {
            "feature_name": selected_feature_columns,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    feature_importance_table["feature_version"] = "v2_sandwich_detection"

    return XGBoostArtifacts(
        model=model,
        feature_columns=selected_feature_columns,
        train_size=len(train_df),
        validation_size=len(validation_df),
        test_size=len(test_df),
        validation_metrics={
            **validation_metrics,
            "selected_threshold": float(best_threshold),
            "threshold_candidates": threshold_table.to_dict(orient="records"),
        },
        test_metrics={
            **test_metrics,
            "selected_threshold": float(best_threshold),
        },
        feature_importance_table=feature_importance_table,
    )


def evaluate_xgboost_model(
    model: Any,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
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


def generate_xgboost_predictions(
    model: Any,
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

    prediction_df["xgb_probability"] = probabilities
    prediction_df["xgb_predicted_flag"] = predicted_flags
    prediction_df["xgb_threshold_used"] = threshold

    return prediction_df


def select_best_threshold(
    model: Any,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    candidate_thresholds: list[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Select the threshold that gives the best F1 score on validation data.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    x_values = dataframe[feature_columns]
    y_true = dataframe["binary_target"]
    y_score = model.predict_proba(x_values)[:, 1]

    rows = []
    for threshold in candidate_thresholds:
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(rows).sort_values(
        ["f1", "precision", "recall"],
        ascending=False,
    ).reset_index(drop=True)

    best_threshold = float(threshold_df.iloc[0]["threshold"])
    return best_threshold, threshold_df
