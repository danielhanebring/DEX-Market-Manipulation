from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.baselines.random_forest_model import (
    DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS,
    prepare_random_forest_dataset,
    time_split_for_random_forest,
)


@dataclass
class LogisticRegressionArtifacts:

    model: Pipeline
    feature_columns: list[str]
    train_size: int
    validation_size: int
    test_size: int
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    coefficient_table: pd.DataFrame


def prepare_logistic_regression_dataset(
    event_features_df: pd.DataFrame,
    event_labels_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reuse the same dataset builder as the RandomForest baseline.
    """
    return prepare_random_forest_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
        feature_columns=feature_columns or DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS,
    )


def train_logistic_regression_model(
    dataset_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
    c: float = 1.0,
    max_iter: int = 2000,
) -> LogisticRegressionArtifacts:
    """
    Train a simple logistic regression classifier on event-level weak labels.
    """
    selected_feature_columns = feature_columns or [
        column for column in DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS if column in dataset_df.columns
    ]

    if not selected_feature_columns:
        raise ValueError("No feature columns available for Logistic Regression training.")

    train_df, validation_df, test_df = time_split_for_random_forest(dataset_df)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logreg",
                LogisticRegression(
                    C=float(c),
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    class_weight="balanced",
                    random_state=int(random_state),
                ),
            ),
        ]
    )

    model.fit(
        train_df[selected_feature_columns],
        train_df["binary_target"],
    )

    best_threshold, threshold_table = select_best_threshold(
        model=model,
        dataframe=validation_df,
        feature_columns=selected_feature_columns,
    )

    validation_metrics = evaluate_logistic_regression_model(
        model=model,
        dataframe=validation_df,
        feature_columns=selected_feature_columns,
        threshold=best_threshold,
    )

    test_metrics = evaluate_logistic_regression_model(
        model=model,
        dataframe=test_df,
        feature_columns=selected_feature_columns,
        threshold=best_threshold,
    )

    coefficient_table = build_coefficient_table(
        model=model,
        feature_columns=selected_feature_columns,
    )
    coefficient_table["feature_version"] = "v2_sandwich_detection"

    return LogisticRegressionArtifacts(
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
        coefficient_table=coefficient_table,
    )


def evaluate_logistic_regression_model(
    model: Pipeline,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate Logistic Regression predictions for a given split.
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


def generate_logistic_regression_predictions(
    model: Pipeline,
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

    prediction_df["lr_probability"] = probabilities
    prediction_df["lr_predicted_flag"] = predicted_flags
    prediction_df["lr_threshold_used"] = threshold

    return prediction_df


def build_coefficient_table(
    model: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Extract coefficients from the logistic regression step.
    """
    if "logreg" not in model.named_steps:
        raise ValueError("Expected pipeline step named 'logreg'.")

    logreg: LogisticRegression = model.named_steps["logreg"]

    coefs = logreg.coef_[0]
    rows = []
    for name, coef in zip(feature_columns, coefs, strict=False):
        rows.append(
            {
                "feature_name": name,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
            }
        )

    table = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    table["intercept"] = float(logreg.intercept_[0])
    return table


def select_best_threshold(
    model: Pipeline,
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
