from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.models.baselines.isolation_forest_model import (
    generate_isolation_forest_predictions,
    prepare_isolation_forest_dataset,
    train_isolation_forest_model,
)
from src.models.baselines.random_forest_model import time_split_for_random_forest

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run the full Isolation Forest pipeline for event-level sandwich detection.
    """
    load_environment()
    setup_logging()

    event_features_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    event_labels_path = PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"

    model_output_path = PROJECT_ROOT / "outputs" / "reports" / "isolation_forest_model.joblib"
    metrics_output_path = PROJECT_ROOT / "outputs" / "metrics" / "isolation_forest_metrics.json"
    predictions_output_path = PROJECT_ROOT / "outputs" / "predictions" / "isolation_forest_test_predictions.parquet"

    ensure_directory(model_output_path.parent)
    ensure_directory(metrics_output_path.parent)
    ensure_directory(predictions_output_path.parent)

    logger.info("Loading event features from: %s", event_features_path)
    event_features_df = pd.read_parquet(event_features_path)

    logger.info("Loading event labels from: %s", event_labels_path)
    event_labels_df = pd.read_parquet(event_labels_path)

    dataset_df = prepare_isolation_forest_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
    )

    logger.info("Prepared Isolation Forest dataset shape: %s", dataset_df.shape)
    logger.info(
        "Binary target distribution:\n%s",
        dataset_df["binary_target"].value_counts(dropna=False).to_string(),
    )

    artifacts = train_isolation_forest_model(
        dataset_df=dataset_df,
        contamination=0.01,
    )

    logger.info("Validation metrics: %s", artifacts.validation_metrics)
    logger.info("Test metrics: %s", artifacts.test_metrics)

    selected_threshold = float(artifacts.validation_metrics["selected_threshold"])

    _, _, test_df = time_split_for_random_forest(dataset_df)
    test_predictions_df = generate_isolation_forest_predictions(
        model=artifacts.model,
        dataframe=test_df,
        feature_columns=artifacts.feature_columns,
        threshold=selected_threshold,
    )

    joblib.dump(artifacts.model, model_output_path)
    test_predictions_df.to_parquet(predictions_output_path, index=False)

    metrics_payload = {
        "model": "isolation_forest_event_anomaly_detector",
        "feature_columns": artifacts.feature_columns,
        "train_size": artifacts.train_size,
        "validation_size": artifacts.validation_size,
        "test_size": artifacts.test_size,
        "validation_metrics": artifacts.validation_metrics,
        "test_metrics": artifacts.test_metrics,
        "dataset_notes": {
            "training_strategy": "fit_on_normal_train_rows_only",
            "positive_class_for_evaluation": "weak_anomaly",
            "negative_class_for_evaluation": "normal",
            "excluded_class": "suspicious",
            "split_strategy": "time_based",
            "contamination": 0.01,
        },
    }

    with metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    logger.info("Saved Isolation Forest model to: %s", model_output_path)
    logger.info("Saved metrics to: %s", metrics_output_path)
    logger.info("Saved test predictions to: %s", predictions_output_path)


if __name__ == "__main__":
    main()