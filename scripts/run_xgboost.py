from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_environment 
from src.common.logging_utils import setup_logging  
from src.common.paths import PROJECT_ROOT, ensure_directory  
from src.models.baselines.random_forest_model import time_split_for_random_forest 
from src.models.baselines.xgboost_model import ( 
    generate_xgboost_predictions,
    prepare_xgboost_dataset,
    train_xgboost_model,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Train and evaluate an XGBoost event-level baseline.
    """
    load_environment()
    setup_logging()

    event_features_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    event_labels_path = PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"

    model_output_path = PROJECT_ROOT / "outputs" / "reports" / "xgboost_model.joblib"
    metrics_output_path = PROJECT_ROOT / "outputs" / "metrics" / "xgboost_metrics.json"
    importance_output_path = PROJECT_ROOT / "outputs" / "reports" / "xgboost_feature_importance.csv"
    predictions_output_path = PROJECT_ROOT / "outputs" / "predictions" / "xgboost_test_predictions.parquet"

    ensure_directory(model_output_path.parent)
    ensure_directory(metrics_output_path.parent)
    ensure_directory(importance_output_path.parent)
    ensure_directory(predictions_output_path.parent)

    logger.info("Loading event features from: %s", event_features_path)
    event_features_df = pd.read_parquet(event_features_path)

    logger.info("Loading event labels from: %s", event_labels_path)
    event_labels_df = pd.read_parquet(event_labels_path)

    dataset_df = prepare_xgboost_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
    )

    logger.info("Prepared XGBoost dataset shape: %s", dataset_df.shape)
    logger.info(
        "Binary target distribution:\n%s",
        dataset_df["binary_target"].value_counts(dropna=False).to_string(),
    )

    artifacts = train_xgboost_model(dataset_df=dataset_df)
    logger.info("Validation metrics: %s", artifacts.validation_metrics)
    logger.info("Test metrics: %s", artifacts.test_metrics)

    selected_threshold = float(artifacts.validation_metrics["selected_threshold"])

    _, _, test_df = time_split_for_random_forest(dataset_df)
    test_predictions_df = generate_xgboost_predictions(
        model=artifacts.model,
        dataframe=test_df,
        feature_columns=artifacts.feature_columns,
        threshold=selected_threshold,
    )

    joblib.dump(artifacts.model, model_output_path)
    artifacts.feature_importance_table.to_csv(importance_output_path, index=False)
    test_predictions_df.to_parquet(predictions_output_path, index=False)

    metrics_payload = {
        "model": "xgboost_event_classifier",
        "feature_columns": artifacts.feature_columns,
        "train_size": artifacts.train_size,
        "validation_size": artifacts.validation_size,
        "test_size": artifacts.test_size,
        "validation_metrics": artifacts.validation_metrics,
        "test_metrics": artifacts.test_metrics,
        "feature_version": "v2_sandwich_detection",
        "dataset_notes": {
            "positive_class": "weak_anomaly",
            "negative_class": "normal",
            "excluded_class": "suspicious",
            "split_strategy": "time_based",
            "tree_method": "hist",
            "early_stopping": True,
        },
    }

    with metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    logger.info("Saved XGBoost model to: %s", model_output_path)
    logger.info("Saved metrics to: %s", metrics_output_path)
    logger.info("Saved feature importance to: %s", importance_output_path)
    logger.info("Saved test predictions to: %s", predictions_output_path)


if __name__ == "__main__":
    main()

