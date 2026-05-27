from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

# Allow `from src...` when running as a script.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.models.baselines.random_forest_model import (
    generate_random_forest_predictions,
    prepare_random_forest_dataset,
    time_split_for_random_forest,
    train_random_forest_model,
    NO_LEAKAGE_RANDOM_FOREST_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def main() -> None:
    load_environment()
    setup_logging()

    event_features_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    event_labels_path = PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"

    ensure_directory((PROJECT_ROOT / "outputs" / "reports"))
    ensure_directory((PROJECT_ROOT / "outputs" / "metrics"))
    ensure_directory((PROJECT_ROOT / "outputs" / "predictions"))

    logger.info("Loading event features from: %s", event_features_path)
    event_features_df = pd.read_parquet(event_features_path)

    logger.info("Loading event labels from: %s", event_labels_path)
    event_labels_df = pd.read_parquet(event_labels_path)

    dataset_df = prepare_random_forest_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
    )

    logger.info("Prepared Random Forest dataset shape: %s", dataset_df.shape)
    logger.info(
        "Binary target distribution:\n%s",
        dataset_df["binary_target"].value_counts(dropna=False).to_string(),
    )

    def _run_one(*, name: str, feature_columns: list[str] | None, legacy_outputs: bool = False) -> None:
        logger.info("=== Random Forest run: %s ===", name)

        if legacy_outputs:
            model_output_path = PROJECT_ROOT / "outputs" / "reports" / "random_forest_model.joblib"
            metrics_output_path = PROJECT_ROOT / "outputs" / "metrics" / "random_forest_metrics.json"
            importance_output_path = PROJECT_ROOT / "outputs" / "reports" / "random_forest_feature_importance.csv"
            predictions_output_path = PROJECT_ROOT / "outputs" / "predictions" / "random_forest_test_predictions.parquet"
        else:
            model_output_path = PROJECT_ROOT / "outputs" / "reports" / f"random_forest_model_{name}.joblib"
            metrics_output_path = PROJECT_ROOT / "outputs" / "metrics" / f"random_forest_metrics_{name}.json"
            importance_output_path = PROJECT_ROOT / "outputs" / "reports" / f"random_forest_feature_importance_{name}.csv"
            predictions_output_path = PROJECT_ROOT / "outputs" / "predictions" / f"random_forest_test_predictions_{name}.parquet"

        artifacts = train_random_forest_model(dataset_df=dataset_df, feature_columns=feature_columns)

        logger.info("Validation metrics: %s", artifacts.validation_metrics)
        logger.info("Test metrics: %s", artifacts.test_metrics)

        selected_threshold = float(artifacts.validation_metrics["selected_threshold"])

        _, _, test_df = time_split_for_random_forest(dataset_df)
        test_predictions_df = generate_random_forest_predictions(
            model=artifacts.model,
            dataframe=test_df,
            feature_columns=artifacts.feature_columns,
            threshold=selected_threshold,
        )

        joblib.dump(artifacts.model, model_output_path)
        artifacts.feature_importance_table.to_csv(importance_output_path, index=False)
        test_predictions_df.to_parquet(predictions_output_path, index=False)

        metrics_payload = {
            "model": "random_forest_event_classifier",
            "experiment": name,
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
            },
        }

        with metrics_output_path.open("w", encoding="utf-8") as file:
            json.dump(metrics_payload, file, indent=2)

        logger.info("Saved model: %s", model_output_path)
        logger.info("Saved metrics: %s", metrics_output_path)
        logger.info("Saved importance: %s", importance_output_path)
        logger.info("Saved test predictions: %s", predictions_output_path)

    # Keep the historical filenames for the "full" run so existing thesis scripts keep working.
    _run_one(name="full", feature_columns=None, legacy_outputs=True)
    _run_one(name="no_leakage", feature_columns=NO_LEAKAGE_RANDOM_FOREST_FEATURE_COLUMNS)


if __name__ == "__main__":
    main()
