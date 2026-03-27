from __future__ import annotations

import logging

import pandas as pd

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.labeling.build_labels import build_event_labels, extract_event_label_table, save_event_labels
from src.models.baselines.rule_based_detector import build_rule_based_predictions

logger = logging.getLogger(__name__)


def main() -> None:
    """Run weak labeling and create a first rule-based baseline output."""
    load_environment()
    setup_logging()

    event_features_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    labels_output_path = PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"
    predictions_output_path = PROJECT_ROOT / "outputs" / "predictions" / "rule_based_event_predictions.parquet"

    ensure_directory(labels_output_path.parent)
    ensure_directory(predictions_output_path.parent)

    logger.info("Loading event features from: %s", event_features_path)
    dataframe = pd.read_parquet(event_features_path)

    logger.info("Input feature dataframe shape: %s", dataframe.shape)

    labeled_dataframe = build_event_labels(dataframe)
    label_table = extract_event_label_table(labeled_dataframe)

    saved_labels_path = save_event_labels(label_table, labels_output_path)
    logger.info("Saved event labels to: %s", saved_labels_path)

    prediction_table = build_rule_based_predictions(labeled_dataframe)
    prediction_table.to_parquet(predictions_output_path, index=False)
    logger.info("Saved rule-based predictions to: %s", predictions_output_path)

    if "label_class" in label_table.columns:
        logger.info("Label distribution:\n%s", label_table["label_class"].value_counts(dropna=False).to_string())

    preview_columns = [
        column for column in [
            "swap_id",
            "label_class",
            "label_confidence",
            "rule_score",
            "label_notes",
        ]
        if column in label_table.columns
    ]

    if preview_columns:
        logger.info("Label preview:\n%s", label_table[preview_columns].head(10).to_string())


if __name__ == "__main__":
    main()