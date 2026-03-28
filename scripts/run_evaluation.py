from __future__ import annotations

import logging
import pandas as pd

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT
from src.evaluation.evaluate_models import evaluate_rule_based_model

logger = logging.getLogger(__name__)


def main() -> None:
    load_environment()
    setup_logging()

    labels_path = PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"
    predictions_path = PROJECT_ROOT / "outputs" / "predictions" / "rule_based_event_predictions.parquet"

    logger.info("Loading labels...")
    labels_df = pd.read_parquet(labels_path)

    logger.info("Loading predictions...")
    pred_df = pd.read_parquet(predictions_path)

    df = labels_df.merge(
        pred_df,
        on="swap_id",
        how="inner",
        suffixes=("_label", "_pred"),
    )

    logger.info("Merged dataset shape: %s", df.shape)
    logger.info("Merged columns: %s", list(df.columns))


    if "rule_score_pred" in df.columns:
        df["rule_score"] = df["rule_score_pred"]
    elif "rule_score_label" in df.columns:
        df["rule_score"] = df["rule_score_label"]

    results = evaluate_rule_based_model(df)

    logger.info("Evaluation results:")
    for key, value in results["metrics"].items():
        logger.info("%s: %s", key, value)

    logger.info("Train size: %s | Test size: %s", results["num_train"], results["num_test"])


if __name__ == "__main__":
    main()