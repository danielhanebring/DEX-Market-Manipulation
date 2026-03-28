from __future__ import annotations

import logging

import pandas as pd

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT
from src.features.sequence_features import (
    build_sequence_feature_table,
    save_sequence_features,
)

logger = logging.getLogger(__name__)


def main() -> None:
    load_environment()
    setup_logging()

    labeled_events_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    labels_path = PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"
    output_path = PROJECT_ROOT / "data" / "features" / "sequence_features.parquet"

    logger.info("Loading event features...")
    events_df = pd.read_parquet(labeled_events_path)

    logger.info("Loading event labels...")
    labels_df = pd.read_parquet(labels_path)

    merged_df = events_df.merge(
        labels_df[
            [
                "swap_id",
                "label_class",
                "label_confidence",
                "rule_score",
            ]
        ],
        on="swap_id",
        how="left",
    )

    logger.info("Merged event dataframe shape: %s", merged_df.shape)

    sequence_df = build_sequence_feature_table(
        dataframe=merged_df,
        sequence_length=20,
        step_size=1,
    )

    logger.info("Sequence dataframe shape: %s", sequence_df.shape)
    save_path = save_sequence_features(sequence_df, output_path)
    logger.info("Saved sequence features to: %s", save_path)

    if not sequence_df.empty:
        logger.info("Sequence preview:\n%s", sequence_df.head(3).to_string())


if __name__ == "__main__":
    main()