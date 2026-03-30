from __future__ import annotations

import logging

import pandas as pd

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.labeling.build_labels import build_sandwich_labels, save_label_table
from src.labeling.sandwich_rules import SandwichRuleConfig

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run sandwich-pattern labeling on event-level features.
    """
    load_environment()
    setup_logging()

    event_features_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    event_labels_path = PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"
    sequence_labels_path = PROJECT_ROOT / "data" / "labels" / "sequence_labels.parquet"

    ensure_directory(event_labels_path.parent)
    ensure_directory(sequence_labels_path.parent)

    logger.info("Loading event features from: %s", event_features_path)
    features_dataframe = pd.read_parquet(event_features_path)
    logger.info("Event feature dataframe shape: %s", features_dataframe.shape)

    rule_config = SandwichRuleConfig(
        minimum_attacker_trade_size=0.0,
        require_same_sender_before_after=True,
        require_reversal_pattern=True,
        require_same_block=True,
        suspicious_gas_multiplier=1.2,
    )

    event_labels_df, sequence_labels_df = build_sandwich_labels(
        features_dataframe=features_dataframe,
        config=rule_config,
    )

    saved_event_path = save_label_table(event_labels_df, event_labels_path)
    saved_sequence_path = save_label_table(sequence_labels_df, sequence_labels_path)

    logger.info("Saved event labels to: %s", saved_event_path)
    logger.info("Saved sequence labels to: %s", saved_sequence_path)

    if "label_class" in event_labels_df.columns:
        logger.info(
            "Event label distribution:\n%s",
            event_labels_df["label_class"].value_counts(dropna=False).to_string(),
        )

    if "label_class" in sequence_labels_df.columns:
        logger.info(
            "Sequence label distribution:\n%s",
            sequence_labels_df["label_class"].value_counts(dropna=False).to_string(),
        )

    preview_columns = [
        column for column in [
            "sequence_id",
            "block_number",
            "attacker_address",
            "victim_address",
            "same_sender_before_after_flag",
            "different_middle_sender_flag",
            "reversal_pattern_flag",
            "relative_trade_size",
            "label_class",
            "label_confidence",
            "notes",
        ]
        if column in sequence_labels_df.columns
    ]

    if preview_columns and not sequence_labels_df.empty:
        logger.info(
            "Sequence label preview:\n%s",
            sequence_labels_df[preview_columns].head(10).to_string(),
        )


if __name__ == "__main__":
    main()