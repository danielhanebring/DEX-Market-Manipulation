from __future__ import annotations

import json
import logging

import pandas as pd

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.features.event_features import build_event_features

logger = logging.getLogger(__name__)


def main() -> None:
    load_environment()
    setup_logging()

    processed_swaps_path = PROJECT_ROOT / "data" / "processed" / "swaps_clean.parquet"
    output_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    metadata_path = PROJECT_ROOT / "data" / "features" / "event_features_metadata.json"

    ensure_directory(output_path.parent)

    logger.info("Loading processed swaps from: %s", processed_swaps_path)
    processed_dataframe = pd.read_parquet(processed_swaps_path)

    logger.info("Input dataframe shape: %s", processed_dataframe.shape)
    feature_dataframe = build_event_features(processed_dataframe)
    logger.info("Feature dataframe shape: %s", feature_dataframe.shape)

    feature_dataframe.to_parquet(output_path, index=False)
    logger.info("Saved event features to: %s", output_path)

    feature_metadata = {
        "feature_version": "v2_sandwich_detection",
        "input_rows": int(len(processed_dataframe)),
        "output_rows": int(len(feature_dataframe)),
        "output_columns": list(feature_dataframe.columns),
        "notes": [
            "Sandwich-specific features added",
            "Three event helper features included",
            "Block relative gas and trade size features included",
        ],
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(feature_metadata, file, indent=2)

    logger.info("Saved feature metadata to: %s", metadata_path)

    preview_columns = [
        column
        for column in [
            "swap_id",
            "block_number",
            "position_in_block",
            "same_sender_before_after_flag",
            "different_middle_sender_from_neighbors_flag",
            "reversal_pattern_flag",
            "three_event_pattern_indicator",
            "gas_price_relative_to_block_median",
            "relative_trade_size_token0",
            "sandwich_support_score",
        ]
        if column in feature_dataframe.columns
    ]

    if preview_columns:
        logger.info(
            "Feature preview:\n%s",
            feature_dataframe[preview_columns].head(10).to_string(),
        )


if __name__ == "__main__":
    main()