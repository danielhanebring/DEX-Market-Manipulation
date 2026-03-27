from __future__ import annotations

import logging

import pandas as pd

from src.common.config import load_environment, load_yaml_config
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.features.event_features import build_event_features

logger = logging.getLogger(__name__)


def main() -> None:
    """Run event-level feature engineering on processed swap data."""
    load_environment()
    setup_logging()

    config_path = PROJECT_ROOT / "configs" / "data.yaml"
    config = load_yaml_config(config_path)

    processed_swaps_path = PROJECT_ROOT / "data" / "processed" / "swaps_clean.parquet"
    output_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"

    ensure_directory(output_path.parent)

    logger.info("Loading processed swaps from: %s", processed_swaps_path)
    dataframe = pd.read_parquet(processed_swaps_path)

    logger.info("Input dataframe shape: %s", dataframe.shape)
    features_dataframe = build_event_features(dataframe)
    logger.info("Feature dataframe shape: %s", features_dataframe.shape)

    features_dataframe.to_parquet(output_path, index=False)
    logger.info("Saved event features to: %s", output_path)

    preview_columns = [
        column for column in [
            "swap_id",
            "pool_address",
            "timestamp",
            "trade_direction",
            "swap_size_token0",
            "swap_size_token1",
            "interarrival_seconds",
            "same_block_event_count",
            "gas_price_gwei",
            "gas_spike_flag",
            "same_sender_recent_count_20",
            "same_recipient_recent_count_20",
            "tick_change_from_previous",
            "abs_tick_change",
        ]
        if column in features_dataframe.columns
    ]

    if preview_columns:
        logger.info("Feature preview:\n%s", features_dataframe[preview_columns].head(10).to_string())


if __name__ == "__main__":
    main()