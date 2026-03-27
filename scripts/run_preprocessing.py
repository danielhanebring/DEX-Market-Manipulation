from __future__ import annotations

import logging

from src.common.config import load_environment, load_yaml_config
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT
from src.preprocessing.build_processed_tables import (
    build_swaps_processed_table,
    save_processed_swaps,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run preprocessing for raw swap data"""
    load_environment()
    setup_logging()

    config_path = PROJECT_ROOT / "configs" / "data.yaml"
    config = load_yaml_config(config_path)

    raw_swaps_dir = PROJECT_ROOT / config["output"]["raw_swaps_dir"]
    processed_output_file = PROJECT_ROOT / "data" / "processed" / "swaps_clean.parquet"

    logger.info("Loading raw swaps from: %s", raw_swaps_dir)
    processed_dataframe = build_swaps_processed_table(raw_swaps_dir)

    logger.info("Processed dataframe shape: %s", processed_dataframe.shape)

    saved_path = save_processed_swaps(processed_dataframe, processed_output_file)

    logger.info("Saved processed swaps to: %s", saved_path)


if __name__ == "__main__":
    main()