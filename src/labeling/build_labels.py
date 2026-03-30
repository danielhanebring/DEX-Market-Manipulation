from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import ensure_directory
from src.labeling.sandwich_rules import (
    SandwichRuleConfig,
    detect_sandwich_candidates,
)


def build_sandwich_labels(
    features_dataframe: pd.DataFrame,
    config: SandwichRuleConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build sandwich-specific event and sequence labels from event features.
    """
    return detect_sandwich_candidates(
        dataframe=features_dataframe,
        config=config,
    )


def save_label_table(dataframe: pd.DataFrame, output_file: str | Path) -> Path:
    """
    Save any label dataframe to parquet.
    """
    output_path = Path(output_file)
    ensure_directory(output_path.parent)
    dataframe.to_parquet(output_path, index=False)
    return output_path