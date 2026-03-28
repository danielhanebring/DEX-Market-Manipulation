from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import ensure_directory
from src.features.sequence_builder import build_sliding_window_sequences


def build_sequence_feature_table(
    dataframe: pd.DataFrame,
    sequence_length: int = 20,
    step_size: int = 1,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build sequence level feature table from labeled event level data.
    """
    return build_sliding_window_sequences(
        dataframe=dataframe,
        sequence_length=sequence_length,
        feature_columns=feature_columns,
        step_size=step_size,
    )


def save_sequence_features(dataframe: pd.DataFrame, output_file: str | Path) -> Path:
    output_path = Path(output_file)
    ensure_directory(output_path.parent)
    dataframe.to_parquet(output_path, index=False)
    return output_path