from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import ensure_directory
from src.preprocessing.clean_swaps import clean_swaps
from src.preprocessing.ordering import sort_swaps_deterministically


def load_raw_swap_files(raw_swaps_dir: str | Path) -> pd.DataFrame:
    """
    Load and concatenate all swap parquet files 
    """
    raw_dir = Path(raw_swaps_dir)
    parquet_files = sorted(raw_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No raw swap parquet files found in: {raw_dir}")

    dataframes = [pd.read_parquet(file_path) for file_path in parquet_files]
    return pd.concat(dataframes, ignore_index=True)


def build_swaps_processed_table(raw_swaps_dir: str | Path) -> pd.DataFrame:
    """
    Build the processed swaps table from raw parquet files.
    """
    raw_dataframe = load_raw_swap_files(raw_swaps_dir)
    cleaned_dataframe = clean_swaps(raw_dataframe)
    ordered_dataframe = sort_swaps_deterministically(cleaned_dataframe)

    return ordered_dataframe.reset_index(drop=True)


def save_processed_swaps(dataframe: pd.DataFrame, output_file: str | Path) -> Path:
    """
    Save processed swaps dataframe to parquet.
    """
    output_path = Path(output_file)
    ensure_directory(output_path.parent)
    dataframe.to_parquet(output_path, index=False)
    return output_path