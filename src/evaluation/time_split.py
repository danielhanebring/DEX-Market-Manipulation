from __future__ import annotations

import pandas as pd


def time_based_split(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset by time (chronological split).
    """
    df = dataframe.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(df) * train_ratio)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df
