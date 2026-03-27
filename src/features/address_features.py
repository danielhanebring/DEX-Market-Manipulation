from __future__ import annotations

import pandas as pd


def add_address_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Address repetition and concentration features.
    - same_sender_recent_count_20
    - same_recipient_recent_count_20
    - sender_recipient_pair_recent_count_20
    - same_sender_same_block_count
    - same_recipient_same_block_count
    """
    df = dataframe.copy()

    required_columns = {"sender_address", "recipient_address", "pool_address", "block_number", "swap_id"}
    if not required_columns.issubset(df.columns):
        return df

    df = df.sort_values(["pool_address", "block_number", "log_index", "swap_id"]).reset_index(drop=True)

    df["sender_recipient_pair"] = (
        df["sender_address"].fillna("missing") + "->" + df["recipient_address"].fillna("missing")
    )

    df["same_sender_recent_count_20"] = _rolling_same_value_count(
        dataframe=df,
        group_column="pool_address",
        value_column="sender_address",
        window_size=20,
    )

    df["same_recipient_recent_count_20"] = _rolling_same_value_count(
        dataframe=df,
        group_column="pool_address",
        value_column="recipient_address",
        window_size=20,
    )

    df["sender_recipient_pair_recent_count_20"] = _rolling_same_value_count(
        dataframe=df,
        group_column="pool_address",
        value_column="sender_recipient_pair",
        window_size=20,
    )

    df["same_sender_same_block_count"] = (
        df.groupby(["pool_address", "block_number", "sender_address"])["swap_id"].transform("count")
    )

    df["same_recipient_same_block_count"] = (
        df.groupby(["pool_address", "block_number", "recipient_address"])["swap_id"].transform("count")
    )

    return df.drop(columns=["sender_recipient_pair"])


def _rolling_same_value_count(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
    window_size: int,
) -> pd.Series:
    """
    Count how many times the same value appeared in the recent rolling window,
    excluding the current row.
    """
    result = pd.Series(index=dataframe.index, dtype="Int64")

    for _, group_df in dataframe.groupby(group_column, sort=False):
        values = group_df[value_column].tolist()
        counts: list[int] = []

        for current_index, current_value in enumerate(values):
            start_index = max(0, current_index - window_size)
            previous_window = values[start_index:current_index]
            counts.append(sum(1 for value in previous_window if value == current_value))

        result.loc[group_df.index] = counts

    return result.astype("Int64")