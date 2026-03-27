from __future__ import annotations

import pandas as pd


def sort_swaps_deterministically(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Sort swap records by following prio:
    1. block_number
    2. transcation_index
    3. log_index
    4. timestamp
    5. swap_id
    """
    sort_columns: list[str] = []

    for column in [
        "block_number",
        "transaction_index",
        "log_index",
        "timestamp",
        "swap_id",
    ]:
        if column in dataframe.columns:
            sort_columns.append(column)

    if not sort_columns:
        return dataframe.copy()

    return dataframe.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)