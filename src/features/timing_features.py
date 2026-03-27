from __future__ import annotations

import pandas as pd


def add_timing_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Timing features;
    - interarrival_seconds
    - same_block_event_count
    - same_block_pattern_flag
    - local_event_density_10
    - time_since_last_swap_in_pool
    """
    df = dataframe.copy()

    if "timestamp" not in df.columns:
        return df

    df = df.sort_values(["pool_address", "block_number", "log_index", "swap_id"]).reset_index(drop=True)

    df["time_since_last_swap_in_pool"] = (
        df.groupby("pool_address")["timestamp"].diff()
    )

    df["interarrival_seconds"] = df["time_since_last_swap_in_pool"]

    if "block_number" in df.columns:
        df["same_block_event_count"] = (
            df.groupby(["pool_address", "block_number"])["swap_id"].transform("count")
        )
        df["same_block_pattern_flag"] = (df["same_block_event_count"] > 1).astype(int)
    else:
        df["same_block_event_count"] = 1
        df["same_block_pattern_flag"] = 0

    df["local_event_density_10"] = (
        df.groupby("pool_address")["timestamp"]
        .rolling(window=10, min_periods=1)
        .apply(_density_from_window, raw=False)
        .reset_index(level=0, drop=True)
    )

    return df


def _density_from_window(window: pd.Series) -> float:
    """
    Approximate local event density as events per second in the rolling window.
    """
    if len(window) <= 1:
        return 0.0

    time_span = float(window.iloc[-1] - window.iloc[0])
    if time_span <= 0:
        return float(len(window))

    return float(len(window) / time_span)