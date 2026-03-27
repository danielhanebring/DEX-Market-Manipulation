from __future__ import annotations

import pandas as pd


def add_gas_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Gas event features:
    - gas_price_gwei (if missing but gas_price_raw exists)
    - gas_price_local_mean_20
    - gas_price_local_median_20
    - gas_price_relative_to_local_mean
    - gas_price_relative_to_local_median
    - gas_spike_flag
    """
    df = dataframe.copy()

    if "gas_price_gwei" not in df.columns and "gas_price_raw" in df.columns:
        df["gas_price_gwei"] = pd.to_numeric(df["gas_price_raw"], errors="coerce") / 1_000_000_000

    if "gas_price_gwei" not in df.columns:
        return df

    df = df.sort_values(["pool_address", "block_number", "log_index", "swap_id"]).reset_index(drop=True)

    grouped = df.groupby("pool_address")["gas_price_gwei"]

    df["gas_price_local_mean_20"] = (
        grouped.rolling(window=20, min_periods=3).mean().reset_index(level=0, drop=True)
    )

    df["gas_price_local_median_20"] = (
        grouped.rolling(window=20, min_periods=3).median().reset_index(level=0, drop=True)
    )

    df["gas_price_relative_to_local_mean"] = (
        df["gas_price_gwei"] / df["gas_price_local_mean_20"]
    )

    df["gas_price_relative_to_local_median"] = (
        df["gas_price_gwei"] / df["gas_price_local_median_20"]
    )

    df["gas_spike_flag"] = (
        df["gas_price_relative_to_local_median"].fillna(1.0) > 1.5
    ).astype(int)

    return df