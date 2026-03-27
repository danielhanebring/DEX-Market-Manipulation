from __future__ import annotations

import pandas as pd


def clean_swaps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data and returns a cleaned dataframe
    """
    df = dataframe.copy()

    df = _drop_duplicate_swaps(df)
    df = _cast_numeric_columns(df)
    df = _add_datetime_columns(df)
    df = _add_convenience_columns(df)

    return df.reset_index(drop=True)


def _drop_duplicate_swaps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate swap rows using swap_id as primary unique key.
    """
    if "swap_id" not in dataframe.columns:
        return dataframe.copy()

    return dataframe.drop_duplicates(subset=["swap_id"], keep="first").reset_index(drop=True)


def _cast_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known numeric columns to numeric dtypes where possible.
    """
    df = dataframe.copy()

    integer_columns = [
        "log_index",
        "block_number",
        "timestamp",
        "tick",
        "token0_decimals",
        "token1_decimals",
        "fee_tier",
    ]

    float_columns = [
        "amount0",
        "amount1",
    ]

    optional_float_columns = [
        "gas_price_raw",
        "sqrt_price_x96_raw",
    ]

    for column in integer_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    for column in float_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in optional_float_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def _add_datetime_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add UTC datetime columns from the event timestamp.
    """
    df = dataframe.copy()

    if "timestamp" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
        df["date_utc"] = df["datetime_utc"].dt.date.astype("string")

    return df


def _add_convenience_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns useful for early analysis and features.
    """
    df = dataframe.copy()

    if "amount0" in df.columns:
        df["abs_amount0"] = df["amount0"].abs()
        df["trade_direction_token0"] = df["amount0"].apply(_direction_from_value)

    if "amount1" in df.columns:
        df["abs_amount1"] = df["amount1"].abs()
        df["trade_direction_token1"] = df["amount1"].apply(_direction_from_value)

    if "gas_price_raw" in df.columns:
        df["gas_price_gwei"] = df["gas_price_raw"] / 1_000_000_000

    return df


def _direction_from_value(value: float | int | None) -> str | None:
    """
    Convert signed trade amount into a simple direction label.
    """
    if pd.isna(value):
        return None
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"