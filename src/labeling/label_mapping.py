from __future__ import annotations

import pandas as pd


def add_binary_model_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary target for experiments that need it.

    Mapping:
    - weak_anomaly -> 1
    - normal -> 0
    - suspicious -> <NA> (excluded by default)
    - unlabeled -> <NA>
    """
    df = dataframe.copy()

    mapping = {
        "weak_anomaly": 1,
        "normal": 0,
    }

    df["binary_target"] = df["label_class"].map(mapping).astype("Float64")
    return df