from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    PyTorch dataset for sequence-level anomaly classification.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "target_contains_weak_anomaly",
        *,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.target_column = target_column
        self.mean = mean
        self.std = std

        if target_column not in self.dataframe.columns:
            raise ValueError(f"Missing target column: {target_column}")

        if "sequence_features" not in self.dataframe.columns:
            raise ValueError("Missing 'sequence_features' column.")

        self.dataframe = self.dataframe[self.dataframe[target_column].notna()].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]

        sequence_features = row["sequence_features"]
        sequence_array = _coerce_sequence_to_float32_array(sequence_features)

        if self.mean is not None and self.std is not None:
            eps = np.float32(1e-8)
            sequence_array = (sequence_array - self.mean) / (self.std + eps)

        if sequence_array.ndim != 2:
            raise ValueError(
                f"Expected 2D sequence array, got shape {sequence_array.shape} "
                f"for row index {index}"
            )

        target_value = np.float32(row[self.target_column])

        features_tensor = torch.tensor(sequence_array, dtype=torch.float32)
        target_tensor = torch.tensor(target_value, dtype=torch.float32)

        return features_tensor, target_tensor


def _coerce_sequence_to_float32_array(sequence_features: Any) -> np.ndarray:
    """
    Convert stored sequence features into a strict float32 2D NumPy array.
    """
    if isinstance(sequence_features, str):
        sequence_features = ast.literal_eval(sequence_features)

    if isinstance(sequence_features, np.ndarray):
        if sequence_features.dtype != object:
            return sequence_features.astype(np.float32)

        sequence_features = sequence_features.tolist()

    if isinstance(sequence_features, list):
        cleaned_rows: list[list[float]] = []

        for row in sequence_features:
            if isinstance(row, np.ndarray):
                row = row.tolist()

            if not isinstance(row, (list, tuple)):
                raise ValueError(
                    f"Sequence row is not list-like. Got type: {type(row)}"
                )

            cleaned_row: list[float] = []
            for value in row:
                if value is None or pd.isna(value):
                    cleaned_row.append(0.0)
                else:
                    cleaned_row.append(float(value))

            cleaned_rows.append(cleaned_row)

        return np.array(cleaned_rows, dtype=np.float32)

    raise ValueError(
        f"Unsupported sequence_features type: {type(sequence_features)}"
    )


def fit_sequence_mean_std(
    dataframe: pd.DataFrame,
    *,
    max_sequences: int = 25000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit feature-wise mean/std on a sample of sequences (flattened over timesteps).
    Returns (mean, std) as float32 arrays of length F.
    """
    df = dataframe
    if len(df) > max_sequences:
        df = df.sample(n=max_sequences, random_state=42).reset_index(drop=True)

    mats = []
    for mat in df["sequence_features"].tolist():
        arr = _coerce_sequence_to_float32_array(mat)
        mats.append(arr)

    big = np.concatenate(mats, axis=0)  # [N*L, F]
    mean = big.mean(axis=0).astype(np.float32)
    std = big.std(axis=0).astype(np.float32)
    std = np.where(std <= 1e-8, 1.0, std).astype(np.float32)
    return mean, std
