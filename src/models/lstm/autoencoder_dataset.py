from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.lstm.dataset import _coerce_sequence_to_float32_array


@dataclass
class SequenceNormalizer:
    """
    Feature-wise normalizer for 3D sequence data.

    mean/std are 1D arrays of length F.
    """

    mean: np.ndarray
    std: np.ndarray

    def transform(self, seq: np.ndarray) -> np.ndarray:
        eps = 1e-8
        return (seq - self.mean) / (self.std + eps)


class SequenceAutoencoderDataset(Dataset):
    """
    Dataset returning only sequence features (no labels required).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        normalizer: SequenceNormalizer | None = None,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)

        if "sequence_features" not in self.dataframe.columns:
            raise ValueError("Missing 'sequence_features' column.")
        if "feature_columns" not in self.dataframe.columns:
            raise ValueError("Missing 'feature_columns' column.")

        self.normalizer = normalizer

        if feature_columns is not None:
            # Project to selected columns.
            base_cols = self.dataframe["feature_columns"].iloc[0]
            if not isinstance(base_cols, list):
                base_cols = list(base_cols)
            base_cols = [str(c) for c in base_cols]
            index_map = {c: i for i, c in enumerate(base_cols)}
            keep_idx = [index_map[c] for c in feature_columns if c in index_map]
            if not keep_idx:
                raise ValueError("Requested feature_columns do not exist in this sequence dataset.")

            def _project(mat: Any) -> list[list[float]]:
                arr = _coerce_sequence_to_float32_array(mat)
                return arr[:, keep_idx].tolist()

            self.dataframe = self.dataframe.copy()
            self.dataframe["sequence_features"] = self.dataframe["sequence_features"].apply(_project)
            self.dataframe["feature_columns"] = [feature_columns] * len(self.dataframe)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.dataframe.iloc[index]
        seq = _coerce_sequence_to_float32_array(row["sequence_features"])
        if self.normalizer is not None:
            seq = self.normalizer.transform(seq)
        return torch.tensor(seq, dtype=torch.float32)


def fit_sequence_normalizer(
    dataframe: pd.DataFrame,
    *,
    max_sequences: int = 25000,
) -> SequenceNormalizer:
    """
    Fit mean/std over flattened (N * L, F) from a sample of sequences.
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
    return SequenceNormalizer(mean=mean, std=std)

