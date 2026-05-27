from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.lstm.dataset import SequenceDataset
from src.models.lstm.model import LSTMAnomalyClassifier


def predict_lstm_probabilities(
    model: LSTMAnomalyClassifier,
    dataframe: pd.DataFrame,
    target_column: str = "target_contains_weak_anomaly",
    batch_size: int = 128,
    device: str | None = None,
    threshold: float = 0.5,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Generate sequence level probabilities from a trained LSTM model.
    """
    dataset = SequenceDataset(dataframe, target_column=target_column, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    probabilities: list[float] = []

    with torch.no_grad():
        for batch_features, _ in loader:
            batch_features = batch_features.to(resolved_device)
            logits = model(batch_features)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            probabilities.extend(batch_probabilities.tolist())

    prediction_df = dataset.dataframe.copy()
    prediction_df["lstm_probability"] = np.asarray(probabilities, dtype=np.float32)
    prediction_df["lstm_predicted_flag"] = (prediction_df["lstm_probability"] >= float(threshold)).astype(int)
    prediction_df["lstm_threshold"] = float(threshold)

    return prediction_df
