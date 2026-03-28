from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from src.models.lstm.dataset import SequenceDataset
from src.models.lstm.model import LSTMAnomalyClassifier


@dataclass
class TrainingArtifacts:
    model: LSTMAnomalyClassifier
    train_history: list[dict]
    validation_metrics: dict


def time_split_sequences(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time based split for sequence data using window_end_time.
    """
    df = dataframe.sort_values("window_end_time").reset_index(drop=True)

    n_total = len(df)
    train_end = int(n_total * train_ratio)
    validation_end = int(n_total * (train_ratio + validation_ratio))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def train_lstm_model(
    dataframe: pd.DataFrame,
    target_column: str = "target_contains_weak_anomaly",
    hidden_size: int = 64,
    num_layers: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 5,
    device: str | None = None,
) -> tuple[TrainingArtifacts, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train a first simple LSTM model on sequence data.
    """
    train_df, validation_df, test_df = time_split_sequences(dataframe)

    train_dataset = SequenceDataset(train_df, target_column=target_column)
    validation_dataset = SequenceDataset(validation_df, target_column=target_column)

    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training or validation dataset is empty after target filtering.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    sample_features, _ = train_dataset[0]
    input_size = sample_features.shape[1]

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMAnomalyClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(resolved_device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        validation_metrics = evaluate_lstm_model(
            model=model,
            dataframe=validation_df,
            target_column=target_column,
            batch_size=batch_size,
            device=resolved_device,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else None,
                **validation_metrics,
            }
        )

    final_validation_metrics = evaluate_lstm_model(
        model=model,
        dataframe=validation_df,
        target_column=target_column,
        batch_size=batch_size,
        device=resolved_device,
    )

    artifacts = TrainingArtifacts(
        model=model,
        train_history=history,
        validation_metrics=final_validation_metrics,
    )

    return artifacts, train_df, validation_df, test_df


def evaluate_lstm_model(
    model: LSTMAnomalyClassifier,
    dataframe: pd.DataFrame,
    target_column: str = "target_contains_weak_anomaly",
    batch_size: int = 128,
    device: str | None = None,
) -> dict:
    """
    Evaluate trained LSTM on a sequence dataframe split.
    """
    dataset = SequenceDataset(dataframe, target_column=target_column)
    if len(dataset) == 0:
        return {"error": "No valid rows available for evaluation."}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    all_targets: list[float] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(resolved_device)
            logits = model(batch_features)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            all_probabilities.extend(probabilities.tolist())
            all_targets.extend(batch_targets.numpy().tolist())

    y_true = np.asarray(all_targets, dtype=np.float32)
    y_score = np.asarray(all_probabilities, dtype=np.float32)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else None,
        "support": int(len(y_true)),
    }

    return metrics