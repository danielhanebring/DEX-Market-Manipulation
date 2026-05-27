from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.lstm.dataset import SequenceDataset, fit_sequence_mean_std
from src.models.lstm.model import LSTMAnomalyClassifier


@dataclass
class TrainingArtifacts:
    model: LSTMAnomalyClassifier
    train_history: list[dict]
    validation_metrics: dict


def set_lstm_seed(seed: int = 42) -> None:
    """
    Set random seeds for repeatable runs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _pick_threshold_by_f1(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float] | None = None,
) -> tuple[float, list[dict]]:
    """
    Choose a probability threshold on validation data by maximizing F1.
    Returns selected_threshold, and a list of candidate rows.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    candidates: list[dict] = []
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        y_pred = (y_score >= float(t)).astype(int)
        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f = float(f1_score(y_true, y_pred, zero_division=0))
        row = {"threshold": float(t), "precision": p, "recall": r, "f1": f}
        candidates.append(row)
        if f > best["f1"]:
            best = row

    return float(best["threshold"]), candidates


def _compute_pos_weight(y: np.ndarray) -> float:
    """
    For BCEWithLogitsLoss(pos_weight=...), a typical choice is:
      pos_weight = (num_negative / num_positive)
    """
    y = np.asarray(y, dtype=np.float32)
    pos = float(np.sum(y >= 0.5))
    neg = float(np.sum(y < 0.5))
    if pos <= 0:
        return 1.0
    return max(neg / pos, 1.0)


def train_lstm_model(
    dataframe: pd.DataFrame,
    target_column: str = "target_contains_weak_anomaly",
    hidden_size: int = 64,
    num_layers: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 8,
    device: str | None = None,
    seed: int = 42,
    use_weighted_sampler: bool = True,
    dropout: float = 0.1,
    early_stopping_patience: int = 3,
) -> tuple[TrainingArtifacts, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train an LSTM model on sequence data.

    Notes vs earlier version:
    - Handles class imbalance via `pos_weight` and optional weighted sampling.
    - Shuffles the training data (within the time-split).
    - Returns validation metrics for the *selected* threshold.
    """
    set_lstm_seed(seed)
    train_df, validation_df, test_df = time_split_sequences(dataframe)

    # Normalize features using mean/std fitted on the training split only.
    mean, std = fit_sequence_mean_std(train_df)

    train_dataset = SequenceDataset(train_df, target_column=target_column, mean=mean, std=std)
    validation_dataset = SequenceDataset(validation_df, target_column=target_column, mean=mean, std=std)

    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training or validation dataset is empty after target filtering.")

    # Class imbalance handling:
    # - pos_weight affects the loss (penalizes false negatives more)
    # - weighted sampler increases the chance to see positive sequences during training
    train_targets = train_dataset.dataframe[target_column].to_numpy(dtype=np.float32)
    pos_weight_value = _compute_pos_weight(train_targets)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    sampler = None
    if use_weighted_sampler:
        weights = np.where(train_targets >= 0.5, pos_weight_value, 1.0).astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights),
            num_samples=len(weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    sample_features, _ = train_dataset[0]
    input_size = sample_features.shape[1]

    model = LSTMAnomalyClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(resolved_device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=resolved_device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    history: list[dict] = []
    best_metric = -1.0
    best_state = None
    bad_epochs = 0

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
            mean=mean,
            std=std,
        )
        # Use PR-AUC for early stopping if available, otherwise fall back to F1.
        score = validation_metrics.get("pr_auc")
        if score is None:
            score = validation_metrics.get("f1", 0.0)
        score = float(score) if score is not None else 0.0
        scheduler.step(score)

        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss_mean": float(np.mean(epoch_losses)) if epoch_losses else None,
                "val_selection_metric": score,
                **validation_metrics,
            }
        )

        if bad_epochs >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_validation_metrics = evaluate_lstm_model(
        model=model,
        dataframe=validation_df,
        target_column=target_column,
        batch_size=batch_size,
        device=resolved_device,
        mean=mean,
        std=std,
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
    threshold: float | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict:
    """
    Evaluate trained LSTM on a sequence dataframe split.
    """
    dataset = SequenceDataset(dataframe, target_column=target_column, mean=mean, std=std)
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
    positive_rate = float(np.mean(y_true >= 0.5)) if len(y_true) else 0.0

    if threshold is None:
        selected_threshold, threshold_candidates = _pick_threshold_by_f1(
            y_true=y_true.astype(int),
            y_score=y_score,
        )
    else:
        selected_threshold = float(threshold)
        threshold_candidates = []

    y_pred = (y_score >= selected_threshold).astype(int)
    predicted_positive_rate = float(np.mean(y_pred == 1)) if len(y_pred) else 0.0

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else None,
        "support": int(len(y_true)),
        "positive_rate": positive_rate,
        "predicted_positive_rate": predicted_positive_rate,
        "selected_threshold": selected_threshold,
        "threshold_candidates": threshold_candidates,
    }

    return metrics
