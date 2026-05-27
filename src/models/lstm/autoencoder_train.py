from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from src.models.lstm.autoencoder_dataset import (
    SequenceAutoencoderDataset,
    SequenceNormalizer,
    fit_sequence_normalizer,
)
from src.models.lstm.autoencoder_model import LSTMAutoencoder
from src.models.lstm.train import set_lstm_seed, time_split_sequences


@dataclass
class AutoencoderArtifacts:
    model: LSTMAutoencoder
    normalizer: SequenceNormalizer
    train_history: list[dict]
    validation_metrics: dict


def _reconstruction_errors(
    *,
    model: LSTMAutoencoder,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - errors: shape [N] mean squared error per sequence
    - targets: shape [N] optional targets from dataframe (if present), else zeros
    """
    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)  # [B, L, F]
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=(1, 2))
            errors.extend(mse.detach().cpu().numpy().tolist())
    return np.asarray(errors, dtype=np.float32), None  # targets handled outside


def _pick_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, list[dict]]:
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    candidates: list[dict] = []
    best = {"threshold": thresholds[0], "f1": -1.0}
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


def train_lstm_autoencoder(
    dataframe: pd.DataFrame,
    *,
    hidden_size: int = 64,
    num_layers: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 8,
    device: str | None = None,
    seed: int = 42,
    # If provided, we evaluate against this label column (weak supervision) but do not train on it.
    eval_target_column: str = "target_contains_weak_anomaly",
) -> tuple[AutoencoderArtifacts, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_lstm_seed(seed)

    train_df, val_df, test_df = time_split_sequences(dataframe)

    # Fit normalizer on training sequences.
    normalizer = fit_sequence_normalizer(train_df)

    train_ds = SequenceAutoencoderDataset(train_df, normalizer=normalizer)
    val_ds = SequenceAutoencoderDataset(val_df, normalizer=normalizer)
    test_ds = SequenceAutoencoderDataset(test_df, normalizer=normalizer)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    sample = train_ds[0]
    input_size = int(sample.shape[1])
    model = LSTMAutoencoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(resolved_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="mean")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(resolved_device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        # Compute validation reconstruction errors.
        val_errors, _ = _reconstruction_errors(model=model, loader=val_loader, device=resolved_device)
        # Normalize to [0,1] by percentile rank for threshold selection.
        val_scores = pd.Series(val_errors).rank(pct=True).to_numpy(dtype=np.float32)

        if eval_target_column in val_df.columns:
            y_true = pd.to_numeric(val_df[eval_target_column], errors="coerce").fillna(0).astype(int).to_numpy()
            thr, candidates = _pick_threshold_by_f1(y_true, val_scores)
            y_pred = (val_scores >= thr).astype(int)
            val_metrics = {
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "pr_auc": float(average_precision_score(y_true, val_scores)) if len(np.unique(y_true)) > 1 else None,
                "support": int(len(y_true)),
                "positive_rate": float(np.mean(y_true == 1)),
                "predicted_positive_rate": float(np.mean(y_pred == 1)),
                "selected_threshold": float(thr),
                "threshold_candidates": candidates,
            }
        else:
            val_metrics = {"note": f"Missing eval_target_column={eval_target_column}"}

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss_mean": float(np.mean(losses)) if losses else None,
                **val_metrics,
            }
        )

    # Final validation metrics (from last epoch history, if present)
    final_val_metrics = history[-1] if history else {}

    artifacts = AutoencoderArtifacts(
        model=model,
        normalizer=normalizer,
        train_history=history,
        validation_metrics=final_val_metrics,
    )

    return artifacts, train_df, val_df, test_df


def score_lstm_autoencoder(
    *,
    artifacts: AutoencoderArtifacts,
    dataframe: pd.DataFrame,
    batch_size: int = 256,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Return dataframe with:
    - ae_reconstruction_error
    - ae_score_percentile (percentile rank, higher => more anomalous)
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = SequenceAutoencoderDataset(dataframe, normalizer=artifacts.normalizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    errors, _ = _reconstruction_errors(model=artifacts.model, loader=loader, device=resolved_device)
    scores = pd.Series(errors).rank(pct=True).to_numpy(dtype=np.float32)

    out = dataframe.reset_index(drop=True).copy()
    out["ae_reconstruction_error"] = errors
    out["ae_score_percentile"] = scores
    return out

