from __future__ import annotations

"""
Run an unsupervised LSTM autoencoder on sequences.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow `from src...` when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_environment  # noqa: E402
from src.common.logging_utils import setup_logging  # noqa: E402
from src.common.paths import PROJECT_ROOT, ensure_directory  # noqa: E402
from src.models.lstm.autoencoder_train import score_lstm_autoencoder, train_lstm_autoencoder  # noqa: E402
from src.models.lstm.dataset import _coerce_sequence_to_float32_array  # noqa: E402

logger = logging.getLogger(__name__)

BOT_JARED = "0xae2fc483527b8ef99eb5d9b44875f005ba1fae13"

SEQUENCE_BLOCK_CONTEXT_FEATURES = [
    "same_block_event_count",
    "local_event_density_10",
]


def _lower_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.lower()


def _compute_jared_swap_ids(event_df: pd.DataFrame) -> set[str]:
    required = ["swap_id", "sender_address", "recipient_address"]
    missing = [c for c in required if c not in event_df.columns]
    if missing:
        raise ValueError(f"event_features missing required columns: {missing}")

    sender = _lower_series(event_df["sender_address"])
    recipient = _lower_series(event_df["recipient_address"])
    mask = (sender == BOT_JARED) | (recipient == BOT_JARED)

    if "origin_address" in event_df.columns:
        origin = _lower_series(event_df["origin_address"])
        mask = mask | (origin == BOT_JARED)

    return set(event_df.loc[mask, "swap_id"].astype("string").tolist())


def _project_sequence_df(sequence_df: pd.DataFrame, *, keep_features: list[str]) -> pd.DataFrame:
    """
    Project sequence_features to a subset (keeps shape [L x F]).
    """
    df = sequence_df.copy()
    base_cols = df["feature_columns"].iloc[0]
    if not isinstance(base_cols, list):
        base_cols = list(base_cols)
    base_cols = [str(c) for c in base_cols]
    idx = {c: i for i, c in enumerate(base_cols)}
    keep_idx = [idx[c] for c in keep_features if c in idx]
    if not keep_idx:
        raise ValueError("Projection would produce zero features.")

    def _proj(mat):
        arr = _coerce_sequence_to_float32_array(mat)
        return arr[:, keep_idx].tolist()

    df["sequence_features"] = df["sequence_features"].apply(_proj)
    df["feature_columns"] = [keep_features] * len(df)
    return df


def main() -> int:
    load_environment()
    setup_logging()

    ensure_directory(PROJECT_ROOT / "outputs" / "metrics")
    ensure_directory(PROJECT_ROOT / "outputs" / "reports")
    ensure_directory(PROJECT_ROOT / "outputs" / "predictions")

    seq_path = PROJECT_ROOT / "data" / "features" / "sequence_features.parquet"
    if not seq_path.exists():
        raise SystemExit(f"Missing: {seq_path}")

    logger.info("Loading sequence features: %s", seq_path)
    seq_df_base = pd.read_parquet(seq_path)
    logger.info("sequence_features shape: %s", seq_df_base.shape)

    base_cols = seq_df_base["feature_columns"].iloc[0]
    if not isinstance(base_cols, list):
        base_cols = list(base_cols)
    base_cols = [str(c) for c in base_cols]
    logger.info("Sequence feature columns (dims=%s): %s", len(base_cols), base_cols)

    # Detect Jared swap ids for filtering sequences.
    event_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    if not event_path.exists():
        raise SystemExit(f"Missing: {event_path}")
    event_min = pd.read_parquet(event_path, columns=["swap_id", "sender_address", "recipient_address", "origin_address"])
    jared_swap_ids = _compute_jared_swap_ids(event_min)
    logger.info("Detected Jared swap_ids: %s", len(jared_swap_ids))

    def _contains_jared(ids: list) -> bool:
        return any(str(x) in jared_swap_ids for x in ids)

    seq_df_base = seq_df_base.copy()
    seq_df_base["contains_jared_swap"] = seq_df_base["source_event_ids"].apply(_contains_jared)

    experiments = [
        {"name": "with_jared_full", "include_jared": True, "ablation": "full"},
        {"name": "without_jared_full", "include_jared": False, "ablation": "full"},
        {"name": "with_jared_no_block_context", "include_jared": True, "ablation": "no_block_context"},
        {"name": "without_jared_no_block_context", "include_jared": False, "ablation": "no_block_context"},
    ]

    for exp in experiments:
        name = exp["name"]
        include_jared = bool(exp["include_jared"])
        ablation = str(exp["ablation"])

        logger.info("=== LSTM-AE experiment: %s ===", name)

        metrics_path = PROJECT_ROOT / "outputs" / "metrics" / f"lstm_ae_{name}_metrics.json"
        pred_path = PROJECT_ROOT / "outputs" / "predictions" / f"lstm_ae_{name}_test_predictions.parquet"
        jared_path = PROJECT_ROOT / "outputs" / "predictions" / f"lstm_ae_{name}_jared_scored.parquet"
        model_path = PROJECT_ROOT / "outputs" / "reports" / f"lstm_ae_{name}.pt"

        if metrics_path.exists() and pred_path.exists() and jared_path.exists() and model_path.exists():
            logger.info("Outputs already exist for %s; skipping.", name)
            continue

        seq_df = seq_df_base.copy()
        if not include_jared:
            seq_df = seq_df[~seq_df["contains_jared_swap"]].copy()

        keep_features = base_cols
        if ablation == "no_block_context":
            keep_features = [c for c in base_cols if c not in SEQUENCE_BLOCK_CONTEXT_FEATURES]
            seq_df = _project_sequence_df(seq_df, keep_features=keep_features)

        artifacts, train_df, val_df, test_df = train_lstm_autoencoder(
            seq_df,
            hidden_size=64,
            num_layers=1,
            learning_rate=1e-3,
            batch_size=256,
            epochs=8,
            seed=42,
        )

        # Selected threshold in validation_metrics is percentile threshold (0..1)
        thr = float(artifacts.validation_metrics.get("selected_threshold", 0.95))

        test_scored = score_lstm_autoencoder(artifacts=artifacts, dataframe=test_df)
        y_true = pd.to_numeric(test_scored["target_contains_weak_anomaly"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_score = pd.to_numeric(test_scored["ae_score_percentile"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y_pred = (y_score >= thr).astype(int)

        from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

        test_metrics = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
            "support": int(len(y_true)),
            "positive_rate": float(np.mean(y_true == 1)),
            "predicted_positive_rate": float(np.mean(y_pred == 1)),
            "selected_threshold": thr,
        }

        # Save model artifact
        torch.save(
            {
                "state_dict": artifacts.model.state_dict(),
                "normalizer_mean": artifacts.normalizer.mean,
                "normalizer_std": artifacts.normalizer.std,
                "feature_columns": keep_features,
                "sequence_length": 20,
                "hidden_size": 64,
                "num_layers": 1,
                "selected_threshold": thr,
            },
            model_path,
        )

        # Save predictions
        out = test_scored.copy()
        out["ae_predicted_flag"] = (out["ae_score_percentile"] >= thr).astype(int)
        out.to_parquet(pred_path, index=False)

        # Jared scoring (unlabeled)
        score_df = seq_df_base[seq_df_base["contains_jared_swap"]].copy()
        if ablation == "no_block_context":
            score_df = _project_sequence_df(score_df, keep_features=keep_features)
        jared_scored = score_lstm_autoencoder(artifacts=artifacts, dataframe=score_df)
        jared_scored["ae_threshold"] = thr
        jared_scored.to_parquet(jared_path, index=False)

        metrics = {
            "model": "lstm_autoencoder_unsupervised",
            "experiment": name,
            "include_jared_in_training": include_jared,
            "sequence_ablation": ablation,
            "feature_dim": len(keep_features),
            "sequence_length": 20,
            "validation_metrics": artifacts.validation_metrics,
            "test_metrics": test_metrics,
            "training_history": artifacts.train_history,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        logger.info("Wrote: %s", metrics_path)
        logger.info("Wrote: %s", pred_path)
        logger.info("Wrote: %s", jared_path)
        logger.info("Wrote: %s", model_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
