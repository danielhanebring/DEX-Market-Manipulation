from __future__ import annotations

import json
import logging

import pandas as pd
import torch

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.models.lstm.predict import predict_lstm_probabilities
from src.models.lstm.train import evaluate_lstm_model, train_lstm_model

logger = logging.getLogger(__name__)


def main() -> None:
    load_environment()
    setup_logging()

    sequence_features_path = PROJECT_ROOT / "data" / "features" / "sequence_features.parquet"
    model_output_path = PROJECT_ROOT / "outputs" / "reports" / "lstm_model.pt"
    metrics_output_path = PROJECT_ROOT / "outputs" / "metrics" / "lstm_metrics.json"
    predictions_output_path = PROJECT_ROOT / "outputs" / "predictions" / "lstm_sequence_predictions.parquet"

    ensure_directory(model_output_path.parent)
    ensure_directory(metrics_output_path.parent)
    ensure_directory(predictions_output_path.parent)

    logger.info("Loading sequence features from: %s", sequence_features_path)
    sequence_df = pd.read_parquet(sequence_features_path)
    logger.info("Sequence dataframe shape: %s", sequence_df.shape)

    artifacts, train_df, validation_df, test_df = train_lstm_model(
        dataframe=sequence_df,
        target_column="target_contains_weak_anomaly",
        hidden_size=64,
        num_layers=1,
        learning_rate=1e-3,
        batch_size=128,
        epochs=5,
    )

    test_metrics = evaluate_lstm_model(
        model=artifacts.model,
        dataframe=test_df,
        target_column="target_contains_weak_anomaly",
        batch_size=128,
    )

    logger.info("Validation metrics: %s", artifacts.validation_metrics)
    logger.info("Test metrics: %s", test_metrics)

    torch.save(artifacts.model.state_dict(), model_output_path)
    logger.info("Saved LSTM model to: %s", model_output_path)

    predictions_df = predict_lstm_probabilities(
        model=artifacts.model,
        dataframe=test_df,
        target_column="target_contains_weak_anomaly",
        batch_size=128,
    )
    predictions_df.to_parquet(predictions_output_path, index=False)
    logger.info("Saved LSTM predictions to: %s", predictions_output_path)

    metrics_payload = {
        "model": "lstm_sequence_classifier",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "validation_metrics": artifacts.validation_metrics,
        "test_metrics": test_metrics,
        "training_history": artifacts.train_history,
    }

    with open(metrics_output_path, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    logger.info("Saved metrics to: %s", metrics_output_path)


if __name__ == "__main__":
    main()