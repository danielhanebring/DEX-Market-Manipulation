from __future__ import annotations

"""
Run LSTM experiments on the sequence dataset.

We use time-based splits and save metrics + model files.
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

from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.features.sequence_builder import build_sliding_window_sequences
from src.models.lstm.dataset import _coerce_sequence_to_float32_array
from src.models.lstm.predict import predict_lstm_probabilities
from src.models.lstm.dataset import fit_sequence_mean_std
from src.models.lstm.train import evaluate_lstm_model, train_lstm_model

logger = logging.getLogger(__name__)

BOT_JARED = "0xae2fc483527b8ef99eb5d9b44875f005ba1fae13"
EXPERIMENT_VERSION = "v2_norm_pr_auc_earlystop"

# Rule-like features (used in some ablations).
RULE_SIGNATURE_FEATURES = [
    "same_block_pattern_flag",
    "strict_sandwich_support_flag",
    "three_event_pattern_indicator",
    "same_origin_before_after_flag",
]

# The default sequence dataset has a small feature set (12 dims).
# We also test a variant without the strongest block-context features.
SEQUENCE_BLOCK_CONTEXT_FEATURES = [
    "same_block_event_count",
    "local_event_density_10",
]

# Features to drop in the no_leakage run.
NO_LEAKAGE_FEATURES_TO_DROP = [
    # Rule-like features
    "same_block_pattern_flag",
    "same_origin_before_after_flag",
    "three_event_pattern_indicator",
    "strict_sandwich_support_flag",
    "sandwich_support_score",
    # Neighbor/triple flags
    "same_sender_before_after_flag",
    "different_middle_sender_from_neighbors_flag",
    "reversal_pattern_flag",
    "tick_change_before",
    "tick_change_after",
    "combined_reversal_magnitude",
    # Size ratios from neighbors
    "relative_trade_size_token0",
    "attacker_vs_victim_size_ratio_token0",
    "relative_trade_size_token1",
    "attacker_vs_victim_size_ratio_token1",
    # Helper flags
    "high_block_gas_context_flag",
    "high_relative_trade_size_flag",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _lower_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.lower()


def _compute_jared_swap_ids(event_df: pd.DataFrame) -> set[str]:
    """
    Find swap_ids where Jared is involved.
    """
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


def _build_sequence_table_from_events(
    *,
    merged_event_df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = 20,
    step_size: int = 1,
) -> pd.DataFrame:
    """
    Build sliding-window sequences from an event dataframe that already includes labels.
    """
    # The sequence builder coerces missing columns away. To keep it strict for experiments,
    # we pass only the columns that exist in the dataframe.
    available = [c for c in feature_columns if c in merged_event_df.columns]
    if not available:
        raise ValueError("None of the requested feature columns exist in the merged event dataframe.")

    seq_df = build_sliding_window_sequences(
        dataframe=merged_event_df,
        sequence_length=sequence_length,
        feature_columns=available,
        step_size=step_size,
    )
    return seq_df


def _project_sequence_features(
    sequence_df: pd.DataFrame,
    *,
    keep_feature_columns: list[str],
) -> pd.DataFrame:
    """
    Project `sequence_features` down to a subset of feature columns.

    Used for ablations on a precomputed sequence dataset.
    """
    df = sequence_df.copy()
    if df.empty:
        return df

    if "feature_columns" not in df.columns or "sequence_features" not in df.columns:
        raise ValueError("sequence dataframe missing required columns: feature_columns / sequence_features")

    base_cols = df["feature_columns"].iloc[0]
    if not isinstance(base_cols, list):
        base_cols = list(base_cols)
    base_cols = [str(c) for c in base_cols]

    index_map = {c: i for i, c in enumerate(base_cols)}
    keep_indices = [index_map[c] for c in keep_feature_columns if c in index_map]
    if not keep_indices:
        raise ValueError("Projection would result in zero sequence feature columns.")

    def _project_matrix(mat):
        # Use the same conversion as the dataset loader.
        arr = _coerce_sequence_to_float32_array(mat)
        if arr.ndim != 2:
            return arr.tolist() if hasattr(arr, "tolist") else mat
        return arr[:, keep_indices].tolist()

    df["sequence_features"] = df["sequence_features"].apply(_project_matrix)
    df["feature_columns"] = [keep_feature_columns] * len(df)
    return df


def _score_jared_sequences(
    *,
    model,
    sequence_df_all_events: pd.DataFrame,
    jared_swap_ids: set[str],
    threshold: float,
    output_path: Path,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> None:
    """
    Score only the sequences that *contain* at least one Jared swap_id.
    This is an unlabeled ranking output; we do not apply sandwich rules here.
    """
    if sequence_df_all_events.empty:
        logger.warning("Sequence table empty; skipping Jared scoring.")
        return

    def _contains_jared(ids: list) -> bool:
        # ids is stored as a list; we keep it simple.
        return any(str(x) in jared_swap_ids for x in ids)

    mask = sequence_df_all_events["source_event_ids"].apply(_contains_jared)
    jared_seq = sequence_df_all_events[mask].copy()
    if jared_seq.empty:
        logger.warning("No sequences contained Jared events; skipping Jared scoring.")
        return

    scored = predict_lstm_probabilities(
        model=model,
        dataframe=jared_seq,
        target_column="target_contains_weak_anomaly",  # exists but is not used semantically for Jared evaluation
        threshold=threshold,
        batch_size=128,
        mean=mean,
        std=std,
    )
    ensure_directory(output_path.parent)
    scored.to_parquet(output_path, index=False)
    logger.info("Wrote Jared scored sequences: %s (rows=%s)", output_path, len(scored))


def main() -> int:
    load_environment()
    setup_logging()

    ensure_directory(PROJECT_ROOT / "outputs" / "metrics")
    ensure_directory(PROJECT_ROOT / "outputs" / "reports")
    ensure_directory(PROJECT_ROOT / "outputs" / "predictions")

    # Load the precomputed sequence dataset (used for LSTM).
    # We default to the base L=20 dataset, but allow using variants built by
    # scripts/build_sequence_feature_variants.py via env var LSTM_SEQUENCE_PATH.
    sequence_features_path = Path(
        str(
            (PROJECT_ROOT / "data" / "features" / "sequence_features_L20_base.parquet")
        )
    )
    env_override = None
    experiment_suffix = ""
    try:
        import os

        env_override = os.environ.get("LSTM_SEQUENCE_PATH")
        experiment_suffix = str(os.environ.get("LSTM_EXPERIMENT_SUFFIX") or "")
    except Exception:
        env_override = None
    if env_override:
        sequence_features_path = Path(env_override)
    if not sequence_features_path.exists():
        raise SystemExit(
            f"Missing: {sequence_features_path}. Build it with: py -3 scripts\\run_sequence_building.py"
        )

    logger.info("Loading sequence features from: %s", sequence_features_path)
    seq_df_base = pd.read_parquet(sequence_features_path)
    logger.info("sequence_features shape: %s", seq_df_base.shape)

    # Normalize suffix formatting.
    experiment_suffix = experiment_suffix.strip()
    if experiment_suffix and not experiment_suffix.startswith("_"):
        experiment_suffix = "_" + experiment_suffix

    # Infer sequence length from the stored matrices (don’t hardcode 20, since we test L=10/L=40 etc).
    inferred_sequence_length = None
    if not seq_df_base.empty and "sequence_features" in seq_df_base.columns:
        try:
            inferred_sequence_length = int(
                _coerce_sequence_to_float32_array(seq_df_base["sequence_features"].iloc[0]).shape[0]
            )
        except Exception:
            inferred_sequence_length = None
    if inferred_sequence_length is None:
        inferred_sequence_length = 20
    logger.info("Inferred sequence_length=%s from dataset", inferred_sequence_length)

    # Compute Jared swap_ids from event_features (minimal columns).
    event_features_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    if not event_features_path.exists():
        raise SystemExit(f"Missing: {event_features_path}")

    logger.info("Loading event features (minimal) to detect Jared swaps...")
    event_min_cols = ["swap_id", "sender_address", "recipient_address"]
    # origin_address is optional
    try:
        tmp_cols = pd.read_parquet(event_features_path, columns=["swap_id", "origin_address"]).columns
        if "origin_address" in tmp_cols:
            event_min_cols.append("origin_address")
    except Exception:
        pass
    event_df_min = pd.read_parquet(event_features_path, columns=list(dict.fromkeys(event_min_cols)))
    jared_swap_ids = _compute_jared_swap_ids(event_df_min)
    logger.info("Detected Jared swap_ids in event_features: %s", len(jared_swap_ids))

    # Mark sequences that contain any Jared swap.
    def _contains_jared(ids: list) -> bool:
        return any(str(x) in jared_swap_ids for x in ids)

    seq_df_base = seq_df_base.copy()
    seq_df_base["contains_jared_swap"] = seq_df_base["source_event_ids"].apply(_contains_jared)

    # We'll use the full sequence table for Jared scoring outputs.
    seq_all_events = seq_df_base

    base_feature_columns = seq_df_base["feature_columns"].iloc[0]
    if not isinstance(base_feature_columns, list):
        base_feature_columns = list(base_feature_columns)
    base_feature_columns = [str(c) for c in base_feature_columns]
    logger.info("Sequence feature columns (dims=%s): %s", len(base_feature_columns), base_feature_columns)

    experiments = [
        {"name": "with_jared_full", "include_jared": True, "ablation": "full"},
        {"name": "without_jared_full", "include_jared": False, "ablation": "full"},
        {"name": "with_jared_no_rule", "include_jared": True, "ablation": "no_rule"},
        {"name": "without_jared_no_rule", "include_jared": False, "ablation": "no_rule"},
        {"name": "with_jared_no_block_context", "include_jared": True, "ablation": "no_block_context"},
        {"name": "without_jared_no_block_context", "include_jared": False, "ablation": "no_block_context"},
        {"name": "with_jared_no_leakage", "include_jared": True, "ablation": "no_leakage"},
        {"name": "without_jared_no_leakage", "include_jared": False, "ablation": "no_leakage"},
    ]

    for exp in experiments:
        exp_name = exp["name"]
        include_jared = bool(exp["include_jared"])
        ablation = str(exp["ablation"])
        run_name = f"{exp_name}{experiment_suffix}"

        logger.info("=== LSTM experiment: %s ===", run_name)

        metrics_path = PROJECT_ROOT / "outputs" / "metrics" / f"lstm_{run_name}_metrics.json"
        pred_path = PROJECT_ROOT / "outputs" / "predictions" / f"lstm_{run_name}_test_predictions.parquet"

        # We only skip if outputs exist *and* the saved predictions use the expected feature set.
        existing_pred_feature_cols: list[str] | None = None
        existing_experiment_version: str | None = None
        existing_sequence_features_path: str | None = None
        if metrics_path.exists() and pred_path.exists():
            try:
                existing_metrics = _load_json(metrics_path)
                existing_experiment_version = existing_metrics.get("experiment_version")
                existing_sequence_features_path = existing_metrics.get("sequence_features_path")
                existing_cols = pd.read_parquet(pred_path, columns=["feature_columns"])["feature_columns"].iloc[0]
                if not isinstance(existing_cols, list):
                    existing_cols = list(existing_cols)
                existing_pred_feature_cols = [str(c) for c in existing_cols]
            except Exception:
                existing_pred_feature_cols = None

        seq_df = seq_df_base.copy()
        if not include_jared:
            seq_df = seq_df[~seq_df["contains_jared_swap"]].copy()

        exp_features = base_feature_columns.copy()
        if ablation == "no_rule":
            # Default sequence features usually do not include these columns.
            removed = [c for c in RULE_SIGNATURE_FEATURES if c in base_feature_columns]
            if not removed:
                logger.warning(
                    "no_rule ablation: explicit rule-signature features are not present in sequence_features.parquet; "
                    "this run is effectively identical to full."
                )
                exp_features = base_feature_columns.copy()
            else:
                exp_features = [c for c in exp_features if c not in RULE_SIGNATURE_FEATURES]
                seq_df = _project_sequence_features(seq_df, keep_feature_columns=exp_features)
        elif ablation == "no_block_context":
            exp_features = [c for c in exp_features if c not in SEQUENCE_BLOCK_CONTEXT_FEATURES]
            seq_df = _project_sequence_features(seq_df, keep_feature_columns=exp_features)
        elif ablation == "no_leakage":
            removed = [c for c in NO_LEAKAGE_FEATURES_TO_DROP if c in exp_features]
            if removed:
                exp_features = [c for c in exp_features if c not in NO_LEAKAGE_FEATURES_TO_DROP]
                seq_df = _project_sequence_features(seq_df, keep_feature_columns=exp_features)
            else:
                logger.warning(
                    "no_leakage ablation: none of the leakage features were present in this sequence dataset; "
                    "this run is effectively identical to full."
                )

        logger.info("Sequences used for training after filtering/ablation: %s", len(seq_df))

        if existing_pred_feature_cols is not None:
            expected_path = str(sequence_features_path)
            version_ok = (existing_experiment_version == EXPERIMENT_VERSION)
            path_ok = (existing_sequence_features_path == expected_path)
            features_ok = (len(existing_pred_feature_cols) == len(exp_features)) and (
                set(existing_pred_feature_cols) == set(exp_features)
            )
            if version_ok and path_ok and features_ok:
                logger.info("Outputs already consistent for %s, skipping re-run: %s", run_name, metrics_path)
                continue
            logger.warning(
                "Existing predictions for %s have mismatching feature_columns (existing=%s, expected=%s). Re-running.",
                run_name,
                len(existing_pred_feature_cols),
                len(exp_features),
            )

        if seq_df.empty:
            logger.warning("Empty sequence table for %s; skipping.", exp_name)
            continue

        artifacts, train_df, val_df, test_df = train_lstm_model(
            dataframe=seq_df,
            target_column="target_contains_weak_anomaly",
            hidden_size=96,
            num_layers=1,
            learning_rate=1e-3,
            batch_size=128,
            epochs=25,
            seed=42,
            use_weighted_sampler=True,
            dropout=0.2,
            early_stopping_patience=4,
        )

        # Evaluate on validation to get threshold, then re-evaluate test with fixed threshold.
        val_metrics = artifacts.validation_metrics
        selected_threshold = float(val_metrics.get("selected_threshold", 0.5))

        mean, std = fit_sequence_mean_std(train_df)
        test_metrics = evaluate_lstm_model(
            model=artifacts.model,
            dataframe=test_df,
            target_column="target_contains_weak_anomaly",
            batch_size=128,
            threshold=selected_threshold,
            mean=mean,
            std=std,
        )

        # Save model state_dict + minimal metadata.
        model_path = PROJECT_ROOT / "outputs" / "reports" / f"lstm_{run_name}.pt"
        torch.save(
            {
                "state_dict": artifacts.model.state_dict(),
                "input_feature_columns": exp_features,
                "sequence_length": inferred_sequence_length,
                "hidden_size": 96,
                "num_layers": 1,
                "selected_threshold": selected_threshold,
                "experiment_version": EXPERIMENT_VERSION,
            },
            model_path,
        )

        # Save predictions on test split.
        pred_path = PROJECT_ROOT / "outputs" / "predictions" / f"lstm_{run_name}_test_predictions.parquet"
        preds = predict_lstm_probabilities(
            model=artifacts.model,
            dataframe=test_df,
            target_column="target_contains_weak_anomaly",
            batch_size=128,
            threshold=selected_threshold,
            mean=mean,
            std=std,
        )
        preds.to_parquet(pred_path, index=False)

        # Score Jared sequences (unlabeled ranking output).
        jared_score_path = PROJECT_ROOT / "outputs" / "predictions" / f"lstm_{run_name}_jared_scored.parquet"
        score_df = seq_all_events
        # If this experiment uses a projected feature set (different input dimension),
        # we must project the Jared scoring dataframe to the same feature columns.
        if ablation == "no_block_context":
            score_df = _project_sequence_features(score_df, keep_feature_columns=exp_features)
        elif ablation == "no_rule":
            # no_rule only projects when it actually removes columns.
            if exp_features != base_feature_columns:
                score_df = _project_sequence_features(score_df, keep_feature_columns=exp_features)
        elif ablation == "no_leakage":
            if exp_features != base_feature_columns:
                score_df = _project_sequence_features(score_df, keep_feature_columns=exp_features)

        _score_jared_sequences(
            model=artifacts.model,
            sequence_df_all_events=score_df,
            jared_swap_ids=jared_swap_ids,
            threshold=selected_threshold,
            output_path=jared_score_path,
            mean=mean,
            std=std,
        )

        metrics_payload = {
            "model": "lstm_sequence_classifier",
            "experiment": run_name,
            "experiment_base": exp_name,
            "experiment_suffix": experiment_suffix,
            "experiment_version": EXPERIMENT_VERSION,
            "include_jared_in_training": include_jared,
            "sequence_ablation": ablation,
            "rule_signature_features": RULE_SIGNATURE_FEATURES,
            "sequence_length": inferred_sequence_length,
            "feature_columns": exp_features,
            "sequence_features_path": str(sequence_features_path),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "training_history": artifacts.train_history,
            "selected_threshold": selected_threshold,
        }

        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        logger.info("Wrote metrics: %s", metrics_path)
        logger.info("Wrote model: %s", model_path)
        logger.info("Wrote test predictions: %s", pred_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
