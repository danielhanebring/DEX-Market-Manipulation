from __future__ import annotations

import argparse
import json
import logging

import joblib
import pandas as pd

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_environment 
from src.common.logging_utils import setup_logging 
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.models.baselines.random_forest_model import (
    DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS,
    prepare_random_forest_dataset,
    time_split_for_random_forest,
)
from src.models.baselines.xgboost_model import ( 
    generate_xgboost_predictions,
    train_xgboost_model,
)

logger = logging.getLogger(__name__)

BOT = "0xAE2Fc483527B8EF99EB5D9B44875F005ba1FaE13".lower()


RULE_SIGNATURE_FEATURES = [
    "same_block_pattern_flag",
    "strict_sandwich_support_flag",
    "three_event_pattern_indicator",
    "same_origin_before_after_flag",
]


def _lower_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower()


def _coerce_x(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=feature_columns).copy()
    for c in feature_columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.fillna(0.0)


def _build_feature_columns(*, no_rule_features: bool) -> list[str]:
    """
    Return the feature columns used for training/scoring.
    """
    feature_columns = list(DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS)
    if no_rule_features:
        feature_columns = [c for c in feature_columns if c not in RULE_SIGNATURE_FEATURES]
    return feature_columns


def _score_summary(scores: pd.Series, *, threshold: float) -> dict[str, float | int | None]:

    if scores.empty:
        return {
            "count": 0,
            "threshold": float(threshold),
            "flagged_count": 0,
            "flagged_rate": 0.0,
            "score_min": None,
            "score_median": None,
            "score_p90": None,
            "score_p95": None,
            "score_p99": None,
            "score_max": None,
            "score_mean": None,
            "score_std": None,
        }

    s = pd.to_numeric(scores, errors="coerce").fillna(0.0)
    flags = (s >= float(threshold)).astype(int)
    quantiles = s.quantile([0.5, 0.9, 0.95, 0.99])

    return {
        "count": int(len(s)),
        "threshold": float(threshold),
        "flagged_count": int(flags.sum()),
        "flagged_rate": float(flags.mean()),
        "score_min": float(s.min()),
        "score_median": float(quantiles.loc[0.5]),
        "score_p90": float(quantiles.loc[0.9]),
        "score_p95": float(quantiles.loc[0.95]),
        "score_p99": float(quantiles.loc[0.99]),
        "score_max": float(s.max()),
        "score_mean": float(s.mean()),
        "score_std": float(s.std(ddof=0)),
    }


def _compute_bot_swap_ids(features_df: pd.DataFrame) -> tuple[set[str], int]:

    sender = _lower_series(features_df["sender_address"]) if "sender_address" in features_df.columns else None
    recipient = _lower_series(features_df["recipient_address"]) if "recipient_address" in features_df.columns else None
    origin = _lower_series(features_df["origin_address"]) if "origin_address" in features_df.columns else None

    mask = pd.Series(False, index=features_df.index)
    if sender is not None:
        mask = mask | (sender == BOT)
    if recipient is not None:
        mask = mask | (recipient == BOT)
    if origin is not None:
        mask = mask | (origin == BOT)

    bot_swap_ids = set(features_df.loc[mask, "swap_id"].astype("string").tolist())
    return bot_swap_ids, int(mask.sum())


def _run_variant(
    *,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    bot_swap_ids: set[str],
    bot_rows_in_event_features: int,
    out_dir: Path,
    out_metrics_dir: Path,
    out_pred_dir: Path,
    no_rule_features: bool,
    suffix: str,
) -> None:
    """
    Train XGBoost with and without Jared interactions and write all outputs for this variant.
    """
    feature_columns = _build_feature_columns(no_rule_features=no_rule_features)
    if no_rule_features:
        logger.info("Rule-feature ablation enabled. Dropping: %s", RULE_SIGNATURE_FEATURES)
    logger.info("Feature column count (%s): %s", suffix or "baseline", len(feature_columns))

    # Build baseline dataset (normal vs weak_anomaly only).
    base_dataset = prepare_random_forest_dataset(
        event_features_df=features_df,
        event_labels_df=labels_df,
        feature_columns=feature_columns,
    )

    base_dataset["is_jared_interaction"] = base_dataset["swap_id"].astype("string").isin(bot_swap_ids).astype(int)

    dataset_with = base_dataset.copy()
    dataset_without = base_dataset[base_dataset["is_jared_interaction"] == 0].copy()

    logger.info("Dataset with Jared (%s): %s", suffix or "baseline", dataset_with.shape)
    logger.info("Dataset without Jared (%s): %s", suffix or "baseline", dataset_without.shape)

    artifacts_with = train_xgboost_model(dataset_df=dataset_with, feature_columns=feature_columns)
    artifacts_without = train_xgboost_model(dataset_df=dataset_without, feature_columns=feature_columns)

    model_with_path = out_dir / f"xgboost_model_with_jared{suffix}.joblib"
    model_without_path = out_dir / f"xgboost_model_without_jared{suffix}.joblib"
    joblib.dump(artifacts_with.model, model_with_path)
    joblib.dump(artifacts_without.model, model_without_path)

    metrics_payload: dict[str, object] = {
        "model": "xgboost_event_classifier",
        "feature_columns": artifacts_with.feature_columns,
        "no_rule_features": bool(no_rule_features),
        "rule_signature_features": RULE_SIGNATURE_FEATURES,
        "with_jared": {
            "train_size": artifacts_with.train_size,
            "validation_size": artifacts_with.validation_size,
            "test_size": artifacts_with.test_size,
            "validation_metrics": artifacts_with.validation_metrics,
            "test_metrics": artifacts_with.test_metrics,
        },
        "without_jared": {
            "train_size": artifacts_without.train_size,
            "validation_size": artifacts_without.validation_size,
            "test_size": artifacts_without.test_size,
            "validation_metrics": artifacts_without.validation_metrics,
            "test_metrics": artifacts_without.test_metrics,
        },
        "jared_rows_in_event_features": int(bot_rows_in_event_features),
    }

    metrics_path = out_metrics_dir / f"xgboost_jared_ablation_metrics{suffix}.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    jared_events = features_df[features_df["swap_id"].astype("string").isin(bot_swap_ids)].copy()
    if not jared_events.empty:
        x_jared = _coerce_x(jared_events, artifacts_with.feature_columns)
        jared_events["xgb_probability_with_jared"] = artifacts_with.model.predict_proba(x_jared)[:, 1]
        jared_events["xgb_probability_without_jared"] = artifacts_without.model.predict_proba(x_jared)[:, 1]

        th_with = float(artifacts_with.validation_metrics["selected_threshold"])
        th_without = float(artifacts_without.validation_metrics["selected_threshold"])
        jared_events["xgb_flag_with_jared"] = (jared_events["xgb_probability_with_jared"] >= th_with).astype(int)
        jared_events["xgb_flag_without_jared"] = (jared_events["xgb_probability_without_jared"] >= th_without).astype(int)

        metrics_payload["jared_scoring_summary"] = {
            "with_jared": _score_summary(jared_events["xgb_probability_with_jared"], threshold=th_with),
            "without_jared": _score_summary(jared_events["xgb_probability_without_jared"], threshold=th_without),
            "spearman_score_corr_with_vs_without": float(
                jared_events[["xgb_probability_with_jared", "xgb_probability_without_jared"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "flag_overlap_rate": float(
                (
                    (jared_events["xgb_flag_with_jared"] == 1)
                    & (jared_events["xgb_flag_without_jared"] == 1)
                ).mean()
            ),
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        scored_out = out_dir / f"jared_scored_by_xgboost_ablation{suffix}.csv"
        keep_cols = [
            "swap_id",
            "block_number",
            "pool_address",
            "transaction_hash",
            "log_index",
            "token0_symbol",
            "token1_symbol",
            "xgb_probability_with_jared",
            "xgb_probability_without_jared",
            "xgb_flag_with_jared",
            "xgb_flag_without_jared",
        ]
        keep_cols = [c for c in keep_cols if c in jared_events.columns]
        jared_events.sort_values("xgb_probability_without_jared", ascending=False)[keep_cols].to_csv(scored_out, index=False)
        logger.info("Wrote: %s", scored_out)

    for name, dataset_df, artifacts in [
        ("with_jared", dataset_with, artifacts_with),
        ("without_jared", dataset_without, artifacts_without),
    ]:
        _, _, test_df = time_split_for_random_forest(dataset_df)
        pred_df = generate_xgboost_predictions(
            model=artifacts.model,
            dataframe=test_df,
            feature_columns=artifacts.feature_columns,
            threshold=float(artifacts.validation_metrics["selected_threshold"]),
        )
        pred_path = out_pred_dir / f"xgboost_test_predictions_{name}{suffix}.parquet"
        pred_df.to_parquet(pred_path, index=False)
        logger.info("Wrote: %s", pred_path)

    logger.info("Saved models: %s | %s", model_with_path, model_without_path)
    logger.info("Saved metrics: %s", metrics_path)


def main() -> None:
    load_environment()
    setup_logging()

    event_features_path = PROJECT_ROOT / "data" / "features" / "event_features.parquet"
    event_labels_path = PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"

    out_dir = PROJECT_ROOT / "outputs" / "reports"
    out_metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    out_pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    ensure_directory(out_dir)
    ensure_directory(out_metrics_dir)
    ensure_directory(out_pred_dir)

    logger.info("Loading event features from: %s", event_features_path)
    features_df = pd.read_parquet(event_features_path)
    logger.info("Loading event labels from: %s", event_labels_path)
    labels_df = pd.read_parquet(event_labels_path)

    bot_swap_ids, bot_rows_in_event_features = _compute_bot_swap_ids(features_df)
    logger.info("Detected Jared interactions in event_features: %s rows", int(bot_rows_in_event_features))

    if len(sys.argv) == 1:
        runs = [
            {"no_rule_features": False, "suffix": ""},
            {"no_rule_features": True, "suffix": "_no_rule_features"},
        ]
    else:
        parser = argparse.ArgumentParser(description="XGBoost: ablation study for Jared interactions and rule features.")
        parser.add_argument(
            "--no-rule-features",
            action="store_true",
            help=(
                "Drop explicit sandwich rule-signature features from the model feature set "
                f"({', '.join(RULE_SIGNATURE_FEATURES)})."
            ),
        )
        parser.add_argument(
            "--suffix",
            default="",
            help="Optional filename suffix for outputs, e.g. _no_rule_features (default: empty).",
        )
        args = parser.parse_args()
        runs = [{"no_rule_features": bool(args.no_rule_features), "suffix": str(args.suffix or "")}]

    for run in runs:
        _run_variant(
            features_df=features_df,
            labels_df=labels_df,
            bot_swap_ids=bot_swap_ids,
            bot_rows_in_event_features=bot_rows_in_event_features,
            out_dir=out_dir,
            out_metrics_dir=out_metrics_dir,
            out_pred_dir=out_pred_dir,
            no_rule_features=bool(run["no_rule_features"]),
            suffix=str(run["suffix"]),
        )


if __name__ == "__main__":
    main()
