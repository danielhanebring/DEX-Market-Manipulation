from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_environment  
from src.common.logging_utils import setup_logging  
from src.common.paths import PROJECT_ROOT, ensure_directory 
from src.models.baselines.isolation_forest_model import (  
    train_isolation_forest_model,
)
from src.models.baselines.logistic_regression_model import (
    train_logistic_regression_model,
)
from src.models.baselines.random_forest_model import (  
    prepare_random_forest_dataset,
    train_random_forest_model,
)


DEFAULT_BOT_ADDRESS = "0xAE2Fc483527B8EF99EB5D9B44875F005ba1FaE13"

INDEPENDENT_FEATURE_COLUMNS: list[str] = [
    "swap_size_token0",
    "swap_size_token1",
    "interarrival_seconds",
    "same_block_event_count",
    "same_block_pattern_flag",
    "local_event_density_10",
    "gas_price_gwei",
    "gas_price_relative_to_local_mean",
    "gas_price_relative_to_local_median",
    "gas_spike_flag",
    "same_sender_recent_count_20",
    "same_recipient_recent_count_20",
    "sender_recipient_pair_recent_count_20",
    "same_sender_same_block_count",
    "same_recipient_same_block_count",
    "tick_change_from_previous",
    "abs_tick_change",
    "relative_sqrt_price_change",
    "burst_activity_flag",
    "position_in_block",
    "block_event_count",
    "swap_size_token0_relative_to_block_mean",
    "swap_size_token1_relative_to_block_mean",
    "gas_price_relative_to_block_mean",
    "gas_price_relative_to_block_median",
    "gas_price_relative_to_neighbors_mean",
    "high_block_gas_context_flag",
    "high_relative_trade_size_flag",
]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent evaluation: train on rule-based weak labels, apply models to an UNLABELED Jared dataset.\n"
            "\n"
            "Key constraint implemented here:\n"
            "- We do NOT run 3-swap sandwich detection rules on the Jared dataset.\n"
            "- We do NOT pre-label Jared data. We only filter by bot address presence.\n"
            "- We also train a model that excludes explicit sandwich-signature features to reduce circularity.\n"
            "\n"
            "Outputs ranked anomalies + block extracts for manual inspection.\n"
        )
    )
    parser.add_argument(
        "--event-features",
        default=str(PROJECT_ROOT / "data" / "features" / "event_features.parquet"),
        help="Path to event_features parquet.",
    )
    parser.add_argument(
        "--event-labels",
        default=str(PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"),
        help="Path to weak event_labels parquet.",
    )
    parser.add_argument(
        "--swaps",
        default=str(PROJECT_ROOT / "data" / "processed" / "swaps_clean.parquet"),
        help="Path to swaps_clean parquet (used for block-level extraction).",
    )
    parser.add_argument(
        "--bot",
        default=DEFAULT_BOT_ADDRESS,
        help="Bot address for unlabeled evaluation dataset construction.",
    )
    parser.add_argument(
        "--include-origin",
        action="store_true",
        help=(
            "Also treat origin_address == bot as a Jared interaction. "
            "Recommended for datasets where MEV bots appear as origin but swaps use router/aggregator as sender."
        ),
    )
    parser.add_argument(
        "--exclude-bot-from-train",
        action="store_true",
        help="Exclude bot-address interactions from the weak-labeled training pool to avoid leakage.",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=0.01,
        # NOTE: Avoid '%' in argparse help strings (argparse may treat it as formatting).
        help="Top fraction of scored Jared events to export for manual inspection (e.g. 0.01 = top 1 percent).",
    )
    parser.add_argument(
        "--top-blocks",
        type=int,
        default=10,
        help="How many unique top-scoring blocks to export for manual block-level inspection.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs" / "reports"),
        help="Output directory for reports and exports.",
    )
    return parser.parse_args(argv)


def _lower_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower()


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _percent(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.1f}%"


def _fmt_float(x: float | None, ndigits: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{x:.{ndigits}f}"


def _coerce_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=feature_columns).copy()
    for col in feature_columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.fillna(0.0)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    load_environment()
    setup_logging()

    event_features_path = Path(args.event_features)
    event_labels_path = Path(args.event_labels)
    swaps_path = Path(args.swaps)
    out_dir = Path(args.out_dir)

    if not event_features_path.exists():
        print(f"Missing event_features: {event_features_path}")
        return 2
    if not event_labels_path.exists():
        print(f"Missing event_labels: {event_labels_path}")
        return 2
    if not swaps_path.exists():
        print(f"Missing swaps_clean: {swaps_path}")
        return 2

    ensure_directory(out_dir)

    bot = str(args.bot).lower()
    top_pct = float(args.top_pct)
    top_blocks = int(args.top_blocks)
    if top_pct <= 0 or top_pct >= 1:
        raise ValueError("--top-pct must be between 0 and 1 (e.g. 0.01 for top 1%).")

    event_features_df = pd.read_parquet(event_features_path)
    event_labels_df = pd.read_parquet(event_labels_path)

    dataset_df = prepare_random_forest_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
        feature_columns=INDEPENDENT_FEATURE_COLUMNS,
    )

    if args.exclude_bot_from_train:
        sender = _lower_series(event_features_df["sender_address"]) if "sender_address" in event_features_df.columns else None
        recipient = _lower_series(event_features_df["recipient_address"]) if "recipient_address" in event_features_df.columns else None
        origin = _lower_series(event_features_df["origin_address"]) if "origin_address" in event_features_df.columns else None
        mask = pd.Series(False, index=event_features_df.index)
        if sender is not None:
            mask = mask | (sender == bot)
        if recipient is not None:
            mask = mask | (recipient == bot)
        if origin is not None:
            mask = mask | (origin == bot)

        bot_swap_ids = set(event_features_df.loc[mask, "swap_id"].astype("string").tolist())
        before = len(dataset_df)
        dataset_df = dataset_df[~dataset_df["swap_id"].astype("string").isin(bot_swap_ids)].copy()
        after = len(dataset_df)
        print(f"Excluded bot interactions from training pool: {before - after} rows removed.")

    rf_artifacts = train_random_forest_model(dataset_df=dataset_df, feature_columns=INDEPENDENT_FEATURE_COLUMNS)
    rf_threshold = _safe_float(rf_artifacts.validation_metrics.get("selected_threshold")) or 0.5

    if_artifacts = train_isolation_forest_model(dataset_df=dataset_df, feature_columns=INDEPENDENT_FEATURE_COLUMNS)
    if_threshold = _safe_float(if_artifacts.validation_metrics.get("selected_threshold"))
    if if_threshold is None:
        if_threshold = 0.0

    lr_artifacts = train_logistic_regression_model(dataset_df=dataset_df, feature_columns=INDEPENDENT_FEATURE_COLUMNS)
    lr_threshold = _safe_float(lr_artifacts.validation_metrics.get("selected_threshold")) or 0.5

    rf_model_path = out_dir / "random_forest_model_independent.joblib"
    if_model_path = out_dir / "isolation_forest_model_independent.joblib"
    lr_model_path = out_dir / "logistic_regression_model_independent.joblib"
    rf_metrics_path = out_dir / "random_forest_model_independent_metrics.json"
    if_metrics_path = out_dir / "isolation_forest_model_independent_metrics.json"
    lr_metrics_path = out_dir / "logistic_regression_model_independent_metrics.json"

    joblib.dump(rf_artifacts.model, rf_model_path)
    joblib.dump(if_artifacts.model, if_model_path)
    joblib.dump(lr_artifacts.model, lr_model_path)
    rf_metrics_path.write_text(json.dumps(rf_artifacts.validation_metrics, indent=2), encoding="utf-8")
    if_metrics_path.write_text(json.dumps(if_artifacts.validation_metrics, indent=2), encoding="utf-8")
    lr_metrics_path.write_text(json.dumps(lr_artifacts.validation_metrics, indent=2), encoding="utf-8")

    required_cols = ["sender_address", "recipient_address"]
    missing_cols = [c for c in required_cols if c not in event_features_df.columns]
    if missing_cols:
        raise ValueError(f"event_features missing required columns for bot filtering: {missing_cols}")

    sender = _lower_series(event_features_df["sender_address"])
    recipient = _lower_series(event_features_df["recipient_address"])
    jared_mask = (sender == bot) | (recipient == bot)

    if args.include_origin:
        if "origin_address" not in event_features_df.columns:
            raise ValueError("--include-origin set but event_features has no origin_address column.")
        origin = _lower_series(event_features_df["origin_address"])
        jared_mask = jared_mask | (origin == bot)
    jared_df = event_features_df[jared_mask].copy()

    if jared_df.empty:
        print("No Jared interactions found using sender/recipient filter.")
        return 0

    x_jared = _coerce_feature_matrix(jared_df, INDEPENDENT_FEATURE_COLUMNS)
    rf_prob = rf_artifacts.model.predict_proba(x_jared)[:, 1]
    jared_df["rf_probability"] = rf_prob
    jared_df["rf_predicted_flag"] = (jared_df["rf_probability"] >= float(rf_threshold)).astype(int)

    if_scores = -if_artifacts.model.decision_function(x_jared)
    jared_df["if_anomaly_score"] = if_scores
    jared_df["if_predicted_flag"] = (jared_df["if_anomaly_score"] >= float(if_threshold)).astype(int)

    lr_prob = lr_artifacts.model.predict_proba(x_jared)[:, 1]
    jared_df["lr_probability"] = lr_prob
    jared_df["lr_predicted_flag"] = (jared_df["lr_probability"] >= float(lr_threshold)).astype(int)

    jared_df = jared_df.sort_values("rf_probability", ascending=False).reset_index(drop=True)
    jared_df["rank"] = jared_df.index + 1
    jared_df["rank_percentile"] = jared_df["rank"] / float(len(jared_df))

    top_n = max(1, int(math.ceil(top_pct * len(jared_df))))
    top_df = jared_df.head(top_n).copy()

    scored_path = out_dir / "jared_unlabeled_scored.parquet"
    scored_csv_path = out_dir / "jared_unlabeled_scored.csv"
    jared_df.to_parquet(scored_path, index=False)
    jared_df.to_csv(scored_csv_path, index=False)

    top_df["block_number"] = pd.to_numeric(top_df["block_number"], errors="coerce")
    unique_blocks = (
        top_df.dropna(subset=["block_number"])
        .drop_duplicates(subset=["block_number"], keep="first")
        .head(top_blocks)
    )
    blocks = [int(b) for b in unique_blocks["block_number"].tolist()]

    swaps_df = pd.read_parquet(swaps_path)
    block_context_df = swaps_df[swaps_df["block_number"].isin(blocks)].copy()
    block_context_df = block_context_df.sort_values(["block_number", "log_index", "swap_id"]).reset_index(drop=True)

    block_out_path = out_dir / "jared_top_blocks_swaps.parquet"
    block_out_csv = out_dir / "jared_top_blocks_swaps.csv"
    block_context_df.to_parquet(block_out_path, index=False)
    block_context_df.to_csv(block_out_csv, index=False)

    template_cols = [
        "case_id",
        "swap_id",
        "block_number",
        "pool_address",
        "token0_symbol",
        "token1_symbol",
        "transaction_hash",
        "log_index",
        "rf_probability",
        "lr_probability",
        "if_anomaly_score",
        "sandwich_pattern_yes_no",
        "price_reversal_yes_no",
        "gas_spike_yes_no",
        "notes",
        "verdict",
    ]
    review_rows = top_df[
        [
            c
            for c in [
                "swap_id",
                "block_number",
                "pool_address",
                "token0_symbol",
                "token1_symbol",
                "transaction_hash",
                "log_index",
                "rf_probability",
                "lr_probability",
                "if_anomaly_score",
            ]
            if c in top_df.columns
        ]
    ].copy()
    review_rows.insert(0, "case_id", [f"case_{i+1:03d}" for i in range(len(review_rows))])
    for col in ["sandwich_pattern_yes_no", "price_reversal_yes_no", "gas_spike_yes_no", "notes", "verdict"]:
        review_rows[col] = ""
    review_rows = review_rows.reindex(columns=[c for c in template_cols if c in review_rows.columns])

    review_template_path = out_dir / "jared_manual_review_template.csv"
    review_rows.to_csv(review_template_path, index=False)

    md_path = out_dir / "jared_independent_eval_readme.md"
    md_lines: list[str] = []
    md_lines.append("# Jared Independent Evaluation (Unlabeled)")
    md_lines.append("")
    md_lines.append("This run follows the constraint: **no sandwich rules were applied to label Jared data**.")
    if args.include_origin:
        md_lines.append("Jared dataset construction used only an address filter (sender/recipient/origin == bot).")
    else:
        md_lines.append("Jared dataset construction used only an address filter (sender/recipient == bot).")
    md_lines.append("")
    md_lines.append("## Outputs")
    md_lines.append("")
    md_lines.append(f"- Scored unlabeled Jared events (all): `{scored_path}` and `{scored_csv_path}`")
    md_lines.append(f"- Full block context for top blocks: `{block_out_path}` and `{block_out_csv}`")
    md_lines.append(f"- Manual review template (fill in Yes/No + verdict): `{review_template_path}`")
    md_lines.append("")
    md_lines.append("## How To Manually Inspect")
    md_lines.append("")
    md_lines.append("1. Open `jared_manual_review_template.csv` and pick the top cases (highest rf_probability).")
    md_lines.append("2. For each case, open `jared_top_blocks_swaps.csv` and filter on the matching block_number.")
    md_lines.append("3. Sort by log_index and look for: 3-swap structure, tick reversal, gas behavior, timing tightness, trade sizes.")
    md_lines.append("")
    md_lines.append("## Model Notes")
    md_lines.append("")
    md_lines.append("- The model was trained on weak labels from `event_labels.parquet` (normal vs weak_anomaly).")
    md_lines.append("- The feature set used here excludes explicit sandwich signature features to reduce circularity.")
    md_lines.append("")
    md_lines.append("## Training Metrics (weak labels)")
    md_lines.append("")
    md_lines.append("```text")
    md_lines.append(f"RF selected_threshold={_fmt_float(rf_threshold, 3)}")
    md_lines.append(str(rf_artifacts.validation_metrics))
    md_lines.append(f"IF selected_threshold={_fmt_float(if_threshold, 6)}")
    md_lines.append(str(if_artifacts.validation_metrics))
    md_lines.append(f"LR selected_threshold={_fmt_float(lr_threshold, 3)}")
    md_lines.append(str(lr_artifacts.validation_metrics))
    md_lines.append("```")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("Wrote outputs:")
    print(f"- {scored_path}")
    print(f"- {block_out_path}")
    print(f"- {review_template_path}")
    print(f"- {md_path}")
    print(f"Top pct exported for review: {top_pct} -> top_n={top_n} rows")
    print(f"Top unique blocks exported: {len(blocks)} -> {blocks}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
