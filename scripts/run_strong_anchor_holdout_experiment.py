from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

# Allow `from src...` when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_environment  # noqa: E402
from src.common.logging_utils import setup_logging  # noqa: E402
from src.common.paths import PROJECT_ROOT, ensure_directory  # noqa: E402
from src.models.baselines.isolation_forest_model import (  # noqa: E402
    generate_isolation_forest_predictions,
    train_isolation_forest_model,
)
from src.models.baselines.random_forest_model import (  # noqa: E402
    generate_random_forest_predictions,
    prepare_random_forest_dataset,
    train_random_forest_model,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train models on weak labels (excluding Jared anchors) and evaluate on Jared anchors as strong anomalies.\n"
            "Reads:\n"
            "- data/features/event_features.parquet\n"
            "- data/labels/event_labels.parquet\n"
            "- data/labels/jared_sandwich_cases.parquet\n"
            "Writes:\n"
            "- outputs/reports/strong_anchor_holdout_eval.md\n"
            "- outputs/reports/random_forest_model_weak_only.joblib\n"
            "- outputs/reports/isolation_forest_model_weak_only.joblib\n"
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
        "--jared-cases",
        default=str(PROJECT_ROOT / "data" / "labels" / "jared_sandwich_cases.parquet"),
        help="Path to Jared high-confidence cases parquet.",
    )
    parser.add_argument(
        "--out-md",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "strong_anchor_holdout_eval.md"),
        help="Output markdown report path.",
    )
    return parser.parse_args(argv)


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


def _fmt_float(x: float | None, ndigits: int = 3) -> str:
    if x is None:
        return "NA"
    return f"{x:.{ndigits}f}"


def _build_anchor_maps(cases_df: pd.DataFrame) -> tuple[pd.DataFrame, set[str], set[str]]:
    """
    Returns:
    - anchors_df: per-swap mapping (swap_id -> role, case id)
    - all_anchor_swap_ids: tx1+tx2+tx3 ids
    - victim_anchor_swap_ids: tx2 ids only (recommended "strong anomaly" definition)
    """
    required = ["sequence_id", "tx1_id", "tx2_id", "tx3_id", "block_number", "pool_address"]
    missing = [c for c in required if c not in cases_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in jared cases: {missing}")

    rows: list[dict] = []
    for _, row in cases_df.iterrows():
        seq = str(row["sequence_id"])
        block_number = int(row["block_number"]) if row.get("block_number") is not None and not pd.isna(row.get("block_number")) else None
        pool = str(row["pool_address"])
        rows.append(
            {
                "swap_id": str(row["tx1_id"]),
                "anchor_role": "attacker_entry",
                "anchor_sequence_id": seq,
                "anchor_block_number": block_number,
                "anchor_pool_address": pool,
            }
        )
        rows.append(
            {
                "swap_id": str(row["tx2_id"]),
                "anchor_role": "victim",
                "anchor_sequence_id": seq,
                "anchor_block_number": block_number,
                "anchor_pool_address": pool,
            }
        )
        rows.append(
            {
                "swap_id": str(row["tx3_id"]),
                "anchor_role": "attacker_exit",
                "anchor_sequence_id": seq,
                "anchor_block_number": block_number,
                "anchor_pool_address": pool,
            }
        )

    anchors_df = pd.DataFrame(rows).drop_duplicates(subset=["swap_id"], keep="first").reset_index(drop=True)
    all_ids = set(anchors_df["swap_id"].astype("string").tolist())
    victim_ids = set(anchors_df.loc[anchors_df["anchor_role"] == "victim", "swap_id"].astype("string").tolist())
    return anchors_df, all_ids, victim_ids


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    load_environment()
    setup_logging()

    event_features_path = Path(args.event_features)
    event_labels_path = Path(args.event_labels)
    jared_cases_path = Path(args.jared_cases)
    out_md_path = Path(args.out_md)

    if not event_features_path.exists():
        print(f"Missing event features: {event_features_path}")
        return 2
    if not event_labels_path.exists():
        print(f"Missing event labels: {event_labels_path}")
        return 2
    if not jared_cases_path.exists():
        print(f"Missing Jared cases: {jared_cases_path}")
        return 2

    event_features_df = pd.read_parquet(event_features_path)
    event_labels_df = pd.read_parquet(event_labels_path)
    jared_cases_df = pd.read_parquet(jared_cases_path)

    anchors_df, all_anchor_ids, victim_anchor_ids = _build_anchor_maps(jared_cases_df)

    # Build the standard event-modeling dataset (normal vs weak_anomaly only).
    full_dataset_df = prepare_random_forest_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
    )

    full_dataset_df["is_jared_anchor"] = full_dataset_df["swap_id"].astype("string").isin(all_anchor_ids).astype(int)
    full_dataset_df = full_dataset_df.merge(anchors_df, on="swap_id", how="left")

    # Holdout strategy: remove anchors from training completely (avoid leakage).
    train_pool_df = full_dataset_df[full_dataset_df["is_jared_anchor"] == 0].copy()
    anchor_eval_df = full_dataset_df[full_dataset_df["is_jared_anchor"] == 1].copy()
    anchor_victim_eval_df = anchor_eval_df[anchor_eval_df["anchor_role"] == "victim"].copy()

    if train_pool_df.empty:
        raise ValueError("Training pool is empty after excluding anchors.")

    # Train a new RF on weak labels excluding anchors.
    rf_artifacts = train_random_forest_model(dataset_df=train_pool_df)
    rf_threshold = _safe_float(rf_artifacts.validation_metrics.get("selected_threshold")) or 0.5

    rf_model_path = PROJECT_ROOT / "outputs" / "reports" / "random_forest_model_weak_only.joblib"
    ensure_directory(rf_model_path.parent)
    joblib.dump(rf_artifacts.model, rf_model_path)

    # Evaluate RF on anchor victims and on all anchor events.
    rf_anchor_pred = generate_random_forest_predictions(
        model=rf_artifacts.model,
        dataframe=anchor_eval_df,
        feature_columns=rf_artifacts.feature_columns,
        threshold=float(rf_threshold),
    )
    rf_victim_pred = rf_anchor_pred[rf_anchor_pred["swap_id"].astype("string").isin(victim_anchor_ids)].copy()

    rf_any_hit_rate = float(rf_anchor_pred["rf_predicted_flag"].mean()) if len(rf_anchor_pred) else None
    rf_victim_hit_rate = float(rf_victim_pred["rf_predicted_flag"].mean()) if len(rf_victim_pred) else None

    # Score percentiles among non-anchor events (stronger story than "among all incl anchors").
    rf_all_non_anchor_pred = generate_random_forest_predictions(
        model=rf_artifacts.model,
        dataframe=train_pool_df,
        feature_columns=rf_artifacts.feature_columns,
        threshold=float(rf_threshold),
    )
    rf_non_anchor_scores = rf_all_non_anchor_pred["rf_probability"].astype(float)
    rf_victim_scores = rf_victim_pred["rf_probability"].astype(float)
    rf_victim_percentiles = rf_victim_scores.apply(lambda v: float((rf_non_anchor_scores <= v).mean()))
    rf_victim_percentile_median = float(rf_victim_percentiles.median()) if not rf_victim_percentiles.empty else None

    # Train a new IsolationForest on weak labels excluding anchors (fit only on normal rows within that pool).
    if_artifacts = train_isolation_forest_model(dataset_df=train_pool_df)
    if_threshold = _safe_float(if_artifacts.validation_metrics.get("selected_threshold"))
    if if_threshold is None:
        if_threshold = 0.0

    if_model_path = PROJECT_ROOT / "outputs" / "reports" / "isolation_forest_model_weak_only.joblib"
    ensure_directory(if_model_path.parent)
    joblib.dump(if_artifacts.model, if_model_path)

    if_anchor_pred = generate_isolation_forest_predictions(
        model=if_artifacts.model,
        dataframe=anchor_eval_df,
        feature_columns=if_artifacts.feature_columns,
        threshold=float(if_threshold),
    )
    if_victim_pred = if_anchor_pred[if_anchor_pred["swap_id"].astype("string").isin(victim_anchor_ids)].copy()

    if_any_hit_rate = float(if_anchor_pred["if_predicted_flag"].mean()) if len(if_anchor_pred) else None
    if_victim_hit_rate = float(if_victim_pred["if_predicted_flag"].mean()) if len(if_victim_pred) else None

    if_all_non_anchor_pred = generate_isolation_forest_predictions(
        model=if_artifacts.model,
        dataframe=train_pool_df,
        feature_columns=if_artifacts.feature_columns,
        threshold=float(if_threshold),
    )
    if_non_anchor_scores = if_all_non_anchor_pred["if_anomaly_score"].astype(float)
    if_victim_scores = if_victim_pred["if_anomaly_score"].astype(float)
    if_victim_percentiles = if_victim_scores.apply(lambda v: float((if_non_anchor_scores <= v).mean()))
    if_victim_percentile_median = float(if_victim_percentiles.median()) if not if_victim_percentiles.empty else None

    # Report.
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append("# Strong Anchor Holdout Experiment (Train on Weak, Test on Strong)")
    md.append("")
    md.append("Definition used here:")
    md.append("- Weak labels: existing `event_labels.parquet` (`normal` vs `weak_anomaly`), generated by heuristic rules.")
    md.append("- Strong anomalies: Jaredfromsubway.eth anchor cases (known bot-pattern), evaluated as a holdout set.")
    md.append("")
    md.append("Holdout / leakage control:")
    md.append("- We **exclude all Jared anchor events** (tx1+tx2+tx3) from training/validation/test splits.")
    md.append("- We train models on the remaining weak-labeled data, then evaluate detection on the anchor set.")
    md.append("")
    md.append(f"- Anchor triples: **{len(jared_cases_df)}**")
    md.append(f"- Anchor events (tx1+tx2+tx3): **{len(anchor_eval_df)}**")
    md.append(f"- Anchor victim events (tx2): **{len(anchor_victim_eval_df)}**")
    md.append("")

    md.append("## Models (Trained on Weak Labels Only)")
    md.append("")
    md.append(f"- RandomForest saved to: `{rf_model_path}`")
    md.append(f"- IsolationForest saved to: `{if_model_path}`")
    md.append("")

    md.append("## Strong-Anchor Detection Results")
    md.append("")
    md.append("| Model | Victim hit rate (tx2) | Any-of-triple hit rate (tx1/tx2/tx3) | Victim median percentile vs non-anchor events |")
    md.append("|---|---:|---:|---:|")
    md.append(
        f"| RandomForest (threshold={_fmt_float(rf_threshold, 3)}) | "
        f"{_percent(rf_victim_hit_rate)} | {_percent(rf_any_hit_rate)} | {_percent(rf_victim_percentile_median)} |"
    )
    md.append(
        f"| IsolationForest (threshold={_fmt_float(if_threshold, 6)}) | "
        f"{_percent(if_victim_hit_rate)} | {_percent(if_any_hit_rate)} | {_percent(if_victim_percentile_median)} |"
    )
    md.append("")

    md.append("## Notes")
    md.append("")
    md.append("- If these hit rates stay high even when anchors are excluded from training, it supports generalization from weak labels to strong anchors.")
    md.append("- If they drop a lot, it suggests the model may be overfitting to patterns that overlap heavily with the anchor construction.")
    md.append("")

    out_md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote report: {out_md_path}")

    # Also write a machine-readable JSON summary for thesis plots if needed.
    json_path = out_md_path.with_suffix(".json")
    payload = {
        "anchor_triples": int(len(jared_cases_df)),
        "anchor_events": int(len(anchor_eval_df)),
        "anchor_victim_events": int(len(anchor_victim_eval_df)),
        "rf": {
            "threshold": float(rf_threshold),
            "victim_hit_rate": rf_victim_hit_rate,
            "any_hit_rate": rf_any_hit_rate,
            "victim_percentile_median_vs_non_anchor": rf_victim_percentile_median,
            "validation_metrics": rf_artifacts.validation_metrics,
            "test_metrics": rf_artifacts.test_metrics,
        },
        "isolation_forest": {
            "threshold": float(if_threshold),
            "victim_hit_rate": if_victim_hit_rate,
            "any_hit_rate": if_any_hit_rate,
            "victim_percentile_median_vs_non_anchor": if_victim_percentile_median,
            "validation_metrics": if_artifacts.validation_metrics,
            "test_metrics": if_artifacts.test_metrics,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

