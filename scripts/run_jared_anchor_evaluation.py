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
from src.common.paths import PROJECT_ROOT


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether existing models detect the high-confidence Jaredfromsubway.eth cases.\n"
            "Read jared_sandwich_cases.parquet, even_features.pq, event_Features.pq"
        )
    )
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "data" / "labels" / "jared_sandwich_cases.parquet"),
        help="Path to the Jared high-confidence cases parquet.",
    )
    parser.add_argument(
        "--event-features",
        default=str(PROJECT_ROOT / "data" / "features" / "event_features.parquet"),
        help="Path to event_features parquet.",
    )
    parser.add_argument(
        "--event-labels",
        default=str(PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"),
        help="Path to event_labels parquet (used for context only; not modified by default).",
    )
    parser.add_argument(
        "--rf-model",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "random_forest_model.joblib"),
        help="Path to trained RandomForest model joblib.",
    )
    parser.add_argument(
        "--rf-metrics",
        default=str(PROJECT_ROOT / "outputs" / "metrics" / "random_forest_metrics.json"),
        help="Path to RandomForest metrics json (for feature list + threshold).",
    )
    parser.add_argument(
        "--if-model",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "isolation_forest_model.joblib"),
        help="Path to trained IsolationForest model joblib.",
    )
    parser.add_argument(
        "--if-metrics",
        default=str(PROJECT_ROOT / "outputs" / "metrics" / "isolation_forest_metrics.json"),
        help="Path to IsolationForest metrics json (for feature list + threshold).",
    )
    parser.add_argument(
        "--rule-predictions",
        default=str(PROJECT_ROOT / "outputs" / "predictions" / "rule_based_event_predictions.parquet"),
        help="Path to rule-based predictions parquet (if missing, will be computed from rule_score).",
    )
    parser.add_argument(
        "--out-md",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "jared_anchor_model_eval.md"),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--write-augmented-labels",
        action="store_true",
        help=(
            "If set, writes data/labels/event_labels_with_jared_anchors.parquet that adds "
            "jared_anchor_* columns without changing label_class."
        ),
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=feature_columns).copy()
    for col in feature_columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.fillna(0.0)


def _build_anchor_event_table(cases_df: pd.DataFrame) -> pd.DataFrame:
    required = ["sequence_id", "block_number", "pool_address", "tx1_id", "tx2_id", "tx3_id"]
    missing = [c for c in required if c not in cases_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cases_df: {missing}")

    rows: list[dict] = []
    for _, row in cases_df.iterrows():
        seq = str(row["sequence_id"])
        block_number = int(row["block_number"]) if row.get("block_number") is not None and not pd.isna(row.get("block_number")) else None
        pool = str(row["pool_address"])

        rows.append(
            {
                "swap_id": str(row["tx1_id"]),
                "jared_anchor_flag": 1,
                "jared_anchor_role": "attacker_entry",
                "jared_case_sequence_id": seq,
                "jared_case_block_number": block_number,
                "jared_case_pool_address": pool,
            }
        )
        rows.append(
            {
                "swap_id": str(row["tx2_id"]),
                "jared_anchor_flag": 1,
                "jared_anchor_role": "victim",
                "jared_case_sequence_id": seq,
                "jared_case_block_number": block_number,
                "jared_case_pool_address": pool,
            }
        )
        rows.append(
            {
                "swap_id": str(row["tx3_id"]),
                "jared_anchor_flag": 1,
                "jared_anchor_role": "attacker_exit",
                "jared_case_sequence_id": seq,
                "jared_case_block_number": block_number,
                "jared_case_pool_address": pool,
            }
        )

    anchors = pd.DataFrame(rows)

    anchors = anchors.drop_duplicates(subset=["swap_id"], keep="first").reset_index(drop=True)
    return anchors


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    load_environment()
    setup_logging()

    cases_path = Path(args.cases)
    event_features_path = Path(args.event_features)
    event_labels_path = Path(args.event_labels)
    rule_predictions_path = Path(args.rule_predictions)
    out_md_path = Path(args.out_md)

    rf_model_path = Path(args.rf_model)
    rf_metrics_path = Path(args.rf_metrics)
    if_model_path = Path(args.if_model)
    if_metrics_path = Path(args.if_metrics)

    if not cases_path.exists():
        print(f"Missing cases file: {cases_path}")
        return 2
    if not event_features_path.exists():
        print(f"Missing event_features file: {event_features_path}")
        return 2

    cases_df = pd.read_parquet(cases_path)
    if cases_df.empty:
        print("Cases dataset is empty. Run detection first.")
        return 0

    anchors_df = _build_anchor_event_table(cases_df)
    anchor_swap_ids = set(anchors_df["swap_id"].astype("string").tolist())
    anchor_victim_ids = set(anchors_df.loc[anchors_df["jared_anchor_role"] == "victim", "swap_id"].astype("string").tolist())

    event_features_df = pd.read_parquet(event_features_path)
    if "swap_id" not in event_features_df.columns:
        raise ValueError("event_features.parquet missing required column: swap_id")

    features_anchor_df = event_features_df[event_features_df["swap_id"].astype("string").isin(anchor_swap_ids)].copy()
    features_victim_df = event_features_df[event_features_df["swap_id"].astype("string").isin(anchor_victim_ids)].copy()

    weak_labels_df = pd.DataFrame()
    if event_labels_path.exists():
        weak_labels_df = pd.read_parquet(event_labels_path)

    if not weak_labels_df.empty and "swap_id" in weak_labels_df.columns:
        features_anchor_df = features_anchor_df.merge(
            weak_labels_df[["swap_id", "label_class", "label_confidence"]].copy(),
            on="swap_id",
            how="left",
        )
        features_victim_df = features_victim_df.merge(
            weak_labels_df[["swap_id", "label_class", "label_confidence"]].copy(),
            on="swap_id",
            how="left",
        )

    if rule_predictions_path.exists():
        rule_pred_df = pd.read_parquet(rule_predictions_path)
    else:
        if "rule_score" not in event_features_df.columns:
            raise ValueError("No rule_based predictions file, and event_features missing rule_score.")
        rule_pred_df = pd.DataFrame(
            {
                "swap_id": event_features_df["swap_id"],
                "rule_score": event_features_df["rule_score"],
                "predicted_suspicious_flag": (pd.to_numeric(event_features_df["rule_score"], errors="coerce").fillna(0.0) >= 1).astype(int),
                "predicted_weak_anomaly_flag": (pd.to_numeric(event_features_df["rule_score"], errors="coerce").fillna(0.0) >= 3).astype(int),
            }
        )

    rule_anchor = anchors_df.merge(rule_pred_df, on="swap_id", how="left")
    rule_victims = rule_anchor[rule_anchor["jared_anchor_role"] == "victim"].copy()

    rule_hit_any = float((pd.to_numeric(rule_anchor.get("predicted_weak_anomaly_flag"), errors="coerce").fillna(0) == 1).mean())
    rule_hit_victim = float((pd.to_numeric(rule_victims.get("predicted_weak_anomaly_flag"), errors="coerce").fillna(0) == 1).mean())

    rf_payload = None
    rf_hit_any = None
    rf_hit_victim = None
    rf_anchor_table = None
    if rf_model_path.exists() and rf_metrics_path.exists():
        rf_payload = _read_json(rf_metrics_path)
        rf_features = list(rf_payload.get("feature_columns", []))
        rf_threshold = _safe_float(rf_payload.get("validation_metrics", {}).get("selected_threshold"))
        if rf_threshold is None:
            rf_threshold = _safe_float(rf_payload.get("test_metrics", {}).get("selected_threshold")) or 0.5

        rf_model = joblib.load(rf_model_path)
        x_anchor = _coerce_feature_matrix(features_anchor_df, rf_features)
        rf_prob = rf_model.predict_proba(x_anchor)[:, 1]

        rf_anchor_table = anchors_df.copy()
        rf_anchor_table["rf_probability"] = rf_prob
        rf_anchor_table["rf_predicted_flag"] = (rf_anchor_table["rf_probability"] >= float(rf_threshold)).astype(int)

        rf_hit_any = float(rf_anchor_table["rf_predicted_flag"].mean())
        rf_hit_victim = float(rf_anchor_table.loc[rf_anchor_table["jared_anchor_role"] == "victim", "rf_predicted_flag"].mean())

        x_all = _coerce_feature_matrix(event_features_df, rf_features)
        all_prob = pd.Series(rf_model.predict_proba(x_all)[:, 1], name="rf_probability")
        victim_prob = rf_anchor_table.loc[rf_anchor_table["jared_anchor_role"] == "victim", "rf_probability"].astype(float)
        victim_percentiles = victim_prob.apply(lambda v: float((all_prob <= v).mean()))
        rf_anchor_table.loc[rf_anchor_table["jared_anchor_role"] == "victim", "rf_percentile_among_all"] = victim_percentiles.values
        rf_victim_percentile_median = float(victim_percentiles.median()) if not victim_percentiles.empty else None
    else:
        rf_features = []
        rf_threshold = None
        rf_victim_percentile_median = None

    if_payload = None
    if_hit_any = None
    if_hit_victim = None
    if_anchor_table = None
    if if_model_path.exists() and if_metrics_path.exists():
        if_payload = _read_json(if_metrics_path)
        if_features = list(if_payload.get("feature_columns", []))
        if_threshold = _safe_float(if_payload.get("validation_metrics", {}).get("selected_threshold"))
        if if_threshold is None:
            if_threshold = _safe_float(if_payload.get("test_metrics", {}).get("selected_threshold"))
        if if_threshold is None:
            if_threshold = 0.0

        if_model = joblib.load(if_model_path)
        x_anchor = _coerce_feature_matrix(features_anchor_df, if_features)
        if_scores = -if_model.decision_function(x_anchor)

        if_anchor_table = anchors_df.copy()
        if_anchor_table["if_anomaly_score"] = if_scores
        if_anchor_table["if_predicted_flag"] = (if_anchor_table["if_anomaly_score"] >= float(if_threshold)).astype(int)

        if_hit_any = float(if_anchor_table["if_predicted_flag"].mean())
        if_hit_victim = float(if_anchor_table.loc[if_anchor_table["jared_anchor_role"] == "victim", "if_predicted_flag"].mean())

        x_all = _coerce_feature_matrix(event_features_df, if_features)
        all_scores = pd.Series(-if_model.decision_function(x_all), name="if_anomaly_score")
        victim_scores = if_anchor_table.loc[if_anchor_table["jared_anchor_role"] == "victim", "if_anomaly_score"].astype(float)
        victim_percentiles = victim_scores.apply(lambda v: float((all_scores <= v).mean()))
        if_anchor_table.loc[if_anchor_table["jared_anchor_role"] == "victim", "if_percentile_among_all"] = victim_percentiles.values
        if_victim_percentile_median = float(victim_percentiles.median()) if not victim_percentiles.empty else None
    else:
        if_features = []
        if_threshold = None
        if_victim_percentile_median = None

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append("# Jaredfromsubway.eth Anchor Evaluation (Do Our Models Detect Them?)")
    md.append("")
    md.append("These are high-confidence, documented bot-pattern cases used as evaluation anchors.")
    md.append("They are **not** a claim of proven malicious intent, but highly structured sequences consistent with sandwich behavior.")
    md.append("")
    md.append(f"- Anchor cases (3-swap triples): **{len(cases_df)}**")
    md.append(f"- Anchor events (tx1+tx2+tx3): **{len(anchors_df)}**")
    md.append(f"- Anchor victim events (tx2 only): **{len(anchor_victim_ids)}**")
    md.append("")

    if not features_anchor_df.empty and "label_class" in features_anchor_df.columns:
        md.append("## Weak-Label Context (What The Current Labeling Thinks)")
        md.append("")
        vc = features_anchor_df["label_class"].value_counts(dropna=False)
        md.append("```text")
        md.append(vc.to_string())
        md.append("```")
        md.append("")

    md.append("## Event Models: Hit Rate On Jared Anchors")
    md.append("")
    md.append("| Model | Victim hit rate (tx2) | Any-of-triple hit rate (tx1/tx2/tx3) | Notes |")
    md.append("|---|---:|---:|---|")
    md.append(f"| Rule-based (rule_score>=3) | {_percent(rule_hit_victim)} | {_percent(rule_hit_any)} | Uses `rule_score` from features |")

    if rf_payload is not None:
        md.append(
            f"| RandomForest (threshold={_fmt_float(_safe_float(rf_threshold), 3)}) | "
            f"{_percent(rf_hit_victim)} | {_percent(rf_hit_any)} | "
            f"Victim score median percentile among all events: {_percent(rf_victim_percentile_median)} |"
        )
    else:
        md.append("| RandomForest | NA | NA | Missing model/metrics files |")

    if if_payload is not None:
        md.append(
            f"| IsolationForest (threshold={_fmt_float(_safe_float(if_threshold), 6)}) | "
            f"{_percent(if_hit_victim)} | {_percent(if_hit_any)} | "
            f"Victim score median percentile among all events: {_percent(if_victim_percentile_median)} |"
        )
    else:
        md.append("| IsolationForest | NA | NA | Missing model/metrics files |")
    md.append("")

    md.append("## Quick Read: What This Means")
    md.append("")
    md.append(
        "- Victim hit rate: does the model flag the middle transaction (tx2)?"
    )
    md.append(
        "- Any-of-triple hit rate answers: if you flag any of the three swaps, do you catch the case at all (useful for triage systems)?"
    )
    md.append("")

    md.append("## Example Anchors With Model Scores (Victim Events)")
    md.append("")

    victim_rows = anchors_df[anchors_df["jared_anchor_role"] == "victim"].merge(
        cases_df[["sequence_id", "block_number", "pool_address", "tx1_hash", "tx2_hash", "tx3_hash", "tick_before", "tick_middle", "tick_after"]],
        left_on="jared_case_sequence_id",
        right_on="sequence_id",
        how="left",
    )
    victim_rows = victim_rows.drop(columns=["sequence_id"], errors="ignore")

    victim_rows = victim_rows.merge(
        rule_pred_df[["swap_id", "rule_score", "predicted_weak_anomaly_flag"]],
        on="swap_id",
        how="left",
    )
    if rf_anchor_table is not None:
        victim_rows = victim_rows.merge(
            rf_anchor_table[["swap_id", "rf_probability", "rf_predicted_flag"]],
            on="swap_id",
            how="left",
        )
    if if_anchor_table is not None:
        victim_rows = victim_rows.merge(
            if_anchor_table[["swap_id", "if_anomaly_score", "if_predicted_flag"]],
            on="swap_id",
            how="left",
        )

    sort_cols = []
    if "rf_probability" in victim_rows.columns:
        sort_cols = ["rf_probability", "rule_score"]
    else:
        sort_cols = ["rule_score"]
    victim_rows = victim_rows.sort_values(sort_cols, ascending=False).head(5)

    keep_cols = [
        "jared_case_block_number",
        "jared_case_pool_address",
        "tx1_hash",
        "tx2_hash",
        "tx3_hash",
        "tick_before",
        "tick_middle",
        "tick_after",
        "rule_score",
        "predicted_weak_anomaly_flag",
        "rf_probability",
        "rf_predicted_flag",
        "if_anomaly_score",
        "if_predicted_flag",
    ]
    keep_cols = [c for c in keep_cols if c in victim_rows.columns]
    md.append("```text")
    md.append(victim_rows[keep_cols].to_string(index=False))
    md.append("```")
    md.append("")

    out_md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote report: {out_md_path}")

    if args.write_augmented_labels:
        if not event_labels_path.exists():
            print(f"Cannot write augmented labels: missing {event_labels_path}")
            return 0

        base_labels = pd.read_parquet(event_labels_path)
        augmented = base_labels.merge(anchors_df, on="swap_id", how="left")
        for col in ["jared_anchor_flag"]:
            if col in augmented.columns:
                augmented[col] = pd.to_numeric(augmented[col], errors="coerce").fillna(0).astype(int)

        out_labels = PROJECT_ROOT / "data" / "labels" / "event_labels_with_jared_anchors.parquet"
        out_labels.parent.mkdir(parents=True, exist_ok=True)
        augmented.to_parquet(out_labels, index=False)
        print(f"Wrote augmented labels: {out_labels}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
