from __future__ import annotations

"""
Build an alternative anchor set and run a holdout test.

This script finds simple round-trip patterns inside a block (same sender enters and exits),
and checks if the models rank those cases high when they were not used in training.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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
from src.models.baselines.logistic_regression_model import (  # noqa: E402
    generate_logistic_regression_predictions,
    train_logistic_regression_model,
)
from src.models.baselines.random_forest_model import (  # noqa: E402
    DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS,
    NO_LEAKAGE_RANDOM_FOREST_FEATURE_COLUMNS,
    generate_random_forest_predictions,
    prepare_random_forest_dataset,
    train_random_forest_model,
)
from src.models.baselines.xgboost_model import (  # noqa: E402
    generate_xgboost_predictions,
    train_xgboost_model,
)


@dataclass
class EconomicAnchorConfig:
    # Position neutrality (approx): abs(net_amount0) <= frac * max(abs(entry), abs(exit))
    net_position_frac: float = 0.20
    # Minimum temporary tick movement during the round-trip window.
    tick_excursion_min: float = 15.0
    # Reversion: abs(tick_exit - tick_entry) <= frac * excursion
    tick_reversion_frac: float = 0.40
    # Higher gas (approx) vs median of swaps between entry and exit.
    gas_multiplier: float = 1.10


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Economic anchor holdout experiment (alternative sandwich definition).")
    p.add_argument(
        "--swaps",
        default=str(PROJECT_ROOT / "data" / "processed" / "swaps_clean.parquet"),
        help="Path to swaps_clean.parquet",
    )
    p.add_argument(
        "--event-features",
        default=str(PROJECT_ROOT / "data" / "features" / "event_features.parquet"),
        help="Path to event_features.parquet",
    )
    p.add_argument(
        "--event-labels",
        default=str(PROJECT_ROOT / "data" / "labels" / "event_labels.parquet"),
        help="Path to event_labels.parquet",
    )
    p.add_argument(
        "--out-md",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "economic_anchor_holdout_eval.md"),
        help="Output markdown report path",
    )
    p.add_argument(
        "--no-leakage-features",
        action="store_true",
        help="Use no_leakage feature set (drops rule-like features).",
    )
    p.add_argument("--suffix", default="", help="Optional filename suffix for outputs, e.g. _no_leakage.")
    return p.parse_args(argv)


def _safe_float(value: Any) -> float | None:
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


def _median_percentile_vs_non_anchor(*, non_anchor_scores: pd.Series, victim_scores: pd.Series) -> float | None:
    if non_anchor_scores.empty or victim_scores.empty:
        return None
    s_non = pd.to_numeric(non_anchor_scores, errors="coerce").dropna().astype(float)
    s_vic = pd.to_numeric(victim_scores, errors="coerce").dropna().astype(float)
    if s_non.empty or s_vic.empty:
        return None
    pct = s_vic.apply(lambda v: float((s_non <= v).mean()))
    return float(pct.median()) if not pct.empty else None


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _build_case_id(*, pool: str, block: int, attacker: str, entry_swap_id: str, exit_swap_id: str) -> str:
    return f"econ|{pool}|{block}|{attacker}|{entry_swap_id}|{exit_swap_id}"


def extract_economic_anchor_cases(swaps_df: pd.DataFrame, *, config: EconomicAnchorConfig) -> pd.DataFrame:
    """
    Build a cases dataframe with columns:
    - anchor_id
    - pool_address, block_number
    - attacker_address
    - tx1_id (entry), tx2_id (middle swap), tx3_id (exit)
    - tick_excursion, tick_reversion_abs
    - gas_entry, gas_exit, gas_median_between
    - net_amount0, net_amount1
    """
    required = [
        "swap_id",
        "pool_address",
        "block_number",
        "log_index",
        "sender_address",
        "amount0",
        "amount1",
        "tick",
        "gas_price_gwei",
        "abs_amount0",
        "abs_amount1",
    ]
    missing = [c for c in required if c not in swaps_df.columns]
    if missing:
        raise ValueError(f"swaps dataframe missing required columns: {missing}")

    df = swaps_df.copy()
    for c in ["block_number", "log_index", "tick", "gas_price_gwei", "amount0", "amount1", "abs_amount0", "abs_amount1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["pool_address", "block_number", "log_index", "sender_address", "swap_id"]).copy()

    df = df.sort_values(["pool_address", "block_number", "log_index", "swap_id"]).reset_index(drop=True)

    cases: list[dict[str, Any]] = []

    for (pool, block), g in df.groupby(["pool_address", "block_number"], sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue

        # Precompute arrays for speed
        swap_ids = g["swap_id"].astype("string").tolist()
        senders = g["sender_address"].astype("string").tolist()
        amount0 = g["amount0"].astype(float).to_numpy()
        amount1 = g["amount1"].astype(float).to_numpy()
        ticks = g["tick"].astype(float).to_numpy()
        gas = g["gas_price_gwei"].astype(float).to_numpy()
        abs0 = g["abs_amount0"].astype(float).to_numpy()
        abs1 = g["abs_amount1"].astype(float).to_numpy()

        # Indices per sender
        idx_by_sender: dict[str, list[int]] = {}
        for i, s in enumerate(senders):
            idx_by_sender.setdefault(str(s), []).append(i)

        for attacker, idxs in idx_by_sender.items():
            if len(idxs) < 2:
                continue

            # Consider adjacent pairs of the attacker's own swaps in this block (simpler, avoids O(n^2)).
            for a_i, b_i in zip(idxs[:-1], idxs[1:], strict=False):
                if b_i <= a_i:
                    continue
                # Must be at least one other swap between
                if (b_i - a_i) < 2:
                    continue

                s0 = _sign(float(amount0[a_i]))
                s1 = _sign(float(amount0[b_i]))
                if s0 == 0 or s1 == 0 or s0 == s1:
                    continue

                # Ensure there is at least one non-attacker swap between
                between = list(range(a_i + 1, b_i))
                if not between:
                    continue
                if all(str(senders[k]) == attacker for k in between):
                    continue

                # Position neutrality in token0
                net0 = float(amount0[a_i] + amount0[b_i])
                net1 = float(amount1[a_i] + amount1[b_i])
                denom0 = max(float(abs(amount0[a_i])), float(abs(amount0[b_i])), 1e-9)
                if abs(net0) > float(config.net_position_frac) * denom0:
                    continue

                # Tick excursion & reversion
                base_tick = float(ticks[a_i])
                window_ticks = ticks[a_i : b_i + 1]
                excursion = float(np.max(np.abs(window_ticks - base_tick)))
                if excursion < float(config.tick_excursion_min):
                    continue
                reversion_abs = float(abs(float(ticks[b_i]) - base_tick))
                if reversion_abs > float(config.tick_reversion_frac) * excursion:
                    continue

                # Higher gas vs median between (excluding attacker swaps)
                between_idx_non_attacker = [k for k in between if str(senders[k]) != attacker]
                if not between_idx_non_attacker:
                    continue
                gas_median_between = float(np.median(gas[between_idx_non_attacker]))
                gas_entry = float(gas[a_i])
                gas_exit = float(gas[b_i])
                gas_ok = False
                if gas_median_between > 0:
                    gas_ok = (gas_entry >= gas_median_between * float(config.gas_multiplier)) or (
                        gas_exit >= gas_median_between * float(config.gas_multiplier)
                    )
                if not gas_ok:
                    continue

                # Victim pick: largest swap (abs_amount0) between entry/exit (non-attacker)
                victim_idx = max(
                    between_idx_non_attacker,
                    key=lambda k: float(abs0[k]) if np.isfinite(abs0[k]) else 0.0,
                )

                cases.append(
                    {
                        "anchor_id": _build_case_id(
                            pool=str(pool),
                            block=int(block),
                            attacker=str(attacker),
                            entry_swap_id=str(swap_ids[a_i]),
                            exit_swap_id=str(swap_ids[b_i]),
                        ),
                        "pool_address": str(pool),
                        "block_number": int(block),
                        "attacker_address": str(attacker),
                        "tx1_id": str(swap_ids[a_i]),
                        "tx2_id": str(swap_ids[victim_idx]),
                        "tx3_id": str(swap_ids[b_i]),
                        "tx1_log_index": int(g["log_index"].iloc[a_i]),
                        "tx2_log_index": int(g["log_index"].iloc[victim_idx]),
                        "tx3_log_index": int(g["log_index"].iloc[b_i]),
                        "tick_entry": base_tick,
                        "tick_exit": float(ticks[b_i]),
                        "tick_excursion": excursion,
                        "tick_reversion_abs": reversion_abs,
                        "gas_entry": gas_entry,
                        "gas_exit": gas_exit,
                        "gas_median_between": gas_median_between,
                        "net_amount0_attacker_pair": net0,
                        "net_amount1_attacker_pair": net1,
                        "victim_abs_amount0": float(abs0[victim_idx]),
                        "victim_abs_amount1": float(abs1[victim_idx]),
                        "definition": "economic_round_trip_v1",
                    }
                )

    cases_df = pd.DataFrame(cases)
    if cases_df.empty:
        return cases_df

    # Deduplicate aggressively: keep one anchor per (pool, block, attacker) with strongest tick_excursion.
    cases_df = (
        cases_df.sort_values(["tick_excursion"], ascending=False)
        .drop_duplicates(subset=["pool_address", "block_number", "attacker_address"], keep="first")
        .reset_index(drop=True)
    )
    return cases_df


def _build_anchor_maps(cases_df: pd.DataFrame) -> tuple[pd.DataFrame, set[str], set[str]]:
    """
    Returns:
    - anchors_df: per-swap mapping (swap_id -> role, case id)
    - all_anchor_swap_ids: tx1+tx2+tx3 ids
    - victim_anchor_swap_ids: tx2 ids only
    """
    required = ["anchor_id", "tx1_id", "tx2_id", "tx3_id", "block_number", "pool_address"]
    missing = [c for c in required if c not in cases_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cases_df: {missing}")

    rows: list[dict[str, Any]] = []
    for _, row in cases_df.iterrows():
        seq = str(row["anchor_id"])
        block_number = int(row["block_number"])
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


def _build_feature_columns(*, no_leakage: bool) -> list[str]:
    if no_leakage:
        return list(NO_LEAKAGE_RANDOM_FOREST_FEATURE_COLUMNS)
    return list(DEFAULT_RANDOM_FOREST_FEATURE_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    # If argv is None, parse from CLI args (sys.argv[1:]) to support normal script usage.
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    load_environment()
    setup_logging()

    swaps_path = Path(args.swaps)
    event_features_path = Path(args.event_features)
    event_labels_path = Path(args.event_labels)
    out_md_path = Path(args.out_md)

    suffix = str(args.suffix or "")
    if bool(args.no_leakage_features) and not suffix:
        suffix = "_no_leakage_features"
    if suffix:
        out_md_path = out_md_path.with_name(out_md_path.stem + suffix + out_md_path.suffix)

    if not swaps_path.exists():
        print(f"Missing swaps: {swaps_path}")
        return 2
    if not event_features_path.exists():
        print(f"Missing event features: {event_features_path}")
        return 2
    if not event_labels_path.exists():
        print(f"Missing event labels: {event_labels_path}")
        return 2

    swaps_df = pd.read_parquet(swaps_path)
    cases_df = extract_economic_anchor_cases(swaps_df, config=EconomicAnchorConfig())

    cases_out = PROJECT_ROOT / "data" / "labels" / "economic_anchor_cases.parquet"
    ensure_directory(cases_out.parent)
    cases_df.to_parquet(cases_out, index=False)
    print(f"Wrote cases: {cases_out} (rows={len(cases_df)})")

    if cases_df.empty:
        print("No economic anchors found with current config.")
        return 0

    anchors_df, all_anchor_ids, victim_anchor_ids = _build_anchor_maps(cases_df)

    event_features_df = pd.read_parquet(event_features_path)
    event_labels_df = pd.read_parquet(event_labels_path)

    feature_columns = _build_feature_columns(no_leakage=bool(args.no_leakage_features))

    # Build the training pool dataset (normal vs weak_anomaly only).
    full_dataset_df = prepare_random_forest_dataset(
        event_features_df=event_features_df,
        event_labels_df=event_labels_df,
        feature_columns=feature_columns,
    )

    # Mark anchors and build holdout pools.
    full_dataset_df["is_anchor"] = full_dataset_df["swap_id"].astype("string").isin(all_anchor_ids).astype(int)
    train_pool_df = full_dataset_df[full_dataset_df["is_anchor"] == 0].copy()

    # For evaluation, we score the anchor swaps and align the feature columns.
    anchor_eval_df = event_features_df[event_features_df["swap_id"].astype("string").isin(all_anchor_ids)].copy()
    anchor_eval_df = anchor_eval_df.merge(anchors_df, on="swap_id", how="left")
    anchor_victim_eval_df = anchor_eval_df[anchor_eval_df["swap_id"].astype("string").isin(victim_anchor_ids)].copy()

    # The baseline prediction helpers expect these columns even when we do unlabeled evaluation.
    if "timestamp" not in anchor_eval_df.columns:
        anchor_eval_df["timestamp"] = pd.NA
    anchor_eval_df["label_class"] = "anchor"
    anchor_eval_df["binary_target"] = 0

    # Coerce missing feature columns to 0.0 (consistent with other baselines).
    for c in feature_columns:
        if c not in anchor_eval_df.columns:
            anchor_eval_df[c] = 0.0
        anchor_eval_df[c] = pd.to_numeric(anchor_eval_df[c], errors="coerce")
    anchor_eval_df[feature_columns] = anchor_eval_df[feature_columns].fillna(0.0)

    # Train and evaluate models on anchor set.
    rf_artifacts = train_random_forest_model(dataset_df=train_pool_df, feature_columns=feature_columns)
    rf_threshold = _safe_float(rf_artifacts.validation_metrics.get("selected_threshold")) or 0.5
    rf_model_path = PROJECT_ROOT / "outputs" / "reports" / f"random_forest_model_weak_only_econ{suffix}.joblib"
    ensure_directory(rf_model_path.parent)
    joblib.dump(rf_artifacts.model, rf_model_path)

    rf_anchor_pred = generate_random_forest_predictions(
        model=rf_artifacts.model,
        dataframe=anchor_eval_df,
        feature_columns=rf_artifacts.feature_columns,
        threshold=float(rf_threshold),
    )
    rf_victim_pred = rf_anchor_pred[rf_anchor_pred["swap_id"].astype("string").isin(victim_anchor_ids)].copy()
    rf_any_hit_rate = float(rf_anchor_pred["rf_predicted_flag"].mean()) if len(rf_anchor_pred) else None
    rf_victim_hit_rate = float(rf_victim_pred["rf_predicted_flag"].mean()) if len(rf_victim_pred) else None

    rf_non_anchor_pred = generate_random_forest_predictions(
        model=rf_artifacts.model,
        dataframe=train_pool_df,
        feature_columns=rf_artifacts.feature_columns,
        threshold=float(rf_threshold),
    )
    rf_victim_percentile_median = _median_percentile_vs_non_anchor(
        non_anchor_scores=rf_non_anchor_pred["rf_probability"],
        victim_scores=rf_victim_pred["rf_probability"],
    )

    lr_artifacts = train_logistic_regression_model(dataset_df=train_pool_df, feature_columns=feature_columns)
    lr_threshold = _safe_float(lr_artifacts.validation_metrics.get("selected_threshold")) or 0.5
    lr_model_path = PROJECT_ROOT / "outputs" / "reports" / f"logistic_regression_model_weak_only_econ{suffix}.joblib"
    joblib.dump(lr_artifacts.model, lr_model_path)

    lr_anchor_pred = generate_logistic_regression_predictions(
        model=lr_artifacts.model,
        dataframe=anchor_eval_df,
        feature_columns=lr_artifacts.feature_columns,
        threshold=float(lr_threshold),
    )
    lr_victim_pred = lr_anchor_pred[lr_anchor_pred["swap_id"].astype("string").isin(victim_anchor_ids)].copy()
    lr_any_hit_rate = float(lr_anchor_pred["lr_predicted_flag"].mean()) if len(lr_anchor_pred) else None
    lr_victim_hit_rate = float(lr_victim_pred["lr_predicted_flag"].mean()) if len(lr_victim_pred) else None

    lr_non_anchor_pred = generate_logistic_regression_predictions(
        model=lr_artifacts.model,
        dataframe=train_pool_df,
        feature_columns=lr_artifacts.feature_columns,
        threshold=float(lr_threshold),
    )
    lr_victim_percentile_median = _median_percentile_vs_non_anchor(
        non_anchor_scores=lr_non_anchor_pred["lr_probability"],
        victim_scores=lr_victim_pred["lr_probability"],
    )

    xgb_artifacts = train_xgboost_model(dataset_df=train_pool_df, feature_columns=feature_columns)
    xgb_threshold = _safe_float(xgb_artifacts.validation_metrics.get("selected_threshold")) or 0.5
    xgb_model_path = PROJECT_ROOT / "outputs" / "reports" / f"xgboost_model_weak_only_econ{suffix}.joblib"
    joblib.dump(xgb_artifacts.model, xgb_model_path)

    xgb_anchor_pred = generate_xgboost_predictions(
        model=xgb_artifacts.model,
        dataframe=anchor_eval_df,
        feature_columns=xgb_artifacts.feature_columns,
        threshold=float(xgb_threshold),
    )
    xgb_victim_pred = xgb_anchor_pred[xgb_anchor_pred["swap_id"].astype("string").isin(victim_anchor_ids)].copy()
    xgb_any_hit_rate = float(xgb_anchor_pred["xgb_predicted_flag"].mean()) if len(xgb_anchor_pred) else None
    xgb_victim_hit_rate = float(xgb_victim_pred["xgb_predicted_flag"].mean()) if len(xgb_victim_pred) else None

    xgb_non_anchor_pred = generate_xgboost_predictions(
        model=xgb_artifacts.model,
        dataframe=train_pool_df,
        feature_columns=xgb_artifacts.feature_columns,
        threshold=float(xgb_threshold),
    )
    xgb_victim_percentile_median = _median_percentile_vs_non_anchor(
        non_anchor_scores=xgb_non_anchor_pred["xgb_probability"],
        victim_scores=xgb_victim_pred["xgb_probability"],
    )

    if_artifacts = train_isolation_forest_model(dataset_df=train_pool_df, feature_columns=feature_columns)
    if_threshold = _safe_float(if_artifacts.validation_metrics.get("selected_threshold"))
    if if_threshold is None:
        if_threshold = 0.0
    if_model_path = PROJECT_ROOT / "outputs" / "reports" / f"isolation_forest_model_weak_only_econ{suffix}.joblib"
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

    if_non_anchor_pred = generate_isolation_forest_predictions(
        model=if_artifacts.model,
        dataframe=train_pool_df,
        feature_columns=if_artifacts.feature_columns,
        threshold=float(if_threshold),
    )
    if_victim_percentile_median = _median_percentile_vs_non_anchor(
        non_anchor_scores=if_non_anchor_pred["if_anomaly_score"],
        victim_scores=if_victim_pred["if_anomaly_score"],
    )

    ensure_directory(out_md_path.parent)
    md: list[str] = []
    md.append("# Economic Anchor Holdout Evaluation")
    md.append("")
    md.append("Anchor definition: economic round-trip (not strict consecutive 3-swap rule).")
    md.append("")
    md.append("Holdout / leakage control:")
    md.append("- Exclude all anchor swaps (tx1+tx2+tx3) from weak-label training pool.")
    md.append("- Train models on remaining weak-labeled data and evaluate detection on anchors.")
    md.append("")
    md.append(f"- Feature set: **{'no-leakage-features' if bool(args.no_leakage_features) else 'full'}**")
    md.append(f"- Anchor cases: **{len(cases_df)}**")
    md.append(f"- Anchor events (tx1+tx2+tx3): **{len(all_anchor_ids)}**")
    md.append(f"- Anchor victim events (tx2): **{len(victim_anchor_ids)}**")
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
        f"| LogisticRegression (threshold={_fmt_float(lr_threshold, 3)}) | "
        f"{_percent(lr_victim_hit_rate)} | {_percent(lr_any_hit_rate)} | {_percent(lr_victim_percentile_median)} |"
    )
    md.append(
        f"| XGBoost (threshold={_fmt_float(xgb_threshold, 3)}) | "
        f"{_percent(xgb_victim_hit_rate)} | {_percent(xgb_any_hit_rate)} | {_percent(xgb_victim_percentile_median)} |"
    )
    md.append(
        f"| IsolationForest (threshold={_fmt_float(if_threshold, 6)}) | "
        f"{_percent(if_victim_hit_rate)} | {_percent(if_any_hit_rate)} | {_percent(if_victim_percentile_median)} |"
    )
    md.append("")

    out_md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote report: {out_md_path}")

    json_path = out_md_path.with_suffix(".json")
    payload = {
        "anchor_cases": int(len(cases_df)),
        "anchor_events": int(len(all_anchor_ids)),
        "anchor_victim_events": int(len(victim_anchor_ids)),
        "definition": "economic_round_trip_v1",
        "config": EconomicAnchorConfig().__dict__,
        "no_leakage_features": bool(args.no_leakage_features),
        "feature_columns": feature_columns,
        "rf": {
            "threshold": float(rf_threshold),
            "victim_hit_rate": rf_victim_hit_rate,
            "any_hit_rate": rf_any_hit_rate,
            "victim_percentile_median_vs_non_anchor": rf_victim_percentile_median,
            "validation_metrics": rf_artifacts.validation_metrics,
            "test_metrics": rf_artifacts.test_metrics,
        },
        "logistic_regression": {
            "threshold": float(lr_threshold),
            "victim_hit_rate": lr_victim_hit_rate,
            "any_hit_rate": lr_any_hit_rate,
            "victim_percentile_median_vs_non_anchor": lr_victim_percentile_median,
            "validation_metrics": lr_artifacts.validation_metrics,
            "test_metrics": lr_artifacts.test_metrics,
        },
        "xgboost": {
            "threshold": float(xgb_threshold),
            "victim_hit_rate": xgb_victim_hit_rate,
            "any_hit_rate": xgb_any_hit_rate,
            "victim_percentile_median_vs_non_anchor": xgb_victim_percentile_median,
            "validation_metrics": xgb_artifacts.validation_metrics,
            "test_metrics": xgb_artifacts.test_metrics,
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
