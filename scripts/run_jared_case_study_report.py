from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.jaredfromsubway import DEFAULT_BOT_ADDRESS
from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a case-study report for Jaredfromsubway.eth sandwich patterns.\n"
        )
    )
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "data" / "labels" / "jared_sandwich_cases.parquet"),
        help="Path to jared cases parquet.",
    )
    parser.add_argument(
        "--event-features",
        default=str(PROJECT_ROOT / "data" / "features" / "event_features.parquet"),
        help="Path to event_features parquet.",
    )
    parser.add_argument(
        "--out-md",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "jared_case_study.md"),
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--bot",
        default=DEFAULT_BOT_ADDRESS,
        help="Bot address label to include in the report.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="How many example cases to print in the report.",
    )
    parser.add_argument(
        "--normal-sample",
        type=int,
        default=5000,
        help="How many 'normal' events to sample for comparisons.",
    )
    return parser.parse_args(argv)


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt_float(value: float | None, ndigits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "NA"
    return f"{value:.{ndigits}f}"


def _direction_arrow(delta: float | None) -> str:
    if delta is None:
        return "?"
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def _pick_example_cases(cases_df: pd.DataFrame, n: int) -> pd.DataFrame:
    if cases_df.empty:
        return cases_df

    df = cases_df.copy()

    df["reversal_magnitude"] = (
        (df["tick_middle"] - df["tick_before"]).abs().fillna(0.0)
        + (df["tick_after"] - df["tick_middle"]).abs().fillna(0.0)
    )
    df["victim_size"] = pd.to_numeric(df.get("trade_size_victim"), errors="coerce").fillna(0.0)
    df["gas_pattern"] = pd.to_numeric(df.get("suspicious_gas_pattern_flag"), errors="coerce").fillna(0.0)

    df = df.sort_values(["gas_pattern", "reversal_magnitude", "victim_size"], ascending=False).reset_index(drop=True)

    return df.head(max(1, n))


def _build_profit_proxy_table(cases_df: pd.DataFrame) -> pd.DataFrame:

    df = cases_df.copy()

    df["tick_change_before"] = df["tick_middle"] - df["tick_before"]
    df["tick_change_after"] = df["tick_after"] - df["tick_middle"]
    df["net_tick_change"] = df["tick_after"] - df["tick_before"]
    df["reversal_magnitude"] = df["tick_change_before"].abs().fillna(0.0) + df["tick_change_after"].abs().fillna(0.0)
    df["net_magnitude"] = df["net_tick_change"].abs().fillna(0.0)

    attacker1 = pd.to_numeric(df.get("trade_size_attacker_1"), errors="coerce").fillna(0.0)
    attacker2 = pd.to_numeric(df.get("trade_size_attacker_2"), errors="coerce").fillna(0.0)
    df["attacker_avg_trade_size"] = ((attacker1 + attacker2) / 2.0).fillna(0.0)

    df["roundtrip_strength"] = (df["reversal_magnitude"] - df["net_magnitude"]).clip(lower=0.0)
    df["profit_proxy"] = df["roundtrip_strength"] * df["attacker_avg_trade_size"]

    keep = [
        "sequence_id",
        "block_number",
        "pool_address",
        "token0_symbol",
        "token1_symbol",
        "tick_before",
        "tick_middle",
        "tick_after",
        "tick_change_before",
        "tick_change_after",
        "net_tick_change",
        "reversal_magnitude",
        "roundtrip_strength",
        "attacker_avg_trade_size",
        "profit_proxy",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    load_environment()
    setup_logging()

    cases_path = Path(args.cases)
    event_features_path = Path(args.event_features)
    out_md_path = Path(args.out_md)

    try:
        cases_df = pd.read_parquet(cases_path)
    except Exception as exc:
        print(f"Failed to read cases parquet: {cases_path}\n{exc}")
        return 2

    if cases_df.empty:
        print("Cases dataset is empty. Run detection first:")
        return 0

    try:
        event_features_df = pd.read_parquet(event_features_path)
    except Exception as exc:
        event_features_df = pd.DataFrame()

    bot = str(args.bot).lower()

    examples_df = _pick_example_cases(cases_df, int(args.examples))

    gas_before = pd.to_numeric(cases_df.get("gas_before"), errors="coerce")
    gas_middle = pd.to_numeric(cases_df.get("gas_middle"), errors="coerce")
    gas_after = pd.to_numeric(cases_df.get("gas_after"), errors="coerce")

    gas_valid = gas_before.notna() & gas_middle.notna() & gas_after.notna()
    gb = gas_before[gas_valid]
    gm = gas_middle[gas_valid]
    ga = gas_after[gas_valid]

    gas_before_gt_middle = (gb > gm).mean() if len(gb) else float("nan")
    gas_after_gt_middle = (ga > gm).mean() if len(gb) else float("nan")

    profit_df = _build_profit_proxy_table(cases_df)
    profit_summary = profit_df["profit_proxy"].describe() if "profit_proxy" in profit_df.columns else None

    comparison_lines: list[str] = []
    if not event_features_df.empty and "swap_id" in event_features_df.columns:
        victim_ids = cases_df["tx2_id"].astype("string")
        victim_features = event_features_df[event_features_df["swap_id"].astype("string").isin(victim_ids)].copy()

        normal_pool = event_features_df.copy()
        normal_pool = normal_pool[~normal_pool["swap_id"].astype("string").isin(victim_ids)].copy()

        sample_n = min(int(args.normal_sample), len(normal_pool))
        normal_sample = normal_pool.sample(n=sample_n, random_state=7) if sample_n > 0 else normal_pool.head(0)

        feature_cols = [
            "three_event_pattern_indicator",
            "strict_sandwich_support_flag",
            "reversal_pattern_flag",
            "same_sender_before_after_flag",
            "different_middle_sender_from_neighbors_flag",
            "gas_spike_flag",
            "high_block_gas_context_flag",
            "high_relative_trade_size_flag",
            "sandwich_support_score",
            "gas_price_relative_to_block_median",
            "gas_price_relative_to_neighbors_mean",
        ]
        feature_cols = [c for c in feature_cols if c in event_features_df.columns]

        def _mean_or_median(series: pd.Series) -> str:
            s = pd.to_numeric(series, errors="coerce")
            if s.dropna().empty:
                return "NA"
            unique_vals = set(s.dropna().unique().tolist())
            if unique_vals.issubset({0, 1}):
                return _percent(float(s.mean()))
            return _fmt_float(float(s.median()), 3)

        comparison_lines.append("| Feature | Normal (sample) | Sandwich victims (Jared cases) |")
        comparison_lines.append("|---|---:|---:|")
        for col in feature_cols:
            comparison_lines.append(
                f"| `{col}` | {_mean_or_median(normal_sample[col])} | {_mean_or_median(victim_features[col])} |"
            )

    out_md_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines: list[str] = []
    md_lines.append("# Jaredfromsubway.eth Case Study (Sandwich-Like Patterns)")
    md_lines.append("")
    md_lines.append("This report describes **highly structured transaction patterns consistent with sandwich behavior**.")
    md_lines.append("It does **not** claim proven malicious intent.")
    md_lines.append("")
    md_lines.append(f"- Bot address: `{bot}`")
    md_lines.append(f"- Total detected cases: **{len(cases_df)}**")
    md_lines.append("")

    md_lines.append("## Step A: Real Cases (1-3 Examples)")
    md_lines.append("")
    for idx, row in examples_df.iterrows():
        block_number = int(row.get("block_number")) if row.get("block_number") is not None and not pd.isna(row.get("block_number")) else None
        pool_address = str(row.get("pool_address"))
        pair = f"{row.get('token0_symbol', 'NA')}/{row.get('token1_symbol', 'NA')}"
        tx1 = str(row.get("tx1_hash") or row.get("tx1_id"))
        tx2 = str(row.get("tx2_hash") or row.get("tx2_id"))
        tx3 = str(row.get("tx3_hash") or row.get("tx3_id"))
        victim = str(row.get("victim_address"))

        t1 = _safe_float(row.get("tick_before"))
        t2 = _safe_float(row.get("tick_middle"))
        t3 = _safe_float(row.get("tick_after"))
        d12 = None if t1 is None or t2 is None else (t2 - t1)
        d23 = None if t2 is None or t3 is None else (t3 - t2)

        g1 = _safe_float(row.get("gas_before"))
        g2 = _safe_float(row.get("gas_middle"))
        g3 = _safe_float(row.get("gas_after"))

        md_lines.append(f"### Case {idx + 1}")
        md_lines.append("")
        md_lines.append(f"- Block: `{block_number}`")
        md_lines.append(f"- Pool: `{pool_address}` ({pair})")
        md_lines.append(f"- TX1: `{tx1}` (attacker)")
        md_lines.append(f"- TX2: `{tx2}` (victim: `{victim}`)")
        md_lines.append(f"- TX3: `{tx3}` (attacker)")
        md_lines.append("")
        md_lines.append("Tick movement (tick after each swap):")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.append(f"TX1 tick: {t1}  ->  TX2 tick: {t2}  ->  TX3 tick: {t3}")
        md_lines.append(
            f"d12: {_fmt_float(d12, 1)} ({_direction_arrow(d12)}) | d23: {_fmt_float(d23, 1)} ({_direction_arrow(d23)})"
        )
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("Gas (gwei):")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.append(f"gas_before={_fmt_float(g1, 3)} | gas_middle={_fmt_float(g2, 3)} | gas_after={_fmt_float(g3, 3)}")
        if g1 is not None and g2 is not None:
            md_lines.append(f"gas_before > gas_middle: {g1 > g2}")
        if g3 is not None and g2 is not None:
            md_lines.append(f"gas_after  > gas_middle: {g3 > g2}")
        md_lines.append("```")
        md_lines.append("")

    md_lines.append("## Step B: Gas Analysis (Across All Detected Cases)")
    md_lines.append("")
    if len(gb):
        md_lines.append(f"- Valid gas rows: **{len(gb)} / {len(cases_df)}**")
        md_lines.append(f"- `gas_before > gas_middle`: **{_percent(float(gas_before_gt_middle))}**")
        md_lines.append(f"- `gas_after  > gas_middle`: **{_percent(float(gas_after_gt_middle))}**")
        md_lines.append("")
        md_lines.append("Interpretation:")
        md_lines.append("- Higher attacker gas is consistent with ordering pressure (front-run/back-run), but is indirect evidence.")
    else:
        md_lines.append("- Gas columns not available in the cases dataset.")
    md_lines.append("")

    md_lines.append("## Step C: Profit Estimate (Simple)")
    md_lines.append("")
    md_lines.append("We report a simple estimate based on round-trip structure: big reversal with small net move.")
    md_lines.append("")
    if profit_summary is not None:
        md_lines.append("`profit_proxy` summary:")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.append(profit_summary.to_string())
        md_lines.append("```")
    else:
        md_lines.append("Profit estimate not available (missing columns).")
    md_lines.append("")

    md_lines.append("## Step D: Compare With Normal Trades (Feature-Level)")
    md_lines.append("")
    if comparison_lines:
        md_lines.extend(comparison_lines)
    else:
        md_lines.append("Skipped (event_features.parquet not available).")
    md_lines.append("")

    md_lines.append("## ML Tie-In")
    md_lines.append("")
    md_lines.append(
        "These high-confidence Jared cases can be used as **evaluation anchors** to sanity-check models trained on weak labels."
    )
    md_lines.append("")

    out_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote report: {out_md_path}")

    print(f"Cases: {len(cases_df)} | Examples in report: {len(examples_df)}")
    if len(gb):
        print(f"gas_before > gas_middle: {_percent(float(gas_before_gt_middle))}")
        print(f"gas_after  > gas_middle: {_percent(float(gas_after_gt_middle))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
