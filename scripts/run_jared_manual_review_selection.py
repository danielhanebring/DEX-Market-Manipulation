from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.paths import PROJECT_ROOT, ensure_directory


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select top Jared cases for manual review using a chosen score column (rf_probability/lr_probability/if_anomaly_score).\n"
            "Reads the scored, UNLABELED Jared table and writes a manual-review template + block context export.\n"
        )
    )
    parser.add_argument(
        "--scored",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "jared_unlabeled_scored.parquet"),
        help="Path to scored Jared table (parquet preferred).",
    )
    parser.add_argument(
        "--swaps",
        default=str(PROJECT_ROOT / "data" / "processed" / "swaps_clean.parquet"),
        help="Path to swaps_clean parquet (used to export full block context).",
    )
    parser.add_argument(
        "--rank-col",
        default="rf_probability",
        choices=["rf_probability", "lr_probability", "if_anomaly_score"],
        help="Which score column to rank by when selecting top events.",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=0.02,
        help="Top fraction of events to include in the manual-review template (e.g. 0.02 = top 2 percent).",
    )
    parser.add_argument(
        "--top-blocks",
        type=int,
        default=40,
        help="How many unique top blocks to export for manual inspection.",
    )
    parser.add_argument(
        "--out-prefix",
        default="jared",
        help="Prefix for output filenames under outputs/reports (e.g. jared_rf, jared_lr).",
    )
    return parser.parse_args(argv)


def _read_scored(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    scored_path = Path(args.scored)
    swaps_path = Path(args.swaps)
    out_dir = PROJECT_ROOT / "outputs" / "reports"

    if not scored_path.exists():
        raise SystemExit(f"Missing scored file: {scored_path}")
    if not swaps_path.exists():
        raise SystemExit(f"Missing swaps_clean file: {swaps_path}")

    ensure_directory(out_dir)

    df = _read_scored(scored_path)

    rank_col = str(args.rank_col)
    if rank_col not in df.columns:
        raise SystemExit(f"Rank column '{rank_col}' not found in {scored_path}")

    top_pct = float(args.top_pct)
    if not (0.0 < top_pct < 1.0):
        raise SystemExit("--top-pct must be between 0 and 1.")

    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    df = df.dropna(subset=[rank_col]).copy()
    df = df.sort_values(rank_col, ascending=False).reset_index(drop=True)

    top_n = max(1, int(math.ceil(top_pct * len(df))))
    top_df = df.head(top_n).copy()

    top_df["block_number"] = pd.to_numeric(top_df.get("block_number"), errors="coerce")
    unique_blocks = (
        top_df.dropna(subset=["block_number"])
        .drop_duplicates(subset=["block_number"], keep="first")
        .head(int(args.top_blocks))
    )
    blocks = [int(b) for b in unique_blocks["block_number"].tolist()]

    swaps_df = pd.read_parquet(swaps_path)
    ctx_df = swaps_df[swaps_df["block_number"].isin(blocks)].copy()
    ctx_df = ctx_df.sort_values(["block_number", "log_index", "swap_id"]).reset_index(drop=True)

    prefix = str(args.out_prefix)
    ctx_parquet = out_dir / f"{prefix}_top_blocks_swaps_{rank_col}.parquet"
    ctx_csv = out_dir / f"{prefix}_top_blocks_swaps_{rank_col}.csv"
    ctx_df.to_parquet(ctx_parquet, index=False)
    ctx_df.to_csv(ctx_csv, index=False)

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

    keep = [c for c in template_cols if c in top_df.columns]
    template = top_df[keep].copy()
    template.insert(0, "case_id", [f"{prefix}_{rank_col}_case_{i+1:03d}" for i in range(len(template))])

    for col in ["sandwich_pattern_yes_no", "price_reversal_yes_no", "gas_spike_yes_no", "notes", "verdict"]:
        if col not in template.columns:
            template[col] = ""

    template = template.reindex(columns=template_cols)

    out_template = out_dir / f"{prefix}_manual_review_template_{rank_col}.csv"
    template.to_csv(out_template, index=False)

    print("Wrote outputs:")
    print(f"- {out_template}")
    print(f"- {ctx_parquet}")
    print(f"- {ctx_csv}")
    print(f"Top pct: {top_pct} -> top_n={top_n} rows")
    print(f"Unique blocks exported: {len(blocks)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

