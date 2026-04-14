from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.jaredfromsubway import (
    DEFAULT_BOT_ADDRESS,
    DetectionConfig,
    detect_sandwich_like_sequences_for_bot,
    filter_bot_interactions,
    load_swaps_table,
    write_cases_to_parquet,
)
from src.common.config import load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT

"""
    This allows to find swaps from a custom "attacker" address. In this specific case we use the address of jaredfromsubway which is one of the most known sandwich bots. 
    We found 1728 interractions with the known address in our dataset where 245 where confirmed sequences of the 3 part pattern.
"""

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detects sandwich pattern from known bot address"
        )
    )

    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "swaps_clean.parquet"),
        help="Path to processed swaps parquet (default: data/processed/swaps_clean.parquet).",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "labels" / "jared_sandwich_cases.parquet"),
        help="Output parquet path (default: data/labels/jared_sandwich_cases.parquet).",
    )
    parser.add_argument(
        "--bot",
        default=DEFAULT_BOT_ADDRESS,
        help="Bot address to search for (default: Jaredfromsubway.eth known address).",
    )
    parser.add_argument(
        "--include-origin",
        action="store_true",
        help="Count matches where origin_address == bot when checking interactions.",
    )
    parser.add_argument(
        "--bot-must-be-sender",
        action="store_true",
        help="Require tx1.sender and tx3.sender to be the bot. This is high confidence",
    )
    parser.add_argument(
        "--min-attacker-trade-size",
        type=float,
        default=0.0,
        help="Minimum attacker trade size required for tx1 and tx3.",
    )
    parser.add_argument(
        "--gas-multiplier",
        type=float,
        default=1.2,
        help="Flag suspicious gas if attacker gas >= victim gas * multiplier.",
    )
    parser.add_argument(
        "--no-reversal",
        action="store_true",
        help="Do not require tick reversal in the triple. This is lower confidence",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    load_environment()
    setup_logging()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading swaps from: {input_path}")
    df = load_swaps_table(input_path)
    print(f"Loaded swaps table shape: {df.shape}")

    bot = str(args.bot).lower()
    print(f"Bot address: {bot}")

    bot_df = filter_bot_interactions(
        dataframe=df,
        bot_address=bot,
        include_origin=bool(args.include_origin),
    )
    print(f"Total interactions (sender/recipient{'/origin' if args.include_origin else ''}): {len(bot_df)}")


    token0_col = "token0_symbol" if "token0_symbol" in bot_df.columns else None
    token1_col = "token1_symbol" if "token1_symbol" in bot_df.columns else None
    if token0_col and token1_col and not bot_df.empty:
        pairs = bot_df[[token0_col, token1_col]].drop_duplicates()
        print("Token pairs (distinct):")
        print(pairs.to_string(index=False))
    elif "pool_address" in bot_df.columns and not bot_df.empty:
        pools = bot_df[["pool_address"]].drop_duplicates()
        print("Pools (distinct pool_address):")
        print(pools.to_string(index=False))

    if bot_df.empty:
        print("\nBot address not found in the local dataset.\n")
        return 0

    detection_config = DetectionConfig(
        require_tick_reversal=not bool(args.no_reversal),
        bot_must_be_sender=bool(args.bot_must_be_sender),
        min_attacker_trade_size=float(args.min_attacker_trade_size),
        suspicious_gas_multiplier=float(args.gas_multiplier),
    )

    print("\nDetecting sandwich sequences...")
    cases_df = detect_sandwich_like_sequences_for_bot(
        dataframe=df,
        bot_address=bot,
        config=detection_config,
    )
    print(f"Detected sequences: {len(cases_df)}")

    if not cases_df.empty:
        print("\nTop targeted pairs (by count):")
        if "token0_symbol" in cases_df.columns and "token1_symbol" in cases_df.columns:
            pair_counts = (
                cases_df.groupby(["token0_symbol", "token1_symbol"])
                .size()
                .sort_values(ascending=False)
                .head(20)
            )
            print(pair_counts.to_string())
        else:
            pool_counts = (
                cases_df.groupby(["pool_address"])
                .size()
                .sort_values(ascending=False)
                .head(20)
            )
            print(pool_counts.to_string())

        if {"tick_before", "tick_after", "price_reversal_flag"}.issubset(cases_df.columns):
            cases_df["net_tick_change"] = (cases_df["tick_after"] - cases_df["tick_before"]).fillna(0.0)
            print(cases_df["net_tick_change"].describe().to_string())

        print(f"\nWriting cases to: {output_path}")
        written = write_cases_to_parquet(cases_df, output_path)
        print(f"Wrote: {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
