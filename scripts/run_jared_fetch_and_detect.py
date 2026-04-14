from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.jaredfromsubway import (
    DEFAULT_BOT_ADDRESS,
    DetectionConfig,
    detect_sandwich_like_sequences_for_bot,
    write_cases_to_parquet,
)
from src.common.config import get_required_env_var, load_environment
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.ingestion.fetch_swaps import fetch_all_swaps_for_pool
from src.ingestion.fetch_swaps_by_sender import fetch_all_swaps_for_sender
from src.ingestion.save_raw_data import build_extraction_metadata, save_json_snapshot, save_records_to_parquet
from src.ingestion.subgraph_client import SubgraphClient


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Targeted fetch + detect for Jaredfromsubway.eth.\n"
        )
    )

    parser.add_argument("--bot", default=DEFAULT_BOT_ADDRESS, help="Bot address (default: Jaredfromsubway.eth known address).")
    parser.add_argument("--start-ts", type=int, required=True, help="Start timestamp (UNIX seconds, UTC).")
    parser.add_argument("--end-ts", type=int, required=True, help="End timestamp (UNIX seconds, UTC).")
    parser.add_argument("--window-seconds", type=int, default=180, help="Pool window size around bot swap timestamps (+/- seconds).")
    parser.add_argument("--max-blocks", type=int, default=200, help="Max unique (pool, block) targets to fetch context for.")
    parser.add_argument("--page-size", type=int, default=1000, help="Subgraph page size.")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages per query.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "labels" / "jared_sandwich_cases.parquet"))
    parser.add_argument("--raw-out-dir", default=str(PROJECT_ROOT / "data" / "raw" / "swaps" / "jared_blocks"))
    parser.add_argument("--endpoint-env", default="UNISWAP_V3_SUBGRAPH_URL", help="Env var for subgraph endpoint.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    load_environment()
    setup_logging()

    endpoint = get_required_env_var(args.endpoint_env)
    client = SubgraphClient(endpoint=endpoint, timeout_seconds=30)

    bot = str(args.bot).lower()
    start_ts = int(args.start_ts)
    end_ts = int(args.end_ts)
    window = int(args.window_seconds)

    raw_out_dir = ensure_directory(args.raw_out_dir)
    meta_out_dir = ensure_directory(PROJECT_ROOT / "data" / "raw" / "metadata")

    print(f"Fetching bot swaps by sender for {bot} in [{start_ts}, {end_ts}) ...")
    bot_swaps = fetch_all_swaps_for_sender(
        client=client,
        sender_address=bot,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        page_size=int(args.page_size),
        max_pages=int(args.max_pages),
        pool_address=None,
    )

    if not bot_swaps:
        print("No bot swaps found in this time range via sender filter.")
        return 0

    bot_df = pd.DataFrame(bot_swaps)
    print(f"Bot swaps fetched: {len(bot_df)}")

    targets: dict[tuple[str, int], int] = {}
    for _, row in bot_df.iterrows():
        pool_address = str(row.get("pool_address") or "").lower()
        block_number = int(row.get("block_number")) if row.get("block_number") is not None and not pd.isna(row.get("block_number")) else None
        ts = int(row.get("timestamp")) if row.get("timestamp") is not None and not pd.isna(row.get("timestamp")) else None
        if not pool_address or block_number is None or ts is None:
            continue
        targets.setdefault((pool_address, block_number), ts)

    target_items = list(targets.items())[: int(args.max_blocks)]
    print(f"Unique (pool, block) targets: {len(targets)} | processing: {len(target_items)}")

    context_records: list[dict] = []
    seen_swap_ids: set[str] = set()

    for (pool_address, block_number), ts in target_items:
        window_start = max(0, ts - window)
        window_end = ts + window

        swaps = fetch_all_swaps_for_pool(
            client=client,
            pool_address=pool_address,
            start_timestamp=window_start,
            end_timestamp=window_end,
            page_size=int(args.page_size),
            max_pages=int(args.max_pages),
        )

        for rec in swaps:
            if rec.get("block_number") != block_number:
                continue
            swap_id = str(rec.get("swap_id") or "")
            if not swap_id or swap_id in seen_swap_ids:
                continue
            seen_swap_ids.add(swap_id)
            context_records.append(rec)

    if not context_records:
        return 0

    extraction_id = f"jared_blocks_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    raw_parquet = Path(raw_out_dir) / f"{extraction_id}_swaps_raw.parquet"
    save_records_to_parquet(context_records, raw_parquet)

    meta = build_extraction_metadata(
        pool_name="jared_blocks_multi_pool",
        pool_address="multiple",
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        page_size=int(args.page_size),
        total_records=len(context_records),
    )
    meta["notes"] = {
        "bot": bot,
        "window_seconds": window,
        "max_blocks": int(args.max_blocks),
        "source": "seed_by_sender_then_fetch_pool_windows_then_filter_block",
    }
    meta_file = Path(meta_out_dir) / f"{extraction_id}_metadata.json"
    save_json_snapshot(meta, meta_file)

    print(f"Context swaps saved: {raw_parquet}")
    print(f"Metadata saved: {meta_file}")

    context_df = pd.DataFrame(context_records)

    detection_config = DetectionConfig(
        bot_must_be_sender=True,
        require_tick_reversal=True,
        require_same_block=True,
        require_same_pool=True,
    )
    cases_df = detect_sandwich_like_sequences_for_bot(
        dataframe=context_df,
        bot_address=bot,
        config=detection_config,
    )

    print(f"Detected sandwich cases: {len(cases_df)}")
    if not cases_df.empty:
        output_path = Path(args.output)
        ensure_directory(output_path.parent)
        written = write_cases_to_parquet(cases_df, output_path)
        print(f"Wrote cases to: {written}")

        counts = cases_df.groupby("pool_address").size().sort_values(ascending=False).head(20)
        print("\nTop pools by detected cases:")
        print(counts.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
