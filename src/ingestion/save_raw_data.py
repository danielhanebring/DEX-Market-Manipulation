from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.paths import ensure_directory


def save_records_to_parquet(records: list[dict[str, Any]], output_file: str | Path) -> Path:
    """
    Takes list of records and saves to parquet file. Returns path to file.
    """
    output_path = Path(output_file)
    ensure_directory(output_path.parent)

    dataframe = pd.DataFrame(records)
    dataframe.to_parquet(output_path, index=False)

    return output_path


def save_json_snapshot(payload: dict[str, Any], output_file: str | Path) -> Path:
    """
    Saves dictionary as JSON, used for debugging
    """
    output_path = Path(output_file)
    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return output_path


def build_extraction_metadata(
    pool_name: str,
    pool_address: str,
    start_timestamp: int,
    end_timestamp: int,
    page_size: int,
    total_records: int,
) -> dict[str, Any]:
    """
    Build metadata for a raw extraction run.
    """
    return {
        "source_name": "uniswap_v3_subgraph",
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "pool_name": pool_name,
        "pool_address": pool_address,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "page_size": page_size,
        "total_records": total_records,
    }