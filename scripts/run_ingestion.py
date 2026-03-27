from __future__ import annotations

import logging
from pathlib import Path

from src.common.config import (
    get_required_env_var,
    load_environment,
    load_yaml_config,
)
from src.common.logging_utils import setup_logging
from src.common.paths import PROJECT_ROOT, ensure_directory
from src.ingestion.fetch_swaps import fetch_all_swaps_for_pool
from src.ingestion.save_raw_data import (
    build_extraction_metadata,
    save_json_snapshot,
    save_records_to_parquet,
)
from src.ingestion.subgraph_client import SubgraphClient

logger = logging.getLogger(__name__)


def main() -> None:
    """Run raw swap ingestion for all selected pools."""
    load_environment()
    setup_logging()

    config_path = PROJECT_ROOT / "configs" / "data.yaml"
    config = load_yaml_config(config_path)

    endpoint_env_var = config["source"]["endpoint_env_var"]
    endpoint = get_required_env_var(endpoint_env_var)

    ingestion_config = config["ingestion"]
    output_config = config["output"]
    time_range = config["time_range"]
    selected_pools = config["pools"]["selected"]

    page_size = int(ingestion_config["page_size"])
    max_pages = int(ingestion_config["max_pages"])
    timeout_seconds = int(ingestion_config["request_timeout_seconds"])

    raw_swaps_dir = ensure_directory(PROJECT_ROOT / output_config["raw_swaps_dir"])
    metadata_dir = ensure_directory(PROJECT_ROOT / output_config["metadata_dir"])
    write_json_snapshot = bool(config["storage"].get("write_json_snapshot", True))

    client = SubgraphClient(endpoint=endpoint, timeout_seconds=timeout_seconds)

    for pool in selected_pools:
        pool_name = pool["name"]
        pool_address = pool["address"]

        logger.info("Starting swap ingestion for pool: %s (%s)", pool_name, pool_address)

        swap_records = fetch_all_swaps_for_pool(
            client=client,
            pool_address=pool_address,
            start_timestamp=int(time_range["start_timestamp"]),
            end_timestamp=int(time_range["end_timestamp"]),
            page_size=page_size,
            max_pages=max_pages,
        )

        if not swap_records:
            logger.warning("No swap records found for pool %s", pool_name)
            continue

        parquet_file = raw_swaps_dir / f"{pool_name.lower()}_swaps_raw.parquet"
        save_records_to_parquet(swap_records, parquet_file)

        metadata = build_extraction_metadata(
            pool_name=pool_name,
            pool_address=pool_address,
            start_timestamp=int(time_range["start_timestamp"]),
            end_timestamp=int(time_range["end_timestamp"]),
            page_size=page_size,
            total_records=len(swap_records),
        )

        metadata_file = metadata_dir / f"{pool_name.lower()}_swaps_metadata.json"
        save_json_snapshot(metadata, metadata_file)

        if write_json_snapshot:
            snapshot_file = metadata_dir / f"{pool_name.lower()}_swaps_sample.json"
            sample_payload = {
                "pool_name": pool_name,
                "pool_address": pool_address,
                "record_count": len(swap_records),
                "sample_records": swap_records[:5],
            }
            save_json_snapshot(sample_payload, snapshot_file)

        logger.info(
            "Finished pool %s | saved %s records to %s",
            pool_name,
            len(swap_records),
            parquet_file,
        )


if __name__ == "__main__":
    main()