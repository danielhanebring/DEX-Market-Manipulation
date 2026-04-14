from __future__ import annotations

import logging
from typing import Any

from src.ingestion.fetch_swaps import flatten_swap_record
from src.ingestion.queries import build_swaps_by_sender_query
from src.ingestion.subgraph_client import SubgraphClient

logger = logging.getLogger(__name__)


def fetch_all_swaps_for_sender(
    client: SubgraphClient,
    sender_address: str,
    start_timestamp: int,
    end_timestamp: int,
    page_size: int = 1000,
    max_pages: int = 100,
    pool_address: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch swaps for specific sender.
    """
    include_pool = pool_address is not None
    query = build_swaps_by_sender_query(include_pool_filter=include_pool)

    all_records: list[dict[str, Any]] = []

    for page_number in range(max_pages):
        skip = page_number * page_size
        variables: dict[str, Any] = {
            "sender": sender_address.lower(),
            "startTimestamp": int(start_timestamp),
            "endTimestamp": int(end_timestamp),
            "first": int(page_size),
            "skip": int(skip),
        }
        if include_pool:
            variables["poolAddress"] = str(pool_address).lower()

        logger.info(
            "Fetching swaps by sender | sender=%s | page=%s | skip=%s",
            sender_address,
            page_number + 1,
            skip,
        )

        response_data = client.execute(query=query, variables=variables)
        raw_swaps = response_data.get("swaps", [])

        if not raw_swaps:
            break

        flattened_records = [flatten_swap_record(raw_swap) for raw_swap in raw_swaps]
        all_records.extend(flattened_records)

        if len(raw_swaps) < page_size:
            break

    return all_records

