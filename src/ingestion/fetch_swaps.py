from __future__ import annotations

import logging
from typing import Any

from src.ingestion.queries import build_swaps_query
from src.ingestion.subgraph_client import SubgraphClient

logger = logging.getLogger(__name__)


def flatten_swap_record(raw_swap: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested swap object to raw record
    """
    transaction = raw_swap.get("transaction", {})
    pool = raw_swap.get("pool", {})
    token0 = pool.get("token0", {})
    token1 = pool.get("token1", {})

    return {
        "swap_id": raw_swap.get("id"),
        "pool_address": pool.get("id"),
        "transaction_hash": transaction.get("id"),
        "log_index": _safe_int(raw_swap.get("logIndex")),
        "block_number": _safe_int(transaction.get("blockNumber")),
        "timestamp": _safe_int(transaction.get("timestamp")),
        "sender_address": raw_swap.get("sender"),
        "recipient_address": raw_swap.get("recipient"),
        "origin_address": raw_swap.get("origin"),
        "amount0": _safe_float(raw_swap.get("amount0")),
        "amount1": _safe_float(raw_swap.get("amount1")),
        "sqrt_price_x96_raw": raw_swap.get("sqrtPriceX96"),
        "tick": _safe_int(raw_swap.get("tick")),
        "gas_price_raw": transaction.get("gasPrice"),
        "token0_id": token0.get("id"),
        "token0_symbol": token0.get("symbol"),
        "token0_decimals": _safe_int(token0.get("decimals")),
        "token1_id": token1.get("id"),
        "token1_symbol": token1.get("symbol"),
        "token1_decimals": _safe_int(token1.get("decimals")),
        "fee_tier": _safe_int(pool.get("feeTier")),
    }


def fetch_all_swaps_for_pool(
    client: SubgraphClient,
    pool_address: str,
    start_timestamp: int,
    end_timestamp: int,
    page_size: int = 1000,
    max_pages: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch swap event pages from liquidity pool and returns list of flattened swap records
    """
    query = build_swaps_query()
    all_records: list[dict[str, Any]] = []

    for page_number in range(max_pages):
        skip = page_number * page_size
        variables = {
            "poolAddress": pool_address.lower(),
            "startTimestamp": start_timestamp,
            "endTimestamp": end_timestamp,
            "first": page_size,
            "skip": skip,
        }

        logger.info(
            "Fetching swaps | pool=%s | page=%s | skip=%s",
            pool_address,
            page_number + 1,
            skip,
        )

        response_data = client.execute(query=query, variables=variables)
        raw_swaps = response_data.get("swaps", [])

        if not raw_swaps:
            logger.info("No more swaps returned for pool %s", pool_address)
            break

        flattened_records = [flatten_swap_record(raw_swap) for raw_swap in raw_swaps]
        all_records.extend(flattened_records)

        if len(raw_swaps) < page_size:
            logger.info("Reached final page for pool %s", pool_address)
            break

    return all_records


def _safe_int(value: Any) -> int | None:
    """Convert a value to int when possible, otherwise return None"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
    
def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible, otherwise return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None