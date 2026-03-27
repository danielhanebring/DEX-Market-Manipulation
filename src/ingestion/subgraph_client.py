from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class SubgraphClient:
    """
    Wrapper to query Uniswap
    """

    def __init__(self, endpoint: str, timeout_seconds: int = 30) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """
        Executes GraphQL queary and returns data
        """
        response = requests.post(
            self.endpoint,
            json={"query": query, "variables": variables},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()

        if "errors" in payload:
            logger.error("GraphQL returned errors: %s", payload["errors"])
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")

        if "data" not in payload:
            raise RuntimeError("GraphQL response did not contain a 'data' field.")

        return payload["data"]