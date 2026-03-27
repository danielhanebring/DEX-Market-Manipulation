from __future__ import annotations


def build_swaps_query() -> str:
    """
    Build the GraphQL query used to fetch Uniswap V3 swap events.

    Returns:
        GraphQL query string.
    """
    return """
    query FetchSwaps(
      $poolAddress: String!,
      $startTimestamp: BigInt!,
      $endTimestamp: BigInt!,
      $first: Int!,
      $skip: Int!
    ) {
      swaps(
        first: $first
        skip: $skip
        orderBy: timestamp
        orderDirection: asc
        where: {
          pool: $poolAddress,
          timestamp_gte: $startTimestamp,
          timestamp_lt: $endTimestamp
        }
      ) {
        id
        logIndex
        sender
        recipient
        origin
        amount0
        amount1
        sqrtPriceX96
        tick
        transaction {
          id
          blockNumber
          timestamp
          gasPrice
        }
        pool {
          id
          feeTier
          token0 {
            id
            symbol
            decimals
          }
          token1 {
            id
            symbol
            decimals
          }
        }
      }
    }
    """