from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BOT_ADDRESS = "0xAE2Fc483527B8EF99EB5D9B44875F005ba1FaE13"


@dataclass(frozen=True)
class SandwichCase:
    """
    One detected 3-swap sandwich-like sequence attributed to a known bot address.

    This is a pattern dataset (not proof of intent).
    """

    sequence_id: str
    block_number: int
    pool_address: str
    token0_symbol: str | None
    token1_symbol: str | None
    attacker_address: str
    victim_address: str
    tx1_id: str
    tx2_id: str
    tx3_id: str


def _lower_address(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).lower()


def _lower_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower()


def _get_first_existing_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in dataframe.columns:
            return name
    return None


def load_swaps_table(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Parquet support is missing in the environment. "
        ) from exc


def filter_bot_interactions(
    dataframe: pd.DataFrame,
    bot_address: str,
    include_origin: bool = True,
) -> pd.DataFrame:
    """
    Return rows where bot address appears
    """
    bot = bot_address.lower()

    if "sender_address" not in dataframe.columns or "recipient_address" not in dataframe.columns:
        raise ValueError("Expected columns: sender_address, recipient_address")

    sender = _lower_series(dataframe["sender_address"])
    recipient = _lower_series(dataframe["recipient_address"])

    mask = (sender == bot) | (recipient == bot)

    if include_origin and "origin_address" in dataframe.columns:
        origin = _lower_series(dataframe["origin_address"])
        mask = mask | (origin == bot)

    return dataframe[mask].copy()


@dataclass(frozen=True)
class DetectionConfig:
    require_same_pool: bool = True
    require_same_block: bool = True
    require_tick_reversal: bool = True
    bot_must_be_sender: bool = True
    require_victim_sender_not_bot: bool = True
    min_attacker_trade_size: float = 0.0
    suspicious_gas_multiplier: float = 1.2


def _infer_tick_column(dataframe: pd.DataFrame) -> str:
    tick_col = _get_first_existing_column(dataframe, ["tick_after", "tick"])
    if tick_col is None:
        raise ValueError("Expected a tick column: tick_after or tick")
    return tick_col


def _infer_gas_column(dataframe: pd.DataFrame) -> str | None:
    return _get_first_existing_column(dataframe, ["gas_price_gwei"])


def _infer_trade_size_columns(dataframe: pd.DataFrame) -> tuple[str | None, str | None]:
    amount0 = _get_first_existing_column(dataframe, ["amount0_normalized", "amount0"])
    amount1 = _get_first_existing_column(dataframe, ["amount1_normalized", "amount1"])
    return amount0, amount1


def _trade_size_for_row(row: pd.Series, amount0_col: str | None, amount1_col: str | None) -> float | None:
    for col in [amount0_col, amount1_col]:
        if col and col in row.index:
            value = row.get(col)
            try:
                if value is None or pd.isna(value):
                    continue
                return float(abs(float(value)))
            except Exception:
                continue
    return None


def _build_sequence_id(tx1_id: str, tx2_id: str, tx3_id: str) -> str:
    return f"{tx1_id}|{tx2_id}|{tx3_id}"


def detect_sandwich_like_sequences_for_bot(
    dataframe: pd.DataFrame,
    bot_address: str,
    config: DetectionConfig | None = None,
) -> pd.DataFrame:
    """
    Detect 3-swap sandwich sequences to a known bot address.

    Output: a dataset with one row per detected (tx1, tx2, tx3).
    """
    resolved = config or DetectionConfig()
    bot = bot_address.lower()

    required = ["pool_address", "block_number", "log_index", "swap_id", "sender_address"]
    missing = [c for c in required if c not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tick_col = _infer_tick_column(dataframe)
    gas_col = _infer_gas_column(dataframe)
    amount0_col, amount1_col = _infer_trade_size_columns(dataframe)

    df = dataframe.copy()
    df["sender_address_norm"] = _lower_series(df["sender_address"])
    df["recipient_address_norm"] = _lower_series(df["recipient_address"]) if "recipient_address" in df.columns else pd.Series(pd.NA, index=df.index)
    df["origin_address_norm"] = _lower_series(df["origin_address"]) if "origin_address" in df.columns else pd.Series(pd.NA, index=df.index)

    df = df.sort_values(["pool_address", "block_number", "log_index", "swap_id"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    group_keys = ["pool_address", "block_number"] if resolved.require_same_block else ["pool_address"]

    for (pool_address, block_number), group_df in df.groupby(group_keys, sort=False):
        group_df = group_df.reset_index(drop=True)

        if len(group_df) < 3:
            continue

        for i in range(len(group_df) - 2):
            tx1 = group_df.iloc[i]
            tx2 = group_df.iloc[i + 1]
            tx3 = group_df.iloc[i + 2]

            if resolved.require_same_pool:
                if not (tx1["pool_address"] == tx2["pool_address"] == tx3["pool_address"]):
                    continue

            tx1_sender = str(tx1["sender_address_norm"])
            tx2_sender = str(tx2["sender_address_norm"])
            tx3_sender = str(tx3["sender_address_norm"])

            if resolved.bot_must_be_sender:
                is_bot_pattern = (tx1_sender == bot) and (tx3_sender == bot)
            else:
                tx1_any = {tx1_sender, str(tx1.get("recipient_address_norm")), str(tx1.get("origin_address_norm"))}
                tx3_any = {tx3_sender, str(tx3.get("recipient_address_norm")), str(tx3.get("origin_address_norm"))}
                is_bot_pattern = (bot in tx1_any) and (bot in tx3_any)

            if not is_bot_pattern:
                continue

            if resolved.require_victim_sender_not_bot and tx2_sender == bot:
                continue

            tick1 = tx1.get(tick_col)
            tick2 = tx2.get(tick_col)
            tick3 = tx3.get(tick_col)

            tick1 = None if tick1 is None or pd.isna(tick1) else float(tick1)
            tick2 = None if tick2 is None or pd.isna(tick2) else float(tick2)
            tick3 = None if tick3 is None or pd.isna(tick3) else float(tick3)

            price_reversal = False
            if tick1 is not None and tick2 is not None and tick3 is not None:
                price_reversal = (tick2 > tick1 and tick3 < tick2) or (tick2 < tick1 and tick3 > tick2)

            if resolved.require_tick_reversal and not price_reversal:
                continue

            attacker_size_1 = _trade_size_for_row(tx1, amount0_col, amount1_col)
            attacker_size_2 = _trade_size_for_row(tx3, amount0_col, amount1_col)

            if (
                attacker_size_1 is None
                or attacker_size_2 is None
                or attacker_size_1 < resolved.min_attacker_trade_size
                or attacker_size_2 < resolved.min_attacker_trade_size
            ):
                continue

            victim_size = _trade_size_for_row(tx2, amount0_col, amount1_col)

            gas1 = None
            gas2 = None
            gas3 = None
            if gas_col is not None:
                gas1 = tx1.get(gas_col)
                gas2 = tx2.get(gas_col)
                gas3 = tx3.get(gas_col)
                gas1 = None if gas1 is None or pd.isna(gas1) else float(gas1)
                gas2 = None if gas2 is None or pd.isna(gas2) else float(gas2)
                gas3 = None if gas3 is None or pd.isna(gas3) else float(gas3)

            suspicious_gas_pattern_flag = 0
            if gas1 is not None and gas2 is not None and gas3 is not None and gas2 > 0:
                suspicious_gas_pattern_flag = int(
                    gas1 >= gas2 * resolved.suspicious_gas_multiplier
                    or gas3 >= gas2 * resolved.suspicious_gas_multiplier
                )

            tx1_id = str(tx1["swap_id"])
            tx2_id = str(tx2["swap_id"])
            tx3_id = str(tx3["swap_id"])

            tx1_hash = tx1.get("transaction_hash")
            tx2_hash = tx2.get("transaction_hash")
            tx3_hash = tx3.get("transaction_hash")

            tx1_hash = None if tx1_hash is None or pd.isna(tx1_hash) else str(tx1_hash)
            tx2_hash = None if tx2_hash is None or pd.isna(tx2_hash) else str(tx2_hash)
            tx3_hash = None if tx3_hash is None or pd.isna(tx3_hash) else str(tx3_hash)

            token0_symbol = tx1.get("token0_symbol") if "token0_symbol" in df.columns else None
            token1_symbol = tx1.get("token1_symbol") if "token1_symbol" in df.columns else None

            rows.append(
                {
                    "sequence_id": _build_sequence_id(tx1_id, tx2_id, tx3_id),
                    "block_number": int(tx1["block_number"]) if not pd.isna(tx1["block_number"]) else None,
                    "pool_address": str(tx1["pool_address"]),
                    "token0_symbol": None if token0_symbol is None or pd.isna(token0_symbol) else str(token0_symbol),
                    "token1_symbol": None if token1_symbol is None or pd.isna(token1_symbol) else str(token1_symbol),
                    "attacker_address": bot,
                    "victim_address": tx2_sender,
                    "tx1_hash": tx1_hash,
                    "tx2_hash": tx2_hash,
                    "tx3_hash": tx3_hash,
                    "tx1_id": tx1_id,
                    "tx2_id": tx2_id,
                    "tx3_id": tx3_id,
                    "tick_before": tick1,
                    "tick_middle": tick2,
                    "tick_after": tick3,
                    "tick_change_before": None if (tick1 is None or tick2 is None) else (tick2 - tick1),
                    "tick_change_after": None if (tick2 is None or tick3 is None) else (tick3 - tick2),
                    "price_reversal_flag": int(price_reversal),
                    "gas_before": gas1,
                    "gas_middle": gas2,
                    "gas_after": gas3,
                    "suspicious_gas_pattern_flag": suspicious_gas_pattern_flag,
                    "trade_size_attacker_1": attacker_size_1,
                    "trade_size_victim": victim_size,
                    "trade_size_attacker_2": attacker_size_2,
                    "label_class": "documented_case",
                    "label_source": "known_mev_bot",
                    "label_confidence": "high",
                    "notes": "highly_structured_pattern_consistent_with_sandwich_behavior",
                }
            )

    return pd.DataFrame(rows)


def write_cases_to_parquet(cases_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cases_df.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Parquet support is missing in the environment. "
        ) from exc

    return output_path
