from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

@dataclass
class SandwichRuleConfig:
    """Config for sandwich pattern"""
    minimum_attacker_trade_size: float = 0.0
    require_same_sender_before_after: bool = True
    require_reversal_pattern: bool = True
    require_same_block: bool = True
    suspicious_gas_multiplier: float = 1.2

def detect_sandwich_candidates(dataframe: pd.DataFrame, config: SandwichRuleConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect attack by checking 3 swap pappern. Bot before and bot after and order in middle
    All 3 swaps must belong to same pool, same block, executed in consecutive order meaning first and last order is attacker and middle is victim.
    """
    resolved_config = config or SandwichRuleConfig()

    required_columns = [
        "swap_id",
        "pool_address",
        "block_number",
        "timestamp",
        "sender_address",
        "transaction_hash",
        "log_index",
        "tick",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for sandwich labeling: {missing_columns}"
        )

    ordered_dataframe = _prepare_event_dataframe(dataframe)
    sequence_rows: list[dict[str, Any]] = []

    for (pool_address, block_number), block_df in ordered_dataframe.groupby(
        ["pool_address", "block_number"],
        sort=False,
    ):
        block_df = block_df.reset_index(drop=True)

        if len(block_df) < 3:
            continue

        for start_index in range(len(block_df) - 2):
            triple_df = block_df.iloc[start_index:start_index + 3].copy()
            candidate = _evaluate_three_swap_candidate(
                triple_df=triple_df,
                config=resolved_config,
                pool_address=str(pool_address),
                block_number=int(block_number),
            )
            sequence_rows.append(candidate)

    sequence_labels_df = pd.DataFrame(sequence_rows)

    if sequence_labels_df.empty:
        event_labels_df = _build_default_event_labels(ordered_dataframe)
        sequence_labels_df = _build_empty_sequence_labels()
        return event_labels_df, sequence_labels_df

    event_labels_df = _build_event_labels_from_sequences(
        ordered_dataframe=ordered_dataframe,
        sequence_labels_df=sequence_labels_df,
    )

    return event_labels_df, sequence_labels_df

def _prepare_event_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure event ordering for rule evaluation.
    """
    ordered_dataframe = dataframe.copy()

    numeric_columns = [
        "block_number",
        "timestamp",
        "log_index",
        "tick",
        "gas_price_gwei",
        "swap_size_token0",
        "swap_size_token1",
    ]
    for column in numeric_columns:
        if column in ordered_dataframe.columns:
            ordered_dataframe[column] = pd.to_numeric(
                ordered_dataframe[column],
                errors="coerce",
            )

    sort_columns = [
        column for column in [
            "pool_address",
            "block_number",
            "timestamp",
            "log_index",
            "swap_id",
        ]
        if column in ordered_dataframe.columns
    ]

    ordered_dataframe = (
        ordered_dataframe
        .sort_values(sort_columns, ascending=True)
        .reset_index(drop=True)
    )

    return ordered_dataframe


def _evaluate_three_swap_candidate(
    triple_df: pd.DataFrame,
    config: SandwichRuleConfig,
    pool_address: str,
    block_number: int,
) -> dict[str, Any]:
    first_swap = triple_df.iloc[0]
    middle_swap = triple_df.iloc[1]
    third_swap = triple_df.iloc[2]

    attacker_sender = first_swap["sender_address"]
    third_sender = third_swap["sender_address"]
    victim_sender = middle_swap["sender_address"]

    same_sender_before_after_flag = int(
        pd.notna(attacker_sender) and
        pd.notna(third_sender) and
        attacker_sender == third_sender
    )

    different_middle_sender_flag = int(
        pd.notna(victim_sender) and
        victim_sender != attacker_sender
    )

    tick_before = _safe_float(first_swap.get("tick"))
    tick_middle = _safe_float(middle_swap.get("tick"))
    tick_after = _safe_float(third_swap.get("tick"))

    tick_change_before = None
    tick_change_after = None
    reversal_pattern_flag = 0

    if tick_before is not None and tick_middle is not None and tick_after is not None:
        tick_change_before = tick_middle - tick_before
        tick_change_after = tick_after - tick_middle

        upward_then_downward = tick_change_before > 0 and tick_change_after < 0
        downward_then_upward = tick_change_before < 0 and tick_change_after > 0

        reversal_pattern_flag = int(upward_then_downward or downward_then_upward)

    attacker_trade_size = _resolve_trade_size(first_swap)
    victim_trade_size = _resolve_trade_size(middle_swap)
    attacker_exit_trade_size = _resolve_trade_size(third_swap)

    attacker_trade_size_flag = int(
        attacker_trade_size is not None and
        attacker_exit_trade_size is not None and
        attacker_trade_size >= config.minimum_attacker_trade_size and
        attacker_exit_trade_size >= config.minimum_attacker_trade_size
    )

    relative_trade_size = None
    if (
        attacker_trade_size is not None and
        victim_trade_size is not None and
        victim_trade_size > 0
    ):
        relative_trade_size = attacker_trade_size / victim_trade_size

    gas_before = _safe_float(first_swap.get("gas_price_gwei"))
    gas_middle = _safe_float(middle_swap.get("gas_price_gwei"))
    gas_after = _safe_float(third_swap.get("gas_price_gwei"))

    suspicious_gas_pattern_flag = 0
    if gas_before is not None and gas_middle is not None and gas_after is not None:
        suspicious_gas_pattern_flag = int(
            gas_before >= gas_middle * config.suspicious_gas_multiplier or
            gas_after >= gas_middle * config.suspicious_gas_multiplier
        )

    heuristic_score = (
        same_sender_before_after_flag +
        different_middle_sender_flag +
        reversal_pattern_flag +
        attacker_trade_size_flag
    )

    is_weak_anomaly = (
        same_sender_before_after_flag == 1 and
        different_middle_sender_flag == 1 and
        (reversal_pattern_flag == 1 if config.require_reversal_pattern else True)
    )

    label_class = "weak_anomaly" if is_weak_anomaly else _map_candidate_to_label(
        same_sender_before_after_flag=same_sender_before_after_flag,
        different_middle_sender_flag=different_middle_sender_flag,
        reversal_pattern_flag=reversal_pattern_flag,
        heuristic_score=heuristic_score,
    )

    label_confidence = _map_label_confidence(
        label_class=label_class,
        heuristic_score=heuristic_score,
        suspicious_gas_pattern_flag=suspicious_gas_pattern_flag,
    )

    source_event_ids = triple_df["swap_id"].tolist()

    return {
        "sequence_id": _build_sequence_id(source_event_ids),
        "pool_address": pool_address,
        "block_number": block_number,
        "tx1_id": first_swap["swap_id"],
        "tx2_id": middle_swap["swap_id"],
        "tx3_id": third_swap["swap_id"],
        "attacker_address": attacker_sender if same_sender_before_after_flag == 1 else None,
        "victim_address": victim_sender if different_middle_sender_flag == 1 else None,
        "tick_before": tick_before,
        "tick_middle": tick_middle,
        "tick_after": tick_after,
        "tick_change_before": tick_change_before,
        "tick_change_after": tick_change_after,
        "gas_before": gas_before,
        "gas_middle": gas_middle,
        "gas_after": gas_after,
        "same_sender_before_after_flag": same_sender_before_after_flag,
        "different_middle_sender_flag": different_middle_sender_flag,
        "reversal_pattern_flag": reversal_pattern_flag,
        "attacker_trade_size_flag": attacker_trade_size_flag,
        "suspicious_gas_pattern_flag": suspicious_gas_pattern_flag,
        "relative_trade_size": relative_trade_size,
        "candidate_pattern_type": "sandwich_like_three_swap",
        "label_class": label_class,
        "label_source": "rule_based",
        "label_confidence": label_confidence,
        "source_event_ids": source_event_ids,
        "notes": _build_sequence_notes(
            same_sender_before_after_flag=same_sender_before_after_flag,
            different_middle_sender_flag=different_middle_sender_flag,
            reversal_pattern_flag=reversal_pattern_flag,
            suspicious_gas_pattern_flag=suspicious_gas_pattern_flag,
        ),
    }


def _build_event_labels_from_sequences(
    ordered_dataframe: pd.DataFrame,
    sequence_labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert sequence-level sandwich candidates into event-level labels.
    """
    event_label_rows: list[dict[str, Any]] = []

    sequence_membership: dict[str, list[dict[str, Any]]] = {}
    for _, sequence_row in sequence_labels_df.iterrows():
        for swap_id in sequence_row["source_event_ids"]:
            sequence_membership.setdefault(swap_id, []).append(sequence_row.to_dict())

    for _, event_row in ordered_dataframe.iterrows():
        swap_id = event_row["swap_id"]
        related_sequences = sequence_membership.get(swap_id, [])

        if not related_sequences:
            event_label_rows.append(
                {
                    "swap_id": swap_id,
                    "pool_address": event_row["pool_address"],
                    "transaction_hash": event_row.get("transaction_hash"),
                    "block_number": event_row.get("block_number"),
                    "timestamp": event_row.get("timestamp"),
                    "label_class": "normal",
                    "label_source": "rule_based",
                    "label_confidence": "low",
                    "candidate_pattern_type": None,
                    "sequence_id": None,
                    "role_in_sequence": None,
                    "notes": "no_sandwich_rule_triggered",
                }
            )
            continue

        best_sequence = _select_best_sequence(related_sequences)
        role_in_sequence = _resolve_event_role(
            swap_id=swap_id,
            tx1_id=best_sequence["tx1_id"],
            tx2_id=best_sequence["tx2_id"],
            tx3_id=best_sequence["tx3_id"],
        )

        event_label_rows.append(
            {
                "swap_id": swap_id,
                "pool_address": event_row["pool_address"],
                "transaction_hash": event_row.get("transaction_hash"),
                "block_number": event_row.get("block_number"),
                "timestamp": event_row.get("timestamp"),
                "label_class": best_sequence["label_class"],
                "label_source": "inherited_from_sequence",
                "label_confidence": best_sequence["label_confidence"],
                "candidate_pattern_type": best_sequence["candidate_pattern_type"],
                "sequence_id": best_sequence["sequence_id"],
                "role_in_sequence": role_in_sequence,
                "notes": best_sequence["notes"],
            }
        )

    return pd.DataFrame(event_label_rows)


def _build_default_event_labels(ordered_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create all-normal event labels when no candidate sequences are found.
    """
    rows = []
    for _, event_row in ordered_dataframe.iterrows():
        rows.append(
            {
                "swap_id": event_row["swap_id"],
                "pool_address": event_row["pool_address"],
                "transaction_hash": event_row.get("transaction_hash"),
                "block_number": event_row.get("block_number"),
                "timestamp": event_row.get("timestamp"),
                "label_class": "normal",
                "label_source": "rule_based",
                "label_confidence": "low",
                "candidate_pattern_type": None,
                "sequence_id": None,
                "role_in_sequence": None,
                "notes": "no_sandwich_rule_triggered",
            }
        )
    return pd.DataFrame(rows)


def _build_empty_sequence_labels() -> pd.DataFrame:
    """
    Return an empty sequence-label dataframe with stable columns.
    """
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "pool_address",
            "block_number",
            "tx1_id",
            "tx2_id",
            "tx3_id",
            "attacker_address",
            "victim_address",
            "tick_before",
            "tick_middle",
            "tick_after",
            "tick_change_before",
            "tick_change_after",
            "gas_before",
            "gas_middle",
            "gas_after",
            "same_sender_before_after_flag",
            "different_middle_sender_flag",
            "reversal_pattern_flag",
            "attacker_trade_size_flag",
            "suspicious_gas_pattern_flag",
            "relative_trade_size",
            "candidate_pattern_type",
            "label_class",
            "label_source",
            "label_confidence",
            "source_event_ids",
            "notes",
        ]
    )


def _resolve_trade_size(event_row: pd.Series) -> float | None:
    for column in ["swap_size_token0", "abs_amount0", "swap_size_token1", "abs_amount1"]:
        if column in event_row.index:
            value = _safe_float(event_row.get(column))
            if value is not None:
                return abs(value)
    return None


def _map_candidate_to_label(
    same_sender_before_after_flag: int,
    different_middle_sender_flag: int,
    reversal_pattern_flag: int,
    heuristic_score: int,
) -> str:
    """
    Map partial rule matches into suspicious vs normal.
    """
    if same_sender_before_after_flag and different_middle_sender_flag:
        return "suspicious"
    if heuristic_score >= 2 and reversal_pattern_flag:
        return "suspicious"
    return "normal"


def _map_label_confidence(
    label_class: str,
    heuristic_score: int,
    suspicious_gas_pattern_flag: int,
) -> str:
    """
    Set confidence for rule labels.
    """
    if label_class == "weak_anomaly":
        if heuristic_score >= 4 or suspicious_gas_pattern_flag == 1:
            return "high"
        return "medium"

    if label_class == "suspicious":
        return "low"

    return "low"


def _build_sequence_notes(
    same_sender_before_after_flag: int,
    different_middle_sender_flag: int,
    reversal_pattern_flag: int,
    suspicious_gas_pattern_flag: int,
) -> str:
    """
    Build compact rule-explanation notes.
    """
    active_notes: list[str] = []

    if same_sender_before_after_flag == 1:
        active_notes.append("same_sender_before_after")
    if different_middle_sender_flag == 1:
        active_notes.append("different_middle_sender")
    if reversal_pattern_flag == 1:
        active_notes.append("tick_reversal")
    if suspicious_gas_pattern_flag == 1:
        active_notes.append("gas_pattern")

    if not active_notes:
        return "no_sandwich_rule_triggered"

    return ",".join(active_notes)


def _select_best_sequence(related_sequences: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Prefer weak_anomaly over suspicious over normal if an event belongs to multiple windows.
    """
    label_priority = {
        "weak_anomaly": 3,
        "suspicious": 2,
        "normal": 1,
    }

    return sorted(
        related_sequences,
        key=lambda row: label_priority.get(str(row["label_class"]), 0),
        reverse=True,
    )[0]


def _resolve_event_role(
    swap_id: str,
    tx1_id: str,
    tx2_id: str,
    tx3_id: str,
) -> str | None:
    """
    Map event position within the three-swap candidate.
    """
    if swap_id == tx1_id:
        return "attacker_entry"
    if swap_id == tx2_id:
        return "victim"
    if swap_id == tx3_id:
        return "attacker_exit"
    return None


def _build_sequence_id(source_event_ids: list[str]) -> str:
    """
    Build deterministic sequence identifier from the three swap IDs.
    """
    return "|".join(source_event_ids)


def _safe_float(value: Any) -> float | None:
    """
    Convert value to float where possible.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
