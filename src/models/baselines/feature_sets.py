from __future__ import annotations

"""
Feature-set helpers for event-level models.

Some features are based on the rule labels. For "no_leakage" runs we drop those features.
"""


RULE_SIGNATURE_FEATURES: list[str] = [
    "same_block_pattern_flag",
    "strict_sandwich_support_flag",
    "three_event_pattern_indicator",
    "same_origin_before_after_flag",
]


NO_LEAKAGE_FEATURES_TO_DROP: list[str] = [
    # Rule-like features
    "same_block_pattern_flag",
    "same_origin_before_after_flag",
    "three_event_pattern_indicator",
    "strict_sandwich_support_flag",
    "sandwich_support_score",
    # Neighbor/triple flags
    "same_sender_before_after_flag",
    "different_middle_sender_from_neighbors_flag",
    "reversal_pattern_flag",
    "tick_change_before",
    "tick_change_after",
    "combined_reversal_magnitude",
    # Size ratios from neighbors
    "relative_trade_size_token0",
    "attacker_vs_victim_size_ratio_token0",
    "relative_trade_size_token1",
    "attacker_vs_victim_size_ratio_token1",
    # Helper flags
    "high_block_gas_context_flag",
    "high_relative_trade_size_flag",
]


def drop_features(feature_columns: list[str], drop: list[str]) -> list[str]:
    drop_set = set(drop)
    return [c for c in feature_columns if c not in drop_set]
