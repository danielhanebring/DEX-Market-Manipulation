from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BOT = "0xae2fc483527b8ef99eb5d9b44875f005ba1fae13"


def _fmt_float(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{float(value):.{ndigits}f}"


def _fmt_int(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return str(int(value))


def build_block_observations(block_context: pd.DataFrame) -> dict[int, dict[str, str]]:
    df = block_context.copy()
    df["origin_is_bot"] = df["origin_address"].astype("string").str.lower().eq(BOT)

    obs: dict[int, dict[str, str]] = {}

    for block_number, g in df.groupby("block_number"):
        b = int(block_number)
        g = g.sort_values(["log_index", "swap_id"]).copy()

        bot = g[g["origin_is_bot"]]
        non = g[~g["origin_is_bot"]]

        if len(bot) < 2:
            obs[b] = {
                "sandwich": "No",
                "reversal": "",
                "gas_spike": "",
                "verdict": "False Positive",
                "note": "<2 bot-origin swaps in exported context",
            }
            continue

        bot1 = bot.iloc[0]
        bot2 = bot.iloc[-1]

        between = g[(g["log_index"] > bot1["log_index"]) & (g["log_index"] < bot2["log_index"])].copy()
        between_non = between[~between["origin_is_bot"]]

        sandwich = "Yes" if len(between_non) > 0 else "No"

        t1 = float(bot1["tick"]) if pd.notna(bot1.get("tick")) else None
        t3 = float(bot2["tick"]) if pd.notna(bot2.get("tick")) else None

        ticks_between = between["tick"].dropna().astype(float)
        tmin = float(ticks_between.min()) if len(ticks_between) else None
        tmax = float(ticks_between.max()) if len(ticks_between) else None

        reversal = ""
        if t1 is not None and t3 is not None and len(ticks_between):
            if (tmax is not None and tmax > max(t1, t3) and t3 < tmax and t1 < tmax) or (
                tmin is not None and tmin < min(t1, t3) and t3 > tmin and t1 > tmin
            ):
                reversal = "Yes"
            else:
                reversal = "No"

        non_gas = non["gas_price_gwei"].dropna().astype(float)
        med_non = float(non_gas.median()) if len(non_gas) else None
        gas2 = float(bot2["gas_price_gwei"]) if pd.notna(bot2.get("gas_price_gwei")) else None
        gas_ratio = (gas2 / med_non) if (gas2 is not None and med_non is not None and med_non > 0) else None

        gas_spike = "Yes" if (gas_ratio is not None and gas_ratio >= 3.0) else "No"

        if sandwich == "Yes" and reversal == "Yes" and gas_spike == "Yes":
            verdict = "True Positive"
        elif sandwich == "Yes" and (reversal == "Yes" or gas_spike == "Yes"):
            verdict = "Partial"
        else:
            verdict = "False Positive"

        note_parts = [
            f"bot1@{_fmt_int(bot1.get('log_index'))} tick={_fmt_int(t1)} gas={_fmt_float(_safe_float(bot1.get('gas_price_gwei')))}",
            f"bot2@{_fmt_int(bot2.get('log_index'))} tick={_fmt_int(t3)} gas={_fmt_float(gas2)}",
            f"between_nonbot={len(between_non)}",
            f"tick_between[min,max]=[{_fmt_int(tmin)},{_fmt_int(tmax)}]",
            f"gas_ratio2~{_fmt_float(gas_ratio, 1)}x",
        ]

        obs[b] = {
            "sandwich": sandwich,
            "reversal": reversal,
            "gas_spike": gas_spike,
            "verdict": verdict,
            "note": "; ".join(note_parts),
        }

    return obs


def _safe_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill a Jared manual review template using the exported block-context parquet.\n"
            "This is a first-pass helper: it adds sandwich/reversal/gas_spike booleans + a preliminary verdict.\n"
            "You can (and should) override after human inspection."
        )
    )
    parser.add_argument(
        "--block-context",
        default=str(Path("outputs") / "reports" / "jared_top_blocks_swaps.parquet"),
        help="Path to block-context parquet exported for the selected top blocks.",
    )
    parser.add_argument(
        "--template",
        default=str(Path("outputs") / "reports" / "jared_manual_review_template.csv"),
        help="Path to the manual review template CSV.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(Path("outputs") / "reports" / "jared_manual_review_filled_by_codex.csv"),
        help="Output CSV path for the filled review sheet.",
    )
    parser.add_argument(
        "--out-json",
        default=str(Path("outputs") / "reports" / "jared_manual_review_aggregate.json"),
        help="Output JSON path for a small verdict summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    block_context_path = Path(args.block_context)
    template_path = Path(args.template)

    if not block_context_path.exists():
        raise SystemExit(f"Missing: {block_context_path}")
    if not template_path.exists():
        raise SystemExit(f"Missing: {template_path}")

    ctx = pd.read_parquet(block_context_path)
    template = pd.read_csv(template_path)

    obs = build_block_observations(ctx)

    def _get(b: int, key: str) -> str:
        return obs.get(int(b), {}).get(key, "")

    filled = template.copy()
    filled["sandwich_pattern_yes_no"] = filled["block_number"].map(lambda b: _get(int(b), "sandwich"))
    filled["price_reversal_yes_no"] = filled["block_number"].map(lambda b: _get(int(b), "reversal"))
    filled["gas_spike_yes_no"] = filled["block_number"].map(lambda b: _get(int(b), "gas_spike"))
    filled["verdict"] = filled["block_number"].map(lambda b: _get(int(b), "verdict"))
    filled["notes"] = filled["block_number"].map(lambda b: _get(int(b), "note"))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(out_csv, index=False)

    vc = filled["verdict"].value_counts(dropna=False)
    aggregate = {
        "rows_total": int(len(filled)),
        "verdict_counts": {str(k): int(v) for k, v in vc.items()},
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    print(vc.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
