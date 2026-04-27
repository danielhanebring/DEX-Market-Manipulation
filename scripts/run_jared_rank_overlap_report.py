from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.paths import PROJECT_ROOT, ensure_directory


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare overlap between top cases ranked by rf_probability vs lr_probability on UNLABELED Jared."
    )
    parser.add_argument(
        "--scored",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "jared_unlabeled_scored.parquet"),
        help="Path to scored Jared table (parquet preferred).",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=0.02,
        help="Top fraction used for overlap comparison (e.g. 0.02 = top 2 percent).",
    )
    parser.add_argument(
        "--out-md",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "jared_rf_vs_lr_overlap.md"),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--out-json",
        default=str(PROJECT_ROOT / "outputs" / "reports" / "jared_rf_vs_lr_overlap.json"),
        help="Output json summary path.",
    )
    return parser.parse_args(argv)


def _read_scored(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _top_set(df: pd.DataFrame, col: str, top_n: int) -> pd.DataFrame:
    tmp = df[["swap_id", "block_number", col]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna(subset=[col]).sort_values(col, ascending=False).head(top_n)
    return tmp


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    scored_path = Path(args.scored)
    if not scored_path.exists():
        raise SystemExit(f"Missing scored file: {scored_path}")

    top_pct = float(args.top_pct)
    if not (0.0 < top_pct < 1.0):
        raise SystemExit("--top-pct must be between 0 and 1.")

    df = _read_scored(scored_path)
    required = ["swap_id", "block_number", "rf_probability", "lr_probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in scored table: {missing}")

    n_total = int(len(df))
    top_n = max(1, int(math.ceil(top_pct * n_total)))

    top_rf = _top_set(df, "rf_probability", top_n)
    top_lr = _top_set(df, "lr_probability", top_n)

    rf_ids = set(top_rf["swap_id"].astype("string").tolist())
    lr_ids = set(top_lr["swap_id"].astype("string").tolist())
    overlap_ids = rf_ids & lr_ids

    rf_blocks = set(pd.to_numeric(top_rf["block_number"], errors="coerce").dropna().astype(int).tolist())
    lr_blocks = set(pd.to_numeric(top_lr["block_number"], errors="coerce").dropna().astype(int).tolist())
    overlap_blocks = rf_blocks & lr_blocks

    jaccard_ids = (len(overlap_ids) / len(rf_ids | lr_ids)) if (rf_ids | lr_ids) else None
    jaccard_blocks = (len(overlap_blocks) / len(rf_blocks | lr_blocks)) if (rf_blocks | lr_blocks) else None

    merged = df[["swap_id", "block_number", "rf_probability", "lr_probability"]].copy()
    merged["rf_probability"] = pd.to_numeric(merged["rf_probability"], errors="coerce")
    merged["lr_probability"] = pd.to_numeric(merged["lr_probability"], errors="coerce")

    top10_rf = merged.sort_values("rf_probability", ascending=False).head(10)
    top10_lr = merged.sort_values("lr_probability", ascending=False).head(10)

    corr = merged[["rf_probability", "lr_probability"]].dropna().corr(method="spearman").iloc[0, 1]
    corr = float(corr) if pd.notna(corr) else None

    payload = {
        "n_total": n_total,
        "top_pct": top_pct,
        "top_n": top_n,
        "event_overlap_count": int(len(overlap_ids)),
        "event_overlap_rate_vs_rf": float(len(overlap_ids) / max(1, len(rf_ids))),
        "event_overlap_rate_vs_lr": float(len(overlap_ids) / max(1, len(lr_ids))),
        "event_jaccard": jaccard_ids,
        "block_overlap_count": int(len(overlap_blocks)),
        "block_overlap_rate_vs_rf": float(len(overlap_blocks) / max(1, len(rf_blocks))),
        "block_overlap_rate_vs_lr": float(len(overlap_blocks) / max(1, len(lr_blocks))),
        "block_jaccard": jaccard_blocks,
        "spearman_corr_all": corr,
    }

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    ensure_directory(out_md.parent)
    ensure_directory(out_json.parent)

    md: list[str] = []
    md.append("# Jared Top-Case Overlap: RandomForest vs LogisticRegression")
    md.append("")
    md.append(f"Scored table: `{scored_path}`")
    md.append(f"- Total Jared events scored: **{n_total}**")
    md.append(f"- Top fraction compared: **{top_pct}** (top_n={top_n})")
    md.append("")
    md.append("## Overlap Summary")
    md.append("")
    md.append("| Item | Value |")
    md.append("|---|---:|")
    md.append(f"| Event overlap count | {payload['event_overlap_count']} |")
    md.append(f"| Event overlap rate vs RF top set | {100.0 * payload['event_overlap_rate_vs_rf']:.1f}% |")
    md.append(f"| Event overlap rate vs LR top set | {100.0 * payload['event_overlap_rate_vs_lr']:.1f}% |")
    md.append(f"| Event Jaccard | {payload['event_jaccard']:.3f} |" if payload["event_jaccard"] is not None else "| Event Jaccard | NA |")
    md.append(f"| Block overlap count | {payload['block_overlap_count']} |")
    md.append(f"| Block overlap rate vs RF top blocks | {100.0 * payload['block_overlap_rate_vs_rf']:.1f}% |")
    md.append(f"| Block overlap rate vs LR top blocks | {100.0 * payload['block_overlap_rate_vs_lr']:.1f}% |")
    md.append(f"| Block Jaccard | {payload['block_jaccard']:.3f} |" if payload["block_jaccard"] is not None else "| Block Jaccard | NA |")
    md.append(f"| Spearman corr (all events) | {payload['spearman_corr_all']:.3f} |" if payload["spearman_corr_all"] is not None else "| Spearman corr (all events) | NA |")
    md.append("")

    md.append("## Top 10 By RF (with LR scores)")
    md.append("")
    md.append("```text")
    md.append(top10_rf.to_string(index=False))
    md.append("```")
    md.append("")
    md.append("## Top 10 By LR (with RF scores)")
    md.append("")
    md.append("```text")
    md.append(top10_lr.to_string(index=False))
    md.append("```")
    md.append("")

    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

