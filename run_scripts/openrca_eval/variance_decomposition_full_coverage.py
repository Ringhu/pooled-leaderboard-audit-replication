#!/usr/bin/env python3
"""Descriptive family vs subsystem variance decomposition.

For each full-coverage method pair, decompose the unweighted variance of
per-system paired deltas into:

  - between-family sum of squares
  - within-family / between-subsystem sum of squares

This is a descriptive diagnostic, not a formal hierarchical model: there are
only three benchmark families.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_REPO_ROOT = Path("$HOME/mnt/3090/auditstack")
DEFAULT_INPUT = DEFAULT_REPO_ROOT / "results_stats" / "multiverse_full_coverage" / "per_system_deltas.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "results_stats" / "variance_decomposition_full_coverage"


def decompose_pair(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    vals = df["delta_pp"].to_numpy(dtype=float)
    grand = float(vals.mean())
    ss_total = float(((vals - grand) ** 2).sum())
    family_rows = []
    ss_between = 0.0
    for fam, sub in df.groupby("benchmark"):
        mean = float(sub["delta_pp"].mean())
        n = int(len(sub))
        ss = float(n * (mean - grand) ** 2)
        ss_between += ss
        family_rows.append({
            "pair": df["pair"].iloc[0],
            "benchmark": fam,
            "n_systems": n,
            "family_mean_delta_pp": mean,
            "family_min_delta_pp": float(sub["delta_pp"].min()),
            "family_max_delta_pp": float(sub["delta_pp"].max()),
            "family_range_pp": float(sub["delta_pp"].max() - sub["delta_pp"].min()),
            "family_has_sign_change": bool((sub["delta_pp"] > 0).any() and (sub["delta_pp"] < 0).any()),
            "ss_between_contribution": ss,
        })
    ss_within = ss_total - ss_between
    summary = {
        "pair": df["pair"].iloc[0],
        "k": int(len(df)),
        "grand_mean_delta_pp": grand,
        "total_ss": ss_total,
        "between_family_ss": ss_between,
        "within_family_ss": ss_within,
        "between_family_share": ss_between / ss_total if ss_total > 0 else np.nan,
        "within_family_share": ss_within / ss_total if ss_total > 0 else np.nan,
        "n_families_with_internal_sign_change": int(sum(r["family_has_sign_change"] for r in family_rows)),
        "max_family_range_pp": float(max(r["family_range_pp"] for r in family_rows)),
    }
    return summary, pd.DataFrame(family_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    summaries = []
    family_tables = []
    for _, sub in df.groupby("pair", sort=False):
        summary, family = decompose_pair(sub)
        summaries.append(summary)
        family_tables.append(family)
    summary_df = pd.DataFrame(summaries)
    family_df = pd.concat(family_tables, ignore_index=True)

    summary_df.to_csv(args.output_dir / "variance_decomposition.csv", index=False)
    family_df.to_csv(args.output_dir / "family_delta_summary.csv", index=False)
    print(f"[written] {args.output_dir}")
    print(summary_df.to_string(index=False))
    print("\n[family summaries]")
    print(family_df.to_string(index=False))


if __name__ == "__main__":
    main()

