#!/usr/bin/env python3
"""Cross-benchmark LOSO.

Unlike loso_rcaeval.py (single file_pattern), this script accepts multiple
(benchmark_label, in_dir, file_pattern, acc_col) triples and pools them into
one big per-case DataFrame tagged by `system` (= f"{bench}::{subsystem}").

Usage:
  python loso_xbench.py \\
    --source rcaeval:/path/results_rcaeval:rcaeval_{ds}_results_v2.csv:acc1_v2:\\
             online-boutique,sock-shop-2,train-ticket \\
    --source petshop:/path/results_petshop:petshop_{ds}_results.csv:acc@1:\\
             high_traffic,low_traffic,temporal_traffic1,temporal_traffic2 \\
    --out-dir /path/results_xbench

Emits: view_accuracy.csv, pairwise_deltas.csv, rank_flips.csv (same schema
as loso_rcaeval.py).
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

# Reuse the heavy logic
from loso_rcaeval import per_view_accuracy, pairwise_deltas, rank_flip_table


def parse_source(s):
    parts = s.split(":", 4)
    if len(parts) != 5:
        raise ValueError(
            f"--source must be label:dir:pattern:acc_col:ds1,ds2,... got {s!r}")
    label, directory, pattern, acc_col, datasets_csv = parts
    datasets = [d.strip() for d in datasets_csv.split(",") if d.strip()]
    return label, Path(directory), pattern, acc_col, datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="Repeat. Format: label:dir:pattern:acc_col:ds1,ds2")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for src in args.source:
        label, d, pattern, acc_col, datasets = parse_source(src)
        for ds in datasets:
            f = d / pattern.format(ds=ds)
            if not f.exists():
                print(f"[skip] {f} missing")
                continue
            df = pd.read_csv(f)
            if acc_col != "acc1_v2":
                df = df.rename(columns={acc_col: "acc1_v2"})
            df["system"] = f"{label}::{ds}"
            frames.append(df)
    if not frames:
        raise SystemExit("no inputs loaded")
    big = pd.concat(frames, ignore_index=True)
    systems = sorted(big["system"].unique())
    methods = sorted(big["method"].unique())
    print(f"Loaded {len(big)} rows, {len(systems)} systems, methods={methods}")

    view_tbl = per_view_accuracy(big, systems)
    view_tbl.to_csv(args.out_dir / "view_accuracy.csv", index=False)
    deltas = pairwise_deltas(view_tbl)
    deltas.to_csv(args.out_dir / "pairwise_deltas.csv", index=False)
    flips = rank_flip_table(deltas)
    flips.to_csv(args.out_dir / "rank_flips.csv", index=False)

    # Summary
    print("\n=== Per-view acc@1 (method × view, tdelta=0 only) ===")
    v0 = view_tbl[view_tbl["tdelta"] == 0].pivot_table(
        index="view", columns="method", values="acc1"
    ).round(3)
    print(v0.to_string())

    if not deltas.empty:
        print("\n=== Pairwise Δ (tdelta=0) ===")
        d0 = deltas[deltas["tdelta"] == 0].copy()
        d0["pair"] = d0["method_a"] + " - " + d0["method_b"]
        piv = d0.pivot_table(index="view", columns="pair", values="delta").round(3)
        print(piv.to_string())

    print("\n=== Rank flips ===")
    if flips.empty:
        print("  (none)")
    else:
        print(flips.to_string(index=False))


if __name__ == "__main__":
    main()
