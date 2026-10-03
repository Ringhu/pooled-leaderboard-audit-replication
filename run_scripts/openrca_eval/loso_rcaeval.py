#!/usr/bin/env python3
"""Leave-one-system-out analysis on RCAEval acc@1 (v2-rescored).

Reads per-case `rcaeval_<ds>_results_v2.csv` files (one per subsystem),
pools them, and for each (method_pair, tdelta) reports:
  - acc@1 per subsystem
  - pooled acc@1
  - LOSO rankings: drop each subsystem in turn
  - pairwise Δ (method_a − method_b) per view

Emits a human-readable markdown table plus a machine CSV.
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def load_all(in_dir: Path, datasets, file_pattern="rcaeval_{ds}_results_v2.csv",
             acc_col="acc1_v2"):
    """Load per-system result CSVs, tag each row with `system`, and normalize
    the accuracy column name to `acc1_v2` so downstream aggregation is stable.

    For PetShop inputs (acc_col='acc@1') the column is renamed in place.
    """
    frames = []
    for ds in datasets:
        f = in_dir / file_pattern.format(ds=ds)
        if not f.exists():
            print(f"[skip] {f} missing")
            continue
        df = pd.read_csv(f)
        if acc_col != "acc1_v2":
            if acc_col not in df.columns:
                raise SystemExit(f"acc column {acc_col!r} not in {f}")
            df = df.rename(columns={acc_col: "acc1_v2"})
        df["system"] = ds
        frames.append(df)
    if not frames:
        raise SystemExit("no result CSVs found")
    return pd.concat(frames, ignore_index=True)


def per_view_accuracy(df, systems):
    """Return DataFrame: (view, method, tdelta) -> (n, acc1).

    Views: each individual system, 'pooled' (all), and 'drop-<system>' for each.
    """
    rows = []
    views = {}
    views["pooled"] = df
    for s in systems:
        views[f"only-{s}"] = df[df["system"] == s]
        views[f"drop-{s}"] = df[df["system"] != s]

    for view_name, view_df in views.items():
        g = view_df.groupby(["method", "tdelta"]).agg(
            n=("acc1_v2", "size"),
            acc1=("acc1_v2", "mean"),
        ).reset_index()
        g["view"] = view_name
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def pairwise_deltas(view_table):
    """For each view × tdelta, compute acc1(method_a) - acc1(method_b)
    across all unordered method pairs."""
    out = []
    for (view, td), sub in view_table.groupby(["view", "tdelta"]):
        methods = sorted(sub["method"].unique())
        acc = dict(zip(sub["method"], sub["acc1"]))
        for a, b in combinations(methods, 2):
            out.append({
                "view": view, "tdelta": td,
                "method_a": a, "method_b": b,
                "acc_a": acc[a], "acc_b": acc[b],
                "delta": acc[a] - acc[b],
            })
    return pd.DataFrame(out)


def rank_flip_table(deltas):
    """Detect whether sign(delta) changes across views for a given (pair, tdelta)."""
    flips = []
    if deltas.empty:
        return pd.DataFrame(flips)
    for (a, b, td), sub in deltas.groupby(["method_a", "method_b", "tdelta"]):
        signs = sub.set_index("view")["delta"].apply(
            lambda x: "+" if x > 1e-9 else ("-" if x < -1e-9 else "0")
        )
        uniq = set(signs.values) - {"0"}
        if len(uniq) > 1:
            flips.append({
                "pair": f"{a} vs {b}",
                "tdelta": td,
                **signs.to_dict(),
                "min_delta": sub["delta"].min(),
                "max_delta": sub["delta"].max(),
            })
    return pd.DataFrame(flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--file-pattern", default="rcaeval_{ds}_results_v2.csv",
                    help="Filename template with '{ds}' placeholder. "
                         "Use 'petshop_{ds}_results.csv' for PetShop outputs.")
    ap.add_argument("--acc-col", default="acc1_v2",
                    help="Name of per-case acc@1 column in the CSVs. "
                         "Use 'acc@1' for PetShop outputs.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(args.in_dir, args.datasets,
                  file_pattern=args.file_pattern, acc_col=args.acc_col)
    print(f"Loaded {len(df)} rows across {df['system'].nunique()} systems, "
          f"{df['method'].nunique()} methods, {df['tdelta'].nunique()} tdeltas")

    view_tbl = per_view_accuracy(df, args.datasets)
    view_tbl.to_csv(args.out_dir / "view_accuracy.csv", index=False)
    deltas = pairwise_deltas(view_tbl)
    deltas.to_csv(args.out_dir / "pairwise_deltas.csv", index=False)
    flips = rank_flip_table(deltas)
    flips.to_csv(args.out_dir / "rank_flips.csv", index=False)

    print("\n=== Per-view acc@1 (method × view × tdelta=0) ===")
    v0 = view_tbl[view_tbl["tdelta"] == 0].pivot_table(
        index="view", columns="method", values="acc1"
    ).round(3)
    print(v0.to_string())

    print("\n=== Pairwise Δ (tdelta=0 only, per view) ===")
    if deltas.empty:
        print("  (single method loaded — no pairwise deltas to report)")
    else:
        d0 = deltas[deltas["tdelta"] == 0].copy()
        d0["pair"] = d0["method_a"] + " - " + d0["method_b"]
        print(d0.pivot_table(index="view", columns="pair", values="delta").round(3).to_string())

    print("\n=== Rank flips detected across views ===")
    if flips.empty:
        print("  (none — all views agree on sign)")
    else:
        print(flips.to_string(index=False))


if __name__ == "__main__":
    main()
