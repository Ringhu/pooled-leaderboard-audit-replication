#!/usr/bin/env python3
"""Convert OpenRCA's per-incident scores (with tolerance sweep) into per-case
CSVs matching the RCAEval/PetShop schema used by loso_xbench.py.

OpenRCA's scores_per_incident.csv has columns:
  method, system, system_key, task_index, tolerance_s, score

For cross-benchmark pooling with acc@1-style metrics, we pick a single
tolerance (default 60s) and treat `score` as the binary/fractional per-case
'acc@1' column. One output CSV per system.

Usage:
  python openrca_to_xbench.py \\
    --in-csv $HOME/auditstack/results_full335/clean_partial/scores_per_incident.csv \\
    --tolerance 60 \\
    --out-dir $HOME/auditstack/results_openrca_xbench
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=Path, required=True)
    ap.add_argument("--tolerance", type=int, default=60,
                    help="Which tolerance_s value to project to acc@1 (default 60)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    df = df[df["tolerance_s"] == args.tolerance].copy()
    if df.empty:
        raise SystemExit(f"no rows at tolerance={args.tolerance}")

    print(f"Loaded {len(df)} rows at tolerance={args.tolerance}s")
    print(f"  methods: {sorted(df['method'].unique())}")
    print(f"  systems: {sorted(df['system'].unique())}")

    # Emit one CSV per system, in the RCAEval result_v2 schema loso_xbench
    # understands (with acc1_v2 as the canonical acc@1 column).
    for system, sub in df.groupby("system_key"):
        out_rows = []
        for _, r in sub.iterrows():
            out_rows.append({
                "method": r["method"],
                "tdelta": 0,  # OpenRCA doesn't sweep tdelta in this csv
                "case": f"{system}/{r['task_index']}",
                "true_service": "",  # not needed for LOSO aggregate metric
                "fault": "",
                "top1": "",
                "acc1_v2": float(r["score"]),
                "acc@3": None, "acc@5": None, "acc@10": None,
                "n_ranks": None,
            })
        out = pd.DataFrame(out_rows)
        out_file = args.out_dir / f"openrca_{system}_results_v2.csv"
        out.to_csv(out_file, index=False)
        print(f"  wrote {out_file} ({len(out)} rows, "
              f"methods={sorted(out['method'].unique())})")


if __name__ == "__main__":
    main()
