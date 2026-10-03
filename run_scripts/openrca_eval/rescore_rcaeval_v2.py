#!/usr/bin/env python3
"""Post-hoc rescore of RCAEval results with a fixed `extract_service`.

Only acc@1 is recoverable: `top1` was stored as the output of the old
extractor applied to the rank-1 metric column. For sock-shop-2 and
train-ticket the old extractor was a no-op, so stored `top1` is actually
the raw rank-1 metric — we can re-apply the v2 extractor and recompute
match-against-true_service. acc@{3,5,10} cannot be recovered without the
full rank list (not stored).

Usage:
  python rescore_rcaeval_v2.py --in-dir ../../results_rcaeval \
      --datasets online-boutique sock-shop-2 train-ticket \
      --out-suffix _v2
"""
import argparse
from pathlib import Path

import pandas as pd

# Import the patched extractor from run_rcaeval so the two stay in sync.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_rcaeval import extract_service  # noqa: E402


def rescore(results_csv, summary_out, per_case_out):
    df = pd.read_csv(results_csv)
    # Keep rows that have a top1 prediction
    df = df[df["top1"].notna() & (df["top1"].astype(str) != "")].copy()
    df["top1_v2"] = df["top1"].astype(str).map(extract_service)
    df["acc1_v2"] = (df["top1_v2"] == df["true_service"]).astype(float)

    df.to_csv(per_case_out, index=False)

    summ = df.groupby(["method", "tdelta"]).agg(
        n=("acc1_v2", "size"),
        acc1=("acc1_v2", "mean"),
    ).reset_index()
    summ.to_csv(summary_out, index=False)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out-suffix", default="_v2")
    args = ap.parse_args()

    for ds in args.datasets:
        res = args.in_dir / f"rcaeval_{ds}_results.csv"
        if not res.exists():
            print(f"[skip] {res} not found")
            continue
        summ_out = args.in_dir / f"rcaeval_{ds}_summary{args.out_suffix}.csv"
        per_out = args.in_dir / f"rcaeval_{ds}_results{args.out_suffix}.csv"
        summ = rescore(res, summ_out, per_out)
        print(f"\n=== {ds} (v2) ===")
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
