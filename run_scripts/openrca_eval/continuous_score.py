#!/usr/bin/env python3
"""Continuous time-credit scoring under a tolerance-distribution prior.

Replaces the hard-threshold time credit `1[d <= θ]` with the survival function
`S_F(d) = 1 - F(d) = P_{θ~F}(θ >= d)` for a distribution F over tolerances.

Three families:
  - point-mass at θ_0   -> `1[d <= θ_0]` (baseline; reproduces OpenRCA)
  - Uniform[0, H]       -> `max(0, 1 - d/H)`        (main paper fix)
  - Exponential(λ)      -> `exp(-λ*d)`              (appendix sensitivity)

Inputs: same prediction CSVs + query CSVs as openrca_evaluate_sweep.py.
Outputs: scores_continuous.csv (per-incident continuous score per method),
         summary_continuous.csv (mean score per method/system/family).

Usage:
  python continuous_score.py --eval-dir full335 \\
      --methods baro cd1min causal zbase rcaagent \\
      --family uniform --H 600 \\
      --out-dir ../../results_full335/continuous_uniform_H600
"""
import argparse
import itertools
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_analysis import (
    SYSTEM_KEYS, SYSTEM_LABELS, METHOD_FILE_PREFIX,
)


# -- Survival functions (time credit) --

def uniform_survival(d, H):
    return np.maximum(0.0, 1.0 - d / H)


def exponential_survival(d, lam):
    return np.exp(-lam * d)


def point_survival(d, theta):
    return (d <= theta).astype(float)


# -- Composite score per incident --

PREDICT_RE = re.compile(
    r'{\s*'
    r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
    r'(?:"root cause component":\s*"(.*?)")?,?\s*'
    r'(?:"root cause reason":\s*"(.*?)")?\s*}'
)
COMP_RE = re.compile(r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)")
REASON_RE = re.compile(r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)")
TIME_RE = re.compile(r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)")


def parse_pred(s):
    m = PREDICT_RE.findall(str(s))
    return [{"dt": dt, "comp": c, "reason": r} for dt, c, r in m]


def parse_gt(s):
    comps = COMP_RE.findall(str(s))
    reasons = REASON_RE.findall(str(s))
    times = TIME_RE.findall(str(s))
    return comps, reasons, times


def time_diff_seconds(t1, t2):
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        return abs((datetime.strptime(t1, fmt) - datetime.strptime(t2, fmt))
                    .total_seconds())
    except ValueError:
        return np.inf


def score_continuous_one(pred_str, scoring_str, survival_fn):
    """Compute composite score = mean over criteria present.

    Criteria: component-match (binary), reason-match (binary), time-credit
    (continuous via survival_fn). Follows OpenRCA's equal-weight convention but
    on the continuous time credit.
    """
    preds = parse_pred(pred_str)
    comps_gt, reasons_gt, times_gt = parse_gt(scoring_str)
    L = max(len(comps_gt), len(reasons_gt), len(times_gt))
    C = len(comps_gt) + len(reasons_gt) + len(times_gt)
    if C == 0 or L != len(preds):
        return 0.0

    # OpenRCA's protocol: try all permutations of predictions against GT slots,
    # pick max score.
    best = -1.0
    for perm in itertools.permutations(preds):
        total = 0.0
        for i in range(L):
            if len(comps_gt) == L:
                total += 1.0 if perm[i]["comp"] == comps_gt[i] else 0.0
            if len(reasons_gt) == L:
                total += 1.0 if perm[i]["reason"] == reasons_gt[i] else 0.0
            if len(times_gt) == L:
                d = time_diff_seconds(times_gt[i], perm[i]["dt"])
                total += float(survival_fn(d))
        best = max(best, total)

    return best / C  # equal-weight mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--family", choices=["uniform", "exponential", "point"],
                    required=True)
    ap.add_argument("--H", type=float, default=600,
                    help="Horizon for uniform[0,H] (seconds)")
    ap.add_argument("--lam", type=float, default=1/300.0,
                    help="Rate for exponential (1/tau, default tau=300s)")
    ap.add_argument("--theta", type=float, default=60.0,
                    help="Threshold for point-mass baseline")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.family == "uniform":
        survival = lambda d: uniform_survival(d, args.H)
        family_label = f"uniform[0,{int(args.H)}s]"
    elif args.family == "exponential":
        survival = lambda d: exponential_survival(d, args.lam)
        family_label = f"exp(lam={args.lam:.4f})"
    else:
        survival = lambda d: point_survival(d, args.theta)
        family_label = f"point({int(args.theta)}s)"

    # Compute per-incident continuous scores
    long_rows = []
    for sys_key in SYSTEM_KEYS:
        q = pd.read_csv(args.eval_dir / f"{sys_key}_q.csv")
        for method in args.methods:
            prefix = METHOD_FILE_PREFIX[method]
            pred_file = args.eval_dir / f"{prefix}_{sys_key}.csv"
            if not pred_file.exists():
                continue
            p = pd.read_csv(pred_file)
            if len(p) != len(q):
                print(f"skip {method}/{sys_key}: length mismatch")
                continue
            for i in range(len(q)):
                sc = score_continuous_one(p.iloc[i].prediction,
                                           q.iloc[i].scoring_points,
                                           survival)
                long_rows.append({
                    "method": method,
                    "system": SYSTEM_LABELS[sys_key],
                    "task_index": q.iloc[i].task_index,
                    "score": sc,
                })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(args.out_dir / "scores_continuous.csv", index=False)

    # Summary
    summary = long_df.groupby(["method", "system"]).agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
    ).reset_index()
    summary["family"] = family_label
    summary.to_csv(args.out_dir / "summary_continuous.csv", index=False)

    # Pooled
    pooled = long_df.groupby("method").agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
    ).reset_index()
    pooled["system"] = "POOLED"
    pooled["family"] = family_label
    pooled.to_csv(args.out_dir / "summary_continuous_pooled.csv", index=False)

    print(f"Family: {family_label}")
    print(summary.to_string(index=False))
    print()
    print("Pooled:")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
