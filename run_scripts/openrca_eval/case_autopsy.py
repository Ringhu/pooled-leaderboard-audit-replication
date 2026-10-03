#!/usr/bin/env python3
"""Case-level autopsy: find incidents where tolerance flips the verdict between
two specific methods, and dump their raw predictions + ground-truth for paper
storytelling.

For a pair (A, B) and tolerances (t1, t2) with opposite-direction dominance:
  - Find incidents where A correct at t1 but wrong at t2, AND B wrong at t1 but
    correct at t2 (or the symmetric case).
  - Show: task_index, system, instruction, ground truth, A's pred, B's pred,
    time diff (sec) for each method, verdict at each tolerance.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openrca_evaluate_sweep import evaluate


METHOD_FILE_PREFIX = {
    "cd5min": "preds",
    "cd1min": "preds_1min",
    "baro":   "preds_baro",
    "causal": "preds_causal",
    "zbase":  "preds_zbaseline",
    "circa":  "preds_circa",
    "rcaagent": "preds_rcaagent",
}
SYSTEM_KEYS = ["bank", "mkt1", "mkt2", "tel"]
SYSTEM_LABELS = {"bank": "Bank", "mkt1": "Mkt-cb1", "mkt2": "Mkt-cb2", "tel": "Telecom"}


def score_one(pred_str, scoring_str, tol):
    p, f, s = evaluate(pred_str, scoring_str, time_tolerance_seconds=tol)
    return s


def extract_first_pred_time(pred_str):
    """Extract the first 'root cause occurrence datetime' from a prediction."""
    import re
    m = re.search(r'"root cause occurrence datetime":\s*"(.*?)"', str(pred_str))
    return m.group(1) if m else None


def extract_gt_time(scoring_str):
    """Extract GT datetime from scoring_points."""
    import re
    m = re.search(r"<=1min\) of (\S+ \S+)", str(scoring_str))
    return m.group(1) if m else None


def time_diff_seconds(pred_time_str, gt_time_str):
    from datetime import datetime
    try:
        t1 = datetime.strptime(pred_time_str, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(gt_time_str, "%Y-%m-%d %H:%M:%S")
        return abs((t1 - t2).total_seconds())
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True,
                    help="Directory with {sys}_q.csv and preds_*.csv")
    ap.add_argument("--method-a", required=True)
    ap.add_argument("--method-b", required=True)
    ap.add_argument("--t1", type=int, required=True)
    ap.add_argument("--t2", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5,
                    help="How many example cases to keep per system")
    args = ap.parse_args()

    prefix_a = METHOD_FILE_PREFIX[args.method_a]
    prefix_b = METHOD_FILE_PREFIX[args.method_b]

    rows = []
    for sys_key in SYSTEM_KEYS:
        q_file = Path(args.eval_dir) / f"{sys_key}_q.csv"
        a_file = Path(args.eval_dir) / f"{prefix_a}_{sys_key}.csv"
        b_file = Path(args.eval_dir) / f"{prefix_b}_{sys_key}.csv"
        if not a_file.exists() or not b_file.exists():
            continue

        q = pd.read_csv(q_file)
        a = pd.read_csv(a_file)
        b = pd.read_csv(b_file)

        n = min(len(q), len(a), len(b))
        for i in range(n):
            gt = q.iloc[i].scoring_points
            pa, pb = a.iloc[i].prediction, b.iloc[i].prediction
            if pd.isna(pa) or pd.isna(pb):
                continue

            sa_t1 = score_one(pa, gt, args.t1)
            sa_t2 = score_one(pa, gt, args.t2)
            sb_t1 = score_one(pb, gt, args.t1)
            sb_t2 = score_one(pb, gt, args.t2)

            # Binary verdict: score == 1.0 = strict correct
            ca_t1, ca_t2 = sa_t1 == 1.0, sa_t2 == 1.0
            cb_t1, cb_t2 = sb_t1 == 1.0, sb_t2 == 1.0

            # Direction of dominance at each tolerance
            # We want: A beats B at t1, B beats A at t2 (one variant of flip)
            # Accept partial wins too (higher score wins)
            delta_t1 = sa_t1 - sb_t1
            delta_t2 = sa_t2 - sb_t2
            if delta_t1 <= 0 or delta_t2 >= 0:
                continue  # not a flip (A doesn't beat B at t1, or B doesn't at t2)

            pa_time = extract_first_pred_time(pa)
            pb_time = extract_first_pred_time(pb)
            gt_time = extract_gt_time(gt)
            diff_a = time_diff_seconds(pa_time, gt_time)
            diff_b = time_diff_seconds(pb_time, gt_time)

            rows.append({
                "system": SYSTEM_LABELS[sys_key],
                "task_index": q.iloc[i].task_index,
                "instruction": q.iloc[i].instruction[:200] + "...",
                "gt_time": gt_time,
                f"{args.method_a}_pred_time": pa_time,
                f"{args.method_a}_time_diff_s": diff_a,
                f"{args.method_a}_score_at_{args.t1}s": sa_t1,
                f"{args.method_a}_score_at_{args.t2}s": sa_t2,
                f"{args.method_b}_pred_time": pb_time,
                f"{args.method_b}_time_diff_s": diff_b,
                f"{args.method_b}_score_at_{args.t1}s": sb_t1,
                f"{args.method_b}_score_at_{args.t2}s": sb_t2,
                "flip_magnitude": (delta_t1 - delta_t2),
            })

    if not rows:
        print("No incidents match the flip condition.")
        return

    df = pd.DataFrame(rows).sort_values("flip_magnitude", ascending=False)
    # Keep top-K per system for diversity
    kept = pd.concat([g.head(args.top_k) for _, g in df.groupby("system")],
                     ignore_index=True)
    kept.to_csv(args.out, index=False)

    print(f"Found {len(df)} flipping incidents (A={args.method_a} wins at "
          f"{args.t1}s, B={args.method_b} wins at {args.t2}s):")
    print(df.groupby("system").size().to_string())
    print(f"\nTop {args.top_k}/system saved to {args.out}")
    print("\n=== Sample rows ===")
    cols = ["system", "task_index", "gt_time",
            f"{args.method_a}_pred_time", f"{args.method_a}_time_diff_s",
            f"{args.method_b}_pred_time", f"{args.method_b}_time_diff_s",
            f"{args.method_a}_score_at_{args.t1}s",
            f"{args.method_a}_score_at_{args.t2}s",
            f"{args.method_b}_score_at_{args.t1}s",
            f"{args.method_b}_score_at_{args.t2}s"]
    print(kept[cols].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
