#!/usr/bin/env python3
"""
M0 sanity bootstrap for Minute Matters.

Re-analyzes existing pilot predictions with paired stratified bootstrap
to compute pairwise Correct% differences with 95% CI and detect
"robust inversions" across tolerance pairs.

Robust inversion (non-stochastic methods):
  Delta_A-B(t1) and Delta_A-B(t2) have opposite signs AND
  both 95% bootstrap CIs exclude 0.

Usage:
  python bootstrap_analysis.py \
      --methods cd5min cd1min baro causal zbase \
      --tolerances 30 60 120 300 600 \
      --B 1000 --seed 42 \
      --out-dir ../../results_m0
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the scoring function from the existing evaluator
from openrca_evaluate_sweep import evaluate

METHOD_FILE_PREFIX = {
    "cd5min": "preds",           # default CD-detector (5-min)
    "cd1min": "preds_1min",
    "baro":   "preds_baro",
    "causal": "preds_causal",
    "zbase":  "preds_zbaseline",
    "circa":  "preds_circa",
    "rcaagent": "preds_rcaagent",
}
SYSTEM_KEYS = ["bank", "mkt1", "mkt2", "tel"]
SYSTEM_LABELS = {"bank": "Bank", "mkt1": "Mkt-cb1", "mkt2": "Mkt-cb2", "tel": "Telecom"}
DEFAULT_TOLERANCES = [30, 60, 120, 300, 600]


def load_predictions(eval_dir: Path, methods, tolerances):
    """Score each incident at every tolerance for every (method, system).

    Returns
    -------
    scores : dict[(method, system)] -> np.ndarray  shape (n_incidents, n_tol), dtype float
             score in {0.0, partial in (0,1), 1.0}
    task_index : dict[system] -> list of task_index (aligned with rows)
    """
    scores = {}
    task_indices = {}
    for sys_key in SYSTEM_KEYS:
        q = pd.read_csv(eval_dir / f"{sys_key}_q.csv")
        task_indices[sys_key] = q["task_index"].tolist()
        for method in methods:
            prefix = METHOD_FILE_PREFIX[method]
            pred_file = eval_dir / f"{prefix}_{sys_key}.csv"
            p = pd.read_csv(pred_file)
            n = len(q)
            if len(p) != n:
                raise ValueError(
                    f"Row mismatch {method}/{sys_key}: pred={len(p)} q={n}")
            # Row-by-row alignment (OpenRCA convention; task_index may repeat)
            mat = np.zeros((n, len(tolerances)), dtype=float)
            for i in range(n):
                pred = p.iloc[i]["prediction"]
                scoring = q.iloc[i]["scoring_points"]
                if pd.isna(pred):
                    continue
                for ti, tol in enumerate(tolerances):
                    _, _, s = evaluate(pred, scoring, time_tolerance_seconds=tol)
                    mat[i, ti] = s
            scores[(method, sys_key)] = mat
    return scores, task_indices


def bootstrap_pair(scores_A, scores_B, counts_per_system, B=1000, seed=42,
                   metric="strict", ci_pct=(2.5, 97.5)):
    """Stratified paired bootstrap for two aligned score arrays.

    Parameters
    ----------
    scores_A, scores_B : np.ndarray shape (n_tol, n_total_incidents)
    counts_per_system : list[int]
    metric : "strict" uses (score==1) indicator; "partial" uses raw score
    ci_pct : (low, high) percentile pair for CI endpoints
    """
    rng = np.random.default_rng(seed)
    n_tol = scores_A.shape[0]
    assert scores_B.shape == scores_A.shape

    if metric == "strict":
        A_hit = (scores_A == 1.0).astype(float)
        B_hit = (scores_B == 1.0).astype(float)
    elif metric == "partial":
        A_hit = scores_A.astype(float)
        B_hit = scores_B.astype(float)
    else:
        raise ValueError(metric)

    delta_obs = 100.0 * (A_hit.mean(axis=1) - B_hit.mean(axis=1))

    offsets = np.concatenate(([0], np.cumsum(counts_per_system)))
    boot_deltas = np.zeros((B, n_tol), dtype=float)
    for b in range(B):
        idx_parts = []
        for sys_i, cnt in enumerate(counts_per_system):
            lo = offsets[sys_i]
            idx_parts.append(rng.integers(lo, lo + cnt, size=cnt))
        idx = np.concatenate(idx_parts)
        boot_deltas[b] = 100.0 * (A_hit[:, idx].mean(axis=1) -
                                   B_hit[:, idx].mean(axis=1))

    ci_low = np.percentile(boot_deltas, ci_pct[0], axis=0)
    ci_high = np.percentile(boot_deltas, ci_pct[1], axis=0)

    p_value = np.zeros(n_tol)
    for ti in range(n_tol):
        obs = delta_obs[ti]
        if obs >= 0:
            p = 2 * (boot_deltas[:, ti] <= 0).mean()
        else:
            p = 2 * (boot_deltas[:, ti] >= 0).mean()
        p_value[ti] = min(1.0, p)

    return delta_obs, ci_low, ci_high, p_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path,
                    default=Path(__file__).parent)
    ap.add_argument("--methods", nargs="+",
                    default=["cd5min", "cd1min", "baro", "causal", "zbase"])
    ap.add_argument("--tolerances", nargs="+", type=int,
                    default=DEFAULT_TOLERANCES)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pooled", action="store_true",
                    help="Also run pooled (cross-system) bootstrap")
    ap.add_argument("--metric", choices=["strict", "partial"], default="strict",
                    help="Score indicator: strict (score==1) or partial (raw)")
    ap.add_argument("--ci", type=float, default=95.0,
                    help="CI confidence level (e.g., 95 or 90)")
    args = ap.parse_args()
    ci_tails = ((100 - args.ci) / 2, 100 - (100 - args.ci) / 2)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions for methods={args.methods}, "
          f"tolerances={args.tolerances}")
    scores, task_indices = load_predictions(args.eval_dir, args.methods,
                                            args.tolerances)

    # Flatten per-incident scores to a long DataFrame
    rows = []
    for (m, s), mat in scores.items():
        tidx = task_indices[s]
        for i, tk in enumerate(tidx):
            for ti, tol in enumerate(args.tolerances):
                rows.append({
                    "method": m, "system": SYSTEM_LABELS[s],
                    "system_key": s, "task_index": tk,
                    "tolerance_s": tol, "score": mat[i, ti],
                })
    long_df = pd.DataFrame(rows)
    long_df.to_csv(args.out_dir / "scores_per_incident.csv", index=False)

    # Per-system Correct% summary
    summary_rows = []
    for m in args.methods:
        for s in SYSTEM_KEYS:
            mat = scores[(m, s)]
            for ti, tol in enumerate(args.tolerances):
                correct = (mat[:, ti] == 1.0).sum()
                total = mat.shape[0]
                summary_rows.append({
                    "method": m, "system": SYSTEM_LABELS[s],
                    "tolerance_s": tol,
                    "correct": int(correct), "total": total,
                    "correct_pct": 100 * correct / total if total else 0.0,
                })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / "summary_per_system.csv", index=False)

    # Pooled Correct% (all systems)
    pooled_rows = []
    counts_per_system = [scores[(args.methods[0], s)].shape[0]
                          for s in SYSTEM_KEYS]
    for m in args.methods:
        stacked = np.concatenate([scores[(m, s)] for s in SYSTEM_KEYS], axis=0)
        for ti, tol in enumerate(args.tolerances):
            correct = (stacked[:, ti] == 1.0).sum()
            total = stacked.shape[0]
            pooled_rows.append({
                "method": m, "system": "POOLED",
                "tolerance_s": tol,
                "correct": int(correct), "total": total,
                "correct_pct": 100 * correct / total,
            })
    pd.DataFrame(pooled_rows).to_csv(
        args.out_dir / "summary_pooled.csv", index=False)

    # Build (n_tol, n_total) matrices per method for bootstrap
    stacked = {}
    for m in args.methods:
        blocks = [scores[(m, s)] for s in SYSTEM_KEYS]
        stacked[m] = np.concatenate(blocks, axis=0).T  # (n_tol, n_total)

    # Pairwise bootstrap
    pair_rows = []
    for A, B in itertools.combinations(args.methods, 2):
        d_obs, lo, hi, p = bootstrap_pair(
            stacked[A], stacked[B], counts_per_system, B=args.B, seed=args.seed,
            metric=args.metric, ci_pct=ci_tails,
        )
        for ti, tol in enumerate(args.tolerances):
            pair_rows.append({
                "method_A": A, "method_B": B,
                "tolerance_s": tol,
                "delta": d_obs[ti],
                "ci_low": lo[ti], "ci_high": hi[ti],
                "sig": bool((lo[ti] > 0) or (hi[ti] < 0)),
                "p_boot": p[ti],
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(args.out_dir / "pairwise_deltas.csv", index=False)

    # Inversion scan: robust (both sig + opposite signs), weak (one sig), and
    # raw (opposite point-estimate signs regardless of significance)
    robust_rows, weak_rows, raw_rows = [], [], []
    for (A, B), sub in pair_df.groupby(["method_A", "method_B"], sort=False):
        sub = sub.reset_index(drop=True)
        for i, j in itertools.combinations(range(len(sub)), 2):
            r1, r2 = sub.iloc[i], sub.iloc[j]
            opposite = (r1.delta > 0 and r2.delta < 0) or \
                       (r1.delta < 0 and r2.delta > 0)
            if not opposite:
                continue
            row = {
                "method_A": A, "method_B": B,
                "t1_s": int(r1.tolerance_s),
                "t2_s": int(r2.tolerance_s),
                "delta_t1": r1.delta, "ci_low_t1": r1.ci_low,
                "ci_high_t1": r1.ci_high, "sig_t1": bool(r1.sig),
                "delta_t2": r2.delta, "ci_low_t2": r2.ci_low,
                "ci_high_t2": r2.ci_high, "sig_t2": bool(r2.sig),
                "flip_magnitude": abs(r1.delta) + abs(r2.delta),
            }
            raw_rows.append(row)
            if r1.sig and r2.sig:
                robust_rows.append(row)
            elif r1.sig or r2.sig:
                weak_rows.append(row)
    pd.DataFrame(robust_rows).to_csv(args.out_dir / "robust_inversions.csv",
                                      index=False)
    pd.DataFrame(weak_rows).to_csv(args.out_dir / "weak_inversions.csv",
                                    index=False)
    pd.DataFrame(raw_rows).to_csv(args.out_dir / "raw_inversions.csv",
                                    index=False)
    inv_df = pd.DataFrame(robust_rows)

    # Per-system pairwise (useful for per-system Tables)
    per_sys_rows = []
    for sys_key in SYSTEM_KEYS:
        n_sys = scores[(args.methods[0], sys_key)].shape[0]
        for A, B in itertools.combinations(args.methods, 2):
            a_mat = scores[(A, sys_key)].T  # (n_tol, n_sys)
            b_mat = scores[(B, sys_key)].T
            d_obs, lo, hi, p = bootstrap_pair(
                a_mat, b_mat, [n_sys], B=args.B, seed=args.seed,
                metric=args.metric, ci_pct=ci_tails,
            )
            for ti, tol in enumerate(args.tolerances):
                per_sys_rows.append({
                    "method_A": A, "method_B": B,
                    "system": SYSTEM_LABELS[sys_key],
                    "tolerance_s": tol,
                    "delta": d_obs[ti],
                    "ci_low": lo[ti], "ci_high": hi[ti],
                    "sig": bool((lo[ti] > 0) or (hi[ti] < 0)),
                    "p_boot": p[ti],
                })
    pd.DataFrame(per_sys_rows).to_csv(
        args.out_dir / "pairwise_deltas_per_system.csv", index=False)

    meta = {
        "methods": args.methods,
        "tolerances_s": args.tolerances,
        "metric": args.metric,
        "ci": args.ci,
        "B": args.B,
        "seed": args.seed,
        "counts_per_system": dict(zip(SYSTEM_KEYS, counts_per_system)),
        "n_total": int(sum(counts_per_system)),
        "n_pairs": len(list(itertools.combinations(args.methods, 2))),
        "n_significant_deltas_pooled": int(pair_df["sig"].sum()),
        "n_robust_inversions_pooled": int(len(robust_rows)),
        "n_weak_inversions_pooled": int(len(weak_rows)),
        "n_raw_inversions_pooled": int(len(raw_rows)),
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== M0 sanity bootstrap summary ===")
    print(json.dumps(meta, indent=2))

    if len(inv_df):
        print("\nTop 5 robust inversions by flip magnitude:")
        print(inv_df.sort_values("flip_magnitude", ascending=False)
              .head(5).to_string(index=False))
    else:
        print("\nNo robust inversions survive stratified bootstrap.")


if __name__ == "__main__":
    main()
