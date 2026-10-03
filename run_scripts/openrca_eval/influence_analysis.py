#!/usr/bin/env python3
"""Influence analysis: does a robust inversion between methods A and B
survive when we remove influential cases or entire systems?

Input: the per-incident score matrix from bootstrap_analysis.py
       (scores_per_incident.csv), plus a list of (A, B) pairs and tolerances.

For each (A, B) pair:
  - Compute full-data Δ(t) at each tolerance; check robust_inversion (opposite signs).
  - Leave-one-system-out: drop each system in turn, recompute Δ(t), check survival.
  - Leave-top-k-influential: rank incidents by |contribution to mean Δ|,
    drop top-k most influential (k=1,2,5), recompute, check survival.
  - Subsampling: draw 80% of incidents (stratified by system) S=200 times, report
    fraction in which robust_inversion persists.

Output:
  influence_robustness.csv: (A, B, check, delta_t1, delta_t2, survived_inv)
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_ORDER = ["Bank", "Mkt-cb1", "Mkt-cb2", "Telecom"]


def score_vec(df, method, system, tol, metric="partial"):
    """Return (n_incidents,) score vector for (method, system, tol)."""
    sub = df[(df.method == method) & (df.system == system)
             & (df.tolerance_s == tol)]
    sub = sub.sort_values("task_index")  # deterministic order
    s = sub.score.values
    if metric == "strict":
        return (s == 1.0).astype(float)
    return s.astype(float)


def get_pair_deltas(df, A, B, tolerances, incident_mask=None, metric="partial"):
    """Per-system incident-level scores for A and B, with optional mask.

    Returns delta(t) aggregated by pooling all incidents (system-stratified
    average, equivalent to unweighted pooled mean since we do not re-weight).
    """
    deltas = []
    for t in tolerances:
        a_all, b_all = [], []
        for sys in SYSTEM_ORDER:
            a = score_vec(df, A, sys, t, metric)
            b = score_vec(df, B, sys, t, metric)
            if incident_mask is not None:
                m = incident_mask.get(sys)
                if m is not None:
                    a = a[m]; b = b[m]
            a_all.extend(a); b_all.extend(b)
        a_all, b_all = np.array(a_all), np.array(b_all)
        deltas.append(100.0 * (a_all.mean() - b_all.mean()))
    return np.array(deltas)


def has_inversion(deltas):
    """Does deltas(t) change sign across the tolerance grid?"""
    above = (deltas > 0).any()
    below = (deltas < 0).any()
    return above and below


def system_sizes(df, method, tolerance):
    """Return dict {system: n_incidents} from score_per_incident df."""
    sub = df[(df.method == method) & (df.tolerance_s == tolerance)]
    return {sys: int((sub.system == sys).sum()) for sys in SYSTEM_ORDER}


def leave_top_k_mask(df, A, B, tolerances, k, metric="partial"):
    """Identify top-k most influential incidents by per-incident |(sA - sB) at
    the two extreme tolerances|. Returns mask dict excluding them.
    """
    # Rank incidents by (sA - sB) contribution summed at min/max tol
    t_lo, t_hi = min(tolerances), max(tolerances)
    rows = []
    for sys in SYSTEM_ORDER:
        tasks = df[(df.method == A) & (df.system == sys)
                    & (df.tolerance_s == t_lo)].sort_values("task_index")
        for i, task in enumerate(tasks.task_index.tolist()):
            sA_lo = score_vec(df, A, sys, t_lo, metric)[i]
            sA_hi = score_vec(df, A, sys, t_hi, metric)[i]
            sB_lo = score_vec(df, B, sys, t_lo, metric)[i]
            sB_hi = score_vec(df, B, sys, t_hi, metric)[i]
            contrib = abs(sA_lo - sB_lo) + abs(sA_hi - sB_hi)
            rows.append((contrib, sys, i, task))
    rows.sort(reverse=True)
    drop = [(sys, i) for _, sys, i, _ in rows[:k]]

    masks = {sys: np.ones(0, dtype=bool) for sys in SYSTEM_ORDER}
    for sys in SYSTEM_ORDER:
        sz = len(score_vec(df, A, sys, tolerances[0], metric))
        mask = np.ones(sz, dtype=bool)
        for (dsys, di) in drop:
            if dsys == sys:
                mask[di] = False
        masks[sys] = mask
    dropped = [(df[(df.method == A) & (df.system == s) & (df.tolerance_s == t_lo)]
                .sort_values("task_index").iloc[i].task_index, s)
               for s, i in drop]
    return masks, dropped


def loso_mask(sys_out):
    return {sys: (None if sys != sys_out else np.zeros(0, dtype=bool))
            for sys in SYSTEM_ORDER}


def subsample_mask(df, A, tolerances, rng, frac=0.8):
    masks = {}
    for sys in SYSTEM_ORDER:
        sz = len(score_vec(df, A, sys, tolerances[0]))
        k = int(sz * frac)
        idx = rng.choice(sz, size=k, replace=False)
        mask = np.zeros(sz, dtype=bool)
        mask[idx] = True
        masks[sys] = mask
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", required=True,
                    help="scores_per_incident.csv from bootstrap_analysis.py")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="Method pairs, formatted 'A:B'")
    ap.add_argument("--tolerances", nargs="+", type=int, required=True)
    ap.add_argument("--metric", choices=["strict", "partial"], default="partial")
    ap.add_argument("--S-subsample", type=int, default=200,
                    help="Subsampling replicates")
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    rng = np.random.default_rng(args.seed)

    rows = []
    for pair in args.pairs:
        A, B = pair.split(":")
        # Full
        d_full = get_pair_deltas(df, A, B, args.tolerances, None, args.metric)
        full_inv = has_inversion(d_full)
        rows.append({"pair": pair, "check": "full", "detail": "",
                     "min_delta": d_full.min(), "max_delta": d_full.max(),
                     "inverted": full_inv,
                     "stable_sign": "pos" if d_full.min() > 0 else ("neg" if d_full.max() < 0 else "mixed")})
        if not full_inv:
            print(f"  {pair}: no inversion in full data — running stability checks "
                  f"anyway to verify monotone ordering.")

        # LOSO
        for sys in SYSTEM_ORDER:
            m = loso_mask(sys)
            d = get_pair_deltas(df, A, B, args.tolerances, m, args.metric)
            sign = "pos" if d.min() > 0 else ("neg" if d.max() < 0 else "mixed")
            rows.append({"pair": pair, "check": f"loso:{sys}",
                         "detail": f"dropped {sys}",
                         "min_delta": d.min(), "max_delta": d.max(),
                         "inverted": has_inversion(d),
                         "stable_sign": sign})

        # Leave-top-k
        for k in (1, 2, 5):
            m, dropped = leave_top_k_mask(df, A, B, args.tolerances, k, args.metric)
            d = get_pair_deltas(df, A, B, args.tolerances, m, args.metric)
            sign = "pos" if d.min() > 0 else ("neg" if d.max() < 0 else "mixed")
            rows.append({"pair": pair, "check": f"leave_top_{k}",
                         "detail": ";".join(f"{t}@{s}" for t, s in dropped),
                         "min_delta": d.min(), "max_delta": d.max(),
                         "inverted": has_inversion(d),
                         "stable_sign": sign})

        # 80% subsampling
        n_inv = 0
        sign_counts = {"pos": 0, "neg": 0, "mixed": 0}
        for s in range(args.S_subsample):
            m = subsample_mask(df, A, args.tolerances, rng, frac=args.frac)
            d = get_pair_deltas(df, A, B, args.tolerances, m, args.metric)
            if has_inversion(d):
                n_inv += 1
            if d.min() > 0:
                sign_counts["pos"] += 1
            elif d.max() < 0:
                sign_counts["neg"] += 1
            else:
                sign_counts["mixed"] += 1
        rows.append({"pair": pair, "check": f"subsample_{int(args.frac*100)}pct",
                     "detail": f"S={args.S_subsample} | sign_counts={sign_counts}",
                     "min_delta": np.nan, "max_delta": np.nan,
                     "inverted": f"{n_inv}/{args.S_subsample}",
                     "stable_sign": max(sign_counts, key=sign_counts.get)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
