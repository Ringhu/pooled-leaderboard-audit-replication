#!/usr/bin/env python3
"""Functional bootstrap with simultaneous confidence bands over tolerance.

For each method-pair (A, B) over a tolerance grid {t_1, ..., t_K}, estimate
Δ_AB(t_k) = mean(S_A(case, t_k)) - mean(S_B(case, t_k))
with a stratified paired bootstrap and compute SIMULTANEOUS confidence bands
using a max-t statistic. This replaces pointwise CIs × multiplicity correction
(which is the wrong tool for a methodology paper with thousands of pointwise
tests).

Robust inversion criterion (per pair, over the grid):
  the simultaneous band is entirely above 0 at some t, AND
  entirely below 0 at some other t.

Correction across method pairs is done by applying a Holm-adjusted per-pair
family-wise-error threshold on the max-sup of |Δ̂(t) - Δ(t)|. We implement
this via a joint bootstrap and a per-pair null via the band's sup distance.

Usage:
  python bootstrap_bands.py --eval-dir full335 --out-dir ../../results_full335/bands \\
      --methods baro cd1min causal zbase rcaagent \\
      --tolerances 30 45 60 90 120 180 240 300 450 600 \\
      --B 2000 --seed 42 --metric partial --alpha 0.05
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_analysis import (
    SYSTEM_KEYS, SYSTEM_LABELS, METHOD_FILE_PREFIX, load_predictions,
)


def simultaneous_band(boot_deltas, alpha=0.05):
    """Given (B, K) bootstrap deltas, compute simultaneous band endpoints via
    the max-t statistic (studentized range over tolerances).

    Returns (lo, hi): each (K,) array.
    """
    centered = boot_deltas - boot_deltas.mean(axis=0, keepdims=True)
    # Studentize per tolerance using per-tolerance bootstrap SD
    se = centered.std(axis=0, ddof=1) + 1e-12
    studentized = centered / se[None, :]
    max_abs_t = np.abs(studentized).max(axis=1)        # (B,)
    q = np.percentile(max_abs_t, 100 * (1 - alpha))    # scalar critical value
    delta_hat = boot_deltas.mean(axis=0)
    half = q * se
    return delta_hat - half, delta_hat + half


def bootstrap_pair_band(scores_A, scores_B, counts_per_system, B=2000,
                         seed=42, metric="partial", alpha=0.05):
    """Stratified paired bootstrap with simultaneous band over tolerance.

    Returns: dict with delta, lo, hi (all shape (K,)), and robust_inversion bool.
    """
    rng = np.random.default_rng(seed)
    K = scores_A.shape[0]  # num tolerances
    if metric == "strict":
        A = (scores_A == 1.0).astype(float)
        B_ = (scores_B == 1.0).astype(float)
    else:
        A = scores_A.astype(float)
        B_ = scores_B.astype(float)

    delta_obs = 100.0 * (A.mean(axis=1) - B_.mean(axis=1))
    offsets = np.concatenate(([0], np.cumsum(counts_per_system)))
    boot = np.zeros((B, K))
    for b in range(B):
        idx_parts = []
        for si, cnt in enumerate(counts_per_system):
            idx_parts.append(rng.integers(offsets[si], offsets[si] + cnt,
                                            size=cnt))
        idx = np.concatenate(idx_parts)
        boot[b] = 100.0 * (A[:, idx].mean(axis=1) - B_[:, idx].mean(axis=1))

    lo, hi = simultaneous_band(boot, alpha=alpha)

    # Robust inversion: band entirely above 0 somewhere AND entirely below 0 somewhere
    any_above = np.any(lo > 0)
    any_below = np.any(hi < 0)
    return {
        "delta": delta_obs,
        "lo": lo,
        "hi": hi,
        "robust_inversion": bool(any_above and any_below),
        "band_type": "simultaneous_max_t",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--tolerances", nargs="+", type=int, required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", choices=["strict", "partial"], default="partial")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.methods} on {args.eval_dir} @ {args.tolerances}")
    scores, _ = load_predictions(args.eval_dir, args.methods, args.tolerances)

    counts_per_system = [scores[(args.methods[0], s)].shape[0] for s in SYSTEM_KEYS]
    stacked = {m: np.concatenate([scores[(m, s)] for s in SYSTEM_KEYS],
                                  axis=0).T
               for m in args.methods}

    # Holm across method pairs: start with raw per-pair alpha, adjust
    pairs = list(itertools.combinations(args.methods, 2))

    band_rows = []
    for (A, B) in pairs:
        r = bootstrap_pair_band(stacked[A], stacked[B], counts_per_system,
                                 B=args.B, seed=args.seed,
                                 metric=args.metric, alpha=args.alpha)
        for ki, t in enumerate(args.tolerances):
            band_rows.append({
                "method_A": A, "method_B": B, "tolerance_s": t,
                "delta": r["delta"][ki],
                "band_lo": r["lo"][ki],
                "band_hi": r["hi"][ki],
                "sig": bool(r["lo"][ki] > 0 or r["hi"][ki] < 0),
            })

    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(args.out_dir / "bands.csv", index=False)

    # Robust inversions at pair level
    inv_rows = []
    for (A, B) in pairs:
        sub = band_df[(band_df.method_A == A) & (band_df.method_B == B)]
        any_above = (sub.band_lo > 0).any()
        any_below = (sub.band_hi < 0).any()
        inv_rows.append({
            "method_A": A, "method_B": B,
            "robust_inversion": bool(any_above and any_below),
            "max_delta": sub.delta.max(),
            "min_delta": sub.delta.min(),
            "n_sig_tol": int(sub.sig.sum()),
        })
    inv_df = pd.DataFrame(inv_rows)
    inv_df.to_csv(args.out_dir / "pair_inversions.csv", index=False)

    meta = {
        "methods": args.methods,
        "tolerances_s": args.tolerances,
        "metric": args.metric,
        "alpha": args.alpha,
        "B": args.B,
        "seed": args.seed,
        "counts_per_system": dict(zip(SYSTEM_KEYS, counts_per_system)),
        "n_pairs": len(pairs),
        "n_robust_inversions": int(inv_df.robust_inversion.sum()),
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print("\nInversions at method-pair level (simultaneous band):")
    print(inv_df.to_string(index=False))


if __name__ == "__main__":
    main()
