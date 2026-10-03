#!/usr/bin/env python3
"""Stress Test A — Leave-K-Systems-Out (k=2) jackknife for F5 robustness.

For the headline BARO vs z_family regret pair (n=11 systems), drop every
C(11,2)=55 pair of systems and recompute mis_selection_rate, mean_regret,
max_regret on the remaining 9 systems.

Output: distribution of those three quantities across the 55 sub-samples,
plus per-sub-sample detail for traceability.

Question answered: "Is the 27.3% headline mis-rate driven by a narrow subset
of flip systems, or does it survive worst-case 2-system drops?"

Uses the unified 11-system block inputs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bootstrap_pairwise as bp  # noqa: E402
import regret_analysis as ra     # noqa: E402


def patch_system_files(inputs_dir: Path) -> list:
    patched = []
    for (benchmark, system, path, acc_col) in bp.SYSTEM_FILES:
        new_path = inputs_dir / path.name
        patched.append((benchmark, system, new_path, acc_col))
    return patched


def compute_one_subsample(
    df: pd.DataFrame,
    drop_systems: tuple,
    method_a: str = "baro",
    method_b: str = "z_family",
    bootstrap_iters: int = 0,
) -> dict:
    """Drop the specified systems, compute regret on remainder.

    bootstrap_iters=0 disables bootstrap CI (much faster; we only need
    point estimates per sub-sample for the distribution).
    """
    sub = df[~df["system_id"].isin(drop_systems)].copy()
    res = ra.compute_regret_pair(
        sub,
        method_a=method_a,
        method_b=method_b,
        min_systems=2,       # allow small retained sets
        n_iter=bootstrap_iters if bootstrap_iters > 0 else 1,
        seed=42,
    )
    # Skip bootstrap CI by post-processing: if bootstrap_iters == 0,
    # we already ran 1 iter (cheap) but will not report it.
    return {
        "dropped": list(drop_systems),
        "n_retained": res["n_systems"],
        "decision_systems": res["decision_systems"],
        "mis_rate": res["mis_selection_rate"],
        "mean_regret": res["mean_regret"],
        "max_regret": res["max_regret"],
        "per_system": res["per_system_details"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--method-a", default="baro")
    ap.add_argument("--method-b", default="z_family")
    ap.add_argument("--k", type=int, default=2, help="number of systems to drop")
    args = ap.parse_args()

    print(f"[load] unified inputs: {args.inputs_dir}")
    patched = patch_system_files(args.inputs_dir)
    missing = [str(p) for (_, _, p, _) in patched if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(missing))

    original = list(bp.SYSTEM_FILES)
    bp.SYSTEM_FILES = patched
    try:
        df = bp.load_all_systems()
        systems = sorted(df["system_id"].unique())
        print(f"       n_systems={len(systems)}, methods={sorted(df['method'].unique())}")
        print(f"       systems: {systems}")

        # Baseline (k=0): full 11 systems.
        baseline = compute_one_subsample(df, drop_systems=(),
                                         method_a=args.method_a,
                                         method_b=args.method_b)
        print(f"\n[baseline k=0]  mis_rate={baseline['mis_rate']*100:.2f}%  "
              f"mean={baseline['mean_regret']*100:.3f}pp  "
              f"max={baseline['max_regret']*100:.2f}pp  "
              f"n_retained={baseline['n_retained']}")

        # LKSO k=2: all C(11,2) = 55 pairs.
        pairs = list(itertools.combinations(systems, args.k))
        print(f"\n[LKSO k={args.k}]  evaluating {len(pairs)} sub-samples ...")
        records = []
        for pair in pairs:
            r = compute_one_subsample(df, drop_systems=pair,
                                      method_a=args.method_a,
                                      method_b=args.method_b)
            records.append(r)

        # Summary.
        mis = np.array([r["mis_rate"] for r in records])
        mean_r = np.array([r["mean_regret"] for r in records])
        max_r = np.array([r["max_regret"] for r in records])

        def pct(x, q): return float(np.percentile(x, q))
        summary = {
            "method_a": args.method_a,
            "method_b": args.method_b,
            "k": args.k,
            "n_subsamples": len(pairs),
            "baseline_k0": {
                "mis_rate": baseline["mis_rate"],
                "mean_regret": baseline["mean_regret"],
                "max_regret": baseline["max_regret"],
            },
            "mis_rate": {
                "min": float(mis.min()),
                "p5": pct(mis, 5),
                "p25": pct(mis, 25),
                "median": float(np.median(mis)),
                "mean": float(mis.mean()),
                "p75": pct(mis, 75),
                "p95": pct(mis, 95),
                "max": float(mis.max()),
                "frac_gte_10pct": float((mis >= 0.10).mean()),
                "frac_gte_15pct": float((mis >= 0.15).mean()),
                "frac_gte_20pct": float((mis >= 0.20).mean()),
            },
            "mean_regret_pp": {
                "min": float(mean_r.min()*100),
                "median": float(np.median(mean_r)*100),
                "mean": float(mean_r.mean()*100),
                "max": float(mean_r.max()*100),
            },
            "max_regret_pp": {
                "min": float(max_r.min()*100),
                "median": float(np.median(max_r)*100),
                "mean": float(max_r.mean()*100),
                "max": float(max_r.max()*100),
            },
        }

        print("\n[LKSO summary]")
        print(json.dumps(summary, indent=2))

        # Write outputs.
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Detailed per-sub-sample CSV.
        detail_rows = []
        for r in records:
            detail_rows.append({
                "dropped_1": r["dropped"][0] if len(r["dropped"]) > 0 else "",
                "dropped_2": r["dropped"][1] if len(r["dropped"]) > 1 else "",
                "n_retained": r["n_retained"],
                "decision_systems": r["decision_systems"],
                "mis_rate": r["mis_rate"],
                "mean_regret": r["mean_regret"],
                "max_regret": r["max_regret"],
            })
        det_df = pd.DataFrame(detail_rows).sort_values(
            by=["mis_rate", "mean_regret"], ascending=[True, True]
        )
        det_path = args.output_dir / f"lkso_k{args.k}_detail.csv"
        det_df.to_csv(det_path, index=False)
        print(f"\n[written] {det_path}")

        # Summary JSON.
        sum_path = args.output_dir / f"lkso_k{args.k}_summary.json"
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[written] {sum_path}")

        # Per-system flip frequency: how often is each system a flip-system
        # across the sub-samples that retain it?
        sys_flip_counts = {s: {"retained": 0, "flipped_on": 0} for s in systems}
        for r in records:
            retained = set(systems) - set(r["dropped"])
            for pd_ in r["per_system"]:
                s = pd_["system"]
                if s in retained:
                    sys_flip_counts[s]["retained"] += 1
                    if pd_["mis_selected"] == 1:
                        sys_flip_counts[s]["flipped_on"] += 1
        per_sys = []
        for s, v in sys_flip_counts.items():
            per_sys.append({
                "system": s,
                "sub_samples_retained": v["retained"],
                "times_flipped": v["flipped_on"],
                "p_flip_in_subsamples": (v["flipped_on"] / v["retained"]
                                          if v["retained"] else 0.0),
            })
        per_sys_df = pd.DataFrame(per_sys).sort_values("p_flip_in_subsamples",
                                                       ascending=False)
        per_sys_path = args.output_dir / f"lkso_k{args.k}_per_system_flips.csv"
        per_sys_df.to_csv(per_sys_path, index=False)
        print(f"[written] {per_sys_path}")

    finally:
        bp.SYSTEM_FILES = original


if __name__ == "__main__":
    main()
