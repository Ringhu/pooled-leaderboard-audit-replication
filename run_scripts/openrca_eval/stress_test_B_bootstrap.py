#!/usr/bin/env python3
"""Stress Test B — case-level bootstrap flip stability for F5.

For each of the 11 systems, bootstrap resample cases (with replacement)
within that system, recompute BARO vs z_family acc@1, and count how often
the observed winner (BARO vs z_family) is reproduced.

Output: per-system P(BARO wins | bootstrap) for interpretation of flip
robustness. Particular interest is whether the 3 mis-selection systems
(bank: 3.35pp margin, petshop::high: 7.69pp margin, rcaeval::OB: 1.60pp
margin) have bootstrap flip probabilities that justify calling them
"robust flips" vs "coin flips".

Also computes joint distribution across the 11 systems to check whether
the observed "3 flips out of 11" count is typical or an outlier.

Uses the unified 11-system block inputs. 5000 iters, seed=42.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bootstrap_pairwise as bp  # noqa: E402


def patch_system_files(inputs_dir: Path) -> list:
    patched = []
    for (benchmark, system, path, acc_col) in bp.SYSTEM_FILES:
        new_path = inputs_dir / path.name
        patched.append((benchmark, system, new_path, acc_col))
    return patched


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--method-a", default="baro")
    ap.add_argument("--method-b", default="z_family")
    ap.add_argument("--n-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
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
        print(f"       n_systems={len(systems)}")

        # Prep per-system arrays of per-case correctness for each method.
        per_sys_arrays = {}
        for s in systems:
            sub = df[df["system_id"] == s]
            a_vec = (sub[sub["method"] == args.method_a]
                     .groupby("case_id")["correct"].mean().to_numpy())
            b_vec = (sub[sub["method"] == args.method_b]
                     .groupby("case_id")["correct"].mean().to_numpy())
            if a_vec.size == 0 or b_vec.size == 0:
                continue
            per_sys_arrays[s] = {
                "a": a_vec, "b": b_vec,
                "n_a": a_vec.size, "n_b": b_vec.size,
                "obs_a": float(a_vec.mean()),
                "obs_b": float(b_vec.mean()),
            }

        valid_systems = list(per_sys_arrays.keys())
        print(f"       valid systems: {len(valid_systems)}")
        print(f"       methods: {args.method_a} vs {args.method_b}")

        # Bootstrap: for each iter, resample cases in each system, record
        # bootstrapped (acc_a, acc_b) per system.
        rng = np.random.default_rng(args.seed)
        n_iter = args.n_iter
        n_sys = len(valid_systems)
        boot_a = np.zeros((n_iter, n_sys))
        boot_b = np.zeros((n_iter, n_sys))

        for it in range(n_iter):
            for j, s in enumerate(valid_systems):
                a = per_sys_arrays[s]["a"]
                b = per_sys_arrays[s]["b"]
                ia = rng.integers(0, a.size, size=a.size)
                ib = rng.integers(0, b.size, size=b.size)
                boot_a[it, j] = a[ia].mean()
                boot_b[it, j] = b[ib].mean()

        # Per-system P(A > B), P(A < B), P(tie).
        rows = []
        for j, s in enumerate(valid_systems):
            diff = boot_a[:, j] - boot_b[:, j]
            p_a_wins = float((diff > 0).mean())
            p_b_wins = float((diff < 0).mean())
            p_tie = float((diff == 0).mean())
            obs_a = per_sys_arrays[s]["obs_a"]
            obs_b = per_sys_arrays[s]["obs_b"]
            obs_winner = (args.method_a if obs_a > obs_b
                          else (args.method_b if obs_b > obs_a else "tie"))
            obs_margin_pp = (obs_a - obs_b) * 100

            # CI on A_acc and B_acc and on (A-B).
            a_ci = np.quantile(boot_a[:, j], [0.025, 0.975])
            b_ci = np.quantile(boot_b[:, j], [0.025, 0.975])
            d_ci = np.quantile(diff, [0.025, 0.975])

            rows.append({
                "system": s,
                "n_cases_a": per_sys_arrays[s]["n_a"],
                "n_cases_b": per_sys_arrays[s]["n_b"],
                "obs_acc_a": obs_a,
                "obs_acc_b": obs_b,
                "obs_margin_pp": obs_margin_pp,
                "obs_winner": obs_winner,
                "p_A_wins": p_a_wins,
                "p_B_wins": p_b_wins,
                "p_tie": p_tie,
                "a_ci_lo": float(a_ci[0]),
                "a_ci_hi": float(a_ci[1]),
                "b_ci_lo": float(b_ci[0]),
                "b_ci_hi": float(b_ci[1]),
                "diff_ci_lo_pp": float(d_ci[0])*100,
                "diff_ci_hi_pp": float(d_ci[1])*100,
                "winner_confidence": max(p_a_wins, p_b_wins),
            })

        rows_df = pd.DataFrame(rows).sort_values("winner_confidence",
                                                 ascending=False)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rows_df.to_csv(args.output_dir / "per_system_flip_stability.csv", index=False)
        print("\n=== Per-system flip stability (sorted by winner confidence) ===")
        print(rows_df.to_string(index=False))
        print(f"\n[written] {args.output_dir / 'per_system_flip_stability.csv'}")

        # --- Joint distribution: number of systems where A wins across iters ---
        # "A wins on s" per iter, count across systems per iter.
        a_wins_mask = boot_a > boot_b    # shape (n_iter, n_sys)
        n_a_wins_per_iter = a_wins_mask.sum(axis=1)
        observed_a_wins = int(sum(1 for r in rows if r["obs_winner"] == args.method_a))
        dist = pd.Series(n_a_wins_per_iter).value_counts().sort_index()
        print(f"\n=== Joint: number of systems where {args.method_a} wins in each bootstrap iter ===")
        print(dist)
        print(f"  observed (point estimate): {observed_a_wins} systems")

        # Also compute P(3 flip systems all flip in BOTH directions consistently)
        # i.e. P(baro wins simultaneously on bank, petshop::high, rcaeval::OB).
        flip_candidates = ["openrca::bank", "petshop::high_traffic", "rcaeval::OB"]
        idx_map = {s: j for j, s in enumerate(valid_systems)}
        flip_idx = [idx_map[s] for s in flip_candidates if s in idx_map]
        if len(flip_idx) == len(flip_candidates):
            all_three_flip = a_wins_mask[:, flip_idx].all(axis=1)
            any_flip = a_wins_mask[:, flip_idx].any(axis=1)
            p_all_three = float(all_three_flip.mean())
            p_any = float(any_flip.mean())
            print(f"\n=== Joint on 3 flip candidates ({flip_candidates}) ===")
            print(f"  P(all 3 flip simultaneously in bootstrap) = {p_all_three:.4f}")
            print(f"  P(at least 1 flips in bootstrap)          = {p_any:.4f}")
            # Per-count distribution.
            counts = a_wins_mask[:, flip_idx].sum(axis=1)
            counts_dist = pd.Series(counts).value_counts().sort_index()
            print(f"  distribution of flip-count among these 3:")
            print(counts_dist)
        else:
            p_all_three = None
            p_any = None

        summary = {
            "method_a": args.method_a,
            "method_b": args.method_b,
            "n_iter": args.n_iter,
            "seed": args.seed,
            "n_systems": len(valid_systems),
            "per_system": rows,
            "joint": {
                "n_systems_where_A_wins_observed": observed_a_wins,
                "n_systems_where_A_wins_distribution": {
                    int(k): int(v) for k, v in dist.items()
                },
                "flip_candidates": flip_candidates,
                "p_all_three_flip": p_all_three,
                "p_any_of_three_flip": p_any,
            },
        }
        with open(args.output_dir / "flip_stability_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[written] {args.output_dir / 'flip_stability_summary.json'}")

    finally:
        bp.SYSTEM_FILES = original


if __name__ == "__main__":
    main()
