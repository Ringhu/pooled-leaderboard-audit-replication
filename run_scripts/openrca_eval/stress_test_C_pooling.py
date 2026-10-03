#!/usr/bin/env python3
"""Stress Test C — alternative pooling protocols for F5 robustness.

Baseline: pooled_winner = argmax of simple macro-mean over held-out systems
(equal-weight). This test recomputes the regret with 3 alternate poolings:

  1. Median over held-out systems
  2. n_cases-weighted mean (per-case weighting — so Bank's 136 cases dominate
     PetShop temporal's 8 cases)
  3. 10% trimmed mean (drop 1 min + 1 max from 10 held-out => pool over 8)

Question: are the 3 flip systems (bank / petshop::high / rcaeval::OB) still
mis-selected under these alternative protocols? Is F5 pooling-robust?

Uses the unified 11-system block inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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


# ------- Pooling functions ------- #
def pool_mean(accs: List[float], weights: List[float] = None) -> float:
    return float(np.mean(accs))


def pool_median(accs: List[float], weights: List[float] = None) -> float:
    return float(np.median(accs))


def pool_weighted(accs: List[float], weights: List[float]) -> float:
    a = np.asarray(accs, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.average(a, weights=w))


def pool_trimmed(accs: List[float], weights: List[float] = None,
                 trim_frac: float = 0.10) -> float:
    a = np.asarray(accs, dtype=float)
    n = len(a)
    k = int(np.floor(trim_frac * n))
    if 2 * k >= n:
        return float(np.median(a))
    a_sorted = np.sort(a)
    trimmed = a_sorted[k:n - k]
    return float(np.mean(trimmed))


POOLINGS = {
    "mean_baseline": lambda accs, w: pool_mean(accs, w),
    "median": lambda accs, w: pool_median(accs, w),
    "weighted_ncases": lambda accs, w: pool_weighted(accs, w),
    "trimmed10pct": lambda accs, w: pool_trimmed(accs, w, trim_frac=0.10),
}


def per_system_aggs(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Return {system: {method: {"acc1": float, "n_cases": int}}}."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    grp = df.groupby(["system_id", "method"])["correct"].agg(["mean", "count"])
    for (s, m), row in grp.iterrows():
        out.setdefault(s, {})[m] = {
            "acc1": float(row["mean"]),
            "n_cases": int(row["count"]),
        }
    return out


def compute_regret_one_pooling(
    aggs: Dict[str, Dict[str, Dict[str, float]]],
    method_a: str,
    method_b: str,
    pooling_name: str,
) -> dict:
    """LOSO regret with a custom pooling protocol."""
    pool_fn = POOLINGS[pooling_name]
    systems = sorted([s for s, m in aggs.items()
                      if method_a in m and method_b in m])

    per_sys_details = []
    mis_flags, regrets = [], []

    for s_i in systems:
        others = [s for s in systems if s != s_i]
        accs_a = [aggs[s][method_a]["acc1"] for s in others]
        accs_b = [aggs[s][method_b]["acc1"] for s in others]
        weights = [aggs[s][method_a]["n_cases"] for s in others]

        pooled_a = pool_fn(accs_a, weights)
        pooled_b = pool_fn(accs_b, weights)
        actual_a = aggs[s_i][method_a]["acc1"]
        actual_b = aggs[s_i][method_b]["acc1"]

        if pooled_a == pooled_b:
            pooled_winner = "tie"
            mis = None
            regret = 0.0
        else:
            pooled_winner = method_a if pooled_a > pooled_b else method_b
            pooled_recommended = actual_a if pooled_winner == method_a else actual_b
            best_actual = max(actual_a, actual_b)
            regret = max(0.0, best_actual - pooled_recommended)
            if actual_a == actual_b:
                mis = 0
            else:
                actual_winner = method_a if actual_a > actual_b else method_b
                mis = int(pooled_winner != actual_winner)

        actual_winner_label = ("tie" if actual_a == actual_b
                               else (method_a if actual_a > actual_b else method_b))
        per_sys_details.append({
            "system": s_i,
            "pooled_a": pooled_a,
            "pooled_b": pooled_b,
            "actual_a": actual_a,
            "actual_b": actual_b,
            "pooled_winner": pooled_winner,
            "actual_winner": actual_winner_label,
            "mis_selected": mis,
            "regret": regret,
        })
        if mis is not None:
            mis_flags.append(mis)
        regrets.append(regret)

    return {
        "pooling": pooling_name,
        "n_systems": len(systems),
        "decision_systems": len(mis_flags),
        "mis_rate": float(np.mean(mis_flags)) if mis_flags else float("nan"),
        "mean_regret": float(np.mean(regrets)),
        "max_regret": float(np.max(regrets)),
        "per_system": per_sys_details,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--method-a", default="baro")
    ap.add_argument("--method-b", default="z_family")
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
        aggs = per_system_aggs(df)
        print(f"       n_systems={len(aggs)}, methods={sorted(df['method'].unique())}")

        all_results = {}
        per_system_wide = {}  # system -> {pooling: flip_flag}

        for pooling_name in POOLINGS.keys():
            print(f"\n=== pooling: {pooling_name} ===")
            res = compute_regret_one_pooling(aggs, args.method_a, args.method_b,
                                             pooling_name)
            all_results[pooling_name] = {
                "n_systems": res["n_systems"],
                "decision_systems": res["decision_systems"],
                "mis_rate": res["mis_rate"],
                "mean_regret_pp": res["mean_regret"] * 100,
                "max_regret_pp": res["max_regret"] * 100,
            }
            print(f"  mis_rate={res['mis_rate']*100:.2f}%  "
                  f"mean={res['mean_regret']*100:.3f}pp  "
                  f"max={res['max_regret']*100:.3f}pp")
            print(f"  flip systems:")
            flip_systems = []
            for d in res["per_system"]:
                if d["mis_selected"] == 1:
                    flip_systems.append(d["system"])
                    print(f"    {d['system']:<35s}  pooled={d['pooled_winner']:<10s} "
                          f"actual={d['actual_winner']:<10s}  regret={d['regret']*100:.2f}pp "
                          f"(pa={d['pooled_a']:.4f} pb={d['pooled_b']:.4f} "
                          f"aa={d['actual_a']:.4f} ab={d['actual_b']:.4f})")
            all_results[pooling_name]["flip_systems"] = flip_systems

            # Build wide table of flip flags per system across poolings.
            for d in res["per_system"]:
                per_system_wide.setdefault(d["system"], {})[pooling_name] = {
                    "pooled_winner": d["pooled_winner"],
                    "actual_winner": d["actual_winner"],
                    "mis_selected": d["mis_selected"],
                    "regret_pp": d["regret"] * 100,
                }

        # Agreement summary: do all 4 poolings agree on which systems flip?
        print("\n=== Cross-pooling agreement ===")
        all_flip_sets = {k: set(v["flip_systems"]) for k, v in all_results.items()}
        print(f"  mean_baseline flips:   {sorted(all_flip_sets['mean_baseline'])}")
        print(f"  median flips:          {sorted(all_flip_sets['median'])}")
        print(f"  weighted_ncases flips: {sorted(all_flip_sets['weighted_ncases'])}")
        print(f"  trimmed10pct flips:    {sorted(all_flip_sets['trimmed10pct'])}")

        intersect = set.intersection(*all_flip_sets.values())
        union = set.union(*all_flip_sets.values())
        print(f"\n  intersection (always flip):  {sorted(intersect)}")
        print(f"  union (ever flip):           {sorted(union)}")

        # Write outputs.
        args.output_dir.mkdir(parents=True, exist_ok=True)

        with open(args.output_dir / "pooling_summary.json", "w") as f:
            json.dump({
                "method_a": args.method_a,
                "method_b": args.method_b,
                "results": all_results,
                "intersection_flips": sorted(intersect),
                "union_flips": sorted(union),
            }, f, indent=2)
        print(f"\n[written] {args.output_dir / 'pooling_summary.json'}")

        # Wide CSV: rows = systems, columns = (pooling, mis_selected) etc.
        wide_rows = []
        for s, per_pool in per_system_wide.items():
            row = {"system": s}
            for pooling_name in POOLINGS.keys():
                info = per_pool.get(pooling_name, {})
                row[f"{pooling_name}__mis"] = info.get("mis_selected")
                row[f"{pooling_name}__regret_pp"] = info.get("regret_pp")
                row[f"{pooling_name}__pooled_winner"] = info.get("pooled_winner")
            wide_rows.append(row)
        wide_df = pd.DataFrame(wide_rows).sort_values("system")
        wide_df.to_csv(args.output_dir / "pooling_per_system.csv", index=False)
        print(f"[written] {args.output_dir / 'pooling_per_system.csv'}")

    finally:
        bp.SYSTEM_FILES = original


if __name__ == "__main__":
    main()
