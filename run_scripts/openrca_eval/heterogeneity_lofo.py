#!/usr/bin/env python3
"""Leave-one-family-out sensitivity for the heterogeneity meta-analysis.

For each benchmark family B ∈ {openrca, rcaeval, petshop}, drop all B's
subsystems and recompute meta-analysis on the remaining ~7-8 systems.
Also: leave-one-system-out (k=1) sensitivity on I² / PI.

Writes: results_stats/heterogeneity/lofo_sensitivity.csv
        results_stats/heterogeneity/loso_i2_sensitivity.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from heterogeneity_analysis import (  # noqa: E402
    load_unified_long,
    per_system_paired_vectors,
    meta_analyze,
    assess_tier,
)


def _recompute_meta(df, method_a, method_b, drop_systems=()):
    paired = per_system_paired_vectors(df, method_a, method_b)
    keep = {s: d for s, d in paired.items() if s not in drop_systems}
    if len(keep) < 3:
        return None
    effects = []
    variances = []
    names = []
    for s in sorted(keep.keys()):
        d = keep[s]
        effects.append(float(d.mean()))
        variances.append(float(np.var(d, ddof=1) / len(d)))
        names.append(s)
    effects_arr = np.array(effects)
    var_arr = np.array(variances)
    meta = meta_analyze(effects_arr, var_arr, row_names=names)
    tier = assess_tier(meta["I2"],
                        meta["pi_lo"] * 100,
                        meta["pi_hi"] * 100)
    return {
        "k": meta["k"],
        "Q": meta["Q"],
        "p_Q": meta["p_Q"],
        "I2": meta["I2"],
        "tau2_pp2": meta["tau2"] * 1e4,
        "tau_pp": float(np.sqrt(meta["tau2"])) * 100,
        "delta_FE_pp": meta["delta_FE"] * 100,
        "delta_RE_pp": meta["delta_RE"] * 100,
        "re_ci_lo_pp": meta["re_ci_lo_hksj"] * 100,
        "re_ci_hi_pp": meta["re_ci_hi_hksj"] * 100,
        "pi_lo_pp": meta["pi_lo"] * 100,
        "pi_hi_pp": meta["pi_hi"] * 100,
        "pi_crosses_zero": tier["pi_crosses_zero"],
        "effective_tier": tier["effective_tier"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--method-a", default="baro")
    ap.add_argument("--method-b", default="z_family")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_unified_long(args.inputs_dir)
    all_systems = sorted(df["system_id"].unique())

    # Baseline (full 11)
    baseline = _recompute_meta(df, args.method_a, args.method_b, drop_systems=())

    # --- LOFO (leave-one-family-out) --- #
    families = {"openrca": [], "rcaeval": [], "petshop": []}
    for s in all_systems:
        families[s.split("::")[0]].append(s)

    lofo_rows = [{"drop_family": "NONE (baseline k=11)", **baseline}]
    for fam, members in families.items():
        r = _recompute_meta(df, args.method_a, args.method_b,
                            drop_systems=set(members))
        if r is None:
            continue
        lofo_rows.append({"drop_family": fam, **r})

    lofo_df = pd.DataFrame(lofo_rows)
    lofo_path = args.output_dir / "lofo_sensitivity.csv"
    lofo_df.to_csv(lofo_path, index=False)
    print("\n=== LOFO (leave-one-family-out) ===")
    print(lofo_df.to_string(index=False))
    print(f"\n[written] {lofo_path}")

    # --- LOSO (leave-one-system-out, on I² / PI only) --- #
    loso_rows = []
    for drop in all_systems:
        r = _recompute_meta(df, args.method_a, args.method_b,
                            drop_systems={drop})
        loso_rows.append({"drop_system": drop, **r})
    loso_df = pd.DataFrame(loso_rows)
    loso_path = args.output_dir / "loso_i2_sensitivity.csv"
    loso_df.to_csv(loso_path, index=False)
    print("\n=== LOSO I² sensitivity (drop each single system) ===")
    print(loso_df[["drop_system", "I2", "delta_RE_pp",
                    "pi_lo_pp", "pi_hi_pp", "pi_crosses_zero",
                    "effective_tier"]].to_string(index=False))
    print(f"\n[written] {loso_path}")


if __name__ == "__main__":
    main()
