#!/usr/bin/env python3
"""Driver: run regret_analysis.py on the unified 11-system block.

The stock `regret_analysis.py --inputs-dir` hook only rewrites paths whose
`parent.name == "_inputs"` — OpenRCA entries in SYSTEM_FILES live in
`results_openrca_xbench/` and therefore are NOT rewritten. That was fine
for the original z_family run because OpenRCA data was only rebuilt in
the unified block.

This driver monkey-patches SYSTEM_FILES in full so every one of the 11
CSVs is loaded from `results_xbench_11sys_unified/_inputs/`, then calls
`run_all_pairs`. We do NOT edit `regret_analysis.py`.

Usage (on the 3090 server):
  conda activate auditstack
  python scripts/openrca_eval/run_regret_unified.py \
      --inputs-dir $HOME/auditstack/results_xbench_11sys_unified/_inputs \
      --output-csv $HOME/auditstack/results_stats/regret_table_unified.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bootstrap_pairwise as bp  # noqa: E402
import regret_analysis as ra      # noqa: E402


def patch_system_files(inputs_dir: Path) -> list:
    """Return a patched SYSTEM_FILES list where every CSV path points into
    `inputs_dir` (including OpenRCA, which the stock hook leaves alone)."""
    patched = []
    for (benchmark, system, path, acc_col) in bp.SYSTEM_FILES:
        new_path = inputs_dir / path.name
        patched.append((benchmark, system, new_path, acc_col))
    return patched


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--min-systems", type=int, default=ra.MIN_SYSTEMS_DEFAULT)
    ap.add_argument("--n-iter", type=int, default=ra.BOOTSTRAP_ITERS)
    ap.add_argument("--seed", type=int, default=ra.BOOTSTRAP_SEED)
    args = ap.parse_args()

    print(f"[load] unified inputs: {args.inputs_dir}")

    # Verify every file exists before patching.
    patched = patch_system_files(args.inputs_dir)
    missing = [str(p) for (_, _, p, _) in patched if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing unified-block inputs:\n  " + "\n  ".join(missing))

    original = list(bp.SYSTEM_FILES)
    bp.SYSTEM_FILES = patched
    try:
        df = bp.load_all_systems()
        print(f"       n_rows={len(df)}  "
              f"n_systems={df['system_id'].nunique()}  "
              f"n_methods={df['method'].nunique()}")
        print(f"       methods (after z_family collapse): "
              f"{sorted(df['method'].unique())}")
        print(f"       methods_raw: {sorted(df['method_raw'].unique())}")
        cov = df.groupby(["system_id", "method"]).size().unstack(fill_value=0)
        print("       coverage (cases per system x method):")
        print(cov.to_string())

        print("\n[regret] computing all pairs on unified block ...")
        out = ra.run_all_pairs(
            df,
            min_systems=args.min_systems,
            output_csv=args.output_csv,
            n_iter=args.n_iter,
            seed=args.seed,
        )
        print(f"[regret] wrote {args.output_csv} ({len(out)} rows)")
        per_sys_csv = args.output_csv.with_name(args.output_csv.stem + "_per_system.csv")
        print(f"         per-system dump: {per_sys_csv}")

        ra._print_headline(out, label=f"unified:{args.inputs_dir.name}")

        # Focus on BARO vs z_family (still the headline pair after the loader
        # collapses zbase_univ -> z_family).
        baro_zf = out[
            ((out["method_a"] == "baro") & (out["method_b"] == "z_family")) |
            ((out["method_a"] == "z_family") & (out["method_b"] == "baro"))
        ]
        if len(baro_zf) == 1 and not bool(baro_zf.iloc[0]["skipped"]):
            r = baro_zf.iloc[0]
            rate = float(r["mis_selection_rate"])
            print(f"\n[sanity] BARO vs z_family mis_selection_rate = {rate:.4f}  "
                  f"(rate in (0,1) expected).")
    finally:
        bp.SYSTEM_FILES = original


if __name__ == "__main__":
    main()
