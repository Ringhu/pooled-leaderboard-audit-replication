#!/usr/bin/env python3
"""Held-out-system regret metric for the "System Matters" RCA audit.

Converts "method × system interaction is significant" into a practical
statement: if you pick your RCA method based on pooled acc@1 over all-but-one
systems, how often would you pick wrong on the held-out system, and by how
much would you lose?

Operational definition (per method pair (A, B) with data on >= K systems,
K >= 6):

  For each held-out system s_i:
    1. pooled_A = mean over other systems of mean_acc@1(A on s_j)
    2. pooled_B = mean over other systems of mean_acc@1(B on s_j)
    3. pooled_winner = argmax(pooled_A, pooled_B); ties flagged and
       treated as "no mis-selection, zero regret" (conservative).
    4. actual_A = mean_acc@1(A on s_i); actual_B = mean_acc@1(B on s_i)
    5. actual_winner = argmax(actual_A, actual_B); ties flagged.
    6. mis_selection_i = 1 iff pooled_winner != actual_winner (both non-tie).
    7. regret_i = max(0, acc_on_si(actual_winner) - acc_on_si(pooled_winner)).

  mis_selection_rate = mean(mis_selection_i)  (excluding ties from numerator
                                                AND from denominator — so the
                                                rate is out of "decision-making
                                                systems").
  mean_regret        = mean(regret_i over all i, including ties as 0).
  max_regret         = max(regret_i).

Bootstrap CI on mean_regret: case-level resampling *within* each system,
5000 iters, seed=42 (same seed as the rest of the audit). Yields
[mean_regret_lo, mean_regret_hi] at 95%.

Reuse: loads data via `load_all_systems` from bootstrap_pairwise.py so that
method collapse (zbase + zbase_univ -> z_family) and case_id construction
stay consistent.

Usage:
  python scripts/openrca_eval/regret_analysis.py                           # default inputs dir
  python scripts/openrca_eval/regret_analysis.py --inputs-dir <path>       # override inputs
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Reuse the established loader for consistent method collapse + case_id.
from bootstrap_pairwise import ROOT, SYSTEM_FILES, Z_FAMILY_RAWS, load_all_systems  # noqa: E402

# ------------------------------------------------------------------ #
# Constants                                                          #
# ------------------------------------------------------------------ #
DEFAULT_OUT_CSV = ROOT / "results_stats" / "regret_table.csv"
BOOTSTRAP_ITERS = 5000
BOOTSTRAP_SEED = 42
MIN_SYSTEMS_DEFAULT = 6


# ------------------------------------------------------------------ #
# Loader override: honor --inputs-dir                                #
# ------------------------------------------------------------------ #
def load_with_inputs_dir(inputs_dir: Optional[Path]) -> pd.DataFrame:
    """Wraps load_all_systems with an optional self-contained inputs dir.

    When --inputs-dir is given, every system CSV is read from that directory by
    filename. This makes the regret script runnable from an anonymous artifact
    without the original /cluster path layout.
    """
    if inputs_dir is None:
        return load_all_systems()

    import bootstrap_pairwise as bp
    # Build patched system_files in place.
    original = list(bp.SYSTEM_FILES)
    patched = []
    for (benchmark, system, path, acc_col) in original:
        new_path = inputs_dir / path.name
        patched.append((benchmark, system, new_path, acc_col))
    bp.SYSTEM_FILES = patched
    try:
        return load_all_systems()
    finally:
        bp.SYSTEM_FILES = original


# ------------------------------------------------------------------ #
# Per-system aggregates                                              #
# ------------------------------------------------------------------ #
def method_system_means(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame indexed by (method, system_id) with mean acc@1 and
    case count (using case_id as the unit — matches bootstrap_pairwise)."""
    agg = (
        df.groupby(["method", "system_id"])["correct"]
          .agg(["mean", "count"])
          .reset_index()
          .rename(columns={"mean": "acc1", "count": "n_cases"})
    )
    return agg


# ------------------------------------------------------------------ #
# Core regret computation for one pair                               #
# ------------------------------------------------------------------ #
def _pooled_leaveout(acc_by_sys: Dict[str, float], held_out: str) -> float:
    """Mean acc@1 averaged across systems != held_out (macro, equal-weight).
    `acc_by_sys` must contain `held_out` plus >=1 other key."""
    others = [v for s, v in acc_by_sys.items() if s != held_out]
    return float(np.mean(others))


def _bootstrap_mean_regret_ci(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    systems: List[str],
    n_iter: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> (float, float):
    """Case-level resample WITHIN each system (preserves per-system case
    counts), re-compute LOSO regret per iteration, return 2.5/97.5 quantiles
    of mean_regret across the LOSO loop."""
    rng = np.random.default_rng(seed)

    # Prep aligned per-system paired arrays so bootstrap resamples the same
    # case rows for both methods.
    per_sys_arrays: Dict[str, np.ndarray] = {}
    for s in systems:
        sub = df[df["system_id"] == s]
        pivot = (
            sub[sub["method"].isin([method_a, method_b])]
            .pivot_table(
                index="case_id",
                columns="method",
                values="correct",
                aggfunc="mean",
            )
            .dropna(subset=[method_a, method_b])
        )
        if pivot.empty:
            continue
        per_sys_arrays[s] = pivot[[method_a, method_b]].to_numpy(dtype=float)

    valid_systems = list(per_sys_arrays.keys())
    if len(valid_systems) < 2:
        return (float("nan"), float("nan"))

    mean_regrets = np.empty(n_iter, dtype=float)
    for it in range(n_iter):
        # Resample cases within each system.
        resampled_a: Dict[str, float] = {}
        resampled_b: Dict[str, float] = {}
        for s in valid_systems:
            arr = per_sys_arrays[s]
            idx = rng.integers(0, arr.shape[0], size=arr.shape[0])
            sample = arr[idx]
            resampled_a[s] = float(sample[:, 0].mean())
            resampled_b[s] = float(sample[:, 1].mean())

        regrets = []
        for s_i in valid_systems:
            if len(valid_systems) < 2:
                continue
            pooled_a = _pooled_leaveout(resampled_a, s_i)
            pooled_b = _pooled_leaveout(resampled_b, s_i)
            actual_a = resampled_a[s_i]
            actual_b = resampled_b[s_i]

            if pooled_a == pooled_b:
                # Tie on pooled recommendation: conservative zero regret.
                regrets.append(0.0)
                continue
            pooled_winner_a = pooled_a > pooled_b
            # Regret: actual winner's acc minus pooled-recommended method's acc.
            best_actual = max(actual_a, actual_b)
            pooled_recommended_acc = actual_a if pooled_winner_a else actual_b
            regrets.append(max(0.0, best_actual - pooled_recommended_acc))
        mean_regrets[it] = float(np.mean(regrets)) if regrets else 0.0

    ci_lo, ci_hi = np.quantile(mean_regrets, [0.025, 0.975])
    return (float(ci_lo), float(ci_hi))


def compute_regret_pair(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    min_systems: int = MIN_SYSTEMS_DEFAULT,
    n_iter: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Compute held-out-system regret for a single method pair.

    Returns dict:
      {
        method_a, method_b,
        n_systems, systems,
        mis_selection_rate, mean_regret, max_regret,
        pooled_tie_count, actual_tie_count, decision_systems,
        regret_ci_lo, regret_ci_hi,
        per_system_details: [{system, pooled_a, pooled_b, actual_a, actual_b,
                              pooled_winner, actual_winner, mis_selected, regret}, ...],
        skipped (bool), skip_reason (str),
      }
    """
    ms = method_system_means(df)
    a_rows = ms[ms["method"] == method_a].set_index("system_id")
    b_rows = ms[ms["method"] == method_b].set_index("system_id")

    # Systems where both methods have >= 1 case.
    systems = sorted(set(a_rows.index) & set(b_rows.index))
    n_systems = len(systems)

    if n_systems < min_systems:
        return {
            "method_a": method_a,
            "method_b": method_b,
            "n_systems": n_systems,
            "systems": systems,
            "mis_selection_rate": float("nan"),
            "mean_regret": float("nan"),
            "max_regret": float("nan"),
            "pooled_tie_count": 0,
            "actual_tie_count": 0,
            "decision_systems": 0,
            "regret_ci_lo": float("nan"),
            "regret_ci_hi": float("nan"),
            "per_system_details": [],
            "skipped": True,
            "skip_reason": f"insufficient coverage ({n_systems} < {min_systems})",
        }

    acc_a = {s: float(a_rows.loc[s, "acc1"]) for s in systems}
    acc_b = {s: float(b_rows.loc[s, "acc1"]) for s in systems}

    # LOSO loop.
    per_system = []
    mis_flags: List[int] = []        # 1 if mis-selected, 0 if correct; ties excluded
    regrets: List[float] = []        # length = n_systems, ties count as 0 regret
    pooled_ties = 0
    actual_ties = 0

    for s_i in systems:
        pooled_a = _pooled_leaveout(acc_a, s_i)
        pooled_b = _pooled_leaveout(acc_b, s_i)
        actual_a = acc_a[s_i]
        actual_b = acc_b[s_i]

        pooled_tie = (pooled_a == pooled_b)
        actual_tie = (actual_a == actual_b)
        if pooled_tie:
            pooled_ties += 1
        if actual_tie:
            actual_ties += 1

        if pooled_tie:
            # Conservative: no mis-selection, zero regret; not counted in
            # decision-systems denominator.
            pooled_winner = "tie"
            mis = None
            regret = 0.0
        else:
            pooled_winner = method_a if pooled_a > pooled_b else method_b
            pooled_recommended_acc = actual_a if pooled_winner == method_a else actual_b
            best_actual = max(actual_a, actual_b)
            regret = max(0.0, best_actual - pooled_recommended_acc)

            if actual_tie:
                # Pooled picked a method; both methods tie on s_i => zero regret,
                # and no unique "actual_winner" to compare against. Treat as
                # NOT mis-selected (regret is 0, nothing lost).
                mis = 0
            else:
                actual_winner = method_a if actual_a > actual_b else method_b
                mis = int(pooled_winner != actual_winner)

        actual_winner_label = (
            "tie" if actual_tie
            else (method_a if actual_a > actual_b else method_b)
        )

        per_system.append({
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

    decision_systems = len(mis_flags)
    mis_selection_rate = float(np.mean(mis_flags)) if mis_flags else float("nan")
    mean_regret = float(np.mean(regrets))
    max_regret = float(np.max(regrets))

    ci_lo, ci_hi = _bootstrap_mean_regret_ci(
        df, method_a, method_b, systems, n_iter=n_iter, seed=seed,
    )

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_systems": n_systems,
        "systems": systems,
        "mis_selection_rate": mis_selection_rate,
        "mean_regret": mean_regret,
        "max_regret": max_regret,
        "pooled_tie_count": pooled_ties,
        "actual_tie_count": actual_ties,
        "decision_systems": decision_systems,
        "regret_ci_lo": ci_lo,
        "regret_ci_hi": ci_hi,
        "per_system_details": per_system,
        "skipped": False,
        "skip_reason": "",
    }


# ------------------------------------------------------------------ #
# Driver                                                             #
# ------------------------------------------------------------------ #
def run_all_pairs(
    df: pd.DataFrame,
    min_systems: int = MIN_SYSTEMS_DEFAULT,
    output_csv: Path = DEFAULT_OUT_CSV,
    n_iter: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Iterate over all method pairs with >=min_systems shared coverage;
    write CSV."""
    methods = sorted(df["method"].unique())
    rows = []
    skipped_rows = []
    per_system_dump = []

    for a, b in itertools.combinations(methods, 2):
        res = compute_regret_pair(
            df, a, b, min_systems=min_systems, n_iter=n_iter, seed=seed,
        )
        common_row = {
            "method_a": res["method_a"],
            "method_b": res["method_b"],
            "n_systems": res["n_systems"],
            "decision_systems": res["decision_systems"],
            "pooled_tie_count": res["pooled_tie_count"],
            "actual_tie_count": res["actual_tie_count"],
            "mis_selection_rate": res["mis_selection_rate"],
            "mean_regret": res["mean_regret"],
            "max_regret": res["max_regret"],
            "regret_ci_lo": res["regret_ci_lo"],
            "regret_ci_hi": res["regret_ci_hi"],
            "skipped": res["skipped"],
            "skip_reason": res["skip_reason"],
        }
        if res["skipped"]:
            skipped_rows.append(common_row)
        else:
            rows.append(common_row)
            for d in res["per_system_details"]:
                per_system_dump.append({
                    "method_a": a, "method_b": b, **d,
                })

    out = pd.DataFrame(rows + skipped_rows)
    out = out.sort_values(
        by=["skipped", "mis_selection_rate", "mean_regret"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    # Write per-system details alongside the aggregate for traceability.
    per_sys_csv = output_csv.with_name(output_csv.stem + "_per_system.csv")
    pd.DataFrame(per_system_dump).to_csv(per_sys_csv, index=False)

    return out


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #
def _format_pair_row(r: pd.Series) -> str:
    if r["skipped"]:
        return (f"  {r['method_a']:>10s} vs {r['method_b']:<10s}  "
                f"[SKIPPED: {r['skip_reason']}]")
    return (
        f"  {r['method_a']:>10s} vs {r['method_b']:<10s}  "
        f"systems={int(r['n_systems']):>2d}  "
        f"decisions={int(r['decision_systems']):>2d}  "
        f"mis_rate={r['mis_selection_rate']*100:5.1f}%  "
        f"mean_regret={r['mean_regret']*100:5.2f}pp "
        f"[{r['regret_ci_lo']*100:5.2f}, {r['regret_ci_hi']*100:5.2f}]  "
        f"max_regret={r['max_regret']*100:5.2f}pp"
    )


def _print_headline(out: pd.DataFrame, label: str) -> None:
    print(f"\n[{label}]")
    print(f"  total pairs evaluated: {len(out)}  "
          f"(analyzed={int((~out['skipped']).sum())}, "
          f"skipped={int(out['skipped'].sum())})")

    baro_zf = out[
        ((out["method_a"] == "baro") & (out["method_b"] == "z_family")) |
        ((out["method_a"] == "z_family") & (out["method_b"] == "baro"))
    ]
    if len(baro_zf) > 0:
        r = baro_zf.iloc[0]
        print(f"\n[{label} :: BARO vs z_family headline]")
        print(_format_pair_row(r))
    else:
        print(f"[{label}] WARN: no BARO vs z_family row found")

    analyzed = out[~out["skipped"]].copy()
    if len(analyzed) > 0:
        print(f"\n[{label} :: all analyzed pairs, sorted by mis_rate desc]")
        for _, r in analyzed.iterrows():
            print(_format_pair_row(r))
    skipped = out[out["skipped"]]
    if len(skipped) > 0:
        print(f"\n[{label} :: skipped pairs]")
        for _, r in skipped.iterrows():
            print(_format_pair_row(r))


def main():
    ap = argparse.ArgumentParser(description="Held-out-system regret metric.")
    ap.add_argument(
        "--inputs-dir", type=Path, default=None,
        help="Override merged inputs dir (default: results_xbench_11sys_with_zbase/_inputs). "
             "For refresh hook after R1, pass results_xbench_11sys_unified/_inputs.",
    )
    ap.add_argument(
        "--output-csv", type=Path, default=DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUT_CSV}).",
    )
    ap.add_argument(
        "--min-systems", type=int, default=MIN_SYSTEMS_DEFAULT,
        help="Minimum shared-coverage systems for a pair to be analyzed (default 6).",
    )
    ap.add_argument(
        "--n-iter", type=int, default=BOOTSTRAP_ITERS,
        help=f"Bootstrap iterations for regret CI (default {BOOTSTRAP_ITERS}).",
    )
    ap.add_argument(
        "--seed", type=int, default=BOOTSTRAP_SEED,
        help=f"Bootstrap seed (default {BOOTSTRAP_SEED}).",
    )
    ap.add_argument(
        "--compare-with", type=Path, default=None,
        help="Second --inputs-dir to also evaluate and summarize side-by-side.",
    )
    args = ap.parse_args()

    print("[load] primary inputs:",
          args.inputs_dir if args.inputs_dir else "(default with_zbase/_inputs)")
    df = load_with_inputs_dir(args.inputs_dir)
    print(f"       n_rows={len(df)}  "
          f"n_systems={df['system_id'].nunique()}  "
          f"n_methods={df['method'].nunique()}")
    cov = df.groupby(["system_id", "method"]).size().unstack(fill_value=0)
    print("       coverage (cases per system x method):")
    print(cov.to_string())

    print("\n[regret] computing all pairs ...")
    out = run_all_pairs(
        df,
        min_systems=args.min_systems,
        output_csv=args.output_csv,
        n_iter=args.n_iter,
        seed=args.seed,
    )
    print(f"[regret] wrote {args.output_csv} ({len(out)} rows)")
    print(f"         per-system dump: {args.output_csv.with_name(args.output_csv.stem + '_per_system.csv')}")

    _print_headline(out, label=str(args.inputs_dir) if args.inputs_dir else "default:with_zbase")

    # Sanity check on BARO vs z_family.
    baro_zf = out[
        ((out["method_a"] == "baro") & (out["method_b"] == "z_family")) |
        ((out["method_a"] == "z_family") & (out["method_b"] == "baro"))
    ]
    if len(baro_zf) == 1 and not bool(baro_zf.iloc[0]["skipped"]):
        r = baro_zf.iloc[0]
        rate = float(r["mis_selection_rate"])
        if rate == 0.0 or rate == 1.0:
            print(f"\n[SANITY-CHECK WARNING] BARO vs z_family mis_selection_rate = {rate:.3f}; "
                  f"expected strictly in (0, 1) given interaction significance. Investigate.")
        else:
            print(f"\n[sanity-check] BARO vs z_family mis_selection_rate = {rate:.3f}  (in (0,1), OK).")

    # Optional second run for refresh hook.
    if args.compare_with is not None:
        print("\n" + "=" * 78)
        print(f"[load] secondary inputs: {args.compare_with}")
        df2 = load_with_inputs_dir(args.compare_with)
        print(f"       n_rows={len(df2)}  "
              f"n_systems={df2['system_id'].nunique()}  "
              f"n_methods={df2['method'].nunique()}")
        second_out_csv = args.output_csv.with_name(args.output_csv.stem + "_unified.csv")
        out2 = run_all_pairs(
            df2,
            min_systems=args.min_systems,
            output_csv=second_out_csv,
            n_iter=args.n_iter,
            seed=args.seed,
        )
        print(f"[regret] wrote {second_out_csv} ({len(out2)} rows)")
        _print_headline(out2, label=str(args.compare_with))

        # Side-by-side BARO vs z_family.
        def _row(df_o):
            row = df_o[
                ((df_o["method_a"] == "baro") & (df_o["method_b"] == "z_family")) |
                ((df_o["method_a"] == "z_family") & (df_o["method_b"] == "baro"))
            ]
            return row.iloc[0] if len(row) else None

        r1 = _row(out); r2 = _row(out2)
        if r1 is not None and r2 is not None:
            print("\n[side-by-side :: BARO vs z_family]")
            print(f"  primary   ({args.inputs_dir or 'with_zbase'}):")
            print(_format_pair_row(r1))
            print(f"  secondary ({args.compare_with}):")
            print(_format_pair_row(r2))


if __name__ == "__main__":
    main()
