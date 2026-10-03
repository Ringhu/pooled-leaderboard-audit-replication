#!/usr/bin/env python3
"""Verify main-text random-effects meta-analysis numbers against cached JSON.

Recomputes per-comparator stats from raw effects/variances using
``statsmodels.stats.meta_analysis.combine_effects`` plus an independent
manual Cochran's Q and IntHout prediction interval, then compares against
the JSON ``meta_pp`` block. Writes a CSV verification table and exits 0
when every absolute diff is < 1e-3 (loose tolerance for HKSJ small-sample
floating drift).

Usage:
    python verify_main_text_numbers.py --repo-root /path/to/auditstack
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.meta_analysis import combine_effects


REPO_ROOT = Path("$HOME/auditstack")
RESULTS = REPO_ROOT / "results_stats"
COMPARATORS = [
    ("zbase_univ", RESULTS / "heterogeneity" / "meta_analysis_summary.json"),
    ("alertcount", RESULTS / "heterogeneity_alertcount" / "meta_analysis_summary.json"),
]
CSV_OUT = RESULTS / "verification_table.csv"
TOL = 1e-3

# Statistic name -> JSON key in meta_pp block (pp-scale).
STATS = [
    "k",
    "Q",
    "df",
    "p_Q",
    "I2",
    "tau2_pp2",
    "delta_RE_pp",
    "re_ci_lo_hksj_pp",
    "re_ci_hi_hksj_pp",
    "pi_lo_pp",
    "pi_hi_pp",
]


def parse_args() -> argparse.Namespace:
    default_root = Path.cwd()
    if not (default_root / "results_stats").exists():
        default_root = REPO_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_root,
        help="Repository root containing results_stats/.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <repo-root>/results_stats/verification_table.csv.",
    )
    return parser.parse_args()


def recompute(eff_pp: np.ndarray, var_prop2: np.ndarray) -> dict:
    """Recompute pp-scale meta-analysis stats from raw effects + variances.

    eff_pp: per-system effect sizes in percentage points.
    var_prop2: per-system variances in proportion-squared scale.
    """
    eff_prop = eff_pp / 100.0  # statsmodels takes effects in proportion scale.
    var_prop2 = np.asarray(var_prop2, dtype=float)

    # Sanity-check: variances should be ~1e-3 (proportion-squared), not pp-squared.
    assert var_prop2.max() < 1.0, f"variances look too large: {var_prop2.max()}"

    res = combine_effects(eff_prop, var_prop2, method_re="iterated", use_t=True)

    k = int(res.k)

    # Independent Cochran's Q on the proportion scale (invariant to /100 rescale
    # because Q = sum w_i * (eff_i - eff_FE)^2 with w_i = 1/v_i; weights and
    # squared deviations both rescale, but Q itself is scale-equivariant
    # consistently with how the JSON computed it).
    w = 1.0 / var_prop2
    eff_FE = (w * eff_prop).sum() / w.sum()
    Q_manual = float((w * (eff_prop - eff_FE) ** 2).sum())
    # Cross-check with statsmodels Q (must agree to numerical noise).
    assert abs(Q_manual - float(res.q)) < 1e-9, (Q_manual, float(res.q))

    df = k - 1
    p_Q = float(stats.chi2.sf(Q_manual, df=df))
    I2 = max(0.0, (Q_manual - df) / Q_manual) if Q_manual > 0 else 0.0

    tau2_prop2 = float(res.tau2)

    # Random-effects mean and HKSJ CI on the proportion scale.
    delta_RE_prop = float(res.mean_effect_re)
    se_RE_hksj_prop = float(res.sd_eff_w_re_hksj)
    t_crit_re = float(stats.t.ppf(0.975, df=df))  # k-1 df for HKSJ.
    re_ci_lo_prop = delta_RE_prop - t_crit_re * se_RE_hksj_prop
    re_ci_hi_prop = delta_RE_prop + t_crit_re * se_RE_hksj_prop

    # IntHout prediction interval: t_{k-2, 0.975} * sqrt(tau2 + se_RE^2).
    t_crit_pi = float(stats.t.ppf(0.975, df=k - 2))
    pi_half = t_crit_pi * np.sqrt(tau2_prop2 + se_RE_hksj_prop ** 2)
    pi_lo_prop = delta_RE_prop - pi_half
    pi_hi_prop = delta_RE_prop + pi_half

    return {
        "k": k,
        "Q": Q_manual,
        "df": df,
        "p_Q": p_Q,
        "I2": I2,
        "tau2_pp2": tau2_prop2 * 1e4,  # proportion^2 -> pp^2
        "delta_RE_pp": delta_RE_prop * 100.0,
        "re_ci_lo_hksj_pp": re_ci_lo_prop * 100.0,
        "re_ci_hi_hksj_pp": re_ci_hi_prop * 100.0,
        "pi_lo_pp": pi_lo_prop * 100.0,
        "pi_hi_pp": pi_hi_prop * 100.0,
    }


def main() -> int:
    global RESULTS, COMPARATORS, CSV_OUT
    args = parse_args()
    RESULTS = args.repo_root / "results_stats"
    COMPARATORS = [
        ("zbase_univ", RESULTS / "heterogeneity" / "meta_analysis_summary.json"),
        ("alertcount", RESULTS / "heterogeneity_alertcount" / "meta_analysis_summary.json"),
    ]
    CSV_OUT = args.output_csv or (RESULTS / "verification_table.csv")

    rows = []
    all_pass = True
    print(f"{'comparator':<14} {'statistic':<20} {'json_value':>16} {'recomputed':>16} {'abs_diff':>14} status")
    print("-" * 90)
    for comparator, json_path in COMPARATORS:
        with open(json_path) as f:
            d = json.load(f)
        eff_pp = np.array(d["effects_pp"], dtype=float)
        var_prop2 = np.array(d["variances"], dtype=float)
        meta_pp = d["meta_pp"]

        recomputed = recompute(eff_pp, var_prop2)

        for stat in STATS:
            json_val = float(meta_pp[stat])
            recomp_val = float(recomputed[stat])
            abs_diff = abs(json_val - recomp_val)
            ok = abs_diff < TOL
            if not ok:
                all_pass = False
            status = "PASS" if ok else "FAIL"
            print(
                f"{comparator:<14} {stat:<20} {json_val:>16.6f} {recomp_val:>16.6f} "
                f"{abs_diff:>14.3e} {status}"
            )
            rows.append(
                {
                    "comparator": comparator,
                    "statistic": stat,
                    "json_value": json_val,
                    "recomputed_value": recomp_val,
                    "abs_diff": abs_diff,
                    "status": status,
                }
            )

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["comparator", "statistic", "json_value", "recomputed_value", "abs_diff", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    max_diff = max(r["abs_diff"] for r in rows)
    print("-" * 90)
    print(
        f"SUMMARY: verified={n_total} pass={n_total - n_fail} fail={n_fail} "
        f"max_abs_diff={max_diff:.3e} csv={CSV_OUT}"
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
