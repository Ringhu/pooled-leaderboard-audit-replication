#!/usr/bin/env python3
"""P2 confirmatory interaction tests for the 4-method CD-1min universe.

Tests whether method performance depends on either:

  - subsystem identity: correct ~ C(method) * C(system_id)
  - benchmark-family identity: correct ~ C(method) * C(benchmark)

For each grouping, the interaction model is compared against the additive
model by likelihood-ratio test. Cluster-robust SEs by case_id are fit for model
coefficient summaries, while the LRT uses model log-likelihoods, matching the
existing interaction test scripts in this project.
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


ROOT = Path("$HOME/mnt/3090/auditstack")
DEFAULT_INPUT = ROOT / "results_stats/multiverse_cd1min_main/full_coverage_cd1min_long.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results_stats/p2_cd1min_confirmatory_interaction"

METHODS = ["baro", "z_family", "alertcount", "cd1min"]
DISPLAY_NAMES = {
    "baro": "BARO",
    "z_family": "max-|Z|",
    "alertcount": "alert-count",
    "cd1min": "CD-1min",
}


def fit_glm(df: pd.DataFrame, formula: str):
    return smf.glm(
        formula,
        data=df,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})


def fit_null(df: pd.DataFrame) -> float:
    return float(fit_glm(df, "correct ~ 1").llf)


def mcfadden_r2(llf: float, llf_null: float) -> float:
    return float(1.0 - llf / llf_null)


def lrt(ll_big: float, ll_small: float, df_diff: int) -> tuple[float, float]:
    stat = 2.0 * (ll_big - ll_small)
    p = float(1.0 - chi2.cdf(stat, df_diff)) if df_diff > 0 else float("nan")
    return float(stat), p


def coverage(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return df.groupby(["method", group_col]).size().unstack(fill_value=0)


def interaction_test(
    df: pd.DataFrame,
    methods: list[str],
    group_col: str,
    label: str,
) -> dict:
    sub = df[df["method"].isin(methods)].copy()
    sub["correct"] = sub["correct"].clip(0.0, 1.0).astype(float)
    cov = coverage(sub, group_col)
    if cov.shape[0] != len(methods) or not (cov.values > 0).all():
        raise RuntimeError(f"{label}: incomplete coverage for {methods} x {group_col}\n{cov}")

    group_expr = f"C({group_col})"
    full_formula = f"correct ~ C(method) * {group_expr}"
    reduced_formula = f"correct ~ C(method) + {group_expr}"

    warn_log: list[str] = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        full = fit_glm(sub, full_formula)
        reduced = fit_glm(sub, reduced_formula)
        for w in wlist:
            warn_log.append(f"{w.category.__name__}: {w.message}")
    warn_log.append(f"full converged={bool(getattr(full, 'converged', True))}")
    warn_log.append(f"reduced converged={bool(getattr(reduced, 'converged', True))}")

    llf_full = float(full.llf)
    llf_reduced = float(reduced.llf)
    df_full = int(len(full.params))
    df_reduced = int(len(reduced.params))
    df_diff = df_full - df_reduced
    stat, p_value = lrt(llf_full, llf_reduced, df_diff)
    llf_null = fit_null(sub)

    return {
        "label": label,
        "group_col": group_col,
        "methods": methods,
        "method_names": [DISPLAY_NAMES.get(m, m) for m in methods],
        "n_methods": len(methods),
        "n_groups": int(sub[group_col].nunique()),
        "n_obs": int(len(sub)),
        "n_cases": int(sub["case_id"].nunique()),
        "coverage_matrix": cov.to_dict(),
        "full_formula": full_formula,
        "reduced_formula": reduced_formula,
        "llf_with_interaction": llf_full,
        "llf_without_interaction": llf_reduced,
        "df_full_model_params": df_full,
        "df_reduced_model_params": df_reduced,
        "df_interaction_terms": int(df_diff),
        "lrt_statistic": stat,
        "p_interaction": p_value,
        "mcfadden_r2_full": mcfadden_r2(llf_full, llf_null),
        "mcfadden_r2_reduced": mcfadden_r2(llf_reduced, llf_null),
        "interpretation": "PASS (p<0.05)" if p_value < 0.05 else "FAIL (p>=0.05)",
        "warnings": warn_log,
    }


def row_from_result(res: dict) -> dict:
    return {
        "label": res["label"],
        "group_col": res["group_col"],
        "methods": "+".join(res["methods"]),
        "method_names": " vs ".join(res["method_names"]),
        "n_methods": res["n_methods"],
        "n_groups": res["n_groups"],
        "n_obs": res["n_obs"],
        "n_cases": res["n_cases"],
        "lrt_statistic": res["lrt_statistic"],
        "df_interaction_terms": res["df_interaction_terms"],
        "p_interaction": res["p_interaction"],
        "mcfadden_r2_full": res["mcfadden_r2_full"],
        "mcfadden_r2_reduced": res["mcfadden_r2_reduced"],
        "interpretation": res["interpretation"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-long", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_long)
    missing = sorted(set(METHODS) - set(df["method"].unique()))
    if missing:
        raise RuntimeError(f"Input missing methods: {missing}")

    results: dict[str, dict] = {}
    rows = []

    for group_col in ["system_id", "benchmark"]:
        key = f"omnibus_{group_col}"
        res = interaction_test(df, METHODS, group_col, key)
        results[key] = res
        rows.append(row_from_result(res))

        for a, b in itertools.combinations(METHODS, 2):
            key = f"{a}_vs_{b}_{group_col}"
            res = interaction_test(df, [a, b], group_col, key)
            results[key] = res
            rows.append(row_from_result(res))

    out_csv = args.output_dir / "p2_interaction_tests.csv"
    out_json = args.output_dir / "p2_interaction_tests.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_long": str(args.input_long),
                "model_spec": (
                    "GLM-Binomial; full correct ~ C(method)*C(group), "
                    "reduced correct ~ C(method)+C(group); cluster SE by case_id"
                ),
                "results": results,
            },
            f,
            indent=2,
            default=float,
        )

    print(f"[written] {out_csv}")
    print(f"[written] {out_json}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
