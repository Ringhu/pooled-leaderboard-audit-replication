#!/usr/bin/env python3
"""Clustered logistic regression with method x system interaction test —
UNIFIED version (R1 correction).

Differs from interaction_test.py by:
  * reading from results_xbench_11sys_unified/_inputs/ (the R1 inputs that
    replaced OpenRCA's legacy zbase with the universal zbase_univ
    predictor, eliminating the z_family collapse).
  * restricting the block to {BARO, zbase_univ} (not {baro, z_family}).

Everything else — GLM-Binomial fractional logit with cluster SE by
case_id, LRT on interaction terms, McFadden R^2, main-effect Wald tests,
full coefficient table — is identical to interaction_test.py.

Output: results_stats/interaction_test_unified.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

ROOT = Path("$HOME/auditstack")
UNIFIED_INPUTS = ROOT / "results_xbench_11sys_unified" / "_inputs"
OUT_JSON = ROOT / "results_stats" / "interaction_test_unified.json"

# (benchmark, system, file, acc_col)
SYSTEM_FILES = [
    ("openrca", "bank",     UNIFIED_INPUTS / "openrca_bank_results_v2.csv",  "acc1_v2"),
    ("openrca", "mkt1",     UNIFIED_INPUTS / "openrca_mkt1_results_v2.csv",  "acc1_v2"),
    ("openrca", "mkt2",     UNIFIED_INPUTS / "openrca_mkt2_results_v2.csv",  "acc1_v2"),
    ("openrca", "tel",      UNIFIED_INPUTS / "openrca_tel_results_v2.csv",   "acc1_v2"),
    ("rcaeval", "OB",       UNIFIED_INPUTS / "rcaeval_merged_online-boutique_results.csv", "acc1_v2"),
    ("rcaeval", "SS",       UNIFIED_INPUTS / "rcaeval_merged_sock-shop-2_results.csv",     "acc1_v2"),
    ("rcaeval", "TT",       UNIFIED_INPUTS / "rcaeval_merged_train-ticket_results.csv",    "acc1_v2"),
    ("petshop", "high_traffic",      UNIFIED_INPUTS / "petshop_merged_high_traffic_results.csv",      "acc@1"),
    ("petshop", "low_traffic",       UNIFIED_INPUTS / "petshop_merged_low_traffic_results.csv",       "acc@1"),
    ("petshop", "temporal_traffic1", UNIFIED_INPUTS / "petshop_merged_temporal_traffic1_results.csv", "acc@1"),
    ("petshop", "temporal_traffic2", UNIFIED_INPUTS / "petshop_merged_temporal_traffic2_results.csv", "acc@1"),
]


def load_all_systems_unified() -> pd.DataFrame:
    """Like bootstrap_pairwise.load_all_systems but reads the unified block
    and does NOT collapse zbase/zbase_univ into a synthetic z_family label
    (they are all zbase_univ after R1)."""
    frames: List[pd.DataFrame] = []
    for benchmark, system, path, acc_col in SYSTEM_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")
        df = pd.read_csv(path)
        if "tdelta" in df.columns:
            df = df[df["tdelta"] == 0].copy()
        if acc_col not in df.columns:
            raise KeyError(f"{path} missing expected acc column {acc_col}")
        if "case" not in df.columns:
            raise KeyError(f"{path} missing 'case' column")
        if "method" not in df.columns:
            raise KeyError(f"{path} missing 'method' column")
        df = df.dropna(subset=[acc_col, "method", "case"]).copy()
        df["q_idx"] = df.groupby(["method", "case"]).cumcount()
        df["method_raw"] = df["method"].astype(str)
        df["benchmark"] = benchmark
        df["system"] = system
        df["system_id"] = f"{benchmark}::{system}"
        df["case_id"] = (
            df["system_id"] + "|" + df["case"].astype(str)
            + "#" + df["q_idx"].astype(str)
        )
        df["correct"] = df[acc_col].astype(float)
        keep = ["benchmark", "system", "system_id", "method", "method_raw",
                "case", "q_idx", "case_id", "correct"]
        frames.append(df[keep])
    return pd.concat(frames, ignore_index=True)


def _mcfadden_r2(llf_full: float, llf_null: float) -> float:
    return float(1.0 - llf_full / llf_null)


def _fit_null(df: pd.DataFrame) -> float:
    m = smf.glm(
        "correct ~ 1", data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})
    return float(m.llf)


def _fit_glm(df: pd.DataFrame, formula: str):
    return smf.glm(
        formula, data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})


def main():
    df = load_all_systems_unified()
    print(f"Methods found: {sorted(df['method'].unique())}")
    # Restrict to the BARO + zbase_univ complete 2 x 11 block.
    sub = df[df["method"].isin(["baro", "zbase_univ"])].copy()
    sub["correct"] = sub["correct"].clip(0.0, 1.0).astype(float)

    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    print("Coverage matrix (obs counts), method x system:")
    print(cov.to_string())

    assert cov.shape == (2, 11), f"Expected 2x11 block, got {cov.shape}"
    assert (cov.values > 0).all(), "Some cells are empty; interaction not estimable."

    n_obs = int(len(sub))
    n_cases = int(sub["case_id"].nunique())
    n_systems = int(sub["system_id"].nunique())
    print(f"\nn_obs={n_obs}  n_cases={n_cases}  n_systems={n_systems}")

    convergence_warnings = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model_full = _fit_glm(sub, "correct ~ C(method) * C(system_id)")
        model_reduced = _fit_glm(sub, "correct ~ C(method) + C(system_id)")
        for w in wlist:
            convergence_warnings.append(f"{w.category.__name__}: {w.message}")
    convergence_warnings.append(
        f"full converged={bool(getattr(model_full, 'converged', True))}"
    )
    convergence_warnings.append(
        f"reduced converged={bool(getattr(model_reduced, 'converged', True))}"
    )

    llf_full = float(model_full.llf)
    llf_reduced = float(model_reduced.llf)
    df_full = int(len(model_full.params))
    df_reduced = int(len(model_reduced.params))
    df_diff = df_full - df_reduced
    lrt = 2.0 * (llf_full - llf_reduced)
    p_interaction = float(1.0 - chi2.cdf(lrt, df_diff))

    llf_null = _fit_null(sub)
    r2_full = _mcfadden_r2(llf_full, llf_null)
    r2_reduced = _mcfadden_r2(llf_reduced, llf_null)

    def _wald_for_factor(model, prefix: str) -> float:
        names = [n for n in model.params.index if n.startswith(prefix)]
        if not names:
            return float("nan")
        t = model.wald_test(names, scalar=True)
        return float(t.pvalue)

    p_method_main = _wald_for_factor(model_reduced, "C(method)")
    p_system_main = _wald_for_factor(model_reduced, "C(system_id)")
    p_interaction_wald = _wald_for_factor(
        model_full, "C(method)[T.zbase_univ]:C(system_id)"
    )

    coef_rows = []
    params = model_full.params
    bse = model_full.bse
    zvals = params / bse
    pvals = model_full.pvalues
    for name in params.index:
        coef_rows.append({
            "term": str(name),
            "coef": float(params[name]),
            "std_err": float(bse[name]),
            "z": float(zvals[name]),
            "p_value": float(pvals[name]),
        })

    out = {
        "model_spec": (
            "GLM-Binomial (logit link, IRLS) "
            "correct ~ C(method) * C(system_id), "
            "cluster SE by case_id, "
            "BARO + zbase_univ only (complete 2x11 block under R1). "
            "Fractional logit (Papke-Wooldridge) "
            "for OpenRCA's partial-credit acc1_v2."
        ),
        "inputs_dir": str(UNIFIED_INPUTS),
        "method_labels_used": ["baro", "zbase_univ"],
        "n_obs": n_obs,
        "n_cases": n_cases,
        "n_systems": n_systems,
        "coverage_matrix": cov.to_dict(),
        "llf_with_interaction": llf_full,
        "llf_without_interaction": llf_reduced,
        "df_full_model_params": df_full,
        "df_reduced_model_params": df_reduced,
        "lrt_statistic": float(lrt),
        "df_interaction_terms": int(df_diff),
        "p_interaction": p_interaction,
        "p_interaction_wald_on_full_model": p_interaction_wald,
        "mcfadden_r2_full": r2_full,
        "mcfadden_r2_reduced": r2_reduced,
        "main_effect_method_p_reduced_model": p_method_main,
        "main_effect_system_p_reduced_model": p_system_main,
        "coef_table_full_model": coef_rows,
        "convergence_warnings": convergence_warnings,
        "previous_z_family_p_for_reference": 8.63e-4,
        "interpretation": (
            "PASS G1 gate (p<0.05)" if p_interaction < 0.05
            else "FAIL G1 gate (p>=0.05)"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {OUT_JSON}")

    print("\n=== Interaction test (UNIFIED) ===")
    print(f"  n_obs       = {n_obs}")
    print(f"  n_cases     = {n_cases}")
    print(f"  llf full    = {llf_full:.4f}   ({df_full} params)")
    print(f"  llf reduced = {llf_reduced:.4f}   ({df_reduced} params)")
    print(f"  LRT stat    = {lrt:.4f}   df_diff = {df_diff}")
    print(f"  p_interact  = {p_interaction:.3e}")
    print(f"  prev z_family ref = 8.63e-04")
    print(f"  McFadden R2 full    = {r2_full:.4f}")
    print(f"  McFadden R2 reduced = {r2_reduced:.4f}")
    print(f"  Main effect method p = {p_method_main:.3e}")
    print(f"  Main effect system p = {p_system_main:.3e}")
    print(f"  Interpretation: {out['interpretation']}")
    if convergence_warnings:
        print("\nWarnings during fit:")
        for w in convergence_warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
