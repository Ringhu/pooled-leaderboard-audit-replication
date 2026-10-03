#!/usr/bin/env python3
"""Clustered logistic regression with method x system interaction test.

Reviewer's weak-accept condition: `method x system` interaction evidence for
the "System Matters" claim. We fit two logit models on the complete
{BARO, z_family} x 11 systems block, with SEs clustered by case_id, and
compare them with a likelihood-ratio test.

  correct ~ C(method) * C(system)   (full)
  correct ~ C(method) + C(system)   (reduced, no interaction)
  LRT = 2 * (llf_full - llf_reduced) ~ chi2(df_diff)

Notes
-----
* We intentionally use clustered SEs on a logit rather than a GLMM. statsmodels
  GLMM support is thin, and case-level clustered SEs + LRT on the difference
  in log-likelihood is robust and reviewer-defensible.
* case_id is globally unique (<system_id>|<case>#<q_idx>) so clustering is
  well-defined even though cases are nested in systems.

Output: results_stats/interaction_test.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

from bootstrap_pairwise import load_all_systems  # same directory

ROOT = Path("$HOME/auditstack")
OUT_JSON = ROOT / "results_stats" / "interaction_test.json"


def _mcfadden_r2(llf_full: float, llf_null: float) -> float:
    return float(1.0 - llf_full / llf_null)


def _fit_null(df: pd.DataFrame) -> float:
    """Intercept-only GLM-Binomial (logit link), clustered by case_id.
    Returns llf."""
    m = smf.glm(
        "correct ~ 1", data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})
    return float(m.llf)


def _fit_glm(df: pd.DataFrame, formula: str):
    """GLM Binomial (logit link), clustered SEs by case_id.

    We use GLM rather than `logit` because OpenRCA's `acc1_v2` contains
    fractional success rates (partial credit: 0.25, 0.33, 0.5, 0.67, 0.75)
    from multi-subtask tasks. GLM Binomial with IRLS handles fractional
    endogenous variables correctly (Papke-Wooldridge fractional logit),
    whereas `Logit` requires strict 0/1 and forced us to round, distorting
    accuracy. For rows that are already 0/1 the two are equivalent.
    """
    return smf.glm(
        formula, data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})


def main():
    df = load_all_systems()
    # Restrict to the BARO + z_family complete 2 x 11 block.
    sub = df[df["method"].isin(["baro", "z_family"])].copy()
    # Clip to [0,1] (defensive; OpenRCA acc1_v2 has legitimate fractional
    # values 0.25/0.33/0.5/0.67/0.75 representing partial credit — we keep
    # them and let GLM-Binomial handle fractional endog).
    sub["correct"] = sub["correct"].clip(0.0, 1.0).astype(float)

    # Coverage matrix (rows = method, cols = system_id) — must be dense 2x11.
    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    print("Coverage matrix (obs counts), method x system:")
    print(cov.to_string())

    assert cov.shape == (2, 11), f"Expected 2x11 block, got {cov.shape}"
    assert (cov.values > 0).all(), "Some cells are empty; interaction not estimable."

    n_obs = int(len(sub))
    n_cases = int(sub["case_id"].nunique())
    n_systems = int(sub["system_id"].nunique())
    print(f"\nn_obs={n_obs}  n_cases={n_cases}  n_systems={n_systems}")

    # -------- Fit the two models --------
    convergence_warnings = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model_full = _fit_glm(sub, "correct ~ C(method) * C(system_id)")
        model_reduced = _fit_glm(sub, "correct ~ C(method) + C(system_id)")
        for w in wlist:
            convergence_warnings.append(f"{w.category.__name__}: {w.message}")
    convergence_warnings.append(f"full converged={bool(getattr(model_full, 'converged', True))}")
    convergence_warnings.append(f"reduced converged={bool(getattr(model_reduced, 'converged', True))}")

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

    # Main-effect p-values from the REDUCED model (so that interaction terms
    # aren't soaking up the variance; reported "for context" per the task).
    # statsmodels' f_test/wald_test takes a set of constraint rows.
    def _wald_for_factor(model, prefix: str) -> float:
        names = [n for n in model.params.index if n.startswith(prefix)]
        if not names:
            return float("nan")
        t = model.wald_test(names, scalar=True)
        return float(t.pvalue)

    p_method_main = _wald_for_factor(model_reduced, "C(method)")
    p_system_main = _wald_for_factor(model_reduced, "C(system_id)")

    # Interaction-only Wald (complement to the LRT).
    p_interaction_wald = _wald_for_factor(model_full, "C(method)[T.z_family]:C(system_id)")

    # Full coefficient table.
    # statsmodels GLM summary2 uses different column names; use the params
    # directly.
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
        "model_spec": ("GLM-Binomial (logit link, IRLS) "
                       "correct ~ C(method) * C(system_id), "
                       "cluster SE by case_id, "
                       "BARO + z_family only (complete 2x11 block). "
                       "GLM-Binomial rather than Logit because OpenRCA's "
                       "acc1_v2 has fractional partial-credit values that "
                       "cannot be handled by strict Bernoulli logit."),
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
        "interpretation": "PASS G1 gate (p<0.05)" if p_interaction < 0.05 else "FAIL G1 gate (p>=0.05)",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {OUT_JSON}")

    print("\n=== Interaction test ===")
    print(f"  n_obs       = {n_obs}")
    print(f"  n_cases     = {n_cases}")
    print(f"  llf full    = {llf_full:.4f}   ({df_full} params)")
    print(f"  llf reduced = {llf_reduced:.4f}   ({df_reduced} params)")
    print(f"  LRT stat    = {lrt:.4f}   df_diff = {df_diff}")
    print(f"  p_interact  = {p_interaction:.3e}")
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
