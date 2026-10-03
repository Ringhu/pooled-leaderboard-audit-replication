#!/usr/bin/env python3
"""Interaction test with per-system difficulty covariates.

Extension of `interaction_test.py`. Reviewer's "weak-accept" concern: the 12.7x
cross-system BARO spread might be explained by per-system difficulty
(class imbalance, service-count variation) rather than genuine system identity
effects. We fit three nested GLM-Binomial models (cluster SE by case_id on the
complete 2 x 11 block for {BARO, z_family}):

    X (reduced):                 correct ~ C(method) + majority_baseline + root_entropy
    Y (difficulty x method):     correct ~ C(method) * (majority_baseline + root_entropy)
    Z (full system interaction): correct ~ C(method) * C(system_id)   [== main-track model]

LRTs
  * Y vs X : does difficulty (interacted with method) explain any method differences?
  * Z vs Y : does residual system heterogeneity remain AFTER controlling for
             difficulty? THIS is the paper's main defensive claim.

PASS condition:  p(Z vs Y) < 0.05 => difficulty cannot absorb the interaction;
                  system identity carries explanatory power beyond difficulty.

Output: results_stats/interaction_test_difficulty.json
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

from bootstrap_pairwise import load_all_systems
from difficulty_indicators import compute_all_systems

ROOT = Path("$HOME/auditstack")
OUT_JSON = ROOT / "results_stats" / "interaction_test_difficulty.json"


SPEC_X = "correct ~ C(method) + majority_baseline + root_entropy"
SPEC_Y = "correct ~ C(method) * (majority_baseline + root_entropy)"
SPEC_Z = "correct ~ C(method) * C(system_id)"


def _fit(df: pd.DataFrame, formula: str):
    """GLM-Binomial (logit), cluster SE by case_id. Mirrors interaction_test.py."""
    return smf.glm(
        formula, data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})


def _lrt(ll_big: float, ll_small: float, df_diff: int) -> dict:
    stat = 2.0 * (ll_big - ll_small)
    p = float(1.0 - chi2.cdf(stat, df_diff)) if df_diff > 0 else float("nan")
    return {"stat": float(stat), "df": int(df_diff), "p": p}


def _system_key(benchmark: str, system: str) -> str:
    """Match bootstrap_pairwise.load_all_systems system_id convention."""
    # bootstrap_pairwise uses labels: openrca::{bank,mkt1,mkt2,tel},
    # rcaeval::{OB,SS,TT}, petshop::{...}. difficulty_indicators emits the
    # same short labels.
    return f"{benchmark}::{system}"


def main():
    # -------- Load observation-level data (BARO + z_family 2x11 block) -------- #
    df = load_all_systems()
    sub = df[df["method"].isin(["baro", "z_family"])].copy()
    sub["correct"] = sub["correct"].clip(0.0, 1.0).astype(float)

    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    assert cov.shape == (2, 11), f"Expected 2x11, got {cov.shape}"
    assert (cov.values > 0).all(), "Some cells empty"

    # -------- Join per-system difficulty indicators -------- #
    diff_table = compute_all_systems()
    diff_table["system_id"] = diff_table.apply(
        lambda r: _system_key(r["benchmark"], r["system"]), axis=1
    )
    cols = ["system_id", "majority_baseline", "root_entropy",
            "service_count", "n_cases"]
    merged = sub.merge(diff_table[cols], on="system_id", how="left")
    assert merged["majority_baseline"].notna().all(), (
        "Some rows missing difficulty covariates — check system_id mapping"
    )

    n_obs = int(len(merged))
    n_cases = int(merged["case_id"].nunique())
    n_systems = int(merged["system_id"].nunique())
    print(f"n_obs={n_obs}  n_cases={n_cases}  n_systems={n_systems}")
    print("Coverage (method x system):")
    print(cov.to_string())

    # -------- Fit 3 nested models -------- #
    warn_log = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model_X = _fit(merged, SPEC_X)
        model_Y = _fit(merged, SPEC_Y)
        model_Z = _fit(merged, SPEC_Z)
        for w in wlist:
            warn_log.append(f"{w.category.__name__}: {w.message}")
    for name, m in [("X", model_X), ("Y", model_Y), ("Z", model_Z)]:
        warn_log.append(f"{name} converged={bool(getattr(m, 'converged', True))}")

    llf_X = float(model_X.llf)
    llf_Y = float(model_Y.llf)
    llf_Z = float(model_Z.llf)
    n_params_X = int(len(model_X.params))
    n_params_Y = int(len(model_Y.params))
    n_params_Z = int(len(model_Z.params))

    print(f"\nllf_X = {llf_X:.4f}  (params={n_params_X})")
    print(f"llf_Y = {llf_Y:.4f}  (params={n_params_Y})")
    print(f"llf_Z = {llf_Z:.4f}  (params={n_params_Z})")

    # Nesting check: Y nests X (adds interactions of same covariates).
    # Z does NOT structurally nest Y (Y uses continuous difficulty covariates,
    # Z uses system-id dummies). However, any continuous-covariate model of
    # system effects is a strict special case of the system-id dummy model
    # when there is >1 system: Z can reproduce any per-system intercept + any
    # per-(system x method) deviation, so Y is a constrained version of Z in
    # the full-rank sense (d.o.f. saved by Y = #system-related params in Z -
    # #system-related params in Y). We compute df_diff from the parameter
    # count, which is the standard practice for non-strictly-nested but
    # nonetheless rank-comparable GLMs, and flag the caveat.
    lrt_Y_vs_X = _lrt(llf_Y, llf_X, n_params_Y - n_params_X)
    lrt_Z_vs_Y = _lrt(llf_Z, llf_Y, n_params_Z - n_params_Y)

    print(f"\nLRT Y vs X: stat={lrt_Y_vs_X['stat']:.4f}  "
          f"df={lrt_Y_vs_X['df']}  p={lrt_Y_vs_X['p']:.3e}")
    print(f"LRT Z vs Y: stat={lrt_Z_vs_Y['stat']:.4f}  "
          f"df={lrt_Z_vs_Y['df']}  p={lrt_Z_vs_Y['p']:.3e}")

    # Guard against rank deficiency / degenerate df (e.g. if Z happened to
    # have fewer params than Y because of dummy absorption).
    z_vs_y_rank_ok = (n_params_Z - n_params_Y) > 0
    if not z_vs_y_rank_ok:
        print("[warn] Z has <= params than Y — LRT Z vs Y is degenerate / "
              "non-comparable. Reporting honestly.")

    # -------- Conclusion -------- #
    pass_condition = (
        z_vs_y_rank_ok
        and np.isfinite(lrt_Z_vs_Y["p"])
        and lrt_Z_vs_Y["p"] < 0.05
    )
    conclusion = (
        "PASS — difficulty covariates do NOT absorb the method x system "
        "interaction; residual system heterogeneity remains significant "
        "(p(Z vs Y) < 0.05)."
        if pass_condition else
        "FAIL — either difficulty absorbs the interaction or LRT(Z vs Y) is "
        "not significant at alpha=0.05; system identity does not add "
        "significant explanatory power beyond the difficulty covariates."
    )

    out = {
        "model_X_spec": SPEC_X,
        "model_Y_spec": SPEC_Y,
        "model_Z_spec": SPEC_Z + "   [== original main-track interaction test]",
        "n_obs": n_obs,
        "n_cases": n_cases,
        "n_systems": n_systems,
        "coverage_matrix_2x11": cov.to_dict(),
        "llf_X": llf_X,
        "llf_Y": llf_Y,
        "llf_Z": llf_Z,
        "n_params_X": n_params_X,
        "n_params_Y": n_params_Y,
        "n_params_Z": n_params_Z,
        "lrt_Y_vs_X": {
            **lrt_Y_vs_X,
            "interpretation": (
                "Does difficulty x method interaction explain any of the "
                "method-level differences? (p<0.05 => yes.)"
            ),
        },
        "lrt_Z_vs_Y": {
            **lrt_Z_vs_Y,
            "rank_ok": bool(z_vs_y_rank_ok),
            "interpretation": (
                "Residual system heterogeneity beyond difficulty covariates: "
                "p<0.05 means system identity matters even after controlling "
                "for majority_baseline + root_entropy and their interaction "
                "with method. THIS IS THE PAPER'S MAIN DEFENSIVE CLAIM."
            ),
            "nesting_caveat": (
                "Y (continuous difficulty) is not a strictly-formula-nested "
                "submodel of Z (per-system dummies), but any affine function "
                "of majority_baseline and root_entropy with method "
                "interactions is representable in Z's full dummy-based "
                "parameterization. LRT df is computed from parameter-count "
                "difference, which is the standard practice."
            ),
        },
        "conclusion": conclusion,
        "warnings": warn_log,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {OUT_JSON}")
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
