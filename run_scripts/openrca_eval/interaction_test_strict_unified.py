#!/usr/bin/env python3
"""Strict 0/1 logit sensitivity analysis — UNIFIED version (R1 correction).

Same four approaches as interaction_test_strict.py (fractional reference,
Firth, L2, drop-quasi-sep), but reading from the unified inputs and using
{baro, zbase_univ} instead of {baro, z_family}.

Output: results_stats/interaction_test_strict_unified.json
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

# Pull the unified loader + SYSTEM_FILES from interaction_test_unified.
from interaction_test_unified import load_all_systems_unified

ROOT = Path("$HOME/auditstack")
UNIFIED_INPUTS = ROOT / "results_xbench_11sys_unified" / "_inputs"
OUT_JSON = ROOT / "results_stats" / "interaction_test_strict_unified.json"

# Paper's previous fractional reference (pre-R1, z_family) for sanity ordering.
FRACTIONAL_P_REFERENCE_PREVIOUS = 8.63e-4


def prepare_block() -> pd.DataFrame:
    df = load_all_systems_unified()
    sub = df[df["method"].isin(["baro", "zbase_univ"])].copy()
    sub["correct_frac"] = sub["correct"].clip(0.0, 1.0).astype(float)
    sub["correct"] = (sub["correct_frac"] > 0.5).astype(int)
    return sub


def _coverage_and_rates(sub: pd.DataFrame):
    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    rate = sub.groupby(["method", "system_id"])["correct"].mean().unstack(fill_value=np.nan)
    return cov, rate


def _sanitize(name: str) -> str:
    return str(name).strip()


def _design_matrices(sub: pd.DataFrame):
    from patsy import dmatrices
    y_full, X_full = dmatrices(
        "correct ~ C(method) * C(system_id)", data=sub, return_type="dataframe"
    )
    y_red, X_red = dmatrices(
        "correct ~ C(method) + C(system_id)", data=sub, return_type="dataframe"
    )
    assert np.array_equal(y_full.values.ravel(), y_red.values.ravel())
    n_interaction_params = X_full.shape[1] - X_red.shape[1]
    return y_full.values.ravel().astype(int), X_full, X_red, int(n_interaction_params)


def approach_firth(sub: pd.DataFrame) -> dict:
    try:
        from pyfirth.PyFirth import PyFirth
    except Exception as e:
        return {"p_interaction": None, "status": "failed_install",
                "notes": f"pyfirth import failed: {e!r}"}
    y, X_full, X_red, k_interaction = _design_matrices(sub)
    data = X_full.copy()
    data.columns = [_sanitize(c) for c in data.columns]
    y_col = "_y_"
    data[y_col] = y
    x_cols = [c for c in data.columns if c != y_col]
    red_cols = {_sanitize(c) for c in X_red.columns}
    interaction_cols = [c for c in x_cols if c not in red_cols]
    assert len(interaction_cols) == k_interaction

    notes = []
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            firth = PyFirth(data, x_cols, y_col, hasconst=True)
            result = firth.fit(
                interaction_cols,
                num_iters=2000, step_limit=200,
                max_step=5.0, convergence_limit=1e-6,
            )
            for w in wlist:
                notes.append(f"{w.category.__name__}: {w.message}")
    except Exception as e:
        return {"p_interaction": None, "status": "failed_converge",
                "notes": f"pyfirth fit raised: {e!r}"}
    p_val = result.get("PVal")
    flag = result.get("Flag", 0)
    if p_val is None or (isinstance(p_val, float) and (np.isnan(p_val) or not np.isfinite(p_val))):
        return {"p_interaction": None, "status": "failed_converge",
                "notes": f"pyfirth returned non-finite PVal; flag={flag}"}
    return {
        "p_interaction": float(p_val),
        "status": "success",
        "k_interaction_params": int(k_interaction),
        "firth_flag": int(flag),
        "notes": (
            f"Firth penalized LRT over {k_interaction} interaction dummies; "
            f"flag={flag} (0=OK). " + "; ".join(notes[:5])
        ).strip("; "),
    }


def approach_l2(sub: pd.DataFrame, C: float = 1.0) -> dict:
    from sklearn.linear_model import LogisticRegression
    y, X_full, X_red, k_interaction = _design_matrices(sub)

    def _drop_intercept(X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in X.columns if c.lower() != "intercept"]
        return X[cols].values
    Xf = _drop_intercept(X_full)
    Xr = _drop_intercept(X_red)
    notes = []

    def _fit_loglik(X: np.ndarray):
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            clf = LogisticRegression(
                penalty="l2", C=C, solver="lbfgs",
                max_iter=5000, fit_intercept=True, tol=1e-8,
            )
            clf.fit(X, y)
            converged = True
            for w in wlist:
                msg = f"{w.category.__name__}: {w.message}"
                notes.append(msg)
                if "converge" in str(w.message).lower():
                    converged = False
        proba = clf.predict_proba(X)[:, 1]
        eps = 1e-12
        proba = np.clip(proba, eps, 1 - eps)
        ll = float(np.sum(np.log(proba[y == 1])) + np.sum(np.log(1.0 - proba[y == 0])))
        return ll, converged

    ll_full, conv_full = _fit_loglik(Xf)
    ll_red, conv_red = _fit_loglik(Xr)
    lrt = 2.0 * (ll_full - ll_red)
    if lrt < 0:
        notes.append(f"LRT negative ({lrt:.4f}); clipping to 0.")
        lrt_clip = 0.0
    else:
        lrt_clip = lrt
    p = float(1.0 - chi2.cdf(lrt_clip, df=k_interaction))
    return {
        "p_interaction": p,
        "status": "success" if (conv_full and conv_red) else "success_with_warnings",
        "C": float(C),
        "lrt_stat": float(lrt),
        "df": int(k_interaction),
        "ll_full": ll_full,
        "ll_reduced": ll_red,
        "notes": (
            f"sklearn L2 logit (C={C}); LRT on unpenalized LLF from predicted probs. "
            f"Full converged={conv_full}, reduced converged={conv_red}. "
            + "; ".join(notes[:5])
        ).strip("; "),
    }


def approach_drop_cell(sub: pd.DataFrame, sep_cell: dict) -> dict:
    """Drop the single most quasi-separating cell and refit strict logit."""
    filt = ~((sub["system_id"] == sep_cell["system_id"]) &
             (sub["method"] == sep_cell["method"]))
    sub2 = sub[filt].copy()
    n_obs = int(len(sub2))

    convergence_warnings = []
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            m_full = smf.glm(
                "correct ~ C(method) * C(system_id)",
                data=sub2, family=sm.families.Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": sub2["case_id"]})
            m_red = smf.glm(
                "correct ~ C(method) + C(system_id)",
                data=sub2, family=sm.families.Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": sub2["case_id"]})
            for w in wlist:
                convergence_warnings.append(f"{w.category.__name__}: {w.message}")
        conv_full = bool(getattr(m_full, "converged", True))
        conv_red = bool(getattr(m_red, "converged", True))
        llf_full = float(m_full.llf)
        llf_red = float(m_red.llf)
        df_full = int(len(m_full.params))
        df_red = int(len(m_red.params))
        df_diff = df_full - df_red
        lrt = 2.0 * (llf_full - llf_red)
        p = float(1.0 - chi2.cdf(lrt, df=df_diff))
    except Exception as e:
        return {"p_interaction": None, "status": "failed_converge",
                "n_obs": n_obs, "notes": f"statsmodels GLM raised: {e!r}"}
    status = "success" if (conv_full and conv_red) else "success_with_warnings"
    return {
        "p_interaction": p,
        "status": status,
        "dropped_cell": sep_cell,
        "n_obs": n_obs,
        "lrt_stat": float(lrt),
        "df": int(df_diff),
        "llf_full": llf_full,
        "llf_reduced": llf_red,
        "full_converged": conv_full,
        "reduced_converged": conv_red,
        "notes": (
            f"Dropped quasi-separating cell ({sep_cell}); "
            "statsmodels GLM-Binomial logit, cluster SE by case_id. "
            + "; ".join(convergence_warnings[:5])
        ).strip("; "),
    }


def main():
    print("[load] reading UNIFIED inputs ...")
    sub = prepare_block()
    cov, rate = _coverage_and_rates(sub)
    n_obs = int(len(sub))
    n_cases = int(sub["case_id"].nunique())

    print("Coverage matrix (obs counts), method x system:")
    print(cov.to_string())
    print("\nStrict 0/1 accuracy rate per cell:")
    print(rate.round(4).to_string())

    sep_cells = []
    for m in rate.index:
        for s in rate.columns:
            r = rate.loc[m, s]
            if np.isclose(r, 0.0) or np.isclose(r, 1.0):
                sep_cells.append({"method": m, "system_id": s, "rate": float(r)})
    print(f"\nQuasi-separating cells (rate in {{0,1}}): {sep_cells}")
    print(f"\nn_obs={n_obs}  n_cases={n_cases}")

    print("\n[A] Firth penalized logit (pyfirth) ...")
    a_result = approach_firth(sub)
    print(f"    -> {a_result}")
    print("\n[B] L2-regularized logit (sklearn, C=1.0) ...")
    b_result = approach_l2(sub, C=1.0)
    print(f"    -> {b_result}")

    # Pick the first quasi-separating cell (if any) for the drop approach.
    # Prefer the sep cell with rate=0.0 (more conservative drop).
    c_result = None
    if sep_cells:
        sep_cells_sorted = sorted(sep_cells, key=lambda c: (c["rate"], c["system_id"]))
        drop_target = sep_cells_sorted[0]
        print(f"\n[C] Drop quasi-separating cell {drop_target} + unreg GLM ...")
        c_result = approach_drop_cell(sub, drop_target)
    else:
        print("\n[C] No quasi-separating cells found; refitting full strict GLM.")
        # No drop needed -> fit strict GLM directly on sub.
        try:
            m_full = smf.glm(
                "correct ~ C(method) * C(system_id)", data=sub,
                family=sm.families.Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["case_id"]})
            m_red = smf.glm(
                "correct ~ C(method) + C(system_id)", data=sub,
                family=sm.families.Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["case_id"]})
            llf_full = float(m_full.llf)
            llf_red = float(m_red.llf)
            df_full = int(len(m_full.params))
            df_red = int(len(m_red.params))
            df_diff = df_full - df_red
            lrt = 2.0 * (llf_full - llf_red)
            p = float(1.0 - chi2.cdf(lrt, df=df_diff))
            c_result = {
                "p_interaction": p, "status": "success",
                "dropped_cell": None,
                "n_obs": int(len(sub)),
                "lrt_stat": float(lrt), "df": int(df_diff),
                "llf_full": llf_full, "llf_reduced": llf_red,
                "notes": "No quasi-separating cells; full strict GLM fit.",
            }
        except Exception as e:
            c_result = {"p_interaction": None, "status": "failed_converge",
                        "notes": f"{e!r}"}
    print(f"    -> {c_result}")

    p_values = {}
    if a_result.get("p_interaction") is not None:
        p_values["firth"] = a_result["p_interaction"]
    if b_result.get("p_interaction") is not None:
        p_values["l2_regularized"] = b_result["p_interaction"]
    if c_result.get("p_interaction") is not None:
        p_values["drop_cell"] = c_result["p_interaction"]

    all_below_05 = all(p < 0.05 for p in p_values.values()) if p_values else False
    max_p = max(p_values.values()) if p_values else None
    min_p = min(p_values.values()) if p_values else None

    if not p_values:
        conclusion = ("AMBIGUOUS — all three strict approaches failed; "
                      "fall back to fractional logit.")
    elif all_below_05:
        orders = {
            k: int(np.floor(np.log10(max(v, 1e-300))))
            for k, v in p_values.items()
        }
        ref_order = int(np.floor(np.log10(FRACTIONAL_P_REFERENCE_PREVIOUS)))
        close = all(abs(o - ref_order) <= 2 for o in orders.values())
        if close:
            conclusion = (
                f"ROBUST — all strict approaches give p<0.05 (max p={max_p:.3e}) "
                f"within 2 orders of magnitude of previous z_family reference "
                f"({FRACTIONAL_P_REFERENCE_PREVIOUS:.3e})."
            )
        else:
            conclusion = (
                f"ROBUST (attenuated) — all p<0.05 but span "
                f"{min_p:.2e} to {max_p:.2e}."
            )
    else:
        conclusion = (
            f"SENSITIVE — at least one strict approach gives p>=0.05 "
            f"(max p={max_p:.3e}); result sensitive to strict 0/1 coding."
        )

    out = {
        "model_spec_reference_previous_fractional": (
            f"Previous z_family fractional logit, p={FRACTIONAL_P_REFERENCE_PREVIOUS:.3e} "
            "(pre-R1)."
        ),
        "binarization": "correct_strict = int(correct_fractional > 0.5)",
        "inputs_dir": str(UNIFIED_INPUTS),
        "method_labels_used": ["baro", "zbase_univ"],
        "n_obs": n_obs,
        "n_cases": n_cases,
        "coverage_matrix": cov.to_dict(),
        "rate_matrix_strict": rate.round(6).to_dict(),
        "quasi_separating_cells": sep_cells,
        "strict_logit_approaches": {
            "firth": a_result,
            "l2_regularized": b_result,
            "drop_cell": c_result,
        },
        "p_interaction_summary": {
            "previous_z_family_reference": FRACTIONAL_P_REFERENCE_PREVIOUS,
            **p_values,
        },
        "conclusion": conclusion,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {OUT_JSON}")

    print("\n=== Strict logit sensitivity summary (UNIFIED) ===")
    print(f"  Prev z_family ref (fractional):  p = {FRACTIONAL_P_REFERENCE_PREVIOUS:.3e}")
    for k in ("firth", "l2_regularized", "drop_cell"):
        res = out["strict_logit_approaches"][k]
        p = res.get("p_interaction")
        status = res.get("status")
        if p is None:
            print(f"  [{k:>15}]  p = n/a         (status: {status})")
        else:
            print(f"  [{k:>15}]  p = {p:.3e}   (status: {status})")
    print(f"\nConclusion: {conclusion}")


if __name__ == "__main__":
    main()
