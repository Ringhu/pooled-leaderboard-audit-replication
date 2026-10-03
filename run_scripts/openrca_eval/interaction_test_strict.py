#!/usr/bin/env python3
"""Strict 0/1 logit sensitivity analysis for method x system interaction.

This is a sanity check companion to interaction_test.py (GLM-Binomial
fractional logit, Papke-Wooldridge). Here we:

  1. Binarize `correct` as int(correct > 0.5) — the standard threshold
     turning OpenRCA's partial-credit values (0.25, 0.33, 0.5, 0.67, 0.75)
     into strict Bernoulli outcomes.
  2. Fit the same full / reduced logit and report an LRT on the
     interaction terms.

The catch: strict 0/1 logit on OpenRCA::mkt1 hits quasi-separation because
BARO on mkt1 has essentially 0% acc@1 → MLE fails to converge / coefficients
blow up. We therefore try three remedies in order, falling through until
one succeeds:

  A.  Firth's penalized logit  (gold standard)
        - uses pyfirth (pure-Python reference implementation).
        - inference via penalized-LL LRT over the block of interaction
          dummy coefficients.

  B.  L2-regularized logistic regression (sklearn)
        - mild ridge (C=1.0) biases point estimates slightly but keeps
          MLE finite under separation; LRT on fit log-likelihoods is an
          acceptable approximation.

  C.  Drop the single quasi-separating cell  (openrca::mkt1 / baro) and
        refit unregularized statsmodels GLM-Binomial on strict 0/1.

Outputs results_stats/interaction_test_strict.json.

Usage (from 3090, auditstack env):
  python scripts/openrca_eval/interaction_test_strict.py
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
OUT_JSON = ROOT / "results_stats" / "interaction_test_strict.json"

FRACTIONAL_P_REFERENCE = 8.63e-4


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------
def prepare_block() -> pd.DataFrame:
    """Return the 2 x 11 BARO/z_family block with STRICT 0/1 `correct`."""
    df = load_all_systems()
    sub = df[df["method"].isin(["baro", "z_family"])].copy()
    # Strict 0/1 binarization at 0.5 threshold.
    sub["correct_frac"] = sub["correct"].clip(0.0, 1.0).astype(float)
    sub["correct"] = (sub["correct_frac"] > 0.5).astype(int)
    return sub


def _coverage_and_rates(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    rate = sub.groupby(["method", "system_id"])["correct"].mean().unstack(fill_value=np.nan)
    return cov, rate


def _design_matrices(sub: pd.DataFrame):
    """Build full-model (with interactions) and reduced-model (additive) X
    matrices via patsy so we can hand them to sklearn / pyfirth.

    Returns (y, X_full, X_reduced, n_interaction_params)."""
    from patsy import dmatrices
    y_full, X_full = dmatrices(
        "correct ~ C(method) * C(system_id)", data=sub, return_type="dataframe"
    )
    y_red, X_red = dmatrices(
        "correct ~ C(method) + C(system_id)", data=sub, return_type="dataframe"
    )
    # Sanity: same y vector in both models.
    assert np.array_equal(y_full.values.ravel(), y_red.values.ravel())
    n_interaction_params = X_full.shape[1] - X_red.shape[1]
    return y_full.values.ravel().astype(int), X_full, X_red, int(n_interaction_params)


# ----------------------------------------------------------------------
# Approach A — Firth penalized logit (via pyfirth)
# ----------------------------------------------------------------------
def approach_firth(sub: pd.DataFrame) -> dict:
    """Firth-penalized logit LRT on the block of interaction dummies."""
    try:
        from pyfirth.PyFirth import PyFirth
    except Exception as e:
        return {
            "p_interaction": None,
            "status": "failed_install",
            "notes": f"pyfirth import failed: {e!r}",
        }

    # Build design matrix via patsy (drop the patsy Intercept column — pyfirth
    # adds its own, controlled via `hasconst`). We pass hasconst=True so pyfirth
    # keeps all columns including the patsy-generated `Intercept`.
    y, X_full, X_red, k_interaction = _design_matrices(sub)

    # pyfirth expects a single DataFrame with a y column.
    data = X_full.copy()
    data.columns = [_sanitize(c) for c in data.columns]
    y_col = "_y_"
    data[y_col] = y
    x_cols = [c for c in data.columns if c != y_col]

    # Identify which columns are interaction dummies — those in X_full but
    # not in X_red (matched by sanitized name).
    red_cols = {_sanitize(c) for c in X_red.columns}
    interaction_cols = [c for c in x_cols if c not in red_cols]
    assert len(interaction_cols) == k_interaction, (
        f"k_interaction mismatch: {len(interaction_cols)} vs {k_interaction}"
    )

    notes = []
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            firth = PyFirth(data, x_cols, y_col, hasconst=True)
            # Joint LRT across all interaction dummies.
            result = firth.fit(
                interaction_cols,
                num_iters=2000,
                step_limit=200,
                max_step=5.0,
                convergence_limit=1e-6,
            )
            for w in wlist:
                notes.append(f"{w.category.__name__}: {w.message}")
    except Exception as e:
        return {
            "p_interaction": None,
            "status": "failed_converge",
            "notes": f"pyfirth fit raised: {e!r}",
        }

    p_val = result.get("PVal")
    flag = result.get("Flag", 0)
    if p_val is None or (isinstance(p_val, float) and (np.isnan(p_val) or not np.isfinite(p_val))):
        return {
            "p_interaction": None,
            "status": "failed_converge",
            "notes": f"pyfirth returned non-finite PVal; flag={flag}; notes={notes}",
        }

    return {
        "p_interaction": float(p_val),
        "status": "success",
        "k_interaction_params": int(k_interaction),
        "firth_flag": int(flag),  # 0=OK, 1=convergence uncertain
        "notes": (
            f"Firth penalized LRT over {k_interaction} interaction dummies; "
            f"flag={flag} (0=OK). " + "; ".join(notes[:5])
        ).strip("; "),
    }


# ----------------------------------------------------------------------
# Approach B — L2-regularized logistic regression (sklearn)
# ----------------------------------------------------------------------
def approach_l2(sub: pd.DataFrame, C: float = 1.0) -> dict:
    """Mild L2 ridge + LRT on fit log-likelihoods."""
    from sklearn.linear_model import LogisticRegression

    y, X_full, X_red, k_interaction = _design_matrices(sub)

    # Drop patsy intercept because sklearn adds its own.
    def _drop_intercept(X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in X.columns if c.lower() != "intercept"]
        return X[cols].values

    Xf = _drop_intercept(X_full)
    Xr = _drop_intercept(X_red)

    notes = []

    def _fit_loglik(X: np.ndarray) -> tuple[float, bool]:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            clf = LogisticRegression(
                penalty="l2",
                C=C,
                solver="lbfgs",
                max_iter=5000,
                fit_intercept=True,
                tol=1e-8,
            )
            clf.fit(X, y)
            converged = True
            for w in wlist:
                msg = f"{w.category.__name__}: {w.message}"
                notes.append(msg)
                if "converge" in str(w.message).lower():
                    converged = False
        proba = clf.predict_proba(X)[:, 1]
        # Numerical guard.
        eps = 1e-12
        proba = np.clip(proba, eps, 1 - eps)
        ll = float(np.sum(np.log(proba[y == 1])) + np.sum(np.log(1.0 - proba[y == 0])))
        return ll, converged

    ll_full, conv_full = _fit_loglik(Xf)
    ll_red, conv_red = _fit_loglik(Xr)
    lrt = 2.0 * (ll_full - ll_red)
    if lrt < 0:
        # Rare but can happen under regularization if the full model doesn't
        # strictly improve the LL (it usually does here). Clip defensively.
        notes.append(f"LRT negative ({lrt:.4f}); clipping to 0 — regularized LLF comparison.")
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
            f"sklearn L2 logit (C={C}); LRT on unpenalized LLF computed from "
            f"predicted probabilities. Full converged={conv_full}, "
            f"reduced converged={conv_red}. " + "; ".join(notes[:5])
        ).strip("; "),
    }


# ----------------------------------------------------------------------
# Approach C — drop the single quasi-separating cell
# ----------------------------------------------------------------------
def approach_drop_cell(sub: pd.DataFrame) -> dict:
    """Drop openrca::mkt1 / baro rows and refit strict Bernoulli logit."""
    filt = ~((sub["system_id"] == "openrca::mkt1") & (sub["method"] == "baro"))
    sub2 = sub[filt].copy()
    n_obs = int(len(sub2))

    # Coverage is now 2x11 minus 1 cell. For the interaction model with a
    # single missing cell, patsy will simply omit the corresponding
    # dummy, so the degrees of freedom shrink by 1.
    convergence_warnings = []
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            m_full = smf.glm(
                "correct ~ C(method) * C(system_id)",
                data=sub2,
                family=sm.families.Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": sub2["case_id"]})
            m_red = smf.glm(
                "correct ~ C(method) + C(system_id)",
                data=sub2,
                family=sm.families.Binomial(),
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
        return {
            "p_interaction": None,
            "status": "failed_converge",
            "n_obs": n_obs,
            "notes": f"statsmodels GLM raised: {e!r}",
        }

    status = "success" if (conv_full and conv_red) else "success_with_warnings"
    return {
        "p_interaction": p,
        "status": status,
        "n_obs": n_obs,
        "lrt_stat": float(lrt),
        "df": int(df_diff),
        "llf_full": llf_full,
        "llf_reduced": llf_red,
        "full_converged": conv_full,
        "reduced_converged": conv_red,
        "notes": (
            "Dropped 1 quasi-separating cell (openrca::mkt1, baro); "
            "statsmodels GLM-Binomial logit, cluster SE by case_id. "
            + "; ".join(convergence_warnings[:5])
        ).strip("; "),
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sanitize(name: str) -> str:
    """Make patsy column names safe as DataFrame keys (they already are,
    but strip whitespace defensively)."""
    return str(name).strip()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("[load] reading 11 per-system CSVs and building strict-binary block ...")
    sub = prepare_block()
    cov, rate = _coverage_and_rates(sub)
    n_obs = int(len(sub))
    n_cases = int(sub["case_id"].nunique())

    print("Coverage matrix (obs counts), method x system:")
    print(cov.to_string())
    print("\nStrict 0/1 accuracy rate per cell:")
    print(rate.round(4).to_string())

    # Identify quasi-separating cells (rate = 0 or 1).
    sep_cells = []
    for m in rate.index:
        for s in rate.columns:
            r = rate.loc[m, s]
            if np.isclose(r, 0.0) or np.isclose(r, 1.0):
                sep_cells.append({"method": m, "system_id": s, "rate": float(r)})
    print(f"\nQuasi-separating cells (rate in {{0,1}}): {sep_cells}")
    print(f"\nn_obs={n_obs}  n_cases={n_cases}")

    # -------- Run all three approaches --------
    print("\n[A] Firth penalized logit (pyfirth) ...")
    a_result = approach_firth(sub)
    print(f"    -> {a_result}")

    print("\n[B] L2-regularized logit (sklearn, C=1.0) ...")
    b_result = approach_l2(sub, C=1.0)
    print(f"    -> {b_result}")

    print("\n[C] Drop mkt1/baro cell + unregularized GLM ...")
    c_result = approach_drop_cell(sub)
    print(f"    -> {c_result}")

    # -------- Interpretation --------
    p_values = {}
    if a_result.get("p_interaction") is not None:
        p_values["firth"] = a_result["p_interaction"]
    if b_result.get("p_interaction") is not None:
        p_values["l2_regularized"] = b_result["p_interaction"]
    if c_result.get("p_interaction") is not None:
        p_values["drop_mkt1_baro"] = c_result["p_interaction"]

    all_below_05 = all(p < 0.05 for p in p_values.values()) if p_values else False
    max_p = max(p_values.values()) if p_values else None
    min_p = min(p_values.values()) if p_values else None

    if not p_values:
        conclusion = "AMBIGUOUS — all three strict approaches failed; fall back to fractional logit (p=8.63e-4)."
    elif all_below_05:
        # Check order-of-magnitude agreement.
        # Fractional reference is ~1e-3.
        orders = {
            k: int(np.floor(np.log10(max(v, 1e-300))))
            for k, v in p_values.items()
        }
        ref_order = int(np.floor(np.log10(FRACTIONAL_P_REFERENCE)))
        close = all(abs(o - ref_order) <= 2 for o in orders.values())
        if close:
            conclusion = (
                f"ROBUST — all strict approaches give p<0.05 (max p={max_p:.3e}) "
                f"within 2 orders of magnitude of fractional reference ({FRACTIONAL_P_REFERENCE:.3e}); "
                f"method x system interaction is not an artifact of fractional coding."
            )
        else:
            conclusion = (
                f"ROBUST (with attenuation) — all p<0.05 but spread across orders "
                f"({min_p:.2e} to {max_p:.2e}); direction consistent with fractional logit."
            )
    else:
        conclusion = (
            f"SENSITIVE — at least one strict approach gives p>=0.05 "
            f"(max p={max_p:.3e}); result is sensitive to rounding 0.25-0.75 partial credit to 0/1."
        )

    # -------- Assemble output --------
    out = {
        "model_spec_reference_fractional": (
            "GLM-Binomial fractional logit, p=8.63e-4 "
            "(interaction_test.json, Papke-Wooldridge)."
        ),
        "binarization": "correct_strict = int(correct_fractional > 0.5)",
        "n_obs": n_obs,
        "n_cases": n_cases,
        "coverage_matrix": cov.to_dict(),
        "rate_matrix_strict": rate.round(6).to_dict(),
        "quasi_separating_cells": sep_cells,
        "strict_logit_approaches": {
            "firth": a_result,
            "l2_regularized": b_result,
            "drop_mkt1_baro": c_result,
        },
        "p_interaction_summary": {
            "fractional_reference": FRACTIONAL_P_REFERENCE,
            **p_values,
        },
        "conclusion": conclusion,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {OUT_JSON}")

    # Pretty summary
    print("\n=== Strict logit sensitivity summary ===")
    print(f"  Fractional reference (GLM-Binomial):  p = {FRACTIONAL_P_REFERENCE:.3e}")
    for k in ("firth", "l2_regularized", "drop_mkt1_baro"):
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
