#!/usr/bin/env python3
"""Step 3 of propagation-aware difficulty proxy.

Extension of `interaction_test_difficulty.py`: adds the structural propagation
proxy `anomalous_metric_count_mean` as a third continuous covariate, then
refits the 3 nested GLM-Binomial models (cluster SE by case_id on the
{BARO, z_family} x 11-system block) and compares LRTs to the v1 run.

Models
  X' (reduced):
      correct ~ C(method) + majority_baseline + root_entropy
                 + anomalous_metric_count_mean
  Y' (difficulty x method):
      correct ~ C(method) * (majority_baseline + root_entropy
                 + anomalous_metric_count_mean)
  Z  (full system interaction, unchanged):
      correct ~ C(method) * C(system_id)

LRTs
  * LRT(Y' vs X') : does difficulty-x-method (now including structural
                    proxy) explain method-level differences?
  * LRT(Z vs Y')  : does residual system heterogeneity remain AFTER
                    controlling for label + structural difficulty?
                    THIS IS THE PAPER'S MAIN DEFENSIVE CLAIM.

Compare with v1 (`interaction_test_difficulty.json`):
  * If LRT(Z vs Y')'s chi2 SHRINKS when we add structural proxy -> the proxy
    absorbs part of system heterogeneity. Magnitude of shrinkage quantifies
    how much propagation structure "explains" the system-identity effect.
  * If LRT(Z vs Y') still p < 0.05 -> residual heterogeneity beyond BOTH
    label and structural difficulty; system identity carries additional
    explanatory weight.

Output: results_stats/interaction_test_difficulty_v2.json
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
# v2 difficulty indicators: emits the extra 2 structural columns.
from difficulty_indicators_v2 import (
    _load_full_per_case,
    _agg_anomalous,
)
from difficulty_indicators import compute_all_systems

ROOT = Path("$HOME/auditstack")
OUT_JSON = ROOT / "results_stats" / "interaction_test_difficulty_v2.json"
V1_JSON = ROOT / "results_stats" / "interaction_test_difficulty.json"

SPEC_X2 = ("correct ~ C(method) + majority_baseline + root_entropy"
           " + anomalous_metric_count_mean")
SPEC_Y2 = ("correct ~ C(method) * (majority_baseline + root_entropy"
           " + anomalous_metric_count_mean)")
SPEC_Z = "correct ~ C(method) * C(system_id)"


def _fit(df: pd.DataFrame, formula: str):
    return smf.glm(
        formula, data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})


def _lrt(ll_big: float, ll_small: float, df_diff: int) -> dict:
    stat = 2.0 * (ll_big - ll_small)
    p = float(1.0 - chi2.cdf(stat, df_diff)) if df_diff > 0 else float("nan")
    return {"stat": float(stat), "df": int(df_diff), "p": p}


def _system_key(benchmark: str, system: str) -> str:
    return f"{benchmark}::{system}"


def main():
    # -------- Observation-level data (BARO + z_family 2x11 block) -------- #
    df = load_all_systems()
    sub = df[df["method"].isin(["baro", "z_family"])].copy()
    sub["correct"] = sub["correct"].clip(0.0, 1.0).astype(float)

    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    assert cov.shape == (2, 11), f"Expected 2x11, got {cov.shape}"
    assert (cov.values > 0).all(), "Some cells empty"

    # -------- Join per-system difficulty indicators (v2: adds struct) ---- #
    v1_table = compute_all_systems()
    struct_agg = _agg_anomalous(_load_full_per_case())
    diff_table = v1_table.merge(
        struct_agg[["benchmark", "system",
                    "anomalous_metric_count_mean",
                    "anomalous_metric_count_cv"]],
        on=["benchmark", "system"], how="left"
    )
    diff_table["system_id"] = diff_table.apply(
        lambda r: _system_key(r["benchmark"], r["system"]), axis=1
    )
    cols = ["system_id", "majority_baseline", "root_entropy",
            "service_count", "n_cases",
            "anomalous_metric_count_mean", "anomalous_metric_count_cv"]
    merged = sub.merge(diff_table[cols], on="system_id", how="left")
    assert merged["majority_baseline"].notna().all()
    assert merged["anomalous_metric_count_mean"].notna().all(), \
        "Some rows missing structural proxy — check merge keys"

    n_obs = int(len(merged))
    n_cases = int(merged["case_id"].nunique())
    n_systems = int(merged["system_id"].nunique())
    print(f"n_obs={n_obs}  n_cases={n_cases}  n_systems={n_systems}")

    # Quick descriptive view of the structural proxy on OB/SS/TT
    print("\nDifficulty covariates per system_id (v2):")
    per_sys = merged.drop_duplicates("system_id")[[
        "system_id", "majority_baseline", "root_entropy",
        "anomalous_metric_count_mean", "anomalous_metric_count_cv"]]
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 160, "display.max_columns", 20):
        print(per_sys.sort_values("system_id").to_string(index=False))

    # -------- Fit 3 nested models -------- #
    warn_log: list[str] = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model_X2 = _fit(merged, SPEC_X2)
        model_Y2 = _fit(merged, SPEC_Y2)
        model_Z = _fit(merged, SPEC_Z)
        for w in wlist:
            warn_log.append(f"{w.category.__name__}: {w.message}")
    for name, m in [("X2", model_X2), ("Y2", model_Y2), ("Z", model_Z)]:
        warn_log.append(f"{name} converged={bool(getattr(m, 'converged', True))}")

    llf_X2 = float(model_X2.llf)
    llf_Y2 = float(model_Y2.llf)
    llf_Z = float(model_Z.llf)
    np_X2 = int(len(model_X2.params))
    np_Y2 = int(len(model_Y2.params))
    np_Z = int(len(model_Z.params))

    print(f"\nllf_X' = {llf_X2:.4f}  (params={np_X2})")
    print(f"llf_Y' = {llf_Y2:.4f}  (params={np_Y2})")
    print(f"llf_Z  = {llf_Z:.4f}  (params={np_Z})")

    lrt_Y_vs_X = _lrt(llf_Y2, llf_X2, np_Y2 - np_X2)
    lrt_Z_vs_Y = _lrt(llf_Z, llf_Y2, np_Z - np_Y2)

    print(f"\nLRT Y' vs X': stat={lrt_Y_vs_X['stat']:.4f}  "
          f"df={lrt_Y_vs_X['df']}  p={lrt_Y_vs_X['p']:.3e}")
    print(f"LRT Z  vs Y': stat={lrt_Z_vs_Y['stat']:.4f}  "
          f"df={lrt_Z_vs_Y['df']}  p={lrt_Z_vs_Y['p']:.3e}")

    # -------- Compare to v1 -------- #
    v1_delta = {}
    if V1_JSON.exists():
        try:
            v1 = json.loads(V1_JSON.read_text())
            v1_z_vs_y = v1.get("lrt_Z_vs_Y", {})
            v1_y_vs_x = v1.get("lrt_Y_vs_X", {})
            v1_delta = {
                "v1_lrt_Z_vs_Y": v1_z_vs_y,
                "v1_lrt_Y_vs_X": v1_y_vs_x,
                "delta_chi2_Z_vs_Y": float(
                    lrt_Z_vs_Y["stat"] - v1_z_vs_y.get("stat", float("nan"))
                ),
                "delta_df_Z_vs_Y": int(
                    lrt_Z_vs_Y["df"] - v1_z_vs_y.get("df", 0)
                ),
                "delta_chi2_Y_vs_X": float(
                    lrt_Y_vs_X["stat"] - v1_y_vs_x.get("stat", float("nan"))
                ),
                "delta_df_Y_vs_X": int(
                    lrt_Y_vs_X["df"] - v1_y_vs_x.get("df", 0)
                ),
            }
            print("\n[v1 comparison]")
            print(f"  v1 LRT(Z vs Y)  chi2={v1_z_vs_y.get('stat'):.4f} df={v1_z_vs_y.get('df')} "
                  f"p={v1_z_vs_y.get('p'):.3e}")
            print(f"  v2 LRT(Z vs Y') chi2={lrt_Z_vs_Y['stat']:.4f} df={lrt_Z_vs_Y['df']} "
                  f"p={lrt_Z_vs_Y['p']:.3e}")
            print(f"  Δchi2(Z vs Y') = {v1_delta['delta_chi2_Z_vs_Y']:+.4f}  "
                  f"(negative = structural proxy absorbed heterogeneity)")
        except Exception as e:
            print(f"[warn] could not read v1 JSON: {e}")
    else:
        print("[info] v1 JSON not found — skipping comparison block")

    z_vs_y_rank_ok = (np_Z - np_Y2) > 0
    pass_condition = (
        z_vs_y_rank_ok
        and np.isfinite(lrt_Z_vs_Y["p"])
        and lrt_Z_vs_Y["p"] < 0.05
    )
    conclusion = (
        "PASS — even after adding structural propagation proxy, residual "
        "system heterogeneity remains significant (p(Z vs Y') < 0.05). "
        "System identity carries explanatory power beyond label-centric AND "
        "propagation-structural difficulty."
        if pass_condition else
        "FAIL — structural proxy absorbs the residual heterogeneity, or LRT "
        "is non-significant at alpha=0.05."
    )

    out = {
        "model_X_spec": SPEC_X2,
        "model_Y_spec": SPEC_Y2,
        "model_Z_spec": SPEC_Z + "   [unchanged from v1]",
        "n_obs": n_obs,
        "n_cases": n_cases,
        "n_systems": n_systems,
        "coverage_matrix_2x11": cov.to_dict(),
        "llf_X": llf_X2, "llf_Y": llf_Y2, "llf_Z": llf_Z,
        "n_params_X": np_X2, "n_params_Y": np_Y2, "n_params_Z": np_Z,
        "lrt_Y_vs_X": {
            **lrt_Y_vs_X,
            "interpretation": (
                "Does difficulty x method interaction (now with 3 covariates "
                "including structural proxy) explain any method-level "
                "differences? (p<0.05 => yes.)"
            ),
        },
        "lrt_Z_vs_Y": {
            **lrt_Z_vs_Y,
            "rank_ok": bool(z_vs_y_rank_ok),
            "interpretation": (
                "Residual system heterogeneity beyond label + structural "
                "difficulty covariates. p<0.05 means system identity matters "
                "even after controlling for majority_baseline + root_entropy "
                "+ anomalous_metric_count_mean and their interaction with "
                "method. THIS IS THE PAPER'S MAIN DEFENSIVE CLAIM (v2)."
            ),
        },
        "v1_comparison": v1_delta,
        "difficulty_table_v2_snapshot": per_sys.sort_values("system_id").to_dict(orient="records"),
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
