#!/usr/bin/env python3
"""Sensitivity analysis (UNIFIED block, R1): drop PetShop temporal_traffic{1,2}.

Mirror of `sensitivity_no_temporal.py`, but rebased on the R1 unified inputs
(`results_xbench_11sys_unified/_inputs/`) and the {BARO, zbase_univ} method
pair. This gives a clean apples-to-apples comparison against the unified
interaction reference (p = 5.26e-05 in `interaction_test_unified.json`),
whereas the original sensitivity driver was tied to the pre-R1 z_family
collapse (p = 8.63e-04).

Reproducibility: bootstrap seed=42.

Outputs
-------
  results_stats/sensitivity_temporal_traffic_unified.csv
      Rows = analysis type, columns = [with_temporal, without_temporal, change]
      Followed by a "OLD z_family reference" block for side-by-side viewing.
  results_stats/sensitivity_narrative_unified.md
      One-paragraph robustness narrative (PASS or WARN).
  results_stats/sensitivity_view_accuracy_no_temporal_unified.csv
      Filtered per-view accuracy table (9 rows).
  results_stats/sensitivity_bootstrap_no_temporal_unified.csv
      Filtered headline baro vs zbase_univ bootstrap CIs (9 rows).
  results_stats/sensitivity_no_temporal_unified_raw.json
      Full raw numeric dump for transparency.

Constraints
-----------
* Does NOT modify `sensitivity_no_temporal.py`, `interaction_test_unified.py`,
  `bootstrap_pairwise.py`, or `interaction_test_difficulty.py`.
* Reuses:
    - `interaction_test_unified.load_all_systems_unified` for the data frame
    - `bootstrap_pairwise.paired_bootstrap` for the per-system CI
    - `difficulty_indicators.compute_all_systems` for difficulty covariates
      (method-agnostic: we only use `majority_baseline` + `root_entropy`)
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

from interaction_test_unified import load_all_systems_unified
from bootstrap_pairwise import paired_bootstrap
from difficulty_indicators import compute_all_systems

ROOT = Path("$HOME/auditstack")
OUT_DIR = ROOT / "results_stats"
OUT_COMPARISON = OUT_DIR / "sensitivity_temporal_traffic_unified.csv"
OUT_NARRATIVE = OUT_DIR / "sensitivity_narrative_unified.md"
OUT_VIEW = OUT_DIR / "sensitivity_view_accuracy_no_temporal_unified.csv"
OUT_BOOT = OUT_DIR / "sensitivity_bootstrap_no_temporal_unified.csv"
OUT_RAW = OUT_DIR / "sensitivity_no_temporal_unified_raw.json"

# Existing (with-temporal) headline numbers on the UNIFIED block. Verified
# 2026-04-20 against:
#   results_stats/interaction_test_unified.json (p_interaction, n_obs)
# The difficulty LRT on the unified block is computed here (no canonical
# unified `interaction_test_difficulty_unified.json` exists yet); the
# with-temporal reference is taken from THIS run (full 11-system frame)
# rather than from the old z_family difficulty artifact, to keep the
# comparison apples-to-apples within the unified block.
WITH_TEMPORAL_UNIFIED = {
    "interaction_p": 5.257987047457835e-05,     # unified reference (R1)
    "n_systems": 11,
    "n_obs": 1556,
}

# For visual side-by-side only: the OLD R4 (z_family, pre-R1) reference.
# From results_stats/sensitivity_temporal_traffic.csv and
# results_stats/interaction_test_difficulty.json.
OLD_Z_FAMILY_REFERENCE = {
    "interaction_p_with_temporal": 8.624715295846297e-04,
    "interaction_p_without_temporal_zfamily": None,  # filled from old CSV
    "difficulty_LRT_Z_vs_Y_chi2_with_temporal": 143.85056639797176,
    "difficulty_LRT_Z_vs_Y_p_with_temporal": 0.0,
}

DROP_SYSTEMS = {"petshop::temporal_traffic1", "petshop::temporal_traffic2"}


# ------------------------------------------------------------------ #
# Fit helpers (mirror interaction_test.py / interaction_test_difficulty.py)
# ------------------------------------------------------------------ #
def _fit_glm(df: pd.DataFrame, formula: str):
    return smf.glm(
        formula, data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})


def _lrt(ll_big: float, ll_small: float, df_diff: int) -> dict:
    stat = 2.0 * (ll_big - ll_small)
    p = float(1.0 - chi2.cdf(stat, df_diff)) if df_diff > 0 else float("nan")
    return {"stat": float(stat), "df": int(df_diff), "p": p}


def _fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "nan"
    if p == 0.0:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.3e}"
    return f"{p:.4f}"


def _fmt_change(old, new) -> str:
    if old is None or new is None:
        return "n/a"
    if not (np.isfinite(old) and np.isfinite(new)):
        return "n/a"
    return f"{new - old:+.4g}"


# ------------------------------------------------------------------ #
# Core analyses                                                      #
# ------------------------------------------------------------------ #
def run_interaction_test(sub: pd.DataFrame) -> dict:
    """Mirror interaction_test_unified.py on the filtered frame."""
    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    n_obs = int(len(sub))
    n_cases = int(sub["case_id"].nunique())
    n_systems = int(sub["system_id"].nunique())

    warn_log = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        full = _fit_glm(sub, "correct ~ C(method) * C(system_id)")
        reduced = _fit_glm(sub, "correct ~ C(method) + C(system_id)")
        for w in wlist:
            warn_log.append(f"{w.category.__name__}: {w.message}")
    warn_log.append(f"full converged={bool(getattr(full, 'converged', True))}")
    warn_log.append(f"reduced converged={bool(getattr(reduced, 'converged', True))}")

    llf_full = float(full.llf)
    llf_reduced = float(reduced.llf)
    df_full = int(len(full.params))
    df_reduced = int(len(reduced.params))
    df_diff = df_full - df_reduced
    stat = 2.0 * (llf_full - llf_reduced)
    p_interaction = float(1.0 - chi2.cdf(stat, df_diff))

    return {
        "n_obs": n_obs,
        "n_cases": n_cases,
        "n_systems": n_systems,
        "coverage_matrix": cov.to_dict(),
        "llf_full": llf_full,
        "llf_reduced": llf_reduced,
        "df_full_params": df_full,
        "df_reduced_params": df_reduced,
        "df_diff": df_diff,
        "lrt_statistic": float(stat),
        "p_interaction": p_interaction,
        "warnings": warn_log,
    }


def run_difficulty_test(sub: pd.DataFrame) -> dict:
    """Mirror interaction_test_difficulty.py Z-vs-Y LRT on the unified block.

    Only `majority_baseline` + `root_entropy` are merged in — both are
    method-agnostic, so reusing the original z_family-derived difficulty
    table is sound (the covariates do not change when the z-method label
    renames from 'z_family' to 'zbase_univ')."""
    diff_table = compute_all_systems()
    diff_table["system_id"] = (
        diff_table["benchmark"].astype(str) + "::" + diff_table["system"].astype(str)
    )
    cols = ["system_id", "majority_baseline", "root_entropy"]
    merged = sub.merge(diff_table[cols], on="system_id", how="left")
    assert merged["majority_baseline"].notna().all(), \
        "Missing difficulty covariates after filter — check system_id mapping"

    warn_log = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model_X = _fit_glm(merged, "correct ~ C(method) + majority_baseline + root_entropy")
        model_Y = _fit_glm(merged, "correct ~ C(method) * (majority_baseline + root_entropy)")
        model_Z = _fit_glm(merged, "correct ~ C(method) * C(system_id)")
        for w in wlist:
            warn_log.append(f"{w.category.__name__}: {w.message}")
    warn_log.append(f"X converged={bool(getattr(model_X, 'converged', True))}")
    warn_log.append(f"Y converged={bool(getattr(model_Y, 'converged', True))}")
    warn_log.append(f"Z converged={bool(getattr(model_Z, 'converged', True))}")

    llf_X = float(model_X.llf)
    llf_Y = float(model_Y.llf)
    llf_Z = float(model_Z.llf)
    n_X = int(len(model_X.params))
    n_Y = int(len(model_Y.params))
    n_Z = int(len(model_Z.params))

    lrt_ZvsY = _lrt(llf_Z, llf_Y, n_Z - n_Y)
    lrt_YvsX = _lrt(llf_Y, llf_X, n_Y - n_X)

    return {
        "llf_X": llf_X,
        "llf_Y": llf_Y,
        "llf_Z": llf_Z,
        "n_params_X": n_X,
        "n_params_Y": n_Y,
        "n_params_Z": n_Z,
        "lrt_Y_vs_X": lrt_YvsX,
        "lrt_Z_vs_Y": lrt_ZvsY,
        "warnings": warn_log,
    }


def run_view_accuracy(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-system acc@1 for BARO and zbase_univ on the filtered block."""
    g = sub.groupby(["system_id", "method"]).agg(
        n=("correct", "size"),
        acc1=("correct", "mean"),
    ).reset_index()
    piv = g.pivot_table(index="system_id", columns="method", values="acc1").reset_index()
    piv.columns.name = None
    return piv


def run_bootstrap(df_all: pd.DataFrame, systems_kept: list[str],
                  n_iter: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Headline baro vs zbase_univ bootstrap CIs on each retained system."""
    rows = []
    for system in sorted(systems_kept):
        res = paired_bootstrap(
            df_all, "baro", "zbase_univ", system, n_iter=n_iter, seed=seed
        )
        rows.append({
            "system": system,
            "n_cases": res["n_cases"],
            "mean_delta": res["mean_delta"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
            "ci_width_pp": 100.0 * (res["ci_high"] - res["ci_low"]),
            "p_two_sided": res["p_value_twosided"],
        })
    return pd.DataFrame(rows)


def run_bootstrap_all11(df_all: pd.DataFrame,
                        n_iter: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Full 11-system baro vs zbase_univ bootstrap (for with-temporal ref CIs)."""
    all_systems = sorted(df_all["system_id"].unique())
    return run_bootstrap(df_all, all_systems, n_iter=n_iter, seed=seed)


# ------------------------------------------------------------------ #
# Load OLD (z_family) sensitivity for side-by-side visual            #
# ------------------------------------------------------------------ #
def load_old_sensitivity_table() -> pd.DataFrame | None:
    path = OUT_DIR / "sensitivity_temporal_traffic.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main():
    print("[load] reading UNIFIED 11-system long frame ...")
    df = load_all_systems_unified()
    print(f"       n_rows={len(df)}  n_systems={df['system_id'].nunique()}  "
          f"methods={sorted(df['method'].unique())}")

    # Restrict to BARO + zbase_univ (R1 complete 2x11 block).
    sub_all = df[df["method"].isin(["baro", "zbase_univ"])].copy()
    sub_all["correct"] = sub_all["correct"].clip(0.0, 1.0).astype(float)

    cov_all = sub_all.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    assert cov_all.shape == (2, 11), f"Expected 2x11, got {cov_all.shape}"
    assert (cov_all.values > 0).all(), "Some cells empty on unified block."

    # --- Filtered frame ---
    sub_filt = sub_all[~sub_all["system_id"].isin(DROP_SYSTEMS)].copy()
    kept_systems = sorted(sub_filt["system_id"].unique())
    print(f"[filter] kept {len(kept_systems)} systems: {kept_systems}")
    print(f"[filter] n_obs: with_temporal={len(sub_all)}  "
          f"without_temporal={len(sub_filt)}")

    # --- 1. Interaction test on filtered frame ---
    print("\n[interaction] fitting GLM-Binomial on filtered frame ...")
    inter_new = run_interaction_test(sub_filt)
    print(f"  n_obs         = {inter_new['n_obs']}")
    print(f"  n_systems     = {inter_new['n_systems']}")
    print(f"  llf_full      = {inter_new['llf_full']:.4f}  ({inter_new['df_full_params']} params)")
    print(f"  llf_reduced   = {inter_new['llf_reduced']:.4f}  ({inter_new['df_reduced_params']} params)")
    print(f"  LRT stat      = {inter_new['lrt_statistic']:.4f}  df={inter_new['df_diff']}")
    print(f"  p_interaction = {_fmt_p(inter_new['p_interaction'])}")

    # --- 1b. Interaction test on FULL frame (for unified-block diff LRT ref) ---
    print("\n[interaction] fitting GLM-Binomial on FULL unified frame (ref) ...")
    inter_ref = run_interaction_test(sub_all)
    print(f"  p_interaction (with temporal) = {_fmt_p(inter_ref['p_interaction'])}")
    # Sanity: should match WITH_TEMPORAL_UNIFIED['interaction_p']
    assert abs(inter_ref["p_interaction"] - WITH_TEMPORAL_UNIFIED["interaction_p"]) < 1e-9, \
        (f"Interaction p on full frame ({inter_ref['p_interaction']}) "
         f"does not match cached reference "
         f"({WITH_TEMPORAL_UNIFIED['interaction_p']}). "
         "Inputs may have changed.")

    # --- 2a. Difficulty LRT on filtered frame ---
    print("\n[difficulty] fitting X/Y/Z on filtered unified frame ...")
    diff_new = run_difficulty_test(sub_filt)
    print(f"  llf_X = {diff_new['llf_X']:.4f}  (params={diff_new['n_params_X']})")
    print(f"  llf_Y = {diff_new['llf_Y']:.4f}  (params={diff_new['n_params_Y']})")
    print(f"  llf_Z = {diff_new['llf_Z']:.4f}  (params={diff_new['n_params_Z']})")
    print(f"  LRT Z vs Y = {diff_new['lrt_Z_vs_Y']['stat']:.4f}  "
          f"df={diff_new['lrt_Z_vs_Y']['df']}  p={_fmt_p(diff_new['lrt_Z_vs_Y']['p'])}")
    print(f"  LRT Y vs X = {diff_new['lrt_Y_vs_X']['stat']:.4f}  "
          f"df={diff_new['lrt_Y_vs_X']['df']}  p={_fmt_p(diff_new['lrt_Y_vs_X']['p'])}")

    # --- 2b. Difficulty LRT on FULL unified frame (reference) ---
    print("\n[difficulty] fitting X/Y/Z on FULL unified frame (ref) ...")
    diff_ref = run_difficulty_test(sub_all)
    print(f"  LRT Z vs Y = {diff_ref['lrt_Z_vs_Y']['stat']:.4f}  "
          f"df={diff_ref['lrt_Z_vs_Y']['df']}  p={_fmt_p(diff_ref['lrt_Z_vs_Y']['p'])}")

    # --- 3. Filtered view accuracy ---
    view_tbl = run_view_accuracy(sub_filt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    view_tbl.to_csv(OUT_VIEW, index=False)
    print(f"\n[view] wrote {OUT_VIEW}")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 160,
                           "display.max_columns", 20):
        print(view_tbl.to_string(index=False))

    # --- 4. Filtered headline BARO vs zbase_univ bootstrap (9 systems) ---
    print("\n[bootstrap] running paired bootstrap on 9 retained systems ...")
    boot_new = run_bootstrap(df, kept_systems, n_iter=10_000, seed=42)
    boot_new.to_csv(OUT_BOOT, index=False)
    print(f"[bootstrap] wrote {OUT_BOOT}")
    with pd.option_context("display.float_format", "{:.5f}".format,
                           "display.width", 200,
                           "display.max_columns", 20):
        print(boot_new.to_string(index=False))

    # --- 4b. Full 11-system bootstrap for with-temporal reference CIs ---
    print("\n[bootstrap] running paired bootstrap on all 11 systems (ref) ...")
    boot_all11 = run_bootstrap_all11(df, n_iter=10_000, seed=42)
    # CI significance at alpha=0.05 via CI excluding zero.
    def sig_from_ci(lo, hi):
        if lo > 0 or hi < 0:
            return True
        return False

    boot_all11["sig"] = boot_all11.apply(
        lambda r: sig_from_ci(r["ci_low"], r["ci_high"]), axis=1)
    boot_new["sig"] = boot_new.apply(
        lambda r: sig_from_ci(r["ci_low"], r["ci_high"]), axis=1)

    # --- 5. Compare per-system: retained-system paired bootstraps are identical
    # by construction (paired resamples are seeded per-system). Just confirm
    # that statement holds numerically and surface any CI significance / sign
    # flips on retained systems.
    boot_old_same9 = boot_all11[~boot_all11["system"].isin(DROP_SYSTEMS)].copy()
    boot_old_same9 = boot_old_same9.rename(columns={
        "mean_delta": "mean_delta_with",
        "ci_low": "ci_low_with",
        "ci_high": "ci_high_with",
        "ci_width_pp": "ci_width_pp_with",
        "p_two_sided": "p_two_sided_with",
        "sig": "sig_with",
    })
    merged_boot = boot_new.rename(columns={
        "mean_delta": "mean_delta_without",
        "ci_low": "ci_low_without",
        "ci_high": "ci_high_without",
        "ci_width_pp": "ci_width_pp_without",
        "p_two_sided": "p_two_sided_without",
        "sig": "sig_without",
    }).merge(
        boot_old_same9[["system", "mean_delta_with", "ci_low_with", "ci_high_with",
                        "ci_width_pp_with", "p_two_sided_with", "sig_with"]],
        on="system", how="inner",
    )

    flipped_sig = merged_boot[merged_boot["sig_with"] != merged_boot["sig_without"]].copy()

    def sign(x): return 1 if x > 1e-9 else (-1 if x < -1e-9 else 0)
    merged_boot["sign_with"] = merged_boot["mean_delta_with"].apply(sign)
    merged_boot["sign_without"] = merged_boot["mean_delta_without"].apply(sign)
    flipped_sign = merged_boot[
        merged_boot["sign_with"] != merged_boot["sign_without"]
    ].copy()

    print("\n[bootstrap-compare] per-system deltas (with vs without temporal):")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 220,
                           "display.max_columns", 20):
        print(merged_boot[[
            "system", "n_cases",
            "mean_delta_with", "mean_delta_without",
            "ci_width_pp_with", "ci_width_pp_without",
            "sig_with", "sig_without",
        ]].to_string(index=False))
    if not flipped_sig.empty:
        print("[bootstrap-compare] WARNING: significance flipped for:")
        print(flipped_sig[["system", "ci_low_with", "ci_high_with",
                           "ci_low_without", "ci_high_without"]].to_string(index=False))
    else:
        print("[bootstrap-compare] No CI significance flipped on retained systems.")
    if not flipped_sign.empty:
        print("[bootstrap-compare] WARNING: mean-delta sign flipped for:")
        print(flipped_sign[["system", "mean_delta_with",
                            "mean_delta_without"]].to_string(index=False))

    mean_width_with = float(boot_old_same9["ci_width_pp_with"].mean())
    mean_width_without = float(boot_new["ci_width_pp"].mean())

    # --- 6. Side-by-side CSV (UNIFIED + OLD z_family ref)                  ---
    rows = [
        {
            "analysis": "interaction_p",
            "with_temporal": f"{WITH_TEMPORAL_UNIFIED['interaction_p']:.3e}",
            "without_temporal": _fmt_p(inter_new["p_interaction"]),
            "change": _fmt_change(
                WITH_TEMPORAL_UNIFIED["interaction_p"], inter_new["p_interaction"]
            ),
        },
        {
            "analysis": "difficulty_LRT_Z_vs_Y_chi2",
            "with_temporal": f"{diff_ref['lrt_Z_vs_Y']['stat']:.4f}",
            "without_temporal": f"{diff_new['lrt_Z_vs_Y']['stat']:.4f}",
            "change": _fmt_change(
                diff_ref["lrt_Z_vs_Y"]["stat"], diff_new["lrt_Z_vs_Y"]["stat"]
            ),
        },
        {
            "analysis": "difficulty_LRT_Z_vs_Y_p",
            "with_temporal": _fmt_p(diff_ref["lrt_Z_vs_Y"]["p"]),
            "without_temporal": _fmt_p(diff_new["lrt_Z_vs_Y"]["p"]),
            "change": _fmt_change(
                diff_ref["lrt_Z_vs_Y"]["p"], diff_new["lrt_Z_vs_Y"]["p"]
            ),
        },
        {
            "analysis": "baro_vs_zbase_univ_mean_CI_width_pp",
            "with_temporal": f"{mean_width_with:.3f}",
            "without_temporal": f"{mean_width_without:.3f}",
            "change": _fmt_change(mean_width_with, mean_width_without),
        },
        {
            "analysis": "n_systems",
            "with_temporal": str(WITH_TEMPORAL_UNIFIED["n_systems"]),
            "without_temporal": str(inter_new["n_systems"]),
            "change": str(inter_new["n_systems"] - WITH_TEMPORAL_UNIFIED["n_systems"]),
        },
        {
            "analysis": "n_obs",
            "with_temporal": str(WITH_TEMPORAL_UNIFIED["n_obs"]),
            "without_temporal": str(inter_new["n_obs"]),
            "change": str(inter_new["n_obs"] - WITH_TEMPORAL_UNIFIED["n_obs"]),
        },
    ]
    cmp_df = pd.DataFrame(rows, columns=["analysis", "with_temporal",
                                         "without_temporal", "change"])

    # Append OLD z_family reference rows (visual side-by-side only).
    old_tbl = load_old_sensitivity_table()
    old_rows = []
    if old_tbl is not None:
        # Old table columns: analysis, with_temporal, without_temporal, change
        # Tag each row with a block marker for readability.
        for _, r in old_tbl.iterrows():
            old_rows.append({
                "analysis": f"[OLD z_family R4] {r['analysis']}",
                "with_temporal": r["with_temporal"],
                "without_temporal": r["without_temporal"],
                "change": r["change"],
            })
    if old_rows:
        # Separator row + old rows.
        sep_row = {
            "analysis": "=== OLD R4 REFERENCE (z_family, pre-R1) ===",
            "with_temporal": "", "without_temporal": "", "change": "",
        }
        cmp_df = pd.concat(
            [cmp_df, pd.DataFrame([sep_row] + old_rows)], ignore_index=True
        )

    cmp_df.to_csv(OUT_COMPARISON, index=False)
    print(f"\n[comparison] wrote {OUT_COMPARISON}")
    with pd.option_context("display.max_colwidth", 80, "display.width", 220):
        print(cmp_df.to_string(index=False))

    # --- 7. Narrative ---
    p_new = inter_new["p_interaction"]
    z_p_new = diff_new["lrt_Z_vs_Y"]["p"]
    p_stable = np.isfinite(p_new) and p_new < 0.05
    z_stable = np.isfinite(z_p_new) and z_p_new < 0.05

    # Did dropping temporal IMPROVE interaction p (smaller / more significant)?
    delta_p = p_new - WITH_TEMPORAL_UNIFIED["interaction_p"]
    improved = p_new < WITH_TEMPORAL_UNIFIED["interaction_p"]

    if p_stable and z_stable and flipped_sig.empty and flipped_sign.empty:
        verdict = "PASS"
        improvement_clause = (
            f"dropping the two subsystems {'further lowered' if improved else 'raised'} "
            f"the interaction p from {_fmt_p(WITH_TEMPORAL_UNIFIED['interaction_p'])} "
            f"to {_fmt_p(p_new)} (Δp={delta_p:+.3e})"
        )
        narrative = (
            f"Removing the two small-sample PetShop `temporal_traffic` "
            f"subsystems (n=8 each) from the R1 unified 11-system block "
            f"leaves the headline statistical conclusions intact: the "
            f"`method × system` interaction remains significant ({improvement_clause}), "
            f"and the residual system heterogeneity beyond difficulty "
            f"covariates remains significant "
            f"(LRT χ² = {diff_new['lrt_Z_vs_Y']['stat']:.2f}, "
            f"df = {diff_new['lrt_Z_vs_Y']['df']}, p = {_fmt_p(z_p_new)}; "
            f"unified-block ref with-temporal: "
            f"χ² = {diff_ref['lrt_Z_vs_Y']['stat']:.2f}, "
            f"p = {_fmt_p(diff_ref['lrt_Z_vs_Y']['p'])}). "
            f"Per-system BARO vs zbase_univ paired bootstraps on the 9 "
            f"retained systems are unchanged by construction (paired "
            f"resamples are system-local and seed-identical), and no "
            f"retained pair changed its CI significance status or "
            f"mean-delta sign."
        )
    else:
        warn_bits = []
        if not p_stable:
            warn_bits.append(
                f"interaction p goes from "
                f"{_fmt_p(WITH_TEMPORAL_UNIFIED['interaction_p'])} to {_fmt_p(p_new)}"
            )
        if not z_stable:
            warn_bits.append(
                f"difficulty LRT(Z vs Y) p goes from "
                f"{_fmt_p(diff_ref['lrt_Z_vs_Y']['p'])} to {_fmt_p(z_p_new)} "
                f"(χ² {diff_ref['lrt_Z_vs_Y']['stat']:.2f} → "
                f"{diff_new['lrt_Z_vs_Y']['stat']:.2f})"
            )
        if not flipped_sig.empty:
            syss = ", ".join(flipped_sig["system"].tolist())
            warn_bits.append(f"CI significance flipped on: {syss}")
        if not flipped_sign.empty:
            syss = ", ".join(flipped_sign["system"].tolist())
            warn_bits.append(f"mean-delta sign flipped on: {syss}")
        verdict = "WARN"
        narrative = (
            "WARNING: headline claims (UNIFIED R1) are sensitive to "
            "temporal_traffic inclusion. " + "; ".join(warn_bits) + "."
        )

    with open(OUT_NARRATIVE, "w") as f:
        f.write(narrative + "\n")
    print(f"\n[narrative] wrote {OUT_NARRATIVE}  (verdict: {verdict})")
    print(narrative)

    # --- 8. Raw JSON dump ---
    with open(OUT_RAW, "w") as f:
        json.dump({
            "with_temporal_unified_reference": WITH_TEMPORAL_UNIFIED,
            "old_z_family_reference": OLD_Z_FAMILY_REFERENCE,
            "dropped_systems": sorted(DROP_SYSTEMS),
            "kept_systems": kept_systems,
            "interaction_with_temporal": inter_ref,
            "interaction_without_temporal": inter_new,
            "difficulty_with_temporal": diff_ref,
            "difficulty_without_temporal": diff_new,
            "view_accuracy": view_tbl.to_dict(orient="records"),
            "bootstrap_with_temporal_all11": boot_all11.to_dict(orient="records"),
            "bootstrap_without_temporal_9": boot_new.to_dict(orient="records"),
            "sig_flips": flipped_sig.to_dict(orient="records"),
            "sign_flips": flipped_sign.to_dict(orient="records"),
            "mean_ci_width_pp_with": mean_width_with,
            "mean_ci_width_pp_without": mean_width_without,
            "delta_interaction_p_vs_with": float(delta_p),
            "interaction_p_improved": bool(improved),
            "verdict": verdict,
        }, f, indent=2, default=float)
    print(f"[raw] wrote {OUT_RAW}")


if __name__ == "__main__":
    main()
