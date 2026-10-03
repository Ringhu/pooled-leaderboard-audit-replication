#!/usr/bin/env python3
"""Sensitivity analysis: drop PetShop temporal_traffic{1,2} subsystems.

Both temporal_traffic subsystems have n_cases=8 (smallest in the block) and
are suspected carriers of the method x system interaction signal due to
their large per-case effect sizes in the bootstrap table. This driver
re-runs the three headline analyses on the 9-system subset and writes a
side-by-side comparison to results_stats/sensitivity_temporal_traffic.csv.

Reproducibility: bootstrap seed=42 (same as bootstrap_pairwise.main()).

Outputs
-------
  results_stats/sensitivity_temporal_traffic.csv
      Rows = analysis type, columns = [with_temporal, without_temporal, change]
  results_stats/sensitivity_narrative.md
      One-paragraph robustness narrative (PASS or WARN).
  results_stats/sensitivity_view_accuracy_no_temporal.csv
      Filtered per-view accuracy table (9 rows — one per retained system).
  results_stats/sensitivity_bootstrap_no_temporal.csv
      Filtered headline baro vs z_family bootstrap CIs (9 rows).

Constraints
-----------
* Does NOT modify existing interaction_test.py / bootstrap_pairwise.py /
  interaction_test_difficulty.py. Re-implements the same fits on the
  filtered frame, importing only the data loader and helpers.
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

from bootstrap_pairwise import load_all_systems, paired_bootstrap
from difficulty_indicators import compute_all_systems

ROOT = Path("$HOME/auditstack")
OUT_DIR = ROOT / "results_stats"
OUT_COMPARISON = OUT_DIR / "sensitivity_temporal_traffic.csv"
OUT_NARRATIVE = OUT_DIR / "sensitivity_narrative.md"
OUT_VIEW = OUT_DIR / "sensitivity_view_accuracy_no_temporal.csv"
OUT_BOOT = OUT_DIR / "sensitivity_bootstrap_no_temporal.csv"

# Existing (with-temporal) headline numbers — hard-coded from the current
# artifacts so the comparison is self-contained. Verified 2026-04-20 against:
#   results_stats/interaction_test.json        (p_interaction, n_obs)
#   results_stats/interaction_test_difficulty.json (llf_Z, llf_Y, LRT)
WITH_TEMPORAL = {
    "interaction_p": 8.624715295846297e-04,
    "difficulty_LRT_Z_vs_Y_chi2": 143.85056639797176,
    "difficulty_LRT_Z_vs_Y_p": 0.0,  # reported as 0.0 (< machine eps for chi2 sf)
    "n_systems": 11,
    "n_obs": 1556,
}

# Systems dropped in the sensitivity analysis.
DROP_SYSTEMS = {"petshop::temporal_traffic1", "petshop::temporal_traffic2"}


# ------------------------------------------------------------------ #
# Fit helpers (mirrors interaction_test.py / interaction_test_difficulty.py)
# ------------------------------------------------------------------ #
def _fit_glm(df: pd.DataFrame, formula: str):
    return smf.glm(
        formula, data=df, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": df["case_id"]})


def _lrt(ll_big: float, ll_small: float, df_diff: int) -> dict:
    stat = 2.0 * (ll_big - ll_small)
    p = float(1.0 - chi2.cdf(stat, df_diff)) if df_diff > 0 else float("nan")
    return {"stat": float(stat), "df": int(df_diff), "p": p}


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    if p == 0.0:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.3e}"
    return f"{p:.4f}"


def _fmt_change(old: float, new: float) -> str:
    if not (np.isfinite(old) and np.isfinite(new)):
        return "n/a"
    return f"{new - old:+.4g}"


# ------------------------------------------------------------------ #
# Core analyses on filtered frame                                    #
# ------------------------------------------------------------------ #
def run_interaction_test(sub: pd.DataFrame) -> dict:
    """Mirror interaction_test.py on the filtered frame."""
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
    """Mirror interaction_test_difficulty.py Z-vs-Y LRT."""
    diff_table = compute_all_systems()
    diff_table["system_id"] = diff_table["benchmark"].astype(str) + "::" + diff_table["system"].astype(str)
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
    """Per-system acc@1 for BARO and z_family on the filtered block."""
    g = sub.groupby(["system_id", "method"]).agg(
        n=("correct", "size"),
        acc1=("correct", "mean"),
    ).reset_index()
    piv = g.pivot_table(index="system_id", columns="method", values="acc1").reset_index()
    piv.columns.name = None
    return piv


def run_bootstrap(df_all: pd.DataFrame, systems_kept: list[str],
                  n_iter: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Headline baro vs z_family bootstrap CIs on each remaining system."""
    rows = []
    for system in sorted(systems_kept):
        res = paired_bootstrap(
            df_all, "baro", "z_family", system, n_iter=n_iter, seed=seed
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


# ------------------------------------------------------------------ #
# With-temporal bootstrap CI widths (for comparison row).            #
# Loaded from the frozen pairwise_bootstrap.csv — do not re-run.     #
# ------------------------------------------------------------------ #
def load_with_temporal_bootstrap() -> pd.DataFrame:
    path = ROOT / "results_stats" / "pairwise_bootstrap.csv"
    df = pd.read_csv(path)
    mask = (df["method_a"] == "baro") & (df["method_b"] == "z_family")
    out = df[mask].copy()
    out["ci_width_pp"] = 100.0 * (out["ci_high"] - out["ci_low"])
    return out


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main():
    print("[load] reading 11-system long frame ...")
    df = load_all_systems()
    print(f"       n_rows={len(df)}  n_systems={df['system_id'].nunique()}")

    # Restrict to BARO + z_family (the paper's complete 2x11 block).
    sub_all = df[df["method"].isin(["baro", "z_family"])].copy()
    sub_all["correct"] = sub_all["correct"].clip(0.0, 1.0).astype(float)

    # --- Filtered frame: drop the two temporal_traffic subsystems ---
    sub_filt = sub_all[~sub_all["system_id"].isin(DROP_SYSTEMS)].copy()
    kept_systems = sorted(sub_filt["system_id"].unique())
    print(f"[filter] kept {len(kept_systems)} systems: {kept_systems}")
    print(f"[filter] n_obs: with_temporal={len(sub_all)}  "
          f"without_temporal={len(sub_filt)}")

    # --- 1. Interaction test (fractional logit LRT) on filtered frame ---
    print("\n[interaction] fitting GLM-Binomial on filtered frame ...")
    inter = run_interaction_test(sub_filt)
    print(f"  n_obs         = {inter['n_obs']}")
    print(f"  n_systems     = {inter['n_systems']}")
    print(f"  llf_full      = {inter['llf_full']:.4f}  ({inter['df_full_params']} params)")
    print(f"  llf_reduced   = {inter['llf_reduced']:.4f}  ({inter['df_reduced_params']} params)")
    print(f"  LRT stat      = {inter['lrt_statistic']:.4f}  df={inter['df_diff']}")
    print(f"  p_interaction = {_fmt_p(inter['p_interaction'])}")

    # --- 2. Difficulty LRT(Z vs Y) on filtered frame ---
    print("\n[difficulty] fitting X/Y/Z on filtered frame ...")
    diff = run_difficulty_test(sub_filt)
    print(f"  llf_X = {diff['llf_X']:.4f}  (params={diff['n_params_X']})")
    print(f"  llf_Y = {diff['llf_Y']:.4f}  (params={diff['n_params_Y']})")
    print(f"  llf_Z = {diff['llf_Z']:.4f}  (params={diff['n_params_Z']})")
    print(f"  LRT Z vs Y = {diff['lrt_Z_vs_Y']['stat']:.4f}  "
          f"df={diff['lrt_Z_vs_Y']['df']}  p={_fmt_p(diff['lrt_Z_vs_Y']['p'])}")
    print(f"  LRT Y vs X = {diff['lrt_Y_vs_X']['stat']:.4f}  "
          f"df={diff['lrt_Y_vs_X']['df']}  p={_fmt_p(diff['lrt_Y_vs_X']['p'])}")

    # --- 3. Filtered view accuracy ---
    view_tbl = run_view_accuracy(sub_filt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    view_tbl.to_csv(OUT_VIEW, index=False)
    print(f"\n[view] wrote {OUT_VIEW}")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 160,
                           "display.max_columns", 20):
        print(view_tbl.to_string(index=False))

    # --- 4. Filtered headline BARO vs z_family bootstrap (9 systems) ---
    print("\n[bootstrap] running paired bootstrap on 9 retained systems ...")
    boot_new = run_bootstrap(df, kept_systems, n_iter=10_000, seed=42)
    boot_new.to_csv(OUT_BOOT, index=False)
    print(f"[bootstrap] wrote {OUT_BOOT}")
    with pd.option_context("display.float_format", "{:.5f}".format,
                           "display.width", 200,
                           "display.max_columns", 20):
        print(boot_new.to_string(index=False))

    # --- 5. Compare CIs: did any pair flip significance direction? ---
    boot_old = load_with_temporal_bootstrap()
    boot_old_f = boot_old[~boot_old["system"].isin(DROP_SYSTEMS)].copy()
    boot_old_f = boot_old_f.rename(columns={
        "mean_delta": "mean_delta_old",
        "ci_low": "ci_low_old",
        "ci_high": "ci_high_old",
        "ci_width_pp": "ci_width_pp_old",
        "p_two_sided": "p_two_sided_old",
    })
    merged_boot = boot_new.merge(
        boot_old_f[["system", "mean_delta_old", "ci_low_old", "ci_high_old",
                    "ci_width_pp_old", "p_two_sided_old"]],
        on="system", how="inner",
    )

    # Significance at alpha=0.05 via CI excluding zero.
    def sig_from_ci(lo, hi):
        if lo > 0 or hi < 0:
            return True
        return False

    merged_boot["sig_old"] = merged_boot.apply(
        lambda r: sig_from_ci(r["ci_low_old"], r["ci_high_old"]), axis=1)
    merged_boot["sig_new"] = merged_boot.apply(
        lambda r: sig_from_ci(r["ci_low"], r["ci_high"]), axis=1)
    flipped_sig = merged_boot[merged_boot["sig_old"] != merged_boot["sig_new"]].copy()
    # Sign flips on the point estimate itself.
    def sign(x): return 1 if x > 1e-9 else (-1 if x < -1e-9 else 0)
    merged_boot["sign_old"] = merged_boot["mean_delta_old"].apply(sign)
    merged_boot["sign_new"] = merged_boot["mean_delta"].apply(sign)
    flipped_sign = merged_boot[merged_boot["sign_old"] != merged_boot["sign_new"]].copy()

    print("\n[bootstrap-compare] per-system deltas (with vs without temporal):")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 220,
                           "display.max_columns", 20):
        print(merged_boot[[
            "system", "n_cases",
            "mean_delta_old", "mean_delta",
            "ci_width_pp_old", "ci_width_pp",
            "sig_old", "sig_new",
        ]].to_string(index=False))
    if not flipped_sig.empty:
        print("[bootstrap-compare] WARNING: significance flipped for:")
        print(flipped_sig[["system", "ci_low_old", "ci_high_old", "ci_low", "ci_high"]].to_string(index=False))
    else:
        print("[bootstrap-compare] No CI significance flipped on retained systems.")
    if not flipped_sign.empty:
        print("[bootstrap-compare] WARNING: mean-delta sign flipped for:")
        print(flipped_sign[["system", "mean_delta_old", "mean_delta"]].to_string(index=False))

    mean_width_old = float(boot_old_f["ci_width_pp_old"].mean())
    mean_width_new = float(boot_new["ci_width_pp"].mean())

    # --- 6. Side-by-side CSV ---
    rows = [
        {
            "analysis": "interaction_p",
            "with_temporal": f"{WITH_TEMPORAL['interaction_p']:.3e}",
            "without_temporal": _fmt_p(inter["p_interaction"]),
            "change": _fmt_change(WITH_TEMPORAL["interaction_p"], inter["p_interaction"]),
        },
        {
            "analysis": "difficulty_LRT_Z_vs_Y_chi2",
            "with_temporal": f"{WITH_TEMPORAL['difficulty_LRT_Z_vs_Y_chi2']:.4f}",
            "without_temporal": f"{diff['lrt_Z_vs_Y']['stat']:.4f}",
            "change": _fmt_change(
                WITH_TEMPORAL["difficulty_LRT_Z_vs_Y_chi2"],
                diff["lrt_Z_vs_Y"]["stat"],
            ),
        },
        {
            "analysis": "difficulty_LRT_Z_vs_Y_p",
            "with_temporal": _fmt_p(WITH_TEMPORAL["difficulty_LRT_Z_vs_Y_p"]),
            "without_temporal": _fmt_p(diff["lrt_Z_vs_Y"]["p"]),
            "change": _fmt_change(
                WITH_TEMPORAL["difficulty_LRT_Z_vs_Y_p"],
                diff["lrt_Z_vs_Y"]["p"],
            ),
        },
        {
            "analysis": "baro_vs_zfamily_mean_CI_width_pp",
            "with_temporal": f"{mean_width_old:.3f}",
            "without_temporal": f"{mean_width_new:.3f}",
            "change": _fmt_change(mean_width_old, mean_width_new),
        },
        {
            "analysis": "n_systems",
            "with_temporal": str(WITH_TEMPORAL["n_systems"]),
            "without_temporal": str(inter["n_systems"]),
            "change": str(inter["n_systems"] - WITH_TEMPORAL["n_systems"]),
        },
        {
            "analysis": "n_obs",
            "with_temporal": str(WITH_TEMPORAL["n_obs"]),
            "without_temporal": str(inter["n_obs"]),
            "change": str(inter["n_obs"] - WITH_TEMPORAL["n_obs"]),
        },
    ]
    cmp_df = pd.DataFrame(rows, columns=["analysis", "with_temporal",
                                         "without_temporal", "change"])
    cmp_df.to_csv(OUT_COMPARISON, index=False)
    print(f"\n[comparison] wrote {OUT_COMPARISON}")
    print(cmp_df.to_string(index=False))

    # --- 7. Narrative ---
    # Decision rule: PASS if p_interaction_new < 0.05 AND Z_vs_Y_p_new < 0.05
    # AND no sign flips on retained systems. Otherwise WARN.
    p_new = inter["p_interaction"]
    z_p_new = diff["lrt_Z_vs_Y"]["p"]
    p_stable = np.isfinite(p_new) and p_new < 0.05
    z_stable = np.isfinite(z_p_new) and z_p_new < 0.05
    # Per-system bootstraps are independent across systems (paired resample
    # within each system_id, seeded identically), so retained-system CIs are
    # unchanged by definition — the sensitivity is what the cross-system
    # tests (interaction, difficulty LRT) say. We report the per-system CI
    # stability as a separate check rather than a "widen/tighten" narrative.
    ci_width_delta_mean = float(np.mean(np.abs(
        boot_new["ci_width_pp"].to_numpy() - boot_old_f["ci_width_pp_old"].to_numpy()
    )))

    if p_stable and z_stable and flipped_sig.empty and flipped_sign.empty:
        verdict = "PASS"
        narrative = (
            f"Removing the two small-sample PetShop `temporal_traffic` "
            f"subsystems (n=8 each) from the 11-system block leaves the main "
            f"statistical conclusions intact. The `method × system` "
            f"interaction remains significant (p = {_fmt_p(p_new)}; "
            f"was {_fmt_p(WITH_TEMPORAL['interaction_p'])}), and the residual "
            f"system heterogeneity beyond difficulty covariates remains "
            f"significant (LRT χ² = {diff['lrt_Z_vs_Y']['stat']:.2f}, "
            f"df = {diff['lrt_Z_vs_Y']['df']}, p = {_fmt_p(z_p_new)}; "
            f"was χ² = {WITH_TEMPORAL['difficulty_LRT_Z_vs_Y_chi2']:.2f}). "
            f"Per-system BARO vs z_family bootstrap CIs on the 9 retained "
            f"systems are unchanged by construction (paired resamples are "
            f"system-local; mean |Δwidth| = {ci_width_delta_mean:.3f}pp), "
            f"and no retained pair changed its CI significance status or "
            f"mean-delta sign. Headline claims do not rest on the "
            f"temporal_traffic subsystems."
        )
    else:
        warn_bits = []
        if not p_stable:
            warn_bits.append(
                f"interaction p goes from "
                f"{_fmt_p(WITH_TEMPORAL['interaction_p'])} to {_fmt_p(p_new)}"
            )
        if not z_stable:
            warn_bits.append(
                f"difficulty LRT(Z vs Y) p goes from "
                f"{_fmt_p(WITH_TEMPORAL['difficulty_LRT_Z_vs_Y_p'])} "
                f"to {_fmt_p(z_p_new)} (χ² "
                f"{WITH_TEMPORAL['difficulty_LRT_Z_vs_Y_chi2']:.2f} → "
                f"{diff['lrt_Z_vs_Y']['stat']:.2f})"
            )
        if not flipped_sig.empty:
            syss = ", ".join(flipped_sig["system"].tolist())
            warn_bits.append(f"CI significance flipped on: {syss}")
        if not flipped_sign.empty:
            syss = ", ".join(flipped_sign["system"].tolist())
            warn_bits.append(f"mean-delta sign flipped on: {syss}")
        verdict = "WARN"
        narrative = (
            "⚠️ WARNING: headline claims are sensitive to temporal_traffic "
            "inclusion. " + "; ".join(warn_bits) + ". Re-examine F1/F2 framing."
        )

    with open(OUT_NARRATIVE, "w") as f:
        f.write(narrative + "\n")
    print(f"\n[narrative] wrote {OUT_NARRATIVE}  (verdict: {verdict})")
    print(narrative)

    # --- 8. JSON dump (full raw numbers for transparency) ---
    raw_dump = OUT_DIR / "sensitivity_no_temporal_raw.json"
    with open(raw_dump, "w") as f:
        json.dump({
            "with_temporal_reference": WITH_TEMPORAL,
            "dropped_systems": sorted(DROP_SYSTEMS),
            "kept_systems": kept_systems,
            "interaction": inter,
            "difficulty": diff,
            "view_accuracy": view_tbl.to_dict(orient="records"),
            "bootstrap_no_temporal": boot_new.to_dict(orient="records"),
            "bootstrap_with_temporal_same_9": boot_old_f.to_dict(orient="records"),
            "sig_flips": flipped_sig.to_dict(orient="records"),
            "sign_flips": flipped_sign.to_dict(orient="records"),
            "mean_ci_width_pp_with": mean_width_old,
            "mean_ci_width_pp_without": mean_width_new,
            "verdict": verdict,
        }, f, indent=2, default=float)
    print(f"[raw] wrote {raw_dump}")


if __name__ == "__main__":
    main()
