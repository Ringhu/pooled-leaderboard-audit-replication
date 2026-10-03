#!/usr/bin/env python3
"""F6-C Rescue — higher-power tests on the OpenRCA-only 3-method block.

Original F6-C: 6-df omnibus LRT on {baro, zbase, zbase_univ} x 4 OpenRCA
subsystems gave p = 0.253. Inadequate power because 6 df spread across
~335 fractional cases.

This rescue replaces the omnibus with 3 alternatives that directly target
the "Bank flip" hypothesis:

  R1. 1-df targeted contrast: test whether (zbase - zbase_univ) is
      significantly different on Bank vs. the mean of the other 3 systems.
      Adds a single Bank × (zbase-vs-univ) interaction term to a
      GLM-Binomial, tests its coefficient via Wald.

  R2. Permutation test for the 6-df interaction. Permute case_id labels
      (preserving within-case method triple + within-system case counts)
      to break system × method linkage; recompute observed LRT 5000 times;
      report empirical p = fraction ≥ observed.

  R3. Paired bootstrap CI on the focal contrast Δ_Bank − Δ_others, where
      Δ_s = mean(zbase on s) − mean(zbase_univ on s). Case-level resampling
      within each system, 5000 iters, seed=42. If 95% CI excludes 0, the
      flip is empirically robust.

Run on 3090:
  python scripts/openrca_eval/stress_test_F6C_rescue.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stress_test_F6 import load_openrca_triple  # noqa: E402

ROOT = Path("$HOME/auditstack")


def _fit_glm(df, formula):
    return smf.glm(formula, data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df["case_id"]}
    )


# ---------------- R1: 1-df targeted contrast ---------------- #
def rescue_R1_targeted_contrast(df: pd.DataFrame) -> dict:
    """Test the specific contrast: (zbase - zbase_univ) on Bank vs others."""
    # Keep only zbase and zbase_univ (2 methods, 4 systems) for a clean
    # 1-df interaction test on the focal contrast.
    sub = df[df["method"].isin(["zbase", "zbase_univ"])].copy()
    sub["is_bank"] = (sub["system_id"] == "openrca::bank").astype(int)
    sub["is_legacy"] = (sub["method"] == "zbase").astype(int)
    sub["correct"] = sub["correct"].clip(0, 1).astype(float)

    # Formula: correct ~ method + system + (is_bank * is_legacy)
    # The interaction is_bank*is_legacy is the 1-df contrast.
    # Using system_id instead of is_bank alone to keep all system main effects.
    formula_full = "correct ~ C(method) + C(system_id) + is_bank:is_legacy"
    formula_red = "correct ~ C(method) + C(system_id)"
    m_full = _fit_glm(sub, formula_full)
    m_red = _fit_glm(sub, formula_red)

    # LRT on 1 added parameter.
    lrt = 2.0 * (m_full.llf - m_red.llf)
    df_diff = len(m_full.params) - len(m_red.params)
    p_lrt = float(1.0 - chi2.cdf(lrt, df_diff))

    # Wald on the bank:legacy coefficient.
    bl_name = [n for n in m_full.params.index if "is_bank:is_legacy" in n]
    if len(bl_name) == 1:
        b = bl_name[0]
        coef = float(m_full.params[b])
        se = float(m_full.bse[b])
        z = coef / se
        p_wald = float(m_full.pvalues[b])
    else:
        coef = se = z = p_wald = float("nan")

    return {
        "design": "2 methods (zbase, zbase_univ) x 4 systems, with 1-df bank×legacy contrast",
        "n_obs": int(len(sub)),
        "n_cases": int(sub["case_id"].nunique()),
        "contrast_name": "is_bank:is_legacy (= [(zbase-zbase_univ)|Bank] - [(zbase-zbase_univ)|others])",
        "coef_logit_scale": coef,
        "se": se,
        "z": z,
        "p_wald": p_wald,
        "lrt_stat": float(lrt),
        "df_lrt": int(df_diff),
        "p_lrt": p_lrt,
        "llf_full": float(m_full.llf),
        "llf_reduced": float(m_red.llf),
    }


# ---------------- R2: Permutation test for 6-df interaction ---------------- #
def observed_lrt_3method(df: pd.DataFrame) -> tuple:
    sub = df[df["method"].isin(["baro", "zbase", "zbase_univ"])].copy()
    sub["correct"] = sub["correct"].clip(0, 1).astype(float)
    m_full = _fit_glm(sub, "correct ~ C(method) * C(system_id)")
    m_red = _fit_glm(sub, "correct ~ C(method) + C(system_id)")
    lrt = 2.0 * (m_full.llf - m_red.llf)
    df_diff = len(m_full.params) - len(m_red.params)
    return float(lrt), int(df_diff)


def rescue_R2_permutation(df: pd.DataFrame, n_perm: int = 2000,
                          seed: int = 42) -> dict:
    """Permute case_id -> system assignments within each method (equivalently,
    permute the system_id label across cases while preserving per-method
    accuracy distribution). Under H0 of no method×system interaction,
    systems can be swapped.

    Practically: for each case_id (which has 3 rows, one per method),
    relabel its system_id by uniform random sampling from the empirical
    system distribution (weighted by system case counts, preserving the
    marginal).
    """
    lrt_obs, df_int = observed_lrt_3method(df)
    print(f"  observed LRT = {lrt_obs:.3f}  df = {df_int}")

    sub = df[df["method"].isin(["baro", "zbase", "zbase_univ"])].copy()
    sub["correct"] = sub["correct"].clip(0, 1).astype(float)

    # Build a case_id -> system_id map.
    case_to_sys = sub.drop_duplicates("case_id").set_index("case_id")["system_id"]
    systems = sorted(sub["system_id"].unique())
    system_counts = sub.drop_duplicates("case_id")["system_id"].value_counts()
    case_list = list(case_to_sys.index)

    rng = np.random.default_rng(seed)
    n_ge_obs = 0
    perm_lrts = np.zeros(n_perm, dtype=float)

    for it in range(n_perm):
        # Permute system labels across cases (break case-system link while
        # preserving system marginal counts). We shuffle the existing
        # system labels among cases.
        perm_systems = np.array(case_to_sys.values, copy=True)
        rng.shuffle(perm_systems)
        perm_map = dict(zip(case_list, perm_systems))
        sub_perm = sub.copy()
        sub_perm["system_id_perm"] = sub_perm["case_id"].map(perm_map)

        try:
            m_full = smf.glm(
                "correct ~ C(method) * C(system_id_perm)",
                data=sub_perm, family=sm.families.Binomial()
            ).fit(cov_type="cluster", cov_kwds={"groups": sub_perm["case_id"]})
            m_red = smf.glm(
                "correct ~ C(method) + C(system_id_perm)",
                data=sub_perm, family=sm.families.Binomial()
            ).fit(cov_type="cluster", cov_kwds={"groups": sub_perm["case_id"]})
            lrt_p = 2.0 * (m_full.llf - m_red.llf)
        except Exception as e:
            lrt_p = float("nan")

        perm_lrts[it] = lrt_p
        if not np.isnan(lrt_p) and lrt_p >= lrt_obs:
            n_ge_obs += 1

        if (it + 1) % 200 == 0:
            print(f"    perm {it+1}/{n_perm}  running p_perm = "
                  f"{(n_ge_obs / (it+1)):.4f}")

    valid = ~np.isnan(perm_lrts)
    p_perm = float(n_ge_obs / int(valid.sum()))
    return {
        "observed_lrt": lrt_obs,
        "df_interaction": df_int,
        "n_perm": n_perm,
        "n_perm_valid": int(valid.sum()),
        "n_perm_ge_obs": n_ge_obs,
        "p_perm": p_perm,
        "perm_lrt_summary": {
            "mean": float(np.nanmean(perm_lrts)),
            "median": float(np.nanmedian(perm_lrts)),
            "p95": float(np.nanpercentile(perm_lrts, 95)),
            "p99": float(np.nanpercentile(perm_lrts, 99)),
            "max": float(np.nanmax(perm_lrts)),
        },
    }


# ---------------- R3: Bootstrap CI on focal contrast ---------------- #
def rescue_R3_bootstrap_contrast(df: pd.DataFrame, n_iter: int = 5000,
                                 seed: int = 42) -> dict:
    """Compute (Δ_Bank - Δ_others) where Δ_s = mean(zbase) - mean(zbase_univ)
    per system. Bootstrap case-level within each system.
    """
    sub = df[df["method"].isin(["zbase", "zbase_univ"])].copy()
    sub["correct"] = sub["correct"].clip(0, 1).astype(float)

    systems = sorted(sub["system_id"].unique())
    # Build per-system case-level vectors for each method.
    per_sys_per_method = {}
    for s in systems:
        ss = sub[sub["system_id"] == s]
        legacy = ss[ss["method"] == "zbase"].groupby("case_id")["correct"].mean().to_numpy()
        univ = ss[ss["method"] == "zbase_univ"].groupby("case_id")["correct"].mean().to_numpy()
        per_sys_per_method[s] = {"legacy": legacy, "univ": univ}

    # Observed contrast.
    delta_obs = {s: per_sys_per_method[s]["legacy"].mean()
                    - per_sys_per_method[s]["univ"].mean()
                 for s in systems}
    delta_bank = delta_obs["openrca::bank"]
    delta_others_mean = float(np.mean([delta_obs[s] for s in systems
                                        if s != "openrca::bank"]))
    contrast_obs = delta_bank - delta_others_mean

    print(f"  Δ_Bank = {delta_bank*100:.3f} pp")
    for s in systems:
        if s != "openrca::bank":
            print(f"  Δ_{s} = {delta_obs[s]*100:.3f} pp")
    print(f"  Δ_others (mean) = {delta_others_mean*100:.3f} pp")
    print(f"  contrast (Δ_Bank - Δ_others) = {contrast_obs*100:.3f} pp")

    rng = np.random.default_rng(seed)
    contrasts = np.zeros(n_iter)
    delta_bank_boots = np.zeros(n_iter)
    delta_others_boots = np.zeros(n_iter)
    for it in range(n_iter):
        boot_delta = {}
        for s in systems:
            L = per_sys_per_method[s]["legacy"]
            U = per_sys_per_method[s]["univ"]
            li = rng.integers(0, L.size, size=L.size)
            ui = rng.integers(0, U.size, size=U.size)
            boot_delta[s] = L[li].mean() - U[ui].mean()
        delta_bank_boots[it] = boot_delta["openrca::bank"]
        delta_others_boots[it] = float(np.mean([
            boot_delta[s] for s in systems if s != "openrca::bank"
        ]))
        contrasts[it] = delta_bank_boots[it] - delta_others_boots[it]

    ci = np.quantile(contrasts, [0.025, 0.5, 0.975])
    p_boot = float((contrasts <= 0).mean())  # one-sided: P(contrast <= 0)

    return {
        "delta_bank_pp": delta_bank * 100,
        "delta_others_mean_pp": delta_others_mean * 100,
        "observed_contrast_pp": contrast_obs * 100,
        "per_system_delta_pp": {s: delta_obs[s]*100 for s in systems},
        "bootstrap": {
            "n_iter": n_iter,
            "contrast_ci_lo_pp": float(ci[0] * 100),
            "contrast_median_pp": float(ci[1] * 100),
            "contrast_ci_hi_pp": float(ci[2] * 100),
            "p_boot_one_sided": p_boot,
            "delta_bank_ci_pp": [
                float(np.quantile(delta_bank_boots, 0.025) * 100),
                float(np.quantile(delta_bank_boots, 0.975) * 100),
            ],
            "delta_others_ci_pp": [
                float(np.quantile(delta_others_boots, 0.025) * 100),
                float(np.quantile(delta_others_boots, 0.975) * 100),
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results_stats" / "stress_test_F6")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[load] OpenRCA 3-method triple block")
    df = load_openrca_triple()
    print(f"       n_obs = {len(df)}, n_cases = {df['case_id'].nunique()}")
    print(f"       methods = {sorted(df['method'].unique())}")
    print(f"       systems = {sorted(df['system_id'].unique())}")

    print("\n" + "=" * 70)
    print("F6-C-R1: 1-df targeted contrast (bank × legacy-vs-univ)")
    print("=" * 70)
    r1 = rescue_R1_targeted_contrast(df)
    print(f"  LRT = {r1['lrt_stat']:.3f}  df = {r1['df_lrt']}  "
          f"p_LRT = {r1['p_lrt']:.3e}")
    print(f"  contrast coef = {r1['coef_logit_scale']:.4f}  SE = {r1['se']:.4f}")
    print(f"  z = {r1['z']:.3f}  p_Wald = {r1['p_wald']:.3e}")

    print("\n" + "=" * 70)
    print(f"F6-C-R2: Permutation test for 6-df interaction (n_perm={args.n_perm})")
    print("=" * 70)
    r2 = rescue_R2_permutation(df, n_perm=args.n_perm, seed=args.seed)
    print(f"  observed LRT = {r2['observed_lrt']:.3f}")
    print(f"  permutation p = {r2['p_perm']:.4f} "
          f"({r2['n_perm_ge_obs']}/{r2['n_perm_valid']})")

    print("\n" + "=" * 70)
    print(f"F6-C-R3: Bootstrap CI on Δ_Bank − Δ_others (n_boot={args.n_boot})")
    print("=" * 70)
    r3 = rescue_R3_bootstrap_contrast(df, n_iter=args.n_boot, seed=args.seed)
    b = r3["bootstrap"]
    print(f"  observed contrast = {r3['observed_contrast_pp']:.3f} pp")
    print(f"  95% CI = [{b['contrast_ci_lo_pp']:.3f}, {b['contrast_ci_hi_pp']:.3f}] pp")
    print(f"  P(contrast <= 0) = {b['p_boot_one_sided']:.4f}")

    summary = {"R1_targeted_contrast": r1,
               "R2_permutation": r2,
               "R3_bootstrap_contrast": r3}

    with open(args.output_dir / "F6C_rescue.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[written] {args.output_dir / 'F6C_rescue.json'}")

    # Final verdict.
    print("\n=== F6-C rescue verdict ===")
    p_thresholds = [
        ("R1 targeted LRT p", r1["p_lrt"]),
        ("R1 targeted Wald p", r1["p_wald"]),
        ("R2 permutation p", r2["p_perm"]),
    ]
    for name, p in p_thresholds:
        mark = "PASS" if (p < 0.05) else ("MARGINAL" if (p < 0.10) else "FAIL")
        print(f"  {name:<30s} = {p:.4f}   {mark}")
    ci_excludes_zero = (r3["bootstrap"]["contrast_ci_lo_pp"] > 0 or
                        r3["bootstrap"]["contrast_ci_hi_pp"] < 0)
    print(f"  R3 bootstrap CI excludes 0: {ci_excludes_zero}  "
          f"(CI = [{b['contrast_ci_lo_pp']:.3f}, {b['contrast_ci_hi_pp']:.3f}] pp)")


if __name__ == "__main__":
    main()
