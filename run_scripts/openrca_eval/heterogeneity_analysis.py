#!/usr/bin/env python3
"""Stream 1 — System-level heterogeneity meta-analysis for BARO vs z_family.

For each of 11 audited subsystems under the unified block, compute the
per-system paired effect Δ_s = mean(BARO_correct - z_family_correct) and
its sampling variance v_s = Var(d_s) / n_s, where d_s is the vector of
per-case paired differences. Then run a random-effects meta-analysis:

  - Cochran's Q and its p-value
  - I² = max(0, (Q-df)/Q) with df = k-1
  - Between-system variance τ² via Paule-Mandel iterated estimator
  - Fixed-effect pooled estimate Δ_FE
  - Random-effects pooled estimate Δ_RE with Hartung-Knapp (HKSJ) CI
  - 95% prediction interval for a new audited-like system

This operationalises Codex review Round 2 (2026-04-21) recommendation:
complement the case-level interaction LRT with a system-level heterogeneity
analysis to address the unit-of-analysis gap.

Reuse:
  * bootstrap_pairwise.load_all_systems() — loads 11-system unified block
    with collapsed `z_family` label
  * bootstrap_pairwise.paired_bootstrap() — per-system case-level paired
    bootstrap for the CI column in Table 1

Usage (on 3090):
  python scripts/openrca_eval/heterogeneity_analysis.py \
      --inputs-dir results_xbench_11sys_unified/_inputs \
      --output-dir results_stats/heterogeneity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as student_t
from statsmodels.stats.meta_analysis import combine_effects

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bootstrap_pairwise as bp  # noqa: E402


def patch_system_files(inputs_dir: Path) -> list:
    patched = []
    for (benchmark, system, path, acc_col) in bp.SYSTEM_FILES:
        new_path = inputs_dir / path.name
        patched.append((benchmark, system, new_path, acc_col))
    return patched


def load_unified_long(inputs_dir: Path) -> pd.DataFrame:
    patched = patch_system_files(inputs_dir)
    missing = [str(p) for (_, _, p, _) in patched if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(missing))
    original = list(bp.SYSTEM_FILES)
    bp.SYSTEM_FILES = patched
    try:
        df = bp.load_all_systems()
    finally:
        bp.SYSTEM_FILES = original
    return df


def per_system_paired_vectors(df: pd.DataFrame,
                              method_a: str, method_b: str) -> dict:
    """Return dict system_id -> np.array of per-case paired differences d_i =
    correct_A(i) - correct_B(i), with cases matched by case_id (inner join)."""
    out = {}
    for s in sorted(df["system_id"].unique()):
        sub = df[df["system_id"] == s]
        a = (sub[sub["method"] == method_a][["case_id", "correct"]]
                .groupby("case_id", as_index=False)["correct"].mean()
                .rename(columns={"correct": "a"}))
        b = (sub[sub["method"] == method_b][["case_id", "correct"]]
                .groupby("case_id", as_index=False)["correct"].mean()
                .rename(columns={"correct": "b"}))
        wide = a.merge(b, on="case_id", how="inner")
        if len(wide) == 0:
            continue
        out[s] = wide["a"].to_numpy() - wide["b"].to_numpy()
    return out


def per_system_stats(paired: dict,
                     df: pd.DataFrame,
                     method_a: str,
                     method_b: str,
                     bootstrap_n_iter: int = 5000,
                     seed: int = 42) -> pd.DataFrame:
    """Compute Δ_s, v_s, paired-bootstrap CI, n_cases per system."""
    rows = []
    for s in sorted(paired.keys()):
        d = paired[s]
        n = len(d)
        delta = float(d.mean())
        # Sampling variance of the mean = Var(d) / n.
        # Use ddof=1 for unbiased sample variance.
        v = float(np.var(d, ddof=1) / n) if n > 1 else float("nan")
        # Paired-bootstrap CI for the forest plot (reuse existing utility).
        bs = bp.paired_bootstrap(
            df, method_a, method_b, s,
            n_iter=bootstrap_n_iter, seed=seed,
        )
        benchmark = s.split("::", 1)[0]
        baro_advantage_flag = int(delta > 0)
        rows.append({
            "system": s,
            "benchmark": benchmark,
            "n_cases": n,
            "delta_pp": delta * 100.0,
            "var": v,                     # v_s on raw (0..1) scale
            "var_pp": v * 1e4,             # for display (pp² units)
            "se_pp": float(np.sqrt(v)) * 100.0 if not np.isnan(v) else float("nan"),
            "ci_lo_pp": bs["ci_low"] * 100.0,
            "ci_hi_pp": bs["ci_high"] * 100.0,
            "baro_advantage_flag": baro_advantage_flag,
            # Backward-compatible alias for older plotting scripts. This is
            # not the LOSO pooled mis-selection flag used in the paper.
            "flip_flag": baro_advantage_flag,
        })
    return pd.DataFrame(rows)


def meta_analyze(effects: np.ndarray,
                 variances: np.ndarray,
                 row_names=None) -> dict:
    """Combine per-system effects with Paule-Mandel τ² estimation.

    Returns all numbers needed for the paper's Table 2 + §5.2 narrative.
    """
    k = len(effects)
    # Fixed-effect inverse-variance pooling.
    w_fe = 1.0 / variances
    delta_fe = float(np.sum(w_fe * effects) / np.sum(w_fe))
    se_fe = float(np.sqrt(1.0 / np.sum(w_fe)))
    # Cochran's Q around the FE estimate.
    Q = float(np.sum(w_fe * (effects - delta_fe) ** 2))
    df = k - 1
    p_Q = float(1.0 - chi2.cdf(Q, df)) if df > 0 else float("nan")
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0

    # Paule-Mandel / iterated τ² via statsmodels, with Hartung-Knapp CI.
    # use_t=True applies Knapp-Hartung variance correction (HKSJ).
    res_pm = combine_effects(
        effects, variances, method_re="iterated", use_t=True,
        row_names=row_names,
    )
    tau2 = float(res_pm.tau2)
    w_re = 1.0 / (variances + tau2)
    delta_re = float(np.sum(w_re * effects) / np.sum(w_re))

    # HKSJ variance:
    #   var_hk = scale * 1/sum(w_re),
    #   scale = max(1, (1/(k-1)) * Σ w_re_i (Δ_s - Δ_RE)² / (1/sum w_re))
    # statsmodels applies this automatically when use_t=True.
    try:
        hk_ci_low, hk_ci_high = res_pm.conf_int(use_t=True, alpha=0.05)[1]  # row 1 = RE
    except Exception:
        # Fallback: recompute manually from the result object.
        se_re_hk = float(res_pm.sd_eff_w_re_hksj)
        t_crit = student_t.ppf(0.975, df=k - 1)
        hk_ci_low = delta_re - t_crit * se_re_hk
        hk_ci_high = delta_re + t_crit * se_re_hk

    # 95% prediction interval for a new audited-like system:
    #   Δ_RE ± t_{k-2, 0.975} · sqrt(τ² + SE(Δ_RE)²)
    # SE(Δ_RE) under HKSJ is the standard error consistent with hk_ci.
    half_width_re = (hk_ci_high - hk_ci_low) / 2.0
    se_re_for_pi = half_width_re / student_t.ppf(0.975, df=k - 1)
    t_crit_pi = student_t.ppf(0.975, df=k - 2) if k > 2 else float("nan")
    pi_half = t_crit_pi * np.sqrt(tau2 + se_re_for_pi ** 2)
    pi_low = delta_re - pi_half
    pi_high = delta_re + pi_half

    return {
        "k": k,
        "Q": Q,
        "df": df,
        "p_Q": p_Q,
        "I2": I2,
        "tau2": tau2,
        "delta_FE": delta_fe,
        "se_FE": se_fe,
        "delta_RE": delta_re,
        "re_ci_lo_hksj": float(hk_ci_low),
        "re_ci_hi_hksj": float(hk_ci_high),
        "pi_lo": float(pi_low),
        "pi_hi": float(pi_high),
        "pi_t_crit_k_minus_2": float(t_crit_pi),
    }


def assess_tier(I2: float, pi_lo_pp: float, pi_hi_pp: float) -> dict:
    """Codex's 3-tier interpretation + PI override rule."""
    pi_crosses_zero = (pi_lo_pp <= 0 <= pi_hi_pp)
    if I2 > 0.50:
        base_tier = "substantial (I² > 50%)"
        base_level = 3
    elif I2 >= 0.25:
        base_tier = "moderate (25% ≤ I² ≤ 50%)"
        base_level = 2
    else:
        base_tier = "limited (I² < 25%)"
        base_level = 1
    # Override: if PI does NOT cross zero, downgrade by one level
    # (PI NOT crossing zero means effect sign IS transportable, so
    # heterogeneity is weaker in practical terms per Codex's rule).
    if not pi_crosses_zero:
        effective_level = max(1, base_level - 1)
        override_applied = True
    else:
        effective_level = base_level
        override_applied = False
    tier_map = {
        3: "substantial",
        2: "moderate",
        1: "limited",
    }
    return {
        "base_tier_description": base_tier,
        "pi_crosses_zero": bool(pi_crosses_zero),
        "pi_override_applied": override_applied,
        "effective_tier": tier_map[effective_level],
        "effective_level": effective_level,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--method-a", default="baro")
    ap.add_argument("--method-b", default="z_family")
    ap.add_argument("--bootstrap-n-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] unified inputs: {args.inputs_dir}")
    df = load_unified_long(args.inputs_dir)
    print(f"       n_rows={len(df)}, n_systems={df['system_id'].nunique()}, "
          f"methods={sorted(df['method'].unique())}")

    paired = per_system_paired_vectors(df, args.method_a, args.method_b)
    print(f"       matched systems: {len(paired)}")

    print(f"\n[per-system] computing Δ_s, v_s, bootstrap CIs ...")
    per_sys = per_system_stats(
        paired, df, args.method_a, args.method_b,
        bootstrap_n_iter=args.bootstrap_n_iter, seed=args.seed,
    )
    per_sys_sorted = per_sys.sort_values(
        ["benchmark", "system"]
    ).reset_index(drop=True)
    print(per_sys_sorted.to_string(index=False))

    per_sys_path = args.output_dir / "per_system_deltas.csv"
    per_sys_sorted.to_csv(per_sys_path, index=False)
    print(f"\n[written] {per_sys_path}")

    # --- meta-analysis on the full 11-system block --- #
    effects = per_sys_sorted["delta_pp"].to_numpy() / 100.0  # back to raw scale
    variances = per_sys_sorted["var"].to_numpy()
    row_names = per_sys_sorted["system"].tolist()

    print(f"\n[meta] running combine_effects(method_re='iterated', use_t=True) ...")
    meta = meta_analyze(effects, variances, row_names=row_names)
    # Convert key statistics from raw to pp for reporting.
    meta_pp = dict(meta)
    for key in ["delta_FE", "delta_RE", "re_ci_lo_hksj", "re_ci_hi_hksj",
                "pi_lo", "pi_hi", "se_FE"]:
        meta_pp[f"{key}_pp"] = meta_pp[key] * 100.0
    # tau2 is in pp² when effects are in pp; convert for reporting.
    meta_pp["tau2_pp2"] = meta["tau2"] * 1e4
    meta_pp["tau_pp"] = float(np.sqrt(meta["tau2"])) * 100.0

    tier = assess_tier(meta["I2"], meta_pp["pi_lo_pp"], meta_pp["pi_hi_pp"])
    meta_pp["tier"] = tier

    print(f"  k = {meta['k']}, df = {meta['df']}")
    print(f"  Q = {meta['Q']:.3f}, p_Q = {meta['p_Q']:.4f}")
    print(f"  I² = {meta['I2']*100:.2f}%")
    print(f"  τ² = {meta_pp['tau2_pp2']:.4f} pp² (τ = {meta_pp['tau_pp']:.3f} pp)")
    print(f"  Δ_FE = {meta_pp['delta_FE_pp']:.3f} pp  (SE = {meta_pp['se_FE_pp']:.3f} pp)")
    print(f"  Δ_RE = {meta_pp['delta_RE_pp']:.3f} pp  "
          f"(HKSJ 95% CI [{meta_pp['re_ci_lo_hksj_pp']:.3f}, "
          f"{meta_pp['re_ci_hi_hksj_pp']:.3f}] pp)")
    print(f"  95% prediction interval: [{meta_pp['pi_lo_pp']:.3f}, "
          f"{meta_pp['pi_hi_pp']:.3f}] pp")
    print(f"  tier (base):      {tier['base_tier_description']}")
    print(f"  PI crosses zero:  {tier['pi_crosses_zero']}")
    print(f"  PI override:      {tier['pi_override_applied']}")
    print(f"  effective tier:   {tier['effective_tier']}")

    meta_path = args.output_dir / "meta_analysis_summary.json"
    with open(meta_path, "w") as f:
        json.dump({
            "method_a": args.method_a,
            "method_b": args.method_b,
            "inputs_dir": str(args.inputs_dir),
            "systems": row_names,
            "effects_pp": (effects * 100.0).tolist(),
            "variances": variances.tolist(),
            "meta": meta,
            "meta_pp": {k: v for k, v in meta_pp.items()
                         if k not in ("tier",)},
            "tier": tier,
        }, f, indent=2, default=float)
    print(f"\n[written] {meta_path}")


if __name__ == "__main__":
    main()
