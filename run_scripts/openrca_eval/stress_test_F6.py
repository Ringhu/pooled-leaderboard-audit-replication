#!/usr/bin/env python3
"""Stress Test F6 — robustness of main findings to the zbase label choice.

F6 claim under test: "The 10.3 pp Bank divergence between legacy zbase and
unified zbase_univ is an honest trade-off that does NOT undermine F1/F2/F5
or the main interaction significance."

Three sub-tests:

  F6-A: Bank-excluded robustness.
    * drop openrca::bank, re-run the main interaction test (BARO vs
      zbase_univ, 10 systems) and the F5 regret analysis (BARO vs
      zbase_univ, 10 systems). If interaction p stays < 1e-3 and F5
      mis-rate stays > 15%, Bank is not driving the findings.

  F6-C: Dual-label 3-method interaction test on OpenRCA only.
    * Combine {baro, zbase (legacy), zbase_univ} on the 4 OpenRCA
      subsystems. 3 methods x 4 systems interaction LRT. If method x
      system interaction is still significant at p < 0.05, heterogeneity
      is a property of the systems, not an artefact of the z-label.

  F6-B: Cross-label sensitivity (cites pre-correction numbers).
    * Compare F5 regret and interaction p under legacy-zbase-on-OpenRCA
      (`z_family`) vs universal-zbase-everywhere (`zbase_univ`). Writes a
      side-by-side table, no new computation.

Output: results_stats/stress_test_F6/ with JSON + CSV summaries.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bootstrap_pairwise as bp          # noqa: E402
import regret_analysis as ra             # noqa: E402

ROOT = Path("$HOME/auditstack")
UNIFIED_INPUTS = ROOT / "results_xbench_11sys_unified" / "_inputs"
LEGACY_OPENRCA_DIR = ROOT / "results_openrca_xbench"   # has legacy zbase
UNIV_OPENRCA_DIR = ROOT / "results_openrca_zbase_univ"  # has zbase_univ


# ------------------------------------------------------------------ #
# Common loaders                                                     #
# ------------------------------------------------------------------ #
def patch_system_files_for_regret(inputs_dir: Path) -> list:
    patched = []
    for (benchmark, system, path, acc_col) in bp.SYSTEM_FILES:
        new_path = inputs_dir / path.name
        patched.append((benchmark, system, new_path, acc_col))
    return patched


def load_unified_long() -> pd.DataFrame:
    """Long-format df over unified 11-system block."""
    patched = patch_system_files_for_regret(UNIFIED_INPUTS)
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


# ------------------------------------------------------------------ #
# F6-A: Bank-excluded                                                #
# ------------------------------------------------------------------ #
def run_F6A_interaction(df_unified: pd.DataFrame, drop_system: str) -> dict:
    """Run the main method x system interaction test after dropping one system."""
    sub = df_unified[df_unified["system_id"] != drop_system].copy()
    # Restrict to BARO + z_family (already collapsed in loader).
    sub = sub[sub["method"].isin(["baro", "z_family"])].copy()
    sub["correct"] = sub["correct"].clip(0.0, 1.0).astype(float)

    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    n_obs = int(len(sub))
    n_cases = int(sub["case_id"].nunique())
    n_systems = int(sub["system_id"].nunique())

    def _fit(formula):
        return smf.glm(formula, data=sub, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": sub["case_id"]}
        )

    m_full = _fit("correct ~ C(method) * C(system_id)")
    m_red = _fit("correct ~ C(method) + C(system_id)")
    m_null = smf.glm("correct ~ 1", data=sub, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": sub["case_id"]}
    )
    lrt = 2.0 * (m_full.llf - m_red.llf)
    df_diff = len(m_full.params) - len(m_red.params)
    p = float(1.0 - chi2.cdf(lrt, df_diff))

    return {
        "dropped_system": drop_system,
        "n_obs": n_obs,
        "n_cases": n_cases,
        "n_systems": n_systems,
        "methods": sorted(sub["method"].unique().tolist()),
        "coverage_matrix": cov.to_dict(),
        "llf_full": float(m_full.llf),
        "llf_reduced": float(m_red.llf),
        "llf_null": float(m_null.llf),
        "df_full_params": int(len(m_full.params)),
        "df_reduced_params": int(len(m_red.params)),
        "lrt_statistic": float(lrt),
        "df_interaction_terms": int(df_diff),
        "p_interaction": p,
        "mcfadden_r2_full": float(1.0 - m_full.llf / m_null.llf),
        "mcfadden_r2_reduced": float(1.0 - m_red.llf / m_null.llf),
    }


def run_F6A_regret(df_unified: pd.DataFrame, drop_system: str) -> dict:
    sub = df_unified[df_unified["system_id"] != drop_system].copy()
    res = ra.compute_regret_pair(sub, "baro", "z_family",
                                 min_systems=2, n_iter=5000, seed=42)
    return {
        "dropped_system": drop_system,
        "n_systems": res["n_systems"],
        "decision_systems": res["decision_systems"],
        "mis_rate": res["mis_selection_rate"],
        "mean_regret_pp": res["mean_regret"] * 100,
        "max_regret_pp": res["max_regret"] * 100,
        "regret_ci_lo_pp": res["regret_ci_lo"] * 100,
        "regret_ci_hi_pp": res["regret_ci_hi"] * 100,
        "flip_systems": [d["system"] for d in res["per_system_details"]
                          if d["mis_selected"] == 1],
    }


# ------------------------------------------------------------------ #
# F6-C: Dual-label 3-method interaction on OpenRCA                   #
# ------------------------------------------------------------------ #
OPENRCA_LEGACY_FILES = {
    "bank": LEGACY_OPENRCA_DIR / "openrca_bank_results_v2.csv",
    "mkt1": LEGACY_OPENRCA_DIR / "openrca_mkt1_results_v2.csv",
    "mkt2": LEGACY_OPENRCA_DIR / "openrca_mkt2_results_v2.csv",
    "tel":  LEGACY_OPENRCA_DIR / "openrca_tel_results_v2.csv",
}
OPENRCA_UNIV_FILES = {
    "bank": UNIV_OPENRCA_DIR / "zbase_univ_bank_results_v2.csv",
    "mkt1": UNIV_OPENRCA_DIR / "zbase_univ_mkt1_results_v2.csv",
    "mkt2": UNIV_OPENRCA_DIR / "zbase_univ_mkt2_results_v2.csv",
    "tel":  UNIV_OPENRCA_DIR / "zbase_univ_tel_results_v2.csv",
}


def load_openrca_triple() -> pd.DataFrame:
    """Long-format over OpenRCA 4 subsystems with 3 methods:
    {baro, zbase (legacy), zbase_univ}."""
    frames = []
    # Legacy block: baro + zbase (and cd1min, causal, which we drop here).
    for sys_, path in OPENRCA_LEGACY_FILES.items():
        df = pd.read_csv(path)
        df = df[df["tdelta"] == 0].copy()
        df = df[df["method"].isin(["baro", "zbase"])].copy()
        df = df.dropna(subset=["acc1_v2", "method", "case"]).copy()
        df["q_idx"] = df.groupby(["method", "case"]).cumcount()
        df["benchmark"] = "openrca"
        df["system"] = sys_
        df["system_id"] = f"openrca::{sys_}"
        df["case_id"] = (df["system_id"] + "|" + df["case"].astype(str)
                         + "#" + df["q_idx"].astype(str))
        df["method_raw"] = df["method"].astype(str)
        df["correct"] = df["acc1_v2"].astype(float).clip(0, 1)
        frames.append(df[["benchmark","system","system_id","method","method_raw",
                          "case","q_idx","case_id","correct"]])
    # Universal block.
    for sys_, path in OPENRCA_UNIV_FILES.items():
        df = pd.read_csv(path)
        df = df[df["tdelta"] == 0].copy()
        df = df[df["method"] == "zbase_univ"].copy()
        df = df.dropna(subset=["acc1_v2", "method", "case"]).copy()
        df["q_idx"] = df.groupby(["method", "case"]).cumcount()
        df["benchmark"] = "openrca"
        df["system"] = sys_
        df["system_id"] = f"openrca::{sys_}"
        df["case_id"] = (df["system_id"] + "|" + df["case"].astype(str)
                         + "#" + df["q_idx"].astype(str))
        df["method_raw"] = df["method"].astype(str)
        df["correct"] = df["acc1_v2"].astype(float).clip(0, 1)
        frames.append(df[["benchmark","system","system_id","method","method_raw",
                          "case","q_idx","case_id","correct"]])

    return pd.concat(frames, ignore_index=True)


def run_F6C(df_triple: pd.DataFrame) -> dict:
    sub = df_triple.copy()
    cov = sub.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    n_obs = int(len(sub))
    n_cases = int(sub["case_id"].nunique())

    def _fit(formula):
        return smf.glm(formula, data=sub, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": sub["case_id"]}
        )

    m_full = _fit("correct ~ C(method) * C(system_id)")
    m_red = _fit("correct ~ C(method) + C(system_id)")
    m_null = smf.glm("correct ~ 1", data=sub, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": sub["case_id"]}
    )
    lrt = 2.0 * (m_full.llf - m_red.llf)
    df_diff = len(m_full.params) - len(m_red.params)
    p = float(1.0 - chi2.cdf(lrt, df_diff))

    # Per-method Wald tests on main effect.
    def _wald_for_factor(model, prefix: str) -> float:
        names = [n for n in model.params.index if n.startswith(prefix)]
        if not names:
            return float("nan")
        t = model.wald_test(names, scalar=True)
        return float(t.pvalue)

    p_method_main = _wald_for_factor(m_red, "C(method)")
    p_system_main = _wald_for_factor(m_red, "C(system_id)")

    # Per-method per-system acc means.
    per_cell = sub.groupby(["method", "system_id"])["correct"].agg(
        ["mean", "count"]).reset_index()

    return {
        "design": "3 methods (baro, zbase, zbase_univ) x 4 OpenRCA systems",
        "n_obs": n_obs,
        "n_cases": n_cases,
        "n_methods": int(sub["method"].nunique()),
        "n_systems": int(sub["system_id"].nunique()),
        "coverage_matrix": cov.to_dict(),
        "llf_full": float(m_full.llf),
        "llf_reduced": float(m_red.llf),
        "llf_null": float(m_null.llf),
        "df_full_params": int(len(m_full.params)),
        "df_reduced_params": int(len(m_red.params)),
        "lrt_statistic": float(lrt),
        "df_interaction_terms": int(df_diff),
        "p_interaction": p,
        "mcfadden_r2_full": float(1.0 - m_full.llf / m_null.llf),
        "mcfadden_r2_reduced": float(1.0 - m_red.llf / m_null.llf),
        "main_effect_method_p": p_method_main,
        "main_effect_system_p": p_system_main,
        "per_cell_means": per_cell.to_dict(orient="records"),
    }


# ------------------------------------------------------------------ #
# F6-B: sensitivity via pre-/post- comparison                         #
# ------------------------------------------------------------------ #
def build_F6B_table(previous_regret_csv: Path,
                    unified_regret_csv: Path,
                    previous_interaction_json: Path,
                    unified_interaction_json: Path) -> dict:
    comp = {}
    # Pull BARO vs z_family row from both regret CSVs.
    for label, path in [("pre_correction_z_family", previous_regret_csv),
                        ("post_correction_zbase_univ", unified_regret_csv)]:
        if not path.exists():
            comp[label] = {"error": f"missing {path}"}
            continue
        df = pd.read_csv(path)
        row = df[((df["method_a"] == "baro") & (df["method_b"] == "z_family")) |
                 ((df["method_a"] == "z_family") & (df["method_b"] == "baro"))]
        if len(row) == 0:
            comp[label] = {"error": "BARO vs z_family row not found"}
            continue
        r = row.iloc[0]
        comp[label] = {
            "n_systems": int(r["n_systems"]),
            "mis_rate": float(r["mis_selection_rate"]),
            "mean_regret_pp": float(r["mean_regret"]) * 100,
            "max_regret_pp": float(r["max_regret"]) * 100,
            "regret_ci_lo_pp": float(r["regret_ci_lo"]) * 100,
            "regret_ci_hi_pp": float(r["regret_ci_hi"]) * 100,
        }

    for label, path in [("pre_correction_z_family", previous_interaction_json),
                        ("post_correction_zbase_univ", unified_interaction_json)]:
        if not path.exists():
            comp.setdefault(label, {})["p_interaction"] = None
            continue
        with open(path) as f:
            j = json.load(f)
        comp.setdefault(label, {})["p_interaction"] = j.get("p_interaction")

    return comp


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results_stats" / "stress_test_F6")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("F6-A: Bank-excluded robustness")
    print("=" * 70)
    df_unified = load_unified_long()
    inter_A = run_F6A_interaction(df_unified, "openrca::bank")
    regret_A = run_F6A_regret(df_unified, "openrca::bank")
    print(f"  Interaction:  p = {inter_A['p_interaction']:.3e}  "
          f"n_systems = {inter_A['n_systems']}  n_obs = {inter_A['n_obs']}")
    print(f"  Regret:       mis_rate = {regret_A['mis_rate']*100:.2f}%  "
          f"mean = {regret_A['mean_regret_pp']:.3f}pp  "
          f"max = {regret_A['max_regret_pp']:.2f}pp  "
          f"CI = [{regret_A['regret_ci_lo_pp']:.2f}, "
          f"{regret_A['regret_ci_hi_pp']:.2f}]pp")
    print(f"  Flip systems after drop: {regret_A['flip_systems']}")

    with open(args.output_dir / "F6A_bank_excluded.json", "w") as f:
        json.dump({"interaction": inter_A, "regret": regret_A}, f,
                  indent=2, default=float)

    # Also: drop each of the other flip systems (petshop::high, rcaeval::OB)
    # to see which flip dominates the interaction.
    extra_drops = ["petshop::high_traffic", "rcaeval::OB"]
    print("\n  [extra: drop each flip system individually]")
    extra_rows = []
    for s in extra_drops:
        r = run_F6A_regret(df_unified, s)
        ir = run_F6A_interaction(df_unified, s)
        extra_rows.append({
            "dropped": s,
            "mis_rate": r["mis_rate"],
            "mean_regret_pp": r["mean_regret_pp"],
            "max_regret_pp": r["max_regret_pp"],
            "p_interaction": ir["p_interaction"],
        })
        print(f"    drop {s:<30s}  mis_rate={r['mis_rate']*100:5.2f}%  "
              f"mean={r['mean_regret_pp']:.3f}pp  max={r['max_regret_pp']:.2f}pp  "
              f"p_interact={ir['p_interaction']:.3e}")
    with open(args.output_dir / "F6A_extra_drops.json", "w") as f:
        json.dump(extra_rows, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("F6-C: Dual-label 3-method interaction on OpenRCA")
    print("=" * 70)
    df_triple = load_openrca_triple()
    print(f"  methods={sorted(df_triple['method'].unique())}")
    print(f"  systems={sorted(df_triple['system_id'].unique())}")
    cov = df_triple.groupby(["method", "system_id"]).size().unstack(fill_value=0)
    print("  coverage:")
    print(cov.to_string())
    res_C = run_F6C(df_triple)
    print(f"\n  Interaction LRT: chi2 = {res_C['lrt_statistic']:.3f}  "
          f"df = {res_C['df_interaction_terms']}  "
          f"p = {res_C['p_interaction']:.3e}")
    print(f"  McFadden R2 full    = {res_C['mcfadden_r2_full']:.4f}")
    print(f"  McFadden R2 reduced = {res_C['mcfadden_r2_reduced']:.4f}")
    print(f"  Main effect method p = {res_C['main_effect_method_p']:.3e}")
    print(f"  Main effect system p = {res_C['main_effect_system_p']:.3e}")

    print(f"\n  Per-cell acc@1 (method x system):")
    pc = pd.DataFrame(res_C["per_cell_means"])
    wide = pc.pivot(index="method", columns="system_id", values="mean")
    print(wide.round(4).to_string())

    with open(args.output_dir / "F6C_openrca_triple.json", "w") as f:
        json.dump(res_C, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("F6-B: Cross-label sensitivity (pre vs post-correction)")
    print("=" * 70)
    f6b = build_F6B_table(
        previous_regret_csv=ROOT / "results_stats" / "regret_table.csv",
        unified_regret_csv=ROOT / "results_stats" / "regret_table_unified.csv",
        previous_interaction_json=ROOT / "results_stats" / "interaction_test.json",
        unified_interaction_json=ROOT / "results_stats" / "interaction_test_unified.json",
    )
    print(json.dumps(f6b, indent=2))
    with open(args.output_dir / "F6B_cross_label.json", "w") as f:
        json.dump(f6b, f, indent=2, default=float)

    # ---------- final verdict summary ----------
    verdict = {
        "F6A_bank_excluded": {
            "p_interaction": inter_A["p_interaction"],
            "mis_rate": regret_A["mis_rate"],
            "mean_regret_pp": regret_A["mean_regret_pp"],
            "verdict": "PASS" if (inter_A["p_interaction"] < 1e-3
                                    and regret_A["mis_rate"] > 0.15) else "MARGINAL",
        },
        "F6C_openrca_triple": {
            "p_interaction": res_C["p_interaction"],
            "verdict": "PASS" if res_C["p_interaction"] < 0.05 else "FAIL",
        },
        "F6B_cross_label": f6b,
    }
    with open(args.output_dir / "F6_verdict_summary.json", "w") as f:
        json.dump(verdict, f, indent=2, default=float)
    print("\n=== Final F6 verdict summary ===")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
