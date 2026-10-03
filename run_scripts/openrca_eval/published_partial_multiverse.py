#!/usr/bin/env python3
"""Partial-coverage multiverse for published RCA method outputs.

This is appendix-only evidence. It reports method-pair variation where matched
published-method outputs already exist, while keeping coverage explicit:

  - OpenRCA subset: BARO, Causal, cd1min, max-|Z| across 4 systems.
  - CIRCA subset: BARO, CIRCA, max-|Z| across the 6 systems where CIRCA has
    completed matched outputs in the unified block.

These results must not be mixed into the 11-system full-coverage main claim.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from heterogeneity_analysis import assess_tier, meta_analyze  # noqa: E402
from multiverse_pair_analysis import DISPLAY_NAMES as BASE_DISPLAY_NAMES  # noqa: E402
from multiverse_pair_analysis import bootstrap_delta_ci, load_with_inputs_dir  # noqa: E402
from regret_analysis import compute_regret_pair  # noqa: E402

DEFAULT_REPO_ROOT = Path("$HOME/mnt/3090/auditstack")
DEFAULT_UNIFIED_INPUTS = DEFAULT_REPO_ROOT / "results_xbench_11sys_unified" / "_inputs"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "results_stats" / "multiverse_published_partial"

DISPLAY_NAMES = {
    **BASE_DISPLAY_NAMES,
    "causal": "Causal",
    "cd1min": "CD-1min",
    "circa": "CIRCA",
}


def complete_subset(df: pd.DataFrame, methods: list[str], systems: list[str]) -> pd.DataFrame:
    cols = ["benchmark", "system", "system_id", "case", "q_idx", "case_id", "method", "correct"]
    sub = df[(df["method"].isin(methods)) & (df["system_id"].isin(systems))][cols].copy()
    wide = sub.pivot_table(
        index=["benchmark", "system", "system_id", "case", "q_idx", "case_id"],
        columns="method",
        values="correct",
        aggfunc="mean",
    ).reset_index()
    complete = wide.dropna(subset=methods).copy()
    return complete.melt(
        id_vars=["benchmark", "system", "system_id", "case", "q_idx", "case_id"],
        value_vars=methods,
        var_name="method",
        value_name="correct",
    ).sort_values(["system_id", "case_id", "method"]).reset_index(drop=True)


def paired_arrays(df: pd.DataFrame, method_a: str, method_b: str) -> dict[str, pd.DataFrame]:
    out = {}
    for system in sorted(df["system_id"].unique()):
        sub = df[df["system_id"] == system]
        wide = (
            sub[sub["method"].isin([method_a, method_b])]
            .pivot_table(index="case_id", columns="method", values="correct", aggfunc="mean")
            .dropna(subset=[method_a, method_b])
        )
        if len(wide):
            out[system] = wide[[method_a, method_b]]
    return out


def compute_pair_outputs(df: pd.DataFrame, methods: list[str], n_iter: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_rows = []
    summary_rows = []
    loso_rows = []
    loso_per_rows = []

    for method_a, method_b in itertools.combinations(methods, 2):
        arrays = paired_arrays(df, method_a, method_b)
        for system, wide in arrays.items():
            arr = wide.to_numpy(dtype=float)
            delta = arr[:, 0] - arr[:, 1]
            ci_lo, ci_hi = bootstrap_delta_ci(delta, n_iter=n_iter, seed=seed)
            n = len(delta)
            var = float(np.var(delta, ddof=1) / n) if n > 1 else float("nan")
            per_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
                "system": system,
                "benchmark": system.split("::", 1)[0],
                "n_cases": n,
                "delta": float(delta.mean()),
                "delta_pp": float(delta.mean() * 100.0),
                "var": var,
                "ci_lo_pp": ci_lo * 100.0,
                "ci_hi_pp": ci_hi * 100.0,
            })

        ps = pd.DataFrame([r for r in per_rows if r["method_a"] == method_a and r["method_b"] == method_b])
        n_pos = int((ps["delta_pp"] > 0).sum())
        n_neg = int((ps["delta_pp"] < 0).sum())
        n_zero = int((ps["delta_pp"] == 0).sum())
        variances = ps["var"].to_numpy(dtype=float)
        effects = ps["delta"].to_numpy(dtype=float)
        meta_fields = {
            "Q": np.nan,
            "df": np.nan,
            "p_Q": np.nan,
            "I2": np.nan,
            "I2_pct": np.nan,
            "tau2_pp2": np.nan,
            "delta_RE_pp": np.nan,
            "re_ci_lo_hksj_pp": np.nan,
            "re_ci_hi_hksj_pp": np.nan,
            "pi_lo_pp": np.nan,
            "pi_hi_pp": np.nan,
            "pi_crosses_zero": np.nan,
            "effective_tier": "not_computed_zero_variance",
        }
        if len(ps) >= 3 and np.all(np.isfinite(variances)) and np.all(variances > 0):
            meta = meta_analyze(effects, variances, row_names=ps["system"].tolist())
            meta_pp = dict(meta)
            for key in ["delta_RE", "re_ci_lo_hksj", "re_ci_hi_hksj", "pi_lo", "pi_hi"]:
                meta_pp[f"{key}_pp"] = meta_pp[key] * 100.0
            tier = assess_tier(meta["I2"], meta_pp["pi_lo_pp"], meta_pp["pi_hi_pp"])
            meta_fields = {
                "Q": meta["Q"],
                "df": meta["df"],
                "p_Q": meta["p_Q"],
                "I2": meta["I2"],
                "I2_pct": meta["I2"] * 100.0,
                "tau2_pp2": meta["tau2"] * 1e4,
                "delta_RE_pp": meta_pp["delta_RE_pp"],
                "re_ci_lo_hksj_pp": meta_pp["re_ci_lo_hksj_pp"],
                "re_ci_hi_hksj_pp": meta_pp["re_ci_hi_hksj_pp"],
                "pi_lo_pp": meta_pp["pi_lo_pp"],
                "pi_hi_pp": meta_pp["pi_hi_pp"],
                "pi_crosses_zero": tier["pi_crosses_zero"],
                "effective_tier": tier["effective_tier"],
            }

        res = compute_regret_pair(df, method_a, method_b, min_systems=2, n_iter=n_iter, seed=seed)
        flip_systems = [d["system"] for d in res["per_system_details"] if d["mis_selected"] == 1]
        loso_rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
            "n_systems": res["n_systems"],
            "decision_systems": res["decision_systems"],
            "mis_selection_count": int(sum(d["mis_selected"] == 1 for d in res["per_system_details"])),
            "mis_selection_rate": res["mis_selection_rate"],
            "mean_regret_pp": res["mean_regret"] * 100.0,
            "max_regret_pp": res["max_regret"] * 100.0,
            "regret_ci_lo_pp": res["regret_ci_lo"] * 100.0,
            "regret_ci_hi_pp": res["regret_ci_hi"] * 100.0,
            "flip_systems": ";".join(flip_systems),
        })
        for d in res["per_system_details"]:
            loso_per_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
                **d,
                "regret_pp": d["regret"] * 100.0,
            })

        summary_rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
            "k": int(len(ps)),
            "n_cases_sum": int(ps["n_cases"].sum()),
            "n_positive_systems": n_pos,
            "n_negative_systems": n_neg,
            "n_zero_systems": n_zero,
            "sign_changing_effects": bool(n_pos > 0 and n_neg > 0),
            **meta_fields,
        })

    return (
        pd.DataFrame(per_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(loso_rows),
        pd.DataFrame(loso_per_rows),
    )


def write_block(df: pd.DataFrame, methods: list[str], label: str, out_dir: Path, n_iter: int, seed: int) -> None:
    per, summary, loso, loso_per = compute_pair_outputs(df, methods, n_iter=n_iter, seed=seed)
    df.to_csv(out_dir / f"{label}_long.csv", index=False)
    per.to_csv(out_dir / f"{label}_per_system_deltas.csv", index=False)
    summary.to_csv(out_dir / f"{label}_pair_summary.csv", index=False)
    loso.to_csv(out_dir / f"{label}_loso_summary.csv", index=False)
    loso_per.to_csv(out_dir / f"{label}_loso_per_system.csv", index=False)
    matrix = per.pivot(index="pair", columns="system", values="delta_pp").reset_index()
    matrix.to_csv(out_dir / f"{label}_delta_matrix.csv", index=False)
    print(f"\n[{label} pair summary]")
    print(summary.to_string(index=False))
    print(f"\n[{label} LOSO summary]")
    print(loso.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified-inputs", type=Path, default=DEFAULT_UNIFIED_INPUTS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--bootstrap-n-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    unified = load_with_inputs_dir(args.unified_inputs)
    coverage = (
        unified.groupby(["system_id", "method"])["case_id"]
        .nunique()
        .reset_index(name="n_scoring_units")
    )
    coverage.to_csv(args.output_dir / "method_coverage.csv", index=False)

    openrca_systems = sorted(unified.loc[unified["benchmark"] == "openrca", "system_id"].unique())
    openrca_methods = ["baro", "causal", "cd1min", "z_family"]
    openrca = complete_subset(unified, openrca_methods, openrca_systems)
    write_block(openrca, openrca_methods, "openrca", args.output_dir, args.bootstrap_n_iter, args.seed)

    circa_methods = ["baro", "circa", "z_family"]
    circa_candidate_systems = []
    for system, sub in unified.groupby("system_id"):
        if all(m in set(sub["method"].unique()) for m in circa_methods):
            circa_candidate_systems.append(system)
    circa = complete_subset(unified, circa_methods, sorted(circa_candidate_systems))
    write_block(circa, circa_methods, "circa_subset", args.output_dir, args.bootstrap_n_iter, args.seed)

    print(f"\n[written] {args.output_dir}")


if __name__ == "__main__":
    main()
