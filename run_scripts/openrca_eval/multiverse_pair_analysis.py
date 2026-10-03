#!/usr/bin/env python3
"""Full-coverage method-pair multiverse for the RCA benchmark audit.

This script combines the existing unified max-|Z| block and alert-count block
into one matched long table for the three methods that cover all 11 audited
subsystems:

  - baro
  - z_family      (the unified portable max-|Z| predictor)
  - alertcount

It then computes every full-coverage method pair:

  - per-system paired deltas and bootstrap CIs
  - random-effects heterogeneity summary
  - leave-one-system-out pooled-selection regret

The script intentionally writes to a new result directory so these multiverse
outputs cannot be confused with older two-pair artifacts.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bootstrap_pairwise as bp  # noqa: E402
from heterogeneity_analysis import assess_tier, meta_analyze  # noqa: E402
from regret_analysis import compute_regret_pair  # noqa: E402

DEFAULT_REPO_ROOT = Path("$HOME/mnt/3090/auditstack")
DEFAULT_UNIFIED_INPUTS = DEFAULT_REPO_ROOT / "results_xbench_11sys_unified" / "_inputs"
DEFAULT_ALERT_INPUTS = DEFAULT_REPO_ROOT / "results_xbench_11sys_alertcount" / "_inputs"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "results_stats" / "multiverse_full_coverage"

FULL_METHODS = ["baro", "z_family", "alertcount"]
DISPLAY_NAMES = {
    "baro": "BARO",
    "z_family": "max-|Z|",
    "alertcount": "alert-count",
}


def load_with_inputs_dir(inputs_dir: Path) -> pd.DataFrame:
    """Load an 11-system input block using bootstrap_pairwise's canonical
    system-file metadata, but patch file paths to the supplied artifact dir."""
    patched = []
    for benchmark, system, path, acc_col in bp.SYSTEM_FILES:
        patched.append((benchmark, system, inputs_dir / path.name, acc_col))
    missing = [str(path) for _, _, path, _ in patched if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(missing))

    original = list(bp.SYSTEM_FILES)
    bp.SYSTEM_FILES = patched
    try:
        return bp.load_all_systems()
    finally:
        bp.SYSTEM_FILES = original


def _method_slice(df: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = ["benchmark", "system", "system_id", "case", "q_idx", "case_id", "method", "correct"]
    out = df[df["method"] == method][cols].copy()
    return out


def build_full_coverage_long(unified: pd.DataFrame, alert: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Combine unified {baro,z_family} and alert-count {alertcount}, checking
    that BARO rows agree exactly between both blocks."""
    unified_baro = _method_slice(unified, "baro").rename(columns={"correct": "correct_unified"})
    alert_baro = _method_slice(alert, "baro").rename(columns={"correct": "correct_alert"})
    merged_baro = unified_baro.merge(
        alert_baro[["case_id", "correct_alert"]],
        on="case_id",
        how="outer",
        indicator=True,
    )
    missing_or_extra = int((merged_baro["_merge"] != "both").sum())
    both = merged_baro[merged_baro["_merge"] == "both"].copy()
    max_baro_abs_diff = (
        float((both["correct_unified"] - both["correct_alert"]).abs().max())
        if len(both) else float("nan")
    )

    full = pd.concat(
        [
            _method_slice(unified, "baro"),
            _method_slice(unified, "z_family"),
            _method_slice(alert, "alertcount"),
        ],
        ignore_index=True,
    )

    # Keep only case_ids where all full-coverage methods are present.
    wide = full.pivot_table(
        index=["benchmark", "system", "system_id", "case", "q_idx", "case_id"],
        columns="method",
        values="correct",
        aggfunc="mean",
    ).reset_index()
    complete = wide.dropna(subset=FULL_METHODS).copy()

    full_complete = complete.melt(
        id_vars=["benchmark", "system", "system_id", "case", "q_idx", "case_id"],
        value_vars=FULL_METHODS,
        var_name="method",
        value_name="correct",
    )

    sanity = {
        "baro_alignment_missing_or_extra": missing_or_extra,
        "baro_alignment_max_abs_diff": max_baro_abs_diff,
        "n_complete_scoring_units": int(len(complete)),
        "n_long_rows": int(len(full_complete)),
        "n_systems": int(complete["system_id"].nunique()),
        "methods": FULL_METHODS,
        "complete_units_by_system": complete.groupby("system_id").size().to_dict(),
    }
    return full_complete.sort_values(["system_id", "case_id", "method"]).reset_index(drop=True), sanity


def method_coverage(df: pd.DataFrame) -> pd.DataFrame:
    cov = (
        df.groupby(["system_id", "method"])["case_id"]
        .nunique()
        .reset_index(name="n_scoring_units")
    )
    return cov.pivot(index="system_id", columns="method", values="n_scoring_units").fillna(0).astype(int).reset_index()


def paired_arrays(df: pd.DataFrame, method_a: str, method_b: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for system in sorted(df["system_id"].unique()):
        sub = df[df["system_id"] == system]
        wide = (
            sub[sub["method"].isin([method_a, method_b])]
            .pivot_table(index="case_id", columns="method", values="correct", aggfunc="mean")
            .dropna(subset=[method_a, method_b])
        )
        if len(wide):
            out[system] = wide[[method_a, method_b]].copy()
    return out


def bootstrap_delta_ci(delta: np.ndarray, n_iter: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_iter, len(delta)))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def compute_pair_deltas(df: pd.DataFrame, n_iter: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_system_rows = []
    summary_rows = []
    for method_a, method_b in itertools.combinations(FULL_METHODS, 2):
        arrays = paired_arrays(df, method_a, method_b)
        for system, wide in arrays.items():
            arr = wide.to_numpy(dtype=float)
            delta = arr[:, 0] - arr[:, 1]
            ci_lo, ci_hi = bootstrap_delta_ci(delta, n_iter=n_iter, seed=seed)
            n = len(delta)
            var = float(np.var(delta, ddof=1) / n) if n > 1 else float("nan")
            benchmark = system.split("::", 1)[0]
            per_system_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "method_a_display": DISPLAY_NAMES[method_a],
                "method_b_display": DISPLAY_NAMES[method_b],
                "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
                "system": system,
                "benchmark": benchmark,
                "n_cases": n,
                "delta": float(delta.mean()),
                "delta_pp": float(delta.mean() * 100.0),
                "var": var,
                "var_pp": var * 1e4 if not np.isnan(var) else float("nan"),
                "se_pp": float(np.sqrt(var) * 100.0) if not np.isnan(var) else float("nan"),
                "ci_lo_pp": ci_lo * 100.0,
                "ci_hi_pp": ci_hi * 100.0,
                "sign": int(np.sign(delta.mean())),
            })

        pair_rows = [r for r in per_system_rows if r["method_a"] == method_a and r["method_b"] == method_b]
        ps = pd.DataFrame(pair_rows).sort_values(["benchmark", "system"])
        effects = ps["delta"].to_numpy()
        variances = ps["var"].to_numpy()
        meta = meta_analyze(effects, variances, row_names=ps["system"].tolist())
        meta_pp = dict(meta)
        for key in ["delta_FE", "delta_RE", "re_ci_lo_hksj", "re_ci_hi_hksj", "pi_lo", "pi_hi", "se_FE"]:
            meta_pp[f"{key}_pp"] = meta_pp[key] * 100.0
        meta_pp["tau2_pp2"] = meta["tau2"] * 1e4
        meta_pp["tau_pp"] = float(np.sqrt(meta["tau2"])) * 100.0
        tier = assess_tier(meta["I2"], meta_pp["pi_lo_pp"], meta_pp["pi_hi_pp"])

        n_pos = int((ps["delta_pp"] > 0).sum())
        n_neg = int((ps["delta_pp"] < 0).sum())
        n_zero = int((ps["delta_pp"] == 0).sum())
        summary_rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
            "k": int(meta["k"]),
            "n_cases_sum": int(ps["n_cases"].sum()),
            "n_positive_systems": n_pos,
            "n_negative_systems": n_neg,
            "n_zero_systems": n_zero,
            "sign_changing_effects": bool(n_pos > 0 and n_neg > 0),
            "Q": meta["Q"],
            "df": meta["df"],
            "p_Q": meta["p_Q"],
            "I2": meta["I2"],
            "I2_pct": meta["I2"] * 100.0,
            "tau2_pp2": meta_pp["tau2_pp2"],
            "delta_RE_pp": meta_pp["delta_RE_pp"],
            "re_ci_lo_hksj_pp": meta_pp["re_ci_lo_hksj_pp"],
            "re_ci_hi_hksj_pp": meta_pp["re_ci_hi_hksj_pp"],
            "pi_lo_pp": meta_pp["pi_lo_pp"],
            "pi_hi_pp": meta_pp["pi_hi_pp"],
            "pi_crosses_zero": tier["pi_crosses_zero"],
            "effective_tier": tier["effective_tier"],
        })

    return (
        pd.DataFrame(per_system_rows).sort_values(["method_a", "method_b", "benchmark", "system"]).reset_index(drop=True),
        pd.DataFrame(summary_rows).sort_values(["method_a", "method_b"]).reset_index(drop=True),
    )


def compute_loso(df: pd.DataFrame, n_iter: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    per_system_rows = []
    for method_a, method_b in itertools.combinations(FULL_METHODS, 2):
        res = compute_regret_pair(
            df,
            method_a,
            method_b,
            min_systems=2,
            n_iter=n_iter,
            seed=seed,
        )
        flip_systems = [
            d["system"] for d in res["per_system_details"]
            if d["mis_selected"] == 1
        ]
        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
            "n_systems": res["n_systems"],
            "decision_systems": res["decision_systems"],
            "mis_selection_count": int(sum(d["mis_selected"] == 1 for d in res["per_system_details"])),
            "mis_selection_rate": res["mis_selection_rate"],
            "mean_regret": res["mean_regret"],
            "mean_regret_pp": res["mean_regret"] * 100.0,
            "max_regret": res["max_regret"],
            "max_regret_pp": res["max_regret"] * 100.0,
            "regret_ci_lo": res["regret_ci_lo"],
            "regret_ci_hi": res["regret_ci_hi"],
            "regret_ci_lo_pp": res["regret_ci_lo"] * 100.0,
            "regret_ci_hi_pp": res["regret_ci_hi"] * 100.0,
            "pooled_tie_count": res["pooled_tie_count"],
            "actual_tie_count": res["actual_tie_count"],
            "flip_systems": ";".join(flip_systems),
        })
        for d in res["per_system_details"]:
            per_system_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
                **d,
                "regret_pp": d["regret"] * 100.0,
            })
    return pd.DataFrame(rows), pd.DataFrame(per_system_rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified-inputs", type=Path, default=DEFAULT_UNIFIED_INPUTS)
    ap.add_argument("--alert-inputs", type=Path, default=DEFAULT_ALERT_INPUTS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--bootstrap-n-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] unified:    {args.unified_inputs}")
    unified = load_with_inputs_dir(args.unified_inputs)
    print(f"[load] alertcount: {args.alert_inputs}")
    alert = load_with_inputs_dir(args.alert_inputs)

    full, sanity = build_full_coverage_long(unified, alert)
    sanity["unified_inputs"] = str(args.unified_inputs)
    sanity["alert_inputs"] = str(args.alert_inputs)
    sanity["seed"] = args.seed
    sanity["bootstrap_n_iter"] = args.bootstrap_n_iter

    if sanity["baro_alignment_missing_or_extra"] != 0:
        raise RuntimeError(f"BARO case-id mismatch between blocks: {sanity}")
    if sanity["baro_alignment_max_abs_diff"] != 0.0:
        raise RuntimeError(f"BARO score mismatch between blocks: {sanity}")
    if sanity["n_complete_scoring_units"] != 778:
        raise RuntimeError(f"Expected 778 complete scoring units, got {sanity['n_complete_scoring_units']}")
    if sanity["n_systems"] != 11:
        raise RuntimeError(f"Expected 11 systems, got {sanity['n_systems']}")

    write_json(args.output_dir / "input_sanity.json", sanity)
    full.to_csv(args.output_dir / "full_coverage_long.csv", index=False)
    cov = method_coverage(full)
    cov.to_csv(args.output_dir / "method_coverage.csv", index=False)

    print("[compute] per-system deltas + random-effects summaries")
    per_system, pair_summary = compute_pair_deltas(full, args.bootstrap_n_iter, args.seed)
    per_system.to_csv(args.output_dir / "per_system_deltas.csv", index=False)
    pair_summary.to_csv(args.output_dir / "pair_summary.csv", index=False)

    matrix = per_system.pivot(index="pair", columns="system", values="delta_pp").reset_index()
    matrix.to_csv(args.output_dir / "multiverse_delta_matrix.csv", index=False)

    print("[compute] LOSO regrets")
    loso_summary, loso_per_system = compute_loso(full, args.bootstrap_n_iter, args.seed)
    loso_summary.to_csv(args.output_dir / "loso_pair_summary.csv", index=False)
    loso_per_system.to_csv(args.output_dir / "loso_pair_per_system.csv", index=False)

    print(f"[written] {args.output_dir}")
    print("\n[pair summary]")
    print(pair_summary.to_string(index=False))
    print("\n[LOSO summary]")
    print(loso_summary.to_string(index=False))


if __name__ == "__main__":
    main()
