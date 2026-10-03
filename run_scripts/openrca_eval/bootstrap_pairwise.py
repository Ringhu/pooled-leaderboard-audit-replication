#!/usr/bin/env python3
"""Paired case-resampling bootstrap for RCA benchmark audit ("System Matters").

Loads all 11 per-system CSVs (OpenRCA x4, RCAEval x3, PetShop x4) into a
long-format DataFrame and runs paired bootstraps across every
(method_a, method_b, system) triple where both methods have data.

The zbase (OpenRCA) and zbase_univ (non-OpenRCA) labels are collapsed into a
single 'z_family' label for coverage-complete {BARO, z_family} analysis; the
original label is preserved in the `method_raw` column.

Outputs:
  results_stats/pairwise_bootstrap.csv

Usage (from 3090, auditstack env):
  python scripts/openrca_eval/bootstrap_pairwise.py
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Layout                                                             #
# ------------------------------------------------------------------ #
ROOT = Path("$HOME/auditstack")
OPENRCA_DIR = ROOT / "results_openrca_xbench"
MERGED_INPUTS = ROOT / "results_xbench_11sys_with_zbase" / "_inputs"
OUT_CSV = ROOT / "results_stats" / "pairwise_bootstrap.csv"

# (benchmark, system, file, acc_col)
SYSTEM_FILES = [
    ("openrca", "bank",     OPENRCA_DIR / "openrca_bank_results_v2.csv",  "acc1_v2"),
    ("openrca", "mkt1",     OPENRCA_DIR / "openrca_mkt1_results_v2.csv",  "acc1_v2"),
    ("openrca", "mkt2",     OPENRCA_DIR / "openrca_mkt2_results_v2.csv",  "acc1_v2"),
    ("openrca", "tel",      OPENRCA_DIR / "openrca_tel_results_v2.csv",   "acc1_v2"),
    ("rcaeval", "OB",       MERGED_INPUTS / "rcaeval_merged_online-boutique_results.csv", "acc1_v2"),
    ("rcaeval", "SS",       MERGED_INPUTS / "rcaeval_merged_sock-shop-2_results.csv",     "acc1_v2"),
    ("rcaeval", "TT",       MERGED_INPUTS / "rcaeval_merged_train-ticket_results.csv",    "acc1_v2"),
    ("petshop", "high_traffic",      MERGED_INPUTS / "petshop_merged_high_traffic_results.csv",      "acc@1"),
    ("petshop", "low_traffic",       MERGED_INPUTS / "petshop_merged_low_traffic_results.csv",       "acc@1"),
    ("petshop", "temporal_traffic1", MERGED_INPUTS / "petshop_merged_temporal_traffic1_results.csv", "acc@1"),
    ("petshop", "temporal_traffic2", MERGED_INPUTS / "petshop_merged_temporal_traffic2_results.csv", "acc@1"),
]

Z_FAMILY_RAWS = {"zbase", "zbase_univ"}


# ------------------------------------------------------------------ #
# Loading                                                            #
# ------------------------------------------------------------------ #
def load_all_systems(inputs_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all 11 per-system CSVs into one long DataFrame.

    Columns: [benchmark, system, system_id, method, method_raw, case,
              case_id, q_idx, correct (0/1)].

    * For all benchmarks only tdelta==0 rows are kept (the canonical slice;
      view_accuracy.csv and the task reference numbers all use tdelta=0).
    * OpenRCA has multiple rows per (method, case) — each is a query/subtask.
      We assign a within-group q_idx via cumcount so each row becomes a
      distinct "pairing unit" (aligned across methods since row counts match
      per case).
    * acc column is renamed to `correct` (0/1).
    * `system_id` = "<benchmark>::<system>"; `case_id` = "<system_id>|<case>#<q_idx>"
      — globally unique, safe to cluster on in the logit.
    * zbase + zbase_univ are collapsed into method="z_family"; the original
      label is kept in method_raw.
    """
    frames: List[pd.DataFrame] = []
    for benchmark, system, path, acc_col in SYSTEM_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")
        df = pd.read_csv(path)
        if "tdelta" in df.columns:
            df = df[df["tdelta"] == 0].copy()
        if acc_col not in df.columns:
            raise KeyError(f"{path} missing expected acc column {acc_col}; has {df.columns.tolist()}")
        if "case" not in df.columns:
            raise KeyError(f"{path} missing 'case' column")
        if "method" not in df.columns:
            raise KeyError(f"{path} missing 'method' column")

        # Drop rows with NaN in the acc column (shouldn't happen, defensive).
        df = df.dropna(subset=[acc_col, "method", "case"]).copy()

        # Per-group query index for OpenRCA (multi-row per case); 0 elsewhere.
        df["q_idx"] = df.groupby(["method", "case"]).cumcount()

        # Collapse zbase / zbase_univ into z_family.
        df["method_raw"] = df["method"].astype(str)
        df["method"] = df["method_raw"].where(
            ~df["method_raw"].isin(Z_FAMILY_RAWS), "z_family"
        )

        df["benchmark"] = benchmark
        df["system"] = system
        df["system_id"] = f"{benchmark}::{system}"
        df["case_id"] = (
            df["system_id"] + "|" + df["case"].astype(str) + "#" + df["q_idx"].astype(str)
        )
        df["correct"] = df[acc_col].astype(float)

        keep = ["benchmark", "system", "system_id", "method", "method_raw",
                "case", "q_idx", "case_id", "correct"]
        frames.append(df[keep])

    long = pd.concat(frames, ignore_index=True)
    return long


# ------------------------------------------------------------------ #
# Paired bootstrap                                                   #
# ------------------------------------------------------------------ #
def _wide_for_pair(df_sys: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    """Return a per-case_id wide frame with columns [case_id, a, b] containing
    the `correct` values, aligned by case_id (intersection)."""
    a = df_sys[df_sys["method"] == method_a][["case_id", "correct"]].rename(columns={"correct": "a"})
    b = df_sys[df_sys["method"] == method_b][["case_id", "correct"]].rename(columns={"correct": "b"})
    # Average duplicates within a method on same case_id (should be none after q_idx
    # differentiation, but defensive).
    a = a.groupby("case_id", as_index=False)["a"].mean()
    b = b.groupby("case_id", as_index=False)["b"].mean()
    merged = a.merge(b, on="case_id", how="inner")
    return merged


def paired_bootstrap(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    system: str,
    n_iter: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Case-resampling paired bootstrap for the difference acc(A) - acc(B).

    Returns dict with mean_delta, ci_low, ci_high, p_value_twosided, n_cases.

    `system` is matched against df['system_id'] first, falling back to the
    short `system` label for convenience.
    """
    sub = df[df["system_id"] == system]
    if sub.empty:
        sub = df[df["system"] == system]
    if sub.empty:
        raise ValueError(f"No rows for system={system!r}")

    wide = _wide_for_pair(sub, method_a, method_b)
    n = len(wide)
    if n == 0:
        raise ValueError(
            f"No common case_ids for ({method_a}, {method_b}) on {system}"
        )

    a_n = int(sub[sub["method"] == method_a]["case_id"].nunique())
    b_n = int(sub[sub["method"] == method_b]["case_id"].nunique())
    if n < max(a_n, b_n):
        warnings.warn(
            f"Case sets differ on {system} for ({method_a},{method_b}): "
            f"A={a_n}, B={b_n}, intersection={n}"
        )

    a_arr = wide["a"].to_numpy(dtype=float)
    b_arr = wide["b"].to_numpy(dtype=float)
    delta = a_arr - b_arr
    mean_delta = float(delta.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    boot_means = delta[idx].mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])

    # Two-sided p-value via bootstrap (null: mean_delta == 0).
    # Center the bootstrap distribution at zero, then compute the mass as
    # extreme as the observed mean.
    centered = boot_means - mean_delta
    if mean_delta >= 0:
        p_one = float((centered >= mean_delta).mean())
    else:
        p_one = float((centered <= mean_delta).mean())
    p_two = min(1.0, 2.0 * p_one)

    return {
        "mean_delta": mean_delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value_twosided": p_two,
        "n_cases": int(n),
    }


def run_full(df: pd.DataFrame, output_csv: Path, n_iter: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Iterate over all (method_a, method_b, system) triples where both
    methods have data on that system; write CSV."""
    rows = []
    systems = sorted(df["system_id"].unique())
    for system in systems:
        sub = df[df["system_id"] == system]
        methods = sorted(sub["method"].unique())
        for a, b in itertools.combinations(methods, 2):
            wide = _wide_for_pair(sub, a, b)
            if len(wide) == 0:
                continue
            res = paired_bootstrap(df, a, b, system, n_iter=n_iter, seed=seed)
            rows.append({
                "method_a": a,
                "method_b": b,
                "system": system,
                "n_cases": res["n_cases"],
                "mean_delta": res["mean_delta"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "p_two_sided": res["p_value_twosided"],
            })
    out = pd.DataFrame(rows).sort_values(["system", "method_a", "method_b"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


# ------------------------------------------------------------------ #
# S2 smoke test                                                      #
# ------------------------------------------------------------------ #
def s2_smoke(df: pd.DataFrame, n_iter: int = 10_000, seed: int = 42) -> dict:
    """Bank vs Mkt-1, (baro, causal). Returns a dict with full diagnostics +
    PASS/FAIL flag."""
    bank = paired_bootstrap(df, "baro", "causal", "openrca::bank", n_iter=n_iter, seed=seed)
    mkt1 = paired_bootstrap(df, "baro", "causal", "openrca::mkt1", n_iter=n_iter, seed=seed)

    bank_sign_ok = bank["mean_delta"] > 0
    mkt1_sign_ok = mkt1["mean_delta"] < 0
    # Non-overlap of the two CIs (for the flip to be "outside noise").
    overlap = not (bank["ci_high"] < mkt1["ci_low"] or mkt1["ci_high"] < bank["ci_low"])
    non_overlap_ok = not overlap

    def _ci_width(r): return (r["ci_high"] - r["ci_low"]) * 100.0  # pp
    bank_width = _ci_width(bank)
    mkt1_width = _ci_width(mkt1)
    width_bank_ok = 5.0 <= bank_width <= 30.0
    width_mkt1_ok = 5.0 <= mkt1_width <= 30.0

    passed = all([bank_sign_ok, mkt1_sign_ok, non_overlap_ok, width_bank_ok, width_mkt1_ok])
    return {
        "bank": bank, "mkt1": mkt1,
        "bank_sign_ok": bank_sign_ok, "mkt1_sign_ok": mkt1_sign_ok,
        "non_overlap_ok": non_overlap_ok,
        "bank_width_pp": bank_width, "mkt1_width_pp": mkt1_width,
        "width_bank_ok": width_bank_ok, "width_mkt1_ok": width_mkt1_ok,
        "passed": passed,
    }


def _fmt(x):
    return f"{x:+.4f}" if isinstance(x, float) else str(x)


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main():
    print("[load] reading 11 per-system CSVs ...")
    df = load_all_systems()
    print(f"       n_rows={len(df)}  "
          f"n_systems={df['system_id'].nunique()}  "
          f"n_methods={df['method'].nunique()}")
    print(f"       method distribution: "
          f"{df.groupby(['system_id','method']).size().unstack(fill_value=0)}")

    print("\n[S2 smoke] baro vs causal on openrca::bank and openrca::mkt1 ...")
    smoke = s2_smoke(df)
    bank, mkt1 = smoke["bank"], smoke["mkt1"]
    print(f"  Bank : Δ={_fmt(bank['mean_delta'])}  "
          f"CI=[{_fmt(bank['ci_low'])}, {_fmt(bank['ci_high'])}]  "
          f"n={bank['n_cases']}  width={smoke['bank_width_pp']:.2f}pp")
    print(f"  Mkt1 : Δ={_fmt(mkt1['mean_delta'])}  "
          f"CI=[{_fmt(mkt1['ci_low'])}, {_fmt(mkt1['ci_high'])}]  "
          f"n={mkt1['n_cases']}  width={smoke['mkt1_width_pp']:.2f}pp")
    print(f"  Bank sign (Δ>0): {smoke['bank_sign_ok']}")
    print(f"  Mkt1 sign (Δ<0): {smoke['mkt1_sign_ok']}")
    print(f"  CI non-overlap: {smoke['non_overlap_ok']}")
    print(f"  Bank CI width in [5,30]pp: {smoke['width_bank_ok']}")
    print(f"  Mkt1 CI width in [5,30]pp: {smoke['width_mkt1_ok']}")
    print(f"  S2 GATE: {'PASS' if smoke['passed'] else 'FAIL'}")

    if not smoke["passed"]:
        raise SystemExit(
            "[S2 gate] FAILED — aborting before full bootstrap. "
            "Review the diagnostics above."
        )

    print("\n[full] running paired bootstrap across all triples ...")
    out = run_full(df, OUT_CSV)
    print(f"       wrote {OUT_CSV} with {len(out)} rows")

    # Summary highlights
    sig = out.assign(abs_delta=out["mean_delta"].abs()).sort_values("abs_delta", ascending=False)
    print("\n[top 10 largest |Δ|]")
    print(sig.head(10)[["system", "method_a", "method_b", "n_cases",
                        "mean_delta", "ci_low", "ci_high", "p_two_sided"]].to_string(index=False))

    narrow = out.assign(ci_width=out["ci_high"] - out["ci_low"]).sort_values("ci_width")
    print("\n[top 5 narrowest CIs]")
    print(narrow.head(5)[["system","method_a","method_b","n_cases","mean_delta","ci_low","ci_high"]].to_string(index=False))
    print("\n[top 5 widest CIs]")
    print(narrow.tail(5)[["system","method_a","method_b","n_cases","mean_delta","ci_low","ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
