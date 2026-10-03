#!/usr/bin/env python3
"""Per-system difficulty indicators for the 11-system RCA benchmark audit.

Reviewer's weak-accept condition #2: quantify per-system difficulty so we can
test whether the 12.7x cross-system BARO spread is explained by class imbalance
/ service-count variation.

Five indicators per system:
  1. majority_class_baseline = max(class_counts) / total_cases
  2. root_entropy            = Shannon entropy (bits) of GT-root distribution
  3. service_count           = distinct services in the GT label space
  4. baro_lift               = mean acc@1(baro) - majority_baseline
  5. zbase_lift              = mean acc@1(z_family) - majority_baseline

Ground-truth extraction conventions (decided after inspecting per-system CSVs
and prep.py — see reviewer note at bottom of file):

* RCAEval + PetShop: `true_service` column is the per-case ground-truth service
  label. Class = true_service. n_cases = unique cases.
* OpenRCA: per-case CSVs (openrca_{sys}_results_v2.csv) do NOT carry the GT
  root-cause service on each row (`true_service`/`fault` are NaN). Rows are
  scoring-subtask observations; `case = <sys>/task_N`. We take:
    - class = case (= task_N), since each task is a distinct incident scenario
      with its own underlying component root cause. `acc1_v2` is already the
      per-subtask scored accuracy under the OpenRCA official protocol.
    - service_count = distinct "root cause component is X" strings extracted
      from {sys}_q.csv `scoring_points`.
  The "majority baseline" is then the frequency of the most-sampled task, which
  also happens to be how often a blind "guess the modal case" would score under
  the same row-weighting the eval uses.

Outputs:
  results_stats/difficulty_table.csv
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Paths                                                              #
# ------------------------------------------------------------------ #
ROOT = Path("$HOME/auditstack")
INPUTS_DIR = ROOT / "results_xbench_11sys_with_zbase" / "_inputs"
OPENRCA_DIR = ROOT / "results_openrca_xbench"
Q_DIR = ROOT / "scripts" / "openrca_eval"
OUT_CSV = ROOT / "results_stats" / "difficulty_table.csv"

OPENRCA_SYSTEMS = ["bank", "mkt1", "mkt2", "tel"]
RCAEVAL_SYSTEMS = ["online-boutique", "sock-shop-2", "train-ticket"]
PETSHOP_SYSTEMS = ["high_traffic", "low_traffic",
                   "temporal_traffic1", "temporal_traffic2"]

Z_FAMILY_RAWS = {"zbase", "zbase_univ"}

RX_COMPONENT = re.compile(r"root cause component is ([A-Za-z0-9_\-]+)")


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _shannon_bits(counts: pd.Series) -> float:
    """Shannon entropy in bits of a counts series."""
    probs = counts.astype(float) / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _load_openrca(system: str) -> pd.DataFrame:
    path = OPENRCA_DIR / f"openrca_{system}_results_v2.csv"
    df = pd.read_csv(path)
    df = df[df["tdelta"] == 0].copy()
    df["correct"] = df["acc1_v2"].astype(float)
    df["method_raw"] = df["method"].astype(str)
    df["method"] = df["method_raw"].where(
        ~df["method_raw"].isin(Z_FAMILY_RAWS), "z_family"
    )
    return df


def _load_rcaeval(system: str) -> pd.DataFrame:
    path = INPUTS_DIR / f"rcaeval_merged_{system}_results.csv"
    df = pd.read_csv(path)
    if "tdelta" in df.columns:
        df = df[df["tdelta"] == 0].copy()
    df["correct"] = df["acc1_v2"].astype(float)
    df["method_raw"] = df["method"].astype(str)
    df["method"] = df["method_raw"].where(
        ~df["method_raw"].isin(Z_FAMILY_RAWS), "z_family"
    )
    return df


def _load_petshop(traffic: str) -> pd.DataFrame:
    path = INPUTS_DIR / f"petshop_merged_{traffic}_results.csv"
    df = pd.read_csv(path)
    if "tdelta" in df.columns:
        df = df[df["tdelta"] == 0].copy()
    df["correct"] = df["acc@1"].astype(float)
    df["method_raw"] = df["method"].astype(str)
    df["method"] = df["method_raw"].where(
        ~df["method_raw"].isin(Z_FAMILY_RAWS), "z_family"
    )
    return df


def _openrca_service_count(system: str) -> int:
    """Distinct 'root cause component is X' strings in {sys}_q.csv."""
    q_path = Q_DIR / f"{system}_q.csv"
    q = pd.read_csv(q_path)
    comps = set()
    for s in q["scoring_points"].fillna(""):
        comps.update(RX_COMPONENT.findall(s))
    return len(comps)


# ------------------------------------------------------------------ #
# Core API                                                           #
# ------------------------------------------------------------------ #
def compute_difficulty(
    system_df: pd.DataFrame,
    method_baro_df: pd.DataFrame,
    method_zfamily_df: pd.DataFrame,
    *,
    benchmark: str,
    system: str,
    service_count_override: int | None = None,
) -> dict:
    """Compute the 5 difficulty indicators for one system.

    Parameters
    ----------
    system_df : full per-system DataFrame (all methods), must have `case` +
                `correct` + `method`.
    method_baro_df : subset of system_df where method == 'baro'.
    method_zfamily_df : subset where method == 'z_family' (i.e. zbase or
                       zbase_univ collapsed).
    benchmark : 'openrca' / 'rcaeval' / 'petshop' — controls class and
                service_count source.
    system : system short label.
    service_count_override : for OpenRCA we pass the q_csv-derived count,
                             since the v2 per-row CSV doesn't encode services.

    Returns dict of indicators for this system.
    """
    # ---- class distribution ----
    if benchmark == "openrca":
        # class = case (each task is a distinct incident). Count observations
        # across all baro rows to match the row-weighting used by the eval.
        class_counts = method_baro_df["case"].value_counts()
    else:
        # RCAEval / PetShop: class = true_service (per-case label).
        # One observation per (case), so use groupby first to avoid
        # double-counting if tdelta duplicates still linger.
        per_case = (
            method_baro_df.groupby("case")["true_service"].first().dropna()
        )
        class_counts = per_case.value_counts()

    total_cases = int(class_counts.sum())
    majority_baseline = (
        float(class_counts.max()) / total_cases if total_cases > 0 else float("nan")
    )
    entropy_bits = _shannon_bits(class_counts) if total_cases > 0 else float("nan")

    # ---- service count ----
    if service_count_override is not None:
        service_count = int(service_count_override)
    else:
        # RCAEval / PetShop: distinct true_service
        service_count = int(method_baro_df["true_service"].dropna().nunique())

    # ---- method accuracies ----
    baro_acc1 = float(method_baro_df["correct"].mean()) if len(method_baro_df) else float("nan")
    zbase_acc1 = (
        float(method_zfamily_df["correct"].mean())
        if len(method_zfamily_df) else float("nan")
    )

    return {
        "benchmark": benchmark,
        "system": system,
        "n_cases": total_cases,
        "n_classes": int(len(class_counts)),
        "majority_baseline": majority_baseline,
        "root_entropy": entropy_bits,
        "service_count": service_count,
        "baro_acc1": baro_acc1,
        "zbase_acc1": zbase_acc1,
        "baro_lift": baro_acc1 - majority_baseline,
        "zbase_lift": zbase_acc1 - majority_baseline,
    }


def compute_all_systems(_unused_inputs_dir: Path | None = None) -> pd.DataFrame:
    """Build the 11-system difficulty table.

    Parameter is kept for API compat with the task spec; paths are resolved
    from module-level constants.
    """
    rows = []

    # ---- OpenRCA (4 systems) ----
    for sys in OPENRCA_SYSTEMS:
        df = _load_openrca(sys)
        baro = df[df["method"] == "baro"]
        zf = df[df["method"] == "z_family"]
        svc_count = _openrca_service_count(sys)
        rows.append(
            compute_difficulty(
                df, baro, zf,
                benchmark="openrca", system=sys,
                service_count_override=svc_count,
            )
        )

    # ---- RCAEval (3 systems) ----
    RCAEVAL_LABEL_MAP = {"online-boutique": "OB",
                         "sock-shop-2": "SS",
                         "train-ticket": "TT"}
    for sys in RCAEVAL_SYSTEMS:
        df = _load_rcaeval(sys)
        baro = df[df["method"] == "baro"]
        zf = df[df["method"] == "z_family"]
        rows.append(
            compute_difficulty(
                df, baro, zf,
                benchmark="rcaeval", system=RCAEVAL_LABEL_MAP[sys],
            )
        )

    # ---- PetShop (4 traffic regimes) ----
    for t in PETSHOP_SYSTEMS:
        df = _load_petshop(t)
        baro = df[df["method"] == "baro"]
        zf = df[df["method"] == "z_family"]
        rows.append(
            compute_difficulty(
                df, baro, zf,
                benchmark="petshop", system=t,
            )
        )

    out = pd.DataFrame(rows, columns=[
        "benchmark", "system", "n_cases", "n_classes",
        "majority_baseline", "root_entropy", "service_count",
        "baro_acc1", "zbase_acc1", "baro_lift", "zbase_lift",
    ])
    return out


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main():
    df = compute_all_systems()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print()
    # Print a readable table with 4-decimal floats
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200,
                           "display.max_columns", 20):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
