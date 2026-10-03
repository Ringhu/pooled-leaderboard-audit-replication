#!/usr/bin/env python3
"""Qualitative diagnostics for LOSO flip systems.

This script summarizes the systems where full-coverage LOSO pooled selection
mis-selects the held-out winner. It joins:

  - LOSO per-system details
  - per-system deltas
  - structural difficulty indicators
  - method prediction concentration where top-1 labels are available

The output is intended for appendix/descriptive use. It should not be used to
claim causal mechanisms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_REPO_ROOT = Path("$HOME/mnt/3090/auditstack")
DEFAULT_MULTI_DIR = DEFAULT_REPO_ROOT / "results_stats" / "multiverse_full_coverage"
DEFAULT_DIFF = DEFAULT_REPO_ROOT / "results_stats" / "difficulty_table_v2.csv"
DEFAULT_UNIFIED_INPUTS = DEFAULT_REPO_ROOT / "results_xbench_11sys_unified" / "_inputs"
DEFAULT_ALERT_INPUTS = DEFAULT_REPO_ROOT / "results_xbench_11sys_alertcount" / "_inputs"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "results_stats" / "flip_system_diagnostic"

SYSTEM_FILES = [
    ("openrca", "bank", "openrca_bank_results_v2.csv", "acc1_v2"),
    ("openrca", "mkt1", "openrca_mkt1_results_v2.csv", "acc1_v2"),
    ("openrca", "mkt2", "openrca_mkt2_results_v2.csv", "acc1_v2"),
    ("openrca", "tel", "openrca_tel_results_v2.csv", "acc1_v2"),
    ("petshop", "high_traffic", "petshop_merged_high_traffic_results.csv", "acc@1"),
    ("petshop", "low_traffic", "petshop_merged_low_traffic_results.csv", "acc@1"),
    ("petshop", "temporal_traffic1", "petshop_merged_temporal_traffic1_results.csv", "acc@1"),
    ("petshop", "temporal_traffic2", "petshop_merged_temporal_traffic2_results.csv", "acc@1"),
    ("rcaeval", "OB", "rcaeval_merged_online-boutique_results.csv", "acc1_v2"),
    ("rcaeval", "SS", "rcaeval_merged_sock-shop-2_results.csv", "acc1_v2"),
    ("rcaeval", "TT", "rcaeval_merged_train-ticket_results.csv", "acc1_v2"),
]

SYSTEM_ID_TO_DIFF_SYSTEM = {
    "openrca::bank": ("openrca", "bank"),
    "openrca::mkt1": ("openrca", "mkt1"),
    "openrca::mkt2": ("openrca", "mkt2"),
    "openrca::tel": ("openrca", "tel"),
    "petshop::high_traffic": ("petshop", "high_traffic"),
    "petshop::low_traffic": ("petshop", "low_traffic"),
    "petshop::temporal_traffic1": ("petshop", "temporal_traffic1"),
    "petshop::temporal_traffic2": ("petshop", "temporal_traffic2"),
    "rcaeval::OB": ("rcaeval", "OB"),
    "rcaeval::SS": ("rcaeval", "SS"),
    "rcaeval::TT": ("rcaeval", "TT"),
}


def _read_method_rows(inputs_dir: Path, method_map: dict[str, str]) -> pd.DataFrame:
    frames = []
    for benchmark, system, filename, acc_col in SYSTEM_FILES:
        path = inputs_dir / filename
        df = pd.read_csv(path)
        if "tdelta" in df.columns:
            df = df[df["tdelta"] == 0].copy()
        if "method" not in df.columns or acc_col not in df.columns:
            continue
        df = df[df["method"].isin(method_map.keys())].copy()
        if df.empty:
            continue
        df["method"] = df["method"].map(method_map)
        df["benchmark"] = benchmark
        df["system"] = system
        df["system_id"] = f"{benchmark}::{system}"
        df["q_idx"] = df.groupby(["method", "case"]).cumcount()
        df["case_id"] = df["system_id"] + "|" + df["case"].astype(str) + "#" + df["q_idx"].astype(str)
        top_col = "top1_v2" if "top1_v2" in df.columns else ("top1" if "top1" in df.columns else None)
        df["top_prediction"] = df[top_col].astype(str) if top_col else ""
        df.loc[df["top_prediction"].isin(["", "nan", "None"]), "top_prediction"] = pd.NA
        df["correct"] = df[acc_col].astype(float)
        frames.append(df[["system_id", "case_id", "method", "correct", "top_prediction"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prediction_concentration(unified_inputs: Path, alert_inputs: Path) -> pd.DataFrame:
    unified = _read_method_rows(unified_inputs, {"baro": "baro", "zbase_univ": "z_family"})
    alert = _read_method_rows(alert_inputs, {"alertcount": "alertcount"})
    df = pd.concat([unified, alert], ignore_index=True)
    rows = []
    for (system, method), sub in df.groupby(["system_id", "method"]):
        nonmissing = sub.dropna(subset=["top_prediction"])
        if len(nonmissing):
            counts = nonmissing["top_prediction"].value_counts()
            modal_top1 = str(counts.index[0])
            modal_freq = float(counts.iloc[0] / len(nonmissing))
            n_unique = int(counts.size)
            n_top1 = int(len(nonmissing))
        else:
            modal_top1 = ""
            modal_freq = float("nan")
            n_unique = 0
            n_top1 = 0
        rows.append({
            "system": system,
            "method": method,
            "acc1": float(sub["correct"].mean()),
            "n_cases": int(sub["case_id"].nunique()),
            "n_top1_available": n_top1,
            "modal_top1": modal_top1,
            "modal_freq": modal_freq,
            "n_unique_top1": n_unique,
        })
    return pd.DataFrame(rows).sort_values(["system", "method"]).reset_index(drop=True)


def load_difficulty(path: Path) -> pd.DataFrame:
    diff = pd.read_csv(path)
    diff["system"] = diff.apply(lambda r: f"{r['benchmark']}::{r['system']}", axis=1)
    return diff[[
        "system",
        "majority_baseline",
        "root_entropy",
        "service_count",
        "anomalous_metric_count_mean",
        "anomalous_metric_count_cv",
    ]]


def classify_regret(regret_pp: float) -> str:
    if regret_pp >= 10:
        return "large"
    if regret_pp >= 2:
        return "moderate"
    return "near_tie_or_mild"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multiverse-dir", type=Path, default=DEFAULT_MULTI_DIR)
    ap.add_argument("--difficulty-csv", type=Path, default=DEFAULT_DIFF)
    ap.add_argument("--unified-inputs", type=Path, default=DEFAULT_UNIFIED_INPUTS)
    ap.add_argument("--alert-inputs", type=Path, default=DEFAULT_ALERT_INPUTS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loso = pd.read_csv(args.multiverse_dir / "loso_pair_per_system.csv")
    deltas = pd.read_csv(args.multiverse_dir / "per_system_deltas.csv")
    diff = load_difficulty(args.difficulty_csv)
    conc = prediction_concentration(args.unified_inputs, args.alert_inputs)

    flips = loso[loso["mis_selected"] == 1].copy()
    flips["regret_class"] = flips["regret_pp"].map(classify_regret)
    flips = flips.merge(diff, on="system", how="left")
    flips = flips.merge(
        deltas[["method_a", "method_b", "system", "delta_pp", "ci_lo_pp", "ci_hi_pp"]],
        on=["method_a", "method_b", "system"],
        how="left",
    )

    # Add concentration for pooled and actual winners where available.
    conc_key = conc.set_index(["system", "method"])
    for role in ["pooled_winner", "actual_winner"]:
        for field in ["acc1", "modal_freq", "modal_top1", "n_unique_top1", "n_top1_available"]:
            vals = []
            for _, row in flips.iterrows():
                method = row[role]
                key = (row["system"], method)
                vals.append(conc_key.loc[key, field] if key in conc_key.index else pd.NA)
            flips[f"{role}_{field}"] = vals

    union_systems = sorted(flips["system"].unique())
    conc_flip_systems = conc[conc["system"].isin(union_systems)].copy()

    notes = {
        "n_flip_rows": int(len(flips)),
        "n_unique_flip_systems": int(len(union_systems)),
        "unique_flip_systems": union_systems,
        "interpretation_boundary": (
            "Descriptive diagnostic only. These features help explain where "
            "mis-selection appears but do not establish a causal mechanism."
        ),
        "top1_availability_note": (
            "OpenRCA input CSVs often lack top1 labels for these methods; "
            "modal prediction fields are therefore missing for those rows."
        ),
    }

    flips.to_csv(args.output_dir / "flip_system_summary.csv", index=False)
    conc.to_csv(args.output_dir / "method_prediction_concentration_all.csv", index=False)
    conc_flip_systems.to_csv(args.output_dir / "method_prediction_concentration_flip_systems.csv", index=False)
    with open(args.output_dir / "diagnostic_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=float)

    print(f"[written] {args.output_dir}")
    show_cols = [
        "pair", "system", "pooled_winner", "actual_winner", "regret_pp",
        "regret_class", "delta_pp", "majority_baseline", "root_entropy",
        "anomalous_metric_count_mean", "pooled_winner_modal_freq",
        "actual_winner_modal_freq",
    ]
    print(flips[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()

