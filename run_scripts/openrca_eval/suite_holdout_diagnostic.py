#!/usr/bin/env python3
"""Leave-one-suite-out method-selection diagnostic for full-coverage methods.

Consumes the long table produced by multiverse_pair_analysis.py and reports
whether a method chosen from two benchmark families would also be the winner
on the held-out family. This is a diagnostic only: there are three families,
so no population-level inference should be drawn from these folds.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_REPO_ROOT = Path("$HOME/mnt/3090/auditstack")
DEFAULT_INPUT = DEFAULT_REPO_ROOT / "results_stats" / "multiverse_full_coverage" / "full_coverage_long.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "results_stats" / "suite_holdout_full_coverage"

FULL_METHODS = ["baro", "z_family", "alertcount"]
DISPLAY_NAMES = {
    "baro": "BARO",
    "z_family": "max-|Z|",
    "alertcount": "alert-count",
}


def system_means(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["benchmark", "system_id", "method"])
        .agg(acc1=("correct", "mean"), n_cases=("case_id", "nunique"))
        .reset_index()
    )


def _winner(a: float, b: float, method_a: str, method_b: str) -> str:
    if a == b:
        return "tie"
    return method_a if a > b else method_b


def macro_suite_holdout(sys_mean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    per_system_rows = []
    families = sorted(sys_mean["benchmark"].unique())

    for method_a, method_b in itertools.combinations(FULL_METHODS, 2):
        pair_df = sys_mean[sys_mean["method"].isin([method_a, method_b])].copy()
        wide = pair_df.pivot_table(
            index=["benchmark", "system_id", "n_cases"],
            columns="method",
            values="acc1",
            aggfunc="mean",
        ).dropna(subset=[method_a, method_b]).reset_index()

        for heldout in families:
            train = wide[wide["benchmark"] != heldout]
            test = wide[wide["benchmark"] == heldout]
            train_a = float(train[method_a].mean())
            train_b = float(train[method_b].mean())
            test_a = float(test[method_a].mean())
            test_b = float(test[method_b].mean())

            selected = _winner(train_a, train_b, method_a, method_b)
            heldout_winner = _winner(test_a, test_b, method_a, method_b)
            selected_acc = np.nan if selected == "tie" else float(test[selected].mean())
            best_acc = max(test_a, test_b)
            regret = 0.0 if selected == "tie" else max(0.0, best_acc - selected_acc)
            mis_selected = 0 if selected == "tie" or heldout_winner == "tie" else int(selected != heldout_winner)

            summary_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
                "heldout_family": heldout,
                "train_families": "+".join(sorted(set(families) - {heldout})),
                "n_train_systems": int(train["system_id"].nunique()),
                "n_heldout_systems": int(test["system_id"].nunique()),
                "train_acc_a": train_a,
                "train_acc_b": train_b,
                "heldout_acc_a": test_a,
                "heldout_acc_b": test_b,
                "selected_method": selected,
                "heldout_winner": heldout_winner,
                "mis_selected": mis_selected,
                "family_regret": regret,
                "family_regret_pp": regret * 100.0,
            })

            for _, row in test.iterrows():
                actual_winner = _winner(float(row[method_a]), float(row[method_b]), method_a, method_b)
                if selected == "tie":
                    sys_regret = 0.0
                    sys_mis = 0
                else:
                    sys_best = max(float(row[method_a]), float(row[method_b]))
                    sys_regret = max(0.0, sys_best - float(row[selected]))
                    sys_mis = 0 if actual_winner == "tie" else int(selected != actual_winner)
                per_system_rows.append({
                    "method_a": method_a,
                    "method_b": method_b,
                    "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
                    "heldout_family": heldout,
                    "system": row["system_id"],
                    "n_cases": int(row["n_cases"]),
                    "system_acc_a": float(row[method_a]),
                    "system_acc_b": float(row[method_b]),
                    "selected_method": selected,
                    "system_winner": actual_winner,
                    "mis_selected": sys_mis,
                    "system_regret": sys_regret,
                    "system_regret_pp": sys_regret * 100.0,
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(per_system_rows)


def weighted_suite_holdout(sys_mean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = sorted(sys_mean["benchmark"].unique())
    for method_a, method_b in itertools.combinations(FULL_METHODS, 2):
        pair_df = sys_mean[sys_mean["method"].isin([method_a, method_b])].copy()
        wide = pair_df.pivot_table(
            index=["benchmark", "system_id", "n_cases"],
            columns="method",
            values="acc1",
            aggfunc="mean",
        ).dropna(subset=[method_a, method_b]).reset_index()
        for heldout in families:
            train = wide[wide["benchmark"] != heldout]
            test = wide[wide["benchmark"] == heldout]
            train_a = float(np.average(train[method_a], weights=train["n_cases"]))
            train_b = float(np.average(train[method_b], weights=train["n_cases"]))
            test_a = float(np.average(test[method_a], weights=test["n_cases"]))
            test_b = float(np.average(test[method_b], weights=test["n_cases"]))
            selected = _winner(train_a, train_b, method_a, method_b)
            heldout_winner = _winner(test_a, test_b, method_a, method_b)
            selected_acc = np.nan if selected == "tie" else float(np.average(test[selected], weights=test["n_cases"]))
            regret = 0.0 if selected == "tie" else max(0.0, max(test_a, test_b) - selected_acc)
            rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
                "heldout_family": heldout,
                "aggregation": "n_cases_weighted",
                "train_acc_a": train_a,
                "train_acc_b": train_b,
                "heldout_acc_a": test_a,
                "heldout_acc_b": test_b,
                "selected_method": selected,
                "heldout_winner": heldout_winner,
                "mis_selected": 0 if selected == "tie" or heldout_winner == "tie" else int(selected != heldout_winner),
                "family_regret": regret,
                "family_regret_pp": regret * 100.0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-long", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_long)
    missing = sorted(set(FULL_METHODS) - set(df["method"].unique()))
    if missing:
        raise RuntimeError(f"Input is missing methods: {missing}")
    sys_mean = system_means(df)
    sys_mean.to_csv(args.output_dir / "system_method_means.csv", index=False)

    summary, per_system = macro_suite_holdout(sys_mean)
    summary.to_csv(args.output_dir / "suite_holdout_summary.csv", index=False)
    per_system.to_csv(args.output_dir / "suite_holdout_per_system.csv", index=False)
    weighted = weighted_suite_holdout(sys_mean)
    weighted.to_csv(args.output_dir / "suite_holdout_weighted_sensitivity.csv", index=False)

    print(f"[written] {args.output_dir}")
    print("\n[macro suite-held-out summary]")
    print(summary.to_string(index=False))
    print("\n[n-cases weighted sensitivity]")
    print(weighted.to_string(index=False))


if __name__ == "__main__":
    main()
