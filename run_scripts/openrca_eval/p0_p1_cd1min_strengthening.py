#!/usr/bin/env python3
"""P0/P1 strengthening analyses for the CD-1min main universe.

P0:
  - leave-one-suite-out diagnostic on the 4-method 11-system universe
  - descriptive family/subsystem variance decomposition
  - no-temporal PetShop sensitivity

P1:
  - PetShop exact-vs-canonical scoring sensitivity for CD-1min
  - RCAEval RE1-vs-RE2 secondary replication table
  - CD-1min adapter threshold/window sweep summary
  - method inclusion gate table
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

import multiverse_pair_analysis as mp  # noqa: E402
from variance_decomposition_full_coverage import decompose_pair  # noqa: E402


ROOT = Path("$HOME/mnt/3090/auditstack")
DEFAULT_LONG = ROOT / "results_stats/multiverse_cd1min_main/full_coverage_cd1min_long.csv"
DEFAULT_PER_SYSTEM_DELTAS = ROOT / "results_stats/multiverse_cd1min_main/per_system_deltas.csv"
DEFAULT_PAIR_SUMMARY = ROOT / "results_stats/multiverse_cd1min_main/pair_summary.csv"
DEFAULT_LOSO_SUMMARY = ROOT / "results_stats/multiverse_cd1min_main/loso_pair_summary.csv"
DEFAULT_CD1MIN_MAIN = ROOT / "results_stats/cd1min_adapt_main_11sys/per_case.csv"
DEFAULT_CD1MIN_MAIN_SUMMARY = ROOT / "results_stats/cd1min_adapt_main_11sys/summary.csv"
DEFAULT_CD1MIN_RE2_SUMMARY = ROOT / "results_stats/cd1min_adapt_full_secondary/summary.csv"
DEFAULT_SWEEP_ROOT = ROOT / "results_stats/cd1min_adapter_robustness"
DEFAULT_OUT = ROOT / "results_stats/p0_p1_cd1min_strengthening"

METHODS = ["baro", "z_family", "alertcount", "cd1min"]
DISPLAY_NAMES = {
    "baro": "BARO",
    "z_family": "max-|Z|",
    "alertcount": "alert-count",
    "cd1min": "CD-1min",
}
DROP_TEMPORAL = {"petshop::temporal_traffic1", "petshop::temporal_traffic2"}
RCAEVAL_SYSTEM_MAP = {
    "online-boutique": "OB",
    "sock-shop-2": "SS",
    "train-ticket": "TT",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)


def winner(a: float, b: float, method_a: str, method_b: str) -> str:
    if a == b:
        return "tie"
    return method_a if a > b else method_b


def system_means(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["benchmark", "system_id", "method"])
        .agg(acc1=("correct", "mean"), n_cases=("case_id", "nunique"))
        .reset_index()
    )


def suite_holdout(sys_mean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = sorted(sys_mean["benchmark"].unique())
    summary_rows = []
    per_system_rows = []
    weighted_rows = []
    for method_a, method_b in itertools.combinations(METHODS, 2):
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
            selected = winner(train_a, train_b, method_a, method_b)
            heldout_winner = winner(test_a, test_b, method_a, method_b)
            selected_acc = np.nan if selected == "tie" else float(test[selected].mean())
            regret = 0.0 if selected == "tie" else max(0.0, max(test_a, test_b) - selected_acc)
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

            train_aw = float(np.average(train[method_a], weights=train["n_cases"]))
            train_bw = float(np.average(train[method_b], weights=train["n_cases"]))
            test_aw = float(np.average(test[method_a], weights=test["n_cases"]))
            test_bw = float(np.average(test[method_b], weights=test["n_cases"]))
            selected_w = winner(train_aw, train_bw, method_a, method_b)
            heldout_winner_w = winner(test_aw, test_bw, method_a, method_b)
            selected_acc_w = np.nan if selected_w == "tie" else float(np.average(test[selected_w], weights=test["n_cases"]))
            regret_w = 0.0 if selected_w == "tie" else max(0.0, max(test_aw, test_bw) - selected_acc_w)
            weighted_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "pair": f"{DISPLAY_NAMES[method_a]} vs {DISPLAY_NAMES[method_b]}",
                "heldout_family": heldout,
                "aggregation": "n_cases_weighted",
                "train_acc_a": train_aw,
                "train_acc_b": train_bw,
                "heldout_acc_a": test_aw,
                "heldout_acc_b": test_bw,
                "selected_method": selected_w,
                "heldout_winner": heldout_winner_w,
                "mis_selected": 0 if selected_w == "tie" or heldout_winner_w == "tie" else int(selected_w != heldout_winner_w),
                "family_regret": regret_w,
                "family_regret_pp": regret_w * 100.0,
            })

            for _, row in test.iterrows():
                actual = winner(float(row[method_a]), float(row[method_b]), method_a, method_b)
                if selected == "tie":
                    sys_regret = 0.0
                    sys_mis = 0
                else:
                    sys_regret = max(0.0, max(float(row[method_a]), float(row[method_b])) - float(row[selected]))
                    sys_mis = 0 if actual == "tie" else int(selected != actual)
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
                    "system_winner": actual,
                    "mis_selected": sys_mis,
                    "system_regret": sys_regret,
                    "system_regret_pp": sys_regret * 100.0,
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(per_system_rows), pd.DataFrame(weighted_rows)


def variance_decomposition(per_system: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    families = []
    for _, sub in per_system.groupby("pair", sort=False):
        summary, family = decompose_pair(sub)
        summaries.append(summary)
        families.append(family)
    return pd.DataFrame(summaries), pd.concat(families, ignore_index=True)


def with_methods(methods: list[str], display_names: dict[str, str]):
    class MethodPatch:
        def __enter__(self):
            self.old_methods = mp.FULL_METHODS
            self.old_display = mp.DISPLAY_NAMES
            mp.FULL_METHODS = methods
            mp.DISPLAY_NAMES = display_names

        def __exit__(self, exc_type, exc, tb):
            mp.FULL_METHODS = self.old_methods
            mp.DISPLAY_NAMES = self.old_display
    return MethodPatch()


def recompute_multiverse(df: pd.DataFrame, n_iter: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with with_methods(METHODS, DISPLAY_NAMES):
        per_system, pair_summary = mp.compute_pair_deltas(df, n_iter, seed)
        loso_summary, loso_per_system = mp.compute_loso(df, n_iter, seed)
    return per_system, pair_summary, loso_summary, loso_per_system


def no_temporal_sensitivity(df: pd.DataFrame, full_pair: pd.DataFrame, full_loso: pd.DataFrame, n_iter: int, seed: int, out_dir: Path) -> pd.DataFrame:
    filt = df[~df["system_id"].isin(DROP_TEMPORAL)].copy()
    per_system, pair_summary, loso_summary, loso_per_system = recompute_multiverse(filt, n_iter, seed)
    per_system.to_csv(out_dir / "no_temporal_per_system_deltas.csv", index=False)
    pair_summary.to_csv(out_dir / "no_temporal_pair_summary.csv", index=False)
    loso_summary.to_csv(out_dir / "no_temporal_loso_summary.csv", index=False)
    loso_per_system.to_csv(out_dir / "no_temporal_loso_per_system.csv", index=False)

    merged = full_pair.merge(
        pair_summary,
        on=["method_a", "method_b", "pair"],
        suffixes=("_full", "_no_temporal"),
    ).merge(
        full_loso[["method_a", "method_b", "mis_selection_count", "mean_regret_pp", "max_regret_pp"]],
        on=["method_a", "method_b"],
        how="left",
    ).merge(
        loso_summary[["method_a", "method_b", "mis_selection_count", "mean_regret_pp", "max_regret_pp"]],
        on=["method_a", "method_b"],
        how="left",
        suffixes=("_loso_full", "_loso_no_temporal"),
    )
    keep = [
        "pair",
        "k_full", "k_no_temporal",
        "sign_changing_effects_full", "sign_changing_effects_no_temporal",
        "I2_pct_full", "I2_pct_no_temporal",
        "pi_crosses_zero_full", "pi_crosses_zero_no_temporal",
        "delta_RE_pp_full", "delta_RE_pp_no_temporal",
        "mis_selection_count_loso_full", "mis_selection_count_loso_no_temporal",
        "mean_regret_pp_loso_full", "mean_regret_pp_loso_no_temporal",
        "max_regret_pp_loso_full", "max_regret_pp_loso_no_temporal",
    ]
    comp = merged[keep].copy()
    comp.to_csv(out_dir / "no_temporal_comparison.csv", index=False)
    return comp


def cd1min_canonical_long(df: pd.DataFrame, cd1min_per_case: Path) -> pd.DataFrame:
    out = df.copy()
    per_case = pd.read_csv(cd1min_per_case)
    canon_rows = []
    for row in per_case.to_dict("records"):
        if row["family"] != "petshop":
            continue
        system_id = f"petshop::{row['system']}"
        case = str(row["case"])
        canon_rows.append({
            "case_id": f"{system_id}|{case}#0",
            "correct_canon": float(row["acc@1_canon"]),
        })
    canon = pd.DataFrame(canon_rows)
    out = out.merge(canon, on="case_id", how="left")
    mask = (out["method"] == "cd1min") & out["correct_canon"].notna()
    out.loc[mask, "correct"] = out.loc[mask, "correct_canon"].astype(float)
    return out.drop(columns=["correct_canon"])


def canonical_sensitivity(df: pd.DataFrame, full_pair: pd.DataFrame, full_loso: pd.DataFrame, cd1min_per_case: Path, n_iter: int, seed: int, out_dir: Path) -> pd.DataFrame:
    canon_df = cd1min_canonical_long(df, cd1min_per_case)
    per_system, pair_summary, loso_summary, loso_per_system = recompute_multiverse(canon_df, n_iter, seed)
    per_system.to_csv(out_dir / "canonical_petshop_per_system_deltas.csv", index=False)
    pair_summary.to_csv(out_dir / "canonical_petshop_pair_summary.csv", index=False)
    loso_summary.to_csv(out_dir / "canonical_petshop_loso_summary.csv", index=False)
    loso_per_system.to_csv(out_dir / "canonical_petshop_loso_per_system.csv", index=False)

    cd_pairs = pair_summary[(pair_summary["method_a"] == "cd1min") | (pair_summary["method_b"] == "cd1min")]
    merged = full_pair.merge(
        cd_pairs,
        on=["method_a", "method_b", "pair"],
        suffixes=("_exact", "_canon"),
    ).merge(
        full_loso[["method_a", "method_b", "mis_selection_count", "mean_regret_pp", "max_regret_pp"]],
        on=["method_a", "method_b"],
        how="left",
    ).merge(
        loso_summary[["method_a", "method_b", "mis_selection_count", "mean_regret_pp", "max_regret_pp"]],
        on=["method_a", "method_b"],
        how="left",
        suffixes=("_loso_exact", "_loso_canon"),
    )
    comp = merged[[
        "pair",
        "sign_changing_effects_exact", "sign_changing_effects_canon",
        "I2_pct_exact", "I2_pct_canon",
        "pi_crosses_zero_exact", "pi_crosses_zero_canon",
        "delta_RE_pp_exact", "delta_RE_pp_canon",
        "mis_selection_count_loso_exact", "mis_selection_count_loso_canon",
        "mean_regret_pp_loso_exact", "mean_regret_pp_loso_canon",
        "max_regret_pp_loso_exact", "max_regret_pp_loso_canon",
    ]].copy()
    comp.to_csv(out_dir / "canonical_petshop_comparison.csv", index=False)

    sys_mean = system_means(canon_df)
    sys_mean[sys_mean["method"] == "cd1min"].to_csv(
        out_dir / "canonical_petshop_cd1min_system_means.csv",
        index=False,
    )
    return comp


def re1_re2_replication(main_summary_path: Path, re2_summary_path: Path, out_dir: Path) -> pd.DataFrame:
    main = pd.read_csv(main_summary_path)
    re2 = pd.read_csv(re2_summary_path)
    cols = ["family", "system", "method", "n_attempted", "n_ok", "n_error", "acc1", "acc3", "acc5", "acc10"]
    main = main[cols].rename(columns={c: f"{c}_RE1_main" for c in cols if c not in {"family", "system", "method"}})
    re2 = re2[cols].rename(columns={c: f"{c}_RE2_secondary" for c in cols if c not in {"family", "system", "method"}})
    merged = main.merge(re2, on=["family", "system", "method"], how="outer")
    merged["acc1_delta_RE1_minus_RE2"] = merged["acc1_RE1_main"] - merged["acc1_RE2_secondary"]
    merged.to_csv(out_dir / "cd1min_re1_re2_secondary_replication.csv", index=False)
    return merged


def summarize_sweeps(sweep_root: Path, default_summary_path: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    if default_summary_path.exists():
        default = pd.read_csv(default_summary_path)
        default = default.assign(config="default_z3_pct5_win10", source=str(default_summary_path))
        rows.append(default)
    if sweep_root.exists():
        for path in sorted(sweep_root.glob("*/summary.csv")):
            df = pd.read_csv(path)
            df = df.assign(config=path.parent.name, source=str(path))
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    all_summ = pd.concat(rows, ignore_index=True)
    all_summ.to_csv(out_dir / "cd1min_adapter_sweep_by_system.csv", index=False)

    macro = all_summ.groupby("config").agg(
        n_systems=("system", "nunique"),
        total_attempted=("n_attempted", "sum"),
        total_error=("n_error", "sum"),
        macro_acc1=("acc1", "mean"),
        min_acc1=("acc1", "min"),
        max_acc1=("acc1", "max"),
        macro_acc3=("acc3", "mean"),
        macro_acc10=("acc10", "mean"),
        mean_modal_freq=("modal_freq", "mean"),
    ).reset_index()
    macro.to_csv(out_dir / "cd1min_adapter_sweep_macro.csv", index=False)
    return macro


def inclusion_gate(out_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "method": "BARO",
            "openrca": "yes",
            "rcaeval": "yes",
            "petshop": "yes",
            "main_experiment": "yes",
            "reason": "Existing 11/11 full-coverage rows in the main universe.",
        },
        {
            "method": "max-|Z|",
            "openrca": "yes",
            "rcaeval": "yes",
            "petshop": "yes",
            "main_experiment": "yes",
            "reason": "Unified z-family baseline covers all 11 systems.",
        },
        {
            "method": "alert-count",
            "openrca": "yes",
            "rcaeval": "yes",
            "petshop": "yes",
            "main_experiment": "yes",
            "reason": "Existing alert-count block aligns with all 778 scoring units.",
        },
        {
            "method": "CD-1min adapter",
            "openrca": "yes",
            "rcaeval": "yes",
            "petshop": "yes",
            "main_experiment": "yes",
            "reason": "Fresh full adapter run completed 11/11 systems and aligns with all 778 scoring units.",
        },
        {
            "method": "CIRCA",
            "openrca": "adapter-only",
            "rcaeval": "partial",
            "petshop": "partial",
            "main_experiment": "no",
            "reason": "Fresh diagnosis: RCAEval Train-Ticket timed out at limit=1; PetShop temporal systems timed out under bounded smoke.",
        },
        {
            "method": "Causal adapter",
            "openrca": "yes",
            "rcaeval": "smoke only",
            "petshop": "smoke only",
            "main_experiment": "no",
            "reason": "Cross-benchmark graph protocol is adapter-defined and only smoke-tested, not a full 11/11 matched main run.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "method_inclusion_gate.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-long", type=Path, default=DEFAULT_LONG)
    ap.add_argument("--per-system-deltas", type=Path, default=DEFAULT_PER_SYSTEM_DELTAS)
    ap.add_argument("--pair-summary", type=Path, default=DEFAULT_PAIR_SUMMARY)
    ap.add_argument("--loso-summary", type=Path, default=DEFAULT_LOSO_SUMMARY)
    ap.add_argument("--cd1min-per-case", type=Path, default=DEFAULT_CD1MIN_MAIN)
    ap.add_argument("--cd1min-main-summary", type=Path, default=DEFAULT_CD1MIN_MAIN_SUMMARY)
    ap.add_argument("--cd1min-re2-summary", type=Path, default=DEFAULT_CD1MIN_RE2_SUMMARY)
    ap.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap-n-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_long)
    full_pair = pd.read_csv(args.pair_summary)
    full_loso = pd.read_csv(args.loso_summary)
    per_system = pd.read_csv(args.per_system_deltas)

    cov = (
        df.groupby(["system_id", "method"])["case_id"]
        .nunique()
        .unstack(fill_value=0)
        .astype(int)
        .reset_index()
    )
    cov.to_csv(args.output_dir / "p0_method_coverage.csv", index=False)

    sys_mean = system_means(df)
    sys_mean.to_csv(args.output_dir / "p0_system_method_means.csv", index=False)

    suite_summary, suite_per_system, suite_weighted = suite_holdout(sys_mean)
    suite_summary.to_csv(args.output_dir / "p0_suite_holdout_summary.csv", index=False)
    suite_per_system.to_csv(args.output_dir / "p0_suite_holdout_per_system.csv", index=False)
    suite_weighted.to_csv(args.output_dir / "p0_suite_holdout_weighted_sensitivity.csv", index=False)

    var_summary, var_family = variance_decomposition(per_system)
    var_summary.to_csv(args.output_dir / "p0_variance_decomposition.csv", index=False)
    var_family.to_csv(args.output_dir / "p0_family_delta_summary.csv", index=False)

    no_temp = no_temporal_sensitivity(
        df, full_pair, full_loso, args.bootstrap_n_iter, args.seed, args.output_dir
    )
    canon = canonical_sensitivity(
        df, full_pair, full_loso, args.cd1min_per_case, args.bootstrap_n_iter, args.seed, args.output_dir
    )
    re1_re2 = re1_re2_replication(args.cd1min_main_summary, args.cd1min_re2_summary, args.output_dir)
    sweeps = summarize_sweeps(args.sweep_root, args.cd1min_main_summary, args.output_dir)
    gates = inclusion_gate(args.output_dir)

    summary = {
        "input_long": str(args.input_long),
        "n_rows": int(len(df)),
        "n_scoring_units": int(df["case_id"].nunique()),
        "n_systems": int(df["system_id"].nunique()),
        "methods": METHODS,
        "p0_suite_holdout_rows": int(len(suite_summary)),
        "p0_variance_rows": int(len(var_summary)),
        "p0_no_temporal_pairs": int(len(no_temp)),
        "p1_canonical_pairs": int(len(canon)),
        "p1_re1_re2_rows": int(len(re1_re2)),
        "p1_sweep_configs": int(sweeps["config"].nunique()) if not sweeps.empty else 0,
        "p1_gate_methods": int(len(gates)),
    }
    write_json(args.output_dir / "summary.json", summary)

    print(f"[written] {args.output_dir}")
    print("\n[P0 suite-holdout]")
    print(suite_summary.to_string(index=False))
    print("\n[P0 variance]")
    print(var_summary.to_string(index=False))
    print("\n[P0 no-temporal comparison]")
    print(no_temp.to_string(index=False))
    print("\n[P1 canonical PetShop comparison]")
    print(canon.to_string(index=False))
    print("\n[P1 RE1-vs-RE2]")
    print(re1_re2.to_string(index=False))
    if not sweeps.empty:
        print("\n[P1 adapter sweep macro]")
        print(sweeps.to_string(index=False))
    print("\n[P1 inclusion gate]")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
