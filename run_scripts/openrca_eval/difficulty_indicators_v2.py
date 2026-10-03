#!/usr/bin/env python3
"""Step 2 of propagation-aware difficulty proxy.

Extends `difficulty_indicators.py` by appending two structural columns:

  * ``anomalous_metric_count_mean`` — mean over cases of the per-case
    count of metric columns whose |max Z-score| in the anomaly window
    exceeds 3 (same Z-score logic as `predict_zbase_universal`).
  * ``anomalous_metric_count_cv`` — coefficient of variation (std / mean,
    ddof=0) across cases within the system. Captures whether propagation
    spread is case-stable or case-variable.

Input CSVs (produced by compute_anomalous_metric_count.py):
  * results_stats/anomalous_metric_count_per_case.csv — may have OpenRCA
    rows sub-sampled to 30/system. We prefer the full side-car for
    OpenRCA when present so that per-system aggregates are unbiased.
  * results_stats/anomalous_metric_count_openrca_full.csv — full (non-
    sampled) OpenRCA per-case rows.

Output:
  results_stats/difficulty_table_v2.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from difficulty_indicators import compute_all_systems as _v1_compute_all_systems

ROOT = Path("$HOME/auditstack")
PER_CASE_CSV = ROOT / "results_stats" / "anomalous_metric_count_per_case.csv"
PER_CASE_OPENRCA_FULL = ROOT / "results_stats" / "anomalous_metric_count_openrca_full.csv"
OUT_CSV = ROOT / "results_stats" / "difficulty_table_v2.csv"


def _load_full_per_case() -> pd.DataFrame:
    """Return a per-case DataFrame with the FULL (non-sub-sampled) rows.

    For OpenRCA we swap in the side-car (if it exists) which was written by
    compute_anomalous_metric_count.py before any sub-sampling.
    """
    base = pd.read_csv(PER_CASE_CSV)
    if PER_CASE_OPENRCA_FULL.exists():
        full_openrca = pd.read_csv(PER_CASE_OPENRCA_FULL)
        # drop base's (possibly sampled) openrca rows, replace with full.
        base = base[base["benchmark"] != "openrca"]
        base = pd.concat([base, full_openrca], ignore_index=True)
    # Drop failed rows (NaN anom count) — they shouldn't bias the mean/CV.
    base = base.dropna(subset=["anomalous_metric_count"]).copy()
    return base


def _agg_anomalous(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["benchmark", "system"], as_index=False)
    out = grp.agg(
        n_cases_struct=("anomalous_metric_count", "size"),
        anomalous_metric_count_mean=("anomalous_metric_count", "mean"),
        anomalous_metric_count_std=("anomalous_metric_count",
                                    lambda x: float(x.std(ddof=0))),
    )
    out["anomalous_metric_count_cv"] = np.where(
        out["anomalous_metric_count_mean"] > 0,
        out["anomalous_metric_count_std"] / out["anomalous_metric_count_mean"],
        np.nan,
    )
    return out


def main():
    v1 = _v1_compute_all_systems()
    per_case = _load_full_per_case()
    agg = _agg_anomalous(per_case)

    merged = v1.merge(agg, on=["benchmark", "system"], how="left")

    # Spec-specified sanity checks
    absurd_low = merged["anomalous_metric_count_mean"] < 1
    absurd_high = merged["anomalous_metric_count_mean"] > 1000
    warn_systems: list[str] = []
    for _, r in merged[absurd_low | absurd_high].iterrows():
        warn_systems.append(
            f"  [sanity] {r['benchmark']}::{r['system']} "
            f"anom_mean={r['anomalous_metric_count_mean']:.2f} "
            f"(total={r.get('service_count', 'n/a')})"
        )
    if warn_systems:
        print("[WARN] some systems have anom_mean outside [1, 1000]:")
        print("\n".join(warn_systems))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Column order: existing v1 cols + new propagation proxy cols
    v1_cols = list(v1.columns)
    new_cols = ["n_cases_struct", "anomalous_metric_count_mean",
                "anomalous_metric_count_std", "anomalous_metric_count_cv"]
    merged = merged.reindex(columns=v1_cols + new_cols)
    merged.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 220, "display.max_columns", 40):
        print()
        print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
