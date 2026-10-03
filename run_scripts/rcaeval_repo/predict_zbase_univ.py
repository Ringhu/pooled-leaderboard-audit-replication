#!/usr/bin/env python3
"""
Universal Z-score anomaly baseline.

Plugs into the existing run_rcaeval.py / run_petshop.py adapters without
modification — matches RCAEval's baro() call signature.

Core logic (stripped from predict_anomaly_baseline.py + predict_rcaagent_1min.py):
  - Split the input DataFrame by ``inject_time`` using the ``time`` column.
  - For each numeric metric column, compute |max Z-score| in the anomaly
    window relative to the baseline (pre-inject) window's mean/std.
  - Rank columns by |max Z-score| descending; ties broken alphabetically.

No parquet IO, no CANDIDATES, no onset timing, no causal graphs.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


def _degenerate_ranks(columns: list[str]) -> dict:
    """Fallback: stable alphabetic order of non-time columns."""
    return {"ranks": sorted(columns)}


def zbase_universal(
    data: pd.DataFrame,
    inject_time,
    *,
    dataset=None,
    sli=None,
    anomalies=None,
    n_iter=None,
    verbose: bool = False,
    **kwargs: Any,
) -> dict:
    """Universal Z-score anomaly baseline.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a ``time`` column (monotone, in the same units as
        ``inject_time``) plus one column per metric. Non-numeric metric
        columns are ignored. Adapters (run_rcaeval / run_petshop's
        ``build_combined``) are expected to have already synthesized a
        well-formed ``time`` axis — this function does not re-parse real
        timestamps.
    inject_time : numeric
        Split point. Rows with ``time < inject_time`` form the baseline;
        rows with ``time >= inject_time`` form the anomaly window.

    Returns
    -------
    dict
        ``{"ranks": [metric_col_name, ...]}`` sorted by |max Z-score|
        descending (ties: stable alphabetic).

    Extra kwargs (``dataset``, ``sli``, ``anomalies``, ``n_iter``,
    ``verbose``, plus any unknown kwargs via ``**kwargs``) are accepted
    for forward-compat with baro()'s signature and ignored.
    """
    # --- 1. Isolate candidate metric columns (numeric, not 'time') ---
    if "time" not in data.columns:
        warnings.warn(
            "zbase_universal: 'time' column missing; returning alphabetic ranks.",
            RuntimeWarning,
        )
        numeric_cols = [
            c for c in data.columns
            if pd.api.types.is_numeric_dtype(data[c])
        ]
        return _degenerate_ranks(numeric_cols)

    candidate_cols: list[str] = []
    for c in data.columns:
        if c == "time":
            continue
        if pd.api.types.is_numeric_dtype(data[c]):
            candidate_cols.append(c)

    if not candidate_cols:
        warnings.warn(
            "zbase_universal: no numeric metric columns found.",
            RuntimeWarning,
        )
        return {"ranks": []}

    # --- 2. Split by inject_time ---
    baseline = data[data["time"] < inject_time]
    anomaly = data[data["time"] >= inject_time]

    if baseline.empty or anomaly.empty:
        warnings.warn(
            f"zbase_universal: degenerate split "
            f"(baseline={len(baseline)} rows, anomaly={len(anomaly)} rows); "
            f"falling back to alphabetic ranks.",
            RuntimeWarning,
        )
        return _degenerate_ranks(candidate_cols)

    # --- 3. Per-column max |Z-score| ---
    scored: list[tuple[str, float]] = []
    for c in candidate_cols:
        base_col = baseline[c]
        mu = base_col.mean()
        sigma = base_col.std(ddof=0)

        # Degenerate metric: constant baseline or all-NaN -> z = 0
        if sigma is None or np.isnan(sigma) or sigma == 0:
            scored.append((c, 0.0))
            continue
        if np.isnan(mu):
            scored.append((c, 0.0))
            continue

        anom_col = anomaly[c].to_numpy(dtype=float, copy=False)
        # All-NaN anomaly window -> z = 0
        if not np.isfinite(anom_col).any():
            scored.append((c, 0.0))
            continue

        z_series = (anom_col - mu) / sigma
        abs_z = np.abs(z_series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_abs_z = float(np.nanmax(abs_z))
        if not np.isfinite(max_abs_z):
            max_abs_z = 0.0

        scored.append((c, max_abs_z))

    # --- 4. Rank: |max_z| descending; ties stable alphabetic ---
    # Python's sort is stable, so sort alphabetically first, then by -score.
    scored.sort(key=lambda t: t[0])
    scored.sort(key=lambda t: t[1], reverse=True)

    ranks = [name for name, _ in scored]

    if verbose:
        top = scored[: min(5, len(scored))]
        print(f"[zbase_universal] top-{len(top)}: {top}")

    return {"ranks": ranks}


# ----------------------------------------------------------------------
# Minimal self-test (offline, no datasets needed)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    inject = 50

    time_col = np.arange(n, dtype=float)
    metric_a = rng.normal(0.0, 1.0, size=n)
    metric_b = rng.normal(0.0, 1.0, size=n)
    metric_c = rng.normal(0.0, 1.0, size=n)
    # Inject a strong shift into metric_b after inject_time
    metric_b[inject:] = rng.normal(10.0, 1.0, size=n - inject)

    df = pd.DataFrame({
        "time": time_col,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "metric_c": metric_c,
    })

    out = zbase_universal(df, inject, dataset="synthetic", verbose=True)
    ranks = out["ranks"]
    print(f"ranks = {ranks}")
    assert ranks[0] == "metric_b", f"expected metric_b first, got {ranks!r}"

    # Edge case: empty baseline (inject_time before all rows)
    out_empty = zbase_universal(df, inject_time=-1.0)
    assert out_empty["ranks"] == sorted(["metric_a", "metric_b", "metric_c"]), (
        f"expected alphabetic fallback, got {out_empty['ranks']!r}"
    )

    # Edge case: empty anomaly (inject_time after all rows)
    out_empty2 = zbase_universal(df, inject_time=1e9)
    assert out_empty2["ranks"] == sorted(["metric_a", "metric_b", "metric_c"]), (
        f"expected alphabetic fallback, got {out_empty2['ranks']!r}"
    )

    print("SELF-TEST PASSED")
