#!/usr/bin/env python3
"""Alert-count portable root-cause baseline (Stream 2).

Per-service count of metrics whose anomaly-window |max Z-score| exceeds a
threshold (default 3.0). Services ranked by count descending; ties broken
alphabetically. Within each service, metrics are ordered by |max_z| so
the returned ranks list (metric columns) preserves service-rank order
for acc_at_k() semantics.

Matches `zbase_universal`'s call signature so it plugs into the existing
RCAEval / PetShop / OpenRCA adapters without modification.

Formula:
  for each metric col c:
      |max_z|_c = |max_t (X_c(t) - μ_base) / σ_base|
  for each service s:
      alert_count_s = # {c in s : |max_z|_c > threshold}
  services ranked by alert_count_s desc
  returned metric ranks = flatten (metrics of top-ranked service first, …)
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


def _default_extract_service(col: str) -> str:
    """Fallback: split on first '__'. Matches PetShop / RCAEval-merged schema."""
    return col.split("__", 1)[0]


def alertcount_universal(
    data: pd.DataFrame,
    inject_time,
    *,
    extract_service: Optional[Callable[[str], str]] = None,
    threshold: float = 3.0,
    dataset=None,
    sli=None,
    anomalies=None,
    n_iter=None,
    verbose: bool = False,
    **kwargs: Any,
) -> dict:
    """Portable alert-count baseline.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a ``time`` column (monotone, same units as inject_time)
        plus numeric metric columns.
    inject_time : numeric
        Split point.
    extract_service : callable metric_col -> service_name, optional
        Benchmark-specific function that maps a metric column name to its
        owning service. If None, defaults to splitting on first '__'.
    threshold : float
        |max_z| threshold for a metric to count as an "alert" (default 3.0).

    Returns
    -------
    {"ranks": [metric_col, ...]}
        Metric columns ordered so that top-ranked service's metrics come
        first (thereby ensuring `extract_service(ranks[0])` = top service).
    """
    if extract_service is None:
        extract_service = _default_extract_service

    # --- Isolate numeric metric columns --- #
    if "time" not in data.columns:
        warnings.warn(
            "alertcount_universal: 'time' column missing; alphabetic fallback.",
            RuntimeWarning,
        )
        numeric_cols = [c for c in data.columns
                        if pd.api.types.is_numeric_dtype(data[c])]
        return {"ranks": sorted(numeric_cols)}

    candidate_cols: list[str] = []
    for c in data.columns:
        if c == "time":
            continue
        if pd.api.types.is_numeric_dtype(data[c]):
            candidate_cols.append(c)
    if not candidate_cols:
        warnings.warn(
            "alertcount_universal: no numeric metric columns found.",
            RuntimeWarning,
        )
        return {"ranks": []}

    # --- Split by inject_time --- #
    baseline = data[data["time"] < inject_time]
    anomaly = data[data["time"] >= inject_time]
    if baseline.empty or anomaly.empty:
        warnings.warn(
            f"alertcount_universal: degenerate split "
            f"(baseline={len(baseline)} rows, anomaly={len(anomaly)} rows); "
            f"alphabetic fallback.",
            RuntimeWarning,
        )
        return {"ranks": sorted(candidate_cols)}

    # --- Per-column |max_z| --- #
    max_abs_z: dict[str, float] = {}
    for c in candidate_cols:
        base_col = baseline[c]
        mu = base_col.mean()
        sigma = base_col.std(ddof=0)
        if sigma is None or np.isnan(sigma) or sigma == 0 or np.isnan(mu):
            max_abs_z[c] = 0.0
            continue
        anom_col = anomaly[c].to_numpy(dtype=float, copy=False)
        if not np.isfinite(anom_col).any():
            max_abs_z[c] = 0.0
            continue
        z = (anom_col - mu) / sigma
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mz = float(np.nanmax(np.abs(z)))
        max_abs_z[c] = mz if np.isfinite(mz) else 0.0

    # --- Aggregate per service --- #
    service_alerts: dict[str, int] = {}
    service_metrics: dict[str, list[tuple[str, float]]] = {}
    for c in candidate_cols:
        svc = extract_service(c)
        service_alerts.setdefault(svc, 0)
        service_metrics.setdefault(svc, [])
        service_metrics[svc].append((c, max_abs_z[c]))
        if max_abs_z[c] > threshold:
            service_alerts[svc] += 1

    # --- Rank services: count desc, alphabetic tiebreak --- #
    services_sorted = sorted(
        service_alerts.keys(),
        key=lambda s: (-service_alerts[s], s),
    )

    # --- Emit metric ranks: service-by-service, within service by |max_z| desc --- #
    ranks: list[str] = []
    for svc in services_sorted:
        metrics = service_metrics[svc]
        # Sort metrics within service: |max_z| desc, alphabetic tiebreak
        metrics_sorted = sorted(metrics, key=lambda t: t[0])
        metrics_sorted.sort(key=lambda t: t[1], reverse=True)
        ranks.extend(name for name, _ in metrics_sorted)

    if verbose:
        top_svc = services_sorted[:5]
        print(f"[alertcount] threshold={threshold}")
        print(f"[alertcount] top-5 services (by alert count): "
              f"{[(s, service_alerts[s]) for s in top_svc]}")

    return {"ranks": ranks}


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    inject = 50
    time_col = np.arange(n, dtype=float)
    df = pd.DataFrame({"time": time_col})

    # service_A has 3 metrics, all anomalous post-inject
    # service_B has 2 metrics, none anomalous
    # service_C has 2 metrics, 1 anomalous
    specs = [
        ("serviceA__cpu__mean", 8.0, True),    # anomalous
        ("serviceA__mem__mean", 7.0, True),    # anomalous
        ("serviceA__io__mean", 6.0, True),     # anomalous
        ("serviceB__cpu__mean", 0.0, False),
        ("serviceB__mem__mean", 0.0, False),
        ("serviceC__cpu__mean", 5.0, True),    # anomalous
        ("serviceC__mem__mean", 0.0, False),
    ]
    for name, shift, anomalous in specs:
        base_mean = 0.0
        base_std = 1.0
        col = rng.normal(base_mean, base_std, size=n)
        if anomalous:
            col[inject:] = rng.normal(base_mean + shift, base_std, size=n - inject)
        df[name] = col

    out = alertcount_universal(df, inject, verbose=True)
    ranks = out["ranks"]
    top_svc = ranks[0].split("__", 1)[0]
    assert top_svc == "serviceA", f"expected serviceA, got {top_svc!r}"
    print(f"top metric: {ranks[0]}  (service={top_svc})")

    # Verify service order: A (3 alerts), C (1 alert), B (0 alerts)
    seen = []
    for m in ranks:
        s = m.split("__", 1)[0]
        if s not in seen:
            seen.append(s)
    assert seen == ["serviceA", "serviceC", "serviceB"], (
        f"service order wrong: {seen}"
    )
    print(f"service order: {seen}")
    print("SELF-TEST PASSED")
