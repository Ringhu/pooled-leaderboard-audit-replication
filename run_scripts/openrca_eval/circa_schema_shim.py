#!/usr/bin/env python3
"""Schema-repair shim for CIRCA (rht.py) on OpenRCA parquets.

Why: RCAEval's rht.py (graph_heads/rht.py:305-351) constructs its node graph
by splitting each column name on the FIRST single underscore:
    service = col.split("_")[0]
    metric  = col.split("_")[1]
and later reconstructs column access with f"{service}_{metric}".

OpenRCA parquet columns use a DOUBLE underscore between service and metric,
and BOTH halves can themselves contain single underscores / hyphens / dots.
Examples:
    IG01__JVM-Memory_7778_JVM_Memory_HeapMemoryUsage          (Bank)
    adservice-0.destination.frontend.adservice__istio_...     (Market)
    db_001__CPU_Used_Pct                                      (Telecom; service has '_')

Under rht.py's split:
    "IG01__JVM-Memory_..."        -> service="IG01",  metric=""     -> degenerate
    "db_001__CPU_Used_Pct"        -> service="db",    metric="001"  -> wrong + collision

This shim rewrites column names BEFORE they enter rht. It collapses the
metric part into a single hyphenated token, and guarantees the service part
contains no underscore. After the rewrite:
    col.split("_", 1)[0]  == service   (stable)
    col.split("_", 1)[1]  == compact_metric  (single token, contains only
                                              letters, digits, '-' or '.')
    f"{service}_{metric}" == col       (round-trips)

The shim is a pure DataFrame preprocessor. It does NOT modify any RCAEval
package file.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def _compact(name: str) -> str:
    """Replace underscores with hyphens so the token has no '_'.

    Hyphens and dots are preserved: rht.py never splits on them, so they are
    safe to keep and make it easier to eyeball the repaired names.
    """
    return name.replace("_", "-")


def repair_openrca_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Rename columns so rht.py's split("_")[0/1] gives sensible tokens.

    Rule:
      * "time" is passed through unchanged.
      * If column contains "__", split on the FIRST "__":
            service_part, metric_part = col.split("__", 1)
        Both halves are compacted (underscore -> hyphen). The new column
        name is f"{_compact(service_part)}_{_compact(metric_part)}" and
        contains exactly ONE underscore.
      * Otherwise the column is assumed to already be in single-underscore
        form and is left unchanged.

    Guarantees:
      * All returned column names (except "time") contain at most one "_".
      * The mapping is 1-to-1; in the unlikely event of a collision (two
        distinct original names compact to the same new name), a numeric
        suffix is appended to de-duplicate.

    Returns:
      (renamed_df, mapping)  where mapping[new_name] == original_name.
    """
    mapping: Dict[str, str] = {}
    new_cols: list[str] = []
    seen_counts: Dict[str, int] = {}

    for col in df.columns:
        if col == "time":
            new_cols.append("time")
            mapping["time"] = "time"
            continue

        if "__" in col:
            service_raw, metric_raw = col.split("__", 1)
            service = _compact(service_raw)
            metric = _compact(metric_raw)
            new_name = f"{service}_{metric}"
        else:
            # Single-underscore schema (e.g., RCAEval native). Leave alone;
            # caller is responsible for not double-applying this shim.
            new_name = col

        # Collision guard: if this compacted name already exists, suffix it.
        if new_name in mapping and mapping[new_name] != col:
            seen_counts[new_name] = seen_counts.get(new_name, 0) + 1
            new_name = f"{new_name}-dup{seen_counts[new_name]}"

        new_cols.append(new_name)
        mapping[new_name] = col

    renamed = df.copy()
    renamed.columns = new_cols
    return renamed, mapping


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    # Case 1: Bank-style double-underscore
    df1 = pd.DataFrame({
        "time": np.arange(3),
        "IG01__JVM-Memory_7778_JVM_Memory_HeapMemoryUsage": [1.0, 2.0, 3.0],
        "Tomcat01__OSLinux-CPU_CPU-0_SingleCpuUtil": [0.1, 0.2, 0.3],
    })
    out1, map1 = repair_openrca_columns(df1)
    expected1 = {
        "time",
        "IG01_JVM-Memory-7778-JVM-Memory-HeapMemoryUsage",
        "Tomcat01_OSLinux-CPU-CPU-0-SingleCpuUtil",
    }
    assert set(out1.columns) == expected1, (
        f"Case1 cols mismatch: got {set(out1.columns)}, want {expected1}"
    )
    # Round-trip: split("_",1) back to service, and f"{s}_{m}" == new col
    for new_name in out1.columns:
        if new_name == "time":
            continue
        s, m = new_name.split("_", 1)
        assert "_" not in s, f"service has underscore: {s}"
        assert f"{s}_{m}" == new_name
    # rht.py uses split("_")[0] and split("_")[1], verify both are non-empty
    for new_name in out1.columns:
        if new_name == "time":
            continue
        parts = new_name.split("_")
        assert parts[0] and parts[1], f"rht split fails on {new_name}"

    # Case 2: Telecom-style service that ITSELF contains an underscore
    # ("db_001") and a short metric.
    df2 = pd.DataFrame({
        "time": np.arange(2),
        "db_001__CPU_Used_Pct": [10.0, 11.0],
        "db_001__ACS": [1.0, 2.0],
    })
    out2, map2 = repair_openrca_columns(df2)
    # After shim, service "db_001" -> "db-001"; metric "CPU_Used_Pct" ->
    # "CPU-Used-Pct".
    assert "db-001_CPU-Used-Pct" in out2.columns
    assert "db-001_ACS" in out2.columns
    for new_name in out2.columns:
        if new_name == "time":
            continue
        parts = new_name.split("_")
        assert parts[0] == "db-001"
        assert parts[1], f"metric empty for {new_name}"

    # Case 3: Market-style service with dots+hyphens and complex metric
    df3 = pd.DataFrame({
        "time": np.arange(2),
        "adservice-0.destination.frontend.adservice__istio_request_bytes.grpc.200.0.0": [1, 2],
        "checkoutservice-0__istio_requests.grpc.200.0.0": [3, 4],
    })
    out3, map3 = repair_openrca_columns(df3)
    for new_name in out3.columns:
        if new_name == "time":
            continue
        # Service part must have NO underscore (so split("_")[0] is clean)
        s = new_name.split("_", 1)[0]
        assert "_" not in s, f"service still has '_': {s}"
        # Mapping round-trip
        assert map3[new_name] in df3.columns

    # Case 4: mapping must round-trip
    for m in (map1, map2, map3):
        for new_name, orig_name in m.items():
            assert orig_name in list(df1.columns) + list(df2.columns) + list(df3.columns)

    # Case 5: no double-underscore -> unchanged (RCAEval native format)
    df5 = pd.DataFrame({
        "time": np.arange(2),
        "checkoutservice_mem": [1.0, 2.0],
        "frontend_cpu": [0.1, 0.2],
    })
    out5, map5 = repair_openrca_columns(df5)
    assert list(out5.columns) == list(df5.columns)
    assert map5["checkoutservice_mem"] == "checkoutservice_mem"

    print("SELF-TEST PASSED")
