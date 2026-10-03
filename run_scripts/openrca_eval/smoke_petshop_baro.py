#!/usr/bin/env python3
"""Smoke test: can RCAEval's BARO process a flattened PetShop case?

Loads one case from PetShop's high_traffic via its own load_scenario(),
flattens the MultiIndex columns, concatenates noissue baseline + issue
abnormal data, calls BARO, prints top-10 ranks. Compares against true
root_cause from target.json.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Make PetShop's rca_task available
PETSHOP = Path("$HOME/petshop_repo")
sys.path.insert(0, str(PETSHOP / "code"))

from rca_task import load_scenario  # noqa: E402

# RCAEval BARO — same entry as run_rcaeval.py
from RCAEval.e2e.baro import baro  # noqa: E402


def flatten_cols(df):
    """MultiIndex (node, metric, statistic) -> 'node__metric__statistic'.

    Also sanitize: replace spaces, double-colons, slashes so downstream
    string parsers don't split on them.
    """
    def clean(s):
        return (str(s)
                .replace(" ", "_")
                .replace("::", "-")
                .replace("/", "-"))

    flat = ["__".join(clean(x) for x in col) for col in df.columns]
    out = df.copy()
    out.columns = flat
    return out


def main():
    scenario = PETSHOP / "dataset" / "high_traffic"
    print(f"Loading scenario: {scenario}")
    graph, normal_metrics, issues = load_scenario(str(scenario))
    print(f"  normal_metrics: {normal_metrics.shape}  (rows=timesteps, cols=flat metric)")
    print(f"  test issues:    {len(issues['test'])}")
    print(f"  train issues:   {len(issues['train'])}")
    print(f"  graph nodes:    {graph.number_of_nodes()}")

    # Pick the first test issue
    abnormal_metrics, target = issues["test"][0]
    print(f"\n=== issue_0 target ===")
    print(f"  target node:    {target['target']['node']}")
    print(f"  target metric:  {target['target']['metric']}")
    print(f"  true root:      {target['root_cause']['node']}")
    print(f"  abnormal_metrics shape: {abnormal_metrics.shape}")

    # Flatten MultiIndex
    normal_flat = flatten_cols(normal_metrics)
    abnormal_flat = flatten_cols(abnormal_metrics)
    print(f"\n  normal_flat cols: {normal_flat.shape[1]}, abnormal_flat cols: {abnormal_flat.shape[1]}")

    # Align columns (intersection)
    common = normal_flat.columns.intersection(abnormal_flat.columns)
    print(f"  common cols:     {len(common)}")
    normal_flat = normal_flat[common]
    abnormal_flat = abnormal_flat[common]

    # Build a single time-indexed DataFrame. BARO expects a `time` column
    # (unix seconds). Normal_metrics has timestamp as index; abnormal too.
    # Concatenate and synthesize time column.
    combined = pd.concat([normal_flat, abnormal_flat], axis=0, ignore_index=False)
    combined = combined.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    combined = combined.reset_index()
    # First column after reset is the original timestamp index (named or `index`)
    ts_col = combined.columns[0]
    combined = combined.rename(columns={ts_col: "time"})
    # Unique monotone time (baseline may overlap abnormal if timestamps reused)
    combined["time"] = combined["time"].astype(float)
    combined = combined.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    print(f"  combined rows:   {len(combined)}")
    print(f"  time range:      {combined['time'].min()} .. {combined['time'].max()}")

    # inject_time = start of abnormal period
    inject_time = float(target["target"]["timestamp"])
    print(f"  inject_time:     {inject_time}")

    # SLI — PetShop's target is (node, metric, statistic=Average usually).
    # After flattening, the SLI column is "<clean_node>__<metric>__<statistic>"
    def clean(s):
        return str(s).replace(" ", "_").replace("::", "-").replace("/", "-")

    sli = "__".join([
        clean(target["target"]["node"]),
        target["target"]["metric"],
        target["target"].get("agg", "Average"),
    ])
    if sli not in combined.columns:
        # Fallback: any column starting with target node + metric
        cands = [c for c in combined.columns
                 if c.startswith(clean(target["target"]["node"]) + "__" + target["target"]["metric"])]
        sli = cands[0] if cands else combined.columns[-1]
    print(f"  sli column:      {sli}")

    # Call BARO
    print("\n=== Running BARO ===")
    try:
        out = baro(
            combined, inject_time,
            dataset="petshop",
            anomalies=None, dk_select_useful=False,
            sli=sli, verbose=False,
            n_iter=len(combined.columns) - 1,
        )
    except Exception as e:
        print(f"  BARO failed: {e}")
        return

    ranks = out.get("ranks", [])
    print(f"  n_ranks: {len(ranks)}")

    # Top-10 with service extraction
    def extract_service(col):
        # PetShop flat format: <node>__<metric>__<statistic>
        return col.split("__")[0]

    print("\n=== Top-10 predictions ===")
    true_root_clean = clean(target["root_cause"]["node"])
    seen = set()
    k = 0
    for m in ranks:
        svc = extract_service(m)
        if svc in seen:
            continue
        seen.add(svc)
        k += 1
        marker = "   *TRUE*" if svc == true_root_clean else ""
        print(f"  {k:2d}. {svc:60s}{marker}")
        if k >= 10:
            break

    hit_at_1 = extract_service(ranks[0]) == true_root_clean if ranks else False
    hit_at_10 = true_root_clean in seen
    print(f"\n  hit@1:  {hit_at_1}")
    print(f"  hit@10: {hit_at_10}")


if __name__ == "__main__":
    main()
