#!/usr/bin/env python3
"""CIRCA degeneracy audit.

For each Bank incident, look at what CIRCA's PC+RHT outputs BEFORE our
component-mapping step. Specifically:
  - Top-10 metric-level ranks (not component-mapped)
  - Unique services in top-10
  - Does the top-ranked METRIC map to Tomcat01 every time, or is the
    degeneracy introduced at the component-mapping step?

This distinguishes:
  (a) CIRCA itself is degenerate on OpenRCA-style data (inherent method-benchmark
      interaction) — "faithful reproduction, not a bug"
  (b) Our component-mapping (extract_component_from_metric) flattens diverse
      metric ranks into the same component (our bug)
"""
import argparse
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_rcaagent_1min import (
    CANDIDATES, EXTRACT, parse_instruction, count_failures,
    _fallback_json, _combined_classify,
)
from predict_baro import parquet_to_baro_df, extract_component_from_metric
from predict_circa import prefilter_top_k, run_circa


def audit_one(system, query_row, parquet_cache, dirs, top_k):
    instr = query_row["instruction"]
    mid_utc8, mid_utc, date_label = parse_instruction(instr)
    if mid_utc is None:
        return None

    paths = [
        Path(dirs["1min"]) / system / date_label / "metric_data.parquet",
        Path(dirs["ext"]) / system / date_label / "metric_data.parquet",
        Path(dirs["pool"]) / system / "metric_data.parquet",
    ]
    md = None
    for p in paths:
        if p.exists():
            if str(p) not in parquet_cache:
                parquet_cache[str(p)] = pd.read_parquet(p)
            m = parquet_cache[str(p)]
            if m is not None and not m.empty \
               and m.index.min() <= mid_utc <= m.index.max():
                md = m
                break
    if md is None:
        return None

    baro_df, inject_time = parquet_to_baro_df(md, mid_utc)
    if baro_df is None:
        return None

    filt_df, kept_cols = prefilter_top_k(baro_df, inject_time, top_k=top_k)
    if filt_df is None:
        return None

    ranks = run_circa(filt_df, inject_time)  # metric-level
    if not ranks:
        return None

    extract = EXTRACT[system]
    top10 = ranks[:10]
    top_components = [extract_component_from_metric(m, extract) for m in top10]

    return {
        "top10_metric": top10,
        "top10_component": top_components,
        "unique_components_top10": len(set(top_components) - {None, ""}),
        "all_returned": ranks,
        "top_metric": ranks[0],
        "top_component": top_components[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="Bank")
    ap.add_argument("--query_csv", default="run_scripts/openrca_eval/full335/bank_q.csv")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    dirs = {
        "1min": "$HOME/auditstack/results_1min",
        "ext": "$HOME/auditstack/results_ext",
        "pool": "$HOME/auditstack/results",
    }
    qdf = pd.read_csv(args.query_csv).head(args.limit)
    cache = {}

    top_component_counter = Counter()
    top_metric_counter = Counter()
    unique_component_dist = []

    print(f"{'task':<10} {'top_metric':<50} {'top_comp':<12} {'uniq@10':<8}")
    print("-" * 85)
    for _, row in qdf.iterrows():
        r = audit_one(args.system, row, cache, dirs, args.top_k)
        if r is None:
            print(f"{row.task_index:<10} <no data>")
            continue
        top_component_counter[r["top_component"]] += 1
        top_metric_counter[r["top_metric"]] += 1
        unique_component_dist.append(r["unique_components_top10"])
        print(f"{row.task_index:<10} {r['top_metric'][:48]:<50} "
              f"{str(r['top_component'])[:11]:<12} {r['unique_components_top10']:<8}")

    print()
    print("=== Top-component distribution (out of top-1 predictions) ===")
    for c, n in top_component_counter.most_common():
        print(f"  {c}: {n}")
    print()
    print("=== Top-metric distribution (out of top-1 metric picks) ===")
    for m, n in top_metric_counter.most_common(8):
        print(f"  {m}: {n}")
    print()
    print(f"=== Components diversity in top-10 metrics (per incident) ===")
    if unique_component_dist:
        arr = np.array(unique_component_dist)
        print(f"  mean={arr.mean():.2f}, median={int(np.median(arr))}, "
              f"min={arr.min()}, max={arr.max()}, n={len(arr)}")
        print(f"  distribution: {Counter(unique_component_dist.__iter__()).most_common()}")


if __name__ == "__main__":
    main()
