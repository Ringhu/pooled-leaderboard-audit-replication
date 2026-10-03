#!/usr/bin/env python3
"""CIRCA (PC + RHT) baseline for OpenRCA.

Pre-filters metrics to the top-K most anomalous per incident (RobustScaler
z-score magnitude), runs PC for causal discovery + RHT for root-cause ranking,
then maps the top-ranked metric back to a component and emits an OpenRCA-
format prediction per query.
"""
import argparse
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_rcaagent_1min import (
    CANDIDATES, EXTRACT,
    UTC_OFFSET,
    parse_instruction, count_failures,
    map_reason, _fallback_json,
    _combined_classify,
)
from predict_baro import parquet_to_baro_df, extract_component_from_metric

from RCAEval.graph_construction.pc import pc_default
from RCAEval.graph_heads.rht import rht


def prefilter_top_k(baro_df, inject_time, top_k, corr_thresh=0.98):
    """Keep only the top-K metrics by |RobustScaler z-score| over the post-
    incident window, then drop near-duplicate columns so PC's fisher-z test
    does not hit a singular correlation matrix.
    """
    normal = baro_df[baro_df["time"] < inject_time].drop(columns=["time"])
    anomal = baro_df[baro_df["time"] >= inject_time].drop(columns=["time"])
    cols = list(normal.columns)
    scores = []
    for c in cols:
        a = normal[c].to_numpy()
        b = anomal[c].to_numpy()
        if len(a) < 5 or len(b) < 2:
            continue
        try:
            scaler = RobustScaler().fit(a.reshape(-1, 1))
            z = scaler.transform(b.reshape(-1, 1))[:, 0]
        except Exception:
            continue
        scores.append((c, float(np.max(np.abs(z)))))
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_cols = [c for c, _ in scores[:top_k * 3]]  # wider pool before dedup
    if len(ranked_cols) < 3:
        return None, []

    # Greedy drop near-duplicates (|corr| > corr_thresh) keeping higher-z one
    full = baro_df[ranked_cols].copy()
    kept = []
    for c in ranked_cols:
        if not kept:
            kept.append(c)
            continue
        # correlation with already-kept columns
        corr = full[kept + [c]].corr().iloc[-1, :-1].abs()
        if corr.max() < corr_thresh:
            kept.append(c)
        if len(kept) >= top_k:
            break

    if len(kept) < 3:
        return None, []
    out = baro_df[["time"] + kept].copy()
    return out, kept


def run_circa(baro_df, inject_time):
    """Run CIRCA (pc_default + rht) and return the ranked metric list."""
    time_col = baro_df["time"]
    data = baro_df.copy()
    pc_input = data.drop(columns=["time"])

    # Drop zero-variance cols that may survive pre-filter after slicing
    keep = pc_input.std() > 1e-8
    pc_input = pc_input.loc[:, keep]
    if pc_input.shape[1] < 3:
        return []

    data = pc_input.copy()
    data["time"] = time_col

    try:
        adj = pc_default(pc_input, show_progress=False)
    except Exception as e:
        print(f"  PC failed: {e}", file=sys.stderr)
        return []
    try:
        ranks = rht(adj, inject_time, data)
    except Exception as e:
        print(f"  RHT failed: {e}", file=sys.stderr)
        return []
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranks]


def predict_one(system, query_row, parquet_cache, results_1min_dir,
                results_ext_dir, results_dir, top_k):
    instruction = query_row["instruction"]
    scoring_points = str(query_row["scoring_points"])
    n_fail = count_failures(scoring_points)

    mid_utc8, mid_utc, date_label = parse_instruction(instruction)
    if mid_utc is None:
        return _fallback_json(system, n_fail, fallback_dt=None)

    p_1min = Path(results_1min_dir) / system / date_label / "metric_data.parquet"
    p_ext = Path(results_ext_dir) / system / date_label / "metric_data.parquet"
    p_pool = Path(results_dir) / system / "metric_data.parquet"

    md = None
    for p in [p_1min, p_ext, p_pool]:
        if p.exists():
            key = str(p)
            if key not in parquet_cache:
                parquet_cache[key] = pd.read_parquet(p)
            m = parquet_cache[key]
            if (m is not None and not m.empty
                and m.index.min() <= mid_utc <= m.index.max()):
                md = m
                break

    if md is None:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    baro_df, inject_time = parquet_to_baro_df(
        md, mid_utc, window_minutes=30, baseline_hours=4
    )
    if baro_df is None:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    # Pre-filter to top-K most anomalous metrics (makes PC tractable)
    filt_df, _ = prefilter_top_k(baro_df, inject_time, top_k=top_k)
    if filt_df is None:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    ranked_metrics = run_circa(filt_df, inject_time)
    if not ranked_metrics:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    extract = EXTRACT[system]
    candidates = set(CANDIDATES[system])

    picks = []
    seen_comps = set()
    for metric_col in ranked_metrics:
        comp = extract_component_from_metric(metric_col, extract)
        if not comp or comp in seen_comps:
            continue
        if comp in candidates:
            ftype = _combined_classify(metric_col)
            fault_types = Counter({ftype: 1})
            info = {
                "anomalies": [{"metric": metric_col, "z": 0, "pct": 0,
                                 "fault_type": ftype}],
                "max_z": 0, "max_pct": 0, "fault_types": fault_types,
            }
            picks.append((comp, info, metric_col))
            seen_comps.add(comp)
            if len(picks) >= n_fail:
                break

    if not picks:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)
    while len(picks) < n_fail:
        picks.append(picks[-1])

    objs = []
    for comp, info, top_metric in picks[:n_fail]:
        reason = map_reason(system, comp, info)
        onset_dt_utc8 = mid_utc8
        if top_metric in md.columns:
            inc_idx = md.index.get_indexer([mid_utc], method="nearest")[0]
            ws = max(0, inc_idx - 30)
            we = min(len(md), inc_idx + 30 + 1)
            base_start = max(0, ws - 240)
            base = md[top_metric].iloc[base_start:ws].dropna()
            if len(base) >= 10:
                b_mean = base.mean()
                window = md[top_metric].iloc[ws:we].dropna()
                if len(window) > 0:
                    deviations = (window - b_mean).abs()
                    peak_utc = deviations.idxmax()
                    onset_dt_utc8 = pd.Timestamp(peak_utc) + UTC_OFFSET
        dt_str = onset_dt_utc8.strftime("%Y-%m-%d %H:%M:%S")
        objs.append(
            f'{{"root cause occurrence datetime": "{dt_str}", '
            f'"root cause component": "{comp}", '
            f'"root cause reason": "{reason}"}}'
        )
    return "".join(objs)


def main():
    ap = argparse.ArgumentParser(description="CIRCA baseline for OpenRCA")
    ap.add_argument("--system", required=True, choices=list(CANDIDATES.keys()))
    ap.add_argument("--query_csv", required=True)
    ap.add_argument("--results_1min_dir",
                    default="$HOME/auditstack/results_1min")
    ap.add_argument("--results_ext_dir",
                    default="$HOME/auditstack/results_ext")
    ap.add_argument("--results_dir",
                    default="$HOME/auditstack/results")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    qdf = pd.read_csv(args.query_csv)
    cache = {}
    preds = []
    for i, row in qdf.iterrows():
        pred = predict_one(
            args.system, row, cache,
            args.results_1min_dir, args.results_ext_dir, args.results_dir,
            top_k=args.top_k,
        )
        preds.append(pred)
        if i < 3 or (i + 1) % 10 == 0:
            print(f"[{args.system} #{i+1}/{len(qdf)} task={row['task_index']}] "
                  f"pred: {pred[:160]}...")

    out = pd.DataFrame({"task_index": qdf["task_index"], "prediction": preds})
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")


if __name__ == "__main__":
    main()
