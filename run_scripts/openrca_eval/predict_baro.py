#!/usr/bin/env python3
"""
BARO baseline adapted for OpenRCA evaluation.

Converts OpenRCA parquets to BARO's expected format, runs BARO,
maps ranked metrics back to components, and outputs predictions
in OpenRCA's JSON-like format.

BARO: rank each metric independently by RobustScaler z-score,
pick the service with the top-ranked metric.
"""
import argparse
import os
import re
import sys
import warnings
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Inline BARO logic (avoids RCAEval dependency)
from sklearn.preprocessing import RobustScaler

def baro_inline(data, inject_time):
    """BARO: RobustScaler z-score ranking per metric.
    data: DataFrame with 'time' column + metric columns.
    inject_time: unix timestamp separating normal/anomalous.
    """
    normal_df = data[data["time"] < inject_time].drop(columns=["time"])
    anomal_df = data[data["time"] >= inject_time].drop(columns=["time"])

    # Drop constant columns
    std = normal_df.std()
    keep = std[std > 1e-8].index.tolist()
    normal_df = normal_df[keep]
    anomal_df = anomal_df[[c for c in keep if c in anomal_df.columns]]

    intersects = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []
    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()
        if len(a) < 5 or len(b) < 2:
            continue
        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = float(np.max(zscores))
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    return {"ranks": [x[0] for x in ranks]}

# Import shared logic from CD-detector predictor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_rcaagent_1min import (
    CANDIDATES, EXTRACT, DEFAULT, REASON_MAP,
    UTC_OFFSET, MONTHS,
    parse_instruction, count_failures,
    map_reason, _fallback_json,
    _combined_classify,
)


def parquet_to_baro_df(md, incident_utc, window_minutes=30, baseline_hours=4):
    """Convert a CD-detector parquet (DatetimeIndex, component__metric cols)
    to BARO format (time column in unix seconds + metric columns).

    Returns (DataFrame, inject_time_unix) or (None, None) if insufficient data.
    """
    inc_idx = md.index.get_indexer([incident_utc], method='nearest')[0]

    # Window: baseline_hours before + window_minutes around incident
    baseline_steps = baseline_hours * 60  # 1-min resolution
    window_steps = window_minutes

    start_idx = max(0, inc_idx - baseline_steps - window_steps)
    end_idx = min(len(md), inc_idx + window_steps + 1)

    if end_idx - start_idx < 30:
        return None, None

    sub = md.iloc[start_idx:end_idx].copy()

    # Drop constant/near-constant columns
    sub = sub.loc[:, sub.std() > 1e-8]
    # Drop columns with >50% NaN
    sub = sub.loc[:, sub.isna().mean() < 0.5]
    # Forward fill, then fill remaining NaN with 0
    sub = sub.ffill().fillna(0)

    if sub.shape[1] < 5:
        return None, None

    # Add time column in unix seconds — unit-safe across datetime64[ns|us|ms|s]
    sub.insert(0, 'time', sub.index.astype('datetime64[s]').astype(np.int64))

    # inject_time = incident midpoint
    inject_time = int(md.index[inc_idx].timestamp())

    return sub, inject_time


def extract_component_from_metric(metric_col, extract_func):
    """Extract service/component name from parquet column name."""
    cmdb_id = metric_col.split('__', 1)[0] if '__' in metric_col else metric_col
    return extract_func(cmdb_id)


def predict_one(system, query_row, parquet_cache, results_1min_dir,
                results_ext_dir, results_dir):
    instruction = query_row['instruction']
    scoring_points = str(query_row['scoring_points'])
    n_fail = count_failures(scoring_points)

    mid_utc8, mid_utc, date_label = parse_instruction(instruction)
    if mid_utc is None:
        return _fallback_json(system, n_fail, fallback_dt=None)

    # Load parquet (same cascade as CD-detector)
    p_1min = Path(results_1min_dir) / system / date_label / 'metric_data.parquet'
    p_ext = Path(results_ext_dir) / system / date_label / 'metric_data.parquet'
    p_pool = Path(results_dir) / system / 'metric_data.parquet'

    md = None
    for p in [p_1min, p_ext, p_pool]:
        if p.exists():
            key = str(p)
            if key not in parquet_cache:
                parquet_cache[key] = pd.read_parquet(p)
            m = parquet_cache[key]
            if m is not None and not m.empty and m.index.min() <= mid_utc <= m.index.max():
                md = m
                break

    if md is None:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    # Convert parquet to BARO format
    baro_df, inject_time = parquet_to_baro_df(md, mid_utc,
                                               window_minutes=30,
                                               baseline_hours=4)
    if baro_df is None:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    # Run BARO
    try:
        result = baro_inline(baro_df, inject_time=inject_time)
    except Exception:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    ranked_metrics = result.get('ranks', [])
    if not ranked_metrics:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    # Map metrics to components
    extract = EXTRACT[system]
    candidates = set(CANDIDATES[system])

    # Find top component from BARO's ranked metrics
    picks = []
    seen_comps = set()
    for metric_col in ranked_metrics:
        comp = extract_component_from_metric(metric_col, extract)
        if not comp or comp in seen_comps:
            continue
        if comp in candidates:
            # Get fault type for reason mapping
            ftype = _combined_classify(metric_col)
            fault_types = Counter({ftype: 1})
            info = {
                'anomalies': [{'metric': metric_col, 'z': 0, 'pct': 0, 'fault_type': ftype}],
                'max_z': 0,
                'max_pct': 0,
                'fault_types': fault_types,
            }
            picks.append((comp, info, metric_col))
            seen_comps.add(comp)
            if len(picks) >= n_fail:
                break

    if not picks:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    while len(picks) < n_fail:
        picks.append(picks[-1])

    # Build prediction JSON
    objs = []
    for comp, info, top_metric in picks[:n_fail]:
        reason = map_reason(system, comp, info)

        # Onset: find peak deviation time of the top metric
        onset_dt_utc8 = mid_utc8
        if top_metric in md.columns:
            inc_idx = md.index.get_indexer([mid_utc], method='nearest')[0]
            ws = max(0, inc_idx - 30)
            we = min(len(md), inc_idx + 30 + 1)
            base_start = max(0, ws - 240)
            base = md[top_metric].iloc[base_start:ws].dropna()
            if len(base) >= 10:
                b_mean = base.mean()
                window = md[top_metric].iloc[ws:we].dropna()
                if len(window) > 0:
                    deviations = abs(window - b_mean)
                    peak_utc = deviations.idxmax()
                    onset_dt_utc8 = pd.Timestamp(peak_utc) + UTC_OFFSET

        dt_str = onset_dt_utc8.strftime('%Y-%m-%d %H:%M:%S')
        objs.append(
            f'{{"root cause occurrence datetime": "{dt_str}", '
            f'"root cause component": "{comp}", '
            f'"root cause reason": "{reason}"}}'
        )

    return ''.join(objs)


def main():
    ap = argparse.ArgumentParser(description='BARO baseline for OpenRCA')
    ap.add_argument('--system', required=True, choices=list(CANDIDATES.keys()))
    ap.add_argument('--query_csv', required=True)
    ap.add_argument('--results_1min_dir',
                    default='$HOME/auditstack/results_1min')
    ap.add_argument('--results_ext_dir',
                    default='$HOME/auditstack/results_ext')
    ap.add_argument('--results_dir',
                    default='$HOME/auditstack/results')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    qdf = pd.read_csv(args.query_csv)
    cache = {}
    preds = []
    for i, row in qdf.iterrows():
        pred = predict_one(args.system, row, cache,
                          args.results_1min_dir,
                          args.results_ext_dir,
                          args.results_dir)
        preds.append(pred)
        if i < 3:
            print(f"[{args.system} #{i} task={row['task_index']}] pred: {pred[:200]}...")

    out = pd.DataFrame({'task_index': qdf['task_index'], 'prediction': preds})
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")


if __name__ == '__main__':
    main()
