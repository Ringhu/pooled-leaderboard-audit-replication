#!/usr/bin/env python3
"""
Anomaly-ranking baseline for OpenRCA.

Differences from CD-detector (predict_rcaagent_1min.py):
  - Rank services by max |z-score| instead of max |pct deviation|
  - Use peak deviation timestamp as onset (NOT first-crossing adaptive detection)

Same data, same candidates, same reason mapping → only ranking & timing differ.
This produces genuinely different predictions to test tolerance sensitivity.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import shared logic from CD-detector predictor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_rcaagent_1min import (
    CANDIDATES, EXTRACT, DEFAULT, REASON_MAP,
    UTC_OFFSET,
    parse_instruction, count_failures,
    detect_anomalies_system_1min,
    map_reason, _fallback_json,
)


def pick_by_zscore(svc_info, candidates, n=1):
    """Pick top-n components by max |z-score| (instead of max |pct|)."""
    if not svc_info:
        return []
    # Sort by z-score descending
    ranking = sorted(svc_info.items(), key=lambda x: -x[1]['max_z'])
    cand_set = set(candidates)
    picks = []
    seen = set()
    for comp, info in ranking:
        if comp in cand_set and comp not in seen:
            picks.append((comp, info))
            seen.add(comp)
            if len(picks) >= n:
                return picks
    return picks


def find_peak_time(metric_data, metric_id, incident_dt, window_minutes=30):
    """Find timestamp of maximum |deviation| from baseline (NOT first-crossing).

    This is simpler and produces different timestamps than CD-detector's
    adaptive onset detection.
    """
    inc_idx = metric_data.index.get_indexer([incident_dt], method='nearest')[0]
    ws = max(0, inc_idx - window_minutes)
    we = min(len(metric_data), inc_idx + window_minutes + 1)
    bs = max(0, ws - 240)  # 4 hours baseline
    be = ws

    if metric_id not in metric_data.columns:
        return None

    base = metric_data[metric_id].iloc[bs:be].dropna()
    if len(base) < 10:
        return None
    b_mean = base.mean()

    window = metric_data[metric_id].iloc[ws:we].dropna()
    if len(window) == 0:
        return None

    deviations = abs(window - b_mean)
    peak_idx = deviations.idxmax()
    return peak_idx


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

    extract = EXTRACT[system]
    svc_info = detect_anomalies_system_1min(md, mid_utc, extract,
                                            window_minutes=30, baseline_hours=4)

    candidates = CANDIDATES[system]
    # KEY DIFFERENCE 1: rank by z-score, not pct
    picks = pick_by_zscore(svc_info, candidates, n=n_fail)

    if not picks:
        picks = [(DEFAULT[system][0], None)] * n_fail
    while len(picks) < n_fail:
        picks.append(picks[-1])

    objs = []
    for comp, info in picks[:n_fail]:
        reason = map_reason(system, comp, info)
        # KEY DIFFERENCE 2: use peak deviation time, not first-crossing onset
        onset_dt_utc8 = mid_utc8  # fallback
        if info and info.get('anomalies'):
            top_metric = max(info['anomalies'], key=lambda a: a['z'])['metric']
            peak_utc = find_peak_time(md, top_metric, mid_utc, window_minutes=30)
            if peak_utc is not None:
                onset_dt_utc8 = pd.Timestamp(peak_utc) + UTC_OFFSET

        dt_str = onset_dt_utc8.strftime('%Y-%m-%d %H:%M:%S')
        objs.append(
            f'{{"root cause occurrence datetime": "{dt_str}", '
            f'"root cause component": "{comp}", '
            f'"root cause reason": "{reason}"}}'
        )
    return ''.join(objs)


def main():
    ap = argparse.ArgumentParser(description='Anomaly-ranking baseline for OpenRCA')
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

    out = pd.DataFrame({'task_index': qdf['task_index'], 'prediction': preds})
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")


if __name__ == '__main__':
    main()
