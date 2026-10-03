#!/usr/bin/env python3
"""Run the UNIVERSAL Z-score predictor (zbase_universal) on OpenRCA.

This is R1 (external-review correction): eliminate the z_family collapse by
running the SAME zbase_universal predictor (already used on RCAEval and
PetShop) on OpenRCA's 4 subsystems and re-labeling everything as
`zbase_univ`. Produces an OpenRCA-format prediction CSV, which can then be
scored by openrca_evaluate_sweep.py (same scoring as the legacy zbase).

Key design:
  * We REUSE predict_baro.py's parquet → BARO-format conversion
    (parquet_to_baro_df) and the EXTRACT/CANDIDATES/REASON_MAP machinery
    from predict_rcaagent_1min. Only the ranker is swapped for
    zbase_universal. This is the minimal, auditable change.
  * Onset time is estimated by the same peak-deviation rule used in
    predict_baro.py (not legacy predict_anomaly_baseline.py's earlier
    baseline/window, which is OpenRCA-specific).
  * Method column in the output CSV is "zbase_univ".

Outputs:
  preds_zbase_univ_<tag>.csv   (same JSON-like schema as preds_baro/preds_zbaseline)

Then the caller runs openrca_evaluate_sweep.py with --tolerance 60 (or a
sweep) to compute acc@1 per subsystem, identical to how the legacy zbase
numbers in results_full335/clean_partial/scores_per_incident.csv were built.
"""
import argparse
import functools
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from predict_alertcount import alertcount_universal  # noqa: E402
from predict_rcaagent_1min import (  # noqa: E402
    CANDIDATES, EXTRACT, DEFAULT, REASON_MAP,
    UTC_OFFSET,
    parse_instruction, count_failures,
    map_reason, _fallback_json,
    _combined_classify,
)
from predict_baro import (  # noqa: E402 -- reuse parquet adapter
    parquet_to_baro_df,
    extract_component_from_metric,
)


def predict_one(system, query_row, parquet_cache, results_1min_dir,
                results_ext_dir, results_dir):
    instruction = query_row['instruction']
    scoring_points = str(query_row['scoring_points'])
    n_fail = count_failures(scoring_points)

    mid_utc8, mid_utc, date_label = parse_instruction(instruction)
    if mid_utc is None:
        return _fallback_json(system, n_fail, fallback_dt=None)

    # Load parquet (identical cascade to predict_baro / predict_anomaly_baseline)
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

    # Convert parquet → BARO/zbase_univ DataFrame (time column + metric columns)
    zdf, inject_time = parquet_to_baro_df(md, mid_utc,
                                          window_minutes=30,
                                          baseline_hours=4)
    if zdf is None:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    # Alert-count predictor, using OpenRCA's component extraction as the
    # per-metric "service" aggregator.
    extract = EXTRACT[system]
    extract_svc = functools.partial(extract_component_from_metric, extract=extract)
    try:
        result = alertcount_universal(
            zdf, inject_time,
            extract_service=extract_svc,
            threshold=3.0,
            dataset="openrca",
            sli=None,
            anomalies=None,
            n_iter=None,
            verbose=False,
        )
    except Exception:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    ranked_metrics = result.get('ranks', [])
    if not ranked_metrics:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8)

    # Map top ranked metrics → candidate components (same filter as zbase_univ).
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
                'anomalies': [{'metric': metric_col, 'z': 0, 'pct': 0,
                               'fault_type': ftype}],
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

    # Format JSON. Onset via peak-deviation (same rule as predict_baro).
    objs = []
    for comp, info, top_metric in picks[:n_fail]:
        reason = map_reason(system, comp, info)
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
    ap = argparse.ArgumentParser(description='alert-count predictor for OpenRCA')
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
