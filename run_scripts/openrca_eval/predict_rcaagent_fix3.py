"""FIX3: sibling disambiguation via onset.

Diff from predict_rcaagent_1min.py: after pick_components selects a top-1,
look at all candidates in the SAME service family (e.g. Tomcat01/02/04,
os_001/002, frontend-0/1/2). Among those, pick the one with earliest onset
on its top anomalous metric. Only applied within-family, never across.
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Import v2 detector (we will use only the helpers)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.insert(0, os.path.join(SCRIPTS_DIR, 'scripts'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v2_rca import classify_metric_type as _orig_classify, is_counter  # noqa: E402
from v4_rca import (  # noqa: E402
    extract_component_bank,
    extract_component_telecom,
    extract_component_market,
)

# ------------------------------------------------------------------
# Extended classifier (Market + Telecom KPI patterns)
# ------------------------------------------------------------------
import v2_rca as _v2  # noqa: E402

MARKET_PATTERNS = [
    ('cpu', ['system.cpu.', 'system.load', 'container_cpu', '.cpu.']),
    ('memory', ['container_memory_failures', 'container_memory_usage',
                'container_memory_max', 'container_memory_rss',
                'container_memory_cache', 'container_memory_swap',
                'container_memory_mapped', 'container_memory_working',
                'system.mem.', '.mem.', 'system.swap']),
    ('disk_io', ['system.io.', 'container_fs_read', 'container_fs_write',
                 'system.disk.read', 'system.disk.write', 'container_blkio']),
    ('disk_space', ['system.disk.free', 'system.disk.used', 'system.disk.pct',
                    'system.fs.inodes', 'container_fs_usage']),
    ('network', ['system.net.', 'container_network', '.tcp.', '.udp.']),
]

TELECOM_PATTERNS = [
    ('cpu', ['cpu_used', 'cpu_free', 'proc_used', 'proc_user', 'sys_cpu', 'system_cpu']),
    ('memory', ['mem_', 'memory', 'swap_', 'pga_', 'sga_']),
    ('disk_io', ['read_per_sec', 'write_per_sec', 'seqread', 'sctread', 'physical_read',
                 'logic_read', 'dfparawrite', 'lfparawrite', 'lfsync', 'redo_']),
    ('disk_space', ['new_tbs_free', 'new_tbs_used', 'tbs_used']),
    ('network', ['tnsping', 'icmp_ping', 'incoming_network', 'outgoing_network',
                 'received_packets', 'sent_packets', 'send_total', 'receive_total',
                 'network_traffic', 'ping_result']),
    ('db', ['sess_active', 'sess_connect', 'session_pct', 'login_per', 'exec_per',
            'tps_per', 'call_per', 'logon']),
]


def _combined_classify(metric_id):
    orig = _orig_classify(metric_id)
    if orig != 'other':
        return orig
    kpi = metric_id.split('__', 1)[1] if '__' in metric_id else metric_id
    kpi_l = kpi.lower()
    for t, patterns in MARKET_PATTERNS + TELECOM_PATTERNS:
        for p in patterns:
            if p in kpi_l:
                return t
    return 'other'


# Patch v2 namespace (used by some helpers)
_v2.classify_metric_type = _combined_classify


# ------------------------------------------------------------------
# 1-min step-aware anomaly detector
# ------------------------------------------------------------------

def detect_anomalies_system_1min(metric_data, incident_dt, extract_component,
                                 z_threshold=3.0, pct_threshold=5.0,
                                 window_minutes=5, baseline_hours=4):
    """1-min variant of v4_rca.detect_anomalies_system.

    At 1-min resolution, 1 step = 1 minute, so:
      window_steps = window_minutes
      baseline_steps = baseline_hours * 60

    Uses extended classifier and includes 'mesh-cmdb' demangling for Market.
    """
    inc_idx = metric_data.index.get_indexer([incident_dt], method='nearest')[0]
    window_steps = window_minutes  # 1 step = 1 min
    baseline_steps = baseline_hours * 60

    ws = max(0, inc_idx - window_steps)
    we = min(len(metric_data), inc_idx + 2)
    bs = max(0, ws - baseline_steps)
    be = ws

    # Adaptive: need at least 30 baseline minutes, but if less available accept 15
    if be - bs < 30:
        if be - bs < 15:
            return {}

    inc_window = metric_data.iloc[ws:we]
    base_window = metric_data.iloc[bs:be]

    svc_info = defaultdict(lambda: {
        'anomalies': [],
        'max_z': 0.0,
        'max_pct': 0.0,
        'fault_types': Counter(),
    })

    for col in metric_data.columns:
        cmdb_id = col.split('__', 1)[0]
        comp = extract_component(cmdb_id)
        if not comp or is_counter(col):
            continue

        base = base_window[col].dropna()
        if len(base) < 10:
            continue
        b_mean = base.mean()
        b_std = base.std()
        if b_std < 1e-8:
            continue

        inc = inc_window[col].dropna()
        if len(inc) == 0:
            continue

        w_max = inc.max()
        w_min = inc.min()
        peak = w_max if abs(w_max - b_mean) > abs(w_min - b_mean) else w_min
        z = abs(peak - b_mean) / b_std
        pct = (peak - b_mean) / (abs(b_mean) + 1e-8) * 100

        if z < z_threshold or abs(pct) < pct_threshold:
            continue

        ftype = _combined_classify(col)
        svc_info[comp]['anomalies'].append({
            'metric': col, 'z': z, 'pct': pct,
            'baseline': b_mean, 'peak': peak, 'fault_type': ftype,
        })
        svc_info[comp]['max_z'] = max(svc_info[comp]['max_z'], z)
        svc_info[comp]['max_pct'] = max(svc_info[comp]['max_pct'], abs(pct))
        svc_info[comp]['fault_types'][ftype] += 1

    return dict(svc_info)


def find_onset_time_1min(metric_data, metric_id, incident_dt,
                         lookback_minutes=60, baseline_minutes=60,
                         min_base_samples=10):
    """1-min step-aware onset finder. Find earliest minute the given metric
    deviated significantly inside [incident - lookback_minutes, incident + 2].

    Adaptive baseline: if baseline_minutes of pre-history is not available
    (common when incident is near the start of a parquet), shrink the lookback
    symmetrically until we have at least `min_base_samples` baseline points.
    If we still cannot get enough baseline, return None (caller will fall back
    to incident midpoint).
    """
    inc_idx = metric_data.index.get_indexer([incident_dt], method='nearest')[0]

    # Adaptive: reduce lookback if we don't have enough pre-history
    for lb in [lookback_minutes, 60, 30, 15]:
        start_idx = max(0, inc_idx - lb)
        pre_base_start = max(0, start_idx - baseline_minutes)
        base = metric_data[metric_id].iloc[pre_base_start:start_idx].dropna()
        if len(base) >= min_base_samples:
            break
    else:
        return None

    b_mean = base.mean()
    b_std = base.std()
    if b_std < 1e-8:
        return None

    # Smooth the lookback with a 3-min rolling mean to reduce 1-min noise spikes
    series = metric_data[metric_id].iloc[start_idx:inc_idx + 2]
    smoothed = series.rolling(3, min_periods=1).mean()
    zs = abs(smoothed - b_mean) / b_std
    pcts = (smoothed - b_mean) / (abs(b_mean) + 1e-8) * 100
    anom_mask = (zs > 3.0) & (abs(pcts) > 5.0)
    anom = smoothed[anom_mask]
    if len(anom) > 0:
        return anom.index[0]
    return None


# ------------------------------------------------------------------
# Reason mapping + candidate lists (verbatim from predict_rcaagent.py)
# ------------------------------------------------------------------

MONTHS = {m: i for i, m in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], start=1)}

UTC_OFFSET = timedelta(hours=8)


def _parse_candidates(text):
    return [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith('- ')]


import rd_prompt as _rdp  # noqa: E402

CANDIDATES = {
    'Bank': _parse_candidates(_rdp.CAND_BANK_PROMPT),
    'Telecom': _parse_candidates(_rdp.CAND_TELECOM_PROMPT),
    'Market_cloudbed-1': _parse_candidates(_rdp.CAND_MARKET_PROMPT),
    'Market_cloudbed-2': _parse_candidates(_rdp.CAND_MARKET_PROMPT),
}

EXTRACT = {
    'Bank': extract_component_bank,
    'Telecom': extract_component_telecom,
    'Market_cloudbed-1': extract_component_market,
    'Market_cloudbed-2': extract_component_market,
}

REASON_MAP = {
    'Telecom': {
        'cpu': 'CPU fault', 'memory': 'CPU fault', 'disk_io': 'CPU fault',
        'disk_space': 'CPU fault', 'network': 'network delay',
        'db': 'db connection limit', 'tomcat_thread': 'CPU fault',
        'tomcat_req': 'CPU fault', 'jvm': 'CPU fault',
        'redis_ops': 'CPU fault', 'other': 'CPU fault',
    },
    'Bank': {
        'cpu': 'high CPU usage', 'memory': 'high memory usage',
        'disk_io': 'high disk I/O read usage', 'disk_space': 'high disk space usage',
        'network': 'network latency', 'db': 'high memory usage',
        'tomcat_thread': 'high CPU usage', 'tomcat_req': 'high CPU usage',
        'jvm': 'JVM Out of Memory (OOM) Heap', 'redis_ops': 'high memory usage',
        'other': 'high memory usage',
    },
    'Market_cloudbed-1': {
        'cpu': 'container CPU load', 'memory': 'container memory load',
        'disk_io': 'node disk read I/O consumption', 'disk_space': 'node disk space consumption',
        'network': 'container network latency', 'db': 'container CPU load',
        'tomcat_thread': 'container CPU load', 'tomcat_req': 'container CPU load',
        'jvm': 'container CPU load', 'redis_ops': 'container CPU load',
        'other': 'container CPU load',
    },
    'Market_cloudbed-2': {
        'cpu': 'container CPU load', 'memory': 'container memory load',
        'disk_io': 'container read I/O load', 'disk_space': 'node disk space consumption',
        'network': 'container network latency', 'db': 'container CPU load',
        'tomcat_thread': 'container CPU load', 'tomcat_req': 'container CPU load',
        'jvm': 'container CPU load', 'redis_ops': 'container CPU load',
        'other': 'container CPU load',
    },
}

NODE_PREFIX_MARKET = {
    'cpu': 'node CPU spike', 'memory': 'node memory consumption',
    'disk_io': 'node disk read I/O consumption', 'disk_space': 'node disk space consumption',
    'network': 'node CPU spike',
}
NODE_PREFIX_MARKET_CB2 = {
    'cpu': 'node CPU load', 'memory': 'node memory consumption',
    'disk_io': 'node disk write I/O consumption', 'disk_space': 'node disk space consumption',
    'network': 'node CPU load',
}

DEFAULT = {
    'Bank': ('Tomcat01', 'high memory usage'),
    'Telecom': ('docker_001', 'CPU fault'),
    'Market_cloudbed-1': ('node-1', 'node CPU spike'),
    'Market_cloudbed-2': ('node-1', 'node CPU load'),
}

# ------------------------------------------------------------------
# Instruction parsing
# ------------------------------------------------------------------

TIME_PAT = re.compile(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)')
DATE_PAT = re.compile(
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+(\d{1,2}),?\s*(\d{4})')


def parse_instruction(instruction):
    dates = DATE_PAT.findall(instruction)
    times = TIME_PAT.findall(instruction)
    if not dates or len(times) < 2:
        return None, None, None
    month = MONTHS[dates[0][0]]
    day = int(dates[0][1])
    year = int(dates[0][2])
    h1, m1 = int(times[0][0]), int(times[0][1])
    h2, m2 = int(times[1][0]), int(times[1][1])
    t1 = pd.Timestamp(year=year, month=month, day=day, hour=h1, minute=m1)
    t2 = pd.Timestamp(year=year, month=month, day=day, hour=h2, minute=m2)
    if t2 < t1:
        t2 = t2 + pd.Timedelta(days=1)
    mid_utc8 = t1 + (t2 - t1) / 2
    mid_utc = mid_utc8 - UTC_OFFSET
    date_label = mid_utc8.strftime('%Y-%m-%d')
    return mid_utc8, mid_utc, date_label


def count_failures(scoring_points):
    ords = re.findall(r'(\d+)-th', scoring_points)
    if ords:
        return max(int(x) for x in ords)
    if re.search(r'\bonly\b', scoring_points):
        return 1
    return 1


# ------------------------------------------------------------------
# Component picking & reason mapping (verbatim)
# ------------------------------------------------------------------

def map_reason(system, comp, info):
    if info is None or not info.get('fault_types'):
        if system.startswith('Market') and str(comp).startswith('node-'):
            return 'node CPU load' if system.endswith('cloudbed-2') else 'node CPU spike'
        return DEFAULT[system][1]
    ft = info['fault_types']
    if ft.get('jvm', 0) >= 2:
        ftype = 'jvm'
    elif ft.get('memory', 0) >= 2:
        ftype = 'memory'
    elif ft.get('db', 0) >= 3:
        ftype = 'db'
    elif ft.get('network', 0) >= 1:
        ftype = 'network'
    elif ft.get('cpu', 0) >= 2:
        ftype = 'cpu'
    elif ft.get('disk_space', 0) >= 2:
        ftype = 'disk_space'
    elif ft.get('disk_io', 0) >= 2:
        ftype = 'disk_io'
    elif ft.get('memory', 0) >= 1:
        ftype = 'memory'
    elif ft.get('cpu', 0) >= 1:
        ftype = 'cpu'
    else:
        most = [(k, v) for k, v in ft.most_common() if k != 'other']
        ftype = most[0][0] if most else 'other'
    if system.startswith('Market') and str(comp).startswith('node-'):
        tbl = NODE_PREFIX_MARKET_CB2 if system.endswith('cloudbed-2') else NODE_PREFIX_MARKET
        if ftype in tbl:
            return tbl[ftype]
        return 'node CPU load' if system.endswith('cloudbed-2') else 'node CPU spike'
    return REASON_MAP[system].get(ftype, DEFAULT[system][1])


def pick_components(svc_info, candidates, extract, n=1):
    if not svc_info:
        return []
    ranking = sorted(svc_info.items(), key=lambda x: -x[1]['max_pct'])
    cand_set = set(candidates)
    aware = []
    for comp, info in ranking:
        fam = re.sub(r'-\d+$', '', comp)
        fam = re.sub(r'2$', '', fam)
        mapped = []
        if comp in cand_set:
            mapped.append(comp)
        if fam in cand_set and fam != comp:
            mapped.append(fam)
        if not mapped:
            base = comp.split('.')[0] if '.' in comp else comp
            if base in cand_set:
                mapped.append(base)
        for m in mapped:
            aware.append((info['max_pct'], m, info, fam if fam else m))

    picks = []
    seen_fams = set()
    seen_names = set()
    if n == 1:
        for _, m, info, fam in aware:
            if m not in seen_names:
                return [(m, info)]
        return []
    for _, m, info, fam in aware:
        if m in seen_names or fam in seen_fams:
            continue
        picks.append((m, info))
        seen_fams.add(fam)
        seen_names.add(m)
        if len(picks) >= n:
            return picks
    for _, m, info, fam in aware:
        if m in seen_names:
            continue
        picks.append((m, info))
        seen_names.add(m)
        if len(picks) >= n:
            break
    return picks


def _family_key(system, comp):
    """Return a family key stripping replica suffix. Returns None if there is
    no meaningful family (e.g., unique node) so sibling logic is a no-op."""
    if not isinstance(comp, str) or not comp:
        return None
    if system == 'Bank':
        # Tomcat04 -> Tomcat, Mysql02 -> Mysql
        m = re.match(r'^([A-Za-z]+?)(\d+)$', comp)
        return m.group(1) if m else None
    if system == 'Telecom':
        # os_001 -> os, docker_001 -> docker, db_007 -> db
        m = re.match(r'^([a-z]+)_(\d+)$', comp)
        return m.group(1) if m else None
    if system.startswith('Market'):
        # frontend-1 -> frontend, frontend2-0 -> frontend2, node-5 -> node
        m = re.match(r'^([A-Za-z][A-Za-z0-9]*?)-(\d+)$', comp)
        return m.group(1) if m else None
    return None


def _quick_onset(md_primary, info, mid_utc, is_1min_primary, _onset_fn):
    """Earliest onset timestamp across the top-3 anomalous metrics of info."""
    if not (info and info.get('anomalies') and is_1min_primary):
        return None
    top = sorted(info['anomalies'], key=lambda a: -abs(a.get('pct', 0)))[:3]
    best = None
    for a in top:
        m = a['metric']
        if m not in md_primary.columns:
            continue
        try:
            t = _onset_fn(md_primary, m, mid_utc,
                         lookback_minutes=120, baseline_minutes=120)
        except Exception:
            t = None
        if t is None:
            continue
        if best is None or t < best:
            best = t
    return best


def pick_component_sibling(picks, svc_info, system, candidates, md_primary,
                           mid_utc, is_1min_primary, _onset_fn):
    """Within each pick's family, re-rank by earliest onset."""
    if not picks or not svc_info:
        return picks
    cand_set = set(candidates)
    new_picks = []
    seen = set()
    for (comp, info) in picks:
        fam = _family_key(system, comp)
        if fam is None:
            new_picks.append((comp, info))
            seen.add(comp)
            continue
        # Collect all anomalous components that map to same family
        fam_members = []
        for c2, i2 in svc_info.items():
            c2_fam = _family_key(system, c2)
            if c2_fam == fam and c2 in cand_set and c2 not in seen:
                fam_members.append((c2, i2))
        if len(fam_members) <= 1:
            new_picks.append((comp, info))
            seen.add(comp)
            continue
        # Score each member by onset (earliest wins)
        scored = []
        for c2, i2 in fam_members:
            onset = _quick_onset(md_primary, i2, mid_utc, is_1min_primary, _onset_fn)
            # Use max_pct as tiebreaker (negative so larger pct wins on tie)
            scored.append((onset, -i2.get('max_pct', 0), c2, i2))
        # Members without an onset get sentinel (sort to end)
        scored.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0, x[1]))
        best_c, best_i = scored[0][2], scored[0][3]
        new_picks.append((best_c, best_i))
        seen.add(best_c)
    return new_picks


def _fallback_json(system, n_fail, fallback_dt):
    comp, reason = DEFAULT[system]
    dt_str = fallback_dt.strftime('%Y-%m-%d %H:%M:%S') if fallback_dt is not None else ''
    obj = f'{{"root cause occurrence datetime": "{dt_str}", "root cause component": "{comp}", "root cause reason": "{reason}"}}'
    return ''.join([obj] * n_fail)


# ------------------------------------------------------------------
# Predict one query (1-min)
# ------------------------------------------------------------------

def predict_one(system, query_row, parquet_cache, results_1min_dir,
                results_ext_dir, results_dir, verbose=False):
    instruction = query_row['instruction']
    scoring_points = str(query_row['scoring_points'])
    n_fail = count_failures(scoring_points)

    mid_utc8, mid_utc, date_label = parse_instruction(instruction)
    if mid_utc is None:
        return _fallback_json(system, n_fail, fallback_dt=None), []

    # Load 1-min parquet (primary), with graceful fallbacks to 5-min if missing
    p_1min = Path(results_1min_dir) / system / date_label / 'metric_data.parquet'
    p_ext = Path(results_ext_dir) / system / date_label / 'metric_data.parquet'
    p_pool = Path(results_dir) / system / 'metric_data.parquet'
    mds = []
    for p in [p_1min, p_ext, p_pool]:
        if p.exists():
            key = str(p)
            if key not in parquet_cache:
                parquet_cache[key] = pd.read_parquet(p)
            m = parquet_cache[key]
            if m is not None and not m.empty and m.index.min() <= mid_utc <= m.index.max():
                mds.append((p, m))

    if not mds:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8), []

    extract = EXTRACT[system]
    # Run 1-min detector on the 1-min parquet first; if no anomalies,
    # try the next available parquet. We pass through the primary 1-min one
    # for the onset finder regardless.
    md_primary = mds[0][1]
    is_1min_primary = mds[0][0] == p_1min

    svc_info = {}
    md_used_for_detect = md_primary
    for (path, candidate_md) in mds:
        if path == p_1min:
            svc_info = detect_anomalies_system_1min(candidate_md, mid_utc, extract,
                                                    window_minutes=30, baseline_hours=4)
        else:
            # Fall back to legacy 5-min detector
            from v4_rca import detect_anomalies_system as _legacy_detect
            svc_info = _legacy_detect(candidate_md, mid_utc, extract)
        if svc_info:
            md_used_for_detect = candidate_md
            break

    candidates = CANDIDATES[system]
    top3 = sorted(svc_info.items(), key=lambda x: -x[1]['max_pct'])[:3] if svc_info else []

    picks = pick_components(svc_info, candidates, extract, n=n_fail)
    # FIX3: sibling disambiguation via onset
    picks = pick_component_sibling(picks, svc_info, system, candidates,
                                   md_primary, mid_utc, is_1min_primary,
                                   find_onset_time_1min)
    if not picks:
        picks = [(DEFAULT[system][0], None)] * n_fail
    while len(picks) < n_fail:
        picks.append(picks[-1])

    objs = []
    for comp, info in picks[:n_fail]:
        reason = map_reason(system, comp, info)
        onset_dt_utc8 = mid_utc8
        if info and info.get('anomalies') and is_1min_primary:
            top_metric = max(info['anomalies'], key=lambda a: abs(a['pct']))['metric']
            if top_metric in md_primary.columns:
                onset_utc = find_onset_time_1min(md_primary, top_metric, mid_utc,
                                                 lookback_minutes=120,
                                                 baseline_minutes=120)
                if onset_utc is not None:
                    onset_dt_utc8 = pd.Timestamp(onset_utc) + UTC_OFFSET
        dt_str = onset_dt_utc8.strftime('%Y-%m-%d %H:%M:%S')
        objs.append(
            f'{{"root cause occurrence datetime": "{dt_str}", '
            f'"root cause component": "{comp}", '
            f'"root cause reason": "{reason}"}}'
        )
    return ''.join(objs), top3


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
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
        pred, top3 = predict_one(args.system, row, cache,
                                 args.results_1min_dir,
                                 args.results_ext_dir,
                                 args.results_dir,
                                 verbose=(i < 3))
        preds.append(pred)
        if i < 3:
            print(f"[{args.system} #{i} task={row['task_index']}]")
            print(f"  top3: {[(c, round(inf['max_pct'], 1), dict(inf['fault_types'])) for c, inf in top3]}")
            print(f"  pred: {pred[:240]}...")
    out = pd.DataFrame({'task_index': qdf['task_index'], 'prediction': preds})
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")


if __name__ == '__main__':
    main()
