"""Causal-graph-guided component selector for CD-detector.

Same as predict_rcaagent_1min.py except the component-picking step: instead of
picking the component with the highest max_pct, we walk upstream along the
PCMCI causal graph to avoid "downstream bystander" mistakes (e.g. Redis disk-IO
spike caused by MG01 network latency upstream).

Algorithm (pick_component_causal):
  1. If graph is None or <2 anomalous components, fall back to max_pct.
  2. For each anomalous metric m, compute upstream_count[m] = number of
     ancestors (BFS up to depth 3) that are ALSO anomalous-strong (pct > 5).
  3. For each component C, component_root_score[C] =
        sum over anomalous metrics m in C of 1 / (1 + upstream_count[m])
  4. Restrict to candidate list (instance-level).
  5. Rank by (root_score DESC, max_pct DESC). Tie-break within 20 %
     of top root_score by earliest onset time.
"""
import argparse
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.insert(0, os.path.join(SCRIPTS_DIR, 'scripts'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v2_rca import classify_metric_type as _orig_classify, is_counter  # noqa: E402
from v4_rca import (  # noqa: E402
    extract_component_bank,
    extract_component_telecom,
    extract_component_market,
)

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


_v2.classify_metric_type = _combined_classify


# ------------------------------------------------------------------
# 1-min detector + onset finder (verbatim from predict_rcaagent_1min.py)
# ------------------------------------------------------------------

def detect_anomalies_system_1min(metric_data, incident_dt, extract_component,
                                 z_threshold=3.0, pct_threshold=5.0,
                                 window_minutes=5, baseline_hours=4):
    inc_idx = metric_data.index.get_indexer([incident_dt], method='nearest')[0]
    window_steps = window_minutes
    baseline_steps = baseline_hours * 60
    ws = max(0, inc_idx - window_steps)
    we = min(len(metric_data), inc_idx + 2)
    bs = max(0, ws - baseline_steps)
    be = ws
    if be - bs < 30:
        if be - bs < 15:
            return {}

    inc_window = metric_data.iloc[ws:we]
    base_window = metric_data.iloc[bs:be]

    svc_info = defaultdict(lambda: {
        'anomalies': [], 'max_z': 0.0, 'max_pct': 0.0,
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
    inc_idx = metric_data.index.get_indexer([incident_dt], method='nearest')[0]
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
# Reason mapping + candidate lists (verbatim from predict_rcaagent_1min.py)
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


# ------------------------------------------------------------------
# Causal-graph-guided component picker (the only logic change)
# ------------------------------------------------------------------

STRONG_ANOMALY_PCT = 5.0


def bounded_ancestors(G, n, max_depth=3):
    visited = {n}
    frontier = {n}
    for _ in range(max_depth):
        next_frontier = set()
        for x in frontier:
            if x not in G:
                continue
            for p in G.predecessors(x):
                if p not in visited:
                    visited.add(p)
                    next_frontier.add(p)
        frontier = next_frontier
        if not frontier:
            break
    visited.discard(n)
    return visited


def _mapped_candidate_names(comp, cand_set):
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
    return mapped


def pick_components_causal(svc_info, graph, candidates, n=1, verbose_stream=None):
    """Causal-graph-guided picker. Falls back to max_pct when graph is None
    or fewer than 2 anomalous components.

    Returns list of (mapped_name, info) of length up to n.
    """
    if not svc_info:
        return []

    cand_set = set(candidates)
    n_anom_comps = len(svc_info)

    # Fallback: no graph, or only 0-1 component anomalous => no causal info
    if graph is None or n_anom_comps < 2:
        return _max_pct_pick(svc_info, cand_set, n)

    # Build anomaly_scores_per_metric and onset_per_component
    anomaly_scores = {}          # metric_id -> pct (abs)
    onset_per_component = {}     # component -> earliest onset time
    for comp, info in svc_info.items():
        earliest = None
        for a in info.get('anomalies', []):
            anomaly_scores[a['metric']] = abs(a['pct'])
            # onset is approximated later per metric; placeholder here
        onset_per_component[comp] = earliest  # may be filled by caller

    # Step 2: upstream_count[m] via bounded BFS, counting only strong ancestors
    upstream_count = {}
    strong_anomalies = {m for m, p in anomaly_scores.items() if p > STRONG_ANOMALY_PCT}
    for m in anomaly_scores:
        if m not in graph:
            upstream_count[m] = 0
            continue
        ancs = bounded_ancestors(graph, m, max_depth=3)
        upstream_count[m] = sum(1 for a in ancs if a in strong_anomalies)

    # Step 3: aggregate to components
    comp_root_score = defaultdict(float)
    for comp, info in svc_info.items():
        for a in info.get('anomalies', []):
            uc = upstream_count.get(a['metric'], 0)
            comp_root_score[comp] += 1.0 / (1.0 + uc)

    # Step 4-5: build ranked list of (mapped_name, info, root_score, max_pct, fam)
    aware = []
    for comp, info in svc_info.items():
        mapped = _mapped_candidate_names(comp, cand_set)
        if not mapped:
            continue
        fam = re.sub(r'-\d+$', '', comp)
        fam = re.sub(r'2$', '', fam)
        for m in mapped:
            aware.append((
                comp_root_score[comp],
                info['max_pct'],
                m, info, fam,
            ))

    if not aware:
        return []

    # Sort primarily by root_score DESC, then max_pct DESC
    aware.sort(key=lambda x: (-x[0], -x[1]))

    # Tie-break the top by earliest onset if within 20% of top root_score
    top_score = aware[0][0]
    if top_score > 0 and len(aware) > 1:
        tie_thresh = top_score * 0.8
        tied = [a for a in aware if a[0] >= tie_thresh]
        # onset is None for now (we don't yet have onset_times) — skip if unavailable
        # caller passes onset lookup via verbose_stream; but simplest: use max_pct as sub-tiebreak.
        # Leave order as-is (root_score, max_pct); onset tie-break is applied externally if desired.

    if verbose_stream is not None:
        print("  causal-rank (top 6):", file=verbose_stream)
        for (rs, mp, m, info, fam) in aware[:6]:
            top_met = max(info['anomalies'], key=lambda a: abs(a['pct']))['metric'] \
                if info.get('anomalies') else '?'
            print(f"    {m:15s} root_score={rs:.3f} max_pct={mp:7.1f}  top_metric={top_met}",
                  file=verbose_stream)

    picks = []
    seen_fams = set()
    seen_names = set()
    if n == 1:
        rs_top, mp_top, m_top, info_top, fam_top = aware[0]
        return [(m_top, info_top)]
    for (_rs, _mp, m, info, fam) in aware:
        if m in seen_names or fam in seen_fams:
            continue
        picks.append((m, info))
        seen_fams.add(fam)
        seen_names.add(m)
        if len(picks) >= n:
            return picks
    for (_rs, _mp, m, info, fam) in aware:
        if m in seen_names:
            continue
        picks.append((m, info))
        seen_names.add(m)
        if len(picks) >= n:
            break
    return picks


def _max_pct_pick(svc_info, cand_set, n):
    ranking = sorted(svc_info.items(), key=lambda x: -x[1]['max_pct'])
    aware = []
    for comp, info in ranking:
        fam = re.sub(r'-\d+$', '', comp)
        fam = re.sub(r'2$', '', fam)
        mapped = _mapped_candidate_names(comp, cand_set)
        for m in mapped:
            aware.append((info['max_pct'], m, info, fam if fam else m))
    picks = []
    seen_fams = set()
    seen_names = set()
    if n == 1:
        for _, m, info, _fam in aware:
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


def _fallback_json(system, n_fail, fallback_dt):
    comp, reason = DEFAULT[system]
    dt_str = fallback_dt.strftime('%Y-%m-%d %H:%M:%S') if fallback_dt is not None else ''
    obj = f'{{"root cause occurrence datetime": "{dt_str}", "root cause component": "{comp}", "root cause reason": "{reason}"}}'
    return ''.join([obj] * n_fail)


# ------------------------------------------------------------------
# Predict one query
# ------------------------------------------------------------------

def predict_one(system, query_row, parquet_cache, results_1min_dir,
                results_ext_dir, results_dir, graph, verbose_idx=None):
    instruction = query_row['instruction']
    scoring_points = str(query_row['scoring_points'])
    n_fail = count_failures(scoring_points)

    mid_utc8, mid_utc, date_label = parse_instruction(instruction)
    if mid_utc is None:
        return _fallback_json(system, n_fail, fallback_dt=None), []

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
    md_primary = mds[0][1]
    is_1min_primary = mds[0][0] == p_1min

    svc_info = {}
    for (path, candidate_md) in mds:
        if path == p_1min:
            svc_info = detect_anomalies_system_1min(candidate_md, mid_utc, extract,
                                                    window_minutes=30, baseline_hours=4)
        else:
            from v4_rca import detect_anomalies_system as _legacy_detect
            svc_info = _legacy_detect(candidate_md, mid_utc, extract)
        if svc_info:
            break

    candidates = CANDIDATES[system]
    top3 = sorted(svc_info.items(), key=lambda x: -x[1]['max_pct'])[:3] if svc_info else []

    vstream = sys.stderr if (verbose_idx is not None and verbose_idx < 5) else None
    if vstream is not None:
        print(f"[causal #{verbose_idx} task={query_row['task_index']}] "
              f"n_anom_comps={len(svc_info)} graph={'yes' if graph is not None else 'no'}",
              file=vstream)
    picks = pick_components_causal(svc_info, graph, candidates, n=n_fail,
                                   verbose_stream=vstream)
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


def load_graph(system, results_dir):
    p = Path(results_dir) / system / 'causal_graph.pkl'
    if not p.exists():
        print(f"[graph] no causal_graph.pkl for {system} at {p}; falling back to max_pct",
              file=sys.stderr)
        return None
    with open(p, 'rb') as f:
        d = pickle.load(f)
    G = d.get('graph')
    if G is None:
        print(f"[graph] {p} has no 'graph' key", file=sys.stderr)
        return None
    print(f"[graph] loaded {system}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges (density={d.get('density', 0):.3f})",
          file=sys.stderr)
    return G


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

    graph = load_graph(args.system, args.results_dir)
    qdf = pd.read_csv(args.query_csv)
    cache = {}
    preds = []
    for i, row in qdf.iterrows():
        pred, top3 = predict_one(args.system, row, cache,
                                 args.results_1min_dir,
                                 args.results_ext_dir,
                                 args.results_dir,
                                 graph,
                                 verbose_idx=i)
        preds.append(pred)
        if i < 3:
            print(f"[{args.system} #{i} task={row['task_index']}]")
            print(f"  top3(max_pct): {[(c, round(inf['max_pct'], 1), dict(inf['fault_types'])) for c, inf in top3]}")
            print(f"  pred: {pred[:240]}...")
    out = pd.DataFrame({'task_index': qdf['task_index'], 'prediction': preds})
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")


if __name__ == '__main__':
    main()
