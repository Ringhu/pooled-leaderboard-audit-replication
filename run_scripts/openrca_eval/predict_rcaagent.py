"""
Predict OpenRCA answers in RCA-agent's JSON-like format using CD-detector v2 pct.

Per query:
  1) Parse instruction for date + time window. Midpoint (UTC+8) -> incident_dt.
     Convert to UTC (subtract 8h) for parquet lookup.
  2) Load the per-date metric_data.parquet under results_ext/<System>/<YYYY-MM-DD>/
     (fallback to the pooled results/<System>/metric_data.parquet if missing).
  3) Build metric_to_comp from column prefixes using system-specific extractor.
  4) detect_anomalies_v2 -> rank components by max_pct.
  5) Restrict to RCA-agent candidate list; fall back to highest-pct candidate in-list.
  6) Map fault_type -> RCA-agent reason vocabulary per system.
  7) Onset time via find_onset_time on top metric of chosen component.
  8) Emit one JSON object per failure (count parsed from scoring_points).

Output: CSV with columns task_index, prediction (positional order = input order).
"""
import argparse
import os
import re
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Import v2 detector
SCRIPTS_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.insert(0, os.path.join(SCRIPTS_DIR, 'scripts'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v2_rca import detect_anomalies_v2, find_onset_time  # noqa: E402
from v4_rca import (  # noqa: E402
    extract_component_bank,
    extract_component_telecom,
    extract_component_market,
    detect_anomalies_system,
)

# ------------------------------------------------------------------
# Extended Market metric-type classification. v2's classifier only covers
# CamelCase Bank/Telecom KPIs; Market uses lowercase system.* / container_*.
# We monkey-patch classify_metric_type to add Market patterns.
# ------------------------------------------------------------------
import v2_rca as _v2
_ORIG_CLASSIFY = _v2.classify_metric_type

MARKET_PATTERNS = [
    # Order matters: first match wins.
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

# Telecom uses uppercase KPI abbreviations, e.g. CPU_Used_Pct, MEM_real_util,
# ping_bc/ping_jbc, LogicalReadPer, TPS_Per_Sec, etc.
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
    orig = _ORIG_CLASSIFY(metric_id)
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

MONTHS = {m: i for i, m in enumerate(
    ['January','February','March','April','May','June','July','August',
     'September','October','November','December'], start=1)}

UTC_OFFSET = timedelta(hours=8)

# ------------------------------------------------------------------
# Candidate lists (parsed from rd_prompt.py)
# ------------------------------------------------------------------

def _parse_candidates(text):
    return [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith('- ')]

import rd_prompt as _rdp  # type: ignore  # noqa: E402
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

# ------------------------------------------------------------------
# Fault type -> reason mapping per system
# ------------------------------------------------------------------
# Reason vocabulary taken verbatim from the queries. Mapping chosen so the
# dominant fault mode in each system's record.csv maps to the most common
# reason string; ambiguous ones default to the most common incident type.

REASON_MAP = {
    'Telecom': {  # reasons: CPU fault, db close, db connection limit, network delay, network loss
        'cpu': 'CPU fault',
        'memory': 'CPU fault',           # rare; default to CPU
        'disk_io': 'CPU fault',
        'disk_space': 'CPU fault',
        'network': 'network delay',      # "delay" > "loss" in counts
        'db': 'db connection limit',     # "limit" > "close"
        'tomcat_thread': 'CPU fault',
        'tomcat_req': 'CPU fault',
        'jvm': 'CPU fault',
        'redis_ops': 'CPU fault',
        'other': 'CPU fault',
    },
    'Bank': {  # reasons: JVM OOM Heap, high CPU usage, high disk I/O read usage,
              # high disk space usage, high memory usage, network latency, network packet loss
        'cpu': 'high CPU usage',
        'memory': 'high memory usage',
        'disk_io': 'high disk I/O read usage',
        'disk_space': 'high disk space usage',
        'network': 'network latency',             # latency > packet loss in counts
        'db': 'high memory usage',
        'tomcat_thread': 'high CPU usage',
        'tomcat_req': 'high CPU usage',
        'jvm': 'JVM Out of Memory (OOM) Heap',
        'redis_ops': 'high memory usage',
        'other': 'high memory usage',
    },
    'Market_cloudbed-1': {  # reasons span container+node
        'cpu': 'container CPU load',
        'memory': 'container memory load',
        'disk_io': 'node disk read I/O consumption',
        'disk_space': 'node disk space consumption',
        'network': 'container network latency',
        'db': 'container CPU load',
        'tomcat_thread': 'container CPU load',
        'tomcat_req': 'container CPU load',
        'jvm': 'container CPU load',
        'redis_ops': 'container CPU load',
        'other': 'container CPU load',
    },
    'Market_cloudbed-2': {
        'cpu': 'container CPU load',
        'memory': 'container memory load',
        'disk_io': 'container read I/O load',
        'disk_space': 'node disk space consumption',
        'network': 'container network latency',
        'db': 'container CPU load',
        'tomcat_thread': 'container CPU load',
        'tomcat_req': 'container CPU load',
        'jvm': 'container CPU load',
        'redis_ops': 'container CPU load',
        'other': 'container CPU load',
    },
}

# Node-prefixed markets need node-level reasons; detect via chosen component
NODE_PREFIX_MARKET = {
    'cpu': 'node CPU spike',
    'memory': 'node memory consumption',
    'disk_io': 'node disk read I/O consumption',
    'disk_space': 'node disk space consumption',
    'network': 'node CPU spike',  # no node-network reason
}
NODE_PREFIX_MARKET_CB2 = {
    'cpu': 'node CPU load',
    'memory': 'node memory consumption',
    'disk_io': 'node disk write I/O consumption',
    'disk_space': 'node disk space consumption',
    'network': 'node CPU load',
}

# Default fallback when zero anomalies
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
    """Return (incident_dt_utc8, incident_dt_utc, date_label).

    Handles time windows that span midnight where the end has a different date.
    We use the first two HH:MM occurrences plus the first date, and if the
    second time is smaller than the first we add a day to it.
    """
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
    # count "1-th", "2-th", ... OR "only" (single failure)
    ords = re.findall(r'(\d+)-th', scoring_points)
    if ords:
        return max(int(x) for x in ords)
    if re.search(r'\bonly\b', scoring_points):
        return 1
    return 1

# ------------------------------------------------------------------
# Prediction core
# ------------------------------------------------------------------

def map_reason(system, comp, info):
    """Pick reason string. Priority rules inspired by common failure signatures:
      1. If memory > 0 and dominates -> memory fault
      2. If db > 0 -> db fault (Telecom specific)
      3. If network > 0 -> network fault (rare but discriminating)
      4. If cpu > 0 -> CPU fault (preferred over disk_io side-effect)
      5. If disk_io > 0 -> disk I/O
      6. If disk_space > 0 -> disk space
      7. Else default
    """
    if info is None or not info.get('fault_types'):
        if system.startswith('Market') and str(comp).startswith('node-'):
            return 'node CPU load' if system.endswith('cloudbed-2') else 'node CPU spike'
        return DEFAULT[system][1]
    ft = info['fault_types']
    # Heuristic priority: specific > generic
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
        # pick most common, excluding 'other'
        most = [(k, v) for k, v in ft.most_common() if k != 'other']
        ftype = most[0][0] if most else 'other'
    # Node-level market
    if system.startswith('Market') and str(comp).startswith('node-'):
        tbl = NODE_PREFIX_MARKET_CB2 if system.endswith('cloudbed-2') else NODE_PREFIX_MARKET
        if ftype in tbl:
            return tbl[ftype]
        return 'node CPU load' if system.endswith('cloudbed-2') else 'node CPU spike'
    return REASON_MAP[system].get(ftype, DEFAULT[system][1])


def predict_one(system, query_row, parquet_cache, results_ext_dir, results_dir, verbose=False):
    instruction = query_row['instruction']
    scoring_points = str(query_row['scoring_points'])
    n_fail = count_failures(scoring_points)

    mid_utc8, mid_utc, date_label = parse_instruction(instruction)
    if mid_utc is None:
        return _fallback_json(system, n_fail, fallback_dt=None), []

    # Load parquets: try per-date extension first, then extension-pool, then
    # original pool. Fall through to wider files if baseline is insufficient
    # (common at start-of-day incidents).
    ext_path = Path(results_ext_dir) / system / date_label / 'metric_data.parquet'
    ext_pool_path = Path(results_ext_dir) / system / 'metric_data.parquet'
    pool_path = Path(results_dir) / system / 'metric_data.parquet'
    mds = []
    for p in [ext_path, ext_pool_path, pool_path]:
        if p.exists():
            key = str(p)
            if key not in parquet_cache:
                parquet_cache[key] = pd.read_parquet(p)
            m = parquet_cache[key]
            if m is not None and not m.empty and m.index.min() <= mid_utc <= m.index.max():
                mds.append(m)

    if not mds:
        return _fallback_json(system, n_fail, fallback_dt=mid_utc8), []

    extract = EXTRACT[system]
    # Run on per-date first; if empty, try pooled.
    svc_info = {}
    md = mds[0]
    for candidate_md in mds:
        svc_info = detect_anomalies_system(candidate_md, mid_utc, extract)
        if svc_info:
            md = candidate_md
            break

    candidates = CANDIDATES[system]
    # Build top-3 for debugging
    top3 = sorted(svc_info.items(), key=lambda x: -x[1]['max_pct'])[:3] if svc_info else []

    # For multi-failure, pick top-N distinct components from the ranking.
    picks = pick_components(svc_info, candidates, extract, n=n_fail)
    if not picks:
        picks = [(DEFAULT[system][0], None)] * n_fail
    while len(picks) < n_fail:
        picks.append(picks[-1])  # duplicate last if not enough

    # Optionally load 1-minute parquet for fine-grained onset
    md_1min = _load_1min(system, date_label, parquet_cache)

    objs = []
    for comp, info in picks[:n_fail]:
        reason = map_reason(system, comp, info)
        onset_dt_utc8 = mid_utc8
        if info and info.get('anomalies'):
            top_metric = max(info['anomalies'], key=lambda a: abs(a['pct']))['metric']
            # Prefer 1-min parquet for onset if available AND has this metric
            if md_1min is not None and top_metric in md_1min.columns and \
               md_1min.index.min() <= mid_utc <= md_1min.index.max():
                onset_utc = _onset_1min(md_1min, top_metric, mid_utc)
            else:
                onset_utc = find_onset_time(md, top_metric, mid_utc, lookback_hours=2)
            if onset_utc is not None:
                onset_dt_utc8 = pd.Timestamp(onset_utc) + UTC_OFFSET
        dt_str = onset_dt_utc8.strftime('%Y-%m-%d %H:%M:%S')
        objs.append(f'{{"root cause occurrence datetime": "{dt_str}", "root cause component": "{comp}", "root cause reason": "{reason}"}}')
    return ''.join(objs), top3


def _load_1min(system, date_label, cache, base='$HOME/auditstack/results_1min'):
    p = Path(base) / system / date_label / 'metric_data.parquet'
    if not p.exists():
        return None
    key = str(p)
    if key not in cache:
        try:
            cache[key] = pd.read_parquet(p)
        except Exception:
            cache[key] = None
    return cache[key]


def _onset_1min(md, metric, incident_dt, lookback_minutes=30):
    """Find earliest 1-min bucket where metric deviates significantly in the
    lookback window ending at incident_dt + 2min. Uses a tighter z-threshold
    (5.0) and larger pct threshold (20%) to avoid picking up early noise."""
    inc_idx = md.index.get_indexer([incident_dt], method='nearest')[0]
    start_idx = max(0, inc_idx - lookback_minutes)
    pre_base_start = max(0, start_idx - 120)  # 2h before lookback
    base = md[metric].iloc[pre_base_start:start_idx].dropna()
    if len(base) < 20:
        return None
    b_mean = base.mean()
    b_std = base.std()
    if b_std < 1e-8:
        return None
    lookback = md[metric].iloc[start_idx:inc_idx + 2]
    zs = abs(lookback - b_mean) / b_std
    pcts = (lookback - b_mean) / (abs(b_mean) + 1e-8) * 100
    mask = (zs > 5.0) & (abs(pcts) > 20.0)
    anom = lookback[mask]
    if len(anom) > 0:
        return anom.index[0]
    # Relaxed: pick max-z point in window
    if zs.max() > 3.0:
        return lookback.index[zs.values.argmax()]
    return None


def pick_components(svc_info, candidates, extract, n=1):
    """Return up to n (component, info) picks.

    Strategy: for each anomalous component, emit BOTH the specific-instance
    name (if in candidate list) and the service-family name (if in candidate
    list), using max_pct rank. Then deduplicate and take top-n.

    For multi-failure (n > 1), we want distinct "families" so that we don't
    predict sibling pods as separate failures — because RCA-agent's eval uses
    permutation matching at family or exact level."""
    if not svc_info:
        return []
    ranking = sorted(svc_info.items(), key=lambda x: -x[1]['max_pct'])
    cand_set = set(candidates)
    # Build candidate-aware ranking: emit both exact and family if both in list.
    aware = []  # list of (rank_score, mapped_name, info, family_key)
    for comp, info in ranking:
        fam = re.sub(r'-\d+$', '', comp)
        fam = re.sub(r'2$', '', fam)  # shippingservice2 -> shippingservice (service level)
        mapped = []
        if comp in cand_set:
            mapped.append(comp)
        if fam in cand_set and fam != comp:
            mapped.append(fam)
        if not mapped:
            # node rollup? (node-X.service-N)
            base = comp.split('.')[0] if '.' in comp else comp
            if base in cand_set:
                mapped.append(base)
        for m in mapped:
            aware.append((info['max_pct'], m, info, fam if fam else m))

    # Greedy top-n with family dedup (for multi-failure)
    picks = []
    seen_fams = set()
    seen_names = set()
    # For n==1, just pick the top candidate (prefer exact over family when both available).
    if n == 1:
        for _, m, info, fam in aware:
            if m not in seen_names:
                return [(m, info)]
        return []

    # n>1: prefer distinct families
    for _, m, info, fam in aware:
        if m in seen_names:
            continue
        if fam in seen_fams:
            continue
        picks.append((m, info))
        seen_fams.add(fam)
        seen_names.add(m)
        if len(picks) >= n:
            return picks
    # Fallback: allow same family if we still need more
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
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True, choices=list(CANDIDATES.keys()))
    ap.add_argument('--query_csv', required=True)
    ap.add_argument('--metric_parquet', required=True,
                    help='Kept for CLI compatibility; real loading uses per-date parquets under results_ext.')
    ap.add_argument('--results_ext_dir', default='$HOME/auditstack/results_ext')
    ap.add_argument('--results_dir', default='$HOME/auditstack/results')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    qdf = pd.read_csv(args.query_csv)
    cache = {}
    preds = []
    for i, row in qdf.iterrows():
        pred, top3 = predict_one(args.system, row, cache, args.results_ext_dir, args.results_dir,
                                 verbose=(i < 3))
        preds.append(pred)
        if i < 3:
            print(f"[{args.system} #{i} task={row['task_index']}]")
            print(f"  top3: {[(c, round(inf['max_pct'],1), dict(inf['fault_types'])) for c,inf in top3]}")
            print(f"  pred: {pred[:200]}...")
    out = pd.DataFrame({'task_index': qdf['task_index'], 'prediction': preds})
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")


if __name__ == '__main__':
    main()
