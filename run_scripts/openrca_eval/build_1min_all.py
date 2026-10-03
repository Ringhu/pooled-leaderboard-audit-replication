"""Build per-date 1-minute resampled parquets for ALL 4 systems (Bank, Telecom,
Market cb-1, cb-2), scoped to dates that the RCA-agent test queries reference.

Output: results_1min/{system}/{YYYY-MM-DD}/metric_data.parquet

Per-system merging:
  Bank: metric_container.csv only (skip metric_app.csv -- different schema).
        Timestamps already in seconds. Native 60s -> resample 1min mean.
  Telecom: metric_container.csv, metric_middleware.csv, metric_service.csv,
           metric_node.csv. Skip metric_app.csv. schema: name -> kpi_name;
           timestamp is ms (divide by 1000). metric_node.csv is 1s native
           -- downsample via resample('1min').mean(). Others are 60s.
  Market cb-1/cb-2: metric_container.csv, metric_mesh.csv, metric_node.csv.
                    All 60s already. Just pivot.

Cleaning:
  - drop metric_ids with <20 samples
  - .resample('1min').mean()
  - .ffill(limit=3)
  - drop columns with >30% null
  - drop constant columns
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


MONTHS = {m: i for i, m in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], start=1)}

DATE_PAT = re.compile(
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+(\d{1,2}),?\s*(\d{4})')
TIME_PAT = re.compile(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)')


def collect_query_dates(query_csv):
    """Parse the query CSV and return set of UTC+8 date labels the queries touch."""
    df = pd.read_csv(query_csv)
    dates = set()
    for ins in df['instruction']:
        d = DATE_PAT.findall(ins)
        t = TIME_PAT.findall(ins)
        if not d or len(t) < 2:
            continue
        mo = MONTHS[d[0][0]]
        dy = int(d[0][1])
        y = int(d[0][2])
        h1, m1 = int(t[0][0]), int(t[0][1])
        h2, m2 = int(t[1][0]), int(t[1][1])
        t1 = pd.Timestamp(year=y, month=mo, day=dy, hour=h1, minute=m1)
        t2 = pd.Timestamp(year=y, month=mo, day=dy, hour=h2, minute=m2)
        if t2 < t1:
            t2 = t2 + pd.Timedelta(days=1)
        mid_utc8 = t1 + (t2 - t1) / 2
        dates.add(mid_utc8.strftime('%Y_%m_%d'))
    return dates


def pivot_1min(df, min_samples=20, max_null_frac=0.30, ffill_limit=3):
    df = df.copy()
    df['metric_id'] = df['cmdb_id'].astype(str) + '__' + df['kpi_name'].astype(str)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    counts = df.groupby('metric_id').size()
    good = counts[counts >= min_samples].index
    df = df[df['metric_id'].isin(good)]

    pivot = df.pivot_table(index='datetime', columns='metric_id',
                           values='value', aggfunc='mean')
    pivot = pivot.resample('1min').mean()
    pivot = pivot.ffill(limit=ffill_limit)

    null_frac = pivot.isnull().mean()
    keep = null_frac[null_frac < max_null_frac].index
    pivot = pivot[keep]

    # Drop constant columns
    const = pivot.columns[pivot.nunique(dropna=True) <= 1]
    pivot = pivot.drop(columns=const)

    # Note: Do NOT dropna rows globally -- v2 detector handles per-column dropna.
    return pivot


def load_bank(mdir):
    f = mdir / 'metric_container.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # Standardize schema: expects timestamp(sec), cmdb_id, kpi_name, value
    if 'kpi_name' not in df.columns and 'name' in df.columns:
        df = df.rename(columns={'name': 'kpi_name'})
    df = df[['timestamp', 'cmdb_id', 'kpi_name', 'value']]
    df['timestamp'] = df['timestamp'].astype(float)
    return df


def load_telecom(mdir):
    dfs = []
    target_files = ['metric_container.csv', 'metric_middleware.csv',
                    'metric_service.csv', 'metric_node.csv']
    for fname in target_files:
        f = mdir / fname
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
            if 'cmdb_id' not in df.columns or 'name' not in df.columns:
                continue
            df = df.rename(columns={'name': 'kpi_name'})
            df['timestamp'] = df['timestamp'].astype(float) / 1000.0
            df = df[['timestamp', 'cmdb_id', 'kpi_name', 'value']]
            dfs.append(df)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_market(mdir):
    dfs = []
    for fname in ['metric_container.csv', 'metric_mesh.csv', 'metric_node.csv']:
        f = mdir / fname
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
            if 'cmdb_id' not in df.columns:
                continue
            if 'kpi_name' not in df.columns and 'name' in df.columns:
                df = df.rename(columns={'name': 'kpi_name'})
            if 'kpi_name' not in df.columns:
                continue
            df = df[['timestamp', 'cmdb_id', 'kpi_name', 'value']]
            df['timestamp'] = df['timestamp'].astype(float)
            dfs.append(df)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


SYSTEM_LOADERS = {
    'Bank': load_bank,
    'Telecom': load_telecom,
    'Market_cloudbed-1': load_market,
    'Market_cloudbed-2': load_market,
}


QUERY_CSVS = {
    'Bank': 'bank_q.csv',
    'Telecom': 'tel_q.csv',
    'Market_cloudbed-1': 'mkt1_q.csv',
    'Market_cloudbed-2': 'mkt2_q.csv',
}


def system_tele_dir(data_root, system):
    if system.startswith('Market_'):
        cloudbed = system.replace('Market_', '')
        return Path(data_root) / 'Market' / cloudbed / 'telemetry'
    return Path(data_root) / system / 'telemetry'


def build_one(system, data_root, output_dir, query_csv_dir):
    loader = SYSTEM_LOADERS[system]
    tele = system_tele_dir(data_root, system)
    out_base = Path(output_dir) / system
    out_base.mkdir(parents=True, exist_ok=True)

    query_csv = Path(query_csv_dir) / QUERY_CSVS[system]
    needed = collect_query_dates(query_csv)
    print(f"[{system}] needs {len(needed)} date(s): {sorted(needed)}")

    summary = []
    for date_folder in sorted(needed):
        # Some dates might not exist as a folder -- use raw folder name YYYY_MM_DD
        dd = tele / date_folder
        if not dd.exists() or not dd.is_dir():
            print(f"  [{system}] MISSING telemetry dir: {dd}")
            continue
        mdir = dd / 'metric'
        if not mdir.exists():
            print(f"  [{system}] no metric/ in {dd}")
            continue
        full = loader(mdir)
        if full is None or full.empty:
            print(f"  [{system}] empty load for {dd}")
            continue
        pivot = pivot_1min(full)
        if pivot.empty:
            print(f"  [{system}] empty pivot for {dd}")
            continue
        label = dd.name.replace('_', '-')
        out_dir = out_base / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / 'metric_data.parquet'
        pivot.to_parquet(out)
        null_med = pivot.isnull().mean().median()
        summary.append((label, pivot.shape, null_med))
        print(f"  {label}: shape={pivot.shape} null_median={null_med:.2%}")

    print(f"[{system}] DONE, wrote {len(summary)} parquets")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True,
                    choices=['Bank', 'Telecom', 'Market_cloudbed-1',
                             'Market_cloudbed-2', 'all'])
    ap.add_argument('--data_root',
                    default='$HOME/auditstack/data/OpenRCA/dataset')
    ap.add_argument('--output_dir',
                    default='$HOME/auditstack/results_1min')
    ap.add_argument('--query_csv_dir',
                    default='$HOME/auditstack/scripts/openrca_eval')
    args = ap.parse_args()

    systems = list(SYSTEM_LOADERS.keys()) if args.system == 'all' else [args.system]
    for sysn in systems:
        build_one(sysn, args.data_root, args.output_dir, args.query_csv_dir)


if __name__ == '__main__':
    main()
