"""Build per-date 1-minute resampled parquets for fine-grained onset detection."""
import pandas as pd
from pathlib import Path
import argparse


def pivot_1min(df, min_samples=200):
    df = df.copy()
    df['metric_id'] = df['cmdb_id'].astype(str) + '__' + df['kpi_name'].astype(str)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    counts = df.groupby('metric_id').size()
    good = counts[counts >= min_samples].index
    df = df[df['metric_id'].isin(good)]
    pivot = df.pivot_table(index='datetime', columns='metric_id', values='value', aggfunc='mean')
    pivot = pivot.resample('1min').mean().ffill(limit=2)
    null_frac = pivot.isnull().mean()
    pivot = pivot[null_frac[null_frac < 0.5].index]
    const = pivot.columns[pivot.nunique() <= 1]
    pivot = pivot.drop(columns=const)
    return pivot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True,
                    choices=['Bank', 'Telecom', 'Market_cloudbed-1', 'Market_cloudbed-2'])
    ap.add_argument('--data_root', default='$HOME/auditstack/data/OpenRCA/dataset')
    ap.add_argument('--output_dir', default='$HOME/auditstack/results_1min')
    args = ap.parse_args()

    if args.system.startswith('Market_'):
        cloudbed = args.system.replace('Market_', '')
        tele = Path(args.data_root) / 'Market' / cloudbed / 'telemetry'
    else:
        tele = Path(args.data_root) / args.system / 'telemetry'
    out_base = Path(args.output_dir) / args.system
    out_base.mkdir(parents=True, exist_ok=True)

    for dd in sorted([d for d in tele.iterdir() if d.is_dir()]):
        mdir = dd / 'metric'
        if not mdir.exists():
            continue
        dfs = []
        if args.system == 'Bank':
            f = mdir / 'metric_container.csv'
            if f.exists():
                dfs.append(pd.read_csv(f))
        elif args.system.startswith('Market_'):
            for fname in ['metric_container.csv', 'metric_node.csv']:
                f = mdir / fname
                if f.exists():
                    df = pd.read_csv(f)
                    if 'cmdb_id' in df.columns:
                        df = df[['timestamp', 'cmdb_id', 'kpi_name', 'value']]
                        df['timestamp'] = df['timestamp'].astype(float)
                        dfs.append(df)
        else:  # Telecom
            for f in sorted(mdir.glob('metric_*.csv')):
                if f.name == 'metric_app.csv':
                    continue
                try:
                    df = pd.read_csv(f)
                    if 'cmdb_id' not in df.columns or 'name' not in df.columns:
                        continue
                    df = df.rename(columns={'name': 'kpi_name'})
                    df['timestamp'] = df['timestamp'] / 1000.0
                    df = df[['timestamp', 'cmdb_id', 'kpi_name', 'value']]
                    dfs.append(df)
                except Exception as e:
                    print(f"  skip {f.name}: {e}")
        if not dfs:
            continue
        full = pd.concat(dfs, ignore_index=True)
        pivot = pivot_1min(full)
        label = dd.name.replace('_', '-')
        out_dir = out_base / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / 'metric_data.parquet'
        pivot.to_parquet(out)
        print(f"{dd.name}: {pivot.shape} idx {pivot.index.min()} -> {pivot.index.max()}")


if __name__ == '__main__':
    main()
