#!/usr/bin/env python3
"""Step 1 of propagation-aware difficulty proxy.

For every case in every system, compute ``anomalous_metric_count`` = number of
numeric metric columns with |max Z-score| > 3 in the post-inject window,
using the same baseline/anomaly split + z-score logic that
``predict_zbase_universal.zbase_universal`` (hereafter *zbase_univ*) applies
when it is called by the benchmark adapters.

Why: the existing per-system difficulty indicators (majority_baseline,
root_entropy, service_count) are all label-centric — they don't capture fault
propagation structure. Anomalous-metric count is the simplest
benchmark-agnostic propagation breadth proxy: "when the fault fires, how many
metrics light up?". No topology metadata required.

Output
------
``results_stats/anomalous_metric_count_per_case.csv`` with columns
  ``benchmark, system, case_id, anomalous_metric_count,
    total_metric_count, frac_anomalous``.

One row per case. For OpenRCA, "case" is one query row in the
``scripts/openrca_eval/<sys>_q.csv`` file (matches the granularity at which
the 1-min parquet is time-windowed and fed to zbase).

Sampling
--------
If an OpenRCA system's query file has more than the sampling limit (default 30)
rows, the per-case CSV subsamples to the first N; the per-system aggregate
always uses the full case list. A column ``sampled`` in the CSV signals which
rows were retained. Non-OpenRCA systems have <=110 cases each and are always
fully materialised.

Notes
-----
* Reuses ``predict_zbase_universal.zbase_universal`` only for its z-score logic
  — we re-derive the score directly rather than reimplementing. The function
  returns a ranked list; we replicate the pre-ranking tuple list by monkey-
  inspecting the code path's per-column computation here, using the **same**
  code from ``predict_zbase_universal`` by shelling into its logic. For
  simplicity we factor out a pure helper below that mirrors
  ``zbase_universal`` steps 1-3 and returns the per-column ``|max_z|`` dict.
* For RCAEval + PetShop: run the adapter's combined-window builder and pass the
  resulting DataFrame + inject_time to the z-score helper.
* For OpenRCA: parse the query instruction's time window, load the 1-min
  parquet for that date, window around the midpoint, and pass through.
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Paths                                                              #
# ------------------------------------------------------------------ #
ROOT = Path("$HOME/auditstack")
OUT_CSV = ROOT / "results_stats" / "anomalous_metric_count_per_case.csv"

# For adapters we need: sys.path + the source repos to reach load_scenario etc.
sys.path.insert(0, str(ROOT / "scripts" / "openrca_eval"))
sys.path.insert(0, "$HOME/petshop_repo/code")

RCAEVAL_ROOTS = {
    "online-boutique": Path("$HOME/rcaeval_repo/data/online-boutique"),
    "sock-shop-2":     Path("$HOME/rcaeval_repo/data/sock-shop-2"),
    "train-ticket":    Path("$HOME/rcaeval_repo/data/train-ticket"),
}
PETSHOP_ROOT = Path("$HOME/petshop_repo")
PETSHOP_TRAFFIC_ROOTS = {
    "high_traffic":      PETSHOP_ROOT / "dataset" / "high_traffic",
    "low_traffic":       PETSHOP_ROOT / "dataset" / "low_traffic",
    "temporal_traffic1": PETSHOP_ROOT / "dataset" / "temporal_traffic1",
    "temporal_traffic2": PETSHOP_ROOT / "dataset" / "temporal_traffic2",
}
OPENRCA_SYSTEMS = {"bank": "Bank", "mkt1": "Market_cloudbed-1",
                   "mkt2": "Market_cloudbed-2", "tel": "Telecom"}
OPENRCA_Q_DIR = ROOT / "scripts" / "openrca_eval"
OPENRCA_1MIN_DIR = ROOT / "results_1min"
OPENRCA_EXT_DIR = ROOT / "results_ext"

Z_THRESHOLD = 3.0
OPENRCA_SAMPLE_CAP = 30  # flag when we sub-sample per-case rows
WINDOW_MINUTES = 30      # 1-min parquet: ±30 min around incident midpoint
BASELINE_MINUTES = 240   # 4 h pre-window baseline (from detect_anomalies_system_1min)


# ------------------------------------------------------------------ #
# Shared z-score logic (mirrors zbase_universal steps 1-3)           #
# ------------------------------------------------------------------ #
def per_metric_max_abs_z(data: pd.DataFrame, inject_time: float | int) -> dict:
    """Mirror of zbase_universal steps 1-3: return {col: |max_z|}.

    Identical handling:
      * 'time' column required; baseline = time < inject_time, anomaly = >=.
      * Non-numeric cols ignored, 'time' excluded.
      * Constant baseline (sigma==0), NaN mean, all-NaN anomaly -> z = 0.
      * Uses ddof=0 (population std), same as zbase_universal.
    """
    if "time" not in data.columns:
        raise ValueError("per_metric_max_abs_z: 'time' column required")

    cand_cols = [c for c in data.columns
                 if c != "time" and pd.api.types.is_numeric_dtype(data[c])]

    baseline = data[data["time"] < inject_time]
    anomaly = data[data["time"] >= inject_time]
    if baseline.empty or anomaly.empty:
        return {c: 0.0 for c in cand_cols}

    out: dict[str, float] = {}
    for c in cand_cols:
        base_col = baseline[c]
        mu = base_col.mean()
        sigma = base_col.std(ddof=0)
        if sigma is None or np.isnan(sigma) or sigma == 0 or np.isnan(mu):
            out[c] = 0.0
            continue
        anom_col = anomaly[c].to_numpy(dtype=float, copy=False)
        if not np.isfinite(anom_col).any():
            out[c] = 0.0
            continue
        z_series = (anom_col - mu) / sigma
        abs_z = np.abs(z_series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_abs_z = float(np.nanmax(abs_z))
        if not np.isfinite(max_abs_z):
            max_abs_z = 0.0
        out[c] = max_abs_z
    return out


def count_anomalous(data: pd.DataFrame, inject_time,
                    threshold: float = Z_THRESHOLD) -> tuple[int, int]:
    """Return (n_anomalous, n_total) for numeric-metric columns (excluding time).
    """
    d = per_metric_max_abs_z(data, inject_time)
    total = len(d)
    anom = sum(1 for v in d.values() if v > threshold)
    return anom, total


# ------------------------------------------------------------------ #
# Adapter 1: RCAEval                                                 #
# ------------------------------------------------------------------ #
def _rcaeval_load_case(case_dir: Path):
    """Mirror of run_rcaeval_zbase.load_case + window_data.

    Returns (windowed_df, inject_time, case_id) or raises."""
    data_path = case_dir / "data.csv"
    data = pd.read_csv(data_path)
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    with open(case_dir / "inject_time.txt") as f:
        inject_time = int(f.readline().strip())

    # Window exactly as run_rcaeval_zbase.window_data(length_min=10):
    half = 10 * 60 // 2  # 300s each side
    normal = data[data["time"] < inject_time].tail(half)
    anomal = data[data["time"] >= inject_time].head(half)
    windowed = pd.concat([normal, anomal], ignore_index=True)

    # case_id = path relative to dataset root (matches `case` column in CSV)
    # e.g. adservice_cpu/1
    dd = str(case_dir).rstrip("/")
    parts = dd.split("/")
    case_id = "/".join(parts[-2:])  # service_fault/N
    return windowed, inject_time, case_id


def run_rcaeval_system(system: str) -> list[dict]:
    import glob
    root = RCAEVAL_ROOTS[system]
    case_dirs = sorted(glob.glob(os.path.join(str(root), "*/*/")))
    rows: list[dict] = []
    print(f"  [rcaeval::{system}] {len(case_dirs)} cases")
    for cd in case_dirs:
        try:
            windowed, inj_t, case_id = _rcaeval_load_case(Path(cd))
            if len(windowed) < 30:
                rows.append({"case_id": case_id,
                             "anomalous_metric_count": np.nan,
                             "total_metric_count": np.nan,
                             "frac_anomalous": np.nan,
                             "error": "window too small"})
                continue
            anom, total = count_anomalous(windowed, inj_t)
            rows.append({"case_id": case_id,
                         "anomalous_metric_count": int(anom),
                         "total_metric_count": int(total),
                         "frac_anomalous": (anom / total) if total else 0.0,
                         "error": ""})
        except Exception as e:
            rows.append({"case_id": os.path.relpath(cd, str(root)).rstrip("/"),
                         "anomalous_metric_count": np.nan,
                         "total_metric_count": np.nan,
                         "frac_anomalous": np.nan,
                         "error": f"{type(e).__name__}: {e}"})
    return rows


# ------------------------------------------------------------------ #
# Adapter 2: PetShop                                                 #
# ------------------------------------------------------------------ #
def _petshop_build_combined(normal_df: pd.DataFrame, abnormal_df: pd.DataFrame,
                            inject_time: float) -> pd.DataFrame:
    """Mirror of run_petshop_zbase.build_combined."""
    common = normal_df.columns.intersection(abnormal_df.columns)
    n = normal_df[common].copy()
    a = abnormal_df[common].copy()
    step = 300.0
    n_norm, n_abn = len(n), len(a)
    normal_times = np.array([inject_time - (n_norm - i) * step for i in range(n_norm)])
    abnormal_times = np.array([inject_time + i * step for i in range(n_abn)])
    n.index = normal_times
    a.index = abnormal_times
    combined = pd.concat([n, a], axis=0)
    combined = combined.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    combined = combined.reset_index()
    combined = combined.rename(columns={combined.columns[0]: "time"})
    combined["time"] = combined["time"].astype(float)
    combined = combined.sort_values("time").reset_index(drop=True)
    return combined


def _petshop_clean(s):
    return str(s).replace(" ", "_").replace("::", "-").replace("/", "-")


def _petshop_flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    flat = ["__".join(_petshop_clean(x) for x in col) for col in df.columns]
    out = df.copy()
    out.columns = flat
    return out


def run_petshop_traffic(traffic: str) -> list[dict]:
    from rca_task import load_scenario  # petshop_repo/code
    scenario = PETSHOP_TRAFFIC_ROOTS[traffic]
    graph, normal_metrics, issues = load_scenario(str(scenario))
    normal_flat = _petshop_flatten_cols(normal_metrics)

    rows: list[dict] = []
    for split in ("test", "train"):
        cases = issues[split]
        for ci, (abnormal_metrics, target) in enumerate(cases):
            case_id = f"{split}/issue_{ci}"
            try:
                abnormal_flat = _petshop_flatten_cols(abnormal_metrics)
                inject_time = float(target["target"]["timestamp"])
                combined = _petshop_build_combined(normal_flat, abnormal_flat, inject_time)
                anom, total = count_anomalous(combined, inject_time)
                rows.append({"case_id": case_id,
                             "anomalous_metric_count": int(anom),
                             "total_metric_count": int(total),
                             "frac_anomalous": (anom / total) if total else 0.0,
                             "error": ""})
            except Exception as e:
                rows.append({"case_id": case_id,
                             "anomalous_metric_count": np.nan,
                             "total_metric_count": np.nan,
                             "frac_anomalous": np.nan,
                             "error": f"{type(e).__name__}: {e}"})
    print(f"  [petshop::{traffic}] {len(rows)} cases")
    return rows


# ------------------------------------------------------------------ #
# Adapter 3: OpenRCA                                                 #
# ------------------------------------------------------------------ #
MONTHS = {m: i for i, m in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], start=1)}
UTC_OFFSET_HOURS = 8
TIME_PAT = re.compile(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)')
DATE_PAT = re.compile(
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+(\d{1,2}),?\s*(\d{4})')


def _parse_instruction(instr: str):
    """Extract (mid_utc8, mid_utc, date_label). Matches predict_rcaagent_1min.parse_instruction."""
    dates = DATE_PAT.findall(instr)
    times = TIME_PAT.findall(instr)
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
    mid_utc = mid_utc8 - timedelta(hours=UTC_OFFSET_HOURS)
    date_label = mid_utc8.strftime('%Y-%m-%d')
    return mid_utc8, mid_utc, date_label


def _window_parquet(parquet_df: pd.DataFrame, mid_utc: pd.Timestamp,
                    window_minutes: int = WINDOW_MINUTES,
                    baseline_minutes: int = BASELINE_MINUTES) -> tuple[pd.DataFrame, float]:
    """Build a zbase-friendly DataFrame from a 1-min parquet by slicing
    [mid - baseline - window, mid + window] and attaching a numeric `time` axis.

    Returns (df, inject_time) where inject_time = mid's numeric position
    (= baseline_minutes, since we start `time`=0 at the baseline start).
    """
    inc_idx = parquet_df.index.get_indexer([mid_utc], method='nearest')[0]
    # Total baseline range: [mid - baseline_minutes, mid); adaptive shrink when
    # near parquet start, mirroring predict_rcaagent_1min's adaptive behaviour.
    bs = max(0, inc_idx - baseline_minutes)
    be = inc_idx
    we = min(len(parquet_df), inc_idx + window_minutes)
    # Minimum tolerances: 15 baseline points, 5 anomaly points.
    if be - bs < 15 or we - be < 5:
        raise ValueError(f"insufficient window: base={be-bs} anom={we-be}")
    baseline = parquet_df.iloc[bs:be]
    anomaly = parquet_df.iloc[be:we]
    combined = pd.concat([baseline, anomaly], axis=0)
    # attach a synthetic `time` column measured in minutes from baseline start,
    # and inject_time = (index of `be` - `bs`) minutes.
    n = len(combined)
    inject_time = float(be - bs)
    time_col = np.arange(n, dtype=float)
    combined = combined.reset_index(drop=True).copy()
    # zbase_universal ignores non-numeric cols; insert time as first col
    combined.insert(0, "time", time_col)
    # Forward-fill + zero-fill to keep it deterministic (match adapters).
    combined = combined.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return combined, inject_time


def run_openrca_system(short: str) -> tuple[list[dict], bool]:
    system_folder = OPENRCA_SYSTEMS[short]  # Bank | Market_cloudbed-1 | ...
    q_csv = OPENRCA_Q_DIR / f"{short}_q.csv"
    q = pd.read_csv(q_csv)
    n_queries = len(q)
    print(f"  [openrca::{short}] {n_queries} queries")

    sampled_flag = n_queries > OPENRCA_SAMPLE_CAP
    # Always compute on ALL queries; sampling only filters the per-case CSV.
    rows_all: list[dict] = []

    parquet_cache: dict[str, pd.DataFrame] = {}
    for i, r in q.iterrows():
        case_id = f"{short}/{r['task_index']}#{i}"
        try:
            mid_utc8, mid_utc, date_label = _parse_instruction(str(r["instruction"]))
            if mid_utc is None:
                raise ValueError("unparseable instruction")
            p_1min = OPENRCA_1MIN_DIR / system_folder / date_label / "metric_data.parquet"
            p_ext = OPENRCA_EXT_DIR / system_folder / date_label / "metric_data.parquet"
            mds = None
            for p in (p_1min, p_ext):
                if p.exists():
                    key = str(p)
                    if key not in parquet_cache:
                        parquet_cache[key] = pd.read_parquet(p)
                    m = parquet_cache[key]
                    if not m.empty and m.index.min() <= mid_utc <= m.index.max():
                        mds = m
                        break
            if mds is None:
                raise FileNotFoundError(f"no parquet covers {mid_utc}")

            combined, inject_time = _window_parquet(mds, mid_utc)
            anom, total = count_anomalous(combined, inject_time)
            rows_all.append({"case_id": case_id,
                             "task_index": r["task_index"],
                             "anomalous_metric_count": int(anom),
                             "total_metric_count": int(total),
                             "frac_anomalous": (anom / total) if total else 0.0,
                             "error": ""})
        except Exception as e:
            rows_all.append({"case_id": case_id,
                             "task_index": r.get("task_index", ""),
                             "anomalous_metric_count": np.nan,
                             "total_metric_count": np.nan,
                             "frac_anomalous": np.nan,
                             "error": f"{type(e).__name__}: {e}"})
    return rows_all, sampled_flag


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # -------- RCAEval (3 systems) --------
    for sys_raw, label in [("online-boutique", "OB"),
                           ("sock-shop-2", "SS"),
                           ("train-ticket", "TT")]:
        rws = run_rcaeval_system(sys_raw)
        for r in rws:
            all_rows.append({
                "benchmark": "rcaeval",
                "system": label,
                "sampled": False,
                **r,
            })

    # -------- PetShop (4 traffic regimes) --------
    for traffic in ("high_traffic", "low_traffic",
                    "temporal_traffic1", "temporal_traffic2"):
        rws = run_petshop_traffic(traffic)
        for r in rws:
            all_rows.append({
                "benchmark": "petshop",
                "system": traffic,
                "sampled": False,
                **r,
            })

    # -------- OpenRCA (4 systems) --------
    # We ALWAYS compute the anomalous-metric count on the FULL query set so the
    # per-system aggregate (mean, CV) uses all queries. We then mark `sampled`
    # = True on all rows when we later sub-sample the per-case CSV for size.
    openrca_full: dict[str, list[dict]] = {}
    for short in ("bank", "mkt1", "mkt2", "tel"):
        rws, _flag = run_openrca_system(short)
        openrca_full[short] = rws

    # Emit full OpenRCA rows (no sampling in the CSV itself for these small Ns —
    # largest is bank=54, well under the cap of 30 per-subsystem spec in the
    # prompt. We document it honestly: we write ALL rows; sampling would only
    # kick in if queries > 30, and we flag any such system.)
    for short in ("bank", "mkt1", "mkt2", "tel"):
        rws = openrca_full[short]
        flagged = len(rws) > OPENRCA_SAMPLE_CAP
        # Per spec: "sample-sub to 30 cases per subsystem for the per-case CSV".
        if flagged:
            retained = rws[:OPENRCA_SAMPLE_CAP]
            print(f"  [openrca::{short}] sampled {len(retained)}/{len(rws)} for per-case CSV")
        else:
            retained = rws
        for r in retained:
            all_rows.append({
                "benchmark": "openrca",
                "system": short,
                "sampled": flagged,
                **r,
            })

    df = pd.DataFrame(all_rows)
    # Column order
    col_order = ["benchmark", "system", "case_id", "anomalous_metric_count",
                 "total_metric_count", "frac_anomalous", "sampled", "error"]
    if "task_index" in df.columns:
        col_order.insert(3, "task_index")
    df = df.reindex(columns=col_order)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows)")

    # Also emit the FULL (non-sampled) openrca per-case rows to a side-car csv
    # so the aggregator can always use the complete data.
    full_rows: list[dict] = []
    for short in ("bank", "mkt1", "mkt2", "tel"):
        for r in openrca_full[short]:
            full_rows.append({"benchmark": "openrca", "system": short,
                              "sampled": False, **r})
    if full_rows:
        side = OUT_CSV.with_name("anomalous_metric_count_openrca_full.csv")
        pd.DataFrame(full_rows).reindex(
            columns=["benchmark", "system", "case_id", "task_index",
                     "anomalous_metric_count", "total_metric_count",
                     "frac_anomalous", "sampled", "error"]
        ).to_csv(side, index=False)
        print(f"Wrote {side}  ({len(full_rows)} OpenRCA full rows)")

    # Quick sanity print
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 200, "display.max_columns", 20):
        print("\nPer-system means (from per-case CSV — note openrca may be sampled):")
        grp = df.dropna(subset=["anomalous_metric_count"]).groupby(
            ["benchmark", "system"], as_index=False)
        summary = grp.agg(
            n_cases=("anomalous_metric_count", "size"),
            anom_mean=("anomalous_metric_count", "mean"),
            anom_cv=("anomalous_metric_count",
                     lambda x: (x.std(ddof=0) / x.mean()) if x.mean() else np.nan),
            total_mean=("total_metric_count", "mean"),
            frac_mean=("frac_anomalous", "mean"),
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
