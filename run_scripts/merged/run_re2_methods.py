"""RCAEval-official methods on RE2 that the released main.py cannot run end-to-end, plus CIRCA.

Methods are loaded by file path from rcaeval_repo/RCAEval/e2e/*.py (RCAEval.e2e.__init__ only
registers them under py3.10/3.12; our env is py3.11 - same approach as the A/B paper runs).

  circa      metric table, main.py preprocessing (see load_metric)
  tracerca   spans (traces.csv), inject_time in microseconds, as in e2e/tracerca.py
  microrank  spans, same
  mmbaro     {"metric", "logs", "logts", "traces", "tracets_lat", "tracets_err", "cluster_info"} as in
             docs/multi-source-rca-demo.ipynb; metric prepared like main.py; logts/tracets windowed
             like main.py's multi-source (torai) branch (length*4//2 rows each side).
             --mm-dataset-name: the string passed as `dataset`. mmnsigma only uses the trace series when
             dataset is "mm-ob"/"mm-tt"; main.py would pass "re2-ob"/"re2-tt".

Case enumeration: <root>/<service>_<fault>/<n> with numeric n (skips the multi-source-data demo folder).
Scoring: main.py Evaluator rule - service = rank.split("_")[0].replace("-db", ""), dedup in order.

Usage: python run_re2_methods.py --method circa --dataset re2-tt --length 20 --out X.json
"""
import argparse
import datetime
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

THIS_FILE = os.path.abspath(__file__)
REPO = "$HOME/rcaeval_repo"
sys.path.insert(0, REPO)
ROOTS = {"re2-ob": f"{REPO}/data/RE2/RE2-OB", "re2-ss": f"{REPO}/data/RE2/RE2-SS", "re2-tt": f"{REPO}/data/RE2/RE2-TT"}
METHOD_FILES = {"circa": "circa.py", "tracerca": "tracerca.py", "microrank": "microrank.py", "mmbaro": "baro.py"}


def load_fn(method):
    spec = importlib.util.spec_from_file_location(f"{method}_mod", f"{REPO}/RCAEval/e2e/{METHOD_FILES[method]}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, method)


def list_cases(ds):
    out = []
    for d in sorted(glob.glob(os.path.join(ROOTS[ds], "*", "*"))):
        if os.path.isdir(d) and os.path.basename(d).isdigit():
            out.append(d)
    return out


def sli_for(ds, service, data):
    # main.py: RE2-TT -> ts-ui-dashboard_latency; RE2-OB/SS -> frontend_latency; prefer <service>_latency
    sli = "ts-ui-dashboard_latency" if ds == "re2-tt" else "frontend_latency"
    if f"{service}_latency" in data:
        sli = f"{service}_latency"
    elif ds != "re2-tt" and "frontend_1" in data:
        sli = "frontend_1"
    return sli


def load_metric(case_dir, inject_time, length, metric_file):
    """main.py process(): drop _latency-50, inf->nan, ffill, fillna(0), length/2 min each side, rename latency-90."""
    data = pd.read_csv(os.path.join(case_dir, metric_file))
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    half = length * 60 // 2
    data = pd.concat([data[data["time"] < inject_time].tail(half), data[data["time"] >= inject_time].head(half)],
                     ignore_index=True)
    data = data.rename(columns={c: c.replace("_latency-90", "_latency") for c in data.columns if c.endswith("_latency-90")})
    return data


def window_ts(df, inject_time, rows):
    if df.shape[0] == 0:
        return df
    return pd.concat([df[df["time"] < inject_time].tail(rows), df[df["time"] >= inject_time].head(rows)], ignore_index=True)


def service_ranks(ranks):
    seen = []
    for r in ranks:
        s = str(r).split("_")[0].replace("-db", "")
        if s not in seen:
            seen.append(s)
    return seen


def run_case(job):
    a, case_dir = job
    fn = load_fn(a.method)
    service, fault = os.path.basename(os.path.dirname(case_dir)).split("_")
    rec = {"case": "/".join(case_dir.split("/")[-2:]), "service": service, "fault": fault}
    t0 = time.time()
    try:
        with open(os.path.join(case_dir, "inject_time.txt")) as f:
            inject_time = int(f.readlines()[0].strip())
        if a.method == "circa":
            data = load_metric(case_dir, inject_time, a.length, a.metric_file)
            out = fn(data, inject_time, dataset=a.dataset, anomalies=None, dk_select_useful=False,
                     sli=sli_for(a.dataset, service, data), verbose=False, n_iter=len(data.columns) - 1)
        elif a.method in ("tracerca", "microrank"):
            spans = pd.read_csv(os.path.join(case_dir, "traces.csv"))
            out = fn(spans, inject_time * 1_000_000, dataset=a.dataset, anomalies=None, dk_select_useful=False,
                     sli=None, verbose=False, n_iter=None)
        elif a.method == "mmbaro":
            metric = load_metric(case_dir, inject_time, a.length, a.metric_file)
            rows = a.length * 4 // 2

            def opt(name):
                p = os.path.join(case_dir, name)
                return window_ts(pd.read_csv(p), inject_time, rows) if os.path.exists(p) else pd.DataFrame()

            mm = {"metric": metric, "logs": None, "logts": opt("logts.csv"), "traces": None,
                  "tracets_lat": opt("tracets_lat.csv"), "tracets_err": opt("tracets_err.csv"), "cluster_info": None}
            out = fn(mm, inject_time, dataset=a.mm_dataset_name or a.dataset)
        ranks = [str(r) for r in (out.get("ranks") or [])]
        rec["error"] = None
    except Exception as e:
        ranks = []
        rec["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-2000:]
    rec["ranks"] = ranks[:50]
    rec["n_ranks"] = len(ranks)
    srv = service_ranks(ranks)
    rec["service_ranks"] = srv[:10]
    for k in (1, 3, 5):
        rec[f"acc{k}"] = int(service in srv[:k])
    rec["avg5"] = float(np.mean([service in srv[:k] for k in range(1, 6)]))
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHOD_FILES))
    ap.add_argument("--dataset", required=True, choices=sorted(ROOTS))
    ap.add_argument("--length", type=int, default=20, help="minutes; main.py default is 10")
    ap.add_argument("--metric-file", default="simple_metrics.csv")
    ap.add_argument("--mm-dataset-name", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cases = list_cases(a.dataset)
    if a.method in ("tracerca", "microrank"):
        cases = [c for c in cases if os.path.exists(os.path.join(c, "traces.csv"))]
    if a.limit:
        cases = cases[: a.limit]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(run_case, [(a, c) for c in cases], chunksize=1)

    n = len(rows)
    src = {f: hashlib.sha256(open(f"{REPO}/RCAEval/e2e/{f}", "rb").read()).hexdigest()[:12] for f in set(METHOD_FILES.values())}
    src["run_re2_methods.py"] = hashlib.sha256(open(THIS_FILE, "rb").read()).hexdigest()[:12]
    res = {
        "config": {
            "method": a.method, "dataset": a.dataset,
            "input_file": "traces.csv" if a.method in ("tracerca", "microrank") else
                          (a.metric_file + (" + logts.csv + tracets_{lat,err}.csv" if a.method == "mmbaro" else "")),
            "length_min": a.length, "mm_dataset_name": a.mm_dataset_name,
            "code_commit": "rcaeval_repo@497505b", "code_sha256_12": src,
            "env": sys.executable, "python": sys.version.split()[0], "argv": sys.argv,
            "run_started_utc": started, "limit": a.limit,
        },
        "n": n,
        "coverage": sum(r["n_ranks"] > 0 for r in rows),
        "n_errors": sum(r["error"] is not None for r in rows),
        **{f"acc{k}": round(sum(r[f"acc{k}"] for r in rows) / n, 4) for k in (1, 3, 5)},
        "avg5": round(sum(r["avg5"] for r in rows) / n, 4),
        "wall_s": round(time.time() - t0, 1),
        "cases": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"method": a.method, "dataset": a.dataset, "length": a.length, "mm": a.mm_dataset_name,
                      "metric_file": a.metric_file, **{k: v for k, v in res.items() if k not in ("cases", "config")}}),
          flush=True)


if __name__ == "__main__":
    main()
