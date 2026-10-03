#!/usr/bin/env python3
"""Run Alert-count baseline on RCAEval's 3 subsystems. Direct clone of
run_rcaeval_zbase.py with predictor swapped for alertcount_universal.

method column is written as "alertcount".
"""
import argparse
import functools
import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Alert-count: we pass RCAEval's own extract_service via functools.partial
# so the predictor's per-service aggregation matches this benchmark's schema.
from predict_alertcount import alertcount_universal


DATASET_ROOTS = {
    "online-boutique": "$HOME/rcaeval_repo/data/online-boutique",
    "sock-shop-2":     "$HOME/rcaeval_repo/data/sock-shop-2",
    "train-ticket":    "$HOME/rcaeval_repo/data/train-ticket",
}


DATA_FILE = "data.csv"


def load_case(data_dir):
    """Return (data_df, inject_time, true_service, fault_type)."""
    data_path = os.path.join(data_dir, DATA_FILE)
    if not os.path.exists(data_path):
        data_path = os.path.join(data_dir, "data.csv")
    data = pd.read_csv(data_path)
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    with open(os.path.join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readline().strip())

    dd = data_dir.rstrip("/")
    service_fault = os.path.basename(os.path.dirname(dd))
    parts = service_fault.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"cannot parse service/fault from {service_fault!r} "
                         f"(data_dir={data_dir})")
    service, fault = parts[0], parts[1]
    return data, inject_time, service, fault


def window_data(data, inject_time, length_min=10):
    half = length_min * 60 // 2
    normal = data[data["time"] < inject_time].tail(half)
    anomal = data[data["time"] >= inject_time].head(half)
    return pd.concat([normal, anomal], ignore_index=True)


def extract_service(metric_col):
    for suffix in ["_cpu", "_mem", "_latency", "_load", "_error",
                   "_duration", "_rps", "_lat_90", "_latency-90",
                   "_loss", "_disk", "_delay", "_workload"]:
        if metric_col.endswith(suffix):
            return metric_col[:-len(suffix)]
    if "_" in metric_col:
        head = metric_col.split("_", 1)[0]
        if head and head[0].isdigit():
            return head
        for subsfx in ["-db", "-rb"]:
            if head.endswith(subsfx):
                return head[:-len(subsfx)]
        return head
    return metric_col


def acc_at_k(ranks, true_service, k):
    services_seen = set()
    for m in ranks:
        s = extract_service(m)
        services_seen.add(s)
        if len(services_seen) >= k:
            break
    return 1.0 if true_service in services_seen else 0.0


def run_one(method_name, method_fn, data_dir, tdelta, sli_hint):
    try:
        data, inj_t, true_svc, fault = load_case(data_dir)
    except Exception as e:
        return {"error": f"load: {e}"}

    eff_inj = inj_t + tdelta
    try:
        windowed = window_data(data, eff_inj, length_min=10)
        if len(windowed) < 30:
            return {"error": "window too small"}

        sli = sli_hint
        if f"{true_svc}_latency" in windowed.columns:
            sli = f"{true_svc}_latency"

        out = method_fn(
            windowed, eff_inj,
            extract_service=extract_service,
            threshold=3.0,
            dataset="ob" if "ob" in method_name else "online-boutique",
            anomalies=None, dk_select_useful=False,
            sli=sli, verbose=False,
            n_iter=len(windowed.columns) - 1,
        )
        ranks = out.get("ranks", [])
    except Exception as e:
        return {"error": f"run: {e}"}

    row = {
        "method": method_name, "tdelta": tdelta,
        "case": os.path.relpath(data_dir, DATASET_ROOTS[args_dataset]),
        "true_service": true_svc, "fault": fault,
        "top1": extract_service(ranks[0]) if ranks else None,
        "acc@1": acc_at_k(ranks, true_svc, 1),
        "acc@3": acc_at_k(ranks, true_svc, 3),
        "acc@5": acc_at_k(ranks, true_svc, 5),
        "acc@10": acc_at_k(ranks, true_svc, 10),
        "n_ranks": len(ranks),
    }
    return row


args_dataset = None


def main():
    global args_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_ROOTS.keys()))
    ap.add_argument("--methods", nargs="+", default=["alertcount"])
    ap.add_argument("--tdeltas", nargs="+", type=int,
                    default=[0, 30, 60, 120, 300, 600])
    ap.add_argument("--datafile", default="data.csv",
                    help="Per-case CSV filename; falls back to data.csv if absent.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N cases (smoke test)")
    args = ap.parse_args()
    global DATA_FILE
    DATA_FILE = args.datafile
    args_dataset = args.dataset
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fn_map = {"alertcount": alertcount_universal}
    sli_hint = "frontend_latency"
    if args.dataset == "sock-shop-2":
        sli_hint = "front-end_cpu"
    elif args.dataset == "train-ticket":
        sli_hint = "ts-ui-dashboard_latency"

    case_dirs = sorted(glob.glob(os.path.join(DATASET_ROOTS[args.dataset], "*/*/")))
    if args.limit:
        case_dirs = case_dirs[:args.limit]
    print(f"Found {len(case_dirs)} cases in {args.dataset}")

    rows = []
    for mi, method in enumerate(args.methods):
        fn = fn_map[method]
        for ti, td in enumerate(args.tdeltas):
            n_done = 0
            for cd in case_dirs:
                r = run_one(method, fn, cd, td, sli_hint)
                if "error" in r:
                    rows.append({"method": method, "tdelta": td,
                                 "case": os.path.relpath(cd, DATASET_ROOTS[args.dataset]),
                                 "error": r["error"]})
                else:
                    rows.append(r)
                    n_done += 1
            print(f"  {method} tdelta={td}s: {n_done}/{len(case_dirs)} ok")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / f"rcaeval_alertcount_{args.dataset}_results.csv", index=False)

    if "error" in df.columns:
        err_df = df[df["error"].notna()] if df["error"].isna().any() else df[df["error"].astype(str) != ""]
        if not err_df.empty:
            print("\nFirst few errors:")
            print(err_df.head(3).to_string(index=False))

    if "acc@1" not in df.columns:
        print("No successful runs — nothing to summarize.")
        return
    ok = df[df["acc@1"].notna()]
    summ = ok.groupby(["method", "tdelta"]).agg(
        n=("acc@1", "size"),
        acc1=("acc@1", "mean"),
        acc3=("acc@3", "mean"),
        acc5=("acc@5", "mean"),
        acc10=("acc@10", "mean"),
    ).reset_index()
    summ.to_csv(args.out_dir / f"rcaeval_alertcount_{args.dataset}_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
