#!/usr/bin/env python3
"""Run Z-base (zbase_universal) on PetShop via its own load_scenario().
Direct clone of run_petshop.py with baro swapped for zbase_universal.

Output: petshop_zbase_<traffic>_results.csv
  Columns: method, tdelta, case, true_service, fault, top1,
           acc@1, acc@3, acc@5, acc@10, n_ranks, error

method column is written as "zbase" (NOT "baro").
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PETSHOP_ROOT = Path("$HOME/petshop_repo")
sys.path.insert(0, str(PETSHOP_ROOT / "code"))
from rca_task import load_scenario  # noqa: E402

# Aliased import: zbase_universal is a drop-in replacement for baro() here.
from predict_zbase_universal import zbase_universal as baro  # noqa: E402
from RCAEval.e2e.circa import circa  # noqa: E402


TRAFFIC_ROOTS = {
    "high_traffic":      PETSHOP_ROOT / "dataset" / "high_traffic",
    "low_traffic":       PETSHOP_ROOT / "dataset" / "low_traffic",
    "temporal_traffic1": PETSHOP_ROOT / "dataset" / "temporal_traffic1",
    "temporal_traffic2": PETSHOP_ROOT / "dataset" / "temporal_traffic2",
}


def clean(s):
    return str(s).replace(" ", "_").replace("::", "-").replace("/", "-")


def flatten_cols(df):
    flat = ["__".join(clean(x) for x in col) for col in df.columns]
    out = df.copy()
    out.columns = flat
    return out


def extract_service(metric_col):
    return metric_col.split("__", 1)[0] if "__" in metric_col else metric_col


def acc_at_k(ranks, true_service, k):
    services_seen = set()
    for m in ranks:
        s = extract_service(m)
        services_seen.add(s)
        if len(services_seen) >= k:
            break
    return 1.0 if true_service in services_seen else 0.0


def build_combined(normal_df, abnormal_df, inject_time):
    common = normal_df.columns.intersection(abnormal_df.columns)
    n = normal_df[common].copy()
    a = abnormal_df[common].copy()

    step = 300.0  # 5 min
    n_norm = len(n)
    n_abn = len(a)
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


def pick_sli(target, columns):
    sli = "__".join([
        clean(target["target"]["node"]),
        target["target"]["metric"],
        target["target"].get("agg", "Average"),
    ])
    if sli in columns:
        return sli
    prefix = clean(target["target"]["node"]) + "__" + target["target"]["metric"]
    cands = [c for c in columns if c.startswith(prefix)]
    return cands[0] if cands else columns[-1]


def run_one(method_name, method_fn, abnormal_df, target, normal_df):
    inject_time = float(target["target"]["timestamp"])
    try:
        combined = build_combined(normal_df, abnormal_df, inject_time)
    except Exception as e:
        return {"error": f"combine: {e}"}

    true_svc = clean(target["root_cause"]["node"])
    fault = ""

    sli = pick_sli(target, list(combined.columns))

    try:
        out = method_fn(
            combined, inject_time,
            dataset="petshop",
            anomalies=None, dk_select_useful=False,
            sli=sli, verbose=False,
            n_iter=len(combined.columns) - 1,
        )
        ranks = out.get("ranks", [])
    except Exception as e:
        return {"error": f"run: {e}"}

    top1 = extract_service(ranks[0]) if ranks else None
    row = {
        "method": method_name,
        "tdelta": 0,
        "true_service": true_svc,
        "fault": fault,
        "top1": top1,
        "acc@1": acc_at_k(ranks, true_svc, 1),
        "acc@3": acc_at_k(ranks, true_svc, 3),
        "acc@5": acc_at_k(ranks, true_svc, 5),
        "acc@10": acc_at_k(ranks, true_svc, 10),
        "n_ranks": len(ranks),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", required=True,
                    choices=list(TRAFFIC_ROOTS.keys()))
    # Default method is zbase (aliased above). "circa" is still reachable.
    ap.add_argument("--methods", nargs="+", default=["zbase"],
                    choices=["zbase", "circa"])
    ap.add_argument("--splits", nargs="+", default=["test", "train"],
                    choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N cases per split (smoke test)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenario = TRAFFIC_ROOTS[args.traffic]
    print(f"Loading scenario: {scenario}")
    graph, normal_metrics, issues = load_scenario(str(scenario))
    normal_flat = flatten_cols(normal_metrics)
    print(f"  normal_metrics: {normal_flat.shape}, graph nodes: {graph.number_of_nodes()}")
    for split in args.splits:
        print(f"  {split}: {len(issues[split])} cases")

    fn_map = {"zbase": baro, "circa": circa}

    rows = []
    for split in args.splits:
        cases = issues[split]
        if args.limit:
            cases = cases[: args.limit]
        for mi, method_name in enumerate(args.methods):
            fn = fn_map[method_name]
            n_done = 0
            for ci, (abnormal_metrics, target) in enumerate(cases):
                abnormal_flat = flatten_cols(abnormal_metrics)
                case_id = f"{split}/issue_{ci}"
                r = run_one(method_name, fn, abnormal_flat, target, normal_flat)
                base = {
                    "method": method_name, "tdelta": 0,
                    "case": case_id,
                    "true_service": clean(target["root_cause"]["node"]),
                    "fault": "",
                }
                if "error" in r:
                    rows.append({**base, "error": r["error"]})
                else:
                    r["case"] = case_id
                    rows.append(r)
                    n_done += 1
            print(f"  {method_name} {split}: {n_done}/{len(cases)} ok")

    df = pd.DataFrame(rows)
    # Output filename uses `zbase` label.
    out_csv = args.out_dir / f"petshop_zbase_{args.traffic}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)")

    if "acc@1" in df.columns:
        ok = df[df["acc@1"].notna()]
        if not ok.empty:
            summ = ok.groupby(["method", "tdelta"]).agg(
                n=("acc@1", "size"),
                acc1=("acc@1", "mean"),
                acc3=("acc@3", "mean"),
                acc5=("acc@5", "mean"),
                acc10=("acc@10", "mean"),
            ).reset_index()
            summ_csv = args.out_dir / f"petshop_zbase_{args.traffic}_summary.csv"
            summ.to_csv(summ_csv, index=False)
            print(summ.to_string(index=False))

    if "error" in df.columns:
        err_df = df[df["error"].notna() & (df["error"].astype(str) != "")]
        if not err_df.empty:
            print("\nFirst errors:")
            print(err_df.head(3)[["method", "case", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
