#!/usr/bin/env python3
"""Diversity-balanced smoke test: zbase_universal on 5 DIFFERENT fault types.

The prior smoke script (smoke_zbase_ob.py) picked the first 5 cases via
sorted glob, which all landed in `adservice_cpu/` — biased single-fault-type
sample. This script hardcodes 5 cases from 5 distinct (service, fault_mode)
combinations to get an unbiased plausibility check of Z-base.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from predict_zbase_universal import zbase_universal


DATASET_ROOT = "$HOME/rcaeval_repo/data/online-boutique"
SLI_HINT = "frontend_latency"

# 5 cases spanning 5 different services AND 5 different fault modes.
CASE_DIRS = [
    os.path.join(DATASET_ROOT, "adservice_cpu", "1"),
    os.path.join(DATASET_ROOT, "cartservice_delay", "1"),
    os.path.join(DATASET_ROOT, "checkoutservice_mem", "1"),
    os.path.join(DATASET_ROOT, "currencyservice_loss", "1"),
    os.path.join(DATASET_ROOT, "productcatalogservice_disk", "1"),
]
N_CASES = len(CASE_DIRS)


def load_case(data_dir):
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
        raise ValueError(f"cannot parse service/fault from {service_fault!r}")
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


def main():
    case_dirs = [cd if cd.endswith("/") else cd + "/" for cd in CASE_DIRS]
    print(f"Running {N_CASES} diverse cases:")
    for cd in case_dirs:
        print(f"  {os.path.relpath(cd, DATASET_ROOT)}")
    print()

    results = []
    for cd in case_dirs:
        case_id = os.path.relpath(cd, DATASET_ROOT)
        try:
            data, inj_t, true_svc, fault = load_case(cd)
        except Exception as e:
            print(f"[LOAD-FAIL] {case_id}: {e}")
            results.append({"case": case_id, "error": f"load: {e}"})
            continue

        try:
            windowed = window_data(data, inj_t, length_min=10)
            if len(windowed) < 30:
                raise RuntimeError("window too small")

            sli = SLI_HINT
            if f"{true_svc}_latency" in windowed.columns:
                sli = f"{true_svc}_latency"

            t0 = time.perf_counter()
            out = zbase_universal(
                windowed, inj_t,
                dataset="online-boutique",
                anomalies=None, dk_select_useful=False,
                sli=sli, verbose=False,
                n_iter=len(windowed.columns) - 1,
            )
            dt = time.perf_counter() - t0

            ranks = out.get("ranks", [])
        except Exception as e:
            print(f"[RUN-FAIL] {case_id}: {e}")
            results.append({"case": case_id, "error": f"run: {e}"})
            continue

        top1_metric = ranks[0] if ranks else None
        top1_service = extract_service(top1_metric) if top1_metric else None
        a1 = acc_at_k(ranks, true_svc, 1)

        results.append({
            "case": case_id,
            "true_service": true_svc,
            "fault": fault,
            "top1_metric": top1_metric,
            "top1_service": top1_service,
            "acc@1": a1,
            "wall_s": dt,
            "n_ranks": len(ranks),
        })

        print(f"[OK] {case_id:40s}  true={true_svc:25s}  "
              f"top1={top1_service or '<none>':25s}  "
              f"acc@1={a1:.1f}  t={dt:.2f}s")

    print("\n=== SUMMARY ===")
    ok_rows = [r for r in results if "error" not in r]
    fail_rows = [r for r in results if "error" in r]

    c1 = (len(ok_rows) == N_CASES) and all(
        r["acc@1"] is not None and not np.isnan(r["acc@1"]) for r in ok_rows
    )
    print(f"C1 (all {N_CASES} non-NaN acc@1): {'PASS' if c1 else 'FAIL'} "
          f"({len(ok_rows)}/{N_CASES} ok, {len(fail_rows)} failed)")

    if ok_rows:
        top1s = [r["top1_service"] for r in ok_rows]
        c2 = len(set(top1s)) > 1
        print(f"C2 (top-1 not all identical): {'PASS' if c2 else 'FAIL'}  top1s={top1s}")
    else:
        print("C2: FAIL (no ok rows)")

    if ok_rows:
        times = [r["wall_s"] for r in ok_rows]
        c3 = all(t < 60.0 for t in times)
        print(f"C3 (<60s per case): {'PASS' if c3 else 'FAIL'}  "
              f"times={[f'{t:.2f}' for t in times]}s")
    else:
        print("C3: FAIL (no ok rows)")

    if ok_rows:
        accs = [r["acc@1"] for r in ok_rows]
        mean_acc = float(np.mean(accs))
        c4 = 0.20 <= mean_acc <= 0.60
        print(f"C4 (mean acc@1 in [0.20, 0.60]): {'PASS' if c4 else 'FAIL'}  "
              f"accs={accs}  mean={mean_acc:.3f}")
    else:
        print("C4: FAIL (no ok rows)")

    if fail_rows:
        print("\nFailures:")
        for r in fail_rows:
            print(f"  {r['case']}: {r['error']}")


if __name__ == "__main__":
    main()
