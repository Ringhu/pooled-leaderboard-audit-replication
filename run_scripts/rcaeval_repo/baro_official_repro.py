"""Official BARO reproduction on RE1-{OB,SS,TT}.

Follows main.py exactly: length=20 min (600 timesteps at 2s sampling),
uses data.csv, preprocesses with the RCAEval package's preprocess(),
BARO fits RobustScaler on normal, transforms anomal, takes max z-score per col.
"""
import argparse, glob, os, sys, json, re
from datetime import datetime
from os.path import basename, dirname, join
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "$HOME/rcaeval_repo")
# Import baro DIRECTLY (bypass RCAEval.e2e.__init__ which needs causallearn)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "baro_mod", "$HOME/rcaeval_repo/RCAEval/e2e/baro.py"
)
baro_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baro_mod)
baro = baro_mod.baro


DATASET_MAP = {
    "re1-ob": "$HOME/rcaeval_repo/data/online-boutique",
    "re1-ss": "$HOME/rcaeval_repo/data/sock-shop-2",
    "re1-tt": "$HOME/rcaeval_repo/data/train-ticket",
    "re2-ob": "$HOME/data/rcaeval/RE2/RE2-OB",
    "re2-ss": "$HOME/data/rcaeval/RE2/RE2-SS",
    "re2-tt": "$HOME/data/rcaeval/RE2/RE2-TT",
}


def process_case(data_path, dataset, length_min=20):
    data_dir = dirname(data_path)
    service, fault = basename(dirname(dirname(data_path))).split("_")

    data = pd.read_csv(data_path)
    # match main.py: drop _latency-50, ffill, fillna(0), truncate to window
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill().fillna(0)
    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip())
    win = length_min * 60 // 2
    normal_df = data[data["time"] < inject_time].tail(win)
    anomal_df = data[data["time"] >= inject_time].head(win)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # rename latency-90 to latency
    data = data.rename(columns={c: c.replace("_latency-90", "_latency")
                                for c in data.columns if c.endswith("_latency-90")})

    # sli (unused by baro but passed by main.py)
    sli = None
    if "RE2-OB" in data_path or "RE2-SS" in data_path:
        sli = "frontend_latency"
    elif "RE2-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
    elif "online-boutique" in data_path:
        sli = "frontend_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
    elif "train-ticket" in data_path:
        sli = "ts-ui-dashboard_latency"

    out = baro(data, inject_time=inject_time, dataset=dataset, sli=sli,
               dk_select_useful=False, verbose=False)
    return {"service": service, "fault": fault, "ranks": out["ranks"]}


def service_from_col(col):
    """Match main.py line 380: `x.split("_")[0].replace("-db", "")`.

    The -db strip is critical on RE1-SS where `carts-db` / `orders-db` metrics
    must be folded into their application services (`carts`, `orders`).
    """
    return col.split("_")[0].replace("-db", "")


def acc_at_k(ranks, truth, k):
    """Acc@k = 1 if truth service is in top-k services of ranks (dedup by service order)."""
    seen = []
    for col in ranks:
        s = service_from_col(col)
        if s not in seen:
            seen.append(s)
        if len(seen) == k:
            break
    return int(truth in seen)


def eval_dataset(dataset, length_min=20):
    root = DATASET_MAP[dataset]
    paths = sorted(glob.glob(os.path.join(root, "**/data.csv"), recursive=True))
    if not paths:
        # why: RE2 uses simple_metrics.csv — fallback per main.py:149-152
        paths = sorted(glob.glob(os.path.join(root, "**/simple_metrics.csv"), recursive=True))
    results = []
    for p in tqdm(paths, desc=dataset):
        try:
            r = process_case(p, dataset, length_min=length_min)
            results.append(r)
        except Exception as e:
            print(f"FAIL {p}: {e}")
    return results


def summarize(results):
    """Compute Acc@1..Acc@5 and Avg@5 overall + per fault type."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in results:
        buckets["_all"].append(r)
        buckets[r["fault"]].append(r)
    out = {}
    for key, rs in buckets.items():
        accs = []
        for k in range(1, 6):
            hits = [acc_at_k(r["ranks"], r["service"], k) for r in rs]
            accs.append(np.mean(hits) if hits else 0.0)
        out[key] = {
            "n": len(rs),
            "acc@1": accs[0],
            "acc@3": accs[2],
            "acc@5": accs[4],
            "avg@5": float(np.mean(accs)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_MAP))
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    t0 = datetime.now()
    results = eval_dataset(args.dataset, length_min=args.length)
    dt = (datetime.now() - t0).total_seconds()
    summary = summarize(results)

    print(f"\n=== {args.dataset} (length={args.length} min) | n={len(results)} | {dt:.1f}s ===")
    print(f"{'fault':>10s} {'n':>4s} {'acc@1':>7s} {'acc@3':>7s} {'acc@5':>7s} {'avg@5':>7s}")
    for key in ["_all", "cpu", "mem", "disk", "delay", "loss"]:
        if key not in summary:
            continue
        s = summary[key]
        print(f"{key:>10s} {s['n']:>4d} {s['acc@1']:>7.3f} {s['acc@3']:>7.3f} {s['acc@5']:>7.3f} {s['avg@5']:>7.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "cases": results, "wall_s": dt}, f)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
