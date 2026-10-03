"""Shared helpers for RCAEval baseline reproduction scripts.

Extracted from `14_baro_official_repro.py` so that `15_baselines_other.py`
can reuse the same dataset map, preprocessing, and Acc@k math.
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict
from os.path import basename, dirname, join
from typing import Any

import numpy as np
import pandas as pd

DATASET_MAP: dict[str, str] = {
    "re1-ob": "$HOME/rcaeval_repo/data/online-boutique",
    "re1-ss": "$HOME/rcaeval_repo/data/sock-shop-2",
    "re1-tt": "$HOME/rcaeval_repo/data/train-ticket",
    "re2-ob": "$HOME/data/rcaeval/RE2/RE2-OB",
    "re2-ss": "$HOME/data/rcaeval/RE2/RE2-SS",
    "re2-tt": "$HOME/data/rcaeval/RE2/RE2-TT",
}


def sli_for_path(data_path: str) -> str | None:
    """Return the SLI metric column that RCAEval's main.py assumes for each system."""
    if "RE2-OB" in data_path or "RE2-SS" in data_path:
        return "frontend_latency"
    if "RE2-TT" in data_path:
        return "ts-ui-dashboard_latency"
    if "online-boutique" in data_path:
        return "frontend_latency"
    if "sock-shop" in data_path:
        return "front-end_cpu"
    if "train-ticket" in data_path:
        return "ts-ui-dashboard_latency"
    return None


def load_case(data_path: str, length_min: int = 20) -> tuple[pd.DataFrame, int, str, str, str | None]:
    """Load and preprocess a single case following main.py's conventions.

    Returns:
        data: the truncated+cleaned dataframe (normal + anomal concatenated)
        inject_time: fault injection timestamp
        service: ground-truth root-cause service
        fault: fault type (cpu/mem/delay/disk/loss)
        sli: canonical SLI column name for the system
    """
    data_dir = dirname(data_path)
    service, fault = basename(dirname(dirname(data_path))).split("_")

    data = pd.read_csv(data_path)
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill().fillna(0)
    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip())
    win = length_min * 60 // 2
    normal_df = data[data["time"] < inject_time].tail(win)
    anomal_df = data[data["time"] >= inject_time].head(win)
    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # rename latency-90 -> latency so RCAEval SLI lookups work
    data = data.rename(
        columns={c: c.replace("_latency-90", "_latency") for c in data.columns if c.endswith("_latency-90")}
    )

    sli = sli_for_path(data_path)
    # RE1 full-metric-name variants: resolve canonical SLI to an existing column.
    # sock-shop-2 RE1 data.csv has no short "front-end_cpu"; use the container-CPU
    # analog of RCAEval main.py's own front-end CPU choice. Documented for followup.
    if sli is not None and sli not in data.columns:
        fallbacks = {
            "front-end_cpu": "front-end_container-cpu-usage-seconds-total",
        }
        alt = fallbacks.get(sli)
        if alt is not None and alt in data.columns:
            sli = alt
        else:
            cand = [c for c in data.columns if c.split("_")[0] == sli.split("_")[0]]
            if cand:
                sli = cand[0]
    return data, inject_time, service, fault, sli


def list_case_paths(dataset: str) -> list[str]:
    """List data.csv paths (or simple_metrics.csv for RE2) for a dataset."""
    root = DATASET_MAP[dataset]
    paths = sorted(glob.glob(os.path.join(root, "**/data.csv"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "**/simple_metrics.csv"), recursive=True))
    return paths


def service_from_col(col: str) -> str:
    """Match main.py line 380: ``x.split("_")[0].replace("-db", "")``.

    The ``-db`` strip is critical on RE1-SS where ``carts-db`` / ``orders-db``
    metrics must be folded back into their application services.
    """
    return col.split("_")[0].replace("-db", "")


def acc_at_k(ranks: list[str], truth: str, k: int) -> int:
    """Acc@k: 1 if truth service is in top-k services of ranks (dedup by service order)."""
    seen: list[str] = []
    for col in ranks:
        s = service_from_col(col)
        if s not in seen:
            seen.append(s)
        if len(seen) == k:
            break
    return int(truth in seen)


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute Acc@1/3/5 and Avg@5 overall and per fault type."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        buckets["_all"].append(r)
        buckets[r["fault"]].append(r)
    out: dict[str, dict[str, float]] = {}
    for key, rs in buckets.items():
        accs = []
        for k in range(1, 6):
            hits = [acc_at_k(r["ranks"], r["service"], k) for r in rs]
            accs.append(float(np.mean(hits)) if hits else 0.0)
        out[key] = {
            "n": len(rs),
            "acc@1": accs[0],
            "acc@3": accs[2],
            "acc@5": accs[4],
            "avg@5": float(np.mean(accs)),
        }
    return out


def print_summary(dataset: str, length_min: int, summary: dict[str, dict[str, float]], n: int, wall_s: float) -> None:
    print(f"\n=== {dataset} (length={length_min} min) | n={n} | {wall_s:.1f}s ===")
    print(f"{'fault':>10s} {'n':>4s} {'acc@1':>7s} {'acc@3':>7s} {'acc@5':>7s} {'avg@5':>7s}")
    for key in ["_all", "cpu", "mem", "disk", "delay", "loss"]:
        if key not in summary:
            continue
        s = summary[key]
        print(
            f"{key:>10s} {int(s['n']):>4d} "
            f"{s['acc@1']:>7.3f} {s['acc@3']:>7.3f} {s['acc@5']:>7.3f} {s['avg@5']:>7.3f}"
        )
