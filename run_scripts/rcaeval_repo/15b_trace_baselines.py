"""Trace-based RCA baselines on RCAEval RE2 (OB and TT).

RCAEval does NOT ship a method literally named "MicroRCA" (Li Wu et al. NOMS'20).
The trace-based baselines available in ``RCAEval.e2e`` are:

- ``tracerca``  — TraceRCA-style support/confidence ranking on span anomalies.
  Entry: ``RCAEval/e2e/tracerca.py::tracerca(data, inject_time, dataset, **kwargs)``
  Source comment: "refactor from https://github.com/IntelligentDDS/MicroRank"
- ``microrank`` — MicroRank (IntelligentDDS) PageRank over trace-operation graph.
  Entry: ``RCAEval/e2e/microrank.py::microrank(data, inject_time, dataset, **kwargs)``

Both expect ``data`` to be a span DataFrame (columns: ``serviceName``, ``methodName``,
``operationName``, ``startTime`` [microseconds], ``duration``, ``traceID``, ...).
The ``inject_time`` argument must be in microseconds (paper's convention).

We run BOTH methods as separate baselines. Only RE2-OB and RE2-TT are supported
(RE2-SS has no traces per the README). RE1 has no trace CSVs. RE3 is code-level.

Each case directory contains ``traces.csv`` alongside ``simple_metrics.csv`` and
``inject_time.txt`` (seconds; we multiply by 1_000_000 to get microseconds).

CPU-only on 3090. Large trace CSVs (up to ~100MB) get loaded per case, so this
script is I/O-heavy. Use ``--max-cases`` for a smoke test first.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from os.path import basename, dirname, join
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure RCAEval is importable
sys.path.insert(0, "$HOME/rcaeval_repo")

# Reuse Acc@k / summary helpers from the metric baseline script
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from _baseline_utils import acc_at_k, print_summary  # noqa: E402

DATASET_MAP: dict[str, str] = {
    "re2-ob": "$HOME/data/rcaeval/RE2/RE2-OB",
    "re2-tt": "$HOME/data/rcaeval/RE2/RE2-TT",
}


def _load_baseline_from_file(path: str, attr: str):
    """Load a function from a .py file via importlib (bypasses __init__.py)."""
    spec = importlib.util.spec_from_file_location(f"{attr}_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, attr)


def resolve_method(method: str):
    if method == "tracerca":
        return _load_baseline_from_file(
            "$HOME/rcaeval_repo/RCAEval/e2e/tracerca.py", "tracerca"
        )
    if method == "microrank":
        return _load_baseline_from_file(
            "$HOME/rcaeval_repo/RCAEval/e2e/microrank.py", "microrank"
        )
    raise ValueError(f"Unknown trace-based method: {method}")


def list_case_dirs(dataset: str) -> list[str]:
    """Return a list of case directories (each containing traces.csv + inject_time.txt)."""
    root = DATASET_MAP[dataset]
    dirs = sorted(glob.glob(os.path.join(root, "*", "*")))
    # Filter to directories that have both traces.csv and inject_time.txt
    out = []
    for d in dirs:
        if os.path.isdir(d) and os.path.exists(join(d, "traces.csv")) and os.path.exists(
            join(d, "inject_time.txt")
        ):
            out.append(d)
    return out


def load_trace_case(case_dir: str, length_min: int = 20) -> tuple[pd.DataFrame, int, str, str]:
    """Load the span DataFrame + inject_time (microseconds) for one case.

    The RCAEval baselines partition spans by ``startTime + duration < inject_time``,
    so no manual windowing is required. We do NOT truncate the span set — the
    method handles its own partitioning. ``length_min`` is accepted for API
    symmetry with the metric script but currently unused (spans already cover
    a natural window around the injection).

    Returns:
        span_df, inject_time_us, service, fault
    """
    case_name = basename(dirname(case_dir))  # e.g. "checkoutservice_cpu"
    service, fault = case_name.split("_", 1)

    span_df = pd.read_csv(join(case_dir, "traces.csv"))
    with open(join(case_dir, "inject_time.txt")) as f:
        inject_time_s = int(f.readlines()[0].strip())
    # RCAEval trace methods compare startTime (us) vs inject_time -> need us.
    inject_time_us = inject_time_s * 1_000_000
    return span_df, inject_time_us, service, fault


def service_from_op(op: str) -> str:
    """Ranks from tracerca/microrank are operation names like ``svc_method``.

    Recover the service by splitting on the first underscore. The RE2 service
    naming uses hyphens (``ts-auth-service``, ``front-end``) so the first ``_``
    separates service from method name safely.
    """
    if "_" not in op:
        return op
    return op.split("_", 1)[0]


def trace_acc_at_k(ranks: list[str], truth: str, k: int) -> int:
    """Acc@k over service-deduped operation ranks."""
    seen: list[str] = []
    for op in ranks:
        s = service_from_op(op)
        if s not in seen:
            seen.append(s)
        if len(seen) == k:
            break
    return int(truth in seen)


def summarize_trace(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Acc@1/3/5 and Avg@5 overall and per fault type, using operation->service dedup."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        buckets["_all"].append(r)
        buckets[r["fault"]].append(r)
    out: dict[str, dict[str, float]] = {}
    for key, rs in buckets.items():
        accs = []
        for k in range(1, 6):
            hits = [trace_acc_at_k(r["ranks"], r["service"], k) for r in rs]
            accs.append(float(np.mean(hits)) if hits else 0.0)
        out[key] = {
            "n": len(rs),
            "acc@1": accs[0],
            "acc@3": accs[2],
            "acc@5": accs[4],
            "avg@5": float(np.mean(accs)),
        }
    return out


def process_case(
    func, method: str, case_dir: str, dataset: str, length_min: int = 20
) -> dict:
    span_df, inject_time_us, service, fault = load_trace_case(case_dir, length_min=length_min)
    out = func(
        span_df,
        inject_time_us,
        dataset=dataset,
        anomalies=None,
        dk_select_useful=False,
        sli=None,
        verbose=False,
        n_iter=None,
    )
    ranks = out.get("ranks", [])
    # tracerca returns list[str], microrank returns list[str] (operation names)
    ranks = [str(r) for r in ranks]
    return {"service": service, "fault": fault, "ranks": ranks, "case_dir": case_dir}


def eval_dataset(
    method: str,
    dataset: str,
    length_min: int = 20,
    max_cases: int | None = None,
) -> list[dict]:
    func = resolve_method(method)
    dirs = list_case_dirs(dataset)
    if max_cases is not None:
        dirs = dirs[:max_cases]

    results: list[dict] = []
    for d in tqdm(dirs, desc=f"{method}/{dataset}"):
        try:
            r = process_case(func, method, d, dataset, length_min=length_min)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            case_name = basename(dirname(d))
            service, _, fault = case_name.partition("_")
            print(f"FAIL {d}: {type(e).__name__}: {e}")
            results.append(
                {
                    "service": service,
                    "fault": fault or "unknown",
                    "ranks": [],
                    "case_dir": d,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["tracerca", "microrank"],
        help="Trace-based baseline. Note: 'microrca' is NOT in RCAEval; "
        "use 'tracerca' or 'microrank' as the trace-based stand-in.",
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_MAP))
    parser.add_argument("--length", type=int, default=20, help="(currently unused)")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    t0 = datetime.now()
    results = eval_dataset(
        args.method, args.dataset, length_min=args.length, max_cases=args.max_cases
    )
    dt = (datetime.now() - t0).total_seconds()
    ok_results = [r for r in results if "error" not in r]
    summary = summarize_trace(ok_results)
    print_summary(f"{args.method}/{args.dataset}", args.length, summary, len(ok_results), dt)
    n_failed = len(results) - len(ok_results)
    if n_failed:
        print(f"  ({n_failed} cases failed; see saved JSON for details)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "method": args.method,
                    "dataset": args.dataset,
                    "length_min": args.length,
                    "summary": summary,
                    "cases": results,
                    "wall_s": dt,
                    "n_failed": n_failed,
                },
                f,
            )
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
