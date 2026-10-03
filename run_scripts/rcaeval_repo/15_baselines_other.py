"""Official reproduction of ε-Diagnosis, MicroCause, and CausalRCA on RE1+RE2.

Parallels `14_baro_official_repro.py` but parameterizes the method. Follows
main.py's `process()` contract: same CLI dataset names, same window size,
same kwargs passed to the baseline entry point.

The three methods live in different places inside `RCAEval.e2e`:
- ``e_diagnosis``  — defined inside ``RCAEval/e2e/__init__.py`` itself
  (loaded via normal package import; requires ``causallearn`` which is now
  installed in the ``ts`` env).
- ``microcause``   — ``RCAEval/e2e/microcause.py`` (loaded via importlib
  to match the BARO script's style, also works via normal import now).
- ``causalrca``    — ``RCAEval/e2e/causalrca.py`` (same pattern).

Run on 3090 from the RCAEval repo root so relative imports inside the
baselines resolve correctly. CPU-only — do not touch GPU 0 or 1.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Ensure RCAEval is importable (sibling-path style used by main.py)
sys.path.insert(0, "$HOME/rcaeval_repo")

# Allow running from any directory by importing _baseline_utils via explicit path
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from _baseline_utils import (  # noqa: E402
    DATASET_MAP,
    list_case_paths,
    load_case,
    print_summary,
    summarize,
)


def _load_baseline_from_file(path: str, attr: str):
    """Load a function from a .py file via importlib (bypasses __init__.py)."""
    spec = importlib.util.spec_from_file_location(f"{attr}_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, attr)


def resolve_method(method: str):
    """Return the baseline function for `method`."""
    if method == "e_diagnosis":
        # Defined inside __init__.py, so import from the package
        from RCAEval.e2e import e_diagnosis  # noqa: PLC0415

        return e_diagnosis
    if method == "microcause":
        return _load_baseline_from_file(
            "$HOME/rcaeval_repo/RCAEval/e2e/microcause.py", "microcause"
        )
    if method == "causalrca":
        return _load_baseline_from_file(
            "$HOME/rcaeval_repo/RCAEval/e2e/causalrca.py", "causalrca"
        )
    if method == "circa":
        return _load_baseline_from_file(
            "$HOME/rcaeval_repo/RCAEval/e2e/circa.py", "circa"
        )
    raise ValueError(f"Unknown method: {method}")


def process_case(func, method: str, data_path: str, dataset: str, length_min: int = 20) -> dict:
    """Run the baseline on one case and return {service, fault, ranks}."""
    data, inject_time, service, fault, sli = load_case(data_path, length_min=length_min)

    # Match main.py:292-309 call signature for process()
    num_node = len([c for c in data.columns if c != "time"])
    out = func(
        data,
        inject_time,
        dataset=dataset,
        anomalies=None,
        dk_select_useful=False,
        sli=sli,
        verbose=False,
        n_iter=num_node,
    )
    # All three baselines return a dict with "ranks" key
    ranks = out.get("ranks", [])
    return {"service": service, "fault": fault, "ranks": list(ranks)}


def eval_dataset(
    method: str,
    dataset: str,
    length_min: int = 20,
    max_cases: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> list[dict]:
    func = resolve_method(method)
    paths = list_case_paths(dataset)
    if max_cases is not None:
        paths = paths[:max_cases]
    if start is not None or end is not None:
        paths = paths[(start or 0):(end if end is not None else len(paths))]

    results: list[dict] = []
    for p in tqdm(paths, desc=f"{method}/{dataset}"):
        try:
            r = process_case(func, method, p, dataset, length_min=length_min)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            # Keep a placeholder so case counts are preserved; surface error
            print(f"FAIL {p}: {type(e).__name__}: {e}")
            results.append(
                {
                    "service": Path(p).parent.parent.name.split("_")[0],
                    "fault": Path(p).parent.parent.name.split("_")[1]
                    if "_" in Path(p).parent.parent.name
                    else "unknown",
                    "ranks": [],
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["e_diagnosis", "microcause", "causalrca", "circa"],
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_MAP))
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit to first N cases (smoke testing).",
    )
    parser.add_argument("--start", type=int, default=None,
                        help="Shard: first case index (inclusive), applied to the sorted path list.")
    parser.add_argument("--end", type=int, default=None,
                        help="Shard: last case index (exclusive).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    t0 = datetime.now()
    results = eval_dataset(
        args.method, args.dataset, length_min=args.length, max_cases=args.max_cases,
        start=args.start, end=args.end,
    )
    dt = (datetime.now() - t0).total_seconds()
    # Only count non-failing cases in summary, but save all
    ok_results = [r for r in results if "error" not in r]
    summary = summarize(ok_results)
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
