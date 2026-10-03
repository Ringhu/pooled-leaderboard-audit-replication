#!/usr/bin/env python3
"""Post-repair CIRCA runner for OpenRCA.

Wraps predict_circa.run_circa so that, before PC/RHT, the pre-filtered
DataFrame's column names are rewritten by circa_schema_shim.repair_openrca_columns.
The shim's mapping dict is then used to map CIRCA's returned metric names
back to the ORIGINAL column names before the downstream component extraction.

Writes predictions to <output_dir>/circa_repaired_<system>_results.csv.
Does NOT modify predict_circa.py or any RCAEval package file.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import predict_circa  # noqa: E402
from circa_schema_shim import repair_openrca_columns  # noqa: E402
from RCAEval.graph_construction.pc import pc_default  # noqa: E402
from RCAEval.graph_heads.rht import rht  # noqa: E402


def _make_run_circa_shim():
    """Return a drop-in replacement for predict_circa.run_circa that applies
    the schema shim before PC/RHT and unmaps returned metric names."""

    def run_circa_repaired(baro_df: pd.DataFrame, inject_time: int):
        # Keep the exact variance-filter logic from predict_circa.run_circa.
        time_col = baro_df["time"]
        pc_input = baro_df.drop(columns=["time"])
        keep = pc_input.std() > 1e-8
        pc_input = pc_input.loc[:, keep]
        if pc_input.shape[1] < 3:
            return []
        data = pc_input.copy()
        data["time"] = time_col

        # SHIM: rewrite column names so rht.py's split("_")[0/1] works.
        renamed_df, mapping = repair_openrca_columns(data)
        renamed_pc_input = renamed_df.drop(columns=["time"])

        try:
            adj = pc_default(renamed_pc_input, show_progress=False)
        except Exception as e:
            print(f"  PC failed: {e}", file=sys.stderr)
            return []
        try:
            ranks = rht(adj, inject_time, renamed_df)
        except Exception as e:
            print(f"  RHT failed: {e}", file=sys.stderr)
            return []
        ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
        # Map each renamed metric back to the ORIGINAL column name.
        return [mapping.get(x[0], x[0]) for x in ranks]

    return run_circa_repaired


def main():
    ap = argparse.ArgumentParser(description="CIRCA with OpenRCA schema repair")
    ap.add_argument("--system", required=True)
    ap.add_argument("--query_csv", required=True)
    ap.add_argument(
        "--results_1min_dir",
        default="$HOME/auditstack/results_1min",
    )
    ap.add_argument(
        "--results_ext_dir",
        default="$HOME/auditstack/results_ext",
    )
    ap.add_argument(
        "--results_dir",
        default="$HOME/auditstack/results",
    )
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Monkey-patch predict_circa.run_circa with our shim'd version. This
    # works because predict_one() calls run_circa by attribute lookup on
    # the module namespace (i.e., predict_circa.run_circa), and our patch
    # rebinds that name before we invoke predict_one.
    predict_circa.run_circa = _make_run_circa_shim()

    qdf = pd.read_csv(args.query_csv)
    cache: dict = {}
    preds = []
    for i, row in qdf.iterrows():
        pred = predict_circa.predict_one(
            args.system, row, cache,
            args.results_1min_dir, args.results_ext_dir, args.results_dir,
            top_k=args.top_k,
        )
        preds.append(pred)
        if i < 3 or (i + 1) % 10 == 0:
            print(
                f"[{args.system} #{i+1}/{len(qdf)} task={row['task_index']}] "
                f"pred: {pred[:160]}...",
                flush=True,
            )

    Path(os.path.dirname(os.path.abspath(args.output))).mkdir(
        parents=True, exist_ok=True
    )
    out = pd.DataFrame({"task_index": qdf["task_index"], "prediction": preds})
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")


if __name__ == "__main__":
    main()
