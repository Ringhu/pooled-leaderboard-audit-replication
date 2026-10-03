#!/usr/bin/env python3
"""Score alertcount predictions on OpenRCA's 4 subsystems.

Reads:
  preds_alertcount_<tag>.csv          (per-case prediction JSON)
  <tag>_q.csv                          (per-case ground truth)

Emits:
  results_openrca_alertcount/alertcount_<tag>_results.csv
      columns: task_index, score, tolerance_s=60 (single-tolerance per-case)
  results_openrca_alertcount/alertcount_<tag>_results_v2.csv
      RCAEval-compatible schema:
      method, tdelta=0, case=<tag>/task_idx_i, true_service="",
      fault="", top1="", acc1_v2=score, acc@3/5/10=None, n_ranks=None
  results_openrca_alertcount/scores_per_incident.csv
      long format at tol in [30,60,120,300,600] (for comparison w/ legacy)
  results_openrca_alertcount/summary_acc1.csv
      per-system acc@1 at tolerance=60 (for equivalence comparison)

Acc@1 at tolerance_s=60 is the canonical metric used by loso_xbench and
the System Matters paper.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from openrca_evaluate_sweep import evaluate  # noqa: E402

BASE = Path("$HOME/auditstack")
EVAL_DIR = Path(HERE)
OUT_DIR = BASE / "results_openrca_alertcount"

# query.csv is the full-335 ground truth; tag -> (pretty label, query path)
OPEN_RCA = BASE / "data" / "OpenRCA" / "dataset"
SYSTEMS = [
    ("bank", "Bank",    OPEN_RCA / "Bank" / "query.csv"),
    ("mkt1", "Mkt-cb1", OPEN_RCA / "Market" / "cloudbed-1" / "query.csv"),
    ("mkt2", "Mkt-cb2", OPEN_RCA / "Market" / "cloudbed-2" / "query.csv"),
    ("tel",  "Telecom", OPEN_RCA / "Telecom" / "query.csv"),
]

TOLERANCES = [30, 60, 120, 300, 600]
CANONICAL_TOL = 60


def score_one(tag: str, label: str, q_path: Path, tol_list=TOLERANCES):
    pred_path = EVAL_DIR / f"preds_alertcount_{tag}.csv"
    assert pred_path.exists(), pred_path
    assert q_path.exists(), q_path
    preds = pd.read_csv(pred_path)
    q = pd.read_csv(q_path)
    assert len(preds) == len(q), (
        f"{tag}: pred rows={len(preds)}  query rows={len(q)} (mismatch)"
    )
    rows_long = []
    rows_v2 = []
    n = len(preds)
    for i in range(n):
        pred = preds.iloc[i]["prediction"]
        sp = q.iloc[i]["scoring_points"]
        tidx = q.iloc[i]["task_index"]
        for tol in tol_list:
            _, _, s = evaluate(str(pred), str(sp), time_tolerance_seconds=tol)
            rows_long.append({
                "method": "alertcount",
                "system": label,
                "system_key": tag,
                "task_index": tidx,
                "tolerance_s": tol,
                "score": float(s),
            })
        _, _, s60 = evaluate(str(pred), str(sp), time_tolerance_seconds=CANONICAL_TOL)
        rows_v2.append({
            "method": "alertcount",
            "tdelta": 0,
            "case": f"{tag}/{tidx}",
            "true_service": "",
            "fault": "",
            "top1": "",
            "acc1_v2": float(s60),
            "acc@3": None,
            "acc@5": None,
            "acc@10": None,
            "n_ranks": None,
        })
    return rows_long, rows_v2


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_long = []
    summary_rows = []
    for tag, label, q_path in SYSTEMS:
        print(f"\n=== {label} ({tag}) ===")
        rows_long, rows_v2 = score_one(tag, label, q_path)
        df_long = pd.DataFrame(rows_long)
        df_v2 = pd.DataFrame(rows_v2)
        out_v2 = OUT_DIR / f"alertcount_{tag}_results_v2.csv"
        df_v2.to_csv(out_v2, index=False)
        out_csv = OUT_DIR / f"alertcount_{tag}_results.csv"
        df_long.to_csv(out_csv, index=False)
        all_long.append(df_long)

        # Acc@1 at canonical tolerance = mean of binary score==1 indicator.
        can = df_long[df_long["tolerance_s"] == CANONICAL_TOL]
        acc1_strict = float((can["score"] == 1.0).mean())
        acc1_partial = float(can["score"].mean())
        summary_rows.append({
            "system": label,
            "system_key": tag,
            "n": len(can),
            "acc1_strict_tol60": acc1_strict,
            "acc1_partial_tol60": acc1_partial,
        })
        print(f"  n={len(can)}  acc1_strict(tol=60)={acc1_strict:.4f}  "
              f"acc1_partial(tol=60)={acc1_partial:.4f}")
        print(f"  wrote {out_v2}")
        print(f"  wrote {out_csv}")

    # Save pooled long CSV and per-system summary.
    pd.concat(all_long, ignore_index=True).to_csv(
        OUT_DIR / "scores_per_incident.csv", index=False
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_acc1.csv", index=False)
    print("\n=== Summary (tol=60s) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSummary: {OUT_DIR / 'summary_acc1.csv'}")


if __name__ == "__main__":
    main()
