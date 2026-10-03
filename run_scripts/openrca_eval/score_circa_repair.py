#!/usr/bin/env python3
"""Score CIRCA pre/post-repair on all 4 OpenRCA subsystems.

For each system:
  - Load pre-repair preds (preds_circa_<sys>.csv)
  - Load post-repair preds (results_circa_repair/circa_repaired_<sys>_results.csv)
  - Evaluate each row via openrca_evaluate.evaluate() to get binary acc@1 (score==1.0)
  - Compute modal top-1 component frequency (from the FIRST JSON dict in prediction)

Outputs: a summary table to stdout and writes score_circa_repair_summary.csv.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from openrca_evaluate import evaluate  # noqa: E402

BASE = "$HOME/auditstack"
SYSTEMS = [
    ("bank",  "bank",  os.path.join(HERE, "bank_q.csv"),
     os.path.join(HERE, "preds_circa_bank.csv"),
     os.path.join(BASE, "results_circa_repair/circa_repaired_bank_results.csv")),
    ("mkt1",  "mkt1",  os.path.join(HERE, "mkt1_q.csv"),
     os.path.join(HERE, "preds_circa_mkt1.csv"),
     os.path.join(BASE, "results_circa_repair/circa_repaired_mkt1_results.csv")),
    ("mkt2",  "mkt2",  os.path.join(HERE, "mkt2_q.csv"),
     os.path.join(HERE, "preds_circa_mkt2.csv"),
     os.path.join(BASE, "results_circa_repair/circa_repaired_mkt2_results.csv")),
    ("tel",   "tel",   os.path.join(HERE, "tel_q.csv"),
     os.path.join(HERE, "preds_circa_tel.csv"),
     os.path.join(BASE, "results_circa_repair/circa_repaired_tel_results.csv")),
]


FIRST_COMP_RE = re.compile(r'"root cause component":\s*"([^"]+)"')


def first_component(pred_str: str) -> str:
    m = FIRST_COMP_RE.search(str(pred_str))
    return m.group(1) if m else ""


def score_file(pred_path: str, query_path: str) -> tuple[float, float, int, str, int]:
    """Return (acc1, modal_freq, n, modal_value, n_unique_top1)."""
    preds = pd.read_csv(pred_path)
    qdf = pd.read_csv(query_path)
    # Align by row order (matches predict_circa's output convention).
    if len(preds) != len(qdf):
        # tolerate mismatch by taking min
        n = min(len(preds), len(qdf))
        preds = preds.iloc[:n].reset_index(drop=True)
        qdf = qdf.iloc[:n].reset_index(drop=True)
    n = len(preds)
    n_correct = 0
    top1s: list[str] = []
    for i in range(n):
        pred = preds.iloc[i]["prediction"]
        sp = qdf.iloc[i]["scoring_points"]
        try:
            _, _, score = evaluate(str(pred), str(sp))
        except Exception:
            score = 0.0
        if float(score) >= 1.0:
            n_correct += 1
        top1s.append(first_component(pred))
    acc1 = n_correct / n if n else 0.0
    counter = Counter(top1s)
    modal_val, modal_count = counter.most_common(1)[0]
    modal_freq = modal_count / n if n else 0.0
    n_unique = len({t for t in top1s if t})
    return acc1, modal_freq, n, modal_val, n_unique


def main():
    rows = []
    for sys_key, label, q, pre, post in SYSTEMS:
        print(f"\n=== System: {label} ===")
        for tag, path in (("pre", pre), ("post", post)):
            if not os.path.exists(path):
                print(f"  [{tag}] MISSING {path}")
                rows.append({
                    "system": label, "variant": tag, "n": 0,
                    "acc1": None, "modal_freq": None,
                    "modal_top1": "", "n_unique_top1": 0,
                    "status": "missing",
                })
                continue
            acc1, mfreq, n, mval, nu = score_file(path, q)
            status = "degenerate" if mfreq >= 0.9 else "valid"
            print(f"  [{tag}] n={n}  acc1={acc1:.4f}  modal='{mval}' ({mfreq:.1%})  "
                  f"unique_top1={nu}  -> {status}")
            rows.append({
                "system": label, "variant": tag, "n": n,
                "acc1": round(acc1, 4),
                "modal_freq": round(mfreq, 4),
                "modal_top1": mval, "n_unique_top1": nu,
                "status": status,
            })
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(BASE, "results_circa_repair",
                            "score_circa_repair_summary.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("\n=== Final summary ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
