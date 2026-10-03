"""audit SimpleRCA rewrite (metric branch as described in Fang et al. §3.1.2) on RE1, per-case output.

Imports ~/earlier-b/scripts/pilot/64_simplerca_rcaeval.py unchanged and calls its process_case on the
project's fixed RE1 inputs (OB data.csv; SS/TT simple_data.csv; the audit run used data.csv for all three).
Sensitivity control only (decision 2026-09-27, SimpleRCA scope option A). Window 20 min (RCAEval default).
Scoring: rcaeval_repo/_baseline_utils.acc_at_k.

Usage: python run_simplerca_rewrite_re1.py re1-ss OUT.json [--input-file F]
"""
import argparse
import datetime
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

THIS_FILE = os.path.abspath(__file__)
CWD0 = os.getcwd()
B = "$HOME/earlier-b/scripts/pilot"
REPO = "$HOME/rcaeval_repo"
sys.path.insert(0, B)
spec = importlib.util.spec_from_file_location("b_simplerca", f"{B}/64_simplerca_rcaeval.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)
sys.path.insert(0, REPO)
import _baseline_utils as rbu  # noqa: E402  (rcaeval_repo copy: scorer)

ROOTS = {"re1-ob": f"{REPO}/data/online-boutique", "re1-ss": f"{REPO}/data/sock-shop-2",
         "re1-tt": f"{REPO}/data/train-ticket"}
FNAME = {"re1-ob": "data.csv", "re1-ss": "simple_data.csv", "re1-tt": "simple_data.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(ROOTS))
    ap.add_argument("out")
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--input-file", default=None)
    a = ap.parse_args()
    a.out = os.path.join(CWD0, a.out)
    fname = a.input_file or FNAME[a.dataset]
    paths = sorted(p for p in glob.glob(os.path.join(ROOTS[a.dataset], "*", "*", fname))
                   if os.path.basename(os.path.dirname(p)).isdigit())
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    rows = []
    for p in paths:
        rec = {"case": "/".join(p.split("/")[-3:-1])}
        try:
            r = bs.process_case(p, length_min=a.length)
            ranks = [str(x) for x in r["ranks"]]
            rec.update(true_service=r["service"], fault=r["fault"], error=None)
        except Exception as e:
            ranks = []
            rec["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:]
        rec["ranks"] = ranks
        rec["n_ranks"] = len(ranks)
        truth = rec.get("true_service")
        for k in (1, 3, 5):
            rec[f"acc{k}"] = float(rbu.acc_at_k(ranks, truth, k)) if ranks and truth else 0.0
        rows.append(rec)
    n = len(rows)
    src = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]
           for f in (f"{B}/64_simplerca_rcaeval.py", f"{B}/_baseline_utils.py", THIS_FILE)}
    res = {"config": {"method": "simplerca_rewrite", "dataset": a.dataset, "input_file": fname,
                      "length_min": a.length, "code_commit": "earlier-b snapshot/pre-merge-2026-09-27 (B paper)",
                      "code_sha256_12": src, "env": sys.executable, "python": sys.version.split()[0],
                      "argv": sys.argv, "run_started_utc": started},
           "method": "simplerca_rewrite", "dataset": a.dataset, "file": fname, "n": n,
           **{f"acc{k}": round(sum(r[f"acc{k}"] for r in rows) / n, 4) for k in (1, 3, 5)},
           "coverage": sum(r["n_ranks"] > 0 for r in rows), "n_errors": sum(r["error"] is not None for r in rows),
           "wall_s": round(time.time() - t0, 1), "cases": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "config")}), flush=True)


if __name__ == "__main__":
    main()
