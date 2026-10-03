"""BARO (RCAEval RCAEval/e2e/baro.py) on RE1 (OB data.csv; SS/TT simple_data.csv) or RE2 (simple_metrics.csv), per-case output.

Same loader/call as the RE1 CIRCA / e-Diagnosis / RCD runs (rcaeval_repo/_baseline_utils.load_case,
dataset=<re1-xx>, dk_select_useful=False, main.py --length default 20 min). --length 10 is the
window the audit long-table BARO rows used (auditstack openrca_eval/run_rcaeval.py).

Usage: python run_re1_baro.py re1-ob OUT.json [--length 20] [--input-file data.csv]
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import time
import traceback

THIS_FILE = os.path.abspath(__file__)
CWD0 = os.getcwd()  # resolve OUT before chdir(REPO)
REPO = "$HOME/rcaeval_repo"
sys.path.insert(0, REPO)
os.chdir(REPO)

from _baseline_utils import acc_at_k, load_case, service_from_col  # noqa: E402
from RCAEval.e2e.baro import baro  # noqa: E402

ROOTS = {"re1-ob": "data/online-boutique", "re1-ss": "data/sock-shop-2", "re1-tt": "data/train-ticket",
         "re2-ob": "data/RE2/RE2-OB", "re2-ss": "data/RE2/RE2-SS", "re2-tt": "data/RE2/RE2-TT"}
FNAME = {"re1-ob": "data.csv", "re1-ss": "simple_data.csv", "re1-tt": "simple_data.csv",
         "re2-ob": "simple_metrics.csv", "re2-ss": "simple_metrics.csv", "re2-tt": "simple_metrics.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(ROOTS))
    ap.add_argument("out")
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--input-file", default=None)
    a = ap.parse_args()
    a.out = os.path.join(CWD0, a.out)
    fname = a.input_file or FNAME[a.dataset]
    # numbered case directories only (RE2-OB also has a multi-source-data example directory)
    paths = sorted(p for p in glob.glob(os.path.join(ROOTS[a.dataset], "*", "*", fname))
                   if os.path.basename(os.path.dirname(p)).isdigit())
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    rows = []
    for p in paths:
        rec = {"case": "/".join(p.split("/")[-3:-1])}
        try:
            data, it, service, fault, sli = load_case(p, length_min=a.length)
            rec.update(true_service=service, fault=fault, n_cols=len(data.columns) - 1)
            out = baro(data, inject_time=it, dataset=a.dataset, anomalies=None, dk_select_useful=False, sli=sli)
            ranks = [str(r) for r in (out.get("ranks") or [])]
            rec["error"] = None
        except Exception as e:
            ranks = []
            rec["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-2000:]
        rec["ranks"] = ranks[:20]
        rec["n_ranks"] = len(ranks)
        rec["top1"] = service_from_col(ranks[0]) if ranks else None
        truth = rec.get("true_service")
        for k in (1, 3, 5):
            rec[f"acc{k}"] = acc_at_k(ranks, truth, k) if ranks and truth else 0
        rows.append(rec)
    n = len(rows)
    src = {f: hashlib.sha256(open(os.path.join(REPO, f), "rb").read()).hexdigest()[:12]
           for f in ("RCAEval/e2e/baro.py", "_baseline_utils.py")}
    src[os.path.basename(THIS_FILE)] = hashlib.sha256(open(THIS_FILE, "rb").read()).hexdigest()[:12]
    res = {
        "config": {"method": "baro", "dataset": a.dataset, "input_file": fname, "length_min": a.length,
                   "code_commit": "rcaeval_repo@497505b", "code_sha256_12": src, "env": sys.executable,
                   "python": sys.version.split()[0], "argv": sys.argv, "run_started_utc": started},
        "method": "baro", "dataset": a.dataset, "file": fname, "n": n,
        **{f"acc{k}": round(sum(r[f"acc{k}"] for r in rows) / n, 4) for k in (1, 3, 5)},
        "coverage": sum(r["n_ranks"] > 0 for r in rows), "n_errors": sum(r["error"] is not None for r in rows),
        "wall_s": round(time.time() - t0, 1), "cases": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "config")}), flush=True)


if __name__ == "__main__":
    main()
