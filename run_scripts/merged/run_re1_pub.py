"""CIRCA / e-Diagnosis (RCAEval official implementations) on RE1, per-case output with ranks.

Same loader and call as rcaeval_repo/pub_simple.py (the audit followup runs used for RE1-SS/TT):
_baseline_utils.load_case(length_min=20), fn(data, it, dataset=ds, anomalies=None,
dk_select_useful=False, sli=sli, verbose=False, n_iter=<n metric cols>). e-Diagnosis bootstraps with
np.random; this script seeds np.random with 1000 + case index (pub_simple.py did not seed).
Written 2026-09-30 for the RE1-OB duplicate-"time"-header rerun (see timefix_wrapper.py).

Usage: python run_re1_pub.py {circa,e_diagnosis} re1-ob OUT.json [--input-file data.csv] [--only-faults cpu,mem]
"""
import argparse
import datetime
import glob
import hashlib
import importlib.util as il
import json
import os
import sys
import time
import traceback

THIS_FILE = os.path.abspath(__file__)
CWD0 = os.getcwd()
REPO = "$HOME/rcaeval_repo"
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np  # noqa: E402
from _baseline_utils import acc_at_k, load_case, service_from_col  # noqa: E402

ROOTS = {"re1-ob": "data/online-boutique", "re1-ss": "data/sock-shop-2", "re1-tt": "data/train-ticket"}
FNAME = {"re1-ob": "data.csv", "re1-ss": "simple_data.csv", "re1-tt": "simple_data.csv"}
SRC = {"circa": "RCAEval/e2e/circa.py", "e_diagnosis": "RCAEval/e2e/__init__.py"}  # e_diagnosis is defined in e2e/__init__.py


def get_fn(which):
    if which == "e_diagnosis":
        from RCAEval.e2e import e_diagnosis
        return e_diagnosis
    s = il.spec_from_file_location("circa_mod", os.path.join(REPO, SRC["circa"]))
    m = il.module_from_spec(s)
    s.loader.exec_module(m)
    return m.circa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=sorted(SRC))
    ap.add_argument("dataset", choices=sorted(ROOTS))
    ap.add_argument("out")
    ap.add_argument("--input-file", default=None)
    ap.add_argument("--only-faults", default="")
    a = ap.parse_args()
    a.out = os.path.join(CWD0, a.out)
    fname = a.input_file or FNAME[a.dataset]
    fn = get_fn(a.method)
    paths = sorted(p for p in glob.glob(os.path.join(ROOTS[a.dataset], "*", "*", fname))
                   if os.path.basename(os.path.dirname(p)).isdigit())
    if a.only_faults:
        keep = set(a.only_faults.split(","))
        paths = [p for p in paths if p.split("/")[-3].split("_")[-1] in keep]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    rows = []
    for i, p in enumerate(paths):
        rec = {"case": "/".join(p.split("/")[-3:-1])}
        tc = time.time()
        try:
            np.random.seed(1000 + i)
            data, it, service, fault, sli = load_case(p, length_min=20)
            n = len([c for c in data.columns if c != "time"])
            rec.update(true_service=service, fault=fault, n_cols=n)
            out = fn(data, it, dataset=a.dataset, anomalies=None, dk_select_useful=False,
                     sli=sli, verbose=False, n_iter=n)
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
        rec["wall_s"] = round(time.time() - tc, 2)
        rows.append(rec)
        print(rec["case"], rec["top1"], rec["acc1"], rec["wall_s"], "ERR" if rec["error"] else "", flush=True)
    n = len(rows)
    src = {f: hashlib.sha256(open(os.path.join(REPO, f), "rb").read()).hexdigest()[:12]
           for f in (SRC[a.method], "_baseline_utils.py")}
    src[os.path.basename(THIS_FILE)] = hashlib.sha256(open(THIS_FILE, "rb").read()).hexdigest()[:12]
    res = {
        "config": {"method": a.method, "dataset": a.dataset, "input_file": fname, "length_min": 20,
                   "code_commit": "rcaeval_repo@497505b", "code_sha256_12": src, "env": sys.executable,
                   "python": sys.version.split()[0], "argv": sys.argv, "run_started_utc": started,
                   "np_seed": "1000+case_index", "only_faults": a.only_faults},
        "method": a.method, "dataset": a.dataset, "file": fname, "n": n,
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
