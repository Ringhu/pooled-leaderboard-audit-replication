"""RCD (RCAEval official implementation) on RE1 (OB data.csv; SS/TT simple_data.csv), one seed per invocation.

Loading/windowing/scoring reuse rcaeval_repo/_baseline_utils.py (same as the audit followup
runs of BARO/CIRCA/e-Diagnosis in pub_simple.py): length_min=20 (10 min each side of
inject_time), `_latency-50` dropped, `_latency-90` renamed, service = col.split("_")[0] minus "-db".
The call mirrors RCAEval main.py (dataset=<re1-xx>, dk_select_useful=False).

Usage: python run_rcd_re1.py re1-ob 0 OUT.json [--limit N] [--procs P]
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
from multiprocessing import Pool

THIS_FILE = os.path.abspath(__file__)
REPO = "$HOME/rcaeval_repo"
sys.path.insert(0, REPO)
os.chdir(REPO)

from _baseline_utils import acc_at_k, load_case, service_from_col  # noqa: E402

ROOTS = {"re1-ob": "data/online-boutique", "re1-ss": "data/sock-shop-2", "re1-tt": "data/train-ticket"}
# RE1-OB ships only data.csv, which is already the reduced 50-metric table; SS/TT ship both a raw
# data.csv (438/1242 cols) and the intended simple_data.csv (60/221 cols).
FNAME = {"re1-ob": "data.csv", "re1-ss": "simple_data.csv", "re1-tt": "simple_data.csv"}


def run_case(job):
    ds, seed, p = job
    from RCAEval.e2e import rcd

    t0 = time.time()
    rec = {"case": "/".join(p.split("/")[-3:-1])}
    try:
        data, it, service, fault, sli = load_case(p, length_min=20)
        rec.update(true_service=service, fault=fault, n_cols=len(data.columns) - 1)
        out = rcd(data, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=sli,
                  verbose=False, n_iter=len(data.columns) - 1, seed=seed)
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
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(ROOTS))
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--input-file", default=None,
                    help="override input (e.g. data.csv on re1-ss/tt for the RQ2(a) raw-export contrast)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    fname = a.input_file or FNAME[a.dataset]

    paths = sorted(glob.glob(os.path.join(ROOTS[a.dataset], "**", fname), recursive=True))
    if a.limit:
        paths = paths[: a.limit]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(run_case, [(a.dataset, a.seed, p) for p in paths], chunksize=1)

    import collections
    n = len(rows)
    modal = collections.Counter(r["top1"] for r in rows).most_common(1)[0]
    src = {f: hashlib.sha256(open(os.path.join(REPO, f), "rb").read()).hexdigest()[:12]
           for f in ("RCAEval/e2e/rcd.py", "_baseline_utils.py")}
    src[os.path.basename(THIS_FILE)] = hashlib.sha256(open(THIS_FILE, "rb").read()).hexdigest()[:12]
    res = {
        "config": {
            "method": "rcd", "dataset": a.dataset, "input_file": fname, "seed": a.seed,
            "code_commit": "rcaeval_repo@497505b", "code_sha256_12": src,
            "env": sys.executable, "python": sys.version.split()[0], "argv": sys.argv,
            "run_started_utc": started, "length_min": 20, "limit": a.limit,
        },
        "method": "rcd", "dataset": a.dataset, "file": fname, "seed": a.seed, "n": n,
        "acc1": round(sum(r["acc1"] for r in rows) / n, 4),
        "acc3": round(sum(r["acc3"] for r in rows) / n, 4),
        "acc5": round(sum(r["acc5"] for r in rows) / n, 4),
        "coverage": sum(r["n_ranks"] > 0 for r in rows),
        "n_errors": sum(r["error"] is not None for r in rows),
        "modal_pred": modal[0], "modal_freq": round(modal[1] / n, 3),
        "n_distinct_top1": len(set(r["top1"] for r in rows)),
        "wall_s": round(time.time() - t0, 1), "cases": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "config")}), flush=True)


if __name__ == "__main__":
    main()
