"""CausalRCA (RCAEval implementation, unchanged) on RE1 and RE2, seeds 0-2, all cases, in parallel.

Decision 2026-09-29 (user): CausalRCA enters the main table, run in the `rcaeval_rcd` env
(scikit-network 0.31.0; under 0.33.0 `PageRank.fit_transform` no longer exists and causalrca.py
returns the input column order from its `except Exception` branch).

causalrca.py defines `seed = 42` but never applies it; each job sets random / numpy / torch seeds to
the run seed before the call. torch threads 1.
--device cpu: rcaeval_rcd env, CUDA hidden (first attempt 2026-09-29: TT jobs had not finished after 9 h).
--device gpu: `ts` env (torch 2.6.0+cu124) with PYTHONPATH=work/pylib_skn031 (scikit-network 0.31.0 and
numpy 1.26.4 installed there with pip --target; the ts env itself is unchanged); causalrca.py moves the
model to CUDA when torch.cuda.is_available(). Each job takes a GPU slot from a shared queue (--per-gpu
slots per GPU) and returns it when done, so running jobs stay balanced across GPUs.
Input: RE1-OB data.csv, RE1-SS/TT simple_data.csv, RE2 simple_metrics.csv (numbered case dirs only);
loading via rcaeval_repo/_baseline_utils.load_case(length_min=20) (main.py conventions, as in the RCD
and permutation runs); call mirrors main.py (dataset=<re1-xx|re2-xx>, dk_select_useful=False).
Scoring: _baseline_utils.acc_at_k (service = col.split("_")[0] minus "-db").

Jobs (dataset, seed, case) run in one pool, largest inputs first; each job writes its own file under
work/causalrca_jobs/ (outside runs/, so not registered) and is skipped if that file exists (resumable).

Usage:
  python run_causalrca_all.py run --procs 48
  python run_causalrca_all.py collect      # -> runs/causalrca_full/causalrca_<ds>_seed<s>.json
  python run_causalrca_all.py status
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import random
import sys
import time
import traceback
from multiprocessing import Pool, Queue

THIS_FILE = os.path.abspath(__file__)
M = "$AUDIT_ROOT"
REPO = "$HOME/rcaeval_repo"
JOBS = f"{M}/work/causalrca_jobs"  # + "_gpu" for --device gpu
OUT = f"{M}/runs/causalrca_full"
sys.path.insert(0, REPO)

from _baseline_utils import acc_at_k, load_case, service_from_col  # noqa: E402

SOURCES = {"re1-ob": ("online-boutique", "data.csv"), "re1-ss": ("sock-shop-2", "simple_data.csv"),
           "re1-tt": ("train-ticket", "simple_data.csv"), "re2-ob": ("RE2/RE2-OB", "simple_metrics.csv"),
           "re2-ss": ("RE2/RE2-SS", "simple_metrics.csv"), "re2-tt": ("RE2/RE2-TT", "simple_metrics.csv")}
SEEDS = (0, 1, 2)
SUFFIX = ""
SLOTS = None  # set in worker initializer


def init_worker(q):
    global SLOTS
    SLOTS = q


def case_paths(ds):
    root, fname = SOURCES[ds]
    return sorted(p for p in glob.glob(f"{REPO}/data/{root}/*/*/{fname}")
                  if os.path.basename(os.path.dirname(p)).isdigit())


def job_file(ds, seed, p):
    return f"{JOBS}{SUFFIX}/{ds}/seed{seed}/" + "__".join(p.split("/")[-3:-1]) + ".json"


def all_jobs():
    jobs = []
    for ds in SOURCES:
        for p in case_paths(ds):
            with open(p) as f:
                n_cols = len(f.readline().split(","))
            for s in SEEDS:
                jobs.append((n_cols, ds, s, p))
    return sorted(jobs, key=lambda j: (-j[0], j[1], j[2], j[3]))


def run_job(job):
    n_cols, ds, seed, p = job
    gpu = SLOTS.get()
    try:
        return _run_job(n_cols, ds, seed, p, gpu)
    finally:
        SLOTS.put(gpu)


def _run_job(n_cols, ds, seed, p, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # "" = CPU; set before torch is imported (fresh process per job)
    out = job_file(ds, seed, p)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from RCAEval.e2e.causalrca import causalrca

    rec = {"case": "/".join(p.split("/")[-3:-1]), "seed": seed, "gpu": gpu, "cuda": torch.cuda.is_available()}
    t0 = time.time()
    try:
        data, it, service, fault, _ = load_case(p, length_min=20)
        rec.update(true_service=service, fault=fault, n_cols=len(data.columns) - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        res = causalrca(data, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=None, verbose=False)
        ranks = [str(r) for r in (res.get("ranks") or [])]
        names = [str(r) for r in (res.get("node_names") or [])]
        rec["ranks_equal_column_order"] = bool(ranks) and ranks == names
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
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(out + ".tmp", out)
    return ds, seed, rec["case"], rec["error"] is None, rec["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "collect", "status"])
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--per-gpu", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated datasets (smoke/timing)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    global SUFFIX
    SUFFIX = "_gpu" if a.device == "gpu" else ""
    jobs = all_jobs()
    if a.only:
        jobs = [j for j in jobs if j[1] in a.only.split(",")]
    todo = [j for j in jobs if not os.path.exists(job_file(j[1], j[2], j[3]))]
    if a.mode == "status":
        done = len(jobs) - len(todo)
        print(json.dumps({"jobs": len(jobs), "done": done, "todo": len(todo)}))
        return
    if a.mode == "run":
        todo = todo[:a.limit] if a.limit else todo
        q = Queue()
        slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)] if a.device == "gpu" else [""] * a.procs
        for g in slots:
            q.put(g)
        procs = min(a.procs, len(slots))
        print(f"{len(jobs)} jobs, {len(todo)} to run, procs {procs}", flush=True)
        with Pool(procs, initializer=init_worker, initargs=(q,), maxtasksperchild=1) as pool:
            for i, r in enumerate(pool.imap_unordered(run_job, todo, chunksize=1), 1):
                print(f"{i}/{len(todo)} {r[0]} seed{r[1]} {r[2]} ok={r[3]} {r[4]}s", flush=True)
        print("ALL_DONE", flush=True)
        return
    # collect
    assert not todo, f"{len(todo)} jobs not finished"
    import sknetwork
    import torch
    src = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]
           for f in (f"{REPO}/RCAEval/e2e/causalrca.py", f"{REPO}/_baseline_utils.py", THIS_FILE)}
    os.makedirs(OUT, exist_ok=True)
    for ds in SOURCES:
        for s in SEEDS:
            rows = [json.load(open(job_file(ds, s, p))) for p in case_paths(ds)]
            n = len(rows)
            res = {"config": {"method": "causalrca", "dataset": ds, "input_file": SOURCES[ds][1], "seed": s,
                              "length_min": 20, "code_commit": "rcaeval_repo@497505b", "code_sha256_12": src,
                              "env": sys.executable, "python": sys.version.split()[0],
                              "scikit_network": sknetwork.__version__, "torch": torch.__version__,
                              "device": a.device, "pythonpath": os.environ.get("PYTHONPATH"),
                              "argv": "run_causalrca_all.py run/collect",
                              "run_started_utc": min(r["finished_utc"] for r in rows)},
                   "method": "causalrca", "dataset": ds, "file": SOURCES[ds][1], "seed": s, "n": n,
                   **{f"acc{k}": round(sum(r[f"acc{k}"] for r in rows) / n, 4) for k in (1, 3, 5)},
                   "coverage": sum(r["n_ranks"] > 0 for r in rows), "n_errors": sum(r["error"] is not None for r in rows),
                   "n_ranks_equal_column_order": sum(bool(r.get("ranks_equal_column_order")) for r in rows),
                   "wall_s": round(sum(r["wall_s"] for r in rows), 1), "cases": rows}
            assert all(r["cuda"] == (a.device == "gpu") for r in rows), (ds, s)
            with open(f"{OUT}/causalrca_{ds}_seed{s}.json", "w") as f:
                json.dump(res, f, indent=1)
            print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "config")}), flush=True)


if __name__ == "__main__":
    main()
