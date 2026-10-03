"""WP2.3 CausalRCA output-integrity test: does the ranking follow the input column order?

For each sampled case, run RCAEval's causalrca (loaded by file path, as in the audit runs) twice:
on the table as loaded, and on the same table with the non-time columns permuted (fixed seed per
case). Record whether the returned ranking equals the (post-preprocess) column order, and whether the
learned adjacency returned by causalrca is empty / contains NaN (the two code paths that return
column order: PageRank exception -> `ranks = node_names`, or an empty graph -> uniform scores ->
stable sort keeps column order).

Loading uses rcaeval_repo/_baseline_utils.load_case(length_min=20), as in the audit runs.
CPU only (CUDA_VISIBLE_DEVICES="").
"""
import datetime
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = ""
THIS_FILE = os.path.abspath(__file__)
REPO = "$HOME/rcaeval_repo"
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from _baseline_utils import load_case, service_from_col  # noqa: E402

SOURCES = {  # (glob, stride) -> 10 cases each, spread over services and fault types
    "re1-ob": (f"{REPO}/data/online-boutique/*/*/data.csv", 12),
    "re2-ob": (f"{REPO}/data/RE2/RE2-OB/*/[0-9]*/simple_metrics.csv", 7),
    "re2-ss": (f"{REPO}/data/RE2/RE2-SS/*/[0-9]*/simple_metrics.csv", 7),
}


def load_causalrca():
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("causalrca_mod", f"{REPO}/RCAEval/e2e/causalrca.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.causalrca


def one_run(fn, data, it, ds):
    t0 = time.time()
    out = fn(data, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=None, verbose=False)
    adj = np.asarray(out.get("adj"), dtype=float) if out.get("adj") is not None else np.zeros((0, 0))
    names = [str(x) for x in out.get("node_names", [])]
    ranks = [str(x) for x in out.get("ranks", [])]
    return {
        "node_names": names, "ranks": ranks,
        "ranks_equal_column_order": ranks == names,
        "adj_nonzero": int(np.count_nonzero(np.nan_to_num(adj))), "adj_nan": int(np.isnan(adj).sum()),
        "top1_service": service_from_col(ranks[0]) if ranks else None, "wall_s": round(time.time() - t0, 1),
    }


def run_case(job):
    ds, idx, path = job
    rec = {"dataset": ds, "case": "/".join(path.split("/")[-3:-1])}
    try:
        fn = load_causalrca()
        data, it, service, fault, _ = load_case(path, length_min=20)
        rec.update(true_service=service, fault=fault)
        rec["original"] = one_run(fn, data.copy(), it, ds)
        cols = [c for c in data.columns if c != "time"]
        perm = list(np.random.default_rng(1000 + idx).permutation(cols))
        rec["permuted_input_order"] = perm
        rec["permuted"] = one_run(fn, data[["time"] + perm].copy(), it, ds)
        rec["error"] = None
    except Exception as e:
        rec["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-2000:]
    return rec


def main():
    out_path = sys.argv[1]
    jobs = []
    for ds, (pat, stride) in SOURCES.items():
        paths = sorted(glob.glob(pat))[::stride][:10]
        jobs += [(ds, i, p) for i, p in enumerate(paths)]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as pool:
        rows = pool.map(run_case, jobs, chunksize=1)
    ok = [r for r in rows if r["error"] is None]
    summary = {
        "n_cases": len(rows), "n_ok": len(ok),
        "original_equals_column_order": sum(r["original"]["ranks_equal_column_order"] for r in ok),
        "permuted_equals_permuted_order": sum(r["permuted"]["ranks_equal_column_order"] for r in ok),
        "top1_changes_under_permutation": sum(r["original"]["top1_service"] != r["permuted"]["top1_service"] for r in ok),
        "original_adj_all_zero": sum(r["original"]["adj_nonzero"] == 0 for r in ok),
        "original_adj_has_nan": sum(r["original"]["adj_nan"] > 0 for r in ok),
    }
    res = {"config": {"method": "causalrca", "experiment": "column-permutation", "input_file": "RE1-OB data.csv; RE2 simple_metrics.csv",
                      "code_commit": "rcaeval_repo@497505b",
                      "code_sha256_12": {"causalrca.py": hashlib.sha256(open(f"{REPO}/RCAEval/e2e/causalrca.py", "rb").read()).hexdigest()[:12],
                                         "run_causalrca_permutation.py": hashlib.sha256(open(THIS_FILE, "rb").read()).hexdigest()[:12]},
                      "env": sys.executable, "python": sys.version.split()[0], "argv": sys.argv,
                      "run_started_utc": started, "device": "cpu", "length_min": 20,
                      "scikit_network": __import__("sknetwork").__version__},
           "summary": summary, "cases": rows}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
