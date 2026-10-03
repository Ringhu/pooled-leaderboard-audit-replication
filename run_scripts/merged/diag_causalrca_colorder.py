"""Diagnostic (not a main-table run): rerun RE2-TT seed-1 CausalRCA jobs whose ranks equalled the input
column order, with the same setup as run_causalrca_all.py (GPU, ts env + sknet 0.31, seed 1), and record why
the PageRank step fell into causalrca.py's `except Exception` branch: nonzero / NaN counts of the returned
adjacency and the exception PageRank raises on it. Also reruns 2 seed-1 jobs that did not collapse as a control.
Output: work/causalrca_diag/<case>.json
Usage: python diag_causalrca_colorder.py <gpu> <case_dir_name/idx> [...]
"""
import json, os, random, sys, time, traceback
gpu = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
REPO = "$HOME/rcaeval_repo"
OUT = "$AUDIT_ROOT/work/causalrca_diag"
sys.path.insert(0, REPO)
import numpy as np, torch
torch.set_num_threads(1)
from _baseline_utils import load_case
from RCAEval.e2e.causalrca import causalrca
from sknetwork.ranking import PageRank
os.makedirs(OUT, exist_ok=True)
for case in sys.argv[2:]:
    p = f"{REPO}/data/RE2/RE2-TT/{case}/simple_metrics.csv"
    data, it, service, fault, _ = load_case(p, length_min=20)
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    t0 = time.time()
    res = causalrca(data, it, dataset="re2-tt", anomalies=None, dk_select_useful=False, sli=None, verbose=False)
    adj = np.asarray(res["adj"], dtype=float)
    rec = {"case": case, "seed": 1, "gpu": gpu, "wall_s": round(time.time() - t0, 1),
           "ranks_equal_column_order": list(map(str, res["ranks"])) == list(map(str, res["node_names"])),
           "n_nodes": int(adj.shape[0]), "adj_nonzero": int(np.count_nonzero(np.nan_to_num(adj, nan=0.0))),
           "adj_nan": int(np.isnan(adj).sum()), "adj_inf": int(np.isinf(adj).sum())}
    try:
        PageRank().fit_transform(np.abs(adj.T))
        rec["pagerank_exception"] = None
    except Exception as e:
        rec["pagerank_exception"] = "".join(traceback.format_exception_only(type(e), e))[-600:]
    with open(f"{OUT}/{case.replace('/', '__')}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec), flush=True)
