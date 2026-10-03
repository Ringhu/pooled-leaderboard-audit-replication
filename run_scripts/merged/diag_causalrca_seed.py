"""Diagnostic (not a main-table run) for CausalRCA seed-to-seed variation.
For each (dataset, case, seed) given: rerun with exactly the setup of run_causalrca_all.py (GPU, ts env + sknet 0.31,
random/numpy/torch seeded before the call) and record
  - same_as_main_run: full top-20 ranks equal to the main run's job file (same seed) -> is a seeded run reproducible?
  - n_edges: nonzero entries of the thresholded learned adjacency (causalrca.py sets |w| < 0.3 to 0)
  - n_leading_not_column_order: smallest k such that ranks[k:] is in input column order (nodes whose PageRank
    scores tie keep input column order, because the sort in causalrca.py is stable)
Output: work/causalrca_diag_seed/<ds>__<case>__seed<s>.json
Usage: python diag_causalrca_seed.py <gpu> <ds>:<case>:<seed> [...]
"""
import json, os, random, sys, time
gpu = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
REPO = "$HOME/rcaeval_repo"
M = "$AUDIT_ROOT"
OUT = f"{M}/work/causalrca_diag_seed"
sys.path.insert(0, REPO)
import numpy as np, torch
torch.set_num_threads(1)
from _baseline_utils import load_case
from RCAEval.e2e.causalrca import causalrca
from sknetwork.ranking import PageRank
SOURCES = {"re1-ob": ("online-boutique", "data.csv"), "re1-ss": ("sock-shop-2", "simple_data.csv"),
           "re1-tt": ("train-ticket", "simple_data.csv"), "re2-ob": ("RE2/RE2-OB", "simple_metrics.csv"),
           "re2-ss": ("RE2/RE2-SS", "simple_metrics.csv"), "re2-tt": ("RE2/RE2-TT", "simple_metrics.csv")}
os.makedirs(OUT, exist_ok=True)
for spec in sys.argv[2:]:
    ds, case, seed = spec.split(":"); seed = int(seed)
    root, fname = SOURCES[ds]
    p = f"{REPO}/data/{root}/{case}/{fname}"
    data, it, service, fault, _ = load_case(p, length_min=20)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    res = causalrca(data, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=None, verbose=False)
    ranks = [str(r) for r in res["ranks"]]
    adj = np.asarray(res["adj"], dtype=float)
    n_edges = int(np.count_nonzero(adj))
    main = json.load(open(f"{M}/work/causalrca_jobs_gpu/{ds}/seed{seed}/{case.replace('/', '__')}.json"))
    names = [str(n) for n in res["node_names"]]
    rest = [n for n in names if n in set(ranks)]
    k = 0
    while k < len(ranks) and ranks[k:] != [n for n in rest if n in set(ranks[k:])]:
        k += 1
    rec = {"ds": ds, "case": case, "seed": seed, "gpu": gpu, "wall_s": round(time.time() - t0, 1),
           "n_nodes": len(names), "n_edges": n_edges, "ranks_top20": ranks[:20],
           "main_run_top20": main["ranks"], "same_as_main_run": ranks[:20] == main["ranks"],
           "n_leading_not_column_order": k}
    with open(f"{OUT}/{ds}__{case.replace('/', '__')}__seed{seed}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({x: v for x, v in rec.items() if "top20" not in x}), flush=True)
