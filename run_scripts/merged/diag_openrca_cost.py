"""Cost probe (diagnostic only, not registered): per-query runtime of CIRCA / RCD / CausalRCA on OpenRCA inputs
built by the audit wrapper (the same windows BARO and e-Diagnosis rank in the long table).

  dump  (auditstack env): for each system, the query whose BARO input width (runs/openrca/baro_openrca_<tag>.json
        n_cols) is closest to the system median; runs the wrapper with a capturing ranker and writes the ranker's
        input (data incl. time column, inject_time) to OUT/<tag>_row<i>.csv + .json. The prediction is discarded.
  time  (method env): runs one method on one dumped input, writes OUT/time_<method>_<tag>.json with wall seconds,
        input shape, number of ranks, error. Calls mirror the long-table runners:
          circa      RCAEval.e2e.circa, dk_select_useful=False, after the audit OpenRCA column-name shim
                     (circa_schema_shim.repair_openrca_columns; RHT parses service/metric from the name)
          rcd        RCAEval.e2e.rcd, dk_select_useful=False, n_iter=n_cols, seed 0 (as run_rcd_re1.py)
          causalrca  RCAEval.e2e.causalrca, seeds 0 (as run_causalrca_all.py), CUDA if visible
Usage:
  python diag_openrca_cost.py dump OUT [N_RANDOM]
  python diag_openrca_cost.py time OUT METHOD TAG
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

A_EVAL = "$HOME/auditstack/scripts/openrca_eval"
REPO = "$HOME/rcaeval_repo"
M = "$AUDIT_ROOT"
sys.path.insert(0, A_EVAL)
sys.path.insert(0, REPO)
TAGS = ["bank", "mkt1", "mkt2", "tel"]


def dump(out, n_random=0):
    """n_random = 0: the median-width query per system; > 0: n_random queries per system drawn with
    np.random.default_rng(20260929) (written under OUT/sample/)."""
    sys.path.insert(0, f"{M}/scripts")
    import openrca_wrapped as w
    import predict_baro as pb
    if n_random:
        out = f"{out}/sample"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20260929)
    for tag in TAGS:
        cases = json.load(open(f"{M}/runs/openrca/baro_openrca_{tag}.json"))["cases"]
        med = np.median([c["n_cols"] for c in cases])
        picks = ([cases[i] for i in sorted(rng.choice(len(cases), n_random, replace=False))] if n_random
                 else [min(cases, key=lambda c: (abs(c["n_cols"] - med), c["row"]))])
        for pick in picks:
            dump_one(out, w, pb, tag, pick, med)


def dump_one(out, w, pb, tag, pick, med):
    system, q_path = w.SYSTEMS[tag]
    row = pd.read_csv(q_path).iloc[pick["row"]]
    got = {}

    def capture(data, inject_time):
        got["data"], got["it"] = data.copy(), inject_time
        return {"ranks": []}

    pb.baro_inline = capture
    pb.predict_one(system, row, {}, "$HOME/auditstack/results_1min",
                   "$HOME/auditstack/results_ext", "$HOME/auditstack/results")
    stem = f"{out}/{tag}_row{pick['row']}"
    got["data"].to_csv(stem + ".csv", index=False)
    json.dump(dict(tag=tag, row=pick["row"], task_index=row["task_index"], inject_time=int(got["it"]),
                   shape=list(got["data"].shape), median_n_cols=med, baro_n_cols=pick["n_cols"]),
              open(stem + ".json", "w"), indent=1)
    print(tag, pick["row"], got["data"].shape, flush=True)


def run_time(out, method, tag):
    """tag = system tag (single dumped query) or <tag>_row<i> (a sampled query)."""
    stem = tag if "_row" in tag else [f[:-5] for f in os.listdir(out) if f.startswith(f"{tag}_row") and f.endswith(".json")][0]
    meta = json.load(open(f"{out}/{stem}.json"))
    data = pd.read_csv(f"{out}/{stem}.csv")
    it, ds = meta["inject_time"], f"openrca-{meta['tag']}"
    rec = dict(method=method, tag=tag, row=meta["row"], shape=list(data.shape), python=sys.version.split()[0],
               env=sys.executable, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
               omp=os.environ.get("OMP_NUM_THREADS"), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    t0 = time.time()
    try:
        if method == "circa":
            from circa_schema_shim import repair_openrca_columns
            from RCAEval.e2e.circa import circa
            renamed, _ = repair_openrca_columns(data)
            res = circa(renamed, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=None, verbose=False)
        elif method == "rcd":
            from RCAEval.e2e.rcd import rcd
            res = rcd(data, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=None, verbose=False,
                      n_iter=len(data.columns) - 1, seed=0)
        elif method == "causalrca":
            import random
            import torch
            torch.set_num_threads(1)
            random.seed(0), np.random.seed(0), torch.manual_seed(0)
            rec["cuda"] = torch.cuda.is_available()
            from RCAEval.e2e.causalrca import causalrca
            res = causalrca(data, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=None, verbose=False)
        rec.update(n_ranks=len(res.get("ranks", [])), top3=[str(r) for r in res.get("ranks", [])[:3]], error=None)
    except Exception as e:
        rec.update(n_ranks=None, error="".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:])
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open(f"{out}/time_{method}_{tag}.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("method", "tag", "shape", "wall_s", "n_ranks")}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        run_time(*sys.argv[2:5])
