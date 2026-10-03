"""Run a metric-ranking RCA method on OpenRCA through the audit wrapper, recording per-query status.

The wrapper is auditstack/scripts/openrca_eval/predict_baro.py (used for the audit BARO rows of
the long table): parquet window (4 h baseline + 30 min either side of the query midpoint), metric
ranking -> first candidate component, reason from the top metric's type, occurrence time = peak
deviation of the top metric. Only the ranking function is swapped:

  baro   predict_baro.baro_inline (unchanged; used to check that this harness reproduces
         full335/preds_baro_<tag>.csv row for row)
  ediag  RCAEval.e2e.e_diagnosis, dataset="openrca-<tag>" (RCAEval preprocess: drop time/constant
         columns, memory to MB), E_ALPHA default 0.01 - same call as the audit 65_e_diagnosis_openrca.py
  simplerca  audit SimpleRCA rewrite (earlier-b scripts/pilot/64_simplerca_rcaeval.py
         simplerca_rank_columns, unchanged; sensitivity control only). Its per-column service parser is
         replaced by the wrapper's component parser (extract_component_from_metric with EXTRACT[system];
         unmapped columns keep their cmdb id), so alerts are aggregated per answer component. Components
         are ordered as SimpleRCA ranks them (-alerts, -tiebreak, name); within a component, columns are
         ordered by (alerted first, -|max(anomal) - mean(normal)| / std(normal)), so the wrapper's top
         metric of the chosen component is its most deviating column (drives reason and time).

The wrapper silently replaces any failure by a fixed per-system default answer (_fallback_json).
This harness records, per query, which path produced the prediction:
  method           method ranking mapped to a candidate component
  no_data          fallback before the method ran (no parquet covering the window / window too short)
  method_error     method raised
  method_empty     method returned no ranking
  no_candidate     ranking had no metric of a candidate component

  rcd    RCAEval.e2e.rcd, ranks precomputed in the rcaeval_rcd env on the inputs this wrapper builds
         (openrca_rcd.py dump / rank; runs/openrca/rcd_ranks/<tag>_seed<s>.json), looked up by query row, input
         sha256 checked; --seed selects the ranks file and names the output rcd-seed<s>_openrca_<tag>

Usage: python openrca_wrapped.py --method {baro,ediag,simplerca,rcd} --tag bank --out_dir DIR [--procs P] [--seed S]
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

A_EVAL = "$HOME/auditstack/scripts/openrca_eval"
REPO = "$HOME/rcaeval_repo"
OPEN_RCA = "$HOME/auditstack/data/OpenRCA/dataset"
B_PILOT = "$HOME/earlier-b/scripts/pilot"
THIS_FILE = os.path.abspath(__file__)
SYSTEMS = {
    "bank": ("Bank", f"{OPEN_RCA}/Bank/query.csv"),
    "mkt1": ("Market_cloudbed-1", f"{OPEN_RCA}/Market/cloudbed-1/query.csv"),
    "mkt2": ("Market_cloudbed-2", f"{OPEN_RCA}/Market/cloudbed-2/query.csv"),
    "tel": ("Telecom", f"{OPEN_RCA}/Telecom/query.csv"),
}
sys.path.insert(0, A_EVAL)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import predict_baro as pb  # noqa: E402

STATE = {}


def simplerca_metric_ranks(bs, data, inject_time, system):
    """SimpleRCA component ranking -> metric ranking (see module docstring)."""
    def comp_of(col):
        c = pb.extract_component_from_metric(col, pb.EXTRACT[system])
        return c if c else col.split("__", 1)[0]

    bs.service_from_col = comp_of
    ranking, _, _ = bs.simplerca_rank_columns(data, inject_time)
    normal = data.loc[data["time"] < inject_time]
    anomal = data.loc[data["time"] >= inject_time]
    per_comp = {}
    for col in data.columns:
        if col == "time":
            continue
        a = np.nan_to_num(normal[col].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(anomal[col].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if a.size < 2 or b.size < 1:
            continue
        mu, sd, bmax = float(a.mean()), float(a.std()), float(b.max())
        z = abs(bmax - mu) / sd if sd > 1e-9 else 0.0
        per_comp.setdefault(comp_of(col), []).append((bmax > mu + 3 * sd, z, col))
    ranks = []
    for comp in ranking:
        ranks += [c for _, _, c in sorted(per_comp.get(comp, []), key=lambda t: (not t[0], -t[1], t[2]))]
    return {"ranks": ranks}


def make_ranker(method, tag, seed=None):
    base = pb.baro_inline
    if method == "rcd":
        pre = json.load(open(f"$AUDIT_ROOT/runs/openrca/rcd_ranks/{tag}_seed{seed}.json"))["rows"]

        def base(data, inject_time):
            r = pre[str(STATE["row"])]
            assert hashlib.sha256(data.to_csv(index=False).encode()).hexdigest() == r["input_sha"], STATE["row"]
            if r["error"]:
                raise RuntimeError("RCD failed in rcaeval_rcd env: " + r["error"][-300:])
            return {"ranks": r["ranks"]}
    if method == "simplerca":
        import importlib.util
        spec = importlib.util.spec_from_file_location("b_simplerca", f"{B_PILOT}/64_simplerca_rcaeval.py")
        bs = importlib.util.module_from_spec(spec)
        sys.path.insert(0, B_PILOT)
        spec.loader.exec_module(bs)
        system = SYSTEMS[tag][0]

        def base(data, inject_time):
            return simplerca_metric_ranks(bs, data, inject_time, system)
    if method == "ediag":
        from RCAEval.e2e import e_diagnosis

        def base(data, inject_time):
            return e_diagnosis(data, inject_time, dataset=f"openrca-{tag}", anomalies=None,
                               dk_select_useful=False, sli=None)

    def ranker(data, inject_time):
        STATE["method_called"] = True
        STATE["n_cols"] = int(data.shape[1] - 1)
        try:
            out = base(data, inject_time=inject_time)
        except Exception as e:
            STATE["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:]
            raise
        ranks = [str(r) for r in out.get("ranks", [])]
        STATE["n_ranks"] = len(ranks)
        STATE["top_metrics"] = ranks[:10]
        return {"ranks": ranks}

    return ranker


def init_worker(method, tag, seed=None):
    pb.baro_inline = make_ranker(method, tag, seed)
    orig_fallback = pb._fallback_json

    def fallback(system_, n_fail, fallback_dt):
        STATE["fallback"] = True
        return orig_fallback(system_, n_fail, fallback_dt)

    pb._fallback_json = fallback


CACHE = {}


def run_query(job):
    i, row, system = job
    STATE.clear()
    STATE["row"] = int(i)
    np.random.seed(1000 + int(i))  # e-Diagnosis bootstraps with np.random; per-query seed, independent of worker
    t0 = time.time()
    pred = pb.predict_one(system, row, CACHE, "$HOME/auditstack/results_1min",
                          "$HOME/auditstack/results_ext", "$HOME/auditstack/results")
    if not STATE.get("fallback"):
        status = "method"
    elif not STATE.get("method_called"):
        status = "no_data"
    elif STATE.get("error"):
        status = "method_error"
    elif not STATE.get("n_ranks"):
        status = "method_empty"
    else:
        status = "no_candidate"
    print(f"row {i} {status} {time.time() - t0:.1f}s", flush=True)
    return pred, {"row": int(i), "task_index": row["task_index"], "status": status,
                  "n_cols": STATE.get("n_cols"), "n_ranks": STATE.get("n_ranks"),
                  "top_metrics": STATE.get("top_metrics"), "error": STATE.get("error"),
                  "wall_s": round(time.time() - t0, 1), "prediction": pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["baro", "ediag", "simplerca", "rcd"])
    ap.add_argument("--tag", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    assert (a.method == "rcd") == (a.seed is not None)
    system, q_path = SYSTEMS[a.tag]

    qdf = pd.read_csv(q_path)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    jobs = [(i, row, system) for i, row in qdf.iterrows()]
    # queries are independent; each worker keeps its own parquet cache
    with Pool(a.procs, initializer=init_worker, initargs=(a.method, a.tag, a.seed)) as pool:
        out = pool.map(run_query, jobs, chunksize=1)
    preds, rows = [o[0] for o in out], [o[1] for o in out]

    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"{a.method}_openrca_{a.tag}" if a.seed is None else f"{a.method}-seed{a.seed}_openrca_{a.tag}"
    pd.DataFrame({"task_index": qdf["task_index"], "prediction": preds}).to_csv(
        os.path.join(a.out_dir, f"preds_{stem}.csv"), index=False)
    src = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]
           for f in (f"{A_EVAL}/predict_baro.py", f"{A_EVAL}/predict_rcaagent_1min.py",
                     f"{REPO}/RCAEval/e2e/__init__.py", f"{B_PILOT}/64_simplerca_rcaeval.py", THIS_FILE)}
    counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    res = {"config": {"method": a.method, "dataset": f"openrca-{a.tag}", "system": system,
                      "input_file": "auditstack results_1min/results_ext/results metric_data.parquet (audit wrapper)",
                      "query_csv": q_path, "code_commit": "rcaeval_repo@497505b; auditstack openrca_eval (A paper)",
                      "code_sha256_12": src, "env": sys.executable, "python": sys.version.split()[0],
                      "argv": sys.argv, "run_started_utc": started, "procs": a.procs, "numpy_seed": "1000 + query row",
                      "e_alpha": float(os.environ.get("E_ALPHA", 0.01)) if a.method == "ediag" else None,
                      "seed": a.seed, "rcd_ranks": (f"runs/openrca/rcd_ranks/{a.tag}_seed{a.seed}.json"
                                                    if a.method == "rcd" else None)},
           "n": len(rows), "status_counts": counts, "wall_s": round(time.time() - t0, 1), "cases": rows}
    with open(os.path.join(a.out_dir, f"{stem}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"method": a.method, "tag": a.tag, "n": len(rows), "status_counts": counts,
                      "wall_s": res["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()
