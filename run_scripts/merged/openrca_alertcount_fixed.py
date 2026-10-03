"""audit alert-count probe on OpenRCA with its one-line call bug fixed, recording per-query status.

auditstack/scripts/openrca_eval/run_openrca_alertcount.py builds the per-metric service mapper as
    functools.partial(extract_component_from_metric, extract=extract)
but predict_baro.extract_component_from_metric's parameter is named `extract_func`, so every call
raises TypeError inside alertcount_universal; predict_one catches it and returns the fixed per-system
default answer (_fallback_json). All 335 rows of preds_alertcount_<tag>.csv are that default answer.

Fix (only change): expose extract_component_from_metric under the keyword the partial uses. The audit
files are imported, not edited. Status labels as in openrca_wrapped.py.

Usage: python openrca_alertcount_fixed.py --tag bank --out_dir DIR [--unfixed]
  --unfixed  run without the fix (checks that this harness reproduces preds_alertcount_<tag>.csv)
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback

A_EVAL = "$HOME/auditstack/scripts/openrca_eval"
OPEN_RCA = "$HOME/auditstack/data/OpenRCA/dataset"
THIS_FILE = os.path.abspath(__file__)
SYSTEMS = {
    "bank": ("Bank", f"{OPEN_RCA}/Bank/query.csv"),
    "mkt1": ("Market_cloudbed-1", f"{OPEN_RCA}/Market/cloudbed-1/query.csv"),
    "mkt2": ("Market_cloudbed-2", f"{OPEN_RCA}/Market/cloudbed-2/query.csv"),
    "tel": ("Telecom", f"{OPEN_RCA}/Telecom/query.csv"),
}
sys.path.insert(0, A_EVAL)

import pandas as pd  # noqa: E402

import predict_baro as pb  # noqa: E402
import run_openrca_alertcount as ac  # noqa: E402

STATE = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--unfixed", action="store_true")
    a = ap.parse_args()
    system, q_path = SYSTEMS[a.tag]

    if not a.unfixed:
        ac.extract_component_from_metric = lambda metric_col, extract: pb.extract_component_from_metric(metric_col, extract)

    orig_rank = ac.alertcount_universal

    def ranker(data, inject_time, **kw):
        STATE["method_called"] = True
        STATE["n_cols"] = int(data.shape[1] - 1)
        try:
            out = orig_rank(data, inject_time, **kw)
        except Exception as e:
            STATE["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:]
            raise
        ranks = [str(r) for r in out.get("ranks", [])]
        STATE["n_ranks"] = len(ranks)
        STATE["top_metrics"] = ranks[:10]
        return out

    ac.alertcount_universal = ranker
    orig_fallback = ac._fallback_json

    def fallback(system_, n_fail, fallback_dt):
        STATE["fallback"] = True
        return orig_fallback(system_, n_fail, fallback_dt)

    ac._fallback_json = fallback

    qdf = pd.read_csv(q_path)
    cache, preds, rows = {}, [], []
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    for i, row in qdf.iterrows():
        STATE.clear()
        pred = ac.predict_one(system, row, cache, "$HOME/auditstack/results_1min",
                              "$HOME/auditstack/results_ext", "$HOME/auditstack/results")
        preds.append(pred)
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
        rows.append({"row": int(i), "task_index": row["task_index"], "status": status,
                     "n_cols": STATE.get("n_cols"), "n_ranks": STATE.get("n_ranks"),
                     "top_metrics": STATE.get("top_metrics"), "error": STATE.get("error"), "prediction": pred})

    os.makedirs(a.out_dir, exist_ok=True)
    stem = f"alertcount{'-unfixed' if a.unfixed else ''}_openrca_{a.tag}"
    pd.DataFrame({"task_index": qdf["task_index"], "prediction": preds}).to_csv(
        os.path.join(a.out_dir, f"preds_{stem}.csv"), index=False)
    src = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]
           for f in (f"{A_EVAL}/run_openrca_alertcount.py", f"{A_EVAL}/predict_alertcount.py",
                     f"{A_EVAL}/predict_baro.py", f"{A_EVAL}/predict_rcaagent_1min.py", THIS_FILE)}
    counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    res = {"config": {"method": "alertcount", "dataset": f"openrca-{a.tag}", "system": system,
                      "variant": "unfixed (audit as run)" if a.unfixed else "call bug fixed",
                      "input_file": "auditstack results_1min/results_ext/results metric_data.parquet (audit wrapper)",
                      "query_csv": q_path, "code_commit": "auditstack openrca_eval (A paper)",
                      "code_sha256_12": src, "env": sys.executable, "python": sys.version.split()[0],
                      "argv": sys.argv, "run_started_utc": started},
           "n": len(rows), "status_counts": counts, "wall_s": round(time.time() - t0, 1), "cases": rows}
    with open(os.path.join(a.out_dir, f"{stem}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"method": stem, "n": len(rows), "status_counts": counts, "wall_s": res["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()
