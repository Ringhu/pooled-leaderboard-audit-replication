"""audit probes (max-|Z|, alert-count, CD-1min) on RCAEval RE1/RE2 at a chosen window length, per-case output.

The probe code is imported unchanged from auditstack/scripts/openrca_eval (the files that produced
the long-table RE1 rows):
  zbase       run_rcaeval_zbase.run_one      (predict_zbase_universal.zbase_universal)
  alertcount  run_rcaeval_alertcount.run_one (predict_alertcount.alertcount_universal, threshold 3.0)
  cd1min      smoke_cd_causal_adapters: load_rcaeval_case + window_rcaeval + method_predict("cd1min_adapt",
              z_threshold 3.0, pct_threshold 5.0, no graph)
Those files hard-code a 10-minute window (5 min either side of injection, 1 row per second). The only
change here is the window length: --length 10 must reproduce the long table; --length 20 is the RCAEval
main.py default used by the published-method runs (CIRCA, e-Diagnosis, RCD, BARO len20).

Input (default): RE1-OB data.csv (the only file); RE1-SS/TT simple_data.csv; RE2 simple_metrics.csv.
--input-file overrides (RE1-SS/TT data.csv, RE2 metrics.csv: input-representation contrast). tdelta = 0.
RE2: numbered case directories only (RE2-OB also has a multi-source-data example directory).
Scoring: each probe's own acc_at_k (as in the long table) -> acc{1,3,5}; also the RCAEval
_baseline_utils scorer on the same ranks (acc1_rcaeval) as a cross-check. No ranks -> 0.

Usage: python run_re1_probes.py --method zbase --dataset re1-ob --length 20 --out OUT.json [--input-file F]
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
CWD0 = os.getcwd()
A_EVAL = "$HOME/auditstack/scripts/openrca_eval"
REPO = "$HOME/rcaeval_repo"
sys.path.insert(0, REPO)
sys.path.insert(0, A_EVAL)  # audit probe files take precedence

import pandas as pd  # noqa: E402

from _baseline_utils import acc_at_k as rcaeval_acc  # noqa: E402

ROOTS = {"re1-ob": ("online-boutique", "data.csv"), "re1-ss": ("sock-shop-2", "simple_data.csv"),
         "re1-tt": ("train-ticket", "simple_data.csv"), "re2-ob": ("RE2/RE2-OB", "simple_metrics.csv"),
         "re2-ss": ("RE2/RE2-SS", "simple_metrics.csv"), "re2-tt": ("RE2/RE2-TT", "simple_metrics.csv")}
# SLI hint as in the audit run scripts (by application)
SLI_HINT = {"ob": "frontend_latency", "ss": "front-end_cpu", "tt": "ts-ui-dashboard_latency"}
SRC_FILES = {"zbase": ["run_rcaeval_zbase.py", "predict_zbase_universal.py"],
             "alertcount": ["run_rcaeval_alertcount.py", "predict_alertcount.py"],
             "cd1min": ["smoke_cd_causal_adapters.py"]}


def window(data, inject_time, length_min):
    half = length_min * 60 // 2
    normal = data[data["time"] < inject_time].tail(half)
    anomal = data[data["time"] >= inject_time].head(half)
    return pd.concat([normal, anomal], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(SRC_FILES))
    ap.add_argument("--dataset", required=True, choices=sorted(ROOTS))
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-file", default=None)
    a = ap.parse_args()
    a.out = os.path.join(CWD0, a.out)
    ds_name, fname = ROOTS[a.dataset]
    fname = a.input_file or fname
    sli_hint = SLI_HINT[a.dataset[-2:]]
    root = f"{REPO}/data/{ds_name}"
    case_dirs = sorted(d for d in glob.glob(os.path.join(root, "*/*/"))
                       if os.path.basename(d.rstrip("/")).isdigit() and os.path.exists(os.path.join(d, fname)))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()

    if a.method in ("zbase", "alertcount"):
        import run_rcaeval_alertcount as ra
        import run_rcaeval_zbase as rz
        mod = rz if a.method == "zbase" else ra
        fn = rz.baro if a.method == "zbase" else ra.alertcount_universal
        mod.DATA_FILE = fname
        mod.args_dataset = ds_name
        mod.DATASET_ROOTS[ds_name] = root
        mod.window_data = lambda data, inject_time, length_min=10: window(data, inject_time, a.length)
        acc = mod.acc_at_k
        to_service = mod.extract_service
        # run_one does not return the ranks; capture them from the predictor call
        captured = {}
        orig = fn

        def fn(data, inject_time, **kw):  # noqa: F811
            out = orig(data, inject_time, **kw)
            captured["ranks"] = [str(x) for x in out.get("ranks", [])]
            captured["n_cols"] = int(data.shape[1] - 1)
            return out

        def predict(cd):
            captured.clear()
            r = mod.run_one(a.method, fn, cd, 0, sli_hint)
            if "error" in r:
                raise RuntimeError(r["error"])
            return r["true_service"], r["fault"], captured.get("n_cols"), captured.get("ranks", [])
    else:
        import smoke_cd_causal_adapters as sc
        sc.CD_DATA_FILE = fname
        acc = sc.acc_at_k
        to_service = str

        def predict(cd):
            data, inject_time, true_service, fault = sc.load_rcaeval_case(cd)
            w = window(data, inject_time, a.length)
            pred = sc.method_predict("cd1min_adapt", w, float(inject_time), sc.extract_rcaeval_component,
                                     graph=None, z_threshold=3.0, pct_threshold=5.0)
            if pred["error"] and pred["error"] != "no_component_passed_thresholds":
                raise RuntimeError(pred["error"])
            return true_service, fault, int(w.shape[1] - 1), [str(x) for x in pred["ranks"]]

    rows = []
    for cd in case_dirs:
        rec = {"case": os.path.relpath(cd, root)}
        try:
            svc, fault, n_cols, ranks = predict(cd)
            rec.update(true_service=svc, fault=fault, n_cols=n_cols, error=None)
        except Exception as e:
            ranks = []
            rec["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:]
        rec["ranks"] = ranks[:20]
        rec["n_ranks"] = len(ranks)
        rec["top1"] = to_service(ranks[0]) if ranks else None
        truth = rec.get("true_service")
        for k in (1, 3, 5):
            rec[f"acc{k}"] = float(acc(ranks, truth, k)) if ranks and truth else 0.0
        rec["acc1_rcaeval"] = float(rcaeval_acc(ranks, truth, 1)) if ranks and truth and a.method != "cd1min" else None
        rows.append(rec)

    n = len(rows)
    src = {f: hashlib.sha256(open(f"{A_EVAL}/{f}", "rb").read()).hexdigest()[:12] for f in SRC_FILES[a.method]}
    src[os.path.basename(THIS_FILE)] = hashlib.sha256(open(THIS_FILE, "rb").read()).hexdigest()[:12]
    res = {
        "config": {"method": a.method, "dataset": a.dataset, "input_file": fname, "length_min": a.length,
                   "tdelta": 0, "code_commit": "auditstack openrca_eval (A paper), rcaeval_repo@497505b",
                   "code_sha256_12": src, "env": sys.executable, "python": sys.version.split()[0],
                   "argv": sys.argv, "run_started_utc": started},
        "method": a.method, "dataset": a.dataset, "file": fname, "n": n,
        **{f"acc{k}": round(sum(r[f"acc{k}"] for r in rows) / n, 4) for k in (1, 3, 5)},
        "coverage": sum(r["n_ranks"] > 0 for r in rows), "n_errors": sum(r["error"] is not None for r in rows),
        "wall_s": round(time.time() - t0, 1), "cases": rows,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "config")}), flush=True)


if __name__ == "__main__":
    main()
