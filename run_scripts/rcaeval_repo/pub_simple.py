"""Rerun published methods (e-Diagnosis, CIRCA) on simple_data.csv for RE1-SS/TT."""
import sys, glob, os, json, time
sys.path.insert(0, "$HOME/rcaeval_repo")
os.chdir("$HOME/rcaeval_repo")
import importlib.util as il
from _baseline_utils import load_case, service_from_col, acc_at_k
from RCAEval.e2e import e_diagnosis
_s = il.spec_from_file_location("circa_mod", "RCAEval/e2e/circa.py")
_m = il.module_from_spec(_s); _s.loader.exec_module(_m); circa_fn = _m.circa

ROOTS = {"re1-ob": "data/online-boutique", "re1-ss": "data/sock-shop-2", "re1-tt": "data/train-ticket"}
METH = {"e_diagnosis": e_diagnosis, "circa": circa_fn}
which, ds, fname = sys.argv[1], sys.argv[2], sys.argv[3]
fn = METH[which]
paths = sorted(glob.glob(os.path.join(ROOTS[ds], "**", fname), recursive=True))
rows = []; t0 = time.time()
for p in paths:
    try:
        data, it, service, fault, sli = load_case(p, length_min=20)
        n = len([c for c in data.columns if c != "time"])
        out = fn(data, it, dataset=ds, anomalies=None, dk_select_useful=False,
                 sli=sli, verbose=False, n_iter=n)
        ranks = out.get("ranks", [])
    except Exception as e:
        print(f"FAIL {p}: {type(e).__name__}: {e}", flush=True); ranks = []
    rows.append({"case": "/".join(p.split("/")[-3:-1]), "true_service": service, "fault": fault,
                 "top1": service_from_col(ranks[0]) if ranks else None,
                 "acc1": acc_at_k(ranks, service, 1) if ranks else 0})
acc = sum(r["acc1"] for r in rows)/len(rows)
import collections
modal = collections.Counter(r["top1"] for r in rows).most_common(1)[0]
res = {"method": which, "dataset": ds, "file": fname, "n": len(rows), "acc1": round(acc,4),
       "modal_pred": modal[0], "modal_freq": round(modal[1]/len(rows),3),
       "n_distinct_top1": len(set(r["top1"] for r in rows)), "wall_s": round(time.time()-t0,1), "cases": rows}
json.dump(res, open(f"followup_runs/{which}_{ds}_{fname.replace('.csv','')}.json","w"))
print(json.dumps({k:v for k,v in res.items() if k!="cases"}), flush=True)
