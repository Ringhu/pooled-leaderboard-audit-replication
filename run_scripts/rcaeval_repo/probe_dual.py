"""Run BARO + max|Z| + alert-count on RE1-SS/TT under BOTH data.csv and simple_data.csv."""
import sys, glob, os, json
sys.path.insert(0, "$HOME/rcaeval_repo")
os.chdir("$HOME/rcaeval_repo")
import importlib.util as il
from _baseline_utils import load_case, service_from_col, acc_at_k
from predict_zbase_univ import zbase_universal
from predict_alertcount import alertcount_universal
_s = il.spec_from_file_location("baro_mod", "RCAEval/e2e/baro.py")
_m = il.module_from_spec(_s); _s.loader.exec_module(_m); baro_fn = _m.baro

ROOTS = {"re1-ob": "data/online-boutique", "re1-ss": "data/sock-shop-2", "re1-tt": "data/train-ticket"}

def methods(data, inject_time, ds, sli):
    return {
        "baro":       baro_fn(data, inject_time=inject_time, dataset=ds, sli=sli, dk_select_useful=False)["ranks"],
        "maxz":       zbase_universal(data, inject_time)["ranks"],
        "alertcount": alertcount_universal(data, inject_time, extract_service=service_from_col)["ranks"],
    }

out = []
for ds in ["re1-ob", "re1-ss", "re1-tt"]:
    for fname in ["data.csv", "simple_data.csv"]:
        paths = sorted(glob.glob(os.path.join(ROOTS[ds], "**", fname), recursive=True))
        if not paths:
            continue
        hits = {m: 0 for m in ["baro", "maxz", "alertcount"]}; n = 0
        for p in paths:
            try:
                data, it, service, fault, sli = load_case(p, length_min=20)
                for m, ranks in methods(data, it, ds, sli).items():
                    hits[m] += acc_at_k(ranks, service, 1)
                n += 1
            except Exception as e:
                print(f"FAIL {p}: {type(e).__name__}: {e}", file=sys.stderr)
        rec = {"dataset": ds, "file": fname, "n": n,
               **{f"acc1_{m}": round(hits[m]/n, 4) for m in hits}}
        out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open("followup_runs/probe_dual_input.json", "w"), indent=1)
print("DONE")
