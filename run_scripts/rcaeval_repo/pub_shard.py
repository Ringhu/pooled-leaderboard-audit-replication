import sys, glob, os, json, time
sys.path.insert(0, "$HOME/rcaeval_repo")
os.chdir("$HOME/rcaeval_repo")
import importlib.util as il
from _baseline_utils import load_case, service_from_col, acc_at_k
_s = il.spec_from_file_location("circa_mod", "RCAEval/e2e/circa.py")
_m = il.module_from_spec(_s); _s.loader.exec_module(_m); circa_fn = _m.circa
ROOTS = {"re1-ob":"data/online-boutique","re1-ss":"data/sock-shop-2","re1-tt":"data/train-ticket"}
ds, fname, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
paths = sorted(glob.glob(os.path.join(ROOTS[ds], "**", fname), recursive=True))[start:end]
rows=[]
for p in paths:
    try:
        data, it, service, fault, sli = load_case(p, length_min=20)
        n=len([c for c in data.columns if c!="time"])
        out=circa_fn(data, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=sli, verbose=False, n_iter=n)
        ranks=out.get("ranks",[])
    except Exception as e:
        print(f"FAIL {p}: {type(e).__name__}: {e}", flush=True); ranks=[]
    rows.append({"case":"/".join(p.split("/")[-3:-1]),"true_service":service,"fault":fault,
                 "top1":service_from_col(ranks[0]) if ranks else None,
                 "acc1":acc_at_k(ranks,service,1) if ranks else 0})
json.dump(rows, open(f"followup_runs/circa_shards_simple/{ds}_{start}_{end}.json","w"))
print("SHARD_DONE", start, end, flush=True)
