"""Decisive test: run RCAEval BARO on RE1-SS/TT using simple_data.csv instead of data.csv."""
import sys, glob, os, json
from datetime import datetime
sys.path.insert(0, "$HOME/rcaeval_repo")
os.chdir("$HOME/rcaeval_repo")
from _baseline_utils import load_case, service_from_col, acc_at_k
import importlib.util as _il
_spec=_il.spec_from_file_location("baro_mod","$HOME/rcaeval_repo/RCAEval/e2e/baro.py")
_m=_il.module_from_spec(_spec); _spec.loader.exec_module(_m); baro=_m.baro
from tqdm import tqdm

ROOTS = {"re1-ss": "data/sock-shop-2", "re1-tt": "data/train-ticket", "re1-ob": "data/online-boutique"}

def run(ds, fname):
    paths = sorted(glob.glob(os.path.join(ROOTS[ds], "**", fname), recursive=True))
    if not paths:
        return None
    hits1 = 0; avg5_sum = 0.0; n = 0
    for p in tqdm(paths, desc=f"{ds}/{fname}", file=sys.stderr):
        try:
            data, inject_time, service, fault, sli = load_case(p, length_min=20)
            out = baro(data, inject_time=inject_time, dataset=ds, sli=sli,
                       anomalies=None, dk_select_useful=False)
            ranks = [service_from_col(c) for c in out["ranks"]]
            dedup = []
            for r in ranks:
                if r not in dedup: dedup.append(r)
            hits1 += int(dedup[:1] == [service])
            avg5_sum += sum(int(service in dedup[:k]) for k in range(1, 6)) / 5.0
            n += 1
        except Exception as e:
            print(f"FAIL {p}: {type(e).__name__}: {e}", file=sys.stderr)
    return {"dataset": ds, "file": fname, "n": n,
            "acc@1": round(hits1 / n, 4), "avg@5": round(avg5_sum / n, 4)}

results = []
for ds, fname in [("re1-ss", "data.csv"), ("re1-ss", "simple_data.csv"),
                  ("re1-tt", "data.csv"), ("re1-tt", "simple_data.csv")]:
    r = run(ds, fname)
    if r: results.append(r); print(json.dumps(r), flush=True)
json.dump(results, open("$HOME/rcaeval_repo/followup_runs/baro_simple_vs_raw.json", "w"), indent=1)
