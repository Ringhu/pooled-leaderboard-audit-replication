"""Compare RE1-OB as-shipped outputs (frozen long table af1088a9) with the duplicate-"time"-column rerun
(runs/re1ob_timefix/). Per method: Acc@1 on the 50 affected / 75 unaffected cases, old vs new, and the number of
cases whose score changed. Read-only; prints a markdown table. Run BEFORE rebuilding the long table."""
import json, os
import pandas as pd
M = os.path.expanduser("$AUDIT_ROOT")
d = pd.read_csv(f"{M}/merged_long.csv")
ob = d[(d.subsystem_id == "RE1::OB") & d.role.isin(["published", "probe"])].copy()
ob["seed"] = ob.seed.fillna(-1).astype(int)
FILES = {("circa", -1): "circa_re1-ob_timefix.json", ("e_diagnosis", -1): "e_diagnosis_re1-ob_timefix.json",
         ("baro", -1): "baro_re1-ob_len20_timefix.json", ("max_z", -1): "zbase_re1-ob_len20_timefix.json",
         ("alertcount", -1): "alertcount_re1-ob_len20_timefix.json", ("cd1min", -1): "cd1min_re1-ob_len20_timefix.json",
         **{("rcd", s): f"rcd_re1-ob_seed{s}_timefix.json" for s in (0, 1, 2)}}
print("| method | seed | old aff | new aff | old unaff | new unaff | aff score changed | unaff score changed | new errors |")
print("|---|---|---|---|---|---|---|---|---|")
for (m, s), fn in FILES.items():
    p = f"{M}/runs/re1ob_timefix/{fn}"
    if not os.path.exists(p):
        print(f"| {m} | {s} | missing |"); continue
    j = {c["case"]: c for c in json.load(open(p))["cases"]}
    o = ob[(ob.method == m) & (ob.seed == s)].set_index("case")
    aff = o.fault.isin(["cpu", "mem"])
    new = pd.Series({c: float(j[c]["acc1"]) if j[c].get("error") in (None, "None") else 0.0 for c in o.index})
    ch = new != o.score
    err = sum(j[c].get("error") not in (None, "None") for c in j)
    print(f"| {m} | {'' if s < 0 else s} | {o.score[aff].mean():.3f} | {new[aff].mean():.3f} | {o.score[~aff].mean():.3f} | "
          f"{new[~aff].mean():.3f} | {int(ch[aff].sum())} | {int(ch[~aff].sum())} | {err} |")
