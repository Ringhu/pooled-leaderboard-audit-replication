"""Rescore author-repo SimpleRCA/BARO RE1 outputs under the RCAEval service rule
(service = name.replace("-db", ""), dedup in rank order) and align per case with the
canonical audit long table (BARO etc.)."""
import csv, json, sys
from collections import defaultdict

R = "$AUDIT_ROOT/runs/simplerca_author"
LONG = "$AUDIT_ROOT/raw/earlier_a_followup_data/full_coverage_cd1min_long_CORRECTED.csv"
SYS = {"re1_ob": "re1-ob", "re1_ss": "re1-ss", "re1_tt": "re1-tt"}

long = defaultdict(dict)
methods = set()
for r in csv.DictReader(open(LONG)):
    if r["benchmark"] == "rcaeval":
        long[r["system"]][(r["case"], r["method"])] = float(r["correct"])
        methods.add(r["method"])
systems = sorted(long)
print("long-table rcaeval systems:", systems, "methods:", sorted(methods))
ex = next(iter(long[systems[0]]))
print("example long key:", ex)

def svc(n):
    return None if n is None else n.replace("-db", "")

for ds in ("re1_ob", "re1_ss", "re1_tt"):
    for algo in ("simplerca", "baro"):
        d = json.load(open(f"{R}/{algo}_author_{ds}_win20.json"))
        n = len(d["cases"])
        exact = sum(c["hits"][0] for c in d["cases"]) / n
        folded = []
        for c in d["cases"]:
            seen = []
            for a in c["answers"]:
                s = svc(a["name"])
                if s not in seen:
                    seen.append(s)
            folded.append((c["datapack"], c["label"], seen))
        acc1 = sum(s[:1] == [lab] for _, lab, s in folded) / n
        acc3 = sum(lab in s[:3] for _, lab, s in folded) / n
        print(f"{algo:9s} {ds}: exact-match T1={exact:.3f} | RCAEval-rule T1={acc1:.3f} T3={acc3:.3f} (n={n})")
        if algo == "simplerca":
            sysname = [s for s in systems if s.replace("-", "_").endswith(ds.split("_")[1]) or ds.split("_")[1] in s]
            print("   matching long-table system:", sysname)
