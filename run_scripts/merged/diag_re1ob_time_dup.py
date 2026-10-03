"""Diagnostic (2026-09-30): upstream RCAEval commit 21ff8c9 reports that 50 RE1-OB CPU/MEM cases repeat the
"time" header (pandas reads it as "time.1"), which crashed CIRCA and made it fall back to column order.
For every RE1-OB per-case output used in the frozen long table, count (a) cases whose data.csv has the
duplicate column, (b) cases whose ranking equals the input column order (first 10), (c) cases whose
top-1 is "time.1" or contains "time". Read-only; writes nothing except stdout."""
import json, os, glob, sys
import pandas as pd

ROOT = os.path.expanduser("$AUDIT_ROOT")
DATA = os.path.expanduser("~/rcaeval_repo/data/online-boutique")
d = pd.read_csv(f"{ROOT}/merged_long.csv")
ob = d[(d.subsystem_id == "RE1::OB")]
srcs = sorted(ob.source_file.dropna().unique())

hdr = {}
for cdir in glob.glob(f"{DATA}/*/*"):
    f = f"{cdir}/data.csv"
    if os.path.exists(f):
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        key = os.path.relpath(cdir, DATA)  # e.g. adservice_cpu/1
        hdr[key] = cols
dup = {k for k, c in hdr.items() if "time.1" in c}
print(f"RE1-OB cases with data.csv: {len(hdr)}; with duplicate time column: {len(dup)}")
from collections import Counter
print("by fault:", Counter(k.split('/')[0].split('_')[-1] for k in dup))

def case_key(c):
    s = c.get("service"); f = c.get("fault"); r = c.get("rep", c.get("case_idx", c.get("idx")))
    return s, f, r

for src in srcs:
    p = f"{ROOT}/{src}"
    try:
        j = json.load(open(p))
    except Exception as e:
        print(src, "unreadable", e); continue
    cases = j.get("cases", j) if isinstance(j, dict) else j
    if isinstance(cases, dict):
        cases = [dict(v, _k=k) for k, v in cases.items()]
    n = len(cases); n_time_top = n_time_any = n_colorder = 0; n_dupcase = 0; n_colorder_dup = 0; n_timetop_dup = 0
    for c in cases:
        ranks = c.get("ranks") or c.get("ranking") or []
        if not isinstance(ranks, list):
            continue
        k = c.get("case") or c.get("case_id") or c.get("_k") or ""
        # match to data dir
        key = None
        for cand in (k, str(k).replace("online-boutique/", "")):
            if cand in hdr: key = cand
        is_dup = key in dup if key else (c.get("fault") in ("cpu", "mem"))
        n_dupcase += is_dup
        top = str(ranks[0]) if ranks else ""
        if top.startswith("time"): n_time_top += 1; n_timetop_dup += is_dup
        if any(str(x).startswith("time") for x in ranks[:5]): n_time_any += 1
        if key:
            cols = [x for x in hdr[key] if x != "time"]
            if [str(x) for x in ranks[:10]] == cols[:10]:
                n_colorder += 1; n_colorder_dup += is_dup
    print(f"{src}: n={n} dup_cases={n_dupcase} top1_time={n_time_top} (in dup {n_timetop_dup}) time_in_top5={n_time_any} ranks==colorder={n_colorder} (in dup {n_colorder_dup})")
