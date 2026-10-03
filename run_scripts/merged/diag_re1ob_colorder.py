"""Diagnostic (2026-09-30): for each RE1-OB per-case output, test whether the ranking is in input column
order (positions in data.csv header strictly increasing over the whole ranking). Column positions are
taken from rep 1 of the same service_fault directory. Read-only."""
import json, os, sys
import pandas as pd
from collections import defaultdict
ROOT = os.path.expanduser("$AUDIT_ROOT")
DATA = os.path.expanduser("~/rcaeval_repo/data/online-boutique")
d = pd.read_csv(f"{ROOT}/merged_long.csv")
srcs = sorted(d[d.subsystem_id == "RE1::OB"].source_file.dropna().unique())
pos_cache = {}
def positions(sf):
    if sf not in pos_cache:
        cols = pd.read_csv(f"{DATA}/{sf}/1/data.csv", nrows=0).columns.tolist()
        pos_cache[sf] = {c: i for i, c in enumerate(cols)}
    return pos_cache[sf]
for src in srcs:
    j = json.load(open(f"{ROOT}/{src}"))
    cases = j["cases"] if isinstance(j, dict) and "cases" in j else j
    if isinstance(cases, dict): cases = list(cases.values())
    cnt = defaultdict(lambda: [0, 0])
    for c in cases:
        r = c.get("ranks") or []
        sf = f"{c.get('service')}_{c.get('fault')}"
        if not r or not os.path.isdir(f"{DATA}/{sf}"): continue
        p = positions(sf)
        idx = [p.get(x) for x in r]
        mono = len(r) >= 5 and all(i is not None for i in idx) and all(a < b for a, b in zip(idx, idx[1:]))
        cnt[c.get("fault")][0] += mono; cnt[c.get("fault")][1] += 1
    print(src, {k: f"{v[0]}/{v[1]}" for k, v in sorted(cnt.items())})
