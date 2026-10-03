"""RE1 window check: per-case equality of the 10-min runs with the long table, and 10 vs 20 min AC@1.

Methods: baro (runs/baro/baro_<ds>_len<L>.json) and the three audit probes
(runs/probes/<m>_<ds>_len<L>.json). Long table: raw/earlier_a_followup_data/full_coverage_cd1min_long_CORRECTED.csv.
Also prints the per-case AC@1 of the published-method RE1 runs at 20 min for reference.
"""
import json
import os

import pandas as pd

M = "$AUDIT_ROOT"
LONG = f"{M}/raw/earlier_a_followup_data/full_coverage_cd1min_long_CORRECTED.csv"
SID = {"re1-ob": "rcaeval::OB", "re1-ss": "rcaeval::SS", "re1-tt": "rcaeval::TT"}
LONG_NAME = {"baro": "baro", "zbase": "z_family", "alertcount": "alertcount", "cd1min": "cd1min"}


def path(m, ds, L):
    return f"{M}/runs/baro/baro_{ds}_len{L}.json" if m == "baro" else f"{M}/runs/probes/{m}_{ds}_len{L}.json"


long = pd.read_csv(LONG)
long = long[long.benchmark == "rcaeval"]
rows = []
for m in LONG_NAME:
    for ds in SID:
        rec = {"method": m, "dataset": ds}
        for L in (10, 20):
            p = path(m, ds, L)
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            d = pd.DataFrame(r["cases"])
            rec[f"acc1_len{L}"] = round(d.acc1.mean(), 3)
            rec[f"cov_len{L}"] = int((d.n_ranks > 0).sum())
            if "acc1_rcaeval" in d and d.acc1_rcaeval.notna().any():
                rec[f"scorer_agree_len{L}"] = int((d.acc1 == d.acc1_rcaeval).sum())
            if L == 10:
                lt = long[(long.system_id == SID[ds]) & (long.method == LONG_NAME[m])][["case", "correct"]]
                mm = d.merge(lt, on="case", how="left")
                rec["long_matched"] = int(mm.correct.notna().sum())
                rec["long_equal"] = int((mm.acc1 == mm.correct).sum())
        if "acc1_len10" in rec and "acc1_len20" in rec:
            a = pd.DataFrame(json.load(open(path(m, ds, 10)))["cases"]).set_index("case").acc1
            b = pd.DataFrame(json.load(open(path(m, ds, 20)))["cases"]).set_index("case").acc1
            rec["flip_10to20_lost"] = int(((a == 1) & (b == 0)).sum())
            rec["flip_10to20_gained"] = int(((a == 0) & (b == 1)).sum())
        rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False))
