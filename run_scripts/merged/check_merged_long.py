"""Consistency checks for merged_long.csv against its sources.

1. every source JSON that stores a run-level acc1 (runs/**): mean(score) over its rows == stored acc1
2. RE1 10-min rows (baro, max_z, alertcount, cd1min) == A long table `correct`, case by case
3. OpenRCA audit methods: score == A long table `correct` where valid_output = 1; number of zeroed rows
4. PetShop audit methods == A long table `correct`
5. one row per (case_id, method, seed) within the main-table roles; main-table case sets identical across methods
Usage: python check_merged_long.py [merged_long.csv]
"""
import json
import sys

import pandas as pd

M = "$AUDIT_ROOT"
path = sys.argv[1] if len(sys.argv) > 1 else f"{M}/merged_long.csv"
d = pd.read_csv(path, keep_default_na=False, dtype={"seed": str, "window_min": str})
A = pd.read_csv(f"{M}/raw/earlier_a_followup_data/full_coverage_cd1min_long_CORRECTED.csv")
AMAP = {"baro": "baro", "max_z": "z_family", "alertcount": "alertcount", "cd1min": "cd1min"}
ok = True


def report(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 1
bad = []
for src, g in d[d.source_file.str.startswith("runs/") & d.source_file.str.endswith(".json")
                & ~d.source_file.str.startswith("runs/re1ob_timefix/")].groupby("source_file"):  # timefix files: check 7
    j = json.load(open(f"{M}/{src}"))
    if "acc1" in j:
        if abs(g.score.mean() - j["acc1"]) > 5e-4 or len(g) != j["n"]:
            bad.append((src, g.score.mean(), j["acc1"], len(g), j["n"]))
report("1 run-level acc1 == mean(score)", not bad, f"{len(bad)} mismatches {bad[:3]}")

# 2
a = A[A.benchmark == "rcaeval"].assign(sid=lambda x: x.system_id.str.replace("rcaeval::", "RE1::"))
w = d[(d.release == "RE1") & (d.role == "diagnostic_window")]
n_eq = n_tot = 0
for m, am in AMAP.items():
    x = w[w.method == m].merge(a[a.method == am], left_on=["subsystem_id", "case"], right_on=["sid", "case"])
    n_tot += len(x)
    n_eq += int((x.score == x.correct).sum())
report("2 RE1 10-min rows == A long table", n_eq == n_tot == 1500, f"{n_eq}/{n_tot}")

# 3
o = d[(d.family == "openrca") & d.impl_source.str.startswith(("RCAEval baro via", "audit CD-1min", "audit probe"))
      & d.role.isin(["published", "probe"]) & d.method.isin(["baro", "cd1min", "max_z"])]
ao = A[A.benchmark == "openrca"]
x = o.assign(am=o.method.map(AMAP)).merge(ao, left_on=["subsystem", "case", "q_idx", "am"],
                                          right_on=["system", "case", "q_idx", "method"])
v = x[x.valid_output == 1]
report("3 OpenRCA A methods == A long table (valid rows)", len(x) == 1005 and (v.score == v.correct).all(),
       f"rows {len(x)}; zeroed default answers {int((x.valid_output == 0).sum())} "
       f"(by method {x[x.valid_output == 0].method_x.value_counts().to_dict()})")

# 4
p = d[(d.family == "petshop") & d.impl_source.str.startswith("audit")]
ap = A[A.benchmark == "petshop"]
x = p.assign(am=p.method.map(AMAP)).merge(ap, left_on=["subsystem", "case", "am"], right_on=["system", "case", "method"])
report("4 PetShop A methods == A long table", len(x) == 272 and (x.score == x.correct).all(), f"{len(x)} rows")

# 5
main = d[d.role.isin(["published", "probe", "shipped_baseline"])]
dup = main.duplicated(["case_id", "method", "seed"]).sum()
report("5a unique (case_id, method, seed) in main table", dup == 0, f"dups={dup}")
bad = []
for sid, g in main.groupby("subsystem_id"):
    sets = g.groupby(["method", "seed"]).case_id.apply(frozenset)
    if sets.nunique() != 1:
        bad.append(sid)
report("5b same case set for every method within a subsystem", not bad, str(bad))

# 6 OpenRCA RCD (user ruling 2026-09-29): 3 seeds x 335 queries, score == scores_long.csv where the method answered
r = d[(d.family == "openrca") & (d.method == "rcd")]
sl = pd.read_csv(f"{M}/runs/openrca/scores_long.csv")
sl = sl[sl.method.str.startswith("rcd-seed")].assign(seed=lambda z: z.method.str[-1].astype(int))
x = r.assign(seed=r.seed.astype(int)).merge(sl, left_on=["subsystem", "case", "q_idx", "seed"],
                                            right_on=["tag", "case", "q_idx", "seed"])
report("6 OpenRCA RCD rows == scores_long (3 x 335; default answers zeroed)",
       len(r) == len(x) == 1005 and ((x.valid_output == 1) == (x.status == "method")).all()
       and (x.score_x == x.score_fb0).all(), f"rows {len(r)}; invalid {int((r.valid_output == 0).sum())}")

# 7 RE1-OB duplicate-time rerun (decision 2026-09-30): timefix rows == per-case acc1 of the rerun JSON; every
# contrast_release_defect row has exactly one main-table row for the same (case, method, seed)
t = d[d.source_file.str.startswith("runs/re1ob_timefix/")]
bad = 0
for src, g in t.groupby("source_file"):
    j = {c["case"]: c for c in json.load(open(f"{M}/{src}"))["cases"]}
    bad += int(sum(abs(float(j[c]["acc1"]) - s_) > 1e-9 for c, s_ in zip(g.case, g.score)))
c = d[d.role == "contrast_release_defect"]
k = ["case_id", "method", "seed"]
mm = main.merge(c[k], on=k)
report("7 RE1-OB timefix rows == rerun JSON; one main row per contrast row",
       bad == 0 and len(mm) == len(c) and len(t) == len(c),
       f"timefix rows {len(t)}, contrast rows {len(c)}, mismatches {bad}, by method {t.method.value_counts().to_dict()}")
print("ALL PASS" if ok else "SOME FAIL")
