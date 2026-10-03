"""RQ2(c) fault composition (main-thread implementation).

Fault labels
  RCAEval: long-table `fault` column (RE1: cpu/mem/disk/delay/loss, 25 each per system; RE2: + socket, 15 each).
  OpenRCA: record.csv row i <-> query.csv row i (checked: for every query whose scoring points carry a datetime, record
    row i's datetime is among them; 185/185 rows); multi-failure queries take record row i's reason and are flagged.
    Reasons are system-specific phrases; they are mapped to coarse classes (REASON_CLASS below, Claude's coding) so that
    subsystems can share fault types.
  PetShop: target.json "target.metric" (latency / availability) of each issue, the stratification of PetShop Table 8.
Analyses (mean-over-seeds scores, pair machinery as in rq1_main)
  1. stratified: per pair x subsystem x fault class, delta; per pair x subsystem, whether fault-level deltas disagree in sign.
  2. reweighting: per subsystem, accuracy = equal-weight mean over the fault classes common to all subsystems of the family
     (cases of other classes dropped); pair deltas, pooled (system-equal), reversals, bootstrap CI (resampling cases within
     subsystem x class cells, B and seed as rq1_main); compared with the reference (main table, all cases).
  3. cross-release with RE2 restricted to RE1's fault types (socket dropped).
Usage (3090, auditstack env): python rq2_fault.py
"""
import csv
import itertools
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_main as R  # noqa: E402

OUT = f"{R.M}/analysis/main/rq2"
OPEN_RCA = os.environ.get("OPENRCA_DATASET", "OpenRCA/dataset")
PETSHOP = os.environ.get("PETSHOP_DATASET", "petshop/dataset")
PREFIX = {"RE1": "RE1::", "RE2": "RE2::", "openrca": "openrca::", "petshop": "petshop::"}
REASON_CLASS = {
    "cpu": ["high CPU usage", "high JVM CPU load", "CPU fault", "container CPU load", "node CPU load", "node CPU spike"],
    "memory": ["high memory usage", "JVM Out of Memory (OOM) Heap", "container memory load", "node memory consumption"],
    "disk": ["high disk I/O read usage", "high disk space usage", "container read I/O load", "container write I/O load",
             "node disk read I/O consumption", "node disk write I/O consumption", "node disk space consumption"],
    "network": ["network packet loss", "network latency", "network delay", "network loss",
                "container network packet corruption", "container network packet retransmission",
                "container network latency", "container packet loss"],
    "other": ["db connection limit", "db close", "container process termination"],
}
CLASS_OF = {r: c for c, rs in REASON_CLASS.items() for r in rs}


def openrca_labels():
    rows = []
    for tag, rel in (("bank", "Bank"), ("mkt1", "Market/cloudbed-1"), ("mkt2", "Market/cloudbed-2"), ("tel", "Telecom")):
        q = list(csv.DictReader(open(f"{OPEN_RCA}/{rel}/query.csv")))
        r = list(csv.DictReader(open(f"{OPEN_RCA}/{rel}/record.csv")))
        assert len(q) == len(r)
        seen = {}
        for i, (x, y) in enumerate(zip(q, r)):
            dts = re.findall(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", x["scoring_points"])
            assert not dts or y["datetime"] in dts, (tag, i)
            k = seen.get(x["task_index"], 0)
            seen[x["task_index"]] = k + 1
            rows.append(dict(case_id=f"openrca::{tag}|{tag}/{x['task_index']}#{k}", reason=y["reason"],
                             fault_class=CLASS_OF[y["reason"]], multi_failure=len(dts) > 1))
    return pd.DataFrame(rows)


def petshop_labels(d):
    rows = []
    for cid, sub, case in d[d.family == "petshop"][["case_id", "subsystem", "case"]].drop_duplicates().itertuples(index=False):
        t = json.load(open(f"{PETSHOP}/{sub}/{case}/target.json"))
        rows.append(dict(case_id=cid, fault_class=t["target"]["metric"]))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    full = pd.read_csv(R.LONG, low_memory=False)
    assert R.sha(R.LONG) == R.SHA
    d = full[full.role.isin(["published", "probe", "shipped_baseline"])].copy()
    if os.path.isdir(OPEN_RCA) and os.path.isdir(PETSHOP):
        lab = pd.concat([openrca_labels(), petshop_labels(d)])
    else:  # replication package: labels that the two functions above extract from the benchmarks' ground truth
        lab = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                                       "fault_labels_openrca_petshop.csv"))
    lab.to_csv(f"{OUT}/fault_labels_openrca_petshop.csv", index=False)
    d = d.merge(lab[["case_id", "fault_class"]], on="case_id", how="left")
    rc = d.family == "rcaeval"
    d.loc[rc, "fault_class"] = d.loc[rc, "fault"]
    assert d.fault_class.notna().all()
    cs = d.groupby(["subsystem_id", "case_id", "method", "role", "fault_class"]).score.mean().reset_index()
    comp = cs.drop_duplicates("case_id").groupby(["subsystem_id", "fault_class"]).size().unstack(fill_value=0)
    comp.to_csv(f"{OUT}/fault_composition.csv")

    strat, rew, summ = [], [], {}
    for fam, pre in PREFIX.items():
        x = cs[cs.subsystem_id.str.startswith(pre)]
        subs = sorted(x.subsystem_id.unique())
        common = sorted(set.intersection(*[set(x[x.subsystem_id == s].fault_class) for s in subs]))
        for layer, roles in R.LAYERS.items():
            y = x[x.role.isin(roles)]
            methods = sorted(y.method.unique())
            for a, b in itertools.combinations(methods, 2):
                rng = np.random.default_rng(R.SEED)
                per = []
                for s in subs:
                    ya = y[(y.subsystem_id == s) & (y.method == a)].set_index("case_id")
                    yb = y[(y.subsystem_id == s) & (y.method == b)].set_index("case_id")
                    w = ya[["score", "fault_class"]].join(yb[["score"]], rsuffix="_b", how="inner")
                    if not len(w):
                        continue
                    w["diff"] = w.score - w.score_b
                    fl = w.groupby("fault_class")["diff"].mean()
                    for f, v in fl.items():
                        strat.append(dict(family=fam, layer=layer, a=a, b=b, sub=s, fault_class=f,
                                          n=int((w.fault_class == f).sum()), delta=v))
                    ref = w["diff"].mean()
                    cells = [w.loc[w.fault_class == f, "diff"].to_numpy() for f in common]
                    rw = float(np.mean([c.mean() for c in cells]))
                    bs = np.mean([c[rng.integers(0, len(c), size=(R.B, len(c)))].mean(axis=1) for c in cells], axis=0)
                    per.append(dict(sub=s, ref=ref, rw=rw, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                                    fault_sign_disagree=bool((fl > 0).any() and (fl < 0).any())))
                if len(per) < 2:
                    continue
                pr, pw = np.mean([p["ref"] for p in per]), np.mean([p["rw"] for p in per])
                for p in per:
                    rew.append(dict(family=fam, layer=layer, a=a, b=b, sub=p["sub"], common_classes=";".join(common),
                                    delta_ref=p["ref"], pooled_ref=pr, delta_rw=p["rw"], lo_rw=p["lo"], hi_rw=p["hi"],
                                    pooled_rw=pw, fault_sign_disagree=p["fault_sign_disagree"],
                                    rev_ref=bool(abs(pr) > R.TIE and p["ref"] != 0 and np.sign(p["ref"]) != np.sign(pr)),
                                    rev_rw=bool(abs(pw) > R.TIE and p["rw"] != 0 and np.sign(p["rw"]) != np.sign(pw)),
                                    rev_ci_rw=bool((pw > R.TIE and p["hi"] < 0) or (pw < -R.TIE and p["lo"] > 0)),
                                    sign_change=bool(np.sign(p["ref"]) != np.sign(p["rw"]))))
    S = pd.DataFrame(strat)
    W = pd.DataFrame(rew)
    S.to_csv(f"{OUT}/fault_stratified_deltas.csv", index=False)
    W.to_csv(f"{OUT}/fault_reweighted.csv", index=False)
    for (fam, layer), g in W.groupby(["family", "layer"]):
        summ[f"fault_{fam}_{layer}"] = dict(common_classes=g.common_classes.iloc[0], comparisons=len(g),
                                            fault_sign_disagree=int(g.fault_sign_disagree.sum()),
                                            rev_ref=int(g.rev_ref.sum()), rev_rw=int(g.rev_rw.sum()),
                                            rev_ci_rw=int(g.rev_ci_rw.sum()), rev_ref_gone=int((g.rev_ref & ~g.rev_rw).sum()),
                                            rev_new=int((g.rev_rw & ~g.rev_ref).sum()), sign_changes=int(g.sign_change.sum()))

    # cross-release with RE2 restricted to RE1 fault types
    c1 = cs[cs.subsystem_id.str.startswith("RE1::")]
    c2 = cs[cs.subsystem_id.str.startswith("RE2::") & cs.fault_class.isin(set(c1.fault_class))]
    xr = []
    for layer, roles in R.LAYERS.items():
        a1 = c1[c1.role.isin(roles)].groupby(["subsystem_id", "method"]).score.mean().unstack(0)
        a2 = c2[c2.role.isin(roles)].groupby(["subsystem_id", "method"]).score.mean().unstack(0)
        a2all = cs[cs.subsystem_id.str.startswith("RE2::") & cs.role.isin(roles)].groupby(
            ["subsystem_id", "method"]).score.mean().unstack(0)
        ms = sorted(set(a1.index) & set(a2.index))
        for a, b in itertools.combinations(ms, 2):
            for app in ("OB", "SS", "TT"):
                d1 = a1.loc[a, f"RE1::{app}"] - a1.loc[b, f"RE1::{app}"]
                d2 = a2.loc[a, f"RE2::{app}"] - a2.loc[b, f"RE2::{app}"]
                d2all = a2all.loc[a, f"RE2::{app}"] - a2all.loc[b, f"RE2::{app}"]
                if np.isnan(d1) or np.isnan(d2):
                    continue
                xr.append(dict(layer=layer, a=a, b=b, app=app, delta_re1=d1, delta_re2_all=d2all, delta_re2_re1faults=d2,
                               agree_all=bool(np.sign(d1) == np.sign(d2all)), agree_re1faults=bool(np.sign(d1) == np.sign(d2))))
    X = pd.DataFrame(xr)
    X.to_csv(f"{OUT}/cross_release_re1_faults.csv", index=False)
    for layer, g in X.groupby("layer"):
        summ[f"cross_release_re1faults_{layer}"] = dict(comparisons=len(g), agree_all_faults=int(g.agree_all.sum()),
                                                        agree_re1_faults=int(g.agree_re1faults.sum()))
    cfg = dict(script_sha256=R.sha(os.path.abspath(__file__)), rq1_main_sha256=R.sha(R.__file__), long_sha256=R.SHA,
               B=R.B, seed=R.SEED, reason_class=REASON_CLASS, run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               wall_s=round(time.time() - t0, 1))
    json.dump(dict(config=cfg, summary=summ), open(f"{OUT}/fault_summary.json", "w"), indent=1, default=float)
    print(comp.to_string())
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
