"""RQ1 follow-ups (main-thread implementation), reading rq1_main.py outputs and merged_long.csv.

B  margin vs dispersion: |pooled margin| (system-equal) vs SD of subsystem deltas, per pair; scatter; the same
   computation on two published tables (RCAEval Table 6 AC@1 over 6 fault types, RE2 Train Ticket; PetShop Table 8
   top-1 recall over 6 strata), parsed from the paper texts in analysis/main/inputs/.
C  cross-release: for method pairs present in RE1 and RE2 (L1 and L2), per application compare sign(delta_RE1) and
   sign(delta_RE2).
D  regret of the pooled winner (case-weighted and system-equal) per subsystem, per family and layer.
E  small-sample sensitivity: keep subsystems with n >= 25 / n >= 50 and recount reversals (system-equal pooling).
Usage (3090, auditstack env): python rq1_extra.py   (after rq1_main.py)
"""
import hashlib
import itertools
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

M = os.environ.get("AUDIT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work"))
LONG = f"{M}/merged_long.csv"
SHA = "78b24273444465dadce151d6452519f251fa482b7afa878d8fae5931dd180295"  # released (anonymized) table; frozen original bface2ee64a01c81a7ddd521c62a47ee4e2a127169366a79f24ad0ec6e202bd4
IN = f"{M}/analysis/main/rq1"
INP = f"{M}/analysis/main/inputs"
OUT = f"{M}/analysis/main/rq1"
FAM = {"RE1": "RE1::", "openrca": "openrca::", "petshop": "petshop::", "RE2": "RE2::"}
LAYERS = {"L1": ("published",), "L2": ("published", "probe", "shipped_baseline")}  # shipped_baseline: PetShop only (user 2026-09-29)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def pairs_on_table(tab, label):
    """tab: DataFrame rows = methods, columns = units (strata). Margin = mean over units."""
    rows = []
    for a, b in itertools.combinations(tab.index, 2):
        dl = (tab.loc[a] - tab.loc[b]).to_numpy(float)
        pooled = dl.mean()
        rows.append(dict(table=label, a=a, b=b, k=len(dl), pooled=pooled, abs_margin=abs(pooled),
                         sd_units=dl.std(ddof=1), n_rev=int(((np.sign(dl) != np.sign(pooled)) & (dl != 0)).sum()) if abs(pooled) > 1e-9 else 0,
                         units_a_wins=int((dl > 0).sum()), units_b_wins=int((dl < 0).sum())))
    r = pd.DataFrame(rows)
    means = tab.mean(axis=1).sort_values(ascending=False)
    gaps = pd.DataFrame(dict(method=means.index, mean=means.values, gap_to_next=-np.diff(np.r_[means.values, np.nan])))
    gaps["table"] = label
    return r, gaps


def rcaeval_table6():
    txt = open(f"{INP}/rcaeval_2412.17015.txt").read().splitlines()
    start = next(i for i, l in enumerate(txt) if "Table 6: RCA performance" in l)
    names = ["BARO", "CausalRCA", "CIRCA", "MicroCause", "RCD", "MicroRank", "TraceRCA", "BARO", "CIRCA",
             "PDiagnose", "RCD"]
    src = ["metric"] * 5 + ["trace"] * 2 + ["multi"] * 4
    rows, j = [], 0
    for l in txt[start:start + 30]:
        m = re.match(r"^\s*(?:(?:Metric|Multi-|Source|Trace)\s+)?([A-Za-z]+)\s+((?:[0-9.]+\s+){20}[0-9.]+)\s*$", l)
        if m and j < len(names) and m.group(1) == names[j]:
            v = [float(x) for x in m.group(2).split()]
            rows.append(dict(method=f"{names[j]} ({src[j]})", **{f: v[3 * i] for i, f in
                                                                 enumerate(["cpu", "mem", "disk", "socket", "delay", "loss"])},
                             printed_avg=v[18]))
            j += 1
    assert j == len(names), f"parsed {j} Table 6 rows"
    t = pd.DataFrame(rows).set_index("method")
    return t


def petshop_table8():
    txt = open(f"{INP}/petshop_2311.04806.txt").read().splitlines()
    end = next(i for i, l in enumerate(txt) if l.strip().startswith("Table 8: Top-1 recall"))
    rows = []
    for l in txt[end - 8:end]:
        m = re.match(r"^\s*(low|high|temporal)\s+(latency|availability)\s+((?:[0-9.]+\s+){5}[0-9.]+)\s*$", l)
        if m:
            rows.append([f"{m.group(1)}-{m.group(2)}"] + [float(x) for x in m.group(3).split()])
    assert len(rows) == 6, rows
    t = pd.DataFrame(rows, columns=["stratum", "traversal", "circa", "counterfactual", "epsilon_diagnosis", "rcd",
                                    "correlation"]).set_index("stratum").T
    return t


def main():
    t0 = time.time()
    assert sha(LONG) == SHA
    P = pd.read_csv(f"{IN}/pairs.csv")
    S = pd.read_csv(f"{IN}/pair_subsystem_deltas.csv")
    d = pd.read_csv(LONG, low_memory=False)
    d = d[d.role.isin(["published", "probe", "shipped_baseline"])]
    out = {}

    # ---- B: margin vs dispersion (11-subsystem families)
    pm = P[(P.variant == "mean") & P.family.isin(["RE1", "openrca", "petshop"])].copy()
    pm["abs_margin"] = pm.pooled_se.abs()
    pm["margin_lt_sd"] = pm.abs_margin < pm.sd_sub_delta
    pm["reverses"] = pm.n_rev_se > 0
    pm["reverses_ci"] = pm.n_rev_ci_se > 0
    pm[["family", "layer", "a", "b", "k", "pooled_se", "abs_margin", "sd_sub_delta", "margin_lt_sd", "n_rev_se",
        "reverses", "n_rev_ci_se", "reverses_ci"]].to_csv(f"{OUT}/margin_dispersion.csv", index=False)
    summ = pm.groupby(["layer", "margin_lt_sd"]).agg(pairs=("a", "size"), reverse=("reverses", "sum"),
                                                     reverse_ci=("reverses_ci", "sum")).reset_index()
    summ.to_csv(f"{OUT}/margin_dispersion_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, layer in zip(axes, ("L1", "L2")):
        q = pm[pm.layer == layer]
        for fam, mk in (("RE1", "o"), ("openrca", "s"), ("petshop", "^")):
            z = q[q.family == fam]
            ax.scatter(z.abs_margin[~z.reverses], z.sd_sub_delta[~z.reverses], marker=mk, facecolors="none",
                       edgecolors="gray", label=f"{fam} no reversal")
            ax.scatter(z.abs_margin[z.reverses], z.sd_sub_delta[z.reverses], marker=mk, color="C3",
                       label=f"{fam} reversal")
        lim = max(pm.abs_margin.max(), pm.sd_sub_delta.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlabel("|pooled margin| (system-equal)")
        ax.set_title({"L1": "published vs published", "L2": "incl. probes"}[layer])
    axes[0].set_ylabel("SD of subsystem deltas")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/margin_dispersion.png", dpi=150)

    t6 = rcaeval_table6()
    t8 = petshop_table8()
    t6.to_csv(f"{OUT}/published_rcaeval_table6_ac1.csv")
    t8.to_csv(f"{OUT}/published_petshop_table8_top1.csv")
    r6, g6 = pairs_on_table(t6.drop(columns="printed_avg"), "RCAEval Table 6 (RE2-TT, AC@1, 6 fault types)")
    r8, g8 = pairs_on_table(t8, "PetShop Table 8 (top-1, 6 strata)")
    pub = pd.concat([r6, r8])
    pub["margin_lt_sd"] = pub.abs_margin < pub.sd_units
    pub.to_csv(f"{OUT}/published_tables_margin_dispersion.csv", index=False)
    pd.concat([g6, g8]).to_csv(f"{OUT}/published_tables_adjacent_gaps.csv", index=False)
    out["published_tables"] = pub.groupby(["table", "margin_lt_sd"]).agg(
        pairs=("a", "size"), reverse=("n_rev", lambda x: int((x > 0).sum()))).reset_index().to_dict("records")

    # ---- C: cross-release
    cr = []
    for layer in ("L1", "L2"):
        s1 = S[(S.family == "RE1") & (S.layer == layer) & (S.variant == "mean")].copy()
        s2 = S[(S.family == "RE2") & (S.layer == layer) & (S.variant == "mean")].copy()
        s1["app"], s2["app"] = s1["sub"].str[5:], s2["sub"].str[5:]
        m = s1.merge(s2, on=["a", "b", "app"], suffixes=("_re1", "_re2"))
        m["layer"] = layer
        m["sign_agree"] = np.sign(m.delta_re1) == np.sign(m.delta_re2)
        m["abs_delta_diff"] = (m.delta_re1.abs() - m.delta_re2.abs())
        m["ci_opposite"] = ((m.lo_re1 > 0) & (m.hi_re2 < 0)) | ((m.hi_re1 < 0) & (m.lo_re2 > 0))
        cr.append(m[["layer", "a", "b", "app", "n_re1", "delta_re1", "lo_re1", "hi_re1", "n_re2", "delta_re2", "lo_re2",
                     "hi_re2", "sign_agree", "ci_opposite", "abs_delta_diff"]])
    cr = pd.concat(cr)
    cr.to_csv(f"{OUT}/cross_release.csv", index=False)
    out["cross_release"] = cr.groupby("layer").agg(comparisons=("a", "size"), sign_agree=("sign_agree", "sum"),
                                                   ci_opposite=("ci_opposite", "sum")).reset_index().to_dict("records")

    # ---- D: regret of pooled winner
    cs = d.groupby(["subsystem_id", "case_id", "method", "role"]).score.mean().reset_index()
    rg = []
    for fam, pre in FAM.items():
        for layer, roles in LAYERS.items():
            x = cs[cs.subsystem_id.str.startswith(pre) & cs.role.isin(roles)]
            acc = x.groupby(["subsystem_id", "method"]).score.mean().unstack(0)
            n = x.groupby("subsystem_id").case_id.nunique()
            complete = acc.dropna()  # methods present on every subsystem of the family
            cw = (complete * n).sum(axis=1) / n.sum()
            se = complete.mean(axis=1)
            for wname, pooled in (("case_weighted", cw), ("system_equal", se)):
                win = pooled.idxmax()
                for s in acc.columns:
                    best = acc[s].max()
                    rg.append(dict(family=fam, layer=layer, weighting=wname, winner=win, winner_pooled=pooled[win],
                                   runner_up=pooled.drop(win).idxmax(), runner_up_pooled=pooled.drop(win).max(),
                                   subsystem=s, n=int(n[s]), winner_acc=acc.loc[win, s], best_method=acc[s].idxmax(),
                                   best_acc=best, regret=best - acc.loc[win, s],
                                   methods_ranked=len(complete), methods_excluded_incomplete=";".join(
                                       sorted(set(acc.index) - set(complete.index)))))
    rg = pd.DataFrame(rg)
    rg.to_csv(f"{OUT}/regret.csv", index=False)

    # ---- E: small-sample sensitivity
    ss = []
    for nmin in (0, 25, 50):
        z = S[(S.variant == "mean") & (S.n >= nmin) & S.family.isin(["RE1", "openrca", "petshop"])]
        for (fam, layer, a, b), g in z.groupby(["family", "layer", "a", "b"]):
            if len(g) < 2:
                continue
            pooled = g.delta.mean()
            ss.append(dict(n_min=nmin, family=fam, layer=layer, a=a, b=b, k=len(g), pooled_se=pooled,
                           n_rev=int(((np.sign(g.delta) != np.sign(pooled)) & (g.delta != 0)).sum()) if abs(pooled) > 1e-9 else 0,
                           n_rev_ci=int((((pooled > 1e-9) & (g.hi < 0)) | ((pooled < -1e-9) & (g.lo > 0))).sum())))
    ss = pd.DataFrame(ss)
    ss.to_csv(f"{OUT}/small_sample.csv", index=False)
    out["small_sample"] = ss.groupby(["n_min", "family", "layer"]).agg(
        pairs=("a", "size"), rev=("n_rev", lambda x: int((x > 0).sum())),
        rev_ci=("n_rev_ci", lambda x: int((x > 0).sum()))).reset_index().to_dict("records")

    cfg = dict(script_sha256=sha(os.path.abspath(__file__)), long_sha256=SHA,
               inputs={f: sha(f"{INP}/{f}") for f in os.listdir(INP)},
               rq1_main_config=json.load(open(f"{IN}/config.json")),
               run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_s=round(time.time() - t0, 1))
    json.dump(dict(config=cfg, summary=out), open(f"{OUT}/extra_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
