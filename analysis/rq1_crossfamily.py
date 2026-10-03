"""RQ1 additions restored from the earlier version of the audit (user 2026-09-30: restore held-out and cross-family analyses).

Reads only merged_long.csv (sha256 checked via rq1_main) and reuses rq1_main's pair analysis. Output: analysis/main/rq1_cross/.

(1) Cross-family comparison set: every method scored on all 11 main subsystems (see SET below). For each pair over k = 11: subsystem deltas, system-equal and case-weighted pooled delta, reversals,
    random-effects PI and I^2, case-level interaction LRT (Holm over the six pairs); a joint four-method LRT for
    method x subsystem and method x family.
(2) Leave-one-subsystem-out (LOSO) selection over the 11 subsystems for each pair: pool the other 10 (system-equal; case-
    weighted as a variant), pick the winner, record whether the held-out delta has the opposite sign and the regret.
(3) Robustness for the cross-family set: leave-one-family-out, leave-two-subsystems-out, alternative pooling rules
    (mean, median, case-weighted, trimmed mean), and a case-level bootstrap of whether both signs persist.
(4) Within-family LOSO regret of the pooled winner (published layer and all-pairs layer), for the regret table.
Usage (3090, auditstack env): python rq1_crossfamily.py   (after rq1_main.py)
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_main as R  # noqa: E402

# 2026-10-02: the cross-family set is every method with outputs on all 11 main subsystems. RCD and epsilon-Diagnosis
# qualify (on PetShop through the implementations PetShop ships; PetShop's epsilon-Diagnosis is stored under the
# method name "epsilon_diagnosis" and is merged here). The earlier four-method set (outputs in rq1_cross/) predates
# the OpenRCA RCD runs. Set CROSS_SET=4 to reproduce it.
SET = (["alertcount", "baro", "cd1min", "max_z"] if os.environ.get("CROSS_SET") == "4"
       else ["alertcount", "baro", "cd1min", "e_diagnosis", "max_z", "rcd"])
SET4 = SET  # name kept for the code below
OUT = f"{R.M}/analysis/main/" + ("rq1_cross" if os.environ.get("CROSS_SET") == "4" else "rq1_cross6")
MERGE = {"epsilon_diagnosis": "e_diagnosis"}
MAIN_PREFIX = ("RE1::", "openrca::", "petshop::")
B_REV = 2000


def family_of(sub):
    return sub.split("::")[0]


def pair_deltas(cs, subs, a, b):
    out = []
    for s in subs:
        sa, sb = cs.loc[s].xs(a, level="method"), cs.loc[s].xs(b, level="method")
        w = pd.concat([sa.rename("a"), sb.rename("b")], axis=1, join="inner")
        out.append(dict(sub=s, n=len(w), diff=(w.a - w.b).to_numpy(), acc_a=w.a.mean(), acc_b=w.b.mean()))
    return out


def pooled(dl, n, rule):
    if rule == "mean":
        return float(dl.mean())
    if rule == "median":
        return float(np.median(dl))
    if rule == "case_weighted":
        return float((n * dl).sum() / n.sum())
    if rule == "trimmed":
        s = np.sort(dl)
        return float(s[1:-1].mean())
    raise ValueError(rule)


def n_rev(dl, p):
    return int(((np.sign(dl) != np.sign(p)) & (dl != 0)).sum()) if abs(p) > R.TIE else 0


def meta_or_nan(dl, var):
    if len(dl) >= 4 and (var > 0).all():
        return R.meta_re(dl, var)
    return dict(re_delta=np.nan, re_pi_lo=np.nan, re_pi_hi=np.nan, tau2=np.nan, I2=np.nan, pi_crosses_0=np.nan)


def main():
    t0 = time.time()
    d = R.load()
    d = d[d.subsystem_id.str.startswith(MAIN_PREFIX)]
    os.makedirs(OUT, exist_ok=True)

    # ---- (1) cross-family set, reuse rq1_main.analyse with an empty prefix on the restricted table
    d4 = d.assign(method=d.method.replace(MERGE))
    d4 = d4[d4.method.isin(SET4)]
    subs = sorted(d4.subsystem_id.unique())
    assert len(subs) == 11, subs
    pairs, subrows = R.analyse(d4, "ALL11", "", "X4", ("published", "probe"), "mean", R.SEED)
    P = pd.DataFrame(pairs)
    P["lrt_p_holm"] = R.holm(P.lrt_p.to_numpy())
    P["wald_p_holm"] = R.holm(P.wald_p.to_numpy())
    P.to_csv(f"{OUT}/cross_pairs.csv", index=False)
    pd.DataFrame(subrows).to_csv(f"{OUT}/cross_pair_subsystem_deltas.csv", index=False)

    # joint four-method interaction tests
    cs = R.case_scores(d4, "mean")
    rows = cs.rename("y").reset_index().rename(columns={"subsystem_id": "sub"})
    rows["fam"] = rows["sub"].map(family_of)
    joint = {}
    for name, full_f, red_f in (("method_x_subsystem", "y ~ C(method)*C(sub)", "y ~ C(method)+C(sub)"),
                                ("method_x_family", "y ~ C(method)*C(fam) + C(sub)", "y ~ C(method)+C(sub)")):
        full = smf.glm(full_f, data=rows, family=sm.families.Binomial()).fit()
        red = smf.glm(red_f, data=rows, family=sm.families.Binomial()).fit()
        stat = 2 * (full.llf - red.llf)
        dof = int(round(full.df_model - red.df_model))
        joint[name] = dict(lrt=float(stat), df=dof, p=float(chi2.sf(stat, dof)))

    # ---- (2) LOSO selection per pair over 11, (3) robustness
    loso_rows, lofo_rows, lkso_rows, pool_rows, boot_rows = [], [], [], [], []
    rng = np.random.default_rng(R.SEED)
    for a, b in itertools.combinations(SET4, 2):
        per = pair_deltas(cs, subs, a, b)
        dl = np.array([p["diff"].mean() for p in per])
        n = np.array([p["n"] for p in per])
        var = np.array([p["diff"].var(ddof=1) / p["n"] for p in per])
        for i, p in enumerate(per):
            keep = np.arange(len(per)) != i
            for rule in ("mean", "case_weighted"):
                pool = pooled(dl[keep], n[keep], rule)
                win = a if pool > 0 else b
                held = dl[i]
                wrong = (pool > 0 and held < 0) or (pool < 0 and held > 0)
                loso_rows.append(dict(a=a, b=b, held_out=p["sub"], rule=rule, pooled_train=pool, winner=win,
                                      held_delta=held, reversal=bool(wrong), regret=abs(held) if wrong else 0.0))
        for fam in ("RE1", "openrca", "petshop"):
            keep = np.array([family_of(p["sub"]) != fam for p in per])
            pool = float(dl[keep].mean())
            res = dict(a=a, b=b, dropped_family=fam, k=int(keep.sum()), pooled=pool, n_pos=int((dl[keep] > 0).sum()),
                       n_neg=int((dl[keep] < 0).sum()), n_rev=n_rev(dl[keep], pool))
            res.update(meta_or_nan(dl[keep], var[keep]))
            lofo_rows.append(res)
        both, pic, tot = 0, 0, 0
        for i, j in itertools.combinations(range(len(per)), 2):
            keep = np.ones(len(per), bool)
            keep[[i, j]] = False
            tot += 1
            both += int((dl[keep] > 0).any() and (dl[keep] < 0).any())
            m = meta_or_nan(dl[keep], var[keep])
            pic += int(m["pi_crosses_0"] is True or m["pi_crosses_0"] == 1)
        lkso_rows.append(dict(a=a, b=b, n_drops=tot, both_signs=both, pi_crosses_0=pic))
        for rule in ("mean", "median", "case_weighted", "trimmed"):
            p0 = pooled(dl, n, rule)
            pool_rows.append(dict(a=a, b=b, rule=rule, pooled=p0, n_rev=n_rev(dl, p0)))
        # case-level bootstrap: resample cases within each subsystem, check whether both signs remain
        bs = np.stack([p["diff"][rng.integers(0, p["n"], size=(B_REV, p["n"]))].mean(axis=1) for p in per])  # k x B
        both_b = ((bs > 0).any(axis=0) & (bs < 0).any(axis=0)).mean()
        boot_rows.append(dict(a=a, b=b, B=B_REV, share_both_signs=float(both_b)))
    pd.DataFrame(loso_rows).to_csv(f"{OUT}/cross_loso.csv", index=False)
    pd.DataFrame(lofo_rows).to_csv(f"{OUT}/cross_lofo.csv", index=False)
    pd.DataFrame(lkso_rows).to_csv(f"{OUT}/cross_lkso2.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(f"{OUT}/cross_pooling_rules.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(f"{OUT}/cross_boot_reversal.csv", index=False)

    # ---- (4) within-family LOSO regret of the pooled winner
    acc_all = R.case_scores(d, "mean").groupby(["subsystem_id", "method"]).mean()
    reg_rows = []
    for fam, prefix in (("RE1", "RE1::"), ("openrca", "openrca::"), ("petshop", "petshop::")):
        for layer, roles in R.LAYERS.items():
            x = d[d.subsystem_id.str.startswith(prefix) & d.role.isin(roles)]
            fsubs = sorted(x.subsystem_id.unique())
            meths = [m for m in sorted(x.method.unique()) if x[x.method == m].subsystem_id.nunique() == len(fsubs)]
            A = pd.DataFrame({s: acc_all.loc[s].reindex(meths) for s in fsubs})  # methods x subsystems
            for s in fsubs:
                train = [t for t in fsubs if t != s]
                score = A[train].mean(axis=1)
                winner = score.sort_index().idxmax()  # ties -> alphabetical first
                best = A[s].max()
                reg_rows.append(dict(family=fam, layer=layer, held_out=s, n_methods=len(meths), winner=winner,
                                     winner_acc=float(A.loc[winner, s]), best_acc=float(best),
                                     best_methods=";".join(A.index[A[s] == best]), regret=float(best - A.loc[winner, s])))
    pd.DataFrame(reg_rows).to_csv(f"{OUT}/family_loso_regret.csv", index=False)

    cfg = dict(script=os.path.abspath(__file__), script_sha256=R.sha(os.path.abspath(__file__)), long_sha256=R.SHA,
               set4=SET4, subsystems=subs, B=R.B, B_rev=B_REV, seed=R.SEED, joint_interaction=joint,
               run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_s=round(time.time() - t0, 1))
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
