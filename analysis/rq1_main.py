"""RQ1 (main-thread implementation): stability of pairwise method comparisons across subsystems.

Follows .research/plans/2026-09-29-wp4-analysis-spec.md. Independent of the subagent scripts in scripts/analysis/.
Input: merged_long.csv (G3 frozen; sha256 checked). Output: analysis/main/rq1/*.csv + config.json

Per family (RE1 k=3, openrca k=4, petshop k=4; RE2 descriptive / cross-release only) and layer
(L1 = published, L2 = published + probe + PetShop shipped baselines), for every method pair (a, b):
  subsystem delta = mean over matched cases of (score_a - score_b); seeded methods (rcd, causalrca) averaged over
  seeds per case ("mean" variant) or taken seed by seed ("seed0/1/2"; two seeded methods paired by seed index).
  pooled delta: case-weighted and system-equal; 95% percentile bootstrap CIs (B = 10000, seed 20260929; subsystem CIs
  resample cases within the subsystem, pooled CIs resample cases within every subsystem).
  reversal: sign(subsystem delta) opposite to sign(pooled delta), subsystem delta != 0;
  CI-supported reversal: the subsystem CI lies entirely on the opposite side of 0.
  k >= 4: random-effects (statsmodels combine_effects, iterated, t), PI with t_{k-2} (audit meta.py).
  interaction: case-level GLM Binomial y ~ C(method)*C(sub) vs y ~ C(method)+C(sub), LRT; cluster-robust Wald
  (clusters = case_id) on the interaction terms. Holm within (family, layer, variant).
Usage (3090, auditstack env): python rq1_main.py
"""
import hashlib
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from scipy.stats import t as student_t
from statsmodels.stats.meta_analysis import combine_effects

warnings.filterwarnings("ignore")
M = os.environ.get("AUDIT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work"))
LONG = f"{M}/merged_long.csv"
SHA = "78b24273444465dadce151d6452519f251fa482b7afa878d8fae5931dd180295"  # released (anonymized) table; frozen original bface2ee64a01c81a7ddd521c62a47ee4e2a127169366a79f24ad0ec6e202bd4
OUT = f"{M}/analysis/main/rq1"
B = 10000
SEED = 20260929
TIE = 1e-9  # |pooled| below this = tied pair: no winner, so no reversal is defined
FAMILIES = {"RE1": "RE1::", "openrca": "openrca::", "petshop": "petshop::", "RE2": "RE2::"}
LAYERS = {"L1": ("published",), "L2": ("published", "probe", "shipped_baseline")}  # shipped_baseline: PetShop only (user 2026-09-29)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    assert sha(LONG) == SHA, "merged_long.csv sha256 mismatch"
    d = pd.read_csv(LONG, low_memory=False)
    return d[d.role.isin(["published", "probe", "shipped_baseline"])].copy()


def case_scores(d, variant):
    """(subsystem_id, case_id, method) -> score for the variant ('mean' or 'seed<k>')."""
    if variant == "mean":
        return d.groupby(["subsystem_id", "case_id", "method"]).score.mean()
    k = float(variant[4:])
    keep = d.seed.isna() | (d.seed == k)
    x = d[keep]
    assert not x.duplicated(["subsystem_id", "case_id", "method"]).any()
    return x.set_index(["subsystem_id", "case_id", "method"]).score


def boot_idx(n, rng):
    return rng.integers(0, n, size=(B, n))


def meta_re(eff, var):
    k = len(eff)
    res = combine_effects(eff, var, method_re="iterated", use_t=True)
    tau2 = float(res.tau2)
    wre = 1 / (var + tau2)
    dre = float((wre * eff).sum() / wre.sum())
    lo, hi = res.conf_int(use_t=True, alpha=0.05)[1]
    se = ((hi - lo) / 2) / student_t.ppf(.975, df=k - 1)
    half = student_t.ppf(.975, df=k - 2) * np.sqrt(tau2 + se ** 2)
    wfe = 1 / var
    dfe = float((wfe * eff).sum() / wfe.sum())
    Q = float((wfe * (eff - dfe) ** 2).sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(re_delta=dre, re_pi_lo=dre - half, re_pi_hi=dre + half, tau2=tau2, I2=100 * I2,
                pi_crosses_0=bool(dre - half < 0 < dre + half))


def interaction(rows):
    """rows: DataFrame case_id, sub, method, y (two methods)."""
    out = {}
    try:
        full = smf.glm("y ~ C(method)*C(sub)", data=rows, family=sm.families.Binomial()).fit()
        red = smf.glm("y ~ C(method)+C(sub)", data=rows, family=sm.families.Binomial()).fit()
        stat = 2 * (full.llf - red.llf)
        dof = int(full.df_model - red.df_model)
        out.update(lrt=stat, lrt_df=dof, lrt_p=float(1 - chi2.cdf(stat, dof)))
    except Exception as e:  # perfect separation etc.
        out.update(lrt=np.nan, lrt_df=np.nan, lrt_p=np.nan, lrt_err=type(e).__name__)
    try:
        groups = pd.factorize(rows.case_id)[0]
        f = smf.glm("y ~ C(method)*C(sub)", data=rows, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        names = [n for n in f.params.index if "C(method)" in n and "C(sub)" in n]  # sub labels contain "::"
        R = np.zeros((len(names), len(f.params)))
        for i, n in enumerate(names):
            R[i, list(f.params.index).index(n)] = 1
        w = f.wald_test(R, scalar=True)
        out.update(wald=float(w.statistic), wald_df=len(names), wald_p=float(w.pvalue))
    except Exception as e:
        out.update(wald=np.nan, wald_df=np.nan, wald_p=np.nan, wald_err=type(e).__name__)
    return out


def holm(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    idx = np.argsort(np.where(ok, p, np.inf))
    m = ok.sum()
    run = 0.0
    for r, i in enumerate(idx[:m]):
        run = max(run, min(1.0, p[i] * (m - r)))
        adj[i] = run
    return adj


def analyse(d, fam, prefix, layer, roles, variant, rng_seed):
    x = d[d.subsystem_id.str.startswith(prefix) & d.role.isin(roles)]
    methods = sorted(x.method.unique())
    seeded = set(x.loc[x.seed.notna(), "method"])  # multi-seed methods actually present in this family
    if variant != "mean" and not seeded:
        return [], []
    cs = case_scores(x, variant)
    subs = sorted(x.subsystem_id.unique())
    pair_rows, sub_rows = [], []
    for a, b in itertools.combinations(methods, 2):
        if variant != "mean" and a not in seeded and b not in seeded:
            continue
        rng = np.random.default_rng(rng_seed)
        per = []
        for s in subs:
            try:
                sa, sb = cs.loc[s].xs(a, level="method"), cs.loc[s].xs(b, level="method")
            except KeyError:
                continue
            w = pd.concat([sa.rename("a"), sb.rename("b")], axis=1, join="inner")
            if not len(w):
                continue
            diff = (w.a - w.b).to_numpy()
            bs = diff[boot_idx(len(diff), rng)].mean(axis=1)
            per.append(dict(sub=s, n=len(diff), delta=diff.mean(), var=diff.var(ddof=1) / len(diff),
                            acc_a=w.a.mean(), acc_b=w.b.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                            boot=bs, cases=w.index.tolist(), ya=w.a.to_numpy(), yb=w.b.to_numpy()))
        if len(per) < 2:
            continue
        n = np.array([p["n"] for p in per])
        dl = np.array([p["delta"] for p in per])
        bmat = np.stack([p["boot"] for p in per])
        res = dict(family=fam, layer=layer, variant=variant, a=a, b=b, k=len(per))
        for wname, wt in (("cw", n / n.sum()), ("se", np.full(len(per), 1 / len(per)))):
            pooled = float((wt * dl).sum())
            pb = (wt[:, None] * bmat).sum(axis=0)
            rev = [p["sub"] for p in per if abs(pooled) > TIE and p["delta"] != 0 and np.sign(p["delta"]) != np.sign(pooled)]
            rev_ci = [p["sub"] for p in per if (pooled > TIE and p["hi"] < 0) or (pooled < -TIE and p["lo"] > 0)]
            res.update({f"pooled_{wname}": pooled, f"pooled_{wname}_lo": np.percentile(pb, 2.5),
                        f"pooled_{wname}_hi": np.percentile(pb, 97.5), f"n_rev_{wname}": len(rev),
                        f"rev_{wname}": ";".join(rev), f"n_rev_ci_{wname}": len(rev_ci), f"rev_ci_{wname}": ";".join(rev_ci)})
        res["sd_sub_delta"] = float(np.std(dl, ddof=1))
        res["n_pos"], res["n_neg"] = int((dl > 0).sum()), int((dl < 0).sum())
        var = np.array([p["var"] for p in per])
        if len(per) >= 4 and (var > 0).all():
            res.update(meta_re(dl, var))
        rows = pd.concat([pd.DataFrame(dict(case_id=p["cases"] * 2, sub=p["sub"], method=[a] * p["n"] + [b] * p["n"],
                                            y=np.r_[p["ya"], p["yb"]])) for p in per])
        res.update(interaction(rows))
        # a method with 0 or all hits in some subsystem -> quasi-separation; the GLM Wald (and to a lesser degree the
        # LRT) is then unreliable, so it is flagged rather than dropped
        res["separation"] = any(min(p["acc_a"], p["acc_b"]) == 0 or max(p["acc_a"], p["acc_b"]) == 1 for p in per)
        pair_rows.append(res)
        for p in per:
            sub_rows.append(dict(family=fam, layer=layer, variant=variant, a=a, b=b, sub=p["sub"], n=p["n"],
                                 acc_a=p["acc_a"], acc_b=p["acc_b"], delta=p["delta"], lo=p["lo"], hi=p["hi"]))
    return pair_rows, sub_rows


def main():
    t0 = time.time()
    d = load()
    os.makedirs(OUT, exist_ok=True)
    pairs, subs = [], []
    for fam, prefix in FAMILIES.items():
        for layer, roles in LAYERS.items():
            for variant in ("mean", "seed0", "seed1", "seed2"):
                p, s = analyse(d, fam, prefix, layer, roles, variant, SEED)
                pairs += p
                subs += s
    P = pd.DataFrame(pairs)
    for key, g in P.groupby(["family", "layer", "variant"]):
        for col in ("lrt_p", "wald_p"):
            P.loc[g.index, col + "_holm"] = holm(g[col].to_numpy())
    P.to_csv(f"{OUT}/pairs.csv", index=False)
    pd.DataFrame(subs).to_csv(f"{OUT}/pair_subsystem_deltas.csv", index=False)
    # per-subsystem accuracy table (mean variant)
    acc = case_scores(d, "mean").groupby(["subsystem_id", "method"]).mean().unstack(0)
    acc.to_csv(f"{OUT}/accuracy_by_subsystem.csv")
    cfg = dict(script=os.path.abspath(__file__), script_sha256=sha(os.path.abspath(__file__)), long_sha256=SHA,
               B=B, seed=SEED, python=sys.version.split()[0], pandas=pd.__version__, run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               wall_s=round(time.time() - t0, 1))
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
