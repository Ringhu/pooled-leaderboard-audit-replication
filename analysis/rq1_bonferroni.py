"""Multiplicity check for the CI-supported reversals of RQ1 (published layer, main families).

For every published-pair comparison on one subsystem (RE1 3 subsystems x 10 pairs, OpenRCA 4 x 3, PetShop 4 x 10 = 82),
recompute the percentile bootstrap CI of the subsystem difference at the Bonferroni-adjusted level 1 - 0.05/82
(B = 400,000 resamples, seed 20260929, 'mean' variant: seeded methods averaged over seeds per case, as in rq1_main.py),
and count how many CI-supported reversals (subsystem sign opposite to the system-equal pooled sign, adjusted CI
entirely on that side) remain. Reads the frozen long table; writes paper/data/rq1/bonferroni.csv + .json.
"""
import hashlib, json, os, sys, time, itertools
import numpy as np, pandas as pd

LONG = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mnt/3090/rca_audit_merged/merged_long.csv")
PAIRS = sys.argv[2] if len(sys.argv) > 2 else "paper/data/rq1/pairs.csv"
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "paper/data/rq1"
SHA = "78b24273444465dadce151d6452519f251fa482b7afa878d8fae5931dd180295"
B, SEED, CHUNK = 400_000, 20260929, 20_000
FAMILIES = {"RE1": "RE1::", "openrca": "openrca::", "petshop": "petshop::"}
TIE = 1e-9

d = pd.read_csv(LONG, low_memory=False)
assert hashlib.sha256(open(LONG, "rb").read()).hexdigest() == SHA, "merged_long.csv sha256 mismatch"
d = d[d.role == "published"]
cs = d.groupby(["subsystem_id", "case_id", "method"]).score.mean()
pairs = pd.read_csv(PAIRS)
pairs = pairs[(pairs.layer == "L1") & (pairs.variant == "mean") & pairs.family.isin(FAMILIES)]

rows = []
for _, pr in pairs.iterrows():
    fam, a, b = pr.family, pr.a, pr.b
    pooled = float(pr.pooled_se)
    subs = sorted(s for s in d.subsystem_id.unique() if s.startswith(FAMILIES[fam]))
    for s in subs:
        try:
            sa, sb = cs.loc[s].xs(a, level="method"), cs.loc[s].xs(b, level="method")
        except KeyError:
            continue
        w = pd.concat([sa.rename("a"), sb.rename("b")], axis=1, join="inner")
        if not len(w):
            continue
        rows.append(dict(family=fam, a=a, b=b, sub=s, n=len(w), pooled_se=pooled, diff=(w.a - w.b).to_numpy()))
m = len(rows)
alpha_adj = 0.05 / m
q_lo, q_hi = 100 * alpha_adj / 2, 100 * (1 - alpha_adj / 2)
print(f"comparisons={m} alpha_adj={alpha_adj:.6f} percentiles=({q_lo:.4f}, {q_hi:.4f}) B={B}")
out = []
t0 = time.time()
for r in rows:
    rng = np.random.default_rng(SEED)
    diff = r.pop("diff"); n = len(diff)
    means = np.empty(B)
    for i in range(0, B, CHUNK):
        idx = rng.integers(0, n, size=(min(CHUNK, B - i), n))
        means[i:i + CHUNK] = diff[idx].mean(axis=1)
    delta = diff.mean()
    lo95, hi95 = np.percentile(means, 2.5), np.percentile(means, 97.5)
    lo_adj, hi_adj = np.percentile(means, q_lo), np.percentile(means, q_hi)
    opp = abs(r["pooled_se"]) > TIE and ((delta > 0) != (r["pooled_se"] > 0)) and delta != 0
    rev_ci95 = bool(opp and (lo95 > 0 or hi95 < 0))
    rev_adj = bool(opp and (lo_adj > 0 or hi_adj < 0))
    out.append(dict(**r, delta=delta, lo95=lo95, hi95=hi95, lo_adj=lo_adj, hi_adj=hi_adj,
                    point_reversal=bool(opp), rev_ci95=rev_ci95, rev_ci_adj=rev_adj,
                    p_boot_zero=float((means == 0).mean())))
O = pd.DataFrame(out)
os.makedirs(OUTDIR, exist_ok=True)
O.to_csv(f"{OUTDIR}/bonferroni.csv", index=False)
summ = dict(n_comparisons=m, alpha_adj=alpha_adj, B=B, seed=SEED,
            n_rev_ci95=int(O.rev_ci95.sum()), n_rev_ci_adj=int(O.rev_ci_adj.sum()),
            rev_ci95=O[O.rev_ci95][["family", "a", "b", "sub", "delta", "lo95", "lo_adj", "hi_adj"]].to_dict("records"),
            n_ci95_excl0=int(((O.lo95 > 0) | (O.hi95 < 0)).sum()), n_ci_adj_excl0=int(((O.lo_adj > 0) | (O.hi_adj < 0)).sum()),
            seconds=time.time() - t0)
json.dump(summ, open(f"{OUTDIR}/bonferroni.json", "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
