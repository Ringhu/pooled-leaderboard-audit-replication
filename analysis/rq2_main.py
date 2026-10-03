"""RQ2 (main-thread implementation): what changes the comparison conclusions.

Follows .research/plans/2026-09-29-wp4-analysis-spec.md; reuses the pair machinery of rq1_main.analyse so that deltas,
bootstrap CIs and reversal definitions are identical to RQ1. Each factor compares a reference condition (the main
table) with an alternative condition on the same cases and the same method set.

(a) input representation
    RE1: alternative = the submitted papers' input: SS/TT from data.csv (role contrast_input), OB unchanged (OB only has
    data.csv, which is already the reduced table). Method set = methods with data.csv rows on SS and TT.
    RE2: alternative = metrics.csv (+ logs/traces for mmbaro) instead of simple_metrics.csv; methods with both only.
    G4 = share of reversals under the alternative (submitted) input that are absent under the correct input.
(a2) release defect: RE1-OB duplicate "time" header in 50 cpu/mem cases (fixed upstream in RCAEval 21ff8c9): main table
    (rerun with the duplicate column dropped) vs the as-shipped outputs (role contrast_release_defect).
(b) output completeness: coverage per method x subsystem (valid_output; seeds separately); alternative denominator =
    accuracy over valid outputs only (sum score / n valid) -- point estimates only (case sets differ between methods).
(d) scoring rule (OpenRCA): official partial score (reference) vs strict (score == 1) vs partial score >= 0.5.
(e) summary per factor: share of (pair x subsystem) deltas whose sign differs from the reference; per-subsystem
    Kendall tau between the method orderings of reference and alternative.
(c) fault composition is in rq2_fault.py.
Usage (3090, auditstack env): python rq2_main.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_main as R  # noqa: E402

OUT = f"{R.M}/analysis/main/rq2"
PREFIX = {"RE1": "RE1::", "RE2": "RE2::", "openrca": "openrca::", "petshop": "petshop::"}


def run_pairs(d, fam, layer, roles):
    p, s = R.analyse(d, fam, PREFIX[fam], layer, roles, "mean", R.SEED)
    return pd.DataFrame(p), pd.DataFrame(s)


def compare(ref_s, alt_s, ref_p, alt_p, factor, fam, layer):
    """Join subsystem-level deltas of two conditions; flag sign changes and reversal status in each."""
    k = ["a", "b", "sub"]
    m = ref_s[k + ["n", "acc_a", "acc_b", "delta", "lo", "hi"]].merge(
        alt_s[k + ["acc_a", "acc_b", "delta", "lo", "hi"]], on=k, suffixes=("_ref", "_alt"))
    pooled = {}
    for tag, P in (("ref", ref_p), ("alt", alt_p)):
        for _, r in P.iterrows():
            pooled[(tag, r.a, r.b)] = r.pooled_se
    for tag in ("ref", "alt"):
        pl = np.array([pooled[(tag, a, b)] for a, b in zip(m.a, m.b)])
        dl, lo, hi = m[f"delta_{tag}"], m[f"lo_{tag}"], m[f"hi_{tag}"]
        m[f"pooled_{tag}"] = pl
        m[f"rev_{tag}"] = (np.abs(pl) > R.TIE) & (dl != 0) & (np.sign(dl) != np.sign(pl))
        m[f"rev_ci_{tag}"] = ((pl > R.TIE) & (hi < 0)) | ((pl < -R.TIE) & (lo > 0))
    m["sign_change"] = np.sign(m.delta_ref) != np.sign(m.delta_alt)
    m.insert(0, "layer", layer)
    m.insert(0, "family", fam)
    m.insert(0, "factor", factor)
    return m


def taus(ref, alt, factor, fam, layer):
    """ref/alt: DataFrame index method, columns subsystem (accuracy)."""
    rows = []
    for s in ref.columns:
        common = ref.index.intersection(alt.index)
        t = kendalltau(ref.loc[common, s], alt.loc[common, s])
        rows.append(dict(factor=factor, family=fam, layer=layer, sub=s, n_methods=len(common), kendall_tau=t.statistic,
                         top_ref=ref.loc[common, s].idxmax(), top_alt=alt.loc[common, s].idxmax()))
    return rows


def acc_table(d, prefix, roles):
    x = d[d.subsystem_id.str.startswith(prefix) & d.role.isin(roles)]
    return x.groupby(["subsystem_id", "case_id", "method"]).score.mean().groupby(["subsystem_id", "method"]).mean().unstack(0)


def main():
    t0 = time.time()
    full = pd.read_csv(R.LONG, low_memory=False)
    assert R.sha(R.LONG) == R.SHA
    main_ = full[full.role.isin(["published", "probe", "shipped_baseline"])].copy()
    os.makedirs(OUT, exist_ok=True)
    comps, tau_rows, summ = [], [], {}

    # ---------------- (a) input representation
    main_role = main_.drop_duplicates(["family", "method"]).set_index(["family", "method"]).role
    for fam, subs_alt in (("RE1", ("RE1::SS", "RE1::TT")), ("RE2", ("RE2::OB", "RE2::SS", "RE2::TT"))):
        con = full[(full.role == "contrast_input") & full.subsystem_id.isin(subs_alt)].copy()
        meths = sorted(set.intersection(*[set(con[con.subsystem_id == s].method) for s in subs_alt]))
        con["role"] = [main_role[("rcaeval", m)] for m in con.method]
        ref = main_[main_.subsystem_id.str.startswith(PREFIX[fam]) & main_.method.isin(meths)]
        alt = pd.concat([ref[~ref.subsystem_id.isin(subs_alt)], con[con.method.isin(meths)]])
        for layer, roles in R.LAYERS.items():
            if ref[ref.role.isin(roles)].method.nunique() < 2:
                continue
            rp, rs = run_pairs(ref, fam, layer, roles)
            ap, as_ = run_pairs(alt, fam, layer, roles)
            c = compare(rs, as_, rp, ap, "input_representation", fam, layer)
            comps.append(c)
            tau_rows += taus(acc_table(ref, PREFIX[fam], roles), acc_table(alt, PREFIX[fam], roles),
                             "input_representation", fam, layer)
            key = f"input_{fam}_{layer}"
            summ[key] = dict(methods=sorted(ref[ref.role.isin(roles)].method.unique()), pairs=len(rp),
                             rev_instances_alt=int(c.rev_alt.sum()), rev_instances_ref=int(c.rev_ref.sum()),
                             rev_alt_absent_in_ref=int((c.rev_alt & ~c.rev_ref).sum()),
                             rev_ref_absent_in_alt=int((c.rev_ref & ~c.rev_alt).sum()),
                             rev_ci_instances_alt=int(c.rev_ci_alt.sum()), rev_ci_instances_ref=int(c.rev_ci_ref.sum()),
                             rev_ci_alt_absent_in_ref=int((c.rev_ci_alt & ~c.rev_ci_ref).sum()),
                             rev_pairs_alt=int((ap.n_rev_se > 0).sum()),
                             rev_pairs_ref=int((rp.n_rev_se > 0).sum()),
                             sign_changes=int(c.sign_change.sum()), comparisons=len(c))
            if summ[key]["rev_instances_alt"]:
                summ[key]["G4_share_alt_reversals_gone"] = summ[key]["rev_alt_absent_in_ref"] / summ[key]["rev_instances_alt"]
    # ---------------- (a2) release defect: RE1-OB duplicate "time" header (decision 2026-09-30)
    # reference = main table (50 affected RE1-OB cases rerun with the duplicate column dropped); alternative = the
    # same table with those rows replaced by the as-shipped outputs (role contrast_release_defect).
    con = full[full.role == "contrast_release_defect"].copy()
    if len(con):
        con["role"] = [main_role[("rcaeval", m)] for m in con.method]
        key = lambda x: list(zip(x.case_id, x.method, x.seed.fillna(-1)))  # noqa: E731
        ref = main_[main_.subsystem_id.str.startswith("RE1::")]
        alt = pd.concat([ref[~pd.Series(key(ref), index=ref.index).isin(set(key(con)))], con])
        assert len(alt) == len(ref)
        for layer, roles in R.LAYERS.items():
            rp, rs = run_pairs(ref, "RE1", layer, roles)
            ap, as_ = run_pairs(alt, "RE1", layer, roles)
            c = compare(rs, as_, rp, ap, "release_defect", "RE1", layer)
            comps.append(c)
            tau_rows += taus(acc_table(ref, "RE1::", roles), acc_table(alt, "RE1::", roles), "release_defect",
                             "RE1", layer)
            summ[f"release_defect_RE1_{layer}"] = dict(
                methods_rerun=sorted(con.method.unique()), pairs=len(rp), comparisons=len(c),
                sign_changes=int(c.sign_change.sum()), rev_ref=int(c.rev_ref.sum()), rev_alt=int(c.rev_alt.sum()),
                rev_ci_ref=int(c.rev_ci_ref.sum()), rev_ci_alt=int(c.rev_ci_alt.sum()),
                rev_alt_absent_in_ref=int((c.rev_alt & ~c.rev_ref).sum()),
                rev_ci_alt_absent_in_ref=int((c.rev_ci_alt & ~c.rev_ci_ref).sum()),
                pooled_sign_flips=int(sum(np.sign(a) != np.sign(b) for a, b in zip(rp.pooled_se, ap.pooled_se))))
        ra = []
        for m, g in con.groupby("method"):
            r = main_[(main_.subsystem_id == "RE1::OB") & (main_.method == m)]
            aff = set(g.case_id)
            ra.append(dict(method=m, acc_fixed=r.groupby("case_id").score.mean().mean(),
                           acc_shipped=alt[(alt.subsystem_id == "RE1::OB") & (alt.method == m)]
                           .groupby("case_id").score.mean().mean(),
                           acc_fixed_affected=r[r.case_id.isin(aff)].groupby("case_id").score.mean().mean(),
                           acc_shipped_affected=g.groupby("case_id").score.mean().mean(), n_affected=len(aff)))
        pd.DataFrame(ra).to_csv(f"{OUT}/release_defect_accuracy.csv", index=False)
    # per-method accuracy under both inputs
    acc_in = []
    for fam, subs_alt in (("RE1", ("RE1::SS", "RE1::TT")), ("RE2", ("RE2::OB", "RE2::SS", "RE2::TT"))):
        con = full[(full.role == "contrast_input") & full.subsystem_id.isin(subs_alt)]
        for (s, m), g in con.groupby(["subsystem_id", "method"]):
            r = main_[(main_.subsystem_id == s) & (main_.method == m)]
            acc_in.append(dict(sub=s, method=m, acc_correct_input=r.groupby("case_id").score.mean().mean(),
                               acc_submitted_input=g.groupby("case_id").score.mean().mean(),
                               input_correct=";".join(r.input_repr.unique()), input_submitted=";".join(g.input_repr.unique())))
    pd.DataFrame(acc_in).to_csv(f"{OUT}/input_accuracy.csv", index=False)

    # ---------------- (d) scoring rule (OpenRCA)
    o = main_[main_.family == "openrca"].copy()
    for rule in ("strict", "ge05"):
        alt = o.copy()
        alt["score"] = alt.score_strict if rule == "strict" else (alt.score >= 0.5).astype(float)
        for layer, roles in R.LAYERS.items():
            rp, rs = run_pairs(o, "openrca", layer, roles)
            ap, as_ = run_pairs(alt, "openrca", layer, roles)
            c = compare(rs, as_, rp, ap, f"scoring_{rule}", "openrca", layer)
            comps.append(c)
            tau_rows += taus(acc_table(o, "openrca::", roles), acc_table(alt, "openrca::", roles), f"scoring_{rule}",
                             "openrca", layer)
            pooled_flip = int((np.sign(rp.set_index(["a", "b"]).pooled_se) !=
                               np.sign(ap.set_index(["a", "b"]).pooled_se)).sum())
            summ[f"scoring_{rule}_{layer}"] = dict(pairs=len(rp), pooled_sign_flips=pooled_flip,
                                                   rev_instances_ref=int(c.rev_ref.sum()), rev_instances_alt=int(c.rev_alt.sum()),
                                                   rev_ci_ref=int(c.rev_ci_ref.sum()), rev_ci_alt=int(c.rev_ci_alt.sum()),
                                                   sign_changes=int(c.sign_change.sum()), comparisons=len(c))
    sc = o.groupby(["subsystem_id", "case_id", "method"])[["score", "score_strict"]].mean().reset_index()
    sc["ge05"] = (sc.score >= 0.5).astype(float)
    sc.groupby(["subsystem_id", "method"])[["score", "score_strict", "ge05"]].mean().reset_index().to_csv(
        f"{OUT}/openrca_scoring_accuracy.csv", index=False)

    # ---------------- (b) output completeness
    cov = main_.groupby(["subsystem_id", "method", "seed"], dropna=False).agg(
        n=("score", "size"), coverage=("valid_output", "mean"), acc_all=("score", "mean"),
        n_valid=("valid_output", "sum"), hits=("score", "sum")).reset_index()
    cov["acc_valid_only"] = cov.hits / cov.n_valid.where(cov.n_valid > 0)
    cov.to_csv(f"{OUT}/coverage.csv", index=False)
    vo = []
    for fam in ("RE1", "RE2", "openrca", "petshop"):
        x = main_[main_.subsystem_id.str.startswith(PREFIX[fam])]
        for layer, roles in R.LAYERS.items():
            y = x[x.role.isin(roles)]
            # per method x subsystem: mean over seeds of per-seed accuracy (all cases vs valid-only)
            per_seed = y.groupby(["subsystem_id", "method", "seed"], dropna=False).apply(
                lambda g: pd.Series(dict(all=g.score.mean(), valid=g.score.sum() / g.valid_output.sum()
                                         if g.valid_output.sum() else np.nan)), include_groups=False)
            acc = per_seed.groupby(level=[0, 1]).mean()
            a_all, a_val = acc["all"].unstack(0), acc["valid"].unstack(0)
            tau_rows += taus(a_all, a_val.fillna(0), "denominator_valid_only", fam, layer)
            ms = sorted(a_all.index)
            for i, a in enumerate(ms):
                for b in ms[i + 1:]:
                    d_all = (a_all.loc[a] - a_all.loc[b]).dropna()
                    d_val = (a_val.loc[a] - a_val.loc[b]).dropna()
                    common = d_all.index.intersection(d_val.index)
                    if len(common) < 2:
                        continue
                    pa, pv = d_all[common].mean(), d_val[common].mean()
                    for s in common:
                        vo.append(dict(factor="denominator_valid_only", family=fam, layer=layer, a=a, b=b, sub=s,
                                       delta_ref=d_all[s], delta_alt=d_val[s], pooled_ref=pa, pooled_alt=pv,
                                       rev_ref=bool(abs(pa) > R.TIE and d_all[s] != 0 and np.sign(d_all[s]) != np.sign(pa)),
                                       rev_alt=bool(abs(pv) > R.TIE and d_val[s] != 0 and np.sign(d_val[s]) != np.sign(pv)),
                                       sign_change=bool(np.sign(d_all[s]) != np.sign(d_val[s]))))
    vo = pd.DataFrame(vo)
    vo.to_csv(f"{OUT}/denominator_valid_only.csv", index=False)
    for (fam, layer), g in vo.groupby(["family", "layer"]):
        summ[f"denominator_{fam}_{layer}"] = dict(comparisons=len(g), sign_changes=int(g.sign_change.sum()),
                                                  rev_ref=int(g.rev_ref.sum()), rev_alt=int(g.rev_alt.sum()),
                                                  rev_only_valid_only=int((g.rev_alt & ~g.rev_ref).sum()),
                                                  rev_only_all_cases=int((g.rev_ref & ~g.rev_alt).sum()))

    C = pd.concat(comps)
    C.to_csv(f"{OUT}/condition_comparisons.csv", index=False)
    T = pd.DataFrame(tau_rows)
    T.to_csv(f"{OUT}/kendall_tau.csv", index=False)
    cfg = dict(script_sha256=R.sha(os.path.abspath(__file__)), rq1_main_sha256=R.sha(R.__file__), long_sha256=R.SHA,
               B=R.B, seed=R.SEED, run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               wall_s=round(time.time() - t0, 1))
    json.dump(dict(config=cfg, summary=summ), open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
