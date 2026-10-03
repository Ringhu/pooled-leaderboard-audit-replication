"""Compute every result number quoted in the paper text from paper/data/ and write paper/numbers.tex
(one \\newcommand per number). The text uses only these macros for computed results, so a rerun of the analysis
followed by this script updates the paper. `--check` regenerates in memory and fails if numbers.tex differs.

Usage (repo root): python3 paper/scripts/make_numbers.py [--check]
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "numbers.tex")
N = {}


def rd(p):
    return pd.read_csv(os.path.join(D, p))


def f3(x):
    return f"{x:.3f}"


def sg(x):  # signed, 3 decimals, LaTeX minus
    s = f"{abs(x):.3f}"
    return f"$+${s}" if x > 0 else (f"$-${s}" if x < 0 else s)


def ci(lo, hi):
    return f"[{sg(lo)}, {sg(hi)}]".replace("$+$", "")


def put(name, val):
    assert name.isalpha(), name
    assert name not in N, name
    N[name] = str(val)


# ------------------------------------------------------------------ design
meta = json.load(open(os.path.join(D, "long_meta.json")))
cfg = json.load(open(os.path.join(D, "rq1/config.json")))
assert cfg["long_sha256"] == meta["sha256"]
put("NlongRows", f"{meta['rows']:,}")
put("longShaShort", meta.get("released_sha256", meta["sha256"])[:8])  # digest of the released (anonymized) table
pairs = rd("rq1/pairs.csv")
pm = pairs[pairs.variant == "mean"]
put("NpairAnalyses", len(pm))
put("NcwDiff", int(((pm.n_rev_cw != pm.n_rev_se) | (pm.n_rev_ci_cw != pm.n_rev_ci_se)).sum()))

# ------------------------------------------------------------------ RQ1: published pairs per family
FAM = {"RE1": "ReOne", "openrca": "Openrca", "petshop": "Petshop", "RE2": "ReTwo"}
LAY = {"L1": "Pub", "L2": "All"}
for fam, fn in FAM.items():
    for lay, ln in LAY.items():
        x = pm[(pm.family == fam) & (pm.layer == lay)]
        put(f"N{fn}{ln}Pairs", len(x))
        put(f"N{fn}{ln}Rev", int((x.n_rev_se > 0).sum()))
        put(f"N{fn}{ln}RevCI", int((x.n_rev_ci_se > 0).sum()))
        if x.pi_crosses_0.notna().any():
            put(f"N{fn}{ln}PIn", int(x.pi_crosses_0.notna().sum()))
            put(f"N{fn}{ln}PIcross", int((x.pi_crosses_0 == True).sum()))  # noqa: E712
main3 = pm[pm.family.isin(["RE1", "openrca", "petshop"])]
for lay, ln in LAY.items():
    x = main3[main3.layer == lay]
    put(f"NMain{ln}Pairs", len(x))
    put(f"NMain{ln}Rev", int((x.n_rev_se > 0).sum()))
    put(f"NMain{ln}RevCI", int((x.n_rev_ci_se > 0).sum()))


def pair(fam, lay, a, b, variant="mean"):
    r = pairs[(pairs.family == fam) & (pairs.layer == lay) & (pairs.variant == variant) & (pairs.a == a) & (pairs.b == b)]
    assert len(r) == 1, (fam, lay, a, b, variant)
    return r.iloc[0]


sub = rd("rq1/pair_subsystem_deltas.csv")


def sd(fam, lay, a, b, s, variant="mean"):
    r = sub[(sub.family == fam) & (sub.layer == lay) & (sub.variant == variant) & (sub.a == a) & (sub.b == b) & (sub["sub"] == s)]
    assert len(r) == 1, (fam, a, b, s)
    return r.iloc[0]


# RE1: CausalRCA vs RCD
r = pair("RE1", "L1", "causalrca", "rcd")
put("ReOneCrRcdPooled", sg(r.pooled_se)); put("ReOneCrRcdPooledCI", ci(r.pooled_se_lo, r.pooled_se_hi))
for s, nm in (("RE1::OB", "OB"), ("RE1::SS", "SS"), ("RE1::TT", "TT")):
    put(f"ReOneCrRcd{nm}", sg(sd("RE1", "L1", "causalrca", "rcd", s).delta))
for k, nm in ((0, "Zero"), (1, "One"), (2, "Two")):
    rr = pair("RE1", "L1", "causalrca", "rcd", f"seed{k}")
    put(f"ReOneCrRcdSeed{nm}", sg(rr.pooled_se))
    put(f"ReOneCrRcdSeed{nm}CI", int(rr.n_rev_ci_se))
# RE1: BARO vs others, margin range
bx = pm[(pm.family == "RE1") & (pm.layer == "L1") & (pm.a == "baro")]
assert (bx.n_rev_se == 0).all() and len(bx) == 4
put("ReOneBaroMinMargin", f3(bx.pooled_se.abs().min())); put("ReOneBaroMaxMargin", f3(bx.pooled_se.abs().max()))
# RE1 CausalRCA vs eps-diag (below line, no reversal); PetShop CIRCA vs counterfactual (above line, reverses)
md = rd("rq1/margin_dispersion.csv")


def mdr(fam, lay, a, b):
    r = md[(md.family == fam) & (md.layer == lay) & (md.a == a) & (md.b == b)]
    assert len(r) == 1
    return r.iloc[0]


r = mdr("RE1", "L1", "causalrca", "e_diagnosis"); assert r.margin_lt_sd and not r.reverses
put("ExBelowNoRevMargin", f3(r.abs_margin)); put("ExBelowNoRevSD", f3(r.sd_sub_delta))
r = mdr("petshop", "L1", "circa", "counterfactual_attribution"); assert (not r.margin_lt_sd) and r.reverses
put("ExAboveRevMargin", f3(r.abs_margin)); put("ExAboveRevSD", f3(r.sd_sub_delta))
put("ExAboveRevHigh", sg(sd("petshop", "L1", "circa", "counterfactual_attribution", "petshop::high_traffic").delta))

# OpenRCA
acc = rd("rq1/accuracy_by_subsystem.csv").set_index("method")
ocols = [c for c in acc.columns if c.startswith("openrca::")]
for m, nm in (("baro", "Baro"), ("rcd", "Rcd"), ("e_diagnosis", "Ediag")):
    put(f"Openrca{nm}Mean", f3(acc.loc[m, ocols].mean()))
r = pair("openrca", "L1", "baro", "rcd")
put("OpenrcaBaroRcdPooled", sg(r.pooled_se)); put("OpenrcaBaroRcdPooledCI", ci(r.pooled_se_lo, r.pooled_se_hi))
t = sd("openrca", "L1", "baro", "rcd", "openrca::tel")
put("OpenrcaBaroRcdTel", sg(t.delta)); put("OpenrcaBaroRcdTelCI", ci(t.lo, t.hi))
revs = [str(pair("openrca", "L1", "baro", "rcd", f"seed{k}").rev_se) for k in (0, 1, 2)]
put("OpenrcaBaroRcdSeedsWithRev", sum(v != "nan" for v in revs))

# PetShop
cov = rd("rq2/coverage.csv")
h = cov[(cov.subsystem_id == "petshop::high_traffic") & (cov.method == "rcd")].iloc[0]
put("PetshopRcdHighHits", int(h.hits)); put("PetshopHighN", int(h.n))
for a, nm in (("baro", "Baro"), ("counterfactual_attribution", "Cfa")):
    t = sd("petshop", "L1", a, "rcd", "petshop::high_traffic")
    put(f"Petshop{nm}RcdHigh", sg(t.delta)); put(f"Petshop{nm}RcdHighCI", ci(t.lo, t.hi))
    r = pair("petshop", "L1", a, "rcd")
    assert r.pooled_se_lo < 0 < r.pooled_se_hi  # text: the pooled CI includes zero
    put(f"Petshop{nm}RcdPooled", sg(r.pooled_se)); put(f"Petshop{nm}RcdPooledCI", ci(r.pooled_se_lo, r.pooled_se_hi))
ss = rd("rq1/small_sample.csv")
x = ss[(ss.n_min == 25) & (ss.family == "petshop") & (ss.layer == "L1")]
put("PetshopPubRevNmin", int((x.n_rev > 0).sum())); put("PetshopPubRevCINmin", int((x.n_rev_ci > 0).sum()))
x = ss[(ss.n_min == 25) & (ss.family == "petshop") & (ss.layer == "L2")]
put("PetshopAllRevNmin", int((x.n_rev > 0).sum())); put("PetshopAllRevCINmin", int((x.n_rev_ci > 0).sum()))
pcols = [c for c in acc.columns if c.startswith("petshop::")]
put("PetshopRankedCorrMean", f3(acc.loc["ranked_correlation", pcols].mean()))
put("PetshopCircaMean", f3(acc.loc["circa", pcols].mean()))

# ------------------------------------------------------------------ margin vs spread
for lay, ln in LAY.items():
    x = md[md.layer == lay]
    for below, bn in ((True, "Below"), (False, "Above")):
        y = x[x.margin_lt_sd == below]
        put(f"N{ln}{bn}", len(y)); put(f"N{ln}{bn}Rev", int(y.reverses.sum())); put(f"N{ln}{bn}RevCI", int(y.reverses_ci.sum()))
pt = rd("rq1/published_tables_margin_dispersion.csv")
for key, nm in (("RCAEval", "TabRcaeval"), ("PetShop", "TabPetshop")):
    x = pt[pt.table.str.startswith(key)]
    put(f"N{nm}Pairs", len(x))
    for below, bn in ((True, "Below"), (False, "Above")):
        y = x[x.margin_lt_sd == below]
        put(f"N{nm}{bn}", len(y)); put(f"N{nm}{bn}Rev", int((y.n_rev > 0).sum()))

tb = pt.assign(rev=pt.n_rev > 0)
put("NTabBelow", int(tb.margin_lt_sd.sum())); put("NTabBelowRev", int((tb.margin_lt_sd & tb.rev).sum()))
put("NTabAbove", int((~tb.margin_lt_sd).sum())); put("NTabAboveRev", int((~tb.margin_lt_sd & tb.rev).sum()))

# ------------------------------------------------------------------ cross-release
cr = rd("rq1/cross_release.csv")
x = cr[cr.layer == "L1"]
put("NCrossPub", len(x)); put("NCrossPubAgree", int(x.sign_agree.sum())); put("NCrossPubCIopp", int(x.ci_opposite.sum()))
x2 = cr[cr.layer == "L2"]
put("NCrossAll", len(x2)); put("NCrossAllAgree", int(x2.sign_agree.sum())); put("NCrossAllCIopp", int(x2.ci_opposite.sum()))


def crr(a, b, app):
    r = x[(x.a == a) & (x.b == b) & (x.app == app)]
    assert len(r) == 1
    return r.iloc[0]


for (a, b, app), nm in ((("baro", "circa", "SS"), "BaroCircaSS"), (("baro", "circa", "OB"), "BaroCircaOB"),
                        (("baro", "causalrca", "SS"), "BaroCrSS"), (("baro", "e_diagnosis", "SS"), "BaroEdSS")):
    r = crr(a, b, app)
    put(f"Cross{nm}One", sg(r.delta_re1)); put(f"Cross{nm}OneCI", ci(r.lo_re1, r.hi_re1))
    put(f"Cross{nm}Two", sg(r.delta_re2)); put(f"Cross{nm}TwoCI", ci(r.lo_re2, r.hi_re2))
fr = rd("rq2/cross_release_re1_faults.csv")
put("NCrossPubAgreeReOneFaults", int(fr[fr.layer == "L1"].agree_re1faults.sum()))

# ------------------------------------------------------------------ regret
rg = rd("rq1/regret.csv")
rg = rg[rg.weighting == "system_equal"]


def reg(fam, lay, s):
    r = rg[(rg.family == fam) & (rg.layer == lay) & (rg.subsystem == s)]
    assert len(r) == 1
    return r.iloc[0]


r = reg("RE2", "L1", "RE2::TT")
put("RegReTwoWinnerPooled", f3(r.winner_pooled)); put("RegReTwoTT", f3(r.regret)); put("RegReTwoTTbest", f3(r.best_acc))
put("RegReTwoTTwinner", f3(r.winner_acc))
r = reg("openrca", "L1", "openrca::tel"); put("RegOpenrcaTel", f3(r.regret))
put("RegPetshopHigh", f3(reg("petshop", "L1", "petshop::high_traffic").regret))
put("RegPetshopTmpOne", f3(reg("petshop", "L1", "petshop::temporal_traffic1").regret))
put("RegReOneAllSS", f3(reg("RE1", "L2", "RE1::SS").regret))
put("RegPetshopAllTmpTwo", f3(reg("petshop", "L2", "petshop::temporal_traffic2").regret))


# ------------------------------------------------------------------ RQ2
cc = rd("rq2/condition_comparisons.csv")
summ = json.load(open(os.path.join(D, "rq2/summary.json")))["summary"]
ia = rd("rq2/input_accuracy.csv")


def iacc(s, m):
    r = ia[(ia["sub"] == s) & (ia.method == m)]
    assert len(r) == 1
    return r.iloc[0]


for s, sn in (("RE1::SS", "SS"), ("RE1::TT", "TT")):
    for m, mn in (("baro", "Baro"), ("circa", "Circa"), ("rcd", "Rcd"), ("cd1min", "Cdmin"), ("alertcount", "Alert"),
                  ("max_z", "Maxz"), ("e_diagnosis", "Ediag")):
        r = iacc(s, m)
        put(f"In{sn}{mn}Simple", f3(r.acc_correct_input)); put(f"In{sn}{mn}Raw", f3(r.acc_submitted_input))
cd2 = ia[(ia["sub"].str.startswith("RE2")) & (ia.method == "cd1min")]
drop = cd2.acc_correct_input - cd2.acc_submitted_input
put("InReTwoCdminDropMin", f3(drop.min())); put("InReTwoCdminDropMax", f3(drop.max()))
r = iacc("RE2::TT", "mmbaro"); put("InReTwoTTMmbaroSimple", f3(r.acc_correct_input)); put("InReTwoTTMmbaroRaw", f3(r.acc_submitted_input))
x = cc[(cc.factor == "input_representation") & (cc["sub"] != "RE1::OB")]
for lay, ln in LAY.items():
    y = x[(x.family == "RE1") & (x.layer == lay)]
    put(f"InReOne{ln}Comp", len(y)); put(f"InReOne{ln}Sign", int(y.sign_change.sum()))
    for s, sn in (("RE1::SS", "SS"), ("RE1::TT", "TT")):
        z = y[y["sub"] == s]; put(f"InReOne{ln}Sign{sn}", int(z.sign_change.sum())); put(f"InReOne{ln}Comp{sn}", len(z))
    k = summ[f"input_RE1_{lay}"]
    put(f"InReOne{ln}RevRaw", k["rev_instances_alt"]); put(f"InReOne{ln}RevCIRaw", k["rev_ci_instances_alt"])
    put(f"InReOne{ln}RevOk", k["rev_instances_ref"]); put(f"InReOne{ln}RevCIOk", k["rev_ci_instances_ref"])
    put(f"InReOne{ln}RevRawOnly", k["rev_alt_absent_in_ref"]); put(f"InReOne{ln}RevOkOnly", k["rev_ref_absent_in_alt"])
y = x[(x.family == "RE2")]
put("InReTwoComp", len(y)); put("InReTwoSign", int(y.sign_change.sum()))


def ccr(factor, lay, a, b, s):
    r = cc[(cc.factor == factor) & (cc.layer == lay) & (cc.a == a) & (cc.b == b) & (cc["sub"] == s)]
    assert len(r) == 1, (factor, a, b, s)
    return r.iloc[0]


for s, sn in (("RE1::SS", "SS"), ("RE1::TT", "TT")):
    r = ccr("input_representation", "L1", "circa", "rcd", s)
    put(f"InCircaRcd{sn}Ok", sg(r.delta_ref)); put(f"InCircaRcd{sn}OkCI", ci(r.lo_ref, r.hi_ref))
    put(f"InCircaRcd{sn}Raw", sg(r.delta_alt)); put(f"InCircaRcd{sn}RawCI", ci(r.lo_alt, r.hi_alt))
kt = rd("rq2/kendall_tau.csv")
r = kt[(kt.factor == "input_representation") & (kt.layer == "L1") & (kt["sub"] == "RE1::SS")].iloc[0]
put("InTauSS", f3(r.kendall_tau)); assert (r.top_ref, r.top_alt) == ("baro", "rcd")
# release defect
for lay, ln in LAY.items():
    k = summ[f"release_defect_RE1_{lay}"]
    put(f"Def{ln}Comp", k["comparisons"]); put(f"Def{ln}Sign", k["sign_changes"])
    put(f"Def{ln}RevShipOnly", k["rev_alt_absent_in_ref"]); put(f"Def{ln}RevCIShipOnly", k["rev_ci_alt_absent_in_ref"])
    put(f"Def{ln}PooledFlips", k["pooled_sign_flips"])
ra = rd("rq2/release_defect_accuracy.csv").set_index("method")
put("DefCircaFixed", f3(ra.loc["circa", "acc_fixed"])); put("DefCircaShipped", f3(ra.loc["circa", "acc_shipped"]))
put("DefCircaFixedAff", f3(ra.loc["circa", "acc_fixed_affected"])); put("DefCircaShippedAff", f3(ra.loc["circa", "acc_shipped_affected"]))
r = ccr("release_defect", "L1", "causalrca", "circa", "RE1::OB")
put("DefCrCircaFixed", sg(r.delta_ref)); put("DefCrCircaFixedCI", ci(r.lo_ref, r.hi_ref))
put("DefCrCircaShipped", sg(r.delta_alt)); put("DefCrCircaShippedCI", ci(r.lo_alt, r.hi_alt))
r = ccr("release_defect", "L1", "circa", "rcd", "RE1::OB")
put("DefCircaRcdFixed", sg(r.delta_ref)); put("DefCircaRcdShipped", sg(r.delta_alt))
# completeness / denominator
cv = rd("rq2/coverage.csv")


def cov1(s, m, seed=None):
    r = cv[(cv.subsystem_id == s) & (cv.method == m)]
    if seed is not None:
        r = r[r.seed == seed]
    assert len(r) == 1, (s, m, seed)
    return r.iloc[0]


r = cov1("RE2::TT", "tracerca"); put("CovTraceValid", int(r.n_valid)); put("CovTraceAll", f3(r.acc_all)); put("CovTraceValidAcc", f3(r.acc_valid_only))
for sname, sn in (("RE1::TT", "ReOneTT"), ("RE2::TT", "ReTwoTT")):
    for k, kn in ((0.0, "Zero"), (1.0, "One"), (2.0, "Two")):
        put(f"CovCr{sn}Seed{kn}", f3(cov1(sname, "causalrca", k).acc_all))
put("CovCrReTwoTTMean", f3(acc.loc["causalrca", "RE2::TT"]))
put("CovMicroValid", int(cov1("RE2::TT", "microrank").n_valid))
put("CovCrSeedOneValid", int(cov1("RE2::TT", "causalrca", 1.0).n_valid))
r = cov1("openrca::tel", "e_diagnosis"); put("CovEdTelValid", int(r.n_valid)); put("CovEdTelN", int(r.n))
r = cov1("petshop::high_traffic", "epsilon_diagnosis"); put("CovEdHighValid", int(r.n_valid))
put("NCovBelowOne", int((cv.coverage < 1).sum())); put("NCovCells", len(cv))
dv = rd("rq2/denominator_valid_only.csv")
for fam, fn in (("RE2", "ReTwo"),):
    for lay, ln in LAY.items():
        y = dv[(dv.family == fam) & (dv.layer == lay)]
        put(f"Den{fn}{ln}Comp", len(y)); put(f"Den{fn}{ln}Sign", int(y.sign_change.sum()))
put("DenOtherSign", int(dv[(dv.family != "RE2") & (dv.layer == "L2")].sign_change.sum()))  # L2 (all pairs) includes the published pairs
put("DenOtherComp", int(((dv.family != "RE2") & (dv.layer == "L2")).sum()))
r = dv[(dv.layer == "L1") & (dv.a == "circa") & (dv.b == "tracerca") & (dv["sub"] == "RE2::TT")].iloc[0]
put("DenCircaTraceAll", sg(r.delta_ref)); put("DenCircaTraceValid", sg(r.delta_alt))
# fault composition
fw = rd("rq2/fault_reweighted.csv")
for fam, fn in (("RE1", "ReOne"), ("RE2", "ReTwo"), ("openrca", "Openrca"), ("petshop", "Petshop")):
    for lay, ln in LAY.items():
        y = fw[(fw.family == fam) & (fw.layer == lay)]
        put(f"Fw{fn}{ln}Comp", len(y)); put(f"Fw{fn}{ln}Sign", int(y.sign_change.sum()))
        put(f"Fw{fn}{ln}SignCI", int((y.sign_change & y.rev_ci_rw).sum()))
fc = rd("rq2/fault_composition.csv").set_index("subsystem_id")
re_counts = fc.loc[[i for i in fc.index if i.startswith("RE")]]
assert all(len(set(v[v > 0])) == 1 for _, v in re_counts.iterrows())  # equal counts per fault type
# scoring
for rule, rn in (("scoring_strict", "Strict"), ("scoring_ge", "Half")):
    for lay, ln in LAY.items():
        k = summ[f"{rule if rule != 'scoring_ge' else 'scoring_ge05'}_{lay}"]
        put(f"Sc{rn}{ln}Comp", k["comparisons"]); put(f"Sc{rn}{ln}Sign", k["sign_changes"]); put(f"Sc{rn}{ln}PooledFlips", k["pooled_sign_flips"])
y = cc[(cc.factor == "scoring_strict") & (cc.layer == "L1") & cc.sign_change]
put("ScStrictPubToZero", int((y.delta_alt == 0).sum()))
osa = rd("rq2/openrca_scoring_accuracy.csv")
for m, mn in (("baro", "Baro"), ("rcd", "Rcd"), ("e_diagnosis", "Ediag")):
    r = osa[(osa.subsystem_id == "openrca::mkt1") & (osa.method == m)].iloc[0]
    put(f"ScMktOne{mn}Strict", f3(r.score_strict)); put(f"ScMktOne{mn}Partial", f3(r.score))

# ------------------------------------------------------------------ one-line summary example (OpenRCA, BARO vs RCD)
r = pair("openrca", "L1", "baro", "rcd")
put("LineCw", sg(r.pooled_cw)); put("LineSe", sg(r.pooled_se))
put("LineSD", f3(r.sd_sub_delta)); put("LineWins", int(r.n_pos)); put("LineLosses", int(r.n_neg)); put("LineTies", int(r.k - r.n_pos - r.n_neg))
oc = cv[cv.subsystem_id.str.startswith("openrca::")]
for m, mn in (("baro", "Baro"), ("rcd", "Rcd")):
    y = oc[oc.method == m]
    put(f"LineCov{mn}", f"{y.n_valid.sum() / y.n.sum():.2f}")
t = sd("openrca", "L1", "baro", "rcd", "openrca::tel")
put("LineWorst", sg(t.delta)); put("LineWorstCI", ci(t.lo, t.hi))

# ------------------------------------------------------------------ multiplicity check (rq1_bonferroni.py)
bf = json.load(open(os.path.join(D, "rq1/bonferroni.json")))
assert bf["n_comparisons"] == 82 and str(bf["n_rev_ci95"]) == N["NMainPubRevCI"]
put("NBonfComp", bf["n_comparisons"]); put("NBonfRevCI", bf["n_rev_ci_adj"])
put("NBonfCIExcl", bf["n_ci95_excl0"]); put("NBonfCIAdjExcl", bf["n_ci_adj_excl0"])

# ------------------------------------------------------------------ survey
# Sample: the papers in survey/papers.csv minus survey/excluded.csv (no full text obtained).
# Items reported: per-system reporting (survey/per_system.csv) and the repository release check.
ex_ = rd("survey/excluded.csv")
assert len(ex_) == 3
q1_ = rd("survey/per_system.csv")
rc_ = rd("survey/release_check.csv")
assert len(rc_) == 26 and set(rc_.L1) <= {"yes", "no", "undeterminable"}
assert not q1_.id.isin(ex_.id).any()
rc_ = rc_[~rc_.id.isin(ex_.id)]
assert len(q1_) == 23 and len(rc_) == 23 and set(q1_.id) == set(rc_.id)
assert set(q1_.per_system) == {"yes", "single_system"}  # every multi-system paper reports per-system scores
put("SurveyN", len(q1_))
put("SurveyMulti", int((q1_.per_system == "yes").sum()))
put("SurveySingle", int((q1_.per_system == "single_system").sum()))
kind_ = rc_.kind.fillna("")
assert set(rc_[rc_.L1 == "yes"].id) == {"baro2024", "circa2022", "rcaeval2025", "howfar2024", "chase2024", "openrca2025", "causalrca2023"}  # named in sec:bg-survey
assert set(kind_[rc_.L1 == "yes"]) == {"demo_case", "own_method", "all_methods_one_table"} and (kind_[rc_.L1 != "yes"] == "").all()
put("SurveyRelAny", int((rc_.L1 == "yes").sum()))
put("SurveyRelDemo", int((kind_ == "demo_case").sum()))
put("SurveyRelOwn", int((kind_ == "own_method").sum()))
assert int((kind_ == "all_methods_one_table").sum()) == 1 and set(rc_[kind_ == "all_methods_one_table"].id) == {"circa2022"}
assert set(rc_[rc_.L1 == "undeterminable"].id) == {"tracerca2021", "toomanycooks2025"} and (rc_[rc_.L1 == "undeterminable"].L3 == "undeterminable").all()
put("SurveyRelUndet", int((rc_.L1 == "undeterminable").sum()))
assert int((rc_.L3 == "yes").sum()) == 0 and set(rc_[rc_.L3 == "partial"].id) == {"circa2022"}
put("SurveyRelFullCheckable", int((rc_.L3 != "undeterminable").sum()))

# ------------------------------------------------------------------ restored analyses (user 2026-09-30)
# per-family I^2 and interaction tests, published layer
for fam, fn in (("openrca", "Openrca"), ("petshop", "Petshop")):
    x = pm[(pm.family == fam) & (pm.layer == "L1") & pm.I2.notna()]
    put(f"Isq{fn}PubMin", f"{x.I2.min():.0f}"); put(f"Isq{fn}PubMax", f"{x.I2.max():.0f}")
for fam, fn in (("RE1", "ReOne"), ("openrca", "Openrca"), ("petshop", "Petshop")):
    x = pm[(pm.family == fam) & (pm.layer == "L1")]
    fit = x[x.separation != True]  # noqa: E712
    put(f"Int{fn}PubFit", len(fit)); put(f"Int{fn}PubSig", int((fit.lrt_p_holm < 0.05).sum()))
    put(f"Int{fn}PubSep", int((x.separation == True).sum()))  # noqa: E712

# cross-family comparison set: every method with outputs on all 11 subsystems (2026-10-02: six methods; the earlier
# four-method set in rq1_cross/ predates the OpenRCA RCD runs and is no longer used in the text)
xc = json.load(open(os.path.join(D, "rq1_cross6/config.json")))
assert xc["long_sha256"] == meta["sha256"] and len(xc["subsystems"]) == 11
put("XNMethods", len(xc["set4"]))
xp = rd("rq1_cross6/cross_pairs.csv")
mixed = (xp.n_pos > 0) & (xp.n_neg > 0)
put("XNPairs", len(xp)); put("XK", int(xp.k.iloc[0]))
put("XNBothSigns", int(mixed.sum())); put("XNOneSign", int((~mixed).sum()))
one = xp[~mixed]
assert ((one.a == "e_diagnosis") | (one.b == "e_diagnosis")).all() and len(one) == len(xc["set4"]) - 1
put("XNRevCI", int((xp.n_rev_ci_se > 0).sum())); put("XNBelow", int((xp.pooled_se.abs() < xp.sd_sub_delta).sum()))
fit = xp[xp.I2.notna()]
put("XNFit", len(fit)); put("XIsqMin", f"{fit.I2.min():.0f}"); put("XIsqMax", f"{fit.I2.max():.0f}")
put("XNPIcross", int((fit.pi_crosses_0 == True).sum()))  # noqa: E712
put("XNLRTsig", int((xp.lrt_p_holm < 0.05).sum())); put("XNSep", int((xp.separation == True).sum()))  # noqa: E712
ji = xc["joint_interaction"]
put("XJointLRT", f"{ji['method_x_subsystem']['lrt']:.1f}"); put("XJointDf", ji["method_x_subsystem"]["df"])
assert ji["method_x_subsystem"]["p"] < 1e-10
mk = set(map(tuple, xp.loc[mixed, ["a", "b"]].to_numpy()))
def only_mixed(t):
    return t[[(a, b) in mk for a, b in zip(t.a, t.b)]]
xl = rd("rq1_cross6/cross_loso.csv")
xg = only_mixed(xl[xl.rule == "mean"]).groupby(["a", "b"]).agg(rev=("reversal", "sum"), mx=("regret", "max"), mn=("regret", "mean"))
assert len(xg) == len(mk)
put("XLosoNAllRev", int((xg.rev > 0).sum()))
put("XLosoMin", int(xg.rev.min())); put("XLosoMax", int(xg.rev.max()))
put("XLosoMaxRegret", f3(xg.mx.max())); put("XLosoMeanRegretMax", f3(xg.mn.max()))
assert xg.rev.idxmax() == ("baro", "max_z")
bm = xp[(xp.a == "baro") & (xp.b == "max_z")].iloc[0]
put("XBaroMaxzPooled", sg(bm.pooled_se)); put("XBaroMaxzLoso", int(xg.loc[("baro", "max_z"), "rev"]))
lf = only_mixed(rd("rq1_cross6/cross_lofo.csv"))
put("XLofoN", len(lf)); put("XLofoBoth", int(((lf.n_pos > 0) & (lf.n_neg > 0)).sum()))
put("XLofoPIn", int(lf.pi_crosses_0.notna().sum())); put("XLofoPI", int((lf.pi_crosses_0 == True).sum()))  # noqa: E712
lk = only_mixed(rd("rq1_cross6/cross_lkso2.csv"))
put("XLksoN", int(lk.n_drops.sum())); put("XLksoBoth", int(lk.both_signs.sum()))
assert lk.n_drops.nunique() == 1; put("XLksoPer", int(lk.n_drops.iloc[0]))
pr = rd("rq1_cross6/cross_pooling_rules.csv")
ref = pr[pr.rule == "mean"].set_index(["a", "b"]).pooled
flip = pr[pr.rule != "mean"].apply(lambda r: (r.pooled > 0) != (ref[(r.a, r.b)] > 0), axis=1)
put("XPoolFlipPairs", pr[pr.rule != "mean"][flip].groupby(["a", "b"]).ngroups)
bt = only_mixed(rd("rq1_cross6/cross_boot_reversal.csv"))
put("XBootBothMin", f"{bt.share_both_signs.min():.3f}")

# within-family held-out subsystems (leaderboard of the other subsystems, adjacent pairs, published layer)
hc = json.load(open(os.path.join(D, "holdout/config.json")))
assert hc["long_sha256"] == meta["sha256"]
cv = rd("holdout/curve.csv")
base = dict(layer="L1", set="adjacent", held_filter="all", scope="all")
def cvrow(th, **kw):
    q = cv
    for k, v in {**base, **kw}.items():
        q = q[q[k] == v]
    q = q[pd.to_numeric(q.threshold, errors="coerce") == th]
    assert len(q) == 1
    return q.iloc[0]
c0 = cvrow(0.0); c1 = cvrow(1.0)
put("HNDec", int(c0.n_decisions)); put("HErr", int(c0.system_equal_errors)); put("HTies", int(c0.system_equal_ties))
put("HStrict", int(c0.system_equal_strict_errors)); put("HErrCw", int(c0.pooled_only_errors))
put("HRec", int(c1.recommended)); put("HRecErr", int(c1.rule_errors))
put("HFlag", int(c1.flagged)); put("HFlagErr", int(c1.flagged_errors)); put("HFlagTies", int(c1.flagged_ties))
put("HBudgetErr", int(c1.system_equal_errors)); put("HBudgetCwErr", int(c1.pooled_only_errors))
for sc, nm in (("RE1", "ReOne"), ("openrca", "Openrca"), ("petshop", "Petshop")):
    r_ = cvrow(1.0, scope=sc)
    put(f"H{nm}Rec", int(r_.recommended)); put(f"H{nm}RecErr", int(r_.rule_errors))
    put(f"H{nm}Flag", int(r_.flagged)); put(f"H{nm}FlagErr", int(r_.flagged_errors))
fr = rd("rq1_cross6/family_loso_regret.csv")
for lay, ln in (("L1", "Pub"), ("L2", "All")):
    x = fr[fr.layer == lay]
    put(f"LosoReg{ln}N", int((x.regret > 1e-12).sum())); put(f"LosoReg{ln}Of", len(x))
    put(f"LosoReg{ln}Max", f3(x.regret.max()))

# ------------------------------------------------------------------ write / check
body = "% generated by paper/scripts/make_numbers.py from paper/data/ -- do not edit\n" + "".join(
    f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n" for k, v in N.items())
if "--check" in sys.argv:
    old = open(OUT).read() if os.path.exists(OUT) else ""
    if old != body:
        sys.exit("numbers.tex is stale: rerun make_numbers.py")
    print(f"numbers.tex up to date ({len(N)} numbers)")
else:
    open(OUT, "w").write(body)
    print(f"wrote {OUT} ({len(N)} numbers)")
