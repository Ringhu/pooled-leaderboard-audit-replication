"""RQ3 (main-thread implementation): leave-one-subsystem-out validation of the margin-vs-dispersion rule.

Follows .research/plans/2026-09-29-wp6-rq3-spec.md (user-confirmed 2026-09-30). Input: merged_long.csv (sha256 checked
via rq1_main) and analysis/main/rq1/pair_subsystem_deltas.csv (held-out subsystem CIs, three-seed-mean variant; every
delta is re-checked against the long table here). Output: analysis/main/rq3/*.

Per family (RE1 k=3, openrca k=4, petshop k=4) and layer (L1, L2), each subsystem is held out in turn:
  training leaderboard  = methods present on every subsystem of the family, ranked by the system-equal mean over the
                          k-1 training subsystems (ties by method name)
  decisions             = adjacent pairs of that leaderboard (main set) or all pairs (supplementary); a = higher-ranked
  per decision          m_se  = mean of training subsystem deltas (a - b), m_cw = case-weighted training delta,
                          sd    = SD (ddof 1) of the training subsystem deltas, t = |m_se| / sd,
                          held-out delta and its RQ1 bootstrap CI
  rule                  recommend a over b iff m_se != 0 and t >= c (sd = 0 and m_se != 0 -> t = inf); c = 1 frozen
  baselines at the same number r of recommendations:
                          pooled-only   top r by |m_cw|, direction sign(m_cw)
                          system-equal  top r by |m_se|, direction sign(m_se)
                          random        expected errors r * K / N (K errors among the N decisions with a direction,
                                        rule direction); hypergeometric P(errors <= rule errors) as a descriptive number
  outcome               error: recommended side has the lower held-out mean; tie: held-out delta == 0 (not an error,
                          stays in the denominator); strict error: held-out CI entirely on the wrong side
Cross-release supplement: training = the three RE1 subsystems, held out = each RE2 subsystem (L1, adjacent pairs,
methods present on every RE1 and RE2 subsystem).
Stability lines: all k subsystems, reference BARO; W/T/L from the subsystem CI of (method - BARO).
Usage (3090, auditstack env): python rq3_holdout.py   (after rq1_main.py)
"""
import hashlib
import itertools
import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_main as R  # noqa: E402

M = R.M
RQ1 = f"{M}/analysis/main/rq1"
OUT = f"{M}/analysis/main/rq3"
FAM = {"RE1": "RE1::", "openrca": "openrca::", "petshop": "petshop::"}
THRESH = [round(0.25 * i, 2) for i in range(13)]  # 0, 0.25, ..., 3
C_FROZEN = 1.0
NMIN = 25
REF = "baro"
EPS = 1e-12


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Held:
    """Subsystem delta (a - b) with its RQ1 bootstrap CI, oriented for any (a, b); checked against the long table."""

    def __init__(self):
        p = pd.read_csv(f"{RQ1}/pair_subsystem_deltas.csv")
        p = p[p.variant == "mean"]
        self.t = {(r.family, r.layer, r.a, r.b, r["sub"]): (r.delta, r.lo, r.hi, int(r.n)) for _, r in p.iterrows()}
        self.checked = 0

    def get(self, fam, layer, a, b, sub, own_delta):
        if (fam, layer, a, b, sub) in self.t:
            d, lo, hi, n = self.t[(fam, layer, a, b, sub)]
        else:
            d, lo, hi, n = self.t[(fam, layer, b, a, sub)]
            d, lo, hi = -d, -hi, -lo
        assert abs(d - own_delta) < 1e-9, (fam, layer, a, b, sub, d, own_delta)
        self.checked += 1
        return d, lo, hi, n


def acc_table(d, prefix, roles):
    x = d[d.subsystem_id.str.startswith(prefix) & d.role.isin(roles)]
    cs = x.groupby(["subsystem_id", "case_id", "method"]).score.mean().reset_index()  # three-seed mean per case
    acc = cs.groupby(["subsystem_id", "method"]).score.mean().unstack(0)  # method x subsystem
    n = cs.groupby("subsystem_id").case_id.nunique()
    return x, acc, n


def outcome(direction, d, lo, hi):
    if direction == 0:
        return "none", False
    v = direction * d
    err = "tie" if abs(d) < EPS else ("error" if v < 0 else "correct")
    strict = (direction > 0 and hi < 0) or (direction < 0 and lo > 0)
    return err, bool(strict)


def decisions_for(fam, layer, acc_tr, n_tr, acc_ho, held_sub, held_fam, H, dset):
    """acc_tr: method x training subsystems; acc_ho: method -> held-out mean; one row per decision."""
    se = acc_tr.mean(axis=1)
    order = sorted(se.index, key=lambda m: (-se[m], m))
    if dset == "adjacent":
        pairs = list(zip(order[:-1], order[1:]))
    else:
        pairs = [(a, b) for a, b in itertools.combinations(order, 2)]
    rows = []
    for a, b in pairs:
        dl = (acc_tr.loc[a] - acc_tr.loc[b]).to_numpy()
        w = n_tr.reindex(acc_tr.columns).to_numpy()
        m_se, m_cw = float(dl.mean()), float((w * dl).sum() / w.sum())
        sd = float(np.std(dl, ddof=1))
        t = (abs(m_se) / sd) if sd > 0 else (np.inf if abs(m_se) > EPS else 0.0)
        dh, lo, hi, nh = H.get(held_fam, layer, a, b, held_sub, float(acc_ho[a] - acc_ho[b]))
        rows.append(dict(family=fam, layer=layer, set=dset, held_out=held_sub, k_train=len(dl), a=a, b=b,
                         rank_a=order.index(a) + 1, rank_b=order.index(b) + 1, se_a=float(se[a]), se_b=float(se[b]),
                         m_se=m_se, m_cw=m_cw, sd=sd, t=t, train_wins=int((dl > EPS).sum()),
                         train_ties=int((abs(dl) <= EPS).sum()), train_losses=int((dl < -EPS).sum()),
                         train_subs=";".join(acc_tr.columns), n_held=nh, d_held=dh, lo_held=lo, hi_held=hi))
    return rows


def annotate(D):
    D = D.copy()
    D["dir_rule"] = np.where(D.m_se.abs() > EPS, np.sign(D.m_se), 0).astype(int)
    D["dir_cw"] = np.where(D.m_cw.abs() > EPS, np.sign(D.m_cw), 0).astype(int)
    for tag, col in (("rule", "dir_rule"), ("cw", "dir_cw")):
        o = [outcome(r[col], r.d_held, r.lo_held, r.hi_held) for _, r in D.iterrows()]
        D[f"out_{tag}"] = [x[0] for x in o]
        D[f"strict_{tag}"] = [x[1] for x in o]
    D["tiebreak"] = D.family + "|" + D.layer + "|" + D.held_out + "|" + D.a + "|" + D.b
    return D


def counts(sub, tag):
    return dict(n=len(sub), errors=int((sub[f"out_{tag}"] == "error").sum()),
                ties=int((sub[f"out_{tag}"] == "tie").sum()), strict_errors=int(sub[f"strict_{tag}"].sum()))


def rate(c):
    return c["errors"] / c["n"] if c["n"] else np.nan


def curve(G, label):
    N0 = int((G.dir_rule != 0).sum())
    K = int(((G.dir_rule != 0) & (G.out_rule == "error")).sum())
    Ks = int(((G.dir_rule != 0) & G.strict_rule).sum())
    by_cw = G[G.dir_cw != 0].sort_values(["m_cw", "tiebreak"], key=lambda s: -s.abs() if s.name == "m_cw" else s)
    by_se = G[G.dir_rule != 0].sort_values(["m_se", "tiebreak"], key=lambda s: -s.abs() if s.name == "m_se" else s)
    rows = []
    points = [(c, (G.dir_rule != 0) & (G.t >= c)) for c in THRESH]
    points.append(("allwin", (G.dir_rule != 0) & (G.train_wins == G.k_train)))
    for c, allowed in points:
        A, F = G[allowed], G[~allowed]
        r = len(A)
        ca, cf = counts(A, "rule"), counts(F, "rule")
        cp, cs_ = counts(by_cw.head(r), "cw"), counts(by_se.head(r), "rule")
        rows.append(dict(**label, threshold=c, n_decisions=len(G), n_with_direction=N0, recommended=r,
                         coverage=r / len(G) if len(G) else np.nan,
                         rule_errors=ca["errors"], rule_ties=ca["ties"], rule_strict_errors=ca["strict_errors"],
                         rule_error_rate=rate(ca), rule_strict_error_rate=ca["strict_errors"] / r if r else np.nan,
                         flagged=len(F), flagged_errors=cf["errors"], flagged_ties=cf["ties"],
                         flagged_strict_errors=cf["strict_errors"], flagged_error_rate=rate(cf),
                         pooled_only_n=cp["n"], pooled_only_errors=cp["errors"], pooled_only_ties=cp["ties"],
                         pooled_only_strict_errors=cp["strict_errors"], pooled_only_error_rate=rate(cp),
                         system_equal_n=cs_["n"], system_equal_errors=cs_["errors"], system_equal_ties=cs_["ties"],
                         system_equal_strict_errors=cs_["strict_errors"], system_equal_error_rate=rate(cs_),
                         random_expected_errors=r * K / N0 if N0 else np.nan,
                         random_error_rate=K / N0 if N0 else np.nan,
                         random_expected_strict_errors=r * Ks / N0 if N0 else np.nan,
                         p_random_le_rule=float(hypergeom.cdf(ca["errors"], N0, K, r)) if N0 and r else np.nan))
    return rows


def main():
    t0 = time.time()
    d = R.load()  # sha256 checked
    cfg1 = json.load(open(f"{RQ1}/config.json"))
    assert cfg1["long_sha256"] == R.SHA, "RQ1 outputs are from another long table"
    os.makedirs(OUT, exist_ok=True)
    H = Held()

    # ---- leave-one-subsystem-out decisions
    rows, stab = [], []
    for fam, pre in FAM.items():
        for layer, roles in R.LAYERS.items():
            x, acc, n = acc_table(d, pre, roles)
            complete = acc.dropna()
            subs = list(complete.columns)
            for h in subs:
                tr = [s for s in subs if s != h]
                for dset in ("adjacent", "all"):
                    rows += decisions_for(fam, layer, complete[tr], n, complete[h], h, fam, H, dset)
            # ---- stability lines (all k subsystems, reference BARO)
            if REF not in complete.index:
                continue
            for m in complete.index:
                cov = x[x.method == m].valid_output.mean()
                rec = dict(family=fam, layer=layer, method=m, role=";".join(sorted(x[x.method == m].role.unique())),
                           k=len(subs), pooled_cw=float((complete.loc[m] * n[subs]).sum() / n[subs].sum()),
                           pooled_se=float(complete.loc[m].mean()), valid_coverage=float(cov),
                           input_repr=";".join(sorted(x[x.method == m].input_repr.unique())))
                if m != REF:
                    ds = [(s,) + H.get(fam, layer, m, REF, s, float(complete.loc[m, s] - complete.loc[REF, s])) for s in subs]
                    rec.update(vs_ref=REF, wins=sum(lo > 0 for _, _, lo, _, _ in ds),
                               ties=sum(lo <= 0 <= hi for _, _, lo, hi, _ in ds),
                               losses=sum(hi < 0 for _, _, _, hi, _ in ds))
                    s, dw, lo, hi, _ = min(ds, key=lambda z: z[1])
                    rec.update(worst_subsystem=s, worst_gap=dw, worst_gap_lo=lo, worst_gap_hi=hi,
                               per_subsystem=";".join(f"{s_}:{dd:+.3f}[{l_:+.3f},{h_:+.3f}]" for s_, dd, l_, h_, _ in ds))
                stab.append(rec)
    D = annotate(pd.DataFrame(rows))
    D.to_csv(f"{OUT}/decisions.csv", index=False)
    S = pd.DataFrame(stab).sort_values(["family", "layer", "pooled_se"], ascending=[True, True, False])
    for (fam, layer), g in S.groupby(["family", "layer"]):
        g.to_csv(f"{OUT}/stability_lines_{fam}_{layer}.csv", index=False)

    # ---- curves
    cur = []
    for layer in R.LAYERS:
        for dset in ("adjacent", "all"):
            for filt in ("all", f"n_held>={NMIN}"):
                G0 = D[(D.layer == layer) & (D.set == dset)]
                if filt != "all":
                    G0 = G0[G0.n_held >= NMIN]
                for scope in ("all",) + tuple(FAM):
                    G = G0 if scope == "all" else G0[G0.family == scope]
                    cur += curve(G, dict(layer=layer, set=dset, held_filter=filt, scope=scope))
    C = pd.DataFrame(cur)
    C.to_csv(f"{OUT}/curve.csv", index=False)

    # ---- G5 (spec §7): L1, adjacent, point definition, all held-out subsystems, c = 1
    def g5_of(scope):
        r = C[(C.layer == "L1") & (C.set == "adjacent") & (C.held_filter == "all") & (C.scope == scope)
              & (C.threshold == C_FROZEN)].iloc[0]
        ident = bool(r.flagged > 0 and r.recommended > 0 and r.flagged_error_rate > r.rule_error_rate)
        reduce_ = bool(r.recommended > 0 and r.rule_error_rate < r.pooled_only_error_rate
                       and r.rule_error_rate < r.system_equal_error_rate and r.rule_error_rate < r.random_error_rate)
        keep = ["n_decisions", "recommended", "coverage", "rule_errors", "rule_ties", "rule_error_rate", "flagged",
                "flagged_errors", "flagged_ties", "flagged_error_rate", "pooled_only_errors", "pooled_only_error_rate",
                "system_equal_errors", "system_equal_error_rate", "random_error_rate", "p_random_le_rule"]
        return dict(identification=ident, reduction=reduce_, **{k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in keep})
    g5 = {s: g5_of(s) for s in ("all",) + tuple(FAM)}
    g5["verdict"] = ("reduction" if g5["all"]["reduction"] else "identification_only" if g5["all"]["identification"]
                     else "neither")

    # ---- cross-release supplement: train RE1 (3 subsystems), hold out each RE2 subsystem; L1 adjacent
    xr = []
    _, acc1, n1 = acc_table(d, "RE1::", R.LAYERS["L1"])
    _, acc2, _ = acc_table(d, "RE2::", R.LAYERS["L1"])
    c1 = acc1.dropna()
    # methods on every RE1 and every RE2 subsystem (RCD was run on RE1 only, so it drops out here)
    meths = [m for m in c1.index if m in acc2.index and acc2.loc[m].notna().all()]
    c1 = c1.loc[meths]
    for h in acc2.columns:
        xr += decisions_for("RE1->RE2", "L1", c1, n1, acc2[h], h, "RE2", H, "adjacent")
    X = annotate(pd.DataFrame(xr))
    X["rule_recommends_c1"] = (X.dir_rule != 0) & (X.t >= C_FROZEN)
    X.to_csv(f"{OUT}/crossrelease.csv", index=False)
    xs = dict(n=len(X), recommended=int(X.rule_recommends_c1.sum()),
              recommended_errors=int(((X.out_rule == "error") & X.rule_recommends_c1).sum()),
              flagged_errors=int(((X.out_rule == "error") & ~X.rule_recommends_c1).sum()),
              pooled_only_errors_all=int((X.out_cw == "error").sum()), ties=int((X.out_rule == "tie").sum()))

    # ---- figure: L1 / L2, adjacent, all held-out, all families
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, layer in zip(axes, ("L1", "L2")):
        z = C[(C.layer == layer) & (C.set == "adjacent") & (C.held_filter == "all") & (C.scope == "all")
              & (C.threshold != "allwin")].copy()
        z = z[z.recommended > 0]
        ax.plot(z.coverage, z.rule_error_rate, "o-", color="C3", label="rule |m|/SD >= c")
        ax.plot(z.coverage, z.pooled_only_error_rate, "s--", color="C0", mfc="none", label="pooled-only (|case-weighted m|)")
        ax.plot(z.coverage, z.system_equal_error_rate, "^:", color="C2", mfc="none", label="system-equal |m|")
        ax.axhline(z.random_error_rate.iloc[0], color="grey", lw=1, label="random")
        f = z[z.threshold == C_FROZEN]
        if len(f):
            ax.annotate("c = 1", (f.coverage.iloc[0], f.rule_error_rate.iloc[0]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
        ax.set_title(f"{layer}: {int(z.n_decisions.iloc[0])} adjacent-pair decisions, 11 held-out subsystems")
        ax.set_xlabel("coverage (share of decisions recommended)")
        ax.set_xlim(0, 1.02)
    axes[0].set_ylabel("error rate among recommended")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/curve.png", dpi=150)

    cfg = dict(script=os.path.abspath(__file__), script_sha256=sha(os.path.abspath(__file__)), long_sha256=R.SHA,
               rq1_pair_subsystem_deltas_sha256=sha(f"{RQ1}/pair_subsystem_deltas.csv"),
               rq1_config=dict(B=cfg1.get("B"), seed=cfg1.get("seed"), script_sha256=cfg1.get("script_sha256")),
               thresholds=THRESH, c_frozen=C_FROZEN, n_min=NMIN, reference=REF, held_ci_checked=H.checked,
               spec=".research/plans/2026-09-29-wp6-rq3-spec.md", python=sys.version.split()[0], pandas=pd.__version__,
               run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_s=round(time.time() - t0, 1))
    json.dump(dict(config=cfg, g5=g5, crossrelease=xs,
                   decision_counts=D.groupby(["layer", "set", "family"]).size().rename("n").reset_index().to_dict("records")),
              open(f"{OUT}/g5.json", "w"), indent=1, default=float)
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)
    print(json.dumps(dict(g5=g5, crossrelease=xs), indent=1, default=float))
    print(D.groupby(["layer", "set", "family"]).size().to_string())


if __name__ == "__main__":
    main()
