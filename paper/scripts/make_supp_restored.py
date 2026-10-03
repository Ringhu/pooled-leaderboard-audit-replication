"""Supplementary tables for the held-out and cross-family analyses (restored 2026-09-30), from paper/data/holdout/ and
paper/data/rq1_cross6/ (outputs of scripts/analysis_main/rq3_holdout.py and rq1_crossfamily.py on the frozen long table).
Writes paper/supp/extra/{heldout,cross}.tex.
Usage (repo root): python3 paper/scripts/make_supp_restored.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "paper", "data")
OUT = os.path.join(ROOT, "paper", "supp", "extra")
NAME = {"baro": r"\baro", "max_z": r"\maxz", "alertcount": r"\alertcount", "cd1min": r"\cdmin", "rcd": r"\rcd",
        "e_diagnosis": r"\ediag"}
FAMN = {"all": "All three", "RE1": r"\rcaeval RE1", "openrca": r"\openrca", "petshop": r"\petshop"}
SUBN = {"RE1::OB": "RE1-OB", "RE1::SS": "RE1-SS", "RE1::TT": "RE1-TT", "openrca::bank": "Bank", "openrca::mkt1": "Market-1",
        "openrca::mkt2": "Market-2", "openrca::tel": "Telecom", "petshop::high_traffic": "High",
        "petshop::low_traffic": "Low", "petshop::temporal_traffic1": "Temporal-1", "petshop::temporal_traffic2": "Temporal-2"}


def f3(x):
    return f"{float(x):.3f}"


def sg(x):
    x = float(x)
    return f"$+${abs(x):.3f}" if x > 0 else (f"$-${abs(x):.3f}" if x < 0 else "0.000")


def heldout():
    cv = pd.read_csv(os.path.join(D, "holdout", "curve.csv"))
    cv = cv[cv.threshold != "allwin"].copy()
    cv["c"] = cv.threshold.astype(float)
    rows = []
    for layer, lname in (("L1", "Published"), ("L2", "All pairs")):
        for scope in ("all", "RE1", "openrca", "petshop"):
            for c in (0.0, 0.5, 1.0, 1.5, 2.0):
                q = cv[(cv.layer == layer) & (cv.set == "adjacent") & (cv.held_filter == "all") & (cv.scope == scope)
                       & (cv.c == c)]
                assert len(q) == 1
                r = q.iloc[0]
                rows.append(f"{lname if scope == 'all' and c == 0 else ''} & {FAMN[scope] if c == 0 else ''} & {c:.1f} & "
                            f"{int(r.n_decisions)} & {int(r.recommended)} & {int(r.rule_errors)} & {int(r.flagged)} & "
                            f"{int(r.flagged_errors)} & {int(r.system_equal_errors)} & {int(r.pooled_only_errors)} \\\\")
            rows.append(r"\addlinespace")
    tex = (r"""\begin{table}[h]
\centering
\caption{Held-out subsystems: adjacent pairs of the leaderboard built on the other subsystems of a family. A decision is passed when the margin on the remaining subsystems is at least $c$ times their spread ($c = 1$ is the line of the main text; $c = 0$ passes every decision with a direction). Wrong: the higher-ranked method is less accurate on the held-out subsystem. The last two columns pass the same number of decisions by the largest system-equal and case-weighted pooled margins.}\label{supp:heldout}
\scriptsize
\begin{tabular}{@{}llrrrrrrrr@{}}
\toprule
Layer & Family & $c$ & Decisions & Passed & Wrong & Flagged & Wrong & \shortstack{Margin only\\(equal) wrong} & \shortstack{Margin only\\(case-wtd.) wrong} \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return tex, len(rows)


def cross():
    lo = pd.read_csv(os.path.join(D, "rq1_cross6", "cross_loso.csv"))
    lo = lo[lo.rule == "mean"]
    lf = pd.read_csv(os.path.join(D, "rq1_cross6", "cross_lofo.csv"))
    lk = pd.read_csv(os.path.join(D, "rq1_cross6", "cross_lkso2.csv")).set_index(["a", "b"])
    pr = pd.read_csv(os.path.join(D, "rq1_cross6", "cross_pooling_rules.csv"))
    bt = pd.read_csv(os.path.join(D, "rq1_cross6", "cross_boot_reversal.csv")).set_index(["a", "b"])
    pairs = list(lo.groupby(["a", "b"]).groups)
    # (a) held-out selection per pair
    ra = []
    for a, b in pairs:
        g = lo[(lo.a == a) & (lo.b == b)]
        wrong = ", ".join(SUBN[s] for s in g[g.reversal].held_out)
        ra.append(f"{NAME[a]} vs {NAME[b]} & {int(g.reversal.sum())} of {len(g)} & {f3(g.regret.mean())} & "
                  f"{f3(g.regret.max())} & {wrong} \\\\")
    # (b) robustness per pair
    rb = []
    for a, b in pairs:
        g = lf[(lf.a == a) & (lf.b == b)]
        lofo = "; ".join(f"{FAMN[r.dropped_family]}: {int(r.n_pos)}/{int(r.n_neg)}" for r in g.itertuples())
        pp = pr[(pr.a == a) & (pr.b == b)].set_index("rule").pooled
        rb.append(f"{NAME[a]} vs {NAME[b]} & {lofo} & {int(lk.loc[(a, b), 'both_signs'])}/{int(lk.loc[(a, b), 'n_drops'])} & "
                  f"{sg(pp['mean'])} & {sg(pp['median'])} & {sg(pp['case_weighted'])} & {sg(pp['trimmed'])} & "
                  f"{bt.loc[(a, b), 'share_both_signs']:.3f} \\\\")
    tex = (r"""\begin{table}[h]
\centering
\caption{Cross-family set, held-out selection: for each of the 11 subsystems, the pooled winner of the other ten (subsystems weighted equally) is applied to it. Regret: accuracy lost on the held-out subsystem.}\label{supp:cross-loso}
\scriptsize
\begin{tabular}{@{}lrrrp{6.2cm}@{}}
\toprule
Pair & Reversals & Mean regret & Max regret & Held-out subsystems where the pooled winner loses \\
\midrule
""" + "\n".join(ra) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Cross-family set, robustness. Drop one family: subsystems where $A$ / $B$ is more accurate among the remaining ones. Drop two: cases (out of all 55 ways to drop two subsystems) in which both signs remain. Pooled difference under four pooling rules (trimmed: the largest and smallest subsystem differences removed). Bootstrap: share of 2{,}000 case resamples in which both signs remain.}\label{supp:cross-robust}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lp{4.6cm}rrrrrr@{}}
\toprule
Pair & Drop one family & Drop two & Mean & Median & Case-wtd. & Trimmed & Bootstrap \\
\midrule
""" + "\n".join(rb) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return tex, len(ra) + len(rb)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("heldout", heldout), ("cross", cross)):
        tex, n = fn()
        open(os.path.join(OUT, f"{name}.tex"), "w").write(tex)
        print(f"{name}.tex: {n} rows")


if __name__ == "__main__":
    main()
