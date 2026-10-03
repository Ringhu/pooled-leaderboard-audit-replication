"""Supplementary tables built from the diagnostic outputs in paper/data/supp/ (not from the long table).

Sources (each file is the verbatim stdout of a script on the audit server, see paper/data/supp/README.md):
  re1_window_compare.txt   -> extra/window.tex      (RE1 10- vs 20-minute window, BARO and the three probes)
  simplerca_summary.txt    -> extra/simplerca.tex   (authors' SimpleRCA branch: RE2/RE3 reproduction, RE1 results)
  re2_table6_compare.txt   -> extra/table6.tex      (RE2-TT per-fault values vs RCAEval Table 6)
  ../survey/release_check.csv -> extra/release.tex  (per-paper check of released per-case outputs, 2 Oct 2026)
Paper values for the SimpleRCA reproduction are the values in parentheses in
.research/verifications/2026-09-27-simplerca-author-g1.md section 2 (transcribed from the SimpleRCA paper, Table 2).
Usage (repo root): python3 paper/scripts/make_supp_extra.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "paper", "data", "supp")
OUT = os.path.join(ROOT, "paper", "supp", "extra")
G1 = os.path.join(D, "simplerca_g1_table.md")  # rows extracted from .research/verifications/2026-09-27-simplerca-author-g1.md
NAME = {"baro": r"\baro", "zbase": r"\maxz", "alertcount": r"\alertcount", "cd1min": r"\cdmin"}
SYS = {"ob": "OB", "ss": "SS", "tt": "TT"}


def f3(x):
    return f"{float(x):.3f}"


def window():
    rows = []
    for line in open(os.path.join(D, "re1_window_compare.txt")).read().splitlines()[1:]:
        p = line.split()
        if len(p) < 10:
            continue
        m, ds, a10, c10, _, _, a20, c20, lost, gained = p[:10]
        rows.append(f"{NAME[m]} & RE1-{SYS[ds.split('-')[1]]} & {f3(a10)} & {f3(a20)} & {lost} & {gained} & {c10}/{c20} \\\\")
    body = "\n".join(rows)
    return (r"""\begin{table}[h]
\caption{RE1 top-1 accuracy with a 10- and a 20-minute window (125 cases each). Lost / gained: cases correct at 10 minutes and wrong at 20, and the reverse. Source: \texttt{re1\_window\_compare.txt}.}
\label{supp:window}
\footnotesize
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Method & Subsystem & 10 min & 20 min & Lost & Gained & Valid (10/20) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""", len(rows))


def parse_summary():
    out = {}
    for line in open(os.path.join(D, "simplerca_summary.txt")):
        m = re.match(r"(\S+)\.json\s+n=\s*(\d+)\s+(.*)", line.strip())
        if not m:
            continue
        vals = dict(kv.split("=") for kv in m.group(3).split())
        vals["n"] = m.group(2)
        out[m.group(1)] = vals
    return out


def paper_values():
    """(dataset, method, branch) -> [paper T1, T3, T5, A3, A5, MRR] from the G1 verification table."""
    pv = {}
    for line in open(G1):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 11 and c[0].startswith("re") and "(" in c[4]:
            pv[(c[0], c[1], c[2])] = [re.search(r"\(([\d.]+)\)", x).group(1) for x in c[4:10]]
    return pv


def simplerca():
    s, pv = parse_summary(), paper_values()
    keys = ["T1", "T3", "T5", "A3", "A5", "MRRall"]
    rep = []
    for ds in ["re2_ob", "re2_ss", "re2_tt", "re3_ob", "re3_ss", "re3_tt"]:
        for meth in ["baro", "simplerca"]:
            if ds == "re3_ss":
                run = f"{meth}_author-re3ssbranch_re3_ss"
                branch = "re3ss"
            else:
                run = f"{meth}_author_{ds}"
                branch = "rcaeval"
            ours, paper = s[run], pv[(ds, meth, branch)]
            diff = max(abs(round(float(ours[k]), 2) - float(p)) for k, p in zip(keys, paper))
            cells = " & ".join(f"{float(ours[k]):.2f} ({p})" for k, p in zip(keys, paper))
            label = {"baro": r"\baro (authors')", "simplerca": r"\simplerca"}[meth]
            rep.append(f"{ds.upper().replace('_', '-')} & {label} & {cells} & {diff:.2f} \\\\")
    re1 = []
    for sysn in ["ob", "ss", "tt"]:
        for meth in ["baro", "simplerca"]:
            v = s[f"{meth}_author_re1_{sysn}_win20"]
            label = {"baro": r"\baro (authors')", "simplerca": r"\simplerca"}[meth]
            re1.append(f"RE1-{SYS[sysn]} & {label} & " + " & ".join(f3(v[k]) for k in keys) + r" & -- \\")
    tex = r"""\begin{table}[h]
\caption{Authors' \simplerca branch. Top: our runs on RE2 and RE3 with the values reported in the \simplerca paper in parentheses (Top@1, Top@3, Top@5, Avg@3, Avg@5, MRR with misses counted as 0); the last column is the largest absolute difference after rounding to two decimals. RE3-SS uses the second branch the authors provide. Bottom: the same code on RE1 (20-minute window; OB \texttt{data.csv}, SS/TT \texttt{simple\_data.csv}), scored by exact service-name match; RE1 contains only metrics, so no trace or log evidence is added. Source: \texttt{simplerca\_summary.txt}.}
\label{supp:simplerca}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
Data & Method & Top@1 & Top@3 & Top@5 & Avg@3 & Avg@5 & MRR & Max diff. \\
\midrule
""" + "\n".join(rep) + r"""
\midrule
""" + "\n".join(re1) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return tex, len(rep) + len(re1)


def table6():
    txt = open(os.path.join(D, "re2_table6_compare.txt")).read()
    blocks = re.findall(r"### (\S+)\.json\n\ncoverage (\d+)/(\d+), errors (\d+)\n\n\|[^\n]*\n\|[^\n]*\n((?:\|[^\n]*\n)+)", txt)
    label = {"circa_re2-tt_len20": r"\circa", "microrank_re2-tt": r"\microrank", "tracerca_re2-tt": r"\tracerca",
             "mmbaro_re2-tt_name-mm": r"\baro (multi-source), name \texttt{mm-tt}",
             "mmbaro_re2-tt_name-re2": r"\baro (multi-source), name \texttt{re2-tt}",
             "mmbaro_re2-tt_name-mm_metricscsv": r"\baro (multi-source), \texttt{metrics.csv}"}
    out, n = [], 0
    for run, cov, tot, err, rows in blocks:
        out.append(r"\midrule" + "\n" + rf"\multicolumn{{4}}{{@{{}}l}}{{{label[run]}: {cov}/{tot} valid outputs}} \\")
        for r in rows.strip().splitlines():
            c = [x.strip() for x in r.strip("|").split("|")]
            fault = c[0] if c[0] != "avg" else "all"
            out.append(f"{fault} & {c[1]} & {c[2]} & {c[3]} \\\\")
            n += 1
    return (r"""\begin{longtable}{@{}llll@{}}
\caption{RE2-TT per fault type: our runs over all cases (invalid outputs score 0), over valid outputs only, and the values in \rcaeval's Table 6. Cells are AC@1 / AC@3 / Avg@5. Source: \texttt{re2\_table6\_compare.txt}.}\label{supp:table6}\\
\toprule
Fault & All cases & Valid outputs only & Table 6 \\
\endfirsthead
\toprule
Fault & All cases & Valid outputs only & Table 6 \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(out).replace("(n=", "($n$=") + r"""
\end{longtable}
""", n)


def release():
    import csv
    excluded = {r["id"] for r in csv.DictReader(open(os.path.join(ROOT, "paper", "data", "survey", "excluded.csv")))}
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "paper", "data", "survey", "release_check.csv"))) if r["id"] not in excluded]
    assert len(rows) == 23, len(rows)
    ab = {"undeterminable": "undet."}
    out = []
    for r in rows:
        m = re.search(r"([\w.-]+/[\w.-]+)@([0-9a-f]{7})", r["repo_ref"])
        ref = f"{m.group(1)}@{m.group(2)}" if m else "--"
        ref = ref.replace("_", r"\_")
        out.append(f"{r['id']} & {ab.get(r['L1'], r['L1'])} & {ab.get(r['L3'], r['L3'])} & \\texttt{{\\footnotesize {ref}}} \\\\")
    return r"""\begin{longtable}{lllp{7.6cm}}
\caption{Released per-case outputs in the 23 surveyed papers with full text, checked on 2 October 2026 (UTC). Any: the release contains per-case predictions or scores of any method. All: it contains them for every method of the paper's main comparison. undet.: the linked release had expired or could not be listed. Release: first repository checked, at the commit checked (--: no repository linked, or the link is not a listable repository); the notes in the package give every link and the files found.}\label{supp:release}\\
\toprule
Paper & Any & All & Release \\
\midrule
\endhead
""" + "\n".join(out) + r"""
\bottomrule
\end{longtable}
""", len(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("window", window), ("simplerca", simplerca), ("table6", table6), ("release", release)]:
        tex, n = fn()
        open(os.path.join(OUT, f"{name}.tex"), "w").write(tex)
        print(f"{name}.tex: {n} rows")


if __name__ == "__main__":
    main()
