"""Check every number in the paper text (paper/sections/*.tex).

1. Computed results appear in the text only as macros from paper/numbers.tex; this script regenerates numbers.tex
   from paper/data/ (make_numbers.py --check) and fails if it is stale.
2. Every macro used in the text is defined, and every number macro is used at least once (unused ones are listed).
3. Every remaining numeric literal in the text must be in LITERALS below, with the source it was checked against.
   Unlisted literals are reported and make the check fail.
Usage (repo root): python3 scripts/verify_main_text_numbers.py
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "paper")

# The abstract is also uploaded as plain text, so it carries literal numbers; each must equal the current value of
# one of these macros (checked below against numbers.tex).
ABSTRACT_MACROS = ["NPubAbove", "NPubAboveHold", "NPubBelow", "NPubBelowRev", "HErr", "HNDec", "NCrossPub",
                   "NCrossPubChange", "NCrossPubChangeBaro", "SurveyN", "SurveyRelFullCheckable"]

# literal -> source (dataset facts, published values, code facts); checked by hand against the named source
LITERALS = {
    "0,1": "interval [0,1] of the case score (design, GLM response)",
    "310": "subsystem differences compared between the two RQ1/RQ2 implementations (.research/verifications/2026-09-29-rq1-rq2-crosscheck.md)",
    "4.7": "Grok 4.7 model version (LLM-use disclosure)", "5.1": "Claude Fable 5.1 model version (LLM-use disclosure)",
    "5.5": "Claude Opus 5.5 model version (LLM-use disclosure)",
    "1": "counts / ordinal", "2": "counts / ordinal", "3": "counts (three families, seeds, top three)", "4": "k=4 / four items",
    "5": "five fault types", "6": "six fault types / strata", "8": "PetShop temporal cases (coverage.csv n)",
    "10": "10 published methods / MicroCause >10 min", "11": "k = 11 subsystems", "12": "SimpleRCA rows reproduced (verifications/2026-09-27-simplerca-author-g1.md)",
    "16": "16 methods in main analysis (long table roles published/probe/shipped_baseline)",
    "17": "TraceRCA/MicroRank failed cases (verifications/2026-09-27-re2-rcaeval-table6.md)",
    "18": "cells per method in RCAEval Table 6 (6 faults x 3 metrics)", "20": "RCAEval window minutes", "23": "full-text papers (survey/papers.csv)",
    "25": "n >= 25 cases threshold", "26": "surveyed papers / PetShop High,Low cases", "27": "survey disagreements (final_summary.json)",
    "29": "date of upstream fix (29 September 2026)", "30": "OpenRCA window +-30 min / permutation cases",
    "43": "CausalRCA seed-1 empty-graph cases (verifications/2026-09-29-causalrca-full-run.md)",
    "49": "CIRCA column order / eps-Diag time.1 top-1 (verifications/2026-09-30-re1ob-duplicate-time-column.md)",
    "50": "RE1-OB duplicated-header cases", "51": "Telecom queries", "60": "OpenRCA time tolerance (s)", "70": "Market-1 queries",
    "75": "RE1-OB unaffected cases", "78": "Market-2 queries", "90": "RE2 cases per application", "125": "RE1 cases per application",
    "136": "Bank queries", "270": "RE2 cases", "778": "main-analysis cases per method", "2021": "survey years", "2026": "survey years / fix date",
    "0.32": "scikit-network version that removed PageRank.fit_transform (RCAEval 21ff8c9 message)",
    "0.33.0": "scikit-network in the ts env", "0.31.0": "scikit-network in rcaeval_rcd env",
    "0.960": "BARO Avg@5 RE1-SS simple_data (runs/baro, recomputed 2026-09-30)", "0.779": "BARO Avg@5 RE1-TT simple_data",
    "0.576": "BARO Avg@5 RE1-SS data.csv", "0.261": "BARO Avg@5 RE1-TT data.csv",
    "0.95": "BARO paper Avg@5 SS (FSE'24 §4.6.1)", "0.81": "BARO paper Avg@5 TT", "0.14": "Fang et al. Table 2, BARO AC@1 on RE2-OB and RE2-SS (.research/verifications/2026-09-27-simplerca-author-g1.md)", "0.66": "RCAEval Table 6 TraceRCA RE2-TT AC@1",
    "0.22": "RCAEval Table 6 CausalRCA RE2-TT AC@1", "0.01": "tolerance for Table 6 match", "0.5": "OpenRCA >=0.5 rule",
    "44.8": "TraceRCA abstract quote", "4,900": "CausalRCA OpenRCA GPU-hour estimate (verifications/2026-09-29-openrca-extra-methods-cost.md)",
    "10,000": "bootstrap B (rq1/config.json)", "20260929": "bootstrap seed (rq1/config.json)",
    "421": "SS data.csv columns min", "439": "SS data.csv columns max", "1,180": "TT data.csv columns min", "1,446": "TT data.csv columns max",
    "58": "SS simple_data columns min", "64": "SS simple_data columns max", "199": "TT simple_data columns min", "239": "TT simple_data columns max",
    "206": "main.py line (259ea41)", "214": "main.py line (259ea41)", "497505": "commit hash fragment", "259": "commit hash fragment",
    "21": "commit hash fragment 21ff8c9", "0": "zero", "95": "95% CI", "256": "SHA-256",
}


def strip(t):
    t = re.sub(r"(?<!\\)%.*", "", t)  # comments
    t = t.replace("{,}", ",")
    t = re.sub(r"\\(cite[pt]?|ref|cref|Cref|label|input|includegraphics|citet|citep)(\[[^\]]*\])?\{[^}]*\}", "", t)
    t = re.sub(r"\\texttt\{[0-9a-f]{7}\}", "", t)  # commit hashes
    t = re.sub(r"\\begin\{tabular\*?\}(\{[^}]*\})?(\{[^}]*\})?", "", t)
    t = re.sub(r"p\{[0-9.]+cm\}|\\tabcolsep\}\{[0-9.]+pt\}|[0-9.]+(pt|cm|in)\b", "", t)
    return t


def main():
    ok = True
    r = subprocess.run([sys.executable, os.path.join(P, "scripts", "make_numbers.py"), "--check"], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    ok &= r.returncode == 0
    numbers_src = open(os.path.join(P, "numbers.tex")).read()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers_src))
    values = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}\\]*)", numbers_src))
    abstract_ok = {values[m] for m in ABSTRACT_MACROS}
    used, unknown = set(), {}
    for f in sorted(glob.glob(os.path.join(P, "sections", "*.tex"))):
        t = open(f).read()
        for m in re.findall(r"\\([A-Za-z]+)", t):
            if m in defined:
                used.add(m)
        body = strip(t)
        body = re.sub(r"\\[A-Za-z]+", " ", body)  # drop macro names
        for lit in re.findall(r"(?<![A-Za-z0-9.])\d[\d,]*(?:\.\d+)*", body):
            lit = lit.rstrip(",")
            if os.path.basename(f) == "0_abstract.tex" and lit in abstract_ok:
                continue
            if lit not in LITERALS:
                unknown.setdefault(lit, set()).add(os.path.basename(f))
    unused = sorted(defined - used)
    print(f"number macros: {len(defined)} defined, {len(used)} used in text, {len(unused)} unused")
    if unknown:
        ok = False
        print("UNLISTED numeric literals (add a source to LITERALS or replace by a macro):")
        for k, v in sorted(unknown.items()):
            print(f"  {k}: {sorted(v)}")
    print("ALL CHECKS PASS" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
