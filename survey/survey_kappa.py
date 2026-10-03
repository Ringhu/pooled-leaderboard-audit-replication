"""Reporting-practice survey: agreement between the two AI coders and verbatim check of evidence quotes.

Inputs  .research/survey/coding_A.csv, coding_B.csv (one row per paper x item; codebook.md), texts/<id>.txt
Outputs .research/survey/agreement.json (Cohen's kappa per item, unweighted, 4 nominal values; percent agreement;
        value confusion), disagreements.csv (paper x item where A != B, with both quotes), quote_check.csv
        (every non-empty quote; fragments joined by an ellipsis are checked one by one, each must occur verbatim in
        texts/<id>.txt after whitespace / hyphenation / typographic-quote normalisation).
"""
import collections
import csv
import json
import os
import re

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".research", "survey")
ITEMS = ["Q1", "Q2", "Q3", "Q4"]


def load(name):
    return {(r["id"], r["item"]): r for r in csv.DictReader(open(f"{D}/{name}"))}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return po, pe, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def norm(s):
    s = s.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = re.sub(r"-\s*\n\s*", "", s)  # hyphenated line break
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    A, B = load("coding_A.csv"), load("coding_B.csv")
    assert A.keys() == B.keys()
    out = {}
    for it in ITEMS:
        keys = sorted(k for k in A if k[1] == it)
        va, vb = [A[k]["value"].strip() for k in keys], [B[k]["value"].strip() for k in keys]
        po, pe, k = kappa(va, vb)
        out[it] = dict(n=len(keys), percent_agreement=po, chance_agreement=pe, cohen_kappa=k,
                       values_A=dict(collections.Counter(va)), values_B=dict(collections.Counter(vb)),
                       confusion={f"{x}|{y}": c for (x, y), c in collections.Counter(zip(va, vb)).items()})
    json.dump(out, open(f"{D}/agreement.json", "w"), indent=1)
    with open(f"{D}/disagreements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "item", "value_A", "value_B", "quote_A", "loc_A", "note_A", "quote_B", "loc_B", "note_B"])
        for k in sorted(A):
            a, b = A[k], B[k]
            if a["value"].strip() != b["value"].strip():
                w.writerow([k[0], k[1], a["value"], b["value"], a["evidence_quote"], a["location"], a["note"],
                            b["evidence_quote"], b["location"], b["note"]])
    texts = {}
    with open(f"{D}/quote_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coder", "id", "item", "value", "has_text", "found", "quote", "missing_fragments"])
        for coder, X in (("A", A), ("B", B)):
            for k, r in sorted(X.items()):
                q = r["evidence_quote"].strip()
                if not q:
                    continue
                p = f"{D}/texts/{k[0]}.txt"
                if k[0] not in texts:
                    texts[k[0]] = norm(open(p, errors="ignore").read()) if os.path.exists(p) else None
                t = texts[k[0]]
                # quotes may join fragments with "..." / "…"; every fragment must occur verbatim
                frags = [norm(x) for x in re.split(r"\.\.\.|…", q.strip('"')) if len(x.split()) >= 3]
                miss = [x for x in frags if t is not None and x not in t]
                found = None if t is None else not miss
                w.writerow([coder, k[0], k[1], r["value"], t is not None, found, q, " || ".join(miss)])
    for it in ITEMS:
        print(it, {x: round(v, 3) if isinstance(v, float) else v for x, v in out[it].items() if x != "confusion"})


if __name__ == "__main__":
    main()
