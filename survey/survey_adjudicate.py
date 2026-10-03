"""Reporting-practice survey: adjudicated coding (main thread, 2026-09-29) and summary counts.

Agreements between coder A and coder B are kept. Each disagreement is decided below against codebook.md, with the
reason. Values not chosen by either coder are allowed where the codebook requires them (marked "third value").
Outputs .research/survey/final_coding.csv (id, item, value, source, reason, evidence_quote, location) and
final_summary.json (counts per item; Q3 also without the three abstract-only papers).
"""
import collections
import csv
import json
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".research", "survey")
ABSTRACT_ONLY = {"microrank2021", "mrca2024", "nezha2023"}  # no full text; abstracts checked against Semantic Scholar

# (id, item): (value, coder whose evidence is kept, reason)
ADJ = {
    ("circa2022", "Q1"): ("yes", "A", "simulation study and the real dataset D_O are separate evaluation units, each in its own table"),
    ("petshop2024", "Q1"): ("yes", "A", "Tables 3/8 report every traffic scenario x metric separately (the units this project treats as subsystems); Q1 has no benchmark-paper exemption"),
    ("rcaeval2025", "Q1"): ("not_applicable", "B", "third value: the only results (Table 6, 'preliminary experiments') are on one system, RE2 Train Ticket"),
    ("sparserca2024", "Q2"): ("no", "A", "no code/data release; both coders coded every other paper without released material as 'no', kept consistent"),
    ("tracediag2023", "Q2"): ("no", "A", "same as sparserca2024"),
    ("tvdiag2024", "Q2"): ("partial", "B", "'The source code and experimental data are publicly available [59]' -- code open, per-case outputs not stated (codebook Q2 rule)"),
    ("baro2024", "Q3"): ("yes", "A", "abstract sentence: unqualified 'consistently outperforms', no per-system number in the abstract; one qualifying sentence suffices"),
    ("chase2024", "Q3"): ("partial", "B", "the abstract's next sentence gives per-dataset gains ('respectively' GAIA / AIOps 2020)"),
    ("dynacausal2025", "Q3"): ("yes", "B", "abstract: 'average AC@1 of 0.63' = mean of D1 and D2, no per-system number in the abstract"),
    ("eadro2023", "Q3"): ("yes", "A", "'F1 of 0.988 and HR@1 of 0.982 on average, vastly outperforming all competitors' is in the Abstract (coder A wrote Conclusion); no per-system number there"),
    ("fang2026", "Q3"): ("partial", "B", "claim is hedged ('comparable to or even surpassing'); results paragraph cites per-dataset numbers"),
    ("mabc2024", "Q3"): ("yes", "A", "assertion cites the Average column (64.9); codebook: same-table per-system numbers not cited by the assertion still count as yes"),
    ("microrank2021", "Q3"): ("yes", "B", "abstract only (verified verbatim, Semantic Scholar): '6% to 22% improvement in recall' across the two systems"),
    ("mrca2024", "Q3"): ("yes", "B", "abstract only (verified verbatim): 'Experiments on two widely-used microservice benchmarks demonstrate that MRCA outperforms state-of-the-art approaches'"),
    ("nezha2023", "Q3"): ("yes", "B", "abstract only (verified verbatim): '89.77% on average ... outperforms state-of-the-art approaches by a large margin'"),
    ("mulan2024", "Q3"): ("partial", "B", "the pooled sentence is followed in the same paragraph by per-dataset evidence"),
    ("ocean2024", "Q3"): ("partial", "B", "same pattern as mulan2024"),
    ("openrca2025", "Q3"): ("not_applicable", "A", "benchmark paper; the quoted sentence compares model families, not the authors' method (codebook Q3 not_applicable)"),
    ("tracerca2021", "Q3"): ("yes", "A", "abstract: '44.8%' is the pooled figure, no per-system number in the abstract"),
    ("tvdiag2024", "Q3"): ("partial", "B", "'at least 32.46%' is a minimum over datasets, not a pooled number; results paragraph gives per-dataset evidence and p-values"),
    ("dynacausal2025", "Q4"): ("partial", "A", "says data are 'preprocessed' and taken from RCAEval RE2, no file / column / version"),
    ("mabc2024", "Q4"): ("partial", "A", "data construction described (processes, users, ChaosBlade), no file / column / version; same rule as tracecontrast2024, tracerca2021"),
    ("ocean2024", "Q4"): ("partial", "B", "window lengths given (8-hour history, 10-minute batches), metric set / version not given"),
    ("openrca2025", "Q4"): ("yes", "B", "named sampling strategies, one-minute sampling, removed columns listed (Sec 3.3)"),
    ("rcd2022", "Q4"): ("partial", "A", "collection setup and '150 metrics' only; codebook: a column count without how it was selected is partial"),
    ("tracecontrast2024", "Q4"): ("partial", "A", "dataset construction and size only"),
    ("tracerca2021", "Q4"): ("partial", "A", "dataset construction, size and train/test split only"),
}


def main():
    A = {(r["id"], r["item"]): r for r in csv.DictReader(open(f"{D}/coding_A.csv"))}
    B = {(r["id"], r["item"]): r for r in csv.DictReader(open(f"{D}/coding_B.csv"))}
    rows = []
    for k in sorted(A):
        a, b = A[k], B[k]
        if a["value"].strip() == b["value"].strip():
            assert k not in ADJ, k
            src = a if a["evidence_quote"].strip() else b
            rows.append(dict(id=k[0], item=k[1], value=a["value"].strip(), source="agree", reason="",
                             evidence_quote=src["evidence_quote"], location=src["location"]))
        else:
            v, who, why = ADJ[k]
            src = a if who == "A" else b
            rows.append(dict(id=k[0], item=k[1], value=v, source=f"adjudicated ({who})", reason=why,
                             evidence_quote=src["evidence_quote"], location=src["location"]))
    assert len([r for r in rows if r["source"] != "agree"]) == len(ADJ)
    with open(f"{D}/final_coding.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {it: dict(collections.Counter(r["value"] for r in rows if r["item"] == it)) for it in ("Q1", "Q2", "Q3", "Q4")}
    summ["Q3_full_text_only"] = dict(collections.Counter(r["value"] for r in rows if r["item"] == "Q3" and r["id"] not in ABSTRACT_ONLY))
    summ["n_papers"] = len({r["id"] for r in rows})
    summ["n_adjudicated"] = len(ADJ)
    json.dump(summ, open(f"{D}/final_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
