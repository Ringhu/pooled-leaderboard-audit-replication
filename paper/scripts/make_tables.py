#!/usr/bin/env python3
"""Build booktabs tables for the RCA benchmark audit paper.

Every number is computed from paper/data/. See SPEC.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data"
OUT = ROOT / "paper" / "tables"

METHOD_DISPLAY = {
    "baro": "BARO",
    "circa": "CIRCA",
    "e_diagnosis": r"$\epsilon$-Diagnosis",
    "epsilon_diagnosis": r"$\epsilon$-Diagnosis",
    "rcd": "RCD",
    "causalrca": "CausalRCA",
    "counterfactual_attribution": "Counterfactual",
    "simplerca_author": "SimpleRCA",
    "tracerca": "TraceRCA",
    "microrank": "MicroRank",
    "mmbaro": "BARO (multi-source)",
    "max_z": r"max-$|Z|$",
    "alertcount": "Alert-count",
    "cd1min": "CD-1min",
    "baseline": "Traversal",
    "ranked_correlation": "Ranked correlation",
    "random_walk": "Random walk",
}

SUB_DISPLAY = {
    "RE1::OB": "RE1-OB",
    "RE1::SS": "RE1-SS",
    "RE1::TT": "RE1-TT",
    "RE2::OB": "RE2-OB",
    "RE2::SS": "RE2-SS",
    "RE2::TT": "RE2-TT",
    "openrca::bank": "Bank",
    "openrca::mkt1": "Market-1",
    "openrca::mkt2": "Market-2",
    "openrca::tel": "Telecom",
    "petshop::high_traffic": "High",
    "petshop::low_traffic": "Low",
    "petshop::temporal_traffic1": "Temporal-1",
    "petshop::temporal_traffic2": "Temporal-2",
}

FAMILY_DISPLAY = {
    "RE1": "RCAEval RE1",
    "RE2": "RCAEval RE2",
    "openrca": "OpenRCA",
    "petshop": "PetShop",
}

PUBLISHED = [
    "baro",
    "circa",
    "e_diagnosis",
    "rcd",
    "causalrca",
    "counterfactual_attribution",
    "simplerca_author",
    "tracerca",
    "microrank",
    "mmbaro",
]
PROBES = ["max_z", "alertcount", "cd1min"]
BASELINES = ["baseline", "ranked_correlation", "random_walk"]

# Column groups for Table 4. Header abbreviations are the spec's short names.
TAB4_GROUPS = [
    (
        "RCAEval RE1",
        [("RE1::OB", "OB"), ("RE1::SS", "SS"), ("RE1::TT", "TT")],
    ),
    (
        "RCAEval RE2",
        [("RE2::OB", "OB"), ("RE2::SS", "SS"), ("RE2::TT", "TT")],
    ),
    (
        "OpenRCA",
        [
            ("openrca::bank", "Bank"),
            ("openrca::mkt1", "Mkt-1"),
            ("openrca::mkt2", "Mkt-2"),
            ("openrca::tel", "Tel"),
        ],
    ),
    (
        "PetShop",
        [
            ("petshop::high_traffic", "High"),
            ("petshop::low_traffic", "Low"),
            ("petshop::temporal_traffic1", "Tmp-1"),
            ("petshop::temporal_traffic2", "Tmp-2"),
        ],
    ),
]


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def fmt_signed(x: float) -> str:
    """Three decimals with an explicit sign; minus is LaTeX $-$."""
    if x < 0:
        return r"$-$" + f"{abs(x):.3f}"
    return f"+{x:.3f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{fmt_signed(lo)}, {fmt_signed(hi)}]"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def display_method(raw: str) -> str:
    if raw not in METHOD_DISPLAY:
        raise KeyError(f"no display name for method {raw!r}")
    return METHOD_DISPLAY[raw]


def display_family(raw: str) -> str:
    if raw not in FAMILY_DISPLAY:
        raise KeyError(f"no display name for family {raw!r}")
    return FAMILY_DISPLAY[raw]


def display_sub(raw: str) -> str:
    if raw not in SUB_DISPLAY:
        raise KeyError(f"no display name for subsystem {raw!r}")
    return SUB_DISPLAY[raw]


def write(name: str, body: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(body)
    return path


def coverage_by_method_sub() -> pd.DataFrame:
    """Mean coverage over seeds when a method is seeded; otherwise the single row."""
    cov = pd.read_csv(DATA / "rq2" / "coverage.csv")
    # A method is seeded on a subsystem when at least one row has a seed.
    # Average only those seeded rows (an unseeded companion row is not present
    # alongside seeds in this file; petshop RCD is unseeded and kept as-is).
    rows = []
    for (sub, method), g in cov.groupby(["subsystem_id", "method"], sort=False):
        seeded = g[g["seed"].notna()]
        use = seeded if len(seeded) else g
        rows.append(
            {
                "subsystem_id": sub,
                "method": method,
                "coverage": float(use["coverage"].mean()),
                "n_rows_averaged": int(len(use)),
            }
        )
    return pd.DataFrame(rows)


def case_counts() -> dict[str, int]:
    cov = pd.read_csv(DATA / "rq2" / "coverage.csv")
    nunique = cov.groupby("subsystem_id")["n"].nunique()
    if (nunique != 1).any():
        bad = nunique[nunique != 1]
        raise ValueError(f"case count n is not unique per subsystem: {bad.to_dict()}")
    return {k: int(v) for k, v in cov.groupby("subsystem_id")["n"].first().items()}


def merged_accuracy(acc: pd.DataFrame) -> pd.DataFrame:
    """One row per display method. e_diagnosis and epsilon_diagnosis never overlap."""
    acc = acc.set_index("method")
    subs = [c for c in acc.columns]
    e_name = "e_diagnosis"
    ep_name = "epsilon_diagnosis"
    if e_name in acc.index and ep_name in acc.index:
        both = acc.loc[e_name].notna() & acc.loc[ep_name].notna()
        if bool(both.any()):
            overlap = list(both[both].index)
            raise ValueError(f"e_diagnosis and epsilon_diagnosis overlap on {overlap}")
        merged = acc.loc[e_name].where(acc.loc[e_name].notna(), acc.loc[ep_name])
        acc = acc.drop(index=[e_name, ep_name])
        acc.loc["e_diagnosis"] = merged
    elif ep_name in acc.index and e_name not in acc.index:
        acc = acc.rename(index={ep_name: e_name})
    # Stable row order within the three groups, dropping methods absent from the file.
    order = [m for m in PUBLISHED + PROBES + BASELINES if m in acc.index]
    extra = [m for m in acc.index if m not in order]
    if extra:
        raise ValueError(f"methods not in the published/probe/baseline lists: {extra}")
    return acc.loc[order, subs]


def _tab4_panel(acc, cov_map, counts, best, groups_cols, colspec_prefix=""):
    """One panel of the main table: header for the given column groups, rows for methods with any value in them."""
    subs = [sub for _, cols in groups_cols for sub, _ in cols]
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "r" * len(subs) + "@{}}", r"\toprule"]
    group_cells = [""]
    cmid = []
    col = 2
    for title, cols in groups_cols:
        group_cells.append(r"\multicolumn{%d}{c}{%s}" % (len(cols), title))
        cmid.append(r"\cmidrule(lr){%d-%d}" % (col, col + len(cols) - 1))
        col += len(cols)
    lines.append(" & ".join(group_cells) + r" \\")
    lines.append(" ".join(cmid))
    lines.append(" & ".join(["$n$"] + [str(counts[sub]) for sub in subs]) + r" \\")
    lines.append(" & ".join(["Method"] + [short for _, cols in groups_cols for _, short in cols]) + r" \\")
    lines.append(r"\midrule")
    stats = dict(rows=0, dagger=0, bold=0, missing=0)
    first_group = True
    for label, methods in [("Published methods", PUBLISHED), ("Probes", PROBES), ("Shipped baselines", BASELINES)]:
        present = [m for m in methods if m in acc.index and acc.loc[m, subs].notna().any()]
        if not present:
            continue
        if not first_group:
            lines.append(r"\midrule")
        first_group = False
        lines.append(r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (1 + len(subs), label))
        for method in present:
            cells = [display_method(method)]
            for sub in subs:
                val = acc.at[method, sub]
                if pd.isna(val):
                    cells.append("--")
                    stats["missing"] += 1
                    continue
                cov_key = (sub, method)
                if cov_key not in cov_map:
                    raise KeyError(f"no coverage for {method} @ {sub}")
                text = fmt3(float(val))
                if cov_map[cov_key] < 1.0:
                    text += r"$^\dagger$"
                    stats["dagger"] += 1
                if best[sub] is not None and np.isclose(float(val), best[sub]):
                    text = r"{\bfseries " + text + "}"
                    stats["bold"] += 1
                cells.append(text)
            lines.append(" & ".join(cells) + r" \\")
            stats["rows"] += 1
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return lines, stats



VENUE_FIX = {
    "petshop2024": "CLeaR 2024", "tvdiag2024": "TOSEM", "chase2024": "arXiv 2024", "lemmarca2024": "arXiv 2024",
    "dynacausal2025": "arXiv 2025", "baro2024": "FSE 2024", "fang2026": "FSE 2026", "rcaeval2025": "WWW 2025 Companion",
    "causalrca2023": "JSS 2023", "mabc2024": "EMNLP 2024 Findings",
}

# survey id -> (short name used in the table, key in references.bib); every surveyed paper is cited (2026-10-03)
SURVEY_CITE = {
    "baro2024": ("BARO", "pham2024baro"), "circa2022": ("CIRCA", "li2022circa"), "rcd2022": ("RCD", "ikram2022rcd"),
    "tracerca2021": ("TraceRCA", "li2021tracerca"), "rcaeval2025": ("RCAEval", "pham2024rcaeval"),
    "fang2026": ("SimpleRCA", "fang2025simplerca"), "openrca2025": ("OpenRCA", "xu2025openrca"),
    "petshop2024": ("PetShop", "hardt2024petshop"), "eadro2023": ("Eadro", "lee2023eadro"), "mabc2024": ("mABC", "zhang2024mabc"),
    "mulan2024": ("MULAN", "zheng2024mulan"), "ocean2024": ("OCEAN", "zheng2025ocean"), "tvdiag2024": ("TVDiag", "xie2025tvdiag"),
    "chase2024": ("CHASE", "zhao2024chase"), "howfar2024": ("How Far Are We", "pham2024howfar"),
    "lemmarca2024": ("LEMMA-RCA", "zheng2024lemma"), "sparserca2024": ("SparseRCA", "yao2024sparserca"),
    "tracecontrast2024": ("TraceContrast", "zhang2024tracecontrast"), "tracediag2023": ("TraceDiag", "ding2023tracediag"),
    "run2024": ("RUN", "lin2024run"), "dynacausal2025": ("DynaCausal", "zhang2025dynacausal"),
    "causalrca2023": ("CausalRCA", "causalrca"), "toomanycooks2025": ("Too Many Cooks", "zhang2025toomanycooks"),
}


def make_tab1() -> None:
    """Table 1: the 23 surveyed papers (main text), one row per paper. Source: data/survey/{papers,per_system,release_check,excluded}.csv."""
    papers = pd.read_csv(DATA / "survey" / "papers.csv")
    per_system = pd.read_csv(DATA / "survey" / "per_system.csv").set_index("id")
    rc = pd.read_csv(DATA / "survey" / "release_check.csv").set_index("id")
    excluded = set(pd.read_csv(DATA / "survey" / "excluded.csv").id)
    kinds = {"demo_case": "one demonstration case", "own_method": "own method, all test cases", "all_methods_one_table": "all methods, simulation study only"}
    rows = []
    for r in papers.itertuples(index=False):
        pid = str(r.id)
        if pid in excluded:
            continue
        short, bibkey = SURVEY_CITE[pid]
        author = f"{short}~\\citep{{{bibkey}}}"
        venue = VENUE_FIX.get(pid) or str(r.venue).split("（")[0].split(" (")[0].strip()
        ps = {"yes": "yes", "single_system": "single system"}[str(per_system.loc[pid, "per_system"])]
        l1, kind, l3 = str(rc.loc[pid, "L1"]), str(rc.loc[pid, "kind"]) if pd.notna(rc.loc[pid, "kind"]) else "", str(rc.loc[pid, "L3"])
        if l1 == "undeterminable":
            rel, full = "release unreachable", "--"
        elif l1 == "yes":
            rel, full = kinds[kind], {"no": "no", "partial": "partial"}[l3]
        else:
            rel, full = "--", "no"
        rows.append(f"{author} & {int(r.year)} & {tex_escape(venue)} & {ps} & {rel} & {full} \\\\")
    if len(rows) != 23 or len(SURVEY_CITE) != 23:
        raise ValueError(f"expected 23 papers, got {len(rows)} rows and {len(SURVEY_CITE)} citation keys")
    bib = (ROOT / "paper" / "references.bib").read_text()
    missing = [k for _, k in SURVEY_CITE.values() if f"{{{k}," not in bib]
    if missing:
        raise ValueError(f"bib keys missing from references.bib: {missing}")
    body = "\n".join([
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lllllc@{}}",
        r"\toprule",
        r"Paper & Year & Venue & Per-system & Per-case outputs released & All methods \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"
    write("tab1_survey.tex", body)

def make_tab4() -> None:
    """Main table as two full-width panels (2026-10-03): RCAEval RE1+RE2, then OpenRCA+PetShop.
    Each panel lists only the methods that run on at least one of its families."""
    acc = pd.read_csv(DATA / "rq1" / "accuracy_by_subsystem.csv")
    acc = merged_accuracy(acc)
    cov = coverage_by_method_sub()
    cov_map = {(r.subsystem_id, r.method): r.coverage for r in cov.itertuples(index=False)}
    for (sub, method), c in list(cov_map.items()):
        if method == "epsilon_diagnosis":
            cov_map[(sub, "e_diagnosis")] = c
    counts = case_counts()

    subs = [sub for _, cols in TAB4_GROUPS for sub, _ in cols]
    missing = [s for s in subs if s not in acc.columns]
    if missing:
        raise ValueError(f"accuracy table missing subsystems {missing}")
    best = {}
    for sub in subs:
        vals = acc[sub].dropna()
        best[sub] = float(vals.max()) if len(vals) else None

    groups = dict(TAB4_GROUPS)
    panel_a, st_a = _tab4_panel(acc, cov_map, counts, best, [("RCAEval RE1", groups["RCAEval RE1"]), ("RCAEval RE2", groups["RCAEval RE2"])])
    panel_b, st_b = _tab4_panel(acc, cov_map, counts, best, [("OpenRCA", groups["OpenRCA"]), ("PetShop", groups["PetShop"])])
    # Two full-width panels (2026-10-03): RCAEval RE1+RE2, then OpenRCA+PetShop,
    # both with the same method column; '--' marks a method that does not run there.
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\centering"]
    lines += panel_a
    lines += [r"\par\medskip"]
    lines += panel_b
    lines += [""]
    path = write("tab4_main.tex", "\n".join(lines))
    print(f"tab4_main.tex -> {path}")
    for name, st in [("RCAEval", st_a), ("OpenRCA+PetShop", st_b)]:
        print(f"  panel {name}: rows={st['rows']} dagger={st['dagger']} bold={st['bold']} missing={st['missing']}")
    print("  case counts: " + ", ".join(f"{s}={counts[s]}" for s in subs))


def make_tab5() -> None:
    """Cross-release table, compact form (2026-10-02): one row per published pair, RE1 and RE2 difference per application;
    bold where the two releases' CIs lie on opposite sides of zero. CIs themselves are in the text / package."""
    cr = pd.read_csv(DATA / "rq1" / "cross_release.csv")
    cr = cr[cr["layer"] == "L1"].copy()
    apps = ["OB", "SS", "TT"]
    if not set(cr["app"]).issubset(apps):
        raise ValueError(f"unexpected apps: {sorted(set(cr['app']) - set(apps))}")
    pair_order = []
    for a_, b_ in zip(cr["a"], cr["b"]):
        if (a_, b_) not in pair_order:
            pair_order.append((a_, b_))
    lines = [
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Online Boutique} & \multicolumn{2}{c}{Sock Shop} & \multicolumn{2}{c}{Train Ticket} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Pair $A$ vs $B$ & RE1 & RE2 & RE1 & RE2 & RE1 & RE2 \\",
        r"\midrule",
    ]
    n_same = n_opp = 0
    for a_, b_ in pair_order:
        cells = [f"{display_method(a_)} vs {display_method(b_)}"]
        for app in apps:
            r = cr[(cr["a"] == a_) & (cr["b"] == b_) & (cr["app"] == app)]
            if len(r) != 1:
                raise ValueError(f"{a_} vs {b_} on {app}: {len(r)} rows")
            r = r.iloc[0]
            n_same += int(bool(r["sign_agree"]))
            n_opp += int(bool(r["ci_opposite"]))
            d1, d2 = fmt_signed(r["delta_re1"]), fmt_signed(r["delta_re2"])
            if bool(r["ci_opposite"]):
                d1, d2 = r"{\bfseries " + d1 + "}", r"{\bfseries " + d2 + "}"
            cells += [d1, d2]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = write("tab5_cross_release.tex", "\n".join(lines))
    print(f"tab5_cross_release.tex -> {path}")
    print(f"  pairs={len(pair_order)} comparisons={len(cr)} same_sign={n_same} opposite_cis={n_opp}")


def make_tab6() -> None:
    rg = pd.read_csv(DATA / "rq1" / "regret.csv")
    rg = rg[rg["weighting"] == "system_equal"].copy()
    lo = pd.read_csv(DATA / "rq1_cross6" / "family_loso_regret.csv")  # winner chosen without the held-out subsystem
    family_order = ["RE1", "RE2", "openrca", "petshop"]
    layer_order = ["L1", "L2"]
    layer_name = {"L1": "Published", "L2": "All"}
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lllrcrcr@{}}",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{All subsystems} & \multicolumn{2}{c}{Held out} \\",
        r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"Family & Layer & Pooled winner & \shortstack{Winner\\pooled acc.} & "
        r"\shortstack{Regret\\$> 0$} & "
        r"\shortstack{Max regret\\(subsystem)} & \shortstack{Regret\\$> 0$} & Max \\",
        r"\midrule",
    ]
    n_rows = 0
    summaries = []
    for fam in family_order:
        for layer in layer_order:
            g = rg[(rg["family"] == fam) & (rg["layer"] == layer)]
            if g.empty:
                raise ValueError(f"no regret rows for {fam} {layer}")
            winners = g["winner"].unique()
            pooled = g["winner_pooled"].unique()
            if len(winners) != 1 or len(pooled) != 1:
                raise ValueError(f"inconsistent winner for {fam} {layer}")
            k = len(g)
            n_pos = int((g["regret"] > 1e-9).sum())
            mx = float(g["regret"].max())
            if mx <= 1e-9:
                max_cell = "0"
            else:
                tops = g[g["regret"] > mx - 1e-12]
                # If several subsystems share the max, name all of them.
                names = ", ".join(display_sub(s) for s in tops["subsystem"])
                max_cell = f"{fmt3(mx)} ({names})"
            h = lo[(lo["family"] == fam) & (lo["layer"] == layer)]
            if h.empty:
                held = "-- & --"
            else:
                hm = float(h["regret"].max())
                held = f"{int((h['regret'] > 1e-9).sum())} of {len(h)} & {fmt3(hm) if hm > 1e-9 else '0'}"
            lines.append(
                f"{display_family(fam)} & {layer_name[layer]} & "
                f"{display_method(winners[0])} & {fmt3(float(pooled[0]))} & "
                f"{n_pos} of {k} & {max_cell} & {held} \\\\"
            )
            n_rows += 1
            summaries.append(
                f"{fam}/{layer}: winner={winners[0]} acc={float(pooled[0]):.3f} "
                f"regret>0 {n_pos}/{k} max={mx:.3f}"
            )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = write("tab6_regret.tex", "\n".join(lines))
    print(f"tab6_regret.tex -> {path}")
    print(f"  rows={n_rows}")
    for s in summaries:
        print(f"  {s}")


def _factor_rows(df: pd.DataFrame, rev_alt_col: str) -> pd.DataFrame:
    rows = []
    for (fam, layer), g in df.groupby(["family", "layer"], sort=False):
        rows.append(
            {
                "family": fam,
                "layer": layer,
                "n": int(len(g)),
                "sign": int(g["sign_change"].sum()),
                "rev": int((g["rev_ref"] != g[rev_alt_col]).sum()),
            }
        )
    return pd.DataFrame(rows)


def seed_rows() -> pd.DataFrame:
    """Seed as a condition (2026-10-03): for every comparison (pair x subsystem) that involves a randomized method, the
    three-seed mean (the reference) against any single seed. Sign: the subsystem difference has a different sign under
    some seed; Rev: the comparison is a reversal (opposite sign to the pooled difference of the same variant) under the
    mean or under some seed but not under both. Source: rq1/pairs.csv and rq1/pair_subsystem_deltas.csv, variants
    mean/seed0/seed1/seed2 (PetShop's RCD is run once, so PetShop has no row)."""
    pairs = pd.read_csv(DATA / "rq1" / "pairs.csv")
    deltas = pd.read_csv(DATA / "rq1" / "pair_subsystem_deltas.csv")
    seeds = sorted(v for v in pairs["variant"].unique() if v != "mean")
    if seeds != ["seed0", "seed1", "seed2"]:
        raise ValueError(f"unexpected seed variants: {seeds}")

    def sgn(x: float) -> int:
        return int(np.sign(np.round(x, 12)))

    rows = []
    for (fam, layer), pm in pairs.groupby(["family", "layer"], sort=False):
        seeded = pm[pm["variant"] != "mean"][["a", "b"]].drop_duplicates()
        if seeded.empty:
            continue
        n = sign = rev = 0
        for a, b in seeded.itertuples(index=False):
            pooled = {}
            for v in ["mean"] + seeds:
                r = pm[(pm["a"] == a) & (pm["b"] == b) & (pm["variant"] == v)]
                if len(r) != 1:
                    raise ValueError(f"{fam} {layer} {a} vs {b} {v}: {len(r)} pair rows")
                pooled[v] = float(r["pooled_se"].iloc[0])
            d = deltas[(deltas["family"] == fam) & (deltas["layer"] == layer) & (deltas["a"] == a) & (deltas["b"] == b)]
            for sub in d[d["variant"] == "mean"]["sub"]:
                delta = {}
                for v in pooled:
                    r = d[(d["variant"] == v) & (d["sub"] == sub)]
                    if len(r) != 1:
                        raise ValueError(f"{fam} {layer} {a} vs {b} {sub} {v}: {len(r)} delta rows")
                    delta[v] = float(r["delta"].iloc[0])

                def is_rev(v: str) -> bool:
                    return pooled[v] != 0 and sgn(delta[v]) != 0 and sgn(delta[v]) != sgn(pooled[v])

                n += 1
                sign += any(sgn(delta[v]) != sgn(delta["mean"]) for v in seeds)
                rev += any(is_rev(v) != is_rev("mean") for v in seeds)
        rows.append({"family": fam, "layer": layer, "n": n, "sign": sign, "rev": rev})
    return pd.DataFrame(rows)


def make_tab7() -> None:
    cond = pd.read_csv(DATA / "rq2" / "condition_comparisons.csv")
    blocks = []

    inp = cond[cond["factor"] == "input_representation"].copy()
    n_ob = int((inp["sub"] == "RE1::OB").sum())
    inp = inp[inp["sub"] != "RE1::OB"]
    blocks.append(("Input representation", _factor_rows(inp, "rev_alt")))

    rd = cond[cond["factor"] == "release_defect"]  # RE1-OB duplicate "time" header, 50 cases (added 2026-09-30)
    if len(rd):
        blocks.append(("Silent failure (duplicated header)", _factor_rows(rd, "rev_alt")))

    # row order follows the four pipeline stages of RQ2 (2026-10-02): input, failure handling, denominator, weighting/scoring
    den = pd.read_csv(DATA / "rq2" / "denominator_valid_only.csv")
    blocks.append(("Denominator (valid outputs only)", _factor_rows(den, "rev_alt")))

    fw = pd.read_csv(DATA / "rq2" / "fault_reweighted.csv")
    blocks.append(("Fault re-weighting", _factor_rows(fw, "rev_rw")))

    strict = cond[cond["factor"] == "scoring_strict"]
    blocks.append(("Scoring (strict $=1$)", _factor_rows(strict, "rev_alt")))

    ge05 = cond[cond["factor"] == "scoring_ge05"]
    blocks.append((r"Scoring ($\geq 0.5$)", _factor_rows(ge05, "rev_alt")))

    # not a stage of the released pipeline, but an unstated part of a reported number (2026-10-03)
    blocks.append(("Seed (one seed vs.\\ three-seed mean)", seed_rows()))

    family_order = ["RE1", "RE2", "openrca", "petshop"]
    layer_order = ["L1", "L2"]
    layer_name = {"L1": "Published", "L2": "All"}

    # one row per factor x family; published-only and all-pairs counts side by side (compact layout, 2026-09-30)
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Published methods} & \multicolumn{3}{c}{All pairs} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"Factor & Family & Comp. & Sign & Rev. & Comp. & Sign & Rev. \\",
        r"\midrule",
    ]
    plain = []
    n_rows = 0
    for fi, (factor, rec) in enumerate(blocks):
        if fi:
            lines.append(r"\addlinespace")
        rec = rec.copy()
        if not rec["family"].isin(family_order).all() or not rec["layer"].isin(layer_order).all():
            raise ValueError(f"unexpected family/layer in {factor}:\n{rec}")
        first = True
        for fam in family_order:
            g = rec[rec["family"] == fam].set_index("layer")
            if g.empty:
                continue
            cells = []
            for lay in layer_order:
                if lay in g.index:
                    r = g.loc[lay]
                    cells += [str(int(r["n"])), str(int(r["sign"])), str(int(r["rev"]))]
                    plain.append(f"{factor} | {display_family(fam)} | {layer_name[lay]} | "
                                 f"{int(r['n'])} | {int(r['sign'])} | {int(r['rev'])}")
                else:
                    cells += ["--", "--", "--"]
            lines.append(f"{factor if first else ''} & {display_family(fam)} & " + " & ".join(cells) + r" \\")
            first = False
            n_rows += 1
    lines += [r"\bottomrule", r"\end{tabular*}", ""]
    path = write("tab7_factors.tex", "\n".join(lines))
    print(f"tab7_factors.tex -> {path}")
    print(f"  rows={n_rows} input_RE1_OB_excluded={n_ob}")
    print("  Factor | Family | Layer | Comparisons | Sign changes | Reversal-status changes")
    for row in plain:
        print(f"  {row}")


def make_tab8() -> None:
    """Cross-family comparison set (six methods on all 11 subsystems; rq1_cross6): sign-based columns only,
    because OpenRCA partial credit and top-1 accuracy are not averaged across families."""
    xp = pd.read_csv(DATA / "rq1_cross6" / "cross_pairs.csv")
    xd = pd.read_csv(DATA / "rq1_cross6" / "cross_pair_subsystem_deltas.csv")
    lofo = pd.read_csv(DATA / "rq1_cross6" / "cross_lofo.csv")
    lkso = pd.read_csv(DATA / "rq1_cross6" / "cross_lkso2.csv")
    lines = [
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Pair $A$ vs $B$ & \shortstack{Subsystems\\$+$ / $-$} & \shortstack{CI excl.\ 0\\$+$ / $-$} & "
        r"\shortstack{Both signs,\\family dropped} & \shortstack{Both signs,\\two subsystems dropped} \\",
        r"\midrule",
    ]
    for _, r in xp.iterrows():
        d = xd[(xd["a"] == r["a"]) & (xd["b"] == r["b"])]
        assert len(d) == 11, (r["a"], r["b"], len(d))
        ci_pos = int((d["lo"] > 0).sum()); ci_neg = int((d["hi"] < 0).sum())
        assert int((d["delta"] > 0).sum()) == int(r["n_pos"]) and int((d["delta"] < 0).sum()) == int(r["n_neg"])
        f = lofo[(lofo["a"] == r["a"]) & (lofo["b"] == r["b"])]
        assert len(f) == 3, (r["a"], r["b"], len(f))
        f_both = int(((f["n_pos"] > 0) & (f["n_neg"] > 0)).sum())
        k = lkso[(lkso["a"] == r["a"]) & (lkso["b"] == r["b"])]
        assert len(k) == 1 and int(k.iloc[0]["n_drops"]) == 55
        lines.append(
            f"{display_method(r['a'])} vs {display_method(r['b'])} & {int(r['n_pos'])} / {int(r['n_neg'])} & "
            f"{ci_pos} / {ci_neg} & {f_both} of 3 & {int(k.iloc[0]['both_signs'])} of 55 \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = write("tab8_cross.tex", "\n".join(lines))
    print(f"tab8_cross.tex -> {path} rows={len(xp)}")


def main() -> None:
    make_tab1()
    make_tab4()
    make_tab5()
    make_tab6()
    make_tab7()
    make_tab8()


if __name__ == "__main__":
    main()
