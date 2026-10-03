#!/usr/bin/env python3
"""Build additional longtables S1--S12.

Every number is read from paper/data/. See SUPP_SPEC.md and SPEC.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_tables import (  # noqa: E402
    METHOD_DISPLAY,
    display_family,
    display_method,
    display_sub,
    fmt3,
    fmt_ci,
    fmt_signed,
    tex_escape,
)

DATA = ROOT / "paper" / "data"
OUT = ROOT / "paper" / "supp" / "tables"

FAMILY_ORDER = ["RE1", "RE2", "openrca", "petshop"]
LAYER_ORDER = ["L1", "L2"]
LAYER_LABEL = {"L1": "Published", "L2": "All"}

# Names that appear in published-table dumps but not in the audit method map.
# Display is the published label, with LaTeX specials escaped; raw audit ids
# still go through display_method.
PUBLISHED_NAME_DISPLAY = {
    "epsilon_diagnosis": r"$\epsilon$-Diagnosis",
    "counterfactual": "Counterfactual",
    "correlation": "Ranked correlation",
    "traversal": "Traversal",
    "circa": "CIRCA",
    "rcd": "RCD",
}


def layer_label(raw: str) -> str:
    if raw not in LAYER_LABEL:
        raise KeyError(f"no display name for layer {raw!r}")
    return LAYER_LABEL[raw]


def checkmark(flag: bool) -> str:
    return r"\checkmark" if bool(flag) else ""


def fmt_p(x: float) -> str:
    """Three decimals, or scientific notation when that would print as 0.000."""
    if pd.isna(x):
        return "--"
    x = float(x)
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax < 5e-4:
        mant, exp = f"{ax:.2e}".split("e")
        exp_i = int(exp)
        return rf"{mant}$\times 10^{{{exp_i}}}$"
    return f"{x:.3f}"


def fmt_delta_ci(delta: float, lo: float, hi: float) -> str:
    return f"{fmt_signed(delta)} {fmt_ci(lo, hi)}"


def pair_label(a: str, b: str) -> str:
    return f"{display_method(a)} vs {display_method(b)}"


def published_pair_name(raw: str) -> str:
    if raw in METHOD_DISPLAY:
        return display_method(raw)
    if raw in PUBLISHED_NAME_DISPLAY:
        return PUBLISHED_NAME_DISPLAY[raw]
    return tex_escape(raw)


def seed_cell(seed) -> str:
    if pd.isna(seed):
        return "--"
    return str(int(seed))


def is_true(value) -> bool:
    """Literal True, including the string 'True' if a CSV was read as text."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def sort_family_layer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_fam"] = pd.Categorical(out["family"], categories=FAMILY_ORDER, ordered=True)
    out["_lay"] = pd.Categorical(out["layer"], categories=LAYER_ORDER, ordered=True)
    out = out.sort_values(["_fam", "_lay"], kind="mergesort").drop(columns=["_fam", "_lay"])
    return out


def longtable(colspec: str, caption: str, label: str, header: str, body: str) -> str:
    """One longtable with a repeated header."""
    n = ncols_of(colspec)
    cap = r"\caption{" + caption + r"}\\"
    first_cap = cap + r"\label{" + label + r"}\\"
    return (
        "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
        f"\\begin{{longtable}}{{{colspec}}}\n"
        f"{first_cap}\n"
        r"\toprule" + "\n"
        f"{header} \\\\\n"
        r"\midrule" + "\n"
        r"\endfirsthead" + "\n"
        f"{cap}\n"
        r"\toprule" + "\n"
        f"{header} \\\\\n"
        r"\midrule" + "\n"
        r"\endhead" + "\n"
        r"\midrule" + "\n"
        rf"\multicolumn{{{n}}}{{r}}{{\emph{{Continued on next page}}}} \\" + "\n"
        r"\bottomrule" + "\n"
        r"\endfoot" + "\n"
        r"\bottomrule" + "\n"
        r"\endlastfoot" + "\n"
        f"{body}"
        r"\end{longtable}" + "\n"
        "}\n"
    )


def ncols_of(colspec: str) -> int:
    return colspec.count("l") + colspec.count("c") + colspec.count("r")


def write_tex(name: str, body: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(body)
    return path


def load(rel: str) -> pd.DataFrame:
    return pd.read_csv(DATA / rel)


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------

def make_s1() -> int:
    df = load("rq1/pairs.csv")
    df = df[df["variant"] == "mean"].copy()
    df = sort_family_layer(df)
    colspec = "llrrrrrrrlc"
    header = (
        "Pair & $k$ & \\shortstack{Pooled $\\Delta$\\\\(equal) [CI]} & \\shortstack{Pooled $\\Delta$\\\\(case-wtd.)} "
        "& \\#rev & \\shortstack{\\#rev\\\\CI} & \\shortstack{SD of\\\\$\\Delta_s$} & \\shortstack{Below\\\\line} "
        "& 95\\% PI & \\shortstack{LRT $p$\\\\(Holm)} & Sep."
    )
    lines = []
    prev_key = None
    for r in df.itertuples(index=False):
        key = (r.family, r.layer)
        if key != prev_key:
            if prev_key is not None:
                lines.append(r"\addlinespace" + "\n")
            title = f"{display_family(r.family)}, {layer_label(r.layer)}"
            lines.append(
                rf"\multicolumn{{{ncols_of(colspec)}}}{{l}}"
                rf"{{\emph{{{title}}}}} \\" + "\n"
            )
            prev_key = key
        below = abs(float(r.pooled_se)) < float(r.sd_sub_delta)
        if pd.notna(r.re_pi_lo) and pd.notna(r.re_pi_hi):
            pi = fmt_ci(float(r.re_pi_lo), float(r.re_pi_hi))
        else:
            pi = "--"
        cells = [
            pair_label(r.a, r.b),
            str(int(r.k)),
            fmt_delta_ci(r.pooled_se, r.pooled_se_lo, r.pooled_se_hi),
            fmt_signed(r.pooled_cw),
            str(int(r.n_rev_se)),
            str(int(r.n_rev_ci_se)),
            fmt3(r.sd_sub_delta),
            checkmark(below),
            pi,
            fmt_p(r.lrt_p_holm),
            checkmark(r.separation),
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        "All method pairs in the main analysis (seed-averaged). "
        "Pooled $\\Delta$ (equal) weights subsystems equally and is shown with its 95\\% CI; "
        "the case-weighted pooled $\\Delta$ weights subsystems by case count. "
        "\\#rev is the number of subsystems whose $\\Delta$ has the opposite sign of the "
        "equal-weight pooled $\\Delta$; \\#rev CI counts those whose CI excludes 0. "
        "Below line marks $|\\mathrm{pooled}\\,\\Delta| < \\mathrm{SD}$ of subsystem $\\Delta$. "
        "The 95\\% prediction interval is reported only where the random-effects fit supplies one. "
        "LRT $p$ is Holm-adjusted. Sep.\\ marks complete separation."
    )
    body = longtable(colspec, caption, "supp:s1-pairs", header, "".join(lines))
    write_tex("s1_pairs.tex", body)
    return len(df)


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------

def make_s2() -> int:
    df = load("rq1/pairs.csv")
    df = df[df["variant"].isin(["seed0", "seed1", "seed2"])].copy()
    # itertuples renames columns that start with "_", so the seed number is
    # parsed from `variant` again when the row is written.
    df["seed_n"] = df["variant"].map(lambda v: int(v[len("seed"):]))
    df["_fam"] = pd.Categorical(df["family"], FAMILY_ORDER, ordered=True)
    df["_lay"] = pd.Categorical(df["layer"], LAYER_ORDER, ordered=True)
    df = df.sort_values(["_fam", "_lay", "a", "b", "seed_n"], kind="mergesort")
    colspec = "llllrr"
    header = (
        "Family & Layer & Pair & Seed & Pooled $\\Delta$ [CI] & \\#rev (\\#rev CI)"
    )
    lines = []
    for r in df.itertuples(index=False):
        rev = f"{int(r.n_rev_se)} ({int(r.n_rev_ci_se)})"
        cells = [
            display_family(r.family),
            layer_label(r.layer),
            pair_label(r.a, r.b),
            str(int(r.seed_n)),
            fmt_delta_ci(r.pooled_se, r.pooled_se_lo, r.pooled_se_hi),
            rev,
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        "Per-seed pair results (seeds 0, 1, and 2). "
        "Pooled $\\Delta$ is the equal-subsystem-weight difference with its 95\\% CI. "
        "\\#rev is the number of subsystems reversing the pooled sign; "
        "the parenthetical count is the number of those reversals whose CI excludes 0."
    )
    body = longtable(colspec, caption, "supp:s2-seeds", header, "".join(lines))
    write_tex("s2_seeds.tex", body)
    return len(df)


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def make_s3() -> int:
    deltas = load("rq1/pair_subsystem_deltas.csv")
    deltas = deltas[(deltas["variant"] == "mean") & (deltas["layer"] == "L1")].copy()
    pairs = load("rq1/pairs.csv")
    pairs = pairs[(pairs["variant"] == "mean") & (pairs["layer"] == "L1")]
    pooled = pairs.set_index(["family", "a", "b"])["pooled_se"]
    deltas["_fam"] = pd.Categorical(deltas["family"], FAMILY_ORDER, ordered=True)
    deltas = deltas.sort_values(["_fam", "a", "b", "sub"], kind="mergesort")
    colspec = "llllrrr"
    header = (
        r"Family & Pair & Subsystem & $n$ & Acc.\ A & Acc.\ B & $\Delta$ [CI]"
    )
    lines = []
    marked = 0
    for r in deltas.itertuples(index=False):
        pooled_se = float(pooled.loc[(r.family, r.a, r.b)])
        opposite = (float(r.delta) * pooled_se) < 0
        if opposite:
            marked += 1
        star = r"$^\ast$" if opposite else ""
        cells = [
            display_family(r.family),
            pair_label(r.a, r.b),
            display_sub(r.sub),
            str(int(r.n)),
            fmt3(r.acc_a),
            fmt3(r.acc_b),
            fmt_delta_ci(r.delta, r.lo, r.hi) + star,
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        "Per-subsystem accuracy differences for published-method pairs (seed-averaged). "
        r"$\Delta$ is accuracy of A minus accuracy of B, with its 95\% CI. "
        r"A row marked $^\ast$ has a $\Delta$ whose sign is opposite to that pair's "
        r"equal-subsystem-weight pooled $\Delta$."
    )
    body = longtable(colspec, caption, "supp:s3-subsystem-deltas", header, "".join(lines))
    write_tex("s3_subsystem_deltas.tex", body)
    print(f"  S3 opposite-sign rows: {marked}")
    return len(deltas)


# ---------------------------------------------------------------------------
# S4
# ---------------------------------------------------------------------------

def make_s4() -> int:
    df = load("rq1/small_sample.csv")
    rows = []
    for (n_min, family, layer), g in df.groupby(["n_min", "family", "layer"], sort=False):
        rows.append(
            {
                "n_min": int(n_min),
                "family": family,
                "layer": layer,
                "n_pairs": int(len(g)),
                "n_rev": int((g["n_rev"] >= 1).sum()),
                "n_rev_ci": int((g["n_rev_ci"] >= 1).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["_fam"] = pd.Categorical(out["family"], FAMILY_ORDER, ordered=True)
    out["_lay"] = pd.Categorical(out["layer"], LAYER_ORDER, ordered=True)
    out = out.sort_values(["n_min", "_fam", "_lay"], kind="mergesort")
    colspec = "lllrrr"
    header = (
        r"$n_{\min}$ & Family & Layer & Pairs & \shortstack{Pairs with\\$\geq 1$ reversal} "
        r"& \shortstack{With CI-supported\\reversal}"
    )
    lines = []
    for r in out.itertuples(index=False):
        cells = [
            str(int(r.n_min)),
            display_family(r.family),
            layer_label(r.layer),
            str(int(r.n_pairs)),
            str(int(r.n_rev)),
            str(int(r.n_rev_ci)),
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        r"Small-sample sensitivity. For each minimum case count $n_{\min}$, family, and layer, "
        "the number of pairs, the number of pairs with at least one subsystem reversal, "
        "and the number of pairs with at least one CI-supported reversal."
    )
    body = longtable(colspec, caption, "supp:s4-small-sample", header, "".join(lines))
    write_tex("s4_small_sample.tex", body)
    return len(out)


# ---------------------------------------------------------------------------
# S5
# ---------------------------------------------------------------------------

def make_s5() -> int:
    df = load("rq2/coverage.csv")
    # File order of subsystems, then method, then seed (NaN seed sorts as -1,
    # which keeps the unseeded row before seeds 0/1/2).
    df["sub_order"] = pd.Categorical(
        df["subsystem_id"], categories=list(dict.fromkeys(df["subsystem_id"])), ordered=True
    )
    df["seed_order"] = df["seed"].fillna(-1)
    df = df.sort_values(["sub_order", "method", "seed_order"], kind="mergesort")
    colspec = "llrrrrrr"
    header = (
        r"Subsystem & Method & Seed & $n$ & Valid & Coverage "
        r"& Acc.\ (all cases) & Acc.\ (valid only)"
    )
    lines = []
    for r in df.itertuples(index=False):
        cells = [
            display_sub(r.subsystem_id),
            display_method(r.method),
            seed_cell(r.seed),
            str(int(r.n)),
            str(int(r.n_valid)),
            fmt3(r.coverage),
            fmt3(r.acc_all),
            fmt3(r.acc_valid_only),
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        "Output coverage and accuracy for every subsystem, method, and seed. "
        "Valid is the number of cases with a usable ranking. "
        "Accuracy (all cases) uses every case as the denominator (an unusable output counts as a miss); "
        "accuracy (valid only) uses only cases with a usable ranking. "
        "A seed of -- means the method was not re-run across seeds."
    )
    body = longtable(colspec, caption, "supp:s5-coverage", header, "".join(lines))
    write_tex("s5_coverage.tex", body)
    return len(df)


# ---------------------------------------------------------------------------
# S6
# ---------------------------------------------------------------------------

def make_s6() -> int:
    df = load("rq2/input_accuracy.csv")
    colspec = "llrrl"
    header = (
        r"Subsystem & Method & \shortstack{Acc.\ reduced\\input} & \shortstack{Acc.\ raw\\input} & Files (reduced / raw)"
    )
    lines = []
    for r in df.itertuples(index=False):
        files = tex_escape(f"{r.input_correct} / {r.input_submitted}")
        cells = [
            display_sub(r.sub),
            display_method(r.method),
            fmt3(r.acc_correct_input),
            fmt3(r.acc_submitted_input),
            files,
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        "Accuracy under the reduced (simple) input released for analysis and under the raw export. "
        "The Files column names the reduced file and the raw file."
    )
    body = longtable(colspec, caption, "supp:s6-input", header, "".join(lines))
    write_tex("s6_input.tex", body)
    return len(df)


# ---------------------------------------------------------------------------
# S7
# ---------------------------------------------------------------------------

def make_s7() -> int:
    df = load("rq2/openrca_scoring_accuracy.csv")
    colspec = "llrrr"
    header = r"Subsystem & Method & Partial credit & Strict ($=1$) & $\geq 0.5$"
    lines = []
    for r in df.itertuples(index=False):
        cells = [
            display_sub(r.subsystem_id),
            display_method(r.method),
            fmt3(r.score),
            fmt3(r.score_strict),
            fmt3(r.ge05),
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        "OpenRCA scoring variants. Partial credit is the released fractional score; "
        r"strict counts only a score of exactly 1; $\geq 0.5$ counts a score of at least 0.5."
    )
    body = longtable(colspec, caption, "supp:s7-openrca-scoring", header, "".join(lines))
    write_tex("s7_openrca_scoring.tex", body)
    return len(df)


# ---------------------------------------------------------------------------
# S8 (two longtables in one file)
# ---------------------------------------------------------------------------

def make_s8() -> tuple[int, int]:
    acc = load("rq2/release_defect_accuracy.csv")
    colspec_a = "lrrrrr"
    header_a = (
        r"Method & \shortstack{Acc.\\fixed} & \shortstack{Acc.\\as shipped} & \shortstack{Acc.\ fixed\\(affected)} "
        r"& \shortstack{Acc.\ as shipped\\(affected)} & \shortstack{$n$\\affected}"
    )
    lines_a = []
    for r in acc.itertuples(index=False):
        cells = [
            display_method(r.method),
            fmt3(r.acc_fixed),
            fmt3(r.acc_shipped),
            fmt3(r.acc_fixed_affected),
            fmt3(r.acc_shipped_affected),
            str(int(r.n_affected)),
        ]
        lines_a.append(" & ".join(cells) + r" \\" + "\n")
    caption_a = (
        "Accuracy with the released RE1-OB duplicate-header silent failure left as shipped, "
        "and with that header fixed. Affected columns restrict both accuracies to the "
        "cases whose input row is changed by the fix."
    )
    part_a = longtable(
        colspec_a, caption_a, "supp:s8-release-defect", header_a, "".join(lines_a)
    )

    cc = load("rq2/condition_comparisons.csv")
    cc = cc[cc["factor"] == "release_defect"].copy()
    keep = cc["sign_change"].map(is_true) | (cc["rev_ref"] != cc["rev_alt"])
    cc = cc[keep]
    cc["_lay"] = pd.Categorical(cc["layer"], LAYER_ORDER, ordered=True)
    cc = cc.sort_values(["_lay", "a", "b", "sub"], kind="mergesort")
    colspec_b = "lllll"
    header_b = r"Layer & Pair & Subsystem & $\Delta$ fixed [CI] & $\Delta$ as shipped [CI]"
    lines_b = []
    for r in cc.itertuples(index=False):
        cells = [
            layer_label(r.layer),
            pair_label(r.a, r.b),
            display_sub(r.sub),
            fmt_delta_ci(r.delta_ref, r.lo_ref, r.hi_ref),
            fmt_delta_ci(r.delta_alt, r.lo_alt, r.hi_alt),
        ]
        lines_b.append(" & ".join(cells) + r" \\" + "\n")
    caption_b = (
        "Pair--subsystem comparisons whose sign, or whose reversal status, changes "
        "between the fixed release and the release as shipped. "
        r"$\Delta$ fixed is the reference (header repaired); $\Delta$ as shipped is the alternative."
    )
    part_b = longtable(
        colspec_b,
        caption_b,
        "supp:s8-release-defect-changes",
        header_b,
        "".join(lines_b),
    )
    write_tex("s8_release_defect.tex", part_a + "\n" + part_b)
    return len(acc), len(cc)


# ---------------------------------------------------------------------------
# S9
# ---------------------------------------------------------------------------

def make_s9() -> int:
    df = load("rq1/published_tables_margin_dispersion.csv")
    colspec = "llrrrrrc"
    header = (
        r"Table & Pair & $k$ & Pooled & $|\mathrm{margin}|$ & SD "
        r"& \shortstack{Below\\line} & \shortstack{\#strata\\reversing}"
    )
    lines = []
    prev = None
    for r in df.itertuples(index=False):
        if r.table != prev:
            if prev is not None:
                lines.append(r"\addlinespace" + "\n")
            prev = r.table
        cells = [
            tex_escape(str(r.table)),
            f"{published_pair_name(r.a)} vs {published_pair_name(r.b)}",
            str(int(r.k)),
            fmt_signed(r.pooled),
            fmt3(r.abs_margin),
            fmt3(r.sd_units),
            checkmark(r.margin_lt_sd),
            str(int(r.n_rev)),
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        "Margin and between-stratum dispersion inside two published tables "
        "(RCAEval Table 6 and PetShop Table 8). "
        r"Below line marks $|\mathrm{pooled\ margin}| < \mathrm{SD}$ across strata. "
        r"\#strata reversing is the number of strata whose sign disagrees with the pooled margin."
    )
    body = longtable(
        colspec, caption, "supp:s9-published-tables", header, "".join(lines)
    )
    write_tex("s9_published_tables.tex", body)
    return len(df)


# ---------------------------------------------------------------------------
# S10 (two longtables)
# ---------------------------------------------------------------------------

def make_s10() -> tuple[int, int]:
    comp = load("rq2/fault_composition.csv")
    id_col = "subsystem_id"
    value_cols = [c for c in comp.columns if c != id_col]
    keep_cols = []
    for c in value_cols:
        if not pd.api.types.is_numeric_dtype(comp[c]):
            keep_cols.append(c)
            continue
        if (comp[c].fillna(0) == 0).all():
            continue
        keep_cols.append(c)
    shown = comp[[id_col] + keep_cols]
    colspec_a = "l" + "r" * len(keep_cols)
    header_a = "Subsystem & " + " & ".join(tex_escape(c) for c in keep_cols)
    lines_a = []
    for r in shown.itertuples(index=False):
        cells = [display_sub(getattr(r, id_col))]
        for c in keep_cols:
            val = getattr(r, c)
            cells.append(str(int(val)) if pd.notna(val) else "--")
        lines_a.append(" & ".join(cells) + r" \\" + "\n")
    caption_a = (
        "Case counts per fault class and subsystem. "
        "Fault classes that are zero in every subsystem are omitted."
    )
    part_a = longtable(
        colspec_a, caption_a, "supp:s10-fault-composition", header_a, "".join(lines_a)
    )

    rw = load("rq2/fault_reweighted.csv")
    rw = rw[rw["sign_change"].map(is_true)].copy()
    rw["_fam"] = pd.Categorical(rw["family"], FAMILY_ORDER, ordered=True)
    rw["_lay"] = pd.Categorical(rw["layer"], LAYER_ORDER, ordered=True)
    rw = rw.sort_values(["_fam", "_lay", "a", "b", "sub"], kind="mergesort")
    colspec_b = "llllrr"
    header_b = (
        r"Family & Layer & Pair & Subsystem & $\Delta$ & $\Delta$ re-weighted [CI]"
    )
    lines_b = []
    for r in rw.itertuples(index=False):
        cells = [
            display_family(r.family),
            layer_label(r.layer),
            pair_label(r.a, r.b),
            display_sub(r.sub),
            fmt_signed(r.delta_ref),
            fmt_delta_ci(r.delta_rw, r.lo_rw, r.hi_rw),
        ]
        lines_b.append(" & ".join(cells) + r" \\" + "\n")
    caption_b = (
        "Comparisons whose sign changes when each fault class is re-weighted to a common mix. "
        r"$\Delta$ is the original per-subsystem difference; "
        r"$\Delta$ re-weighted is the re-weighted difference with its 95\% CI."
    )
    part_b = longtable(
        colspec_b, caption_b, "supp:s10-fault-reweighted", header_b, "".join(lines_b)
    )
    write_tex("s10_fault_composition.tex", part_a + "\n" + part_b)
    return len(shown), len(rw)


# ---------------------------------------------------------------------------
# S11
# ---------------------------------------------------------------------------

def shorten_venue(venue: str, limit: int = 30) -> str:
    """Shorten a venue to at most `limit` characters.

    papers.csv records several venues as an ASCII name plus a Chinese note,
    and two of them are ``arXiv preprint'' written in Chinese. The acceptance
    document is compiled with pdflatex, which cannot typeset those characters.
    Keep the ASCII prefix; where the prefix is only ``arXiv'' and the note
    says the paper is a preprint, append ``preprint'' so the cell still says
    what the source says. Cut a still-too-long ASCII string on a word boundary.
    """
    text = str(venue).strip()
    note = ""
    for opener in ("（", "("):
        head, sep, tail = text.partition(opener)
        if sep and any(ord(ch) > 127 for ch in tail):
            note = tail
            text = head.strip()
            break
    if any(ord(ch) > 127 for ch in text):
        # ``预印本'' is the Chinese wording of preprint in this file.
        if text.startswith("arXiv") and "预印本" in text:
            text = "arXiv preprint"
        else:
            text = "".join(ch for ch in text if ord(ch) < 128).strip()
    elif text == "arXiv" and "预印本" in note:
        text = "arXiv preprint"
    if len(text) <= limit:
        return tex_escape(text)
    cut = text[:limit].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return tex_escape(cut) + r"\ldots"


def make_s11() -> int:
    papers = load("survey/papers.csv")
    per_system = load("survey/per_system.csv").set_index("id")
    excluded = load("survey/excluded.csv").set_index("id")
    cell = {"yes": "Yes", "single_system": "Single system"}
    colspec = "lllcp{5.2cm}"
    header = r"ID & Year & Venue & Per-system results & Location"
    lines = []
    n = 0
    for r in papers.itertuples(index=False):
        pid = str(r.id)
        if pid in excluded.index:
            status, loc = "Excluded (no full text)", "--"
        else:
            if pid not in per_system.index:
                raise KeyError(f"per_system.csv has no row for {pid}")
            status = cell[str(per_system.loc[pid, "per_system"])]
            loc = tex_escape(str(per_system.loc[pid, "location"]))
            n += 1
        cells = [tex_escape(pid), str(int(r.year)), shorten_venue(r.venue), status, loc]
        lines.append(" & ".join(cells) + r" \\" + "\n")
        title = tex_escape(str(r.title)).replace("\u2014", "---")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\scriptsize\textit{{{title}}}}} \\" + "\n")
    if n != 23:
        raise ValueError(f"expected 23 full-text papers, found {n}")
    caption = (
        "The surveyed papers. Per-system results: Yes if the paper evaluates more than one system "
        "and reports a score for each; Single system if it evaluates one system; "
        "Excluded if no full text was obtained (not counted in the paper). "
        "Location: the table, figure, or passage checked. "
        "Venue names longer than 30 characters are truncated. "
        "The italic line under each row is the paper title."
    )
    body = longtable(colspec, caption, "supp:s11-survey", header, "".join(lines))
    write_tex("s11_survey.tex", body)
    return n


# ---------------------------------------------------------------------------
# S12
# ---------------------------------------------------------------------------

def make_s12() -> int:
    df = load("rq1/cross_release.csv")
    df = df[df["layer"] == "L2"].copy()
    app_order = list(dict.fromkeys(df["app"]))
    df["app_order"] = pd.Categorical(df["app"], categories=app_order, ordered=True)
    df = df.sort_values(["a", "b", "app_order"], kind="mergesort")
    colspec = "llrrcc"
    header = (
        r"Pair & App & $\Delta$ RE1 [CI] & $\Delta$ RE2 [CI] & Same sign & CIs opposite"
    )
    lines = []
    prev = None
    for r in df.itertuples(index=False):
        key = (r.a, r.b)
        pair = pair_label(r.a, r.b) if key != prev else ""
        prev = key
        cells = [
            pair,
            tex_escape(str(r.app)),
            fmt_delta_ci(r.delta_re1, r.lo_re1, r.hi_re1),
            fmt_delta_ci(r.delta_re2, r.lo_re2, r.hi_re2),
            checkmark(r.sign_agree),
            checkmark(r.ci_opposite),
        ]
        lines.append(" & ".join(cells) + r" \\" + "\n")
    caption = (
        "Cross-release comparison for every pair, including probes "
        "(the all-pairs layer). "
        r"$\Delta$ is accuracy on the RE1 release minus the paired method, and the same "
        r"contrast on the RE2 release, each with its 95\% CI. "
        "Same sign marks agreement of the two differences; "
        "CIs opposite marks intervals that lie on opposite sides of zero."
    )
    body = longtable(colspec, caption, "supp:s12-cross-release", header, "".join(lines))
    write_tex("s12_cross_release.tex", body)
    return len(df)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

INDEX_ENTRIES = [
    (
        "s1_pairs.tex",
        "All pairs, main analysis",
        "Every seed-averaged method pair, with both pooled differences, reversal counts, dispersion, and the Holm-adjusted likelihood-ratio test.",
        "paper/data/rq1/pairs.csv",
    ),
    (
        "s2_seeds.tex",
        "Per-seed pair results",
        "The same pair contrasts computed separately on each of the three seeds.",
        "paper/data/rq1/pairs.csv",
    ),
    (
        "s3_subsystem_deltas.tex",
        "Per-subsystem differences",
        "Accuracy of each published method on each subsystem, and the per-subsystem difference, for published-method pairs.",
        "paper/data/rq1/pair_subsystem_deltas.csv",
    ),
    (
        "s4_small_sample.tex",
        "Small-sample sensitivity",
        "How many pairs still reverse when subsystems below a minimum case count are set aside.",
        "paper/data/rq1/small_sample.csv",
    ),
    (
        "s5_coverage.tex",
        "Coverage and accuracy",
        "Valid-output coverage and accuracy under both denominators, for every subsystem, method, and seed.",
        "paper/data/rq2/coverage.csv",
    ),
    (
        "s6_input.tex",
        "Input representation",
        "Accuracy on the reduced input file and on the raw export, with the file names.",
        "paper/data/rq2/input_accuracy.csv",
    ),
    (
        "s7_openrca_scoring.tex",
        "OpenRCA scoring variants",
        "Partial-credit, exact-match, and at-least-0.5 accuracy on each OpenRCA subsystem.",
        "paper/data/rq2/openrca_scoring_accuracy.csv",
    ),
    (
        "s8_release_defect.tex",
        "Duplicated header: fixed vs as shipped",
        "Accuracy with the RE1-OB duplicate header fixed and as shipped, and the pair contrasts whose sign or reversal status changes.",
        "paper/data/rq2/release_defect_accuracy.csv",
    ),
    (
        "s9_published_tables.tex",
        "Published tables",
        "Pooled margin against between-stratum dispersion for every pair inside two published tables.",
        "paper/data/rq1/published_tables_margin_dispersion.csv",
    ),
    (
        "s10_fault_composition.tex",
        "Fault composition",
        "Case counts by fault class, and the comparisons whose sign changes after re-weighting fault classes.",
        "paper/data/rq2/fault_composition.csv",
    ),
    (
        "s11_survey.tex",
        "Surveyed papers",
        "The paper list with the per-system reporting check and the three papers excluded for lack of full text.",
        "paper/data/survey/papers.csv",
    ),
    (
        "s12_cross_release.tex",
        "Cross-release, all pairs",
        "RE1 versus RE2 differences for every pair, including probes and shipped baselines.",
        "paper/data/rq1/cross_release.csv",
    ),
]


def make_index() -> None:
    parts = [
        "% Auto-generated by paper/scripts/make_supp_tables.py. Do not edit.\n",
    ]
    for filename, title, sentence, csv_name in INDEX_ENTRIES:
        parts.append(f"\\subsection*{{{title}}}\n")
        parts.append(f"{sentence} Source: \\texttt{{{tex_escape(csv_name)}}}.\n\n")
        parts.append(f"\\input{{tables/{filename}}}\n\n")
    write_tex("index.tex", "".join(parts))


def main() -> None:
    counts = [
        ("S1 s1_pairs.tex", make_s1()),
        ("S2 s2_seeds.tex", make_s2()),
        ("S3 s3_subsystem_deltas.tex", make_s3()),
        ("S4 s4_small_sample.tex", make_s4()),
        ("S5 s5_coverage.tex", make_s5()),
        ("S6 s6_input.tex", make_s6()),
        ("S7 s7_openrca_scoring.tex", make_s7()),
    ]
    n_acc, n_chg = make_s8()
    counts.append(("S8 s8_release_defect.tex (accuracy rows)", n_acc))
    counts.append(("S8 s8_release_defect.tex (changed comparisons)", n_chg))
    counts.append(("S9 s9_published_tables.tex", make_s9()))
    n_comp, n_rw = make_s10()
    counts.append(("S10 s10_fault_composition.tex (composition rows)", n_comp))
    counts.append(("S10 s10_fault_composition.tex (sign-change rows)", n_rw))
    counts.append(("S11 s11_survey.tex", make_s11()))
    counts.append(("S12 s12_cross_release.tex", make_s12()))
    make_index()
    print("row counts:")
    for name, n in counts:
        print(f"  {name}: {n}")
    print(f"wrote {OUT / 'index.tex'}")


if __name__ == "__main__":
    main()
