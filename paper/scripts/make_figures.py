#!/usr/bin/env python3
"""Build the three vector figures for the RCA benchmark audit paper.

Every number is computed from paper/data/. See SPEC.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data"
OUT = ROOT / "paper" / "figures"

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

# Okabe–Ito, colour-blind safe.
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_BLACK = "#000000"
NEUTRAL = "#0072B2"
HIGHLIGHT = "#D55E00"
GREY_BAND = "#E6E6E6"

FULL_W = 5.5

# Marker cycle for subsystems inside one forest panel.
SUB_MARKERS = ["o", "s", "D", "^", "v", "P"]


def display_method(raw: str) -> str:
    if raw not in METHOD_DISPLAY:
        raise KeyError(f"no display name for method {raw!r}")
    return METHOD_DISPLAY[raw]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def load_forest():
    # Figure 1 panels are RE1, OpenRCA, PetShop. RE2 is the cross-release
    # check and is not in margin_dispersion.csv, so it is not drawn here.
    families = ["RE1", "openrca", "petshop"]
    deltas = pd.read_csv(DATA / "rq1" / "pair_subsystem_deltas.csv")
    deltas = deltas[
        (deltas["layer"] == "L1")
        & (deltas["variant"] == "mean")
        & (deltas["family"].isin(families))
    ].copy()
    pairs = pd.read_csv(DATA / "rq1" / "pairs.csv")
    pairs = pairs[
        (pairs["layer"] == "L1")
        & (pairs["variant"] == "mean")
        & (pairs["family"].isin(families))
    ].copy()
    margin = pd.read_csv(DATA / "rq1" / "margin_dispersion.csv")
    margin = margin[margin["layer"] == "L1"].copy()
    keep_p = pairs[
        ["family", "a", "b", "pooled_se", "sd_sub_delta", "k"]
    ].drop_duplicates()
    keep_m = margin[["family", "a", "b", "margin_lt_sd", "abs_margin"]].drop_duplicates()
    merged = deltas.merge(keep_p, on=["family", "a", "b"], how="left", validate="many_to_one")
    merged = merged.merge(keep_m, on=["family", "a", "b"], how="left", validate="many_to_one")
    if merged[["pooled_se", "sd_sub_delta", "margin_lt_sd"]].isna().any().any():
        raise ValueError("forest join left some pairs without pooled Δ or margin flag")
    return merged


def _forest_panel(ax, g, subs, sub_labels, title):
    """Draw one family's forest panel on ax. Returns (n_rows, n_opposite, n_grey)."""
    pair_keys = g.groupby(["a", "b"], sort=False)["pooled_se"].first().reset_index()
    pair_keys["abs"] = pair_keys["pooled_se"].abs()
    pair_keys = pair_keys.sort_values("abs", ascending=False, kind="mergesort").reset_index(drop=True)
    n = len(pair_keys)
    n_opposite = 0
    n_grey = 0
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(0, color="#666666", ls="--", lw=0.6, zorder=1)
    marker_for = {s: SUB_MARKERS[i % len(SUB_MARKERS)] for i, s in enumerate(subs)}
    lo = float(g["lo"].min())
    hi = float(g["hi"].max())
    span = max(hi - lo, 0.2)
    pad = 0.05 * span
    ax.set_xlim(lo - pad, hi + pad)
    for i, row in pair_keys.iterrows():
        subg = g[(g["a"] == row["a"]) & (g["b"] == row["b"])]
        pooled = float(row["pooled_se"])
        grey = bool(subg["margin_lt_sd"].iloc[0])
        if grey:
            ax.axhspan(i - 0.5, i + 0.5, color=GREY_BAND, zorder=0, lw=0)
            n_grey += 1
        ax.plot([pooled, pooled], [i - 0.34, i + 0.34], color=OI_BLACK, lw=1.0,
                solid_capstyle="butt", zorder=3)
        for _, p in subg.iterrows():
            delta = float(p["delta"])
            opposite = (delta > 0 and pooled < 0) or (delta < 0 and pooled > 0)
            if pooled == 0 or delta == 0:
                opposite = False
            excludes_0 = float(p["lo"]) > 0 or float(p["hi"]) < 0
            colour = HIGHLIGHT if opposite else NEUTRAL
            n_opposite += int(opposite)
            filled = opposite and excludes_0
            ax.plot([float(p["lo"]), float(p["hi"])], [i, i], color=colour, lw=0.7,
                    zorder=2, solid_capstyle="butt")
            ax.plot(delta, i, marker=marker_for[p["sub"]], ms=3.4, color=colour,
                    markerfacecolor=colour if filled else "white", markeredgewidth=0.7,
                    linestyle="none", zorder=4)
    labels = [f"{display_method(a)} vs {display_method(b)}" for a, b in zip(pair_keys["a"], pair_keys["b"])]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=6.5, pad=1.5, length=2)
    ax.set_xlabel(r"$\Delta = \mathrm{acc}_A - \mathrm{acc}_B$", fontsize=7, labelpad=1.5)
    ax.set_title(title, fontsize=8, loc="left", pad=12, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [
        Line2D([0], [0], marker=marker_for[s], color="#444444", markerfacecolor="white",
               markeredgewidth=0.7, linestyle="none", ms=3.4, label=lab)
        for s, lab in zip(subs, sub_labels)
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, borderpad=0.0,
              handletextpad=0.3, columnspacing=0.8, labelspacing=0.0, fontsize=6,
              ncol=len(handles), bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)
    return n, n_opposite, n_grey


def make_fig1() -> None:
    df = load_forest()
    panels = {
        "RE1": ("RCAEval RE1", ["RE1::OB", "RE1::SS", "RE1::TT"], ["OB", "SS", "TT"]),
        "openrca": ("OpenRCA",
                    ["openrca::bank", "openrca::mkt1", "openrca::mkt2", "openrca::tel"],
                    ["Bank", "Market-1", "Market-2", "Telecom"]),
        "petshop": ("PetShop",
                    ["petshop::high_traffic", "petshop::low_traffic",
                     "petshop::temporal_traffic1", "petshop::temporal_traffic2"],
                    ["High", "Low", "Temporal-1", "Temporal-2"]),
    }
    # Two columns (2026-10-03): RE1 and PetShop (10 pairs each) side by side,
    # OpenRCA (3 pairs) below left, shared legend below right.
    apply_style()
    fig = plt.figure(figsize=(FULL_W, 3.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 3.6], width_ratios=[1, 1],
                          left=0.19, right=0.99, top=0.91, bottom=0.07,
                          wspace=0.78, hspace=0.62)
    ax_re1 = fig.add_subplot(gs[0, 0])
    ax_pet = fig.add_subplot(gs[0, 1])
    ax_orca = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[1, 1])
    totals = [0, 0, 0]
    for ax, fam in ((ax_re1, "RE1"), (ax_pet, "petshop"), (ax_orca, "openrca")):
        title, subs, sub_labels = panels[fam]
        g = df[df["family"] == fam].copy()
        n, n_opp, n_grey = _forest_panel(ax, g, subs, sub_labels, title)
        totals[0] += n; totals[1] += n_opp; totals[2] += n_grey
        print(f"fig1 {fam}: pairs={n} subsystems={len(subs)} points={len(g)} grey_rows={n_grey}")
    ax_leg.axis("off")
    key = [
        Line2D([0], [0], marker="o", color=NEUTRAL, markerfacecolor="white", markeredgewidth=0.7,
               linestyle="-", lw=0.7, ms=3.4, label=r"$\Delta_s$ with 95% CI, same sign as pooled $\Delta$"),
        Line2D([0], [0], marker="o", color=HIGHLIGHT, markerfacecolor="white", markeredgewidth=0.7,
               linestyle="-", lw=0.7, ms=3.4, label="opposite sign, CI includes 0"),
        Line2D([0], [0], marker="o", color=HIGHLIGHT, markerfacecolor=HIGHLIGHT, markeredgewidth=0.7,
               linestyle="-", lw=0.7, ms=3.4, label="opposite sign, CI excludes 0"),
        Line2D([0], [0], color=OI_BLACK, lw=1.0, label=r"pooled $\Delta$ (subsystems weighted equally)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=GREY_BAND, edgecolor="none",
                      label="pooled margin < between-subsystem spread"),
    ]
    ax_leg.legend(handles=key, loc="center left", frameon=False, fontsize=6.3,
                  handlelength=1.6, handletextpad=0.5, labelspacing=0.55,
                  bbox_to_anchor=(-0.42, 0.5), borderaxespad=0.0)
    path = save(fig, "fig1_forest.pdf")
    print(f"fig1_forest.pdf -> {path} rows={totals[0]} opposite_points={totals[1]} grey_rows={totals[2]}")


def _scatter_counts(x, y, flag) -> tuple[int, int, int, int]:
    """below line: reverse/total ; above line: reverse/total. On-line points counted neither."""
    below = y < x
    above = y > x
    return (
        int((below & flag).sum()),
        int(below.sum()),
        int((above & flag).sum()),
        int(above.sum()),
    )


def make_fig2() -> None:
    md = pd.read_csv(DATA / "rq1" / "margin_dispersion.csv")
    pub = pd.read_csv(DATA / "rq1" / "published_tables_margin_dispersion.csv")
    # L2 (all pairs) already contains every L1 (published) pair: plot each pair once,
    # from L2, and fill the markers of the pairs that are also in L1.
    key = ["family", "a", "b"]
    l1_keys = set(map(tuple, md.loc[md["layer"] == "L1", key].to_numpy()))
    ours = md[md["layer"] == "L2"].copy()
    ours["layer"] = ["L1" if tuple(r) in l1_keys else "L2" for r in ours[key].to_numpy()]
    assert len(ours) == (md["layer"] == "L2").sum() and (ours["layer"] == "L1").sum() == len(l1_keys)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35), sharex=True, sharey=True)

    family_marker = {"RE1": "o", "openrca": "s", "petshop": "D"}
    ax = axes[0]
    for fam, marker in family_marker.items():
        for layer, filled in (("L1", True), ("L2", False)):
            for rev, colour in ((True, HIGHLIGHT), (False, NEUTRAL)):
                g = ours[
                    (ours["family"] == fam)
                    & (ours["layer"] == layer)
                    & (ours["reverses"] == rev)
                ]
                if g.empty:
                    continue
                ax.scatter(
                    g["sd_sub_delta"], g["abs_margin"],
                    marker=marker, s=16,
                    facecolors=colour if filled else "none",
                    edgecolors=colour, linewidths=0.7, zorder=3,
                )
    b_rev, b_n, a_rev, a_n = _scatter_counts(
        ours["sd_sub_delta"].to_numpy(),
        ours["abs_margin"].to_numpy(),
        ours["reverses"].to_numpy(),
    )
    p1 = ours[ours["layer"] == "L1"]
    pb_rev1, pb_n1, pa_rev1, pa_n1 = _scatter_counts(
        p1["sd_sub_delta"].to_numpy(), p1["abs_margin"].to_numpy(), p1["reverses"].to_numpy(),
    )
    ax.set_title("Our audit")
    ax.set_xlabel("Between-subsystem SD of $\\Delta$")
    ax.set_ylabel("|pooled margin|")

    ax = axes[1]
    table_marker = {}
    # Spec: shape = table (RCAEval Table 6 / PetShop Table 8).
    # Map by a stable substring so a longer source label still matches.
    rules = (
        ("RCAEval Table 6", "o", "RCAEval Table 6"),
        ("PetShop Table 8", "s", "PetShop Table 8"),
    )
    for label, marker, short in rules:
        table_marker[label] = (marker, short)
    seen = set()
    for table, g0 in pub.groupby("table"):
        matched = None
        for label, marker, short in rules:
            if label in str(table):
                matched = (marker, short)
                break
        if matched is None:
            raise ValueError(f"unrecognised published table label: {table}")
        marker, short = matched
        seen.add(short)
        for rev, colour in ((True, HIGHLIGHT), (False, NEUTRAL)):
            g = g0[(g0["n_rev"] > 0) == rev]
            if g.empty:
                continue
            ax.scatter(
                g["sd_units"], g["abs_margin"],
                marker=marker, s=16,
                facecolors=colour, edgecolors=colour,
                linewidths=0.6, zorder=3,
            )
    if seen != {"RCAEval Table 6", "PetShop Table 8"}:
        raise ValueError(f"published tables present: {seen}")
    pb_rev, pb_n, pa_rev, pa_n = _scatter_counts(
        pub["sd_units"].to_numpy(),
        pub["abs_margin"].to_numpy(),
        (pub["n_rev"] > 0).to_numpy(),
    )
    ax.set_title("Published tables")
    ax.set_xlabel("Between-stratum SD of $\\Delta$")

    xmax = float(max(ours["sd_sub_delta"].max(), pub["sd_units"].max()))
    ymax = float(max(ours["abs_margin"].max(), pub["abs_margin"].max()))
    lim = max(xmax, ymax) * 1.08
    for a in axes:
        a.set_xlim(0, lim)
        a.set_ylim(0, lim)
        a.plot([0, lim], [0, lim], color="#888888", lw=0.6, zorder=1)
        a.set_aspect("equal", adjustable="box")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    # Left-aligned at x = 0.40: every point in both panels has SD <= 0.45, and
    # none of those high-SD points has |margin| above 0.30, so the block sits
    # right of the diagonal and clear of the markers.
    axes[0].text(
        0.40, 0.30,
        f"all pairs, below line: {b_rev}/{b_n} reverse\nabove: {a_rev}/{a_n}\n"
        f"published, below: {pb_rev1}/{pb_n1}\nabove: {pa_rev1}/{pa_n1}",
        va="bottom", ha="left", fontsize=5.5,
    )
    axes[1].text(
        0.40, 0.30,
        f"below line: {pb_rev}/{pb_n} reverse\nabove: {pa_rev}/{pa_n}",
        va="bottom", ha="left", fontsize=5.5,
    )

    # Legends sit above the axes: the upper-left holds real points.
    left_handles = [
        Line2D([0], [0], marker="o", color="#444444", linestyle="none",
               markerfacecolor="none", ms=4, label="RE1"),
        Line2D([0], [0], marker="s", color="#444444", linestyle="none",
               markerfacecolor="none", ms=4, label="OpenRCA"),
        Line2D([0], [0], marker="D", color="#444444", linestyle="none",
               markerfacecolor="none", ms=4, label="PetShop"),
        Line2D([0], [0], marker="o", color=NEUTRAL, linestyle="none",
               markerfacecolor=NEUTRAL, ms=4, label="published pair (filled)"),
        Line2D([0], [0], marker="o", color=NEUTRAL, linestyle="none",
               markerfacecolor="none", ms=4, label="other pair (hollow)"),
        Line2D([0], [0], marker="o", color=HIGHLIGHT, linestyle="none",
               markerfacecolor=HIGHLIGHT, ms=4, label="reverses"),
    ]
    right_handles = [
        Line2D([0], [0], marker="o", color="#444444", linestyle="none",
               markerfacecolor="#444444", ms=4, label="RCAEval Table 6"),
        Line2D([0], [0], marker="s", color="#444444", linestyle="none",
               markerfacecolor="#444444", ms=4, label="PetShop Table 8"),
        Line2D([0], [0], marker="o", color=HIGHLIGHT, linestyle="none",
               markerfacecolor=HIGHLIGHT, ms=4, label=r"$n_{\mathrm{rev}}>0$"),
        Line2D([0], [0], marker="o", color=NEUTRAL, linestyle="none",
               markerfacecolor=NEUTRAL, ms=4, label=r"$n_{\mathrm{rev}}=0$"),
    ]
    fig.legend(
        handles=left_handles, loc="lower left", frameon=False, fontsize=5.5,
        ncol=3, bbox_to_anchor=(0.02, -0.01), borderaxespad=0.0,
    )
    fig.legend(
        handles=right_handles, loc="lower right", frameon=False, fontsize=5.5,
        ncol=2, bbox_to_anchor=(0.98, -0.01), borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = save(fig, "fig2_margin_spread.pdf")
    print(f"fig2_margin_spread.pdf -> {path}")
    print(
        f"  our audit: n={len(ours)} L1={(ours.layer=='L1').sum()} "
        f"L2={(ours.layer=='L2').sum()} below {b_rev}/{b_n} reverse; above {a_rev}/{a_n}"
    )
    print(
        f"  published: n={len(pub)} below {pb_rev}/{pb_n} reverse; above {pa_rev}/{pa_n}"
    )


def make_fig3() -> None:
    acc = pd.read_csv(DATA / "rq2" / "input_accuracy.csv")
    ss = acc[acc["sub"] == "RE1::SS"].copy()
    tt = acc[acc["sub"] == "RE1::TT"].copy()
    if set(ss["method"]) != set(tt["method"]):
        raise ValueError("RE1-SS and RE1-TT method sets differ")
    ss = ss.sort_values(["acc_correct_input", "method"], ascending=[False, True], kind="mergesort")
    order = list(ss["method"])
    tt = tt.set_index("method").loc[order].reset_index()
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.3), sharex=True, sharey=True)
    n = len(order)
    y = np.arange(n)
    reduced_c = OI_BLUE
    raw_c = OI_VERMILLION
    for ax, frame, title in ((axes[0], ss, "Sock Shop (RE1-SS)"), (axes[1], tt, "Train Ticket (RE1-TT)")):
        reduced = frame["acc_correct_input"].to_numpy()
        raw = frame["acc_submitted_input"].to_numpy()
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DDDDDD", lw=0.5)
        for i in range(n):
            if abs(raw[i] - reduced[i]) > 0.015:
                ax.annotate("", xy=(raw[i], i), xytext=(reduced[i], i),
                            arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.8,
                                            shrinkA=3.5, shrinkB=3.5, mutation_scale=7), zorder=1)
            else:
                ax.plot([reduced[i], raw[i]], [i, i], color="#777777", lw=0.8, zorder=1)
            ax.text(max(reduced[i], raw[i]) + 0.035, i, (f"{raw[i] - reduced[i]:+.2f}" if abs(raw[i] - reduced[i]) >= 0.005 else "0.00"),
                    fontsize=5.8, color="#555555", va="center", ha="left", zorder=2)
        ax.scatter(reduced, y, s=20, color=reduced_c, zorder=3, label="reduced table (reference)")
        ax.scatter(raw, y, s=20, facecolors="white", edgecolors=raw_c, linewidths=0.9,
                   zorder=3, label="raw export (read by the released runner)")
        ax.set_yticks(y)
        ax.set_yticklabels([display_method(m) for m in frame["method"]], fontsize=7)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xlim(-0.02, 1.12)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Accuracy (Acc@1)", labelpad=1.5)
        ax.set_title(title, fontsize=8, loc="left", pad=3, fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6.5, length=2, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=6.5,
               ncol=2, bbox_to_anchor=(0.55, -0.02), handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = save(fig, "fig3_input_dumbbell.pdf")
    print(f"fig3_input_dumbbell.pdf -> {path}")
    print(f"  methods={n} order (by RE1-SS reduced-table Acc@1):")
    for method, val in zip(order, ss["acc_correct_input"]):
        print(f"    {method} {val:.3f}")


def main() -> None:
    make_fig1()
    make_fig2()
    make_fig3()


if __name__ == "__main__":
    main()
