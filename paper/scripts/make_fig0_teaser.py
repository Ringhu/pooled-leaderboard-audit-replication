#!/usr/bin/env python3
"""Build the teaser figure (Figure 1) for the RCA benchmark audit paper.

Every number drawn or printed is read from paper/data/ CSVs at run time; none
is hard-coded. See .research/plans/2026-09-30-teaser-figure.md for the design
and paper/scripts/make_figures.py for the shared style this reuses (rcParams,
FULL_W, colour palette, marker conventions, display-name mapping).

Run from the repo root: python3 paper/scripts/make_fig0_teaser.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (  # noqa: E402
    DATA,
    FULL_W,
    GREY_BAND,
    HIGHLIGHT,
    NEUTRAL,
    OI_BLACK,
    OI_BLUE,
    OI_VERMILLION,
    OUT,
    apply_style,
)

ROOT = Path(__file__).resolve().parents[2]


def marker_style(pooled: float, delta: float, lo: float, hi: float):
    """Same opposite-sign / CI-excludes-0 convention as make_figures.py."""
    opposite = (delta > 0 and pooled < 0) or (delta < 0 and pooled > 0)
    if pooled == 0 or delta == 0:
        opposite = False
    excludes_0 = lo > 0 or hi < 0
    colour = HIGHLIGHT if opposite else NEUTRAL
    filled = opposite and excludes_0
    return colour, filled, opposite, excludes_0


def sub_key_to_phrase(key: str) -> str:
    """'petshop::high_traffic' -> 'High traffic'."""
    name = key.split("::")[-1]
    parts = name.split("_")
    return " ".join([parts[0].capitalize()] + parts[1:])


def load_pair1() -> dict:
    """BARO - CIRCA, RCAEval RE1 (holds on every subsystem). Same A-B order as
    the card in panel (a), so the points sit on the BARO side of zero."""
    margin = pd.read_csv(DATA / "rq1" / "margin_dispersion.csv")
    row = margin[
        (margin["family"] == "RE1")
        & (margin["layer"] == "L1")
        & (margin["a"] == "baro")
        & (margin["b"] == "circa")
    ]
    if len(row) != 1:
        raise ValueError("expected exactly one RE1 baro-circa L1 margin row")
    row = row.iloc[0]
    pooled = float(row["pooled_se"])  # BARO - CIRCA, as stored
    sd = float(row["sd_sub_delta"])
    k = int(row["k"])
    margin_lt_sd = bool(row["margin_lt_sd"])

    deltas = pd.read_csv(DATA / "rq1" / "pair_subsystem_deltas.csv")
    g = deltas[
        (deltas["family"] == "RE1")
        & (deltas["layer"] == "L1")
        & (deltas["variant"] == "mean")
        & (deltas["a"] == "baro")
        & (deltas["b"] == "circa")
    ]
    sub_order = ["RE1::OB", "RE1::SS", "RE1::TT"]
    sub_labels = ["OB", "SS", "TT"]
    rows = []
    for key, label in zip(sub_order, sub_labels):
        r = g[g["sub"] == key]
        if len(r) != 1:
            raise ValueError(f"expected one pair_subsystem_deltas row for {key}")
        r = r.iloc[0]
        rows.append(
            {
                "key": key,
                "label": label,
                "delta": float(r["delta"]),
                "lo": float(r["lo"]),
                "hi": float(r["hi"]),
            }
        )
    return {
        "name_a": "BARO",
        "name_b": "CIRCA",
        "family": "RCAEval RE1",
        "pooled": pooled,
        "sd": sd,
        "k": k,
        "margin_lt_sd": margin_lt_sd,
        "rows": rows,
        "group_title": "BARO vs CIRCA, RCAEval RE1",
    }


def load_pair2() -> dict:
    """RCD - BARO, PetShop. CSV stores a=baro,b=rcd (pooled -0.135); we flip
    sign (and swap lo/hi on each subsystem) so the pooled leader (RCD) is
    reported first, per the approved design."""
    margin = pd.read_csv(DATA / "rq1" / "margin_dispersion.csv")
    row = margin[
        (margin["family"] == "petshop")
        & (margin["layer"] == "L1")
        & (margin["a"] == "baro")
        & (margin["b"] == "rcd")
    ]
    if len(row) != 1:
        raise ValueError("expected exactly one petshop baro-rcd L1 margin row")
    row = row.iloc[0]
    pooled = -float(row["pooled_se"])  # flip: RCD - BARO
    sd = float(row["sd_sub_delta"])
    k = int(row["k"])
    margin_lt_sd = bool(row["margin_lt_sd"])

    deltas = pd.read_csv(DATA / "rq1" / "pair_subsystem_deltas.csv")
    g = deltas[
        (deltas["family"] == "petshop")
        & (deltas["layer"] == "L1")
        & (deltas["variant"] == "mean")
        & (deltas["a"] == "baro")
        & (deltas["b"] == "rcd")
    ]
    sub_order = [
        "petshop::high_traffic",
        "petshop::low_traffic",
        "petshop::temporal_traffic1",
        "petshop::temporal_traffic2",
    ]
    sub_labels = ["High", "Low", "Tmp-1", "Tmp-2"]
    rows = []
    for key, label in zip(sub_order, sub_labels):
        r = g[g["sub"] == key]
        if len(r) != 1:
            raise ValueError(f"expected one pair_subsystem_deltas row for {key}")
        r = r.iloc[0]
        delta = -float(r["delta"])
        lo = -float(r["hi"])
        hi = -float(r["lo"])
        rows.append({"key": key, "label": label, "delta": delta, "lo": lo, "hi": hi})
    return {
        "name_a": "RCD",
        "name_b": "BARO",
        "family": "PetShop",
        "pooled": pooled,
        "sd": sd,
        "k": k,
        "margin_lt_sd": margin_lt_sd,
        "rows": rows,
        "group_title": "RCD vs BARO, PetShop",
    }


def load_panel_c() -> dict:
    """CIRCA - RCD, RE1-SS and RE1-TT, reduced (simple_data.csv, used in our
    analysis) vs raw (data.csv, read by the released runner)."""
    cc = pd.read_csv(DATA / "rq2" / "condition_comparisons.csv")
    g = cc[
        (cc["factor"] == "input_representation")
        & (cc["layer"] == "L1")
        & (cc["a"] == "circa")
        & (cc["b"] == "rcd")
        & (cc["sub"].isin(["RE1::SS", "RE1::TT"]))
    ]
    out = {}
    for sub_key, label in (("RE1::SS", "RE1-SS"), ("RE1::TT", "RE1-TT")):
        r = g[g["sub"] == sub_key]
        if len(r) != 1:
            raise ValueError(f"expected one condition_comparisons row for {sub_key}")
        r = r.iloc[0]
        out[label] = {
            "reduced": (float(r["delta_ref"]), float(r["lo_ref"]), float(r["hi_ref"])),
            "raw": (float(r["delta_alt"]), float(r["lo_alt"]), float(r["hi_alt"])),
        }
    return out


def draw_panel_a(ax, pair1: dict, pair2: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box_h = 0.36
    for y0, pair in ((0.56, pair1), (0.08, pair2)):
        box = FancyBboxPatch(
            (0.03, y0), 0.94, box_h,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            linewidth=0.6, edgecolor="#555555", facecolor="#F2F3F5",
            transform=ax.transAxes, zorder=1,
        )
        ax.add_patch(box)
        leader, other = ((pair["name_a"], pair["name_b"]) if pair["pooled"] > 0
                         else (pair["name_b"], pair["name_a"]))
        ax.text(0.5, y0 + 0.285, f"{leader} > {other}", ha="center", va="center",
                fontsize=6.5, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, y0 + 0.185, pair["family"], ha="center", va="center",
                fontsize=5.8, color="#444444", transform=ax.transAxes)
        ax.text(0.5, y0 + 0.08, f"{abs(pair['pooled']):+.3f}", ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.03, "pooled difference,\n$k$ subsystems averaged", ha="center", va="top",
            linespacing=1.1,
            fontsize=5.5, color="#666666", style="italic", transform=ax.transAxes)


def style_axis(ax):
    ax.tick_params(axis="y", length=0, pad=2, labelsize=6)
    ax.tick_params(axis="x", labelsize=5.5, pad=1, length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def direction_labels(ax, left_name: str, right_name: str, y: float) -> None:
    """Name the method each side of zero favours, so the sign needs no decoding."""
    ax.text(-0.012, y, f"\u2190 {left_name} better", ha="right", va="center",
            fontsize=5.4, color="#555555", transform=ax.transData)
    ax.text(0.012, y, f"{right_name} better \u2192", ha="left", va="center",
            fontsize=5.4, color="#555555", transform=ax.transData)


def draw_group(ax, pair: dict, xlim) -> None:
    rows = pair["rows"]
    n = len(rows)
    ax.axvline(0, color="#777777", ls="--", lw=0.6, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(n - 0.5, -1.3)
    direction_labels(ax, pair["name_b"], pair["name_a"], -0.95)
    ax.plot([pair["pooled"]] * 2, [-0.45, n - 0.55], color=OI_BLACK, lw=1.0, zorder=3,
            solid_capstyle="butt")
    for i, r in enumerate(rows):
        colour, filled, _, _ = marker_style(pair["pooled"], r["delta"], r["lo"], r["hi"])
        lo, hi = max(r["lo"], xlim[0]), min(r["hi"], xlim[1])
        ax.plot([lo, hi], [i, i], color=colour, lw=0.9, zorder=2, solid_capstyle="butt")
        if r["hi"] > xlim[1]:  # whisker clipped at the axis edge
            ax.plot(xlim[1], i, marker=">", ms=2.5, color=colour, zorder=2, clip_on=False)
        ax.plot(r["delta"], i, marker="o", ms=3.6, color=colour,
                markerfacecolor=colour if filled else "white", markeredgewidth=0.8,
                linestyle="none", zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows])
    style_axis(ax)
    # Subtitle line 2 is derived from the data: which subsystems are opposite
    # to the pooled sign with a CI that excludes 0.
    flags = [marker_style(pair["pooled"], r["delta"], r["lo"], r["hi"]) for r in rows]
    reversed_rows = [r for r, f in zip(rows, flags) if f[2] and f[3]]
    if reversed_rows:
        names = " and ".join(sub_key_to_phrase(r["key"]) for r in reversed_rows)
        sub_text, sub_colour = f"reversed on {names}, CI excludes 0", HIGHLIGHT
    else:
        assert all(not f[2] for f in flags) and len(rows) == 3
        sub_text, sub_colour = "same sign on all three applications", "#333333"
    # Title and two subtitle lines, spaced in points so both groups look alike (2026-10-03): the second line gives
    # the number the leaderboard reports (pooled margin) next to the one it withholds (between-subsystem spread).
    ax.set_title(pair["group_title"], fontsize=6.3, loc="left", pad=18)
    rel = "<" if pair["margin_lt_sd"] else ">"
    assert (abs(pair["pooled"]) < pair["sd"]) == pair["margin_lt_sd"]
    for dy, text, colour in ((10.5, sub_text, sub_colour),
                             (2.5, f"margin {abs(pair['pooled']):.3f} {rel} spread {pair['sd']:.3f}", "#555555")):
        ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                    ha="left", va="bottom", fontsize=5.6, color=colour)


def draw_panel_b(fig, cell, pair1: dict, pair2: dict):
    sub_gs = cell.subgridspec(2, 1, height_ratios=[3, 4], hspace=1.25)
    ax1 = fig.add_subplot(sub_gs[0, 0])
    ax2 = fig.add_subplot(sub_gs[1, 0], sharex=ax1)
    lo = min(r["lo"] for p in (pair1, pair2) for r in p["rows"])
    xlim = (min(-0.35, lo - 0.03), 0.6)
    draw_group(ax1, pair1, xlim)
    draw_group(ax2, pair2, xlim)
    ax2.set_xlabel("accuracy difference per subsystem, 95% CI", fontsize=6.3, labelpad=2)
    return ax1, ax2


def draw_panel_c(fig, cell, panel_c: dict):
    sub_gs = cell.subgridspec(2, 1, height_ratios=[3, 4], hspace=1.25)
    ax = fig.add_subplot(sub_gs[:, 0])
    rows, labels = [], []
    for app, key in (("Sock Shop", "RE1-SS"), ("Train Ticket", "RE1-TT")):
        rows.append((app, "reduced table", panel_c[key]["reduced"], NEUTRAL))
        rows.append((app, "raw export", panel_c[key]["raw"], HIGHLIGHT))
    ypos = [0, 1, 2.6, 3.6]
    ax.axvline(0, color="#777777", ls="--", lw=0.6, zorder=1)
    for y, (_, _, (d, lo, hi), colour) in zip(ypos, rows):
        ax.plot([lo, hi], [y, y], color=colour, lw=1.0, zorder=2, solid_capstyle="butt")
        ax.plot(d, y, marker="o", ms=3.8, color=colour, markeredgewidth=0.8, linestyle="none", zorder=4)
    for y0 in (0, 2.6):
        (d0, _, _), (d1, _, _) = rows[ypos.index(y0)][2], rows[ypos.index(y0) + 1][2]
        ax.annotate("", xy=(d1, y0 + 0.78), xytext=(d0, y0 + 0.22),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.6,
                                    connectionstyle="arc3,rad=-0.25"), zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[1] for r in rows])
    for y, app in ((0.5, "Sock Shop"), (3.1, "Train Ticket")):
        ax.text(-0.02, y - 1.05, app, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6, fontweight="bold")
    ax.set_ylim(4.1, -1.1)
    direction_labels(ax, "RCD", "CIRCA", -0.8)
    ax.set_xlim(-0.25, 0.5)
    style_axis(ax)
    ax.set_xlabel("accuracy difference, RE1, 95% CI", fontsize=6.3, labelpad=2)
    ax.set_title("CIRCA vs RCD, RCAEval RE1", fontsize=6.3, loc="left", pad=9)
    ax.text(0.0, 1.02, "same cases, two input files", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=5.6, color="#333333")
    return ax


def main() -> None:
    apply_style()
    pair1 = load_pair1()
    pair2 = load_pair2()
    panel_c = load_panel_c()

    fig = plt.figure(figsize=(FULL_W, 2.45))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.76, bottom=0.20)
    outer = fig.add_gridspec(1, 3, width_ratios=[0.62, 1.25, 1.0], wspace=0.42)

    ax_a = fig.add_subplot(outer[0, 0])
    draw_panel_a(ax_a, pair1, pair2)
    ax_b1, _ = draw_panel_b(fig, outer[0, 1], pair1, pair2)
    ax_c = draw_panel_c(fig, outer[0, 2], panel_c)

    label_y = 0.90
    for x0, text in (
        (ax_a.get_position().x0, "(a) Leaderboard:\none number per method"),
        (ax_b1.get_position().x0 - 0.02, "(b) Premise 1:\non which systems does it hold?"),
        (ax_c.get_position().x0 - 0.11, "(c) Premise 2:\nhow was the score produced?"),
    ):
        fig.text(x0, label_y, text, fontsize=7, fontweight="bold", ha="left", va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "fig0_teaser.pdf"
    png_path = OUT / "fig0_teaser_preview.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print()
    print("panel (a)/(b) pair 1: BARO - CIRCA, RCAEval RE1 (as stored in CSV)")
    print(f"  pooled = {pair1['pooled']:+.3f}  SD = {pair1['sd']:.3f}  k = {pair1['k']}"
          f"  margin_lt_sd = {pair1['margin_lt_sd']}")
    for r in pair1["rows"]:
        print(f"    {r['label']:>3}  delta={r['delta']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    print()
    print("panel (a)/(b) pair 2: RCD - BARO, PetShop (sign-flipped from CSV's baro-rcd row)")
    print(f"  pooled = {pair2['pooled']:+.3f}  SD = {pair2['sd']:.3f}  k = {pair2['k']}"
          f"  margin_lt_sd = {pair2['margin_lt_sd']}")
    for r in pair2["rows"]:
        print(f"    {r['label']:>10}  delta={r['delta']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    print()
    print("panel (c): CIRCA - RCD, reduced vs raw input")
    for label, d in panel_c.items():
        rd = d["reduced"]
        rw = d["raw"]
        print(f"    {label}  reduced={rd[0]:+.3f} [{rd[1]:+.3f}, {rd[2]:+.3f}]"
              f"   raw={rw[0]:+.3f} [{rw[1]:+.3f}, {rw[2]:+.3f}]")


if __name__ == "__main__":
    main()
