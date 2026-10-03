#!/usr/bin/env python3
"""Forest plot for the System Matters BARO vs z_family heterogeneity analysis.

One row per system, grouped by benchmark family (OpenRCA / RCAEval / PetShop).
X-axis = Δ_s in percentage points (BARO − z_family).
  - Flip systems (Δ > 0, BARO wins) use filled square markers.
  - Non-flip systems (Δ ≤ 0, z_family wins) use filled circle markers.
  - 95% paired-bootstrap CIs as horizontal whiskers.

Bottom panel:
  - Fixed-effect diamond (Δ_FE ± 1.96 · SE_FE).
  - Random-effects diamond (HKSJ 95% CI on Δ_RE).
  - 95% prediction interval as a separate bar.

Reads: results_stats/heterogeneity/per_system_deltas.csv
       results_stats/heterogeneity/meta_analysis_summary.json
Writes: results_stats/heterogeneity/forest_plot.pdf + .png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCH_ORDER = ["openrca", "rcaeval", "petshop"]
BENCH_LABEL = {
    "openrca": "OpenRCA",
    "rcaeval": "RCAEval",
    "petshop": "PetShop",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir

    per_sys = pd.read_csv(args.input_dir / "per_system_deltas.csv")
    with open(args.input_dir / "meta_analysis_summary.json") as f:
        meta = json.load(f)
    m = meta["meta_pp"]
    I2_pct = meta["meta"]["I2"] * 100
    Q = meta["meta"]["Q"]
    p_Q = meta["meta"]["p_Q"]
    k = meta["meta"]["k"]
    tau_pp = m["tau_pp"]
    delta_FE = m["delta_FE_pp"]
    se_FE = m["se_FE_pp"]
    delta_RE = m["delta_RE_pp"]
    re_lo = m["re_ci_lo_hksj_pp"]
    re_hi = m["re_ci_hi_hksj_pp"]
    pi_lo = m["pi_lo_pp"]
    pi_hi = m["pi_hi_pp"]

    # Organise row order: by benchmark family, then by delta within family.
    per_sys = per_sys.copy()
    per_sys["bench_order"] = per_sys["benchmark"].map(
        {b: i for i, b in enumerate(BENCH_ORDER)}
    )
    per_sys = per_sys.sort_values(["bench_order", "delta_pp"])
    per_sys = per_sys.reset_index(drop=True)

    # Layout: one "row" per system + 3 spacer rows (bench labels) + 4 summary rows
    # We lay out using y-coordinates from top (high y) to bottom (low y=0).
    y_positions = []
    row_info = []
    y = 0.0
    # Summary rows at the bottom
    y_pi = y; y += 1.0
    y_re = y; y += 1.0
    y_fe = y; y += 1.0
    y += 1.0  # spacer above summary block
    # System rows, grouped with labels
    current_bench = None
    bench_label_positions = {}
    # Iterate in REVERSE so systems appear in top-to-bottom = family-order order.
    for _, row in per_sys[::-1].iterrows():
        bench = row["benchmark"]
        if current_bench is None or bench != current_bench:
            if current_bench is not None:
                y += 0.8  # gap between families
            current_bench = bench
            bench_label_positions[bench] = None  # set after we know min/max y
        y_positions.append(y)
        row_info.append(row)
        y += 1.0
    # Compute bench label y as midpoint of family's rows.
    bench_ys = {b: [] for b in BENCH_ORDER}
    for yp, info in zip(y_positions, row_info):
        bench_ys[info["benchmark"]].append(yp)
    bench_label_y = {b: float(np.mean(ys)) for b, ys in bench_ys.items() if ys}

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Determine x-axis limits
    x_min = min(per_sys["ci_lo_pp"].min(), pi_lo, re_lo) - 3
    x_max = max(per_sys["ci_hi_pp"].max(), pi_hi, re_hi) + 3

    # Vertical reference line at 0
    ax.axvline(0.0, color="#888888", lw=0.8, zorder=1)

    # Plot per-system markers + CIs
    for yp, info in zip(y_positions, row_info):
        is_flip = bool(info["flip_flag"])
        marker = "s" if is_flip else "o"
        color = "#C44E52" if is_flip else "#4C72B0"
        # CI whisker
        ax.hlines(yp, info["ci_lo_pp"], info["ci_hi_pp"],
                  color=color, lw=1.4, zorder=2)
        # Point
        ax.plot(info["delta_pp"], yp, marker=marker, markersize=8,
                color=color, zorder=3, markeredgecolor="white",
                markeredgewidth=0.7)
        # Row label (left)
        short = info["system"].split("::", 1)[1]
        ax.text(x_min, yp, short, ha="left", va="center", fontsize=9)
        # n and Δ (right)
        ax.text(
            x_max, yp,
            f"Δ = {info['delta_pp']:+5.1f}pp  [{info['ci_lo_pp']:+5.1f}, {info['ci_hi_pp']:+5.1f}]  n={int(info['n_cases'])}",
            ha="right", va="center", fontsize=8, family="monospace",
        )

    # Benchmark family labels (small, left of system labels)
    for bench, by in bench_label_y.items():
        ax.text(x_min - (x_max - x_min) * 0.12, by, BENCH_LABEL[bench],
                ha="left", va="center", fontsize=10, fontweight="bold",
                color="#444444")

    # --- Summary block: FE diamond --- #
    def draw_diamond(center_y, x_lo, x_hi, color, label_left):
        x_mid = (x_lo + x_hi) / 2.0
        xs = [x_lo, x_mid, x_hi, x_mid]
        ys = [center_y, center_y + 0.22, center_y, center_y - 0.22]
        ax.fill(xs, ys, color=color, edgecolor="black", linewidth=0.7, zorder=3)
        ax.text(x_min, center_y, label_left, ha="left", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x_max, center_y,
                f"[{x_lo:+5.1f}, {x_hi:+5.1f}]",
                ha="right", va="center", fontsize=8, family="monospace")

    fe_lo = delta_FE - 1.96 * se_FE
    fe_hi = delta_FE + 1.96 * se_FE
    draw_diamond(y_fe, fe_lo, fe_hi, "#4C72B0",
                 f"FE pooled (Δ={delta_FE:+.2f} pp)")
    draw_diamond(y_re, re_lo, re_hi, "#55A868",
                 f"RE pooled (Δ={delta_RE:+.2f} pp, HKSJ 95% CI)")
    # --- PI bar --- #
    ax.hlines(y_pi, pi_lo, pi_hi, color="#8172B2", lw=5, zorder=3)
    ax.plot(delta_RE, y_pi, "D", markersize=6, color="#8172B2",
            markeredgecolor="black", markeredgewidth=0.5, zorder=4)
    ax.text(x_min, y_pi, "95% Prediction Interval",
            ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(x_max, y_pi, f"[{pi_lo:+5.1f}, {pi_hi:+5.1f}]",
            ha="right", va="center", fontsize=8, family="monospace")

    # Y-axis tidying
    ax.set_ylim(-1.0, max(y_positions) + 1.2)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("Δ = BARO acc@1 − z_family acc@1  (percentage points)",
                  fontsize=10)
    ax.set_xlim(x_min - (x_max - x_min) * 0.15, x_max)

    # Top annotation: Q / I² / τ²
    title = (f"System-level heterogeneity: BARO vs z_family  "
             f"(k={k}, Q={Q:.1f}, p$_Q$={p_Q:.1e}, "
             f"I²={I2_pct:.1f}%, τ={tau_pp:.2f} pp)")
    ax.set_title(title, fontsize=10.5, pad=12)

    # Clean up spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out_pdf = args.output_dir / "forest_plot.pdf"
    out_png = args.output_dir / "forest_plot.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[written] {out_pdf}")
    print(f"[written] {out_png}")


if __name__ == "__main__":
    main()
