#!/usr/bin/env python3
"""Plot Figure 1 (tolerance-response curves) and Figure 2 (pairwise Δ heatmap)
from the bootstrap analysis outputs.

Inputs:
  --summary-per-system   summary_per_system.csv from bootstrap_analysis.py
  --pairwise             pairwise_deltas.csv from bootstrap_analysis.py
  --metric               'correct_pct' (strict) or 'partial_pct' (partial)

Outputs (PNG + PDF):
  fig1_tolerance_curves.{png,pdf}
  fig2_pairwise_heatmap.{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABEL = {
    "cd5min":   "CD-detector-5min",
    "cd1min":   "CD-detector-1min",
    "baro":     "BARO",
    "causal":   "Causal-guided",
    "zbase":    "Z-baseline",
    "circa":    "CIRCA",
    "rcaagent": "RCA-Agent",
}
METHOD_COLOR = {
    "cd5min":   "#999999",
    "cd1min":   "#1f77b4",
    "baro":     "#2ca02c",
    "causal":   "#9467bd",
    "zbase":    "#8c564b",
    "circa":    "#d62728",
    "rcaagent": "#ff7f0e",
}
MARKER = {
    "cd5min": "v", "cd1min": "o", "baro": "s",
    "causal": "D", "zbase": "^", "circa": "P", "rcaagent": "X",
}


def plot_fig1(summary_per_sys, out_dir, metric_col="correct_pct"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    systems = ["Bank", "Mkt-cb1", "Mkt-cb2", "Telecom"]
    for ax, sys in zip(axes.flat, systems):
        sub = summary_per_sys[summary_per_sys.system == sys]
        for m, g in sub.groupby("method", sort=False):
            g = g.sort_values("tolerance_s")
            ax.plot(g.tolerance_s, g[metric_col],
                    marker=MARKER.get(m, "o"),
                    color=METHOD_COLOR.get(m, "black"),
                    label=METHOD_LABEL.get(m, m),
                    linewidth=1.8, markersize=6)
        ax.set_xscale("log")
        ax.set_xticks([30, 60, 120, 300, 600])
        ax.set_xticklabels(["30s", "60s", "120s", "300s", "600s"])
        ax.set_title(sys, fontsize=11)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time tolerance $T_{win}$")
    y_label = "Correct %" if metric_col == "correct_pct" else "Partial %"
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(labels), bbox_to_anchor=(0.5, 1.02),
               frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_tolerance_curves.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig1_tolerance_curves.png")


def plot_fig2(pairwise_df, out_dir):
    """Heatmap: one row per (A,B) pair, one col per tolerance.
    Cell = Δ (Correct%(A) - Correct%(B)) ; red = A wins, blue = B wins.
    Hatched cells = bootstrap-significant.
    """
    tols = sorted(pairwise_df.tolerance_s.unique())
    pairs = list(pairwise_df[["method_A", "method_B"]].drop_duplicates()
                 .itertuples(index=False, name=None))
    n_pairs = len(pairs)
    mat = np.zeros((n_pairs, len(tols)))
    sig = np.zeros((n_pairs, len(tols)), dtype=bool)
    labels = []
    for i, (A, B) in enumerate(pairs):
        labels.append(f"{METHOD_LABEL.get(A, A)} vs {METHOD_LABEL.get(B, B)}")
        for j, t in enumerate(tols):
            r = pairwise_df[(pairwise_df.method_A == A)
                             & (pairwise_df.method_B == B)
                             & (pairwise_df.tolerance_s == t)]
            if len(r):
                mat[i, j] = r.delta.iloc[0]
                sig[i, j] = bool(r.sig.iloc[0])

    vmax = max(8, np.abs(mat).max())
    fig_h = 0.35 * n_pairs + 1.5
    fig, ax = plt.subplots(figsize=(7, fig_h))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(tols)))
    ax.set_xticklabels([f"±{t}s" for t in tols])
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n_pairs):
        for j in range(len(tols)):
            val = mat[i, j]
            txt = f"{val:+.1f}"
            color = "white" if abs(val) > 0.6 * vmax else "black"
            weight = "bold" if sig[i, j] else "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=color, weight=weight)
            if sig[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="black", linewidth=1.3
                ))
    ax.set_xlabel("Time tolerance $T_{win}$")
    ax.set_title("Pairwise Δ = Correct%(A) - Correct%(B), pooled\n"
                 "(boxed cells: bootstrap 95% CI excludes 0; "
                 "bold = same)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cb.set_label("Δ (percentage points)")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_pairwise_heatmap.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig2_pairwise_heatmap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-per-system", required=True)
    ap.add_argument("--pairwise", required=True)
    ap.add_argument("--metric", choices=["correct_pct", "partial_pct"],
                    default="correct_pct")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sps = pd.read_csv(args.summary_per_system)
    pw = pd.read_csv(args.pairwise)

    # If partial_pct not present, compute from score mean in per-system?
    if args.metric == "partial_pct" and "partial_pct" not in sps.columns:
        print("WARN: partial_pct not available in summary_per_system.csv; "
              "falling back to correct_pct. Re-run bootstrap_analysis to "
              "emit partial_pct if needed.")
        args.metric = "correct_pct"

    plot_fig1(sps, args.out_dir, metric_col=args.metric)
    plot_fig2(pw, args.out_dir)


if __name__ == "__main__":
    main()
