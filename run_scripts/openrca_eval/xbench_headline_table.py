#!/usr/bin/env python3
"""Produce the headline 11-system cross-benchmark table and BARO-only
deep-dive (BARO is the only method that has data across all 11 systems)."""
import pandas as pd
from pathlib import Path

IN = Path("$HOME/auditstack/results_xbench_11sys/view_accuracy.csv")
OUT = Path("$HOME/auditstack/results_xbench_11sys")

df = pd.read_csv(IN)
df = df[df["tdelta"] == 0]  # only tdelta=0 for all OpenRCA; RCAEval/PetShop also 0

# BARO-only: per-system acc@1 min/max spread and LOSO deltas
baro = df[df["method"] == "baro"].copy()
only = baro[baro["view"].str.startswith("only-")].sort_values("acc1")
print("=== BARO per-system acc@1 (sorted, 11 systems) ===")
print(only[["view", "n", "acc1"]].to_string(index=False))
print(f"\nmin={only['acc1'].min():.3f}, max={only['acc1'].max():.3f}, "
      f"ratio={only['acc1'].max() / max(only['acc1'].min(), 1e-6):.1f}×")

drop = baro[baro["view"].str.startswith("drop-")]
pooled = baro[baro["view"] == "pooled"]["acc1"].iloc[0]
print(f"\n=== BARO pooled acc@1 = {pooled:.3f} ===")
print("\n=== BARO LOSO (drop-one-system) spread ===")
print(drop[["view", "acc1"]].sort_values("acc1").to_string(index=False))
print(f"\nLOSO range: {drop['acc1'].min():.3f} .. {drop['acc1'].max():.3f}")

# Wide table: method × only-system
wide = (df[df["view"].str.startswith("only-")]
        .pivot_table(index="view", columns="method", values="acc1"))
wide.to_csv(OUT / "headline_per_system_acc1.csv")
print("\n=== Per-system × per-method acc@1 (11 systems × 5 methods) ===")
print(wide.round(3).to_string())

# OpenRCA-internal sign flip (F1) detection
oi = wide.loc[wide.index.str.startswith("only-openrca::")]
print("\n=== OpenRCA-internal ranking (F1 sign-flip witness) ===")
ranks = oi.rank(axis=1, ascending=False)
print(ranks.astype(int).to_string())

# Pairs where OpenRCA has sign flip across systems
print("\n=== Method-pair sign changes across OpenRCA systems ===")
for a in ["baro", "cd1min", "causal", "zbase"]:
    for b in ["baro", "cd1min", "causal", "zbase"]:
        if a >= b:
            continue
        diff = oi[a] - oi[b]
        if (diff > 1e-3).any() and (diff < -1e-3).any():
            print(f"  {a} vs {b}: flip — " +
                  ", ".join(f"{idx.split('::')[1]}={v:+.3f}"
                            for idx, v in diff.items()))
