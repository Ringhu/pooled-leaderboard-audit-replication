#!/usr/bin/env python3
"""Check: is the true root-cause column in the 35 abnormal-only columns?"""
import sys
from pathlib import Path
import pandas as pd

PETSHOP = Path("$HOME/petshop_repo")
sys.path.insert(0, str(PETSHOP / "code"))
from rca_task import load_scenario


def clean(s):
    return str(s).replace(" ", "_").replace("::", "-").replace("/", "-")


scenario = PETSHOP / "dataset" / "high_traffic"
graph, normal, issues = load_scenario(str(scenario))


def flat(df):
    return ["__".join(clean(x) for x in col) for col in df.columns]


normal_cols = set(flat(normal))

missing_per_case = []
for i, (abn, tgt) in enumerate(issues["test"]):
    abn_cols = set(flat(abn))
    only_abn = abn_cols - normal_cols
    only_norm = normal_cols - abn_cols
    true_root = clean(tgt["root_cause"]["node"])
    # Does true root appear in normal's node set?
    normal_nodes = {c.split("__")[0] for c in normal_cols}
    abn_nodes = {c.split("__")[0] for c in abn_cols}
    missing_per_case.append({
        "case": i,
        "true_root": true_root,
        "root_in_normal_nodes": true_root in normal_nodes,
        "root_in_abn_nodes": true_root in abn_nodes,
        "only_abn_cols": len(only_abn),
        "only_norm_cols": len(only_norm),
    })

df = pd.DataFrame(missing_per_case)
print(df.to_string(index=False))
print()
print("Summary:")
print(f"  cases where root IS in normal nodes: {df['root_in_normal_nodes'].sum()}/{len(df)}")
print(f"  cases where root IS in abnormal nodes: {df['root_in_abn_nodes'].sum()}/{len(df)}")
print(f"  mean abn-only cols: {df['only_abn_cols'].mean():.1f}")
print(f"  mean norm-only cols: {df['only_norm_cols'].mean():.1f}")
