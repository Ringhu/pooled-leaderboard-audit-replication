#!/usr/bin/env python3
"""Compute causal-graph-distance metric for PetShop predictions.

Given:
  - `graph.csv`  — per-traffic-pattern adjacency matrix
  - `petshop_<traffic>_results.csv` (or *_circa.csv) — per-case top1 prediction
    plus true_service label

Compute for each case:
  dag_dist_top1 = undirected shortest-path length from top1 to true_service
                  in the scenario's true causal graph. Unreachable → max_dist + 1.

Also emit a summary (method × system × mean dag_dist) so downstream LOSO-style
analysis can check whether ranking by `dag_dist` differs from ranking by `acc@1`.
"""
import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


PETSHOP = Path("$HOME/petshop_repo")
TRAFFIC_DIRS = {
    "high_traffic":      PETSHOP / "dataset" / "high_traffic",
    "low_traffic":       PETSHOP / "dataset" / "low_traffic",
    "temporal_traffic1": PETSHOP / "dataset" / "temporal_traffic1",
    "temporal_traffic2": PETSHOP / "dataset" / "temporal_traffic2",
}


def clean(s):
    return str(s).replace(" ", "_").replace("::", "-").replace("/", "-")


def load_graph(scenario_dir: Path):
    adj = pd.read_csv(scenario_dir / "graph.csv", index_col=0)
    # Column/row labels come from PetShop raw naming; apply same clean() as
    # run_petshop.py so node names line up with predicted top1 / true_service.
    adj.index = [clean(x) for x in adj.index]
    adj.columns = [clean(x) for x in adj.columns]
    g = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)
    return g.to_undirected()  # distance is symmetric for our purposes


def graph_dist(g, a, b, fallback):
    if a not in g.nodes or b not in g.nodes:
        return fallback
    if a == b:
        return 0
    try:
        return nx.shortest_path_length(g, a, b)
    except nx.NetworkXNoPath:
        return fallback


def rescore(results_csv: Path, graph_file_scenario: Path, traffic: str):
    df = pd.read_csv(results_csv)
    df = df[df["top1"].notna() & (df["top1"].astype(str) != "")].copy()
    g = load_graph(graph_file_scenario)
    max_diam = 0
    try:
        max_diam = nx.diameter(g)
    except nx.NetworkXError:
        # disconnected graph — diameter is per-component; use max+1 as fallback
        max_diam = max((nx.diameter(g.subgraph(cc))
                        for cc in nx.connected_components(g)),
                       default=1)
    fallback = max_diam + 1

    df["dag_dist_top1"] = df.apply(
        lambda r: graph_dist(g, r["top1"], r["true_service"], fallback),
        axis=1,
    )
    df["dag_in_graph"] = df["true_service"].apply(lambda s: s in g.nodes)
    df["traffic"] = traffic
    return df, fallback, g.number_of_nodes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Directory with petshop_<traffic>_results.csv files")
    ap.add_argument("--method-tag", default="",
                    help="Optional suffix added to output filenames (e.g., "
                         "'circa' for results_petshop_circa).")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for traffic, scenario_dir in TRAFFIC_DIRS.items():
        res = args.in_dir / f"petshop_{traffic}_results.csv"
        if not res.exists():
            print(f"[skip] {res} missing")
            continue
        df, fallback, n_nodes = rescore(res, scenario_dir, traffic)
        print(f"{traffic}: n={len(df)}, graph={n_nodes} nodes, fallback={fallback}, "
              f"mean dag_dist_top1={df['dag_dist_top1'].mean():.2f}, "
              f"coverage={df['dag_in_graph'].mean():.2%}")
        all_rows.append(df)

    if not all_rows:
        raise SystemExit("No results found")
    big = pd.concat(all_rows, ignore_index=True)
    tag = f"_{args.method_tag}" if args.method_tag else ""
    per_case_out = args.out_dir / f"petshop_dag_distance_per_case{tag}.csv"
    big.to_csv(per_case_out, index=False)
    print(f"\nWrote {per_case_out} ({len(big)} rows)")

    # Summary: method × system × mean_dag_dist + acc@1 (for comparison)
    summ = big.groupby(["method", "traffic"]).agg(
        n=("dag_dist_top1", "size"),
        mean_dag_dist=("dag_dist_top1", "mean"),
        mean_acc1=("acc@1", "mean"),
    ).reset_index()
    summ_out = args.out_dir / f"petshop_dag_distance_summary{tag}.csv"
    summ.to_csv(summ_out, index=False)
    print(summ.to_string(index=False))

    # Pooled across all traffic
    pooled = big.groupby("method").agg(
        n=("dag_dist_top1", "size"),
        mean_dag_dist=("dag_dist_top1", "mean"),
        mean_acc1=("acc@1", "mean"),
    ).reset_index()
    print("\n=== Pooled across all traffic patterns ===")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
