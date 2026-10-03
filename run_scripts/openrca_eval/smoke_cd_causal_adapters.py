#!/usr/bin/env python3
"""Smoke-test cross-benchmark CD-1min / causal-style adapters.

This is a bounded feasibility test, not a paper-ready reproduction.

It checks whether the OpenRCA-style component-level detector can be adapted to
RCAEval and PetShop inputs:

* cd1min_adapt: rank components by the largest post-injection percentage shift
  among metrics that pass z-score and percentage thresholds.
* causal_adapt: use a component graph when one is available; otherwise record
  that the method falls back to cd1min_adapt-style ranking.

Outputs:
  per_case.csv
  summary.csv
  notes.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd


def default_root() -> Path:
    """Work both on the 3090 host and through the local SSHFS mount."""
    remote_root = Path("$HOME")
    if remote_root.exists():
        return remote_root
    return Path("$HOME/mnt/3090")


MOUNT_ROOT = default_root()
RCAEVAL_DATA_ROOT = MOUNT_ROOT / "rcaeval_repo/data"
_RCAEVAL_CANONICAL = {
    "online-boutique": RCAEVAL_DATA_ROOT / "online-boutique",
    "sock-shop-2": RCAEVAL_DATA_ROOT / "sock-shop-2",
    "train-ticket": RCAEVAL_DATA_ROOT / "train-ticket",
}
_RCAEVAL_RE2 = {
    "online-boutique": RCAEVAL_DATA_ROOT / "RE2/RE2-OB",
    "sock-shop-2": RCAEVAL_DATA_ROOT / "RE2/RE2-SS",
    "train-ticket": RCAEVAL_DATA_ROOT / "RE2/RE2-TT",
}
RCAEVAL_DATA_ROOTS = {
    name: path if path.exists() else _RCAEVAL_RE2[name]
    for name, path in _RCAEVAL_CANONICAL.items()
}
PETSHOP_ROOT = MOUNT_ROOT / "petshop_repo"
PETSHOP_TRAFFIC_ROOTS = {
    "high_traffic": PETSHOP_ROOT / "dataset/high_traffic",
    "low_traffic": PETSHOP_ROOT / "dataset/low_traffic",
    "temporal_traffic1": PETSHOP_ROOT / "dataset/temporal_traffic1",
    "temporal_traffic2": PETSHOP_ROOT / "dataset/temporal_traffic2",
}


AWS_SUFFIXES_TO_STRIP = [
    "-function", "-table", "-stage", "-queue", "-statemachine",
]


def clean(s) -> str:
    return str(s).replace(" ", "_").replace("::", "-").replace("/", "-")


def canonical_petshop_node(s) -> str:
    """A conservative alias layer for PetShop node labels.

    PetShop target labels sometimes use a more specific AWS resource type than
    metric labels, e.g. `..._AWS-Lambda-Function` vs `..._AWS-Lambda`. Exact
    matching remains the primary score; this canonical score is an audit column
    for label-normalization sensitivity.
    """
    out = clean(s).lower()
    changed = True
    while changed:
        changed = False
        for suffix in AWS_SUFFIXES_TO_STRIP:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def flatten_petshop_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["__".join(clean(x) for x in col) for col in df.columns]
    return out


def extract_rcaeval_component(metric_col: str) -> str:
    """Map RCAEval metric columns to service-like component names."""
    for suffix in [
        "_cpu", "_mem", "_latency", "_load", "_error", "_duration", "_rps",
        "_lat_90", "_latency-90", "_loss", "_disk", "_delay", "_workload",
    ]:
        if metric_col.endswith(suffix):
            return metric_col[:-len(suffix)]
    if "_" in metric_col:
        head = metric_col.split("_", 1)[0]
        if head and head[0].isdigit():
            return head
        for subsfx in ["-db", "-rb"]:
            if head.endswith(subsfx):
                return head[:-len(subsfx)]
        return head
    return metric_col


def extract_petshop_component(metric_col: str) -> str:
    """PetShop flat format: '<node>__<metric>__<statistic>'."""
    return metric_col.split("__", 1)[0] if "__" in metric_col else metric_col


def acc_at_k(ranks: list[str], true_component: str, k: int) -> float:
    seen = []
    for r in ranks:
        if r not in seen:
            seen.append(r)
        if len(seen) >= k:
            break
    return 1.0 if true_component in seen else 0.0


def acc_at_k_canonical(
    ranks: list[str],
    true_component: str,
    k: int,
    canonicalize: Callable[[str], str],
) -> float:
    true_c = canonicalize(true_component)
    seen = []
    for r in ranks:
        rc = canonicalize(r)
        if rc not in seen:
            seen.append(rc)
        if len(seen) >= k:
            break
    return 1.0 if true_c in seen else 0.0


def _numeric_metric_cols(data: pd.DataFrame) -> list[str]:
    return [
        c for c in data.columns
        if c != "time" and pd.api.types.is_numeric_dtype(data[c])
    ]


def detect_component_anomalies(
    data: pd.DataFrame,
    inject_time: float,
    component_of: Callable[[str], str],
    *,
    z_threshold: float = 3.0,
    pct_threshold: float = 5.0,
) -> tuple[dict, str | None]:
    if "time" not in data.columns:
        return {}, "missing_time_column"

    baseline = data[data["time"] < inject_time]
    anomaly = data[data["time"] >= inject_time]
    if len(baseline) < 10 or len(anomaly) < 1:
        return {}, f"bad_split: baseline={len(baseline)}, anomaly={len(anomaly)}"

    svc_info = defaultdict(lambda: {
        "anomalies": [],
        "max_z": 0.0,
        "max_pct": 0.0,
    })

    for col in _numeric_metric_cols(data):
        comp = component_of(col)
        if not comp:
            continue

        base = baseline[col].replace([np.inf, -np.inf], np.nan).dropna()
        anom = anomaly[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(base) < 10 or len(anom) == 0:
            continue

        mu = float(base.mean())
        sigma = float(base.std(ddof=1))
        if not math.isfinite(mu) or not math.isfinite(sigma) or sigma < 1e-8:
            continue

        w_max = float(anom.max())
        w_min = float(anom.min())
        peak = w_max if abs(w_max - mu) >= abs(w_min - mu) else w_min
        z = abs(peak - mu) / sigma
        pct = (peak - mu) / (abs(mu) + 1e-8) * 100.0

        if z < z_threshold or abs(pct) < pct_threshold:
            continue

        svc_info[comp]["anomalies"].append({
            "metric": col,
            "z": z,
            "pct": pct,
        })
        svc_info[comp]["max_z"] = max(svc_info[comp]["max_z"], z)
        svc_info[comp]["max_pct"] = max(svc_info[comp]["max_pct"], abs(pct))

    if not svc_info:
        return {}, "no_component_passed_thresholds"
    return dict(svc_info), None


def rank_cd1min(svc_info: dict) -> list[str]:
    ranking = sorted(
        svc_info.items(),
        key=lambda kv: (-kv[1]["max_pct"], -kv[1]["max_z"], kv[0]),
    )
    return [comp for comp, _ in ranking]


def normalized_graph(graph: nx.Graph | None) -> nx.DiGraph | None:
    if graph is None:
        return None
    out = nx.DiGraph()
    for n in graph.nodes:
        out.add_node(clean(n))
    for u, v in graph.edges:
        out.add_edge(clean(u), clean(v))
    return out


def maybe_reverse_graph(graph: nx.DiGraph | None, protocol: str) -> nx.DiGraph | None:
    if graph is None:
        return None
    if protocol == "none":
        return None
    if protocol in {"original", "callgraph-original"}:
        return graph
    if protocol in {"reverse", "callgraph-reverse"}:
        return graph.reverse(copy=True)
    raise ValueError(f"unknown graph protocol: {protocol}")


def bounded_ancestors(graph: nx.DiGraph, node: str, max_depth: int = 3) -> set[str]:
    seen = set()
    q = deque([(node, 0)])
    while q:
        cur, depth = q.popleft()
        if depth >= max_depth:
            continue
        for pred in graph.predecessors(cur):
            if pred not in seen:
                seen.add(pred)
                q.append((pred, depth + 1))
    seen.discard(node)
    return seen


def aggregate_component_series(
    data: pd.DataFrame,
    component_of: Callable[[str], str],
) -> pd.DataFrame:
    """Aggregate metric columns into one standardized series per component."""
    out = {}
    for col in _numeric_metric_cols(data):
        comp = component_of(col)
        if not comp:
            continue
        x = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 5:
            continue
        sigma = float(x.std(ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-8:
            continue
        z = ((x - float(x.mean())) / sigma).abs()
        out.setdefault(comp, []).append(z.to_numpy(dtype=float))

    agg = {}
    for comp, arrs in out.items():
        mat = np.vstack(arrs)
        agg[comp] = np.nanmax(mat, axis=0)
    return pd.DataFrame(agg)


def build_lagcorr_graph(
    data: pd.DataFrame,
    component_of: Callable[[str], str],
    *,
    threshold: float = 0.35,
    margin: float = 0.05,
) -> nx.DiGraph | None:
    """Build an explicit RCAEval component graph from lagged correlations.

    Edge u -> v when corr(u[t-1], v[t]) is positive, exceeds `threshold`, and
    is stronger than the reverse lag by at least `margin`.
    """
    comp_df = aggregate_component_series(data, component_of)
    if comp_df.shape[0] < 5 or comp_df.shape[1] < 2:
        return None

    comps = list(comp_df.columns)
    g = nx.DiGraph()
    g.add_nodes_from(comps)
    vals = comp_df.to_numpy(dtype=float)
    for i, u in enumerate(comps):
        for j, v in enumerate(comps):
            if i == j:
                continue
            x = vals[:-1, i]
            y = vals[1:, j]
            xr = vals[:-1, j]
            yr = vals[1:, i]
            if np.nanstd(x) < 1e-8 or np.nanstd(y) < 1e-8:
                continue
            c_uv = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(c_uv):
                continue
            if np.nanstd(xr) < 1e-8 or np.nanstd(yr) < 1e-8:
                c_vu = 0.0
            else:
                c_vu = float(np.corrcoef(xr, yr)[0, 1])
                if not np.isfinite(c_vu):
                    c_vu = 0.0
            if c_uv >= threshold and (c_uv - c_vu) >= margin:
                g.add_edge(u, v, weight=c_uv)
    return g if g.number_of_edges() > 0 else None


def rank_causal(svc_info: dict, graph: nx.DiGraph | None) -> tuple[list[str], bool]:
    if graph is None or len(svc_info) < 2:
        return rank_cd1min(svc_info), False

    anom_components = set(svc_info)
    scored = []
    for comp, info in svc_info.items():
        if comp in graph:
            upstream = bounded_ancestors(graph, comp, max_depth=3)
            upstream_anom = len(upstream & anom_components)
        else:
            upstream_anom = 0
        root_score = len(info.get("anomalies", [])) / (1.0 + upstream_anom)
        scored.append((root_score, info["max_pct"], info["max_z"], comp))

    scored.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))
    return [comp for _, _, _, comp in scored], True


def method_predict(
    method: str,
    data: pd.DataFrame,
    inject_time: float,
    component_of: Callable[[str], str],
    graph: nx.DiGraph | None = None,
    z_threshold: float = 3.0,
    pct_threshold: float = 5.0,
) -> dict:
    svc_info, error = detect_component_anomalies(
        data,
        inject_time,
        component_of,
        z_threshold=z_threshold,
        pct_threshold=pct_threshold,
    )
    if error is not None:
        return {
            "ranks": [],
            "error": error,
            "n_anom_components": 0,
            "graph_used": False,
        }

    if method == "cd1min_adapt":
        ranks = rank_cd1min(svc_info)
        graph_used = False
    elif method == "causal_adapt":
        ranks, graph_used = rank_causal(svc_info, graph)
    else:
        raise ValueError(f"unknown method: {method}")

    return {
        "ranks": ranks,
        "error": None,
        "n_anom_components": len(svc_info),
        "graph_used": graph_used,
    }


def choose_case_dirs(case_dirs: list[str], limit: int, strategy: str, seed: int) -> list[str]:
    if limit is None or limit <= 0 or len(case_dirs) <= limit:
        return case_dirs
    if strategy == "first":
        return case_dirs[:limit]
    if strategy != "stratified":
        raise ValueError(f"unknown sampling strategy: {strategy}")

    groups = defaultdict(list)
    for d in case_dirs:
        key = os.path.basename(os.path.dirname(d.rstrip("/")))
        groups[key].append(d)

    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    for key in keys:
        groups[key] = list(groups[key])
        rng.shuffle(groups[key])

    selected = []
    while len(selected) < limit:
        added = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


CD_DATA_FILE = "data.csv"


def load_rcaeval_case(data_dir: str) -> tuple[pd.DataFrame, int, str, str]:
    data_path = os.path.join(data_dir, CD_DATA_FILE)
    if not os.path.exists(data_path):
        data_path = os.path.join(data_dir, "data.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(data_dir, "simple_metrics.csv")
    data = pd.read_csv(data_path)
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    with open(os.path.join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readline().strip())

    dd = data_dir.rstrip("/")
    service_fault = os.path.basename(os.path.dirname(dd))
    service, fault = service_fault.rsplit("_", 1)
    return data, inject_time, service, fault


def window_rcaeval(data: pd.DataFrame, inject_time: int, length_min: int = 10) -> pd.DataFrame:
    half = length_min * 60 // 2
    normal = data[data["time"] < inject_time].tail(half)
    anomal = data[data["time"] >= inject_time].head(half)
    return pd.concat([normal, anomal], ignore_index=True)


def build_petshop_combined(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    inject_time: float,
) -> pd.DataFrame:
    common = normal_df.columns.intersection(abnormal_df.columns)
    n = normal_df[common].copy()
    a = abnormal_df[common].copy()

    step = 300.0
    normal_times = np.array([inject_time - (len(n) - i) * step for i in range(len(n))])
    abnormal_times = np.array([inject_time + i * step for i in range(len(a))])
    n.index = normal_times
    a.index = abnormal_times

    combined = pd.concat([n, a], axis=0)
    combined = combined.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    combined = combined.reset_index()
    combined = combined.rename(columns={combined.columns[0]: "time"})
    combined["time"] = combined["time"].astype(float)
    return combined.sort_values("time").reset_index(drop=True)


def run_rcaeval(
    limit: int,
    sampling: str,
    seed: int,
    graph_protocol: str,
    methods: list[str],
    window_min: int,
    z_threshold: float,
    pct_threshold: float,
) -> list[dict]:
    rows = []
    for dataset, root in RCAEVAL_DATA_ROOTS.items():
        case_dirs = [
            d for d in sorted(glob.glob(str(root / "*/*/")))
            if os.path.exists(os.path.join(d, "data.csv"))
            or os.path.exists(os.path.join(d, "simple_metrics.csv"))
        ]
        case_dirs = choose_case_dirs(case_dirs, limit, sampling, seed)
        for data_dir in case_dirs:
            case_rel = os.path.relpath(data_dir, root)
            try:
                data, inject_time, true_service, fault = load_rcaeval_case(data_dir)
                windowed = window_rcaeval(data, inject_time, length_min=window_min)
            except Exception as e:
                for method in methods:
                    rows.append({
                        "family": "rcaeval",
                        "system": dataset,
                        "method": method,
                        "case": case_rel,
                        "true_service": None,
                        "fault": None,
                        "error": f"load: {e}",
                    })
                continue

            graph = None
            graph_protocol_used = "none"
            if graph_protocol == "lagcorr":
                graph = build_lagcorr_graph(windowed, extract_rcaeval_component)
                graph_protocol_used = "lagcorr" if graph is not None else "lagcorr_empty"
            elif graph_protocol != "none":
                raise ValueError(f"unknown RCAEval graph protocol: {graph_protocol}")

            for method in methods:
                pred = method_predict(
                    method,
                    windowed,
                    float(inject_time),
                    extract_rcaeval_component,
                    graph=graph,
                    z_threshold=z_threshold,
                    pct_threshold=pct_threshold,
                )
                ranks = pred["ranks"]
                top1 = ranks[0] if ranks else None
                rows.append({
                    "family": "rcaeval",
                    "system": dataset,
                    "method": method,
                    "case": case_rel,
                    "true_service": true_service,
                    "fault": fault,
                    "top1": top1,
                    "acc@1": acc_at_k(ranks, true_service, 1) if ranks else np.nan,
                    "acc@3": acc_at_k(ranks, true_service, 3) if ranks else np.nan,
                    "acc@5": acc_at_k(ranks, true_service, 5) if ranks else np.nan,
                    "acc@10": acc_at_k(ranks, true_service, 10) if ranks else np.nan,
                    "n_ranks": len(ranks),
                    "n_anom_components": pred["n_anom_components"],
                    "graph_used": pred["graph_used"],
                    "graph_protocol": graph_protocol_used if method == "causal_adapt" else "none",
                    "graph_edges": graph.number_of_edges() if graph is not None else 0,
                    "error": pred["error"],
                })
    return rows


def run_petshop(
    limit: int,
    splits: list[str],
    graph_protocol: str,
    methods: list[str],
    z_threshold: float,
    pct_threshold: float,
) -> list[dict]:
    sys.path.insert(0, str(PETSHOP_ROOT / "code"))
    from rca_task import load_scenario  # noqa: PLC0415

    rows = []
    for traffic, scenario in PETSHOP_TRAFFIC_ROOTS.items():
        graph, normal_metrics, issues = load_scenario(str(scenario))
        raw_graph = normalized_graph(graph)
        graph = maybe_reverse_graph(raw_graph, graph_protocol)
        normal_flat = flatten_petshop_cols(normal_metrics)
        for split in splits:
            cases = issues[split] if limit is None or limit <= 0 else issues[split][:limit]
            for ci, (abnormal_metrics, target) in enumerate(cases):
                case_id = f"{split}/issue_{ci}"
                true_service = clean(target["root_cause"]["node"])
                inject_time = float(target["target"]["timestamp"])
                try:
                    abnormal_flat = flatten_petshop_cols(abnormal_metrics)
                    combined = build_petshop_combined(normal_flat, abnormal_flat, inject_time)
                except Exception as e:
                    for method in methods:
                        rows.append({
                            "family": "petshop",
                            "system": traffic,
                            "method": method,
                            "case": case_id,
                            "true_service": true_service,
                            "fault": "",
                            "error": f"load: {e}",
                        })
                    continue

                for method in methods:
                    pred = method_predict(
                        method,
                        combined,
                        inject_time,
                        extract_petshop_component,
                        graph=graph,
                        z_threshold=z_threshold,
                        pct_threshold=pct_threshold,
                    )
                    ranks = pred["ranks"]
                    top1 = ranks[0] if ranks else None
                    rows.append({
                        "family": "petshop",
                        "system": traffic,
                        "method": method,
                        "case": case_id,
                        "true_service": true_service,
                        "fault": "",
                        "top1": top1,
                        "acc@1": acc_at_k(ranks, true_service, 1) if ranks else np.nan,
                        "acc@3": acc_at_k(ranks, true_service, 3) if ranks else np.nan,
                        "acc@5": acc_at_k(ranks, true_service, 5) if ranks else np.nan,
                        "acc@10": acc_at_k(ranks, true_service, 10) if ranks else np.nan,
                        "true_service_canon": canonical_petshop_node(true_service),
                        "top1_canon": canonical_petshop_node(top1) if top1 else None,
                        "acc@1_canon": acc_at_k_canonical(
                            ranks, true_service, 1, canonical_petshop_node
                        ) if ranks else np.nan,
                        "acc@3_canon": acc_at_k_canonical(
                            ranks, true_service, 3, canonical_petshop_node
                        ) if ranks else np.nan,
                        "acc@5_canon": acc_at_k_canonical(
                            ranks, true_service, 5, canonical_petshop_node
                        ) if ranks else np.nan,
                        "acc@10_canon": acc_at_k_canonical(
                            ranks, true_service, 10, canonical_petshop_node
                        ) if ranks else np.nan,
                        "true_in_graph": true_service in raw_graph.nodes if raw_graph is not None else False,
                        "top1_in_graph": top1 in raw_graph.nodes if top1 and raw_graph is not None else False,
                        "n_ranks": len(ranks),
                        "n_anom_components": pred["n_anom_components"],
                        "graph_used": pred["graph_used"],
                        "graph_protocol": graph_protocol if method == "causal_adapt" else "none",
                        "graph_edges": graph.number_of_edges() if graph is not None else 0,
                        "error": pred["error"],
                    })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    optional_cols = [
        "acc@1_canon", "acc@3_canon", "acc@5_canon", "acc@10_canon",
        "true_in_graph", "top1_in_graph", "graph_edges",
    ]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    def modal_value(s: pd.Series):
        s = s.dropna().astype(str)
        if s.empty:
            return None
        return s.value_counts().index[0]

    def modal_freq(s: pd.Series):
        s = s.dropna().astype(str)
        if s.empty:
            return np.nan
        return float(s.value_counts().iloc[0] / len(s))

    ok_mask = df["error"].isna()
    df = df.copy()
    df["ok"] = ok_mask
    return df.groupby(["family", "system", "method"]).agg(
        n_attempted=("case", "size"),
        n_ok=("ok", "sum"),
        n_error=("error", lambda s: int(s.notna().sum())),
        acc1=("acc@1", "mean"),
        acc3=("acc@3", "mean"),
        acc5=("acc@5", "mean"),
        acc10=("acc@10", "mean"),
        acc1_canon=("acc@1_canon", "mean"),
        acc3_canon=("acc@3_canon", "mean"),
        acc5_canon=("acc@5_canon", "mean"),
        acc10_canon=("acc@10_canon", "mean"),
        true_graph_coverage=("true_in_graph", "mean"),
        top1_graph_coverage=("top1_in_graph", "mean"),
        mean_n_ranks=("n_ranks", "mean"),
        mean_n_anom_components=("n_anom_components", "mean"),
        graph_use_rate=("graph_used", "mean"),
        mean_graph_edges=("graph_edges", "mean"),
        modal_top1=("top1", modal_value),
        modal_freq=("top1", modal_freq),
    ).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5,
                    help="Max cases per RCAEval system and per PetShop split.")
    ap.add_argument("--rcaeval-sampling", default="stratified",
                    choices=["first", "stratified"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rcaeval-graph-protocol", default="lagcorr",
                    choices=["none", "lagcorr"])
    ap.add_argument("--petshop-graph-protocol", default="callgraph-reverse",
                    choices=["none", "callgraph-original", "callgraph-reverse"])
    ap.add_argument("--petshop-splits", nargs="+", default=["test"],
                    choices=["test", "train"])
    ap.add_argument("--methods", nargs="+",
                    default=["cd1min_adapt", "causal_adapt"],
                    choices=["cd1min_adapt", "causal_adapt"])
    ap.add_argument("--rcaeval-window-min", type=int, default=10,
                    help="RCAEval window length in minutes, centered around injection time.")
    ap.add_argument("--z-threshold", type=float, default=3.0,
                    help="Absolute z-score threshold for component anomaly detection.")
    ap.add_argument("--pct-threshold", type=float, default=5.0,
                    help="Absolute percentage-shift threshold for component anomaly detection.")
    ap.add_argument("--datafile", default="data.csv",
                    help="Per-case CSV filename for RCAEval; falls back to data.csv.")
    ap.add_argument("--out-dir", type=Path,
                    default=MOUNT_ROOT / "auditstack/results_stats/smoke_published_adapters")
    args = ap.parse_args()
    global CD_DATA_FILE
    CD_DATA_FILE = args.datafile

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(run_rcaeval(
        args.limit,
        args.rcaeval_sampling,
        args.seed,
        args.rcaeval_graph_protocol,
        args.methods,
        args.rcaeval_window_min,
        args.z_threshold,
        args.pct_threshold,
    ))
    rows.extend(run_petshop(
        args.limit,
        args.petshop_splits,
        args.petshop_graph_protocol,
        args.methods,
        args.z_threshold,
        args.pct_threshold,
    ))

    df = pd.DataFrame(rows)
    summary = summarize(df)

    per_case_path = args.out_dir / "per_case.csv"
    summary_path = args.out_dir / "summary.csv"
    notes_path = args.out_dir / "notes.json"
    df.to_csv(per_case_path, index=False)
    summary.to_csv(summary_path, index=False)

    notes = {
        "purpose": "Bounded smoke test for cross-benchmark CD-1min/Causal adapters.",
        "limit": args.limit,
        "rcaeval_sampling": args.rcaeval_sampling,
        "seed": args.seed,
        "rcaeval_graph_protocol": args.rcaeval_graph_protocol,
        "petshop_graph_protocol": args.petshop_graph_protocol,
        "petshop_splits": args.petshop_splits,
        "methods": args.methods,
        "rcaeval_window_min": args.rcaeval_window_min,
        "z_threshold": args.z_threshold,
        "pct_threshold": args.pct_threshold,
        "important_caveat": (
            "RCAEval has no supplied benchmark graph. The lagcorr graph, when "
            "enabled, is an explicit adapter protocol inferred from each case "
            "window, not a faithful reproduction of the OpenRCA PCMCI graph. "
            "PetShop graph protocol is configurable; callgraph-reverse follows "
            "the direction used by PetShop baseline methods that reverse the "
            "call graph into a causal graph."
        ),
        "not_paper_ready": (
            "This is an executability/degeneracy smoke test. Full reproduction "
            "would need graph construction choices, broader runs, and scorer audit."
        ),
    }
    notes_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print(f"Wrote {per_case_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {notes_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
