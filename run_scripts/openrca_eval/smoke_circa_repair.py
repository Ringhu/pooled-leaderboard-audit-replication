#!/usr/bin/env python3
"""S4 smoke: compare pre-repair vs post-repair CIRCA on 3 Bank cases.

For each case we:
  1. Load the parquet + build the BARO-shape DataFrame + pre-filter top-K
     metrics (identical to predict_circa.py).
  2. Run CIRCA with ORIGINAL columns -> record top-1 component.
  3. Apply the schema shim to the pre-filtered DataFrame -> run CIRCA ->
     map the top-ranked metric back to its ORIGINAL name via the shim's
     mapping dict -> record top-1 component.
  4. Extract true_root from the scoring_points.

GATE: pass iff at least one of:
  - Post-repair top-1 varies across the 3 cases.
  - Post-repair top-1 matches true_root on >=1 case.

Kill switch: if a single case takes >10 min in pre- or post-repair, abort
and report partial results.
"""
from __future__ import annotations

import os
import re
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from predict_rcaagent_1min import (  # noqa: E402
    CANDIDATES, EXTRACT, parse_instruction, count_failures,
)
from predict_baro import (  # noqa: E402
    parquet_to_baro_df, extract_component_from_metric,
)
from predict_circa import prefilter_top_k  # noqa: E402
from circa_schema_shim import repair_openrca_columns  # noqa: E402

from RCAEval.graph_construction.pc import pc_default  # noqa: E402
from RCAEval.graph_heads.rht import rht  # noqa: E402


RESULTS_1MIN = "$HOME/auditstack/results_1min"
RESULTS_EXT = "$HOME/auditstack/results_ext"
RESULTS_POOL = "$HOME/auditstack/results"
QUERY_CSV = os.path.join(HERE, "bank_q.csv")
SYSTEM = "Bank"
TOP_K = 30
N_CASES = 3
PER_CASE_TIMEOUT_SEC = 600  # 10 min kill-switch


class CaseTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise CaseTimeout("10-min wall-clock limit hit")


def _find_md(system: str, date_label: str, mid_utc) -> pd.DataFrame | None:
    for base in (RESULTS_1MIN, RESULTS_EXT):
        p = Path(base) / system / date_label / "metric_data.parquet"
        if p.exists():
            m = pd.read_parquet(p)
            if (not m.empty and m.index.min() <= mid_utc <= m.index.max()):
                return m
    p = Path(RESULTS_POOL) / system / "metric_data.parquet"
    if p.exists():
        m = pd.read_parquet(p)
        if (not m.empty and m.index.min() <= mid_utc <= m.index.max()):
            return m
    return None


def _run_circa_on_df(baro_df: pd.DataFrame, inject_time: int) -> list[str]:
    """Run PC + RHT; return the ranked metric list."""
    time_col = baro_df["time"]
    pc_input = baro_df.drop(columns=["time"])
    keep = pc_input.std() > 1e-8
    pc_input = pc_input.loc[:, keep]
    if pc_input.shape[1] < 3:
        return []
    data = pc_input.copy()
    data["time"] = time_col
    adj = pc_default(pc_input, show_progress=False)
    ranks = rht(adj, inject_time, data)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranks]


def _metric_to_component(metric_name: str, extract_fn) -> str:
    """Mimic predict_circa.predict_one's metric -> comp step."""
    cmdb = metric_name.split("__", 1)[0] if "__" in metric_name else metric_name
    return extract_fn(cmdb)


def _true_root(scoring_points: str) -> str:
    m = re.search(
        r"root cause component is ([^\s\n\.]+)", str(scoring_points)
    )
    return m.group(1) if m else ""


def main():
    qdf = pd.read_csv(QUERY_CSV)
    # Pick the first N_CASES that have a component in scoring_points
    # (the pure-onset cases have no component to score against).
    kept_rows = []
    for _, row in qdf.iterrows():
        tr = _true_root(row["scoring_points"])
        if tr:
            kept_rows.append((row, tr))
        if len(kept_rows) >= N_CASES:
            break
    if len(kept_rows) < N_CASES:
        print(f"ERR: only found {len(kept_rows)} labelled cases", file=sys.stderr)
        sys.exit(2)

    extract_fn = EXTRACT[SYSTEM]
    candidates = set(CANDIDATES[SYSTEM])

    rows_out = []
    signal.signal(signal.SIGALRM, _alarm_handler)

    for idx, (row, true_root) in enumerate(kept_rows):
        task_id = row["task_index"]
        n_fail = count_failures(str(row["scoring_points"]))
        mid_utc8, mid_utc, date_label = parse_instruction(row["instruction"])
        if mid_utc is None:
            rows_out.append((task_id, true_root, "PARSE_FAIL", "PARSE_FAIL"))
            continue

        md = _find_md(SYSTEM, date_label, mid_utc)
        if md is None:
            rows_out.append((task_id, true_root, "NO_PARQUET", "NO_PARQUET"))
            continue

        baro_df, inject_time = parquet_to_baro_df(
            md, mid_utc, window_minutes=30, baseline_hours=4
        )
        if baro_df is None:
            rows_out.append((task_id, true_root, "NO_WINDOW", "NO_WINDOW"))
            continue

        filt_df, _ = prefilter_top_k(baro_df, inject_time, top_k=TOP_K)
        if filt_df is None:
            rows_out.append((task_id, true_root, "NO_FILTER", "NO_FILTER"))
            continue

        # ===== Pre-repair =====
        pre_top = "CRASH"
        pre_top_comp = "CRASH"
        t0 = time.time()
        try:
            signal.alarm(PER_CASE_TIMEOUT_SEC)
            ranked = _run_circa_on_df(filt_df, inject_time)
        except CaseTimeout:
            ranked = None
            pre_top = "TIMEOUT"
            pre_top_comp = "TIMEOUT"
        except Exception as e:
            ranked = None
            pre_top = f"CRASH:{type(e).__name__}"
            pre_top_comp = "CRASH"
            print(f"[{task_id}] pre-repair crash: {e}", file=sys.stderr)
        finally:
            signal.alarm(0)
        pre_elapsed = time.time() - t0

        if ranked is not None:
            for m in ranked:
                c = _metric_to_component(m, extract_fn)
                if c and c in candidates:
                    pre_top_comp = c
                    pre_top = m
                    break
            else:
                pre_top = ranked[0] if ranked else "EMPTY"
                pre_top_comp = _metric_to_component(pre_top, extract_fn) or "NO_COMP"

        # ===== Post-repair =====
        post_top = "CRASH"
        post_top_comp = "CRASH"
        t0 = time.time()
        try:
            signal.alarm(PER_CASE_TIMEOUT_SEC)
            renamed_df, mapping = repair_openrca_columns(filt_df)
            ranked_new = _run_circa_on_df(renamed_df, inject_time)
        except CaseTimeout:
            ranked_new = None
            post_top = "TIMEOUT"
            post_top_comp = "TIMEOUT"
        except Exception as e:
            ranked_new = None
            post_top = f"CRASH:{type(e).__name__}"
            post_top_comp = "CRASH"
            print(f"[{task_id}] post-repair crash: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            signal.alarm(0)
        post_elapsed = time.time() - t0

        if ranked_new is not None:
            # Map renamed metric back to ORIGINAL column name.
            for m_new in ranked_new:
                m_orig = mapping.get(m_new, m_new)
                c = _metric_to_component(m_orig, extract_fn)
                if c and c in candidates:
                    post_top = m_orig
                    post_top_comp = c
                    break
            else:
                first = ranked_new[0] if ranked_new else "EMPTY"
                post_top = mapping.get(first, first)
                post_top_comp = _metric_to_component(post_top, extract_fn) or "NO_COMP"

        rows_out.append((task_id, true_root, pre_top_comp, post_top_comp))
        print(
            f"[case {idx+1}/{N_CASES}] {task_id}: true={true_root}  "
            f"pre={pre_top_comp} ({pre_elapsed:.1f}s)  "
            f"post={post_top_comp} ({post_elapsed:.1f}s)",
            flush=True,
        )

    print("\n=== S4 smoke summary ===")
    print(f"{'task':<10} {'true_root':<20} {'pre_top1':<20} {'post_top1':<20}")
    for r in rows_out:
        print(f"{r[0]:<10} {r[1]:<20} {r[2]:<20} {r[3]:<20}")

    # Gate
    post_vals = [r[3] for r in rows_out]
    post_counter = Counter(post_vals)
    unique_post = len({v for v in post_vals if not v.startswith(("CRASH", "TIMEOUT", "NO_", "PARSE_"))})
    hit_true = sum(1 for r in rows_out if r[3] == r[1])

    varies = unique_post >= 2
    hits = hit_true >= 1

    print(f"\npost unique (excl. errors): {unique_post}  varies>=2?: {varies}")
    print(f"post matches true_root: {hit_true}  hit>=1?: {hits}")
    if varies or hits:
        print("S4 GATE: PASS (schema repair changes CIRCA's behaviour)")
        sys.exit(0)
    else:
        print("S4 GATE: FAIL (schema repair does not help — degeneracy is method-inherent)")
        sys.exit(1)


if __name__ == "__main__":
    main()
