#!/usr/bin/env python3
"""Build the UNIFIED portability matrix: 11 systems x 5 methods (+ circa_post_repair).

R1 unified rerun variant of build_portability_matrix.py:
  * Reads from results_xbench_11sys_unified/_inputs/ instead of
    results_xbench_11sys_with_zbase/_inputs/.
  * Column `z_family` is renamed to `zbase_univ`. The underlying OpenRCA
    predictor is now the universal Z-score (same algorithm as RCAEval /
    PetShop) — sourced from preds_zbase_univ_<tag>.csv instead of the
    legacy preds_zbaseline_<tag>.csv.
  * CIRCA pre-repair (OpenRCA) and CIRCA post-repair (OpenRCA) are
    unchanged — same prediction files as the legacy run.

Labels per (system, method) cell:
  * 'valid'       -- method ran, modal top-1 freq < 90%
  * 'degenerate'  -- method ran, modal top-1 freq >= 90%
  * 'timeout'     -- method never completed (e.g., CIRCA on RCAEval::TT)
  * 'n/a'         -- not attempted / no data

Output:
  * results_stats/portability_matrix_unified.csv
  * results_stats/portability_matrix_unified_detail.csv
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE = "$HOME/auditstack"
XBENCH_INPUTS = os.path.join(BASE, "results_xbench_11sys_unified", "_inputs")
CIRCA_REPAIR_DIR = os.path.join(BASE, "results_circa_repair")

METHODS = ["baro", "causal", "cd1min", "circa", "zbase_univ"]

SYSTEMS = [
    # (display_key, source_type, source_path, filter_column_value)
    ("OpenRCA::Bank", "openrca",
     os.path.join(XBENCH_INPUTS, "openrca_bank_results_v2.csv"), None),
    ("OpenRCA::Market1", "openrca",
     os.path.join(XBENCH_INPUTS, "openrca_mkt1_results_v2.csv"), None),
    ("OpenRCA::Market2", "openrca",
     os.path.join(XBENCH_INPUTS, "openrca_mkt2_results_v2.csv"), None),
    ("OpenRCA::Telecom", "openrca",
     os.path.join(XBENCH_INPUTS, "openrca_tel_results_v2.csv"), None),
    ("RCAEval::OnlineBoutique", "xbench",
     os.path.join(XBENCH_INPUTS, "rcaeval_merged_online-boutique_results.csv"), None),
    ("RCAEval::SockShop2", "xbench",
     os.path.join(XBENCH_INPUTS, "rcaeval_merged_sock-shop-2_results.csv"), None),
    ("RCAEval::TrainTicket", "xbench",
     os.path.join(XBENCH_INPUTS, "rcaeval_merged_train-ticket_results.csv"), None),
    ("PetShop::HighTraffic", "xbench",
     os.path.join(XBENCH_INPUTS, "petshop_merged_high_traffic_results.csv"), None),
    ("PetShop::LowTraffic", "xbench",
     os.path.join(XBENCH_INPUTS, "petshop_merged_low_traffic_results.csv"), None),
    ("PetShop::TemporalTraffic1", "xbench",
     os.path.join(XBENCH_INPUTS, "petshop_merged_temporal_traffic1_results.csv"), None),
    ("PetShop::TemporalTraffic2", "xbench",
     os.path.join(XBENCH_INPUTS, "petshop_merged_temporal_traffic2_results.csv"), None),
]


def _canonical_method(raw: str) -> str:
    r = raw.strip()
    if r == "baro":
        return "baro"
    if r == "causal":
        return "causal"
    if r == "cd1min":
        return "cd1min"
    if r == "circa":
        return "circa"
    # Legacy 'zbase' and 'z_family' collapse under the unified name.
    if r in ("zbase", "zbase_univ", "z_family"):
        return "zbase_univ"
    return r


def _modal_freq_from_top1(series: pd.Series) -> tuple[float, str, int]:
    vals = [v for v in series.tolist() if isinstance(v, str) and v]
    if not vals:
        return 0.0, "", 0
    c = Counter(vals)
    m, mcnt = c.most_common(1)[0]
    return mcnt / len(vals), m, len(vals)


def _label(modal_freq: float, n: int) -> str:
    if n == 0:
        return "n/a"
    return "degenerate" if modal_freq >= 0.9 else "valid"


def load_merged(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method_canon"] = df["method"].apply(_canonical_method)
    # keep only tdelta=0 (no tolerance)
    if "tdelta" in df.columns:
        df = df[df["tdelta"] == 0]
    return df


# --- CIRCA pre-repair on OpenRCA: compute from preds_circa_<sys>.csv ---
FIRST_COMP_RE = re.compile(r'"root cause component":\s*"([^"]+)"')


def first_component(s: str) -> str:
    m = FIRST_COMP_RE.search(str(s))
    return m.group(1) if m else ""


def load_openrca_preds(pred_path: str) -> tuple[float, str, int]:
    if not os.path.exists(pred_path):
        return 0.0, "", 0
    preds = pd.read_csv(pred_path)
    top1 = preds["prediction"].apply(first_component)
    return _modal_freq_from_top1(top1)


def main():
    cells: dict[tuple[str, str], str] = {}
    detail_rows = []

    # OpenRCA per-method prediction file prefixes.
    openrca_pred_prefix = {
        "baro": "preds_baro",
        "causal": "preds_causal",
        "cd1min": "preds_1min",
        "circa": "preds_circa",
        # UNIFIED: use the universal Z-score preds, not the legacy zbaseline.
        "zbase_univ": "preds_zbase_univ",
    }
    FULL335 = os.path.join(HERE, "full335")
    openrca_sys_tag = {
        "OpenRCA::Bank": "bank",
        "OpenRCA::Market1": "mkt1",
        "OpenRCA::Market2": "mkt2",
        "OpenRCA::Telecom": "tel",
    }

    for sys_disp, src_type, path, _ in SYSTEMS:
        if src_type == "openrca":
            tag = openrca_sys_tag[sys_disp]
            # Cross-check per-method pred file rowcount vs unified v2 CSV rowcount.
            expected_n = None
            if os.path.exists(path):
                try:
                    v2df = pd.read_csv(path)
                    # Per-method row count from the v2 CSV (tdelta=0 only).
                    if "tdelta" in v2df.columns:
                        v2df = v2df[v2df["tdelta"] == 0]
                    method_counts = v2df.groupby("method").size().to_dict()
                    expected_n = method_counts
                except Exception:
                    expected_n = None
            for meth in METHODS:
                prefix = openrca_pred_prefix.get(meth)
                if prefix is None:
                    cells[(sys_disp, meth)] = "n/a"
                    continue
                # Prefer full335/ (post-BARO-fix) when present. The unified
                # zbase_univ preds live only at the top level.
                cand1 = os.path.join(FULL335, f"{prefix}_{tag}.csv")
                cand2 = os.path.join(HERE, f"{prefix}_{tag}.csv")
                pred_path = cand1 if os.path.exists(cand1) else cand2
                mfreq, mval, n = load_openrca_preds(pred_path)
                cells[(sys_disp, meth)] = _label(mfreq, n)
                detail_rows.append({
                    "system": sys_disp, "method": meth,
                    "n": n, "modal_freq": round(mfreq, 4),
                    "modal_top1": mval, "label": cells[(sys_disp, meth)],
                })
                # Flag rowcount mismatch vs the unified v2 CSV's method count.
                if expected_n is not None:
                    # v2 CSV uses raw method names; map back for comparison.
                    raw_key = "zbase_univ" if meth == "zbase_univ" else meth
                    exp = expected_n.get(raw_key, 0)
                    if exp and n != exp:
                        print(
                            f"  [WARN] {sys_disp}/{meth}: pred file has "
                            f"n={n} but unified v2 CSV has {exp} rows."
                        )
            continue
        if not os.path.exists(path):
            for meth in METHODS:
                cells[(sys_disp, meth)] = "n/a"
            continue
        df = load_merged(path)
        methods_present = set(df["method_canon"].unique())
        for meth in METHODS:
            if meth not in methods_present:
                # Special-case: CIRCA missing on RCAEval::TrainTicket -> timeout
                if meth == "circa" and sys_disp == "RCAEval::TrainTicket":
                    cells[(sys_disp, meth)] = "timeout"
                    detail_rows.append({
                        "system": sys_disp, "method": meth,
                        "n": 0, "modal_freq": None,
                        "modal_top1": "", "label": "timeout",
                    })
                    continue
                cells[(sys_disp, meth)] = "n/a"
                detail_rows.append({
                    "system": sys_disp, "method": meth,
                    "n": 0, "modal_freq": None,
                    "modal_top1": "", "label": "n/a",
                })
                continue
            sub = df[df["method_canon"] == meth]
            mfreq, mval, n = _modal_freq_from_top1(sub["top1"])
            cells[(sys_disp, meth)] = _label(mfreq, n)
            detail_rows.append({
                "system": sys_disp, "method": meth,
                "n": n, "modal_freq": round(mfreq, 4),
                "modal_top1": mval, "label": cells[(sys_disp, meth)],
            })

    # Matrix: rows = systems, cols = methods, plus circa_post_repair on OpenRCA.
    col_names = METHODS + ["circa_post_repair"]
    rows = []
    for sys_disp, _, _, _ in SYSTEMS:
        row = {"system": sys_disp}
        for meth in METHODS:
            row[meth] = cells.get((sys_disp, meth), "n/a")
        # circa_post_repair: only on the 4 OpenRCA systems.
        if sys_disp.startswith("OpenRCA::"):
            short = sys_disp.split("::")[-1].lower()
            short_map = {"bank": "bank", "market1": "mkt1",
                         "market2": "mkt2", "telecom": "tel"}
            tag = short_map.get(short, short)
            post_path = os.path.join(
                CIRCA_REPAIR_DIR, f"circa_repaired_{tag}_results.csv"
            )
            mfreq, mval, n = load_openrca_preds(post_path)
            row["circa_post_repair"] = _label(mfreq, n)
            detail_rows.append({
                "system": sys_disp, "method": "circa_post_repair",
                "n": n, "modal_freq": round(mfreq, 4),
                "modal_top1": mval, "label": row["circa_post_repair"],
            })
        else:
            row["circa_post_repair"] = "n/a"
        rows.append(row)

    mat_df = pd.DataFrame(rows, columns=["system"] + col_names)
    out_dir = os.path.join(BASE, "results_stats")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mat_path = os.path.join(out_dir, "portability_matrix_unified.csv")
    mat_df.to_csv(mat_path, index=False)
    print(f"Wrote {mat_path}")
    print(mat_df.to_string(index=False))

    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(out_dir, "portability_matrix_unified_detail.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"\nWrote {detail_path}")


if __name__ == "__main__":
    main()
