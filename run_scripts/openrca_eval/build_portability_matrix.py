#!/usr/bin/env python3
"""Build the portability matrix: 11 systems x 5 methods (+ circa_post_repair).

For each (system, method) cell, label as:
  * 'valid'       -- method ran, modal top-1 freq < 90%
  * 'degenerate'  -- method ran, modal top-1 freq >= 90%
  * 'timeout'     -- method never completed (e.g., CIRCA on RCAEval::TT, 13-day ETA)
  * 'n/a'         -- not attempted / no data

Input sources:
  * results_xbench_11sys_with_zbase/_inputs/*.csv   (7 non-OpenRCA systems)
      methods present: baro, circa (most), zbase_univ
      plus separately: causal, cd1min from other merged files if available.
  * results_openrca_xbench/*_v2.csv                 (4 OpenRCA systems)
      methods present: baro, causal, cd1min, zbase
  * preds_circa_<sys>.csv + *_q.csv                 (CIRCA pre-repair for OpenRCA)
  * results_circa_repair/*_results.csv              (CIRCA post-repair for OpenRCA)

Method name normalization:
  * 'baro'                 -> baro
  * 'causal' (OpenRCA)     -> causal
  * 'cd1min' (OpenRCA)     -> cd1min
  * 'circa'                -> circa
  * 'zbase' / 'zbase_univ' -> z_family
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
XBENCH_INPUTS = os.path.join(BASE, "results_xbench_11sys_with_zbase", "_inputs")
OPENRCA_DIR = os.path.join(BASE, "results_openrca_xbench")
CIRCA_REPAIR_DIR = os.path.join(BASE, "results_circa_repair")

METHODS = ["baro", "causal", "cd1min", "circa", "z_family"]

SYSTEMS = [
    # (display_key, source_type, source_path, filter_column_value)
    ("OpenRCA::Bank", "openrca", os.path.join(OPENRCA_DIR, "openrca_bank_results_v2.csv"), None),
    ("OpenRCA::Market1", "openrca", os.path.join(OPENRCA_DIR, "openrca_mkt1_results_v2.csv"), None),
    ("OpenRCA::Market2", "openrca", os.path.join(OPENRCA_DIR, "openrca_mkt2_results_v2.csv"), None),
    ("OpenRCA::Telecom", "openrca", os.path.join(OPENRCA_DIR, "openrca_tel_results_v2.csv"), None),
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
    if r in ("baro",):
        return "baro"
    if r in ("causal",):
        return "causal"
    if r in ("cd1min",):
        return "cd1min"
    if r in ("circa",):
        return "circa"
    if r in ("zbase", "zbase_univ"):
        return "z_family"
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


def load_openrca_circa_preds(pred_path: str) -> tuple[float, str, int]:
    if not os.path.exists(pred_path):
        return 0.0, "", 0
    preds = pd.read_csv(pred_path)
    top1 = preds["prediction"].apply(first_component)
    return _modal_freq_from_top1(top1)


def main():
    # Build per-system per-method top1 extractor.
    cells: dict[tuple[str, str], str] = {}
    detail_rows = []

    # For OpenRCA systems, the v2 merged csv does NOT store top1 per case
    # (it only stores the binary acc1_v2 score). So we read from per-method
    # prediction files (preds_<method>_<sys>.csv) instead.
    # full335/ holds the post-BARO-fix preds; top-level scripts/openrca_eval/
    # holds pre-fix (stale) BARO preds and the canonical CIRCA preds.
    # Prefer full335 when available.
    openrca_pred_prefix = {
        "baro": "preds_baro",
        "causal": "preds_causal",
        "cd1min": "preds_1min",
        "circa": "preds_circa",
        "z_family": "preds_zbaseline",
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
            # Use per-method prediction files.
            tag = openrca_sys_tag[sys_disp]
            for meth in METHODS:
                prefix = openrca_pred_prefix.get(meth)
                if prefix is None:
                    cells[(sys_disp, meth)] = "n/a"
                    continue
                # Prefer full335/ (post-BARO-fix) when present.
                cand1 = os.path.join(FULL335, f"{prefix}_{tag}.csv")
                cand2 = os.path.join(HERE, f"{prefix}_{tag}.csv")
                pred_path = cand1 if os.path.exists(cand1) else cand2
                mfreq, mval, n = load_openrca_circa_preds(pred_path)
                cells[(sys_disp, meth)] = _label(mfreq, n)
                detail_rows.append({
                    "system": sys_disp, "method": meth,
                    "n": n, "modal_freq": round(mfreq, 4),
                    "modal_top1": mval, "label": cells[(sys_disp, meth)],
                })
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
            mfreq, mval, n = load_openrca_circa_preds(post_path)
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
    mat_path = os.path.join(out_dir, "portability_matrix.csv")
    mat_df.to_csv(mat_path, index=False)
    print(f"Wrote {mat_path}")
    print(mat_df.to_string(index=False))

    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(out_dir, "portability_matrix_detail.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"\nWrote {detail_path}")


if __name__ == "__main__":
    main()
