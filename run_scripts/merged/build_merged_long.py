"""Build the canonical long table merged_long.csv (one row per case x method x run variant).

Columns
  family         rcaeval | openrca | petshop
  release        RE1 | RE2 (rcaeval); "" otherwise
  subsystem      OB/SS/TT, bank/mkt1/mkt2/tel, high_traffic/...
  subsystem_id   <family or release>::<subsystem>; the 11 inference units are RE1::*, openrca::*, petshop::*
  case, q_idx    case directory / task (OpenRCA: task_index + position within it)
  case_id        <subsystem_id>|<case>#<q_idx>
  fault          RCAEval fault type (empty elsewhere)
  method         method name
  seed           RCD seeds 0/1/2; empty otherwise
  impl_source    where the implementation comes from
  role           published | probe          -> main table
                 appendix                    -> SimpleRCA author impl where it reduces to a metric-only BARO variant
                 sensitivity                 -> SimpleRCA rewritten from the paper text (B paper)
                 shipped_baseline            -> PetShop authors' own baselines (traversal, ranked correlation, random walk)
                 contrast_input              -> raw-export input (RE1 data.csv, RE2 metrics.csv): RQ2(a) only
                 contrast_release_defect     -> RE1-OB 50 cases with the duplicate "time" header as shipped (RQ2)
                 diagnostic_window           -> RE1/RE2 10-min window (A long-table window)
                 diagnostic                  -> reproduction checks / non-official configurations
  input_repr     input file(s)
  window_min     RCAEval window length (minutes); empty elsewhere
  score          top-1 hit (RCAEval, PetShop node-level) or OpenRCA official partial score; 0 when valid_output = 0
  score_strict   OpenRCA: score == 1; RCAEval/PetShop: = score
  valid_output   1 if the method produced its own answer (RCAEval: non-empty ranking, no error, not CausalRCA's
                 empty-graph column-order fallback; OpenRCA: not the
                 wrapper's fixed default answer; PetShop: non-empty, no error)
  source_file    path relative to $AUDIT_ROOT (or absolute for files outside it)

Scoring rules (unchanged from each source):
  RCAEval  runs/{baro,probes,rcd,re2,causalrca_full}: per-case acc1 written by the runner (RCAEval main.py Evaluator rule; probes use
           the audit probe scorer, equal to it on every case). A-followup / B-pilot files that store only ranks are
           scored here with rcaeval_repo/_baseline_utils.acc_at_k. SimpleRCA author runs: hits[0] (exact match,
           platform rule).
  OpenRCA  official evaluate() at 60 s (audit convention). audit prediction files are scored from the A long
           table `correct` column (verified equal to evaluate() 1340/1340); default answers detected as in
           openrca_default_answer_check.py.
  PetShop  hit1_nodeonly of the per-case harness output (= audit petshop_analysis.py) for the 7 shipped
           implementations; A long-table `correct` for BARO and the three probes.

Usage (3090, env auditstack): python build_merged_long.py [--out merged_long.csv]
"""
import argparse
import ast
import json
import os
import re
import sys

import pandas as pd

M = "$AUDIT_ROOT"
REPO = "$HOME/rcaeval_repo"
A_EVAL = "$HOME/auditstack/scripts/openrca_eval"
OPEN_RCA = "$HOME/auditstack/data/OpenRCA/dataset"
PETSHOP = "$HOME/petshop_percase"
LONG_A = f"{M}/raw/earlier_a_followup_data/full_coverage_cd1min_long_CORRECTED.csv"
sys.path.insert(0, REPO)
sys.path.insert(0, A_EVAL)
from _baseline_utils import acc_at_k  # noqa: E402

SUB = {"ob": "OB", "ss": "SS", "tt": "TT"}
RE_ROOT = {"RE1": {"OB": "data/online-boutique", "SS": "data/sock-shop-2", "TT": "data/train-ticket"},
           "RE2": {"OB": "data/RE2/RE2-OB", "SS": "data/RE2/RE2-SS", "TT": "data/RE2/RE2-TT"}}
RE1_DEFAULT_INPUT = {"OB": "data.csv", "SS": "simple_data.csv", "TT": "simple_data.csv"}
ROWS = []


def rel(p):
    return os.path.relpath(p, M) if p.startswith(M) else p


def case_dirs(release, sub):
    """Sorted numbered case directories (order used by the A-followup / B-pilot runners that store no case id)."""
    root = f"{REPO}/{RE_ROOT[release][sub]}"
    out = []
    for sf in sorted(os.listdir(root)):
        d = os.path.join(root, sf)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d), key=lambda x: int(x) if x.isdigit() else 10 ** 9):
            if n.isdigit():
                out.append(f"{sf}/{n}")
    return out


def as_list(x):
    if isinstance(x, list):
        return [str(v) for v in x]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [str(v) for v in ast.literal_eval(str(x))]


def add(family, release, sub, case, q_idx, method, role, impl, inp, score, valid, src, *, fault="", seed="",
        window="", strict=None):
    sid = f"{release}::{sub}" if family == "rcaeval" else f"{family}::{sub}"
    score = float(score) if valid else 0.0
    ROWS.append(dict(family=family, release=release, subsystem=sub, subsystem_id=sid, case=case, q_idx=q_idx,
                     case_id=f"{sid}|{case}#{q_idx}", fault=fault, method=method, seed=seed, impl_source=impl,
                     role=role, input_repr=inp, window_min=window, score=score,
                     score_strict=float(strict) if strict is not None and valid else score,
                     valid_output=int(bool(valid)), source_file=rel(src)))


# ---------------------------------------------------------------- RCAEval: runs written by this project's runners
def rcaeval_runner_file(path, release, sub, method, role, impl, inp, window, seed=""):
    d = json.load(open(path))
    expect = case_dirs(release, sub)
    got = [c["case"] for c in d["cases"]]
    assert sorted(got) == sorted(expect), (path, len(got), len(expect))
    for c in d["cases"]:
        # CausalRCA: an empty learned graph makes PageRank raise and causalrca.py return the input column order
        # (its `except Exception` branch); that fixed fallback is not the method's own answer
        ok = (c.get("error") in (None, "None") and int(c.get("n_ranks") or 0) > 0
              and not c.get("ranks_equal_column_order"))
        add("rcaeval", release, sub, c["case"], 0, method, role, impl, inp, c["acc1"] if ok else 0, ok, path,
            fault=c.get("fault", ""), seed=seed, window=window)


def rcaeval_ranks_file(path, release, sub, method, role, impl, inp, window, case_key=None):
    """A-followup / B-pilot files: list of {service, fault, ranks} (sorted-case order) or {case, acc1}."""
    d = json.load(open(path))
    cases = d["cases"]
    expect = case_dirs(release, sub)
    assert len(cases) == len(expect), (path, len(cases), len(expect))
    for i, c in enumerate(cases):
        case = c.get("case") or expect[i]
        svc, fault = case.split("/")[0].rsplit("_", 1)
        if "case" in c:
            assert c["case"] == expect[i] or c["case"] in expect, (path, c["case"])
        else:
            assert c["service"] == svc and c["fault"] == fault, (path, i, c["service"], svc)
        if "ranks" in c:
            ranks = as_list(c["ranks"])
            ok = len(ranks) > 0
            s = acc_at_k(ranks, svc, 1) if ok else 0
        else:  # pub_simple.py output: acc1 + top1 only
            ok = c.get("top1") not in (None, "", "None")
            s = float(c["acc1"])
        add("rcaeval", release, sub, case, 0, method, role, impl, inp, s, ok, path, fault=fault, window=window)


def simplerca_author_file(path, release, sub, method, role, inp, window):
    d = json.load(open(path))
    expect = {c.replace("/", "_"): c for c in case_dirs(release, sub)}
    assert len(d["cases"]) == len(expect), path
    for c in d["cases"]:
        case = expect[c["datapack"]]
        hits = c.get("hits") or []
        ok = c.get("exception") in (None, "None") and len(c.get("answers") or []) > 0
        add("rcaeval", release, sub, case, 0, method, role, "author repo LGU-SE-Internal/Simple-RCA "
            + d["config"]["author_branch_commit"], inp, float(bool(hits and hits[0])), ok, path,
            fault=case.split("/")[0].rsplit("_", 1)[1], window=window)


def build_rcaeval():
    OFF = "RCAEval official (rcaeval_repo@497505b)"
    APROBE = "audit probe (auditstack openrca_eval)"
    REWRITE = "audit rewrite of Fang et al. §3.1.2 metric branch (earlier-b 64_simplerca_rcaeval.py)"
    for s, sub in SUB.items():
        # ---------------- RE1
        inp = RE1_DEFAULT_INPUT[sub]
        rcaeval_runner_file(f"{M}/runs/baro/baro_re1-{s}_len20.json", "RE1", sub, "baro", "published", OFF, inp, 20)
        rcaeval_runner_file(f"{M}/runs/baro/baro_re1-{s}_len10.json", "RE1", sub, "baro", "diagnostic_window", OFF, inp, 10)
        for m, name in (("zbase", "max_z"), ("alertcount", "alertcount"), ("cd1min", "cd1min")):
            for L, role in ((20, "probe"), (10, "diagnostic_window")):
                rcaeval_runner_file(f"{M}/runs/probes/{m}_re1-{s}_len{L}.json", "RE1", sub, name, role, APROBE, inp, L)
        for seed in (0, 1, 2):
            rcaeval_runner_file(f"{M}/runs/rcd/rcd_re1-{s}_seed{seed}.json", "RE1", sub, "rcd", "published", OFF, inp,
                                20, seed=seed)
        # CausalRCA in the rcaeval_rcd env (scikit-network 0.31.0), seeds 0-2 (decision 2026-09-29)
        for seed in (0, 1, 2):
            p = f"{M}/runs/causalrca_full/causalrca_re1-{s}_seed{seed}.json"
            if os.path.exists(p):
                rcaeval_runner_file(p, "RE1", sub, "causalrca", "published", OFF, inp, 20, seed=seed)
        A = f"{M}/raw/earlier_a_followup_runs"
        if sub == "OB":
            rcaeval_ranks_file(f"{A}/circa_re1-ob.json", "RE1", sub, "circa", "published", OFF, inp, 20)
            rcaeval_ranks_file(f"{A}/ediag_re1-ob.json", "RE1", sub, "e_diagnosis", "published", OFF, inp, 20)
        else:
            rcaeval_ranks_file(f"{A}/circa_re1-{s}_simple_data.json", "RE1", sub, "circa", "published", OFF, inp, 20)
            rcaeval_ranks_file(f"{A}/e_diagnosis_re1-{s}_simple_data.json", "RE1", sub, "e_diagnosis", "published", OFF,
                               inp, 20)
            # raw-export data.csv contrast (RQ2(a))
            rcaeval_runner_file(f"{M}/runs/baro/baro_re1-{s}_datacsv_len20.json", "RE1", sub, "baro", "contrast_input",
                                OFF, "data.csv", 20)
            for m, name in (("zbase", "max_z"), ("alertcount", "alertcount"), ("cd1min", "cd1min")):
                rcaeval_runner_file(f"{M}/runs/probes/{m}_re1-{s}_datacsv_len20.json", "RE1", sub, name,
                                    "contrast_input", APROBE, "data.csv", 20)
            for seed in (0, 1, 2):
                p = f"{M}/runs/rcd/rcd_re1-{s}_datacsv_seed{seed}.json"
                if os.path.exists(p):
                    rcaeval_runner_file(p, "RE1", sub, "rcd", "contrast_input", OFF, "data.csv", 20, seed=seed)
            circa_raw = f"{A}/circa_re1-ss.json" if s == "ss" else f"{M}/raw/earlier_b/circa_re1-tt.json"
            rcaeval_ranks_file(circa_raw, "RE1", sub, "circa", "contrast_input", OFF, "data.csv", 20)
            rcaeval_ranks_file(f"{A}/ediag_re1-{s}.json", "RE1", sub, "e_diagnosis", "contrast_input", OFF, "data.csv", 20)
        simplerca_author_file(f"{M}/runs/simplerca_author/simplerca_author_re1_{s}_win20.json", "RE1", sub,
                              "simplerca_author", "appendix", inp, 20)
        rcaeval_runner_file(f"{M}/runs/simplerca_rewrite/simplerca_rewrite_re1-{s}.json", "RE1", sub,
                            "simplerca_rewrite", "sensitivity", REWRITE, inp, 20)
        # ---------------- RE2
        R = f"{M}/runs/re2"
        rcaeval_runner_file(f"{M}/runs/baro/baro_re2-{s}_len20.json", "RE2", sub, "baro", "published", OFF,
                            "simple_metrics.csv", 20)
        rcaeval_runner_file(f"{M}/runs/baro/baro_re2-{s}_len10.json", "RE2", sub, "baro", "diagnostic_window", OFF,
                            "simple_metrics.csv", 10)
        for m, name in (("zbase", "max_z"), ("alertcount", "alertcount"), ("cd1min", "cd1min")):
            for L, role in ((20, "probe"), (10, "diagnostic_window")):
                rcaeval_runner_file(f"{M}/runs/probes/{m}_re2-{s}_len{L}.json", "RE2", sub, name, role, APROBE,
                                    "simple_metrics.csv", L)
            rcaeval_runner_file(f"{M}/runs/probes/{m}_re2-{s}_metricscsv_len20.json", "RE2", sub, name,
                                "contrast_input", APROBE, "metrics.csv", 20)
        for L, role in ((20, "published"), (10, "diagnostic_window")):
            p = f"{R}/circa_re2-{s}_len{L}.json"
            if os.path.exists(p):
                rcaeval_runner_file(p, "RE2", sub, "circa", role, OFF, "simple_metrics.csv", L)
        rcaeval_ranks_file(f"{M}/raw/earlier_b/e_diagnosis_re2{s}.json", "RE2", sub, "e_diagnosis", "published", OFF,
                           "simple_metrics.csv", 20)
        mm_in = "simple_metrics.csv + logts.csv + tracets_{lat,err}.csv"
        rcaeval_runner_file(f"{R}/mmbaro_re2-{s}_name-mm.json", "RE2", sub, "mmbaro", "published", OFF, mm_in, 20)
        rcaeval_runner_file(f"{R}/mmbaro_re2-{s}_name-re2.json", "RE2", sub, "mmbaro", "diagnostic", OFF,
                            mm_in + " (dataset name re2-*: trace branch off)", 20)
        rcaeval_runner_file(f"{R}/mmbaro_re2-{s}_name-mm_metricscsv.json", "RE2", sub, "mmbaro", "contrast_input", OFF,
                            mm_in.replace("simple_metrics.csv", "metrics.csv"), 20)
        for seed in (0, 1, 2):
            p = f"{M}/runs/causalrca_full/causalrca_re2-{s}_seed{seed}.json"
            if os.path.exists(p):
                rcaeval_runner_file(p, "RE2", sub, "causalrca", "published", OFF, "simple_metrics.csv", 20, seed=seed)
        for m in ("tracerca", "microrank"):
            p = f"{R}/{m}_re2-{s}.json"
            if os.path.exists(p):
                rcaeval_runner_file(p, "RE2", sub, m, "published", OFF, "traces.csv", 20)
        rcaeval_ranks_file(f"{M}/raw/earlier_b/simplerca_re2{s}.json", "RE2", sub, "simplerca_rewrite", "sensitivity",
                           REWRITE, "simple_metrics.csv", 20)
        simplerca_author_file(f"{M}/runs/simplerca_author/simplerca_author_re2_{s}.json", "RE2", sub,
                              "simplerca_author", "published", "simple_metrics.csv + traces.csv + logs.csv", "")


# ---------------------------------------------------------------- OpenRCA
def build_openrca():
    from openrca_evaluate_sweep import evaluate  # noqa: F401  (scores_long.csv was written with it)
    from predict_rcaagent_1min import DEFAULT, parse_instruction
    systems = {"bank": ("Bank", f"{OPEN_RCA}/Bank/query.csv"),
               "mkt1": ("Market_cloudbed-1", f"{OPEN_RCA}/Market/cloudbed-1/query.csv"),
               "mkt2": ("Market_cloudbed-2", f"{OPEN_RCA}/Market/cloudbed-2/query.csv"),
               "tel": ("Telecom", f"{OPEN_RCA}/Telecom/query.csv")}
    along = pd.read_csv(LONG_A)
    along = along[along.benchmark == "openrca"]
    a_files = {"baro": ("baro", "full335/preds_baro_{tag}.csv", "published", "RCAEval baro via audit OpenRCA wrapper"),
               "cd1min": ("cd1min", "full335/preds_1min_{tag}.csv", "probe", "audit CD-1min"),
               "z_family": ("max_z", "preds_zbase_univ_{tag}.csv", "probe", "audit probe")}
    inp = "OpenRCA metric parquet (audit wrapper: 4 h baseline + 30 min either side)"
    for tag, (sysname, qp) in systems.items():
        q = pd.read_csv(qp)
        q["q_idx"] = q.groupby("task_index").cumcount()
        comp, reason = DEFAULT[sysname]
        default_ans = []
        for ins in q.instruction:
            mid8, _, _ = parse_instruction(ins)
            dt = mid8.strftime('%Y-%m-%d %H:%M:%S') if mid8 is not None else ''
            default_ans.append(f'{{"root cause occurrence datetime": "{dt}", "root cause component": "{comp}", '
                               f'"root cause reason": "{reason}"}}')
        for lm, (name, f, role, impl) in a_files.items():
            preds = pd.read_csv(f"{A_EVAL}/{f.format(tag=tag)}")
            assert (preds.task_index == q.task_index).all()
            lt = along[(along.method == lm) & (along.system == tag)].set_index(["case", "q_idx"]).correct
            for i, r in q.iterrows():
                case = f"{tag}/{r.task_index}"
                s = float(lt.loc[(case, r.q_idx)])
                p = str(preds.prediction[i])
                is_default = p != "nan" and set(re.findall(r"\{[^}]*\}", p)) == {default_ans[i]}
                add("openrca", "", tag, case, int(r.q_idx), name, role, impl, inp, s, not is_default,
                    f"{A_EVAL}/{f.format(tag=tag)}", strict=float(s == 1.0))
    sc = pd.read_csv(f"{M}/runs/openrca/scores_long.csv")
    names = {"ediag": ("e_diagnosis", "published", "RCAEval e_diagnosis via audit OpenRCA wrapper"),
             "simplerca": ("simplerca_rewrite", "sensitivity",
                           "audit rewrite (64_simplerca_rcaeval.py) via audit OpenRCA wrapper, alerts per component"),
             "alertcount": ("alertcount", "probe", "audit probe, call bug fixed"),
             "alertcount-unfixed": ("alertcount", "diagnostic", "audit probe as run (335/335 default answers)"),
             "baro": ("baro", "diagnostic", "harness reproduction of audit BARO predictions")}
    # RCD, seeds 0-2 (user ruling 2026-09-29, .research/decisions/2026-09-29-openrca-rcd-unfreeze.md)
    for s in (0, 1, 2):
        names[f"rcd-seed{s}"] = ("rcd", "published",
                                 "RCAEval rcd via audit OpenRCA wrapper (ranks computed in rcaeval_rcd env)")
    for m, (name, role, impl) in names.items():
        d = sc[sc.method == m]
        seed = int(m.split("seed")[1]) if m.startswith("rcd-seed") else ""
        assert len(d) == 335 or not m.startswith("rcd-seed"), (m, len(d))
        for _, r in d.iterrows():
            add("openrca", "", r.tag, r.case, int(r.q_idx), name, role, impl, inp, r.score, r.status == "method",
                f"{M}/runs/openrca/preds_{m}_openrca_{r.tag}.csv", strict=r.strict, seed=seed)


# ---------------------------------------------------------------- PetShop
def build_petshop():
    shipped = ["baseline", "circa", "counterfactual_attribution", "epsilon_diagnosis", "random_walk",
               "ranked_correlation", "rcd"]
    along = pd.read_csv(LONG_A)
    along = along[along.benchmark == "petshop"]
    for sc in ("high_traffic", "low_traffic", "temporal_traffic1", "temporal_traffic2"):
        cases = None
        for m in shipped:
            p = f"{PETSHOP}/{m}__{sc}.csv"
            d = pd.read_csv(p)
            d["case"] = d.split + "/" + d.issue
            cases = set(d.case) if cases is None else cases
            assert set(d.case) == cases and len(d) == len(cases), p
            for _, r in d.iterrows():
                ok = int(r["empty"]) == 0 and (pd.isna(r["err"]) or str(r["err"]) == "")
                # baseline (= traversal), ranked_correlation, random_walk are the benchmark authors' own baselines,
                # not counted as published comparators (A followup convention)
                role = "shipped_baseline" if m in ("baseline", "ranked_correlation", "random_walk") else "published"
                add("petshop", "", sc, r.case, 0, m, role, "PetShop shipped implementation (petshop_repo)",
                    "PetShop official harness", r.hit1_nodeonly, ok, p)
        for lm, name, role in (("baro", "baro", "published"), ("cd1min", "cd1min", "probe"),
                               ("z_family", "max_z", "probe"), ("alertcount", "alertcount", "probe")):
            d = along[(along.system == sc) & (along.method == lm)]
            assert set(d.case) == cases, (sc, lm)
            # audit PetShop outputs (auditstack results_petshop/, results_alertcount/,
            # results_stats/cd1min_adapt_main_11sys/per_case.csv) have no error rows and no empty rankings
            for _, r in d.iterrows():
                add("petshop", "", sc, r.case, int(r.q_idx), name, role, "audit (auditstack openrca_eval)",
                    "PetShop metrics (audit adapter)", r.correct, True, LONG_A)


# ---------------------------------------------------------------- RE1-OB duplicate "time" header (2026-09-30)
# 50 RE1-OB cpu/mem data.csv files repeat the "time" header (pandas: "time.1"); fixed upstream in RCAEval 21ff8c9.
# Decision .research/decisions/2026-09-30-re1ob-timefix-unfreeze.md: on those 50 cases the main-table rows come from
# the rerun with the duplicate column dropped (runs/re1ob_timefix/, scripts/timefix_wrapper.py); the original rows
# stay in the table with role contrast_release_defect (RQ2). CIRCA, e-Diagnosis and RCD are always replaced; BARO and
# the probes (rerun only as a check) keep their original rows when no score changes on the 50 affected cases.
FORCE_REPLACE = {"circa", "e_diagnosis", "rcd"}
TIMEFIX = {  # method: [(timefix file, original main-table source relative to M, seed)]
    "circa": [("circa_re1-ob_timefix.json", "raw/earlier_a_followup_runs/circa_re1-ob.json", "")],
    "e_diagnosis": [("e_diagnosis_re1-ob_timefix.json", "raw/earlier_a_followup_runs/ediag_re1-ob.json", "")],
    "rcd": [(f"rcd_re1-ob_seed{s}_timefix.json", f"runs/rcd/rcd_re1-ob_seed{s}.json", s) for s in (0, 1, 2)],
    "baro": [("baro_re1-ob_len20_timefix.json", "runs/baro/baro_re1-ob_len20.json", "")],
    "max_z": [("zbase_re1-ob_len20_timefix.json", "runs/probes/zbase_re1-ob_len20.json", "")],
    "alertcount": [("alertcount_re1-ob_len20_timefix.json", "runs/probes/alertcount_re1-ob_len20.json", "")],
    "cd1min": [("cd1min_re1-ob_len20_timefix.json", "runs/probes/cd1min_re1-ob_len20.json", "")],
}


def apply_timefix(df):
    root = f"{REPO}/{RE_ROOT['RE1']['OB']}"
    affected = sorted(c for c in case_dirs("RE1", "OB")
                      if "time.1" in pd.read_csv(f"{root}/{c}/data.csv", nrows=0).columns)
    assert len(affected) == 50, len(affected)
    keep, new, log = [df], [], []
    for method, runs in TIMEFIX.items():
        for fn, orig_src, seed in runs:
            path = f"{M}/runs/re1ob_timefix/{fn}"
            j = json.load(open(path))
            cases = {c["case"]: c for c in j["cases"]}
            assert sorted(cases) == sorted(case_dirs("RE1", "OB")), path
            sel = ((df.subsystem_id == "RE1::OB") & (df.method == method) & (df.source_file == orig_src)
                   & (df.seed.astype(str) == str(seed)) & df.case.isin(affected))
            old = df[sel]
            assert len(old) == 50 and old.role.nunique() == 1, (method, fn, len(old))
            role = old.role.iloc[0]
            changed = 0
            for _, r in old.iterrows():
                c = cases[r.case]
                ok = c.get("error") in (None, "None") and int(c.get("n_ranks") or 0) > 0
                s = float(c["acc1"]) if ok else 0.0
                changed += int(s != r.score or int(ok) != r.valid_output)
            log.append((method, seed, changed))
            if changed == 0 and method not in FORCE_REPLACE:
                continue
            df.loc[sel, "role"] = "contrast_release_defect"
            for _, r in old.iterrows():
                c = cases[r.case]
                ok = c.get("error") in (None, "None") and int(c.get("n_ranks") or 0) > 0
                new.append(dict(r.to_dict(), role=role, score=float(c["acc1"]) if ok else 0.0,
                                score_strict=float(c["acc1"]) if ok else 0.0, valid_output=int(ok),
                                source_file=rel(path),
                                input_repr="data.csv (duplicate time column dropped)"))
    print("timefix: cases whose score/validity changed on the 50 affected cases:", log)
    return pd.concat(keep + [pd.DataFrame(new)], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{M}/merged_long.csv")
    a = ap.parse_args()
    build_rcaeval()
    build_openrca()
    build_petshop()
    df = pd.DataFrame(ROWS)
    df = apply_timefix(df)
    key = ["case_id", "method", "role", "input_repr", "window_min", "seed", "impl_source"]
    dup = df.duplicated(key, keep=False)
    assert not dup.any(), df[dup].head()
    df.to_csv(a.out, index=False)
    main_ = df[df.role.isin(["published", "probe", "shipped_baseline"])]
    summ = main_.groupby(["subsystem_id", "method", "seed"], dropna=False).agg(
        n=("score", "size"), coverage=("valid_output", "mean"), score=("score", "mean")).round(3)
    print(f"rows={len(df)} main-table rows={len(main_)}")
    print(df.groupby(["family", "role"]).size().to_string())
    print(summ.to_string())


if __name__ == "__main__":
    main()
