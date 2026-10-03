"""Idempotently add/refresh ledger rows for new runs under $AUDIT_ROOT/runs/.
Rows for the same path are replaced; all other ledger rows are left untouched."""
import csv, glob, hashlib, json, os

M = os.path.expanduser("$AUDIT_ROOT")
L = f"{M}/runs_ledger.csv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def status_for(rel, cfg):
    b = os.path.basename(rel)
    if rel.startswith(("runs/baro/", "runs/probes/")) and "_datacsv_" in b:
        return "contaminated"  # raw-export data.csv on RE1-SS/TT: RQ2(a) contrast only
    if rel.startswith("runs/probes/") and "_metricscsv_" in b:
        return "diagnostic"  # RE2 metrics.csv instead of simple_metrics.csv: input contrast
    if rel.startswith("runs/re2/"):
        # primary = RE2 protocol (simple_metrics, 20 min); mmbaro primary uses the dataset name that
        # enables its trace branch ("mm-*"); other variants are sensitivity/diagnostic runs
        if b.endswith("_len10.json") or "_name-re2" in b or "metricscsv" in b:
            return "diagnostic"
        return "valid"
    if rel.startswith(("runs/baro/", "runs/probes/")):
        # RE1 window fixed at 20 min (decision 2026-09-27); len10 = A long-table window, kept as reproduction check
        return "diagnostic" if b.endswith("_len10.json") else "valid"
    if rel.startswith("runs/openrca/rcd_ranks/"):
        return "diagnostic"  # intermediate RCD ranks; the wrapper predictions built from them are the valid rows
    if rel.startswith("runs/openrca/"):
        return "diagnostic" if b.startswith("baro_") else "valid"  # baro = harness reproduction check
    if rel.startswith("runs/causalrca/"):
        return "diagnostic"
    if "_datacsv_" in b:
        return "contaminated"  # raw-export data.csv on RE1-SS/TT: RQ2(a) contrast only
    if b.startswith(("barodk_", "simplerca_author_nologs_")) or b.endswith("_winfull.json"):
        return "diagnostic"
    if b.startswith("baro_author-re3ssbranch") or b.startswith("simplerca_author-re3ssbranch"):
        return "valid"
    if cfg.get("dataset") == "rcaeval_re3_ss" and str(cfg.get("author_branch_commit", "")).startswith("author_rcaeval:"):
        return "diagnostic"  # rcaeval-branch run on RE3-SS; Fang Table 2 used the rcaeval_re3_ss branch
    return "valid"


with open(L) as f:
    rd = csv.DictReader(f)
    keys = rd.fieldnames
    old = list(rd)

new = []
for p in sorted(glob.glob(f"{M}/runs/**/*.json", recursive=True)):
    rel = os.path.relpath(p, M)
    d = json.load(open(p))
    cfg = d.get("config", {})
    if rel.startswith("runs/causalrca/"):
        s = d["summary"]
        n, acc1, cov, commit = s["n_cases"], "", f"{s['n_ok']}/{s['n_cases']}", cfg.get("code_commit")
        note = "column-permutation test; " + "; ".join(f"{k}={v}" for k, v in s.items())
    elif rel.startswith("runs/openrca/rcd_ranks/"):
        n, acc1, commit = len(d["rows"]), "", cfg.get("code_commit")
        cov = f"{sum(r['error'] is None for r in d['rows'].values())}/{n}"
        note = ("RCD ranks on the wrapper's dumped inputs (rcaeval_rcd env), intermediate; predictions in "
                f"runs/openrca/rcd-seed{cfg.get('seed')}_openrca_<tag>.json")
    elif rel.startswith("runs/openrca/"):
        n, acc1, commit = d["n"], "", cfg.get("code_commit")
        cov = f"{d['status_counts'].get('method', 0)}/{n}"
        note = "audit OpenRCA wrapper; status " + json.dumps(d["status_counts"]) + "; scored by score_openrca_wrapped.py"
    elif rel.startswith("runs/probes/"):
        n, acc1, cov, commit = d["n"], d["acc1"], f"{d['coverage']}/{d['n']}", cfg.get("code_commit")
        note = f"length_min={cfg.get('length_min')}; audit probe code and its own acc_at_k scorer"
    elif "summary" in d:  # author SimpleRCA harness
        s = d["summary"]
        n, acc1 = s["n"], s["AC@1"]
        cov = f"{n - s['n_empty']}/{n}"
        commit = cfg.get("author_branch_commit")
        note = f"converter={cfg.get('converter')}; exact-match scoring (platform rule)"
    else:  # rcd
        n, acc1 = d["n"], d["acc1"]
        cov = f"{d['coverage']}/{n}"
        commit = cfg.get("code_commit")
        note = (f"seed={cfg.get('seed')}; " if cfg.get("seed") is not None else
                f"length_min={cfg.get('length_min')}; mm_dataset_name={cfg.get('mm_dataset_name')}; ") + "RCAEval -db fold scoring"
    new.append(dict(path=rel, source="merged_runs", method=cfg.get("method"), dataset=cfg.get("dataset"),
                    variant=cfg.get("author_branch_commit") or (f"seed{cfg.get('seed')}" if cfg.get("seed") is not None else f"len{cfg.get('length_min')}"),
                    input_file=cfg.get("input_file"), code_commit=commit, env=cfg.get("env"),
                    run_utc=cfg.get("run_started_utc"), n_cases=n, coverage=cov, acc1_recomputed=acc1,
                    status=status_for(rel, cfg), note=note, sha12=sha(p), bytes=os.path.getsize(p)))

paths = {r["path"] for r in new}
rows = [r for r in old if r["path"] not in paths] + new
with open(L, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"ledger: {len(old)} -> {len(rows)} rows ({len(new)} from runs/)")
