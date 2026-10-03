"""RCD on OpenRCA through the audit wrapper (user ruling 2026-09-29, .research/decisions/2026-09-29-openrca-rcd-unfreeze.md).

RCD needs the rcaeval_rcd env (py3.8, RCAEval's patched causal-learn), which cannot read the wrapper's parquet
(no pyarrow). So the run is split; the wrapper logic itself stays in openrca_wrapped.py:
  dump  (auditstack env)  every query through the wrapper with a capturing ranker; the ranker input (metric window
                          incl. time column) -> work/openrca_rcd_inputs/<tag>/row<i>.csv, + index.json with
                          inject_time and sha256 of the CSV text. Queries where the wrapper never calls the
                          ranker (no data) are recorded as such.
  rank  (rcaeval_rcd env) RCAEval rcd on every dumped input, same call as run_rcd_re1.py (dataset=openrca-<tag>,
                          dk_select_useful=False, sli=None, n_iter = n_cols, seed); -> runs/openrca/rcd_ranks/
                          <tag>_seed<s>.json with config + per-row ranks / error / wall_s / input sha.
  then  (auditstack env)  python openrca_wrapped.py --method rcd --seed <s> --tag <tag> --out_dir runs/openrca
                          looks the ranks up by row and checks the input sha, so the prediction, status and
                          reason/time extraction are exactly the wrapper's.
Usage:
  python openrca_rcd.py dump [--procs P]
  python openrca_rcd.py rank --seed S [--procs P]
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

M = "$AUDIT_ROOT"
REPO = "$HOME/rcaeval_repo"
INPUTS = f"{M}/work/openrca_rcd_inputs"
RANKS = f"{M}/runs/openrca/rcd_ranks"
TAGS = ["bank", "mkt1", "mkt2", "tel"]
THIS_FILE = os.path.abspath(__file__)


def csv_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dump_tag(tag):
    sys.path.insert(0, f"{M}/scripts")
    import pandas as pd
    import openrca_wrapped as w
    import predict_baro as pb
    system, q_path = w.SYSTEMS[tag]
    qdf = pd.read_csv(q_path)
    os.makedirs(f"{INPUTS}/{tag}", exist_ok=True)
    got = {}

    def capture(data, inject_time):  # serialise at call time: exactly what the ranker receives
        got["text"], got["shape"], got["it"] = data.to_csv(index=False), list(data.shape), inject_time
        return {"ranks": []}

    pb.baro_inline = capture
    index, cache = {}, {}
    for i, row in qdf.iterrows():
        got.clear()
        pb.predict_one(system, row, cache, "$HOME/auditstack/results_1min",
                       "$HOME/auditstack/results_ext", "$HOME/auditstack/results")
        if "text" not in got:
            index[str(i)] = dict(task_index=row["task_index"], ranker_called=False)
            continue
        with open(f"{INPUTS}/{tag}/row{i}.csv", "w") as f:
            f.write(got["text"])
        index[str(i)] = dict(task_index=row["task_index"], ranker_called=True, inject_time=int(got["it"]),
                             shape=got["shape"], input_sha=csv_sha(got["text"]))
    json.dump(dict(tag=tag, n=len(qdf), dumped_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   script_sha256=hashlib.sha256(open(THIS_FILE, "rb").read()).hexdigest(), rows=index),
              open(f"{INPUTS}/{tag}/index.json", "w"), indent=1)
    print(tag, len(qdf), sum(v["ranker_called"] for v in index.values()), flush=True)


def rank_one(job):
    tag, i, meta, seed = job
    import pandas as pd
    sys.path.insert(0, REPO)
    from RCAEval.e2e import rcd
    t0 = time.time()
    rec = dict(input_sha=meta["input_sha"])
    try:
        text = open(f"{INPUTS}/{tag}/row{i}.csv").read()
        assert csv_sha(text) == meta["input_sha"]
        from io import StringIO
        data = pd.read_csv(StringIO(text))
        out = rcd(data, meta["inject_time"], dataset=f"openrca-{tag}", anomalies=None, dk_select_useful=False, sli=None,
                  verbose=False, n_iter=len(data.columns) - 1, seed=seed)
        rec.update(ranks=[str(r) for r in (out.get("ranks") or [])], error=None)
    except Exception as e:
        rec.update(ranks=[], error="".join(traceback.format_exception(type(e), e, e.__traceback__))[-2000:])
    rec["wall_s"] = round(time.time() - t0, 2)
    return tag, i, rec


def rank(seed, procs):
    import numpy
    import pandas
    os.makedirs(RANKS, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    idx = {t: json.load(open(f"{INPUTS}/{t}/index.json")) for t in TAGS}
    jobs = [(t, i, m, seed) for t in TAGS for i, m in idx[t]["rows"].items() if m["ranker_called"]]
    jobs.sort(key=lambda j: -j[2]["shape"][1])  # widest first
    res = {t: {} for t in TAGS}
    with Pool(procs) as pool:
        for k, (t, i, rec) in enumerate(pool.imap_unordered(rank_one, jobs), 1):
            res[t][i] = rec
            print(f"{k}/{len(jobs)} {t} row{i} err={rec['error'] is not None} n_ranks={len(rec['ranks'])} {rec['wall_s']}s",
                  flush=True)
    import RCAEval
    for t in TAGS:
        cfg = dict(method="rcd", dataset=f"openrca-{t}", seed=seed,
                   input_file=f"{INPUTS}/{t}/row<i>.csv (audit wrapper window, dumped by openrca_rcd.py dump)",
                   code_commit="rcaeval_repo@497505b", rcaeval_path=os.path.dirname(RCAEval.__file__),
                   script_sha256=hashlib.sha256(open(THIS_FILE, "rb").read()).hexdigest(), env=sys.executable,
                   python=sys.version.split()[0], numpy=numpy.__version__, pandas=pandas.__version__,
                   argv=sys.argv, run_started_utc=started, procs=procs,
                   call="rcd(data, inject_time, dataset='openrca-<tag>', anomalies=None, dk_select_useful=False, "
                        "sli=None, verbose=False, n_iter=n_cols, seed=seed)")
        json.dump(dict(config=cfg, rows=res[t]), open(f"{RANKS}/{t}_seed{seed}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dump", "rank"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "dump":
        with Pool(min(a.procs, len(TAGS))) as pool:
            pool.map(dump_tag, TAGS)
    else:
        rank(a.seed, a.procs)


if __name__ == "__main__":
    main()
