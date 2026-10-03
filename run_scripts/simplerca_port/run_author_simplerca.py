"""Run the author-repo SimpleRCA (rcaeval branches of LGU-SE-Internal/Simple-RCA) on RCAEval RE2/RE3.

Algorithm code under author_rcaeval/ (branch rcaeval @ 5fd1f7f) and author_re3ss/ (branch
rcaeval_re3_ss @ 7f7f295) is copied unmodified from the author repository. On those branches
main.py registers "simplerca" = NezhaRCA (BARO metric ranking + trace/log bonus for BARO top-3)
and "baro" = Baro.

Only the I/O layer is ours:
  * CSV -> parquet conversion uses a verbatim copy of rcabench_platform 0.3.27
    sources/rcaeval.py (first release with logs.parquet / attr.has_error, which nezha_rca.py
    reads), including the same schema validation the platform applies before saving.
  * Datapack enumeration and labels follow RcaevalDatasetLoader (label = folder.split("_")[0]).
  * Hits/metrics follow rcabench_platform experiments/single.py + evaluation/ranking.py
    (exact (level, name) match; empty answer list -> one null answer at rank 1).

Usage: python run_author_simplerca.py --dataset rcaeval_re2_ob --algo simplerca --out X.json
       [--no-logs]  (sensitivity: logs.parquet absent, as with rcabench_platform <= 0.3.26)
"""

import argparse
import datetime
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(HERE))

import polars as pl  # noqa: E402

from rcabench_platform.v2.algorithms.spec import AlgorithmArgs  # noqa: E402
from rcabench_platform.v2.sources import rcaeval as conv  # noqa: E402

DATA_ROOT = Path.home() / "rcaeval_repo" / "data"
DATASET_DIRS = {
    "rcaeval_re2_ob": DATA_ROOT / "RE2" / "RE2-OB",
    "rcaeval_re2_ss": DATA_ROOT / "RE2" / "RE2-SS",
    "rcaeval_re2_tt": DATA_ROOT / "RE2" / "RE2-TT",
    "rcaeval_re3_ob": DATA_ROOT / "RE3" / "RE3-OB",
    "rcaeval_re3_ss": DATA_ROOT / "RE3" / "RE3-SS",
    "rcaeval_re3_tt": DATA_ROOT / "RE3" / "RE3-TT",
    # RE1 has metrics only; the author repo never ran on it. We feed the metric table through the
    # same converter as simple_metrics.csv (OB ships only data.csv, already the reduced table).
    "rcaeval_re1_ob": DATA_ROOT / "online-boutique",
    "rcaeval_re1_ss": DATA_ROOT / "sock-shop-2",
    "rcaeval_re1_tt": DATA_ROOT / "train-ticket",
}
RE1_METRIC_FILE = {"rcaeval_re1_ob": "data.csv", "rcaeval_re1_ss": "simple_data.csv", "rcaeval_re1_tt": "simple_data.csv"}
AUTHOR_COMMITS = {"author_rcaeval": "5fd1f7f", "author_re3ss": "7f7f295"}


# ---- verbatim copy of rcabench_platform 0.3.27 sources/convert.py validators ----
def validate_by_model(df, model, extra_prefix):
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    for name, dtype in model.items():
        if name not in schema:
            raise ValueError(f"Missing required column: {name} {dtype}")
        if schema[name] != dtype:
            raise ValueError(f"Column {name} has incorrect type: {schema[name]}")
    for name, dtype in schema.items():
        if name not in model and not name.startswith(extra_prefix):
            raise ValueError(f"Unexpected column: {name} {dtype}")


VALIDATORS = {
    "traces": {"time": pl.Datetime, "trace_id": pl.String, "span_id": pl.String, "parent_span_id": pl.String,
               "service_name": pl.String, "span_name": pl.String, "duration": pl.UInt64},
    "metrics": {"time": pl.Datetime, "metric": pl.String, "value": pl.Float64, "service_name": pl.String},
    "logs": {"time": pl.Datetime, "trace_id": pl.String, "span_id": pl.String, "service_name": pl.String,
             "level": pl.String, "message": pl.String},
}


def list_datapacks(src_folder: Path):
    """Same enumeration as RcaevalDatasetLoader.__init__ (sorted for determinism)."""
    packs = []
    for service_path in sorted(src_folder.iterdir()):
        if not service_path.is_dir():
            continue
        for num_path in sorted(service_path.iterdir()):
            if not num_path.is_dir() or num_path.name == "multi-source-data":
                continue
            packs.append((f"{service_path.name}_{num_path.name}", num_path, service_path.name.split("_")[0]))
    return packs


def re1_metrics(src: Path, dataset: str, window: str) -> pl.LazyFrame:
    """RE1 metric table through the verbatim converter, optionally cut to the RE1 protocol window
    (last/first 600 rows around inject_time, as rcaeval_repo/_baseline_utils.load_case length_min=20).
    The cut is applied on time after conversion so column dtypes are inferred from the full file."""
    path = src / RE1_METRIC_FILE[dataset]
    lf = conv.convert_metrics(path)
    if window == "full":
        return lf
    inject = int((src / "inject_time.txt").read_text().strip())
    t = pl.read_csv(path, columns=["time"])["time"]
    kept = pl.concat([t.filter(t < inject).tail(600), t.filter(t >= inject).head(600)])
    lo = datetime.datetime.fromtimestamp(int(kept.min()), tz=datetime.timezone.utc)
    hi = datetime.datetime.fromtimestamp(int(kept.max()), tz=datetime.timezone.utc)
    return lf.filter((pl.col("time") >= lo) & (pl.col("time") <= hi))


def convert_datapack(src: Path, dst: Path, no_logs: bool, dataset: str = "", re1_window: str = "full"):
    if dataset.startswith("rcaeval_re1"):
        data = {"inject_time.txt": src / "inject_time.txt",
                "simple_metrics.parquet": re1_metrics(src, dataset, re1_window)}
    else:
        loader = conv.RcaevalDatapackLoader(src_folder=src, dataset="", datapack="", service="")
        data = loader.data()
    if no_logs:
        data.pop("logs.parquet", None)
    for k, v in data.items():
        stem = Path(k).stem
        for key, model in VALIDATORS.items():
            if stem.endswith(key):
                validate_by_model(v, model, extra_prefix="attr.")
        if isinstance(v, Path):
            shutil.copyfile(v, dst / k)
        else:
            v.sink_parquet(dst / k)


def run_one(job):
    dataset, algo, variant, no_logs, re1_window, (datapack, src, label) = job
    nr = importlib.import_module(f"{variant}.nezha_rca")
    baro_mod = importlib.import_module(f"{variant}.baro")
    rec = {"datapack": datapack, "label": label, "src": str(src)}
    branches = {}
    orig = nr.nezha_rca

    def recording_nezha_rca(**kw):
        out = orig(**kw)
        # Re-run the branch detectors on the same inputs purely for logging; ranks come from `out`.
        branches["trace_top5"] = nr._detect_trace_anomalies(kw["normal_traces"], kw["abnormal_traces"])
        branches["log_top5"] = nr._detect_log_anomalies(kw["normal_logs"], kw["abnormal_logs"])
        try:
            mr = baro_mod.baro(kw["data"], kw["inject_time"], dataset=kw["dataset"],
                               dk_select_useful=(variant == "author_re3ss" and kw["dataset"] == "rcaeval_re3_ss"))
            branches["metric_top5"] = [[s, float(v)] for s, v in nr._extract_service_scores(mr)]
        except Exception as e:  # mirrors the try/except in nezha_rca
            branches["metric_error"] = repr(e)
        branches["n_abnormal_logs"] = int(len(kw["abnormal_logs"]))
        branches["n_abnormal_traces"] = int(len(kw["abnormal_traces"]))
        return out

    tmp = Path(tempfile.mkdtemp(prefix="srca_", dir=os.environ.get("SRCA_TMP", "/tmp")))
    t0 = time.time()
    try:
        convert_datapack(src, tmp, no_logs, dataset, re1_window)
        args = AlgorithmArgs(dataset=dataset, datapack=datapack, input_folder=tmp, output_folder=tmp)
        t1 = time.time()
        if algo == "simplerca":
            nr.nezha_rca = recording_nezha_rca
            answers = nr.NezhaRCA()(args)
        elif algo == "baro":
            answers = baro_mod.Baro()(args)
        elif algo == "baro_dk":
            # Diagnostic only: the author's Baro adapter with dk_select_useful=True, i.e. the setting
            # that author_re3ss/nezha_rca.py uses internally for rcaeval_re3_ss.
            from functools import partial
            answers = baro_mod.SimpleMetricsAdapter(partial(baro_mod.baro, dk_select_useful=True))(args)
        else:
            raise ValueError(algo)
        rec["runtime_s"] = time.time() - t1
        rec["exception"] = None
    except Exception as e:
        answers = []
        rec["runtime_s"] = None
        rec["exception"] = "".join(traceback.format_exception(e))
    finally:
        nr.nezha_rca = orig
        shutil.rmtree(tmp, ignore_errors=True)
    rec["convert_plus_run_s"] = time.time() - t0
    answers = sorted(answers, key=lambda a: a.rank)
    ans = [{"level": a.level, "name": a.name, "rank": a.rank} for a in answers]
    if not ans:
        ans = [{"level": None, "name": None, "rank": 1}]
    rec["answers"] = ans
    rec["hits"] = [(a["level"], a["name"]) == ("service", label) for a in ans]
    rec.update(branches)
    return rec


def summarize(recs):
    """AC@k / Avg@k as in evaluation/ranking.py; MRR both platform-style (hit cases only) and over all cases."""
    n = len(recs)
    ac = {}
    for k in range(1, 6):
        ac[k] = sum(any(h and a["rank"] <= k for h, a in zip(r["hits"], r["answers"])) for r in recs) / n
    first = [min((a["rank"] for h, a in zip(r["hits"], r["answers"]) if h), default=None) for r in recs]
    hit_rr = [1 / f for f in first if f is not None]
    return {
        "n": n,
        **{f"AC@{k}": round(ac[k], 6) for k in ac},
        "Avg@3": round(sum(ac[i] for i in (1, 2, 3)) / 3, 6),
        "Avg@5": round(sum(ac[i] for i in range(1, 6)) / 5, 6),
        "MRR_platform_hit_only": round(sum(hit_rr) / len(hit_rr), 6) if hit_rr else 0.0,
        "MRR_all_cases": round(sum(hit_rr) / n, 6),
        "n_exceptions": sum(r["exception"] is not None for r in recs),
        "n_empty": sum(r["answers"][0]["name"] is None for r in recs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_DIRS))
    ap.add_argument("--algo", required=True, choices=["simplerca", "baro", "baro_dk"])
    ap.add_argument("--variant", default="author_rcaeval", choices=sorted(AUTHOR_COMMITS))
    ap.add_argument("--no-logs", action="store_true")
    ap.add_argument("--re1-window", default="full", choices=["full", "20"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    packs = list_datapacks(DATASET_DIRS[a.dataset])
    if a.limit:
        packs = packs[: a.limit]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    jobs = [(a.dataset, a.algo, a.variant, a.no_logs, a.re1_window, p) for p in packs]
    with Pool(a.procs) as pool:
        recs = pool.map(run_one, jobs, chunksize=1)

    import hashlib
    code_sha = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()[:12]
                for p in sorted(HERE.rglob("*.py")) if "__pycache__" not in p.parts}
    out = {
        "config": {
            "method": {"simplerca": "simplerca-author", "baro": "baro-author-repo", "baro_dk": "baro-author-repo-dk_select_useful"}[a.algo],
            "dataset": a.dataset,
            "input_file": (f"{RE1_METRIC_FILE[a.dataset]} (window={a.re1_window})" if a.dataset in RE1_METRIC_FILE
                           else "simple_metrics.csv + traces.csv + logs.csv" + (" (logs dropped)" if a.no_logs else "")),
            "author_repo": "LGU-SE-Internal/Simple-RCA",
            "author_branch_commit": f"{a.variant}:{AUTHOR_COMMITS[a.variant]}",
            "converter": "rcabench_platform==0.3.27 sources/rcaeval.py (verbatim)",
            "no_logs": a.no_logs,
            "code_sha256_12": code_sha,
            "env": sys.executable,
            "python": sys.version.split()[0],
            "polars": pl.__version__,
            "argv": sys.argv,
            "run_started_utc": started,
            "limit": a.limit,
        },
        "summary": summarize(recs),
        "cases": recs,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"dataset": a.dataset, "algo": a.algo, "variant": a.variant, **out["summary"]}))


if __name__ == "__main__":
    main()
