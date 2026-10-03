"""Score OpenRCA prediction CSVs with the official evaluate() at 60 s tolerance (audit convention).

For every (method, tag) prediction file:
  score           official partial score (the long-table `correct` column for OpenRCA)
  strict          score == 1 (OpenRCA report() "Correct")
  score_fb0       score, but queries answered by the wrapper's fixed default (status != "method") set to 0
Per-query rows go to runs/openrca/scores_long.csv; a summary table is printed.

Check mode: also scores A's prediction files (A_FILES) and compares with the long table
(raw/earlier_a_followup_data/full_coverage_cd1min_long_CORRECTED.csv, same method), matched by row position
(case = <tag>/<task_index>, q_idx = cumcount within task_index).
"""
import json
import os
import sys

import pandas as pd

A_EVAL = "$HOME/auditstack/scripts/openrca_eval"
sys.path.insert(0, A_EVAL)
from openrca_evaluate_sweep import evaluate  # noqa: E402

M = "$AUDIT_ROOT"
OPEN_RCA = "$HOME/auditstack/data/OpenRCA/dataset"
Q = {"bank": f"{OPEN_RCA}/Bank/query.csv", "mkt1": f"{OPEN_RCA}/Market/cloudbed-1/query.csv",
     "mkt2": f"{OPEN_RCA}/Market/cloudbed-2/query.csv", "tel": f"{OPEN_RCA}/Telecom/query.csv"}
LONG = f"{M}/raw/earlier_a_followup_data/full_coverage_cd1min_long_CORRECTED.csv"
# long-table method -> A prediction file it was scored from
A_FILES = {"baro": "full335/preds_baro_{tag}.csv", "alertcount": "preds_alertcount_{tag}.csv",
           "cd1min": "full335/preds_1min_{tag}.csv", "z_family": "preds_zbase_univ_{tag}.csv"}
# harness run -> A prediction file it must reproduce
REPRO = {"baro": "full335/preds_baro_{tag}.csv", "alertcount-unfixed": "preds_alertcount_{tag}.csv"}


def score_file(pred_csv, tag):
    preds, q = pd.read_csv(pred_csv), pd.read_csv(Q[tag])
    assert len(preds) == len(q) and (preds.task_index == q.task_index).all(), pred_csv
    q = q.assign(q_idx=q.groupby("task_index").cumcount())
    s = [float(evaluate(str(p), str(sp), time_tolerance_seconds=60)[2]) for p, sp in zip(preds.prediction, q.scoring_points)]
    return pd.DataFrame({"tag": tag, "case": tag + "/" + q.task_index, "q_idx": q.q_idx, "score": s})


def main():
    long = pd.read_csv(LONG)
    long = long[long.benchmark == "openrca"]
    for lm, a_file in A_FILES.items():
        print(f"## check: A predictions {a_file.format(tag='<tag>')} vs long table method={lm}")
        for tag in Q:
            d = score_file(f"{A_EVAL}/{a_file.format(tag=tag)}", tag)
            m = d.merge(long[long.method == lm][["case", "q_idx", "correct"]], on=["case", "q_idx"], how="left")
            print(f"{tag}: n={len(d)} matched={m.correct.notna().sum()} equal={(abs(m.score - m.correct) < 1e-6).sum()} "
                  f"mean={d.score.mean():.4f} long_mean={m.correct.mean():.4f}")

    rows = []
    for method in ("baro", "ediag", "simplerca", "alertcount", "alertcount-unfixed", "rcd-seed0", "rcd-seed1", "rcd-seed2"):
        for tag in Q:
            stem = f"{M}/runs/openrca/{method}_openrca_{tag}"
            if not os.path.exists(stem + ".json"):
                continue
            meta = json.load(open(stem + ".json"))
            d = score_file(f"{M}/runs/openrca/preds_{method}_openrca_{tag}.csv", tag)
            d["status"] = [c["status"] for c in meta["cases"]]
            d["method"] = method
            if method in REPRO:
                a = pd.read_csv(f"{A_EVAL}/{REPRO[method].format(tag=tag)}").prediction
                b = pd.read_csv(f"{M}/runs/openrca/preds_{method}_openrca_{tag}.csv").prediction
                print(f"harness {method} {tag}: identical to A predictions {(a == b).sum()}/{len(a)}")
            rows.append(d)
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    df["strict"] = (df.score == 1.0).astype(float)
    df["score_fb0"] = df.score.where(df.status == "method", 0.0)
    df.to_csv(f"{M}/runs/openrca/scores_long.csv", index=False)
    g = df.groupby(["method", "tag"])
    summ = pd.DataFrame({"n": g.size(), "method_output": g.status.apply(lambda s: (s == "method").sum()),
                         "score": g.score.mean(), "strict": g.strict.mean(), "score_fb0": g.score_fb0.mean(),
                         "fallback_score_sum": g.apply(lambda x: x.score[x.status != "method"].sum())})
    print("\n## summary\n" + summ.round(4).to_string())
    print("\n## status counts\n" + df.groupby(["method", "tag"]).status.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
