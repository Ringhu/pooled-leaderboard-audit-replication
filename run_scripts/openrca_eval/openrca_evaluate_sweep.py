#!/usr/bin/env python3
"""
Tolerance-aware OpenRCA evaluator.

Fork of openrca_evaluate.py with parameterized time tolerance
for the Minute Matters pilot experiment.

Usage:
  # Single tolerance (drop-in replacement for original)
  python openrca_evaluate_sweep.py -p preds.csv -q query.csv -r report.csv --tolerance 60

  # Sweep multiple tolerances, output summary
  python openrca_evaluate_sweep.py -p preds.csv -q query.csv --tolerances 30,60,120,300,600 --summary sweep_out.csv

  # Batch: multiple prediction files
  python openrca_evaluate_sweep.py \
    -p preds_bank.csv preds_tel.csv \
    -q query_bank.csv query_tel.csv \
    --labels Bank Telecom \
    --tolerances 30,60,120,300,600 \
    --summary sweep_summary.csv
"""

import sys
import os
import re
import argparse
import csv
from datetime import datetime
import itertools

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)


def evaluate(prediction: str, scoring_points: str, time_tolerance_seconds: int = 60):
    """
    Evaluate single prediction against scoring points.

    Args:
        prediction: JSON-like prediction string
        scoring_points: ground truth scoring criteria string
        time_tolerance_seconds: max allowed time deviation in seconds (default: 60)

    Returns:
        (passing_criteria, failing_criteria, score)
    """
    predict_pattern = (
        r'{\s*'
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )
    predict_matches = re.findall(predict_pattern, prediction)

    predict_results = []
    for match in predict_matches:
        datetime_str, component, reason = match
        predict_results.append({
            "root cause occurrence datetime": datetime_str,
            "root cause component": component,
            "root cause reason": reason
        })

    prediction_length = len(predict_results)

    component_pattern = r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
    reason_pattern = r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
    time_pattern = r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)"

    components = re.findall(component_pattern, scoring_points)
    reasons = re.findall(reason_pattern, scoring_points)
    times = re.findall(time_pattern, scoring_points)

    scoringpoints_length = max(len(components), len(reasons), len(times))
    socres_num = len(components) + len(reasons) + len(times)

    def check_time(time1_str, time2_str):
        time_format = "%Y-%m-%d %H:%M:%S"
        try:
            t1 = datetime.strptime(time1_str, time_format)
            t2 = datetime.strptime(time2_str, time_format)
        except ValueError:
            return False
        return abs(t1 - t2).total_seconds() <= time_tolerance_seconds

    scores_get = 0
    passing_criteria = []
    failing_criteria = []

    if scoringpoints_length == prediction_length:
        best_score = -1
        for perm in itertools.permutations(predict_results):
            current_score = 0
            current_passing = []
            for i in range(scoringpoints_length):
                if len(components) == scoringpoints_length:
                    if perm[i]['root cause component'] == components[i]:
                        current_score += 1
                        current_passing.append(components[i])
                if len(reasons) == scoringpoints_length:
                    if perm[i]['root cause reason'] == reasons[i]:
                        current_score += 1
                        current_passing.append(reasons[i])
                if len(times) == scoringpoints_length:
                    if check_time(times[i], perm[i]['root cause occurrence datetime']):
                        current_score += 1
                        current_passing.append(times[i])
            if current_score > best_score:
                best_score = current_score
                passing_criteria = current_passing
        scores_get = best_score

    failing_criteria = list(set(components + reasons + times) - set(passing_criteria))

    if socres_num == 0:
        return passing_criteria, failing_criteria, 0.0

    final_score = scores_get / socres_num
    return passing_criteria, failing_criteria, round(final_score, 2)


def evaluate_file_at_tolerance(prediction_file, query_file, tolerance_s):
    """
    Evaluate a prediction file at a specific tolerance.

    Returns:
        dict with correct_count, partial_count, total, correct_pct, partial_pct
    """
    import pandas as pd

    pred_df = pd.read_csv(prediction_file)
    query_df = pd.read_csv(query_file)

    if len(pred_df) != len(query_df):
        raise ValueError(
            f"Length mismatch: {prediction_file} has {len(pred_df)} rows, "
            f"{query_file} has {len(query_df)} rows"
        )

    correct = 0
    partial = 0
    total = len(pred_df)

    for idx in range(total):
        prediction = pred_df.loc[idx, "prediction"]
        scoring_points = query_df.loc[idx, "scoring_points"]
        _, _, score = evaluate(prediction, scoring_points, time_tolerance_seconds=tolerance_s)
        if score == 1.0:
            correct += 1
        if score > 0:
            partial += 1

    return {
        "correct": correct,
        "partial": partial,
        "total": total,
        "correct_pct": round(100 * correct / total, 2) if total > 0 else 0,
        "partial_pct": round(100 * partial / total, 2) if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Tolerance-aware OpenRCA evaluator")
    parser.add_argument("-p", type=str, nargs='+', required=True,
                        help="Prediction file(s)")
    parser.add_argument("-q", type=str, nargs='+', required=True,
                        help="Query file(s) (ground truth)")
    parser.add_argument("-r", type=str, default=None,
                        help="Report file (detailed per-case, single tolerance only)")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
                        help="Labels for each prediction file (e.g., Bank Telecom)")
    parser.add_argument("--tolerance", type=int, default=None,
                        help="Single time tolerance in seconds")
    parser.add_argument("--tolerances", type=str, default=None,
                        help="Comma-separated list of tolerances (e.g., 30,60,120,300,600)")
    parser.add_argument("--summary", type=str, default=None,
                        help="Output CSV for sweep summary")
    parser.add_argument("--method", type=str, default="unknown",
                        help="Method name for summary CSV")
    args = parser.parse_args()

    if len(args.p) != len(args.q):
        raise ValueError("Number of prediction files must match query files")

    # Determine tolerances to sweep
    if args.tolerances:
        tolerances = [int(t) for t in args.tolerances.split(",")]
    elif args.tolerance:
        tolerances = [args.tolerance]
    else:
        tolerances = [60]  # default: original behavior

    # Labels
    if args.labels:
        labels = args.labels
    else:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.p]

    if len(labels) != len(args.p):
        raise ValueError("Number of labels must match prediction files")

    # Run sweep
    results = []
    for pred_file, query_file, label in zip(args.p, args.q, labels):
        for tol in tolerances:
            r = evaluate_file_at_tolerance(pred_file, query_file, tol)
            row = {
                "method": args.method,
                "system": label,
                "tolerance_s": tol,
                "correct": r["correct"],
                "partial": r["partial"],
                "total": r["total"],
                "correct_pct": r["correct_pct"],
                "partial_pct": r["partial_pct"],
            }
            results.append(row)
            print(f"  {args.method:20s} | {label:12s} | tol={tol:>4d}s | "
                  f"Correct={r['correct_pct']:5.1f}% ({r['correct']}/{r['total']}) | "
                  f"Partial={r['partial_pct']:5.1f}% ({r['partial']}/{r['total']})")

    # Write summary CSV
    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
        file_exists = os.path.exists(args.summary)
        with open(args.summary, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "method", "system", "tolerance_s",
                "correct", "partial", "total",
                "correct_pct", "partial_pct"
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary appended to {args.summary}")

    # Backward-compatible detailed report (single tolerance only)
    if args.r and len(tolerances) == 1:
        import pandas as pd
        tol = tolerances[0]
        if os.path.exists(args.r):
            os.remove(args.r)
        for pred_file, query_file in zip(args.p, args.q):
            pred_df = pd.read_csv(pred_file)
            query_df = pd.read_csv(query_file)
            eval_rows = []
            for idx in range(len(pred_df)):
                prediction = pred_df.loc[idx, "prediction"]
                scoring_points = query_df.loc[idx, "scoring_points"]
                passing, failing, score = evaluate(prediction, scoring_points, tol)
                eval_rows.append({
                    "query": query_df.loc[idx, "instruction"],
                    "answer": prediction,
                    "groundtruth": scoring_points,
                    "passed": passing,
                    "failed": failing,
                    "score": score,
                    "task_index": query_df.loc[idx, "task_index"],
                })
            eval_df = pd.DataFrame(eval_rows)
            if os.path.exists(args.r):
                eval_df.to_csv(args.r, mode='a', header=False, index=False)
            else:
                eval_df.to_csv(args.r, index=False)
        print(f"\nDetailed report saved to {args.r}")


if __name__ == "__main__":
    main()
