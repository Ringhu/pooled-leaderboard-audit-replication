#!/usr/bin/env python3
"""Post-repair CIRCA stats: modal top-1 freq + component-acc@1 on the
queries whose scoring_points include a root-cause-component criterion.
"""
import collections
import re

import pandas as pd

PRED_DIR = "$HOME/auditstack/results_circa_repair"
QDIR = "$HOME/auditstack/scripts/openrca_eval"

SYSTEMS = [
    ("Bank",               f"{PRED_DIR}/circa_repaired_bank_results.csv", f"{QDIR}/bank_q.csv"),
    ("Market_cloudbed-1",  f"{PRED_DIR}/circa_repaired_mkt1_results.csv", f"{QDIR}/mkt1_q.csv"),
    ("Market_cloudbed-2",  f"{PRED_DIR}/circa_repaired_mkt2_results.csv", f"{QDIR}/mkt2_q.csv"),
    ("Telecom",            f"{PRED_DIR}/circa_repaired_tel_results.csv",  f"{QDIR}/tel_q.csv"),
]


def extract_top1(pred_str: str) -> str:
    m = re.search(r'"root cause component":\s*"([^"]+)"', str(pred_str))
    return m.group(1) if m else ""


def all_true_components(scoring_points: str) -> list[str]:
    pattern = (
        r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
    )
    return [c.strip() for c in re.findall(pattern, str(scoring_points))]


def main():
    print(f"{'system':<22} {'n_total':<8} {'n_comp':<8} "
          f"{'modal_top1':<22} {'modal_freq':<12} {'acc@1':<10}")
    for sys_name, pred_csv, q_csv in SYSTEMS:
        pdf = pd.read_csv(pred_csv)
        qdf = pd.read_csv(q_csv)
        merged = pdf.merge(qdf, on="task_index", how="left")
        preds_top1 = merged["prediction"].map(extract_top1)
        counter = collections.Counter(preds_top1)
        modal = counter.most_common(1)[0] if counter else ("", 0)
        modal_freq = modal[1] / max(len(merged), 1)

        # acc@1 on cases that actually have a labelled component.
        n_with_comp = 0
        hits = 0
        for _, row in merged.iterrows():
            trues = all_true_components(row["scoring_points"])
            if not trues:
                continue
            n_with_comp += 1
            top = extract_top1(row["prediction"])
            if top in set(trues):
                hits += 1
        acc = hits / n_with_comp if n_with_comp else float("nan")

        print(f"{sys_name:<22} {len(merged):<8} {n_with_comp:<8} "
              f"{modal[0][:20]:<22} {modal_freq:<12.2%} {acc:<10.2%}")


if __name__ == "__main__":
    main()
