#!/usr/bin/env python3
"""Compute MTEP preset recall from a full-sweep inversion set.

Inputs:
  --full-inv   CSV of inversions from the full T_win sweep (weak_inversions.csv
               or robust_inversions.csv from bootstrap_analysis.py).
  --mtep-inv   CSV of inversions computed AT the MTEP preset tolerances only.
               Must be the output of bootstrap_analysis.py restricted to
               --tolerances 60 300 600.

Output: prints recall = (inversions also present in mtep-inv) / (full inversions).
        Also writes an audit CSV showing which flips were captured.

Capture rule:
  A full-sweep inversion (A,B,t1,t2) is "captured" by MTEP iff in the mtep-inv
  set there exists (A,B,p1,p2) such that:
    - sign(delta_t1) == sign(delta_p1) AND sign(delta_t2) == sign(delta_p2)
    - p1 <= t1 and p2 >= t2   (MTEP presets bracket the flip)
"""
import argparse
import pandas as pd


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-inv", required=True)
    ap.add_argument("--mtep-inv", required=True)
    ap.add_argument("--presets", nargs="+", type=int, default=[60, 300, 600])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    full = pd.read_csv(args.full_inv)
    mtep = pd.read_csv(args.mtep_inv)

    rows = []
    for _, r in full.iterrows():
        A, B = r.method_A, r.method_B
        t1, t2 = int(r.t1_s), int(r.t2_s)
        d1, d2 = r.delta_t1, r.delta_t2

        cand = mtep[(mtep.method_A == A) & (mtep.method_B == B)]
        captured = False
        cap_p1, cap_p2 = None, None
        for _, m in cand.iterrows():
            p1, p2 = int(m.t1_s), int(m.t2_s)
            if sign(m.delta_t1) == sign(d1) and sign(m.delta_t2) == sign(d2) \
               and p1 <= t1 and p2 >= t2:
                captured = True
                cap_p1, cap_p2 = p1, p2
                break
        rows.append({
            "method_A": A, "method_B": B,
            "t1_s": t1, "t2_s": t2,
            "delta_t1": d1, "delta_t2": d2,
            "captured": captured,
            "cap_p1": cap_p1, "cap_p2": cap_p2,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    n_full = len(full)
    n_cap = int(out.captured.sum())
    recall = (n_cap / n_full) if n_full else 0.0
    print(f"Full-sweep inversions: {n_full}")
    print(f"Captured by MTEP presets {args.presets}: {n_cap}")
    print(f"Recall: {recall:.3f}")
    print(f"Audit CSV: {args.out}")


if __name__ == "__main__":
    main()
