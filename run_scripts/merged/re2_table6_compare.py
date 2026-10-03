"""Per-fault AC@1 / AC@3 / Avg@5 of the RE2-TT runs in runs/re2/, next to RCAEval Table 6 (arXiv 2412.17015).

Two denominators per run: all cases (invalid output = 0) and the valid-output subset only.
Also prints the all-fault summary for the RE2-OB / RE2-SS runs (no published per-system table).

Usage (on 3090): python re2_table6_compare.py [runs_dir] > out.md
"""
import glob
import json
import os
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("$AUDIT_ROOT/runs/re2")
FAULTS = ["cpu", "mem", "disk", "socket", "delay", "loss"]
# Table 6, RE2-TT: (AC@1, AC@3, Avg@5) per fault + average, transcribed from the paper text
TABLE6 = {
    "metric BARO": {"cpu": (0.47, 0.8, 0.72), "mem": (0.93, 1, 0.99), "disk": (1, 1, 1), "socket": (0.6, 0.87, 0.83),
                    "delay": (0.47, 0.67, 0.63), "loss": (0.53, 0.6, 0.64), "avg": (0.67, 0.82, 0.8)},
    "metric CIRCA": {"cpu": (0.27, 0.27, 0.28), "mem": (0.47, 0.73, 0.68), "disk": (0.53, 0.67, 0.64),
                     "socket": (0.27, 0.53, 0.52), "delay": (0.2, 0.27, 0.28), "loss": (0.2, 0.33, 0.35),
                     "avg": (0.32, 0.47, 0.46)},
    "MicroRank": {"cpu": (0.21, 0.43, 0.34), "mem": (0.25, 0.38, 0.33), "disk": (0, 0.36, 0.27), "socket": (0.3, 0.4, 0.36),
                  "delay": (0.08, 0.31, 0.23), "loss": (0.14, 0.36, 0.3), "avg": (0.16, 0.37, 0.31)},
    "TraceRCA": {"cpu": (0.64, 0.79, 0.74), "mem": (0.63, 0.88, 0.83), "disk": (0.64, 0.71, 0.74), "socket": (0.6, 0.8, 0.76),
                 "delay": (0.85, 0.85, 0.88), "loss": (0.57, 0.71, 0.67), "avg": (0.66, 0.79, 0.77)},
    "multi-source BARO": {"cpu": (0.47, 0.8, 0.75), "mem": (0.93, 1, 0.99), "disk": (1, 1, 1), "socket": (0.6, 0.8, 0.79),
                          "delay": (0.47, 0.67, 0.61), "loss": (0.67, 0.67, 0.71), "avg": (0.69, 0.82, 0.81)},
}
PAPER_ROW = {"circa": "metric CIRCA", "microrank": "MicroRank", "tracerca": "TraceRCA", "mmbaro": "multi-source BARO"}


def stats(rows):
    n = len(rows)
    if n == 0:
        return None
    return (sum(r["acc1"] for r in rows) / n, sum(r["acc3"] for r in rows) / n, sum(r["avg5"] for r in rows) / n, n)


def fmt(t):
    return "–" if t is None else f"{t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f} (n={t[3]})"


def main():
    files = sorted(glob.glob(os.path.join(RUNS, "*.json")))
    print(f"# RE2 runs vs RCAEval Table 6\n\nsource dir: `{RUNS}`; cells = AC@1 / AC@3 / Avg@5\n")
    print("## RE2-TT, per fault\n")
    for f in files:
        r = json.load(open(f))
        if r["config"]["dataset"] != "re2-tt":
            continue
        cases = r["cases"]
        valid = [c for c in cases if c["n_ranks"] > 0]
        paper = TABLE6.get(PAPER_ROW.get(r["config"]["method"], ""), {})
        print(f"### {os.path.basename(f)}\n")
        print(f"coverage {len(valid)}/{len(cases)}, errors {r['n_errors']}\n")
        print("| fault | all cases (invalid = 0) | valid-output subset | Table 6 |")
        print("|---|---|---|---|")
        for flt in FAULTS + ["avg"]:
            sub = cases if flt == "avg" else [c for c in cases if c["fault"] == flt]
            vsub = [c for c in sub if c["n_ranks"] > 0]
            p = paper.get(flt)
            print(f"| {flt} | {fmt(stats(sub))} | {fmt(stats(vsub))} | {'–' if p is None else ' / '.join(map(str, p))} |")
        print()
    print("## RE2-OB / RE2-SS, all faults\n")
    print("| run | all cases (invalid = 0) | valid-output subset |")
    print("|---|---|---|")
    for f in files:
        r = json.load(open(f))
        if r["config"]["dataset"] == "re2-tt":
            continue
        cases = r["cases"]
        print(f"| {os.path.basename(f)} | {fmt(stats(cases))} | {fmt(stats([c for c in cases if c['n_ranks'] > 0]))} |")


if __name__ == "__main__":
    main()
