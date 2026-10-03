"""G1: compare author-repo SimpleRCA/BARO runs with Fang et al. FSE'26 Table 2 (arXiv 2510.04711v2).
Paper values transcribed from RCAAgent/.research/papers/2510.04711v2.txt lines 528-750."""
import json

R = "$AUDIT_ROOT/runs/simplerca_author"
PAPER = {  # Top@1 Top@3 Top@5 Avg@3 Avg@5 MRR
    ("re2_ob", "baro"): (0.14, 0.93, 0.98, 0.64, 0.77, 0.54), ("re2_ob", "simplerca"): (0.47, 0.93, 1.00, 0.68, 0.80, 0.67),
    ("re2_tt", "baro"): (0.67, 0.84, 0.89, 0.77, 0.81, 0.76), ("re2_tt", "simplerca"): (0.80, 0.87, 0.93, 0.84, 0.87, 0.85),
    ("re2_ss", "baro"): (0.14, 0.82, 0.94, 0.50, 0.67, 0.47), ("re2_ss", "simplerca"): (0.24, 0.88, 0.97, 0.59, 0.74, 0.55),
    ("re3_ob", "baro"): (0.00, 0.90, 0.90, 0.59, 0.71, 0.45), ("re3_ob", "simplerca"): (0.30, 0.90, 0.90, 0.62, 0.73, 0.56),
    ("re3_tt", "baro"): (0.50, 0.93, 1.00, 0.77, 0.86, 0.72), ("re3_tt", "simplerca"): (0.83, 1.00, 1.00, 0.93, 0.96, 0.91),
    ("re3_ss", "baro"): (0.00, 0.83, 0.93, 0.46, 0.65, 0.40), ("re3_ss", "simplerca"): (0.83, 0.90, 0.90, 0.88, 0.89, 0.87),
}
KEYS = ("AC@1", "AC@3", "AC@5", "Avg@3", "Avg@5", "MRR_all_cases")


def fname(ds, algo, branch):
    tag = "author-re3ssbranch" if branch == "re3ss" else "author"
    return f"{R}/{algo}_{tag}_{ds}.json"


print("| dataset | method | author branch | n | Top@1 | Top@3 | Top@5 | Avg@3 | Avg@5 | MRR | max abs diff vs paper (2 d.p.) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (ds, algo), pv in PAPER.items():
    branches = ["rcaeval", "re3ss"] if ds == "re3_ss" else ["rcaeval"]
    for br in branches:
        s = json.load(open(fname(ds, algo, br)))["summary"]
        ours = [s[k] for k in KEYS]
        diff = max(abs(round(o, 2) - p) for o, p in zip(ours, pv))
        cells = " | ".join(f"{o:.3f} ({p:.2f})" for o, p in zip(ours, pv))
        print(f"| {ds} | {algo} | {br} | {s['n']} | {cells} | {diff:.2f} |")
