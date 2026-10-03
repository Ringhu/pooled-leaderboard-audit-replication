"""2026-09-27: RE1-OB ships only data.csv (51 cols, already the reduced table), so RE1-OB runs on
data.csv are NOT contaminated. Also: B's baro_v7_re1{ss,tt} are per-case BARO on raw data.csv
(usable for the RQ2(a) contrast) -> contaminated instead of drop; baro_v7_re1ob -> valid."""
import csv, os
L = os.path.expanduser("$AUDIT_ROOT/runs_ledger.csv")
rows = list(csv.DictReader(open(L))); keys = list(rows[0].keys())
fix = {
    "raw/earlier_a_followup_runs/circa_re1-ob.json": "valid",
    "raw/earlier_a_followup_runs/ediag_re1-ob.json": "valid",
    "raw/earlier_b/circa_re1-ob.json": "rescore",   # 102/125 valid outputs; denominator must be 125
    "raw/earlier_b/e_diagnosis_re1ob.json": "valid",
    "raw/earlier_b/simplerca_re1ob.json": "superseded",
    "raw/earlier_b/causalrca_re1ob.json": "forensics_only",
    "raw/earlier_b/baro_v7_re1ob.json": "valid",
    "raw/earlier_b/baro_v7_re1ss.json": "contaminated",
    "raw/earlier_b/baro_v7_re1tt.json": "contaminated",
}
n = 0
for r in rows:
    if r["path"] in fix:
        old = r["status"]; r["status"] = fix[r["path"]]
        r["note"] = (r.get("note") or "") + f" | 2026-09-27 status {old}->{fix[r['path']]}: RE1-OB has only data.csv"
        n += 1
with open(L, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("updated", n)
