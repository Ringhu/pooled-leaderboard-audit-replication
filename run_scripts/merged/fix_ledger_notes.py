"""2026-09-27: correct the note text fix_ledger_re1ob.py wrote on B's baro_v7_re1{ss,tt} rows
(the reason given there, "RE1-OB has only data.csv", applies to the OB row only). Idempotent."""
import csv, os
L = os.path.expanduser("$AUDIT_ROOT/runs_ledger.csv")
rows = list(csv.DictReader(open(L))); keys = list(rows[0].keys())
OLD = "contaminated: RE1-OB has only data.csv"
NEW = "contaminated: BARO on raw data.csv (RE1-SS/TT), RQ2(a) contrast only"
n = 0
for r in rows:
    if r["path"] in ("raw/earlier_b/baro_v7_re1ss.json", "raw/earlier_b/baro_v7_re1tt.json") and OLD in r["note"]:
        r["note"] = r["note"].replace(OLD, NEW); n += 1
with open(L, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("updated", n)
