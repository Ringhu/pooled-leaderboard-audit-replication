"""Column and row counts of every RCAEval metric table we read (diagnostic; run on the 3090 workspace).
Writes paper/data/supp/table_shapes.json: per table kind, min/max of the number of columns and of data rows over all cases.
Usage: python3 table_shapes.py <out.json>   (reads only the header line and counts lines; no pandas)"""
import glob, json, os, sys
DATA = os.path.expanduser("~/rcaeval_repo/data")
KINDS = {
    "re1-ob-data": f"{DATA}/online-boutique/*/*/data.csv",
    "re1-ss-data": f"{DATA}/sock-shop-2/*/*/data.csv", "re1-ss-simple": f"{DATA}/sock-shop-2/*/*/simple_data.csv",
    "re1-tt-data": f"{DATA}/train-ticket/*/*/data.csv", "re1-tt-simple": f"{DATA}/train-ticket/*/*/simple_data.csv",
    "re2-ob-simple": f"{DATA}/RE2/RE2-OB/*/*/simple_metrics.csv", "re2-ss-simple": f"{DATA}/RE2/RE2-SS/*/*/simple_metrics.csv",
    "re2-tt-simple": f"{DATA}/RE2/RE2-TT/*/*/simple_metrics.csv",
}
out = {}
for k, pat in KINDS.items():
    cols, rows = [], []
    for f in sorted(glob.glob(pat)):
        with open(f) as fh:
            cols.append(fh.readline().rstrip("\n").count(",") + 1)
            rows.append(sum(1 for _ in fh))
    assert cols, pat
    mode = max(set(rows), key=rows.count)
    out[k] = {"n_cases": len(cols), "n_cols_min": min(cols), "n_cols_max": max(cols), "n_rows_min": min(rows), "n_rows_max": max(rows),
              "n_rows_mode": mode, "n_cases_rows_mode": rows.count(mode)}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
