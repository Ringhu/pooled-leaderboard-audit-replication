"""Run an existing RE1 run script with the RE1-OB duplicate-"time"-header fix applied.

Upstream RCAEval commit 21ff8c9 (2026-09-29): "The 50 RE1-OB CPU and MEM cases repeat the "time" header,
which pandas reads as "time.1". ... main.py and drop_time now drop it." We apply the same fix without
touching any run script or RCAEval file: pandas.read_csv is wrapped so that any column named
time.<digits> is dropped from the returned frame. Everything else (loader, window, call, scoring) is the
unchanged target script. After the target finishes, its output JSON config gets a "preprocess_patch" field.

Usage: python timefix_wrapper.py OUT.json TARGET_SCRIPT [target args...]   (OUT must be the target's output path;
       pass it as an absolute path: some targets chdir into the RCAEval repo before writing)
       python timefix_wrapper.py --annotate OUT.json TARGET_SCRIPT N_FRAMES "<original target argv>"   (annotate a finished output)
"""
import json
import os
import re
import runpy
import sys

import pandas as pd

ANNOTATE = sys.argv[1] == "--annotate"
if ANNOTATE:
    sys.argv.pop(1)
OUT = os.path.abspath(sys.argv[1])
TARGET = os.path.abspath(sys.argv[2])
_orig = pd.read_csv
_pat = re.compile(r"^time\.\d+$")
N_DROPPED = [0]


def read_csv_timefix(*args, **kw):
    df = _orig(*args, **kw)
    if isinstance(df, pd.DataFrame):
        drop = [c for c in df.columns if isinstance(c, str) and _pat.match(c)]
        if drop:
            N_DROPPED[0] += 1
            df = df.drop(columns=drop)
    return df


if ANNOTATE:  # argv: OUT TARGET N_FRAMES "original argv"
    N_DROPPED[0] = int(sys.argv[3])
    sys.argv = sys.argv[4].split()
else:
    pd.read_csv = read_csv_timefix
    sys.argv = [TARGET] + sys.argv[3:]
    runpy.run_path(TARGET, run_name="__main__")

with open(OUT) as f:
    res = json.load(f)
res.setdefault("config", {})["preprocess_patch"] = {
    "what": "drop columns matching ^time\\.\\d+$ after pandas.read_csv (duplicate 'time' header)",
    "why": "RE1-OB cpu/mem data.csv repeat the time header; fixed upstream in RCAEval 21ff8c9 (2026-09-29)",
    "wrapper": os.path.basename(__file__), "frames_with_dropped_cols": N_DROPPED[0],
    "wrapper_argv": [os.path.basename(p) for p in sys.argv],
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("timefix: frames with dropped time.N columns =", N_DROPPED[0], flush=True)
