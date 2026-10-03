#!/usr/bin/env python3
"""Quick pre-repair CIRCA stat: modal top-1 component frequency per system."""
import re
import collections

import pandas as pd

PRE_DIR = "$HOME/auditstack/scripts/openrca_eval"
FILES = {
    "bank": f"{PRE_DIR}/preds_circa_bank.csv",
    "mkt1": f"{PRE_DIR}/preds_circa_mkt1.csv",
    "mkt2": f"{PRE_DIR}/preds_circa_mkt2.csv",
    "tel": f"{PRE_DIR}/preds_circa_tel.csv",
}

for name, path in FILES.items():
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(name, "ERR", e)
        continue
    comps = []
    for p in df["prediction"]:
        m = re.search(r'"root cause component":\s*"([^"]+)"', str(p))
        if m:
            comps.append(m.group(1))
    c = collections.Counter(comps)
    total = sum(c.values()) or 1
    top = c.most_common(3)
    modal = top[0][1] / total if top else 0.0
    print(
        f"{name}: n={total} unique={len(c)} "
        f"top={top} modal_freq={modal:.2%}"
    )
