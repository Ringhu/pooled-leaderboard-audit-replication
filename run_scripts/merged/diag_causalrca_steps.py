"""Cost probe (diagnostic only): count CausalRCA training steps and time per step.

RCAEval causalrca.py calls the module-level matrix_poly() once per training batch (inside _h_A in train()) and once per
outer update; wrapping it gives the number of steps and the time between consecutive steps. The method is not
otherwise changed. Seeds 0 as run_causalrca_all.py; ts env + PYTHONPATH=work/pylib_skn031, CUDA as visible.

  re2  CASE_DIR           full run on an RE2 case (simple_metrics.csv, load_case as run_causalrca_all.py)
  openrca DUMP_STEM SEC   OpenRCA dumped input (diag_openrca_cost.py), stopped after SEC seconds (SIGALRM)
Writes OUT_JSON (last argument) with n_steps, per-step seconds (median of consecutive intervals), wall_s, stopped.
"""
import json
import os
import signal
import sys
import time

import numpy as np
import pandas as pd

M = "$AUDIT_ROOT"
sys.path.insert(0, "$HOME/rcaeval_repo")
sys.path.insert(0, f"{M}/scripts")


class Stop(BaseException):  # BaseException: not swallowed by causalrca.py's `except Exception`
    pass


def main():
    mode, out = sys.argv[1], sys.argv[-1]
    import random
    import torch
    torch.set_num_threads(1)
    import RCAEval.e2e.causalrca as C
    stamps = []
    orig = C.matrix_poly

    def counted(matrix, d):
        stamps.append(time.time())
        return orig(matrix, d)

    C.matrix_poly = counted
    if mode == "re2":
        from run_causalrca_all import load_case
        data, it, *_ = load_case(sys.argv[2], length_min=20)
        ds, limit = "re2-tt", None
    else:
        stem = sys.argv[2]
        meta = json.load(open(stem + ".json"))
        data, it, ds, limit = pd.read_csv(stem + ".csv"), meta["inject_time"], f"openrca-{meta['tag']}", int(sys.argv[3])
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    fn = C.causalrca.__wrapped__ if hasattr(C.causalrca, "__wrapped__") else C.causalrca
    if limit:
        signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(Stop()))
        signal.alarm(limit)
    t0, stopped = time.time(), False
    try:
        fn(data, it, dataset=ds, anomalies=None, dk_select_useful=False, sli=None, verbose=False)
    except Stop:
        stopped = True
    wall = time.time() - t0
    gaps = np.diff(stamps)
    rec = dict(mode=mode, input=sys.argv[2], shape=list(data.shape), cuda=torch.cuda.is_available(),
               gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), n_steps=len(stamps), wall_s=round(wall, 1), stopped=stopped,
               step_s_median=float(np.median(gaps)) if len(gaps) else None,
               step_s_mean=float(np.mean(gaps)) if len(gaps) else None,
               first_step_after_s=round(stamps[0] - t0, 1) if stamps else None)
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
