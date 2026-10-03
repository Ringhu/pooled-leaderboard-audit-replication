"""Count OpenRCA predictions that equal the audit wrapper's fixed per-system default answer
(_fallback_json: DEFAULT component + reason, occurrence time = query-window midpoint), per prediction file,
and how much official partial score those default answers earn. Run on 3090 (env auditstack)."""
import sys, re, pandas as pd
sys.path.insert(0, "$HOME/auditstack/scripts/openrca_eval")
from predict_rcaagent_1min import DEFAULT, parse_instruction
from openrca_evaluate_sweep import evaluate
E = "$HOME/auditstack/scripts/openrca_eval"
O = "$HOME/auditstack/data/OpenRCA/dataset"
SYS = {"bank": ("Bank", f"{O}/Bank/query.csv"), "mkt1": ("Market_cloudbed-1", f"{O}/Market/cloudbed-1/query.csv"),
       "mkt2": ("Market_cloudbed-2", f"{O}/Market/cloudbed-2/query.csv"), "tel": ("Telecom", f"{O}/Telecom/query.csv")}
for pref in ["full335/preds_baro_", "full335/preds_1min_", "preds_1min_", "preds_alertcount_", "preds_zbase_univ_"]:
    for tag, (sysname, qp) in SYS.items():
        try:
            p = pd.read_csv(f"{E}/{pref}{tag}.csv")
        except FileNotFoundError:
            print(pref, tag, "missing"); continue
        q = pd.read_csv(qp)
        comp, reason = DEFAULT[sysname]
        nfb, sfb, tot = 0, 0.0, 0.0
        for pred, ins, sp in zip(p.prediction, q.instruction, q.scoring_points):
            mid8, _, _ = parse_instruction(ins)
            dt = mid8.strftime('%Y-%m-%d %H:%M:%S') if mid8 is not None else ''
            one = f'{{"root cause occurrence datetime": "{dt}", "root cause component": "{comp}", "root cause reason": "{reason}"}}'
            s = float(evaluate(str(pred), str(sp), 60)[2]); tot += s
            if str(pred) != "nan" and set(re.findall(r"\{[^}]*\}", str(pred))) == {one}:
                nfb += 1; sfb += s
        print(f"{pref}{tag}: n={len(p)} default_answers={nfb} score_from_defaults={sfb:.2f} total_score={tot:.2f}")
