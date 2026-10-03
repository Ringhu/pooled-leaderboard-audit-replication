#!/bin/bash
# Quick check: regenerate every analysis output, number, table, and figure of the paper from data/merged_long.csv.
# Needs Python 3.11 with numpy, pandas, scipy, statsmodels, matplotlib (see envs/auditstack.txt). A few minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
export AUDIT_ROOT="$PWD/work" POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1
A=work/analysis/main
rm -rf work && mkdir -p $A/rq1 $A/rq2 $A/rq3 $A/rq1_cross6 $A/inputs
cp data/merged_long.csv work/ && cp data/inputs/*.txt $A/inputs/
for s in rq1_main rq1_extra rq2_main rq2_fault rq3_holdout rq1_crossfamily; do
  echo "== $s"; $PY analysis/$s.py > work/$s.log 2>&1 || { echo "$s failed, see work/$s.log"; exit 1; }
done
echo "== rq1_bonferroni"; $PY analysis/rq1_bonferroni.py work/merged_long.csv $A/rq1/pairs.csv $A/rq1 > work/rq1_bonferroni.log 2>&1 || { echo "rq1_bonferroni failed, see work/rq1_bonferroni.log"; exit 1; }
D=paper/data
rm -rf $D/rq1 $D/rq2 $D/holdout $D/rq1_cross6 && mkdir -p $D/rq1 $D/rq2 $D/holdout $D/rq1_cross6
cp $A/rq1/*.csv $A/rq1/*.json $D/rq1/
cp $A/rq2/*.csv $A/rq2/*.json $D/rq2/
cp $A/rq3/config.json $A/rq3/crossrelease.csv $A/rq3/curve.csv $A/rq3/decisions.csv $A/rq3/g5.json $D/holdout/
cp $A/rq1_cross6/* $D/rq1_cross6/
$PY - <<PY
import hashlib, json
h = hashlib.sha256(open("data/merged_long.csv", "rb").read()).hexdigest()
n = sum(1 for _ in open("data/merged_long.csv")) - 1
json.dump(dict(rows=n, sha256=h, released_sha256=h), open("$D/long_meta.json", "w"))
PY
cp paper/numbers.tex work/numbers.shipped.tex
(cd paper && for s in make_numbers make_tables make_figures make_fig0_teaser make_supp_tables make_supp_extra make_supp_restored; do
   $PY scripts/$s.py > ../work/$s.log 2>&1 || { echo "$s failed, see work/$s.log"; exit 1; }; done)
if diff -q work/numbers.shipped.tex paper/numbers.tex >/dev/null; then echo "numbers.tex: identical to the shipped version"
else echo "numbers.tex differs from the shipped version:"; diff work/numbers.shipped.tex paper/numbers.tex || true; fi
$PY scripts/verify_main_text_numbers.py
