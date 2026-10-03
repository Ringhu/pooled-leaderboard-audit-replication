#!/bin/bash
# alert-count on OpenRCA: audit code as run (--unfixed, reproduction check) and with the call bug fixed.
cd $AUDIT_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=$CONDA_ENVS/auditstack/bin/python
for t in bank mkt1 mkt2 tel; do
  $PY scripts/openrca_alertcount_fixed.py --tag $t --out_dir runs/openrca > logs/openrca_alertcount_$t.log 2>&1 &
  $PY scripts/openrca_alertcount_fixed.py --tag $t --out_dir runs/openrca --unfixed > logs/openrca_alertcount-unfixed_$t.log 2>&1 &
done
wait
echo ALL_DONE
