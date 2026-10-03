#!/bin/bash
# BARO (harness check) + e-Diagnosis on OpenRCA via the audit wrapper; queries in parallel within each system.
cd $AUDIT_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2
PY=$CONDA_ENVS/auditstack/bin/python
for t in bank mkt1 mkt2 tel; do
  $PY scripts/openrca_wrapped.py --method baro --tag $t --out_dir runs/openrca --procs 4 > logs/openrca_baro_${t}.log 2>&1 &
done
wait
declare -A P=([bank]=8 [mkt1]=8 [mkt2]=8 [tel]=6)
for t in bank mkt1 mkt2 tel; do
  $PY scripts/openrca_wrapped.py --method ediag --tag $t --out_dir runs/openrca --procs ${P[$t]} > logs/openrca_ediag_${t}.log 2>&1 &
done
wait
echo ALL_DONE
