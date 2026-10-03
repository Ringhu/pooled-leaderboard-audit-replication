#!/bin/bash
# audit SimpleRCA rewrite on OpenRCA via the audit wrapper (sensitivity control); queries in parallel within each system.
cd $AUDIT_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2
PY=$CONDA_ENVS/auditstack/bin/python
for t in bank mkt1 mkt2 tel; do
  $PY scripts/openrca_wrapped.py --method simplerca --tag $t --out_dir runs/openrca --procs 4 > logs/openrca_simplerca_${t}.log 2>&1 &
done
wait
echo ALL_DONE
