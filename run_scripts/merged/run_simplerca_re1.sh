#!/bin/bash
# Author-repo SimpleRCA (rcaeval branch) + author BARO on RE1 (metrics only), two windows.
cd $AUDIT_ROOT/scripts/simplerca_port
PY=~/anaconda3/envs/simplerca/bin/python
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=$AUDIT_ROOT/runs/simplerca_author
for ds in re1_ob re1_ss re1_tt; do
  for w in 20 full; do
    $PY run_author_simplerca.py --dataset rcaeval_$ds --algo simplerca --re1-window $w --procs 8 --out $R/simplerca_author_${ds}_win$w.json
    $PY run_author_simplerca.py --dataset rcaeval_$ds --algo baro --re1-window $w --procs 8 --out $R/baro_author_${ds}_win$w.json
  done
done
echo ALL_DONE
