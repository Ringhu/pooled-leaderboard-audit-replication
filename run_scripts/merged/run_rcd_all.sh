#!/bin/bash
# RCD (RCAEval official, patched causal-learn via script/link.sh) on RE1 x 3 seeds.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $AUDIT_ROOT/scripts
PY=~/anaconda3/envs/rcaeval_rcd/bin/python
R=$AUDIT_ROOT/runs/rcd
mkdir -p $R
for seed in 0 1 2; do
  for ds in re1-ob re1-ss re1-tt; do
    $PY run_rcd_re1.py $ds $seed $R/rcd_${ds}_seed${seed}.json --procs 12
  done
done
echo ALL_DONE
