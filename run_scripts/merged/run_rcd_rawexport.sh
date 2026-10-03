#!/bin/bash
# RQ2(a): RCD on the raw-export data.csv of RE1-SS/TT (contrast to simple_data.csv), 3 seeds.
# Waits for run_rcd_all.sh to finish so the two batches do not compete for CPU.
while pgrep -f "^/bin/bash .*run_rcd_all.sh" > /dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $AUDIT_ROOT/scripts
PY=~/anaconda3/envs/rcaeval_rcd/bin/python
R=$AUDIT_ROOT/runs/rcd
for seed in 0 1 2; do
  for ds in re1-ss re1-tt; do
    $PY run_rcd_re1.py $ds $seed $R/rcd_${ds}_datacsv_seed${seed}.json --input-file data.csv --procs 12
  done
done
echo ALL_DONE
