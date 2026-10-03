#!/bin/bash
# G1 calibration: author-repo SimpleRCA (rcaeval branches) + author BARO on RCAEval RE2/RE3.
cd $AUDIT_ROOT/scripts/simplerca_port
PY=~/anaconda3/envs/simplerca/bin/python
# polars defaults to one thread per core (72); 12 workers x 72 threads hit the 4096-thread user limit
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=$AUDIT_ROOT/runs/simplerca_author
mkdir -p $R
for ds in re2_ob re2_ss re2_tt re3_ob re3_ss re3_tt; do
  $PY run_author_simplerca.py --dataset rcaeval_$ds --algo baro --procs 12 --out $R/baro_author_$ds.json
  $PY run_author_simplerca.py --dataset rcaeval_$ds --algo simplerca --procs 12 --out $R/simplerca_author_$ds.json
  $PY run_author_simplerca.py --dataset rcaeval_$ds --algo simplerca --no-logs --procs 12 --out $R/simplerca_author_nologs_$ds.json
done
for ds in re3_ss; do
  $PY run_author_simplerca.py --dataset rcaeval_$ds --algo simplerca --variant author_re3ss --procs 12 --out $R/simplerca_author-re3ssbranch_$ds.json
  $PY run_author_simplerca.py --dataset rcaeval_$ds --algo baro --variant author_re3ss --procs 12 --out $R/baro_author-re3ssbranch_$ds.json
done
echo ALL_DONE
