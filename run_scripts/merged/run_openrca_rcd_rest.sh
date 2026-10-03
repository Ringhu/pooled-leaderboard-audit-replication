#!/bin/bash
# continuation of run_openrca_rcd.sh (2026-09-29 23:1x): seed 0 ranks done under the old runner; seed 1 was interrupted
# (runner stopped) and is recomputed from scratch. Seeds 1 and 2 with 32 procs, then wrapper predictions for the
# 3 seeds in parallel (3 x 4 procs).
set -e
cd $AUDIT_ROOT
export POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
C=~/anaconda3/envs/auditstack/bin/python; R=~/anaconda3/envs/rcaeval_rcd/bin/python
for t in bank mkt1 mkt2 tel; do test -s runs/openrca/rcd_ranks/${t}_seed0.json; done
for s in 1 2; do $R scripts/openrca_rcd.py rank --seed $s --procs 32; done
for s in 0 1 2; do ( for t in bank mkt1 mkt2 tel; do
  $C scripts/openrca_wrapped.py --method rcd --seed $s --tag $t --out_dir runs/openrca --procs 4
done ) > logs/openrca_rcd_wrap_seed$s.log 2>&1 & done
wait
echo ALLDONE
