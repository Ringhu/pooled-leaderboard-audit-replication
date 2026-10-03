#!/bin/bash
# RCD on OpenRCA (user ruling 2026-09-29): dump wrapper inputs -> RCD ranks (seeds 0-2) -> wrapper predictions
set -e
cd $AUDIT_ROOT
export POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
C=~/anaconda3/envs/auditstack/bin/python; R=~/anaconda3/envs/rcaeval_rcd/bin/python
$C scripts/openrca_rcd.py dump --procs 4
for s in 0 1 2; do $R scripts/openrca_rcd.py rank --seed $s --procs 16; done
for s in 0 1 2; do for t in bank mkt1 mkt2 tel; do
  $C scripts/openrca_wrapped.py --method rcd --seed $s --tag $t --out_dir runs/openrca --procs 4
done; done
echo ALLDONE
