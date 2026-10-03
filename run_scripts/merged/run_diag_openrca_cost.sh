#!/bin/bash
# cost probe: 4 OpenRCA queries x {circa, rcd, causalrca}, all in parallel, capped (circa/rcd 1 h, causalrca 2 h)
cd $AUDIT_ROOT
O=scratch/openrca_cost; S=scripts/diag_openrca_cost.py
export POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
C=~/anaconda3/envs/auditstack/bin/python; R=~/anaconda3/envs/rcaeval_rcd/bin/python
GPUS=(0 3 5 6); i=0
for t in bank mkt1 mkt2 tel; do
  CUDA_VISIBLE_DEVICES= timeout 3600 $C $S time $O circa $t > $O/log_circa_$t.txt 2>&1 &
  CUDA_VISIBLE_DEVICES= timeout 3600 $R $S time $O rcd $t > $O/log_rcd_$t.txt 2>&1 &
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} timeout 7200 $R $S time $O causalrca $t > $O/log_causalrca_$t.txt 2>&1 &
  i=$((i+1))
done
wait
echo ALLDONE
