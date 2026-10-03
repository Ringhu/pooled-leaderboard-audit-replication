#!/bin/bash
# RE1-OB duplicate-"time"-header rerun (2026-09-30; decision .research/decisions/2026-09-30-re1ob-timefix-unfreeze.md).
# All 125 RE1-OB cases are rerun so the 75 unaffected cases double as a reproduction check.
cd $AUDIT_ROOT
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TS=~/anaconda3/envs/ts/bin/python
CD=~/anaconda3/envs/auditstack/bin/python
RC=~/anaconda3/envs/rcaeval_rcd/bin/python
O=runs/re1ob_timefix
W=scripts/timefix_wrapper.py
$TS $W $O/circa_re1-ob_timefix.json scripts/run_re1_pub.py circa re1-ob $O/circa_re1-ob_timefix.json > logs/timefix_circa.log 2>&1 &
$TS $W $O/e_diagnosis_re1-ob_timefix.json scripts/run_re1_pub.py e_diagnosis re1-ob $O/e_diagnosis_re1-ob_timefix.json > logs/timefix_ediag.log 2>&1 &
( for s in 0 1 2; do $RC $W $O/rcd_re1-ob_seed${s}_timefix.json scripts/run_rcd_re1.py re1-ob $s $O/rcd_re1-ob_seed${s}_timefix.json --procs 8; done ) > logs/timefix_rcd.log 2>&1 &
( $CD $W $O/baro_re1-ob_len20_timefix.json scripts/run_re1_baro.py re1-ob $O/baro_re1-ob_len20_timefix.json --length 20
  for m in zbase alertcount cd1min; do $CD $W $O/${m}_re1-ob_len20_timefix.json scripts/run_re1_probes.py --method $m --dataset re1-ob --length 20 --out $O/${m}_re1-ob_len20_timefix.json; done ) > logs/timefix_baro_probes.log 2>&1 &
wait
echo ALLDONE
