#!/bin/bash
cd $AUDIT_ROOT/scripts
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 PYTHONPATH=$AUDIT_ROOT/work/pylib_skn031
PY=$CONDA_ENVS/ts/bin/python
L=$AUDIT_ROOT/logs/causalrca_diag_seed
$PY diag_causalrca_seed.py 0 re1-ob:checkoutservice_cpu/2:0 re1-ob:checkoutservice_cpu/3:0 re1-ob:checkoutservice_cpu/2:1 > ${L}_0.log 2>&1 &
$PY diag_causalrca_seed.py 3 re1-ss:orders_cpu/2:0 re1-ss:orders_cpu/3:0 re1-ss:orders_cpu/2:1 > ${L}_3.log 2>&1 &
$PY diag_causalrca_seed.py 5 re1-tt:ts-route-service_cpu/2:0 re1-tt:ts-route-service_cpu/3:0 re1-tt:ts-route-service_cpu/2:1 > ${L}_5.log 2>&1 &
$PY diag_causalrca_seed.py 6 re2-ob:emailservice_socket/2:0 re2-ob:emailservice_socket/1:0 re2-ob:emailservice_socket/2:1 re2-ss:orders_socket/2:0 > ${L}_6.log 2>&1 &
$PY diag_causalrca_seed.py 7 re2-tt:ts-route-service_socket/2:0 re2-tt:ts-route-service_socket/1:0 re2-tt:ts-route-service_socket/2:1 > ${L}_7.log 2>&1 &
$PY diag_causalrca_seed.py 0 re2-ss:orders_socket/1:0 re2-ss:orders_socket/2:1 > ${L}_0b.log 2>&1 &
wait
