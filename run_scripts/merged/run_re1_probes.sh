#!/bin/bash
# audit probes on RCAEval at 10 min (A long-table window; must reproduce it) and 20 min (RCAEval default,
# RE1 window decision 2026-09-27), plus input-representation contrasts at 20 min.
cd $AUDIT_ROOT
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1
PY=~/anaconda3/envs/auditstack/bin/python
for m in zbase alertcount cd1min; do
  for ds in re1-ob re1-ss re1-tt re2-ob re2-ss re2-tt; do
    for L in 10 20; do
      $PY scripts/run_re1_probes.py --method $m --dataset $ds --length $L --out runs/probes/${m}_${ds}_len${L}.json &
    done
  done
  for ds in re1-ss re1-tt; do
    $PY scripts/run_re1_probes.py --method $m --dataset $ds --length 20 --input-file data.csv --out runs/probes/${m}_${ds}_datacsv_len20.json &
  done
  for ds in re2-ob re2-ss re2-tt; do
    $PY scripts/run_re1_probes.py --method $m --dataset $ds --length 20 --input-file metrics.csv --out runs/probes/${m}_${ds}_metricscsv_len20.json &
  done
  wait
done
for ds in re1-ss re1-tt; do
  $PY scripts/run_re1_baro.py $ds runs/baro/baro_${ds}_datacsv_len20.json --length 20 --input-file data.csv &
done
wait
echo ALL_DONE
