#!/bin/bash
# WP2: RCAEval-official methods on RE2 with full config; see run_re2_methods.py docstring.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $AUDIT_ROOT/scripts
PY=~/anaconda3/envs/auditstack/bin/python
R=$AUDIT_ROOT/runs/re2
mkdir -p $R
for s in ob ss tt; do
  $PY run_re2_methods.py --method mmbaro --dataset re2-$s --out $R/mmbaro_re2-${s}_name-re2.json
  $PY run_re2_methods.py --method mmbaro --dataset re2-$s --mm-dataset-name mm-$s --out $R/mmbaro_re2-${s}_name-mm.json
  $PY run_re2_methods.py --method mmbaro --dataset re2-$s --mm-dataset-name mm-$s --metric-file metrics.csv --out $R/mmbaro_re2-${s}_name-mm_metricscsv.json
done
for s in ob tt; do
  $PY run_re2_methods.py --method tracerca --dataset re2-$s --out $R/tracerca_re2-${s}.json
done
for s in ob ss; do
  for L in 20 10; do
    $PY run_re2_methods.py --method circa --dataset re2-$s --length $L --out $R/circa_re2-${s}_len$L.json
  done
done
for s in ob tt; do
  $PY run_re2_methods.py --method microrank --dataset re2-$s --out $R/microrank_re2-${s}.json
done
for L in 20 10; do
  $PY run_re2_methods.py --method circa --dataset re2-tt --length $L --out $R/circa_re2-tt_len$L.json
done
echo ALL_DONE
