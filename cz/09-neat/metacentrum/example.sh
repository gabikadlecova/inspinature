#!/bin/bash

echo "Hello from script"

OUTDIR=/storage/plzen1/home/gabisuchoparova/outputs

module load py-pip/21.3.1-gcc-10.2.1-mjt74tn
pip install --user scikit-learn numpy deap

cd $OUTDIR/..

python cmaes.py  > "$SCRATCHDIR"/out.txt 2> "$SCRATCHDIR"/err.txt


cp "$SCRATCHDIR"/out.txt "$OUTDIR"/"$PBS_JOBID"_out.txt
cp "$SCRATCHDIR"/err.txt "$OUTDIR"/"$PBS_JOBID"_err.txt

[[ -n $SCRATCHDIR ]] && rm -r "$SCRATCHDIR"/*

