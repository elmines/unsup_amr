#!/bin/bash

TIMESTAMP=$(date +%s)

prefix=`pwd`/outputs/random_$TIMESTAMP
PREDS_OUTPUT=${prefix}_predictions.txt
LOGGED_OUTPUT=${prefix}_output.txt
METRICS_OUTPUT=${prefix}_metrics.txt

module load conda
conda activate unsup_amr
python -m unsupamr.predict --output_path $PREDS_OUTPUT | tee $LOGGED_OUTPUT

conda activate unsup_amr_eval
./eval.sh $PREDS_OUTPUT | tee $METRICS_OUTPUT

echo Predictions available at $PREDS_OUTPUT
echo "Full prediction output (DFS and Penman) available at $LOGGED_OUTPUT"
echo Metrics available at $METRICS_OUTPUT