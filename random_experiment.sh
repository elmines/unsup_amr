#!/bin/bash

OUTPUT_PATH=`pwd`/random_preds_$(date +%s).txt

module load conda
conda activate unsup_amr
python -m unsupamr.predict --output_path $OUTPUT_PATH

conda activate unsup_amr_eval
./eval.sh $OUTPUT_PATH

echo Predictions available at $OUTPUT_PATH