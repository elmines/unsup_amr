#!/bin/bash

TIMESTAMP=$(date +%s)

VERSION=grad_analysis_$TIMESTAMP

module load conda
conda activate unsup_amr

EXTRA_ARGS=$@

python -m unsupamr.fit \
    --model.log_gradients true \
    --data.debug_subset true \
    --trainer.max_steps 1000 \
    --trainer.logger.version $VERSION $EXTRA_ARGS
