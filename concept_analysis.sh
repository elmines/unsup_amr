#!/bin/bash

TIMESTAMP=$(date +%s)

VERSION=concept_analysis_$TIMESTAMP

module load conda
conda activate unsup_amr

EXTRA_ARGS=$@

python -m unsupamr.fit \
    --model.log_concept_rates true \
    --trainer.log_every_n_steps 1 \
    --data.debug_subset true \
    --trainer.max_steps 1000 \
    --trainer.logger.version $VERSION $EXTRA_ARGS
