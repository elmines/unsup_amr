#!/bin/bash

#SBATCH --job-name=unsup_amr
#SBATCH -o unsup_amr.out                   
#SBATCH -e unsup_amr.err                   
#SBATCH --output=%j.log               
#SBATCH --cpus-per-task=1
#SBATCH --qos=cai6307             
#SBATCH --account=cai6307
#SBATCH --mem=8G                     
#SBATCH --time=40:00:00               
#SBATCH --mail-type=begin             
#SBATCH --mail-type=end               
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

TIMESTAMP=$(date +%s)

prefix=`pwd`/outputs/random_$TIMESTAMP
PREDS_OUTPUT=${prefix}_predictions.txt
LOGGED_OUTPUT=${prefix}_output.txt
METRICS_OUTPUT=${prefix}_metrics.txt

module load conda
conda activate unsup_amr
python -m unsupamr.predict --output_path $PREDS_OUTPUT $COMMON_ARGS | tee $LOGGED_OUTPUT

conda activate unsup_amr_eval
./eval.sh $PREDS_OUTPUT | tee $METRICS_OUTPUT

echo Predictions available at $PREDS_OUTPUT
echo "Full prediction output (DFS and Penman) available at $LOGGED_OUTPUT"
echo Metrics available at $METRICS_OUTPUT