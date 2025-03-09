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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load conda
conda env create -f environment.yml
conda activate unsup_amr

srun python -m unsupamr.fit --data.source_lang en --data.target_lang de --data.batch_size 2 --model.vocab_path vocab.json --data.debug_subset true

conda deactivate
