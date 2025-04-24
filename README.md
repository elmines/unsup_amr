# Unsupervised AMR
Training of an AMR parser without labelled samples

## Getting Started

```bash
git clone --recurse-submodules https://github.com/elmines/unsup_amr.git
# Or if you didn't read these instructions before cloning:
# git submodule update --init --recursive
conda env create -f environment.yml
conda env create -f eval_env.yml
```

Running experiments:
```bash
## Experiments with training
# Base
./experiment.sh
# τ = 1.5
COMMON_ARGS="--model.temperature 1.5" ./experiment.sh
# RLH
TRAIN_ARGS="--model.new_lm_head_scheme true" ./experiment.sh
# ENT
COMMON_ARGS="--model.limit_frame_ids true" ./experiment.sh
# RLH + ENT
TRAIN_ARGS="--model.new_lm_head_scheme true" COMMON_ARGS="--model.limit_frame_ids true" ./experiment.sh
# No Training
./random_experiment.sh
# No Training + ENT
COMMON_ARGS="--model.limit_frame_ids true" ./random_experiment.sh
```

