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

Evaluate a model with random weights on AMR 3.0 data:
```bash
./random_experiment.sh
```

Training a model on English samples from EuroParl, and then evaluate on AMR 3.0 data:
```bash
./experiment.sh
```
