# Unsupervised AMR
Training of an AMR parser without labelled samples

## Dependencies

```bash
conda env create -f environment.yml
conda activate unsup_amr
```

## Training

Pytorch Lightning has many optional CLI params:
```bash
python -m unsupamr.fit --help
```

At least right now we've managed to make our training CLI arguments all optional.
`--data.batch_size` and `--data.debug_subset` are handy though for light testing:
Here's a sample run:
```bash
python -m unsupamr.fit \
    --data.batch_size 4 \
    --data.debug_subset true 
```

## Prediction
Required arguments to run the prediction module,
- `--output_path any/output/result/path`
- `--data.amr_version 3.0|2.0`

Example:
```bash
python -m unsupamr.predict --output_path prediction_output.txt --data.amr_version 3.0 
```
