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
    --trainer.logger.version ethans_run_mar16 \
    --data.batch_size 4 \
    --data.debug_subset true 
```

## Prediction
Required arguments to run the prediction module,
- `--output_path any/output/result/path`
- `--model.version <version>` where `<version>` is basename of the experiment dir `./lightning_logs/<version>` created by `unsupamr.fit`

Example:
```bash
python -m unsupamr.predict --model.version ethans_run_mar16 --output_path prediction_output.txt
```

## Putting it All Together

Here's a simple way to run both in succession:
```bash
RUN_NAME=ethans_run_mar16 

python -m unsupamr.fit \
    --trainer.logger.version $RUN_NAME \
python -m unsupamr.predict \
    --model.version $RUN_NAME \
    --output_path output.txt
```
