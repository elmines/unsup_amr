"""
The script we use for training runs
"""
from .cli import cli_main

if __name__ == "__main__":
    cli = cli_main(run=False)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)