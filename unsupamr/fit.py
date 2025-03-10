"""
The script we use for training runs
"""
from .cli import CustomCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from .lightning_mods import TrainingMod
from .data_mods import EuroParlDataModule

if __name__ == "__main__":

    STOPPING_METRIC = "loss"
    model_callback = ModelCheckpoint(
        monitor=STOPPING_METRIC,
        mode='min',
        filename="{epoch:02d}-{loss:.3f}"
    )
    early_stopping_callback = EarlyStopping(
        monitor=STOPPING_METRIC,
        patience=3,
        mode='min'
    )

    cli = CustomCLI(
        model_class=TrainingMod, subclass_mode_model=False,
        datamodule_class=EuroParlDataModule, subclass_mode_data=False,
        trainer_defaults={
            "max_epochs": 10,
            "callbacks": [
                model_callback,
                early_stopping_callback
            ]
        },
        run=False
    )
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)