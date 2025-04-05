"""
The script we use for training runs
"""
# STL
import os
# Local
from .cli import CustomCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from .lightning_mods import TrainingMod
from .data_mods import EuroParlDataModule
from .constants import STOPPING_METRIC

if __name__ == "__main__":

    validation_steps = 1000 # TODO: Make thsi configurable? 

    best_checkpoint_callback = ModelCheckpoint(
        monitor=STOPPING_METRIC,
        mode='min',
        filename="best-{epoch:02d}-{loss:.3f}",
        every_n_train_steps=validation_steps,
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
            "logger": dict(
                class_path="lightning.pytorch.loggers.CSVLogger",
                init_args=dict(
                    save_dir=os.path.join(os.path.dirname(__file__), ".."),
                )
            ),
            "callbacks": [
                early_stopping_callback,
                best_checkpoint_callback
            ],
            "val_check_interval": validation_steps
        },
        run=False
    )
    cli.datamodule.vocab_ext = cli.model.vocab_ext
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, )