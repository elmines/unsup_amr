"""
The script we use for training runs
"""
import time
from .cli import CustomCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from .lightning_mods import TrainingMod
from .data_mods import EuroParlDataModule
from .constants import STOPPING_METRIC

if __name__ == "__main__":
    # model_callback = ModelCheckpoint(
    #     monitor=STOPPING_METRIC,
    #     mode='min',
    #     filename="{epoch:02d}-{loss:.3f}"
    # )
    early_stopping_callback = EarlyStopping(
        monitor=STOPPING_METRIC,
        patience=3,
        mode='min'
    )

    best_checkpoint_callback = ModelCheckpoint(
        monitor=STOPPING_METRIC,
        mode='min',
        filename=f'best_{int(time.time())}',
        every_n_train_steps=10 # TODO: remove this
    )

    cli = CustomCLI(
        model_class=TrainingMod, subclass_mode_model=False,
        datamodule_class=EuroParlDataModule, subclass_mode_data=False,
        trainer_defaults={
            "max_epochs": 10,
            "callbacks": [
                early_stopping_callback,
                best_checkpoint_callback
            ]
        },
        run=False
    )

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, )