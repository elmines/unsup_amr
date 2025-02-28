# 3rd Party
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
# Local
from .constants import DEFAULT_SEQ_MODEL
from .lightning_mods import TrainingMod
from .data_mods import EuroParlDataModule

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--pretrained_model", default=DEFAULT_SEQ_MODEL)
        parser.link_arguments("pretrained_model", "model.pretrained_model")
        parser.link_arguments("pretrained_model", "data.tokenizer")

def cli_main(**cli_kwargs):

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

    return CustomCLI(
        model_class=TrainingMod, subclass_mode_model=False,
        model_class=EuroParlDataModule, subclass_mode_data=False,
        trainer_defaults={
            "max_epochs": 10,
            "callbacks": [
                model_callback,
                early_stopping_callback
            ]
        }
    )