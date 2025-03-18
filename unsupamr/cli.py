# 3rd Party
from lightning.pytorch.cli import LightningCLI
# Local
from .constants import DEFAULT_SEQ_MODEL

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--pretrained_model", default=DEFAULT_SEQ_MODEL)
        parser.link_arguments("pretrained_model", "model.pretrained_model")
        parser.link_arguments("pretrained_model", "data.pretrained_model")


