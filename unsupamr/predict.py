# STL
import os
# Local
from .cli import CustomCLI
from. predict_mods import PredictMod
from .data_mods import AMRDataModule

class PredictCLI(CustomCLI):

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser=parser)
        parser.add_argument('--output_path', type=str, required=True, help="Path to save predictions.")

if __name__ == "__main__":
    cli = PredictCLI(
        model_class = PredictMod,
        datamodule_class = AMRDataModule,
        trainer_defaults = {
            "max_epochs": 1,   #setting it 1 for now
            },
        run = False,   
        )

    cli.datamodule.vocab_ext = cli.model.vocab_ext
    output_path = cli.config.output_path
    print(f"Predictions will be saved to: {output_path}")

    prediction_batches = cli.trainer.predict(model=cli.model, datamodule=cli.datamodule)

    with open(output_path, "w") as f:
        for prediction_batch in prediction_batches:
            for pred in prediction_batch:
                f.write(str(pred) + "\n\n")
    
    print(f"Predictions saved to {output_path}")
