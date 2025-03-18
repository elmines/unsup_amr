from .cli import CustomCLI
from. predict_mods import PredictMod
from .data_mods import AMRDataModule

class PredictCLI(CustomCLI):

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser=parser)
        parser.add_argument('--output_path', type=str, required=True, help="Path to save predictions.")

    
    def after_instantiate_classes(self):
        args = self.parser.parse_args()
        print("Parsed Arguments:", args)  # Debugging step
        self.output_path = args.output_path
        print(f"Predictions will be saved to: {self.output_path}")


if __name__ == "__main__":
    cli = PredictCLI(
        model_class = PredictMod,
        datamodule_class = AMRDataModule,
        trainer_defaults = {
            "max_epochs": 1,   #setting it 1 for now
            },
        run = False,   
        )

    predictions = cli.trainer.predict(model=cli.model, datamodule=cli.datamodule)

    with open(cli.output_path, "w") as f:
        for prediction in predictions:
            for pred in prediction:
                f.write(str(pred) + "\n")
    
    print(f"Predictions saved to {cli.output_path}")
